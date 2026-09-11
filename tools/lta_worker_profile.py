"""Heatsoaked, repeated functional profiling of persistent production LTA tasks.

The adapter wraps the production worker without changing masks or scheduling.
Unprofiled repeats provide timings; a separate cProfile repeat attributes CPU
and synchronization cost. Inclusive phase times overlap and must not be added.
"""
from __future__ import annotations

import argparse
import cProfile
from contextlib import ExitStack
import csv
from dataclasses import asdict
import functools
import io
import json
import math
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import threading
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lta_production_smoke import prepare_fixture, validate_chain, _sha256_file, _source_snapshot
from XTA.lta_outputs import write_json_atomically
from XTA.lta_workers import LtaWorkerInit, LtaWorkerPool, LtaWorkerTask
from XTA.lta_execution import _load_chain_manifest, _consume_chain_manifest


def build_profile_predictor(config):
    from XTA.lta_worker_adapter import build_worker_predictor
    started = time.perf_counter()
    context = build_worker_predictor(config)
    context.profile_build_seconds = time.perf_counter() - started
    context.profile_source_sha256 = _source_snapshot()
    seconds = float(config.get("heat_seconds", 60))
    torch = context.torch_module
    print(f"Worker modules imported; heatsoaking for {seconds:g} seconds.", flush=True)
    with torch.inference_mode():
        a = torch.full((4096, 4096), 0.001, device="cuda", dtype=torch.float32)
        b = torch.empty_like(a)
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            torch.mm(a, a, out=b)
            torch.cuda.synchronize()
        context.profile_heat_seconds = time.perf_counter() - started
        del a, b
    torch.cuda.empty_cache()
    return context


def close_profile_predictor(context):
    from XTA.lta_worker_adapter import close_worker_predictor
    close_worker_predictor(context)


def _profile_rows(stats, sort_index):
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        rows.append({"file": filename, "line": line, "function": name,
                     "primitive_calls": primitive, "calls": calls,
                     "self_seconds": own, "cumulative_seconds": cumulative})
    return sorted(rows, key=lambda row: row[sort_index], reverse=True)[:100]


def execute_profile_task(context, kind, payload):
    from XTA import lta_worker_adapter as worker, lta_propagation as propagation
    from XTA import lta_rendering as rendering, lta_postprocessing as postprocessing
    from XTA import lta_experimental as experimental
    phases = {}
    def instrument(label, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                row = phases.setdefault(label, {"calls": 0, "inclusive_wall_seconds": 0.0})
                row["calls"] += 1
                row["inclusive_wall_seconds"] += time.perf_counter() - started
        return wrapped
    boundaries = (
        (rendering, "render_native_tile_window", "native_tile_render"),
        (propagation, "run_mask_injected_session", "sam_session_and_publication"),
        (propagation, "partition_mask_seed_sessions", "seed_partition"),
        (propagation, "read_seed_artifact", "seed_artifact_read"),
        (postprocessing, "fill_binary_mask_holes_2d", "cpu_hole_fill"),
        (experimental, "mask_metrics", "diagnostic_mask_metrics"),
        (worker, "_relay_observation", "relay_observation"),
        (worker, "_write_relay_artifacts", "relay_artifacts"),
        (worker, "_sha256_file", "worker_output_hash"),
        (worker.gc, "collect", "garbage_collection"),
    )
    torch = context.torch_module
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    profile = cProfile.Profile() if payload.get("capture_cprofile") else None
    with ExitStack() as stack:
        for module, name, label in boundaries:
            stack.enter_context(mock.patch.object(module, name, instrument(label, getattr(module, name))))
        started = time.perf_counter()
        if profile is not None:
            profile.enable()
        try:
            result = dict(worker.execute_worker_task(context, kind, payload))
            torch.cuda.synchronize()
        finally:
            if profile is not None:
                profile.disable()
        elapsed = time.perf_counter() - started
    receipt = {
        "worker_task_seconds": elapsed,
        "inclusive_phases": phases,
        "phase_semantics": "inclusive wall times overlap; CUDA synchronization only at task boundaries",
        "profiler_enabled": profile is not None,
        "build_seconds": context.profile_build_seconds,
        "heat_seconds": context.profile_heat_seconds,
        "source_sha256_at_worker_build": context.profile_source_sha256,
        "cuda_allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "cuda_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    profile_root = Path(payload["output_dir"])
    if profile is not None:
        profile.dump_stats(str(profile_root / "worker.pstats"))
        stats = pstats.Stats(profile)
        receipt["cprofile_top_self"] = _profile_rows(stats, "self_seconds")
        receipt["cprofile_top_cumulative"] = _profile_rows(stats, "cumulative_seconds")
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(100)
        (profile_root / "worker_profile.txt").write_text(stream.getvalue(), encoding="utf-8")
    receipt_path = write_json_atomically(profile_root / "profile.json", receipt)
    result["metadata"] = {**result.get("metadata", {}), "profile_path": str(receipt_path)}
    return result


class GpuSampler:
    def __init__(self):
        self.stage = "startup"
        self.samples = []
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        fields = "index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm"
        while not self.stopping.is_set():
            record = {"time_unix": time.time(), "stage": self.stage}
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields,
                                         "--format=csv,noheader,nounits"], capture_output=True,
                                        text=True, check=True, timeout=5)
                record["devices"] = list(csv.reader(result.stdout.strip().splitlines()))
            except (OSError, subprocess.SubprocessError) as exc:
                record["error"] = str(exc)
            self.samples.append(record)
            self.stopping.wait(1.0)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="JSON list of name/cache/cache_shape/seed plus optional neighbors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("auto", "egpu", "h100"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heat-seconds", type=float, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not math.isfinite(args.heat_seconds) or args.heat_seconds < 30 or args.repeats < 2:
        parser.error("profiling requires at least 30 seconds heatsoak and two timed repetitions")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    prepared = []
    for fixture in fixtures:
        plan, seeds = prepare_fixture(cache_path=Path(fixture["cache"]),
                                      cache_shape=tuple(fixture["cache_shape"]),
                                      seed_path=Path(fixture["seed"]), output_dir=output,
                                      tile_left=fixture.get("tile_left", 0),
                                      tile_top=fixture.get("tile_top", 0),
                                      tile_index=fixture.get("tile_index", 0),
                                      require_both_dogfood=fixture.get("require_both_dogfood", False))
        plan["payload"]["neighbors"] = fixture.get("neighbors", [])
        prepared.append((fixture["name"], plan, seeds))
    record = {"status": "running", "command": [sys.executable, *sys.argv],
              "source_sha256": _source_snapshot(), "fixtures": fixtures,
              "fixture_plans": {name: plan for name, plan, seeds in prepared},
              "profile_tool_sha256": _sha256_file(Path(__file__)),
              "heat_seconds": args.heat_seconds, "repeats": args.repeats,
              "results": [], "forced_shutdown_devices": None}
    write_json_atomically(output / "run_plan.json", record)
    sampler = GpuSampler()
    sampler.start()
    pool = None
    try:
        pool = LtaWorkerPool((args.device,), LtaWorkerInit(
            adapter_module="tools.lta_worker_profile", adapter_factory="build_profile_predictor",
            adapter_execute="execute_profile_task", adapter_shutdown="close_profile_predictor",
            adapter_config={"model_path": args.model, "profile": args.profile,
                            "conf": 0.15, "heat_seconds": args.heat_seconds}), startup_timeout=600)
        record["ready_events"] = [asdict(event) for event in pool.ready_events]
        for name, plan, seeds in prepared:
            for repeat in range(-1, args.repeats + 1):
                mode = "warmup" if repeat < 0 else "profiled" if repeat == args.repeats else f"timed-{repeat}"
                label = f"{name}-{mode}"
                sampler.stage = label
                payload = {**plan["payload"], "work_id": label,
                           "output_dir": str(output / label), "capture_cprofile": mode == "profiled"}
                task = LtaWorkerTask(label, label, "propagation_chain", payload)
                started = time.perf_counter()
                pool.submit(task, execution_device_id=args.device)
                result = pool.wait_result(timeout=600)
                elapsed = time.perf_counter() - started
                sampler.stage = label + "-validation"
                started = time.perf_counter()
                manifest = _load_chain_manifest(result)
                load_manifest_seconds = time.perf_counter() - started
                import numpy as np
                commit_path = output / (label + ".coordinator.raw")
                view_union = np.memmap(commit_path, dtype=np.uint8, mode="w+", shape=tuple(plan["payload"]["cache_ref"]["shape"]))
                started = time.perf_counter()
                _consume_chain_manifest(manifest, view_union=view_union)
                coordinator_consume_seconds = time.perf_counter() - started
                view_union._mmap.close()
                commit_path.unlink()
                started = time.perf_counter()
                audit = validate_chain(manifest, plan, seeds)
                profile = json.loads(Path(result.metadata["profile_path"]).read_text())
                parent_seconds = time.perf_counter() - started
                row = {"case": name, "mode": mode, "coordinator_wait_seconds": elapsed,
                       "parent_validation_seconds": parent_seconds,
                       "coordinator_load_manifest_seconds": load_manifest_seconds,
                       "coordinator_consume_seconds": coordinator_consume_seconds,
                       "worker_task_seconds": profile["worker_task_seconds"],
                       "inclusive_phases": profile["inclusive_phases"],
                       "union_sha256": audit["union_sha256"], "active_frame_count": len(audit["active_frames"]),
                       "relay_count": len(manifest["relays"]), "profile_path": result.metadata["profile_path"],
                       "manifest_path": result.artifact_path,
                       "cuda_peak_allocated_mib": profile["cuda_peak_allocated_mib"]}
                record["results"].append(row)
                write_json_atomically(output / "summary.json", record)
                print(json.dumps(row, sort_keys=True), flush=True)
        record["case_summaries"] = {}
        for name, plan, seeds in prepared:
            rows = [row for row in record["results"] if row["case"] == name]
            timed = [row for row in rows if row["mode"].startswith("timed-")]
            hashes = {row["union_sha256"] for row in rows}
            if len(hashes) != 1:
                raise RuntimeError(f"repeated fixture outputs changed: {name}")
            record["case_summaries"][name] = {"all_output_hashes_identical": True,
                "median_worker_seconds": statistics.median(row["worker_task_seconds"] for row in timed),
                "timed_worker_seconds": [row["worker_task_seconds"] for row in timed],
                "union_sha256": next(iter(hashes))}
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        sampler.stage = "shutdown"
        if pool is not None:
            record["forced_shutdown_devices"] = list(pool.shutdown(timeout=30))
        sampler.stop()
        record["gpu_sample_columns"] = ["index", "gpu_utilization_percent", "memory_utilization_percent",
                                         "device_memory_used_mib", "temperature_c", "power_w", "sm_clock_mhz"]
        write_json_atomically(output / "gpu_samples.json", sampler.samples)
        record["source_sha256_after"] = _source_snapshot()
        write_json_atomically(output / "summary.json", record)


if __name__ == "__main__":
    main()
