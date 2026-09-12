"""Compare isolated LTA process slots on one physical GPU using cached fixtures.

The fixed workload is identical at one and two workers per GPU. Models are
loaded once per process, heatsoaked, warmed up, then timed without per-prediction
hashing. A separate concurrent qualification batch checks exact object masks,
scores and relay artifacts. This measures independent propagation-chain tasks,
not a complete volume or the production relay fixed point. No MPS assumption is
made; overlapping host work is not proof of concurrent CUDA kernel execution.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import uuid
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lta_production_smoke import prepare_fixture, validate_chain, _sha256_file
from XTA.lta_outputs import write_json_atomically
from XTA.lta_propagation import read_seed_artifact
from XTA.lta_telemetry import lta_source_fingerprint
from XTA.lta_workers import LtaWorkerInit, LtaWorkerPool, LtaWorkerTask
from XTA import __version__


def build_overlap_context(config):
    from XTA.lta_worker_adapter import build_worker_predictor
    started = time.monotonic()
    context = build_worker_predictor(config)
    context.overlap_build_seconds = time.monotonic() - started
    context.overlap_source_fingerprint = lta_source_fingerprint()
    context.overlap_runtime = {
        "torch_version": str(context.torch_module.__version__),
        "torch_cuda_runtime": context.torch_module.version.cuda,
        "device_name": context.torch_module.cuda.get_device_name(0),
        "platform": sys.platform,
        "mps_environment": {key: os.environ.get(key) for key in (
            "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        )},
        "mps_activation_verified": False,
    }
    return context


def close_overlap_context(context):
    from XTA.lta_worker_adapter import close_worker_predictor
    close_worker_predictor(context)


def execute_overlap_task(context, kind, payload):
    """Worker entry point; production code is unchanged outside this process."""
    import numpy as np
    from XTA import lta_propagation as propagation
    from XTA.lta_worker_adapter import execute_worker_task

    torch = context.torch_module
    directory = Path(payload["output_dir"])
    if kind == "heatsoak":
        seconds = float(payload["heat_seconds"])
        if not math.isfinite(seconds) or seconds < 30:
            raise ValueError("benchmark heatsoak must last at least 30 seconds")
        started = time.monotonic()
        with torch.inference_mode():
            left = torch.full((4096, 4096), 0.001, device="cuda", dtype=torch.float32)
            right = torch.empty_like(left)
            while time.monotonic() - started < seconds:
                torch.mm(left, left, out=right)
                torch.cuda.synchronize()
            del left, right
        torch.cuda.empty_cache()
        path = write_json_atomically(directory / "heatsoak.json", {
            "requested_seconds": seconds, "wall_seconds": time.monotonic() - started,
            "worker_build_seconds": context.overlap_build_seconds,
        })
        return {"artifact_path": str(path)}
    if kind != "propagation_chain":
        raise ValueError(f"unsupported overlap task kind {kind!r}")

    records = []
    original = propagation.run_mask_injected_session

    def traced(measured, raw, **kwargs):
        request = kwargs["request"]
        callback = kwargs["prediction_callback"]

        def observe(item):
            prediction = item.prediction
            mask = np.ascontiguousarray(prediction.binary_mask, dtype=np.uint8)
            records.append({
                "lineage": item.lineage.token,
                "frame": int(prediction.frame_index),
                "object_id": int(prediction.object_id),
                "session_range": [request.session.frame_start, request.session.frame_stop],
                "direction": request.direction,
                "initial_detection_score": prediction.initial_detection_score,
                "frame_tracker_score": prediction.frame_tracker_score,
                "mask_sha256": hashlib.sha256(memoryview(mask)).hexdigest(),
            })
            callback(item)

        kwargs["prediction_callback"] = observe
        return original(measured, raw, **kwargs)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_ns = time.monotonic_ns()
    if payload.get("collect_prediction_evidence", False):
        with mock.patch.object(propagation, "run_mask_injected_session", traced):
            result = dict(execute_worker_task(context, kind, payload))
    else:
        result = dict(execute_worker_task(context, kind, payload))
    torch.cuda.synchronize()
    ended_ns = time.monotonic_ns()
    evidence = None
    if payload.get("collect_prediction_evidence", False):
        records.sort(key=lambda record: (record["lineage"], record["frame"], record["session_range"], record["direction"], record["object_id"]))
        evidence = write_json_atomically(directory / "prediction_evidence.json", records)
    profile_path = write_json_atomically(directory / "overlap_task.json", {
        "work_id": payload["work_id"], "started_ns": started_ns, "ended_ns": ended_ns,
        "worker_task_seconds": (ended_ns - started_ns) / 1e9,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cuda_allocated_bytes_after_task": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes_after_task": int(torch.cuda.memory_reserved()),
        "source_fingerprint": context.overlap_source_fingerprint,
        "runtime": context.overlap_runtime,
        "prediction_evidence_path": None if evidence is None else str(evidence),
        "timing_semantics": "host task time with current-process CUDA synchronization at its boundaries",
    })
    result["metadata"] = {**result.get("metadata", {}), "overlap_profile_path": str(profile_path)}
    return result


class DeviceSampler:
    """Sample every physical GPU; retain identities so logical indices are unambiguous."""
    def __init__(self, interval):
        self.interval = interval
        self.stage = "startup"
        self.samples = []
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        fields = "index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm,driver_version"
        while not self.stopping.is_set():
            row = {"monotonic_ns": time.monotonic_ns(), "time_unix_ns": time.time_ns(), "stage": self.stage}
            try:
                output = subprocess.run(["nvidia-smi", "--query-gpu=" + fields,
                                         "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, check=True, timeout=5)
                row["devices"] = [[value.strip() for value in values] for values in csv.reader(output.stdout.splitlines())]
            except (OSError, subprocess.SubprocessError) as error:
                row["error"] = str(error)
            self.samples.append(row)
            self.stopping.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.thread.join(timeout=6)


def interval_overlap(intervals, start_ns, end_ns):
    events = {}
    for start, end in intervals:
        start, end = max(start_ns, start), min(end_ns, end)
        if end <= start:
            continue
        events[start] = events.get(start, 0) + 1
        events[end] = events.get(end, 0) - 1
    current = maximum = area = 0
    previous = start_ns
    histogram = {}
    for instant, delta in sorted(events.items()):
        duration = instant - previous
        histogram[current] = histogram.get(current, 0) + duration
        area += current * duration
        current += delta
        maximum = max(maximum, current)
        previous = instant
    histogram[current] = histogram.get(current, 0) + end_ns - previous
    span = end_ns - start_ns
    return {"maximum": maximum, "time_weighted_mean": area / span if span else 0,
            "seconds_by_active_count": {str(key): value / 1e9 for key, value in sorted(histogram.items())}}


def phase_overlap(trace_root, start_ns, end_ns):
    starts, intervals = {}, []
    for path in trace_root.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            instant = int(event.get("monotonic_ns", 0))
            if not start_ns <= instant <= end_ns:
                continue
            key = event.get("pid"), event.get("span_id")
            if event.get("event") == "phase_start":
                starts[key] = event
            elif event.get("event") == "phase_end" and key in starts:
                begin = starts.pop(key)
                intervals.append((begin.get("phase"), begin.get("pid"), int(begin["monotonic_ns"]), instant))
    sessions = [(pid, start, end) for phase, pid, start, end in intervals if phase == "sam_session"]
    preparation = [(pid, start, end) for phase, pid, start, end in intervals if phase in {"render_window", "seed_partition"}]
    overlapped_ns = 0
    for pid, start, end in preparation:
        clipped = sorted((max(start, other_start), min(end, other_end))
                         for other_pid, other_start, other_end in sessions
                         if other_pid != pid and max(start, other_start) < min(end, other_end))
        cursor = start
        for left, right in clipped:
            if right > cursor:
                overlapped_ns += right - max(cursor, left)
                cursor = right
    return {
        "sam_session_context_concurrency": interval_overlap([(start, end) for _pid, start, end in sessions], start_ns, end_ns),
        "cpu_preparation_seconds": sum(end - start for _pid, start, end in preparation) / 1e9,
        "cpu_preparation_overlapped_by_other_process_session_seconds": overlapped_ns / 1e9,
        "unclosed_phase_count": len(starts),
        "semantics": "host intervals only; session presence includes CPU work and does not measure CUDA kernel overlap",
    }


def _relay_signature(manifest):
    import numpy as np
    records = []
    for record in manifest.get("relays", ()):
        seeds = read_seed_artifact(record["seed_artifact_path"], expected_sha256=record["seed_artifact_sha256"])
        records.append({
            "lineage": record["lineage"], "destination": record["destination_tile_index"],
            "frame": record["frame_index"], "direction": record["temporal_direction"],
            "generation": record["generation"], "artifact_sha256": record["seed_artifact_sha256"],
            "seeds": [{"lineage": seed.lineage.token, "object_id": seed.object_id,
                       "probability": seed.tracker_probability, "visited_tiles": list(seed.visited_tile_indices),
                       "mask_sha256": hashlib.sha256(np.ascontiguousarray(seed.mask, dtype=np.uint8).tobytes()).hexdigest()}
                      for seed in seeds],
        })
    return sorted(records, key=lambda record: json.dumps(record, sort_keys=True))


def _sample_summary(samples, token, start_ns, end_ns):
    selected = []
    for sample in samples:
        if not start_ns <= sample["monotonic_ns"] <= end_ns:
            continue
        matches = [row for row in sample.get("devices", ()) if row[0] == str(token) or row[1].startswith(str(token))]
        if len(matches) == 1:
            selected.append(matches[0])
    def numeric(column):
        values = []
        for row in selected:
            try:
                values.append(float(row[column]))
            except (IndexError, ValueError):
                pass
        return values
    result = {"matched_device_sample_count": len(selected),
              "driver_versions": sorted({row[9] for row in selected if len(row) > 9})}
    for name, column in (("memory_used_mib", 3), ("gpu_utilization_percent", 5), ("temperature_c", 6), ("power_w", 7), ("sm_clock_mhz", 8)):
        values = numeric(column)
        result[name] = None if not values else {"mean": statistics.mean(values), "sampled_maximum": max(values)}
    return result


def _run_batch(pool, jobs, *, root, device, collect_prediction_evidence=False):
    pending = deque(jobs)
    active, results = {}, []
    start_ns = time.monotonic_ns()

    def submit(worker_index):
        job = pending.popleft()
        name, ordinal, plan, _seeds = job
        work_id = f"overlap::{name}::job-{ordinal:04d}"
        payload = {**plan["payload"], "work_id": work_id,
                   "output_dir": str(root / f"{name}-{ordinal:04d}"),
                   "collect_prediction_evidence": collect_prediction_evidence,
                   "worker_trace_root": str(root.parent / "worker-traces")}
        task = LtaWorkerTask(work_id, uuid.uuid4().hex, "propagation_chain", payload)
        pool.submit(task, execution_device_id=device, worker_index=worker_index)
        active[worker_index] = (job, work_id)

    for _device, worker_index in pool.worker_slots:
        if pending:
            submit(worker_index)
    while active:
        result = pool.wait_result(timeout=900)
        worker_index = int(result.worker_index)
        job, work_id = active.pop(worker_index)
        if result.work_id != work_id or result.execution_device_id != device:
            raise RuntimeError("overlap result does not match the assigned process slot")
        results.append((job, result))
        if pending:
            submit(worker_index)
    end_ns = time.monotonic_ns()
    return {"start_ns": start_ns, "end_ns": end_ns, "wall_seconds": (end_ns - start_ns) / 1e9}, results


def _validate_batch(results, baseline_basic, baseline_predictions, *, qualification):
    rows, intervals = [], []
    for (name, ordinal, plan, seeds), result in results:
        path = Path(result.artifact_path)
        if _sha256_file(path) != result.artifact_sha256:
            raise RuntimeError("overlap worker manifest changed after completion")
        manifest = json.loads(path.read_text())
        audit = validate_chain(manifest, plan, seeds)
        key = f"{name}::{ordinal}"
        basic = {"union_sha256": audit["union_sha256"], "relays": _relay_signature(manifest)}
        if baseline_basic.setdefault(key, basic) != basic:
            raise RuntimeError(f"exact union or relay parity failed for {key}")
        profile = json.loads(Path(result.metadata["overlap_profile_path"]).read_text())
        if qualification:
            evidence = Path(profile["prediction_evidence_path"])
            predictions = json.loads(evidence.read_text())
            if baseline_predictions.setdefault(key, predictions) != predictions:
                raise RuntimeError(f"exact per-object mask/score parity failed for {key}")
        intervals.append((int(profile["started_ns"]), int(profile["ended_ns"])))
        rows.append({"fixture": name, "job_ordinal": ordinal, "worker_index": result.worker_index,
                     "worker_pid": result.worker_pid, "manifest_path": result.artifact_path,
                     "profile": profile, "union_sha256": audit["union_sha256"],
                     "tracker_session_count": audit["tracker_session_count"],
                     "relay_count": len(manifest.get("relays", ()))})
    return rows, intervals


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True, help="Existing local SAM3.1 bundle; one model copy per process slot")
    parser.add_argument("--fixtures", type=Path, required=True, help="JSON list of name/cache/cache_shape/seed with optional tile fields and neighbors")
    parser.add_argument("--output", type=Path, required=True, help="New or empty persistent evidence directory")
    parser.add_argument("--device", type=int, default=0, help="One logical CUDA device; every process slot shares this physical GPU")
    parser.add_argument("--workers-per-gpu", nargs="+", type=int, choices=range(1, 5), default=[1, 2], help="Configurations in measurement order; default compares 1 then 2 isolated processes")
    parser.add_argument("--jobs-per-fixture", type=int, default=4, help="Same independent task count in each timed repeat/configuration")
    parser.add_argument("--repeats", type=int, default=3, help="Unprofiled timed repetitions; exact prediction evidence is collected separately")
    parser.add_argument("--heat-seconds", type=float, default=60, help="Sustained GPU heatsoak before warmup/timing; minimum 30 seconds")
    parser.add_argument("--sample-interval", type=float, default=0.5, help="nvidia-smi sampling interval; sampled peaks may miss brief allocations")
    parser.add_argument("--profile", choices=("auto", "egpu", "h100"), default="auto")
    return parser


def main():
    args = _parser().parse_args()
    if not math.isfinite(args.heat_seconds) or args.heat_seconds < 30 or args.repeats < 2:
        raise SystemExit("Use at least 30 seconds heatsoak and two timed repetitions")
    if args.jobs_per_fixture < max(args.workers_per_gpu):
        raise SystemExit("jobs-per-fixture must be at least the largest worker count")
    if not math.isfinite(args.sample_interval) or args.sample_interval < 0.2:
        raise SystemExit("sample-interval must be finite and at least 0.2 seconds")
    if len(set(args.workers_per_gpu)) != len(args.workers_per_gpu):
        raise SystemExit("workers-per-gpu configurations must be unique")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit("output must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(args.fixtures.read_text())
    if not isinstance(fixtures, list) or not fixtures:
        raise SystemExit("fixtures must be a nonempty JSON list")
    prepared, names = [], set()
    for index, fixture in enumerate(fixtures):
        name = str(fixture["name"])
        if not name or name in names or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in name):
            raise SystemExit("fixture names must be unique letters, digits, hyphens or underscores")
        names.add(name)
        plan, seeds = prepare_fixture(
            cache_path=Path(fixture["cache"]), cache_shape=tuple(fixture["cache_shape"]),
            seed_path=Path(fixture["seed"]), output_dir=root / "fixture-plans" / name,
            tile_left=int(fixture.get("tile_left", 0)), tile_top=int(fixture.get("tile_top", 0)),
            tile_index=int(fixture.get("tile_index", 0)), conf=float(fixture.get("conf", 0.15)),
            require_both_dogfood=bool(fixture.get("require_both_dogfood", False)),
        )
        plan["payload"]["neighbors"] = fixture.get("neighbors", [])
        prepared.append((name, plan, seeds))
    jobs = [(name, ordinal, plan, seeds) for name, plan, seeds in prepared for ordinal in range(args.jobs_per_fixture)]
    report = {"status": "running", "command": [sys.executable, *sys.argv],
              "pipeline_version": __version__, "platform": sys.platform,
              "source_fingerprint": lta_source_fingerprint(), "tool_sha256": _sha256_file(Path(__file__)),
              "fixture_plans": {name: plan for name, plan, _seeds in prepared},
              "heat_seconds": args.heat_seconds, "repeats": args.repeats,
              "jobs_per_timed_repeat": len(jobs), "configurations": [],
              "limits": ["Independent cached-chain throughput, not full-volume/fixed-point runtime.",
                         "Separate process models duplicate GPU memory; no shared-weight thread safety claim.",
                         "Host phase overlap is not CUDA kernel concurrency; no MPS assumption.",
                         "Per-object mask/score hashes are collected only in separate qualification batches."]}
    baseline_basic, baseline_predictions = {}, {}
    sampler = DeviceSampler(args.sample_interval)
    sampler.start()
    try:
        for count in args.workers_per_gpu:
            directory = root / f"workers-{count}"
            configuration = {"workers_per_gpu": count, "timed_batches": [], "status": "running"}
            report["configurations"].append(configuration)
            pool = None
            try:
                sampler.stage = f"workers-{count}:startup"
                started = time.monotonic()
                pool = LtaWorkerPool((args.device,), LtaWorkerInit(
                    adapter_module="tools.lta_overlap_profile", adapter_factory="build_overlap_context",
                    adapter_execute="execute_overlap_task", adapter_shutdown="close_overlap_context",
                    adapter_config={"model_path": args.model, "profile": args.profile, "conf": 0.15,
                                    "trace_dir": str(directory / "worker-traces")}),
                    startup_timeout=900, workers_per_device=count)
                configuration["startup_seconds"] = time.monotonic() - started
                configuration["ready_events"] = [asdict(event) for event in pool.ready_events]
                configuration["pids_by_slot"] = {f"{device}:{slot}": pid for (device, slot), pid in pool.pids_by_slot.items()}
                configuration["visible_device_token"] = pool.visible_device_tokens[args.device]
                sampler.stage = f"workers-{count}:heatsoak"
                for device, slot in pool.worker_slots:
                    task = LtaWorkerTask(f"heat-{slot}", uuid.uuid4().hex, "heatsoak", {
                        "heat_seconds": args.heat_seconds, "output_dir": str(directory / f"heat-{slot}")})
                    pool.submit(task, execution_device_id=device, worker_index=slot)
                configuration["heatsoak_receipts"] = []
                for _ in pool.worker_slots:
                    result = pool.wait_result(timeout=max(900, args.heat_seconds * count + 120))
                    configuration["heatsoak_receipts"].append(json.loads(Path(result.artifact_path).read_text()))
                for name, plan, seeds in prepared:
                    sampler.stage = f"workers-{count}:warmup:{name}"
                    _batch, results = _run_batch(pool, [(name, slot, plan, seeds) for slot in range(count)], root=directory / f"warmup-{name}", device=args.device)
                    _validate_batch(results, {}, {}, qualification=False)
                for repeat in range(args.repeats):
                    label = f"workers-{count}:timed-{repeat}"
                    sampler.stage = label
                    batch, results = _run_batch(pool, jobs, root=directory / f"timed-{repeat}", device=args.device)
                    sampler.stage = label + ":validation"
                    rows, intervals = _validate_batch(results, baseline_basic, baseline_predictions, qualification=False)
                    batch.update({"repeat": repeat, "tasks": rows,
                                  "jobs_per_second": len(jobs) / batch["wall_seconds"],
                                  "tracker_sessions_per_second": sum(row["tracker_session_count"] for row in rows) / batch["wall_seconds"],
                                  "host_task_concurrency": interval_overlap(intervals, batch["start_ns"], batch["end_ns"]),
                                  "phases": phase_overlap(directory / "worker-traces", batch["start_ns"], batch["end_ns"])})
                    configuration["timed_batches"].append(batch)
                    write_json_atomically(root / "summary.json", report)
                    print(f"workers_per_gpu={count} repeat={repeat} jobs={len(jobs)} wall_seconds={batch['wall_seconds']:.3f} jobs_per_second={batch['jobs_per_second']:.3f}", flush=True)
                sampler.stage = f"workers-{count}:qualification"
                _batch, results = _run_batch(pool, jobs, root=directory / "qualification", device=args.device, collect_prediction_evidence=True)
                rows, _intervals = _validate_batch(results, baseline_basic, baseline_predictions, qualification=True)
                configuration["qualification"] = {"exact_masks_scores_relays": True, "tasks": rows}
                configuration["median_batch_seconds"] = statistics.median(batch["wall_seconds"] for batch in configuration["timed_batches"])
                configuration["status"] = "complete"
            except BaseException as error:
                configuration["status"] = "failed"
                configuration["error"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                sampler.stage = f"workers-{count}:shutdown"
                if pool is not None:
                    forced = pool.shutdown(timeout=60, force=True)
                    configuration["forced_shutdown_devices"] = list(forced)
                    if forced:
                        raise RuntimeError(f"overlap workers required forced shutdown: {forced}")
        baseline = report["configurations"][0]["median_batch_seconds"]
        for configuration in report["configurations"]:
            configuration["throughput_speedup_vs_first_configuration"] = baseline / configuration["median_batch_seconds"]
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        sampler.stage = "finished"
        sampler.stop()
        report["gpu_sample_columns"] = ["physical_index", "uuid", "name", "memory_used_mib", "memory_total_mib", "gpu_utilization_percent", "temperature_c", "power_w", "sm_clock_mhz", "driver_version"]
        for configuration in report["configurations"]:
            for batch in configuration["timed_batches"]:
                batch["sampled_device"] = _sample_summary(sampler.samples, configuration.get("visible_device_token"), batch["start_ns"], batch["end_ns"])
        report["source_fingerprint_after"] = lta_source_fingerprint()
        write_json_atomically(root / "gpu_samples.json", sampler.samples)
        write_json_atomically(root / "summary.json", report)


if __name__ == "__main__":
    main()
