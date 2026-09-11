"""Replay immutable cached frames through the persistent production LTA worker.

This bounded functional diagnostic uses the production temporal planner, mask
injection, dogfood continuation, and worker shutdown. It does not measure model
accuracy, benchmark throughput, or qualify final-volume publication. Input paths
and dimensions are explicit so retained fixtures can move between machines.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.lta_outputs import write_json_atomically
from XTA.lta_propagation import LtaSeedProvenance, read_seed_artifact
from XTA.lta_rendering import reference_existing_physical_view_cache
from XTA.lta_tiles import TilePlan
from XTA.lta_windows import AnchorDomain, owned_frame_range, plan_domain_windows
from XTA.lta_workers import LtaWorkerInit, LtaWorkerPool, LtaWorkerTask


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot() -> dict[str, str]:
    files = [*sorted((ROOT / "XTA").glob("lta_*.py")), Path(__file__)]
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in files}


def _gpu_snapshot(stage: str) -> dict[str, object]:
    """Observe whole-device usage without importing CUDA in the coordinator."""
    record: dict[str, object] = {"stage": stage, "time_unix": time.time()}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        record["devices"] = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        record["unavailable"] = str(exc)
    return record


def prepare_fixture(
    *,
    cache_path: Path,
    cache_shape: tuple[int, int, int],
    seed_path: Path,
    output_dir: Path,
    tile_left: int = 0,
    tile_top: int = 0,
    tile_index: int = 0,
    conf: float = 0.15,
    require_both_dogfood: bool = False,
) -> tuple[dict[str, object], tuple[object, ...]]:
    """Validate geometry and adopt current paths without editing old evidence."""
    cache_path = Path(cache_path).resolve(strict=True)
    seed_path = Path(seed_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve(strict=False)
    if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
        raise ValueError("confidence must be finite and in [0,1]")
    if len(cache_shape) != 3 or any(value < 1 for value in cache_shape):
        raise ValueError("cache shape must contain three positive TYX dimensions")
    if cache_path.stat().st_size != math.prod(cache_shape):
        raise ValueError("cache byte length does not match uint8 TYX shape")
    seed_digest = _sha256_file(seed_path)
    seeds = read_seed_artifact(seed_path, expected_sha256=seed_digest)
    if not seeds:
        raise ValueError("a production replay requires at least one seed")
    if any(seed.provenance != LtaSeedProvenance.AUTHORITATIVE for seed in seeds):
        raise ValueError("the initial cached replay seed must be authoritative")
    anchors = {seed.frame_index for seed in seeds}
    if len(anchors) != 1:
        raise ValueError("initial masks must share one authoritative anchor frame")
    lineages = {seed.lineage for seed in seeds}
    if len(lineages) != len(seeds):
        raise ValueError("initial masks must have distinct lineages")
    view_ids = {seed.lineage.physical_view_id for seed in seeds}
    sequence_ids = {
        (seed.lineage.volume_id, seed.lineage.runtime_view_id) for seed in seeds
    }
    if len(view_ids) != 1 or len(sequence_ids) != 1:
        raise ValueError("cached replay masks must belong to one physical/runtime view")
    mask_shapes = {tuple(seed.mask.shape) for seed in seeds}
    if len(mask_shapes) != 1:
        raise ValueError("all initial masks must have the same tile shape")
    mask_height, mask_width = next(iter(mask_shapes))
    if mask_height != mask_width:
        raise ValueError("production replay requires square native tile masks")
    tile = TilePlan(
        left=tile_left,
        top=tile_top,
        size=int(mask_width),
        source_width=cache_shape[2],
        source_height=cache_shape[1],
    )
    if tile_index < 0:
        raise ValueError("tile index must be nonnegative")
    windows = plan_domain_windows(
        AnchorDomain(next(iter(anchors)), 0, cache_shape[0])
    )
    dogfood_branches = {w.branch for w in windows if w.seed_kind == "dogfood"}
    if require_both_dogfood and dogfood_branches != {"backward", "forward"}:
        raise ValueError("fixture must plan backward and forward dogfood continuation")
    # Independently check the planner's ownership contract before model startup.
    ownership = [0] * cache_shape[0]
    for window in windows:
        start, stop = owned_frame_range(window)
        for frame in range(start, stop):
            ownership[frame] += 1
    if any(count != 1 for count in ownership):
        raise RuntimeError("planned windows do not own every cached frame exactly once")
    cache_digest = _sha256_file(cache_path)
    cache = reference_existing_physical_view_cache(
        cache_path,
        shape=cache_shape,
        physical_view_id=next(iter(view_ids)),
        source_identity=cache_digest,
    )
    sequence_id = "::".join(next(iter(sequence_ids)))
    payload = {
        "work_id": "cached-production-replay",
        "sequence_id": sequence_id,
        "cache_ref": cache.payload(),
        "tile_index": tile_index,
        "tile": asdict(tile),
        "neighbors": [],
        "seed_artifact_path": str(seed_path),
        "seed_artifact_sha256": seed_digest,
        "windows": [asdict(window) for window in windows],
        "output_frame_start": 0,
        "output_frame_stop": cache_shape[0],
        "conf": conf,
        "empty_frame_limit": 30,
        "relay_generation": 0,
        "relay_min_pixels": 16,
        "relay_min_probability": 0.5,
        "output_dir": str(output_dir / "worker-task"),
    }
    return {
        "schema": "lta.production-smoke/1",
        "cache_sha256": cache_digest,
        "seed_sha256": seed_digest,
        "frame_ownership_counts": ownership,
        "model_frame_visit_bound_per_seed": sum(w.frame_count for w in windows),
        "payload": payload,
    }, seeds


def validate_chain(
    manifest: dict[str, object],
    plan: dict[str, object],
    seeds: tuple[object, ...],
    *,
    require_all_active: bool = False,
) -> dict[str, object]:
    """Check independent output/provenance invariants, not model accuracy."""
    import numpy as np
    from XTA.lta_union_artifacts import (
        iter_union_crops, logical_union_sha256, validate_union_artifact,
    )

    payload = plan["payload"]
    planned_windows = payload["windows"]
    expected_frames = payload["output_frame_stop"]
    if manifest.get("schema") != "lta.propagation-chain/1":
        raise RuntimeError("worker emitted an unsupported propagation manifest schema")
    if manifest.get("status") != "complete":
        raise RuntimeError("production worker did not complete the propagation chain")
    if manifest.get("output_frame_range") != [0, expected_frames]:
        raise RuntimeError("worker output frame range differs from the immutable plan")
    receipts = manifest["windows"]
    for planned in planned_windows:
        matching = [receipt for receipt in receipts if receipt["window"] == planned]
        if not matching or any(item["status"] != "complete" for item in matching):
            raise RuntimeError(f"planned {planned['branch']} window did not complete")
        partition_count = matching[0]["seed_partition_count"]
        if sorted(item["seed_partition_index"] for item in matching) != list(range(partition_count)):
            raise RuntimeError("worker receipts do not cover all seed partitions exactly once")
        for receipt in matching:
            adapter = receipt["adapter"]
            if receipt["retained_prediction_count"] != 0 or adapter.get("canonical_predictions_retained") is not False:
                raise RuntimeError("production worker retained dense predictions")
            expected_provenance = (
                "authoritative" if planned["seed_kind"] == "authoritative" else "temporal_dogfood"
            )
            provenance = adapter.get("prompt_provenance", [])
            if not provenance or set(provenance) != {expected_provenance}:
                raise RuntimeError("worker prompt provenance differs from planned continuation")
    if any(receipt["window"] not in planned_windows for receipt in receipts):
        raise RuntimeError("worker executed an unplanned window")
    union_record = manifest["union"]
    expected_shape = (expected_frames, *seeds[0].mask.shape)
    artifact = validate_union_artifact(union_record, frame_start=0)
    if artifact.shape != expected_shape:
        raise RuntimeError("worker union geometry differs from fixture geometry")
    areas = [0] * expected_frames
    hard_positive_missing = [int(np.count_nonzero(seed.mask)) for seed in seeds]
    for frame, (top, left), crop in iter_union_crops(artifact):
        areas[frame] = int(np.count_nonzero(crop))
        for index, seed in enumerate(seeds):
            if seed.frame_index == frame:
                expected = seed.mask[top:top+crop.shape[0],left:left+crop.shape[1]]
                hard_positive_missing[index] -= int(np.count_nonzero(expected & (crop != 0)))
    if any(hard_positive_missing):
        raise RuntimeError(f"authoritative foreground was lost: {hard_positive_missing}")
    active_frames = [index for index, area in enumerate(areas) if area]
    if require_all_active and len(active_frames) != expected_frames:
        raise RuntimeError("the required all-frame activity gate failed")
    active_lineages = manifest.get("lineage_active_frame_ranges", {})
    if any(seed.lineage.token not in active_lineages for seed in seeds):
        raise RuntimeError("a seeded lineage is absent from the activity audit")
    return {
        "planned_window_count": len(planned_windows),
        "tracker_session_count": len(receipts),
        "window_branches": [window["branch"] for window in planned_windows],
        "window_statuses": [receipt["status"] for receipt in receipts],
        "prompt_provenance": [receipt["adapter"]["prompt_provenance"] for receipt in receipts],
        "retained_prediction_counts": [receipt["retained_prediction_count"] for receipt in receipts],
        "hole_fill_added_pixels": [receipt["hole_fill_added_pixels"] for receipt in receipts],
        "anchor_propagation_ious": [
            receipt["adapter"].get("anchor_preview_propagation_iou") for receipt in receipts
        ],
        "active_frames": active_frames,
        "foreground_pixels_per_frame": areas,
        "union_foreground_pixels": sum(areas),
        "hard_positive_missing_pixels": hard_positive_missing,
        "lineage_active_frame_ranges": active_lineages,
        "union_sha256": logical_union_sha256(artifact),
        "stored_union_sha256": artifact.sha256,
        "stored_union_bytes": artifact.size_bytes,
        "union_encoding": artifact.encoding,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True, help="Existing SAM 3.1 model bundle")
    parser.add_argument("--cache", required=True, type=Path, help="Immutable raw uint8 TYX cache")
    parser.add_argument("--cache-shape", required=True, type=int, nargs=3, metavar=("T", "Y", "X"))
    parser.add_argument("--seed", required=True, type=Path, help="Existing production seed NPZ")
    parser.add_argument("--output", required=True, type=Path, help="New or empty evidence directory")
    parser.add_argument("--tile-left", type=int, default=0)
    parser.add_argument("--tile-top", type=int, default=0)
    parser.add_argument("--tile-index", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--profile", choices=("auto", "h100", "egpu"), default="auto")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--task-timeout", type=float, default=900.0)
    parser.add_argument("--require-both-dogfood", action="store_true")
    parser.add_argument("--require-all-active", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    from XTA.lta_sam import resolve_local_sam_bundle

    args = _build_parser().parse_args()
    for name in ("startup_timeout", "task_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    output = args.output.resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"--output must be new or empty: {output}")
    plan, seeds = prepare_fixture(
        cache_path=args.cache,
        cache_shape=tuple(args.cache_shape),
        seed_path=args.seed,
        output_dir=output,
        tile_left=args.tile_left,
        tile_top=args.tile_top,
        tile_index=args.tile_index,
        conf=args.conf,
        require_both_dogfood=args.require_both_dogfood,
    )
    bundle = resolve_local_sam_bundle(args.model)
    plan["controls"] = {
        "profile": args.profile,
        "device": args.device,
        "conf": args.conf,
        "require_both_dogfood": args.require_both_dogfood,
        "require_all_active": args.require_all_active,
        "qualification": "bounded production CUDA/control flow only; no accuracy or performance claim",
    }
    plan["model"] = {
        "checkpoint_path": str(bundle.checkpoint_path),
        "checkpoint_sha256": _sha256_file(bundle.checkpoint_path),
    }
    plan["source_sha256"] = _source_snapshot()
    plan["command"] = [sys.executable, *sys.argv]
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomically(output / "run_plan.json", plan)
    if args.plan_only:
        print(json.dumps({"status": "planned", "output": str(output)}, sort_keys=True), flush=True)
        return
    summary: dict[str, object] = {
        "schema": "lta.production-smoke/1",
        "status": "running",
        "gpu_memory_semantics": "nvidia-smi whole-device MiB, including other processes; sampled, not allocator peak",
        "gpu_snapshots": [_gpu_snapshot("before_worker")],
        "forced_shutdown_devices": None,
    }
    pool = None
    failure = None
    try:
        pool = LtaWorkerPool(
            (args.device,),
            LtaWorkerInit(
                adapter_module="XTA.lta_worker_adapter",
                adapter_factory="build_worker_predictor",
                adapter_execute="execute_worker_task",
                adapter_shutdown="close_worker_predictor",
                adapter_config={"model_path": str(bundle.checkpoint_path), "profile": args.profile, "conf": args.conf},
            ),
            startup_timeout=args.startup_timeout,
        )
        summary["ready_events"] = [asdict(event) for event in pool.ready_events]
        summary["gpu_snapshots"].append(_gpu_snapshot("worker_ready"))
        task = LtaWorkerTask(
            work_id=plan["payload"]["work_id"],
            attempt_token="cached-replay-attempt",
            kind="propagation_chain",
            payload=plan["payload"],
        )
        pool.submit(task, execution_device_id=args.device)
        deadline = time.monotonic() + args.task_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("production replay exceeded its task watchdog")
            try:
                result = pool.wait_result(timeout=min(15.0, remaining))
                break
            except TimeoutError:
                summary["gpu_snapshots"].append(_gpu_snapshot("worker_running"))
        summary["worker_result"] = asdict(result)
        if result.worker_pid != pool.ready_events[0].worker_pid:
            raise RuntimeError("the completing worker differs from the initialized worker")
        manifest_path = Path(result.artifact_path)
        if _sha256_file(manifest_path) != result.artifact_sha256:
            raise RuntimeError("worker manifest digest changed before validation")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary.update(validate_chain(manifest, plan, seeds, require_all_active=args.require_all_active))
        summary["status"] = "complete"
    except BaseException as exc:
        failure = exc
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if pool is not None:
            try:
                forced = pool.shutdown(timeout=30.0)
                summary["forced_shutdown_devices"] = list(forced)
                summary["worker_pool_closed"] = pool.closed
                if forced:
                    raise RuntimeError(f"worker shutdown required forced termination: {forced}")
            except BaseException as exc:
                summary["status"] = "failed"
                summary["shutdown_error"] = str(exc)
                failure = failure or exc
        summary["gpu_snapshots"].append(_gpu_snapshot("after_shutdown"))
        summary["source_sha256_after"] = _source_snapshot()
        summary["source_unchanged"] = summary["source_sha256_after"] == plan["source_sha256"]
        if _sha256_file(args.cache) != plan["cache_sha256"] or _sha256_file(args.seed) != plan["seed_sha256"]:
            summary["status"] = "failed"
            summary["fixture_unchanged"] = False
            failure = failure or RuntimeError("input fixture changed during replay")
        else:
            summary["fixture_unchanged"] = True
        write_json_atomically(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "output": str(output)}, sort_keys=True), flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
