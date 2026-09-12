"""Replay the complete bounded LTA relay fixed point from a selected source tree.

Run this tool separately against retained and candidate source roots, then
compare their native_union.npy files and per-object evidence. This is a GPU
functional qualification, not a throughput benchmark; it makes no speed claim
and does not heatsoak. The selected source's existing helper limits relay
generations to 20 and keeps the normal model/seed validation intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock


def _select_source_root():
    """Run before any XTA/tools import, including in spawned child bootstrap."""
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args, _remaining = bootstrap.parse_known_args()
    root = args.source_root.expanduser().resolve(strict=True)
    if not (root / "XTA" / "lta_execution.py").is_file():
        raise SystemExit(f"source-root does not contain XTA/lta_execution.py: {root}")
    if not (root / "tools" / "lta_window_gpu_smoke.py").is_file():
        raise SystemExit(f"source-root does not contain tools/lta_window_gpu_smoke.py: {root}")
    root_text = str(root)
    sys.path[:] = [root_text, *(value for value in sys.path if value != root_text)]
    return root


SOURCE_ROOT = _select_source_root()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT,
                        help="Retained or candidate repository; selected before XTA imports in parent and workers")
    parser.add_argument("--model", required=True, help="Existing SAM3.1 model bundle")
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="Existing JSON fixture list: single-dogfood, multi-dogfood and/or east-relay")
    parser.add_argument("--output", type=Path, required=True, help="New or empty evidence directory")
    parser.add_argument("--device", type=int, default=0, help="One logical CUDA device")
    parser.add_argument("--profile", choices=("auto", "egpu", "h100"), default="auto")
    parser.add_argument("--plan-only", action="store_true", help="Validate source and portable fixtures without starting a GPU worker")
    return parser


def main():
    args = _parser().parse_args()
    if args.device < 0:
        raise SystemExit("device must be nonnegative")

    # Keep these imports inside main. Spawned children execute only the source
    # bootstrap above; their worker entry point binds CUDA before adapter import.
    import numpy as np
    package = importlib.import_module("XTA")
    execution = importlib.import_module("XTA.lta_execution")
    helper = importlib.import_module("tools.lta_window_gpu_smoke")
    workers = importlib.import_module("XTA.lta_workers")
    normalize_path = getattr(execution, "_artifact_ownership_path", lambda value: value)
    for module in (package, execution, helper, workers):
        try:
            normalize_path(Path(module.__file__).resolve(strict=True)).relative_to(normalize_path(SOURCE_ROOT))
        except ValueError as error:
            raise RuntimeError(f"module was imported outside selected source-root: {module.__name__}: {module.__file__}") from error

    root = args.output.expanduser().resolve(strict=False)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise SystemExit("output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or not fixtures:
        raise SystemExit("fixtures must be a nonempty JSON list")
    allowed = {"single-dogfood", "multi-dogfood", "east-relay", "east-relay-long"}
    names = [str(fixture["name"]) for fixture in fixtures]
    if len(set(names)) != len(names) or any(name not in allowed for name in names):
        raise SystemExit("fixtures must use unique supported names: single-dogfood, multi-dogfood, east-relay, east-relay-long")
    prepared = []
    for fixture in fixtures:
        name = str(fixture["name"])
        if any(int(fixture.get(key, 0)) != 0 for key in ("tile_index", "tile_left", "tile_top")):
            raise SystemExit(f"fixture must seed tile 0 at its native origin: {name}")
        cache_path = Path(fixture["cache"])
        seed_path = Path(fixture["seed"])
        if not cache_path.is_absolute():
            cache_path = args.fixtures.resolve().parent / cache_path
        if not seed_path.is_absolute():
            seed_path = args.fixtures.resolve().parent / seed_path
        cache_path = cache_path.resolve(strict=True)
        seed_path = seed_path.resolve(strict=True)
        cache_hash, seed_hash = _sha256(cache_path), _sha256(seed_path)
        seeds = helper.read_seed_artifact(seed_path, expected_sha256=seed_hash)
        shape = tuple(int(value) for value in fixture["cache_shape"])
        if len(shape) != 3 or any(value < 1 for value in shape) or cache_path.stat().st_size != int(np.prod(shape)):
            raise ValueError(f"fixture cache shape/byte length is invalid: {name}")
        if not seeds or len({seed.frame_index for seed in seeds}) != 1:
            raise ValueError(f"fixture must contain masks at one authoritative frame: {name}")
        size = int(seeds[0].mask.shape[0])
        scope = seeds[0].lineage
        if any(
            seed.provenance.value != "authoritative" or tuple(seed.mask.shape) != (size, size)
            or (seed.lineage.volume_id, seed.lineage.physical_view_id, seed.lineage.runtime_view_id, seed.lineage.tile_config_id)
            != (scope.volume_id, scope.physical_view_id, scope.runtime_view_id, scope.tile_config_id)
            or not 0 <= seed.frame_index < shape[0]
            for seed in seeds
        ):
            raise ValueError(f"fixture seed identity/geometry/provenance is inconsistent: {name}")
        cache = helper.reference_existing_physical_view_cache(
            cache_path, shape=shape, physical_view_id=scope.physical_view_id,
            source_identity="immutable-" + name,
        )
        grid = helper.LtaTileGridPlan(
            scope.tile_config_id, size, 3 * size // 4,
            helper.plan_tile_grid(source_width=shape[2], source_height=shape[1],
                                  tile_size=size, tile_stride=3 * size // 4),
        )
        view = SimpleNamespace(
            volume_id=scope.volume_id, physical_view_id=scope.physical_view_id,
            runtime_view_id=scope.runtime_view_id, frame_count=shape[0],
            frame_height=shape[1], frame_width=shape[2], tile_grids=(grid,),
        )
        prepared.append((name, cache, grid, view, seeds, {
            "cache_path": str(cache_path), "cache_sha256": cache_hash,
            "seed_path": str(seed_path), "seed_sha256": seed_hash,
            "cache_shape_tyx": list(shape), "initial_seed_count": len(seeds),
            "fixture_provenance": fixture.get("provenance", {}),
        }))

    fingerprint = helper.lta_source_fingerprint()
    report = {
        "schema": "lta.relay-frontier-smoke/1", "status": "running",
        "qualification": "bounded single-GPU fixed-point functional replay; no speed claim",
        "source_root": str(SOURCE_ROOT), "pipeline_version": package.__version__,
        "source_fingerprint": fingerprint, "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": _sha256(Path(__file__)),
        "selected_helper_path": str(Path(helper.__file__).resolve()),
        "selected_helper_sha256": _sha256(Path(helper.__file__)),
        "command": [sys.executable, *sys.argv], "device": args.device, "profile": args.profile,
        "max_relay_generation": 20, "windowed": True,
        "fixtures": {name: inputs for name, _cache, _grid, _view, _seeds, inputs in prepared},
        "fixture_results": {}, "scores_root": str(root / "scores"),
        "forced_shutdown_devices": None,
    }
    helper.write_json_atomically(root / "summary.json", report)
    print(f"LTA relay frontier replay: source_root={SOURCE_ROOT} source_sha256={fingerprint['sha256']} output={root}", flush=True)
    if args.plan_only:
        report["status"] = "planned"
        helper.write_json_atomically(root / "summary.json", report)
        return
    pool = None
    primary_error = None
    try:
        pool = helper.LtaWorkerPool((args.device,), helper.LtaWorkerInit(
            adapter_module="tools.lta_window_gpu_smoke", adapter_factory="build_context",
            adapter_execute="execute_traced", adapter_shutdown="close_worker_predictor",
            adapter_config={"model_path": args.model, "profile": args.profile, "conf": 0.15,
                            "score_root": str(root / "scores"), "trace_dir": str(root / "worker-traces")},
        ), startup_timeout=600)
        report["ready_events"] = [asdict(event) for event in pool.ready_events]
        for name, cache, grid, view, seeds, inputs in prepared:
            case_root = root / name
            capture = {}
            original_driver = execution._drive_workers_to_fixed_point

            def save_native_union(*driver_args, **driver_kwargs):
                result = original_driver(*driver_args, **driver_kwargs)
                native = np.asarray(driver_kwargs["view_union"])
                if native.dtype != np.uint8 or tuple(native.shape) != tuple(cache.shape):
                    raise RuntimeError("settled native union does not match fixture storage")
                destination = case_root / "native_union.npy"
                temporary = destination.with_suffix(".npy.partial")
                try:
                    with temporary.open("wb") as stream:
                        np.save(stream, native, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, destination)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
                tile = grid.tiles[0]
                hard_positives = []
                for seed in seeds:
                    frame = native[seed.frame_index, tile.top:tile.top + tile.size,
                                   tile.left:tile.left + tile.size] != 0
                    hard_positives.append({
                        "lineage": seed.lineage.token, "frame_index": seed.frame_index,
                        "seed_foreground_pixels": int(np.count_nonzero(seed.mask)),
                        "missing_pixels": int(np.count_nonzero(seed.mask & ~frame)),
                    })
                capture.update({
                    "native_union_path": str(destination.resolve()),
                    "native_union_file_sha256": _sha256(destination),
                    "native_union_logical_sha256": hashlib.sha256(memoryview(native)).hexdigest(),
                    "native_union_shape_tyx": list(native.shape),
                    "native_union_foreground_pixels": int(np.count_nonzero(native)),
                    "foreground_pixels_per_frame": [int(np.count_nonzero(frame)) for frame in native],
                    "hard_positive_retention": hard_positives,
                    "coordinate_space": "fixture-cache native TYX; seeded tile 0 embedded at its grid origin",
                })
                if any(item["missing_pixels"] for item in hard_positives):
                    raise RuntimeError(f"authoritative seed pixels were lost in {name}")
                return result

            with mock.patch.object(execution, "_drive_workers_to_fixed_point", save_native_union):
                result = helper.run_case(
                    case_root, pool=pool, cache=cache, view=view, seeds=seeds,
                    windowed=True, role=name, device=args.device,
                )
            if not capture or capture["native_union_logical_sha256"] != result["logical_union_sha256"]:
                raise RuntimeError("saved native union differs from the selected helper's result")
            if _sha256(inputs["cache_path"]) != inputs["cache_sha256"] or _sha256(inputs["seed_path"]) != inputs["seed_sha256"]:
                raise RuntimeError(f"immutable fixture input changed during replay: {name}")
            audit = result["worker_audit"]
            score_files = sorted((root / "scores").glob(name + "-*.json"))
            row = {
                **result, **capture,
                "counts": {
                    "actual_dispatched_tasks": len(result["dispatches"]),
                    "actual_tracker_sessions": audit.get("tracker_session_count"),
                    "logical_chains": audit.get("chain_count"),
                    "planned_windows": audit.get("planned_window_count"),
                    "skipped_windows": audit.get("skipped_window_count"),
                    "last_relay_generation": result["generation"],
                },
                "relay_admission": audit.get("relay_admission"),
                "prediction_evidence": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in score_files],
                "prediction_record_count": sum(len(json.loads(path.read_text())) for path in score_files),
            }
            report["fixture_results"][name] = row
            helper.write_json_atomically(case_root / "frontier_summary.json", row)
            helper.write_json_atomically(root / "summary.json", report)
            print(json.dumps({"fixture": name, **row["counts"], "hard_positive_missing_pixels": 0,
                              "native_union": capture["native_union_path"]}, sort_keys=True), flush=True)
        report["source_fingerprint_after"] = helper.lta_source_fingerprint()
        if report["source_fingerprint_after"] != fingerprint:
            raise RuntimeError("selected source tree changed during functional replay")
        report["status"] = "passed"
    except BaseException as error:
        primary_error = error
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        try:
            if pool is not None:
                report["forced_shutdown_devices"] = list(pool.shutdown(timeout=30))
                if report["forced_shutdown_devices"]:
                    raise RuntimeError("relay frontier replay required forced worker shutdown")
        except BaseException as error:
            report["status"] = "failed"
            report["shutdown_error"] = str(error)
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(str(error))
        finally:
            helper.write_json_atomically(root / "summary.json", report)


if __name__ == "__main__":
    main()
