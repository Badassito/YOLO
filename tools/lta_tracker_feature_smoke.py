"""Compare the pinned grounding and tracker-only visual caches on real frames."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.lta_outputs import write_json_atomically
from XTA.lta_worker_adapter import build_worker_predictor, close_worker_predictor
from XTA.lta_workers import LtaWorkerInit, LtaWorkerPool, LtaWorkerTask
from tools.lta_production_smoke import prepare_fixture, _source_snapshot


def compare_propagation(context, payload, output):
    """Record compact per-object masks/scores while comparing both bridges."""
    import numpy as np
    from XTA import lta_experimental as experimental, lta_propagation as propagation
    from XTA import lta_tracker_features as feature_module
    from XTA.lta_worker_adapter import execute_worker_task
    from XTA.lta_union_artifacts import logical_union_sha256
    spec = importlib.util.spec_from_file_location("XTA._lta_score_reference", payload["reference_module"])
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    actual_scores = experimental._sigmoid_tracker_score_logits
    score_checks = []
    def checked_scores(scores, **kwargs):
        actual = actual_scores(scores, **kwargs)
        expected = reference._sigmoid_tracker_score_logits(scores, **kwargs)
        score_checks.append(actual == expected)
        if actual != expected:
            raise RuntimeError("packed score normalization differs from original on real tracker logits")
        return actual
    actual_session = propagation.run_mask_injected_session
    traces = {}
    manifests = {}
    for role in ("original", "feature_only"):
        trace = []
        def traced_session(*args, **kwargs):
            callback = kwargs["prediction_callback"]
            request = kwargs["request"]
            def observe(item):
                prediction = item.prediction
                mask = np.ascontiguousarray(prediction.binary_mask, dtype=np.uint8)
                trace.append({"session": request.session.session_index,
                              "frame": int(prediction.frame_index),
                              "object_id": int(prediction.object_id), "lineage": item.lineage.token,
                              "mask_sha256": hashlib.sha256(memoryview(mask)).hexdigest(),
                              "initial_detection_score": prediction.initial_detection_score,
                              "frame_tracker_score": prediction.frame_tracker_score})
                callback(item)
            kwargs["prediction_callback"] = observe
            return actual_session(*args, **kwargs)
        def original_features(model, state, frame_idx, reverse):
            model._prepare_backbone_feats(state, frame_idx, reverse=reverse)
            return {"policy": "original_grounding_reference", "fallback_reason": None}
        actual_features = feature_module.prepare_tracker_frame_features
        with (mock.patch.object(propagation, "run_mask_injected_session", traced_session),
              mock.patch.object(experimental, "_sigmoid_tracker_score_logits", checked_scores),
              mock.patch.object(feature_module, "prepare_tracker_frame_features",
                                original_features if role == "original" else actual_features)):
            task_payload = {**payload["propagation_payload"], "work_id": "trace-" + role,
                            "output_dir": str(output / ("propagation-" + role))}
            result = execute_worker_task(context, "propagation_chain", task_payload)
        manifests[role] = json.loads(Path(result["artifact_path"]).read_text())
        traces[role] = sorted(trace, key=lambda row: (row["session"], row["frame"], row["object_id"], row["lineage"]))
    equal = traces["original"] == traces["feature_only"]
    audit = {"status": "passed" if equal and all(score_checks) else "failed",
             "original_prediction_count": len(traces["original"]),
             "feature_only_prediction_count": len(traces["feature_only"]),
             "exact_per_object_mask_and_probability_trace": equal,
             "real_logit_normalizer_comparisons": len(score_checks),
             "real_logit_normalizer_exact": all(score_checks),
             "union_sha256": {role: logical_union_sha256(manifest["union"], frame_start=manifest["output_frame_range"][0])
                              for role, manifest in manifests.items()},
             "feature_only_session_audits": [row["adapter"]["tracker_feature_preparation"]
                                               for row in manifests["feature_only"]["windows"]]}
    for row in audit["feature_only_session_audits"]:
        if row.get("feature_only_preparations", 0) <= 0 or row.get("fallback_preparations", 0) != 0:
            raise RuntimeError("propagation comparison fell back from feature-only preparation")
    write_json_atomically(output / "prediction_traces.json", traces)
    return audit


def compare_worker_features(context, kind, payload):
    if kind != "feature_equivalence":
        raise ValueError("unsupported feature qualification task")
    from XTA.lta_rendering import LtaPhysicalViewCacheRef, render_native_tile_window
    from XTA.lta_tracker_features import prepare_tracker_frame_features
    torch = context.torch_module
    cache = LtaPhysicalViewCacheRef.from_payload(payload["cache_ref"])
    resource = render_native_tile_window(cache, frame_start=payload["frame_start"],
                                        frame_stop=payload["frame_start"] + 3,
                                        tile_xyxy=payload["tile_xyxy"])
    predictor = context.predictor
    session_id = predictor.handle_request({"type": "start_session", "resource_path": resource})["session_id"]
    result = {"status": "running", "autocast_cuda": torch.is_autocast_enabled("cuda"),
              "comparisons": [], "profile": dict(context.profile),
              "sam_runtime": dict(context.sam_runtime), "source_sha256": _source_snapshot()}
    output = Path(payload["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        if not result["autocast_cuda"]:
            raise RuntimeError("pinned feature comparison requires inherited CUDA autocast")
        base = predictor._all_inference_states[session_id]["state"]
        old_state = {**base, "feature_cache": {}}
        new_state = {**base, "feature_cache": {}}
        model = predictor.model
        def tensors(state, frame):
            image, features = state["feature_cache"][frame]
            values = {"cached_image": image}
            for branch in ("interactive", "sam2_backbone_out"):
                assert features[branch]["vision_mask"] is None
                values[branch + ".vision_features"] = features[branch]["vision_features"]
                for index, feature in enumerate(features[branch]["backbone_fpn"]):
                    assert feature.mask is None
                    values[f"{branch}.fpn.{index}"] = feature.tensors
                for index, position in enumerate(features[branch]["vision_pos_enc"]):
                    values[f"{branch}.position.{index}"] = position
            return values
        with torch.inference_mode():
            for ordinal, (frame, reverse) in enumerate(((1, False), (1, False), (2, False), (0, True))):
                model._prepare_backbone_feats(old_state, frame, reverse=reverse)
                originals = tensors(old_state, frame)
                expected = {name: tensor.detach().cpu().clone() for name, tensor in originals.items()}
                properties = {name: (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
                              for name, tensor in originals.items()}
                receipt = prepare_tracker_frame_features(model, new_state, frame, reverse=reverse)
                if receipt["fallback_reason"] is not None:
                    raise RuntimeError("feature comparison did not use the recognized optimized path")
                actual = tensors(new_state, frame)
                if set(actual) != set(expected):
                    raise RuntimeError("feature cache keys differ")
                for name, tensor in actual.items():
                    observed = tensor.detach().cpu()
                    shape, dtype, device = properties[name]
                    equal = (tuple(tensor.shape) == shape and str(tensor.dtype) == dtype
                             and str(tensor.device) == device and torch.equal(expected[name], observed))
                    row = {"ordinal": ordinal, "frame": frame, "reverse": reverse, "tensor": name,
                           "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                           "device": str(tensor.device), "exact_equal": equal}
                    if not equal and tuple(tensor.shape) == shape:
                        row["maximum_absolute_difference"] = float((expected[name].double() - observed.double()).abs().max())
                    result["comparisons"].append(row)
                del expected, originals, actual
        result["helper_audit"] = new_state["lta_tracker_feature_preparation"]
        result["status"] = "passed" if all(row["exact_equal"] for row in result["comparisons"]) else "failed"
        result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        artifact = write_json_atomically(output / "tensor_equivalence.json", result)
    if payload.get("compare_propagation"):
        old_state["feature_cache"].clear()
        new_state["feature_cache"].clear()
        try:
            result["propagation_comparison"] = compare_propagation(context, payload, output)
        except BaseException as exc:
            result["status"] = "failed"
            result["propagation_error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json_atomically(output / "tensor_equivalence.json", result)
            raise
        if result["propagation_comparison"]["status"] != "passed":
            result["status"] = "failed"
        artifact = write_json_atomically(output / "tensor_equivalence.json", result)
    return {"artifact_path": str(artifact), "metrics": {"status": result["status"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-shape", type=int, nargs=3, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("auto", "egpu", "h100"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compare-propagation", action="store_true")
    parser.add_argument("--reference-module", type=Path,
                        help="Saved original XTA/lta_experimental.py for actual-logit comparison")
    args = parser.parse_args()
    if args.compare_propagation and args.reference_module is None:
        parser.error("--compare-propagation requires --reference-module")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be new or empty")
    plan, seeds = prepare_fixture(cache_path=args.cache, cache_shape=tuple(args.cache_shape),
                                  seed_path=args.seed, output_dir=output)
    prompt = seeds[0].frame_index
    if not 0 < prompt < args.cache_shape[0] - 1:
        parser.error("seed must have a frame on each side for directional comparison")
    tile = plan["payload"]["tile"]
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomically(output / "run_plan.json", plan)
    pool = LtaWorkerPool((args.device,), LtaWorkerInit(
        adapter_module="tools.lta_tracker_feature_smoke", adapter_factory="build_worker_predictor",
        adapter_execute="compare_worker_features", adapter_shutdown="close_worker_predictor",
        adapter_config={"model_path": args.model, "profile": args.profile, "conf": 0.15}), startup_timeout=600)
    summary = {"status": "running", "ready_events": [asdict(event) for event in pool.ready_events]}
    primary_error = None
    try:
        task = LtaWorkerTask("feature-equivalence", "feature-equivalence", "feature_equivalence", {
            "cache_ref": plan["payload"]["cache_ref"], "frame_start": prompt - 1,
            "compare_propagation": args.compare_propagation,
            "reference_module": str(args.reference_module) if args.reference_module else None,
            "propagation_payload": plan["payload"],
            "tile_xyxy": [tile["left"], tile["top"], tile["left"] + tile["size"], tile["top"] + tile["size"]],
            "output_dir": str(output / "worker-task")})
        pool.submit(task, execution_device_id=args.device)
        result = pool.wait_result(timeout=600)
        summary["result"] = asdict(result)
        summary["status"] = result.metrics["status"]
        if summary["status"] != "passed":
            raise RuntimeError("tracker feature tensors are not exactly equal")
    except BaseException as exc:
        primary_error = exc
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            forced = list(pool.shutdown(timeout=30))
            summary["forced_shutdown_devices"] = forced
            if forced:
                raise RuntimeError(f"feature qualification required forced worker shutdown: {forced}")
        except BaseException as shutdown_error:
            summary["status"] = "failed"
            summary["shutdown_error"] = str(shutdown_error)
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(f"Worker shutdown also failed: {shutdown_error}")
        finally:
            write_json_atomically(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
