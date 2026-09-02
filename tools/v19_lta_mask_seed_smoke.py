"""Experimentally seed SAM 3.1 tracking from an authoritative tile mask.

This uses a pinned private SAM2 tracker boundary after creating one interactive
object. It is diagnostic-only and must not be treated as a public SAM API or an
approved LTA publication path.
"""

from __future__ import annotations

import argparse
import gc
import json
import operator
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for entry in (ROOT, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from v19_lta_gpu_smoke import (
    DEFAULT_EXEMPLAR_INDEX,
    DEFAULT_LABEL_ROW,
    DEFAULT_PROMPT_FRAME,
    DEFAULT_START_FRAME,
    _MeasuredPredictor,
    configure_constrained_gpu_batches,
    cuda_snapshot,
    find_case_video,
    find_indexed_exemplar,
    install_sdpa_fallback,
    resolve_pinned_sam_runtime_provenance,
    validate_smoke_window,
)
from v19_lta_point_smoke import (
    PointClick,
    _distance_peaks,
    _pixel_center_normalized,
    mask_metrics,
    rasterize_polygons,
    write_strategy_artifacts,
)
from v19_lta_tile_smoke import (
    decode_rgb_tile_frames,
    plan_object_tile,
    probe_video_dimensions,
    select_polygon_row,
    transform_polygon_to_tile,
)
from XTA.lta_inputs import parse_yolo_segmentation_label
from XTA.lta_sam import (
    LTA_MAX_NUM_OBJECTS,
    LTA_MULTIPLEX_COUNT,
    LTA_SESSION_FRAMES,
    SamFramePrediction,
    SamSessionPlan,
    build_local_sam_predictor,
    normalize_video_frame_output,
    resolve_local_sam_bundle,
)


MASK_SEED_ANCHOR_IOU = 0.99


def resolve_mask_seed_capacity(object_count: int) -> int:
    """Size the diagnostic tracker explicitly without silently truncating objects."""

    if isinstance(object_count, bool):
        raise TypeError("mask-seed object count must be an integer")
    try:
        count = operator.index(object_count)
    except Exception as exc:
        raise TypeError("mask-seed object count must be an integer") from exc
    if not 1 <= count <= LTA_MAX_NUM_OBJECTS:
        raise ValueError(
            f"mask-seed object count must be in [1,{LTA_MAX_NUM_OBJECTS}]; got {count}"
        )
    quantum = 16
    return min(LTA_MAX_NUM_OBJECTS, max(quantum, ((count + quantum - 1) // quantum) * quantum))


def run_mask_seed_session(
    measured: Any,
    raw_predictor: Any,
    *,
    resource: list[Any],
    session: SamSessionPlan,
    prompt_frame: int,
    ground_truth: Any,
    seed: PointClick | None,
    object_masks: tuple[Any, ...] | None,
    conf: float,
    propagation_mode: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    local_prompt = int(prompt_frame) - int(session.frame_start)
    started = measured.handle_request(
        {
            "type": "start_session",
            "resource_path": resource,
            "offload_video_to_cpu": True,
        }
    )
    if not isinstance(started, Mapping) or not str(started.get("session_id", "")).strip():
        raise RuntimeError("mask-seed start_session returned no session_id")
    session_id = str(started["session_id"])
    stream = None
    predictions = []
    seen = set()
    tracker_state = None
    inference_state = None
    feature = None
    video_masks = None
    active_error: BaseException | None = None
    try:
        registry = getattr(raw_predictor, "_all_inference_states", None)
        if not isinstance(registry, Mapping) or session_id not in registry:
            raise RuntimeError("pinned predictor session registry is unavailable")
        inference_state = registry[session_id]["state"]
        model = raw_predictor.model
        tracker = model.tracker
        add_masks = getattr(tracker, "add_new_masks", None)
        if not callable(add_masks):
            raise RuntimeError("pinned multiplex tracker exposes no add_new_masks boundary")
        if object_masks is None:
            if seed is None:
                raise ValueError("selected-mask seeding requires an object-creation point")
            created = measured.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": local_prompt,
                    "obj_id": 0,
                    "points": [[seed.x, seed.y]],
                    "point_labels": [1],
                    "clear_old_points": True,
                    "rel_coordinates": True,
                }
            )
            if not isinstance(created, Mapping) or not isinstance(created.get("outputs"), Mapping):
                raise RuntimeError("could not create the tracker object before mask seeding")
            get_states = getattr(model, "_get_sam2_inference_states_by_obj_ids", None)
            if not callable(get_states):
                raise RuntimeError("pinned SAM 3.1 model exposes no SAM2 state lookup")
            states = get_states(inference_state, [0])
            if len(states) != 1:
                raise RuntimeError(f"expected one SAM2 state for object 0; got {len(states)}")
            tracker_state = states[0]
            expected_ids = (0,)
            mask_tensor = torch.from_numpy(np.asarray(ground_truth, dtype=np.float32))[None]
            reconditioning = True
        else:
            if propagation_mode != "tracker-only":
                raise ValueError("multi-mask seeding is supported only by tracker-only propagation")
            if not object_masks:
                raise ValueError("multi-mask seeding requires at least one nonempty object mask")
            initialize = getattr(model, "_init_new_sam2_state", None)
            if not callable(initialize):
                raise RuntimeError("pinned SAM model exposes no tracker-state initializer")
            tracker_state = initialize(inference_state)
            prepare_anchor = getattr(model, "_prepare_backbone_feats", None)
            if not callable(prepare_anchor):
                raise RuntimeError("pinned SAM model exposes no shared anchor feature bridge")
            with torch.inference_mode():
                prepare_anchor(inference_state, local_prompt, reverse=False)
            expected_ids = tuple(range(len(object_masks)))
            mask_tensor = torch.from_numpy(
                np.stack([np.asarray(mask, dtype=np.float32) for mask in object_masks])
            )
            reconditioning = False
        _frame, object_ids, _low_res, video_masks = add_masks(
            tracker_state,
            frame_idx=local_prompt,
            obj_ids=list(expected_ids),
            masks=mask_tensor,
            reconditioning=reconditioning,
        )
        if int(_frame) != local_prompt or tuple(int(value) for value in object_ids) != expected_ids:
            raise RuntimeError("private mask seed returned an unexpected object/frame")
        if video_masks is None or len(video_masks) < len(expected_ids):
            raise RuntimeError("private mask seed returned no anchor mask")
        seeded_masks = (video_masks[: len(expected_ids)] > 0).to(torch.bool)
        expected_seed_masks = (
            (np.asarray(ground_truth, dtype=bool),)
            if object_masks is None
            else tuple(np.asarray(mask, dtype=bool) for mask in object_masks)
        )
        seeded_arrays = tuple(
            seeded_masks[index].squeeze().detach().cpu().numpy().astype(bool)
            for index in range(len(expected_ids))
        )
        seed_object_metrics = tuple(
            mask_metrics(expected_mask, seeded_array)
            for expected_mask, seeded_array in zip(expected_seed_masks, seeded_arrays)
        )
        changed = [
            index
            for index, metrics in enumerate(seed_object_metrics)
            if float(metrics["iou"]) < 0.999999
        ]
        if changed:
            details = {index: seed_object_metrics[index] for index in changed}
            raise RuntimeError(f"private mask seed changed object mask(s): {details}")
        seeded_mask = seeded_masks.any(dim=0)
        seeded_np = np.logical_or.reduce(seeded_arrays)
        seed_metrics = mask_metrics(ground_truth, seeded_np)
        if float(seed_metrics["iou"]) < 0.999999:
            raise RuntimeError(f"private mask seed changed the anchor mask: {seed_metrics}")
        if object_masks is None:
            cache = getattr(model, "_cache_frame_outputs", None)
            if not callable(cache):
                raise RuntimeError("pinned SAM model exposes no frame-cache boundary")
            cache(inference_state, local_prompt, {0: seeded_mask})

        if propagation_mode == "merged":
            stream = measured.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both",
                    "start_frame_index": local_prompt,
                    "max_frame_num_to_track": int(session.frame_count),
                    "output_prob_thresh": float(conf),
                }
            )
            for response in stream:
                if not isinstance(response, Mapping):
                    raise RuntimeError("mask-seed propagation yielded a non-mapping response")
                local_frame = int(response.get("frame_index", -1))
                if not 0 <= local_frame < session.frame_count or local_frame in seen:
                    raise RuntimeError(f"mask-seed propagation yielded invalid frame {local_frame}")
                seen.add(local_frame)
                outputs = response.get("outputs")
                if not isinstance(outputs, Mapping):
                    raise RuntimeError("mask-seed propagation response has no outputs")
                normalized = normalize_video_frame_output(
                    outputs,
                    sequence_id="m1__private_mask_seed",
                    session_index=0,
                    global_frame_index=int(session.frame_start) + local_frame,
                    # A mask-seeded high-level run is propagation_partial: it
                    # performs no detector admission and upstream therefore
                    # emits no mandatory Multiplex drop statistics.  The
                    # normalizer still validates and rejects any nonzero stats
                    # that are present.
                    require_drop_stats=False,
                )
                predictions.extend(item for item in normalized if item.object_id == 0)
        elif propagation_mode == "tracker-only":
            preflight = getattr(tracker, "propagate_in_video_preflight", None)
            propagate = getattr(tracker, "propagate_in_video", None)
            if not callable(preflight) or not callable(propagate):
                raise RuntimeError("pinned tracker exposes no direct propagation boundary")
            preflight(tracker_state, run_mem_encoder=True)
            prepare_features = getattr(model, "_prepare_backbone_feats", None)
            if not callable(prepare_features):
                raise RuntimeError("pinned SAM model exposes no shared feature bridge")
            orders = (
                (False, range(local_prompt, int(session.frame_count))),
                (True, range(local_prompt - 1, -1, -1)),
            )
            for reverse, order in orders:
                for requested_frame in order:
                    with torch.inference_mode():
                        prepare_features(inference_state, requested_frame, reverse=reverse)
                        feature = inference_state["feature_cache"].get(requested_frame)
                        if feature is None:
                            raise RuntimeError(
                                f"shared SAM feature was not cached for frame {requested_frame}"
                            )
                        tracker_state["cached_features"] = {requested_frame: feature}
                        one_frame = propagate(
                            tracker_state,
                            start_frame_idx=requested_frame,
                            max_frame_num_to_track=0,
                            reverse=reverse,
                            tqdm_disable=True,
                            run_mem_encoder=True,
                        )
                        outputs = tuple(one_frame)
                    if len(outputs) != 1:
                        raise RuntimeError(
                            f"tracker-only one-frame bridge returned {len(outputs)} frames"
                        )
                    local_frame, object_ids, _low_res, video_masks, _scores = outputs[0]
                    local_frame = int(local_frame)
                    if local_frame != requested_frame or local_frame in seen:
                        raise RuntimeError(
                            f"tracker-only propagation yielded invalid frame {local_frame}"
                        )
                    seen.add(local_frame)
                    ids = tuple(int(value) for value in object_ids)
                    if ids != expected_ids or video_masks is None or len(video_masks) < len(expected_ids):
                        raise RuntimeError(
                            f"tracker-only propagation returned ids={ids}, masks={getattr(video_masks, 'shape', None)}"
                        )
                    frame_masks = tuple(
                        (video_masks[index].squeeze() > 0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool)
                        for index in range(len(expected_ids))
                    )
                    for object_id, binary in zip(expected_ids, frame_masks):
                        predictions.append(
                            SamFramePrediction(
                                sequence_id="m1__private_tracker_only_mask_seed",
                                session_index=0,
                                frame_index=int(session.frame_start) + local_frame,
                                object_id=object_id,
                                initial_detection_score=1.0,
                                binary_mask=binary,
                            )
                        )
        else:
            raise ValueError(f"unknown propagation_mode: {propagation_mode}")
        expected = set(range(session.frame_count))
        if seen != expected:
            raise RuntimeError(f"mask-seed propagation missing frames {sorted(expected-seen)}")
    except BaseException as exc:
        active_error = exc
        predictions.clear()
        raise
    finally:
        cleanup_errors: list[tuple[str, Exception]] = []
        close_stream = getattr(stream, "close", None)
        if callable(close_stream):
            try:
                close_stream()
            except Exception as exc:
                cleanup_errors.append(("propagation iterator", exc))
        try:
            measured.handle_request({"type": "close_session", "session_id": session_id})
        except Exception as exc:
            cleanup_errors.append(("close_session", exc))
        try:
            if isinstance(tracker_state, dict):
                tracker_state.clear()
        except Exception as exc:
            cleanup_errors.append(("private tracker-state clear", exc))
        feature = None
        video_masks = None
        mask_tensor = None
        seeded_masks = None
        seeded_mask = None
        one_frame = None
        outputs = None
        _low_res = None
        _scores = None
        tracker_state = None
        inference_state = None
        if active_error is not None:
            add_note = getattr(active_error, "add_note", None)
            if callable(add_note):
                for boundary, cleanup_error in cleanup_errors:
                    add_note(
                        f"SAM {boundary} also failed while handling the primary error: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        elif cleanup_errors:
            boundary, cleanup_error = cleanup_errors[0]
            add_note = getattr(cleanup_error, "add_note", None)
            if callable(add_note):
                for secondary_boundary, secondary_error in cleanup_errors[1:]:
                    add_note(
                        f"SAM {secondary_boundary} also failed during cleanup: "
                        f"{type(secondary_error).__name__}: {secondary_error}"
                    )
            raise cleanup_error

    anchor = [item for item in predictions if item.frame_index == int(prompt_frame)]
    anchor_iou = None
    anchor_object_metrics = None
    anchor_ids = tuple(sorted(item.object_id for item in anchor))
    if anchor_ids == expected_ids:
        by_id = {item.object_id: np.asarray(item.binary_mask, dtype=bool) for item in anchor}
        anchor_masks = tuple(by_id[object_id] for object_id in expected_ids)
        anchor_union = np.logical_or.reduce(anchor_masks)
        anchor_iou = mask_metrics(ground_truth, anchor_union)["iou"]
        anchor_object_metrics = tuple(
            mask_metrics(expected_mask, actual_mask)
            for expected_mask, actual_mask in zip(expected_seed_masks, anchor_masks)
        )
    passed = (
        anchor_iou is not None
        and float(anchor_iou) >= MASK_SEED_ANCHOR_IOU
        and anchor_object_metrics is not None
        and all(
            float(metrics["iou"]) >= MASK_SEED_ANCHOR_IOU
            for metrics in anchor_object_metrics
        )
    )
    active_frames = sorted(
        {
            item.frame_index
            for item in predictions
            if bool(np.asarray(item.binary_mask, dtype=bool).any())
        }
    )
    non_anchor_active_frames = [
        frame_index for frame_index in active_frames if frame_index != int(prompt_frame)
    ]
    anchor_integrity_passed = bool(passed)
    diagnostic_propagation_gate_passed = bool(
        anchor_integrity_passed and non_anchor_active_frames
    )
    return {
        "strategy": "private_mask_seed" if object_masks is None else "private_multi_mask_seed",
        "initial_clicks": () if seed is None else (seed,),
        "final_clicks": () if seed is None else (seed,),
        "iterations": (
            {
                "round": 0,
                "new_clicks": [] if seed is None else [asdict(seed)],
                "click_count": 0 if seed is None else 1,
                "metrics": seed_metrics,
                "preview_active": True,
                "mask": seeded_np,
            },
        ),
        "stop_reason": "authoritative_mask_seed",
        "final_metrics": seed_metrics,
        "best_metrics": seed_metrics,
        "best_round": 0,
        "best_click_count": 0 if seed is None else 1,
        # ``success`` remains only as the compatibility input consumed by the
        # shared diagnostic artifact writer.  It means the explicitly named
        # activity gate below, never label quality or publication acceptance.
        "success": diagnostic_propagation_gate_passed,
        "success_definition": (
            "diagnostic propagation gate only: anchor integrity plus at least one "
            "non-anchor active frame; not a quality or publication claim"
        ),
        "anchor_integrity_passed": anchor_integrity_passed,
        "anchor_integrity_minimum_iou": MASK_SEED_ANCHOR_IOU,
        "diagnostic_propagation_gate_passed": diagnostic_propagation_gate_passed,
        "non_anchor_active_frames": non_anchor_active_frames,
        "drop_stats_applicable": False,
        "propagated_revision": {
            "selection": "authoritative_mask_seed",
            "round": 0,
            "click_count": 0 if seed is None else 1,
            "equals_best": True,
        },
        "propagation": tuple(
            sorted(predictions, key=lambda item: (item.frame_index, item.object_id))
        ),
        "propagation_response_count": len(seen),
        "propagation_active_frames": active_frames,
        "anchor_preview_propagation_iou": anchor_iou,
        "anchor_propagation_object_metrics": anchor_object_metrics,
        "anchor_expected_object_ids": expected_ids,
        "anchor_returned_object_ids": anchor_ids,
        "propagation_mode": str(propagation_mode),
        "seeded_object_count": len(expected_ids),
        "seed_object_metrics": seed_object_metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--exemplar-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=DEFAULT_START_FRAME)
    parser.add_argument("--prompt-frame", type=int, default=DEFAULT_PROMPT_FRAME)
    parser.add_argument("--exemplar-index", type=int, default=DEFAULT_EXEMPLAR_INDEX)
    parser.add_argument("--label-row", type=int, default=DEFAULT_LABEL_ROW)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument(
        "--seed-scope",
        choices=("selected", "all-in-tile"),
        default="selected",
    )
    parser.add_argument(
        "--propagation-mode",
        choices=("merged", "tracker-only"),
        default="tracker-only",
    )
    parser.add_argument(
        "--weight-storage",
        choices=("bfloat16_egpu", "float32"),
        default="bfloat16_egpu",
    )
    return parser


def main() -> None:
    import numpy as np
    import torch

    args = _build_parser().parse_args()
    validate_smoke_window(args.start_frame, args.prompt_frame)
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"--output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = resolve_local_sam_bundle(args.model)
    video = find_case_video(Path(args.input_root).resolve(strict=True), "m1")
    _image, label = find_indexed_exemplar(Path(args.exemplar_root).resolve(strict=True), args.exemplar_index)
    label_digest, polygons = parse_yolo_segmentation_label(label)
    selected_polygon = select_polygon_row(polygons, int(args.label_row))
    width, height = probe_video_dimensions(video)
    tile = plan_object_tile(
        selected_polygon,
        source_width=width,
        source_height=height,
    )
    local_polygons = tuple(transform_polygon_to_tile(item, tile) for item in polygons)
    local_selected_polygon = transform_polygon_to_tile(selected_polygon, tile)
    if args.seed_scope == "selected":
        ground_truth = rasterize_polygons((local_selected_polygon,))
        object_masks = None
        peak = _distance_peaks(ground_truth, 1)[0]
        seed = PointClick(*_pixel_center_normalized(peak[0], peak[1], tile.size), True, "object_creation_seed")
    else:
        masks = tuple(rasterize_polygons((polygon,)) for polygon in local_polygons)
        object_masks = tuple(mask for mask in masks if bool(mask.any()))
        ground_truth = rasterize_polygons(
            tuple(polygon for polygon, mask in zip(local_polygons, masks) if bool(mask.any()))
        )
        seed = None
        if args.propagation_mode != "tracker-only":
            raise ValueError("--seed-scope all-in-tile requires --propagation-mode tracker-only")
    seeded_object_count = 1 if object_masks is None else len(object_masks)
    model_object_capacity = resolve_mask_seed_capacity(seeded_object_count)
    runtime = resolve_pinned_sam_runtime_provenance()
    session = SamSessionPlan(
        sequence_id="m1__private_mask_seed",
        session_index=0,
        frame_start=args.start_frame,
        frame_stop=args.start_frame + LTA_SESSION_FRAMES,
    )

    torch.cuda.set_device(args.device)
    predictor = None
    restore_sdpa = None
    measured = None
    resource = None
    active_error: BaseException | None = None
    memory = [cuda_snapshot(torch, args.device, "before_builder")]
    try:
        predictor = build_local_sam_predictor(
            bundle,
            device_id=args.device,
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            conf=args.conf,
            weight_storage=args.weight_storage,
            max_num_objects=model_object_capacity,
            construction_device="meta",
        )
        constrained = configure_constrained_gpu_batches(predictor)
        restore_sdpa = install_sdpa_fallback()
        measured = _MeasuredPredictor(predictor, torch, args.device, memory)
        resource = decode_rgb_tile_frames(video, args.start_frame, tile)
        result = run_mask_seed_session(
            measured,
            predictor,
            resource=resource,
            session=session,
            prompt_frame=args.prompt_frame,
            ground_truth=ground_truth,
            seed=seed,
            object_masks=object_masks,
            conf=args.conf,
            propagation_mode=args.propagation_mode,
        )
        result["peak_allocated_mib"] = int(torch.cuda.max_memory_allocated(args.device) // (1024 * 1024))
        artifacts = write_strategy_artifacts(
            output,
            result=result,
            resource=resource,
            ground_truth=ground_truth,
            session=session,
            prompt_frame=args.prompt_frame,
        )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_errors: list[tuple[str, Exception]] = []
        if restore_sdpa is not None:
            try:
                restore_sdpa()
            except Exception as exc:
                cleanup_errors.append(("SDPA restoration", exc))
        if predictor is not None:
            shutdown = getattr(predictor, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    cleanup_errors.append(("predictor shutdown", exc))
        measured = None
        resource = None
        predictor = None
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            cleanup_errors.append(("CUDA garbage collection", exc))
        if active_error is not None:
            add_note = getattr(active_error, "add_note", None)
            if callable(add_note):
                for boundary, cleanup_error in cleanup_errors:
                    add_note(
                        f"SAM {boundary} also failed while handling the primary error: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        elif cleanup_errors:
            boundary, cleanup_error = cleanup_errors[0]
            add_note = getattr(cleanup_error, "add_note", None)
            if callable(add_note):
                for secondary_boundary, secondary_error in cleanup_errors[1:]:
                    add_note(
                        f"SAM {secondary_boundary} also failed during cleanup: "
                        f"{type(secondary_error).__name__}: {secondary_error}"
                    )
            raise cleanup_error

    summary = {
        "status": "execution_complete",
        "execution_completed": True,
        "experiment": "v19_lta_private_mask_seed",
        "private_unstable_api": True,
        "propagation_mode": str(args.propagation_mode),
        "seed_scope": str(args.seed_scope),
        "seeded_object_count": result["seeded_object_count"],
        "model_object_capacity": model_object_capacity,
        "runtime": runtime,
        "model": {
            "bundle_root": str(bundle.root),
            "checkpoint_path": str(bundle.checkpoint_path),
            "model_version": bundle.model_version,
            "checkpoint_identity_sha256": bundle.checkpoint_identity_sha256,
        },
        "device": {
            "index": int(args.device),
            "name": torch.cuda.get_device_name(int(args.device)),
            "capability": list(torch.cuda.get_device_capability(int(args.device))),
        },
        "video": {
            "path": str(video),
            "source_width": int(width),
            "source_height": int(height),
        },
        "label": {
            "path": str(label),
            "sha256": str(label_digest),
            "encoded_index": int(args.exemplar_index),
            "zero_based_row": int(args.label_row),
        },
        "tile_xyxy": list(tile.xyxy),
        "session": {
            "sequence_id": session.sequence_id,
            "session_index": int(session.session_index),
            "frame_start": int(session.frame_start),
            "frame_stop_exclusive": int(session.frame_stop),
            "frame_count": int(session.frame_count),
            "prompt_frame_global": int(args.prompt_frame),
            "prompt_frame_local": int(args.prompt_frame) - int(session.frame_start),
        },
        "settings": {
            "conf": float(args.conf),
            "weight_storage": str(args.weight_storage),
            "construction_device": "meta",
            "use_fa3": False,
            "use_rope_real": False,
            "compile": False,
            "warm_up": False,
            "async_loading_frames": False,
            "offload_video_to_cpu_requested": True,
            "multiplex_count": int(LTA_MULTIPLEX_COUNT),
            "max_num_objects": int(model_object_capacity),
            "sdpa_backends": ["flash_attention", "efficient_attention", "math"],
        },
        "anchor_integrity_gate": {
            "minimum_per_object_iou": float(MASK_SEED_ANCHOR_IOU),
            "requires_exact_object_ids": True,
            "passed": bool(result["anchor_integrity_passed"]),
        },
        "diagnostic_propagation_gate": {
            "definition": (
                "anchor integrity passed and at least one non-anchor frame returned "
                "a nonempty mask"
            ),
            "minimum_non_anchor_active_frames": 1,
            "non_anchor_active_frames": result["non_anchor_active_frames"],
            "passed": bool(result["diagnostic_propagation_gate_passed"]),
            "quality_or_publication_claim": False,
        },
        "drop_stats_applicable": bool(result["drop_stats_applicable"]),
        "anchor_seed_metrics": result["final_metrics"],
        "anchor_seed_object_metrics": result["seed_object_metrics"],
        "propagation_response_count": result["propagation_response_count"],
        "propagation_active_frames": result["propagation_active_frames"],
        "anchor_preview_propagation_iou": result["anchor_preview_propagation_iou"],
        "anchor_propagation_object_metrics": result[
            "anchor_propagation_object_metrics"
        ],
        "anchor_expected_object_ids": result["anchor_expected_object_ids"],
        "anchor_returned_object_ids": result["anchor_returned_object_ids"],
        "peak_allocated_mib": result["peak_allocated_mib"],
        "constrained_batches": constrained,
        "artifacts": artifacts,
        "memory_events": memory,
    }
    path = output / "summary.json"
    summary["summary_path"] = str(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
