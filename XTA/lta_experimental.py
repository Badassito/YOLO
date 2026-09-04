"""Pinned private SAM tracker adapter used by the experimental LTA path.

This module is the single quarantine boundary for SAM 3.1 internals used to
seed tracker state from authoritative masks.  Imports remain accelerator-light;
NumPy and Torch are loaded only while a session is executing.
"""

from __future__ import annotations

import math
import operator
from dataclasses import asdict
from typing import Any, Mapping

from .lta_sam import (
    LTA_MAX_NUM_OBJECTS,
    SamFramePrediction,
    SamSessionPlan,
    normalize_video_frame_output,
)


MASK_SEED_ANCHOR_IOU = 0.99


def mask_metrics(expected: Any, actual: Any) -> dict[str, Any]:
    """Return the established binary overlap and component diagnostics."""

    import cv2
    import numpy as np

    left = np.asarray(expected, dtype=bool)
    right = np.asarray(actual, dtype=bool)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(
            f"mask metrics require equal two-dimensional shapes; got {left.shape}/{right.shape}"
        )
    true_positive = int(np.count_nonzero(left & right))
    false_positive = int(np.count_nonzero(~left & right))
    false_negative = int(np.count_nonzero(left & ~right))
    expected_pixels = int(np.count_nonzero(left))
    actual_pixels = int(np.count_nonzero(right))
    union = true_positive + false_positive + false_negative
    denominator = 2 * true_positive + false_positive + false_negative
    count, labels = cv2.connectedComponents(right.astype(np.uint8), connectivity=8)
    component_areas = [
        int(np.count_nonzero(labels == index)) for index in range(1, int(count))
    ]
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "iou": 1.0 if union == 0 else float(true_positive / union),
        "dice": 1.0 if denominator == 0 else float(2 * true_positive / denominator),
        "precision": 1.0 if actual_pixels == 0 else float(true_positive / actual_pixels),
        "recall": 1.0 if expected_pixels == 0 else float(true_positive / expected_pixels),
        "ground_truth_pixels": expected_pixels,
        "prediction_pixels": actual_pixels,
        "area_ratio": (
            None if expected_pixels == 0 else float(actual_pixels / expected_pixels)
        ),
        "component_count": len(component_areas),
        "largest_component_fraction": (
            float(max(component_areas) / actual_pixels)
            if component_areas and actual_pixels
            else 0.0
        ),
    }


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


def _resolve_empty_frame_limit(value: int | None) -> int | None:
    """Validate the optional tracker-only empty-mask termination threshold."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("empty-frame limit must be a positive integer or None")
    try:
        limit = operator.index(value)
    except Exception as exc:
        raise TypeError("empty-frame limit must be a positive integer or None") from exc
    if limit < 1:
        raise ValueError(f"empty-frame limit must be >= 1; got {limit}")
    return int(limit)


def _global_half_open_ranges(
    indices: set[int],
    *,
    frame_start: int,
) -> tuple[tuple[int, int], ...]:
    """Compress local frame indexes into sorted global half-open ranges."""

    ordered = sorted(int(value) for value in indices)
    if not ordered:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append((int(frame_start) + start, int(frame_start) + previous + 1))
            start = value
        previous = value
    ranges.append((int(frame_start) + start, int(frame_start) + previous + 1))
    return tuple(ranges)


def _sigmoid_tracker_score_logits(
    scores: Any,
    *,
    expected_count: int,
    torch_module: Any,
) -> tuple[float, ...]:
    """Normalize SAM's fifth tracker return, ``object_score_logits``, safely."""

    if not isinstance(scores, torch_module.Tensor):
        raise RuntimeError(
            "tracker-only propagation returned non-tensor object_score_logits: "
            f"{type(scores).__name__}"
        )
    shape = tuple(int(value) for value in scores.shape)
    allowed = {(int(expected_count),), (int(expected_count), 1)}
    if shape not in allowed:
        raise RuntimeError(
            "tracker-only object_score_logits shape must contain one scalar per object; "
            f"got {shape}, expected one of {sorted(allowed)}"
        )
    flattened = scores.reshape(int(expected_count))
    if not torch_module.is_floating_point(flattened):
        raise RuntimeError(
            "tracker-only object_score_logits must have a real floating dtype; "
            f"got {flattened.dtype}"
        )
    if not bool(torch_module.isfinite(flattened).all().item()):
        raise RuntimeError("tracker-only object_score_logits contains a non-finite value")
    # The pinned SAM tracker names and stores this tensor as object_score_logits
    # (shape N x 1, with 10.0 documented as sigmoid(10) ~= 1).  Persist a
    # probability because SamFramePrediction deliberately accepts only [0, 1].
    probabilities = torch_module.sigmoid(flattened.to(dtype=torch_module.float64))
    normalized = tuple(float(value) for value in probabilities.detach().cpu().tolist())
    if len(normalized) != int(expected_count) or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in normalized
    ):
        raise RuntimeError("could not normalize tracker-only object_score_logits")
    return normalized


def run_mask_seed_session(
    measured: Any,
    raw_predictor: Any,
    *,
    resource: list[Any],
    session: SamSessionPlan,
    prompt_frame: int,
    ground_truth: Any,
    seed: Any | None,
    object_masks: tuple[Any, ...] | None,
    conf: float,
    propagation_mode: str,
    empty_frame_limit: int | None = None,
) -> dict[str, Any]:
    resolved_empty_frame_limit = _resolve_empty_frame_limit(empty_frame_limit)
    if resolved_empty_frame_limit is not None and propagation_mode != "tracker-only":
        raise ValueError("empty-frame termination is supported only by tracker-only propagation")
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
    policy_zero_frames: set[int] = set()
    termination_receipts: list[dict[str, Any]] = []
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
                raise ValueError(
                    "multi-mask seeding is supported only by tracker-only propagation"
                )
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
        if (
            int(_frame) != local_prompt
            or tuple(int(value) for value in object_ids) != expected_ids
        ):
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
                    raise RuntimeError(
                        f"mask-seed propagation yielded invalid frame {local_frame}"
                    )
                seen.add(local_frame)
                outputs = response.get("outputs")
                if not isinstance(outputs, Mapping):
                    raise RuntimeError("mask-seed propagation response has no outputs")
                normalized = normalize_video_frame_output(
                    outputs,
                    sequence_id=str(session.sequence_id),
                    session_index=int(session.session_index),
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
                ("forward", False, range(local_prompt, int(session.frame_count))),
                ("backward", True, range(local_prompt - 1, -1, -1)),
            )
            for direction, reverse, order in orders:
                consecutive_empty = 0
                empty_streak_start = None
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
                    if (
                        ids != expected_ids
                        or video_masks is None
                        or len(video_masks) < len(expected_ids)
                    ):
                        raise RuntimeError(
                            "tracker-only propagation returned "
                            f"ids={ids}, masks={getattr(video_masks, 'shape', None)}"
                        )
                    tracker_scores = _sigmoid_tracker_score_logits(
                        _scores,
                        expected_count=len(expected_ids),
                        torch_module=torch,
                    )
                    frame_masks = tuple(
                        (video_masks[index].squeeze() > 0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(bool)
                        for index in range(len(expected_ids))
                    )
                    for object_id, binary, tracker_score in zip(
                        expected_ids, frame_masks, tracker_scores
                    ):
                        predictions.append(
                            SamFramePrediction(
                                sequence_id=str(session.sequence_id),
                                session_index=int(session.session_index),
                                frame_index=int(session.frame_start) + local_frame,
                                object_id=object_id,
                                initial_detection_score=1.0,
                                frame_tracker_score=tracker_score,
                                binary_mask=binary,
                            )
                        )
                    # The authoritative prompt is visited in the forward leg so
                    # its identity can be audited, but it never contributes to
                    # an empty-propagation streak.
                    if (
                        resolved_empty_frame_limit is not None
                        and local_frame != local_prompt
                    ):
                        all_objects_empty = not any(bool(mask.any()) for mask in frame_masks)
                        if all_objects_empty:
                            if consecutive_empty == 0:
                                empty_streak_start = local_frame
                            consecutive_empty += 1
                        else:
                            consecutive_empty = 0
                            empty_streak_start = None
                        if consecutive_empty >= resolved_empty_frame_limit:
                            if empty_streak_start is None:  # pragma: no cover - guarded above
                                raise RuntimeError("empty-frame streak has no start frame")
                            if reverse:
                                skipped_start, skipped_stop = 0, local_frame
                            else:
                                skipped_start = local_frame + 1
                                skipped_stop = int(session.frame_count)
                            skipped = set(range(skipped_start, skipped_stop))
                            if seen & skipped or policy_zero_frames & skipped:
                                raise RuntimeError(
                                    "empty-frame termination produced overlapping frame ownership"
                                )
                            policy_zero_frames.update(skipped)
                            streak_low = min(int(empty_streak_start), local_frame)
                            streak_high = max(int(empty_streak_start), local_frame) + 1
                            global_start = int(session.frame_start)
                            termination_receipts.append(
                                {
                                    "direction": direction,
                                    "reason": "consecutive_all_object_empty_masks",
                                    "empty_frame_limit": int(resolved_empty_frame_limit),
                                    "empty_definition": (
                                        "all expected object masks contain zero pixels after "
                                        "the tracker mask-logit > 0 threshold"
                                    ),
                                    "first_empty_frame_in_propagation_order": (
                                        global_start + int(empty_streak_start)
                                    ),
                                    "threshold_frame": global_start + local_frame,
                                    "empty_streak_frame_range": [
                                        global_start + streak_low,
                                        global_start + streak_high,
                                    ],
                                    "last_model_visited_frame": global_start + local_frame,
                                    "policy_zero_frame_range": [
                                        global_start + skipped_start,
                                        global_start + skipped_stop,
                                    ],
                                    "policy_zero_frame_count": len(skipped),
                                    "terminated_early": bool(skipped),
                                }
                            )
                            break
        else:
            raise ValueError(f"unknown propagation_mode: {propagation_mode}")
        expected = set(range(session.frame_count))
        overlap = seen & policy_zero_frames
        missing = expected - seen - policy_zero_frames
        extra = (seen | policy_zero_frames) - expected
        if overlap or missing or extra:
            raise RuntimeError(
                "mask-seed propagation did not partition model-visited and policy-zero frames; "
                f"missing={sorted(missing)}, overlap={sorted(overlap)}, extra={sorted(extra)}"
            )
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
        "model_visited_frame_ranges": _global_half_open_ranges(
            seen, frame_start=int(session.frame_start)
        ),
        "policy_zero_frame_ranges": _global_half_open_ranges(
            policy_zero_frames, frame_start=int(session.frame_start)
        ),
        "policy_zero_frame_count": len(policy_zero_frames),
        "empty_frame_limit": resolved_empty_frame_limit,
        "empty_frame_termination_receipts": tuple(termination_receipts),
        "propagation_active_frames": active_frames,
        "anchor_preview_propagation_iou": anchor_iou,
        "anchor_propagation_object_metrics": anchor_object_metrics,
        "anchor_expected_object_ids": expected_ids,
        "anchor_returned_object_ids": anchor_ids,
        "propagation_mode": str(propagation_mode),
        "frame_tracker_score_semantics": {
            "source": (
                "tracker.propagate_in_video fifth return value: object_score_logits"
                if propagation_mode == "tracker-only"
                else None
            ),
            "source_representation": (
                "logit" if propagation_mode == "tracker-only" else None
            ),
            "stored_representation": (
                "sigmoid_probability" if propagation_mode == "tracker-only" else None
            ),
            "raw_logits_persisted": False,
        },
        "seeded_object_count": len(expected_ids),
        "seed_object_metrics": seed_object_metrics,
    }

__all__ = (
    "MASK_SEED_ANCHOR_IOU",
    "mask_metrics",
    "resolve_mask_seed_capacity",
    "run_mask_seed_session",
)
