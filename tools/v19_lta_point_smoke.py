"""Compare two deterministic SAM 3.1 point-prompt strategies on bounded M1.

Both strategies refine one point-seeded tracker object against the same known
YOLO-seg polygon on the prompt frame, then propagate once through the same
30-frame session. Outputs are diagnostic artifacts, never LTA publication.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for entry in (ROOT, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from v19_lta_gpu_smoke import (
    CANVAS_SIZE,
    DEFAULT_EXEMPLAR_INDEX,
    DEFAULT_LABEL_ROW,
    DEFAULT_PROMPT_FRAME,
    DEFAULT_START_FRAME,
    _MeasuredPredictor,
    configure_constrained_gpu_batches,
    cuda_snapshot,
    decode_rgb_frames,
    find_case_video,
    find_indexed_exemplar,
    install_sdpa_fallback,
    resize_square,
    resolve_pinned_sam_runtime_provenance,
    validate_smoke_window,
)
from XTA.lta_inputs import parse_yolo_segmentation_label
from XTA.lta_sam import (
    LTA_SESSION_FRAMES,
    SamFramePrediction,
    SamSessionPlan,
    build_local_sam_predictor,
    normalize_video_frame_output,
    resolve_local_sam_bundle,
)


MAX_CLICKS = 8
TARGET_IOU = 0.90
TARGET_DICE = 0.95
TARGET_PRECISION = 0.90
TARGET_RECALL = 0.90


@dataclass(frozen=True)
class PointClick:
    x: float
    y: float
    positive: bool
    source: str

    def __post_init__(self) -> None:
        x = float(self.x)
        y = float(self.y)
        if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("point coordinates must be finite normalized x,y values")
        if not isinstance(self.positive, bool):
            raise TypeError("point polarity must be a strict boolean")
        source = str(self.source).strip()
        if not source:
            raise ValueError("point source must not be empty")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "source", source)

    @property
    def label(self) -> int:
        return 1 if self.positive else 0

    def pixel_xy(self, size: int = CANVAS_SIZE) -> tuple[int, int]:
        size = int(size)
        return (
            min(size - 1, max(0, int(math.floor(self.x * size)))),
            min(size - 1, max(0, int(math.floor(self.y * size)))),
        )


def _pixel_center_normalized(x: int, y: int, size: int) -> tuple[float, float]:
    return (float(x) + 0.5) / int(size), (float(y) + 0.5) / int(size)


def rasterize_polygons(polygons: Sequence[Any], size: int = CANVAS_SIZE):
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("1", (int(size), int(size)), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(
            [(float(x) * int(size), float(y) * int(size)) for x, y in polygon.points],
            fill=1,
        )
    return np.asarray(image, dtype=bool)


def _distance_peaks(mask: Any, count: int, *, separation: int = 60) -> list[tuple[int, int, float]]:
    import cv2
    import numpy as np

    work = cv2.distanceTransform(np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, 5)
    peaks = []
    for _ in range(int(count)):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        score = float(work[y, x])
        if score <= 0.0:
            break
        peaks.append((int(x), int(y), score))
        cv2.circle(work, (int(x), int(y)), int(separation), 0.0, -1)
    return peaks


def _safe_exterior_candidates(
    polygon: Any,
    *,
    all_foreground: Any,
    fov_mask: Any,
    count: int,
) -> list[tuple[int, int]]:
    import cv2
    import numpy as np

    size = int(all_foreground.shape[0])
    x0, y0, x1, y1 = (float(value) for value in polygon.box_xyxy)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    margin = max(8.0 / size, 0.1 * min(x1 - x0, y1 - y0))
    candidates = (
        (x0 - margin, cy),
        (cx, y0 - margin),
        (cx, y1 + margin),
        (x1 + margin, cy),
    )
    inverse_foreground = (~np.asarray(all_foreground, dtype=bool)).astype(np.uint8)
    clearance = cv2.distanceTransform(inverse_foreground, cv2.DIST_L2, 5)
    accepted = []
    for nx, ny in candidates:
        if not 0.0 <= nx <= 1.0 or not 0.0 <= ny <= 1.0:
            continue
        x = min(size - 1, max(0, int(math.floor(nx * size))))
        y = min(size - 1, max(0, int(math.floor(ny * size))))
        if not bool(fov_mask[y, x]) or bool(all_foreground[y, x]) or float(clearance[y, x]) < 8.0:
            continue
        accepted.append((x, y))
        if len(accepted) == int(count):
            break
    if len(accepted) < int(count):
        px0 = max(0, int(math.floor(x0 * size)) - 80)
        py0 = max(0, int(math.floor(y0 * size)) - 80)
        px1 = min(size, int(math.ceil(x1 * size)) + 80)
        py1 = min(size, int(math.ceil(y1 * size)) + 80)
        valid = (
            np.asarray(fov_mask, dtype=bool)
            & ~np.asarray(all_foreground, dtype=bool)
            & (clearance >= 8.0)
        )
        region = np.zeros_like(valid)
        region[py0:py1, px0:px1] = True
        candidate_yx = np.argwhere(valid & region)
        ordered = sorted(
            ((int(x), int(y), float(clearance[y, x])) for y, x in candidate_yx),
            key=lambda item: (abs(item[2] - 16.0), item[1], item[0]),
        )
        for x, y, _clearance in ordered:
            if any((x - old_x) ** 2 + (y - old_y) ** 2 < 60**2 for old_x, old_y in accepted):
                continue
            accepted.append((x, y))
            if len(accepted) == int(count):
                break
    if len(accepted) != int(count):
        raise RuntimeError(
            f"could not derive {count} safe exterior negatives; found {accepted}"
        )
    return accepted


def build_distance_strategy(
    polygon: Any,
    *,
    target_mask: Any,
    all_foreground: Any,
    fov_mask: Any,
) -> tuple[PointClick, ...]:
    size = int(target_mask.shape[0])
    bbox_width = (float(polygon.box_xyxy[2]) - float(polygon.box_xyxy[0])) * size
    bbox_height = (float(polygon.box_xyxy[3]) - float(polygon.box_xyxy[1])) * size
    separation = int(math.ceil(0.25 * max(bbox_width, bbox_height)))
    positives = _distance_peaks(target_mask, 3, separation=separation)
    if len(positives) != 3:
        raise RuntimeError("distance strategy could not derive three interior positives")
    negatives = _safe_exterior_candidates(
        polygon,
        all_foreground=all_foreground,
        fov_mask=fov_mask,
        count=3,
    )
    return tuple(
        [
            PointClick(*_pixel_center_normalized(x, y, size), True, f"distance_peak_{index}")
            for index, (x, y, _score) in enumerate(positives, start=1)
        ]
        + [
            PointClick(*_pixel_center_normalized(x, y, size), False, f"safe_exterior_{index}")
            for index, (x, y) in enumerate(negatives, start=1)
        ]
    )


def _nearest_boundary(center: tuple[int, int], boundary_xy: Any) -> tuple[int, int]:
    import numpy as np

    x, y = center
    delta = boundary_xy - np.asarray([x, y], dtype=np.int32)
    index = int(np.argmin(np.sum(delta.astype(np.int64) ** 2, axis=1)))
    return int(boundary_xy[index, 0]), int(boundary_xy[index, 1])


def build_centerline_strategy(
    polygon: Any,
    *,
    target_mask: Any,
    all_foreground: Any,
    fov_mask: Any,
) -> tuple[PointClick, ...]:
    import cv2
    import numpy as np

    size = int(target_mask.shape[0])
    distance = cv2.distanceTransform(np.asarray(target_mask, dtype=np.uint8), cv2.DIST_L2, 5)
    center_y, center_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    center = np.asarray([int(center_x), int(center_y)], dtype=np.float64)
    exterior_yx = np.argwhere(~np.asarray(target_mask, dtype=bool))
    exterior_xy = exterior_yx[:, ::-1].astype(np.float64)
    outside_index = int(np.argmin(np.sum((exterior_xy - center) ** 2, axis=1)))
    outside = exterior_xy[outside_index]

    interior_yx = np.argwhere(np.asarray(target_mask, dtype=bool))
    interior_xy = interior_yx[:, ::-1].astype(np.float64)
    midpoint_target = (center + outside) * 0.5
    midpoint_distance = np.sum((interior_xy - midpoint_target) ** 2, axis=1)
    best_midpoint_distance = float(midpoint_distance.min())
    midpoint_indexes = np.where(midpoint_distance == best_midpoint_distance)[0]
    midpoint_index = min(
        midpoint_indexes,
        key=lambda index: (
            -float(distance[int(interior_xy[index, 1]), int(interior_xy[index, 0])]),
            int(interior_xy[index, 1]),
            int(interior_xy[index, 0]),
        ),
    )
    midpoint = interior_xy[int(midpoint_index)].astype(int)

    valid_background = (
        np.asarray(fov_mask, dtype=bool) & ~np.asarray(all_foreground, dtype=bool)
    )
    background_yx = np.argwhere(valid_background)
    background_xy = background_yx[:, ::-1].astype(np.float64)
    negative_target = center + 1.5 * (outside - center)
    negative_distance = np.sum((background_xy - negative_target) ** 2, axis=1)
    nearest = np.where(negative_distance == float(negative_distance.min()))[0]
    inverse_foreground = (~np.asarray(all_foreground, dtype=bool)).astype(np.uint8)
    clearance = cv2.distanceTransform(inverse_foreground, cv2.DIST_L2, 5)
    negative_index = min(
        nearest,
        key=lambda index: (
            -float(clearance[int(background_xy[index, 1]), int(background_xy[index, 0])]),
            int(background_xy[index, 1]),
            int(background_xy[index, 0]),
        ),
    )
    negative = background_xy[int(negative_index)].astype(int)
    return (
        PointClick(*_pixel_center_normalized(int(center[0]), int(center[1]), size), True, "centerline"),
        PointClick(*_pixel_center_normalized(int(midpoint[0]), int(midpoint[1]), size), True, "centerline_to_edge"),
        PointClick(*_pixel_center_normalized(int(negative[0]), int(negative[1]), size), False, "across_polygon_edge"),
    )


def validate_clicks(clicks: Sequence[PointClick], *, require_positive: bool = True) -> None:
    if not clicks:
        raise ValueError("point set must not be empty")
    if require_positive and not any(click.positive for click in clicks):
        raise ValueError("an initial point set requires at least one positive")
    seen: dict[tuple[float, float], bool] = {}
    for click in clicks:
        key = (click.x, click.y)
        if key in seen:
            raise ValueError(f"duplicate/conflicting point coordinate: {key}")
        seen[key] = click.positive


def mask_metrics(ground_truth: Any, prediction: Any) -> dict[str, Any]:
    import cv2
    import numpy as np

    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError(f"metric shape mismatch: {gt.shape} != {pred.shape}")
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    union = tp + fp + fn
    gt_area = int(gt.sum())
    pred_area = int(pred.sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / union if union else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    count, labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    component_areas = [int((labels == index).sum()) for index in range(1, int(count))]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "ground_truth_pixels": gt_area,
        "prediction_pixels": pred_area,
        "area_ratio": pred_area / gt_area if gt_area else None,
        "component_count": len(component_areas),
        "largest_component_fraction": (
            max(component_areas) / pred_area if component_areas and pred_area else 0.0
        ),
    }


def _success(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics["iou"]) >= TARGET_IOU
        and float(metrics["dice"]) >= TARGET_DICE
        and float(metrics["precision"]) >= TARGET_PRECISION
        and float(metrics["recall"]) >= TARGET_RECALL
    )


def next_error_clicks(
    ground_truth: Any,
    prediction: Any,
    existing: Sequence[PointClick],
    *,
    capacity: int,
    all_foreground: Any | None = None,
    fov_mask: Any | None = None,
) -> tuple[PointClick, ...]:
    import cv2
    import numpy as np

    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    size = int(gt.shape[0])
    valid_background = ~gt if all_foreground is None else ~np.asarray(all_foreground, dtype=bool)
    if fov_mask is not None:
        valid_background &= np.asarray(fov_mask, dtype=bool)
    occupied = {click.pixel_xy(gt.shape[0]) for click in existing}
    candidates = []
    for positive, error, source in (
        (True, gt & ~pred, "largest_false_negative"),
        (False, pred & valid_background, "largest_false_positive"),
    ):
        distance = cv2.distanceTransform(error.astype(np.uint8), cv2.DIST_L2, 5)
        while float(distance.max()) > 0.0:
            y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            if (int(x), int(y)) not in occupied:
                nx, ny = _pixel_center_normalized(int(x), int(y), size)
                candidates.append((float(distance[y, x]), PointClick(nx, ny, positive, source)))
                break
            cv2.circle(distance, (int(x), int(y)), 4, 0.0, -1)
    candidates.sort(key=lambda item: (-item[0], not item[1].positive, item[1].y, item[1].x))
    return tuple(item[1] for item in candidates[: max(0, int(capacity))])


def _point_preview(
    predictor: Any,
    *,
    session_id: str,
    local_frame: int,
    global_frame: int,
    object_id: int,
    clicks: Sequence[PointClick],
    clear_old_points: bool,
) -> tuple[Any, Any, bool]:
    import numpy as np

    response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(local_frame),
            "obj_id": int(object_id),
            "points": [[click.x, click.y] for click in clicks],
            "point_labels": [click.label for click in clicks],
            "clear_old_points": bool(clear_old_points),
            "rel_coordinates": True,
        }
    )
    if not isinstance(response, Mapping) or int(response.get("frame_index", -1)) != int(local_frame):
        raise RuntimeError("SAM point prompt returned an invalid frame response")
    outputs = response.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("SAM point prompt returned no outputs (object cap or API failure)")
    predictions = normalize_video_frame_output(
        outputs,
        sequence_id="m1__point_prompt",
        session_index=0,
        global_frame_index=int(global_frame),
        require_drop_stats=False,
    )
    selected = [item for item in predictions if item.object_id == int(object_id)]
    if len(selected) == 1:
        return np.asarray(selected[0].binary_mask, dtype=bool), outputs.get("frame_stats"), True
    if len(selected) > 1:
        raise RuntimeError(
            f"SAM point prompt did not return object {object_id} exactly once"
        )
    raw_ids = tuple(int(value) for value in outputs.get("out_obj_ids", ()))
    raw_masks = np.asarray(outputs.get("out_binary_masks"))
    if raw_ids or raw_masks.ndim != 3 or raw_masks.shape[0] != 0:
        raise RuntimeError(
            "SAM point prompt omitted the requested object in a malformed/nonempty output: "
            f"ids={raw_ids}, mask_shape={raw_masks.shape}"
        )
    return np.zeros(raw_masks.shape[1:], dtype=bool), outputs.get("frame_stats"), False


def _run_cleanup_steps(
    active_error: BaseException | None,
    steps: Sequence[tuple[str, Any]],
) -> None:
    """Run every cleanup step without replacing an active primary failure."""

    cleanup_error: BaseException | None = None
    for description, step in steps:
        try:
            step()
        except BaseException as exc:
            owner = active_error if active_error is not None else cleanup_error
            if owner is None:
                cleanup_error = exc
                continue
            add_note = getattr(owner, "add_note", None)
            if callable(add_note):
                add_note(
                    f"point-smoke cleanup {description} also failed: "
                    f"{type(exc).__name__}: {exc}"
                )
    if active_error is None and cleanup_error is not None:
        raise cleanup_error


def run_point_strategy(
    predictor: Any,
    *,
    strategy: str,
    resource: list[Any],
    session: SamSessionPlan,
    prompt_frame: int,
    initial_clicks: Sequence[PointClick],
    ground_truth: Any,
    all_foreground: Any,
    fov_mask: Any,
    conf: float,
) -> dict[str, Any]:
    import numpy as np

    validate_clicks(initial_clicks)
    local_prompt = int(prompt_frame) - int(session.frame_start)
    started = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": resource,
            "offload_video_to_cpu": True,
        }
    )
    if not isinstance(started, Mapping) or not str(started.get("session_id", "")).strip():
        raise RuntimeError("SAM point start_session returned no session_id")
    session_id = str(started["session_id"])
    stream = None
    iterations = []
    propagation = []
    seen_frames: set[int] = set()
    clicks = list(initial_clicks)
    stop_reason = "click_budget"
    plateau_count = 0
    active_error: BaseException | None = None
    try:
        preview_mask, _stats, preview_active = _point_preview(
            predictor,
            session_id=session_id,
            local_frame=local_prompt,
            global_frame=prompt_frame,
            object_id=0,
            clicks=initial_clicks,
            clear_old_points=True,
        )
        metrics = mask_metrics(ground_truth, preview_mask)
        iterations.append(
            {
                "round": 0,
                "new_clicks": [asdict(click) for click in initial_clicks],
                "click_count": len(clicks),
                "metrics": metrics,
                "preview_active": bool(preview_active),
                "mask": preview_mask,
            }
        )
        while len(clicks) < MAX_CLICKS:
            if _success(metrics):
                stop_reason = "quality_target"
                break
            corrections = next_error_clicks(
                ground_truth,
                preview_mask,
                clicks,
                capacity=MAX_CLICKS - len(clicks),
                all_foreground=all_foreground,
                fov_mask=fov_mask,
            )
            if not corrections:
                stop_reason = "no_error_candidate"
                break
            validate_clicks(tuple(clicks) + corrections)
            prior_iou = float(metrics["iou"])
            preview_mask, _stats, preview_active = _point_preview(
                predictor,
                session_id=session_id,
                local_frame=local_prompt,
                global_frame=prompt_frame,
                object_id=0,
                clicks=corrections,
                clear_old_points=False,
            )
            clicks.extend(corrections)
            metrics = mask_metrics(ground_truth, preview_mask)
            delta = float(metrics["iou"]) - prior_iou
            iterations.append(
                {
                    "round": len(iterations),
                    "new_clicks": [asdict(click) for click in corrections],
                    "click_count": len(clicks),
                    "metrics": metrics,
                    "preview_active": bool(preview_active),
                    "delta_iou": delta,
                    "mask": preview_mask,
                }
            )
            if delta <= -0.05:
                stop_reason = "catastrophic_regression"
                break
            plateau_count = plateau_count + 1 if delta < 0.002 else 0
            if plateau_count >= 2:
                stop_reason = "plateau"
                break
        else:
            stop_reason = "click_budget"

        stream = predictor.handle_stream_request(
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
                raise RuntimeError("SAM point propagation yielded a non-mapping response")
            local_frame = int(response.get("frame_index", -1))
            if not 0 <= local_frame < int(session.frame_count) or local_frame in seen_frames:
                raise RuntimeError(f"SAM point propagation yielded invalid frame {local_frame}")
            seen_frames.add(local_frame)
            outputs = response.get("outputs")
            if not isinstance(outputs, Mapping):
                raise RuntimeError("SAM point propagation response has no outputs")
            frame_predictions = normalize_video_frame_output(
                outputs,
                sequence_id=f"m1__point_{strategy}",
                session_index=0,
                global_frame_index=int(session.frame_start) + local_frame,
                require_drop_stats=False,
            )
            propagation.extend(item for item in frame_predictions if item.object_id == 0)
        expected = set(range(int(session.frame_count)))
        if seen_frames != expected:
            raise RuntimeError(
                f"SAM point propagation coverage mismatch: missing={sorted(expected-seen_frames)}"
            )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        close_stream = getattr(stream, "close", None)
        cleanup_steps = []
        if callable(close_stream):
            cleanup_steps.append(("stream close", close_stream))
        cleanup_steps.append(
            (
                "session close",
                lambda: predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                ),
            )
        )
        _run_cleanup_steps(active_error, cleanup_steps)

    final_preview = np.asarray(iterations[-1]["mask"], dtype=bool)
    best_iteration = max(
        iterations,
        key=lambda item: (float(item["metrics"]["iou"]), -int(item["click_count"])),
    )
    anchor = [item for item in propagation if item.frame_index == int(prompt_frame)]
    anchor_iou = None
    if len(anchor) == 1:
        anchor_iou = mask_metrics(final_preview, np.asarray(anchor[0].binary_mask, dtype=bool))["iou"]
    final_round = int(iterations[-1]["round"])
    best_round = int(best_iteration["round"])
    return {
        "strategy": str(strategy),
        "initial_clicks": tuple(initial_clicks),
        "final_clicks": tuple(clicks),
        "iterations": iterations,
        "stop_reason": stop_reason,
        "final_metrics": iterations[-1]["metrics"],
        "best_metrics": best_iteration["metrics"],
        "best_round": best_round,
        "best_click_count": int(best_iteration["click_count"]),
        "success": _success(iterations[-1]["metrics"]),
        "propagation": tuple(sorted(propagation, key=lambda item: (item.frame_index, item.object_id))),
        "propagation_response_count": len(seen_frames),
        "propagation_active_frames": sorted({item.frame_index for item in propagation}),
        "anchor_preview_propagation_iou": anchor_iou,
        "drop_stats_applicable": False,
        "propagated_revision": {
            "selection": "final",
            "round": final_round,
            "click_count": len(clicks),
            "equals_best": final_round == best_round,
        },
    }


def _draw_clicks(draw: Any, clicks: Sequence[PointClick], size: int) -> None:
    for index, click in enumerate(clicks, start=1):
        x, y = click.pixel_xy(size)
        color = (80, 255, 80) if click.positive else (255, 80, 255)
        radius = 7
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=2)
        draw.text((x + 8, y - 8), f"{'+' if click.positive else '-'}{index}", fill=color)


def _error_overlay(frame: Any, ground_truth: Any, prediction: Any, clicks: Sequence[PointClick]):
    import numpy as np
    from PIL import Image, ImageDraw

    rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred
    for mask, color in ((tp, (40, 220, 40)), (fp, (255, 40, 40)), (fn, (40, 100, 255))):
        target = np.asarray(color, dtype=np.float32)
        rgb[mask] = np.rint(rgb[mask].astype(np.float32) * 0.35 + target * 0.65).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    _draw_clicks(ImageDraw.Draw(image), clicks, image.width)
    return image


def write_strategy_artifacts(
    root: Path,
    *,
    result: Mapping[str, Any],
    resource: Sequence[Any],
    ground_truth: Any,
    session: SamSessionPlan,
    prompt_frame: int,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw

    strategy = str(result["strategy"])
    target = Path(root) / strategy
    target.mkdir(parents=True, exist_ok=False)
    Image.fromarray(np.asarray(ground_truth, dtype=np.uint8) * 255, mode="L").save(target / "ground_truth.png")
    prompt_image = resource[int(prompt_frame) - int(session.frame_start)]
    iteration_images = []
    serial_iterations = []
    cumulative: list[PointClick] = []
    for iteration in result["iterations"]:
        cumulative.extend(PointClick(**item) for item in iteration["new_clicks"])
        mask = np.asarray(iteration["mask"], dtype=bool)
        index = int(iteration["round"])
        mask_path = target / f"prompt_mask_round_{index:02d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        overlay = _error_overlay(prompt_image, ground_truth, mask, cumulative)
        overlay_path = target / f"prompt_error_round_{index:02d}.png"
        overlay.save(overlay_path, format="PNG", compress_level=1)
        iteration_images.append(overlay)
        serial = {key: value for key, value in iteration.items() if key != "mask"}
        serial["mask_path"] = str(mask_path)
        serial["overlay_path"] = str(overlay_path)
        serial_iterations.append(serial)

    if iteration_images:
        thumbs = []
        for image in iteration_images:
            thumb = image.copy()
            thumb.thumbnail((360, 360), Image.Resampling.BICUBIC)
            thumbs.append(thumb)
        sheet = Image.new("RGB", (360 * len(thumbs), 360), (24, 24, 24))
        for index, thumb in enumerate(thumbs):
            sheet.paste(thumb, (360 * index, 0))
        sheet.save(target / "refinement_contact_sheet.png", format="PNG", compress_level=1)

    by_frame: dict[int, Any] = {}
    object_masks_by_frame: dict[int, dict[int, Any]] = {}
    for item in result["propagation"]:
        frame_index = int(item.frame_index)
        object_id = int(item.object_id)
        binary = np.asarray(item.binary_mask, dtype=bool)
        if binary.shape != np.asarray(ground_truth).shape:
            raise ValueError(
                f"propagation mask shape mismatch at frame {frame_index}: "
                f"{binary.shape} != {np.asarray(ground_truth).shape}"
            )
        if frame_index not in by_frame:
            by_frame[frame_index] = np.zeros_like(binary, dtype=bool)
            object_masks_by_frame[frame_index] = {}
        if object_id in object_masks_by_frame[frame_index]:
            raise ValueError(
                f"duplicate propagation object {object_id} at frame {frame_index}"
            )
        by_frame[frame_index] |= binary
        object_masks_by_frame[frame_index][object_id] = binary
    multi_object_output = any(
        len(object_masks) > 1 for object_masks in object_masks_by_frame.values()
    )
    propagation_thumbs = []
    propagation_records = []
    for offset, frame in enumerate(resource):
        frame_index = int(session.frame_start) + offset
        mask = by_frame.get(frame_index, np.zeros_like(ground_truth, dtype=bool))
        rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
        if mask.any():
            red = np.asarray([255, 32, 32], dtype=np.float32)
            rgb[mask] = np.rint(rgb[mask].astype(np.float32) * 0.4 + red * 0.6).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.text((5, 4), f"frame {frame_index}{' ACTIVE' if mask.any() else ''}", fill=(255, 255, 255))
        thumb = image.copy()
        thumb.thumbnail((252, 252), Image.Resampling.BICUBIC)
        propagation_thumbs.append(thumb)
        frame_object_masks = object_masks_by_frame.get(frame_index, {})
        per_object = []
        for object_id, object_mask in sorted(frame_object_masks.items()):
            pixels = int(object_mask.sum())
            object_mask_path = None
            if multi_object_output and pixels:
                path = target / f"frame_{frame_index:04d}_object_{object_id:03d}_mask.png"
                Image.fromarray(object_mask.astype(np.uint8) * 255, mode="L").save(path)
                object_mask_path = str(path)
            per_object.append(
                {
                    "object_id": object_id,
                    "active": bool(pixels),
                    "mask_pixels": pixels,
                    "mask_path": object_mask_path,
                }
            )
        propagation_records.append(
            {
                "frame_index": frame_index,
                "mask_pixels": int(mask.sum()),
                "returned_object_ids": [item["object_id"] for item in per_object],
                "active_object_ids": [
                    item["object_id"] for item in per_object if item["active"]
                ],
                "objects": per_object,
            }
        )
        if mask.any() or frame_index == int(prompt_frame):
            image.save(target / f"frame_{frame_index:04d}_overlay.png", format="PNG", compress_level=1)
        if mask.any():
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                target / f"frame_{frame_index:04d}_mask.png"
            )
    sheet = Image.new("RGB", (6 * 252, 5 * 252), (24, 24, 24))
    for index, thumb in enumerate(propagation_thumbs):
        sheet.paste(thumb, ((index % 6) * 252, (index // 6) * 252))
    sheet.save(target / "propagation_contact_sheet.png", format="PNG", compress_level=1)

    summary = {
        "strategy": strategy,
        "stop_reason": result["stop_reason"],
        "success": bool(result["success"]),
        "initial_clicks": [asdict(item) for item in result["initial_clicks"]],
        "final_clicks": [asdict(item) for item in result["final_clicks"]],
        "iterations": serial_iterations,
        "final_metrics": result["final_metrics"],
        "best_metrics": result["best_metrics"],
        "best_round": result["best_round"],
        "best_click_count": result["best_click_count"],
        "propagation_response_count": result["propagation_response_count"],
        "propagation_active_frames": result["propagation_active_frames"],
        "anchor_preview_propagation_iou": result["anchor_preview_propagation_iou"],
        "drop_stats_applicable": bool(result["drop_stats_applicable"]),
        "propagated_revision": dict(result["propagated_revision"]),
        "propagation_frames": propagation_records,
    }
    for field_name in (
        "anchor_integrity_passed",
        "diagnostic_propagation_gate_passed",
        "non_anchor_active_frames",
        "anchor_propagation_object_metrics",
        "anchor_expected_object_ids",
        "anchor_returned_object_ids",
        "propagation_mode",
        "seeded_object_count",
        "seed_object_metrics",
        "anchor_integrity_minimum_iou",
        "success_definition",
    ):
        if field_name in result:
            summary[field_name] = result[field_name]
    summary_path = target / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "directory": str(target),
        "summary": str(summary_path),
        "refinement_contact_sheet": str(target / "refinement_contact_sheet.png"),
        "propagation_contact_sheet": str(target / "propagation_contact_sheet.png"),
    }


def compare_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        results,
        key=lambda item: (
            not bool(item["success"]),
            len(item["final_clicks"]),
            -float(item["final_metrics"]["iou"]),
        ),
    )
    first, second = ranked
    iou_delta = float(first["final_metrics"]["iou"]) - float(second["final_metrics"]["iou"])
    if bool(first["success"]) and (
        not bool(second["success"]) or len(first["final_clicks"]) < len(second["final_clicks"])
    ):
        winner = str(first["strategy"])
        reason = "reached the quality target with fewer clicks or while the other strategy missed"
    elif abs(iou_delta) >= 0.01:
        winner = str(first["strategy"])
        reason = f"higher final prompt IoU by {abs(iou_delta):.4f}"
    else:
        winner = "inconclusive"
        reason = "final prompt IoU differed by less than 0.01 at this single polygon"
    return {"winner": winner, "reason": reason}


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
        "--weight-storage",
        choices=("bfloat16_egpu", "float32"),
        default="bfloat16_egpu",
    )
    return parser


def main() -> None:
    import numpy as np
    import torch
    from PIL import Image

    args = _build_parser().parse_args()
    validate_smoke_window(args.start_frame, args.prompt_frame)
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"--output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = resolve_local_sam_bundle(args.model)
    if bundle.model_version != "sam3.1":
        raise ValueError("point comparison requires SAM 3.1")
    input_root = Path(args.input_root).expanduser().resolve(strict=True)
    exemplar_root = Path(args.exemplar_root).expanduser().resolve(strict=True)
    video = find_case_video(input_root, "m1")
    exemplar_path, label_path = find_indexed_exemplar(exemplar_root, args.exemplar_index)
    _label_digest, polygons = parse_yolo_segmentation_label(label_path)
    selected = [item for item in polygons if int(item.row_index) == int(args.label_row)]
    if len(selected) != 1:
        raise ValueError("selected YOLO row is not unique")
    polygon = selected[0]
    target_mask = rasterize_polygons((polygon,))
    all_foreground = rasterize_polygons(polygons)
    sam_runtime = resolve_pinned_sam_runtime_provenance()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(int(args.device))
    predictor = None
    restore_sdpa = None
    memory_events = [cuda_snapshot(torch, args.device, "before_builder")]
    active_error: BaseException | None = None
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
        )
        constrained = configure_constrained_gpu_batches(predictor)
        restore_sdpa = install_sdpa_fallback()
        measured = _MeasuredPredictor(predictor, torch, args.device, memory_events)
        decoded = decode_rgb_frames(video, args.start_frame)
        resource = resize_square(decoded)
        del decoded
        prompt_image = np.asarray(resource[args.prompt_frame - args.start_frame].convert("L"))
        fov_mask = prompt_image > 3
        strategies = {
            "distance": build_distance_strategy(
                polygon,
                target_mask=target_mask,
                all_foreground=all_foreground,
                fov_mask=fov_mask,
            ),
            "centerline": build_centerline_strategy(
                polygon,
                target_mask=target_mask,
                all_foreground=all_foreground,
                fov_mask=fov_mask,
            ),
        }
        session = SamSessionPlan(
            sequence_id="m1__point_comparison",
            session_index=0,
            frame_start=args.start_frame,
            frame_stop=args.start_frame + LTA_SESSION_FRAMES,
        )
        results = []
        artifacts = {}
        for name, clicks in strategies.items():
            validate_clicks(clicks)
            torch.cuda.reset_peak_memory_stats(args.device)
            print(f"Running point strategy {name} with {len(clicks)} initial clicks...", file=sys.stderr, flush=True)
            result = run_point_strategy(
                measured,
                strategy=name,
                resource=resource,
                session=session,
                prompt_frame=args.prompt_frame,
                initial_clicks=clicks,
                ground_truth=target_mask,
                all_foreground=all_foreground,
                fov_mask=fov_mask,
                conf=args.conf,
            )
            result["peak_allocated_mib"] = int(torch.cuda.max_memory_allocated(args.device) // (1024 * 1024))
            artifacts[name] = write_strategy_artifacts(
                output,
                result=result,
                resource=resource,
                ground_truth=target_mask,
                session=session,
                prompt_frame=args.prompt_frame,
            )
            results.append(result)
            torch.cuda.empty_cache()
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_steps = []
        if restore_sdpa is not None:
            cleanup_steps.append(("SDPA restore", restore_sdpa))
        if predictor is not None:
            shutdown = getattr(predictor, "shutdown", None)
            if callable(shutdown):
                cleanup_steps.append(("predictor shutdown", shutdown))
        predictor = None
        cleanup_steps.append(("garbage collection", gc.collect))
        if torch.cuda.is_available():
            cleanup_steps.append(("CUDA cache release", torch.cuda.empty_cache))
        _run_cleanup_steps(active_error, cleanup_steps)

    serial_results = []
    for result in results:
        serial_results.append(
            {
                "strategy": result["strategy"],
                "initial_clicks": [asdict(item) for item in result["initial_clicks"]],
                "final_clicks": [asdict(item) for item in result["final_clicks"]],
                "stop_reason": result["stop_reason"],
                "final_metrics": result["final_metrics"],
                "best_metrics": result["best_metrics"],
                "best_round": result["best_round"],
                "best_click_count": result["best_click_count"],
                "success": result["success"],
                "propagation_response_count": result["propagation_response_count"],
                "propagation_active_frames": result["propagation_active_frames"],
                "anchor_preview_propagation_iou": result["anchor_preview_propagation_iou"],
                "drop_stats_applicable": bool(result["drop_stats_applicable"]),
                "propagated_revision": dict(result["propagated_revision"]),
                "peak_allocated_mib": result["peak_allocated_mib"],
                "artifacts": artifacts[result["strategy"]],
            }
        )
    summary = {
        "status": "complete",
        "experiment": "v19_lta_m1_point_comparison",
        "model": str(bundle.checkpoint_path),
        "sam_runtime": sam_runtime,
        "video": str(video),
        "exemplar": str(exemplar_path),
        "label": str(label_path),
        "encoded_index": int(args.exemplar_index),
        "zero_based_label_row": int(args.label_row),
        "frame_start": int(args.start_frame),
        "frame_stop": int(args.start_frame + LTA_SESSION_FRAMES),
        "prompt_frame": int(args.prompt_frame),
        "max_clicks": MAX_CLICKS,
        "quality_target": {
            "iou": TARGET_IOU,
            "dice": TARGET_DICE,
            "precision": TARGET_PRECISION,
            "recall": TARGET_RECALL,
        },
        "constrained_batches": constrained,
        "results": serial_results,
        "comparison": compare_results(results),
        "memory_events": memory_events,
    }
    summary_path = output / "comparison.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
