"""Compare bounded SAM 3.1 prompts on a fixed native-resolution M1 tile.

The source video is cropped to one object-centered 1008x1008 tile before SAM,
so no whole-frame downsampling occurs. One box baseline and four point strategies
receive fresh 30-frame sessions. All outputs are diagnostic-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


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
    find_case_video,
    find_indexed_exemplar,
    install_sdpa_fallback,
    resolve_pinned_sam_runtime_provenance,
    validate_smoke_window,
    write_case_diagnostics,
)
from v19_lta_point_smoke import (
    MAX_CLICKS,
    PointClick,
    _distance_peaks,
    _error_overlay,
    _pixel_center_normalized,
    _success,
    build_centerline_strategy,
    build_distance_strategy,
    mask_metrics,
    rasterize_polygons,
    run_point_strategy,
    write_strategy_artifacts,
)
from XTA.lta_inputs import parse_yolo_segmentation_label
from XTA.lta_sam import (
    LTA_SESSION_FRAMES,
    SamPromptBox,
    SamSessionPlan,
    build_local_sam_predictor,
    resolve_local_sam_bundle,
    run_video_session,
)


@dataclass(frozen=True)
class TilePlan:
    left: int
    top: int
    size: int
    source_width: int
    source_height: int

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.size, self.top + self.size


def probe_video_dimensions(path: Path) -> tuple[int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if width < 1 or height < 1:
        raise RuntimeError(f"invalid video dimensions {width}x{height}")
    return width, height


def plan_object_tile(
    polygon: Any,
    *,
    source_width: int,
    source_height: int,
    size: int = CANVAS_SIZE,
) -> TilePlan:
    source_width = int(source_width)
    source_height = int(source_height)
    size = int(size)
    if source_width <= 0 or source_height <= 0 or size <= 0:
        raise ValueError("source dimensions and tile size must be positive")
    if source_width < size or source_height < size:
        raise ValueError(
            f"source dimensions {source_width}x{source_height} cannot contain "
            f"a {size}x{size} native tile"
        )
    x0, y0, x1, y1 = (float(value) for value in polygon.box_xyxy)
    pixel_box = (
        x0 * source_width,
        y0 * source_height,
        x1 * source_width,
        y1 * source_height,
    )
    if pixel_box[2] - pixel_box[0] > size or pixel_box[3] - pixel_box[1] > size:
        raise ValueError(f"selected polygon bbox does not fit a {size}x{size} source tile")
    center_x = (pixel_box[0] + pixel_box[2]) * 0.5
    center_y = (pixel_box[1] + pixel_box[3]) * 0.5
    left = min(max(0, int(round(center_x - size * 0.5))), source_width - size)
    top = min(max(0, int(round(center_y - size * 0.5))), source_height - size)
    plan = TilePlan(left, top, size, source_width, source_height)
    if plan.left > pixel_box[0] or plan.top > pixel_box[1] or plan.left + plan.size < pixel_box[2] or plan.top + plan.size < pixel_box[3]:
        raise RuntimeError("object-centered tile does not contain the complete polygon bbox")
    return plan


def transform_polygon_to_tile(polygon: Any, plan: TilePlan) -> Any:
    points = tuple(
        (
            (float(x) * plan.source_width - plan.left) / plan.size,
            (float(y) * plan.source_height - plan.top) / plan.size,
        )
        for x, y in polygon.points
    )
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    box = min(xs), min(ys), max(xs), max(ys)
    return SimpleNamespace(
        points=points,
        box_xyxy=box,
        row_index=getattr(polygon, "row_index", -1),
    )


def select_polygon_row(polygons: Sequence[Any], row_index: int) -> Any:
    """Return the polygon with one source-row identity, not a compact-list offset."""

    matches = [
        polygon
        for polygon in polygons
        if int(getattr(polygon, "row_index", -1)) == int(row_index)
    ]
    if len(matches) != 1:
        raise ValueError(f"selected polygon row {row_index} is not unique")
    return matches[0]


def decode_rgb_tile_frames(video_path: Path, start_frame: int, plan: TilePlan) -> list[Any]:
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    frames = []
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame)):
            raise RuntimeError(f"OpenCV could not seek to frame {start_frame}")
        reported = int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES))))
        if reported != int(start_frame):
            raise RuntimeError(
                f"OpenCV reported frame {reported} after seek to {start_frame}: "
                f"{video_path}"
            )
        for offset in range(LTA_SESSION_FRAMES):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"OpenCV stopped at frame {start_frame + offset}")
            if frame.shape[1] != plan.source_width or frame.shape[0] != plan.source_height:
                raise RuntimeError(f"video dimensions changed inside tile session: {frame.shape}")
            if frame.ndim == 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            left, top, right, bottom = plan.xyxy
            tile = rgb[top:bottom, left:right]
            if tile.shape != (plan.size, plan.size, 3):
                raise RuntimeError(f"decoded tile has wrong shape {tile.shape}")
            frames.append(Image.fromarray(tile.copy(), mode="RGB"))
    finally:
        capture.release()
    return frames


def run_box_tile(
    predictor: Any,
    *,
    resource: list[Any],
    session: SamSessionPlan,
    prompt_frame: int,
    polygon: Any,
    ground_truth: Any,
    conf: float,
) -> dict[str, Any]:
    import numpy as np

    x0, y0, x1, y1 = polygon.box_xyxy
    prompt = SamPromptBox(
        exemplar_id="tile_box",
        frame_index=int(prompt_frame),
        xywh=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
    )
    predictions = run_video_session(
        predictor,
        resource=resource,
        session=session,
        prompt=prompt,
        conf=float(conf),
        offload_video_to_cpu=True,
    )
    anchor = [item for item in predictions if item.frame_index == int(prompt_frame)]
    scored = [
        (mask_metrics(ground_truth, np.asarray(item.binary_mask, dtype=bool)), item)
        for item in anchor
    ]
    if scored:
        best_metrics, best_item = max(
            scored,
            key=lambda pair: (
                float(pair[0]["iou"]),
                float(pair[1].initial_detection_score),
                -int(pair[1].object_id),
            ),
        )
        selected_id = int(best_item.object_id)
        selected = tuple(item for item in predictions if item.object_id == selected_id)
    else:
        best_metrics = mask_metrics(ground_truth, np.zeros_like(ground_truth, dtype=bool))
        selected_id = None
        selected = ()
    return {
        "strategy": "box",
        "prompt": prompt,
        "all_predictions": predictions,
        "selected_predictions": selected,
        "selected_object_id": selected_id,
        "anchor_metrics": best_metrics,
        "acceptance_passed": bool(selected_id is not None and _success(best_metrics)),
        "active_frames": sorted({item.frame_index for item in selected}),
    }


def write_box_artifacts(
    root: Path,
    *,
    result: Mapping[str, Any],
    resource: Sequence[Any],
    session: SamSessionPlan,
    prompt_frame: int,
    ground_truth: Any,
) -> dict[str, Any]:
    import numpy as np

    target = Path(root) / "box"
    target.mkdir(parents=True, exist_ok=False)
    diagnostic = write_case_diagnostics(
        root,
        case="box",
        resource=resource,
        predictions=result["selected_predictions"],
        session=session,
        prompt=result["prompt"],
    )
    anchor = [item for item in result["selected_predictions"] if item.frame_index == int(prompt_frame)]
    mask = np.asarray(anchor[0].binary_mask, dtype=bool) if len(anchor) == 1 else np.zeros_like(ground_truth, dtype=bool)
    overlay = _error_overlay(resource[prompt_frame - session.frame_start], ground_truth, mask, ())
    error_path = target / "anchor_error.png"
    overlay.save(error_path, format="PNG", compress_level=1)
    summary = {
        "strategy": "box",
        "prompt_xywh": list(result["prompt"].xywh),
        "selected_object_id": result["selected_object_id"],
        "anchor_metrics": result["anchor_metrics"],
        "active_frames": result["active_frames"],
        "diagnostics": diagnostic,
        "anchor_error": str(error_path),
    }
    summary_path = target / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": str(summary_path), "anchor_error": str(error_path), **diagnostic}


def _serial_point_result(result: Mapping[str, Any], artifacts: Mapping[str, Any]) -> dict[str, Any]:
    anchor_iou = result["anchor_preview_propagation_iou"]
    acceptance_passed = bool(
        result["success"]
        and anchor_iou is not None
        and float(anchor_iou) >= 0.99
    )
    return {
        "strategy": result["strategy"],
        "initial_clicks": [asdict(item) for item in result["initial_clicks"]],
        "final_clicks": [asdict(item) for item in result["final_clicks"]],
        "stop_reason": result["stop_reason"],
        "final_metrics": result["final_metrics"],
        "best_metrics": result["best_metrics"],
        "best_round": result["best_round"],
        "best_click_count": result["best_click_count"],
        "success": result["success"],
        "acceptance_passed": acceptance_passed,
        "prompt_budget": {
            "box_prompts": 0,
            "max_point_clicks": int(MAX_CLICKS),
            "point_clicks_used": len(result["final_clicks"]),
        },
        "propagation_response_count": result["propagation_response_count"],
        "propagation_active_frames": result["propagation_active_frames"],
        "anchor_preview_propagation_iou": result["anchor_preview_propagation_iou"],
        "peak_allocated_mib": result["peak_allocated_mib"],
        "artifacts": dict(artifacts),
    }


def compare_tile_results(
    box_result: Mapping[str, Any],
    point_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare accepted arms using the exact prompt state that was propagated."""

    arms = [
        {
            "strategy": "box",
            "acceptance_passed": bool(box_result["acceptance_passed"]),
            "final_prompt_iou": float(box_result["anchor_metrics"]["iou"]),
            "active_frames": len(box_result["active_frames"]),
            "prompt_budget": {
                "box_prompts": 1,
                "max_point_clicks": 0,
                "point_clicks_used": 0,
            },
        }
    ]
    for result in point_results:
        anchor_iou = result["anchor_preview_propagation_iou"]
        arms.append(
            {
                "strategy": str(result["strategy"]),
                "acceptance_passed": bool(
                    result["success"]
                    and anchor_iou is not None
                    and float(anchor_iou) >= 0.99
                ),
                "final_prompt_iou": float(result["final_metrics"]["iou"]),
                "active_frames": len(result["propagation_active_frames"]),
                "prompt_budget": {
                    "box_prompts": 0,
                    "max_point_clicks": int(MAX_CLICKS),
                    "point_clicks_used": len(result["final_clicks"]),
                },
            }
        )

    accepted = [arm for arm in arms if bool(arm["acceptance_passed"])]
    accepted.sort(
        key=lambda arm: (
            -float(arm["final_prompt_iou"]),
            str(arm["strategy"]),
        )
    )
    if not accepted:
        winner = "inconclusive"
        reason = "no strategy met prompt and propagated-anchor acceptance"
        material_delta = None
    elif len(accepted) == 1:
        winner = str(accepted[0]["strategy"])
        reason = "only this strategy met prompt and propagated-anchor acceptance"
        material_delta = None
    else:
        material_delta = float(accepted[0]["final_prompt_iou"]) - float(
            accepted[1]["final_prompt_iou"]
        )
        if material_delta >= 0.01:
            winner = str(accepted[0]["strategy"])
            reason = "higher propagated-state prompt IoU by at least 0.01"
        else:
            winner = "inconclusive"
            reason = "accepted strategies differ by less than 0.01 prompt IoU"
    return {
        "winner": winner,
        "reason": reason,
        "material_iou_delta": material_delta,
        "accepted_strategy_count": len(accepted),
        "arms": arms,
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
    input_root = Path(args.input_root).expanduser().resolve(strict=True)
    exemplar_root = Path(args.exemplar_root).expanduser().resolve(strict=True)
    video = find_case_video(input_root, "m1")
    _exemplar, label_path = find_indexed_exemplar(exemplar_root, args.exemplar_index)
    _digest, polygons = parse_yolo_segmentation_label(label_path)
    selected_polygon = select_polygon_row(polygons, int(args.label_row))
    source_width, source_height = probe_video_dimensions(video)
    tile = plan_object_tile(
        selected_polygon,
        source_width=source_width,
        source_height=source_height,
    )
    local_polygons = tuple(transform_polygon_to_tile(item, tile) for item in polygons)
    local_polygon = transform_polygon_to_tile(selected_polygon, tile)
    target_mask = rasterize_polygons((local_polygon,))
    all_foreground = rasterize_polygons(local_polygons)
    runtime = resolve_pinned_sam_runtime_provenance()

    torch.cuda.set_device(int(args.device))
    predictor = None
    restore_sdpa = None
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
            max_num_objects=16,
            construction_device="meta",
        )
        constrained = configure_constrained_gpu_batches(predictor)
        restore_sdpa = install_sdpa_fallback()
        measured = _MeasuredPredictor(predictor, torch, args.device, memory)
        resource = decode_rgb_tile_frames(video, args.start_frame, tile)
        fov_mask = np.asarray(resource[args.prompt_frame - args.start_frame].convert("L")) > 3
        distance = build_distance_strategy(
            local_polygon,
            target_mask=target_mask,
            all_foreground=all_foreground,
            fov_mask=fov_mask,
        )
        centerline = build_centerline_strategy(
            local_polygon,
            target_mask=target_mask,
            all_foreground=all_foreground,
            fov_mask=fov_mask,
        )
        peaks = _distance_peaks(target_mask, 3, separation=180)
        positive_clicks = tuple(
            PointClick(*_pixel_center_normalized(x, y, CANVAS_SIZE), True, f"tile_distance_peak_{index}")
            for index, (x, y, _score) in enumerate(peaks, start=1)
        )
        strategies = {
            "single_positive": positive_clicks[:1],
            "three_positive": positive_clicks,
            "distance_balanced": distance,
            "centerline": centerline,
        }
        session = SamSessionPlan(
            sequence_id="m1__native_tile",
            session_index=0,
            frame_start=args.start_frame,
            frame_stop=args.start_frame + LTA_SESSION_FRAMES,
        )

        torch.cuda.reset_peak_memory_stats(args.device)
        print("Running native-tile box baseline...", file=sys.stderr, flush=True)
        box = run_box_tile(
            measured,
            resource=resource,
            session=session,
            prompt_frame=args.prompt_frame,
            polygon=local_polygon,
            ground_truth=target_mask,
            conf=args.conf,
        )
        box["peak_allocated_mib"] = int(torch.cuda.max_memory_allocated(args.device) // (1024 * 1024))
        box_artifacts = write_box_artifacts(
            output,
            result=box,
            resource=resource,
            session=session,
            prompt_frame=args.prompt_frame,
            ground_truth=target_mask,
        )
        point_results = []
        point_artifacts = {}
        for name, clicks in strategies.items():
            torch.cuda.reset_peak_memory_stats(args.device)
            print(f"Running native-tile point strategy {name}...", file=sys.stderr, flush=True)
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
            point_artifacts[name] = write_strategy_artifacts(
                output,
                result=result,
                resource=resource,
                ground_truth=target_mask,
                session=session,
                prompt_frame=args.prompt_frame,
            )
            point_results.append(result)
            torch.cuda.empty_cache()
    finally:
        if restore_sdpa is not None:
            restore_sdpa()
        if predictor is not None:
            shutdown = getattr(predictor, "shutdown", None)
            if callable(shutdown):
                shutdown()
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    serial_box = {
        "strategy": "box",
        "anchor_metrics": box["anchor_metrics"],
        "acceptance_passed": box["acceptance_passed"],
        "selected_object_id": box["selected_object_id"],
        "active_frames": box["active_frames"],
        "prompt_budget": {
            "box_prompts": 1,
            "max_point_clicks": 0,
            "point_clicks_used": 0,
        },
        "peak_allocated_mib": box["peak_allocated_mib"],
        "artifacts": box_artifacts,
    }
    serial_points = [
        _serial_point_result(result, point_artifacts[result["strategy"]])
        for result in point_results
    ]
    comparison = compare_tile_results(box, point_results)
    accepted_strategies = [
        str(arm["strategy"])
        for arm in comparison["arms"]
        if bool(arm["acceptance_passed"])
    ]
    summary = {
        "execution": {"completed": True},
        "acceptance": {
            "passed": bool(accepted_strategies),
            "accepted_strategies": accepted_strategies,
        },
        "experiment": "v19_lta_native_1008_tile",
        "runtime": runtime,
        "model": str(bundle.checkpoint_path),
        "video": str(video),
        "label": str(label_path),
        "zero_based_label_row": int(args.label_row),
        "session": [int(args.start_frame), int(args.start_frame + LTA_SESSION_FRAMES)],
        "prompt_frame": int(args.prompt_frame),
        "source_dimensions": [source_width, source_height],
        "tile_xyxy": list(tile.xyxy),
        "tile_size": tile.size,
        "tile_polygon_box_xyxy": list(local_polygon.box_xyxy),
        "ground_truth_pixels": int(target_mask.sum()),
        "constrained_batches": constrained,
        "box": serial_box,
        "points": serial_points,
        "comparison": comparison,
        "winner": comparison["winner"],
        "memory_events": memory,
    }
    path = output / "comparison.json"
    summary["summary_path"] = str(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
