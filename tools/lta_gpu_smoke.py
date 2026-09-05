"""Run a bounded, non-publishing SAM 3.1 LTA hardware smoke.

The tool accepts only an explicit local model bundle and exactly thirty decoded
frames. It never downloads a checkpoint and writes nothing by default. An
explicit ``--diagnostic-output`` may emit review-only overlays and masks; these
are not LTA publication artifacts. The ``direct`` case exercises a same-source
visual exemplar; ``composite`` exercises the cross-image conditioning seam.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.lta_inputs import parse_yolo_segmentation_label, split_indexed_stem
from XTA.lta_sam import (
    LTA_SESSION_FRAMES,
    PINNED_SAM_COMMIT,
    PINNED_SAM_PACKAGE_FILE_COUNT,
    PINNED_SAM_PACKAGE_TREE_SHA256,
    PINNED_SAM_SOURCE_URL,
    PINNED_SAM_VERSION,
    SamPromptBox,
    SamSessionPlan,
    _canonical_sam_package_fingerprint as _shared_sam_package_fingerprint,
    build_local_sam_predictor,
    configure_constrained_gpu_batches,
    install_sdpa_fallback,
    resolve_installed_sam_bpe,
    resolve_local_sam_bundle,
    resolve_pinned_sam_runtime_provenance as _shared_sam_runtime_provenance,
    run_video_session,
)


CANVAS_SIZE = 1008


def _existing_directory(value: str | Path, *, name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except Exception as exc:
        raise ValueError(f"{name} is not an existing local path: {value!r}") from exc
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory: {path}")
    return path


def validate_smoke_window(start_frame: int, prompt_frame: int) -> None:
    if int(start_frame) < 0:
        raise ValueError("--start-frame must be non-negative")
    stop_frame = int(start_frame) + LTA_SESSION_FRAMES
    if not int(start_frame) <= int(prompt_frame) < stop_frame:
        raise ValueError(
            "--prompt-frame must fall inside the fixed 30-frame session "
            f"[{start_frame},{stop_frame})"
        )


def _canonical_sam_package_fingerprint(distribution: Any) -> tuple[str, int, Path]:
    return _shared_sam_package_fingerprint(distribution)


def resolve_pinned_sam_runtime_provenance() -> dict[str, object]:
    return _shared_sam_runtime_provenance(
        pinned_version=PINNED_SAM_VERSION,
        pinned_commit=PINNED_SAM_COMMIT,
        pinned_source_url=PINNED_SAM_SOURCE_URL,
        pinned_package_file_count=PINNED_SAM_PACKAGE_FILE_COUNT,
        pinned_package_tree_sha256=PINNED_SAM_PACKAGE_TREE_SHA256,
        bpe_resolver=resolve_installed_sam_bpe,
    )


def find_case_video(input_root: Path, case: str) -> Path:
    """Resolve a sample-neutral source video for one diagnostic arm.

    A file may be supplied directly.  A directory may contain one MKV, or one
    file prefixed with the arm name when both diagnostic arms are staged.
    """

    arm = str(case).lower()
    if arm not in {"direct", "composite"}:
        raise ValueError(f"unknown smoke case: {case}")
    root = Path(input_root)
    if root.is_file():
        if root.suffix.lower() != ".mkv":
            raise ValueError(f"LTA smoke input must be an MKV: {root}")
        return root.resolve()
    all_matches = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".mkv"
    )
    if len(all_matches) == 1:
        return all_matches[0]
    matches = [
        path for path in all_matches if path.name.lower().startswith(f"{arm}_")
    ]
    if len(matches) != 1:
        names = [path.name for path in all_matches]
        raise ValueError(
            f"expected one MKV or one {arm}_*.mkv under {root}; found {names}"
        )
    return matches[0]


def find_indexed_exemplar(exemplar_root: Path, encoded_index: int) -> tuple[Path, Path]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    matches = []
    for path in sorted(Path(exemplar_root).iterdir()):
        if not path.is_file() or path.suffix.lower() not in image_suffixes:
            continue
        _stem, parsed_index = split_indexed_stem(path)
        if parsed_index == int(encoded_index):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError(
            f"expected one exemplar image with embedded index {encoded_index}; "
            f"found {[path.name for path in matches]}"
        )
    image_path = matches[0]
    label_path = image_path.with_suffix(".txt")
    if not label_path.is_file():
        raise ValueError(f"exemplar has no stem-matched YOLO-seg label: {image_path}")
    return image_path, label_path.resolve()


def prompt_xywh_from_label(label_path: Path, row_index: int) -> tuple[float, float, float, float]:
    _digest, polygons = parse_yolo_segmentation_label(Path(label_path))
    selected = [polygon for polygon in polygons if polygon.row_index == int(row_index)]
    if len(selected) != 1:
        raise ValueError(
            f"YOLO-seg label {label_path} has no unique zero-based row {row_index}"
        )
    min_x, min_y, max_x, max_y = selected[0].box_xyxy
    return min_x, min_y, max_x - min_x, max_y - min_y


def decode_rgb_frames(video_path: Path, start_frame: int) -> list[Any]:
    """Decode exactly one fixed LTA session as ordered RGB PIL images."""

    import cv2
    from PIL import Image

    if int(start_frame) < 0:
        raise ValueError("start_frame must be non-negative")
    capture = cv2.VideoCapture(str(Path(video_path)))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    frames = []
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame)):
            raise RuntimeError(f"OpenCV could not seek {video_path} to frame {start_frame}")
        reported = int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES))))
        if reported != int(start_frame):
            raise RuntimeError(
                f"OpenCV reported frame {reported} after seek to {start_frame}: {video_path}"
            )
        for offset in range(LTA_SESSION_FRAMES):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"OpenCV stopped at frame {int(start_frame) + offset}: {video_path}"
                )
            if frame.ndim == 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError(f"unsupported decoded frame shape {frame.shape}")
            frames.append(Image.fromarray(rgb, mode="RGB"))
    finally:
        capture.release()
    if len(frames) != LTA_SESSION_FRAMES:
        raise RuntimeError(f"decoded {len(frames)} frames, expected {LTA_SESSION_FRAMES}")
    return frames


def resize_square(frames: Iterable[Any], size: int = CANVAS_SIZE) -> list[Any]:
    from PIL import Image

    resampling = Image.Resampling.BICUBIC
    return [frame.convert("RGB").resize((int(size), int(size)), resampling) for frame in frames]


def build_composite_frames(
    target_frames: Sequence[Any],
    exemplar_image: Any,
    exemplar_xywh: Sequence[float],
    *,
    canvas_size: int = CANVAS_SIZE,
    exemplar_frame_offset: int | None = None,
) -> tuple[list[Any], tuple[float, float, float, float], tuple[int, int, int, int]]:
    """Build the fixed exemplar-left, target-right conditioning canvas."""

    from PIL import Image

    if int(canvas_size) < 3 or int(canvas_size) % 3:
        raise ValueError("canvas_size must be a positive multiple of three")
    if len(tuple(exemplar_xywh)) != 4:
        raise ValueError("exemplar_xywh must contain four values")
    if len(target_frames) != LTA_SESSION_FRAMES:
        raise ValueError(
            f"composite diagnostic requires {LTA_SESSION_FRAMES} frames; got {len(target_frames)}"
        )
    if exemplar_frame_offset is not None and not (
        0 <= int(exemplar_frame_offset) < LTA_SESSION_FRAMES
    ):
        raise ValueError("exemplar_frame_offset must address the fixed session")

    canvas_size = int(canvas_size)
    exemplar_size = canvas_size // 3
    target_lane_x = exemplar_size
    target_lane_width = canvas_size - exemplar_size
    exemplar_top = (canvas_size - exemplar_size) // 2
    resampling = Image.Resampling.BICUBIC
    exemplar = exemplar_image.convert("RGB").resize(
        (exemplar_size, exemplar_size), resampling
    )

    composites = []
    target_rect: tuple[int, int, int, int] | None = None
    for frame_offset, frame in enumerate(target_frames):
        target = frame.convert("RGB")
        source_width, source_height = target.size
        scale = min(target_lane_width / source_width, canvas_size / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = target.resize((resized_width, resized_height), resampling)
        target_left = target_lane_x + (target_lane_width - resized_width) // 2
        target_top = (canvas_size - resized_height) // 2
        current_rect = (
            target_left,
            target_top,
            target_left + resized_width,
            target_top + resized_height,
        )
        if target_rect is None:
            target_rect = current_rect
        elif current_rect != target_rect:
            raise RuntimeError("composite target geometry changed inside one fixed session")
        canvas = Image.new("RGB", (canvas_size, canvas_size), (128, 128, 128))
        if exemplar_frame_offset is None or frame_offset == int(exemplar_frame_offset):
            canvas.paste(exemplar, (0, exemplar_top))
        canvas.paste(resized, (target_left, target_top))
        composites.append(canvas)

    assert target_rect is not None
    x, y, width, height = (float(value) for value in exemplar_xywh)
    mapped = (
        (x * exemplar_size) / canvas_size,
        (exemplar_top + y * exemplar_size) / canvas_size,
        (width * exemplar_size) / canvas_size,
        (height * exemplar_size) / canvas_size,
    )
    return composites, mapped, target_rect


def summarize_predictions(
    predictions: Sequence[Any],
    *,
    target_rect: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    import numpy as np

    scores = [float(prediction.initial_detection_score) for prediction in predictions]
    frame_indices = sorted({int(prediction.frame_index) for prediction in predictions})
    object_ids = sorted({int(prediction.object_id) for prediction in predictions})
    summary: dict[str, Any] = {
        "prediction_count": len(predictions),
        "frames_with_predictions": len(frame_indices),
        "frame_indices_with_predictions": frame_indices,
        "distinct_object_ids": len(object_ids),
        "score_min": min(scores) if scores else None,
        "score_mean": (sum(scores) / len(scores)) if scores else None,
        "score_max": max(scores) if scores else None,
        "mask_pixels": int(
            sum(np.asarray(prediction.binary_mask, dtype=bool).sum() for prediction in predictions)
        ),
    }
    if target_rect is not None:
        x0, y0, x1, y1 = target_rect
        target_hits = 0
        exemplar_only_hits = 0
        cross_region_hits = 0
        target_pixels = 0
        exemplar_pixels = 0
        for prediction in predictions:
            mask = np.asarray(prediction.binary_mask, dtype=bool)
            target_count = int(mask[y0:y1, x0:x1].sum())
            exemplar_count = int(mask[:, :x0].sum())
            target_pixels += target_count
            exemplar_pixels += exemplar_count
            has_target = target_count > 0
            has_exemplar = exemplar_count > 0
            target_hits += int(has_target)
            exemplar_only_hits += int(has_exemplar and not has_target)
            cross_region_hits += int(has_exemplar and has_target)
        summary.update(
            {
                "target_rect_xyxy": list(target_rect),
                "target_prediction_hits": target_hits,
                "exemplar_only_prediction_hits": exemplar_only_hits,
                "cross_region_prediction_hits": cross_region_hits,
                "target_mask_pixels": target_pixels,
                "left_exemplar_mask_pixels": exemplar_pixels,
            }
        )
    return summary


def evaluate_case_acceptance(case: str, summary: dict[str, Any]) -> dict[str, Any]:
    case = str(case).lower()
    if case == "direct":
        passed = int(summary.get("prediction_count", 0)) > 0
        reason = (
            "at least one active object-frame mask was produced"
            if passed
            else "no active object-frame mask was produced"
        )
    elif case == "composite":
        passed = int(summary.get("target_prediction_hits", 0)) > 0
        reason = (
            "at least one prediction intersected the composite target panel"
            if passed
            else "no prediction intersected the composite target panel"
        )
    else:
        raise ValueError(f"unknown smoke case: {case}")
    return {"passed": bool(passed), "reason": reason}


def validate_diagnostic_output(
    value: str | Path | None,
    cases: Sequence[str],
) -> Path | None:
    if value is None:
        return None
    root = Path(value).expanduser().resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise ValueError(f"--diagnostic-output must be a directory path: {root}")
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "--diagnostic-output must be new or empty; refusing to overwrite: "
            f"{root}"
        )
    for case in cases:
        case_dir = root / str(case).lower()
        if case_dir.exists() and any(case_dir.iterdir()):
            raise ValueError(
                "--diagnostic-output refuses to overwrite a nonempty case directory: "
                f"{case_dir}"
            )
    return root


def write_case_diagnostics(
    output_root: Path,
    *,
    case: str,
    resource: Sequence[Any],
    predictions: Sequence[Any],
    session: SamSessionPlan,
    prompt: SamPromptBox,
    target_rect: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Write review-only masks, overlays, and a 30-frame contact sheet."""

    import numpy as np
    from PIL import Image, ImageDraw

    if len(resource) != session.frame_count:
        raise ValueError("diagnostic resource does not match the session frame count")
    case_dir = Path(output_root) / str(case).lower()
    if case_dir.exists() and any(case_dir.iterdir()):
        raise ValueError(f"diagnostic case directory is not empty: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[Any]] = {}
    for prediction in predictions:
        by_frame.setdefault(int(prediction.frame_index), []).append(prediction)

    thumbnails = []
    interesting = []
    mask_files = []
    overlay_files = []
    for offset, frame in enumerate(resource):
        global_frame = int(session.frame_start) + offset
        rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
        height, width = rgb.shape[:2]
        union = np.zeros((height, width), dtype=bool)
        frame_predictions = by_frame.get(global_frame, [])
        for prediction in frame_predictions:
            mask = np.asarray(prediction.binary_mask, dtype=bool)
            if mask.shape != union.shape:
                raise ValueError(
                    f"diagnostic mask shape {mask.shape} does not match frame {union.shape}"
                )
            union |= mask
        if union.any():
            red = np.asarray([255, 32, 32], dtype=np.float32)
            rgb[union] = np.rint(
                rgb[union].astype(np.float32) * 0.4 + red * 0.6
            ).astype(np.uint8)
        overlay = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(overlay)
        if global_frame == int(prompt.frame_index):
            x, y, box_width, box_height = prompt.xywh
            draw.rectangle(
                (
                    int(round(x * width)),
                    int(round(y * height)),
                    int(round((x + box_width) * width)),
                    int(round((y + box_height) * height)),
                ),
                outline=(0, 220, 255),
                width=max(2, width // 252),
            )
        if target_rect is not None:
            draw.rectangle(target_rect, outline=(255, 220, 0), width=max(2, width // 252))
        label = f"frame {global_frame}"
        if global_frame == int(prompt.frame_index):
            label += "  PROMPT"
        if frame_predictions:
            label += f"  masks={len(frame_predictions)}"
        draw.rectangle((0, 0, min(width, 230), 20), fill=(0, 0, 0))
        draw.text((5, 4), label, fill=(255, 255, 255))

        thumb = overlay.copy()
        thumb.thumbnail((252, 252), Image.Resampling.BICUBIC)
        thumbnails.append((global_frame, thumb))
        if union.any() or global_frame == int(prompt.frame_index):
            interesting.append(global_frame)
            overlay_path = case_dir / f"frame_{global_frame:04d}_overlay.png"
            overlay.save(overlay_path, format="PNG", compress_level=1)
            overlay_files.append(str(overlay_path))
        if union.any():
            mask_path = case_dir / f"frame_{global_frame:04d}_mask.png"
            Image.fromarray(union.astype(np.uint8) * 255, mode="L").save(
                mask_path,
                format="PNG",
                compress_level=1,
            )
            mask_files.append(str(mask_path))

    columns = 6
    rows = (len(thumbnails) + columns - 1) // columns
    cell_width = max(image.width for _, image in thumbnails)
    cell_height = max(image.height for _, image in thumbnails)
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        (24, 24, 24),
    )
    for index, (_frame_index, thumbnail) in enumerate(thumbnails):
        left = (index % columns) * cell_width
        top = (index // columns) * cell_height
        sheet.paste(thumbnail, (left, top))
    contact_sheet = case_dir / "contact_sheet.png"
    sheet.save(contact_sheet, format="PNG", compress_level=1)

    record = {
        "diagnostic_only": True,
        "case": str(case).lower(),
        "frame_start": int(session.frame_start),
        "frame_stop": int(session.frame_stop),
        "prompt_frame": int(prompt.frame_index),
        "interesting_frames": interesting,
        "contact_sheet": str(contact_sheet),
        "overlay_files": overlay_files,
        "mask_files": mask_files,
    }
    manifest = case_dir / "diagnostics.json"
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    record["manifest"] = str(manifest)
    return record


def cuda_snapshot(torch_module: Any, device_id: int, phase: str) -> dict[str, Any]:
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(int(device_id))
    mib = 1024 * 1024
    return {
        "phase": str(phase),
        "allocated_mib": int(torch_module.cuda.memory_allocated(int(device_id)) // mib),
        "reserved_mib": int(torch_module.cuda.memory_reserved(int(device_id)) // mib),
        "peak_allocated_mib": int(
            torch_module.cuda.max_memory_allocated(int(device_id)) // mib
        ),
        "free_mib": int(free_bytes // mib),
        "total_mib": int(total_bytes // mib),
    }




class _MeasuredPredictor:
    def __init__(self, predictor: Any, torch_module: Any, device_id: int, events: list[dict]):
        self.predictor = predictor
        self.torch = torch_module
        self.device_id = int(device_id)
        self.events = events

    def handle_request(self, request: dict[str, Any]) -> Any:
        request_type = str(request.get("type", "request"))
        self.events.append(cuda_snapshot(self.torch, self.device_id, f"before_{request_type}"))
        response = self.predictor.handle_request(request)
        self.events.append(cuda_snapshot(self.torch, self.device_id, f"after_{request_type}"))
        return response

    def handle_stream_request(self, request: dict[str, Any]) -> Iterable[Any]:
        request_type = str(request.get("type", "stream"))
        self.events.append(cuda_snapshot(self.torch, self.device_id, f"before_{request_type}"))
        try:
            yield from self.predictor.handle_stream_request(request)
        finally:
            self.events.append(cuda_snapshot(self.torch, self.device_id, f"after_{request_type}"))


def run_case(
    predictor: Any,
    *,
    case: str,
    video_path: Path,
    exemplar_image: Any,
    exemplar_xywh: tuple[float, float, float, float],
    start_frame: int,
    prompt_frame: int,
    conf: float,
    composite_exemplar_visibility: str = "prompt-only",
    diagnostic_output: Path | None = None,
) -> dict[str, Any]:
    if not int(start_frame) <= int(prompt_frame) < int(start_frame) + LTA_SESSION_FRAMES:
        raise ValueError("prompt_frame must fall inside the fixed 30-frame session")
    started = time.perf_counter()
    decoded = decode_rgb_frames(video_path, int(start_frame))
    target_rect = None
    mapped_xywh = exemplar_xywh
    if str(case).lower() == "direct":
        resource = resize_square(decoded)
    elif str(case).lower() == "composite":
        exemplar_frame_offset = (
            int(prompt_frame) - int(start_frame)
            if str(composite_exemplar_visibility) == "prompt-only"
            else None
        )
        resource, mapped_xywh, target_rect = build_composite_frames(
            decoded,
            exemplar_image,
            exemplar_xywh,
            exemplar_frame_offset=exemplar_frame_offset,
        )
    else:
        raise ValueError(f"unknown smoke case: {case}")
    del decoded

    session = SamSessionPlan(
        sequence_id=f"{str(case).lower()}__transverse_smoke",
        session_index=0,
        frame_start=int(start_frame),
        frame_stop=int(start_frame) + LTA_SESSION_FRAMES,
    )
    prompt = SamPromptBox(
        exemplar_id="curated_visual_exemplar",
        frame_index=int(prompt_frame),
        xywh=tuple(float(value) for value in mapped_xywh),
    )
    predictions = run_video_session(
        predictor,
        resource=resource,
        session=session,
        prompt=prompt,
        conf=float(conf),
        offload_video_to_cpu=True,
    )
    result = {
        "case": str(case).lower(),
        "video": str(video_path),
        "frame_start": int(start_frame),
        "frame_stop": int(start_frame) + LTA_SESSION_FRAMES,
        "prompt_frame": int(prompt_frame),
        "prompt_xywh": list(prompt.xywh),
        "composite_exemplar_visibility": (
            str(composite_exemplar_visibility)
            if str(case).lower() == "composite"
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": summarize_predictions(predictions, target_rect=target_rect),
    }
    result["acceptance"] = evaluate_case_acceptance(case, result["predictions"])
    if diagnostic_output is not None:
        result["diagnostics"] = write_case_diagnostics(
            diagnostic_output,
            case=case,
            resource=resource,
            predictions=predictions,
            session=session,
            prompt=prompt,
            target_rect=target_rect,
        )
    del predictions
    del resource
    gc.collect()
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local SAM 3.1 file/directory")
    parser.add_argument("--input-root", required=True, help="Source MKV or directory of diagnostic MKVs")
    parser.add_argument(
        "--exemplar-root",
        required=True,
        help="Directory containing indexed exemplar JPG/TXT pairs",
    )
    parser.add_argument("--case", choices=("direct", "composite", "both"), default="direct")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--prompt-frame", type=int, required=True)
    parser.add_argument("--exemplar-index", type=int, required=True)
    parser.add_argument(
        "--label-row",
        type=int,
        required=True,
        help="Zero-based YOLO-seg row within the selected exemplar label",
    )
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument(
        "--diagnostic-output",
        default=None,
        help="Optional new/empty directory for review-only overlays, masks, and contact sheets",
    )
    parser.add_argument(
        "--composite-exemplar-visibility",
        choices=("prompt-only", "all"),
        default="prompt-only",
        help="Show the left exemplar only on the prompt frame or on every composite frame",
    )
    parser.add_argument(
        "--weight-storage",
        choices=("bfloat16_egpu", "float32"),
        default="bfloat16_egpu",
        help="Use mixed BF16/FP32 storage for a constrained eGPU or full FP32",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    validate_smoke_window(int(args.start_frame), int(args.prompt_frame))
    bundle = resolve_local_sam_bundle(args.model)
    if bundle.model_version != "sam3.1":
        raise ValueError("the LTA GPU smoke requires a local SAM 3.1 bundle")
    input_root = Path(args.input_root).expanduser().resolve(strict=True)
    if not (input_root.is_file() or input_root.is_dir()):
        raise ValueError(f"--input-root is not a file or directory: {input_root}")
    exemplar_root = _existing_directory(args.exemplar_root, name="--exemplar-root")
    exemplar_path, label_path = find_indexed_exemplar(
        exemplar_root, int(args.exemplar_index)
    )
    exemplar_xywh = prompt_xywh_from_label(label_path, int(args.label_row))
    selected_cases = ("direct", "composite") if args.case == "both" else (args.case,)
    if input_root.is_file() and len(selected_cases) != 1:
        raise ValueError("--case both requires a directory with direct_ and composite_ MKVs")
    diagnostic_output = validate_diagnostic_output(args.diagnostic_output, selected_cases)
    videos = {case: find_case_video(input_root, case) for case in selected_cases}
    sam_runtime = resolve_pinned_sam_runtime_provenance()

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if int(args.device) < 0 or int(args.device) >= int(torch.cuda.device_count()):
        raise ValueError(
            f"--device {args.device} is outside CUDA device_count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(int(args.device))
    torch.cuda.reset_peak_memory_stats(int(args.device))
    memory_events = [cuda_snapshot(torch, int(args.device), "before_builder")]
    exemplar_image = Image.open(exemplar_path).convert("RGB")
    predictor = None
    started = time.perf_counter()
    case_results = []
    constrained_batches = None
    restore_sdpa = None
    try:
        print("Loading the explicit local SAM 3.1 bundle...", file=sys.stderr, flush=True)
        predictor = build_local_sam_predictor(
            bundle,
            device_id=int(args.device),
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            conf=float(args.conf),
            weight_storage=str(args.weight_storage),
        )
        constrained_batches = configure_constrained_gpu_batches(predictor)
        restore_sdpa = install_sdpa_fallback()
        memory_events.append(cuda_snapshot(torch, int(args.device), "after_builder"))
        measured = _MeasuredPredictor(predictor, torch, int(args.device), memory_events)
        for case in selected_cases:
            torch.cuda.reset_peak_memory_stats(int(args.device))
            print(
                f"Running {case.upper()} as one fixed {LTA_SESSION_FRAMES}-frame session...",
                file=sys.stderr,
                flush=True,
            )
            case_result = run_case(
                measured,
                case=case,
                video_path=videos[case],
                exemplar_image=exemplar_image,
                exemplar_xywh=exemplar_xywh,
                start_frame=int(args.start_frame),
                prompt_frame=int(args.prompt_frame),
                conf=float(args.conf),
                composite_exemplar_visibility=str(args.composite_exemplar_visibility),
                diagnostic_output=diagnostic_output,
            )
            torch.cuda.empty_cache()
            after_case = cuda_snapshot(torch, int(args.device), f"after_{case}")
            memory_events.append(after_case)
            case_result["peak_allocated_mib"] = after_case["peak_allocated_mib"]
            case_results.append(case_result)
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

    accepted = all(bool(case["acceptance"]["passed"]) for case in case_results)
    result = {
        "status": "ok" if accepted else "acceptance_failed",
        "execution_completed": True,
        "diagnostic_schema": "lta.gpu-smoke/1",
        "model": {
            "bundle": str(bundle.root),
            "checkpoint": str(bundle.checkpoint_path),
            "checkpoint_identity_sha256": bundle.checkpoint_identity_sha256,
            "checkpoint_size_bytes": int(bundle.checkpoint_path.stat().st_size),
            "version": bundle.model_version,
        },
        "sam_runtime": sam_runtime,
        "exemplar": {
            "image": str(exemplar_path),
            "label": str(label_path),
            "encoded_index": int(args.exemplar_index),
            "zero_based_label_row": int(args.label_row),
            "prompt_xywh": list(exemplar_xywh),
        },
        "device": {
            "index": int(args.device),
            "name": torch.cuda.get_device_name(int(args.device)),
            "capability": list(torch.cuda.get_device_capability(int(args.device))),
        },
        "settings": {
            "conf": float(args.conf),
            "frame_count": LTA_SESSION_FRAMES,
            "offload_video_to_cpu_requested": True,
            "use_fa3": False,
            "use_rope_real": False,
            "compile": False,
            "warm_up": False,
            "weight_storage": str(args.weight_storage),
            "constrained_batches": constrained_batches,
            "sdpa_backends": ["flash_attention", "efficient_attention", "math"],
            "composite_exemplar_visibility": str(args.composite_exemplar_visibility),
        },
        "cases": case_results,
        "cuda_memory": memory_events,
        "elapsed_seconds": time.perf_counter() - started,
        "writes_outputs": False,
        "diagnostic_output": (
            None if diagnostic_output is None else str(diagnostic_output)
        ),
    }
    if diagnostic_output is not None:
        diagnostic_output.mkdir(parents=True, exist_ok=True)
        result_path = diagnostic_output / "smoke_result.json"
        result["diagnostic_result"] = str(result_path)
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
