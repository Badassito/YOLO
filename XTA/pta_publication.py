"""PTA dataset and image publication primitives shared with render workers.

The module owns filesystem publication, image encoding, and the canonical
frame-carrying dataset sink.  It is independent of :mod:`XTA.pta` and accepts
only the warning-sink protocol owned by :mod:`XTA.pta_dataset`.
"""

from __future__ import annotations

import argparse
import os
import stat as statlib
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

from .pta_augmentation import (
    LoadedAugmentation,
    apply_augmentation_pair,
    assert_augmentation_did_not_synthesize_mask,
)
from .pta_dataset import AUGMENTATION_TAG_LENGTH, OutputCandidate, WarningSink
from .render_batch import RenderBatch as CanonicalRenderBatch, RenderBatchItem
from .unification.contracts import DataRole, FrameAddress, RasterPlan, RenderItem


OUTPUT_IMAGE_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff"}


def parse_output_image_format(value: str) -> str:
    """Validate an output image format and canonicalize aliases."""
    normalized = str(value).strip().lower().lstrip(".")
    if normalized not in OUTPUT_IMAGE_FORMATS:
        supported = ", ".join(sorted(OUTPUT_IMAGE_FORMATS))
        raise argparse.ArgumentTypeError(f"unsupported output image format {value!r}; choose one of: {supported}")
    if normalized == "jpeg":
        return "jpg"
    if normalized == "tiff":
        return "tif"
    return normalized


def output_image_suffix(image_format: str) -> str:
    normalized = parse_output_image_format(image_format)
    return f".{normalized}"

def mask_to_yolo_lines(mask01: np.ndarray, *, warnings: WarningSink, context: str, known_empty: bool = False) -> List[str]:
    if known_empty:
        return []
    m = (np.asarray(mask01, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(m):
        return []
    h, w = m.shape
    contours, hierarchy = cv2.findContours(m * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hinfo = hierarchy[0]
        if np.any(hinfo[:, 3] >= 0):
            warnings.add("holes_discarded_during_polygon_export", context)
    lines: List[str] = []
    hierarchy_arr = [None] * len(contours) if hierarchy is None else list(hierarchy[0])
    for cnt, hrow in zip(contours, hierarchy_arr):
        if hrow is not None and int(hrow[3]) >= 0:
            continue
        if cnt is None or len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        if approx is None or len(approx) < 3:
            warnings.add("degenerate_export_polygon", context)
            continue
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:
            warnings.add("degenerate_export_polygon", context)
            continue
        coords: List[str] = []
        for x, y in pts:
            xn = min(1.0, max(0.0, float(x) / float(max(1, w))))
            yn = min(1.0, max(0.0, float(y) / float(max(1, h))))
            coords.append(f"{xn:.6f}")
            coords.append(f"{yn:.6f}")
        lines.append("0 " + " ".join(coords))
    return lines


_CREATED_OUTPUT_DIRS: set[str] = set()
_CREATED_OUTPUT_DIRS_LOCK = threading.Lock()


def ensure_output_parent_once(path: Path) -> None:
    parent_key = str(path.parent)
    if parent_key in _CREATED_OUTPUT_DIRS:
        return
    with _CREATED_OUTPUT_DIRS_LOCK:
        if parent_key not in _CREATED_OUTPUT_DIRS:
            path.parent.mkdir(parents=True, exist_ok=True)
            _CREATED_OUTPUT_DIRS.add(parent_key)


def _private_image_stage_path(path: Path, family: str) -> Path:
    """Return a hidden same-directory image stage retaining the codec suffix."""

    token = uuid.uuid4().hex
    return path.with_name(
        f".{path.stem}.{str(family)}.{os.getpid()}.{threading.get_ident()}.{token}{path.suffix}"
    )


def _validate_nonempty_regular_file(path: Path, *, context: str) -> int:
    try:
        stat = path.stat()
    except OSError as exc:
        raise RuntimeError(f"{context} did not create a readable file: {path}") from exc
    if not statlib.S_ISREG(stat.st_mode):
        raise RuntimeError(f"{context} output is not a regular file: {path}")
    if int(stat.st_size) <= 0:
        raise RuntimeError(f"{context} produced an empty file: {path}")
    return int(stat.st_size)


def _validate_jpeg_file(path: Path, *, context: str) -> int:
    size = _validate_nonempty_regular_file(path, context=context)
    if size < 4:
        raise RuntimeError(f"{context} produced a truncated JPEG ({size} bytes): {path}")
    with path.open("rb") as handle:
        start = handle.read(2)
        handle.seek(max(0, size - 64))
        tail = handle.read()
    if start != b"\xff\xd8" or b"\xff\xd9" not in tail:
        raise RuntimeError(f"{context} produced an invalid JPEG marker sequence: {path}")
    return size


def _write_nvjpeg_batch_atomically(
    *,
    encoder: object,
    images: Sequence[object],
    final_paths: Sequence[Path],
    params: object,
    cuda_stream: object,
    synchronize_device: Optional[Callable[[], None]] = None,
) -> None:
    """Encode, settle, validate, then publish one nvJPEG batch.

    nvImageCodec 0.9's native file sink can report encoder success even when a
    sink write leaves an empty file. Encode to in-memory CodeStreams, validate
    their buffers, and let Python own the checked same-directory file write.
    """

    finals = tuple(Path(path) for path in final_paths)
    if not finals or len(finals) != len(images):
        raise ValueError(
            f"nvJPEG atomic batch mismatch: images={len(images)}, paths={len(finals)}"
        )
    resolved_finals = [path.resolve(strict=False) for path in finals]
    if len(set(resolved_finals)) != len(resolved_finals):
        raise ValueError("nvJPEG atomic batch contains duplicate destination paths")
    stages = tuple(_private_image_stage_path(path, "nvjpeg") for path in finals)
    for path in finals:
        ensure_output_parent_once(path)
    try:
        raw_results = encoder.encode(  # type: ignore[union-attr]
            list(images),
            codec=".jpg",
            params=params,
            cuda_stream=int(getattr(cuda_stream, "cuda_stream")),
        )
        synchronize = getattr(cuda_stream, "synchronize", None)
        if callable(synchronize):
            synchronize()
        if synchronize_device is not None:
            synchronize_device()
        results = (
            list(raw_results)
            if isinstance(raw_results, (list, tuple))
            else [raw_results]
        )
        if len(results) != len(stages):
            raise RuntimeError(
                f"nvImageCodec returned {len(results)} result(s) for {len(stages)} JPEG file(s)"
            )
        payloads: List[memoryview] = []
        for index, result in enumerate(results):
            if result is None:
                raise RuntimeError(f"nvImageCodec failed JPEG batch index {index}")
            try:
                payload = memoryview(result)
                if payload.format != "B" or payload.ndim != 1:
                    payload = payload.cast("B")
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"nvImageCodec returned a non-buffer CodeStream at JPEG batch index {index}: "
                    f"{type(result).__name__}"
                ) from exc
            declared_size = int(getattr(result, "size", payload.nbytes))
            if declared_size < 0 or declared_size > int(payload.nbytes):
                raise RuntimeError(
                    f"nvImageCodec returned an invalid JPEG CodeStream size at batch index "
                    f"{index}: size={declared_size}, buffer={payload.nbytes}"
                )
            payload = payload[:declared_size]
            if int(payload.nbytes) < 4:
                raise RuntimeError(
                    f"nvImageCodec produced an empty/truncated JPEG CodeStream at batch index "
                    f"{index}: {int(payload.nbytes)} bytes"
                )
            if bytes(payload[:2]) != b"\xff\xd8" or b"\xff\xd9" not in bytes(payload[-64:]):
                raise RuntimeError(
                    f"nvImageCodec produced invalid JPEG markers at batch index {index}"
                )
            payloads.append(payload)
        for index, (payload, stage) in enumerate(zip(payloads, stages)):
            with stage.open("wb") as handle:
                written = handle.write(payload)
                if int(written) != int(payload.nbytes):
                    raise RuntimeError(
                        f"Python JPEG stage write was short at batch index {index}: "
                        f"{written}/{payload.nbytes} bytes"
                    )
            _validate_jpeg_file(stage, context=f"nvJPEG batch index {index}")
        for stage, final in zip(stages, finals):
            os.replace(stage, final)
    finally:
        for stage in stages:
            try:
                stage.unlink(missing_ok=True)
            except OSError:
                pass


def verify_published_image_tree(
    out_dir: Path,
    *,
    expected_count: int,
    image_format: str,
) -> Dict[str, object]:
    """Fail closed before a complete manifest can describe missing/empty images."""

    expected = max(0, int(expected_count))
    image_root = Path(out_dir) / "images"
    expected_suffix = output_image_suffix(str(image_format)).lower()
    if not image_root.exists():
        if expected == 0:
            return {
                "selected": True,
                "verified_image_count": 0,
                "verified_total_bytes": 0,
                "suffix": expected_suffix,
            }
        raise RuntimeError(
            f"PTA image publication is missing its image root: {image_root}; expected={expected}"
        )

    count = 0
    total_bytes = 0
    problems: List[str] = []
    for path in image_root.rglob("*"):
        if path.is_symlink():
            if len(problems) < 12:
                problems.append(f"unexpected_symlink={path}")
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() != expected_suffix:
            if len(problems) < 12:
                problems.append(f"unexpected={path}")
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            if len(problems) < 12:
                problems.append(f"unreadable={path} ({type(exc).__name__}: {exc})")
            continue
        if not statlib.S_ISREG(stat.st_mode) or int(stat.st_size) <= 0:
            if len(problems) < 12:
                problems.append(f"empty_or_irregular={path} size={int(stat.st_size)}")
            continue
        count += 1
        total_bytes += int(stat.st_size)
    if problems or count != expected:
        detail = "; ".join(problems) if problems else "none"
        raise RuntimeError(
            "PTA image publication integrity check failed: "
            f"expected={expected}, verified={count}, suffix={expected_suffix}, problems={detail}"
        )
    return {
        "selected": True,
        "verified_image_count": int(count),
        "verified_total_bytes": int(total_bytes),
        "suffix": expected_suffix,
        "all_files_nonempty": True,
    }


def write_yolo_lines(lines: Sequence[str], path: Path) -> None:
    ensure_output_parent_once(path)
    if lines:
        path.write_text("\n".join(lines) + "\n")
    else:
        path.write_text("")


def write_label_from_mask(mask01: np.ndarray, path: Path, *, warnings: WarningSink, context: str) -> None:
    write_yolo_lines(mask_to_yolo_lines(mask01, warnings=warnings, context=context), path)


def ensure_tiff_output_available() -> None:
    """Require OpenCV's multi-page TIFF writer for custom channel stacks."""
    if not callable(getattr(cv2, "imwritemulti", None)):
        raise RuntimeError(
            "TIFF output for custom C...S... channel stacks requires an OpenCV "
            "build that provides cv2.imwritemulti(); upgrade opencv-python"
        )


def write_image(
    path: Path,
    frame: np.ndarray,
    png_compression: int,
    *,
    channel_kind: str = "gray",
    jpeg_quality: int = 95,
) -> None:
    """Write one uint8 HxW/HxWxC frame using the codec selected by ``path``.

    Custom C...S... arrays use the stock-Ultralytics multispectral TIFF
    representation: one two-dimensional uint8 grayscale TIFF page per image
    channel.  Stock Ultralytics decodes all pages and stacks them into HxWxC.
    """
    ensure_output_parent_once(path)
    suffix = path.suffix.lower()
    kind = str(channel_kind).strip().lower()
    arr = np.ascontiguousarray(np.asarray(frame), dtype=np.uint8)
    if arr.ndim not in (2, 3) or arr.size == 0:
        raise ValueError(f"Output image must be a nonempty HxW or HxWxC array, got shape={arr.shape}")
    if arr.ndim == 3 and int(arr.shape[2]) < 1:
        raise ValueError(f"Output image must contain at least one channel, got shape={arr.shape}")

    if kind == "gray":
        if arr.ndim == 3:
            if int(arr.shape[2]) != 1:
                raise ValueError(f"Gray output requires one channel, got shape={arr.shape}")
            arr = np.ascontiguousarray(arr[:, :, 0])
    elif kind == "rgb":
        if arr.ndim != 3 or int(arr.shape[2]) != 3:
            raise ValueError(f"RGB output requires exactly three channels, got shape={arr.shape}")
    elif kind == "custom":
        if suffix not in {".tif", ".tiff"}:
            raise ValueError("Custom C...S... channel stacks require TIFF output")
        if arr.ndim != 3:
            raise ValueError(f"Custom channel output requires an HxWxC array, got shape={arr.shape}")
    else:
        raise ValueError(f"Unknown channel kind: {channel_kind!r}")

    if suffix == ".png":
        encode_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if kind == "rgb" else arr
        ok = cv2.imwrite(
            str(path),
            encode_arr,
            [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)],
        )
    elif suffix in {".jpg", ".jpeg"}:
        encode_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if kind == "rgb" else arr
        ok = cv2.imwrite(
            str(path),
            encode_arr,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
    elif suffix in {".tif", ".tiff"}:
        if kind == "custom":
            ensure_tiff_output_available()
            pages = [
                np.ascontiguousarray(arr[:, :, channel_index])
                for channel_index in range(int(arr.shape[2]))
            ]
            ok = bool(cv2.imwritemulti(str(path), pages))
        elif kind == "rgb":
            # Conventional single-page RGB TIFF. OpenCV accepts BGR input.
            ok = bool(cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)))
        else:
            gray = arr if arr.ndim == 2 else np.ascontiguousarray(arr[:, :, 0])
            ok = bool(cv2.imwrite(str(path), gray))
    else:
        raise ValueError(f"Unsupported output image extension: {path.suffix or '<none>'}")

    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    if suffix in {".jpg", ".jpeg"}:
        _validate_jpeg_file(path, context="OpenCV JPEG encoder")
    else:
        _validate_nonempty_regular_file(path, context="OpenCV image encoder")


def write_image_gray(path: Path, frame: np.ndarray, png_compression: int) -> None:
    """Single-channel convenience wrapper for render paths."""
    write_image(path, frame, png_compression, channel_kind="gray")

def candidate_output_paths(out_dir: Path, cand: OutputCandidate, *, split_active: bool, image_format: str) -> Tuple[Path, Optional[Path]]:
    subset = cand.split_subset if split_active else None
    img_dir = out_dir / "images" / subset if subset else out_dir / "images"
    lbl_dir = out_dir / "labels" / subset if subset else out_dir / "labels"
    name = f"{cand.volume_name}_{cand.output_tag}_{int(cand.frame_idx) + 1:04d}"
    if int(cand.augmentation_index) > 0:
        tag = str(cand.augmentation_tag or "")
        if len(tag) != AUGMENTATION_TAG_LENGTH or not tag.isalnum() or not tag.isascii():
            raise RuntimeError(f"Invalid internal augmentation tag for {name}: {tag!r}")
        name = f"{name}_{tag}"
    img_path = img_dir / f"{name}{output_image_suffix(image_format)}"
    lbl_path = (lbl_dir / f"{name}.txt") if cand.label_enabled else None
    return img_path, lbl_path


@dataclass(frozen=True)
class PtaDatasetImageSink:
    """PTA image publication consumer for the canonical frame-carrying batch."""

    png_compression: int
    channel_kind: str
    jpeg_quality: int

    def __call__(self, batch: CanonicalRenderBatch) -> None:
        for path_value, item in zip(batch.paths, batch.items):
            if item.synthetic_padding or item.radial_padding:
                continue
            write_image(
                Path(path_value),
                item.frame,
                int(self.png_compression),
                channel_kind=str(self.channel_kind),
                jpeg_quality=int(self.jpeg_quality),
            )


def publish_pta_candidate_image_batch(
    *,
    cand: OutputCandidate,
    image: np.ndarray,
    img_path: Path,
    canonical_plan: RasterPlan,
    png_compression: int,
    jpeg_quality: int,
) -> CanonicalRenderBatch:
    """Build once and deliver the exact dataset image frame to its sink."""

    frame = np.ascontiguousarray(np.asarray(image), dtype=np.uint8)
    request = RenderItem(
        plan=canonical_plan,
        data_role=DataRole.INTENSITY,
        frame_address=FrameAddress(int(cand.frame_idx)),
        metadata={
            "physical_view_id": str(cand.physical_view_id),
            "presentation_variant_id": str(cand.presentation_variant_id),
            "geometry_item_id": str(cand.geometry_item_id),
            "augmentation_index": int(cand.augmentation_index),
            "augmentation_tag": cand.augmentation_tag,
        },
    )
    frames = [frame]
    batch = CanonicalRenderBatch(
        paths=[str(img_path)],
        frames=frames,
        info=[
            f"PTA dataset item {cand.volume_name}/{cand.output_tag}/"
            f"frame={int(cand.frame_idx) + 1:04d}"
        ],
        items=(
            RenderBatchItem(
                result_index=int(cand.frame_idx),
                center_index=int(cand.frame_idx),
                synthetic_padding=False,
                radial_padding=False,
                frame=frame,
                request=request,
            ),
        ),
        raster_plan=canonical_plan,
    )
    PtaDatasetImageSink(
        png_compression=int(png_compression),
        channel_kind=str(cand.channel_kind),
        jpeg_quality=int(jpeg_quality),
    )(batch)
    return batch


def write_selected_candidate_version(
    *,
    cand: OutputCandidate,
    image: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    split_active: bool,
    image_format: str,
    png_compression: int,
    jpeg_quality: int,
    warnings: WarningSink,
    augmentation: Optional[LoadedAugmentation],
    inputs_are_private: bool = False,
    save_images: bool = True,
    save_labels: bool = True,
    canonical_plan: Optional[RasterPlan] = None,
) -> str:
    """Render and write one retained candidate version.

    Returns "written", or "flip_dropped" when a presumed-foreground augmented
    copy rendered background: such copies are dropped rather than
    written so an augmented background can never enter ahead of the
    unique-source background ordering or exceed the cap.
    """
    image_out = image
    mask_out = mask
    if int(cand.augmentation_index) > 0:
        if augmentation is None or cand.augmentation_seed is None or not cand.augmentation_tag:
            raise RuntimeError(f"Internal error: augmented candidate is missing its pipeline, seed, or tag: {cand}")
        replay_context = (
            f"rendering {cand.volume_name}/{cand.parent_view_tag}/{cand.item_key}/"
            f"frame={int(cand.frame_idx)+1:04d}/augmentation_copy={int(cand.augmentation_index)}/"
            f"tag={cand.augmentation_tag}"
        )
        image_out, mask_out = apply_augmentation_pair(
            augmentation,
            image,
            mask,
            seed=int(cand.augmentation_seed),
            context=replay_context,
            copy_inputs=not bool(inputs_are_private),
        )
        # This is unconditional: background classification is skipped when
        # --background_percent=1, but label-preserving augmentations must still
        # never create a mask from a truly empty source.
        assert_augmentation_did_not_synthesize_mask(
            mask,
            mask_out,
            context=replay_context,
            original_known_empty=not bool(cand.foreground),
        )

    img_path, lbl_path = candidate_output_paths(out_dir, cand, split_active=split_active, image_format=image_format)
    label_lines: Optional[List[str]] = None
    if cand.label_enabled and lbl_path is not None:
        label_context = f"{cand.volume_name} {cand.output_tag} frame {int(cand.frame_idx)+1:04d}"
        label_lines = mask_to_yolo_lines(
            mask_out,
            warnings=warnings,
            context=label_context,
            known_empty=not bool(cand.foreground),
        )
        if int(cand.augmentation_index) > 0 and bool(cand.foreground) and not label_lines:
            warnings.add(
                "augmented_foreground_flip_dropped",
                f"{cand.volume_name}/{cand.output_tag}/frame={int(cand.frame_idx)+1:04d}/tag={cand.augmentation_tag}",
            )
            return "flip_dropped"
    if bool(save_images):
        if canonical_plan is None:
            write_image(
                img_path,
                image_out,
                int(png_compression),
                channel_kind=str(cand.channel_kind),
                jpeg_quality=int(jpeg_quality),
            )
        else:
            publish_pta_candidate_image_batch(
                cand=cand,
                image=image_out,
                img_path=img_path,
                canonical_plan=canonical_plan,
                png_compression=int(png_compression),
                jpeg_quality=int(jpeg_quality),
            )
    if bool(save_labels) and label_lines is not None and lbl_path is not None:
        write_yolo_lines(label_lines, lbl_path)
    return "written"

__all__ = [
    "OUTPUT_IMAGE_FORMATS",
    "PtaDatasetImageSink",
    "candidate_output_paths",
    "ensure_output_parent_once",
    "ensure_tiff_output_available",
    "mask_to_yolo_lines",
    "output_image_suffix",
    "parse_output_image_format",
    "publish_pta_candidate_image_batch",
    "verify_published_image_tree",
    "write_image",
    "write_image_gray",
    "write_label_from_mask",
    "write_selected_candidate_version",
    "write_yolo_lines",
]
