#!/usr/bin/env python3
"""yolo_tta_seg.py

Test-time augmentation (TTA) for Ultralytics YOLO segmentation on videos.

Key features
- Rotation (including non-90°) with output clamped to the original frame size (cropped corners, black fill).
- Scaling to --imgsz and pixel shifts (center, left/right/up/down) with clamped output size.
- Sequential inference over all augmented videos (optionally multiple models for ensembling).
- Inverse-transform masks back to the original orientation; union masks per-frame across all augments/models.
- Fill fully-enclosed holes in the union mask (donut -> filled).
- Optional temporal refinement using optical flow to interpolate/extrapolate missing detections.
- Final overlay video (blue mask @ 50% alpha) encoded as FFV1 in MKV.
- Optional saving of YOLO-seg TXT labels and a binary TIFF stack.

Requirements
- ffmpeg + ffprobe on PATH
- Ultralytics (pip install ultralytics)
- OpenCV (pip install opencv-python)

Example
  python yolo_tta_seg.py --source input.mp4 --model yolov8n-seg.pt --output out.mkv \
    --imgsz 1024 --shift 16 --conf 0.10 --temporal 2 --save_labels --save_binary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# CLI parsing
# -----------------------------

def _parse_csv_ints(s: str) -> List[int]:
    # Accept: "0,45,90" or "0 45 90" (argparse may pass joined strings)
    s = s.strip()
    if not s:
        return []
    parts: List[str] = []
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend(chunk.split())
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        out.append(int(float(p)))
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TTA script for YOLO segmentation on a video (rot/scale/shift + mask union).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--source", required=True, type=Path, help="Input video path")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MKV path (FFV1). Default: <source_stem>_tta.mkv next to source.",
    )

    # Allow multiple --model, and also allow space-separated list after one --model.
    p.add_argument(
        "--model",
        required=True,
        action="append",
        nargs="+",
        type=Path,
        help="Model path(s). Repeat --model for multiple models to ensemble.",
    )

    p.add_argument(
        "--angle",
        nargs="*",
        default=None,
        help=(
            "Rotation angles in degrees. Accepts either space-separated values "
            "(e.g. --angle 0 45 90) or comma-separated (e.g. --angle 0,45,90). "
            "If omitted, uses 0,45,90,135,180,225,270,315."
        ),
    )
    p.add_argument(
        "--shift",
        type=int,
        default=0,
        help="Pixel shift magnitude applied after scaling. 0 disables shifts (only center).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Square inference size. Also used to scale the augmented videos.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Confidence for YOLO predict.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device string passed to YOLO predict (e.g. '0', '0,1', 'cpu').",
    )

    p.add_argument("--half", action="store_true", help="Enable half precision during prediction")
    p.add_argument("--int8", action="store_true", help="Enable int8 during prediction")

    p.add_argument(
        "--save_labels",
        action="store_true",
        help="Save final flattened labels (YOLO-seg TXT). Blank TXT for no detections.",
    )
    p.add_argument(
        "--save_binary",
        action="store_true",
        help="Save a binary mask TIFF stack (white mask, black background).",
    )
    p.add_argument(
        "--no_temp",
        action="store_true",
        help="Clean up temporary files (temp videos, intermediates). Temp files are kept by default.",
    )

    p.add_argument(
        "--temporal",
        type=int,
        default=0,
        help="Temporal refinement window N (check +-N frames). 0 disables temporal refinement.",
    )

    return p.parse_args(argv)


# -----------------------------
# ffprobe / ffmpeg helpers
# -----------------------------


def _run(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    nb_frames: Optional[int]


def _parse_fps(fr: str) -> float:
    fr = fr.strip()
    if not fr or fr == "0/0":
        return 0.0
    try:
        return float(Fraction(fr))
    except Exception:
        try:
            return float(fr)
        except Exception:
            return 0.0


def ffprobe_info(video_path: Path) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    cp = _run(cmd)
    data = json.loads(cp.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream: {video_path}")
    s0 = streams[0]
    w = int(s0["width"])
    h = int(s0["height"])
    fps = _parse_fps(s0.get("avg_frame_rate", "")) or _parse_fps(s0.get("r_frame_rate", ""))
    nb = s0.get("nb_frames", None)
    nb_frames = int(nb) if nb not in (None, "N/A") else None
    if fps <= 0:
        # conservative fallback
        fps = 30.0
    return VideoInfo(width=w, height=h, fps=fps, nb_frames=nb_frames)


def has_encoder(encoder_name: str) -> bool:
    # Cache encoder list once per process (ffmpeg -encoders can be slow).
    if not hasattr(has_encoder, "_cache"):
        setattr(has_encoder, "_cache", {})
    cache: Dict[str, bool] = getattr(has_encoder, "_cache")
    if encoder_name in cache:
        return cache[encoder_name]
    try:
        cp = _run(["ffmpeg", "-hide_banner", "-encoders"], check=True)
        cache[encoder_name] = (encoder_name in cp.stdout)
    except Exception:
        cache[encoder_name] = False
    return cache[encoder_name]


def make_augmented_video(
    *,
    src: Path,
    dst: Path,
    angle_deg: float,
    dx: int,
    dy: int,
    imgsz: int,
    prefer_nvenc: bool = True,
) -> None:
    """Create an augmented video file using ffmpeg.

    Order: rotate (clamped to src size) -> scale to imgsz -> shift (clamped) -> encode.

    Rotation direction:
      ffmpeg rotate uses positive angles as CLOCKWISE in practice (empirically).
      We keep ffmpeg's convention here; inverse mapping code compensates.
    """

    # Rotate with clamped output size (crop corners, fill black)
    rot = f"rotate={angle_deg}*PI/180:ow=iw:oh=ih:c=black"

    # Scale to square imgsz
    scl = f"scale={imgsz}:{imgsz}"

    vf_parts = [rot, scl]

    if dx != 0 or dy != 0:
        absx = abs(dx)
        absy = abs(dy)
        # pad larger, place image offset, then crop back to original (post-scale) size
        # crop size = (iw - 2*absx, ih - 2*absy)
        pad = f"pad=iw+{2*absx}:ih+{2*absy}:{absx+dx}:{absy+dy}:color=black"
        crop = f"crop=iw-{2*absx}:ih-{2*absy}:{absx}:{absy}"
        vf_parts.extend([pad, crop])

    vf = ",".join(vf_parts)

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Prefer NVENC lossless if available
    use_nvenc = prefer_nvenc and has_encoder("h264_nvenc")
    if use_nvenc:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-an",
            "-vf",
            vf,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "lossless",
            "-rc",
            "constqp",
            "-qp",
            "0",
            str(dst),
        ]
        try:
            _run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            # fall back to CPU encoder
            sys.stderr.write(
                "[WARN] h264_nvenc failed; falling back to libx264 lossless.\n"
                + e.stderr
                + "\n"
            )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "0",
        str(dst),
    ]
    _run(cmd, check=True)


# -----------------------------
# Geometry: forward + inverse transforms
# -----------------------------


def affine_original_to_aug(
    *,
    w0: int,
    h0: int,
    imgsz: int,
    angle_deg: float,
    dx: int,
    dy: int,
) -> np.ndarray:
    """Return 2x3 affine transform mapping ORIGINAL -> AUGMENTED.

    This must match make_augmented_video() order.

    Notes:
    - ffmpeg rotate uses positive degrees as clockwise (empirically),
      while OpenCV getRotationMatrix2D uses positive as counter-clockwise.
      Therefore we negate angle_deg when building the OpenCV matrix.
    - Rotation is applied at original size, clamped to original size.
    - Then scale to imgsz x imgsz.
    - Then shift (dx, dy) in the scaled coordinate system.
    """

    # 1) Rotate at original size around original center.
    c0 = (w0 / 2.0, h0 / 2.0)
    M_rot = cv2.getRotationMatrix2D(c0, -float(angle_deg), 1.0)  # NOTE sign

    # 2) Scale to square imgsz.
    sx = float(imgsz) / float(w0)
    sy = float(imgsz) / float(h0)
    A_scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    # 3) Shift after scaling.
    A_shift = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Homogeneous compose: A = A_shift * A_scale * A_rot
    A_rot = np.vstack([M_rot.astype(np.float32), [0.0, 0.0, 1.0]])
    A = A_shift @ A_scale @ A_rot
    return A[:2, :]


def invert_affine(M: np.ndarray) -> np.ndarray:
    if M.shape != (2, 3):
        raise ValueError("Expected 2x3 affine")
    return cv2.invertAffineTransform(M)


# -----------------------------
# Mask post-processing
# -----------------------------


def fill_holes(mask01: np.ndarray) -> np.ndarray:
    """Fill fully enclosed holes in a binary mask.

    mask01: uint8 or bool, values 0/1.
    Returns uint8 0/1.
    """
    m = (mask01.astype(np.uint8) * 255)
    inv = cv2.bitwise_not(m)  # outside + holes are 255

    h, w = inv.shape
    flood = inv.copy()
    ffmask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ffmask, seedPoint=(0, 0), newVal=0)
    holes = flood  # holes remain 255
    filled = cv2.bitwise_or(m, holes)
    return (filled > 0).astype(np.uint8)


def mask_to_yolo_seg_line(mask01: np.ndarray) -> Optional[str]:
    """Convert a binary mask to a single YOLO-seg polygon line.

    Strategy:
    - External contours only.
    - Keep the largest contour by area.
    - Approximate with approxPolyDP for size.

    Returns None if mask is empty or contour is degenerate.
    """
    h, w = mask01.shape
    m = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # largest contour
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 1.0:
        return None

    peri = cv2.arcLength(c, True)
    eps = max(1.0, 0.002 * peri)
    approx = cv2.approxPolyDP(c, eps, True)
    if approx is None or len(approx) < 3:
        return None

    pts = approx.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / float(w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(h), 0.0, 1.0)

    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
    return f"0 {coords}"  # single class


# -----------------------------
# Optical flow temporal refinement
# -----------------------------


class FlowCache:
    def __init__(self, frames_gray: List[np.ndarray]):
        self.frames = frames_gray
        self.cache: Dict[Tuple[int, int], np.ndarray] = {}

    def flow(self, i: int, j: int) -> np.ndarray:
        """Optical flow from frame i -> frame j, where |i-j| == 1."""
        if abs(i - j) != 1:
            raise ValueError("FlowCache only supports adjacent frames")
        key = (i, j)
        if key in self.cache:
            return self.cache[key]

        prev = self.frames[i]
        nxt = self.frames[j]

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        self.cache[key] = flow
        return flow


def warp_mask_with_flow(mask01: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp a binary mask forward using optical flow (prev -> next).

    mask01: uint8 0/1
    flow: (h,w,2) float32 from prev->next
    Returns uint8 0/1 mask in next frame coordinates.
    """
    h, w = mask01.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs - flow[..., 0].astype(np.float32)
    map_y = ys - flow[..., 1].astype(np.float32)

    src = (mask01.astype(np.uint8) * 255)
    warped = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped > 0).astype(np.uint8)


def warp_chain(mask01: np.ndarray, flow_cache: FlowCache, start: int, end: int) -> np.ndarray:
    """Warp mask from frame `start` to frame `end` using chained adjacent flows."""
    m = mask01.copy()
    if start == end:
        return m
    if start < end:
        for k in range(start, end):
            f = flow_cache.flow(k, k + 1)
            m = warp_mask_with_flow(m, f)
    else:
        for k in range(start, end, -1):
            f = flow_cache.flow(k, k - 1)
            m = warp_mask_with_flow(m, f)
    return m


def temporal_refine(
    masks01: List[np.ndarray],
    confs: List[float],
    frames_gray: List[np.ndarray],
    *,
    base_conf: float,
    window: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """Temporal refinement using a simple optical-flow strategy.

    Rules implemented (per prompt):
    - High confidence: conf >= 2*base_conf
    - If a detection is low confidence and has NO high-conf neighbors within +-window: drop it.
    - If a frame is missing detections but has high-conf neighbors within +-window: interpolate using optical flow.
    - If only one side has high-conf neighbor, extrapolate if that side is part of a high-conf series.
    """

    if window <= 0:
        return masks01, confs

    hi_thr = 2.0 * float(base_conf)
    n = len(masks01)
    if len(confs) != n or len(frames_gray) != n:
        raise ValueError("temporal_refine: mismatched lengths")

    # High-confidence indicator (mask must be non-empty)
    high = np.array([(confs[i] >= hi_thr) and bool(masks01[i].any()) for i in range(n)], dtype=bool)

    # 1) Drop isolated low-confidence detections.
    for i in range(n):
        if masks01[i].any() and confs[i] < hi_thr:
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            has_high_neighbor = any(high[j] for j in range(lo, hi) if j != i)
            if not has_high_neighbor:
                masks01[i][:] = 0
                confs[i] = 0.0

    # 2) Fill gaps using optical flow from nearby high-confidence frames.
    flow_cache = FlowCache(frames_gray)

    for i in range(n):
        if masks01[i].any():
            continue

        left = None
        for j in range(i - 1, max(-1, i - window - 1), -1):
            if high[j]:
                left = j
                break

        right = None
        for j in range(i + 1, min(n, i + window + 1)):
            if high[j]:
                right = j
                break

        candidates: List[np.ndarray] = []

        if left is not None and right is not None:
            candidates.append(warp_chain(masks01[left], flow_cache, left, i))
            candidates.append(warp_chain(masks01[right], flow_cache, right, i))

        elif left is not None:
            # Extrapolate if left is part of a high-confidence series
            if left - 1 >= 0 and high[left - 1]:
                flow = flow_cache.flow(left - 1, left)
                m = masks01[left].copy()
                for _ in range(i - left):
                    m = warp_mask_with_flow(m, flow)
                candidates.append(m)

        elif right is not None:
            # Extrapolate backward if right is part of a high-confidence series
            if right + 1 < n and high[right + 1]:
                flow = flow_cache.flow(right + 1, right)
                m = masks01[right].copy()
                for _ in range(right - i):
                    m = warp_mask_with_flow(m, flow)
                candidates.append(m)

        if candidates:
            m = candidates[0]
            for c in candidates[1:]:
                m = np.logical_or(m, c)
            masks01[i] = m.astype(np.uint8)
            confs[i] = hi_thr

    return masks01, confs


# -----------------------------
# Output writers
# -----------------------------


def overlay_blue(frame_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay blue mask with given alpha (expects BGR frame)."""
    out = frame_bgr.copy()
    m = mask01.astype(bool)
    if not m.any():
        return out
    blue = np.array([255, 0, 0], dtype=np.float32)  # BGR blue
    # Blend only masked pixels
    out_f = out.astype(np.float32)
    out_f[m] = out_f[m] * (1.0 - alpha) + blue * alpha
    return out_f.astype(np.uint8)


def write_ffv1_mkv(
    *,
    frames_bgr: Iterable[np.ndarray],
    width: int,
    height: int,
    fps: float,
    output_path: Path,
) -> None:
    """Encode frames to FFV1 MKV via ffmpeg stdin."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    try:
        for fr in frames_bgr:
            if fr.shape[0] != height or fr.shape[1] != width or fr.shape[2] != 3:
                raise ValueError("Frame has unexpected shape; expected HxWx3 BGR")
            proc.stdin.write(fr.tobytes())
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg FFV1 encode failed with exit code {ret}")


# -----------------------------
# Main pipeline
# -----------------------------


@dataclass(frozen=True)
class AugSpec:
    angle: int
    dx: int
    dy: int
    video_path: Path
    M_orig_to_aug: np.ndarray  # 2x3
    M_aug_to_orig: np.ndarray  # 2x3


def load_models(model_paths: List[Path]):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is not installed. Install with: pip install ultralytics\n"
            f"Import error: {e}"
        )

    models = []
    for p in model_paths:
        models.append(YOLO(str(p)))
    return models


def build_aug_specs(
    *,
    src: Path,
    work_dir: Path,
    angles: List[int],
    shift_px: int,
    imgsz: int,
    w0: int,
    h0: int,
) -> List[AugSpec]:
    shifts: List[Tuple[int, int]]
    if shift_px == 0:
        shifts = [(0, 0)]
    else:
        n = int(shift_px)
        shifts = [(0, 0), (n, 0), (-n, 0), (0, -n), (0, n)]  # right, left, up, down

    specs: List[AugSpec] = []
    for a in angles:
        for dx, dy in shifts:
            name = f"a{a:03d}_dx{dx:+05d}_dy{dy:+05d}.mp4"
            vp = work_dir / "aug_videos" / name
            M = affine_original_to_aug(w0=w0, h0=h0, imgsz=imgsz, angle_deg=a, dx=dx, dy=dy)
            Minv = invert_affine(M)
            specs.append(AugSpec(angle=a, dx=dx, dy=dy, video_path=vp, M_orig_to_aug=M, M_aug_to_orig=Minv))
    return specs


def union_masks_from_result(result) -> Tuple[Optional[np.ndarray], float]:
    """Return (mask01_union, max_conf) for a single Ultralytics Results frame."""
    # result.masks can be None
    if getattr(result, "masks", None) is None:
        return None, 0.0

    masks = result.masks.data
    # to numpy
    try:
        masks_np = masks.detach().float().cpu().numpy()
    except Exception:
        masks_np = np.asarray(masks)

    if masks_np.size == 0:
        return None, 0.0

    # Union
    union = (masks_np > 0.5).any(axis=0).astype(np.uint8)

    # Confidence: take max over boxes, if present
    max_conf = 0.0
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        try:
            conf_np = boxes.conf.detach().float().cpu().numpy()
            if conf_np.size:
                max_conf = float(conf_np.max())
        except Exception:
            pass

    return union, max_conf


def run_inference_and_accumulate(
    *,
    models,
    specs: List[AugSpec],
    src_info: VideoInfo,
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], List[float]]:
    """Run sequential inference over all augmented videos and union masks in original coords."""

    w0, h0 = src_info.width, src_info.height

    masks01: List[np.ndarray] = []
    confs: List[float] = []

    # Iterate over augmented videos
    for spec in tqdm(specs, desc="TTA variants", unit="variant"):
        if not spec.video_path.exists():
            make_augmented_video(
                src=args.source,
                dst=spec.video_path,
                angle_deg=spec.angle,
                dx=spec.dx,
                dy=spec.dy,
                imgsz=args.imgsz,
                prefer_nvenc=True,
            )

        for model in models:
            preds = model.predict(
                source=str(spec.video_path),
                save=False,
                stream=True,
                conf=float(args.conf),
                iou=1.0,
                imgsz=int(args.imgsz),
                retina_masks=True,
                device=args.device,
                half=bool(args.half),
                int8=bool(args.int8),
                verbose=False,
            )

            for idx, res in enumerate(preds):
                if idx >= len(masks01):
                    masks01.append(np.zeros((h0, w0), dtype=np.uint8))
                    confs.append(0.0)

                union_aug, max_conf = union_masks_from_result(res)
                if union_aug is None:
                    # keep max conf anyway
                    if max_conf > confs[idx]:
                        confs[idx] = max_conf
                    continue

                # Map augmented mask -> original coords
                # union_aug is imgsz x imgsz, we warp to w0 x h0
                mask_orig = cv2.warpAffine(
                    union_aug,
                    spec.M_aug_to_orig,
                    (w0, h0),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                masks01[idx] = np.logical_or(masks01[idx], mask_orig > 0).astype(np.uint8)
                if max_conf > confs[idx]:
                    confs[idx] = max_conf

    return masks01, confs


def read_video_frames(source: Path) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_video_frames_gray_list(source: Path) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for fr in tqdm(read_video_frames(source), desc="Loading frames for optical flow", unit="frame"):
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    return frames


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    args.source = args.source.expanduser().resolve()
    if args.output is None:
        args.output = args.source.with_name(f"{args.source.stem}_tta.mkv")
    args.output = args.output.expanduser().resolve()

    # Flatten model args (since argparse used action=append,nargs=+)
    model_paths = [p.expanduser().resolve() for group in args.model for p in group]

    if args.angle is None or len(args.angle) == 0:
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
    else:
        angles = []
        for token in args.angle:
            angles.extend(_parse_csv_ints(str(token)))
    if not angles:
        raise ValueError("--angle parsed to an empty list")

    # Video info
    src_info = ffprobe_info(args.source)
    w0, h0 = src_info.width, src_info.height

    # Work dir
    work_dir = args.output.parent / f"{args.output.stem}__tta_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Persist a small manifest for reproducibility
    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "models": [str(p) for p in model_paths],
        "angles": angles,
        "shift": int(args.shift),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "device": str(args.device),
        "half": bool(args.half),
        "int8": bool(args.int8),
        "temporal": int(args.temporal),
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Build augmentation specs
    specs = build_aug_specs(
        src=args.source,
        work_dir=work_dir,
        angles=angles,
        shift_px=int(args.shift),
        imgsz=int(args.imgsz),
        w0=w0,
        h0=h0,
    )

    # Load YOLO models
    models = load_models(model_paths)

    # Inference + accumulation
    masks01, confs = run_inference_and_accumulate(models=models, specs=specs, src_info=src_info, args=args)

    # Fill holes (donut -> filled)
    masks01 = [fill_holes(m) for m in tqdm(masks01, desc="Filling holes", unit="frame")]

    # Temporal refinement
    if int(args.temporal) > 0:
        frames_gray = read_video_frames_gray_list(args.source)
        n = min(len(frames_gray), len(masks01))
        masks01 = masks01[:n]
        confs = confs[:n]
        frames_gray = frames_gray[:n]

        masks01, confs = temporal_refine(
            masks01,
            confs,
            frames_gray,
            base_conf=float(args.conf),
            window=int(args.temporal),
        )

        # Fill holes again in case flow warping created enclosed gaps
        masks01 = [fill_holes(m) for m in tqdm(masks01, desc="Filling holes (post-temporal)", unit="frame")]

    # Save binary TIFF stack
    if args.save_binary:
        import tifffile

        tiff_path = args.output.with_suffix("")
        tiff_path = tiff_path.with_name(f"{tiff_path.name}_mask.tif")

        with tifffile.TiffWriter(str(tiff_path), bigtiff=True) as tw:
            for m in tqdm(masks01, desc="Writing TIFF stack", unit="frame"):
                tw.write((m.astype(np.uint8) * 255), photometric="minisblack")

    # Save YOLO labels
    if args.save_labels:
        label_dir = args.output.with_suffix("")
        label_dir = label_dir.with_name(f"{label_dir.name}_labels")
        label_dir.mkdir(parents=True, exist_ok=True)

        pad = max(6, len(str(len(masks01))))
        for i, m in tqdm(list(enumerate(masks01)), desc="Writing labels", unit="frame"):
            txt = label_dir / f"{i:0{pad}d}.txt"
            line = mask_to_yolo_seg_line(m)
            if line is None:
                txt.write_text("")
            else:
                txt.write_text(line + "\n")

    # Write final overlay video
    def overlay_frames() -> Iterable[np.ndarray]:
        for i, fr in enumerate(read_video_frames(args.source)):
            if i >= len(masks01):
                break
            yield overlay_blue(fr, masks01[i], alpha=0.5)

    write_ffv1_mkv(
        frames_bgr=tqdm(overlay_frames(), desc="Writing FFV1 MKV", unit="frame"),
        width=w0,
        height=h0,
        fps=src_info.fps,
        output_path=args.output,
    )

    # Cleanup
    if args.no_temp:
        try:
            shutil.rmtree(work_dir)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to remove temp dir {work_dir}: {e}\n")

    print(f"Done. Output video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
