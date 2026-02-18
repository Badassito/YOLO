
#!/usr/bin/env python3
"""
YOLO Segmentation TTA for a single input video.

Spec implemented:
1) Rotate video by --angle list (default "0,45,90,135,180,225,270,315"),
   including non-90° rotations with clamped output size and black fill (crop corners).
2) Scale to imgsz x imgsz. Then shift by N pixels in 4 directions + none (5 variants/angle),
   or 1 if shift=0. Clamp-to-size with black fill. Save each augmented video to output/temp
   using h264_nvenc -preset lossless (fallback to libx264 CRF 0 if NVENC unavailable).
3) Run inference sequentially per augmented video (and per model if ensemble):
   task=segment, save=False, stream=True, iou=1.0, retina_masks=True, conf=--conf, imgsz=--imgsz.
4) Undo shifts/scale/rotations back to original orientation, union masks per frame (1 class).
5) Fill fully enclosed holes in the final per-frame union mask.
6) Output final overlay video (original + 50% transparent blue mask) as FFV1 in MKV.

Additional flags:
- --save_labels: write YOLO segmentation labels (blank .txt when no detections).
- --save_binary: write binary mask TIFF sequence + binary mask FFV1 MKV.
- --keep_temp: keep temp augmented videos.
- --int8 / --half: passed to YOLO predict.
- --model can be repeated to ensemble by unioning results.
- --temporal N (default 1): temporal stabilization via optical flow mask warping.

Requires:
  pip install ultralytics opencv-python numpy
  ffmpeg in PATH
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class AugVariant:
    angle_deg: float
    dx: int
    dy: int
    path: Path


# ------------------------ CLI parsing helpers ------------------------


def parse_angles(s: str) -> List[float]:
    # Accept "0,45,90" and/or whitespace-separated values.
    parts: List[str] = []
    for chunk in str(s).split():
        parts.extend([p.strip() for p in chunk.split(",") if p.strip()])
    return [float(p) for p in parts]


def normalize_models(models: List[str]) -> List[str]:
    # Accept: --model m1 m2
    # Also:  --model "m1,m2"
    out: List[str] = []
    for m in models:
        out.extend([x.strip() for x in m.split(",") if x.strip()])
    # de-dupe
    seen = set()
    uniq: List[str] = []
    for m in out:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


# ------------------------ ffmpeg helpers ------------------------


def ffmpeg_exists() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def ffmpeg_has_encoder(enc: str) -> bool:
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        return enc in p.stdout
    except Exception:
        return False


class FFmpegWriter:
    def __init__(self, cmd: List[str]) -> None:
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if self.proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin pipe.")

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            raise ValueError("FFmpegWriter expects uint8 frames.")
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            raise RuntimeError("ffmpeg pipe broke (check ffmpeg stderr/loglevel).") from e

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        finally:
            ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}: {' '.join(self.cmd)}")


def start_temp_h264_writer(out_path: Path, w: int, h: int, fps: float, has_nvenc: bool) -> FFmpegWriter:
    # Raw BGR24 in -> H.264 lossless temp video.
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
    ]
    if has_nvenc:
        cmd = base + ["-c:v", "h264_nvenc", "-preset", "lossless", str(out_path)]
    else:
        # fallback if NVENC isn't present
        cmd = base + ["-c:v", "libx264", "-crf", "0", "-preset", "veryfast", str(out_path)]
    return FFmpegWriter(cmd)


def start_ffv1_writer(out_path: Path, w: int, h: int, fps: float, pix_in: str, pix_out: str) -> FFmpegWriter:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_in,
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        pix_out,
        str(out_path),
    ]
    return FFmpegWriter(cmd)


# ------------------------ video / geometry helpers ------------------------


def probe_video(path: Path) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        fps = 30.0
    if n <= 0:
        # fallback count
        cap = cv2.VideoCapture(str(path))
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
    return w, h, fps, n


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Clamp-to-size integer shift with black fill. dx>0 shifts content right; dy>0 shifts content down."""
    if dx == 0 and dy == 0:
        return img
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    if dx >= 0:
        src_x0, src_x1 = 0, w - dx
        dst_x0, dst_x1 = dx, w
    else:
        src_x0, src_x1 = -dx, w
        dst_x0, dst_x1 = 0, w + dx

    if dy >= 0:
        src_y0, src_y1 = 0, h - dy
        dst_y0, dst_y1 = dy, h
    else:
        src_y0, src_y1 = -dy, h
        dst_y0, dst_y1 = 0, h + dy

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def fill_holes(mask_bool: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Fill fully enclosed holes in a binary mask."""
    if mask_bool is None:
        return None
    mask_bool = mask_bool.astype(bool)
    if not mask_bool.any():
        return mask_bool

    mask = (mask_bool.astype(np.uint8) * 255)
    # Pad so (0,0) is guaranteed background.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, flood_inv)[1:-1, 1:-1]
    return (filled > 0)


def mask_to_yolo_seg_lines(mask_bool: Optional[np.ndarray]) -> List[str]:
    """YOLO segmentation label lines: cls x1 y1 x2 y2 ... (normalized)."""
    if mask_bool is None or not mask_bool.any():
        return []
    h, w = mask_bool.shape[:2]
    mask = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: List[str] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 3:
            continue
        pts[:, 0] = np.clip(pts[:, 0] / float(w), 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / float(h), 0.0, 1.0)
        flat = pts.reshape(-1)
        if flat.size < 6:
            continue
        coords = " ".join(f"{v:.6f}" for v in flat.tolist())
        lines.append(f"0 {coords}")
    return lines


# ------------------------ TTA generation ------------------------


def generate_augmented_videos(
    input_video: Path,
    base: str,
    temp_dir: Path,
    angles: List[float],
    shift: int,
    imgsz: int,
    fps: float,
    has_nvenc: bool,
) -> List[AugVariant]:
    temp_dir.mkdir(parents=True, exist_ok=True)

    shifts: List[Tuple[int, int]] = [(0, 0)] if shift == 0 else [(0, 0), (shift, 0), (-shift, 0), (0, shift), (0, -shift)]
    variants: List[AugVariant] = []

    for angle in angles:
        writers: List[FFmpegWriter] = []
        for dx, dy in shifts:
            tag = f"A{angle:.1f}_DX{dx}_DY{dy}".replace(".", "p")
            out_path = temp_dir / f"{base}_{tag}.mkv"
            writers.append(start_temp_h264_writer(out_path, imgsz, imgsz, fps, has_nvenc=has_nvenc))
            variants.append(AugVariant(angle_deg=angle, dx=dx, dy=dy, path=out_path))

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open input video: {input_video}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        center = (w / 2.0, h / 2.0)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # rotate, clamp to original size, black fill
            rotated = cv2.warpAffine(
                frame,
                M_rot,
                dsize=(w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            # scale to imgsz x imgsz
            scaled = cv2.resize(rotated, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

            # shift variants
            for (dx, dy), wr in zip(shifts, writers):
                wr.write(shift_image(scaled, dx=dx, dy=dy))

        cap.release()
        for wr in writers:
            wr.close()

    return variants


# ------------------------ YOLO result handling ------------------------


def extract_union_mask_and_conf(result) -> Tuple[Optional[np.ndarray], float]:
    """Union all instance masks in a frame; return (mask_bool or None, max_conf)."""
    max_conf = 0.0
    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
        try:
            conf_t = result.boxes.conf
            if conf_t is not None and len(conf_t) > 0:
                max_conf = float(conf_t.max().item()) if hasattr(conf_t, "max") else float(np.max(conf_t))
        except Exception:
            pass

    if getattr(result, "masks", None) is None or getattr(result.masks, "data", None) is None:
        return None, max_conf

    masks = result.masks.data
    try:
        if hasattr(masks, "detach"):
            m = masks.detach().float().cpu().numpy()
        else:
            m = np.asarray(masks)
    except Exception:
        m = np.asarray(masks)

    if m.ndim != 3 or m.shape[0] == 0:
        return None, max_conf

    mask = (m > 0.5).any(axis=0)
    if not mask.any():
        return None, max_conf
    return mask.astype(bool), max_conf


def undo_tta_mask_to_original(mask_aug: np.ndarray, angle_deg: float, dx: int, dy: int, orig_w: int, orig_h: int) -> np.ndarray:
    # undo shift in imgsz-space (mask_aug already in imgsz x imgsz)
    m = (mask_aug.astype(np.uint8) * 255)
    unshift = shift_image(m, dx=-dx, dy=-dy)

    # undo scale back to original frame size
    resized = cv2.resize(unshift, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # undo rotation
    center = (orig_w / 2.0, orig_h / 2.0)
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M_inv = cv2.invertAffineTransform(M_rot)
    unrot = cv2.warpAffine(
        resized,
        M_inv,
        dsize=(orig_w, orig_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (unrot > 0)


def safe_yolo_predict(model, **kwargs):
    """
    Call model.predict(**kwargs) but auto-strip unsupported kwargs based on TypeError message.
    This keeps the script working across Ultralytics versions while still attempting to pass
    task=segment, retina_masks=True, etc.
    """
    k = dict(kwargs)
    while True:
        try:
            return model.predict(**k)
        except TypeError as e:
            msg = str(e)
            m = re.search(r"(?:got an unexpected keyword argument|unexpected keyword argument) '([^']+)'", msg)
            if m:
                bad = m.group(1)
                if bad in k:
                    k.pop(bad)
                    continue
            raise


# ------------------------ temporal stabilization ------------------------


def apply_temporal_optical_flow(
    input_video: Path,
    union_masks: List[Optional[np.ndarray]],
    max_confs: List[float],
    conf_th: float,
    temporal_n: int,
) -> Tuple[List[Optional[np.ndarray]], List[float]]:
    """
    Temporal logic (per spec):
    - High confidence: conf >= 2*--conf
    - Low confidence: conf < 2*--conf
    - High conf detections never dropped.
    - Low conf detections dropped if no neighbors within +-N contain any detection.
    - If a frame is missing detections and all neighbors within +-N are high, interpolate via optical flow warping.
    - If missing right after/before a run of high detections, extrapolate 1 frame via optical flow warping.
    """
    if temporal_n <= 0:
        return union_masks, max_confs

    n = len(union_masks)
    has = [m is not None and bool(m.any()) for m in union_masks]
    high = [(has[i] and (max_confs[i] >= 2.0 * conf_th)) for i in range(n)]

    # Drop isolated low-confidence predictions.
    for i in range(n):
        if not has[i] or high[i]:
            continue
        lo, hi = max(0, i - temporal_n), min(n - 1, i + temporal_n)
        if not any(has[j] for j in range(lo, hi + 1) if j != i):
            union_masks[i] = None
            max_confs[i] = 0.0
            has[i] = False

    has = [m is not None and bool(m.any()) for m in union_masks]
    high = [(has[i] and (max_confs[i] >= 2.0 * conf_th)) for i in range(n)]

    need_interp = [False] * n
    need_fwd = [False] * n
    need_bwd = [False] * n

    for i in range(n):
        if has[i]:
            continue

        # interpolation if full window exists and all neighbors are high
        if i - temporal_n >= 0 and i + temporal_n < n:
            ok = True
            for j in range(i - temporal_n, i + temporal_n + 1):
                if j == i:
                    continue
                if not high[j]:
                    ok = False
                    break
            if ok:
                need_interp[i] = True
                continue

        # forward extrapolation if previous N frames are all high
        if i - temporal_n >= 0:
            if all(high[j] for j in range(i - temporal_n, i)):
                need_fwd[i] = True

        # backward extrapolation if next N frames are all high
        if i + temporal_n < n:
            if all(high[j] for j in range(i + 1, i + temporal_n + 1)):
                need_bwd[i] = True

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video for temporal pass: {input_video}")

    ok, curr = cap.read()
    if not ok:
        cap.release()
        return union_masks, max_confs

    ok, nf = cap.read()
    next_frame = nf if ok else None
    prev_frame = None

    h, w = curr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    def warp_mask_from_other_to_curr(other_mask: np.ndarray, other_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        # flow computed as (curr -> other); then remap other_mask into curr coordinates
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        other_gray = cv2.cvtColor(other_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            curr_gray,
            other_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        ).astype(np.float32)
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        src = (other_mask.astype(np.uint8) * 255)
        warped = cv2.remap(
            src,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return (warped > 0)

    for i in range(n):
        if not has[i] and (need_interp[i] or need_fwd[i] or need_bwd[i]):
            new_mask: Optional[np.ndarray] = None

            if need_interp[i]:
                if prev_frame is not None and next_frame is not None and i - 1 >= 0 and i + 1 < n:
                    mp = union_masks[i - 1]
                    mn = union_masks[i + 1]
                    if mp is not None and mn is not None:
                        wp = warp_mask_from_other_to_curr(mp, prev_frame, curr)
                        wn = warp_mask_from_other_to_curr(mn, next_frame, curr)
                        new_mask = np.logical_or(wp, wn)

            if new_mask is None and need_fwd[i]:
                if prev_frame is not None and i - 1 >= 0:
                    mp = union_masks[i - 1]
                    if mp is not None:
                        new_mask = warp_mask_from_other_to_curr(mp, prev_frame, curr)

            if new_mask is None and need_bwd[i]:
                if next_frame is not None and i + 1 < n:
                    mn = union_masks[i + 1]
                    if mn is not None:
                        new_mask = warp_mask_from_other_to_curr(mn, next_frame, curr)

            if new_mask is not None and new_mask.any():
                union_masks[i] = fill_holes(new_mask)
                max_confs[i] = conf_th  # interpolated/extrapolated
                has[i] = True

        # advance window
        prev_frame = curr
        curr = next_frame if next_frame is not None else curr
        ok, nf = cap.read()
        next_frame = nf if ok else None

        if i == n - 1:
            break

    cap.release()
    return union_masks, max_confs


# ------------------------ main ------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Single input video.")
    ap.add_argument("--output", default=None, type=str, help="Output dir (default: ./{Filename}/).")
    ap.add_argument("--model", required=True, nargs="+", type=str, help="One or more model paths (ensemble by union).")
    ap.add_argument("--angle", default="0,45,90,135,180,225,270,315", type=str, help="Angles in degrees, comma-separated.")
    ap.add_argument("--shift", default=0, type=int, help="Shift in pixels after scaling (0 => no shifts).")
    ap.add_argument("--imgsz", default=1024, type=int, help="imgsz for YOLO predict + scaling temp videos.")
    ap.add_argument("--conf", default=0.10, type=float, help="Confidence threshold for YOLO predict.")
    ap.add_argument("--device", default="0", type=str, help="Device passed to YOLO predict (default 0).")
    ap.add_argument("--save_labels", action="store_true", help="Save final flattened labels in YOLO segmentation format.")
    ap.add_argument("--save_binary", action="store_true", help="Save binary mask TIFF seq + FFV1 MKV.")
    ap.add_argument("--keep_temp", action="store_true", help="Keep temp/ videos.")
    ap.add_argument("--int8", action="store_true", help="Pass int8=True to YOLO predict.")
    ap.add_argument("--half", action="store_true", help="Pass half=True to YOLO predict.")
    ap.add_argument("--temporal", default=1, type=int, help="Temporal strictness N (0 disables).")

    args = ap.parse_args()

    input_video = Path(args.input)
    if not input_video.exists():
        print(f"ERROR: input not found: {input_video}", file=sys.stderr)
        return 2

    if not ffmpeg_exists():
        print("ERROR: ffmpeg not found in PATH.", file=sys.stderr)
        return 2

    has_nvenc = ffmpeg_has_encoder("h264_nvenc")
    if not has_nvenc:
        print("[WARN] h264_nvenc not available; falling back to libx264 CRF 0 for temp videos.", file=sys.stderr)

    base = input_video.stem
    out_dir = Path(args.output) if args.output else (Path(".") / base)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = out_dir / "labels"
    binary_dir = out_dir / "binary_masks"
    temp_dir = out_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    if args.save_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
    if args.save_binary:
        binary_dir.mkdir(parents=True, exist_ok=True)

    orig_w, orig_h, fps, n_frames = probe_video(input_video)
    angles = parse_angles(args.angle)
    models = normalize_models(args.model)

    print(f"[INFO] Input: {input_video}")
    print(f"[INFO] Output: {out_dir}")
    print(f"[INFO] Video: {n_frames} frames @ {fps:.3f} FPS, {orig_w}x{orig_h}")
    print(f"[INFO] Angles: {angles} | shift={args.shift} | imgsz={args.imgsz}")
    print(f"[INFO] Models ({len(models)}): {models}")

    # 1-2) Generate temp videos (rotate -> scale -> shift).
    print("[INFO] Generating augmented videos...")
    variants = generate_augmented_videos(
        input_video=input_video,
        base=base,
        temp_dir=temp_dir,
        angles=angles,
        shift=args.shift,
        imgsz=args.imgsz,
        fps=fps,
        has_nvenc=has_nvenc,
    )
    print(f"[INFO] Variants: {len(variants)} (stored in {temp_dir})")

    # Accumulators (in-memory results).
    union_masks: List[Optional[np.ndarray]] = [None] * n_frames
    max_confs: List[float] = [0.0] * n_frames

    # 3) YOLO inference per model + variant.
    try:
        from ultralytics import YOLO
    except Exception:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        return 2

    for mi, mp in enumerate(models):
        print(f"[INFO] Loading model {mi+1}/{len(models)}: {mp}")
        model = YOLO(mp)

        for vi, var in enumerate(variants):
            print(f"[INFO] Predict {mi+1}/{len(models)} | {vi+1}/{len(variants)}: angle={var.angle_deg}, dx={var.dx}, dy={var.dy}")

            results_iter = safe_yolo_predict(
                model,
                source=str(var.path),
                task="segment",
                save=False,
                stream=True,
                iou=1.0,
                retina_masks=True,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                half=bool(args.half),
                int8=bool(args.int8),
                verbose=False,
            )

            for fi, result in enumerate(results_iter):
                if fi >= n_frames:
                    break
                mask_aug, cmax = extract_union_mask_and_conf(result)
                if cmax > max_confs[fi]:
                    max_confs[fi] = cmax
                if mask_aug is None:
                    continue
                mask_orig = undo_tta_mask_to_original(mask_aug, var.angle_deg, var.dx, var.dy, orig_w, orig_h)
                if not mask_orig.any():
                    continue
                if union_masks[fi] is None:
                    union_masks[fi] = mask_orig
                else:
                    union_masks[fi] |= mask_orig

    # 4-5) Fill holes per frame.
    print("[INFO] Filling enclosed holes...")
    for i in range(n_frames):
        if union_masks[i] is not None and union_masks[i].any():
            union_masks[i] = fill_holes(union_masks[i])

    # 6) Temporal stabilization (optional).
    if args.temporal and args.temporal > 0:
        print(f"[INFO] Temporal post-process N={args.temporal}...")
        union_masks, max_confs = apply_temporal_optical_flow(
            input_video=input_video,
            union_masks=union_masks,
            max_confs=max_confs,
            conf_th=args.conf,
            temporal_n=args.temporal,
        )

    # 7) Write outputs (overlay FFV1 MKV; optional binary + labels).
    final_video = out_dir / f"{base}_Final.mkv"
    binary_video = out_dir / f"{base}_Binary.mkv"

    print(f"[INFO] Writing final overlay video: {final_video}")
    overlay_writer = start_ffv1_writer(final_video, orig_w, orig_h, fps, pix_in="bgr24", pix_out="yuv444p")

    binary_writer = None
    if args.save_binary:
        print(f"[INFO] Writing binary mask video: {binary_video}")
        binary_writer = start_ffv1_writer(binary_video, orig_w, orig_h, fps, pix_in="gray", pix_out="gray")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to reopen video: {input_video}")

    blue = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    blue[:] = (255, 0, 0)  # BGR blue

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        mb = union_masks[i]
        mask_u8 = np.zeros((orig_h, orig_w), dtype=np.uint8) if (mb is None) else (mb.astype(np.uint8) * 255)

        if args.save_labels:
            p = labels_dir / f"{base}_{i:04d}.txt"
            lines = mask_to_yolo_seg_lines(mb)
            with open(p, "w", encoding="utf-8") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")

        if args.save_binary:
            tiff = binary_dir / f"{base}_Binary_{i:04d}.tiff"
            cv2.imwrite(str(tiff), mask_u8)

        if mb is not None and mb.any():
            blended = cv2.addWeighted(frame, 0.5, blue, 0.5, 0.0)
            out_frame = frame.copy()
            out_frame[mb] = blended[mb]
        else:
            out_frame = frame

        overlay_writer.write(out_frame)
        if binary_writer is not None:
            binary_writer.write(mask_u8)

    cap.release()
    overlay_writer.close()
    if binary_writer is not None:
        binary_writer.close()

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
