#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) for Ultralytics YOLO segmentation on video.

Pipeline:
1) Create rotated+scaled versions of the input video for each angle (clockwise degrees),
   clamped to original size (corners clipped, black fill).
2) Encode those temp videos with h264_nvenc -preset lossless.
3) Run YOLO segmentation inference sequentially on each rotated video (and each model if multiple).
4) Convert predicted masks to a single union mask per frame, resize to original resolution,
   undo the rotation, and accumulate across angles/models (logical OR union).
5) Optional temporal union across ±N frames and hole filling.
6) Produce final overlay video (FFV1). Optionally save binary mask video and YOLO-format labels.

Requirements:
- Python: ultralytics, opencv-python, numpy
- ffmpeg installed with: h264_nvenc, ffv1 encoders
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

# Ultralytics
from ultralytics import YOLO


# ----------------------------
# Utilities
# ----------------------------

def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def run_cmd(cmd: List[str]) -> None:
    # Print command for reproducibility
    print("[CMD]", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        die(f"Command failed with code {p.returncode}")


def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def get_video_props_cv2(video_path: Path) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        die(f"Could not open input video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    if w <= 0 or h <= 0:
        die(f"Invalid video dimensions read from {video_path}: {w}x{h}")
    if fps <= 0:
        # Fallback if FPS not available
        fps = 30.0
        print("[WARN] FPS not detected; defaulting to 30.0")

    return w, h, fps, n


def parse_angles(angle_str: str) -> List[float]:
    # Accept "0,45,90" and also tolerate spaces
    parts = [p.strip() for p in angle_str.split(",")]
    angles: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            angles.append(float(p))
        except ValueError:
            die(f"Could not parse angle '{p}' from --angle '{angle_str}'")
    if not angles:
        die("No angles provided. Example: --angle 0,45,90,135")
    return angles


def sanitize_angle(a: float) -> str:
    # Stable filename token
    s = f"{a}".replace("-", "m").replace(".", "p")
    return s


def ensure_even(x: int) -> int:
    # H.264 encoders often require even dimensions.
    return x if (x % 2 == 0) else (x + 1)


def ffmpeg_rotate_scale_nvenc(
    in_video: Path,
    out_video: Path,
    angle_deg_clockwise: float,
    imgsz: int,
) -> None:
    """
    Rotate by arbitrary angle with output clamped to original size (ow=iw, oh=ih),
    fill exposed regions with black, then scale to imgsz x imgsz.
    Encode using h264_nvenc lossless preset.
    """
    if not ffmpeg_exists():
        die("ffmpeg not found in PATH. Install ffmpeg with nvenc + ffv1 support.")

    imgsz2 = ensure_even(imgsz)
    if imgsz2 != imgsz:
        print(f"[WARN] imgsz {imgsz} is odd; bumped to {imgsz2} for H.264 compatibility.")
        imgsz = imgsz2

    # FFmpeg rotate: positive is clockwise. :contentReference[oaicite:3]{index=3}
    vf = f"rotate={angle_deg_clockwise}*PI/180:ow=iw:oh=ih:c=black,scale={imgsz}:{imgsz}:flags=bilinear"

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(in_video),
        "-an",
        "-vf", vf,
        "-c:v", "h264_nvenc",
        "-preset", "lossless",
        "-rc", "constqp",
        "-qp", "0",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ]
    run_cmd(cmd)


def start_ffmpeg_writer_ffv1_bgr0(out_path: Path, fps: float, w: int, h: int) -> subprocess.Popen:
    """
    Start an ffmpeg process that accepts raw BGR24 frames via stdin and writes FFV1 (lossless).
    Output pixel format bgr0 (lossless, no chroma subsampling).
    """
    if not ffmpeg_exists():
        die("ffmpeg not found in PATH. Install ffmpeg.")

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
        "-c:v", "ffv1",
        "-pix_fmt", "bgr0",
        str(out_path),
    ]
    print("[CMD]", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def start_ffmpeg_writer_ffv1_gray(out_path: Path, fps: float, w: int, h: int) -> subprocess.Popen:
    """
    Start an ffmpeg process that accepts raw GRAY8 frames via stdin and writes FFV1 (lossless).
    """
    if not ffmpeg_exists():
        die("ffmpeg not found in PATH. Install ffmpeg.")

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{w}x{h}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
        "-c:v", "ffv1",
        "-pix_fmt", "gray",
        str(out_path),
    ]
    print("[CMD]", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def union_instance_masks(result) -> Optional[np.ndarray]:
    """
    Given a single Ultralytics Results object, return a single uint8 mask (0 or 255)
    that is the union across all predicted instance masks for that frame.
    """
    if result.masks is None:
        return None
    data = result.masks.data  # torch tensor [n, h, w]
    if data is None or len(data) == 0:
        return None

    # Convert to numpy on CPU
    m = data
    try:
        m = m.detach()
    except Exception:
        pass
    try:
        m = m.cpu()
    except Exception:
        pass

    mnp = m.numpy()
    # Union across instances
    u = (np.sum(mnp, axis=0) > 0).astype(np.uint8) * 255
    return u


def undo_rotation_mask(
    mask_u8: np.ndarray,
    angle_deg_clockwise: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    We rotated the video clockwise by angle. To map mask back to original,
    rotate mask counter-clockwise by the same angle.
    OpenCV getRotationMatrix2D uses positive angles as CCW.
    """
    center = (out_w / 2.0, out_h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg_clockwise, 1.0)  # CCW by angle
    out = cv2.warpAffine(
        mask_u8,
        M,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out


def fill_enclosed_holes(mask_u8: np.ndarray) -> np.ndarray:
    """
    Fill fully enclosed holes in a binary mask (0/255).
    This is the classic flood-fill background then invert approach.
    """
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    if mask_u8.max() == 0:
        return mask_u8

    # Pad with black so (0,0) is guaranteed background
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()

    # Flood fill the background from the corner with white (255)
    h, w = flood.shape[:2]
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, flood_inv)

    return filled[1:-1, 1:-1]


def overlay_mask_bgr(frame_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    """
    Overlay mask on top of original frame. (No boxes/classes/conf shown.)
    """
    if mask_u8 is None or mask_u8.max() == 0:
        return frame_bgr

    # Green overlay (BGR)
    overlay = frame_bgr.copy()
    overlay[mask_u8 > 0] = (0, 255, 0)

    out = cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0)
    return out


def mask_to_yolo_seg_lines(mask_u8: np.ndarray, cls_id: int, w: int, h: int) -> List[str]:
    """
    Convert a binary mask into YOLO segmentation label lines:
      class x1 y1 x2 y2 ... (all normalized 0..1)
    Uses external contours. If multiple contours exist, it outputs multiple lines.
    """
    if mask_u8 is None or mask_u8.max() == 0:
        return []

    # Find external contours
    cnts, _hier = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: List[str] = []

    for cnt in cnts:
        if cnt is None or len(cnt) < 3:
            continue
        # Optional simplification to reduce label size
        perim = cv2.arcLength(cnt, True)
        eps = max(1.0, 0.001 * perim)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2)

        if pts.shape[0] < 3:
            continue

        coords: List[str] = []
        for (x, y) in pts:
            xn = float(x) / float(w)
            yn = float(y) / float(h)
            # clamp to [0,1]
            xn = 0.0 if xn < 0 else (1.0 if xn > 1 else xn)
            yn = 0.0 if yn < 0 else (1.0 if yn > 1 else yn)
            coords.append(f"{xn:.6f}")
            coords.append(f"{yn:.6f}")

        lines.append(f"{cls_id} " + " ".join(coords))

    return lines


def read_mask_png(mask_path: Path, h: int, w: int) -> np.ndarray:
    if not mask_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((h, w), dtype=np.uint8)
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.uint8) * 255
    return m


def write_mask_png(mask_path: Path, mask_u8: np.ndarray) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask_u8)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="TTA script for Ultralytics YOLO segmentation on video.")
    ap.add_argument("video", type=str, help="Input video path")

    ap.add_argument("--model", type=str, nargs="+", required=True,
                    help="One or more YOLO segmentation models (e.g., best.pt or exported engine). "
                         "If multiple are provided, outputs are ensembled via mask union.")

    ap.add_argument("--angle", type=str, default="0",
                    help="Comma-separated clockwise angles in degrees. Example: --angle 0,45,90,135,180,225,270,315")

    ap.add_argument("--imgsz", type=int, default=640, help="Square inference/resize size (e.g., 640)")
    ap.add_argument("--conf", type=float, default=0.10, help="Confidence threshold (default 0.10)")
    ap.add_argument("--device", type=str, default="0", help="Device passed to YOLO predict (default '0')")
    ap.add_argument("--half", action="store_true", help="Enable FP16 inference in YOLO predict")
    ap.add_argument("--int8", action="store_true",
                    help="Attempt INT8 inference (typically requires an INT8-exported model). "
                         "If unsupported by backend, it will be ignored or may raise an error.")

    ap.add_argument("--temporal", type=int, default=0,
                    help="Temporal merge window radius: union masks from frames [t-N, t+N]. Default 0.")

    ap.add_argument("--save_labels", action="store_true",
                    help="Save final flattened labels in YOLO segmentation format (blank .txt for empty frames).")

    ap.add_argument("--save_binary", action="store_true",
                    help="Save a binary mask video (black bg, white fg), encoded with FFV1.")

    ap.add_argument("--no_temp", action="store_true",
                    help="Clean up temporary files at the end. (Temp files are kept by default.)")

    ap.add_argument("--output", type=str, default=None,
                    help="Output overlay video path. Default: <input_stem>_tta.mkv (FFV1)")

    ap.add_argument("--alpha", type=float, default=0.40,
                    help="Overlay alpha for mask visualization (default 0.40)")

    args = ap.parse_args()

    in_video = Path(args.video)
    if not in_video.exists():
        die(f"Input video not found: {in_video}")

    angles = parse_angles(args.angle)
    imgsz = args.imgsz

    # Output paths
    if args.output is None:
        out_overlay = in_video.with_name(in_video.stem + "_tta.mkv")
    else:
        out_overlay = Path(args.output)

    out_binary = out_overlay.with_name(out_overlay.stem + "_binary.mkv")
    out_labels_dir = out_overlay.with_name(out_overlay.stem + "_labels")

    # Read original properties
    orig_w, orig_h, fps, nframes = get_video_props_cv2(in_video)
    print(f"[INFO] Input: {in_video}")
    print(f"[INFO] Resolution: {orig_w}x{orig_h}, FPS: {fps:.3f}, Frames: {nframes}")

    # Temp workspace
    # Keep by default; delete only if --no_temp was passed.
    tmp_root = Path(tempfile.mkdtemp(prefix="yolo_tta_"))
    print(f"[INFO] Temp dir: {tmp_root} (kept by default; pass --no_temp to remove)")

    rotated_dir = tmp_root / "rotated_videos"
    runs_dir = tmp_root / "ultra_runs"
    accum_dir = tmp_root / "accum_masks"
    rotated_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    accum_dir.mkdir(parents=True, exist_ok=True)

    # Load models once
    models: List[YOLO] = []
    model_tags: List[str] = []
    for mpath in args.model:
        mp = Path(mpath)
        tag = mp.stem if mp.suffix else mp.name
        model_tags.append(tag)
        print(f"[INFO] Loading model: {mpath}")
        models.append(YOLO(mpath))

    # ----------------------------
    # 1-4) Build rotated videos, run inference, accumulate union masks
    # ----------------------------
    for a in angles:
        a_tag = sanitize_angle(a)
        rot_vid = rotated_dir / f"angle_{a_tag}_imgsz{imgsz}.mp4"

        if not rot_vid.exists():
            print(f"[INFO] Creating rotated+scaled temp video for angle={a} -> {rot_vid}")
            ffmpeg_rotate_scale_nvenc(in_video, rot_vid, angle_deg_clockwise=a, imgsz=imgsz)
        else:
            print(f"[INFO] Reusing existing temp video: {rot_vid}")

        for model, tag in zip(models, model_tags):
            run_name = f"{tag}_angle_{a_tag}"
            print(f"[INFO] Predicting: model={tag}, angle={a}, source={rot_vid}")

            # Ultralytics predict args: stream=True yields generator; retina_masks gives high-res masks. :contentReference[oaicite:4]{index=4}
            predict_kwargs = dict(
                source=str(rot_vid),
                stream=True,
                save=False,
                save_txt=True,
                conf=float(args.conf),
                iou=1.0,
                retina_masks=True,
                imgsz=int(imgsz),
                device=str(args.device),
                half=bool(args.half),
                project=str(runs_dir),
                name=run_name,
                exist_ok=True,
                vid_stride=1,
                verbose=False,
            )

            # Try to pass int8 if requested. Some backends may ignore or error.
            if args.int8:
                predict_kwargs["int8"] = True

            try:
                results_iter = model.predict(**predict_kwargs)
            except Exception as e:
                if args.int8:
                    print(f"[WARN] predict() failed with int8=True ({e}). Retrying without int8...")
                    predict_kwargs.pop("int8", None)
                    results_iter = model.predict(**predict_kwargs)
                else:
                    raise

            frame_idx = 0
            for r in results_iter:
                um = union_instance_masks(r)
                if um is None:
                    frame_idx += 1
                    continue

                # Resize union mask back to original resolution
                um_orig = cv2.resize(um, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                # Undo rotation back to original orientation
                um_unrot = undo_rotation_mask(um_orig, angle_deg_clockwise=a, out_w=orig_w, out_h=orig_h)

                if um_unrot.max() == 0:
                    frame_idx += 1
                    continue

                # Accumulate (union) into per-frame stored mask
                mask_path = accum_dir / f"{frame_idx:06d}.png"
                if mask_path.exists():
                    prev = read_mask_png(mask_path, orig_h, orig_w)
                    merged = cv2.bitwise_or(prev, um_unrot)
                else:
                    merged = um_unrot

                write_mask_png(mask_path, merged)
                frame_idx += 1

            print(f"[INFO] Done model={tag}, angle={a}. Processed frames: {frame_idx}")

    # ----------------------------
    # 5) Final output pass: temporal merge, hole fill, overlay render, save labels/binary
    # ----------------------------
    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        die(f"Could not reopen input video: {in_video}")

    out_overlay.parent.mkdir(parents=True, exist_ok=True)

    writer_overlay = start_ffmpeg_writer_ffv1_bgr0(out_overlay, fps=fps, w=orig_w, h=orig_h)
    writer_binary = None
    if args.save_binary:
        writer_binary = start_ffmpeg_writer_ffv1_gray(out_binary, fps=fps, w=orig_w, h=orig_h)

    if args.save_labels:
        out_labels_dir.mkdir(parents=True, exist_ok=True)

    temporal = int(args.temporal)
    alpha = float(args.alpha)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Base (already unioned across angles/models) mask for this frame
        base_path = accum_dir / f"{frame_idx:06d}.png"

        if temporal <= 0:
            m = read_mask_png(base_path, orig_h, orig_w)
        else:
            # Union across window [t-N, t+N]
            m = np.zeros((orig_h, orig_w), dtype=np.uint8)
            t0 = max(0, frame_idx - temporal)
            t1 = frame_idx + temporal
            # If frame count is unknown, just try the files; missing => empty
            for t in range(t0, t1 + 1):
                mp = accum_dir / f"{t:06d}.png"
                if mp.exists():
                    m = cv2.bitwise_or(m, read_mask_png(mp, orig_h, orig_w))

        # Fill enclosed holes (low-priority request)
        m = fill_enclosed_holes(m)

        # Save binary video frame if requested
        if writer_binary is not None and writer_binary.stdin is not None:
            writer_binary.stdin.write(m.tobytes())

        # Save labels if requested (blank file for no detections)
        if args.save_labels:
            label_path = out_labels_dir / f"{frame_idx:06d}.txt"
            if m.max() == 0:
                label_path.write_text("")  # blank
            else:
                lines = mask_to_yolo_seg_lines(m, cls_id=0, w=orig_w, h=orig_h)
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        # Overlay into output frame
        vis = overlay_mask_bgr(frame, m, alpha=alpha)

        if writer_overlay.stdin is None:
            die("FFmpeg overlay writer stdin is None (failed to start writer?)")
        writer_overlay.stdin.write(vis.tobytes())

        frame_idx += 1

    cap.release()

    # Close writers
    if writer_overlay.stdin is not None:
        writer_overlay.stdin.close()
    writer_overlay.wait()

    if writer_binary is not None:
        if writer_binary.stdin is not None:
            writer_binary.stdin.close()
        writer_binary.wait()

    print(f"[INFO] Wrote overlay video: {out_overlay}")
    if args.save_binary:
        print(f"[INFO] Wrote binary mask video: {out_binary}")
    if args.save_labels:
        print(f"[INFO] Wrote labels to: {out_labels_dir}")

    # Cleanup temp if requested
    if args.no_temp:
        print(f"[INFO] Cleaning temp dir: {tmp_root}")
        shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        print(f"[INFO] Temp kept at: {tmp_root} (use --no_temp to remove)")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()