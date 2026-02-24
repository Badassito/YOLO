#!/usr/bin/env python3
"""
YOLO Segmentation TTA Script (v3.0 spec)

Pipeline summary (per spec):
1) Build Transverse (X,Y), Sagittal (X,t) and Coronal (Y,t) views from an input video interpreted as (X,Y,t).
2) For each view, generate FFV1/MKV augmented videos for each --angle and (optionally) 8-direction --shift variants:
   - rotate (clamped to input size, fill black)
   - scale to --imgsz x --imgsz
   - shift by N pixels (clamped to imgsz, fill black)
3) Run YOLO segment inference on each augmented video sequentially for each model in --model:
   save=False, stream=True, iou=1.0, retina_masks=True, batch=1, plus imgsz/conf/device/half/int8 flags.
   Store union-per-frame masks+confidence to disk.
4) Read results, invert affine (undo shift+scale+rotation) back to native view coordinates, and union across augs.
5) Fill 2D holes (enclosed) per slice; drop masks with long axis-aligned contour edges (>30 px in imgsz space).
6) Union across views into a 3D volume (t,Y,X). If --disable_multiplanar: only Transverse.
7) Fill 3D holes (enclosed) in the final volume (multiplanar only).
8) Output only Transverse:
   - final overlay video (blue mask, 50% alpha) encoded FFV1/MKV
   - optional YOLO-format labels per frame (blank files for empty)
   - optional binary masks as TIFF sequence + FFV1/MKV

Notes:
- Requires ffmpeg + ffprobe in PATH.
- Requires: ultralytics, opencv-python, numpy, scipy
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
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from scipy import ndimage

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


# ----------------------------
# Utilities
# ----------------------------

def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command with streaming stdout/stderr."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )
    return p


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_fraction(fr: str) -> float:
    # e.g. "30000/1001"
    if not fr or fr == "0/0":
        return 0.0
    if "/" in fr:
        a, b = fr.split("/")
        a = float(a)
        b = float(b)
        return a / b if b != 0 else 0.0
    return float(fr)


def ffprobe_video_info(path: Path) -> Dict[str, float | int]:
    """
    Returns dict with: width, height, frames, fps
    Uses -count_frames to get nb_read_frames when nb_frames isn't present.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=width,height,nb_frames,nb_read_frames,avg_frame_rate",
        "-of", "json",
        str(path)
    ]
    out = _run(cmd).stdout
    j = json.loads(out)
    if "streams" not in j or not j["streams"]:
        raise RuntimeError(f"No video stream found in: {path}")
    s = j["streams"][0]
    w = int(s.get("width"))
    h = int(s.get("height"))
    nb_frames = s.get("nb_frames")
    nb_read_frames = s.get("nb_read_frames")
    frames = int(nb_frames) if nb_frames not in (None, "N/A") else int(nb_read_frames) if nb_read_frames not in (None, "N/A") else -1
    fps = _parse_fraction(s.get("avg_frame_rate", "0/0"))
    return {"width": w, "height": h, "frames": frames, "fps": fps}


def parse_angles(arg: str) -> List[float]:
    # Accept "0,120,240" or "0 120 240" if passed as a single string.
    parts = []
    for token in arg.replace(",", " ").split():
        if token.strip():
            parts.append(float(token))
    if not parts:
        return [0.0, 120.0, 240.0]
    return parts


# ----------------------------
# FFmpeg: views and augmentations
# ----------------------------

def ffmpeg_reencode_ffv1(in_path: Path, out_path: Path, *, pix_fmt: str = "bgr0") -> None:
    """
    Lossless-ish master using FFV1 in MKV.
    pix_fmt=bgr0 avoids RGB<->YUV conversion rounding.
    """
    _ensure_dir(out_path.parent)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-an",
        "-c:v", "ffv1",
        "-level", "3",
        "-pix_fmt", pix_fmt,
        str(out_path),
    ]
    _run(cmd)


def ffmpeg_decode_to_raw_bgr24(in_path: Path, raw_out: Path) -> None:
    """Decode video to raw BGR24 frames (for multiplanar view creation)."""
    _ensure_dir(raw_out.parent)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-an",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        str(raw_out),
    ]
    _run(cmd)


def _ffmpeg_open_rawvideo_writer(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    """
    Open an ffmpeg process that reads raw bgr24 frames from stdin and writes FFV1/MKV.
    """
    _ensure_dir(out_path.parent)
    # Note: -r here defines input timestamps; for inference it doesn't really matter, but keep consistent.
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}" if fps > 0 else "30",
        "-i", "-",
        "-an",
        "-c:v", "ffv1",
        "-level", "3",
        "-pix_fmt", "bgr0",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def create_sagittal_and_coronal_views(
    transverse_raw_bgr24: Path,
    *,
    out_sagittal: Path,
    out_coronal: Path,
    X: int,
    Y: int,
    T: int,
    fps: float,
) -> None:
    """
    Input transverse_raw_bgr24 is frames in order t=0..T-1, each frame size YxX with 3 channels (BGR).
    Creates:
      - Sagittal video: frames indexed by y (0..Y-1), each frame is (T x X)
      - Coronal  video: frames indexed by x (0..X-1), each frame is (T x Y)
    """
    expected_bytes = T * Y * X * 3
    actual_bytes = transverse_raw_bgr24.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Raw file size mismatch.\n"
            f"Expected: {expected_bytes} bytes (T*Y*X*3) = {T}*{Y}*{X}*3\n"
            f"Actual:   {actual_bytes} bytes\n"
            f"Raw path: {transverse_raw_bgr24}"
        )

    mm = np.memmap(transverse_raw_bgr24, dtype=np.uint8, mode="r", shape=(T, Y, X, 3))

    # Sagittal: width=X, height=T, frames=Y
    sag_p = _ffmpeg_open_rawvideo_writer(out_sagittal, width=X, height=T, fps=fps)
    assert sag_p.stdin is not None
    for y in range(Y):
        frame = np.ascontiguousarray(mm[:, y, :, :])  # (T, X, 3)
        sag_p.stdin.write(frame.tobytes())
    sag_p.stdin.close()
    _, sag_err = sag_p.communicate()
    if sag_p.returncode != 0:
        raise RuntimeError(f"FFmpeg sagittal encode failed:\n{sag_err.decode('utf-8', errors='ignore')}")

    # Coronal: width=Y, height=T, frames=X
    cor_p = _ffmpeg_open_rawvideo_writer(out_coronal, width=Y, height=T, fps=fps)
    assert cor_p.stdin is not None
    for x in range(X):
        frame = np.ascontiguousarray(mm[:, :, x, :])  # (T, Y, 3)
        cor_p.stdin.write(frame.tobytes())
    cor_p.stdin.close()
    _, cor_err = cor_p.communicate()
    if cor_p.returncode != 0:
        raise RuntimeError(f"FFmpeg coronal encode failed:\n{cor_err.decode('utf-8', errors='ignore')}")


def ffmpeg_make_augmented_video(
    *,
    in_path: Path,
    out_path: Path,
    angle_deg: float,
    imgsz: int,
    shift: int,
    dx: int,
    dy: int,
) -> None:
    """
    Builds augmented FFV1/MKV:
      rotate (clamped to input size), scale to imgsz x imgsz, optional shift via pad+crop (clamped), fill black.
    """
    _ensure_dir(out_path.parent)

    # rotate expects radians; positive is clockwise.
    # Clamp output to input size by forcing ow=iw, oh=ih and fill black.
    vf = f"rotate={angle_deg}*PI/180:ow=iw:oh=ih:c=black,scale={imgsz}:{imgsz}:flags=bicubic"

    if shift > 0:
        pad_w = imgsz + 2 * shift
        pad_h = imgsz + 2 * shift
        # Place scaled image at (shift, shift), then crop a window shifted by (dx, dy)
        crop_x = shift - dx
        crop_y = shift - dy
        vf += f",pad={pad_w}:{pad_h}:{shift}:{shift}:color=black,crop={imgsz}:{imgsz}:{crop_x}:{crop_y}"

    # Keep in RGB/BGR domain (avoid YUV rounding)
    vf += ",format=bgr0"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-vf", vf,
        "-an",
        "-c:v", "ffv1",
        "-level", "3",
        "-pix_fmt", "bgr0",
        str(out_path),
    ]
    _run(cmd)


# ----------------------------
# Affine matrices (for inversion)
# ----------------------------

def affine_original_to_aug(
    *,
    orig_w: int,
    orig_h: int,
    imgsz: int,
    angle_deg: float,
    dx: int,
    dy: int,
) -> np.ndarray:
    """
    Returns 2x3 affine mapping original (orig_w,orig_h) -> augmented (imgsz,imgsz)
    Composite: rotate around center (clockwise for +angle in image coords), then scale to imgsz, then shift (dx,dy).
    """
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)

    cx = (orig_w - 1) / 2.0
    cy = (orig_h - 1) / 2.0

    # Rotation about center in image coords (y down), +theta yields clockwise.
    R = np.array([
        [c, -s, cx - c * cx + s * cy],
        [s,  c, cy - s * cx - c * cy],
        [0,  0, 1],
    ], dtype=np.float32)

    sx = float(imgsz) / float(orig_w)
    sy = float(imgsz) / float(orig_h)

    S = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
    ], dtype=np.float32)

    T = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1],
    ], dtype=np.float32)

    M = (T @ S @ R)[:2, :]
    return M


def affine_aug_to_original(
    *,
    orig_w: int,
    orig_h: int,
    imgsz: int,
    angle_deg: float,
    dx: int,
    dy: int,
) -> np.ndarray:
    """Returns 2x3 affine mapping augmented -> original (invert of original->aug)."""
    M_fwd = affine_original_to_aug(orig_w=orig_w, orig_h=orig_h, imgsz=imgsz, angle_deg=angle_deg, dx=dx, dy=dy)
    M_inv = cv2.invertAffineTransform(M_fwd)
    return M_inv.astype(np.float32)


# ----------------------------
# Mask ops: holes + edge filter + labels
# ----------------------------

def fill_holes_2d(mask_u8_255: np.ndarray) -> np.ndarray:
    """Fill fully enclosed holes in a 2D mask. Input/output: uint8 0 or 255."""
    if mask_u8_255.max() == 0:
        return mask_u8_255
    filled = ndimage.binary_fill_holes(mask_u8_255 > 0)
    return (filled.astype(np.uint8) * 255)


def has_long_axis_aligned_edge(mask_u8_255: np.ndarray, *, thresh_x: float, thresh_y: float) -> bool:
    """
    Delete masks with a straight horizontal or vertical edge longer than threshold.
    Thresholds are in THIS mask's pixel space (converted from imgsz space).
    """
    if mask_u8_255.max() == 0:
        return False

    contours, _ = cv2.findContours((mask_u8_255 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False

    max_h = 0.0
    max_v = 0.0

    for cnt in contours:
        pts = cnt[:, 0, :]  # (N,2)
        if pts.shape[0] < 2:
            continue
        # close loop
        pts2 = np.vstack([pts, pts[:1]])
        run_h = 0.0
        run_v = 0.0
        for i in range(1, pts2.shape[0]):
            x0, y0 = pts2[i - 1]
            x1, y1 = pts2[i]
            dx = abs(float(x1 - x0))
            dy = abs(float(y1 - y0))

            if dy == 0.0 and dx > 0.0:
                run_h += dx
            else:
                run_h = 0.0

            if dx == 0.0 and dy > 0.0:
                run_v += dy
            else:
                run_v = 0.0

            if run_h > max_h:
                max_h = run_h
            if run_v > max_v:
                max_v = run_v

            if max_h >= thresh_x or max_v >= thresh_y:
                return True

    return (max_h >= thresh_x) or (max_v >= thresh_y)


def mask_to_yolo_seg_line(mask_u8_255: np.ndarray) -> Optional[str]:
    """
    Convert a binary mask to a single YOLO segmentation polygon line for class 0.
    If multiple contours exist, uses the largest contour by area.
    Output format: "0 x1 y1 x2 y2 ...", normalized.
    """
    h, w = mask_u8_255.shape[:2]
    if mask_u8_255.max() == 0:
        return None

    contours, _ = cv2.findContours((mask_u8_255 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]

    # Optional simplification (keeps file sizes sane)
    eps = 1.0
    approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
    pts = approx[:, 0, :]  # (N,2)

    # Need at least 3 points (polygon)
    if pts.shape[0] < 3:
        return None

    # Normalize
    coords = []
    for x, y in pts:
        coords.append(f"{(x / w):.6f}")
        coords.append(f"{(y / h):.6f}")

    return "0 " + " ".join(coords)


# ----------------------------
# YOLO inference on augmented videos
# ----------------------------

@dataclass
class AugVariant:
    view: str
    angle_deg: float
    dx: int
    dy: int
    aug_video: Path
    pred_masks_bits_npy: Path
    pred_confs_npy: Path
    meta_json: Path
    M_aug_to_orig: np.ndarray  # 2x3 float32


def run_yolo_segment_on_video(
    *,
    model,
    source_video: Path,
    expected_frames: int,
    imgsz: int,
    conf: float,
    device: str,
    half: bool,
    int8: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs YOLO segmentation on a video and returns:
      - mask_bits: (expected_frames, bytes_per_mask) uint8 (bitpacked)
      - confs:     (expected_frames,) float32
    Each frame's masks are unioned to one object; conf is max among instances.
    """
    bytes_per = (imgsz * imgsz + 7) // 8
    mask_bits = np.zeros((expected_frames, bytes_per), dtype=np.uint8)
    confs = np.zeros((expected_frames,), dtype=np.float32)

    kwargs = dict(
        source=str(source_video),
        save=False,
        stream=True,
        iou=1.0,
        retina_masks=True,
        batch=1,
        imgsz=imgsz,
        conf=conf,
        device=device,
        half=half,
        verbose=False,
    )
    # Some backends may accept int8; if not, we retry without it.
    if int8:
        kwargs["int8"] = True

    try:
        results_iter = model.predict(**kwargs)
    except TypeError as e:
        if "int8" in str(e) and "int8" in kwargs:
            kwargs.pop("int8", None)
            results_iter = model.predict(**kwargs)
        else:
            raise

    i = 0
    for r in results_iter:
        if i >= expected_frames:
            break

        frame_mask = np.zeros((imgsz, imgsz), dtype=np.uint8)  # 0/1
        frame_conf = 0.0

        if getattr(r, "masks", None) is not None and r.masks is not None and getattr(r.masks, "data", None) is not None:
            m = r.masks.data  # torch tensor (N,H,W)
            if m is not None and len(m) > 0:
                m_np = (m.detach().float().cpu().numpy() > 0.5)  # bool (N,H,W)
                frame_mask = np.any(m_np, axis=0).astype(np.uint8)  # 0/1

                if getattr(r, "boxes", None) is not None and r.boxes is not None and getattr(r.boxes, "conf", None) is not None:
                    c = r.boxes.conf.detach().float().cpu().numpy()
                    if c.size > 0:
                        frame_conf = float(np.max(c))

        # Pack bits
        packed = np.packbits(frame_mask.reshape(-1), bitorder="big")
        mask_bits[i, :] = packed[:bytes_per]
        confs[i] = frame_conf
        i += 1

    if i != expected_frames:
        # If ffprobe frame count differs from what YOLO yielded, trim down.
        mask_bits = mask_bits[:i]
        confs = confs[:i]

    return mask_bits, confs


# ----------------------------
# View processing (union augs -> insert into volume)
# ----------------------------

def build_aug_variants(
    *,
    view_name: str,
    view_video: Path,
    angles: List[float],
    imgsz: int,
    shift: int,
    tmp_aug_dir: Path,
    tmp_pred_dir: Path,
    orig_w: int,
    orig_h: int,
) -> List[AugVariant]:
    variants: List[AugVariant] = []
    directions: List[Tuple[int, int]] = [(0, 0)]
    if shift > 0:
        directions += [
            (0, -shift),
            (0, shift),
            (-shift, 0),
            (shift, 0),
            (-shift, -shift),
            (shift, -shift),
            (-shift, shift),
            (shift, shift),
        ]

    for a in angles:
        for (dx, dy) in directions:
            tag = f"{view_name}_a{int(a)}_dx{dx}_dy{dy}"
            aug_video = tmp_aug_dir / f"{tag}.mkv"
            pred_masks = tmp_pred_dir / f"{tag}_masks_bits.npy"
            pred_confs = tmp_pred_dir / f"{tag}_confs.npy"
            meta_json = tmp_pred_dir / f"{tag}_meta.json"
            M_inv = affine_aug_to_original(orig_w=orig_w, orig_h=orig_h, imgsz=imgsz, angle_deg=a, dx=dx, dy=dy)

            variants.append(AugVariant(
                view=view_name,
                angle_deg=a,
                dx=dx,
                dy=dy,
                aug_video=aug_video,
                pred_masks_bits_npy=pred_masks,
                pred_confs_npy=pred_confs,
                meta_json=meta_json,
                M_aug_to_orig=M_inv,
            ))
    return variants


def process_view_into_volume(
    *,
    model,
    view_name: str,
    view_video: Path,
    angles: List[float],
    imgsz: int,
    conf: float,
    shift: int,
    device: str,
    half: bool,
    int8: bool,
    expected_frames: int,
    orig_w: int,
    orig_h: int,
    volume_u8_01: np.memmap,  # (T, Y, X) uint8 0/1
    conf_time: np.ndarray,    # (T,) float32
    tmp_aug_dir: Path,
    tmp_pred_dir: Path,
    keep_temp: bool,
    # insertion mode:
    #   transverse: slice_index -> volume[t,:,:]
    #   sagittal:   slice_index=y -> volume[:,y,:]
    #   coronal:    slice_index=x -> volume[:,:,x]
    T: int,
    X: int,
    Y: int,
) -> None:
    """
    Full per-view pipeline:
      - generate augmented videos
      - run inference per augmented video and store union masks/conf to disk
      - read back, invert affine, union across augs per slice
      - fill 2D holes, edge-filter, insert into 3D volume
    """
    _ensure_dir(tmp_aug_dir)
    _ensure_dir(tmp_pred_dir)

    variants = build_aug_variants(
        view_name=view_name,
        view_video=view_video,
        angles=angles,
        imgsz=imgsz,
        shift=shift,
        tmp_aug_dir=tmp_aug_dir,
        tmp_pred_dir=tmp_pred_dir,
        orig_w=orig_w,
        orig_h=orig_h,
    )

    # 1) Generate aug videos + predict + save results to disk
    for v in variants:
        # Augmented video
        ffmpeg_make_augmented_video(
            in_path=view_video,
            out_path=v.aug_video,
            angle_deg=v.angle_deg,
            imgsz=imgsz,
            shift=shift,
            dx=v.dx,
            dy=v.dy,
        )

        # Predict -> store
        mask_bits, confs = run_yolo_segment_on_video(
            model=model,
            source_video=v.aug_video,
            expected_frames=expected_frames,
            imgsz=imgsz,
            conf=conf,
            device=device,
            half=half,
            int8=int8,
        )
        np.save(v.pred_masks_bits_npy, mask_bits)
        np.save(v.pred_confs_npy, confs)
        with open(v.meta_json, "w", encoding="utf-8") as f:
            json.dump({
                "view": v.view,
                "angle_deg": v.angle_deg,
                "dx": v.dx,
                "dy": v.dy,
                "imgsz": imgsz,
                "orig_w": orig_w,
                "orig_h": orig_h,
                "expected_frames": expected_frames,
                "pred_masks_bits_npy": str(v.pred_masks_bits_npy),
                "pred_confs_npy": str(v.pred_confs_npy),
                "aug_video": str(v.aug_video),
            }, f, indent=2)

        if not keep_temp:
            # keep preds; drop heavy aug video
            if v.aug_video.exists():
                v.aug_video.unlink()

    # 2) Read results (as arrays) for slice-wise fusion
    loaded_masks_bits: List[np.ndarray] = []
    loaded_confs: List[np.ndarray] = []
    for v in variants:
        loaded_masks_bits.append(np.load(v.pred_masks_bits_npy, mmap_mode=None))
        loaded_confs.append(np.load(v.pred_confs_npy, mmap_mode=None))

    bytes_per = (imgsz * imgsz + 7) // 8

    # Thresholds in ORIGINAL view pixel space, derived from "30 pixels in imgsz space"
    # Horizontal edges measured in X direction -> scale factor (orig_w/imgsz)
    # Vertical edges measured in Y direction -> scale factor (orig_h/imgsz)
    thresh_x = 30.0 * (float(orig_w) / float(imgsz))
    thresh_y = 30.0 * (float(orig_h) / float(imgsz))

    # 3) For each slice/frame in this view, union across augs (after inverse affine)
    for s in range(expected_frames):
        acc = np.zeros((orig_h, orig_w), dtype=np.uint8)  # 0/255
        best_conf = 0.0

        for idx, v in enumerate(variants):
            conf_s = float(loaded_confs[idx][s]) if s < loaded_confs[idx].shape[0] else 0.0
            if conf_s <= 0.0:
                continue

            row = loaded_masks_bits[idx][s]
            # fast skip if fully empty
            if not np.any(row):
                continue

            bits = np.unpackbits(row, bitorder="big")
            bits = bits[:imgsz * imgsz].reshape((imgsz, imgsz)).astype(np.uint8) * 255

            warped = cv2.warpAffine(
                bits,
                v.M_aug_to_orig,
                (orig_w, orig_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            acc = cv2.bitwise_or(acc, warped)
            if conf_s > best_conf:
                best_conf = conf_s

        # 2D hole filling + edge filter
        if acc.max() > 0:
            acc = fill_holes_2d(acc)
            if has_long_axis_aligned_edge(acc, thresh_x=thresh_x, thresh_y=thresh_y):
                acc.fill(0)
                best_conf = 0.0

        acc01 = (acc > 0).astype(np.uint8)  # 0/1

        # Insert into 3D volume + update per-time confidence
        if view_name == "transverse":
            # s is time index (0..T-1), acc is (Y,X)
            volume_u8_01[s, :, :] |= acc01
            # confidence for time s
            if best_conf > conf_time[s]:
                conf_time[s] = best_conf

        elif view_name == "sagittal":
            # s is y index (0..Y-1), acc is (T,X)
            volume_u8_01[:, s, :] |= acc01
            # if any voxel exists at time t in this slice, update conf_time[t]
            if best_conf > 0.0 and acc01.max() > 0:
                present_t = np.any(acc01 > 0, axis=1)  # (T,)
                conf_time[present_t] = np.maximum(conf_time[present_t], best_conf)

        elif view_name == "coronal":
            # s is x index (0..X-1), acc is (T,Y)
            volume_u8_01[:, :, s] |= acc01
            if best_conf > 0.0 and acc01.max() > 0:
                present_t = np.any(acc01 > 0, axis=1)  # (T,)
                conf_time[present_t] = np.maximum(conf_time[present_t], best_conf)

        else:
            raise ValueError(f"Unknown view: {view_name}")

    # 4) Cleanup per-view preds if not keeping temp
    if not keep_temp:
        for v in variants:
            for p in [v.pred_masks_bits_npy, v.pred_confs_npy, v.meta_json]:
                if p.exists():
                    p.unlink(missing_ok=True)


# ----------------------------
# Temporal refinement (optical flow mask warping)
# ----------------------------

def read_small_gray_frames(video_path: Path, *, target: int = 512) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open for temporal frames: {video_path}")

    frames = []
    ok, frame = cap.read()
    while ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if max(h, w) > target:
            scale = target / float(max(h, w))
            nw = int(round(w * scale))
            nh = int(round(h * scale))
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        frames.append(gray)
        ok, frame = cap.read()

    cap.release()
    return frames


def compute_dis_flows(frames_small: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (fwd, bwd) optical flows between consecutive frames at small resolution:
      fwd[i] = flow from i -> i+1   (H,W,2) float16
      bwd[i] = flow from i+1 -> i   (H,W,2) float16
    """
    if len(frames_small) < 2:
        raise ValueError("Need >=2 frames for optical flow.")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    n = len(frames_small) - 1
    h, w = frames_small[0].shape[:2]
    fwd = np.zeros((n, h, w, 2), dtype=np.float16)
    bwd = np.zeros((n, h, w, 2), dtype=np.float16)

    for i in range(n):
        a = frames_small[i]
        b = frames_small[i + 1]
        flow_ab = dis.calc(a, b, None)  # float32
        flow_ba = dis.calc(b, a, None)
        fwd[i] = flow_ab.astype(np.float16)
        bwd[i] = flow_ba.astype(np.float16)
    return fwd, bwd


def upsample_flow(flow_small: np.ndarray, full_w: int, full_h: int) -> np.ndarray:
    """
    Upsample flow from small-res to full-res and scale vector magnitudes accordingly.
    flow_small shape: (hs, ws, 2)
    """
    hs, ws = flow_small.shape[:2]
    flow = flow_small.astype(np.float32)
    flow_up = cv2.resize(flow, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    flow_up[..., 0] *= (full_w / float(ws))
    flow_up[..., 1] *= (full_h / float(hs))
    return flow_up


def warp_mask_with_flow(mask_src_u8: np.ndarray, flow_dest_to_src_full: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """
    Warp mask from src frame into dest frame using flow defined as dest->src (backward mapping).
    """
    map_x = (grid_x + flow_dest_to_src_full[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_dest_to_src_full[..., 1]).astype(np.float32)
    warped = cv2.remap(
        mask_src_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def warp_mask_along_time(
    masks_u8_255: np.memmap,
    *,
    src_idx: int,
    dst_idx: int,
    fwd_small: np.ndarray,
    bwd_small: np.ndarray,
    full_w: int,
    full_h: int,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """
    Warp a full-res mask from src_idx to dst_idx by chaining 1-step warps.
    Uses:
      - to go forward (k -> k+1): use bwd_small[k] (flow k+1 -> k)
      - to go backward (k -> k-1): use fwd_small[k-1] (flow k-1 -> k)
    """
    out = np.array(masks_u8_255[src_idx], copy=True)  # full-res

    if src_idx == dst_idx:
        return out

    if src_idx < dst_idx:
        # forward chain
        for k in range(src_idx, dst_idx):
            flow_full = upsample_flow(bwd_small[k], full_w, full_h)  # (k+1)->k
            out = warp_mask_with_flow(out, flow_full, grid_x, grid_y)
    else:
        # backward chain
        for k in range(src_idx, dst_idx, -1):
            flow_full = upsample_flow(fwd_small[k - 1], full_w, full_h)  # (k-1)->k
            out = warp_mask_with_flow(out, flow_full, grid_x, grid_y)

    return out


def temporal_refine(
    masks_u8_255: np.memmap,
    conf_time: np.ndarray,
    *,
    transverse_video: Path,
    N: int,
    conf_base: float,
) -> None:
    """
    Implements v3.0 spec temporal logic on final transverse masks.

    High confidence: >= 2*conf_base
    Low confidence:  (0, 2*conf_base)
    Interp/extrap conf: set to conf_base
    """
    if N <= 0:
        return

    T = masks_u8_255.shape[0]
    H = masks_u8_255.shape[1]
    W = masks_u8_255.shape[2]
    high = 2.0 * conf_base

    # Preload small grayscale frames + flows
    frames_small = read_small_gray_frames(transverse_video, target=512)
    if len(frames_small) != T:
        # If ffmpeg/cv2 differs, clamp
        minT = min(len(frames_small), T)
        frames_small = frames_small[:minT]
        T = minT

    fwd_small, bwd_small = compute_dis_flows(frames_small)

    # Precompute full-res grids once
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    def has_mask(i: int) -> bool:
        return masks_u8_255[i].max() > 0

    # 1) Interpolation
    for i in range(N, T - N):
        if has_mask(i):
            continue
        ok_neighbors = True
        for k in range(1, N + 1):
            if not (conf_time[i - k] >= high and has_mask(i - k)):
                ok_neighbors = False
                break
            if not (conf_time[i + k] >= high and has_mask(i + k)):
                ok_neighbors = False
                break
        if not ok_neighbors:
            continue

        interp = np.zeros((H, W), dtype=np.uint8)
        for k in range(1, N + 1):
            m1 = warp_mask_along_time(
                masks_u8_255,
                src_idx=i - k,
                dst_idx=i,
                fwd_small=fwd_small,
                bwd_small=bwd_small,
                full_w=W,
                full_h=H,
                grid_x=grid_x,
                grid_y=grid_y,
            )
            m2 = warp_mask_along_time(
                masks_u8_255,
                src_idx=i + k,
                dst_idx=i,
                fwd_small=fwd_small,
                bwd_small=bwd_small,
                full_w=W,
                full_h=H,
                grid_x=grid_x,
                grid_y=grid_y,
            )
            interp = cv2.bitwise_or(interp, m1)
            interp = cv2.bitwise_or(interp, m2)

        masks_u8_255[i] = interp
        conf_time[i] = conf_base

    # 2) Forward extrapolation
    for i in range(N, T):
        if has_mask(i):
            continue
        ok_prev = True
        for k in range(1, N + 1):
            if not (conf_time[i - k] >= high and has_mask(i - k)):
                ok_prev = False
                break
        if not ok_prev:
            continue

        pred = warp_mask_along_time(
            masks_u8_255,
            src_idx=i - 1,
            dst_idx=i,
            fwd_small=fwd_small,
            bwd_small=bwd_small,
            full_w=W,
            full_h=H,
            grid_x=grid_x,
            grid_y=grid_y,
        )
        masks_u8_255[i] = pred
        conf_time[i] = conf_base

    # 3) Backward extrapolation
    for i in range(T - N - 1, -1, -1):
        if has_mask(i):
            continue
        ok_next = True
        for k in range(1, N + 1):
            if not (conf_time[i + k] >= high and has_mask(i + k)):
                ok_next = False
                break
        if not ok_next:
            continue

        pred = warp_mask_along_time(
            masks_u8_255,
            src_idx=i + 1,
            dst_idx=i,
            fwd_small=fwd_small,
            bwd_small=bwd_small,
            full_w=W,
            full_h=H,
            grid_x=grid_x,
            grid_y=grid_y,
        )
        masks_u8_255[i] = pred
        conf_time[i] = conf_base

    # 4) Drop low confidence predictions if +-N neighbors contain no objects of any confidence
    for i in range(T):
        if not has_mask(i):
            continue
        if conf_time[i] >= high:
            continue  # never drop high conf
        if conf_time[i] <= 0:
            continue

        neighbor_has_any = False
        for k in range(1, N + 1):
            if i - k >= 0 and has_mask(i - k):
                neighbor_has_any = True
                break
            if i + k < T and has_mask(i + k):
                neighbor_has_any = True
                break

        if not neighbor_has_any:
            masks_u8_255[i].fill(0)
            conf_time[i] = 0.0


# ----------------------------
# Outputs
# ----------------------------

def save_labels_yolo(
    *,
    labels_dir: Path,
    base: str,
    masks_u8_255: np.memmap,
) -> None:
    _ensure_dir(labels_dir)
    T = masks_u8_255.shape[0]
    for i in range(T):
        p = labels_dir / f"{base}_{i:04d}.txt"
        line = mask_to_yolo_seg_line(np.array(masks_u8_255[i], copy=False))
        if line is None:
            # blank file
            p.write_text("", encoding="utf-8")
        else:
            p.write_text(line + "\n", encoding="utf-8")


def save_binary_masks(
    *,
    binary_dir: Path,
    base: str,
    masks_u8_255: np.memmap,
    mask_video_path: Path,
    fps: float,
    save_tiffs: bool,
) -> None:
    _ensure_dir(binary_dir)
    T, H, W = masks_u8_255.shape

    # ffmpeg writer for mask video
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{W}x{H}",
        "-r", f"{fps:.6f}" if fps > 0 else "30",
        "-i", "-",
        "-an",
        "-c:v", "ffv1",
        "-level", "3",
        "-pix_fmt", "gray",
        str(mask_video_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdin is not None

    for i in range(T):
        m = np.array(masks_u8_255[i], copy=False)  # uint8 0/255
        if save_tiffs:
            tif = binary_dir / f"{base}_Binary_{i:04d}.tiff"
            # black background, white masks already satisfied by 0/255
            cv2.imwrite(str(tif), m)
        p.stdin.write(m.tobytes())

    p.stdin.close()
    _, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg binary mask video encode failed:\n{err.decode('utf-8', errors='ignore')}")


def make_final_overlay_video(
    *,
    transverse_video: Path,
    mask_video: Path,
    out_final: Path,
    width: int,
    height: int,
) -> None:
    """
    Blue overlay with 50% alpha where mask is white.
    """
    _ensure_dir(out_final.parent)

    # Build blue RGBA stream, use mask as alpha scaled by 0.5.
    # alpha = mask * 0.5 (mask is 0 or 255)
    filt = (
        f"[1:v]format=gray,lut='val*0.5'[a];"
        f"color=c=blue:s={width}x{height},format=rgba[c];"
        f"[c][a]alphamerge[ov];"
        f"[0:v][ov]overlay=0:0:format=auto,format=bgr0"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(transverse_video),
        "-i", str(mask_video),
        "-filter_complex", filt,
        "-an",
        "-c:v", "ffv1",
        "-level", "3",
        "-pix_fmt", "bgr0",
        str(out_final),
    ]
    _run(cmd)


# ----------------------------
# Main
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("tta_yolo_seg_v3.py")

    p.add_argument("--input", required=True, type=str, help="Single input video")
    p.add_argument("--output", default=None, type=str, help="Output directory (default: ./{Filename}/)")
    p.add_argument("--model", required=True, nargs="+", type=str, help="One or more model paths (ensemble by union)")

    p.add_argument("--disable_multiplanar", action="store_true", help="Only use Transverse view if set")
    p.add_argument("--angle", default="0,120,240", type=str, help="Rotation angles in degrees (comma or space separated)")
    p.add_argument("--imgsz", default=1536, type=int, help="Scale augmented inputs to imgsz x imgsz; passed to YOLO")
    p.add_argument("--conf", default=0.15, type=float, help="YOLO conf; also temporal base conf")
    p.add_argument("--shift", default=0, type=int, help="Shift magnitude in pixels in imgsz-space (0 disables shifting)")

    p.add_argument("--save_labels", action="store_true", help="Save final flattened YOLO segmentation labels")
    p.add_argument("--save_binary", action="store_true", help="Save TIFF sequence + FFV1 MKV of final binary masks")
    p.add_argument("--keep_temp", action="store_true", help="Keep temp files")
    p.add_argument("--int8", action="store_true", help="Enable int8 if backend supports it")
    p.add_argument("--half", action="store_true", help="Enable fp16/half during prediction")

    p.add_argument("--temporal", default=1, type=int, help="Temporal window N (0 disables)")

    p.add_argument("--device", default="0", type=str, help="Device string passed to YOLO (default: 0)")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    if YOLO is None:
        raise RuntimeError("ultralytics is not importable. Install with: pip install ultralytics")

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    base = in_path.stem
    out_dir = Path(args.output).expanduser().resolve() if args.output else (Path(".").resolve() / base)
    labels_dir = out_dir / "labels"
    binary_dir = out_dir / "binary_masks"
    temp_dir = out_dir / "temp"
    views_dir = temp_dir / "views"
    aug_dir = temp_dir / "aug"
    preds_dir = temp_dir / "preds"

    _ensure_dir(out_dir)
    _ensure_dir(temp_dir)
    _ensure_dir(views_dir)
    _ensure_dir(aug_dir)
    _ensure_dir(preds_dir)

    angles = parse_angles(args.angle)

    # 1) Create transverse master (FFV1)
    transverse_view = views_dir / f"{base}_Transverse_FFV1.mkv"
    if not transverse_view.exists():
        ffmpeg_reencode_ffv1(in_path, transverse_view, pix_fmt="bgr0")

    info = ffprobe_video_info(transverse_view)
    X = int(info["width"])
    Y = int(info["height"])
    T = int(info["frames"])
    fps = float(info["fps"])

    if T <= 0:
        raise RuntimeError("Could not determine frame count with ffprobe.")

    # Spec expects 3072x3072x1930, but we don't hard-fail; we just warn.
    if (X, Y, T) != (3072, 3072, 1930):
        print(f"[WARN] Input dims not (3072,3072,1930). Detected: X={X}, Y={Y}, T={T}", file=sys.stderr)

    sagittal_view = views_dir / f"{base}_Sagittal_FFV1.mkv"
    coronal_view = views_dir / f"{base}_Coronal_FFV1.mkv"

    # 1) Create multiplanar views if enabled
    if not args.disable_multiplanar:
        raw_bgr24 = views_dir / f"{base}_Transverse_bgr24.raw"
        if not raw_bgr24.exists():
            ffmpeg_decode_to_raw_bgr24(transverse_view, raw_bgr24)

        if not sagittal_view.exists() or not coronal_view.exists():
            create_sagittal_and_coronal_views(
                raw_bgr24,
                out_sagittal=sagittal_view,
                out_coronal=coronal_view,
                X=X, Y=Y, T=T, fps=fps
            )
    else:
        print("[INFO] --disable_multiplanar set: skipping sagittal/coronal view creation.", file=sys.stderr)

    # Combined volume (t, Y, X) as uint8 0/1
    vol_path = temp_dir / "combined_volume_u8_01.dat"
    volume = np.memmap(vol_path, dtype=np.uint8, mode="w+", shape=(T, Y, X))
    volume[:] = 0

    # Per-time confidence after unions (used by temporal)
    conf_time = np.zeros((T,), dtype=np.float32)

    # Process each model sequentially (ensemble by union into the same volume)
    for mi, mpath in enumerate(args.model):
        mpath_p = Path(mpath).expanduser().resolve()
        if not mpath_p.exists():
            raise FileNotFoundError(mpath_p)

        print(f"[INFO] Loading model {mi+1}/{len(args.model)}: {mpath_p}", file=sys.stderr)
        model = YOLO(str(mpath_p), task="segment", verbose=False)

        model_tmp_aug = aug_dir / f"model_{mi:02d}"
        model_tmp_pred = preds_dir / f"model_{mi:02d}"
        _ensure_dir(model_tmp_aug)
        _ensure_dir(model_tmp_pred)

        # Transverse view: orig_w=X, orig_h=Y, slices=T
        print("[INFO] Processing Transverse view...", file=sys.stderr)
        process_view_into_volume(
            model=model,
            view_name="transverse",
            view_video=transverse_view,
            angles=angles,
            imgsz=args.imgsz,
            conf=args.conf,
            shift=args.shift,
            device=args.device,
            half=args.half,
            int8=args.int8,
            expected_frames=T,
            orig_w=X,
            orig_h=Y,
            volume_u8_01=volume,
            conf_time=conf_time,
            tmp_aug_dir=model_tmp_aug / "transverse",
            tmp_pred_dir=model_tmp_pred / "transverse",
            keep_temp=args.keep_temp,
            T=T, X=X, Y=Y,
        )

        if not args.disable_multiplanar:
            # Sagittal view: frames=Y, each frame is (T x X), so orig_w=X, orig_h=T
            print("[INFO] Processing Sagittal view...", file=sys.stderr)
            process_view_into_volume(
                model=model,
                view_name="sagittal",
                view_video=sagittal_view,
                angles=angles,
                imgsz=args.imgsz,
                conf=args.conf,
                shift=args.shift,
                device=args.device,
                half=args.half,
                int8=args.int8,
                expected_frames=Y,
                orig_w=X,
                orig_h=T,
                volume_u8_01=volume,
                conf_time=conf_time,
                tmp_aug_dir=model_tmp_aug / "sagittal",
                tmp_pred_dir=model_tmp_pred / "sagittal",
                keep_temp=args.keep_temp,
                T=T, X=X, Y=Y,
            )

            # Coronal view: frames=X, each frame is (T x Y), so orig_w=Y, orig_h=T
            print("[INFO] Processing Coronal view...", file=sys.stderr)
            process_view_into_volume(
                model=model,
                view_name="coronal",
                view_video=coronal_view,
                angles=angles,
                imgsz=args.imgsz,
                conf=args.conf,
                shift=args.shift,
                device=args.device,
                half=args.half,
                int8=args.int8,
                expected_frames=X,
                orig_w=Y,
                orig_h=T,
                volume_u8_01=volume,
                conf_time=conf_time,
                tmp_aug_dir=model_tmp_aug / "coronal",
                tmp_pred_dir=model_tmp_pred / "coronal",
                keep_temp=args.keep_temp,
                T=T, X=X, Y=Y,
            )

        # free model between runs
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 7) 3D hole fill (multiplanar only per spec)
    if not args.disable_multiplanar:
        print("[INFO] Performing 3D hole filling on combined volume...", file=sys.stderr)
        vol_bool = np.asarray(volume, dtype=bool)  # loads into RAM
        filled_bool = ndimage.binary_fill_holes(vol_bool)  # bool (T,Y,X)
        del vol_bool

        # Write final transverse masks (uint8 0/255)
        final_mask_path = temp_dir / "final_transverse_masks_u8_255.dat"
        final_masks = np.memmap(final_mask_path, dtype=np.uint8, mode="w+", shape=(T, Y, X))
        for t in range(T):
            final_masks[t] = filled_bool[t].astype(np.uint8) * 255

        del filled_bool
    else:
        # No 3D fill; just use transverse-derived volume (0/1) as final
        print("[INFO] Multiplanar disabled: skipping 3D hole filling.", file=sys.stderr)
        final_mask_path = temp_dir / "final_transverse_masks_u8_255.dat"
        final_masks = np.memmap(final_mask_path, dtype=np.uint8, mode="w+", shape=(T, Y, X))
        for t in range(T):
            final_masks[t] = volume[t].astype(np.uint8) * 255

    # Final safety pass: fill 2D holes per frame (covers union-induced donut holes)
    print("[INFO] Final 2D hole fill pass on transverse masks...", file=sys.stderr)
    for t in range(T):
        if final_masks[t].max() > 0:
            final_masks[t] = fill_holes_2d(np.array(final_masks[t], copy=False))

    # Temporal postprocess
    if args.temporal and args.temporal > 0:
        print(f"[INFO] Temporal refinement enabled (N={args.temporal})...", file=sys.stderr)
        temporal_refine(
            final_masks,
            conf_time,
            transverse_video=transverse_view,
            N=int(args.temporal),
            conf_base=float(args.conf),
        )

    # Save labels (blank files if empty)
    if args.save_labels:
        print("[INFO] Saving YOLO segmentation labels...", file=sys.stderr)
        save_labels_yolo(labels_dir=labels_dir, base=base, masks_u8_255=final_masks)

    # Save binary masks (TIFFs + video) if requested; always make a mask video for overlay
    if args.save_binary:
        print("[INFO] Saving binary TIFF sequence + mask video...", file=sys.stderr)
        mask_video_out = out_dir / f"{base}_Binary.mkv"
        save_binary_masks(
            binary_dir=binary_dir,
            base=base,
            masks_u8_255=final_masks,
            mask_video_path=mask_video_out,
            fps=fps,
            save_tiffs=True,
        )
        mask_video_for_overlay = mask_video_out
    else:
        # temp mask video
        mask_video_tmp = temp_dir / f"{base}_Binary_TMP.mkv"
        print("[INFO] Creating temporary binary mask video for overlay...", file=sys.stderr)
        save_binary_masks(
            binary_dir=binary_dir,   # directory unused when save_tiffs=False
            base=base,
            masks_u8_255=final_masks,
            mask_video_path=mask_video_tmp,
            fps=fps,
            save_tiffs=False,
        )
        mask_video_for_overlay = mask_video_tmp

    # Final overlay video
    final_video_out = out_dir / f"{base}_Final.mkv"
    print("[INFO] Creating final overlay video (blue @ 50%)...", file=sys.stderr)
    make_final_overlay_video(
        transverse_video=transverse_view,
        mask_video=mask_video_for_overlay,
        out_final=final_video_out,
        width=X,
        height=Y,
    )

    # Cleanup temp
    if not args.keep_temp:
        print("[INFO] Cleaning temp directory...", file=sys.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("[DONE]")
    print(f"Output directory: {out_dir}")
    print(f"Final video:      {final_video_out}")
    if args.save_binary:
        print(f"Binary mask video:{out_dir / f'{base}_Binary.mkv'}")
        print(f"Binary TIFFs:     {binary_dir}")
    if args.save_labels:
        print(f"Labels:           {labels_dir}")


if __name__ == "__main__":
    main()