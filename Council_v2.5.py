#!/usr/bin/env python3
"""
YOLO Segmentation Test-Time Augmentation (TTA)

Pipeline:
    1. Rotate input video to specified angles (non-90° supported, clamped to input size)
    2. Scale to imgsz x imgsz, shift in 4 directions + center (5 variants per angle)
    3. Run YOLO segment inference (stream=True) on each temp video
    4. Inverse-transform masks back to original space, pixel-wise union
    5. Fill fully enclosed holes (donut → filled circle)
    6. Temporal smoothing: interpolate/extrapolate missing frames via optical flow
    7. Generate final overlay video, optional labels and binary masks

Requirements:
    pip install ultralytics opencv-python numpy
    ffmpeg with h264_nvenc support (for temp videos)
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="YOLO Segmentation TTA")
    p.add_argument("--input", required=True, help="Path to input video")
    p.add_argument("--output", default=None,
                   help="Output directory (default ./{Filename}/)")
    p.add_argument("--model", nargs="+", required=True,
                   help="One or more YOLO model paths")
    p.add_argument("--angle", type=str,
                   default="0,45,90,135,180,225,270,315",
                   help="Comma-separated rotation angles")
    p.add_argument("--shift", type=int, default=0,
                   help="Pixel shift amount (0 = no shift)")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="Image size for scaling and YOLO inference")
    p.add_argument("--conf", type=float, default=0.10,
                   help="Confidence threshold")
    p.add_argument("--temporal", type=int, default=1,
                   help="Temporal window N (higher = stricter)")
    p.add_argument("--device", type=str, default="0",
                   help="Device for YOLO inference")
    p.add_argument("--half", action="store_true", help="FP16 inference")
    p.add_argument("--int8", action="store_true", help="INT8 inference")
    p.add_argument("--save_labels", action="store_true",
                   help="Save YOLO-format segmentation labels")
    p.add_argument("--save_binary", action="store_true",
                   help="Save binary masks (TIFF sequence + FFV1 MKV)")
    p.add_argument("--keep_temp", action="store_true",
                   help="Keep temporary files")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Video Helpers
# ═══════════════════════════════════════════════════════════════
def get_video_dims(path):
    """Get dimensions and FPS only. Frame count is determined during decode."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return w, h, fps


def ffmpeg_pipe(output_path, w, h, fps, mode="temp"):
    """Open ffmpeg subprocess: rawvideo stdin -> encoded file.

    CRITICAL: Input pixel format must be specified BEFORE '-i -'.

    Modes:
        temp  - h264_nvenc lossless yuv444p in MP4
        final - FFV1 bgr24 in MKV
        mask  - FFV1 gray in MKV
    """
    in_pix = "gray" if mode == "mask" else "bgr24"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", in_pix,
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
    ]

    if mode == "temp":
#        cmd += ["-c:v", "h264_nvenc", "-preset", "lossless",

        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
                "-pix_fmt", "yuv444p", str(output_path)]
    elif mode == "final":
        cmd += ["-c:v", "ffv1", "-pix_fmt", "bgr24", str(output_path)]
    elif mode == "mask":
        cmd += ["-c:v", "ffv1", "-pix_fmt", "gray", str(output_path)]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)


class VideoSeeker:
    """Persistent reader to avoid thrashing file handles during temporal pass."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def read_at(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        self.cap.release()


# ═══════════════════════════════════════════════════════════════
# Augmentation Transform
# ═══════════════════════════════════════════════════════════════
class AugTransform:
    """Rotate -> scale to imgsz x imgsz -> shift."""

    def __init__(self, angle, sx, sy, orig_w, orig_h, imgsz):
        self.angle = angle
        self.sx, self.sy = sx, sy
        self.ow, self.oh = orig_w, orig_h
        self.sz = imgsz
        self.center = (orig_w / 2.0, orig_h / 2.0)

    def forward(self, frame):
        M = cv2.getRotationMatrix2D(self.center, self.angle, 1.0)
        rot = cv2.warpAffine(frame, M, (self.ow, self.oh),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
        scaled = cv2.resize(rot, (self.sz, self.sz),
                            interpolation=cv2.INTER_LINEAR)
        if self.sx != 0 or self.sy != 0:
            Ms = np.float32([[1, 0, self.sx], [0, 1, self.sy]])
            scaled = cv2.warpAffine(scaled, Ms, (self.sz, self.sz),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
        return scaled

    def inverse(self, mask):
        """Undo shift -> undo scale -> undo rotation on a binary mask."""
        if self.sx != 0 or self.sy != 0:
            Ms = np.float32([[1, 0, -self.sx], [0, 1, -self.sy]])
            mask = cv2.warpAffine(mask, Ms, (self.sz, self.sz),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        mask = cv2.resize(mask, (self.ow, self.oh),
                          interpolation=cv2.INTER_NEAREST)
        M = cv2.getRotationMatrix2D(self.center, -self.angle, 1.0)
        mask = cv2.warpAffine(mask, M, (self.ow, self.oh),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
        return mask


def build_augmentations(angles, shift, orig_w, orig_h, imgsz):
    if shift == 0:
        shift_pairs = [(0, 0)]
    else:
        shift_pairs = [
            (0, 0),
            (0, -shift),   # up
            (0, shift),    # down
            (-shift, 0),   # left
            (shift, 0),    # right
        ]
    augs = []
    for a in angles:
        for sx, sy in shift_pairs:
            augs.append(AugTransform(a, sx, sy, orig_w, orig_h, imgsz))
    return augs


# ═══════════════════════════════════════════════════════════════
# Hole Filling (OpenCV only, multi-seed)
# ═══════════════════════════════════════════════════════════════
def fill_holes(mask):
    """Fill fully enclosed holes via flood-fill from border seeds.

    Multi-seed approach handles foreground touching corners.
    Only fills regions completely surrounded by foreground.
    """
    inv = cv2.bitwise_not(mask)
    h, w = inv.shape
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if inv[seed[1], seed[0]] == 255:
            cv2.floodFill(inv, ff_mask, seed, 0)
    return cv2.bitwise_or(mask, inv)


# ═══════════════════════════════════════════════════════════════
# Optical-Flow Mask Warping
# ═══════════════════════════════════════════════════════════════
def warp_mask(src_frame, dst_frame, src_mask):
    """Warp src_mask from src_frame's perspective to dst_frame's.

    Uses backward flow (dst->src) for correct cv2.remap sampling.
    """
    gray_src = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(dst_frame, cv2.COLOR_BGR2GRAY)

    flow_back = cv2.calcOpticalFlowFarneback(
        gray_dst, gray_src, None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    h, w = flow_back.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x + flow_back[..., 0]
    map_y = grid_y + flow_back[..., 1]

    return cv2.remap(src_mask, map_x, map_y,
                     interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ═══════════════════════════════════════════════════════════════
# YOLO Label Conversion
# ═══════════════════════════════════════════════════════════════
def mask_to_yolo_lines(mask, img_w, img_h, class_id=0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 3:
            continue
        norm = pts.astype(np.float64)
        norm[:, 0] /= img_w
        norm[:, 1] /= img_h
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
        lines.append(f"{class_id} {coords}")
    return lines


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # -- Resolve model paths (handle nargs="+" and embedded commas) --
    model_paths = []
    for m in args.model:
        for part in m.split(","):
            part = part.strip()
            if part:
                model_paths.append(part)

    # -- Paths --
    input_path = Path(args.input).resolve()
    stem = input_path.stem
    input_str = str(input_path)

    if args.output is None:
        out_dir = Path(f"./{stem}")
    else:
        out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = out_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    label_dir = out_dir / "labels"
    binary_dir = out_dir / "binary_masks"
    if args.save_labels:
        label_dir.mkdir(exist_ok=True)
    if args.save_binary:
        binary_dir.mkdir(exist_ok=True)

    # -- Video info --
    orig_w, orig_h, fps = get_video_dims(input_str)
    print(f"[INFO] Input: {orig_w}x{orig_h} @ {fps:.2f} fps")

    # -- Build augmentations --
    angles = [float(a.strip()) for a in args.angle.split(",")]
    augs = build_augmentations(angles, args.shift,
                               orig_w, orig_h, args.imgsz)
    print(f"[INFO] {len(augs)} augmentation variant(s), "
          f"{len(model_paths)} model(s)")

    # ══════════════════════════════════════════════════════════════
    # STEP 1-2: Generate temp videos
    #   Sequential to avoid NVENC session limits on consumer GPUs.
    #   Frame count determined by actual decode (not header).
    # ══════════════════════════════════════════════════════════════
    temp_paths = []
    n_frames = None

    for idx, aug in enumerate(augs):
        t_path = temp_dir / f"aug_{idx}.mp4"
        temp_paths.append(str(t_path))

        cap = cv2.VideoCapture(input_str)
        writer = ffmpeg_pipe(t_path, args.imgsz, args.imgsz, fps, "temp")

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            aug_frame = aug.forward(frame)
            writer.stdin.write(aug_frame.tobytes())
            count += 1

        cap.release()
        writer.stdin.close()
        writer.wait()

        if n_frames is None:
            n_frames = count
        elif count != n_frames:
            print(f"[WARN] Frame count mismatch: variant {idx} "
                  f"had {count}, expected {n_frames}")

        print(f"  Variant {idx + 1}/{len(augs)} "
              f"(angle={aug.angle}, shift=({aug.sx},{aug.sy})) "
              f"— {count} frames")

    print(f"[INFO] Confirmed {n_frames} frames")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Inference — sequential per model per variant
    # ══════════════════════════════════════════════════════════════
    union_masks = [np.zeros((orig_h, orig_w), dtype=np.uint8)
                   for _ in range(n_frames)]
    max_confs = [0.0] * n_frames

    for m_path in model_paths:
        print(f"[INFER] Model: {m_path}")
        model = YOLO(m_path)

        for v_idx, t_path in enumerate(temp_paths):
            aug = augs[v_idx]
            results = model.predict(
                source=t_path,
                task="segment",
                save=False,
                stream=True,
                iou=1.0,
                retina_masks=True,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                half=args.half,
                int8=args.int8,
                verbose=False,
            )

            fi = 0
            for r in results:
                if fi >= n_frames:
                    break

                if r.masks is not None and len(r.masks.data) > 0:
                    conf = float(r.boxes.conf.max().cpu())
                    if conf > max_confs[fi]:
                        max_confs[fi] = conf

                    # Union detections: pure numpy, no torch import needed
                    masks_np = r.masks.data.cpu().numpy()  # (N, H, W) float32
                    var_union = (masks_np.max(axis=0) > 0.5).astype(np.uint8) * 255

                    if var_union.shape != (args.imgsz, args.imgsz):
                        var_union = cv2.resize(var_union,
                                               (args.imgsz, args.imgsz),
                                               interpolation=cv2.INTER_NEAREST)

                    orig_mask = aug.inverse(var_union)
                    union_masks[fi] = cv2.bitwise_or(union_masks[fi],
                                                     orig_mask)
                fi += 1

            print(f"  Variant {v_idx + 1}/{len(augs)} done")

    # ══════════════════════════════════════════════════════════════
    # STEP 4-5: Hole filling
    # ══════════════════════════════════════════════════════════════
    print("[POST] Filling holes...")
    for i in range(n_frames):
        if np.any(union_masks[i]):
            union_masks[i] = fill_holes(union_masks[i])

    # ══════════════════════════════════════════════════════════════
    # STEP 6: Temporal smoothing
    # ══════════════════════════════════════════════════════════════
    print("[POST] Temporal smoothing...")
    high_thresh = args.conf * 2.0
    N = args.temporal

    # Snapshot original confidences — Pass 1 uses this for neighbor
    # lookups so dropping frame i doesn't cascade to frame i+1
    orig_confs = list(max_confs)
    final_masks = [m.copy() for m in union_masks]
    working_confs = list(max_confs)

    # Persistent reader for temporal phase I/O
    seeker = VideoSeeker(input_str)

    # -- Pass 1: Drop isolated low-confidence detections --
    for i in range(n_frames):
        c = orig_confs[i]
        if c <= 0 or c >= high_thresh:
            continue  # missing or high-conf: skip
        # Low confidence: check +/-N neighbors using ORIGINAL confs
        found_neighbor = False
        for d in range(1, N + 1):
            if (i - d >= 0 and orig_confs[i - d] > 0) or \
               (i + d < n_frames and orig_confs[i + d] > 0):
                found_neighbor = True
                break
        if not found_neighbor:
            final_masks[i] = np.zeros((orig_h, orig_w), dtype=np.uint8)
            working_confs[i] = 0.0

    # -- Pass 2: Interpolation and extrapolation --
    # Uses working_confs (post-drop) for window checks.
    # Updates are deferred to prevent interpolated frames from
    # serving as sources for other interpolations.
    updates = {}

    for i in range(n_frames):
        if working_confs[i] >= high_thresh:
            continue

        # Backward window: frames [i-N .. i-1] all high?
        back_ok = True
        if i - N < 0:
            back_ok = False
        else:
            for k in range(1, N + 1):
                if working_confs[i - k] < high_thresh:
                    back_ok = False
                    break

        # Forward window: frames [i+1 .. i+N] all high?
        fwd_ok = True
        if i + N >= n_frames:
            fwd_ok = False
        else:
            for k in range(1, N + 1):
                if working_confs[i + k] < high_thresh:
                    fwd_ok = False
                    break

        # Helper for warping
        def get_warped(src_idx, dst_idx):
            src_f = seeker.read_at(src_idx)
            dst_f = seeker.read_at(dst_idx)
            if src_f is not None and dst_f is not None:
                return warp_mask(src_f, dst_f, final_masks[src_idx])
            return None

        # Interpolation: both sides have full high-conf runs
        if back_ok and fwd_ok:
            w_prev = get_warped(i - 1, i)
            w_next = get_warped(i + 1, i)
            if w_prev is not None and w_next is not None:
                updates[i] = cv2.bitwise_or(w_prev, w_next)
            continue

        # Forward extrapolation: back window all high, current missing
        if back_ok and working_confs[i] == 0.0:
            w_prev = get_warped(i - 1, i)
            if w_prev is not None:
                updates[i] = w_prev
            continue

        # Backward extrapolation: forward window all high, current missing
        if fwd_ok and working_confs[i] == 0.0:
            w_next = get_warped(i + 1, i)
            if w_next is not None:
                updates[i] = w_next
            continue

    seeker.close()

    # Apply deferred temporal updates
    for i, m in updates.items():
        final_masks[i] = m
        working_confs[i] = args.conf

    # ══════════════════════════════════════════════════════════════
    # STEP 7: Generate outputs
    # ══════════════════════════════════════════════════════════════
    print("[OUTPUT] Rendering...")

    # Final overlay video (FFV1 MKV)
    final_vid_path = out_dir / f"{stem}_final.mkv"
    vid_pipe = ffmpeg_pipe(final_vid_path, orig_w, orig_h, fps, "final")

    # Binary mask video (FFV1 MKV)
    bin_pipe = None
    if args.save_binary:
        bin_vid_path = out_dir / f"{stem}_Binary.mkv"
        bin_pipe = ffmpeg_pipe(bin_vid_path, orig_w, orig_h, fps, "mask")

    # Preallocate blue overlay
    blue_layer = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    blue_layer[:] = (255, 0, 0)  # BGR blue

    cap = cv2.VideoCapture(input_str)
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        mask = final_masks[i]

        # -- Labels --
        if args.save_labels:
            txt_path = label_dir / f"{stem}_{i:04d}.txt"
            lines = mask_to_yolo_lines(mask, orig_w, orig_h)
            with open(txt_path, "w") as f:
                for line in lines:
                    f.write(line + "\n")
                # File created even if empty (no detections)

        # -- Binary mask --
        if args.save_binary:
            bin_pipe.stdin.write(mask.tobytes())
            tiff_path = binary_dir / f"{stem}_Binary_{i:04d}.tiff"
            cv2.imwrite(str(tiff_path), mask)

        # -- Overlay (blue, 50% transparency) --
        overlay = frame.copy()
        roi = mask > 0
        if roi.any():
            overlay[roi] = cv2.addWeighted(
                frame[roi], 0.5,
                blue_layer[roi], 0.5, 0)
        vid_pipe.stdin.write(overlay.tobytes())

        if (i + 1) % 100 == 0 or i == n_frames - 1:
            print(f"  Frame {i + 1}/{n_frames}", end="\r")

    cap.release()

    vid_pipe.stdin.close()
    vid_pipe.wait()
    if bin_pipe is not None:
        bin_pipe.stdin.close()
        bin_pipe.wait()

    # -- Cleanup --
    if not args.keep_temp:
        shutil.rmtree(temp_dir)

    print(f"\n[DONE] Output in {out_dir}")


if __name__ == "__main__":
    main()
