"""
YOLO Video TTA (Test Time Augmentation) Segmentation
----------------------------------------------------
Rotates input video to various angles, runs YOLO segmentation inference,
and merges the results to improve detection robustness.

Key Features:
- Lossless intermediate storage (h264_nvenc/FFV1) to prevent artifact accumulation.
- Streaming pipeline to handle videos of arbitrary length without RAM exhaustion.
- Temporal smoothing to stabilize masks across frames.
- Topological hole filling to close enclosed gaps in masks.
- Robust error handling for OpenCV/FFmpeg edge cases.

Usage:
    # Basic usage (default angles 0-315 by 45)
    python Council.py input.mp4 --model yolov8x-seg.pt

    # Production usage with ensemble and temporal smoothing
    python Council.py input.mp4 \
        --model yolov8x-seg.pt,yolov8l-seg.pt \
        --angle 0,90,180,270 \
        --imgsz 640 \
        --conf 0.25 \
        --temporal 2 \
        --save_labels \
        --save_binary \
        --no_temp

    # High performance (requires TensorRT exported model)
    python Council.py input.mp4 --model yolov8x-seg.engine --int8
"""

import argparse
import cv2
import numpy as np
import os
import shutil
import subprocess
import sys
from pathlib import Path
from ultralytics import YOLO
from scipy.ndimage import binary_fill_holes


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Video TTA Segmentation")
    parser.add_argument("input", type=str, help="Path to input video")
    parser.add_argument("--model", type=str, required=True,
                        help="Comma-separated paths to YOLO models (e.g. 'yolo1.pt,yolo2.pt')")
    parser.add_argument("--angle", type=str, default="0,45,90,135,180,225,270,315",
                        help="Comma-separated rotation angles (default: 0-315 steps of 45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--temporal", type=int, default=0,
                        help="Temporal smoothing (+/- N frames)")

    # Precision flags
    parser.add_argument("--half", action="store_true", help="Use FP16 half precision")
    parser.add_argument("--int8", action="store_true",
                        help="Use Int8 precision (requires exported TensorRT/ONNX model)")

    # Output flags
    parser.add_argument("--save_labels", action="store_true",
                        help="Save YOLO format TXT labels")
    parser.add_argument("--save_binary", action="store_true",
                        help="Save binary mask video")
    parser.add_argument("--no_temp", action="store_true",
                        help="Clean up temp files after completion (default: keep them)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# FFmpeg Writer Class
# ---------------------------------------------------------------------------

class FFmpegWriter:
    """Writes video via FFmpeg subprocess pipe with robust cleanup."""

    def __init__(self, filename, w, h, fps, codec="ffv1",
                 preset=None, pixel_format=None):
        self.w, self.h = w, h
        self.filename = filename

        input_pix = 'gray' if pixel_format == 'gray' else 'bgr24'

        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-r', str(fps),
            '-pix_fmt', input_pix,
            '-i', '-',
            '-c:v', codec,
        ]

        if codec == "h264_nvenc":
            cmd.extend(['-preset', preset if preset else 'lossless'])
            cmd.extend(['-pix_fmt', 'yuv444p'])
        elif codec == "ffv1":
            if pixel_format == 'gray':
                cmd.extend(['-pix_fmt', 'gray'])
            else:
                cmd.extend(['-pix_fmt', 'bgr0'])

        cmd.append(str(filename))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        if self.proc.stdin.closed:
            return
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h))
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print(f"Error: FFmpeg pipe broken for {self.filename}",
                  file=sys.stderr)

    def close(self):
        if self.proc:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_video_metadata(path):
    """Return (width, height, fps, frame_count) for a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, cnt


def rotate_frame(image, angle, interpolation=cv2.INTER_LINEAR):
    """Rotate frame around its center; output clamped to input size."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def fill_holes_scipy(mask):
    """Fill fully enclosed holes in a binary uint8 mask (0/255)."""
    filled = binary_fill_holes(mask > 127)
    return filled.astype(np.uint8) * 255


def mask_to_yolo_poly(mask, w, h):
    """Convert a binary mask to YOLO polygon label lines (class 0)."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        pts = cnt.reshape(-1, 2).astype(float)
        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0, 1)
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"0 {coords}")
    return lines


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: {input_path} not found.")

    if args.int8:
        has_engine = any(
            m.strip().endswith(('.engine', '.onnx'))
            for m in args.model.split(','))
        if not has_engine:
            print("WARNING: --int8 passed but no .engine/.onnx model found.")
            print("         Standard .pt models will ignore this flag.")

    orig_w, orig_h, fps, total_frames = get_video_metadata(str(input_path))
    angles = [float(a) for a in args.angle.split(',')]
    models = [m.strip() for m in args.model.split(',')]

    temp_dir = Path("temp_tta_" + input_path.stem)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input : {input_path} ({orig_w}x{orig_h} @ {fps:.2f}fps, "
          f"{total_frames} frames)")
    print(f"Config: {len(angles)} angles, {len(models)} model(s), "
          f"temporal +/-{args.temporal}")
    print(f"Temp  : {temp_dir}")

    aug_video_paths = {}
    mask_video_paths = {}

    try:
        # -------------------------------------------------------------------
        # Step 1: Generate rotated + scaled temporary videos (h264_nvenc)
        # -------------------------------------------------------------------
        print("\n--- Step 1/3: Generating Augmented Videos ---")

        cap = cv2.VideoCapture(str(input_path))
        writers = {}

        try:
            for ang in angles:
                p = temp_dir / f"aug_{ang:.1f}.mp4"
                aug_video_paths[ang] = p
                writers[ang] = FFmpegWriter(
                    str(p), args.imgsz, args.imgsz, fps,
                    codec="h264_nvenc", preset="lossless")

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                for ang in angles:
                    rot = rotate_frame(frame, ang)
                    scaled = cv2.resize(rot, (args.imgsz, args.imgsz),
                                        interpolation=cv2.INTER_LINEAR)
                    writers[ang].write(scaled)
                idx += 1
                if idx % 100 == 0:
                    print(f"  Augmenting frame {idx}/{total_frames}",
                          end='\r')
        finally:
            cap.release()
            for w in writers.values():
                w.close()

        print(f"\n  Augmentation complete ({idx} frames).")

        # -------------------------------------------------------------------
        # Step 2: Run inference, save masks as FFV1 grayscale videos
        # -------------------------------------------------------------------
        print("\n--- Step 2/3: Inference ---")

        for m_idx, m_path in enumerate(models):
            print(f"  Loading model {m_idx + 1}/{len(models)}: {m_path}")
            model = YOLO(m_path)

            for ang in angles:
                src = str(aug_video_paths[ang])
                dst = temp_dir / f"mask_m{m_idx}_a{ang:.1f}.mkv"
                mask_video_paths[(m_idx, ang)] = dst

                mw = FFmpegWriter(str(dst), args.imgsz, args.imgsz, fps,
                                  codec="ffv1", pixel_format="gray")

                print(f"    Angle {ang:>7.1f}°...", end=" ", flush=True)

                results = model.predict(
                    source=src, save=False, stream=True, save_txt=False,
                    conf=args.conf, iou=1.0, imgsz=args.imgsz,
                    retina_masks=True, device=args.device,
                    half=args.half, int8=args.int8, verbose=False)

                count = 0
                for res in results:
                    count += 1
                    combined = np.zeros(
                        (args.imgsz, args.imgsz), dtype=np.uint8)
                    if res.masks is not None and len(res.masks.data) > 0:
                        m_sum = (res.masks.data.sum(dim=0)
                                 .clamp(0, 1).cpu().numpy())
                        combined = (m_sum * 255).astype(np.uint8)
                        if combined.shape != (args.imgsz, args.imgsz):
                            combined = cv2.resize(
                                combined, (args.imgsz, args.imgsz),
                                interpolation=cv2.INTER_NEAREST)
                    mw.write(combined)

                mw.close()
                print(f"{count} frames")

            del model

        # -------------------------------------------------------------------
        # Step 3: Aggregate masks, apply temporal smoothing, write outputs
        # -------------------------------------------------------------------
        print("\n--- Step 3/3: Aggregation & Output ---")

        out_name = f"{input_path.stem}_tta.mkv"
        final_writer = FFmpegWriter(
            out_name, orig_w, orig_h, fps, codec="ffv1")

        bin_writer = None
        if args.save_binary:
            bin_writer = FFmpegWriter(
                f"{input_path.stem}_binary.mkv",
                orig_w, orig_h, fps, codec="ffv1")

        lbl_dir = None
        if args.save_labels:
            lbl_dir = Path(f"{input_path.stem}_labels")
            lbl_dir.mkdir(parents=True, exist_ok=True)

        cap_orig = cv2.VideoCapture(str(input_path))
        mask_caps = {
            key: cv2.VideoCapture(str(path))
            for key, path in mask_video_paths.items()
        }

        buffer_data = []
        write_idx = 0
        read_idx = 0

        try:
            while write_idx < total_frames:

                # Fill buffer with enough future context
                while (read_idx <= write_idx + args.temporal
                       and read_idx < total_frames):
                    ret, frame_orig = cap_orig.read()
                    if not ret:
                        break

                    acc = np.zeros((orig_h, orig_w), dtype=np.uint8)

                    for (m_idx, ang), mcap in mask_caps.items():
                        ret_m, frame_m = mcap.read()
                        if not ret_m:
                            continue
                        mask_s = frame_m[:, :, 0]
                        if not np.any(mask_s):
                            continue
                        m_full = cv2.resize(
                            mask_s, (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST)
                        m_rot = rotate_frame(
                            m_full, -ang,
                            interpolation=cv2.INTER_NEAREST)
                        acc = cv2.bitwise_or(acc, m_rot)

                    buffer_data.append(
                        {'orig': frame_orig, 'mask': acc,
                         'idx': read_idx})
                    read_idx += 1

                # Find the frame we need to write
                target = next(
                    (x for x in buffer_data if x['idx'] == write_idx),
                    None)

                if target is None:
                    print(f"\n  Warning: frame {write_idx} missing from "
                          f"buffer. Stopping early.")
                    break

                # Temporal union
                t_lo = write_idx - args.temporal
                t_hi = write_idx + args.temporal
                temporal_acc = np.zeros((orig_h, orig_w), dtype=np.uint8)
                for item in buffer_data:
                    if t_lo <= item['idx'] <= t_hi:
                        temporal_acc = cv2.bitwise_or(
                            temporal_acc, item['mask'])

                # Hole filling
                final_mask = fill_holes_scipy(temporal_acc)

                # Save labels
                if lbl_dir is not None:
                    txt_p = lbl_dir / f"frame_{write_idx + 1:07d}.txt"
                    if np.any(final_mask):
                        polys = mask_to_yolo_poly(
                            final_mask, orig_w, orig_h)
                        with open(txt_p, 'w') as f:
                            f.write('\n'.join(polys))
                    else:
                        open(txt_p, 'w').close()

                # Binary video
                if bin_writer:
                    bin_writer.write(
                        cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))

                # Overlay video
                # FIX: Blend full frames first to avoid OpenCV errors on sliced arrays
                overlay = target['orig'].copy()
                green = np.zeros_like(overlay)
                green[:, :, 1] = 255

                # Blend (addWeighted works reliably on full contiguous frames)
                blended = cv2.addWeighted(overlay, 0.5, green, 0.5, 0)

                # Assign using numpy mask (works reliably on slices)
                m_bool = final_mask > 0
                overlay[m_bool] = blended[m_bool]

                final_writer.write(overlay)

                # Prune buffer
                cutoff = write_idx - args.temporal
                buffer_data[:] = [
                    x for x in buffer_data if x['idx'] >= cutoff]

                write_idx += 1
                if write_idx % 50 == 0:
                    print(f"  Aggregated {write_idx}/{total_frames}",
                          end='\r')

        finally:
            cap_orig.release()
            for c in mask_caps.values():
                c.release()
            final_writer.close()
            if bin_writer:
                bin_writer.close()

        print(f"\n  Done. Output: {out_name}")

    finally:
        # Cleanup
        if args.no_temp:
            print("Cleaning up temp files...")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        else:
            print(f"Temp files preserved at: {temp_dir}")


if __name__ == "__main__":
    main()
