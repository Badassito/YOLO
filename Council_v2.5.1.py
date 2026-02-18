#!/usr/bin/env python3
"""
YOLO Segmentation TTA — Performance-Optimized
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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


def ffmpeg_pipe(output_path, w, h, fps, mode="final"):
    in_pix = "gray" if mode == "mask" else "bgr24"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", in_pix,
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
    ]
    if mode == "temp":
        cmd += ["-c:v", "h264_nvenc", "-preset", "lossless",
                "-pix_fmt", "yuv444p", str(output_path)]
    elif mode == "final":
        cmd += ["-c:v", "ffv1", "-pix_fmt", "bgr24", str(output_path)]
    elif mode == "mask":
        cmd += ["-c:v", "ffv1", "-pix_fmt", "gray", str(output_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)


class AugTransform:
    def __init__(self, angle, sx, sy, orig_w, orig_h, imgsz):
        self.angle = angle
        self.sx, self.sy = sx, sy
        self.ow, self.oh = orig_w, orig_h
        self.sz = imgsz
        self.center = (orig_w / 2.0, orig_h / 2.0)

        # Pre-compute rotation matrix for forward pass
        self.M_rot = cv2.getRotationMatrix2D(self.center, angle, 1.0)

        # Pre-compute unified inverse matrix
        M_rot_3x3 = np.vstack([self.M_rot, [0, 0, 1]])
        M_scale = np.array([
            [imgsz / orig_w, 0, 0],
            [0, imgsz / orig_h, 0],
            [0, 0, 1]
        ])
        M_shift = np.array([
            [1, 0, sx],
            [0, 1, sy],
            [0, 0, 1]
        ])
        self.M_inv = np.linalg.inv(M_shift @ M_scale @ M_rot_3x3)[:2, :]

    def forward(self, frame):
        # Multi-step preserves original corner-clamping behavior
        rot = cv2.warpAffine(frame, self.M_rot, (self.ow, self.oh),
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
        # Single-pass via pre-computed matrix
        return cv2.warpAffine(mask, self.M_inv, (self.ow, self.oh),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)


def build_augmentations(angles, shift, orig_w, orig_h, imgsz):
    if shift == 0:
        shift_pairs = [(0, 0)]
    else:
        shift_pairs = [
            (0, 0), (0, -shift), (0, shift), (-shift, 0), (shift, 0),
        ]
    augs = []
    for a in angles:
        for sx, sy in shift_pairs:
            augs.append(AugTransform(a, sx, sy, orig_w, orig_h, imgsz))
    return augs


def fill_holes(mask):
    inv = cv2.bitwise_not(mask)
    h, w = inv.shape
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if inv[seed[1], seed[0]] == 255:
            cv2.floodFill(inv, ff_mask, seed, 0)
    return cv2.bitwise_or(mask, inv)


def warp_mask(src_frame, dst_frame, src_mask):
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


def main():
    args = parse_args()

    # Resolve model paths
    model_paths = []
    for m in args.model:
        for part in m.split(","):
            part = part.strip()
            if part:
                model_paths.append(part)

    # Paths
    input_path = Path(args.input).resolve()
    stem = input_path.stem
    input_str = str(input_path)

    if args.output is None:
        out_dir = Path(f"./{stem}")
    else:
        out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = out_dir / "temp"
    label_dir = out_dir / "labels"
    binary_dir = out_dir / "binary_masks"
    if args.save_labels:
        label_dir.mkdir(exist_ok=True)
    if args.save_binary:
        binary_dir.mkdir(exist_ok=True)

    # Video info
    cap = cv2.VideoCapture(input_str)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_str}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # ══════════════════════════════════════════════════════════
    # STEP 1: Cache input frames (single decode)
    # ══════════════════════════════════════════════════════════
    print("[INFO] Caching input frames...")
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    n_frames = len(raw_frames)

    mem_gb = n_frames * orig_w * orig_h * 3 / (1024 ** 3)
    print(f"[INFO] {orig_w}x{orig_h} @ {fps:.2f} fps, "
          f"{n_frames} frames ({mem_gb:.1f} GB RAM)")
    if mem_gb > 16:
        print("[WARN] High memory usage — ensure sufficient RAM")

    # Build augmentations
    angles = [float(a.strip()) for a in args.angle.split(",")]
    augs = build_augmentations(angles, args.shift,
                               orig_w, orig_h, args.imgsz)
    print(f"[INFO] {len(augs)} augmentation variant(s), "
          f"{len(model_paths)} model(s)")

    # ══════════════════════════════════════════════════════════
    # STEP 1b: Write temp videos only if --keep_temp
    # ══════════════════════════════════════════════════════════
    if args.keep_temp:
        temp_dir.mkdir(exist_ok=True)
        print("[INFO] Writing temp videos...")
        for idx, aug in enumerate(augs):
            t_path = temp_dir / f"aug_{idx}.mp4"
            writer = ffmpeg_pipe(t_path, args.imgsz, args.imgsz,
                                 fps, "temp")
            for frame in raw_frames:
                writer.stdin.write(aug.forward(frame).tobytes())
            writer.stdin.close()
            writer.wait()
            print(f"  Variant {idx + 1}/{len(augs)} "
                  f"(angle={aug.angle}, shift=({aug.sx},{aug.sy}))")

    # ══════════════════════════════════════════════════════════
    # STEP 2-3: Inference via generator (no disk I/O)
    # ══════════════════════════════════════════════════════════
    union_masks = [np.zeros((orig_h, orig_w), dtype=np.uint8)
                   for _ in range(n_frames)]
    max_confs = [0.0] * n_frames

    for m_path in model_paths:
        print(f"[INFER] Model: {m_path}")
        model = YOLO(m_path)

        # Warmup once per model (not per variant)
        model.predict(
            source=np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
            task="segment", imgsz=args.imgsz,
            device=args.device, half=args.half, int8=args.int8,
            verbose=False,
        )

        for v_idx, aug in enumerate(augs):
            # Default arg captures aug by value
            def _aug_gen(_a=aug):
                for _f in raw_frames:
                    yield _a.forward(_f)

            results = model.predict(
                source=_aug_gen(),
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

                    masks_np = r.masks.data.cpu().numpy()
                    var_union = (masks_np.max(axis=0) > 0.5
                                 ).astype(np.uint8) * 255

                    if var_union.shape != (args.imgsz, args.imgsz):
                        var_union = cv2.resize(
                            var_union, (args.imgsz, args.imgsz),
                            interpolation=cv2.INTER_NEAREST)

                    orig_mask = aug.inverse(var_union)
                    union_masks[fi] = cv2.bitwise_or(
                        union_masks[fi], orig_mask)
                fi += 1

            print(f"  Variant {v_idx + 1}/{len(augs)} done "
                  f"(angle={aug.angle}, shift=({aug.sx},{aug.sy}))")

    # ══════════════════════════════════════════════════════════
    # STEP 4-5: Hole filling
    # ══════════════════════════════════════════════════════════
    print("[POST] Filling holes...")
    for i in range(n_frames):
        if np.any(union_masks[i]):
            union_masks[i] = fill_holes(union_masks[i])

    # ══════════════════════════════════════════════════════════
    # STEP 6: Temporal smoothing
    # ══════════════════════════════════════════════════════════
    print("[POST] Temporal smoothing...")
    high_thresh = args.conf * 2.0
    N = args.temporal

    # Snapshot confidences to prevent cascade during drop pass
    orig_confs = list(max_confs)
    final_masks = [m.copy() for m in union_masks]
    working_confs = list(max_confs)

    # Pass 1: Drop isolated low-confidence detections
    for i in range(n_frames):
        c = orig_confs[i]
        if c <= 0 or c >= high_thresh:
            continue
        found_neighbor = False
        for d in range(1, N + 1):
            if (i - d >= 0 and orig_confs[i - d] > 0) or \
               (i + d < n_frames and orig_confs[i + d] > 0):
                found_neighbor = True
                break
        if not found_neighbor:
            final_masks[i] = np.zeros((orig_h, orig_w), dtype=np.uint8)
            working_confs[i] = 0.0

    # Pass 2: Interpolation and extrapolation (deferred updates)
    updates = {}
    for i in range(n_frames):
        if working_confs[i] >= high_thresh:
            continue

        back_ok = (i >= N) and all(
            working_confs[i - k] >= high_thresh
            for k in range(1, N + 1))
        fwd_ok = (i + N < n_frames) and all(
            working_confs[i + k] >= high_thresh
            for k in range(1, N + 1))

        def get_warped(src_idx, dst_idx):
            return warp_mask(raw_frames[src_idx], raw_frames[dst_idx],
                             final_masks[src_idx])

        if back_ok and fwd_ok:
            w_prev = get_warped(i - 1, i)
            w_next = get_warped(i + 1, i)
            updates[i] = cv2.bitwise_or(w_prev, w_next)
        elif back_ok and working_confs[i] == 0.0:
            updates[i] = get_warped(i - 1, i)
        elif fwd_ok and working_confs[i] == 0.0:
            updates[i] = get_warped(i + 1, i)

    for i, m in updates.items():
        final_masks[i] = m
        working_confs[i] = args.conf

    # ══════════════════════════════════════════════════════════
    # STEP 7: Generate outputs
    # ══════════════════════════════════════════════════════════
    print("[OUTPUT] Rendering...")
    final_vid_path = out_dir / f"{stem}_final.mkv"
    vid_pipe = ffmpeg_pipe(final_vid_path, orig_w, orig_h, fps, "final")

    bin_pipe = None
    if args.save_binary:
        bin_vid_path = out_dir / f"{stem}_Binary.mkv"
        bin_pipe = ffmpeg_pipe(bin_vid_path, orig_w, orig_h, fps, "mask")

    blue_layer = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    blue_layer[:] = (255, 0, 0)

    for i in range(n_frames):
        frame = raw_frames[i]
        mask = final_masks[i]

        if args.save_labels:
            txt_path = label_dir / f"{stem}_{i:04d}.txt"
            lines = mask_to_yolo_lines(mask, orig_w, orig_h)
            with open(txt_path, "w") as f:
                for line in lines:
                    f.write(line + "\n")

        if args.save_binary:
            bin_pipe.stdin.write(mask.tobytes())
            tiff_path = binary_dir / f"{stem}_Binary_{i:04d}.tiff"
            cv2.imwrite(str(tiff_path), mask)

        overlay = frame.copy()
        roi = mask > 0
        if roi.any():
            overlay[roi] = cv2.addWeighted(
                frame[roi], 0.5, blue_layer[roi], 0.5, 0)
        vid_pipe.stdin.write(overlay.tobytes())

        if (i + 1) % 100 == 0 or i == n_frames - 1:
            print(f"  Frame {i + 1}/{n_frames}", end="\r")

    vid_pipe.stdin.close()
    vid_pipe.wait()
    if bin_pipe is not None:
        bin_pipe.stdin.close()
        bin_pipe.wait()

    if not args.keep_temp and temp_dir.exists():
        shutil.rmtree(temp_dir)

    print(f"\n[DONE] Output in {out_dir}")


if __name__ == "__main__":
    main()
