#!/usr/bin/env python3
"""
Test Time Augmentation for YOLO Segmentation using rotation.

Rotates input video to multiple angles, runs inference on each,
then fuses the segmentation masks back together.
"""

import argparse
import tempfile
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="TTA for YOLO segmentation with rotation augmentation"
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument(
        "--angles",
        default="0,90,180,270",
        help="Comma-separated rotation angles (default: 0,90,180,270)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for inference"
    )
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument(
        "--save_labels", type=str, default=None, help="Directory to save flattened labels"
    )
    parser.add_argument(
        "--save_binary", type=str, default=None, help="Path to save binary mask video"
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.5,
        help="Alpha for mask overlay (default: 0.5)",
    )
    parser.add_argument(
        "--mask_color",
        type=str,
        default="0,255,0",
        help="BGR color for mask overlay (default: 0,255,0 = green)",
    )
    return parser.parse_args()


def get_video_info(video_path: str) -> tuple[int, int, float, int]:
    """Get video width, height, fps, and frame count."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count


def rotate_video_ffmpeg(
    input_path: str, output_path: str, angle: int, imgsz: int
) -> None:
    """Rotate and scale video using ffmpeg with h264_nvenc lossless."""
    # Build filter chain
    filters = []

    # Rotation using transpose
    if angle == 90:
        filters.append("transpose=1")  # 90 CW
    elif angle == 180:
        filters.append("transpose=1,transpose=1")  # 180
    elif angle == 270:
        filters.append("transpose=2")  # 90 CCW (270 CW)
    # angle == 0: no rotation filter needed

    # Scale to imgsz (maintain aspect ratio, scale to fit)
    filters.append(f"scale={imgsz}:{imgsz}:force_original_aspect_ratio=decrease")
    # Pad to exact imgsz x imgsz if needed (center the content)
    filters.append(f"pad={imgsz}:{imgsz}:(ow-iw)/2:(oh-ih)/2:black")

    filter_str = ",".join(filters) if filters else "null"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        filter_str,
        "-c:v",
        "ffv1",
        "-an",  # No audio
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True)


def undo_rotation_polygon(
    polygon: np.ndarray, angle: int, rotated_w: int, rotated_h: int
) -> np.ndarray:
    """
    Undo rotation on polygon coordinates.

    polygon: Nx2 array of (x, y) normalized coordinates [0, 1]
    angle: the angle the video was rotated BY (so we undo it)
    rotated_w, rotated_h: dimensions of the rotated video

    Returns: Nx2 array in original orientation
    """
    if angle == 0:
        return polygon

    # Work with a copy
    poly = polygon.copy()

    if angle == 90:
        # Video was rotated 90 CW, so to undo:
        # (x', y') in rotated -> (y', 1 - x') in original
        new_x = poly[:, 1]
        new_y = 1.0 - poly[:, 0]
        poly = np.stack([new_x, new_y], axis=1)

    elif angle == 180:
        # Undo 180: (x', y') -> (1 - x', 1 - y')
        poly = 1.0 - poly

    elif angle == 270:
        # Video was rotated 270 CW (90 CCW), to undo:
        # (x', y') -> (1 - y', x')
        new_x = 1.0 - poly[:, 1]
        new_y = poly[:, 0]
        poly = np.stack([new_x, new_y], axis=1)

    return poly


def parse_yolo_seg_label(label_path: Path) -> list[np.ndarray]:
    """
    Parse YOLO segmentation label file.

    Returns list of polygons, each as Nx2 numpy array of normalized coordinates.
    """
    polygons = []
    if not label_path.exists():
        return polygons

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 points (6 coords)
                continue
            # Skip class id, rest are x y x y x y ...
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue
            points = np.array(coords).reshape(-1, 2)
            polygons.append(points)

    return polygons


def polygons_to_mask(
    polygons: list[np.ndarray], width: int, height: int
) -> np.ndarray:
    """Convert list of normalized polygons to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in polygons:
        # Convert normalized coords to pixel coords
        pixel_coords = poly.copy()
        pixel_coords[:, 0] *= width
        pixel_coords[:, 1] *= height
        pixel_coords = pixel_coords.astype(np.int32)

        cv2.fillPoly(mask, [pixel_coords], 255)

    return mask


def mask_to_polygons_normalized(
    mask: np.ndarray, epsilon_factor: float = 0.001
) -> list[np.ndarray]:
    """
    Convert binary mask back to normalized polygon coordinates.

    Uses contour detection and approximation.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = mask.shape
    polygons = []

    for contour in contours:
        # Approximate contour to reduce points
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # Reshape from (N, 1, 2) to (N, 2) and normalize
        points = approx.reshape(-1, 2).astype(float)
        points[:, 0] /= width
        points[:, 1] /= height

        polygons.append(points)

    return polygons


def save_labels_yolo_format(
    labels_dir: Path, frame_idx: int, polygons: list[np.ndarray], class_id: int = 0
) -> None:
    """Save polygons in YOLO segmentation format."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"frame_{frame_idx:06d}.txt"

    with open(label_path, "w") as f:
        for poly in polygons:
            coords = poly.flatten()
            coords_str = " ".join(f"{c:.6f}" for c in coords)
            f.write(f"{class_id} {coords_str}\n")


def run_inference_and_collect_labels(
    model: YOLO, video_path: str, angle: int, imgsz: int
) -> dict[int, list[np.ndarray]]:
    """
    Run inference on video and return per-frame polygons with rotation undone.

    Returns: dict mapping frame_idx -> list of polygons (normalized coords, original orientation)
    """
    # Get rotated video dimensions for coordinate transformation
    rot_w, rot_h, _, _ = get_video_info(video_path)

    frame_polygons = defaultdict(list)

    # Run inference with streaming
    results = model.predict(
        source=video_path,
        save=False,
        save_txt=False,  # We'll handle labels ourselves
        stream=True,
        iou=1.0,
        conf=0.10,
        imgsz=imgsz,
        verbose=False,
    )

    for frame_idx, result in enumerate(results):
        if result.masks is None:
            continue

        # Get mask polygons from result
        # result.masks.xyn gives normalized polygon coordinates
        for mask_poly in result.masks.xyn:
            if len(mask_poly) < 3:
                continue

            poly = np.array(mask_poly)

            # Undo rotation
            poly_original = undo_rotation_polygon(poly, angle, rot_w, rot_h)

            frame_polygons[frame_idx].append(poly_original)

    return frame_polygons


def main():
    args = parse_args()

    # Parse arguments
    input_path = Path(args.input)
    output_path = Path(args.output)
    angles = [int(a.strip()) for a in args.angles.split(",")]
    mask_color = tuple(int(c) for c in args.mask_color.split(","))

    # Validate angles
    valid_angles = {0, 90, 180, 270}
    for angle in angles:
        if angle not in valid_angles:
            raise ValueError(f"Invalid angle {angle}. Must be one of {valid_angles}")

    # Get original video info
    orig_w, orig_h, fps, frame_count = get_video_info(str(input_path))
    print(f"Input video: {orig_w}x{orig_h}, {fps:.2f} fps, {frame_count} frames")

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Create temp directory for rotated videos
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1 & 2: Create rotated videos and run inference
        all_frame_polygons = defaultdict(list)  # frame_idx -> list of all polygons

        for angle in angles:
            print(f"\nProcessing angle: {angle}°")

            # Create rotated video
            rotated_path = tmpdir / f"rotated_{angle}.mp4"
            print(f"  Creating rotated video...")
            rotate_video_ffmpeg(str(input_path), str(rotated_path), angle, args.imgsz)

            # Run inference
            print(f"  Running inference...")
            frame_polygons = run_inference_and_collect_labels(
                model, str(rotated_path), angle, args.imgsz
            )

            # Merge into global collection
            for frame_idx, polygons in frame_polygons.items():
                all_frame_polygons[frame_idx].extend(polygons)

            print(f"  Found detections in {len(frame_polygons)} frames")

        # Step 3: Generate output video(s)
        print(f"\nGenerating output video...")

        # Open input video for reading frames
        cap = cv2.VideoCapture(str(input_path))

        # Setup ffmpeg process for main output
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{orig_w}x{orig_h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "ffv1",
            str(output_path),
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        # Setup binary mask output if requested
        binary_proc = None
        if args.save_binary:
            binary_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{orig_w}x{orig_h}",
                "-pix_fmt",
                "gray",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "ffv1",
                args.save_binary,
            ]
            binary_proc = subprocess.Popen(binary_cmd, stdin=subprocess.PIPE)

        # Setup labels directory if requested
        labels_dir = Path(args.save_labels) if args.save_labels else None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get polygons for this frame
            polygons = all_frame_polygons.get(frame_idx, [])

            # Create fused mask
            fused_mask = polygons_to_mask(polygons, orig_w, orig_h)

            # Create overlay frame
            overlay = frame.copy()
            overlay[fused_mask > 0] = mask_color
            output_frame = cv2.addWeighted(
                frame, 1 - args.overlay_alpha, overlay, args.overlay_alpha, 0
            )

            # Write to main output
            ffmpeg_proc.stdin.write(output_frame.tobytes())

            # Write binary mask if requested
            if binary_proc:
                binary_proc.stdin.write(fused_mask.tobytes())

            # Save labels if requested
            if labels_dir and polygons:
                # Convert mask back to polygons for cleaner output
                final_polygons = mask_to_polygons_normalized(fused_mask)
                save_labels_yolo_format(labels_dir, frame_idx, final_polygons)

            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{frame_count} frames")

        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

        if binary_proc:
            binary_proc.stdin.close()
            binary_proc.wait()

    print(f"\nDone! Output saved to: {output_path}")
    if args.save_binary:
        print(f"Binary masks saved to: {args.save_binary}")
    if args.save_labels:
        print(f"Labels saved to: {args.save_labels}")


if __name__ == "__main__":
    main()
