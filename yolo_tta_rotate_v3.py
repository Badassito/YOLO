"""
YOLO Segmentation with Test Time Augmentation (TTA) using ffmpeg-based video rotation.

Workflow:
1. Pre-rotate input video to multiple angles using ffmpeg (h264_nvenc, lossless)
2. Run YOLO inference on each rotated video, saving txt labels
3. Read labels, rotate polygons back to original orientation, merge via OR

Designed for microscopy images with circular FOV (black corners).
"""

import subprocess
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
import argparse
from tqdm import tqdm
import tempfile
import shutil
import re


def get_video_info(video_path):
    """Get video frame count, dimensions, and fps using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets,width,height,r_frame_rate',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split(',')
    width, height = int(parts[0]), int(parts[1])

    # Parse frame rate (can be "30/1" format)
    fps_str = parts[2]
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    num_frames = int(parts[3])
    return width, height, fps, num_frames


def rotate_video_ffmpeg(input_path, output_path, angle, imgsz, gpu_device=0):
    """
    Rotate video using ffmpeg with NVENC hardware acceleration.

    For circular FOV microscopy: maintains square frame, crops overflow,
    pads empty regions with black.

    Args:
        input_path: Input video path
        output_path: Output rotated video path
        angle: Rotation angle in degrees (counter-clockwise)
        imgsz: Output size (square)
        gpu_device: GPU device index for nvenc
    """
    angle_rad = angle * np.pi / 180

    # Filter chain:
    # 1. Scale to fit within imgsz while maintaining aspect ratio
    # 2. Pad to exact imgsz x imgsz with black
    # 3. Rotate by angle, filling empty corners with black
    #    rotate filter outputs same dimensions as input by default
    vf_filters = [
        f"scale={imgsz}:{imgsz}:force_original_aspect_ratio=decrease",
        f"pad={imgsz}:{imgsz}:(ow-iw)/2:(oh-ih)/2:black",
    ]

    # Only add rotation filter if angle != 0
    if angle != 0:
        vf_filters.append(f"rotate={angle_rad}:fillcolor=black")

    vf = ','.join(vf_filters)

    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-hwaccel_device', str(gpu_device),
        '-i', str(input_path),
        '-vf', vf,
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',           # Slowest/highest quality preset
        '-tune', 'lossless',       # Lossless encoding mode
        '-pix_fmt', 'yuv444p',     # No chroma subsampling for lossless
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try fallback without lossless tune (older drivers)
        cmd_fallback = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_device),
            '-i', str(input_path),
            '-vf', vf,
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',
            '-rc', 'constqp',
            '-qp', '0',            # Constant QP=0 (near-lossless)
            '-pix_fmt', 'yuv444p',
            str(output_path)
        ]
        subprocess.run(cmd_fallback, check=True, capture_output=True)


def run_yolo_inference(model, video_path, output_dir, conf=0.3, imgsz=1536, device='0'):
    """
    Run YOLO inference on video, saving txt labels.

    Equivalent to CLI: yolo model=... source=... retina_masks=True iou=1.0 save=False save_txt=True

    Returns:
        Path to directory containing label txt files
    """
    results = model.predict(
        source=str(video_path),
        conf=conf,
        iou=1.0,
        retina_masks=True,
        save=False,
        save_txt=True,
        project=str(output_dir),
        name='predict',
        exist_ok=True,
        device=device,
        half=True,
        imgsz=imgsz,
        verbose=False,
        stream=True
    )

    # YOLO saves labels in project/name/labels/
    return output_dir / 'predict' / 'labels'


def rotate_polygon_back(polygon, angle):
    """
    Rotate normalized polygon coordinates back to original orientation.

    Args:
        polygon: List of (x, y) normalized coordinates [0, 1]
        angle: Original rotation angle in degrees (counter-clockwise)

    Returns:
        Rotated polygon coordinates (normalized)
    """
    if angle == 0:
        return polygon

    # Convert to radians (negative to rotate back)
    theta = -angle * np.pi / 180
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Center is at (0.5, 0.5) in normalized coords
    cx, cy = 0.5, 0.5

    rotated = []
    for x, y in polygon:
        # Translate to center, rotate, translate back
        dx, dy = x - cx, y - cy
        new_x = dx * cos_t - dy * sin_t + cx
        new_y = dx * sin_t + dy * cos_t + cy
        rotated.append((new_x, new_y))

    return rotated


def parse_yolo_label(label_path):
    """
    Parse YOLO segmentation label file.

    Returns:
        List of (class_id, polygon) where polygon is list of (x, y) normalized coords
    """
    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 points (6 coords)
                continue

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Parse as (x, y) pairs
            polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            objects.append((class_id, polygon))

    return objects


def polygon_to_mask(polygon, img_size):
    """Convert normalized polygon to binary mask."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Convert normalized to pixel coordinates
    pts = np.array([(x * img_size, y * img_size) for x, y in polygon], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_yolo_polygons(mask, class_id=0):
    """Convert binary mask to YOLO polygon format strings."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Simplify contour
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(contour) < 3:
            continue

        points = contour.reshape(-1, 2)
        normalized = []
        for x, y in points:
            normalized.extend([x / w, y / h])

        poly_str = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized)
        polygons.append(poly_str)

    return polygons


def find_label_files(label_dir):
    """
    Find all label files and extract frame indices.

    YOLO names files like: video_name_0.txt, video_name_1.txt, ...
    or frame_0.txt, etc.

    Returns:
        Dict mapping frame_index -> label_path
    """
    label_files = {}

    if not label_dir.exists():
        return label_files

    for path in label_dir.glob('*.txt'):
        # Extract frame number from filename
        # Try to find trailing number before .txt
        match = re.search(r'(\d+)\.txt$', path.name)
        if match:
            frame_idx = int(match.group(1))
            label_files[frame_idx] = path

    return label_files


def merge_frame_labels(label_dirs, angles, frame_idx, imgsz):
    """
    Merge labels from multiple rotations for a single frame.

    Args:
        label_dirs: List of dicts mapping frame_idx -> label_path for each rotation
        angles: List of rotation angles
        frame_idx: Frame index to process
        imgsz: Image size for mask generation

    Returns:
        Merged binary mask
    """
    merged_mask = np.zeros((imgsz, imgsz), dtype=np.uint8)

    for label_files, angle in zip(label_dirs, angles):
        if frame_idx not in label_files:
            continue

        label_path = label_files[frame_idx]
        objects = parse_yolo_label(label_path)

        for class_id, polygon in objects:
            # Rotate polygon back to original orientation
            rotated_poly = rotate_polygon_back(polygon, angle)

            # Clip coordinates to valid range [0, 1]
            rotated_poly = [(max(0, min(1, x)), max(0, min(1, y))) for x, y in rotated_poly]

            # Convert to mask and OR merge
            obj_mask = polygon_to_mask(rotated_poly, imgsz)
            merged_mask = np.logical_or(merged_mask, obj_mask).astype(np.uint8)

    return merged_mask


def save_binary_mask(mask, path):
    """Save binary mask as image (0 and 255 values)."""
    cv2.imwrite(str(path), mask * 255)


def save_yolo_annotation(mask, path):
    """Save YOLO format annotation file from mask."""
    polygons = mask_to_yolo_polygons(mask)
    with open(path, 'w') as f:
        f.write("\n".join(polygons))


def create_overlay_video(source_video, masks_dir, output_video, fps, imgsz,
                         color=(0, 255, 0), alpha=0.5):
    """Create video with mask overlay for visualization."""
    cap = cv2.VideoCapture(str(source_video))

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (imgsz, imgsz))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to match mask size
        frame = cv2.resize(frame, (imgsz, imgsz))

        # Load mask
        mask_path = masks_dir / f"{frame_idx:06d}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)

            # Overlay
            overlay = frame.copy()
            overlay[mask > 0] = color
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def process_video(model_path, source, conf, imgsz, output_dir, angles, device,
                  save_video=True):
    """
    Main processing pipeline.

    1. Rotate video to each angle using ffmpeg
    2. Run YOLO inference on each rotated video
    3. Merge results from all rotations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(source)
    width, height, fps, num_frames = get_video_info(source)

    print(f"Video: {source.name} - {width}x{height} @ {fps:.2f}fps, {num_frames} frames")
    print(f"Model: {model_path}")
    print(f"TTA angles: {angles}")
    print(f"Output size: {imgsz}x{imgsz}")
    print(f"Output: {output_dir}")

    # Load model once
    model = YOLO(model_path)
    model.overrides['imgsz'] = imgsz

    # Create temp directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix='yolo_tta_'))
    print(f"Temp dir: {temp_dir}")

    try:
        all_label_files = []  # List of dicts: frame_idx -> label_path

        for angle in angles:
            print(f"\n{'='*50}")
            print(f"Processing angle {angle}°")
            print('='*50)

            # Step 1: Rotate video
            rotated_video = temp_dir / f"rotated_{angle}.mp4"
            print(f"Rotating video...")
            rotate_video_ffmpeg(source, rotated_video, angle, imgsz, gpu_device=device)
            print(f"  -> {rotated_video}")

            # Step 2: Run inference
            inference_out = temp_dir / f"inference_{angle}"
            print(f"Running inference...")
            label_dir = run_yolo_inference(
                model, rotated_video, inference_out,
                conf=conf, imgsz=imgsz, device=str(device)
            )
            print(f"  -> {label_dir}")

            # Index label files
            label_files = find_label_files(label_dir)
            print(f"  Found {len(label_files)} label files")
            all_label_files.append(label_files)

        # Step 3: Merge results
        print(f"\n{'='*50}")
        print("Merging results")
        print('='*50)

        labels_out = output_dir / 'labels'
        masks_out = output_dir / 'masks'
        labels_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

        # Determine frame indices to process
        all_frame_indices = set()
        for label_files in all_label_files:
            all_frame_indices.update(label_files.keys())

        if not all_frame_indices:
            # No detections in any rotation, create empty outputs for all frames
            all_frame_indices = set(range(num_frames))

        for frame_idx in tqdm(sorted(all_frame_indices), desc="Merging"):
            merged_mask = merge_frame_labels(all_label_files, angles, frame_idx, imgsz)

            # Save outputs
            save_binary_mask(merged_mask, masks_out / f"{frame_idx:06d}.png")
            save_yolo_annotation(merged_mask, labels_out / f"{frame_idx:06d}.txt")

        print(f"\nLabels saved to: {labels_out}")
        print(f"Masks saved to: {masks_out}")

        # Optional: Create overlay video
        if save_video:
            print("\nCreating overlay video...")
            video_out = output_dir / f"{source.stem}_overlay.avi"
            create_overlay_video(source, masks_out, video_out, fps, imgsz)
            print(f"Video saved to: {video_out}")

    finally:
        # Cleanup temp directory
        print(f"\nCleaning up temp files...")
        shutil.rmtree(temp_dir)

    print("\nDone!")


def parse_angles(angles_str):
    """Parse angles from comma-separated string."""
    return [float(a.strip()) for a in angles_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Segmentation TTA with FFmpeg-based video rotation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 90° rotations
  python %(prog)s --model best.pt --source video.mp4 --output ./output --angles 0,90,180,270

  # 45° increments (8 angles)
  python %(prog)s --model best.pt --source video.mp4 --output ./output --angles 0,45,90,135,180,225,270,315

  # Higher resolution inference
  python %(prog)s --model best.pt --source video.mp4 --output ./output --imgsz 2048
        """
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--conf', type=float, default=0.30,
                        help='Confidence threshold (default: 0.30)')
    parser.add_argument('--imgsz', type=int, default=1536,
                        help='Inference image size, will be square (default: 1536)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for labels and masks')
    parser.add_argument('--angles', type=str, default='0,90,180,270',
                        help='Comma-separated rotation angles in degrees (default: 0,90,180,270)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device index (default: 0)')
    parser.add_argument('--no-video', action='store_true',
                        help='Skip creating overlay video')

    args = parser.parse_args()
    angles = parse_angles(args.angles)

    process_video(
        args.model,
        args.source,
        args.conf,
        args.imgsz,
        args.output,
        angles,
        args.device,
        save_video=not args.no_video
    )


if __name__ == '__main__':
    main()
