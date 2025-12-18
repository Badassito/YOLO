"""
YOLO Segmentation with Test Time Augmentation (TTA) using 90°, 180°, 270° rotations. then merges masks using OR operation. Optional temporal smoothing using ± N frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from collections import deque
from pathlib import Path
from tqdm import tqdm


def rotate_image(image, angle):
    """Rotate image by 90, 180, or 270 degrees."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def rotate_mask_back(mask, angle):
    """Rotate mask back to original orientation."""
    if mask is None:
        return None
    if angle == 90:
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(mask, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    return mask


def get_combined_mask(results, shape):
    """Extract and combine all masks from YOLO results into a single binary mask."""
    h, w = shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for mask in masks:
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            combined = np.logical_or(combined, mask_resized > 0.5).astype(np.uint8)
    
    return combined


def run_tta_inference(model, frame, conf, retina_masks=True):
    """Run TTA inference with rotations and merge masks."""
    angles = [0, 90, 180, 270]
    h, w = frame.shape[:2]
    merged_mask = np.zeros((h, w), dtype=np.uint8)
    
    for angle in angles:
        rotated = rotate_image(frame, angle) if angle != 0 else frame
        
        results = model.predict(
            rotated,
            conf=conf,
            iou=1.0,
            retina_masks=retina_masks,
            verbose=False,
            # device='cpu'
        )
        
        mask = get_combined_mask(results, rotated.shape)
        
        if angle != 0:
            mask = rotate_mask_back(mask, angle)
        
        if mask is not None:
            merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8)
    
    return merged_mask


def merge_temporal_masks(masks):
    """OR together a list of masks."""
    result = masks[0].copy()
    for m in masks[1:]:
        result = np.logical_or(result, m).astype(np.uint8)
    return result


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay binary mask on frame."""
    overlay = frame.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def save_binary_mask(mask, path):
    """Save binary mask as image (0 and 255 values)."""
    cv2.imwrite(str(path), mask * 255)


def mask_to_yolo_polygons(mask):
    """
    Convert binary mask to YOLO polygon format.
    Returns list of normalized polygon strings for a single flattened object.
    Each contour becomes one annotation line.
    """
    h, w = mask.shape
    
    # Find contours of the flattened mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Simplify contour to reduce points
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Need at least 3 points for a polygon
        if len(contour) < 3:
            continue
        
        # Flatten and normalize coordinates
        points = contour.reshape(-1, 2)
        normalized = []
        for x, y in points:
            normalized.extend([x / w, y / h])
        
        # Format: class_id x1 y1 x2 y2 ...
        # Class 0 since single class
        poly_str = "0 " + " ".join(f"{coord:.6f}" for coord in normalized)
        polygons.append(poly_str)
    
    return polygons


def save_yolo_annotation(mask, path):
    """Save YOLO format annotation file from flattened mask."""
    polygons = mask_to_yolo_polygons(mask)
    with open(path, 'w') as f:
        f.write("\n".join(polygons))


def process_video(model_path, source, conf, imgsz, output_path, temporal_n=0,
                  save_txt=False, save_binary=False):
    """Process video with TTA and optional temporal smoothing."""
    model = YOLO(model_path)
    model.overrides['imgsz'] = imgsz
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")
    print(f"Model: {model_path}")
    print(f"TTA: 0°, 90°, 180°, 270° rotations with OR merge")
    if temporal_n > 0:
        print(f"Temporal: ±{temporal_n} frames ({2*temporal_n + 1} frame window)")
    print(f"Output: {output_path}")
    
    # Setup output directories
    output_base = Path(output_path).parent
    video_stem = Path(source).stem
    
    txt_dir = None
    binary_dir = None
    
    if save_txt:
        txt_dir = output_base / f"{video_stem}_labels"
        txt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Annotations: {txt_dir}")
    
    if save_binary:
        binary_dir = output_base / f"{video_stem}_masks"
        binary_dir.mkdir(parents=True, exist_ok=True)
        print(f"Binary masks: {binary_dir}")
    
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {output_path}")
    
    window_size = 2 * temporal_n + 1
    buffer = deque(maxlen=window_size)  # stores (frame, mask, frame_idx) tuples
    
    pbar = tqdm(total=total_frames, desc="Processing")
    frame_idx = 0
    output_idx = 0
    
    def save_outputs(frame, mask, idx):
        """Save video frame and optional txt/binary outputs."""
        vis_frame = overlay_mask(frame, mask)
        writer.write(vis_frame)
        
        frame_name = f"{idx:06d}"
        
        if save_txt:
            save_yolo_annotation(mask, txt_dir / f"{frame_name}.txt")
        
        if save_binary:
            save_binary_mask(mask, binary_dir / f"{frame_name}.png")
    
    # Read and process all frames, outputting when buffer is ready
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        mask = run_tta_inference(model, frame, conf, retina_masks=True)
        buffer.append((frame, mask, frame_idx))
        frame_idx += 1
        
        # Once buffer is full, start outputting center frames
        if len(buffer) == window_size:
            center_idx = temporal_n
            center_frame, center_mask, center_frame_idx = buffer[center_idx]
            
            if temporal_n > 0:
                temporal_mask = merge_temporal_masks([b[1] for b in buffer])
            else:
                temporal_mask = center_mask
            
            save_outputs(center_frame, temporal_mask, center_frame_idx)
            output_idx += 1
        
        pbar.update(1)
    
    # Flush remaining frames in buffer (handle end of video)
    # The center shifts right as we can no longer look ahead
    while len(buffer) > temporal_n:
        if temporal_n == 0:
            break
        buffer.popleft()
        if len(buffer) == 0:
            break
        
        # Center is now at min(temporal_n, len(buffer)-1)
        center_idx = min(temporal_n, len(buffer) - 1)
        center_frame, center_mask, center_frame_idx = buffer[center_idx]
        temporal_mask = merge_temporal_masks([b[1] for b in buffer])
        save_outputs(center_frame, temporal_mask, center_frame_idx)
        output_idx += 1
    
    pbar.close()
    cap.release()
    writer.release()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='YOLO Segmentation with Rotation TTA')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to video')
    parser.add_argument('--conf', type=float, default=0.30, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=1536, help='Inference image size')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--temporal', type=int, default=0, 
                        help='Temporal window: N frames before/after (0 = disabled)')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save YOLO format annotations (flattened to single object)')
    parser.add_argument('--save_binary', action='store_true',
                        help='Save binary mask images')
    
    args = parser.parse_args()
    
    process_video(
        args.model,
        args.source,
        args.conf,
        args.imgsz,
        args.output,
        args.temporal,
        args.save_txt,
        args.save_binary
    )


if __name__ == '__main__':
    main()
