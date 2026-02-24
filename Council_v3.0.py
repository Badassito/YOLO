import argparse
import os
import shutil
import cv2
import numpy as np
import torch
import gc
import tifffile
import math
from pathlib import Path
from scipy import ndimage
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Argument Parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Segmentation TTA v3.0 Final")
    parser.add_argument("--input", type=str, required=True, help="Input video file")
    parser.add_argument("--output", type=str, default=None, help="Output directory root")
    parser.add_argument("--model", type=str, required=True, help="Comma-separated model paths")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--int8", action="store_true", help="Use int8 precision")
    parser.add_argument("--half", action="store_true", help="Use half precision")
    parser.add_argument("--disable_multiplanar", action="store_true", help="Only Transverse view")
    parser.add_argument("--angle", type=str, default="0,120,240", help="Rotation angles")
    parser.add_argument("--imgsz", type=int, default=1536, help="Inference size")
    parser.add_argument("--shift", type=int, default=0, help="Pixel shift value")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--temporal", type=int, default=1, help="Temporal window N")
    parser.add_argument("--save_labels", action="store_true", help="Save YOLO labels")
    parser.add_argument("--save_binary", action="store_true", help="Save binary masks")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")

    args = parser.parse_args()

    args.input_stem = Path(args.input).stem
    if args.output is None:
        args.output = Path(f"./{args.input_stem}/")
    else:
        args.output = Path(args.output)

    args.model_list = [x.strip() for x in args.model.split(',')]
    args.angle_list = [float(x) for x in args.angle.split(',')]

    return args

# -----------------------------------------------------------------------------
# 2. Affine Transformation Logic
# -----------------------------------------------------------------------------

def get_affine_matrix(src_shape, imgsz, angle, shift_val, shift_dir_idx):
    """
    Constructs the 2x3 Affine Matrix.
    Order: Center->Origin, Rotate, Scale, Origin->DstCenter, Shift
    """
    h, w = src_shape
    src_cx, src_cy = w / 2.0, h / 2.0
    dst_cx, dst_cy = imgsz / 2.0, imgsz / 2.0

    T_origin = np.array([[1, 0, -src_cx], [0, 1, -src_cy], [0, 0, 1]])

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    scale_x, scale_y = imgsz / w, imgsz / h
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

    T_dst = np.array([[1, 0, dst_cx], [0, 1, dst_cy], [0, 0, 1]])

    shifts = [
        (0,0),
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (1, -1), (-1, 1), (1, 1)
    ]
    dx, dy = shifts[shift_dir_idx]

    T_shift = np.array([[1, 0, dx * shift_val], [0, 1, dy * shift_val], [0, 0, 1]])

    M_final = T_shift @ T_dst @ S @ R @ T_origin
    M_inv = np.linalg.inv(M_final)

    return M_final[:2, :], M_inv[:2, :]

# -----------------------------------------------------------------------------
# 3. Filtering Logic (Mathematically Perfected)
# -----------------------------------------------------------------------------

def filter_straight_edges(mask_imgsz_space, threshold=1999):
    """
    Delete masks with a straight horizontal or vertical edge >= 30px in --imgsz space.
    Correctly accounts for contour wrap-around via array doubling.
    """
    contours, _ = cv2.findContours(mask_imgsz_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask_imgsz_space

    valid_mask = np.zeros_like(mask_imgsz_space)

    def max_run_wrap(diffs):
        if len(diffs) == 0: return 0
        is_zero = diffs == 0
        doubled = np.concatenate([is_zero, is_zero])
        max_r = current = 0
        cap = len(is_zero)
        for v in doubled:
            if v:
                current += 1
                if current > cap: return cap
                max_r = max(max_r, current)
            else:
                current = 0
        return max_r

    for cnt in contours:
        pts = cnt[:, 0, :]
        if len(pts) < threshold:
            cv2.drawContours(valid_mask, [cnt], -1, 255, -1)
            continue

        # Close the contour explicitly to calculate the final segment difference
        pts_closed = np.vstack([pts, pts[0]])

        # >= threshold (e.g., 30 zeros = 31 pixels long)
        if max_run_wrap(np.diff(pts_closed[:, 1])) >= threshold: continue
        if max_run_wrap(np.diff(pts_closed[:, 0])) >= threshold: continue

        cv2.drawContours(valid_mask, [cnt], -1, 255, -1)

    return valid_mask

def fill_2d_holes(mask):
    mask_padded = np.pad(mask, 1, mode='constant', constant_values=0)
    flood = mask_padded.copy()
    cv2.floodFill(flood, None, (0,0), 255)
    return (mask_padded | cv2.bitwise_not(flood))[1:-1, 1:-1]

def fill_3d_holes_multiplanar(mask_vol):
    """
    Fill 3D holes by applying 2D hole fill along all three axes.
    Completes in minutes, avoids Scipy's 72GB+ int32 RAM spike.
    Incorporates `np.maximum` and full boolean guards.
    """
    T, H, W = mask_vol.shape

    print("      Pass 1/3: Transverse (XY)...")
    for t in tqdm(range(T), desc="        T-axis", leave=False):
        slc = mask_vol[t]
        if np.any(slc) and not np.all(slc):
            filled = ndimage.binary_fill_holes(slc > 0)
            mask_vol[t] = np.maximum(slc, (filled * 255).astype(np.uint8))

    print("      Pass 2/3: Sagittal (Xt)...")
    for y in tqdm(range(H), desc="        Y-axis", leave=False):
        slc = mask_vol[:, y, :]
        if np.any(slc) and not np.all(slc):
            filled = ndimage.binary_fill_holes(slc > 0)
            mask_vol[:, y, :] = np.maximum(slc, (filled * 255).astype(np.uint8))

    print("      Pass 3/3: Coronal (Yt)...")
    for x in tqdm(range(W), desc="        X-axis", leave=False):
        slc = mask_vol[:, :, x]
        if np.any(slc) and not np.all(slc):
            filled = ndimage.binary_fill_holes(slc > 0)
            mask_vol[:, :, x] = np.maximum(slc, (filled * 255).astype(np.uint8))

    if hasattr(mask_vol, 'flush'):
        mask_vol.flush()

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    labels_dir = args.output / "labels"
    binary_dir = args.output / "binary_masks"
    temp_dir = args.output / "temp"

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    if args.save_labels: labels_dir.mkdir()
    if args.save_binary: binary_dir.mkdir()
    temp_dir.mkdir()

    # ---------------------------------------------------------
    # Step 1: Load Volume (Grayscale)
    # ---------------------------------------------------------
    print(f"[1/8] Loading Input Video into RAM (Grayscale)...")
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    vol_orig = np.stack(frames)
    T, H, W = vol_orig.shape
    print(f"      Volume Loaded: {vol_orig.shape} (T, H, W)")

    # ---------------------------------------------------------
    # Initialization (MEMMAP Architecture)
    # ---------------------------------------------------------
    mask_path = temp_dir / "final_mask.dat"
    conf_path = temp_dir / "final_conf.dat"
    final_mask_vol = np.memmap(mask_path, dtype=np.uint8, mode='w+', shape=(T, H, W))
    final_conf_vol = np.memmap(conf_path, dtype=np.float16, mode='w+', shape=(T, H, W))
    final_mask_vol[:] = 0
    final_conf_vol[:] = 0

    # ---------------------------------------------------------
    # Step 2 & 3: Augment, Infer, & Accumulate
    # ---------------------------------------------------------
    print(f"[2/8 & 3/8] Generating Augmentations & Running Multiplanar Inference...")

    views = ['Transverse']
    if not args.disable_multiplanar:
        views.extend(['Sagittal', 'Coronal'])

    shift_indices = [0]
    if args.shift > 0:
        shift_indices.extend(range(1, 9))

    for model_path in args.model_list:
        model = YOLO(model_path)

        for view in views:
            if view == 'Transverse':
                source, src_h, src_w = vol_orig, H, W
            elif view == 'Sagittal':
                source, src_h, src_w = vol_orig.transpose(1, 0, 2), T, W
            elif view == 'Coronal':
                source, src_h, src_w = vol_orig.transpose(2, 0, 1), T, H

            for angle in args.angle_list:
                for s_idx in shift_indices:
                    M_fwd, M_inv = get_affine_matrix((src_h, src_w), args.imgsz, angle, args.shift, s_idx)

                    vid_path = temp_dir / f"{view}_a{angle}_s{s_idx}.mkv"

                    # Write BGR frames for Ultralytics cv2.VideoCapture compatibility
                    writer = cv2.VideoWriter(
                        str(vid_path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (args.imgsz, args.imgsz), isColor=True
                    )
                    for i in range(source.shape[0]):
                        warped = cv2.warpAffine(
                            source[i], M_fwd, (args.imgsz, args.imgsz),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                        )
                        writer.write(cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR))
                    writer.release()

                    # Inference
                    results = model.predict(
                        source=str(vid_path), stream=True, save=False, task='segment',
                        iou=1.0, retina_masks=True, batch=1,
                        conf=args.conf, imgsz=args.imgsz,
                        device=args.device, int8=args.int8, half=args.half, verbose=False
                    )

                    for i, r in enumerate(results):
                        if r.masks is None or len(r.masks.data) == 0: continue

                        union_mask = np.any(r.masks.data.cpu().numpy(), axis=0).astype(np.uint8) * 255
                        max_conf = float(r.boxes.conf.max().cpu().numpy())

                        union_mask = fill_2d_holes(union_mask)
                        union_mask = filter_straight_edges(union_mask, threshold=30)
                        if not np.any(union_mask): continue

                        restored = cv2.warpAffine(
                            union_mask, M_inv, (src_w, src_h),
                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                        )

                        # Inject using np.maximum to avoid cv2 slice striding issues
                        if view == 'Transverse':
                            final_mask_vol[i] = np.maximum(final_mask_vol[i], restored)
                            mask_bool = restored > 0
                            final_conf_vol[i][mask_bool] = np.maximum(final_conf_vol[i][mask_bool], max_conf)
                        elif view == 'Sagittal':
                            final_mask_vol[:, i, :] = np.maximum(final_mask_vol[:, i, :], restored)
                            mask_bool = restored > 0
                            final_conf_vol[:, i, :][mask_bool] = np.maximum(final_conf_vol[:, i, :][mask_bool], max_conf)
                        elif view == 'Coronal':
                            final_mask_vol[:, :, i] = np.maximum(final_mask_vol[:, :, i], restored)
                            mask_bool = restored > 0
                            final_conf_vol[:, :, i][mask_bool] = np.maximum(final_conf_vol[:, :, i][mask_bool], max_conf)

                    # Clean up temp video immediately to prevent 400GB+ bloat
                    if not args.keep_temp:
                        vid_path.unlink(missing_ok=True)

            del source
            gc.collect()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if hasattr(final_mask_vol, 'flush'):
        final_mask_vol.flush()
        final_conf_vol.flush()

    # ---------------------------------------------------------
    # Step 4: 3D Processing
    # ---------------------------------------------------------
    print(f"[4/8] 3D Hole Filling (Multiplanar 2D Strategy)...")
    fill_3d_holes_multiplanar(final_mask_vol)

    # ---------------------------------------------------------
    # Step 5: Temporal Processing
    # ---------------------------------------------------------
    print(f"[5/8] Temporal Processing (N={args.temporal})...")

    high_conf_thresh = args.conf * 2.0
    N = args.temporal
    flow_size = 512

    if N > 0:
        # SNAPSHOT ORIGINAL STATE
        orig_has_obj = np.array([np.max(final_mask_vol[t]) > 0 for t in range(T)])
        orig_high_conf = np.array([np.max(final_conf_vol[t]) >= high_conf_thresh for t in range(T)])

        def get_pull_flow(t_dst, t_src):
            """
            Calculates backward flow from t_dst -> t_src.
            For every pixel in t_dst, gives the displacement to sample from t_src.
            """
            f_dst = cv2.resize(vol_orig[t_dst], (flow_size, flow_size))
            f_src = cv2.resize(vol_orig[t_src], (flow_size, flow_size))
            flow = cv2.calcOpticalFlowFarneback(f_dst, f_src, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            full_flow = cv2.resize(flow, (W, H))
            full_flow[..., 0] *= (W / flow_size)
            full_flow[..., 1] *= (H / flow_size)
            return full_flow

        def warp_mask_with_pull(mask_src, flow_pull):
            h, w = mask_src.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow_pull[..., 0]).astype(np.float32)
            map_y = (grid_y + flow_pull[..., 1]).astype(np.float32)
            return cv2.remap(mask_src, map_x, map_y, cv2.INTER_NEAREST)

        # 1. Interpolation
        for t in tqdm(range(T), desc="  Interpolation"):
            if not orig_has_obj[t]:
                neighbors_ok = True
                for dt in range(-N, N+1):
                    if dt == 0: continue
                    idx = t + dt
                    if 0 <= idx < T:
                        if not orig_high_conf[idx]:
                            neighbors_ok = False; break
                    else: neighbors_ok = False; break

                if neighbors_ok:
                    flow = get_pull_flow(t, t-1)
                    final_mask_vol[t] = warp_mask_with_pull(final_mask_vol[t-1], flow)
                    final_conf_vol[t][final_mask_vol[t] > 0] = args.conf

        # 2. Extrapolation Fwd
        for t in tqdm(range(T), desc="  Extrapolation Fwd"):
            if not np.any(final_mask_vol[t]):
                series_ok = True
                for dt in range(1, N+1):
                    idx = t - dt
                    if idx < 0 or not orig_high_conf[idx]:
                        series_ok = False; break
                if series_ok:
                    flow = get_pull_flow(t, t-1)
                    final_mask_vol[t] = warp_mask_with_pull(final_mask_vol[t-1], flow)
                    final_conf_vol[t][final_mask_vol[t] > 0] = args.conf

        # 3. Extrapolation Bwd
        for t in tqdm(range(T-1, -1, -1), desc="  Extrapolation Bwd"):
            if not np.any(final_mask_vol[t]):
                series_ok = True
                for dt in range(1, N+1):
                    idx = t + dt
                    if idx >= T or not orig_high_conf[idx]:
                        series_ok = False; break
                if series_ok:
                    flow = get_pull_flow(t, t+1)
                    final_mask_vol[t] = warp_mask_with_pull(final_mask_vol[t+1], flow)
                    final_conf_vol[t][final_mask_vol[t] > 0] = args.conf

        # 4. Drop Low Conf (Checking Current State)
        for t in tqdm(range(T), desc="  Drop Logic"):
            if np.max(final_conf_vol[t]) > 0 and np.max(final_conf_vol[t]) < high_conf_thresh:
                has_obj_neighbor = False
                for dt in range(-N, N+1):
                    if dt == 0: continue
                    idx = t + dt
                    if 0 <= idx < T and np.max(final_mask_vol[idx]) > 0:
                        has_obj_neighbor = True; break

                if not has_obj_neighbor:
                    final_mask_vol[t] = 0
                    final_conf_vol[t] = 0

        if hasattr(final_mask_vol, 'flush'):
            final_mask_vol.flush()
    else:
        print("      Temporal processing disabled (N=0).")

    # ---------------------------------------------------------
    # Step 6: Output
    # ---------------------------------------------------------
    print(f"[6/8] Saving Outputs...")

    fname = args.input_stem
    vid_ov_path = args.output / f"{fname}_Overlay.mkv"
    writer = cv2.VideoWriter(str(vid_ov_path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (W, H), isColor=True)

    bin_writer = None
    if args.save_binary:
        vid_bin_path = args.output / f"{fname}_Binary.mkv"
        bin_writer = cv2.VideoWriter(str(vid_bin_path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (W, H), isColor=False)

    for t in tqdm(range(T), desc="Writing Frames"):
        mask = final_mask_vol[t]

        # Labels (Polygon)
        if args.save_labels:
            txt_path = labels_dir / f"{fname}_{t:04d}.txt"
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                with open(txt_path, 'w') as f:
                    for cnt in contours:
                        pts = cnt.reshape(-1, 2).astype(np.float32)
                        pts[:, 0] /= W
                        pts[:, 1] /= H
                        coord_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts])
                        f.write(f"0 {coord_str}\n")
            else:
                open(txt_path, 'w').close()

        # Binary
        if args.save_binary:
            tifffile.imwrite(binary_dir / f"{fname}_Binary_{t:04d}.tiff", mask)
            bin_writer.write(mask)

        # Overlay
        frame_color = cv2.cvtColor(vol_orig[t], cv2.COLOR_GRAY2BGR)
        if np.any(mask):
            overlay = np.zeros_like(frame_color)
            overlay[mask > 0] = [255, 0, 0]
            mask_bool = mask > 0

            roi_src = frame_color[mask_bool]
            roi_over = overlay[mask_bool]
            weighted = cv2.addWeighted(roi_src, 0.5, roi_over, 0.5, 0)
            frame_color[mask_bool] = weighted

        writer.write(frame_color)

    writer.release()
    if bin_writer: bin_writer.release()

    # Final Cleanup
    if not args.keep_temp:
        del final_mask_vol
        del final_conf_vol
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Pipeline Complete.")

if __name__ == "__main__":
    main()
