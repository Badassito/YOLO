#!/usr/bin/env python3
"""
YOLO segmentation test-time augmentation (TTA) for large transverse video volumes.

This v5.0-oriented script:
  - builds transverse, sagittal and coronal views (unless --disable_multiplanar)
  - generates rotated / scaled / shifted FFV1 MKV augmentations via a single affine transform per variant
  - runs Ultralytics YOLO segmentation sequentially on the pre-generated augmentation videos
  - stores per-augmentation results to disk, undoes the affine transforms and unions masks per slice
  - applies --min_conf directly to per-slice unions, fills 2D holes, and optionally interpolates each view volume independently
  - unions the final per-view / per-model volumes, fills enclosed 3D voids, optionally filters small transverse-plane components via --min_radius
  - writes the final transverse overlay video plus optional YOLO labels, binary TIFF/MKV outputs, NRRD, and a summary text file

Dependencies (Python):
  pip install opencv-python numpy scipy scikit-image tifffile tqdm ultralytics
  pip install pynrrd   # only needed for --save_nrrd

System:
  ffmpeg + ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# --- third-party ---
try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("OpenCV (cv2) is required: pip install opencv-python") from e

from scipy import ndimage as ndi  # type: ignore

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize  # type: ignore
    try:
        from skimage.morphology import skeletonize_3d as _skimage_skeletonize_3d  # type: ignore
    except Exception:
        _skimage_skeletonize_3d = None
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-image is required: pip install scikit-image") from e

try:
    import tifffile  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("tifffile is required: pip install tifffile") from e

try:
    from tqdm import tqdm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("tqdm is required: pip install tqdm") from e


# --------------------------
# CLI / args
# --------------------------

def _parse_angles(s: str) -> List[float]:
    """Accepts comma or whitespace separated angles."""
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    return [float(p) for p in parts if p != ""]


def _parse_models(values: Sequence[str]) -> List[str]:
    """
    Accepts:
      --model m1.pt m2.pt
      --model "m1.pt,m2.pt"
    """
    out: List[str] = []
    for v in values:
        if "," in v:
            out.extend([x.strip() for x in v.split(",") if x.strip()])
        else:
            out.append(v.strip())
    return [x for x in out if x]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multiplanar YOLO-segmentation TTA (rotation/shift) for large square videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", required=True, type=str, help="Input video path")
    p.add_argument("--output", default=None, type=str, help="Output directory (default ./{Filename}/)")
    p.add_argument("--device", default="0", type=str, help="Device for YOLO predict")
    p.add_argument("--model", required=True, nargs="+", type=str, help="One or more YOLO segmentation model paths")

    p.add_argument("--disable_multiplanar", action="store_true",
                   help="Only use Transverse view (skip Sagittal/Coronal)")
    p.add_argument("--angle", default="0,120,240", type=str,
                   help="Rotation angles in degrees for augmentation (comma/space separated)")
    p.add_argument("--imgsz", default=1536, type=int, help="Square input size for YOLO predict")
    p.add_argument("--shift", default=0, type=int,
                   help="If nonzero, create 4 shifted variants (U/D/L/R) per rotation per view, plus unshifted")

    p.add_argument("--conf", default=0.15, type=float, help="Passed to YOLO predict")
    p.add_argument("--min_conf", default=0.30, type=float,
                   help="Remove per-slice unions whose confidence is below this threshold (must be >= --conf)")
    p.add_argument("--min_radius", default=0.0, type=float,
                   help="Remove final transverse-plane connected components whose radius is smaller than this value")
    p.add_argument("--half", action="store_true", help="Enable FP16 inference (Ultralytics half=True)")
    p.add_argument("--int8", action="store_true", help="Enable INT8 inference if supported (Ultralytics int8=True)")

    p.add_argument("--save_labels", nargs="?", const="__DEFAULT__", default=None, type=str,
                   help="Save final YOLO segmentation labels per frame. Optional custom pattern, e.g. labels/{Filename}_%%04d.txt")
    p.add_argument("--save_binary", nargs="?", const="__DEFAULT__", default=None, type=str,
                   help="Save final binary masks as TIFF sequence + FFV1 MKV. Optional custom TIFF pattern, e.g. binary_masks/{Filename}_Binary_%%04d.tiff")
    p.add_argument("--save_nrrd", action="store_true", help="Save the final binary mask volume as an NRRD file")
    p.add_argument("--voxel_volume", action="store_true", help="Count white voxels in the final binary output and save to the summary text file")
    p.add_argument("--binary", action="store_true", dest="voxel_volume", help=argparse.SUPPRESS)
    p.add_argument("--troubleshooting", action="store_true",
                   help="Keep temp files and save pass snapshots before each interpolation pass")

    p.add_argument("--interpolate", default=15, type=int,
                   help="Maximum slice-distance used to search for interpolation candidates. 0 disables interpolation")
    p.add_argument("--interpolation_walk_back", default=3, type=int,
                   help="Additional source slices to bridge before the endpoint slice. 0 disables walk-back bridges")
    p.add_argument("--interpolate_passes", default=1, type=int,
                   help="Run the interpolation process this many passes, treating the previous pass as real")
    p.add_argument("--interpolate_min_radius", default=3, type=float,
                   help="Skip interpolation bridges whose effective radius is <= this value")
    p.add_argument("--interpolation_search_angle", default=15.0, type=float,
                   help="Projection growth angle in degrees. Must be greater than -90 and less than 90")
    p.add_argument("--cone_half_angle", dest="interpolation_search_angle", type=float, help=argparse.SUPPRESS)
    return p


# --------------------------
# ffmpeg helpers
# --------------------------

def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")


def ffprobe_info(video_path: Path) -> Dict[str, object]:
    """Return dict with width, height, fps, num_frames."""
    _require_bin("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames",
        "-of", "json",
        str(video_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(p.stdout)
    if "streams" not in info or not info["streams"]:
        raise RuntimeError(f"ffprobe: no video stream found in {video_path}")
    st = info["streams"][0]

    width = int(st.get("width"))
    height = int(st.get("height"))

    def _parse_ratio(r: str) -> float:
        if not r or r == "0/0":
            return 0.0
        num, den = r.split("/")
        den_i = int(den)
        return float(num) / float(den_i) if den_i != 0 else 0.0

    fps = _parse_ratio(str(st.get("avg_frame_rate", "0/0")))
    if fps <= 0:
        fps = _parse_ratio(str(st.get("r_frame_rate", "0/0")))
    if fps <= 0:
        fps = 30.0

    nf = st.get("nb_read_frames", None)
    if nf is None or str(nf).strip() == "" or str(nf) == "N/A":
        nf = st.get("nb_frames", None)
    if nf is None or str(nf).strip() == "" or str(nf) == "N/A":
        raise RuntimeError(
            "ffprobe could not determine frame count (nb_read_frames/nb_frames missing). "
            "Please provide an input with a known frame count."
        )
    num_frames = int(nf)
    return {"width": width, "height": height, "fps": fps, "num_frames": num_frames}


def decode_video_to_memmap_rgb24(
    input_video: Path,
    out_dat: Path,
    num_frames: int,
    width: int,
    height: int,
    overwrite: bool = False,
) -> np.memmap:
    """Decode input video to a (T,H,W,3) uint8 memmap in RGB24."""
    _require_bin("ffmpeg")
    if out_dat.exists() and not overwrite:
        return np.memmap(out_dat, dtype=np.uint8, mode="r+", shape=(num_frames, height, width, 3))

    if out_dat.exists():
        out_dat.unlink()

    out_dat.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_dat, dtype=np.uint8, mode="w+", shape=(num_frames, height, width, 3))
    frame_bytes = width * height * 3

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(input_video),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-vsync", "0",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    try:
        for i in tqdm(range(num_frames), desc="Decoding input -> memmap (rgb24)"):
            buf = proc.stdout.read(frame_bytes)
            if buf is None or len(buf) < frame_bytes:
                raise RuntimeError(f"Unexpected EOF while decoding frame {i}/{num_frames}")
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
            mm[i, :, :, :] = frame
        mm.flush()
    finally:
        if proc.stdout:
            proc.stdout.close()
        _, err = proc.communicate()
        if proc.returncode not in (0, None):
            msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
            raise RuntimeError(f"ffmpeg decode failed: {msg}")
    return mm


def ffmpeg_rawvideo_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    pix_fmt_in: str = "rgb24",
    codec: str = "ffv1",
    pix_fmt_out: str = "yuv444p",
) -> subprocess.Popen:
    """Return a Popen with stdin open for writing raw frames."""
    _require_bin("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt_in,
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-an",
        "-c:v", codec,
        "-level", "3",
        "-pix_fmt", pix_fmt_out,
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    return proc


def close_ffmpeg_writer(proc: subprocess.Popen) -> None:
    """Close an ffmpeg writer Popen safely (Python 3.12+ compatible).

    IMPORTANT:
      Calling proc.stdin.close() and then proc.communicate() triggers
      'ValueError: flush of closed file' on Python 3.12, because communicate()
      tries to flush stdin even if it is already closed. We therefore:
        1) close stdin (if open)
        2) set proc.stdin = None
        3) call communicate() to drain stdout/stderr
    """
    if proc.stdin is not None and not proc.stdin.closed:
        try:
            proc.stdin.close()
        except Exception:
            pass

    # Prevent subprocess.communicate() from flushing a closed stdin (Py3.12 behavior)
    proc.stdin = None  # type: ignore[attr-defined]

    _, err = proc.communicate()
    if proc.returncode not in (0, None):
        msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
        raise RuntimeError(f"ffmpeg write failed: {msg}")



# --------------------------
# Affine transforms
# --------------------------

@dataclass(frozen=True)
class AffineSpec:
    view: str
    angle_deg: float
    shift_dx: int
    shift_dy: int
    src_w: int
    src_h: int
    out_size: int
    canvas_w: int
    canvas_h: int
    pad_size: int
    pad_off_x: float
    pad_off_y: float
    M_out_to_src: np.ndarray  # 2x3 float32
    M_src_to_out: np.ndarray  # 2x3 float32


def _expanded_pad_size(w: int, h: int, angle_deg: float) -> int:
    """Square canvas size P to fit a WxH rectangle rotated by angle_deg."""
    theta = math.radians(angle_deg % 360.0)
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    w_rot = w * c + h * s
    h_rot = w * s + h * c
    P = int(math.ceil(max(w_rot, h_rot)))
    return max(P, max(w, h))


def _affine2x3_to_3x3(M: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, :3] = np.asarray(M, dtype=np.float64)
    return out


def build_affine(
    view: str,
    src_w: int,
    src_h: int,
    out_size: int,
    angle_deg: float,
    shift_dx: int,
    shift_dy: int,
    pad_mode: str,
) -> AffineSpec:
    """
    Build a single affine transform that performs, in one pass:

      source(native view) -> optional black-padding canvas -> rotation around canvas center
      -> scale to out_size x out_size -> optional shift in output space

    Transverse uses pad_mode='clamp', which rotates directly on the source-sized canvas so
    non-90° content that leaves the source frame is discarded. Sagittal/Coronal use pad_mode='pad'
    with a square black canvas large enough to preserve the full rotation before scaling.
    """
    if pad_mode not in ("clamp", "pad"):
        raise ValueError("pad_mode must be 'clamp' or 'pad'")

    if pad_mode == "pad":
        canvas_w = canvas_h = _expanded_pad_size(src_w, src_h, angle_deg)
    else:
        canvas_w = src_w
        canvas_h = src_h

    off_x = (canvas_w - src_w) / 2.0
    off_y = (canvas_h - src_h) / 2.0

    cx_canvas = (canvas_w - 1) / 2.0
    cy_canvas = (canvas_h - 1) / 2.0
    cx_out = (out_size - 1) / 2.0
    cy_out = (out_size - 1) / 2.0

    M_pad = np.array(
        [
            [1.0, 0.0, off_x],
            [0.0, 1.0, off_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    M_rot = _affine2x3_to_3x3(cv2.getRotationMatrix2D((cx_canvas, cy_canvas), angle_deg, 1.0))

    sx = out_size / float(canvas_w)
    sy = out_size / float(canvas_h)
    M_scale_shift = np.array(
        [
            [sx, 0.0, cx_out + float(shift_dx) - sx * cx_canvas],
            [0.0, sy, cy_out + float(shift_dy) - sy * cy_canvas],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    M_src_to_out3 = M_scale_shift @ M_rot @ M_pad
    M_out_to_src3 = np.linalg.inv(M_src_to_out3)

    return AffineSpec(
        view=view,
        angle_deg=float(angle_deg),
        shift_dx=int(shift_dx),
        shift_dy=int(shift_dy),
        src_w=src_w,
        src_h=src_h,
        out_size=out_size,
        canvas_w=int(canvas_w),
        canvas_h=int(canvas_h),
        pad_size=int(max(canvas_w, canvas_h)),
        pad_off_x=float(off_x),
        pad_off_y=float(off_y),
        M_out_to_src=M_out_to_src3[:2, :3].astype(np.float32),
        M_src_to_out=M_src_to_out3[:2, :3].astype(np.float32),
    )


# --------------------------
# View slicing
# --------------------------

@dataclass(frozen=True)
class ViewInfo:
    name: str
    num_slices: int
    src_h: int
    src_w: int
    pad_mode: str  # 'clamp' or 'pad'


def get_view_infos(T: int, H: int, W: int, disable_multiplanar: bool) -> List[ViewInfo]:
    views = [
        ViewInfo(name="transverse", num_slices=T, src_h=H, src_w=W, pad_mode="clamp"),
    ]
    if not disable_multiplanar:
        views.append(ViewInfo(name="sagittal", num_slices=H, src_h=T, src_w=W, pad_mode="pad"))
        views.append(ViewInfo(name="coronal", num_slices=W, src_h=T, src_w=H, pad_mode="pad"))
    return views


def iter_view_frames(volume_rgb: np.memmap, view: ViewInfo) -> Iterator[np.ndarray]:
    """Yield frames for a view, in slice order (0..num_slices-1)."""
    T, H, W, C = volume_rgb.shape
    assert C == 3

    if view.name == "transverse":
        for t in range(T):
            yield np.asarray(volume_rgb[t])  # (H,W,3)
    elif view.name == "sagittal":
        for y in range(H):
            yield np.ascontiguousarray(volume_rgb[:, y, :, :])  # (T,W,3)
    elif view.name == "coronal":
        for x in range(W):
            yield np.ascontiguousarray(volume_rgb[:, :, x, :])  # (T,H,3)
    else:
        raise ValueError(f"Unknown view: {view.name}")


# --------------------------
# Packed mask IO
# --------------------------

def bytes_for_packbits(h: int, w: int) -> int:
    return (h * w + 7) // 8


def pack_mask(mask01: np.ndarray) -> np.ndarray:
    """Pack a 2D/1D binary mask (bool or 0/1 uint8) into np.packbits uint8."""
    flat = np.asarray(mask01).reshape(-1)
    # np.packbits supports bool directly; for numeric inputs we treat nonzero as 1.
    if flat.dtype == np.bool_:
        return np.packbits(flat)
    else:
        return np.packbits((flat != 0).astype(np.uint8, copy=False))


def unpack_mask(packed: np.ndarray, h: int, w: int) -> np.ndarray:
    """Unpack a packed-bit mask into a 2D uint8 array of shape (h,w) with values {0,1}."""
    flat = np.unpackbits(packed)[: h * w]
    return flat.reshape((h, w)).astype(np.uint8, copy=False)


def any_mask(packed: np.ndarray) -> bool:
    return bool(np.any(packed))


def overlap_any(packed_a: np.ndarray, packed_b: np.ndarray) -> bool:
    return bool(np.any(np.bitwise_and(packed_a, packed_b)))


# --------------------------
# YOLO inference
# --------------------------

def load_ultralytics_model(path: str):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is required. Install with: pip install ultralytics\n"
            f"Import error: {e}"
        ) from e
    return YOLO(path)


@dataclass
class PredictConfig:
    imgsz: int
    conf: float
    device: str
    half: bool
    int8: bool


def predict_video_and_accumulate(
    model,
    video_path: Path,
    num_frames: int,
    out_size: int,
    pred_out_prefix: Path,
    cfg: PredictConfig,
    # accumulation into per-view union stack (native resolution, packed bits)
    view_union_mm: np.memmap,          # uint8 packbits, shape (num_slices, bytes_native)
    view_conf_mm: np.memmap,           # float16, shape (num_slices,)
    M_out_to_native: np.ndarray,       # 2x3, maps augmented(out)->native for cv2.warpAffine (src->dst)
    native_h: int,
    native_w: int,
) -> None:
    """
    Runs YOLO predict(stream=True) on a pre-generated augmented video,
    stores per-frame union mask + max confidence to disk, and accumulates
    the inverse-transformed mask into the per-view union stack.
    """
    pred_out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_bytes = bytes_for_packbits(out_size, out_size)

    pred_mask_mm = np.memmap(
        pred_out_prefix.with_suffix(".mask.packbits.dat"),
        dtype=np.uint8,
        mode="w+",
        shape=(num_frames, out_bytes),
    )
    pred_conf_mm = np.memmap(
        pred_out_prefix.with_suffix(".conf.f16.dat"),
        dtype=np.float16,
        mode="w+",
        shape=(num_frames,),
    )
    pred_mask_mm[:] = 0
    pred_conf_mm[:] = 0

    results = model.predict(
        source=str(video_path),
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=1.0,
        save=False,
        stream=True,
        task="segment",
        retina_masks=True,
        batch=1,
        device=cfg.device,
        half=cfg.half,
        int8=cfg.int8,
        verbose=False,
    )

    for idx, r in enumerate(results):
        if idx >= num_frames:
            break

        # Union all instance masks in this frame -> single binary mask
        if getattr(r, "masks", None) is None or r.masks is None or r.masks.data is None:
            union = np.zeros((out_size, out_size), dtype=np.uint8)
            conf_val = 0.0
        else:
            m = r.masks.data  # (n,h,w)
            try:
                import torch  # type: ignore
                union_t = (m.sum(dim=0) > 0).to(torch.uint8)
                union = union_t.cpu().numpy()
            except Exception:
                union = (np.sum(np.asarray(m), axis=0) > 0).astype(np.uint8)

            conf_val = 0.0
            if getattr(r, "boxes", None) is not None and r.boxes is not None and getattr(r.boxes, "conf", None) is not None:
                try:
                    conf_val = float(r.boxes.conf.max().cpu().item())
                except Exception:
                    conf_val = float(np.max(np.asarray(r.boxes.conf)))

        # Ensure union mask matches expected size (Ultralytics should return masks at image size, but be defensive)
        if union.shape[0] != out_size or union.shape[1] != out_size:
            union = cv2.resize(union.astype(np.uint8), (out_size, out_size), interpolation=cv2.INTER_NEAREST)

        # Store per-augmentation result to disk (packed)
        pred_mask_mm[idx, :] = pack_mask(union)
        pred_conf_mm[idx] = np.float16(conf_val)

    pred_mask_mm.flush()
    pred_conf_mm.flush()

    # Read the stored per-augmentation results back from disk, undo the augmentation,
    # and accumulate into the native-orientation union stack.
    for idx in range(num_frames):
        packed_pred = np.asarray(pred_mask_mm[idx])
        if not any_mask(packed_pred):
            continue

        union = unpack_mask(packed_pred, out_size, out_size)
        conf_val = float(pred_conf_mm[idx])

        mask_back = cv2.warpAffine(
            union,
            M_out_to_native,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        packed_back = pack_mask((mask_back > 0).astype(np.uint8))

        view_union_mm[idx, :] |= packed_back
        if conf_val > float(view_conf_mm[idx]):
            view_conf_mm[idx] = np.float16(conf_val)

    meta = {
        "video": str(video_path),
        "num_frames": int(num_frames),
        "out_size": int(out_size),
        "mask_packbits_bytes": int(out_bytes),
        "cfg": {
            "imgsz": int(cfg.imgsz),
            "conf": float(cfg.conf),
            "device": str(cfg.device),
            "half": bool(cfg.half),
            "int8": bool(cfg.int8),
        },
    }
    pred_out_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


# --------------------------
# Per-view postprocessing
# --------------------------

def apply_min_conf_filter_inplace(
    union_mm: np.memmap,
    conf_mm: np.memmap,
    min_conf: float,
) -> None:
    """Remove any per-slice union whose stored confidence is below ``min_conf``."""
    n = union_mm.shape[0]
    for i in tqdm(range(n), desc="min_conf thresholding"):
        if float(conf_mm[i]) >= float(min_conf):
            continue
        union_mm[i, :] = 0
        conf_mm[i] = np.float16(0.0)


def fill_2d_holes_inplace(union_mm: np.memmap, h: int, w: int) -> None:
    """Fill all 2D holes per slice (donut-hole fill).

    NOTE:
      We intentionally avoid scipy.ndimage.binary_fill_holes() here. On very large
      slices (e.g., 3072x3072) and long runs, SciPy's implementation can create
      large transient allocations and may destabilize the process depending on
      the system / BLAS / OpenMP configuration.

      This implementation uses an OpenCV flood-fill on the inverted mask with a
      1-pixel constant-black border, which is robust and memory-stable:

        holes = background components NOT connected to the padded image boundary
        filled = mask OR holes
    """
    n = union_mm.shape[0]

    # Preallocate scratch buffers (reduces allocator churn / fragmentation).
    pad = np.zeros((h + 2, w + 2), dtype=np.uint8)          # 0/1 with a black border
    inv = np.empty_like(pad)                                # 0/1 inverse of pad
    flood = np.empty_like(pad)                              # working image for floodFill
    ffmask = np.zeros((h + 4, w + 4), dtype=np.uint8)        # floodFill mask: (H+2, W+2)
    holes = np.empty(pad.shape, dtype=np.bool_)              # boolean scratch

    for i in tqdm(range(n), desc="2D hole fill"):
        packed = np.asarray(union_mm[i])
        if not any_mask(packed):
            continue

        m = unpack_mask(packed, h, w)  # uint8 {0,1}

        # Build padded mask
        pad.fill(0)
        pad[1:-1, 1:-1] = m

        # inv = 1 - pad (still {0,1})
        inv[:] = 1
        inv -= pad

        # Flood-fill the outside background (connected to the padded boundary)
        flood[:] = inv
        ffmask.fill(0)
        # New value=2 so we can distinguish "reached outside background" from "unreached holes" (=1)
        cv2.floodFill(flood, ffmask, seedPoint=(0, 0), newVal=2)

        # Holes are inverse-background pixels still == 1 after flood fill
        np.equal(flood, 1, out=holes)
        if holes.any():
            pad[holes] = 1

        filled = pad[1:-1, 1:-1]
        union_mm[i, :] = pack_mask(filled)

# --------------------------
# 3D assembly + postprocessing
# --------------------------

def assemble_model_volume_into_ensemble(
    ensemble_mm: np.memmap,  # uint8 (0/1) shape (T,H,W)
    view_union_mms: Dict[str, np.memmap],
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
) -> None:
    """
    OR the per-view union stacks for ONE model into the 3D ensemble volume.

    Views:
      - transverse: slices along t, each slice is HxW
      - sagittal: slices along y, each slice is T x W, mapped into ensemble[:, y, :]
      - coronal: slices along x, each slice is T x H, mapped into ensemble[:, :, x]
    """
    transverse = view_union_mms["transverse"]
    bytes_xy = bytes_for_packbits(H, W)
    assert transverse.shape == (T, bytes_xy)

    for t in tqdm(range(T), desc="Assembling volume from transverse"):
        m = unpack_mask(np.asarray(transverse[t]), H, W)
        ensemble_mm[t, :, :] |= m

    if disable_multiplanar:
        return

    sagittal = view_union_mms["sagittal"]
    bytes_tx = bytes_for_packbits(T, W)
    assert sagittal.shape == (H, bytes_tx)

    for y in tqdm(range(H), desc="Assembling volume from sagittal"):
        m = unpack_mask(np.asarray(sagittal[y]), T, W)
        ensemble_mm[:, y, :] |= m

    coronal = view_union_mms["coronal"]
    bytes_ty = bytes_for_packbits(T, H)
    assert coronal.shape == (W, bytes_ty)

    for x in tqdm(range(W), desc="Assembling volume from coronal"):
        m = unpack_mask(np.asarray(coronal[x]), T, H)  # (T,H) cols are y
        ensemble_mm[:, :, x] |= m


def fill_3d_voids(mask_u8: np.ndarray) -> np.ndarray:
    """Fill enclosed 3D voids by labeling background CCs and keeping only those connected to a boundary."""
    mask_bool = np.asarray(mask_u8, dtype=bool)
    if mask_bool.size == 0:
        return mask_bool.astype(np.uint8)

    bg = np.logical_not(mask_bool)
    labels_bg, num_bg = ndi.label(bg, structure=np.ones((3, 3, 3), dtype=bool))
    if num_bg == 0:
        return mask_bool.astype(np.uint8)

    boundary_labels: List[np.ndarray] = []
    for slab in (
        labels_bg[0, :, :],
        labels_bg[-1, :, :],
        labels_bg[:, 0, :],
        labels_bg[:, -1, :],
        labels_bg[:, :, 0],
        labels_bg[:, :, -1],
    ):
        vals = slab[slab > 0]
        if vals.size:
            boundary_labels.append(np.unique(vals))

    if boundary_labels:
        touching = np.unique(np.concatenate(boundary_labels))
        touches_boundary = np.zeros(num_bg + 1, dtype=bool)
        touches_boundary[touching] = True
        enclosed = (labels_bg > 0) & (~touches_boundary[labels_bg])
    else:
        enclosed = labels_bg > 0

    out = mask_bool.copy()
    out[enclosed] = True
    return out.astype(np.uint8)


def unpack_view_union_to_volume(
    union_mm: np.memmap,
    num_slices: int,
    h: int,
    w: int,
    out_path: Path,
    desc: str,
) -> np.memmap:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol_mm = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(num_slices, h, w))
    for i in tqdm(range(num_slices), desc=desc):
        packed = np.asarray(union_mm[i])
        if any_mask(packed):
            vol_mm[i, :, :] = unpack_mask(packed, h, w)
        else:
            vol_mm[i, :, :] = 0
    vol_mm.flush()
    return vol_mm


def apply_transverse_min_radius_filter(mask_u8: np.ndarray, min_radius: float) -> np.ndarray:
    """Remove 2D connected components whose transverse-plane radius is smaller than ``min_radius``."""
    if float(min_radius) <= 0:
        return np.asarray(mask_u8, dtype=np.uint8)

    out = np.asarray(mask_u8, dtype=np.uint8).copy()
    struct2 = np.ones((3, 3), dtype=bool)

    for t in tqdm(range(out.shape[0]), desc="Transverse min-radius filter"):
        sl = out[t] > 0
        if not sl.any():
            continue

        labels2d, num = ndi.label(sl, structure=struct2)
        if num <= 0:
            out[t, :, :] = 0
            continue

        keep = np.zeros(sl.shape, dtype=bool)
        for lbl in range(1, int(num) + 1):
            comp = labels2d == lbl
            if not comp.any():
                continue
            radius = float(np.max(ndi.distance_transform_edt(comp)))
            if radius >= float(min_radius):
                keep |= comp

        out[t, :, :] = keep.astype(np.uint8)

    return out.astype(np.uint8)


def assemble_model_volume_from_view_volumes(
    ensemble_mm: np.memmap,
    view_volume_mms: Dict[str, np.memmap],
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
) -> None:
    transverse = np.asarray(view_volume_mms["transverse"])
    assert transverse.shape == (T, H, W)
    ensemble_mm[:, :, :] |= transverse

    if disable_multiplanar:
        return

    sagittal = np.asarray(view_volume_mms["sagittal"])
    assert sagittal.shape == (H, T, W)
    for y in tqdm(range(H), desc="Assembling volume from sagittal view volume"):
        ensemble_mm[:, y, :] |= sagittal[y, :, :]

    coronal = np.asarray(view_volume_mms["coronal"])
    assert coronal.shape == (W, T, H)
    for x in tqdm(range(W), desc="Assembling volume from coronal view volume"):
        ensemble_mm[:, :, x] |= coronal[x, :, :]


def assemble_current_ensemble_volume(
    view_volumes_by_model: Dict[str, Dict[str, np.memmap]],
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
    out_path: Path,
) -> np.memmap:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_mm = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(T, H, W))
    ensemble_mm[:] = 0
    ensemble_mm.flush()

    for model_name in sorted(view_volumes_by_model.keys()):
        print(f"\n=== Assembling model into ensemble volume: {model_name} ===")
        assemble_model_volume_from_view_volumes(
            ensemble_mm=ensemble_mm,
            view_volume_mms=view_volumes_by_model[model_name],
            T=T,
            H=H,
            W=W,
            disable_multiplanar=disable_multiplanar,
        )
        ensemble_mm.flush()

    return ensemble_mm


# --------------------------
# Skeleton-based interpolation (optional)
# --------------------------

def _neighbors26() -> List[Tuple[int, int, int]]:
    out = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                out.append((dz, dy, dx))
    return out


NEIGH26 = _neighbors26()
KERNEL_3 = np.ones((3, 3, 3), dtype=np.uint8)
STRUCTURE26 = np.ones((3, 3, 3), dtype=bool)


@dataclass(frozen=True)
class EndpointSeed:
    label: int
    point: Tuple[int, int, int]      # (z, y, x)
    direction: np.ndarray            # unit vector in (z, y, x)
    support_radius: float            # robust local radius for this endpoint


@dataclass(frozen=True)
class BridgeCandidate:
    source_label: int
    target_label: int
    source_point: Tuple[int, int, int]
    target_point: Tuple[int, int, int]
    hit_point: Tuple[int, int, int]
    distance: float
    source_radius: float
    target_radius: float
    bridge_radius: float


@dataclass(frozen=True)
class PlannedBridge:
    pair_labels: Tuple[int, int]
    label0: int
    label1: int
    p0: Tuple[int, int, int]
    p1: Tuple[int, int, int]
    radius0: float
    radius1: float
    bridge_radius: float


def skeletonize_volume(mask_bool: np.ndarray) -> np.ndarray:
    """Best-effort 3D skeletonization across skimage versions."""
    arr = np.asarray(mask_bool, dtype=bool)

    try:
        return np.asarray(_skimage_skeletonize(arr, method="lee"), dtype=bool)
    except TypeError:
        pass
    except Exception:
        pass

    if _skimage_skeletonize_3d is not None:
        try:
            return np.asarray(_skimage_skeletonize_3d(arr), dtype=bool)
        except Exception:
            pass

    return np.asarray(_skimage_skeletonize(arr), dtype=bool)


def _skeleton_neighbors(skel: np.ndarray, p: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    z, y, x = p
    out: List[Tuple[int, int, int]] = []
    for dz, dy, dx in NEIGH26:
        zz, yy, xx = z + dz, y + dy, x + dx
        if 0 <= zz < skel.shape[0] and 0 <= yy < skel.shape[1] and 0 <= xx < skel.shape[2]:
            if skel[zz, yy, xx]:
                out.append((zz, yy, xx))
    return out


def _trace_inward_path(
    skel: np.ndarray,
    endpoint: Tuple[int, int, int],
    max_steps: int,
) -> List[Tuple[int, int, int]]:
    """
    Trace inward from a skeleton endpoint to estimate the local tangent, preferring smooth continuation
    if the walk reaches a bifurcation.
    """
    path = [endpoint]
    prev: Optional[Tuple[int, int, int]] = None
    cur = endpoint

    for _ in range(max_steps):
        nbs = _skeleton_neighbors(skel, cur)
        if prev is not None:
            nbs = [n for n in nbs if n != prev]
        if not nbs:
            break
        if len(nbs) == 1 or prev is None:
            nxt = nbs[0]
        else:
            prev_vec = np.asarray(cur, dtype=np.float32) - np.asarray(prev, dtype=np.float32)
            prev_norm = float(np.linalg.norm(prev_vec))
            if prev_norm <= 0:
                nxt = nbs[0]
            else:
                prev_vec /= prev_norm

                def _continuation_score(n: Tuple[int, int, int]) -> float:
                    step_vec = np.asarray(n, dtype=np.float32) - np.asarray(cur, dtype=np.float32)
                    step_norm = float(np.linalg.norm(step_vec))
                    if step_norm <= 0:
                        return -1.0
                    step_vec /= step_norm
                    return float(np.dot(prev_vec, step_vec))

                nxt = max(nbs, key=_continuation_score)
        path.append(nxt)
        prev, cur = cur, nxt

    return path


def _estimate_seed_support_radius(
    dist_sub: np.ndarray,
    path: List[Tuple[int, int, int]],
    extension_radius: int,
) -> float:
    """
    Estimate a robust local support radius for an endpoint by sampling several voxels
    inward along the skeleton path instead of trusting only the terminal endpoint voxel.
    This suppresses tiny bead-like bridges caused by noisy terminal skeleton branches.
    """
    if not path:
        return 1.0

    max_samples = max(2, min(len(path), max(3, min(8, int(extension_radius) + 1))))
    vals: List[float] = []
    for pt in path[:max_samples]:
        v = float(dist_sub[pt])
        if v > 0:
            vals.append(v)

    if not vals:
        return 1.0

    radius_cap = max(1.0, float(extension_radius))
    return max(1.0, min(max(vals), radius_cap))


def _build_endpoint_seeds(
    labels_real: np.ndarray,
    extension_radius: int,
) -> Tuple[List[EndpointSeed], int]:
    """Skeletonize every real 3D object and return endpoint seeds with local tangent directions."""
    objs = ndi.find_objects(labels_real)
    seeds: List[EndpointSeed] = []
    endpoint_count = 0
    direction_depth = max(2, min(8, int(extension_radius)))

    for lbl, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = (labels_real[sl] == lbl)
        if not sub.any():
            continue

        skel = skeletonize_volume(sub)
        if not skel.any():
            continue

        dist_sub = ndi.distance_transform_edt(sub)
        neigh = ndi.convolve(skel.astype(np.uint8), KERNEL_3, mode="constant", cval=0) - skel.astype(np.uint8)
        ep_coords = np.argwhere(np.logical_and(skel, neigh == 1))
        if ep_coords.size == 0:
            continue

        z0, y0, x0 = sl[0].start, sl[1].start, sl[2].start

        for ep in ep_coords:
            ep_t = (int(ep[0]), int(ep[1]), int(ep[2]))
            path = _trace_inward_path(skel, ep_t, max_steps=direction_depth)
            ref = path[-1] if len(path) > 1 else ep_t

            direction = np.asarray(ep_t, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 0:
                nbs = _skeleton_neighbors(skel, ep_t)
                if not nbs:
                    continue
                direction = np.asarray(ep_t, dtype=np.float32) - np.asarray(nbs[0], dtype=np.float32)
                norm = float(np.linalg.norm(direction))
                if norm <= 0:
                    continue
            direction /= norm

            seeds.append(
                EndpointSeed(
                    label=int(lbl),
                    point=(z0 + ep_t[0], y0 + ep_t[1], x0 + ep_t[2]),
                    direction=direction.astype(np.float32),
                    support_radius=_estimate_seed_support_radius(
                        dist_sub=dist_sub,
                        path=path,
                        extension_radius=extension_radius,
                    ),
                )
            )
            endpoint_count += 1

    return seeds, endpoint_count


def _estimate_local_object_radius(
    labels_real: np.ndarray,
    label: int,
    center: Tuple[int, int, int],
    radius_cap: int,
) -> float:
    """
    Approximate the largest integer-radius sphere fully contained in a labeled object around `center`.
    This avoids allocating a full-volume EDT while still producing a stable bridge radius estimate.
    """
    z0, y0, x0 = center
    Z, Y, X = labels_real.shape
    if not (0 <= z0 < Z and 0 <= y0 < Y and 0 <= x0 < X):
        return 0.0
    if int(labels_real[z0, y0, x0]) != int(label):
        return 0.0

    best = 1
    for r in range(2, max(2, int(radius_cap)) + 1):
        offsets = _integer_sphere_offsets(r)
        dz = offsets[:, 0].astype(np.int64, copy=False)
        dy = offsets[:, 1].astype(np.int64, copy=False)
        dx = offsets[:, 2].astype(np.int64, copy=False)

        zz = z0 + dz
        yy = y0 + dy
        xx = x0 + dx
        inside = (
            (zz >= 0) & (zz < Z) &
            (yy >= 0) & (yy < Y) &
            (xx >= 0) & (xx < X)
        )
        if not np.all(inside):
            break
        if not np.all(labels_real[zz, yy, xx] == int(label)):
            break
        best = r

    return float(max(1, best))


def _estimate_target_anchor_and_radius(
    labels_real: np.ndarray,
    target_label: int,
    hit_point: Tuple[int, int, int],
    direction: np.ndarray,
    radius_cap: int,
    max_steps: int,
) -> Tuple[Tuple[int, int, int], float]:
    """
    Starting from the first target intercept, walk a few voxels deeper into the target
    along the connection direction and keep the in-object point with the strongest local support.
    This avoids deriving bridge width from a single surface voxel.
    """
    direction = np.asarray(direction, dtype=np.float32)
    dnorm = float(np.linalg.norm(direction))
    if dnorm <= 0:
        return hit_point, max(1.0, _estimate_local_object_radius(labels_real, target_label, hit_point, radius_cap))

    direction = direction / dnorm
    best_point = hit_point
    best_radius = max(1.0, _estimate_local_object_radius(labels_real, target_label, hit_point, radius_cap))
    Z, Y, X = labels_real.shape
    seen: set[Tuple[int, int, int]] = {hit_point}

    for step in range(1, max(1, int(max_steps)) + 1):
        q = tuple(int(round(hit_point[i] + float(direction[i]) * float(step))) for i in range(3))
        if q in seen:
            continue
        seen.add(q)

        qz, qy, qx = q
        if not (0 <= qz < Z and 0 <= qy < Y and 0 <= qx < X):
            break
        if int(labels_real[qz, qy, qx]) != int(target_label):
            break

        r = _estimate_local_object_radius(labels_real, target_label, q, radius_cap)
        if r > best_radius + 1e-6:
            best_radius = r
            best_point = q

    return best_point, max(1.0, best_radius)


def _sorted_sphere_offsets(radius: int) -> List[Tuple[float, int, int, int]]:
    out: List[Tuple[float, int, int, int]] = []
    r2 = float(radius * radius)
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                d2 = float(dz * dz + dy * dy + dx * dx)
                if d2 <= 0.0 or d2 > r2:
                    continue
                out.append((math.sqrt(d2), dz, dy, dx))
    out.sort(key=lambda item: item[0])
    return out


def _closest_sector_hit(
    labels_real: np.ndarray,
    source_label: int,
    point: Tuple[int, int, int],
    direction: np.ndarray,
    extension_radius: int,
    cone_half_angle_deg: float,
    sphere_offsets: List[Tuple[float, int, int, int]],
) -> Optional[Tuple[Tuple[int, int, int], int, float]]:
    """
    Return the closest intercept inside the spherical sector.
    For cone_half_angle == 0 the sector degenerates to a ray sampled at unit steps.
    """
    z0, y0, x0 = point
    Z, Y, X = labels_real.shape
    direction = np.asarray(direction, dtype=np.float32)
    dnorm = float(np.linalg.norm(direction))
    if dnorm <= 0:
        return None
    direction = direction / dnorm

    if cone_half_angle_deg <= 0.0:
        seen: set[Tuple[int, int, int]] = set()
        for step in range(1, int(extension_radius) + 1):
            q = tuple(int(round(point[i] + float(direction[i]) * float(step))) for i in range(3))
            if q in seen:
                continue
            seen.add(q)
            qz, qy, qx = q
            if not (0 <= qz < Z and 0 <= qy < Y and 0 <= qx < X):
                break
            other = int(labels_real[qz, qy, qx])
            if other != 0 and other != int(source_label):
                return q, other, float(step)
        return None

    cos_thresh = math.cos(math.radians(min(90.0, max(0.0, cone_half_angle_deg))))
    for dist, dz, dy, dx in sphere_offsets:
        dot = (dz * float(direction[0]) + dy * float(direction[1]) + dx * float(direction[2])) / dist
        if dot + 1e-9 < cos_thresh:
            continue
        qz = z0 + dz
        qy = y0 + dy
        qx = x0 + dx
        if not (0 <= qz < Z and 0 <= qy < Y and 0 <= qx < X):
            continue
        other = int(labels_real[qz, qy, qx])
        if other != 0 and other != int(source_label):
            return (qz, qy, qx), other, dist

    return None


def _build_bridge_candidates(
    labels_real: np.ndarray,
    seeds: List[EndpointSeed],
    extension_radius: int,
    cone_half_angle_deg: float,
) -> List[BridgeCandidate]:
    sphere_offsets = _sorted_sphere_offsets(int(extension_radius))
    anchor_depth = max(2, min(6, int(extension_radius)))
    radius_cap = max(1, int(extension_radius))

    candidates: List[BridgeCandidate] = []
    for seed in seeds:
        hit = _closest_sector_hit(
            labels_real=labels_real,
            source_label=seed.label,
            point=seed.point,
            direction=seed.direction,
            extension_radius=int(extension_radius),
            cone_half_angle_deg=float(cone_half_angle_deg),
            sphere_offsets=sphere_offsets,
        )
        if hit is None:
            continue

        hit_point, hit_label, hit_distance = hit
        target_point, target_radius = _estimate_target_anchor_and_radius(
            labels_real=labels_real,
            target_label=int(hit_label),
            hit_point=hit_point,
            direction=seed.direction,
            radius_cap=radius_cap,
            max_steps=anchor_depth,
        )
        bridge_radius = min(float(seed.support_radius), float(target_radius))
        candidates.append(
            BridgeCandidate(
                source_label=int(seed.label),
                target_label=int(hit_label),
                source_point=seed.point,
                target_point=target_point,
                hit_point=hit_point,
                distance=float(hit_distance),
                source_radius=float(seed.support_radius),
                target_radius=float(target_radius),
                bridge_radius=float(bridge_radius),
            )
        )

    return candidates


def _euclidean_distance(p0: Tuple[int, int, int], p1: Tuple[int, int, int]) -> float:
    dz = float(p0[0] - p1[0])
    dy = float(p0[1] - p1[1])
    dx = float(p0[2] - p1[2])
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def _reciprocal_pair_score(
    c_ab: BridgeCandidate,
    c_ba: BridgeCandidate,
) -> Tuple[float, float, float]:
    """
    Prefer reciprocal candidates that point toward each other's supported anchor region,
    then prefer shorter intercepts, then prefer wider connections.
    """
    consistency = (
        _euclidean_distance(c_ab.target_point, c_ba.source_point) +
        _euclidean_distance(c_ba.target_point, c_ab.source_point)
    )
    return (
        float(consistency),
        float(c_ab.distance + c_ba.distance),
        -float(min(c_ab.bridge_radius, c_ba.bridge_radius)),
    )


def _plan_pairwise_bridges(
    candidates: List[BridgeCandidate],
    interpolate_min_radius: float,
) -> Tuple[List[PlannedBridge], Dict[str, int]]:
    """
    Collapse many endpoint-level candidate hits into at most one bridge per unordered object pair.
    This directly suppresses the failure mode where multiple nearby endpoints create many small
    circular bridges instead of one larger bridge.
    """
    by_pair: Dict[Tuple[int, int], List[BridgeCandidate]] = {}
    for cand in candidates:
        key = (
            min(int(cand.source_label), int(cand.target_label)),
            max(int(cand.source_label), int(cand.target_label)),
        )
        by_pair.setdefault(key, []).append(cand)

    planned: List[PlannedBridge] = []
    skipped_by_min_radius = 0

    for pair_key, group in by_pair.items():
        a, b = pair_key
        ab = [c for c in group if int(c.source_label) == int(a) and int(c.target_label) == int(b)]
        ba = [c for c in group if int(c.source_label) == int(b) and int(c.target_label) == int(a)]

        if ab and ba:
            best_pair: Optional[Tuple[BridgeCandidate, BridgeCandidate]] = None
            best_score: Optional[Tuple[float, float, float]] = None
            for c_ab in ab:
                for c_ba in ba:
                    score = _reciprocal_pair_score(c_ab, c_ba)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pair = (c_ab, c_ba)

            assert best_pair is not None
            c_ab, c_ba = best_pair

            radius0 = max(float(c.source_radius) for c in ab)
            radius1 = max(float(c.source_radius) for c in ba)
            bridge_radius = min(radius0, radius1)
            if bridge_radius <= float(interpolate_min_radius):
                skipped_by_min_radius += 1
                continue

            planned.append(
                PlannedBridge(
                    pair_labels=pair_key,
                    label0=int(a),
                    label1=int(b),
                    p0=c_ab.source_point,
                    p1=c_ba.source_point,
                    radius0=float(radius0),
                    radius1=float(radius1),
                    bridge_radius=float(bridge_radius),
                )
            )
            continue

        best = min(group, key=lambda c: (float(c.distance), -float(c.bridge_radius)))

        # With only one directional hit, the target-side radius estimate is usually the least stable
        # because the first intercept occurs at the target surface. Use the source-side support radius
        # to decide whether the bridge is too small, and keep the deeper target anchor only for geometry.
        radius0 = max(float(c.source_radius) for c in group if int(c.source_label) == int(best.source_label))
        target_radius = max(float(c.target_radius) for c in group)
        radius1 = max(radius0, target_radius)
        bridge_radius = radius0
        if bridge_radius <= float(interpolate_min_radius):
            skipped_by_min_radius += 1
            continue

        planned.append(
            PlannedBridge(
                pair_labels=pair_key,
                label0=int(best.source_label),
                label1=int(best.target_label),
                p0=best.source_point,
                p1=best.target_point,
                radius0=float(radius0),
                radius1=float(radius1),
                bridge_radius=float(bridge_radius),
            )
        )

    return planned, {
        "pair_groups": int(len(by_pair)),
        "accepted_connections": int(len(planned)),
        "skipped_by_min_radius": int(skipped_by_min_radius),
    }


_SPHERE_PAINT_CACHE: Dict[int, np.ndarray] = {}


def _integer_sphere_offsets(radius: int) -> np.ndarray:
    cached = _SPHERE_PAINT_CACHE.get(int(radius))
    if cached is not None:
        return cached

    pts: List[Tuple[int, int, int, int]] = []
    r2 = radius * radius
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                d2 = dz * dz + dy * dy + dx * dx
                if d2 <= r2:
                    pts.append((dz, dy, dx, d2))
    arr = np.asarray(pts, dtype=np.int16 if radius <= 127 else np.int32)
    _SPHERE_PAINT_CACHE[int(radius)] = arr
    return arr


def _paint_sphere(mask: np.ndarray, center: Tuple[int, int, int], radius: float) -> int:
    radius = max(0.5, float(radius))
    ir = int(math.ceil(radius))
    offsets = _integer_sphere_offsets(ir)
    rr2 = float(radius * radius) + 1e-6

    z0, y0, x0 = center
    Z, Y, X = mask.shape
    added = 0

    for dz, dy, dx, d2 in offsets.tolist():
        if float(d2) > rr2:
            continue
        zz = z0 + int(dz)
        yy = y0 + int(dy)
        xx = x0 + int(dx)
        if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X and not mask[zz, yy, xx]:
            mask[zz, yy, xx] = True
            added += 1

    return added


def bresenham3d(p0: Tuple[int, int, int], p1: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """3D Bresenham line between integer points p0 and p1 (z,y,x)."""
    z0, y0, x0 = p0
    z1, y1, x1 = p1
    dz = abs(z1 - z0)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sz = 1 if z1 >= z0 else -1
    sy = 1 if y1 >= y0 else -1
    sx = 1 if x1 >= x0 else -1

    if dx >= dy and dx >= dz:
        p1_err = 2 * dy - dx
        p2_err = 2 * dz - dx
        z, y, x = z0, y0, x0
        out = []
        for _ in range(dx + 1):
            out.append((z, y, x))
            if p1_err >= 0:
                y += sy
                p1_err -= 2 * dx
            if p2_err >= 0:
                z += sz
                p2_err -= 2 * dx
            p1_err += 2 * dy
            p2_err += 2 * dz
            x += sx
        return out
    elif dy >= dx and dy >= dz:
        p1_err = 2 * dx - dy
        p2_err = 2 * dz - dy
        z, y, x = z0, y0, x0
        out = []
        for _ in range(dy + 1):
            out.append((z, y, x))
            if p1_err >= 0:
                x += sx
                p1_err -= 2 * dy
            if p2_err >= 0:
                z += sz
                p2_err -= 2 * dy
            p1_err += 2 * dx
            p2_err += 2 * dz
            y += sy
        return out
    else:
        p1_err = 2 * dy - dz
        p2_err = 2 * dx - dz
        z, y, x = z0, y0, x0
        out = []
        for _ in range(dz + 1):
            out.append((z, y, x))
            if p1_err >= 0:
                y += sy
                p1_err -= 2 * dz
            if p2_err >= 0:
                x += sx
                p2_err -= 2 * dz
            p1_err += 2 * dy
            p2_err += 2 * dx
            z += sz
        return out


_PLANE_GRID_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n <= 0:
        return arr
    return arr / n


def _orthonormal_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_u = _normalize_vec(axis)
    if float(np.linalg.norm(axis_u)) <= 0:
        axis_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    if abs(float(axis_u[0])) < 0.9:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    u_axis = np.cross(axis_u, ref)
    if float(np.linalg.norm(u_axis)) <= 1e-6:
        ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        u_axis = np.cross(axis_u, ref)
    u_axis = _normalize_vec(u_axis)
    v_axis = _normalize_vec(np.cross(axis_u, u_axis))
    return axis_u, u_axis, v_axis


def _plane_uv_grid(half_width: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _PLANE_GRID_CACHE.get(int(half_width))
    if cached is not None:
        return cached

    coords = np.arange(-int(half_width), int(half_width) + 1, dtype=np.float32)
    vv, uu = np.meshgrid(coords, coords, indexing="ij")
    _PLANE_GRID_CACHE[int(half_width)] = (uu, vv)
    return uu, vv


def _keep_center_component_2d(mask2d: np.ndarray) -> np.ndarray:
    mask2d = np.asarray(mask2d, dtype=bool)
    if not mask2d.any():
        return mask2d

    labels2d, num = ndi.label(mask2d, structure=np.ones((3, 3), dtype=bool))
    if num <= 1:
        return np.asarray(ndi.binary_fill_holes(mask2d), dtype=bool)

    cy = mask2d.shape[0] // 2
    cx = mask2d.shape[1] // 2
    keep = int(labels2d[cy, cx])
    if keep == 0:
        pts = np.argwhere(labels2d > 0)
        if pts.size == 0:
            return np.zeros_like(mask2d, dtype=bool)
        d2 = (pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2
        py, px = pts[int(np.argmin(d2))]
        keep = int(labels2d[py, px])

    kept = (labels2d == keep)
    return np.asarray(ndi.binary_fill_holes(kept), dtype=bool)


def _mask_touches_border(mask2d: np.ndarray) -> bool:
    if mask2d.size == 0 or not np.any(mask2d):
        return False
    return bool(
        np.any(mask2d[0, :]) or
        np.any(mask2d[-1, :]) or
        np.any(mask2d[:, 0]) or
        np.any(mask2d[:, -1])
    )


def _disk_mask_2d(half_width: int, radius: float) -> np.ndarray:
    radius = max(0.5, float(radius))
    uu, vv = _plane_uv_grid(int(half_width))
    return np.asarray((uu * uu + vv * vv) <= (radius * radius + 1e-6), dtype=bool)


def _signed_distance_2d(mask2d: np.ndarray) -> np.ndarray:
    mask2d = np.asarray(mask2d, dtype=bool)
    inside = ndi.distance_transform_edt(mask2d)
    outside = ndi.distance_transform_edt(~mask2d)
    return np.asarray(inside - outside, dtype=np.float32)


def _best_in_object_axis_anchor(
    labels_real: np.ndarray,
    label: int,
    start: Tuple[int, int, int],
    direction: np.ndarray,
    radius_hint: float,
    max_steps: int,
) -> Tuple[Tuple[int, int, int], float]:
    direction = _normalize_vec(direction)
    if float(np.linalg.norm(direction)) <= 0:
        return start, max(1.0, float(radius_hint))

    start_arr = np.asarray(start, dtype=np.float32)
    Z, Y, X = labels_real.shape
    radius_cap = max(1, int(math.ceil(max(1.0, float(radius_hint))) + 2))

    best_point: Optional[Tuple[int, int, int]] = None
    best_radius = 0.0
    best_step = -1
    seen: set[Tuple[int, int, int]] = set()

    for step in range(0, max(0, int(max_steps)) + 1):
        q = tuple(int(round(float(start_arr[i] + float(direction[i]) * float(step)))) for i in range(3))
        if q in seen:
            continue
        seen.add(q)
        qz, qy, qx = q
        if not (0 <= qz < Z and 0 <= qy < Y and 0 <= qx < X):
            continue
        if int(labels_real[qz, qy, qx]) != int(label):
            continue

        r = float(_estimate_local_object_radius(labels_real, int(label), q, radius_cap))
        if (
            best_point is None or
            r > best_radius + 1e-6 or
            (abs(r - best_radius) <= 1e-6 and step > best_step)
        ):
            best_point = q
            best_radius = r
            best_step = step

    if best_point is None:
        return start, max(1.0, float(radius_hint))
    return best_point, max(1.0, float(best_radius))


def _extract_local_plane_mask(
    labels_real: np.ndarray,
    label: int,
    center: np.ndarray,
    axis: np.ndarray,
    half_width: int,
    plane_half_thickness: float = 0.75,
) -> np.ndarray:
    axis_u, u_axis, v_axis = _orthonormal_basis(axis)
    uu, vv = _plane_uv_grid(int(half_width))

    center = np.asarray(center, dtype=np.float32)
    base = center.reshape(1, 1, 3) + (
        uu[..., None] * u_axis.reshape(1, 1, 3) +
        vv[..., None] * v_axis.reshape(1, 1, 3)
    )

    if plane_half_thickness > 0:
        offsets = (-float(plane_half_thickness), 0.0, float(plane_half_thickness))
    else:
        offsets = (0.0,)

    acc = np.zeros(uu.shape, dtype=bool)
    Z, Y, X = labels_real.shape
    for off in offsets:
        coords = base + axis_u.reshape(1, 1, 3) * float(off)
        zz = np.rint(coords[..., 0]).astype(np.int32)
        yy = np.rint(coords[..., 1]).astype(np.int32)
        xx = np.rint(coords[..., 2]).astype(np.int32)

        valid = (
            (zz >= 0) & (zz < Z) &
            (yy >= 0) & (yy < Y) &
            (xx >= 0) & (xx < X)
        )
        if not np.any(valid):
            continue
        hit = np.zeros_like(acc, dtype=bool)
        hit[valid] = (labels_real[zz[valid], yy[valid], xx[valid]] == int(label))
        acc |= hit

    if not np.any(acc):
        return acc
    return _keep_center_component_2d(acc)


def _paint_plane_section(
    mask: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    section_mask: np.ndarray,
    plane_half_thickness: float = 0.49,
) -> int:
    section_mask = np.asarray(section_mask, dtype=bool)
    if not section_mask.any():
        return 0

    axis_u, u_axis, v_axis = _orthonormal_basis(axis)
    center = np.asarray(center, dtype=np.float32)
    half_width = section_mask.shape[0] // 2
    pts2d = np.argwhere(section_mask)
    if pts2d.size == 0:
        return 0

    vv = (pts2d[:, 0].astype(np.float32) - float(half_width))
    uu = (pts2d[:, 1].astype(np.float32) - float(half_width))
    base = (
        center.reshape(1, 3) +
        uu[:, None] * u_axis.reshape(1, 3) +
        vv[:, None] * v_axis.reshape(1, 3)
    )

    if plane_half_thickness > 0:
        offsets = (-float(plane_half_thickness), 0.0, float(plane_half_thickness))
    else:
        offsets = (0.0,)

    clouds = [base + axis_u.reshape(1, 3) * float(off) for off in offsets]
    coords = np.concatenate(clouds, axis=0)
    ijk = np.rint(coords).astype(np.int32)

    Z, Y, X = mask.shape
    valid = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < Z) &
        (ijk[:, 1] >= 0) & (ijk[:, 1] < Y) &
        (ijk[:, 2] >= 0) & (ijk[:, 2] < X)
    )
    if not np.any(valid):
        return 0

    pts = np.unique(ijk[valid], axis=0)
    current = mask[pts[:, 0], pts[:, 1], pts[:, 2]]
    mask[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    return int(np.count_nonzero(~current))


def _paint_connection_tube(
    mask: np.ndarray,
    labels_real: np.ndarray,
    label0: int,
    label1: int,
    p0: Tuple[int, int, int],
    p1: Tuple[int, int, int],
    radius0: float,
    radius1: Optional[float] = None,
) -> int:
    if radius1 is None:
        radius1 = radius0

    radius0 = max(0.5, float(radius0))
    radius1 = max(0.5, float(radius1))

    p0_arr = np.asarray(p0, dtype=np.float32)
    p1_arr = np.asarray(p1, dtype=np.float32)
    axis = p1_arr - p0_arr
    axis_len = float(np.linalg.norm(axis))
    if axis_len <= 0:
        return 0
    axis_u = axis / axis_len

    anchor_steps0 = max(2, min(12, int(math.ceil(radius0 * 2.0)) + 2))
    anchor_steps1 = max(2, min(12, int(math.ceil(radius1 * 2.0)) + 2))
    c0, local_r0 = _best_in_object_axis_anchor(
        labels_real=labels_real,
        label=int(label0),
        start=p0,
        direction=-axis_u,
        radius_hint=radius0,
        max_steps=anchor_steps0,
    )
    c1, local_r1 = _best_in_object_axis_anchor(
        labels_real=labels_real,
        label=int(label1),
        start=p1,
        direction=axis_u,
        radius_hint=radius1,
        max_steps=anchor_steps1,
    )

    c0_arr = np.asarray(c0, dtype=np.float32)
    c1_arr = np.asarray(c1, dtype=np.float32)
    section_axis = c1_arr - c0_arr
    section_len = float(np.linalg.norm(section_axis))
    if section_len <= 0:
        section_axis = axis_u.copy()
        section_len = axis_len
    else:
        section_axis = section_axis / section_len

    base_radius = max(radius0, radius1, float(local_r0), float(local_r1))
    half_width = max(3, int(math.ceil(base_radius * 2.0)) + 1)
    max_half_width = max(half_width, min(96, int(math.ceil(base_radius * 4.0)) + 6))

    while True:
        section0 = _extract_local_plane_mask(
            labels_real=labels_real,
            label=int(label0),
            center=c0_arr,
            axis=section_axis,
            half_width=half_width,
            plane_half_thickness=0.75,
        )
        section1 = _extract_local_plane_mask(
            labels_real=labels_real,
            label=int(label1),
            center=c1_arr,
            axis=section_axis,
            half_width=half_width,
            plane_half_thickness=0.75,
        )
        if (
            half_width >= max_half_width or
            (not _mask_touches_border(section0) and not _mask_touches_border(section1))
        ):
            break
        half_width = min(max_half_width, max(half_width + 1, half_width * 2))

    if not np.any(section0):
        section0 = _disk_mask_2d(half_width, max(radius0, float(local_r0)))
    if not np.any(section1):
        section1 = _disk_mask_2d(half_width, max(radius1, float(local_r1)))

    section0 = _keep_center_component_2d(section0)
    section1 = _keep_center_component_2d(section1)

    sdf0 = _signed_distance_2d(section0)
    sdf1 = _signed_distance_2d(section1)

    steps = max(1, int(math.ceil(section_len)))
    if steps <= 1:
        alphas = [0.5]
    else:
        alphas = [float(idx) / float(steps) for idx in range(1, steps)]

    added = 0
    for alpha in alphas:
        center = (1.0 - alpha) * c0_arr + alpha * c1_arr
        section = ((1.0 - alpha) * sdf0 + alpha * sdf1) >= 0.0
        section = _keep_center_component_2d(section)
        added += _paint_plane_section(
            mask=mask,
            center=center,
            axis=section_axis,
            section_mask=section,
            plane_half_thickness=0.49,
        )

    return int(added)


def interpolate_spherical_sector_pass(
    mask_u8: np.ndarray,
    extension_radius: int,
    cone_half_angle_deg: float,
    interpolate_min_radius: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    One interpolation pass using spherical-sector interception from skeleton endpoints.

    Rules implemented:
      - The current mask at the start of the pass is the set of "real" sections for this pass.
      - Sector interception is computed only against those real sections.
      - New interpolation added during this pass is transparent to the search, and only becomes "real"
        in the next pass (handled by the caller).
      - If multiple objects are intercepted, the closest intercept is used.
      - Multiple endpoint-level hits between the same unordered object pair are collapsed into at most
        one painted bridge for the pass, which suppresses the common "many small circular bridges"
        failure mode.
      - Spherical sectors are used only to choose connection targets. The bridge geometry itself is
        produced by linear interpolation between local real cross-sectional masks, which yields tubes
        instead of chains of nearly spherical blobs.
    """
    if extension_radius <= 0:
        return np.asarray(mask_u8, dtype=np.uint8), {
            "num_objects": 0,
            "num_endpoints": 0,
            "candidate_connections": 0,
            "pair_groups": 0,
            "accepted_connections": 0,
            "skipped_by_min_radius": 0,
            "added_voxels": 0,
            "skipped": True,
        }

    pass_real = np.asarray(mask_u8, dtype=bool)
    labels_real, num_objects = ndi.label(pass_real, structure=STRUCTURE26)
    if num_objects <= 1:
        return pass_real.astype(np.uint8), {
            "num_objects": int(num_objects),
            "num_endpoints": 0,
            "candidate_connections": 0,
            "pair_groups": 0,
            "accepted_connections": 0,
            "skipped_by_min_radius": 0,
            "added_voxels": 0,
            "skipped": num_objects <= 1,
        }

    seeds, num_endpoints = _build_endpoint_seeds(labels_real, extension_radius=extension_radius)
    if not seeds:
        return pass_real.astype(np.uint8), {
            "num_objects": int(num_objects),
            "num_endpoints": int(num_endpoints),
            "candidate_connections": 0,
            "pair_groups": 0,
            "accepted_connections": 0,
            "skipped_by_min_radius": 0,
            "added_voxels": 0,
            "skipped": False,
        }

    candidates = _build_bridge_candidates(
        labels_real=labels_real,
        seeds=seeds,
        extension_radius=int(extension_radius),
        cone_half_angle_deg=float(cone_half_angle_deg),
    )
    planned_bridges, plan_stats = _plan_pairwise_bridges(
        candidates=candidates,
        interpolate_min_radius=float(interpolate_min_radius),
    )

    work_mask = pass_real.copy()
    added_voxels = 0
    for bridge in planned_bridges:
        added_voxels += _paint_connection_tube(
            work_mask,
            labels_real=labels_real,
            label0=int(bridge.label0),
            label1=int(bridge.label1),
            p0=bridge.p0,
            p1=bridge.p1,
            radius0=bridge.radius0,
            radius1=bridge.radius1,
        )

    stats: Dict[str, object] = {
        "num_objects": int(num_objects),
        "num_endpoints": int(num_endpoints),
        "candidate_connections": int(len(candidates)),
        "pair_groups": int(plan_stats["pair_groups"]),
        "accepted_connections": int(plan_stats["accepted_connections"]),
        "skipped_by_min_radius": int(plan_stats["skipped_by_min_radius"]),
        "added_voxels": int(added_voxels),
        "skipped": False,
    }
    return work_mask.astype(np.uint8), stats


@dataclass(frozen=True)
class SliceEndpointSeed:
    label: int
    point: Tuple[int, int, int]  # (slice, row, col)
    direction_sign: int          # -1 or +1 along the slice axis


@dataclass(frozen=True)
class SliceProjectionCandidate:
    source_label: int
    target_label: int
    source_point: Tuple[int, int, int]
    target_point: Tuple[int, int, int]
    slice_distance: int


def _nearest_point_in_mask(mask2d: np.ndarray, ref_yx: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return None
    d2 = (ys.astype(np.int64) - int(ref_yx[0])) ** 2 + (xs.astype(np.int64) - int(ref_yx[1])) ** 2
    idx = int(np.argmin(d2))
    return int(ys[idx]), int(xs[idx])


def _component_mask_and_anchor(mask2d: np.ndarray, point_yx: Tuple[int, int]) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    mask2d = np.asarray(mask2d, dtype=bool)
    if not mask2d.any():
        return np.zeros(mask2d.shape, dtype=bool), None

    labels2d, num = ndi.label(mask2d, structure=np.ones((3, 3), dtype=bool))
    if num <= 0:
        return np.zeros(mask2d.shape, dtype=bool), None

    py = int(np.clip(int(point_yx[0]), 0, mask2d.shape[0] - 1))
    px = int(np.clip(int(point_yx[1]), 0, mask2d.shape[1] - 1))
    lbl = int(labels2d[py, px])
    if lbl <= 0:
        nearest = _nearest_point_in_mask(mask2d, (py, px))
        if nearest is None:
            return np.zeros(mask2d.shape, dtype=bool), None
        lbl = int(labels2d[nearest[0], nearest[1]])
        py, px = nearest

    comp = labels2d == lbl
    anchor = _nearest_point_in_mask(comp, (py, px))
    return comp.astype(bool), anchor


def _component_centroid_anchor(mask2d: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return None
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    d2 = (ys.astype(np.float32) - cy) ** 2 + (xs.astype(np.float32) - cx) ** 2
    idx = int(np.argmin(d2))
    return int(ys[idx]), int(xs[idx])


def _component_max_radius(mask2d: np.ndarray) -> float:
    if not np.any(mask2d):
        return 0.0
    return float(np.max(ndi.distance_transform_edt(np.asarray(mask2d, dtype=bool))))


def _follow_branch_component(
    slice_mask: np.ndarray,
    prev_component_mask: np.ndarray,
    prev_anchor: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    slice_mask = np.asarray(slice_mask, dtype=bool)
    if not slice_mask.any():
        return np.zeros(slice_mask.shape, dtype=bool), None

    labels2d, num = ndi.label(slice_mask, structure=np.ones((3, 3), dtype=bool))
    if num <= 0:
        return np.zeros(slice_mask.shape, dtype=bool), None

    if num == 1:
        comp = labels2d == 1
        return comp.astype(bool), _nearest_point_in_mask(comp, prev_anchor)

    dil_prev = ndi.binary_dilation(np.asarray(prev_component_mask, dtype=bool), structure=np.ones((3, 3), dtype=bool))

    best_lbl = 0
    best_score: Optional[Tuple[int, int, int]] = None
    for lbl in range(1, int(num) + 1):
        comp = labels2d == lbl
        overlap = int(np.count_nonzero(comp & dil_prev))
        anchor = _nearest_point_in_mask(comp, prev_anchor)
        if anchor is None:
            continue
        d2 = int((anchor[0] - int(prev_anchor[0])) ** 2 + (anchor[1] - int(prev_anchor[1])) ** 2)
        area = int(np.count_nonzero(comp))
        score = (overlap, -d2, area)
        if best_score is None or score > best_score:
            best_score = score
            best_lbl = int(lbl)

    if best_lbl <= 0:
        return np.zeros(slice_mask.shape, dtype=bool), None

    comp = labels2d == best_lbl
    return comp.astype(bool), _nearest_point_in_mask(comp, prev_anchor)


def _component_to_local_canvas(mask2d: np.ndarray, anchor_yx: Tuple[int, int], half_width: int) -> np.ndarray:
    size = int(2 * half_width + 1)
    local = np.zeros((size, size), dtype=bool)
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return local

    yy = ys.astype(np.int64) - int(anchor_yx[0]) + int(half_width)
    xx = xs.astype(np.int64) - int(anchor_yx[1]) + int(half_width)
    valid = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
    local[yy[valid], xx[valid]] = True
    return local


def _paste_local_mask_onto_slice(dest_slice: np.ndarray, local_mask: np.ndarray, center_yx: Tuple[float, float]) -> int:
    if not np.any(local_mask):
        return 0

    size_y, size_x = local_mask.shape
    hh_y = size_y // 2
    hh_x = size_x // 2
    cy = int(round(float(center_yx[0])))
    cx = int(round(float(center_yx[1])))

    y0 = cy - hh_y
    x0 = cx - hh_x
    y1 = y0 + size_y
    x1 = x0 + size_x

    dy0 = max(0, -y0)
    dx0 = max(0, -x0)
    dy1 = max(0, y1 - dest_slice.shape[0])
    dx1 = max(0, x1 - dest_slice.shape[1])

    src_y0 = dy0
    src_y1 = size_y - dy1
    src_x0 = dx0
    src_x1 = size_x - dx1

    dst_y0 = y0 + dy0
    dst_y1 = y1 - dy1
    dst_x0 = x0 + dx0
    dst_x1 = x1 - dx1

    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return 0

    patch = np.asarray(local_mask[src_y0:src_y1, src_x0:src_x1], dtype=bool)
    current = np.asarray(dest_slice[dst_y0:dst_y1, dst_x0:dst_x1], dtype=bool)
    added = int(np.count_nonzero(patch & (~current)))
    dest_slice[dst_y0:dst_y1, dst_x0:dst_x1] |= patch
    return added


def _local_half_width_for_components(
    source_component: np.ndarray,
    source_anchor: Tuple[int, int],
    target_component: np.ndarray,
    target_anchor: Tuple[int, int],
) -> int:
    max_extent = 0.0
    for comp, anchor in ((source_component, source_anchor), (target_component, target_anchor)):
        ys, xs = np.nonzero(comp)
        if ys.size == 0:
            continue
        max_extent = max(
            max_extent,
            float(np.max(np.abs(ys.astype(np.int64) - int(anchor[0])))),
            float(np.max(np.abs(xs.astype(np.int64) - int(anchor[1])))),
        )

    max_extent = max(
        max_extent,
        float(abs(int(target_anchor[0]) - int(source_anchor[0]))),
        float(abs(int(target_anchor[1]) - int(source_anchor[1]))),
    )
    return max(4, int(math.ceil(max_extent)) + 4)


def _build_slice_endpoint_seeds(
    labels_real: np.ndarray,
    extension_slices: int,
) -> Tuple[List[SliceEndpointSeed], int]:
    objs = ndi.find_objects(labels_real)
    seeds: List[SliceEndpointSeed] = []
    direction_depth = max(2, min(8, int(extension_slices) + 1))

    for lbl, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = labels_real[sl] == lbl
        if not np.any(sub):
            continue

        grouped: Dict[Tuple[int, int, int, int, int], SliceEndpointSeed] = {}
        slice_start = int(sl[0].start)
        slice_stop = int(sl[0].stop) - 1

        skel = skeletonize_volume(sub)
        if np.any(skel):
            neigh = ndi.convolve(skel.astype(np.uint8), KERNEL_3, mode="constant", cval=0) - skel.astype(np.uint8)
            ep_coords = np.argwhere(np.logical_and(skel, neigh == 1))
        else:
            ep_coords = np.zeros((0, 3), dtype=np.int64)

        for ep in ep_coords:
            ep_t = (int(ep[0]), int(ep[1]), int(ep[2]))
            path = _trace_inward_path(skel, ep_t, max_steps=direction_depth)
            ref = path[-1] if len(path) > 1 else ep_t
            outward = np.asarray(ep_t, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
            if float(abs(outward[0])) > 1e-6:
                direction_sign = 1 if float(outward[0]) > 0 else -1
            else:
                gz = slice_start + int(ep_t[0])
                dist_min = abs(gz - slice_start)
                dist_max = abs(slice_stop - gz)
                direction_sign = -1 if dist_min <= dist_max else 1

            gpoint = (slice_start + int(ep_t[0]), int(sl[1].start) + int(ep_t[1]), int(sl[2].start) + int(ep_t[2]))
            comp, _ = _component_mask_and_anchor(labels_real[gpoint[0]] == lbl, (gpoint[1], gpoint[2]))
            cent = _component_centroid_anchor(comp)
            if cent is None:
                continue
            key = (int(lbl), int(gpoint[0]), int(direction_sign), int(cent[0]), int(cent[1]))
            grouped[key] = SliceEndpointSeed(label=int(lbl), point=gpoint, direction_sign=int(direction_sign))

        if not grouped:
            slice_any = np.any(sub, axis=(1, 2))
            slice_indices = np.flatnonzero(slice_any)
            if slice_indices.size:
                extremes: List[Tuple[int, int]] = []
                first_local = int(slice_indices[0])
                last_local = int(slice_indices[-1])
                extremes.append((first_local, -1))
                if last_local != first_local:
                    extremes.append((last_local, 1))
                else:
                    extremes.append((last_local, 1))

                for local_slice_idx, direction_sign in extremes:
                    global_slice_idx = slice_start + int(local_slice_idx)
                    labels2d, num = ndi.label(sub[local_slice_idx], structure=np.ones((3, 3), dtype=bool))
                    for comp_lbl in range(1, int(num) + 1):
                        comp = labels2d == comp_lbl
                        cent = _component_centroid_anchor(comp)
                        if cent is None:
                            continue
                        gpoint = (int(global_slice_idx), int(sl[1].start) + int(cent[0]), int(sl[2].start) + int(cent[1]))
                        key = (int(lbl), int(gpoint[0]), int(direction_sign), int(gpoint[1]), int(gpoint[2]))
                        grouped[key] = SliceEndpointSeed(label=int(lbl), point=gpoint, direction_sign=int(direction_sign))

        seeds.extend(grouped.values())

    return seeds, int(len(seeds))


def _find_slice_projection_candidate(
    labels_real: np.ndarray,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
) -> Optional[SliceProjectionCandidate]:
    if int(max_slice_distance) <= 0:
        return None

    s0, y0, x0 = seed.point
    source_component, source_anchor = _component_mask_and_anchor(labels_real[s0] == int(seed.label), (y0, x0))
    if source_anchor is None or not np.any(source_component):
        return None

    sdf = _signed_distance_2d(source_component)
    slope = math.tan(math.radians(float(search_angle_deg)))
    num_slices = labels_real.shape[0]

    for step in range(1, int(max_slice_distance) + 1):
        s = int(s0 + int(seed.direction_sign) * step)
        if s < 0 or s >= num_slices:
            break

        threshold = -float(slope) * float(step)
        projection = sdf >= threshold
        if not np.any(projection):
            if float(search_angle_deg) < 0.0:
                break
            continue

        labels2d = labels_real[s]
        overlap = projection & (labels2d > 0) & (labels2d != int(seed.label))
        if not np.any(overlap):
            continue

        ys, xs = np.nonzero(overlap)
        lbls = labels2d[ys, xs].astype(np.int64, copy=False)

        best: Optional[Tuple[int, int, int, int]] = None
        for target_label in np.unique(lbls):
            if int(target_label) <= 0 or int(target_label) == int(seed.label):
                continue
            use = lbls == int(target_label)
            ys_t = ys[use]
            xs_t = xs[use]
            if ys_t.size == 0:
                continue
            d2 = (ys_t.astype(np.int64) - int(source_anchor[0])) ** 2 + (xs_t.astype(np.int64) - int(source_anchor[1])) ** 2
            idx = int(np.argmin(d2))
            cand = (int(d2[idx]), int(target_label), int(ys_t[idx]), int(xs_t[idx]))
            if best is None or cand < best:
                best = cand

        if best is None:
            continue

        _, target_label, ty, tx = best
        return SliceProjectionCandidate(
            source_label=int(seed.label),
            target_label=int(target_label),
            source_point=(int(s0), int(y0), int(x0)),
            target_point=(int(s), int(ty), int(tx)),
            slice_distance=int(step),
        )

    return None


def _collect_walkback_source_points(
    labels_real: np.ndarray,
    label: int,
    start_point: Tuple[int, int, int],
    direction_sign: int,
    walk_back: int,
) -> List[Tuple[int, int, int]]:
    if int(walk_back) <= 0:
        return []

    s0, y0, x0 = start_point
    current_component, current_anchor = _component_mask_and_anchor(labels_real[s0] == int(label), (y0, x0))
    if current_anchor is None or not np.any(current_component):
        return []

    out: List[Tuple[int, int, int]] = []
    current_slice = int(s0)
    num_slices = labels_real.shape[0]

    for _ in range(int(walk_back)):
        next_slice = int(current_slice - int(direction_sign))
        if next_slice < 0 or next_slice >= num_slices:
            break

        next_slice_mask = labels_real[next_slice] == int(label)
        if not np.any(next_slice_mask):
            break

        next_component, next_anchor = _follow_branch_component(next_slice_mask, current_component, current_anchor)
        if next_anchor is None or not np.any(next_component):
            break

        out.append((int(next_slice), int(next_anchor[0]), int(next_anchor[1])))
        current_slice = int(next_slice)
        current_component = next_component
        current_anchor = next_anchor

    return out


def _paint_linear_slice_bridge(
    bridge_volume: np.ndarray,
    labels_real: np.ndarray,
    source_label: int,
    target_label: int,
    source_point: Tuple[int, int, int],
    target_point: Tuple[int, int, int],
) -> int:
    s0, y0, x0 = source_point
    s1, y1, x1 = target_point
    if int(s0) == int(s1):
        return 0

    source_component, source_anchor = _component_mask_and_anchor(labels_real[s0] == int(source_label), (y0, x0))
    target_component, target_anchor = _component_mask_and_anchor(labels_real[s1] == int(target_label), (y1, x1))
    if source_anchor is None or target_anchor is None:
        return 0
    if not np.any(source_component) or not np.any(target_component):
        return 0

    half_width = _local_half_width_for_components(source_component, source_anchor, target_component, target_anchor)
    source_local = _component_to_local_canvas(source_component, source_anchor, half_width)
    target_local = _component_to_local_canvas(target_component, target_anchor, half_width)
    if not np.any(source_local) or not np.any(target_local):
        return 0

    sdf0 = _signed_distance_2d(source_local)
    sdf1 = _signed_distance_2d(target_local)

    steps = int(abs(int(s1) - int(s0)))
    if steps <= 0:
        return 0

    sign = 1 if int(s1) > int(s0) else -1
    added = 0
    for idx in range(1, steps):
        alpha = float(idx) / float(steps)
        section = ((1.0 - alpha) * sdf0 + alpha * sdf1) >= 0.0
        if not np.any(section):
            continue
        section = _keep_center_component_2d(section)
        s = int(s0 + sign * idx)
        center = (
            (1.0 - alpha) * float(source_anchor[0]) + alpha * float(target_anchor[0]),
            (1.0 - alpha) * float(source_anchor[1]) + alpha * float(target_anchor[1]),
        )
        added += _paste_local_mask_onto_slice(bridge_volume[s], section, center)

    return int(added)


def interpolate_view_volume_pass(
    mask_u8: np.ndarray,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolate_min_radius: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if int(max_slice_distance) <= 0:
        return np.asarray(mask_u8, dtype=np.uint8), {
            "num_objects": 0,
            "num_endpoints": 0,
            "candidate_connections": 0,
            "accepted_connections": 0,
            "walk_back_bridges": 0,
            "skipped_by_min_radius": 0,
            "added_voxels": 0,
            "skipped": True,
        }

    pass_real = np.asarray(mask_u8, dtype=bool)
    labels_real, num_objects = ndi.label(pass_real, structure=STRUCTURE26)
    if int(num_objects) <= 1:
        return pass_real.astype(np.uint8), {
            "num_objects": int(num_objects),
            "num_endpoints": 0,
            "candidate_connections": 0,
            "accepted_connections": 0,
            "walk_back_bridges": 0,
            "skipped_by_min_radius": 0,
            "added_voxels": 0,
            "skipped": int(num_objects) <= 1,
        }

    seeds, num_endpoints = _build_slice_endpoint_seeds(labels_real, extension_slices=max_slice_distance)
    if not seeds:
        return pass_real.astype(np.uint8), {
            "num_objects": int(num_objects),
            "num_endpoints": int(num_endpoints),
            "candidate_connections": 0,
            "accepted_connections": 0,
            "walk_back_bridges": 0,
            "skipped_by_min_radius": 0,
            "added_voxels": 0,
            "skipped": False,
        }

    bridge_volume = np.zeros(pass_real.shape, dtype=bool)
    candidate_connections = 0
    accepted_connections = 0
    walk_back_bridges = 0
    skipped_by_min_radius = 0
    added_voxels = 0

    for seed in seeds:
        candidate = _find_slice_projection_candidate(
            labels_real=labels_real,
            seed=seed,
            max_slice_distance=int(max_slice_distance),
            search_angle_deg=float(search_angle_deg),
        )
        if candidate is None:
            continue

        candidate_connections += 1
        source_points = [candidate.source_point] + _collect_walkback_source_points(
            labels_real=labels_real,
            label=int(candidate.source_label),
            start_point=candidate.source_point,
            direction_sign=int(seed.direction_sign),
            walk_back=int(interpolation_walk_back),
        )

        target_component, target_anchor = _component_mask_and_anchor(
            labels_real[candidate.target_point[0]] == int(candidate.target_label),
            (candidate.target_point[1], candidate.target_point[2]),
        )
        target_radius = _component_max_radius(target_component) if target_anchor is not None else 0.0

        for walk_idx, src_point in enumerate(source_points):
            source_component, source_anchor = _component_mask_and_anchor(
                labels_real[src_point[0]] == int(candidate.source_label),
                (src_point[1], src_point[2]),
            )
            source_radius = _component_max_radius(source_component) if source_anchor is not None else 0.0
            bridge_radius = min(float(source_radius), float(target_radius))
            if bridge_radius <= float(interpolate_min_radius):
                skipped_by_min_radius += 1
                continue

            accepted_connections += 1
            if walk_idx > 0:
                walk_back_bridges += 1

            added_voxels += _paint_linear_slice_bridge(
                bridge_volume=bridge_volume,
                labels_real=labels_real,
                source_label=int(candidate.source_label),
                target_label=int(candidate.target_label),
                source_point=src_point,
                target_point=candidate.target_point,
            )

    work = np.asarray(pass_real | bridge_volume, dtype=np.uint8)
    stats: Dict[str, object] = {
        "num_objects": int(num_objects),
        "num_endpoints": int(num_endpoints),
        "candidate_connections": int(candidate_connections),
        "accepted_connections": int(accepted_connections),
        "walk_back_bridges": int(walk_back_bridges),
        "skipped_by_min_radius": int(skipped_by_min_radius),
        "added_voxels": int(added_voxels),
        "skipped": False,
    }
    return work.astype(np.uint8), stats


# --------------------------
# Final outputs
# --------------------------

def write_binary_outputs(
    mask_u8: np.ndarray,  # (T,H,W) 0/1
    out_dir: Path,
    stem: str,
    fps: float,
    *,
    tiff_subdir: str = "binary_masks",
    tiff_prefix: Optional[str] = None,
    video_name: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Write TIFF sequence + FFV1 MKV for a binary mask volume."""
    T, H, W = mask_u8.shape
    tiff_dir = out_dir / tiff_subdir
    tiff_dir.mkdir(parents=True, exist_ok=True)

    if tiff_prefix is None:
        tiff_prefix = f"{stem}_Binary"
    if video_name is None:
        video_name = f"{stem}_Binary.mkv"

    for t in tqdm(range(T), desc=f"Writing binary TIFF sequence ({tiff_subdir})"):
        img = (mask_u8[t] * 255).astype(np.uint8)
        tifffile.imwrite(str(tiff_dir / f"{tiff_prefix}_{t+1:04d}.tiff"), img)

    vid_path = out_dir / video_name
    proc = ffmpeg_rawvideo_writer(
        vid_path,
        width=W,
        height=H,
        fps=fps,
        pix_fmt_in="gray",
        codec="ffv1",
        pix_fmt_out="gray",
    )
    try:
        assert proc.stdin is not None
        for t in tqdm(range(T), desc=f"Writing binary MKV ({video_name})"):
            img = (mask_u8[t] * 255).astype(np.uint8)
            proc.stdin.write(img.tobytes())
    finally:
        close_ffmpeg_writer(proc)

    return tiff_dir, vid_path


def write_overlay_video(
    volume_rgb: np.memmap,  # (T,H,W,3) RGB
    mask_u8: np.ndarray,    # (T,H,W) 0/1
    out_path: Path,
    fps: float,
) -> None:
    """Overlay blue masks (50% alpha) on original transverse frames."""
    T, H, W, _ = volume_rgb.shape
    assert mask_u8.shape == (T, H, W)

    proc = ffmpeg_rawvideo_writer(
        out_path,
        width=W,
        height=H,
        fps=fps,
        pix_fmt_in="rgb24",
        codec="ffv1",
        pix_fmt_out="yuv444p",
    )

    blue = np.array([0, 0, 255], dtype=np.uint8)  # RGB blue

    try:
        assert proc.stdin is not None
        for t in tqdm(range(T), desc=f"Writing overlay video ({out_path.name})"):
            frame = np.asarray(volume_rgb[t]).copy()
            m = mask_u8[t].astype(bool)
            if m.any():
                frame[m] = ((frame[m].astype(np.uint16) + blue.astype(np.uint16)) // 2).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
    finally:
        close_ffmpeg_writer(proc)


def mask_to_yolo_polygons(mask01: np.ndarray) -> List[List[Tuple[float, float]]]:
    """Convert a binary mask (H,W) to a list of external polygons with normalized coords."""
    h, w = mask01.shape
    m = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[List[Tuple[float, float]]] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        if approx is None or len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        polys.append([(float(x) / float(w), float(y) / float(h)) for (x, y) in pts])
    return polys


def write_yolo_labels(
    mask_u8: np.ndarray,
    out_dir: Path,
    stem: str,
    *,
    labels_subdir: str = "labels",
    frame_prefix: Optional[str] = None,
) -> Path:
    """Write YOLO seg labels, including blank files for frames with no detections."""
    labels_dir = out_dir / labels_subdir
    labels_dir.mkdir(parents=True, exist_ok=True)
    T, H, W = mask_u8.shape

    if frame_prefix is None:
        frame_prefix = stem

    for t in tqdm(range(T), desc=f"Writing YOLO labels ({labels_subdir})"):
        fp = labels_dir / f"{frame_prefix}_{t+1:04d}.txt"
        m = (mask_u8[t] > 0)
        if not m.any():
            fp.write_text("")
            continue
        polys = mask_to_yolo_polygons(m.astype(np.uint8))
        if not polys:
            fp.write_text("")
            continue
        lines = []
        for poly in polys:
            coords = []
            for (x, y) in poly:
                coords.append(f"{x:.6f}")
                coords.append(f"{y:.6f}")
            lines.append("0 " + " ".join(coords))
        fp.write_text("\n".join(lines) + "\n")
    return labels_dir


def write_result_set(
    *,
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    out_dir: Path,
    stem: str,
    fps: float,
    save_binary: bool,
    save_labels: bool,
    tag: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Write one result set. When tag is None, standard final-output names are used.
    For troubleshooting tags, outputs are suffixed with _{tag} and placed into tag-specific folders.
    """
    if tag is None:
        overlay_name = f"{stem}_Overlay.mkv"
        binary_subdir = "binary_masks"
        binary_prefix = f"{stem}_Binary"
        binary_video = f"{stem}_Binary.mkv"
        labels_subdir = "labels"
        labels_prefix = stem
    else:
        overlay_name = f"{stem}_Overlay_{tag}.mkv"
        binary_subdir = f"binary_masks_{tag.lower()}"
        binary_prefix = f"{stem}_Binary_{tag}"
        binary_video = f"{stem}_Binary_{tag}.mkv"
        labels_subdir = f"labels_{tag.lower()}"
        labels_prefix = f"{stem}_{tag}"

    overlay_path = out_dir / overlay_name
    write_overlay_video(volume_rgb, mask_u8, overlay_path, fps=fps)

    result_paths: Dict[str, Path] = {"overlay": overlay_path}

    if save_binary:
        tiff_dir, vid_path = write_binary_outputs(
            mask_u8,
            out_dir=out_dir,
            stem=stem,
            fps=fps,
            tiff_subdir=binary_subdir,
            tiff_prefix=binary_prefix,
            video_name=binary_video,
        )
        result_paths["binary_tiff_dir"] = tiff_dir
        result_paths["binary_video"] = vid_path

    if save_labels:
        labels_dir = write_yolo_labels(
            mask_u8,
            out_dir=out_dir,
            stem=stem,
            labels_subdir=labels_subdir,
            frame_prefix=labels_prefix,
        )
        result_paths["labels_dir"] = labels_dir

    return result_paths


DEFAULT_LABEL_PATTERN = "labels/{Filename}_%04d.txt"
DEFAULT_BINARY_PATTERN = "binary_masks/{Filename}_Binary_%04d.tiff"


def _resolve_output_pattern(pattern_value: Optional[str], default_pattern: str, out_dir: Path, stem: str) -> Optional[Path]:
    if pattern_value is None:
        return None
    pattern = default_pattern if str(pattern_value) == "__DEFAULT__" else str(pattern_value)
    pattern = pattern.replace("{Filename}", stem)
    path = Path(pattern)
    if not path.is_absolute():
        path = out_dir / path
    return path


def _tag_regular_path(path: Path, tag: str) -> Path:
    suffix = "".join(path.suffixes)
    base = path.name[:-len(suffix)] if suffix else path.name
    return path.with_name(f"{base}_{tag}{suffix}")


def _tag_frame_pattern(path: Path, tag: str) -> Path:
    parent = path.parent
    if parent.name:
        parent = parent.with_name(f"{parent.name}_{tag.lower()}")

    name = path.name
    m = re.search(r"(%0\d+d)", name)
    if m is not None:
        prefix = name[:m.start()]
        if prefix.endswith("_"):
            name = prefix + f"{tag}_" + name[m.start():]
        else:
            name = prefix + f"_{tag}_" + name[m.start():]
    else:
        suffix = "".join(path.suffixes)
        base = name[:-len(suffix)] if suffix else name
        name = f"{base}_{tag}{suffix}"
    return parent / name


def _format_frame_path(pattern_path: Path, frame_idx_1based: int) -> Path:
    as_str = str(pattern_path)
    if "%" in as_str:
        try:
            return Path(as_str % int(frame_idx_1based))
        except TypeError:
            pass
    return pattern_path.with_name(f"{pattern_path.stem}_{int(frame_idx_1based):04d}{pattern_path.suffix}")


def write_yolo_labels_from_pattern(mask_u8: np.ndarray, pattern_path: Path) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W = mask_u8.shape
    for t in tqdm(range(T), desc=f"Writing YOLO labels ({pattern_path.parent.name})"):
        fp = _format_frame_path(pattern_path, t + 1)
        fp.parent.mkdir(parents=True, exist_ok=True)
        m = mask_u8[t] > 0
        if not np.any(m):
            fp.write_text("")
            continue
        polys = mask_to_yolo_polygons(m.astype(np.uint8))
        if not polys:
            fp.write_text("")
            continue
        lines: List[str] = []
        for poly in polys:
            coords: List[str] = []
            for x, y in poly:
                coords.append(f"{x:.6f}")
                coords.append(f"{y:.6f}")
            lines.append("0 " + " ".join(coords))
        fp.write_text("\n".join(lines) + "\n")
    return pattern_path.parent


def write_binary_outputs_from_pattern(
    mask_u8: np.ndarray,
    pattern_path: Path,
    video_path: Path,
    fps: float,
) -> Tuple[Path, Path]:
    T, H, W = mask_u8.shape
    pattern_path.parent.mkdir(parents=True, exist_ok=True)

    for t in tqdm(range(T), desc=f"Writing binary TIFF sequence ({pattern_path.parent.name})"):
        img = (mask_u8[t] * 255).astype(np.uint8)
        fp = _format_frame_path(pattern_path, t + 1)
        fp.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(fp), img)

    proc = ffmpeg_rawvideo_writer(
        video_path,
        width=W,
        height=H,
        fps=fps,
        pix_fmt_in="gray",
        codec="ffv1",
        pix_fmt_out="gray",
    )
    try:
        assert proc.stdin is not None
        for t in tqdm(range(T), desc=f"Writing binary MKV ({video_path.name})"):
            img = (mask_u8[t] * 255).astype(np.uint8)
            proc.stdin.write(img.tobytes())
    finally:
        close_ffmpeg_writer(proc)

    return pattern_path.parent, video_path


def write_nrrd(mask_u8: np.ndarray, out_path: Path) -> Path:
    try:
        import nrrd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pynrrd is required for --save_nrrd: pip install pynrrd") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(out_path), np.asarray(mask_u8, dtype=np.uint8))
    return out_path


def write_pipeline_outputs(
    *,
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    out_dir: Path,
    stem: str,
    fps: float,
    save_binary_pattern_value: Optional[str],
    save_labels_pattern_value: Optional[str],
    save_nrrd_flag: bool,
    tag: Optional[str] = None,
) -> Dict[str, Path]:
    tag_suffix = f"_{tag}" if tag else ""

    overlay_path = out_dir / f"{stem}_Overlay{tag_suffix}.mkv"
    write_overlay_video(volume_rgb, mask_u8, overlay_path, fps=fps)
    result_paths: Dict[str, Path] = {"overlay": overlay_path}

    labels_pattern = _resolve_output_pattern(save_labels_pattern_value, DEFAULT_LABEL_PATTERN, out_dir, stem)
    if labels_pattern is not None:
        if tag is not None:
            labels_pattern = _tag_frame_pattern(labels_pattern, tag)
        labels_dir = write_yolo_labels_from_pattern(mask_u8, labels_pattern)
        result_paths["labels_dir"] = labels_dir

    binary_pattern = _resolve_output_pattern(save_binary_pattern_value, DEFAULT_BINARY_PATTERN, out_dir, stem)
    if binary_pattern is not None:
        if tag is not None:
            binary_pattern = _tag_frame_pattern(binary_pattern, tag)
        binary_video_path = out_dir / f"{stem}_Binary{tag_suffix}.mkv"
        tiff_dir, binary_video_path = write_binary_outputs_from_pattern(mask_u8, binary_pattern, binary_video_path, fps=fps)
        result_paths["binary_tiff_dir"] = tiff_dir
        result_paths["binary_video"] = binary_video_path

    if bool(save_nrrd_flag):
        nrrd_path = out_dir / f"{stem}{tag_suffix}.nrrd"
        result_paths["nrrd"] = write_nrrd(mask_u8, nrrd_path)

    return result_paths


def write_summary_file(
    out_path: Path,
    *,
    command: str,
    input_path: Path,
    out_dir: Path,
    volume_shape: Tuple[int, int, int],
    fps: float,
    model_paths: Sequence[str],
    view_names: Sequence[str],
    interpolation_stats: List[Dict[str, object]],
    voxel_volume: Optional[int],
    final_paths: Dict[str, Path],
) -> Path:
    lines: List[str] = []
    lines.append(f"Command: {command}")
    lines.append(f"Input: {input_path}")
    lines.append(f"Output directory: {out_dir}")
    lines.append(f"Volume shape (t, Y, X): {volume_shape}")
    lines.append(f"FPS: {fps}")
    lines.append(f"Models: {', '.join(str(Path(m)) for m in model_paths)}")
    lines.append(f"Views: {', '.join(view_names)}")
    lines.append("")
    lines.append("Interpolation statistics:")
    lines.append(json.dumps(interpolation_stats, indent=2))
    if voxel_volume is not None:
        lines.append("")
        lines.append(f"voxel_volume: {int(voxel_volume)}")
    lines.append("")
    lines.append("Final outputs:")
    for key in sorted(final_paths.keys()):
        lines.append(f"{key}: {final_paths[key]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


# --------------------------
# Main
# --------------------------

def main() -> None:
    args = build_argparser().parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    model_paths = _parse_models(args.model)
    if not model_paths:
        raise ValueError("--model must specify at least one model path")
    for m in model_paths:
        if not Path(m).expanduser().exists():
            raise FileNotFoundError(m)

    angles = _parse_angles(args.angle) or [0.0, 120.0, 240.0]

    if args.min_conf > 0 and args.min_conf < args.conf:
        raise ValueError("--min_conf must be equal to or greater than --conf")
    if int(args.interpolate) < 0:
        raise ValueError("--interpolate must be >= 0")
    if int(args.interpolation_walk_back) < 0:
        raise ValueError("--interpolation_walk_back must be >= 0")
    if int(args.interpolate_passes) < 1:
        raise ValueError("--interpolate_passes must be >= 1")
    if float(args.interpolate_min_radius) < 0:
        raise ValueError("--interpolate_min_radius must be >= 0")
    if float(args.min_radius) < 0:
        raise ValueError("--min_radius must be >= 0")
    if not (-90.0 < float(args.interpolation_search_angle) < 90.0):
        raise ValueError("--interpolation_search_angle must be greater than -90 and less than 90")

    out_dir = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / input_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = out_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    info = ffprobe_info(input_path)
    W = int(info["width"])
    H = int(info["height"])
    T = int(info["num_frames"])
    fps = float(info["fps"])

    vol_path = temp_dir / "input_volume.rgb24.dat"
    volume_rgb = decode_video_to_memmap_rgb24(
        input_video=input_path,
        out_dat=vol_path,
        num_frames=T,
        width=W,
        height=H,
        overwrite=False,
    )
    (temp_dir / "input_volume.meta.json").write_text(
        json.dumps({"shape": [T, H, W, 3], "dtype": "uint8", "fps": fps}, indent=2)
    )

    views = get_view_infos(T=T, H=H, W=W, disable_multiplanar=args.disable_multiplanar)

    shifts: List[Tuple[int, int, str]] = [(0, 0, "none")]
    if int(args.shift) != 0:
        s = abs(int(args.shift))
        shifts = [
            (0, 0, "none"),
            (0, -s, "up"),
            (0, +s, "down"),
            (-s, 0, "left"),
            (+s, 0, "right"),
        ]

    yolo_models: List[Tuple[str, object]] = []
    for m in model_paths:
        name = Path(m).stem
        print(f"Loading model: {name} ({m})")
        yolo_models.append((name, load_ultralytics_model(m)))

    pred_cfg = PredictConfig(
        imgsz=args.imgsz,
        conf=args.conf,
        device=str(args.device),
        half=bool(args.half),
        int8=bool(args.int8),
    )

    union_by_model: Dict[str, Dict[str, np.memmap]] = {}
    conf_by_model: Dict[str, Dict[str, np.memmap]] = {}

    for model_name, _ in yolo_models:
        union_by_model[model_name] = {}
        conf_by_model[model_name] = {}
        for view in views:
            bytes_native = bytes_for_packbits(view.src_h, view.src_w)
            union_path = temp_dir / "union" / model_name / f"{view.name}.union.packbits.dat"
            conf_path = temp_dir / "union" / model_name / f"{view.name}.union_conf.f16.dat"
            union_path.parent.mkdir(parents=True, exist_ok=True)
            union_mm = np.memmap(union_path, dtype=np.uint8, mode="w+", shape=(view.num_slices, bytes_native))
            conf_mm = np.memmap(conf_path, dtype=np.float16, mode="w+", shape=(view.num_slices,))
            union_mm[:] = 0
            conf_mm[:] = 0
            union_mm.flush()
            conf_mm.flush()
            union_by_model[model_name][view.name] = union_mm
            conf_by_model[model_name][view.name] = conf_mm

    for view in views:
        print(f"\n=== View: {view.name} ({view.src_w}x{view.src_h}, slices={view.num_slices}) ===")
        for angle in angles:
            for dx, dy, sname in shifts:
                aug_id = f"a{angle:g}_dx{dx}_dy{dy}_{sname}"
                aug_dir = temp_dir / "aug" / view.name
                aug_video = aug_dir / f"{view.name}_{aug_id}.mkv"
                aug_meta = aug_dir / f"{view.name}_{aug_id}.meta.json"

                aff = build_affine(
                    view=view.name,
                    src_w=view.src_w,
                    src_h=view.src_h,
                    out_size=args.imgsz,
                    angle_deg=float(angle),
                    shift_dx=int(dx),
                    shift_dy=int(dy),
                    pad_mode=view.pad_mode,
                )

                if not aug_video.exists():
                    aug_dir.mkdir(parents=True, exist_ok=True)
                    writer = ffmpeg_rawvideo_writer(
                        aug_video,
                        width=args.imgsz,
                        height=args.imgsz,
                        fps=fps,
                        pix_fmt_in="rgb24",
                        codec="ffv1",
                        pix_fmt_out="yuv444p",
                    )
                    try:
                        assert writer.stdin is not None
                        for frame in tqdm(
                            iter_view_frames(volume_rgb, view),
                            total=view.num_slices,
                            desc=f"Augment {view.name} {aug_id}",
                        ):
                            out = cv2.warpAffine(
                                frame,
                                aff.M_src_to_out,
                                dsize=(args.imgsz, args.imgsz),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0),
                            )
                            writer.stdin.write(out.tobytes())
                    finally:
                        close_ffmpeg_writer(writer)

                    aug_meta.write_text(
                        json.dumps(
                            {
                                "view": view.name,
                                "angle_deg": float(angle),
                                "shift_dx": int(dx),
                                "shift_dy": int(dy),
                                "src_w": view.src_w,
                                "src_h": view.src_h,
                                "out_size": int(args.imgsz),
                                "pad_mode": view.pad_mode,
                                "canvas_w": int(aff.canvas_w),
                                "canvas_h": int(aff.canvas_h),
                                "pad_size": int(aff.pad_size),
                                "pad_off_x": float(aff.pad_off_x),
                                "pad_off_y": float(aff.pad_off_y),
                                "M_out_to_src": aff.M_out_to_src.tolist(),
                                "M_src_to_out": aff.M_src_to_out.tolist(),
                            },
                            indent=2,
                        )
                    )

                for model_name, yolo in yolo_models:
                    pred_prefix = temp_dir / "preds" / model_name / view.name / f"{view.name}_{aug_id}"
                    predict_video_and_accumulate(
                        model=yolo,
                        video_path=aug_video,
                        num_frames=view.num_slices,
                        out_size=args.imgsz,
                        pred_out_prefix=pred_prefix,
                        cfg=pred_cfg,
                        view_union_mm=union_by_model[model_name][view.name],
                        view_conf_mm=conf_by_model[model_name][view.name],
                        M_out_to_native=aff.M_out_to_src,
                        native_h=view.src_h,
                        native_w=view.src_w,
                    )

                if not args.troubleshooting:
                    try:
                        aug_video.unlink(missing_ok=True)
                        aug_meta.unlink(missing_ok=True)
                    except Exception:
                        pass

        for model_name, _ in yolo_models:
            print(f"\n--- Postprocessing view '{view.name}' for model '{model_name}' ---")
            if args.min_conf > 0:
                apply_min_conf_filter_inplace(
                    union_by_model[model_name][view.name],
                    conf_by_model[model_name][view.name],
                    float(args.min_conf),
                )
                union_by_model[model_name][view.name].flush()
                conf_by_model[model_name][view.name].flush()

            fill_2d_holes_inplace(union_by_model[model_name][view.name], view.src_h, view.src_w)
            union_by_model[model_name][view.name].flush()

    view_volumes_by_model: Dict[str, Dict[str, np.memmap]] = {}
    for model_name, _ in yolo_models:
        view_volumes_by_model[model_name] = {}
        for view in views:
            out_path = temp_dir / "view_volumes" / model_name / f"{view.name}.u8.dat"
            view_volumes_by_model[model_name][view.name] = unpack_view_union_to_volume(
                union_mm=union_by_model[model_name][view.name],
                num_slices=view.num_slices,
                h=view.src_h,
                w=view.src_w,
                out_path=out_path,
                desc=f"Unpacking {model_name}/{view.name}",
            )

    def build_current_finalized_volume(snapshot_stem: str) -> np.ndarray:
        ensemble_mm = assemble_current_ensemble_volume(
            view_volumes_by_model=view_volumes_by_model,
            T=T,
            H=H,
            W=W,
            disable_multiplanar=args.disable_multiplanar,
            out_path=temp_dir / f"{snapshot_stem}.u8.dat",
        )
        ensemble_u8 = fill_3d_voids(np.asarray(ensemble_mm))
        if float(args.min_radius) > 0:
            ensemble_u8 = apply_transverse_min_radius_filter(ensemble_u8, float(args.min_radius))
        return ensemble_u8

    interpolation_stats: List[Dict[str, object]] = []

    if bool(args.troubleshooting) and int(args.interpolate) > 0:
        print("\n=== Writing troubleshooting outputs: pass 0 (pre-interpolation) ===")
        pass0_u8 = build_current_finalized_volume("ensemble_pass0")
        write_pipeline_outputs(
            volume_rgb=volume_rgb,
            mask_u8=pass0_u8,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_binary_pattern_value=args.save_binary,
            save_labels_pattern_value=args.save_labels,
            save_nrrd_flag=bool(args.save_nrrd),
            tag="Pass0",
        )

    if int(args.interpolate) > 0:
        total_passes = int(args.interpolate_passes)
        for pass_idx in range(1, total_passes + 1):
            print(
                f"\n=== Interpolation pass {pass_idx}/{total_passes} "
                f"(distance={int(args.interpolate)}, walk_back={int(args.interpolation_walk_back)}, "
                f"min_radius={float(args.interpolate_min_radius):g}, "
                f"search_angle={float(args.interpolation_search_angle):g}) ==="
            )
            for model_name in sorted(view_volumes_by_model.keys()):
                for view in views:
                    print(f"\n--- Interpolating model '{model_name}' view '{view.name}' ---")
                    current_mm = view_volumes_by_model[model_name][view.name]
                    next_u8, stats = interpolate_view_volume_pass(
                        np.asarray(current_mm),
                        max_slice_distance=int(args.interpolate),
                        search_angle_deg=float(args.interpolation_search_angle),
                        interpolation_walk_back=int(args.interpolation_walk_back),
                        interpolate_min_radius=float(args.interpolate_min_radius),
                    )
                    current_mm[:, :, :] = next_u8
                    current_mm.flush()

                    stats = dict(stats)
                    stats.update({
                        "pass_index": int(pass_idx),
                        "model": str(model_name),
                        "view": str(view.name),
                        "max_slice_distance": int(args.interpolate),
                        "interpolation_walk_back": int(args.interpolation_walk_back),
                        "interpolation_search_angle": float(args.interpolation_search_angle),
                    })
                    interpolation_stats.append(stats)

            if bool(args.troubleshooting) and pass_idx < total_passes:
                print(f"\n=== Writing troubleshooting outputs: pass {pass_idx} ===")
                pass_u8 = build_current_finalized_volume(f"ensemble_pass{pass_idx}")
                write_pipeline_outputs(
                    volume_rgb=volume_rgb,
                    mask_u8=pass_u8,
                    out_dir=out_dir,
                    stem=input_path.stem,
                    fps=fps,
                    save_binary_pattern_value=args.save_binary,
                    save_labels_pattern_value=args.save_labels,
                    save_nrrd_flag=bool(args.save_nrrd),
                    tag=f"Pass{pass_idx}",
                )
    else:
        for model_name in sorted(view_volumes_by_model.keys()):
            for view in views:
                interpolation_stats.append({
                    "pass_index": 0,
                    "model": str(model_name),
                    "view": str(view.name),
                    "max_slice_distance": int(args.interpolate),
                    "interpolation_walk_back": int(args.interpolation_walk_back),
                    "interpolation_search_angle": float(args.interpolation_search_angle),
                    "num_objects": 0,
                    "num_endpoints": 0,
                    "candidate_connections": 0,
                    "accepted_connections": 0,
                    "walk_back_bridges": 0,
                    "skipped_by_min_radius": 0,
                    "added_voxels": 0,
                    "skipped": True,
                })

    final_ensemble_mm = assemble_current_ensemble_volume(
        view_volumes_by_model=view_volumes_by_model,
        T=T,
        H=H,
        W=W,
        disable_multiplanar=args.disable_multiplanar,
        out_path=temp_dir / "ensemble_volume_final.u8.dat",
    )

    print("\n=== 3D void fill after multiplanar/model union ===")
    ensemble_u8 = fill_3d_voids(np.asarray(final_ensemble_mm))
    if float(args.min_radius) > 0:
        ensemble_u8 = apply_transverse_min_radius_filter(ensemble_u8, float(args.min_radius))

    final_paths = write_pipeline_outputs(
        volume_rgb=volume_rgb,
        mask_u8=ensemble_u8,
        out_dir=out_dir,
        stem=input_path.stem,
        fps=fps,
        save_binary_pattern_value=args.save_binary,
        save_labels_pattern_value=args.save_labels,
        save_nrrd_flag=bool(args.save_nrrd),
        tag=None,
    )

    voxel_volume = int(np.count_nonzero(ensemble_u8)) if bool(args.voxel_volume) else None
    summary_path = write_summary_file(
        out_dir / f"{input_path.stem}_Summary.txt",
        command=shlex.join([str(x) for x in sys.argv]),
        input_path=input_path,
        out_dir=out_dir,
        volume_shape=(T, H, W),
        fps=fps,
        model_paths=model_paths,
        view_names=[v.name for v in views],
        interpolation_stats=interpolation_stats,
        voxel_volume=voxel_volume,
        final_paths=final_paths,
    )

    if not args.troubleshooting:
        for child in list(temp_dir.iterdir()):
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except Exception:
                pass

    print("\nDone.")
    print(f"Output dir: {out_dir}")
    print(f"Final overlay: {final_paths['overlay']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
