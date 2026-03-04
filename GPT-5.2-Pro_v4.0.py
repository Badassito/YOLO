
#!/usr/bin/env python3
"""
YOLO Segmentation Test-Time Augmentation (TTA) for large videos / volumes.

Implements the requested pipeline:

1) Input: 3072x3072x1930 (X,Y,t) video (frames are transverse XY).
   - Creates Transverse (X,Y), Sagittal (X,t), Coronal (Y,t) views
     unless --disable_multiplanar is active.

2) Augmentation via ONE affine matrix per variant:
   - Rotate by each --angle
   - Scale to --imgsz (square)
   - Optional shifts by --shift pixels: up/down/left/right
   - Transverse: clamp (no pad) for non-90° rotations; OOB -> black; rotated-out content discarded
   - Sagittal/Coronal: pad canvas so rotation doesn't crop; OOB -> black
   - Each variant saved as FFV1 MKV in output/temp (unless deleted when not troubleshooting)

3) Inference:
   - Runs Ultralytics YOLO predict on each pre-generated augmented video (faster than JIT per-frame aug)
   - save=False, task=segment, stream=True, iou=1.0, retina_masks=True, batch=1
   - Results stored to disk (packed-bit masks + per-frame max conf)

4) Undo aug (inverse affine) and per-frame union:
   - Union all instance masks in a frame into one binary mask
   - Keep the highest confidence among combined masks

5) Apply --min_conf (if > 0):
   - For frames with conf < min_conf, remove slice if it has no overlapping segment in either neighbor slice.

6) 2D hole fill per slice (donut hole fill).

7) Multiplanar union (or transverse only if disabled).

8) 3D void fill (background CCs not touching boundary become foreground).

9) Optional skeleton-based interpolation (--interpolate > 0).

10) Outputs:
   - Final transverse overlay MKV (FFV1): original frames + blue (50% alpha) mask.
   - Optional YOLO segmentation labels (blank files for no-detection frames).
   - Optional binary mask TIFF sequence + binary MKV (FFV1).
   - Optional white-pixel count marker file (named "{count}.txt").

Dependencies (Python):
  pip install opencv-python numpy scipy scikit-image tifffile tqdm ultralytics

System:
  ffmpeg + ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
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
    from skimage.morphology import skeletonize  # type: ignore
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
                   help="If >0: low-confidence slice pruning threshold (must be >= --conf). 0 disables.")
    p.add_argument("--half", action="store_true", help="Enable FP16 inference (Ultralytics half=True)")
    p.add_argument("--int8", action="store_true", help="Enable INT8 inference if supported (Ultralytics int8=True)")

    p.add_argument("--save_labels", action="store_true", help="Save final YOLO segmentation labels per frame")
    p.add_argument("--save_binary", action="store_true",
                   help="Save final binary masks as TIFF sequence + FFV1 MKV")
    p.add_argument("--binary", action="store_true",
                   help="Count white pixels in final binary output and create an empty file named with that count")
    p.add_argument("--troubleshooting", action="store_true",
                   help="Keep all temp files and also output a no-interpolation result set")

    p.add_argument("--interpolate", default=3, type=int,
                   help="Skeleton extension length (voxels). 0 disables interpolation.")
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
    assert proc.stdin is not None
    proc.stdin.close()
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
    pad_size: int
    pad_off_x: float
    pad_off_y: float
    M_out_to_src: np.ndarray  # 2x3 float32
    M_src_to_out: np.ndarray  # inverse, 2x3 float32


def _expanded_pad_size(w: int, h: int, angle_deg: float) -> int:
    """Square canvas size P to fit a WxH rectangle rotated by angle_deg."""
    theta = math.radians(angle_deg % 360.0)
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    w_rot = w * c + h * s
    h_rot = w * s + h * c
    P = int(math.ceil(max(w_rot, h_rot)))
    return max(P, max(w, h))


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
    Single affine matrix that does:
      - (virtual) padding for sag/cor (pad_mode='pad')
      - rotation around padded center
      - scale to out_size x out_size
      - shift in destination space
    OpenCV warpAffine uses a matrix mapping destination -> source.
    """
    if pad_mode not in ("clamp", "pad"):
        raise ValueError("pad_mode must be 'clamp' or 'pad'")

    if pad_mode == "clamp":
        P = max(src_w, src_h)
    else:
        P = _expanded_pad_size(src_w, src_h, angle_deg)

    off_x = (P - src_w) / 2.0
    off_y = (P - src_h) / 2.0

    s = P / float(out_size)

    c_d = (out_size - 1) / 2.0
    c_p = (P - 1) / 2.0

    theta = math.radians(angle_deg % 360.0)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # R(-theta) because dst->src
    r11 = cos_t
    r12 = sin_t
    r21 = -sin_t
    r22 = cos_t

    a11 = s * r11
    a12 = s * r12
    a21 = s * r21
    a22 = s * r22

    dx0 = shift_dx + c_d
    dy0 = shift_dy + c_d

    t1 = -(s * (r11 * dx0 + r12 * dy0)) + c_p - off_x
    t2 = -(s * (r21 * dx0 + r22 * dy0)) + c_p - off_y

    M = np.array([[a11, a12, t1],
                  [a21, a22, t2]], dtype=np.float32)

    M3 = np.eye(3, dtype=np.float64)
    M3[:2, :3] = M.astype(np.float64)
    M3_inv = np.linalg.inv(M3)
    Minv = M3_inv[:2, :3].astype(np.float32)

    return AffineSpec(
        view=view,
        angle_deg=float(angle_deg),
        shift_dx=int(shift_dx),
        shift_dy=int(shift_dy),
        src_w=src_w,
        src_h=src_h,
        out_size=out_size,
        pad_size=P,
        pad_off_x=off_x,
        pad_off_y=off_y,
        M_out_to_src=M,
        M_src_to_out=Minv,
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
    flat = mask01.reshape(-1).astype(np.uint8)
    return np.packbits(flat)


def unpack_mask(packed: np.ndarray, h: int, w: int) -> np.ndarray:
    flat = np.unpackbits(packed)[: h * w]
    return flat.reshape((h, w)).astype(np.bool_)


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
    M_native_to_out: np.ndarray,       # 2x3, maps destination(native)->source(out) for warpAffine
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

# Store per-aug result to disk (packed)
        pred_mask_mm[idx, :] = pack_mask(union)
        pred_conf_mm[idx] = np.float16(conf_val)

        # Undo augmentation (warp from out->native by supplying native->out matrix)
        mask_back = cv2.warpAffine(
            union,
            M_native_to_out,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        packed_back = pack_mask((mask_back > 0).astype(np.uint8))

        # OR into the union stack at this slice
        view_union_mm[idx, :] |= packed_back

        # Max confidence across augmentations for this slice
        if conf_val > float(view_conf_mm[idx]):
            view_conf_mm[idx] = np.float16(conf_val)

    pred_mask_mm.flush()
    pred_conf_mm.flush()

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
    """
    For any slice with conf < min_conf:
      - if no overlap with either neighbor slice, delete it (mask->0)
    """
    n = union_mm.shape[0]
    for i in tqdm(range(n), desc="min_conf neighbor pruning"):
        if float(conf_mm[i]) >= min_conf:
            continue
        cur = np.asarray(union_mm[i])
        if not any_mask(cur):
            continue

        keep = False
        if i - 1 >= 0:
            prev = np.asarray(union_mm[i - 1])
            if any_mask(prev) and overlap_any(cur, prev):
                keep = True
        if not keep and i + 1 < n:
            nxt = np.asarray(union_mm[i + 1])
            if any_mask(nxt) and overlap_any(cur, nxt):
                keep = True

        if not keep:
            union_mm[i, :] = 0
            conf_mm[i] = np.float16(0.0)


def fill_2d_holes_inplace(union_mm: np.memmap, h: int, w: int) -> None:
    """Fill all 2D holes per slice (donut-hole fill)."""
    n = union_mm.shape[0]
    for i in tqdm(range(n), desc="2D hole fill"):
        packed = np.asarray(union_mm[i])
        if not any_mask(packed):
            continue
        mask = unpack_mask(packed, h, w)
        filled = ndi.binary_fill_holes(mask)
        union_mm[i, :] = pack_mask(filled.astype(np.uint8))


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
        ensemble_mm[t, :, :] |= m.astype(np.uint8)

    if disable_multiplanar:
        return

    sagittal = view_union_mms["sagittal"]
    bytes_tx = bytes_for_packbits(T, W)
    assert sagittal.shape == (H, bytes_tx)

    for y in tqdm(range(H), desc="Assembling volume from sagittal"):
        m = unpack_mask(np.asarray(sagittal[y]), T, W)
        ensemble_mm[:, y, :] |= m.astype(np.uint8)

    coronal = view_union_mms["coronal"]
    bytes_ty = bytes_for_packbits(T, H)
    assert coronal.shape == (W, bytes_ty)

    for x in tqdm(range(W), desc="Assembling volume from coronal"):
        m = unpack_mask(np.asarray(coronal[x]), T, H)  # (T,H) cols are y
        ensemble_mm[:, :, x] |= m.astype(np.uint8)


def fill_3d_voids(mask_u8: np.ndarray) -> np.ndarray:
    """Fill enclosed 3D voids (background CCs not touching boundary become foreground)."""
    mask_bool = mask_u8.astype(bool)
    filled = ndi.binary_fill_holes(mask_bool)
    return filled.astype(np.uint8)


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


def interpolate_skeleton_extensions(
    mask_u8: np.ndarray,
    extension: int,
    max_iters: Optional[int] = None,
) -> np.ndarray:
    """
    Practical interpolation per spec:

    - Connected components = 3D objects
    - Skeletonize each object (bbox-scoped)
    - Extend skeleton endpoints by N voxels; connect if we hit another object's *real* voxel
    - Iterate so extensions can traverse interpolated regions to reach real regions
    """
    if extension <= 0:
        return mask_u8

    if max_iters is None:
        max_iters = max(1, extension)

    mask = mask_u8.astype(bool)
    real = mask.copy()  # only original voxels are targets

    Z, Y, X = mask.shape
    structure26 = np.ones((3, 3, 3), dtype=bool)

    for _it in range(max_iters):
        labels, num = ndi.label(mask, structure=structure26)
        if num <= 1:
            break

        objs = ndi.find_objects(labels)
        connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []

        for lbl in range(1, num + 1):
            sl = objs[lbl - 1]
            if sl is None:
                continue
            sub = (labels[sl] == lbl)
            if sub.sum() < 8:
                continue

            skel = skeletonize_3d(sub).astype(bool)
            if not skel.any():
                continue

            neigh = ndi.convolve(skel.astype(np.uint8), KERNEL_3, mode="constant", cval=0) - skel.astype(np.uint8)
            endpoints = np.logical_and(skel, neigh == 1)
            ep_coords = np.argwhere(endpoints)
            if ep_coords.size == 0:
                continue

            zsl, ysl, xsl = sl
            z0, y0, x0 = zsl.start, ysl.start, xsl.start

            for ez, ey, ex in ep_coords:
                # find the single skeleton neighbor
                nb = None
                for dz, dy, dx in NEIGH26:
                    zz, yy, xx = ez + dz, ey + dy, ex + dx
                    if 0 <= zz < skel.shape[0] and 0 <= yy < skel.shape[1] and 0 <= xx < skel.shape[2]:
                        if skel[zz, yy, xx]:
                            nb = (zz, yy, xx)
                            break
                if nb is None:
                    continue

                dirz = ez - nb[0]
                diry = ey - nb[1]
                dirx = ex - nb[2]
                if dirz == 0 and diry == 0 and dirx == 0:
                    continue

                p = (z0 + ez, y0 + ey, x0 + ex)

                hit = None
                for step in range(1, extension + 1):
                    qz = p[0] + step * dirz
                    qy = p[1] + step * diry
                    qx = p[2] + step * dirx
                    if not (0 <= qz < Z and 0 <= qy < Y and 0 <= qx < X):
                        break

                    if not real[qz, qy, qx]:
                        # transparent: can pass through interpolated regions / background
                        continue

                    other_lbl = labels[qz, qy, qx]
                    if other_lbl != 0 and other_lbl != lbl:
                        hit = (qz, qy, qx)
                        break

                if hit is not None:
                    connections.append((p, hit))

        if not connections:
            break

        added = 0
        for p, q in connections:
            for (z, y, x) in bresenham3d(p, q):
                if not mask[z, y, x]:
                    mask[z, y, x] = True
                    added += 1

        if added == 0:
            break

    return mask.astype(np.uint8)


# --------------------------
# Final outputs
# --------------------------

def write_binary_outputs(
    mask_u8: np.ndarray,  # (T,H,W) 0/1
    out_dir: Path,
    stem: str,
    fps: float,
) -> Tuple[Path, Path]:
    """Write TIFF sequence + FFV1 MKV for the final binary mask."""
    T, H, W = mask_u8.shape
    tiff_dir = out_dir / "binary_masks"
    tiff_dir.mkdir(parents=True, exist_ok=True)

    for t in tqdm(range(T), desc="Writing binary TIFF sequence"):
        img = (mask_u8[t] * 255).astype(np.uint8)
        tifffile.imwrite(str(tiff_dir / f"{stem}_Binary_{t:04d}.tiff"), img)

    vid_path = out_dir / f"{stem}_Binary.mkv"
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
        for t in tqdm(range(T), desc="Writing binary MKV (FFV1)"):
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
        for t in tqdm(range(T), desc="Writing final overlay video"):
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


def write_yolo_labels(mask_u8: np.ndarray, out_dir: Path, stem: str) -> Path:
    """Write YOLO seg labels: labels/{stem}_%04d.txt (blank files if no detections)."""
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    T, H, W = mask_u8.shape

    for t in tqdm(range(T), desc="Writing YOLO labels"):
        fp = labels_dir / f"{stem}_{t:04d}.txt"
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
        raise ValueError("--min_conf must be 0 (disabled) or >= --conf")

    out_dir = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / input_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = out_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Probe input
    info = ffprobe_info(input_path)
    W = int(info["width"])
    H = int(info["height"])
    T = int(info["num_frames"])
    fps = float(info["fps"])

    # Decode input to memmap (needed for sagittal/coronal slicing)
    vol_path = temp_dir / "input_volume.rgb24.dat"
    volume_rgb = decode_video_to_memmap_rgb24(
        input_video=input_path,
        out_dat=vol_path,
        num_frames=T,
        width=W,
        height=H,
        overwrite=False,
    )
    (temp_dir / "input_volume.meta.json").write_text(json.dumps({"shape": [T, H, W, 3], "dtype": "uint8", "fps": fps}, indent=2))

    # Views
    views = get_view_infos(T=T, H=H, W=W, disable_multiplanar=args.disable_multiplanar)

    # Shift variants
    shifts: List[Tuple[int, int, str]] = [(0, 0, "none")]
    if int(args.shift) != 0:
        s = int(args.shift)
        shifts = [
            (0, 0, "none"),
            (0, -s, "up"),
            (0, +s, "down"),
            (-s, 0, "left"),
            (+s, 0, "right"),
        ]

    # Load models
    yolo_models: List[Tuple[str, object]] = []
    for m in model_paths:
        name = Path(m).stem
        print(f"Loading model: {name} ({m})")
        yolo_models.append((name, load_ultralytics_model(m)))

    pred_cfg = PredictConfig(imgsz=args.imgsz, conf=args.conf, device=str(args.device), half=bool(args.half), int8=bool(args.int8))

    # Allocate per-model, per-view union stacks (packed bits) + confidences
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

    # Augmentation + prediction loops (generate each augmented video ONCE, run all models on it, then delete)
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

                # Generate augmented video if needed
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
                        for frame in tqdm(iter_view_frames(volume_rgb, view), total=view.num_slices, desc=f"Augment {view.name} {aug_id}"):
                            out = cv2.warpAffine(
                                frame,
                                aff.M_out_to_src,
                                dsize=(args.imgsz, args.imgsz),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0),
                            )
                            writer.stdin.write(out.tobytes())
                    finally:
                        close_ffmpeg_writer(writer)

                    aug_meta.write_text(json.dumps({
                        "view": view.name,
                        "angle_deg": float(angle),
                        "shift_dx": int(dx),
                        "shift_dy": int(dy),
                        "src_w": view.src_w,
                        "src_h": view.src_h,
                        "out_size": int(args.imgsz),
                        "pad_mode": view.pad_mode,
                        "pad_size": int(aff.pad_size),
                        "pad_off_x": float(aff.pad_off_x),
                        "pad_off_y": float(aff.pad_off_y),
                        "M_out_to_src": aff.M_out_to_src.tolist(),
                        "M_src_to_out": aff.M_src_to_out.tolist(),
                    }, indent=2))

                # Run inference sequentially for each model on this augmented video
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
                        M_native_to_out=aff.M_src_to_out,  # native (dst) -> out (src)
                        native_h=view.src_h,
                        native_w=view.src_w,
                    )

                # Cleanup augmented video unless troubleshooting
                if not args.troubleshooting:
                    try:
                        aug_video.unlink(missing_ok=True)
                        aug_meta.unlink(missing_ok=True)
                    except Exception:
                        pass

        # Postprocess this view for each model: min_conf + 2D hole fill
        for model_name, _ in yolo_models:
            print(f"\n--- Postprocessing view '{view.name}' for model '{model_name}' ---")
            if args.min_conf > 0:
                apply_min_conf_filter_inplace(union_by_model[model_name][view.name],
                                              conf_by_model[model_name][view.name],
                                              float(args.min_conf))
                union_by_model[model_name][view.name].flush()
                conf_by_model[model_name][view.name].flush()

            fill_2d_holes_inplace(union_by_model[model_name][view.name], view.src_h, view.src_w)
            union_by_model[model_name][view.name].flush()

    # Assemble all models into an ensemble volume on disk (uint8)
    ensemble_path = temp_dir / "ensemble_volume.u8.dat"
    ensemble_mm = np.memmap(ensemble_path, dtype=np.uint8, mode="w+", shape=(T, H, W))
    ensemble_mm[:] = 0
    ensemble_mm.flush()

    for model_name, _ in yolo_models:
        print(f"\n=== Assembling model into ensemble: {model_name} ===")
        assemble_model_volume_into_ensemble(
            ensemble_mm=ensemble_mm,
            view_union_mms=union_by_model[model_name],
            T=T,
            H=H,
            W=W,
            disable_multiplanar=args.disable_multiplanar,
        )
        ensemble_mm.flush()

        if not args.troubleshooting:
            # Remove heavy per-augmentation prediction outputs for this model
            shutil.rmtree(temp_dir / "preds" / model_name, ignore_errors=True)

    # 3D void fill after multiplanar + model ensemble union
    print("\n=== 3D void fill (binary_fill_holes) ===")
    ensemble_u8 = np.asarray(ensemble_mm)  # bring into RAM for 3D ops
    ensemble_u8 = fill_3d_voids(ensemble_u8)

    # Save a no-interp snapshot for troubleshooting outputs
    nointerp_mask = ensemble_u8.copy() if args.troubleshooting else None

    # Interpolation
    if int(args.interpolate) > 0:
        print(f"\n=== Interpolation (skeleton extension N={int(args.interpolate)}) ===")
        ensemble_u8 = interpolate_skeleton_extensions(ensemble_u8, extension=int(args.interpolate), max_iters=None)

    # Persist final mask back to disk
    ensemble_mm[:] = ensemble_u8
    ensemble_mm.flush()

    # Final overlay video (transverse only)
    final_overlay = out_dir / f"{input_path.stem}_Overlay.mkv"
    write_overlay_video(volume_rgb, ensemble_u8, final_overlay, fps=fps)

    if args.save_binary:
        write_binary_outputs(ensemble_u8, out_dir=out_dir, stem=input_path.stem, fps=fps)

    if args.save_labels:
        write_yolo_labels(ensemble_u8, out_dir=out_dir, stem=input_path.stem)

    if args.binary:
        count = int(np.sum(ensemble_u8))
        (out_dir / f"{count}.txt").write_text(str(count) + "\n")

    # Troubleshooting outputs without interpolation
    if args.troubleshooting and nointerp_mask is not None:
        print("\n=== Writing troubleshooting outputs (no interpolation) ===")
        overlay_nointerp = out_dir / f"{input_path.stem}_Overlay_NoInterp.mkv"
        write_overlay_video(volume_rgb, nointerp_mask, overlay_nointerp, fps=fps)

        if args.save_binary:
            tiff_dir = out_dir / "binary_masks_nointerp"
            tiff_dir.mkdir(parents=True, exist_ok=True)
            for t in tqdm(range(T), desc="Writing no-interp binary TIFF sequence"):
                img = (nointerp_mask[t] * 255).astype(np.uint8)
                tifffile.imwrite(str(tiff_dir / f"{input_path.stem}_Binary_NoInterp_{t:04d}.tiff"), img)

            vid_nointerp = out_dir / f"{input_path.stem}_Binary_NoInterp.mkv"
            proc = ffmpeg_rawvideo_writer(
                vid_nointerp,
                width=W,
                height=H,
                fps=fps,
                pix_fmt_in="gray",
                codec="ffv1",
                pix_fmt_out="gray",
            )
            try:
                assert proc.stdin is not None
                for t in tqdm(range(T), desc="Writing no-interp binary MKV (FFV1)"):
                    img = (nointerp_mask[t] * 255).astype(np.uint8)
                    proc.stdin.write(img.tobytes())
            finally:
                close_ffmpeg_writer(proc)

        if args.save_labels:
            labels_nointerp = out_dir / "labels_nointerp"
            labels_nointerp.mkdir(parents=True, exist_ok=True)
            for t in tqdm(range(T), desc="Writing no-interp YOLO labels"):
                fp = labels_nointerp / f"{input_path.stem}_NoInterp_{t:04d}.txt"
                m = (nointerp_mask[t] > 0)
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

    # Cleanup temp if not troubleshooting (keep an empty temp/ folder in the output tree)
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
    print(f"Final overlay: {final_overlay}")


if __name__ == "__main__":
    main()
