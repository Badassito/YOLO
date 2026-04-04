#!/usr/bin/env python3
"""
YOLO segmentation test-time augmentation (TTA) for large video volumes.

This v6.1.0 specification-aligned script:
  - builds transverse, sagittal, coronal and optional radial view families
  - generates rotated / scaled / shifted FFV1 MKV augmentations via a single affine transform per variant
  - runs Ultralytics YOLO segmentation sequentially on the pre-generated augmentation videos
  - stores per-augmentation traces to disk, undoes the affine transforms, unions masks per slice and tracks per-pixel max confidence for --min_conf
  - fills 2D holes after per-frame unions, interpolates active orthogonal views in their native slice direction, and interpolates the radial backprojection in both Cartesian sagittal and coronal directions before the final multiplanar / multi-model union
  - fills enclosed 3D voids with a streamed boundary-background pass and applies --min_radius in the transverse plane before interpolation
  - prefers in-memory workspaces on the SLURM target and falls back to disk-backed scratch only when the working set is too large
  - parallelizes augmentation generation across independent slices and interpolation across independent view/model volumes
  - extracts radial diameter slices with exact Lanczos-5 interpolation by default

Dependencies (Python):
  pip install opencv-python numpy scipy scikit-image tifffile tqdm ultralytics
  pip install pynrrd   # only needed for --save_nrrd

System:
  ffmpeg + ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

GIB = 1024 ** 3
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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
                   help="Only use Transverse view for Cartesian augmentation (skip Sagittal/Coronal)")
    p.add_argument("--azimuth_angle", default=0.0, type=float,
                   help="Angular spacing in degrees for radial diameter slices over [0,180). 0 disables radial views")
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
    p.add_argument("--save_multiplanar", action="store_true",
                   help="Save additional Sagittal and Coronal final outputs even though the default output is transverse-only")
    p.add_argument("--voxel_volume", action="store_true", help="Count white voxels in the final binary output and save to the summary text file")
    p.add_argument("--binary", action="store_true", dest="voxel_volume", help=argparse.SUPPRESS)
    p.add_argument("--troubleshooting", action="store_true",
                   help="Keep temp files and save pass snapshots before each interpolation pass")
    p.add_argument("--scratch_dir", default=None, type=str,
                   help="Optional bulk scratch/temp root for disk-backed working files. Defaults to {output}/temp")
    p.add_argument("--augmentation_workers", default=0, type=int,
                   help="Worker threads for augmentation generation. 0 = auto-tuned for the SLURM target")
    p.add_argument("--interpolation_workers", default=0, type=int,
                   help="Worker threads for running independent per-view interpolation passes. 0 = auto-tuned")

    p.add_argument("--interpolate", default=15, type=int,
                   help="Maximum slice-distance used to search for interpolation candidates. 0 disables interpolation")
    p.add_argument("--interpolation_walk_back", default=3, type=int,
                   help="Additional source slices to bridge before the endpoint slice. 0 disables walk-back bridges")
    p.add_argument("--interpolation_candidates", default=1, type=int,
                   help="Accept up to the Nth nearest interpolation candidate per endpoint projection")
    p.add_argument("--interpolate_passes", default=1, type=int,
                   help="Run the interpolation process this many passes, treating the previous pass as real")
    p.add_argument("--interpolate_min_radius", default=3, type=float,
                   help="Skip interpolation bridges whose effective radius is <= this value")
    p.add_argument("--interpolation_search_angle", default=15.0, type=float,
                   help="Projection growth angle in degrees. Must be greater than -90 and less than 90")
    p.add_argument("--cone_half_angle", dest="interpolation_search_angle", type=float, help=argparse.SUPPRESS)
    return p


# --------------------------
# Scratch / temp layout
# --------------------------

def _path_free_bytes(path: Path) -> int:
    try:
        usage = shutil.disk_usage(str(path))
        return int(usage.free)
    except Exception:
        return 0


def _read_meminfo_bytes() -> Dict[str, int]:
    info: Dict[str, int] = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            raw_val = parts[1].strip().split()[0]
            info[key] = int(raw_val) * 1024
    except Exception:
        pass
    return info


def available_anon_work_bytes() -> int:
    info = _read_meminfo_bytes()
    mem_avail = int(info.get('MemAvailable', 0))
    swap_free = int(info.get('SwapFree', 0))
    return max(0, mem_avail + swap_free)


def total_anon_capacity_bytes() -> int:
    info = _read_meminfo_bytes()
    mem_total = int(info.get('MemTotal', 0))
    swap_total = int(info.get('SwapTotal', 0))
    return max(0, mem_total + swap_total)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def default_augmentation_workers() -> int:
    cpu = _cpu_count()
    if cpu >= 128:
        return 24
    if cpu >= 64:
        return 16
    if cpu >= 32:
        return 12
    if cpu >= 16:
        return 8
    if cpu >= 8:
        return 4
    return 1


def default_interpolation_workers() -> int:
    cpu = _cpu_count()
    if cpu >= 128:
        return 8
    if cpu >= 64:
        return 6
    if cpu >= 32:
        return 4
    if cpu >= 16:
        return 3
    if cpu >= 8:
        return 2
    return 1


def resolve_worker_count(requested: int, env_name: str, auto_value: int, max_tasks: Optional[int] = None) -> int:
    workers = int(requested)
    if workers <= 0:
        workers = _env_int(env_name, int(auto_value))
    workers = max(1, int(workers))
    if max_tasks is not None:
        workers = max(1, min(int(workers), int(max_tasks)))
    return workers


def array_nbytes(shape: Sequence[int], dtype: np.dtype | str | type) -> int:
    dtype_obj = np.dtype(dtype)
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total) * int(dtype_obj.itemsize)


def flush_array(arr: object) -> None:
    if arr is None:
        return

    try:
        if isinstance(arr, np.memmap):
            arr.flush()
            return
    except Exception:
        pass

    base = getattr(arr, 'base', None)
    try:
        if isinstance(base, np.memmap):
            base.flush()
    except Exception:
        pass


def allocate_workspace_array(
    shape: Sequence[int],
    dtype: np.dtype | str | type,
    path: Optional[Path],
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    reuse_existing: bool = False,
) -> np.ndarray:
    dtype_obj = np.dtype(dtype)
    need_bytes = array_nbytes(shape, dtype_obj)
    budget = workspace_budget_summary(need_bytes, reserve_bytes=reserve_bytes)
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(need_bytes, reserve_bytes=reserve_bytes)

    if use_in_memory:
        try:
            print(f"{desc}: in-memory ({budget})")
            return np.zeros(tuple(int(x) for x in shape), dtype=dtype_obj)
        except MemoryError:
            print(f"{desc}: in-memory allocation failed, falling back to disk ({budget})")

    if path is None:
        raise ValueError(f"{desc}: disk fallback requires a filesystem path")

    path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and path.exists():
        print(f"{desc}: disk-backed reuse ({budget}) -> {path}")
        return np.memmap(path, dtype=dtype_obj, mode='r+', shape=tuple(int(x) for x in shape))

    if path.exists():
        path.unlink()
    print(f"{desc}: disk-backed ({budget}) -> {path}")
    return np.memmap(path, dtype=dtype_obj, mode='w+', shape=tuple(int(x) for x in shape))


def parallel_map_in_order(
    func: Callable[[int], object],
    items: Iterable[int],
    *,
    max_workers: int,
    max_pending: Optional[int] = None,
) -> Iterator[object]:
    workers = max(1, int(max_workers))
    if workers <= 1:
        for item in items:
            yield func(int(item))
        return

    pending_limit = max(workers, int(max_pending) if max_pending is not None else workers + 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        queue: List[object] = []
        for item in items:
            queue.append(executor.submit(func, int(item)))
            if len(queue) >= pending_limit:
                fut = queue.pop(0)
                yield fut.result()
        while queue:
            fut = queue.pop(0)
            yield fut.result()


def workspace_anon_cap_bytes() -> int:
    """Return the soft cap for anonymous in-memory workspaces.

    The v6.1.0 target system has large RAM+ZRAM capacity, so the default policy now prefers
    a capacity-relative cap instead of a small fixed hard limit:
      - YOLO_TTA_MAX_ANON_WORKSPACE_GIB: hard GiB cap (default 0 = disabled)
      - YOLO_TTA_MAX_ANON_WORKSPACE_FRACTION: fraction of total RAM+swap (default 0.25)
    The lower non-zero cap wins.
    """
    hard_cap_gib = max(0.0, _env_float('YOLO_TTA_MAX_ANON_WORKSPACE_GIB', 0.0))
    fraction = max(0.0, _env_float('YOLO_TTA_MAX_ANON_WORKSPACE_FRACTION', 0.25))

    hard_cap = int(hard_cap_gib * GIB)
    total_cap = int(total_anon_capacity_bytes() * fraction) if fraction > 0.0 else 0

    caps = [v for v in (hard_cap, total_cap) if int(v) > 0]
    if not caps:
        return 0
    return int(min(caps))


def workspace_budget_summary(required_bytes: int, reserve_bytes: int = 16 * GIB) -> str:
    avail = available_anon_work_bytes()
    cap = workspace_anon_cap_bytes()
    reserve = int(max(0, reserve_bytes))
    parts = [
        f'need={required_bytes / GIB:.1f} GiB',
        f'avail+swap={avail / GIB:.1f} GiB',
        f'reserve={reserve / GIB:.1f} GiB',
    ]
    if cap > 0:
        parts.append(f'anon-cap={cap / GIB:.1f} GiB')
    return ', '.join(parts)


def should_use_in_memory_workspace(required_bytes: int, reserve_bytes: int = 16 * GIB) -> bool:
    if int(required_bytes) <= 0:
        return False

    avail = available_anon_work_bytes()
    reserve = int(max(0, reserve_bytes))
    cap = workspace_anon_cap_bytes()

    if cap > 0 and int(required_bytes) > cap:
        return False
    return avail >= int(required_bytes) + reserve


def choose_scratch_dir(preferred: Optional[str], out_dir: Path, stem: str) -> Path:
    """Pick the bulk disk-backed scratch root.

    v6.0.2 hotfix:
      - no longer auto-selects tmpfs locations like /dev/shm for the whole pipeline
      - defaults to {output}/temp so large persistent memmaps stay off tmpfs unless the user explicitly opts in
    """
    candidates: List[Path] = []
    seen: set[str] = set()

    def _add_candidate(p: Path) -> None:
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        candidates.append(p)

    if preferred:
        _add_candidate(Path(preferred).expanduser())
    env_pref = os.environ.get('YOLO_TTA_SCRATCH_DIR', '').strip()
    if env_pref:
        _add_candidate(Path(env_pref).expanduser())
    _add_candidate(out_dir)

    chosen_root: Optional[Path] = None
    for cand in candidates:
        try:
            cand = cand.resolve()
        except Exception:
            cand = cand
        if cand.exists() and os.access(str(cand), os.W_OK):
            chosen_root = cand
            break

    if chosen_root is None:
        chosen_root = out_dir

    if chosen_root == out_dir:
        scratch_dir = out_dir / 'temp'
    else:
        scratch_dir = chosen_root / f'{stem}_{os.getpid()}_temp'

    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir


def estimate_voidfill_workspace_bytes(shape: Tuple[int, int, int]) -> int:
    z_dim, h, w = shape
    return int(z_dim) * int(h) * int(w) * np.dtype(np.uint32).itemsize


def estimate_interpolation_workspace_bytes(shape: Tuple[int, int, int]) -> int:
    z_dim, h, w = shape
    voxels = int(z_dim) * int(h) * int(w)
    return voxels * (np.dtype(np.uint32).itemsize + np.dtype(np.uint8).itemsize)


def expose_scratch_in_output(out_dir: Path, scratch_dir: Path) -> Path:
    """Expose the active scratch directory from the output tree when possible."""
    temp_entry = out_dir / 'temp'
    try:
        if temp_entry.exists() or temp_entry.is_symlink():
            if temp_entry.is_symlink() or temp_entry.is_file():
                temp_entry.unlink(missing_ok=True)
            elif temp_entry.resolve() != scratch_dir.resolve():
                shutil.rmtree(temp_entry, ignore_errors=True)
    except Exception:
        pass

    try:
        if temp_entry.resolve() == scratch_dir.resolve():
            return temp_entry
    except Exception:
        pass

    if temp_entry.exists() or temp_entry.is_symlink():
        return temp_entry

    try:
        os.symlink(str(scratch_dir), str(temp_entry), target_is_directory=True)
        return temp_entry
    except Exception:
        temp_entry.mkdir(parents=True, exist_ok=True)
        (temp_entry / 'SCRATCH_LOCATION.txt').write_text(str(scratch_dir) + '\n')
        return temp_entry


def close_memmap_array(arr: object) -> None:
    if arr is None:
        return
    flush_array(arr)
    try:
        if isinstance(arr, np.memmap):
            mmap_obj = getattr(arr, '_mmap', None)
            if mmap_obj is not None:
                mmap_obj.close()
            return
    except Exception:
        pass
    try:
        base = getattr(arr, 'base', None)
        if isinstance(base, np.memmap):
            mmap_obj = getattr(base, '_mmap', None)
            if mmap_obj is not None:
                mmap_obj.close()
    except Exception:
        pass


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
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Decode input video to a (T,H,W,3) uint8 workspace in RGB24.

    On the v6.1.0 SLURM target the default policy is to keep the decoded source volume in RAM.
    A disk-backed memmap is used only when the working set would be too large.
    """
    _require_bin("ffmpeg")

    shape = (num_frames, height, width, 3)
    reuse_existing = bool(not overwrite and out_dat.exists() and not prefer_memory)
    arr = allocate_workspace_array(
        shape=shape,
        dtype=np.uint8,
        path=out_dat,
        desc='Decoded RGB24 input volume',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        reuse_existing=bool(reuse_existing),
    )
    if reuse_existing and isinstance(arr, np.memmap):
        return arr

    raw_bytes = memoryview(np.ascontiguousarray(arr) if not arr.flags['C_CONTIGUOUS'] else arr).cast('B')
    if raw_bytes.obj is not arr:
        arr = np.asarray(raw_bytes.obj).reshape(shape)
        raw_bytes = memoryview(arr).cast('B')

    frame_bytes = int(width) * int(height) * 3
    chunk_frames = max(1, min(64, max(1, (256 * 1024 * 1024) // max(1, frame_bytes))))

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
        with tqdm(total=num_frames, desc='Decoding input volume (rgb24)') as pbar:
            for start in range(0, num_frames, chunk_frames):
                nframes = min(chunk_frames, num_frames - start)
                need = int(nframes) * int(frame_bytes)
                offset = int(start) * int(frame_bytes)
                view = raw_bytes[offset:offset + need]
                filled = 0
                while filled < need:
                    nread = proc.stdout.readinto(view[filled:])
                    if nread is None or nread <= 0:
                        raise RuntimeError(f'Unexpected EOF while decoding frames starting at {start}/{num_frames}')
                    filled += int(nread)
                pbar.update(int(nframes))
        flush_array(arr)
    finally:
        if proc.stdout:
            proc.stdout.close()
        _, err = proc.communicate()
        if proc.returncode not in (0, None):
            msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
            raise RuntimeError(f"ffmpeg decode failed: {msg}")
    return arr


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


@dataclass(frozen=True)
class AugJob:
    aug_id: str
    angle_deg: float
    shift_dx: int
    shift_dy: int
    shift_name: str
    video_path: Path
    meta_path: Path
    aff: AffineSpec


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
      -> optional shift in native/canvas pixels -> scale to out_size x out_size

    This matches the v5.0.1 expectation that --shift is expressed in pre-scale view pixels, not
    in already-resized imgsz pixels. Transverse uses pad_mode='clamp', which rotates directly on the
    source-sized canvas so non-90° content that leaves the source frame is discarded.
    Sagittal/Coronal use pad_mode='pad' with a square black canvas large enough to preserve the full
    rotation before scaling.
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

    M_shift_canvas = np.array(
        [
            [1.0, 0.0, float(shift_dx)],
            [0.0, 1.0, float(shift_dy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    sx = out_size / float(canvas_w)
    sy = out_size / float(canvas_h)
    M_scale = np.array(
        [
            [sx, 0.0, cx_out - sx * cx_canvas],
            [0.0, sy, cy_out - sy * cy_canvas],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    M_src_to_out3 = M_scale @ M_shift_canvas @ M_rot @ M_pad
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
    family: str = 'orthogonal'
    azimuths_deg: Tuple[float, ...] = ()
    diameter: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    roi_radius: float = 0.0
    full_h: int = 0
    full_w: int = 0


def build_radial_azimuths(azimuth_angle: float) -> List[float]:
    if float(azimuth_angle) <= 0.0:
        return []
    out: List[float] = []
    a = 0.0
    step = float(azimuth_angle)
    while a < 180.0 - 1e-9:
        out.append(float(a))
        a += step
    if not out:
        out.append(0.0)
    return out


def get_view_infos(
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
    azimuth_angle: float = 0.0,
    include_radial: bool = True,
) -> List[ViewInfo]:
    views = [
        ViewInfo(name='transverse', num_slices=T, src_h=H, src_w=W, pad_mode='clamp', family='orthogonal', full_h=H, full_w=W),
    ]
    if not disable_multiplanar:
        views.append(ViewInfo(name='sagittal', num_slices=H, src_h=T, src_w=W, pad_mode='pad', family='orthogonal', full_h=H, full_w=W))
        views.append(ViewInfo(name='coronal', num_slices=W, src_h=T, src_w=H, pad_mode='pad', family='orthogonal', full_h=H, full_w=W))

    if include_radial and float(azimuth_angle) > 0.0:
        azimuths = tuple(build_radial_azimuths(float(azimuth_angle)))
        diameter = int(min(W, H))
        roi_radius = float(max(0.0, (diameter - 1) / 2.0))
        views.append(
            ViewInfo(
                name='radial',
                num_slices=len(azimuths),
                src_h=T,
                src_w=diameter,
                pad_mode='pad',
                family='radial',
                azimuths_deg=azimuths,
                diameter=diameter,
                center_x=float((W - 1) / 2.0),
                center_y=float((H - 1) / 2.0),
                roi_radius=roi_radius,
                full_h=H,
                full_w=W,
            )
        )
    return views


def orthogonal_views_only(views: Sequence[ViewInfo]) -> List[ViewInfo]:
    return [v for v in views if v.family == 'orthogonal']


@dataclass(frozen=True)
class RadialSampler:
    angle_deg: float
    diameter: int
    x_idx: np.ndarray
    y_idx: np.ndarray
    x_w: np.ndarray
    y_w: np.ndarray
    nn_x: np.ndarray
    nn_y: np.ndarray


_RADIAL_SAMPLER_CACHE: Dict[Tuple[int, int, int, float], RadialSampler] = {}


def _lanczos_kernel(x: np.ndarray, a: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.sinc(x) * np.sinc(x / float(a))
    out[np.abs(x) >= float(a)] = 0.0
    return out.astype(np.float32, copy=False)


def get_radial_sampler(view: ViewInfo, angle_deg: float) -> RadialSampler:
    if view.family != 'radial':
        raise ValueError('Radial sampler requested for a non-radial view')

    key = (int(view.full_w), int(view.full_h), int(view.diameter), round(float(angle_deg), 6))
    cached = _RADIAL_SAMPLER_CACHE.get(key)
    if cached is not None:
        return cached

    diameter = int(view.diameter)
    coords = np.linspace(-float(view.roi_radius), float(view.roi_radius), diameter, dtype=np.float32)
    theta = math.radians(float(angle_deg))
    xs = np.asarray(float(view.center_x) + coords * math.cos(theta), dtype=np.float32)
    ys = np.asarray(float(view.center_y) + coords * math.sin(theta), dtype=np.float32)

    offsets = np.arange(-4, 6, dtype=np.int32)
    x0 = np.floor(xs).astype(np.int32, copy=False)
    y0 = np.floor(ys).astype(np.int32, copy=False)

    x_idx_raw = x0[:, None] + offsets[None, :]
    y_idx_raw = y0[:, None] + offsets[None, :]

    x_w = _lanczos_kernel(xs[:, None] - x_idx_raw, a=5)
    y_w = _lanczos_kernel(ys[:, None] - y_idx_raw, a=5)

    x_valid = (x_idx_raw >= 0) & (x_idx_raw < int(view.full_w))
    y_valid = (y_idx_raw >= 0) & (y_idx_raw < int(view.full_h))
    x_w *= x_valid.astype(np.float32, copy=False)
    y_w *= y_valid.astype(np.float32, copy=False)

    x_idx = np.clip(x_idx_raw, 0, int(view.full_w) - 1).astype(np.int32, copy=False)
    y_idx = np.clip(y_idx_raw, 0, int(view.full_h) - 1).astype(np.int32, copy=False)

    sampler = RadialSampler(
        angle_deg=float(angle_deg),
        diameter=diameter,
        x_idx=x_idx,
        y_idx=y_idx,
        x_w=x_w.astype(np.float32, copy=False),
        y_w=y_w.astype(np.float32, copy=False),
        nn_x=np.clip(np.rint(xs).astype(np.int32, copy=False), 0, int(view.full_w) - 1),
        nn_y=np.clip(np.rint(ys).astype(np.int32, copy=False), 0, int(view.full_h) - 1),
    )
    _RADIAL_SAMPLER_CACHE[key] = sampler
    return sampler


def sample_radial_line_rgb_lanczos5(frame_rgb: np.ndarray, sampler: RadialSampler) -> np.ndarray:
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError('Expected frame_rgb with shape (H,W,3)')

    out = np.zeros((sampler.diameter, 3), dtype=np.float32)
    for yi in range(sampler.y_idx.shape[1]):
        samples = frame_rgb[sampler.y_idx[:, yi][:, None], sampler.x_idx, :].astype(np.float32, copy=False)
        row = np.sum(samples * sampler.x_w[:, :, None], axis=1)
        out += row * sampler.y_w[:, yi][:, None]

    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


def extract_radial_slice_frame(volume_rgb: np.memmap, sampler: RadialSampler) -> np.ndarray:
    t_dim = int(volume_rgb.shape[0])
    out = np.empty((t_dim, sampler.diameter, 3), dtype=np.uint8)
    for t in range(t_dim):
        out[t, :, :] = sample_radial_line_rgb_lanczos5(np.asarray(volume_rgb[t]), sampler)
    return out


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if s in ("", "0", "false", "no", "off"):
        return False
    return True


def radial_fast_path_enabled() -> bool:
    """Fast blocked radial sampler override.

    Default: disabled so the default v6.1.0 behavior remains exact Lanczos-5.
    Set YOLO_TTA_RADIAL_FAST=1 to opt into the faster OpenCV remap path.
    """
    return _env_flag('YOLO_TTA_RADIAL_FAST', False)


def choose_radial_block_size(view: ViewInfo, target_bytes: int = 1 * GIB) -> int:
    env = os.environ.get('YOLO_TTA_RADIAL_BLOCK_ANGLES', '').strip()
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass

    bytes_per_angle = max(1, int(view.src_h) * int(view.src_w) * 3)
    block = max(1, int(target_bytes // bytes_per_angle))
    return max(1, min(32, block))


def build_radial_block_maps(view: ViewInfo, angles_deg: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    if view.family != 'radial':
        raise ValueError('Radial block maps requested for a non-radial view')

    diameter = int(view.diameter)
    coords = np.linspace(-float(view.roi_radius), float(view.roi_radius), diameter, dtype=np.float32)
    map_x = np.empty((len(angles_deg), diameter), dtype=np.float32)
    map_y = np.empty((len(angles_deg), diameter), dtype=np.float32)

    for i, angle_deg in enumerate(angles_deg):
        theta = math.radians(float(angle_deg))
        map_x[i, :] = np.asarray(float(view.center_x) + coords * math.cos(theta), dtype=np.float32)
        map_y[i, :] = np.asarray(float(view.center_y) + coords * math.sin(theta), dtype=np.float32)

    return map_x, map_y


def write_aug_job_meta(job: AugJob, view: ViewInfo) -> None:
    job.meta_path.parent.mkdir(parents=True, exist_ok=True)
    job.meta_path.write_text(
        json.dumps(
            {
                'view': view.name,
                'family': view.family,
                'angle_deg': float(job.angle_deg),
                'shift_dx': int(job.shift_dx),
                'shift_dy': int(job.shift_dy),
                'src_w': view.src_w,
                'src_h': view.src_h,
                'out_size': int(job.aff.out_size),
                'pad_mode': view.pad_mode,
                'canvas_w': int(job.aff.canvas_w),
                'canvas_h': int(job.aff.canvas_h),
                'pad_size': int(job.aff.pad_size),
                'pad_off_x': float(job.aff.pad_off_x),
                'pad_off_y': float(job.aff.pad_off_y),
                'M_out_to_src': job.aff.M_out_to_src.tolist(),
                'M_src_to_out': job.aff.M_src_to_out.tolist(),
            },
            indent=2,
        )
    )


def build_aug_jobs_for_view(
    view: ViewInfo,
    angles: Sequence[float],
    shifts: Sequence[Tuple[int, int, str]],
    out_size: int,
    temp_dir: Path,
) -> List[AugJob]:
    aug_dir = temp_dir / 'aug' / view.name
    jobs: List[AugJob] = []

    for angle in angles:
        for dx, dy, sname in shifts:
            aug_id = f'a{angle:g}_dx{dx}_dy{dy}_{sname}'
            aff = build_affine(
                view=view.name,
                src_w=view.src_w,
                src_h=view.src_h,
                out_size=out_size,
                angle_deg=float(angle),
                shift_dx=int(dx),
                shift_dy=int(dy),
                pad_mode=view.pad_mode,
            )
            jobs.append(
                AugJob(
                    aug_id=aug_id,
                    angle_deg=float(angle),
                    shift_dx=int(dx),
                    shift_dy=int(dy),
                    shift_name=str(sname),
                    video_path=aug_dir / f'{view.name}_{aug_id}.mkv',
                    meta_path=aug_dir / f'{view.name}_{aug_id}.meta.json',
                    aff=aff,
                )
            )
    return jobs


def ensure_radial_augmented_videos(
    volume_rgb: np.memmap,
    view: ViewInfo,
    aug_jobs: Sequence[AugJob],
    fps: float,
) -> None:
    if view.family != 'radial':
        raise ValueError('ensure_radial_augmented_videos expects a radial view')

    missing_jobs = [job for job in aug_jobs if not job.video_path.exists()]
    for job in aug_jobs:
        if not job.meta_path.exists():
            write_aug_job_meta(job, view)

    if not missing_jobs:
        return

    for job in missing_jobs:
        job.video_path.parent.mkdir(parents=True, exist_ok=True)

    writers: Dict[str, subprocess.Popen] = {}
    try:
        for job in missing_jobs:
            writers[job.aug_id] = ffmpeg_rawvideo_writer(
                job.video_path,
                width=int(job.aff.out_size),
                height=int(job.aff.out_size),
                fps=fps,
                pix_fmt_in='rgb24',
                codec='ffv1',
                pix_fmt_out='yuv444p',
            )

        if radial_fast_path_enabled():
            block_size = choose_radial_block_size(view)
            print(
                f"Radial fast path: OpenCV remap blocks of {block_size} azimuths, "
                f"reused across {len(missing_jobs)} augmentation(s)"
            )
            for block_start in range(0, len(view.azimuths_deg), block_size):
                block_end = min(len(view.azimuths_deg), block_start + block_size)
                block_angles = view.azimuths_deg[block_start:block_end]
                map_x, map_y = build_radial_block_maps(view, block_angles)
                native_block = np.empty((len(block_angles), int(view.src_h), int(view.src_w), 3), dtype=np.uint8)

                for t in tqdm(
                    range(int(view.src_h)),
                    total=int(view.src_h),
                    desc=f'Radial slice block {block_start + 1}-{block_end}',
                ):
                    frame_rgb = np.asarray(volume_rgb[t])
                    sampled = cv2.remap(
                        frame_rgb,
                        map_x,
                        map_y,
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    if sampled.ndim == 2:
                        sampled = sampled[:, :, None]
                    native_block[:, t, :, :] = sampled

                for local_idx in tqdm(
                    range(len(block_angles)),
                    total=len(block_angles),
                    desc=f'Radial augment block {block_start + 1}-{block_end}',
                ):
                    native_frame = np.ascontiguousarray(native_block[local_idx])
                    for job in missing_jobs:
                        out = cv2.warpAffine(
                            native_frame,
                            job.aff.M_src_to_out,
                            dsize=(int(job.aff.out_size), int(job.aff.out_size)),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0),
                        )
                        writer = writers[job.aug_id]
                        assert writer.stdin is not None
                        writer.stdin.write(out.tobytes())

                del native_block
        else:
            print(
                f"Radial strict path: exact Lanczos-5, reused across {len(missing_jobs)} augmentation(s) "
                f"(set YOLO_TTA_RADIAL_FAST=1 to enable the faster block remap path)"
            )
            for angle_deg in tqdm(view.azimuths_deg, total=len(view.azimuths_deg), desc='Radial slicing (exact Lanczos-5)'):
                sampler = get_radial_sampler(view, float(angle_deg))
                native_frame = np.ascontiguousarray(extract_radial_slice_frame(volume_rgb, sampler))
                for job in missing_jobs:
                    out = cv2.warpAffine(
                        native_frame,
                        job.aff.M_src_to_out,
                        dsize=(int(job.aff.out_size), int(job.aff.out_size)),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    writer = writers[job.aug_id]
                    assert writer.stdin is not None
                    writer.stdin.write(out.tobytes())
    finally:
        for writer in writers.values():
            close_ffmpeg_writer(writer)


def iter_view_frames(volume_rgb: np.memmap, view: ViewInfo) -> Iterator[np.ndarray]:
    """Yield frames for a view, in slice order (0..num_slices-1)."""
    T, H, W, C = volume_rgb.shape
    assert C == 3

    if view.name == 'transverse':
        for t in range(T):
            yield np.asarray(volume_rgb[t])  # (H,W,3)
    elif view.name == 'sagittal':
        for y in range(H):
            yield np.ascontiguousarray(volume_rgb[:, y, :, :])  # (T,W,3)
    elif view.name == 'coronal':
        for x in range(W):
            yield np.ascontiguousarray(volume_rgb[:, :, x, :])  # (T,H,3)
    elif view.name == 'radial':
        for angle_deg in view.azimuths_deg:
            sampler = get_radial_sampler(view, float(angle_deg))
            yield np.ascontiguousarray(extract_radial_slice_frame(volume_rgb, sampler))  # (T,D,3)
    else:
        raise ValueError(f'Unknown view: {view.name}')


def get_view_frame_by_index(volume_rgb: np.ndarray, view: ViewInfo, index: int) -> np.ndarray:
    T, H, W, C = volume_rgb.shape
    assert C == 3

    if view.name == 'transverse':
        return np.asarray(volume_rgb[int(index)])
    if view.name == 'sagittal':
        return np.ascontiguousarray(volume_rgb[:, int(index), :, :])
    if view.name == 'coronal':
        return np.ascontiguousarray(volume_rgb[:, :, int(index), :])
    if view.name == 'radial':
        angle_deg = float(view.azimuths_deg[int(index)])
        if radial_fast_path_enabled():
            map_x, map_y = build_radial_block_maps(view, [angle_deg])
            map_x = np.ascontiguousarray(map_x.astype(np.float32, copy=False))
            map_y = np.ascontiguousarray(map_y.astype(np.float32, copy=False))
            out = np.empty((int(view.src_h), int(view.src_w), 3), dtype=np.uint8)
            for t in range(T):
                sampled = cv2.remap(
                    np.asarray(volume_rgb[t]),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                if sampled.ndim == 2:
                    sampled = sampled[:, :, None]
                out[t, :, :] = np.asarray(sampled[0])
            return out

        sampler = get_radial_sampler(view, angle_deg)
        return np.ascontiguousarray(extract_radial_slice_frame(volume_rgb, sampler))

    raise ValueError(f'Unknown view: {view.name}')


def _render_augmented_bundle_for_index(volume_rgb: np.ndarray, view: ViewInfo, jobs: Sequence[AugJob], index: int) -> List[np.ndarray]:
    native_frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(index)))
    bundle: List[np.ndarray] = []
    for job in jobs:
        bundle.append(cv2.warpAffine(
            native_frame,
            job.aff.M_src_to_out,
            dsize=(int(job.aff.out_size), int(job.aff.out_size)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        ))
    return bundle


def ensure_augmented_videos(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    aug_jobs: Sequence[AugJob],
    fps: float,
    augmentation_workers: int,
) -> None:
    missing_jobs = [job for job in aug_jobs if not job.video_path.exists()]
    for job in aug_jobs:
        if not job.meta_path.exists():
            write_aug_job_meta(job, view)

    if not missing_jobs:
        return

    worker_count = max(1, min(int(augmentation_workers), int(view.num_slices)))
    mode_suffix = ''
    if view.family == 'radial':
        mode_suffix = ' [OpenCV remap fast path]' if radial_fast_path_enabled() else ' [exact Lanczos-5]'
    print(
        f"Generating {len(missing_jobs)} augmented {view.name} video(s) over {view.num_slices} slice(s) "
        f"with {worker_count} worker thread(s){mode_suffix}"
    )

    writers: Dict[str, subprocess.Popen] = {}
    try:
        for job in missing_jobs:
            job.video_path.parent.mkdir(parents=True, exist_ok=True)
            writers[job.aug_id] = ffmpeg_rawvideo_writer(
                job.video_path,
                width=int(job.aff.out_size),
                height=int(job.aff.out_size),
                fps=fps,
                pix_fmt_in='rgb24',
                codec='ffv1',
                pix_fmt_out='yuv444p',
            )

        render = lambda idx: _render_augmented_bundle_for_index(volume_rgb, view, missing_jobs, idx)
        for bundle in tqdm(
            parallel_map_in_order(render, range(view.num_slices), max_workers=worker_count, max_pending=worker_count + 1),
            total=view.num_slices,
            desc=f'Augment {view.name} -> {len(missing_jobs)} video(s)',
        ):
            for job, out in zip(missing_jobs, bundle):
                writer = writers[job.aug_id]
                assert writer.stdin is not None
                writer.stdin.write(np.ascontiguousarray(out).tobytes())
    finally:
        for writer in writers.values():
            close_ffmpeg_writer(writer)


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
    view_confmap_mm: np.memmap,        # float16 confidence map, shape (num_slices, native_h, native_w)
    M_out_to_native: np.ndarray,       # 2x3, maps augmented(out)->native for cv2.warpAffine (src->dst)
    native_h: int,
    native_w: int,
) -> Dict[str, int]:
    """
    Run YOLO predict(stream=True) on a pre-generated augmented video, store a lightweight
    per-augmentation trace to disk, and accumulate the inverse-transformed native masks.

    v5.0.1 requires unioning overlapping masks with the highest confidence score of the
    combined masks. To preserve that behavior without exploding memory, we keep two native
    accumulators per slice:
      1) a packed-bit union mask
      2) a per-pixel max-confidence map

    Later, --min_conf is applied per connected component on the native-orientation union.
    """
    pred_out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_bytes = bytes_for_packbits(out_size, out_size)

    pred_mask_mm = np.memmap(
        pred_out_prefix.with_suffix('.mask.packbits.dat'),
        dtype=np.uint8,
        mode='w+',
        shape=(num_frames, out_bytes),
    )
    pred_conf_mm = np.memmap(
        pred_out_prefix.with_suffix('.conf.f16.dat'),
        dtype=np.float16,
        mode='w+',
        shape=(num_frames,),
    )

    prediction_count = 0
    frames_with_predictions = 0

    results = model.predict(
        source=str(video_path),
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=1.0,
        save=False,
        stream=True,
        task='segment',
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

        frame_union = np.zeros((out_size, out_size), dtype=np.uint8)
        frame_max_conf = 0.0

        if getattr(r, 'masks', None) is None or r.masks is None or r.masks.data is None:
            pred_mask_mm[idx, :] = pack_mask(frame_union)
            pred_conf_mm[idx] = np.float16(0.0)
            continue

        masks_data = r.masks.data  # (n,h,w)
        try:
            masks_np = masks_data.cpu().numpy()
        except Exception:
            masks_np = np.asarray(masks_data)

        num_inst = int(masks_np.shape[0]) if masks_np.ndim == 3 else 0
        if num_inst <= 0:
            pred_mask_mm[idx, :] = pack_mask(frame_union)
            pred_conf_mm[idx] = np.float16(0.0)
            continue

        prediction_count += int(num_inst)
        frames_with_predictions += 1

        if getattr(r, 'boxes', None) is not None and r.boxes is not None and getattr(r.boxes, 'conf', None) is not None:
            try:
                confs_np = r.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception:
                confs_np = np.asarray(r.boxes.conf, dtype=np.float32)
        else:
            confs_np = np.zeros((num_inst,), dtype=np.float32)

        if confs_np.ndim == 0:
            confs_np = np.full((num_inst,), float(confs_np), dtype=np.float32)
        elif confs_np.shape[0] != num_inst:
            confs_np = np.resize(confs_np, (num_inst,)).astype(np.float32, copy=False)

        conf_slice = view_confmap_mm[idx]
        native_union_frame = np.zeros((native_h, native_w), dtype=np.uint8)

        for inst_idx in range(num_inst):
            inst = np.asarray(masks_np[inst_idx], dtype=np.uint8)
            if inst.shape[0] != out_size or inst.shape[1] != out_size:
                inst = cv2.resize(inst, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
            inst = (inst > 0).astype(np.uint8, copy=False)
            if not np.any(inst):
                continue

            frame_union |= inst
            conf_val = float(confs_np[inst_idx])
            if conf_val > frame_max_conf:
                frame_max_conf = conf_val

            native_mask = cv2.warpAffine(
                inst,
                M_out_to_native,
                dsize=(native_w, native_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            native_mask_bool = native_mask > 0
            if not np.any(native_mask_bool):
                continue

            native_union_frame[native_mask_bool] = 1
            conf_slice[native_mask_bool] = np.maximum(
                conf_slice[native_mask_bool],
                np.float16(conf_val),
            )

        if np.any(native_union_frame):
            view_union_mm[idx, :] |= pack_mask(native_union_frame)

        pred_mask_mm[idx, :] = pack_mask(frame_union)
        pred_conf_mm[idx] = np.float16(frame_max_conf)

    flush_array(pred_mask_mm)
    flush_array(pred_conf_mm)
    flush_array(view_confmap_mm)
    flush_array(view_union_mm)

    meta = {
        'video': str(video_path),
        'num_frames': int(num_frames),
        'out_size': int(out_size),
        'mask_packbits_bytes': int(out_bytes),
        'prediction_count': int(prediction_count),
        'frames_with_predictions': int(frames_with_predictions),
        'cfg': {
            'imgsz': int(cfg.imgsz),
            'conf': float(cfg.conf),
            'device': str(cfg.device),
            'half': bool(cfg.half),
            'int8': bool(cfg.int8),
        },
    }
    pred_out_prefix.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    return {
        'prediction_count': int(prediction_count),
        'frames_with_predictions': int(frames_with_predictions),
    }


# --------------------------
# Per-view postprocessing
# --------------------------

def apply_min_conf_filter_with_confmap_inplace(
    union_mm: np.memmap,
    confmap_mm: np.memmap,
    min_conf: float,
    h: int,
    w: int,
) -> None:
    """Remove native 2D connected components whose max confidence is below ``min_conf``.

    This matches the v5.0.1 rule to union overlapping masks and attach the highest confidence
    score of the combined mask, instead of using a single slice-wide score.
    """
    n = union_mm.shape[0]
    structure2 = np.ones((3, 3), dtype=bool)

    for i in tqdm(range(n), desc='min_conf thresholding'):
        packed = np.asarray(union_mm[i])
        if not any_mask(packed):
            confmap_mm[i, :, :] = np.float16(0.0)
            continue

        union = unpack_mask(packed, h, w).astype(bool, copy=False)
        labels2d, num = ndi.label(union, structure=structure2)
        if int(num) <= 0:
            union_mm[i, :] = 0
            confmap_mm[i, :, :] = np.float16(0.0)
            continue

        conf_slice = np.asarray(confmap_mm[i], dtype=np.float32)
        label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
        maxima = ndi.maximum(conf_slice, labels=labels2d, index=label_ids)
        maxima = np.asarray(maxima, dtype=np.float32)
        keep_ids = label_ids[maxima >= float(min_conf)]

        if keep_ids.size == 0:
            union_mm[i, :] = 0
            confmap_mm[i, :, :] = np.float16(0.0)
            continue

        keep = np.isin(labels2d, keep_ids)
        union_mm[i, :] = pack_mask(keep.astype(np.uint8, copy=False))
        conf_slice[~keep] = 0.0
        confmap_mm[i, :, :] = conf_slice.astype(np.float16, copy=False)


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

    for i in tqdm(range(n), desc='2D hole fill'):
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
    ensemble_mm: np.ndarray,  # uint8 (0/1) shape (T,H,W)
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
      - radial: optional already-backprojected (T,H,W) volume
    """
    transverse = view_union_mms["transverse"]
    bytes_xy = bytes_for_packbits(H, W)
    assert transverse.shape == (T, bytes_xy)

    for t in tqdm(range(T), desc="Assembling volume from transverse"):
        m = unpack_mask(np.asarray(transverse[t]), H, W)
        ensemble_mm[t, :, :] |= m

    if not disable_multiplanar and "sagittal" in view_union_mms:
        sagittal = view_union_mms["sagittal"]
        bytes_tx = bytes_for_packbits(T, W)
        assert sagittal.shape == (H, bytes_tx)

        for y in tqdm(range(H), desc="Assembling volume from sagittal"):
            m = unpack_mask(np.asarray(sagittal[y]), T, W)
            ensemble_mm[:, y, :] |= m

    if not disable_multiplanar and "coronal" in view_union_mms:
        coronal = view_union_mms["coronal"]
        bytes_ty = bytes_for_packbits(T, H)
        assert coronal.shape == (W, bytes_ty)

        for x in tqdm(range(W), desc="Assembling volume from coronal"):
            m = unpack_mask(np.asarray(coronal[x]), T, H)  # (T,H) cols are y
            ensemble_mm[:, :, x] |= m

    if "radial" in view_union_mms:
        radial = np.asarray(view_union_mms["radial"])
        assert radial.shape == (T, H, W)
        ensemble_mm[:, :, :] |= radial


class _UnionFind:
    """Simple dynamic union-find for slice-streamed 3D connected-components."""

    def __init__(self) -> None:
        self.parent: List[int] = [0]
        self.rank: List[int] = [0]
        self.touches_boundary: List[bool] = [False]

    def new_ids(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0,), dtype=np.uint32)
        start = len(self.parent)
        stop = start + int(count)
        if stop >= 2 ** 32:
            raise RuntimeError('3D component id space exceeded uint32 capacity')
        self.parent.extend(range(start, stop))
        self.rank.extend([0] * int(count))
        self.touches_boundary.extend([False] * int(count))
        return np.arange(start, stop, dtype=np.uint32)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return ra

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.touches_boundary[ra] = bool(self.touches_boundary[ra] or self.touches_boundary[rb])
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra

    def mark_boundary(self, x: int) -> None:
        self.touches_boundary[self.find(int(x))] = True

    def root_map(self) -> np.ndarray:
        out = np.zeros((len(self.parent),), dtype=np.uint32)
        for i in range(1, len(self.parent)):
            out[i] = np.uint32(self.find(i))
        return out


def _iter_adjacent_gid_pairs(prev_gid: np.ndarray, curr_gid: np.ndarray) -> Iterator[Tuple[int, int]]:
    """Yield unique touching component-id pairs across adjacent z-slices using 26-connectivity."""
    h, w = prev_gid.shape
    seen: set[int] = set()

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            py0 = max(0, -dy)
            py1 = min(h, h - dy)
            cy0 = max(0, dy)
            cy1 = min(h, h + dy)
            px0 = max(0, -dx)
            px1 = min(w, w - dx)
            cx0 = max(0, dx)
            cx1 = min(w, w + dx)

            a = prev_gid[py0:py1, px0:px1]
            b = curr_gid[cy0:cy1, cx0:cx1]
            overlap = (a > 0) & (b > 0)
            if not np.any(overlap):
                continue

            codes = (a[overlap].astype(np.uint64, copy=False) << np.uint64(32)) | b[overlap].astype(np.uint64, copy=False)
            for code in np.unique(codes):
                code_i = int(code)
                if code_i in seen:
                    continue
                seen.add(code_i)
                yield (code_i >> 32), (code_i & 0xFFFFFFFF)


def _mark_boundary_components_from_local_labels(
    uf: _UnionFind,
    local_to_gid: np.ndarray,
    labels2d: np.ndarray,
    z: int,
    z_max: int,
) -> None:
    if labels2d.size == 0 or local_to_gid.size <= 1:
        return

    if z == 0 or z == z_max:
        for gid in local_to_gid[1:]:
            uf.mark_boundary(int(gid))

    border_vals = np.unique(np.concatenate([
        labels2d[0, :],
        labels2d[-1, :],
        labels2d[:, 0],
        labels2d[:, -1],
    ]))
    border_vals = border_vals[border_vals > 0]
    for local_lbl in border_vals.tolist():
        uf.mark_boundary(int(local_to_gid[int(local_lbl)]))


def fill_3d_voids_inplace_streaming(
    mask_mm: np.memmap,
    work_prefix: Path,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> None:
    """Fill enclosed 3D voids with an in-memory-first streamed implementation.

    v6.0.2 hotfix:
      - prefers anonymous RAM/swap-backed arrays for the 3D background-ID workspace
      - falls back to a disk-backed memmap only when the estimated working set would be too large
      - avoids tmpfs-backed bulk scratch files that could previously SIGBUS when /dev/shm filled
    """
    z_dim, h, w = mask_mm.shape
    if z_dim <= 0:
        return

    estimated_bytes = estimate_voidfill_workspace_bytes((z_dim, h, w))
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(estimated_bytes, reserve_bytes=reserve_bytes)
    budget = workspace_budget_summary(estimated_bytes, reserve_bytes=reserve_bytes)
    if use_in_memory:
        print(f"3D void fill workspace: in-memory ({budget})")
        bg_gid_store: np.ndarray = np.zeros((z_dim, h, w), dtype=np.uint32)
        bg_gid_path: Optional[Path] = None
    else:
        print(f"3D void fill workspace: disk-backed ({budget}) -> {work_prefix.parent}")
        bg_gid_path = work_prefix.with_suffix('.bg_gid.u32.dat')
        bg_gid_path.parent.mkdir(parents=True, exist_ok=True)
        bg_gid_store = np.memmap(bg_gid_path, dtype=np.uint32, mode='w+', shape=(z_dim, h, w))

    uf = _UnionFind()
    prev_gid_slice: Optional[np.ndarray] = None

    for z in tqdm(range(z_dim), desc='3D void fill: slice labeling'):
        bg = (np.asarray(mask_mm[z]) == 0).astype(np.uint8, copy=False)
        num_labels, labels2d = cv2.connectedComponents(bg, connectivity=8, ltype=cv2.CV_32S)
        if int(num_labels) <= 1:
            bg_gid_store[z, :, :] = 0
            prev_gid_slice = None
            continue

        local_to_gid = np.zeros((int(num_labels),), dtype=np.uint32)
        local_to_gid[1:] = uf.new_ids(int(num_labels) - 1)
        gid_slice = local_to_gid[labels2d]
        bg_gid_store[z, :, :] = gid_slice

        _mark_boundary_components_from_local_labels(uf, local_to_gid, labels2d, z=z, z_max=z_dim - 1)

        if prev_gid_slice is not None and np.any(prev_gid_slice) and np.any(gid_slice):
            for a, b in _iter_adjacent_gid_pairs(prev_gid_slice, gid_slice):
                uf.union(int(a), int(b))

        prev_gid_slice = np.asarray(gid_slice)

    root_map = uf.root_map()
    touches_by_gid = np.zeros(root_map.shape, dtype=bool)
    for gid in range(1, root_map.shape[0]):
        touches_by_gid[gid] = bool(uf.touches_boundary[int(root_map[gid])])

    for z in tqdm(range(z_dim), desc='3D void fill: apply enclosed components'):
        gid_slice = np.asarray(bg_gid_store[z])
        enclosed = (gid_slice > 0) & (~touches_by_gid[gid_slice])
        if np.any(enclosed):
            mask_mm[z, enclosed] = np.uint8(1)

    flush_array(mask_mm)
    flush_array(bg_gid_store)
    del bg_gid_store
    if bg_gid_path is not None and not keep_temp:
        try:
            bg_gid_path.unlink(missing_ok=True)
        except Exception:
            pass



def label_foreground_volume_streaming(
    mask_mm: np.ndarray,
    work_prefix: Path,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> Tuple[np.ndarray, int, List[Path]]:
    """Label a 3D foreground volume using slice-streamed 26-connectivity.

    v6.0.2 hotfix:
      - prefers an anonymous in-memory uint32 label volume when enough RAM+swap is available
      - otherwise uses a single disk-backed provisional label volume and compacts labels in place,
        avoiding the previous second full uint32 relabel volume
    """
    z_dim, h, w = mask_mm.shape
    estimated_bytes = estimate_voidfill_workspace_bytes((z_dim, h, w))
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(estimated_bytes, reserve_bytes=reserve_bytes)

    work_prefix.parent.mkdir(parents=True, exist_ok=True)
    provisional_path = work_prefix.with_suffix('.fg_labels.u32.dat')
    label_paths: List[Path] = []

    budget = workspace_budget_summary(estimated_bytes, reserve_bytes=reserve_bytes)
    if use_in_memory:
        print(f"Interpolation label workspace: in-memory ({budget})")
        labels_store: np.ndarray = np.zeros((z_dim, h, w), dtype=np.uint32)
    else:
        print(f"Interpolation label workspace: disk-backed ({budget}) -> {work_prefix.parent}")
        labels_store = np.memmap(provisional_path, dtype=np.uint32, mode='w+', shape=(z_dim, h, w))
        label_paths = [provisional_path]

    uf = _UnionFind()
    prev_gid_slice: Optional[np.ndarray] = None

    for z in tqdm(range(z_dim), desc='Interpolation: slice labeling'):
        fg = (np.asarray(mask_mm[z]) > 0).astype(np.uint8, copy=False)
        num_labels, labels2d = cv2.connectedComponents(fg, connectivity=8, ltype=cv2.CV_32S)
        if int(num_labels) <= 1:
            labels_store[z, :, :] = 0
            prev_gid_slice = None
            continue

        local_to_gid = np.zeros((int(num_labels),), dtype=np.uint32)
        local_to_gid[1:] = uf.new_ids(int(num_labels) - 1)
        gid_slice = local_to_gid[labels2d]
        labels_store[z, :, :] = gid_slice

        if prev_gid_slice is not None and np.any(prev_gid_slice) and np.any(gid_slice):
            for a, b in _iter_adjacent_gid_pairs(prev_gid_slice, gid_slice):
                uf.union(int(a), int(b))

        prev_gid_slice = np.asarray(gid_slice)

    root_map = uf.root_map()
    if root_map.shape[0] <= 1:
        flush_array(labels_store)
        return labels_store, 0, label_paths

    unique_roots = np.unique(root_map[1:])
    unique_roots = unique_roots[unique_roots > 0]
    compact_root_ids = np.zeros(root_map.shape, dtype=np.uint32)
    compact_root_ids[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.uint32)

    for z in tqdm(range(z_dim), desc='Interpolation: compact relabel'):
        gid_slice = np.asarray(labels_store[z])
        labels_store[z, :, :] = compact_root_ids[root_map[gid_slice]]

    flush_array(labels_store)
    return labels_store, int(unique_roots.size), label_paths



def build_slice_endpoint_seeds_from_label_volume(labels_real: np.ndarray) -> Tuple[List[SliceEndpointSeed], int]:
    """Legacy faster endpoint scan retained for reference.

    v6.1.0 now uses per-object 3D skeletonization via ``_build_slice_endpoint_seeds`` for actual
    interpolation passes, but this slice-graph terminal scan is kept as an alternative helper.
    """
    z_dim = labels_real.shape[0]
    if z_dim <= 0:
        return [], 0

    seeds: List[SliceEndpointSeed] = []
    kernel2 = np.ones((3, 3), dtype=np.uint8)

    prev_slice: Optional[np.ndarray] = None
    curr_slice = np.asarray(labels_real[0])
    next_slice = np.asarray(labels_real[1]) if z_dim > 1 else None

    for z in tqdm(range(z_dim), desc='Interpolation: endpoint scan'):
        if z > 0:
            prev_slice = curr_slice
            curr_slice = next_slice if next_slice is not None else np.zeros_like(curr_slice)
            next_slice = np.asarray(labels_real[z + 1]) if (z + 1) < z_dim else None

        present = np.unique(curr_slice)
        present = present[present > 0]
        if present.size == 0:
            continue

        for obj_id in present.tolist():
            obj_mask = (curr_slice == int(obj_id)).astype(np.uint8, copy=False)
            num_cc, cc = cv2.connectedComponents(obj_mask, connectivity=8, ltype=cv2.CV_32S)
            if int(num_cc) <= 1:
                continue

            prev_same = (prev_slice == int(obj_id)) if prev_slice is not None else None
            next_same = (next_slice == int(obj_id)) if next_slice is not None else None

            for local_lbl in range(1, int(num_cc)):
                comp = cc == int(local_lbl)
                if not np.any(comp):
                    continue
                anchor = _component_centroid_anchor(comp)
                if anchor is None:
                    continue

                dil = cv2.dilate(comp.astype(np.uint8, copy=False), kernel2, iterations=1) > 0
                has_prev = bool(prev_same is not None and np.any(dil & prev_same))
                has_next = bool(next_same is not None and np.any(dil & next_same))

                if not has_prev:
                    seeds.append(SliceEndpointSeed(
                        label=int(obj_id),
                        point=(int(z), int(anchor[0]), int(anchor[1])),
                        direction_sign=-1,
                    ))
                if not has_next:
                    seeds.append(SliceEndpointSeed(
                        label=int(obj_id),
                        point=(int(z), int(anchor[0]), int(anchor[1])),
                        direction_sign=1,
                    ))

    return seeds, int(len(seeds))


def apply_transverse_min_radius_filter_inplace(mask_mm: np.ndarray, min_radius: float) -> None:
    """In-place transverse-plane radius filter to avoid a full extra volume copy."""
    if float(min_radius) <= 0:
        return

    struct2 = np.ones((3, 3), dtype=bool)
    for t in tqdm(range(mask_mm.shape[0]), desc='Transverse min-radius filter'):
        sl = np.asarray(mask_mm[t]) > 0
        if not np.any(sl):
            continue

        labels2d, num = ndi.label(sl, structure=struct2)
        if int(num) <= 0:
            mask_mm[t, :, :] = 0
            continue

        keep = np.zeros(sl.shape, dtype=bool)
        for lbl in range(1, int(num) + 1):
            comp = labels2d == lbl
            if not np.any(comp):
                continue
            radius = float(np.max(ndi.distance_transform_edt(comp)))
            if radius >= float(min_radius):
                keep |= comp

        mask_mm[t, :, :] = keep.astype(np.uint8, copy=False)
    flush_array(mask_mm)


def apply_view_min_radius_filter_inplace(mask_mm: np.ndarray, view: ViewInfo, min_radius: float) -> None:
    if float(min_radius) <= 0:
        return

    if view.name in ('transverse', 'radial'):
        transverse_view = mask_mm
    elif view.name == 'sagittal':
        transverse_view = np.transpose(mask_mm, (1, 0, 2))
    elif view.name == 'coronal':
        transverse_view = np.transpose(mask_mm, (1, 2, 0))
    else:  # pragma: no cover
        raise ValueError(f'Unsupported view for min-radius filtering: {view.name}')

    print(f"Applying --min_radius in the transverse plane for view '{view.name}'")
    apply_transverse_min_radius_filter_inplace(transverse_view, float(min_radius))
    flush_array(mask_mm)


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
    union_mm: np.ndarray,
    num_slices: int,
    h: int,
    w: int,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    vol_mm = allocate_workspace_array(
        shape=(num_slices, h, w),
        dtype=np.uint8,
        path=out_path,
        desc=f'{desc} workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    for i in tqdm(range(num_slices), desc=desc):
        packed = np.asarray(union_mm[i])
        if any_mask(packed):
            vol_mm[i, :, :] = unpack_mask(packed, h, w)
        else:
            vol_mm[i, :, :] = 0
    flush_array(vol_mm)
    return vol_mm


def backproject_radial_union_to_volume(
    union_mm: np.ndarray,
    radial_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    if radial_view.family != 'radial':
        raise ValueError('backproject_radial_union_to_volume expects a radial view')

    t_dim = int(radial_view.src_h)
    out_h = int(radial_view.full_h)
    out_w = int(radial_view.full_w)
    diameter = int(radial_view.src_w)

    vol_mm = allocate_workspace_array(
        shape=(t_dim, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=f'{desc} workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    for angle_idx, angle_deg in enumerate(tqdm(radial_view.azimuths_deg, desc=desc)):
        packed = np.asarray(union_mm[angle_idx])
        if not any_mask(packed):
            continue
        radial_mask = unpack_mask(packed, t_dim, diameter).astype(bool, copy=False)
        tt, uu = np.nonzero(radial_mask)
        if tt.size == 0:
            continue
        sampler = get_radial_sampler(radial_view, float(angle_deg))
        vol_mm[tt, sampler.nn_y[uu], sampler.nn_x[uu]] = 1

    flush_array(vol_mm)
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
    ensemble_mm: np.ndarray,
    view_volume_mms: Dict[str, np.ndarray],
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
) -> None:
    transverse = np.asarray(view_volume_mms["transverse"])
    assert transverse.shape == (T, H, W)
    for t in tqdm(range(T), desc="Assembling volume from transverse view volume"):
        ensemble_mm[t, :, :] |= transverse[t, :, :]

    if not disable_multiplanar and "sagittal" in view_volume_mms:
        sagittal = np.asarray(view_volume_mms["sagittal"])
        assert sagittal.shape == (H, T, W)
        for y in tqdm(range(H), desc="Assembling volume from sagittal view volume"):
            ensemble_mm[:, y, :] |= sagittal[y, :, :]

    if not disable_multiplanar and "coronal" in view_volume_mms:
        coronal = np.asarray(view_volume_mms["coronal"])
        assert coronal.shape == (W, T, H)
        for x in tqdm(range(W), desc="Assembling volume from coronal view volume"):
            ensemble_mm[:, :, x] |= coronal[x, :, :]

    if "radial" in view_volume_mms:
        radial = np.asarray(view_volume_mms["radial"])
        assert radial.shape == (T, H, W)
        for t in tqdm(range(T), desc="Assembling volume from radial view volume"):
            ensemble_mm[t, :, :] |= radial[t, :, :]


def assemble_current_ensemble_volume(
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
    out_path: Path,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    ensemble_mm = allocate_workspace_array(
        shape=(T, H, W),
        dtype=np.uint8,
        path=out_path,
        desc='Ensemble volume',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

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
        flush_array(ensemble_mm)

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


def _find_slice_projection_candidates(
    labels_real: np.ndarray,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    max_candidates: int,
) -> List[SliceProjectionCandidate]:
    if int(max_slice_distance) <= 0 or int(max_candidates) <= 0:
        return []

    s0, y0, x0 = seed.point
    source_component, source_anchor = _component_mask_and_anchor(labels_real[s0] == int(seed.label), (y0, x0))
    if source_anchor is None or not np.any(source_component):
        return []

    sdf = _signed_distance_2d(source_component)
    slope = math.tan(math.radians(float(search_angle_deg)))
    num_slices = labels_real.shape[0]
    found: Dict[int, SliceProjectionCandidate] = {}

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
        for target_label in np.unique(lbls):
            target_label_i = int(target_label)
            if target_label_i <= 0 or target_label_i == int(seed.label) or target_label_i in found:
                continue
            use = lbls == target_label_i
            ys_t = ys[use]
            xs_t = xs[use]
            if ys_t.size == 0:
                continue
            d2 = (ys_t.astype(np.int64) - int(source_anchor[0])) ** 2 + (xs_t.astype(np.int64) - int(source_anchor[1])) ** 2
            idx = int(np.argmin(d2))
            found[target_label_i] = SliceProjectionCandidate(
                source_label=int(seed.label),
                target_label=target_label_i,
                source_point=(int(s0), int(y0), int(x0)),
                target_point=(int(s), int(ys_t[idx]), int(xs_t[idx])),
                slice_distance=int(step),
            )

        if len(found) >= int(max_candidates):
            break

    ordered = sorted(
        found.values(),
        key=lambda c: (
            int(c.slice_distance),
            (int(c.target_point[1]) - int(source_anchor[0])) ** 2 + (int(c.target_point[2]) - int(source_anchor[1])) ** 2,
            int(c.target_label),
        ),
    )
    return ordered[: int(max_candidates)]

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


def _estimate_linear_slice_bridge_min_radius(
    labels_real: np.ndarray,
    source_label: int,
    target_label: int,
    source_point: Tuple[int, int, int],
    target_point: Tuple[int, int, int],
) -> float:
    """Return the smallest effective radius encountered along a linear slice bridge."""
    s0, y0, x0 = source_point
    s1, y1, x1 = target_point
    if int(s0) == int(s1):
        return 0.0

    source_component, source_anchor = _component_mask_and_anchor(labels_real[s0] == int(source_label), (y0, x0))
    target_component, target_anchor = _component_mask_and_anchor(labels_real[s1] == int(target_label), (y1, x1))
    if source_anchor is None or target_anchor is None:
        return 0.0
    if not np.any(source_component) or not np.any(target_component):
        return 0.0

    half_width = _local_half_width_for_components(source_component, source_anchor, target_component, target_anchor)
    source_local = _component_to_local_canvas(source_component, source_anchor, half_width)
    target_local = _component_to_local_canvas(target_component, target_anchor, half_width)
    if not np.any(source_local) or not np.any(target_local):
        return 0.0

    min_radius = min(_component_max_radius(source_local), _component_max_radius(target_local))
    sdf0 = _signed_distance_2d(source_local)
    sdf1 = _signed_distance_2d(target_local)

    steps = int(abs(int(s1) - int(s0)))
    if steps <= 0:
        return 0.0

    for idx in range(1, steps):
        alpha = float(idx) / float(steps)
        section = ((1.0 - alpha) * sdf0 + alpha * sdf1) >= 0.0
        if not np.any(section):
            return 0.0
        section = _keep_center_component_2d(section)
        min_radius = min(min_radius, _component_max_radius(section))

    return float(min_radius)


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


def interpolate_view_volume_pass_inplace(
    mask_mm: np.ndarray,
    work_dir: Path,
    pass_tag: str,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_min_radius: float,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> Dict[str, object]:
    """Apply one interpolation pass directly to a view-volume stack.

    The pass keeps bridge creation simultaneous by searching against a frozen label snapshot and
    merging all newly created bridge voxels only after planning is complete. Endpoint discovery
    uses per-object 3D skeletonization with a one-slice fallback for orphaned objects so the
    implementation matches the v6.1.0 slice-direction interpolation rules more closely.
    """
    if int(max_slice_distance) <= 0:
        return {
            'num_objects': 0,
            'num_endpoints': 0,
            'candidate_connections': 0,
            'accepted_connections': 0,
            'default_bridges': 0,
            'walk_back_bridges': 0,
            'skipped_by_min_radius': 0,
            'added_voxels': 0,
            'skipped': True,
        }

    work_dir.mkdir(parents=True, exist_ok=True)
    estimated_bytes = estimate_interpolation_workspace_bytes(tuple(int(x) for x in mask_mm.shape))
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(estimated_bytes, reserve_bytes=reserve_bytes)
    if use_in_memory:
        print(f"Interpolation workspace ({pass_tag}): in-memory ({estimated_bytes / GIB:.1f} GiB estimated)")
    else:
        print(f"Interpolation workspace ({pass_tag}): disk-backed ({estimated_bytes / GIB:.1f} GiB estimated) -> {work_dir}")

    labels_mm, num_objects, label_paths = label_foreground_volume_streaming(
        mask_mm,
        work_dir / f'{pass_tag}_labels',
        keep_temp=True,
        prefer_memory=use_in_memory,
        reserve_bytes=reserve_bytes,
    )

    if int(num_objects) <= 1:
        del labels_mm
        if not keep_temp:
            for p in label_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        return {
            'num_objects': int(num_objects),
            'num_endpoints': 0,
            'candidate_connections': 0,
            'accepted_connections': 0,
            'default_bridges': 0,
            'walk_back_bridges': 0,
            'skipped_by_min_radius': 0,
            'added_voxels': 0,
            'skipped': int(num_objects) <= 1,
        }

    seeds, num_endpoints = _build_slice_endpoint_seeds(labels_mm, extension_slices=int(max_slice_distance))
    if not seeds:
        del labels_mm
        if not keep_temp:
            for p in label_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        return {
            'num_objects': int(num_objects),
            'num_endpoints': int(num_endpoints),
            'candidate_connections': 0,
            'accepted_connections': 0,
            'default_bridges': 0,
            'walk_back_bridges': 0,
            'skipped_by_min_radius': 0,
            'added_voxels': 0,
            'skipped': False,
        }

    bridge_path: Optional[Path] = None
    if use_in_memory:
        bridge_mm: np.ndarray = np.zeros(mask_mm.shape, dtype=np.uint8)
    else:
        bridge_path = work_dir / f'{pass_tag}_bridges.u8.dat'
        bridge_mm = np.memmap(bridge_path, dtype=np.uint8, mode='w+', shape=mask_mm.shape)

    candidate_connections = 0
    accepted_connections = 0
    default_bridges = 0
    walk_back_bridges = 0
    skipped_by_min_radius = 0
    added_voxels = 0

    for seed in seeds:
        candidates = _find_slice_projection_candidates(
            labels_real=labels_mm,
            seed=seed,
            max_slice_distance=int(max_slice_distance),
            search_angle_deg=float(search_angle_deg),
            max_candidates=int(interpolation_candidates),
        )
        if not candidates:
            continue

        for candidate in candidates:
            candidate_connections += 1
            target_component, target_anchor = _component_mask_and_anchor(
                labels_mm[candidate.target_point[0]] == int(candidate.target_label),
                (candidate.target_point[1], candidate.target_point[2]),
            )
            target_radius = _component_max_radius(target_component) if target_anchor is not None else 0.0

            source_points = [candidate.source_point] + _collect_walkback_source_points(
                labels_real=labels_mm,
                label=int(candidate.source_label),
                start_point=candidate.source_point,
                direction_sign=int(seed.direction_sign),
                walk_back=int(interpolation_walk_back),
            )

            accepted_this_candidate = False
            for walk_idx, src_point in enumerate(source_points):
                source_component, source_anchor = _component_mask_and_anchor(
                    labels_mm[src_point[0]] == int(candidate.source_label),
                    (src_point[1], src_point[2]),
                )
                source_radius = _component_max_radius(source_component) if source_anchor is not None else 0.0
                bridge_radius = min(float(source_radius), float(target_radius))
                if float(interpolate_min_radius) > 0.0:
                    bridge_radius = _estimate_linear_slice_bridge_min_radius(
                        labels_real=labels_mm,
                        source_label=int(candidate.source_label),
                        target_label=int(candidate.target_label),
                        source_point=src_point,
                        target_point=candidate.target_point,
                    )
                    if bridge_radius <= float(interpolate_min_radius):
                        skipped_by_min_radius += 1
                        continue

                if walk_idx == 0:
                    default_bridges += 1
                else:
                    walk_back_bridges += 1

                if not accepted_this_candidate:
                    accepted_connections += 1
                    accepted_this_candidate = True

                added_voxels += _paint_linear_slice_bridge(
                    bridge_volume=bridge_mm,
                    labels_real=labels_mm,
                    source_label=int(candidate.source_label),
                    target_label=int(candidate.target_label),
                    source_point=src_point,
                    target_point=candidate.target_point,
                )

    for z in tqdm(range(mask_mm.shape[0]), desc='Interpolation: merge bridges'):
        bridge_slice = np.asarray(bridge_mm[z])
        if np.any(bridge_slice):
            mask_mm[z, :, :] |= bridge_slice
    flush_array(mask_mm)

    if isinstance(bridge_mm, np.memmap):
        flush_array(bridge_mm)
    del bridge_mm
    if isinstance(labels_mm, np.memmap):
        flush_array(labels_mm)
    del labels_mm
    if not keep_temp:
        for p in label_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        if bridge_path is not None:
            try:
                bridge_path.unlink(missing_ok=True)
            except Exception:
                pass

    return {
        'num_objects': int(num_objects),
        'num_endpoints': int(num_endpoints),
        'candidate_connections': int(candidate_connections),
        'accepted_connections': int(accepted_connections),
        'default_bridges': int(default_bridges),
        'walk_back_bridges': int(walk_back_bridges),
        'skipped_by_min_radius': int(skipped_by_min_radius),
        'added_voxels': int(added_voxels),
        'skipped': False,
    }


def _copy_volume_to_oriented_memmap(
    src_volume_mm: np.ndarray,
    orientation: str,
    out_path: Path,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    """Copy a Cartesian (T,H,W) volume into an oriented stack for slice-direction interpolation."""
    t_dim, h_dim, w_dim = src_volume_mm.shape

    if orientation == 'sagittal':
        oriented_mm = allocate_workspace_array(
            shape=(h_dim, t_dim, w_dim),
            dtype=np.uint8,
            path=out_path,
            desc=f'Cartesian->{orientation} interpolation workspace',
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
        for y in tqdm(range(h_dim), desc=f'Copy radial volume -> {orientation}'):
            oriented_mm[y, :, :] = np.asarray(src_volume_mm[:, y, :], dtype=np.uint8)
    elif orientation == 'coronal':
        oriented_mm = allocate_workspace_array(
            shape=(w_dim, t_dim, h_dim),
            dtype=np.uint8,
            path=out_path,
            desc=f'Cartesian->{orientation} interpolation workspace',
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
        for x in tqdm(range(w_dim), desc=f'Copy radial volume -> {orientation}'):
            oriented_mm[x, :, :] = np.asarray(src_volume_mm[:, :, x], dtype=np.uint8)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported orientation for radial interpolation: {orientation}')

    flush_array(oriented_mm)
    return oriented_mm


def _merge_added_oriented_volume_into_bridge_union(
    src_volume_mm: np.ndarray,
    oriented_mm: np.ndarray,
    orientation: str,
    bridge_union_mm: np.ndarray,
) -> None:
    """Collect only newly added voxels from an oriented interpolation run back into Cartesian space."""
    t_dim, h_dim, w_dim = src_volume_mm.shape

    if orientation == 'sagittal':
        for y in tqdm(range(h_dim), desc=f'Collect radial {orientation} bridges'):
            src_slice = np.asarray(src_volume_mm[:, y, :], dtype=np.uint8)
            out_slice = np.asarray(oriented_mm[y], dtype=np.uint8)
            added = (out_slice > 0) & (src_slice == 0)
            if np.any(added):
                bridge_union_mm[:, y, :] |= added.astype(np.uint8, copy=False)
    elif orientation == 'coronal':
        for x in tqdm(range(w_dim), desc=f'Collect radial {orientation} bridges'):
            src_slice = np.asarray(src_volume_mm[:, :, x], dtype=np.uint8)
            out_slice = np.asarray(oriented_mm[x], dtype=np.uint8)
            added = (out_slice > 0) & (src_slice == 0)
            if np.any(added):
                bridge_union_mm[:, :, x] |= added.astype(np.uint8, copy=False)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported orientation for radial interpolation: {orientation}')


def interpolate_radial_view_pass_inplace(
    mask_mm: np.ndarray,
    work_dir: Path,
    pass_tag: str,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_min_radius: float,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> Dict[str, object]:
    """Interpolate a radial backprojection in Cartesian sagittal and coronal directions simultaneously."""
    if int(max_slice_distance) <= 0:
        return {
            'num_objects': 0,
            'num_endpoints': 0,
            'candidate_connections': 0,
            'accepted_connections': 0,
            'default_bridges': 0,
            'walk_back_bridges': 0,
            'skipped_by_min_radius': 0,
            'added_voxels': 0,
            'direction_modes': 'sagittal+coronal',
            'skipped': True,
        }

    work_dir.mkdir(parents=True, exist_ok=True)

    bridge_union_bytes = int(np.prod(mask_mm.shape, dtype=np.int64)) * np.dtype(np.uint8).itemsize
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(bridge_union_bytes, reserve_bytes=reserve_bytes)
    budget = workspace_budget_summary(bridge_union_bytes, reserve_bytes=reserve_bytes)
    if use_in_memory:
        print(f"Radial Cartesian interpolation bridge union ({pass_tag}): in-memory ({budget})")
        bridge_union_mm: np.ndarray = np.zeros(mask_mm.shape, dtype=np.uint8)
        bridge_union_path: Optional[Path] = None
    else:
        bridge_union_path = work_dir / f'{pass_tag}_radial_bridge_union.u8.dat'
        print(f"Radial Cartesian interpolation bridge union ({pass_tag}): disk-backed ({budget}) -> {bridge_union_path.parent}")
        bridge_union_mm = np.memmap(bridge_union_path, dtype=np.uint8, mode='w+', shape=mask_mm.shape)

    directional_stats: List[Tuple[str, Dict[str, object]]] = []
    try:
        for orientation in ('sagittal', 'coronal'):
            oriented_path = work_dir / f'{pass_tag}_{orientation}.u8.dat'
            oriented_mm = _copy_volume_to_oriented_memmap(
                mask_mm,
                orientation,
                oriented_path,
                prefer_memory=bool(prefer_memory),
                reserve_bytes=int(reserve_bytes),
            )
            try:
                stats = interpolate_view_volume_pass_inplace(
                    mask_mm=oriented_mm,
                    work_dir=work_dir / orientation,
                    pass_tag=f'{pass_tag}_{orientation}',
                    max_slice_distance=int(max_slice_distance),
                    search_angle_deg=float(search_angle_deg),
                    interpolation_walk_back=int(interpolation_walk_back),
                    interpolation_candidates=int(interpolation_candidates),
                    interpolate_min_radius=float(interpolate_min_radius),
                    keep_temp=bool(keep_temp),
                    prefer_memory=bool(prefer_memory),
                    reserve_bytes=int(reserve_bytes),
                )
                directional_stats.append((orientation, dict(stats)))
                _merge_added_oriented_volume_into_bridge_union(mask_mm, oriented_mm, orientation, bridge_union_mm)
            finally:
                close_memmap_array(oriented_mm)
                del oriented_mm
                if not keep_temp:
                    try:
                        oriented_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        for t in tqdm(range(mask_mm.shape[0]), desc='Interpolation: merge radial sagittal/coronal bridges'):
            bridge_slice = np.asarray(bridge_union_mm[t])
            if np.any(bridge_slice):
                mask_mm[t, :, :] |= bridge_slice
        flush_array(mask_mm)

        if directional_stats:
            aggregated: Dict[str, object] = {
                'num_objects': int(max(int(s.get('num_objects', 0)) for _, s in directional_stats)),
                'num_endpoints': int(sum(int(s.get('num_endpoints', 0)) for _, s in directional_stats)),
                'candidate_connections': int(sum(int(s.get('candidate_connections', 0)) for _, s in directional_stats)),
                'accepted_connections': int(sum(int(s.get('accepted_connections', 0)) for _, s in directional_stats)),
                'default_bridges': int(sum(int(s.get('default_bridges', 0)) for _, s in directional_stats)),
                'walk_back_bridges': int(sum(int(s.get('walk_back_bridges', 0)) for _, s in directional_stats)),
                'skipped_by_min_radius': int(sum(int(s.get('skipped_by_min_radius', 0)) for _, s in directional_stats)),
                'added_voxels': int(sum(int(s.get('added_voxels', 0)) for _, s in directional_stats)),
                'direction_modes': 'sagittal+coronal',
                'skipped': bool(all(bool(s.get('skipped', False)) for _, s in directional_stats)),
            }
        else:
            aggregated = {
                'num_objects': 0,
                'num_endpoints': 0,
                'candidate_connections': 0,
                'accepted_connections': 0,
                'default_bridges': 0,
                'walk_back_bridges': 0,
                'skipped_by_min_radius': 0,
                'added_voxels': 0,
                'direction_modes': 'sagittal+coronal',
                'skipped': True,
            }

        return aggregated
    finally:
        close_memmap_array(bridge_union_mm)
        del bridge_union_mm
        if bridge_union_path is not None and not keep_temp:
            try:
                bridge_union_path.unlink(missing_ok=True)
            except Exception:
                pass


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


def iter_view_mask_frames(mask_u8: np.ndarray, view: ViewInfo) -> Iterator[np.ndarray]:
    if view.name == 'transverse':
        for t in range(view.num_slices):
            yield np.asarray(mask_u8[t])
    elif view.name == 'sagittal':
        for y in range(view.num_slices):
            yield np.ascontiguousarray(mask_u8[:, y, :])
    elif view.name == 'coronal':
        for x in range(view.num_slices):
            yield np.ascontiguousarray(mask_u8[:, :, x])
    else:  # pragma: no cover
        raise ValueError(f'Unknown view: {view.name}')


def write_overlay_video_for_view(
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    view: ViewInfo,
    out_path: Path,
    fps: float,
) -> None:
    if view.name == 'transverse':
        write_overlay_video(volume_rgb, mask_u8, out_path, fps=fps)
        return

    proc = ffmpeg_rawvideo_writer(
        out_path,
        width=view.src_w,
        height=view.src_h,
        fps=fps,
        pix_fmt_in='rgb24',
        codec='ffv1',
        pix_fmt_out='yuv444p',
    )
    blue = np.array([0, 0, 255], dtype=np.uint8)

    try:
        assert proc.stdin is not None
        for frame_rgb, frame_mask in tqdm(
            zip(iter_view_frames(volume_rgb, view), iter_view_mask_frames(mask_u8, view)),
            total=view.num_slices,
            desc=f'Writing {view.name} overlay video ({out_path.name})',
        ):
            frame = np.asarray(frame_rgb).copy()
            m = np.asarray(frame_mask, dtype=bool)
            if np.any(m):
                frame[m] = ((frame[m].astype(np.uint16) + blue.astype(np.uint16)) // 2).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
    finally:
        close_ffmpeg_writer(proc)


def write_view_yolo_labels_from_pattern(
    mask_u8: np.ndarray,
    view: ViewInfo,
    pattern_path: Path,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, frame_mask in enumerate(tqdm(iter_view_mask_frames(mask_u8, view), total=view.num_slices, desc=f'Writing YOLO labels ({pattern_path.parent.name})')):
        fp = _format_frame_path(pattern_path, idx + 1)
        fp.parent.mkdir(parents=True, exist_ok=True)
        m = np.asarray(frame_mask) > 0
        if not np.any(m):
            fp.write_text('')
            continue
        polys = mask_to_yolo_polygons(m.astype(np.uint8))
        if not polys:
            fp.write_text('')
            continue
        lines: List[str] = []
        for poly in polys:
            coords: List[str] = []
            for x, y in poly:
                coords.append(f'{x:.6f}')
                coords.append(f'{y:.6f}')
            lines.append('0 ' + ' '.join(coords))
        fp.write_text('\n'.join(lines) + '\n')
    return pattern_path.parent


def write_view_binary_outputs_from_pattern(
    mask_u8: np.ndarray,
    view: ViewInfo,
    pattern_path: Path,
    video_path: Path,
    fps: float,
) -> Tuple[Path, Path]:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, frame_mask in enumerate(tqdm(iter_view_mask_frames(mask_u8, view), total=view.num_slices, desc=f'Writing binary TIFF sequence ({pattern_path.parent.name})')):
        img = (np.asarray(frame_mask) * 255).astype(np.uint8)
        fp = _format_frame_path(pattern_path, idx + 1)
        fp.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(fp), img)

    proc = ffmpeg_rawvideo_writer(
        video_path,
        width=view.src_w,
        height=view.src_h,
        fps=fps,
        pix_fmt_in='gray',
        codec='ffv1',
        pix_fmt_out='gray',
    )
    try:
        assert proc.stdin is not None
        for frame_mask in tqdm(iter_view_mask_frames(mask_u8, view), total=view.num_slices, desc=f'Writing binary MKV ({video_path.name})'):
            img = (np.asarray(frame_mask) * 255).astype(np.uint8)
            proc.stdin.write(img.tobytes())
    finally:
        close_ffmpeg_writer(proc)

    return pattern_path.parent, video_path


def write_additional_view_outputs(
    *,
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    view: ViewInfo,
    out_dir: Path,
    stem: str,
    fps: float,
    save_binary_pattern_value: Optional[str],
    save_labels_pattern_value: Optional[str],
    tag: Optional[str] = None,
) -> Dict[str, Path]:
    pretty = view.name.capitalize()
    tag_suffix = f'_{tag}' if tag else ''
    overlay_path = out_dir / f'{stem}_{pretty}_Overlay{tag_suffix}.mkv'
    write_overlay_video_for_view(volume_rgb, mask_u8, view, overlay_path, fps=fps)
    result_paths: Dict[str, Path] = {f'{view.name}_overlay': overlay_path}

    labels_pattern = _resolve_output_pattern(save_labels_pattern_value, DEFAULT_LABEL_PATTERN, out_dir, stem)
    if labels_pattern is not None:
        labels_pattern = _tag_frame_pattern(labels_pattern, pretty if tag is None else f'{pretty}_{tag}')
        labels_dir = write_view_yolo_labels_from_pattern(mask_u8, view, labels_pattern)
        result_paths[f'{view.name}_labels_dir'] = labels_dir

    binary_pattern = _resolve_output_pattern(save_binary_pattern_value, DEFAULT_BINARY_PATTERN, out_dir, stem)
    if binary_pattern is not None:
        binary_pattern = _tag_frame_pattern(binary_pattern, pretty if tag is None else f'{pretty}_{tag}')
        binary_video_path = out_dir / f'{stem}_{pretty}_Binary{tag_suffix}.mkv'
        tiff_dir, binary_video_path = write_view_binary_outputs_from_pattern(mask_u8, view, binary_pattern, binary_video_path, fps)
        result_paths[f'{view.name}_binary_tiff_dir'] = tiff_dir
        result_paths[f'{view.name}_binary_video'] = binary_video_path

    return result_paths


def write_multiplanar_outputs(
    *,
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    out_dir: Path,
    stem: str,
    fps: float,
    save_binary_pattern_value: Optional[str],
    save_labels_pattern_value: Optional[str],
    tag: Optional[str] = None,
) -> Dict[str, Path]:
    t_dim, h_dim, w_dim = mask_u8.shape
    views = {v.name: v for v in get_view_infos(t_dim, h_dim, w_dim, disable_multiplanar=False, azimuth_angle=0.0, include_radial=False)}
    result_paths: Dict[str, Path] = {}
    for view_name in ('sagittal', 'coronal'):
        result_paths.update(write_additional_view_outputs(
            volume_rgb=volume_rgb,
            mask_u8=mask_u8,
            view=views[view_name],
            out_dir=out_dir,
            stem=stem,
            fps=fps,
            save_binary_pattern_value=save_binary_pattern_value,
            save_labels_pattern_value=save_labels_pattern_value,
            tag=tag,
        ))
    return result_paths


def write_summary_file(
    out_path: Path,
    *,
    command: str,
    input_path: Path,
    out_dir: Path,
    scratch_dir: Path,
    volume_shape: Tuple[int, int, int],
    fps: float,
    model_paths: Sequence[str],
    view_names: Sequence[str],
    view_prediction_stats: Dict[str, int],
    interpolation_stats: List[Dict[str, object]],
    voxel_volume: Optional[int],
    final_paths: Dict[str, Path],
    augmentation_workers: int,
    interpolation_workers: int,
) -> Path:
    lines: List[str] = []
    lines.append(f'Command: {command}')
    lines.append(f'Input: {input_path}')
    lines.append(f'Output directory: {out_dir}')
    lines.append(f'Volume shape (t, Y, X): {volume_shape}')
    lines.append(f'FPS: {fps}')
    lines.append(f'Scratch directory: {scratch_dir}')
    lines.append('Workspace policy: in-memory first with disk fallback when the working set exceeds available RAM/swap')
    lines.append(f'Augmentation workers: {int(augmentation_workers)}')
    lines.append(f'Interpolation workers: {int(interpolation_workers)}')
    lines.append(f'Models: {", ".join(str(Path(m)) for m in model_paths)}')
    lines.append(f'Views: {", ".join(view_names)}')

    lines.append('')
    lines.append('View statistics:')
    total_prediction_count = 0
    for view_key in ('transverse', 'sagittal', 'coronal', 'radial'):
        count = int(view_prediction_stats.get(view_key, 0))
        total_prediction_count += count
        lines.append(f'  {view_key.capitalize()}: predictions={count}')
    lines.append(f'  Total prediction count: {int(total_prediction_count)}')

    if interpolation_stats:
        lines.append('')
        lines.append('Interpolation statistics (per pass):')
        pass_indices = sorted({int(s.get('pass_index', 0)) for s in interpolation_stats})
        for pass_idx in pass_indices:
            stats_this_pass = [s for s in interpolation_stats if int(s.get('pass_index', 0)) == pass_idx]
            total_objects = sum(int(s.get('num_objects', 0)) for s in stats_this_pass)
            total_endpoints = sum(int(s.get('num_endpoints', 0)) for s in stats_this_pass)
            total_candidates = sum(int(s.get('candidate_connections', 0)) for s in stats_this_pass)
            total_accepted = sum(int(s.get('accepted_connections', 0)) for s in stats_this_pass)
            total_default_bridges = sum(int(s.get('default_bridges', 0)) for s in stats_this_pass)
            total_walk_back = sum(int(s.get('walk_back_bridges', 0)) for s in stats_this_pass)
            total_skipped = sum(int(s.get('skipped_by_min_radius', 0)) for s in stats_this_pass)
            total_added_voxels = sum(int(s.get('added_voxels', 0)) for s in stats_this_pass)
            lines.append(
                f'  Pass {pass_idx}: objects={total_objects}, endpoints={total_endpoints}, '
                f'candidate_connections={total_candidates}, accepted_connections={total_accepted}, '
                f'default_bridges={total_default_bridges}, walk_back_bridges={total_walk_back}, '
                f'bridges_skipped_by_--interpolate_min_radius={total_skipped}, added_voxels={total_added_voxels}'
            )
            for s in sorted(stats_this_pass, key=lambda d: (str(d.get('model', '')), str(d.get('view', '')))):
                mode_suffix = ''
                if s.get('direction_modes'):
                    mode_suffix = f", direction_modes={s.get('direction_modes')}"
                lines.append(
                    f"    {s.get('model', '?')}/{s.get('view', '?')}: "
                    f"objects={int(s.get('num_objects', 0))}, "
                    f"endpoints={int(s.get('num_endpoints', 0))}, "
                    f"candidate_connections={int(s.get('candidate_connections', 0))}, "
                    f"accepted_connections={int(s.get('accepted_connections', 0))}, "
                    f"default_bridges={int(s.get('default_bridges', 0))}, "
                    f"walk_back_bridges={int(s.get('walk_back_bridges', 0))}, "
                    f"bridges_skipped_by_--interpolate_min_radius={int(s.get('skipped_by_min_radius', 0))}, "
                    f"added_voxels={int(s.get('added_voxels', 0))}, "
                    f"skipped={bool(s.get('skipped', False))}{mode_suffix}"
                )

    if voxel_volume is not None:
        lines.append('')
        lines.append(f'voxel_volume: {int(voxel_volume)}')

    lines.append('')
    lines.append('Final outputs:')
    for key in sorted(final_paths.keys()):
        lines.append(f'{key}: {final_paths[key]}')

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
        raise ValueError('--model must specify at least one model path')
    for m in model_paths:
        if not Path(m).expanduser().exists():
            raise FileNotFoundError(m)

    angles = _parse_angles(args.angle) or [0.0, 120.0, 240.0]

    if args.min_conf > 0 and args.min_conf < args.conf:
        raise ValueError('--min_conf must be equal to or greater than --conf')
    if int(args.interpolate) < 0:
        raise ValueError('--interpolate must be >= 0')
    if int(args.interpolation_walk_back) < 0:
        raise ValueError('--interpolation_walk_back must be >= 0')
    if int(args.interpolation_candidates) < 1:
        raise ValueError('--interpolation_candidates must be >= 1')
    if int(args.interpolate_passes) < 1:
        raise ValueError('--interpolate_passes must be >= 1')
    if float(args.interpolate_min_radius) < 0:
        raise ValueError('--interpolate_min_radius must be >= 0')
    if float(args.min_radius) < 0:
        raise ValueError('--min_radius must be >= 0')
    if float(args.azimuth_angle) < 0:
        raise ValueError('--azimuth_angle must be >= 0')
    if not (-90.0 < float(args.interpolation_search_angle) < 90.0):
        raise ValueError('--interpolation_search_angle must be greater than -90 and less than 90')

    out_dir = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / input_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = choose_scratch_dir(args.scratch_dir, out_dir, input_path.stem)
    expose_scratch_in_output(out_dir, temp_dir)
    print(f"Bulk scratch dir: {temp_dir}")

    info = ffprobe_info(input_path)
    W = int(info['width'])
    H = int(info['height'])
    T = int(info['num_frames'])
    fps = float(info['fps'])

    vol_path = temp_dir / 'input_volume.rgb24.dat'
    volume_rgb = decode_video_to_memmap_rgb24(
        input_video=input_path,
        out_dat=vol_path,
        num_frames=T,
        width=W,
        height=H,
        overwrite=False,
        prefer_memory=True,
    )
    (temp_dir / 'input_volume.meta.json').write_text(
        json.dumps({'shape': [T, H, W, 3], 'dtype': 'uint8', 'fps': fps}, indent=2)
    )

    views = get_view_infos(
        T=T,
        H=H,
        W=W,
        disable_multiplanar=bool(args.disable_multiplanar),
        azimuth_angle=float(args.azimuth_angle),
        include_radial=True,
    )
    cartesian_views = orthogonal_views_only(views)

    shifts: List[Tuple[int, int, str]] = [(0, 0, 'none')]
    if int(args.shift) != 0:
        s = abs(int(args.shift))
        shifts = [
            (0, 0, 'none'),
            (0, -s, 'up'),
            (0, +s, 'down'),
            (-s, 0, 'left'),
            (+s, 0, 'right'),
        ]

    yolo_models: List[Tuple[str, object]] = []
    for m in model_paths:
        name = Path(m).stem
        print(f'Loading model: {name} ({m})')
        yolo_models.append((name, load_ultralytics_model(m)))

    pred_cfg = PredictConfig(
        imgsz=args.imgsz,
        conf=args.conf,
        device=str(args.device),
        half=bool(args.half),
        int8=bool(args.int8),
    )

    augmentation_workers = resolve_worker_count(
        int(args.augmentation_workers),
        'YOLO_TTA_AUG_WORKERS',
        default_augmentation_workers(),
        max_tasks=max(1, max(v.num_slices for v in views)),
    )
    interpolation_workers = resolve_worker_count(
        int(args.interpolation_workers),
        'YOLO_TTA_INTERPOLATION_WORKERS',
        default_interpolation_workers(),
        max_tasks=max(1, len(yolo_models) * len(views)),
    )
    print(f'Augmentation workers: {augmentation_workers}')
    print(f'Interpolation workers: {interpolation_workers}')

    if augmentation_workers > 1 or interpolation_workers > 1:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    view_prediction_stats: Dict[str, int] = {
        'transverse': 0,
        'sagittal': 0,
        'coronal': 0,
        'radial': 0,
    }

    for view in views:
        extra = ''
        if view.family == 'radial':
            extra = f', azimuths={view.num_slices}'
        print(f"\n=== View: {view.name} ({view.src_w}x{view.src_h}, slices={view.num_slices}{extra}) ===")

        union_by_model_view: Dict[str, np.ndarray] = {}
        confmap_by_model_view: Dict[str, np.ndarray] = {}
        union_paths: Dict[str, Path] = {}
        confmap_paths: Dict[str, Path] = {}

        for model_name, _ in yolo_models:
            bytes_native = bytes_for_packbits(view.src_h, view.src_w)
            union_path = temp_dir / 'union' / model_name / f'{view.name}.union.packbits.dat'
            confmap_path = temp_dir / 'union' / model_name / f'{view.name}.confmap.f16.dat'
            union_path.parent.mkdir(parents=True, exist_ok=True)

            union_mm = allocate_workspace_array(
                shape=(view.num_slices, bytes_native),
                dtype=np.uint8,
                path=union_path,
                desc=f'{model_name}/{view.name} union workspace',
                prefer_memory=True,
            )
            confmap_mm = allocate_workspace_array(
                shape=(view.num_slices, view.src_h, view.src_w),
                dtype=np.float16,
                path=confmap_path,
                desc=f'{model_name}/{view.name} confidence workspace',
                prefer_memory=True,
            )

            union_by_model_view[model_name] = union_mm
            confmap_by_model_view[model_name] = confmap_mm
            union_paths[model_name] = union_path
            confmap_paths[model_name] = confmap_path

        aug_jobs = build_aug_jobs_for_view(
            view=view,
            angles=angles,
            shifts=shifts,
            out_size=args.imgsz,
            temp_dir=temp_dir,
        )

        ensure_augmented_videos(
            volume_rgb=volume_rgb,
            view=view,
            aug_jobs=aug_jobs,
            fps=fps,
            augmentation_workers=augmentation_workers,
        )

        for job in aug_jobs:
            aug_id = job.aug_id
            aug_video = job.video_path
            aug_meta = job.meta_path
            aff = job.aff

            for model_name, yolo in yolo_models:
                pred_prefix = temp_dir / 'preds' / model_name / view.name / f'{view.name}_{aug_id}'
                pred_stats = predict_video_and_accumulate(
                    model=yolo,
                    video_path=aug_video,
                    num_frames=view.num_slices,
                    out_size=args.imgsz,
                    pred_out_prefix=pred_prefix,
                    cfg=pred_cfg,
                    view_union_mm=union_by_model_view[model_name],
                    view_confmap_mm=confmap_by_model_view[model_name],
                    M_out_to_native=aff.M_out_to_src,
                    native_h=view.src_h,
                    native_w=view.src_w,
                )
                view_prediction_stats[view.name] = int(view_prediction_stats.get(view.name, 0)) + int(pred_stats.get('prediction_count', 0))

            if not args.troubleshooting:
                try:
                    aug_video.unlink(missing_ok=True)
                    aug_meta.unlink(missing_ok=True)
                except Exception:
                    pass

        for model_name, _ in yolo_models:
            print(f"\n--- Postprocessing view '{view.name}' for model '{model_name}' ---")
            if args.min_conf > 0:
                apply_min_conf_filter_with_confmap_inplace(
                    union_by_model_view[model_name],
                    confmap_by_model_view[model_name],
                    float(args.min_conf),
                    view.src_h,
                    view.src_w,
                )
                flush_array(union_by_model_view[model_name])
                flush_array(confmap_by_model_view[model_name])

            fill_2d_holes_inplace(union_by_model_view[model_name], view.src_h, view.src_w)
            flush_array(union_by_model_view[model_name])

            out_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.u8.dat'
            if view.family == 'radial':
                view_volumes_by_model[model_name][view.name] = backproject_radial_union_to_volume(
                    union_mm=union_by_model_view[model_name],
                    radial_view=view,
                    out_path=out_path,
                    desc=f'Backprojecting {model_name}/{view.name}',
                    prefer_memory=True,
                )
            else:
                view_volumes_by_model[model_name][view.name] = unpack_view_union_to_volume(
                    union_mm=union_by_model_view[model_name],
                    num_slices=view.num_slices,
                    h=view.src_h,
                    w=view.src_w,
                    out_path=out_path,
                    desc=f'Unpacking {model_name}/{view.name}',
                    prefer_memory=True,
                )

            if float(args.min_radius) > 0:
                apply_view_min_radius_filter_inplace(
                    view_volumes_by_model[model_name][view.name],
                    view,
                    float(args.min_radius),
                )

            confmap_mm_done = confmap_by_model_view.pop(model_name, None)
            union_mm_done = union_by_model_view.pop(model_name, None)
            close_memmap_array(confmap_mm_done)
            close_memmap_array(union_mm_done)
            del confmap_mm_done, union_mm_done

            if not args.troubleshooting:
                try:
                    confmap_paths[model_name].unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    union_paths[model_name].unlink(missing_ok=True)
                except Exception:
                    pass

        union_by_model_view.clear()
        confmap_by_model_view.clear()
        gc.collect()

    gc.collect()

    def build_current_finalized_volume(snapshot_stem: str) -> np.ndarray:
        ensemble_mm = assemble_current_ensemble_volume(
            view_volumes_by_model=view_volumes_by_model,
            T=T,
            H=H,
            W=W,
            disable_multiplanar=bool(args.disable_multiplanar),
            out_path=temp_dir / f'{snapshot_stem}.u8.dat',
            prefer_memory=True,
        )
        fill_3d_voids_inplace_streaming(
            ensemble_mm,
            temp_dir / f'{snapshot_stem}_voidfill',
            keep_temp=bool(args.troubleshooting),
            prefer_memory=True,
        )
        return ensemble_mm

    interpolation_stats: List[Dict[str, object]] = []

    if bool(args.troubleshooting) and int(args.interpolate) > 0:
        print('\n=== Writing troubleshooting outputs: pass 0 (pre-interpolation) ===')
        pass0_mm = build_current_finalized_volume('ensemble_pass0')
        pass0_paths = write_pipeline_outputs(
            volume_rgb=volume_rgb,
            mask_u8=pass0_mm,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_binary_pattern_value=args.save_binary,
            save_labels_pattern_value=args.save_labels,
            save_nrrd_flag=bool(args.save_nrrd),
            tag='Pass0',
        )
        if bool(args.save_multiplanar):
            pass0_paths.update(write_multiplanar_outputs(
                volume_rgb=volume_rgb,
                mask_u8=pass0_mm,
                out_dir=out_dir,
                stem=input_path.stem,
                fps=fps,
                save_binary_pattern_value=args.save_binary,
                save_labels_pattern_value=args.save_labels,
                tag='Pass0',
            ))
        close_memmap_array(pass0_mm)
        del pass0_mm
        gc.collect()

    if int(args.interpolate) > 0:
        total_passes = int(args.interpolate_passes)
        for pass_idx in range(1, total_passes + 1):
            print(
                f"\n=== Interpolation pass {pass_idx}/{total_passes} "
                f"(distance={int(args.interpolate)}, walk_back={int(args.interpolation_walk_back)}, "
                f"candidates={int(args.interpolation_candidates)}, "
                f"min_radius={float(args.interpolate_min_radius):g}, "
                f"search_angle={float(args.interpolation_search_angle):g}) ==="
            )
            task_specs: List[Tuple[str, ViewInfo]] = [
                (model_name, view)
                for model_name in sorted(view_volumes_by_model.keys())
                for view in views
            ]

            def _run_interpolation_task(task_index: int) -> Dict[str, object]:
                model_name, view = task_specs[int(task_index)]
                print(f"\n--- Interpolating model '{model_name}' view '{view.name}' ---")
                current_mm = view_volumes_by_model[model_name][view.name]
                if view.family == 'radial':
                    stats_local = interpolate_radial_view_pass_inplace(
                        mask_mm=current_mm,
                        work_dir=temp_dir / 'interpolation' / model_name / view.name,
                        pass_tag=f'pass{pass_idx}',
                        max_slice_distance=int(args.interpolate),
                        search_angle_deg=float(args.interpolation_search_angle),
                        interpolation_walk_back=int(args.interpolation_walk_back),
                        interpolation_candidates=int(args.interpolation_candidates),
                        interpolate_min_radius=float(args.interpolate_min_radius),
                        keep_temp=bool(args.troubleshooting),
                        prefer_memory=True,
                    )
                else:
                    stats_local = interpolate_view_volume_pass_inplace(
                        mask_mm=current_mm,
                        work_dir=temp_dir / 'interpolation' / model_name / view.name,
                        pass_tag=f'pass{pass_idx}',
                        max_slice_distance=int(args.interpolate),
                        search_angle_deg=float(args.interpolation_search_angle),
                        interpolation_walk_back=int(args.interpolation_walk_back),
                        interpolation_candidates=int(args.interpolation_candidates),
                        interpolate_min_radius=float(args.interpolate_min_radius),
                        keep_temp=bool(args.troubleshooting),
                        prefer_memory=True,
                    )

                stats_local = dict(stats_local)
                stats_local.update({
                    'pass_index': int(pass_idx),
                    'model': str(model_name),
                    'view': str(view.name),
                    'max_slice_distance': int(args.interpolate),
                    'interpolation_walk_back': int(args.interpolation_walk_back),
                    'interpolation_candidates': int(args.interpolation_candidates),
                    'interpolation_search_angle': float(args.interpolation_search_angle),
                })
                return stats_local

            if interpolation_workers > 1 and len(task_specs) > 1:
                for stats in parallel_map_in_order(
                    _run_interpolation_task,
                    range(len(task_specs)),
                    max_workers=min(interpolation_workers, len(task_specs)),
                    max_pending=min(interpolation_workers, len(task_specs)),
                ):
                    interpolation_stats.append(dict(stats))
            else:
                for task_idx in range(len(task_specs)):
                    interpolation_stats.append(_run_interpolation_task(task_idx))

            if bool(args.troubleshooting) and pass_idx < total_passes:
                print(f"\n=== Writing troubleshooting outputs: pass {pass_idx} ===")
                pass_mm = build_current_finalized_volume(f'ensemble_pass{pass_idx}')
                pass_paths = write_pipeline_outputs(
                    volume_rgb=volume_rgb,
                    mask_u8=pass_mm,
                    out_dir=out_dir,
                    stem=input_path.stem,
                    fps=fps,
                    save_binary_pattern_value=args.save_binary,
                    save_labels_pattern_value=args.save_labels,
                    save_nrrd_flag=bool(args.save_nrrd),
                    tag=f'Pass{pass_idx}',
                )
                if bool(args.save_multiplanar):
                    pass_paths.update(write_multiplanar_outputs(
                        volume_rgb=volume_rgb,
                        mask_u8=pass_mm,
                        out_dir=out_dir,
                        stem=input_path.stem,
                        fps=fps,
                        save_binary_pattern_value=args.save_binary,
                        save_labels_pattern_value=args.save_labels,
                        tag=f'Pass{pass_idx}',
                    ))
                close_memmap_array(pass_mm)
                del pass_mm
                gc.collect()
    else:
        for model_name in sorted(view_volumes_by_model.keys()):
            for view in views:
                entry: Dict[str, object] = {
                    'pass_index': 0,
                    'model': str(model_name),
                    'view': str(view.name),
                    'max_slice_distance': int(args.interpolate),
                    'interpolation_walk_back': int(args.interpolation_walk_back),
                    'interpolation_candidates': int(args.interpolation_candidates),
                    'interpolation_search_angle': float(args.interpolation_search_angle),
                    'num_objects': 0,
                    'num_endpoints': 0,
                    'candidate_connections': 0,
                    'accepted_connections': 0,
                    'default_bridges': 0,
                    'walk_back_bridges': 0,
                    'skipped_by_min_radius': 0,
                    'added_voxels': 0,
                    'skipped': True,
                }
                if view.family == 'radial':
                    entry['direction_modes'] = 'sagittal+coronal'
                interpolation_stats.append(entry)

    final_ensemble_mm = assemble_current_ensemble_volume(
        view_volumes_by_model=view_volumes_by_model,
        T=T,
        H=H,
        W=W,
        disable_multiplanar=bool(args.disable_multiplanar),
        out_path=temp_dir / 'ensemble_volume_final.u8.dat',
        prefer_memory=True,
    )

    print('\n=== 3D void fill after multiplanar/model union ===')
    fill_3d_voids_inplace_streaming(
        final_ensemble_mm,
        temp_dir / 'ensemble_volume_final_voidfill',
        keep_temp=bool(args.troubleshooting),
        prefer_memory=True,
    )
    final_paths = write_pipeline_outputs(
        volume_rgb=volume_rgb,
        mask_u8=final_ensemble_mm,
        out_dir=out_dir,
        stem=input_path.stem,
        fps=fps,
        save_binary_pattern_value=args.save_binary,
        save_labels_pattern_value=args.save_labels,
        save_nrrd_flag=bool(args.save_nrrd),
        tag=None,
    )
    if bool(args.save_multiplanar):
        final_paths.update(write_multiplanar_outputs(
            volume_rgb=volume_rgb,
            mask_u8=final_ensemble_mm,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_binary_pattern_value=args.save_binary,
            save_labels_pattern_value=args.save_labels,
            tag=None,
        ))

    voxel_volume = None
    if bool(args.voxel_volume):
        voxel_volume = 0
        for z in tqdm(range(final_ensemble_mm.shape[0]), desc='Counting voxel_volume'):
            voxel_volume += int(np.count_nonzero(np.asarray(final_ensemble_mm[z])))

    summary_path = write_summary_file(
        out_dir / f'{input_path.stem}_Summary.txt',
        command=shlex.join([str(x) for x in sys.argv]),
        input_path=input_path,
        out_dir=out_dir,
        scratch_dir=temp_dir,
        volume_shape=(T, H, W),
        fps=fps,
        model_paths=model_paths,
        view_names=[v.name for v in views],
        view_prediction_stats=view_prediction_stats,
        interpolation_stats=interpolation_stats,
        voxel_volume=voxel_volume,
        final_paths=final_paths,
        augmentation_workers=augmentation_workers,
        interpolation_workers=interpolation_workers,
    )

    close_memmap_array(final_ensemble_mm)
    for model_views in view_volumes_by_model.values():
        for mm in model_views.values():
            close_memmap_array(mm)
        model_views.clear()
    close_memmap_array(volume_rgb)
    gc.collect()

    if not args.troubleshooting:
        try:
            for child in list(temp_dir.iterdir()):
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if temp_dir != out_dir / 'temp':
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_link = out_dir / 'temp'
                if temp_link.is_symlink():
                    temp_link.unlink(missing_ok=True)
        except Exception:
            pass

    print('\nDone.')
    print(f'Output dir: {out_dir}')
    print(f'Scratch dir: {temp_dir}')
    print(f"Final overlay: {final_paths['overlay']}")
    print(f'Summary: {summary_path}')


if __name__ == "__main__":
    main()
