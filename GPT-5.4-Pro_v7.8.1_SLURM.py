#!/usr/bin/env python3
"""
YOLO segmentation test-time augmentation (TTA) for large cylindrical video volumes.

This v7.8.1_SLURM-aligned script:
  - builds Transverse, optional Sagittal/Coronal, and optional Radial view families
  - prefetches active-view render assets so view creation can overlap across active families
  - generates rotated FFV1 MKV augmentation videos ahead of inference
  - supports dense tiled inference for all active views, including multi-configuration sliding-window sets,
    with direct per-tile gated OR back into the parent full-frame view volume
  - reuses a native radial frame cache during dense tiled rendering so radial tiles do not recompute
    the same Lanczos-5 slices for every tile location
  - runs Ultralytics YOLO segmentation sequentially on each pre-generated video source
  - inverse-maps predictions into native view coordinates, applies confidence filtering, hole filling, interpolation, and final 3D void fill
  - saves the transverse default overlay plus optional labels, binary masks, NRRD, multiplanar, radial, image-sequence, and TTA outputs

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
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

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


def _parse_int_list(values: Sequence[str] | str | int | None) -> List[int]:
    """Accept comma and/or whitespace separated integer lists."""
    if values is None:
        return []
    if isinstance(values, int):
        return [int(values)]
    raw_values = [values] if isinstance(values, str) else [str(v) for v in values]
    parts: List[str] = []
    for raw in raw_values:
        raw = str(raw).strip()
        if not raw:
            continue
        parts.extend([p for p in re.split(r"[,\s]+", raw) if p])
    return [int(p) for p in parts]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLO segmentation TTA for large cylindrical video volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", required=True, type=str, help="Input video path")
    p.add_argument("--output", default=None, type=str, help="Output directory (default ./{Filename}/)")
    p.add_argument("--device", default="0", type=str, help="Device passed to YOLO predict")
    p.add_argument("--model", required=True, nargs="+", type=str, help="One or more YOLO segmentation model paths")

    p.add_argument("--imgsz", default=1536, type=int, help="Square input size used for YOLO predict")
    p.add_argument("--conf", default=0.15, type=float, help="Passed to YOLO predict")
    p.add_argument("--min_conf", default=0.30, type=float,
                   help="Remove prediction-set objects whose combined confidence is below this threshold. 0 disables the check")
    p.add_argument("--half", action="store_true", help="Enable FP16 inference")
    p.add_argument("--int8", action="store_true", help="Enable INT8 inference if supported")

    p.add_argument("--angle", default="0,120,240", type=str,
                   help="Rotation angles in degrees for augmentation (comma or whitespace separated)")
    p.add_argument("--min_radius", default=0.0, type=float,
                   help="Remove objects with a transverse-plane radius smaller than this value. 0 disables the check")

    p.add_argument("--enable_multiplanar", action="store_true",
                   help="Enable Sagittal and Coronal Cartesian views in addition to the required Transverse view")
    p.add_argument("--azimuth_angle", default=0.0, type=float,
                   help="Angular spacing in degrees for radial diameter slices over [0,180]. 0 disables radial views")

    p.add_argument("--tile_size", nargs="+", default=["0"], type=str,
                   help="One or more square dense-tile side lengths in source pixels for all active views. 0 disables dense tiled predictions")
    p.add_argument("--tile_stride", nargs="+", default=["0"], type=str,
                   help="One or more dense-tile strides in source pixels. Must match --tile_size when dense tiling is active")

    p.add_argument("--save_images", action="store_true", help="Save unlabeled image sequences for all active views")
    p.add_argument("--save_labels", nargs="?", const="__DEFAULT__", default=None, type=str,
                   help="Save final YOLO segmentation labels per frame. Optional custom pattern, e.g. labels/{Filename}_%%04d.txt")
    p.add_argument("--save_TTA", action="store_true",
                   help="Save the rotated augmentation videos together with the final labels mapped to each augmentation")
    p.add_argument("--save_binary", nargs="?", const="__DEFAULT__", default=None, type=str,
                   help="Save final binary masks as a TIFF sequence plus an FFV1 MKV. Optional custom TIFF pattern, e.g. binary_masks/{Filename}_Binary_%%04d.tiff")
    p.add_argument("--save_nrrd", action="store_true", help="Save the final binary mask volume as an NRRD file")
    p.add_argument("--save_multiplanar", action="store_true",
                   help="Save additional Sagittal and Coronal outputs. If multiplanar inference is disabled, reslice the final unified volume for saving only")
    p.add_argument("--save_radial", action="store_true",
                   help="Save additional Radial outputs. A warning is emitted and nothing is saved when --azimuth_angle is 0")
    p.add_argument("--voxel_volume", action="store_true", help="Count white voxels in the final binary output and save the value to the summary text file")

    p.add_argument("--troubleshooting", action="store_true",
                   help="Keep temporary files and save outputs before each interpolation pass")

    p.add_argument("--interpolate", default=15, type=int,
                   help="Maximum slice distance used to search for interpolation candidates in Cartesian views. 0 disables interpolation")
    p.add_argument("--interpolation_walk_back", default=3, type=int,
                   help="Additional source slices to bridge before the endpoint slice. 0 disables walk-back bridges")
    p.add_argument("--interpolation_candidates", default=1, type=int,
                   help="Accept up to the Nth nearest interpolation candidate per endpoint projection")
    p.add_argument("--interpolate_passes", default=1, type=int,
                   help="Run the interpolation process this many passes, treating the previous pass as real")
    p.add_argument("--interpolate_min_radius", default=3, type=float,
                   help="Reject a candidate connection if the bridge radius is equal to, or smaller than, this value. 0 disables the check")
    p.add_argument("--interpolation_search_angle", default=15.0, type=float,
                   help="Projection growth angle in degrees. Must be greater than -90 and less than 90")

    return p


# --------------------------
# Scratch / temp layout
# --------------------------


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


def _slurm_allocated_cpu_count() -> Optional[int]:
    for env_name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        raw = os.environ.get(env_name, '').strip()
        if not raw:
            continue
        m = re.search(r'(\d+)', raw)
        if m is not None:
            try:
                return max(1, int(m.group(1)))
            except Exception:
                continue
    return None


def _cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, int(len(affinity)))
    except Exception:
        pass

    slurm_cpus = _slurm_allocated_cpu_count()
    if slurm_cpus is not None:
        return max(1, int(slurm_cpus))

    return max(1, int(os.cpu_count() or 1))


def default_worker_budget() -> int:
    return max(1, int(_cpu_count()))


def default_augmentation_workers() -> int:
    return int(default_worker_budget())


def default_interpolation_workers() -> int:
    return int(default_worker_budget())


def default_output_workers() -> int:
    return int(default_worker_budget())


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


def copy_workspace_array(
    src: np.ndarray,
    path: Optional[Path],
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
) -> np.ndarray:
    """Allocate a workspace matching ``src`` and copy the contents in parallel."""
    dst = allocate_workspace_array(
        shape=src.shape,
        dtype=src.dtype,
        path=path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    if src.ndim <= 1:
        np.copyto(dst, src)
        flush_array(dst)
        return dst

    total = int(src.shape[0])

    def _copy(idx: int) -> None:
        np.copyto(dst[int(idx)], src[int(idx)])

    parallel_for_indices(
        total,
        _copy,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=f'{desc} copy',
        show_progress=False,
    )
    flush_array(dst)
    return dst


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


def parallel_for_indices(
    count: int,
    func: Callable[[int], None],
    *,
    max_workers: int,
    desc: str,
    show_progress: bool = True,
) -> None:
    total = max(0, int(count))
    if total <= 0:
        return

    workers = max(1, min(int(max_workers), total))
    if workers <= 1:
        iterable = tqdm(range(total), desc=desc) if show_progress else range(total)
        for idx in iterable:
            func(int(idx))
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, int(idx)) for idx in range(total)]
        if show_progress:
            with tqdm(total=total, desc=desc) as pbar:
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)
        else:
            for fut in as_completed(futures):
                fut.result()


def workspace_anon_cap_bytes() -> int:
    """Return the soft cap for anonymous in-memory workspaces.

    The SLURM target exposes large RAM and swap capacity, so the default policy prefers
    a capacity-relative cap instead of a small fixed hard limit:
      - YOLO_TTA_MAX_ANON_WORKSPACE_GIB: hard GiB cap (default 0 = disabled)
      - YOLO_TTA_MAX_ANON_WORKSPACE_FRACTION: fraction of total RAM+swap (default 0.50)
    The lower non-zero cap wins.
    """
    hard_cap_gib = max(0.0, _env_float('YOLO_TTA_MAX_ANON_WORKSPACE_GIB', 0.0))
    fraction = max(0.0, _env_float('YOLO_TTA_MAX_ANON_WORKSPACE_FRACTION', 0.50))

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


def augmentation_staging_cap_bytes() -> int:
    hard_cap_gib = max(0.0, _env_float('YOLO_TTA_MAX_AUG_STAGING_GIB', 0.0))
    fraction = max(0.0, _env_float('YOLO_TTA_MAX_AUG_STAGING_FRACTION', 0.75))

    hard_cap = int(hard_cap_gib * GIB)
    total_cap = int(total_anon_capacity_bytes() * fraction) if fraction > 0.0 else 0

    caps = [v for v in (hard_cap, total_cap) if int(v) > 0]
    if not caps:
        return 0
    return int(min(caps))


def augmentation_staging_budget_summary(required_bytes: int, reserve_bytes: int = 64 * GIB) -> str:
    avail = available_anon_work_bytes()
    cap = augmentation_staging_cap_bytes()
    reserve = int(max(0, reserve_bytes))
    parts = [
        f'need={required_bytes / GIB:.1f} GiB',
        f'avail+swap={avail / GIB:.1f} GiB',
        f'reserve={reserve / GIB:.1f} GiB',
    ]
    if cap > 0:
        parts.append(f'aug-staging-cap={cap / GIB:.1f} GiB')
    return ', '.join(parts)


def should_use_in_memory_augmentation_staging(required_bytes: int, reserve_bytes: int = 64 * GIB) -> bool:
    if int(required_bytes) <= 0:
        return False

    avail = available_anon_work_bytes()
    reserve = int(max(0, reserve_bytes))
    cap = augmentation_staging_cap_bytes()

    if cap > 0 and int(required_bytes) > cap:
        return False
    return avail >= int(required_bytes) + reserve


def estimate_augmented_view_staging_bytes(num_slices: int, out_size: int, num_jobs: int) -> int:
    return array_nbytes((int(num_jobs), int(num_slices), int(out_size), int(out_size), 3), np.uint8)


def choose_slice_parallel_workers(requested_workers: int, num_items: int) -> int:
    return max(1, min(int(requested_workers), int(max(1, num_items))))


def choose_augmentation_writer_workers(num_jobs: int) -> int:
    return max(1, int(num_jobs))


def choose_scratch_dir(preferred: Optional[str], out_dir: Path, stem: str) -> Path:
    """Pick the bulk disk-backed scratch root.

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
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
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

    nf = st.get("nb_frames", None)
    if nf is None or str(nf).strip() == "" or str(nf) == "N/A":
        # Fast fallback: count packets without decoding
        fallback_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "json",
            str(video_path),
        ]
        p2 = subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
        info2 = json.loads(p2.stdout)
        nf = info2["streams"][0].get("nb_read_packets", None)
    if nf is None or str(nf).strip() == "" or str(nf) == "N/A":
        raise RuntimeError(
            "ffprobe could not determine frame count (nb_frames/nb_read_packets missing)."
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

    The default policy keeps the decoded source volume in RAM when the workspace budget allows it.
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
    src_w: int
    src_h: int
    out_size: int
    canvas_w: int
    canvas_h: int
    pad_size: int
    pad_off_x: float
    pad_off_y: float
    M_out_to_src: np.ndarray      # 2x3 float32
    M_src_to_out: np.ndarray      # 2x3 float32
    M_canvas_to_src: np.ndarray   # 2x3 float32, pre-scale augmented canvas -> native view
    M_src_to_canvas: np.ndarray   # 2x3 float32, native view -> pre-scale augmented canvas


@dataclass(frozen=True)
class AugJob:
    aug_id: str
    angle_deg: float
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


def _format_angle_aug_id(angle_deg: float) -> str:
    token = f'{float(angle_deg):g}'.replace('-', 'm').replace('.', 'p')
    return f'a{token}'


def build_affine(
    view: str,
    src_w: int,
    src_h: int,
    out_size: int,
    angle_deg: float,
    pad_mode: str,
) -> AffineSpec:
    """
    Build a single affine transform that performs, in one pass:

      source(native view) -> optional black-padding canvas -> rotation around canvas center
      -> scale to out_size x out_size

    Transverse uses pad_mode='clamp', which rotates directly on the source-sized canvas so non-90°
    content that leaves the source frame is discarded. Sagittal/Coronal/Radial use pad_mode='pad'
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
    M_scale = np.array(
        [
            [sx, 0.0, cx_out - sx * cx_canvas],
            [0.0, sy, cy_out - sy * cy_canvas],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    M_src_to_canvas3 = M_rot @ M_pad
    M_canvas_to_src3 = np.linalg.inv(M_src_to_canvas3)
    M_src_to_out3 = M_scale @ M_src_to_canvas3
    M_out_to_src3 = np.linalg.inv(M_src_to_out3)

    return AffineSpec(
        view=view,
        angle_deg=float(angle_deg),
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
        M_canvas_to_src=M_canvas_to_src3[:2, :3].astype(np.float32),
        M_src_to_canvas=M_src_to_canvas3[:2, :3].astype(np.float32),
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


def choose_radial_exact_block_frames(diameter: int, target_bytes: int = 256 * 1024 * 1024) -> int:
    env = os.environ.get('YOLO_TTA_RADIAL_EXACT_BLOCK_FRAMES', '').strip()
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass

    bytes_per_frame = max(1, int(diameter) * 10 * 3 * np.dtype(np.float32).itemsize)
    block = max(1, int(target_bytes // bytes_per_frame))
    return max(1, min(256, block))



def extract_radial_slice_frame(volume_rgb: np.ndarray, sampler: RadialSampler) -> np.ndarray:
    t_dim = int(volume_rgb.shape[0])
    out = np.empty((t_dim, sampler.diameter, 3), dtype=np.uint8)

    x_w = np.asarray(sampler.x_w, dtype=np.float32)[None, :, :, None]
    y_w = np.asarray(sampler.y_w, dtype=np.float32)
    block_frames = choose_radial_exact_block_frames(sampler.diameter)

    for start in range(0, t_dim, block_frames):
        stop = min(t_dim, start + block_frames)
        block = np.asarray(volume_rgb[start:stop])
        acc = np.zeros((stop - start, sampler.diameter, 3), dtype=np.float32)
        for yi in range(sampler.y_idx.shape[1]):
            samples = block[:, sampler.y_idx[:, yi][:, None], sampler.x_idx, :].astype(np.float32, copy=False)
            row = np.sum(samples * x_w, axis=2)
            acc += row * y_w[:, yi][None, :, None]
        out[start:stop, :, :] = np.clip(np.rint(acc), 0.0, 255.0).astype(np.uint8)

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

    Default: disabled so the exact Lanczos-5 path remains the default.
    Set YOLO_TTA_RADIAL_FAST=1 to opt into the faster OpenCV remap path.
    """
    return _env_flag('YOLO_TTA_RADIAL_FAST', False)


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
    out_size: int,
    temp_dir: Path,
) -> List[AugJob]:
    aug_dir = temp_dir / 'aug' / view.name
    jobs: List[AugJob] = []

    for angle in angles:
        aug_id = _format_angle_aug_id(float(angle))
        aff = build_affine(
            view=view.name,
            src_w=view.src_w,
            src_h=view.src_h,
            out_size=out_size,
            angle_deg=float(angle),
            pad_mode=view.pad_mode,
        )
        jobs.append(
            AugJob(
                aug_id=aug_id,
                angle_deg=float(angle),
                video_path=aug_dir / f'{view.name}_{aug_id}.mkv',
                meta_path=aug_dir / f'{view.name}_{aug_id}.meta.json',
                aff=aff,
            )
        )
    return jobs


@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    tile_stride: int
    config_id: str


def resolve_tile_configs(tile_sizes_raw: Sequence[str] | str | int | None, tile_strides_raw: Sequence[str] | str | int | None) -> List[TileConfig]:
    tile_sizes = _parse_int_list(tile_sizes_raw)
    tile_strides = _parse_int_list(tile_strides_raw)

    if not tile_sizes:
        tile_sizes = [0]
    if not tile_strides:
        tile_strides = [0]

    if len(tile_sizes) != len(tile_strides):
        raise ValueError('--tile_size and --tile_stride must contain the same number of values')

    if len(tile_sizes) == 1 and int(tile_sizes[0]) == 0:
        if any(int(v) != 0 for v in tile_strides):
            raise ValueError('--tile_stride must be 0 when --tile_size disables tiled predictions')
        return []

    configs: List[TileConfig] = []
    seen: set[str] = set()
    for tile_size, tile_stride in zip(tile_sizes, tile_strides):
        if int(tile_size) <= 0:
            raise ValueError('--tile_size values must be > 0 when dense tiling is active')
        if int(tile_stride) <= 0:
            raise ValueError('--tile_stride values must be > 0 when dense tiling is active')
        if int(tile_stride) > int(tile_size):
            raise ValueError('--tile_stride must be less than or equal to the corresponding --tile_size')
        config_id = f's{int(tile_size)}_st{int(tile_stride)}'
        if config_id in seen:
            raise ValueError(f'Duplicate dense tile configuration: tile_size={int(tile_size)}, tile_stride={int(tile_stride)}')
        seen.add(config_id)
        configs.append(TileConfig(tile_size=int(tile_size), tile_stride=int(tile_stride), config_id=config_id))

    return configs


@dataclass(frozen=True)
class DenseTileJob:
    view: str
    aug_id: str
    config_id: str
    tile_id: str
    tile_x: int
    tile_y: int
    tile_size: int
    tile_stride: int
    out_size: int
    video_path: Path
    meta_path: Path
    M_out_to_src: np.ndarray
    M_src_to_out: np.ndarray


def _center_preserving_scale_matrix(src_w: int, src_h: int, out_w: int, out_h: int) -> np.ndarray:
    cx_src = (int(src_w) - 1) / 2.0
    cy_src = (int(src_h) - 1) / 2.0
    cx_out = (int(out_w) - 1) / 2.0
    cy_out = (int(out_h) - 1) / 2.0
    sx = float(out_w) / float(src_w)
    sy = float(out_h) / float(src_h)
    return np.array(
        [
            [sx, 0.0, cx_out - sx * cx_src],
            [0.0, sy, cy_out - sy * cy_src],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def dense_tile_positions(length: int, tile_size: int, stride: int) -> List[int]:
    length = int(length)
    tile_size = int(tile_size)
    stride = int(stride)
    if tile_size <= 0:
        return []
    if stride <= 0:
        raise ValueError('tile stride must be > 0 when dense tiling is enabled')

    last = max(0, length - tile_size)
    starts = list(range(0, last + 1, stride)) if last > 0 else [0]
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return [int(x) for x in starts]


def build_dense_tile_jobs_for_aug(
    view: ViewInfo,
    aug_job: AugJob,
    tile_cfg: TileConfig,
    out_size: int,
    temp_dir: Path,
) -> List[DenseTileJob]:
    tile_dir = temp_dir / 'tiles' / view.name / tile_cfg.config_id / aug_job.aug_id
    xs = dense_tile_positions(int(aug_job.aff.canvas_w), int(tile_cfg.tile_size), int(tile_cfg.tile_stride))
    ys = dense_tile_positions(int(aug_job.aff.canvas_h), int(tile_cfg.tile_size), int(tile_cfg.tile_stride))
    jobs: List[DenseTileJob] = []

    M_src_to_canvas3 = _affine2x3_to_3x3(aug_job.aff.M_src_to_canvas)
    for tile_y in ys:
        for tile_x in xs:
            tile_id = f'{tile_cfg.config_id}_{aug_job.aug_id}_x{int(tile_x):04d}_y{int(tile_y):04d}'
            M_crop = np.array(
                [
                    [1.0, 0.0, -float(tile_x)],
                    [0.0, 1.0, -float(tile_y)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            M_scale = _center_preserving_scale_matrix(int(tile_cfg.tile_size), int(tile_cfg.tile_size), int(out_size), int(out_size))
            M_src_to_out3 = M_scale @ M_crop @ M_src_to_canvas3
            M_out_to_src3 = np.linalg.inv(M_src_to_out3)
            jobs.append(
                DenseTileJob(
                    view=view.name,
                    aug_id=aug_job.aug_id,
                    config_id=tile_cfg.config_id,
                    tile_id=tile_id,
                    tile_x=int(tile_x),
                    tile_y=int(tile_y),
                    tile_size=int(tile_cfg.tile_size),
                    tile_stride=int(tile_cfg.tile_stride),
                    out_size=int(out_size),
                    video_path=tile_dir / f'{view.name}_{tile_id}.mkv',
                    meta_path=tile_dir / f'{view.name}_{tile_id}.meta.json',
                    M_out_to_src=M_out_to_src3[:2, :3].astype(np.float32),
                    M_src_to_out=M_src_to_out3[:2, :3].astype(np.float32),
                )
            )
    return jobs


def write_dense_tile_job_meta(job: DenseTileJob) -> None:
    job.meta_path.parent.mkdir(parents=True, exist_ok=True)
    job.meta_path.write_text(
        json.dumps(
            {
                'view': job.view,
                'aug_id': job.aug_id,
                'config_id': job.config_id,
                'tile_id': job.tile_id,
                'tile_x': int(job.tile_x),
                'tile_y': int(job.tile_y),
                'tile_size': int(job.tile_size),
                'tile_stride': int(job.tile_stride),
                'out_size': int(job.out_size),
                'M_out_to_src': job.M_out_to_src.tolist(),
                'M_src_to_out': job.M_src_to_out.tolist(),
            },
            indent=2,
        )
    )


def render_view_video_with_affine(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    out_path: Path,
    fps: float,
    M_src_to_out: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    view_frames: Optional[np.ndarray] = None,
    workers: int = 1,
    interpolation: int = cv2.INTER_LINEAR,
    desc: Optional[str] = None,
    show_progress: bool = True,
) -> None:
    proc = ffmpeg_rawvideo_writer(
        out_path,
        width=int(out_w),
        height=int(out_h),
        fps=float(fps),
        pix_fmt_in='rgb24',
        codec='ffv1',
        pix_fmt_out='yuv444p',
    )
    try:
        assert proc.stdin is not None

        def _render(idx: int) -> np.ndarray:
            native_frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(idx), view_frames=view_frames))
            return cv2.warpAffine(
                native_frame,
                M_src_to_out,
                dsize=(int(out_w), int(out_h)),
                flags=int(interpolation),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        pending = min(int(view.num_slices), max(2, int(workers) * 4))
        iterable = parallel_map_in_order(_render, range(int(view.num_slices)), max_workers=max(1, int(workers)), max_pending=pending)
        for frame in tqdm(
            iterable,
            total=int(view.num_slices),
            desc=desc or f'Rendering {out_path.name}',
            disable=not bool(show_progress),
        ):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)


def ensure_dense_tile_video(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: DenseTileJob,
    fps: float,
    workers: int,
    view_frames: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> None:
    if not job.meta_path.exists():
        write_dense_tile_job_meta(job)
    if job.video_path.exists():
        return
    job.video_path.parent.mkdir(parents=True, exist_ok=True)
    render_view_video_with_affine(
        volume_rgb=volume_rgb,
        view=view,
        out_path=job.video_path,
        fps=float(fps),
        M_src_to_out=job.M_src_to_out,
        out_w=int(job.out_size),
        out_h=int(job.out_size),
        view_frames=view_frames,
        workers=max(1, int(workers)),
        interpolation=cv2.INTER_LINEAR,
        desc=f'Dense tile {view.name} {job.tile_id}',
        show_progress=bool(show_progress),
    )


def submit_dense_tile_video_jobs(
    executor: ThreadPoolExecutor,
    volume_rgb: np.ndarray,
    view: ViewInfo,
    tile_jobs: Sequence[DenseTileJob],
    fps: float,
    workers_per_job: int,
    view_frames: Optional[np.ndarray] = None,
) -> Tuple[List[DenseTileJob], Dict[Future, DenseTileJob]]:
    ready_jobs: List[DenseTileJob] = []
    future_to_job: Dict[Future, DenseTileJob] = {}

    for job in tile_jobs:
        if not job.meta_path.exists():
            write_dense_tile_job_meta(job)
        if job.video_path.exists():
            ready_jobs.append(job)
            continue
        fut = executor.submit(
            ensure_dense_tile_video,
            volume_rgb,
            view,
            job,
            float(fps),
            max(1, int(workers_per_job)),
            view_frames,
            False,
        )
        future_to_job[fut] = job

    return ready_jobs, future_to_job


def iter_dense_tile_jobs_in_completion_order(
    ready_jobs: Sequence[DenseTileJob],
    future_to_job: Dict[Future, DenseTileJob],
) -> Iterator[DenseTileJob]:
    for job in ready_jobs:
        yield job
    if not future_to_job:
        return
    for fut in as_completed(list(future_to_job.keys())):
        fut.result()
        yield future_to_job[fut]


def should_cache_view_frames(view: ViewInfo, dense_tiling_active: bool) -> bool:
    """Return True when precomputing native RGB frames is worthwhile for this view.

    Radial view extraction is the dominant CPU cost during dense tiled rendering because every
    tile video currently traverses the same expensive Lanczos-5 radial slices independently.
    Caching the native radial frames once lets both full-frame augmentation and all tile videos
    reuse that work.
    """
    return bool(dense_tiling_active) and view.family == 'radial'


def estimate_view_frame_cache_bytes(view: ViewInfo) -> int:
    return array_nbytes((int(view.num_slices), int(view.src_h), int(view.src_w), 3), np.uint8)


def build_view_frame_cache(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 64 * GIB,
    workers: int = 1,
) -> np.ndarray:
    """Materialize native RGB frames for a view into a reusable cache volume.

    This is used primarily for the radial view when dense tiling is enabled so later tile video
    generation no longer recomputes the same radial slices for every tile location.
    """
    cache_mm = allocate_workspace_array(
        shape=(int(view.num_slices), int(view.src_h), int(view.src_w), 3),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    worker_count = choose_slice_parallel_workers(int(workers), int(view.num_slices))
    print(
        f"Preparing reusable native frame cache for view '{view.name}' over {int(view.num_slices)} slice(s) "
        f"with {int(worker_count)} worker thread(s)"
    )

    def _build(idx: int) -> None:
        cache_mm[int(idx), :, :, :] = np.ascontiguousarray(
            get_view_frame_by_index(volume_rgb, view, int(idx), view_frames=None)
        )

    parallel_for_indices(
        int(view.num_slices),
        _build,
        max_workers=worker_count,
        desc=f'Caching {view.name} native frames',
        show_progress=False,
    )
    flush_array(cache_mm)
    return cache_mm


def iter_view_frames(
    volume_rgb: np.memmap,
    view: ViewInfo,
    view_frames: Optional[np.ndarray] = None,
) -> Iterator[np.ndarray]:
    """Yield frames for a view, in slice order (0..num_slices-1)."""
    if view_frames is not None:
        for idx in range(int(view.num_slices)):
            yield np.asarray(view_frames[int(idx)])
        return

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


def get_view_frame_by_index(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    index: int,
    view_frames: Optional[np.ndarray] = None,
) -> np.ndarray:
    if view_frames is not None:
        return np.asarray(view_frames[int(index)])

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


def _render_augmented_bundle_for_index(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    jobs: Sequence[AugJob],
    index: int,
    view_frames: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    native_frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(index), view_frames=view_frames))
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


def _write_augmented_video_from_stage(
    job: AugJob,
    staged_frames: np.ndarray,
    ready_flags: np.ndarray,
    ready_cond: threading.Condition,
    stop_event: threading.Event,
    fps: float,
    num_slices: int,
) -> None:
    proc: Optional[subprocess.Popen] = None
    caught: Optional[BaseException] = None

    try:
        proc = ffmpeg_rawvideo_writer(
            job.video_path,
            width=int(job.aff.out_size),
            height=int(job.aff.out_size),
            fps=fps,
            pix_fmt_in='rgb24',
            codec='ffv1',
            pix_fmt_out='yuv444p',
        )
        assert proc.stdin is not None

        for idx in range(int(num_slices)):
            with ready_cond:
                while not bool(ready_flags[idx]) and not stop_event.is_set():
                    ready_cond.wait()
                if stop_event.is_set() and not bool(ready_flags[idx]):
                    return
            proc.stdin.write(np.ascontiguousarray(staged_frames[idx]).tobytes())
    except BaseException as exc:  # pragma: no cover - propagated through the future result path
        caught = exc
    finally:
        if proc is not None:
            try:
                close_ffmpeg_writer(proc)
            except BaseException as exc:  # pragma: no cover - propagated through the future result path
                if caught is None:
                    caught = exc
        if caught is not None:
            stop_event.set()
            with ready_cond:
                ready_cond.notify_all()
            raise caught



def _ensure_augmented_videos_with_staging(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    missing_jobs: Sequence[AugJob],
    fps: float,
    worker_count: int,
    view_frames: Optional[np.ndarray] = None,
) -> bool:
    if not missing_jobs:
        return True

    out_size = int(missing_jobs[0].aff.out_size)
    stage_bytes = estimate_augmented_view_staging_bytes(view.num_slices, out_size, len(missing_jobs))
    budget = augmentation_staging_budget_summary(stage_bytes)
    if not should_use_in_memory_augmentation_staging(stage_bytes):
        print(f'Augmentation staging disabled for {view.name}: {budget}')
        return False

    print(f'Augmentation staging for {view.name}: in-memory ({budget})')
    staged_frames: Dict[str, np.ndarray] = {}
    try:
        for job in missing_jobs:
            staged_frames[job.aug_id] = np.empty((int(view.num_slices), out_size, out_size, 3), dtype=np.uint8)
    except MemoryError:
        staged_frames.clear()
        gc.collect()
        print(f'Augmentation staging allocation failed for {view.name}; falling back to streaming writers')
        return False

    ready_flags = np.zeros((int(view.num_slices),), dtype=np.uint8)
    ready_cond = threading.Condition()
    stop_event = threading.Event()
    writer_workers = choose_augmentation_writer_workers(len(missing_jobs))

    for job in missing_jobs:
        job.video_path.parent.mkdir(parents=True, exist_ok=True)

    def render_index(idx: int) -> None:
        if stop_event.is_set():
            return
        bundle = _render_augmented_bundle_for_index(volume_rgb, view, missing_jobs, int(idx), view_frames=view_frames)
        for job, out in zip(missing_jobs, bundle):
            staged_frames[job.aug_id][int(idx), :, :, :] = np.ascontiguousarray(out)
        with ready_cond:
            ready_flags[int(idx)] = 1
            ready_cond.notify_all()

    try:
        with ThreadPoolExecutor(max_workers=writer_workers) as writer_pool:
            writer_futures = [
                writer_pool.submit(
                    _write_augmented_video_from_stage,
                    job,
                    staged_frames[job.aug_id],
                    ready_flags,
                    ready_cond,
                    stop_event,
                    fps,
                    int(view.num_slices),
                )
                for job in missing_jobs
            ]

            try:
                parallel_for_indices(
                    int(view.num_slices),
                    render_index,
                    max_workers=worker_count,
                    desc=f'Augment {view.name} -> {len(missing_jobs)} video(s) [staged]',
                )
            except BaseException:
                stop_event.set()
                with ready_cond:
                    ready_cond.notify_all()
                raise
            finally:
                with ready_cond:
                    ready_cond.notify_all()

            for fut in writer_futures:
                fut.result()
        return True
    finally:
        staged_frames.clear()
        gc.collect()



def _ensure_augmented_videos_streaming(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    missing_jobs: Sequence[AugJob],
    fps: float,
    worker_count: int,
    view_frames: Optional[np.ndarray] = None,
) -> None:
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

        render = lambda idx: _render_augmented_bundle_for_index(volume_rgb, view, missing_jobs, idx, view_frames=view_frames)
        pending = min(int(view.num_slices), max(worker_count + 1, worker_count * 8))
        for bundle in tqdm(
            parallel_map_in_order(render, range(view.num_slices), max_workers=worker_count, max_pending=pending),
            total=view.num_slices,
            desc=f'Augment {view.name} -> {len(missing_jobs)} video(s) [streaming]',
        ):
            for job, out in zip(missing_jobs, bundle):
                writer = writers[job.aug_id]
                assert writer.stdin is not None
                writer.stdin.write(np.ascontiguousarray(out).tobytes())
    finally:
        for writer in writers.values():
            close_ffmpeg_writer(writer)



def ensure_augmented_videos(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    aug_jobs: Sequence[AugJob],
    fps: float,
    augmentation_workers: int,
    view_frames: Optional[np.ndarray] = None,
) -> None:
    missing_jobs = [job for job in aug_jobs if not job.video_path.exists()]
    for job in aug_jobs:
        if not job.meta_path.exists():
            write_aug_job_meta(job, view)

    if not missing_jobs:
        return

    worker_count = choose_slice_parallel_workers(int(augmentation_workers), int(view.num_slices))
    mode_suffix = ''
    if view.family == 'radial':
        mode_suffix = ' [OpenCV remap fast path]' if radial_fast_path_enabled() else ' [exact Lanczos-5]'
    print(
        f"Generating {len(missing_jobs)} augmented {view.name} video(s) over {view.num_slices} slice(s) "
        f"with {worker_count} worker thread(s){mode_suffix}"
    )
    if view_frames is not None:
        print(f"Augmented {view.name} rendering will reuse the native frame cache")

    if _ensure_augmented_videos_with_staging(
        volume_rgb=volume_rgb,
        view=view,
        missing_jobs=missing_jobs,
        fps=fps,
        worker_count=worker_count,
        view_frames=view_frames,
    ):
        return

    _ensure_augmented_videos_streaming(
        volume_rgb=volume_rgb,
        view=view,
        missing_jobs=missing_jobs,
        fps=fps,
        worker_count=worker_count,
        view_frames=view_frames,
    )


@dataclass
class ViewPrefetchResult:
    view_frames: Optional[np.ndarray]
    view_frame_cache_path: Optional[Path]


def prepare_view_render_assets(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    aug_jobs: Sequence[AugJob],
    fps: float,
    augmentation_workers: int,
    dense_tiling_active: bool,
    temp_dir: Path,
) -> ViewPrefetchResult:
    """Prefetch reusable render assets for a view.

    This runs ahead of the main view-processing loop so expensive view construction can overlap
    across active views within the global CPU budget.
    """
    view_frames: Optional[np.ndarray] = None
    cache_path: Optional[Path] = None

    print(f"Prefetching render assets for view '{view.name}'")

    if should_cache_view_frames(view, dense_tiling_active):
        cache_path = temp_dir / 'view_frames' / f'{view.name}.rgb24.dat'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        view_frames = build_view_frame_cache(
            volume_rgb=volume_rgb,
            view=view,
            out_path=cache_path,
            desc=f'{view.name} native frame cache',
            prefer_memory=True,
            workers=max(1, int(augmentation_workers)),
        )

    ensure_augmented_videos(
        volume_rgb=volume_rgb,
        view=view,
        aug_jobs=aug_jobs,
        fps=float(fps),
        augmentation_workers=max(1, int(augmentation_workers)),
        view_frames=view_frames,
    )
    return ViewPrefetchResult(view_frames=view_frames, view_frame_cache_path=cache_path)


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


def _extract_result_masks_and_confs(r) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Detach one streamed YOLO result into CPU-owned numpy arrays for asynchronous postprocess."""
    if getattr(r, 'masks', None) is None or r.masks is None or r.masks.data is None:
        return None, None

    masks_data = r.masks.data  # (n,h,w)
    try:
        masks_np = np.asarray(masks_data.detach().cpu().numpy(), dtype=np.uint8)
    except Exception:
        try:
            masks_np = np.asarray(masks_data.cpu().numpy(), dtype=np.uint8)
        except Exception:
            masks_np = np.asarray(masks_data, dtype=np.uint8)

    if masks_np.ndim != 3 or int(masks_np.shape[0]) <= 0:
        return None, None

    num_inst = int(masks_np.shape[0])
    if getattr(r, 'boxes', None) is not None and r.boxes is not None and getattr(r.boxes, 'conf', None) is not None:
        try:
            confs_np = np.asarray(r.boxes.conf.detach().cpu().numpy(), dtype=np.float32)
        except Exception:
            try:
                confs_np = np.asarray(r.boxes.conf.cpu().numpy(), dtype=np.float32)
            except Exception:
                confs_np = np.asarray(r.boxes.conf, dtype=np.float32)
    else:
        confs_np = np.zeros((num_inst,), dtype=np.float32)

    if confs_np.ndim == 0:
        confs_np = np.full((num_inst,), float(confs_np), dtype=np.float32)
    elif int(confs_np.shape[0]) != num_inst:
        confs_np = np.resize(confs_np, (num_inst,)).astype(np.float32, copy=False)

    return np.ascontiguousarray(masks_np), np.ascontiguousarray(confs_np)


def _process_prediction_frame(
    idx: int,
    masks_np: Optional[np.ndarray],
    confs_np: Optional[np.ndarray],
    out_size: int,
    view_union_mm: np.ndarray,
    view_confmap_mm: np.ndarray,
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
) -> Tuple[int, int]:
    """Collapse one streamed result directly into the native-view union + confidence accumulators.

    Key optimization:
      - build the augmented-space union mask once
      - build the augmented-space per-pixel max-confidence map once
      - inverse-warp each of those exactly once

    This preserves the required semantics of unioning masks and retaining the highest confidence
    score per covered pixel, while avoiding one native warp per detected instance and avoiding
    per-augmentation trace files on disk.
    """
    if masks_np is None or confs_np is None or masks_np.ndim != 3 or int(masks_np.shape[0]) <= 0:
        return 0, 0

    frame_union = np.zeros((out_size, out_size), dtype=np.uint8)
    frame_confmap = np.zeros((out_size, out_size), dtype=np.float32)
    frame_max_conf = 0.0
    num_inst = int(masks_np.shape[0])

    for inst_idx in range(num_inst):
        inst = np.asarray(masks_np[inst_idx], dtype=np.uint8)
        if inst.shape[0] != out_size or inst.shape[1] != out_size:
            inst = cv2.resize(inst, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        inst = (inst > 0).astype(np.uint8, copy=False)
        if not np.any(inst):
            continue

        conf_val = float(confs_np[inst_idx]) if inst_idx < int(confs_np.shape[0]) else 0.0
        frame_union |= inst
        if conf_val > frame_max_conf:
            frame_max_conf = conf_val
        inst_bool = inst > 0
        frame_confmap[inst_bool] = np.maximum(frame_confmap[inst_bool], np.float32(conf_val))

    if not np.any(frame_union):
        return int(num_inst), 1

    native_union = cv2.warpAffine(
        frame_union,
        M_out_to_native,
        dsize=(native_w, native_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if np.any(native_union):
        view_union_mm[idx, :] |= pack_mask(native_union)

    if frame_max_conf > 0.0:
        native_conf = cv2.warpAffine(
            frame_confmap,
            M_out_to_native,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        if np.any(native_conf > 0.0):
            conf_slice = view_confmap_mm[idx]
            native_conf_f16 = native_conf.astype(np.float16, copy=False)
            np.maximum(conf_slice, native_conf_f16, out=conf_slice)

    return int(num_inst), 1


def predict_video_and_accumulate(
    model,
    video_path: Path,
    num_frames: int,
    out_size: int,
    pred_out_prefix: Path,
    cfg: PredictConfig,
    # accumulation into per-view union stack (native resolution, packed bits)
    view_union_mm: np.ndarray,          # uint8 packbits, shape (num_slices, bytes_native)
    view_confmap_mm: np.ndarray,        # float16 confidence map, shape (num_slices, native_h, native_w)
    M_out_to_native: np.ndarray,       # 2x3, maps augmented(out)->native for cv2.warpAffine (src->dst)
    native_h: int,
    native_w: int,
    postprocess_workers: int = 1,
) -> Dict[str, int]:
    """
    Run YOLO predict(stream=True) on a pre-generated augmented video and accumulate the inverse-
    transformed native masks in memory-backed workspaces.

    The sequential portion remains the YOLO inference stream itself. CPU-side result handling is
    overlapped with that stream when ``postprocess_workers > 1`` so native reorientation and
    confidence-map updates do not unnecessarily serialize GPU inference.
    """
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

    worker_count = max(1, min(int(postprocess_workers), int(num_frames)))
    pending_limit = max(worker_count, worker_count * 2)

    if worker_count <= 1:
        for idx, r in enumerate(results):
            if idx >= num_frames:
                break
            masks_np, confs_np = _extract_result_masks_and_confs(r)
            pred_inc, frame_inc = _process_prediction_frame(
                idx=idx,
                masks_np=masks_np,
                confs_np=confs_np,
                out_size=out_size,
                view_union_mm=view_union_mm,
                view_confmap_mm=view_confmap_mm,
                M_out_to_native=M_out_to_native,
                native_h=native_h,
                native_w=native_w,
            )
            prediction_count += int(pred_inc)
            frames_with_predictions += int(frame_inc)
    else:
        pending: List[object] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for idx, r in enumerate(results):
                if idx >= num_frames:
                    break

                masks_np, confs_np = _extract_result_masks_and_confs(r)
                pending.append(executor.submit(
                    _process_prediction_frame,
                    idx,
                    masks_np,
                    confs_np,
                    out_size,
                    view_union_mm,
                    view_confmap_mm,
                    M_out_to_native,
                    native_h,
                    native_w,
                ))
                if len(pending) >= pending_limit:
                    fut = pending.pop(0)
                    pred_inc, frame_inc = fut.result()
                    prediction_count += int(pred_inc)
                    frames_with_predictions += int(frame_inc)

            while pending:
                fut = pending.pop(0)
                pred_inc, frame_inc = fut.result()
                prediction_count += int(pred_inc)
                frames_with_predictions += int(frame_inc)

    flush_array(view_confmap_mm)
    flush_array(view_union_mm)

    return {
        'prediction_count': int(prediction_count),
        'frames_with_predictions': int(frames_with_predictions),
    }


def _process_tile_prediction_frame(
    idx: int,
    masks_np: Optional[np.ndarray],
    confs_np: Optional[np.ndarray],
    out_size: int,
    baseline_support_mm: np.ndarray,
    baseline_mask_mm: np.ndarray,
    baseline_confmap_mm: np.ndarray,
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    min_conf: float,
) -> Dict[str, int]:
    """Gate one tiled frame directly against the frozen parent full-frame slice.

    v7.8.0 requires tiled masks to be compared against the parent full-frame slice without
    consolidating tiles across locations first. We therefore keep tiled masks instance-wise,
    inverse-map each mask to the parent native view, test it against the frozen full-frame support
    slice, and only then union accepted masks into the mutable parent view volume.
    """
    stats = {
        'prediction_count': 0,
        'frames_with_predictions': 0,
        'accepted_masks': 0,
        'rejected_masks': 0,
        'rejected_min_conf': 0,
        'gated_added_voxels': 0,
    }

    if masks_np is None or confs_np is None or masks_np.ndim != 3 or int(masks_np.shape[0]) <= 0:
        return stats

    support_slice = np.asarray(baseline_support_mm[int(idx)], dtype=np.uint8)
    parent_slice = np.asarray(baseline_mask_mm[int(idx)], dtype=np.uint8)
    conf_slice = baseline_confmap_mm[int(idx)]

    num_inst = int(masks_np.shape[0])
    stats['prediction_count'] = int(num_inst)
    stats['frames_with_predictions'] = 1

    for inst_idx in range(num_inst):
        conf_val = float(confs_np[inst_idx]) if inst_idx < int(confs_np.shape[0]) else 0.0
        if float(min_conf) > 0.0 and conf_val < float(min_conf):
            stats['rejected_min_conf'] += 1
            continue

        inst = np.asarray(masks_np[inst_idx], dtype=np.uint8)
        if inst.shape[0] != out_size or inst.shape[1] != out_size:
            inst = cv2.resize(inst, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        inst = (inst > 0).astype(np.uint8, copy=False)
        if not np.any(inst):
            stats['rejected_masks'] += 1
            continue

        native_mask = cv2.warpAffine(
            inst,
            M_out_to_native,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        native_mask_bool = np.asarray(native_mask, dtype=bool)
        if not np.any(native_mask_bool):
            stats['rejected_masks'] += 1
            continue

        if not np.any(native_mask_bool & (support_slice > 0)):
            stats['rejected_masks'] += 1
            continue

        added = native_mask_bool & (parent_slice == 0)
        if np.any(added):
            stats['gated_added_voxels'] += int(np.count_nonzero(added))
        parent_slice[native_mask_bool] = np.uint8(1)
        if conf_val > 0.0:
            conf_val_f16 = np.float16(conf_val)
            conf_slice[native_mask_bool] = np.maximum(conf_slice[native_mask_bool], conf_val_f16)
        stats['accepted_masks'] += 1

    return stats


def predict_tile_video_and_gate(
    model,
    video_path: Path,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    baseline_support_mm: np.ndarray,
    baseline_mask_mm: np.ndarray,
    baseline_confmap_mm: np.ndarray,
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    min_conf: float,
    postprocess_workers: int = 1,
) -> Dict[str, int]:
    """Run tiled inference and gate accepted masks directly into the parent full-frame volume."""
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

    agg = {
        'prediction_count': 0,
        'frames_with_predictions': 0,
        'accepted_masks': 0,
        'rejected_masks': 0,
        'rejected_min_conf': 0,
        'gated_added_voxels': 0,
    }

    worker_count = max(1, min(int(postprocess_workers), int(num_frames)))
    pending_limit = max(worker_count, worker_count * 2)

    def _accumulate(stats_local: Dict[str, int]) -> None:
        for key in agg.keys():
            agg[key] += int(stats_local.get(key, 0))

    if worker_count <= 1:
        for idx, r in enumerate(results):
            if idx >= num_frames:
                break
            masks_np, confs_np = _extract_result_masks_and_confs(r)
            _accumulate(_process_tile_prediction_frame(
                idx=idx,
                masks_np=masks_np,
                confs_np=confs_np,
                out_size=out_size,
                baseline_support_mm=baseline_support_mm,
                baseline_mask_mm=baseline_mask_mm,
                baseline_confmap_mm=baseline_confmap_mm,
                M_out_to_native=M_out_to_native,
                native_h=native_h,
                native_w=native_w,
                min_conf=float(min_conf),
            ))
    else:
        pending: List[object] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for idx, r in enumerate(results):
                if idx >= num_frames:
                    break

                masks_np, confs_np = _extract_result_masks_and_confs(r)
                pending.append(executor.submit(
                    _process_tile_prediction_frame,
                    idx,
                    masks_np,
                    confs_np,
                    out_size,
                    baseline_support_mm,
                    baseline_mask_mm,
                    baseline_confmap_mm,
                    M_out_to_native,
                    native_h,
                    native_w,
                    float(min_conf),
                ))
                if len(pending) >= pending_limit:
                    _accumulate(pending.pop(0).result())

            while pending:
                _accumulate(pending.pop(0).result())

    flush_array(baseline_mask_mm)
    flush_array(baseline_confmap_mm)

    return {k: int(v) for k, v in agg.items()}


# --------------------------
# Per-view postprocessing
# --------------------------

def apply_min_conf_filter_with_confmap_inplace(
    union_mm: np.ndarray,
    confmap_mm: np.ndarray,
    min_conf: float,
    h: int,
    w: int,
    *,
    workers: int = 1,
) -> None:
    """Remove native 2D connected components whose max confidence is below ``min_conf``.

    This keeps the highest confidence score for each connected combined object after unioning overlapping masks,
    score of the combined mask, instead of using a single slice-wide score.
    """
    n = int(union_mm.shape[0])
    structure2 = np.ones((3, 3), dtype=bool)

    def _process(i: int) -> None:
        packed = np.asarray(union_mm[int(i)])
        if not any_mask(packed):
            confmap_mm[int(i), :, :] = np.float16(0.0)
            return

        union = unpack_mask(packed, h, w).astype(bool, copy=False)
        labels2d, num = ndi.label(union, structure=structure2)
        if int(num) <= 0:
            union_mm[int(i), :] = 0
            confmap_mm[int(i), :, :] = np.float16(0.0)
            return

        conf_slice = np.asarray(confmap_mm[int(i)], dtype=np.float32)
        label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
        maxima = ndi.maximum(conf_slice, labels=labels2d, index=label_ids)
        maxima = np.asarray(maxima, dtype=np.float32)
        keep_ids = label_ids[maxima >= float(min_conf)]

        if keep_ids.size == 0:
            union_mm[int(i), :] = 0
            confmap_mm[int(i), :, :] = np.float16(0.0)
            return

        keep = np.isin(labels2d, keep_ids)
        union_mm[int(i), :] = pack_mask(keep.astype(np.uint8, copy=False))
        conf_slice[~keep] = 0.0
        confmap_mm[int(i), :, :] = conf_slice.astype(np.float16, copy=False)

    parallel_for_indices(
        n,
        _process,
        max_workers=choose_slice_parallel_workers(int(workers), n),
        desc='min_conf thresholding',
    )


def fill_2d_holes_volume_inplace(
    mask_mm: np.ndarray,
    *,
    workers: int = 1,
) -> None:
    num_slices = int(mask_mm.shape[0])

    def _process(i: int) -> None:
        sl = np.asarray(mask_mm[int(i)], dtype=np.uint8)
        if not np.any(sl):
            return

        h, w = sl.shape
        pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
        inv = np.empty_like(pad)
        flood = np.empty_like(pad)
        ffmask = np.zeros((h + 4, w + 4), dtype=np.uint8)
        holes = np.empty(pad.shape, dtype=np.bool_)

        pad[1:-1, 1:-1] = sl
        inv[:] = 1
        inv -= pad

        flood[:] = inv
        ffmask.fill(0)
        cv2.floodFill(flood, ffmask, seedPoint=(0, 0), newVal=2)

        np.equal(flood, 1, out=holes)
        if holes.any():
            pad[holes] = 1

        mask_mm[int(i), :, :] = pad[1:-1, 1:-1]

    parallel_for_indices(
        num_slices,
        _process,
        max_workers=choose_slice_parallel_workers(int(workers), num_slices),
        desc='2D hole fill',
    )
    flush_array(mask_mm)


def objectwise_gated_or_inplace(
    baseline_mask_mm: np.ndarray,
    baseline_confmap_mm: np.ndarray,
    tiled_mask_mm: np.ndarray,
    tiled_confmap_mm: np.ndarray,
    work_dir: Path,
    *,
    baseline_labels_mm: Optional[np.ndarray] = None,
    baseline_objects: Optional[int] = None,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> Dict[str, int]:
    created_baseline_labels = False
    created_baseline_paths: List[Path] = []

    if baseline_labels_mm is None or baseline_objects is None:
        if not np.any(np.asarray(baseline_mask_mm)):
            baseline_objects = 0
            baseline_labels_mm = None
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            baseline_labels_mm, baseline_objects, created_baseline_paths = label_foreground_volume_streaming(
                baseline_mask_mm,
                work_dir / 'baseline',
                keep_temp=True,
                prefer_memory=bool(prefer_memory),
                reserve_bytes=int(reserve_bytes),
            )
            created_baseline_labels = True

    if not np.any(np.asarray(tiled_mask_mm)):
        if created_baseline_labels and baseline_labels_mm is not None:
            close_memmap_array(baseline_labels_mm)
        if not keep_temp:
            for p in created_baseline_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        return {
            'baseline_objects': int(baseline_objects or 0),
            'tiled_objects': 0,
            'accepted_tiled_objects': 0,
            'rejected_tiled_objects': 0,
            'gated_added_voxels': 0,
        }

    work_dir.mkdir(parents=True, exist_ok=True)
    tiled_labels, tiled_objects, tiled_paths = label_foreground_volume_streaming(
        tiled_mask_mm,
        work_dir / 'tiled',
        keep_temp=True,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    accepted_tiled = np.zeros((int(tiled_objects) + 1,), dtype=bool)
    if int(tiled_objects) > 0 and int(baseline_objects or 0) > 0 and baseline_labels_mm is not None:
        for z in tqdm(range(int(tiled_mask_mm.shape[0])), desc='Dense tiled gating: overlap scan'):
            tile_lbl = np.asarray(tiled_labels[int(z)])
            base_lbl = np.asarray(baseline_labels_mm[int(z)])
            overlap = (tile_lbl > 0) & (base_lbl > 0)
            if not np.any(overlap):
                continue
            accepted_tiled[np.unique(tile_lbl[overlap])] = True
        accepted_tiled[0] = False

    added_counts = np.zeros((int(tiled_mask_mm.shape[0]),), dtype=np.int64)
    for z in tqdm(range(int(tiled_mask_mm.shape[0])), desc='Dense tiled gating: apply accepted objects'):
        tile_lbl = np.asarray(tiled_labels[int(z)])
        accept_mask = accepted_tiled[tile_lbl]
        if not np.any(accept_mask):
            continue

        baseline_slice = baseline_mask_mm[int(z)]
        before = np.asarray(baseline_slice, dtype=bool)
        after = before | accept_mask
        baseline_mask_mm[int(z), :, :] = after.astype(np.uint8, copy=False)
        np.maximum(
            np.asarray(baseline_confmap_mm[int(z)], dtype=np.float16),
            np.where(accept_mask, np.asarray(tiled_confmap_mm[int(z)], dtype=np.float16), np.float16(0.0)),
            out=baseline_confmap_mm[int(z)],
        )
        added_counts[int(z)] = np.int64(np.count_nonzero(after & (~before)))

    stats = {
        'baseline_objects': int(baseline_objects or 0),
        'tiled_objects': int(tiled_objects),
        'accepted_tiled_objects': int(np.count_nonzero(accepted_tiled[1:])),
        'rejected_tiled_objects': int(max(0, int(tiled_objects) - int(np.count_nonzero(accepted_tiled[1:])))),
        'gated_added_voxels': int(np.sum(added_counts, dtype=np.int64)),
    }

    flush_array(baseline_mask_mm)
    flush_array(baseline_confmap_mm)
    if created_baseline_labels and baseline_labels_mm is not None:
        close_memmap_array(baseline_labels_mm)
    close_memmap_array(tiled_labels)
    if not keep_temp:
        for p in created_baseline_paths + tiled_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    return stats

# --------------------------
# 3D assembly + postprocessing
# --------------------------


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



def build_slice_endpoint_seeds_from_label_volume(
    labels_real: np.ndarray,
    workers: int = 1,
    desc: str = 'Interpolation: endpoint seeds [scan]',
) -> Tuple[List[SliceEndpointSeed], int]:
    """Fast slice-graph endpoint scan for slice-direction interpolation.

    This avoids per-object 3D voxel skeletonization on large relabeled volumes, which can become
    prohibitively slow when an object's bounding box spans a large fraction of the volume. Endpoints
    are identified from connected components in each slice that do not continue into the previous or
    next slice of the same relabeled object.
    """
    z_dim = int(labels_real.shape[0])
    if z_dim <= 0:
        return [], 0

    kernel2 = np.ones((3, 3), dtype=np.uint8)

    def _scan_slice(z: int) -> List[SliceEndpointSeed]:
        curr_slice = np.asarray(labels_real[int(z)])
        if not np.any(curr_slice):
            return []

        prev_slice = np.asarray(labels_real[int(z) - 1]) if int(z) > 0 else None
        next_slice = np.asarray(labels_real[int(z) + 1]) if (int(z) + 1) < z_dim else None

        seeds_local: List[SliceEndpointSeed] = []
        present = np.unique(curr_slice)
        present = present[present > 0]
        if present.size == 0:
            return seeds_local

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
                    seeds_local.append(SliceEndpointSeed(
                        label=int(obj_id),
                        point=(int(z), int(anchor[0]), int(anchor[1])),
                        direction_sign=-1,
                    ))
                if not has_next:
                    seeds_local.append(SliceEndpointSeed(
                        label=int(obj_id),
                        point=(int(z), int(anchor[0]), int(anchor[1])),
                        direction_sign=1,
                    ))

        return seeds_local

    worker_count = choose_slice_parallel_workers(int(workers), z_dim)
    seeds: List[SliceEndpointSeed] = []

    if worker_count <= 1:
        for z in tqdm(range(z_dim), desc=desc):
            seeds.extend(_scan_slice(int(z)))
    else:
        pending = max(worker_count, worker_count * 2)
        for seeds_local in tqdm(
            parallel_map_in_order(_scan_slice, range(z_dim), max_workers=worker_count, max_pending=pending),
            total=z_dim,
            desc=desc,
        ):
            seeds.extend(seeds_local)

    seeds.sort(key=lambda s: (int(s.label), int(s.point[0]), int(s.direction_sign), int(s.point[1]), int(s.point[2])))
    return seeds, int(len(seeds))


def apply_transverse_min_radius_filter_inplace(
    mask_mm: np.ndarray,
    min_radius: float,
    *,
    workers: int = 1,
) -> None:
    """In-place transverse-plane radius filter to avoid a full extra volume copy."""
    if float(min_radius) <= 0:
        return

    struct2 = np.ones((3, 3), dtype=bool)
    num_slices = int(mask_mm.shape[0])

    def _process(t: int) -> None:
        sl = np.asarray(mask_mm[int(t)]) > 0
        if not np.any(sl):
            return

        labels2d, num = ndi.label(sl, structure=struct2)
        if int(num) <= 0:
            mask_mm[int(t), :, :] = 0
            return

        keep = np.zeros(sl.shape, dtype=bool)
        for lbl in range(1, int(num) + 1):
            comp = labels2d == lbl
            if not np.any(comp):
                continue
            radius = float(np.max(ndi.distance_transform_edt(comp)))
            if radius >= float(min_radius):
                keep |= comp

        mask_mm[int(t), :, :] = keep.astype(np.uint8, copy=False)

    parallel_for_indices(
        num_slices,
        _process,
        max_workers=choose_slice_parallel_workers(int(workers), num_slices),
        desc='Transverse min-radius filter',
    )
    flush_array(mask_mm)


def apply_view_min_radius_filter_inplace(
    mask_mm: np.ndarray,
    view: ViewInfo,
    min_radius: float,
    *,
    workers: int = 1,
) -> None:
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
    apply_transverse_min_radius_filter_inplace(
        transverse_view,
        float(min_radius),
        workers=choose_slice_parallel_workers(int(workers), int(transverse_view.shape[0])),
    )
    flush_array(mask_mm)


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
    workers: int = 1,
) -> np.ndarray:
    vol_mm = allocate_workspace_array(
        shape=(num_slices, h, w),
        dtype=np.uint8,
        path=out_path,
        desc=f'{desc} workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    def _process(i: int) -> None:
        packed = np.asarray(union_mm[int(i)])
        if any_mask(packed):
            vol_mm[int(i), :, :] = unpack_mask(packed, h, w)
        else:
            vol_mm[int(i), :, :] = 0

    parallel_for_indices(
        int(num_slices),
        _process,
        max_workers=choose_slice_parallel_workers(int(workers), int(num_slices)),
        desc=desc,
    )
    flush_array(vol_mm)
    return vol_mm


def backproject_radial_volume_to_volume(
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    if radial_view.family != 'radial':
        raise ValueError('backproject_radial_volume_to_volume expects a radial view')

    t_dim = int(radial_view.src_h)
    out_h = int(radial_view.full_h)
    out_w = int(radial_view.full_w)

    vol_mm = allocate_workspace_array(
        shape=(t_dim, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=f'{desc} workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    for angle_idx, angle_deg in enumerate(tqdm(radial_view.azimuths_deg, desc=desc)):
        radial_mask = np.asarray(radial_mask_mm[int(angle_idx)], dtype=bool)
        if not np.any(radial_mask):
            continue
        tt, uu = np.nonzero(radial_mask)
        if tt.size == 0:
            continue
        sampler = get_radial_sampler(radial_view, float(angle_deg))
        vol_mm[tt, sampler.nn_y[uu], sampler.nn_x[uu]] = 1

    flush_array(vol_mm)
    return vol_mm


def assemble_model_volume_from_view_volumes(
    ensemble_mm: np.ndarray,
    view_volume_mms: Dict[str, np.ndarray],
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
    *,
    workers: int = 1,
) -> None:
    transverse = np.asarray(view_volume_mms["transverse"])
    assert transverse.shape == (T, H, W)

    transverse_workers = choose_slice_parallel_workers(int(workers), int(T))

    def _merge_transverse(t: int) -> None:
        ensemble_mm[int(t), :, :] |= transverse[int(t), :, :]

    parallel_for_indices(
        int(T),
        _merge_transverse,
        max_workers=transverse_workers,
        desc="Assembling volume from transverse view volume",
    )

    if not disable_multiplanar and "sagittal" in view_volume_mms:
        sagittal = np.asarray(view_volume_mms["sagittal"])
        assert sagittal.shape == (H, T, W)
        sagittal_workers = choose_slice_parallel_workers(int(workers), int(H))

        def _merge_sagittal(y: int) -> None:
            ensemble_mm[:, int(y), :] |= sagittal[int(y), :, :]

        parallel_for_indices(
            int(H),
            _merge_sagittal,
            max_workers=sagittal_workers,
            desc="Assembling volume from sagittal view volume",
        )

    if not disable_multiplanar and "coronal" in view_volume_mms:
        coronal = np.asarray(view_volume_mms["coronal"])
        assert coronal.shape == (W, T, H)
        coronal_workers = choose_slice_parallel_workers(int(workers), int(W))

        def _merge_coronal(x: int) -> None:
            ensemble_mm[:, :, int(x)] |= coronal[int(x), :, :]

        parallel_for_indices(
            int(W),
            _merge_coronal,
            max_workers=coronal_workers,
            desc="Assembling volume from coronal view volume",
        )

    if "radial" in view_volume_mms:
        radial = np.asarray(view_volume_mms["radial"])
        assert radial.shape == (T, H, W)
        radial_workers = choose_slice_parallel_workers(int(workers), int(T))

        def _merge_radial(t: int) -> None:
            ensemble_mm[int(t), :, :] |= radial[int(t), :, :]

        parallel_for_indices(
            int(T),
            _merge_radial,
            max_workers=radial_workers,
            desc="Assembling volume from radial view volume",
        )


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
    workers: int = 1,
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
            workers=int(workers),
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


_SPHERE_PAINT_CACHE: Dict[int, np.ndarray] = {}


_PLANE_GRID_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


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


def _signed_distance_2d(mask2d: np.ndarray) -> np.ndarray:
    mask2d = np.asarray(mask2d, dtype=bool)
    inside = ndi.distance_transform_edt(mask2d)
    outside = ndi.distance_transform_edt(~mask2d)
    return np.asarray(inside - outside, dtype=np.float32)


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


@dataclass(frozen=True)
class SliceBridgeRenderPlan:
    source_label: int
    target_label: int
    source_point: Tuple[int, int, int]
    target_point: Tuple[int, int, int]
    source_anchor: Tuple[int, int]
    target_anchor: Tuple[int, int]
    steps: int
    sign: int
    sdf0: np.ndarray
    sdf1: np.ndarray


@dataclass
class SliceSeedBridgePlanResult:
    candidate_connections: int = 0
    accepted_connections: int = 0
    default_bridges: int = 0
    walk_back_bridges: int = 0
    skipped_by_min_radius: int = 0
    plans: List[SliceBridgeRenderPlan] = field(default_factory=list)


def _build_slice_endpoint_seeds_for_label(
    labels_real: np.ndarray,
    lbl: int,
    sl: Tuple[slice, slice, slice],
    direction_depth: int,
) -> List[SliceEndpointSeed]:
    sub = labels_real[sl] == int(lbl)
    if not np.any(sub):
        return []

    grouped: Dict[Tuple[int, int, int, int, int], SliceEndpointSeed] = {}
    slice_start = int(sl[0].start)
    slice_stop = int(sl[0].stop) - 1
    structure2 = np.ones((3, 3), dtype=bool)
    slice_cache: Dict[int, Tuple[np.ndarray, Dict[int, Tuple[int, int]]]] = {}

    def _slice_centroid(local_slice_idx: int, local_y: int, local_x: int) -> Optional[Tuple[int, int]]:
        entry = slice_cache.get(int(local_slice_idx))
        if entry is None:
            labels2d, num = ndi.label(sub[int(local_slice_idx)], structure=structure2)
            centroids: Dict[int, Tuple[int, int]] = {}
            for comp_lbl in range(1, int(num) + 1):
                cent = _component_centroid_anchor(labels2d == comp_lbl)
                if cent is not None:
                    centroids[int(comp_lbl)] = cent
            entry = (labels2d, centroids)
            slice_cache[int(local_slice_idx)] = entry

        labels2d, centroids = entry
        comp_lbl = int(labels2d[int(local_y), int(local_x)])
        if comp_lbl <= 0:
            return None
        cent = centroids.get(comp_lbl)
        if cent is not None:
            return cent
        return _component_centroid_anchor(labels2d == comp_lbl)

    skel = skeletonize_volume(sub)
    if np.any(skel):
        neigh = ndi.convolve(skel.astype(np.uint8), KERNEL_3, mode='constant', cval=0) - skel.astype(np.uint8)
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

        cent = _slice_centroid(int(ep_t[0]), int(ep_t[1]), int(ep_t[2]))
        if cent is None:
            continue
        gpoint = (slice_start + int(ep_t[0]), int(sl[1].start) + int(ep_t[1]), int(sl[2].start) + int(ep_t[2]))
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
            extremes.append((last_local, 1))

            for local_slice_idx, direction_sign in extremes:
                entry = slice_cache.get(int(local_slice_idx))
                if entry is None:
                    labels2d, num = ndi.label(sub[int(local_slice_idx)], structure=structure2)
                    centroids: Dict[int, Tuple[int, int]] = {}
                    for comp_lbl in range(1, int(num) + 1):
                        cent = _component_centroid_anchor(labels2d == comp_lbl)
                        if cent is not None:
                            centroids[int(comp_lbl)] = cent
                    entry = (labels2d, centroids)
                    slice_cache[int(local_slice_idx)] = entry
                labels2d, centroids = entry
                num = int(np.max(labels2d))
                for comp_lbl in range(1, num + 1):
                    cent = centroids.get(int(comp_lbl))
                    if cent is None:
                        cent = _component_centroid_anchor(labels2d == comp_lbl)
                    if cent is None:
                        continue
                    gpoint = (
                        slice_start + int(local_slice_idx),
                        int(sl[1].start) + int(cent[0]),
                        int(sl[2].start) + int(cent[1]),
                    )
                    key = (int(lbl), int(gpoint[0]), int(direction_sign), int(gpoint[1]), int(gpoint[2]))
                    grouped[key] = SliceEndpointSeed(label=int(lbl), point=gpoint, direction_sign=int(direction_sign))

    return [grouped[k] for k in sorted(grouped.keys())]


def _build_slice_endpoint_seeds(
    labels_real: np.ndarray,
    extension_slices: int,
    workers: int = 1,
) -> Tuple[List[SliceEndpointSeed], int]:
    """Build interpolation endpoint seeds.

    Endpoint mode is controlled by ``YOLO_TTA_INTERPOLATION_ENDPOINT_MODE``:
      - ``scan`` (default): always use the fast slice-graph terminal scan
      - ``hybrid``: use per-object 3D skeletonization only when every object bounding box is small
      - ``skeleton``: always use per-object 3D skeletonization

    The scan path is the safe default for SLURM-scale volumes because a single object can have a
    massive bounding box after compact relabel, making voxel skeletonization effectively intractable.
    """
    mode = os.environ.get('YOLO_TTA_INTERPOLATION_ENDPOINT_MODE', 'scan').strip().lower()
    if mode not in {'scan', 'hybrid', 'skeleton'}:
        mode = 'scan'

    if mode == 'scan':
        return build_slice_endpoint_seeds_from_label_volume(
            labels_real,
            workers=int(workers),
            desc='Interpolation: endpoint seeds [scan]',
        )

    objs = ndi.find_objects(labels_real)
    tasks = [(lbl, sl) for lbl, sl in enumerate(objs, start=1) if sl is not None]
    if not tasks:
        return [], 0

    if mode == 'hybrid':
        max_bbox_voxels = max(
            (int(sl[0].stop - sl[0].start) * int(sl[1].stop - sl[1].start) * int(sl[2].stop - sl[2].start) for _, sl in tasks),
            default=0,
        )
        max_allowed_bbox_voxels = max(0, _env_int('YOLO_TTA_MAX_SKELETON_BBOX_VOXELS', 64 * 1024 * 1024))
        if max_allowed_bbox_voxels > 0 and int(max_bbox_voxels) > int(max_allowed_bbox_voxels):
            print(
                'Interpolation endpoint discovery: switching to slice scan '
                f'(largest compact-relabel bbox={int(max_bbox_voxels):,} voxels exceeds '
                f'YOLO_TTA_MAX_SKELETON_BBOX_VOXELS={int(max_allowed_bbox_voxels):,})'
            )
            return build_slice_endpoint_seeds_from_label_volume(
                labels_real,
                workers=int(workers),
                desc='Interpolation: endpoint seeds [scan]',
            )

    seeds: List[SliceEndpointSeed] = []
    direction_depth = max(2, min(8, int(extension_slices) + 1))
    worker_count = choose_slice_parallel_workers(int(workers), len(tasks))

    def _process(idx: int) -> List[SliceEndpointSeed]:
        lbl, sl = tasks[int(idx)]
        return _build_slice_endpoint_seeds_for_label(labels_real, int(lbl), sl, direction_depth)

    if worker_count <= 1:
        for idx in tqdm(range(len(tasks)), desc='Interpolation: endpoint seeds [skeleton]'):
            seeds.extend(_process(int(idx)))
    else:
        pending = max(worker_count, worker_count * 2)
        for seed_group in tqdm(
            parallel_map_in_order(_process, range(len(tasks)), max_workers=worker_count, max_pending=pending),
            total=len(tasks),
            desc='Interpolation: endpoint seeds [skeleton]',
        ):
            seeds.extend(seed_group)

    seeds.sort(key=lambda s: (int(s.label), int(s.point[0]), int(s.direction_sign), int(s.point[1]), int(s.point[2])))
    return seeds, int(len(seeds))


def _build_linear_slice_bridge_plan(
    labels_real: np.ndarray,
    source_label: int,
    target_label: int,
    source_point: Tuple[int, int, int],
    target_point: Tuple[int, int, int],
) -> Optional[SliceBridgeRenderPlan]:
    s0, y0, x0 = source_point
    s1, y1, x1 = target_point
    if int(s0) == int(s1):
        return None

    source_component, source_anchor = _component_mask_and_anchor(labels_real[s0] == int(source_label), (y0, x0))
    target_component, target_anchor = _component_mask_and_anchor(labels_real[s1] == int(target_label), (y1, x1))
    if source_anchor is None or target_anchor is None:
        return None
    if not np.any(source_component) or not np.any(target_component):
        return None

    half_width = _local_half_width_for_components(source_component, source_anchor, target_component, target_anchor)
    source_local = _component_to_local_canvas(source_component, source_anchor, half_width)
    target_local = _component_to_local_canvas(target_component, target_anchor, half_width)
    if not np.any(source_local) or not np.any(target_local):
        return None

    steps = int(abs(int(s1) - int(s0)))
    if steps <= 0:
        return None

    return SliceBridgeRenderPlan(
        source_label=int(source_label),
        target_label=int(target_label),
        source_point=(int(s0), int(y0), int(x0)),
        target_point=(int(s1), int(y1), int(x1)),
        source_anchor=(int(source_anchor[0]), int(source_anchor[1])),
        target_anchor=(int(target_anchor[0]), int(target_anchor[1])),
        steps=int(steps),
        sign=1 if int(s1) > int(s0) else -1,
        sdf0=np.ascontiguousarray(_signed_distance_2d(source_local)),
        sdf1=np.ascontiguousarray(_signed_distance_2d(target_local)),
    )


def _estimate_linear_slice_bridge_min_radius_from_plan(plan: SliceBridgeRenderPlan) -> float:
    source_local = np.asarray(plan.sdf0 >= 0.0, dtype=bool)
    target_local = np.asarray(plan.sdf1 >= 0.0, dtype=bool)
    if not np.any(source_local) or not np.any(target_local):
        return 0.0

    min_radius = min(_component_max_radius(source_local), _component_max_radius(target_local))
    for idx in range(1, int(plan.steps)):
        alpha = float(idx) / float(plan.steps)
        section = ((1.0 - alpha) * plan.sdf0 + alpha * plan.sdf1) >= 0.0
        if not np.any(section):
            return 0.0
        section = _keep_center_component_2d(section)
        min_radius = min(min_radius, _component_max_radius(section))
    return float(min_radius)


def _paint_linear_slice_bridge_plan_onto_slice(
    dest_slice: np.ndarray,
    plan: SliceBridgeRenderPlan,
    step_idx: int,
) -> int:
    if int(step_idx) <= 0 or int(step_idx) >= int(plan.steps):
        return 0

    alpha = float(step_idx) / float(plan.steps)
    section = ((1.0 - alpha) * plan.sdf0 + alpha * plan.sdf1) >= 0.0
    if not np.any(section):
        return 0
    section = _keep_center_component_2d(section)
    center = (
        (1.0 - alpha) * float(plan.source_anchor[0]) + alpha * float(plan.target_anchor[0]),
        (1.0 - alpha) * float(plan.source_anchor[1]) + alpha * float(plan.target_anchor[1]),
    )
    return _paste_local_mask_onto_slice(dest_slice, section, center)


def _plan_slice_seed_bridges(
    labels_real: np.ndarray,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_min_radius: float,
) -> SliceSeedBridgePlanResult:
    result = SliceSeedBridgePlanResult()

    candidates = _find_slice_projection_candidates(
        labels_real=labels_real,
        seed=seed,
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        max_candidates=int(interpolation_candidates),
    )
    if not candidates:
        return result

    result.candidate_connections = int(len(candidates))
    source_points = [seed.point] + _collect_walkback_source_points(
        labels_real=labels_real,
        label=int(seed.label),
        start_point=seed.point,
        direction_sign=int(seed.direction_sign),
        walk_back=int(interpolation_walk_back),
    )

    for candidate in candidates:
        accepted_this_candidate = False
        for walk_idx, src_point in enumerate(source_points):
            plan = _build_linear_slice_bridge_plan(
                labels_real=labels_real,
                source_label=int(candidate.source_label),
                target_label=int(candidate.target_label),
                source_point=src_point,
                target_point=candidate.target_point,
            )
            if plan is None:
                continue

            if float(interpolate_min_radius) > 0.0:
                bridge_radius = _estimate_linear_slice_bridge_min_radius_from_plan(plan)
                if bridge_radius <= float(interpolate_min_radius):
                    result.skipped_by_min_radius += 1
                    continue

            if walk_idx == 0:
                result.default_bridges += 1
            else:
                result.walk_back_bridges += 1

            if not accepted_this_candidate:
                result.accepted_connections += 1
                accepted_this_candidate = True

            result.plans.append(plan)

    return result


def _build_slice_bridge_render_schedule(
    plans: Sequence[SliceBridgeRenderPlan],
    num_slices: int,
) -> List[List[Tuple[int, int]]]:
    schedule: List[List[Tuple[int, int]]] = [[] for _ in range(int(num_slices))]
    for plan_idx, plan in enumerate(plans):
        start_slice = int(plan.source_point[0])
        for step_idx in range(1, int(plan.steps)):
            s = int(start_slice + int(plan.sign) * step_idx)
            if 0 <= s < int(num_slices):
                schedule[s].append((int(plan_idx), int(step_idx)))
    return schedule

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
    workers: int = 1,
) -> Dict[str, object]:
    """Apply one interpolation pass directly to a view-volume stack.

    The pass keeps bridge creation simultaneous by searching against a frozen label snapshot and
    merging all newly created bridge voxels only after planning is complete. Endpoint discovery,
    candidate search, bridge planning and slice rendering are parallelized across independent
    objects / seeds / slices to reduce the long single-threaded stretch after compact relabel.
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

    worker_count = choose_slice_parallel_workers(int(workers), max(1, int(num_objects)))
    seeds, num_endpoints = _build_slice_endpoint_seeds(
        labels_mm,
        extension_slices=int(max_slice_distance),
        workers=worker_count,
    )
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
    plans: List[SliceBridgeRenderPlan] = []

    try:
        plan_workers = choose_slice_parallel_workers(int(workers), len(seeds))

        def _plan_seed(idx: int) -> SliceSeedBridgePlanResult:
            return _plan_slice_seed_bridges(
                labels_real=labels_mm,
                seed=seeds[int(idx)],
                max_slice_distance=int(max_slice_distance),
                search_angle_deg=float(search_angle_deg),
                interpolation_walk_back=int(interpolation_walk_back),
                interpolation_candidates=int(interpolation_candidates),
                interpolate_min_radius=float(interpolate_min_radius),
            )

        pending = max(plan_workers, plan_workers * 2)
        if plan_workers <= 1:
            iterable = (_plan_seed(int(idx)) for idx in range(len(seeds)))
        else:
            iterable = parallel_map_in_order(
                _plan_seed,
                range(len(seeds)),
                max_workers=plan_workers,
                max_pending=pending,
            )

        for seed_result in tqdm(iterable, total=len(seeds), desc='Interpolation: seed planning'):
            candidate_connections += int(seed_result.candidate_connections)
            accepted_connections += int(seed_result.accepted_connections)
            default_bridges += int(seed_result.default_bridges)
            walk_back_bridges += int(seed_result.walk_back_bridges)
            skipped_by_min_radius += int(seed_result.skipped_by_min_radius)
            if seed_result.plans:
                plans.extend(seed_result.plans)

        if plans:
            schedule = _build_slice_bridge_render_schedule(plans, int(mask_mm.shape[0]))
            added_counts = np.zeros((int(mask_mm.shape[0]),), dtype=np.int64)
            render_workers = choose_slice_parallel_workers(int(workers), int(mask_mm.shape[0]))

            def _render_slice(z: int) -> None:
                contribs = schedule[int(z)]
                if not contribs:
                    return
                bridge_slice = bridge_mm[int(z)]
                local_added = 0
                for plan_idx, step_idx in contribs:
                    local_added += _paint_linear_slice_bridge_plan_onto_slice(
                        bridge_slice,
                        plans[int(plan_idx)],
                        int(step_idx),
                    )
                added_counts[int(z)] = np.int64(local_added)

            parallel_for_indices(
                int(mask_mm.shape[0]),
                _render_slice,
                max_workers=render_workers,
                desc='Interpolation: render bridges',
            )
            added_voxels = int(np.sum(added_counts, dtype=np.int64))
            del schedule
            del added_counts

            def _merge_slice(z: int) -> None:
                bridge_slice = np.asarray(bridge_mm[int(z)])
                if np.any(bridge_slice):
                    mask_mm[int(z), :, :] |= bridge_slice

            parallel_for_indices(
                int(mask_mm.shape[0]),
                _merge_slice,
                max_workers=render_workers,
                desc='Interpolation: merge bridges',
            )
            flush_array(mask_mm)
    finally:
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



# --------------------------
# Final outputs
# --------------------------


def write_overlay_video(
    volume_rgb: np.memmap,  # (T,H,W,3) RGB
    mask_u8: np.ndarray,    # (T,H,W) 0/1
    out_path: Path,
    fps: float,
    show_progress: bool = True,
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
        for t in tqdm(range(T), desc=f"Writing overlay video ({out_path.name})", disable=not show_progress):
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


def _write_label_file_from_mask(mask2d: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mask2d) > 0
    if not np.any(m):
        out_path.write_text("")
        return

    polys = mask_to_yolo_polygons(m.astype(np.uint8))
    if not polys:
        out_path.write_text("")
        return

    lines: List[str] = []
    for poly in polys:
        coords: List[str] = []
        for x, y in poly:
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")
        lines.append("0 " + " ".join(coords))
    out_path.write_text("\n".join(lines) + "\n")


def _write_binary_tiff_frame(mask2d: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), (np.asarray(mask2d) * 255).astype(np.uint8))


def write_yolo_labels_from_pattern(
    mask_u8: np.ndarray,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(mask_u8.shape[0])

    def _write_frame(t: int) -> None:
        fp = _format_frame_path(pattern_path, int(t) + 1)
        _write_label_file_from_mask(np.asarray(mask_u8[int(t)]), fp)

    parallel_for_indices(
        total,
        _write_frame,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=f"Writing YOLO labels ({pattern_path.parent.name})",
        show_progress=show_progress,
    )
    return pattern_path.parent


def write_binary_tiff_sequence_from_pattern(
    mask_u8: np.ndarray,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(mask_u8.shape[0])

    def _write_frame(t: int) -> None:
        fp = _format_frame_path(pattern_path, int(t) + 1)
        _write_binary_tiff_frame(np.asarray(mask_u8[int(t)]), fp)

    parallel_for_indices(
        total,
        _write_frame,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=f"Writing binary TIFF sequence ({pattern_path.parent.name})",
        show_progress=show_progress,
    )
    return pattern_path.parent


def write_binary_video_from_mask_volume(
    mask_u8: np.ndarray,
    video_path: Path,
    fps: float,
    show_progress: bool = True,
) -> Path:
    T, H, W = mask_u8.shape
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
        for t in tqdm(range(T), desc=f"Writing binary MKV ({video_path.name})", disable=not show_progress):
            proc.stdin.write((np.asarray(mask_u8[t]) * 255).astype(np.uint8).tobytes())
    finally:
        close_ffmpeg_writer(proc)
    return video_path


def write_nrrd(mask_u8: np.ndarray, out_path: Path) -> Path:
    try:
        import nrrd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pynrrd is required for --save_nrrd: pip install pynrrd") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(out_path), np.asarray(mask_u8, dtype=np.uint8))
    return out_path


def extract_radial_slice_mask_frame(mask_u8: np.ndarray, sampler: RadialSampler) -> np.ndarray:
    t_dim = int(mask_u8.shape[0])
    tt = np.arange(t_dim, dtype=np.int64)[:, None]
    yy = np.broadcast_to(sampler.nn_y[None, :], (t_dim, sampler.diameter))
    xx = np.broadcast_to(sampler.nn_x[None, :], (t_dim, sampler.diameter))
    return np.ascontiguousarray(mask_u8[tt, yy, xx].astype(np.uint8, copy=False))


def get_view_mask_frame_by_index(mask_u8: np.ndarray, view: ViewInfo, index: int) -> np.ndarray:
    if view.name == 'transverse':
        return np.asarray(mask_u8[int(index)])
    if view.name == 'sagittal':
        return np.ascontiguousarray(mask_u8[:, int(index), :])
    if view.name == 'coronal':
        return np.ascontiguousarray(mask_u8[:, :, int(index)])
    if view.name == 'radial':
        sampler = get_radial_sampler(view, float(view.azimuths_deg[int(index)]))
        return extract_radial_slice_mask_frame(mask_u8, sampler)
    raise ValueError(f'Unknown view: {view.name}')


def iter_view_mask_frames(mask_u8: np.ndarray, view: ViewInfo) -> Iterator[np.ndarray]:
    for idx in range(int(view.num_slices)):
        yield get_view_mask_frame_by_index(mask_u8, view, int(idx))


def write_overlay_video_for_view(
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    view: ViewInfo,
    out_path: Path,
    fps: float,
    show_progress: bool = True,
) -> None:
    if view.name == 'transverse':
        write_overlay_video(volume_rgb, mask_u8, out_path, fps, show_progress=show_progress)
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
        iterator = zip(iter_view_frames(volume_rgb, view), iter_view_mask_frames(mask_u8, view))
        for frame_rgb, frame_mask in tqdm(
            iterator,
            total=view.num_slices,
            desc=f'Writing {view.name} overlay video ({out_path.name})',
            disable=not show_progress,
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
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(view.num_slices)

    def _write_frame(idx: int) -> None:
        fp = _format_frame_path(pattern_path, int(idx) + 1)
        _write_label_file_from_mask(get_view_mask_frame_by_index(mask_u8, view, int(idx)), fp)

    parallel_for_indices(
        total,
        _write_frame,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=f'Writing YOLO labels ({pattern_path.parent.name})',
        show_progress=show_progress,
    )
    return pattern_path.parent


def write_view_binary_tiff_sequence_from_pattern(
    mask_u8: np.ndarray,
    view: ViewInfo,
    pattern_path: Path,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(view.num_slices)

    def _write_frame(idx: int) -> None:
        fp = _format_frame_path(pattern_path, int(idx) + 1)
        _write_binary_tiff_frame(get_view_mask_frame_by_index(mask_u8, view, int(idx)), fp)

    parallel_for_indices(
        total,
        _write_frame,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=f'Writing binary TIFF sequence ({pattern_path.parent.name})',
        show_progress=show_progress,
    )
    return pattern_path.parent


def write_view_binary_video_from_mask_volume(
    mask_u8: np.ndarray,
    view: ViewInfo,
    video_path: Path,
    fps: float,
    show_progress: bool = True,
) -> Path:
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
        for idx in tqdm(range(view.num_slices), total=view.num_slices, desc=f'Writing binary MKV ({video_path.name})', disable=not show_progress):
            proc.stdin.write((np.asarray(get_view_mask_frame_by_index(mask_u8, view, int(idx))) * 255).astype(np.uint8).tobytes())
    finally:
        close_ffmpeg_writer(proc)
    return video_path


def write_view_binary_outputs_from_pattern(
    mask_u8: np.ndarray,
    view: ViewInfo,
    pattern_path: Path,
    video_path: Path,
    fps: float,
    workers: int = 1,
    show_progress: bool = True,
) -> Tuple[Path, Path]:
    tiff_dir = write_view_binary_tiff_sequence_from_pattern(
        mask_u8,
        view,
        pattern_path,
        workers=int(workers),
        show_progress=show_progress,
    )
    binary_video_path = write_view_binary_video_from_mask_volume(
        mask_u8,
        view,
        video_path,
        fps,
        show_progress=show_progress,
    )
    return tiff_dir, binary_video_path


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
    workers: int = 1,
    show_progress: bool = True,
) -> Dict[str, Path]:
    pretty = view.name.capitalize()
    tag_suffix = f'_{tag}' if tag else ''
    overlay_path = out_dir / f'{stem}_{pretty}_Overlay{tag_suffix}.mkv'
    write_overlay_video_for_view(volume_rgb, mask_u8, view, overlay_path, fps, show_progress=show_progress)
    result_paths: Dict[str, Path] = {f'{view.name}_overlay': overlay_path}

    labels_pattern = _resolve_output_pattern(save_labels_pattern_value, DEFAULT_LABEL_PATTERN, out_dir, stem)
    if labels_pattern is not None:
        labels_pattern = _tag_frame_pattern(labels_pattern, pretty if tag is None else f'{pretty}_{tag}')
        labels_dir = write_view_yolo_labels_from_pattern(
            mask_u8,
            view,
            labels_pattern,
            workers=int(workers),
            show_progress=show_progress,
        )
        result_paths[f'{view.name}_labels_dir'] = labels_dir

    binary_pattern = _resolve_output_pattern(save_binary_pattern_value, DEFAULT_BINARY_PATTERN, out_dir, stem)
    if binary_pattern is not None:
        binary_pattern = _tag_frame_pattern(binary_pattern, pretty if tag is None else f'{pretty}_{tag}')
        binary_video_path = out_dir / f'{stem}_{pretty}_Binary{tag_suffix}.mkv'
        tiff_dir, binary_video_path = write_view_binary_outputs_from_pattern(
            mask_u8,
            view,
            binary_pattern,
            binary_video_path,
            fps,
            workers=int(workers),
            show_progress=show_progress,
        )
        result_paths[f'{view.name}_binary_tiff_dir'] = tiff_dir
        result_paths[f'{view.name}_binary_video'] = binary_video_path

    return result_paths


@dataclass
class BackgroundOutputSubmission:
    label: str
    result_paths: Dict[str, Path]
    futures: List[Future] = field(default_factory=list)
    resources: List[object] = field(default_factory=list)

    def wait(self) -> Dict[str, Path]:
        error: Optional[BaseException] = None
        try:
            for fut in self.futures:
                fut.result()
        except BaseException as exc:  # pragma: no cover - surfaced to main
            error = exc
        finally:
            for resource in self.resources:
                close_memmap_array(resource)
        if error is not None:
            raise RuntimeError(f'Background output generation failed for {self.label}') from error
        return self.result_paths


class BackgroundOutputManager:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, int(max_workers))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='yolo-output')
        self.pending: List[BackgroundOutputSubmission] = []

    def submit(self, submission: BackgroundOutputSubmission) -> Dict[str, Path]:
        self.pending.append(submission)
        return submission.result_paths

    def reap_completed(self) -> None:
        remaining: List[BackgroundOutputSubmission] = []
        for submission in self.pending:
            if all(fut.done() for fut in submission.futures):
                submission.wait()
            else:
                remaining.append(submission)
        self.pending = remaining

    def wait(self) -> None:
        error: Optional[BaseException] = None
        try:
            while self.pending:
                submission = self.pending.pop(0)
                try:
                    submission.wait()
                except BaseException as exc:  # pragma: no cover - surfaced to main
                    if error is None:
                        error = exc
        finally:
            self.executor.shutdown(wait=True)
        if error is not None:
            raise error


def collect_pipeline_output_futures(
    executor: ThreadPoolExecutor,
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
    frame_workers: int = 1,
    show_progress: bool = False,
) -> Tuple[Dict[str, Path], List[Future]]:
    futures: List[Future] = []
    result_paths: Dict[str, Path] = {}
    tag_suffix = f"_{tag}" if tag else ""

    overlay_path = out_dir / f"{stem}_Overlay{tag_suffix}.mkv"
    futures.append(executor.submit(write_overlay_video, volume_rgb, mask_u8, overlay_path, fps, show_progress))
    result_paths["overlay"] = overlay_path

    labels_pattern = _resolve_output_pattern(save_labels_pattern_value, DEFAULT_LABEL_PATTERN, out_dir, stem)
    if labels_pattern is not None:
        if tag is not None:
            labels_pattern = _tag_frame_pattern(labels_pattern, tag)
        futures.append(executor.submit(write_yolo_labels_from_pattern, mask_u8, labels_pattern, int(frame_workers), show_progress))
        result_paths["labels_dir"] = labels_pattern.parent

    binary_pattern = _resolve_output_pattern(save_binary_pattern_value, DEFAULT_BINARY_PATTERN, out_dir, stem)
    if binary_pattern is not None:
        if tag is not None:
            binary_pattern = _tag_frame_pattern(binary_pattern, tag)
        binary_video_path = out_dir / f"{stem}_Binary{tag_suffix}.mkv"
        futures.append(executor.submit(write_binary_tiff_sequence_from_pattern, mask_u8, binary_pattern, int(frame_workers), show_progress))
        futures.append(executor.submit(write_binary_video_from_mask_volume, mask_u8, binary_video_path, fps, show_progress))
        result_paths["binary_tiff_dir"] = binary_pattern.parent
        result_paths["binary_video"] = binary_video_path

    if bool(save_nrrd_flag):
        nrrd_path = out_dir / f"{stem}{tag_suffix}.nrrd"
        futures.append(executor.submit(write_nrrd, mask_u8, nrrd_path))
        result_paths["nrrd"] = nrrd_path

    return result_paths, futures


def collect_multiplanar_output_futures(
    executor: ThreadPoolExecutor,
    *,
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    out_dir: Path,
    stem: str,
    fps: float,
    save_binary_pattern_value: Optional[str],
    save_labels_pattern_value: Optional[str],
    tag: Optional[str] = None,
    frame_workers: int = 1,
    show_progress: bool = False,
) -> Tuple[Dict[str, Path], List[Future]]:
    t_dim, h_dim, w_dim = mask_u8.shape
    views = {v.name: v for v in get_view_infos(t_dim, h_dim, w_dim, disable_multiplanar=False, azimuth_angle=0.0, include_radial=False)}
    futures: List[Future] = []
    result_paths: Dict[str, Path] = {}

    for view_name in ('sagittal', 'coronal'):
        view = views[view_name]
        pretty = view.name.capitalize()
        tag_suffix = f'_{tag}' if tag else ''

        overlay_path = out_dir / f'{stem}_{pretty}_Overlay{tag_suffix}.mkv'
        futures.append(executor.submit(write_overlay_video_for_view, volume_rgb, mask_u8, view, overlay_path, fps, show_progress))
        result_paths[f'{view.name}_overlay'] = overlay_path

        labels_pattern = _resolve_output_pattern(save_labels_pattern_value, DEFAULT_LABEL_PATTERN, out_dir, stem)
        if labels_pattern is not None:
            labels_pattern = _tag_frame_pattern(labels_pattern, pretty if tag is None else f'{pretty}_{tag}')
            futures.append(executor.submit(write_view_yolo_labels_from_pattern, mask_u8, view, labels_pattern, int(frame_workers), show_progress))
            result_paths[f'{view.name}_labels_dir'] = labels_pattern.parent

        binary_pattern = _resolve_output_pattern(save_binary_pattern_value, DEFAULT_BINARY_PATTERN, out_dir, stem)
        if binary_pattern is not None:
            binary_pattern = _tag_frame_pattern(binary_pattern, pretty if tag is None else f'{pretty}_{tag}')
            binary_video_path = out_dir / f'{stem}_{pretty}_Binary{tag_suffix}.mkv'
            futures.append(executor.submit(write_view_binary_tiff_sequence_from_pattern, mask_u8, view, binary_pattern, int(frame_workers), show_progress))
            futures.append(executor.submit(write_view_binary_video_from_mask_volume, mask_u8, view, binary_video_path, fps, show_progress))
            result_paths[f'{view.name}_binary_tiff_dir'] = binary_pattern.parent
            result_paths[f'{view.name}_binary_video'] = binary_video_path

    return result_paths, futures


def link_or_copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)
    return dst


def write_view_images(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    out_dir: Path,
    stem: str,
    *,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    view_dir = out_dir / 'images' / view.name
    view_dir.mkdir(parents=True, exist_ok=True)
    total = int(view.num_slices)

    def _write_frame(idx: int) -> None:
        frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(idx)))
        out_path = view_dir / f'{stem}_{view.name}_{int(idx) + 1:04d}.png'
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(out_path), bgr):
            raise RuntimeError(f'Failed to write image: {out_path}')

    parallel_for_indices(
        total,
        _write_frame,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=f'Writing {view.name} image sequence',
        show_progress=show_progress,
    )
    return view_dir


def union_view_volume_across_models(
    view_name: str,
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
) -> np.ndarray:
    source_shape: Optional[Tuple[int, ...]] = None
    for model_name in sorted(view_volumes_by_model.keys()):
        if view_name in view_volumes_by_model[model_name]:
            source_shape = tuple(int(x) for x in np.asarray(view_volumes_by_model[model_name][view_name]).shape)
            break
    if source_shape is None:
        raise KeyError(f'No per-model view volume found for {view_name}')

    union_mm = allocate_workspace_array(
        shape=source_shape,
        dtype=np.uint8,
        path=out_path,
        desc=f'Union view volume {view_name}',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    num_slices = int(source_shape[0]) if len(source_shape) > 0 else 0

    def _merge_slice(idx: int) -> None:
        dst = union_mm[int(idx)]
        for model_name in sorted(view_volumes_by_model.keys()):
            if view_name not in view_volumes_by_model[model_name]:
                continue
            dst |= np.asarray(view_volumes_by_model[model_name][view_name][int(idx)], dtype=np.uint8)

    parallel_for_indices(
        num_slices,
        _merge_slice,
        max_workers=choose_slice_parallel_workers(int(workers), num_slices),
        desc=f'Union per-model view volumes ({view_name})',
        show_progress=False,
    )
    flush_array(union_mm)
    return union_mm


def write_tta_labels_for_view(
    mask_source: np.ndarray,
    view: ViewInfo,
    aug_jobs: Sequence[AugJob],
    out_dir: Path,
    stem: str,
    *,
    mask_source_is_native: bool = False,
    workers: int = 1,
    show_progress: bool = True,
) -> Dict[str, Path]:
    result_paths: Dict[str, Path] = {}
    root = out_dir / 'TTA' / view.name
    videos_dir = root / 'videos'
    labels_root = root / 'labels'
    videos_dir.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    for job in aug_jobs:
        if job.video_path.exists():
            copied_video = link_or_copy_file(job.video_path, videos_dir / job.video_path.name)
            result_paths[f'{view.name}_tta_video_{job.aug_id}'] = copied_video

        labels_dir = labels_root / job.aug_id
        labels_dir.mkdir(parents=True, exist_ok=True)
        total = int(view.num_slices)

        def _write_frame(idx: int) -> None:
            if bool(mask_source_is_native):
                native_mask = np.asarray(mask_source[int(idx)], dtype=np.uint8)
            else:
                native_mask = np.asarray(get_view_mask_frame_by_index(mask_source, view, int(idx)), dtype=np.uint8)
            warped = cv2.warpAffine(
                native_mask,
                job.aff.M_src_to_out,
                dsize=(int(job.aff.out_size), int(job.aff.out_size)),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            _write_label_file_from_mask(warped, labels_dir / f'{stem}_{job.aug_id}_{int(idx) + 1:04d}.txt')

        parallel_for_indices(
            total,
            _write_frame,
            max_workers=choose_slice_parallel_workers(int(workers), total),
            desc=f'Writing {view.name} TTA labels ({job.aug_id})',
            show_progress=show_progress,
        )
        result_paths[f'{view.name}_tta_labels_{job.aug_id}'] = labels_dir

    result_paths[f'{view.name}_tta_root'] = root
    return result_paths


def save_tta_outputs(
    *,
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    views: Sequence[ViewInfo],
    aug_jobs_by_view: Dict[str, List[AugJob]],
    temp_dir: Path,
    out_dir: Path,
    stem: str,
    workers: int = 1,
    show_progress: bool = True,
) -> Dict[str, Path]:
    result_paths: Dict[str, Path] = {}
    for view in views:
        aug_jobs = list(aug_jobs_by_view.get(view.name, []))
        if not aug_jobs:
            continue
        union_mask = union_view_volume_across_models(
            view.name,
            view_volumes_by_model,
            temp_dir / 'tta' / f'{view.name}.u8.dat',
            prefer_memory=True,
            workers=int(workers),
        )
        try:
            result_paths.update(write_tta_labels_for_view(
                mask_source=union_mask,
                view=view,
                aug_jobs=aug_jobs,
                out_dir=out_dir,
                stem=stem,
                mask_source_is_native=(view.family == 'orthogonal' and view.name != 'radial'),
                workers=int(workers),
                show_progress=show_progress,
            ))
        finally:
            close_memmap_array(union_mask)
            try:
                (temp_dir / 'tta' / f'{view.name}.u8.dat').unlink(missing_ok=True)
            except Exception:
                pass
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
    slice_postprocess_workers: int,
    interpolation_workers: int,
    output_workers: int,
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
    lines.append(f'Slice-parallel postprocess workers: {int(slice_postprocess_workers)}')
    lines.append(f'Interpolation workers: {int(interpolation_workers)}')
    lines.append(f'Output workers: {int(output_workers)}')
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
                    f"skipped={bool(s.get('skipped', False))}"
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
    tile_configs = resolve_tile_configs(args.tile_size, args.tile_stride)

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

    temp_dir = choose_scratch_dir(None, out_dir, input_path.stem)
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
        disable_multiplanar=not bool(args.enable_multiplanar),
        azimuth_angle=float(args.azimuth_angle),
        include_radial=True,
    )
    cartesian_views = orthogonal_views_only(views)

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

    worker_budget = int(default_worker_budget())
    augmentation_workers = resolve_worker_count(
        0,
        'YOLO_TTA_AUG_WORKERS',
        worker_budget,
        max_tasks=max(1, max(v.num_slices for v in views)),
    )
    interpolation_workers = resolve_worker_count(
        0,
        'YOLO_TTA_INTERPOLATION_WORKERS',
        worker_budget,
        max_tasks=max(1, len(yolo_models) * len(cartesian_views)),
    )
    output_workers = resolve_worker_count(
        0,
        'YOLO_TTA_OUTPUT_WORKERS',
        worker_budget,
        max_tasks=24,
    )
    output_frame_workers = max(1, min(16, _env_int('YOLO_TTA_OUTPUT_FRAME_WORKERS', max(1, min(8, output_workers)))))
    slice_postprocess_workers = max(1, int(augmentation_workers))
    predict_postprocess_workers = max(1, min(32, _env_int('YOLO_TTA_PREDICT_POSTPROCESS_WORKERS', slice_postprocess_workers)))
    print(f'Allocated CPU count: {_cpu_count()}')
    print(f'Worker budget: {worker_budget}')
    print(f'Augmentation workers: {augmentation_workers}')
    print(f'Slice-parallel postprocess workers: {slice_postprocess_workers}')
    print(f'Inference postprocess workers: {predict_postprocess_workers}')
    print(f'Interpolation workers: {interpolation_workers}')
    print(f'Background output workers: {output_workers} (frame workers per labels/TIFF task: {output_frame_workers})')

    output_manager = BackgroundOutputManager(max_workers=output_workers)

    if augmentation_workers > 1 or interpolation_workers > 1 or slice_postprocess_workers > 1 or output_workers > 1:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    aug_jobs_by_view: Dict[str, List[AugJob]] = {}
    dense_tiling_active = len(tile_configs) > 0
    view_prediction_stats: Dict[str, int] = {
        'transverse': 0,
        'sagittal': 0,
        'coronal': 0,
        'radial': 0,
    }

    for view in views:
        aug_jobs_by_view[view.name] = build_aug_jobs_for_view(
            view=view,
            angles=angles,
            out_size=args.imgsz,
            temp_dir=temp_dir,
        )

    view_prefetch_workers = max(
        1,
        min(len(views), max(1, augmentation_workers), _env_int('YOLO_TTA_VIEW_PREFETCH_WORKERS', max(1, len(views)))),
    )
    per_view_prefetch_workers = max(1, augmentation_workers // max(1, view_prefetch_workers))
    print(
        f'View render prefetch workers: {view_prefetch_workers} '
        f'(per-view render workers: {per_view_prefetch_workers})'
    )

    view_prefetch_executor = ThreadPoolExecutor(
        max_workers=int(view_prefetch_workers),
        thread_name_prefix='view-prefetch',
    )
    view_prefetch_futures: Dict[str, Future] = {}
    for view in views:
        view_prefetch_futures[view.name] = view_prefetch_executor.submit(
            prepare_view_render_assets,
            volume_rgb,
            view,
            list(aug_jobs_by_view[view.name]),
            float(fps),
            int(per_view_prefetch_workers),
            bool(dense_tiling_active),
            temp_dir,
        )

    for view in views:
        extra = ''
        if view.family == 'radial':
            extra = f', azimuths={view.num_slices}'
        print(f"\n=== View: {view.name} ({view.src_w}x{view.src_h}, slices={view.num_slices}{extra}) ===")

        if not view_prefetch_futures[view.name].done():
            print(f"Waiting for prefetched render assets for view '{view.name}' to finish...")
        prefetch_result = view_prefetch_futures[view.name].result()
        view_frame_cache = prefetch_result.view_frames
        view_frame_cache_path = prefetch_result.view_frame_cache_path
        aug_jobs = list(aug_jobs_by_view[view.name])
        if view_frame_cache is not None:
            print(f"Using reusable native frame cache for view '{view.name}'")

        baseline_union_by_model_view: Dict[str, np.ndarray] = {}
        baseline_confmap_by_model_view: Dict[str, np.ndarray] = {}
        baseline_native_by_model_view: Dict[str, np.ndarray] = {}
        baseline_support_by_model_view: Dict[str, np.ndarray] = {}
        baseline_union_paths: Dict[str, Path] = {}
        baseline_confmap_paths: Dict[str, Path] = {}
        baseline_native_paths: Dict[str, Path] = {}
        baseline_support_paths: Dict[str, Path] = {}
        tile_gate_stats_by_model: Dict[str, Dict[str, int]] = {
            model_name: {
                'prediction_count': 0,
                'frames_with_predictions': 0,
                'accepted_masks': 0,
                'rejected_masks': 0,
                'rejected_min_conf': 0,
                'gated_added_voxels': 0,
            }
            for model_name, _ in yolo_models
        }

        for model_name, _ in yolo_models:
            bytes_native = bytes_for_packbits(view.src_h, view.src_w)
            union_path = temp_dir / 'union' / model_name / f'{view.name}.union.packbits.dat'
            confmap_path = temp_dir / 'union' / model_name / f'{view.name}.confmap.f16.dat'
            union_path.parent.mkdir(parents=True, exist_ok=True)

            union_mm = allocate_workspace_array(
                shape=(view.num_slices, bytes_native),
                dtype=np.uint8,
                path=union_path,
                desc=f'{model_name}/{view.name} baseline union workspace',
                prefer_memory=True,
            )
            confmap_mm = allocate_workspace_array(
                shape=(view.num_slices, view.src_h, view.src_w),
                dtype=np.float16,
                path=confmap_path,
                desc=f'{model_name}/{view.name} baseline confidence workspace',
                prefer_memory=True,
            )

            baseline_union_by_model_view[model_name] = union_mm
            baseline_confmap_by_model_view[model_name] = confmap_mm
            baseline_union_paths[model_name] = union_path
            baseline_confmap_paths[model_name] = confmap_path

        tile_jobs_for_view: List[DenseTileJob] = []
        tile_ready_jobs: List[DenseTileJob] = []
        tile_future_to_job: Dict[Future, DenseTileJob] = {}
        tile_executor: Optional[ThreadPoolExecutor] = None
        tile_video_workers = 0
        tile_workers_per_job = 1

        if dense_tiling_active:
            for tile_cfg in tile_configs:
                for job in aug_jobs:
                    tile_jobs_for_view.extend(build_dense_tile_jobs_for_aug(
                        view=view,
                        aug_job=job,
                        tile_cfg=tile_cfg,
                        out_size=int(args.imgsz),
                        temp_dir=temp_dir,
                    ))
            if tile_jobs_for_view:
                if view_frame_cache is not None:
                    tile_workers_per_job = max(1, min(8, int(worker_budget // max(1, len(tile_jobs_for_view)))))
                tile_video_workers = max(1, min(int(len(tile_jobs_for_view)), int(worker_budget // max(1, tile_workers_per_job))))
                print(
                    f"Starting background dense tile rendering for {view.name}: "
                    f"{len(tile_jobs_for_view)} tile video(s) across {len(tile_configs)} tile configuration(s) "
                    f"with {tile_video_workers} outer worker(s)"
                )
                if view_frame_cache is not None:
                    print(
                        f"Dense tile rendering for {view.name} will reuse the native frame cache "
                        f"with {tile_workers_per_job} slice worker(s) per tile video"
                    )
                tile_executor = ThreadPoolExecutor(
                    max_workers=int(tile_video_workers),
                    thread_name_prefix=f'dense-tile-{view.name}',
                )
                tile_ready_jobs, tile_future_to_job = submit_dense_tile_video_jobs(
                    executor=tile_executor,
                    volume_rgb=volume_rgb,
                    view=view,
                    tile_jobs=tile_jobs_for_view,
                    fps=float(fps),
                    workers_per_job=int(tile_workers_per_job),
                    view_frames=view_frame_cache,
                )

        try:
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
                        view_union_mm=baseline_union_by_model_view[model_name],
                        view_confmap_mm=baseline_confmap_by_model_view[model_name],
                        M_out_to_native=aff.M_out_to_src,
                        native_h=view.src_h,
                        native_w=view.src_w,
                        postprocess_workers=predict_postprocess_workers,
                    )
                    view_prediction_stats[view.name] = int(view_prediction_stats.get(view.name, 0)) + int(pred_stats.get('prediction_count', 0))

                if not args.troubleshooting and not args.save_TTA:
                    try:
                        aug_video.unlink(missing_ok=True)
                        aug_meta.unlink(missing_ok=True)
                    except Exception:
                        pass

            for model_name, _ in yolo_models:
                print(f"\n--- Preparing full-frame baseline for view '{view.name}' and model '{model_name}' ---")
                if args.min_conf > 0:
                    apply_min_conf_filter_with_confmap_inplace(
                        baseline_union_by_model_view[model_name],
                        baseline_confmap_by_model_view[model_name],
                        float(args.min_conf),
                        view.src_h,
                        view.src_w,
                        workers=slice_postprocess_workers,
                    )
                    flush_array(baseline_union_by_model_view[model_name])
                    flush_array(baseline_confmap_by_model_view[model_name])

                native_volume_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.native.u8.dat'
                baseline_native_volume = unpack_view_union_to_volume(
                    union_mm=baseline_union_by_model_view[model_name],
                    num_slices=view.num_slices,
                    h=view.src_h,
                    w=view.src_w,
                    out_path=native_volume_path,
                    desc=f'Unpacking {model_name}/{view.name}',
                    prefer_memory=True,
                    workers=slice_postprocess_workers,
                )
                baseline_native_by_model_view[model_name] = baseline_native_volume
                baseline_native_paths[model_name] = native_volume_path

                if dense_tiling_active and tile_jobs_for_view:
                    support_volume_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.support.u8.dat'
                    baseline_support_by_model_view[model_name] = copy_workspace_array(
                        baseline_native_volume,
                        support_volume_path,
                        desc=f'{model_name}/{view.name} frozen full-frame support volume',
                        prefer_memory=True,
                        workers=slice_postprocess_workers,
                    )
                    baseline_support_paths[model_name] = support_volume_path

                union_mm_done = baseline_union_by_model_view.pop(model_name, None)
                close_memmap_array(union_mm_done)
                if not args.troubleshooting:
                    try:
                        baseline_union_paths[model_name].unlink(missing_ok=True)
                    except Exception:
                        pass

            if dense_tiling_active and tile_jobs_for_view:
                if not tile_ready_jobs and tile_future_to_job:
                    print(f"Waiting for the first background dense tile video for view '{view.name}' to finish rendering...")
                ready_iter = iter_dense_tile_jobs_in_completion_order(tile_ready_jobs, tile_future_to_job)
                for tile_job in tqdm(
                    ready_iter,
                    total=len(tile_jobs_for_view),
                    desc=f'Dense tiled inference {view.name}',
                ):
                    for model_name, yolo in yolo_models:
                        pred_stats = predict_tile_video_and_gate(
                            model=yolo,
                            video_path=tile_job.video_path,
                            num_frames=view.num_slices,
                            out_size=args.imgsz,
                            cfg=pred_cfg,
                            baseline_support_mm=baseline_support_by_model_view[model_name],
                            baseline_mask_mm=baseline_native_by_model_view[model_name],
                            baseline_confmap_mm=baseline_confmap_by_model_view[model_name],
                            M_out_to_native=tile_job.M_out_to_src,
                            native_h=view.src_h,
                            native_w=view.src_w,
                            min_conf=float(args.min_conf),
                            postprocess_workers=predict_postprocess_workers,
                        )
                        view_prediction_stats[view.name] = int(view_prediction_stats.get(view.name, 0)) + int(pred_stats.get('prediction_count', 0))
                        for key in tile_gate_stats_by_model[model_name].keys():
                            tile_gate_stats_by_model[model_name][key] += int(pred_stats.get(key, 0))

                    if not args.troubleshooting:
                        try:
                            tile_job.video_path.unlink(missing_ok=True)
                            tile_job.meta_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                for model_name in sorted(tile_gate_stats_by_model.keys()):
                    gate_stats = tile_gate_stats_by_model[model_name]
                    print(
                        f"Dense tiled gating {model_name}/{view.name}: "
                        f"predictions={gate_stats['prediction_count']}, "
                        f"accepted_masks={gate_stats['accepted_masks']}, "
                        f"rejected_masks={gate_stats['rejected_masks']}, "
                        f"rejected_min_conf={gate_stats['rejected_min_conf']}, "
                        f"gated_added_voxels={gate_stats['gated_added_voxels']}"
                    )
        finally:
            if tile_executor is not None:
                tile_executor.shutdown(wait=True)

        for model_name, _ in yolo_models:
            print(f"\n--- Finalizing view '{view.name}' for model '{model_name}' ---")
            baseline_native_volume = baseline_native_by_model_view.pop(model_name)
            try:
                support_mm_done = baseline_support_by_model_view.pop(model_name, None)
                close_memmap_array(support_mm_done)
                if not args.troubleshooting and model_name in baseline_support_paths:
                    try:
                        baseline_support_paths[model_name].unlink(missing_ok=True)
                    except Exception:
                        pass

                confmap_mm_done = baseline_confmap_by_model_view.pop(model_name, None)
                close_memmap_array(confmap_mm_done)
                if not args.troubleshooting:
                    try:
                        baseline_confmap_paths[model_name].unlink(missing_ok=True)
                    except Exception:
                        pass

                fill_2d_holes_volume_inplace(
                    baseline_native_volume,
                    workers=slice_postprocess_workers,
                )

                if view.family == 'radial':
                    out_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.u8.dat'
                    baseline_volume = backproject_radial_volume_to_volume(
                        radial_mask_mm=baseline_native_volume,
                        radial_view=view,
                        out_path=out_path,
                        desc=f'Backprojecting {model_name}/{view.name}',
                        prefer_memory=True,
                    )
                    close_memmap_array(baseline_native_volume)
                    if not args.troubleshooting:
                        try:
                            baseline_native_paths[model_name].unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    baseline_volume = baseline_native_volume

                if float(args.min_radius) > 0:
                    apply_view_min_radius_filter_inplace(
                        baseline_volume,
                        view,
                        float(args.min_radius),
                        workers=slice_postprocess_workers,
                    )

                view_volumes_by_model[model_name][view.name] = baseline_volume
            finally:
                baseline_native_by_model_view.pop(model_name, None)

        baseline_union_by_model_view.clear()
        baseline_confmap_by_model_view.clear()
        baseline_native_by_model_view.clear()
        baseline_support_by_model_view.clear()

        close_memmap_array(view_frame_cache)
        if not args.troubleshooting and view_frame_cache_path is not None:
            try:
                view_frame_cache_path.unlink(missing_ok=True)
            except Exception:
                pass

        gc.collect()

    view_prefetch_executor.shutdown(wait=True)

    gc.collect()
    gc.collect()

    def build_current_snapshot_volume(snapshot_stem: str, *, apply_void_fill: bool) -> np.ndarray:
        ensemble_mm = assemble_current_ensemble_volume(
            view_volumes_by_model=view_volumes_by_model,
            T=T,
            H=H,
            W=W,
            disable_multiplanar=not bool(args.enable_multiplanar),
            out_path=temp_dir / f'{snapshot_stem}.u8.dat',
            prefer_memory=True,
            workers=slice_postprocess_workers,
        )
        if bool(apply_void_fill):
            fill_3d_voids_inplace_streaming(
                ensemble_mm,
                temp_dir / f'{snapshot_stem}_voidfill',
                keep_temp=bool(args.troubleshooting),
                prefer_memory=True,
            )
        return ensemble_mm

    interpolation_stats: List[Dict[str, object]] = []

    if bool(args.troubleshooting) and int(args.interpolate) > 0:
        print('\n=== Scheduling troubleshooting outputs: pass 0 (pre-interpolation, pre-void-fill) ===')
        pass0_mm = build_current_snapshot_volume('ensemble_pass0', apply_void_fill=False)
        pass0_paths, pass0_futures = collect_pipeline_output_futures(
            output_manager.executor,
            volume_rgb=volume_rgb,
            mask_u8=pass0_mm,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_binary_pattern_value=args.save_binary,
            save_labels_pattern_value=args.save_labels,
            save_nrrd_flag=bool(args.save_nrrd),
            tag='Pass0',
            frame_workers=output_frame_workers,
            show_progress=False,
        )
        if bool(args.save_multiplanar):
            extra_paths, extra_futures = collect_multiplanar_output_futures(
                output_manager.executor,
                volume_rgb=volume_rgb,
                mask_u8=pass0_mm,
                out_dir=out_dir,
                stem=input_path.stem,
                fps=fps,
                save_binary_pattern_value=args.save_binary,
                save_labels_pattern_value=args.save_labels,
                tag='Pass0',
                frame_workers=output_frame_workers,
                show_progress=False,
            )
            pass0_paths.update(extra_paths)
            pass0_futures.extend(extra_futures)
        output_manager.submit(BackgroundOutputSubmission(
            label='Pass0 outputs',
            result_paths=pass0_paths,
            futures=pass0_futures,
            resources=[pass0_mm],
        ))
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
            output_manager.reap_completed()

            task_specs: List[Tuple[str, ViewInfo]] = [
                (model_name, view)
                for model_name in sorted(view_volumes_by_model.keys())
                for view in cartesian_views
            ]
            outer_interpolation_workers = min(int(interpolation_workers), max(1, len(task_specs)))
            per_task_interpolation_workers = max(
                1,
                min(16, _env_int('YOLO_TTA_INTERPOLATION_TASK_WORKERS', max(1, _cpu_count() // max(1, outer_interpolation_workers)))),
            )
            print(
                f"Interpolation pass worker layout: outer={outer_interpolation_workers}, "
                f"per-task={per_task_interpolation_workers}"
            )

            def _run_interpolation_task(task_index: int) -> Dict[str, object]:
                model_name, view = task_specs[int(task_index)]
                print(f"\n--- Interpolating model '{model_name}' view '{view.name}' ---")
                current_mm = view_volumes_by_model[model_name][view.name]
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
                    workers=per_task_interpolation_workers,
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

            current_pass_stats: List[Dict[str, object]] = []
            if outer_interpolation_workers > 1 and len(task_specs) > 1:
                for stats in parallel_map_in_order(
                    _run_interpolation_task,
                    range(len(task_specs)),
                    max_workers=outer_interpolation_workers,
                    max_pending=outer_interpolation_workers,
                ):
                    stats_dict = dict(stats)
                    interpolation_stats.append(stats_dict)
                    current_pass_stats.append(stats_dict)
            else:
                for task_idx in range(len(task_specs)):
                    stats_dict = dict(_run_interpolation_task(task_idx))
                    interpolation_stats.append(stats_dict)
                    current_pass_stats.append(stats_dict)

            added_this_pass = sum(int(s.get('added_voxels', 0)) for s in current_pass_stats)
            if int(added_this_pass) <= 0:
                print(
                    f"Interpolation terminated early after pass {pass_idx}: "
                    "no new voxels were added across any Cartesian view"
                )
                break

            if bool(args.troubleshooting) and pass_idx < total_passes:
                print(f"\n=== Scheduling troubleshooting outputs: pass {pass_idx} (pre-next-pass, pre-void-fill) ===")
                pass_mm = build_current_snapshot_volume(f'ensemble_pass{pass_idx}', apply_void_fill=False)
                pass_paths, pass_futures = collect_pipeline_output_futures(
                    output_manager.executor,
                    volume_rgb=volume_rgb,
                    mask_u8=pass_mm,
                    out_dir=out_dir,
                    stem=input_path.stem,
                    fps=fps,
                    save_binary_pattern_value=args.save_binary,
                    save_labels_pattern_value=args.save_labels,
                    save_nrrd_flag=bool(args.save_nrrd),
                    tag=f'Pass{pass_idx}',
                    frame_workers=output_frame_workers,
                    show_progress=False,
                )
                if bool(args.save_multiplanar):
                    extra_paths, extra_futures = collect_multiplanar_output_futures(
                        output_manager.executor,
                        volume_rgb=volume_rgb,
                        mask_u8=pass_mm,
                        out_dir=out_dir,
                        stem=input_path.stem,
                        fps=fps,
                        save_binary_pattern_value=args.save_binary,
                        save_labels_pattern_value=args.save_labels,
                        tag=f'Pass{pass_idx}',
                        frame_workers=output_frame_workers,
                        show_progress=False,
                    )
                    pass_paths.update(extra_paths)
                    pass_futures.extend(extra_futures)
                output_manager.submit(BackgroundOutputSubmission(
                    label=f'Pass{pass_idx} outputs',
                    result_paths=pass_paths,
                    futures=pass_futures,
                    resources=[pass_mm],
                ))
                del pass_mm
                gc.collect()
    else:
        for model_name in sorted(view_volumes_by_model.keys()):
            for view in cartesian_views:
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
                interpolation_stats.append(entry)

    output_manager.reap_completed()

    final_ensemble_mm = assemble_current_ensemble_volume(
        view_volumes_by_model=view_volumes_by_model,
        T=T,
        H=H,
        W=W,
        disable_multiplanar=not bool(args.enable_multiplanar),
        out_path=temp_dir / 'ensemble_volume_final.u8.dat',
        prefer_memory=True,
        workers=slice_postprocess_workers,
    )

    print('\n=== 3D void fill after multiplanar/model union ===')
    fill_3d_voids_inplace_streaming(
        final_ensemble_mm,
        temp_dir / 'ensemble_volume_final_voidfill',
        keep_temp=bool(args.troubleshooting),
        prefer_memory=True,
    )

    print('\n=== Scheduling final outputs in background ===')
    final_paths, final_futures = collect_pipeline_output_futures(
        output_manager.executor,
        volume_rgb=volume_rgb,
        mask_u8=final_ensemble_mm,
        out_dir=out_dir,
        stem=input_path.stem,
        fps=fps,
        save_binary_pattern_value=args.save_binary,
        save_labels_pattern_value=args.save_labels,
        save_nrrd_flag=bool(args.save_nrrd),
        tag=None,
        frame_workers=output_frame_workers,
        show_progress=False,
    )
    if bool(args.save_multiplanar):
        extra_paths, extra_futures = collect_multiplanar_output_futures(
            output_manager.executor,
            volume_rgb=volume_rgb,
            mask_u8=final_ensemble_mm,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_binary_pattern_value=args.save_binary,
            save_labels_pattern_value=args.save_labels,
            tag=None,
            frame_workers=output_frame_workers,
            show_progress=False,
        )
        final_paths.update(extra_paths)
        final_futures.extend(extra_futures)
    output_manager.submit(BackgroundOutputSubmission(
        label='final outputs',
        result_paths=final_paths,
        futures=final_futures,
        resources=[],
    ))

    voxel_volume = None
    if bool(args.voxel_volume):
        voxel_counts = np.zeros((int(final_ensemble_mm.shape[0]),), dtype=np.int64)

        def _count_voxels(z: int) -> None:
            voxel_counts[int(z)] = np.int64(np.count_nonzero(np.asarray(final_ensemble_mm[int(z)])))

        parallel_for_indices(
            int(final_ensemble_mm.shape[0]),
            _count_voxels,
            max_workers=choose_slice_parallel_workers(int(slice_postprocess_workers), int(final_ensemble_mm.shape[0])),
            desc='Counting voxel_volume',
        )
        voxel_volume = int(np.sum(voxel_counts, dtype=np.int64))

    output_manager.wait()

    if bool(args.save_images):
        print('\n=== Saving active-view image sequences ===')
        for view in views:
            image_dir = write_view_images(
                volume_rgb=volume_rgb,
                view=view,
                out_dir=out_dir,
                stem=input_path.stem,
                workers=output_frame_workers,
                show_progress=False,
            )
            final_paths[f'{view.name}_images_dir'] = image_dir

    if bool(args.save_radial):
        radial_view = next((v for v in views if v.name == 'radial'), None)
        if radial_view is None:
            print('Warning: --save_radial requested but --azimuth_angle is 0; skipping radial outputs')
        else:
            print('\n=== Saving radial outputs ===')
            final_paths.update(write_additional_view_outputs(
                volume_rgb=volume_rgb,
                mask_u8=final_ensemble_mm,
                view=radial_view,
                out_dir=out_dir,
                stem=input_path.stem,
                fps=fps,
                save_binary_pattern_value=args.save_binary,
                save_labels_pattern_value=args.save_labels,
                tag=None,
                workers=output_frame_workers,
                show_progress=False,
            ))

    if bool(args.save_TTA):
        print('\n=== Saving TTA videos and mapped labels ===')
        final_paths.update(save_tta_outputs(
            view_volumes_by_model=view_volumes_by_model,
            views=views,
            aug_jobs_by_view=aug_jobs_by_view,
            temp_dir=temp_dir,
            out_dir=out_dir,
            stem=input_path.stem,
            workers=output_frame_workers,
            show_progress=False,
        ))

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
        slice_postprocess_workers=slice_postprocess_workers,
        interpolation_workers=interpolation_workers,
        output_workers=output_workers,
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
