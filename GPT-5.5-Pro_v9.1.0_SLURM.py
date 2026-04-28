#!/usr/bin/env python3
"""
YOLO segmentation test-time augmentation (TTA) for large cylindrical video volumes.

This v9.1.0_SLURM single-channel-aligned script:
  - builds Transverse, optional Tilted Transverse, optional Sagittal/Coronal, and optional Radial view families using single-channel intermediates
  - renders parent/full-frame videos directly from the native view transform so inference no longer waits on
    a separately rendered canvas video, and derives non-radial tile videos from shared canvas batches so the
    same rotated frames are not re-decoded once per tile
  - keeps parent/full-frame videos and non-radial tile batches rendering concurrently so the GPU can consume
    whichever inference job becomes ready first, while still preferring ready parent/full-frame jobs over tile jobs
  - keeps tile acceptance mask-wise gated against the frozen parent full-frame support, consolidates
    accepted tile masks per parent view, and interpolates the consolidated gated-tile volume once per parent view
  - removes dense-tile pruning and keeps temporary/intermediate mask volumes unpacked throughout
  - fuses per-slice cleanup work where the slice orientation matches the required semantics
    (notably min_conf filtering, 2D hole filling, and min_radius where applicable)
  - overlaps GPU inference with CPU-side view interpolation, consolidated-tile interpolation, and output writing
  - reuses a native radial frame cache during dense tiled rendering so radial tiles do not recompute
    the same Lanczos-5 slices for every tile location
  - inverse-maps predictions only into each generated video's native view space, keeps Radial and Tilted
    Transverse results view-native through cleanup/interpolation, then backprojects them after per-view processing
  - treats --model as a single YOLO segmentation model path; multiple-model inference is not supported in v9.1.0
  - applies final 3D void fill only when --enable_3d_void_fill is active, and only once after the global union
  - applies --object_interpolation_smoothing as independent per-view frozen-state smoothing jobs
    (Transverse, and Sagittal/Coronal when multiplanar inference is enabled) before unioning their
    smoothed deltas back into the final native volume
  - supports v9.1.0 Radial and Tilted Transverse view-native interpolation, and expands Tilted
    Transverse frame counts according to tilt angle so edge frames are represented with black padding
  - saves the transverse default color overlay plus optional labels, single-channel binary masks, NRRD, multiplanar, radial,
    tilted-transverse, image-sequence, low-quality, and TTA outputs, with FFV1 used for spec-required MKVs

Dependencies (Python):
  pip install opencv-python numpy scipy scikit-image tifffile tqdm ultralytics
  pip install pynrrd   # only needed for --save_nrrd

System:
  ffmpeg + ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import math
import os
import re
import shlex
import struct
import shutil
import subprocess
import sys
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
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


def _parse_token_list(values: Sequence[str] | str | None) -> List[str]:
    """Accept comma and/or whitespace separated string tokens."""
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else [str(v) for v in values]
    parts: List[str] = []
    for raw in raw_values:
        raw = str(raw).strip()
        if not raw:
            continue
        parts.extend([p for p in re.split(r"[,\s]+", raw) if p])
    return parts


def resolve_tilt_angles(values: Sequence[str] | str | None) -> List[float]:
    raw = _parse_angles(' '.join(_parse_token_list(values)))
    out: List[float] = []
    seen: set[float] = set()
    for angle in raw:
        angle_f = float(angle)
        if abs(angle_f) <= 0.0:
            continue
        angle_pos = abs(angle_f)
        if angle_pos in seen:
            continue
        seen.add(angle_pos)
        out.append(angle_pos)
    return out


def resolve_tilt_directions(values: Sequence[str] | str | None) -> List[str]:
    raw = [str(v).strip().lower() for v in _parse_token_list(values)]
    out: List[str] = []
    for token in raw:
        if token == 'both':
            for expanded in ('vertical', 'horizontal'):
                if expanded not in out:
                    out.append(expanded)
            continue
        if token not in ('vertical', 'horizontal'):
            raise ValueError("--tilt_direction values must be vertical, horizontal, or both")
        if token not in out:
            out.append(token)
    if not out:
        out = ['vertical']
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLO segmentation TTA for large cylindrical video volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", required=True, type=str, help="Input video path")
    p.add_argument("--output", default=None, type=str, help="Output directory (default ./{Filename}/)")
    p.add_argument("--device", default="0", type=str, help="Device passed to YOLO predict")
    p.add_argument("--model", required=True, type=str, help="Path to a single YOLO segmentation model")

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
    p.add_argument("--keep_objects", default=0, type=int,
                   help="Keep the top N largest final 3D objects by volume. 0 keeps all objects")

    p.add_argument("--enable_multiplanar", action="store_true",
                   help="Enable Sagittal and Coronal Cartesian views in addition to the required Transverse view")
    p.add_argument("--azimuth_angle", default=0.0, type=float,
                   help="Angular spacing in degrees for radial diameter slices over [0,180]. 0 disables radial views")
    p.add_argument("--tilt_angle", nargs="+", default=["0"], type=str,
                   help="One or more positive Tilted Transverse angles in degrees. Each value creates both positive and negative variants. 0 disables tilted transverse views")
    p.add_argument("--tilt_direction", nargs="+", default=["vertical"], type=str,
                   help="One or more Tilted Transverse directions: vertical, horizontal, or both")

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
    p.add_argument("--save_tilted_transverse", action="store_true",
                   help="Save additional Tilted Transverse outputs. Filenames encode the direction and signed tilt angle. A warning is emitted and nothing is saved when --tilt_angle is 0")
    p.add_argument("--save_low_quality", action="store_true",
                   help="Save additional low-quality presentation copies using libx264, preset slow, yuv420p, and a 1024px maximum dimension")
    p.add_argument("--voxel_volume", action="store_true", help="Count white voxels in the final binary output and save the value to the summary text file")
    p.add_argument("--enable_3d_void_fill", action="store_true",
                   help="Apply one final 3D enclosed-void fill after the global union. Disabled by default")
    p.add_argument("--object_interpolation_smoothing", action="store_true",
                   help="Apply frozen-state object mask interpolation smoothing independently per eligible view, then union the smoothed deltas")

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
    """Intentional CPU oversubscription for mixed GPU/IO/CPU workloads.

    The v9.1.0 SLURM target has enough CPU headroom that running roughly 2x the visible CPU count
    helps keep ffmpeg, view rendering, interpolation planning, and output writers busy while the GPU
    is inferencing or waiting on a different stage.
    """
    return max(1, int(_cpu_count()) * 2)


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



def choose_parallel_chunk_size(
    total_items: int,
    max_workers: int,
    *,
    target_chunks_per_worker: int = 4,
    min_chunk_size: int = 1,
    max_chunk_size: Optional[int] = None,
) -> int:
    total = max(0, int(total_items))
    workers = max(1, int(max_workers))
    if total <= 0:
        return max(1, int(min_chunk_size))

    denom = max(1, workers * max(1, int(target_chunks_per_worker)))
    chunk = max(int(min_chunk_size), int(math.ceil(float(total) / float(denom))))
    if max_chunk_size is not None:
        chunk = min(int(chunk), max(1, int(max_chunk_size)))
    return max(1, int(chunk))



def parallel_for_indices_chunked(
    count: int,
    func: Callable[[int], None],
    *,
    max_workers: int,
    desc: str,
    show_progress: bool = True,
    chunk_size: Optional[int] = None,
    target_chunks_per_worker: int = 4,
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

    if chunk_size is None or int(chunk_size) <= 0:
        chunk = choose_parallel_chunk_size(
            total,
            workers,
            target_chunks_per_worker=int(target_chunks_per_worker),
            min_chunk_size=1,
        )
    else:
        chunk = max(1, int(chunk_size))

    ranges = [(int(start), int(min(total, start + chunk))) for start in range(0, total, chunk)]

    def _run_range(range_idx: int) -> int:
        start, stop = ranges[int(range_idx)]
        for idx in range(int(start), int(stop)):
            func(int(idx))
        return int(stop - start)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_range, int(range_idx)) for range_idx in range(len(ranges))]
        if show_progress:
            with tqdm(total=total, desc=desc) as pbar:
                for fut in as_completed(futures):
                    pbar.update(int(fut.result()))
        else:
            for fut in as_completed(futures):
                fut.result()



def workspace_anon_cap_bytes() -> int:
    """Return the optional anonymous-workspace cap.

    Task overrides remove the previous conservative default fractional cap. Anonymous workspaces are
    only capped when the user explicitly sets YOLO_TTA_MAX_ANON_WORKSPACE_GIB.
    """
    hard_cap_gib = max(0.0, _env_float('YOLO_TTA_MAX_ANON_WORKSPACE_GIB', 0.0))
    if hard_cap_gib <= 0.0:
        return 0
    return int(hard_cap_gib * GIB)


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










def choose_slice_parallel_workers(requested_workers: int, num_items: int) -> int:
    return max(1, min(int(requested_workers), int(max(1, num_items))))




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


def decode_video_to_memmap_gray8(
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
    """Decode input video to a (T,H,W) uint8 gray/luma workspace.

    ffmpeg's ``gray`` pixel format accepts RGB, YUV, and already single-channel inputs and
    normalizes them to one luma channel. This keeps the source volume and all view-rendering
    intermediates single-channel; color expansion is deferred to optional presentation outputs.
    """
    _require_bin("ffmpeg")

    shape = (int(num_frames), int(height), int(width))
    reuse_existing = bool(not overwrite and out_dat.exists() and not prefer_memory)
    arr = allocate_workspace_array(
        shape=shape,
        dtype=np.uint8,
        path=out_dat,
        desc='Decoded gray8 input volume',
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

    frame_bytes = int(width) * int(height)
    chunk_frames = max(1, min(128, max(1, (256 * 1024 * 1024) // max(1, frame_bytes))))

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(input_video),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-vsync", "0",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    try:
        with tqdm(total=num_frames, desc='Decoding input volume (gray8)') as pbar:
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
    pix_fmt_in: str = "gray",
    codec: str = "ffv1",
    pix_fmt_out: Optional[str] = "gray",
    codec_args: Optional[Sequence[str]] = None,
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
    ]

    if str(codec) == 'ffv1':
        cmd.extend(["-c:v", "ffv1", "-level", "3", "-slices", "30", "-threads", "4"])
    else:
        cmd.extend(["-c:v", str(codec)])

    if codec_args:
        cmd.extend([str(x) for x in codec_args])

    if pix_fmt_out:
        cmd.extend(["-pix_fmt", str(pix_fmt_out)])

    cmd.append(str(out_path))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    return proc


def ffmpeg_lossless_rgb_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    """Lossless MKV writer using libx264rgb -preset ultrafast -qp 0."""
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='rgb24',
        codec='libx264rgb',
        pix_fmt_out='rgb24',
        codec_args=['-preset', 'ultrafast', '-qp', '0'],
    )



def ffmpeg_ffv1_gray_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    """FFV1 MKV writer for single-channel temporary, prediction, and binary videos."""
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='gray',
        codec='ffv1',
        pix_fmt_out='gray',
    )


def ffmpeg_ffv1_rgb_writer(
    out_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    """FFV1 MKV writer for color presentation overlays."""
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='rgb24',
        codec='ffv1',
        pix_fmt_out='yuv444p',
    )


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
    canvas_video_path: Path
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
    summary_family: str = 'transverse'
    display_name: str = 'Transverse'
    azimuths_deg: Tuple[float, ...] = ()
    diameter: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    roi_radius: float = 0.0
    full_t: int = 0
    full_h: int = 0
    full_w: int = 0
    tilt_angle_deg: float = 0.0
    tilt_direction: str = ''
    tilt_frame_start: int = 0
    tilt_frame_stop: int = 0


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


def compute_tilt_frame_range(
    t_dim: int,
    axis_len: int,
    signed_tilt_angle_deg: float,
) -> Tuple[int, int]:
    """Return inclusive native-center frame range for a Tilted Transverse stack.

    A tilted frame centered at native slice coordinate ``N`` samples
    ``t = N + tan(alpha) * (axis - center)``.  v9.1.0 requires the generated
    tilted video to include the incomplete edge frames introduced by this shear,
    rather than forcing the frame count to remain equal to the input transverse
    frame count.  We therefore include every integer center coordinate whose
    tilted plane intersects the native t-domain [0, t_dim - 1].
    """
    t_dim_i = int(t_dim)
    if t_dim_i <= 0:
        return 0, -1

    axis_len_i = max(1, int(axis_len))
    tan_alpha = float(math.tan(math.radians(float(signed_tilt_angle_deg))))
    center = float((axis_len_i - 1) / 2.0)
    off0 = tan_alpha * (0.0 - center)
    off1 = tan_alpha * (float(axis_len_i - 1) - center)
    min_off = min(float(off0), float(off1))
    max_off = max(float(off0), float(off1))

    start = int(math.ceil(-max_off - 1e-9))
    stop = int(math.floor(float(t_dim_i - 1) - min_off + 1e-9))
    if stop < start:
        return 0, -1
    return int(start), int(stop)


def tilted_frame_center(view: ViewInfo, frame_idx: int) -> int:
    return int(view.tilt_frame_start) + int(frame_idx)


def _format_signed_angle_token(angle_deg: float) -> str:
    sign = 'p' if float(angle_deg) >= 0.0 else 'm'
    mag = f'{abs(float(angle_deg)):g}'.replace('.', 'p')
    return f'{sign}{mag}'


def _format_signed_angle_label(angle_deg: float) -> str:
    sign = '+' if float(angle_deg) >= 0.0 else '-'
    return f'{sign}{abs(float(angle_deg)):g}°'


def pretty_view_name(view: ViewInfo) -> str:
    return str(view.display_name)


def view_output_token(view: ViewInfo) -> str:
    if view.family == 'tilted_transverse':
        direction = str(view.tilt_direction or 'vertical').capitalize()
        return f'TiltedTransverse_{direction}_{_format_signed_angle_token(float(view.tilt_angle_deg))}'
    return pretty_view_name(view).replace(' ', '_')


def get_view_infos(
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
    azimuth_angle: float = 0.0,
    include_radial: bool = True,
    tilt_angles: Optional[Sequence[float]] = None,
    tilt_directions: Optional[Sequence[str]] = None,
) -> List[ViewInfo]:
    views = [
        ViewInfo(
            name='transverse',
            num_slices=T,
            src_h=H,
            src_w=W,
            pad_mode='clamp',
            family='orthogonal',
            summary_family='transverse',
            display_name='Transverse',
            full_t=T,
            full_h=H,
            full_w=W,
        ),
    ]
    if not disable_multiplanar:
        views.append(ViewInfo(
            name='sagittal',
            num_slices=H,
            src_h=T,
            src_w=W,
            pad_mode='pad',
            family='orthogonal',
            summary_family='sagittal',
            display_name='Sagittal',
            full_t=T,
            full_h=H,
            full_w=W,
        ))
        views.append(ViewInfo(
            name='coronal',
            num_slices=W,
            src_h=T,
            src_w=H,
            pad_mode='pad',
            family='orthogonal',
            summary_family='coronal',
            display_name='Coronal',
            full_t=T,
            full_h=H,
            full_w=W,
        ))

    tilt_angles_resolved = [float(a) for a in (tilt_angles or []) if float(a) > 0.0]
    tilt_dirs_resolved = [str(v) for v in (tilt_directions or [])]
    for tilt_direction in tilt_dirs_resolved:
        for tilt_angle in tilt_angles_resolved:
            for sign in (+1.0, -1.0):
                signed_angle = float(sign * tilt_angle)
                token = _format_signed_angle_token(signed_angle)
                axis_len = int(H) if str(tilt_direction) == 'vertical' else int(W)
                tilt_frame_start, tilt_frame_stop = compute_tilt_frame_range(
                    t_dim=int(T),
                    axis_len=int(axis_len),
                    signed_tilt_angle_deg=float(signed_angle),
                )
                views.append(ViewInfo(
                    name=f'tilted_transverse_{tilt_direction}_{token}',
                    num_slices=max(0, int(tilt_frame_stop) - int(tilt_frame_start) + 1),
                    src_h=H,
                    src_w=W,
                    pad_mode='clamp',
                    family='tilted_transverse',
                    summary_family='tilted_transverse',
                    display_name=f'Tilted Transverse {tilt_direction} {_format_signed_angle_label(signed_angle)}',
                    full_t=T,
                    full_h=H,
                    full_w=W,
                    tilt_angle_deg=signed_angle,
                    tilt_direction=str(tilt_direction),
                    tilt_frame_start=int(tilt_frame_start),
                    tilt_frame_stop=int(tilt_frame_stop),
                ))

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
                summary_family='radial',
                display_name='Radial',
                full_t=T,
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

    bytes_per_frame = max(1, int(diameter) * 10 * np.dtype(np.float32).itemsize)
    block = max(1, int(target_bytes // bytes_per_frame))
    return max(1, min(256, block))



def extract_radial_slice_frame(volume_rgb: np.ndarray, sampler: RadialSampler) -> np.ndarray:
    t_dim = int(volume_rgb.shape[0])
    out = np.empty((t_dim, sampler.diameter), dtype=np.uint8)

    x_w = np.asarray(sampler.x_w, dtype=np.float32)[None, :, :]
    y_w = np.asarray(sampler.y_w, dtype=np.float32)
    block_frames = choose_radial_exact_block_frames(sampler.diameter)

    for start in range(0, t_dim, block_frames):
        stop = min(t_dim, start + block_frames)
        block = np.asarray(volume_rgb[start:stop])
        acc = np.zeros((stop - start, sampler.diameter), dtype=np.float32)
        for yi in range(sampler.y_idx.shape[1]):
            samples = block[:, sampler.y_idx[:, yi][:, None], sampler.x_idx].astype(np.float32, copy=False)
            row = np.sum(samples * x_w, axis=2)
            acc += row * y_w[:, yi][None, :]
        out[start:stop, :] = np.clip(np.rint(acc), 0.0, 255.0).astype(np.uint8)

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
                'num_slices': int(view.num_slices),
                'full_t': int(view.full_t),
                'angle_deg': float(job.angle_deg),
                'src_w': view.src_w,
                'src_h': view.src_h,
                'tilt_angle_deg': float(view.tilt_angle_deg),
                'tilt_direction': str(view.tilt_direction),
                'tilt_frame_start': int(view.tilt_frame_start),
                'tilt_frame_stop': int(view.tilt_frame_stop),
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
                canvas_video_path=aug_dir / f'{view.name}_{aug_id}.canvas.mkv',
                video_path=aug_dir / f'{view.name}_{aug_id}.mkv',
                meta_path=aug_dir / f'{view.name}_{aug_id}.meta.json',
                aff=aff,
            )
        )
    return jobs


def iter_aug_jobs_round_robin(
    views: Sequence[ViewInfo],
    aug_jobs_by_view: Dict[str, Sequence[AugJob]],
) -> Iterator[Tuple[ViewInfo, AugJob]]:
    """Yield augmentation jobs one round per view so later view families start earlier.

    The prior FIFO submission order rendered every augmentation for early views before later
    views even entered the render queue. That could leave later families, notably tilted
    transverse variants, sitting behind a long tail of earlier canvas/tile work even while the GPU
    had no ready inference job. Round-robin submission starts one job per active view first, then
    the second job per view, and so on.
    """
    queues: Dict[str, deque[AugJob]] = {
        str(view.name): deque(aug_jobs_by_view.get(view.name, ()))
        for view in views
    }
    while True:
        emitted = False
        for view in views:
            q = queues[str(view.name)]
            if not q:
                continue
            emitted = True
            yield view, q.popleft()
        if not emitted:
            break


def split_video_render_workers(total_workers: int, total_fullframe_tasks: int, total_tile_tasks: int) -> Tuple[int, int]:
    """Reserve render capacity for full-frame jobs so later parents do not queue behind tiles.

    Returns ``(fullframe_workers, tile_workers)``. Tile rendering gets a smaller share because the
    GPU scheduler already prefers ready full-frame videos over ready tile videos, and keeping parent
    rendering moving prevents the GPU from stalling while it waits for later views to appear.
    """
    total_workers = max(1, int(total_workers))
    total_fullframe_tasks = max(0, int(total_fullframe_tasks))
    total_tile_tasks = max(0, int(total_tile_tasks))

    if total_tile_tasks <= 0:
        return max(1, min(total_workers, max(1, total_fullframe_tasks))), 0

    if total_workers <= 1:
        return 1, 0

    tile_workers = min(total_tile_tasks, max(1, total_workers // 3))
    fullframe_workers = max(1, total_workers - tile_workers)

    if total_fullframe_tasks > 0:
        fullframe_workers = min(fullframe_workers, total_fullframe_tasks)
    else:
        fullframe_workers = 0

    tile_workers = total_workers - fullframe_workers
    if tile_workers <= 0 and total_tile_tasks > 0:
        tile_workers = 1
        fullframe_workers = max(1, total_workers - tile_workers)

    if total_fullframe_tasks > 0 and fullframe_workers <= 0:
        fullframe_workers = 1
        tile_workers = max(0, total_workers - fullframe_workers)

    return int(fullframe_workers), int(min(tile_workers, total_tile_tasks))


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


@dataclass(frozen=True)
class TiltedRenderPlan:
    x_idx: np.ndarray
    y_idx: np.ndarray
    valid_xy: np.ndarray
    axis_offset: np.ndarray


_TILTED_RENDER_PLAN_CACHE: Dict[Tuple[str, int, int, Tuple[float, ...]], TiltedRenderPlan] = {}


def ffmpeg_rawvideo_reader(
    input_path: Path,
    width: int,
    height: int,
    pix_fmt: str = 'gray',
) -> subprocess.Popen:
    _require_bin('ffmpeg')
    cmd = [
        'ffmpeg',
        '-v', 'error',
        '-i', str(input_path),
        '-f', 'rawvideo',
        '-pix_fmt', pix_fmt,
        '-vsync', '0',
        '-',
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    return proc


def iter_ffmpeg_gray8_frames(
    input_path: Path,
    width: int,
    height: int,
    num_frames: int,
) -> Iterator[np.ndarray]:
    proc = ffmpeg_rawvideo_reader(input_path, int(width), int(height), pix_fmt='gray')
    frame_bytes = int(width) * int(height)
    assert proc.stdout is not None
    try:
        for idx in range(int(num_frames)):
            buf = bytearray(frame_bytes)
            mv = memoryview(buf)
            filled = 0
            while filled < frame_bytes:
                nread = proc.stdout.readinto(mv[filled:])
                if nread is None or nread <= 0:
                    raise RuntimeError(f'Unexpected EOF while decoding {input_path} at frame {idx}/{num_frames}')
                filled += int(nread)
            yield np.frombuffer(buf, dtype=np.uint8).reshape((int(height), int(width))).copy()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        _, err = proc.communicate()
        if proc.returncode not in (0, None):
            msg = err.decode('utf-8', errors='ignore') if isinstance(err, (bytes, bytearray)) else str(err)
            raise RuntimeError(f'ffmpeg read failed: {msg}')


def _extract_padded_tile_frame(frame: np.ndarray, x0: int, y0: int, tile_size: int) -> np.ndarray:
    tile = np.zeros((int(tile_size), int(tile_size)), dtype=frame.dtype)
    src_x0 = max(0, int(x0))
    src_y0 = max(0, int(y0))
    src_x1 = min(int(frame.shape[1]), int(x0) + int(tile_size))
    src_y1 = min(int(frame.shape[0]), int(y0) + int(tile_size))
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return tile
    dst_x0 = src_x0 - int(x0)
    dst_y0 = src_y0 - int(y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    tile[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return tile


def _resize_frame_centered(frame: np.ndarray, out_w: int, out_h: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    scale_M = _center_preserving_scale_matrix(int(frame.shape[1]), int(frame.shape[0]), int(out_w), int(out_h))
    return cv2.warpAffine(
        np.ascontiguousarray(frame),
        scale_M[:2, :].astype(np.float32),
        dsize=(int(out_w), int(out_h)),
        flags=int(interpolation),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _tilted_plan_cache_key(view: ViewInfo, M_grid_to_src: np.ndarray, grid_h: int, grid_w: int) -> Tuple[str, int, int, Tuple[float, ...]]:
    mat = tuple(round(float(x), 6) for x in np.asarray(M_grid_to_src, dtype=np.float32).reshape(-1).tolist())
    return (str(view.name), int(grid_h), int(grid_w), mat)


def get_tilted_render_plan(
    view: ViewInfo,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> TiltedRenderPlan:
    if view.family != 'tilted_transverse':
        raise ValueError('Tilted render plan requested for a non-tilted view')

    key = _tilted_plan_cache_key(view, M_grid_to_src, int(grid_h), int(grid_w))
    cached = _TILTED_RENDER_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    yy, xx = np.indices((int(grid_h), int(grid_w)), dtype=np.float32)
    M = np.asarray(M_grid_to_src, dtype=np.float32)
    src_x = (M[0, 0] * xx) + (M[0, 1] * yy) + M[0, 2]
    src_y = (M[1, 0] * xx) + (M[1, 1] * yy) + M[1, 2]

    x_nn = np.rint(src_x).astype(np.int32, copy=False)
    y_nn = np.rint(src_y).astype(np.int32, copy=False)
    valid_xy = (
        (x_nn >= 0) & (x_nn < int(view.full_w)) &
        (y_nn >= 0) & (y_nn < int(view.full_h))
    )
    x_idx = np.clip(x_nn, 0, int(view.full_w) - 1).astype(np.int32, copy=False)
    y_idx = np.clip(y_nn, 0, int(view.full_h) - 1).astype(np.int32, copy=False)

    if str(view.tilt_direction) == 'vertical':
        axis_offset = src_y - float((view.full_h - 1) / 2.0)
    elif str(view.tilt_direction) == 'horizontal':
        axis_offset = src_x - float((view.full_w - 1) / 2.0)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported tilt direction: {view.tilt_direction}')

    plan = TiltedRenderPlan(
        x_idx=x_idx,
        y_idx=y_idx,
        valid_xy=valid_xy.astype(bool, copy=False),
        axis_offset=np.asarray(axis_offset, dtype=np.float32),
    )
    _TILTED_RENDER_PLAN_CACHE[key] = plan
    return plan


def render_tilted_frame_on_grid(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
    block_rows: int = 256,
) -> np.ndarray:
    plan = get_tilted_render_plan(view, M_grid_to_src, int(grid_h), int(grid_w))
    tan_alpha = float(math.tan(math.radians(float(view.tilt_angle_deg))))
    t_dim = int(volume_rgb.shape[0])
    out = np.zeros((int(grid_h), int(grid_w)), dtype=np.uint8)

    for y0 in range(0, int(grid_h), int(block_rows)):
        y1 = min(int(grid_h), y0 + int(block_rows))
        valid_xy = np.asarray(plan.valid_xy[y0:y1], dtype=bool)
        if not np.any(valid_xy):
            continue

        frame_center = float(tilted_frame_center(view, int(frame_idx)))
        t_src = frame_center + tan_alpha * np.asarray(plan.axis_offset[y0:y1], dtype=np.float32)
        valid = valid_xy & (t_src >= 0.0) & (t_src <= float(t_dim - 1))
        if not np.any(valid):
            continue

        t0 = np.floor(t_src).astype(np.int32, copy=False)
        t1 = np.clip(t0 + 1, 0, t_dim - 1).astype(np.int32, copy=False)
        t0 = np.clip(t0, 0, t_dim - 1).astype(np.int32, copy=False)
        alpha = (t_src - t0).astype(np.float32, copy=False)

        ys = np.asarray(plan.y_idx[y0:y1], dtype=np.int32)
        xs = np.asarray(plan.x_idx[y0:y1], dtype=np.int32)
        f0 = np.asarray(volume_rgb[t0, ys, xs], dtype=np.float32)
        f1 = np.asarray(volume_rgb[t1, ys, xs], dtype=np.float32)
        blend = np.clip(np.rint(((1.0 - alpha) * f0) + (alpha * f1)), 0.0, 255.0).astype(np.uint8)
        out_block = out[y0:y1]
        out_block[valid] = blend[valid]

    return out


def render_tilted_mask_on_grid(
    volume_mask: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
    block_rows: int = 256,
) -> np.ndarray:
    plan = get_tilted_render_plan(view, M_grid_to_src, int(grid_h), int(grid_w))
    tan_alpha = float(math.tan(math.radians(float(view.tilt_angle_deg))))
    t_dim = int(volume_mask.shape[0])
    out = np.zeros((int(grid_h), int(grid_w)), dtype=np.uint8)

    for y0 in range(0, int(grid_h), int(block_rows)):
        y1 = min(int(grid_h), y0 + int(block_rows))
        valid_xy = np.asarray(plan.valid_xy[y0:y1], dtype=bool)
        if not np.any(valid_xy):
            continue

        frame_center = float(tilted_frame_center(view, int(frame_idx)))
        t_src = frame_center + tan_alpha * np.asarray(plan.axis_offset[y0:y1], dtype=np.float32)
        valid = valid_xy & (t_src >= 0.0) & (t_src <= float(t_dim - 1))
        if not np.any(valid):
            continue

        t0 = np.floor(t_src).astype(np.int32, copy=False)
        t1 = np.clip(t0 + 1, 0, t_dim - 1).astype(np.int32, copy=False)
        t0 = np.clip(t0, 0, t_dim - 1).astype(np.int32, copy=False)
        alpha = (t_src - t0).astype(np.float32, copy=False)

        ys = np.asarray(plan.y_idx[y0:y1], dtype=np.int32)
        xs = np.asarray(plan.x_idx[y0:y1], dtype=np.int32)
        f0 = np.asarray(volume_mask[t0, ys, xs], dtype=np.float32)
        f1 = np.asarray(volume_mask[t1, ys, xs], dtype=np.float32)
        blend = (((1.0 - alpha) * f0) + (alpha * f1) >= 0.5).astype(np.uint8)
        out_block = out[y0:y1]
        out_block[valid] = blend[valid]

    return out


def render_tilted_canvas_frame(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    aff: AffineSpec,
    block_rows: int = 256,
) -> np.ndarray:
    return render_tilted_frame_on_grid(
        volume_rgb=volume_rgb,
        view=view,
        frame_idx=int(frame_idx),
        M_grid_to_src=aff.M_canvas_to_src,
        grid_h=int(aff.canvas_h),
        grid_w=int(aff.canvas_w),
        block_rows=int(block_rows),
    )


def render_tilted_mask_canvas_frame(
    volume_mask: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    aff: AffineSpec,
    block_rows: int = 256,
) -> np.ndarray:
    return render_tilted_mask_on_grid(
        volume_mask=volume_mask,
        view=view,
        frame_idx=int(frame_idx),
        M_grid_to_src=aff.M_canvas_to_src,
        grid_h=int(aff.canvas_h),
        grid_w=int(aff.canvas_w),
        block_rows=int(block_rows),
    )


_TILTED_NATIVE_AFFINE_CACHE: Dict[Tuple[str, int, int], AffineSpec] = {}


def get_tilted_native_affine(view: ViewInfo) -> AffineSpec:
    if view.family != 'tilted_transverse':
        raise ValueError('Tilted native affine requested for a non-tilted view')
    key = (str(view.name), int(view.src_w), int(view.src_h))
    cached = _TILTED_NATIVE_AFFINE_CACHE.get(key)
    if cached is not None:
        return cached
    aff = build_affine(
        view=view.name,
        src_w=int(view.src_w),
        src_h=int(view.src_h),
        out_size=max(int(view.src_w), int(view.src_h), 1),
        angle_deg=0.0,
        pad_mode='clamp',
    )
    _TILTED_NATIVE_AFFINE_CACHE[key] = aff
    return aff


def render_tilted_native_frame(volume_rgb: np.ndarray, view: ViewInfo, frame_idx: int) -> np.ndarray:
    aff = get_tilted_native_affine(view)
    return render_tilted_canvas_frame(volume_rgb, view, int(frame_idx), aff)


def render_tilted_native_mask_frame(mask_u8: np.ndarray, view: ViewInfo, frame_idx: int) -> np.ndarray:
    aff = get_tilted_native_affine(view)
    return render_tilted_mask_canvas_frame(mask_u8, view, int(frame_idx), aff)


def render_canvas_frame_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    frame_idx: int,
    view_frames: Optional[np.ndarray] = None,
) -> np.ndarray:
    if view.family == 'tilted_transverse':
        return render_tilted_canvas_frame(
            volume_rgb=volume_rgb,
            view=view,
            frame_idx=int(frame_idx),
            aff=job.aff,
        )

    native_frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(frame_idx), view_frames=view_frames))
    return cv2.warpAffine(
        native_frame,
        job.aff.M_src_to_canvas,
        dsize=(int(job.aff.canvas_w), int(job.aff.canvas_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def render_canvas_video_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    fps: float,
    *,
    view_frames: Optional[np.ndarray] = None,
    workers: int = 1,
    show_progress: bool = True,
) -> None:
    proc = ffmpeg_ffv1_gray_writer(
        job.canvas_video_path,
        width=int(job.aff.canvas_w),
        height=int(job.aff.canvas_h),
        fps=float(fps),
    )
    try:
        assert proc.stdin is not None

        def _render(idx: int) -> np.ndarray:
            return render_canvas_frame_for_job(
                volume_rgb=volume_rgb,
                view=view,
                job=job,
                frame_idx=int(idx),
                view_frames=view_frames,
            )

        pending = min(int(view.num_slices), max(2, int(workers) * 2))
        iterable = parallel_map_in_order(_render, range(int(view.num_slices)), max_workers=max(1, int(workers)), max_pending=pending)
        for frame in tqdm(
            iterable,
            total=int(view.num_slices),
            desc=f'Rendering canvas {view.name} {job.aug_id}',
            disable=not bool(show_progress),
        ):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)


def render_tilted_fullframe_video_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    fps: float,
    *,
    workers: int = 1,
    show_progress: bool = True,
) -> None:
    proc = ffmpeg_ffv1_gray_writer(
        job.video_path,
        width=int(job.aff.out_size),
        height=int(job.aff.out_size),
        fps=float(fps),
    )
    try:
        assert proc.stdin is not None

        def _render(idx: int) -> np.ndarray:
            return render_tilted_frame_on_grid(
                volume_rgb=volume_rgb,
                view=view,
                frame_idx=int(idx),
                M_grid_to_src=job.aff.M_out_to_src,
                grid_h=int(job.aff.out_size),
                grid_w=int(job.aff.out_size),
            )

        pending = min(int(view.num_slices), max(2, int(workers) * 2))
        iterable = parallel_map_in_order(_render, range(int(view.num_slices)), max_workers=max(1, int(workers)), max_pending=pending)
        for frame in tqdm(
            iterable,
            total=int(view.num_slices),
            desc=f'Rendering full-frame {view.name} {job.aug_id}',
            disable=not bool(show_progress),
        ):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)


def derive_fullframe_video_from_canvas(
    canvas_video_path: Path,
    out_path: Path,
    fps: float,
    canvas_w: int,
    canvas_h: int,
    out_size: int,
    num_frames: int,
    *,
    show_progress: bool = True,
) -> None:
    proc = ffmpeg_ffv1_gray_writer(
        out_path,
        width=int(out_size),
        height=int(out_size),
        fps=float(fps),
    )
    try:
        assert proc.stdin is not None
        for frame in tqdm(
            iter_ffmpeg_gray8_frames(canvas_video_path, int(canvas_w), int(canvas_h), int(num_frames)),
            total=int(num_frames),
            desc=f'Deriving full-frame {out_path.name}',
            disable=not bool(show_progress),
        ):
            out_frame = _resize_frame_centered(frame, int(out_size), int(out_size), interpolation=cv2.INTER_LINEAR)
            proc.stdin.write(np.ascontiguousarray(out_frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)


def derive_dense_tile_video_from_canvas(
    canvas_video_path: Path,
    job: DenseTileJob,
    fps: float,
    canvas_w: int,
    canvas_h: int,
    num_frames: int,
    *,
    show_progress: bool = True,
) -> None:
    proc = ffmpeg_ffv1_gray_writer(
        job.video_path,
        width=int(job.out_size),
        height=int(job.out_size),
        fps=float(fps),
    )
    try:
        assert proc.stdin is not None
        for frame in tqdm(
            iter_ffmpeg_gray8_frames(canvas_video_path, int(canvas_w), int(canvas_h), int(num_frames)),
            total=int(num_frames),
            desc=f'Deriving dense tile {job.view} {job.tile_id}',
            disable=not bool(show_progress),
        ):
            tile = _extract_padded_tile_frame(frame, int(job.tile_x), int(job.tile_y), int(job.tile_size))
            out_frame = _resize_frame_centered(tile, int(job.out_size), int(job.out_size), interpolation=cv2.INTER_LINEAR)
            proc.stdin.write(np.ascontiguousarray(out_frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)



def render_fullframe_frame_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    frame_idx: int,
    view_frames: Optional[np.ndarray] = None,
) -> np.ndarray:
    if view.family == 'tilted_transverse':
        return render_tilted_frame_on_grid(
            volume_rgb=volume_rgb,
            view=view,
            frame_idx=int(frame_idx),
            M_grid_to_src=job.aff.M_out_to_src,
            grid_h=int(job.aff.out_size),
            grid_w=int(job.aff.out_size),
        )

    native_frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(frame_idx), view_frames=view_frames))
    return cv2.warpAffine(
        native_frame,
        job.aff.M_src_to_out,
        dsize=(int(job.aff.out_size), int(job.aff.out_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def render_fullframe_video_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    fps: float,
    *,
    view_frames: Optional[np.ndarray] = None,
    workers: int = 1,
    show_progress: bool = True,
) -> None:
    if view.family == 'tilted_transverse':
        render_tilted_fullframe_video_for_job(
            volume_rgb=volume_rgb,
            view=view,
            job=job,
            fps=float(fps),
            workers=max(1, int(workers)),
            show_progress=bool(show_progress),
        )
        return

    proc = ffmpeg_ffv1_gray_writer(
        job.video_path,
        width=int(job.aff.out_size),
        height=int(job.aff.out_size),
        fps=float(fps),
    )
    try:
        assert proc.stdin is not None

        def _render(idx: int) -> np.ndarray:
            return render_fullframe_frame_for_job(
                volume_rgb=volume_rgb,
                view=view,
                job=job,
                frame_idx=int(idx),
                view_frames=view_frames,
            )

        pending = min(int(view.num_slices), max(2, int(workers) * 2))
        iterable = parallel_map_in_order(_render, range(int(view.num_slices)), max_workers=max(1, int(workers)), max_pending=pending)
        for frame in tqdm(
            iterable,
            total=int(view.num_slices),
            desc=f'Rendering full-frame {view.name} {job.aug_id}',
            disable=not bool(show_progress),
        ):
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)


def ensure_canvas_and_fullframe_video(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    fps: float,
    workers: int,
    view_frames: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> None:
    if not job.meta_path.exists():
        write_aug_job_meta(job, view)
    if not job.video_path.exists():
        job.video_path.parent.mkdir(parents=True, exist_ok=True)
        render_fullframe_video_for_job(
            volume_rgb=volume_rgb,
            view=view,
            job=job,
            fps=float(fps),
            view_frames=view_frames,
            workers=max(1, int(workers)),
            show_progress=bool(show_progress),
        )
    if not job.canvas_video_path.exists():
        job.canvas_video_path.parent.mkdir(parents=True, exist_ok=True)
        render_canvas_video_for_job(
            volume_rgb=volume_rgb,
            view=view,
            job=job,
            fps=float(fps),
            view_frames=view_frames,
            workers=max(1, int(workers)),
            show_progress=bool(show_progress),
        )


def ensure_canvas_video_only(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    fps: float,
    workers: int,
    view_frames: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> None:
    if not job.meta_path.exists():
        write_aug_job_meta(job, view)
    if job.canvas_video_path.exists():
        return
    job.canvas_video_path.parent.mkdir(parents=True, exist_ok=True)
    render_canvas_video_for_job(
        volume_rgb=volume_rgb,
        view=view,
        job=job,
        fps=float(fps),
        view_frames=view_frames,
        workers=max(1, int(workers)),
        show_progress=bool(show_progress),
    )


def ensure_fullframe_video_only(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    fps: float,
    workers: int,
    view_frames: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> None:
    if not job.meta_path.exists():
        write_aug_job_meta(job, view)
    if job.video_path.exists():
        return
    job.video_path.parent.mkdir(parents=True, exist_ok=True)
    render_fullframe_video_for_job(
        volume_rgb=volume_rgb,
        view=view,
        job=job,
        fps=float(fps),
        view_frames=view_frames,
        workers=max(1, int(workers)),
        show_progress=bool(show_progress),
    )


def _run_ffmpeg_checked(cmd: Sequence[str], description: str) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode('utf-8', errors='ignore') if isinstance(proc.stderr, (bytes, bytearray)) else str(proc.stderr)
        stdout = proc.stdout.decode('utf-8', errors='ignore') if isinstance(proc.stdout, (bytes, bytearray)) else str(proc.stdout)
        detail = stderr.strip() or stdout.strip()
        raise RuntimeError(f'{description} failed: {detail}')


def dense_tile_fanout_limit() -> int:
    return max(1, _env_int('YOLO_TTA_TILE_FANOUT', 32))


def _chunk_dense_tile_jobs(jobs: Sequence[DenseTileJob], chunk_size: int) -> Iterator[List[DenseTileJob]]:
    step = max(1, int(chunk_size))
    for start in range(0, len(jobs), step):
        yield list(jobs[start:start + step])


def derive_dense_tile_videos_from_canvas_batch(
    canvas_video_path: Path,
    jobs: Sequence[DenseTileJob],
    fps: float,
    canvas_w: int,
    canvas_h: int,
    num_frames: int,
    *,
    show_progress: bool = True,
) -> None:
    del fps
    del num_frames

    missing_jobs = [job for job in jobs if not job.video_path.exists()]
    if not missing_jobs:
        return

    filter_threads = max(1, _env_int('YOLO_TTA_TILE_FILTER_THREADS', min(8, _cpu_count())))
    encoder_threads = max(1, _env_int('YOLO_TTA_TILE_FFV1_THREADS', 1))
    batch_limit = dense_tile_fanout_limit()

    for batch_idx, batch_jobs in enumerate(_chunk_dense_tile_jobs(missing_jobs, batch_limit), start=1):
        pad_w = max(int(canvas_w), max(int(job.tile_x) + int(job.tile_size) for job in batch_jobs))
        pad_h = max(int(canvas_h), max(int(job.tile_y) + int(job.tile_size) for job in batch_jobs))

        base_labels = [f'v{i}' for i in range(len(batch_jobs))]
        filter_parts: List[str] = []
        if len(batch_jobs) == 1:
            ops: List[str] = []
            if pad_w != int(canvas_w) or pad_h != int(canvas_h):
                ops.append(f'pad={pad_w}:{pad_h}:0:0:black')
            else:
                ops.append('null')
            filter_parts.append(f'[0:v]{",".join(ops)}[{base_labels[0]}]')
        else:
            ops = []
            if pad_w != int(canvas_w) or pad_h != int(canvas_h):
                ops.append(f'pad={pad_w}:{pad_h}:0:0:black')
            ops.append(f'split={len(batch_jobs)}' + ''.join(f'[{label}]' for label in base_labels))
            filter_parts.append(f'[0:v]{",".join(ops)}')

        for out_idx, job in enumerate(batch_jobs):
            filter_parts.append(
                f'[{base_labels[out_idx]}]'
                f'crop={int(job.tile_size)}:{int(job.tile_size)}:{int(job.tile_x)}:{int(job.tile_y)},'
                f'scale={int(job.out_size)}:{int(job.out_size)}:flags=bilinear'
                f'[t{out_idx}]'
            )

        cmd: List[str] = [
            'ffmpeg',
            '-y',
            '-v', 'error',
            '-filter_complex_threads', str(filter_threads),
            '-i', str(canvas_video_path),
            '-vsync', '0',
            '-filter_complex', ';'.join(filter_parts),
        ]

        for out_idx, job in enumerate(batch_jobs):
            job.video_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend([
                '-map', f'[t{out_idx}]',
                '-an',
                '-c:v', 'ffv1',
                '-level', '3',
                '-slices', '30',
                '-threads', str(encoder_threads),
                '-pix_fmt', 'gray',
                str(job.video_path),
            ])

        if show_progress:
            print(
                f'Deriving dense tiles from {canvas_video_path.name} '
                f'(batch {batch_idx}, outputs={len(batch_jobs)})'
            )
        _run_ffmpeg_checked(cmd, f'dense tile fan-out from {canvas_video_path.name} (batch {batch_idx})')


def ensure_dense_tile_video_batch_after_canvas(
    canvas_future: Future,
    aug_job: AugJob,
    view: ViewInfo,
    jobs: Sequence[DenseTileJob],
    fps: float,
    show_progress: bool = True,
) -> None:
    canvas_future.result()
    ensure_dense_tile_video_batch_from_canvas(
        aug_job=aug_job,
        view=view,
        jobs=jobs,
        fps=float(fps),
        show_progress=bool(show_progress),
    )


def ensure_dense_tile_video_batch_from_canvas(
    aug_job: AugJob,
    view: ViewInfo,
    jobs: Sequence[DenseTileJob],
    fps: float,
    show_progress: bool = True,
) -> None:
    jobs = list(jobs)
    if not jobs:
        return

    for job in jobs:
        if not job.meta_path.exists():
            write_dense_tile_job_meta(job)

    if all(job.video_path.exists() for job in jobs):
        return

    derive_dense_tile_videos_from_canvas_batch(
        canvas_video_path=aug_job.canvas_video_path,
        jobs=jobs,
        fps=float(fps),
        canvas_w=int(aug_job.aff.canvas_w),
        canvas_h=int(aug_job.aff.canvas_h),
        num_frames=int(view.num_slices),
        show_progress=bool(show_progress),
    )









def should_cache_view_frames(view: ViewInfo, dense_tiling_active: bool) -> bool:
    """Return True when precomputing native single-channel frames is worthwhile for this view.

    Radial view extraction is the dominant CPU cost during dense tiled rendering because every
    tile video currently traverses the same expensive Lanczos-5 radial slices independently.
    Caching the native radial frames once lets both full-frame augmentation and all tile videos
    reuse that work.
    """
    return bool(dense_tiling_active) and view.family == 'radial'




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
    """Materialize native single-channel frames for a view into a reusable cache volume.

    This is used primarily for the radial view when dense tiling is enabled so later tile video
    generation no longer recomputes the same radial slices for every tile location.
    """
    cache_mm = allocate_workspace_array(
        shape=(int(view.num_slices), int(view.src_h), int(view.src_w)),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    worker_count = choose_slice_parallel_workers(int(workers), int(view.num_slices))
    print(
        f"Preparing reusable native single-channel frame cache for view '{view.name}' over {int(view.num_slices)} slice(s) "
        f"with {int(worker_count)} worker thread(s)"
    )

    def _build(idx: int) -> None:
        cache_mm[int(idx), :, :] = np.ascontiguousarray(
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
    """Yield single-channel frames for a view, in slice order (0..num_slices-1)."""
    if view_frames is not None:
        for idx in range(int(view.num_slices)):
            yield np.asarray(view_frames[int(idx)])
        return

    T, H, W = volume_rgb.shape

    if view.name == 'transverse':
        for t in range(T):
            yield np.asarray(volume_rgb[t])  # (H,W)
    elif view.name == 'sagittal':
        for y in range(H):
            yield np.ascontiguousarray(volume_rgb[:, y, :])  # (T,W)
    elif view.name == 'coronal':
        for x in range(W):
            yield np.ascontiguousarray(volume_rgb[:, :, x])  # (T,H)
    elif view.name == 'radial':
        for angle_deg in view.azimuths_deg:
            sampler = get_radial_sampler(view, float(angle_deg))
            yield np.ascontiguousarray(extract_radial_slice_frame(volume_rgb, sampler))  # (T,D)
    elif view.family == 'tilted_transverse':
        for t in range(int(view.num_slices)):
            yield render_tilted_native_frame(volume_rgb, view, int(t))
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

    T, H, W = volume_rgb.shape

    if view.name == 'transverse':
        return np.asarray(volume_rgb[int(index)])
    if view.name == 'sagittal':
        return np.ascontiguousarray(volume_rgb[:, int(index), :])
    if view.name == 'coronal':
        return np.ascontiguousarray(volume_rgb[:, :, int(index)])
    if view.name == 'radial':
        angle_deg = float(view.azimuths_deg[int(index)])
        if radial_fast_path_enabled():
            map_x, map_y = build_radial_block_maps(view, [angle_deg])
            map_x = np.ascontiguousarray(map_x.astype(np.float32, copy=False))
            map_y = np.ascontiguousarray(map_y.astype(np.float32, copy=False))
            out = np.empty((int(view.src_h), int(view.src_w)), dtype=np.uint8)
            for t in range(T):
                sampled = cv2.remap(
                    np.asarray(volume_rgb[t]),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                out[t, :] = np.asarray(sampled[0])
            return out

        sampler = get_radial_sampler(view, angle_deg)
        return np.ascontiguousarray(extract_radial_slice_frame(volume_rgb, sampler))
    if view.family == 'tilted_transverse':
        return render_tilted_native_frame(volume_rgb, view, int(index))

    raise ValueError(f'Unknown view: {view.name}')




# --------------------------
# Native mask accumulation# --------------------------
# Native mask accumulation
# --------------------------

# --------------------------
# YOLO inference
# --------------------------

def load_ultralytics_model(path: str, task: str = 'segment'):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is required. Install with: pip install ultralytics\n"
            f"Import error: {e}"
        ) from e
    return YOLO(path, task=task)


def canonical_single_device(device: str) -> str:
    raw = str(device or '').strip()
    if not raw:
        return 'cpu'

    token = raw.split(',')[0].strip()
    low = token.lower()
    if low in ('cpu', 'mps'):
        return low
    if low.startswith('cuda'):
        return low
    if token.isdigit():
        return f'cuda:{token}'
    return token


def ensure_yolo_ready_for_predict(model: object, cfg: 'PredictConfig') -> None:
    """Keep the active YOLO backend resident on the requested device.

    The scheduling contract wants the GPU to stay hot until work is exhausted. Offloading the model
    between videos can leave Ultralytics with CUDA inputs but CPU weights on the next predict() call.
    This helper lazily restores the requested device/dtype when needed and is a cheap no-op while the
    model already matches the active predict configuration.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return

    target = canonical_single_device(str(cfg.device))
    wants_half = bool(cfg.half) and str(target).startswith('cuda')
    state = (str(target), bool(wants_half))
    if getattr(model, '_tta_predict_state', None) == state:
        return

    candidates: List[object] = [model]
    direct_model = getattr(model, 'model', None)
    if direct_model is not None:
        candidates.append(direct_model)
    predictor = getattr(model, 'predictor', None)
    predictor_model = getattr(predictor, 'model', None) if predictor is not None else None
    if predictor_model is not None and predictor_model is not direct_model:
        candidates.append(predictor_model)

    seen_ids: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        cid = id(candidate)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        to_fn = getattr(candidate, 'to', None)
        if callable(to_fn):
            try:
                to_fn(target)
            except Exception:
                pass

        precision_fn = getattr(candidate, 'half', None) if wants_half else getattr(candidate, 'float', None)
        if callable(precision_fn):
            try:
                precision_fn()
            except Exception:
                pass

    if predictor is not None:
        try:
            setattr(predictor, 'device', torch.device(target))
        except Exception:
            pass
        args_obj = getattr(predictor, 'args', None)
        if args_obj is not None:
            try:
                setattr(args_obj, 'device', target)
            except Exception:
                pass

    try:
        setattr(model, '_tta_predict_state', state)
    except Exception:
        pass


def offload_between_jobs_enabled() -> bool:
    return _env_flag('YOLO_TTA_OFFLOAD_BETWEEN_JOBS', False)


def trim_cuda_memory() -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return

    try:
        if not bool(torch.cuda.is_available()):
            return
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, 'ipc_collect', None)
        if callable(ipc_collect):
            ipc_collect()
    except Exception:
        pass


def offload_yolo_from_gpu(model: object) -> None:
    modules: List[object] = []
    direct_model = getattr(model, 'model', None)
    if direct_model is not None:
        modules.append(direct_model)

    predictor = getattr(model, 'predictor', None)
    predictor_model = getattr(predictor, 'model', None) if predictor is not None else None
    if predictor_model is not None and predictor_model is not direct_model:
        modules.append(predictor_model)

    for module in modules:
        try:
            to_fn = getattr(module, 'to', None)
            if callable(to_fn):
                to_fn('cpu')
        except Exception:
            pass

    try:
        setattr(model, '_tta_predict_state', None)
    except Exception:
        pass

    trim_cuda_memory()


def unload_yolo_model(model: object) -> None:
    offload_yolo_from_gpu(model)
    predictor = getattr(model, 'predictor', None)
    if predictor is not None:
        try:
            setattr(model, 'predictor', None)
        except Exception:
            pass
    trim_cuda_memory()


@dataclass
class PredictConfig:
    imgsz: int
    conf: float
    device: str
    half: bool
    int8: bool


CONF_U8_MAX = 255


def quantize_conf_to_u8(conf: float) -> np.uint8:
    conf_clamped = min(1.0, max(0.0, float(conf)))
    return np.uint8(int(round(conf_clamped * float(CONF_U8_MAX))))


def min_conf_to_u8_threshold(min_conf: float) -> int:
    conf_clamped = min(1.0, max(0.0, float(min_conf)))
    return int(math.ceil(conf_clamped * float(CONF_U8_MAX) - 1e-9))


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



def _scatter_tilted_native_xy_to_volume(
    frame_idx: int,
    native_union: np.ndarray,
    native_conf: Optional[np.ndarray],
    tilted_view: ViewInfo,
    view_union_mm: np.ndarray,
    view_confmap_mm: np.ndarray,
) -> None:
    """Back-project one Tilted Transverse prediction frame into the native (t, Y, X) volume.

    Tilted transverse rendering samples the native XY grid while shifting only the slice
    coordinate: ``t = frame_idx + tan(alpha) * axis_offset``. The inverse therefore must
    not place every predicted pixel back into ``frame_idx``. Doing so creates duplicated
    structures displaced along the transverse slice axis.
    """
    if tilted_view.family != 'tilted_transverse':
        raise ValueError('Tilted inverse projection requested for a non-tilted view')

    mask_bool = np.asarray(native_union, dtype=bool)
    if not np.any(mask_bool):
        return

    ys, xs = np.nonzero(mask_bool)
    if ys.size <= 0:
        return

    tan_alpha = float(math.tan(math.radians(float(tilted_view.tilt_angle_deg))))
    frame_center = float(tilted_frame_center(tilted_view, int(frame_idx)))
    if str(tilted_view.tilt_direction) == 'vertical':
        cy = float((int(tilted_view.full_h) - 1) / 2.0)
        t_float = frame_center + tan_alpha * (ys.astype(np.float32, copy=False) - cy)
    elif str(tilted_view.tilt_direction) == 'horizontal':
        cx = float((int(tilted_view.full_w) - 1) / 2.0)
        t_float = frame_center + tan_alpha * (xs.astype(np.float32, copy=False) - cx)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported tilt direction: {tilted_view.tilt_direction}')

    t_idx = np.rint(t_float).astype(np.int32, copy=False)
    valid = (t_idx >= 0) & (t_idx < int(view_union_mm.shape[0]))
    if not np.any(valid):
        return

    tt = t_idx[valid]
    yy = ys[valid]
    xx = xs[valid]
    view_union_mm[tt, yy, xx] = np.uint8(1)

    if native_conf is None:
        return

    native_conf_u8 = np.asarray(native_conf, dtype=np.uint8)
    conf_vals = native_conf_u8[yy, xx]
    has_conf = conf_vals > 0
    if np.any(has_conf):
        np.maximum.at(
            view_confmap_mm,
            (tt[has_conf], yy[has_conf], xx[has_conf]),
            conf_vals[has_conf],
        )


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
    tilted_view: Optional[ViewInfo] = None,
) -> Tuple[int, int]:
    """Collapse one streamed result directly into unpacked native-view union + confidence volumes."""
    if masks_np is None or confs_np is None or masks_np.ndim != 3 or int(masks_np.shape[0]) <= 0:
        return 0, 0

    frame_union = np.zeros((out_size, out_size), dtype=np.uint8)
    frame_confmap = np.zeros((out_size, out_size), dtype=np.uint8)
    num_inst = int(masks_np.shape[0])

    for inst_idx in range(num_inst):
        inst = np.asarray(masks_np[inst_idx], dtype=np.uint8)
        if inst.shape[0] != out_size or inst.shape[1] != out_size:
            inst = cv2.resize(inst, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        inst = (inst > 0).astype(np.uint8, copy=False)
        if not np.any(inst):
            continue

        conf_val = float(confs_np[inst_idx]) if inst_idx < int(confs_np.shape[0]) else 0.0
        conf_u8 = quantize_conf_to_u8(conf_val)
        frame_union |= inst
        inst_bool = inst > 0
        frame_confmap[inst_bool] = np.maximum(frame_confmap[inst_bool], conf_u8)

    if not np.any(frame_union):
        return int(num_inst), 1

    native_union = cv2.warpAffine(
        frame_union,
        M_out_to_native,
        dsize=(native_w, native_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    native_conf: Optional[np.ndarray] = None
    if np.any(frame_confmap):
        native_conf = cv2.warpAffine(
            frame_confmap,
            M_out_to_native,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8, copy=False)

    if tilted_view is not None:
        _scatter_tilted_native_xy_to_volume(
            frame_idx=int(idx),
            native_union=native_union,
            native_conf=native_conf,
            tilted_view=tilted_view,
            view_union_mm=view_union_mm,
            view_confmap_mm=view_confmap_mm,
        )
    else:
        if np.any(native_union):
            view_union_mm[idx, :, :] |= native_union.astype(np.uint8, copy=False)

        if native_conf is not None and np.any(native_conf):
            conf_slice = view_confmap_mm[idx]
            np.maximum(conf_slice, native_conf, out=conf_slice)

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
    view_confmap_mm: np.ndarray,        # uint8 confidence map, shape (num_slices, native_h, native_w)
    M_out_to_native: np.ndarray,       # 2x3, maps augmented(out)->native for cv2.warpAffine (src->dst)
    native_h: int,
    native_w: int,
    postprocess_workers: int = 1,
    tilted_view: Optional[ViewInfo] = None,
) -> Dict[str, int]:
    """
    Run YOLO predict(stream=True) on a pre-generated augmented video and accumulate the inverse-
    transformed native masks in memory-backed workspaces.

    The sequential portion remains the YOLO inference stream itself. CPU-side result handling is
    overlapped with that stream when ``postprocess_workers > 1`` so native reorientation and
    confidence-map updates do not unnecessarily serialize GPU inference.
    """
    ensure_yolo_ready_for_predict(model, cfg)

    prediction_count = 0
    frames_with_predictions = 0

    results = model.predict(
        source=str(video_path),
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=1.0,
        save=False,
        stream=True,
        retina_masks=True,
        batch=1,
        device=cfg.device,
        half=cfg.half,
        int8=cfg.int8,
        verbose=False,
    )

    # Tilted transverse inverse projection scatters one prediction frame across native
    # t-slices. Process those frames serially to avoid non-atomic writes from multiple
    # postprocess workers into the same native volume voxels/confidence map.
    if tilted_view is not None:
        worker_count = 1
    else:
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
                tilted_view=tilted_view,
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
                    tilted_view,
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






















def _union_supported_native_components_into_parent(
    native_mask_bool: np.ndarray,
    support_slice_bool: np.ndarray,
    parent_slice: np.ndarray,
    conf_slice: np.ndarray,
    conf_val: float,
) -> Dict[str, int]:
    """Union only the connected components that are supported by the frozen full-frame slice.

    A single YOLO mask can contain multiple disconnected islands after inverse mapping. Treat those
    disconnected regions independently for dense tiled gating so an unsupported island cannot hitch a
    ride on another supported component from the same tiled instance.
    """
    stats = {
        'accepted_components': 0,
        'rejected_components': 0,
        'gated_added_voxels': 0,
    }
    if not np.any(native_mask_bool):
        return stats

    num_labels, labels2d = cv2.connectedComponents(
        np.asarray(native_mask_bool, dtype=np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    if int(num_labels) <= 1:
        return stats

    conf_val_u8 = quantize_conf_to_u8(float(conf_val)) if float(conf_val) > 0.0 else np.uint8(0)
    for comp_lbl in range(1, int(num_labels)):
        comp = labels2d == int(comp_lbl)
        if not np.any(comp):
            continue
        if not np.any(comp & support_slice_bool):
            stats['rejected_components'] += 1
            continue

        added = comp & (parent_slice == 0)
        if np.any(added):
            stats['gated_added_voxels'] += int(np.count_nonzero(added))
        parent_slice[comp] = np.uint8(1)
        if float(conf_val) > 0.0:
            conf_slice[comp] = np.maximum(conf_slice[comp], conf_val_u8)
        stats['accepted_components'] += 1

    return stats



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
) -> Dict[str, int]:
    """Gate one tiled frame against the frozen parent full-frame slice into an isolated tiled volume."""
    stats = {
        'prediction_count': 0,
        'frames_with_predictions': 0,
        'accepted_masks': 0,
        'rejected_masks': 0,
        'accepted_components': 0,
        'rejected_components': 0,
        'rejected_min_conf': 0,
        'gated_added_voxels': 0,
    }

    if masks_np is None or confs_np is None or masks_np.ndim != 3 or int(masks_np.shape[0]) <= 0:
        return stats

    support_slice = np.asarray(baseline_support_mm[int(idx)], dtype=bool)
    tiled_slice = np.asarray(baseline_mask_mm[int(idx)], dtype=np.uint8)
    conf_slice = baseline_confmap_mm[int(idx)]

    num_inst = int(masks_np.shape[0])
    stats['prediction_count'] = int(num_inst)
    stats['frames_with_predictions'] = 1

    for inst_idx in range(num_inst):
        conf_val = float(confs_np[inst_idx]) if inst_idx < int(confs_np.shape[0]) else 0.0

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

        component_stats = _union_supported_native_components_into_parent(
            native_mask_bool=native_mask_bool,
            support_slice_bool=support_slice,
            parent_slice=tiled_slice,
            conf_slice=conf_slice,
            conf_val=conf_val,
        )
        stats['accepted_components'] += int(component_stats['accepted_components'])
        stats['rejected_components'] += int(component_stats['rejected_components'])
        stats['gated_added_voxels'] += int(component_stats['gated_added_voxels'])

        if int(component_stats['accepted_components']) > 0:
            stats['accepted_masks'] += 1
        else:
            stats['rejected_masks'] += 1

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
    postprocess_workers: int = 1,
) -> Dict[str, int]:
    """Run tiled inference and gate accepted masks into an isolated tiled-view volume."""
    ensure_yolo_ready_for_predict(model, cfg)

    results = model.predict(
        source=str(video_path),
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=1.0,
        save=False,
        stream=True,
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
        'accepted_components': 0,
        'rejected_components': 0,
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
# --------------------------
# Per-view postprocessing
# --------------------------




def _fill_holes_2d(mask_bool: np.ndarray) -> np.ndarray:
    return np.asarray(ndi.binary_fill_holes(np.asarray(mask_bool, dtype=bool)), dtype=bool)



def _filter_connected_components_by_min_radius(
    mask_bool: np.ndarray,
    structure2: np.ndarray,
    min_radius: float,
) -> np.ndarray:
    labels2d, num = ndi.label(mask_bool, structure=structure2)
    if int(num) <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)

    label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
    dist = np.asarray(ndi.distance_transform_edt(mask_bool), dtype=np.float32)
    radii = np.asarray(ndi.maximum(dist, labels=labels2d, index=label_ids), dtype=np.float32)
    keep_ids = label_ids[radii >= float(min_radius)]
    if keep_ids.size <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    return np.isin(labels2d, keep_ids)



def _build_supported_component_mask(
    tile_slice: np.ndarray,
    support_slice: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    num_labels, labels2d = cv2.connectedComponents(
        np.asarray(tile_slice, dtype=np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    if int(num_labels) <= 1:
        return np.zeros(tile_slice.shape, dtype=bool), 0, 0

    label_ids = np.arange(1, int(num_labels), dtype=np.int32)
    supported = np.asarray(
        ndi.maximum(
            np.asarray(support_slice, dtype=np.uint8),
            labels=labels2d,
            index=label_ids,
        ),
        dtype=np.uint8,
    )
    keep_ids = label_ids[supported > 0]
    if keep_ids.size <= 0:
        return np.zeros(tile_slice.shape, dtype=bool), 0, int(num_labels) - 1

    keep = np.isin(labels2d, keep_ids)
    return keep, int(keep_ids.size), int((int(num_labels) - 1) - int(keep_ids.size))



def fused_slice_cleanup_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray] = None,
    *,
    min_conf: float = 0.0,
    min_radius: float = 0.0,
    workers: int = 1,
    desc: str = 'Fused slice cleanup',
) -> None:
    """Slice-parallel cleanup with chunked worker fan-out.

    The old path delegated one future per slice and, when ``--min_radius`` was active, performed a
    Python loop plus one EDT per connected component. On large sparse slices that makes the stage
    look effectively single-threaded even though it nominally uses a thread pool. The updated path
    keeps the same semantics but:
      - submits chunked slice ranges so worker threads stay busy with lower dispatch overhead
      - computes connected-component radii with one EDT + one reduce per slice instead of one EDT
        per component
    """
    num_slices = int(mask_mm.shape[0])
    structure2 = np.ones((3, 3), dtype=bool)
    min_conf_u8 = int(min_conf_to_u8_threshold(float(min_conf)))
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)
    chunk_size = choose_parallel_chunk_size(num_slices, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _process(i: int) -> None:
        mask_slice = np.asarray(mask_mm[int(i)], dtype=bool)
        conf_slice = None if confmap_mm is None else np.asarray(confmap_mm[int(i)], dtype=np.uint8)

        if np.any(mask_slice) and conf_slice is not None and float(min_conf) > 0.0:
            labels2d, num = ndi.label(mask_slice, structure=structure2)
            if int(num) > 0:
                label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
                maxima = np.asarray(ndi.maximum(conf_slice, labels=labels2d, index=label_ids), dtype=np.uint8)
                keep_ids = label_ids[maxima >= int(min_conf_u8)]
                if keep_ids.size > 0:
                    mask_slice = np.isin(labels2d, keep_ids)
                else:
                    mask_slice = np.zeros(mask_slice.shape, dtype=bool)
            else:
                mask_slice = np.zeros(mask_slice.shape, dtype=bool)

        if np.any(mask_slice):
            mask_slice = _fill_holes_2d(mask_slice)

        if np.any(mask_slice) and float(min_radius) > 0.0:
            mask_slice = _filter_connected_components_by_min_radius(
                mask_slice,
                structure2,
                float(min_radius),
            )

        mask_mm[int(i), :, :] = mask_slice.astype(np.uint8, copy=False)
        if conf_slice is not None:
            if np.any(mask_slice):
                conf_slice[~mask_slice] = np.uint8(0)
            else:
                conf_slice.fill(np.uint8(0))
            confmap_mm[int(i), :, :] = conf_slice.astype(np.uint8, copy=False)

    parallel_for_indices_chunked(
        num_slices,
        _process,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
    )
    flush_array(mask_mm)
    if confmap_mm is not None:
        flush_array(confmap_mm)


def cleanup_view_volume_after_prediction_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray],
    view: ViewInfo,
    min_conf: float,
    min_radius: float,
    *,
    workers: int = 1,
) -> None:
    native_min_radius = 0.0
    if view.name == 'transverse' or view.family == 'tilted_transverse':
        native_min_radius = float(min_radius)

    fused_slice_cleanup_inplace(
        mask_mm,
        confmap_mm,
        min_conf=float(min_conf),
        min_radius=float(native_min_radius),
        workers=int(workers),
        desc=f'Fused cleanup ({view.name})',
    )

    if view.name in ('sagittal', 'coronal') and float(min_radius) > 0.0:
        apply_view_min_radius_filter_inplace(
            mask_mm,
            view,
            float(min_radius),
            workers=int(workers),
        )


# --------------------------
# 3D assembly + postprocessing
# --------------------------
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


def _adjacent_xy_offsets_for_3d_connectivity(connectivity: int) -> Tuple[Tuple[int, int], ...]:
    """Return XY offsets that connect components across adjacent z-slices.

    Connectivity is interpreted in the standard cubic-neighborhood sense:
      - 6-connected: only face adjacency across z, so the same (y, x) position
      - 18-connected: face/edge adjacency, so same position plus cardinal XY offsets
      - 26-connected: face/edge/corner adjacency, so the full 3x3 XY neighborhood
    """
    connectivity_i = int(connectivity)
    if connectivity_i == 6:
        return ((0, 0),)
    if connectivity_i == 18:
        return tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if abs(dy) + abs(dx) <= 1)
    if connectivity_i == 26:
        return tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    raise ValueError('3D connectivity must be one of 6, 18, or 26')


def _slice_connectivity_for_3d_connectivity(connectivity: int) -> int:
    connectivity_i = int(connectivity)
    if connectivity_i == 6:
        return 4
    if connectivity_i in (18, 26):
        return 8
    raise ValueError('3D connectivity must be one of 6, 18, or 26')


def _iter_adjacent_gid_pairs(
    prev_gid: np.ndarray,
    curr_gid: np.ndarray,
    xy_offsets: Optional[Sequence[Tuple[int, int]]] = None,
) -> Iterator[Tuple[int, int]]:
    """Yield unique touching component-id pairs across adjacent z-slices.

    ``xy_offsets`` controls the 3D connectivity across adjacent slices. When omitted, the
    existing 26-connected behavior is used for foreground interpolation labeling.
    """
    h, w = prev_gid.shape
    seen: set[int] = set()
    offsets = tuple(xy_offsets) if xy_offsets is not None else _adjacent_xy_offsets_for_3d_connectivity(26)

    for dy, dx in offsets:
        dy_i = int(dy)
        dx_i = int(dx)
        py0 = max(0, -dy_i)
        py1 = min(h, h - dy_i)
        cy0 = max(0, dy_i)
        cy1 = min(h, h + dy_i)
        px0 = max(0, -dx_i)
        px1 = min(w, w - dx_i)
        cx0 = max(0, dx_i)
        cx1 = min(w, w + dx_i)

        if py0 >= py1 or px0 >= px1 or cy0 >= cy1 or cx0 >= cx1:
            continue

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
    mask_mm: np.ndarray,
    work_prefix: Path,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    connectivity: int = 6,
) -> None:
    """Fill enclosed 3D voids by labeling background connected components once.

    v9.1.0 allows 3D void fill only as an optional final global-union step. This
    function performs that operation by labeling the background volume, marking any
    background component connected to the volume boundary, and converting all other
    background components to foreground. The default 6-connected background matches
    the usual enclosed-void interpretation used by binary hole filling; set
    ``YOLO_TTA_VOIDFILL_CONNECTIVITY`` or pass ``connectivity`` explicitly to use 18
    or 26 connectivity.

      - prefers anonymous RAM/swap-backed arrays for the 3D background-ID workspace
      - falls back to a disk-backed memmap only when the estimated working set would be too large
      - avoids tmpfs-backed bulk scratch files that could previously SIGBUS when /dev/shm filled
    """
    env_conn = os.environ.get('YOLO_TTA_VOIDFILL_CONNECTIVITY', '').strip()
    if env_conn:
        try:
            connectivity = int(env_conn)
        except Exception:
            pass
    connectivity = int(connectivity)
    if connectivity not in (6, 18, 26):
        raise ValueError('3D void fill connectivity must be one of 6, 18, or 26')

    z_dim, h, w = mask_mm.shape
    if z_dim <= 0:
        return

    slice_connectivity = _slice_connectivity_for_3d_connectivity(connectivity)
    adjacent_offsets = _adjacent_xy_offsets_for_3d_connectivity(connectivity)

    estimated_bytes = estimate_voidfill_workspace_bytes((z_dim, h, w))
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(estimated_bytes, reserve_bytes=reserve_bytes)
    budget = workspace_budget_summary(estimated_bytes, reserve_bytes=reserve_bytes)
    if use_in_memory:
        print(f"3D void fill workspace: in-memory ({budget}, background connectivity={connectivity})")
        bg_gid_store: np.ndarray = np.zeros((z_dim, h, w), dtype=np.uint32)
        bg_gid_path: Optional[Path] = None
    else:
        print(f"3D void fill workspace: disk-backed ({budget}, background connectivity={connectivity}) -> {work_prefix.parent}")
        bg_gid_path = work_prefix.with_suffix('.bg_gid.u32.dat')
        bg_gid_path.parent.mkdir(parents=True, exist_ok=True)
        bg_gid_store = np.memmap(bg_gid_path, dtype=np.uint32, mode='w+', shape=(z_dim, h, w))

    uf = _UnionFind()
    prev_gid_slice: Optional[np.ndarray] = None

    for z in tqdm(range(z_dim), desc='3D void fill: slice labeling'):
        bg = (np.asarray(mask_mm[z]) == 0).astype(np.uint8, copy=False)
        num_labels, labels2d = cv2.connectedComponents(bg, connectivity=slice_connectivity, ltype=cv2.CV_32S)
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
            for a, b in _iter_adjacent_gid_pairs(prev_gid_slice, gid_slice, adjacent_offsets):
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
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)
    chunk_size = choose_parallel_chunk_size(num_slices, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _process(t: int) -> None:
        sl = np.asarray(mask_mm[int(t)]) > 0
        if not np.any(sl):
            return

        keep = _filter_connected_components_by_min_radius(
            sl,
            struct2,
            float(min_radius),
        )
        mask_mm[int(t), :, :] = keep.astype(np.uint8, copy=False)

    parallel_for_indices_chunked(
        num_slices,
        _process,
        max_workers=worker_count,
        desc='Transverse min-radius filter',
        chunk_size=chunk_size,
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

    if view.name == 'transverse' or view.family in ('radial', 'tilted_transverse'):
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


def backproject_tilted_volume_to_volume(
    tilted_mask_mm: np.ndarray,
    tilted_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    """Backproject a Tilted Transverse view-native mask stack into native (t, Y, X).

    The tilted view stack remains in generated-video coordinates through cleanup and
    interpolation.  Only at final assembly do we map each foreground pixel from frame
    center ``N`` and native XY coordinates back to the original orthogonal volume using
    ``t = N + tan(alpha) * axis_offset``.
    """
    if tilted_view.family != 'tilted_transverse':
        raise ValueError('backproject_tilted_volume_to_volume expects a tilted transverse view')

    t_dim = int(tilted_view.full_t)
    out_h = int(tilted_view.full_h)
    out_w = int(tilted_view.full_w)
    if t_dim <= 0:
        raise ValueError(f'Tilted view {tilted_view.name} has invalid full_t={tilted_view.full_t}')

    vol_mm = allocate_workspace_array(
        shape=(t_dim, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=f'{desc} workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    tan_alpha = float(math.tan(math.radians(float(tilted_view.tilt_angle_deg))))
    if str(tilted_view.tilt_direction) == 'vertical':
        axis_center = float((out_h - 1) / 2.0)
    elif str(tilted_view.tilt_direction) == 'horizontal':
        axis_center = float((out_w - 1) / 2.0)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported tilt direction: {tilted_view.tilt_direction}')

    for frame_idx in tqdm(range(int(tilted_view.num_slices)), desc=desc):
        tilted_mask = np.asarray(tilted_mask_mm[int(frame_idx)], dtype=bool)
        if not np.any(tilted_mask):
            continue
        yy, xx = np.nonzero(tilted_mask)
        if yy.size <= 0:
            continue

        frame_center = float(tilted_frame_center(tilted_view, int(frame_idx)))
        if str(tilted_view.tilt_direction) == 'vertical':
            t_float = frame_center + tan_alpha * (yy.astype(np.float32, copy=False) - axis_center)
        else:
            t_float = frame_center + tan_alpha * (xx.astype(np.float32, copy=False) - axis_center)

        tt = np.rint(t_float).astype(np.int32, copy=False)
        valid = (tt >= 0) & (tt < t_dim)
        if not np.any(valid):
            continue
        vol_mm[tt[valid], yy[valid], xx[valid]] = np.uint8(1)

    flush_array(vol_mm)
    return vol_mm


def assemble_view_volumes_into_native_union(
    final_union_mm: np.ndarray,
    view_volume_mms: Dict[str, np.ndarray],
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
    *,
    workers: int = 1,
) -> None:
    """OR all per-view prediction volumes from the single active model into native (t, Y, X).

    Transverse, already-backprojected Tilted Transverse, and already-backprojected Radial volumes
    all have native (T,H,W) shape and are merged slice-wise. Sagittal and Coronal retain their native view stacks
    and are mapped back into the transverse coordinate system.
    """
    consumed: set[str] = set()

    if "transverse" in view_volume_mms:
        transverse = np.asarray(view_volume_mms["transverse"])
        assert transverse.shape == (T, H, W)
        consumed.add("transverse")

        transverse_workers = choose_slice_parallel_workers(int(workers), int(T))

        def _merge_transverse(t: int) -> None:
            final_union_mm[int(t), :, :] |= transverse[int(t), :, :]

        parallel_for_indices(
            int(T),
            _merge_transverse,
            max_workers=transverse_workers,
            desc="Assembling final union from transverse view volume",
        )

    if not disable_multiplanar and "sagittal" in view_volume_mms:
        sagittal = np.asarray(view_volume_mms["sagittal"])
        assert sagittal.shape == (H, T, W)
        consumed.add("sagittal")
        sagittal_workers = choose_slice_parallel_workers(int(workers), int(H))

        def _merge_sagittal(y: int) -> None:
            final_union_mm[:, int(y), :] |= sagittal[int(y), :, :]

        parallel_for_indices(
            int(H),
            _merge_sagittal,
            max_workers=sagittal_workers,
            desc="Assembling final union from sagittal view volume",
        )

    if not disable_multiplanar and "coronal" in view_volume_mms:
        coronal = np.asarray(view_volume_mms["coronal"])
        assert coronal.shape == (W, T, H)
        consumed.add("coronal")
        coronal_workers = choose_slice_parallel_workers(int(workers), int(W))

        def _merge_coronal(x: int) -> None:
            final_union_mm[:, :, int(x)] |= coronal[int(x), :, :]

        parallel_for_indices(
            int(W),
            _merge_coronal,
            max_workers=coronal_workers,
            desc="Assembling final union from coronal view volume",
        )

    for view_name in sorted(view_volume_mms.keys()):
        if view_name in consumed:
            continue
        vol = np.asarray(view_volume_mms[view_name])
        if vol.shape != (T, H, W):
            raise ValueError(
                f"View volume '{view_name}' has shape {tuple(vol.shape)}; expected native volume shape {(T, H, W)} "
                "or a handled sagittal/coronal stack."
            )
        native_workers = choose_slice_parallel_workers(int(workers), int(T))

        def _merge_native(t: int, *, _vol: np.ndarray = vol) -> None:
            final_union_mm[int(t), :, :] |= _vol[int(t), :, :]

        parallel_for_indices(
            int(T),
            _merge_native,
            max_workers=native_workers,
            desc=f"Assembling final union from {view_name} view volume",
        )


def assemble_current_view_union_volume(
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
    """Build the single-model final view union.

    The mapping is kept keyed by model name because earlier pipeline stages store temporary
    workspaces under the model stem. v9.1.0 rejects multiple model entries and never
    combines outputs from more than one model.
    """
    if len(view_volumes_by_model) != 1:
        raise ValueError('v9.1.0_SLURM supports exactly one --model; multiple-model inference has been removed')

    model_name = next(iter(view_volumes_by_model.keys()))
    final_union_mm = allocate_workspace_array(
        shape=(T, H, W),
        dtype=np.uint8,
        path=out_path,
        desc='Final single-model view-union volume',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    print(f"\n=== Assembling final view union for model: {model_name} ===")
    assemble_view_volumes_into_native_union(
        final_union_mm=final_union_mm,
        view_volume_mms=view_volumes_by_model[model_name],
        T=T,
        H=H,
        W=W,
        disable_multiplanar=disable_multiplanar,
        workers=int(workers),
    )
    flush_array(final_union_mm)
    return final_union_mm


def union_volume_into_volume(
    dst_mm: np.ndarray,
    src_mm: np.ndarray,
    *,
    workers: int = 1,
    desc: str = 'Union volumes',
) -> None:
    num_slices = int(dst_mm.shape[0]) if int(dst_mm.ndim) > 0 else 0

    def _merge_slice(idx: int) -> None:
        dst_mm[int(idx), :, :] |= np.asarray(src_mm[int(idx)], dtype=np.uint8)

    parallel_for_indices(
        num_slices,
        _merge_slice,
        max_workers=choose_slice_parallel_workers(int(workers), num_slices),
        desc=desc,
        show_progress=False,
    )
    flush_array(dst_mm)


def apply_keep_largest_objects_inplace(
    mask_mm: np.ndarray,
    keep_objects: int,
    temp_dir: Path,
    *,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
) -> Dict[str, int]:
    """Keep only the largest N connected foreground components in the final 3D volume."""
    keep_n = int(keep_objects)
    if keep_n <= 0:
        return {'enabled': 0, 'num_objects': 0, 'kept_objects': 0, 'removed_objects': 0, 'removed_voxels': 0}

    work_dir = temp_dir / 'keep_objects'
    work_dir.mkdir(parents=True, exist_ok=True)
    labels_mm, num_objects, label_paths = label_foreground_volume_streaming(
        mask_mm,
        work_dir / 'final_keep_objects',
        keep_temp=True,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    if int(num_objects) <= keep_n:
        close_memmap_array(labels_mm)
        if not bool(keep_temp):
            for lp in label_paths:
                try:
                    lp.unlink(missing_ok=True)
                except Exception:
                    pass
        return {
            'enabled': 1,
            'num_objects': int(num_objects),
            'kept_objects': int(num_objects),
            'removed_objects': 0,
            'removed_voxels': 0,
        }

    counts = np.zeros((int(num_objects) + 1,), dtype=np.int64)
    count_workers = choose_slice_parallel_workers(int(workers), int(labels_mm.shape[0]))
    partial_counts = [np.zeros_like(counts) for _ in range(count_workers)]
    partial_locks = [threading.Lock() for _ in range(count_workers)]

    def _count_slice(idx: int) -> None:
        worker_slot = int(idx) % int(count_workers)
        binc = np.bincount(np.asarray(labels_mm[int(idx)]).ravel(), minlength=int(num_objects) + 1)
        with partial_locks[worker_slot]:
            partial_counts[worker_slot][:] += binc.astype(np.int64, copy=False)

    parallel_for_indices(
        int(labels_mm.shape[0]),
        _count_slice,
        max_workers=count_workers,
        desc='keep_objects: count object volumes',
        show_progress=True,
    )
    for part in partial_counts:
        counts += part

    object_ids = np.arange(1, int(num_objects) + 1, dtype=np.int64)
    order = np.argsort(counts[1:])[::-1]
    keep_ids = object_ids[order[:keep_n]]
    keep_lookup = np.zeros((int(num_objects) + 1,), dtype=bool)
    keep_lookup[keep_ids] = True

    removed_by_slice = np.zeros((int(mask_mm.shape[0]),), dtype=np.int64)

    def _apply_slice(idx: int) -> None:
        labels_slice = np.asarray(labels_mm[int(idx)])
        keep_slice = keep_lookup[labels_slice]
        current = np.asarray(mask_mm[int(idx)], dtype=bool)
        removed_by_slice[int(idx)] = np.int64(np.count_nonzero(current & (~keep_slice)))
        mask_mm[int(idx), :, :] = keep_slice.astype(np.uint8, copy=False)

    parallel_for_indices(
        int(mask_mm.shape[0]),
        _apply_slice,
        max_workers=choose_slice_parallel_workers(int(workers), int(mask_mm.shape[0])),
        desc=f'keep_objects: keep largest {keep_n}',
        show_progress=True,
    )
    flush_array(mask_mm)

    close_memmap_array(labels_mm)
    if not bool(keep_temp):
        for lp in label_paths:
            try:
                lp.unlink(missing_ok=True)
            except Exception:
                pass

    return {
        'enabled': 1,
        'num_objects': int(num_objects),
        'kept_objects': int(min(keep_n, int(num_objects))),
        'removed_objects': int(max(0, int(num_objects) - keep_n)),
        'removed_voxels': int(np.sum(removed_by_slice, dtype=np.int64)),
    }


def assemble_final_union_after_view_union(
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    T: int,
    H: int,
    W: int,
    disable_multiplanar: bool,
    out_path: Path,
    temp_dir: Path,
    *,
    enable_3d_void_fill: bool = False,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
) -> np.ndarray:
    """Build the final single-model view union and optionally apply one 3D void fill."""
    final_union_mm = assemble_current_view_union_volume(
        view_volumes_by_model=view_volumes_by_model,
        T=T,
        H=H,
        W=W,
        disable_multiplanar=disable_multiplanar,
        out_path=out_path,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        workers=int(workers),
    )

    if bool(enable_3d_void_fill):
        print('\n=== Optional 3D void fill after final global union ===')
        final_void_dir = temp_dir / 'final_global_void_fill'
        final_void_dir.mkdir(parents=True, exist_ok=True)
        fill_3d_voids_inplace_streaming(
            final_union_mm,
            final_void_dir / 'final_union',
            keep_temp=bool(keep_temp),
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
    else:
        print('\n=== Optional 3D void fill disabled (--enable_3d_void_fill not set) ===')
    return final_union_mm


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
      - ``hybrid`` (default): use per-object 3D skeletonization when the compact-relabel bounding box
        is tractable, falling back to the fast slice-graph terminal scan for oversized objects
      - ``scan``: always use the fast slice-graph terminal scan
      - ``skeleton``: always use per-object 3D skeletonization

    The hybrid path keeps 3D skeletonization as the default v9.1.0-compliant behavior while still
    protecting large SLURM-scale volumes from pathological all-voxel skeletonization costs.
    """
    mode = os.environ.get('YOLO_TTA_INTERPOLATION_ENDPOINT_MODE', 'hybrid').strip().lower()
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




@dataclass
class PreparedViewResult:
    model_name: str
    view_name: str
    native_support_mm: np.ndarray
    final_view_volume_mm: Optional[np.ndarray]
    interpolation_stats: List[Dict[str, object]]
    pass_snapshots: Dict[int, VolumeSnapshotRef] = field(default_factory=dict)


@dataclass
class TilePostprocessTask:
    model_name: str
    view_name: str
    config_id: str
    tile_id: str
    tile_mask_mm: np.ndarray
    tile_confmap_mm: np.ndarray
    tile_mask_path: Path
    tile_confmap_path: Path


@dataclass
class TilePostprocessResult:
    model_name: str
    view_name: str
    config_id: str
    tile_id: str
    tile_mask_mm: np.ndarray
    tile_mask_path: Path


@dataclass(frozen=True)
class DeferredTilePostprocessResult:
    model_name: str
    view_name: str
    config_id: str
    tile_id: str
    tile_mask_path: Path
    tile_shape: Tuple[int, int, int]


@dataclass(frozen=True)
class VolumeSnapshotRef:
    model_name: str
    view_name: str
    source: str  # fullframe or tile
    pass_index: int
    path: Path
    shape: Tuple[int, int, int]
    dtype: str = 'uint8'


@dataclass(frozen=True)
class TileGateResult:
    model_name: str
    view_name: str
    config_id: str
    tile_id: str
    gate_stats: Dict[str, int]


@dataclass(frozen=True)
class TileConsolidationResult:
    model_name: str
    view_name: str
    interpolation_stats: List[Dict[str, object]]
    pass_snapshots: Dict[int, VolumeSnapshotRef]


def _view_uses_interpolation(view: ViewInfo, interpolate: int) -> bool:
    return bool(view.family in ('orthogonal', 'radial', 'tilted_transverse') and int(interpolate) > 0)


def _snapshot_volume_for_troubleshooting(
    volume: np.ndarray,
    *,
    temp_dir: Path,
    model_name: str,
    view_name: str,
    source: str,
    pass_index: int,
    enabled: bool,
    workers: int = 1,
) -> Optional[VolumeSnapshotRef]:
    if not bool(enabled):
        return None

    path = (
        temp_dir / 'troubleshooting_pass_snapshots' / str(source) / str(model_name) /
        str(view_name) / f'pass{int(pass_index)}.u8.dat'
    )
    snap_mm = copy_workspace_array(
        np.asarray(volume),
        path,
        desc=f'Troubleshooting snapshot {source}/{model_name}/{view_name}/pass{int(pass_index)}',
        prefer_memory=False,
        workers=int(workers),
    )
    shape = tuple(int(x) for x in np.asarray(snap_mm).shape)
    close_memmap_array(snap_mm)
    return VolumeSnapshotRef(
        model_name=str(model_name),
        view_name=str(view_name),
        source=str(source),
        pass_index=int(pass_index),
        path=path,
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        dtype='uint8',
    )


def _drain_volume_to_mmap(
    volume: np.ndarray,
    path: Path,
    desc: str,
    *,
    workers: int = 1,
) -> np.ndarray:
    drained = copy_workspace_array(
        np.asarray(volume),
        path,
        desc=desc,
        prefer_memory=False,
        workers=int(workers),
    )
    flush_array(drained)
    return drained


def _open_snapshot_ref(ref: VolumeSnapshotRef, mode: str = 'r') -> np.memmap:
    return np.memmap(
        ref.path,
        dtype=np.dtype(ref.dtype),
        mode=str(mode),
        shape=tuple(int(x) for x in ref.shape),
    )


def _resolve_snapshot_ref(
    snapshot_refs: Dict[Tuple[str, str, str, int], VolumeSnapshotRef],
    *,
    source: str,
    model_name: str,
    view_name: str,
    pass_index: int,
) -> Optional[VolumeSnapshotRef]:
    for idx in range(int(pass_index), -1, -1):
        ref = snapshot_refs.get((str(source), str(model_name), str(view_name), int(idx)))
        if ref is not None:
            return ref
    return None


def _volume_has_foreground(mask_mm: np.ndarray) -> bool:
    for idx in range(int(mask_mm.shape[0])):
        if np.any(np.asarray(mask_mm[int(idx)], dtype=bool)):
            return True
    return False


def prepare_view_volume_after_fullframe(
    *,
    model_name: str,
    view: ViewInfo,
    union_mm: np.ndarray,
    confmap_mm: np.ndarray,
    union_path: Path,
    confmap_path: Path,
    temp_dir: Path,
    dense_tiling_active: bool,
    min_conf: float,
    min_radius: float,
    interpolate: int,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_passes: int,
    interpolate_min_radius: float,
    interpolation_search_angle: float,
    keep_temp: bool,
    slice_workers: int,
    interpolation_task_workers: int,
) -> PreparedViewResult:
    baseline_native_volume = union_mm

    cleanup_view_volume_after_prediction_inplace(
        baseline_native_volume,
        confmap_mm,
        view,
        float(min_conf),
        float(min_radius),
        workers=int(slice_workers),
    )

    close_memmap_array(confmap_mm)
    if not keep_temp:
        try:
            confmap_path.unlink(missing_ok=True)
        except Exception:
            pass

    pass_snapshots: Dict[int, VolumeSnapshotRef] = {}
    snap0 = _snapshot_volume_for_troubleshooting(
        baseline_native_volume,
        temp_dir=temp_dir,
        model_name=str(model_name),
        view_name=str(view.name),
        source='fullframe',
        pass_index=0,
        enabled=bool(keep_temp) and int(interpolate) > 0,
        workers=int(slice_workers),
    )
    if snap0 is not None:
        pass_snapshots[0] = snap0

    interpolation_stats: List[Dict[str, object]] = []
    if _view_uses_interpolation(view, int(interpolate)):
        total_passes = int(interpolate_passes)
        for pass_idx in range(1, total_passes + 1):
            stats_local = interpolate_view_volume_pass_inplace(
                mask_mm=baseline_native_volume,
                work_dir=temp_dir / 'interpolation' / model_name / view.name,
                pass_tag=f'pass{pass_idx}',
                max_slice_distance=int(interpolate),
                search_angle_deg=float(interpolation_search_angle),
                interpolation_walk_back=int(interpolation_walk_back),
                interpolation_candidates=int(interpolation_candidates),
                interpolate_min_radius=float(interpolate_min_radius),
                keep_temp=bool(keep_temp),
                prefer_memory=True,
                workers=int(interpolation_task_workers),
            )
            stats_local = dict(stats_local)
            stats_local.update({
                'pass_index': int(pass_idx),
                'model': str(model_name),
                'view': str(view.name),
                'source': 'fullframe',
                'max_slice_distance': int(interpolate),
                'interpolation_walk_back': int(interpolation_walk_back),
                'interpolation_candidates': int(interpolation_candidates),
                'interpolation_search_angle': float(interpolation_search_angle),
            })
            interpolation_stats.append(stats_local)

            snap = _snapshot_volume_for_troubleshooting(
                baseline_native_volume,
                temp_dir=temp_dir,
                model_name=str(model_name),
                view_name=str(view.name),
                source='fullframe',
                pass_index=int(pass_idx),
                enabled=bool(keep_temp) and int(interpolate) > 0,
                workers=int(slice_workers),
            )
            if snap is not None:
                pass_snapshots[int(pass_idx)] = snap

            if int(stats_local.get('added_voxels', 0)) <= 0:
                break
    else:
        # When interpolation is disabled, drain the view-native volume to disk so it can wait
        # cheaply until final backprojection / final union.
        drained_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.noninterpolated_native.u8.dat'
        old_volume = baseline_native_volume
        baseline_native_volume = _drain_volume_to_mmap(
            old_volume,
            drained_path,
            desc=f'{model_name}/{view.name} non-interpolated native drain',
            workers=int(slice_workers),
        )
        if old_volume is not baseline_native_volume:
            close_memmap_array(old_volume)
            if not keep_temp:
                try:
                    union_path.unlink(missing_ok=True)
                except Exception:
                    pass

    final_view_volume: Optional[np.ndarray] = None
    if bool(dense_tiling_active):
        if _view_uses_interpolation(view, int(interpolate)):
            final_copy_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.final_fullframe.u8.dat'
            final_view_volume = copy_workspace_array(
                baseline_native_volume,
                final_copy_path,
                desc=f'{model_name}/{view.name} frozen full-frame support copy',
                prefer_memory=True,
                workers=int(slice_workers),
            )
        else:
            final_view_volume = baseline_native_volume
    elif view.family == 'radial':
        out_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.u8.dat'
        final_view_volume = backproject_radial_volume_to_volume(
            radial_mask_mm=baseline_native_volume,
            radial_view=view,
            out_path=out_path,
            desc=f'Backprojecting {model_name}/{view.name}',
            prefer_memory=False,
        )
        if float(min_radius) > 0.0:
            print(f"Applying --min_radius in the transverse plane for backprojected view '{view.name}'")
            apply_transverse_min_radius_filter_inplace(
                final_view_volume,
                float(min_radius),
                workers=int(slice_workers),
            )
    elif view.family == 'tilted_transverse':
        out_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.u8.dat'
        final_view_volume = backproject_tilted_volume_to_volume(
            tilted_mask_mm=baseline_native_volume,
            tilted_view=view,
            out_path=out_path,
            desc=f'Backprojecting {model_name}/{view.name}',
            prefer_memory=False,
        )
    else:
        final_view_volume = baseline_native_volume

    return PreparedViewResult(
        model_name=str(model_name),
        view_name=str(view.name),
        native_support_mm=baseline_native_volume,
        final_view_volume_mm=final_view_volume,
        interpolation_stats=interpolation_stats,
        pass_snapshots=pass_snapshots,
    )


def gate_tile_volume_against_parent_inplace(
    tile_mask_mm: np.ndarray,
    parent_support_mm: np.ndarray,
    *,
    workers: int = 1,
    desc: str = 'Tile gated OR',
) -> Dict[str, int]:
    """Keep only tile components that intersect the frozen parent support on the same slice.

    The gating result for one slice never affects another slice, so this stage can be parallelized
    aggressively across the slice axis without violating the v9.1.0_SLURM semantics.
    """
    num_slices = int(tile_mask_mm.shape[0])
    accepted_components = np.zeros((num_slices,), dtype=np.int64)
    rejected_components = np.zeros((num_slices,), dtype=np.int64)
    kept_voxels = np.zeros((num_slices,), dtype=np.int64)
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)
    chunk_size = choose_parallel_chunk_size(num_slices, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _process(idx: int) -> None:
        tile_slice = np.asarray(tile_mask_mm[int(idx)], dtype=bool)
        if not np.any(tile_slice):
            tile_mask_mm[int(idx), :, :] = np.uint8(0)
            return

        support_slice = np.asarray(parent_support_mm[int(idx)], dtype=bool)
        keep, accepted, rejected = _build_supported_component_mask(tile_slice, support_slice)
        tile_mask_mm[int(idx), :, :] = keep.astype(np.uint8, copy=False)
        accepted_components[int(idx)] = np.int64(int(accepted))
        rejected_components[int(idx)] = np.int64(int(rejected))
        kept_voxels[int(idx)] = np.int64(np.count_nonzero(keep))

    parallel_for_indices_chunked(
        num_slices,
        _process,
        max_workers=worker_count,
        desc=desc,
        show_progress=False,
        chunk_size=chunk_size,
    )
    flush_array(tile_mask_mm)

    return {
        'accepted_components': int(np.sum(accepted_components, dtype=np.int64)),
        'rejected_components': int(np.sum(rejected_components, dtype=np.int64)),
        'kept_voxels': int(np.sum(kept_voxels, dtype=np.int64)),
    }


def postprocess_tile_volume_after_inference(
    task: TilePostprocessTask,
    *,
    view: ViewInfo,
    min_conf: float,
    min_radius: float,
    keep_temp: bool,
    slice_workers: int,
) -> Optional[TilePostprocessResult]:
    cleanup_view_volume_after_prediction_inplace(
        task.tile_mask_mm,
        task.tile_confmap_mm,
        view,
        float(min_conf),
        float(min_radius),
        workers=int(slice_workers),
    )

    close_memmap_array(task.tile_confmap_mm)
    if not keep_temp:
        try:
            task.tile_confmap_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not _volume_has_foreground(task.tile_mask_mm):
        close_memmap_array(task.tile_mask_mm)
        if not keep_temp:
            try:
                task.tile_mask_path.unlink(missing_ok=True)
            except Exception:
                pass
        return None

    return TilePostprocessResult(
        model_name=str(task.model_name),
        view_name=str(task.view_name),
        config_id=str(task.config_id),
        tile_id=str(task.tile_id),
        tile_mask_mm=task.tile_mask_mm,
        tile_mask_path=task.tile_mask_path,
    )


def spill_waiting_tile_result_to_mmap(
    result: TilePostprocessResult,
    temp_dir: Path,
    *,
    workers: int = 1,
    keep_original: bool = False,
) -> DeferredTilePostprocessResult:
    """Drain a postprocessed tile volume to temp storage while it waits for parent gated OR.

    Tile cleanup can complete long before the matching parent view finishes interpolation. Holding
    those cleaned tile volumes open in RAM needlessly inflates the working set and can delay other
    CPU work. Persist them to a dedicated temp-directory memmap and reopen them only after the
    parent support becomes available.
    """
    wait_path = temp_dir / 'waiting_tiles' / result.model_name / result.view_name / result.config_id / f'{result.tile_id}.u8.dat'
    wait_path.parent.mkdir(parents=True, exist_ok=True)
    tile_shape = tuple(int(x) for x in np.asarray(result.tile_mask_mm).shape)

    is_disk_backed = bool(isinstance(result.tile_mask_mm, np.memmap) and result.tile_mask_path.exists())
    if is_disk_backed:
        flush_array(result.tile_mask_mm)
        close_memmap_array(result.tile_mask_mm)
        if result.tile_mask_path.resolve() != wait_path.resolve():
            if bool(keep_original):
                shutil.copy2(str(result.tile_mask_path), str(wait_path))
            else:
                shutil.move(str(result.tile_mask_path), str(wait_path))
    else:
        spilled_mm = copy_workspace_array(
            np.asarray(result.tile_mask_mm),
            wait_path,
            desc=f'Waiting tile spill {result.model_name}/{result.view_name}/{result.tile_id}',
            prefer_memory=False,
            workers=int(workers),
        )
        close_memmap_array(spilled_mm)
        close_memmap_array(result.tile_mask_mm)

    return DeferredTilePostprocessResult(
        model_name=str(result.model_name),
        view_name=str(result.view_name),
        config_id=str(result.config_id),
        tile_id=str(result.tile_id),
        tile_mask_path=wait_path,
        tile_shape=(int(tile_shape[0]), int(tile_shape[1]), int(tile_shape[2])),
    )


def load_waiting_tile_result_from_mmap(waiting: DeferredTilePostprocessResult) -> TilePostprocessResult:
    tile_mask_mm = np.memmap(
        waiting.tile_mask_path,
        dtype=np.uint8,
        mode='r+',
        shape=tuple(int(x) for x in waiting.tile_shape),
    )
    return TilePostprocessResult(
        model_name=str(waiting.model_name),
        view_name=str(waiting.view_name),
        config_id=str(waiting.config_id),
        tile_id=str(waiting.tile_id),
        tile_mask_mm=tile_mask_mm,
        tile_mask_path=waiting.tile_mask_path,
    )


def gate_tile_volume_into_consolidated_parent(
    task: TilePostprocessResult,
    *,
    parent_support_mm: np.ndarray,
    tile_accumulator_mm: np.ndarray,
    tile_accumulator_lock: threading.Lock,
    keep_temp: bool,
    slice_workers: int,
) -> TileGateResult:
    """Gate one tile volume, then OR accepted components into the parent-view tile accumulator.

    v9.1.0 intentionally consolidates all accepted tiles for a parent view before interpolation.
    This replaces the previous per-tile interpolation path while preserving the mask-wise OR gate:
    accepted tile components still must intersect the frozen parent full-frame support, and accepted
    tiles cannot accept other tiles because the support image is not the accumulator.
    """
    gate_stats = gate_tile_volume_against_parent_inplace(
        task.tile_mask_mm,
        parent_support_mm,
        workers=int(slice_workers),
        desc=f'Gated OR ({task.model_name}/{task.view_name}/{task.tile_id})',
    )

    if int(gate_stats.get('accepted_components', 0)) > 0 and int(gate_stats.get('kept_voxels', 0)) > 0:
        with tile_accumulator_lock:
            union_volume_into_volume(
                tile_accumulator_mm,
                task.tile_mask_mm,
                workers=int(slice_workers),
                desc=f'Consolidate accepted tile {task.model_name}/{task.view_name}/{task.tile_id}',
            )

    close_memmap_array(task.tile_mask_mm)
    if not keep_temp:
        try:
            task.tile_mask_path.unlink(missing_ok=True)
        except Exception:
            pass

    return TileGateResult(
        model_name=str(task.model_name),
        view_name=str(task.view_name),
        config_id=str(task.config_id),
        tile_id=str(task.tile_id),
        gate_stats={k: int(v) for k, v in gate_stats.items()},
    )


def finalize_consolidated_tile_volume_for_parent(
    *,
    model_name: str,
    view: ViewInfo,
    tile_accumulator_mm: np.ndarray,
    destination_mm: np.ndarray,
    destination_lock: threading.Lock,
    temp_dir: Path,
    interpolate: int,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_passes: int,
    interpolate_min_radius: float,
    interpolation_search_angle: float,
    keep_temp: bool,
    slice_workers: int,
    interpolation_task_workers: int,
) -> TileConsolidationResult:
    """Interpolate the consolidated gated-tile volume once for the parent view, then union it.

    The input accumulator already contains the OR of every accepted tile mask for this parent view.
    Interpolation is now performed once on that consolidated volume instead of once per tile.
    """
    interpolation_stats: List[Dict[str, object]] = []
    pass_snapshots: Dict[int, VolumeSnapshotRef] = {}

    if not _volume_has_foreground(tile_accumulator_mm):
        return TileConsolidationResult(
            model_name=str(model_name),
            view_name=str(view.name),
            interpolation_stats=interpolation_stats,
            pass_snapshots=pass_snapshots,
        )

    snap0 = _snapshot_volume_for_troubleshooting(
        tile_accumulator_mm,
        temp_dir=temp_dir,
        model_name=str(model_name),
        view_name=str(view.name),
        source='tile',
        pass_index=0,
        enabled=bool(keep_temp) and int(interpolate) > 0,
        workers=int(slice_workers),
    )
    if snap0 is not None:
        pass_snapshots[0] = snap0

    if _view_uses_interpolation(view, int(interpolate)):
        total_passes = int(interpolate_passes)
        for pass_idx in range(1, total_passes + 1):
            stats_local = interpolate_view_volume_pass_inplace(
                mask_mm=tile_accumulator_mm,
                work_dir=temp_dir / 'tile_interpolation' / str(model_name) / view.name / 'consolidated',
                pass_tag=f'pass{pass_idx}',
                max_slice_distance=int(interpolate),
                search_angle_deg=float(interpolation_search_angle),
                interpolation_walk_back=int(interpolation_walk_back),
                interpolation_candidates=int(interpolation_candidates),
                interpolate_min_radius=float(interpolate_min_radius),
                keep_temp=bool(keep_temp),
                prefer_memory=True,
                workers=int(interpolation_task_workers),
            )
            stats_local = dict(stats_local)
            stats_local.update({
                'pass_index': int(pass_idx),
                'model': str(model_name),
                'view': f'{view.name}[tiles:consolidated]',
                'source': 'tile',
                'max_slice_distance': int(interpolate),
                'interpolation_walk_back': int(interpolation_walk_back),
                'interpolation_candidates': int(interpolation_candidates),
                'interpolation_search_angle': float(interpolation_search_angle),
            })
            interpolation_stats.append(stats_local)

            snap = _snapshot_volume_for_troubleshooting(
                tile_accumulator_mm,
                temp_dir=temp_dir,
                model_name=str(model_name),
                view_name=str(view.name),
                source='tile',
                pass_index=int(pass_idx),
                enabled=bool(keep_temp) and int(interpolate) > 0,
                workers=int(slice_workers),
            )
            if snap is not None:
                pass_snapshots[int(pass_idx)] = snap

            if int(stats_local.get('added_voxels', 0)) <= 0:
                break

    with destination_lock:
        union_volume_into_volume(
            destination_mm,
            tile_accumulator_mm,
            workers=int(slice_workers),
            desc=f'Union consolidated gated tiles {model_name}/{view.name}',
        )

    return TileConsolidationResult(
        model_name=str(model_name),
        view_name=str(view.name),
        interpolation_stats=interpolation_stats,
        pass_snapshots=pass_snapshots,
    )


# --------------------------
# Final object interpolation smoothing (optional)
# --------------------------


def _bbox2d_for_masks(mask_a: np.ndarray, mask_b: np.ndarray, margin: int = 4) -> Optional[Tuple[slice, slice]]:
    ys_a, xs_a = np.nonzero(mask_a)
    ys_b, xs_b = np.nonzero(mask_b)
    if ys_a.size == 0 or ys_b.size == 0:
        return None
    y0 = max(0, int(min(int(np.min(ys_a)), int(np.min(ys_b)))) - int(margin))
    y1 = min(mask_a.shape[0], int(max(int(np.max(ys_a)), int(np.max(ys_b)))) + int(margin) + 1)
    x0 = max(0, int(min(int(np.min(xs_a)), int(np.min(xs_b)))) - int(margin))
    x1 = min(mask_a.shape[1], int(max(int(np.max(xs_a)), int(np.max(xs_b)))) + int(margin) + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    return slice(y0, y1), slice(x0, x1)


def _smooth_one_axis_from_frozen_labels(
    dst_axis_view: np.ndarray,
    frozen_axis_view: np.ndarray,
    labels_axis_view: np.ndarray,
    *,
    axis_name: str,
    workers: int = 1,
) -> int:
    """Create frozen-state N/N+2 -> N+1 object-smoothing deltas along one axis.

    ``dst_axis_view`` is expected to be an initially empty per-view delta volume in the same
    orientation as ``frozen_axis_view``. The source masks and labels are frozen before the pass.
    Newly synthesized middle-slice masks are written only when they are absent from the frozen
    source, so Transverse/Sagittal/Coronal smoothing jobs can run independently and be unioned
    after all view jobs finish.
    """
    axis_len = int(frozen_axis_view.shape[0])
    if axis_len < 3:
        return 0

    added_by_plane = np.zeros((axis_len,), dtype=np.int64)
    worker_count = choose_slice_parallel_workers(int(workers), max(1, axis_len - 2))
    chunk_size = choose_parallel_chunk_size(max(1, axis_len - 2), worker_count, target_chunks_per_worker=2, min_chunk_size=1)
    locks = [threading.Lock() for _ in range(axis_len)]

    def _process(i: int) -> None:
        i0 = int(i)
        i1 = i0 + 2
        im = i0 + 1
        labels0 = np.asarray(labels_axis_view[i0])
        labels1 = np.asarray(labels_axis_view[i1])
        if not np.any(labels0) or not np.any(labels1):
            return

        common = np.intersect1d(np.unique(labels0[labels0 > 0]), np.unique(labels1[labels1 > 0]), assume_unique=False)
        if common.size == 0:
            return

        local = np.zeros(frozen_axis_view.shape[1:], dtype=np.uint8)
        for lbl in common.tolist():
            src = labels0 == int(lbl)
            tgt = labels1 == int(lbl)
            if not np.any(src) or not np.any(tgt):
                continue
            bbox = _bbox2d_for_masks(src, tgt, margin=4)
            if bbox is None:
                continue
            ys, xs = bbox
            src_local = src[ys, xs]
            tgt_local = tgt[ys, xs]
            if not np.any(src_local) or not np.any(tgt_local):
                continue
            synth = (_signed_distance_2d(src_local) + _signed_distance_2d(tgt_local)) >= 0.0
            if not np.any(synth):
                continue
            local_slice = local[ys, xs]
            local_slice[synth] = np.uint8(1)

        if not np.any(local):
            return

        local_new = np.asarray(local, dtype=bool) & (np.asarray(frozen_axis_view[im]) == 0)
        if not np.any(local_new):
            return

        with locks[im]:
            dst = dst_axis_view[im]
            before_new = local_new & (np.asarray(dst) == 0)
            if np.any(before_new):
                added_by_plane[im] += np.int64(np.count_nonzero(before_new))
            dst[local_new] = np.uint8(1)

    parallel_for_indices_chunked(
        axis_len - 2,
        _process,
        max_workers=worker_count,
        desc=f'Object interpolation smoothing ({axis_name})',
        chunk_size=chunk_size,
    )
    flush_array(dst_axis_view)
    return int(np.sum(added_by_plane, dtype=np.int64))


@dataclass
class ObjectSmoothingViewResult:
    view_name: str
    display_name: str
    added_voxels: int
    num_objects: int
    delta_mm: Optional[np.ndarray]
    delta_path: Optional[Path]


def _eligible_object_smoothing_views(views: Sequence[ViewInfo], enable_multiplanar: bool) -> List[ViewInfo]:
    view_by_name = {str(v.name): v for v in views}
    wanted = ['transverse']
    if bool(enable_multiplanar):
        wanted.extend(['sagittal', 'coronal'])
    out: List[ViewInfo] = []
    for name in wanted:
        view = view_by_name.get(name)
        if view is not None and view.family == 'orthogonal':
            out.append(view)
    return out


def _smooth_one_view_volume_to_delta(
    *,
    model_name: str,
    view: ViewInfo,
    source_mm: np.ndarray,
    smoothing_dir: Path,
    keep_temp: bool,
    workers: int,
) -> ObjectSmoothingViewResult:
    """Smooth one frozen parent-view volume and return a view-oriented delta volume.

    The input is a parent full-frame support volume in that view's own orientation. This keeps
    object interpolation smoothing independent per view and avoids using Radial, Tilted Transverse,
    or tiled predictions as smoothing sources.
    """
    view_dir = smoothing_dir / str(model_name) / str(view.name)
    view_dir.mkdir(parents=True, exist_ok=True)
    label_paths: List[Path] = []
    labels_mm: Optional[np.ndarray] = None
    delta_mm: Optional[np.ndarray] = None
    delta_path: Optional[Path] = None

    try:
        labels_mm, num_objects, label_paths = label_foreground_volume_streaming(
            source_mm,
            view_dir / 'frozen_labels',
            keep_temp=True,
            prefer_memory=False,
        )
        if int(num_objects) <= 0:
            return ObjectSmoothingViewResult(
                view_name=str(view.name),
                display_name=pretty_view_name(view),
                added_voxels=0,
                num_objects=int(num_objects),
                delta_mm=None,
                delta_path=None,
            )

        delta_path = view_dir / f'{view.name}.smoothing_delta.u8.dat'
        delta_mm = allocate_workspace_array(
            shape=tuple(int(x) for x in np.asarray(source_mm).shape),
            dtype=np.uint8,
            path=delta_path,
            desc=f'{model_name}/{view.name} object-smoothing delta',
            prefer_memory=False,
        )
        added = _smooth_one_axis_from_frozen_labels(
            dst_axis_view=delta_mm,
            frozen_axis_view=np.asarray(source_mm),
            labels_axis_view=np.asarray(labels_mm),
            axis_name=pretty_view_name(view),
            workers=int(workers),
        )

        if int(added) <= 0:
            close_memmap_array(delta_mm)
            if not bool(keep_temp) and delta_path is not None:
                try:
                    delta_path.unlink(missing_ok=True)
                except Exception:
                    pass
            delta_mm = None
            delta_path = None

        return ObjectSmoothingViewResult(
            view_name=str(view.name),
            display_name=pretty_view_name(view),
            added_voxels=int(added),
            num_objects=int(num_objects),
            delta_mm=delta_mm,
            delta_path=delta_path,
        )
    finally:
        if labels_mm is not None:
            close_memmap_array(labels_mm)
        if not bool(keep_temp):
            for lp in label_paths:
                try:
                    lp.unlink(missing_ok=True)
                except Exception:
                    pass


def union_view_oriented_delta_into_native_inplace(
    dst_native_mm: np.ndarray,
    delta_mm: np.ndarray,
    view: ViewInfo,
    *,
    workers: int = 1,
    desc: str = 'Union object-smoothing delta',
) -> int:
    """Union one view-oriented smoothing delta into the native transverse volume."""
    added_by_slice = np.zeros((max(1, int(delta_mm.shape[0])),), dtype=np.int64)

    if view.name == 'transverse':
        if tuple(int(x) for x in delta_mm.shape) != tuple(int(x) for x in dst_native_mm.shape):
            raise ValueError(
                f"Transverse smoothing delta shape {tuple(delta_mm.shape)} does not match native shape {tuple(dst_native_mm.shape)}"
            )
        num_slices = int(dst_native_mm.shape[0])

        def _merge_transverse(t: int) -> None:
            src = np.asarray(delta_mm[int(t)], dtype=np.uint8)
            if not np.any(src):
                return
            dst = dst_native_mm[int(t)]
            src_bool = src > 0
            added_by_slice[int(t)] = np.int64(np.count_nonzero(src_bool & (np.asarray(dst) == 0)))
            np.maximum(dst, src, out=dst)

        parallel_for_indices(
            num_slices,
            _merge_transverse,
            max_workers=choose_slice_parallel_workers(int(workers), num_slices),
            desc=desc,
            show_progress=False,
        )
    elif view.name == 'sagittal':
        expected = (int(dst_native_mm.shape[1]), int(dst_native_mm.shape[0]), int(dst_native_mm.shape[2]))
        if tuple(int(x) for x in delta_mm.shape) != expected:
            raise ValueError(f"Sagittal smoothing delta shape {tuple(delta_mm.shape)}; expected {expected}")
        num_slices = int(delta_mm.shape[0])
        added_by_slice = np.zeros((num_slices,), dtype=np.int64)

        def _merge_sagittal(y: int) -> None:
            src = np.asarray(delta_mm[int(y)], dtype=np.uint8)
            if not np.any(src):
                return
            dst = dst_native_mm[:, int(y), :]
            src_bool = src > 0
            added_by_slice[int(y)] = np.int64(np.count_nonzero(src_bool & (np.asarray(dst) == 0)))
            np.maximum(dst, src, out=dst)

        parallel_for_indices(
            num_slices,
            _merge_sagittal,
            max_workers=choose_slice_parallel_workers(int(workers), num_slices),
            desc=desc,
            show_progress=False,
        )
    elif view.name == 'coronal':
        expected = (int(dst_native_mm.shape[2]), int(dst_native_mm.shape[0]), int(dst_native_mm.shape[1]))
        if tuple(int(x) for x in delta_mm.shape) != expected:
            raise ValueError(f"Coronal smoothing delta shape {tuple(delta_mm.shape)}; expected {expected}")
        num_slices = int(delta_mm.shape[0])
        added_by_slice = np.zeros((num_slices,), dtype=np.int64)

        def _merge_coronal(x: int) -> None:
            src = np.asarray(delta_mm[int(x)], dtype=np.uint8)
            if not np.any(src):
                return
            dst = dst_native_mm[:, :, int(x)]
            src_bool = src > 0
            added_by_slice[int(x)] = np.int64(np.count_nonzero(src_bool & (np.asarray(dst) == 0)))
            np.maximum(dst, src, out=dst)

        parallel_for_indices(
            num_slices,
            _merge_coronal,
            max_workers=choose_slice_parallel_workers(int(workers), num_slices),
            desc=desc,
            show_progress=False,
        )
    else:
        raise ValueError(f'Object interpolation smoothing is not defined for view: {view.name}')

    flush_array(dst_native_mm)
    return int(np.sum(added_by_slice, dtype=np.int64))


def apply_object_interpolation_smoothing_inplace(
    mask_mm: np.ndarray,
    view_support_by_model: Dict[str, Dict[str, np.ndarray]],
    views: Sequence[ViewInfo],
    temp_dir: Path,
    *,
    enable_multiplanar: bool,
    keep_temp: bool = False,
    workers: int = 1,
) -> Dict[str, int]:
    """Apply v9.1.0 object interpolation smoothing as independent per-view jobs.

    Transverse smoothing is always sourced from the frozen Transverse parent-view support. Sagittal
    and Coronal smoothing are sourced from their own frozen parent-view supports only when
    ``--enable_multiplanar`` is active. Radial, Tilted Transverse, and tiled predictions are not
    used as smoothing sources. Each eligible view writes a separate delta volume; all deltas are
    unioned into ``mask_mm`` only after the view-level smoothing jobs have completed.
    """
    if len(view_support_by_model) != 1:
        raise ValueError('v9.1.0_SLURM supports exactly one --model; multiple-model smoothing has been removed')

    model_name = next(iter(view_support_by_model.keys()))
    support_by_view = view_support_by_model[model_name]
    eligible_views = [v for v in _eligible_object_smoothing_views(views, bool(enable_multiplanar)) if v.name in support_by_view]

    stats: Dict[str, int] = {
        'transverse_added_voxels': 0,
        'sagittal_added_voxels': 0,
        'coronal_added_voxels': 0,
        'transverse_union_added_voxels': 0,
        'sagittal_union_added_voxels': 0,
        'coronal_union_added_voxels': 0,
        'total_union_added_voxels': 0,
    }
    if not eligible_views:
        return stats

    smoothing_dir = temp_dir / 'object_interpolation_smoothing'
    smoothing_dir.mkdir(parents=True, exist_ok=True)

    requested_view_workers = _env_int('YOLO_TTA_OBJECT_SMOOTHING_VIEW_WORKERS', len(eligible_views))
    view_workers = max(1, min(len(eligible_views), int(requested_view_workers)))
    per_view_workers = max(1, int(workers) // max(1, view_workers))
    print(
        'Object interpolation smoothing view workers: '
        f'{int(view_workers)} (per-view slice workers: {int(per_view_workers)})'
    )

    results: List[ObjectSmoothingViewResult] = []
    with ThreadPoolExecutor(max_workers=int(view_workers), thread_name_prefix='object-smoothing-view') as executor:
        future_to_view: Dict[Future, ViewInfo] = {}
        for view in eligible_views:
            future = executor.submit(
                _smooth_one_view_volume_to_delta,
                model_name=str(model_name),
                view=view,
                source_mm=support_by_view[view.name],
                smoothing_dir=smoothing_dir,
                keep_temp=bool(keep_temp),
                workers=int(per_view_workers),
            )
            future_to_view[future] = view

        for future in as_completed(future_to_view):
            result = future.result()
            results.append(result)
            stats[f'{result.view_name}_added_voxels'] = int(result.added_voxels)
            print(
                f'Object interpolation smoothing finished for {result.display_name}: '
                f'objects={int(result.num_objects)}, added_voxels={int(result.added_voxels)}'
            )

    for result in sorted(results, key=lambda r: ('transverse', 'sagittal', 'coronal').index(r.view_name) if r.view_name in ('transverse', 'sagittal', 'coronal') else 99):
        if result.delta_mm is None or int(result.added_voxels) <= 0:
            continue
        view = next(v for v in eligible_views if v.name == result.view_name)
        try:
            union_added = union_view_oriented_delta_into_native_inplace(
                mask_mm,
                result.delta_mm,
                view,
                workers=int(workers),
                desc=f'Union object-smoothing delta ({result.display_name})',
            )
            stats[f'{result.view_name}_union_added_voxels'] = int(union_added)
            stats['total_union_added_voxels'] += int(union_added)
        finally:
            close_memmap_array(result.delta_mm)
            if not bool(keep_temp) and result.delta_path is not None:
                try:
                    result.delta_path.unlink(missing_ok=True)
                except Exception:
                    pass

    flush_array(mask_mm)
    return stats

# --------------------------
# Final outputs
# --------------------------


# --------------------------
# Final outputs
# --------------------------


def _gray_to_rgb_frame(frame_gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame_gray, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return np.ascontiguousarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f'Expected a 2D gray frame, got shape {arr.shape}')
    return np.repeat(arr[:, :, None], 3, axis=2)


def write_overlay_video(
    volume_rgb: np.memmap,  # (T,H,W) gray/luma
    mask_u8: np.ndarray,    # (T,H,W) 0/1
    out_path: Path,
    fps: float,
    show_progress: bool = True,
) -> None:
    """Overlay blue masks (50% alpha) on original transverse frames.

    The working source volume is single-channel; frames are expanded to RGB only for this
    presentation video so the segmentation can remain blue.
    """
    T, H, W = volume_rgb.shape
    assert mask_u8.shape == (T, H, W)

    proc = ffmpeg_ffv1_rgb_writer(
        out_path,
        width=W,
        height=H,
        fps=fps,
    )

    blue = np.array([0, 0, 255], dtype=np.uint8)  # RGB blue

    try:
        assert proc.stdin is not None
        for t in tqdm(range(T), desc=f"Writing overlay video ({out_path.name})", disable=not show_progress):
            frame = _gray_to_rgb_frame(np.asarray(volume_rgb[t]))
            m = mask_u8[t].astype(bool)
            if m.any():
                frame[m] = ((frame[m].astype(np.uint16) + blue.astype(np.uint16)) // 2).astype(np.uint8)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
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
    proc = ffmpeg_ffv1_gray_writer(
        video_path,
        width=W,
        height=H,
        fps=fps,
    )
    try:
        assert proc.stdin is not None
        for t in tqdm(range(T), desc=f"Writing binary MKV ({video_path.name})", disable=not show_progress):
            gray = (np.asarray(mask_u8[t]) * 255).astype(np.uint8)
            proc.stdin.write(np.ascontiguousarray(gray).tobytes())
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
    if view.family == 'tilted_transverse':
        return render_tilted_native_mask_frame(mask_u8, view, int(index))
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

    proc = ffmpeg_ffv1_rgb_writer(
        out_path,
        width=view.src_w,
        height=view.src_h,
        fps=fps,
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
            frame = _gray_to_rgb_frame(np.asarray(frame_rgb))
            m = np.asarray(frame_mask, dtype=bool)
            if np.any(m):
                frame[m] = ((frame[m].astype(np.uint16) + blue.astype(np.uint16)) // 2).astype(np.uint8)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
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
    proc = ffmpeg_ffv1_gray_writer(
        video_path,
        width=view.src_w,
        height=view.src_h,
        fps=fps,
    )
    try:
        assert proc.stdin is not None
        for idx in tqdm(range(view.num_slices), total=view.num_slices, desc=f'Writing binary MKV ({video_path.name})', disable=not show_progress):
            gray = (np.asarray(get_view_mask_frame_by_index(mask_u8, view, int(idx))) * 255).astype(np.uint8)
            proc.stdin.write(np.ascontiguousarray(gray).tobytes())
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
    pretty = view_output_token(view)
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
        pretty = view_output_token(view)
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




def _build_snapshot_native_union_for_view(
    *,
    snapshot_refs: Dict[Tuple[str, str, str, int], VolumeSnapshotRef],
    model_name: str,
    view: ViewInfo,
    pass_index: int,
    temp_dir: Path,
    workers: int,
) -> Optional[np.ndarray]:
    refs: List[VolumeSnapshotRef] = []
    full_ref = _resolve_snapshot_ref(
        snapshot_refs,
        source='fullframe',
        model_name=str(model_name),
        view_name=str(view.name),
        pass_index=int(pass_index),
    )
    tile_ref = _resolve_snapshot_ref(
        snapshot_refs,
        source='tile',
        model_name=str(model_name),
        view_name=str(view.name),
        pass_index=int(pass_index),
    )
    if full_ref is not None:
        refs.append(full_ref)
    if tile_ref is not None:
        refs.append(tile_ref)
    if not refs:
        return None

    shape = tuple(int(x) for x in refs[0].shape)
    for ref in refs[1:]:
        if tuple(int(x) for x in ref.shape) != shape:
            raise ValueError(
                f'Troubleshooting snapshot shape mismatch for {model_name}/{view.name}/pass{pass_index}: '
                f'{shape} vs {tuple(int(x) for x in ref.shape)}'
            )

    union_path = (
        temp_dir / 'troubleshooting_pass_assembly' / f'pass{int(pass_index)}' /
        str(model_name) / f'{view.name}.native_union.u8.dat'
    )
    native_union = allocate_workspace_array(
        shape=shape,
        dtype=np.uint8,
        path=union_path,
        desc=f'Troubleshooting pass {int(pass_index)} native union {model_name}/{view.name}',
        prefer_memory=False,
    )

    for ref in refs:
        src = _open_snapshot_ref(ref, mode='r')
        try:
            union_volume_into_volume(
                native_union,
                src,
                workers=int(workers),
                desc=f'Assembling troubleshooting snapshot {ref.source}/{model_name}/{view.name}/pass{int(ref.pass_index)}',
            )
        finally:
            close_memmap_array(src)

    return native_union


def build_troubleshooting_pass_union(
    *,
    snapshot_refs: Dict[Tuple[str, str, str, int], VolumeSnapshotRef],
    model_names: Sequence[str],
    views: Sequence[ViewInfo],
    pass_index: int,
    T: int,
    H: int,
    W: int,
    enable_multiplanar: bool,
    min_radius: float,
    temp_dir: Path,
    keep_temp: bool,
    workers: int,
) -> np.ndarray:
    if len(model_names) != 1:
        raise ValueError('v9.1.0_SLURM supports exactly one --model; troubleshooting outputs cannot combine multiple models')
    view_volumes_for_pass: Dict[str, Dict[str, np.ndarray]] = {str(model_name): {} for model_name in model_names}
    intermediate_volumes: List[np.ndarray] = []

    try:
        for model_name in model_names:
            for view in views:
                native_union = _build_snapshot_native_union_for_view(
                    snapshot_refs=snapshot_refs,
                    model_name=str(model_name),
                    view=view,
                    pass_index=int(pass_index),
                    temp_dir=temp_dir,
                    workers=int(workers),
                )
                if native_union is None:
                    continue
                intermediate_volumes.append(native_union)

                if view.family == 'radial':
                    if tuple(int(x) for x in native_union.shape) == (int(T), int(H), int(W)):
                        radial_volume = native_union
                    else:
                        radial_volume = backproject_radial_volume_to_volume(
                            radial_mask_mm=native_union,
                            radial_view=view,
                            out_path=(
                                temp_dir / 'troubleshooting_pass_assembly' / f'pass{int(pass_index)}' /
                                str(model_name) / f'{view.name}.backprojected.u8.dat'
                            ),
                            desc=f'Backprojecting troubleshooting pass {int(pass_index)} {model_name}/{view.name}',
                            prefer_memory=False,
                        )
                        intermediate_volumes.append(radial_volume)
                    if float(min_radius) > 0.0:
                        apply_transverse_min_radius_filter_inplace(
                            radial_volume,
                            float(min_radius),
                            workers=int(workers),
                        )
                    view_volumes_for_pass[str(model_name)][view.name] = radial_volume
                elif view.family == 'tilted_transverse':
                    if tuple(int(x) for x in native_union.shape) == (int(T), int(H), int(W)):
                        tilted_volume = native_union
                    else:
                        tilted_volume = backproject_tilted_volume_to_volume(
                            tilted_mask_mm=native_union,
                            tilted_view=view,
                            out_path=(
                                temp_dir / 'troubleshooting_pass_assembly' / f'pass{int(pass_index)}' /
                                str(model_name) / f'{view.name}.backprojected.u8.dat'
                            ),
                            desc=f'Backprojecting troubleshooting pass {int(pass_index)} {model_name}/{view.name}',
                            prefer_memory=False,
                        )
                        intermediate_volumes.append(tilted_volume)
                    view_volumes_for_pass[str(model_name)][view.name] = tilted_volume
                else:
                    view_volumes_for_pass[str(model_name)][view.name] = native_union

        # Troubleshooting pass outputs are stage snapshots taken before the corresponding
        # interpolation pass. They must not run the final 3D void fill; v9.1.0 keeps
        # 3D void fill out of troubleshooting pre-pass outputs and applies it only when
        # --enable_3d_void_fill is active for the normal final output.
        pass_union = assemble_current_view_union_volume(
            view_volumes_by_model=view_volumes_for_pass,
            T=int(T),
            H=int(H),
            W=int(W),
            disable_multiplanar=not bool(enable_multiplanar),
            out_path=temp_dir / 'troubleshooting_pass_outputs' / f'final_union_pass{int(pass_index)}.u8.dat',
            prefer_memory=False,
            workers=int(workers),
        )
        return pass_union
    finally:
        for vol in intermediate_volumes:
            close_memmap_array(vol)


def schedule_troubleshooting_pass_outputs(
    *,
    output_manager: BackgroundOutputManager,
    snapshot_refs: Dict[Tuple[str, str, str, int], VolumeSnapshotRef],
    model_names: Sequence[str],
    views: Sequence[ViewInfo],
    volume_rgb: np.ndarray,
    out_dir: Path,
    stem: str,
    fps: float,
    save_binary_pattern_value: Optional[str],
    save_labels_pattern_value: Optional[str],
    save_nrrd_flag: bool,
    save_multiplanar_flag: bool,
    total_passes: int,
    T: int,
    H: int,
    W: int,
    enable_multiplanar: bool,
    min_radius: float,
    temp_dir: Path,
    keep_temp: bool,
    frame_workers: int,
    workers: int,
) -> Dict[str, Path]:
    if len(model_names) != 1:
        raise ValueError('v9.1.0_SLURM supports exactly one --model; troubleshooting outputs cannot combine multiple models')
    if int(total_passes) < 0 or not snapshot_refs:
        return {}

    # Save only the additional pre-interpolation troubleshooting passes. For example,
    # --interpolate_passes 2 saves Pass0 and Pass1 here; the normal final output is Pass2
    # and is saved untagged by the main output path. Use actual captured snapshot indices
    # so early interpolation termination does not synthesize duplicate pass outputs.
    available_prepass_indices = sorted({
        int(ref.pass_index)
        for ref in snapshot_refs.values()
        if 0 <= int(ref.pass_index) < int(total_passes)
    })
    if 0 not in available_prepass_indices:
        available_prepass_indices.insert(0, 0)

    all_paths: Dict[str, Path] = {}
    for pass_idx in available_prepass_indices:
        print(f"\n=== Scheduling troubleshooting pre-interpolation outputs: pass {int(pass_idx)} ===")
        pass_union = build_troubleshooting_pass_union(
            snapshot_refs=snapshot_refs,
            model_names=model_names,
            views=views,
            pass_index=int(pass_idx),
            T=int(T),
            H=int(H),
            W=int(W),
            enable_multiplanar=bool(enable_multiplanar),
            min_radius=float(min_radius),
            temp_dir=temp_dir,
            keep_temp=bool(keep_temp),
            workers=int(workers),
        )
        tag = f'Pass{int(pass_idx)}'
        pass_paths, pass_futures = collect_pipeline_output_futures(
            output_manager.executor,
            volume_rgb=volume_rgb,
            mask_u8=pass_union,
            out_dir=out_dir,
            stem=stem,
            fps=float(fps),
            save_binary_pattern_value=save_binary_pattern_value,
            save_labels_pattern_value=save_labels_pattern_value,
            save_nrrd_flag=bool(save_nrrd_flag),
            tag=tag,
            frame_workers=int(frame_workers),
            show_progress=False,
        )
        if bool(save_multiplanar_flag):
            extra_paths, extra_futures = collect_multiplanar_output_futures(
                output_manager.executor,
                volume_rgb=volume_rgb,
                mask_u8=pass_union,
                out_dir=out_dir,
                stem=stem,
                fps=float(fps),
                save_binary_pattern_value=save_binary_pattern_value,
                save_labels_pattern_value=save_labels_pattern_value,
                tag=tag,
                frame_workers=int(frame_workers),
                show_progress=False,
            )
            pass_paths.update(extra_paths)
            pass_futures.extend(extra_futures)

        output_manager.submit(BackgroundOutputSubmission(
            label=f'troubleshooting pass {int(pass_idx)} outputs',
            result_paths=pass_paths,
            futures=pass_futures,
            resources=[pass_union],
        ))
        for key, path in pass_paths.items():
            all_paths[f'troubleshooting_pass{int(pass_idx)}_{key}'] = path
    return all_paths

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


def write_low_quality_video_copy(src: Path, dst: Path) -> Path:
    _require_bin('ffmpeg')
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg',
        '-y',
        '-v', 'error',
        '-i', str(src),
        '-vsync', '0',
        '-an',
        '-vf', 'scale=1024:1024:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-pix_fmt', 'yuv420p',
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def save_low_quality_video_copies(
    root_dir: Path,
    *,
    workers: int = 1,
    show_progress: bool = True,
) -> Path:
    low_root = root_dir / 'low_quality'
    low_root.mkdir(parents=True, exist_ok=True)
    src_videos = [
        p for p in sorted(root_dir.rglob('*.mkv'))
        if low_root not in p.parents and 'temp' not in p.parts
    ]
    if not src_videos:
        return low_root

    def _transcode(idx: int) -> None:
        src = src_videos[int(idx)]
        rel = src.relative_to(root_dir)
        dst = (low_root / rel).with_suffix('.mp4')
        write_low_quality_video_copy(src, dst)

    parallel_for_indices(
        len(src_videos),
        _transcode,
        max_workers=choose_slice_parallel_workers(int(workers), len(src_videos)),
        desc='Writing low-quality video copies',
        show_progress=show_progress,
    )
    return low_root


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
        if not cv2.imwrite(str(out_path), frame):
            raise RuntimeError(f'Failed to write image: {out_path}')

    parallel_for_indices(
        total,
        _write_frame,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=f'Writing {view.name} image sequence',
        show_progress=show_progress,
    )
    return view_dir


def union_view_volume_for_single_model(
    view_name: str,
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
) -> np.ndarray:
    if len(view_volumes_by_model) != 1:
        raise ValueError('v9.1.0_SLURM supports exactly one --model; multiple-model inference has been removed')
    model_name = next(iter(view_volumes_by_model.keys()))
    if view_name not in view_volumes_by_model[model_name]:
        raise KeyError(f'No view volume found for {view_name}')

    source = np.asarray(view_volumes_by_model[model_name][view_name])
    source_shape = tuple(int(x) for x in source.shape)
    union_mm = allocate_workspace_array(
        shape=source_shape,
        dtype=np.uint8,
        path=out_path,
        desc=f'Single-model view volume copy for TTA labels ({view_name})',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    num_slices = int(source_shape[0]) if len(source_shape) > 0 else 0

    def _copy_slice(idx: int) -> None:
        union_mm[int(idx)] = np.asarray(source[int(idx)], dtype=np.uint8)

    parallel_for_indices(
        num_slices,
        _copy_slice,
        max_workers=choose_slice_parallel_workers(int(workers), num_slices),
        desc=f'Copy single-model view volume ({view_name})',
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
        union_mask = union_view_volume_for_single_model(
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
    enable_3d_void_fill: bool,
    object_interpolation_smoothing_stats: Optional[Dict[str, int]],
    keep_objects_stats: Optional[Dict[str, int]],
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
    lines.append(f'3D void fill: {"enabled" if bool(enable_3d_void_fill) else "disabled"}; when enabled, it is applied once after the final global union')
    if bool(enable_3d_void_fill):
        lines.append('3D void fill background connectivity: default 6-connected; override with YOLO_TTA_VOIDFILL_CONNECTIVITY=18 or 26 if needed')
    lines.append(f'Augmentation workers: {int(augmentation_workers)}')
    lines.append(f'Slice-parallel postprocess workers: {int(slice_postprocess_workers)}')
    lines.append(f'Interpolation workers: {int(interpolation_workers)}')
    lines.append(f'Output workers: {int(output_workers)}')
    lines.append('Worker oversubscription: intentional; the default worker budget targets 2x the visible CPU allocation to overlap GPU waits, ffmpeg IO, and slice-parallel CPU work.')
    lines.append(f'Model: {str(Path(model_paths[0])) if model_paths else "<none>"}')
    lines.append(f'Views: {", ".join(view_names)}')

    lines.append('')
    lines.append('View statistics:')
    total_prediction_count = 0
    for view_key, label in (
        ('transverse', 'Transverse'),
        ('tilted_transverse', 'Tilted Transverse'),
        ('sagittal', 'Sagittal'),
        ('coronal', 'Coronal'),
        ('radial', 'Radial'),
    ):
        count = int(view_prediction_stats.get(view_key, 0))
        total_prediction_count += count
        lines.append(f'  {label}: predictions={count}')
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

    if object_interpolation_smoothing_stats is not None:
        lines.append('')
        lines.append('Object interpolation smoothing: enabled; per-view jobs are run independently and unioned after all smoothing jobs finish')
        lines.append(
            '  per-view added_voxels: '
            f"transverse={int(object_interpolation_smoothing_stats.get('transverse_added_voxels', 0))}, "
            f"sagittal={int(object_interpolation_smoothing_stats.get('sagittal_added_voxels', 0))}, "
            f"coronal={int(object_interpolation_smoothing_stats.get('coronal_added_voxels', 0))}"
        )
        lines.append(
            '  union_added_voxels: '
            f"transverse={int(object_interpolation_smoothing_stats.get('transverse_union_added_voxels', 0))}, "
            f"sagittal={int(object_interpolation_smoothing_stats.get('sagittal_union_added_voxels', 0))}, "
            f"coronal={int(object_interpolation_smoothing_stats.get('coronal_union_added_voxels', 0))}, "
            f"total={int(object_interpolation_smoothing_stats.get('total_union_added_voxels', 0))}"
        )
    else:
        lines.append('')
        lines.append('Object interpolation smoothing: disabled')

    if keep_objects_stats is not None:
        lines.append('')
        lines.append(
            'keep_objects: '
            f"enabled, objects={int(keep_objects_stats.get('num_objects', 0))}, "
            f"kept={int(keep_objects_stats.get('kept_objects', 0))}, "
            f"removed_objects={int(keep_objects_stats.get('removed_objects', 0))}, "
            f"removed_voxels={int(keep_objects_stats.get('removed_voxels', 0))}"
        )
    else:
        lines.append('')
        lines.append('keep_objects: disabled')

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

    model_path = str(args.model).strip()
    if not model_path:
        raise ValueError('--model must specify one YOLO segmentation model path')
    if ',' in model_path:
        raise ValueError('v9.1.0_SLURM accepts a single --model path; multiple-model inference has been removed')
    model_path_resolved = str(Path(model_path).expanduser().resolve())
    if not Path(model_path_resolved).exists():
        raise FileNotFoundError(model_path_resolved)
    model_paths = [model_path_resolved]

    angles = _parse_angles(args.angle) or [0.0, 120.0, 240.0]
    tilt_angles = resolve_tilt_angles(args.tilt_angle)
    tilt_directions = resolve_tilt_directions(args.tilt_direction)
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
    if int(args.keep_objects) < 0:
        raise ValueError('--keep_objects must be >= 0')
    if float(args.azimuth_angle) < 0:
        raise ValueError('--azimuth_angle must be >= 0')
    for tilt_angle in tilt_angles:
        if not (0.0 < float(tilt_angle) < 45.0):
            raise ValueError('--tilt_angle values must be greater than 0 and less than 45')
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

    vol_path = temp_dir / 'input_volume.gray8.dat'
    volume_rgb = decode_video_to_memmap_gray8(
        input_video=input_path,
        out_dat=vol_path,
        num_frames=T,
        width=W,
        height=H,
        overwrite=False,
        prefer_memory=True,
    )
    (temp_dir / 'input_volume.meta.json').write_text(
        json.dumps({'shape': [T, H, W], 'dtype': 'uint8', 'channels': 1, 'fps': fps}, indent=2)
    )

    views = get_view_infos(
        T=T,
        H=H,
        W=W,
        disable_multiplanar=not bool(args.enable_multiplanar),
        azimuth_angle=float(args.azimuth_angle),
        include_radial=True,
        tilt_angles=tilt_angles,
        tilt_directions=tilt_directions,
    )
    cartesian_views = orthogonal_views_only(views)
    interpolating_views = [v for v in views if _view_uses_interpolation(v, int(args.interpolate))]

    model_name = Path(model_paths[0]).stem
    print(f'Loading model: {model_name} ({model_paths[0]})')
    yolo_model = load_ultralytics_model(model_paths[0], task='segment')
    # v9.1.0 has no multiple-model inference. A single-item list is retained only to minimize churn in
    # scheduling structures keyed by model stem.
    yolo_models: List[Tuple[str, object]] = [(model_name, yolo_model)]
    yolo_by_model_name: Dict[str, object] = {model_name: yolo_model}

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
        max_tasks=max(1, len(yolo_models) * max(1, len(interpolating_views))),
    )
    output_workers = resolve_worker_count(
        0,
        'YOLO_TTA_OUTPUT_WORKERS',
        worker_budget,
    )
    output_frame_workers = max(1, _env_int('YOLO_TTA_OUTPUT_FRAME_WORKERS', max(1, min(8, output_workers))))
    slice_postprocess_workers = max(1, int(augmentation_workers))
    predict_postprocess_workers = max(1, min(32, _env_int('YOLO_TTA_PREDICT_POSTPROCESS_WORKERS', slice_postprocess_workers)))

    parent_postprocess_workers = max(1, min(int(interpolation_workers), max(1, len(yolo_models) * max(1, len(views)))))
    parent_interpolation_task_workers = max(
        1,
        min(16, _env_int('YOLO_TTA_INTERPOLATION_TASK_WORKERS', max(1, _cpu_count() // max(1, parent_postprocess_workers)))),
    )

    tile_postprocess_workers_default = int(worker_budget)
    tile_postprocess_workers = max(1, _env_int('YOLO_TTA_TILE_POSTPROCESS_WORKERS', tile_postprocess_workers_default))
    tile_slice_postprocess_workers_default = int(worker_budget)
    tile_slice_postprocess_workers = max(
        1,
        _env_int('YOLO_TTA_TILE_SLICE_WORKERS', tile_slice_postprocess_workers_default),
    )
    tile_interpolation_task_workers = max(1, _env_int('YOLO_TTA_TILE_INTERPOLATION_TASK_WORKERS', 1))

    print(f'Allocated CPU count: {_cpu_count()}')
    print(f'Worker budget: {worker_budget}')
    print('Worker oversubscription is intentional (default budget = 2x visible CPUs).')
    print(f'Augmentation workers: {augmentation_workers}')
    print(f'Slice-parallel postprocess workers: {slice_postprocess_workers}')
    print(f'Inference postprocess workers: {predict_postprocess_workers}')
    print(
        'Parent full-frame postprocess workers: '
        f'{parent_postprocess_workers} (per-parent interpolation workers: {parent_interpolation_task_workers})'
    )
    print(
        'Tile postprocess workers: '
        f'{tile_postprocess_workers} (per-tile slice workers: {tile_slice_postprocess_workers}, '
        f'consolidated-tile interpolation workers: {tile_interpolation_task_workers})'
    )
    print(f'Background output workers: {output_workers} (frame workers per labels/TIFF task: {output_frame_workers})')

    output_manager = BackgroundOutputManager(max_workers=output_workers)

    if augmentation_workers > 1 or interpolation_workers > 1 or slice_postprocess_workers > 1 or output_workers > 1:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    dense_tiling_active = len(tile_configs) > 0
    view_infos_by_name: Dict[str, ViewInfo] = {view.name: view for view in views}
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    native_view_support_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    radial_native_output_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    tilted_native_output_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    view_volume_locks: Dict[Tuple[str, str], threading.Lock] = {
        (model_name, view.name): threading.Lock()
        for model_name, _ in yolo_models
        for view in views
    }

    aug_jobs_by_view: Dict[str, List[AugJob]] = {}
    aug_job_lookup_by_view: Dict[str, Dict[str, AugJob]] = {}
    tile_jobs_by_view_config: Dict[str, Dict[str, List[DenseTileJob]]] = {view.name: {} for view in views}
    tile_jobs_by_aug: Dict[Tuple[str, str], List[DenseTileJob]] = {}
    view_prediction_stats: Dict[str, int] = {
        'transverse': 0,
        'tilted_transverse': 0,
        'sagittal': 0,
        'coronal': 0,
        'radial': 0,
    }
    interpolation_stats: List[Dict[str, object]] = []
    pass_snapshot_refs: Dict[Tuple[str, str, str, int], VolumeSnapshotRef] = {}

    def _register_pass_snapshots(snapshots: Dict[int, VolumeSnapshotRef]) -> None:
        for ref in snapshots.values():
            pass_snapshot_refs[(str(ref.source), str(ref.model_name), str(ref.view_name), int(ref.pass_index))] = ref

    for view in views:
        jobs = build_aug_jobs_for_view(
            view=view,
            angles=angles,
            out_size=args.imgsz,
            temp_dir=temp_dir,
        )
        aug_jobs_by_view[view.name] = jobs
        aug_job_lookup_by_view[view.name] = {job.aug_id: job for job in jobs}
        if dense_tiling_active:
            jobs_by_config: Dict[str, List[DenseTileJob]] = {}
            for tile_cfg in tile_configs:
                cfg_jobs: List[DenseTileJob] = []
                for aug_job in jobs:
                    built_jobs = build_dense_tile_jobs_for_aug(
                        view=view,
                        aug_job=aug_job,
                        tile_cfg=tile_cfg,
                        out_size=int(args.imgsz),
                        temp_dir=temp_dir,
                    )
                    cfg_jobs.extend(built_jobs)
                    tile_jobs_by_aug.setdefault((view.name, aug_job.aug_id), []).extend(built_jobs)
                if cfg_jobs:
                    jobs_by_config[tile_cfg.config_id] = cfg_jobs
            if jobs_by_config:
                tile_jobs_by_view_config[view.name] = jobs_by_config

    tile_expected_by_parent: Dict[Tuple[str, str], int] = {}
    if dense_tiling_active:
        for view in views:
            expected_for_view = sum(
                len(tile_jobs_by_aug.get((view.name, aug_job.aug_id), []))
                for aug_job in aug_jobs_by_view.get(view.name, [])
            )
            if expected_for_view <= 0:
                continue
            for model_name, _ in yolo_models:
                tile_expected_by_parent[(str(model_name), str(view.name))] = int(expected_for_view)

    view_frame_caches: Dict[str, np.ndarray] = {}
    view_frame_cache_paths: Dict[str, Path] = {}
    for view in views:
        if should_cache_view_frames(view, dense_tiling_active):
            cache_path = temp_dir / 'view_frames' / f'{view.name}.gray8.dat'
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            view_frame_caches[view.name] = build_view_frame_cache(
                volume_rgb=volume_rgb,
                view=view,
                out_path=cache_path,
                desc=f'{view.name} native frame cache',
                prefer_memory=True,
                workers=max(1, int(augmentation_workers)),
            )
            view_frame_cache_paths[view.name] = cache_path

    baseline_union_by_model_view: Dict[Tuple[str, str], np.ndarray] = {}
    baseline_confmap_by_model_view: Dict[Tuple[str, str], np.ndarray] = {}
    baseline_union_paths: Dict[Tuple[str, str], Path] = {}
    baseline_confmap_paths: Dict[Tuple[str, str], Path] = {}
    fullframe_remaining: Dict[Tuple[str, str], int] = {}

    for view in views:
        for model_name, _ in yolo_models:
            union_path = temp_dir / 'union' / model_name / f'{view.name}.union.u8.dat'
            confmap_path = temp_dir / 'union' / model_name / f'{view.name}.confmap.u8.dat'
            union_path.parent.mkdir(parents=True, exist_ok=True)

            baseline_union_by_model_view[(model_name, view.name)] = allocate_workspace_array(
                shape=(view.num_slices, view.src_h, view.src_w),
                dtype=np.uint8,
                path=union_path,
                desc=f'{model_name}/{view.name} baseline union workspace',
                prefer_memory=True,
            )
            baseline_confmap_by_model_view[(model_name, view.name)] = allocate_workspace_array(
                shape=(view.num_slices, view.src_h, view.src_w),
                dtype=np.uint8,
                path=confmap_path,
                desc=f'{model_name}/{view.name} baseline confidence workspace',
                prefer_memory=True,
            )
            baseline_union_paths[(model_name, view.name)] = union_path
            baseline_confmap_paths[(model_name, view.name)] = confmap_path
            fullframe_remaining[(model_name, view.name)] = int(len(aug_jobs_by_view[view.name]))

    total_fullframe_jobs = sum(len(v) for v in aug_jobs_by_view.values())
    total_initial_tile_jobs = sum(
        len(tile_jobs_by_aug.get((view.name, aug_job.aug_id), []))
        for view in views if view.family != 'radial'
        for aug_job in aug_jobs_by_view[view.name]
    )
    total_canvas_render_tasks = sum(
        1
        for view in views if view.family != 'radial'
        for aug_job in aug_jobs_by_view[view.name]
        if tile_jobs_by_aug.get((view.name, aug_job.aug_id), [])
    )
    total_tile_render_batches = sum(
        1
        for view in views
        for aug_job in aug_jobs_by_view[view.name]
        if tile_jobs_by_aug.get((view.name, aug_job.aug_id), [])
    )
    total_tile_render_tasks = int(total_canvas_render_tasks + total_tile_render_batches)
    total_render_tasks = int(total_fullframe_jobs + total_tile_render_tasks)
    video_render_workers = max(
        1,
        min(
            int(augmentation_workers),
            max(1, _env_int('YOLO_TTA_VIDEO_RENDER_WORKERS', max(1, min(augmentation_workers, total_render_tasks)))),
        ),
    )
    use_split_render_executors = bool(dense_tiling_active) and int(video_render_workers) > 1 and int(total_tile_render_tasks) > 0
    if use_split_render_executors:
        fullframe_render_workers, tile_render_workers = split_video_render_workers(
            int(video_render_workers),
            int(total_fullframe_jobs),
            int(total_tile_render_tasks),
        )
    else:
        fullframe_render_workers = int(video_render_workers)
        tile_render_workers = 0
    per_video_render_workers = max(1, int(max(1, augmentation_workers) // max(1, video_render_workers)))
    print(
        f'Video render workers: {video_render_workers} '
        f'(full-frame queue: {fullframe_render_workers}, tile/canvas queue: {tile_render_workers}, '
        f'per-video slice workers: {per_video_render_workers}, render tasks: {total_render_tasks})'
    )

    fullframe_render_executor = ThreadPoolExecutor(max_workers=int(max(1, fullframe_render_workers)), thread_name_prefix='fullframe-render')
    tile_render_executor = (
        ThreadPoolExecutor(max_workers=int(tile_render_workers), thread_name_prefix='tile-render')
        if int(tile_render_workers) > 0 else None
    )
    parent_postprocess_executor = ThreadPoolExecutor(max_workers=int(parent_postprocess_workers), thread_name_prefix='parent-postprocess')
    tile_postprocess_executor = ThreadPoolExecutor(max_workers=int(tile_postprocess_workers), thread_name_prefix='tile-postprocess')

    def _tile_render_submit(fn: Callable[..., object], /, *fn_args: object) -> Future:
        executor = tile_render_executor if tile_render_executor is not None else fullframe_render_executor
        return executor.submit(fn, *fn_args)


    canvas_futures_by_aug: Dict[Tuple[str, str], Future] = {}
    canvas_future_processed: set[Tuple[str, str]] = set()
    pending_tile_batches_by_aug: Dict[Tuple[str, str], Tuple[ViewInfo, AugJob, Tuple[DenseTileJob, ...]]] = {}
    fullframe_video_futures: Dict[Future, Tuple[ViewInfo, AugJob]] = {}
    tile_batch_video_futures: Dict[Future, Tuple[ViewInfo, Tuple[DenseTileJob, ...]]] = {}
    pending_fullframe_futures: set[Future] = set()
    pending_tile_video_futures: set[Future] = set()
    rendered_tile_jobs_by_view: Dict[str, Dict[str, DenseTileJob]] = {view.name: {} for view in views}
    tile_batch_render_submitted: set[Tuple[str, str]] = set()

    ready_fullframe: deque[Tuple[ViewInfo, AugJob]] = deque()
    ready_tile_infer: deque[Tuple[str, ViewInfo, DenseTileJob]] = deque()
    tile_inference_enqueued: set[Tuple[str, str, str]] = set()
    tile_inference_done: set[Tuple[str, str, str]] = set()

    view_processing_futures: Dict[Future, Tuple[str, str]] = {}
    view_processing_submitted: set[Tuple[str, str]] = set()
    tile_cleanup_futures: Dict[Future, Tuple[str, str, str]] = {}
    postprocessed_tiles_waiting_by_parent: Dict[Tuple[str, str], Dict[str, DeferredTilePostprocessResult]] = {}
    tile_finalize_futures: Dict[Future, Tuple[str, str, str]] = {}
    tile_consolidation_futures: Dict[Future, Tuple[str, str]] = {}
    tile_accumulator_by_parent: Dict[Tuple[str, str], np.ndarray] = {}
    tile_accumulator_paths: Dict[Tuple[str, str], Path] = {}
    tile_completed_by_parent: Dict[Tuple[str, str], set[str]] = {}
    tile_consolidation_submitted: set[Tuple[str, str]] = set()

    def _get_canvas_future(view: ViewInfo, job: AugJob) -> Future:
        key = (view.name, job.aug_id)
        fut = canvas_futures_by_aug.get(key)
        if fut is not None:
            return fut
        fut = _tile_render_submit(
            ensure_canvas_video_only,
            volume_rgb,
            view,
            job,
            float(fps),
            int(per_video_render_workers),
            view_frame_caches.get(view.name),
            False,
        )
        canvas_futures_by_aug[key] = fut
        return fut

    def _submit_fullframe_video(view: ViewInfo, job: AugJob) -> None:
        fut = fullframe_render_executor.submit(
            ensure_fullframe_video_only,
            volume_rgb,
            view,
            job,
            float(fps),
            int(per_video_render_workers),
            view_frame_caches.get(view.name),
            False,
        )
        fullframe_video_futures[fut] = (view, job)
        pending_fullframe_futures.add(fut)

    def _submit_tile_video_batch_now(view: ViewInfo, aug_job: AugJob, tile_jobs: Sequence[DenseTileJob]) -> None:
        batch_key = (view.name, aug_job.aug_id)
        if batch_key in tile_batch_render_submitted:
            return
        tile_jobs = tuple(tile_jobs)
        if not tile_jobs:
            return
        fut = _tile_render_submit(
            ensure_dense_tile_video_batch_from_canvas,
            aug_job,
            view,
            tile_jobs,
            float(fps),
            False,
        )
        tile_batch_video_futures[fut] = (view, tile_jobs)
        pending_tile_video_futures.add(fut)
        tile_batch_render_submitted.add(batch_key)

    def _submit_tile_video_batch(view: ViewInfo, aug_job: AugJob, tile_jobs: Sequence[DenseTileJob]) -> None:
        batch_key = (view.name, aug_job.aug_id)
        if batch_key in tile_batch_render_submitted:
            return
        tile_jobs = tuple(tile_jobs)
        if not tile_jobs:
            return
        canvas_future = _get_canvas_future(view, aug_job)
        if canvas_future.done():
            canvas_future.result()
            _submit_tile_video_batch_now(view, aug_job, tile_jobs)
            return
        pending_tile_batches_by_aug[batch_key] = (view, aug_job, tile_jobs)

    for view, aug_job in iter_aug_jobs_round_robin(views, aug_jobs_by_view):
        _submit_fullframe_video(view, aug_job)

    for view, aug_job in iter_aug_jobs_round_robin(views, aug_jobs_by_view):
        if dense_tiling_active and view.family != 'radial':
            tile_jobs = tile_jobs_by_aug.get((view.name, aug_job.aug_id), [])
            if tile_jobs:
                _submit_tile_video_batch(view, aug_job, tile_jobs)

    def _push_ready_fullframe(view: ViewInfo, job: AugJob) -> None:
        ready_fullframe.append((view, job))

    def _push_ready_tile(model_name: str, view: ViewInfo, tile_job: DenseTileJob) -> None:
        ready_tile_infer.append((str(model_name), view, tile_job))


    def _maybe_enqueue_tile_inference(model_name: str, view_name: str, tile_job: DenseTileJob) -> None:
        ready_key = (str(model_name), str(view_name), str(tile_job.tile_id))
        if ready_key in tile_inference_done or ready_key in tile_inference_enqueued:
            return
        _push_ready_tile(str(model_name), view_infos_by_name[view_name], tile_job)
        tile_inference_enqueued.add(ready_key)

    def _submit_view_prepare(model_name: str, view: ViewInfo) -> None:
        key = (str(model_name), str(view.name))
        if key in view_processing_submitted:
            return
        view_processing_submitted.add(key)
        union_mm = baseline_union_by_model_view.pop(key)
        confmap_mm = baseline_confmap_by_model_view.pop(key)
        union_path = baseline_union_paths.pop(key)
        confmap_path = baseline_confmap_paths.pop(key)
        fut = parent_postprocess_executor.submit(
            prepare_view_volume_after_fullframe,
            model_name=str(model_name),
            view=view,
            union_mm=union_mm,
            confmap_mm=confmap_mm,
            union_path=union_path,
            confmap_path=confmap_path,
            temp_dir=temp_dir,
            dense_tiling_active=bool(dense_tiling_active),
            min_conf=float(args.min_conf),
            min_radius=float(args.min_radius),
            interpolate=int(args.interpolate),
            interpolation_walk_back=int(args.interpolation_walk_back),
            interpolation_candidates=int(args.interpolation_candidates),
            interpolate_passes=int(args.interpolate_passes),
            interpolate_min_radius=float(args.interpolate_min_radius),
            interpolation_search_angle=float(args.interpolation_search_angle),
            keep_temp=bool(args.troubleshooting),
            slice_workers=int(slice_postprocess_workers),
            interpolation_task_workers=int(parent_interpolation_task_workers),
        )
        view_processing_futures[fut] = key

    def _get_tile_accumulator(model_name: str, view_name: str) -> np.ndarray:
        key = (str(model_name), str(view_name))
        acc = tile_accumulator_by_parent.get(key)
        if acc is not None:
            return acc
        view = view_infos_by_name[str(view_name)]
        acc_path = temp_dir / 'tile_consolidated' / str(model_name) / str(view_name) / 'gated_or.u8.dat'
        acc = allocate_workspace_array(
            shape=(int(view.num_slices), int(view.src_h), int(view.src_w)),
            dtype=np.uint8,
            path=acc_path,
            desc=f'{model_name}/{view_name} consolidated gated-tile accumulator',
            prefer_memory=False,
        )
        tile_accumulator_by_parent[key] = acc
        tile_accumulator_paths[key] = acc_path
        return acc

    def _parent_destination_ready(model_name: str, view_name: str) -> bool:
        view = view_infos_by_name[str(view_name)]
        if view.family == 'radial':
            return str(view_name) in radial_native_output_by_model.get(str(model_name), {})
        if view.family == 'tilted_transverse':
            return str(view_name) in tilted_native_output_by_model.get(str(model_name), {})
        return str(view_name) in view_volumes_by_model.get(str(model_name), {})

    def _parent_destination_volume(model_name: str, view_name: str) -> np.ndarray:
        view = view_infos_by_name[str(view_name)]
        if view.family == 'radial':
            return radial_native_output_by_model[str(model_name)][str(view_name)]
        if view.family == 'tilted_transverse':
            return tilted_native_output_by_model[str(model_name)][str(view_name)]
        return view_volumes_by_model[str(model_name)][str(view_name)]

    def _maybe_submit_tile_consolidation(model_name: str, view_name: str) -> None:
        parent_key = (str(model_name), str(view_name))
        if parent_key in tile_consolidation_submitted:
            return
        expected = int(tile_expected_by_parent.get(parent_key, 0))
        if expected <= 0:
            return
        if len(tile_completed_by_parent.get(parent_key, set())) < expected:
            return
        if not _parent_destination_ready(str(model_name), str(view_name)):
            return

        tile_consolidation_submitted.add(parent_key)
        acc = tile_accumulator_by_parent.get(parent_key)
        if acc is None:
            # All tiles completed but none survived cleanup/gating. Nothing to interpolate or union.
            return

        view = view_infos_by_name[str(view_name)]
        fut = tile_postprocess_executor.submit(
            finalize_consolidated_tile_volume_for_parent,
            model_name=str(model_name),
            view=view,
            tile_accumulator_mm=acc,
            destination_mm=_parent_destination_volume(str(model_name), str(view_name)),
            destination_lock=view_volume_locks[(str(model_name), str(view_name))],
            temp_dir=temp_dir,
            interpolate=int(args.interpolate),
            interpolation_walk_back=int(args.interpolation_walk_back),
            interpolation_candidates=int(args.interpolation_candidates),
            interpolate_passes=int(args.interpolate_passes),
            interpolate_min_radius=float(args.interpolate_min_radius),
            interpolation_search_angle=float(args.interpolation_search_angle),
            keep_temp=bool(args.troubleshooting),
            slice_workers=int(tile_slice_postprocess_workers),
            interpolation_task_workers=int(tile_interpolation_task_workers),
        )
        tile_consolidation_futures[fut] = parent_key

    def _mark_tile_complete(model_name: str, view_name: str, tile_id: str) -> None:
        parent_key = (str(model_name), str(view_name))
        if parent_key not in tile_expected_by_parent:
            return
        completed = tile_completed_by_parent.setdefault(parent_key, set())
        completed.add(str(tile_id))
        _maybe_submit_tile_consolidation(str(model_name), str(view_name))

    def _submit_tile_finalize(result: TilePostprocessResult) -> None:
        parent_key = (str(result.model_name), str(result.view_name))
        support_by_view = native_view_support_by_model.get(result.model_name, {})
        if result.view_name not in support_by_view:
            waiting = postprocessed_tiles_waiting_by_parent.setdefault(parent_key, {})
            waiting[str(result.tile_id)] = spill_waiting_tile_result_to_mmap(
                result,
                temp_dir,
                workers=int(tile_slice_postprocess_workers),
                keep_original=bool(args.troubleshooting),
            )
            return

        waiting = postprocessed_tiles_waiting_by_parent.get(parent_key)
        if waiting is not None:
            waiting.pop(str(result.tile_id), None)
            if not waiting:
                postprocessed_tiles_waiting_by_parent.pop(parent_key, None)

        tile_accumulator_mm = _get_tile_accumulator(result.model_name, result.view_name)

        fut = tile_postprocess_executor.submit(
            gate_tile_volume_into_consolidated_parent,
            result,
            parent_support_mm=support_by_view[result.view_name],
            tile_accumulator_mm=tile_accumulator_mm,
            tile_accumulator_lock=view_volume_locks[(result.model_name, result.view_name)],
            keep_temp=bool(args.troubleshooting),
            slice_workers=int(tile_slice_postprocess_workers),
        )
        tile_finalize_futures[fut] = (str(result.model_name), str(result.view_name), str(result.tile_id))

    def _flush_ready_postprocessed_tiles() -> None:
        ready_results: List[TilePostprocessResult] = []
        for parent_key, waiting in list(postprocessed_tiles_waiting_by_parent.items()):
            model_name, view_name = parent_key
            if view_name not in native_view_support_by_model.get(model_name, {}):
                continue
            ready_results.extend(load_waiting_tile_result_from_mmap(wait_result) for wait_result in waiting.values())
            del postprocessed_tiles_waiting_by_parent[parent_key]

        for result in ready_results:
            _submit_tile_finalize(result)


    def _drain_completed_render_futures() -> None:
        for key, fut in list(canvas_futures_by_aug.items()):
            if key in canvas_future_processed or not fut.done():
                continue
            fut.result()
            canvas_future_processed.add(key)
            pending = pending_tile_batches_by_aug.pop(key, None)
            if pending is not None:
                pending_view, pending_aug_job, pending_tile_jobs = pending
                _submit_tile_video_batch_now(pending_view, pending_aug_job, pending_tile_jobs)

        for fut in list(pending_fullframe_futures):
            if not fut.done():
                continue
            pending_fullframe_futures.remove(fut)
            view, job = fullframe_video_futures.pop(fut)
            fut.result()
            _push_ready_fullframe(view, job)
            if dense_tiling_active and view.family == 'radial':
                tile_jobs = tile_jobs_by_aug.get((view.name, job.aug_id), [])
                if tile_jobs:
                    _submit_tile_video_batch(view, job, tile_jobs)

        for fut in list(pending_tile_video_futures):
            if not fut.done():
                continue
            pending_tile_video_futures.remove(fut)
            view, tile_jobs = tile_batch_video_futures.pop(fut)
            fut.result()
            for tile_job in tile_jobs:
                rendered_tile_jobs_by_view[view.name][tile_job.tile_id] = tile_job
                for model_name, _ in yolo_models:
                    _maybe_enqueue_tile_inference(model_name, view.name, tile_job)

    def _drain_completed_background_futures() -> None:
        for fut in list(view_processing_futures.keys()):
            if not fut.done():
                continue
            result = fut.result()
            del view_processing_futures[fut]
            native_view_support_by_model[result.model_name][result.view_name] = result.native_support_mm
            interpolation_stats.extend(result.interpolation_stats)
            _register_pass_snapshots(result.pass_snapshots)

            view_info = view_infos_by_name[result.view_name]
            if result.final_view_volume_mm is not None:
                if view_info.family == 'radial' and dense_tiling_active:
                    radial_native_output_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
                elif view_info.family == 'tilted_transverse' and dense_tiling_active:
                    tilted_native_output_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
                else:
                    view_volumes_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
            _maybe_submit_tile_consolidation(result.model_name, result.view_name)

        for fut in list(tile_cleanup_futures.keys()):
            if not fut.done():
                continue
            ready_key = tile_cleanup_futures.pop(fut)
            result = fut.result()
            if result is None:
                _mark_tile_complete(str(ready_key[0]), str(ready_key[1]), str(ready_key[2]))
                continue
            _submit_tile_finalize(result)

        _flush_ready_postprocessed_tiles()

        for fut in list(tile_finalize_futures.keys()):
            if not fut.done():
                continue
            model_name, view_name, tile_id = tile_finalize_futures.pop(fut)
            fut.result()
            _mark_tile_complete(str(model_name), str(view_name), str(tile_id))

        for fut in list(tile_consolidation_futures.keys()):
            if not fut.done():
                continue
            result = fut.result()
            del tile_consolidation_futures[fut]
            interpolation_stats.extend(result.interpolation_stats)
            _register_pass_snapshots(result.pass_snapshots)

        output_manager.reap_completed()


    try:
        while True:
            _drain_completed_render_futures()
            _drain_completed_background_futures()

            if ready_fullframe:
                view, job = ready_fullframe.popleft()
                print(f"Inferencing full-frame video: {view.name}/{job.aug_id}")
                for model_name, yolo in yolo_models:
                    pred_prefix = temp_dir / 'preds' / model_name / view.name / f'{view.name}_{job.aug_id}'
                    pred_stats = predict_video_and_accumulate(
                        model=yolo,
                        video_path=job.video_path,
                        num_frames=view.num_slices,
                        out_size=args.imgsz,
                        pred_out_prefix=pred_prefix,
                        cfg=pred_cfg,
                        view_union_mm=baseline_union_by_model_view[(model_name, view.name)],
                        view_confmap_mm=baseline_confmap_by_model_view[(model_name, view.name)],
                        M_out_to_native=job.aff.M_out_to_src,
                        native_h=view.src_h,
                        native_w=view.src_w,
                        postprocess_workers=predict_postprocess_workers,
                        tilted_view=None,
                    )
                    if offload_between_jobs_enabled():
                        offload_yolo_from_gpu(yolo)
                    view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))

                    remaining_key = (model_name, view.name)
                    fullframe_remaining[remaining_key] = int(fullframe_remaining.get(remaining_key, 0)) - 1
                    if int(fullframe_remaining.get(remaining_key, 0)) == 0:
                        _submit_view_prepare(model_name, view)
                continue

            if ready_tile_infer:
                model_name, view, tile_job = ready_tile_infer.popleft()
                ready_key = (str(model_name), str(view.name), str(tile_job.tile_id))
                tile_inference_enqueued.discard(ready_key)
                if ready_key in tile_inference_done:
                    continue

                print(f"Inferencing tile video: {model_name}/{view.name}/{tile_job.tile_id}")
                tile_shape = (int(view.num_slices), int(view.src_h), int(view.src_w))
                tile_mask_path = temp_dir / 'tile_volumes' / model_name / view.name / f'{tile_job.tile_id}.u8.dat'
                tile_conf_path = temp_dir / 'tile_volumes' / model_name / view.name / f'{tile_job.tile_id}.confmap.u8.dat'
                tile_mask_path.parent.mkdir(parents=True, exist_ok=True)

                tile_mask_mm = allocate_workspace_array(
                    shape=tile_shape,
                    dtype=np.uint8,
                    path=tile_mask_path,
                    desc=f'{model_name}/{view.name}/{tile_job.tile_id} raw tile volume',
                    prefer_memory=True,
                )
                tile_conf_mm = allocate_workspace_array(
                    shape=tile_shape,
                    dtype=np.uint8,
                    path=tile_conf_path,
                    desc=f'{model_name}/{view.name}/{tile_job.tile_id} raw tile confidence workspace',
                    prefer_memory=True,
                )

                yolo = yolo_by_model_name[str(model_name)]
                pred_prefix = temp_dir / 'tile_preds' / model_name / view.name / f'{tile_job.tile_id}'
                pred_stats = predict_video_and_accumulate(
                    model=yolo,
                    video_path=tile_job.video_path,
                    num_frames=view.num_slices,
                    out_size=int(args.imgsz),
                    pred_out_prefix=pred_prefix,
                    cfg=pred_cfg,
                    view_union_mm=tile_mask_mm,
                    view_confmap_mm=tile_conf_mm,
                    M_out_to_native=tile_job.M_out_to_src,
                    native_h=view.src_h,
                    native_w=view.src_w,
                    postprocess_workers=predict_postprocess_workers,
                    tilted_view=None,
                )
                if offload_between_jobs_enabled():
                    offload_yolo_from_gpu(yolo)
                tile_inference_done.add(ready_key)
                view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))

                if int(pred_stats.get('frames_with_predictions', 0)) <= 0:
                    close_memmap_array(tile_mask_mm)
                    close_memmap_array(tile_conf_mm)
                    if not args.troubleshooting:
                        try:
                            tile_mask_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        try:
                            tile_conf_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    _mark_tile_complete(str(model_name), str(view.name), str(tile_job.tile_id))
                    continue

                task = TilePostprocessTask(
                    model_name=str(model_name),
                    view_name=str(view.name),
                    config_id=str(tile_job.config_id),
                    tile_id=str(tile_job.tile_id),
                    tile_mask_mm=tile_mask_mm,
                    tile_confmap_mm=tile_conf_mm,
                    tile_mask_path=tile_mask_path,
                    tile_confmap_path=tile_conf_path,
                )
                fut = tile_postprocess_executor.submit(
                    postprocess_tile_volume_after_inference,
                    task,
                    view=view,
                    min_conf=float(args.min_conf),
                    min_radius=float(args.min_radius),
                    keep_temp=bool(args.troubleshooting),
                    slice_workers=int(tile_slice_postprocess_workers),
                )
                tile_cleanup_futures[fut] = ready_key
                continue

            waitables: List[Future] = [
                fut
                for key, fut in canvas_futures_by_aug.items()
                if key not in canvas_future_processed
            ]
            waitables.extend(list(pending_fullframe_futures))
            waitables.extend(list(pending_tile_video_futures))
            waitables.extend(list(view_processing_futures.keys()))
            waitables.extend(list(tile_cleanup_futures.keys()))
            waitables.extend(list(tile_finalize_futures.keys()))
            waitables.extend(list(tile_consolidation_futures.keys()))
            if not waitables:
                _flush_ready_postprocessed_tiles()
                if (
                    not tile_finalize_futures and
                    not tile_cleanup_futures and
                    not tile_consolidation_futures and
                    not view_processing_futures
                ):
                    break
                continue
            wait(waitables, return_when=FIRST_COMPLETED)

    finally:
        fullframe_render_executor.shutdown(wait=True)
        if tile_render_executor is not None:
            tile_render_executor.shutdown(wait=True)
        parent_postprocess_executor.shutdown(wait=True)
        tile_postprocess_executor.shutdown(wait=True)

    _drain_completed_render_futures()
    _drain_completed_background_futures()

    for cache_name, cache_mm in list(view_frame_caches.items()):
        close_memmap_array(cache_mm)
        cache_path = view_frame_cache_paths.get(cache_name)
        if not args.troubleshooting and cache_path is not None:
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass
    view_frame_caches.clear()
    view_frame_cache_paths.clear()

    for view in views:
        if view.family != 'radial':
            continue
        for model_name, _ in yolo_models:
            if view.name in view_volumes_by_model[model_name]:
                continue
            radial_native = radial_native_output_by_model[model_name].get(view.name)
            if radial_native is None:
                radial_native = native_view_support_by_model[model_name].get(view.name)
            if radial_native is None:
                continue
            out_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.u8.dat'
            radial_volume = backproject_radial_volume_to_volume(
                radial_mask_mm=radial_native,
                radial_view=view,
                out_path=out_path,
                desc=f'Backprojecting final {model_name}/{view.name}',
                prefer_memory=False,
            )
            if float(args.min_radius) > 0:
                print(f"Applying --min_radius in the transverse plane for backprojected view '{view.name}'")
                apply_transverse_min_radius_filter_inplace(
                    radial_volume,
                    float(args.min_radius),
                    workers=slice_postprocess_workers,
                )
            view_volumes_by_model[model_name][view.name] = radial_volume

    for view in views:
        if view.family != 'tilted_transverse':
            continue
        for model_name, _ in yolo_models:
            if view.name in view_volumes_by_model[model_name]:
                continue
            tilted_native = tilted_native_output_by_model[model_name].get(view.name)
            if tilted_native is None:
                tilted_native = native_view_support_by_model[model_name].get(view.name)
            if tilted_native is None:
                continue
            out_path = temp_dir / 'view_volumes' / model_name / f'{view.name}.u8.dat'
            tilted_volume = backproject_tilted_volume_to_volume(
                tilted_mask_mm=tilted_native,
                tilted_view=view,
                out_path=out_path,
                desc=f'Backprojecting final {model_name}/{view.name}',
                prefer_memory=False,
            )
            view_volumes_by_model[model_name][view.name] = tilted_volume

    output_manager.reap_completed()

    print('\n=== Building final single-model view union after the global view union ===')
    final_union_mm = assemble_final_union_after_view_union(
        view_volumes_by_model=view_volumes_by_model,
        T=T,
        H=H,
        W=W,
        disable_multiplanar=not bool(args.enable_multiplanar),
        out_path=temp_dir / 'final_union_volume.u8.dat',
        temp_dir=temp_dir,
        enable_3d_void_fill=bool(args.enable_3d_void_fill),
        keep_temp=bool(args.troubleshooting),
        prefer_memory=True,
        workers=slice_postprocess_workers,
    )

    object_interpolation_smoothing_stats: Optional[Dict[str, int]] = None
    if bool(args.object_interpolation_smoothing):
        print('\n=== Applying object interpolation smoothing ===')
        object_interpolation_smoothing_stats = apply_object_interpolation_smoothing_inplace(
            final_union_mm,
            view_support_by_model=native_view_support_by_model,
            views=views,
            temp_dir=temp_dir,
            enable_multiplanar=bool(args.enable_multiplanar),
            keep_temp=bool(args.troubleshooting),
            workers=slice_postprocess_workers,
        )

    keep_objects_stats: Optional[Dict[str, int]] = None
    if int(args.keep_objects) > 0:
        print(f'\n=== Keeping largest {int(args.keep_objects)} final object(s) ===')
        keep_objects_stats = apply_keep_largest_objects_inplace(
            final_union_mm,
            int(args.keep_objects),
            temp_dir=temp_dir,
            keep_temp=bool(args.troubleshooting),
            prefer_memory=True,
            workers=slice_postprocess_workers,
        )

    final_paths: Dict[str, Path] = {}

    if bool(args.troubleshooting) and int(args.interpolate) > 0:
        final_paths.update(schedule_troubleshooting_pass_outputs(
            output_manager=output_manager,
            snapshot_refs=pass_snapshot_refs,
            model_names=[name for name, _ in yolo_models],
            views=views,
            volume_rgb=volume_rgb,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_binary_pattern_value=args.save_binary,
            save_labels_pattern_value=args.save_labels,
            save_nrrd_flag=bool(args.save_nrrd),
            save_multiplanar_flag=bool(args.save_multiplanar),
            total_passes=int(args.interpolate_passes),
            T=T,
            H=H,
            W=W,
            enable_multiplanar=bool(args.enable_multiplanar),
            min_radius=float(args.min_radius),
            temp_dir=temp_dir,
            keep_temp=bool(args.troubleshooting),
            frame_workers=output_frame_workers,
            workers=slice_postprocess_workers,
        ))

    print('\n=== Scheduling final outputs in background ===')
    final_output_paths, final_futures = collect_pipeline_output_futures(
        output_manager.executor,
        volume_rgb=volume_rgb,
        mask_u8=final_union_mm,
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
            mask_u8=final_union_mm,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_binary_pattern_value=args.save_binary,
            save_labels_pattern_value=args.save_labels,
            tag=None,
            frame_workers=output_frame_workers,
            show_progress=False,
        )
        final_output_paths.update(extra_paths)
        final_futures.extend(extra_futures)
    final_paths.update(final_output_paths)
    output_manager.submit(BackgroundOutputSubmission(
        label='final outputs',
        result_paths=final_output_paths,
        futures=final_futures,
        resources=[],
    ))

    voxel_volume = None
    if bool(args.voxel_volume):
        voxel_counts = np.zeros((int(final_union_mm.shape[0]),), dtype=np.int64)

        def _count_voxels(z: int) -> None:
            voxel_counts[int(z)] = np.int64(np.count_nonzero(np.asarray(final_union_mm[int(z)])))

        parallel_for_indices(
            int(final_union_mm.shape[0]),
            _count_voxels,
            max_workers=choose_slice_parallel_workers(int(slice_postprocess_workers), int(final_union_mm.shape[0])),
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
                mask_u8=final_union_mm,
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

    if bool(args.save_tilted_transverse):
        tilted_views = [v for v in views if v.family == 'tilted_transverse']
        if not tilted_views:
            print('Warning: --save_tilted_transverse requested but --tilt_angle is 0; skipping tilted transverse outputs')
        else:
            print('\n=== Saving tilted transverse outputs ===')
            for view in tilted_views:
                final_paths.update(write_additional_view_outputs(
                    volume_rgb=volume_rgb,
                    mask_u8=final_union_mm,
                    view=view,
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

    if bool(args.save_low_quality):
        print('\n=== Saving low-quality video copies ===')
        low_quality_dir = save_low_quality_video_copies(
            out_dir,
            workers=output_workers,
            show_progress=False,
        )
        final_paths['low_quality_dir'] = low_quality_dir

    summary_path = write_summary_file(
        out_dir / f'{input_path.stem}_Summary.txt',
        command=shlex.join([str(x) for x in sys.argv]),
        input_path=input_path,
        out_dir=out_dir,
        scratch_dir=temp_dir,
        volume_shape=(T, H, W),
        fps=fps,
        model_paths=model_paths,
        view_names=[
            (
                f'{v.name} ({int(v.num_slices)} frames; centers {int(v.tilt_frame_start)}..{int(v.tilt_frame_stop)})'
                if v.family == 'tilted_transverse'
                else f'{v.name} ({int(v.num_slices)} frames)'
            )
            for v in views
        ],
        view_prediction_stats=view_prediction_stats,
        interpolation_stats=interpolation_stats,
        enable_3d_void_fill=bool(args.enable_3d_void_fill),
        object_interpolation_smoothing_stats=object_interpolation_smoothing_stats,
        keep_objects_stats=keep_objects_stats,
        voxel_volume=voxel_volume,
        final_paths=final_paths,
        augmentation_workers=augmentation_workers,
        slice_postprocess_workers=slice_postprocess_workers,
        interpolation_workers=interpolation_workers,
        output_workers=output_workers,
    )

    close_memmap_array(final_union_mm)
    for model_support in native_view_support_by_model.values():
        for mm in model_support.values():
            close_memmap_array(mm)
        model_support.clear()
    for model_views in radial_native_output_by_model.values():
        for mm in model_views.values():
            close_memmap_array(mm)
        model_views.clear()
    for model_views in tilted_native_output_by_model.values():
        for mm in model_views.values():
            close_memmap_array(mm)
        model_views.clear()
    for model_views in view_volumes_by_model.values():
        for mm in model_views.values():
            close_memmap_array(mm)
        model_views.clear()
    for mm in tile_accumulator_by_parent.values():
        close_memmap_array(mm)
    tile_accumulator_by_parent.clear()
    for mm in baseline_union_by_model_view.values():
        close_memmap_array(mm)
    for mm in baseline_confmap_by_model_view.values():
        close_memmap_array(mm)
    for _, yolo in yolo_models:
        unload_yolo_model(yolo)
    close_memmap_array(volume_rgb)
    trim_cuda_memory()
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
