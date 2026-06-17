#!/usr/bin/env python3
"""
YOLO segmentation test-time augmentation (TTA) for large cylindrical video volumes.

This v12.2.12_SLURM implementation of the v12.2.0_SLURM specification plus v12.2.1-v12.2.12 performance patches:
  - v12.2.12 changes full-frame/tile YOLO inputs to bounded streaming render sources, so inference starts after the first prefetch batch is rendered instead of after a complete (slice,imgsz,imgsz) prediction volume is materialized
  - v12.2.11 removes GPU backprojection from the final Radial/Tilted path, gives CPU-only backprojection the full slice-worker budget, keeps tile waiting/staging/consolidation canvases in RAM by default, overlaps YOLO model loading with decode/cube preparation, adds threaded ffmpeg decode defaults, and tightens the NRRD writer around the Target_Dummy trailing-list layout with direct raw-bbox chunk fills and full-CPU pigz defaults
  - v12.2.8 adds a dedicated single --angle streaming cleanup path: when exactly one augmentation angle is requested, min_conf filtering, 2D hole filling, and view-native per-slice min_radius filtering are applied as YOLO slices stream in; only true volume-level cleanup waits for the completed view volume
  - v12.2.9 keeps the GPU-fed queue hot by sizing prediction-volume builders to the active prefetch window, skipping scratch memmap flushes on the prediction hot path by default, and allowing single-angle CPU result accumulation to finish behind the next prediction volume
  - v12.2.1 restores the low-quality downbin/output helper path and NrrdLayerRef metadata class that were erroneously pruned in v12.2.0
  - v12.2.2 adds batched in-memory inference with synthetic final-batch padding for fixed-batch backends, larger RAM-aware NRRD streaming slabs, parallel low-quality NRRD scheduling, and the now-CPU-only radial/tilted backprojection queue
  - v12.2.4 aligns the script header with the filename, moves Radial before Tilted in scheduling order, plans interpolation from cached per-slice component tables and local SDF crops, consumes variable-cost seed plans through an unordered bounded completion queue, keeps transient NRRD projection/delta workspaces in RAM when possible, and raises the CPU mask postprocess pending cap to 4096
  - v12.2.5 improves CPU occupancy during interpolation labeling by compact-relabeling row blocks with per-slice local-to-compact lookup tables, prebuilding per-slice component tables through an unordered bounded queue, and testing endpoint continuation from component-table crops instead of full adjacent slices
  - v12.2.6 sizes per-parent full-frame interpolation seed planning for the expected live overlap, not the total number of active parent views, so compact relabel and endpoint scans can use most of the SLURM CPU allocation when interpolation is effectively serial or only two parents overlap
  - v12.2.7 moves full-frame and consolidated-tile interpolation passes into a shared ProcessPoolExecutor by reopening disk-backed memmaps in worker processes, so GIL-heavy interpolation cannot starve prediction-volume builders in the main process
  - v12.2.7 adds optional Numba no-GIL projection-candidate kernels for interpolation seed planning; set YOLO_TTA_INTERPOLATION_COMPILED_KERNELS=0 to force the Python fallback
  - builds Transverse, optional generalized Tilted Views for Transverse/Sagittal/Coronal, independently optional upright Sagittal/Coronal, and optional Radial view families using single-channel intermediates
  - renders YOLO-ready full-frame and tiled sources from memory through bounded streaming slice iterators instead of writing prediction videos or requiring a full prediction volume before inference
  - feeds those sources to Ultralytics through custom HxWx1 in-memory loaders with stream=True, while v12.2.9 defaults to deferred CPU retina-mask reconstruction so large GPU mask tensors are not copied on the scheduler/model-stream thread
  - bounds streaming prediction-source prefetch and legacy prediction-volume creation, preserving priority order while preventing unbounded RAM growth
  - stages cleaned tiled masks into one parent-view canvas per tile-size/stride set, gates each consolidated canvas
    at the connected-component level against frozen parent full-frame support, then interpolates the accepted tiled volume once per parent view
  - spills postprocessed waiting tile masks and decomposed NRRD/support binary volumes to raw slice-bbox stores with no bitpacking or LZ4 compression
  - validates/resizes tile mask volumes to their parent view-native shape before raw waiting-tile store spill so ctile stores always match the parent canvas
  - removes dense-tile pruning and keeps temporary/intermediate mask volumes unpacked throughout
  - fuses per-slice cleanup work where the slice orientation matches the required semantics
    (notably min_conf filtering, 2D hole filling, and min_radius where applicable)
  - overlaps GPU inference with CPU-side view interpolation, consolidated-tile interpolation, and output writing
  - defaults to deferred CPU retina-mask reconstruction for live inference; set YOLO_TTA_CPU_RETINA_MASKS=0 to restore Ultralytics native retina_masks=True compatibility mode
  - reuses a native radial frame cache during dense tiled rendering so radial tiles do not recompute
    the same Lanczos-3 slices for every tile location
  - inverse-maps predictions only into each generated prediction volume's native view space, keeps Radial and Tilted View results view-native through cleanup/interpolation, then backprojects them after per-view processing
  - treats --model as a single YOLO segmentation model path; multiple-model inference is not supported in this script
  - applies final 3D void fill only when --enable_3d_void_fill is active, and only once after the global union
  - activates Gaussian smoothing only when --gaussian_smoothing or --gaussian_smoothing_passes is explicitly set; either flag set to 0 disables it
  - supports Radial and generalized Tilted View-native interpolation, and keeps every Tilted View frame N centered on its base view native slice N with black-padded out-of-bounds shear samples
  - resizes the working volume to an approximately cubic orthogonal volume when needed, restores the final mask to the original input dimensions for default outputs,
    and saves the transverse default color overlay plus optional labels, bilevel compressed TIFF binary masks, binary MKVs,
    decomposed multi-layer NRRD, active-view image sequences, optional skeleton NRRDs/layers, and isotropically downsampled
    low-quality presentation outputs, with FFV1 used for spec-required MKVs
  - writes the default NRRD as independently toggleable layers for full-frame YOLO masks, interpolation bridges, tiled masks accepted by parent YOLO masks,
    tiled masks accepted by parent bridges, consolidated tile bridges, the pre-smoothing global union, smoothing-pass results, optional skeleton, and the final output layer
  - streams NRRD gzip-encoded payloads through pigz from per-layer backing arrays or raw cvol stores without materializing a full 4D decomposed payload, reducing peak scratch pressure during final NRRD creation
  - records NRRD SegmentN_Extent metadata while each layer is materialized, so final NRRD packaging reuses stored extents instead of rescanning every backing layer

Dependencies (Python):
  pip install opencv-python numpy scipy scikit-image tifffile tqdm ultralytics
  # Optional CPU acceleration: numba for no-GIL interpolation seed-planning kernels.
  # Optional GPU acceleration: cupy-cuda12x for cupyx.scipy.ndimage smoothing only; final Radial/Tilted backprojection is CPU-only in v12.2.11.
  # --save_nrrd uses the built-in streaming NRRD writer; pynrrd is no longer required for default exports.

System:
  ffmpeg + ffprobe + pigz on PATH.
"""

from __future__ import annotations

import argparse
import colorsys
import gc
import heapq
import json
import math
import multiprocessing as mp
import os
import queue
import re
import shlex
import struct
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field

GIB = 1024 ** 3
NRRD_SPACE = "left-posterior-superior"
NRRD_AXIS_ORDER_NOTE = "internal mask (t,Y,X) is exported as Slicer spatial axes (X,Y,t); decomposed Segment list axis is trailing (X,Y,t,layer)"
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

try:
    import numba as _numba  # type: ignore
except Exception as _numba_exc:  # pragma: no cover - optional acceleration
    _numba = None  # type: ignore[assignment]
    _NUMBA_IMPORT_ERROR: Optional[BaseException] = _numba_exc
else:
    _NUMBA_IMPORT_ERROR = None

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
        if angle_f == 0.0:
            continue
        if angle_f < 0.0:
            raise ValueError('--tilt_angle values must be positive; each non-zero value creates both signed tilt variants')
        if angle_f in seen:
            continue
        seen.add(angle_f)
        out.append(angle_f)
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


def resolve_tilt_views(values: Sequence[str] | str | None) -> List[str]:
    """Resolve v12 Tilted View base-view selections.

    Tilted Views are derived from Cartesian base views only.  The selected base
    view does not need to be enabled as an upright view; e.g. ``--tilt_view
    sagittal`` is valid even without ``--enable_sagittal``.
    """
    raw = [str(v).strip().lower() for v in _parse_token_list(values)]
    out: List[str] = []
    for token in raw:
        if token not in ('transverse', 'sagittal', 'coronal'):
            raise ValueError("--tilt_view values must be transverse, sagittal, or coronal")
        if token not in out:
            out.append(token)
    if not out:
        out = ['transverse']
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

    p.add_argument("--imgsz", default=2048, type=int, help="Square input size used for YOLO predict")
    p.add_argument("--batch", default=1, type=int, help="Batch size passed to YOLO predict. The in-memory source pads the final batch by repeating the last real slice and discards synthetic results, which supports fixed-batch engines such as TensorRT builds without dynamic=True")
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

    p.add_argument("--enable_sagittal", action="store_true",
                   help="Enable Sagittal (X,t) Cartesian views in addition to the required Transverse view")
    p.add_argument("--enable_coronal", action="store_true",
                   help="Enable Coronal (Y,t) Cartesian views in addition to the required Transverse view")
    p.add_argument("--disable_transverse", action="store_true",
                   help="Skip standard Transverse full-frame and tiled inferencing only; Transverse output geometry remains available")
    p.add_argument("--enable_radial", action="store_true", help="Enable Radial views")
    p.add_argument("--azimuth_angle", default=None, type=float,
                   help="Angular spacing in degrees for radial diameter slices over [0,180]. When --enable_radial is active and this is omitted, defaults to the largest angle that guarantees full ROI coverage. 0 disables radial views")
    p.add_argument("--tilt_view", nargs="+", default=["transverse"], type=str,
                   help="One or more Cartesian base views for v12 Tilted Views: transverse, sagittal, or coronal. A tilted base view does not need to be enabled as an upright view")
    p.add_argument("--tilt_angle", nargs="+", default=["0"], type=str,
                   help="One or more positive Tilted View angles in degrees. Each value creates both positive and negative variants. 0 disables all Tilted Views. Values must be greater than 0 and less than or equal to 45")
    p.add_argument("--tilt_direction", nargs="+", default=["vertical"], type=str,
                   help="One or more Tilted View directions: vertical, horizontal, or both")

    p.add_argument("--tile_size", nargs="+", default=["0"], type=str,
                   help="One or more square dense-tile side lengths in source pixels for all active views. 0 disables dense tiled predictions")
    p.add_argument("--tile_stride", nargs="+", default=["0"], type=str,
                   help="One or more dense-tile strides in source pixels. v12 forms a Cartesian product with --tile_size; each stride must be <= each active tile size")

    p.add_argument("--save_images", action="store_true", help="Save unlabeled image sequences for all active views")
    p.add_argument("--save_labels", nargs="?", const="__DEFAULT__", default=None, type=str,
                   help="Save final YOLO segmentation labels per frame. Optional custom pattern, e.g. labels/{Filename}_%%04d.txt")
    p.add_argument("--save_binary", nargs="?", const="__DEFAULT__", default=None, type=str,
                   help="Save final binary masks as a TIFF sequence plus an FFV1 MKV. Optional custom TIFF pattern, e.g. binary_masks/{Filename}_Binary_%%04d.tiff")
    p.add_argument("--save_nrrd", action="store_true", help="Save a decomposed multi-layer NRRD plus manifest JSON")
    p.add_argument("--save_skeleton", action="store_true",
                   help="Compute a 3D skeleton after final postprocessing. With --save_nrrd, add it as a decomposed layer; otherwise write a skeleton-only NRRD")
    p.add_argument("--save_low_quality", action="store_true",
                   help="Save additional isotropically downsampled low-quality presentation videos and NRRDs using libx264, preset slow, yuv420p")
    p.add_argument("--save_low_quality_downbin", nargs="+", default=None, type=str,
                   help="One or more isotropic low-quality downbins. Floats scale each X/Y/t dimension, e.g. 0.5. Integers scale the largest dimension to that value. Providing this flag implies --save_low_quality")
    p.add_argument("--voxel_volume", action="store_true", help="Count white voxels in the final binary output after restoration to native input geometry and save the value to the summary text file")
    p.add_argument("--enable_3d_void_fill", action="store_true",
                   help="Apply one final 3D enclosed-void fill after the global union. Disabled by default")
    p.add_argument("--gaussian_smoothing", nargs="?", const=3.0, default=None, type=float, metavar="SIGMA",
                   help="Final 3D Gaussian smoothing sigma in voxel units. Unset uses default 3.0 when smoothing is activated by either Gaussian flag; explicitly set 0 to disable")
    p.add_argument("--gaussian_smoothing_passes", default=None, type=int,
                   help="Number of Gaussian smoothing passes. Unset uses default 1 when smoothing is activated by either Gaussian flag; explicitly set 0 to disable")

    p.add_argument("--troubleshooting", action="store_true",
                   help="Save FFV1 MKV troubleshooting overlays for each active full-frame view and consolidated tiled prediction set")
    p.add_argument("--interpolate", default=15, type=int,
                   help="Maximum view-native slice/frame distance used to search for interpolation candidates. Radial interpolation wraps around frame order. 0 disables interpolation")
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

    The v12.2.0 SLURM target has enough CPU headroom that running roughly 2x the visible CPU count
    helps keep in-memory view-volume construction, interpolation planning, and output writers busy while the GPU
    is inferencing or waiting on a different stage.
    """
    return max(1, int(_cpu_count()) * 2)


def resolve_parent_interpolation_worker_allocation(
    worker_budget: int,
    parent_postprocess_workers: int,
) -> Tuple[int, int, int]:
    """Resolve per-parent interpolation workers from expected live overlap.

    Parent full-frame interpolation used to divide the global worker budget by the total
    number of parent postprocess workers, which also tracked the number of active views.
    Current scheduling normally completes a view's interpolation before the next parent
    begins and rarely has more than two parent views interpolating at once.  The default
    therefore partitions the global worker budget by expected live overlap instead of by
    total active view count.

    Returns ``(expected_overlap, default_per_parent_workers, resolved_per_parent_workers)``.
    ``YOLO_TTA_INTERPOLATION_CONCURRENT_PARENTS`` changes the overlap estimate and
    ``YOLO_TTA_INTERPOLATION_TASK_WORKERS`` still overrides the final per-parent worker count.
    """
    budget = max(1, int(worker_budget))
    parent_workers = max(1, int(parent_postprocess_workers))
    default_overlap = max(1, min(1, parent_workers))
    overlap = max(
        1,
        min(
            parent_workers,
            _env_int('YOLO_TTA_INTERPOLATION_CONCURRENT_PARENTS', default_overlap),
        ),
    )
    default_workers = max(1, budget // max(1, int(overlap)))
    resolved_workers = max(
        1,
        _env_int('YOLO_TTA_INTERPOLATION_TASK_WORKERS', default_workers),
    )
    return int(overlap), int(default_workers), int(resolved_workers)


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
    initialize_zero: bool = True,
) -> np.ndarray:
    dtype_obj = np.dtype(dtype)
    need_bytes = array_nbytes(shape, dtype_obj)
    budget = workspace_budget_summary(need_bytes, reserve_bytes=reserve_bytes)
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(need_bytes, reserve_bytes=reserve_bytes)

    if use_in_memory:
        try:
            print(f"{desc}: in-memory ({budget})")
            shape_tuple = tuple(int(x) for x in shape)
            return (
                np.zeros(shape_tuple, dtype=dtype_obj)
                if bool(initialize_zero)
                else np.empty(shape_tuple, dtype=dtype_obj)
            )
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
        initialize_zero=False,
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


def parallel_map_unordered(
    func: Callable[[int], object],
    items: Iterable[int],
    *,
    max_workers: int,
    max_pending: Optional[int] = None,
) -> Iterator[object]:
    """Bounded parallel map that yields results as soon as tasks complete.

    This is intended for variable-cost tasks whose output order does not affect the
    final binary result, such as interpolation endpoint seed planning.  Unlike
    ``parallel_map_in_order``, one slow early item cannot block result consumption
    or prevent later items from being submitted once the pending bound is reached.
    """
    workers = max(1, int(max_workers))
    if workers <= 1:
        for item in items:
            yield func(int(item))
        return

    pending_limit = max(workers, int(max_pending) if max_pending is not None else workers + 1)
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: set[Future] = set()

        def _submit_until_full() -> None:
            while len(pending) < pending_limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.add(executor.submit(func, int(item)))

        _submit_until_full()
        while pending:
            done, pending_remainder = wait(pending, return_when=FIRST_COMPLETED)
            pending = set(pending_remainder)
            for fut in done:
                yield fut.result()
            _submit_until_full()


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


def close_memmap_array_without_flush(arr: object) -> None:
    """Close a scratch memmap mapping without forcing dirty pages to storage."""
    if arr is None:
        return
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


def prediction_volume_build_flush_enabled() -> bool:
    """Return True to force flushing YOLO input volumes before inference."""
    return _env_flag('YOLO_TTA_FLUSH_PREDICTION_VOLUME_ON_BUILD', False)


def prediction_hot_path_flush_enabled() -> bool:
    """Return True to force per-source prediction accumulation memmap flushes."""
    return _env_flag('YOLO_TTA_PREDICT_FLUSH_EACH_VOLUME', False)


# --------------------------
# Interpolation process isolation
# --------------------------

_INTERPOLATION_PROCESS_EXECUTOR: Optional[ProcessPoolExecutor] = None
_INTERPOLATION_PROCESS_MAX_WORKERS = 0
_INTERPOLATION_PROCESS_WORKER = False


def _sanitize_filesystem_token(value: object) -> str:
    token = re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(value).strip()).strip('_')
    return token or 'unnamed'


def interpolation_process_backend_enabled() -> bool:
    """Return True when interpolation passes should run in process workers.

    The default is enabled for v12.2.7 because interpolation seed planning contains
    Python-heavy control flow.  Running the pass in a separate process prevents that
    GIL-bound work from starving prediction-volume builder threads in the main
    process.  Set YOLO_TTA_INTERPOLATION_PROCESS_BACKEND=0 to recover the legacy
    in-process thread-pool path.
    """
    return _env_flag('YOLO_TTA_INTERPOLATION_PROCESS_BACKEND', True)


def interpolation_process_fallback_enabled() -> bool:
    """Allow an in-process fallback if a process interpolation task fails.

    The default is fail-fast so accidental reintroduction of the GIL bottleneck is
    visible.  Set YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK=1 when completing a run is
    preferred over failing on a worker-process exception.
    """
    return _env_flag('YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK', False)


def interpolation_process_start_method() -> str:
    method = os.environ.get('YOLO_TTA_INTERPOLATION_PROCESS_START_METHOD', 'spawn').strip().lower()
    if method not in {'spawn', 'forkserver', 'fork'}:
        method = 'spawn'
    return method


def interpolation_process_cv2_threads() -> int:
    return max(1, _env_int('YOLO_TTA_INTERPOLATION_PROCESS_CV2_THREADS', 1))


def _interpolation_process_initializer() -> None:
    global _INTERPOLATION_PROCESS_WORKER
    _INTERPOLATION_PROCESS_WORKER = True
    try:
        cv2.setNumThreads(int(interpolation_process_cv2_threads()))
    except Exception:
        pass


def create_interpolation_process_executor(max_workers: int) -> Optional[ProcessPoolExecutor]:
    if not interpolation_process_backend_enabled():
        return None
    workers = max(1, int(max_workers))
    start_method = interpolation_process_start_method()
    try:
        ctx = mp.get_context(start_method)
    except Exception:
        start_method = 'spawn'
        ctx = mp.get_context(start_method)
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_interpolation_process_initializer,
    )


def set_interpolation_process_executor(executor: Optional[ProcessPoolExecutor], max_workers: int = 0) -> None:
    global _INTERPOLATION_PROCESS_EXECUTOR, _INTERPOLATION_PROCESS_MAX_WORKERS
    _INTERPOLATION_PROCESS_EXECUTOR = executor
    _INTERPOLATION_PROCESS_MAX_WORKERS = max(0, int(max_workers))


def _interpolation_array_backing_path(arr: object) -> Optional[Path]:
    if arr is None:
        return None
    try:
        arr_np = np.asarray(arr)
        if (
            isinstance(arr, np.memmap)
            and bool(arr_np.flags['C_CONTIGUOUS'])
            and int(getattr(arr, 'offset', 0) or 0) == 0
        ):
            filename = getattr(arr, 'filename', None)
            return Path(filename) if filename else None
    except Exception:
        pass
    # Deliberately do not reuse a base memmap for ndarray views here.  A child
    # process would need the view's byte offset and strides to reopen it exactly;
    # interpolation volumes are expected to be full C-contiguous arrays, so views
    # are copied into a dedicated process memmap instead.
    return None


def _ensure_process_backed_interpolation_volume(
    mask_mm: np.ndarray,
    *,
    work_dir: Path,
    pass_tag: str,
    workers: int,
) -> Tuple[np.ndarray, Path, bool]:
    """Return a memmap-backed volume that a process worker can reopen by path."""
    arr = np.asarray(mask_mm)
    if arr.ndim != 3:
        raise ValueError(f'Interpolation expects a 3D mask volume, got shape {arr.shape}')
    if arr.dtype != np.dtype(np.uint8):
        raise ValueError(f'Interpolation process backend expects uint8 mask volume, got dtype {arr.dtype}')

    backing_path = _interpolation_array_backing_path(mask_mm)
    if backing_path is not None:
        flush_array(mask_mm)
        return mask_mm, Path(backing_path), False

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    process_path = work_dir / f'{_sanitize_filesystem_token(pass_tag)}.process_input.u8.dat'
    process_mm = copy_workspace_array(
        arr,
        process_path,
        desc=f'Interpolation process input {pass_tag}',
        prefer_memory=False,
        workers=int(workers),
    )
    flush_array(process_mm)
    return process_mm, process_path, True


def _interpolation_process_entry(
    *,
    mask_path: str,
    mask_shape: Tuple[int, int, int],
    mask_dtype: str,
    work_dir: str,
    pass_tag: str,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_min_radius: float,
    keep_temp: bool,
    reserve_bytes: int,
    workers: int,
    wrap_axis: bool,
) -> Dict[str, object]:
    global _INTERPOLATION_PROCESS_WORKER
    _INTERPOLATION_PROCESS_WORKER = True
    try:
        cv2.setNumThreads(int(interpolation_process_cv2_threads()))
    except Exception:
        pass

    mask_mm = np.memmap(
        Path(mask_path),
        dtype=np.dtype(mask_dtype),
        mode='r+',
        shape=tuple(int(x) for x in mask_shape),
    )
    try:
        stats = interpolate_view_volume_pass_inplace(
            mask_mm=mask_mm,
            work_dir=Path(work_dir),
            pass_tag=str(pass_tag),
            max_slice_distance=int(max_slice_distance),
            search_angle_deg=float(search_angle_deg),
            interpolation_walk_back=int(interpolation_walk_back),
            interpolation_candidates=int(interpolation_candidates),
            interpolate_min_radius=float(interpolate_min_radius),
            keep_temp=bool(keep_temp),
            prefer_memory=True,
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
            wrap_axis=bool(wrap_axis),
        )
        stats = dict(stats)
        stats.update({
            'process_backend': 'process_pool_memmap',
            'process_pid': int(os.getpid()),
            'process_workers_inside_pass': int(workers),
            'process_memmap_path': str(mask_path),
        })
        flush_array(mask_mm)
        return stats
    finally:
        close_memmap_array(mask_mm)


def interpolate_view_volume_pass_maybe_process(
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
    wrap_axis: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Run one interpolation pass, using a child process when configured.

    The returned array may be a new memmap when the input was an anonymous ndarray.
    Callers must keep using the returned volume for later passes and downstream work.
    """
    executor = _INTERPOLATION_PROCESS_EXECUTOR
    if (
        _INTERPOLATION_PROCESS_WORKER
        or executor is None
        or not interpolation_process_backend_enabled()
    ):
        stats = interpolate_view_volume_pass_inplace(
            mask_mm=mask_mm,
            work_dir=work_dir,
            pass_tag=pass_tag,
            max_slice_distance=int(max_slice_distance),
            search_angle_deg=float(search_angle_deg),
            interpolation_walk_back=int(interpolation_walk_back),
            interpolation_candidates=int(interpolation_candidates),
            interpolate_min_radius=float(interpolate_min_radius),
            keep_temp=bool(keep_temp),
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
            wrap_axis=bool(wrap_axis),
        )
        stats = dict(stats)
        stats.setdefault('process_backend', 'disabled_or_unconfigured')
        return mask_mm, stats

    process_mm, process_path, copied_to_memmap = _ensure_process_backed_interpolation_volume(
        mask_mm,
        work_dir=Path(work_dir),
        pass_tag=str(pass_tag),
        workers=max(1, min(int(workers), int(mask_mm.shape[0]) if getattr(mask_mm, 'ndim', 0) else int(workers))),
    )
    shape = tuple(int(x) for x in np.asarray(process_mm).shape)
    dtype_str = str(np.asarray(process_mm).dtype)
    flush_array(process_mm)

    fut = executor.submit(
        _interpolation_process_entry,
        mask_path=str(process_path),
        mask_shape=shape,
        mask_dtype=dtype_str,
        work_dir=str(work_dir),
        pass_tag=str(pass_tag),
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        interpolation_walk_back=int(interpolation_walk_back),
        interpolation_candidates=int(interpolation_candidates),
        interpolate_min_radius=float(interpolate_min_radius),
        keep_temp=bool(keep_temp),
        reserve_bytes=int(reserve_bytes),
        workers=int(workers),
        wrap_axis=bool(wrap_axis),
    )
    try:
        stats = dict(fut.result())
    except Exception as exc:
        if not interpolation_process_fallback_enabled():
            raise RuntimeError(
                f'Interpolation process worker failed for {pass_tag} at {process_path}. '
                'Set YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK=1 to rerun this pass in-process for recovery.'
            ) from exc
        print(
            f'Warning: interpolation process worker failed for {pass_tag} ({exc}); '
            'falling back to legacy in-process interpolation because YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK=1.'
        )
        stats = interpolate_view_volume_pass_inplace(
            mask_mm=process_mm,
            work_dir=work_dir,
            pass_tag=pass_tag,
            max_slice_distance=int(max_slice_distance),
            search_angle_deg=float(search_angle_deg),
            interpolation_walk_back=int(interpolation_walk_back),
            interpolation_candidates=int(interpolation_candidates),
            interpolate_min_radius=float(interpolate_min_radius),
            keep_temp=bool(keep_temp),
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
            wrap_axis=bool(wrap_axis),
        )
        stats = dict(stats)
        stats['process_backend'] = 'fallback_in_process_after_worker_failure'

    stats.setdefault('process_backend', 'process_pool_memmap')
    stats['process_memmap_copied_from_anonymous_array'] = bool(copied_to_memmap)
    stats['process_pool_workers'] = int(_INTERPOLATION_PROCESS_MAX_WORKERS)
    flush_array(process_mm)
    return process_mm, stats


# --------------------------
# ffmpeg helpers
# --------------------------

def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")


def ffmpeg_decode_threads() -> int:
    """Return the decoder thread count for input-volume materialization.

    The earlier decode path left FFmpeg to choose its own threading.  On large
    FFV1/Matroska inputs this can silently become a single-decoder bottleneck
    before the first prediction volume is even scheduled.  Default to the full
    visible CPU allocation; set ``YOLO_TTA_FFMPEG_DECODE_THREADS`` to pin a
    smaller value for codecs or filesystems that prefer less parallel decode.
    """
    return max(1, _env_int('YOLO_TTA_FFMPEG_DECODE_THREADS', max(1, _cpu_count())))


class VolumeReadiness:
    """Slice-level readiness gate for streaming decode/cube preprocessing."""

    def __init__(self, total_slices: int, desc: str = 'volume') -> None:
        self.total_slices = max(0, int(total_slices))
        self.desc = str(desc)
        self._slice_events = [threading.Event() for _ in range(self.total_slices)]
        self._all_event = threading.Event()
        self._lock = threading.Lock()
        self._ready_count = 0
        self._exception: Optional[BaseException] = None
        if self.total_slices <= 0:
            self._all_event.set()

    def mark_slice_ready(self, idx: int) -> None:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= int(self.total_slices):
            return
        ev = self._slice_events[idx_i]
        if ev.is_set():
            return
        with self._lock:
            if not ev.is_set():
                ev.set()
                self._ready_count += 1
                if self._ready_count >= self.total_slices:
                    self._all_event.set()

    def mark_all_ready(self) -> None:
        with self._lock:
            for ev in self._slice_events:
                ev.set()
            self._ready_count = self.total_slices
            self._all_event.set()

    def mark_failed(self, exc: BaseException) -> None:
        with self._lock:
            self._exception = exc
            for ev in self._slice_events:
                ev.set()
            self._all_event.set()

    def _raise_if_failed(self) -> None:
        exc = self._exception
        if exc is not None:
            raise RuntimeError(f'{self.desc} producer failed before required slice data was ready') from exc

    def wait_for_slice(self, idx: int) -> None:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= int(self.total_slices):
            raise IndexError(idx_i)
        self._slice_events[idx_i].wait()
        self._raise_if_failed()

    def wait_all(self) -> None:
        self._all_event.wait()
        self._raise_if_failed()

    @property
    def ready_count(self) -> int:
        with self._lock:
            return int(self._ready_count)


_VOLUME_READINESS_BY_ARRAY_ID: Dict[int, VolumeReadiness] = {}


def streaming_preprocess_enabled() -> bool:
    """Return True when decode/cube preprocessing may run ahead of consumers."""
    return _env_flag('YOLO_TTA_STREAMING_PREPROCESS', True)


def register_volume_readiness(arr: object, readiness: VolumeReadiness) -> None:
    _VOLUME_READINESS_BY_ARRAY_ID[id(arr)] = readiness


def volume_readiness(arr: object) -> Optional[VolumeReadiness]:
    return _VOLUME_READINESS_BY_ARRAY_ID.get(id(arr))


def wait_for_volume_slice_ready(arr: object, idx: int) -> None:
    readiness = volume_readiness(arr)
    if readiness is not None:
        readiness.wait_for_slice(int(idx))


def wait_for_volume_ready(arr: object) -> None:
    readiness = volume_readiness(arr)
    if readiness is not None:
        readiness.wait_all()


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
        initialize_zero=False,
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
        "-threads", str(ffmpeg_decode_threads()),
        "-i", str(input_video),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-vsync", "0",
        "-",
    ]
    print(f'FFmpeg gray8 decode threads: {ffmpeg_decode_threads()}')
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


def decode_video_to_memmap_gray8_streaming(
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
    """Start ffmpeg gray8 decode in the background and return its destination array immediately."""
    _require_bin("ffmpeg")
    shape = (int(num_frames), int(height), int(width))
    reuse_existing = bool(not overwrite and out_dat.exists() and not prefer_memory)
    arr = allocate_workspace_array(
        shape=shape,
        dtype=np.uint8,
        path=out_dat,
        desc='Decoded gray8 input volume [streaming producer]',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        reuse_existing=bool(reuse_existing),
        initialize_zero=False,
    )
    readiness = VolumeReadiness(int(num_frames), desc='streaming ffmpeg gray8 decode')
    register_volume_readiness(arr, readiness)
    if reuse_existing and isinstance(arr, np.memmap):
        readiness.mark_all_ready()
        return arr

    raw_bytes = memoryview(np.ascontiguousarray(arr) if not arr.flags['C_CONTIGUOUS'] else arr).cast('B')
    if raw_bytes.obj is not arr:
        arr = np.asarray(raw_bytes.obj).reshape(shape)
        raw_bytes = memoryview(arr).cast('B')
        register_volume_readiness(arr, readiness)

    frame_bytes = int(width) * int(height)
    chunk_frames = max(1, min(128, max(1, (256 * 1024 * 1024) // max(1, frame_bytes))))

    def _decode_worker() -> None:
        cmd = [
            "ffmpeg", "-v", "error", "-threads", str(ffmpeg_decode_threads()),
            "-i", str(input_video), "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-",
        ]
        print(f'FFmpeg gray8 streaming decode threads: {ffmpeg_decode_threads()}')
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            with tqdm(total=num_frames, desc='Streaming decode input volume (gray8)') as pbar:
                for start in range(0, int(num_frames), int(chunk_frames)):
                    nframes = min(int(chunk_frames), int(num_frames) - int(start))
                    need = int(nframes) * int(frame_bytes)
                    offset = int(start) * int(frame_bytes)
                    view = raw_bytes[offset:offset + need]
                    filled = 0
                    while filled < need:
                        nread = proc.stdout.readinto(view[filled:])
                        if nread is None or nread <= 0:
                            raise RuntimeError(f'Unexpected EOF while decoding frames starting at {start}/{num_frames}')
                        filled += int(nread)
                    for frame_idx in range(int(start), int(start) + int(nframes)):
                        readiness.mark_slice_ready(int(frame_idx))
                    pbar.update(int(nframes))
            flush_array(arr)
            readiness.mark_all_ready()
        except BaseException as exc:
            readiness.mark_failed(exc)
            raise
        finally:
            if proc.stdout:
                proc.stdout.close()
            _out, err = proc.communicate()
            if proc.returncode not in (0, None):
                msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
                readiness.mark_failed(RuntimeError(f"ffmpeg decode failed: {msg}"))

    threading.Thread(target=_decode_worker, name='streaming-gray8-decode', daemon=False).start()
    return arr


def compute_cube_resize_shape(
    t_dim: int,
    h_dim: int,
    w_dim: int,
    tolerance: float = 0.05,
) -> Tuple[int, int, int]:
    """Return the minimal processing shape whose dimensions are within tolerance of the longest axis.

    v12.2.0 requires the orthogonal working volume to be approximately cubic.  The implementation
    preserves any axis that is already within the tolerated band and upsamples shorter axes only
    enough to reach ``(1 - tolerance) * longest``.  This avoids unnecessary XY resampling for the
    common 3072x3072x1930 case while still satisfying the 5% cube constraint.
    """
    t_i = max(1, int(t_dim))
    h_i = max(1, int(h_dim))
    w_i = max(1, int(w_dim))
    longest = max(t_i, h_i, w_i)
    lower_bound = int(math.ceil(float(longest) * (1.0 - float(tolerance))))
    return (
        max(t_i, lower_bound),
        max(h_i, lower_bound),
        max(w_i, lower_bound),
    )




def _linear_source_index(out_idx: int, out_count: int, in_count: int) -> float:
    if int(out_count) <= 1 or int(in_count) <= 1:
        return 0.0
    return float(out_idx) * float(in_count - 1) / float(out_count - 1)


def _resize_gray_slice_nearest_or_linear(
    frame: np.ndarray,
    out_w: int,
    out_h: int,
    interpolation: int,
) -> np.ndarray:
    frame_arr = np.asarray(frame, dtype=np.uint8)
    if int(frame_arr.shape[0]) == int(out_h) and int(frame_arr.shape[1]) == int(out_w):
        return np.ascontiguousarray(frame_arr)
    return cv2.resize(
        np.ascontiguousarray(frame_arr),
        (int(out_w), int(out_h)),
        interpolation=int(interpolation),
    )


def _cube_t_axis_resize_backend() -> str:
    """Backend used when cubic resizing only changes the slice axis.

    ``slab`` is the default because the common 3072x3072x1930 -> approximately
    cubic case does not need XY resampling.  It processes row slabs through
    OpenCV's native vertical resize and fans the slabs out across Python worker
    threads.  ``slice_exact`` preserves the older endpoint-aligned per-output-slice
    interpolation path for regression testing.
    """
    backend = os.environ.get('YOLO_TTA_CUBE_T_RESIZE_BACKEND', 'slab').strip().lower()
    if backend not in {'slab', 'slice_exact'}:
        backend = 'slab'
    return backend


def _cube_t_axis_slab_rows(in_w: int, in_t: int, out_t: int, workers: int) -> int:
    """Choose a bounded row-slab height for T-axis-only cubic resizing."""
    del in_t, out_t
    env_rows = _env_int('YOLO_TTA_CUBE_T_RESIZE_SLAB_ROWS', 0)
    if env_rows > 0:
        return max(1, int(env_rows))

    # Keep OpenCV's temporary 2D image width below conservative historical limits,
    # while also keeping per-task input/output buffers comfortably bounded.
    max_cv_width = max(1, _env_int('YOLO_TTA_CUBE_T_RESIZE_MAX_CV_WIDTH', 32760))
    rows_by_width = max(1, int(max_cv_width) // max(1, int(in_w)))

    target_mib = max(16, _env_int('YOLO_TTA_CUBE_T_RESIZE_TARGET_MIB_PER_TASK', 384))
    bytes_per_row = max(1, int(in_w)) * 2  # one input byte + one output byte, approximate
    rows_by_memory = max(1, int((target_mib * 1024 * 1024) // max(1, bytes_per_row * max(1, workers))))
    return max(1, min(rows_by_width, rows_by_memory, 16))


def resize_volume_t_axis_only_gray8_slab(
    volume_gray: np.ndarray,
    out_t: int,
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Resize only the first/T axis by processing independent row slabs.

    This is optimized for the dominant SLURM input geometry where X/Y are already
    at the target size and only the shorter frame axis is upsampled to satisfy the
    approximate-cube requirement.  Each task views a ``(T, rows*X)`` slab as a 2D
    OpenCV image and resizes only its height to ``out_t``.
    """
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t_i = int(out_t)
    if int(in_t) == out_t_i:
        return volume_gray

    out_mm = allocate_workspace_array(
        shape=(out_t_i, in_h, in_w),
        dtype=np.uint8,
        path=out_path,
        desc='v12.2.0 cubic processing volume (parallel T-axis slab resize)',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=False,
    )

    worker_count = choose_slice_parallel_workers(int(workers), int(in_h))
    rows_per_slab = _cube_t_axis_slab_rows(int(in_w), int(in_t), int(out_t_i), worker_count)
    ranges = [(int(y0), int(min(in_h, y0 + rows_per_slab))) for y0 in range(0, int(in_h), int(rows_per_slab))]
    cv_threads = max(1, _env_int('YOLO_TTA_CUBE_T_RESIZE_CV2_THREADS', 1))

    try:
        previous_cv_threads = cv2.getNumThreads()
    except Exception:
        previous_cv_threads = None
    try:
        cv2.setNumThreads(int(cv_threads))
    except Exception:
        pass

    print(
        'Cubic resize T-axis slab backend: '
        f'in=(t,Y,X)=({in_t},{in_h},{in_w}) -> out_t={out_t_i}, '
        f'slab_rows={rows_per_slab}, slab_tasks={len(ranges)}, workers={worker_count}, '
        f'cv2_threads_per_task={cv_threads}'
    )

    def _resize_slab(range_idx: int) -> None:
        y0, y1 = ranges[int(range_idx)]
        rows = int(y1 - y0)
        slab = np.ascontiguousarray(volume_gray[:, y0:y1, :], dtype=np.uint8)
        slab_2d = slab.reshape((int(in_t), int(rows) * int(in_w)))
        resized_2d = cv2.resize(
            slab_2d,
            (int(rows) * int(in_w), int(out_t_i)),
            interpolation=cv2.INTER_LINEAR,
        )
        out_mm[:, y0:y1, :] = np.ascontiguousarray(resized_2d.reshape((int(out_t_i), int(rows), int(in_w))))

    try:
        parallel_for_indices_chunked(
            len(ranges),
            _resize_slab,
            max_workers=worker_count,
            desc='Resizing orthogonal volume to v12.2.0 cube (T-axis slabs)',
            show_progress=True,
            chunk_size=1,
        )
    finally:
        if previous_cv_threads is not None:
            try:
                cv2.setNumThreads(int(previous_cv_threads))
            except Exception:
                pass

    flush_array(out_mm)
    return out_mm


def resize_volume_to_processing_cube_gray8(
    volume_gray: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Resample a gray8 (t,Y,X) volume to the v12.2.0 approximately-cubic processing shape."""
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return volume_gray

    if in_h == out_h and in_w == out_w and _cube_t_axis_resize_backend() == 'slab':
        return resize_volume_t_axis_only_gray8_slab(
            volume_gray,
            out_t,
            out_path,
            workers=int(workers),
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc='v12.2.0 cubic processing volume',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=False,
    )

    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _render_target_slice(out_z: int) -> None:
        src_z = _linear_source_index(int(out_z), int(out_t), int(in_t))
        z0 = int(math.floor(src_z))
        z1 = min(in_t - 1, z0 + 1)
        alpha = float(src_z - float(z0))

        f0 = _resize_gray_slice_nearest_or_linear(volume_gray[z0], out_w, out_h, cv2.INTER_LINEAR)
        if z1 == z0 or alpha <= 1e-7:
            out_mm[int(out_z), :, :] = f0
            return
        f1 = _resize_gray_slice_nearest_or_linear(volume_gray[z1], out_w, out_h, cv2.INTER_LINEAR)
        blended = cv2.addWeighted(
            np.ascontiguousarray(f0),
            float(1.0 - alpha),
            np.ascontiguousarray(f1),
            float(alpha),
            0.0,
        )
        out_mm[int(out_z), :, :] = blended

    parallel_for_indices_chunked(
        int(out_t),
        _render_target_slice,
        max_workers=worker_count,
        desc='Resizing orthogonal volume to v12.2.0 cube',
        chunk_size=chunk_size,
    )
    flush_array(out_mm)
    return out_mm


def resize_volume_to_processing_cube_gray8_streaming(
    volume_gray: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 32 * GIB,
) -> np.ndarray:
    """Start cubic gray8 preprocessing in the background and return its output array immediately."""
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        input_ready = volume_readiness(volume_gray)
        if input_ready is not None:
            register_volume_readiness(volume_gray, input_ready)
        return volume_gray

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w), dtype=np.uint8, path=out_path,
        desc='v12.2.0 cubic processing volume [streaming producer]',
        prefer_memory=bool(prefer_memory), reserve_bytes=int(reserve_bytes), initialize_zero=False,
    )
    readiness = VolumeReadiness(int(out_t), desc='streaming cubic processing volume')
    register_volume_readiness(out_mm, readiness)
    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _render_target_slice(out_z: int) -> None:
        src_z = _linear_source_index(int(out_z), int(out_t), int(in_t))
        z0 = int(math.floor(src_z))
        z1 = min(in_t - 1, z0 + 1)
        alpha = float(src_z - float(z0))
        wait_for_volume_slice_ready(volume_gray, int(z0))
        f0 = _resize_gray_slice_nearest_or_linear(volume_gray[z0], out_w, out_h, cv2.INTER_LINEAR)
        if z1 == z0 or alpha <= 1e-7:
            out_mm[int(out_z), :, :] = f0
            readiness.mark_slice_ready(int(out_z))
            return
        wait_for_volume_slice_ready(volume_gray, int(z1))
        f1 = _resize_gray_slice_nearest_or_linear(volume_gray[z1], out_w, out_h, cv2.INTER_LINEAR)
        out_mm[int(out_z), :, :] = cv2.addWeighted(np.ascontiguousarray(f0), float(1.0 - alpha), np.ascontiguousarray(f1), float(alpha), 0.0)
        readiness.mark_slice_ready(int(out_z))

    def _resize_worker() -> None:
        try:
            print(
                'Streaming cubic resize producer: '
                f'in=(t,Y,X)=({in_t},{in_h},{in_w}) -> out=(t,Y,X)=({out_t},{out_h},{out_w}), '
                f'workers={int(worker_count)}, chunk_size={int(chunk_size)}'
            )
            parallel_for_indices_chunked(
                int(out_t), _render_target_slice, max_workers=worker_count,
                desc='Streaming resize orthogonal volume to v12.2.0 cube', chunk_size=chunk_size,
            )
            flush_array(out_mm)
            readiness.mark_all_ready()
        except BaseException as exc:
            readiness.mark_failed(exc)
            raise

    threading.Thread(target=_resize_worker, name='streaming-cubic-resize', daemon=False).start()
    return out_mm


def restore_mask_volume_to_original_shape(
    mask_u8: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    """Map a processing-space binary mask back to the input video's original (t,Y,X) shape."""
    in_t, in_h, in_w = (int(mask_u8.shape[0]), int(mask_u8.shape[1]), int(mask_u8.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return mask_u8

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc='Restored final mask in original input geometry',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=False,
    )
    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _resize_mask_to_output_xy(frame: np.ndarray) -> np.ndarray:
        frame_u8 = (np.asarray(frame, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
        if int(frame_u8.shape[0]) == int(out_h) and int(frame_u8.shape[1]) == int(out_w):
            return np.ascontiguousarray(frame_u8)
        interp = cv2.INTER_AREA if (int(frame_u8.shape[0]) >= int(out_h) and int(frame_u8.shape[1]) >= int(out_w)) else cv2.INTER_NEAREST
        scaled = cv2.resize(
            np.ascontiguousarray(frame_u8 * np.uint8(255)),
            (int(out_w), int(out_h)),
            interpolation=int(interp),
        )
        return (scaled > 0).astype(np.uint8, copy=False)

    def _restore_slice(out_z: int) -> None:
        if int(in_t) >= int(out_t):
            src_start = int(math.floor(float(out_z) * float(in_t) / float(out_t)))
            src_stop = int(math.ceil(float(out_z + 1) * float(in_t) / float(out_t)))
            src_start = int(np.clip(src_start, 0, in_t - 1))
            src_stop = int(np.clip(max(src_start + 1, src_stop), 1, in_t))
            restored = np.zeros((int(out_h), int(out_w)), dtype=np.uint8)
            for src_idx in range(src_start, src_stop):
                restored |= _resize_mask_to_output_xy(mask_u8[int(src_idx)])
            out_mm[int(out_z), :, :] = restored
            return

        src_z = _linear_source_index(int(out_z), int(out_t), int(in_t))
        src_idx = int(np.clip(int(round(src_z)), 0, in_t - 1))
        out_mm[int(out_z), :, :] = _resize_mask_to_output_xy(mask_u8[src_idx])

    parallel_for_indices_chunked(
        int(out_t),
        _restore_slice,
        max_workers=worker_count,
        desc='Restoring final mask to original input geometry',
        chunk_size=chunk_size,
    )
    flush_array(out_mm)
    return out_mm


def resolve_radial_azimuth_angle(
    requested_angle: Optional[float],
    *,
    enable_radial: bool,
    diameter: int,
) -> float:
    """Resolve v12.2.0 radial activation and default full-coverage angle."""
    if requested_angle is not None:
        if float(requested_angle) <= 0.0:
            return 0.0
        return float(requested_angle)
    if bool(enable_radial):
        return radial_full_coverage_angle_deg(int(diameter))
    return 0.0


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


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def purge_temporary_mkv(
    video_path: Optional[Path],
    *,
    temp_dir: Path,
    keep_temp: bool,
    reason: str = '',
) -> bool:
    """Delete one temporary MKV once its last pipeline consumer has finished.

    The guard is intentionally narrow: only ``*.mkv`` files below the active scratch/temp
    directory are eligible. Final output MKVs live in the output directory and are never
    removed through this helper.
    """
    if bool(keep_temp) or video_path is None:
        return False

    path = Path(video_path)
    if path.suffix.lower() != '.mkv':
        return False
    if not _path_is_relative_to(path, Path(temp_dir)):
        return False
    if not path.exists():
        return False

    try:
        path.unlink(missing_ok=True)
        if reason:
            print(f'Purged temporary MKV after last use ({reason}): {path.name}')
        return True
    except Exception as exc:
        print(f'Warning: failed to purge temporary MKV {path} ({exc})')
        return False


def purge_remaining_temporary_mkvs(temp_dir: Path, *, keep_temp: bool) -> int:
    """Best-effort final sweep for temporary MKVs that survived targeted lifecycle deletion."""
    if bool(keep_temp):
        return 0
    root = Path(temp_dir)
    if not root.exists():
        return 0

    purged = 0
    for path in list(root.rglob('*.mkv')):
        if purge_temporary_mkv(path, temp_dir=root, keep_temp=False, reason='final scratch sweep'):
            purged += 1
    return int(purged)



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

    Cartesian views (Transverse, Sagittal, Coronal, and Tilted View in-plane rotation) use
    pad_mode='clamp', which rotates directly on the source-sized canvas so non-90° content that
    leaves the source frame is discarded. Radial uses pad_mode='pad' so non-90° radial rotations are
    black-padded before scaling.
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

TILTED_VIEW_FAMILY = 'tilted'


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
    tilt_base_view: str = ''
    horizontal_axis: str = ''
    vertical_axis: str = ''
    stack_axis: str = ''


def is_tilted_view(view: ViewInfo) -> bool:
    """Return True for v12.2.0 generalized Tilted Views."""
    return str(view.family) == TILTED_VIEW_FAMILY


def tilted_base_view_name(view: ViewInfo) -> str:
    if str(view.tilt_base_view):
        return str(view.tilt_base_view)
    return str(view.name)


def cartesian_view_axis_spec(base_view: str, T: int, H: int, W: int) -> Dict[str, object]:
    base = str(base_view).lower()
    if base == 'transverse':
        return {
            'name': 'transverse',
            'display_name': 'Transverse',
            'num_slices': int(T),
            'src_h': int(H),
            'src_w': int(W),
            'summary_family': 'transverse',
            'horizontal_axis': 'x',
            'vertical_axis': 'y',
            'stack_axis': 't',
        }
    if base == 'sagittal':
        return {
            'name': 'sagittal',
            'display_name': 'Sagittal',
            'num_slices': int(H),
            'src_h': int(T),
            'src_w': int(W),
            'summary_family': 'sagittal',
            'horizontal_axis': 'x',
            'vertical_axis': 't',
            'stack_axis': 'y',
        }
    if base == 'coronal':
        return {
            'name': 'coronal',
            'display_name': 'Coronal',
            'num_slices': int(W),
            'src_h': int(T),
            'src_w': int(H),
            'summary_family': 'coronal',
            'horizontal_axis': 'y',
            'vertical_axis': 't',
            'stack_axis': 'x',
        }
    raise ValueError(f'Unsupported Cartesian base view: {base_view}')


def tilted_stack_axis_length(view: ViewInfo) -> int:
    axis = str(view.stack_axis or cartesian_view_axis_spec(tilted_base_view_name(view), view.full_t, view.full_h, view.full_w)['stack_axis'])
    if axis == 't':
        return int(view.full_t)
    if axis == 'y':
        return int(view.full_h)
    if axis == 'x':
        return int(view.full_w)
    raise ValueError(f'Unsupported Tilted View stacking axis: {axis}')


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
    if is_tilted_view(view):
        base = str(tilted_base_view_name(view)).capitalize()
        direction = str(view.tilt_direction or 'vertical').capitalize()
        return f'Tilted{base}_{direction}_{_format_signed_angle_token(float(view.tilt_angle_deg))}'
    return pretty_view_name(view).replace(' ', '_')


def get_view_infos(
    T: int,
    H: int,
    W: int,
    disable_multiplanar: Optional[bool] = None,
    azimuth_angle: float = 0.0,
    include_radial: bool = True,
    tilt_views: Optional[Sequence[str]] = None,
    tilt_angles: Optional[Sequence[float]] = None,
    tilt_directions: Optional[Sequence[str]] = None,
    enable_sagittal: bool = False,
    enable_coronal: bool = False,
) -> List[ViewInfo]:
    # Backward-compatible alias for older command lines; v12.2.0 controls Sagittal and
    # Coronal independently via --enable_sagittal and --enable_coronal.
    if disable_multiplanar is not None:
        enable_sagittal = bool(enable_sagittal or (not bool(disable_multiplanar)))
        enable_coronal = bool(enable_coronal or (not bool(disable_multiplanar)))

    transverse_spec = cartesian_view_axis_spec('transverse', T, H, W)
    views = [
        ViewInfo(
            name=str(transverse_spec['name']),
            num_slices=int(transverse_spec['num_slices']),
            src_h=int(transverse_spec['src_h']),
            src_w=int(transverse_spec['src_w']),
            pad_mode='clamp',
            family='orthogonal',
            summary_family=str(transverse_spec['summary_family']),
            display_name=str(transverse_spec['display_name']),
            full_t=T,
            full_h=H,
            full_w=W,
            tilt_base_view='transverse',
            horizontal_axis=str(transverse_spec['horizontal_axis']),
            vertical_axis=str(transverse_spec['vertical_axis']),
            stack_axis=str(transverse_spec['stack_axis']),
        ),
    ]
    if bool(enable_sagittal):
        spec = cartesian_view_axis_spec('sagittal', T, H, W)
        views.append(ViewInfo(
            name=str(spec['name']),
            num_slices=int(spec['num_slices']),
            src_h=int(spec['src_h']),
            src_w=int(spec['src_w']),
            pad_mode='clamp',
            family='orthogonal',
            summary_family=str(spec['summary_family']),
            display_name=str(spec['display_name']),
            full_t=T,
            full_h=H,
            full_w=W,
            tilt_base_view='sagittal',
            horizontal_axis=str(spec['horizontal_axis']),
            vertical_axis=str(spec['vertical_axis']),
            stack_axis=str(spec['stack_axis']),
        ))
    if bool(enable_coronal):
        spec = cartesian_view_axis_spec('coronal', T, H, W)
        views.append(ViewInfo(
            name=str(spec['name']),
            num_slices=int(spec['num_slices']),
            src_h=int(spec['src_h']),
            src_w=int(spec['src_w']),
            pad_mode='clamp',
            family='orthogonal',
            summary_family=str(spec['summary_family']),
            display_name=str(spec['display_name']),
            full_t=T,
            full_h=H,
            full_w=W,
            tilt_base_view='coronal',
            horizontal_axis=str(spec['horizontal_axis']),
            vertical_axis=str(spec['vertical_axis']),
            stack_axis=str(spec['stack_axis']),
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
                horizontal_axis='r',
                vertical_axis='t',
                stack_axis='azimuth',
            )
        )

    # Scheduling order in v12.2.0 is Cartesian upright views, then Radial, then Tilted.
    # Tilted variants themselves are deterministic: base view Transverse/Sagittal/Coronal,
    # direction horizontal/vertical, signed angle positive/negative, angle value ascending.
    tilt_angles_resolved = sorted({float(a) for a in (tilt_angles or []) if float(a) > 0.0})
    requested_dirs = set(resolve_tilt_directions(tilt_directions if tilt_directions is not None else ['vertical']))
    tilt_dirs_resolved = [d for d in ('horizontal', 'vertical') if d in requested_dirs]
    tilt_views_requested = set(resolve_tilt_views(tilt_views if tilt_views is not None else ['transverse']))
    tilt_views_resolved = [base for base in ('transverse', 'sagittal', 'coronal') if base in tilt_views_requested]
    for base_view in tilt_views_resolved:
        spec = cartesian_view_axis_spec(str(base_view), T, H, W)
        for tilt_direction in tilt_dirs_resolved:
            for tilt_angle in tilt_angles_resolved:
                for sign in (+1.0, -1.0):
                    signed_angle = float(sign * tilt_angle)
                    token = _format_signed_angle_token(signed_angle)
                    tilt_frame_start, tilt_frame_stop = 0, int(spec['num_slices']) - 1
                    base_label = str(spec['display_name'])
                    direction_label = str(tilt_direction).capitalize()
                    views.append(ViewInfo(
                        name=f'tilted_{base_view}_{tilt_direction}_{token}',
                        num_slices=int(spec['num_slices']),
                        src_h=int(spec['src_h']),
                        src_w=int(spec['src_w']),
                        pad_mode='clamp',
                        family=TILTED_VIEW_FAMILY,
                        summary_family=f'tilted_{base_view}_{tilt_direction}_{token}',
                        display_name=f'Tilted {base_label} {direction_label} {_format_signed_angle_label(signed_angle)}',
                        full_t=T,
                        full_h=H,
                        full_w=W,
                        tilt_angle_deg=signed_angle,
                        tilt_direction=str(tilt_direction),
                        tilt_frame_start=int(tilt_frame_start),
                        tilt_frame_stop=int(tilt_frame_stop),
                        tilt_base_view=str(base_view),
                        horizontal_axis=str(spec['horizontal_axis']),
                        vertical_axis=str(spec['vertical_axis']),
                        stack_axis=str(spec['stack_axis']),
                    ))

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

RADIAL_LANCZOS_A = 3


def _lanczos_kernel(x: np.ndarray, a: int = RADIAL_LANCZOS_A) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.sinc(x) * np.sinc(x / float(a))
    out[np.abs(x) >= float(a)] = 0.0
    return out.astype(np.float32, copy=False)


def _lanczos_offsets(a: int = RADIAL_LANCZOS_A) -> np.ndarray:
    """Integer support offsets for a Lanczos-a kernel around floor(sample)."""
    a_i = max(1, int(a))
    return np.arange(-(a_i - 1), a_i + 1, dtype=np.int32)


def _lanczos_tap_count(a: int = RADIAL_LANCZOS_A) -> int:
    return int(_lanczos_offsets(int(a)).size)


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

    offsets = _lanczos_offsets(RADIAL_LANCZOS_A)
    x0 = np.floor(xs).astype(np.int32, copy=False)
    y0 = np.floor(ys).astype(np.int32, copy=False)

    x_idx_raw = x0[:, None] + offsets[None, :]
    y_idx_raw = y0[:, None] + offsets[None, :]

    x_w = _lanczos_kernel(xs[:, None] - x_idx_raw, a=RADIAL_LANCZOS_A)
    y_w = _lanczos_kernel(ys[:, None] - y_idx_raw, a=RADIAL_LANCZOS_A)

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

    bytes_per_frame = max(1, int(diameter) * _lanczos_tap_count(RADIAL_LANCZOS_A) * np.dtype(np.float32).itemsize)
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
                'tilt_base_view': str(view.tilt_base_view),
                'horizontal_axis': str(view.horizontal_axis),
                'vertical_axis': str(view.vertical_axis),
                'stack_axis': str(view.stack_axis),
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


@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    tile_stride: int
    config_id: str


def resolve_tile_configs(tile_sizes_raw: Sequence[str] | str | int | None, tile_strides_raw: Sequence[str] | str | int | None) -> List[TileConfig]:
    """Resolve v12 dense tile configurations as a Cartesian product.

    v12 specifies that ``--tile_size`` and ``--tile_stride`` form a Cartesian
    product, not paired zip entries.  ``--tile_size 1536,2048 --tile_stride
    256,512`` therefore creates four tile configurations.
    """
    tile_sizes = _parse_int_list(tile_sizes_raw)
    tile_strides = _parse_int_list(tile_strides_raw)

    if not tile_sizes:
        tile_sizes = [0]
    if not tile_strides:
        tile_strides = [0]

    if any(int(v) == 0 for v in tile_sizes):
        if len(tile_sizes) == 1 and int(tile_sizes[0]) == 0:
            if any(int(v) != 0 for v in tile_strides):
                raise ValueError('--tile_stride must be 0 when --tile_size disables tiled predictions')
            return []
        raise ValueError('--tile_size 0 cannot be mixed with active tile sizes')

    if any(int(v) <= 0 for v in tile_sizes):
        raise ValueError('--tile_size values must be > 0 when dense tiling is active')
    if any(int(v) <= 0 for v in tile_strides):
        raise ValueError('--tile_stride values must be > 0 when dense tiling is active')

    configs: List[TileConfig] = []
    seen: set[str] = set()
    for tile_size in tile_sizes:
        for tile_stride in tile_strides:
            if int(tile_stride) > int(tile_size):
                raise ValueError('--tile_stride must be less than or equal to every --tile_size value in the Cartesian product')
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
    meta_path: Path
    M_out_to_src: np.ndarray
    M_src_to_out: np.ndarray


@dataclass
class PredictionVolumeRef:
    """One active YOLO input job.

    ``array`` is used by the legacy dense-materialization path.  v12.2.12 normally
    uses ``source``: an Ultralytics-compatible streaming source that renders a
    bounded prefetch window of slices and lets YOLO start on the first ready batch.
    """

    array: Optional[np.ndarray]
    path: Optional[Path]
    name: str
    view_name: str
    job_id: str
    kind: str = 'fullframe'
    source: Optional[object] = None


def close_prediction_volume_ref(ref: Optional[PredictionVolumeRef], *, keep_temp: bool = False) -> None:
    """Release one prediction source and remove any fallback backing file."""
    if ref is None:
        return

    close_fn = getattr(getattr(ref, 'source', None), 'close', None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception as exc:
            print(f'Warning: failed to close streaming prediction source {ref.name} ({exc})')

    arr = getattr(ref, 'array', None)
    path = getattr(ref, 'path', None)
    if arr is not None:
        if bool(keep_temp):
            close_memmap_array(arr)
        else:
            close_memmap_array_without_flush(arr)
    if not bool(keep_temp) and path is not None:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


class InMemoryYoloVolumeSource:
    """Ultralytics-compatible in-memory source that streams fixed-size slice batches.

    Ultralytics' public Python API accepts in-memory numpy inputs, but its built-in
    ``LoadPilAndNumpy`` loader treats a list of arrays as one large batch.  This
    loader is registered as an in-memory loader so the predictor consumes a 3-D
    slice volume incrementally with ``stream=True`` and the requested ``--batch``.
    Slices are yielded as single-channel ``H×W×1`` uint8 arrays; this intentionally
    avoids the older gray-to-BGR expansion path so single-channel YOLO models receive
    a single input channel.  The final batch is padded by repeating the final real
    slice so fixed-batch engines, e.g. TensorRT without dynamic=True, always receive
    the same batch size.  Downstream accumulation discards synthetic padded results.
    """

    def __init__(self, volume_gray: np.ndarray, name: str, batch_size: int = 1, max_frames: Optional[int] = None) -> None:
        if volume_gray is None:
            raise ValueError('InMemoryYoloVolumeSource requires a volume array')
        self.volume_gray = np.asarray(volume_gray)
        if self.volume_gray.ndim == 4:
            if int(self.volume_gray.shape[3]) != 1:
                raise ValueError(
                    f'Prediction volume must be single-channel with shape (N,H,W) or (N,H,W,1); got {self.volume_gray.shape}'
                )
        elif self.volume_gray.ndim != 3:
            raise ValueError(f'Prediction volume must have shape (N,H,W) or (N,H,W,1); got {self.volume_gray.shape}')
        self.name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'in_memory_volume'
        self.nf = int(self.volume_gray.shape[0])
        if max_frames is not None:
            self.nf = max(0, min(self.nf, int(max_frames)))
        self.bs = max(1, int(batch_size))
        self.yield_nf = int(math.ceil(float(self.nf) / float(self.bs)) * self.bs) if self.nf > 0 else 0
        self.synthetic_count = max(0, int(self.yield_nf) - int(self.nf))
        self.mode = 'image'
        self.count = 0
        try:
            from ultralytics.data.loaders import SourceTypes  # type: ignore
            self.source_type = SourceTypes(stream=False, screenshot=False, from_img=True, tensor=False)
        except Exception:
            # Older/forked Ultralytics builds only require these attributes.
            self.source_type = argparse.Namespace(stream=False, screenshot=False, from_img=True, tensor=False)

    def __iter__(self) -> 'InMemoryYoloVolumeSource':
        self.count = 0
        return self

    def __len__(self) -> int:
        return int(math.ceil(float(self.nf) / float(self.bs))) if self.nf > 0 else 0

    @staticmethod
    def _frame_to_single_channel(frame: np.ndarray) -> np.ndarray:
        """Return one prediction frame as contiguous H×W×1 uint8, never H×W×3."""
        frame_u8 = np.asarray(frame, dtype=np.uint8)
        if frame_u8.ndim == 2:
            return np.ascontiguousarray(frame_u8[:, :, None])
        if frame_u8.ndim == 3 and int(frame_u8.shape[2]) == 1:
            return np.ascontiguousarray(frame_u8)
        raise ValueError(f'Unsupported non-single-channel prediction frame shape for YOLO source: {frame_u8.shape}')

    def __next__(self) -> Tuple[List[str], List[np.ndarray], List[str]]:
        if self.count >= self.yield_nf:
            raise StopIteration
        if self.nf <= 0:
            raise StopIteration
        start = int(self.count)
        stop = min(int(self.yield_nf), start + int(self.bs))
        self.count = int(stop)
        paths: List[str] = []
        imgs: List[np.ndarray] = []
        info: List[str] = []
        last_real_idx = max(0, int(self.nf) - 1)
        for idx in range(start, stop):
            real_idx = int(idx) if int(idx) < int(self.nf) else int(last_real_idx)
            synthetic = int(idx) >= int(self.nf)
            suffix = '_synthetic' if synthetic else ''
            paths.append(f'{self.name}_{idx + 1:06d}{suffix}.png')
            imgs.append(self._frame_to_single_channel(self.volume_gray[int(real_idx)]))
            if synthetic:
                info.append(f'in-memory {self.name} synthetic padded slice {idx + 1}/{self.yield_nf} repeats real slice {real_idx + 1}/{self.nf}: ')
            else:
                info.append(f'in-memory {self.name} slice {idx + 1}/{self.nf}: ')
        return paths, imgs, info



class StreamingYoloVolumeSource:
    """Ultralytics-compatible source that renders prediction slices just ahead of YOLO.

    The legacy v12 path first materialized a complete ``(N,imgsz,imgsz)`` uint8
    array for every full-frame/tile job.  This source keeps only a bounded window
    of rendered futures alive: the first batch can enter YOLO as soon as those
    frames are ready, while CPU workers continue rendering later slices behind the
    GPU stream.
    """

    def __init__(
        self,
        renderer: Callable[[int], np.ndarray],
        *,
        num_frames: int,
        name: str,
        batch_size: int = 1,
        max_frames: Optional[int] = None,
        out_size: Optional[int] = None,
        render_workers: int = 1,
        prefetch_frames: Optional[int] = None,
        autostart: bool = True,
    ) -> None:
        self.renderer = renderer
        self.name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'streaming_volume'
        self.nf = max(0, int(num_frames))
        if max_frames is not None:
            self.nf = max(0, min(self.nf, int(max_frames)))
        self.out_size = None if out_size is None else int(out_size)
        self.bs = max(1, int(batch_size))
        self.yield_nf = int(math.ceil(float(self.nf) / float(self.bs)) * self.bs) if self.nf > 0 else 0
        self.synthetic_count = max(0, int(self.yield_nf) - int(self.nf))
        self.render_workers = max(1, min(int(render_workers), max(1, int(self.nf))))
        default_prefetch = streaming_prediction_source_prefetch_frames(self.bs)
        self.prefetch_frames = max(self.bs, int(prefetch_frames if prefetch_frames is not None else default_prefetch))
        self.mode = 'image'
        self.count = 0
        self._next_submit = 0
        self._futures: Dict[int, Future] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._closed = False
        try:
            from ultralytics.data.loaders import SourceTypes  # type: ignore
            self.source_type = SourceTypes(stream=False, screenshot=False, from_img=True, tensor=False)
        except Exception:
            self.source_type = argparse.Namespace(stream=False, screenshot=False, from_img=True, tensor=False)
        if bool(autostart):
            self.start()

    def __len__(self) -> int:
        return int(math.ceil(float(self.nf) / float(self.bs))) if self.nf > 0 else 0

    def __iter__(self) -> 'StreamingYoloVolumeSource':
        self.start()
        self.count = 0
        self._fill_prefetch_locked(target_index=0)
        return self

    @staticmethod
    def _frame_to_single_channel(frame: np.ndarray) -> np.ndarray:
        return InMemoryYoloVolumeSource._frame_to_single_channel(frame)

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError(f'Streaming YOLO source {self.name} is already closed')
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=int(self.render_workers),
                    thread_name_prefix=f'yolo-render-{self.name[:24]}',
                )
            self._fill_prefetch_locked(target_index=0)

    def close(self) -> None:
        executor: Optional[ThreadPoolExecutor]
        with self._lock:
            self._closed = True
            executor = self._executor
            self._executor = None
            self._futures.clear()
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)

    def _submit_locked(self, idx: int) -> None:
        idx_i = int(idx)
        if idx_i in self._futures or idx_i >= int(self.nf):
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=int(self.render_workers),
                thread_name_prefix=f'yolo-render-{self.name[:24]}',
            )
        self._futures[idx_i] = self._executor.submit(self._render_one, idx_i)
        self._next_submit = max(int(self._next_submit), idx_i + 1)

    def _fill_prefetch_locked(self, target_index: int) -> None:
        if self._closed or int(self.nf) <= 0:
            return
        submit_stop = min(int(self.nf), max(int(self._next_submit), int(target_index) + int(self.prefetch_frames)))
        while int(self._next_submit) < int(submit_stop):
            self._submit_locked(int(self._next_submit))

    def _ensure_submitted(self, stop_exclusive: int) -> None:
        with self._lock:
            self._fill_prefetch_locked(target_index=max(0, int(stop_exclusive)))
            while int(self._next_submit) < min(int(self.nf), int(stop_exclusive)):
                self._submit_locked(int(self._next_submit))

    def _render_one(self, idx: int) -> np.ndarray:
        frame = np.asarray(self.renderer(int(idx)), dtype=np.uint8)
        if frame.ndim == 3 and int(frame.shape[-1]) == 1:
            frame = frame[:, :, 0]
        if frame.ndim != 2:
            raise ValueError(f'{self.name}: renderer returned unsupported frame shape {frame.shape} for slice {idx}')
        if self.out_size is not None and (int(frame.shape[0]) != int(self.out_size) or int(frame.shape[1]) != int(self.out_size)):
            raise ValueError(f'{self.name}: renderer returned {frame.shape}, expected ({int(self.out_size)}, {int(self.out_size)})')
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _get_real_frame(self, idx: int) -> np.ndarray:
        idx_i = int(np.clip(int(idx), 0, max(0, int(self.nf) - 1)))
        self._ensure_submitted(idx_i + 1)
        with self._lock:
            fut = self._futures.get(idx_i)
        if fut is None:
            # Should only happen if close() raced with iteration; render synchronously so the
            # YOLO stream can still finish deterministically.
            return self._render_one(idx_i)
        frame = fut.result()
        with self._lock:
            self._futures.pop(idx_i, None)
        return frame

    def __next__(self) -> Tuple[List[str], List[np.ndarray], List[str]]:
        if self.count >= self.yield_nf or self.nf <= 0:
            self.close()
            raise StopIteration
        self.start()
        start = int(self.count)
        stop = min(int(self.yield_nf), start + int(self.bs))
        # Ensure the actual batch is ready or rendering before waiting for ordered frames.
        self._ensure_submitted(min(int(stop), int(self.nf)))
        paths: List[str] = []
        imgs: List[np.ndarray] = []
        info: List[str] = []
        last_real_idx = max(0, int(self.nf) - 1)
        for idx in range(start, stop):
            real_idx = int(idx) if int(idx) < int(self.nf) else int(last_real_idx)
            synthetic = int(idx) >= int(self.nf)
            suffix = '_synthetic' if synthetic else ''
            frame = self._get_real_frame(real_idx)
            paths.append(f'{self.name}_{idx + 1:06d}{suffix}.png')
            imgs.append(self._frame_to_single_channel(frame))
            if synthetic:
                info.append(f'streaming {self.name} synthetic padded slice {idx + 1}/{self.yield_nf} repeats real slice {real_idx + 1}/{self.nf}: ')
            else:
                info.append(f'streaming {self.name} slice {idx + 1}/{self.nf}: ')
        self.count = int(stop)
        with self._lock:
            self._fill_prefetch_locked(target_index=int(self.count))
        return paths, imgs, info


def streaming_prediction_sources_enabled() -> bool:
    """Return True when YOLO input slices should be rendered lazily instead of prebuilt."""
    if os.environ.get('YOLO_TTA_STREAMING_PREDICTION_SOURCES') is not None:
        return _env_flag('YOLO_TTA_STREAMING_PREDICTION_SOURCES', True)
    if os.environ.get('YOLO_TTA_STREAM_RENDERED_PREDICTION_SOURCES') is not None:
        return _env_flag('YOLO_TTA_STREAM_RENDERED_PREDICTION_SOURCES', True)
    return _env_flag('YOLO_TTA_STREAM_RENDERED_PREDICTION', True)


def streaming_prediction_source_autostart_enabled() -> bool:
    if os.environ.get('YOLO_TTA_STREAMING_SOURCE_AUTOSTART') is not None:
        return _env_flag('YOLO_TTA_STREAMING_SOURCE_AUTOSTART', True)
    return _env_flag('YOLO_TTA_STREAM_RENDER_AUTOSTART', True)


def streaming_prediction_source_prefetch_frames(batch_size: int) -> int:
    explicit = _env_int('YOLO_TTA_STREAMING_SOURCE_PREFETCH_FRAMES', 0)
    if explicit <= 0 and os.environ.get('YOLO_TTA_STREAM_RENDER_PREFETCH_FRAMES') is not None:
        explicit = _env_int('YOLO_TTA_STREAM_RENDER_PREFETCH_FRAMES', 0)
    if explicit > 0:
        return max(1, int(explicit))
    batches = max(1, _env_int('YOLO_TTA_STREAMING_SOURCE_PREFETCH_BATCHES', _env_int('YOLO_TTA_STREAM_RENDER_PREFETCH_BATCHES', 8)))
    max_frames = max(1, _env_int('YOLO_TTA_STREAMING_SOURCE_PREFETCH_MAX_FRAMES', _env_int('YOLO_TTA_STREAM_RENDER_PREFETCH_MAX_FRAMES', 384)))
    return max(1, min(int(max_frames), max(int(batch_size), int(batch_size) * int(batches))))


def streaming_prediction_source_workers(default_workers: int, num_frames: int) -> int:
    requested = _env_int('YOLO_TTA_STREAMING_SOURCE_WORKERS', _env_int('YOLO_TTA_STREAM_RENDER_WORKERS', int(default_workers)))
    return choose_slice_parallel_workers(max(1, int(requested)), max(1, int(num_frames)))


def make_streaming_yolo_source(
    renderer: Callable[[int], np.ndarray],
    name: str,
    *,
    num_frames: int,
    batch_size: int = 1,
    max_frames: Optional[int] = None,
    out_size: Optional[int] = None,
    render_workers: int = 1,
    prefetch_frames: Optional[int] = None,
    autostart: Optional[bool] = None,
) -> StreamingYoloVolumeSource:
    ensure_ultralytics_accepts_in_memory_volume_source()
    source = StreamingYoloVolumeSource(
        renderer,
        num_frames=int(num_frames),
        name=str(name),
        batch_size=max(1, int(batch_size)),
        max_frames=max_frames,
        out_size=out_size,
        render_workers=max(1, int(render_workers)),
        prefetch_frames=prefetch_frames,
        autostart=streaming_prediction_source_autostart_enabled() if autostart is None else bool(autostart),
    )
    return source


def ensure_ultralytics_accepts_in_memory_volume_source() -> None:
    """Register the v12 in-memory volume loader with Ultralytics' source checker."""
    try:
        import ultralytics.data.build as ultralytics_build  # type: ignore
    except Exception as exc:  # pragma: no cover - ultralytics is imported lazily on SLURM
        raise RuntimeError(f'Unable to import ultralytics.data.build for in-memory prediction source registration: {exc}') from exc

    loaders = getattr(ultralytics_build, 'LOADERS', ())
    try:
        loaders_tuple = tuple(loaders)
    except Exception:
        loaders_tuple = ()
    additions: List[object] = []
    for loader_cls in (InMemoryYoloVolumeSource, StreamingYoloVolumeSource):
        if loader_cls not in loaders_tuple:
            additions.append(loader_cls)
    if additions:
        setattr(ultralytics_build, 'LOADERS', loaders_tuple + tuple(additions))


def make_in_memory_yolo_source(
    volume_gray: np.ndarray,
    name: str,
    *,
    batch_size: int = 1,
    max_frames: Optional[int] = None,
) -> InMemoryYoloVolumeSource:
    ensure_ultralytics_accepts_in_memory_volume_source()
    return InMemoryYoloVolumeSource(volume_gray, name=name, batch_size=max(1, int(batch_size)), max_frames=max_frames)


def make_prediction_ref_yolo_source(
    prediction_volume: PredictionVolumeRef,
    *,
    batch_size: int = 1,
    max_frames: Optional[int] = None,
) -> object:
    ensure_ultralytics_accepts_in_memory_volume_source()
    source = getattr(prediction_volume, 'source', None)
    if source is not None:
        return source
    if prediction_volume.array is None:
        raise ValueError(f'Prediction input {prediction_volume.name} has neither source nor array')
    return make_in_memory_yolo_source(
        prediction_volume.array,
        prediction_volume.name,
        batch_size=max(1, int(batch_size)),
        max_frames=max_frames,
    )


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










def _tilted_plan_cache_key(view: ViewInfo, M_grid_to_src: np.ndarray, grid_h: int, grid_w: int) -> Tuple[str, int, int, Tuple[float, ...]]:
    mat = tuple(round(float(x), 6) for x in np.asarray(M_grid_to_src, dtype=np.float32).reshape(-1).tolist())
    return (str(view.name), int(grid_h), int(grid_w), mat)


def get_tilted_render_plan(
    view: ViewInfo,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> TiltedRenderPlan:
    if not is_tilted_view(view):
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
        (x_nn >= 0) & (x_nn < int(view.src_w)) &
        (y_nn >= 0) & (y_nn < int(view.src_h))
    )
    x_idx = np.clip(x_nn, 0, int(view.src_w) - 1).astype(np.int32, copy=False)
    y_idx = np.clip(y_nn, 0, int(view.src_h) - 1).astype(np.int32, copy=False)

    if str(view.tilt_direction) == 'vertical':
        axis_offset = src_y - float((int(view.src_h) - 1) / 2.0)
    elif str(view.tilt_direction) == 'horizontal':
        axis_offset = src_x - float((int(view.src_w) - 1) / 2.0)
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


def _render_tilted_array_on_grid(
    volume_arr: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
    *,
    mask_mode: bool,
    block_rows: int = 256,
) -> np.ndarray:
    """Render one generalized v12 Tilted View frame.

    ``M_grid_to_src`` maps output/grid pixels into the selected Cartesian base view's
    native raster coordinates.  The two in-plane axes are rounded to their native
    grid, while the base view's stacking coordinate is linearly interpolated after
    applying the signed shear.  This covers Tilted Transverse, Tilted Sagittal, and
    Tilted Coronal without expanding the output canvas.
    """
    if not is_tilted_view(view):
        raise ValueError('Tilted rendering requested for a non-tilted view')

    # A tilted output frame can sample a sheared range of the base stack. Wait for
    # the streaming preprocessing producer to finish before using these views.
    wait_for_volume_ready(volume_arr)

    plan = get_tilted_render_plan(view, M_grid_to_src, int(grid_h), int(grid_w))
    tan_alpha = float(math.tan(math.radians(float(view.tilt_angle_deg))))
    stack_len = int(tilted_stack_axis_length(view))
    base_view = tilted_base_view_name(view)
    out = np.zeros((int(grid_h), int(grid_w)), dtype=np.uint8)
    if stack_len <= 0:
        return out

    for y0 in range(0, int(grid_h), int(block_rows)):
        y1 = min(int(grid_h), y0 + int(block_rows))
        valid_xy = np.asarray(plan.valid_xy[y0:y1], dtype=bool)
        if not np.any(valid_xy):
            continue

        frame_center = float(tilted_frame_center(view, int(frame_idx)))
        stack_src = frame_center + tan_alpha * np.asarray(plan.axis_offset[y0:y1], dtype=np.float32)
        valid = valid_xy & (stack_src >= 0.0) & (stack_src <= float(stack_len - 1))
        if not np.any(valid):
            continue

        s0 = np.floor(stack_src).astype(np.int32, copy=False)
        s1 = np.clip(s0 + 1, 0, stack_len - 1).astype(np.int32, copy=False)
        s0 = np.clip(s0, 0, stack_len - 1).astype(np.int32, copy=False)
        alpha = (stack_src - s0).astype(np.float32, copy=False)

        u_idx = np.asarray(plan.x_idx[y0:y1], dtype=np.int32)  # base-view horizontal axis
        v_idx = np.asarray(plan.y_idx[y0:y1], dtype=np.int32)  # base-view vertical axis
        if base_view == 'transverse':
            f0 = np.asarray(volume_arr[s0, v_idx, u_idx], dtype=np.float32)
            f1 = np.asarray(volume_arr[s1, v_idx, u_idx], dtype=np.float32)
        elif base_view == 'sagittal':
            # Sagittal in-plane axes are (X, t); stacking axis is Y.
            f0 = np.asarray(volume_arr[v_idx, s0, u_idx], dtype=np.float32)
            f1 = np.asarray(volume_arr[v_idx, s1, u_idx], dtype=np.float32)
        elif base_view == 'coronal':
            # Coronal in-plane axes are (Y, t); stacking axis is X.
            f0 = np.asarray(volume_arr[v_idx, u_idx, s0], dtype=np.float32)
            f1 = np.asarray(volume_arr[v_idx, u_idx, s1], dtype=np.float32)
        else:  # pragma: no cover
            raise ValueError(f'Unsupported Tilted View base: {base_view}')

        values = ((1.0 - alpha) * f0) + (alpha * f1)
        if bool(mask_mode):
            rendered = (values >= 0.5).astype(np.uint8, copy=False)
        else:
            rendered = np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)
        out_block = out[y0:y1]
        out_block[valid] = rendered[valid]

    return out


def render_tilted_frame_on_grid(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
    block_rows: int = 256,
) -> np.ndarray:
    return _render_tilted_array_on_grid(
        volume_rgb,
        view,
        int(frame_idx),
        M_grid_to_src,
        int(grid_h),
        int(grid_w),
        mask_mode=False,
        block_rows=int(block_rows),
    )


def render_tilted_mask_on_grid(
    volume_mask: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
    block_rows: int = 256,
) -> np.ndarray:
    return _render_tilted_array_on_grid(
        volume_mask,
        view,
        int(frame_idx),
        M_grid_to_src,
        int(grid_h),
        int(grid_w),
        mask_mode=True,
        block_rows=int(block_rows),
    )


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
    if not is_tilted_view(view):
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







def render_fullframe_frame_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    frame_idx: int,
    view_frames: Optional[np.ndarray] = None,
) -> np.ndarray:
    if is_tilted_view(view):
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



def render_dense_tile_frame_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    tile_job: DenseTileJob,
    frame_idx: int,
    view_frames: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render one dense-tile inference range directly from the native view volume.

    This replaces the legacy canvas-video -> crop -> scale FFmpeg path with the
    same transform collapsed into one in-memory reslice.  ``tile_job.M_src_to_out``
    maps native view coordinates directly to the tile's ``--imgsz`` inference
    raster; for Tilted Views the inverse grid-to-native transform is passed into
    the tilted sampler so the stacking-axis shear, in-plane augmentation, crop,
    and scale are sampled in one pass.
    """
    if is_tilted_view(view):
        return render_tilted_frame_on_grid(
            volume_rgb=volume_rgb,
            view=view,
            frame_idx=int(frame_idx),
            M_grid_to_src=tile_job.M_out_to_src,
            grid_h=int(tile_job.out_size),
            grid_w=int(tile_job.out_size),
        )

    native_frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(frame_idx), view_frames=view_frames))
    return cv2.warpAffine(
        native_frame,
        tile_job.M_src_to_out,
        dsize=(int(tile_job.out_size), int(tile_job.out_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _materialize_prediction_volume_from_renderer(
    *,
    num_slices: int,
    out_size: int,
    out_path: Path,
    desc: str,
    renderer: Callable[[int], np.ndarray],
    workers: int = 1,
    show_progress: bool = True,
    view_name: str = '',
    job_id: str = '',
    kind: str = 'fullframe',
) -> PredictionVolumeRef:
    """Return a YOLO-ready prediction input for one rendered view job.

    v12.2.12 defaults to a lazy render-backed source instead of building a
    complete ``(slice,--imgsz,--imgsz)`` volume before inference. YOLO can
    consume the first ready batch while CPU workers continue resizing/warping
    later slices. Set ``YOLO_TTA_STREAMING_PREDICTION_SOURCES=0`` to recover
    the legacy whole-volume materialization path for regression testing.
    """
    stream_name = str(desc).replace('Materializing in-memory ', 'Streaming render-backed ')
    if streaming_prediction_sources_enabled():
        stream_workers = streaming_prediction_source_workers(max(1, int(workers)), max(1, int(num_slices)))
        stream_prefetch = streaming_prediction_source_prefetch_frames(inference_batch_size())
        if bool(show_progress):
            print(
                f'{stream_name}: streaming source active '
                f'(frames={int(num_slices)}, out_size={int(out_size)}, '
                f'workers={stream_workers}, prefetch_frames={stream_prefetch}, '
                f'autostart={bool(streaming_prediction_source_autostart_enabled())})'
            )
        source = StreamingYoloVolumeSource(
            renderer,
            num_frames=int(num_slices),
            name=stream_name,
            batch_size=inference_batch_size(),
            out_size=int(out_size),
            render_workers=stream_workers,
            prefetch_frames=stream_prefetch,
            autostart=streaming_prediction_source_autostart_enabled(),
        )
        return PredictionVolumeRef(
            array=None,
            path=None,
            name=stream_name,
            view_name=str(view_name),
            job_id=str(job_id),
            kind=str(kind),
            source=source,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_volume = allocate_workspace_array(
        shape=(int(num_slices), int(out_size), int(out_size)),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=True,
        reserve_bytes=32 * GIB,
        initialize_zero=False,
    )
    worker_count = choose_slice_parallel_workers(int(workers), int(num_slices))
    chunk_size = choose_parallel_chunk_size(int(num_slices), worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _render(idx: int) -> None:
        pred_volume[int(idx), :, :] = np.ascontiguousarray(renderer(int(idx)), dtype=np.uint8)

    parallel_for_indices_chunked(
        int(num_slices),
        _render,
        max_workers=worker_count,
        desc=desc,
        show_progress=bool(show_progress),
        chunk_size=chunk_size,
    )
    if prediction_volume_build_flush_enabled():
        flush_array(pred_volume)
    return PredictionVolumeRef(
        array=pred_volume,
        path=out_path if isinstance(pred_volume, np.memmap) else None,
        name=str(desc),
        view_name=str(view_name),
        job_id=str(job_id),
        kind=str(kind),
    )


def materialize_fullframe_prediction_volume_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    *,
    out_path: Path,
    view_frames: Optional[np.ndarray] = None,
    workers: int = 1,
    show_progress: bool = True,
) -> PredictionVolumeRef:
    """Create the model-sized full-frame prediction volume for one view/angle in RAM."""
    if not job.meta_path.exists():
        write_aug_job_meta(job, view)

    def _render(idx: int) -> np.ndarray:
        return render_fullframe_frame_for_job(
            volume_rgb=volume_rgb,
            view=view,
            job=job,
            frame_idx=int(idx),
            view_frames=view_frames,
        )

    return _materialize_prediction_volume_from_renderer(
        num_slices=int(view.num_slices),
        out_size=int(job.aff.out_size),
        out_path=out_path,
        desc=f'Materializing in-memory full-frame prediction volume {view.name}/{job.aug_id}',
        renderer=_render,
        workers=max(1, int(workers)),
        show_progress=bool(show_progress),
        view_name=str(view.name),
        job_id=str(job.aug_id),
        kind='fullframe',
    )


def materialize_dense_tile_prediction_volume_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    tile_job: DenseTileJob,
    *,
    out_path: Path,
    view_frames: Optional[np.ndarray] = None,
    workers: int = 1,
    show_progress: bool = True,
) -> PredictionVolumeRef:
    """Create one dense-tile prediction range as an in-memory YOLO volume."""
    if not tile_job.meta_path.exists():
        write_dense_tile_job_meta(tile_job)

    def _render(idx: int) -> np.ndarray:
        return render_dense_tile_frame_for_job(
            volume_rgb=volume_rgb,
            view=view,
            tile_job=tile_job,
            frame_idx=int(idx),
            view_frames=view_frames,
        )

    return _materialize_prediction_volume_from_renderer(
        num_slices=int(view.num_slices),
        out_size=int(tile_job.out_size),
        out_path=out_path,
        desc=f'Materializing in-memory tile prediction volume {view.name}/{tile_job.tile_id}',
        renderer=_render,
        workers=max(1, int(workers)),
        show_progress=bool(show_progress),
        view_name=str(view.name),
        job_id=str(tile_job.tile_id),
        kind='tile',
    )









def should_cache_view_frames(view: ViewInfo, dense_tiling_active: bool) -> bool:
    """Return True when precomputing native single-channel frames is worthwhile for this view.

    Radial frame caches can be useful for dense-tile throughput, but building the entire
    radial cache before inference creates a large time-to-first-prediction barrier.  v12.2.12
    therefore keeps this prebuild opt-in; streaming sources render only the slices they need.
    """
    return bool(_env_flag('YOLO_TTA_PREBUILD_VIEW_FRAME_CACHES', False)) and bool(dense_tiling_active) and view.family == 'radial'




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
        initialize_zero=False,
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
    """Yield single-channel frames for a view, in slice order (0..num_slices-1).

    Streaming preprocessing marks Transverse processing slices ready one by one,
    so Transverse rendering waits only for the required slice. Sagittal,
    Coronal, Radial, and Tilted views can sample across the stack and therefore
    intentionally wait for the preprocessing producer to finish.
    """
    if view_frames is not None:
        for idx in range(int(view.num_slices)):
            yield np.asarray(view_frames[int(idx)])
        return

    T, H, W = volume_rgb.shape

    if view.name == 'transverse':
        for t in range(T):
            wait_for_volume_slice_ready(volume_rgb, int(t))
            yield np.asarray(volume_rgb[t])  # (H,W)
    elif view.name == 'sagittal':
        wait_for_volume_ready(volume_rgb)
        for y in range(H):
            yield np.ascontiguousarray(volume_rgb[:, y, :])  # (T,W)
    elif view.name == 'coronal':
        wait_for_volume_ready(volume_rgb)
        for x in range(W):
            yield np.ascontiguousarray(volume_rgb[:, :, x])  # (T,H)
    elif view.name == 'radial':
        wait_for_volume_ready(volume_rgb)
        for angle_deg in view.azimuths_deg:
            sampler = get_radial_sampler(view, float(angle_deg))
            yield np.ascontiguousarray(extract_radial_slice_frame(volume_rgb, sampler))  # (T,D)
    elif is_tilted_view(view):
        wait_for_volume_ready(volume_rgb)
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
        wait_for_volume_slice_ready(volume_rgb, int(index))
        return np.asarray(volume_rgb[int(index)])
    if view.name == 'sagittal':
        wait_for_volume_ready(volume_rgb)
        return np.ascontiguousarray(volume_rgb[:, int(index), :])
    if view.name == 'coronal':
        wait_for_volume_ready(volume_rgb)
        return np.ascontiguousarray(volume_rgb[:, :, int(index)])
    if view.name == 'radial':
        wait_for_volume_ready(volume_rgb)
        angle_deg = float(view.azimuths_deg[int(index)])
        sampler = get_radial_sampler(view, angle_deg)
        return np.ascontiguousarray(extract_radial_slice_frame(volume_rgb, sampler))
    if is_tilted_view(view):
        wait_for_volume_ready(volume_rgb)
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


def background_model_load_enabled() -> bool:
    """Return True when YOLO model loading should overlap CPU volume setup.

    Model construction is not expected to be the multi-minute startup bottleneck, so
    the overlap thread is opt-in.  The important fast-start path is streaming CPU
    resize/render work into the GPU consumer once the processing volume is available.
    """
    return _env_flag('YOLO_TTA_BACKGROUND_MODEL_LOAD', False)


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
    batch: int = 1


def async_predict_postprocess_enabled() -> bool:
    """Return True when single-angle prediction CPU tails may run behind the GPU."""
    return _env_flag('YOLO_TTA_ASYNC_PREDICT_POSTPROCESS', True)


def async_predict_join_workers(default_value: int) -> int:
    return max(1, _env_int('YOLO_TTA_ASYNC_PREDICT_JOIN_WORKERS', max(1, int(default_value))))


def async_predict_pending_frame_limit(num_frames: int) -> int:
    """Optional cap for queued async result-worker futures per source.

    The default 0 means source-sized buffering: the GPU-facing iterator will not
    intentionally wait for CPU result workers while a prediction volume streams.
    Set YOLO_TTA_ASYNC_PREDICT_PENDING_FRAMES to a positive value to reduce
    transient CPU/GPU result memory at the cost of some continuity.
    """
    requested = _env_int('YOLO_TTA_ASYNC_PREDICT_PENDING_FRAMES', 0)
    if int(requested) <= 0:
        return 0
    return max(1, min(max(1, int(num_frames)), int(requested)))


@dataclass
class PredictionAccumulationHandle:
    """Background CPU accumulation tail for one streamed prediction source."""
    source_label: str
    futures: List[Future]
    view_union_mm: np.ndarray
    view_confmap_mm: Optional[np.ndarray]
    submitted_frames: int = 0
    synthetic_discarded: int = 0
    precompleted_prediction_count: int = 0
    precompleted_frames_with_predictions: int = 0
    pending_limit: int = 0

    def wait(self) -> Dict[str, int]:
        prediction_count = int(self.precompleted_prediction_count)
        frames_with_predictions = int(self.precompleted_frames_with_predictions)
        try:
            for fut in as_completed(list(self.futures)):
                pred_inc, frame_inc = fut.result()
                prediction_count += int(pred_inc)
                frames_with_predictions += int(frame_inc)
        finally:
            if prediction_hot_path_flush_enabled():
                if self.view_confmap_mm is not None:
                    flush_array(self.view_confmap_mm)
                flush_array(self.view_union_mm)
        return {
            'prediction_count': int(prediction_count),
            'frames_with_predictions': int(frames_with_predictions),
            'submitted_frames': int(self.submitted_frames),
            'synthetic_discarded': int(self.synthetic_discarded),
            'async_accumulation': 1,
        }


DEFAULT_GAUSSIAN_SMOOTHING_SIGMA = 3.0
DEFAULT_GAUSSIAN_SMOOTHING_PASSES = 1


def resolve_gaussian_smoothing_settings(
    gaussian_smoothing_arg: Optional[float],
    gaussian_smoothing_passes_arg: Optional[int],
) -> Tuple[bool, float, int]:
    """Resolve Gaussian smoothing enablement from the v12.2.0 CLI contract.

    Gaussian smoothing is active iff at least one Gaussian flag is explicitly set
    and neither explicitly supplied value is 0. Unset values use the defaults
    from the specification: sigma=3.0 and passes=1.
    """
    sigma_explicit = gaussian_smoothing_arg is not None
    passes_explicit = gaussian_smoothing_passes_arg is not None
    sigma_f = (
        float(gaussian_smoothing_arg)
        if sigma_explicit
        else float(DEFAULT_GAUSSIAN_SMOOTHING_SIGMA)
    )
    passes_i = (
        int(gaussian_smoothing_passes_arg)
        if passes_explicit
        else int(DEFAULT_GAUSSIAN_SMOOTHING_PASSES)
    )
    enabled = bool((sigma_explicit or passes_explicit) and sigma_f > 0.0 and passes_i > 0)
    if not enabled:
        return False, 0.0, max(0, int(passes_i))
    return True, float(sigma_f), int(passes_i)


CONF_U8_MAX = 255


def quantize_conf_to_u8(conf: float) -> np.uint8:
    conf_clamped = min(1.0, max(0.0, float(conf)))
    return np.uint8(int(round(conf_clamped * float(CONF_U8_MAX))))


def min_conf_to_u8_threshold(min_conf: float) -> int:
    conf_clamped = min(1.0, max(0.0, float(min_conf)))
    return int(math.ceil(conf_clamped * float(CONF_U8_MAX) - 1e-9))


@dataclass(frozen=True)
class CpuRetinaMaskPayload:
    """CPU-owned YOLO segmentation tensors needed to reconstruct retina/native masks off-GPU."""

    proto: np.ndarray             # (C, mask_h, mask_w), float32 CPU
    coeffs: np.ndarray            # (N, C), float32 CPU
    boxes_xyxy: np.ndarray        # (N, 4), scaled to prediction-video pixel coordinates
    confs: np.ndarray             # (N,), float32 CPU
    orig_shape: Tuple[int, int]   # (height, width) of the prediction-video frame
    img_shape: Tuple[int, int]    # (height, width) of the network input tensor
    frame_path: str = ''



@dataclass(frozen=True)
class DeferredCpuRetinaMaskPayload:
    """Compact GPU tensors captured from Ultralytics and copied in result-worker threads."""

    pred: object
    proto: object
    orig_shape: Tuple[int, int]
    img_shape: Tuple[int, int]
    frame_path: str = ''


_ULTRALYTICS_CPU_RETINA_PATCHED = False
_ULTRALYTICS_ORIGINAL_SEG_CONSTRUCT_RESULT: Optional[object] = None


def cpu_retina_masks_enabled() -> bool:
    """Use deferred CPU retina reconstruction as the default live-inference result path.

    Native ``retina_masks=True`` can hand large GPU mask tensors to Python result
    handling. v12.2.9 instead captures compact mask coefficients/protos and
    reconstructs bbox-local masks in prediction-result workers. Set
    ``YOLO_TTA_CPU_RETINA_MASKS=0`` to restore Ultralytics native-retina
    compatibility mode.
    """
    return _env_flag('YOLO_TTA_CPU_RETINA_MASKS', True)


def cpu_retina_roi_only_enabled() -> bool:
    """Use bbox-ROI-only CPU upsampling instead of reconstructing every full-size instance mask."""
    return _env_flag('YOLO_TTA_CPU_RETINA_ROI_ONLY', True)


def cpu_retina_block_detections() -> int:
    """Number of mask logits to reconstruct per CPU matrix-multiply block."""
    return max(1, _env_int('YOLO_TTA_CPU_RETINA_BLOCK_DETECTIONS', 8))




def cpu_retina_deferred_payload_enabled() -> bool:
    """Defer compact pred/proto CPU copies from Ultralytics construct_result to result workers."""
    if os.environ.get('YOLO_TTA_CPU_RETINA_DEFER_GPU_COPY') is not None:
        return _env_flag('YOLO_TTA_CPU_RETINA_DEFER_GPU_COPY', True)
    return _env_flag('YOLO_TTA_CPU_RETINA_DEFERRED_PAYLOAD', True)


def predict_async_gpu_copy_enabled() -> bool:
    """Use worker-owned CUDA streams and pinned CPU buffers for tensor copies when possible."""
    return _env_flag('YOLO_TTA_PREDICT_ASYNC_GPU_COPY', True)


def cpu_mask_postprocess_pending_limit(worker_count: int, num_frames: int) -> int:
    """Bound queued CPU mask-reconstruction frames while allowing RAM-backed buffering.

    Setting YOLO_TTA_CPU_MASK_PENDING_FRAMES=0 removes the frame-count cap for sites that
    intentionally want to absorb the entire CPU backlog in RAM. The default is deliberately much
    larger than the worker count so GPU inference is not throttled by CPU retina reconstruction under
    normal SLURM allocations.
    """
    workers = max(1, int(worker_count))
    frames = max(1, int(num_frames))
    requested = _env_int('YOLO_TTA_CPU_MASK_PENDING_FRAMES', 4096)
    if int(requested) <= 0:
        return max(workers, frames)
    return max(workers, min(frames, max(int(requested), workers * 2)))




_ULTRALYTICS_SINGLE_CHANNEL_PREPROCESS_PATCHED = False
_ULTRALYTICS_ORIGINAL_BASE_PREPROCESS: Optional[object] = None


def ensure_single_channel_yolo_preprocess_patch() -> bool:
    """Patch Ultralytics preprocessing so H×W×1 in-memory slices stay one-channel tensors.

    The v12 in-memory source deliberately yields single-channel image arrays. Recent
    Ultralytics preprocessors assume OpenCV-style H×W×3 BGR arrays for numpy-image
    batches and transpose them as BHWC. This patch handles batches that are already
    single-channel by constructing a BCHW tensor directly, bypassing any gray-to-BGR
    duplication while leaving normal RGB/BGR and video sources on the original path.
    """
    global _ULTRALYTICS_SINGLE_CHANNEL_PREPROCESS_PATCHED, _ULTRALYTICS_ORIGINAL_BASE_PREPROCESS

    if _ULTRALYTICS_SINGLE_CHANNEL_PREPROCESS_PATCHED:
        return True

    try:
        import torch  # type: ignore
        from ultralytics.engine.predictor import BasePredictor  # type: ignore
    except Exception as exc:  # pragma: no cover - ultralytics is imported lazily on SLURM
        print(f'Warning: single-channel YOLO preprocess patch could not be installed ({exc})')
        return False

    original_preprocess = BasePredictor.preprocess
    _ULTRALYTICS_ORIGINAL_BASE_PREPROCESS = original_preprocess

    def _tta_single_channel_preprocess(self, im):  # type: ignore[no-untyped-def]
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            if isinstance(im, np.ndarray):
                seq = [im]
            elif isinstance(im, (list, tuple)):
                seq = list(im)
            else:
                seq = []

            if seq:
                gray_frames: List[np.ndarray] = []
                single_channel_batch = True
                for item in seq:
                    arr = np.asarray(item)
                    if arr.ndim == 2:
                        gray = arr
                    elif arr.ndim == 3 and int(arr.shape[2]) == 1:
                        gray = arr[:, :, 0]
                    else:
                        single_channel_batch = False
                        break
                    gray_frames.append(np.ascontiguousarray(gray, dtype=np.uint8))

                if single_channel_batch:
                    try:
                        shapes = {tuple(int(v) for v in frame.shape) for frame in gray_frames}
                        if len(shapes) != 1:
                            raise ValueError(f'single-channel batch contains mixed shapes: {sorted(shapes)}')
                        batch = np.stack(gray_frames, axis=0)[:, None, :, :]  # BCHW, C=1
                        tensor = torch.from_numpy(np.ascontiguousarray(batch))
                        device = getattr(self, 'device', None)
                        if device is not None:
                            tensor = tensor.to(device)
                        model_obj = getattr(self, 'model', None)
                        use_half = bool(getattr(model_obj, 'fp16', False))
                        tensor = tensor.half() if use_half else tensor.float()
                        tensor /= 255.0
                        return tensor
                    except Exception as exc:
                        raise RuntimeError(f'Failed to preprocess single-channel in-memory YOLO batch: {exc}') from exc

        return original_preprocess(self, im)

    BasePredictor.preprocess = _tta_single_channel_preprocess
    _ULTRALYTICS_SINGLE_CHANNEL_PREPROCESS_PATCHED = True
    print('Single-channel YOLO preprocess enabled: in-memory H×W×1 slices are passed as BCHW tensors with C=1.')
    return True


def _as_numpy_float32_cpu(x: object) -> np.ndarray:
    """Detach torch/array-like tensors into owned, contiguous CPU float32 NumPy arrays."""
    try:
        detach = getattr(x, 'detach', None)
        if callable(detach):
            x = detach()
        to_fn = getattr(x, 'to', None)
        if callable(to_fn):
            try:
                import torch  # type: ignore
                x = to_fn(device='cpu', dtype=torch.float32)
            except Exception:
                cpu_fn = getattr(x, 'cpu', None)
                if callable(cpu_fn):
                    x = cpu_fn()
        cpu_fn = getattr(x, 'cpu', None)
        if callable(cpu_fn):
            x = cpu_fn()
        numpy_fn = getattr(x, 'numpy', None)
        if callable(numpy_fn):
            arr = numpy_fn()
        else:
            arr = np.asarray(x)
    except Exception:
        arr = np.asarray(x)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _clip_boxes_np(boxes: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    h = int(shape[0])
    w = int(shape[1])
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(w))
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(h))
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(w))
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(h))
    return boxes


def _scale_boxes_np(
    img1_shape: Sequence[int],
    boxes: np.ndarray,
    img0_shape: Sequence[int],
    *,
    padding: bool = True,
) -> np.ndarray:
    """NumPy equivalent of Ultralytics ops.scale_boxes for xyxy boxes."""
    if boxes.size <= 0:
        return boxes.astype(np.float32, copy=False)

    img1_h, img1_w = int(img1_shape[0]), int(img1_shape[1])
    img0_h, img0_w = int(img0_shape[0]), int(img0_shape[1])
    gain = min(float(img1_h) / max(1.0, float(img0_h)), float(img1_w) / max(1.0, float(img0_w)))
    if gain <= 0.0:
        return _clip_boxes_np(boxes.astype(np.float32, copy=False), img0_shape)

    pad_x = round((float(img1_w) - round(float(img0_w) * gain)) / 2.0 - 0.1)
    pad_y = round((float(img1_h) - round(float(img0_h) * gain)) / 2.0 - 0.1)
    out = boxes.astype(np.float32, copy=True)
    if bool(padding):
        out[:, [0, 2]] -= float(pad_x)
        out[:, [1, 3]] -= float(pad_y)
    out[:, :4] /= float(gain)
    return _clip_boxes_np(out, img0_shape)


def _scale_masks_crop_slices_np(mask_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> Tuple[slice, slice]:
    """Return the low-resolution crop used by Ultralytics scale_masks before interpolation."""
    im1_h, im1_w = int(mask_shape[0]), int(mask_shape[1])
    im0_h, im0_w = int(target_shape[0]), int(target_shape[1])
    if im1_h == im0_h and im1_w == im0_w:
        return slice(0, im1_h), slice(0, im1_w)

    gain = min(float(im1_h) / max(1.0, float(im0_h)), float(im1_w) / max(1.0, float(im0_w)))
    pad_w = (float(im1_w) - round(float(im0_w) * gain)) / 2.0
    pad_h = (float(im1_h) - round(float(im0_h) * gain)) / 2.0
    top = int(round(pad_h - 0.1))
    left = int(round(pad_w - 0.1))
    bottom = int(im1_h - round(pad_h + 0.1))
    right = int(im1_w - round(pad_w + 0.1))
    top = int(np.clip(top, 0, im1_h))
    left = int(np.clip(left, 0, im1_w))
    bottom = int(np.clip(bottom, top + 1, im1_h)) if im1_h > 0 else 0
    right = int(np.clip(right, left + 1, im1_w)) if im1_w > 0 else 0
    return slice(top, bottom), slice(left, right)


def _bbox_to_integer_roi(box_xyxy: np.ndarray, target_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    h, w = int(target_shape[0]), int(target_shape[1])
    if h <= 0 or w <= 0:
        return None
    x1 = int(math.ceil(max(0.0, float(box_xyxy[0]))))
    y1 = int(math.ceil(max(0.0, float(box_xyxy[1]))))
    x2 = int(math.ceil(min(float(w), float(box_xyxy[2]))))
    y2 = int(math.ceil(min(float(h), float(box_xyxy[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return y1, y2, x1, x2


def _resize_lowres_logits_roi(
    low_logits: np.ndarray,
    target_shape: Tuple[int, int],
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    """Upsample only the requested high-resolution ROI using align-corners=False geometry.

    This avoids allocating an N x H x W tensor for hundreds of detections. It is equivalent in
    coordinate mapping to resizing the low-resolution retina logits to the prediction-video frame and
    then cropping the bbox ROI, while doing the expensive interpolation only inside the bbox.
    """
    y1, y2, x1, x2 = (int(v) for v in roi)
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    low = np.asarray(low_logits, dtype=np.float32)
    low_h, low_w = int(low.shape[0]), int(low.shape[1])
    roi_h = int(y2 - y1)
    roi_w = int(x2 - x1)

    if roi_h <= 0 or roi_w <= 0 or low_h <= 0 or low_w <= 0:
        return np.zeros((max(0, roi_h), max(0, roi_w)), dtype=np.float32)

    if low_h == target_h and low_w == target_w:
        return np.ascontiguousarray(low[y1:y2, x1:x2], dtype=np.float32)

    if not cpu_retina_roi_only_enabled():
        full = cv2.resize(low, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(full[y1:y2, x1:x2], dtype=np.float32)

    # Destination pixel-center to source-coordinate mapping for bilinear resize with
    # align_corners=False. Coordinates outside the low-resolution raster are edge-clamped,
    # matching torch.nn.functional.interpolate and cv2.resize behavior.
    xs = ((np.arange(x1, x2, dtype=np.float32) + 0.5) * (float(low_w) / float(target_w))) - 0.5
    ys = ((np.arange(y1, y2, dtype=np.float32) + 0.5) * (float(low_h) / float(target_h))) - 0.5
    xs = np.clip(xs, 0.0, float(low_w - 1))
    ys = np.clip(ys, 0.0, float(low_h - 1))

    x0 = np.floor(xs).astype(np.int32, copy=False)
    y0 = np.floor(ys).astype(np.int32, copy=False)
    x1_idx = np.minimum(x0 + 1, low_w - 1).astype(np.int32, copy=False)
    y1_idx = np.minimum(y0 + 1, low_h - 1).astype(np.int32, copy=False)
    wx = (xs - x0.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    wy = (ys - y0.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    top = (low[y0[:, None], x0[None, :]] * (1.0 - wx[None, :])) + (low[y0[:, None], x1_idx[None, :]] * wx[None, :])
    bottom = (low[y1_idx[:, None], x0[None, :]] * (1.0 - wx[None, :])) + (low[y1_idx[:, None], x1_idx[None, :]] * wx[None, :])
    return np.ascontiguousarray((top * (1.0 - wy[:, None])) + (bottom * wy[:, None]), dtype=np.float32)


def _iter_cpu_retina_payload_rois(
    payload: CpuRetinaMaskPayload,
    target_shape: Tuple[int, int],
) -> Iterator[Tuple[int, float, int, int, int, int, np.ndarray]]:
    """Yield per-instance high-resolution ROI masks reconstructed from CPU protos/coefficients."""
    proto = np.asarray(payload.proto, dtype=np.float32)
    if proto.ndim == 4 and int(proto.shape[0]) == 1:
        proto = proto[0]
    if proto.ndim != 3:
        return

    coeffs = np.asarray(payload.coeffs, dtype=np.float32)
    boxes = np.asarray(payload.boxes_xyxy, dtype=np.float32)
    confs = np.asarray(payload.confs, dtype=np.float32)
    if coeffs.ndim != 2 or coeffs.shape[0] <= 0 or coeffs.shape[1] != int(proto.shape[0]):
        return

    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        return

    mask_crop_y, mask_crop_x = _scale_masks_crop_slices_np(
        (int(proto.shape[1]), int(proto.shape[2])),
        (target_h, target_w),
    )
    c, mh, mw = int(proto.shape[0]), int(proto.shape[1]), int(proto.shape[2])
    proto_flat = np.ascontiguousarray(proto.reshape(c, mh * mw), dtype=np.float32)
    block = cpu_retina_block_detections()
    n = int(coeffs.shape[0])

    for start in range(0, n, block):
        stop = min(n, start + block)
        logits_block = np.matmul(
            np.ascontiguousarray(coeffs[start:stop], dtype=np.float32),
            proto_flat,
        ).reshape((stop - start, mh, mw))

        for local_idx in range(stop - start):
            inst_idx = int(start + local_idx)
            if inst_idx >= int(boxes.shape[0]):
                continue
            roi = _bbox_to_integer_roi(boxes[inst_idx], (target_h, target_w))
            if roi is None:
                continue
            y1, y2, x1, x2 = roi
            low_logits = np.ascontiguousarray(logits_block[local_idx, mask_crop_y, mask_crop_x], dtype=np.float32)
            roi_logits = _resize_lowres_logits_roi(low_logits, (target_h, target_w), roi)
            roi_mask = np.asarray(roi_logits > 0.0, dtype=bool)
            if not np.any(roi_mask):
                continue
            conf_val = float(confs[inst_idx]) if inst_idx < int(confs.shape[0]) else 0.0
            yield inst_idx, conf_val, y1, y2, x1, x2, roi_mask


def _accumulate_cpu_retina_payload_to_prediction_frame(
    payload: CpuRetinaMaskPayload,
    out_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build a prediction-space union/confidence frame from CPU-side retina payload tensors."""
    target_shape = (int(out_size), int(out_size))
    frame_union = np.zeros(target_shape, dtype=np.uint8)
    frame_confmap = np.zeros(target_shape, dtype=np.uint8)
    kept_instances = 0

    # The generated inference videos are square --imgsz frames. If a future source ever reports a
    # different original shape, keep the current --imgsz target because the affine matrices in this
    # pipeline are defined in that prediction-video coordinate system.
    for _inst_idx, conf_val, y1, y2, x1, x2, roi_mask in _iter_cpu_retina_payload_rois(payload, target_shape):
        roi_u8 = roi_mask.astype(np.uint8, copy=False)
        frame_union[y1:y2, x1:x2] |= roi_u8
        conf_u8 = quantize_conf_to_u8(float(conf_val))
        conf_patch = frame_confmap[y1:y2, x1:x2]
        conf_patch[roi_mask] = np.maximum(conf_patch[roi_mask], conf_u8)
        kept_instances += 1

    return frame_union, frame_confmap, int(kept_instances)


def _realize_deferred_cpu_retina_payload(payload: DeferredCpuRetinaMaskPayload) -> CpuRetinaMaskPayload:
    """Copy compact segmentation tensors to CPU and build a CPU-retina payload."""
    pred_cpu = _as_numpy_float32_cpu(payload.pred)
    if pred_cpu.ndim != 2 or int(pred_cpu.shape[0]) <= 0:
        return CpuRetinaMaskPayload(
            proto=np.zeros((0, 0, 0), dtype=np.float32),
            coeffs=np.zeros((0, 0), dtype=np.float32),
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            confs=np.zeros((0,), dtype=np.float32),
            orig_shape=(int(payload.orig_shape[0]), int(payload.orig_shape[1])),
            img_shape=(int(payload.img_shape[0]), int(payload.img_shape[1])),
            frame_path=str(payload.frame_path),
        )

    boxes_scaled = _scale_boxes_np(payload.img_shape, pred_cpu[:, :4], payload.orig_shape)
    pred_cpu[:, :4] = boxes_scaled
    proto_cpu = _as_numpy_float32_cpu(payload.proto)
    if proto_cpu.ndim == 4 and int(proto_cpu.shape[0]) == 1:
        proto_cpu = proto_cpu[0]
    if proto_cpu.ndim != 3:
        proto_cpu = np.zeros((0, 0, 0), dtype=np.float32)
    coeffs = pred_cpu[:, 6:] if int(pred_cpu.shape[1]) > 6 else np.zeros((int(pred_cpu.shape[0]), 0), dtype=np.float32)
    confs = pred_cpu[:, 4] if int(pred_cpu.shape[1]) > 4 else np.zeros((int(pred_cpu.shape[0]),), dtype=np.float32)
    return CpuRetinaMaskPayload(
        proto=np.ascontiguousarray(proto_cpu, dtype=np.float32),
        coeffs=np.ascontiguousarray(coeffs, dtype=np.float32),
        boxes_xyxy=np.ascontiguousarray(pred_cpu[:, :4], dtype=np.float32),
        confs=np.ascontiguousarray(confs, dtype=np.float32),
        orig_shape=(int(payload.orig_shape[0]), int(payload.orig_shape[1])),
        img_shape=(int(payload.img_shape[0]), int(payload.img_shape[1])),
        frame_path=str(payload.frame_path),
    )


def ensure_cpu_retina_mask_predictor_patch() -> bool:
    """Patch Ultralytics segmentation postprocess to return CPU-retina payloads, not GPU masks."""
    global _ULTRALYTICS_CPU_RETINA_PATCHED, _ULTRALYTICS_ORIGINAL_SEG_CONSTRUCT_RESULT

    if not cpu_retina_masks_enabled():
        return False
    if _ULTRALYTICS_CPU_RETINA_PATCHED:
        return True

    try:
        from ultralytics.engine.results import Results  # type: ignore
        from ultralytics.models.yolo.segment.predict import SegmentationPredictor  # type: ignore
    except Exception as exc:
        print(f'Warning: CPU retina-mask predictor patch could not be installed; falling back to Ultralytics masks ({exc})')
        return False

    original_construct_result = SegmentationPredictor.construct_result
    _ULTRALYTICS_ORIGINAL_SEG_CONSTRUCT_RESULT = original_construct_result

    def _tta_cpu_retina_construct_result(self, pred, img, orig_img, img_path, proto):  # type: ignore[no-untyped-def]
        if not cpu_retina_masks_enabled():
            return original_construct_result(self, pred, img, orig_img, img_path, proto)

        try:
            img_shape = tuple(int(x) for x in img.shape[2:])
        except Exception:
            img_shape = (int(getattr(orig_img, 'shape', (0, 0))[0]), int(getattr(orig_img, 'shape', (0, 0))[1]))
        try:
            orig_shape = (int(orig_img.shape[0]), int(orig_img.shape[1]))
        except Exception:
            orig_shape = (int(img_shape[0]), int(img_shape[1]))

        if bool(cpu_retina_deferred_payload_enabled()):
            # Keep the Results object lightweight; compact pred/proto tensors are realized in
            # the prediction-result worker, not inside Ultralytics construct_result.
            result = Results(orig_img, path=img_path, names=self.model.names, boxes=None, masks=None)
            setattr(result, '_tta_deferred_cpu_retina_payload', DeferredCpuRetinaMaskPayload(
                pred=pred,
                proto=proto,
                orig_shape=(int(orig_shape[0]), int(orig_shape[1])),
                img_shape=(int(img_shape[0]), int(img_shape[1])),
                frame_path=str(img_path),
            ))
            return result

        payload = _realize_deferred_cpu_retina_payload(DeferredCpuRetinaMaskPayload(
            pred=pred,
            proto=proto,
            orig_shape=(int(orig_shape[0]), int(orig_shape[1])),
            img_shape=(int(img_shape[0]), int(img_shape[1])),
            frame_path=str(img_path),
        ))
        boxes_for_result = np.zeros((0, 6), dtype=np.float32)
        if payload.confs.size > 0 and payload.boxes_xyxy.shape[0] == payload.confs.shape[0]:
            boxes_for_result = np.concatenate(
                [payload.boxes_xyxy, payload.confs[:, None], np.zeros((payload.confs.shape[0], 1), dtype=np.float32)],
                axis=1,
            ).astype(np.float32, copy=False)
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=boxes_for_result, masks=None)
        setattr(result, '_tta_cpu_retina_payload', payload)
        return result

    SegmentationPredictor.construct_result = _tta_cpu_retina_construct_result
    _ULTRALYTICS_CPU_RETINA_PATCHED = True
    print(
        'Deferred CPU retina masks enabled: Ultralytics GPU mask upsampling is bypassed; '
        'compact mask protos/coefficients are copied and reconstructed in prediction-result workers.'
    )
    return True


def _extract_result_masks_and_confs(r) -> Tuple[Optional[object], Optional[np.ndarray]]:
    """Detach one streamed YOLO result into CPU-owned data for asynchronous postprocess."""
    deferred_payload = getattr(r, '_tta_deferred_cpu_retina_payload', None)
    if isinstance(deferred_payload, DeferredCpuRetinaMaskPayload):
        payload = _realize_deferred_cpu_retina_payload(deferred_payload)
        return payload, np.ascontiguousarray(payload.confs, dtype=np.float32)

    cpu_payload = getattr(r, '_tta_cpu_retina_payload', None)
    if isinstance(cpu_payload, CpuRetinaMaskPayload):
        return cpu_payload, np.ascontiguousarray(cpu_payload.confs, dtype=np.float32)

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


# DEAD_CODE_MARKER(v12.2.0-post-refactor): helper became unreachable after pruning deprecated retina validation; retained for one release for regression notebook compatibility.
def _result_to_prediction_union_u8(result: object, out_size: int) -> np.ndarray:
    masks_np, confs_np = _extract_result_masks_and_confs(result)
    if isinstance(masks_np, CpuRetinaMaskPayload):
        frame_union, _frame_conf, _kept = _accumulate_cpu_retina_payload_to_prediction_frame(masks_np, int(out_size))
        return (frame_union > 0).astype(np.uint8, copy=False)
    if masks_np is None:
        return np.zeros((int(out_size), int(out_size)), dtype=np.uint8)
    masks_arr = np.asarray(masks_np)
    if masks_arr.ndim != 3 or int(masks_arr.shape[0]) <= 0:
        return np.zeros((int(out_size), int(out_size)), dtype=np.uint8)
    out = np.zeros((int(out_size), int(out_size)), dtype=np.uint8)
    for inst_idx in range(int(masks_arr.shape[0])):
        inst = np.asarray(masks_arr[int(inst_idx)], dtype=np.uint8)
        if int(inst.shape[0]) != int(out_size) or int(inst.shape[1]) != int(out_size):
            inst = cv2.resize(inst, (int(out_size), int(out_size)), interpolation=cv2.INTER_NEAREST)
        out |= (inst > 0).astype(np.uint8, copy=False)
    return out


def _process_cpu_retina_prediction_frame(
    idx: int,
    payload: CpuRetinaMaskPayload,
    out_size: int,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    slice_lock: Optional[threading.Lock] = None,
) -> Tuple[int, int]:
    """CPU equivalent of retina_masks=True accumulation without allocating GPU HxW masks."""
    frame_union, frame_confmap, kept_instances = _accumulate_cpu_retina_payload_to_prediction_frame(
        payload,
        int(out_size),
    )
    if int(kept_instances) <= 0 or not np.any(frame_union):
        return int(kept_instances), 0

    native_union = cv2.warpAffine(
        frame_union,
        M_out_to_native,
        dsize=(native_w, native_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    native_conf: Optional[np.ndarray] = None
    if frame_confmap is not None and np.any(frame_confmap):
        native_conf = cv2.warpAffine(
            frame_confmap,
            M_out_to_native,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8, copy=False)

    def _write_native_outputs() -> None:
        if np.any(native_union):
            view_union_mm[int(idx), :, :] |= native_union.astype(np.uint8, copy=False)

        if view_confmap_mm is not None and native_conf is not None and np.any(native_conf):
            conf_slice = view_confmap_mm[int(idx)]
            np.maximum(conf_slice, native_conf, out=conf_slice)

    if slice_lock is None:
        _write_native_outputs()
    else:
        with slice_lock:
            _write_native_outputs()

    return int(kept_instances), 1


def _process_prediction_frame(
    idx: int,
    masks_np: Optional[object],
    confs_np: Optional[np.ndarray],
    out_size: int,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    slice_lock: Optional[threading.Lock] = None,
) -> Tuple[int, int]:
    """Collapse one streamed result directly into unpacked native-view union + confidence volumes."""
    if isinstance(masks_np, CpuRetinaMaskPayload):
        return _process_cpu_retina_prediction_frame(
            idx=idx,
            payload=masks_np,
            out_size=out_size,
            view_union_mm=view_union_mm,
            view_confmap_mm=view_confmap_mm,
            M_out_to_native=M_out_to_native,
            native_h=native_h,
            native_w=native_w,
            slice_lock=slice_lock,
        )

    if masks_np is None:
        return 0, 0
    masks_arr = np.asarray(masks_np)
    if masks_arr.ndim != 3 or int(masks_arr.shape[0]) <= 0:
        return 0, 0

    track_conf = view_confmap_mm is not None
    frame_union = np.zeros((out_size, out_size), dtype=np.uint8)
    frame_confmap = np.zeros((out_size, out_size), dtype=np.uint8) if track_conf else None
    num_inst = int(masks_arr.shape[0])

    for inst_idx in range(num_inst):
        inst = np.asarray(masks_arr[inst_idx], dtype=np.uint8)
        if inst.shape[0] != out_size or inst.shape[1] != out_size:
            inst = cv2.resize(inst, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        inst = (inst > 0).astype(np.uint8, copy=False)
        if not np.any(inst):
            continue

        frame_union |= inst
        if track_conf and frame_confmap is not None:
            conf_val = float(confs_np[inst_idx]) if (confs_np is not None and inst_idx < int(confs_np.shape[0])) else 0.0
            conf_u8 = quantize_conf_to_u8(conf_val)
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
    if frame_confmap is not None and np.any(frame_confmap):
        native_conf = cv2.warpAffine(
            frame_confmap,
            M_out_to_native,
            dsize=(native_w, native_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8, copy=False)

    def _write_native_outputs() -> None:
        if np.any(native_union):
            view_union_mm[int(idx), :, :] |= native_union.astype(np.uint8, copy=False)

        if view_confmap_mm is not None and native_conf is not None and np.any(native_conf):
            conf_slice = view_confmap_mm[int(idx)]
            np.maximum(conf_slice, native_conf, out=conf_slice)

    if slice_lock is None:
        _write_native_outputs()
    else:
        with slice_lock:
            _write_native_outputs()

    return int(num_inst), 1

def predict_source_and_accumulate(
    model,
    source: object,
    *,
    source_label: str,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    postprocess_workers: int = 1,
    streaming_cleanup_enabled: bool = False,
    streaming_cleanup_min_conf: float = 0.0,
    streaming_cleanup_min_radius: float = 0.0,
    slice_locks: Optional[Sequence[threading.Lock]] = None,
) -> Dict[str, int]:
    """Run YOLO predict(stream=True) on an in-memory source and accumulate native masks.

    The predictor consumes ``InMemoryYoloVolumeSource`` instances in the v12 path,
    so the GPU no longer reads augmented FFV1 videos from scratch.  CPU-side result
    handling remains bounded by ``cpu_mask_postprocess_pending_limit`` and runs
    behind the streamed GPU inference.  When the v12.2.8 single-angle cleanup
    path is enabled, slice-local filtering is appended to the same streamed
    postprocess unit so a completed prediction slice is already cleaned before
    the full view volume has finished inferencing.
    """
    ensure_yolo_ready_for_predict(model, cfg)
    if isinstance(source, (InMemoryYoloVolumeSource, StreamingYoloVolumeSource)):
        ensure_single_channel_yolo_preprocess_patch()
    use_custom_cpu_retina = False
    if cpu_retina_masks_enabled():
        use_custom_cpu_retina = bool(ensure_cpu_retina_mask_predictor_patch())
        if not use_custom_cpu_retina:
            print('Warning: CPU retina predictor patch unavailable; using Ultralytics native retina_masks=True for this source.')

    prediction_count = 0
    frames_with_predictions = 0

    results = model.predict(
        source=source,
        task='segment',
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=1.0,
        save=False,
        stream=True,
        retina_masks=not bool(use_custom_cpu_retina),
        batch=max(1, int(cfg.batch)),
        device=cfg.device,
        half=cfg.half,
        int8=cfg.int8,
        verbose=False,
    )

    worker_count = max(1, min(int(postprocess_workers), int(num_frames)))
    pending_limit = cpu_mask_postprocess_pending_limit(worker_count, int(num_frames))
    stream_cleanup = bool(streaming_cleanup_enabled)
    stream_backend = cleanup_backend() if stream_cleanup else ''
    stream_structure2 = np.ones((3, 3), dtype=bool) if stream_cleanup else None
    stream_min_conf = float(streaming_cleanup_min_conf)
    stream_min_radius = float(streaming_cleanup_min_radius)
    stream_min_conf_u8 = int(min_conf_to_u8_threshold(stream_min_conf)) if stream_min_conf > 0.0 else 0

    def _process_prediction_unit(idx_i: int, masks_obj: Optional[object], confs_arr: Optional[np.ndarray]) -> Tuple[int, int]:
        slice_lock = None
        if slice_locks is not None and len(slice_locks) > 0:
            slice_lock = slice_locks[int(idx_i) % len(slice_locks)]
        pred_inc, frame_inc = _process_prediction_frame(
            idx=int(idx_i),
            masks_np=masks_obj,
            confs_np=confs_arr,
            out_size=out_size,
            view_union_mm=view_union_mm,
            view_confmap_mm=view_confmap_mm,
            M_out_to_native=M_out_to_native,
            native_h=native_h,
            native_w=native_w,
            slice_lock=slice_lock,
        )
        if stream_cleanup:
            cleaned_has_foreground = _cleanup_prediction_slice_inplace(
                view_union_mm,
                view_confmap_mm,
                int(idx_i),
                min_conf=stream_min_conf,
                min_radius=stream_min_radius,
                backend=stream_backend,
                structure2=stream_structure2,
                min_conf_u8=stream_min_conf_u8,
            )
            frame_inc = 1 if bool(cleaned_has_foreground) else 0
        return int(pred_inc), int(frame_inc)

    def _extract_and_process_result(idx_i: int, result_obj: object) -> Tuple[int, int]:
        masks_np, confs_np = _extract_result_masks_and_confs(result_obj)
        try:
            del result_obj
        except Exception:
            pass
        return _process_prediction_unit(int(idx_i), masks_np, confs_np)

    if worker_count <= 1:
        for idx, r in enumerate(results):
            if idx >= num_frames:
                # Synthetic final-batch padding is required for fixed-batch inference engines;
                # consume but discard those results so the predictor stream drains cleanly.
                continue
            masks_np, confs_np = _extract_result_masks_and_confs(r)
            pred_inc, frame_inc = _process_prediction_unit(int(idx), masks_np, confs_np)
            prediction_count += int(pred_inc)
            frames_with_predictions += int(frame_inc)
    else:
        pending: List[object] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for idx, r in enumerate(results):
                if idx >= num_frames:
                    # Discard synthetic repeated-slice results from the padded final batch.
                    continue

                pending.append(executor.submit(_extract_and_process_result, int(idx), r))
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

    if prediction_hot_path_flush_enabled():
        if view_confmap_mm is not None:
            flush_array(view_confmap_mm)
        flush_array(view_union_mm)

    return {
        'prediction_count': int(prediction_count),
        'frames_with_predictions': int(frames_with_predictions),
    }



def predict_source_and_submit_accumulation(
    model,
    source: object,
    *,
    source_label: str,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    postprocess_executor: ThreadPoolExecutor,
    streaming_cleanup_enabled: bool = False,
    streaming_cleanup_min_conf: float = 0.0,
    streaming_cleanup_min_radius: float = 0.0,
    slice_locks: Optional[Sequence[threading.Lock]] = None,
) -> PredictionAccumulationHandle:
    """Run YOLO streaming inference and enqueue result accumulation without draining it."""
    ensure_yolo_ready_for_predict(model, cfg)
    if isinstance(source, (InMemoryYoloVolumeSource, StreamingYoloVolumeSource)):
        ensure_single_channel_yolo_preprocess_patch()
    use_custom_cpu_retina = False
    if cpu_retina_masks_enabled():
        use_custom_cpu_retina = bool(ensure_cpu_retina_mask_predictor_patch())
        if not use_custom_cpu_retina:
            print('Warning: CPU retina predictor patch unavailable; using Ultralytics native retina_masks=True for this source.')

    results = model.predict(
        source=source,
        task='segment',
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=1.0,
        save=False,
        stream=True,
        retina_masks=not bool(use_custom_cpu_retina),
        batch=max(1, int(cfg.batch)),
        device=cfg.device,
        half=cfg.half,
        int8=cfg.int8,
        verbose=False,
    )

    stream_cleanup = bool(streaming_cleanup_enabled)
    stream_backend = cleanup_backend() if stream_cleanup else ''
    stream_structure2 = np.ones((3, 3), dtype=bool) if stream_cleanup else None
    stream_min_conf = float(streaming_cleanup_min_conf)
    stream_min_radius = float(streaming_cleanup_min_radius)
    stream_min_conf_u8 = int(min_conf_to_u8_threshold(stream_min_conf)) if stream_min_conf > 0.0 else 0

    def _process_prediction_unit(idx_i: int, masks_obj: Optional[object], confs_arr: Optional[np.ndarray]) -> Tuple[int, int]:
        slice_lock = None
        if slice_locks is not None and len(slice_locks) > 0:
            slice_lock = slice_locks[int(idx_i) % len(slice_locks)]
        pred_inc, frame_inc = _process_prediction_frame(
            idx=int(idx_i),
            masks_np=masks_obj,
            confs_np=confs_arr,
            out_size=out_size,
            view_union_mm=view_union_mm,
            view_confmap_mm=view_confmap_mm,
            M_out_to_native=M_out_to_native,
            native_h=native_h,
            native_w=native_w,
            slice_lock=slice_lock,
        )
        if stream_cleanup:
            cleaned_has_foreground = _cleanup_prediction_slice_inplace(
                view_union_mm,
                view_confmap_mm,
                int(idx_i),
                min_conf=stream_min_conf,
                min_radius=stream_min_radius,
                backend=stream_backend,
                structure2=stream_structure2,
                min_conf_u8=stream_min_conf_u8,
            )
            frame_inc = 1 if bool(cleaned_has_foreground) else 0
        return int(pred_inc), int(frame_inc)

    def _extract_and_process_result(idx_i: int, result_obj: object) -> Tuple[int, int]:
        masks_np, confs_np = _extract_result_masks_and_confs(result_obj)
        try:
            del result_obj
        except Exception:
            pass
        return _process_prediction_unit(int(idx_i), masks_np, confs_np)

    futures: List[Future] = []
    submitted_frames = 0
    synthetic_discarded = 0
    precompleted_prediction_count = 0
    precompleted_frames_with_predictions = 0
    pending_limit = async_predict_pending_frame_limit(int(num_frames))

    def _join_one_pending() -> None:
        nonlocal futures, precompleted_prediction_count, precompleted_frames_with_predictions
        if not futures:
            return
        done, remaining = wait(set(futures), return_when=FIRST_COMPLETED)
        futures = list(remaining)
        for fut_done in done:
            pred_inc, frame_inc = fut_done.result()
            precompleted_prediction_count += int(pred_inc)
            precompleted_frames_with_predictions += int(frame_inc)

    for idx, r in enumerate(results):
        if int(idx) >= int(num_frames):
            synthetic_discarded += 1
            continue
        submitted_frames += 1
        futures.append(postprocess_executor.submit(_extract_and_process_result, int(idx), r))
        while int(pending_limit) > 0 and len(futures) >= int(pending_limit):
            _join_one_pending()

    return PredictionAccumulationHandle(
        source_label=str(source_label),
        futures=futures,
        view_union_mm=view_union_mm,
        view_confmap_mm=view_confmap_mm,
        submitted_frames=int(submitted_frames),
        synthetic_discarded=int(synthetic_discarded),
        precompleted_prediction_count=int(precompleted_prediction_count),
        precompleted_frames_with_predictions=int(precompleted_frames_with_predictions),
        pending_limit=int(pending_limit),
    )


def predict_in_memory_volume_and_submit_accumulation(
    model,
    prediction_volume: PredictionVolumeRef,
    *,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    postprocess_executor: ThreadPoolExecutor,
    streaming_cleanup_enabled: bool = False,
    streaming_cleanup_min_conf: float = 0.0,
    streaming_cleanup_min_radius: float = 0.0,
    slice_locks: Optional[Sequence[threading.Lock]] = None,
) -> PredictionAccumulationHandle:
    source = make_prediction_ref_yolo_source(
        prediction_volume,
        batch_size=max(1, int(cfg.batch)),
        max_frames=int(num_frames),
    )
    return predict_source_and_submit_accumulation(
        model,
        source,
        source_label=prediction_volume.name,
        num_frames=int(num_frames),
        out_size=int(out_size),
        cfg=cfg,
        view_union_mm=view_union_mm,
        view_confmap_mm=view_confmap_mm,
        M_out_to_native=M_out_to_native,
        native_h=int(native_h),
        native_w=int(native_w),
        postprocess_executor=postprocess_executor,
        streaming_cleanup_enabled=bool(streaming_cleanup_enabled),
        streaming_cleanup_min_conf=float(streaming_cleanup_min_conf),
        streaming_cleanup_min_radius=float(streaming_cleanup_min_radius),
        slice_locks=slice_locks,
    )

def predict_in_memory_volume_and_accumulate(
    model,
    prediction_volume: PredictionVolumeRef,
    *,
    num_frames: int,
    out_size: int,
    cfg: PredictConfig,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    postprocess_workers: int = 1,
    streaming_cleanup_enabled: bool = False,
    streaming_cleanup_min_conf: float = 0.0,
    streaming_cleanup_min_radius: float = 0.0,
    slice_locks: Optional[Sequence[threading.Lock]] = None,
) -> Dict[str, int]:
    source = make_prediction_ref_yolo_source(
        prediction_volume,
        batch_size=max(1, int(cfg.batch)),
        max_frames=int(num_frames),
    )
    return predict_source_and_accumulate(
        model,
        source,
        source_label=prediction_volume.name,
        num_frames=int(num_frames),
        out_size=int(out_size),
        cfg=cfg,
        view_union_mm=view_union_mm,
        view_confmap_mm=view_confmap_mm,
        M_out_to_native=M_out_to_native,
        native_h=int(native_h),
        native_w=int(native_w),
        postprocess_workers=int(postprocess_workers),
        streaming_cleanup_enabled=bool(streaming_cleanup_enabled),
        streaming_cleanup_min_conf=float(streaming_cleanup_min_conf),
        streaming_cleanup_min_radius=float(streaming_cleanup_min_radius),
        slice_locks=slice_locks,
    )





























# --------------------------
# Per-view postprocessing
# --------------------------
# --------------------------
# Per-view postprocessing
# --------------------------




def cleanup_backend() -> str:
    """Return the per-slice cleanup backend.

    OpenCV is the default because the hot operations used here release the GIL and scale better
    under Python thread pools. Set YOLO_TTA_CLEANUP_BACKEND=scipy to recover the older scipy.ndimage
    cleanup path for debugging or strict regression comparison.
    """
    backend = os.environ.get('YOLO_TTA_CLEANUP_BACKEND', 'opencv').strip().lower()
    if backend not in {'opencv', 'scipy'}:
        backend = 'opencv'
    return backend


def _cv2_connected_components(mask_u8: np.ndarray, connectivity: int = 8) -> Tuple[int, np.ndarray]:
    return cv2.connectedComponents(
        np.ascontiguousarray(mask_u8, dtype=np.uint8),
        connectivity=int(connectivity),
        ltype=cv2.CV_32S,
    )


def _fill_holes_2d_scipy(mask_bool: np.ndarray) -> np.ndarray:
    return np.asarray(ndi.binary_fill_holes(np.asarray(mask_bool, dtype=bool)), dtype=bool)


def _fill_holes_2d_opencv(mask_bool: np.ndarray) -> np.ndarray:
    """Fill 2D holes using background connected components.

    This matches scipy.ndimage.binary_fill_holes' default 2D background connectivity (4-connected)
    while avoiding the slower Python-visible scipy path for thousands of large slices.
    """
    mask_u8 = np.ascontiguousarray(np.asarray(mask_bool, dtype=np.uint8))
    if mask_u8.size == 0 or not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)
    if bool(np.all(mask_u8)):
        return np.ones(mask_u8.shape, dtype=bool)

    bg_u8 = (mask_u8 == 0).astype(np.uint8, copy=False)
    num_labels, labels2d = _cv2_connected_components(bg_u8, connectivity=4)
    if int(num_labels) <= 1:
        return mask_u8.astype(bool, copy=False)

    touches_boundary = np.zeros((int(num_labels),), dtype=bool)
    touches_boundary[np.unique(labels2d[0, :])] = True
    touches_boundary[np.unique(labels2d[-1, :])] = True
    touches_boundary[np.unique(labels2d[:, 0])] = True
    touches_boundary[np.unique(labels2d[:, -1])] = True

    enclosed_bg = (labels2d > 0) & (~touches_boundary[labels2d])
    if np.any(enclosed_bg):
        mask_u8 = mask_u8.copy()
        mask_u8[enclosed_bg] = np.uint8(1)
    return mask_u8.astype(bool, copy=False)


def _fill_holes_2d(mask_bool: np.ndarray) -> np.ndarray:
    if cleanup_backend() == 'scipy':
        return _fill_holes_2d_scipy(mask_bool)
    return _fill_holes_2d_opencv(mask_bool)


def _filter_connected_components_by_min_radius_scipy(
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


def _filter_connected_components_by_min_radius_opencv(
    mask_bool: np.ndarray,
    min_radius: float,
) -> np.ndarray:
    mask_u8 = np.ascontiguousarray(np.asarray(mask_bool, dtype=np.uint8))
    if mask_u8.size == 0 or not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)

    num_labels, labels2d = _cv2_connected_components(mask_u8, connectivity=8)
    if int(num_labels) <= 1:
        return np.zeros(mask_u8.shape, dtype=bool)

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radii = np.zeros((int(num_labels),), dtype=np.float32)
    np.maximum.at(radii, labels2d.ravel(), np.asarray(dist, dtype=np.float32).ravel())
    keep_lookup = radii >= float(min_radius)
    keep_lookup[0] = False
    return keep_lookup[labels2d]


def _filter_connected_components_by_min_radius(
    mask_bool: np.ndarray,
    structure2: np.ndarray,
    min_radius: float,
) -> np.ndarray:
    if cleanup_backend() == 'scipy':
        return _filter_connected_components_by_min_radius_scipy(mask_bool, structure2, float(min_radius))
    return _filter_connected_components_by_min_radius_opencv(mask_bool, float(min_radius))


def _filter_connected_components_by_min_conf_opencv(
    mask_bool: np.ndarray,
    conf_slice: np.ndarray,
    min_conf_u8: int,
) -> np.ndarray:
    mask_u8 = np.ascontiguousarray(np.asarray(mask_bool, dtype=np.uint8))
    if mask_u8.size == 0 or not np.any(mask_u8):
        return np.zeros(mask_u8.shape, dtype=bool)

    num_labels, labels2d = _cv2_connected_components(mask_u8, connectivity=8)
    if int(num_labels) <= 1:
        return np.zeros(mask_u8.shape, dtype=bool)

    maxima = np.zeros((int(num_labels),), dtype=np.uint8)
    np.maximum.at(maxima, labels2d.ravel(), np.asarray(conf_slice, dtype=np.uint8).ravel())
    keep_lookup = maxima >= int(min_conf_u8)
    keep_lookup[0] = False
    return keep_lookup[labels2d]


def _filter_connected_components_by_min_conf_scipy(
    mask_bool: np.ndarray,
    conf_slice: np.ndarray,
    min_conf_u8: int,
    structure2: np.ndarray,
) -> np.ndarray:
    labels2d, num = ndi.label(mask_bool, structure=structure2)
    if int(num) <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    label_ids = np.arange(1, int(num) + 1, dtype=np.int32)
    maxima = np.asarray(ndi.maximum(conf_slice, labels=labels2d, index=label_ids), dtype=np.uint8)
    keep_ids = label_ids[maxima >= int(min_conf_u8)]
    if keep_ids.size <= 0:
        return np.zeros(mask_bool.shape, dtype=bool)
    return np.isin(labels2d, keep_ids)



def _view_native_slice_min_radius(view: ViewInfo, min_radius: float) -> float:
    """Return the part of --min_radius that is valid for independent view-native slices."""
    if float(min_radius) <= 0.0:
        return 0.0
    if view.name == 'transverse' or (is_tilted_view(view) and tilted_base_view_name(view) == 'transverse'):
        return float(min_radius)
    return 0.0


def _view_needs_deferred_volume_cleanup_after_streaming(view: ViewInfo, min_radius: float) -> bool:
    """Return True when streaming slice cleanup cannot finish all view cleanup semantics."""
    return bool(view.name in ('sagittal', 'coronal') and float(min_radius) > 0.0)


def _cleanup_prediction_slice_inplace(
    mask_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray],
    idx: int,
    *,
    min_conf: float = 0.0,
    min_radius: float = 0.0,
    backend: Optional[str] = None,
    structure2: Optional[np.ndarray] = None,
    min_conf_u8: Optional[int] = None,
) -> bool:
    """Clean one accumulated native-view prediction slice in place.

    This is the unit used by the v12.2.8 single-angle streaming path and by the
    older whole-volume cleanup pass. It applies only slice-local semantics:
    confidence gating, 2D hole filling, and view-native 2D min-radius filtering.
    Cleanup that depends on a transposed/volume-level view, such as Sagittal or
    Coronal transverse-plane min-radius filtering, is deliberately deferred until
    the full view volume exists.
    """
    idx_i = int(idx)
    backend_norm = cleanup_backend() if backend is None else str(backend)
    structure = np.ones((3, 3), dtype=bool) if structure2 is None else structure2
    min_conf_u8_i = (
        int(min_conf_to_u8_threshold(float(min_conf)))
        if min_conf_u8 is None and float(min_conf) > 0.0
        else int(min_conf_u8 or 0)
    )

    mask_slice = np.asarray(mask_mm[idx_i], dtype=bool)
    conf_slice = None if confmap_mm is None else np.asarray(confmap_mm[idx_i], dtype=np.uint8)

    if np.any(mask_slice) and conf_slice is not None and float(min_conf) > 0.0:
        if backend_norm == 'opencv':
            mask_slice = _filter_connected_components_by_min_conf_opencv(
                mask_slice,
                conf_slice,
                int(min_conf_u8_i),
            )
        else:
            mask_slice = _filter_connected_components_by_min_conf_scipy(
                mask_slice,
                conf_slice,
                int(min_conf_u8_i),
                structure,
            )

    if np.any(mask_slice):
        if backend_norm == 'opencv':
            mask_slice = _fill_holes_2d_opencv(mask_slice)
        else:
            mask_slice = _fill_holes_2d_scipy(mask_slice)

    if np.any(mask_slice) and float(min_radius) > 0.0:
        if backend_norm == 'opencv':
            mask_slice = _filter_connected_components_by_min_radius_opencv(
                mask_slice,
                float(min_radius),
            )
        else:
            mask_slice = _filter_connected_components_by_min_radius_scipy(
                mask_slice,
                structure,
                float(min_radius),
            )

    has_foreground = bool(np.any(mask_slice))
    mask_mm[idx_i, :, :] = mask_slice.astype(np.uint8, copy=False)
    if conf_slice is not None:
        if has_foreground:
            conf_slice[~mask_slice] = np.uint8(0)
        else:
            conf_slice.fill(np.uint8(0))
        confmap_mm[idx_i, :, :] = conf_slice.astype(np.uint8, copy=False)
    return bool(has_foreground)






_INFERENCE_BATCH_SIZE = 1


def set_inference_batch_size(batch_size: int) -> None:
    global _INFERENCE_BATCH_SIZE
    _INFERENCE_BATCH_SIZE = max(1, int(batch_size))


def inference_batch_size() -> int:
    return max(1, int(_INFERENCE_BATCH_SIZE))


def _cupy_scalar_to_int(value: object) -> int:
    try:
        item = getattr(value, 'item', None)
        if callable(item):
            return int(item())
    except Exception:
        pass
    return int(value)  # type: ignore[arg-type]


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
    min_conf_u8 = int(min_conf_to_u8_threshold(float(min_conf))) if float(min_conf) > 0.0 else 0
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)
    chunk_size = choose_parallel_chunk_size(num_slices, worker_count, target_chunks_per_worker=2, min_chunk_size=1)
    backend = cleanup_backend()

    def _process(i: int) -> None:
        _cleanup_prediction_slice_inplace(
            mask_mm,
            confmap_mm,
            int(i),
            min_conf=float(min_conf),
            min_radius=float(min_radius),
            backend=backend,
            structure2=structure2,
            min_conf_u8=int(min_conf_u8),
        )

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
    precleaned_slice_cleanup: bool = False,
) -> None:
    native_min_radius = _view_native_slice_min_radius(view, float(min_radius))

    if bool(precleaned_slice_cleanup):
        # v12.2.8 single-angle inference has already applied every slice-local cleanup
        # operation as results streamed in. Only cleanup whose semantics require the
        # completed view volume remains here.
        if _view_needs_deferred_volume_cleanup_after_streaming(view, float(min_radius)):
            apply_view_min_radius_filter_inplace(
                mask_mm,
                view,
                float(min_radius),
                workers=int(workers),
            )
        flush_array(mask_mm)
        if confmap_mm is not None:
            flush_array(confmap_mm)
        return

    effective_confmap_mm = confmap_mm if float(min_conf) > 0.0 else None

    fused_slice_cleanup_inplace(
        mask_mm,
        effective_confmap_mm,
        min_conf=float(min_conf),
        min_radius=float(native_min_radius),
        workers=int(workers),
        desc=f'Fused cleanup ({view.name})',
    )

    if _view_needs_deferred_volume_cleanup_after_streaming(view, float(min_radius)):
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


def _adjacent_gid_pair_codes(
    prev_gid: np.ndarray,
    curr_gid: np.ndarray,
    xy_offsets: Optional[Sequence[Tuple[int, int]]] = None,
    prev_offset: int = 0,
    curr_offset: int = 0,
) -> np.ndarray:
    """Return unique touching component-id pairs encoded as uint64 values.

    The upper 32 bits contain the previous-slice gid and the lower 32 bits contain the
    current-slice gid. ``prev_offset`` and ``curr_offset`` allow callers to pass per-slice
    local labels and encode global provisional ids without first rewriting the whole volume.
    Pair extraction is independent for each adjacent slice pair, so callers can run this
    function concurrently and then apply union-find merges serially.
    """
    h, w = prev_gid.shape
    offsets = tuple(xy_offsets) if xy_offsets is not None else _adjacent_xy_offsets_for_3d_connectivity(26)
    code_parts: List[np.ndarray] = []

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

        a_vals = a[overlap].astype(np.uint64, copy=False)
        b_vals = b[overlap].astype(np.uint64, copy=False)
        if int(prev_offset) != 0:
            a_vals = a_vals + np.uint64(int(prev_offset))
        if int(curr_offset) != 0:
            b_vals = b_vals + np.uint64(int(curr_offset))
        codes = (a_vals << np.uint64(32)) | b_vals
        if codes.size > 0:
            code_parts.append(np.unique(codes))

    if not code_parts:
        return np.zeros((0,), dtype=np.uint64)
    if len(code_parts) == 1:
        return np.asarray(code_parts[0], dtype=np.uint64)
    return np.unique(np.concatenate(code_parts).astype(np.uint64, copy=False))


def _iter_adjacent_gid_pairs(
    prev_gid: np.ndarray,
    curr_gid: np.ndarray,
    xy_offsets: Optional[Sequence[Tuple[int, int]]] = None,
) -> Iterator[Tuple[int, int]]:
    """Yield unique touching component-id pairs across adjacent z-slices.

    ``xy_offsets`` controls the 3D connectivity across adjacent slices. When omitted, the
    existing 26-connected behavior is used for foreground interpolation labeling.
    """
    for code in _adjacent_gid_pair_codes(prev_gid, curr_gid, xy_offsets):
        code_i = int(code)
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

    v12.2.0 allows 3D void fill only as an optional final global-union step. This
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
    wrap_axis: bool = False,
    workers: int = 1,
) -> Tuple[np.ndarray, int, List[Path]]:
    """Label a 3D foreground volume using parallel 2D slice labeling plus serial union-find.

      - the expensive per-slice 2D connected-component labeling runs concurrently across slices
      - local slice labels are promoted to global provisional ids with a parallel pass
      - adjacent-slice pair extraction can also run concurrently; only the final union-find merges
        remain serial to preserve deterministic 26-connected object identities
      - the compact relabel pass is slice-parallel
      - when wrap_axis is true, the first and last slices are treated as adjacent. This is used
        for Radial view-native interpolation over the [0°, 180°) diameter-frame domain.
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

    if int(z_dim) <= 0:
        flush_array(labels_store)
        return labels_store, 0, label_paths

    worker_count = choose_slice_parallel_workers(int(workers), int(z_dim))
    label_workers = choose_slice_parallel_workers(
        _env_int('YOLO_TTA_INTERPOLATION_LABEL_WORKERS', worker_count),
        int(z_dim),
    )
    compact_workers = choose_slice_parallel_workers(
        _env_int('YOLO_TTA_INTERPOLATION_COMPACT_WORKERS', worker_count),
        int(z_dim),
    )
    pair_workers = choose_slice_parallel_workers(
        _env_int('YOLO_TTA_INTERPOLATION_PAIR_WORKERS', worker_count),
        max(1, int(z_dim) - 1),
    )

    component_counts = np.zeros((int(z_dim),), dtype=np.uint32)

    def _label_slice_local(z: int) -> None:
        fg = (np.asarray(mask_mm[int(z)]) > 0).astype(np.uint8, copy=False)
        if fg.size == 0 or not np.any(fg):
            labels_store[int(z), :, :] = np.uint32(0)
            component_counts[int(z)] = np.uint32(0)
            return

        num_labels, labels2d = _cv2_connected_components(fg, connectivity=8)
        if int(num_labels) <= 1:
            labels_store[int(z), :, :] = np.uint32(0)
            component_counts[int(z)] = np.uint32(0)
            return

        labels_store[int(z), :, :] = np.asarray(labels2d, dtype=np.uint32)
        component_counts[int(z)] = np.uint32(int(num_labels) - 1)

    parallel_for_indices_chunked(
        int(z_dim),
        _label_slice_local,
        max_workers=label_workers,
        desc='Interpolation: 2D slice labeling',
        show_progress=True,
        target_chunks_per_worker=2,
    )

    total_components = int(np.sum(component_counts, dtype=np.uint64))
    if total_components <= 0:
        flush_array(labels_store)
        return labels_store, 0, label_paths
    if total_components >= 2 ** 32:
        raise RuntimeError('3D component id space exceeded uint32 capacity')

    # Slice z's provisional global id range is
    # [slice_offsets[z] + 1, slice_offsets[z] + component_counts[z]].  The label volume remains
    # in local per-slice ids until compact relabel so we avoid a full-volume promotion write pass.
    cumsum = np.cumsum(component_counts.astype(np.uint64, copy=False), dtype=np.uint64)
    slice_offsets = np.zeros((int(z_dim),), dtype=np.uint32)
    if int(z_dim) > 1:
        slice_offsets[1:] = cumsum[:-1].astype(np.uint32, copy=False)

    uf = _UnionFind()
    uf.new_ids(total_components)

    def _pair_codes_for_z(z: int) -> np.ndarray:
        prev_local_slice = np.asarray(labels_store[int(z) - 1])
        curr_local_slice = np.asarray(labels_store[int(z)])
        if not np.any(prev_local_slice) or not np.any(curr_local_slice):
            return np.zeros((0,), dtype=np.uint64)
        return _adjacent_gid_pair_codes(
            prev_local_slice,
            curr_local_slice,
            prev_offset=int(slice_offsets[int(z) - 1]),
            curr_offset=int(slice_offsets[int(z)]),
        )

    if int(z_dim) > 1:
        if pair_workers <= 1:
            pair_iter: Iterable[np.ndarray] = (_pair_codes_for_z(int(z)) for z in range(1, int(z_dim)))
        else:
            pair_iter = parallel_map_in_order(
                _pair_codes_for_z,
                range(1, int(z_dim)),
                max_workers=pair_workers,
                max_pending=max(pair_workers, pair_workers * 2),
            )
        for codes in tqdm(pair_iter, total=max(0, int(z_dim) - 1), desc='Interpolation: cross-slice unions'):
            for code in np.asarray(codes, dtype=np.uint64):
                code_i = int(code)
                uf.union(code_i >> 32, code_i & 0xFFFFFFFF)

    if bool(wrap_axis) and int(z_dim) > 1:
        first_gid_slice = np.asarray(labels_store[0])
        last_gid_slice = np.asarray(labels_store[int(z_dim) - 1])
        if np.any(first_gid_slice) and np.any(last_gid_slice):
            for code in _adjacent_gid_pair_codes(
                last_gid_slice,
                first_gid_slice,
                prev_offset=int(slice_offsets[int(z_dim) - 1]),
                curr_offset=int(slice_offsets[0]),
            ):
                code_i = int(code)
                uf.union(code_i >> 32, code_i & 0xFFFFFFFF)

    root_map = uf.root_map()
    if root_map.shape[0] <= 1:
        flush_array(labels_store)
        return labels_store, 0, label_paths

    unique_roots = np.unique(root_map[1:])
    unique_roots = unique_roots[unique_roots > 0]
    compact_root_ids = np.zeros(root_map.shape, dtype=np.uint32)
    compact_root_ids[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.uint32)
    gid_to_compact = compact_root_ids[root_map]

    local_to_compact_by_slice: List[np.ndarray] = []
    compact_tasks: List[Tuple[int, int, int]] = []
    row_block = max(1, _env_int('YOLO_TTA_INTERPOLATION_COMPACT_RELABEL_ROWS', 256))
    for z in range(int(z_dim)):
        count = int(component_counts[int(z)])
        local_to_compact = np.zeros((count + 1,), dtype=np.uint32)
        if count > 0:
            offset = int(slice_offsets[int(z)])
            local_to_compact[1:] = gid_to_compact[int(offset) + 1:int(offset) + int(count) + 1]
            for y0 in range(0, int(h), int(row_block)):
                compact_tasks.append((int(z), int(y0), int(min(int(h), int(y0) + int(row_block)))))
        local_to_compact_by_slice.append(local_to_compact)

    def _compact_block(task_idx: int) -> int:
        z, y0, y1 = compact_tasks[int(task_idx)]
        local_to_compact = local_to_compact_by_slice[int(z)]
        block = np.asarray(labels_store[int(z), int(y0):int(y1), :])
        if np.any(block):
            labels_store[int(z), int(y0):int(y1), :] = local_to_compact[block]
        else:
            labels_store[int(z), int(y0):int(y1), :].fill(np.uint32(0))
        return int(y1) - int(y0)

    if compact_tasks:
        pending = max(compact_workers, compact_workers * 8)
        if compact_workers <= 1:
            for task_idx in tqdm(range(len(compact_tasks)), desc='Interpolation: compact relabel'):
                _compact_block(int(task_idx))
        else:
            for _rows_done in tqdm(
                parallel_map_unordered(
                    _compact_block,
                    range(len(compact_tasks)),
                    max_workers=compact_workers,
                    max_pending=pending,
                ),
                total=len(compact_tasks),
                desc='Interpolation: compact relabel',
            ):
                pass

    flush_array(labels_store)
    return labels_store, int(unique_roots.size), label_paths


def build_slice_endpoint_seeds_from_label_volume(
    labels_real: np.ndarray,
    workers: int = 1,
    desc: str = 'Interpolation: endpoint seeds [scan]',
    wrap_axis: bool = False,
    component_cache: Optional['SliceComponentTableCache'] = None,
) -> Tuple[List[SliceEndpointSeed], int]:
    """Fast slice-graph endpoint scan for slice-direction interpolation.

    This avoids per-object 3D voxel skeletonization on large relabeled volumes, which can become
    prohibitively slow when an object's bounding box spans a large fraction of the volume. Endpoints
    are identified from connected components in each slice that do not continue into the previous or
    next slice of the same relabeled object. When wrap_axis is true, slice 0 and the
    final slice are also considered adjacent for Radial interpolation.

    When a component cache is supplied, the scan reuses cached per-slice component tables and tests
    continuation only inside each component's crop. This makes endpoint discovery and later seed
    planning share the same component records instead of relabeling slices again per seed/bridge.
    """
    z_dim = int(labels_real.shape[0])
    if z_dim <= 0:
        return [], 0

    def _scan_slice_from_cache(z: int) -> List[SliceEndpointSeed]:
        if component_cache is None:
            return []
        z_i = int(z)
        table = component_cache.get(z_i)
        if not table.components:
            return []

        prev_table: Optional[SliceComponentTable] = None
        next_table: Optional[SliceComponentTable] = None
        if bool(wrap_axis) and z_dim > 1:
            prev_table = component_cache.get((z_i - 1) % z_dim)
            next_table = component_cache.get((z_i + 1) % z_dim)
        else:
            if z_i > 0:
                prev_table = component_cache.get(z_i - 1)
            if (z_i + 1) < z_dim:
                next_table = component_cache.get(z_i + 1)

        seeds_local: List[SliceEndpointSeed] = []
        for record in table.components:
            prev_records = prev_table.by_label.get(int(record.label), []) if prev_table is not None else []
            next_records = next_table.by_label.get(int(record.label), []) if next_table is not None else []
            has_prev = any(_component_records_directly_overlap(record, other) for other in prev_records)
            has_next = any(_component_records_directly_overlap(record, other) for other in next_records)

            if not has_prev:
                seeds_local.append(SliceEndpointSeed(
                    label=int(record.label),
                    point=(z_i, int(record.anchor[0]), int(record.anchor[1])),
                    direction_sign=-1,
                ))
            if not has_next:
                seeds_local.append(SliceEndpointSeed(
                    label=int(record.label),
                    point=(z_i, int(record.anchor[0]), int(record.anchor[1])),
                    direction_sign=1,
                ))
        return seeds_local

    def _scan_slice(z: int) -> List[SliceEndpointSeed]:
        if component_cache is not None:
            return _scan_slice_from_cache(int(z))

        curr_slice = np.asarray(labels_real[int(z)])
        if not np.any(curr_slice):
            return []

        if bool(wrap_axis) and z_dim > 1:
            prev_slice = np.asarray(labels_real[(int(z) - 1) % z_dim])
            next_slice = np.asarray(labels_real[(int(z) + 1) % z_dim])
        else:
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

                # v12.2.0 endpoint continuation is defined by direct footprint overlap in
                # the adjacent slice; do not dilate/skeletonize the component for endpoint discovery.
                has_prev = bool(prev_same is not None and np.any(comp & prev_same))
                has_next = bool(next_same is not None and np.any(comp & next_same))

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
    if component_cache is not None:
        component_cache.prebuild(
            workers=worker_count,
            desc=f'{desc}: component tables',
        )

    seeds: List[SliceEndpointSeed] = []

    if worker_count <= 1:
        for z in tqdm(range(z_dim), desc=desc):
            seeds.extend(_scan_slice(int(z)))
    else:
        pending = max(worker_count, worker_count * 8)
        for seeds_local in tqdm(
            parallel_map_unordered(_scan_slice, range(z_dim), max_workers=worker_count, max_pending=pending),
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

    if view.name == 'transverse' or view.family == 'radial' or (is_tilted_view(view) and tilted_base_view_name(view) == 'transverse'):
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


@dataclass(frozen=True)
class RadialBackprojectionSample:
    angle_deg: float
    source_index: int
    reverse_u: bool = False


@dataclass(frozen=True)
class DenseRadialBackprojectionMap:
    valid_mask: np.ndarray
    source_idx_map: np.ndarray
    u_idx_map: np.ndarray


_DENSE_RADIAL_BACKPROJECT_MAP_CACHE: Dict[Tuple[int, int, int, Tuple[Tuple[float, int, bool], ...]], DenseRadialBackprojectionMap] = {}


def radial_full_coverage_angle_deg(diameter: int) -> float:
    """Return the v12.2.0 radial spacing that gives approximately 1x ROI edge coverage."""
    diameter_i = max(1, int(diameter))
    return float(360.0 / (math.pi * float(diameter_i)))


def _radial_view_nominal_spacing_deg(radial_view: ViewInfo) -> float:
    az = list(float(a) for a in radial_view.azimuths_deg)
    if len(az) >= 2:
        return float(abs(az[1] - az[0]))
    return 180.0


def _nearest_radial_source_for_backprojection(
    target_angle_deg: float,
    source_angles_deg: np.ndarray,
) -> Tuple[int, bool]:
    """Nearest source plane over the unoriented [0, 180) radial diameter domain.

    When the nearest match crosses the 0/180 boundary, the same diameter plane is reused with
    the radial coordinate reversed so source raster coordinates still map to the target plane.
    """
    if source_angles_deg.size <= 0:
        raise ValueError('No radial source angles are available for backprojection')

    raw = float(target_angle_deg) - source_angles_deg.astype(np.float64, copy=False)
    wrapped = ((raw + 90.0) % 180.0) - 90.0
    idx = int(np.argmin(np.abs(wrapped)))
    reverse_u = bool(abs(float(raw[idx])) > 90.0)
    return idx, reverse_u


def build_radial_backprojection_plan(radial_view: ViewInfo) -> Tuple[List[RadialBackprojectionSample], Dict[str, float]]:
    """Build the angular plan used to backproject a radial view into Cartesian space.

    If the user-requested radial spacing is coarser than the v12.2.0 full-coverage spacing,
    backprojection is densified to the full-coverage spacing and each dense angle samples the
    nearest completed radial prediction frame. This keeps Radial masks view-native through
    postprocessing/interpolation while preventing sparse spoke-like Cartesian backprojections.
    """
    source_angles = np.asarray([float(a) for a in radial_view.azimuths_deg], dtype=np.float64)
    if source_angles.size <= 0:
        return [], {
            'provided_spacing_deg': 0.0,
            'coverage_spacing_deg': radial_full_coverage_angle_deg(int(radial_view.diameter)),
            'effective_spacing_deg': 0.0,
            'source_frames': 0.0,
            'backprojection_angles': 0.0,
            'densified': 0.0,
        }

    provided_spacing = _radial_view_nominal_spacing_deg(radial_view)
    coverage_spacing = radial_full_coverage_angle_deg(int(radial_view.diameter))
    # The specification's guarantee formula is a maximum safe angular spacing.  A smaller
    # user spacing is already dense enough; a larger spacing is densified during backprojection.
    effective_spacing = min(float(provided_spacing), float(coverage_spacing))

    if float(provided_spacing) <= float(coverage_spacing) * (1.0 + 1e-9):
        samples = [
            RadialBackprojectionSample(angle_deg=float(angle), source_index=int(idx), reverse_u=False)
            for idx, angle in enumerate(source_angles.tolist())
        ]
        densified = False
    else:
        dense_angles = build_radial_azimuths(float(effective_spacing))
        samples = []
        for dense_angle in dense_angles:
            source_idx, reverse_u = _nearest_radial_source_for_backprojection(float(dense_angle), source_angles)
            samples.append(RadialBackprojectionSample(
                angle_deg=float(dense_angle),
                source_index=int(source_idx),
                reverse_u=bool(reverse_u),
            ))
        densified = True

    return samples, {
        'provided_spacing_deg': float(provided_spacing),
        'coverage_spacing_deg': float(coverage_spacing),
        'effective_spacing_deg': float(effective_spacing),
        'source_frames': float(source_angles.size),
        'backprojection_angles': float(len(samples)),
        'densified': 1.0 if bool(densified) else 0.0,
    }


def _radial_plan_signature(plan: Sequence[RadialBackprojectionSample]) -> Tuple[Tuple[float, int, bool], ...]:
    return tuple((round(float(s.angle_deg), 6), int(s.source_index), bool(s.reverse_u)) for s in plan)


def build_dense_radial_backprojection_map(
    radial_view: ViewInfo,
    plan: Sequence[RadialBackprojectionSample],
) -> DenseRadialBackprojectionMap:
    """Map every Cartesian ROI pixel to its nearest dense radial source frame and radial coordinate.

    This is the dense v12.2.0 backprojection path: instead of painting only the pixels that lie on
    a set of spokes, every pixel inside the circular ROI receives a sample from the Radial video's
    own frame/raster coordinate system.  When the user-provided azimuth spacing is too coarse, the
    supplied ``plan`` is already densified to the full-coverage angular spacing; pixels then select
    the nearest dense plan angle and that angle's nearest completed source frame.
    """
    if not plan:
        return DenseRadialBackprojectionMap(
            valid_mask=np.zeros((int(radial_view.full_h), int(radial_view.full_w)), dtype=bool),
            source_idx_map=np.zeros((int(radial_view.full_h), int(radial_view.full_w)), dtype=np.int32),
            u_idx_map=np.zeros((int(radial_view.full_h), int(radial_view.full_w)), dtype=np.int32),
        )

    key = (
        int(radial_view.full_h),
        int(radial_view.full_w),
        int(radial_view.diameter),
        _radial_plan_signature(plan),
    )
    cached = _DENSE_RADIAL_BACKPROJECT_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    out_h = int(radial_view.full_h)
    out_w = int(radial_view.full_w)
    diameter = int(radial_view.diameter)
    radius = float(radial_view.roi_radius)
    if radius <= 0.0:
        radius = max(1.0, float(diameter - 1) / 2.0)

    plan_angles = np.asarray([float(s.angle_deg) % 180.0 for s in plan], dtype=np.float32)
    plan_sources = np.asarray([int(s.source_index) for s in plan], dtype=np.int32)
    plan_reverses = np.asarray([bool(s.reverse_u) for s in plan], dtype=bool)
    n_plan = int(plan_angles.size)
    if n_plan <= 0:
        raise ValueError('Dense radial backprojection requires at least one angular plan sample')

    if n_plan >= 2:
        diffs = np.diff(plan_angles.astype(np.float64, copy=False))
        positive_diffs = diffs[diffs > 1e-9]
        step = float(np.median(positive_diffs)) if positive_diffs.size else 180.0 / float(n_plan)
    else:
        step = 180.0
    step = max(float(step), 1e-9)

    yy, xx = np.indices((out_h, out_w), dtype=np.float32)
    dx = xx - float(radial_view.center_x)
    dy = yy - float(radial_view.center_y)
    rr = np.sqrt((dx * dx) + (dy * dy)).astype(np.float32, copy=False)
    valid = rr <= (float(radius) + 0.5)

    theta = np.degrees(np.arctan2(dy, dx)).astype(np.float32, copy=False)
    theta = np.mod(theta, 180.0).astype(np.float32, copy=False)
    nearest_plan_idx = np.mod(np.rint(theta / float(step)).astype(np.int32, copy=False), n_plan)

    target_angles = plan_angles[nearest_plan_idx]
    cos_t = np.cos(np.deg2rad(target_angles)).astype(np.float32, copy=False)
    sin_t = np.sin(np.deg2rad(target_angles)).astype(np.float32, copy=False)
    signed_r = (dx * cos_t) + (dy * sin_t)
    signed_r[plan_reverses[nearest_plan_idx]] *= -1.0

    u_float = ((signed_r + float(radius)) / max(1e-6, 2.0 * float(radius))) * float(diameter - 1)
    u_idx = np.clip(np.rint(u_float).astype(np.int32, copy=False), 0, diameter - 1)
    source_idx = plan_sources[nearest_plan_idx].astype(np.int32, copy=False)

    source_idx[~valid] = 0
    u_idx[~valid] = 0

    dense_map = DenseRadialBackprojectionMap(
        valid_mask=np.ascontiguousarray(valid),
        source_idx_map=np.ascontiguousarray(source_idx),
        u_idx_map=np.ascontiguousarray(u_idx),
    )
    _DENSE_RADIAL_BACKPROJECT_MAP_CACHE[key] = dense_map
    return dense_map


def backproject_radial_volume_to_volume(
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    backend: str = 'cpu',
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

    plan, plan_stats = build_radial_backprojection_plan(radial_view)
    if bool(plan_stats.get('densified', 0.0)):
        print(
            f"{desc}: densifying radial backprojection for continuity "
            f"from {int(plan_stats['source_frames'])} source frame(s) at "
            f"{float(plan_stats['provided_spacing_deg']):.6g}° spacing to "
            f"{int(plan_stats['backprojection_angles'])} backprojection angle(s) at "
            f"{float(plan_stats['effective_spacing_deg']):.6g}° spacing "
            f"(full-coverage threshold {float(plan_stats['coverage_spacing_deg']):.6g}°)"
        )

    if not plan:
        flush_array(vol_mm)
        return vol_mm

    dense_map = build_dense_radial_backprojection_map(radial_view, plan)
    valid = np.asarray(dense_map.valid_mask, dtype=bool)
    source_idx_map = np.asarray(dense_map.source_idx_map, dtype=np.int32)
    u_idx_map = np.asarray(dense_map.u_idx_map, dtype=np.int32)

    backend_norm = str(backend or 'cpu').strip().lower()
    if backend_norm in ('gpu', 'auto'):
        gpu_done = try_backproject_radial_volume_to_volume_gpu(
            radial_mask_mm=radial_mask_mm,
            radial_view=radial_view,
            vol_mm=vol_mm,
            dense_map=dense_map,
            desc=desc,
        )
        if bool(gpu_done):
            flush_array(vol_mm)
            return vol_mm
        if backend_norm == 'gpu':
            print(f'{desc}: GPU radial backprojection unavailable or failed; falling back to CPU backend.')

    worker_count = choose_slice_parallel_workers(int(workers), int(t_dim))

    def _backproject_t_slice(t_idx: int) -> None:
        radial_plane = np.asarray(radial_mask_mm[:, int(t_idx), :], dtype=np.uint8)
        dst = vol_mm[int(t_idx)]
        # Dense gather: every in-ROI Cartesian pixel is assigned from the nearest dense radial
        # angle's source frame and radial raster coordinate. Out-of-ROI remains black.
        gathered = radial_plane[source_idx_map, u_idx_map]
        dst[valid] = gathered[valid]

    parallel_for_indices_chunked(
        int(t_dim),
        _backproject_t_slice,
        max_workers=worker_count,
        desc=desc,
        show_progress=True,
        target_chunks_per_worker=2,
    )

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
    workers: int = 1,
    backend: str = 'cpu',
) -> np.ndarray:
    """Backproject a v12 Tilted View-native mask stack into native (t, Y, X).

    The input stack remains in the generated Tilted View's own frame order and
    base-view raster coordinates through cleanup and interpolation.  Only after
    all per-view operations are complete do we map each foreground pixel to the
    orthogonal processing volume by applying the same stacking-axis shear used
    during rendering:

      * Tilted Transverse: base axes (X, Y), stacking axis t
      * Tilted Sagittal:  base axes (X, t), stacking axis Y
      * Tilted Coronal:   base axes (Y, t), stacking axis X
    """
    if not is_tilted_view(tilted_view):
        raise ValueError('backproject_tilted_volume_to_volume expects a Tilted View')

    t_dim = int(tilted_view.full_t)
    out_h = int(tilted_view.full_h)
    out_w = int(tilted_view.full_w)
    if t_dim <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError(
            f'Tilted view {tilted_view.name} has invalid output geometry '
            f'(t,Y,X)=({t_dim},{out_h},{out_w})'
        )

    src = np.asarray(tilted_mask_mm)
    expected_shape = (int(tilted_view.num_slices), int(tilted_view.src_h), int(tilted_view.src_w))
    if tuple(int(x) for x in src.shape) != expected_shape:
        raise ValueError(f'{desc}: tilted layer shape {tuple(src.shape)} != {expected_shape}')

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
        axis_center = float((int(tilted_view.src_h) - 1) / 2.0)
    elif str(tilted_view.tilt_direction) == 'horizontal':
        axis_center = float((int(tilted_view.src_w) - 1) / 2.0)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported tilt direction: {tilted_view.tilt_direction}')

    base_view = tilted_base_view_name(tilted_view)
    stack_len = tilted_stack_axis_length(tilted_view)

    backend_norm = str(backend or 'cpu').strip().lower()
    if backend_norm in ('gpu', 'auto'):
        gpu_done = try_backproject_tilted_volume_to_volume_gpu(
            tilted_mask_mm=src,
            tilted_view=tilted_view,
            vol_mm=vol_mm,
            desc=desc,
        )
        if bool(gpu_done):
            flush_array(vol_mm)
            return vol_mm
        if backend_norm == 'gpu':
            print(f'{desc}: GPU tilted backprojection unavailable or failed; falling back to CPU backend.')

    worker_count = choose_slice_parallel_workers(int(workers), int(tilted_view.num_slices))

    def _backproject_frame(frame_idx: int) -> None:
        tilted_mask = np.asarray(src[int(frame_idx)], dtype=bool)
        if not np.any(tilted_mask):
            return
        vv, uu = np.nonzero(tilted_mask)
        if vv.size <= 0:
            return

        frame_center = float(tilted_frame_center(tilted_view, int(frame_idx)))
        if str(tilted_view.tilt_direction) == 'vertical':
            stack_float = frame_center + tan_alpha * (vv.astype(np.float32, copy=False) - axis_center)
        else:
            stack_float = frame_center + tan_alpha * (uu.astype(np.float32, copy=False) - axis_center)

        ss = np.rint(stack_float).astype(np.int32, copy=False)
        valid = (ss >= 0) & (ss < int(stack_len))
        if not np.any(valid):
            return

        ss_v = ss[valid]
        vv_v = vv[valid]
        uu_v = uu[valid]
        if base_view == 'transverse':
            # base in-plane: horizontal X=uu, vertical Y=vv; stack t=ss
            vol_mm[ss_v, vv_v, uu_v] = np.uint8(1)
        elif base_view == 'sagittal':
            # base in-plane: horizontal X=uu, vertical t=vv; stack Y=ss
            vol_mm[vv_v, ss_v, uu_v] = np.uint8(1)
        elif base_view == 'coronal':
            # base in-plane: horizontal Y=uu, vertical t=vv; stack X=ss
            vol_mm[vv_v, uu_v, ss_v] = np.uint8(1)
        else:  # pragma: no cover
            raise ValueError(f'Unsupported Tilted View base: {base_view}')

    parallel_for_indices_chunked(
        int(tilted_view.num_slices),
        _backproject_frame,
        max_workers=worker_count,
        desc=desc,
        show_progress=True,
        target_chunks_per_worker=4,
    )

    flush_array(vol_mm)
    return vol_mm



def _try_import_cupy_for_backprojection() -> Optional[object]:
    """Backprojection is intentionally CPU-only in v12.2.11.

    CuPy remains available to the optional Gaussian smoothing backend, but Radial
    and Tilted final backprojection no longer import or schedule CUDA work.  This
    keeps the GPU dedicated to inference and removes the utilization spikes/dips
    previously observed around final view projection and NRRD preparation.
    """
    return None


def _cupy_free_bytes(cp: object) -> int:
    try:
        free_b, _total_b = cp.cuda.runtime.memGetInfo()
        return int(free_b)
    except Exception:
        return 0


def _copy_cupy_volume_to_cpu_workspace(cp: object, vol_cp: object, vol_mm: np.ndarray, desc: str) -> None:
    z_dim = int(vol_mm.shape[0])
    copy_chunk = max(1, _env_int('YOLO_TTA_BACKPROJECT_GPU_COPY_SLICES', 8))
    for z0 in tqdm(range(0, z_dim, copy_chunk), desc=f'{desc}: GPU result copy'):
        z1 = min(z_dim, int(z0) + int(copy_chunk))
        vol_mm[int(z0):int(z1), :, :] = cp.asnumpy(vol_cp[int(z0):int(z1), :, :]).astype(np.uint8, copy=False)


def try_backproject_radial_volume_to_volume_gpu(
    *,
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
    vol_mm: np.ndarray,
    dense_map: DenseRadialBackprojectionMap,
    desc: str,
) -> bool:
    """GPU dense radial backprojection, returning False when unavailable/unsafe."""
    cp = _try_import_cupy_for_backprojection()
    if cp is None:
        return False
    try:
        t_dim = int(radial_view.src_h)
        out_h = int(radial_view.full_h)
        out_w = int(radial_view.full_w)
        map_bytes = int(out_h) * int(out_w) * (np.dtype(np.int32).itemsize * 2 + np.dtype(np.bool_).itemsize)
        batch_slices = max(1, _env_int('YOLO_TTA_RADIAL_BACKPROJECT_GPU_BATCH', max(1, min(16, inference_batch_size()))))
        source_bytes_per_t = int(radial_mask_mm.shape[0]) * int(radial_mask_mm.shape[2])
        need_bytes = int(map_bytes) + int(batch_slices) * (int(source_bytes_per_t) + int(out_h) * int(out_w)) + 2 * GIB
        free_bytes = _cupy_free_bytes(cp)
        if free_bytes and int(need_bytes) > int(free_bytes):
            batch_slices = max(1, min(batch_slices, int(max(1, (int(free_bytes) - 2 * GIB - int(map_bytes)) // max(1, int(source_bytes_per_t) + int(out_h) * int(out_w))))))
        if batch_slices <= 0:
            return False

        print(f'{desc}: using GPU radial backprojection (batch_t={batch_slices})')
        valid_cp = cp.asarray(np.asarray(dense_map.valid_mask, dtype=bool))
        source_idx_cp = cp.asarray(np.asarray(dense_map.source_idx_map, dtype=np.int32))
        u_idx_cp = cp.asarray(np.asarray(dense_map.u_idx_map, dtype=np.int32))
        for t0 in tqdm(range(0, t_dim, int(batch_slices)), desc=f'{desc} [GPU radial]'):
            t1 = min(t_dim, int(t0) + int(batch_slices))
            planes_np = np.ascontiguousarray(np.asarray(radial_mask_mm[:, int(t0):int(t1), :], dtype=np.uint8))
            planes_cp = cp.asarray(planes_np, dtype=cp.uint8)
            out_batch_cp = cp.zeros((int(t1 - t0), int(out_h), int(out_w)), dtype=cp.uint8)
            for bi in range(int(t1 - t0)):
                gathered = planes_cp[:, int(bi), :][source_idx_cp, u_idx_cp]
                out_slice = out_batch_cp[int(bi)]
                out_slice[valid_cp] = gathered[valid_cp]
            vol_mm[int(t0):int(t1), :, :] = cp.asnumpy(out_batch_cp).astype(np.uint8, copy=False)
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        flush_array(vol_mm)
        return True
    except Exception as exc:
        print(f'{desc}: GPU radial backprojection failed ({exc}); CPU fallback will be used.')
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return False


def try_backproject_tilted_volume_to_volume_gpu(
    *,
    tilted_mask_mm: np.ndarray,
    tilted_view: ViewInfo,
    vol_mm: np.ndarray,
    desc: str,
) -> bool:
    """GPU tilted backprojection using a full output uint8 volume on the GPU when it fits."""
    cp = _try_import_cupy_for_backprojection()
    if cp is None:
        return False
    try:
        t_dim = int(tilted_view.full_t)
        out_h = int(tilted_view.full_h)
        out_w = int(tilted_view.full_w)
        output_bytes = int(t_dim) * int(out_h) * int(out_w)
        free_bytes = _cupy_free_bytes(cp)
        reserve_bytes = int(max(2.0, _env_float('YOLO_TTA_TILTED_BACKPROJECT_GPU_RESERVE_GIB', 8.0)) * GIB)
        if free_bytes and output_bytes + reserve_bytes > free_bytes:
            print(
                f'{desc}: GPU tilted backprojection skipped because output volume needs '
                f'{output_bytes / GIB:.1f} GiB with reserve {reserve_bytes / GIB:.1f} GiB but only {free_bytes / GIB:.1f} GiB is free.'
            )
            return False

        print(f'{desc}: using GPU tilted backprojection (full output volume {output_bytes / GIB:.1f} GiB on GPU)')
        vol_cp = cp.zeros((int(t_dim), int(out_h), int(out_w)), dtype=cp.uint8)
        tan_alpha = float(math.tan(math.radians(float(tilted_view.tilt_angle_deg))))
        if str(tilted_view.tilt_direction) == 'vertical':
            axis_center = float((int(tilted_view.src_h) - 1) / 2.0)
        elif str(tilted_view.tilt_direction) == 'horizontal':
            axis_center = float((int(tilted_view.src_w) - 1) / 2.0)
        else:
            return False
        base_view = tilted_base_view_name(tilted_view)
        stack_len = tilted_stack_axis_length(tilted_view)

        for frame_idx in tqdm(range(int(tilted_view.num_slices)), desc=f'{desc} [GPU tilted]'):
            tilted_mask_np = np.asarray(tilted_mask_mm[int(frame_idx)], dtype=bool)
            if not np.any(tilted_mask_np):
                continue
            mask_cp = cp.asarray(tilted_mask_np, dtype=cp.bool_)
            vv, uu = cp.nonzero(mask_cp)
            if int(vv.size) <= 0:
                continue
            frame_center = float(tilted_frame_center(tilted_view, int(frame_idx)))
            if str(tilted_view.tilt_direction) == 'vertical':
                stack_float = frame_center + tan_alpha * (vv.astype(cp.float32, copy=False) - float(axis_center))
            else:
                stack_float = frame_center + tan_alpha * (uu.astype(cp.float32, copy=False) - float(axis_center))
            ss = cp.rint(stack_float).astype(cp.int32, copy=False)
            valid = cp.logical_and(ss >= 0, ss < int(stack_len))
            if not bool(_cupy_scalar_to_int(cp.any(valid))):
                continue
            ss_v = ss[valid]
            vv_v = vv[valid]
            uu_v = uu[valid]
            if base_view == 'transverse':
                vol_cp[ss_v, vv_v, uu_v] = cp.uint8(1)
            elif base_view == 'sagittal':
                vol_cp[vv_v, ss_v, uu_v] = cp.uint8(1)
            elif base_view == 'coronal':
                vol_cp[vv_v, uu_v, ss_v] = cp.uint8(1)
            else:
                return False
        _copy_cupy_volume_to_cpu_workspace(cp, vol_cp, vol_mm, desc)
        flush_array(vol_mm)
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return True
    except Exception as exc:
        print(f'{desc}: GPU tilted backprojection failed ({exc}); CPU fallback will be used.')
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return False


@dataclass(frozen=True)
class ViewBackprojectionQueueJob:
    model_name: str
    view: ViewInfo
    native_source: np.ndarray
    out_path: Path
    desc: str
    min_radius: float = 0.0
    workers: int = 1


class HybridBackprojectionQueue:
    """CPU-only radial/tilted backprojection queue.

    The historical class name is retained to avoid churn in scheduler call
    sites, but v12.2.11 disables the GPU slot entirely.  Each queued set should
    therefore receive the full resolved CPU slice-worker budget.
    """

    def __init__(self, *, cpu_workers: int = 1, gpu_enabled: bool = True) -> None:
        self.cpu_workers = max(1, int(cpu_workers))
        del gpu_enabled
        self.gpu_enabled = False

    def _run_job(self, job: ViewBackprojectionQueueJob, backend: str) -> Tuple[str, str, np.ndarray]:
        view_local = job.view
        if view_local.family == 'radial':
            projected = backproject_radial_volume_to_volume(
                radial_mask_mm=job.native_source,
                radial_view=view_local,
                out_path=job.out_path,
                desc=job.desc,
                prefer_memory=True,
                workers=int(job.workers),
                backend=str(backend),
            )
        elif is_tilted_view(view_local):
            projected = backproject_tilted_volume_to_volume(
                tilted_mask_mm=job.native_source,
                tilted_view=view_local,
                out_path=job.out_path,
                desc=job.desc,
                prefer_memory=True,
                workers=int(job.workers),
                backend=str(backend),
            )
        else:  # pragma: no cover
            raise ValueError(f'Unsupported queued backprojection view family: {view_local.family}')

        if float(job.min_radius) > 0.0:
            print(f"Applying --min_radius in the transverse plane for backprojected view '{view_local.name}' [{backend}]")
            apply_transverse_min_radius_filter_inplace(
                projected,
                float(job.min_radius),
                workers=int(job.workers),
            )
        return str(job.model_name), str(view_local.name), projected

    def run(self, jobs: Sequence[ViewBackprojectionQueueJob]) -> List[Tuple[str, str, np.ndarray]]:
        queue = deque(jobs)
        results: List[Tuple[str, str, np.ndarray]] = []
        cpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='backproject-cpu')
        gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='backproject-gpu') if self.gpu_enabled else None
        pending: Dict[Future, str] = {}
        try:
            while queue or pending:
                if self.gpu_enabled and gpu_executor is not None and 'gpu' not in pending.values() and queue:
                    job = queue.popleft()
                    print(f'Backprojection queue: assigning {job.model_name}/{job.view.name} to GPU')
                    pending[gpu_executor.submit(self._run_job, job, 'gpu')] = 'gpu'
                if 'cpu' not in pending.values() and queue:
                    job = queue.popleft()
                    print(f'Backprojection queue: assigning {job.model_name}/{job.view.name} to CPU')
                    pending[cpu_executor.submit(self._run_job, job, 'cpu')] = 'cpu'
                if not pending:
                    continue
                done, _not_done = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.pop(fut, None)
                    results.append(fut.result())
        finally:
            cpu_executor.shutdown(wait=True)
            if gpu_executor is not None:
                gpu_executor.shutdown(wait=True)
        return results

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

    Transverse, already-backprojected Tilted Views, and already-backprojected Radial volumes
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

    if "sagittal" in view_volume_mms:
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

    if "coronal" in view_volume_mms:
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
    workspaces under the model stem. v12.2.0 rejects multiple model entries and never
    combines outputs from more than one model.
    """
    if len(view_volumes_by_model) != 1:
        raise ValueError('v12.2.0_SLURM supports exactly one --model; multiple-model inference has been removed')

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
        workers=int(workers),
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
# Optional skeleton output + scan-based interpolation helpers
# --------------------------


def _skimage_skeletonize_bool(mask_bool: np.ndarray) -> np.ndarray:
    """Best-effort 3D Lee skeletonization across scikit-image versions."""
    arr = np.asarray(mask_bool, dtype=bool)
    if arr.size <= 0 or not np.any(arr):
        return np.zeros(arr.shape, dtype=bool)

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


def _draw_line_zyx(mask: np.ndarray, p0: Tuple[int, int, int], p1: Tuple[int, int, int]) -> None:
    z0, y0, x0 = (int(p0[0]), int(p0[1]), int(p0[2]))
    z1, y1, x1 = (int(p1[0]), int(p1[1]), int(p1[2]))
    steps = max(abs(z1 - z0), abs(y1 - y0), abs(x1 - x0), 1)
    for i in range(int(steps) + 1):
        a = float(i) / float(steps)
        z = int(round((1.0 - a) * z0 + a * z1))
        y = int(round((1.0 - a) * y0 + a * y1))
        x = int(round((1.0 - a) * x0 + a * x1))
        if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            mask[z, y, x] = True


def _slice_component_anchors(mask2d: np.ndarray) -> List[Tuple[int, int]]:
    mask_u8 = np.ascontiguousarray(np.asarray(mask2d, dtype=np.uint8))
    if mask_u8.size == 0 or not np.any(mask_u8):
        return []
    num_labels, labels2d = cv2.connectedComponents(mask_u8, connectivity=8, ltype=cv2.CV_32S)
    anchors: List[Tuple[int, int]] = []
    for lbl in range(1, int(num_labels)):
        comp = labels2d == int(lbl)
        if not np.any(comp):
            continue
        comp_u8 = comp.astype(np.uint8, copy=False)
        dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        max_val = float(np.max(dist)) if dist.size else 0.0
        if max_val > 0.0:
            pts = np.argwhere(dist >= (max_val - 1e-6))
        else:
            pts = np.argwhere(comp)
        if pts.size <= 0:
            continue
        anchor = np.mean(pts, axis=0)
        anchors.append((int(round(float(anchor[0]))), int(round(float(anchor[1])))))
    return anchors


def _fallback_slice_anchor_centerline(comp_bool: np.ndarray) -> np.ndarray:
    """Build a smooth-ish centerline graph from per-slice distance-ridge anchors.

    This fallback prevents pathological empty or surface-like skeleton outputs from being
    written as porous blobs.  It is intentionally conservative: every 2D component in a
    slice contributes one distance-transform anchor, and adjacent-slice anchors are linked
    in both nearest-neighbor directions so simple bifurcations remain connected.
    """
    comp = np.asarray(comp_bool, dtype=bool)
    out = np.zeros(comp.shape, dtype=bool)
    prev: List[Tuple[int, int, int]] = []
    for z in range(int(comp.shape[0])):
        anchors2d = _slice_component_anchors(comp[int(z)])
        curr = [(int(z), int(y), int(x)) for y, x in anchors2d]
        for p in curr:
            out[p] = True
        if prev and curr:
            used_pairs: set[Tuple[int, int]] = set()
            for ci, cp in enumerate(curr):
                pi = min(range(len(prev)), key=lambda j: (prev[j][1] - cp[1]) ** 2 + (prev[j][2] - cp[2]) ** 2)
                used_pairs.add((int(pi), int(ci)))
            for pi, pp in enumerate(prev):
                ci = min(range(len(curr)), key=lambda j: (curr[j][1] - pp[1]) ** 2 + (curr[j][2] - pp[2]) ** 2)
                used_pairs.add((int(pi), int(ci)))
            for pi, ci in used_pairs:
                _draw_line_zyx(out, prev[int(pi)], curr[int(ci)])
        prev = curr if curr else prev
    return out


def _skeleton_max_density_fraction() -> float:
    raw = _env_float('YOLO_TTA_SKELETON_MAX_DENSITY_FRACTION', 0.18)
    return max(0.0, float(raw))


def centerline_skeletonize_component(comp_bool: np.ndarray) -> np.ndarray:
    comp = np.asarray(comp_bool, dtype=bool)
    if comp.size <= 0 or not np.any(comp):
        return np.zeros(comp.shape, dtype=bool)
    if _env_flag('YOLO_TTA_SKELETON_FORCE_SLICE_ANCHOR_CENTERLINE', False):
        return _fallback_slice_anchor_centerline(comp)

    skel = _skimage_skeletonize_bool(comp)
    skel_voxels = int(np.count_nonzero(skel))
    comp_voxels = max(1, int(np.count_nonzero(comp)))
    max_density = _skeleton_max_density_fraction()
    if skel_voxels <= 0 or (max_density > 0.0 and float(skel_voxels) / float(comp_voxels) > float(max_density)):
        fallback = _fallback_slice_anchor_centerline(comp)
        if np.any(fallback):
            return fallback
    return skel




def _skeleton_fill_2d_holes_enabled() -> bool:
    return _env_flag('YOLO_TTA_SKELETON_FILL_2D_HOLES', True)


def _skeleton_component_fill_3d_holes_enabled() -> bool:
    return _env_flag('YOLO_TTA_SKELETON_FILL_COMPONENT_3D_HOLES', False)


def _skeleton_chunk_min_voxels() -> int:
    return max(0, _env_int('YOLO_TTA_SKELETON_CHUNK_MIN_VOXELS', 256 * 1024 * 1024))


def _skeleton_chunk_depth() -> int:
    return max(16, _env_int('YOLO_TTA_SKELETON_CHUNK_DEPTH', 384))


def _skeleton_chunk_overlap() -> int:
    return max(4, _env_int('YOLO_TTA_SKELETON_CHUNK_OVERLAP', 32))


def _skeleton_chunk_max_voxels() -> int:
    return max(1024 * 1024, _env_int('YOLO_TTA_SKELETON_CHUNK_MAX_VOXELS', 96 * 1024 * 1024))


def _skeleton_worker_count(requested_workers: int, task_count: int) -> int:
    default_workers = max(1, min(32, _cpu_count()))
    workers = _env_int('YOLO_TTA_SKELETON_WORKERS', default_workers)
    if workers <= 0:
        workers = int(requested_workers)
    return choose_slice_parallel_workers(max(1, int(workers)), max(1, int(task_count)))


@dataclass(frozen=True)
class SkeletonChunkTask:
    label: int
    read_slice: Tuple[slice, slice, slice]
    core_slice: Tuple[slice, slice, slice]


def _pad_slice_for_shape(sl: slice, dim: int, pad_before: int, pad_after: int) -> slice:
    start = 0 if sl.start is None else int(sl.start)
    stop = int(dim) if sl.stop is None else int(sl.stop)
    return slice(max(0, start - int(pad_before)), min(int(dim), stop + int(pad_after)))


def prepare_skeleton_input_volume(
    mask_u8: np.ndarray,
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    """Prepare a cleaner binary input for centerline extraction.

    Small 2D pores in the final mask can make a thinning algorithm return porous
    skeleton islands instead of a centerline tree.  The default preconditioner fills
    per-slice enclosed holes in parallel, leaving 3D topology otherwise unchanged.
    """
    src = np.asarray(mask_u8, dtype=np.uint8)
    if not _skeleton_fill_2d_holes_enabled():
        return mask_u8

    out = allocate_workspace_array(
        shape=tuple(int(x) for x in src.shape),
        dtype=np.uint8,
        path=out_path,
        desc='Skeleton preconditioned mask workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    total = int(src.shape[0])

    def _fill_slice(z: int) -> None:
        sl = np.asarray(src[int(z)], dtype=bool)
        if np.any(sl):
            out[int(z), :, :] = _fill_holes_2d(sl).astype(np.uint8, copy=False)
        else:
            out[int(z), :, :].fill(np.uint8(0))

    parallel_for_indices_chunked(
        total,
        _fill_slice,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc='Skeleton precondition: fill 2D pores',
        show_progress=True,
        target_chunks_per_worker=2,
    )
    flush_array(out)
    return out


def _build_skeleton_chunk_tasks(
    component_slices: Sequence[Optional[Tuple[slice, slice, slice]]],
    volume_shape: Tuple[int, int, int],
) -> Tuple[List[SkeletonChunkTask], Dict[str, int]]:
    z_dim, h_dim, w_dim = (int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2]))
    tasks: List[SkeletonChunkTask] = []
    chunked_components = 0
    full_components = 0
    chunk_min_voxels = _skeleton_chunk_min_voxels()
    chunk_depth = _skeleton_chunk_depth()
    overlap = _skeleton_chunk_overlap()
    max_task_voxels = _skeleton_chunk_max_voxels()

    def _axis_chunks(start: int, stop: int, step: int) -> Iterator[Tuple[int, int]]:
        step_i = max(1, int(step))
        cur = int(start)
        while cur < int(stop):
            nxt = min(int(stop), cur + step_i)
            yield int(cur), int(nxt)
            cur = int(nxt)

    def _append_task(label_i: int, core: Tuple[slice, slice, slice], read_pad: int) -> None:
        cz, cy, cx = core
        read = (
            _pad_slice_for_shape(cz, z_dim, read_pad, read_pad),
            _pad_slice_for_shape(cy, h_dim, read_pad, read_pad),
            _pad_slice_for_shape(cx, w_dim, read_pad, read_pad),
        )
        tasks.append(SkeletonChunkTask(label=int(label_i), read_slice=read, core_slice=core))

    for label, sl in enumerate(component_slices, start=1):
        if sl is None:
            continue
        z_sl, y_sl, x_sl = sl
        z0, z1 = int(z_sl.start or 0), int(z_sl.stop or 0)
        y0, y1 = int(y_sl.start or 0), int(y_sl.stop or 0)
        x0, x1 = int(x_sl.start or 0), int(x_sl.stop or 0)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue
        bbox_voxels = int(z1 - z0) * int(y1 - y0) * int(x1 - x0)

        if chunk_min_voxels > 0 and bbox_voxels >= chunk_min_voxels and ((z1 - z0) > chunk_depth or bbox_voxels > max_task_voxels):
            chunked_components += 1
            core_z_step = min(max(1, chunk_depth), max(1, z1 - z0))
            # Choose XY tile sizes that keep dense label/bool/skel temporaries bounded while
            # retaining overlap to reduce boundary discontinuities.
            area_budget = max(64 * 64, int(max_task_voxels // max(1, core_z_step + 2 * overlap)))
            side = max(64, int(math.sqrt(float(area_budget))))
            x_step = max(64, min(x1 - x0, side))
            y_step = max(64, min(y1 - y0, max(64, int(area_budget // max(1, x_step)))))
            for core_z0 in range(z0, z1, chunk_depth):
                core_z1 = min(z1, core_z0 + chunk_depth)
                for core_y0, core_y1 in _axis_chunks(y0, y1, y_step):
                    for core_x0, core_x1 in _axis_chunks(x0, x1, x_step):
                        _append_task(
                            int(label),
                            (slice(int(core_z0), int(core_z1)), slice(int(core_y0), int(core_y1)), slice(int(core_x0), int(core_x1))),
                            int(overlap),
                        )
        else:
            full_components += 1
            _append_task(int(label), (slice(z0, z1), slice(y0, y1), slice(x0, x1)), 2)

    return tasks, {
        'tasks': int(len(tasks)),
        'full_components': int(full_components),
        'chunked_components': int(chunked_components),
    }


def compute_centerline_skeleton_into_workspace(
    mask_u8: np.ndarray,
    out_mm: np.ndarray,
    *,
    temp_dir: Path,
    workers: int = 1,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> Dict[str, int]:
    """Write a 1-voxel centerline skeleton to ``out_mm``.

    The work is decomposed into 3D connected components and, for very large
    components, overlapping z chunks.  That keeps the optional skeleton output from
    becoming a single monolithic thinning call while preserving independent object
    topology as much as possible.
    """
    out_mm[:, :, :] = np.uint8(0)
    flush_array(out_mm)

    skeleton_dir = Path(temp_dir) / 'skeleton'
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    prepped = prepare_skeleton_input_volume(
        mask_u8,
        skeleton_dir / 'preconditioned_mask.u8.dat',
        workers=int(workers),
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    labels_mm, num_objects, label_paths = label_foreground_volume_streaming(
        prepped,
        skeleton_dir / 'component_labels',
        keep_temp=bool(keep_temp),
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        wrap_axis=False,
        workers=int(workers),
    )
    if int(num_objects) <= 0:
        close_memmap_array(labels_mm)
        if prepped is not mask_u8:
            close_memmap_array(prepped)
        if not bool(keep_temp):
            for lp in label_paths:
                try:
                    lp.unlink(missing_ok=True)
                except Exception:
                    pass
        return {'objects': 0, 'tasks': 0, 'chunked_components': 0, 'skeleton_voxels': 0}

    print(f'Skeleton labeling found {int(num_objects)} object component(s); building component bounding boxes.')
    component_slices = ndi.find_objects(labels_mm, max_label=int(num_objects))
    tasks, task_stats = _build_skeleton_chunk_tasks(component_slices, tuple(int(x) for x in labels_mm.shape))
    worker_count = _skeleton_worker_count(int(workers), max(1, len(tasks)))
    print(
        'Skeleton centerline tasks: '
        f'tasks={int(task_stats["tasks"])}, full_components={int(task_stats["full_components"])}, '
        f'chunked_components={int(task_stats["chunked_components"])}, workers={int(worker_count)}'
    )
    skeleton_voxel_counts = np.zeros((len(tasks),), dtype=np.int64)

    def _process_task(task_idx: int) -> None:
        task = tasks[int(task_idx)]
        z_sl, y_sl, x_sl = task.read_slice
        labels_block = np.asarray(labels_mm[z_sl, y_sl, x_sl])
        comp = labels_block == int(task.label)
        if not np.any(comp):
            return
        if _skeleton_component_fill_3d_holes_enabled():
            comp = np.asarray(ndi.binary_fill_holes(comp), dtype=bool)
        skel = centerline_skeletonize_component(comp)
        if not np.any(skel):
            return

        core_z, core_y, core_x = task.core_slice
        read_z0, read_y0, read_x0 = int(z_sl.start or 0), int(y_sl.start or 0), int(x_sl.start or 0)
        z_rel0 = max(0, int(core_z.start or 0) - read_z0)
        z_rel1 = min(int(skel.shape[0]), int(core_z.stop or 0) - read_z0)
        y_rel0 = max(0, int(core_y.start or 0) - read_y0)
        y_rel1 = min(int(skel.shape[1]), int(core_y.stop or 0) - read_y0)
        x_rel0 = max(0, int(core_x.start or 0) - read_x0)
        x_rel1 = min(int(skel.shape[2]), int(core_x.stop or 0) - read_x0)
        if z_rel1 <= z_rel0 or y_rel1 <= y_rel0 or x_rel1 <= x_rel0:
            return
        skel_core = skel[z_rel0:z_rel1, y_rel0:y_rel1, x_rel0:x_rel1]
        if not np.any(skel_core):
            return

        dst = out_mm[core_z, core_y, core_x]
        dst[skel_core] = np.uint8(1)
        skeleton_voxel_counts[int(task_idx)] = np.int64(np.count_nonzero(skel_core))

    parallel_for_indices_chunked(
        len(tasks),
        _process_task,
        max_workers=worker_count,
        desc='Skeletonize components/chunks',
        show_progress=True,
        target_chunks_per_worker=1,
    )
    flush_array(out_mm)

    skeleton_voxels = int(np.sum(skeleton_voxel_counts, dtype=np.int64))
    close_memmap_array(labels_mm)
    if prepped is not mask_u8:
        close_memmap_array(prepped)
    if not bool(keep_temp):
        for lp in label_paths:
            try:
                lp.unlink(missing_ok=True)
            except Exception:
                pass
    return {
        'objects': int(num_objects),
        'tasks': int(len(tasks)),
        'chunked_components': int(task_stats['chunked_components']),
        'skeleton_voxels': int(skeleton_voxels),
    }


def compute_skeleton_volume_to_workspace(
    mask_u8: np.ndarray,
    out_path: Path,
    *,
    temp_dir: Path,
    workers: int = 1,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    """Compute an optional 3D centerline skeleton as a postprocessing output layer.

    This is intentionally independent from interpolation. Interpolation endpoint seeds are
    discovered by per-slice connected-component scanning; this function runs only when
    --save_skeleton is requested.
    """
    shape = tuple(int(x) for x in np.asarray(mask_u8).shape)
    out_mm = allocate_workspace_array(
        shape=shape,
        dtype=np.uint8,
        path=out_path,
        desc='Skeleton output workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    print('Computing optional 3D centerline skeleton output; this is independent of interpolation endpoint discovery.')
    stats = compute_centerline_skeleton_into_workspace(
        mask_u8,
        out_mm,
        temp_dir=Path(temp_dir),
        workers=int(workers),
        keep_temp=bool(keep_temp),
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    print(
        'Skeleton output stats: '
        f'objects={int(stats.get("objects", 0))}, tasks={int(stats.get("tasks", 0))}, '
        f'chunked_components={int(stats.get("chunked_components", 0))}, '
        f'skeleton_voxels={int(stats.get("skeleton_voxels", 0))}'
    )
    flush_array(out_mm)
    return out_mm


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


INTERPOLATION_LOCAL_SDF_PAD = 2


@dataclass(frozen=True)
class SliceComponentRecord:
    z: int
    label: int
    component_index: int
    bbox: Tuple[int, int, int, int]  # y0, x0, y1, x1 in full-slice coordinates
    anchor: Tuple[int, int]          # centroid-nearest component pixel in full-slice coordinates
    area: int
    mask_crop: np.ndarray            # bool crop in bbox coordinates


@dataclass(frozen=True)
class CroppedProjectionSDF:
    origin_y: int
    origin_x: int
    sdf: np.ndarray


@dataclass
class SliceComponentTable:
    z: int
    shape: Tuple[int, int]
    components: List[SliceComponentRecord]
    by_label: Dict[int, List[SliceComponentRecord]]

    def find_record_for_point(
        self,
        label: int,
        point_yx: Tuple[int, int],
    ) -> Tuple[Optional[SliceComponentRecord], Optional[Tuple[int, int]]]:
        records = self.by_label.get(int(label), [])
        if not records:
            return None, None

        h, w = int(self.shape[0]), int(self.shape[1])
        py = int(np.clip(int(point_yx[0]), 0, max(0, h - 1)))
        px = int(np.clip(int(point_yx[1]), 0, max(0, w - 1)))

        for record in records:
            y0, x0, y1, x1 = record.bbox
            if y0 <= py < y1 and x0 <= px < x1 and bool(record.mask_crop[py - y0, px - x0]):
                return record, (int(py), int(px))

        best_record: Optional[SliceComponentRecord] = None
        best_point: Optional[Tuple[int, int]] = None
        best_d2: Optional[int] = None
        for record in records:
            nearest = _nearest_point_in_component_record(record, (py, px))
            if nearest is None:
                continue
            d2 = int((int(nearest[0]) - py) ** 2 + (int(nearest[1]) - px) ** 2)
            if best_d2 is None or d2 < best_d2:
                best_d2 = int(d2)
                best_record = record
                best_point = nearest
        return best_record, best_point

    def find_branch_continuation(
        self,
        label: int,
        prev_record: SliceComponentRecord,
        prev_anchor: Tuple[int, int],
    ) -> Tuple[Optional[SliceComponentRecord], Optional[Tuple[int, int]]]:
        records = self.by_label.get(int(label), [])
        if not records:
            return None, None
        if len(records) == 1:
            record = records[0]
            return record, _nearest_point_in_component_record(record, prev_anchor)

        best_record: Optional[SliceComponentRecord] = None
        best_anchor: Optional[Tuple[int, int]] = None
        best_score: Optional[Tuple[int, int, int]] = None
        for record in records:
            anchor = _nearest_point_in_component_record(record, prev_anchor)
            if anchor is None:
                continue
            overlap = _component_record_dilated_overlap_count(prev_record, record)
            d2 = int((int(anchor[0]) - int(prev_anchor[0])) ** 2 + (int(anchor[1]) - int(prev_anchor[1])) ** 2)
            score = (int(overlap), -int(d2), int(record.area))
            if best_score is None or score > best_score:
                best_score = score
                best_record = record
                best_anchor = anchor
        return best_record, best_anchor


class SliceComponentTableCache:
    """Lazy per-slice component table and local projection-SDF cache for interpolation.

    Tables are keyed by slice and store cropped masks/bboxes for each 2D component of
    each 3D object label.  Projection candidate search then operates only on the source
    component's bbox expanded by the maximum possible search-angle growth, avoiding
    full-slice SDFs and full-slice boolean projection scans per seed.
    """

    def __init__(self, labels_real: np.ndarray) -> None:
        self.labels_real = labels_real
        self.z_dim = int(labels_real.shape[0])
        self.shape_yx = (int(labels_real.shape[1]), int(labels_real.shape[2]))
        self._tables: Dict[int, SliceComponentTable] = {}
        self._projection_sdfs: Dict[Tuple[int, int, int, int, float, int], CroppedProjectionSDF] = {}
        self._table_locks = [threading.Lock() for _ in range(max(1, self.z_dim))]
        self._sdf_lock = threading.Lock()

    def get(self, z: int) -> SliceComponentTable:
        z_i = int(z)
        if z_i < 0 or z_i >= self.z_dim:
            raise IndexError(z_i)
        cached = self._tables.get(z_i)
        if cached is not None:
            return cached
        with self._table_locks[z_i]:
            cached = self._tables.get(z_i)
            if cached is None:
                cached = _build_slice_component_table(np.asarray(self.labels_real[z_i]), z_i)
                self._tables[z_i] = cached
            return cached

    def prebuild(self, *, workers: int = 1, desc: str = 'Interpolation: per-slice component tables') -> None:
        total = int(self.z_dim)
        if total <= 0:
            return
        worker_count = choose_slice_parallel_workers(int(workers), total)

        def _build(z: int) -> int:
            return int(len(self.get(int(z)).components))

        if worker_count <= 1:
            for z in tqdm(range(total), desc=desc):
                _build(int(z))
            return

        pending = max(worker_count, worker_count * 8)
        for _component_count in tqdm(
            parallel_map_unordered(_build, range(total), max_workers=worker_count, max_pending=pending),
            total=total,
            desc=desc,
        ):
            pass

    def find_record_for_point(
        self,
        z: int,
        label: int,
        point_yx: Tuple[int, int],
    ) -> Tuple[Optional[SliceComponentRecord], Optional[Tuple[int, int]]]:
        return self.get(int(z)).find_record_for_point(int(label), point_yx)

    def get_projection_sdf(
        self,
        record: SliceComponentRecord,
        max_slice_distance: int,
        search_angle_deg: float,
    ) -> CroppedProjectionSDF:
        slope = math.tan(math.radians(float(search_angle_deg)))
        growth = max(0.0, float(slope)) * float(max(0, int(max_slice_distance)))
        pad = max(int(INTERPOLATION_LOCAL_SDF_PAD), int(math.ceil(growth)) + int(INTERPOLATION_LOCAL_SDF_PAD))
        key = (
            int(record.z),
            int(record.label),
            int(record.component_index),
            int(max_slice_distance),
            round(float(search_angle_deg), 6),
            int(pad),
        )
        with self._sdf_lock:
            cached = self._projection_sdfs.get(key)
        if cached is not None:
            return cached

        h, w = self.shape_yx
        y0, x0, y1, x1 = record.bbox
        crop_y0 = max(0, int(y0) - int(pad))
        crop_x0 = max(0, int(x0) - int(pad))
        crop_y1 = min(int(h), int(y1) + int(pad))
        crop_x1 = min(int(w), int(x1) + int(pad))
        source_crop = np.zeros((int(crop_y1 - crop_y0), int(crop_x1 - crop_x0)), dtype=bool)
        dy0 = int(y0) - int(crop_y0)
        dx0 = int(x0) - int(crop_x0)
        source_crop[dy0:dy0 + int(record.mask_crop.shape[0]), dx0:dx0 + int(record.mask_crop.shape[1])] = record.mask_crop
        cropped = CroppedProjectionSDF(
            origin_y=int(crop_y0),
            origin_x=int(crop_x0),
            sdf=np.ascontiguousarray(_signed_distance_2d(source_crop)),
        )
        with self._sdf_lock:
            existing = self._projection_sdfs.setdefault(key, cropped)
        return existing


def _nearest_point_in_component_record(record: SliceComponentRecord, ref_yx: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    ys, xs = np.nonzero(record.mask_crop)
    if ys.size == 0:
        return None
    y0, x0, _y1, _x1 = record.bbox
    gy = ys.astype(np.int64, copy=False) + int(y0)
    gx = xs.astype(np.int64, copy=False) + int(x0)
    d2 = (gy - int(ref_yx[0])) ** 2 + (gx - int(ref_yx[1])) ** 2
    idx = int(np.argmin(d2))
    return int(gy[idx]), int(gx[idx])


def _component_record_dilated_overlap_count(prev_record: SliceComponentRecord, candidate_record: SliceComponentRecord) -> int:
    py0, px0, py1, px1 = prev_record.bbox
    cy0, cx0, cy1, cx1 = candidate_record.bbox
    # The previous component is dilated by one pixel for branch following, so its
    # effective bbox expands by one pixel in every direction.
    iy0 = max(int(py0) - 1, int(cy0))
    ix0 = max(int(px0) - 1, int(cx0))
    iy1 = min(int(py1) + 1, int(cy1))
    ix1 = min(int(px1) + 1, int(cx1))
    if iy0 >= iy1 or ix0 >= ix1:
        return 0

    padded_prev = np.pad(np.asarray(prev_record.mask_crop, dtype=np.uint8), 1, mode='constant', constant_values=0)
    dilated_prev = cv2.dilate(padded_prev, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool, copy=False)
    prev_origin_y = int(py0) - 1
    prev_origin_x = int(px0) - 1
    prev_block = dilated_prev[iy0 - prev_origin_y:iy1 - prev_origin_y, ix0 - prev_origin_x:ix1 - prev_origin_x]
    cand_block = candidate_record.mask_crop[iy0 - int(cy0):iy1 - int(cy0), ix0 - int(cx0):ix1 - int(cx0)]
    return int(np.count_nonzero(prev_block & cand_block))


def _component_records_directly_overlap(a: SliceComponentRecord, b: SliceComponentRecord) -> bool:
    ay0, ax0, ay1, ax1 = a.bbox
    by0, bx0, by1, bx1 = b.bbox
    iy0 = max(int(ay0), int(by0))
    ix0 = max(int(ax0), int(bx0))
    iy1 = min(int(ay1), int(by1))
    ix1 = min(int(ax1), int(bx1))
    if iy0 >= iy1 or ix0 >= ix1:
        return False
    a_block = a.mask_crop[iy0 - int(ay0):iy1 - int(ay0), ix0 - int(ax0):ix1 - int(ax0)]
    b_block = b.mask_crop[iy0 - int(by0):iy1 - int(by0), ix0 - int(bx0):ix1 - int(bx0)]
    return bool(np.any(a_block & b_block))


def _build_slice_component_table(labels2d: np.ndarray, z: int) -> SliceComponentTable:
    labels_arr = np.asarray(labels2d)
    h, w = (int(labels_arr.shape[0]), int(labels_arr.shape[1]))
    components: List[SliceComponentRecord] = []
    by_label: Dict[int, List[SliceComponentRecord]] = {}

    ys_all, xs_all = np.nonzero(labels_arr)
    if ys_all.size <= 0:
        return SliceComponentTable(z=int(z), shape=(h, w), components=components, by_label=by_label)

    # Build each label's 2D components from sparse foreground coordinates instead of scanning the
    # full 9 MP slice once per object label.  This keeps endpoint-table construction local to the
    # actual foreground footprint while preserving the rule that different 3D labels remain separate
    # even when they touch in this slice.
    labels_fg = labels_arr[ys_all, xs_all].astype(np.int64, copy=False)
    order = np.argsort(labels_fg, kind='mergesort')
    labels_sorted = labels_fg[order]
    ys_sorted = ys_all[order]
    xs_sorted = xs_all[order]
    boundaries = np.flatnonzero(labels_sorted[1:] != labels_sorted[:-1]) + 1
    starts = np.concatenate(([0], boundaries)).astype(np.int64, copy=False)
    stops = np.concatenate((boundaries, [labels_sorted.size])).astype(np.int64, copy=False)

    for start_i, stop_i in zip(starts.tolist(), stops.tolist()):
        if int(stop_i) <= int(start_i):
            continue
        label_i = int(labels_sorted[int(start_i)])
        if label_i <= 0:
            continue
        ys_g = ys_sorted[int(start_i):int(stop_i)].astype(np.int64, copy=False)
        xs_g = xs_sorted[int(start_i):int(stop_i)].astype(np.int64, copy=False)
        if ys_g.size <= 0:
            continue

        y0 = int(np.min(ys_g))
        y1 = int(np.max(ys_g)) + 1
        x0 = int(np.min(xs_g))
        x1 = int(np.max(xs_g)) + 1
        crop_h = int(y1 - y0)
        crop_w = int(x1 - x0)
        if crop_h <= 0 or crop_w <= 0:
            continue

        mask_u8 = np.zeros((crop_h, crop_w), dtype=np.uint8)
        mask_u8[ys_g - int(y0), xs_g - int(x0)] = np.uint8(1)
        num_cc, cc, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8, ltype=cv2.CV_32S)
        for local_lbl in range(1, int(num_cc)):
            area = int(stats[int(local_lbl), cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[int(local_lbl), cv2.CC_STAT_LEFT])
            y = int(stats[int(local_lbl), cv2.CC_STAT_TOP])
            width = int(stats[int(local_lbl), cv2.CC_STAT_WIDTH])
            height = int(stats[int(local_lbl), cv2.CC_STAT_HEIGHT])
            if width <= 0 or height <= 0:
                continue
            comp_crop = np.ascontiguousarray(cc[y:y + height, x:x + width] == int(local_lbl))
            ys, xs = np.nonzero(comp_crop)
            if ys.size <= 0:
                continue
            centroid_x = float(centroids[int(local_lbl), 0]) - float(x)
            centroid_y = float(centroids[int(local_lbl), 1]) - float(y)
            d2 = (ys.astype(np.float32, copy=False) - centroid_y) ** 2 + (xs.astype(np.float32, copy=False) - centroid_x) ** 2
            anchor_idx = int(np.argmin(d2))
            record = SliceComponentRecord(
                z=int(z),
                label=int(label_i),
                component_index=int(len(components) + 1),
                bbox=(int(y0 + y), int(x0 + x), int(y0 + y + height), int(x0 + x + width)),
                anchor=(int(y0 + y + int(ys[anchor_idx])), int(x0 + x + int(xs[anchor_idx]))),
                area=int(area),
                mask_crop=comp_crop,
            )
            components.append(record)
            by_label.setdefault(int(label_i), []).append(record)
    return SliceComponentTable(z=int(z), shape=(h, w), components=components, by_label=by_label)


def _component_record_to_local_canvas(record: SliceComponentRecord, anchor_yx: Tuple[int, int], half_width: int) -> np.ndarray:
    size = int(2 * int(half_width) + 1)
    local = np.zeros((size, size), dtype=bool)
    ys, xs = np.nonzero(record.mask_crop)
    if ys.size == 0:
        return local
    y0, x0, _y1, _x1 = record.bbox
    yy = ys.astype(np.int64, copy=False) + int(y0) - int(anchor_yx[0]) + int(half_width)
    xx = xs.astype(np.int64, copy=False) + int(x0) - int(anchor_yx[1]) + int(half_width)
    valid = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
    local[yy[valid], xx[valid]] = True
    return local


def _local_half_width_for_component_records(
    source_record: SliceComponentRecord,
    source_anchor: Tuple[int, int],
    target_record: SliceComponentRecord,
    target_anchor: Tuple[int, int],
) -> int:
    max_extent = 0.0
    for record, anchor in ((source_record, source_anchor), (target_record, target_anchor)):
        ys, xs = np.nonzero(record.mask_crop)
        if ys.size == 0:
            continue
        y0, x0, _y1, _x1 = record.bbox
        gy = ys.astype(np.int64, copy=False) + int(y0)
        gx = xs.astype(np.int64, copy=False) + int(x0)
        max_extent = max(
            max_extent,
            float(np.max(np.abs(gy - int(anchor[0])))),
            float(np.max(np.abs(gx - int(anchor[1])))),
        )

    max_extent = max(
        max_extent,
        float(abs(int(target_anchor[0]) - int(source_anchor[0]))),
        float(abs(int(target_anchor[1]) - int(source_anchor[1]))),
    )
    return max(4, int(math.ceil(max_extent)) + 4)


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



_NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED = False


def compiled_interpolation_kernels_enabled() -> bool:
    return bool(_numba is not None and _env_flag('YOLO_TTA_INTERPOLATION_COMPILED_KERNELS', True))


def interpolation_projection_numba_max_tracked() -> int:
    return max(8, _env_int('YOLO_TTA_INTERPOLATION_NUMBA_MAX_TRACKED_CANDIDATES', 64))


def interpolation_compiled_kernels_status() -> str:
    if _numba is None:
        return f'unavailable ({_NUMBA_IMPORT_ERROR})'
    if not _env_flag('YOLO_TTA_INTERPOLATION_COMPILED_KERNELS', True):
        return 'disabled by YOLO_TTA_INTERPOLATION_COMPILED_KERNELS=0'
    if _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED:
        return 'disabled after runtime compilation/execution failure'
    return 'enabled: numba no-GIL projection-candidate kernel'


def interpolation_planning_backend_name() -> str:
    base = 'cached_per_slice_component_tables_local_sdf_unordered'
    if compiled_interpolation_kernels_enabled() and not _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED:
        base += '+numba_nogil_projection_candidates'
    else:
        base += '+python_projection_candidates'
    if _INTERPOLATION_PROCESS_WORKER:
        base += '+process_isolated_pass'
    return base


if _numba is not None:
    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_find_projection_candidates_kernel(
        labels_real: np.ndarray,
        sdf: np.ndarray,
        crop_y0: int,
        crop_x0: int,
        s0: int,
        source_anchor_y: int,
        source_anchor_x: int,
        seed_label: int,
        direction_sign: int,
        max_steps: int,
        slope: float,
        max_candidates: int,
        wrap_axis: bool,
        search_angle_negative: bool,
        max_tracked: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        out_labels = np.zeros((max_tracked,), dtype=np.int64)
        out_slices = np.zeros((max_tracked,), dtype=np.int64)
        out_ys = np.zeros((max_tracked,), dtype=np.int64)
        out_xs = np.zeros((max_tracked,), dtype=np.int64)
        out_steps = np.zeros((max_tracked,), dtype=np.int64)
        out_d2 = np.zeros((max_tracked,), dtype=np.int64)

        step_labels = np.zeros((max_tracked,), dtype=np.int64)
        step_ys = np.zeros((max_tracked,), dtype=np.int64)
        step_xs = np.zeros((max_tracked,), dtype=np.int64)
        step_d2 = np.zeros((max_tracked,), dtype=np.int64)

        found_count = 0
        num_slices = labels_real.shape[0]
        sdf_h = sdf.shape[0]
        sdf_w = sdf.shape[1]
        overflow = 0

        for step in range(1, max_steps + 1):
            s = s0 + direction_sign * step
            if wrap_axis:
                s = s % num_slices
            else:
                if s < 0 or s >= num_slices:
                    break

            threshold = -slope * float(step)
            any_projection = False
            step_count = 0

            for yy in range(sdf_h):
                gy = crop_y0 + yy
                for xx in range(sdf_w):
                    if sdf[yy, xx] < threshold:
                        continue
                    any_projection = True
                    gx = crop_x0 + xx
                    target_label = int(labels_real[s, gy, gx])
                    if target_label <= 0 or target_label == seed_label:
                        continue

                    already_found = False
                    for prev_idx in range(found_count):
                        if out_labels[prev_idx] == target_label:
                            already_found = True
                            break
                    if already_found:
                        continue

                    step_idx = -1
                    for local_idx in range(step_count):
                        if step_labels[local_idx] == target_label:
                            step_idx = local_idx
                            break

                    dy = gy - source_anchor_y
                    dx = gx - source_anchor_x
                    d2 = dy * dy + dx * dx
                    if step_idx < 0:
                        if step_count >= max_tracked:
                            overflow = 1
                            return out_labels, out_slices, out_ys, out_xs, out_steps, out_d2, found_count, overflow
                        step_labels[step_count] = target_label
                        step_ys[step_count] = gy
                        step_xs[step_count] = gx
                        step_d2[step_count] = d2
                        step_count += 1
                    else:
                        if d2 < step_d2[step_idx]:
                            step_ys[step_idx] = gy
                            step_xs[step_idx] = gx
                            step_d2[step_idx] = d2

            if not any_projection:
                if search_angle_negative:
                    break
                continue

            for local_idx in range(step_count):
                if found_count >= max_tracked:
                    overflow = 1
                    return out_labels, out_slices, out_ys, out_xs, out_steps, out_d2, found_count, overflow
                out_labels[found_count] = step_labels[local_idx]
                out_slices[found_count] = s
                out_ys[found_count] = step_ys[local_idx]
                out_xs[found_count] = step_xs[local_idx]
                out_steps[found_count] = step
                out_d2[found_count] = step_d2[local_idx]
                found_count += 1

            if found_count >= max_candidates:
                break

        return out_labels, out_slices, out_ys, out_xs, out_steps, out_d2, found_count, overflow
else:
    _numba_find_projection_candidates_kernel = None


def _find_slice_projection_candidates_numba(
    labels_real: np.ndarray,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    max_candidates: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
) -> Optional[List[SliceProjectionCandidate]]:
    global _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED
    if (
        _numba_find_projection_candidates_kernel is None
        or not compiled_interpolation_kernels_enabled()
        or _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED
    ):
        return None
    if int(max_slice_distance) <= 0 or int(max_candidates) <= 0:
        return []

    s0, y0, x0 = seed.point
    num_slices = int(labels_real.shape[0])
    if num_slices <= 0:
        return []

    local_cache = component_cache if component_cache is not None else SliceComponentTableCache(labels_real)
    source_record, source_anchor = local_cache.find_record_for_point(int(s0), int(seed.label), (int(y0), int(x0)))
    if source_record is None or source_anchor is None or int(source_record.area) <= 0:
        return []

    max_steps = min(int(max_slice_distance), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(max_slice_distance)
    if int(max_steps) <= 0:
        return []

    cropped_sdf = local_cache.get_projection_sdf(
        source_record,
        max_slice_distance=int(max_steps),
        search_angle_deg=float(search_angle_deg),
    )
    sdf = np.ascontiguousarray(cropped_sdf.sdf, dtype=np.float32)
    crop_y0 = int(cropped_sdf.origin_y)
    crop_x0 = int(cropped_sdf.origin_x)
    slope = math.tan(math.radians(float(search_angle_deg)))
    max_tracked = max(int(max_candidates), int(interpolation_projection_numba_max_tracked()))

    try:
        labels_out, slices_out, ys_out, xs_out, steps_out, d2_out, count, overflow = _numba_find_projection_candidates_kernel(
            labels_real,
            sdf,
            int(crop_y0),
            int(crop_x0),
            int(s0),
            int(source_anchor[0]),
            int(source_anchor[1]),
            int(seed.label),
            int(seed.direction_sign),
            int(max_steps),
            float(slope),
            int(max_candidates),
            bool(wrap_axis),
            bool(float(search_angle_deg) < 0.0),
            int(max_tracked),
        )
    except Exception as exc:
        _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED = True
        print(f'Warning: Numba interpolation projection kernel failed ({exc}); using Python candidate search for remaining calls in this process.')
        return None

    if int(overflow) != 0:
        # Preserve exact semantics by falling back to the Python implementation when a single
        # projection step contains more distinct labels than the fixed-size compiled workspace.
        return None
    if int(count) <= 0:
        return []

    rows: List[Tuple[int, int, int, int, int, int]] = []
    for idx in range(int(count)):
        rows.append((
            int(steps_out[int(idx)]),
            int(d2_out[int(idx)]),
            int(labels_out[int(idx)]),
            int(slices_out[int(idx)]),
            int(ys_out[int(idx)]),
            int(xs_out[int(idx)]),
        ))
    rows.sort(key=lambda item: (int(item[0]), int(item[1]), int(item[2])))
    out: List[SliceProjectionCandidate] = []
    for step, _d2, target_label, target_slice, target_y, target_x in rows[: int(max_candidates)]:
        out.append(SliceProjectionCandidate(
            source_label=int(seed.label),
            target_label=int(target_label),
            source_point=(int(s0), int(y0), int(x0)),
            target_point=(int(target_slice), int(target_y), int(target_x)),
            slice_distance=int(step),
        ))
    return out


def _find_slice_projection_candidates_python(
    labels_real: np.ndarray,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    max_candidates: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
) -> List[SliceProjectionCandidate]:
    if int(max_slice_distance) <= 0 or int(max_candidates) <= 0:
        return []

    s0, y0, x0 = seed.point
    num_slices = int(labels_real.shape[0])
    if num_slices <= 0:
        return []

    local_cache = component_cache if component_cache is not None else SliceComponentTableCache(labels_real)
    source_record, source_anchor = local_cache.find_record_for_point(int(s0), int(seed.label), (int(y0), int(x0)))
    if source_record is None or source_anchor is None or int(source_record.area) <= 0:
        return []

    max_steps = min(int(max_slice_distance), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(max_slice_distance)
    if int(max_steps) <= 0:
        return []

    cropped_sdf = local_cache.get_projection_sdf(
        source_record,
        max_slice_distance=int(max_steps),
        search_angle_deg=float(search_angle_deg),
    )
    sdf = np.asarray(cropped_sdf.sdf, dtype=np.float32)
    crop_y0 = int(cropped_sdf.origin_y)
    crop_x0 = int(cropped_sdf.origin_x)
    crop_y1 = int(crop_y0 + int(sdf.shape[0]))
    crop_x1 = int(crop_x0 + int(sdf.shape[1]))

    slope = math.tan(math.radians(float(search_angle_deg)))
    found: Dict[int, SliceProjectionCandidate] = {}

    for step in range(1, int(max_steps) + 1):
        s = int(s0 + int(seed.direction_sign) * step)
        if bool(wrap_axis):
            s = int(s % int(num_slices))
        elif s < 0 or s >= num_slices:
            break

        threshold = -float(slope) * float(step)
        projection = sdf >= threshold
        if not np.any(projection):
            if float(search_angle_deg) < 0.0:
                break
            continue

        labels_crop = np.asarray(labels_real[int(s), crop_y0:crop_y1, crop_x0:crop_x1])
        overlap = projection & (labels_crop > 0) & (labels_crop != int(seed.label))
        if not np.any(overlap):
            continue

        ys_local, xs_local = np.nonzero(overlap)
        lbls = labels_crop[ys_local, xs_local].astype(np.int64, copy=False)
        for target_label in np.unique(lbls):
            target_label_i = int(target_label)
            if target_label_i <= 0 or target_label_i == int(seed.label) or target_label_i in found:
                continue
            use = lbls == target_label_i
            ys_t = ys_local[use]
            xs_t = xs_local[use]
            if ys_t.size == 0:
                continue
            ys_global = ys_t.astype(np.int64, copy=False) + int(crop_y0)
            xs_global = xs_t.astype(np.int64, copy=False) + int(crop_x0)
            d2 = (ys_global - int(source_anchor[0])) ** 2 + (xs_global - int(source_anchor[1])) ** 2
            idx = int(np.argmin(d2))
            found[target_label_i] = SliceProjectionCandidate(
                source_label=int(seed.label),
                target_label=target_label_i,
                source_point=(int(s0), int(y0), int(x0)),
                target_point=(int(s), int(ys_global[idx]), int(xs_global[idx])),
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



def _find_slice_projection_candidates(
    labels_real: np.ndarray,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    max_candidates: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
) -> List[SliceProjectionCandidate]:
    fast_candidates = _find_slice_projection_candidates_numba(
        labels_real=labels_real,
        seed=seed,
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        max_candidates=int(max_candidates),
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
    )
    if fast_candidates is not None:
        return fast_candidates
    return _find_slice_projection_candidates_python(
        labels_real=labels_real,
        seed=seed,
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        max_candidates=int(max_candidates),
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
    )

def _collect_walkback_source_points(
    labels_real: np.ndarray,
    label: int,
    start_point: Tuple[int, int, int],
    direction_sign: int,
    walk_back: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
) -> List[Tuple[int, int, int]]:
    if int(walk_back) <= 0:
        return []

    s0, y0, x0 = start_point
    num_slices = int(labels_real.shape[0])

    if component_cache is not None:
        current_record, current_anchor = component_cache.find_record_for_point(int(s0), int(label), (int(y0), int(x0)))
        if current_record is None or current_anchor is None:
            return []
        out: List[Tuple[int, int, int]] = []
        current_slice = int(s0)
        max_walk = min(int(walk_back), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(walk_back)
        visited_slices = {int(current_slice)}
        for _ in range(int(max_walk)):
            next_slice = int(current_slice - int(direction_sign))
            if bool(wrap_axis):
                next_slice = int(next_slice % int(num_slices))
                if next_slice in visited_slices:
                    break
            elif next_slice < 0 or next_slice >= num_slices:
                break

            next_record, next_anchor = component_cache.get(next_slice).find_branch_continuation(
                int(label),
                current_record,
                current_anchor,
            )
            if next_record is None or next_anchor is None:
                break
            out.append((int(next_slice), int(next_anchor[0]), int(next_anchor[1])))
            visited_slices.add(int(next_slice))
            current_slice = int(next_slice)
            current_record = next_record
            current_anchor = next_anchor
        return out

    current_component, current_anchor = _component_mask_and_anchor(labels_real[s0] == int(label), (y0, x0))
    if current_anchor is None or not np.any(current_component):
        return []

    out: List[Tuple[int, int, int]] = []
    current_slice = int(s0)

    max_walk = min(int(walk_back), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(walk_back)
    visited_slices = {int(current_slice)}
    for _ in range(int(max_walk)):
        next_slice = int(current_slice - int(direction_sign))
        if bool(wrap_axis):
            next_slice = int(next_slice % int(num_slices))
            if next_slice in visited_slices:
                break
        elif next_slice < 0 or next_slice >= num_slices:
            break

        next_slice_mask = labels_real[next_slice] == int(label)
        if not np.any(next_slice_mask):
            break

        next_component, next_anchor = _follow_branch_component(next_slice_mask, current_component, current_anchor)
        if next_anchor is None or not np.any(next_component):
            break

        out.append((int(next_slice), int(next_anchor[0]), int(next_anchor[1])))
        visited_slices.add(int(next_slice))
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


def _build_slice_endpoint_seeds(
    labels_real: np.ndarray,
    extension_slices: int,
    workers: int = 1,
    wrap_axis: bool = False,
    component_cache: Optional['SliceComponentTableCache'] = None,
) -> Tuple[List[SliceEndpointSeed], int]:
    """Build interpolation endpoint seeds with the v12.2.0 per-slice component scan.

    Interpolation no longer uses skeletonization. Each labeled 3D object is scanned slice by
    slice; every 2D connected component is evaluated independently for overlap continuation
    into the previous and next slice. Components without continuation become endpoint seeds in
    the corresponding direction. Radial interpolation can wrap the slice/frame axis so frame 0
    and the final radial frame are considered adjacent.
    """
    del extension_slices  # retained for call-site compatibility; scan endpoints do not need it.
    return build_slice_endpoint_seeds_from_label_volume(
        labels_real,
        workers=int(workers),
        desc='Interpolation: endpoint seeds [scan]',
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
    )


def _build_linear_slice_bridge_plan(
    labels_real: np.ndarray,
    source_label: int,
    target_label: int,
    source_point: Tuple[int, int, int],
    target_point: Tuple[int, int, int],
    direction_sign: Optional[int] = None,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
) -> Optional[SliceBridgeRenderPlan]:
    s0, y0, x0 = source_point
    s1, y1, x1 = target_point
    if int(s0) == int(s1):
        return None

    if component_cache is not None:
        source_record, source_anchor = component_cache.find_record_for_point(int(s0), int(source_label), (int(y0), int(x0)))
        target_record, target_anchor = component_cache.find_record_for_point(int(s1), int(target_label), (int(y1), int(x1)))
        if source_record is None or target_record is None or source_anchor is None or target_anchor is None:
            return None
        if int(source_record.area) <= 0 or int(target_record.area) <= 0:
            return None

        half_width = _local_half_width_for_component_records(source_record, source_anchor, target_record, target_anchor)
        source_local = _component_record_to_local_canvas(source_record, source_anchor, half_width)
        target_local = _component_record_to_local_canvas(target_record, target_anchor, half_width)
    else:
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

    if bool(wrap_axis):
        num_slices = int(labels_real.shape[0])
        sign = 1 if int(direction_sign or 1) >= 0 else -1
        if sign > 0:
            steps = int((int(s1) - int(s0)) % num_slices)
        else:
            steps = int((int(s0) - int(s1)) % num_slices)
    else:
        steps = int(abs(int(s1) - int(s0)))
        sign = 1 if int(s1) > int(s0) else -1
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
        sign=int(sign),
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
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
) -> SliceSeedBridgePlanResult:
    result = SliceSeedBridgePlanResult()

    candidates = _find_slice_projection_candidates(
        labels_real=labels_real,
        seed=seed,
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        max_candidates=int(interpolation_candidates),
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
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
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
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
                direction_sign=int(seed.direction_sign),
                wrap_axis=bool(wrap_axis),
                component_cache=component_cache,
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
    wrap_axis: bool = False,
) -> List[List[Tuple[int, int]]]:
    schedule: List[List[Tuple[int, int]]] = [[] for _ in range(int(num_slices))]
    for plan_idx, plan in enumerate(plans):
        start_slice = int(plan.source_point[0])
        for step_idx in range(1, int(plan.steps)):
            s = int(start_slice + int(plan.sign) * step_idx)
            if bool(wrap_axis):
                s = int(s % int(num_slices))
                schedule[s].append((int(plan_idx), int(step_idx)))
            elif 0 <= s < int(num_slices):
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
    wrap_axis: bool = False,
) -> Dict[str, object]:
    """Apply one interpolation pass directly to a view-volume stack.

    The pass keeps bridge creation simultaneous by searching against a frozen label snapshot and
    merging all newly created bridge voxels only after planning is complete. Endpoint discovery,
    candidate search, bridge planning and slice rendering are parallelized across independent
    objects / seeds / slices to reduce the long single-threaded stretch after compact relabel.
    For Radial views, wrap_axis lets endpoint search and bridge rendering wrap between the
    final and first azimuth frames.
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
            'wrap_axis': bool(wrap_axis),
            'endpoint_method': 'slice_component_scan',
            'planning_backend': interpolation_planning_backend_name(),
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
        wrap_axis=bool(wrap_axis),
        workers=int(workers),
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
            'wrap_axis': bool(wrap_axis),
            'endpoint_method': 'slice_component_scan',
            'planning_backend': interpolation_planning_backend_name(),
        }

    component_cache = SliceComponentTableCache(labels_mm)
    worker_count = choose_slice_parallel_workers(int(workers), int(labels_mm.shape[0]))
    seeds, num_endpoints = _build_slice_endpoint_seeds(
        labels_mm,
        extension_slices=int(max_slice_distance),
        workers=worker_count,
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
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
            'wrap_axis': bool(wrap_axis),
            'endpoint_method': 'slice_component_scan',
            'planning_backend': interpolation_planning_backend_name(),
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
                wrap_axis=bool(wrap_axis),
                component_cache=component_cache,
            )

        pending = max(plan_workers, plan_workers * 2)
        if plan_workers <= 1:
            iterable = (_plan_seed(int(idx)) for idx in range(len(seeds)))
        else:
            iterable = parallel_map_unordered(
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
            schedule = _build_slice_bridge_render_schedule(plans, int(mask_mm.shape[0]), wrap_axis=bool(wrap_axis))
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
        try:
            del component_cache
        except Exception:
            pass
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
        'wrap_axis': bool(wrap_axis),
        'endpoint_method': 'slice_component_scan',
        'planning_backend': interpolation_planning_backend_name(),
    }




@dataclass
class PreparedViewResult:
    model_name: str
    view_name: str
    native_support_mm: np.ndarray
    final_view_volume_mm: Optional[np.ndarray]
    interpolation_stats: List[Dict[str, object]]
    nrrd_layers: List[NrrdLayerRef] = field(default_factory=list)
    parent_mask_support_mm: Optional[object] = None
    parent_bridge_support_mm: Optional[object] = None


@dataclass
class TilePostprocessTask:
    model_name: str
    view_name: str
    config_id: str
    tile_id: str
    tile_mask_mm: np.ndarray
    tile_confmap_mm: Optional[np.ndarray]
    tile_mask_path: Path
    tile_confmap_path: Optional[Path]
    precleaned_slice_cleanup: bool = False


@dataclass
class TilePostprocessResult:
    model_name: str
    view_name: str
    config_id: str
    tile_id: str
    tile_mask_mm: Optional[np.ndarray] = None
    tile_mask_path: Optional[Path] = None
    tile_mask_store: Optional['RawBBoxMaskStore'] = None


@dataclass(frozen=True)
class DeferredTilePostprocessResult:
    model_name: str
    view_name: str
    config_id: str
    tile_id: str
    tile_mask_path: Path
    tile_shape: Tuple[int, int, int]
    storage_format: str = 'ctile-mask-v2-raw'



CTILE_FORMAT = 'ctile-mask-v2-raw'
CVOL_FORMAT = 'cvol-mask-v2-raw'
MASK_STORE_FORMATS = {CTILE_FORMAT, CVOL_FORMAT}
# RAM cache for raw bbox-store chunks.bin payloads during NRRD streaming.
_NRRD_RAW_STORE_CHUNKS_RAM_CACHE: Dict[Path, bytes] = {}
CTILE_INDEX_DTYPE = np.dtype([
    ('kind', 'u1'),              # 0 = empty/zero slice, 1 = raw uint8 bbox payload
    ('reserved', 'u1', (7,)),
    ('offset', '<u8'),
    ('payload_size', '<u8'),
    ('y0', '<u4'),
    ('x0', '<u4'),
    ('y1', '<u4'),
    ('x1', '<u4'),
    ('payload_nbytes', '<u8'),
])


NrrdSegmentExtent = Tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class NrrdLayerRef:
    """One orthogonal-space layer for decomposed NRRD export.

    The layer is a binary mask in the pipeline's orthogonal ``(t, Y, X)``
    processing geometry. The backing path may be a raw uint8 memmap or a
    v12.2 raw bbox cvol/ctile store. The NRRD writer reads either source
    slice-by-slice and writes a 4D NRRD with a trailing list axis.
    """

    key: str
    name: str
    path: Path
    shape: Tuple[int, int, int]
    dtype: str = 'uint8'
    storage_format: str = 'raw_u8'
    model_name: str = ''
    view_name: str = ''
    view_family: str = ''
    source: str = ''  # fullframe, tile, or global
    mask_kind: str = ''  # yolo, bridge, union, smoothing_result
    pass_index: int = 0
    tile_acceptance: str = ''  # parent_mask, parent_bridge, consolidated, or blank
    stage: str = ''
    description: str = ''
    # Slicer SegmentN_Extent in this layer backing store's own (X,Y,t) index space.
    # Final NRRD packaging maps this extent into the requested output geometry without
    # reopening the layer solely to compute header metadata.
    segment_extent_ijk: Optional[NrrdSegmentExtent] = None
    segment_extent_shape_tyx: Tuple[int, int, int] = (0, 0, 0)
    segment_extent_source: str = ''


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
    nrrd_layers: List[NrrdLayerRef] = field(default_factory=list)


def _view_uses_interpolation(view: ViewInfo, interpolate: int) -> bool:
    return bool((view.family in ('orthogonal', 'radial') or is_tilted_view(view)) and int(interpolate) > 0)


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


def _nrrd_empty_segment_extent() -> NrrdSegmentExtent:
    return (0, -1, 0, -1, 0, -1)


def _coerce_segment_extent(value: object) -> Optional[NrrdSegmentExtent]:
    if value is None:
        return None
    try:
        vals = [int(v) for v in value]  # type: ignore[arg-type]
    except Exception:
        return None
    if len(vals) != 6:
        return None
    return (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]))


def _segment_extent_is_empty(extent: Sequence[int]) -> bool:
    vals = [int(v) for v in extent]
    return len(vals) != 6 or vals[1] < vals[0] or vals[3] < vals[2] or vals[5] < vals[4]


def _segment_extent_to_json(extent: Sequence[int]) -> List[int]:
    coerced = _coerce_segment_extent(extent)
    if coerced is None:
        coerced = _nrrd_empty_segment_extent()
    return [int(v) for v in coerced]


def _raw_store_index_segment_extent(
    index: np.ndarray,
    shape_tyx: Sequence[int],
) -> NrrdSegmentExtent:
    """Compute a Slicer SegmentN extent from a raw cvol/ctile index without decoding payloads."""
    try:
        shape_i = (int(shape_tyx[0]), int(shape_tyx[1]), int(shape_tyx[2]))
    except Exception:
        return _nrrd_empty_segment_extent()
    idx_arr = np.asarray(index, dtype=CTILE_INDEX_DTYPE)
    if idx_arr.size <= 0 or any(v <= 0 for v in shape_i):
        return _nrrd_empty_segment_extent()
    nonempty = np.asarray(idx_arr['kind'] == np.uint8(1), dtype=bool)
    if not np.any(nonempty):
        return _nrrd_empty_segment_extent()

    z_vals = np.flatnonzero(nonempty).astype(np.int64, copy=False)
    x0_vals = np.asarray(idx_arr['x0'][nonempty], dtype=np.int64)
    x1_vals = np.asarray(idx_arr['x1'][nonempty], dtype=np.int64) - 1
    y0_vals = np.asarray(idx_arr['y0'][nonempty], dtype=np.int64)
    y1_vals = np.asarray(idx_arr['y1'][nonempty], dtype=np.int64) - 1
    min_x = int(np.clip(int(np.min(x0_vals)), 0, max(0, shape_i[2] - 1)))
    max_x = int(np.clip(int(np.max(x1_vals)), 0, max(0, shape_i[2] - 1)))
    min_y = int(np.clip(int(np.min(y0_vals)), 0, max(0, shape_i[1] - 1)))
    max_y = int(np.clip(int(np.max(y1_vals)), 0, max(0, shape_i[1] - 1)))
    min_t = int(np.clip(int(np.min(z_vals)), 0, max(0, shape_i[0] - 1)))
    max_t = int(np.clip(int(np.max(z_vals)), 0, max(0, shape_i[0] - 1)))
    if max_x < min_x or max_y < min_y or max_t < min_t:
        return _nrrd_empty_segment_extent()
    return (int(min_x), int(max_x), int(min_y), int(max_y), int(min_t), int(max_t))


@dataclass(frozen=True)
class RawBBoxSlicePayload:
    """One slice payload for the raw bbox mask store.

    Payloads are raw uint8 crop bytes in v12.2.0, not bitpacked or LZ4-compressed.
    """

    idx: int
    is_empty: bool
    y0: int = 0
    x0: int = 0
    y1: int = 0
    x1: int = 0
    payload_nbytes: int = 0
    payload: bytes = b''
    foreground_voxels: int = 0


class RawBBoxMaskStore:
    """Read adapter for v12.2.0 raw slice-bbox binary mask volumes.

    The payload format is uncompressed: empty slices are elided, nonempty slices are cropped to their
    nonzero bbox, and the crop is written as raw uint8 bytes. No NumPy packbits and
    no LZ4 are used for waiting tiles or cvol NRRD/support layers.
    """

    def __init__(
        self,
        root: Path,
        meta: Dict[str, object],
        index: np.ndarray,
        *,
        chunks_bytes: Optional[bytes] = None,
    ) -> None:
        self.root = Path(root)
        self.meta = dict(meta)
        self.index = np.asarray(index, dtype=CTILE_INDEX_DTYPE)
        self.chunks_path = self.root / 'chunks.bin'
        self._chunks_bytes: Optional[bytes] = chunks_bytes
        shape = self.meta.get('shape')
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f'{self.root}: invalid raw-mask shape metadata: {shape!r}')
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        if int(self.index.shape[0]) != int(self.shape[0]):
            raise ValueError(f'{self.root}: index slice count {int(self.index.shape[0])} != shape[0] {int(self.shape[0])}')

    @classmethod
    def open(cls, root: Path, *, cache_payload_in_ram: bool = False) -> 'RawBBoxMaskStore':
        root = Path(root)
        meta_path = root / 'meta.json'
        index_path = root / 'index.bin'
        chunks_path = root / 'chunks.bin'
        if not meta_path.exists() or not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f'Incomplete raw mask store: {root}')
        meta = json.loads(meta_path.read_text())
        fmt = str(meta.get('format', ''))
        if fmt not in MASK_STORE_FORMATS:
            raise ValueError(f'{root}: unsupported raw mask format {meta.get("format")!r}')
        shape = meta.get('shape')
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f'{root}: invalid shape metadata {shape!r}')
        index = np.fromfile(index_path, dtype=CTILE_INDEX_DTYPE, count=int(shape[0]))
        chunks_bytes: Optional[bytes] = None
        if bool(cache_payload_in_ram):
            try:
                cache_key = chunks_path.resolve()
            except Exception:
                cache_key = chunks_path
            chunks_bytes = _NRRD_RAW_STORE_CHUNKS_RAM_CACHE.get(cache_key)
            if chunks_bytes is None:
                chunks_bytes = chunks_path.read_bytes()
                _NRRD_RAW_STORE_CHUNKS_RAM_CACHE[cache_key] = chunks_bytes
                print(
                    f'Raw bbox mask store cached in RAM for NRRD streaming: {root} '
                    f'({len(chunks_bytes) / GIB:.3f} GiB chunks.bin)'
                )
            else:
                print(
                    f'Raw bbox mask store reused from RAM for NRRD streaming: {root} '
                    f'({len(chunks_bytes) / GIB:.3f} GiB chunks.bin)'
                )
        return cls(root, meta, index, chunks_bytes=chunks_bytes)

    # DEAD_CODE_MARKER(v12.2.0-post-refactor): unused diagnostic property retained for raw-store debugging.
    @property
    def chunks_cached_in_ram(self) -> bool:
        return self._chunks_bytes is not None

    def close(self) -> None:
        self._chunks_bytes = None

    def fill_decoded_slice_into(self, idx: int, out: np.ndarray) -> None:
        """Decode one raw-bbox slice directly into an existing ``(Y,X)`` array."""
        idx_i = int(idx)
        z_dim, h, w = self.shape
        if idx_i < 0 or idx_i >= int(z_dim):
            raise IndexError(idx_i)
        out_arr = np.asarray(out)
        if tuple(int(x) for x in out_arr.shape) != (int(h), int(w)):
            raise ValueError(f'{self.root}: output slice shape {tuple(out_arr.shape)} != {(int(h), int(w))}')
        out_arr.fill(np.uint8(0))
        rec = self.index[idx_i]
        if int(rec['kind']) == 0:
            return
        if int(rec['kind']) != 1:
            raise ValueError(f'{self.root}: invalid raw-mask chunk marker {int(rec["kind"])} at slice {idx_i}')

        y0 = int(rec['y0']); x0 = int(rec['x0']); y1 = int(rec['y1']); x1 = int(rec['x1'])
        if not (0 <= y0 < y1 <= int(h) and 0 <= x0 < x1 <= int(w)):
            raise ValueError(f'{self.root}: invalid bbox {(y0, x0, y1, x1)} for shape {(h, w)} at slice {idx_i}')
        payload_size = int(rec['payload_size'])
        payload_nbytes = int(rec['payload_nbytes'])
        if payload_size <= 0 or payload_nbytes <= 0:
            raise ValueError(f'{self.root}: nonempty slice {idx_i} has empty payload metadata')

        if self._chunks_bytes is not None:
            start = int(rec['offset'])
            stop = start + int(payload_size)
            payload = self._chunks_bytes[start:stop]
        else:
            with self.chunks_path.open('rb') as fh:
                fh.seek(int(rec['offset']))
                payload = fh.read(int(payload_size))
        if len(payload) != payload_size or len(payload) != payload_nbytes:
            raise IOError(f'{self.root}: short read for slice {idx_i}: {len(payload)} != {payload_size}')
        crop_h = int(y1 - y0)
        crop_w = int(x1 - x0)
        expected = int(crop_h * crop_w)
        if int(payload_nbytes) != expected:
            raise ValueError(f'{self.root}: raw payload byte count mismatch at slice {idx_i}: {payload_nbytes} != {expected}')
        crop = np.frombuffer(payload, dtype=np.uint8, count=expected).reshape((crop_h, crop_w))
        out_arr[y0:y1, x0:x1] = crop.astype(out_arr.dtype, copy=False)

    def decode_slice(self, idx: int, *, dtype: np.dtype | str | type = np.uint8) -> np.ndarray:
        idx_i = int(idx)
        z_dim, h, w = self.shape
        if idx_i < 0 or idx_i >= int(z_dim):
            raise IndexError(idx_i)
        out = np.zeros((int(h), int(w)), dtype=np.dtype(dtype))
        self.fill_decoded_slice_into(idx_i, out)
        return out

    # DEAD_CODE_MARKER(v12.2.0-post-refactor): unused convenience iterator retained for raw-store debugging.
    def iter_slices(self) -> Iterator[np.ndarray]:
        for idx in range(int(self.shape[0])):
            yield self.decode_slice(idx)

    def unlink(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _encode_bool_mask_slice_payload(idx: int, mask_bool: np.ndarray) -> RawBBoxSlicePayload:
    mask_arr = np.asarray(mask_bool, dtype=bool)
    if mask_arr.size == 0 or not np.any(mask_arr):
        return RawBBoxSlicePayload(idx=int(idx), is_empty=True)

    rows = np.any(mask_arr, axis=1)
    cols = np.any(mask_arr, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(rows.size - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(cols.size - np.argmax(cols[::-1]))
    crop = np.ascontiguousarray(mask_arr[y0:y1, x0:x1], dtype=np.uint8)
    payload = crop.tobytes(order='C')
    return RawBBoxSlicePayload(
        idx=int(idx),
        is_empty=False,
        y0=int(y0),
        x0=int(x0),
        y1=int(y1),
        x1=int(x1),
        payload_nbytes=int(len(payload)),
        payload=payload,
        foreground_voxels=int(np.count_nonzero(crop)),
    )


def _encode_ctile_slice(idx: int, tile_mask_mm: np.ndarray) -> RawBBoxSlicePayload:
    mask_bool = np.asarray(tile_mask_mm[int(idx)], dtype=np.uint8) > 0
    return _encode_bool_mask_slice_payload(int(idx), mask_bool)


def _write_raw_bbox_payload_store(
    *,
    shape: Tuple[int, int, int],
    store_dir: Path,
    encode_slice: Callable[[int], RawBBoxSlicePayload],
    format_name: str,
    desc: str,
    workers: int = 1,
    extra_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Write a slice-chunked raw bbox binary mask store.

    Write raw uint8 bbox payloads; this stage no longer bitpacks or LZ4-compresses payloads.
    """
    fmt = str(format_name)
    if fmt not in MASK_STORE_FORMATS:
        raise ValueError(f'Unsupported raw bbox mask format: {fmt}')

    shape_i = (int(shape[0]), int(shape[1]), int(shape[2]))
    if any(v < 0 for v in shape_i):
        raise ValueError(f'{desc}: invalid raw store shape {shape_i}')

    store_dir = Path(store_dir)
    chunks_path_prewrite = store_dir / 'chunks.bin'
    try:
        _NRRD_RAW_STORE_CHUNKS_RAM_CACHE.pop(chunks_path_prewrite.resolve(), None)
    except Exception:
        _NRRD_RAW_STORE_CHUNKS_RAM_CACHE.pop(chunks_path_prewrite, None)
    if store_dir.exists():
        shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = store_dir / 'chunks.bin'
    index_path = store_dir / 'index.bin'
    meta_path = store_dir / 'meta.json'

    index = np.zeros((int(shape_i[0]),), dtype=CTILE_INDEX_DTYPE)
    worker_count = choose_slice_parallel_workers(int(workers), int(shape_i[0]))
    payload_bytes = 0
    nonempty_slices = 0
    foreground_voxels = 0
    min_t, max_t = int(shape_i[0]), -1
    min_y, max_y = int(shape_i[1]), -1
    min_x, max_x = int(shape_i[2]), -1

    if worker_count <= 1:
        iterable: Iterable[RawBBoxSlicePayload] = (encode_slice(idx) for idx in range(int(shape_i[0])))
    else:
        iterable = parallel_map_in_order(
            encode_slice,
            range(int(shape_i[0])),
            max_workers=worker_count,
            max_pending=max(worker_count, worker_count * 2),
        )

    offset = 0
    with chunks_path.open('wb') as chunks_fh:
        for item in iterable:
            idx = int(item.idx)
            if idx < 0 or idx >= int(shape_i[0]):
                raise ValueError(f'{desc}: encoder returned out-of-range slice index {idx}')
            if bool(item.is_empty):
                index[idx]['kind'] = np.uint8(0)
                continue
            payload = bytes(item.payload)
            payload_size = int(len(payload))
            chunks_fh.write(payload)
            index[idx]['kind'] = np.uint8(1)
            index[idx]['offset'] = np.uint64(offset)
            index[idx]['payload_size'] = np.uint64(payload_size)
            index[idx]['y0'] = np.uint32(item.y0)
            index[idx]['x0'] = np.uint32(item.x0)
            index[idx]['y1'] = np.uint32(item.y1)
            index[idx]['x1'] = np.uint32(item.x1)
            index[idx]['payload_nbytes'] = np.uint64(item.payload_nbytes)
            offset += payload_size
            payload_bytes += payload_size
            foreground_voxels += int(item.foreground_voxels)
            nonempty_slices += 1
            min_t = min(int(min_t), int(idx))
            max_t = max(int(max_t), int(idx))
            min_y = min(int(min_y), int(item.y0))
            max_y = max(int(max_y), int(item.y1) - 1)
            min_x = min(int(min_x), int(item.x0))
            max_x = max(int(max_x), int(item.x1) - 1)

    index.tofile(index_path)
    raw_logical_bytes = int(array_nbytes(shape_i, np.uint8))
    segment_extent_ijk = (
        _nrrd_empty_segment_extent()
        if int(max_t) < 0
        else (int(min_x), int(max_x), int(min_y), int(max_y), int(min_t), int(max_t))
    )
    stats: Dict[str, object] = {
        'nonempty_slices': int(nonempty_slices),
        'empty_slices': int(shape_i[0] - nonempty_slices),
        'foreground_voxels': int(foreground_voxels),
        'logical_raw_uint8_bytes': int(raw_logical_bytes),
        'raw_payload_bytes': int(payload_bytes),
        'index_bytes': int(index.nbytes),
        'segment_extent_ijk': _segment_extent_to_json(segment_extent_ijk),
        'segment_extent_shape_tyx': [int(shape_i[0]), int(shape_i[1]), int(shape_i[2])],
    }
    meta = {
        'format': fmt,
        'shape': [int(shape_i[0]), int(shape_i[1]), int(shape_i[2])],
        'dtype': 'bool',
        'logical_dtype_in_pipeline': 'uint8_0_or_1',
        'chunking': 'slice',
        'precodec': 'none',
        'compressor': 'none',
        'bbox_per_chunk': True,
        'zero_chunk_elision': True,
        'index_dtype': 'ctile-index-v2-raw',
        'index_record_bytes': int(CTILE_INDEX_DTYPE.itemsize),
        'payload_shape_encoding': 'raw_uint8_bbox_shape_from_index',
        'description': str(desc),
        'segment_extent_ijk': _segment_extent_to_json(segment_extent_ijk),
        'segment_extent_axis_order': 'Slicer IJK inclusive extent: minX maxX minY maxY minT maxT for internal layer order (t,Y,X)',
        'segment_extent_shape_tyx': [int(shape_i[0]), int(shape_i[1]), int(shape_i[2])],
        'stats': stats,
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    print(
        f'{desc}: raw bbox mask store {store_dir} '
        f'(logical_raw={raw_logical_bytes / GIB:.2f} GiB, payload={payload_bytes / GIB:.2f} GiB, '
        f'nonempty_slices={nonempty_slices})'
    )
    return stats


def write_raw_bbox_mask_store(
    mask_volume: np.ndarray,
    store_dir: Path,
    *,
    format_name: str = CVOL_FORMAT,
    desc: str = 'Raw bbox mask store',
    workers: int = 1,
    extra_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Write a v12.2.0 raw bbox mask store."""
    arr = np.asarray(mask_volume)
    if arr.ndim != 3:
        raise ValueError(f'{desc}: expected 3D mask volume, got shape {arr.shape}')
    shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))

    def _encode(idx: int) -> RawBBoxSlicePayload:
        return _encode_ctile_slice(int(idx), arr)

    return _write_raw_bbox_payload_store(
        shape=shape,
        store_dir=Path(store_dir),
        encode_slice=_encode,
        format_name=str(format_name),
        desc=str(desc),
        workers=int(workers),
        extra_meta=extra_meta,
    )

def _volume_shape_tuple(volume: object) -> Tuple[int, int, int]:
    shape = getattr(volume, 'shape', None)
    if shape is None:
        shape = np.asarray(volume).shape
    if len(shape) != 3:
        raise ValueError(f'Expected a 3D binary volume, got shape {shape}')
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def _read_binary_volume_slice_u8(volume: object, idx: int) -> np.ndarray:
    if isinstance(volume, RawBBoxMaskStore):
        return volume.decode_slice(int(idx), dtype=np.uint8)
    return np.asarray(volume[int(idx)], dtype=np.uint8)


def _read_binary_volume_slice_bool(volume: object, idx: int) -> np.ndarray:
    return np.asarray(_read_binary_volume_slice_u8(volume, int(idx)), dtype=np.uint8) > 0


def subtract_volume_to_raw_bbox_store(
    after_mm: np.ndarray,
    before_volume: object,
    store_dir: Path,
    *,
    desc: str,
    workers: int = 1,
    format_name: str = CVOL_FORMAT,
) -> Dict[str, object]:
    after_shape = _volume_shape_tuple(after_mm)
    before_shape = _volume_shape_tuple(before_volume)
    if after_shape != before_shape:
        raise ValueError(f'{desc}: shape mismatch {after_shape} vs {before_shape}')

    def _encode_delta(idx: int) -> RawBBoxSlicePayload:
        after_slice = np.asarray(after_mm[int(idx)], dtype=np.uint8) > 0
        before_slice = _read_binary_volume_slice_bool(before_volume, int(idx))
        return _encode_bool_mask_slice_payload(int(idx), after_slice & (~before_slice))

    return _write_raw_bbox_payload_store(
        shape=after_shape,
        store_dir=Path(store_dir),
        encode_slice=_encode_delta,
        format_name=str(format_name),
        desc=str(desc),
        workers=int(workers),
        extra_meta={'operation': 'after_and_not_before'},
    )


def _memmap_backing_path(arr: object) -> Optional[Path]:
    if arr is None:
        return None
    try:
        if isinstance(arr, np.memmap):
            filename = getattr(arr, 'filename', None)
            return Path(filename) if filename else None
    except Exception:
        pass
    try:
        base = getattr(arr, 'base', None)
        if isinstance(base, np.memmap):
            filename = getattr(base, 'filename', None)
            return Path(filename) if filename else None
    except Exception:
        pass
    return None


def temp_binary_archive_enabled() -> bool:
    return _env_flag('YOLO_TTA_ARCHIVE_TEMP_BINARY_VOLUMES', True)


def raw_bbox_nrrd_layers_enabled() -> bool:
    raw_env = os.environ.get('YOLO_TTA_RAW_BBOX_NRRD_LAYERS')
    if raw_env is not None:
        return _env_flag('YOLO_TTA_RAW_BBOX_NRRD_LAYERS', True)
    return _env_flag('YOLO_TTA_COMPRESS_NRRD_LAYERS', True)  # backward-compatible alias; payloads are raw bbox stores, not compressed


def tile_intermediate_accumulators_prefer_memory() -> bool:
    """Keep tile staging/consolidation canvases in anonymous RAM by default."""
    return _env_flag('YOLO_TTA_TILE_ACCUMULATORS_IN_RAM', True)


def tile_intermediate_accumulator_reserve_bytes() -> int:
    return int(max(0.0, _env_float('YOLO_TTA_TILE_ACCUMULATOR_RESERVE_GIB', 64.0)) * GIB)


def waiting_tile_spill_enabled() -> bool:
    """Return True to recover legacy waiting-tile ctile spill behavior.

    v12.2.11 keeps postprocessed tiles in RAM while they wait for parent support
    unless this opt-in escape hatch is enabled.  That avoids the disk write/read
    cycle for the common SLURM case where the allocation has hundreds of GiB of
    anonymous memory available.
    """
    return _env_flag('YOLO_TTA_SPILL_WAITING_TILES', False)


def nrrd_cache_raw_bbox_layers_in_ram_enabled() -> bool:
    """Cache cvol/ctile raw bbox payload files in RAM during NRRD streaming.

    The NRRD writer uses the same layer backing files for both ``NRRD streaming:
    segment extents`` and ``NRRD streaming: decomposed layers``.  When those
    backing files are raw bbox stores, their ``chunks.bin`` payloads
    are intentionally read into RAM before slice traversal to remove repeated
    random NVMe reads.
    """
    return _env_flag('YOLO_TTA_NRRD_CACHE_RAW_BBOX_LAYERS_IN_RAM', True)


def archive_or_delete_binary_volume_storage(
    volume: Optional[np.ndarray],
    *,
    keep_temp: bool,
    workers: int,
    desc: str,
    raw_bbox_dir: Optional[Path] = None,
) -> None:
    """Close a temporary binary volume and replace kept raw uint8 memmaps with cvol."""
    if volume is None:
        return

    raw_path = _memmap_backing_path(volume)
    if bool(keep_temp) and bool(temp_binary_archive_enabled()) and raw_path is not None:
        archive_dir = Path(raw_bbox_dir) if raw_bbox_dir is not None else raw_path.with_name(raw_path.name + '.cvol')
        try:
            write_raw_bbox_mask_store(
                np.asarray(volume),
                archive_dir,
                format_name=CVOL_FORMAT,
                desc=f'{desc} archive',
                workers=int(workers),
                extra_meta={'archived_from_raw_path': str(raw_path)},
            )
            close_memmap_array(volume)
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
            return
        except Exception as exc:
            print(f'Warning: failed to archive {desc} as raw bbox cvol ({exc}); keeping raw volume {raw_path}')

    close_memmap_array(volume)
    if not bool(keep_temp) and raw_path is not None:
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass


def close_raw_store_or_memmap_volume(volume: object, *, keep_temp: bool = True) -> None:
    if volume is None:
        return
    if isinstance(volume, RawBBoxMaskStore):
        volume.close()
        if not bool(keep_temp):
            volume.unlink()
        return
    close_memmap_array(volume)


def _volume_has_foreground(mask_mm: np.ndarray) -> bool:
    for idx in range(int(mask_mm.shape[0])):
        if np.any(np.asarray(mask_mm[int(idx)], dtype=bool)):
            return True
    return False



def _sanitize_nrrd_layer_token(value: object) -> str:
    token = re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(value).strip())
    token = token.strip('_')
    return token or 'unnamed'


def _nrrd_layer_key(
    *,
    model_name: str,
    view_name: str,
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    tile_acceptance: str = '',
    stage: str = '',
) -> str:
    parts = [
        _sanitize_nrrd_layer_token(model_name),
        _sanitize_nrrd_layer_token(view_name),
        _sanitize_nrrd_layer_token(source),
        _sanitize_nrrd_layer_token(mask_kind),
    ]
    if int(pass_index) > 0:
        parts.append(f'pass{int(pass_index):02d}')
    if tile_acceptance:
        parts.append(_sanitize_nrrd_layer_token(tile_acceptance))
    if stage:
        parts.append(_sanitize_nrrd_layer_token(stage))
    return '__'.join(parts)


def _nrrd_layer_name(
    *,
    view: Optional[ViewInfo],
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    tile_acceptance: str = '',
    stage: str = '',
) -> str:
    view_label = pretty_view_name(view) if view is not None else 'Global'
    pieces = [view_label]
    if source:
        pieces.append(str(source))
    if mask_kind == 'yolo':
        pieces.append('YOLO mask')
    elif mask_kind == 'bridge':
        pieces.append('postprocess bridge')
    elif mask_kind == 'smoothing_result':
        pieces.append('smoothing result')
    elif mask_kind == 'union':
        pieces.append('union')
    else:
        pieces.append(str(mask_kind))
    if int(pass_index) > 0:
        pieces.append(f'pass {int(pass_index)}')
    if tile_acceptance:
        pieces.append(f'accepted by {str(tile_acceptance).replace("_", " ")}')
    if stage:
        pieces.append(str(stage).replace('_', ' '))
    return ' / '.join(pieces)


def subtract_volume_to_mmap(
    after_mm: np.ndarray,
    before_mm: np.ndarray,
    out_path: Path,
    desc: str,
    *,
    workers: int = 1,
    prefer_memory: bool = False,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    """Write ``after AND NOT before`` into a uint8 workspace.

    The historical name is retained for call-site stability.  NRRD delta layers use
    ``prefer_memory=True`` so short-lived full-volume deltas can stay in RAM when
    the SLURM allocation has headroom instead of appearing briefly in ``nrrd_work``.
    """
    after_arr = np.asarray(after_mm)
    before_arr = np.asarray(before_mm)
    if tuple(int(x) for x in after_arr.shape) != tuple(int(x) for x in before_arr.shape):
        raise ValueError(f'{desc}: shape mismatch {after_arr.shape} vs {before_arr.shape}')
    out = allocate_workspace_array(
        shape=tuple(int(x) for x in after_arr.shape),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    total = int(after_arr.shape[0]) if after_arr.ndim > 0 else 0

    def _delta_slice(idx: int) -> None:
        after_slice = np.asarray(after_arr[int(idx)], dtype=bool)
        before_slice = np.asarray(before_arr[int(idx)], dtype=bool)
        out[int(idx), :, :] = (after_slice & (~before_slice)).astype(np.uint8, copy=False)

    parallel_for_indices_chunked(
        total,
        _delta_slice,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=desc,
        show_progress=False,
        target_chunks_per_worker=2,
    )
    flush_array(out)
    return out


def project_view_volume_to_orthogonal_volume(
    view_mask_mm: np.ndarray,
    view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    workers: int = 1,
    prefer_memory: bool = False,
    reserve_bytes: int = 16 * GIB,
) -> np.ndarray:
    """Project a view-native binary volume into orthogonal processing geometry (t,Y,X)."""
    t_dim = int(view.full_t) if int(view.full_t) > 0 else int(view_mask_mm.shape[0])
    h_dim = int(view.full_h) if int(view.full_h) > 0 else int(view.src_h)
    w_dim = int(view.full_w) if int(view.full_w) > 0 else int(view.src_w)

    if view.family == 'radial':
        return backproject_radial_volume_to_volume(
            radial_mask_mm=view_mask_mm,
            radial_view=view,
            out_path=out_path,
            desc=desc,
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
        )

    if is_tilted_view(view):
        return backproject_tilted_volume_to_volume(
            tilted_mask_mm=view_mask_mm,
            tilted_view=view,
            out_path=out_path,
            desc=desc,
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
        )

    if view.name == 'transverse':
        if tuple(int(x) for x in np.asarray(view_mask_mm).shape) != (t_dim, h_dim, w_dim):
            raise ValueError(f'{desc}: transverse layer shape {tuple(view_mask_mm.shape)} != {(t_dim, h_dim, w_dim)}')
        return copy_workspace_array(
            np.asarray(view_mask_mm, dtype=np.uint8),
            out_path,
            desc=desc,
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
        )

    out = allocate_workspace_array(
        shape=(t_dim, h_dim, w_dim),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    if view.name == 'sagittal':
        src = np.asarray(view_mask_mm)
        if tuple(int(x) for x in src.shape) != (h_dim, t_dim, w_dim):
            raise ValueError(f'{desc}: sagittal layer shape {tuple(src.shape)} != {(h_dim, t_dim, w_dim)}')
        def _copy_y(y_idx: int) -> None:
            out[:, int(y_idx), :] = np.asarray(src[int(y_idx)], dtype=np.uint8)
        parallel_for_indices_chunked(
            h_dim,
            _copy_y,
            max_workers=choose_slice_parallel_workers(int(workers), h_dim),
            desc=desc,
            show_progress=False,
            target_chunks_per_worker=2,
        )
    elif view.name == 'coronal':
        src = np.asarray(view_mask_mm)
        if tuple(int(x) for x in src.shape) != (w_dim, t_dim, h_dim):
            raise ValueError(f'{desc}: coronal layer shape {tuple(src.shape)} != {(w_dim, t_dim, h_dim)}')
        def _copy_x(x_idx: int) -> None:
            out[:, :, int(x_idx)] = np.asarray(src[int(x_idx)], dtype=np.uint8)
        parallel_for_indices_chunked(
            w_dim,
            _copy_x,
            max_workers=choose_slice_parallel_workers(int(workers), w_dim),
            desc=desc,
            show_progress=False,
            target_chunks_per_worker=2,
        )
    else:  # pragma: no cover
        raise ValueError(f'Unsupported view for orthogonal NRRD projection: {view.name}/{view.family}')

    flush_array(out)
    return out


def materialize_nrrd_view_layer(
    view_volume_mm: np.ndarray,
    *,
    model_name: str,
    view: ViewInfo,
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    tile_acceptance: str = '',
    stage: str = '',
    description: str = '',
    temp_dir: Path,
    workers: int = 1,
) -> Optional[NrrdLayerRef]:
    """Persist a view-derived layer in orthogonal processing geometry for the NRRD writer."""
    if not _volume_has_foreground(view_volume_mm):
        return None

    key = _nrrd_layer_key(
        model_name=str(model_name),
        view_name=str(view.name),
        source=str(source),
        mask_kind=str(mask_kind),
        pass_index=int(pass_index),
        tile_acceptance=str(tile_acceptance),
        stage=str(stage),
    )
    layer_dir = temp_dir / 'nrrd_layers' / str(model_name) / str(view.name)
    storage_format = 'raw_u8'
    if raw_bbox_nrrd_layers_enabled():
        raw_path = temp_dir / 'nrrd_work' / 'projected_layers' / str(model_name) / str(view.name) / f'{key}.orthogonal.u8.dat'
        out_path = layer_dir / f'{key}.orthogonal.cvol'
    else:
        raw_path = layer_dir / f'{key}.orthogonal.u8.dat'
        out_path = raw_path

    transient_projection_in_memory = bool(raw_bbox_nrrd_layers_enabled())
    projected = project_view_volume_to_orthogonal_volume(
        view_volume_mm,
        view,
        raw_path,
        desc=f'NRRD layer {key}',
        workers=int(workers),
        prefer_memory=bool(transient_projection_in_memory),
        reserve_bytes=32 * GIB,
    )
    shape = tuple(int(x) for x in np.asarray(projected).shape)
    if raw_bbox_nrrd_layers_enabled():
        layer_stats = write_raw_bbox_mask_store(
            projected,
            out_path,
            format_name=CVOL_FORMAT,
            desc=f'NRRD layer {key}',
            workers=int(workers),
            extra_meta={
                'nrrd_layer_key': key,
                'source_raw_path': str(raw_path),
                'source_raw_workspace': 'in_memory_when_available' if bool(transient_projection_in_memory) else 'disk_backed',
            },
        )
        segment_extent = _coerce_segment_extent(layer_stats.get('segment_extent_ijk')) or _nrrd_empty_segment_extent()
        segment_extent_source = 'raw_bbox_cvol_index'
        storage_format = CVOL_FORMAT
        close_memmap_array(projected)
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        segment_extent = compute_segment_extent_zyx(projected)
        segment_extent_source = 'raw_layer_materialization_scan'
        close_memmap_array(projected)

    return NrrdLayerRef(
        key=key,
        name=_nrrd_layer_name(
            view=view,
            source=str(source),
            mask_kind=str(mask_kind),
            pass_index=int(pass_index),
            tile_acceptance=str(tile_acceptance),
            stage=str(stage),
        ),
        path=out_path,
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        dtype='uint8',
        storage_format=storage_format,
        model_name=str(model_name),
        view_name=str(view.name),
        view_family=str(view.family),
        source=str(source),
        mask_kind=str(mask_kind),
        pass_index=int(pass_index),
        tile_acceptance=str(tile_acceptance),
        stage=str(stage),
        description=str(description),
        segment_extent_ijk=segment_extent,
        segment_extent_shape_tyx=(int(shape[0]), int(shape[1]), int(shape[2])),
        segment_extent_source=segment_extent_source,
    )

def materialize_nrrd_global_layer(
    volume_mm: np.ndarray,
    *,
    model_name: str,
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    stage: str = '',
    description: str = '',
    temp_dir: Path,
    workers: int = 1,
) -> Optional[NrrdLayerRef]:
    if not _volume_has_foreground(volume_mm):
        return None
    view_name = 'global'
    key = _nrrd_layer_key(
        model_name=str(model_name),
        view_name=view_name,
        source=str(source),
        mask_kind=str(mask_kind),
        pass_index=int(pass_index),
        stage=str(stage),
    )
    layer_dir = temp_dir / 'nrrd_layers' / str(model_name) / view_name
    storage_format = 'raw_u8'
    if raw_bbox_nrrd_layers_enabled():
        raw_path = temp_dir / 'nrrd_work' / 'global_layers' / str(model_name) / f'{key}.orthogonal.u8.dat'
        out_path = layer_dir / f'{key}.orthogonal.cvol'
    else:
        raw_path = layer_dir / f'{key}.orthogonal.u8.dat'
        out_path = raw_path

    transient_copy_in_memory = bool(raw_bbox_nrrd_layers_enabled())
    copied = copy_workspace_array(
        np.asarray(volume_mm, dtype=np.uint8),
        raw_path,
        desc=f'NRRD layer {key}',
        prefer_memory=bool(transient_copy_in_memory),
        reserve_bytes=32 * GIB,
        workers=int(workers),
    )
    shape = tuple(int(x) for x in np.asarray(copied).shape)
    if raw_bbox_nrrd_layers_enabled():
        layer_stats = write_raw_bbox_mask_store(
            copied,
            out_path,
            format_name=CVOL_FORMAT,
            desc=f'NRRD layer {key}',
            workers=int(workers),
            extra_meta={
                'nrrd_layer_key': key,
                'source_raw_path': str(raw_path),
                'source_raw_workspace': 'in_memory_when_available' if bool(transient_copy_in_memory) else 'disk_backed',
            },
        )
        segment_extent = _coerce_segment_extent(layer_stats.get('segment_extent_ijk')) or _nrrd_empty_segment_extent()
        segment_extent_source = 'raw_bbox_cvol_index'
        storage_format = CVOL_FORMAT
        close_memmap_array(copied)
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        segment_extent = compute_segment_extent_zyx(copied)
        segment_extent_source = 'raw_layer_materialization_scan'
        close_memmap_array(copied)

    return NrrdLayerRef(
        key=key,
        name=_nrrd_layer_name(
            view=None,
            source=str(source),
            mask_kind=str(mask_kind),
            pass_index=int(pass_index),
            stage=str(stage),
        ),
        path=out_path,
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        dtype='uint8',
        storage_format=storage_format,
        model_name=str(model_name),
        view_name=view_name,
        view_family='global',
        source=str(source),
        mask_kind=str(mask_kind),
        pass_index=int(pass_index),
        stage=str(stage),
        description=str(description),
        segment_extent_ijk=segment_extent,
        segment_extent_shape_tyx=(int(shape[0]), int(shape[1]), int(shape[2])),
        segment_extent_source=segment_extent_source,
    )

def prepare_view_volume_after_fullframe(
    *,
    model_name: str,
    view: ViewInfo,
    union_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray],
    union_path: Path,
    confmap_path: Optional[Path],
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
    nrrd_layers_enabled: bool = False,
    precleaned_slice_cleanup: bool = False,
) -> PreparedViewResult:
    baseline_native_volume = union_mm
    nrrd_layers: List[NrrdLayerRef] = []
    parent_mask_support_mm: Optional[object] = None
    parent_bridge_support_mm: Optional[object] = None
    parent_mask_support_path: Optional[Path] = None
    parent_bridge_support_path: Optional[Path] = None

    cleanup_view_volume_after_prediction_inplace(
        baseline_native_volume,
        confmap_mm,
        view,
        float(min_conf),
        float(min_radius),
        workers=int(slice_workers),
        precleaned_slice_cleanup=bool(precleaned_slice_cleanup),
    )

    close_memmap_array(confmap_mm)
    if confmap_path is not None and not keep_temp:
        try:
            confmap_path.unlink(missing_ok=True)
        except Exception:
            pass

    if bool(nrrd_layers_enabled):
        layer_ref = materialize_nrrd_view_layer(
            baseline_native_volume,
            model_name=str(model_name),
            view=view,
            source='fullframe',
            mask_kind='yolo',
            pass_index=0,
            stage='pre_interpolation',
            description='Cleaned full-frame YOLO mask before interpolation bridges.',
            temp_dir=temp_dir,
            workers=int(slice_workers),
        )
        if layer_ref is not None:
            nrrd_layers.append(layer_ref)

        # Only dense tiled NRRD category gating needs a persistent parent-YOLO support copy.
        # Store it as a slice-chunked cvol instead of a raw uint8 memmap to avoid one
        # full-volume duplicate per active view.
        if bool(dense_tiling_active):
            parent_mask_support_path = temp_dir / 'nrrd_support' / str(model_name) / view.name / 'fullframe_yolo_support.cvol'
            write_raw_bbox_mask_store(
                baseline_native_volume,
                parent_mask_support_path,
                format_name=CVOL_FORMAT,
                desc=f'NRRD support fullframe YOLO {model_name}/{view.name}',
                workers=int(slice_workers),
                extra_meta={'support_kind': 'fullframe_yolo_pre_interpolation'},
            )
            parent_mask_support_mm = RawBBoxMaskStore.open(parent_mask_support_path)


    interpolation_stats: List[Dict[str, object]] = []
    if _view_uses_interpolation(view, int(interpolate)):
        total_passes = int(interpolate_passes)
        for pass_idx in range(1, total_passes + 1):
            before_pass_mm: Optional[np.ndarray] = None
            before_pass_path: Optional[Path] = None
            if bool(nrrd_layers_enabled):
                before_pass_path = temp_dir / 'nrrd_work' / str(model_name) / view.name / f'fullframe_before_pass{int(pass_idx):02d}.u8.dat'
                before_pass_mm = copy_workspace_array(
                    baseline_native_volume,
                    before_pass_path,
                    desc=f'NRRD fullframe before interpolation pass {int(pass_idx)} {model_name}/{view.name}',
                    prefer_memory=True,
                    reserve_bytes=32 * GIB,
                    workers=int(slice_workers),
                )

            baseline_native_volume, stats_local = interpolate_view_volume_pass_maybe_process(
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
                wrap_axis=bool(view.family == 'radial'),
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

            if bool(nrrd_layers_enabled) and before_pass_mm is not None:
                delta_path = temp_dir / 'nrrd_work' / str(model_name) / view.name / f'fullframe_bridge_pass{int(pass_idx):02d}.u8.dat'
                bridge_delta_mm = subtract_volume_to_mmap(
                    baseline_native_volume,
                    before_pass_mm,
                    delta_path,
                    desc=f'NRRD fullframe bridge delta pass {int(pass_idx)} {model_name}/{view.name}',
                    workers=int(slice_workers),
                    prefer_memory=True,
                    reserve_bytes=32 * GIB,
                )
                layer_ref = materialize_nrrd_view_layer(
                    bridge_delta_mm,
                    model_name=str(model_name),
                    view=view,
                    source='fullframe',
                    mask_kind='bridge',
                    pass_index=int(pass_idx),
                    stage='interpolation',
                    description='Voxels added by this full-frame interpolation pass only.',
                    temp_dir=temp_dir,
                    workers=int(slice_workers),
                )
                if layer_ref is not None:
                    nrrd_layers.append(layer_ref)
                close_memmap_array(bridge_delta_mm)
                if not bool(keep_temp):
                    try:
                        delta_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                close_memmap_array(before_pass_mm)
                if before_pass_path is not None and not bool(keep_temp):
                    try:
                        before_pass_path.unlink(missing_ok=True)
                    except Exception:
                        pass


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

    if bool(nrrd_layers_enabled) and parent_mask_support_mm is not None:
        parent_bridge_support_path = temp_dir / 'nrrd_support' / str(model_name) / view.name / 'fullframe_bridge_support.cvol'
        bridge_stats = subtract_volume_to_raw_bbox_store(
            baseline_native_volume,
            parent_mask_support_mm,
            parent_bridge_support_path,
            desc=f'NRRD support fullframe bridges {model_name}/{view.name}',
            workers=int(slice_workers),
            format_name=CVOL_FORMAT,
        )
        if int(bridge_stats.get('foreground_voxels', 0)) <= 0:
            parent_bridge_support_mm = None
            if parent_bridge_support_path is not None:
                try:
                    shutil.rmtree(parent_bridge_support_path, ignore_errors=True)
                except Exception:
                    pass
        else:
            parent_bridge_support_mm = RawBBoxMaskStore.open(parent_bridge_support_path)

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
        # Keep Radial masks in Radial view-native space here.  The final radial/tilted
        # backprojection queue later assigns at most one set to GPU and one set to CPU.
        final_view_volume = baseline_native_volume
    elif is_tilted_view(view):
        # Keep Tilted masks in Tilted view-native space here so the hybrid final
        # backprojection queue can schedule CPU/GPU work globally across sets.
        final_view_volume = baseline_native_volume
    else:
        final_view_volume = baseline_native_volume

    returned_parent_mask_support = parent_mask_support_mm if (bool(nrrd_layers_enabled) and bool(dense_tiling_active)) else None
    returned_parent_bridge_support = parent_bridge_support_mm if (bool(nrrd_layers_enabled) and bool(dense_tiling_active)) else None
    if parent_mask_support_mm is not None and returned_parent_mask_support is None:
        close_raw_store_or_memmap_volume(parent_mask_support_mm, keep_temp=bool(keep_temp))
    if parent_bridge_support_mm is not None and returned_parent_bridge_support is None:
        close_raw_store_or_memmap_volume(parent_bridge_support_mm, keep_temp=bool(keep_temp))

    return PreparedViewResult(
        model_name=str(model_name),
        view_name=str(view.name),
        native_support_mm=baseline_native_volume,
        final_view_volume_mm=final_view_volume,
        interpolation_stats=interpolation_stats,
        nrrd_layers=nrrd_layers,
        parent_mask_support_mm=returned_parent_mask_support,
        parent_bridge_support_mm=returned_parent_bridge_support,
    )

def gate_tile_volume_against_parent_inplace(
    tile_mask_mm: np.ndarray,
    parent_support_mm: np.ndarray,
    *,
    parent_mask_support_mm: Optional[object] = None,
    parent_bridge_support_mm: Optional[object] = None,
    accepted_by_parent_mask_mm: Optional[np.ndarray] = None,
    accepted_by_parent_bridge_mm: Optional[np.ndarray] = None,
    workers: int = 1,
    desc: str = 'Tile gated OR',
) -> Dict[str, int]:
    """Keep tile components that intersect the frozen parent support on the same slice.

    When parent YOLO-mask and parent-bridge supports are supplied, accepted components are
    also split into two mutually exclusive categories for the decomposed NRRD export.  If a
    component intersects both parent supports, it is assigned to ``parent_mask`` first so the
    category layers can be toggled without double-counting the same tile component.
    """
    num_slices = int(tile_mask_mm.shape[0])
    accepted_components = np.zeros((num_slices,), dtype=np.int64)
    rejected_components = np.zeros((num_slices,), dtype=np.int64)
    accepted_by_parent_mask_components = np.zeros((num_slices,), dtype=np.int64)
    accepted_by_parent_bridge_components = np.zeros((num_slices,), dtype=np.int64)
    kept_voxels = np.zeros((num_slices,), dtype=np.int64)
    accepted_by_parent_mask_voxels = np.zeros((num_slices,), dtype=np.int64)
    accepted_by_parent_bridge_voxels = np.zeros((num_slices,), dtype=np.int64)
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)
    chunk_size = choose_parallel_chunk_size(num_slices, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _process(idx: int) -> None:
        tile_slice = np.asarray(tile_mask_mm[int(idx)], dtype=bool)
        if not np.any(tile_slice):
            tile_mask_mm[int(idx), :, :] = np.uint8(0)
            if accepted_by_parent_mask_mm is not None:
                accepted_by_parent_mask_mm[int(idx), :, :] = np.uint8(0)
            if accepted_by_parent_bridge_mm is not None:
                accepted_by_parent_bridge_mm[int(idx), :, :] = np.uint8(0)
            return

        support_slice = _read_binary_volume_slice_bool(parent_support_mm, int(idx))
        parent_mask_slice = (
            _read_binary_volume_slice_bool(parent_mask_support_mm, int(idx))
            if parent_mask_support_mm is not None else None
        )
        parent_bridge_slice = (
            _read_binary_volume_slice_bool(parent_bridge_support_mm, int(idx))
            if parent_bridge_support_mm is not None else None
        )

        num_labels, labels2d = cv2.connectedComponents(
            np.asarray(tile_slice, dtype=np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        if int(num_labels) <= 1:
            tile_mask_mm[int(idx), :, :] = np.uint8(0)
            if accepted_by_parent_mask_mm is not None:
                accepted_by_parent_mask_mm[int(idx), :, :] = np.uint8(0)
            if accepted_by_parent_bridge_mm is not None:
                accepted_by_parent_bridge_mm[int(idx), :, :] = np.uint8(0)
            return

        keep = np.zeros(tile_slice.shape, dtype=bool)
        mask_category = np.zeros(tile_slice.shape, dtype=bool) if accepted_by_parent_mask_mm is not None else None
        bridge_category = np.zeros(tile_slice.shape, dtype=bool) if accepted_by_parent_bridge_mm is not None else None

        accepted = 0
        rejected = 0
        accepted_mask = 0
        accepted_bridge = 0
        kept_count = 0
        mask_count = 0
        bridge_count = 0

        for comp_lbl in range(1, int(num_labels)):
            comp = labels2d == int(comp_lbl)
            if not np.any(comp):
                continue

            supported_by_mask = bool(parent_mask_slice is not None and np.any(comp & parent_mask_slice))
            supported_by_bridge = bool(parent_bridge_slice is not None and np.any(comp & parent_bridge_slice))
            supported_by_union = bool(np.any(comp & support_slice))
            if not supported_by_union:
                rejected += 1
                continue

            keep[comp] = True
            accepted += 1
            comp_voxels = int(np.count_nonzero(comp))
            kept_count += comp_voxels

            # Mutually exclusive category assignment. Parent mask wins if both supports intersect.
            if supported_by_mask or (parent_mask_slice is None and not supported_by_bridge):
                accepted_mask += 1
                mask_count += comp_voxels
                if mask_category is not None:
                    mask_category[comp] = True
            elif supported_by_bridge:
                accepted_bridge += 1
                bridge_count += comp_voxels
                if bridge_category is not None:
                    bridge_category[comp] = True
            else:
                # Fallback for pathological cases where the supplied category supports do not
                # exactly union to parent_support_mm.
                accepted_mask += 1
                mask_count += comp_voxels
                if mask_category is not None:
                    mask_category[comp] = True

        tile_mask_mm[int(idx), :, :] = keep.astype(np.uint8, copy=False)
        if accepted_by_parent_mask_mm is not None:
            accepted_by_parent_mask_mm[int(idx), :, :] = mask_category.astype(np.uint8, copy=False) if mask_category is not None else np.uint8(0)
        if accepted_by_parent_bridge_mm is not None:
            accepted_by_parent_bridge_mm[int(idx), :, :] = bridge_category.astype(np.uint8, copy=False) if bridge_category is not None else np.uint8(0)

        accepted_components[int(idx)] = np.int64(accepted)
        rejected_components[int(idx)] = np.int64(rejected)
        accepted_by_parent_mask_components[int(idx)] = np.int64(accepted_mask)
        accepted_by_parent_bridge_components[int(idx)] = np.int64(accepted_bridge)
        kept_voxels[int(idx)] = np.int64(kept_count)
        accepted_by_parent_mask_voxels[int(idx)] = np.int64(mask_count)
        accepted_by_parent_bridge_voxels[int(idx)] = np.int64(bridge_count)

    parallel_for_indices_chunked(
        num_slices,
        _process,
        max_workers=worker_count,
        desc=desc,
        show_progress=False,
        chunk_size=chunk_size,
    )
    flush_array(tile_mask_mm)
    if accepted_by_parent_mask_mm is not None:
        flush_array(accepted_by_parent_mask_mm)
    if accepted_by_parent_bridge_mm is not None:
        flush_array(accepted_by_parent_bridge_mm)

    return {
        'accepted_components': int(np.sum(accepted_components, dtype=np.int64)),
        'rejected_components': int(np.sum(rejected_components, dtype=np.int64)),
        'accepted_by_parent_mask_components': int(np.sum(accepted_by_parent_mask_components, dtype=np.int64)),
        'accepted_by_parent_bridge_components': int(np.sum(accepted_by_parent_bridge_components, dtype=np.int64)),
        'kept_voxels': int(np.sum(kept_voxels, dtype=np.int64)),
        'accepted_by_parent_mask_voxels': int(np.sum(accepted_by_parent_mask_voxels, dtype=np.int64)),
        'accepted_by_parent_bridge_voxels': int(np.sum(accepted_by_parent_bridge_voxels, dtype=np.int64)),
    }


def _resize_binary_mask_frame_to_exact_shape_for_tile(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    frame_u8 = (np.asarray(frame, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    if int(frame_u8.shape[0]) == int(out_h) and int(frame_u8.shape[1]) == int(out_w):
        return np.ascontiguousarray(frame_u8)
    scaled = cv2.resize(
        np.ascontiguousarray(frame_u8),
        (int(out_w), int(out_h)),
        interpolation=cv2.INTER_NEAREST,
    )
    return (scaled > 0).astype(np.uint8, copy=False)


def ensure_tile_mask_volume_matches_parent_shape(
    tile_mask_mm: np.ndarray,
    expected_parent_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    desc: str,
    workers: int = 1,
    prefer_memory: bool = True,
) -> np.ndarray:
    """Return a tile mask volume whose shape exactly matches its parent view canvas.

    Prediction-time tile masks should already be inverse-mapped into parent view-native
    coordinates by ``DenseTileJob.M_out_to_src``.  This guard makes the v12.2.0 tile
    waiting-store rule explicit: before a tile can be spilled to the raw waiting store, its volume
    shape must be ``(parent_frames, parent_height, parent_width)``.  If a future code path
    produces a different raster size, the binary slices are nearest-neighbor resized to
    the parent shape before compression/staging.
    """
    src = np.asarray(tile_mask_mm)
    expected = (int(expected_parent_shape[0]), int(expected_parent_shape[1]), int(expected_parent_shape[2]))
    if tuple(int(x) for x in src.shape) == expected:
        return tile_mask_mm
    if src.ndim != 3:
        raise ValueError(f'{desc}: expected a 3D tile mask volume, got shape {src.shape}')

    print(f'{desc}: resizing tile mask volume to parent shape before raw-store spill: {tuple(src.shape)} -> {expected}')
    out = allocate_workspace_array(
        shape=expected,
        dtype=np.uint8,
        path=Path(out_path),
        desc=f'{desc} parent-sized tile mask',
        prefer_memory=bool(prefer_memory),
    )

    in_t = int(src.shape[0])
    out_t, out_h, out_w = expected
    worker_count = choose_slice_parallel_workers(int(workers), max(1, int(out_t)))

    def _resize_out_slice(out_z: int) -> None:
        if int(in_t) == int(out_t):
            restored = _resize_binary_mask_frame_to_exact_shape_for_tile(src[int(out_z)], int(out_h), int(out_w))
        else:
            restored = np.zeros((int(out_h), int(out_w)), dtype=np.uint8)
            for src_z in _restore_source_indices_for_output_z(int(in_t), int(out_t), int(out_z)):
                restored |= _resize_binary_mask_frame_to_exact_shape_for_tile(src[int(src_z)], int(out_h), int(out_w))
        out[int(out_z), :, :] = restored

    parallel_for_indices_chunked(
        int(out_t),
        _resize_out_slice,
        max_workers=worker_count,
        desc=f'{desc}: resize tile mask to parent before compression',
        show_progress=False,
        target_chunks_per_worker=2,
    )
    flush_array(out)
    return out


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
        precleaned_slice_cleanup=bool(task.precleaned_slice_cleanup),
    )

    close_memmap_array(task.tile_confmap_mm)
    if task.tile_confmap_path is not None and not keep_temp:
        try:
            task.tile_confmap_path.unlink(missing_ok=True)
        except Exception:
            pass

    tile_mask_mm = task.tile_mask_mm
    tile_mask_path = task.tile_mask_path
    expected_parent_shape = (int(view.num_slices), int(view.src_h), int(view.src_w))
    parent_sized_path = tile_mask_path.with_name(tile_mask_path.stem + '.parent_sized.u8.dat')
    parent_sized_mm = ensure_tile_mask_volume_matches_parent_shape(
        tile_mask_mm,
        expected_parent_shape,
        parent_sized_path,
        desc=f'{task.model_name}/{task.view_name}/{task.tile_id}',
        workers=int(slice_workers),
        prefer_memory=True,
    )
    if parent_sized_mm is not tile_mask_mm:
        old_mask_path = tile_mask_path
        close_memmap_array(tile_mask_mm)
        if not keep_temp:
            try:
                old_mask_path.unlink(missing_ok=True)
            except Exception:
                pass
        tile_mask_mm = parent_sized_mm
        tile_mask_path = parent_sized_path

    if not _volume_has_foreground(tile_mask_mm):
        close_memmap_array(tile_mask_mm)
        if not keep_temp:
            try:
                tile_mask_path.unlink(missing_ok=True)
            except Exception:
                pass
        return None

    return TilePostprocessResult(
        model_name=str(task.model_name),
        view_name=str(task.view_name),
        config_id=str(task.config_id),
        tile_id=str(task.tile_id),
        tile_mask_mm=tile_mask_mm,
        tile_mask_path=tile_mask_path,
    )



def spill_waiting_tile_result_to_raw_store(
    result: TilePostprocessResult,
    temp_dir: Path,
    *,
    workers: int = 1,
    keep_original: bool = False,
    expected_parent_shape: Optional[Tuple[int, int, int]] = None,
) -> DeferredTilePostprocessResult:
    """Spill a postprocessed waiting tile into the v12.2.0 raw bbox ctile store.

    The function name is retained for stable scheduler call sites. The stored tile
    payload is no longer bitpacked or LZ4-compressed: empty slices are elided,
    nonempty slices are cropped to their nonzero bbox, and the crop is written as
    raw uint8 bytes.
    """
    if result.tile_mask_mm is None:
        raise ValueError('Cannot spill a tile result without a dense tile_mask_mm')

    mask_to_spill = result.tile_mask_mm
    mask_to_spill_path = result.tile_mask_path
    if expected_parent_shape is not None:
        expected = (int(expected_parent_shape[0]), int(expected_parent_shape[1]), int(expected_parent_shape[2]))
        parent_sized_path = temp_dir / 'waiting_tiles_parent_sized' / result.model_name / result.view_name / result.config_id / f'{result.tile_id}.parent_sized.u8.dat'
        parent_sized_mm = ensure_tile_mask_volume_matches_parent_shape(
            mask_to_spill,
            expected,
            parent_sized_path,
            desc=f'Waiting tile spill {result.model_name}/{result.view_name}/{result.tile_id}',
            workers=int(workers),
            prefer_memory=True,
        )
        if parent_sized_mm is not mask_to_spill:
            old_path = mask_to_spill_path
            close_memmap_array(mask_to_spill)
            if not bool(keep_original) and old_path is not None:
                try:
                    Path(old_path).unlink(missing_ok=True)
                except Exception:
                    pass
            mask_to_spill = parent_sized_mm
            mask_to_spill_path = parent_sized_path

    tile_arr = np.asarray(mask_to_spill)
    if tile_arr.ndim != 3:
        raise ValueError(f'Waiting tile spill expects a 3D mask volume, got shape {tile_arr.shape}')
    tile_shape = tuple(int(x) for x in tile_arr.shape)
    if expected_parent_shape is not None and tile_shape != tuple(int(x) for x in expected_parent_shape):
        raise ValueError(f'Waiting tile spill parent-shape guard failed: tile_shape={tile_shape}, expected={tuple(int(x) for x in expected_parent_shape)}')

    store_dir = temp_dir / 'waiting_tiles' / result.model_name / result.view_name / result.config_id / f'{result.tile_id}.ctile'
    write_raw_bbox_mask_store(
        tile_arr,
        store_dir,
        format_name=CTILE_FORMAT,
        desc=f'Waiting tile raw bbox store {result.model_name}/{result.view_name}/{result.config_id}/{result.tile_id}',
        workers=int(workers),
        extra_meta={'waiting_tile_id': str(result.tile_id)},
    )

    flush_array(mask_to_spill)
    close_memmap_array(mask_to_spill)
    if not bool(keep_original) and mask_to_spill_path is not None:
        try:
            Path(mask_to_spill_path).unlink(missing_ok=True)
        except Exception:
            pass

    return DeferredTilePostprocessResult(
        model_name=str(result.model_name),
        view_name=str(result.view_name),
        config_id=str(result.config_id),
        tile_id=str(result.tile_id),
        tile_mask_path=store_dir,
        tile_shape=(int(tile_shape[0]), int(tile_shape[1]), int(tile_shape[2])),
        storage_format=CTILE_FORMAT,
    )

def load_waiting_tile_result_from_raw_store(waiting: DeferredTilePostprocessResult) -> TilePostprocessResult:
    if str(waiting.storage_format) != CTILE_FORMAT:
        raise ValueError(f'Unsupported waiting tile storage format: {waiting.storage_format}')
    store = RawBBoxMaskStore.open(waiting.tile_mask_path)
    if tuple(int(x) for x in store.shape) != tuple(int(x) for x in waiting.tile_shape):
        raise ValueError(f'Raw tile store shape mismatch: store={store.shape}, expected={waiting.tile_shape}')
    return TilePostprocessResult(
        model_name=str(waiting.model_name),
        view_name=str(waiting.view_name),
        config_id=str(waiting.config_id),
        tile_id=str(waiting.tile_id),
        tile_mask_mm=None,
        tile_mask_path=waiting.tile_mask_path,
        tile_mask_store=store,
    )


def _delete_tile_result_storage(result: TilePostprocessResult, *, keep_temp: bool) -> None:
    if bool(keep_temp):
        return
    if result.tile_mask_store is not None:
        result.tile_mask_store.unlink()
        return
    if result.tile_mask_path is not None:
        try:
            Path(result.tile_mask_path).unlink(missing_ok=True)
        except Exception:
            pass


def stage_tile_result_into_config_canvas(
    result: TilePostprocessResult,
    *,
    tile_set_accumulator_mm: np.ndarray,
    tile_set_accumulator_lock: threading.Lock,
    keep_temp: bool,
    slice_workers: int,
) -> Dict[str, int]:
    """Union one cleaned tile into its tile-size/stride canvas before parent gating.

    This implements the v12.2.0 tile-set staging rule: all positions and angle variants for the
    same --tile_size/--tile_stride configuration are first reassembled into one parent-view canvas.
    The consolidated canvas is gated later at the connected-component level after every tile in
    that configuration has either staged or completed empty.
    """
    staged_voxels = 0
    try:
        if result.tile_mask_store is not None:
            store = result.tile_mask_store
            if tuple(int(x) for x in store.shape) != tuple(int(x) for x in np.asarray(tile_set_accumulator_mm).shape):
                raise ValueError(
                    f'Raw tile store shape {store.shape} does not match tile-set accumulator '
                    f'{tuple(int(x) for x in np.asarray(tile_set_accumulator_mm).shape)}'
                )
            z_dim = int(store.shape[0])
            worker_count = choose_slice_parallel_workers(int(slice_workers), z_dim)

            def _decode_and_union(idx: int) -> int:
                tile_slice = store.decode_slice(int(idx), dtype=np.uint8)
                if tile_slice.size == 0 or not np.any(tile_slice):
                    return 0
                count = int(np.count_nonzero(tile_slice))
                with tile_set_accumulator_lock:
                    acc_slice = tile_set_accumulator_mm[int(idx)]
                    acc_slice |= tile_slice.astype(np.uint8, copy=False)
                return count

            if worker_count <= 1:
                for idx in range(z_dim):
                    staged_voxels += int(_decode_and_union(int(idx)))
            else:
                for count in parallel_map_in_order(
                    _decode_and_union,
                    range(z_dim),
                    max_workers=worker_count,
                    max_pending=max(worker_count, worker_count * 2),
                ):
                    staged_voxels += int(count)
            flush_array(tile_set_accumulator_mm)
            return {'staged_voxels': int(staged_voxels), 'source': 'ctile'}

        if result.tile_mask_mm is None:
            return {'staged_voxels': 0, 'source': 'empty'}

        with tile_set_accumulator_lock:
            before_count = None
            if _env_flag('YOLO_TTA_TILE_STAGE_COUNT_EXACT', False):
                before_count = int(np.count_nonzero(tile_set_accumulator_mm))
            union_volume_into_volume(
                tile_set_accumulator_mm,
                result.tile_mask_mm,
                workers=int(slice_workers),
                desc=f'Stage tile into config canvas {result.model_name}/{result.view_name}/{result.config_id}/{result.tile_id}',
            )
            if before_count is None:
                staged_voxels = int(np.count_nonzero(result.tile_mask_mm))
            else:
                staged_voxels = int(np.count_nonzero(tile_set_accumulator_mm)) - int(before_count)
        flush_array(tile_set_accumulator_mm)
        return {'staged_voxels': int(max(0, staged_voxels)), 'source': 'dense'}
    finally:
        close_memmap_array(result.tile_mask_mm)
        _delete_tile_result_storage(result, keep_temp=bool(keep_temp))



def gate_tile_volume_into_consolidated_parent(
    task: TilePostprocessResult,
    *,
    parent_support_mm: np.ndarray,
    tile_accumulator_mm: np.ndarray,
    tile_accumulator_lock: threading.Lock,
    keep_temp: bool,
    slice_workers: int,
    parent_mask_support_mm: Optional[object] = None,
    parent_bridge_support_mm: Optional[object] = None,
    tile_parent_mask_accumulator_mm: Optional[np.ndarray] = None,
    tile_parent_bridge_accumulator_mm: Optional[np.ndarray] = None,
    temp_dir: Optional[Path] = None,
) -> TileGateResult:
    """Gate one consolidated tile-set canvas, then OR accepted components into parent accumulators.

    The input is the already reassembled canvas for one --tile_size/--tile_stride configuration
    across all positions and angle variants of its parent view.  Gating therefore happens at the
    canvas connected-component level, so objects split across tile seams can be accepted as one
    component when any part intersects parent support.  Optional category accumulators are
    populated only for decomposed NRRD export and split accepted tile components into
    parent-YOLO-supported and parent-bridge-supported layers.
    """
    if task.tile_mask_mm is None:
        raise ValueError('Tile-set gate requires a dense staged tile_mask_mm canvas')
    tile_mask_shape = tuple(int(x) for x in np.asarray(task.tile_mask_mm).shape)
    category_enabled = bool(tile_parent_mask_accumulator_mm is not None or tile_parent_bridge_accumulator_mm is not None)
    local_parent_mask_mm: Optional[np.ndarray] = None
    local_parent_bridge_mm: Optional[np.ndarray] = None
    local_parent_mask_path: Optional[Path] = None
    local_parent_bridge_path: Optional[Path] = None

    try:
        if bool(category_enabled):
            if temp_dir is None:
                raise ValueError('temp_dir is required for NRRD tile category gating')
            category_dir = temp_dir / 'nrrd_work' / 'tile_gate_categories' / task.model_name / task.view_name / task.tile_id
            if tile_parent_mask_accumulator_mm is not None:
                local_parent_mask_path = category_dir / 'accepted_by_parent_mask.u8.dat'
                local_parent_mask_mm = allocate_workspace_array(
                    shape=tile_mask_shape,
                    dtype=np.uint8,
                    path=local_parent_mask_path,
                    desc=f'NRRD tile category parent-mask {task.model_name}/{task.view_name}/{task.tile_id}',
                    prefer_memory=True,
                    reserve_bytes=32 * GIB,
                )
            if tile_parent_bridge_accumulator_mm is not None:
                local_parent_bridge_path = category_dir / 'accepted_by_parent_bridge.u8.dat'
                local_parent_bridge_mm = allocate_workspace_array(
                    shape=tile_mask_shape,
                    dtype=np.uint8,
                    path=local_parent_bridge_path,
                    desc=f'NRRD tile category parent-bridge {task.model_name}/{task.view_name}/{task.tile_id}',
                    prefer_memory=True,
                    reserve_bytes=32 * GIB,
                )

        gate_stats = gate_tile_volume_against_parent_inplace(
            task.tile_mask_mm,
            parent_support_mm,
            parent_mask_support_mm=parent_mask_support_mm,
            parent_bridge_support_mm=parent_bridge_support_mm,
            accepted_by_parent_mask_mm=local_parent_mask_mm,
            accepted_by_parent_bridge_mm=local_parent_bridge_mm,
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
                if local_parent_mask_mm is not None and tile_parent_mask_accumulator_mm is not None:
                    union_volume_into_volume(
                        tile_parent_mask_accumulator_mm,
                        local_parent_mask_mm,
                        workers=int(slice_workers),
                        desc=f'Consolidate parent-mask-accepted tile {task.model_name}/{task.view_name}/{task.tile_id}',
                    )
                if local_parent_bridge_mm is not None and tile_parent_bridge_accumulator_mm is not None:
                    union_volume_into_volume(
                        tile_parent_bridge_accumulator_mm,
                        local_parent_bridge_mm,
                        workers=int(slice_workers),
                        desc=f'Consolidate parent-bridge-accepted tile {task.model_name}/{task.view_name}/{task.tile_id}',
                    )

        return TileGateResult(
            model_name=str(task.model_name),
            view_name=str(task.view_name),
            config_id=str(task.config_id),
            tile_id=str(task.tile_id),
            gate_stats={k: int(v) for k, v in gate_stats.items()},
        )
    finally:
        close_memmap_array(local_parent_mask_mm)
        close_memmap_array(local_parent_bridge_mm)
        if not keep_temp:
            for path in (local_parent_mask_path, local_parent_bridge_path):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
        archive_or_delete_binary_volume_storage(
            task.tile_mask_mm,
            keep_temp=bool(keep_temp),
            workers=int(slice_workers),
            desc=f'Gated tile-set canvas {task.model_name}/{task.view_name}/{task.tile_id}',
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
    nrrd_layers_enabled: bool = False,
    tile_parent_mask_accumulator_mm: Optional[np.ndarray] = None,
    tile_parent_bridge_accumulator_mm: Optional[np.ndarray] = None,
) -> TileConsolidationResult:
    """Interpolate the consolidated gated-tile volume once for the parent view, then union it.

    The input accumulator already contains the OR of every accepted tile mask for this parent view.
    Interpolation is now performed once on that consolidated volume instead of once per tile.  When
    NRRD decomposition is enabled, the accepted YOLO tile support is written separately for tiles
    accepted by parent YOLO masks and by parent interpolation bridges; tile interpolation bridges are
    then exported per pass as consolidated tile-bridge layers.
    """
    interpolation_stats: List[Dict[str, object]] = []
    nrrd_layers: List[NrrdLayerRef] = []

    if not _volume_has_foreground(tile_accumulator_mm):
        return TileConsolidationResult(
            model_name=str(model_name),
            view_name=str(view.name),
            interpolation_stats=interpolation_stats,
                nrrd_layers=nrrd_layers,
        )

    if bool(nrrd_layers_enabled):
        emitted_category = False
        if tile_parent_mask_accumulator_mm is not None:
            layer_ref = materialize_nrrd_view_layer(
                tile_parent_mask_accumulator_mm,
                model_name=str(model_name),
                view=view,
                source='tile',
                mask_kind='yolo',
                pass_index=0,
                tile_acceptance='parent_mask',
                stage='pre_tile_interpolation',
                description='Accepted tile YOLO masks whose components intersected parent full-frame YOLO support. Parent-mask support has priority when a component intersects both parent mask and parent bridge.',
                temp_dir=temp_dir,
                workers=int(slice_workers),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)
                emitted_category = True
        if tile_parent_bridge_accumulator_mm is not None:
            layer_ref = materialize_nrrd_view_layer(
                tile_parent_bridge_accumulator_mm,
                model_name=str(model_name),
                view=view,
                source='tile',
                mask_kind='yolo',
                pass_index=0,
                tile_acceptance='parent_bridge',
                stage='pre_tile_interpolation',
                description='Accepted tile YOLO masks whose components did not intersect parent YOLO support but did intersect a parent interpolation bridge.',
                temp_dir=temp_dir,
                workers=int(slice_workers),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)
                emitted_category = True
        if not emitted_category:
            layer_ref = materialize_nrrd_view_layer(
                tile_accumulator_mm,
                model_name=str(model_name),
                view=view,
                source='tile',
                mask_kind='yolo',
                pass_index=0,
                tile_acceptance='parent_support',
                stage='pre_tile_interpolation',
                description='Accepted tile YOLO masks before tile interpolation. Parent mask/bridge category supports were unavailable, so the category is the total parent support.',
                temp_dir=temp_dir,
                workers=int(slice_workers),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)


    if _view_uses_interpolation(view, int(interpolate)):
        total_passes = int(interpolate_passes)
        for pass_idx in range(1, total_passes + 1):
            before_pass_mm: Optional[np.ndarray] = None
            before_pass_path: Optional[Path] = None
            if bool(nrrd_layers_enabled):
                before_pass_path = temp_dir / 'nrrd_work' / str(model_name) / view.name / f'tile_before_pass{int(pass_idx):02d}.u8.dat'
                before_pass_mm = copy_workspace_array(
                    tile_accumulator_mm,
                    before_pass_path,
                    desc=f'NRRD tile before interpolation pass {int(pass_idx)} {model_name}/{view.name}',
                    prefer_memory=True,
                    reserve_bytes=32 * GIB,
                    workers=int(slice_workers),
                )

            tile_accumulator_mm, stats_local = interpolate_view_volume_pass_maybe_process(
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
                wrap_axis=bool(view.family == 'radial'),
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

            if bool(nrrd_layers_enabled) and before_pass_mm is not None:
                delta_path = temp_dir / 'nrrd_work' / str(model_name) / view.name / f'tile_bridge_pass{int(pass_idx):02d}.u8.dat'
                bridge_delta_mm = subtract_volume_to_mmap(
                    tile_accumulator_mm,
                    before_pass_mm,
                    delta_path,
                    desc=f'NRRD tile bridge delta pass {int(pass_idx)} {model_name}/{view.name}',
                    workers=int(slice_workers),
                    prefer_memory=True,
                    reserve_bytes=32 * GIB,
                )
                layer_ref = materialize_nrrd_view_layer(
                    bridge_delta_mm,
                    model_name=str(model_name),
                    view=view,
                    source='tile',
                    mask_kind='bridge',
                    pass_index=int(pass_idx),
                    tile_acceptance='consolidated',
                    stage='tile_interpolation',
                    description='Voxels added by this consolidated tile interpolation pass. Bridges are generated after accepted tile masks are consolidated, so they are not attributed back to parent-mask vs parent-bridge acceptance categories.',
                    temp_dir=temp_dir,
                    workers=int(slice_workers),
                )
                if layer_ref is not None:
                    nrrd_layers.append(layer_ref)
                close_memmap_array(bridge_delta_mm)
                if not bool(keep_temp):
                    try:
                        delta_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                close_memmap_array(before_pass_mm)
                if before_pass_path is not None and not bool(keep_temp):
                    try:
                        before_pass_path.unlink(missing_ok=True)
                    except Exception:
                        pass


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
        nrrd_layers=nrrd_layers,
    )


# --------------------------
# Final Gaussian smoothing
# --------------------------



def gaussian_smoothing_gpu_enabled() -> bool:
    return _env_flag('YOLO_TTA_GAUSSIAN_SMOOTHING_GPU', True)


def gaussian_smoothing_gpu_chunk_mib() -> int:
    return max(256, _env_int('YOLO_TTA_GAUSSIAN_GPU_CHUNK_MIB', 16384))


def _try_get_gpu_gaussian_backend() -> Optional[Tuple[object, object]]:
    try:
        import cupy as cp  # type: ignore
        from cupyx.scipy import ndimage as cpx_ndi  # type: ignore
    except Exception:
        return None
    if not hasattr(cpx_ndi, 'gaussian_filter'):
        return None
    return cp, cpx_ndi


def _gaussian_gpu_core_depth(shape_tyx: Tuple[int, int, int], sigma: float, truncate: float = 4.0) -> Tuple[int, int]:
    z_dim, h_dim, w_dim = (int(shape_tyx[0]), int(shape_tyx[1]), int(shape_tyx[2]))
    halo = max(0, int(math.ceil(float(truncate) * float(sigma))))
    target_bytes = int(gaussian_smoothing_gpu_chunk_mib()) * 1024 * 1024
    bytes_per_slice = max(1, int(h_dim) * int(w_dim) * np.dtype(np.float32).itemsize)
    # cupyx.ndimage may allocate temporaries; budget input/output plus one scratch-equivalent.
    max_read_slices = max(1, int(target_bytes // max(1, bytes_per_slice * 3)))
    core_depth = max(1, int(max_read_slices) - 2 * int(halo))
    return min(max(1, int(core_depth)), max(1, int(z_dim))), int(halo)


def _try_apply_gaussian_smoothing_gpu_chunked_inplace(
    mask_mm: np.ndarray,
    sigma: float,
    passes: int,
    temp_dir: Path,
    *,
    stats: Dict[str, object],
    keep_temp: bool = False,
    workers: int = 1,
    nrrd_layers: Optional[List[NrrdLayerRef]] = None,
    nrrd_model_name: str = 'global',
) -> Optional[Dict[str, object]]:
    """Apply Gaussian smoothing with a CuPy/cupyx chunk+halo implementation.

    Chunks are split along the t/slice axis and include a halo of ceil(truncate*sigma)
    slices on both sides.  Each GPU result writes only its core region back to the
    CPU-backed volume, which makes the chunked result match a whole-volume filter for
    the unchunked Y/X axes and removes seams along the chunked axis.
    """
    if not gaussian_smoothing_gpu_enabled():
        return None
    backend = _try_get_gpu_gaussian_backend()
    if backend is None:
        print('Warning: Gaussian smoothing GPU backend unavailable (CuPy/cupyx.ndimage); falling back to CPU scipy.ndimage.')
        return None
    cp, cpx_ndi = backend

    sigma_f = float(sigma)
    passes_i = int(passes)
    if sigma_f <= 0.0 or passes_i <= 0:
        return stats

    smooth_dir = temp_dir / 'gaussian_smoothing'
    smooth_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(x) for x in np.asarray(mask_mm).shape)
    if len(shape) != 3:
        raise ValueError(f'Gaussian smoothing expects a 3D volume, got shape {shape}')
    num_slices = int(shape[0])
    core_depth, halo = _gaussian_gpu_core_depth(shape, sigma_f, truncate=4.0)
    stats['backend'] = 'cupy_chunked_gpu'
    stats['chunk_core_slices'] = int(core_depth)
    stats['halo_slices'] = int(halo)
    print(
        'Gaussian smoothing GPU backend active: '
        f'shape={shape}, sigma={sigma_f:g}, passes={passes_i}, core_slices={core_depth}, '
        f'halo={halo}, chunk_budget={gaussian_smoothing_gpu_chunk_mib()} MiB'
    )

    for pass_idx in range(1, passes_i + 1):
        print(f'Gaussian smoothing GPU pass {int(pass_idx)}/{int(passes_i)} (sigma={sigma_f:g} voxels)')
        source_path = smooth_dir / f'pass{int(pass_idx):02d}_source.u8.dat'
        source_mm = copy_workspace_array(
            np.asarray(mask_mm, dtype=np.uint8),
            source_path,
            desc=f'Gaussian smoothing GPU pass {int(pass_idx)} source snapshot',
            prefer_memory=False,
            workers=int(workers),
        )
        added_by_slice = np.zeros((num_slices,), dtype=np.int64)
        removed_by_slice = np.zeros((num_slices,), dtype=np.int64)
        try:
            ranges = [(int(z0), int(min(num_slices, z0 + core_depth))) for z0 in range(0, num_slices, core_depth)]
            for z0, z1 in tqdm(ranges, desc=f'Gaussian smoothing GPU pass {int(pass_idx)}: chunked filter'):
                read0 = max(0, int(z0) - int(halo))
                read1 = min(num_slices, int(z1) + int(halo))
                core0 = int(z0) - int(read0)
                core1 = core0 + int(z1 - z0)
                chunk_cpu = np.ascontiguousarray(np.asarray(source_mm[int(read0):int(read1)], dtype=np.float32))
                chunk_gpu = cp.asarray(chunk_cpu)
                smoothed_gpu = cpx_ndi.gaussian_filter(
                    chunk_gpu,
                    sigma=float(sigma_f),
                    mode='constant',
                    cval=0.0,
                    truncate=4.0,
                )
                core_bool = cp.asnumpy(smoothed_gpu[int(core0):int(core1)] >= 0.5).astype(bool, copy=False)
                old_bool = np.asarray(source_mm[int(z0):int(z1)], dtype=bool)
                for local_z in range(int(z1 - z0)):
                    new = core_bool[int(local_z)]
                    old = old_bool[int(local_z)]
                    added_by_slice[int(z0) + int(local_z)] = np.int64(np.count_nonzero(new & (~old)))
                    removed_by_slice[int(z0) + int(local_z)] = np.int64(np.count_nonzero(old & (~new)))
                mask_mm[int(z0):int(z1), :, :] = core_bool.astype(np.uint8, copy=False)
                try:
                    del chunk_gpu, smoothed_gpu, core_bool, chunk_cpu
                    free_all = getattr(cp.get_default_memory_pool(), 'free_all_blocks', None)
                    if callable(free_all):
                        free_all()
                except Exception:
                    pass
        finally:
            close_memmap_array(source_mm)
            if not bool(keep_temp):
                try:
                    source_path.unlink(missing_ok=True)
                except Exception:
                    pass

        flush_array(mask_mm)
        if nrrd_layers is not None:
            layer_ref = materialize_nrrd_global_layer(
                mask_mm,
                model_name=str(nrrd_model_name),
                source='global',
                mask_kind='smoothing_result',
                pass_index=int(pass_idx),
                stage='gaussian_smoothing',
                description='Full-volume result after this GPU Gaussian smoothing pass.',
                temp_dir=temp_dir,
                workers=int(workers),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)

        added = int(np.sum(added_by_slice, dtype=np.int64))
        removed = int(np.sum(removed_by_slice, dtype=np.int64))
        stats[f'pass{int(pass_idx)}_added_voxels'] = int(added)
        stats[f'pass{int(pass_idx)}_removed_voxels'] = int(removed)
        stats['total_added_voxels'] = int(stats.get('total_added_voxels', 0)) + int(added)
        stats['total_removed_voxels'] = int(stats.get('total_removed_voxels', 0)) + int(removed)
        stats['passes_completed'] = int(pass_idx)

    return stats

def apply_gaussian_smoothing_inplace(
    mask_mm: np.ndarray,
    sigma: float,
    passes: int,
    temp_dir: Path,
    *,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    nrrd_layers: Optional[List[NrrdLayerRef]] = None,
    nrrd_model_name: str = 'global',
) -> Dict[str, object]:
    """Smooth the final binary 3D volume with a Gaussian kernel and re-threshold at 0.5.

    The v12.2.0_SLURM smoothing stage is applied after the final view/tile union and optional
    3D void fill, but before --keep_objects and before resizing back to the source geometry.
    A single float32 workspace is reused for every pass to avoid retaining multiple dense
    floating-point copies of the volume.
    """
    sigma_f = float(sigma)
    passes_i = int(passes)
    stats: Dict[str, object] = {
        'enabled': 1 if sigma_f > 0.0 and passes_i > 0 else 0,
        'sigma': float(sigma_f),
        'passes_requested': int(max(0, passes_i)),
        'passes_completed': 0,
        'total_added_voxels': 0,
        'total_removed_voxels': 0,
    }
    if sigma_f <= 0.0 or passes_i <= 0:
        return stats

    gpu_stats = _try_apply_gaussian_smoothing_gpu_chunked_inplace(
        mask_mm,
        sigma_f,
        passes_i,
        temp_dir,
        stats=stats,
        keep_temp=bool(keep_temp),
        workers=int(workers),
        nrrd_layers=nrrd_layers,
        nrrd_model_name=str(nrrd_model_name),
    )
    if gpu_stats is not None:
        return gpu_stats

    stats['backend'] = 'cpu_scipy_ndimage'
    smooth_dir = temp_dir / 'gaussian_smoothing'
    smooth_dir.mkdir(parents=True, exist_ok=True)
    work_path = smooth_dir / 'gaussian_float32.dat'
    work_mm = allocate_workspace_array(
        shape=tuple(int(x) for x in np.asarray(mask_mm).shape),
        dtype=np.float32,
        path=work_path,
        desc='Gaussian smoothing float32 workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    num_slices = int(mask_mm.shape[0])
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)

    try:
        for pass_idx in range(1, passes_i + 1):
            print(f'Gaussian smoothing pass {int(pass_idx)}/{int(passes_i)} (sigma={sigma_f:g} voxels)')

            def _copy_to_float(z: int) -> None:
                work_mm[int(z), :, :] = np.asarray(mask_mm[int(z)], dtype=np.float32)

            parallel_for_indices_chunked(
                num_slices,
                _copy_to_float,
                max_workers=worker_count,
                desc=f'Gaussian smoothing pass {int(pass_idx)}: copy binary volume',
                show_progress=True,
                target_chunks_per_worker=2,
            )
            flush_array(work_mm)

            ndi.gaussian_filter(
                input=work_mm,
                sigma=float(sigma_f),
                output=work_mm,
                mode='constant',
                cval=0.0,
                truncate=4.0,
            )
            flush_array(work_mm)

            added_by_slice = np.zeros((num_slices,), dtype=np.int64)
            removed_by_slice = np.zeros((num_slices,), dtype=np.int64)

            def _threshold_slice(z: int) -> None:
                old = np.asarray(mask_mm[int(z)], dtype=bool)
                new = np.asarray(work_mm[int(z)] >= 0.5, dtype=bool)
                added_by_slice[int(z)] = np.int64(np.count_nonzero(new & (~old)))
                removed_by_slice[int(z)] = np.int64(np.count_nonzero(old & (~new)))
                mask_mm[int(z), :, :] = new.astype(np.uint8, copy=False)

            parallel_for_indices_chunked(
                num_slices,
                _threshold_slice,
                max_workers=worker_count,
                desc=f'Gaussian smoothing pass {int(pass_idx)}: threshold smoothed volume',
                show_progress=True,
                target_chunks_per_worker=2,
            )
            flush_array(mask_mm)

            if nrrd_layers is not None:
                layer_ref = materialize_nrrd_global_layer(
                    mask_mm,
                    model_name=str(nrrd_model_name),
                    source='global',
                    mask_kind='smoothing_result',
                    pass_index=int(pass_idx),
                    stage='gaussian_smoothing',
                    description='Full-volume result after this Gaussian smoothing pass.',
                    temp_dir=temp_dir,
                    workers=int(workers),
                )
                if layer_ref is not None:
                    nrrd_layers.append(layer_ref)

            added = int(np.sum(added_by_slice, dtype=np.int64))
            removed = int(np.sum(removed_by_slice, dtype=np.int64))
            stats[f'pass{int(pass_idx)}_added_voxels'] = int(added)
            stats[f'pass{int(pass_idx)}_removed_voxels'] = int(removed)
            stats['total_added_voxels'] = int(stats.get('total_added_voxels', 0)) + int(added)
            stats['total_removed_voxels'] = int(stats.get('total_removed_voxels', 0)) + int(removed)
            stats['passes_completed'] = int(pass_idx)
    finally:
        close_memmap_array(work_mm)
        if not bool(keep_temp):
            try:
                work_path.unlink(missing_ok=True)
            except Exception:
                pass

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


SKELETON_OVERLAY_BASE_DIM = 3072.0
SKELETON_OVERLAY_BASE_THICKNESS_PX = 10.0


def skeleton_overlay_thickness_px(frame_h: int, frame_w: int) -> int:
    scale = max(float(frame_h), float(frame_w)) / float(SKELETON_OVERLAY_BASE_DIM)
    return max(1, int(round(float(SKELETON_OVERLAY_BASE_THICKNESS_PX) * scale)))


_SKELETON_OVERLAY_KERNEL_CACHE: Dict[int, np.ndarray] = {}


def skeleton_overlay_mask_2d(skeleton_slice: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    skel = np.asarray(skeleton_slice, dtype=np.uint8)
    if skel.shape != (int(frame_h), int(frame_w)):
        skel = cv2.resize(
            (skel > 0).astype(np.uint8, copy=False),
            (int(frame_w), int(frame_h)),
            interpolation=cv2.INTER_NEAREST,
        )
    skel = (skel > 0).astype(np.uint8, copy=False)
    if not np.any(skel):
        return np.zeros((int(frame_h), int(frame_w)), dtype=bool)
    thickness = skeleton_overlay_thickness_px(int(frame_h), int(frame_w))
    if thickness <= 1:
        return skel.astype(bool, copy=False)
    kernel = _SKELETON_OVERLAY_KERNEL_CACHE.get(int(thickness))
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(thickness), int(thickness)))
        _SKELETON_OVERLAY_KERNEL_CACHE[int(thickness)] = kernel
    return cv2.dilate(skel, kernel, iterations=1).astype(bool, copy=False)


def write_overlay_video(
    volume_rgb: np.memmap,  # (T,H,W) gray/luma
    mask_u8: np.ndarray,    # (T,H,W) 0/1
    out_path: Path,
    fps: float,
    skeleton_u8: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> None:
    """Overlay blue masks (50% alpha) and optional red skeleton on transverse frames.

    The working source volume is single-channel; frames are expanded to RGB only for this
    presentation video so the segmentation can remain blue.  When supplied, the skeleton
    is drawn fully opaque red on top of the mask overlay at 10 px for a 3072 px frame,
    scaled isotropically for other output sizes.
    """
    T, H, W = volume_rgb.shape
    assert mask_u8.shape == (T, H, W)
    if skeleton_u8 is not None:
        assert skeleton_u8.shape == (T, H, W)

    proc = ffmpeg_ffv1_rgb_writer(
        out_path,
        width=W,
        height=H,
        fps=fps,
    )

    blue = np.array([0, 0, 255], dtype=np.uint8)  # RGB blue
    red = np.array([255, 0, 0], dtype=np.uint8)   # RGB red

    try:
        assert proc.stdin is not None
        for t in tqdm(range(T), desc=f"Writing overlay video ({out_path.name})", disable=not show_progress):
            frame = _gray_to_rgb_frame(np.asarray(volume_rgb[t]))
            m = mask_u8[t].astype(bool)
            if m.any():
                frame[m] = ((frame[m].astype(np.uint16) + blue.astype(np.uint16)) // 2).astype(np.uint8)
            if skeleton_u8 is not None:
                skel = skeleton_overlay_mask_2d(np.asarray(skeleton_u8[int(t)]), int(H), int(W))
                if np.any(skel):
                    frame[skel] = red
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
    """Write one true bilevel binary TIFF frame with DEFLATE compression.

    tifffile stores bool arrays as 1 bit/sample.  This satisfies the required monob
    TIFF sequence while keeping mask semantics unambiguous: False = black, True = white.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_bool = np.asarray(mask2d, dtype=bool)
    tifffile.imwrite(
        str(out_path),
        mask_bool,
        photometric='minisblack',
        compression='deflate',
    )


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


def _nrrd_ascii_header_text(value: object) -> str:
    """Return ASCII-safe text for pynrrd header fields.

    pynrrd writes headers as ASCII.  View names can contain characters such as the
    degree sign, so normalize those before they reach the header.  The sidecar JSON
    manifest keeps the full Unicode names.
    """
    text = str(value)
    replacements = {
        '°': 'deg',
        '±': '+/-',
        'µ': 'u',
        '–': '-',
        '—': '-',
        '−': '-',
        '\u00a0': ' ',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text


def _nrrd_space_directions_matrix(
    spatial_axes: int = 3,
    list_axis: bool = False,
    list_axis_position: str = 'first',
) -> np.ndarray:
    """Return a NRRD ``space directions`` matrix with optional non-spatial list axis.

    Non-spatial axes are represented by a row of NaNs, which serializes as
    ``none``.  ``list_axis_position='last'`` is used by decomposed segmentation
    NRRDs so the on-disk byte stream is one complete ``(t,Y,X)`` layer block
    followed by the next layer.
    """
    spatial_axes_i = max(1, int(spatial_axes))
    if bool(list_axis):
        position = str(list_axis_position).strip().lower()
        mat = np.full((spatial_axes_i + 1, spatial_axes_i), np.nan, dtype=np.float64)
        if position == 'last':
            mat[:spatial_axes_i, :] = np.eye(spatial_axes_i, dtype=np.float64)
        elif position == 'first':
            mat[1:, :] = np.eye(spatial_axes_i, dtype=np.float64)
        else:
            raise ValueError("list_axis_position must be 'first' or 'last'")
        return mat
    return np.eye(spatial_axes_i, dtype=np.float64)






def nrrd_slicer_header(mask_shape_zyx: Tuple[int, int, int]) -> Dict[str, object]:
    t_dim, h, w = (int(mask_shape_zyx[0]), int(mask_shape_zyx[1]), int(mask_shape_zyx[2]))
    return {
        "space": NRRD_SPACE,
        "kinds": ["domain", "domain", "domain"],
        "space directions": _nrrd_space_directions_matrix(spatial_axes=3, list_axis=False),
        "space origin": np.zeros((3,), dtype=np.float64),
        "content": f"binary segmentation mask; source_shape_tyx=({t_dim},{h},{w}); exported_axes=(X,Y,t)",
    }


def nrrd_pigz_compresslevel() -> int:
    """Compression level passed to pigz for NRRD gzip-encoding."""
    return int(np.clip(_env_int('YOLO_TTA_NRRD_PIGZ_LEVEL', 6), 0, 9))


def nrrd_pigz_threads() -> int:
    default_threads = max(1, min(_cpu_count(), _env_int('YOLO_TTA_NRRD_PIGZ_MAX_THREADS', max(1, _cpu_count()))))
    return max(1, _env_int('YOLO_TTA_NRRD_PIGZ_THREADS', int(default_threads)))


def nrrd_stream_buffer_bytes(required_bytes: Optional[int] = None) -> int:
    """Return the RAM budget for one NRRD streaming payload slab.

    v12.2.1 intentionally avoided materializing the full decomposed NRRD payload.
    v12.2.2 keeps that safety property but uses the hundreds of GiB that are normally
    free under the SLURM allocation to make the streaming slabs much larger.  An
    explicit YOLO_TTA_NRRD_STREAM_BUFFER_MIB still wins; otherwise the buffer is a
    bounded fraction of currently available anonymous memory after a reserve.
    """
    explicit = os.environ.get('YOLO_TTA_NRRD_STREAM_BUFFER_MIB', '').strip()
    if explicit:
        mib = max(1, _env_int('YOLO_TTA_NRRD_STREAM_BUFFER_MIB', 4096))
        target = int(mib) * 1024 * 1024
    else:
        min_mib = max(1, _env_int('YOLO_TTA_NRRD_STREAM_BUFFER_MIN_MIB', 4096))
        reserve_gib = max(1.0, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_RESERVE_GIB', 192.0))
        max_gib = max(1.0, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_MAX_GIB', 384.0))
        fraction = min(0.90, max(0.01, _env_float('YOLO_TTA_NRRD_STREAM_BUFFER_FRACTION', 0.35)))
        avail = int(available_anon_work_bytes())
        usable = max(0, int(avail) - int(reserve_gib * GIB))
        target = int(max(int(min_mib) * 1024 * 1024, min(float(max_gib) * float(GIB), float(usable) * float(fraction))))
    if required_bytes is not None and int(required_bytes) > 0:
        target = min(int(target), int(required_bytes))
    return max(1, int(target))


def nrrd_payload_fill_workers(layer_count: int, z_chunk: int = 1) -> int:
    default_workers = max(1, min(_cpu_count(), int(layer_count) * max(1, int(z_chunk))))
    return max(1, _env_int('YOLO_TTA_NRRD_PAYLOAD_FILL_WORKERS', int(default_workers)))


def _nrrd_full_slice_z_chunk(layer_count: int, width: int, height: int, depth: int) -> int:
    full_slice_bytes = max(1, int(layer_count) * int(width) * int(height) * np.dtype(np.uint8).itemsize)
    full_payload_bytes = full_slice_bytes * max(1, int(depth))
    target = nrrd_stream_buffer_bytes(full_payload_bytes)
    if target < full_slice_bytes:
        return 1
    return max(1, min(int(depth), int(target // full_slice_bytes)))


def nrrd_madvise_dontneed_interval() -> int:
    # 0 disables advisory cache dropping.  The default keeps NRRD source-layer page cache
    # bounded without issuing a syscall after every single slice.
    return max(0, _env_int('YOLO_TTA_NRRD_MADVISE_DONTNEED_INTERVAL', 16))


class _PigzPayloadWriter:
    """Small file-like wrapper that gzip-encodes NRRD payload bytes through pigz."""

    def __init__(self, fh: object) -> None:
        _require_bin('pigz')
        level = int(nrrd_pigz_compresslevel())
        threads = int(nrrd_pigz_threads())
        level_arg = f'-{level}' if level > 0 else '-0'
        cmd = ['pigz', '-c', '-n', '-p', str(threads), level_arg]
        try:
            flush_fn = getattr(fh, 'flush', None)
            if callable(flush_fn):
                flush_fn()
        except Exception:
            pass
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=fh, stderr=subprocess.PIPE)
        if self.proc.stdin is None:
            raise RuntimeError('pigz stdin was not opened')
        self.stdin = self.proc.stdin
        self.cmd = cmd

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self.stdin.closed:
            raise RuntimeError('Cannot write to closed pigz payload stream')
        written = self.stdin.write(data)
        return int(0 if written is None else written)

    def close(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.stdin = None  # type: ignore[assignment]
        _out, err = self.proc.communicate()
        if self.proc.returncode not in (0, None):
            msg = err.decode('utf-8', errors='ignore') if isinstance(err, (bytes, bytearray)) else str(err)
            raise RuntimeError(f'pigz NRRD payload compression failed: {msg}')

    def __enter__(self) -> '_PigzPayloadWriter':
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                if self.proc.stdin is not None and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass


def _open_pigz_payload_writer(fh: object) -> _PigzPayloadWriter:
    return _PigzPayloadWriter(fh)

def _madvise_array_mmap(arr: object, advice_name: str) -> None:
    try:
        import mmap as _mmap_module  # local import so non-POSIX platforms remain unaffected
        advice = getattr(_mmap_module, str(advice_name), None)
        if advice is None:
            return
        mmap_obj = getattr(arr, '_mmap', None)
        if mmap_obj is None:
            base = getattr(arr, 'base', None)
            mmap_obj = getattr(base, '_mmap', None)
        madvise_fn = getattr(mmap_obj, 'madvise', None)
        if callable(madvise_fn):
            madvise_fn(advice)
    except Exception:
        pass


def _nrrd_float_text(value: float) -> str:
    value_f = float(value)
    if math.isnan(value_f):
        return 'none'
    if math.isinf(value_f):
        return 'inf' if value_f > 0 else '-inf'
    return f'{value_f:.17g}'


def _nrrd_vector_text(values: Sequence[object]) -> str:
    return '(' + ','.join(_nrrd_float_text(float(v)) for v in values) + ')'


def _nrrd_space_directions_text(value: object) -> str:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return _nrrd_vector_text(arr.tolist())
    if arr.ndim != 2:
        raise ValueError(f'NRRD space directions must be 1D or 2D, got shape {arr.shape}')
    parts: List[str] = []
    for row in arr:
        if np.all(np.isnan(row)):
            parts.append('none')
        else:
            parts.append(_nrrd_vector_text(row.tolist()))
    return ' '.join(parts)


def _nrrd_header_value_text(key: str, value: object) -> str:
    key_l = str(key).strip().lower()
    if key_l == 'space directions':
        return _nrrd_space_directions_text(value)
    if key_l in {'space origin', 'measurement frame'}:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            return _nrrd_vector_text(arr.tolist())
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 1:
            return ' '.join(_nrrd_float_text(float(v)) for v in arr.tolist())
        return ' '.join(_nrrd_vector_text(row) for row in arr.tolist())
    if isinstance(value, (list, tuple)):
        return ' '.join(_nrrd_ascii_header_text(v) for v in value)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return _nrrd_ascii_header_text(value).replace('\n', ' ').replace('\r', ' ')


_NRRD_STANDARD_FIELDS = {
    'type', 'dimension', 'sizes', 'spacings', 'thicknesses', 'axis mins', 'axis maxs',
    'centers', 'labels', 'units', 'min', 'max', 'old min', 'old max', 'endian',
    'encoding', 'line skip', 'byte skip', 'data file', 'content', 'sample units',
    'space', 'space dimension', 'space units', 'space origin', 'space directions',
    'measurement frame', 'kinds', 'block size',
}


def _nrrd_field_separator(key: str) -> str:
    return ':' if str(key).strip().lower() in _NRRD_STANDARD_FIELDS else ':='


def _write_nrrd_ascii_header(
    fh: object,
    *,
    header: Dict[str, object],
    sizes: Sequence[int],
    dimension: int,
    data_type: str = 'uint8',
    encoding: str = 'gzip',
) -> None:
    header_copy: Dict[str, object] = dict(header)
    for reserved in ('type', 'dimension', 'sizes'):
        header_copy.pop(reserved, None)
    header_copy['encoding'] = str(encoding)

    standard_order = [
        'space', 'space dimension', 'kinds', 'space directions', 'space origin',
        'measurement frame', 'content', 'encoding', 'endian',
    ]

    lines: List[str] = [
        'NRRD0005',
        '# Complete NRRD file generated by GPT-5.5-Pro_v12.2.12_SLURM.py',
        f'type: {str(data_type)}',
        f'dimension: {int(dimension)}',
        'sizes: ' + ' '.join(str(int(v)) for v in sizes),
    ]

    emitted: set[str] = set()
    for key in standard_order:
        if key in header_copy:
            lines.append(f'{key}: {_nrrd_header_value_text(key, header_copy[key])}')
            emitted.add(key)

    for key, value in header_copy.items():
        if key in emitted or key in {'type', 'dimension', 'sizes'}:
            continue
        sep = _nrrd_field_separator(str(key))
        value_text = _nrrd_header_value_text(str(key), value)
        if sep == ':=':
            lines.append(f'{str(key)}{sep}{value_text}')
        else:
            lines.append(f'{str(key)}{sep} {value_text}')

    text = '\n'.join(lines) + '\n\n'
    fh.write(text.encode('ascii', errors='ignore'))



def _nrrd_stream_chunk_rows(layer_count: int, width: int, height: int) -> int:
    required = max(1, int(layer_count) * int(width) * int(height) * np.dtype(np.uint8).itemsize)
    target = max(1, int(nrrd_stream_buffer_bytes(required)))
    bytes_per_row = max(1, int(layer_count) * int(width) * np.dtype(np.uint8).itemsize)
    return max(1, min(int(height), int(target // bytes_per_row)))


def _write_single_nrrd_payload_stream(mask_u8: np.ndarray, payload_writer: object) -> None:
    src = np.asarray(mask_u8)
    if src.ndim != 3:
        raise ValueError(f'NRRD export expects a 3D mask volume with shape (t,Y,X), got {src.shape}')
    t_dim, h_dim, w_dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
    z_chunk = _nrrd_full_slice_z_chunk(1, w_dim, h_dim, t_dim)
    buffer_bytes = int(w_dim) * int(h_dim) * int(z_chunk)
    print(
        f'NRRD streaming single-volume payload: shape=(X,Y,t)=({w_dim},{h_dim},{t_dim}), '
        f'z_chunk={int(z_chunk)}, buffer~{buffer_bytes / GIB:.3f} GiB, '
        f'pigz_level={nrrd_pigz_compresslevel()}, pigz_threads={nrrd_pigz_threads()}'
    )
    for z0 in tqdm(range(0, t_dim, int(z_chunk)), desc='NRRD streaming: single volume'):
        z1 = min(t_dim, int(z0) + int(z_chunk))
        chunk = np.asarray(src[int(z0):int(z1)], dtype=np.uint8)
        if chunk.flags['C_CONTIGUOUS']:
            payload_writer.write(memoryview(chunk).cast('B'))
        else:
            payload_writer.write(np.ascontiguousarray(chunk).tobytes(order='C'))


def write_nrrd(mask_u8: np.ndarray, out_path: Path) -> Path:
    """Write a binary 3D NRRD using bounded-memory streaming pigz gzip-encoded output.

    This replaces the previous pynrrd path that first transposed the whole volume into a
    second in-memory payload.  Data are written in NRRD/Slicer spatial axis order
    ``(X,Y,t)`` with the first axis varying fastest.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = np.asarray(mask_u8)
    if src.ndim != 3:
        raise ValueError(f'NRRD export expects a 3D mask volume with shape (t,Y,X), got {src.shape}')
    t_dim, h_dim, w_dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
    header = nrrd_slicer_header((t_dim, h_dim, w_dim))
    with open(out_path, 'wb') as fh:
        _write_nrrd_ascii_header(
            fh,
            header=header,
            sizes=(w_dim, h_dim, t_dim),
            dimension=3,
            data_type='unsigned char',
            encoding='gzip',
        )
        with _open_pigz_payload_writer(fh) as payload_writer:
            _write_single_nrrd_payload_stream(mask_u8, payload_writer)
    return out_path


def _nrrd_layer_ref_is_raw_bbox_store(ref: NrrdLayerRef) -> bool:
    return str(getattr(ref, 'storage_format', 'raw_u8')) in MASK_STORE_FORMATS or Path(ref.path).is_dir()


def _open_nrrd_layer_ref(ref: NrrdLayerRef) -> object:
    if _nrrd_layer_ref_is_raw_bbox_store(ref):
        return RawBBoxMaskStore.open(
            ref.path,
            cache_payload_in_ram=bool(nrrd_cache_raw_bbox_layers_in_ram_enabled()),
        )
    return np.memmap(
        ref.path,
        dtype=np.dtype(ref.dtype),
        mode='r',
        shape=tuple(int(x) for x in ref.shape),
    )


def _close_nrrd_layer_source(src: object) -> None:
    if isinstance(src, RawBBoxMaskStore):
        src.close()
        return
    close_memmap_array(src)


def _drop_nrrd_raw_store_chunks_ram_cache(src: object) -> None:
    if not isinstance(src, RawBBoxMaskStore):
        return
    try:
        cache_key = src.chunks_path.resolve()
    except Exception:
        cache_key = src.chunks_path
    _NRRD_RAW_STORE_CHUNKS_RAM_CACHE.pop(cache_key, None)


def _log_nrrd_streaming_sources(effective_refs: Sequence[NrrdLayerRef]) -> None:
    raw_bbox = [ref for ref in effective_refs if _nrrd_layer_ref_is_raw_bbox_store(ref)]
    raw = [ref for ref in effective_refs if not _nrrd_layer_ref_is_raw_bbox_store(ref)]
    print(
        'NRRD streaming source files used by decomposed-layer payload packaging: '
        f'{len(effective_refs)} layer backing path(s); raw_bbox_store={len(raw_bbox)}, raw_u8={len(raw)}, '
        f'raw_bbox_payload_ram_cache_per_layer={nrrd_cache_raw_bbox_layers_in_ram_enabled()}, '
        f'precomputed_segment_extents={nrrd_precomputed_segment_extents_enabled()}'
    )
    preview = list(effective_refs[:8])
    for ref in preview:
        print(f'  - {ref.storage_format}: {ref.path}')
    if len(effective_refs) > len(preview):
        print(f'  ... {len(effective_refs) - len(preview)} additional layer backing path(s) listed in the NRRD manifest sidecar')


def compute_segment_extent_zyx(mask_zyx: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """Return Slicer SegmentN_Extent as ``minI maxI minJ maxJ minK maxK``.

    The pipeline layer is ``(t,Y,X)``.  The NRRD payload stores that as spatial
    ``(X,Y,t)``, so Slicer I/J/K map to X/Y/t respectively.
    """
    src = np.asarray(mask_zyx)
    if src.ndim != 3:
        raise ValueError(f'compute_segment_extent_zyx expects a 3D (t,Y,X) layer, got {src.shape}')

    t_dim, h_dim, w_dim = (int(src.shape[0]), int(src.shape[1]), int(src.shape[2]))
    min_t, max_t = t_dim, -1
    min_y, max_y = h_dim, -1
    min_x, max_x = w_dim, -1

    for t_idx in range(t_dim):
        sl = np.asarray(src[int(t_idx)], dtype=bool)
        if not np.any(sl):
            continue
        row_has_fg = np.any(sl, axis=1)
        col_has_fg = np.any(sl, axis=0)
        ys = np.flatnonzero(row_has_fg)
        xs = np.flatnonzero(col_has_fg)
        if xs.size <= 0 or ys.size <= 0:
            continue
        min_t = min(min_t, int(t_idx))
        max_t = max(max_t, int(t_idx))
        min_y = min(min_y, int(ys[0]))
        max_y = max(max_y, int(ys[-1]))
        min_x = min(min_x, int(xs[0]))
        max_x = max(max_x, int(xs[-1]))

    if max_t < 0:
        return _nrrd_empty_segment_extent()
    return (int(min_x), int(max_x), int(min_y), int(max_y), int(min_t), int(max_t))


def _format_segment_extent(extent: Sequence[int]) -> str:
    values = [int(v) for v in extent]
    if len(values) != 6:
        raise ValueError(f'Segment extent must contain 6 values, got {values}')
    return ' '.join(str(v) for v in values)


def _nrrd_segment_color(idx: int) -> str:
    hue = (float(idx) * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
    return f'{r:.6g} {g:.6g} {b:.6g}'


def nrrd_decomposed_header(
    *,
    output_shape_zyx: Tuple[int, int, int],
    layer_refs: Sequence[NrrdLayerRef],
    layer_extents: Optional[Sequence[Tuple[int, int, int, int, int, int]]] = None,
) -> Dict[str, object]:
    t_dim, h, w = (int(output_shape_zyx[0]), int(output_shape_zyx[1]), int(output_shape_zyx[2]))
    if layer_extents is None:
        extents_resolved = [_nrrd_empty_segment_extent() for _ in layer_refs]
    else:
        extents_resolved = [tuple(int(v) for v in extent) for extent in layer_extents]
    if len(extents_resolved) != len(layer_refs):
        raise ValueError(f'NRRD segment extent count {len(extents_resolved)} does not match layer count {len(layer_refs)}')

    header: Dict[str, object] = {
        'space': NRRD_SPACE,
        'kinds': ['domain', 'domain', 'domain', 'list'],
        # The trailing list axis is non-spatial.  Use a NaN row so it serializes
        # as ``none`` after the three spatial direction vectors.
        'space directions': _nrrd_space_directions_matrix(spatial_axes=3, list_axis=True, list_axis_position='last'),
        'space origin': np.zeros((3,), dtype=np.float64),
        'content': (
            'decomposed binary segmentation layers; '
            f'source_shape_tyx=({t_dim},{h},{w}); exported_axes=(X,Y,t,layer); '
            'layer metadata stored in SegmentN_* fields and sidecar manifest JSON'
        ),
        'encoding': 'gzip',
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|',
        'Segmentation_SourceRepresentation': 'Binary labelmap',
        # Kept for compatibility with older Slicer segmentation metadata readers.
        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_ReferenceImageExtentOffset': '0 0 0',
    }
    for idx, ref in enumerate(layer_refs):
        tag_parts = [
            f'model:{ref.model_name}',
            f'view:{ref.view_name}',
            f'view_family:{ref.view_family}',
            f'source:{ref.source}',
            f'mask_kind:{ref.mask_kind}',
            f'pass:{int(ref.pass_index)}',
        ]
        if ref.tile_acceptance:
            tag_parts.append(f'tile_acceptance:{ref.tile_acceptance}')
        if ref.stage:
            tag_parts.append(f'stage:{ref.stage}')
        header[f'Segment{idx}_ID'] = _nrrd_ascii_header_text(ref.key)
        header[f'Segment{idx}_Name'] = _nrrd_ascii_header_text(ref.name)
        header[f'Segment{idx}_NameAutoGenerated'] = '0'
        header[f'Segment{idx}_Color'] = _nrrd_segment_color(int(idx))
        header[f'Segment{idx}_ColorAutoGenerated'] = '0'
        header[f'Segment{idx}_Extent'] = _format_segment_extent(extents_resolved[int(idx)])
        header[f'Segment{idx}_Layer'] = str(idx)
        header[f'Segment{idx}_LabelValue'] = '1'
        header[f'Segment{idx}_Tags'] = _nrrd_ascii_header_text('|'.join(tag_parts))
        header[f'Segment{idx}_Description'] = _nrrd_ascii_header_text(ref.description)
    return header




def _restore_source_indices_for_output_z(in_t: int, out_t: int, out_z: int) -> List[int]:
    in_t_i = max(1, int(in_t))
    out_t_i = max(1, int(out_t))
    out_z_i = int(np.clip(int(out_z), 0, out_t_i - 1))
    if in_t_i >= out_t_i:
        src_start = int(math.floor(float(out_z_i) * float(in_t_i) / float(out_t_i)))
        src_stop = int(math.ceil(float(out_z_i + 1) * float(in_t_i) / float(out_t_i)))
        src_start = int(np.clip(src_start, 0, in_t_i - 1))
        src_stop = int(np.clip(max(src_start + 1, src_stop), 1, in_t_i))
        return [int(v) for v in range(src_start, src_stop)]

    src_z = _linear_source_index(out_z_i, out_t_i, in_t_i)
    return [int(np.clip(int(round(src_z)), 0, in_t_i - 1))]


def _resize_binary_mask_frame_to_output_shape(frame: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    frame_u8 = (np.asarray(frame, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    if int(frame_u8.shape[0]) == int(out_h) and int(frame_u8.shape[1]) == int(out_w):
        return np.ascontiguousarray(frame_u8)
    interp = cv2.INTER_AREA if (int(frame_u8.shape[0]) >= int(out_h) and int(frame_u8.shape[1]) >= int(out_w)) else cv2.INTER_NEAREST
    scaled = cv2.resize(
        np.ascontiguousarray(frame_u8 * np.uint8(255)),
        (int(out_w), int(out_h)),
        interpolation=int(interp),
    )
    return (scaled > 0).astype(np.uint8, copy=False)


def _read_layer_slice_in_output_shape(
    src: object,
    output_shape: Tuple[int, int, int],
    out_z: int,
) -> np.ndarray:
    """Return one output-geometry ``(Y,X)`` slice for an NRRD layer without materializing the full layer."""
    in_t, in_h, in_w = _volume_shape_tuple(src)
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    out_z_i = int(np.clip(int(out_z), 0, out_t - 1))

    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return _read_binary_volume_slice_u8(src, out_z_i)

    source_indices = _restore_source_indices_for_output_z(in_t, out_t, out_z_i)
    if in_h == out_h and in_w == out_w:
        if len(source_indices) == 1:
            return _read_binary_volume_slice_u8(src, int(source_indices[0]))
        restored = np.zeros((out_h, out_w), dtype=np.uint8)
        for src_idx in source_indices:
            restored |= _read_binary_volume_slice_bool(src, int(src_idx)).astype(np.uint8, copy=False)
        return restored

    restored = np.zeros((out_h, out_w), dtype=np.uint8)
    for src_idx in source_indices:
        restored |= _resize_binary_mask_frame_to_output_shape(_read_binary_volume_slice_u8(src, int(src_idx)), out_h, out_w)
    return restored


def _read_layer_row_block_in_output_shape(
    src: object,
    output_shape: Tuple[int, int, int],
    out_z: int,
    y0: int,
    y1: int,
) -> np.ndarray:
    """Return a ``(rows,X)`` block in final output geometry for streaming NRRD writes."""
    in_t, in_h, in_w = _volume_shape_tuple(src)
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    y0_i = int(np.clip(int(y0), 0, out_h))
    y1_i = int(np.clip(int(y1), y0_i, out_h))
    out_z_i = int(np.clip(int(out_z), 0, out_t - 1))

    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return _read_binary_volume_slice_u8(src, out_z_i)[y0_i:y1_i, :]

    source_indices = _restore_source_indices_for_output_z(in_t, out_t, out_z_i)
    if in_h == out_h and in_w == out_w:
        if len(source_indices) == 1:
            return _read_binary_volume_slice_u8(src, int(source_indices[0]))[y0_i:y1_i, :]
        restored = np.zeros((y1_i - y0_i, out_w), dtype=np.uint8)
        for src_idx in source_indices:
            restored |= _read_binary_volume_slice_bool(src, int(src_idx))[y0_i:y1_i, :].astype(np.uint8, copy=False)
        return restored

    # Rare fallback for XY geometry changes.  Keep the code simple and still bounded: one
    # full output slice is materialized for one layer at a time, not for every layer.
    return _read_layer_slice_in_output_shape(src, output_shape, out_z_i)[y0_i:y1_i, :]


def nrrd_precomputed_segment_extents_enabled() -> bool:
    return _env_flag('YOLO_TTA_NRRD_PRECOMPUTED_SEGMENT_EXTENTS', True)


def _map_spatial_extent_conservative(
    min_idx: int,
    max_idx: int,
    in_len: int,
    out_len: int,
) -> Tuple[int, int]:
    if int(out_len) <= 0 or int(in_len) <= 0 or int(max_idx) < int(min_idx):
        return (0, -1)
    in_len_i = int(in_len)
    out_len_i = int(out_len)
    min_i = int(np.clip(int(min_idx), 0, in_len_i - 1))
    max_i = int(np.clip(int(max_idx), 0, in_len_i - 1))
    if in_len_i == out_len_i:
        return (min_i, max_i)
    # Conservative interval-overlap mapping for cv2 area/nearest resizing. It may include a
    # one-pixel border in unusual scale ratios, but it never underestimates the layer extent.
    out_min = int(math.floor(float(min_i) * float(out_len_i) / float(in_len_i)))
    out_max = int(math.ceil(float(max_i + 1) * float(out_len_i) / float(in_len_i)) - 1)
    out_min = int(np.clip(out_min, 0, out_len_i - 1))
    out_max = int(np.clip(out_max, 0, out_len_i - 1))
    if out_max < out_min:
        return (0, -1)
    return (out_min, out_max)


def _map_t_extent_for_nrrd_restore(
    min_t: int,
    max_t: int,
    in_t: int,
    out_t: int,
) -> Tuple[int, int]:
    if int(out_t) <= 0 or int(in_t) <= 0 or int(max_t) < int(min_t):
        return (0, -1)
    in_t_i = int(in_t)
    out_t_i = int(out_t)
    lo = int(np.clip(int(min_t), 0, in_t_i - 1))
    hi = int(np.clip(int(max_t), 0, in_t_i - 1))
    if in_t_i == out_t_i:
        return (lo, hi)

    out_min: Optional[int] = None
    out_max = -1
    for out_z in range(out_t_i):
        src_indices = _restore_source_indices_for_output_z(in_t_i, out_t_i, int(out_z))
        if any(lo <= int(src_idx) <= hi for src_idx in src_indices):
            if out_min is None:
                out_min = int(out_z)
            out_max = int(out_z)
    if out_min is None or out_max < out_min:
        return (0, -1)
    return (int(out_min), int(out_max))


def _transform_segment_extent_to_output_shape(
    extent: Sequence[int],
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
) -> Optional[NrrdSegmentExtent]:
    extent_i = _coerce_segment_extent(extent)
    if extent_i is None:
        return None
    if _segment_extent_is_empty(extent_i):
        return _nrrd_empty_segment_extent()

    in_t, in_h, in_w = (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]))
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return extent_i

    x0, x1, y0, y1, t0, t1 = (int(v) for v in extent_i)
    mapped_t0, mapped_t1 = _map_t_extent_for_nrrd_restore(t0, t1, in_t, out_t)
    mapped_y0, mapped_y1 = _map_spatial_extent_conservative(y0, y1, in_h, out_h)
    mapped_x0, mapped_x1 = _map_spatial_extent_conservative(x0, x1, in_w, out_w)
    mapped = (int(mapped_x0), int(mapped_x1), int(mapped_y0), int(mapped_y1), int(mapped_t0), int(mapped_t1))
    if _segment_extent_is_empty(mapped):
        return _nrrd_empty_segment_extent()
    return mapped


def _read_raw_store_segment_extent(ref: NrrdLayerRef) -> Optional[NrrdSegmentExtent]:
    if not _nrrd_layer_ref_is_raw_bbox_store(ref):
        return None
    meta_path = Path(ref.path) / 'meta.json'
    index_path = Path(ref.path) / 'index.bin'
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    extent = _coerce_segment_extent(meta.get('segment_extent_ijk'))
    if extent is not None:
        return extent
    try:
        shape = meta.get('shape', [int(ref.shape[0]), int(ref.shape[1]), int(ref.shape[2])])
        shape_i = (int(shape[0]), int(shape[1]), int(shape[2]))
        if index_path.exists():
            index = np.fromfile(index_path, dtype=CTILE_INDEX_DTYPE, count=int(shape_i[0]))
            return _raw_store_index_segment_extent(index, shape_i)
    except Exception:
        return None
    return None


def _precomputed_segment_extent_for_layer_ref(
    ref: NrrdLayerRef,
    output_shape: Tuple[int, int, int],
) -> Optional[Tuple[NrrdSegmentExtent, str]]:
    if not nrrd_precomputed_segment_extents_enabled():
        return None
    stored_extent = _coerce_segment_extent(getattr(ref, 'segment_extent_ijk', None))
    source = str(getattr(ref, 'segment_extent_source', '') or 'layer_ref_metadata')
    if stored_extent is None:
        stored_extent = _read_raw_store_segment_extent(ref)
        source = 'raw_bbox_store_metadata'
    if stored_extent is None:
        return None
    transformed = _transform_segment_extent_to_output_shape(
        stored_extent,
        tuple(int(x) for x in ref.shape),
        tuple(int(x) for x in output_shape),
    )
    if transformed is None:
        return None
    if tuple(int(x) for x in ref.shape) != tuple(int(x) for x in output_shape):
        source += '_geometry_mapped'
    return transformed, source


def _scan_segment_extent_for_layer_ref(
    ref: NrrdLayerRef,
    output_shape: Tuple[int, int, int],
) -> NrrdSegmentExtent:
    src = _open_nrrd_layer_ref(ref)
    try:
        if isinstance(src, np.ndarray) and tuple(int(x) for x in ref.shape) == tuple(int(x) for x in output_shape):
            return compute_segment_extent_zyx(src)

        out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
        min_t, max_t = out_t, -1
        min_y, max_y = out_h, -1
        min_x, max_x = out_w, -1
        for z in range(out_t):
            sl = np.asarray(_read_layer_slice_in_output_shape(src, output_shape, int(z)), dtype=bool)
            if not np.any(sl):
                continue
            row_has_fg = np.any(sl, axis=1)
            col_has_fg = np.any(sl, axis=0)
            ys = np.flatnonzero(row_has_fg)
            xs = np.flatnonzero(col_has_fg)
            if xs.size <= 0 or ys.size <= 0:
                continue
            min_t = min(min_t, int(z))
            max_t = max(max_t, int(z))
            min_y = min(min_y, int(ys[0]))
            max_y = max(max_y, int(ys[-1]))
            min_x = min(min_x, int(xs[0]))
            max_x = max(max_x, int(xs[-1]))

        if max_t < 0:
            return _nrrd_empty_segment_extent()
        return (int(min_x), int(max_x), int(min_y), int(max_y), int(min_t), int(max_t))
    finally:
        _madvise_array_mmap(src, 'MADV_DONTNEED')
        _close_nrrd_layer_source(src)


def _resolve_segment_extent_for_layer_ref(
    ref: NrrdLayerRef,
    output_shape: Tuple[int, int, int],
) -> Tuple[NrrdSegmentExtent, str]:
    precomputed = _precomputed_segment_extent_for_layer_ref(ref, output_shape)
    if precomputed is not None:
        return precomputed
    return _scan_segment_extent_for_layer_ref(ref, output_shape), 'fallback_layer_scan'


# DEAD_CODE_MARKER(v12.2.0-post-refactor): compatibility wrapper is no longer called after direct extent/source resolution; keep for one release but do not prune automatically.
def _compute_segment_extent_for_layer_ref(
    ref: NrrdLayerRef,
    output_shape: Tuple[int, int, int],
) -> NrrdSegmentExtent:
    return _resolve_segment_extent_for_layer_ref(ref, output_shape)[0]

def _write_decomposed_nrrd_payload_stream(
    effective_refs: Sequence[NrrdLayerRef],
    output_shape: Tuple[int, int, int],
    payload_writer: object,
) -> None:
    """Stream decomposed ``(X,Y,t,layer)`` payload one complete layer at a time.

    NRRD stores axis 0 as the fastest-varying axis.  With sizes ``(X,Y,t,layer)``,
    each layer occupies one contiguous on-disk block whose byte order matches the
    pipeline's native ``(t,Y,X)`` C-order buffer.  This avoids the old
    layer-fastest voxel interleave and the per-slice ``(Y,X)->(X,Y)`` transpose.
    """
    out_t, out_h, out_w = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    layer_count = int(len(effective_refs))
    if layer_count <= 0:
        return

    z_chunk = _nrrd_full_slice_z_chunk(1, out_w, out_h, out_t)
    layer_slice_bytes = int(out_w) * int(out_h) * np.dtype(np.uint8).itemsize
    buffer_bytes = int(layer_slice_bytes) * int(z_chunk)
    print(
        f'NRRD streaming decomposed payload: layers={layer_count}, shape=(X,Y,t,layer)='
        f'({out_w},{out_h},{out_t},{layer_count}), z_chunk={z_chunk}, '
        f'one_layer_buffer~{buffer_bytes / GIB:.3f} GiB, '
        f'pigz_level={nrrd_pigz_compresslevel()}, pigz_threads={nrrd_pigz_threads()}'
    )

    madvise_interval = nrrd_madvise_dontneed_interval()
    with tqdm(total=int(layer_count) * int(out_t), desc='NRRD streaming: decomposed layer blocks') as pbar:
        for layer_idx, ref in enumerate(effective_refs):
            src: Optional[object] = None
            try:
                src = _open_nrrd_layer_ref(ref)
                _madvise_array_mmap(src, 'MADV_SEQUENTIAL')
                in_t, in_h, in_w = _volume_shape_tuple(src)
                direct_native_stream = (
                    (int(in_t), int(in_h), int(in_w)) == (int(out_t), int(out_h), int(out_w))
                    and not isinstance(src, RawBBoxMaskStore)
                )
                raw_store_native_stream = (
                    isinstance(src, RawBBoxMaskStore)
                    and (int(in_t), int(in_h), int(in_w)) == (int(out_t), int(out_h), int(out_w))
                )

                if bool(direct_native_stream):
                    # The memmap is already a native (t,Y,X) C-order layer.  Write chunks of
                    # it directly; no transpose, no Fortran conversion, and no layer interleave.
                    for z0 in range(0, out_t, int(z_chunk)):
                        z1 = min(out_t, int(z0) + int(z_chunk))
                        chunk = np.asarray(src[int(z0):int(z1)], dtype=np.uint8)
                        if chunk.flags['C_CONTIGUOUS']:
                            payload_writer.write(memoryview(chunk).cast('B'))
                        else:
                            payload_writer.write(np.ascontiguousarray(chunk).tobytes(order='C'))
                        pbar.update(int(z1 - z0))
                        if madvise_interval > 0 and (int(z1) % int(madvise_interval) == 0):
                            _madvise_array_mmap(src, 'MADV_DONTNEED')
                    continue

                if bool(raw_store_native_stream):
                    # Target_Dummy layout keeps the list axis last, so one complete layer is a
                    # native (t,Y,X) byte block.  For cvol/ctile sources, fill the reusable block
                    # directly from each slice bbox payload instead of allocating one full decoded
                    # zeros slice per frame and copying it again into the NRRD write buffer.
                    try:
                        layer_chunk = np.empty((int(z_chunk), int(out_h), int(out_w)), dtype=np.uint8, order='C')
                    except MemoryError:
                        if int(z_chunk) != 1:
                            print(
                                f'Warning: requested raw-bbox NRRD chunk allocation failed for layer {int(layer_idx)}; '
                                'falling back to one output t-slice at a time.'
                            )
                        z_chunk = 1
                        layer_chunk = np.empty((1, int(out_h), int(out_w)), dtype=np.uint8, order='C')

                    for z0 in range(0, out_t, int(z_chunk)):
                        z1 = min(out_t, int(z0) + int(z_chunk))
                        z_count = int(z1 - z0)
                        block = layer_chunk[:z_count, :, :]
                        for zi, z in enumerate(range(int(z0), int(z1))):
                            src.fill_decoded_slice_into(int(z), block[int(zi)])
                        if block.flags['C_CONTIGUOUS']:
                            payload_writer.write(memoryview(block).cast('B'))
                        else:
                            payload_writer.write(np.ascontiguousarray(block).tobytes(order='C'))
                        pbar.update(int(z_count))
                    continue

                try:
                    layer_chunk = np.empty((int(z_chunk), int(out_h), int(out_w)), dtype=np.uint8, order='C')
                except MemoryError:
                    if int(z_chunk) != 1:
                        print(
                            f'Warning: requested one-layer NRRD chunk allocation failed for layer {int(layer_idx)}; '
                            'falling back to one output t-slice at a time.'
                        )
                    z_chunk = 1
                    layer_chunk = np.empty((1, int(out_h), int(out_w)), dtype=np.uint8, order='C')

                for z0 in range(0, out_t, int(z_chunk)):
                    z1 = min(out_t, int(z0) + int(z_chunk))
                    z_count = int(z1 - z0)
                    block = layer_chunk[:z_count, :, :]
                    for zi, z in enumerate(range(int(z0), int(z1))):
                        slice_yx = _read_layer_slice_in_output_shape(src, output_shape, int(z))
                        block[int(zi), :, :] = np.asarray(slice_yx, dtype=np.uint8)
                    payload_writer.write(block.tobytes(order='C'))
                    pbar.update(int(z_count))
                    if madvise_interval > 0 and (int(z1) % int(madvise_interval) == 0):
                        _madvise_array_mmap(src, 'MADV_DONTNEED')
            finally:
                if src is not None:
                    _madvise_array_mmap(src, 'MADV_DONTNEED')
                    _close_nrrd_layer_source(src)
                    _drop_nrrd_raw_store_chunks_ram_cache(src)


def _write_decomposed_nrrd_manifest(
    *,
    out_path: Path,
    output_shape: Tuple[int, int, int],
    effective_refs: Sequence[NrrdLayerRef],
    layer_extents: Optional[Sequence[NrrdSegmentExtent]] = None,
    layer_extent_sources: Optional[Sequence[str]] = None,
) -> None:
    extents_resolved: List[NrrdSegmentExtent] = []
    if layer_extents is None:
        extents_resolved = [_nrrd_empty_segment_extent() for _ in effective_refs]
    else:
        extents_resolved = [(_coerce_segment_extent(extent) or _nrrd_empty_segment_extent()) for extent in layer_extents]
    sources_resolved = list(layer_extent_sources) if layer_extent_sources is not None else ['unknown'] * len(effective_refs)
    if len(sources_resolved) != len(effective_refs):
        sources_resolved = ['unknown'] * len(effective_refs)

    manifest = {
        'nrrd_path': str(out_path),
        'axis_order': '(X, Y, t, layer)',
        'internal_layer_order': '(t, Y, X)',
        'output_shape_tyx': [int(output_shape[0]), int(output_shape[1]), int(output_shape[2])],
        'layer_count': int(len(effective_refs)),
        'layers': [
            {
                'index': int(idx),
                'key': ref.key,
                'name': ref.name,
                'model_name': ref.model_name,
                'view_name': ref.view_name,
                'view_family': ref.view_family,
                'source': ref.source,
                'mask_kind': ref.mask_kind,
                'pass_index': int(ref.pass_index),
                'tile_acceptance': ref.tile_acceptance,
                'stage': ref.stage,
                'description': ref.description,
                'storage_format': getattr(ref, 'storage_format', 'raw_u8'),
                'backing_path': str(ref.path),
                'backing_shape_tyx': [int(ref.shape[0]), int(ref.shape[1]), int(ref.shape[2])],
                'stored_segment_extent_ijk': (
                    _segment_extent_to_json(ref.segment_extent_ijk)
                    if getattr(ref, 'segment_extent_ijk', None) is not None else None
                ),
                'stored_segment_extent_shape_tyx': [
                    int(v) for v in (
                        getattr(ref, 'segment_extent_shape_tyx', None)
                        or (int(ref.shape[0]), int(ref.shape[1]), int(ref.shape[2]))
                    )
                ],
                'stored_segment_extent_source': str(getattr(ref, 'segment_extent_source', '')),
                'nrrd_output_segment_extent_ijk': _segment_extent_to_json(extents_resolved[int(idx)]),
                'segment_extent_source': str(sources_resolved[int(idx)]),
            }
            for idx, ref in enumerate(effective_refs)
        ],
        'notes': [
            'YOLO layers are cleaned masks before interpolation bridges.',
            'Bridge layers contain voxels added by the corresponding interpolation pass only.',
            'Tile components intersecting both parent YOLO support and parent bridge support are assigned to parent_mask to keep categories mutually exclusive.',
            'Consolidated tile bridge layers are not attributed back to parent_mask vs parent_bridge acceptance because interpolation occurs after accepted tile masks are unioned.',
            'The final_output layer is included for direct comparison and can be ignored when recomposing a custom volume from component layers.',
            'NRRD payload was written by the bounded-memory streaming writer; no full 4D decomposed payload was materialized.',
            'Segment extents are recorded during layer materialization and reused during final packaging whenever possible.',
        ],
    }
    manifest_path = out_path.with_suffix(out_path.suffix + '.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2))

def write_decomposed_nrrd(
    layer_refs: Sequence[NrrdLayerRef],
    final_mask_u8: np.ndarray,
    out_path: Path,
    *,
    temp_dir: Path,
    workers: int = 1,
    include_final_output_layer: bool = True,
) -> Path:
    """Write a multi-layer NRRD whose trailing axis decomposes the final segmentation.

    Each supplied layer is an orthogonal ``(t,Y,X)`` binary mask in processing geometry.  Layers are
    restored to the final output geometry on the fly when cubic resizing changed the working shape,
    then streamed as a pigz gzip-encoded 4D payload ordered ``(X, Y, t, layer)``.  A sidecar
    ``*.manifest.json`` is written next to the NRRD so downstream tools can reconstruct unions
    without parsing NRRD custom fields.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_shape = tuple(int(x) for x in np.asarray(final_mask_u8).shape)
    if len(output_shape) != 3:
        raise ValueError(f'Decomposed NRRD final mask must be 3D (t,Y,X), got {output_shape}')

    effective_refs: List[NrrdLayerRef] = []
    for ref in layer_refs:
        if not ref.path.exists():
            print(f'Warning: skipping missing NRRD layer backing file: {ref.path}')
            continue
        effective_refs.append(ref)

    if bool(include_final_output_layer):
        final_ref_path = (
            temp_dir / 'nrrd_layers' / 'global' /
            ('final_output_binary.orthogonal.cvol' if raw_bbox_nrrd_layers_enabled() else 'final_output_binary.orthogonal.u8.dat')
        )
        final_ref = materialize_nrrd_global_layer(
            final_mask_u8,
            model_name='global',
            source='global',
            mask_kind='union',
            pass_index=0,
            stage='final_output_after_all_postprocessing',
            description='Final binary output after view/tile union, optional smoothing, optional keep_objects, and geometry restoration.',
            temp_dir=temp_dir,
            workers=int(workers),
        )
        if final_ref is None:
            # Preserve an empty final layer for empty volumes so the NRRD still documents final output.
            storage_format = 'raw_u8'
            if raw_bbox_nrrd_layers_enabled():
                write_raw_bbox_mask_store(
                    np.asarray(final_mask_u8, dtype=np.uint8),
                    final_ref_path,
                    format_name=CVOL_FORMAT,
                    desc='NRRD layer global final empty output',
                    workers=int(workers),
                    extra_meta={'nrrd_layer_key': 'global__global__global__union__final_output_after_all_postprocessing'},
                )
                storage_format = CVOL_FORMAT
            else:
                copied = copy_workspace_array(
                    np.asarray(final_mask_u8, dtype=np.uint8),
                    final_ref_path,
                    desc='NRRD layer global final empty output',
                    prefer_memory=False,
                    workers=int(workers),
                )
                close_memmap_array(copied)
            final_ref = NrrdLayerRef(
                key='global__global__global__union__final_output_after_all_postprocessing',
                name='Global / global / union / final output after all postprocessing',
                path=final_ref_path,
                shape=output_shape,
                dtype='uint8',
                storage_format=storage_format,
                model_name='global',
                view_name='global',
                view_family='global',
                source='global',
                mask_kind='union',
                pass_index=0,
                stage='final_output_after_all_postprocessing',
                description='Final binary output after all postprocessing; empty volume.',
                segment_extent_ijk=_nrrd_empty_segment_extent(),
            )
        effective_refs.append(final_ref)

    if not effective_refs:
        return write_nrrd(final_mask_u8, out_path)

    estimated_old_payload_bytes = (
        int(len(effective_refs)) * int(output_shape[2]) * int(output_shape[1]) * int(output_shape[0])
    )
    layer_slice_bytes = int(output_shape[2]) * int(output_shape[1])
    z_chunk_est = _nrrd_full_slice_z_chunk(1, int(output_shape[2]), int(output_shape[1]), int(output_shape[0]))
    bounded_buffer_bytes = int(layer_slice_bytes) * int(z_chunk_est)
    print(
        'Decomposed NRRD export is using the layer-slowest bounded streaming writer. '
        f'A full 4D uint8 payload would be ~{estimated_old_payload_bytes / GIB:.1f} GiB; '
        f'the RAM-aware write buffer is one layer chunk, ~{bounded_buffer_bytes / GIB:.3f} GiB '
        f'(estimated z_chunk={int(z_chunk_est)}). '
        'The list axis is last, so each layer is written as its native (t,Y,X) C-order byte stream.'
    )

    _log_nrrd_streaming_sources(effective_refs)

    layer_extents: List[NrrdSegmentExtent] = []
    layer_extent_sources: List[str] = []
    extent_source_counts: Dict[str, int] = {}
    for ref in tqdm(effective_refs, desc='NRRD segment extents: metadata'):
        extent, extent_source = _resolve_segment_extent_for_layer_ref(ref, output_shape)
        layer_extents.append(extent)
        layer_extent_sources.append(str(extent_source))
        extent_source_counts[str(extent_source)] = int(extent_source_counts.get(str(extent_source), 0)) + 1
    print(
        'NRRD segment extents resolved: ' +
        ', '.join(f'{key}={value}' for key, value in sorted(extent_source_counts.items()))
    )

    header = nrrd_decomposed_header(
        output_shape_zyx=output_shape,
        layer_refs=effective_refs,
        layer_extents=layer_extents,
    )

    with open(out_path, 'wb') as fh:
        _write_nrrd_ascii_header(
            fh,
            header=header,
            sizes=(int(output_shape[2]), int(output_shape[1]), int(output_shape[0]), len(effective_refs)),
            dimension=4,
            data_type='unsigned char',
            encoding='gzip',
        )
        with _open_pigz_payload_writer(fh) as payload_writer:
            _write_decomposed_nrrd_payload_stream(effective_refs, output_shape, payload_writer)

    _write_decomposed_nrrd_manifest(
        out_path=out_path,
        output_shape=output_shape,
        effective_refs=effective_refs,
        layer_extents=layer_extents,
        layer_extent_sources=layer_extent_sources,
    )
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
    if is_tilted_view(view):
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
    skeleton_u8: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> None:
    if view.name == 'transverse':
        write_overlay_video(volume_rgb, mask_u8, out_path, fps, skeleton_u8=skeleton_u8, show_progress=show_progress)
        return

    proc = ffmpeg_ffv1_rgb_writer(
        out_path,
        width=view.src_w,
        height=view.src_h,
        fps=fps,
    )
    blue = np.array([0, 0, 255], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)

    try:
        assert proc.stdin is not None
        if skeleton_u8 is None:
            iterator = zip(iter_view_frames(volume_rgb, view), iter_view_mask_frames(mask_u8, view))
        else:
            iterator = zip(iter_view_frames(volume_rgb, view), iter_view_mask_frames(mask_u8, view), iter_view_mask_frames(skeleton_u8, view))
        for items in tqdm(
            iterator,
            total=view.num_slices,
            desc=f'Writing {view.name} overlay video ({out_path.name})',
            disable=not show_progress,
        ):
            if skeleton_u8 is None:
                frame_rgb, frame_mask = items
                frame_skeleton = None
            else:
                frame_rgb, frame_mask, frame_skeleton = items
            frame = _gray_to_rgb_frame(np.asarray(frame_rgb))
            m = np.asarray(frame_mask, dtype=bool)
            if np.any(m):
                frame[m] = ((frame[m].astype(np.uint16) + blue.astype(np.uint16)) // 2).astype(np.uint8)
            if frame_skeleton is not None:
                skel = skeleton_overlay_mask_2d(np.asarray(frame_skeleton), int(view.src_h), int(view.src_w))
                if np.any(skel):
                    frame[skel] = red
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)


def write_native_view_overlay_video(
    volume_rgb: np.ndarray,
    native_mask_u8: np.ndarray,
    view: ViewInfo,
    out_path: Path,
    fps: float,
    *,
    show_progress: bool = True,
) -> Path:
    """Write a v12.2.0 troubleshooting overlay in a view's own slice space.

    The mask is already flattened in the native raster consumed by/produced for
    the view: shape (view.num_slices, view.src_h, view.src_w). Source frames are
    resliced from the single-channel orthogonal processing volume only for the
    presentation overlay.
    """
    if tuple(int(x) for x in np.asarray(native_mask_u8).shape) != (int(view.num_slices), int(view.src_h), int(view.src_w)):
        raise ValueError(
            f'Troubleshooting overlay for {view.name} expected native mask shape '
            f'{(int(view.num_slices), int(view.src_h), int(view.src_w))}, got {tuple(np.asarray(native_mask_u8).shape)}'
        )

    proc = ffmpeg_ffv1_rgb_writer(
        out_path,
        width=int(view.src_w),
        height=int(view.src_h),
        fps=float(fps),
    )
    blue = np.array([0, 0, 255], dtype=np.uint8)
    try:
        assert proc.stdin is not None
        for idx in tqdm(
            range(int(view.num_slices)),
            total=int(view.num_slices),
            desc=f'Writing troubleshooting overlay {view.name} ({out_path.name})',
            disable=not bool(show_progress),
        ):
            frame_gray = get_view_frame_by_index(volume_rgb, view, int(idx))
            frame = _gray_to_rgb_frame(np.asarray(frame_gray))
            m = np.asarray(native_mask_u8[int(idx)], dtype=bool)
            if np.any(m):
                frame[m] = ((frame[m].astype(np.uint16) + blue.astype(np.uint16)) // 2).astype(np.uint8)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)
    return out_path


def collect_troubleshooting_overlay_futures(
    executor: ThreadPoolExecutor,
    *,
    volume_rgb: np.ndarray,
    model_name: str,
    views: Sequence[ViewInfo],
    native_view_support: Dict[str, np.ndarray],
    consolidated_tile_masks: Dict[str, np.ndarray],
    out_dir: Path,
    stem: str,
    fps: float,
    show_progress: bool = False,
) -> Tuple[Dict[str, Path], List[Future]]:
    """Schedule simplified v12.2.0 troubleshooting overlay MKVs."""
    result_paths: Dict[str, Path] = {}
    futures: List[Future] = []
    troubleshooting_dir = Path(out_dir) / 'troubleshooting'
    troubleshooting_dir.mkdir(parents=True, exist_ok=True)

    for view in views:
        view_token = view_output_token(view)
        full_mask = native_view_support.get(str(view.name))
        if full_mask is not None:
            overlay_path = troubleshooting_dir / f'{stem}_Troubleshooting_{view_token}_FullFrame_Overlay.mkv'
            futures.append(executor.submit(
                write_native_view_overlay_video,
                volume_rgb,
                full_mask,
                view,
                overlay_path,
                fps,
                show_progress=show_progress,
            ))
            result_paths[f'troubleshooting_{view.name}_fullframe_overlay'] = overlay_path

        tile_mask = consolidated_tile_masks.get(str(view.name))
        if tile_mask is not None:
            overlay_path = troubleshooting_dir / f'{stem}_Troubleshooting_{view_token}_Tiles_Overlay.mkv'
            futures.append(executor.submit(
                write_native_view_overlay_video,
                volume_rgb,
                tile_mask,
                view,
                overlay_path,
                fps,
                show_progress=show_progress,
            ))
            result_paths[f'troubleshooting_{view.name}_tiles_overlay'] = overlay_path

    if not futures:
        note_path = troubleshooting_dir / f'{stem}_Troubleshooting_NoActiveMasks.txt'
        note_path.write_text(
            f'No full-frame or consolidated tiled native-view masks were available for model {model_name}.\n'
        )
        result_paths['troubleshooting_note'] = note_path

    return result_paths, futures


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



@dataclass(frozen=True)
class LowQualityDownbinSpec:
    raw_value: str
    token: str
    scale: float
    output_shape_t_y_x: Tuple[int, int, int]
    warning: str = ''


def _nearest_multiple_of_four(value: float) -> int:
    return max(4, int(math.floor(float(value) / 4.0 + 0.5)) * 4)


def _round_low_quality_dimension(value: float) -> int:
    # Large SLURM volumes should round to multiples of 4 per spec. For tiny synthetic
    # tests, avoid changing a sub-4 dimension to 4 unless the scaled value warrants it.
    value_f = max(1.0, float(value))
    if value_f < 4.0:
        return max(1, int(round(value_f)))
    return _nearest_multiple_of_four(value_f)


def _low_quality_token(raw: str, shape_t_y_x: Tuple[int, int, int]) -> str:
    safe_raw = str(raw).strip().replace('-', 'm').replace('.', 'p').replace(',', '_')
    t_dim, h_dim, w_dim = (int(shape_t_y_x[0]), int(shape_t_y_x[1]), int(shape_t_y_x[2]))
    return f'{safe_raw}_{int(w_dim)}x{int(h_dim)}x{int(t_dim)}'


def resolve_low_quality_downbin_specs(
    downbin_values: Sequence[str] | str | None,
    low_quality_requested: bool,
    source_shape_t_y_x: Tuple[int, int, int],
) -> Tuple[List[LowQualityDownbinSpec], List[str]]:
    """Resolve v12.2.1 isotropic low-quality downbins in native input geometry."""
    if downbin_values is None and not bool(low_quality_requested):
        return [], []

    raw_tokens = _parse_token_list(downbin_values) if downbin_values is not None else []
    if not raw_tokens:
        raw_tokens = ['1024']

    in_t, in_h, in_w = (int(source_shape_t_y_x[0]), int(source_shape_t_y_x[1]), int(source_shape_t_y_x[2]))
    max_dim = max(1, int(in_t), int(in_h), int(in_w))
    specs: List[LowQualityDownbinSpec] = []
    warnings: List[str] = []
    seen_shapes: set[Tuple[int, int, int]] = set()

    for raw in raw_tokens:
        raw_s = str(raw).strip()
        if not raw_s:
            continue
        try:
            value = float(raw_s)
        except Exception as exc:
            raise ValueError(f'--save_low_quality_downbin value is not numeric: {raw_s!r}') from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError('--save_low_quality_downbin values must be positive finite numbers')

        warning = ''
        raw_lower = raw_s.lower()
        looks_float = ('.' in raw_s) or ('e' in raw_lower)
        if looks_float and value <= 1.0:
            scale = float(value)
        else:
            nearest_int = int(round(value))
            if abs(float(value) - float(nearest_int)) > 1e-6:
                raise ValueError('Integer --save_low_quality_downbin values >= 1 must be whole numbers; use a fraction such as 0.5 for scale factors')
            if nearest_int <= 0:
                raise ValueError('--save_low_quality_downbin integer targets must be positive')
            rounded_target = _nearest_multiple_of_four(float(nearest_int))
            if int(rounded_target) != int(nearest_int):
                warning = (
                    f'--save_low_quality_downbin {int(nearest_int)} is not a multiple of 4; '
                    f'rounded to {int(rounded_target)} for isotropic low-quality output.'
                )
                warnings.append(warning)
            scale = float(rounded_target) / float(max_dim)

        out_t = _round_low_quality_dimension(float(in_t) * float(scale))
        out_h = _round_low_quality_dimension(float(in_h) * float(scale))
        out_w = _round_low_quality_dimension(float(in_w) * float(scale))
        shape = (int(out_t), int(out_h), int(out_w))
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        specs.append(LowQualityDownbinSpec(
            raw_value=raw_s,
            token=_low_quality_token(raw_s, shape),
            scale=float(scale),
            output_shape_t_y_x=shape,
            warning=warning,
        ))

    return specs, warnings


def resize_gray_volume_to_shape(
    volume_gray: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 8 * GIB,
    desc: str = 'Resizing gray volume',
) -> np.ndarray:
    in_t, in_h, in_w = (int(volume_gray.shape[0]), int(volume_gray.shape[1]), int(volume_gray.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return volume_gray

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)
    xy_interp = cv2.INTER_AREA if (out_h <= in_h and out_w <= in_w) else cv2.INTER_LINEAR

    def _render_target_slice(out_z: int) -> None:
        src_z = _linear_source_index(int(out_z), int(out_t), int(in_t))
        z0 = int(math.floor(src_z))
        z1 = min(in_t - 1, z0 + 1)
        alpha = float(src_z - float(z0))
        f0 = _resize_gray_slice_nearest_or_linear(volume_gray[z0], out_w, out_h, xy_interp)
        if z1 == z0 or alpha <= 1e-7:
            out_mm[int(out_z), :, :] = f0
            return
        f1 = _resize_gray_slice_nearest_or_linear(volume_gray[z1], out_w, out_h, xy_interp)
        blended = np.clip(
            np.rint((1.0 - alpha) * f0.astype(np.float32, copy=False) + alpha * f1.astype(np.float32, copy=False)),
            0.0,
            255.0,
        ).astype(np.uint8)
        out_mm[int(out_z), :, :] = blended

    parallel_for_indices_chunked(
        int(out_t),
        _render_target_slice,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
        show_progress=True,
    )
    flush_array(out_mm)
    return out_mm


def resize_binary_mask_volume_to_shape(
    mask_u8: np.ndarray,
    out_shape: Tuple[int, int, int],
    out_path: Path,
    *,
    workers: int = 1,
    prefer_memory: bool = True,
    reserve_bytes: int = 8 * GIB,
    desc: str = 'Resizing binary mask volume',
) -> np.ndarray:
    in_t, in_h, in_w = (int(mask_u8.shape[0]), int(mask_u8.shape[1]), int(mask_u8.shape[2]))
    out_t, out_h, out_w = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    if (in_t, in_h, in_w) == (out_t, out_h, out_w):
        return mask_u8

    out_mm = allocate_workspace_array(
        shape=(out_t, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )
    worker_count = choose_slice_parallel_workers(int(workers), int(out_t))
    chunk_size = choose_parallel_chunk_size(out_t, worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _resize_mask_xy(frame: np.ndarray) -> np.ndarray:
        frame_u8 = (np.asarray(frame, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
        if int(frame_u8.shape[0]) == int(out_h) and int(frame_u8.shape[1]) == int(out_w):
            return np.ascontiguousarray(frame_u8)
        if int(out_h) <= int(frame_u8.shape[0]) and int(out_w) <= int(frame_u8.shape[1]):
            scaled = cv2.resize(
                np.ascontiguousarray(frame_u8 * np.uint8(255)),
                (int(out_w), int(out_h)),
                interpolation=cv2.INTER_AREA,
            )
            return (scaled > 0).astype(np.uint8, copy=False)
        scaled = cv2.resize(
            np.ascontiguousarray(frame_u8),
            (int(out_w), int(out_h)),
            interpolation=cv2.INTER_NEAREST,
        )
        return (scaled > 0).astype(np.uint8, copy=False)

    def _restore_slice(out_z: int) -> None:
        src_start = int(math.floor(float(out_z) * float(in_t) / float(out_t)))
        src_stop = int(math.ceil(float(out_z + 1) * float(in_t) / float(out_t)))
        src_start = int(np.clip(src_start, 0, in_t - 1))
        src_stop = int(np.clip(max(src_start + 1, src_stop), 1, in_t))
        restored = np.zeros((int(out_h), int(out_w)), dtype=np.uint8)
        for src_idx in range(src_start, src_stop):
            restored |= _resize_mask_xy(mask_u8[int(src_idx)])
        out_mm[int(out_z), :, :] = restored

    parallel_for_indices_chunked(
        int(out_t),
        _restore_slice,
        max_workers=worker_count,
        desc=desc,
        chunk_size=chunk_size,
        show_progress=True,
    )
    flush_array(out_mm)
    return out_mm


def ffmpeg_h264_rgb_writer(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='rgb24',
        codec='libx264',
        pix_fmt_out='yuv420p',
        codec_args=['-preset', 'slow'],
    )


def ffmpeg_h264_gray_writer(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    return ffmpeg_rawvideo_writer(
        out_path=out_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        pix_fmt_in='gray',
        codec='libx264',
        pix_fmt_out='yuv420p',
        codec_args=['-preset', 'slow'],
    )


def write_low_quality_overlay_video(
    volume_gray: np.ndarray,
    mask_u8: np.ndarray,
    out_path: Path,
    fps: float,
    skeleton_u8: Optional[np.ndarray] = None,
    show_progress: bool = True,
) -> Path:
    t_dim, h_dim, w_dim = volume_gray.shape
    assert mask_u8.shape == (t_dim, h_dim, w_dim)
    if skeleton_u8 is not None:
        assert skeleton_u8.shape == (t_dim, h_dim, w_dim)
    proc = ffmpeg_h264_rgb_writer(out_path, int(w_dim), int(h_dim), float(fps))
    blue = np.array([0, 0, 255], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    try:
        assert proc.stdin is not None
        for t in tqdm(range(int(t_dim)), desc=f'Writing low-quality overlay ({out_path.name})', disable=not show_progress):
            frame = _gray_to_rgb_frame(np.asarray(volume_gray[int(t)]))
            m = np.asarray(mask_u8[int(t)], dtype=bool)
            if np.any(m):
                frame[m] = ((frame[m].astype(np.uint16) + blue.astype(np.uint16)) // 2).astype(np.uint8)
            if skeleton_u8 is not None:
                skel = skeleton_overlay_mask_2d(np.asarray(skeleton_u8[int(t)]), int(h_dim), int(w_dim))
                if np.any(skel):
                    frame[skel] = red
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)
    return out_path


def write_low_quality_binary_video(
    mask_u8: np.ndarray,
    out_path: Path,
    fps: float,
    show_progress: bool = True,
) -> Path:
    t_dim, h_dim, w_dim = mask_u8.shape
    proc = ffmpeg_h264_gray_writer(out_path, int(w_dim), int(h_dim), float(fps))
    try:
        assert proc.stdin is not None
        for t in tqdm(range(int(t_dim)), desc=f'Writing low-quality binary ({out_path.name})', disable=not show_progress):
            frame = (np.asarray(mask_u8[int(t)], dtype=np.uint8) > 0).astype(np.uint8) * np.uint8(255)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        close_ffmpeg_writer(proc)
    return out_path


def save_low_quality_outputs(
    *,
    volume_gray: np.ndarray,
    mask_u8: np.ndarray,
    skeleton_u8: Optional[np.ndarray] = None,
    out_dir: Path,
    stem: str,
    fps: float,
    downbin_specs: Sequence[LowQualityDownbinSpec],
    temp_dir: Path,
    nrrd_layer_refs: Optional[Sequence[NrrdLayerRef]] = None,
    workers: int = 1,
    show_progress: bool = True,
) -> Dict[str, Path]:
    """Write v12.2.1 low-quality outputs with isotropic X/Y/t resizing."""
    low_root = out_dir / 'low_quality'
    low_root.mkdir(parents=True, exist_ok=True)
    result_paths: Dict[str, Path] = {'low_quality_dir': low_root}
    if not downbin_specs:
        return result_paths

    source_t = max(1, int(mask_u8.shape[0]))
    for spec in downbin_specs:
        out_t, out_h, out_w = (
            int(spec.output_shape_t_y_x[0]),
            int(spec.output_shape_t_y_x[1]),
            int(spec.output_shape_t_y_x[2]),
        )
        spec_dir = low_root / spec.token
        spec_dir.mkdir(parents=True, exist_ok=True)
        fps_lq = max(1e-6, float(fps) * float(out_t) / float(source_t))
        print(
            f'Low-quality downbin {spec.raw_value}: native (t,Y,X)=({source_t},{int(mask_u8.shape[1])},{int(mask_u8.shape[2])}) '
            f'-> ({out_t},{out_h},{out_w}); playback fps adjusted {float(fps):g} -> {fps_lq:g}'
        )

        gray_lq = resize_gray_volume_to_shape(
            volume_gray,
            (out_t, out_h, out_w),
            temp_dir / 'low_quality' / spec.token / 'source.gray8.dat',
            workers=int(workers),
            prefer_memory=True,
            desc=f'Low-quality source resize {spec.token}',
        )
        mask_lq = resize_binary_mask_volume_to_shape(
            mask_u8,
            (out_t, out_h, out_w),
            temp_dir / 'low_quality' / spec.token / 'mask.u8.dat',
            workers=int(workers),
            prefer_memory=True,
            desc=f'Low-quality mask resize {spec.token}',
        )
        skeleton_lq: Optional[np.ndarray] = None
        if skeleton_u8 is not None:
            skeleton_lq = resize_binary_mask_volume_to_shape(
                skeleton_u8,
                (out_t, out_h, out_w),
                temp_dir / 'low_quality' / spec.token / 'skeleton.u8.dat',
                workers=int(workers),
                prefer_memory=True,
                desc=f'Low-quality skeleton resize {spec.token}',
            )
        try:
            overlay_path = spec_dir / f'{stem}_Overlay_LowQuality_{spec.token}.mp4'
            binary_path = spec_dir / f'{stem}_Binary_LowQuality_{spec.token}.mp4'
            nrrd_path = spec_dir / f'{stem}_LowQuality_{spec.token}.nrrd'
            write_low_quality_overlay_video(gray_lq, mask_lq, overlay_path, fps_lq, skeleton_u8=skeleton_lq, show_progress=show_progress)
            write_low_quality_binary_video(mask_lq, binary_path, fps_lq, show_progress=show_progress)
            if nrrd_layer_refs is not None and len(nrrd_layer_refs) > 0:
                write_decomposed_nrrd(
                    list(nrrd_layer_refs),
                    mask_lq,
                    nrrd_path,
                    temp_dir=temp_dir / 'low_quality' / spec.token / 'nrrd_decomposed',
                    workers=int(workers),
                    include_final_output_layer=True,
                )
                result_paths[f'low_quality_{spec.token}_nrrd_manifest'] = nrrd_path.with_suffix(nrrd_path.suffix + '.manifest.json')
            else:
                write_nrrd(mask_lq, nrrd_path)
            result_paths[f'low_quality_{spec.token}_overlay'] = overlay_path
            result_paths[f'low_quality_{spec.token}_binary_video'] = binary_path
            result_paths[f'low_quality_{spec.token}_nrrd'] = nrrd_path
        finally:
            if gray_lq is not volume_gray:
                close_memmap_array(gray_lq)
            if mask_lq is not mask_u8:
                close_memmap_array(mask_lq)
            if skeleton_lq is not None and skeleton_lq is not skeleton_u8:
                close_memmap_array(skeleton_lq)

    return result_paths



def planned_low_quality_output_paths(
    *,
    out_dir: Path,
    stem: str,
    downbin_specs: Sequence[LowQualityDownbinSpec],
    nrrd_layer_refs: Optional[Sequence[NrrdLayerRef]] = None,
) -> Dict[str, Path]:
    """Return the paths that save_low_quality_outputs will create.

    This lets the background output manager launch low-quality videos/NRRDs at the
    same time as full-size outputs, including concurrent full-size and low-quality
    NRRD streaming, instead of waiting for the full-size output group to finish.
    """
    low_root = out_dir / 'low_quality'
    result_paths: Dict[str, Path] = {'low_quality_dir': low_root}
    for spec in downbin_specs:
        spec_dir = low_root / spec.token
        overlay_path = spec_dir / f'{stem}_Overlay_LowQuality_{spec.token}.mp4'
        binary_path = spec_dir / f'{stem}_Binary_LowQuality_{spec.token}.mp4'
        nrrd_path = spec_dir / f'{stem}_LowQuality_{spec.token}.nrrd'
        result_paths[f'low_quality_{spec.token}_overlay'] = overlay_path
        result_paths[f'low_quality_{spec.token}_binary_video'] = binary_path
        result_paths[f'low_quality_{spec.token}_nrrd'] = nrrd_path
        if nrrd_layer_refs is not None and len(nrrd_layer_refs) > 0:
            result_paths[f'low_quality_{spec.token}_nrrd_manifest'] = nrrd_path.with_suffix(nrrd_path.suffix + '.manifest.json')
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
    skeleton_u8: Optional[np.ndarray] = None,
    out_dir: Path,
    stem: str,
    fps: float,
    save_binary_pattern_value: Optional[str],
    save_labels_pattern_value: Optional[str],
    save_nrrd_flag: bool,
    tag: Optional[str] = None,
    frame_workers: int = 1,
    show_progress: bool = False,
    nrrd_layer_refs: Optional[Sequence[NrrdLayerRef]] = None,
    nrrd_temp_dir: Optional[Path] = None,
    nrrd_workers: int = 1,
) -> Tuple[Dict[str, Path], List[Future]]:
    futures: List[Future] = []
    result_paths: Dict[str, Path] = {}
    tag_suffix = f"_{tag}" if tag else ""

    overlay_path = out_dir / f"{stem}_Overlay{tag_suffix}.mkv"
    futures.append(executor.submit(
        write_overlay_video,
        volume_rgb,
        mask_u8,
        overlay_path,
        fps,
        skeleton_u8=skeleton_u8,
        show_progress=show_progress,
    ))
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
        if nrrd_layer_refs is not None and len(nrrd_layer_refs) > 0 and nrrd_temp_dir is not None:
            futures.append(executor.submit(
                write_decomposed_nrrd,
                list(nrrd_layer_refs),
                mask_u8,
                nrrd_path,
                temp_dir=nrrd_temp_dir,
                workers=int(nrrd_workers),
                include_final_output_layer=True,
            ))
            result_paths["nrrd_manifest"] = nrrd_path.with_suffix(nrrd_path.suffix + '.manifest.json')
        else:
            futures.append(executor.submit(write_nrrd, mask_u8, nrrd_path))
        result_paths["nrrd"] = nrrd_path

    return result_paths, futures


def collect_multiplanar_output_futures(
    executor: ThreadPoolExecutor,
    *,
    volume_rgb: np.memmap,
    mask_u8: np.ndarray,
    skeleton_u8: Optional[np.ndarray] = None,
    out_dir: Path,
    stem: str,
    fps: float,
    save_binary_pattern_value: Optional[str],
    save_labels_pattern_value: Optional[str],
    save_sagittal_flag: bool = True,
    save_coronal_flag: bool = True,
    tag: Optional[str] = None,
    frame_workers: int = 1,
    show_progress: bool = False,
) -> Tuple[Dict[str, Path], List[Future]]:
    t_dim, h_dim, w_dim = mask_u8.shape
    views = {
        v.name: v for v in get_view_infos(
            t_dim, h_dim, w_dim,
            disable_multiplanar=None,
            azimuth_angle=0.0,
            include_radial=False,
            enable_sagittal=True,
            enable_coronal=True,
        )
    }
    futures: List[Future] = []
    result_paths: Dict[str, Path] = {}

    requested_view_names: List[str] = []
    if bool(save_sagittal_flag):
        requested_view_names.append('sagittal')
    if bool(save_coronal_flag):
        requested_view_names.append('coronal')

    for view_name in requested_view_names:
        view = views[view_name]
        pretty = view_output_token(view)
        tag_suffix = f'_{tag}' if tag else ''

        overlay_path = out_dir / f'{stem}_{pretty}_Overlay{tag_suffix}.mkv'
        futures.append(executor.submit(
            write_overlay_video_for_view,
            volume_rgb,
            mask_u8,
            view,
            overlay_path,
            fps,
            skeleton_u8=skeleton_u8,
            show_progress=show_progress,
        ))
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


def write_summary_file(
    out_path: Path,
    *,
    command: str,
    input_path: Path,
    out_dir: Path,
    scratch_dir: Path,
    source_shape_x_y_t: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
    fps: float,
    model_paths: Sequence[str],
    view_names: Sequence[str],
    view_prediction_stats: Dict[str, int],
    interpolation_stats: List[Dict[str, object]],
    enable_3d_void_fill: bool,
    gaussian_smoothing_stats: Optional[Dict[str, int | float]],
    keep_objects_stats: Optional[Dict[str, int]],
    voxel_volume: Optional[int],
    final_paths: Dict[str, Path],
    augmentation_workers: int,
    slice_postprocess_workers: int,
    interpolation_workers: int,
    output_workers: int,
    spec_notes: Optional[Sequence[str]] = None,
    view_prediction_labels: Optional[Dict[str, str]] = None,
) -> Path:
    lines: List[str] = []
    lines.append(f'Command: {command}')
    lines.append(f'Input: {input_path}')
    lines.append(f'Output directory: {out_dir}')
    lines.append(f'Source dimensions before cubic resizing (X, Y, t): {source_shape_x_y_t}')
    lines.append(f'Processing volume shape (t, Y, X): {volume_shape}')
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
    if spec_notes:
        lines.append('')
        lines.append('Specification notes:')
        for note in spec_notes:
            lines.append(f'  - {note}')

    lines.append('')
    lines.append('View statistics:')
    total_prediction_count = 0
    labels = dict(view_prediction_labels or {})
    ordered_keys: List[str] = [k for k in ('transverse', 'sagittal', 'coronal', 'radial') if k in view_prediction_stats]
    tilted_keys = [k for k in view_prediction_stats.keys() if str(k).startswith('tilted_')]
    other_keys = [k for k in view_prediction_stats.keys() if k not in set(ordered_keys) and k not in set(tilted_keys)]
    for view_key in ordered_keys + sorted(tilted_keys, key=lambda k: labels.get(k, k)) + sorted(other_keys):
        label = labels.get(view_key, str(view_key).replace('_', ' ').title())
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

    lines.append('')
    if gaussian_smoothing_stats is not None and int(gaussian_smoothing_stats.get('enabled', 0)) > 0:
        lines.append(
            'Gaussian smoothing: enabled; '
            f"sigma={float(gaussian_smoothing_stats.get('sigma', 0.0)):g}, "
            f"passes_requested={int(gaussian_smoothing_stats.get('passes_requested', 0))}, "
            f"passes_completed={int(gaussian_smoothing_stats.get('passes_completed', 0))}"
        )
        lines.append(
            '  voxel changes after thresholding: '
            f"added={int(gaussian_smoothing_stats.get('total_added_voxels', 0))}, "
            f"removed={int(gaussian_smoothing_stats.get('total_removed_voxels', 0))}"
        )
    else:
        lines.append('Gaussian smoothing: disabled')

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
        lines.append(f'voxel_volume_native_input_space: {int(voxel_volume)}')

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
        raise ValueError('v12.2.0_SLURM accepts a single --model path; multiple-model inference has been removed')
    model_path_resolved = str(Path(model_path).expanduser().resolve())
    if not Path(model_path_resolved).exists():
        raise FileNotFoundError(model_path_resolved)
    model_paths = [model_path_resolved]
    model_name = Path(model_paths[0]).stem

    model_load_executor: Optional[ThreadPoolExecutor] = None
    model_load_future: Optional[Future] = None
    if background_model_load_enabled():
        print(f'Loading model in background while input volume is prepared: {model_name} ({model_paths[0]})')
        model_load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='model-load')
        model_load_future = model_load_executor.submit(load_ultralytics_model, model_paths[0], 'segment')

    angles = _parse_angles(args.angle) or [0.0, 120.0, 240.0]
    single_angle_streaming_cleanup_active = bool(
        len(angles) == 1 and _env_flag('YOLO_TTA_SINGLE_ANGLE_STREAMING_CLEANUP', True)
    )
    if int(args.batch) < 1:
        raise ValueError('--batch must be >= 1')
    set_inference_batch_size(int(args.batch))
    tilt_views = resolve_tilt_views(args.tilt_view)
    tilt_angles = resolve_tilt_angles(args.tilt_angle)
    tilt_directions = resolve_tilt_directions(args.tilt_direction)
    tile_configs = resolve_tile_configs(args.tile_size, args.tile_stride)
    enable_sagittal = bool(args.enable_sagittal)
    enable_coronal = bool(args.enable_coronal)
    # v12.2.0 retains the v12 single-transverse default-output behavior and omits old per-view save flags. Default outputs stay transverse-only;
    # active non-transverse views contribute to inference/union but are not separately exported.
    save_sagittal = False
    save_coronal = False
    requested_azimuth_angle = None if args.azimuth_angle is None else float(args.azimuth_angle)

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
    if args.gaussian_smoothing is not None and float(args.gaussian_smoothing) < 0.0:
        raise ValueError('--gaussian_smoothing must be >= 0; use 0 to disable smoothing')
    if args.gaussian_smoothing_passes is not None and int(args.gaussian_smoothing_passes) < 0:
        raise ValueError('--gaussian_smoothing_passes must be >= 0; use 0 to disable smoothing')
    gaussian_smoothing_cli_requested = bool(args.gaussian_smoothing is not None or args.gaussian_smoothing_passes is not None)
    gaussian_smoothing_disabled_by_zero = bool(
        (args.gaussian_smoothing is not None and float(args.gaussian_smoothing) == 0.0)
        or (args.gaussian_smoothing_passes is not None and int(args.gaussian_smoothing_passes) == 0)
    )
    gaussian_smoothing_enabled, gaussian_smoothing_sigma, gaussian_smoothing_passes = resolve_gaussian_smoothing_settings(
        args.gaussian_smoothing,
        args.gaussian_smoothing_passes,
    )
    if float(args.interpolate_min_radius) < 0:
        raise ValueError('--interpolate_min_radius must be >= 0')
    if float(args.min_radius) < 0:
        raise ValueError('--min_radius must be >= 0')
    if int(args.keep_objects) < 0:
        raise ValueError('--keep_objects must be >= 0')
    if requested_azimuth_angle is not None and float(requested_azimuth_angle) < 0:
        raise ValueError('--azimuth_angle must be >= 0')
    for tilt_angle in tilt_angles:
        if not (0.0 < float(tilt_angle) <= 45.0):
            raise ValueError('--tilt_angle values must be greater than 0 and less than or equal to 45')
    if not (-90.0 < float(args.interpolation_search_angle) < 90.0):
        raise ValueError('--interpolation_search_angle must be greater than -90 and less than 90')
    low_quality_requested = bool(args.save_low_quality or args.save_low_quality_downbin is not None)
    # Low-quality NRRDs must mirror the full decomposed NRRD layer stack.  Therefore layer
    # materialization is needed whenever a low-quality NRRD will be written, even if the user did
    # not also request the full-size --save_nrrd output.
    nrrd_layers_needed = bool(args.save_nrrd or low_quality_requested)
    troubleshooting_outputs_enabled = bool(args.troubleshooting)
    # v12.2.0 --troubleshooting creates overlay MKVs only. Temporary scratch retention is a
    # separate hidden maintenance escape hatch so troubleshooting no longer changes cleanup semantics.
    keep_temp_artifacts = bool(_env_flag('YOLO_TTA_KEEP_TEMP', False))

    out_dir = Path(args.output).expanduser().resolve() if args.output else (Path.cwd() / input_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = choose_scratch_dir(None, out_dir, input_path.stem)
    expose_scratch_in_output(out_dir, temp_dir)
    print(f"Bulk scratch dir: {temp_dir}")

    info = ffprobe_info(input_path)
    input_W = int(info['width'])
    input_H = int(info['height'])
    input_T = int(info['num_frames'])
    fps = float(info['fps'])

    low_quality_downbin_specs, low_quality_downbin_warnings = resolve_low_quality_downbin_specs(
        args.save_low_quality_downbin,
        bool(low_quality_requested),
        (input_T, input_H, input_W),
    )

    preprocess_streaming_active = bool(streaming_preprocess_enabled())
    vol_path = temp_dir / 'input_volume.gray8.dat'
    if preprocess_streaming_active:
        print(
            'v12.2.12 streaming preprocessing active: ffmpeg decode and cubic resize return their destination arrays immediately; '
            'Transverse consumers wait only for the needed processing slice.'
        )
        input_volume_rgb = decode_video_to_memmap_gray8_streaming(
            input_video=input_path,
            out_dat=vol_path,
            num_frames=input_T,
            width=input_W,
            height=input_H,
            overwrite=False,
            prefer_memory=True,
        )
    else:
        input_volume_rgb = decode_video_to_memmap_gray8(
            input_video=input_path,
            out_dat=vol_path,
            num_frames=input_T,
            width=input_W,
            height=input_H,
            overwrite=False,
            prefer_memory=True,
        )
    (temp_dir / 'input_volume.meta.json').write_text(
        json.dumps({'shape': [input_T, input_H, input_W], 'dtype': 'uint8', 'channels': 1, 'fps': fps, 'streaming_preprocess': bool(preprocess_streaming_active)}, indent=2)
    )

    processing_shape = compute_cube_resize_shape(input_T, input_H, input_W, tolerance=0.05)
    if processing_shape != (input_T, input_H, input_W):
        print(
            'v12.2.0 cubic resize: '
            f'input shape (t,Y,X)=({input_T},{input_H},{input_W}) -> '
            f'processing shape (t,Y,X)={processing_shape}'
        )
        if preprocess_streaming_active:
            volume_rgb = resize_volume_to_processing_cube_gray8_streaming(
                input_volume_rgb,
                processing_shape,
                temp_dir / 'input_volume.v950_cube.gray8.dat',
                workers=max(1, default_worker_budget()),
                prefer_memory=True,
            )
        else:
            volume_rgb = resize_volume_to_processing_cube_gray8(
                input_volume_rgb,
                processing_shape,
                temp_dir / 'input_volume.v950_cube.gray8.dat',
                workers=max(1, default_worker_budget()),
                prefer_memory=True,
            )
    else:
        print(f'v12.2.0 cubic resize: input shape already within 5% cube tolerance ({processing_shape})')
        volume_rgb = input_volume_rgb

    T, H, W = (int(processing_shape[0]), int(processing_shape[1]), int(processing_shape[2]))
    resolved_azimuth_angle = resolve_radial_azimuth_angle(
        requested_azimuth_angle,
        enable_radial=bool(args.enable_radial),
        diameter=min(W, H),
    )
    if bool(args.enable_radial) and resolved_azimuth_angle > 0.0 and requested_azimuth_angle is None:
        print(f'Radial azimuth default: using full-coverage spacing {resolved_azimuth_angle:.8g}° for processing diameter {min(W, H)}')

    (temp_dir / 'processing_volume.meta.json').write_text(
        json.dumps({
            'input_shape_t_y_x': [input_T, input_H, input_W],
            'processing_shape_t_y_x': [T, H, W],
            'dtype': 'uint8',
            'channels': 1,
            'fps': fps,
            'azimuth_angle_deg': resolved_azimuth_angle,
            'enable_sagittal': bool(enable_sagittal),
            'enable_coronal': bool(enable_coronal),
            'tilt_views': list(tilt_views),
            'tilt_angles_deg': [float(v) for v in tilt_angles],
            'tilt_directions': list(tilt_directions),
        }, indent=2)
    )

    views = get_view_infos(
        T=T,
        H=H,
        W=W,
        disable_multiplanar=None,
        enable_sagittal=bool(enable_sagittal),
        enable_coronal=bool(enable_coronal),
        azimuth_angle=float(resolved_azimuth_angle),
        include_radial=True,
        tilt_views=tilt_views,
        tilt_angles=tilt_angles,
        tilt_directions=tilt_directions,
    )
    cartesian_views = orthogonal_views_only(views)
    transverse_inference_disabled = bool(args.disable_transverse)
    inference_views = [v for v in views if not (transverse_inference_disabled and v.name == 'transverse')]
    interpolating_views = [v for v in inference_views if _view_uses_interpolation(v, int(args.interpolate))]
    spec_notes: List[str] = []
    if preprocess_streaming_active:
        spec_notes.append(
            'v12.2.12 streaming preprocessing is active: decode/cube preprocessing can run concurrently with Transverse rendering and GPU inference; Transverse readers wait only for the slice they need, while stack-sampling view families wait for the completed preprocessing volume.'
        )
    else:
        spec_notes.append(
            'v12.2.12 streaming preprocessing is disabled by YOLO_TTA_STREAMING_PREPROCESS=0; decode and cubic resize finish before inference scheduling begins.'
        )
    if background_model_load_enabled():
        spec_notes.append(
            'v12.2.11 startup overlap is active: the YOLO model load is submitted before ffprobe/decode/cubic preparation and joined only when the scheduler needs the model for prediction. Set YOLO_TTA_BACKGROUND_MODEL_LOAD=0 to restore synchronous model loading.'
        )
    if bool(single_angle_streaming_cleanup_active):
        spec_notes.append(
            'v12.2.8 single-angle streaming cleanup is active: exactly one --angle value was supplied, so YOLO result slices are confidence-filtered, 2D-hole-filled, and view-native min_radius-filtered as they stream in; only Sagittal/Coronal transverse-plane min_radius cleanup waits for the completed view volume before interpolation.'
        )
    else:
        spec_notes.append(
            'v12.2.8 single-angle streaming cleanup is inactive because multiple --angle values are being accumulated or YOLO_TTA_SINGLE_ANGLE_STREAMING_CLEANUP=0; view cleanup therefore runs after all angle volumes for a view have accumulated.'
        )
    spec_notes.append('Input video channel handling: RGB/YUV inputs are flattened to one gray/luma channel during decode; single-channel Y/gray inputs remain single-channel, and YOLO receives H×W×1 frames.')
    spec_notes.append('Voxel-volume reporting, when enabled, counts the final binary mask after restoration to native input geometry, not imgsz or cubic working geometry.')
    spec_notes.append('v12.2.0 tilt-angle validation follows the specification: values must be greater than 0 and less than or equal to 45 degrees.')
    if low_quality_downbin_warnings:
        for warning in low_quality_downbin_warnings:
            print(f'Warning: {warning}')
            spec_notes.append(warning)
    if low_quality_downbin_specs:
        spec_notes.append(
            'Low-quality outputs use isotropic X/Y/t downbinning in native input space; frame count is resampled with the same scale as XY, rather than preserving the original frame count. '
            + '; '.join(
                f'{spec.raw_value}->(t,Y,X)={spec.output_shape_t_y_x}'
                for spec in low_quality_downbin_specs
            )
        )
        spec_notes.append(
            'Low-quality NRRDs use the same decomposed component layer stack as the full NRRD, '
            'streamed directly into the requested low-quality geometry with its own manifest sidecar, and scheduled in the background alongside full-size outputs so full and low-quality NRRD payload streams can run concurrently when output workers are available.'
        )
    if (int(T), int(H), int(W)) != (int(input_T), int(input_H), int(input_W)):
        spec_notes.append(
            f'Working volume resized to v12.2.0 approximately-cubic processing geometry '
            f'(t,Y,X)=({int(T)},{int(H)},{int(W)}); final default outputs are restored to the original '
            f'input geometry (t,Y,X)=({int(input_T)},{int(input_H)},{int(input_W)}).'
        )
    spec_notes.append(
        f'Sagittal enabled={bool(enable_sagittal)}, Coronal enabled={bool(enable_coronal)}; '
        'their non-90 degree Cartesian augmentations use clamp-to-frame black fill rather than expanded padding.'
    )
    if tilt_angles:
        active_tilt_labels = ', '.join(pretty_view_name(v) for v in views if is_tilted_view(v))
        spec_notes.append(
            f'Generalized v12 Tilted Views active from --tilt_view={list(tilt_views)}, '
            f'--tilt_direction={list(tilt_directions)}, --tilt_angle={[float(v) for v in tilt_angles]}. '
            'Tilted base views do not need to be enabled as upright Cartesian views; each base/direction/signed-angle configuration remains independent until final union. '
            f'Active tilted configurations: {active_tilt_labels}'
        )
    else:
        spec_notes.append('Tilted Views disabled because --tilt_angle resolved to 0/no non-zero angles.')
    if float(resolved_azimuth_angle) > 0.0:
        spec_notes.append(
            f'Radial enabled with azimuth spacing {float(resolved_azimuth_angle):.8g} degrees; '
            'Lanczos-3 radial sampling, wraparound Radial interpolation, and dense final backprojection are active.'
        )
    if any((v.family == 'radial' or is_tilted_view(v)) for v in views):
        spec_notes.append(
            'v12.2.11 final Radial and Tilted backprojection is CPU-only. The old CuPy/GPU backprojection slot is disabled so the GPU remains dedicated to YOLO inference; each backprojection set receives the full resolved CPU slice-worker budget.'
        )
    spec_notes.append(
        'NRRD export is decomposed by default in v12.2.0: layers are written on a trailing list axis as '
        '(X,Y,t,layer), with manifest JSON and SegmentN metadata for view, source, YOLO-vs-bridge, '
        'tile acceptance category, pre-smoothing union, smoothing pass results, and final output. '
        'The NRRD payload is pigz-streamed directly from backing layers without materializing the full 4D decomposed payload; '
        'The layer-slowest writer uses a RAM-aware one-layer (t,Y,X) chunk buffer and writes layers sequentially without per-voxel interleave or slice transposes. '
        'Tune YOLO_TTA_NRRD_STREAM_BUFFER_MIB, YOLO_TTA_NRRD_STREAM_BUFFER_FRACTION, YOLO_TTA_NRRD_STREAM_BUFFER_MAX_GIB, '
        'and YOLO_TTA_NRRD_PIGZ_LEVEL if needed. '
        "Segment extents are now recorded when each NRRD layer/cvol store is materialized and reused during final packaging; only missing metadata or disabled YOLO_TTA_NRRD_PRECOMPUTED_SEGMENT_EXTENTS falls back to a layer scan. Raw cvol/ctile backing paths cache each layer's chunks.bin in RAM while that layer is streamed via YOLO_TTA_NRRD_CACHE_RAW_BBOX_LAYERS_IN_RAM=1, then release the per-layer cache. "
        'Transient NRRD projection, before-pass, and bridge-delta workspaces prefer anonymous RAM and fall back to disk only when the workspace budget requires it. '
        f'Legacy single-volume NRRD writing is retained only for deprecated internal callers without layer refs; space={NRRD_SPACE}.'
    )
    if cpu_retina_masks_enabled():
        spec_notes.append(
            'Deferred CPU retina-mask reconstruction active: Ultralytics native mask upsampling is bypassed on GPU; '
            'compact mask protos/coefficients/boxes are copied and bbox-ROI retina masks are reconstructed in prediction-result workers, '
            'so the scheduler/model-stream thread does not perform per-slice full-mask CPU copies.'
        )
    else:
        spec_notes.append('Using Ultralytics native retina_masks=True compatibility mode because YOLO_TTA_CPU_RETINA_MASKS=0 or the CPU-retina patch was unavailable.')
    if bool(troubleshooting_outputs_enabled):
        spec_notes.append(
            '--troubleshooting active: writing FFV1 MKV overlays for each active full-frame native view and each available consolidated tiled prediction set; '
            'temporary scratch retention is not implied.'
        )
    if bool(keep_temp_artifacts):
        spec_notes.append('YOLO_TTA_KEEP_TEMP=1 active: temporary scratch artifacts are retained independently of --troubleshooting.')
    if bool(gaussian_smoothing_enabled):
        spec_notes.append(
            f'Gaussian smoothing active by v12.2.0 explicit-flag rule: sigma={float(gaussian_smoothing_sigma):g} voxel(s), '
            f'passes={int(gaussian_smoothing_passes)}; applied after final union/optional 3D void fill and before --keep_objects. '
            'The default smoothing backend attempts chunked GPU execution through CuPy/cupyx.scipy.ndimage with halo/core writes, then falls back to scipy.ndimage on CPU if the GPU backend is unavailable.'
        )
    else:
        if not bool(gaussian_smoothing_cli_requested):
            spec_notes.append('Gaussian smoothing disabled by v12.2.0 activation rule because neither Gaussian flag was explicitly set.')
        elif bool(gaussian_smoothing_disabled_by_zero):
            spec_notes.append('Gaussian smoothing disabled because at least one explicitly supplied Gaussian flag was set to 0.')
        else:
            spec_notes.append('Gaussian smoothing disabled because the resolved sigma or pass count was not positive.')
    spec_notes.append(
        'Interpolation endpoint discovery uses the v12.2.0 per-slice connected-component scan backed by cached per-slice component tables. '
        'Projection candidate search runs on source-component local SDF crops, and variable-cost seed planning is consumed through a bounded unordered completion queue; '
        'skeletonization is never used for interpolation and runs only when --save_skeleton is requested. '
        'Optional skeleton output uses slice-parallel pore preconditioning plus component/chunk based 3D Lee thinning; '
        'overlay videos draw the skeleton as fully opaque red on top of the 50% blue mask overlay.'
    )
    spec_notes.append(
        f'Cubic resize T-axis backend={_cube_t_axis_resize_backend()}; set YOLO_TTA_CUBE_T_RESIZE_BACKEND=slice_exact '
        'to recover the previous endpoint-aligned per-slice interpolation path.'
    )
    if transverse_inference_disabled:
        note = (
            'v12.2.0 specification note: Section 2.1.1 says Transverse is enabled by default, while '
            'View Flags item 1 adds --disable_transverse to skip Transverse inferencing. This implementation '
            'keeps Transverse geometry/output paths available and removes only the standard Transverse full-frame '
            'and tiled prediction jobs.'
        )
        spec_notes.append(note)
        print(note)

    if model_load_future is not None:
        print(f'Waiting for background model load to finish: {model_name} ({model_paths[0]})')
        yolo_model = model_load_future.result()
        if model_load_executor is not None:
            model_load_executor.shutdown(wait=True)
            model_load_executor = None
    else:
        print(f'Loading model: {model_name} ({model_paths[0]})')
        yolo_model = load_ultralytics_model(model_paths[0], task='segment')
    # v12.2.0 has no multiple-model inference. A single-item list is retained only to minimize churn in
    # scheduling structures keyed by model stem.
    yolo_models: List[Tuple[str, object]] = [(model_name, yolo_model)]
    yolo_by_model_name: Dict[str, object] = {model_name: yolo_model}

    pred_cfg = PredictConfig(
        imgsz=args.imgsz,
        conf=args.conf,
        device=str(args.device),
        half=bool(args.half),
        int8=bool(args.int8),
        batch=max(1, int(args.batch)),
    )


    worker_budget = int(default_worker_budget())
    augmentation_workers = resolve_worker_count(
        0,
        'YOLO_TTA_AUG_WORKERS',
        worker_budget,
        max_tasks=max(1, max((v.num_slices for v in inference_views), default=1)),
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
    output_frame_workers = max(1, _env_int('YOLO_TTA_OUTPUT_FRAME_WORKERS', max(1, min(_cpu_count(), output_workers))))
    slice_postprocess_workers = max(1, int(augmentation_workers))
    predict_postprocess_cap = max(1, _env_int('YOLO_TTA_PREDICT_POSTPROCESS_MAX_WORKERS', max(1, _cpu_count())))
    predict_postprocess_workers = max(
        1,
        min(
            int(predict_postprocess_cap),
            _env_int('YOLO_TTA_PREDICT_POSTPROCESS_WORKERS', slice_postprocess_workers),
        ),
    )

    parent_postprocess_workers = max(1, min(int(interpolation_workers), max(1, len(yolo_models) * max(1, len(inference_views)))))
    (
        parent_interpolation_overlap,
        parent_interpolation_task_workers_default,
        parent_interpolation_task_workers,
    ) = resolve_parent_interpolation_worker_allocation(
        worker_budget=int(worker_budget),
        parent_postprocess_workers=int(parent_postprocess_workers),
    )

    tile_postprocess_workers_default = int(worker_budget)
    tile_postprocess_workers = max(1, _env_int('YOLO_TTA_TILE_POSTPROCESS_WORKERS', tile_postprocess_workers_default))
    tile_slice_postprocess_workers_default = int(worker_budget)
    tile_slice_postprocess_workers = max(
        1,
        _env_int('YOLO_TTA_TILE_SLICE_WORKERS', tile_slice_postprocess_workers_default),
    )
    tile_interpolation_task_workers_default = max(1, _cpu_count())
    tile_interpolation_task_workers = max(
        1,
        _env_int('YOLO_TTA_TILE_INTERPOLATION_TASK_WORKERS', tile_interpolation_task_workers_default),
    )

    interpolation_process_backend_active = bool(interpolation_process_backend_enabled() and int(args.interpolate) > 0)
    interpolation_process_workers_default = max(
        1,
        min(
            2,
            max(1, int(parent_postprocess_workers)) + (1 if len(tile_configs) > 0 else 0),
        ),
    )
    interpolation_process_workers = (
        max(1, _env_int('YOLO_TTA_INTERPOLATION_PROCESS_WORKERS', interpolation_process_workers_default))
        if bool(interpolation_process_backend_active) else 0
    )

    print(f'Allocated CPU count: {_cpu_count()}')
    print(f'Worker budget: {worker_budget}')
    print('Worker oversubscription is intentional (default budget = 2x visible CPUs).')
    print(f'Augmentation workers: {augmentation_workers}')
    print(f'Slice-parallel postprocess workers: {slice_postprocess_workers}')
    print(f'Inference postprocess workers: {predict_postprocess_workers}')
    print(
        'Parent full-frame postprocess workers: '
        f'{parent_postprocess_workers} (expected interpolation overlap: {parent_interpolation_overlap}, '
        f'per-parent interpolation workers: {parent_interpolation_task_workers})'
    )
    print(
        'Tile postprocess workers: '
        f'{tile_postprocess_workers} (per-tile slice workers: {tile_slice_postprocess_workers}, '
        f'consolidated-tile interpolation workers: {tile_interpolation_task_workers})'
    )
    if bool(interpolation_process_backend_active):
        print(
            'Interpolation process backend: enabled '
            f'(process workers: {int(interpolation_process_workers)}, start_method={interpolation_process_start_method()}, '
            f'child cv2_threads={interpolation_process_cv2_threads()}, compiled kernels: {interpolation_compiled_kernels_status()})'
        )
    else:
        print('Interpolation process backend: disabled (legacy in-process interpolation path).')
    print(f'Background output workers: {output_workers} (frame workers per labels/TIFF task: {output_frame_workers})')
    max_predict_video_frames = max(1, max((int(v.num_slices) for v in inference_views), default=1))
    example_cpu_mask_workers = max(1, min(int(predict_postprocess_workers), int(max_predict_video_frames)))
    example_cpu_mask_pending = cpu_mask_postprocess_pending_limit(int(example_cpu_mask_workers), int(max_predict_video_frames))
    spec_notes.append(
        'YOLO result accumulation is bounded per in-memory prediction source by the number of pending CPU postprocess futures. ' 
        'worker_count=max(1, min(YOLO_TTA_PREDICT_POSTPROCESS_WORKERS, num_frames)) for the live v12 in-memory source path; ' 
        'v12.2.11 removes the former hard 32-worker ceiling; YOLO_TTA_PREDICT_POSTPROCESS_MAX_WORKERS now defaults to the visible CPU allocation. '
        'pending_limit=cpu_mask_postprocess_pending_limit(worker_count, num_frames) = max(worker_count, min(num_frames, max(YOLO_TTA_CPU_MASK_PENDING_FRAMES, worker_count*2))). ' 
        'The default YOLO_TTA_CPU_MASK_PENDING_FRAMES is 4096; setting it to 0 permits buffering all frames for that prediction source. ' 
        f'For this run, the largest active prediction source has {int(max_predict_video_frames)} frame(s), ' 
        f'example worker_count={int(example_cpu_mask_workers)}, pending_limit={int(example_cpu_mask_pending)}.'
    )
    spec_notes.append(
        f'Batched inference active with --batch={int(args.batch)}. StreamingYoloVolumeSource is used by default for full-frame/tile sources: it renders a bounded CPU prefetch window of H×W×1 slices and pads the final batch by repeating the last real slice; set YOLO_TTA_STREAMING_PREDICTION_SOURCES=0 to restore dense in-memory prediction-volume materialization.'
    )
    spec_notes.append(
        'Tiled masks are staged into one consolidated parent-view canvas per --tile_size/--tile_stride configuration before gating; '
        'the canvas gate is slice-local and connected-component based, so seam-split tile objects are reassembled before support testing. '
        'A tile mask shape guard validates/resizes every postprocessed tile to its parent view-native shape. '
        f'v12.2.11 keeps waiting tiles and tile-set/category accumulators in RAM by default (YOLO_TTA_SPILL_WAITING_TILES={int(waiting_tile_spill_enabled())}, YOLO_TTA_TILE_ACCUMULATORS_IN_RAM={int(tile_intermediate_accumulators_prefer_memory())}); disk ctile spill is now opt-in.'
    )
    spec_notes.append(
        'Postprocessed tiles waiting for parent support stay in RAM by default and use ctile-mask-v2-raw only when YOLO_TTA_SPILL_WAITING_TILES=1; decomposed NRRD/support layers use cvol-mask-v2-raw where enabled: empty slices are elided, '
        'nonempty slices are cropped to their nonzero bbox and written as raw uint8 payload bytes; bitpacking and LZ4 are not used. '
        'Storage review: dense gray8 source/processing volumes stay raw; uint8 confidence maps are allocated only when --min_conf > 0 and stay raw because they are not sparse binary masks; FFV1/MKV/TIFF outputs remain codec-compressed; labels/summary JSON are negligible; retained sparse binary scratch/tile accumulators can be archived as raw bbox cvol when YOLO_TTA_KEEP_TEMP and YOLO_TTA_ARCHIVE_TEMP_BINARY_VOLUMES are enabled.'
    )
    spec_notes.append(
        f'Fused cleanup backend={cleanup_backend()}; set YOLO_TTA_CLEANUP_BACKEND=scipy to use the previous scipy.ndimage cleanup path.'
    )
    spec_notes.append(
        'Interpolation labeling uses parallel 2D per-slice connected-component labeling, parallel adjacent-slice pair extraction, '
        'row-blocked parallel compact relabeling, and unordered prebuilt per-slice component tables for endpoint scanning. '
        f'Per-parent interpolation task workers default to worker_budget / expected_live_parent_overlap = {int(parent_interpolation_task_workers_default)} '
        f'using YOLO_TTA_INTERPOLATION_CONCURRENT_PARENTS={int(parent_interpolation_overlap)}; '
        'YOLO_TTA_INTERPOLATION_TASK_WORKERS still overrides the exact per-parent worker count. '
        'Tune YOLO_TTA_INTERPOLATION_LABEL_WORKERS, YOLO_TTA_INTERPOLATION_PAIR_WORKERS, YOLO_TTA_INTERPOLATION_COMPACT_WORKERS, '
        'YOLO_TTA_INTERPOLATION_COMPACT_RELABEL_ROWS, and YOLO_TTA_INTERPOLATION_CONCURRENT_PARENTS if needed.'
    )
    spec_notes.append(
        'v12.2.7 interpolation process isolation active by default: full-frame and consolidated-tile interpolation passes reopen uint8 mask volumes from disk-backed memmaps in a ProcessPoolExecutor worker and return only small stats. '
        f'Process backend enabled={bool(interpolation_process_backend_active)}, process_workers={int(interpolation_process_workers)}, start_method={interpolation_process_start_method()}, '
        f'fallback_on_worker_failure={bool(interpolation_process_fallback_enabled())}. Anonymous in-memory mask arrays are copied once to a process memmap before interpolation, avoiding multi-GiB pickle payloads.'
    )
    spec_notes.append(
        f'v12.2.7 compiled interpolation kernels: {interpolation_compiled_kernels_status()}. '
        'The compiled kernel accelerates projection-candidate discovery in seed planning with Numba nogil=True when numba is installed; the exact Python candidate search remains the fallback.'
    )

    output_manager = BackgroundOutputManager(max_workers=output_workers)

    if augmentation_workers > 1 or interpolation_workers > 1 or slice_postprocess_workers > 1 or output_workers > 1:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    interpolation_process_executor: Optional[ProcessPoolExecutor] = None
    if bool(interpolation_process_backend_active):
        interpolation_process_executor = create_interpolation_process_executor(int(interpolation_process_workers))
        set_interpolation_process_executor(interpolation_process_executor, int(interpolation_process_workers))
    else:
        set_interpolation_process_executor(None, 0)

    dense_tiling_active = len(tile_configs) > 0
    view_infos_by_name: Dict[str, ViewInfo] = {view.name: view for view in views}
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    native_view_support_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    parent_mask_support_by_model: Dict[str, Dict[str, object]] = {model_name: {} for model_name, _ in yolo_models}
    parent_bridge_support_by_model: Dict[str, Dict[str, object]] = {model_name: {} for model_name, _ in yolo_models}
    radial_native_output_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    tilted_native_output_by_model: Dict[str, Dict[str, np.ndarray]] = {model_name: {} for model_name, _ in yolo_models}
    nrrd_layer_refs: List[NrrdLayerRef] = []
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
        'sagittal': 0,
        'coronal': 0,
        'radial': 0,
    }
    view_prediction_labels: Dict[str, str] = {
        'transverse': 'Transverse',
        'sagittal': 'Sagittal',
        'coronal': 'Coronal',
        'radial': 'Radial',
    }
    for _view_for_stats in views:
        if is_tilted_view(_view_for_stats):
            view_prediction_stats.setdefault(str(_view_for_stats.summary_family), 0)
            view_prediction_labels[str(_view_for_stats.summary_family)] = pretty_view_name(_view_for_stats)
    interpolation_stats: List[Dict[str, object]] = []

    inference_view_names = {v.name for v in inference_views}
    for view in views:
        jobs = build_aug_jobs_for_view(
            view=view,
            angles=angles,
            out_size=args.imgsz,
            temp_dir=temp_dir,
        )
        aug_jobs_by_view[view.name] = jobs
        aug_job_lookup_by_view[view.name] = {job.aug_id: job for job in jobs}
        if dense_tiling_active and view.name in inference_view_names:
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

    tile_expected_by_parent_config: Dict[Tuple[str, str, str], int] = {}
    tile_config_expected_by_parent: Dict[Tuple[str, str], int] = {}
    if dense_tiling_active:
        for view in inference_views:
            jobs_by_config = tile_jobs_by_view_config.get(view.name, {})
            if not jobs_by_config:
                continue
            expected_for_view = sum(len(jobs) for jobs in jobs_by_config.values())
            if expected_for_view <= 0:
                continue
            for model_name, _ in yolo_models:
                parent_key = (str(model_name), str(view.name))
                active_config_count = 0
                for config_id, cfg_jobs in jobs_by_config.items():
                    if not cfg_jobs:
                        continue
                    tile_expected_by_parent_config[(str(model_name), str(view.name), str(config_id))] = int(len(cfg_jobs))
                    active_config_count += 1
                tile_config_expected_by_parent[parent_key] = int(active_config_count)

    view_frame_caches: Dict[str, np.ndarray] = {}
    view_frame_cache_paths: Dict[str, Path] = {}
    view_frame_cache_lock = threading.Lock()

    def _get_view_frame_cache(view: ViewInfo) -> Optional[np.ndarray]:
        if not should_cache_view_frames(view, dense_tiling_active):
            return None
        cached = view_frame_caches.get(view.name)
        if cached is not None:
            return cached
        with view_frame_cache_lock:
            cached = view_frame_caches.get(view.name)
            if cached is not None:
                return cached
            wait_for_volume_ready(volume_rgb)
            cache_path = temp_dir / 'view_frames' / f'{view.name}.gray8.dat'
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_mm = build_view_frame_cache(
                volume_rgb=volume_rgb,
                view=view,
                out_path=cache_path,
                desc=f'{view.name} native frame cache',
                prefer_memory=True,
                workers=max(1, int(augmentation_workers)),
            )
            view_frame_caches[view.name] = cache_mm
            view_frame_cache_paths[view.name] = cache_path
            return cache_mm

    baseline_union_by_model_view: Dict[Tuple[str, str], np.ndarray] = {}
    baseline_confmap_by_model_view: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    baseline_union_paths: Dict[Tuple[str, str], Path] = {}
    baseline_confmap_paths: Dict[Tuple[str, str], Optional[Path]] = {}
    baseline_slice_locks_by_model_view: Dict[Tuple[str, str], List[threading.Lock]] = {}
    fullframe_remaining: Dict[Tuple[str, str], int] = {}

    for view in inference_views:
        for model_name, _ in yolo_models:
            # v12.2.11 lazy allocation: do not zero every full-view accumulator before the
            # first prediction.  On 20+ view runs those eager zeros can touch hundreds of GiB
            # and dominate time-to-first-prediction.  Allocate a view's union/confidence
            # workspaces only when its first full-frame prediction is about to run.
            fullframe_remaining[(model_name, view.name)] = int(len(aug_jobs_by_view[view.name]))

    total_fullframe_jobs = sum(len(aug_jobs_by_view.get(view.name, [])) for view in inference_views)
    total_tile_prediction_jobs = sum(
        len(tile_jobs_by_aug.get((view.name, aug_job.aug_id), []))
        for view in inference_views
        for aug_job in aug_jobs_by_view[view.name]
    )
    total_prediction_volume_build_tasks = int(total_fullframe_jobs + total_tile_prediction_jobs)
    prediction_volume_queue_slots = max(1, _env_int('YOLO_TTA_PREDICTION_VOLUME_QUEUE_SLOTS', 4))
    active_build_slot_default = max(
        1,
        min(
            int(augmentation_workers),
            int(prediction_volume_queue_slots),
            max(1, int(total_prediction_volume_build_tasks)),
        ),
    )
    requested_build_workers = max(
        1,
        _env_int('YOLO_TTA_VOLUME_BUILD_WORKERS', int(active_build_slot_default)),
    )
    prediction_volume_builder_workers = max(
        1,
        min(
            int(augmentation_workers),
            int(prediction_volume_queue_slots),
            max(1, int(total_prediction_volume_build_tasks)),
            int(requested_build_workers),
        ),
    )
    per_prediction_volume_workers = max(1, int(max(1, augmentation_workers) // max(1, prediction_volume_builder_workers)))
    async_prediction_accumulation_active = bool(async_predict_postprocess_enabled())
    async_prediction_multiview_locking_active = bool(len(angles) > 1 and async_prediction_accumulation_active)
    async_prediction_join_worker_count = async_predict_join_workers(max(2, int(prediction_volume_queue_slots) + 1))
    default_async_result_workers = max(1, min(int(predict_postprocess_workers), max(1, _cpu_count())))
    async_prediction_result_worker_count = max(
        1,
        min(
            int(predict_postprocess_workers),
            _env_int('YOLO_TTA_ASYNC_PREDICT_RESULT_WORKERS', int(default_async_result_workers)),
        ),
    )
    if bool(async_prediction_accumulation_active):
        print(
            'Async prediction accumulation: enabled '
            f'(angles={len(angles)}, result workers={int(async_prediction_result_worker_count)}, '
            f'join workers={int(async_prediction_join_worker_count)}, '
            f'per-slice locks for multi-angle full-frame writes={bool(async_prediction_multiview_locking_active)})'
        )
        spec_notes.append(
            'v12.2.12 async prediction accumulation is active for all angle counts by default. '
            'YOLO result detach/copy, native inverse-mapping, and optional streaming cleanup are queued to a shared prediction-result executor; '
            'for multi-angle full-frame accumulation, per-slice locks protect shared union/confidence slices so the scheduler can start the next source while CPU result work drains.'
        )
    else:
        print('Async prediction accumulation: disabled by YOLO_TTA_ASYNC_PREDICT_POSTPROCESS=0; prediction sources are drained synchronously.')
        spec_notes.append(
            'v12.2.12 async prediction accumulation was disabled by configuration; prediction sources are drained synchronously.'
        )
    spec_notes.append(
        'v12.2.12 streaming prediction sources are active by default: full-frame and tiled YOLO inputs render only a bounded CPU prefetch window before model.predict starts, rather than materializing a complete (slice,--imgsz,--imgsz) volume first. '
        'Set YOLO_TTA_STREAMING_PREDICTION_SOURCES=0 to restore the legacy dense prediction-volume path; per-source prediction memmap flushes are still skipped by default unless YOLO_TTA_FLUSH_PREDICTION_VOLUME_ON_BUILD=1 or YOLO_TTA_PREDICT_FLUSH_EACH_VOLUME=1 is set.'
    )
    print(
        f'Streaming prediction-source preparers: {prediction_volume_builder_workers} '
        f'(per-source render workers: {per_prediction_volume_workers}, source tasks: {total_prediction_volume_build_tasks}, '
        f'queued-source bound: {prediction_volume_queue_slots})'
    )
    spec_notes.append(
        'v12.2.12 prediction scheduler active: full-frame and tiled YOLO sources normally stream H×W×1 frames through StreamingYoloVolumeSource, so the GPU can begin after the first prefetch batch while CPU workers continue 2D resize/warp rendering behind it. '
        f'The build queue is bounded to {int(prediction_volume_queue_slots)} queued prediction source(s) beyond the current inference; the v12.2.0 default is four queued sources, for a normal total bound of five including the current inference source. '
        'v12.2.11 lazily allocates each full-view union/confidence workspace only when that view first reaches inference, avoiding an eager all-views zero-fill before the first prediction.'
    )
    if dense_tiling_active:
        spec_notes.append(
            'Tiled prediction sources follow the deterministic tile footprint, stride order, angle variant, '
            'and inverse-mapping rules from the v12.2.0 specification.'
        )

    prediction_volume_executor = ThreadPoolExecutor(max_workers=int(prediction_volume_builder_workers), thread_name_prefix='prediction-volume')
    prediction_result_executor = ThreadPoolExecutor(max_workers=int(async_prediction_result_worker_count), thread_name_prefix='predict-result')
    prediction_join_executor = ThreadPoolExecutor(max_workers=int(async_prediction_join_worker_count), thread_name_prefix='predict-join')
    parent_postprocess_executor = ThreadPoolExecutor(max_workers=int(parent_postprocess_workers), thread_name_prefix='parent-postprocess')
    tile_postprocess_executor = ThreadPoolExecutor(max_workers=int(tile_postprocess_workers), thread_name_prefix='tile-postprocess')

    pending_prediction_build_jobs: deque[Tuple[str, ViewInfo, object]] = deque()
    for view, aug_job in iter_aug_jobs_round_robin(inference_views, aug_jobs_by_view):
        pending_prediction_build_jobs.append(('fullframe', view, aug_job))
        if dense_tiling_active:
            for tile_job in tile_jobs_by_aug.get((view.name, aug_job.aug_id), []):
                pending_prediction_build_jobs.append(('tile', view, tile_job))

    prediction_volume_futures: Dict[Future, Tuple[str, ViewInfo, object]] = {}
    pending_prediction_volume_futures: set[Future] = set()
    ready_fullframe: deque[Tuple[ViewInfo, AugJob, PredictionVolumeRef]] = deque()
    ready_tile_infer: deque[Tuple[str, ViewInfo, DenseTileJob, PredictionVolumeRef]] = deque()
    tile_inference_done: set[Tuple[str, str, str]] = set()
    prediction_accumulation_futures: Dict[Future, Dict[str, object]] = {}

    view_processing_futures: Dict[Future, Tuple[str, str]] = {}
    view_processing_submitted: set[Tuple[str, str]] = set()
    tile_cleanup_futures: Dict[Future, Tuple[str, str, str, str]] = {}
    postprocessed_tiles_waiting_by_parent: Dict[Tuple[str, str], Dict[str, object]] = {}
    tile_finalize_futures: Dict[Future, Tuple[str, str, str, str]] = {}
    tile_config_gate_futures: Dict[Future, Tuple[str, str, str]] = {}
    tile_consolidation_futures: Dict[Future, Tuple[str, str]] = {}
    tile_config_accumulator_by_key: Dict[Tuple[str, str, str], np.ndarray] = {}
    tile_config_accumulator_paths: Dict[Tuple[str, str, str], Path] = {}
    tile_config_accumulator_locks: Dict[Tuple[str, str, str], threading.Lock] = {}
    tile_accumulator_by_parent: Dict[Tuple[str, str], np.ndarray] = {}
    tile_accumulator_paths: Dict[Tuple[str, str], Path] = {}
    tile_parent_mask_accumulator_by_parent: Dict[Tuple[str, str], np.ndarray] = {}
    tile_parent_bridge_accumulator_by_parent: Dict[Tuple[str, str], np.ndarray] = {}
    tile_completed_by_parent_config: Dict[Tuple[str, str, str], set[str]] = {}
    tile_config_gated_by_parent: Dict[Tuple[str, str], set[str]] = {}
    tile_config_gate_submitted: set[Tuple[str, str, str]] = set()
    tile_consolidation_submitted: set[Tuple[str, str]] = set()

    def _prediction_volume_queue_depth() -> int:
        return int(len(pending_prediction_volume_futures) + len(ready_fullframe) + len(ready_tile_infer))

    def _make_streaming_fullframe_ref(view: ViewInfo, aug_job: AugJob) -> PredictionVolumeRef:
        if not aug_job.meta_path.exists():
            write_aug_job_meta(aug_job, view)
        render_workers = streaming_prediction_source_workers(int(per_prediction_volume_workers), int(view.num_slices))
        prefetch_frames = streaming_prediction_source_prefetch_frames(max(1, int(args.batch)))

        def _render(idx: int) -> np.ndarray:
            return render_fullframe_frame_for_job(
                volume_rgb=volume_rgb,
                view=view,
                job=aug_job,
                frame_idx=int(idx),
                view_frames=_get_view_frame_cache(view),
            )

        name = f'Streaming full-frame prediction source {view.name}/{aug_job.aug_id}'
        source = StreamingYoloVolumeSource(
            _render,
            num_frames=int(view.num_slices),
            name=name,
            batch_size=max(1, int(args.batch)),
            out_size=int(aug_job.aff.out_size),
            render_workers=int(render_workers),
            prefetch_frames=int(prefetch_frames),
            autostart=bool(streaming_prediction_source_autostart_enabled()),
        )
        return PredictionVolumeRef(
            array=None,
            path=None,
            name=name,
            view_name=str(view.name),
            job_id=str(aug_job.aug_id),
            kind='fullframe',
            source=source,
        )

    def _make_streaming_tile_ref(view: ViewInfo, tile_job: DenseTileJob) -> PredictionVolumeRef:
        if not tile_job.meta_path.exists():
            write_dense_tile_job_meta(tile_job)
        render_workers = streaming_prediction_source_workers(int(per_prediction_volume_workers), int(view.num_slices))
        prefetch_frames = streaming_prediction_source_prefetch_frames(max(1, int(args.batch)))

        def _render(idx: int) -> np.ndarray:
            return render_dense_tile_frame_for_job(
                volume_rgb=volume_rgb,
                view=view,
                tile_job=tile_job,
                frame_idx=int(idx),
                view_frames=_get_view_frame_cache(view),
            )

        name = f'Streaming tile prediction source {view.name}/{tile_job.tile_id}'
        source = StreamingYoloVolumeSource(
            _render,
            num_frames=int(view.num_slices),
            name=name,
            batch_size=max(1, int(args.batch)),
            out_size=int(tile_job.out_size),
            render_workers=int(render_workers),
            prefetch_frames=int(prefetch_frames),
            autostart=bool(streaming_prediction_source_autostart_enabled()),
        )
        return PredictionVolumeRef(
            array=None,
            path=None,
            name=name,
            view_name=str(view.name),
            job_id=str(tile_job.tile_id),
            kind='tile',
            source=source,
        )

    def _submit_prediction_volume_build(kind: str, view: ViewInfo, job_obj: object) -> None:
        if str(kind) == 'fullframe':
            aug_job = job_obj
            assert isinstance(aug_job, AugJob)
            if streaming_prediction_sources_enabled():
                fut = prediction_volume_executor.submit(_make_streaming_fullframe_ref, view, aug_job)
            else:
                out_path = temp_dir / 'prediction_volumes' / 'fullframe' / view.name / f'{view.name}_{aug_job.aug_id}.u8.dat'
                fut = prediction_volume_executor.submit(
                    materialize_fullframe_prediction_volume_for_job,
                    volume_rgb,
                    view,
                    aug_job,
                    out_path=out_path,
                    view_frames=_get_view_frame_cache(view),
                    workers=int(per_prediction_volume_workers),
                    show_progress=False,
                )
        elif str(kind) == 'tile':
            tile_job = job_obj
            assert isinstance(tile_job, DenseTileJob)
            if streaming_prediction_sources_enabled():
                fut = prediction_volume_executor.submit(_make_streaming_tile_ref, view, tile_job)
            else:
                out_path = temp_dir / 'prediction_volumes' / 'tiles' / view.name / str(tile_job.config_id) / f'{tile_job.tile_id}.u8.dat'
                fut = prediction_volume_executor.submit(
                    materialize_dense_tile_prediction_volume_for_job,
                    volume_rgb,
                    view,
                    tile_job,
                    out_path=out_path,
                    view_frames=_get_view_frame_cache(view),
                    workers=int(per_prediction_volume_workers),
                    show_progress=False,
                )
        else:  # pragma: no cover
            raise ValueError(f'Unknown prediction volume build kind: {kind}')
        prediction_volume_futures[fut] = (str(kind), view, job_obj)
        pending_prediction_volume_futures.add(fut)

    def _pump_prediction_volume_build_queue() -> None:
        while pending_prediction_build_jobs and _prediction_volume_queue_depth() < int(prediction_volume_queue_slots):
            kind, view, job_obj = pending_prediction_build_jobs.popleft()
            _submit_prediction_volume_build(str(kind), view, job_obj)

    def _drain_completed_prediction_volume_futures() -> None:
        for fut in list(pending_prediction_volume_futures):
            if not fut.done():
                continue
            pending_prediction_volume_futures.remove(fut)
            kind, view, job_obj = prediction_volume_futures.pop(fut)
            pred_ref = fut.result()
            if str(kind) == 'fullframe':
                assert isinstance(job_obj, AugJob)
                ready_fullframe.append((view, job_obj, pred_ref))
            else:
                assert isinstance(job_obj, DenseTileJob)
                for model_name, _ in yolo_models:
                    ready_tile_infer.append((str(model_name), view, job_obj, pred_ref))
        _pump_prediction_volume_build_queue()

    def _ensure_baseline_workspaces(model_name: str, view: ViewInfo) -> None:
        key = (str(model_name), str(view.name))
        if key in baseline_union_by_model_view:
            return
        union_path = temp_dir / 'union' / str(model_name) / f'{view.name}.union.u8.dat'
        confmap_path = temp_dir / 'union' / str(model_name) / f'{view.name}.confmap.u8.dat'
        union_path.parent.mkdir(parents=True, exist_ok=True)

        baseline_union_by_model_view[key] = allocate_workspace_array(
            shape=(int(view.num_slices), int(view.src_h), int(view.src_w)),
            dtype=np.uint8,
            path=union_path,
            desc=f'{model_name}/{view.name} baseline union workspace',
            prefer_memory=True,
        )
        baseline_union_paths[key] = union_path
        baseline_slice_locks_by_model_view[key] = [threading.Lock() for _ in range(int(view.num_slices))]
        if float(args.min_conf) > 0.0:
            baseline_confmap_by_model_view[key] = allocate_workspace_array(
                shape=(int(view.num_slices), int(view.src_h), int(view.src_w)),
                dtype=np.uint8,
                path=confmap_path,
                desc=f'{model_name}/{view.name} baseline confidence workspace',
                prefer_memory=True,
            )
            baseline_confmap_paths[key] = confmap_path
        else:
            baseline_confmap_by_model_view[key] = None
            baseline_confmap_paths[key] = None

    def _submit_view_prepare(model_name: str, view: ViewInfo) -> None:
        key = (str(model_name), str(view.name))
        if key in view_processing_submitted:
            return
        view_processing_submitted.add(key)
        union_mm = baseline_union_by_model_view.pop(key)
        confmap_mm = baseline_confmap_by_model_view.pop(key)
        union_path = baseline_union_paths.pop(key)
        confmap_path = baseline_confmap_paths.pop(key)
        baseline_slice_locks_by_model_view.pop(key, None)
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
            keep_temp=bool(keep_temp_artifacts),
            slice_workers=int(slice_postprocess_workers),
            interpolation_task_workers=int(parent_interpolation_task_workers),
            nrrd_layers_enabled=bool(nrrd_layers_needed),
            precleaned_slice_cleanup=bool(single_angle_streaming_cleanup_active),
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
            prefer_memory=tile_intermediate_accumulators_prefer_memory(),
            reserve_bytes=tile_intermediate_accumulator_reserve_bytes(),
        )
        tile_accumulator_by_parent[key] = acc
        tile_accumulator_paths[key] = acc_path
        return acc

    def _get_tile_config_accumulator(model_name: str, view_name: str, config_id: str) -> np.ndarray:
        key = (str(model_name), str(view_name), str(config_id))
        acc = tile_config_accumulator_by_key.get(key)
        if acc is not None:
            return acc
        view = view_infos_by_name[str(view_name)]
        acc_path = temp_dir / 'tile_consolidated_by_config' / str(model_name) / str(view_name) / str(config_id) / 'raw_canvas.u8.dat'
        acc = allocate_workspace_array(
            shape=(int(view.num_slices), int(view.src_h), int(view.src_w)),
            dtype=np.uint8,
            path=acc_path,
            desc=f'{model_name}/{view_name}/{config_id} raw consolidated tile-set canvas',
            prefer_memory=tile_intermediate_accumulators_prefer_memory(),
            reserve_bytes=tile_intermediate_accumulator_reserve_bytes(),
        )
        tile_config_accumulator_by_key[key] = acc
        tile_config_accumulator_paths[key] = acc_path
        tile_config_accumulator_locks.setdefault(key, threading.Lock())
        return acc


    def _get_tile_category_accumulator(model_name: str, view_name: str, category: str) -> np.ndarray:
        key = (str(model_name), str(view_name))
        category_norm = str(category)
        store = (
            tile_parent_mask_accumulator_by_parent
            if category_norm == 'parent_mask'
            else tile_parent_bridge_accumulator_by_parent
        )
        acc = store.get(key)
        if acc is not None:
            return acc
        view = view_infos_by_name[str(view_name)]
        acc_path = temp_dir / 'tile_consolidated' / str(model_name) / str(view_name) / f'gated_or_accepted_by_{category_norm}.u8.dat'
        acc = allocate_workspace_array(
            shape=(int(view.num_slices), int(view.src_h), int(view.src_w)),
            dtype=np.uint8,
            path=acc_path,
            desc=f'{model_name}/{view_name} consolidated gated-tile accumulator accepted by {category_norm}',
            prefer_memory=tile_intermediate_accumulators_prefer_memory(),
            reserve_bytes=tile_intermediate_accumulator_reserve_bytes(),
        )
        store[key] = acc
        return acc


    def _parent_destination_ready(model_name: str, view_name: str) -> bool:
        view = view_infos_by_name[str(view_name)]
        if view.family == 'radial':
            return str(view_name) in radial_native_output_by_model.get(str(model_name), {})
        if is_tilted_view(view):
            return str(view_name) in tilted_native_output_by_model.get(str(model_name), {})
        return str(view_name) in view_volumes_by_model.get(str(model_name), {})

    def _parent_destination_volume(model_name: str, view_name: str) -> np.ndarray:
        view = view_infos_by_name[str(view_name)]
        if view.family == 'radial':
            return radial_native_output_by_model[str(model_name)][str(view_name)]
        if is_tilted_view(view):
            return tilted_native_output_by_model[str(model_name)][str(view_name)]
        return view_volumes_by_model[str(model_name)][str(view_name)]

    def _maybe_submit_tile_consolidation(model_name: str, view_name: str) -> None:
        parent_key = (str(model_name), str(view_name))
        if parent_key in tile_consolidation_submitted:
            return
        expected_configs = int(tile_config_expected_by_parent.get(parent_key, 0))
        if expected_configs <= 0:
            return
        if len(tile_config_gated_by_parent.get(parent_key, set())) < expected_configs:
            return
        if not _parent_destination_ready(str(model_name), str(view_name)):
            return

        tile_consolidation_submitted.add(parent_key)
        acc = tile_accumulator_by_parent.get(parent_key)
        if acc is None:
            # All tile-set canvases completed but none had parent-supported components.
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
            keep_temp=bool(keep_temp_artifacts),
            slice_workers=int(tile_slice_postprocess_workers),
            interpolation_task_workers=int(tile_interpolation_task_workers),
            nrrd_layers_enabled=bool(nrrd_layers_needed),
            tile_parent_mask_accumulator_mm=tile_parent_mask_accumulator_by_parent.get(parent_key),
            tile_parent_bridge_accumulator_mm=tile_parent_bridge_accumulator_by_parent.get(parent_key),
        )
        tile_consolidation_futures[fut] = parent_key

    def _mark_tile_config_gated(model_name: str, view_name: str, config_id: str) -> None:
        parent_key = (str(model_name), str(view_name))
        gated = tile_config_gated_by_parent.setdefault(parent_key, set())
        gated.add(str(config_id))
        _maybe_submit_tile_consolidation(str(model_name), str(view_name))

    def _maybe_submit_tile_config_gate(model_name: str, view_name: str, config_id: str) -> None:
        key = (str(model_name), str(view_name), str(config_id))
        if key in tile_config_gate_submitted:
            return
        expected = int(tile_expected_by_parent_config.get(key, 0))
        if expected <= 0:
            return
        if len(tile_completed_by_parent_config.get(key, set())) < expected:
            return
        support_by_view = native_view_support_by_model.get(str(model_name), {})
        if str(view_name) not in support_by_view:
            return

        tile_config_gate_submitted.add(key)
        config_acc = tile_config_accumulator_by_key.pop(key, None)
        config_acc_path = tile_config_accumulator_paths.pop(key, None)
        if config_acc is None:
            # Every tile in this size/stride configuration was empty or rejected before staging.
            _mark_tile_config_gated(str(model_name), str(view_name), str(config_id))
            return

        parent_key = (str(model_name), str(view_name))
        tile_accumulator_mm = _get_tile_accumulator(str(model_name), str(view_name))
        parent_mask_support_mm = None
        parent_bridge_support_mm = None
        tile_parent_mask_accumulator_mm = None
        tile_parent_bridge_accumulator_mm = None
        if bool(nrrd_layers_needed):
            parent_mask_support_mm = parent_mask_support_by_model.get(str(model_name), {}).get(str(view_name))
            parent_bridge_support_mm = parent_bridge_support_by_model.get(str(model_name), {}).get(str(view_name))
            tile_parent_mask_accumulator_mm = _get_tile_category_accumulator(str(model_name), str(view_name), 'parent_mask')
            if parent_bridge_support_mm is not None:
                tile_parent_bridge_accumulator_mm = _get_tile_category_accumulator(str(model_name), str(view_name), 'parent_bridge')

        config_result = TilePostprocessResult(
            model_name=str(model_name),
            view_name=str(view_name),
            config_id=str(config_id),
            tile_id=str(config_id),
            tile_mask_mm=config_acc,
            tile_mask_path=config_acc_path,
        )
        fut = tile_postprocess_executor.submit(
            gate_tile_volume_into_consolidated_parent,
            config_result,
            parent_support_mm=support_by_view[str(view_name)],
            tile_accumulator_mm=tile_accumulator_mm,
            tile_accumulator_lock=view_volume_locks[(str(model_name), str(view_name))],
            keep_temp=bool(keep_temp_artifacts),
            slice_workers=int(tile_slice_postprocess_workers),
            parent_mask_support_mm=parent_mask_support_mm,
            parent_bridge_support_mm=parent_bridge_support_mm,
            tile_parent_mask_accumulator_mm=tile_parent_mask_accumulator_mm,
            tile_parent_bridge_accumulator_mm=tile_parent_bridge_accumulator_mm,
            temp_dir=temp_dir,
        )
        tile_config_gate_futures[fut] = key

    def _mark_tile_staged(model_name: str, view_name: str, config_id: str, tile_id: str) -> None:
        key = (str(model_name), str(view_name), str(config_id))
        if key not in tile_expected_by_parent_config:
            return
        completed = tile_completed_by_parent_config.setdefault(key, set())
        completed.add(str(tile_id))
        _maybe_submit_tile_config_gate(str(model_name), str(view_name), str(config_id))

    def _submit_tile_finalize(result: TilePostprocessResult) -> None:
        parent_key = (str(result.model_name), str(result.view_name))
        support_by_view = native_view_support_by_model.get(result.model_name, {})
        if result.view_name not in support_by_view:
            waiting = postprocessed_tiles_waiting_by_parent.setdefault(parent_key, {})
            parent_view = view_infos_by_name[str(result.view_name)]
            if waiting_tile_spill_enabled():
                waiting[str(result.tile_id)] = spill_waiting_tile_result_to_raw_store(
                    result,
                    temp_dir,
                    workers=int(tile_slice_postprocess_workers),
                    keep_original=bool(keep_temp_artifacts),
                    expected_parent_shape=(int(parent_view.num_slices), int(parent_view.src_h), int(parent_view.src_w)),
                )
            else:
                # Keep the already-parent-sized postprocessed tile volume resident instead of
                # round-tripping through a raw ctile store while waiting for parent support.
                waiting[str(result.tile_id)] = result
            return

        waiting = postprocessed_tiles_waiting_by_parent.get(parent_key)
        if waiting is not None:
            waiting.pop(str(result.tile_id), None)
            if not waiting:
                postprocessed_tiles_waiting_by_parent.pop(parent_key, None)

        config_accumulator_mm = _get_tile_config_accumulator(result.model_name, result.view_name, result.config_id)
        config_lock = tile_config_accumulator_locks.setdefault(
            (str(result.model_name), str(result.view_name), str(result.config_id)),
            threading.Lock(),
        )
        fut = tile_postprocess_executor.submit(
            stage_tile_result_into_config_canvas,
            result,
            tile_set_accumulator_mm=config_accumulator_mm,
            tile_set_accumulator_lock=config_lock,
            keep_temp=bool(keep_temp_artifacts),
            slice_workers=int(tile_slice_postprocess_workers),
        )
        tile_finalize_futures[fut] = (str(result.model_name), str(result.view_name), str(result.config_id), str(result.tile_id))

    def _flush_ready_postprocessed_tiles() -> None:
        ready_results: List[TilePostprocessResult] = []
        for parent_key, waiting in list(postprocessed_tiles_waiting_by_parent.items()):
            model_name, view_name = parent_key
            if view_name not in native_view_support_by_model.get(model_name, {}):
                continue
            for wait_result in waiting.values():
                if isinstance(wait_result, DeferredTilePostprocessResult):
                    ready_results.append(load_waiting_tile_result_from_raw_store(wait_result))
                elif isinstance(wait_result, TilePostprocessResult):
                    ready_results.append(wait_result)
                else:
                    raise TypeError(f'Unsupported waiting tile result type: {type(wait_result)!r}')
            del postprocessed_tiles_waiting_by_parent[parent_key]

        for result in ready_results:
            _submit_tile_finalize(result)



    # v12: render futures were replaced by _drain_completed_prediction_volume_futures().

    def _submit_prediction_accumulation_join(handle: PredictionAccumulationHandle, context: Dict[str, object]) -> None:
        fut = prediction_join_executor.submit(handle.wait)
        prediction_accumulation_futures[fut] = dict(context)

    def _drain_completed_prediction_accumulation_futures() -> None:
        for fut in list(prediction_accumulation_futures.keys()):
            if not fut.done():
                continue
            context = prediction_accumulation_futures.pop(fut)
            pred_stats = fut.result()
            kind = str(context.get('kind', ''))

            if kind == 'fullframe':
                model_name = str(context['model_name'])
                view = context['view']
                assert isinstance(view, ViewInfo)
                yolo_obj = context.get('yolo')
                if offload_between_jobs_enabled() and yolo_obj is not None:
                    offload_yolo_from_gpu(yolo_obj)
                view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))
                remaining_key = (model_name, view.name)
                fullframe_remaining[remaining_key] = int(fullframe_remaining.get(remaining_key, 0)) - 1
                if int(fullframe_remaining.get(remaining_key, 0)) == 0:
                    _submit_view_prepare(model_name, view)
                continue

            if kind == 'tile':
                model_name = str(context['model_name'])
                view = context['view']
                tile_job = context['tile_job']
                assert isinstance(view, ViewInfo)
                assert isinstance(tile_job, DenseTileJob)
                tile_mask_mm = context['tile_mask_mm']
                tile_conf_mm = context.get('tile_conf_mm')
                tile_mask_path = Path(context['tile_mask_path'])
                tile_conf_path_obj = context.get('tile_conf_path')
                tile_conf_path = Path(tile_conf_path_obj) if tile_conf_path_obj is not None else None
                ready_key = (str(model_name), str(view.name), str(tile_job.tile_id))
                yolo_obj = context.get('yolo')
                if offload_between_jobs_enabled() and yolo_obj is not None:
                    offload_yolo_from_gpu(yolo_obj)
                tile_inference_done.add(ready_key)
                view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))

                if int(pred_stats.get('frames_with_predictions', 0)) <= 0:
                    close_memmap_array(tile_mask_mm)
                    close_memmap_array(tile_conf_mm)
                    if not keep_temp_artifacts:
                        try:
                            tile_mask_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        if tile_conf_path is not None:
                            try:
                                tile_conf_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                    _mark_tile_staged(str(model_name), str(view.name), str(tile_job.config_id), str(tile_job.tile_id))
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
                    precleaned_slice_cleanup=bool(single_angle_streaming_cleanup_active),
                )
                tile_fut = tile_postprocess_executor.submit(
                    postprocess_tile_volume_after_inference,
                    task,
                    view=view,
                    min_conf=float(args.min_conf),
                    min_radius=float(args.min_radius),
                    keep_temp=bool(keep_temp_artifacts),
                    slice_workers=int(tile_slice_postprocess_workers),
                )
                tile_cleanup_futures[tile_fut] = (str(model_name), str(view.name), str(tile_job.config_id), str(tile_job.tile_id))
                continue

            raise RuntimeError(f'Unknown prediction accumulation kind: {kind!r}')

    def _drain_completed_background_futures() -> None:
        for fut in list(view_processing_futures.keys()):
            if not fut.done():
                continue
            result = fut.result()
            del view_processing_futures[fut]
            native_view_support_by_model[result.model_name][result.view_name] = result.native_support_mm
            if result.parent_mask_support_mm is not None:
                parent_mask_support_by_model[result.model_name][result.view_name] = result.parent_mask_support_mm
            if result.parent_bridge_support_mm is not None:
                parent_bridge_support_by_model[result.model_name][result.view_name] = result.parent_bridge_support_mm
            interpolation_stats.extend(result.interpolation_stats)
            nrrd_layer_refs.extend(result.nrrd_layers)

            view_info = view_infos_by_name[result.view_name]
            if result.final_view_volume_mm is not None:
                if view_info.family == 'radial':
                    radial_native_output_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
                elif is_tilted_view(view_info):
                    tilted_native_output_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
                else:
                    view_volumes_by_model[result.model_name][result.view_name] = result.final_view_volume_mm
            for config_key in list(tile_expected_by_parent_config.keys()):
                cfg_model, cfg_view, cfg_id = config_key
                if cfg_model == result.model_name and cfg_view == result.view_name:
                    _maybe_submit_tile_config_gate(cfg_model, cfg_view, cfg_id)

        for fut in list(tile_cleanup_futures.keys()):
            if not fut.done():
                continue
            ready_key = tile_cleanup_futures.pop(fut)
            result = fut.result()
            if result is None:
                _mark_tile_staged(str(ready_key[0]), str(ready_key[1]), str(ready_key[2]), str(ready_key[3]))
                continue
            _submit_tile_finalize(result)

        _flush_ready_postprocessed_tiles()

        for fut in list(tile_finalize_futures.keys()):
            if not fut.done():
                continue
            model_name, view_name, config_id, tile_id = tile_finalize_futures.pop(fut)
            fut.result()
            _mark_tile_staged(str(model_name), str(view_name), str(config_id), str(tile_id))

        for fut in list(tile_config_gate_futures.keys()):
            if not fut.done():
                continue
            model_name, view_name, config_id = tile_config_gate_futures.pop(fut)
            fut.result()
            _mark_tile_config_gated(str(model_name), str(view_name), str(config_id))

        for fut in list(tile_consolidation_futures.keys()):
            if not fut.done():
                continue
            parent_key = tile_consolidation_futures.pop(fut)
            result = fut.result()
            interpolation_stats.extend(result.interpolation_stats)
            nrrd_layer_refs.extend(result.nrrd_layers)

            # Once the consolidated tile volume has been unioned into its parent destination and
            # any NRRD layers have been materialized, category accumulators are no longer needed.
            # The main consolidated tile accumulator is retained until troubleshooting overlays are
            # scheduled when --troubleshooting is active; final cleanup archives/deletes any survivor.
            for label, store in (
                ('consolidated gated tiles', tile_accumulator_by_parent),
                ('tile components accepted by parent mask', tile_parent_mask_accumulator_by_parent),
                ('tile components accepted by parent bridge', tile_parent_bridge_accumulator_by_parent),
            ):
                if bool(troubleshooting_outputs_enabled) and label == 'consolidated gated tiles':
                    continue
                acc = store.pop(parent_key, None)
                if acc is not None:
                    archive_or_delete_binary_volume_storage(
                        acc,
                        keep_temp=bool(keep_temp_artifacts),
                        workers=int(tile_slice_postprocess_workers),
                        desc=f'{label} {parent_key[0]}/{parent_key[1]}',
                    )
                    if label == 'consolidated gated tiles':
                        tile_accumulator_paths.pop(parent_key, None)

        output_manager.reap_completed()

    last_scheduler_wait_log = 0.0

    def _log_scheduler_wait_state(force: bool = False) -> None:
        nonlocal last_scheduler_wait_log
        now = time.time()
        interval = max(5.0, _env_float('YOLO_TTA_SCHEDULER_STATUS_INTERVAL_SEC', 30.0))
        if not bool(force) and (now - float(last_scheduler_wait_log)) < float(interval):
            return
        last_scheduler_wait_log = float(now)
        waiting_tiles = sum(len(v) for v in postprocessed_tiles_waiting_by_parent.values())
        print(
            'Scheduler wait: no inference-ready in-memory volume; '
            f'pending_volume_builds={len(pending_prediction_volume_futures)}, '
            f'queued_build_jobs={len(pending_prediction_build_jobs)}, '
            f'prediction_accumulation={len(prediction_accumulation_futures)}, '
            f'parent_postprocess={len(view_processing_futures)}, '
            f'tile_cleanup={len(tile_cleanup_futures)}, '
            f'tile_finalize={len(tile_finalize_futures)}, '
            f'tile_config_gate={len(tile_config_gate_futures)}, '
            f'tile_consolidation={len(tile_consolidation_futures)}, '
            f'waiting_tiles_for_parent={waiting_tiles}, '
            f'ready_fullframe={len(ready_fullframe)}, ready_tiles={len(ready_tile_infer)}'
        )

    try:
        _pump_prediction_volume_build_queue()
        while True:
            _drain_completed_prediction_volume_futures()
            _drain_completed_prediction_accumulation_futures()
            _drain_completed_background_futures()
            _pump_prediction_volume_build_queue()

            if ready_fullframe:
                view, job, prediction_ref = ready_fullframe.popleft()
                print(f"Inferencing full-frame in-memory volume: {view.name}/{job.aug_id}")
                try:
                    for model_name, yolo in yolo_models:
                        _ensure_baseline_workspaces(str(model_name), view)
                        if bool(async_prediction_accumulation_active):
                            handle = predict_in_memory_volume_and_submit_accumulation(
                                model=yolo,
                                prediction_volume=prediction_ref,
                                num_frames=view.num_slices,
                                out_size=args.imgsz,
                                cfg=pred_cfg,
                                view_union_mm=baseline_union_by_model_view[(model_name, view.name)],
                                view_confmap_mm=baseline_confmap_by_model_view[(model_name, view.name)],
                                M_out_to_native=job.aff.M_out_to_src,
                                native_h=view.src_h,
                                native_w=view.src_w,
                                postprocess_executor=prediction_result_executor,
                                streaming_cleanup_enabled=bool(single_angle_streaming_cleanup_active),
                                streaming_cleanup_min_conf=float(args.min_conf),
                                streaming_cleanup_min_radius=_view_native_slice_min_radius(view, float(args.min_radius)),
                                slice_locks=baseline_slice_locks_by_model_view.get((str(model_name), str(view.name))),
                            )
                            _submit_prediction_accumulation_join(handle, {
                                'kind': 'fullframe',
                                'model_name': str(model_name),
                                'view': view,
                                'job': job,
                                'yolo': yolo,
                            })
                        else:
                            pred_stats = predict_in_memory_volume_and_accumulate(
                                model=yolo,
                                prediction_volume=prediction_ref,
                                num_frames=view.num_slices,
                                out_size=args.imgsz,
                                cfg=pred_cfg,
                                view_union_mm=baseline_union_by_model_view[(model_name, view.name)],
                                view_confmap_mm=baseline_confmap_by_model_view[(model_name, view.name)],
                                M_out_to_native=job.aff.M_out_to_src,
                                native_h=view.src_h,
                                native_w=view.src_w,
                                postprocess_workers=predict_postprocess_workers,
                                streaming_cleanup_enabled=bool(single_angle_streaming_cleanup_active),
                                streaming_cleanup_min_conf=float(args.min_conf),
                                streaming_cleanup_min_radius=_view_native_slice_min_radius(view, float(args.min_radius)),
                                slice_locks=baseline_slice_locks_by_model_view.get((str(model_name), str(view.name))),
                            )
                            if offload_between_jobs_enabled():
                                offload_yolo_from_gpu(yolo)
                            view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))

                            remaining_key = (model_name, view.name)
                            fullframe_remaining[remaining_key] = int(fullframe_remaining.get(remaining_key, 0)) - 1
                            if int(fullframe_remaining.get(remaining_key, 0)) == 0:
                                _submit_view_prepare(model_name, view)
                finally:
                    close_prediction_volume_ref(prediction_ref, keep_temp=bool(keep_temp_artifacts))
                    _pump_prediction_volume_build_queue()
                continue

            if ready_tile_infer:
                model_name, view, tile_job, prediction_ref = ready_tile_infer.popleft()
                ready_key = (str(model_name), str(view.name), str(tile_job.tile_id))
                if ready_key in tile_inference_done:
                    close_prediction_volume_ref(prediction_ref, keep_temp=bool(keep_temp_artifacts))
                    continue

                print(f"Inferencing tile in-memory volume: {model_name}/{view.name}/{tile_job.tile_id}")
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
                if float(args.min_conf) > 0.0:
                    tile_conf_mm = allocate_workspace_array(
                        shape=tile_shape,
                        dtype=np.uint8,
                        path=tile_conf_path,
                        desc=f'{model_name}/{view.name}/{tile_job.tile_id} raw tile confidence workspace',
                        prefer_memory=True,
                    )
                    tile_conf_store_path: Optional[Path] = tile_conf_path
                else:
                    tile_conf_mm = None
                    tile_conf_store_path = None

                yolo = yolo_by_model_name[str(model_name)]
                try:
                    if bool(async_prediction_accumulation_active):
                        handle = predict_in_memory_volume_and_submit_accumulation(
                            model=yolo,
                            prediction_volume=prediction_ref,
                            num_frames=view.num_slices,
                            out_size=int(args.imgsz),
                            cfg=pred_cfg,
                            view_union_mm=tile_mask_mm,
                            view_confmap_mm=tile_conf_mm,
                            M_out_to_native=tile_job.M_out_to_src,
                            native_h=view.src_h,
                            native_w=view.src_w,
                            postprocess_executor=prediction_result_executor,
                            streaming_cleanup_enabled=bool(single_angle_streaming_cleanup_active),
                            streaming_cleanup_min_conf=float(args.min_conf),
                            streaming_cleanup_min_radius=_view_native_slice_min_radius(view, float(args.min_radius)),
                        )
                        _submit_prediction_accumulation_join(handle, {
                            'kind': 'tile',
                            'model_name': str(model_name),
                            'view': view,
                            'tile_job': tile_job,
                            'tile_mask_mm': tile_mask_mm,
                            'tile_conf_mm': tile_conf_mm,
                            'tile_mask_path': tile_mask_path,
                            'tile_conf_path': tile_conf_store_path,
                            'yolo': yolo,
                        })
                    else:
                        pred_stats = predict_in_memory_volume_and_accumulate(
                            model=yolo,
                            prediction_volume=prediction_ref,
                            num_frames=view.num_slices,
                            out_size=int(args.imgsz),
                            cfg=pred_cfg,
                            view_union_mm=tile_mask_mm,
                            view_confmap_mm=tile_conf_mm,
                            M_out_to_native=tile_job.M_out_to_src,
                            native_h=view.src_h,
                            native_w=view.src_w,
                            postprocess_workers=predict_postprocess_workers,
                            streaming_cleanup_enabled=bool(single_angle_streaming_cleanup_active),
                            streaming_cleanup_min_conf=float(args.min_conf),
                            streaming_cleanup_min_radius=_view_native_slice_min_radius(view, float(args.min_radius)),
                        )
                        if offload_between_jobs_enabled():
                            offload_yolo_from_gpu(yolo)
                        tile_inference_done.add(ready_key)
                        view_prediction_stats[str(view.summary_family)] = int(view_prediction_stats.get(str(view.summary_family), 0)) + int(pred_stats.get('prediction_count', 0))

                        if int(pred_stats.get('frames_with_predictions', 0)) <= 0:
                            close_memmap_array(tile_mask_mm)
                            close_memmap_array(tile_conf_mm)
                            if not keep_temp_artifacts:
                                try:
                                    tile_mask_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                                if tile_conf_path is not None:
                                    try:
                                        tile_conf_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                            _mark_tile_staged(str(model_name), str(view.name), str(tile_job.config_id), str(tile_job.tile_id))
                            continue

                        task = TilePostprocessTask(
                            model_name=str(model_name),
                            view_name=str(view.name),
                            config_id=str(tile_job.config_id),
                            tile_id=str(tile_job.tile_id),
                            tile_mask_mm=tile_mask_mm,
                            tile_confmap_mm=tile_conf_mm,
                            tile_mask_path=tile_mask_path,
                            tile_confmap_path=tile_conf_store_path,
                            precleaned_slice_cleanup=bool(single_angle_streaming_cleanup_active),
                        )
                        fut = tile_postprocess_executor.submit(
                            postprocess_tile_volume_after_inference,
                            task,
                            view=view,
                            min_conf=float(args.min_conf),
                            min_radius=float(args.min_radius),
                            keep_temp=bool(keep_temp_artifacts),
                            slice_workers=int(tile_slice_postprocess_workers),
                        )
                        tile_cleanup_futures[fut] = (str(model_name), str(view.name), str(tile_job.config_id), str(tile_job.tile_id))
                finally:
                    close_prediction_volume_ref(prediction_ref, keep_temp=bool(keep_temp_artifacts))
                    _pump_prediction_volume_build_queue()
                continue

            waitables: List[Future] = list(pending_prediction_volume_futures)
            waitables.extend(list(prediction_accumulation_futures.keys()))
            waitables.extend(list(view_processing_futures.keys()))
            waitables.extend(list(tile_cleanup_futures.keys()))
            waitables.extend(list(tile_finalize_futures.keys()))
            waitables.extend(list(tile_config_gate_futures.keys()))
            waitables.extend(list(tile_consolidation_futures.keys()))
            if not waitables:
                _flush_ready_postprocessed_tiles()
                _pump_prediction_volume_build_queue()
                if (
                    not pending_prediction_build_jobs and
                    not pending_prediction_volume_futures and
                    not prediction_accumulation_futures and
                    not ready_fullframe and
                    not ready_tile_infer and
                    not tile_finalize_futures and
                    not tile_config_gate_futures and
                    not tile_cleanup_futures and
                    not tile_consolidation_futures and
                    not view_processing_futures
                ):
                    break
                continue
            _log_scheduler_wait_state()
            wait(waitables, return_when=FIRST_COMPLETED)

    finally:
        prediction_volume_executor.shutdown(wait=True)
        prediction_join_executor.shutdown(wait=True)
        prediction_result_executor.shutdown(wait=True)
        parent_postprocess_executor.shutdown(wait=True)
        tile_postprocess_executor.shutdown(wait=True)
        set_interpolation_process_executor(None, 0)
        if interpolation_process_executor is not None:
            try:
                interpolation_process_executor.shutdown(wait=True, cancel_futures=False)
            except TypeError:
                interpolation_process_executor.shutdown(wait=True)

    _drain_completed_prediction_volume_futures()
    _drain_completed_prediction_accumulation_futures()
    _drain_completed_background_futures()

    if preprocess_streaming_active:
        print('Ensuring streaming preprocessing producers have completed before final output/backprojection stages.')
        wait_for_volume_ready(volume_rgb)

    for cache_name, cache_mm in list(view_frame_caches.items()):
        close_memmap_array(cache_mm)
        cache_path = view_frame_cache_paths.get(cache_name)
        if not keep_temp_artifacts and cache_path is not None:
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass
    view_frame_caches.clear()
    view_frame_cache_paths.clear()

    if not bool(keep_temp_artifacts):
        swept_mkvs = purge_remaining_temporary_mkvs(temp_dir, keep_temp=False)
        if int(swept_mkvs) > 0:
            print(f'Final legacy temporary MKV sweep removed {int(swept_mkvs)} leftover file(s).')
        spec_notes.append(
            'No prediction MKVs are produced by the v12 in-memory path. A best-effort final sweep still removes '
            f'{int(swept_mkvs)} legacy/interrupted scratch MKV file(s) when YOLO_TTA_KEEP_TEMP is disabled.'
        )
    else:
        spec_notes.append('YOLO_TTA_KEEP_TEMP retained scratch artifacts; the v12.2.0 in-memory inference path itself does not create prediction MKVs.')

    final_backprojection_jobs: List[ViewBackprojectionQueueJob] = []
    for view in views:
        if (view.family != 'radial' and not is_tilted_view(view)):
            continue
        for model_name, _ in yolo_models:
            if view.name in view_volumes_by_model[model_name]:
                continue
            if view.family == 'radial':
                native_source = radial_native_output_by_model[model_name].get(view.name)
            else:
                native_source = tilted_native_output_by_model[model_name].get(view.name)
            if native_source is None:
                native_source = native_view_support_by_model[model_name].get(view.name)
            if native_source is None:
                continue
            final_backprojection_jobs.append(ViewBackprojectionQueueJob(
                model_name=str(model_name),
                view=view,
                native_source=native_source,
                out_path=temp_dir / 'view_volumes' / str(model_name) / f'{view.name}.u8.dat',
                desc=f'Backprojecting final {model_name}/{view.name}',
                min_radius=float(args.min_radius),
                workers=1,
            ))

    if final_backprojection_jobs:
        # v12.2.11: backprojection is CPU-only, so do not halve the CPU budget for a
        # nonexistent GPU slot.  Run one set at a time with the full slice-worker budget.
        per_backproject_workers = max(1, int(slice_postprocess_workers))
        final_backprojection_jobs = [
            ViewBackprojectionQueueJob(
                model_name=job.model_name,
                view=job.view,
                native_source=job.native_source,
                out_path=job.out_path,
                desc=job.desc,
                min_radius=job.min_radius,
                workers=int(per_backproject_workers),
            )
            for job in final_backprojection_jobs
        ]
        gpu_enabled_for_backproject = False
        print(
            f'Final radial/tilted backprojection queue: tasks={len(final_backprojection_jobs)}, '
            f'max_active=1 CPU-only, per-set CPU workers={int(per_backproject_workers)}, '
            f'gpu_enabled={gpu_enabled_for_backproject}'
        )
        backproject_queue = HybridBackprojectionQueue(
            cpu_workers=int(per_backproject_workers),
            gpu_enabled=bool(gpu_enabled_for_backproject),
        )
        for model_name_done, view_name_done, projected_volume in backproject_queue.run(final_backprojection_jobs):
            view_volumes_by_model[model_name_done][view_name_done] = projected_volume

    output_manager.reap_completed()

    print('\n=== Building final single-model view union after the global view union ===')
    final_union_mm = assemble_final_union_after_view_union(
        view_volumes_by_model=view_volumes_by_model,
        T=T,
        H=H,
        W=W,
        disable_multiplanar=None,
        out_path=temp_dir / 'final_union_volume.u8.dat',
        temp_dir=temp_dir,
        enable_3d_void_fill=bool(args.enable_3d_void_fill),
        keep_temp=bool(keep_temp_artifacts),
        prefer_memory=True,
        workers=slice_postprocess_workers,
    )

    if bool(nrrd_layers_needed):
        pre_smoothing_ref = materialize_nrrd_global_layer(
            final_union_mm,
            model_name=str(model_name),
            source='global',
            mask_kind='union',
            pass_index=0,
            stage='pre_smoothing',
            description='Global union after all active view/tile layers and optional 3D void fill, before Gaussian smoothing and keep_objects.',
            temp_dir=temp_dir,
            workers=int(slice_postprocess_workers),
        )
        if pre_smoothing_ref is not None:
            nrrd_layer_refs.append(pre_smoothing_ref)

    gaussian_smoothing_stats: Optional[Dict[str, object]] = None
    if bool(gaussian_smoothing_enabled):
        print('\n=== Applying Gaussian smoothing ===')
        gaussian_smoothing_stats = apply_gaussian_smoothing_inplace(
            final_union_mm,
            sigma=float(gaussian_smoothing_sigma),
            passes=int(gaussian_smoothing_passes),
            temp_dir=temp_dir,
            keep_temp=bool(keep_temp_artifacts),
            prefer_memory=True,
            workers=slice_postprocess_workers,
            nrrd_layers=nrrd_layer_refs if bool(nrrd_layers_needed) else None,
            nrrd_model_name=str(model_name),
        )

    keep_objects_stats: Optional[Dict[str, int]] = None
    if int(args.keep_objects) > 0:
        print(f'\n=== Keeping largest {int(args.keep_objects)} final object(s) ===')
        keep_objects_stats = apply_keep_largest_objects_inplace(
            final_union_mm,
            int(args.keep_objects),
            temp_dir=temp_dir,
            keep_temp=bool(keep_temp_artifacts),
            prefer_memory=True,
            workers=slice_postprocess_workers,
        )

    skeleton_processing_mm: Optional[np.ndarray] = None
    skeleton_output_mm: Optional[np.ndarray] = None
    skeleton_path: Optional[Path] = None
    if bool(args.save_skeleton):
        print('\n=== Computing optional skeleton output ===')
        skeleton_processing_mm = compute_skeleton_volume_to_workspace(
            final_union_mm,
            temp_dir / 'skeleton' / 'final_skeleton_processing_geometry.u8.dat',
            temp_dir=temp_dir,
            workers=int(slice_postprocess_workers),
            keep_temp=bool(keep_temp_artifacts),
            prefer_memory=True,
        )
        if bool(nrrd_layers_needed):
            skeleton_ref = materialize_nrrd_global_layer(
                skeleton_processing_mm,
                model_name=str(model_name),
                source='global',
                mask_kind='skeleton',
                pass_index=0,
                stage='skeleton_after_postprocessing',
                description='Optional skeleton layer computed after final postprocessing and before restoration to native input geometry. Interpolation did not use skeletonization.',
                temp_dir=temp_dir,
                workers=int(slice_postprocess_workers),
            )
            if skeleton_ref is not None:
                nrrd_layer_refs.append(skeleton_ref)

    final_output_mask_mm = final_union_mm
    output_volume_rgb = input_volume_rgb
    output_T, output_H, output_W = int(input_T), int(input_H), int(input_W)
    if (int(T), int(H), int(W)) != (output_T, output_H, output_W):
        print('\n=== Restoring final mask to original input geometry for default outputs ===')
        final_output_mask_mm = restore_mask_volume_to_original_shape(
            final_union_mm,
            (output_T, output_H, output_W),
            temp_dir / 'final_union_original_geometry.u8.dat',
            workers=int(slice_postprocess_workers),
            prefer_memory=True,
        )
    final_output_volume_for_low_quality = output_volume_rgb
    final_paths: Dict[str, Path] = {}

    if bool(args.save_skeleton) and skeleton_processing_mm is not None:
        if (int(T), int(H), int(W)) != (output_T, output_H, output_W):
            skeleton_output_mm = restore_mask_volume_to_original_shape(
                skeleton_processing_mm,
                (output_T, output_H, output_W),
                temp_dir / 'skeleton' / 'final_skeleton_original_geometry.u8.dat',
                workers=int(slice_postprocess_workers),
                prefer_memory=True,
            )
        else:
            skeleton_output_mm = skeleton_processing_mm
        if not bool(args.save_nrrd):
            print('\n=== Writing skeleton-only NRRD ===')
            skeleton_path = out_dir / f'{input_path.stem}_Skeleton.nrrd'
            write_nrrd(skeleton_output_mm, skeleton_path)
            final_paths['skeleton_nrrd'] = skeleton_path


    if bool(troubleshooting_outputs_enabled):
        print('\n=== Scheduling v12.2.0 troubleshooting overlays ===')
        consolidated_tile_masks_for_model = {
            str(view_name): mask
            for (tile_model, view_name), mask in tile_accumulator_by_parent.items()
            if str(tile_model) == str(model_name)
        }
        troubleshooting_paths, troubleshooting_futures = collect_troubleshooting_overlay_futures(
            output_manager.executor,
            volume_rgb=volume_rgb,
            model_name=str(model_name),
            views=inference_views,
            native_view_support=native_view_support_by_model.get(str(model_name), {}),
            consolidated_tile_masks=consolidated_tile_masks_for_model,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            show_progress=False,
        )
        final_paths.update(troubleshooting_paths)
        output_manager.submit(BackgroundOutputSubmission(
            label='troubleshooting overlays',
            result_paths=troubleshooting_paths,
            futures=troubleshooting_futures,
            resources=[],
        ))

    print('\n=== Scheduling final outputs in background ===')
    final_output_paths, final_futures = collect_pipeline_output_futures(
        output_manager.executor,
        volume_rgb=output_volume_rgb,
        mask_u8=final_output_mask_mm,
        skeleton_u8=skeleton_output_mm if bool(args.save_skeleton) else None,
        out_dir=out_dir,
        stem=input_path.stem,
        fps=fps,
        save_binary_pattern_value=args.save_binary,
        save_labels_pattern_value=args.save_labels,
        save_nrrd_flag=bool(args.save_nrrd),
        tag=None,
        frame_workers=output_frame_workers,
        show_progress=False,
        nrrd_layer_refs=nrrd_layer_refs if bool(args.save_nrrd) else None,
        nrrd_temp_dir=temp_dir,
        nrrd_workers=output_frame_workers,
    )
    if bool(save_sagittal) or bool(save_coronal):
        extra_paths, extra_futures = collect_multiplanar_output_futures(
            output_manager.executor,
            volume_rgb=output_volume_rgb,
            mask_u8=final_output_mask_mm,
            skeleton_u8=skeleton_output_mm if bool(args.save_skeleton) else None,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            save_binary_pattern_value=args.save_binary,
            save_labels_pattern_value=args.save_labels,
            save_sagittal_flag=False,
            save_coronal_flag=False,
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

    if bool(low_quality_requested):
        print('\n=== Scheduling low-quality isotropic outputs in background ===')
        low_quality_paths = planned_low_quality_output_paths(
            out_dir=out_dir,
            stem=input_path.stem,
            downbin_specs=low_quality_downbin_specs,
            nrrd_layer_refs=nrrd_layer_refs,
        )
        low_quality_future = output_manager.executor.submit(
            save_low_quality_outputs,
            volume_gray=final_output_volume_for_low_quality,
            mask_u8=final_output_mask_mm,
            skeleton_u8=skeleton_output_mm if bool(args.save_skeleton) else None,
            out_dir=out_dir,
            stem=input_path.stem,
            fps=fps,
            downbin_specs=low_quality_downbin_specs,
            temp_dir=temp_dir,
            nrrd_layer_refs=nrrd_layer_refs,
            workers=output_workers,
            show_progress=False,
        )
        final_paths.update(low_quality_paths)
        output_manager.submit(BackgroundOutputSubmission(
            label='low-quality outputs',
            result_paths=low_quality_paths,
            futures=[low_quality_future],
            resources=[],
        ))

    voxel_volume = None
    if bool(args.voxel_volume):
        voxel_counts = np.zeros((int(final_output_mask_mm.shape[0]),), dtype=np.int64)

        def _count_voxels(z: int) -> None:
            voxel_counts[int(z)] = np.int64(np.count_nonzero(np.asarray(final_output_mask_mm[int(z)])))

        parallel_for_indices(
            int(final_output_mask_mm.shape[0]),
            _count_voxels,
            max_workers=choose_slice_parallel_workers(int(slice_postprocess_workers), int(final_output_mask_mm.shape[0])),
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

    if bool(nrrd_layers_needed):
        spec_notes.append(
            f'Decomposed NRRD component layers prepared: {int(len(nrrd_layer_refs))}; '
            'the writer appends one final_output layer, reuses materialization-time SegmentN extents, and writes .nrrd.manifest.json sidecars for full-size and low-quality decomposed NRRDs.'
        )

    summary_path = write_summary_file(
        out_dir / f'{input_path.stem}_Summary.txt',
        command=shlex.join([str(x) for x in sys.argv]),
        input_path=input_path,
        out_dir=out_dir,
        scratch_dir=temp_dir,
        source_shape_x_y_t=(input_W, input_H, input_T),
        volume_shape=(T, H, W),
        fps=fps,
        model_paths=model_paths,
        view_names=[
            (
                f'{v.name} ({int(v.num_slices)} frames; centers {int(v.tilt_frame_start)}..{int(v.tilt_frame_stop)})'
                if is_tilted_view(v)
                else f'{v.name} ({int(v.num_slices)} frames)'
            )
            for v in views
        ],
        view_prediction_stats=view_prediction_stats,
        interpolation_stats=interpolation_stats,
        view_prediction_labels=view_prediction_labels,
        enable_3d_void_fill=bool(args.enable_3d_void_fill),
        gaussian_smoothing_stats=gaussian_smoothing_stats,
        keep_objects_stats=keep_objects_stats,
        voxel_volume=voxel_volume,
        final_paths=final_paths,
        augmentation_workers=augmentation_workers,
        slice_postprocess_workers=slice_postprocess_workers,
        interpolation_workers=interpolation_workers,
        output_workers=output_workers,
        spec_notes=spec_notes,
    )

    if skeleton_output_mm is not None and skeleton_output_mm is not skeleton_processing_mm:
        close_memmap_array(skeleton_output_mm)
    if skeleton_processing_mm is not None:
        close_memmap_array(skeleton_processing_mm)
    if final_output_mask_mm is not final_union_mm:
        close_memmap_array(final_output_mask_mm)
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
    for model_support in parent_mask_support_by_model.values():
        for mm in model_support.values():
            close_raw_store_or_memmap_volume(mm, keep_temp=bool(keep_temp_artifacts))
        model_support.clear()
    for model_support in parent_bridge_support_by_model.values():
        for mm in model_support.values():
            close_raw_store_or_memmap_volume(mm, keep_temp=bool(keep_temp_artifacts))
        model_support.clear()
    for mm in tile_config_accumulator_by_key.values():
        archive_or_delete_binary_volume_storage(
            mm,
            keep_temp=bool(keep_temp_artifacts),
            workers=int(tile_slice_postprocess_workers),
            desc='remaining tile config accumulator',
        )
    tile_config_accumulator_by_key.clear()
    for mm in tile_accumulator_by_parent.values():
        archive_or_delete_binary_volume_storage(
            mm,
            keep_temp=bool(keep_temp_artifacts),
            workers=int(tile_slice_postprocess_workers),
            desc='remaining consolidated tile accumulator',
        )
    tile_accumulator_by_parent.clear()
    for mm in tile_parent_mask_accumulator_by_parent.values():
        archive_or_delete_binary_volume_storage(
            mm,
            keep_temp=bool(keep_temp_artifacts),
            workers=int(tile_slice_postprocess_workers),
            desc='remaining parent-mask tile category accumulator',
        )
    tile_parent_mask_accumulator_by_parent.clear()
    for mm in tile_parent_bridge_accumulator_by_parent.values():
        archive_or_delete_binary_volume_storage(
            mm,
            keep_temp=bool(keep_temp_artifacts),
            workers=int(tile_slice_postprocess_workers),
            desc='remaining parent-bridge tile category accumulator',
        )
    tile_parent_bridge_accumulator_by_parent.clear()
    for mm in baseline_union_by_model_view.values():
        close_memmap_array(mm)
    for mm in baseline_confmap_by_model_view.values():
        close_memmap_array(mm)
    for _, yolo in yolo_models:
        unload_yolo_model(yolo)
    if volume_rgb is not input_volume_rgb:
        close_memmap_array(volume_rgb)
    close_memmap_array(input_volume_rgb)
    trim_cuda_memory()
    gc.collect()

    if not keep_temp_artifacts:
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
