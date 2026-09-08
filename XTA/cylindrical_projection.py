"""Bounded pull projection of radius-swept cylindrical patch trajectories.

This geometry is separate from the legacy Azimuthal diameter-plane projector.
Every source voxel chooses its nearest *global* shell, then reads every periodic
occurrence in this intrinsic patch. Distinct predictions in repeated wraps are
combined by OR rather than silently choosing one copy.
"""
from __future__ import annotations

import math
import os
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from .geometry import ViewInfo
from .cylindrical_cuda_projection import RadialCudaProjectionUnsafeFailure
from ._deps import _numba
from .runtime import allocate_workspace_array, close_memmap_array_without_flush, runtime_telemetry
from .workspace import _cpu_count

# Coordinate intermediates are bounded independently of source volume size.
_PULL_CHUNK_VOXELS = 128 * 1024
_PLANE_BUILD_CHUNK_PIXELS = 1024 * 1024
_RADIAL_METADATA_CHUNK_VALUES = 1024 * 1024
_PLANE_PLAN_CACHE_BYTES = 256 * 1024 * 1024
_PLANE_PLAN_MAX_BYTES = 384 * 1024 * 1024
_OUTPUT_BLOCK_BYTES = 8 * 1024 * 1024
_INFLIGHT_OUTPUT_BYTES = 512 * 1024 * 1024
_PLANE_PLAN_CACHE = OrderedDict()
_PLANE_PLAN_CACHE_SIZE = 0
_PLANE_PLAN_INFLIGHT = {}
_PLANE_PLAN_CACHE_GENERATION = 0
_PLANE_PLAN_LOCK = threading.Lock()


class _RadialPlanePlanTooLarge(RuntimeError):
    pass


@dataclass(frozen=True)
class RadialPlanePlan:
    """Exact 2D ownership shared across source-stack slices and tilt variants.

    Every output base-plane pixel has one local shell and zero or more native
    periodic columns. No stack-sized geometry table is stored. These readonly
    arrays also define the geometry contract for future accelerated consumers.
    """
    base_id: int
    plane_shape: Tuple[int, int]
    shell_index: np.ndarray
    column_offsets: np.ndarray
    native_columns: np.ndarray

    @property
    def nbytes(self):
        return sum(value.nbytes for value in (self.shell_index, self.column_offsets, self.native_columns))


def clear_radial_plane_plan_cache():
    """Forget retained plans without cancelling callers of an older build."""
    global _PLANE_PLAN_CACHE_SIZE, _PLANE_PLAN_CACHE_GENERATION
    with _PLANE_PLAN_LOCK:
        _PLANE_PLAN_CACHE_GENERATION += 1
        _PLANE_PLAN_CACHE.clear()
        _PLANE_PLAN_CACHE_SIZE = 0
        # Leaders and their existing waiters retain the detached Futures. They
        # still finish normally, but cannot populate or remove newer entries.
        _PLANE_PLAN_INFLIGHT.clear()


def _plane_geometry(view, output_shape):
    base = {'transverse': 0, 'sagittal': 1, 'coronal': 2}[str(view.radial_base_view)]
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    axes = ((1, 2), (0, 2), (0, 1))[base]
    return base, tuple(work[a] for a in axes), tuple(output_shape[a] for a in axes)


def _plane_occurrence_strips(view, radii, output_shape, *, first_pixel=0, stop_pixel=None,
                             chunk_pixels=_PULL_CHUNK_VOXELS):
    """Evaluate the original pull equations once per bounded 2D strip."""
    _, work_plane, plane = _plane_geometry(view, output_shape)
    height, width = plane
    native_width = int(view.src_w)
    stop_pixel = height * width if stop_pixel is None else min(height * width, stop_pixel)
    for first in range(first_pixel, stop_pixel, chunk_pixels):
        flat = np.arange(first, min(stop_pixel, first + chunk_pixels), dtype=np.int64)
        yy, xx = flat // width, flat % width
        xx = (xx + .5) * work_plane[1] / width - .5
        yy = (yy + .5) * work_plane[0] / height - .5
        dx, dy = xx - float(view.center_x), yy - float(view.center_y)
        radius = np.hypot(dx, dy)
        global_shell = _nearest_global_shell(radius, radii)
        shell = global_shell - int(view.radial_shell_start)
        valid = ((radius >= float(view.radial_min_radius)) & (radius <= float(view.radial_max_radius))
                 & (shell >= 0) & (shell < int(view.num_slices)))
        local = np.flatnonzero(valid.reshape(-1))
        if not local.size:
            continue
        positions = local + first
        local_shell = shell.reshape(-1)[local].astype(np.int32)
        selected = radii[global_shell.reshape(-1)[local]]
        period = 2.0 * math.pi * selected
        theta = np.mod(np.arctan2(dy.reshape(-1)[local], dx.reshape(-1)[local]), 2.0 * math.pi)
        tiny = period <= 1.0
        if np.any(tiny):
            for column in range(native_width):
                yield positions[tiny], local_shell[tiny], np.full(np.count_nonzero(tiny), column, np.int32)
        ordinary = ~tiny
        positions, local_shell = positions[ordinary], local_shell[ordinary]
        if not positions.size:
            continue
        selected, period, theta = selected[ordinary], period[ordinary], theta[ordinary]
        column_float = np.mod(theta * selected - float(view.radial_arc_origin) + .5, period) - .5
        while True:
            column = np.rint(column_float).astype(np.int64)
            active = (column >= 0) & (column < native_width)
            if not np.any(active):
                break
            yield positions[active], local_shell[active], column[active].astype(np.int32)
            # Preserve repeated-addition rounding from the reference exactly.
            column_float += period


def _build_radial_plane_plan_reference(view, radii, output_shape):
    base, _, plane = _plane_geometry(view, output_shape)
    pixels = math.prod(plane)
    if pixels > np.iinfo(np.uint32).max or pixels * 12 > _PLANE_PLAN_MAX_BYTES:
        raise _RadialPlanePlanTooLarge('Radial plane ownership exceeds the bounded plan budget')
    counts = np.zeros(pixels, np.uint32)
    shells = np.full(pixels, -1, np.int32)
    for positions, local_shell, _ in _plane_occurrence_strips(view, radii, output_shape):
        counts[positions] += 1
        shells[positions] = local_shell
    total = int(counts.sum(dtype=np.uint64))
    if total > np.iinfo(np.uint32).max or pixels * 8 + 4 + total * 4 > _PLANE_PLAN_MAX_BYTES:
        raise _RadialPlanePlanTooLarge('Radial periodic occurrences exceed the bounded plan budget')
    offsets = np.empty(pixels + 1, np.uint32)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.uint32, out=offsets[1:])
    columns = np.empty(total, np.int32)
    counts[:] = offsets[:-1]
    for positions, _, native_columns in _plane_occurrence_strips(view, radii, output_shape):
        columns[counts[positions]] = native_columns
        counts[positions] += 1
    for array in (shells, offsets, columns):
        array.flags.writeable = False
    return RadialPlanePlan(base, tuple(plane), shells, offsets, columns)


def _build_radial_plane_plan(view, radii, output_shape):
    """Evaluate exact NumPy geometry once, assembling CSR in bounded strips.

    The reference evaluates every transcendental twice. Reusing one strip's
    occurrences also reduces Python/NumPy handoffs during concurrent output
    encoding. Strip payloads are bounded by the final plan budget; concatenation
    temporarily retains at most two copies of the admitted column payload.
    """
    base, _, plane = _plane_geometry(view, output_shape)
    pixels = math.prod(plane)
    if pixels > np.iinfo(np.uint32).max or pixels * 12 > _PLANE_PLAN_MAX_BYTES:
        raise _RadialPlanePlanTooLarge('Radial plane ownership exceeds the bounded plan budget')
    shells = np.full(pixels, -1, np.int32)
    offsets = np.empty(pixels + 1, np.uint32)
    offsets[0] = 0
    parts = []
    total = 0
    for first in range(0, pixels, _PLANE_BUILD_CHUNK_PIXELS):
        stop = min(pixels, first + _PLANE_BUILD_CHUNK_PIXELS)
        counts = np.zeros(stop - first, np.uint32)
        occurrences = []
        strip_total = 0
        for positions, local_shell, columns in _plane_occurrence_strips(
                view, radii, output_shape, first_pixel=first, stop_pixel=stop,
                chunk_pixels=_PLANE_BUILD_CHUNK_PIXELS):
            strip_total += int(columns.size)
            if (total + strip_total > np.iinfo(np.uint32).max or
                    pixels * 8 + 4 + (total + strip_total) * 4 > _PLANE_PLAN_MAX_BYTES):
                raise _RadialPlanePlanTooLarge('Radial periodic occurrences exceed the bounded plan budget')
            shells[positions] = local_shell
            local = (positions - first).astype(np.int32)
            counts[local] += 1
            occurrences.append((local, columns))
        np.cumsum(counts, dtype=np.uint32, out=offsets[first + 1:stop + 1])
        # Local write cursors preserve the reference's per-pixel wrap order.
        counts[0] = 0
        counts[1:] = offsets[first + 1:stop]
        part = np.empty(strip_total, np.int32)
        for local, columns in occurrences:
            part[counts[local]] = columns
            counts[local] += 1
        offsets[first + 1:stop + 1] += np.uint32(total)
        total += strip_total
        parts.append(part)
        del occurrences, counts
    columns = np.concatenate(parts)
    for array in (shells, offsets, columns):
        array.flags.writeable = False
    return RadialPlanePlan(base, tuple(plane), shells, offsets, columns)


def _radial_plane_plan(view, radii, output_shape):
    global _PLANE_PLAN_CACHE_SIZE
    base, work_plane, plane = _plane_geometry(view, output_shape)
    key = (base, work_plane, plane, float(view.center_x), float(view.center_y),
           float(view.radial_min_radius), float(view.radial_max_radius), tuple(radii),
           int(view.radial_shell_start), int(view.num_slices), int(view.src_w), float(view.radial_arc_origin))
    flight = None
    owner = False
    try:
        with _PLANE_PLAN_LOCK:
            cached = _PLANE_PLAN_CACHE.pop(key, None)
            if cached is not None:
                _PLANE_PLAN_CACHE[key] = cached
                return cached, True
            flight = _PLANE_PLAN_INFLIGHT.get(key)
            if flight is None:
                flight = Future()
                # A caller abandoning its wait must not cancel shared work.
                flight.set_running_or_notify_cancel()
                generation = _PLANE_PLAN_CACHE_GENERATION
                owner = True
                _PLANE_PLAN_INFLIGHT[key] = flight
        if not owner:
            return flight.result(), True

        # Build and wait outside the cache lock: unrelated keys remain parallel.
        built = _build_radial_plane_plan(view, radii, output_shape)
        with _PLANE_PLAN_LOCK:
            if (generation == _PLANE_PLAN_CACHE_GENERATION
                    and _PLANE_PLAN_INFLIGHT.get(key) is flight
                    and built.nbytes <= _PLANE_PLAN_CACHE_BYTES):
                while _PLANE_PLAN_CACHE and _PLANE_PLAN_CACHE_SIZE + built.nbytes > _PLANE_PLAN_CACHE_BYTES:
                    _, evicted = _PLANE_PLAN_CACHE.popitem(last=False)
                    _PLANE_PLAN_CACHE_SIZE -= evicted.nbytes
                _PLANE_PLAN_CACHE[key] = built
                _PLANE_PLAN_CACHE_SIZE += built.nbytes
        # Completion can invoke callbacks; never invoke it under our cache lock.
        # Keep the flight discoverable through completion even for uncached plans.
        flight.set_result(built)
        return built, False
    except BaseException as exc:
        if owner and flight is not None and not flight.done():
            flight.set_exception(exc)
        raise
    finally:
        if owner and flight is not None:
            with _PLANE_PLAN_LOCK:
                if _PLANE_PLAN_INFLIGHT.get(key) is flight:
                    _PLANE_PLAN_INFLIGHT.pop(key)


def _radial_projection_metadata(view, source_shape, output_shape, plan):
    stack_axis = (0, 1, 2)[plan.base_id]
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    stack_length = work[stack_axis]
    stack_centers = (np.arange(output_shape[stack_axis], dtype=np.float64) + .5) * stack_length / output_shape[stack_axis] - .5
    _, work_plane, plane = _plane_geometry(view, output_shape)
    vertical = str(view.tilt_direction) == 'vertical'
    axis = 0 if vertical else 1
    center = float(view.center_y if vertical else view.center_x)
    ideal = (np.arange(plane[axis], dtype=np.float64) + .5) * work_plane[axis] / plane[axis] - .5 - center
    sampled = np.zeros((int(view.num_slices), int(view.src_w)), np.float64)
    if bool(view.radial_tilted_source):
        tangent = math.tan(math.radians(float(view.tilt_angle_deg)))
        ideal *= tangent
        columns = np.arange(int(view.src_w), dtype=np.float64)
        arc = float(view.radial_arc_origin) + columns
        radii = np.asarray(view.radial_radii, dtype=np.float64)
        # Preserve NumPy's exact operations, but do not reacquire the interpreter
        # between a handful of small ufuncs for each of thousands of shells.
        rows = max(1, _RADIAL_METADATA_CHUNK_VALUES // max(1, int(view.src_w)))
        for first in range(0, len(radii), rows):
            radius = radii[first:first + rows, None]
            theta = np.remainder(arc[None, :] / radius, 2.0 * math.pi)
            offset = radius * (np.sin(theta) if vertical else np.cos(theta))
            sampled[first:first + rows] = tangent * offset
    else:
        ideal[:] = 0.0
    row_map = _processing_index(np.arange(int(view.src_h), dtype=np.int64), int(view.src_h), int(source_shape[1])).astype(np.int32)
    column_map = _processing_index(np.arange(int(view.src_w), dtype=np.int64), int(view.src_w), int(source_shape[2])).astype(np.int32)
    return stack_centers, ideal, sampled, row_map, column_map, stack_length, vertical


def _gather_radial_pixel(source, shells, offsets, columns, sampled, row_map, column_map,
                         p, stack, ideal_shear, stack_length, height_origin, native_height,
                         bboxes, use_bboxes):
    shell = shells[p]
    if shell < 0:
        return np.uint8(0)
    if use_bboxes and (bboxes[shell, 1] <= bboxes[shell, 0] or bboxes[shell, 3] <= bboxes[shell, 2]):
        return np.uint8(0)
    ideal_height = stack - ideal_shear
    if ideal_height < 0.0 or ideal_height > stack_length - 1:
        return np.uint8(0)
    for at in range(offsets[p], offsets[p + 1]):
        column = columns[at]
        height = stack - sampled[shell, column]
        height = min(max(height, 0.0), float(stack_length - 1))
        row = int(np.rint(height)) - height_origin
        if row < 0 or row >= native_height:
            continue
        pr, pc = row_map[row], column_map[column]
        if use_bboxes and (pr < bboxes[shell, 0] or pr >= bboxes[shell, 1]
                           or pc < bboxes[shell, 2] or pc >= bboxes[shell, 3]):
            continue
        if source[shell, pr, pc] != 0:
            return np.uint8(1)
    return np.uint8(0)


def _project_radial_block(source, shells, offsets, columns, sampled, row_map, column_map,
                          stack_centers, ideal_axis, stack_length, height_origin, native_height,
                          base_id, vertical, plane_width, out_h, out_w, first_z, count,
                          bboxes, use_bboxes):
    result = np.zeros((count, out_h, out_w), np.uint8)
    for dz in range(count):
        z = first_z + dz
        for y in range(out_h):
            for x in range(out_w):
                if base_id == 0:
                    p, stack_index, v, u = y * plane_width + x, z, y, x
                elif base_id == 1:
                    p, stack_index, v, u = z * plane_width + x, y, z, x
                else:
                    p, stack_index, v, u = z * plane_width + y, x, z, y
                result[dz, y, x] = _gather_radial_pixel(
                    source, shells, offsets, columns, sampled, row_map, column_map,
                    p, stack_centers[stack_index], ideal_axis[v if vertical else u],
                    stack_length, height_origin, native_height, bboxes, use_bboxes,
                )
    return result


if _numba is not None:
    _gather_radial_pixel = _numba.njit(cache=True, nogil=True, inline='always', fastmath=False)(_gather_radial_pixel)
    _project_radial_block = _numba.njit(cache=True, nogil=True, fastmath=False)(_project_radial_block)


def _radial_block_schedule(depth, plane_bytes, workers):
    block_depth = max(1, min(depth, _OUTPUT_BLOCK_BYTES // max(1, plane_bytes)))
    block_bytes = plane_bytes * block_depth
    worker_count = max(1, min(int(workers), _cpu_count(), math.ceil(depth / block_depth),
                              max(1, _INFLIGHT_OUTPUT_BYTES // max(1, block_bytes))))
    return block_depth, worker_count


def _ordered_radial_blocks(project, depth, plane_bytes, workers):
    block_depth, worker_count = _radial_block_schedule(depth, plane_bytes, workers)
    starts = iter(range(0, depth, block_depth))
    if worker_count == 1:
        for first in starts:
            yield first, project(first, min(block_depth, depth - first))
        return
    pending = deque()
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='radial-project') as pool:
        try:
            for first in starts:
                pending.append((first, pool.submit(project, first, min(block_depth, depth - first))))
                if len(pending) >= worker_count:
                    z, future = pending.popleft()
                    yield z, future.result()
                    del future
            while pending:
                z, future = pending.popleft()
                yield z, future.result()
                del future
        finally:
            for _, future in pending:
                future.cancel()


def radial_cuda_backproject_enabled():
    return os.environ.get('YOLO_TTA_GPU_RADIAL_BACKPROJECT', '1').strip().lower() not in (
        '0', 'false', 'no', 'off',
    )


class _RadialCudaStage:
    """Keep one device lease until every borrowed source/GPU operation has settled."""
    def __init__(self, projector, lease):
        self.projector = projector
        self.lease = lease
        self.device_index = int(lease.device_index)
        self.max_block_depth = int(projector.max_block_depth)

    def project(self, first, count):
        return self.projector.project(first, count)

    def project_encoded(self, first, count, packed=False):
        return self.projector.project_encoded(first, count, packed=packed)

    def close(self):
        if self.lease is None:
            return
        lease = self.lease
        try:
            _close_radial_cuda_resources(self.projector, lease)
        except RadialCudaProjectionUnsafeFailure:
            raise
        except BaseException:
            self.lease = None
            raise
        else:
            self.lease = None


def _close_radial_cuda_resources(projector, lease):
    try:
        if projector is not None:
            projector.close()
    except RadialCudaProjectionUnsafeFailure as exc:
        # An unfenced operation still owns its source and device. Preserve
        # both on the fatal exception; do not admit another stage here.
        exc.stage_lease = lease
        raise
    except BaseException:
        lease.release()
        raise
    else:
        lease.release()


def _try_radial_cuda_stage(source, plan, metadata, view, shape, bboxes, use_bboxes):
    """Admit before publication; unavailable/busy/undersized GPUs retain CPU progress."""
    if not radial_cuda_backproject_enabled():
        return None
    try:
        import torch
        if not bool(torch.cuda.is_available()):
            return None
    except Exception:
        return None
    from .backprojection import _try_acquire_main_process_gpu_stage

    # This purpose deliberately uses the normal inference-priority policy. The
    # legacy non-D1 "backprojection" exception can borrow idle feeder devices.
    lease = _try_acquire_main_process_gpu_stage(torch, f'Radial source projection {view.name}')
    if lease is None:
        print(f'Radial CUDA fallback {view.name}: eligible devices are busy or retiring; using CPU.', flush=True)
        return None
    projector = None
    try:
        from .cylindrical_cuda_projection import RadialCudaProjector

        projector = RadialCudaProjector(
            source, plan, metadata, view, shape, bboxes, use_bboxes, int(lease.device_index),
        )
        return _RadialCudaStage(projector, lease)
    except RadialCudaProjectionUnsafeFailure as exc:
        exc.stage_lease = lease
        raise
    except Exception as exc:
        _close_radial_cuda_resources(projector, lease)
        print(f'Radial CUDA fallback {view.name}: device admission/preflight failed ({exc}); using CPU.', flush=True)
        return None
    except BaseException:
        _close_radial_cuda_resources(projector, lease)
        raise


def _ordered_radial_cuda_blocks(stage, depth, packed=None):
    """A single GPU producer overlaps the current immutable block's CPU consumer."""
    block_depth = max(1, int(stage.max_block_depth))
    project = stage.project if packed is None else lambda first, count: stage.project_encoded(first, count, packed=packed)
    starts = iter(range(0, depth, block_depth))
    # Each project result owns its host allocation. At most one ready block and
    # one in-flight block exist; executor shutdown precedes GPU/lease cleanup.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='radial-cuda-project') as pool:
        future = None
        try:
            first = next(starts, None)
            if first is None:
                return
            future = pool.submit(project, first, min(block_depth, depth - first))
            for following in starts:
                block = future.result()
                future = pool.submit(project, following, min(block_depth, depth - following))
                yield first, block
                del block
                first = following
            yield first, future.result()
        finally:
            if future is not None:
                future.cancel()


def _nearest_global_shell(radius: np.ndarray, radii: np.ndarray) -> np.ndarray:
    right = np.searchsorted(radii, radius, side='left')
    right = np.clip(right, 0, len(radii) - 1)
    left = np.maximum(right - 1, 0)
    # A midpoint belongs to the inner shell. This also handles exact endpoints.
    return np.where(radius - radii[left] <= radii[right] - radius, left, right)


def _processing_index(native: np.ndarray, native_count: int, processing_count: int) -> np.ndarray:
    if native_count == processing_count:
        return native
    return np.minimum(
        ((native.astype(np.float64) + 0.5) * processing_count / native_count).astype(np.int64),
        processing_count - 1,
    )


def _occurrence_rows(view, source_stack, radii, columns, stack_length):
    """Invert shear at the actual discrete shell sample, independently per wrap.

    Radius/arc quantization changes the in-plane point. Reusing the ideal source
    voxel's inverse-shear height can select zero padding across a source face.
    Height selection is global before patch offsets, so overlapping bands agree
    even at nearest-neighbor ties.
    """
    height = np.asarray(source_stack, dtype=np.float64)
    if bool(view.radial_tilted_source):
        theta = np.remainder((float(view.radial_arc_origin) + columns) / radii, 2.0 * math.pi)
        offset = radii * (np.sin(theta) if view.tilt_direction == 'vertical' else np.cos(theta))
        height = height - math.tan(math.radians(float(view.tilt_angle_deg))) * offset
    global_height = np.rint(np.clip(height, 0.0, float(stack_length - 1))).astype(np.int64)
    return global_height - int(view.radial_height_origin)


def _pull_radial_chunk(
    source: np.ndarray,
    view: ViewInfo,
    radii: np.ndarray,
    output_shape: Tuple[int, int, int],
    z: int,
    first: int,
    stop: int,
) -> np.ndarray:
    """Project one bounded, flattened source-coordinate XY strip."""
    out_t, out_h, out_w = output_shape
    work_t, work_h, work_w = int(view.full_t), int(view.full_h), int(view.full_w)
    flat = np.arange(first, stop, dtype=np.int64)
    wt = (float(z) + 0.5) * work_t / out_t - 0.5
    wy = (flat // out_w + 0.5) * work_h / out_h - 0.5
    wx = (flat % out_w + 0.5) * work_w / out_w - 0.5
    base = str(view.radial_base_view)
    if base == 'transverse':
        stack, py, px, stack_len = wt, wy, wx, work_t
    elif base == 'sagittal':
        stack, py, px, stack_len = wy, np.full(flat.shape, wt), wx, work_h
    elif base == 'coronal':
        stack, py, px, stack_len = wx, np.full(flat.shape, wt), wy, work_w
    else:
        raise ValueError(f'Unsupported Radial base {base!r}')
    dx, dy = px - float(view.center_x), py - float(view.center_y)
    radius = np.hypot(dx, dy)
    source_stack = np.broadcast_to(np.asarray(stack, dtype=np.float64), flat.shape)
    height = source_stack.copy()
    if bool(view.radial_tilted_source):
        direction = str(view.tilt_direction)
        if direction not in ('vertical', 'horizontal'):
            raise ValueError(f'Unsupported Radial tilt direction {direction!r}')
        axis = dy if direction == 'vertical' else dx
        height -= math.tan(math.radians(float(view.tilt_angle_deg))) * axis
    global_shell = _nearest_global_shell(radius, radii)
    shell = global_shell - int(view.radial_shell_start)
    valid = (
        (radius >= float(view.radial_min_radius))
        & (radius <= float(view.radial_max_radius))
        & (height >= 0.0) & (height <= float(stack_len - 1))
        & (shell >= 0) & (shell < source.shape[0])
    )
    result = np.zeros(flat.shape, dtype=np.uint8)
    positions = np.flatnonzero(valid)
    if not positions.size:
        return result
    local_shell = shell[positions]
    selected_radius = radii[global_shell[positions]]
    circumference = 2.0 * math.pi * selected_radius
    theta = np.mod(np.arctan2(dy[positions], dx[positions]), 2.0 * math.pi)
    width = int(view.src_w)

    # When a circumference is <= one native pixel, every native column contains
    # a nearest periodic occurrence. Radius zero similarly represents the axis.
    tiny = circumference <= 1.0
    if np.any(tiny):
        tiny_positions = positions[tiny]
        for column in range(width):
            rows = _occurrence_rows(
                view, source_stack[tiny_positions], selected_radius[tiny], column, stack_len,
            )
            inside = (rows >= 0) & (rows < int(view.src_h))
            if np.any(inside):
                proc_row = _processing_index(rows[inside], int(view.src_h), int(source.shape[1]))
                proc_col = int(_processing_index(np.asarray(column), width, int(source.shape[2])))
                result[tiny_positions[inside]] |= np.asarray(
                    source[local_shell[tiny][inside], proc_row, proc_col] != 0, dtype=np.uint8,
                )
    ordinary = ~tiny
    if not np.any(ordinary):
        return result
    positions = positions[ordinary]
    local_shell = local_shell[ordinary]
    selected_radius = selected_radius[ordinary]
    period = circumference[ordinary]
    arc = theta[ordinary] * selected_radius
    # Include an occurrence just below zero when it rounds onto column zero.
    column_float = np.mod(arc - float(view.radial_arc_origin) + 0.5, period) - 0.5
    while True:
        column = np.rint(column_float).astype(np.int64)
        active = (column >= 0) & (column < width)
        if not np.any(active):
            break
        rows = _occurrence_rows(view, source_stack[positions], selected_radius, column, stack_len)
        active &= (rows >= 0) & (rows < int(view.src_h))
        if np.any(active):
            proc_row = _processing_index(rows[active], int(view.src_h), int(source.shape[1]))
            proc_column = _processing_index(column[active], width, int(source.shape[2]))
            result[positions[active]] |= np.asarray(
                source[local_shell[active], proc_row, proc_column] != 0,
                dtype=np.uint8,
            )
        column_float += period
    return result


def backproject_radial_volume_to_volume(
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 0,
    workers: int = 1,
    out_shape_tyx: Optional[Tuple[int, int, int]] = None,
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    sink_only: bool = False,
    known_slice_bboxes: Optional[np.ndarray] = None,
):
    """Project one intrinsic Radial patch trajectory into source geometry.

    The output is caller-owned. Dense output uses path-backed storage, while an
    optional sink receives completed source blocks in order. Neither path
    materializes a whole-volume coordinate map or changes the borrowed input.
    """
    from .geometry import radial_global_radii
    from .backprojection import SinkOnlyProjectionResult, _emit_projection_block_callback

    if str(radial_view.family) != 'radial':
        raise ValueError('Cylindrical shell projection requires a Radial view')
    source = np.asarray(radial_mask_mm)
    if source.ndim != 3 or min(source.shape) <= 0:
        raise ValueError('Radial projection requires a nonempty three-dimensional mask')
    if int(source.shape[0]) != int(radial_view.num_slices):
        raise ValueError('Radial mask depth differs from the shell trajectory')
    radii = np.asarray(radial_global_radii(radial_view), dtype=np.float64)
    start = int(radial_view.radial_shell_start)
    local_radii = np.asarray(radial_view.radial_radii, dtype=np.float64)
    if (
        radii.ndim != 1 or not radii.size or not np.isfinite(radii).all()
        or np.any(radii < 0) or np.any(np.diff(radii) <= 0)
        or start < 0 or start + source.shape[0] > radii.size
        or local_radii.shape != (source.shape[0],)
        or not np.array_equal(local_radii, radii[start:start + source.shape[0]])
    ):
        raise ValueError('Radial trajectory does not match its global radius grid')
    shape = tuple(int(v) for v in (
        out_shape_tyx if out_shape_tyx is not None
        else (radial_view.full_t, radial_view.full_h, radial_view.full_w)
    ))
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Radial projection requires positive source output dimensions')
    if sink_only and projection_block_callback is None:
        raise ValueError('Radial sink-only projection requires a block consumer')
    total_started = time.perf_counter()
    print(f'Radial projection planning {radial_view.name}: source={tuple(source.shape)}, '
          f'output={shape}, requested_workers={max(1, int(workers))}', flush=True)
    if bool(radial_view.radial_tilted_source) and radial_view.tilt_direction not in ('vertical', 'horizontal'):
        raise ValueError(f'Unsupported Radial tilt direction {radial_view.tilt_direction!r}')
    bboxes = np.zeros((source.shape[0], 4), np.int64)
    use_bboxes = known_slice_bboxes is not None
    if use_bboxes:
        bboxes = np.ascontiguousarray(known_slice_bboxes, dtype=np.int64)
        if (bboxes.shape != (source.shape[0], 4) or np.any(bboxes < 0)
                or np.any(bboxes[:, 0] > bboxes[:, 1]) or np.any(bboxes[:, 2] > bboxes[:, 3])
                or np.any(bboxes[:, 1] > source.shape[1]) or np.any(bboxes[:, 3] > source.shape[2])):
            raise ValueError('Radial source bounding boxes do not match the processing mask grid')
    cuda_stage = None
    output = None
    failed = False
    try:
        project = None
        plan_seconds = setup_seconds = sink_seconds = metadata_host_seconds = cuda_admission_seconds = 0.0
        plan_bytes = 0
        cache_hit = False
        backend_name = 'cpu_numpy_reference'
        if _numba is not None or radial_cuda_backproject_enabled():
            started = time.perf_counter()
            try:
                plan, cache_hit = _radial_plane_plan(radial_view, radii, shape)
            except _RadialPlanePlanTooLarge:
                # Retain the bounded numerical reference for unusual geometry whose
                # explicit occurrence map cannot fit the plan budget.
                plan = None
            plan_seconds = time.perf_counter() - started
            if plan is not None:
                metadata_started = time.perf_counter()
                metadata = _radial_projection_metadata(
                    radial_view, source.shape, shape, plan,
                )
                metadata_host_seconds = time.perf_counter() - metadata_started
                centers, ideal, sampled, row_map, column_map, stack_length, vertical = metadata
                arguments = (
                    source, plan.shell_index, plan.column_offsets, plan.native_columns,
                    sampled, row_map, column_map, centers, ideal, int(stack_length),
                    int(radial_view.radial_height_origin), int(radial_view.src_h), plan.base_id,
                    bool(vertical), int(plan.plane_shape[1]), int(shape[1]), int(shape[2]),
                )
                admission_started = time.perf_counter()
                cuda_stage = _try_radial_cuda_stage(
                    source, plan, metadata, radial_view, shape, bboxes, use_bboxes,
                )
                cuda_admission_seconds = time.perf_counter() - admission_started
                if cuda_stage is not None:
                    project = cuda_stage.project
                    backend_name = 'cuda_factored'
                elif _numba is not None:
                    def project(first, count):
                        return _project_radial_block(*arguments, int(first), int(count), bboxes, bool(use_bboxes))
                    # Compile before starting output or executor threads. A genuine
                    # kernel failure must not silently fall back after partial delivery.
                    project(0, 0)
                    backend_name = 'cpu_numba_factored'
                setup_seconds = time.perf_counter() - started
                plan_bytes = int(plan.nbytes)
                telemetry = runtime_telemetry()
                telemetry.gauge('projection.radial.plan_seconds', plan_seconds)
                telemetry.add('projection.radial.plan_cache_hits', int(cache_hit))
                telemetry.gauge('projection.radial.plan_bytes', int(plan.nbytes))
                telemetry.gauge('projection.radial.backend', backend_name)
        if project is None:
            runtime_telemetry().gauge('projection.radial.backend', 'cpu_numpy_reference')
        block_depth, actual_workers = _radial_block_schedule(shape[0], shape[1] * shape[2], int(workers))
        if cuda_stage is not None:
            block_depth, actual_workers = int(cuda_stage.max_block_depth), 1
        elif project is None:
            block_depth, actual_workers = 1, 1
        encoded_format = getattr(projection_block_callback, 'encoded_slice_format', None)
        compact_output = bool(cuda_stage is not None and sink_only
                              and encoded_format in ('raw_u8', 'packbits_little')
                              and callable(getattr(projection_block_callback, 'consume_encoded_block', None)))
        packed_output = encoded_format == 'packbits_little'
        if compact_output:
            backend_name = 'cuda_factored_compact'
            runtime_telemetry().gauge('projection.radial.backend', backend_name)
        setup_metrics = (f', metadata_host_seconds={metadata_host_seconds:.6f}'
                         f', cuda_admission_seconds={cuda_admission_seconds:.6f}')
        if cuda_stage is not None:
            projector = cuda_stage.projector
            setup_metrics += ''.join(f', {name}={float(getattr(projector, name, 0.0)):.6f}' for name in
                ('source_upload_seconds', 'source_pack_seconds', 'geometry_upload_seconds',
                 'preflight_seconds', 'contract_validation_seconds', 'module_setup_seconds',
                 'buffer_setup_seconds', 'constructor_seconds', 'cuda_graph_setup_seconds'))
            setup_metrics += (f', source_layout={getattr(projector, "source_layout", "unknown")}'
                              f', source_bytes={int(getattr(projector, "source_bytes", source.nbytes))}'
                              f', source_h2d_bytes={int(getattr(projector, "source_h2d_bytes", source.nbytes))}'
                              f', source_pack_backend={getattr(projector, "source_pack_backend", "unknown")}'
                              f', cuda_graph_enabled={int(getattr(projector, "cuda_graph_enabled", False))}'
                              f', cuda_graph_note={getattr(projector, "cuda_graph_note", "unknown")}')
        print(f'Radial projection start {radial_view.name}: backend={backend_name}, '
              f'workers={actual_workers}, block_z={block_depth}, plan_MiB={plan_bytes / 2**20:.2f}, '
              f'cache_hit={int(cache_hit)}, plan_s={plan_seconds:.6f}, setup_s={setup_seconds:.6f}'
              + (f', device=cuda:{cuda_stage.device_index}' if cuda_stage is not None else '')
              + (f', payload={encoded_format}' if compact_output else '') + setup_metrics, flush=True)
        try:
            if not sink_only:
                output = allocate_workspace_array(
                    shape=shape, dtype=np.uint8, path=Path(out_path), desc=f'{desc} workspace',
                    prefer_memory=False, prefer_memfd=False, reserve_bytes=int(reserve_bytes),
                    initialize_zero=False,
                )
            if project is not None:
                blocks = (_ordered_radial_cuda_blocks(cuda_stage, shape[0], packed_output if compact_output else None) if cuda_stage is not None
                          else _ordered_radial_blocks(project, shape[0], shape[1] * shape[2], int(workers)))
                try:
                    for z, block in blocks:
                        if output is not None:
                            output[z:z + len(block)] = block
                        sink_started = time.perf_counter()
                        if compact_output:
                            expected_count = min(int(cuda_stage.max_block_depth), shape[0] - z)
                            if (block.first_z != z or bool(block.packed) != packed_output
                                    or len(block.records) != expected_count):
                                raise RuntimeError('Radial CUDA encoded block identity/format/count differs from its request')
                            projection_block_callback.consume_encoded_block(
                                z, block.records, block.payload, packed=packed_output,
                            )
                        else:
                            _emit_projection_block_callback(
                                projection_block_callback, z, block, desc=desc, required=bool(sink_only),
                            )
                        sink_seconds += time.perf_counter() - sink_started
                        del block
                finally:
                    # Generator shutdown waits/cancels outstanding work before the
                    # borrowed input or caller's writer can be retired on failure.
                    blocks.close()
            else:
                for z in range(shape[0]):
                    plane = np.empty(shape[1:], dtype=np.uint8) if output is None else output[z]
                    flat_plane = plane.reshape(-1)
                    for first in range(0, flat_plane.size, _PULL_CHUNK_VOXELS):
                        stop = min(flat_plane.size, first + _PULL_CHUNK_VOXELS)
                        flat_plane[first:stop] = _pull_radial_chunk(source, radial_view, radii, shape, z, first, stop)
                    sink_started = time.perf_counter()
                    _emit_projection_block_callback(
                        projection_block_callback, z, plane[None], desc=desc, required=bool(sink_only),
                    )
                    sink_seconds += time.perf_counter() - sink_started
            if output is not None:
                flush = getattr(output, 'flush', None)
                if callable(flush):
                    flush()
            device_metrics = ''
            if cuda_stage is not None:
                projector = cuda_stage.projector
                timings = ('kernel_seconds', 'metadata_seconds', 'pack_seconds', 'd2h_seconds', 'cuda_graph_seconds')
                transfers = ('metadata_d2h_bytes', 'payload_d2h_bytes', 'dense_d2h_bytes',
                             'cuda_graph_blocks', 'empty_encoded_blocks')
                device_metrics = ''.join(f', {name}={float(getattr(projector, name, 0.0)):.6f}' for name in timings)
                device_metrics += ''.join(f', {name}={int(getattr(projector, name, 0))}' for name in transfers)
            print(f'Radial projection complete {radial_view.name}: backend={backend_name}, '
                  f'workers={actual_workers}, plan_MiB={plan_bytes / 2**20:.2f}, cache_hit={int(cache_hit)}, '
                  f'plan_s={plan_seconds:.6f}, setup_s={setup_seconds:.6f}, '
                  f'total_s={time.perf_counter() - total_started:.6f}, sink_wall_s={sink_seconds:.6f}'
                  + setup_metrics + device_metrics, flush=True)
            return output if output is not None else SinkOnlyProjectionResult(shape)
        except BaseException as exc:
            failed = True
            close_memmap_array_without_flush(output)
            print(f'Radial projection failed {radial_view.name}: backend={backend_name}, '
                  f'total_s={time.perf_counter() - total_started:.6f}, error={exc}', flush=True)
            raise
    finally:
        if cuda_stage is not None:
            try:
                cuda_stage.close()
            except RadialCudaProjectionUnsafeFailure:
                # Never let a transactional caller retry on a device whose
                # stream still has uncertain ownership.
                raise
            except BaseException as exc:
                if not failed:
                    close_memmap_array_without_flush(output)
                    raise
                print(f'Radial CUDA cleanup failed {radial_view.name}: {exc}', flush=True)
