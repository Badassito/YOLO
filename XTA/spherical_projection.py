"""Bounded source-coordinate pull projection for spherical QSC shell patches.

All radii share one face lattice. Radius and face-pixel selection happen in
global coordinates before patch offsets, so overlapping patches agree on
nearest-neighbor ties. Faces are closed: their incident edges and corners may
contribute independently to the caller's ordinary OR union.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
import math
import operator
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

from .qsc import qsc_forward_face
from .geometry_quality import spherical_cpu_compiled_requested
from .spherical_projection_bounds import spherical_output_bounds
from .spherical_projection_cuda import SphericalCudaProjectionUnsafeFailure
from .cylindrical_cuda_projection import RadialEncodedBlock, RadialEncodedSlice, _MAX_ENCODED_SLICES
from .runtime import allocate_workspace_array, close_memmap_array_without_flush, runtime_telemetry
from .workspace import _cpu_count

if TYPE_CHECKING:
    from .geometry import ViewInfo


_PULL_CHUNK_VOXELS = 128 * 1024
_OUTPUT_BLOCK_BYTES = 8 * 1024 * 1024
_INFLIGHT_WORK_BYTES = 256 * 1024 * 1024
_CHUNK_BYTES_PER_VOXEL = 384
_COMPILED_CHUNK_BYTES_PER_VOXEL = 8
_CPU_ENCODED_SLICE_BYTES = 1024
_CUDA_RECHECK_SLICES = 8
_CUDA_RECHECK_SECONDS = 1.0


def spherical_cpu_compact_enabled():
    return os.environ.get('YOLO_TTA_CPU_SPHERICAL_COMPACT', '1').strip().lower() not in (
        '', '0', 'false', 'no', 'off',
    )


@dataclass(frozen=True)
class SphericalCpuEncodedBlock(RadialEncodedBlock):
    pull_voxels: int = 0
    scan_voxels: int = 0


def spherical_cuda_backproject_enabled():
    return os.environ.get('YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', '1').strip().lower() not in (
        '0', 'false', 'no', 'off',
    )


def _close_spherical_cuda_resources(projector, lease):
    try:
        if projector is not None:
            projector.close()
    except SphericalCudaProjectionUnsafeFailure as exc:
        exc.stage_lease = lease
        raise
    except BaseException:
        lease.release()
        raise
    else:
        lease.release()


class _SphericalCudaStage:
    def __init__(self, projector, lease):
        self.projector, self.lease = projector, lease
        self.device_index = int(lease.device_index)
        self.max_block_depth = int(projector.max_block_depth)

    def project(self, first, count):
        return self.projector.project(first, count)

    def project_encoded(self, first, count, packed=False):
        return self.projector.project_encoded(first, count, packed=packed)

    def close(self):
        if self.lease is None:
            return
        try:
            _close_spherical_cuda_resources(self.projector, self.lease)
        except SphericalCudaProjectionUnsafeFailure:
            raise
        except BaseException:
            self.lease = None
            raise
        else:
            self.lease = None


def _try_spherical_cuda_stage(source, view, shape, bboxes, *, quiet=False):
    if not spherical_cuda_backproject_enabled():
        if not quiet:
            print(f'Spherical CUDA disabled {view.name}: using bounded CPU projection.', flush=True)
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            if not quiet:
                print(f'Spherical CUDA fallback {view.name}: CUDA unavailable; using CPU.', flush=True)
            return None
    except Exception as exc:
        if not quiet:
            print(f'Spherical CUDA fallback {view.name}: runtime unavailable ({exc}); using CPU.', flush=True)
        return None
    from .backprojection import _try_acquire_main_process_gpu_stage, _cancel_main_process_spherical_retirement_request
    from .spherical_projection_cuda import SphericalCudaProjector

    lease = _try_acquire_main_process_gpu_stage(torch, f'Spherical source projection {view.name}')
    if lease is None:
        if not quiet:
            print(f'Spherical CUDA fallback {view.name}: eligible devices are busy or retiring; using CPU.', flush=True)
        return None
    projector = None
    try:
        projector = SphericalCudaProjector(source, view, shape, bboxes, int(lease.device_index))
        return _SphericalCudaStage(projector, lease)
    except SphericalCudaProjectionUnsafeFailure as exc:
        exc.stage_lease = lease
        raise
    except Exception as exc:
        _cancel_main_process_spherical_retirement_request(f'Spherical source projection {view.name}', failed=True)
        _close_spherical_cuda_resources(projector, lease)
        if not quiet:
            print(f'Spherical CUDA fallback {view.name}: admission/preflight failed ({exc}); using CPU.', flush=True)
        return None
    except BaseException:
        _close_spherical_cuda_resources(projector, lease)
        raise


def _ordered_spherical_cuda_blocks(stage, depth, packed=None, *, first_z=0):
    """Overlap one bounded device producer with the preceding host consumer."""
    size = max(1, int(stage.max_block_depth))
    project = stage.project if packed is None else lambda z, n: stage.project_encoded(z, n, packed=packed)
    starts = iter(range(int(first_z), depth, size))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='spherical-cuda-project') as pool:
        future = None
        try:
            first = next(starts, None)
            if first is None:
                return
            future = pool.submit(project, first, min(size, depth - first))
            for following in starts:
                block = future.result()
                future = pool.submit(project, following, min(size, depth - following))
                yield first, block
                del block
                first = following
            yield first, future.result()
        finally:
            if future is not None:
                future.cancel()


def _nearest_global_shell(radius: np.ndarray, radii: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(radii, radius, side='left'), 0, len(radii) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(radius - radii[left] <= radii[right] - radius, left, right)


def _processing_index(native: np.ndarray, native_count: int, processing_count: int) -> np.ndarray:
    """Select the processing pixel containing the native pixel center."""
    if native_count == processing_count:
        return native
    return np.minimum(((native.astype(np.float64) + .5) * processing_count / native_count).astype(np.int64),
                      processing_count - 1)


def _pull_spherical_chunk(source, view, radii, rotation, output_shape, z, first, stop, bboxes=None):
    """Evaluate one flattened source XY strip without any retained 3D map."""
    out_t, out_h, out_w = output_shape
    work_t, work_h, work_w = int(view.full_t), int(view.full_h), int(view.full_w)
    flat = np.arange(first, stop, dtype=np.int64)
    # Output coordinates are voxel centers mapped into the working grid.
    dx = ((flat % out_w + .5) * work_w / out_w - .5) - (work_w - 1) / 2.0
    dy = ((flat // out_w + .5) * work_h / out_h - .5) - (work_h - 1) / 2.0
    dz = ((float(z) + .5) * work_t / out_t - .5) - (work_t - 1) / 2.0
    radius = np.sqrt(dx * dx + dy * dy + dz * dz)
    valid = ((radius >= float(view.spherical_min_radius))
             & (radius <= float(view.spherical_max_radius)))
    result = np.zeros(flat.shape, dtype=np.uint8)
    positions = np.flatnonzero(valid)
    if not positions.size:
        return result
    # Row-vector inverse of the cube's local-to-world orthogonal rotation.
    dx, dy = dx[positions], dy[positions]
    local = np.stack(tuple(dx * rotation[0, a] + dy * rotation[1, a] + dz * rotation[2, a]
                           for a in range(3)), axis=-1)
    u, v, member = qsc_forward_face(local, int(view.spherical_face))
    count = int(view.spherical_face_intervals)
    columns = np.rint((u + 1.0) * count / 2.0).astype(np.int64) - int(view.spherical_u_origin)
    rows = np.rint((1.0 - v) * count / 2.0).astype(np.int64) - int(view.spherical_v_origin)
    member &= ((columns >= 0) & (columns < int(view.src_w))
               & (rows >= 0) & (rows < int(view.src_h)))
    if not np.any(member):
        return result
    positions = positions[member]
    shell = _nearest_global_shell(radius[positions], radii)
    pr = _processing_index(rows[member], int(view.src_h), int(source.shape[1]))
    pc = _processing_index(columns[member], int(view.src_w), int(source.shape[2]))
    if bboxes is not None:
        boxes = bboxes[shell]
        inside = ((pr >= boxes[:, 0]) & (pr < boxes[:, 1])
                  & (pc >= boxes[:, 2]) & (pc < boxes[:, 3]))
        positions, shell, pr, pc = (a[inside] for a in (positions, shell, pr, pc))
    result[positions] = np.asarray(source[shell, pr, pc] != 0, dtype=np.uint8)
    return result


def _project_spherical_block(source, view, radii, rotation, shape, first_z, count, bboxes=None,
                             output_bounds=None, cancel_event=None, cpu_pull=None):
    # Keep the unbounded path as the independent full-plane reference used by
    # CUDA qualification. Production CPU work can skip analytic Z/Y bands, but
    # retains contiguous full-width strips: one NumPy call per cropped row
    # would replace bounded vector work with thousands of tiny QSC calls.
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError('Spherical CPU block retired before publication')
    if output_bounds is None:
        block = np.empty((count, shape[1], shape[2]), dtype=np.uint8)
        z0, z1, first_pixel, stop_pixel = first_z, first_z + count, 0, shape[1] * shape[2]
        pull = _pull_spherical_chunk
    else:
        block = np.zeros((count, shape[1], shape[2]), dtype=np.uint8)
        z0, z1, y0, y1, x0, x1 = output_bounds.block(first_z, count)
        if z0 == z1 or y0 == y1 or x0 == x1:
            return block
        first_pixel, stop_pixel = y0 * shape[2], y1 * shape[2]
        pull = _pull_spherical_chunk if cpu_pull is None else cpu_pull
    for z in range(z0, z1):
        local_z = z - first_z
        plane = block[local_z].reshape(-1)
        for first in range(first_pixel, stop_pixel, _PULL_CHUNK_VOXELS):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError('Spherical CPU block retired before publication')
            stop = min(stop_pixel, first + _PULL_CHUNK_VOXELS)
            plane[first:stop] = pull(
                source, view, radii, rotation, shape, z, first, stop, bboxes,
            )
    return block


def _select_spherical_cpu_pull(source, view, radii, rotation, shape, bboxes):
    """Opt into compiled bounded CPU pulls without changing the NumPy oracle."""
    if not spherical_cpu_compiled_requested() or source.dtype not in (np.uint8, np.bool_):
        return None
    try:
        from .spherical_projection_cpu import prepare_spherical_chunk_numba, SphericalCpuProjectionUnavailable
    except (ImportError, OSError) as exc:
        print(f'Spherical compiled CPU unavailable {view.name}: {exc}; using NumPy.', flush=True)
        return None
    try:
        return prepare_spherical_chunk_numba(source, view, radii, rotation, shape, bboxes)
    except SphericalCpuProjectionUnavailable as exc:
        print(f'Spherical compiled CPU unavailable {view.name}: {exc}; using NumPy.', flush=True)
        return None


def _project_spherical_encoded_block(source, view, radii, rotation, shape, first_z, count,
                                     bboxes=None, output_bounds=None, cancel_event=None,
                                     cpu_pull=None, *, packed=False):
    """Produce exact tight crops on the bounded CPU worker, before publication.

    The conservative bounds never replace the categorical pull's final tests.
    NumPy keeps large contiguous full-width strips; the compiled pull also skips
    X outside the bounds. Only the bounded X/Y crop is scanned by the encoder.
    """
    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError('Spherical CPU block retired before publication')

    check_cancelled()
    if output_bounds is None:
        output_bounds = spherical_output_bounds(view, shape, bboxes)
    z0, z1, y0, y1, x0, x1 = output_bounds.block(first_z, count)
    records, payloads = [], []
    offset = pull_voxels = scan_voxels = 0
    pull = _pull_spherical_chunk if cpu_pull is None else cpu_pull
    rectangle_pull = getattr(cpu_pull, 'rectangle', None)
    for z in range(first_z, first_z + count):
        check_cancelled()
        if not (z0 <= z < z1) or y0 == y1 or x0 == x1:
            records.append(RadialEncodedSlice(z, 0, 0, 0, 0, 0, offset, 0))
            continue
        rectangular = callable(rectangle_pull)
        plane = np.empty((y1 - y0, x1 - x0 if rectangular else shape[2]), dtype=np.uint8)
        flat = plane.reshape(-1)
        base, end = (0, flat.size) if rectangular else (y0 * shape[2], y1 * shape[2])
        for first in range(base, end, _PULL_CHUNK_VOXELS):
            check_cancelled()
            stop = min(end, first + _PULL_CHUNK_VOXELS)
            if rectangular:
                flat[first - base:stop - base] = rectangle_pull(
                    source, view, radii, rotation, shape, z, first, stop, bboxes,
                    bounds_yx=(y0, y1, x0, x1))
            else:
                flat[first - base:stop - base] = pull(
                    source, view, radii, rotation, shape, z, first, stop, bboxes)
        pull_voxels += end - base
        region = plane if rectangular else plane[:, x0:x1]
        scan_voxels += region.size
        check_cancelled()
        rows = np.flatnonzero(np.any(region, axis=1))
        if not rows.size:
            records.append(RadialEncodedSlice(z, 0, 0, 0, 0, 0, offset, 0))
            del plane, flat, region
            continue
        ry0, ry1 = int(rows[0]), int(rows[-1]) + 1
        columns = np.flatnonzero(np.any(region[ry0:ry1], axis=0))
        rx0, rx1 = int(columns[0]), int(columns[-1]) + 1
        crop = region[ry0:ry1, rx0:rx1]
        foreground = int(np.count_nonzero(crop))
        data = (np.packbits(crop, axis=1, bitorder='little').reshape(-1) if packed
                else np.array(crop, dtype=np.uint8, order='C', copy=True).reshape(-1))
        records.append(RadialEncodedSlice(z, y0 + ry0, y0 + ry1, x0 + rx0, x0 + rx1,
                                          foreground, offset, int(data.size)))
        payloads.append(data)
        offset += int(data.size)
        del plane, flat, region, crop
    check_cancelled()
    payload = (np.concatenate(payloads) if len(payloads) > 1 else payloads[0]
               if payloads else np.empty(0, dtype=np.uint8))
    return SphericalCpuEncodedBlock(first_z, tuple(records), payload, bool(packed),
                                    pull_voxels, scan_voxels)


def _spherical_block_schedule(depth, plane_bytes, workers, *, compact=False, compiled=False):
    block_depth = max(1, min(depth, _OUTPUT_BLOCK_BYTES // max(1, plane_bytes)))
    legacy_chunk_bytes = min(plane_bytes, _PULL_CHUNK_VOXELS) * _CHUNK_BYTES_PER_VOXEL
    legacy_worker_bytes = plane_bytes * block_depth + legacy_chunk_bytes
    legacy_worker_count = max(1, min(int(workers), _cpu_count(), math.ceil(depth / block_depth),
                                     max(1, _INFLIGHT_WORK_BYTES // max(1, legacy_worker_bytes))))
    if compact:
        block_depth = min(block_depth, _MAX_ENCODED_SLICES)
    chunk_bytes = min(plane_bytes, _PULL_CHUNK_VOXELS) * (
        _COMPILED_CHUNK_BYTES_PER_VOXEL if compact and compiled else _CHUNK_BYTES_PER_VOXEL)
    worker_bytes = plane_bytes * block_depth + chunk_bytes
    if compact:
        # A block can transiently retain individual crops, their concatenation,
        # one bounded projection plane, and its small Python wire records.
        worker_bytes += plane_bytes * (block_depth + 1) + block_depth * _CPU_ENCODED_SLICE_BYTES
    # Compiled scalar pulls retain no coordinate arrays. Use their actual bounded
    # strip workspace without increasing concurrency beyond the legacy schedule.
    worker_count = max(1, min(legacy_worker_count, math.ceil(depth / block_depth),
                              max(1, _INFLIGHT_WORK_BYTES // max(1, worker_bytes))))
    return block_depth, worker_count


def _ordered_spherical_blocks(project, depth, plane_bytes, workers, *, cancel_event=None,
                               compact=False, compiled=False):
    block_depth, worker_count = _spherical_block_schedule(
        depth, plane_bytes, workers, compact=compact, compiled=compiled)
    starts = iter(range(0, depth, block_depth))
    if worker_count == 1:
        for first in starts:
            yield first, project(first, min(block_depth, depth - first))
        return
    pending = deque()
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='spherical-project') as pool:
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
            if cancel_event is not None:
                cancel_event.set()
            for _, future in pending:
                future.cancel()
            # Executor shutdown joins running readers before the caller may
            # release the borrowed mask, even if a sink or producer failed.


def _validate_spherical_projection(source, view, out_shape_tyx, known_slice_bboxes):
    if str(view.family) != 'spherical':
        raise ValueError('Spherical shell projection requires a Spherical view')
    if source.ndim != 3 or min(source.shape) <= 0:
        raise ValueError('Spherical projection requires a nonempty three-dimensional mask')
    if source.shape[0] != int(view.num_slices):
        raise ValueError('Spherical mask depth differs from its shell trajectory')
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    if min(work) <= 0:
        raise ValueError('Spherical projection requires positive working dimensions')
    shape = tuple(int(n) for n in (work if out_shape_tyx is None else out_shape_tyx))
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Spherical projection requires positive source output dimensions')
    radii = np.asarray(view.spherical_radii, dtype=np.float64)
    minimum, maximum = float(view.spherical_min_radius), float(view.spherical_max_radius)
    if (radii.shape != (source.shape[0],) or not np.isfinite(radii).all()
            or not math.isfinite(minimum) or not math.isfinite(maximum)
            or minimum <= 0 or maximum < minimum
            or radii[0] != minimum or radii[-1] != maximum
            or np.any(np.diff(radii) <= 0) or np.any(np.diff(radii) > 1.0 + 1e-12)
            or maximum > (min(work) - 1) / 2.0):
        raise ValueError('Spherical trajectory does not match its bounded global radius grid')
    face, intervals = int(view.spherical_face), int(view.spherical_face_intervals)
    origin_u, origin_v = int(view.spherical_u_origin), int(view.spherical_v_origin)
    if (face != view.spherical_face or not 0 <= face < 6
            or intervals != view.spherical_face_intervals or intervals <= 0
            or origin_u != view.spherical_u_origin or origin_v != view.spherical_v_origin
            or not -int(view.src_w) < origin_u <= intervals
            or not -int(view.src_h) < origin_v <= intervals
            or int(view.spherical_patch_size) <= 0
            or int(view.src_h) != int(view.spherical_patch_size)
            or int(view.src_w) != int(view.spherical_patch_size)):
        raise ValueError('Spherical face patch does not match its native lattice')
    rotation = np.asarray(view.spherical_rotation_xyz, dtype=np.float64)
    if rotation.shape == (9,):
        rotation = rotation.reshape(3, 3)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-12, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-12, rtol=0)):
        raise ValueError('Spherical cube rotation must be a proper orthogonal XYZ matrix')
    bboxes = None
    if known_slice_bboxes is not None:
        raw_boxes = np.asarray(known_slice_bboxes)
        if raw_boxes.dtype.kind not in 'iu':
            raise ValueError('Spherical source bounding boxes must be integer pixel bounds')
        bboxes = np.ascontiguousarray(raw_boxes, dtype=np.int64)
        if (bboxes.shape != (source.shape[0], 4) or np.any(bboxes < 0)
                or np.any(bboxes[:, 0] > bboxes[:, 1]) or np.any(bboxes[:, 2] > bboxes[:, 3])
                or np.any(bboxes[:, 1] > source.shape[1]) or np.any(bboxes[:, 3] > source.shape[2])):
            raise ValueError('Spherical source bounding boxes do not match the processing mask grid')
    return radii, rotation, shape, bboxes


def backproject_spherical_volume_to_volume(
    spherical_mask_mm: np.ndarray,
    spherical_view: 'ViewInfo',
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
    """Pull one QSC face-patch trajectory into caller-owned source geometry.

    Dense output is path-backed, matching the other nonlinear projectors.
    Sink-only output allocates no complete output volume and publishes bounded
    blocks in source-z order. Input masks are borrowed and never modified.
    The nearest radius is global, with exact midpoints assigned inward; row
    and column nearest ties use NumPy's round-to-even on the global lattice.
    """
    from .backprojection import (
        SinkOnlyProjectionResult, _emit_projection_block_callback,
        _abort_projection_block_callback,
        _cancel_main_process_spherical_retirement_request,
    )

    source = np.asarray(spherical_mask_mm)
    radii, rotation, shape, bboxes = _validate_spherical_projection(
        source, spherical_view, out_shape_tyx, known_slice_bboxes,
    )
    cpu_bounds = spherical_output_bounds(spherical_view, shape, bboxes)
    if sink_only and projection_block_callback is None:
        raise ValueError('Spherical sink-only projection requires a block consumer')
    stage = None
    output = None
    failed = False
    callback_aborted = False
    started = time.perf_counter()
    cpu_cancel = threading.Event()
    metrics_lock = threading.Lock()
    cpu_worker_s = 0.0
    cpu_cancelled_blocks = 0
    cpu_result_wait_s = cpu_publish_s = gpu_result_wait_s = gpu_publish_s = 0.0
    admission_probe_s = cpu_reader_drain_s = 0.0
    admission_attempts = 0
    cpu_compact_slices = cpu_empty_slices = cpu_payload_bytes = 0
    cpu_compact_pull_voxels = cpu_compact_scan_voxels = 0

    def try_stage(*, quiet=False):
        nonlocal admission_attempts, admission_probe_s
        admission_attempts += 1
        probe_started = time.perf_counter()
        try:
            return _try_spherical_cuda_stage(source, spherical_view, shape, bboxes, quiet=quiet)
        finally:
            admission_probe_s += time.perf_counter() - probe_started

    try:
        stage = try_stage()
        cpu_setup_started = time.perf_counter()
        cpu_pull = (_select_spherical_cpu_pull(source, spherical_view, radii, rotation, shape, bboxes)
                    if stage is None else None)
        cpu_setup_seconds = time.perf_counter() - cpu_setup_started
        cpu_backend = ('unused' if stage is not None else
                       ('numba_f64_bounded' if cpu_pull is not None else 'numpy_bounded'))
        encoded_format = getattr(projection_block_callback, 'encoded_slice_format', None)
        compact_supported = bool(sink_only and encoded_format in ('raw_u8', 'packbits_little')
                                 and callable(getattr(projection_block_callback, 'consume_encoded_block', None)))
        cpu_compact = bool(compact_supported and spherical_cpu_compact_enabled())
        cpu_rectangular = bool(cpu_compact and callable(getattr(cpu_pull, 'rectangle', None)))
        compact = bool(compact_supported and (stage is not None or cpu_compact))
        packed = encoded_format == 'packbits_little'
        block_depth, actual_workers = _spherical_block_schedule(
            shape[0], shape[1] * shape[2], workers, compact=cpu_compact, compiled=cpu_rectangular)
        if stage is not None:
            block_depth, actual_workers = stage.max_block_depth, 1
        backend = f'cpu_{cpu_backend}' if stage is None else ('cuda_direct_qsc_compact' if compact else 'cuda_direct_qsc')
        runtime_telemetry().gauge('projection.spherical.backend', backend)
        runtime_telemetry().gauge('projection.spherical.cpu_backend', cpu_backend)
        runtime_telemetry().gauge('projection.spherical.cpu_setup_seconds', cpu_setup_seconds)
        runtime_telemetry().gauge('projection.spherical.workers', actual_workers)
        runtime_telemetry().gauge('projection.spherical.block_depth', block_depth)
        print(f'Spherical projection start {spherical_view.name}: backend={backend}, '
              f'workers={actual_workers}, block_z={block_depth}, source={source.shape}, output={shape}'
              f', cpu_compact_enabled={int(cpu_compact)}'
              + (f', device=cuda:{stage.device_index}, payload={encoded_format if compact else "dense"}'
                 if stage is not None else ''), flush=True)
        if not sink_only:
            output = allocate_workspace_array(
                shape=shape, dtype=np.uint8, path=Path(out_path), desc=f'{desc} workspace',
                prefer_memory=False, prefer_memfd=False, reserve_bytes=int(reserve_bytes), initialize_zero=False,
            )

        def project(first, count):
            nonlocal cpu_worker_s, cpu_cancelled_blocks
            work_started = time.perf_counter()
            cancelled = False
            try:
                project_cpu = _project_spherical_encoded_block if cpu_compact else _project_spherical_block
                return project_cpu(
                    source, spherical_view, radii, rotation, shape, first, count,
                    bboxes, cpu_bounds, cancel_event=cpu_cancel, cpu_pull=cpu_pull,
                    **({'packed': packed} if cpu_compact else {}),
                )
            except CancelledError:
                cancelled = True
                raise
            finally:
                elapsed = time.perf_counter() - work_started
                with metrics_lock:
                    cpu_worker_s += elapsed
                    cpu_cancelled_blocks += int(cancelled)

        next_z = cpu_slices = cuda_slices = 0
        recheck_at = max(1, int(_CUDA_RECHECK_SLICES))
        recheck_time = time.monotonic() + _CUDA_RECHECK_SECONDS
        while next_z < shape[0]:
            cpu_blocks = stage is None
            blocks = (_ordered_spherical_cuda_blocks(stage, shape[0], packed if compact else None, first_z=next_z)
                      if stage is not None else _ordered_spherical_blocks(
                          project, shape[0], shape[1] * shape[2], workers,
                          cancel_event=cpu_cancel, compact=cpu_compact, compiled=cpu_rectangular))
            try:
                while True:
                    wait_started = time.perf_counter()
                    try:
                        z, block = next(blocks)
                    except StopIteration:
                        break
                    if cpu_blocks:
                        cpu_result_wait_s += time.perf_counter() - wait_started
                    else:
                        gpu_result_wait_s += time.perf_counter() - wait_started
                    if z != next_z:
                        raise RuntimeError('Spherical projection duplicated or skipped an output slice')
                    count = len(block.records) if compact else len(block)
                    if not compact and (count <= 0 or z + count > shape[0]):
                        raise RuntimeError('Spherical projection returned an invalid output block size')
                    publish_started = time.perf_counter()
                    if output is not None:
                        output[z:z + count] = block
                    if compact:
                        expected_depth = block_depth if cpu_blocks else stage.max_block_depth
                        if (block.first_z != z or bool(block.packed) != packed
                                or count != min(expected_depth, shape[0] - z)):
                            raise RuntimeError('Spherical encoded block identity/format/count differs from its request')
                        empty_consumer = getattr(projection_block_callback, 'consume_empty_range', None)
                        if cpu_blocks and block.payload.size == 0 and callable(empty_consumer):
                            try:
                                canonical = (block.payload.dtype == np.uint8 and block.payload.ndim == 1
                                    and block.payload.flags.c_contiguous
                                    and all(operator.index(record.z) == z + index
                                        and all(operator.index(getattr(record, field)) == 0
                                            for field in ('y0', 'y1', 'x0', 'x1', 'foreground', 'offset', 'size'))
                                        for index, record in enumerate(block.records)))
                            except (AttributeError, TypeError, ValueError, OverflowError):
                                canonical = False
                            if not canonical:
                                raise RuntimeError('Spherical CPU empty block metadata is not canonical')
                            empty_consumer(z, count)
                        else:
                            projection_block_callback.consume_encoded_block(
                                z, block.records, block.payload, packed=packed)
                        if cpu_blocks:
                            cpu_compact_slices += count
                            cpu_empty_slices += sum(record.foreground == 0 for record in block.records)
                            cpu_payload_bytes += int(block.payload.size)
                            cpu_compact_pull_voxels += block.pull_voxels
                            cpu_compact_scan_voxels += block.scan_voxels
                    else:
                        try:
                            _emit_projection_block_callback(
                                projection_block_callback, z, block, desc=desc, required=bool(sink_only),
                            )
                        except Exception:
                            callback_aborted = True  # The compatibility helper already aborted it.
                            raise
                    if cpu_blocks:
                        cpu_publish_s += time.perf_counter() - publish_started
                    else:
                        gpu_publish_s += time.perf_counter() - publish_started
                    next_z += count
                    if stage is None:
                        cpu_slices += count
                    else:
                        cuda_slices += count
                    del block
                    if (stage is None and next_z < shape[0]
                            and (next_z >= recheck_at or time.monotonic() >= recheck_time)
                            and spherical_cuda_backproject_enabled()):
                        recheck_at = next_z + max(1, int(_CUDA_RECHECK_SLICES))
                        recheck_time = time.monotonic() + _CUDA_RECHECK_SECONDS
                        candidate = try_stage(quiet=True)
                        if candidate is not None:
                            stage = candidate
                            compact = compact_supported
                            backend = 'cuda_direct_qsc_compact' if compact else 'cuda_direct_qsc'
                            runtime_telemetry().gauge('projection.spherical.backend', backend)
                            runtime_telemetry().gauge('projection.spherical.workers', 1)
                            runtime_telemetry().gauge('projection.spherical.block_depth', stage.max_block_depth)
                            print(f'Spherical projection promoted {spherical_view.name}: '
                                  f'CPU completed z=[0,{next_z}); continuing on cuda:{stage.device_index} '
                                  f'with backend={backend}', flush=True)
                            break
            finally:
                # Before switching backends, settle all unconsumed CPU futures.
                # Their output was never published; GPU resumes at next_z only.
                # Cooperative cancellation bounds each running reader's remaining
                # work to its current pull chunk; joining still owns source lifetime.
                drain_started = time.perf_counter()
                if cpu_blocks:
                    cpu_cancel.set()
                blocks.close()
                if cpu_blocks:
                    cpu_reader_drain_s += time.perf_counter() - drain_started
        if output is not None:
            flush = getattr(output, 'flush', None)
            if callable(flush):
                flush()
        if stage is not None:
            for name in ('kernel_seconds', 'metadata_seconds', 'pack_seconds', 'd2h_seconds',
                         'metadata_d2h_bytes', 'payload_d2h_bytes', 'dense_d2h_bytes', 'source_h2d_bytes',
                         'roi_projection_voxels', 'roi_skipped_blocks', 'constructor_seconds', 'preflight_seconds',
                         'preflight_mode', 'preflight_pixels', 'source_upload_seconds',
                         'source_pack_seconds', 'source_pack_backend', 'geometry_upload_seconds',
                         'source_upload_pipeline', 'source_upload_stage_bytes', 'source_upload_copy_count',
                         'source_upload_stream_fences', 'source_upload_lane_wait_seconds'):
                runtime_telemetry().gauge(f'projection.spherical.{name}', getattr(stage.projector, name, None))
        metrics = '' if stage is None else ''.join(
            f', {name}={getattr(stage.projector, name, None)}' for name in
            ('source_h2d_bytes', 'metadata_d2h_bytes', 'payload_d2h_bytes', 'dense_d2h_bytes',
             'roi_projection_voxels', 'roi_skipped_blocks', 'constructor_seconds', 'preflight_seconds',
             'preflight_mode', 'preflight_pixels', 'source_upload_seconds',
             'source_pack_seconds', 'source_pack_backend', 'geometry_upload_seconds',
             'source_upload_pipeline', 'source_upload_stage_bytes', 'source_upload_copy_count',
             'source_upload_stream_fences', 'source_upload_lane_wait_seconds',
             'kernel_seconds', 'metadata_seconds', 'pack_seconds', 'd2h_seconds'))
        print(f'Spherical projection complete {spherical_view.name}: backend={backend}, '
              f'total_s={time.perf_counter() - started:.6f}, cpu_slices={cpu_slices}, '
              f'cuda_slices={cuda_slices}, admission_attempts={admission_attempts}, '
              f'admission_probe_s={admission_probe_s:.6f}, cpu_worker_s={cpu_worker_s:.6f}, '
              f'cpu_result_wait_s={cpu_result_wait_s:.6f}, cpu_publish_s={cpu_publish_s:.6f}, '
              f'cpu_reader_drain_s={cpu_reader_drain_s:.6f}, cpu_cancelled_blocks={cpu_cancelled_blocks}, '
              f'gpu_result_wait_s={gpu_result_wait_s:.6f}, gpu_publish_s={gpu_publish_s:.6f}'
              f', cpu_backend={cpu_backend}, cpu_setup_seconds={cpu_setup_seconds:.6f}'
              f', cpu_compact_slices={cpu_compact_slices}, cpu_empty_slices={cpu_empty_slices}'
              f', cpu_payload_bytes={cpu_payload_bytes}, cpu_compact_pull_voxels={cpu_compact_pull_voxels}'
              f', cpu_compact_scan_voxels={cpu_compact_scan_voxels}'
              f'{metrics}', flush=True)
        return output if output is not None else SinkOnlyProjectionResult(shape)
    except BaseException as exc:
        failed = True
        if sink_only and not callback_aborted:
            _abort_projection_block_callback(projection_block_callback, exc)
        close_memmap_array_without_flush(output)
        raise
    finally:
        _cancel_main_process_spherical_retirement_request(f'Spherical source projection {spherical_view.name}')
        if stage is not None:
            try:
                stage.close()
            except SphericalCudaProjectionUnsafeFailure:
                raise
            except BaseException as exc:
                if not failed:
                    close_memmap_array_without_flush(output)
                    raise
                print(f'Spherical CUDA cleanup failed {spherical_view.name}: {exc}', flush=True)
