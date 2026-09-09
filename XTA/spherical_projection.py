"""Bounded source-coordinate pull projection for spherical QSC shell patches.

All radii share one face lattice. Radius and face-pixel selection happen in
global coordinates before patch offsets, so overlapping patches agree on
nearest-neighbor ties. Faces are closed: their incident edges and corners may
contribute independently to the caller's ordinary OR union.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

from .qsc import qsc_forward_face
from .spherical_projection_cuda import SphericalCudaProjectionUnsafeFailure
from .runtime import allocate_workspace_array, close_memmap_array_without_flush, runtime_telemetry
from .workspace import _cpu_count

if TYPE_CHECKING:
    from .geometry import ViewInfo


_PULL_CHUNK_VOXELS = 128 * 1024
_OUTPUT_BLOCK_BYTES = 8 * 1024 * 1024
_INFLIGHT_WORK_BYTES = 256 * 1024 * 1024
_CHUNK_BYTES_PER_VOXEL = 384
_CUDA_RECHECK_SLICES = 8


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
    from .backprojection import _try_acquire_main_process_gpu_stage
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


def _project_spherical_block(source, view, radii, rotation, shape, first_z, count, bboxes=None):
    block = np.empty((count, shape[1], shape[2]), dtype=np.uint8)
    for local_z in range(count):
        plane = block[local_z].reshape(-1)
        for first in range(0, plane.size, _PULL_CHUNK_VOXELS):
            stop = min(plane.size, first + _PULL_CHUNK_VOXELS)
            plane[first:stop] = _pull_spherical_chunk(
                source, view, radii, rotation, shape, first_z + local_z, first, stop, bboxes,
            )
    return block


def _spherical_block_schedule(depth, plane_bytes, workers):
    block_depth = max(1, min(depth, _OUTPUT_BLOCK_BYTES // max(1, plane_bytes)))
    chunk_bytes = min(plane_bytes, _PULL_CHUNK_VOXELS) * _CHUNK_BYTES_PER_VOXEL
    worker_bytes = plane_bytes * block_depth + chunk_bytes
    worker_count = max(1, min(int(workers), _cpu_count(), math.ceil(depth / block_depth),
                              max(1, _INFLIGHT_WORK_BYTES // max(1, worker_bytes))))
    return block_depth, worker_count


def _ordered_spherical_blocks(project, depth, plane_bytes, workers):
    block_depth, worker_count = _spherical_block_schedule(depth, plane_bytes, workers)
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
    from .backprojection import SinkOnlyProjectionResult, _emit_projection_block_callback

    source = np.asarray(spherical_mask_mm)
    radii, rotation, shape, bboxes = _validate_spherical_projection(
        source, spherical_view, out_shape_tyx, known_slice_bboxes,
    )
    if sink_only and projection_block_callback is None:
        raise ValueError('Spherical sink-only projection requires a block consumer')
    stage = None
    output = None
    failed = False
    started = time.perf_counter()
    try:
        stage = _try_spherical_cuda_stage(source, spherical_view, shape, bboxes)
        block_depth, actual_workers = _spherical_block_schedule(shape[0], shape[1] * shape[2], workers)
        if stage is not None:
            block_depth, actual_workers = stage.max_block_depth, 1
        encoded_format = getattr(projection_block_callback, 'encoded_slice_format', None)
        compact_supported = bool(sink_only and encoded_format in ('raw_u8', 'packbits_little')
                                 and callable(getattr(projection_block_callback, 'consume_encoded_block', None)))
        compact = bool(stage is not None and compact_supported)
        packed = encoded_format == 'packbits_little'
        backend = 'cpu_numpy_bounded' if stage is None else ('cuda_direct_qsc_compact' if compact else 'cuda_direct_qsc')
        runtime_telemetry().gauge('projection.spherical.backend', backend)
        runtime_telemetry().gauge('projection.spherical.workers', actual_workers)
        runtime_telemetry().gauge('projection.spherical.block_depth', block_depth)
        print(f'Spherical projection start {spherical_view.name}: backend={backend}, '
              f'workers={actual_workers}, block_z={block_depth}, source={source.shape}, output={shape}'
              + (f', device=cuda:{stage.device_index}, payload={encoded_format if compact else "dense"}'
                 if stage is not None else ''), flush=True)
        if not sink_only:
            output = allocate_workspace_array(
                shape=shape, dtype=np.uint8, path=Path(out_path), desc=f'{desc} workspace',
                prefer_memory=False, prefer_memfd=False, reserve_bytes=int(reserve_bytes), initialize_zero=False,
            )

        def project(first, count):
            return _project_spherical_block(source, spherical_view, radii, rotation, shape, first, count, bboxes)

        next_z = cpu_slices = cuda_slices = 0
        recheck_at = max(1, int(_CUDA_RECHECK_SLICES))
        while next_z < shape[0]:
            blocks = (_ordered_spherical_cuda_blocks(stage, shape[0], packed if compact else None, first_z=next_z)
                      if stage is not None else _ordered_spherical_blocks(project, shape[0], shape[1] * shape[2], workers))
            try:
                for z, block in blocks:
                    if z != next_z:
                        raise RuntimeError('Spherical projection duplicated or skipped an output slice')
                    count = len(block.records) if compact else len(block)
                    if not compact and (count <= 0 or z + count > shape[0]):
                        raise RuntimeError('Spherical projection returned an invalid output block size')
                    if output is not None:
                        output[z:z + count] = block
                    if compact:
                        if (block.first_z != z or bool(block.packed) != packed
                                or count != min(stage.max_block_depth, shape[0] - z)):
                            raise RuntimeError('Spherical CUDA encoded block identity/format/count differs from its request')
                        projection_block_callback.consume_encoded_block(z, block.records, block.payload, packed=packed)
                    else:
                        _emit_projection_block_callback(
                            projection_block_callback, z, block, desc=desc, required=bool(sink_only),
                        )
                    next_z += count
                    if stage is None:
                        cpu_slices += count
                    else:
                        cuda_slices += count
                    del block
                    if (stage is None and next_z < shape[0] and next_z >= recheck_at
                            and spherical_cuda_backproject_enabled()):
                        recheck_at = next_z + max(1, int(_CUDA_RECHECK_SLICES))
                        candidate = _try_spherical_cuda_stage(source, spherical_view, shape, bboxes, quiet=True)
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
                blocks.close()
        if output is not None:
            flush = getattr(output, 'flush', None)
            if callable(flush):
                flush()
        if stage is not None:
            for name in ('kernel_seconds', 'metadata_seconds', 'pack_seconds', 'd2h_seconds',
                         'metadata_d2h_bytes', 'payload_d2h_bytes', 'dense_d2h_bytes', 'source_h2d_bytes'):
                runtime_telemetry().gauge(f'projection.spherical.{name}', getattr(stage.projector, name, 0))
        metrics = '' if stage is None else ''.join(
            f', {name}={getattr(stage.projector, name, 0)}' for name in
            ('source_h2d_bytes', 'metadata_d2h_bytes', 'payload_d2h_bytes', 'dense_d2h_bytes'))
        print(f'Spherical projection complete {spherical_view.name}: backend={backend}, '
              f'total_s={time.perf_counter() - started:.6f}, cpu_slices={cpu_slices}, '
              f'cuda_slices={cuda_slices}{metrics}', flush=True)
        return output if output is not None else SinkOnlyProjectionResult(shape)
    except BaseException:
        failed = True
        close_memmap_array_without_flush(output)
        raise
    finally:
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
