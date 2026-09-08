"""Native shell chunks -> exact source bitset on their persistent inference owner.

The D1 result envelope provides scheduling/coverage/publication ownership only.
Its old geometry and proto cleanup are deliberately not used for these masks.
"""
from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

from ._deps import _numba
from .cylindrical_cuda_projection import RadialCudaProjectionUnsafeFailure, _CROP_METADATA_DTYPE

RADIAL_OWNER_CONTRACT = 'radial_native_pull_v1'
_WORK_ITEMS = 64 * 1024 * 1024
_RESERVE_BYTES = 2 * 1024**3
_RADIAL_OWNER_STATES = {}
_RADIAL_OWNER_LOCK = threading.RLock()


class DeviceOnlyRadialTarget:
    """Shape metadata for the generic predictor; host access is a contract error."""
    def __init__(self, shape):
        self.shape = tuple(map(int, shape))
        self.dtype = np.dtype(np.uint8)
        self._received = np.zeros(self.shape[0], bool)
        self._lock = threading.Lock()

    def claim_result(self, index):
        with self._lock:
            if index < 0 or index >= len(self._received):
                raise IndexError('Native Radial prediction is outside its radius lease')
            if self._received[index]:
                raise RuntimeError('Native Radial prediction duplicated a radius result')
            self._received[index] = True

    def require_complete(self):
        with self._lock:
            if not self._received.all():
                raise RuntimeError('Native Radial prediction ended without every radius result')

    def __array__(self, *args, **kwargs):
        raise RuntimeError('Native Radial owner forbids materializing a host task mask')

    def __getitem__(self, key):
        raise RuntimeError('Native Radial owner received a host-mask fallback')

    def __setitem__(self, key, value):
        raise RuntimeError('Native Radial owner received a host-mask fallback')


def radial_owner_enabled():
    return os.environ.get('YOLO_TTA_RADIAL_OWNER', '1').strip().lower() not in ('', '0', 'false', 'off', 'no')


def radial_runtime_provenance():
    from . import outputs
    modules = {}
    for name in ('pipeline', 'workers', 'inference', 'cuda_d1', 'cylindrical_owner', 'outputs'):
        module = sys.modules.get('XTA.' + name)
        raw_path = getattr(module, '__file__', None)
        if raw_path:
            path = Path(raw_path).resolve()
            try:
                modules[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            except OSError as exc:
                modules[name] = {'path': str(path), 'sha256': None, 'read_error': type(exc).__name__}
    sink = outputs.nrrd_layer_sink()
    return {'radial_owner_requested': radial_owner_enabled(),
        'nrrd_sink_workers_effective': outputs.nrrd_layer_sink_workers(),
        'nrrd_sink_workers_constructed': getattr(sink, 'max_workers', None),
        'nrrd_gzip_workers_effective': outputs.nrrd_gzip_workers(),
        'nrrd_fill_workers_effective': outputs.nrrd_fill_workers(), 'modules': modules}


def radial_owner_eligible(view, *, d1_active, cpu_workers, kind, interpolation, tiled,
                          angle_count, min_conf, min_radius, batch, gray, retain_native=False):
    return bool(radial_owner_enabled() and d1_active and not cpu_workers and kind == 'fullframe'
        and view.family == 'radial' and int(interpolation) == 0 and not tiled and angle_count == 1
        and float(view.tta_angle_deg) == 0. and float(min_conf) == 0. and float(min_radius) == 0.
        and int(batch) == 1 and gray and not retain_native)


def is_radial_owner_task(task):
    return task.get('projection_contract') == RADIAL_OWNER_CONTRACT


def _bucket_shell_pixels(shells, count):
    offsets = np.zeros(count + 1, np.int64)
    for shell in shells:
        if shell >= 0:
            offsets[shell + 1] += 1
    for i in range(count):
        offsets[i + 1] += offsets[i]
    pixels = np.empty(offsets[-1], np.int32)
    cursor = offsets[:-1].copy()
    for pixel in range(len(shells)):
        shell = shells[pixel]
        if shell >= 0:
            pixels[cursor[shell]] = pixel
            cursor[shell] += 1
    return offsets, pixels


_bucket_shell_pixels_compiled = _numba.njit(cache=True, nogil=True)(_bucket_shell_pixels) if _numba else None


def _bucket_shell_pixels_numpy(shells, count):
    valid = np.flatnonzero(shells >= 0)
    values = shells[valid]
    offsets = np.zeros(count + 1, np.int64)
    np.cumsum(np.bincount(values, minlength=count), out=offsets[1:])
    return offsets, valid[np.argsort(values, kind='stable')].astype(np.int32)


_OWNER_KERNEL_SOURCE = r'''
__device__ int bg_root(int* parents, int node) {
    int next = atomicAdd(parents + node, 0);
    while (next != node) {
        int grand = atomicAdd(parents + next, 0);
        atomicMin(parents + node, grand);
        node = next;
        next = grand;
    }
    return node;
}
__device__ void bg_join(int* parents, int a, int b) {
    while (true) {
        a = bg_root(parents, a); b = bg_root(parents, b);
        if (a == b) return;
        int lo = min(a, b), hi = max(a, b);
        if (atomicCAS(parents + hi, hi, lo) == hi) return;
    }
}
extern "C" __global__ void init_background(
    const unsigned char* mask, int* parents, int* exterior,
    unsigned long long start, int stride, int width, int pixels) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= pixels) return;
    parents[q] = mask[start + (unsigned long long)(q / width) * stride + q % width] ? -1 : q;
    exterior[q] = 0;
}
extern "C" __global__ void join_background(int* parents, int width, int pixels) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= pixels || parents[q] < 0) return;
    if (q % width && parents[q - 1] >= 0) bg_join(parents, q, q - 1);
    if (q >= width && parents[q - width] >= 0) bg_join(parents, q, q - width);
}
extern "C" __global__ void mark_exterior(int* parents, int* exterior, int width, int height) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= width * height || parents[q] < 0) return;
    int x = q % width, y = q / width;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
        atomicExch(exterior + bg_root(parents, q), 1);
}
extern "C" __global__ void fill_enclosed(
    unsigned char* mask, int* parents, const int* exterior,
    unsigned long long start, int stride, int width, int pixels) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= pixels || parents[q] < 0) return;
    if (!exterior[bg_root(parents, q)])
        mask[start + (unsigned long long)(q / width) * stride + q % width] = 1;
}
extern "C" __global__ void gather_shell_chunk(
    const unsigned char* masks, const int* pixels, const int* shells,
    const unsigned int* offsets, const int* columns,
    const double* sampled, const int* rows, const int* mapped_columns,
    const double* centers, const double* ideal, const long long* boxes,
    unsigned int* words, unsigned long long bucket_first, unsigned long long pixel_count,
    unsigned long long work_first, unsigned long long work_stop,
    int shell_start, int mask_h, int mask_w, int native_h, int native_w,
    int stack_length, int height_origin, int base, int vertical, int plane_w, int out_h, int out_w) {
    unsigned long long q = work_first + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= work_stop) return;
    int stack_index = (int)(q / pixel_count);
    int p = pixels[bucket_first + q % pixel_count];
    int shell = shells[p], local = shell - shell_start;
    unsigned long long box = (unsigned long long)local * 4;
    if (boxes[box + 1] <= boxes[box] || boxes[box + 3] <= boxes[box + 2]) return;
    int v = p / plane_w, u = p % plane_w;
    double stack = centers[stack_index];
    double h = __dsub_rn(stack, ideal[vertical ? v : u]);
    if (h < 0.0 || h > (double)(stack_length - 1)) return;
    for (unsigned long long at = offsets[p]; at < (unsigned long long)offsets[p + 1]; ++at) {
        int col = columns[at];
        h = __dsub_rn(stack, sampled[(unsigned long long)shell * native_w + col]);
        h = fmin(fmax(h, 0.0), (double)(stack_length - 1));
        long long row = __double2ll_rn(h) - (long long)height_origin;
        if (row < 0 || row >= native_h) continue;
        int pr = rows[row], pc = mapped_columns[col];
        if (pr < boxes[box] || pr >= boxes[box + 1] || pc < boxes[box + 2] || pc >= boxes[box + 3]) continue;
        if (!masks[((unsigned long long)local * mask_h + pr) * mask_w + pc]) continue;
        unsigned long long flat;
        if (base == 0) flat = (unsigned long long)stack_index * out_h * out_w + p;
        else if (base == 1) flat = ((unsigned long long)v * out_h + stack_index) * out_w + u;
        else flat = ((unsigned long long)v * out_h + u) * out_w + stack_index;
        atomicOr(words + (flat >> 5), 1u << (flat & 31));
        return;
    }
}
'''


class RadialOwner:
    """One worker's source bitset and private bounded cleanup/projection resources."""
    def __init__(self, view, mask_shape, output_shape, *, device_index=0, reserve_bytes=_RESERVE_BYTES):
        from . import cylindrical_projection as reference
        from .geometry import radial_global_radii
        import cupy as cp
        self.view, self.mask_shape, self.output_shape = view, tuple(mask_shape), tuple(output_shape)
        if view.family != 'radial' or len(self.mask_shape) != 2 or len(self.output_shape) != 3:
            raise ValueError('Invalid Radial owner geometry')
        if min(*self.mask_shape, *self.output_shape) <= 0:
            raise ValueError('Radial owner dimensions must be positive')
        if max(*self.mask_shape, *self.output_shape) > np.iinfo(np.int32).max or math.prod(self.mask_shape) > np.iinfo(np.int32).max:
            raise ValueError('Radial owner dimensions exceed int32 kernel addressing')
        self.coverage = np.zeros(view.num_slices, bool)
        self.cp, self.device_index = cp, int(device_index)
        self._stream = self._pool = None
        self._arrays = {}
        self.words = self._parents = self._exterior = self._borrowed = self._borrowed_boxes = None
        self._module = self._meta_module = None
        self._closed = self._failed = False
        self.native_any = False
        self.cleanup_seconds = self.projection_seconds = self.upload_seconds = 0.
        self.created_at = time.perf_counter()
        radii = np.asarray(radial_global_radii(view))
        start = int(view.radial_shell_start)
        if (radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all()
                or np.any(radii < 0) or np.any(np.diff(radii) <= 0)
                or start < 0 or start + view.num_slices > len(radii)
                or not np.array_equal(np.asarray(view.radial_radii), radii[start:start + view.num_slices])
                or (view.radial_tilted_source and view.tilt_direction not in ('vertical', 'horizontal'))):
            raise ValueError('Radial owner trajectory does not match its exact global geometry')
        self.plan, _ = reference._radial_plane_plan(view, radii, self.output_shape)
        self.metadata = reference._radial_projection_metadata(view,
            (view.num_slices, *self.mask_shape), self.output_shape, self.plan)
        if _bucket_shell_pixels_compiled is not None:
            try:
                self.bucket_offsets, pixels = _bucket_shell_pixels_compiled(self.plan.shell_index, view.num_slices)
            except Exception:
                self.bucket_offsets, pixels = _bucket_shell_pixels_numpy(self.plan.shell_index, view.num_slices)
        else:
            self.bucket_offsets, pixels = _bucket_shell_pixels_numpy(self.plan.shell_index, view.num_slices)
        centers, ideal, sampled, rows, mapped_columns, self.stack_length, self.vertical = self.metadata
        arrays = dict(pixels=pixels, shells=self.plan.shell_index, offsets=self.plan.column_offsets,
            columns=self.plan.native_columns, sampled=sampled, rows=rows, mapped_columns=mapped_columns,
            centers=centers, ideal=ideal)
        word_count = (math.prod(self.output_shape) + 31) // 32
        pixels_per_mask = math.prod(self.mask_shape)
        self.required_bytes = word_count * 4 + pixels_per_mask * 8 + sum(a.nbytes for a in arrays.values()) + 256 * 1024**2
        try:
            with cp.cuda.Device(self.device_index):
                self._stream = cp.cuda.Stream(non_blocking=True)
                self._pool = cp.cuda.MemoryPool()
                free, _ = cp.cuda.runtime.memGetInfo()
                if free < self.required_bytes + reserve_bytes:
                    raise RuntimeError(f'Radial owner requires {self.required_bytes / 2**30:.2f} GiB plus '
                        f'{reserve_bytes / 2**30:.2f} GiB reserve; {free / 2**30:.2f} GiB free. '
                        'YOLO_TTA_RADIAL_OWNER=0 selects parent projection.')
                with cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    self._module = cp.RawModule(code=_OWNER_KERNEL_SOURCE, options=('--std=c++11', '--fmad=false'))
                    self._kernels = {name: self._module.get_function(name) for name in
                        ('init_background', 'join_background', 'mark_exterior', 'fill_enclosed', 'gather_shell_chunk')}
                    from .cylindrical_cuda_projection import _KERNEL_SOURCE
                    self._meta_module = cp.RawModule(code=_KERNEL_SOURCE, options=('--std=c++11', '--fmad=false'))
                    self._reset_meta = self._meta_module.get_function('reset_radial_crop_metadata')
                    self._reduce_meta = self._meta_module.get_function('reduce_radial_crop_metadata')
                    self.words = cp.zeros(word_count, cp.uint32)
                    self._parents = cp.empty(pixels_per_mask, cp.int32)
                    self._exterior = cp.empty(pixels_per_mask, cp.int32)
                    started = time.perf_counter()
                    for name, array in arrays.items():
                        self._arrays[name] = cp.asarray(array)
                    self._stream.synchronize()
                    self.upload_seconds = time.perf_counter() - started
        except RadialCudaProjectionUnsafeFailure:
            raise
        except BaseException:
            self.close()
            raise

    def _fence(self):
        try:
            self._stream.synchronize()
        except BaseException as exc:
            raise RadialCudaProjectionUnsafeFailure('Could not fence native Radial owner', self) from exc

    def _boxes(self, masks):
        cp = self.cp
        count, h, w = masks.shape
        meta = cp.empty(count * _CROP_METADATA_DTYPE.itemsize, cp.uint8)
        self._reset_meta(((count + 255) // 256,), (256,),
            (meta, np.int32(count), np.int32(h), np.int32(w)), stream=self._stream)
        self._reduce_meta(((w + 31) // 32, (h + 7) // 8, count), (32, 8),
            (masks, meta, np.int32(count), np.int32(h), np.int32(w)), stream=self._stream)
        host = meta.get(stream=self._stream).view(_CROP_METADATA_DTYPE)
        boxes = np.zeros((count, 4), np.int64)
        for i, row in enumerate(host):
            if row['foreground']:
                boxes[i] = tuple(row[n] for n in ('y0', 'y1', 'x0', 'x1'))
        return boxes

    def _clean(self, masks, boxes):
        h, w = self.mask_shape
        for i, (y0, y1, x0, x1) in enumerate(boxes):
            height, width = int(y1 - y0), int(x1 - x0)
            if height < 3 or width < 3:
                continue
            count = height * width
            grid = ((count + 255) // 256,)
            start = np.uint64((i * h + int(y0)) * w + int(x0))
            self._kernels['init_background'](grid, (256,), (masks, self._parents, self._exterior,
                start, np.int32(w), np.int32(width), np.int32(count)), stream=self._stream)
            self._kernels['join_background'](grid, (256,),
                (self._parents, np.int32(width), np.int32(count)), stream=self._stream)
            self._kernels['mark_exterior'](grid, (256,),
                (self._parents, self._exterior, np.int32(width), np.int32(height)), stream=self._stream)
            self._kernels['fill_enclosed'](grid, (256,), (masks, self._parents, self._exterior,
                start, np.int32(w), np.int32(width), np.int32(count)), stream=self._stream)

    def consume(self, first, masks, *, clean=True):
        first = int(first)
        if self._closed or self._failed:
            raise RuntimeError('Radial owner is closed or failed')
        if not hasattr(masks, '__cuda_array_interface__') and not bool(getattr(masks, 'is_cuda', False)):
            raise ValueError('Radial owner consumes resident device masks; host mask upload is forbidden')
        cp = self.cp
        with cp.cuda.Device(self.device_index), cp.cuda.using_allocator(self._pool.malloc), self._stream:
            self._borrowed = cp.asarray(masks)
            source = self._borrowed
            count = int(source.shape[0]) if source.ndim == 3 else 0
            if (source.dtype != cp.uint8 or not source.flags.c_contiguous or source.shape[1:] != self.mask_shape
                    or count <= 0 or first < 0 or first + count > len(self.coverage)
                    or np.any(self.coverage[first:first + count])):
                self._borrowed = None
                raise ValueError('Invalid or duplicate native Radial chunk')
            try:
                started = time.perf_counter()
                boxes = self._boxes(source)
                nonempty = int(np.count_nonzero(boxes[:, 1] > boxes[:, 0]))
                self.native_any |= bool(nonempty)
                if clean:
                    self._clean(source, boxes)
                self._fence()
                self.cleanup_seconds += time.perf_counter() - started
                started = time.perf_counter()
                bucket_first = int(self.bucket_offsets[first])
                bucket_count = int(self.bucket_offsets[first + count]) - bucket_first
                work = bucket_count * len(self.metadata[0]) if nonempty else 0
                a = self._arrays
                boxes_gpu = self._borrowed_boxes = cp.asarray(boxes)
                for begin in range(0, work, _WORK_ITEMS):
                    stop = min(work, begin + _WORK_ITEMS)
                    self._kernels['gather_shell_chunk'](((stop - begin + 255) // 256,), (256,), (
                        source, a['pixels'], a['shells'], a['offsets'], a['columns'], a['sampled'],
                        a['rows'], a['mapped_columns'], a['centers'], a['ideal'], boxes_gpu, self.words,
                        np.uint64(bucket_first), np.uint64(bucket_count), np.uint64(begin), np.uint64(stop),
                        np.int32(first), np.int32(self.mask_shape[0]), np.int32(self.mask_shape[1]),
                        np.int32(self.view.src_h), np.int32(self.view.src_w), np.int32(self.stack_length),
                        np.int32(self.view.radial_height_origin), np.int32(self.plan.base_id), np.int32(self.vertical),
                        np.int32(self.plan.plane_shape[1]), np.int32(self.output_shape[1]), np.int32(self.output_shape[2])
                    ), stream=self._stream)
                self._fence()
                self.projection_seconds += time.perf_counter() - started
                self.coverage[first:first + count] = True
                self._borrowed = None
                self._borrowed_boxes = None
                return nonempty
            except RadialCudaProjectionUnsafeFailure:
                self._failed = True
                raise
            except BaseException:
                self._failed = True
                self._fence()
                self._borrowed = None
                self._borrowed_boxes = None
                raise

    def host_words(self):
        if self._failed or not self.coverage.all():
            raise RuntimeError('Cannot publish incomplete/failed Radial coverage')
        with self.cp.cuda.Device(self.device_index), self._stream:
            return self.words.get(stream=self._stream)

    def close(self):
        if self._closed:
            return
        if self._stream is not None:
            with self.cp.cuda.Device(self.device_index):
                self._fence()
                self._borrowed = self._borrowed_boxes = self.words = self._parents = self._exterior = None
                self._arrays.clear()
                self._module = self._meta_module = None
                self._pool.free_all_blocks()
        self.plan = self.metadata = None
        self._closed = True


def consume_radial_device_union(task, accumulator, *, target=None):
    if not is_radial_owner_task(task) or task.get('result_mode') != 'd1_owner':
        raise ValueError('Radial owner requires its dedicated native projection contract')
    if task.get('d1_group_id') or task.get('d1_view_shadow_required'):
        raise ValueError('Radial owner currently requires a single owner without native shadow consumers')
    if (task.get('kind', 'fullframe') != 'fullframe'
            or int(task.get('prediction_batch', 1)) != 1
            or float(task.get('streaming_cleanup_min_conf', 0)) != 0.
            or float(task.get('streaming_cleanup_min_radius', 0)) != 0.
            or float(task['view'].tta_angle_deg) != 0.):
        raise ValueError('Radial owner task violates its native cleanup contract')
    if accumulator.union_dev is None or accumulator.host_written:
        raise RuntimeError('Radial owner requires the complete native device union')
    if target is not None:
        target.require_complete()
    view = task['view']
    key = (str(task['model_name']), str(view.name))
    shape = tuple(int(v) for v in accumulator.union_dev.shape)
    first, count = int(task['slice_start']), int(task['slice_count'])
    if shape[0] != count:
        raise ValueError('Radial device chunk depth differs from its lease')
    output_shape = tuple(map(int, task['d1_output_shape']))
    with _RADIAL_OWNER_LOCK:
        owner = _RADIAL_OWNER_STATES.get(key)
        if owner is None:
            from . import cuda_d1
            if _RADIAL_OWNER_STATES or cuda_d1._D1_WORKER_VIEW_STATES:
                raise RuntimeError('This worker already retains a native Radial owner')
            owner = RadialOwner(view, shape[1:], output_shape)
            owner.key, owner.store_dir = key, Path(task['d1_store_dir'])
            owner.projection_kind = RADIAL_OWNER_CONTRACT
            _RADIAL_OWNER_STATES[key] = owner
            print(f'Radial owner admitted {key}: bitset_MiB={owner.words.nbytes / 2**20:.2f}, '
                  f'contract={RADIAL_OWNER_CONTRACT}; native task files and host view union bypassed.', flush=True)
        elif (owner.view != view or owner.mask_shape != shape[1:] or owner.output_shape != output_shape
              or owner.store_dir != Path(task['d1_store_dir'])):
            raise RuntimeError('Radial owner geometry changed during its view')
    nonempty = owner.consume(first, accumulator.union_dev)
    for name in ('union_dev', 'conf_dev', 'prediction_counts_dev', 'slice_bboxes_dev', 'slice_bboxes_written'):
        setattr(accumulator, name, None)
    covered = int(np.count_nonzero(owner.coverage))
    complete = covered == view.num_slices
    result = dict(d1_view_complete=complete, d1_covered_slices=covered, d1_total_slices=view.num_slices,
        d1_backprojected_task_slices=count, d1_nonempty_task_slices=nonempty,
        radial_owner=True, radial_owner_empty=not owner.native_any,
        device_hole_filled_frames=count, proto_hole_treated_frames=0)
    if not complete:
        return result
    words = owner.host_words()
    owner.close()
    with _RADIAL_OWNER_LOCK:
        if _RADIAL_OWNER_STATES.pop(key) is not owner:
            raise RuntimeError('Radial ownership changed during completion')
    from .cuda_d1 import _d1_submit_publication
    result['d1_bitset_words'] = len(words)
    result['_publication_future'] = _d1_submit_publication(words=words, state=owner)
    result['d1_view_compute_seconds'] = time.perf_counter() - owner.created_at
    print(f'Radial owner complete {key}: coverage={covered}/{view.num_slices}, '
          f'native_cleanup_s={owner.cleanup_seconds:.6f}, gather_s={owner.projection_seconds:.6f}; '
          'source-space publication queued without parent radial projection.', flush=True)
    return result


def active_radial_owners():
    with _RADIAL_OWNER_LOCK:
        return tuple(_RADIAL_OWNER_STATES)


def preflight_radial_owner():
    """Compile and execute native cleanup/gather before a worker accepts tasks."""
    from . import geometry, cylindrical_projection as reference
    import cupy as cp
    view = geometry.get_view_infos(7, 9, 11, cartesian_views=(), radial_views=('transverse',),
                                   radial_min_radius=.1, radial_patch_size=8)[0]
    data = np.zeros((view.num_slices, 8, 8), np.uint8)
    data[:, 1:7, 1:7] = 1
    data[:, 2:6, 2:6] = 0
    expected_native = data.copy()
    expected_native[:, 2:6, 2:6] = 1
    radii = np.asarray(geometry.radial_global_radii(view))
    expected = np.stack([reference._pull_radial_chunk(expected_native, view, radii,
        (7, 9, 11), z, 0, 99).reshape(9, 11) for z in range(7)])
    active = RadialOwner(view, (8, 8), (7, 9, 11), reserve_bytes=0)
    try:
        device = cp.asarray(data)
        active.consume(0, device)
        words = active.host_words()
        flat = np.arange(expected.size)
        decoded = ((words[flat // 32] >> (flat % 32).astype(np.uint32)) & 1).astype(np.uint8)
        if not np.array_equal(decoded, expected.reshape(-1)) or not np.array_equal(device.get(), expected_native):
            raise RuntimeError('Native Radial owner preflight disagreed with its CPU reference')
    finally:
        active.close()
    print('Native Radial owner preflight passed: exact native cleanup and shell-bucketed CUDA gather.', flush=True)


def shutdown_radial_owners():
    with _RADIAL_OWNER_LOCK:
        for key, owner in list(_RADIAL_OWNER_STATES.items()):
            owner.close()
            _RADIAL_OWNER_STATES.pop(key)
