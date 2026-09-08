"""Direct CUDA QSC source pull with bounded dense or compact publication.

Only the slice-crop encoder and its wire records are shared with the Radial
projector. Spherical geometry is evaluated directly in float64 on the device;
there are no cylindrical plans or host-generated voxel coordinate uploads.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator
import threading
import time

import numpy as np

from .qsc import QSC_FACE_BASES

from .cylindrical_cuda_projection import (
    RadialCudaProjector, RadialEncodedBlock, RadialCudaProjectionUnsafeFailure,
    _BLOCK_BYTES, _UPLOAD_BYTES, _RESERVE_BYTES, _SETUP_BYTES, _MAX_ENCODED_SLICES,
    _CROP_METADATA_DTYPE, _KERNEL_SOURCE as _CROP_KERNEL_BUNDLE,
)


class SphericalCudaProjectionUnavailable(RuntimeError):
    """Admission failed before publication; a settled CPU fallback is safe."""


class SphericalCudaProjectionUnsafeFailure(RadialCudaProjectionUnsafeFailure):
    """A failed stream fence retains the projector and its stage lease."""


@dataclass(frozen=True)
class _SphericalContract:
    output_shape: tuple[int, int, int]
    source_shape: tuple[int, int, int]
    arrays: dict
    cropped_source: bool


_SPHERICAL_KERNEL = r'''
extern "C" __global__ void project_spherical_qsc(
    const unsigned char* source, const double* radii, const double* rotation,
    const long long* boxes, const unsigned long long* offsets, unsigned char* out,
    int nshells, int source_h, int source_w, int native_h, int native_w,
    int work_t, int work_h, int work_w, int out_t, int out_h, int out_w,
    int face, int intervals, int origin_u, int origin_v, int first_z,
    double minimum, double maximum, unsigned long long voxel_count,
    int use_boxes, int cropped) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= voxel_count) return;
    out[q] = 0;
    unsigned long long plane = (unsigned long long)out_h * out_w;
    int z = first_z + (int)(q / plane);
    unsigned long long rem = q % plane;
    int y = (int)(rem / out_w), x = (int)(rem % out_w);
    double dx = (((x + .5) * work_w / out_w) - .5) - ((work_w - 1) / 2.0);
    double dy = (((y + .5) * work_h / out_h) - .5) - ((work_h - 1) / 2.0);
    double dz = (((z + .5) * work_t / out_t) - .5) - ((work_t - 1) / 2.0);
    double radius = sqrt((dx * dx + dy * dy) + dz * dz);
    if (radius < minimum || radius > maximum) return;
    double lx = (dx * rotation[0] + dy * rotation[3]) + dz * rotation[6];
    double ly = (dx * rotation[1] + dy * rotation[4]) + dz * rotation[7];
    double lz = (dx * rotation[2] + dy * rotation[5]) + dz * rotation[8];
    double scale = fmax(fabs(lx), fmax(fabs(ly), fabs(lz)));
    if (scale == 0.0) return;
    lx /= scale; ly /= scale; lz /= scale;
    double normal, right, up;
    if (face == 0) { normal = lx; right = ly; up = lz; }
    else if (face == 1) { normal = ly; right = -lx; up = lz; }
    else if (face == 2) { normal = -lx; right = -ly; up = lz; }
    else if (face == 3) { normal = -ly; right = lx; up = lz; }
    else if (face == 4) { normal = lz; right = ly; up = -lx; }
    else { normal = -lz; right = ly; up = lx; }
    if (normal <= 0.0 || normal + 1.7763568394002505e-15 < fmax(fabs(right), fabs(up))) return;
    double norm = sqrt((lx * lx + ly * ly) + lz * lz);
    normal /= norm; right /= norm; up /= norm;
    int area;
    double major, minor;
    if (fabs(right) >= fabs(up)) {
        if (right >= 0.0) { area = 0; major = right; minor = up; }
        else { area = 2; major = -right; minor = -up; }
    } else {
        if (up >= 0.0) { area = 1; major = up; minor = -right; }
        else { area = 3; major = -up; minor = right; }
    }
    double theta = atan2(minor, major);
    double ratio = 3.819718634205488 * (theta - asin(sin(theta) * 0.7071067811865475));
    if (fabs(minor) == major) ratio = minor > 0.0 ? 1.0 : (minor < 0.0 ? -1.0 : 0.0);
    double cosine = cos(theta);
    double d = 1.0 - cosine / sqrt(1.0 + cosine * cosine);
    double p = hypot(right, up) / sqrt((1.0 + normal) * d);
    if (major == normal) p = 1.0;
    double m = p * ratio, u, v;
    if (area == 0) { u = p; v = m; }
    else if (area == 1) { u = -m; v = p; }
    else if (area == 2) { u = -p; v = -m; }
    else { u = m; v = -p; }
    u = fmin(1.0, fmax(-1.0, u)); v = fmin(1.0, fmax(-1.0, v));
    long long col = __double2ll_rn((u + 1.0) * intervals / 2.0) - origin_u;
    long long row = __double2ll_rn((1.0 - v) * intervals / 2.0) - origin_v;
    if (row < 0 || row >= native_h || col < 0 || col >= native_w) return;
    int pr = native_h == source_h ? (int)row : min((int)((row + .5) * source_h / native_h), source_h - 1);
    int pc = native_w == source_w ? (int)col : min((int)((col + .5) * source_w / native_w), source_w - 1);
    int lo = 0, hi = nshells;
    while (lo < hi) { int mid = lo + (hi - lo) / 2; if (radii[mid] < radius) lo = mid + 1; else hi = mid; }
    int outer = min(lo, nshells - 1), inner = max(outer - 1, 0);
    int shell = radius - radii[inner] <= radii[outer] - radius ? inner : outer;
    unsigned long long box = (unsigned long long)shell * 4;
    if (use_boxes && (pr < boxes[box] || pr >= boxes[box + 1] || pc < boxes[box + 2] || pc >= boxes[box + 3])) return;
    unsigned long long source_at = ((unsigned long long)shell * source_h + pr) * source_w + pc;
    if (cropped) source_at = offsets[shell] + (unsigned long long)(pr - boxes[box]) * (boxes[box + 3] - boxes[box + 2]) + (pc - boxes[box + 2]);
    out[q] = source[source_at] != 0;
}
'''


class SphericalCudaProjector:
    """Private device owners; every public return has a completed stream fence."""

    # These methods operate exclusively on byte slices and crop metadata.
    # The actual contract and projection launch below are spherical.
    _record = RadialCudaProjector._record
    _elapsed = RadialCudaProjector._elapsed
    _enqueue_crop_metadata = RadialCudaProjector._enqueue_crop_metadata
    _encode_current_output = RadialCudaProjector._encode_current_output
    _validate_encoded_preflight = staticmethod(RadialCudaProjector._validate_encoded_preflight)
    _upload_cropped_source = RadialCudaProjector._upload_cropped_source
    _reset_projection_stats = RadialCudaProjector._reset_projection_stats

    def __init__(self, source, view, output_shape, bboxes=None, device_index=0, *,
                 block_bytes=_BLOCK_BYTES, upload_bytes=_UPLOAD_BYTES, reserve_bytes=_RESERVE_BYTES):
        from .spherical_projection import _validate_spherical_projection, _project_spherical_block

        started = time.perf_counter()
        source = np.asarray(source)
        if source.dtype != np.uint8 or not source.flags.c_contiguous:
            raise SphericalCudaProjectionUnavailable('Spherical CUDA requires a contiguous uint8 mask')
        radii, rotation, shape, boxes = _validate_spherical_projection(source, view, output_shape, bboxes)
        if max((*shape, *source.shape, view.src_h, view.src_w, view.spherical_face_intervals)) > np.iinfo(np.int32).max:
            raise SphericalCudaProjectionUnavailable('Spherical CUDA dimensions exceed int32 addressing')
        self.use_bboxes = boxes is not None
        if boxes is None:
            boxes = np.tile(np.array([0, source.shape[1], 0, source.shape[2]], np.int64), (source.shape[0], 1))
        sizes = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
        cropped = bool(self.use_bboxes and int(sizes.sum()) < source.nbytes)
        offsets = np.zeros(source.shape[0] + 1, np.uint64)
        if cropped:
            np.cumsum(sizes, dtype=np.uint64, out=offsets[1:])
        arrays = {'radii': np.ascontiguousarray(radii), 'rotation': np.ascontiguousarray(rotation),
                  'bboxes': np.ascontiguousarray(boxes), 'source_offsets': offsets}
        self.contract = _SphericalContract(shape, tuple(source.shape), arrays, cropped)
        budget = min(operator.index(block_bytes), _BLOCK_BYTES)
        plane_bytes = shape[1] * shape[2]
        if budget <= 0 or plane_bytes > budget or (shape[1] + 7) // 8 > 65535:
            raise SphericalCudaProjectionUnavailable('One Spherical output plane exceeds the bounded CUDA grid')
        self.max_block_depth = min(shape[0], budget // plane_bytes, _MAX_ENCODED_SLICES)
        self.device_index = operator.index(device_index)
        self.view = view
        self.source_bytes = source.nbytes
        self.source_layout = 'bbox_u8' if cropped else 'dense_u8'
        self.source_h2d_bytes = int(offsets[-1]) if cropped else source.nbytes
        self.geometry_bytes = sum(a.nbytes for a in arrays.values())
        self.output_buffer_bytes = self.max_block_depth * plane_bytes
        self.compact_buffer_bytes = self.output_buffer_bytes
        self.metadata_buffer_bytes = self.max_block_depth * _CROP_METADATA_DTYPE.itemsize
        self.offset_buffer_bytes = (self.max_block_depth + 1) * 8
        self.required_device_bytes = (max(1, self.source_h2d_bytes) + self.geometry_bytes + 2 * self.output_buffer_bytes
                                      + self.metadata_buffer_bytes + self.offset_buffer_bytes + _SETUP_BYTES)
        self.reserve_bytes = max(0, operator.index(reserve_bytes))
        self._upload_bytes = min(operator.index(upload_bytes), _UPLOAD_BYTES)
        if self._upload_bytes <= 0 or self.device_index < 0:
            raise ValueError('Spherical CUDA upload budget/device must be positive/nonnegative')
        self._lock = threading.RLock()
        self._cp = self._stream = self._pool = self._pinned_pool = None
        self._source_gpu = self._output_gpu = self._compact_gpu = self._metadata_gpu = self._offsets_gpu = None
        self._upload_pin = self._upload_stage = self._output_pin = self._output_stage = None
        self._metadata_pin = self._metadata_stage = self._offsets_pin = self._offsets_stage = None
        self._module = self._kernel = self._reset_metadata_kernel = self._reduce_metadata_kernel = self._encode_kernel = None
        self._arrays, self._events = {}, {}
        self._allocations = []
        self._closed = self._failed = False
        self._skip_empty_blocks = False
        self.source_upload_seconds = self.source_pack_seconds = self.geometry_upload_seconds = self.preflight_seconds = 0.0
        self.source_pack_backend = 'dense_copy'
        self._reset_projection_stats()
        try:
            import cupy as cp
            self._cp = cp
            if self.device_index >= cp.cuda.runtime.getDeviceCount():
                raise SphericalCudaProjectionUnavailable('Requested Spherical CUDA device is unavailable')
            with cp.cuda.Device(self.device_index):
                self._stream = cp.cuda.Stream(non_blocking=True)
                self._pool, self._pinned_pool = cp.cuda.MemoryPool(), cp.cuda.PinnedMemoryPool()
                free, _ = cp.cuda.runtime.memGetInfo()
                if free < self.required_device_bytes + self.reserve_bytes:
                    raise SphericalCudaProjectionUnavailable('Spherical CUDA source and bounded buffers exceed available device memory')
                with cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    self._module = cp.RawModule(code=_CROP_KERNEL_BUNDLE + _SPHERICAL_KERNEL,
                                                options=('--std=c++11', '--fmad=false'))
                    self._kernel = self._module.get_function('project_spherical_qsc')
                    self._reset_metadata_kernel = self._module.get_function('reset_radial_crop_metadata')
                    self._reduce_metadata_kernel = self._module.get_function('reduce_radial_crop_metadata')
                    self._encode_kernel = self._module.get_function('encode_radial_crops')
                    self._events = {f'{phase}_{edge}': cp.cuda.Event()
                                    for phase in ('kernel', 'metadata', 'pack', 'copy') for edge in ('start', 'end')}
                    stage_bytes = max(1, min(self._upload_bytes, max(self.source_h2d_bytes, self.geometry_bytes)))
                    self._upload_pin = self._pinned_pool.malloc(stage_bytes)
                    self._upload_stage = np.frombuffer(self._upload_pin, np.uint8, count=stage_bytes)
                    then = time.perf_counter()
                    if cropped:
                        self._upload_cropped_source(source)
                    else:
                        self._source_gpu = self._upload_array(source)
                    self.source_upload_seconds = time.perf_counter() - then
                    then = time.perf_counter()
                    for key, array in arrays.items():
                        self._arrays[key] = self._upload_array(array)
                    self.geometry_upload_seconds = time.perf_counter() - then
                    self._upload_stage = self._upload_pin = None
                    self._pinned_pool.free_all_blocks()
                    self._output_gpu = cp.empty((self.max_block_depth, *shape[1:]), cp.uint8)
                    self._compact_gpu = cp.empty(self.compact_buffer_bytes, cp.uint8)
                    self._metadata_gpu = cp.empty(self.metadata_buffer_bytes, cp.uint8)
                    self._offsets_gpu = cp.empty(self.max_block_depth + 1, cp.uint64)
                    self._output_pin = self._pinned_pool.malloc(self.output_buffer_bytes)
                    self._output_stage = np.frombuffer(self._output_pin, np.uint8, count=self.output_buffer_bytes).reshape(self._output_gpu.shape)
                    self._metadata_pin = self._pinned_pool.malloc(self.metadata_buffer_bytes)
                    self._metadata_stage = np.frombuffer(self._metadata_pin, _CROP_METADATA_DTYPE, count=self.max_block_depth)
                    self._offsets_pin = self._pinned_pool.malloc(self.offset_buffer_bytes)
                    self._offsets_stage = np.frombuffer(self._offsets_pin, np.uint64, count=self.max_block_depth + 1)
                    then = time.perf_counter()
                    # The middle XY plane can miss a polar face entirely.
                    # Also check the plane through this rotated face's interior.
                    normal_world = rotation @ np.asarray(QSC_FACE_BASES[int(view.spherical_face)][0])
                    point_z = (view.full_t - 1) / 2 + (radii[0] + radii[-1]) / 2 * normal_world[2]
                    face_z = int(np.clip(np.rint((point_z + .5) * shape[0] / view.full_t - .5), 0, shape[0] - 1))
                    self.preflight_planes = tuple(dict.fromkeys((shape[0] // 2, face_z)))
                    for checked_z in self.preflight_planes:
                        checked = self._run_block(checked_z, 1)
                        expected = _project_spherical_block(source, view, radii, rotation, shape, checked_z, 1,
                                                            boxes if self.use_bboxes else None)
                        if not np.array_equal(checked, expected):
                            differences = int(np.count_nonzero(checked != expected))
                            raise SphericalCudaProjectionUnavailable(
                                f'Spherical CUDA preflight differs from CPU at {differences} voxels on source Z={checked_z}')
                    for packed in (False, True):
                        self._validate_encoded_preflight(checked, self._encode_current_output(checked_z, 1, packed))
                    self.preflight_seconds = time.perf_counter() - then
                    self._skip_empty_blocks = True
                    self._reset_projection_stats()
            self.constructor_seconds = time.perf_counter() - started
        except SphericalCudaProjectionUnsafeFailure:
            raise
        except BaseException as exc:
            self.close()
            if isinstance(exc, (SphericalCudaProjectionUnavailable, KeyboardInterrupt, SystemExit)):
                raise
            raise SphericalCudaProjectionUnavailable(f'Spherical CUDA startup failed: {type(exc).__name__}: {exc}') from exc

    def _upload_array(self, host):
        device = self._cp.empty(host.shape, host.dtype)
        # Register each owner before enqueueing a copy, including failed uploads.
        self._allocations.append(device)
        raw = host.view(np.uint8).reshape(-1)
        for first in range(0, raw.size, self._upload_stage.size):
            count = min(self._upload_stage.size, raw.size - first)
            np.copyto(self._upload_stage[:count], raw[first:first + count])
            self._cp.cuda.runtime.memcpyAsync(int(device.data.ptr) + first, int(self._upload_pin.ptr), count,
                self._cp.cuda.runtime.memcpyHostToDevice, int(self._stream.ptr))
            self._stream.synchronize()
        return device

    def _launch_projection(self, first, count):
        c, a, v = self.contract, self._arrays, self.view
        voxels = count * c.output_shape[1] * c.output_shape[2]
        self._record('kernel', 'start')
        self._kernel(((voxels + 255) // 256,), (256,), (
            self._source_gpu, a['radii'], a['rotation'], a['bboxes'], a['source_offsets'], self._output_gpu,
            *(np.int32(value) for value in (c.source_shape[0], c.source_shape[1], c.source_shape[2], v.src_h, v.src_w,
                v.full_t, v.full_h, v.full_w, *c.output_shape, v.spherical_face, v.spherical_face_intervals,
                v.spherical_u_origin, v.spherical_v_origin, first)),
            np.float64(v.spherical_min_radius), np.float64(v.spherical_max_radius), np.uint64(voxels),
            np.int32(self.use_bboxes), np.int32(c.cropped_source)), stream=self._stream)
        self._record('kernel', 'end')
        return voxels

    def _run_block(self, first, count):
        voxels = self._launch_projection(first, count)
        self._record('copy', 'start')
        self._cp.cuda.runtime.memcpyAsync(int(self._output_pin.ptr), int(self._output_gpu.data.ptr), voxels,
            self._cp.cuda.runtime.memcpyDeviceToHost, int(self._stream.ptr))
        self._record('copy', 'end')
        self._stream.synchronize()
        self.kernel_seconds += self._elapsed('kernel')
        self.d2h_seconds += self._elapsed('copy')
        self.dense_d2h_bytes += voxels
        return self._output_stage[:count].copy()

    def _validate_block(self, first, count):
        first, count = operator.index(first), operator.index(count)
        if self._closed or self._failed:
            raise RuntimeError('Spherical CUDA projector is closed or failed')
        if first < 0 or count < 0 or count > self.max_block_depth or first + count > self.contract.output_shape[0]:
            raise ValueError('Spherical CUDA output block exceeds its admitted bounds')
        return first, count

    def project(self, first, count):
        with self._lock:
            first, count = self._validate_block(first, count)
            if count == 0:
                return np.empty((0, *self.contract.output_shape[1:]), np.uint8)
            try:
                with self._cp.cuda.Device(self.device_index), self._cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    return self._run_block(first, count)
            except BaseException:
                self._failed = True
                raise

    def project_encoded(self, first, count, packed=False):
        with self._lock:
            first, count = self._validate_block(first, count)
            if count == 0:
                payload = np.empty(0, np.uint8)
                payload.flags.writeable = False
                return RadialEncodedBlock(first, (), payload, bool(packed))
            try:
                with self._cp.cuda.Device(self.device_index), self._cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    self._launch_projection(first, count)
                    result = self._encode_current_output(first, count, bool(packed))
                    self.kernel_seconds += self._elapsed('kernel')
                    return result
            except BaseException:
                self._failed = True
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._stream is not None:
                try:
                    with self._cp.cuda.Device(self.device_index):
                        self._stream.synchronize()
                except BaseException as exc:
                    raise SphericalCudaProjectionUnsafeFailure('Could not settle Spherical CUDA projection stream', self) from exc
            if self._cp is not None and self._stream is not None:
                with self._cp.cuda.Device(self.device_index):
                    self._arrays.clear()
                    self._allocations.clear()
                    self._events.clear()
                    self._source_gpu = self._output_gpu = self._compact_gpu = self._metadata_gpu = self._offsets_gpu = None
                    self._module = self._kernel = self._reset_metadata_kernel = self._reduce_metadata_kernel = self._encode_kernel = None
                    self._upload_stage = self._output_stage = self._metadata_stage = self._offsets_stage = None
                    self._upload_pin = self._output_pin = self._metadata_pin = self._offsets_pin = None
                    if self._pool is not None:
                        self._pool.free_all_blocks()
                    if self._pinned_pool is not None:
                        self._pinned_pool.free_all_blocks()
            self._stream = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
