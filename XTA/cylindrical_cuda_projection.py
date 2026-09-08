"""Bounded CUDA consumption of the exact host Radial pull-projection plan.

No cylindrical geometry is recomputed on CUDA. CPU-generated ownership, periodic
columns, sample shear, and processing-index maps remain authoritative. The caller
owns the stage lease and sink transaction; this module owns only its private CUDA
allocations and returns independent host blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import threading
import time
from typing import Any

import numpy as np

from ._deps import _numba

_BLOCK_BYTES = 64 * 1024 * 1024
_UPLOAD_BYTES = 64 * 1024 * 1024
_RESERVE_BYTES = 2 * 1024**3
_SETUP_BYTES = 256 * 1024**2
_MAX_ENCODED_SLICES = 4096
_CROP_METADATA_DTYPE = np.dtype([
    ('y0', np.int32), ('y1', np.int32), ('x0', np.int32), ('x1', np.int32),
    ('foreground', np.uint64),
], align=True)


def _pack_radial_source_block(source, bboxes, offsets, first, destination):
    """Copy an arbitrary interval of concatenated source rectangles, without GIL handoffs.

    Inputs are validated by the CUDA contract. No normalization or floating point
    addressing occurs, and a block may start/end inside a row or cross empty shells.
    """
    position = np.int64(first)
    source_flat = source.reshape(-1)
    cursor = 0
    shell = 0
    while cursor < destination.size:
        while np.int64(offsets[shell + 1]) <= position:
            shell += 1
        y0, y1, x0, x1 = bboxes[shell]
        width = x1 - x0
        within = position - np.int64(offsets[shell])
        row = y0 + within // width
        column = x0 + within % width
        count = min(destination.size - cursor, x1 - column)
        # Validated nonnegative linear indices avoid per-byte negative-index
        # correction and let LLVM generate contiguous vector loads/stores.
        source_first = np.uint64((shell * source.shape[1] + row) * source.shape[2] + column)
        output_first = np.uint64(cursor)
        for at in range(count):
            destination[output_first + np.uint64(at)] = source_flat[source_first + np.uint64(at)]
        cursor += count
        position += count


_pack_radial_source_block_compiled = (
    _numba.njit(cache=True, nogil=True)(_pack_radial_source_block) if _numba is not None else None
)


@dataclass(frozen=True)
class RadialEncodedSlice:
    z: int
    y0: int
    y1: int
    x0: int
    x1: int
    foreground: int
    offset: int
    size: int


@dataclass(frozen=True)
class RadialEncodedBlock:
    first_z: int
    records: tuple[RadialEncodedSlice, ...]
    payload: np.ndarray
    packed: bool = False


def _encoded_records(first_z, metadata, packed, plane_shape, capacity):
    """Validate small device metadata and prefix the exact on-disk payloads."""
    records = []
    offsets = np.empty(len(metadata) + 1, np.uint64)
    offsets[0] = 0
    total = largest = 0
    height, width = plane_shape
    for local, item in enumerate(metadata):
        foreground = int(item['foreground'])
        if foreground == 0:
            y0 = y1 = x0 = x1 = size = 0
        else:
            y0, y1, x0, x1 = (int(item[name]) for name in ('y0', 'y1', 'x0', 'x1'))
            if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
                raise RuntimeError('Radial CUDA crop metadata has invalid bounds')
            if foreground < 0 or foreground > (y1 - y0) * (x1 - x0):
                raise RuntimeError('Radial CUDA crop metadata has invalid foreground count')
            stride = (x1 - x0 + 7) // 8 if packed else x1 - x0
            size = (y1 - y0) * stride
        records.append(RadialEncodedSlice(first_z + local, y0, y1, x0, x1, foreground, total, size))
        total += size
        largest = max(largest, size)
        offsets[local + 1] = total
    if total > capacity:
        raise RuntimeError('Radial CUDA crop payload exceeds its admitted output buffer')
    return tuple(records), offsets, total, largest


class RadialCudaProjectionUnavailable(RuntimeError):
    """No output has been delivered; the caller may use its CPU projector."""


class RadialCudaProjectionUnsafeFailure(BaseException):
    """A stream could not be settled; abort instead of falling back mid-flight."""

    def __init__(self, message, projector):
        super().__init__(message)
        # Keep owners alive if the caller reports a failed fence before aborting.
        self.projector = projector


@dataclass(frozen=True)
class _ProjectionContract:
    output_shape: tuple[int, int, int]
    source_shape: tuple[int, int, int]
    arrays: dict[str, np.ndarray]
    base_id: int
    vertical: bool
    plane_width: int
    native_height: int
    native_width: int
    height_origin: int
    stack_length: int
    use_bboxes: bool
    max_block_depth: int
    block_bytes: int
    cropped_source: bool

    @property
    def geometry_bytes(self):
        return sum(array.nbytes for array in self.arrays.values())


def _positive_shape(value, name):
    try:
        shape = tuple(operator.index(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain integer dimensions') from exc
    if len(shape) != 3 or min(shape) <= 0 or max(shape) > np.iinfo(np.int32).max:
        raise ValueError(f'{name} requires three positive int32 dimensions')
    return shape


def _contract_array(value, dtype, shape, name):
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype) or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f'{name} requires contiguous {np.dtype(dtype)} shape {shape}')
    return array


def _validate_projection_contract(source, plan, metadata, view, output_shape,
                                  bboxes, use_bboxes, block_bytes):
    array = np.asarray(source)
    if array.ndim != 3 or array.dtype != np.uint8 or not array.flags.c_contiguous:
        raise RadialCudaProjectionUnavailable('CUDA Radial projection requires a contiguous uint8 mask')
    source_shape = _positive_shape(array.shape, 'Radial source')
    shape = _positive_shape(output_shape, 'Radial output')
    if (shape[1] + 7) // 8 > 65535:
        raise RadialCudaProjectionUnavailable('Radial output height exceeds the CUDA crop-reduction grid limit')
    if str(view.family) != 'radial' or source_shape[0] != int(view.num_slices):
        raise ValueError('Radial source depth does not match its view')
    native_height, native_width = operator.index(view.src_h), operator.index(view.src_w)
    stack_height = operator.index(view.radial_height_origin)
    if (min(native_height, native_width) <= 0 or
            max(native_height, native_width, abs(stack_height)) > np.iinfo(np.int32).max):
        raise ValueError('Radial native patch dimensions/origin exceed int32 addressing')
    base = operator.index(plan.base_id)
    if base not in (0, 1, 2) or str(view.radial_base_view) != ('transverse', 'sagittal', 'coronal')[base]:
        raise ValueError('Radial plane plan has the wrong base orientation')
    plane = ((shape[1], shape[2]), (shape[0], shape[2]), (shape[0], shape[1]))[base]
    if tuple(plan.plane_shape) != plane:
        raise ValueError('Radial plane plan does not match source output geometry')
    pixels = math.prod(plane)
    columns = np.asarray(plan.native_columns)
    if columns.ndim != 1:
        raise ValueError('Radial periodic columns must be one-dimensional')
    arrays = {
        'shells': _contract_array(plan.shell_index, np.int32, (pixels,), 'Radial shell indices'),
        'offsets': _contract_array(plan.column_offsets, np.uint32, (pixels + 1,), 'Radial column offsets'),
        'columns': _contract_array(columns, np.int32, (len(columns),), 'Radial periodic columns'),
    }
    shells, offsets = arrays['shells'], arrays['offsets']
    if (np.any(shells < -1) or np.any(shells >= source_shape[0]) or
            offsets[0] != 0 or int(offsets[-1]) != len(columns) or
            np.any(offsets[1:] < offsets[:-1]) or
            np.any(columns < 0) or np.any(columns >= native_width)):
        raise ValueError('Radial plane plan contains invalid gather addresses')
    if len(metadata) != 7:
        raise ValueError('Radial projection metadata requires seven fields')
    centers, ideal, sampled, row_map, column_map, stack_length, vertical = metadata
    stack_length = operator.index(stack_length)
    physical_shape = _positive_shape((view.full_t, view.full_h, view.full_w), 'Radial physical geometry')
    if stack_length != physical_shape[base]:
        raise ValueError('Radial stack metadata does not match physical geometry')
    vertical = bool(vertical)
    arrays.update({
        'centers': _contract_array(centers, np.float64, (shape[base],), 'Radial source stack centers'),
        'ideal': _contract_array(ideal, np.float64, (plane[0 if vertical else 1],), 'Radial ideal shear axis'),
        'sampled': _contract_array(sampled, np.float64, (source_shape[0], native_width), 'Radial sampled shear'),
        'rows': _contract_array(row_map, np.int32, (native_height,), 'Radial processing rows'),
        'mapped_columns': _contract_array(column_map, np.int32, (native_width,), 'Radial processing columns'),
    })
    if (any(not np.isfinite(arrays[name]).all() for name in ('centers', 'ideal', 'sampled')) or
            np.any(arrays['rows'] < 0) or np.any(arrays['rows'] >= source_shape[1]) or
            np.any(arrays['mapped_columns'] < 0) or np.any(arrays['mapped_columns'] >= source_shape[2])):
        raise ValueError('Radial projection metadata contains invalid gather coordinates')
    if use_bboxes:
        boxes = _contract_array(bboxes, np.int64, (source_shape[0], 4), 'Radial source bounding boxes')
        if (np.any(boxes < 0) or np.any(boxes[:, 0] > boxes[:, 1]) or np.any(boxes[:, 2] > boxes[:, 3]) or
                np.any(boxes[:, 1] > source_shape[1]) or np.any(boxes[:, 3] > source_shape[2])):
            raise ValueError('Radial source bounding boxes exceed mask dimensions')
    else:
        boxes = np.zeros((source_shape[0], 4), np.int64)
    arrays['bboxes'] = boxes
    # Known bounds already constrain every gather. Store only those rectangles;
    # this changes byte addressing, not the set of samples the kernel can read.
    sizes = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    cropped_source = bool(use_bboxes and int(sizes.sum()) < array.nbytes)
    source_offsets = np.zeros(source_shape[0] + 1, np.uint64)
    if cropped_source:
        np.cumsum(sizes, dtype=np.uint64, out=source_offsets[1:])
    arrays['source_offsets'] = source_offsets
    budget = min(operator.index(block_bytes), _BLOCK_BYTES)
    plane_bytes = int(shape[1]) * int(shape[2])
    if budget <= 0 or plane_bytes > budget:
        raise RadialCudaProjectionUnavailable('One Radial output slice exceeds the CUDA block budget')
    depth = min(shape[0], budget // plane_bytes, _MAX_ENCODED_SLICES)
    return _ProjectionContract(shape, source_shape, arrays, base, vertical, plane[1],
        native_height, native_width, stack_height, stack_length, bool(use_bboxes), depth, depth * plane_bytes,
        cropped_source)


_KERNEL_SOURCE = r'''
extern "C" __global__ void project_radial_plan(
    const unsigned char* source, const int* shells,
    const unsigned int* offsets, const int* columns,
    const double* sampled, const int* row_map, const int* column_map,
    const double* stack_centers, const double* ideal_axis,
    const long long* bboxes, const unsigned long long* source_offsets,
    const int* launch_first_z, unsigned char* out,
    int source_h, int source_w, int native_height, int native_width,
    int stack_length, int height_origin, int base_id, int vertical,
    int plane_width, int out_h, int out_w, int first_z,
    unsigned long long voxel_count, int use_bboxes, int cropped_source) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= voxel_count) return;
    out[q] = 0;
    unsigned long long plane_pixels = (unsigned long long)out_h * out_w;
    int z = (first_z < 0 ? launch_first_z[0] : first_z) + (int)(q / plane_pixels);
    unsigned long long rem = q % plane_pixels;
    int y = (int)(rem / out_w), x = (int)(rem % out_w);
    unsigned long long p;
    int stack_index, v, u;
    if (base_id == 0) {
        p = (unsigned long long)y * plane_width + x;
        stack_index = z; v = y; u = x;
    } else if (base_id == 1) {
        p = (unsigned long long)z * plane_width + x;
        stack_index = y; v = z; u = x;
    } else {
        p = (unsigned long long)z * plane_width + y;
        stack_index = x; v = z; u = y;
    }
    int shell = shells[p];
    if (shell < 0) return;
    unsigned long long box = (unsigned long long)shell * 4;
    if (use_bboxes && (bboxes[box + 1] <= bboxes[box] || bboxes[box + 3] <= bboxes[box + 2])) return;
    double stack = stack_centers[stack_index];
    double ideal_height = __dsub_rn(stack, ideal_axis[vertical ? v : u]);
    if (ideal_height < 0.0 || ideal_height > (double)(stack_length - 1)) return;
    for (unsigned long long at = offsets[p]; at < (unsigned long long)offsets[p + 1]; ++at) {
        int column = columns[at];
        double height = __dsub_rn(stack, sampled[(unsigned long long)shell * native_width + column]);
        height = fmin(fmax(height, 0.0), (double)(stack_length - 1));
        long long row = __double2ll_rn(height) - (long long)height_origin;
        if (row < 0 || row >= native_height) continue;
        int pr = row_map[row], pc = column_map[column];
        if (use_bboxes && ((long long)pr < bboxes[box] || (long long)pr >= bboxes[box + 1] ||
                          (long long)pc < bboxes[box + 2] || (long long)pc >= bboxes[box + 3])) continue;
        unsigned long long source_at = ((unsigned long long)shell * source_h + pr) * source_w + pc;
        if (cropped_source) {
            source_at = source_offsets[shell] +
                (unsigned long long)(pr - bboxes[box]) * (bboxes[box + 3] - bboxes[box + 2]) +
                (unsigned long long)(pc - bboxes[box + 2]);
        }
        if (source[source_at] != 0) { out[q] = 1; return; }
    }
}

struct RadialCropMetadata {
    int y0, y1, x0, x1;
    unsigned long long foreground;
};

extern "C" __global__ void reset_radial_crop_metadata(
    RadialCropMetadata* metadata, int count, int height, int width) {
    int slice = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (slice >= count) return;
    metadata[slice].y0 = height; metadata[slice].y1 = 0;
    metadata[slice].x0 = width; metadata[slice].x1 = 0;
    metadata[slice].foreground = 0;
}

extern "C" __global__ void reduce_radial_crop_metadata(
    const unsigned char* source, RadialCropMetadata* metadata,
    int count, int height, int width) {
    // Each full warp owns one output row; every lane participates in ballot
    // and the block barrier, including pixels beyond the image boundary.
    __shared__ unsigned int row_masks[8];
    int slice = (int)blockIdx.z;
    if (slice >= count) return;
    int x = (int)blockIdx.x * 32 + (int)threadIdx.x;
    int y = (int)blockIdx.y * 8 + (int)threadIdx.y;
    bool foreground = x < width && y < height &&
        source[((unsigned long long)slice * height + y) * width + x] != 0;
    unsigned int bits = __ballot_sync(0xffffffffu, foreground);
    if (threadIdx.x == 0) row_masks[threadIdx.y] = bits;
    __syncthreads();
    if (threadIdx.x != 0 || threadIdx.y != 0) return;
    int y0 = height, y1 = 0, x0 = width, x1 = 0;
    unsigned int total = 0;
    for (int row = 0; row < 8; ++row) {
        unsigned int mask = row_masks[row];
        if (mask == 0) continue;
        int yy = (int)blockIdx.y * 8 + row;
        y0 = min(y0, yy); y1 = max(y1, yy + 1);
        x0 = min(x0, (int)blockIdx.x * 32 + __ffs(mask) - 1);
        x1 = max(x1, (int)blockIdx.x * 32 + 32 - __clz(mask));
        total += __popc(mask);
    }
    if (total) {
        // Five global atomics per nonempty block, never per foreground voxel.
        atomicMin(&metadata[slice].y0, y0); atomicMax(&metadata[slice].y1, y1);
        atomicMin(&metadata[slice].x0, x0); atomicMax(&metadata[slice].x1, x1);
        atomicAdd(&metadata[slice].foreground, (unsigned long long)total);
    }
}

extern "C" __global__ void encode_radial_crops(
    const unsigned char* source, const RadialCropMetadata* metadata,
    const unsigned long long* offsets, unsigned char* payload,
    int count, int height, int width, int packed) {
    int slice = (int)blockIdx.y;
    if (slice >= count || metadata[slice].foreground == 0) return;
    const RadialCropMetadata box = metadata[slice];
    int crop_width = box.x1 - box.x0;
    int row_bytes = packed ? (crop_width + 7) / 8 : crop_width;
    unsigned long long size = (unsigned long long)(box.y1 - box.y0) * row_bytes;
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= size) return;
    int row = (int)(q / row_bytes);
    int byte_x = (int)(q % row_bytes);
    unsigned long long source_row = ((unsigned long long)slice * height + box.y0 + row) * width;
    unsigned char result = 0;
    if (packed) {
        // Bit zero is the leftmost crop pixel. Padding bits stay zero and every
        // row starts a new byte sequence, matching NumPy axis=1, bitorder=little.
        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            int x = byte_x * 8 + bit;
            if (x < crop_width && source[source_row + box.x0 + x] != 0)
                result |= (unsigned char)(1u << bit);
        }
    } else {
        result = source[source_row + box.x0 + byte_x] != 0 ? 1 : 0;
    }
    payload[offsets[slice] + q] = result;
}
'''


class RadialCudaProjector:
    """Private, thread-transferable CUDA projector with owned host return blocks.

    Construction settles upload and a one-slice preflight before returning. A
    project call is synchronous with respect to its returned host array, but its
    caller may consume a previous array concurrently. close() serializes against
    active calls and fences before releasing any memory. No stage lease is taken.
    """

    def __init__(self, source, plan, metadata_tuple, view, output_shape,
                 bboxes=None, use_bboxes=False, device_index=0, *,
                 block_bytes=_BLOCK_BYTES, upload_bytes=_UPLOAD_BYTES, reserve_bytes=_RESERVE_BYTES,
                 use_graphs=False, skip_empty_blocks=False):
        started = time.perf_counter()
        self.source_upload_seconds = self.geometry_upload_seconds = self.preflight_seconds = 0.0
        self.source_pack_seconds = self.module_setup_seconds = self.buffer_setup_seconds = 0.0
        self.source_pack_backend = 'dense_copy'
        self.cuda_graph_setup_seconds = 0.0
        self.cuda_graph_enabled = False
        self.cuda_graph_note = 'disabled' if not use_graphs else 'not_initialized'
        # Preflight still executes both packing kernels, even for an empty first
        # slice. Empty-packet elision is enabled only after that check completes.
        self._skip_empty_blocks = False
        self.constructor_seconds = 0.0
        self.contract = _validate_projection_contract(source, plan, metadata_tuple, view,
            output_shape, bboxes, use_bboxes, block_bytes)
        self.contract_validation_seconds = time.perf_counter() - started
        self.max_block_depth = self.contract.max_block_depth
        self.device_index = operator.index(device_index)
        self.source_bytes = int(np.asarray(source).nbytes)
        self.source_layout = 'bbox_u8' if self.contract.cropped_source else 'dense_u8'
        self.source_h2d_bytes = (int(self.contract.arrays['source_offsets'][-1])
                                 if self.contract.cropped_source else self.source_bytes)
        self.geometry_bytes = int(self.contract.geometry_bytes)
        self.output_buffer_bytes = int(self.contract.block_bytes)
        self.compact_buffer_bytes = self.output_buffer_bytes
        self.metadata_buffer_bytes = self.max_block_depth * int(_CROP_METADATA_DTYPE.itemsize)
        self.offset_buffer_bytes = (self.max_block_depth + 1) * np.dtype(np.uint64).itemsize
        self.required_device_bytes = (max(1, self.source_h2d_bytes) + self.geometry_bytes + self.output_buffer_bytes
            + self.compact_buffer_bytes + self.metadata_buffer_bytes + self.offset_buffer_bytes + 4 + _SETUP_BYTES)
        self.reserve_bytes = max(0, operator.index(reserve_bytes))
        self._upload_bytes = operator.index(upload_bytes)
        if self._upload_bytes <= 0 or self.device_index < 0:
            raise ValueError('Radial CUDA upload budget/device must be positive/nonnegative')
        self._lock = threading.RLock()
        self._cp = self._stream = self._pool = self._pinned_pool = None
        self._source_gpu = self._output_gpu = self._module = self._kernel = None
        self._compact_gpu = self._metadata_gpu = self._offsets_gpu = None
        self._reset_metadata_kernel = self._reduce_metadata_kernel = self._encode_kernel = None
        self._upload_pin = self._upload_stage = self._output_pin = self._output_stage = None
        self._metadata_pin = self._metadata_stage = self._offsets_pin = self._offsets_stage = None
        self._events = {}
        self._graphs = {}
        self._first_z_gpu = self._first_z_pin = self._first_z_stage = None
        self._arrays: dict[str, Any] = {}
        self._closed = self._failed = False
        self._reset_projection_stats()
        try:
            import cupy as cp  # type: ignore
            self._cp = cp
            if self.device_index >= int(cp.cuda.runtime.getDeviceCount()):
                raise RadialCudaProjectionUnavailable('Requested Radial CUDA device is unavailable')
            with cp.cuda.Device(self.device_index):
                self._stream = cp.cuda.Stream(non_blocking=True)
                self._pool = cp.cuda.MemoryPool()
                self._pinned_pool = cp.cuda.PinnedMemoryPool()
                free_bytes, _ = cp.cuda.runtime.memGetInfo()
                if int(free_bytes) < self.required_device_bytes + self.reserve_bytes:
                    raise RadialCudaProjectionUnavailable(
                        f'Radial CUDA projection needs {self.required_device_bytes / 1024**3:.2f} GiB '
                        f'plus {self.reserve_bytes / 1024**3:.2f} GiB reserve; '
                        f'{int(free_bytes) / 1024**3:.2f} GiB is free')
                with cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    module_started = time.perf_counter()
                    self._module = cp.RawModule(code=_KERNEL_SOURCE, options=('--std=c++11', '--fmad=false'))
                    self._kernel = self._module.get_function('project_radial_plan')
                    self._reset_metadata_kernel = self._module.get_function('reset_radial_crop_metadata')
                    self._reduce_metadata_kernel = self._module.get_function('reduce_radial_crop_metadata')
                    self._encode_kernel = self._module.get_function('encode_radial_crops')
                    self._events = {f'{phase}_{boundary}': cp.cuda.Event()
                        for phase in ('kernel', 'metadata', 'pack', 'copy', 'graph') for boundary in ('start', 'end')}
                    self.module_setup_seconds = time.perf_counter() - module_started
                    buffer_started = time.perf_counter()
                    stage_bytes = min(self._upload_bytes, _UPLOAD_BYTES, max(self.source_h2d_bytes,
                        max((array.nbytes for array in self.contract.arrays.values()), default=1)))
                    self._upload_pin = self._pinned_pool.malloc(int(stage_bytes))
                    self._upload_stage = np.frombuffer(self._upload_pin, dtype=np.uint8, count=int(stage_bytes))
                    self.buffer_setup_seconds = time.perf_counter() - buffer_started
                    upload_started = time.perf_counter()
                    if self.contract.cropped_source:
                        self._upload_cropped_source(np.asarray(source))
                    else:
                        self._source_gpu = self._upload_array(np.asarray(source))
                    self.source_upload_seconds = time.perf_counter() - upload_started
                    geometry_started = time.perf_counter()
                    for name, array in self.contract.arrays.items():
                        self._arrays[name] = self._upload_array(array)
                    self.geometry_upload_seconds = time.perf_counter() - geometry_started
                    buffer_started = time.perf_counter()
                    self._upload_stage = self._upload_pin = None
                    self._pinned_pool.free_all_blocks()
                    self._output_gpu = cp.empty((self.max_block_depth, *self.contract.output_shape[1:]), dtype=cp.uint8)
                    self._output_pin = self._pinned_pool.malloc(self.output_buffer_bytes)
                    self._output_stage = np.frombuffer(self._output_pin, dtype=np.uint8,
                        count=self.output_buffer_bytes).reshape(self._output_gpu.shape)
                    self._compact_gpu = cp.empty(self.compact_buffer_bytes, dtype=cp.uint8)
                    self._metadata_gpu = cp.empty(self.metadata_buffer_bytes, dtype=cp.uint8)
                    self._offsets_gpu = cp.empty(self.max_block_depth + 1, dtype=cp.uint64)
                    self._metadata_pin = self._pinned_pool.malloc(self.metadata_buffer_bytes)
                    self._metadata_stage = np.frombuffer(self._metadata_pin, dtype=_CROP_METADATA_DTYPE,
                        count=self.max_block_depth)
                    self._offsets_pin = self._pinned_pool.malloc(self.offset_buffer_bytes)
                    self._offsets_stage = np.frombuffer(self._offsets_pin, dtype=np.uint64,
                        count=self.max_block_depth + 1)
                    self._first_z_gpu = cp.empty(1, dtype=cp.int32)
                    self._first_z_pin = self._pinned_pool.malloc(4)
                    self._first_z_stage = np.frombuffer(self._first_z_pin, dtype=np.int32, count=1)
                    self.buffer_setup_seconds += time.perf_counter() - buffer_started
                    # Execute the actual kernel and D2H path before a caller can
                    # publish a slice, surfacing launch/copy errors at admission.
                    preflight_started = time.perf_counter()
                    checked = self._run_block(0, 1)
                    if np.any(checked > 1):
                        raise RuntimeError('Radial CUDA preflight did not produce a binary mask')
                    for packed in (False, True):
                        encoded = self._encode_current_output(0, 1, packed)
                        self._validate_encoded_preflight(checked, encoded)
                    if use_graphs:
                        graph_started = time.perf_counter()
                        self._prepare_projection_graphs()
                        self.cuda_graph_setup_seconds = time.perf_counter() - graph_started
                        if self.cuda_graph_enabled:
                            # Compare graph replay with the already-checked direct
                            # projection before this constructor can publish output.
                            self._run_projection_graph(0, 1)
                            encoded = self._encode_current_output(0, 1, False, metadata_ready=True)
                            self._validate_encoded_preflight(checked, encoded)
                    self.preflight_seconds = time.perf_counter() - preflight_started
                    self._skip_empty_blocks = bool(skip_empty_blocks)
                    self._reset_projection_stats()
            self.constructor_seconds = time.perf_counter() - started
        except RadialCudaProjectionUnsafeFailure:
            # A failed capture fence already established uncertain ownership.
            # Do not retry cleanup and downgrade it into an ordinary fallback.
            raise
        except BaseException as exc:
            self.close()
            if isinstance(exc, (RadialCudaProjectionUnavailable, KeyboardInterrupt, SystemExit)):
                raise
            raise RadialCudaProjectionUnavailable(f'Radial CUDA startup failed: {type(exc).__name__}: {exc}') from exc

    def _upload_array(self, host):
        cp = self._cp
        device = cp.empty(host.shape, dtype=host.dtype)
        raw = host.view(np.uint8).reshape(-1)
        for first in range(0, int(raw.size), int(self._upload_stage.size)):
            count = min(int(self._upload_stage.size), int(raw.size) - first)
            np.copyto(self._upload_stage[:count], raw[first:first + count])
            cp.cuda.runtime.memcpyAsync(int(device.data.ptr) + first, int(self._upload_pin.ptr), count,
                cp.cuda.runtime.memcpyHostToDevice, int(self._stream.ptr))
            self._stream.synchronize()
        return device

    def _upload_cropped_source(self, source):
        """Pack strided rectangles directly into one bounded pinned buffer.

        No full host copy or dense device canvas is created. The device owner is
        published before the first asynchronous copy, including failure paths.
        Empty shells occupy no payload bytes and are rejected by the bbox guard.
        """
        cp = self._cp
        self._source_gpu = cp.empty(max(1, self.source_h2d_bytes), dtype=cp.uint8)
        capacity = int(self._upload_stage.size)
        pack = _pack_radial_source_block_compiled
        boxes, offsets = self.contract.arrays['bboxes'], self.contract.arrays['source_offsets']
        if pack is not None:
            # Compile before any source copy. Optional compilation failures can
            # still select the original NumPy uploader without partial delivery.
            try:
                pack(source, boxes, offsets, 0, self._upload_stage[:0])
            except Exception:
                pack = None
        if pack is not None:
            self.source_pack_backend = 'numba_nogil'
            for first in range(0, self.source_h2d_bytes, capacity):
                count = min(capacity, self.source_h2d_bytes - first)
                pack_started = time.perf_counter()
                pack(source, boxes, offsets, first, self._upload_stage[:count])
                self.source_pack_seconds += time.perf_counter() - pack_started
                cp.cuda.runtime.memcpyAsync(int(self._source_gpu.data.ptr) + first,
                    int(self._upload_pin.ptr), count, cp.cuda.runtime.memcpyHostToDevice, int(self._stream.ptr))
                self._stream.synchronize()
            return
        self.source_pack_backend = 'numpy'
        filled = uploaded = 0

        def flush(count):
            cp.cuda.runtime.memcpyAsync(int(self._source_gpu.data.ptr) + uploaded,
                int(self._upload_pin.ptr), count, cp.cuda.runtime.memcpyHostToDevice, int(self._stream.ptr))
            self._stream.synchronize()

        for shell, (y0, y1, x0, x1) in enumerate(self.contract.arrays['bboxes']):
            y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))
            width = x1 - x0
            if not width or y1 == y0:
                continue
            for first_row in range(y0, y1, max(1, capacity // width)):
                rows = min(y1 - first_row, max(1, capacity // width))
                if width <= capacity:
                    count = rows * width
                    if filled + count > capacity:
                        flush(filled)
                        uploaded += filled
                        filled = 0
                    pack_started = time.perf_counter()
                    np.copyto(self._upload_stage[filled:filled + count].reshape(rows, width),
                              source[shell, first_row:first_row + rows, x0:x1])
                    self.source_pack_seconds += time.perf_counter() - pack_started
                    filled += count
                else:
                    # Small explicit test budgets or unusually wide masks can
                    # split a row; concatenated bytes still follow C row order.
                    for first_col in range(x0, x1, capacity):
                        if filled:
                            flush(filled)
                            uploaded += filled
                            filled = 0
                        count = min(capacity, x1 - first_col)
                        pack_started = time.perf_counter()
                        np.copyto(self._upload_stage[:count],
                                  source[shell, first_row, first_col:first_col + count])
                        self.source_pack_seconds += time.perf_counter() - pack_started
                        filled = count
        if filled:
            flush(filled)
            uploaded += filled
        if uploaded != self.source_h2d_bytes:
            raise RuntimeError('Radial CUDA source crop upload did not match its admitted payload')

    def _reset_projection_stats(self):
        self.kernel_seconds = self.metadata_seconds = self.pack_seconds = self.d2h_seconds = 0.0
        self.metadata_d2h_bytes = self.payload_d2h_bytes = self.dense_d2h_bytes = 0
        self.cuda_graph_blocks = 0
        self.cuda_graph_seconds = 0.0
        self.empty_encoded_blocks = 0

    def _record(self, phase, boundary):
        self._events[f'{phase}_{boundary}'].record(self._stream)

    def _elapsed(self, phase):
        return float(self._cp.cuda.get_elapsed_time(
            self._events[f'{phase}_start'], self._events[f'{phase}_end'])) / 1000.0

    def _launch_projection(self, first_z, count, *, record=True):
        c, a = self.contract, self._arrays
        voxels = int(count) * int(c.output_shape[1]) * int(c.output_shape[2])
        if record:
            self._record('kernel', 'start')
        self._kernel(((voxels + 255) // 256,), (256,), (
            self._source_gpu, a['shells'], a['offsets'], a['columns'], a['sampled'],
            a['rows'], a['mapped_columns'], a['centers'], a['ideal'], a['bboxes'],
            a['source_offsets'], self._first_z_gpu, self._output_gpu,
            np.int32(c.source_shape[1]), np.int32(c.source_shape[2]), np.int32(c.native_height),
            np.int32(c.native_width), np.int32(c.stack_length), np.int32(c.height_origin),
            np.int32(c.base_id), np.int32(c.vertical), np.int32(c.plane_width),
            np.int32(c.output_shape[1]), np.int32(c.output_shape[2]), np.int32(first_z),
            np.uint64(voxels), np.int32(c.use_bboxes), np.int32(c.cropped_source),
        ), stream=self._stream)
        if record:
            self._record('kernel', 'end')
        return voxels

    def _run_block(self, first_z, count):
        voxels = self._launch_projection(first_z, count)
        self._record('copy', 'start')
        self._cp.cuda.runtime.memcpyAsync(int(self._output_pin.ptr), int(self._output_gpu.data.ptr), voxels,
            self._cp.cuda.runtime.memcpyDeviceToHost, int(self._stream.ptr))
        self._record('copy', 'end')
        self._stream.synchronize()
        self.kernel_seconds += self._elapsed('kernel')
        self.d2h_seconds += self._elapsed('copy')
        self.dense_d2h_bytes += voxels
        return self._output_stage[:count].copy()

    def _enqueue_crop_metadata(self, count, *, record=True):
        cp = self._cp
        height, width = self.contract.output_shape[1:]
        if record:
            self._record('metadata', 'start')
        self._reset_metadata_kernel(((count + 255) // 256,), (256,),
            (self._metadata_gpu, np.int32(count), np.int32(height), np.int32(width)), stream=self._stream)
        self._reduce_metadata_kernel(((width + 31) // 32, (height + 7) // 8, count), (32, 8),
            (self._output_gpu, self._metadata_gpu, np.int32(count), np.int32(height), np.int32(width)), stream=self._stream)
        if record:
            self._record('metadata', 'end')
        metadata_bytes = count * _CROP_METADATA_DTYPE.itemsize
        if record:
            self._record('copy', 'start')
        cp.cuda.runtime.memcpyAsync(int(self._metadata_pin.ptr), int(self._metadata_gpu.data.ptr), metadata_bytes,
            cp.cuda.runtime.memcpyDeviceToHost, int(self._stream.ptr))
        if record:
            self._record('copy', 'end')

    def _prepare_projection_graphs(self):
        """Capture at most three common block sizes before output publication.

        The graph joins projection, crop metadata and its pinned D2H copy. The
        variable first-Z value has an owned pinned/device scalar; no graph node
        or allocation is edited during replay. Payload compaction keeps its
        existing exact-size copy after the small metadata handshake.
        """
        cp = self._cp
        counts = {1, self.max_block_depth}
        tail = self.contract.output_shape[0] % self.max_block_depth
        if tail:
            counts.add(tail)
        try:
            for count in sorted(counts):
                self._stream.begin_capture(mode=cp.cuda.runtime.streamCaptureModeThreadLocal)
                try:
                    cp.cuda.runtime.memcpyAsync(int(self._first_z_gpu.data.ptr), int(self._first_z_pin.ptr), 4,
                        cp.cuda.runtime.memcpyHostToDevice, int(self._stream.ptr))
                    # Captured ordinary events are not portable elapsed-time
                    # markers. Time the combined replay outside the graph.
                    self._launch_projection(-1, count, record=False)
                    self._enqueue_crop_metadata(count, record=False)
                finally:
                    graph = self._stream.end_capture()
                self._graphs[count] = graph
        except Exception as exc:
            # Ending an invalidated capture normally restores the stream. If it
            # cannot be fenced, retain every owner and fail fatally as elsewhere.
            try:
                self._stream.synchronize()
            except BaseException as fence_error:
                raise RadialCudaProjectionUnsafeFailure('Could not settle Radial CUDA graph capture', self) from fence_error
            self._graphs.clear()
            self.cuda_graph_note = f'capture_unavailable:{type(exc).__name__}'
            return
        self.cuda_graph_enabled = True
        self.cuda_graph_note = 'projection_and_crop_metadata'

    def _run_projection_graph(self, first_z, count):
        self._first_z_stage[0] = first_z
        self._record('graph', 'start')
        self._graphs[count].launch(stream=self._stream)
        self._record('graph', 'end')
        self._stream.synchronize()
        self.cuda_graph_blocks += 1
        self.cuda_graph_seconds += self._elapsed('graph')

    def _encode_current_output(self, first_z, count, packed, *, metadata_ready=False):
        cp = self._cp
        height, width = self.contract.output_shape[1:]
        if not metadata_ready:
            self._enqueue_crop_metadata(count)
            self._stream.synchronize()
        metadata_bytes = count * _CROP_METADATA_DTYPE.itemsize
        if not metadata_ready:
            self.metadata_seconds += self._elapsed('metadata')
            self.d2h_seconds += self._elapsed('copy')
        self.metadata_d2h_bytes += metadata_bytes
        records, offsets, total, largest = _encoded_records(first_z, self._metadata_stage[:count],
            packed, (height, width), self.compact_buffer_bytes)
        if total == 0 and self._skip_empty_blocks:
            # Metadata is already fenced and establishes every plane as empty.
            # Do not upload offsets, launch a no-op encoder or fence it again.
            payload = np.empty(0, np.uint8)
            payload.flags.writeable = False
            self.empty_encoded_blocks += 1
            return RadialEncodedBlock(first_z, records, payload, bool(packed))
        self._offsets_stage[:count + 1] = offsets
        cp.cuda.runtime.memcpyAsync(int(self._offsets_gpu.data.ptr), int(self._offsets_pin.ptr), offsets.nbytes,
            cp.cuda.runtime.memcpyHostToDevice, int(self._stream.ptr))
        self._record('pack', 'start')
        self._encode_kernel((max(1, (largest + 255) // 256), count), (256,),
            (self._output_gpu, self._metadata_gpu, self._offsets_gpu, self._compact_gpu,
             np.int32(count), np.int32(height), np.int32(width), np.int32(packed)), stream=self._stream)
        self._record('pack', 'end')
        if total:
            self._record('copy', 'start')
            cp.cuda.runtime.memcpyAsync(int(self._output_pin.ptr), int(self._compact_gpu.data.ptr), total,
                cp.cuda.runtime.memcpyDeviceToHost, int(self._stream.ptr))
            self._record('copy', 'end')
        self._stream.synchronize()
        self.pack_seconds += self._elapsed('pack')
        if total:
            self.d2h_seconds += self._elapsed('copy')
        self.payload_d2h_bytes += total
        payload = self._output_stage.reshape(-1)[:total].copy()
        payload.flags.writeable = False
        return RadialEncodedBlock(first_z, records, payload, bool(packed))

    @staticmethod
    def _validate_encoded_preflight(dense, encoded):
        for local, record in enumerate(encoded.records):
            plane = dense[local]
            rows, cols = np.flatnonzero(np.any(plane, axis=1)), np.flatnonzero(np.any(plane, axis=0))
            bounds = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1) if len(rows) else (0, 0, 0, 0)
            if bounds != (record.y0, record.y1, record.x0, record.x1) or record.foreground != np.count_nonzero(plane):
                raise RuntimeError('Radial CUDA encoded preflight metadata differs from dense output')
            crop = plane[record.y0:record.y1, record.x0:record.x1]
            expected = np.packbits(crop, axis=1, bitorder='little') if encoded.packed else crop
            actual = encoded.payload[record.offset:record.offset + record.size]
            if not np.array_equal(actual, expected.reshape(-1)):
                raise RuntimeError('Radial CUDA encoded preflight payload differs from dense output')

    def project_encoded(self, first_z, count, packed=False):
        first_z, count = operator.index(first_z), operator.index(count)
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError('Radial CUDA projector is closed or failed')
            if first_z < 0 or count < 0 or count > self.max_block_depth or first_z + count > self.contract.output_shape[0]:
                raise ValueError('Radial CUDA output block is outside its admitted bounds')
            if count == 0:
                payload = np.empty(0, np.uint8)
                payload.flags.writeable = False
                return RadialEncodedBlock(first_z, (), payload, bool(packed))
            try:
                with self._cp.cuda.Device(self.device_index), self._cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    replay = count in self._graphs
                    if replay:
                        self._run_projection_graph(first_z, count)
                    else:
                        self._launch_projection(first_z, count)
                    result = self._encode_current_output(first_z, count, bool(packed), metadata_ready=replay)
                    if not replay:
                        self.kernel_seconds += self._elapsed('kernel')
                    return result
            except BaseException:
                self._failed = True
                raise

    def project(self, first_z, count):
        first_z, count = operator.index(first_z), operator.index(count)
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError('Radial CUDA projector is closed or failed')
            if first_z < 0 or count < 0 or count > self.max_block_depth or first_z + count > self.contract.output_shape[0]:
                raise ValueError('Radial CUDA output block is outside its admitted bounds')
            if count == 0:
                return np.empty((0, *self.contract.output_shape[1:]), np.uint8)
            try:
                with self._cp.cuda.Device(self.device_index), self._cp.cuda.using_allocator(self._pool.malloc), self._stream:
                    return self._run_block(first_z, count)
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
                    raise RadialCudaProjectionUnsafeFailure('Could not settle Radial CUDA projection stream', self) from exc
            if self._cp is not None and self._stream is not None:
                with self._cp.cuda.Device(self.device_index):
                    self._graphs.clear()
                    self._first_z_gpu = self._first_z_stage = self._first_z_pin = None
                    self._arrays.clear()
                    self._source_gpu = self._output_gpu = self._kernel = self._module = None
                    self._compact_gpu = self._metadata_gpu = self._offsets_gpu = None
                    self._reset_metadata_kernel = self._reduce_metadata_kernel = self._encode_kernel = None
                    self._events.clear()
                    self._upload_stage = self._output_stage = None
                    self._upload_pin = self._output_pin = None
                    self._metadata_stage = self._offsets_stage = None
                    self._metadata_pin = self._offsets_pin = None
                    if self._pool is not None:
                        self._pool.free_all_blocks()
                    if self._pinned_pool is not None:
                        self._pinned_pool.free_all_blocks()
            self._stream = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
