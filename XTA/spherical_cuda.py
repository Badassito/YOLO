"""Resident QSC shell rendering with bounded reusable face directions.

The native frame remains rounded gray8 before the existing full-frame/tile
affine. The explicit Torch implementation also runs on CPU for qualification;
CuPy is imported and kernels are compiled only for an actual CUDA tensor.
"""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
import math
import os
import time

import numpy as np

from .qsc import qsc_inverse
from .geometry_quality import spherical_fp32_requested, spherical_fp32_shape_eligible


_DIRECTION_CACHE_BYTES = 256 * 1024**2
_ROTATION_CACHE_ENTRIES = 32
_ROW_BLOCK = 64
_KERNELS = None
_KERNEL_ERROR = ''
_FP32_KERNELS = None
_FP32_KERNEL_ERROR = ''


def clear_spherical_render_cache(engine):
    """Drop this render engine's direction owners at its fenced release boundary."""
    cache = getattr(engine, '_spherical_direction_cache', None)
    if cache is not None:
        cache.clear()
    engine._spherical_direction_cache_bytes = 0
    rotations = getattr(engine, '_spherical_rotation_cache', None)
    if rotations is not None:
        rotations.clear()


def _direction_cache_stats(engine):
    stats = getattr(engine, '_spherical_direction_cache_stats', None)
    if stats is None:
        stats = engine._spherical_direction_cache_stats = {
            'cache_hits': 0, 'cache_misses': 0, 'cache_evictions': 0,
            'cache_host_fallbacks': 0,
            'cache_device_fallbacks': 0,
            'host_build_blocks': 0, 'host_build_seconds': 0.0,
            'upload_calls': 0, 'upload_seconds': 0.0,
            'h2d_copies': 0, 'h2d_bytes': 0,
        }
    return stats


def spherical_direction_cache_stats(engine):
    """Snapshot monotonic render counters for per-task deltas; retain no owners."""
    return dict(_direction_cache_stats(engine))


def _validate_spherical_rotation(engine, rotation):
    """Validate each immutable rotation value once in a bounded engine cache."""
    cache = getattr(engine, '_spherical_rotation_cache', None)
    if cache is None:
        cache = engine._spherical_rotation_cache = OrderedDict()
    if rotation in cache:
        cache.move_to_end(rotation)
        return
    if len(rotation) != 9 or not np.isfinite(rotation).all():
        raise ValueError('Spherical cube rotation must contain nine finite values')
    matrix = np.asarray(rotation).reshape(3, 3)
    if (not np.allclose(matrix @ matrix.T, np.eye(3), rtol=0, atol=1e-10)
            or not math.isclose(float(np.linalg.det(matrix)), 1., abs_tol=1e-10)):
        raise ValueError('Spherical cube rotation must be a proper rigid rotation')
    if len(cache) >= _ROTATION_CACHE_ENTRIES:
        cache.popitem(last=False)
    cache[rotation] = None


def _render_contract(engine, view, index):
    if str(view.family) != 'spherical':
        raise ValueError('QSC rendering requires a spherical view')
    index = int(index)
    radii = view.spherical_radii
    if index < 0 or index >= len(radii):
        raise ValueError('Spherical frame index is outside its radius trajectory')
    radius = float(radii[index])
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('Spherical radius must be finite and nonnegative')
    volume = engine._volume_gpu
    if (volume is None or volume.ndim != 3 or volume.dtype != engine.torch.uint8
            or not bool(volume.is_contiguous()) or min(volume.shape) <= 0):
        raise ValueError('Spherical rendering requires a contiguous resident uint8 volume')
    shape = (int(engine._logical_t), int(volume.shape[1]), int(volume.shape[2]))
    if shape != (int(view.full_t), int(view.full_h), int(view.full_w)) or min(shape) <= 0:
        raise ValueError('Spherical source shape does not match the logical volume')
    rows, columns = int(view.src_h), int(view.src_w)
    intervals, face = int(view.spherical_face_intervals), int(view.spherical_face)
    if min(rows, columns, intervals) <= 0 or face not in range(6):
        raise ValueError('Invalid spherical face or native raster dimensions')
    rotation = tuple(float(value) for value in view.spherical_rotation_xyz)
    _validate_spherical_rotation(engine, rotation)
    key = (str(volume.device), face, intervals, rows, columns,
           int(view.spherical_u_origin), int(view.spherical_v_origin), rotation)
    return radius, shape, key


def _build_host_direction_block(key, row0, row1):
    _, face, intervals, rows, columns, u_origin, v_origin, rotation = key
    column = u_origin + np.arange(columns, dtype=np.float64)[None, :]
    row = v_origin + np.arange(row0, row1, dtype=np.float64)[:, None]
    valid = ((column >= 0) & (column <= intervals) & (row >= 0) & (row <= intervals))
    # Padding has no geometric ownership. Clipping only makes the QSC call
    # well-defined there; the validity mask zeros every padded output sample.
    u = np.clip(-1. + 2. * column / intervals, -1., 1.)
    v = np.clip(1. - 2. * row / intervals, -1., 1.)
    directions = qsc_inverse(face, u, v) @ np.asarray(rotation).reshape(3, 3).T
    return np.ascontiguousarray(directions), np.ascontiguousarray(valid)


def _upload_direction_arrays(engine, host_directions, host_valid):
    stats = _direction_cache_stats(engine)
    started = time.perf_counter()
    on_cuda = str(engine.device).startswith('cuda')
    results = []
    try:
        ray_dtype = engine.torch.float32 if host_directions.dtype == np.float32 else engine.torch.float64
        for host, dtype in ((host_directions, ray_dtype), (host_valid, engine.torch.bool)):
            # Blocking transfers complete before the temporary host owners leave
            # scope. The caller's active render stream remains authoritative.
            results.append(engine.torch.as_tensor(host, dtype=dtype, device=engine.device))
            stats['upload_calls'] += 1
            if on_cuda:
                stats['h2d_copies'] += 1
                stats['h2d_bytes'] += host.nbytes
    finally:
        stats['upload_seconds'] += time.perf_counter() - started
    return tuple(results)


def _build_direction_block(engine, key, row0, row1, dtype=np.float64):
    started = time.perf_counter()
    host_directions, host_valid = _build_host_direction_block(key, row0, row1)
    host_directions = host_directions.astype(dtype, copy=False)
    stats = _direction_cache_stats(engine)
    stats['host_build_blocks'] += 1
    stats['host_build_seconds'] += time.perf_counter() - started
    return _upload_direction_arrays(engine, host_directions, host_valid)


def _build_cached_directions(engine, key, dtype=np.float64):
    """Build at most one cache-budget host entry, then transfer its two arrays."""
    rows, columns = key[3:5]
    started = time.perf_counter()
    host_directions = np.empty((rows, columns, 3), dtype=dtype)
    host_valid = np.empty((rows, columns), dtype=np.bool_)
    stats = _direction_cache_stats(engine)
    for row0 in range(0, rows, _ROW_BLOCK):
        row1 = min(rows, row0 + _ROW_BLOCK)
        block_directions, block_valid = _build_host_direction_block(key, row0, row1)
        host_directions[row0:row1] = block_directions
        host_valid[row0:row1] = block_valid
        stats['host_build_blocks'] += 1
        del block_directions, block_valid
    stats['host_build_seconds'] += time.perf_counter() - started
    return _upload_direction_arrays(engine, host_directions, host_valid)


def _direction_blocks(engine, key, dtype=np.float64):
    """Reuse a whole patch when it fits; otherwise build bounded uncached strips."""
    rows, columns = key[3:5]
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError('Spherical direction plans support only float32 or float64')
    pixel_bytes = 3 * dtype.itemsize + 1
    cache_key = key + (dtype.str,)
    cache = getattr(engine, '_spherical_direction_cache', None)
    if cache is None:
        cache = engine._spherical_direction_cache = OrderedDict()
        engine._spherical_direction_cache_bytes = 0
    entry = cache.get(cache_key)
    stats = _direction_cache_stats(engine)
    if entry is not None:
        stats['cache_hits'] += 1
        cache.move_to_end(cache_key)
    else:
        stats['cache_misses'] += 1
        size = rows * columns * pixel_bytes
        if size <= _DIRECTION_CACHE_BYTES:
            while cache and engine._spherical_direction_cache_bytes + size > _DIRECTION_CACHE_BYTES:
                _, (_, _, old_size) = cache.popitem(last=False)
                engine._spherical_direction_cache_bytes -= old_size
                stats['cache_evictions'] += 1
            # Keep QSC temporaries strip-bounded. Only the final direction and
            # validity arrays occupy a full, cache-budget-limited host entry.
            try:
                directions, valid = _build_cached_directions(engine, key, dtype=dtype)
            except (MemoryError, engine.torch.OutOfMemoryError) as exc:
                # Either staging arena can fail independently. Never publish
                # partial entries; retain the bounded-strip sampler on OOM.
                stats['cache_host_fallbacks' if isinstance(exc, MemoryError)
                      else 'cache_device_fallbacks'] += 1
            else:
                entry = (directions, valid, size)
                cache[cache_key] = entry
                engine._spherical_direction_cache_bytes += size
    block_rows = max(1, min(_ROW_BLOCK, max(1, _DIRECTION_CACHE_BYTES // (columns * pixel_bytes))))
    if entry is None:
        for row0 in range(0, rows, block_rows):
            row1 = min(rows, row0 + block_rows)
            directions, valid = _build_direction_block(engine, key, row0, row1, dtype=dtype)
            yield row0, directions, valid
    else:
        # CUDA can consume the cached plan in one launch. The reference sampler
        # uses bounded strips for its eight-tap intermediates below.
        yield 0, entry[0], entry[1]


def _render_spherical_native_torch(engine, view, index):
    radius, shape, key = _render_contract(engine, view, index)
    torch = engine.torch
    volume = engine._volume_gpu
    source_flat = volume.reshape(-1)
    logical_t, height, width = shape
    native_indices = engine._native_t_indices(logical_t) if int(volume.shape[0]) != logical_t else None
    output = torch.empty((int(view.src_h), int(view.src_w)), dtype=torch.float32, device=engine.device)

    def gather(t, y, x):
        spatial = y * width + x
        stride = height * width
        if native_indices is None:
            return torch.take(source_flat, t * stride + spatial).to(torch.float32)
        r0, r1, alpha = native_indices
        f0 = torch.take(source_flat, r0[t] * stride + spatial).to(torch.float32)
        f1 = torch.take(source_flat, r1[t] * stride + spatial).to(torch.float32)
        # Match the Radial logical integer source, including its gray8 boundary.
        return (f0 + alpha[t] * (f1 - f0)).round_().clamp_(0., 255.)

    for first_row, directions, valid in _direction_blocks(engine, key):
        for local0 in range(0, len(directions), _ROW_BLOCK):
            local1 = min(len(directions), local0 + _ROW_BLOCK)
            rays = directions[local0:local1]
            padding_valid = valid[local0:local1]
            axis_taps = []
            for component, length in ((2, logical_t), (1, height), (0, width)):
                coordinate = ((length - 1) * .5 + radius * rays[..., component]).clamp(-1., float(length))
                lower = torch.floor(coordinate).to(torch.int64)
                fraction = (coordinate - lower).to(torch.float32)
                taps = []
                for offset, weight in ((0, 1. - fraction), (1, fraction)):
                    at = lower + offset
                    weight = weight * ((at >= 0) & (at < length))
                    taps.append((at.clamp(0, length - 1), weight))
                axis_taps.append(taps)
            values = torch.zeros_like(axis_taps[0][0][1])
            for ti, tw in axis_taps[0]:
                for yi, yw in axis_taps[1]:
                    for xi, xw in axis_taps[2]:
                        values.add_(gather(ti, yi, xi) * ((tw * yw) * xw))
            output[first_row + local0:first_row + local1] = values.masked_fill_(~padding_valid, 0.).round_().clamp_(0., 255.)
    return output


# The logical-voxel reconstruction and the eight rounded float32 tap operations
# intentionally match cuda_backend's radial_native_f32. Geometry alone differs.
_KERNEL_SOURCE = r'''
__device__ __forceinline__ int spherical_clip(int value, int length) {
    return value < 0 ? 0 : (value >= length ? length - 1 : value);
}
__device__ __forceinline__ float spherical_logical_voxel(
    const unsigned char* source, int t, int y, int x,
    int native_t, int logical_t, int height, int width) {
    const unsigned long long stride = (unsigned long long)height * width;
    const unsigned long long spatial = (unsigned long long)y * width + x;
    if (native_t == logical_t) return (float)source[(unsigned long long)t * stride + spatial];
    double rf = __dadd_rn(__dmul_rn(__dadd_rn((double)t, 0.5),
                                   __ddiv_rn((double)native_t, (double)logical_t)), -0.5);
    int t0 = spherical_clip((int)floor(rf), native_t);
    int t1 = spherical_clip(t0 + 1, native_t);
    float alpha = __double2float_rn(fmin(1.0, fmax(0.0, rf - (double)t0)));
    float f0 = (float)source[(unsigned long long)t0 * stride + spatial];
    float f1 = (float)source[(unsigned long long)t1 * stride + spatial];
    float value = __fadd_rn(f0, __fmul_rn(alpha, __fsub_rn(f1, f0)));
    return (float)__float2uint_rn(fminf(255.0f, fmaxf(0.0f, value)));
}
extern "C" __global__ void spherical_native_f32(
    const unsigned char* source, const double* directions, const bool* valid,
    float* output, int native_t, int logical_t, int height, int width,
    unsigned long long count, unsigned long long output_offset, double radius) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count) return;
    if (!valid[q]) { output[output_offset + q] = 0.0f; return; }
    double tt = __dadd_rn((double)(logical_t - 1) * 0.5, __dmul_rn(radius, directions[3*q+2]));
    double yy = __dadd_rn((double)(height - 1) * 0.5, __dmul_rn(radius, directions[3*q+1]));
    double xx = __dadd_rn((double)(width - 1) * 0.5, __dmul_rn(radius, directions[3*q]));
    if (!(tt > -1.0 && tt < (double)logical_t && yy > -1.0 && yy < (double)height
          && xx > -1.0 && xx < (double)width)) {
        output[output_offset + q] = 0.0f; return;
    }
    int t0 = (int)floor(tt), y0 = (int)floor(yy), x0 = (int)floor(xx);
    float dt = __double2float_rn(tt - (double)t0);
    float dy = __double2float_rn(yy - (double)y0);
    float dx = __double2float_rn(xx - (double)x0);
    float value = 0.0f;
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int ti = t0 + it;
        float wt = it ? dt : __fsub_rn(1.0f, dt);
        #pragma unroll
        for (int iy = 0; iy < 2; ++iy) {
            int yi = y0 + iy;
            float wy = iy ? dy : __fsub_rn(1.0f, dy);
            #pragma unroll
            for (int ix = 0; ix < 2; ++ix) {
                int xi = x0 + ix;
                float wx = ix ? dx : __fsub_rn(1.0f, dx);
                if (ti >= 0 && ti < logical_t && yi >= 0 && yi < height && xi >= 0 && xi < width) {
                    float voxel = spherical_logical_voxel(source, ti, yi, xi, native_t, logical_t, height, width);
                    value = __fadd_rn(value, __fmul_rn(voxel, __fmul_rn(__fmul_rn(wt, wy), wx)));
                }
            }
        }
    }
    output[output_offset + q] = (float)__float2uint_rn(fminf(255.0f, fmaxf(0.0f, value)));
}
'''


def _spherical_kernels():
    global _KERNELS, _KERNEL_ERROR
    if _KERNELS is not None:
        return _KERNELS
    if _KERNEL_ERROR:
        raise RuntimeError(_KERNEL_ERROR)
    try:
        import cupy as cp
        module = cp.RawModule(code=_KERNEL_SOURCE, options=('--std=c++11', '--fmad=false'))
        _KERNELS = SimpleNamespace(cp=cp, module=module, kernel=module.get_function('spherical_native_f32'))
        return _KERNELS
    except Exception as exc:
        _KERNEL_ERROR = f'Spherical CuPy/NVRTC kernel unavailable: {type(exc).__name__}: {exc}'
        raise RuntimeError(_KERNEL_ERROR) from exc


def _render_spherical_native_cuda(engine, view, index):
    from .geometry import _cupy_external_stream
    radius, shape, key = _render_contract(engine, view, index)
    kernels = _spherical_kernels()
    output = engine.torch.empty((int(view.src_h), int(view.src_w)), dtype=engine.torch.float32, device=engine.device)
    source_ref = engine._fused_cupy_volume(kernels)
    output_ref = kernels.cp.asarray(output)
    stream = _cupy_external_stream(kernels.cp, engine._stream)
    for row0, directions, valid in _direction_blocks(engine, key):
        # Typed cache entries can be evicted while this reference kernel is
        # queued on the render stream. Track those CuPy reads explicitly even
        # when the plans/output were allocated on another Torch stream.
        for tensor in (directions, valid, output):
            if bool(tensor.is_cuda):
                tensor.record_stream(engine._stream)
        count = int(valid.numel())
        kernels.kernel(
            ((count + 255) // 256,), (256,),
            (source_ref, kernels.cp.asarray(directions), kernels.cp.asarray(valid), output_ref,
             np.int32(engine._volume_gpu.shape[0]), np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
             np.uint64(count), np.uint64(row0 * int(view.src_w)), np.float64(radius)), stream=stream,
        )
    return output


def _spherical_fp32_kernels():
    global _FP32_KERNELS, _FP32_KERNEL_ERROR
    if _FP32_KERNELS is not None:
        return _FP32_KERNELS
    if _FP32_KERNEL_ERROR:
        raise RuntimeError(_FP32_KERNEL_ERROR)
    try:
        import cupy as cp
        from .spherical_sampling_cuda import SPHERICAL_FP32_SOURCE
        module = cp.RawModule(code=SPHERICAL_FP32_SOURCE, options=('--std=c++11', '--fmad=true'))
        _FP32_KERNELS = SimpleNamespace(cp=cp, module=module,
                                       kernel=module.get_function('spherical_virtual_fp32'))
        return _FP32_KERNELS
    except Exception as exc:
        _FP32_KERNEL_ERROR = f'Spherical FP32 kernel unavailable: {type(exc).__name__}: {exc}'
        raise RuntimeError(_FP32_KERNEL_ERROR) from exc


def _spherical_fp32_eligible(engine):
    volume = engine._volume_gpu
    return spherical_fp32_shape_eligible((int(volume.shape[0]), int(engine._logical_t),
                                         int(volume.shape[1]), int(volume.shape[2])))


def _render_spherical_fp32_cuda(engine, view, index):
    """Sample the same virtual gray8 cube using FP32 coordinates and FMA."""
    from .geometry import _cupy_external_stream
    from .geometry_quality import SPHERICAL_FP32_BACKEND
    from .unification.contracts import DataRole
    from .unification.sampling import require_forward_sampling
    radius, shape, key = _render_contract(engine, view, index)
    if not _spherical_fp32_eligible(engine):
        raise ValueError('Spherical FP32 sampling requires source axes in [1,4096]')
    require_forward_sampling(SPHERICAL_FP32_BACKEND, DataRole.INTENSITY)
    kernels = _spherical_fp32_kernels()
    output = engine.torch.empty((int(view.src_h), int(view.src_w)), dtype=engine.torch.float32, device=engine.device)
    source_ref = engine._fused_cupy_volume(kernels)
    output_ref = kernels.cp.asarray(output)
    stream = _cupy_external_stream(kernels.cp, engine._stream)
    for row0, directions, valid in _direction_blocks(engine, key, dtype=np.float32):
        # Cache entries may be evicted before queued CuPy readers finish, and
        # callers can allocate on a different Torch stream than the renderer.
        for tensor in (directions, valid, output):
            if bool(tensor.is_cuda):
                tensor.record_stream(engine._stream)
        count = int(valid.numel())
        kernels.kernel(
            ((count + 255) // 256,), (256,),
            (source_ref, kernels.cp.asarray(directions), kernels.cp.asarray(valid), output_ref,
             np.int32(engine._volume_gpu.shape[0]), np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
             np.uint64(count), np.uint64(row0 * int(view.src_w)), np.float32(radius)), stream=stream,
        )
    return output


def render_spherical_native_resident(engine, view, index):
    """Render one radius into float32 gray8 on the engine's active render stream."""
    volume = engine._volume_gpu
    if not bool(getattr(volume, 'is_cuda', False)):
        engine._spherical_sampler_mode = 'reference_torch'
        return _render_spherical_native_torch(engine, view, index)
    enabled = os.environ.get('YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL', '1').strip().lower() not in ('', '0', 'false', 'off', 'no')
    disabled = bool(getattr(engine, '_spherical_native_kernel_disabled', False))
    if (enabled and not disabled and spherical_fp32_requested()
            and not bool(getattr(engine, '_spherical_fp32_disabled', False))):
        if _spherical_fp32_eligible(engine):
            try:
                output = _render_spherical_fp32_cuda(engine, view, index)
                engine._spherical_sampler_mode = 'fp32_virtual_cube'
                if not bool(getattr(engine, '_spherical_fp32_announced', False)):
                    engine._spherical_fp32_announced = True
                    print('Spherical FP32 CUDA sampler active: FP32 directions/FMA with virtual-cube gray8 rounding retained.', flush=True)
                return output
            except Exception as exc:
                engine._spherical_fp32_disabled = True
                print(f'Warning: Spherical FP32 sampler unavailable ({type(exc).__name__}: {exc}); using the FP64 reference sampler.', flush=True)
        elif not bool(getattr(engine, '_spherical_fp32_shape_warned', False)):
            engine._spherical_fp32_shape_warned = True
            print('Spherical FP32 sampler outside its qualified source-axis range [1,4096]; using the FP64 reference sampler.', flush=True)
    if enabled and not disabled:
        try:
            output = _render_spherical_native_cuda(engine, view, index)
            engine._spherical_sampler_mode = 'reference_fp64'
            if not bool(getattr(engine, '_spherical_native_kernel_announced', False)):
                engine._spherical_native_kernel_announced = True
                print('Spherical native CUDA renderer active: cached QSC directions and one source-sampling launch per patch.', flush=True)
            return output
        except Exception as exc:
            engine._spherical_native_kernel_disabled = True
            reason = f'{type(exc).__name__}: {exc}'
    else:
        reason = 'disabled after an earlier failure' if disabled else 'YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL=0'
    if not bool(getattr(engine, '_spherical_native_fallback_warned', False)):
        engine._spherical_native_fallback_warned = True
        print(f'Warning: Spherical native CUDA kernel unavailable ({reason}); using the resident Torch reference renderer.', flush=True)
    engine._spherical_sampler_mode = 'reference_torch'
    return _render_spherical_native_torch(engine, view, index)
