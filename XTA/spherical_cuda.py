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

import numpy as np

from .qsc import qsc_inverse


_DIRECTION_CACHE_BYTES = 256 * 1024**2
_ROW_BLOCK = 64
_KERNELS = None
_KERNEL_ERROR = ''


def clear_spherical_render_cache(engine):
    """Drop this render engine's direction owners at its fenced release boundary."""
    cache = getattr(engine, '_spherical_direction_cache', None)
    if cache is not None:
        cache.clear()
    engine._spherical_direction_cache_bytes = 0


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
    if len(rotation) != 9 or not np.isfinite(rotation).all():
        raise ValueError('Spherical cube rotation must contain nine finite values')
    matrix = np.asarray(rotation).reshape(3, 3)
    if (not np.allclose(matrix @ matrix.T, np.eye(3), rtol=0, atol=1e-10)
            or not math.isclose(float(np.linalg.det(matrix)), 1., abs_tol=1e-10)):
        raise ValueError('Spherical cube rotation must be a proper rigid rotation')
    key = (str(volume.device), face, intervals, rows, columns,
           int(view.spherical_u_origin), int(view.spherical_v_origin), rotation)
    return radius, shape, key


def _build_direction_block(engine, key, row0, row1):
    _, face, intervals, rows, columns, u_origin, v_origin, rotation = key
    column = u_origin + np.arange(columns, dtype=np.float64)[None, :]
    row = v_origin + np.arange(row0, row1, dtype=np.float64)[:, None]
    valid = ((column >= 0) & (column <= intervals) & (row >= 0) & (row <= intervals))
    # Padding has no geometric ownership. Clipping only makes the QSC call
    # well-defined there; the validity mask zeros every padded output sample.
    u = np.clip(-1. + 2. * column / intervals, -1., 1.)
    v = np.clip(1. - 2. * row / intervals, -1., 1.)
    directions = qsc_inverse(face, u, v) @ np.asarray(rotation).reshape(3, 3).T
    return (
        engine.torch.as_tensor(np.ascontiguousarray(directions),
                               dtype=engine.torch.float64, device=engine.device),
        engine.torch.as_tensor(np.ascontiguousarray(valid),
                               dtype=engine.torch.bool, device=engine.device),
    )


def _direction_blocks(engine, key):
    """Reuse a whole patch when it fits; otherwise build bounded uncached strips."""
    rows, columns = key[3:5]
    cache = getattr(engine, '_spherical_direction_cache', None)
    if cache is None:
        cache = engine._spherical_direction_cache = OrderedDict()
        engine._spherical_direction_cache_bytes = 0
    entry = cache.get(key)
    if entry is not None:
        cache.move_to_end(key)
    else:
        size = rows * columns * (3 * 8 + 1)
        if size <= _DIRECTION_CACHE_BYTES:
            while cache and engine._spherical_direction_cache_bytes + size > _DIRECTION_CACHE_BYTES:
                _, (_, _, old_size) = cache.popitem(last=False)
                engine._spherical_direction_cache_bytes -= old_size
            directions = engine.torch.empty((rows, columns, 3), dtype=engine.torch.float64, device=engine.device)
            valid = engine.torch.empty((rows, columns), dtype=engine.torch.bool, device=engine.device)
            # QSC's temporary arrays are limited to a strip even when retaining
            # a whole patch on the GPU. Never build a full-face host mesh.
            for row0 in range(0, rows, _ROW_BLOCK):
                row1 = min(rows, row0 + _ROW_BLOCK)
                block_directions, block_valid = _build_direction_block(engine, key, row0, row1)
                directions[row0:row1].copy_(block_directions)
                valid[row0:row1].copy_(block_valid)
            entry = (directions, valid, size)
            cache[key] = entry
            engine._spherical_direction_cache_bytes += size
    block_rows = max(1, min(_ROW_BLOCK, max(1, _DIRECTION_CACHE_BYTES // (columns * 25))))
    if entry is None:
        for row0 in range(0, rows, block_rows):
            row1 = min(rows, row0 + block_rows)
            directions, valid = _build_direction_block(engine, key, row0, row1)
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
        count = int(valid.numel())
        kernels.kernel(
            ((count + 255) // 256,), (256,),
            (source_ref, kernels.cp.asarray(directions), kernels.cp.asarray(valid), output_ref,
             np.int32(engine._volume_gpu.shape[0]), np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
             np.uint64(count), np.uint64(row0 * int(view.src_w)), np.float64(radius)), stream=stream,
        )
    return output


def render_spherical_native_resident(engine, view, index):
    """Render one radius into float32 gray8 on the engine's active render stream."""
    volume = engine._volume_gpu
    if not bool(getattr(volume, 'is_cuda', False)):
        return _render_spherical_native_torch(engine, view, index)
    enabled = os.environ.get('YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL', '1').strip().lower() not in ('', '0', 'false', 'off', 'no')
    disabled = bool(getattr(engine, '_spherical_native_kernel_disabled', False))
    if enabled and not disabled:
        try:
            output = _render_spherical_native_cuda(engine, view, index)
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
    return _render_spherical_native_torch(engine, view, index)
