"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np
from ._deps import cv2

from .config import (
    DEFAULT_CHANNEL_FORMAT,
)

def open_existing_gray_memmap(path: object, shape: Sequence[int], dtype: object = np.uint8, mode: str = 'r') -> np.memmap:
    return np.memmap(Path(path), dtype=np.dtype(dtype), mode=str(mode), shape=tuple(int(x) for x in shape))

def union_conf_volume_into_volume_inplace(
    dst_mask_mm: np.ndarray,
    dst_conf_mm: Optional[np.ndarray],
    src_mask_mm: np.ndarray,
    src_conf_mm: Optional[np.ndarray],
    *,
    workers: int = 1,
    desc: str = 'Union view-volume contributions',
) -> None:
    """Union one prediction-volume result into the per-view union.

 When confidence maps are present (``--min_conf`` > 0) this performs a per-pixel
 maximum-confidence union (the surviving pixel keeps the higher confidence), matching the
 in-thread accumulation semantics. Otherwise it is a plain binary OR."""
    num_slices = int(dst_mask_mm.shape[0]) if int(dst_mask_mm.ndim) > 0 else 0
    if num_slices <= 0:
        return
    use_conf = bool(dst_conf_mm is not None and src_conf_mm is not None)

    def _merge_slice(idx: int) -> None:
        i = int(idx)
        src_mask = np.asarray(src_mask_mm[i], dtype=np.uint8)
        if not use_conf:
            dst_mask_mm[i, :, :] |= src_mask
            return
        dst_mask = np.asarray(dst_mask_mm[i], dtype=np.uint8)
        dst_conf = np.asarray(dst_conf_mm[i], dtype=np.uint8)
        src_conf = np.asarray(src_conf_mm[i], dtype=np.uint8)
        # A source pixel wins where it is foreground and strictly more confident than the
        # current destination (ties keep the existing pixel, matching first-writer-wins).
        take = (src_mask > 0) & (src_conf > dst_conf)
        if np.any(take):
            dst_mask = np.where(take, src_mask, dst_mask)
            dst_conf = np.where(take, src_conf, dst_conf)
        # Foreground from the source that lands where the destination is empty is also unioned.
        add = (src_mask > 0) & (dst_mask == 0)
        if np.any(add):
            dst_mask = np.where(add, src_mask, dst_mask)
            dst_conf = np.where(add, src_conf, dst_conf)
        dst_mask_mm[i, :, :] = dst_mask
        dst_conf_mm[i, :, :] = dst_conf

    parallel_for_indices(
        num_slices,
        _merge_slice,
        max_workers=choose_slice_parallel_workers(int(workers), num_slices),
        desc=desc,
        show_progress=False,
    )
    flush_array(dst_mask_mm)
    if dst_conf_mm is not None:
        flush_array(dst_conf_mm)

def gpu_worker_render_enabled() -> bool:
    """Render eligible full-frame views on the worker GPU."""
    return _env_flag('YOLO_TTA_GPU_RENDER', True)

def gpu_worker_render_resident_enabled() -> bool:
    """Allow a full source-volume upload when VRAM admission succeeds."""
    return _env_flag('YOLO_TTA_GPU_RENDER_RESIDENT', True)

def gpu_render_reserve_bytes() -> int:
    """VRAM headroom that must remain free AFTER a resident source-volume upload."""
    return int(max(1.0, _env_float('YOLO_TTA_GPU_RENDER_RESERVE_GIB', 12.0)) * GIB)

def gpu_render_tblock_slices() -> int:
    """Transient source t-block size for streaming-mode GPU radial prerendering."""
    return max(16, _env_int('YOLO_TTA_GPU_RENDER_TBLOCK_SLICES', 256))

def gpu_cube_resize_enabled() -> bool:
    """Fold eligible T-axis cube scaling into resident GPU renderers.

    The host cube stays deferred until a CPU, tile, or nonresident fallback requests it.
    """
    return _env_flag('YOLO_TTA_GPU_CUBE_RESIZE', True)

def fused_direct_render_enabled() -> bool:
    """Allow resident-ring renderers to write normalized pixels straight to TRT bindings."""
    return _env_flag('YOLO_TTA_FUSED_DIRECT_RENDER', True)

def fused_radial_render_enabled() -> bool:
    """Use the Radial direct-to-binding kernel when enabled."""
    return fused_direct_render_enabled() and _env_flag('YOLO_TTA_FUSED_RADIAL_RENDER', True)

def fused_tilted_render_enabled() -> bool:
    """Use the Tilted direct-to-binding kernel when enabled."""
    return fused_direct_render_enabled() and _env_flag('YOLO_TTA_FUSED_TILTED_RENDER', True)

def fused_render_cuda_graphs_enabled() -> bool:
    """Capture the stable render kernel; metadata remains a dynamic device-side input."""
    return (
        fused_direct_render_enabled()
        and resident_trt_cuda_graphs_enabled()
        and _env_flag('YOLO_TTA_FUSED_RENDER_CUDA_GRAPHS', True)
    )

def radial_texture_source_copy_reserve_enabled() -> bool:
    """Reserve a second source-volume allocation when admitting GPU residency."""
    return _env_flag('YOLO_TTA_RADIAL_TEXTURE_SOURCE_COPY_RESERVE', True)

_FUSED_DIRECT_RENDER_KERNELS: Optional[object] = None

_FUSED_DIRECT_RENDER_KERNELS_FAILED = False

_FUSED_DIRECT_RENDER_KERNELS_ERROR: Optional[str] = None

_FUSED_DIRECT_RENDER_KERNELS_WARNED = False

def _fused_direct_render_kernels() -> Optional[object]:
    """Compile the allocation-free resident Radial and Tilted render kernels once.
    
    Explicit fp32/fp16 entry points are materialized eagerly so toolchain failures retain a complete diagnostic."""
    global _FUSED_DIRECT_RENDER_KERNELS, _FUSED_DIRECT_RENDER_KERNELS_FAILED
    global _FUSED_DIRECT_RENDER_KERNELS_ERROR, _FUSED_DIRECT_RENDER_KERNELS_WARNED
    if _FUSED_DIRECT_RENDER_KERNELS is not None:
        return _FUSED_DIRECT_RENDER_KERNELS
    if _FUSED_DIRECT_RENDER_KERNELS_FAILED:
        return None
    src = r'''
    #include <cuda_fp16.h>

    __device__ __forceinline__ int clamp_i(int v, int lo, int hi) {
      return v < lo ? lo : (v > hi ? hi : v);
    }
    __device__ __forceinline__ float clamp_f(float v, float lo, float hi) {
      return v < lo ? lo : (v > hi ? hi : v);
    }
    // Match align_corners=False + zero padding when the composed output affine lands
    // within one bilinear-kernel radius of a Radial plane edge.  The native two-stage
    // renderer blends the edge texel with zero in this fringe; a hard [0,N-1] reject
    // creates a sparse but high-amplitude seam after square padding/scaling.
    __device__ __forceinline__ float zero_padded_linear_coord(
        float coord, int size, float* clamped_coord) {
      if (size <= 0 || coord <= -1.0f || coord >= (float)size) {
        *clamped_coord = 0.0f;
        return 0.0f;
      }
      float last = (float)(size - 1);
      if (coord < 0.0f) {
        *clamped_coord = 0.0f;
        return coord + 1.0f;
      }
      if (coord > last) {
        *clamped_coord = last;
        return (float)size - coord;
      }
      *clamped_coord = coord;
      return 1.0f;
    }
    // The independent Torch Radial tap builder masks invalid taps and renormalizes the
    // surviving triangle weights.  Coordinates less than one pixel beyond the projected
    // source plane therefore clamp to the nearest edge texel rather than becoming zero.
    __device__ __forceinline__ bool renormalized_linear_coord(
        float coord, int size, float* clamped_coord) {
      if (size <= 0 || coord <= -1.0f || coord >= (float)size) {
        *clamped_coord = 0.0f;
        return false;
      }
      *clamped_coord = clamp_f(coord, 0.0f, (float)(size - 1));
      return true;
    }
    __device__ __forceinline__ int floor_i(float v) {
      return __float2int_rd(v);
    }
    __device__ __forceinline__ float round_nearest_f(float v) {
      return (float)__float2int_rn(v);
    }
    // Header-free device math: run 126080 showed NVRTC 13.0 could not locate
    // the host system math.h. CUDA device math functions need no host header path.
    __device__ __forceinline__ float norm_u8(float value) {
      return clamp_f(value, 0.0f, 255.0f) * (1.0f / 255.0f);
    }

    extern "C" __global__ void set_render_meta(int* render_meta, int frame_value) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        render_meta[0] = frame_value;
        render_meta[1] = 0;
      }
    }

    __device__ __forceinline__ float logical_to_native_coord(
        float logical_coord, int native_t, int logical_t) {
      float mapped = native_t == logical_t
          ? logical_coord
          : ((logical_coord + 0.5f) * ((float)native_t / (float)logical_t)) - 0.5f;
      return clamp_f(mapped, 0.0f, (float)(native_t - 1));
    }

    __device__ __forceinline__ float radial_row_to_stack(
        float row, int rows, int stack_len) {
      if (rows == stack_len) return row;
      return clamp_f(
          ((row + 0.5f) * ((float)stack_len / (float)rows)) - 0.5f,
          0.0f, (float)(stack_len - 1));
    }

    __device__ __forceinline__ int radial_plane_width(
        int base_id, int full_h, int full_w) {
      return base_id == 2 ? full_h : full_w;
    }

    __device__ __forceinline__ int radial_plane_height(
        int base_id, int full_h, int logical_t) {
      return base_id == 0 ? full_h : logical_t;
    }

    __device__ __forceinline__ float radial_source_texture(
        cudaTextureObject_t volume_tex,
        int native_t, int full_h, int full_w, int logical_t,
        int base_id, float plane_x, float plane_y, float stack) {
      float source_x, source_y, source_t;
      if (base_id == 0) {
        source_x = plane_x;
        source_y = plane_y;
        source_t = logical_to_native_coord(stack, native_t, logical_t);
      } else if (base_id == 1) {
        source_x = plane_x;
        source_y = stack;
        source_t = logical_to_native_coord(plane_y, native_t, logical_t);
      } else {
        source_x = stack;
        source_y = plane_x;
        source_t = logical_to_native_coord(plane_y, native_t, logical_t);
      }
      return tex3D<float>(
          volume_tex, source_x + 0.5f, source_y + 0.5f, source_t + 0.5f);
    }

    __device__ __forceinline__ float radial_texture_sample(
        cudaTextureObject_t volume_tex,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        float tan_tilt, float center_x, float center_y, float roi_radius,
        int angle_idx, float radial_y, float radial_x,
        const float* angle_cos, const float* angle_sin) {
      float clamped_radial_x, clamped_radial_y;
      float border_x = zero_padded_linear_coord(
          radial_x, n_u, &clamped_radial_x);
      float border_y = zero_padded_linear_coord(
          radial_y, rows, &clamped_radial_y);
      float output_border_weight = border_x * border_y;
      if (output_border_weight <= 0.0f) return 0.0f;
      radial_x = clamped_radial_x;
      radial_y = clamped_radial_y;
      float line = n_u > 1
          ? -roi_radius + (2.0f * roi_radius) * (radial_x / (float)(n_u - 1))
          : -roi_radius;
      float px = center_x + line * angle_cos[angle_idx];
      float py = center_y + line * angle_sin[angle_idx];
      int plane_w = radial_plane_width(base_id, full_h, full_w);
      int plane_h = radial_plane_height(base_id, full_h, logical_t);
      float clamped_px, clamped_py;
      if (!renormalized_linear_coord(px, plane_w, &clamped_px)
          || !renormalized_linear_coord(py, plane_h, &clamped_py)) return 0.0f;
      px = clamped_px;
      py = clamped_py;
      float stack = radial_row_to_stack(radial_y, rows, stack_len);
      if (tan_tilt != 0.0f) {
        float axis = direction_id == 0 ? py - center_y : px - center_x;
        stack = __fadd_rn(stack, __fmul_rn(tan_tilt, axis));
      }
      if (stack < 0.0f || stack > (float)(stack_len - 1)) return 0.0f;
      return output_border_weight * radial_source_texture(
          volume_tex, native_t, full_h, full_w, logical_t,
          base_id, px, py, stack);
    }

    __device__ __forceinline__ float radial_texture_direct_value(
        cudaTextureObject_t volume_tex,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12,
        const float* angle_cos, const float* angle_sin, int q) {
      int angle_idx = render_meta[0];
      int oy = q / ow;
      int ox = q - oy * ow;
      float radial_x = __fadd_rn(
          __fadd_rn(__fmul_rn(m00, (float)ox), __fmul_rn(m01, (float)oy)), m02);
      float radial_y = __fadd_rn(
          __fadd_rn(__fmul_rn(m10, (float)ox), __fmul_rn(m11, (float)oy)), m12);
      return radial_texture_sample(
          volume_tex, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id,
          tan_tilt, center_x, center_y, roi_radius,
          angle_idx, radial_y, radial_x, angle_cos, angle_sin);
    }

    extern "C" __global__ void radial_texture_direct_f32(
        cudaTextureObject_t volume_tex,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12,
        const float* angle_cos, const float* angle_sin, float* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= oh * ow) return;
      out[q] = clamp_f(radial_texture_direct_value(
          volume_tex, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id, render_meta, tan_tilt,
          center_x, center_y, roi_radius, oh, ow,
          m00, m01, m02, m10, m11, m12, angle_cos, angle_sin, q), 0.0f, 1.0f);
    }

    extern "C" __global__ void radial_texture_direct_f16(
        cudaTextureObject_t volume_tex,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12,
        const float* angle_cos, const float* angle_sin, __half* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= oh * ow) return;
      float value = clamp_f(radial_texture_direct_value(
          volume_tex, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id, render_meta, tan_tilt,
          center_x, center_y, roi_radius, oh, ow,
          m00, m01, m02, m10, m11, m12, angle_cos, angle_sin, q), 0.0f, 1.0f);
      out[q] = __float2half_rn(value);
    }

    extern "C" __global__ void radial_texture_native_f32(
        cudaTextureObject_t volume_tex,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        int angle_idx, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        const float* angle_cos, const float* angle_sin, float* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= rows * n_u) return;
      int row = q / n_u;
      int u = q - row * n_u;
      float value = radial_texture_sample(
          volume_tex, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id,
          tan_tilt, center_x, center_y, roi_radius,
          angle_idx, (float)row, (float)u, angle_cos, angle_sin);
      out[q] = clamp_f(value, 0.0f, 1.0f) * 255.0f;
    }

    extern "C" __global__ void radial_texture_grid_f32(
        cudaTextureObject_t volume_tex,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        int angle_idx, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12,
        const float* angle_cos, const float* angle_sin, float* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= oh * ow) return;
      int oy = q / ow;
      int ox = q - oy * ow;
      float radial_x = __fadd_rn(
          __fadd_rn(__fmul_rn(m00, (float)ox), __fmul_rn(m01, (float)oy)), m02);
      float radial_y = __fadd_rn(
          __fadd_rn(__fmul_rn(m10, (float)ox), __fmul_rn(m11, (float)oy)), m12);
      float value = radial_texture_sample(
          volume_tex, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id,
          tan_tilt, center_x, center_y, roi_radius,
          angle_idx, radial_y, radial_x, angle_cos, angle_sin);
      out[q] = clamp_f(value, 0.0f, 1.0f) * 255.0f;
    }

    // A3: canonical-pointer Radial sampling. X/Y/stack are nearest-neighbor while
    // the decoded native T axis remains linearly interpolated. This reads the same
    // resident uint8 allocation used by Cartesian/Tilted renderers and avoids a second
    // full-volume CUDA array/texture copy.
    __device__ __forceinline__ float radial_source_pointer(
        const unsigned char* volume,
        int native_t, int full_h, int full_w, int logical_t,
        int base_id, float plane_x, float plane_y, float stack) {
      float source_x, source_y, source_t;
      if (base_id == 0) {
        source_x = plane_x;
        source_y = plane_y;
        source_t = logical_to_native_coord(stack, native_t, logical_t);
      } else if (base_id == 1) {
        source_x = plane_x;
        source_y = stack;
        source_t = logical_to_native_coord(plane_y, native_t, logical_t);
      } else {
        source_x = stack;
        source_y = plane_x;
        source_t = logical_to_native_coord(plane_y, native_t, logical_t);
      }
      int x = __float2int_rn(source_x);
      int y = __float2int_rn(source_y);
      if (x < 0 || x >= full_w || y < 0 || y >= full_h) return 0.0f;
      int t0 = clamp_i(floor_i(source_t), 0, native_t - 1);
      int t1 = min(native_t - 1, t0 + 1);
      float alpha = clamp_f(source_t - (float)t0, 0.0f, 1.0f);
      long long plane_stride = (long long)full_h * (long long)full_w;
      long long spatial = (long long)y * (long long)full_w + (long long)x;
      float a = (float)volume[(long long)t0 * plane_stride + spatial];
      float b = (float)volume[(long long)t1 * plane_stride + spatial];
      return norm_u8(a + alpha * (b - a));
    }

    __device__ __forceinline__ float radial_pointer_sample(
        const unsigned char* volume,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        float tan_tilt, float center_x, float center_y, float roi_radius,
        int angle_idx, float radial_y, float radial_x,
        const float* angle_cos, const float* angle_sin) {
      float clamped_radial_x, clamped_radial_y;
      float border_x = zero_padded_linear_coord(radial_x, n_u, &clamped_radial_x);
      float border_y = zero_padded_linear_coord(radial_y, rows, &clamped_radial_y);
      float output_border_weight = border_x * border_y;
      if (output_border_weight <= 0.0f) return 0.0f;
      radial_x = clamped_radial_x;
      radial_y = clamped_radial_y;
      float line = n_u > 1
          ? -roi_radius + (2.0f * roi_radius) * (radial_x / (float)(n_u - 1))
          : -roi_radius;
      float px = center_x + line * angle_cos[angle_idx];
      float py = center_y + line * angle_sin[angle_idx];
      int plane_w = radial_plane_width(base_id, full_h, full_w);
      int plane_h = radial_plane_height(base_id, full_h, logical_t);
      float clamped_px, clamped_py;
      if (!renormalized_linear_coord(px, plane_w, &clamped_px)
          || !renormalized_linear_coord(py, plane_h, &clamped_py)) return 0.0f;
      px = clamped_px;
      py = clamped_py;
      float stack_coord = radial_row_to_stack(radial_y, rows, stack_len);
      if (tan_tilt != 0.0f) {
        float axis = direction_id == 0 ? py - center_y : px - center_x;
        stack_coord = __fadd_rn(stack_coord, __fmul_rn(tan_tilt, axis));
      }
      if (stack_coord < 0.0f || stack_coord > (float)(stack_len - 1)) return 0.0f;
      return output_border_weight * radial_source_pointer(
          volume, native_t, full_h, full_w, logical_t,
          base_id, px, py, stack_coord);
    }

    __device__ __forceinline__ float radial_pointer_direct_value(
        const unsigned char* volume,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12,
        const float* angle_cos, const float* angle_sin, int q) {
      int angle_idx = render_meta[0];
      int oy = q / ow;
      int ox = q - oy * ow;
      float radial_x = __fadd_rn(
          __fadd_rn(__fmul_rn(m00, (float)ox), __fmul_rn(m01, (float)oy)), m02);
      float radial_y = __fadd_rn(
          __fadd_rn(__fmul_rn(m10, (float)ox), __fmul_rn(m11, (float)oy)), m12);
      return radial_pointer_sample(
          volume, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id,
          tan_tilt, center_x, center_y, roi_radius,
          angle_idx, radial_y, radial_x, angle_cos, angle_sin);
    }

    extern "C" __global__ void radial_pointer_direct_f32(
        const unsigned char* volume,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12,
        const float* angle_cos, const float* angle_sin, float* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= oh * ow) return;
      out[q] = clamp_f(radial_pointer_direct_value(
          volume, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id, render_meta, tan_tilt,
          center_x, center_y, roi_radius, oh, ow,
          m00, m01, m02, m10, m11, m12, angle_cos, angle_sin, q), 0.0f, 1.0f);
    }

    extern "C" __global__ void radial_pointer_direct_f16(
        const unsigned char* volume,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12,
        const float* angle_cos, const float* angle_sin, __half* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= oh * ow) return;
      float value = clamp_f(radial_pointer_direct_value(
          volume, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id, render_meta, tan_tilt,
          center_x, center_y, roi_radius, oh, ow,
          m00, m01, m02, m10, m11, m12, angle_cos, angle_sin, q), 0.0f, 1.0f);
      out[q] = __float2half_rn(value);
    }

    extern "C" __global__ void radial_pointer_native_f32(
        const unsigned char* volume,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        int angle_idx, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        const float* angle_cos, const float* angle_sin, float* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= rows * n_u) return;
      int row = q / n_u;
      int u = q - row * n_u;
      float value = radial_pointer_sample(
          volume, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id,
          tan_tilt, center_x, center_y, roi_radius,
          angle_idx, (float)row, (float)u, angle_cos, angle_sin);
      out[q] = clamp_f(value, 0.0f, 1.0f) * 255.0f;
    }

    extern "C" __global__ void radial_pointer_grid_f32(
        const unsigned char* volume,
        int native_t, int full_h, int full_w, int logical_t,
        int rows, int n_u, int stack_len, int base_id, int direction_id,
        int angle_idx, float tan_tilt,
        float center_x, float center_y, float roi_radius,
        int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12,
        const float* angle_cos, const float* angle_sin, float* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= oh * ow) return;
      int oy = q / ow;
      int ox = q - oy * ow;
      float radial_x = __fadd_rn(
          __fadd_rn(__fmul_rn(m00, (float)ox), __fmul_rn(m01, (float)oy)), m02);
      float radial_y = __fadd_rn(
          __fadd_rn(__fmul_rn(m10, (float)ox), __fmul_rn(m11, (float)oy)), m12);
      float value = radial_pointer_sample(
          volume, native_t, full_h, full_w, logical_t,
          rows, n_u, stack_len, base_id, direction_id,
          tan_tilt, center_x, center_y, roi_radius,
          angle_idx, radial_y, radial_x, angle_cos, angle_sin);
      out[q] = clamp_f(value, 0.0f, 1.0f) * 255.0f;
    }

    // Pointer-based helpers remain for the non-radial Tilted renderer.
    __device__ __forceinline__ void logical_t_taps(
        int logical_idx, int native_t, int logical_t, int* t0, int* t1, float* alpha) {
      float pos = ((float)logical_idx + 0.5f) * ((float)native_t / (float)logical_t) - 0.5f;
      int raw0 = floor_i(pos);
      *t0 = clamp_i(raw0, 0, native_t - 1);
      *t1 = *t0 + 1 < native_t ? *t0 + 1 : native_t - 1;
      *alpha = clamp_f(pos - (float)(*t0), 0.0f, 1.0f);
    }

    __device__ __forceinline__ float rounded_t_lerp(
        const unsigned char* volume, long long plane_stride, long long spatial,
        int t0, int t1, float alpha) {
      float a = (float)volume[(long long)t0 * plane_stride + spatial];
      float b = (float)volume[(long long)t1 * plane_stride + spatial];
      return clamp_f(round_nearest_f(a + alpha * (b - a)), 0.0f, 255.0f);
    }

    // Native tilted raster value at one integer in-plane tap. ``axis_coord`` is the
    // coordinate along the tilt axis used for the shear: the legacy nearest path passes
    // the CONTINUOUS affine coordinate (preserving v16.1.7 output exactly), while the
    // bilinear path passes each tap's own integer coordinate, which makes the 4-tap
    // blend identical to rendering the native frame and warping it bilinearly.
    __device__ __forceinline__ float tilted_native_value(
        const unsigned char* volume, int native_t, int full_h, int full_w, int logical_t,
        int src_h, int src_w, int stack_len, int base_id, int direction_id,
        int frame_center, float tan_tilt, int x, int y, float axis_coord) {
      if (x < 0 || x >= src_w || y < 0 || y >= src_h) return 0.0f;
      float axis = direction_id == 0
          ? axis_coord - 0.5f * (float)(src_h - 1)
          : axis_coord - 0.5f * (float)(src_w - 1);
      float stack = __fadd_rn((float)frame_center, __fmul_rn(tan_tilt, axis));
      if (stack < 0.0f || stack > (float)(stack_len - 1)) return 0.0f;
      int s0 = clamp_i(floor_i(stack), 0, stack_len - 1);
      int s1 = s0 + 1 < stack_len ? s0 + 1 : stack_len - 1;
      float sa = stack - (float)s0;
      long long ps = (long long)full_h * (long long)full_w;
      float v0, v1;
      if (base_id == 0) {
        long long spatial = (long long)y * (long long)full_w + (long long)x;
        if (native_t == logical_t) {
          v0 = (float)volume[(long long)s0 * ps + spatial];
          v1 = (float)volume[(long long)s1 * ps + spatial];
        } else {
          int a0, a1, b0, b1; float aa, ba;
          logical_t_taps(s0, native_t, logical_t, &a0, &a1, &aa);
          logical_t_taps(s1, native_t, logical_t, &b0, &b1, &ba);
          v0 = rounded_t_lerp(volume, ps, spatial, a0, a1, aa);
          v1 = rounded_t_lerp(volume, ps, spatial, b0, b1, ba);
        }
      } else {
        int t0, t1; float ta;
        logical_t_taps(y, native_t, logical_t, &t0, &t1, &ta);
        long long spatial0, spatial1;
        if (base_id == 1) {
          spatial0 = (long long)s0 * (long long)full_w + (long long)x;
          spatial1 = (long long)s1 * (long long)full_w + (long long)x;
        } else {
          spatial0 = (long long)x * (long long)full_w + (long long)s0;
          spatial1 = (long long)x * (long long)full_w + (long long)s1;
        }
        if (native_t == logical_t) {
          v0 = (float)volume[(long long)y * ps + spatial0];
          v1 = (float)volume[(long long)y * ps + spatial1];
        } else {
          v0 = rounded_t_lerp(volume, ps, spatial0, t0, t1, ta);
          v1 = rounded_t_lerp(volume, ps, spatial1, t0, t1, ta);
        }
      }
      return v0 + sa * (v1 - v0);
    }

    __device__ __forceinline__ float tilted_direct_value(
        const unsigned char* volume, int native_t, int full_h, int full_w, int logical_t,
        int src_h, int src_w, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt, int inplane_linear, int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12, int q) {
      int frame_center = render_meta[0];
      int oy = q / ow, ox = q - oy * ow;
      float sx = __fadd_rn(__fadd_rn(__fmul_rn(m00, (float)ox), __fmul_rn(m01, (float)oy)), m02);
      float sy = __fadd_rn(__fadd_rn(__fmul_rn(m10, (float)ox), __fmul_rn(m11, (float)oy)), m12);
      if (!inplane_linear) {
        int x = __float2int_rn(sx);
        int y = __float2int_rn(sy);
        return tilted_native_value(volume, native_t, full_h, full_w, logical_t,
            src_h, src_w, stack_len, base_id, direction_id, frame_center, tan_tilt,
            x, y, direction_id == 0 ? sy : sx);
      }
      // v16.1.8 forward-pass bilinear: match align_corners=False zero-padded warp
      // semantics on the native tilted raster (the same contract the Cartesian
      // grid_sample warp and the radial kernels' edge handling use).
      float cx, cy;
      float border_x = zero_padded_linear_coord(sx, src_w, &cx);
      float border_y = zero_padded_linear_coord(sy, src_h, &cy);
      float border = border_x * border_y;
      if (border <= 0.0f) return 0.0f;
      int x0 = floor_i(cx);
      int y0 = floor_i(cy);
      int x1 = x0 + 1 < src_w ? x0 + 1 : src_w - 1;
      int y1 = y0 + 1 < src_h ? y0 + 1 : src_h - 1;
      float fx = cx - (float)x0;
      float fy = cy - (float)y0;
      float v00 = tilted_native_value(volume, native_t, full_h, full_w, logical_t,
          src_h, src_w, stack_len, base_id, direction_id, frame_center, tan_tilt,
          x0, y0, direction_id == 0 ? (float)y0 : (float)x0);
      float v01 = tilted_native_value(volume, native_t, full_h, full_w, logical_t,
          src_h, src_w, stack_len, base_id, direction_id, frame_center, tan_tilt,
          x1, y0, direction_id == 0 ? (float)y0 : (float)x1);
      float v10 = tilted_native_value(volume, native_t, full_h, full_w, logical_t,
          src_h, src_w, stack_len, base_id, direction_id, frame_center, tan_tilt,
          x0, y1, direction_id == 0 ? (float)y1 : (float)x0);
      float v11 = tilted_native_value(volume, native_t, full_h, full_w, logical_t,
          src_h, src_w, stack_len, base_id, direction_id, frame_center, tan_tilt,
          x1, y1, direction_id == 0 ? (float)y1 : (float)x1);
      float v_top = v00 + fx * (v01 - v00);
      float v_bottom = v10 + fx * (v11 - v10);
      return border * (v_top + fy * (v_bottom - v_top));
    }

    extern "C" __global__ void tilted_direct_f32(
        const unsigned char* volume, int native_t, int full_h, int full_w, int logical_t,
        int src_h, int src_w, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt, int inplane_linear, int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12, float* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= oh * ow) return;
      out[q] = norm_u8(tilted_direct_value(volume, native_t, full_h, full_w, logical_t,
          src_h, src_w, stack_len, base_id, direction_id, render_meta, tan_tilt,
          inplane_linear, oh, ow, m00, m01, m02, m10, m11, m12, q));
    }

    extern "C" __global__ void tilted_direct_f16(
        const unsigned char* volume, int native_t, int full_h, int full_w, int logical_t,
        int src_h, int src_w, int stack_len, int base_id, int direction_id,
        const int* render_meta, float tan_tilt, int inplane_linear, int oh, int ow,
        float m00, float m01, float m02, float m10, float m11, float m12, __half* out) {
      int q = (int)(blockDim.x * blockIdx.x + threadIdx.x);
      if (q >= oh * ow) return;
      float value = norm_u8(tilted_direct_value(volume, native_t, full_h, full_w, logical_t,
          src_h, src_w, stack_len, base_id, direction_id, render_meta, tan_tilt,
          inplane_linear, oh, ow, m00, m01, m02, m10, m11, m12, q));
      out[q] = __float2half_rn(value);
    }
    '''
    try:
        import cupy as cp  # type: ignore
        names = (
            'set_render_meta',
            'radial_texture_direct_f32', 'radial_texture_direct_f16',
            'radial_texture_native_f32', 'radial_texture_grid_f32',
            'radial_pointer_direct_f32', 'radial_pointer_direct_f16',
            'radial_pointer_native_f32', 'radial_pointer_grid_f32',
            'tilted_direct_f32', 'tilted_direct_f16',
        )
        module = cp.RawModule(code=src, options=('--std=c++14',))
        compile_fn = getattr(module, 'compile', None)
        if callable(compile_fn):
            compile_fn()
        functions = {name: module.get_function(name) for name in names}
        _FUSED_DIRECT_RENDER_KERNELS = argparse.Namespace(cp=cp, module=module, **functions)
        _FUSED_DIRECT_RENDER_KERNELS_ERROR = None
        return _FUSED_DIRECT_RENDER_KERNELS
    except Exception as exc:
        _FUSED_DIRECT_RENDER_KERNELS_FAILED = True
        try:
            import hashlib
            source_id = hashlib.sha256(src.encode('utf-8')).hexdigest()[:16]
        except Exception:
            source_id = 'unknown'
        details = [f'{type(exc).__name__}: {exc}', f'cuda_source_sha256={source_id}']
        try:
            import cupy as cp  # type: ignore
            details.append(f'cupy={getattr(cp, "__version__", "unknown")}')
            details.append(f'cuda_runtime={cp.cuda.runtime.runtimeGetVersion()}')
            try:
                details.append(f'nvrtc={cp.cuda.nvrtc.getVersion()}')
            except Exception:
                pass
            try:
                details.append(f'compute_capability={cp.cuda.Device().compute_capability}')
            except Exception:
                pass
        except Exception:
            pass
        _FUSED_DIRECT_RENDER_KERNELS_ERROR = '; '.join(details)
        dump_path = os.environ.get('YOLO_TTA_FUSED_RENDER_DUMP_CUDA', '').strip()
        if dump_path:
            try:
                Path(dump_path).expanduser().write_text(src)
            except Exception:
                pass
        if not _FUSED_DIRECT_RENDER_KERNELS_WARNED:
            _FUSED_DIRECT_RENDER_KERNELS_WARNED = True
            print(
                'Warning: fused Radial/Tilted NVRTC module failed to compile; '
                f'{_FUSED_DIRECT_RENDER_KERNELS_ERROR}. '
                'The reference Torch renderers remain active.'
            )
        return None

def _wait_for_cube_ready_sentinel(
    sentinel_path: str,
    *,
    request_path: Optional[str] = None,
    failed_path: Optional[str] = None,
    poll_seconds: float = 0.5,
) -> None:
    """Wait for the deferred host cube to publish its ready or failure sentinel."""
    sentinel = Path(str(sentinel_path))
    request = Path(str(request_path)) if request_path else None
    failed = Path(str(failed_path)) if failed_path else None
    if request is not None:
        # this is the first proven need for the host cube. A tiny atomic
        # marker wakes the main-process materializer; all workers share the same result.
        request.parent.mkdir(parents=True, exist_ok=True)
        request.touch(exist_ok=True)
    announced = False
    timeout_seconds = max(
        60.0,
        _env_float('YOLO_TTA_LAZY_CUBE_WAIT_TIMEOUT_SEC', 3600.0),
    )
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        # Failure wins if both records somehow exist; publishes them transactionally,
        # but this also makes stale/corrupt shared state fail closed.
        if failed is not None and failed.exists():
            try:
                detail = failed.read_text().strip()
            except Exception:
                detail = 'unknown producer failure'
            raise RuntimeError(f'shared cube fallback construction failed: {detail}')
        if sentinel.exists():
            return
        if time.monotonic() >= float(deadline):
            raise TimeoutError(
                'timed out waiting for the deferred shared cube after '
                f'{float(timeout_seconds):.0f}s; ready={sentinel}, failed={failed}'
            )
        if not announced:
            announced = True
            print(
                'GPU worker: requested the deferred shared cube and is waiting for '
                'file-backed fallback rendering...'
            )
        time.sleep(max(0.05, float(poll_seconds)))

def fused_renderer_preflight_enabled() -> bool:
    """Fail-fast worker probe for fused upright-Radial, Tilted, and tilted-Radial kernels."""
    return _env_flag('YOLO_TTA_FUSED_RENDER_PREFLIGHT', True)

def fused_renderer_fail_fast_enabled() -> bool:
    """Treat an enabled fused-render launch/probe failure as worker-fatal."""
    return _env_flag('YOLO_TTA_FUSED_RENDER_FAIL_FAST', True)

def fused_renderer_preflight_tolerances() -> Tuple[float, float, float, float, float]:
    """Return high-error, mean, mismatch, and fractional preflight tolerances.

    A raw maximum is retained for diagnostics, but one nearest-neighbor tie or texture
    interpolation outlier among millions of pixels is not independently worker-fatal.
    ``max_abs_fraction`` bounds the share exceeding ``max_abs`` instead. The validator
    may raise the default Radial hardware-texture limit to one raster-perimeter-equivalent
    seam; an explicit environment override remains exact.
    """
    return (
        max(0.0, _env_float('YOLO_TTA_FUSED_PREFLIGHT_MAX_ABS', 16.0 / 255.0)),
        max(0.0, min(1.0, _env_float(
            'YOLO_TTA_FUSED_PREFLIGHT_MAX_ABS_FRACTION', 0.001,
        ))),
        max(0.0, _env_float('YOLO_TTA_FUSED_PREFLIGHT_MEAN_ABS', 2.0 / 255.0)),
        max(0.0, _env_float('YOLO_TTA_FUSED_PREFLIGHT_MISMATCH_ABS', 4.0 / 255.0)),
        max(0.0, min(1.0, _env_float('YOLO_TTA_FUSED_PREFLIGHT_MISMATCH_FRACTION', 0.02))),
    )

def _fused_preflight_family(view: ViewInfo) -> str:
    if is_tilted_radial_view(view):
        return 'tilted_radial'
    if is_radial_view(view):
        return 'radial'
    if is_tilted_view(view):
        return 'tilted'
    return ''

def _single_pixel_closed_seam_fraction(height: int, width: int) -> float:
    """Return the fraction occupied by a one-pixel closed seam around an HxW raster.

    A fused hardware-texture Radial launch composes source sampling and output resampling in
    one kernel, while the independent startup reference renders a native Radial plane and then
    resamples it. Their expected numerical disagreement can concentrate along one resampling
    seam even when the image-wide mean and broader mismatch rates remain negligible.
    """
    h = max(1, int(height))
    w = max(1, int(width))
    pixels = int(h) * int(w)
    if pixels <= 1 or h <= 1 or w <= 1:
        return 1.0
    seam_pixels = (2 * int(h)) + (2 * int(w)) - 4
    return min(1.0, max(0.0, float(seam_pixels) / float(pixels)))

def fused_renderer_effective_max_fraction_tolerance(
    configured_tolerance: float,
    *,
    preflight_family: str,
    height: int,
    width: int,
) -> Tuple[float, float]:
    """Return the effective high-error fraction limit and its automatic seam floor.

    The one-pixel floor applies only to the default hardware-texture Radial comparison. An
    explicit ``YOLO_TTA_FUSED_PREFLIGHT_MAX_ABS_FRACTION`` remains authoritative, and the
    independent mean-error and 4/255 mismatch-fraction limits are never relaxed.
    """
    configured = max(0.0, min(1.0, float(configured_tolerance)))
    if os.environ.get('YOLO_TTA_FUSED_PREFLIGHT_MAX_ABS_FRACTION', '').strip():
        return configured, 0.0
    if str(preflight_family) not in ('radial', 'tilted_radial'):
        return configured, 0.0
    if radial_source_mode() != 'texture_linear':
        return configured, 0.0
    seam_floor = _single_pixel_closed_seam_fraction(int(height), int(width))
    return max(configured, float(seam_floor)), float(seam_floor)

_GPU_WORKER_FUSED_PREFLIGHT_SPECS: Tuple[Dict[str, object], ...] = ()

def set_gpu_worker_fused_preflight_specs(specs: Optional[Sequence[Dict[str, object]]]) -> None:
    global _GPU_WORKER_FUSED_PREFLIGHT_SPECS
    _GPU_WORKER_FUSED_PREFLIGHT_SPECS = tuple(dict(spec) for spec in (specs or ()))


def gpu_worker_fused_preflight_specs() -> Tuple[Dict[str, object], ...]:
    """Return the process-local fused-render preflight specifications."""

    return _GPU_WORKER_FUSED_PREFLIGHT_SPECS

class _GpuWorkerRenderEngine:
    """Per-worker-process GPU renderer for full-frame prediction sources."""

    def __init__(self, device_str: str = 'cuda:0') -> None:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        self.torch = torch
        self.F = F
        self.device = torch.device(str(device_str))
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('GPU render engine requires a CUDA device')
        # A worker owns exactly one inference/render GPU. Keep CuPy/NVRTC's implicit
        # current device aligned with the explicit Torch device used below.
        torch.cuda.set_device(self.device)
        # All renders run on a dedicated LOW-priority stream; the resident batch-1 ring
        # gives TensorRT distinct high-priority streams so frame n inference can preempt/
        # overlap frame n+1 rendering. Consumers order through per-slot events.
        self._stream = torch.cuda.Stream(
            device=self.device, priority=_cuda_stream_priority(torch, high=False),
        )
        self._volume_key: Optional[Tuple[str, Tuple[int, int, int], int]] = None
        self._volume_mm: Optional[np.ndarray] = None
        self._volume_gpu: Optional[object] = None
        self._volume_flat: Optional[object] = None
        # _volume_gpu keeps NATIVE t while _logical_t is the
        # approximately-cubic working t seen by ViewInfo/render coordinates.
        self._logical_t = 0
        self._native_t_map_cache: Dict[Tuple[int, int], Tuple[object, object, object]] = {}
        self._mode = 'unresolved'
        self._tilted_plans: 'OrderedDict[Tuple[str, int, int, Tuple[float, ...]], Dict[str, object]]' = OrderedDict()
        self._fold_cache: Dict[Tuple[int, int, int], Tuple[object, object, object]] = {}
        # Small per-azimuth sin/cos geometry plus one optional 3D texture object.
        self._fused_radial_taps: 'OrderedDict[object, object]' = OrderedDict()
        self._fused_volume_ref: Optional[object] = None
        self._radial_texture_ref: Optional[object] = None
        self._radial_texture_lock = threading.RLock()
        self._fused_disabled_families: set = set()
        self._fused_warned_families: set = set()
        self._fused_announced_families: set = set()
        self._fused_graph_announced_families: set = set()
        self._fused_graph_warned_families: set = set()
        self._fused_graph_rejected_keys: set = set()
        self._fused_validated_keys: set = set()
        self._fused_preflight_validated_families: set[str] = set()
        self._fused_preflight_volume_key: Optional[Tuple[str, Tuple[int, int, int], int]] = None
        self._warned_fallback = False
        self._resident_runtime_disabled = False
        # Native planes reused within one angle-local tile source across batches and
        # contextual channel indices; no cross-tile canvas is retained.
        self._native_plane_cache: 'OrderedDict[Tuple[str, int], object]' = OrderedDict()
        self._tilted_plan_cache_floor = 0

    # volume residency ----

    def ensure_volume(
        self,
        path: str,
        shape: Sequence[int],
        dtype: str = 'uint8',
        *,
        resize_to_t: Optional[int] = None,
        require_radial_texture: bool = False,
    ) -> str:
        """Resolve resident or streaming GPU source-volume mode.
        
        Eligible runs retain native T and defer host-cube construction until a CPU, tile, or nonresident fallback requests it."""
        torch = self.torch
        shape_t = tuple(int(x) for x in shape)
        if len(shape_t) != 3:
            raise ValueError(f'GPU render source must be 3D (t,Y,X), got {shape_t}')
        in_t, in_h, in_w = (int(shape_t[0]), int(shape_t[1]), int(shape_t[2]))
        out_t = int(resize_to_t) if resize_to_t else int(in_t)
        if min(in_t, in_h, in_w, out_t) <= 0:
            raise ValueError(f'GPU render source has invalid native/logical shape {shape_t} -> t={out_t}')
        resize_active = int(out_t) != int(in_t)
        key = (str(path), shape_t, int(out_t))
        if self._volume_key == key and self._mode != 'unresolved':
            return self._mode
        if self._volume_key is not None and self._volume_key != key:
            # Texture objects and angle tables can be referenced by queued CuPy
            # launches/graphs. Retire the old render stream before dropping owners.
            try:
                self._stream.synchronize()
            except Exception:
                pass
        self._volume_key = key
        self._volume_mm = np.memmap(Path(str(path)), dtype=np.dtype(str(dtype)), mode='c', shape=shape_t)
        self._volume_gpu = None
        self._volume_flat = None
        self._logical_t = int(out_t)
        self._native_t_map_cache.clear()
        self._native_plane_cache.clear()
        self._fold_cache.clear()
        self._tilted_plans.clear()
        self._fused_radial_taps.clear()
        self._fused_volume_ref = None
        self._radial_texture_ref = None
        self._fused_disabled_families.clear()
        self._fused_graph_rejected_keys.clear()
        self._fused_validated_keys.clear()
        self._fused_preflight_validated_families.clear()
        self._fused_preflight_volume_key = None
        self._mode = 'stream'
        # Residency admission uses the native buffer, not the inflated logical cube.
        nbytes = int(in_t) * int(in_h) * int(in_w)
        if gpu_worker_render_resident_enabled() and not self._resident_runtime_disabled:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                texture_copy_bytes = (
                    int(nbytes)
                    if (
                        bool(require_radial_texture)
                        and radial_source_mode() == 'texture_linear'
                        and radial_texture_source_copy_reserve_enabled()
                    )
                    else 0
                )
                need = nbytes + texture_copy_bytes + gpu_render_reserve_bytes()
                if int(free_bytes) >= int(need):
                    vol = torch.empty((int(in_t), int(in_h), int(in_w)), dtype=torch.uint8, device=self.device)
                    chunk = 256
                    for t0 in range(0, int(in_t), chunk):
                        t1 = min(int(in_t), t0 + chunk)
                        vol[t0:t1].copy_(torch.from_numpy(np.ascontiguousarray(self._volume_mm[t0:t1])))
                    self._volume_gpu = vol
                    self._volume_flat = vol.view(-1)
                    # Build/cache the integer logical-t taps now; transverse and
                    # Cartesian planes use them directly and tilted plans reuse the
                    # same center-aligned convention.
                    self._native_t_indices(int(out_t))
                    self._mode = 'resident'
                else:
                    print(
                        f'GPU render: source volume NOT resident ({nbytes / GIB:.1f} GiB source + '
                        f'{texture_copy_bytes / GIB:.1f} GiB radial-texture copy + '
                        f'{gpu_render_reserve_bytes() / GIB:.1f} GiB reserve > '
                        f'{free_bytes / GIB:.1f} GiB free); '
                        'only upright transverse-Radial tasks retain streamed GPU prerendering. '
                        'Tilted-Radial, Cartesian, and other unsupported nonresident views use CPU rendering. '
                        'A TensorRT engine rebuilt at max batch 1 frees enough VRAM for residency.'
                    )
            except Exception as exc:
                self._volume_gpu = None
                self._volume_flat = None
                self._mode = 'stream'
                print(f'GPU render: resident upload failed ({exc}); falling back to streaming mode.')
        if resize_active and self._mode != 'resident':
            # never leave a native-geometry memmap where streaming-mode
            # consumers (radial slab prerender, shape probes) expect the cube.
            self._volume_key = None
            self._volume_mm = None
            self._logical_t = 0
            self._native_t_map_cache.clear()
            self._mode = 'unresolved'
            return 'stream'
        if self._mode == 'resident':
            resize_note = (
                f' (v13.3.9 E3: native t retained; renderers map {in_t}->{out_t} on device)'
                if resize_active else ''
            )
            print(
                f'GPU render: source volume resident on {self.device} '
                f'({nbytes / GIB:.1f} GiB){resize_note}; full-frame views render on device.'
            )
        return self._mode

    def disable_resident_after_runtime_failure(self) -> None:
        """Release residency and keep this worker on completed-cube fallbacks.

 Rendering is lazy, so a gather-plan error or late OOM can occur after prediction has
 started. Releasing the large resident tensor gives a clean CPU-render retry enough VRAM
 for inference and avoids repeating the same failing resident path on later tasks."""
        self._resident_runtime_disabled = True
        try:
            self._stream.synchronize()
        except Exception:
            pass
        self._volume_gpu = None
        self._volume_flat = None
        self._volume_mm = None
        self._volume_key = None
        self._logical_t = 0
        self._native_t_map_cache.clear()
        self._fold_cache.clear()
        self._tilted_plans.clear()
        self._fused_radial_taps.clear()
        self._fused_volume_ref = None
        self._radial_texture_ref = None
        self._fused_disabled_families.clear()
        self._fused_graph_rejected_keys.clear()
        self._fused_validated_keys.clear()
        self._fused_preflight_validated_families.clear()
        self._fused_preflight_volume_key = None
        self._mode = 'unresolved'
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def _native_t_indices(self, logical_t: Optional[int] = None) -> Tuple[object, object, object]:
        """Integer logical-t planes -> native two-tap indices/weights on device."""
        torch = self.torch
        native_t = int(self._volume_gpu.shape[0])
        logical = int(self._logical_t if logical_t is None else logical_t)
        key = (native_t, logical)
        cached = self._native_t_map_cache.get(key)
        if cached is not None:
            return cached
        rf = (np.arange(logical, dtype=np.float64) + 0.5) * (float(native_t) / float(logical)) - 0.5
        r0 = np.clip(np.floor(rf).astype(np.int64), 0, native_t - 1)
        r1 = np.minimum(r0 + 1, native_t - 1)
        alpha = np.clip(rf - r0, 0.0, 1.0).astype(np.float32)
        mapped = (
            torch.from_numpy(r0).to(self.device),
            torch.from_numpy(r1).to(self.device),
            torch.from_numpy(alpha).to(self.device),
        )
        self._native_t_map_cache[key] = mapped
        return mapped

    def _resample_native_t_axis(self, values: object) -> object:
        """Map a native-t-leading tensor to logical t with endpoint-aligned u8 rounding."""
        torch = self.torch
        native_t = int(values.shape[0])
        logical_t = int(self._logical_t)
        if native_t == logical_t:
            return values.to(torch.float32)
        r0, r1, alpha = self._native_t_indices(logical_t)
        alpha_shape = [logical_t] + [1] * max(0, int(values.ndim) - 1)
        a = alpha.view(*alpha_shape)
        f0 = values.index_select(0, r0).to(torch.float32)
        f1 = values.index_select(0, r1).to(torch.float32)
        # The old resident cube stored uint8 after cv2-compatible lerp.
        return torch.lerp(f0, f1, a).round_().clamp_(0.0, 255.0)

    # fused resident-ring renderers ----

    def _fused_render_fallback(self, family: str, exc: object) -> None:
        """Disable one optional fused family after a synchronous capability failure."""
        family_s = str(family)
        self._fused_disabled_families.add(family_s)
        if family_s not in self._fused_warned_families:
            self._fused_warned_families.add(family_s)
            if family_s == 'tilted_radial':
                gate = (
                    'YOLO_TTA_FUSED_RADIAL_RENDER=0 disables only the direct kernel; '
                    'YOLO_TTA_GPU_TILTED_RADIAL_RENDER=0 restores the v16.0.2 CPU path'
                )
            elif family_s == 'radial':
                gate = 'YOLO_TTA_FUSED_RADIAL_RENDER=0'
            else:
                gate = 'YOLO_TTA_FUSED_TILTED_RENDER=0'
            print(
                f'P4 fused {family_s} renderer unavailable ({exc}); using the reference '
                f'Torch renderer for this worker. {gate} disables the capability probe.'
            )

    def _fused_cupy_volume(self, kernels: object) -> object:
        with self._radial_texture_lock:
            if self._fused_volume_ref is None:
                self._fused_volume_ref = kernels.cp.asarray(self._volume_gpu)
            return self._fused_volume_ref

    def _ensure_radial_texture(self, kernels: object) -> object:
        """Create one normalized-float, hardware-linear 3D texture for the resident u8 source.

        CUDA arrays are allocated outside the Torch/CuPy memory pools, so construction is
        single-flight and performs an explicit free-memory admission check before allocation.
        """
        cached = self._radial_texture_ref
        if cached is not None:
            return cached
        with self._radial_texture_lock:
            cached = self._radial_texture_ref
            if cached is not None:
                return cached
            if self._volume_gpu is None or self._volume_gpu.dtype != self.torch.uint8:
                raise RuntimeError('radial texture requires a resident uint8 source volume')
            if not bool(self._volume_gpu.is_contiguous()):
                raise RuntimeError('radial texture requires a contiguous source volume')

            cp = kernels.cp
            native_t, full_h, full_w = (int(v) for v in self._volume_gpu.shape)
            texture_bytes = int(native_t) * int(full_h) * int(full_w)
            free_bytes, _total = self.torch.cuda.mem_get_info(self.device)
            texture_headroom = int(
                max(0.5, _env_float('YOLO_TTA_RADIAL_TEXTURE_RESERVE_GIB', 2.0)) * GIB
            )
            if int(free_bytes) < int(texture_bytes) + int(texture_headroom):
                raise RuntimeError(
                    f'3D radial texture needs {texture_bytes / GIB:.2f} GiB plus '
                    f'{texture_headroom / GIB:.2f} GiB headroom, only '
                    f'{int(free_bytes) / GIB:.2f} GiB free'
                )

            runtime = cp.cuda.runtime
            with cp.cuda.Device(int(getattr(self.device, 'index', 0) or 0)):
                channel = cp.cuda.texture.ChannelFormatDescriptor(
                    8, 0, 0, 0, runtime.cudaChannelFormatKindUnsigned,
                )
                cuda_array = cp.cuda.texture.CUDAarray(
                    channel, int(full_w), int(full_h), int(native_t),
                )
                source_ref = self._fused_cupy_volume(kernels)
                external = _cupy_external_stream(cp, self._stream)
                cuda_array.copy_from(source_ref, stream=external)
                resource = cp.cuda.texture.ResourceDescriptor(
                    runtime.cudaResourceTypeArray, cuArr=cuda_array,
                )
                descriptor = cp.cuda.texture.TextureDescriptor(
                    addressModes=(
                        runtime.cudaAddressModeBorder,
                        runtime.cudaAddressModeBorder,
                        runtime.cudaAddressModeBorder,
                    ),
                    filterMode=runtime.cudaFilterModeLinear,
                    readMode=runtime.cudaReadModeNormalizedFloat,
                    normalizedCoords=0,
                )
                texture = cp.cuda.texture.TextureObject(resource, descriptor)
                # CUDAarray storage is outside CuPy's pool. Complete the one-time D2D copy so
                # allocation/copy failures surface before any graph captures the texture handle.
                self._stream.synchronize()
            cached = argparse.Namespace(
                source_ref=source_ref,
                channel=channel,
                cuda_array=cuda_array,
                resource=resource,
                descriptor=descriptor,
                texture=texture,
                nbytes=texture_bytes,
                shape=(native_t, full_h, full_w),
            )
            self._radial_texture_ref = cached
            print(
                f'Radial hardware texture allocated on {self.device}: '
                f'{native_t}x{full_h}x{full_w} u8 ({texture_bytes / GIB:.2f} GiB), '
                f'filter={RADIAL_FILTER_LABEL}.'
            )
            return cached

    def _fused_slot_output(self, slot: _ResidentGpuPipelineSlot, kernels: object) -> object:
        ref = slot._render_cupy_refs.get('input_plane')
        ptr = int(slot.input.data_ptr())
        if ref is None or int(slot._render_cupy_refs.get('input_ptr', -1)) != ptr:
            ref = kernels.cp.asarray(slot.input[0, 0])
            slot._render_cupy_refs['input_plane'] = ref
            slot._render_cupy_refs['input_ptr'] = ptr
        return ref

    def _fused_slot_metadata(
        self,
        slot: _ResidentGpuPipelineSlot,
        kernels: object,
        frame_value: Optional[int],
    ) -> object:
        """Update dynamic frame metadata without changing its graph-friendly device address."""
        ref = slot._render_cupy_refs.get('render_meta')
        ptr = int(slot.render_meta.data_ptr())
        if ref is None or int(slot._render_cupy_refs.get('render_meta_ptr', -1)) != ptr:
            ref = kernels.cp.asarray(slot.render_meta)
            slot._render_cupy_refs['render_meta'] = ref
            slot._render_cupy_refs['render_meta_ptr'] = ptr
        if frame_value is not None:
            kernels.set_render_meta(
                (1,), (1,), (ref, np.int32(frame_value)),
                stream=_cupy_external_stream(kernels.cp, self._stream),
            )
        return ref

    def _ensure_fused_radial_taps(self, view: ViewInfo, kernels: object) -> object:
        """Cache the per-azimuth sin/cos table used by texture render kernels.

        Reconstruction state is computed in registers, so each geometry retains only two
        float32 values per azimuth rather than an all-diameter tap descriptor. The cache is
        intentionally large enough to retain a normal full view set and keep CuPy launches
        from outliving their Torch-owned angle tables during view transitions.
        """
        angles_np = np.ascontiguousarray(np.asarray(view.azimuths_deg, dtype=np.float32))
        n_angles = int(angles_np.size)
        n_u = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
        if n_angles <= 0 or n_u <= 0:
            raise RuntimeError('fused Radial texture geometry is empty')
        plane_h, plane_w = radial_plane_shape(view)
        source_mode = radial_source_mode()
        key = (
            RADIAL_FILTER_MODE, str(source_mode), int(plane_h), int(plane_w), int(n_u),
            round(float(view.center_x), 5), round(float(view.center_y), 5),
            round(float(view.roi_radius), 5), angles_np.tobytes(),
        )
        with self._radial_texture_lock:
            cached = self._fused_radial_taps.get(key)
            if cached is not None:
                self._fused_radial_taps.move_to_end(key)
                return cached

            radians = np.deg2rad(angles_np.astype(np.float64)).astype(np.float32)
            cos_np = np.ascontiguousarray(np.cos(radians).astype(np.float32))
            sin_np = np.ascontiguousarray(np.sin(radians).astype(np.float32))
            torch = self.torch
            angle_cos = torch.from_numpy(cos_np).to(self.device)
            angle_sin = torch.from_numpy(sin_np).to(self.device)
            cp = kernels.cp
            refs = argparse.Namespace(
                angle_cos=angle_cos,
                angle_sin=angle_sin,
                cp_angle_cos=cp.asarray(angle_cos),
                cp_angle_sin=cp.asarray(angle_sin),
                n_angles=n_angles,
                n_u=n_u,
                plane_h=int(plane_h),
                plane_w=int(plane_w),
                nbytes=int(cos_np.nbytes + sin_np.nbytes),
            )
            self._fused_radial_taps[key] = refs
            self._fused_radial_taps.move_to_end(key)
            limit = max(8, _env_int('YOLO_TTA_RADIAL_TEXTURE_GEOMETRY_CACHE_ENTRIES', 128))
            while len(self._fused_radial_taps) > int(limit):
                self._fused_radial_taps.popitem(last=False)
            print(
                f'Radial geometry cached on {self.device}: {n_angles} azimuths, '
                f'{n_u} radial samples, {int(plane_h)}x{int(plane_w)} projected plane; '
                f'source_mode={source_mode}, angle_table={refs.nbytes / (1024 ** 2):.3f} MiB.'
            )
            return refs

    def _try_fused_radial_into_slot(
        self,
        slot: _ResidentGpuPipelineSlot,
        view: ViewInfo,
        aff: AffineSpec,
        frame_index: int,
        out_size: int,
        *,
        stage_metadata: bool = True,
        disable_on_failure: bool = True,
    ) -> bool:
        tilted_radial = bool(is_tilted_radial_view(view))
        render_family = 'tilted_radial' if tilted_radial else 'radial'
        if (
            not fused_radial_render_enabled()
            or 'radial' in self._fused_disabled_families
            or render_family in self._fused_disabled_families
            or not radial_fused_render_supported(view)
        ):
            return False
        try:
            if str(view.family) != 'radial':
                return False
            if self._volume_gpu is None or not bool(self._volume_gpu.is_contiguous()):
                raise RuntimeError('resident uint8 source volume is unavailable or non-contiguous')
            if self._volume_gpu.dtype != self.torch.uint8:
                raise RuntimeError(f'expected uint8 source volume, got {self._volume_gpu.dtype}')
            native_t, full_h, full_w = (int(v) for v in self._volume_gpu.shape)
            if int(view.full_h) != full_h or int(view.full_w) != full_w:
                raise RuntimeError('Radial view/source-volume XY geometry mismatch')
            if int(frame_index) < 0 or int(frame_index) >= len(view.azimuths_deg):
                raise RuntimeError(f'Radial frame index {frame_index} is outside its azimuth table')
            kernels = _fused_direct_render_kernels()
            if kernels is None:
                raise RuntimeError(
                    'CuPy/NVRTC direct renderer kernels are unavailable: '
                    + str(_FUSED_DIRECT_RENDER_KERNELS_ERROR or 'no diagnostic')
                )
            geometry = self._ensure_fused_radial_taps(view, kernels)
            source_mode = radial_source_mode()
            if source_mode == 'texture_linear':
                source_arg = self._ensure_radial_texture(kernels).texture
                kernel_prefix = 'radial_texture'
            else:
                source_arg = self._fused_cupy_volume(kernels)
                kernel_prefix = 'radial_pointer'
            matrix = np.asarray(aff.M_out_to_src, dtype=np.float32).reshape(2, 3)
            if not bool(np.all(np.isfinite(matrix))):
                raise RuntimeError('Radial output-to-source affine is non-finite')
            if slot.input.dtype not in (self.torch.float16, self.torch.float32):
                raise RuntimeError(f'unsupported binding dtype {slot.input.dtype}')

            base = str(radial_base_view_name(view))
            base_ids = {'transverse': 0, 'sagittal': 1, 'coronal': 2}
            if base not in base_ids:
                raise RuntimeError(f'unsupported Radial base {base!r}')
            direction = str(view.tilt_direction) if tilted_radial else 'vertical'
            if direction not in ('vertical', 'horizontal'):
                raise RuntimeError(f'unsupported tilted-Radial direction {direction!r}')
            plane_h, plane_w = radial_plane_shape(view)
            if int(geometry.plane_h) != int(plane_h) or int(geometry.plane_w) != int(plane_w):
                raise RuntimeError('Radial angle table/projected-plane geometry mismatch')
            stack_len = int(radial_stack_length(view))
            if stack_len <= 0:
                raise RuntimeError('Radial stack geometry is empty')

            external = _cupy_external_stream(kernels.cp, self._stream)
            metadata = self._fused_slot_metadata(
                slot, kernels, int(frame_index) if bool(stage_metadata) else None,
            )
            output_ref = self._fused_slot_output(slot, kernels)
            pixels = int(out_size) * int(out_size)
            kernel = getattr(
                kernels,
                f'{kernel_prefix}_direct_f16'
                if slot.input.dtype == self.torch.float16
                else f'{kernel_prefix}_direct_f32',
            )
            kernel(
                ((pixels + 255) // 256,), (256,),
                (
                    source_arg,
                    np.int32(native_t), np.int32(full_h), np.int32(full_w),
                    np.int32(self._logical_t), np.int32(view.src_h),
                    np.int32(geometry.n_u), np.int32(stack_len),
                    np.int32(base_ids[base]),
                    np.int32(0 if direction == 'vertical' else 1),
                    metadata,
                    np.float32(
                        math.tan(math.radians(float(view.tilt_angle_deg)))
                        if tilted_radial else 0.0
                    ),
                    np.float32(view.center_x), np.float32(view.center_y),
                    np.float32(view.roi_radius),
                    np.int32(out_size), np.int32(out_size),
                    *(np.float32(v) for v in matrix.reshape(-1)),
                    geometry.cp_angle_cos, geometry.cp_angle_sin,
                    output_ref,
                ),
                stream=external,
            )
            if render_family not in self._fused_announced_families:
                self._fused_announced_families.add(render_family)
                geometry_label = f'tilted {base}' if tilted_radial else f'upright {base}'
                print(
                    f'Fused {geometry_label} Radial renderer active: output affine + radial '
                    f'mapping + source_mode={source_mode} -> TensorRT binding in one launch.'
                )
            return True
        except Exception as exc:
            if not bool(disable_on_failure):
                raise
            self._fused_radial_taps.clear()
            self._fused_render_fallback(render_family, exc)
            return False


    def _try_fused_tilted_into_slot(
        self,
        slot: _ResidentGpuPipelineSlot,
        view: ViewInfo,
        aff: AffineSpec,
        frame_index: int,
        out_size: int,
        *,
        stage_metadata: bool = True,
        disable_on_failure: bool = True,
    ) -> bool:
        if not fused_tilted_render_enabled() or 'tilted' in self._fused_disabled_families:
            return False
        try:
            if not is_tilted_view(view):
                return False
            if self._volume_gpu is None or not bool(self._volume_gpu.is_contiguous()):
                raise RuntimeError('resident uint8 source volume is unavailable or non-contiguous')
            if self._volume_gpu.dtype != self.torch.uint8:
                raise RuntimeError(f'expected uint8 source volume, got {self._volume_gpu.dtype}')
            native_t, full_h, full_w = (int(v) for v in self._volume_gpu.shape)
            if int(view.full_h) != full_h or int(view.full_w) != full_w:
                raise RuntimeError('Tilted view/source-volume XY geometry mismatch')
            base = str(tilted_base_view_name(view))
            base_ids = {'transverse': 0, 'sagittal': 1, 'coronal': 2}
            if base not in base_ids:
                raise RuntimeError(f'unsupported Tilted base {base!r}')
            direction = str(view.tilt_direction)
            if direction not in ('vertical', 'horizontal'):
                raise RuntimeError(f'unsupported Tilted direction {direction!r}')
            expected = cartesian_view_axis_spec(base, int(self._logical_t), full_h, full_w)
            if int(view.src_h) != int(expected['src_h']) or int(view.src_w) != int(expected['src_w']):
                raise RuntimeError('Tilted source raster does not match its base-view geometry')
            stack_len = int(tilted_stack_axis_length(view))
            center = int(tilted_frame_center(view, int(frame_index)))
            if stack_len <= 0 or center < 0 or center >= stack_len:
                raise RuntimeError('Tilted frame center is outside the stack geometry')
            kernels = _fused_direct_render_kernels()
            if kernels is None:
                raise RuntimeError('CuPy/NVRTC direct renderer kernels are unavailable: ' + str(_FUSED_DIRECT_RENDER_KERNELS_ERROR or 'no diagnostic'))
            matrix = np.asarray(aff.M_out_to_src, dtype=np.float32).reshape(2, 3)
            if not bool(np.all(np.isfinite(matrix))):
                raise RuntimeError('Tilted output-to-source affine is non-finite')
            cp = kernels.cp
            external = _cupy_external_stream(cp, self._stream)
            kernel = (
                kernels.tilted_direct_f16
                if slot.input.dtype == self.torch.float16 else kernels.tilted_direct_f32
            )
            if slot.input.dtype not in (self.torch.float16, self.torch.float32):
                raise RuntimeError(f'unsupported binding dtype {slot.input.dtype}')
            pixels = int(out_size) * int(out_size)
            kernel(
                ((pixels + 255) // 256,), (256,),
                (
                    self._fused_cupy_volume(kernels),
                    np.int32(native_t), np.int32(full_h), np.int32(full_w), np.int32(self._logical_t),
                    np.int32(view.src_h), np.int32(view.src_w), np.int32(stack_len),
                    np.int32(base_ids[base]), np.int32(0 if direction == 'vertical' else 1),
                    self._fused_slot_metadata(
                        slot, kernels, int(center) if bool(stage_metadata) else None,
                    ),
                    np.float32(math.tan(math.radians(float(view.tilt_angle_deg)))),
                    np.int32(1 if tilted_inplane_linear_enabled() else 0),
                    np.int32(out_size), np.int32(out_size),
                    *(np.float32(v) for v in matrix.reshape(-1)),
                    self._fused_slot_output(slot, kernels),
                ),
                stream=external,
            )
            if 'tilted' not in self._fused_announced_families:
                self._fused_announced_families.add('tilted')
                inplane_label = 'bilinear' if tilted_inplane_linear_enabled() else 'nearest'
                print(
                    'P4 fused Tilted renderer active: affine/shear/gathers/lerps -> '
                    f'TensorRT binding (in-plane={inplane_label}).'
                )
            return True
        except Exception as exc:
            if not bool(disable_on_failure):
                raise
            self._fused_render_fallback('tilted', exc)
            return False

    def _try_fused_render_into_ring_slot(
        self,
        slot: _ResidentGpuPipelineSlot,
        view: ViewInfo,
        aff: AffineSpec,
        frame_index: int,
        out_size: int,
        *,
        stage_metadata: bool = True,
        allow_graph_replay: bool = True,
        disable_on_failure: bool = True,
    ) -> bool:
        """Render an eligible resident-ring Radial or Tilted frame into its fixed binding."""
        family = 'radial' if str(view.family) == 'radial' else ('tilted' if is_tilted_view(view) else '')
        if not family:
            return False
        if (
            bool(allow_graph_replay)
            and slot.render_graph is not None
            and slot.render_expected_key is not None
            and slot.render_graph_key == slot.render_expected_key
            and family not in self._fused_disabled_families
        ):
            kernels = _fused_direct_render_kernels()
            if kernels is None:
                return False
            if family == 'radial':
                if int(frame_index) < 0 or int(frame_index) >= len(view.azimuths_deg):
                    raise _ResidentTensorRTRingFatalError(
                        f'P4 Radial graph frame {frame_index} is outside its descriptor table'
                    )
                dynamic_value = int(frame_index)
            else:
                dynamic_value = int(tilted_frame_center(view, int(frame_index)))
                if dynamic_value < 0 or dynamic_value >= int(tilted_stack_axis_length(view)):
                    raise _ResidentTensorRTRingFatalError(
                        f'P4 Tilted graph center {dynamic_value} is outside its stack'
                    )
            try:
                self._fused_slot_metadata(slot, kernels, dynamic_value)
                slot.render_graph.replay()
            except BaseException as exc:
                # The ring has already borrowed TRT bindings when this path runs. A graph
                # replay failure cannot enter the worker's generic lazy-render retry safely.
                raise _ResidentTensorRTRingFatalError(
                    f'P4 fused {family} renderer CUDA Graph replay failed'
                ) from exc
            return True
        if family == 'radial':
            return self._try_fused_radial_into_slot(
                slot, view, aff, frame_index, out_size,
                stage_metadata=bool(stage_metadata),
                disable_on_failure=bool(disable_on_failure),
            )
        if family == 'tilted':
            return self._try_fused_tilted_into_slot(
                slot, view, aff, frame_index, out_size,
                stage_metadata=bool(stage_metadata),
                disable_on_failure=bool(disable_on_failure),
            )
        return False

    def _fused_renderer_key(
        self,
        slot: _ResidentGpuPipelineSlot,
        view: ViewInfo,
        aff: AffineSpec,
        out_size: int,
    ) -> Tuple[object, ...]:
        family = 'radial' if str(view.family) == 'radial' else ('tilted' if is_tilted_view(view) else '')
        matrix_key = tuple(
            round(float(x), 7)
            for x in np.asarray(aff.M_out_to_src, dtype=np.float32).reshape(-1).tolist()
        )
        if family == 'radial':
            family_geometry: Tuple[object, ...] = (
                radial_base_view_name(view), bool(is_tilted_radial_view(view)),
                int(view.src_h), int(view.src_w), *radial_plane_shape(view),
                round(float(view.center_x), 6), round(float(view.center_y), 6),
                round(float(view.roi_radius), 6),
                np.ascontiguousarray(np.asarray(view.azimuths_deg, dtype=np.float32)).tobytes(),
            )
        elif family == 'tilted':
            family_geometry = (
                str(tilted_base_view_name(view)), str(view.tilt_direction),
                round(float(view.tilt_angle_deg), 7), int(view.tilt_frame_start),
                int(view.tilt_frame_stop), int(view.src_h), int(view.src_w),
                int(view.full_t), int(view.full_h), int(view.full_w),
            )
        else:
            family_geometry = ()
        return (
            self._volume_key, str(view.name), family, int(out_size), str(slot.input.dtype),
            radial_source_mode() if family == 'radial' else '',
            matrix_key, family_geometry,
        )

    def _reference_fused_frame(
        self,
        view: ViewInfo,
        aff: AffineSpec,
        frame_index: int,
        out_size: int,
    ) -> object:
        """Independent Torch reference used only by the fail-fast startup probe."""
        if is_tilted_view(view):
            return self._render_tilted_frame(
                view, aff.M_out_to_src, int(out_size), int(out_size), int(frame_index),
            )
        if is_tilted_radial_view(view):
            plane = self._render_tilted_radial_native_resident_torch(view, int(frame_index))
        elif is_radial_view(view):
            plane = self._render_radial_native_resident_torch(view, int(frame_index))
        else:
            raise ValueError(f'No fused reference renderer for {view.name!r}')
        if self._affine_is_identity_render(aff):
            return plane
        theta = _affine_theta_from_dst_to_src(
            aff.M_out_to_src,
            int(plane.shape[0]), int(plane.shape[1]), int(out_size), int(out_size),
        )
        grid = _get_cached_affine_grid(theta, int(out_size), int(out_size), self.device)
        return self.F.grid_sample(
            plane.reshape(1, 1, int(plane.shape[0]), int(plane.shape[1])),
            grid, mode='bilinear', padding_mode='zeros', align_corners=False,
        ).reshape(int(out_size), int(out_size))

    def validate_fused_ring_renderer(
        self,
        slot: _ResidentGpuPipelineSlot,
        view: ViewInfo,
        job: AugJob,
        frame_index: int,
        out_size: int,
        *,
        compare_reference: bool = False,
    ) -> None:
        """Prove one fused launch and optionally compare it with an independent renderer.

        Enabled fused renderers are fail-fast by default.  A launch rejection or CUDA fault
        aborts the worker instead of silently routing tens of thousands of frames through the
        slower reference path.  The startup probe compares one representative frame from each
        fused family under explicit, user-adjustable tolerances.
        """
        family = 'radial' if str(view.family) == 'radial' else ('tilted' if is_tilted_view(view) else '')
        preflight_family = _fused_preflight_family(view)
        if not family:
            return
        if family == 'radial' and not fused_radial_render_enabled():
            return
        if family == 'tilted' and not fused_tilted_render_enabled():
            return
        key = self._fused_renderer_key(slot, view, job.aff, int(out_size))
        if key in self._fused_validated_keys and not bool(compare_reference):
            return
        if family in self._fused_disabled_families:
            if fused_renderer_fail_fast_enabled():
                raise _ResidentTensorRTRingFatalError(
                    f'P4 fused {preflight_family or family} renderer was disabled before validation'
                )
            return
        try:
            with self.torch.cuda.stream(self._stream):
                launched = self._try_fused_render_into_ring_slot(
                    slot, view, job.aff, int(frame_index), int(out_size),
                    allow_graph_replay=False,
                    disable_on_failure=not fused_renderer_fail_fast_enabled(),
                )
            self._stream.synchronize()
        except BaseException as exc:
            raise _ResidentTensorRTRingFatalError(
                f'P4 fused {preflight_family or family} renderer failed its CUDA validation probe'
            ) from exc
        if not launched:
            if fused_renderer_fail_fast_enabled():
                raise _ResidentTensorRTRingFatalError(
                    f'P4 fused {preflight_family or family} renderer rejected an enabled validation probe'
                )
            return

        if (
            bool(compare_reference)
            and family == 'radial'
            and radial_source_mode() != 'texture_linear'
        ):
            try:
                fused = slot.input[0, 0].to(self.torch.float32)
                if not bool(self.torch.isfinite(fused).all().item()):
                    raise RuntimeError('pointer Radial renderer produced non-finite pixels')
                min_value = float(fused.min().item()) if int(fused.numel()) else 0.0
                max_value = float(fused.max().item()) if int(fused.numel()) else 0.0
                if min_value < -1e-6 or max_value > 1.0 + 1e-6:
                    raise RuntimeError(
                        f'pointer Radial renderer escaped normalized range [{min_value},{max_value}]'
                    )
            except BaseException as exc:
                raise _ResidentTensorRTRingFatalError(
                    f'P4 fused {preflight_family or family} pointer validation failed'
                ) from exc
            compare_reference = False
            self._fused_preflight_validated_families.add(str(preflight_family or family))
            print(
                f'P4 fused {preflight_family or family} pointer-mode launch validation passed; '
                'reference-delta gating skipped because nearest-XY/linear-T is intentionally '
                'output-changing.'
            )

        if bool(compare_reference):
            try:
                with self.torch.no_grad():
                    reference = self._reference_fused_frame(
                        view, job.aff, int(frame_index), int(out_size),
                    ).to(self.torch.float32).div_(255.0)
                    fused = slot.input[0, 0].to(self.torch.float32)
                    delta = (fused - reference).abs()
                    max_abs = float(delta.max().item()) if int(delta.numel()) else 0.0
                    mean_abs = float(delta.mean().item()) if int(delta.numel()) else 0.0
                    if not bool(self.torch.isfinite(fused).all().item()):
                        raise RuntimeError('fused renderer produced non-finite pixels')
                    if not bool(self.torch.isfinite(reference).all().item()):
                        raise RuntimeError('reference renderer produced non-finite pixels')
                    (
                        max_tol, configured_max_fraction_tol, mean_tol,
                        mismatch_abs, mismatch_fraction_tol,
                    ) = fused_renderer_preflight_tolerances()
                    max_fraction_tol, texture_seam_fraction_floor = (
                        fused_renderer_effective_max_fraction_tolerance(
                            float(configured_max_fraction_tol),
                            preflight_family=str(preflight_family or family),
                            height=int(delta.shape[-2]),
                            width=int(delta.shape[-1]),
                        )
                    )
                    max_exceed_fraction = float(
                        (delta > float(max_tol)).to(self.torch.float32).mean().item()
                    ) if int(delta.numel()) else 0.0
                    mismatch_fraction = float(
                        (delta > float(mismatch_abs)).to(self.torch.float32).mean().item()
                    ) if int(delta.numel()) else 0.0
            except BaseException as exc:
                raise _ResidentTensorRTRingFatalError(
                    f'P4 fused {preflight_family or family} reference comparison failed'
                ) from exc
            if (
                max_exceed_fraction > float(max_fraction_tol)
                or mean_abs > float(mean_tol)
                or mismatch_fraction > float(mismatch_fraction_tol)
            ):
                raise _ResidentTensorRTRingFatalError(
                    f'P4 fused {preflight_family or family} preflight exceeded tolerance: '
                    f'max={max_abs:.6f}, fraction(abs>{max_tol:.6f})='
                    f'{max_exceed_fraction:.6f}/{max_fraction_tol:.6f} '
                    f'(configured={configured_max_fraction_tol:.6f}, '
                    f'texture_seam_floor={texture_seam_fraction_floor:.6f}), '
                    f'mean={mean_abs:.6f}/{mean_tol:.6f}, '
                    f'fraction(abs>{mismatch_abs:.6f})={mismatch_fraction:.6f}/'
                    f'{mismatch_fraction_tol:.6f}'
                )
            self._fused_preflight_validated_families.add(str(preflight_family or family))
            print(
                f'P4 fused {preflight_family or family} preflight passed: '
                f'max_abs={max_abs:.6f}, max_exceed_fraction={max_exceed_fraction:.6f}/'
                f'{max_fraction_tol:.6f} (configured={configured_max_fraction_tol:.6f}, '
                f'texture_seam_floor={texture_seam_fraction_floor:.6f}), '
                f'mean_abs={mean_abs:.6f}, mismatch_fraction={mismatch_fraction:.6f}.'
            )
        self._fused_validated_keys.add(key)

    def run_startup_fused_preflight(
        self,
        specs: Sequence[Dict[str, object]],
        *,
        out_size: int,
        fp16: bool,
    ) -> None:
        """Validate one upright-Radial, Tilted, and tilted-Radial frame per volume."""
        if not fused_renderer_preflight_enabled() or not specs:
            return
        if self._mode != 'resident' or self._volume_key is None:
            return
        if self._fused_preflight_volume_key == self._volume_key:
            return
        torch = self.torch
        dtype = torch.float16 if bool(fp16) else torch.float32
        found: set[str] = set()
        for spec in specs:
            view = spec.get('view')
            job = spec.get('job')
            frame_index = int(spec.get('frame_index', 0))
            if not isinstance(view, ViewInfo) or not isinstance(job, AugJob):
                raise _ResidentTensorRTRingFatalError('invalid fused-render preflight descriptor')
            family = _fused_preflight_family(view)
            if not family or family in found:
                continue
            slot = argparse.Namespace(
                input=torch.empty(
                    (1, 1, int(out_size), int(out_size)), dtype=dtype, device=self.device,
                ),
                # set_render_meta writes two int32 fields; match the production ring slot
                # exactly so startup validation cannot corrupt an adjacent allocator block.
                render_meta=torch.empty((2,), dtype=torch.int32, device=self.device),
                _render_cupy_refs={}, render_graph=None, render_graph_key=None,
                render_expected_key=None,
            )
            self.validate_fused_ring_renderer(
                slot, view, job, int(frame_index), int(out_size), compare_reference=True,
            )
            found.add(family)
        expected = {
            _fused_preflight_family(spec['view'])
            for spec in specs
            if isinstance(spec.get('view'), ViewInfo) and _fused_preflight_family(spec['view'])
        }
        missing = sorted(expected.difference(found))
        if missing:
            raise _ResidentTensorRTRingFatalError(
                f'fused-render startup preflight did not execute required families: {missing}'
            )
        self._fused_preflight_volume_key = self._volume_key

    def capture_fused_ring_renderer(
        self,
        slot: _ResidentGpuPipelineSlot,
        view: ViewInfo,
        job: AugJob,
        frame_index: int,
        out_size: int,
    ) -> None:
        """Capture and validate one slot's stable render kernel opportunistically.

 The one-thread metadata setter intentionally remains outside the graph. Therefore a
 replay has fixed volume/descriptor/output addresses but reads the just-staged frame
 index or Tilted center from ``slot.render_meta``. Capture failures with a healthy
 stream retain the uncaptured fused launch; failed recovery is worker-fatal."""
        if not fused_render_cuda_graphs_enabled():
            return
        family = 'radial' if str(view.family) == 'radial' else ('tilted' if is_tilted_view(view) else '')
        if not family or family in self._fused_disabled_families:
            return
        key = self._fused_renderer_key(slot, view, job.aff, int(out_size))
        slot.render_expected_key = key
        # A slot can be reused after a view/affine/volume change. Never leave an older
        # executable installed when this new key is rejected or capture fails: otherwise
        # the frame path could replay stale descriptor pointers and affine constants.
        if slot.render_graph is not None and slot.render_graph_key != key:
            slot.render_graph = None
            slot.render_graph_key = None
        if key not in self._fused_validated_keys:
            return
        if key in self._fused_graph_rejected_keys:
            return
        if slot.render_graph is not None and slot.render_graph_key == key:
            return
        kernels = _fused_direct_render_kernels()
        if kernels is None:
            return
        dynamic_value = (
            int(frame_index)
            if family == 'radial'
            else int(tilted_frame_center(view, int(frame_index)))
        )
        # Create zero-copy wrappers before stream capture; their addresses remain stable.
        self._fused_slot_metadata(slot, kernels, None)
        self._fused_slot_output(slot, kernels)
        graph = None
        try:
            graph = self.torch.cuda.CUDAGraph()
            with _cuda_graph_capture_context(self.torch, graph, self._stream):
                launched = self._try_fused_render_into_ring_slot(
                    slot, view, job.aff, int(frame_index), int(out_size),
                    stage_metadata=False, allow_graph_replay=False,
                    disable_on_failure=False,
                )
            if not launched:
                graph = None
                self._stream.synchronize()
                self._fused_graph_rejected_keys.add(key)
                return
            # Prove graph instantiation/replay before source consumption. The dynamic
            # metadata write is ordered immediately ahead of replay on the same stream.
            with self.torch.cuda.stream(self._stream):
                self._fused_slot_metadata(slot, kernels, dynamic_value)
                graph.replay()
            self._stream.synchronize()
        except Exception as exc:
            graph = None
            try:
                self._stream.synchronize()
            except BaseException as sync_exc:
                raise _ResidentTensorRTRingFatalError(
                    f'failed to recover P4 fused {family} renderer stream after graph capture'
                ) from sync_exc
            if family not in self._fused_graph_warned_families:
                self._fused_graph_warned_families.add(family)
                print(
                    f'P4 fused {family} renderer CUDA Graph capture unavailable ({exc}); '
                    'using its uncaptured direct kernel.'
                )
            self._fused_graph_rejected_keys.add(key)
            return
        slot.render_graph = graph
        slot.render_graph_key = key
        if family not in self._fused_graph_announced_families:
            self._fused_graph_announced_families.add(family)
            print(f'P4 fused {family} renderer CUDA Graph active (dynamic device metadata).')

    # radial taps / fold (device) ----

    def _radial_taps_gpu(self, view: ViewInfo, angle_deg: float) -> Tuple[object, object]:
        """Device fallback for the active orientation-aware radial reconstruction filter."""
        torch = self.torch
        dev = self.device
        plane_h, plane_w = radial_plane_shape(view)
        n_u = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
        coords = torch.linspace(
            -float(view.roi_radius), float(view.roi_radius), n_u,
            dtype=torch.float32, device=dev,
        )
        theta = math.radians(float(angle_deg))
        xs = float(view.center_x) + coords * float(math.cos(theta))
        ys = float(view.center_y) + coords * float(math.sin(theta))
        offs = torch.tensor((0.0, 1.0), dtype=torch.float32, device=dev)
        x_pos = xs.floor().unsqueeze(1) + offs.unsqueeze(0)
        y_pos = ys.floor().unsqueeze(1) + offs.unsqueeze(0)
        x_w = (1.0 - (xs.unsqueeze(1) - x_pos).abs()).clamp_min_(0.0)
        y_w = (1.0 - (ys.unsqueeze(1) - y_pos).abs()).clamp_min_(0.0)
        x_w = x_w * ((x_pos >= 0) & (x_pos < float(plane_w))).to(torch.float32)
        y_w = y_w * ((y_pos >= 0) & (y_pos < float(plane_h))).to(torch.float32)
        x_sum = x_w.sum(dim=1, keepdim=True)
        y_sum = y_w.sum(dim=1, keepdim=True)
        x_w = torch.where(x_sum.abs() > 1e-6, x_w / x_sum, x_w)
        y_w = torch.where(y_sum.abs() > 1e-6, y_w / y_sum, y_w)
        x_idx = x_pos.clamp(0.0, float(plane_w - 1)).to(torch.int64)
        y_idx = y_pos.clamp(0.0, float(plane_h - 1)).to(torch.int64)
        flat_idx = (y_idx.unsqueeze(2) * int(plane_w) + x_idx.unsqueeze(1)).reshape(n_u, -1)
        w2d = (y_w.unsqueeze(2) * x_w.unsqueeze(1)).reshape(n_u, -1)
        return flat_idx, w2d

    def _radial_fold_indices(
        self,
        t_dim: int,
        rows: int,
        logical_t: Optional[int] = None,
    ) -> Tuple[object, object, object]:
        torch = self.torch
        # The two center-aligned maps (rows -> logical/cube t -> native t)
        # compose to this direct native coordinate. Keep logical_t in the key so
        # cache identity still describes the renderer geometry being composed.
        logical = int(t_dim if logical_t is None else logical_t)
        key = (int(t_dim), int(logical), int(rows))
        cached = self._fold_cache.get(key)
        if cached is not None:
            return cached
        rf = (np.arange(int(rows), dtype=np.float64) + 0.5) * (float(t_dim) / float(rows)) - 0.5
        r0 = np.clip(np.floor(rf).astype(np.int64), 0, int(t_dim) - 1)
        r1 = np.minimum(r0 + 1, int(t_dim) - 1)
        alpha = np.clip(rf - r0, 0.0, 1.0).astype(np.float32)[:, None]
        out = (
            torch.from_numpy(r0).to(self.device),
            torch.from_numpy(r1).to(self.device),
            torch.from_numpy(alpha).to(self.device),
        )
        self._fold_cache[key] = out
        return out

    def _radial_project_blocks(self, block2d: object, flat_idx: object, w2d: object) -> object:
        """(rows, H*W) u8 block -> (rows, u) float32 active-filter projection."""
        samples = block2d[:, flat_idx]
        return (samples.to(self.torch.float32) * w2d.unsqueeze(0)).sum(dim=-1)

    def _render_radial_native_resident(self, view: ViewInfo, frame_idx: int) -> object:
        """Render a Radial native plane, preferring the direct hardware texture kernel."""
        if not radial_resident_gpu_render_supported(view):
            raise RuntimeError(f'resident GPU Radial rendering is disabled for {view.name!r}')
        if is_tilted_radial_view(view):
            return self._render_tilted_radial_native_resident(view, int(frame_idx))
        if _env_flag('YOLO_TTA_GPU_RADIAL_NATIVE_TEXTURE_KERNEL', True):
            try:
                return self._render_radial_native_texture(view, int(frame_idx))
            except Exception as exc:
                if 'radial_native' not in self._fused_warned_families:
                    self._fused_warned_families.add('radial_native')
                    print(
                        f'Warning: upright Radial texture kernel unavailable ({exc}); '
                        'using the resident Torch reconstruction path.'
                    )
        return self._render_radial_native_resident_torch(view, int(frame_idx))

    def _render_radial_native_resident_torch(self, view: ViewInfo, frame_idx: int) -> object:
        """Render an upright or tilted Radial frame directly from the resident source volume."""
        if not radial_resident_gpu_render_supported(view):
            raise RuntimeError(f'resident GPU Radial rendering is disabled for {view.name!r}')
        if is_tilted_radial_view(view):
            return self._render_tilted_radial_native_resident_torch(view, int(frame_idx))
        torch = self.torch
        vol = self._volume_gpu
        base = radial_base_view_name(view)
        rows_out = int(view.src_h)
        u_len = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
        flat_idx, w2d = self._radial_taps_gpu(view, float(view.azimuths_deg[int(frame_idx)]))

        if base == 'transverse':
            native_t = int(vol.shape[0])
            vol2d = vol.view(native_t, -1)
            proj = torch.empty((native_t, u_len), dtype=torch.float32, device=self.device)
            chunk = 512
            for t0 in range(0, native_t, chunk):
                t1 = min(native_t, t0 + chunk)
                proj[t0:t1] = self._radial_project_blocks(vol2d[t0:t1], flat_idx, w2d)
            if rows_out == native_t and int(self._logical_t) == native_t:
                return proj
            r0, r1, alpha = self._radial_fold_indices(
                native_t, rows_out, logical_t=int(self._logical_t),
            )
            return proj[r0] * (1.0 - alpha) + proj[r1] * alpha

        stack_len = int(radial_stack_length(view))
        proj = torch.empty((stack_len, u_len), dtype=torch.float32, device=self.device)
        block = max(1, _env_int('YOLO_TTA_GPU_RADIAL_STACK_BLOCK', 32))
        for s0 in range(0, stack_len, block):
            s1 = min(stack_len, s0 + block)
            if base == 'sagittal':
                # source (native_t, Yblock, X) -> logical (Yblock, t, X)
                oriented = self._resample_native_t_axis(vol[:, s0:s1, :]).permute(1, 0, 2).contiguous()
            elif base == 'coronal':
                # source (native_t, Y, Xblock) -> logical (Xblock, t, Y)
                oriented = self._resample_native_t_axis(vol[:, :, s0:s1]).permute(2, 0, 1).contiguous()
            else:  # pragma: no cover
                raise ValueError(f'Unsupported resident Radial base: {base}')
            proj[s0:s1] = self._radial_project_blocks(
                oriented.view(s1 - s0, -1), flat_idx, w2d,
            )
        if rows_out == stack_len:
            return proj
        r0, r1, alpha = self._radial_fold_indices(stack_len, rows_out)
        return proj[r0] * (1.0 - alpha) + proj[r1] * alpha

    def _render_tilted_radial_native_resident_torch(self, view: ViewInfo, frame_idx: int) -> object:
        """Reference CUDA/Torch implementation matching ``extract_tilted_radial_slice_frame``.

 This path is retained when the allocation-free NVRTC kernel is unavailable or explicitly
 disabled. It keeps every gather and interpolation on the resident GPU and therefore never
 requests the deferred host cube. Row blocking bounds temporary int64 gather tensors."""
        torch = self.torch
        vol = self._volume_gpu
        native_t, full_h, full_w = (int(v) for v in vol.shape)
        logical_t = int(self._logical_t)
        base = str(radial_base_view_name(view))
        stack_len = int(radial_stack_length(view))
        rows = int(view.src_h)
        u_len = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
        plane_h, plane_w = radial_plane_shape(view)
        flat_idx, w2d = self._radial_taps_gpu(
            view, float(view.azimuths_deg[int(frame_idx)]),
        )
        px = torch.remainder(flat_idx, int(plane_w)).to(torch.int64)
        py = torch.div(flat_idx, int(plane_w), rounding_mode='floor').to(torch.int64)
        weights = w2d.to(torch.float32)
        tan_tilt = float(math.tan(math.radians(float(view.tilt_angle_deg))))
        if str(view.tilt_direction) == 'vertical':
            tap_offsets = py.to(torch.float32) - float(view.center_y)
        elif str(view.tilt_direction) == 'horizontal':
            tap_offsets = px.to(torch.float32) - float(view.center_x)
        else:
            raise ValueError(f'Unsupported tilted-Radial direction: {view.tilt_direction!r}')

        if rows == stack_len:
            row_centers = torch.arange(rows, dtype=torch.float32, device=self.device)
        else:
            row_centers = (
                (torch.arange(rows, dtype=torch.float32, device=self.device) + 0.5)
                * (float(stack_len) / float(rows))
                - 0.5
            ).clamp_(0.0, float(max(0, stack_len - 1)))

        out = torch.empty((rows, u_len), dtype=torch.float32, device=self.device)
        row_block = max(
            1,
            min(256, _env_int('YOLO_TTA_GPU_TILTED_RADIAL_ROW_BLOCK', 32)),
        )
        plane_stride = int(full_h) * int(full_w)
        volume_flat = self._volume_flat
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        native_r0: Optional[object] = None
        native_r1: Optional[object] = None
        native_alpha: Optional[object] = None
        if native_t != logical_t:
            native_r0, native_r1, native_alpha = self._native_t_indices(logical_t)

        for row0 in range(0, rows, row_block):
            row1 = min(rows, row0 + row_block)
            centers = row_centers[row0:row1].view(-1, 1, 1)
            stack_src = centers + float(tan_tilt) * tap_offsets.unsqueeze(0)
            valid = (stack_src >= 0.0) & (stack_src <= float(stack_len - 1))
            s0f = stack_src.floor().clamp_(0.0, float(stack_len - 1))
            s1f = (s0f + 1.0).clamp_(max=float(stack_len - 1))
            s0 = s0f.to(torch.int64)
            s1 = s1f.to(torch.int64)
            stack_alpha = stack_src - s0f

            if base == 'transverse':
                spatial = py * int(full_w) + px
                if native_t == logical_t:
                    v0 = torch.take(
                        volume_flat, spatial.unsqueeze(0) + s0 * int(plane_stride),
                    ).to(torch.float32)
                    v1 = torch.take(
                        volume_flat, spatial.unsqueeze(0) + s1 * int(plane_stride),
                    ).to(torch.float32)
                else:
                    n00 = native_r0[s0]; n01 = native_r1[s0]; a0 = native_alpha[s0]
                    n10 = native_r0[s1]; n11 = native_r1[s1]; a1 = native_alpha[s1]
                    f00 = torch.take(volume_flat, spatial.unsqueeze(0) + n00 * int(plane_stride)).to(torch.float32)
                    f01 = torch.take(volume_flat, spatial.unsqueeze(0) + n01 * int(plane_stride)).to(torch.float32)
                    f10 = torch.take(volume_flat, spatial.unsqueeze(0) + n10 * int(plane_stride)).to(torch.float32)
                    f11 = torch.take(volume_flat, spatial.unsqueeze(0) + n11 * int(plane_stride)).to(torch.float32)
                    v0 = (f00 + a0 * (f01 - f00)).round_().clamp_(0.0, 255.0)
                    v1 = (f10 + a1 * (f11 - f10)).round_().clamp_(0.0, 255.0)
            elif base in ('sagittal', 'coronal'):
                if base == 'sagittal':
                    spatial0 = s0 * int(full_w) + px.unsqueeze(0)
                    spatial1 = s1 * int(full_w) + px.unsqueeze(0)
                else:
                    spatial0 = px.unsqueeze(0) * int(full_w) + s0
                    spatial1 = px.unsqueeze(0) * int(full_w) + s1
                if native_t == logical_t:
                    t_base = py.unsqueeze(0) * int(plane_stride)
                    v0 = torch.take(volume_flat, t_base + spatial0).to(torch.float32)
                    v1 = torch.take(volume_flat, t_base + spatial1).to(torch.float32)
                else:
                    t0 = native_r0[py].unsqueeze(0)
                    t1 = native_r1[py].unsqueeze(0)
                    ta = native_alpha[py].unsqueeze(0)
                    f00 = torch.take(volume_flat, t0 * int(plane_stride) + spatial0).to(torch.float32)
                    f01 = torch.take(volume_flat, t1 * int(plane_stride) + spatial0).to(torch.float32)
                    f10 = torch.take(volume_flat, t0 * int(plane_stride) + spatial1).to(torch.float32)
                    f11 = torch.take(volume_flat, t1 * int(plane_stride) + spatial1).to(torch.float32)
                    v0 = (f00 + ta * (f01 - f00)).round_().clamp_(0.0, 255.0)
                    v1 = (f10 + ta * (f11 - f10)).round_().clamp_(0.0, 255.0)
            else:  # pragma: no cover
                raise ValueError(f'Unsupported tilted-Radial base: {base!r}')

            values = v0 + stack_alpha * (v1 - v0)
            values = torch.where(valid, values, zero)
            out[row0:row1] = (values * weights.unsqueeze(0)).sum(dim=-1)
        return out

    def _render_radial_native_texture(self, view: ViewInfo, frame_idx: int) -> object:
        """Render one upright or tilted Radial native plane from the selected source mode."""
        kernels = _fused_direct_render_kernels()
        if kernels is None:
            raise RuntimeError(
                'CuPy/NVRTC kernels unavailable: '
                + str(_FUSED_DIRECT_RENDER_KERNELS_ERROR or 'no diagnostic')
            )
        geometry = self._ensure_fused_radial_taps(view, kernels)
        source_mode = radial_source_mode()
        if source_mode == 'texture_linear':
            source_arg = self._ensure_radial_texture(kernels).texture
            kernel = kernels.radial_texture_native_f32
        else:
            source_arg = self._fused_cupy_volume(kernels)
            kernel = kernels.radial_pointer_native_f32
        base = str(radial_base_view_name(view))
        base_ids = {'transverse': 0, 'sagittal': 1, 'coronal': 2}
        tilted = bool(is_tilted_radial_view(view))
        direction = str(view.tilt_direction) if tilted else 'vertical'
        if base not in base_ids or direction not in ('vertical', 'horizontal'):
            raise RuntimeError(
                f'unsupported Radial texture geometry base={base!r}, direction={direction!r}'
            )
        rows = int(view.src_h)
        n_u = int(geometry.n_u)
        stack_len = int(radial_stack_length(view))
        out = self.torch.empty((rows, n_u), dtype=self.torch.float32, device=self.device)
        cp_out = kernels.cp.asarray(out)
        pixels = int(rows) * int(n_u)
        kernel(
            ((pixels + 255) // 256,), (256,),
            (
                source_arg,
                np.int32(self._volume_gpu.shape[0]), np.int32(self._volume_gpu.shape[1]),
                np.int32(self._volume_gpu.shape[2]), np.int32(self._logical_t),
                np.int32(rows), np.int32(n_u), np.int32(stack_len),
                np.int32(base_ids[base]), np.int32(0 if direction == 'vertical' else 1),
                np.int32(frame_idx),
                np.float32(
                    math.tan(math.radians(float(view.tilt_angle_deg))) if tilted else 0.0
                ),
                np.float32(view.center_x), np.float32(view.center_y),
                np.float32(view.roi_radius),
                geometry.cp_angle_cos, geometry.cp_angle_sin, cp_out,
            ),
            stream=_cupy_external_stream(kernels.cp, self._stream),
        )
        announce_key = 'tilted_radial_native' if tilted else 'radial_native'
        if announce_key not in self._fused_announced_families:
            self._fused_announced_families.add(announce_key)
            print(
                f'Resident {"tilted-" if tilted else ""}Radial native-plane kernel active: '
                f'source_mode={source_mode}.'
            )
        return out

    def _render_radial_texture_grid(
        self,
        view: ViewInfo,
        frame_idx: int,
        M_out_to_src: np.ndarray,
        out_h: int,
        out_w: int,
    ) -> object:
        """Compose the output affine and Radial transform in one selected-source launch."""
        if not is_radial_view(view):
            raise ValueError(f'{view.name!r} is not a Radial view')
        if int(frame_idx) < 0 or int(frame_idx) >= len(view.azimuths_deg):
            raise IndexError(
                f'Radial frame index {int(frame_idx)} is outside [0,{len(view.azimuths_deg)})'
            )
        kernels = _fused_direct_render_kernels()
        if kernels is None:
            raise RuntimeError(
                'CuPy/NVRTC kernels unavailable: '
                + str(_FUSED_DIRECT_RENDER_KERNELS_ERROR or 'no diagnostic')
            )
        geometry = self._ensure_fused_radial_taps(view, kernels)
        source_mode = radial_source_mode()
        if source_mode == 'texture_linear':
            source_arg = self._ensure_radial_texture(kernels).texture
            kernel = kernels.radial_texture_grid_f32
        else:
            source_arg = self._fused_cupy_volume(kernels)
            kernel = kernels.radial_pointer_grid_f32
        base = str(radial_base_view_name(view))
        base_ids = {'transverse': 0, 'sagittal': 1, 'coronal': 2}
        tilted = bool(is_tilted_radial_view(view))
        direction = str(view.tilt_direction) if tilted else 'vertical'
        if base not in base_ids or direction not in ('vertical', 'horizontal'):
            raise RuntimeError(
                f'unsupported Radial texture geometry base={base!r}, direction={direction!r}'
            )
        matrix = np.asarray(M_out_to_src, dtype=np.float32).reshape(2, 3)
        if not bool(np.all(np.isfinite(matrix))):
            raise RuntimeError('Radial output-to-source affine is non-finite')
        rows = int(view.src_h)
        n_u = int(geometry.n_u)
        stack_len = int(radial_stack_length(view))
        if min(rows, n_u, stack_len, int(out_h), int(out_w)) <= 0:
            raise RuntimeError('Radial texture output/source geometry is empty')
        out = self.torch.empty(
            (int(out_h), int(out_w)), dtype=self.torch.float32, device=self.device,
        )
        cp_out = kernels.cp.asarray(out)
        pixels = int(out_h) * int(out_w)
        kernel(
            ((pixels + 255) // 256,), (256,),
            (
                source_arg,
                np.int32(self._volume_gpu.shape[0]), np.int32(self._volume_gpu.shape[1]),
                np.int32(self._volume_gpu.shape[2]), np.int32(self._logical_t),
                np.int32(rows), np.int32(n_u), np.int32(stack_len),
                np.int32(base_ids[base]), np.int32(0 if direction == 'vertical' else 1),
                np.int32(frame_idx),
                np.float32(
                    math.tan(math.radians(float(view.tilt_angle_deg))) if tilted else 0.0
                ),
                np.float32(view.center_x), np.float32(view.center_y),
                np.float32(view.roi_radius),
                np.int32(out_h), np.int32(out_w),
                *(np.float32(v) for v in matrix.reshape(-1)),
                geometry.cp_angle_cos, geometry.cp_angle_sin, cp_out,
            ),
            stream=_cupy_external_stream(kernels.cp, self._stream),
        )
        announce_key = 'tilted_radial_grid' if tilted else 'radial_grid'
        if announce_key not in self._fused_announced_families:
            self._fused_announced_families.add(announce_key)
            print(
                f'Direct {"tilted-" if tilted else ""}Radial grid renderer active: '
                f'output affine + radial mapping + source_mode={source_mode} in one launch.'
            )
        return out

    def _render_tilted_radial_native_resident(self, view: ViewInfo, frame_idx: int) -> object:
        """Render a tilted-Radial plane through the texture kernel, with Torch fallback."""
        if not is_tilted_radial_view(view):
            raise ValueError(f'{view.name!r} is not a tilted-Radial view')
        native_kernel_enabled = _env_flag('YOLO_TTA_GPU_TILTED_RADIAL_NATIVE_KERNEL', True)
        if native_kernel_enabled:
            try:
                return self._render_radial_native_texture(view, int(frame_idx))
            except Exception as exc:
                if 'tilted_radial_native' not in self._fused_warned_families:
                    self._fused_warned_families.add('tilted_radial_native')
                    print(
                        f'Warning: tilted-Radial texture kernel unavailable ({exc}); '
                        'using the resident Torch reconstruction path without requesting the host cube.'
                    )
        return self._render_tilted_radial_native_resident_torch(view, int(frame_idx))


    def prerender_radial_slab(
        self,
        view: ViewInfo,
        frame_indices: Sequence[int],
    ) -> np.ndarray:
        """GPU-render selected upright Radial frames from logical stack-axis slabs.

        Transverse streams t slabs, sagittal streams y slabs arranged as ``(y,t,x)``,
        and coronal streams x slabs arranged as ``(x,t,y)``.  Each bounded slab is
        contiguous before H2D transfer, so the active radial-filter projection contract is
        shared by all three Cartesian Radial bases without materializing a full oriented
        volume on either host or device.
        """
        if not radial_streaming_gpu_render_supported(view):
            raise RuntimeError(
                f'non-resident GPU Radial prerender does not support {view.name!r}'
            )
        if self._volume_mm is None:
            raise RuntimeError('non-resident GPU Radial prerender has no source memmap')
        torch = self.torch
        native_t, full_h, full_w = (int(x) for x in self._volume_mm.shape)
        base = radial_base_view_name(view)
        stack_len = int(radial_stack_length(view))
        plane_h, plane_w = radial_plane_shape(view)
        rows_out = int(view.src_h)
        u_len = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
        indices = tuple(int(value) for value in frame_indices)
        if not indices:
            raise RuntimeError('streamed radial prerender received no frame indices')
        if len(set(indices)) != len(indices):
            raise RuntimeError('streamed radial prerender frame indices must be unique')
        if any(value < 0 or value >= int(view.num_slices) for value in indices):
            raise RuntimeError(
                f'streamed radial prerender indices are outside [0,{int(view.num_slices)})'
            )
        expected_plane = {
            'transverse': (full_h, full_w),
            'sagittal': (native_t, full_w),
            'coronal': (native_t, full_h),
        }[base]
        expected_stack = {'transverse': native_t, 'sagittal': full_h, 'coronal': full_w}[base]
        if (int(plane_h), int(plane_w), int(stack_len)) != (
            int(expected_plane[0]), int(expected_plane[1]), int(expected_stack)
        ):
            raise RuntimeError(
                f'{base} Radial logical geometry mismatch: plane={plane_h}x{plane_w}, '
                f'stack={stack_len}; expected {expected_plane[0]}x{expected_plane[1]}, '
                f'stack={expected_stack}'
            )

        count = len(indices)
        slab = np.empty((count, rows_out, u_len), dtype=np.uint8)
        stack_block = max(1, min(int(gpu_render_tblock_slices()), int(stack_len)))
        tap_count = int(RADIAL_FILTER_TAP_COUNT) ** 2
        per_az_bytes = (
            int(stack_len) * int(u_len) * np.dtype(np.float16).itemsize
            + int(u_len) * int(tap_count) * (
                np.dtype(np.int64).itemsize + np.dtype(np.float32).itemsize
            )
        )

        def _logical_stack_block(s0: int, s1: int) -> np.ndarray:
            if base == 'transverse':
                return np.ascontiguousarray(self._volume_mm[int(s0):int(s1), :, :])
            if base == 'sagittal':
                return np.ascontiguousarray(
                    np.transpose(self._volume_mm[:, int(s0):int(s1), :], (1, 0, 2))
                )
            return np.ascontiguousarray(
                np.transpose(self._volume_mm[:, :, int(s0):int(s1)], (2, 0, 1))
            )

        with torch.cuda.stream(self._stream):
            try:
                free_bytes, _total = torch.cuda.mem_get_info(self.device)
            except Exception:
                free_bytes = 8 * GIB
            source_block_bytes = int(stack_block) * int(plane_h) * int(plane_w)
            gather_temp = int(stack_block) * int(u_len) * int(tap_count) * np.dtype(np.float32).itemsize
            budget = max(0, int(free_bytes) - source_block_bytes - gather_temp - 2 * GIB)
            az_chunk = int(np.clip(budget // max(1, per_az_bytes), 1, count))
            if az_chunk < min(8, count):
                raise RuntimeError(
                    f'insufficient free VRAM for streamed {base} Radial prerender '
                    f'(az_chunk={az_chunk}, free={free_bytes / GIB:.1f} GiB)'
                )
            fold = rows_out != stack_len
            if fold:
                r0, r1, alpha = self._radial_fold_indices(stack_len, rows_out)
            for a0 in range(0, count, az_chunk):
                a1 = min(count, a0 + az_chunk)
                proj = torch.zeros(
                    (a1 - a0, stack_len, u_len), dtype=torch.float16, device=self.device,
                )
                taps = [
                    self._radial_taps_gpu(view, float(view.azimuths_deg[indices[i]]))
                    for i in range(a0, a1)
                ]
                for s0 in range(0, stack_len, stack_block):
                    s1 = min(stack_len, s0 + stack_block)
                    block_np = _logical_stack_block(s0, s1)
                    block = torch.from_numpy(block_np).to(self.device)
                    block2d = block.view(s1 - s0, -1)
                    for j, (flat_idx, w2d) in enumerate(taps):
                        proj[j, s0:s1] = self._radial_project_blocks(
                            block2d, flat_idx, w2d,
                        ).to(torch.float16)
                    del block, block2d, block_np
                for j in range(a1 - a0):
                    pj = proj[j].to(torch.float32)
                    folded = (pj[r0] * (1.0 - alpha) + pj[r1] * alpha) if fold else pj
                    slab[a0 + j] = folded.round().clamp_(0.0, 255.0).to(torch.uint8).cpu().numpy()
                del proj, taps
        return slab

    # tilted (device) ----

    def _tilted_plan_gpu(self, view: ViewInfo, M_grid_to_src: np.ndarray, grid_h: int, grid_w: int) -> Dict[str, object]:
        torch = self.torch
        key = _tilted_plan_cache_key(view, M_grid_to_src, int(grid_h), int(grid_w))
        cached = self._tilted_plans.get(key)
        if cached is not None:
            self._tilted_plans.move_to_end(key)
            return cached
        plan = get_tilted_render_plan(view, M_grid_to_src, int(grid_h), int(grid_w))
        native_t, full_h, full_w = (int(x) for x in self._volume_gpu.shape)
        logical_t = int(self._logical_t)
        base = tilted_base_view_name(view)
        x64 = plan.x_idx.astype(np.int64)
        y64 = plan.y_idx.astype(np.int64)
        if base == 'transverse':
            # X/Y are native and working-identical on the admission path; t is
            # the fractional STACK coordinate and is composed in _render_tilted_frame.
            inplane0 = y64 * full_w + x64
            inplane1 = inplane0
            t_alpha_np = np.zeros_like(plan.axis_offset, dtype=np.float32)
            stride = full_h * full_w
        elif base == 'sagittal':
            # base axes (X,t), stack Y: working-t is an IN-PLANE coordinate.
            t_pos = (y64.astype(np.float64) + 0.5) * (float(native_t) / float(logical_t)) - 0.5
            t0 = np.clip(np.floor(t_pos).astype(np.int64), 0, native_t - 1)
            t1 = np.minimum(t0 + 1, native_t - 1)
            t_alpha_np = np.clip(t_pos - t0, 0.0, 1.0).astype(np.float32)
            inplane0 = t0 * (full_h * full_w) + x64
            inplane1 = t1 * (full_h * full_w) + x64
            stride = full_w
        elif base == 'coronal':
            # base axes (Y,t), stack X: like sagittal, t needs its own two taps.
            t_pos = (y64.astype(np.float64) + 0.5) * (float(native_t) / float(logical_t)) - 0.5
            t0 = np.clip(np.floor(t_pos).astype(np.int64), 0, native_t - 1)
            t1 = np.minimum(t0 + 1, native_t - 1)
            t_alpha_np = np.clip(t_pos - t0, 0.0, 1.0).astype(np.float32)
            inplane0 = t0 * (full_h * full_w) + x64 * full_w
            inplane1 = t1 * (full_h * full_w) + x64 * full_w
            stride = 1
        else:  # pragma: no cover
            raise ValueError(f'Unsupported Tilted View base: {base}')
        tan_alpha = float(math.tan(math.radians(float(view.tilt_angle_deg))))
        info: Dict[str, object] = {
            'base': str(base),
            't_identity': bool(native_t == logical_t),
            'inplane0': torch.from_numpy(np.ascontiguousarray(inplane0)).to(self.device),
            'inplane1': torch.from_numpy(np.ascontiguousarray(inplane1)).to(self.device),
            't_alpha': torch.from_numpy(np.ascontiguousarray(t_alpha_np)).to(self.device),
            'stack_base': torch.from_numpy(
                np.ascontiguousarray(np.asarray(plan.axis_offset, dtype=np.float32) * np.float32(tan_alpha))
            ).to(self.device),
            'valid_xy': torch.from_numpy(np.ascontiguousarray(plan.valid_xy)).to(self.device),
            'stride': int(stride),
            'stack_len': int(tilted_stack_axis_length(view)),
        }
        self._tilted_plans[key] = info
        self._tilted_plans.move_to_end(key)
        # A sequence of angle-local tile tasks may revisit several static Tilted plans.
        # Keep a bounded floor so those plans survive task overlap without recreating any
        # independent tile crop.
        while len(self._tilted_plans) > max(10, int(self._tilted_plan_cache_floor)):
            self._tilted_plans.popitem(last=False)
        return info

    def request_tilted_plan_cache_entries(self, entries: int) -> None:
        self._tilted_plan_cache_floor = max(
            int(self._tilted_plan_cache_floor),
            min(int(entries), max(16, _env_int('YOLO_TTA_GPU_TILTED_PLAN_CACHE_MAX', 256))),
        )

    def _render_tilted_frame(self, view: ViewInfo, M_grid_to_src: np.ndarray, grid_h: int, grid_w: int, frame_idx: int) -> object:
        torch = self.torch
        if (
            tilted_inplane_linear_enabled()
            and not _tilted_grid_is_identity(M_grid_to_src, int(grid_h), int(grid_w), view)
        ):
            # v16.1.8 forward-pass in-plane interpolation: build the exact integer-grid
            # native frame (the identity branch below), then warp it with the same
            # align_corners=False zero-padded bilinear grid_sample the Cartesian views
            # use. This is also the preflight reference for the fused tilted kernel's
            # bilinear branch, so both stages share one definition.
            src_h, src_w = int(view.src_h), int(view.src_w)
            plane = self._render_tilted_frame(
                view, _TILTED_IDENTITY_M, src_h, src_w, int(frame_idx),
            )
            theta = _affine_theta_from_dst_to_src(
                np.asarray(M_grid_to_src, dtype=np.float32),
                src_h, src_w, int(grid_h), int(grid_w),
            )
            grid = _get_cached_affine_grid(theta, int(grid_h), int(grid_w), self.device)
            return self.F.grid_sample(
                plane.reshape(1, 1, src_h, src_w),
                grid, mode='bilinear', padding_mode='zeros', align_corners=False,
            ).reshape(int(grid_h), int(grid_w))
        info = self._tilted_plan_gpu(view, M_grid_to_src, int(grid_h), int(grid_w))
        stack_len = int(info['stack_len'])
        center = float(tilted_frame_center(view, int(frame_idx)))
        stack_src = info['stack_base'] + center
        valid = info['valid_xy'] & (stack_src >= 0.0) & (stack_src <= float(stack_len - 1))
        s0f = stack_src.floor().clamp(0.0, float(stack_len - 1))
        s1f = (s0f + 1.0).clamp(max=float(stack_len - 1))
        alpha = stack_src - s0f
        stride = int(info['stride'])
        if str(info['base']) == 'transverse':
            # Reconstruct the two working-t stack taps from native t (including
            # the former cube-u8 rounding), then apply the existing stack lerp.
            # This is exact to without materializing the full cube volume.
            inplane = info['inplane0']
            if bool(info['t_identity']):
                f0 = torch.take(
                    self._volume_flat, inplane + s0f.to(torch.int64) * stride,
                ).to(torch.float32)
                f1 = torch.take(
                    self._volume_flat, inplane + s1f.to(torch.int64) * stride,
                ).to(torch.float32)
                vals = f0 + alpha * (f1 - f0)
                return torch.where(valid, vals, torch.zeros((), dtype=torch.float32, device=self.device))
            r0, r1, tmap_alpha = self._native_t_indices(int(self._logical_t))
            s0i = s0f.to(torch.int64)
            s1i = s1f.to(torch.int64)
            n00 = r0[s0i]; n01 = r1[s0i]; a0 = tmap_alpha[s0i]
            n10 = r0[s1i]; n11 = r1[s1i]; a1 = tmap_alpha[s1i]
            f00 = torch.take(self._volume_flat, inplane + n00 * stride).to(torch.float32)
            f01 = torch.take(self._volume_flat, inplane + n01 * stride).to(torch.float32)
            f10 = torch.take(self._volume_flat, inplane + n10 * stride).to(torch.float32)
            f11 = torch.take(self._volume_flat, inplane + n11 * stride).to(torch.float32)
            v0 = (f00 + a0 * (f01 - f00)).round_().clamp_(0.0, 255.0)
            v1 = (f10 + a1 * (f11 - f10)).round_().clamp_(0.0, 255.0)
            vals = v0 + alpha * (v1 - v0)
        else:
            # Sagittal/coronal bases carry working t in-plane. Interpolate native
            # t at each of the two spatial stack taps (four gathers total), round
            # each t-lerp as the former resident uint8 cube did, then stack-lerp.
            inplane0 = info['inplane0']
            inplane1 = info['inplane1']
            t_alpha = info['t_alpha']
            s0 = s0f.to(torch.int64) * stride
            s1 = s1f.to(torch.int64) * stride
            if bool(info['t_identity']):
                v0 = torch.take(self._volume_flat, inplane0 + s0).to(torch.float32)
                v1 = torch.take(self._volume_flat, inplane0 + s1).to(torch.float32)
            else:
                f00 = torch.take(self._volume_flat, inplane0 + s0).to(torch.float32)
                f01 = torch.take(self._volume_flat, inplane1 + s0).to(torch.float32)
                f10 = torch.take(self._volume_flat, inplane0 + s1).to(torch.float32)
                f11 = torch.take(self._volume_flat, inplane1 + s1).to(torch.float32)
                v0 = (f00 + t_alpha * (f01 - f00)).round_().clamp_(0.0, 255.0)
                v1 = (f10 + t_alpha * (f11 - f10)).round_().clamp_(0.0, 255.0)
            vals = v0 + alpha * (v1 - v0)
        return torch.where(valid, vals, torch.zeros((), dtype=torch.float32, device=self.device))

    # cartesian / dispatch ----

    @staticmethod
    def _affine_is_identity_render(aff: AffineSpec) -> bool:
        return bool(
            int(aff.src_w) == int(aff.out_size)
            and int(aff.src_h) == int(aff.out_size)
            and int(aff.canvas_w) == int(aff.src_w)
            and int(aff.canvas_h) == int(aff.src_h)
            and float(aff.angle_deg) % 360.0 == 0.0
        )

    def _native_plane_cache_entries(self) -> int:
        """LRU depth for native planes reused inside one angle-local tile source."""
        return max(2, _env_int('YOLO_TTA_GPU_NATIVE_PLANE_CACHE', 8))

    def _render_native_plane_cached(self, view: ViewInfo, frame_idx: int) -> object:
        """Cache native planes for repeated reads inside one tile's inference stream.

 Batches and 2.5D channel formats can request the same center or neighbouring plane more
 than once. This bounded cache serves those reads without introducing cross-tile union or
 per-tile crop state. Full-frame rendering keeps its existing allocation behavior."""
        key = (str(view.name), int(frame_idx))
        cache = self._native_plane_cache
        plane = cache.get(key)
        if plane is not None:
            cache.move_to_end(key)
            return plane
        plane = self._render_native_plane(view, int(frame_idx))
        cache[key] = plane
        cache.move_to_end(key)
        while len(cache) > self._native_plane_cache_entries():
            cache.popitem(last=False)
        return plane

    def clear_native_plane_cache(self) -> None:
        self._native_plane_cache.clear()

    def _render_native_plane(self, view: ViewInfo, frame_idx: int) -> object:
        vol = self._volume_gpu
        name = physical_view_name(view)
        if name == 'transverse':
            if int(vol.shape[0]) == int(self._logical_t):
                return vol[int(frame_idx)].to(self.torch.float32)
            r0, r1, alpha = self._native_t_indices(int(self._logical_t))
            tap0 = r0[int(frame_idx):int(frame_idx) + 1]
            tap1 = r1[int(frame_idx):int(frame_idx) + 1]
            f0 = vol.index_select(0, tap0).squeeze(0).to(self.torch.float32)
            f1 = vol.index_select(0, tap1).squeeze(0).to(self.torch.float32)
            return (f0 + alpha[int(frame_idx)] * (f1 - f0)).round_().clamp_(0.0, 255.0)
        if name == 'sagittal':
            return self._resample_native_t_axis(vol[:, int(frame_idx), :])
        if name == 'coronal':
            return self._resample_native_t_axis(vol[:, :, int(frame_idx)]).contiguous()
        if str(view.family) == 'radial':
            return self._render_radial_native_resident(view, int(frame_idx))
        raise ValueError(f'Unsupported view for GPU native plane: {name}')

    def _render_fullframe_frame(self, view: ViewInfo, aff: AffineSpec, frame_idx: int, out_size: int) -> object:
        if is_tilted_view(view):
            return self._render_tilted_frame(
                view, aff.M_out_to_src, int(out_size), int(out_size), int(frame_idx),
            )
        if is_radial_view(view) and _env_flag('YOLO_TTA_GPU_RADIAL_DIRECT_TEXTURE_GRID', True):
            try:
                return self._render_radial_texture_grid(
                    view, int(frame_idx), aff.M_out_to_src, int(out_size), int(out_size),
                )
            except Exception as exc:
                warning_key = 'tilted_radial_grid_fallback' if is_tilted_radial_view(view) else 'radial_grid_fallback'
                if warning_key not in self._fused_warned_families:
                    self._fused_warned_families.add(warning_key)
                    print(
                        f'Warning: direct Radial texture-grid render unavailable ({exc}); '
                        'using native-plane reconstruction followed by the affine warp.'
                    )
        plane = self._render_native_plane(view, int(frame_idx))
        if self._affine_is_identity_render(aff):
            return plane
        theta = _affine_theta_from_dst_to_src(
            aff.M_out_to_src,
            int(plane.shape[0]), int(plane.shape[1]), int(out_size), int(out_size),
        )
        grid = _get_cached_affine_grid(theta, int(out_size), int(out_size), self.device)
        return self.F.grid_sample(
            plane.reshape(1, 1, int(plane.shape[0]), int(plane.shape[1])),
            grid, mode='bilinear', padding_mode='zeros', align_corners=False,
        ).reshape(int(out_size), int(out_size))

    def _render_tile_plane(
        self,
        view: ViewInfo,
        M_out_to_src: np.ndarray,
        frame_idx: int,
        out_size: int,
    ) -> object:
        """Render one dense-tile inference raster directly when the view supports it."""
        matrix = np.asarray(M_out_to_src, dtype=np.float32)
        if is_tilted_view(view):
            return self._render_tilted_frame(
                view, matrix, int(out_size), int(out_size), int(frame_idx),
            )
        if is_radial_view(view) and _env_flag('YOLO_TTA_GPU_RADIAL_DIRECT_TEXTURE_GRID', True):
            try:
                return self._render_radial_texture_grid(
                    view, int(frame_idx), matrix, int(out_size), int(out_size),
                )
            except Exception as exc:
                warning_key = 'tilted_radial_tile_grid_fallback' if is_tilted_radial_view(view) else 'radial_tile_grid_fallback'
                if warning_key not in self._fused_warned_families:
                    self._fused_warned_families.add(warning_key)
                    print(
                        f'Warning: direct Radial tile texture-grid render unavailable ({exc}); '
                        'using the cached native plane plus affine grid sampling.'
                    )
        plane = self._render_native_plane_cached(view, int(frame_idx))
        theta = _affine_theta_from_dst_to_src(
            matrix,
            int(plane.shape[0]), int(plane.shape[1]), int(out_size), int(out_size),
        )
        grid = _get_cached_affine_grid(theta, int(out_size), int(out_size), self.device)
        return self.F.grid_sample(
            plane.reshape(1, 1, int(plane.shape[0]), int(plane.shape[1])),
            grid, mode='bilinear', padding_mode='zeros', align_corners=False,
        ).reshape(int(out_size), int(out_size))

    def render_tile_batch(
        self,
        view: ViewInfo,
        tile_affine: np.ndarray,
        frame_indices: Sequence[int],
        *,
        out_size: int,
        fp16: bool,
        channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    ) -> Tuple[object, object]:
        """Render one tile's center frames as a normalized BCHW tensor on the GPU.

        The source/native view plane remains resident in the worker. Every contextual plane
        is cropped, warped, and resized directly into the tile inference raster; no grouped
        tile canvas or cross-tile accumulation exists in v16.4.0.
        """
        torch = self.torch
        fmt = resolve_channel_format(channel_format)
        offsets = (0,) if fmt.kind == 'gray' else tuple(int(v) for v in fmt.offsets)
        matrix = np.asarray(tile_affine, dtype=np.float32)
        with torch.cuda.stream(self._stream):
            requested_indices: List[Tuple[int, ...]] = []
            unique_indices: Dict[int, object] = {}
            for frame_idx in frame_indices:
                contextual = tuple(
                    channel_view_slice_index(view, int(frame_idx) + int(offset))
                    for offset in offsets
                )
                requested_indices.append(contextual)
                for source_idx in contextual:
                    if int(source_idx) not in unique_indices:
                        unique_indices[int(source_idx)] = self._render_tile_plane(
                            view, matrix, int(source_idx), int(out_size),
                        )

            frames: List[object] = []
            for contextual in requested_indices:
                if fmt.kind == 'gray':
                    frames.append(unique_indices[int(contextual[0])].unsqueeze(0))
                else:
                    frames.append(torch.stack(
                        [unique_indices[int(source_idx)] for source_idx in contextual], dim=0,
                    ))
            batch = torch.stack(frames, dim=0)
            batch = batch.clamp_(0.0, 255.0).mul_(1.0 / 255.0)
            if bool(fp16):
                batch = batch.to(torch.float16)
            ready_event = torch.cuda.Event()
            ready_event.record(self._stream)
        return batch, ready_event

    def render_fullframe_batch(
        self,
        view: ViewInfo,
        job: AugJob,
        frame_indices: Sequence[int],
        out_size: int,
        fp16: bool,
        channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    ) -> Tuple[object, object]:
        """Render one normalized BCHW batch on the render stream; returns (tensor, ready event)."""
        torch = self.torch
        fmt = resolve_channel_format(channel_format)
        with torch.cuda.stream(self._stream):
            requested_indices: List[Tuple[int, ...]] = []
            unique_indices: Dict[int, object] = {}
            for frame_idx in frame_indices:
                center = int(frame_idx)
                offsets = (0,) if fmt.kind == 'gray' else tuple(int(v) for v in fmt.offsets)
                contextual = tuple(
                    channel_view_slice_index(view, center + int(offset))
                    for offset in offsets
                )
                requested_indices.append(contextual)
                for source_idx in contextual:
                    if source_idx not in unique_indices:
                        unique_indices[source_idx] = self._render_fullframe_frame(
                            view, job.aff, int(source_idx), int(out_size)
                        )

            frames: List[object] = []
            for contextual in requested_indices:
                if fmt.kind == 'gray':
                    frames.append(unique_indices[int(contextual[0])].unsqueeze(0))
                else:
                    frames.append(torch.stack(
                        [unique_indices[int(source_idx)] for source_idx in contextual],
                        dim=0,
                    ))
            batch = torch.stack(frames, dim=0)
            batch = batch.clamp_(0.0, 255.0).mul_(1.0 / 255.0)
            if bool(fp16):
                batch = batch.to(torch.float16)
            ready_event = torch.cuda.Event()
            ready_event.record(self._stream)
        return batch, ready_event

    def render_fullframe_into_ring_slot(
        self,
        slot: _ResidentGpuPipelineSlot,
        view: ViewInfo,
        job: AugJob,
        frame_index: int,
        out_size: int,
        channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    ) -> None:
        """Render one center stack into a slot's persistent TensorRT input address."""
        torch = self.torch
        fmt = resolve_channel_format(channel_format)
        expected_shape = (
            1, int(fmt.channel_count), int(out_size), int(out_size),
        )
        if tuple(int(x) for x in slot.input.shape) != expected_shape:
            raise RuntimeError(
                f'resident ring input shape {tuple(slot.input.shape)} != {expected_shape}'
            )
        with torch.cuda.stream(self._stream):
            if bool(slot.infer_valid):
                # This slot's preceding TensorRT enqueue has consumed its input before
                # the low-priority renderer writes the next frame into the same address.
                self._stream.wait_event(slot.infer_done)
            # The fused direct renderer owns a single output plane. Arbitrary-channel
            # inputs still use the persistent TensorRT contexts/static bindings below,
            # with the channel-aware Torch renderer filling those bindings in place.
            if (
                int(fmt.channel_count) == 1
                and abs(float(job.angle_deg)) <= 1.0e-7
                and self._try_fused_render_into_ring_slot(
                    slot, view, job.aff, int(frame_index), int(out_size),
                )
            ):
                slot.render_done.record(self._stream)
                return

            contextual = tuple(
                channel_view_slice_index(view, int(frame_index) + int(offset))
                for offset in fmt.offsets
            )
            first_channel_by_source: Dict[int, int] = {}
            for channel_idx, source_idx in enumerate(contextual):
                first_channel = first_channel_by_source.get(int(source_idx))
                if first_channel is not None:
                    slot.input[0, int(channel_idx)].copy_(
                        slot.input[0, int(first_channel)], non_blocking=True,
                    )
                    continue
                plane = self._render_fullframe_frame(
                    view, job.aff, int(source_idx), int(out_size)
                )
                plane.clamp_(0.0, 255.0).mul_(1.0 / 255.0)
                slot.input[0, int(channel_idx)].copy_(plane, non_blocking=True)
                first_channel_by_source[int(source_idx)] = int(channel_idx)
            slot.render_done.record(self._stream)


    def render_tile_into_ring_slot(
        self,
        slot: _ResidentGpuPipelineSlot,
        view: ViewInfo,
        tile_affine: np.ndarray,
        frame_index: int,
        *,
        out_size: int,
        channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    ) -> None:
        """Render one tile center stack into a persistent TensorRT input slot."""
        torch = self.torch
        fmt = resolve_channel_format(channel_format)
        expected_shape = (1, int(fmt.channel_count), int(out_size), int(out_size))
        if tuple(int(x) for x in slot.input.shape) != expected_shape:
            raise RuntimeError(
                f'resident tile-ring input shape {tuple(slot.input.shape)} != {expected_shape}'
            )
        matrix = np.asarray(tile_affine, dtype=np.float32)
        with torch.cuda.stream(self._stream):
            if bool(slot.infer_valid):
                self._stream.wait_event(slot.infer_done)
            contextual = tuple(
                channel_view_slice_index(view, int(frame_index) + int(offset))
                for offset in fmt.offsets
            )
            first_channel_by_source: Dict[int, int] = {}
            for channel_index, source_index in enumerate(contextual):
                first_channel = first_channel_by_source.get(int(source_index))
                if first_channel is not None:
                    slot.input[0, int(channel_index)].copy_(
                        slot.input[0, int(first_channel)], non_blocking=True,
                    )
                    continue
                plane = self._render_tile_plane(
                    view, matrix, int(source_index), int(out_size),
                )
                plane.clamp_(0.0, 255.0).mul_(1.0 / 255.0)
                slot.input[0, int(channel_index)].copy_(plane, non_blocking=True)
                first_channel_by_source[int(source_index)] = int(channel_index)
            slot.render_done.record(self._stream)

class GpuRenderedYoloSource:
    """Ultralytics-compatible source whose batches are rendered AND normalized on the GPU.

 Used by CUDA inference workers in resident mode. ``__next__`` yields
 (paths, GpuPrefetchedYoloBatch, info) exactly like GpuPrefetchingYoloSource — the batch
 carries a device-resident normalized BCHW tensor plus a render-stream event, and the
 patched BasePredictor.preprocess consumes it via the _tta_gpu_tensor contract. The
 orig-image entries are zero-strided placeholder frames (only their shape is read)."""

    def __init__(
        self,
        engine: _GpuWorkerRenderEngine,
        view: ViewInfo,
        job: AugJob,
        *,
        slice_offset: int,
        num_frames: int,
        batch_size: int,
        out_size: int,
        fp16: bool,
        name: str,
        channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    ) -> None:
        self.engine = engine
        self.view = view
        self.job = job
        self.slice_offset = int(slice_offset)
        self.name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'gpu_rendered_volume'
        self.channel_format = resolve_channel_format(channel_format)
        self.channel_count = int(self.channel_format.channel_count)
        self.out_size = int(out_size)
        self.fp16 = bool(fp16)
        self.nf = max(0, int(num_frames))
        self.bs = max(1, int(batch_size))
        self.yield_nf = int(math.ceil(float(self.nf) / float(self.bs)) * self.bs) if self.nf > 0 else 0
        self.synthetic_count = max(0, int(self.yield_nf) - int(self.nf))
        self.mode = 'image'
        self.count = 0
        self._direct_count = 0
        self._direct_ring: Optional[List[_ResidentGpuPipelineSlot]] = None
        self._fake_frame = np.broadcast_to(
            np.zeros((1, 1, int(self.channel_count)), dtype=np.uint8),
            (self.out_size, self.out_size, int(self.channel_count)),
        )
        # This source bypasses maybe_wrap_source_with_gpu_input_staging (which is where other
        # sources get registered), so register with Ultralytics' source checker here.
        ensure_ultralytics_accepts_in_memory_volume_source()
        try:
            from ultralytics.data.loaders import SourceTypes  # type: ignore
            self.source_type = SourceTypes(stream=False, screenshot=False, from_img=True, tensor=False)
        except Exception:
            self.source_type = argparse.Namespace(stream=False, screenshot=False, from_img=True, tensor=False)

    def __len__(self) -> int:
        return int(math.ceil(float(self.nf) / float(self.bs))) if self.nf > 0 else 0

    def __iter__(self) -> 'GpuRenderedYoloSource':
        self.count = 0
        self._direct_count = 0
        return self

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def prepare_direct_ring(self, input_dtype: Optional[object] = None) -> List[_ResidentGpuPipelineSlot]:
        """Allocate the two-entry ring in the TensorRT ENGINE input binding dtype.

 Unified ``--quantize`` selects the predictor precision policy, but exported TensorRT
 engines may expose an input binding whose dtype differs from that policy. The capability
 probe resolves the actual binding first; ``self.fp16`` is only the generic fallback."""
        if (
            self.bs != 1
            or self.nf <= 0
            or str(getattr(self.engine, '_mode', '')) != 'resident'
        ):
            raise RuntimeError(
                'direct resident ring requires a nonempty batch-1 resident source'
            )
        torch = self.engine.torch
        resolved_dtype = (
            input_dtype
            if input_dtype is not None
            else (torch.float16 if bool(self.fp16) else torch.float32)
        )
        if resolved_dtype not in (torch.float16, torch.float32):
            raise RuntimeError(
                f'resident ring cannot render normalized images into TensorRT dtype {resolved_dtype}'
            )
        if (
            self._direct_ring is None
            or any(
                slot.input.dtype != resolved_dtype
                or tuple(int(x) for x in slot.input.shape)
                != (1, int(self.channel_count), int(self.out_size), int(self.out_size))
                for slot in self._direct_ring
            )
        ):
            self._direct_ring = [
                _ResidentGpuPipelineSlot(
                    torch, self.engine.device, int(self.out_size), int(self.channel_count),
                    resolved_dtype, slot_id=i,
                )
                for i in range(2)
            ]
        if (
            int(self.channel_count) == 1
            and abs(float(self.job.angle_deg)) <= 1.0e-7
        ):
            family = (
                'radial' if str(self.view.family) == 'radial'
                else ('tilted' if is_tilted_view(self.view) else '')
            )
            for slot in self._direct_ring:
                expected_key = (
                    self.engine._fused_renderer_key(
                        slot, self.view, self.job.aff, int(self.out_size),
                    )
                    if family else None
                )
                slot.render_expected_key = expected_key
                if slot.render_graph is not None and slot.render_graph_key != expected_key:
                    slot.render_graph = None
                    slot.render_graph_key = None
            # Synchronize the first async descriptor/render launch before
            # _try_resident_trt_ring_accumulate consumes a frame or borrows TRT bindings.
            self.engine.validate_fused_ring_renderer(
                self._direct_ring[0], self.view, self.job,
                int(self.slice_offset), int(self.out_size),
            )
            for slot in self._direct_ring:
                self.engine.capture_fused_ring_renderer(
                    slot, self.view, self.job,
                    int(self.slice_offset), int(self.out_size),
                )
        else:
            for slot in self._direct_ring:
                slot.render_graph = None
                slot.render_graph_key = None
                slot.render_expected_key = None
            # Prove the arbitrary-channel renderer before TensorRT borrows the
            # backend context and binding addresses.
            self.engine.render_fullframe_into_ring_slot(
                self._direct_ring[0], self.view, self.job,
                int(self.slice_offset), int(self.out_size),
                channel_format=self.channel_format,
            )
            self.engine._stream.synchronize()
        self._direct_count = 0
        self.count = 0
        return self._direct_ring

    def reset_direct_ring(self) -> None:
        """Reset source position after a pre-consumption capability probe failed."""
        self._direct_count = 0
        self.count = 0
        # The generic fallback never uses these buffers. Drop a rejected probe's static
        # bindings/events immediately so a capability mismatch does not reserve VRAM for
        # the remainder of the worker process.
        self._direct_ring = None

    def next_direct_slot(self) -> Optional[Tuple[int, _ResidentGpuPipelineSlot]]:
        """Queue one render without constructing paths/info/source tuples.

 This entry point belongs exclusively to the specialized accumulator. Generic
 Ultralytics/direct iterators continue to use ``__next__`` unchanged."""
        if self._direct_ring is None:
            raise RuntimeError('direct resident ring has not been prepared')
        if int(self._direct_count) >= int(self.nf):
            return None
        local_index = int(self._direct_count)
        absolute_index = int(self.slice_offset) + local_index
        slot = self._direct_ring[local_index & 1]
        slot.frame_index = int(local_index)
        slot.absolute_index = int(absolute_index)
        slot.synthetic = False
        self.engine.render_fullframe_into_ring_slot(
            slot, self.view, self.job, int(absolute_index), int(self.out_size),
            channel_format=self.channel_format,
        )
        self._direct_count += 1
        return int(local_index), slot

    def __next__(self) -> Tuple[List[str], object, List[str]]:
        if self.count >= self.yield_nf or self.nf <= 0:
            raise StopIteration
        start = int(self.count)
        stop = min(int(self.yield_nf), start + int(self.bs))
        self.count = int(stop)
        paths: List[str] = []
        info: List[str] = []
        abs_indices: List[int] = []
        last_real_idx = max(0, int(self.nf) - 1)
        for idx in range(start, stop):
            real_idx = int(idx) if int(idx) < int(self.nf) else int(last_real_idx)
            synthetic = int(idx) >= int(self.nf)
            suffix = '_synthetic' if synthetic else ''
            abs_indices.append(int(self.slice_offset) + int(real_idx))
            paths.append(f'{self.name}_{idx + 1:06d}{suffix}.png')
            if synthetic:
                info.append(f'gpu-rendered {self.name} synthetic padded slice {idx + 1}/{self.yield_nf} repeats real slice {real_idx + 1}/{self.nf}: ')
            else:
                info.append(f'gpu-rendered {self.name} slice {idx + 1}/{self.nf}: ')
        gpu_tensor, ready_event = self.engine.render_fullframe_batch(
            self.view,
            self.job,
            abs_indices,
            int(self.out_size),
            bool(self.fp16),
            channel_format=self.channel_format,
        )
        batch = GpuPrefetchedYoloBatch(
            [self._fake_frame] * len(abs_indices),
            gpu_tensor=gpu_tensor,
            ready_event=ready_event,
            source_label=self.name,
        )
        return paths, batch, info

class GpuTileRenderedYoloSource:
    """GPU-rendered source for exactly one dense tile.

    Each task retains its own tile identity and affine from render through inference and
    postprocessing. The source never flattens or unions several tiles into one stream.
    """

    def __init__(
        self,
        engine: _GpuWorkerRenderEngine,
        view: ViewInfo,
        tile_job: DenseTileJob,
        *,
        slice_offset: int,
        num_frames: int,
        batch_size: int,
        out_size: int,
        fp16: bool,
        name: str,
        channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    ) -> None:
        self.engine = engine
        self.view = view
        self.tile_job = tile_job
        self.tile_affine = np.asarray(tile_job.M_out_to_src, dtype=np.float32)
        self.slice_offset = int(slice_offset)
        self.name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'gpu_tile_volume'
        self.channel_format = resolve_channel_format(channel_format)
        self.channel_count = int(self.channel_format.channel_count)
        self.out_size = int(out_size)
        self.fp16 = bool(fp16)
        self.nf = max(0, int(num_frames))
        self.bs = max(1, int(batch_size))
        self.yield_nf = int(math.ceil(float(self.nf) / float(self.bs)) * self.bs) if self.nf > 0 else 0
        self.synthetic_count = max(0, int(self.yield_nf) - int(self.nf))
        self.mode = 'image'
        self.count = 0
        self._direct_count = 0
        self._direct_ring: Optional[List[_ResidentGpuPipelineSlot]] = None
        self._fake_frame = np.broadcast_to(
            np.zeros((1, 1, int(self.channel_count)), dtype=np.uint8),
            (self.out_size, self.out_size, int(self.channel_count)),
        )
        ensure_ultralytics_accepts_in_memory_volume_source()
        try:
            from ultralytics.data.loaders import SourceTypes  # type: ignore
            self.source_type = SourceTypes(stream=False, screenshot=False, from_img=True, tensor=False)
        except Exception:
            self.source_type = argparse.Namespace(stream=False, screenshot=False, from_img=True, tensor=False)

    def __len__(self) -> int:
        return int(math.ceil(float(self.nf) / float(self.bs))) if self.nf > 0 else 0

    def __iter__(self) -> 'GpuTileRenderedYoloSource':
        self.count = 0
        self._direct_count = 0
        return self

    def start(self) -> None:
        return None

    def close(self) -> None:
        try:
            self.engine.clear_native_plane_cache()
        except Exception:
            pass

    def prepare_direct_ring(self, input_dtype: Optional[object] = None) -> List[_ResidentGpuPipelineSlot]:
        """Allocate and validate two static TensorRT input slots for this tile."""
        if self.bs != 1 or self.nf <= 0 or str(getattr(self.engine, '_mode', '')) != 'resident':
            raise RuntimeError('direct tile ring requires a nonempty batch-1 resident tile source')
        torch = self.engine.torch
        resolved_dtype = input_dtype if input_dtype is not None else (torch.float16 if self.fp16 else torch.float32)
        if resolved_dtype not in (torch.float16, torch.float32):
            raise RuntimeError(f'resident tile ring cannot render into TensorRT dtype {resolved_dtype}')
        expected_shape = (1, int(self.channel_count), int(self.out_size), int(self.out_size))
        if (
            self._direct_ring is None
            or any(
                slot.input.dtype != resolved_dtype
                or tuple(int(x) for x in slot.input.shape) != expected_shape
                for slot in self._direct_ring
            )
        ):
            self._direct_ring = [
                _ResidentGpuPipelineSlot(
                    torch, self.engine.device, int(self.out_size), int(self.channel_count),
                    resolved_dtype, slot_id=i,
                )
                for i in range(2)
            ]
        for slot in self._direct_ring:
            slot.render_graph = None
            slot.render_graph_key = None
            slot.render_expected_key = None
        self.engine.render_tile_into_ring_slot(
            self._direct_ring[0], self.view, self.tile_affine, int(self.slice_offset),
            out_size=int(self.out_size), channel_format=self.channel_format,
        )
        self.engine._stream.synchronize()
        self._direct_count = 0
        self.count = 0
        return self._direct_ring

    def reset_direct_ring(self) -> None:
        self._direct_count = 0
        self.count = 0
        self._direct_ring = None

    def next_direct_slot(self) -> Optional[Tuple[int, _ResidentGpuPipelineSlot]]:
        if self._direct_ring is None:
            raise RuntimeError('direct resident tile ring has not been prepared')
        if int(self._direct_count) >= int(self.nf):
            return None
        local_index = int(self._direct_count)
        absolute_index = int(self.slice_offset) + int(local_index)
        slot = self._direct_ring[local_index & 1]
        slot.frame_index = int(local_index)
        slot.absolute_index = int(absolute_index)
        slot.synthetic = False
        self.engine.render_tile_into_ring_slot(
            slot, self.view, self.tile_affine, int(absolute_index),
            out_size=int(self.out_size), channel_format=self.channel_format,
        )
        self._direct_count += 1
        return int(local_index), slot

    def __next__(self) -> Tuple[List[str], object, List[str]]:
        if self.count >= self.yield_nf or self.nf <= 0:
            raise StopIteration
        start = int(self.count)
        stop = min(int(self.yield_nf), start + int(self.bs))
        self.count = int(stop)
        paths: List[str] = []
        info: List[str] = []
        absolute_indices: List[int] = []
        last_real_idx = max(0, int(self.nf) - 1)
        for idx in range(start, stop):
            real_idx = int(idx) if int(idx) < int(self.nf) else int(last_real_idx)
            synthetic = int(idx) >= int(self.nf)
            suffix = '_synthetic' if synthetic else ''
            absolute_indices.append(int(self.slice_offset) + int(real_idx))
            paths.append(f'{self.name}_{idx + 1:06d}{suffix}.png')
            if synthetic:
                info.append(
                    f'gpu-tile {self.name} synthetic padded slice {idx + 1}/{self.yield_nf} '
                    f'repeats real slice {real_idx + 1}/{self.nf}: '
                )
            else:
                info.append(f'gpu-tile {self.name} slice {idx + 1}/{self.nf}: ')
        gpu_tensor, ready_event = self.engine.render_tile_batch(
            self.view, self.tile_affine, absolute_indices,
            out_size=int(self.out_size), fp16=bool(self.fp16),
            channel_format=self.channel_format,
        )
        batch = GpuPrefetchedYoloBatch(
            [self._fake_frame] * len(absolute_indices),
            gpu_tensor=gpu_tensor, ready_event=ready_event, source_label=self.name,
        )
        return paths, batch, info

_WORKER_GPU_RENDER_ENGINE: Optional[_GpuWorkerRenderEngine] = None

_WORKER_TILTED_RADIAL_CPU_WARNED: set[str] = set()

def _worker_gpu_render_engine() -> Optional[_GpuWorkerRenderEngine]:
    return _WORKER_GPU_RENDER_ENGINE

def _init_worker_gpu_render_engine(device_str: str = 'cuda:0') -> Optional[_GpuWorkerRenderEngine]:
    """Create this worker process's GPU render engine (call after the model/CUDA context exists)."""
    global _WORKER_GPU_RENDER_ENGINE
    if _WORKER_GPU_RENDER_ENGINE is not None:
        return _WORKER_GPU_RENDER_ENGINE
    if not gpu_worker_render_enabled():
        return None
    try:
        import torch  # type: ignore
        if not bool(torch.cuda.is_available()):
            return None
        _WORKER_GPU_RENDER_ENGINE = _GpuWorkerRenderEngine(str(device_str))
        print('GPU render engine initialized (v13.3.0 R1/R21); volume residency resolves on the first task.')
    except Exception as exc:
        print(f'Warning: GPU render engine unavailable ({exc}); worker uses CPU rendering.')
        _WORKER_GPU_RENDER_ENGINE = None
    return _WORKER_GPU_RENDER_ENGINE

def _radial_slab_context_indices(
    view: ViewInfo,
    center_start: int,
    center_count: int,
    channel_format: ChannelFormat,
) -> Tuple[int, ...]:
    """Return the sparse global plane bank required by one radial task window."""
    fmt = resolve_channel_format(channel_format)
    required = {
        channel_view_slice_index(view, center + int(offset))
        for center in range(int(center_start), int(center_start) + int(center_count))
        for offset in fmt.offsets
    }
    return tuple(sorted(int(value) for value in required))

def _radial_slab_channel_renderer(
    slab: np.ndarray,
    frame_indices: Sequence[int],
    view: ViewInfo,
    job: AugJob,
    *,
    center_start: int,
    channel_format: ChannelFormat,
) -> Callable[[int], np.ndarray]:
    """Apply the job affine to a sparse radial plane bank and assemble HWC stacks."""
    fmt = resolve_channel_format(channel_format)
    indices = tuple(int(value) for value in frame_indices)
    bank = np.asarray(slab, dtype=np.uint8)
    if bank.ndim != 3 or int(bank.shape[0]) != len(indices):
        raise ValueError(
            f'radial slab shape {bank.shape} does not match {len(indices)} frame indices'
        )
    if len(set(indices)) != len(indices):
        raise ValueError('radial slab frame indices must be unique')

    aff = job.aff
    identity = (
        int(aff.src_w) == int(aff.out_size)
        and int(aff.src_h) == int(aff.out_size)
        and int(aff.canvas_w) == int(aff.src_w)
        and int(aff.canvas_h) == int(aff.src_h)
        and float(aff.angle_deg) % 360.0 == 0.0
    )
    transformed = np.empty(
        (len(indices), int(aff.out_size), int(aff.out_size)), dtype=np.uint8,
    )
    for row, native_plane in enumerate(bank):
        plane = np.ascontiguousarray(native_plane, dtype=np.uint8)
        if identity:
            if plane.shape != (int(aff.out_size), int(aff.out_size)):
                raise ValueError(
                    f'identity radial slab plane has shape {plane.shape}, expected '
                    f'({int(aff.out_size)},{int(aff.out_size)})'
                )
            transformed[row] = plane
        else:
            transformed[row] = cv2.warpAffine(
                plane,
                aff.M_src_to_out,
                dsize=(int(aff.out_size), int(aff.out_size)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

    row_by_index = {value: row for row, value in enumerate(indices)}

    def _render_plane(global_idx: int) -> np.ndarray:
        try:
            return transformed[row_by_index[int(global_idx)]]
        except KeyError as exc:
            raise IndexError(
                f'global radial plane {int(global_idx)} was not prerendered'
            ) from exc

    formatted = ChannelFormattedFrameRenderer(
        _render_plane, view, fmt, cache_frames=0,
    )
    start = int(center_start)

    def _render_center(local_idx: int) -> np.ndarray:
        return formatted(start + int(local_idx))

    return _render_center

def _worker_render_callable(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: object,
    kind: str,
    slice_offset: int = 0,
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
) -> Callable[[int], np.ndarray]:
    # local index -> absolute view slice index (sub-range tasks render a contiguous slice window).
    off = int(slice_offset)
    if str(kind) == 'tile':
        formatted = make_dense_tile_channel_renderer(
            volume_rgb,
            view,
            job,  # type: ignore[arg-type]
            channel_format=resolve_channel_format(channel_format),
            view_frames=None,
        )
    else:
        formatted = make_fullframe_channel_renderer(
            volume_rgb,
            view,
            job,  # type: ignore[arg-type]
            channel_format=resolve_channel_format(channel_format),
            view_frames=None,
        )

    def _render_center(local_idx: int) -> np.ndarray:
        # Context is clamped against the GLOBAL ViewInfo range, not this worker's
        # task-local slice window.
        return formatted(off + int(local_idx))

    return _render_center


# Late imports keep callable-only dependency cycles import-safe.
from ._latebind import bind_late_symbols as _bind_late_symbols

_bind_late_symbols(
    __name__,
    globals(),
    {
        "backprojection": (
            "_ResidentTensorRTRingFatalError",
        ),
        "config": (
            "ChannelFormat",
            "GIB",
            "resolve_channel_format",
        ),
        "geometry": (
            "AffineSpec",
            "AugJob",
            "ChannelFormattedFrameRenderer",
            "DenseTileJob",
            "GpuPrefetchedYoloBatch",
            "RADIAL_FILTER_LABEL",
            "RADIAL_FILTER_MODE",
            "RADIAL_FILTER_TAP_COUNT",
            "ViewInfo",
            "_cupy_external_stream",
            "_tilted_plan_cache_key",
            "cartesian_view_axis_spec",
            "channel_view_slice_index",
            "ensure_ultralytics_accepts_in_memory_volume_source",
            "get_tilted_render_plan",
            "is_radial_view",
            "is_tilted_radial_view",
            "is_tilted_view",
            "make_dense_tile_channel_renderer",
            "make_fullframe_channel_renderer",
            "physical_view_name",
            "radial_base_view_name",
            "radial_fused_render_supported",
            "radial_plane_shape",
            "radial_resident_gpu_render_supported",
            "radial_stack_length",
            "radial_streaming_gpu_render_supported",
            "tilted_base_view_name",
            "tilted_frame_center",
            "tilted_stack_axis_length",
        ),
        "inference": (
            "_ResidentGpuPipelineSlot",
            "_affine_theta_from_dst_to_src",
            "_cuda_graph_capture_context",
            "_cuda_stream_priority",
            "_get_cached_affine_grid",
            "resident_trt_cuda_graphs_enabled",
        ),
        "runtime": (
            "choose_slice_parallel_workers",
            "flush_array",
            "parallel_for_indices",
        ),
        "workspace": (
            "_TILTED_IDENTITY_M",
            "_env_flag",
            "_env_float",
            "_env_int",
            "_tilted_grid_is_identity",
            "radial_source_mode",
            "tilted_inplane_linear_enabled",
        ),
    },
)
