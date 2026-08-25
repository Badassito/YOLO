"""D1 owner-GPU backprojection and packed source-space storage."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
import numpy as np

from .geometry import (
    ViewInfo,
    is_radial_view,
    is_tilted_radial_view,
    is_tilted_view,
    physical_view_name,
    pretty_view_name,
    radial_base_view_name,
    radial_stack_length,
    tilted_base_view_name,
    tilted_stack_axis_length,
)

# Explicit lower-layer dependencies keep imports one-way.
from .config import GIB
from .workspace import (
    _env_flag,
    _env_float,
    _env_int,
    available_anon_work_bytes,
    v1613_d1_pipeline_active,
)
from .runtime import (
    _memfd_backing_path_from_array,
    close_memmap_array,
    runtime_telemetry,
)
from .interpolation import (
    CVOL_FORMAT,
    INTERNAL_PACKED_CVOL_FORMAT,
    IncrementalRawBBoxMaskStoreWriter,
    NrrdLayerRef,
    RawBBoxMaskStore,
    RawBBoxSlicePayload,
    _coerce_segment_extent,
    _encode_bool_mask_slice_payload,
    _nrrd_empty_segment_extent,
    _write_raw_bbox_payload_store,
    write_raw_bbox_mask_store,
)

def d1_owner_pipeline_enabled() -> bool:
    """Return the D1 mode admitted and published by the current run."""
    return bool(v1613_d1_pipeline_active())

def d1_owner_bitset_reserve_bytes() -> int:
    return int(max(0.5, _env_float('YOLO_TTA_D1_OWNER_RESERVE_GIB', 2.0)) * GIB)

def d1_publication_credits_per_worker() -> int:
    return max(1, min(8, _env_int('YOLO_TTA_D1_PUBLICATION_CREDITS', 3)))

def d1_publication_max_pending_per_worker() -> int:
    """Bound completed host bitsets without coupling the bound to CPU encoder threads."""
    # An uneven C3 assignment must not make the GPU wait for a path-backed cvol encoder.
    # The prioritized run is still globally bounded by its 30 views.
    default = 12
    return max(
        d1_publication_credits_per_worker(),
        min(32, _env_int('YOLO_TTA_D1_PUBLICATION_MAX_PENDING', default)),
    )

def d1_unpack_target_mib() -> int:
    return max(16, min(1024, _env_int('YOLO_TTA_D1_UNPACK_TARGET_MIB', 256)))

_D1_BACKPROJECT_KERNELS: Optional[object] = None

_D1_BACKPROJECT_KERNELS_FAILED = False

_D1_BACKPROJECT_KERNELS_ERROR: Optional[str] = None

def _d1_backproject_kernels() -> object:
    """Compile the source-geometry bitset backprojection kernel once per worker."""
    global _D1_BACKPROJECT_KERNELS, _D1_BACKPROJECT_KERNELS_FAILED
    global _D1_BACKPROJECT_KERNELS_ERROR
    if _D1_BACKPROJECT_KERNELS is not None:
        return _D1_BACKPROJECT_KERNELS
    if _D1_BACKPROJECT_KERNELS_FAILED:
        raise RuntimeError(
            'D1 backprojection kernels are unavailable: '
            + str(_D1_BACKPROJECT_KERNELS_ERROR or 'unknown NVRTC failure')
        )
    # This NVRTC unit is deliberately header-free. Rootless cluster CUDA installs can
    # expose libnvrtc without the host libc/C++ include tree, so including stdint.h or
    # math.h can fail even though this kernel needs only CUDA intrinsics and primitive types.
    source = r'''
    __device__ __forceinline__ int d1_round_index(float value, int count) {
      int idx = __float2int_rn(value);
      if (idx < 0) return 0;
      if (idx >= count) return count - 1;
      return idx;
    }
    __device__ __forceinline__ float d1_scale_center(
        float value, int in_count, int out_count) {
      if (in_count <= 1 || out_count <= 1) return 0.0f;
      return (value + 0.5f) * ((float)out_count / (float)in_count) - 0.5f;
    }
    __device__ __forceinline__ bool d1_inside(float value, int count) {
      return value >= 0.0f && value <= (float)(count - 1);
    }

    extern "C" __global__ void d1_backproject_bbox_to_bits(
        const unsigned char* mask,
        int proc_h, int proc_w, int local_slice, int slice_start,
        int bbox_y0, int bbox_x0, int bbox_h, int bbox_w,
        int view_slices, int view_h, int view_w,
        int logical_t, int logical_h, int logical_w,
        int out_t, int out_h, int out_w,
        int family_id, int base_id, int direction_id,
        float tan_tilt, int stack_len,
        float center_x, float center_y, float roi_radius,
        const float* angle_cos, const float* angle_sin,
        unsigned int* output_bits) {
      unsigned long long q =
          (unsigned long long)blockDim.x * (unsigned long long)blockIdx.x
          + (unsigned long long)threadIdx.x;
      unsigned long long total =
          (unsigned long long)bbox_h * (unsigned long long)bbox_w;
      if (q >= total) return;

      int py_i = bbox_y0 + (int)(q / (unsigned long long)bbox_w);
      int px_i = bbox_x0 + (int)(q % (unsigned long long)bbox_w);
      if (py_i < 0 || py_i >= proc_h || px_i < 0 || px_i >= proc_w) return;
      unsigned long long mask_index =
          ((unsigned long long)local_slice * (unsigned long long)proc_h
           + (unsigned long long)py_i) * (unsigned long long)proc_w
          + (unsigned long long)px_i;
      if (mask[mask_index] == 0) return;

      int slice_index = slice_start + local_slice;
      if (slice_index < 0 || slice_index >= view_slices) return;

      float native_x = d1_scale_center((float)px_i, proc_w, view_w);
      float native_y = d1_scale_center((float)py_i, proc_h, view_h);
      float plane_x = native_x;
      float plane_y = native_y;
      float stack = (float)slice_index;

      if (family_id == 2) {
        // Radial mask axes are (stack row, diameter sample); slice_index is azimuth.
        float line = view_w > 1
            ? -roi_radius + (2.0f * roi_radius) * (native_x / (float)(view_w - 1))
            : -roi_radius;
        plane_x = center_x + line * angle_cos[slice_index];
        plane_y = center_y + line * angle_sin[slice_index];
        stack = d1_scale_center(native_y, view_h, stack_len);
      }

      if (family_id == 1 || (family_id == 2 && tan_tilt != 0.0f)) {
        float axis = direction_id == 0
            ? plane_y - center_y
            : plane_x - center_x;
        stack += tan_tilt * axis;
      }

      float wt, wy, wx;
      if (base_id == 0) {
        wt = stack; wy = plane_y; wx = plane_x;
      } else if (base_id == 1) {
        wt = plane_y; wy = stack; wx = plane_x;
      } else {
        wt = plane_y; wy = plane_x; wx = stack;
      }
      if (!d1_inside(wt, logical_t)
          || !d1_inside(wy, logical_h)
          || !d1_inside(wx, logical_w)) return;

      int oz = d1_round_index(d1_scale_center(wt, logical_t, out_t), out_t);
      int oy = d1_round_index(d1_scale_center(wy, logical_h, out_h), out_h);
      int ox = d1_round_index(d1_scale_center(wx, logical_w, out_w), out_w);
      unsigned long long linear =
          ((unsigned long long)oz * (unsigned long long)out_h + (unsigned long long)oy)
          * (unsigned long long)out_w + (unsigned long long)ox;
      unsigned long long word = linear >> 5;
      unsigned int bit = 1u << (unsigned int)(linear & 31ull);
      atomicOr(output_bits + word, bit);
    }
    '''
    try:
        import cupy as cp  # type: ignore
        module = cp.RawModule(
            code=source,
            options=('--std=c++14',),
            name_expressions=('d1_backproject_bbox_to_bits',),
        )
        compile_fn = getattr(module, 'compile', None)
        if callable(compile_fn):
            compile_fn()
        _D1_BACKPROJECT_KERNELS = argparse.Namespace(
            cp=cp,
            module=module,
            d1_backproject_bbox_to_bits=module.get_function('d1_backproject_bbox_to_bits'),
        )
        _D1_BACKPROJECT_KERNELS_ERROR = None
        return _D1_BACKPROJECT_KERNELS
    except Exception as exc:
        _D1_BACKPROJECT_KERNELS_FAILED = True
        _D1_BACKPROJECT_KERNELS_ERROR = f'{type(exc).__name__}: {exc}'
        raise RuntimeError(
            'D1 backprojection header-free NVRTC compilation failed: '
            + str(_D1_BACKPROJECT_KERNELS_ERROR)
        ) from exc

@dataclass
class _D1WorkerViewState:
    key: Tuple[str, str]
    view: ViewInfo
    output_shape: Tuple[int, int, int]
    bitset: object
    coverage: np.ndarray
    angle_cos: object
    angle_sin: object
    store_dir: Path
    created_at: float
    view_shadow_writer: Optional[IncrementalRawBBoxMaskStoreWriter] = None
    view_shadow_store_dir: Optional[Path] = None
    view_shadow_shape: Optional[Tuple[int, int, int]] = None

_D1_WORKER_VIEW_STATES: Dict[Tuple[str, str], _D1WorkerViewState] = {}

_D1_WORKER_VIEW_LOCK = threading.RLock()

_D1_PUBLICATION_EXECUTOR: Optional[ThreadPoolExecutor] = None

_D1_PUBLICATION_SEMAPHORE: Optional[threading.BoundedSemaphore] = None

_D1_PIPELINE_ANNOUNCED = False

def _d1_publication_executor() -> ThreadPoolExecutor:
    global _D1_PUBLICATION_EXECUTOR, _D1_PUBLICATION_SEMAPHORE
    if _D1_PUBLICATION_EXECUTOR is None:
        credits = int(d1_publication_credits_per_worker())
        _D1_PUBLICATION_EXECUTOR = ThreadPoolExecutor(
            max_workers=credits,
            thread_name_prefix='d1-cvol-publication',
        )
        _D1_PUBLICATION_SEMAPHORE = threading.BoundedSemaphore(
            int(d1_publication_max_pending_per_worker())
        )
    return _D1_PUBLICATION_EXECUTOR

def _shutdown_d1_worker_pipeline() -> None:
    global _D1_PUBLICATION_EXECUTOR, _D1_PUBLICATION_SEMAPHORE
    with _D1_WORKER_VIEW_LOCK:
        states = list(_D1_WORKER_VIEW_STATES.values())
        _D1_WORKER_VIEW_STATES.clear()
    for state in states:
        try:
            state.bitset = None
        except Exception:
            pass
        shadow_writer = state.view_shadow_writer
        state.view_shadow_writer = None
        if shadow_writer is not None:
            try:
                shadow_writer.discard()
            except Exception:
                pass
    executor = _D1_PUBLICATION_EXECUTOR
    _D1_PUBLICATION_EXECUTOR = None
    _D1_PUBLICATION_SEMAPHORE = None
    if executor is not None:
        try:
            executor.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            executor.shutdown(wait=True)

def _d1_view_family_ids(view: ViewInfo) -> Tuple[int, int, int, float, int, float, float]:
    """Return family/base/direction/shear/stack/center metadata for the CUDA kernel."""
    if is_radial_view(view):
        family_id = 2
        base = radial_base_view_name(view)
        tilted = bool(is_tilted_radial_view(view))
        direction = str(view.tilt_direction or 'vertical')
        tan_tilt = (
            math.tan(math.radians(float(view.tilt_angle_deg))) if tilted else 0.0
        )
        stack_len = int(radial_stack_length(view))
        center_x = float(view.center_x)
        center_y = float(view.center_y)
    elif is_tilted_view(view):
        family_id = 1
        base = tilted_base_view_name(view)
        direction = str(view.tilt_direction or 'vertical')
        tan_tilt = math.tan(math.radians(float(view.tilt_angle_deg)))
        stack_len = int(tilted_stack_axis_length(view))
        center_x = float((int(view.src_w) - 1) * 0.5)
        center_y = float((int(view.src_h) - 1) * 0.5)
    else:
        family_id = 0
        base = physical_view_name(view)
        direction = 'vertical'
        tan_tilt = 0.0
        stack_len = int(view.num_slices)
        center_x = float((int(view.src_w) - 1) * 0.5)
        center_y = float((int(view.src_h) - 1) * 0.5)
    base_ids = {'transverse': 0, 'sagittal': 1, 'coronal': 2}
    if str(base) not in base_ids:
        raise ValueError(f'D1 does not support base view {base!r}')
    if direction not in ('vertical', 'horizontal'):
        raise ValueError(f'D1 does not support tilt direction {direction!r}')
    return (
        int(family_id), int(base_ids[str(base)]),
        int(0 if direction == 'vertical' else 1),
        float(tan_tilt), int(stack_len), float(center_x), float(center_y),
    )

def _d1_get_or_create_state(task: Dict[str, object], accumulator: '_DeviceUnionAccumulator') -> _D1WorkerViewState:
    global _D1_PIPELINE_ANNOUNCED
    view = task.get('view')
    if not isinstance(view, ViewInfo):
        raise TypeError('D1 task is missing ViewInfo metadata')
    key = (str(task.get('model_name', '')), str(view.name))
    output_shape = tuple(int(v) for v in task.get('d1_output_shape', ()))
    if len(output_shape) != 3 or any(int(v) <= 0 for v in output_shape):
        raise ValueError(f'D1 task {task.get("task_id")} has invalid output shape {output_shape}')
    raw_store_dir = str(task.get('d1_store_dir', '') or '').strip()
    if not raw_store_dir:
        raise ValueError(f'D1 task {task.get("task_id")} has no cvol store path')
    store_dir = Path(raw_store_dir)
    raw_shadow_dir = str(task.get('d1_view_shadow_store_dir', '') or '').strip()
    shadow_store_dir = Path(raw_shadow_dir) if raw_shadow_dir else None
    shadow_shape_raw = tuple(int(v) for v in task.get('d1_view_shadow_shape', ()))
    shadow_shape: Optional[Tuple[int, int, int]] = None
    if shadow_store_dir is not None:
        if len(shadow_shape_raw) != 3 or any(int(v) <= 0 for v in shadow_shape_raw):
            raise ValueError(
                f'D1 task {task.get("task_id")} has invalid view-shadow shape {shadow_shape_raw}'
            )
        shadow_shape = (
            int(shadow_shape_raw[0]), int(shadow_shape_raw[1]), int(shadow_shape_raw[2]),
        )
        if int(shadow_shape[0]) != int(view.num_slices):
            raise ValueError(
                f'D1 view-shadow depth {shadow_shape[0]} != view depth {view.num_slices}'
            )

    with _D1_WORKER_VIEW_LOCK:
        existing = _D1_WORKER_VIEW_STATES.get(key)
        if existing is not None:
            if existing.output_shape != output_shape:
                raise RuntimeError(
                    f'D1 owner state {key} output shape changed '
                    f'{existing.output_shape} -> {output_shape}'
                )
            if existing.view_shadow_store_dir != shadow_store_dir or existing.view_shadow_shape != shadow_shape:
                raise RuntimeError(
                    f'D1 owner state {key} view-shadow contract changed '
                    f'{existing.view_shadow_store_dir}/{existing.view_shadow_shape} -> '
                    f'{shadow_store_dir}/{shadow_shape}'
                )
            return existing
        if _D1_WORKER_VIEW_STATES:
            active = ', '.join(f'{m}/{v}' for m, v in _D1_WORKER_VIEW_STATES)
            raise RuntimeError(
                f'D1 worker received {key} while another owner view is active ({active}); '
                'the scheduler must keep one source-space bitset per worker'
            )

        import torch  # type: ignore
        kernels = _d1_backproject_kernels()
        total_voxels = int(output_shape[0]) * int(output_shape[1]) * int(output_shape[2])
        word_count = int((total_voxels + 31) // 32)
        bitset_bytes = int(word_count) * np.dtype(np.uint32).itemsize
        free_bytes, _total_bytes = torch.cuda.mem_get_info(accumulator.device)
        if int(free_bytes) < int(bitset_bytes) + int(d1_owner_bitset_reserve_bytes()):
            raise RuntimeError(
                f'D1 owner bitset for {key} needs {bitset_bytes / GIB:.2f} GiB plus '
                f'{d1_owner_bitset_reserve_bytes() / GIB:.2f} GiB reserve, but only '
                f'{int(free_bytes) / GIB:.2f} GiB is free'
            )
        bitset = torch.zeros(
            (int(word_count),), dtype=torch.int32, device=accumulator.device,
        )
        if is_radial_view(view):
            radians = np.deg2rad(
                np.ascontiguousarray(np.asarray(view.azimuths_deg, dtype=np.float32)).astype(np.float64)
            ).astype(np.float32)
            if int(radians.size) != int(view.num_slices):
                raise RuntimeError(
                    f'D1 Radial angle table has {int(radians.size)} entries for '
                    f'{int(view.num_slices)} slices'
                )
            angle_cos = kernels.cp.asarray(np.ascontiguousarray(np.cos(radians), dtype=np.float32))
            angle_sin = kernels.cp.asarray(np.ascontiguousarray(np.sin(radians), dtype=np.float32))
        else:
            angle_cos = kernels.cp.asarray(np.ones((1,), dtype=np.float32))
            angle_sin = kernels.cp.asarray(np.zeros((1,), dtype=np.float32))
        shadow_writer: Optional[IncrementalRawBBoxMaskStoreWriter] = None
        if shadow_store_dir is not None and shadow_shape is not None:
            shadow_writer = IncrementalRawBBoxMaskStoreWriter(
                shape=shadow_shape,
                store_dir=shadow_store_dir,
                format_name=INTERNAL_PACKED_CVOL_FORMAT,
                desc=f'D1 view-native shadow {key[0]}/{key[1]}',
                force_path_backed=True,
                extra_meta={
                    'producer': 'v17.0.5_d1_sparse_view_shadow',
                    'purpose': 'interpolation_and_tile_parent_compatibility',
                    'physical_view': physical_view_name(view),
                    'tta_aug_id': str(view.tta_aug_id),
                    'tta_angle_deg': float(view.tta_angle_deg),
                },
            )
        state = _D1WorkerViewState(
            key=key,
            view=view,
            output_shape=output_shape,
            bitset=bitset,
            coverage=np.zeros((int(view.num_slices),), dtype=bool),
            angle_cos=angle_cos,
            angle_sin=angle_sin,
            store_dir=store_dir,
            created_at=time.perf_counter(),
            view_shadow_writer=shadow_writer,
            view_shadow_store_dir=shadow_store_dir,
            view_shadow_shape=shadow_shape,
        )
        _D1_WORKER_VIEW_STATES[key] = state
        if not _D1_PIPELINE_ANNOUNCED:
            _D1_PIPELINE_ANNOUNCED = True
            print(
                'v16.1.3 D1 owner-GPU pipeline active: each worker keeps one '
                'source-geometry uint32 bitset, backprojects every completed inference '
                'lease immediately, and publishes one sparse cvol after view completion.'
            )
        print(
            f'D1 owner state opened for {key[0]}/{key[1]} on {accumulator.device}: '
            f'{word_count:,} uint32 words ({bitset_bytes / GIB:.2f} GiB), '
            f'output_shape={output_shape}, '
            f'view_shadow={str(shadow_store_dir) if shadow_store_dir is not None else "disabled"}.'
        )
        return state

def _d1_unpack_bitset_z_block(
    words: np.ndarray,
    output_shape: Tuple[int, int, int],
    z0: int,
    z1: int,
) -> np.ndarray:
    """Decode one source-z block without materializing the complete dense volume."""
    out_t, out_h, out_w = (int(v) for v in output_shape)
    start = int(z0) * int(out_h) * int(out_w)
    stop = int(z1) * int(out_h) * int(out_w)
    if start < 0 or stop < start or stop > int(out_t) * int(out_h) * int(out_w):
        raise IndexError(f'D1 bit range [{start},{stop}) is outside {output_shape}')
    w0 = int(start // 32)
    w1 = int((stop + 31) // 32)
    selected = np.ascontiguousarray(np.asarray(words[w0:w1], dtype=np.uint32))
    if sys.byteorder != 'little':
        selected = selected.byteswap().newbyteorder('<')
    unpacked = np.unpackbits(selected.view(np.uint8), bitorder='little')
    offset = int(start - w0 * 32)
    count = int(stop - start)
    return np.ascontiguousarray(
        unpacked[offset:offset + count].reshape(int(z1 - z0), int(out_h), int(out_w)),
        dtype=np.uint8,
    )

def _d1_finalize_bitset_layer(
    *,
    words: np.ndarray,
    output_shape: Tuple[int, int, int],
    store_dir: Path,
    model_name: str,
    view: ViewInfo,
) -> Dict[str, object]:
    """Stream the completed owner bitset into a path-backed cvol and return its layer ref."""
    key = _nrrd_layer_key(
        view_name=str(view.name), source='fullframe', mask_kind='yolo',
        pass_index=0, stage='pre_interpolation',
    )
    writer = IncrementalRawBBoxMaskStoreWriter(
        shape=tuple(int(v) for v in output_shape),
        store_dir=Path(store_dir),
        format_name=CVOL_FORMAT,
        desc=f'D1 source-space layer {model_name}/{view.name}',
        force_path_backed=True,
        extra_meta={
            'nrrd_layer_key': str(key),
            'producer': 'v16.1.3_d1_owner_gpu_bitset',
            'projection_payload_fusion': 'project_infer_proto_close_backproject',
            'source_geometry_bitset': True,
        },
    )
    plane_bytes = max(1, int(output_shape[1]) * int(output_shape[2]))
    target_bytes = int(d1_unpack_target_mib()) * 1024 * 1024
    block_z = max(1, min(int(output_shape[0]), int(target_bytes // plane_bytes)))
    try:
        for z0 in range(0, int(output_shape[0]), int(block_z)):
            z1 = min(int(output_shape[0]), int(z0) + int(block_z))
            start = int(z0) * int(plane_bytes)
            stop = int(z1) * int(plane_bytes)
            w0 = int(start // 32)
            w1 = int((stop + 31) // 32)
            if not bool(np.any(words[w0:w1])):
                writer.consume_empty_range(int(z0), int(z1 - z0))
                continue
            block = _d1_unpack_bitset_z_block(words, output_shape, int(z0), int(z1))
            writer.consume(int(z0), block)
            del block
        stats = dict(writer.finalize())
    except BaseException as exc:
        writer.abort(exc)
        writer.discard()
        raise
    finally:
        # Permit the 2+ GiB host word copy to be reclaimed before NRRD compression starts.
        try:
            words.resize((0,), refcheck=False)
        except Exception:
            pass

    extent = _coerce_segment_extent(stats.get('segment_extent_ijk')) or _nrrd_empty_segment_extent()
    ref = NrrdLayerRef(
        key=str(key),
        name=_nrrd_layer_name(
            view=view, source='fullframe', mask_kind='yolo',
            pass_index=0, stage='pre_interpolation',
        ),
        path=Path(store_dir),
        shape=tuple(int(v) for v in output_shape),
        dtype='uint8',
        storage_format=CVOL_FORMAT,
        model_name=str(model_name),
        view_name=str(view.name),
        physical_view_name=physical_view_name(view),
        aug_id=str(view.tta_aug_id),
        angle_deg=float(view.tta_angle_deg),
        view_family=str(view.family),
        source='fullframe',
        mask_kind='yolo',
        pass_index=0,
        stage='pre_interpolation',
        description=(
            'Angle-variant YOLO mask with resident proto closing and immediate '
            'owner-GPU backprojection into source geometry.'
        ),
        segment_extent_ijk=extent,
        segment_extent_shape_tyx=tuple(int(v) for v in output_shape),
        segment_extent_source='d1_incremental_source_space_cvol_index',
    )
    return {
        'd1_layer_ref': ref,
        'd1_cvol_stats': stats,
        'd1_publication_seconds': 0.0,
    }

def _d1_submit_publication(
    *,
    words: np.ndarray,
    state: _D1WorkerViewState,
) -> Future:
    executor = _d1_publication_executor()
    semaphore = _D1_PUBLICATION_SEMAPHORE
    if semaphore is None:
        raise RuntimeError('D1 publication semaphore was not initialized')
    semaphore.acquire()

    def _publish() -> Dict[str, object]:
        started = time.perf_counter()
        try:
            result = _d1_finalize_bitset_layer(
                words=words,
                output_shape=state.output_shape,
                store_dir=state.store_dir,
                model_name=state.key[0],
                view=state.view,
            )
            result['d1_publication_seconds'] = max(0.0, time.perf_counter() - started)
            return result
        finally:
            semaphore.release()

    try:
        return executor.submit(_publish)
    except BaseException:
        semaphore.release()
        raise

def _d1_consume_device_union(
    task: Dict[str, object],
    accumulator: '_DeviceUnionAccumulator',
) -> Dict[str, object]:
    """Backproject one completed task union into its persistent owner bitset."""
    if accumulator.union_dev is None:
        raise RuntimeError('D1 received an already-retired device union')
    if bool(accumulator.host_written):
        raise RuntimeError(
            'D1 forbids host-written fallback slices; disable YOLO_TTA_V1613_FAST_BUNDLE '
            'to use the dense compatibility path'
        )
    view = task.get('view')
    if not isinstance(view, ViewInfo):
        raise TypeError('D1 task has no ViewInfo')
    state = _d1_get_or_create_state(task, accumulator)
    s0 = int(task.get('slice_start', 0))
    count = int(task.get('slice_count', 0))
    s1 = int(s0 + count)
    if s0 < 0 or count <= 0 or s1 > int(view.num_slices):
        raise ValueError(f'D1 task slice range [{s0},{s1}) is outside {view.num_slices}')
    if bool(np.any(state.coverage[s0:s1])):
        duplicate = int(s0 + np.flatnonzero(state.coverage[s0:s1])[0])
        raise RuntimeError(f'D1 owner {state.key} received duplicate source slice {duplicate}')

    kernels = _d1_backproject_kernels()
    cp = kernels.cp
    family_id, base_id, direction_id, tan_tilt, stack_len, center_x, center_y = (
        _d1_view_family_ids(view)
    )
    mask_shape = tuple(int(v) for v in accumulator.union_dev.shape)
    if mask_shape[0] != count:
        raise RuntimeError(
            f'D1 task union depth {mask_shape[0]} != dispatched slice count {count}'
        )
    # B1 metadata is already computed from the committed device union. Use it as the
    # launch domain: a 128-slice 3072^2 task would otherwise launch ~1.2 billion threads
    # merely to rediscover zeros. Empty slices launch no kernel and each nonempty slice scans
    # only its known foreground bbox.
    slice_meta = accumulator.compute_slice_metadata()
    if not isinstance(slice_meta, dict):
        raise RuntimeError('D1 requires valid device slice metadata for bbox-limited backprojection')
    slice_any = np.asarray(slice_meta.get('slice_any'), dtype=bool)
    slice_bboxes = np.asarray(slice_meta.get('slice_bboxes'), dtype=np.int64)
    if slice_any.shape != (int(count),) or slice_bboxes.shape != (int(count), 4):
        raise RuntimeError(
            f'D1 metadata shape mismatch: any={slice_any.shape}, bboxes={slice_bboxes.shape}, '
            f'expected ({count},) and ({count},4)'
        )

    stream = cp.cuda.get_current_stream()
    mask_cp = cp.asarray(accumulator.union_dev)
    bitset_cp = cp.asarray(state.bitset)
    scanned_bbox_pixels = 0
    nonempty_slices = 0
    for local_slice in np.flatnonzero(slice_any):
        y0, y1, x0, x1 = (int(v) for v in slice_bboxes[int(local_slice)])
        y0 = max(0, min(int(mask_shape[1]), y0))
        y1 = max(y0, min(int(mask_shape[1]), y1))
        x0 = max(0, min(int(mask_shape[2]), x0))
        x1 = max(x0, min(int(mask_shape[2]), x1))
        bbox_h = int(y1 - y0)
        bbox_w = int(x1 - x0)
        bbox_pixels = int(bbox_h) * int(bbox_w)
        if bbox_pixels <= 0:
            raise RuntimeError(
                f'D1 metadata marks local slice {int(local_slice)} nonempty but gives '
                f'empty bbox {(y0, y1, x0, x1)}'
            )
        kernels.d1_backproject_bbox_to_bits(
            ((int(bbox_pixels) + 255) // 256,), (256,),
            (
                mask_cp,
                np.int32(mask_shape[1]), np.int32(mask_shape[2]),
                np.int32(local_slice), np.int32(s0),
                np.int32(y0), np.int32(x0), np.int32(bbox_h), np.int32(bbox_w),
                np.int32(view.num_slices), np.int32(view.src_h), np.int32(view.src_w),
                np.int32(view.full_t), np.int32(view.full_h), np.int32(view.full_w),
                np.int32(state.output_shape[0]), np.int32(state.output_shape[1]),
                np.int32(state.output_shape[2]),
                np.int32(family_id), np.int32(base_id), np.int32(direction_id),
                np.float32(tan_tilt), np.int32(stack_len),
                np.float32(center_x), np.float32(center_y), np.float32(view.roi_radius),
                state.angle_cos, state.angle_sin,
                bitset_cp,
            ),
            stream=stream,
        )
        scanned_bbox_pixels += int(bbox_pixels)
        nonempty_slices += 1
    stream.synchronize()
    shadow_writer = state.view_shadow_writer
    if shadow_writer is not None:
        crop_specs: Dict[int, Tuple[int, int, int, int, int, int]] = {}
        crop_device_parts: List[object] = []
        packed_size = 0
        for local_slice in np.flatnonzero(slice_any):
            local_i = int(local_slice)
            global_slice = int(s0 + local_i)
            y0, y1, x0, x1 = (int(v) for v in slice_bboxes[local_i])
            y0 = max(0, min(int(mask_shape[1]), y0))
            y1 = max(y0, min(int(mask_shape[1]), y1))
            x0 = max(0, min(int(mask_shape[2]), x0))
            x1 = max(x0, min(int(mask_shape[2]), x1))
            size = int(y1 - y0) * int(x1 - x0)
            if size <= 0:
                raise RuntimeError(
                    f'D1 view shadow received nonempty slice {global_slice} with empty bbox '
                    f'{(y0, y1, x0, x1)}'
                )
            crop_specs[local_i] = (
                int(global_slice), int(y0), int(y1), int(x0), int(x1), int(size),
            )
            crop_device_parts.append(
                mask_cp[local_i, int(y0):int(y1), int(x0):int(x1)].reshape(-1)
            )
            packed_size += int(size)

        if crop_device_parts:
            packed_device = (
                crop_device_parts[0]
                if len(crop_device_parts) == 1
                else cp.concatenate(crop_device_parts)
            )
            packed_host = np.ascontiguousarray(cp.asnumpy(packed_device), dtype=np.uint8)
            del packed_device
        else:
            packed_host = np.empty((0,), dtype=np.uint8)
        del crop_device_parts
        if int(packed_host.size) != int(packed_size):
            raise RuntimeError(
                f'D1 packed view-shadow transfer produced {int(packed_host.size)} byte(s), '
                f'expected {int(packed_size)}'
            )

        cursor = 0
        for local_slice in range(int(count)):
            spec = crop_specs.get(int(local_slice))
            if spec is None:
                shadow_writer.consume_empty_range(int(s0 + local_slice), 1)
                continue
            global_slice, y0, y1, x0, x1, size = spec
            shadow_writer.consume_sparse_slice(
                int(global_slice), int(y0), int(y1), int(x0), int(x1),
                packed_host[int(cursor):int(cursor + size)].reshape(
                    (int(y1 - y0), int(x1 - x0)),
                ),
            )
            cursor += int(size)
        if int(cursor) != int(packed_host.size):
            raise RuntimeError(
                f'D1 packed view-shadow cursor {int(cursor)} != payload {int(packed_host.size)}'
            )
        runtime_telemetry().add('d1.view_shadow_packed_d2h_bytes', int(packed_host.nbytes))
        del packed_host, crop_specs
    state.coverage[s0:s1] = True
    covered = int(np.count_nonzero(state.coverage))
    complete = bool(covered == int(view.num_slices))
    runtime_telemetry().add('d1.backprojected_task_slices', int(count))
    runtime_telemetry().add('d1.nonempty_task_slices', int(nonempty_slices))
    runtime_telemetry().add('d1.scanned_bbox_pixels', int(scanned_bbox_pixels))

    # The task-local mask is no longer needed; release it before the next render/TRT lease.
    accumulator.union_dev = None
    accumulator.conf_dev = None
    accumulator.prediction_counts_dev = None

    result: Dict[str, object] = {
        'd1_view_complete': bool(complete),
        'd1_covered_slices': int(covered),
        'd1_total_slices': int(view.num_slices),
        'd1_backprojected_task_slices': int(count),
        'd1_nonempty_task_slices': int(nonempty_slices),
        'd1_scanned_bbox_pixels': int(scanned_bbox_pixels),
        'slice_meta': slice_meta,
    }
    if not complete:
        return result

    if state.view_shadow_writer is not None:
        shadow_stats = dict(state.view_shadow_writer.finalize())
        state.view_shadow_writer = None
        result['d1_view_shadow_path'] = str(state.view_shadow_store_dir)
        result['d1_view_shadow_format'] = INTERNAL_PACKED_CVOL_FORMAT
        result['d1_view_shadow_stats'] = shadow_stats
        print(
            f'D1 sparse view shadow complete for {state.key[0]}/{state.key[1]}: '
            f'{int(shadow_stats.get("nonempty_slices", 0))}/{int(view.num_slices)} nonempty '
            f'slice(s), payload={int(shadow_stats.get("raw_payload_bytes", 0)) / GIB:.3f} GiB.'
        )

    # One D2H at view completion; sparse cvol construction proceeds on an independent CPU
    # publication credit while this GPU immediately accepts the next owner view.
    # The CuPy backprojection stream was synchronized above. A direct Torch CPU copy is
    # therefore sufficient; retain the tensor-backed NumPy view for publication instead of
    # performing a second 2+ GiB host-to-host copy.
    host_words_tensor = state.bitset.detach().cpu()
    words = host_words_tensor.numpy().view(np.uint32)
    with _D1_WORKER_VIEW_LOCK:
        removed = _D1_WORKER_VIEW_STATES.pop(state.key, None)
        if removed is not state:
            raise RuntimeError(f'D1 owner state {state.key} changed during completion')
    state.bitset = None
    bitset_word_count = int(words.size)
    future = _d1_submit_publication(words=words, state=state)
    result['_publication_future'] = future
    result['d1_bitset_words'] = int(bitset_word_count)
    result['d1_view_compute_seconds'] = max(0.0, time.perf_counter() - state.created_at)
    print(
        f'D1 owner state complete for {state.key[0]}/{state.key[1]}: '
        f'{covered}/{int(view.num_slices)} slices backprojected; sparse cvol publication '
        'continues on an independent CPU credit.'
    )
    return result

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

def _read_binary_volume_slice_crop_bool(
    volume: object,
    idx: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Read only one requested support crop, preserving sparse raw-bbox storage."""
    shape = _volume_shape_tuple(volume)
    z = int(idx)
    y0_i, y1_i, x0_i, x1_i = int(y0), int(y1), int(x0), int(x1)
    if not (
        0 <= z < int(shape[0])
        and 0 <= y0_i <= y1_i <= int(shape[1])
        and 0 <= x0_i <= x1_i <= int(shape[2])
    ):
        raise IndexError(
            f'binary-volume crop {(z, y0_i, y1_i, x0_i, x1_i)} outside {shape}'
        )
    out = np.zeros((int(y1_i - y0_i), int(x1_i - x0_i)), dtype=bool)
    if out.size <= 0:
        return out
    if isinstance(volume, RawBBoxMaskStore):
        decoded = volume.decode_slice_crop(int(z), dtype=np.uint8)
        if decoded is None:
            return out
        sy0, sx0, sy1, sx1, crop = decoded
        iy0 = max(int(y0_i), int(sy0))
        iy1 = min(int(y1_i), int(sy1))
        ix0 = max(int(x0_i), int(sx0))
        ix1 = min(int(x1_i), int(sx1))
        if int(iy1) <= int(iy0) or int(ix1) <= int(ix0):
            return out
        out[
            int(iy0 - y0_i):int(iy1 - y0_i),
            int(ix0 - x0_i):int(ix1 - x0_i),
        ] = np.asarray(crop)[
            int(iy0 - sy0):int(iy1 - sy0),
            int(ix0 - sx0):int(ix1 - sx0),
        ] > 0
        return out
    return np.asarray(
        _read_binary_volume_slice_u8(volume, int(z))[
            int(y0_i):int(y1_i), int(x0_i):int(x1_i)
        ],
        dtype=np.uint8,
    ) > 0

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
    memfd_path = _memfd_backing_path_from_array(arr)
    if memfd_path is not None:
        return memfd_path
    lazy_path = (
        getattr(arr, 'backing_path', None)
        if bool(getattr(arr, '_is_lazy_processing_cube', False))
        else None
    )
    if lazy_path is not None:
        try:
            return Path(lazy_path)
        except Exception:
            pass
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
    return _env_flag('YOLO_TTA_RAW_BBOX_NRRD_LAYERS', True)

def tile_intermediate_accumulators_prefer_memory() -> bool:
    """Keep tile staging/consolidation canvases in process-reopenable RAM by default."""
    return _env_flag('YOLO_TTA_TILE_ACCUMULATORS_IN_RAM', True)

def tile_intermediate_accumulator_reserve_bytes() -> int:
    """Return the anonymous-memory reserve used when admitting tile accumulators."""
    if os.environ.get('YOLO_TTA_TILE_ACCUMULATOR_RESERVE_GIB', '').strip():
        return int(max(0.0, _env_float('YOLO_TTA_TILE_ACCUMULATOR_RESERVE_GIB', 64.0)) * GIB)
    fraction = min(0.9, max(0.0, _env_float('YOLO_TTA_TILE_ACCUMULATOR_RESERVE_FRACTION', 0.15)))
    floor_bytes = int(max(0.0, _env_float('YOLO_TTA_TILE_ACCUMULATOR_RESERVE_MIN_GIB', 8.0)) * GIB)
    cap_bytes = int(max(0.0, _env_float('YOLO_TTA_TILE_ACCUMULATOR_RESERVE_MAX_GIB', 64.0)) * GIB)
    proportional = int(float(max(0, available_anon_work_bytes())) * float(fraction))
    return int(min(max(floor_bytes, proportional), max(floor_bytes, cap_bytes)))

def tile_dense_worker_result_limit_bytes() -> int:
    """Bound live dense per-tile GPU-worker result files before sparse retirement.

    The limit is a hard scheduling backpressure budget, not a storage target. A single tile
    larger than the budget is still admitted when no other dense tile result is live, so valid
    geometries cannot deadlock. Waiting parent/bridge masks are always converted to crop-local
    raw-bbox stores; there is no dense-waiting compatibility path in v16.4.3.
    """
    return int(max(1.0, _env_float('YOLO_TTA_TILE_DENSE_RESULT_MAX_GIB', 96.0)) * GIB)

def tile_dense_worker_result_limit_tasks() -> int:
    """Bound live dense tile mappings/fds even when each crop is individually small."""
    return max(1, _env_int('YOLO_TTA_TILE_DENSE_RESULT_MAX_TASKS', 64))

def tile_dense_worker_result_warn_seconds() -> float:
    """Retention age that emits a diagnostic warning; zero disables the warning."""
    return max(0.0, _env_float('YOLO_TTA_TILE_DENSE_RESULT_WARN_SECONDS', 120.0))

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
    # np.any tests the raw uint8 slice directly — the old dtype=bool asarray
    # made a full-slice cast COPY per slice (~25 GB of alloc+copy for a near-empty volume).
    for idx in range(int(mask_mm.shape[0])):
        if np.any(np.asarray(mask_mm[int(idx)])):
            return True
    return False

def _sanitize_nrrd_layer_token(value: object) -> str:
    token = re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(value).strip())
    token = token.strip('_')
    return token or 'unnamed'

def _nrrd_layer_key(
    *,
    view_name: str,
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    tile_config_id: str = '',
    tile_acceptance: str = '',
    stage: str = '',
) -> str:
    parts = [
        _sanitize_nrrd_layer_token(view_name),
        _sanitize_nrrd_layer_token(source),
        _sanitize_nrrd_layer_token(mask_kind),
    ]
    if int(pass_index) > 0:
        parts.append(f'pass{int(pass_index):02d}')
    if tile_config_id:
        parts.append(_sanitize_nrrd_layer_token(tile_config_id))
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
    tile_config_id: str = '',
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
    if tile_config_id:
        pieces.append(f'tile set {str(tile_config_id)}')
    if tile_acceptance:
        pieces.append(f'accepted by {str(tile_acceptance).replace("_", " ")}')
    if stage:
        pieces.append(str(stage).replace('_', ' '))
    return ' / '.join(pieces)
