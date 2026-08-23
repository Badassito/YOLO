"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

import gc
import json
import math
import mmap
import os
import re
import threading
import time
import multiprocessing as mp
from collections import OrderedDict
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from dataclasses import (
    dataclass,
    field,
    replace as dataclasses_replace,
)
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)
import numpy as np
from ._deps import cv2, ndi, tqdm

from .config import (
    GIB,
)
from .runtime import (
    runtime_telemetry_phase,
)

# Explicit lower-layer dependencies keep imports one-way.
from .config import SCRIPT_VERSION
from .workspace import (
    _cpu_count,
    _env_flag,
    _env_float,
    _env_int,
)
from .runtime import (
    _acquire_parallel_pool,
    _release_parallel_pool,
    allocate_workspace_array,
    choose_slice_parallel_workers,
    close_memmap_array,
    flush_array,
    parallel_for_indices,
    parallel_for_indices_chunked,
    runtime_telemetry,
)
from .geometry import (
    ViewInfo,
    coronal_block_cols,
    physical_view_name,
)
from .interpolation import (
    CVOL_FORMAT,
    IncrementalRawBBoxMaskStoreWriter,
    NrrdLayerRef,
    RawBBoxMaskStore,
    _coerce_segment_extent,
    _nrrd_empty_segment_extent,
    compiled_topology_kernels_enabled,
)
from .cuda_d1 import (
    _nrrd_layer_key,
    _nrrd_layer_name,
    _volume_shape_tuple,
)
from .topology import (
    SparseSliceLabelStore,
    _numba_keep_lut_apply_kernel,
    _numba_sparse_keep_lut_apply_kernel,
    binary_slice_bbox_coverage,
    binary_volume_slice_metadata,
    discard_binary_volume_slice_metadata,
    fill_3d_voids_inplace_streaming,
    gpu_slice_labeling_configured_devices,
    label_foreground_volume_streaming,
    register_binary_volume_slice_metadata,
    scan_binary_volume_slice_metadata,
    topology_slab_slices,
)
from .outputs import (
    _close_nrrd_layer_source,
    _drop_nrrd_raw_store_chunks_ram_cache,
    _ffv1_contiguous_segments,
    _madvise_array_mmap,
    _nrrd_layer_ref_is_raw_bbox_store,
    _open_nrrd_layer_ref,
    _read_layer_slice_in_output_shape,
    _restore_source_indices_for_output_z,
    compute_segment_extent_zyx,
    nrrd_layer_output_suffix,
    nrrd_layer_sink,
)
from .assembly import materialize_nrrd_global_layer

def _union_projected_layer_ref_into_volume(
    ref: 'NrrdLayerRef',
    vol_mm: np.ndarray,
    *,
    workers: int = 1,
    desc: str = 'Layer union',
) -> None:
    """Restore one projected component layer into the destination union.
    
    Reduced orthogonal stores and already-native stores share the same output-z traversal."""
    vol_shape = tuple(int(v) for v in vol_mm.shape)
    z_dim, h_dim, w_dim = vol_shape
    src = _open_nrrd_layer_ref(ref)
    try:
        if tuple(int(v) for v in ref.shape) != vol_shape:
            def _or_restored_slice(z_idx: int) -> None:
                restored = _read_layer_slice_in_output_shape(
                    src, (int(z_dim), int(h_dim), int(w_dim)), int(z_idx),
                )
                if bool(np.any(restored)):
                    vol_mm[int(z_idx), :, :] |= restored

            parallel_for_indices_chunked(
                int(z_dim),
                _or_restored_slice,
                max_workers=choose_slice_parallel_workers(int(workers), int(z_dim)),
                desc=f'{desc} [reduced restore]',
                show_progress=False,
                target_chunks_per_worker=2,
            )
            return
        if isinstance(src, RawBBoxMaskStore):
            index = src.index
            scratch_tls = threading.local()

            def _or_store_slice(z_idx: int) -> None:
                rec = index[int(z_idx)]
                if int(rec['kind']) != 1:
                    return
                buf = getattr(scratch_tls, 'buf', None)
                if buf is None:
                    buf = np.zeros((int(h_dim), int(w_dim)), dtype=np.uint8)
                    scratch_tls.buf = buf
                src.fill_decoded_slice_into(int(z_idx), buf)
                y0 = int(rec['y0']); x0 = int(rec['x0']); y1 = int(rec['y1']); x1 = int(rec['x1'])
                vol_mm[int(z_idx), y0:y1, x0:x1] |= buf[y0:y1, x0:x1]

            parallel_for_indices_chunked(
                int(z_dim),
                _or_store_slice,
                max_workers=choose_slice_parallel_workers(int(workers), int(z_dim)),
                desc=desc,
                show_progress=False,
                target_chunks_per_worker=2,
            )
        else:
            def _or_raw_slice(z_idx: int) -> None:
                vol_mm[int(z_idx)] |= np.asarray(src[int(z_idx)], dtype=np.uint8)

            parallel_for_indices_chunked(
                int(z_dim),
                _or_raw_slice,
                max_workers=choose_slice_parallel_workers(int(workers), int(z_dim)),
                desc=desc,
                show_progress=False,
                target_chunks_per_worker=2,
            )
    finally:
        try:
            _close_nrrd_layer_source(src)
        finally:
            _drop_nrrd_raw_store_chunks_ram_cache(src)

def scheduler_push_drain_enabled() -> bool:
    """Push-drain the GPU-worker result queue instead of timeout polling.

 A transport-only daemon thread blocks on the process result queue and hands messages
 to the main thread through a local deque + wake event; completed scheduler futures set
 the same event through one-time done-callbacks. Results are processed the instant they
 arrive instead of after the 0.1 s futures-poll / 0.5 s queue-poll, and handlers still
 run ONLY on the main thread. YOLO_TTA_SCHEDULER_PUSH_DRAIN=0 restores polling."""
    return _env_flag('YOLO_TTA_SCHEDULER_PUSH_DRAIN', True)

def scheduler_push_drain_heartbeat_seconds() -> float:
    """Upper bound on one push-drain sleep (worker-liveness re-check cadence)."""
    return max(0.05, _env_float('YOLO_TTA_SCHEDULER_PUSH_DRAIN_HEARTBEAT', 1.0))

def fused_final_view_union_enabled() -> bool:
    """Return whether Cartesian and projected-layer restoration may share one output-z pass."""
    return _env_flag('YOLO_TTA_FUSED_FINAL_VIEW_UNION', True)

def fused_final_restore_geometry_groups_enabled() -> bool:
    """Union equal-geometry projected layers before one restore.

 Reduced Tilted component layers share an orthogonal backing geometry. Binary
 union commutes with the positive-support INTER_AREA/NEAREST restore, so can
 decode/OR every layer in one geometry group on its reduced grid and resize the
 group once. The escape hatch retains the per-layer restore for direct
 regression comparisons."""
    return _env_flag('YOLO_TTA_FUSED_FINAL_RESTORE_GEOMETRY_GROUPS', True)

def fused_final_gpu_enabled() -> bool:
    """Move grouped final restore/OR work onto idle inference GPUs by default.

 Admission remains VRAM-aware and every failure rewrites the complete result through the
 established CPU path. Set ``YOLO_TTA_FUSED_FINAL_GPU=0`` for a CPU-only comparison."""
    return _env_flag('YOLO_TTA_FUSED_FINAL_GPU', True)

def fused_final_gpu_batch_slices() -> int:
    return max(1, _env_int('YOLO_TTA_FUSED_FINAL_GPU_BATCH_SLICES', 8))

def fused_final_gpu_pipeline_slots() -> int:
    """Reusable pinned-host/device batches per GPU lane (2 overlaps decode with CUDA work)."""
    return max(1, min(3, _env_int('YOLO_TTA_FUSED_FINAL_GPU_PIPELINE_SLOTS', 2)))

def fused_final_native_sparse_cpu_enabled() -> bool:
    """Bypass GPU staging when every final contributor is already in output geometry.

    D1 source-space cvols require no resampling. Sending their host-assembled union through
    one lane thread per GPU only adds H2D/D2H traffic and caps sparse decode concurrency at
    the GPU count. The source-z-parallel CPU path is byte-identical and keeps the sparse
    crops in host memory. Set YOLO_TTA_FUSED_FINAL_NATIVE_SPARSE_CPU=0 to restore the
    v16.1.5 GPU-lane behavior for comparison.
    """
    return _env_flag('YOLO_TTA_FUSED_FINAL_NATIVE_SPARSE_CPU', True)

def fused_final_native_sparse_dense_pretouch_enabled() -> bool:
    """Materialize every dense destination page while the sparse final union is built.

    ``np.zeros`` leaves untouched anonymous pages mapped to the shared kernel zero page.
    A sparse bbox-only union otherwise defers most of the 16+ GiB page-fault/NUMA placement
    cost to the next dense CUDA consumer (notably keep_objects slice labeling), where
    pageable H2D staging turns it into long idle gaps. First-touching each owned output-z
    plane here preserves the same zero value, distributes placement across the union pool,
    and makes later dense scans DMA-ready. Set
    YOLO_TTA_FUSED_FINAL_NATIVE_SPARSE_DENSE_PRETOUCH=0 only for regression comparison.
    """
    return _env_flag('YOLO_TTA_FUSED_FINAL_NATIVE_SPARSE_DENSE_PRETOUCH', True)

def fused_final_native_sparse_cpu_workers(requested_workers: int, out_t: int) -> int:
    """Worker count for native source-space sparse final union."""
    default = max(
        1,
        min(
            int(max(1, out_t)),
            int(max(1, requested_workers)),
            int(max(1, _cpu_count())),
        ),
    )
    return max(
        1,
        min(
            int(max(1, out_t)),
            _env_int('YOLO_TTA_FUSED_FINAL_NATIVE_SPARSE_WORKERS', int(default)),
        ),
    )

def fused_final_native_sparse_band_slices(worker_count: int, out_t: int) -> int:
    """Bound the active source-z frontier while retaining several chunks per worker."""
    default = max(256, int(max(1, worker_count)) * 4)
    return max(
        1,
        min(
            int(max(1, out_t)),
            _env_int('YOLO_TTA_FUSED_FINAL_NATIVE_SPARSE_BAND_SLICES', int(default)),
        ),
    )

def fused_final_gpu_min_group_bytes() -> int:
    """Reduced geometry groups below this work size stay on the CPU."""
    return max(0, int(max(0.0, _env_float('YOLO_TTA_FUSED_FINAL_GPU_MIN_GROUP_MIB', 4.0)) * 1024 * 1024))

def fused_final_gpu_sparse_ratio() -> float:
    """Very sparse groups below this extent ratio may stay on the CPU."""
    return max(0.0, min(1.0, _env_float('YOLO_TTA_FUSED_FINAL_GPU_SPARSE_RATIO', 0.005)))

def _resize_union_plane_to_out_xy(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Restore one Cartesian binary plane to the requested output geometry."""
    plane_u8 = (np.asarray(plane, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    if int(plane_u8.shape[0]) == int(out_h) and int(plane_u8.shape[1]) == int(out_w):
        return plane_u8
    interp = (
        cv2.INTER_AREA
        if (int(plane_u8.shape[0]) >= int(out_h) and int(plane_u8.shape[1]) >= int(out_w))
        else cv2.INTER_NEAREST
    )
    scaled = cv2.resize(
        np.ascontiguousarray(plane_u8 * np.uint8(255)),
        (int(out_w), int(out_h)),
        interpolation=int(interp),
    )
    return (scaled > 0).astype(np.uint8, copy=False)

def collapse_tta_variant_volumes_to_physical_views(
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    views: Sequence[ViewInfo],
    *,
    workers: int = 1,
    retired_volume_ids: Optional[set[int]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Union completed angle variants only at physical-view finalization.

    Every runtime ``ViewInfo.name`` is angle-specific in v16.4.0. Cartesian working stacks
    must therefore recover their physical names before the terminal axis-aware assembler;
    otherwise a Sagittal/Coronal variant can be mistaken for an already-source-native volume.
    The first completed variant becomes the in-place physical accumulator. Later variants of
    the same physical view are ORed into it and retired immediately. Projected component refs
    are intentionally not materialized here; their independent angle layers remain direct
    contributors to the final source-space union.
    """
    view_by_runtime_name = {str(view.name): view for view in views}
    collapsed_by_model: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, runtime_volumes in view_volumes_by_model.items():
        physical_volumes: Dict[str, np.ndarray] = {}
        physical_owner_names: Dict[str, str] = {}
        for runtime_name, volume in list(runtime_volumes.items()):
            runtime_key = str(runtime_name)
            view = view_by_runtime_name.get(runtime_key)
            if view is None:
                if '__tta_' in runtime_key:
                    raise KeyError(
                        f'Cannot collapse unknown TTA runtime view {runtime_key!r}; '
                        'its physical geometry is unavailable.'
                    )
                physical_name = runtime_key
            else:
                physical_name = physical_view_name(view)

            existing = physical_volumes.get(str(physical_name))
            if existing is None:
                physical_volumes[str(physical_name)] = volume
                physical_owner_names[str(physical_name)] = runtime_key
                continue
            if existing is volume:
                continue
            existing_shape = tuple(int(v) for v in np.asarray(existing).shape)
            incoming_shape = tuple(int(v) for v in np.asarray(volume).shape)
            if existing_shape != incoming_shape:
                raise ValueError(
                    f'Cannot collapse TTA variants for {model_name}/{physical_name}: '
                    f'{physical_owner_names[physical_name]} has shape {existing_shape}, '
                    f'but {runtime_key} has shape {incoming_shape}.'
                )
            union_volume_into_volume(
                existing,
                volume,
                workers=int(workers),
                desc=(
                    f'Collapsing final TTA angle variant {model_name}/{runtime_key} '
                    f'into physical view {physical_name}'
                ),
            )
            close_memmap_array(volume)
            if retired_volume_ids is not None:
                retired_volume_ids.add(id(volume))

        runtime_volumes.clear()
        collapsed_by_model[str(model_name)] = physical_volumes

    return collapsed_by_model

def release_unretained_volume_maps(
    volume_maps: Sequence[Dict[str, Dict[str, np.ndarray]]],
    retained_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    *,
    already_retired_ids: Optional[set[int]] = None,
) -> None:
    """Retire auxiliary per-angle dense volumes that the collapsed map no longer owns."""
    retained_ids = {
        id(volume)
        for model_volumes in retained_volumes_by_model.values()
        for volume in model_volumes.values()
    }
    closed_ids: set[int] = set(already_retired_ids or ())
    for outer_map in volume_maps:
        for model_volumes in outer_map.values():
            for volume in model_volumes.values():
                volume_id = id(volume)
                if volume_id in retained_ids or volume_id in closed_ids:
                    continue
                close_memmap_array(volume)
                closed_ids.add(volume_id)
            model_volumes.clear()

def assemble_view_volumes_into_native_union(
    final_union_mm: np.ndarray,
    view_volume_mms: Dict[str, np.ndarray],
    T: int,
    H: int,
    W: int,
    *,
    out_shape_tyx: Optional[Tuple[int, int, int]] = None,
    workers: int = 1,
) -> None:
    """OR all per-view prediction volumes from the single active model into the final union.

 Already-backprojected Tilted/Radial volumes arrive at the union's own (out) geometry and are
 merged slice-wise. Transverse, Sagittal, and Coronal retain their native working-geometry view
 stacks; restores each of them working->source with ONE resample
 while merging (union-biased: a source slice ORs every working slice it covers and XY planes
 are resized with the same >0-threshold INTER_AREA/NEAREST semantics as the former tail
 restore). When ``out_shape_tyx`` is None or equals the working geometry, the historical
 zero-resample axis-permutation merges are used."""
    out_t, out_h, out_w = (int(T), int(H), int(W)) if out_shape_tyx is None else (
        int(out_shape_tyx[0]), int(out_shape_tyx[1]), int(out_shape_tyx[2]))
    working_equals_out = (out_t, out_h, out_w) == (int(T), int(H), int(W))

    def _resize_plane_to_out_xy(plane: np.ndarray) -> np.ndarray:
        return _resize_union_plane_to_out_xy(plane, int(out_h), int(out_w))

    def _source_plane_indices(plane_count: int, out_z: int) -> range:
        count = max(1, int(plane_count))
        if count >= int(out_t):
            start = int(math.floor(float(out_z) * float(count) / float(out_t)))
            stop = int(math.ceil(float(out_z + 1) * float(count) / float(out_t)))
            start = int(np.clip(start, 0, count - 1))
            stop = int(np.clip(max(start + 1, stop), 1, count))
            return range(start, stop)
        if int(out_t) <= 1 or count <= 1:
            return range(0, 1)
        src_idx = int(round(float(out_z) * float(count - 1) / float(out_t - 1)))
        src_idx = int(np.clip(src_idx, 0, count - 1))
        return range(src_idx, src_idx + 1)

    def _or_restore_working_planes_into_union(
        get_working_plane: Callable[[int], np.ndarray],
        view_label: str,
        plane_count: int = T,
    ) -> None:
        # One terminal resample per source plane: t-range OR + XY resize straight into union.
        def _merge_out_slice(out_z: int) -> None:
            acc: Optional[np.ndarray] = None
            for v_idx in _source_plane_indices(int(plane_count), int(out_z)):
                restored = _resize_plane_to_out_xy(get_working_plane(int(v_idx)))
                if acc is None:
                    acc = restored
                else:
                    np.bitwise_or(acc, restored, out=acc)
            if acc is not None:
                final_union_mm[int(out_z), :, :] |= acc

        parallel_for_indices(
            int(out_t),
            _merge_out_slice,
            max_workers=choose_slice_parallel_workers(int(workers), int(out_t)),
            desc=f"Assembling final union from {view_label} view volume (restored to source geometry)",
        )

    consumed: set[str] = set()

    if "transverse" in view_volume_mms:
        transverse = np.asarray(view_volume_mms["transverse"])
        if int(transverse.ndim) != 3:
            raise ValueError(f'transverse volume must be 3D, got {tuple(transverse.shape)}')
        consumed.add("transverse")

        if working_equals_out and tuple(int(v) for v in transverse.shape) == (int(T), int(H), int(W)):
            transverse_workers = choose_slice_parallel_workers(int(workers), int(T))

            def _merge_transverse(t: int) -> None:
                final_union_mm[int(t), :, :] |= transverse[int(t), :, :]

            parallel_for_indices(
                int(T),
                _merge_transverse,
                max_workers=transverse_workers,
                desc="Assembling final union from transverse view volume",
            )
        else:
            _or_restore_working_planes_into_union(
                lambda v: transverse[int(v), :, :], "transverse", int(transverse.shape[0]),
            )

    if "sagittal" in view_volume_mms:
        sagittal = np.asarray(view_volume_mms["sagittal"])
        if int(sagittal.ndim) != 3:
            raise ValueError(f'sagittal volume must be 3D, got {tuple(sagittal.shape)}')
        consumed.add("sagittal")

        if working_equals_out and tuple(int(v) for v in sagittal.shape) == (int(H), int(T), int(W)):
            # the per-y merge (final_union[:, y,:] |= sagittal[y]) swept the
            # ENTIRE destination volume once per y (T rows of W bytes at stride H*W) — full
            # cache lines, but H passes of page/TLB traffic. Merge K rows per block instead:
            # for each t the destination window [t, y0:y1,:] is one contiguous K*W run, so
            # every destination page is visited once per block (H/K fewer volume sweeps).
            blk_rows = int(coronal_block_cols())
            n_blocks = (int(H) + blk_rows - 1) // blk_rows

            def _merge_sagittal_block(block_idx: int) -> None:
                y0 = int(block_idx) * blk_rows
                y1 = min(int(H), y0 + blk_rows)
                sub = np.asarray(sagittal[y0:y1])
                for t in range(int(T)):
                    final_union_mm[int(t), y0:y1, :] |= sub[:, int(t), :]

            parallel_for_indices_chunked(
                int(n_blocks),
                _merge_sagittal_block,
                max_workers=choose_slice_parallel_workers(int(workers), int(n_blocks)),
                desc="Assembling final union from sagittal view volume",
                show_progress=False,
                target_chunks_per_worker=2,
            )
        else:
            # reduced stack: axis 1 is the inference-pitch t axis; axis 0 remains the
            # sagittal stack/y axis. The final resize restores both Y and X in one pass.
            _or_restore_working_planes_into_union(
                lambda v: sagittal[:, int(v), :], "sagittal", int(sagittal.shape[1]),
            )

    if "coronal" in view_volume_mms:
        coronal = np.asarray(view_volume_mms["coronal"])
        if int(coronal.ndim) != 3:
            raise ValueError(f'coronal volume must be 3D, got {tuple(coronal.shape)}')
        consumed.add("coronal")

        if working_equals_out and tuple(int(v) for v in coronal.shape) == (int(W), int(T), int(H)):
            # the per-x merge (final_union[:,:, x] |= coronal[x]) wrote one
            # byte per cache line across the whole (t, Y) extent — the same ~64x write
            # amplification fixed for the projection copy. Same cure: permute
            # K columns at a time through per-t (K, H) -> (H, K) tile transposes (tiles stay
            # cache-resident) and OR K contiguous bytes per destination row.
            blk_cols = int(coronal_block_cols())
            n_blocks = (int(W) + blk_cols - 1) // blk_cols

            def _merge_coronal_block(block_idx: int) -> None:
                x0 = int(block_idx) * blk_cols
                x1 = min(int(W), x0 + blk_cols)
                sub = np.asarray(coronal[x0:x1])
                for t in range(int(T)):
                    final_union_mm[int(t), :, x0:x1] |= sub[:, int(t), :].T

            parallel_for_indices_chunked(
                int(n_blocks),
                _merge_coronal_block,
                max_workers=choose_slice_parallel_workers(int(workers), int(n_blocks)),
                desc="Assembling final union from coronal view volume",
                show_progress=False,
                target_chunks_per_worker=2,
            )
        else:
            # reduced stack: axis 1 is inference-pitch t; transpose stack/x and plane/y.
            _or_restore_working_planes_into_union(
                lambda v: np.ascontiguousarray(np.asarray(coronal[:, int(v), :]).T),
                "coronal",
                int(coronal.shape[1]),
            )

    for view_name in sorted(view_volume_mms.keys()):
        if view_name in consumed:
            continue
        vol = np.asarray(view_volume_mms[view_name])
        if vol.shape != (out_t, out_h, out_w):
            raise ValueError(
                f"View volume '{view_name}' has shape {tuple(vol.shape)}; expected final output volume shape "
                f"{(out_t, out_h, out_w)} or a handled transverse/sagittal/coronal working stack."
            )
        native_workers = choose_slice_parallel_workers(int(workers), int(out_t))

        def _merge_native(t: int, *, _vol: np.ndarray = vol) -> None:
            final_union_mm[int(t), :, :] |= _vol[int(t), :, :]

        parallel_for_indices(
            int(out_t),
            _merge_native,
            max_workers=native_workers,
            desc=f"Assembling final union from {view_name} view volume",
        )

def assemble_view_volumes_and_projected_layers_fused(
    final_union_mm: np.ndarray,
    view_volume_mms: Dict[str, np.ndarray],
    projected_layer_refs: Sequence['NrrdLayerRef'],
    T: int,
    H: int,
    W: int,
    *,
    out_shape_tyx: Tuple[int, int, int],
    workers: int = 1,
) -> None:
    """Assemble every view and projected layer in one output-z traversal.
    
    Contributors with equal restore geometry are grouped before resizing and OR accumulation."""
    out_t, out_h, out_w = (int(v) for v in out_shape_tyx)
    if (out_t, out_h, out_w) == (int(T), int(H), int(W)):
        raise ValueError('G5 fused restore is scoped to working geometry != output geometry')
    if tuple(int(v) for v in np.asarray(final_union_mm).shape) != (out_t, out_h, out_w):
        raise ValueError(
            f'G5 final union shape {tuple(np.asarray(final_union_mm).shape)} != {(out_t, out_h, out_w)}'
        )

    transverse = np.asarray(view_volume_mms['transverse']) if 'transverse' in view_volume_mms else None
    sagittal = np.asarray(view_volume_mms['sagittal']) if 'sagittal' in view_volume_mms else None
    coronal = np.asarray(view_volume_mms['coronal']) if 'coronal' in view_volume_mms else None
    for _name, _vol in (('transverse', transverse), ('sagittal', sagittal), ('coronal', coronal)):
        if _vol is not None and int(_vol.ndim) != 3:
            raise ValueError(f'G5 {_name} volume must be 3D, got {tuple(_vol.shape)}')

    native_views: List[Tuple[str, np.ndarray]] = []
    for view_name in sorted(view_volume_mms):
        if view_name in {'transverse', 'sagittal', 'coronal'}:
            continue
        vol = np.asarray(view_volume_mms[view_name])
        if tuple(int(v) for v in vol.shape) != (out_t, out_h, out_w):
            raise ValueError(
                f"G5 native view '{view_name}' shape {tuple(vol.shape)} != {(out_t, out_h, out_w)}"
            )
        native_views.append((str(view_name), vol))

    opened: List[Tuple['NrrdLayerRef', object]] = []
    try:
        for ref in projected_layer_refs:
            # Do not pin every projected chunks.bin simultaneously: may open ~90
            # component stores, and the explicit NRRD RAM cache would erase the peak-RAM
            # win from deleting the per-view intermediates. The OS page cache still gives
            # reclaimable reads; a read-only mmap also avoids per-slice file open/seek churn.
            if _nrrd_layer_ref_is_raw_bbox_store(ref):
                src = RawBBoxMaskStore.open(
                    ref.path, cache_payload_in_ram=False, mmap_payload=True,
                )
                try:
                    payload_map = getattr(src, '_chunks_mmap', None)
                    advice = getattr(mmap, 'MADV_SEQUENTIAL', None)
                    if payload_map is not None and advice is not None:
                        payload_map.madvise(advice)
                except Exception:
                    pass
            else:
                src = _open_nrrd_layer_ref(ref)
                _madvise_array_mmap(src, 'MADV_SEQUENTIAL')
            opened.append((ref, src))

        # shape fully defines the restore transform here because every
        # projected layer is already in canonical orthogonal (t,Y,X) coordinates.
        # Grouping is intentionally limited to non-native layers. Native Radial stores
        # continue to contribute bbox crops directly to the output accumulator.
        group_restores = bool(fused_final_restore_geometry_groups_enabled())
        native_projected: List[Tuple['NrrdLayerRef', object]] = []
        restore_groups: 'OrderedDict[Tuple[int, int, int], List[Tuple[NrrdLayerRef, object]]]' = OrderedDict()
        if group_restores:
            for ref, src in opened:
                geometry = tuple(int(v) for v in _volume_shape_tuple(src))
                if geometry == (out_t, out_h, out_w):
                    native_projected.append((ref, src))
                else:
                    restore_groups.setdefault(geometry, []).append((ref, src))
            grouped_layers = sum(len(group) for group in restore_groups.values())
            group_text = ', '.join(
                f'{len(group)}x{tuple(int(v) for v in geometry)}'
                for geometry, group in restore_groups.items()
            ) or 'none'
            print(
                f'v13.3.17 (C1): G5 grouped {int(grouped_layers)} reduced projected '
                f'layer(s) into {len(restore_groups)} restore geometry group(s) '
                f'[{group_text}] ({int(grouped_layers)} -> {len(restore_groups)} '
                f'restores/output-z); {len(native_projected)} native layer(s) remain direct.'
            )

        scratch_tls = threading.local()

        def _scratch_map() -> Dict[object, np.ndarray]:
            buffers = getattr(scratch_tls, 'buffers', None)
            if buffers is None:
                buffers = {}
                scratch_tls.buffers = buffers
            return buffers

        def _or_native_source_slice(dst: np.ndarray, src: object, src_z: int) -> None:
            """OR one source-grid slice, preserving RawBBoxMaskStore crop sparsity."""
            if isinstance(src, RawBBoxMaskStore):
                rec = src.index[int(src_z)]
                if int(rec['kind']) != 1:
                    return
                decoded = src.decode_slice_crop(int(src_z), dtype=np.uint8)
                if decoded is None:  # defensive: index.kind was checked above
                    return
                y0, x0, y1, x1, crop = decoded
                dst[int(y0):int(y1), int(x0):int(x1)] |= crop
                return
            np.bitwise_or(dst, np.asarray(src[int(src_z)], dtype=np.uint8), out=dst)

        def _try_native_sparse_cpu_final_fusion() -> bool:
            """Directly OR output-native contributors with source-z CPU parallelism.

            This is the D1 fast-bundle terminal case: every cvol is already in final
            source geometry, so there is no restore kernel for a GPU to accelerate. One
            worker owns each z slice, explicitly first-touches its dense zero plane, then
            writes only the foreground bbox crops. The dense first-touch prevents the next
            CUDA consumer from inheriting millions of anonymous zero-page faults.
            """
            if not fused_final_native_sparse_cpu_enabled():
                return False
            if transverse is not None or sagittal is not None or coronal is not None:
                return False
            if not native_views and not opened:
                return False

            output_shape = (int(out_t), int(out_h), int(out_w))
            direct_projected: List[Tuple['NrrdLayerRef', object]] = []
            for ref, src in opened:
                if tuple(int(v) for v in _volume_shape_tuple(src)) != output_shape:
                    return False
                direct_projected.append((ref, src))

            raw_sources: List[Tuple['NrrdLayerRef', RawBBoxMaskStore]] = []
            generic_sources: List[Tuple['NrrdLayerRef', object]] = []
            active_source_ids_by_z: List[List[int]] = [
                [] for _ in range(int(out_t))
            ]
            union_slice_any = np.zeros((int(out_t),), dtype=bool)
            union_slice_bboxes = np.zeros((int(out_t), 4), dtype=np.int64)
            payload_bytes = 0
            nonempty_layer_slices = 0
            for ref, src in direct_projected:
                if not isinstance(src, RawBBoxMaskStore):
                    generic_sources.append((ref, src))
                    continue
                source_id = int(len(raw_sources))
                raw_sources.append((ref, src))
                kinds = np.asarray(src.index['kind'], dtype=np.uint8)
                invalid = np.flatnonzero((kinds != np.uint8(0)) & (kinds != np.uint8(1)))
                if int(invalid.size) > 0:
                    raise ValueError(
                        f'{src.root}: invalid raw-mask marker {int(kinds[int(invalid[0])])} '
                        f'at slice {int(invalid[0])}'
                    )
                active_zs = np.flatnonzero(kinds == np.uint8(1))
                nonempty_layer_slices += int(active_zs.size)
                payload_bytes += int(np.sum(
                    np.asarray(src.index['payload_size'], dtype=np.uint64),
                    dtype=np.uint64,
                ))
                for z_idx in active_zs:
                    z_i = int(z_idx)
                    active_source_ids_by_z[z_i].append(int(source_id))
                    rec = src.index[z_i]
                    y0 = int(rec['y0']); y1 = int(rec['y1'])
                    x0 = int(rec['x0']); x1 = int(rec['x1'])
                    if not bool(union_slice_any[z_i]):
                        union_slice_any[z_i] = True
                        union_slice_bboxes[z_i] = np.asarray(
                            (int(y0), int(y1), int(x0), int(x1)), dtype=np.int64,
                        )
                    else:
                        union_slice_bboxes[z_i, 0] = min(
                            int(union_slice_bboxes[z_i, 0]), int(y0),
                        )
                        union_slice_bboxes[z_i, 1] = max(
                            int(union_slice_bboxes[z_i, 1]), int(y1),
                        )
                        union_slice_bboxes[z_i, 2] = min(
                            int(union_slice_bboxes[z_i, 2]), int(x0),
                        )
                        union_slice_bboxes[z_i, 3] = max(
                            int(union_slice_bboxes[z_i, 3]), int(x1),
                        )

            active_source_ids = [tuple(ids) for ids in active_source_ids_by_z]
            metadata_requires_exact_scan = bool(native_views or generic_sources)
            worker_count = fused_final_native_sparse_cpu_workers(int(workers), int(out_t))
            band_slices = fused_final_native_sparse_band_slices(
                int(worker_count), int(out_t),
            )
            logical_bytes = int(out_t) * int(out_h) * int(out_w)
            dense_pretouch = bool(fused_final_native_sparse_dense_pretouch_enabled())
            print(
                'v16.1.7 native sparse final union selected: '
                f'{len(direct_projected)} source-space layer(s) '
                f'({len(raw_sources)} raw-bbox, {len(generic_sources)} generic), '
                f'{len(native_views)} dense native view(s), workers={int(worker_count)}, '
                f'z_band={int(band_slices)}, payload={int(payload_bytes) / GIB:.2f} GiB, '
                f'dense_pretouch={"on" if dense_pretouch else "off"}. '
                'The GPU lane path is bypassed because there is no restore/resample work.',
                flush=True,
            )
            telemetry = runtime_telemetry()
            telemetry.gauge('final_union.native_sparse_cpu.layers', int(len(direct_projected)))
            telemetry.gauge('final_union.native_sparse_cpu.raw_bbox_layers', int(len(raw_sources)))
            telemetry.gauge('final_union.native_sparse_cpu.payload_bytes', int(payload_bytes))
            telemetry.gauge('final_union.native_sparse_cpu.workers', int(worker_count))
            telemetry.gauge('final_union.native_sparse_cpu.dense_pretouch', bool(dense_pretouch))
            telemetry.gauge(
                'final_union.native_sparse_cpu.dense_pretouch_bytes',
                int(logical_bytes) if bool(dense_pretouch) else 0,
            )

            def _merge_source_z(z_idx: int) -> None:
                z_i = int(z_idx)
                dst = np.asarray(final_union_mm[z_i])
                if bool(dense_pretouch):
                    # The destination is newly allocated and this z plane has exactly one
                    # owner. Writing the logical zeros now commits/places every page in one
                    # parallel pass instead of making CUDA pageable-copy staging fault them
                    # serially during keep_objects or final output generation.
                    dst.fill(np.uint8(0))
                # Dense native contributors are uncommon in D1 mode, but retaining them
                # makes this specialization exact for mixed output-native callers too.
                for _view_name, volume in native_views:
                    np.bitwise_or(
                        dst, np.asarray(volume[z_i], dtype=np.uint8), out=dst,
                    )
                for source_id in active_source_ids[z_i]:
                    _ref, src = raw_sources[int(source_id)]
                    decoded = src.decode_slice_crop(z_i, dtype=np.uint8)
                    if decoded is None:
                        continue
                    y0, x0, y1, x1, crop = decoded
                    dst_window = dst[int(y0):int(y1), int(x0):int(x1)]
                    np.bitwise_or(dst_window, crop, out=dst_window)
                for _ref, src in generic_sources:
                    _or_native_source_slice(dst, src, z_i)
                if bool(metadata_requires_exact_scan):
                    # Dense/generic contributors do not expose trustworthy per-z bboxes.
                    # The destination is already hot in this worker, so capture its exact
                    # support now rather than forcing keep_objects to rescan the full volume.
                    x0, y0, bw, bh = (int(v) for v in cv2.boundingRect(dst))
                    if int(bw) <= 0 or int(bh) <= 0:
                        union_slice_any[z_i] = False
                        union_slice_bboxes[z_i] = np.int64(0)
                    else:
                        union_slice_any[z_i] = True
                        union_slice_bboxes[z_i] = np.asarray(
                            (int(y0), int(y0 + bh), int(x0), int(x0 + bw)),
                            dtype=np.int64,
                        )

            started = time.perf_counter()
            with telemetry.span(
                'final_union.native_sparse_cpu',
                layers=int(len(direct_projected)),
                payload_bytes=int(payload_bytes),
            ):
                with tqdm(
                    total=int(out_t),
                    desc='v16.1.7 native sparse final union',
                ) as pbar:
                    for band0 in range(0, int(out_t), int(band_slices)):
                        band1 = min(int(out_t), int(band0) + int(band_slices))
                        band_count = int(band1 - band0)

                        def _merge_band_z(local_z: int, _band0: int = int(band0)) -> None:
                            _merge_source_z(int(_band0) + int(local_z))

                        parallel_for_indices_chunked(
                            int(band_count),
                            _merge_band_z,
                            max_workers=min(int(worker_count), int(band_count)),
                            desc='v16.1.7 native sparse final union band',
                            show_progress=False,
                            target_chunks_per_worker=2,
                        )
                        pbar.update(int(band_count))
            elapsed = max(1e-9, time.perf_counter() - started)
            metadata = register_binary_volume_slice_metadata(
                final_union_mm,
                union_slice_any,
                union_slice_bboxes,
                source=(
                    'v16.1.7 native sparse final union exact destination scan'
                    if bool(metadata_requires_exact_scan)
                    else 'v16.1.7 native sparse cvol index union'
                ),
                exact=bool(metadata_requires_exact_scan),
            )
            nonempty_slices, bbox_pixels, bbox_fraction = binary_slice_bbox_coverage(
                output_shape,
                metadata.slice_any,
                metadata.slice_bboxes,
            )
            print(
                'v16.1.7 native sparse final union completed in '
                f'{elapsed:.3f}s; sparse payload={int(payload_bytes) / GIB:.2f} GiB, '
                f'logical destination={int(logical_bytes) / GIB:.2f} GiB, '
                f'payload throughput={int(payload_bytes) / GIB / elapsed:.2f} GiB/s, '
                f'nonempty layer-slices={int(nonempty_layer_slices)}, '
                f'union support={int(nonempty_slices)}/{int(out_t)} z slice(s), '
                f'bbox coverage={100.0 * float(bbox_fraction):.2f}% '
                f'({int(bbox_pixels) / GIB:.2f} Gpixel-equivalent), '
                f'dense_pretouch={"on" if dense_pretouch else "off"} '
                f'({int(logical_bytes) / GIB:.2f} GiB committed in the same z pass).',
                flush=True,
            )
            telemetry.gauge('final_union.native_sparse_cpu.seconds', float(elapsed))
            telemetry.gauge('final_union.native_sparse_cpu.support_nonempty_slices', int(nonempty_slices))
            telemetry.gauge('final_union.native_sparse_cpu.support_bbox_pixels', int(bbox_pixels))
            telemetry.gauge('final_union.native_sparse_cpu.support_bbox_fraction', float(bbox_fraction))
            return True

        def _try_gpu_final_fusion() -> bool:
            """GPU path with pinned double buffering and geometry streams.

 Independent output-z bands are assigned to every admitted CUDA device. Each lane
 decodes the next host batch while the previous batch is asynchronously copied,
 restored, OR-reduced, and copied back. Reduced geometry groups receive independent
 CUDA streams; a merge stream waits on their events before the single final D2H.
 Source-native sparse stores and very small reduced groups remain on the CPU and are
 folded into one pinned native contributor, where H2D setup would dominate."""
            if not bool(group_restores) or not fused_final_gpu_enabled():
                return False
            try:
                import torch  # type: ignore
                import torch.nn.functional as F  # type: ignore
                if not bool(torch.cuda.is_available()):
                    return False
            except Exception:
                return False

            configured = gpu_slice_labeling_configured_devices()
            devices: List[int] = []
            raw_devices = os.environ.get('YOLO_TTA_FUSED_FINAL_GPU_DEVICES', '').strip()
            if raw_devices:
                try:
                    devices = [
                        int(tok) for tok in re.split(r'[,\s]+', raw_devices) if str(tok).strip()
                    ]
                except Exception as exc:
                    print(
                        f'Warning: invalid YOLO_TTA_FUSED_FINAL_GPU_DEVICES ({exc}); '
                        'using configured devices.'
                    )
                    devices = []
            if not devices and configured:
                devices = [int(v) for v in configured]
            if not devices:
                devices = list(range(int(torch.cuda.device_count())))
            max_devices = max(1, _env_int('YOLO_TTA_FUSED_FINAL_GPU_MAX_DEVICES', 4))
            devices = list(dict.fromkeys(devices))[:min(int(max_devices), int(out_t))]
            if not devices:
                return False

            min_group_bytes = int(fused_final_gpu_min_group_bytes())
            sparse_ratio_limit = float(fused_final_gpu_sparse_ratio())

            def _group_extent_ratio(
                geometry: Tuple[int, int, int],
                group: Sequence[Tuple['NrrdLayerRef', object]],
            ) -> float:
                in_t, in_h, in_w = (int(v) for v in geometry)
                full_per_layer = max(1, int(in_t) * int(in_h) * int(in_w))
                active = 0
                for ref, _src in group:
                    extent = getattr(ref, 'segment_extent_ijk', None)
                    if extent is None or len(extent) != 6:
                        active += int(full_per_layer)
                        continue
                    try:
                        x0, x1, y0, y1, z0, z1 = (int(v) for v in extent)
                        ex = max(0, min(int(in_w) - 1, x1) - max(0, x0) + 1)
                        ey = max(0, min(int(in_h) - 1, y1) - max(0, y0) + 1)
                        ez = max(0, min(int(in_t) - 1, z1) - max(0, z0) + 1)
                        active += int(ex) * int(ey) * int(ez)
                    except Exception:
                        active += int(full_per_layer)
                denom = max(1, int(full_per_layer) * max(1, len(group)))
                return max(0.0, min(1.0, float(active) / float(denom)))

            gpu_restore_groups: 'OrderedDict[Tuple[int, int, int], List[Tuple[NrrdLayerRef, object]]]' = OrderedDict()
            cpu_restore_groups: 'OrderedDict[Tuple[int, int, int], List[Tuple[NrrdLayerRef, object]]]' = OrderedDict()
            for geometry, group in restore_groups.items():
                _in_t, in_h, in_w = (int(v) for v in geometry)
                plane_work = int(in_h) * int(in_w) * max(1, len(group))
                extent_ratio = _group_extent_ratio(geometry, group)
                tiny = int(plane_work) < int(min_group_bytes)
                sparse_and_small = bool(
                    extent_ratio < float(sparse_ratio_limit)
                    and int(plane_work) < max(int(min_group_bytes), 1) * 4
                )
                if tiny or sparse_and_small:
                    cpu_restore_groups[geometry] = list(group)
                else:
                    gpu_restore_groups[geometry] = list(group)

            batch_slices = int(fused_final_gpu_batch_slices())
            pipeline_slots = int(fused_final_gpu_pipeline_slots())
            contributor_input_planes = 0
            contributor_count = 0
            if transverse is not None:
                contributor_input_planes += int(transverse.shape[1]) * int(transverse.shape[2])
                contributor_count += 1
            if sagittal is not None:
                contributor_input_planes += int(sagittal.shape[0]) * int(sagittal.shape[2])
                contributor_count += 1
            if coronal is not None:
                contributor_input_planes += int(coronal.shape[2]) * int(coronal.shape[0])
                contributor_count += 1
            needs_native_contributor = bool(native_views or native_projected or cpu_restore_groups)
            if needs_native_contributor:
                contributor_input_planes += int(out_h) * int(out_w)
                contributor_count += 1
            for geometry in gpu_restore_groups:
                contributor_input_planes += int(geometry[1]) * int(geometry[2])
                contributor_count += 1
            if contributor_count <= 0:
                return False

            reserve = int(max(0.0, _env_float(
                'YOLO_TTA_FUSED_FINAL_GPU_RESERVE_GIB', 2.0,
            )) * GIB)
            out_plane = int(out_h) * int(out_w)
            # Per slot: uint8 device inputs, float interpolation workspace/restored outputs,
            # one uint8 accumulator/output, plus allocator slack. Host-pinned memory is not
            # charged here; the anonymous/cgroup allocator remains the authority for it.
            estimate_per_slot = int(batch_slices) * (
                int(contributor_input_planes)
                + max(1, int(contributor_count)) * 6 * int(out_plane)
                + 3 * int(out_plane)
            )
            estimate = int(pipeline_slots) * int(estimate_per_slot)
            admitted: List[int] = []
            for device_idx in devices:
                try:
                    free_bytes, _total_bytes = torch.cuda.mem_get_info(int(device_idx))
                    if int(free_bytes) >= int(estimate) + int(reserve):
                        admitted.append(int(device_idx))
                except Exception:
                    continue
            if not admitted:
                print(
                    'v16.0.2 GPU final fusion skipped: no selected device satisfied the '
                    f'{estimate / GIB:.2f} GiB lane estimate + {reserve / GIB:.2f} GiB reserve.'
                )
                return False

            print(
                f'v16.0.2 GPU final fusion admitted {len(admitted)} device(s) '
                f'{[f"cuda:{d}" for d in admitted]}; batch={batch_slices}, '
                f'pipeline_slots={pipeline_slots}, GPU_restore_groups={len(gpu_restore_groups)}, '
                f'CPU_tiny_groups={len(cpu_restore_groups)}, lane_estimate={estimate / GIB:.2f} GiB.'
            )

            def _indices(count: int, out_z: int) -> range:
                count_i = max(1, int(count))
                if count_i >= int(out_t):
                    lo = int(math.floor(float(out_z) * float(count_i) / float(out_t)))
                    hi = int(math.ceil(float(out_z + 1) * float(count_i) / float(out_t)))
                    lo = int(np.clip(lo, 0, count_i - 1))
                    hi = int(np.clip(max(lo + 1, hi), 1, count_i))
                    return range(lo, hi)
                if int(out_t) <= 1 or count_i <= 1:
                    return range(0, 1)
                src_i = int(round(float(out_z) * float(count_i - 1) / float(out_t - 1)))
                src_i = int(np.clip(src_i, 0, count_i - 1))
                return range(src_i, src_i + 1)

            def _lane(device_idx: int, z0_lane: int, z1_lane: int) -> None:
                device = torch.device(f'cuda:{int(device_idx)}')
                with torch.cuda.device(device):
                    host_shapes: 'OrderedDict[object, Tuple[int, int, int]]' = OrderedDict()
                    if transverse is not None:
                        host_shapes['transverse'] = (
                            int(batch_slices), int(transverse.shape[1]), int(transverse.shape[2]),
                        )
                    if sagittal is not None:
                        host_shapes['sagittal'] = (
                            int(batch_slices), int(sagittal.shape[0]), int(sagittal.shape[2]),
                        )
                    if coronal is not None:
                        host_shapes['coronal'] = (
                            int(batch_slices), int(coronal.shape[2]), int(coronal.shape[0]),
                        )
                    if needs_native_contributor:
                        host_shapes['native'] = (
                            int(batch_slices), int(out_h), int(out_w),
                        )
                    for geometry in gpu_restore_groups:
                        host_shapes[('restore', geometry)] = (
                            int(batch_slices), int(geometry[1]), int(geometry[2]),
                        )

                    pin_warning_printed = False

                    def _alloc_host(shape: Tuple[int, int, int]) -> object:
                        nonlocal pin_warning_printed
                        try:
                            return torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                        except Exception as exc:
                            if not pin_warning_printed:
                                pin_warning_printed = True
                                print(
                                    f'Warning: cuda:{int(device_idx)} pinned final-fusion staging '
                                    f'unavailable ({exc}); using pageable fallback.'
                                )
                            return torch.empty(shape, dtype=torch.uint8)

                    streams = {
                        key: torch.cuda.Stream(device=device)
                        for key in host_shapes
                    }
                    merge_stream = torch.cuda.Stream(device=device)
                    slots: List[Dict[str, object]] = []
                    for _slot_idx in range(int(pipeline_slots)):
                        host_tensors = {key: _alloc_host(shape) for key, shape in host_shapes.items()}
                        host_arrays = {key: tensor.numpy() for key, tensor in host_tensors.items()}
                        device_tensors = {
                            key: torch.empty(shape, dtype=torch.uint8, device=device)
                            for key, shape in host_shapes.items()
                        }
                        output_host = _alloc_host((int(batch_slices), int(out_h), int(out_w)))
                        slots.append({
                            'host_tensors': host_tensors,
                            'host_arrays': host_arrays,
                            'device_tensors': device_tensors,
                            'output_host': output_host,
                            'done_event': None,
                            'range': None,
                            'refs': [],
                        })

                    cpu_reduced = {
                        geometry: np.zeros((int(geometry[1]), int(geometry[2])), dtype=np.uint8)
                        for geometry in cpu_restore_groups
                    }

                    def _finish_slot(slot: Dict[str, object]) -> None:
                        done_event = slot.get('done_event')
                        batch_range = slot.get('range')
                        if done_event is None or batch_range is None:
                            return
                        done_event.synchronize()
                        batch0_done, batch1_done = (int(v) for v in batch_range)
                        count_done = int(batch1_done - batch0_done)
                        output_host = slot['output_host']
                        np.copyto(
                            np.asarray(final_union_mm[int(batch0_done):int(batch1_done)]),
                            output_host[:count_done].numpy(),
                        )
                        refs = slot.get('refs')
                        if isinstance(refs, list):
                            refs.clear()
                        slot['done_event'] = None
                        slot['range'] = None

                    def _fill_slot(
                        slot: Dict[str, object],
                        batch0: int,
                        batch1: int,
                    ) -> int:
                        count = int(batch1 - batch0)
                        host_arrays = slot['host_arrays']
                        for arr in host_arrays.values():
                            arr[:count].fill(np.uint8(0))

                        if transverse is not None:
                            arr = host_arrays['transverse']
                            for local_z, out_z in enumerate(range(int(batch0), int(batch1))):
                                for src_z in _indices(int(transverse.shape[0]), int(out_z)):
                                    arr[local_z] |= np.asarray(transverse[int(src_z)], dtype=np.uint8)
                        if sagittal is not None:
                            arr = host_arrays['sagittal']
                            for local_z, out_z in enumerate(range(int(batch0), int(batch1))):
                                for src_z in _indices(int(sagittal.shape[1]), int(out_z)):
                                    arr[local_z] |= np.asarray(
                                        sagittal[:, int(src_z), :], dtype=np.uint8,
                                    )
                        if coronal is not None:
                            arr = host_arrays['coronal']
                            for local_z, out_z in enumerate(range(int(batch0), int(batch1))):
                                for src_z in _indices(int(coronal.shape[1]), int(out_z)):
                                    arr[local_z] |= np.ascontiguousarray(
                                        coronal[:, int(src_z), :].T, dtype=np.uint8,
                                    )

                        if needs_native_contributor:
                            native_arr = host_arrays['native']
                            for local_z, out_z in enumerate(range(int(batch0), int(batch1))):
                                dst = native_arr[local_z]
                                for _view_name, vol in native_views:
                                    dst |= np.asarray(vol[int(out_z)], dtype=np.uint8)
                                for _ref, src in native_projected:
                                    _or_native_source_slice(dst, src, int(out_z))
                                for geometry, group in cpu_restore_groups.items():
                                    in_t, _in_h, _in_w = (int(v) for v in geometry)
                                    reduced = cpu_reduced[geometry]
                                    reduced.fill(np.uint8(0))
                                    for src_z in _restore_source_indices_for_output_z(
                                        int(in_t), int(out_t), int(out_z),
                                    ):
                                        for _ref, src in group:
                                            _or_native_source_slice(reduced, src, int(src_z))
                                    dst |= _resize_union_plane_to_out_xy(
                                        reduced, int(out_h), int(out_w),
                                    )

                        for geometry, group in gpu_restore_groups.items():
                            arr = host_arrays[('restore', geometry)]
                            in_t = int(geometry[0])
                            for local_z, out_z in enumerate(range(int(batch0), int(batch1))):
                                dst = arr[local_z]
                                for src_z in _restore_source_indices_for_output_z(
                                    int(in_t), int(out_t), int(out_z),
                                ):
                                    for _ref, src in group:
                                        _or_native_source_slice(dst, src, int(src_z))
                        return int(count)

                    def _submit_slot(
                        slot: Dict[str, object],
                        batch0: int,
                        batch1: int,
                        count: int,
                    ) -> None:
                        host_tensors = slot['host_tensors']
                        device_tensors = slot['device_tensors']
                        contributor_outputs: List[Tuple[object, object]] = []
                        refs: List[object] = []
                        for key, host_tensor in host_tensors.items():
                            stream = streams[key]
                            dev_tensor = device_tensors[key]
                            with torch.cuda.stream(stream):
                                src_dev = dev_tensor[:count]
                                src_host = host_tensor[:count]
                                src_dev.copy_(
                                    src_host,
                                    non_blocking=bool(getattr(src_host, 'is_pinned', lambda: False)()),
                                )
                                in_h = int(src_dev.shape[1])
                                in_w = int(src_dev.shape[2])
                                if (in_h, in_w) == (int(out_h), int(out_w)):
                                    restored = src_dev
                                else:
                                    src_f = src_dev.to(torch.float32).unsqueeze(1)
                                    if in_h >= int(out_h) and in_w >= int(out_w):
                                        scaled = F.interpolate(
                                            src_f, size=(int(out_h), int(out_w)), mode='area',
                                        )
                                        restored = (
                                            scaled.squeeze(1) >= (1.0 / 510.0)
                                        ).to(torch.uint8)
                                    else:
                                        scaled = F.interpolate(
                                            src_f, size=(int(out_h), int(out_w)), mode='nearest',
                                        )
                                        restored = (scaled.squeeze(1) > 0).to(torch.uint8)
                                    refs.extend([src_f, scaled])
                                event = torch.cuda.Event(blocking=False, enable_timing=False)
                                event.record(stream)
                                contributor_outputs.append((restored, event))
                                refs.append(restored)

                        done_event = torch.cuda.Event(blocking=False, enable_timing=False)
                        output_host = slot['output_host']
                        with torch.cuda.stream(merge_stream):
                            acc = torch.zeros(
                                (int(count), int(out_h), int(out_w)),
                                dtype=torch.uint8,
                                device=device,
                            )
                            for restored, event in contributor_outputs:
                                merge_stream.wait_event(event)
                                torch.bitwise_or(acc, restored, out=acc)
                            output_view = output_host[:count]
                            output_view.copy_(
                                acc,
                                non_blocking=bool(
                                    getattr(output_view, 'is_pinned', lambda: False)()
                                ),
                            )
                            done_event.record(merge_stream)
                            refs.extend([acc, contributor_outputs])
                        slot['refs'] = refs
                        slot['done_event'] = done_event
                        slot['range'] = (int(batch0), int(batch1))

                    batch_index = 0
                    for batch0 in range(int(z0_lane), int(z1_lane), int(batch_slices)):
                        batch1 = min(int(z1_lane), int(batch0) + int(batch_slices))
                        slot = slots[int(batch_index) % len(slots)]
                        _finish_slot(slot)
                        count = _fill_slot(slot, int(batch0), int(batch1))
                        _submit_slot(slot, int(batch0), int(batch1), int(count))
                        batch_index += 1
                    for slot in slots:
                        _finish_slot(slot)
                    merge_stream.synchronize()
                    for stream in streams.values():
                        stream.synchronize()

            bands = _ffv1_contiguous_segments(int(out_t), len(admitted))
            lane_pool: Optional[ThreadPoolExecutor] = None
            lane_futures: List[Future] = []
            first_error: Optional[Exception] = None
            try:
                try:
                    lane_pool = ThreadPoolExecutor(
                        max_workers=len(admitted), thread_name_prefix='g5-gpu-fusion',
                    )
                    for dev, band in zip(admitted, bands):
                        lane_futures.append(lane_pool.submit(
                            _lane, int(dev), int(band[0]), int(band[1]),
                        ))
                except Exception as exc:
                    first_error = exc
                for lane_future in lane_futures:
                    try:
                        lane_future.result()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
            finally:
                try:
                    if lane_pool is not None:
                        lane_pool.shutdown(wait=True)
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                lane_futures.clear()
                try:
                    del lane_future
                except UnboundLocalError:
                    pass
                gc.collect()
                for device_idx in admitted:
                    try:
                        with torch.cuda.device(int(device_idx)):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            if first_error is not None:
                error_text = f'{type(first_error).__name__}: {first_error}'
                print(
                    f'Warning: v16.0.2 GPU final fusion failed ({error_text}); '
                    'rewriting the complete result with the CPU G5 path.'
                )
                return False
            print(
                f'v16.0.2: final grouped restore/OR completed on {len(admitted)} GPU lane(s) '
                f'with pinned {pipeline_slots}-slot streaming.'
            )
            return True

        if _try_native_sparse_cpu_final_fusion():
            flush_array(final_union_mm)
            return

        if _try_gpu_final_fusion():
            flush_array(final_union_mm)
            return

        def _merge_out_slice(out_z: int) -> None:
            buffers = _scratch_map()
            acc = buffers.get(-1)
            if acc is None:
                acc = np.zeros((out_h, out_w), dtype=np.uint8)
                buffers[-1] = acc
            else:
                acc.fill(np.uint8(0))

            def _indices(count: int) -> range:
                count_i = max(1, int(count))
                if count_i >= int(out_t):
                    lo = int(math.floor(float(out_z) * float(count_i) / float(out_t)))
                    hi = int(math.ceil(float(out_z + 1) * float(count_i) / float(out_t)))
                    lo = int(np.clip(lo, 0, count_i - 1))
                    hi = int(np.clip(max(lo + 1, hi), 1, count_i))
                    return range(lo, hi)
                if int(out_t) <= 1 or count_i <= 1:
                    return range(0, 1)
                src_i = int(round(float(out_z) * float(count_i - 1) / float(out_t - 1)))
                src_i = int(np.clip(src_i, 0, count_i - 1))
                return range(src_i, src_i + 1)

            if transverse is not None:
                for v_idx in _indices(int(transverse.shape[0])):
                    np.bitwise_or(
                        acc,
                        _resize_union_plane_to_out_xy(transverse[int(v_idx)], out_h, out_w),
                        out=acc,
                    )
            if sagittal is not None:
                for v_idx in _indices(int(sagittal.shape[1])):
                    np.bitwise_or(
                        acc,
                        _resize_union_plane_to_out_xy(sagittal[:, int(v_idx), :], out_h, out_w),
                        out=acc,
                    )
            if coronal is not None:
                for v_idx in _indices(int(coronal.shape[1])):
                    np.bitwise_or(
                        acc,
                        _resize_union_plane_to_out_xy(
                            np.ascontiguousarray(coronal[:, int(v_idx), :].T), out_h, out_w,
                        ),
                        out=acc,
                    )

            for _view_name, vol in native_views:
                np.bitwise_or(acc, np.asarray(vol[int(out_z)], dtype=np.uint8), out=acc)

            if group_restores:
                # Native projected layers (the two Radial refs in the reference run)
                # retain the exact -independent sparse direct path.
                for _ref, src in native_projected:
                    _or_native_source_slice(acc, src, int(out_z))

                # One thread-private reduced plane is reused by groups with equal XY.
                # Union all t-coverage slices and component layers before the single
                # positive-support resize. For binary masks this is exactly equivalent
                # to OR(resize(layer_i)) while eliminating repeated cv2 calls/allocations.
                for geometry, group in restore_groups.items():
                    in_t, in_h, in_w = (int(v) for v in geometry)
                    reduced_key = ('c1_restore', int(in_h), int(in_w))
                    reduced = buffers.get(reduced_key)
                    if reduced is None:
                        reduced = np.zeros((int(in_h), int(in_w)), dtype=np.uint8)
                        buffers[reduced_key] = reduced
                    else:
                        reduced.fill(np.uint8(0))
                    source_zs = _restore_source_indices_for_output_z(
                        int(in_t), int(out_t), int(out_z),
                    )
                    # Keep adjacent source-z reads together to preserve sequential
                    # mmap/readahead locality during the group-wide restore.
                    for _ref, src in group:
                        for src_z in source_zs:
                            _or_native_source_slice(reduced, src, int(src_z))
                    np.bitwise_or(
                        acc,
                        _resize_union_plane_to_out_xy(reduced, int(out_h), int(out_w)),
                        out=acc,
                    )
            else:
                # Exact fallback: restore every reduced component layer
                # independently. Useful for byte-equivalence/performance A/B checks.
                for _ref, src in opened:
                    ref_native = tuple(int(v) for v in _ref.shape) == (out_t, out_h, out_w)
                    if ref_native:
                        _or_native_source_slice(acc, src, int(out_z))
                    else:
                        np.bitwise_or(
                            acc,
                            _read_layer_slice_in_output_shape(
                                src, (int(out_t), int(out_h), int(out_w)), int(out_z),
                            ),
                            out=acc,
                        )

            # final_union is freshly allocated, but assignment is stronger than RMW and
            # guarantees exactly one full-plane destination write for this z.
            final_union_mm[int(out_z), :, :] = acc

        # submitting all z chunks to 320 workers made the first
        # frontier touch roughly z=0..1280 across every projected store simultaneously.
        # The log's 40 s cold start and second 12 s stall at z=1280 line up exactly with
        # that frontier. Process bounded sequential z bands so mmap readahead/page-table
        # locality can work, while retaining enough slice parallelism for cv2 restores.
        g5_workers = choose_slice_parallel_workers(
            max(1, _env_int('YOLO_TTA_FUSED_FINAL_VIEW_UNION_WORKERS', min(64, int(workers)))),
            out_t,
        )
        g5_band = max(
            int(g5_workers),
            _env_int('YOLO_TTA_FUSED_FINAL_VIEW_UNION_BAND_SLICES', 256),
        )
        print(
            f'v13.3.16 (G9): G5 locality scheduler uses {int(g5_workers)} worker(s) '
            f'and sequential {int(g5_band)}-slice z bands.'
        )
        with tqdm(total=int(out_t), desc='v13.3.16 G9 fused final view/layer union') as pbar:
            for band0 in range(0, int(out_t), int(g5_band)):
                band1 = min(int(out_t), int(band0) + int(g5_band))
                band_count = int(band1 - band0)

                def _merge_band_slice(local_z: int, _band0: int = int(band0)) -> None:
                    _merge_out_slice(int(_band0) + int(local_z))

                parallel_for_indices_chunked(
                    band_count,
                    _merge_band_slice,
                    max_workers=min(int(g5_workers), int(band_count)),
                    desc='v13.3.16 G9 fused final view/layer union band',
                    show_progress=False,
                    target_chunks_per_worker=4,
                )
                pbar.update(int(band_count))
        flush_array(final_union_mm)
    finally:
        for _ref, src in opened:
            try:
                _close_nrrd_layer_source(src)
            finally:
                _drop_nrrd_raw_store_chunks_ram_cache(src)

def assemble_current_view_union_volume(
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    T: int,
    H: int,
    W: int,
    out_path: Path,
    *,
    out_shape_tyx: Optional[Tuple[int, int, int]] = None,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    projected_layer_refs: Optional[Sequence['NrrdLayerRef']] = None,
) -> np.ndarray:
    """Assemble the single logical GPU/CPU model pair's current view union.

    GPU and CPU artifacts share one logical segmentation namespace in v17; the destination
    uses source geometry when requested."""
    if len(view_volumes_by_model) != 1:
        raise ValueError(
            f'GPT-5.6-Sol-Ultra v{SCRIPT_VERSION} expected one logical GPU/CPU model pair; '
            f'found {len(view_volumes_by_model)} result namespaces'
        )

    union_shape = (int(T), int(H), int(W)) if out_shape_tyx is None else tuple(int(v) for v in out_shape_tyx)
    model_name = next(iter(view_volumes_by_model.keys()))
    final_union_mm = allocate_workspace_array(
        shape=union_shape,
        dtype=np.uint8,
        path=out_path,
        desc='Final single-model view-union volume',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    print(f"\n=== Assembling final view union for model: {model_name} ===")
    direct_refs = list(projected_layer_refs or [])
    working_equals_out = tuple(int(v) for v in union_shape) == (int(T), int(H), int(W))
    if direct_refs and fused_final_view_union_enabled() and not working_equals_out:
        print(
            f'v13.3.9 (G5): fusing Cartesian restore, native fallback views, and '
            f'{len(direct_refs)} projected component layer(s) into one output-z pass '
            '(YOLO_TTA_FUSED_FINAL_VIEW_UNION=0 restores per-view assembly).'
        )
        assemble_view_volumes_and_projected_layers_fused(
            final_union_mm,
            view_volumes_by_model[model_name],
            direct_refs,
            int(T), int(H), int(W),
            out_shape_tyx=tuple(int(v) for v in union_shape),
            workers=int(workers),
        )
    else:
        assemble_view_volumes_into_native_union(
            final_union_mm=final_union_mm,
            view_volume_mms=view_volumes_by_model[model_name],
            T=T,
            H=H,
            W=W,
            out_shape_tyx=out_shape_tyx,
            workers=int(workers),
        )
        # Defensive mixed-mode support. The normal tail only supplies direct refs when
        # is eligible; if a caller supplies them under equal geometry, preserve the
        # Cartesian path and merge refs without omitting any contribution.
        for ref in direct_refs:
            _union_projected_layer_ref_into_volume(
                ref, final_union_mm, workers=int(workers), desc=f'Fallback final union: OR {ref.key}',
            )
    flush_array(final_union_mm)
    return final_union_mm

def union_volume_into_volume(
    dst_mm: np.ndarray,
    src_mm: np.ndarray,
    *,
    workers: int = 1,
    desc: str = 'Union volumes',
    slice_locks: Optional[Sequence[threading.Lock]] = None,
    count_voxels: bool = False,
) -> int:
    """OR one volume into another, optionally restoring geometry and using per-slice locks."""
    num_slices = int(dst_mm.shape[0]) if int(dst_mm.ndim) > 0 else 0
    lock_count = int(len(slice_locks)) if slice_locks else 0
    counts = np.zeros((num_slices,), dtype=np.int64) if bool(count_voxels) else None

    def _merge_slice(idx: int) -> None:
        src_slice = np.asarray(src_mm[int(idx)], dtype=np.uint8)
        if counts is not None:
            counts[int(idx)] = np.int64(int(np.count_nonzero(src_slice)))
        if lock_count > 0:
            with slice_locks[int(idx) % lock_count]:
                dst_mm[int(idx), :, :] |= src_slice
        else:
            dst_mm[int(idx), :, :] |= src_slice

    parallel_for_indices(
        num_slices,
        _merge_slice,
        max_workers=choose_slice_parallel_workers(int(workers), num_slices),
        desc=desc,
        show_progress=False,
    )
    flush_array(dst_mm)
    return int(np.sum(counts, dtype=np.int64)) if counts is not None else 0

@runtime_telemetry_phase('post.keep_objects')
def apply_keep_largest_objects_inplace(
    mask_mm: np.ndarray,
    keep_objects: int,
    temp_dir: Path,
    *,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
) -> Dict[str, int | float]:
    """Keep only the largest N connected foreground components in the final 3D volume."""
    keep_n = int(keep_objects)
    if keep_n <= 0:
        return {'enabled': 0, 'num_objects': 0, 'kept_objects': 0, 'removed_objects': 0, 'removed_voxels': 0}

    keep_started = time.perf_counter()
    runtime_telemetry().gauge('pipeline.phase', 'keep_objects_3d_connected_components')
    shape_tyx = tuple(int(v) for v in np.asarray(mask_mm).shape)
    print(
        f'keep_objects: resolving 3D connected components for shape={shape_tyx}, '
        f'keep={int(keep_n)}, worker_budget={int(workers)}, '
        f'topology_slab_slices={int(topology_slab_slices())}.'
    )
    work_dir = temp_dir / 'keep_objects'
    work_dir.mkdir(parents=True, exist_ok=True)
    # label with LOCAL per-slice ids only (no compact relabel write pass) and
    # harvest per-component areas during 2D labeling — the old flow ran a full-volume compact
    # relabel, then a full-volume GIL-held bincount, then rewrote every slice. The keep decision
    # is applied through per-slice local->keep LUTs restricted to each slice's foreground bbox,
    # and slices whose components are all kept are never touched.
    comp_stats: Dict[str, object] = {}
    metadata_started = time.perf_counter()
    support_metadata = binary_volume_slice_metadata(mask_mm)
    if support_metadata is None:
        support_metadata = scan_binary_volume_slice_metadata(
            mask_mm,
            workers=min(max(1, int(workers)), max(1, int(_cpu_count()))),
            source='keep_objects exact fallback scan',
        )
    metadata_seconds = float(time.perf_counter() - metadata_started)
    support_nonempty, support_bbox_pixels, support_bbox_fraction = binary_slice_bbox_coverage(
        shape_tyx,
        support_metadata.slice_any,
        support_metadata.slice_bboxes,
    )
    print(
        'keep_objects slice metadata: '
        f'source={support_metadata.source}, exact={bool(support_metadata.exact)}, '
        f'nonempty_z={int(support_nonempty)}/{int(shape_tyx[0])}, '
        f'bbox_coverage={100.0 * float(support_bbox_fraction):.2f}% '
        f'({int(support_bbox_pixels) / GIB:.2f} GiPixels), '
        f'prepare={float(metadata_seconds):.3f}s.',
        flush=True,
    )
    labels_mm, num_objects, label_paths = label_foreground_volume_streaming(
        mask_mm,
        work_dir / 'final_keep_objects',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        workers=int(workers),
        compact_relabel=False,
        component_stats_out=comp_stats,
        known_slice_any=support_metadata.slice_any,
        known_slice_bboxes=support_metadata.slice_bboxes,
        # also covers the common postprocessing keep_objects:1 tail: the same local-label
        # arena and direct packed-LUT apply avoid rebuilding a second dense host cube.
        sparse_local_labels=True,
        prefer_crop_bounded_cpu_labeling=True,
    )
    # The keep pass may mutate foreground below; do not let any later consumer reuse the
    # pre-keep support description. This is harmless when no removal is required.
    discard_binary_volume_slice_metadata(mask_mm)

    if int(num_objects) <= keep_n:
        total_seconds = float(time.perf_counter() - keep_started)
        topology_times = dict(comp_stats.get('topology_phase_seconds', {}))
        print(
            f'v16.0.2 keep_objects phases: no removal required; '
            f'topology={float(topology_times.get("topology_total", 0.0)):.3f}s, '
            f'total={total_seconds:.3f}s.'
        )
        close_memmap_array(labels_mm)
        if not bool(keep_temp):
            for lp in label_paths:
                try:
                    lp.unlink(missing_ok=True)
                except Exception:
                    pass
        return {
            'enabled': 1,
            'num_objects': int(num_objects),
            'kept_objects': int(num_objects),
            'removed_objects': 0,
            'removed_voxels': 0,
            'topology_slab_count': int(comp_stats.get('topology_slab_count', 0)),
            'topology_slab_workers': int(comp_stats.get('topology_slab_workers', 0)),
            'label_seconds': float(topology_times.get('slice_label', 0.0)),
            'pair_extraction_seconds': float(topology_times.get('internal_pair_extraction', 0.0)),
            'local_union_seconds': float(topology_times.get('local_slab_union', 0.0)),
            'boundary_merge_seconds': float(topology_times.get('boundary_merge', 0.0)),
            'root_expansion_seconds': float(topology_times.get('root_expansion', 0.0)),
            'area_reduction_seconds': float(topology_times.get('area_reduction', 0.0)),
            'topology_seconds': float(topology_times.get('topology_total', 0.0)),
            'metadata_seconds': float(metadata_seconds),
            'decision_seconds': 0.0,
            'lut_seconds': 0.0,
            'apply_seconds': 0.0,
            'total_seconds': float(total_seconds),
        }

    z_dim = int(mask_mm.shape[0])
    component_counts = np.asarray(comp_stats['component_counts'])
    slice_offsets = np.asarray(comp_stats['slice_offsets'])
    slice_bboxes = np.asarray(comp_stats['slice_bboxes'])
    root_map = np.asarray(comp_stats['root_map'])
    unique_roots = np.asarray(comp_stats['unique_roots'])
    root_areas = np.asarray(comp_stats['root_areas'])
    total_components = int(comp_stats['total_components'])

    decision_started = time.perf_counter()
    order = np.argsort(root_areas[unique_roots])[::-1]
    keep_roots = unique_roots[order[:keep_n]]
    keep_root_flag = np.zeros((int(total_components) + 1,), dtype=bool)
    keep_root_flag[keep_roots] = True
    gid_keep = keep_root_flag[root_map]
    gid_keep[0] = False
    # removed_voxels falls out of the root area table — no per-slice count pass.
    removed_voxels = int(root_areas[unique_roots].sum() - root_areas[keep_roots].sum())
    decision_seconds = float(time.perf_counter() - decision_started)

    # Per-slice local->keep LUTs in one concatenated uint8 table; only slices that actually
    # contain a dropped component are rewritten (the mask is 0/1 everywhere in this pipeline,
    # so untouched slices are already in their final state).
    keep_lut_started = time.perf_counter()
    lut_sizes = component_counts.astype(np.int64, copy=False) + 1
    lut_offsets = np.zeros((z_dim,), dtype=np.int64)
    if z_dim > 1:
        lut_offsets[1:] = np.cumsum(lut_sizes)[:-1]
    keep_flat = np.zeros((int(lut_sizes.sum()),), dtype=np.uint8)
    apply_slice = np.zeros((z_dim,), dtype=np.uint8)
    for z in range(z_dim):
        count = int(component_counts[int(z)])
        if count <= 0:
            continue
        offset = int(slice_offsets[int(z)])
        lo = int(lut_offsets[int(z)])
        lut = gid_keep[offset + 1:offset + count + 1]
        keep_flat[lo + 1:lo + count + 1] = lut
        if not bool(lut.all()):
            apply_slice[int(z)] = np.uint8(1)
    keep_lut_seconds = float(time.perf_counter() - keep_lut_started)

    apply_started = time.perf_counter()
    kernel_done = False
    if (
        isinstance(labels_mm, SparseSliceLabelStore)
        and compiled_topology_kernels_enabled()
        and _numba_sparse_keep_lut_apply_kernel is not None
    ):
        try:
            print(f'keep_objects: applying keep-largest-{keep_n} from sparse labels via numba nogil kernel')
            _numba_sparse_keep_lut_apply_kernel(
                labels_mm.flat,
                labels_mm.offsets,
                keep_flat,
                lut_offsets,
                np.ascontiguousarray(slice_bboxes),
                apply_slice,
                np.asarray(mask_mm),
            )
            kernel_done = True
        except Exception as exc:
            print(f'keep_objects: sparse numba apply unavailable ({exc}); using the thread pool.')
    elif compiled_topology_kernels_enabled() and _numba_keep_lut_apply_kernel is not None:
        try:
            print(f'keep_objects: applying keep-largest-{keep_n} via numba nogil kernel')
            _numba_keep_lut_apply_kernel(
                np.asarray(labels_mm),
                keep_flat,
                lut_offsets,
                np.ascontiguousarray(slice_bboxes),
                apply_slice,
                np.asarray(mask_mm),
            )
            kernel_done = True
        except Exception as exc:
            print(f'keep_objects: numba apply unavailable ({exc}); using the thread pool.')

    if not kernel_done:
        apply_zs = np.flatnonzero(apply_slice)

        def _apply_slice_fn(i: int) -> None:
            z = int(apply_zs[int(i)])
            y0, y1, x0, x1 = (int(v) for v in slice_bboxes[z])
            lo = int(lut_offsets[z])
            lut_u8 = keep_flat[lo:lo + int(component_counts[z]) + 1]
            labels_window = np.asarray(labels_mm[z, y0:y1, x0:x1])
            mask_mm[z, y0:y1, x0:x1] = lut_u8[labels_window]

        parallel_for_indices(
            int(apply_zs.size),
            _apply_slice_fn,
            max_workers=choose_slice_parallel_workers(int(workers), max(1, int(apply_zs.size))),
            desc=f'keep_objects: keep largest {keep_n}',
            show_progress=True,
        )
    flush_array(mask_mm)
    apply_seconds = float(time.perf_counter() - apply_started)

    close_memmap_array(labels_mm)
    if not bool(keep_temp):
        for lp in label_paths:
            try:
                lp.unlink(missing_ok=True)
            except Exception:
                pass

    topology_times = dict(comp_stats.get('topology_phase_seconds', {}))
    total_seconds = float(time.perf_counter() - keep_started)
    print(
        'v16.0.2 keep_objects phases: '
        f'label={float(topology_times.get("slice_label", 0.0)):.3f}s, '
        f'pairs={float(topology_times.get("internal_pair_extraction", 0.0)):.3f}s, '
        f'local_union={float(topology_times.get("local_slab_union", 0.0)):.3f}s, '
        f'boundary/root={float(topology_times.get("boundary_merge", 0.0)) + float(topology_times.get("root_expansion", 0.0)):.3f}s, '
        f'area={float(topology_times.get("area_reduction", 0.0)):.3f}s, '
        f'decision={decision_seconds:.3f}s, keep_lut={keep_lut_seconds:.3f}s, '
        f'apply={apply_seconds:.3f}s, total={total_seconds:.3f}s.'
    )
    return {
        'enabled': 1,
        'num_objects': int(num_objects),
        'kept_objects': int(min(keep_n, int(num_objects))),
        'removed_objects': int(max(0, int(num_objects) - keep_n)),
        'removed_voxels': int(removed_voxels),
        'topology_slab_count': int(comp_stats.get('topology_slab_count', 0)),
        'topology_slab_workers': int(comp_stats.get('topology_slab_workers', 0)),
        'label_seconds': float(topology_times.get('slice_label', 0.0)),
        'pair_extraction_seconds': float(topology_times.get('internal_pair_extraction', 0.0)),
        'local_union_seconds': float(topology_times.get('local_slab_union', 0.0)),
        'boundary_merge_seconds': float(topology_times.get('boundary_merge', 0.0)),
        'root_expansion_seconds': float(topology_times.get('root_expansion', 0.0)),
        'area_reduction_seconds': float(topology_times.get('area_reduction', 0.0)),
        'topology_seconds': float(topology_times.get('topology_total', 0.0)),
        'metadata_seconds': float(metadata_seconds),
        'decision_seconds': float(decision_seconds),
        'lut_seconds': float(keep_lut_seconds),
        'apply_seconds': float(apply_seconds),
        'total_seconds': float(total_seconds),
    }

def assemble_final_union_after_view_union(
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]],
    T: int,
    H: int,
    W: int,
    out_path: Path,
    temp_dir: Path,
    *,
    out_shape_tyx: Optional[Tuple[int, int, int]] = None,
    enable_3d_void_fill: bool = False,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    projected_layer_refs: Optional[Sequence['NrrdLayerRef']] = None,
) -> np.ndarray:
    """Build the final single-model view union and optionally apply one 3D void fill."""
    final_union_mm = assemble_current_view_union_volume(
        view_volumes_by_model=view_volumes_by_model,
        T=T,
        H=H,
        W=W,
        out_path=out_path,
        out_shape_tyx=out_shape_tyx,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        workers=int(workers),
        projected_layer_refs=projected_layer_refs,
    )

    if bool(enable_3d_void_fill):
        print('\n=== Optional 3D void fill after final global union ===')
        discard_binary_volume_slice_metadata(final_union_mm)
        final_void_dir = temp_dir / 'final_global_void_fill'
        final_void_dir.mkdir(parents=True, exist_ok=True)
        fill_3d_voids_inplace_streaming(
            final_union_mm,
            final_void_dir / 'final_union',
            keep_temp=bool(keep_temp),
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
    else:
        print('\n=== Optional 3D void fill disabled (--postprocessing 3d_void_fill not selected) ===')
    return final_union_mm

@dataclass(frozen=True)
class V14CenterlineSamples:
    """Ordered embedded medial-ridge samples in source ``(t,Y,X)`` coordinates."""

    points_tyx: np.ndarray
    tangents_tyx: np.ndarray
    radii: np.ndarray
    branch_ids: np.ndarray
    arc_indices: np.ndarray
    backend: str
    endpoint_count: int
    automatic_removal_allowed: bool
    details: Dict[str, object] = field(default_factory=dict, compare=False)

@dataclass(frozen=True)
class V14SectionEvidence:
    sample_index: int
    branch_id: int
    arc_index: int
    center_tyx: Tuple[float, float, float]
    radius: float
    voxel_tyx: np.ndarray = field(compare=False, repr=False)
    used_curved_path: bool = False
    event_id: int = -1

@dataclass(frozen=True)
class V14CenterlineEvent:
    event_id: int
    branch_id: int
    first_arc_index: int
    last_arc_index: int
    evidence_indices: Tuple[int, ...]
    clean_left: bool
    clean_right: bool
    min_t: int
    max_t: int

    @property
    def clean_flanks(self) -> bool:
        return bool(self.clean_left and self.clean_right)

def _v1401_axis_plane(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Return a 2D view whose row/column coordinates map deterministically to TYX."""
    if int(axis) == 0:
        return np.asarray(array[int(index), :, :])
    if int(axis) == 1:
        return np.asarray(array[:, int(index), :])
    if int(axis) == 2:
        return np.asarray(array[:, :, int(index)])
    raise ValueError(f'invalid embedded-centerline axis {axis}')

def _v1401_axis_point_tyx(
    axis: int,
    plane_index: int,
    row: int,
    col: int,
    spacing_tyx: Sequence[float],
    origin_tyx: Sequence[float],
) -> np.ndarray:
    if int(axis) == 0:
        coarse = (int(plane_index), int(row), int(col))
    elif int(axis) == 1:
        coarse = (int(row), int(plane_index), int(col))
    elif int(axis) == 2:
        coarse = (int(row), int(col), int(plane_index))
    else:
        raise ValueError(f'invalid embedded-centerline axis {axis}')
    return np.asarray([
        float(origin_tyx[dim]) + float(coarse[dim]) * float(spacing_tyx[dim])
        for dim in range(3)
    ], dtype=np.float64)

def _v1401_embedded_plane_ridges(
    mask_2d: np.ndarray,
    distance_2d: np.ndarray,
    *,
    spacing_rc: Tuple[float, float],
    max_peaks: int,
) -> Tuple[List[Tuple[int, int, float]], bool, bool]:
    """Extract radius-adaptively separated medial-ridge points from one plane.

 Equality with a 3x3 maximum is intentionally retained along flat medial
 plateaus. A second deterministic non-maximum-suppression step samples long
 plateaus instead of collapsing an in-plane airway branch to one point."""
    foreground = np.asarray(mask_2d) != 0
    if not bool(np.any(foreground)):
        return [], False, False
    distance = np.asarray(distance_2d, dtype=np.float64)
    local_maximum = ndi.maximum_filter(distance, size=3, mode='constant', cval=0.0)
    minimum_radius = 0.50 * min(float(spacing_rc[0]), float(spacing_rc[1]))
    candidates = foreground & (distance >= local_maximum - 1.0e-9) & (distance >= minimum_radius)
    coords = np.argwhere(candidates)
    if int(coords.shape[0]) <= 0:
        return [], False, False
    values = distance[coords[:, 0], coords[:, 1]]
    # Large one-voxel sheets can make every point a tied local maximum. Retain a
    # deterministic, radius-prioritized candidate pool before the adaptive NMS.
    candidate_cap = max(4096, int(max_peaks) * 64)
    candidate_pool_truncated = bool(int(coords.shape[0]) > int(candidate_cap))
    if bool(candidate_pool_truncated):
        kth = max(0, int(coords.shape[0]) - int(candidate_cap))
        selected = np.argpartition(values, kth)[kth:]
        coords = coords[selected]
        values = values[selected]
    order = np.lexsort((coords[:, 1], coords[:, 0], -values))
    suppressed = np.zeros(foreground.shape, dtype=np.uint8)
    accepted: List[Tuple[int, int, float]] = []
    peak_limit_reached = False
    row_spacing = max(1.0e-6, float(spacing_rc[0]))
    col_spacing = max(1.0e-6, float(spacing_rc[1]))
    for candidate_index in order:
        row = int(coords[int(candidate_index), 0])
        col = int(coords[int(candidate_index), 1])
        if int(suppressed[row, col]) != 0:
            continue
        radius = float(values[int(candidate_index)])
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        accepted.append((int(row), int(col), float(radius)))
        if len(accepted) >= int(max_peaks):
            peak_limit_reached = True
            break
        suppression = max(0.75 * min(row_spacing, col_spacing), 0.40 * float(radius))
        row_reach = max(1, int(math.ceil(float(suppression) / row_spacing)))
        col_reach = max(1, int(math.ceil(float(suppression) / col_spacing)))
        row0 = max(0, int(row) - int(row_reach))
        row1 = min(int(suppressed.shape[0]), int(row) + int(row_reach) + 1)
        col0 = max(0, int(col) - int(col_reach))
        col1 = min(int(suppressed.shape[1]), int(col) + int(col_reach) + 1)
        rr = (np.arange(row0, row1, dtype=np.float64) - float(row)) * row_spacing
        cc = (np.arange(col0, col1, dtype=np.float64) - float(col)) * col_spacing
        ellipse = rr.reshape(-1, 1) ** 2 + cc.reshape(1, -1) ** 2 <= float(suppression) ** 2
        suppressed[row0:row1, col0:col1][ellipse] = np.uint8(1)
    return accepted, bool(candidate_pool_truncated), bool(peak_limit_reached)

def _v1401_track_axis_ridges(
    ridge_mask: np.ndarray,
    distance: np.ndarray,
    *,
    support_mask: np.ndarray,
    axis: int,
    spacing_tyx: Tuple[float, float, float],
    origin_tyx: Tuple[float, float, float],
) -> Tuple[List[List[Tuple[np.ndarray, float]]], Dict[str, object]]:
    """Join planar ridge samples into ordered, one-to-one centerline tracks."""
    axis_i = int(axis)
    plane_count = int(ridge_mask.shape[axis_i])
    inplane_dims = tuple(dim for dim in range(3) if int(dim) != axis_i)
    spacing_rc = (
        float(spacing_tyx[int(inplane_dims[0])]),
        float(spacing_tyx[int(inplane_dims[1])]),
    )
    # The radius-resolved ridge mask is already globally bounded. A low
    # per-plane cap can nevertheless truncate a real sheet/slab exactly where it
    # is most informative, so retain up to 512 separated peaks on every plane.
    peaks_per_plane = 512
    max_gap = 2
    tracks: List[List[Tuple[np.ndarray, float]]] = []
    last_plane_by_track: List[int] = []
    nonempty_planes = 0
    sampled_planes = 0
    peak_count = 0
    candidate_pool_truncated = False
    peak_limit_reached = False
    for plane_index in range(int(plane_count)):
        support_plane = _v1401_axis_plane(support_mask, axis_i, int(plane_index))
        if not bool(np.any(support_plane)):
            continue
        nonempty_planes += 1
        mask_plane = _v1401_axis_plane(ridge_mask, axis_i, int(plane_index))
        if not bool(np.any(mask_plane)):
            continue
        distance_plane = _v1401_axis_plane(distance, axis_i, int(plane_index))
        ridges, plane_pool_truncated, plane_peak_limit = _v1401_embedded_plane_ridges(
            mask_plane,
            distance_plane,
            spacing_rc=spacing_rc,
            max_peaks=int(peaks_per_plane),
        )
        candidate_pool_truncated = bool(candidate_pool_truncated or plane_pool_truncated)
        peak_limit_reached = bool(peak_limit_reached or plane_peak_limit)
        if not ridges:
            continue
        sampled_planes += 1
        peak_count += int(len(ridges))
        points = [
            (
                _v1401_axis_point_tyx(
                    axis_i, int(plane_index), int(row), int(col),
                    spacing_tyx, origin_tyx,
                ),
                float(radius),
            )
            for row, col, radius in ridges
        ]
        points.sort(key=lambda item: (-float(item[1]),) + tuple(float(v) for v in item[0]))
        active_ids = [
            track_id for track_id, last_plane in enumerate(last_plane_by_track)
            if 0 < int(plane_index) - int(last_plane) <= int(max_gap)
        ]
        used_tracks: set[int] = set()
        for point, radius in points:
            best_track = -1
            best_score = float('inf')
            for track_id in active_ids:
                if int(track_id) in used_tracks:
                    continue
                previous_point, previous_radius = tracks[int(track_id)][-1]
                gap = max(1, int(plane_index) - int(last_plane_by_track[int(track_id)]))
                delta = np.asarray(point, dtype=np.float64) - np.asarray(previous_point, dtype=np.float64)
                inplane_distance = float(np.linalg.norm(delta[list(inplane_dims)]))
                allowed = max(
                    2.50 * max(spacing_rc) * float(gap),
                    0.85 * (float(radius) + float(previous_radius)) * math.sqrt(float(gap)),
                )
                score = float(inplane_distance) / float(max(1.0e-6, allowed))
                radius_ratio = max(float(radius), float(previous_radius)) / max(
                    1.0e-6, min(float(radius), float(previous_radius)),
                )
                if not (
                    float(score) <= 1.0
                    and float(radius_ratio) <= 5.0
                    and float(score) < float(best_score)
                ):
                    continue
                segment_steps = max(
                    2,
                    int(math.ceil(float(np.linalg.norm(delta)) / max(
                        1.0e-6, 0.50 * min(float(v) for v in spacing_tyx),
                    ))) + 1,
                )
                alpha = np.linspace(0.0, 1.0, int(segment_steps), dtype=np.float64)
                segment = (
                    np.asarray(previous_point, dtype=np.float64).reshape(1, 3)
                    + alpha.reshape(-1, 1) * delta.reshape(1, 3)
                )
                coarse_indices = np.rint(
                    (
                        segment - np.asarray(origin_tyx, dtype=np.float64).reshape(1, 3)
                    ) / np.asarray(spacing_tyx, dtype=np.float64).reshape(1, 3)
                ).astype(np.int64)
                segment_valid = (
                    (coarse_indices[:, 0] >= 0) & (coarse_indices[:, 0] < int(support_mask.shape[0]))
                    & (coarse_indices[:, 1] >= 0) & (coarse_indices[:, 1] < int(support_mask.shape[1]))
                    & (coarse_indices[:, 2] >= 0) & (coarse_indices[:, 2] < int(support_mask.shape[2]))
                )
                segment_in_foreground = bool(
                    np.all(segment_valid)
                    and np.all(support_mask[
                        coarse_indices[:, 0], coarse_indices[:, 1], coarse_indices[:, 2]
                    ] != 0)
                )
                if bool(segment_in_foreground):
                    best_track = int(track_id)
                    best_score = float(score)
            if int(best_track) < 0:
                tracks.append([(np.asarray(point, dtype=np.float64), float(radius))])
                last_plane_by_track.append(int(plane_index))
                used_tracks.add(int(len(tracks) - 1))
            else:
                tracks[int(best_track)].append((np.asarray(point, dtype=np.float64), float(radius)))
                last_plane_by_track[int(best_track)] = int(plane_index)
                used_tracks.add(int(best_track))
    # Two-sample chains overwhelmingly represent cap/corner ridges and cannot
    # satisfy the detector's three-sample longitudinal-event requirement.
    retained = [track for track in tracks if len(track) >= 3]
    return retained, {
        'axis': int(axis_i),
        'plane_count': int(plane_count),
        'nonempty_planes': int(nonempty_planes),
        'sampled_planes': int(sampled_planes),
        'sampled_plane_fraction': (
            float(sampled_planes) / float(nonempty_planes) if int(nonempty_planes) > 0 else 1.0
        ),
        'raw_peaks': int(peak_count),
        'candidate_pool_truncated': bool(candidate_pool_truncated),
        'peak_limit_reached': bool(peak_limit_reached),
        'tracks_before_min_length': int(len(tracks)),
        'tracks': int(len(retained)),
        'max_peaks_per_plane': int(peaks_per_plane),
    }

def _v1401_embedded_centerline_arrays(
    mask_tyx: np.ndarray,
    *,
    spacing_tyx: Tuple[float, float, float],
    origin_tyx: Tuple[float, float, float],
    target_points: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Compute ordered exact-EDT medial-ridge centerlines with NumPy/SciPy."""
    coarse = np.ascontiguousarray(np.asarray(mask_tyx) != 0, dtype=np.uint8)
    if not bool(np.any(coarse)):
        raise RuntimeError('embedded centerline received an empty foreground raster')
    # Analyze only the largest 26-connected coarse object so disconnected specks
    # cannot exhaust the track budget. This never edits ``mask_tyx``:
    # unanalysed objects remain byte-identical in the union and cannot be selected
    # for removal because they receive no centerline evidence.
    labels = np.empty(coarse.shape, dtype=np.int32)
    component_count = int(ndi.label(
        coarse,
        structure=np.ones((3, 3, 3), dtype=bool),
        output=labels,
    ))
    if int(component_count) <= 0:
        raise RuntimeError('embedded centerline found no connected foreground object')
    component_sizes = np.bincount(labels.reshape(-1))
    largest_label = 1 + int(np.argmax(component_sizes[1:]))
    total_foreground = int(np.count_nonzero(coarse))
    largest_foreground = int(component_sizes[int(largest_label)])
    if int(component_count) > 1:
        coarse = np.ascontiguousarray(labels == int(largest_label), dtype=np.uint8)
    del labels
    del component_sizes
    print(
        f'v14.0.1 embedded centerline: exact 3D EDT on {tuple(int(v) for v in coarse.shape)} '
        f'with source-voxel sampling {tuple(float(v) for v in spacing_tyx)}; '
        f'largest coarse object {int(largest_foreground)}/{int(total_foreground)} voxels '
        f'across {int(component_count)} component(s).'
    )
    distance = ndi.distance_transform_edt(
        coarse,
        sampling=tuple(float(v) for v in spacing_tyx),
    )
    # Shift center-to-center EDT radii by half a coarse cell to approximate the
    # binary object's 0.5 isosurface.
    half_cell = 0.5 * min(float(v) for v in spacing_tyx)
    foreground = coarse != 0
    distance[foreground] = np.maximum(0.5, distance[foreground] - float(half_cell))
    distance[~foreground] = 0.0
    ridge_votes = np.zeros(coarse.shape, dtype=np.uint8)
    for filter_size in ((1, 3, 3), (3, 1, 3), (3, 3, 1)):
        local_maximum = ndi.maximum_filter(
            distance,
            size=filter_size,
            mode='constant',
            cval=0.0,
            output=np.float32,
        )
        # ``local_maximum`` is float32 to halve peak scratch memory. Its ULP
        # grows with the EDT value, so a fixed epsilon can reject a true float64
        # maximum after rounding. Use a small scale-aware comparison tolerance.
        comparison_floor = np.asarray(local_maximum) * np.float32(1.0 - 2.0e-7) - np.float32(1.0e-6)
        ridge_votes += np.asarray(
            foreground & (distance >= comparison_floor),
            dtype=np.uint8,
        )
        del comparison_floor
        del local_maximum
    # All three plane tests must agree. A two-of-three rule admits the medial
    # *surface* of a capped cylinder and creates false transverse centerlines;
    # three-way agreement retains its 1D axis while still following oblique tubes.
    # One-coarse-voxel sheets are also medial plateaus, but they are surface-like
    # rather than usable lumen centerlines. Prefer a radius-resolved core and
    # relax only when that would otherwise make a genuinely thin object empty.
    ridge_consensus = foreground & (ridge_votes == 3)
    preferred_ridge_radius = 1.25 * min(float(v) for v in spacing_tyx)
    ridge_mask_bool = ridge_consensus & (distance >= float(preferred_ridge_radius))
    ridge_radius_relaxed = False
    if int(np.count_nonzero(ridge_mask_bool)) < 3:
        ridge_radius_relaxed = True
        preferred_ridge_radius = 0.50 * min(float(v) for v in spacing_tyx)
        ridge_mask_bool = ridge_consensus & (distance >= float(preferred_ridge_radius))
    ridge_mask = np.ascontiguousarray(ridge_mask_bool, dtype=np.uint8)
    del ridge_consensus
    del ridge_mask_bool
    ridge_candidate_voxels = int(np.count_nonzero(ridge_mask))
    del ridge_votes
    if int(ridge_candidate_voxels) <= 0:
        raise RuntimeError('embedded orthogonal medial-ridge consensus is empty')
    raw_sample_budget = max(4000, min(50000, int(target_points) * 4))
    all_tracks: List[List[Tuple[np.ndarray, float]]] = []
    axis_stats: List[Dict[str, object]] = []
    for axis in range(3):
        tracks, stats = _v1401_track_axis_ridges(
            ridge_mask,
            distance,
            support_mask=coarse,
            axis=int(axis),
            spacing_tyx=tuple(float(v) for v in spacing_tyx),
            origin_tyx=tuple(float(v) for v in origin_tyx),
        )
        axis_stats.append(stats)
        all_tracks.extend(tracks)
    del distance
    del ridge_mask

    # A plane sweep perpendicular to a clean tube also traces a short path across
    # its diameter. That is not a lumen centerline. Require tube-like
    # longitudinal persistence (path length >= 3 local radii), then suppress
    # near-duplicate paths emitted by the three orthogonal sweeps. Missed paths
    # are conservative: without centerline evidence they cannot trigger removal.
    longitudinal_tracks: List[Tuple[List[Tuple[np.ndarray, float]], float, float]] = []
    rejected_short_tracks = 0
    for track in all_tracks:
        track_points = np.asarray([item[0] for item in track], dtype=np.float64)
        path_length = float(np.sum(np.linalg.norm(np.diff(track_points, axis=0), axis=1)))
        median_track_radius = float(np.median(np.asarray([item[1] for item in track], dtype=np.float64)))
        minimum_length = max(
            2.0 * min(float(v) for v in spacing_tyx),
            3.0 * float(median_track_radius),
        )
        if not np.isfinite(path_length) or float(path_length) < float(minimum_length):
            rejected_short_tracks += 1
            continue
        longitudinal_tracks.append((track, float(path_length), float(median_track_radius)))
    longitudinal_tracks.sort(
        key=lambda item: (
            -float(item[1]) / max(1.0e-6, float(item[2])),
            -float(item[1]),
            -max(float(sample[1]) for sample in item[0]),
            tuple(float(v) for v in item[0][0][0]),
        )
    )
    covered = np.zeros(coarse.shape, dtype=np.uint8)
    deduplicated_tracks: List[List[Tuple[np.ndarray, float]]] = []
    rejected_duplicate_tracks = 0
    spacing_array = np.asarray(spacing_tyx, dtype=np.float64).reshape(1, 3)
    origin_array = np.asarray(origin_tyx, dtype=np.float64).reshape(1, 3)
    for track, _path_length, _median_track_radius in longitudinal_tracks:
        track_points = np.asarray([item[0] for item in track], dtype=np.float64)
        indices = np.rint((track_points - origin_array) / spacing_array).astype(np.int64)
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < int(coarse.shape[0]))
            & (indices[:, 1] >= 0) & (indices[:, 1] < int(coarse.shape[1]))
            & (indices[:, 2] >= 0) & (indices[:, 2] < int(coarse.shape[2]))
        )
        indices = np.unique(indices[valid], axis=0)
        if int(indices.shape[0]) <= 0:
            rejected_duplicate_tracks += 1
            continue
        new_fraction = float(np.mean(covered[indices[:, 0], indices[:, 1], indices[:, 2]] == 0))
        if float(new_fraction) < 0.50:
            rejected_duplicate_tracks += 1
            continue
        deduplicated_tracks.append(track)
        for coarse_index in indices:
            t_idx, y_idx, x_idx = (int(v) for v in coarse_index)
            covered[
                max(0, t_idx - 1):min(int(coarse.shape[0]), t_idx + 2),
                max(0, y_idx - 1):min(int(coarse.shape[1]), y_idx + 2),
                max(0, x_idx - 1):min(int(coarse.shape[2]), x_idx + 2),
            ] = np.uint8(1)
    del covered

    # Prefer long, thick, non-duplicate tracks if pathological sheets still
    # exceed the global sample bound.
    retained_tracks: List[List[Tuple[np.ndarray, float]]] = []
    retained_points = 0
    truncated = False
    for track in deduplicated_tracks:
        if retained_points + len(track) > int(raw_sample_budget):
            truncated = True
            continue
        retained_tracks.append(track)
        retained_points += int(len(track))
    if not retained_tracks:
        raise RuntimeError('embedded medial-ridge tracking returned no multi-plane tracks')

    points: List[np.ndarray] = []
    tangents: List[np.ndarray] = []
    radii: List[float] = []
    branches: List[int] = []
    arcs: List[int] = []
    for branch_id, track in enumerate(retained_tracks):
        track_points = np.asarray([item[0] for item in track], dtype=np.float64)
        track_radii = np.asarray([item[1] for item in track], dtype=np.float64)
        for arc_index in range(int(track_points.shape[0])):
            left = track_points[max(0, int(arc_index) - 1)]
            right = track_points[min(int(track_points.shape[0]) - 1, int(arc_index) + 1)]
            tangent = np.asarray(right - left, dtype=np.float64)
            norm = float(np.linalg.norm(tangent))
            if not np.isfinite(norm) or norm <= 1.0e-8:
                continue
            radius = float(track_radii[int(arc_index)])
            if not np.isfinite(radius) or radius <= 0.0:
                continue
            points.append(np.asarray(track_points[int(arc_index)], dtype=np.float64))
            tangents.append(tangent / norm)
            radii.append(float(radius))
            branches.append(int(branch_id))
            arcs.append(int(arc_index))
    if not points:
        raise RuntimeError('embedded medial-ridge tracks contained no finite samples')
    best_axis_coverage = max(
        (float(item.get('sampled_plane_fraction', 0.0)) for item in axis_stats),
        default=0.0,
    )
    planar_cap_reached = bool(any(
        bool(item.get('candidate_pool_truncated', False))
        or bool(item.get('peak_limit_reached', False))
        for item in axis_stats
    ))
    automatic_allowed = bool(
        not truncated
        and not planar_cap_reached
        and not ridge_radius_relaxed
        and float(best_axis_coverage) >= 0.75
    )
    arrays = {
        'points_tyx': np.asarray(points, dtype=np.float64),
        'tangents_tyx': np.asarray(tangents, dtype=np.float64),
        'radii': np.asarray(radii, dtype=np.float64),
        'branch_ids': np.asarray(branches, dtype=np.int32),
        'arc_indices': np.asarray(arcs, dtype=np.int32),
        'endpoint_count': np.asarray([2 * len(retained_tracks)], dtype=np.int32),
        'automatic_removal_allowed': np.asarray([int(automatic_allowed)], dtype=np.uint8),
    }
    details: Dict[str, object] = {
        'algorithm': 'three_axis_exact_edt_medial_ridge_tracking',
        'coarse_component_count': int(component_count),
        'coarse_total_foreground_voxels': int(total_foreground),
        'coarse_largest_component_voxels': int(largest_foreground),
        'coarse_largest_component_fraction': (
            float(largest_foreground) / float(total_foreground)
            if int(total_foreground) > 0 else 0.0
        ),
        'minimum_ridge_radius': float(preferred_ridge_radius),
        'minimum_ridge_radius_relaxed': bool(ridge_radius_relaxed),
        'axis_stats': axis_stats,
        'raw_sample_budget': int(raw_sample_budget),
        'ridge_candidate_voxels': int(ridge_candidate_voxels),
        'tracks_before_longitudinal_filter': int(len(all_tracks)),
        'tracks_rejected_as_short': int(rejected_short_tracks),
        'tracks_before_deduplication': int(len(longitudinal_tracks)),
        'tracks_rejected_as_duplicates': int(rejected_duplicate_tracks),
        'tracks_before_cap': int(len(deduplicated_tracks)),
        'tracks': int(len(retained_tracks)),
        'pre_dense_sample_count': int(len(points)),
        'raw_sample_cap_reached': bool(truncated),
        'planar_candidate_cap_reached': bool(planar_cap_reached),
        'best_axis_sampled_plane_fraction': float(best_axis_coverage),
        'automatic_removal_allowed': bool(automatic_allowed),
        'maximum_radius': float(np.max(np.asarray(radii, dtype=np.float64))),
        'median_radius': float(np.median(np.asarray(radii, dtype=np.float64))),
    }
    return arrays, details

def _v1401_embedded_worker(
    coarse_npy: str,
    spacing_tyx: Tuple[float, float, float],
    origin_tyx: Tuple[float, float, float],
    target_points: int,
    result_npz: str,
    status_json: str,
) -> None:
    """Isolated bounded-memory worker for the default backend."""
    try:
        coarse = np.load(str(coarse_npy), mmap_mode='r')
        arrays, details = _v1401_embedded_centerline_arrays(
            coarse,
            spacing_tyx=tuple(float(v) for v in spacing_tyx),
            origin_tyx=tuple(float(v) for v in origin_tyx),
            target_points=int(target_points),
        )
        np.savez_compressed(str(result_npz), **arrays)
        Path(status_json).write_text(json.dumps({
            'ok': True,
            'sample_count': int(arrays['points_tyx'].shape[0]),
            'endpoint_count': int(arrays['endpoint_count'][0]),
            **details,
        }, indent=2) + '\n')
    except BaseException as exc:
        import traceback as _traceback
        Path(status_json).write_text(json.dumps({
            'ok': False,
            'error': f'{type(exc).__name__}: {exc}',
            'traceback': _traceback.format_exc(),
        }, indent=2) + '\n')
        raise

def _v14_build_coarse_centerline_raster(
    mask_mm: np.ndarray,
    *,
    temp_dir: Path,
    pass_index: int,
    max_dim: int,
    workers: int,
) -> Tuple[Optional[Path], Dict[str, object]]:
    extent = compute_segment_extent_zyx(mask_mm, workers=int(workers))
    x0, x1, y0, y1, t0, t1 = (int(v) for v in extent)
    if x1 < x0 or y1 < y0 or t1 < t0:
        return None, {'empty': True, 'extent_xyt': [int(v) for v in extent]}
    crop_shape = (int(t1 - t0 + 1), int(y1 - y0 + 1), int(x1 - x0 + 1))
    max_axis = max(int(v) for v in crop_shape)
    max_dim_i = max(64, int(max_dim))
    factor_by_dim = int(math.ceil(float(max_axis) / float(max_dim_i)))
    crop_voxels = int(crop_shape[0]) * int(crop_shape[1]) * int(crop_shape[2])
    factor_by_voxels = int(math.ceil(max(1.0, (float(crop_voxels) / 64_000_000.0) ** (1.0 / 3.0))))
    factor = max(1, int(factor_by_dim), int(factor_by_voxels))
    out_shape = tuple(int(math.ceil(float(dim) / float(factor))) for dim in crop_shape)
    padded_shape = (int(out_shape[0]) + 2, int(out_shape[1]) + 2, int(out_shape[2]) + 2)
    work_dir = temp_dir / 'centerline_filter' / f'pass{int(pass_index):02d}'
    work_dir.mkdir(parents=True, exist_ok=True)
    coarse_path = work_dir / 'centerline_blockmax.npy'
    coarse = np.lib.format.open_memmap(
        coarse_path, mode='w+', dtype=np.uint8, shape=padded_shape,
    )
    coarse[:] = np.uint8(0)
    y_starts = np.arange(0, int(crop_shape[1]), int(factor), dtype=np.int64)
    x_starts = np.arange(0, int(crop_shape[2]), int(factor), dtype=np.int64)
    print(
        f'v14 centerline pass {int(pass_index)}: block-max foreground crop '
        f'{crop_shape} -> {out_shape} (factor={int(factor)}, padded={padded_shape}).'
    )
    # Sequential T blocks keep reads ordered through the source memmap. NumPy's
    # reduceat performs the Y/X block maxima in compiled loops; no thin foreground
    # is lost as it would be by simple striding.
    for out_t in range(int(out_shape[0])):
        src_t0 = int(t0) + int(out_t) * int(factor)
        src_t1 = min(int(t1) + 1, int(src_t0) + int(factor))
        acc = np.zeros((int(crop_shape[1]), int(crop_shape[2])), dtype=np.uint8)
        for src_t in range(int(src_t0), int(src_t1)):
            np.maximum(
                acc,
                np.asarray(mask_mm[int(src_t), int(y0):int(y1) + 1, int(x0):int(x1) + 1], dtype=np.uint8),
                out=acc,
            )
        pooled_y = np.maximum.reduceat(acc, y_starts, axis=0)
        pooled = np.maximum.reduceat(pooled_y, x_starts, axis=1)
        coarse[int(out_t) + 1, 1:int(out_shape[1]) + 1, 1:int(out_shape[2]) + 1] = pooled
    coarse.flush()
    del coarse
    center_offset = (float(factor) - 1.0) * 0.5
    origin_tyx = (
        float(t0) + center_offset - float(factor),
        float(y0) + center_offset - float(factor),
        float(x0) + center_offset - float(factor),
    )
    return coarse_path, {
        'empty': False,
        'extent_xyt': [int(v) for v in extent],
        'crop_shape_tyx': [int(v) for v in crop_shape],
        'coarse_shape_tyx': [int(v) for v in padded_shape],
        'blockmax_factor': int(factor),
        'spacing_tyx': [float(factor), float(factor), float(factor)],
        'origin_tyx': [float(v) for v in origin_tyx],
    }

def _v14_fallback_centerline_samples(
    mask_mm: np.ndarray,
    *,
    extent_xyt: Sequence[int],
    max_samples: int = 512,
) -> V14CenterlineSamples:
    """Last-resort marker-only ridge used when the selected backend fails.

 This is deliberately not authorized to remove voxels. It finds the deepest
 point in the largest 2D component on ordered source slices, forming a bounded
 approximate ridge so audit markers can still be generated and synthetic tests
 can exercise the full plane/component machinery."""
    x0, x1, y0, y1, t0, t1 = (int(v) for v in extent_xyt)
    if x1 < x0 or y1 < y0 or t1 < t0:
        return V14CenterlineSamples(
            np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32), 'geometric_marker_only', 0, False,
            {'reason': 'empty_union'},
        )
    span = int(t1 - t0 + 1)
    stride = max(1, int(math.ceil(float(span) / float(max(2, int(max_samples))))))
    slice_ids = list(range(int(t0), int(t1) + 1, int(stride)))
    if slice_ids and int(slice_ids[-1]) != int(t1):
        slice_ids.append(int(t1))
    points: List[Tuple[float, float, float]] = []
    radii: List[float] = []
    for t_idx in slice_ids:
        crop = np.ascontiguousarray(
            np.asarray(mask_mm[int(t_idx), int(y0):int(y1) + 1, int(x0):int(x1) + 1]) != 0,
            dtype=np.uint8,
        )
        local_x0, local_y0, local_w, local_h = (int(v) for v in cv2.boundingRect(crop))
        if local_w <= 0 or local_h <= 0:
            continue
        crop = np.ascontiguousarray(
            crop[
                int(local_y0):int(local_y0) + int(local_h),
                int(local_x0):int(local_x0) + int(local_w),
            ],
            dtype=np.uint8,
        )
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            crop, connectivity=8, ltype=cv2.CV_32S,
        )
        if int(count) <= 1:
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = np.ascontiguousarray(labels == int(largest), dtype=np.uint8)
        dist = cv2.distanceTransform(component, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(dist)
        if float(max_val) <= 0.0:
            continue
        points.append((
            float(t_idx),
            float(y0 + int(local_y0) + int(max_loc[1])),
            float(x0 + int(local_x0) + int(max_loc[0])),
        ))
        radii.append(float(max_val))
    if len(points) < 2:
        return V14CenterlineSamples(
            np.asarray(points, dtype=np.float64).reshape((-1, 3)),
            np.zeros((len(points), 3), dtype=np.float64), np.asarray(radii, dtype=np.float64),
            np.zeros((len(points),), dtype=np.int32), np.arange(len(points), dtype=np.int32),
            'geometric_marker_only', 0, False,
            {'reason': 'fewer_than_two_ridge_samples', 'stride': int(stride)},
        )
    point_array = np.asarray(points, dtype=np.float64)
    tangents = np.zeros_like(point_array)
    for idx in range(int(point_array.shape[0])):
        left = point_array[max(0, idx - 1)]
        right = point_array[min(int(point_array.shape[0]) - 1, idx + 1)]
        delta = right - left
        norm = float(np.linalg.norm(delta))
        if norm > 0.0:
            tangents[idx] = delta / norm
    return V14CenterlineSamples(
        point_array, tangents, np.asarray(radii, dtype=np.float64),
        np.zeros((len(points),), dtype=np.int32), np.arange(len(points), dtype=np.int32),
        'geometric_marker_only', 0, False,
        {'stride': int(stride), 'sample_count': int(len(points))},
    )

def _v14_run_isolated_process(
    process: mp.Process,
    *,
    timeout_seconds: float,
) -> Tuple[Optional[int], str]:
    """Run a centerline child without allowing it to strand interpreter exit."""
    try:
        process.start()
    except BaseException as exc:
        try:
            process.close()
        except Exception:
            pass
        return None, f'could not start isolated centerline process: {type(exc).__name__}: {exc}'
    error = ''
    try:
        process.join(timeout=max(1.0, float(timeout_seconds)))
        if process.is_alive():
            error = f'timed out after {float(timeout_seconds):g}s'
            process.terminate()
            process.join(timeout=30.0)
        if process.is_alive():
            if hasattr(process, 'kill'):
                process.kill()
            else:  # pragma: no cover - Python versions without Process.kill
                process.terminate()
            process.join(timeout=10.0)
        if process.is_alive():
            raise RuntimeError(
                f'{error}; isolated centerline process could not be stopped safely'
            )
        return int(process.exitcode) if process.exitcode is not None else None, str(error)
    finally:
        if not process.is_alive():
            try:
                process.close()
            except Exception:
                pass

def _v14_extract_centerline_samples(
    mask_mm: np.ndarray,
    *,
    temp_dir: Path,
    pass_index: int,
    backend: str,
    surface_max_dim: int,
    surface_points: int,
    timeout_seconds: float,
    workers: int,
    keep_temp: bool,
) -> V14CenterlineSamples:
    requested = str(backend).strip().lower()
    if requested != 'embedded':
        raise ValueError(f'unsupported centerline backend {backend!r}')

    coarse_path, raster_info = _v14_build_coarse_centerline_raster(
        mask_mm,
        temp_dir=temp_dir,
        pass_index=int(pass_index),
        max_dim=int(surface_max_dim),
        workers=int(workers),
    )
    if coarse_path is None:
        return _v14_fallback_centerline_samples(
            mask_mm, extent_xyt=raster_info.get('extent_xyt', _nrrd_empty_segment_extent()),
        )
    work_dir = Path(coarse_path).parent
    errors: Dict[str, str] = {}

    def _cleanup_attempt(result_path: Path, status_path: Path) -> None:
        if bool(keep_temp):
            return
        for path in (result_path, status_path):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    def _load_contract(
        result_path: Path,
        status_path: Path,
        *,
        backend_name: str,
        automatic_removal_allowed: Optional[bool],
    ) -> V14CenterlineSamples:
        with np.load(result_path, allow_pickle=False) as data:
            points = np.asarray(data['points_tyx'], dtype=np.float64).copy()
            tangents = np.asarray(data['tangents_tyx'], dtype=np.float64).copy()
            radii = np.asarray(data['radii'], dtype=np.float64).reshape(-1).copy()
            branches = np.asarray(data['branch_ids'], dtype=np.int32).reshape(-1).copy()
            arcs = np.asarray(data['arc_indices'], dtype=np.int32).reshape(-1).copy()
            endpoint_count = int(np.asarray(data['endpoint_count']).reshape(-1)[0])
            if automatic_removal_allowed is None:
                if 'automatic_removal_allowed' not in data.files:
                    raise RuntimeError('backend result omitted automatic_removal_allowed')
                allow_remove = bool(int(np.asarray(data['automatic_removal_allowed']).reshape(-1)[0]))
            else:
                allow_remove = bool(automatic_removal_allowed)
        count = int(points.shape[0])
        if points.ndim != 2 or tuple(points.shape[1:]) != (3,):
            raise RuntimeError(f'points_tyx has invalid shape {tuple(points.shape)}')
        if tangents.shape != points.shape:
            raise RuntimeError(f'tangents shape {tuple(tangents.shape)} != points {tuple(points.shape)}')
        if any(int(arr.shape[0]) != count for arr in (radii, branches, arcs)):
            raise RuntimeError('centerline result arrays have inconsistent lengths')
        tangent_norms = np.linalg.norm(tangents, axis=1)
        valid = (
            np.all(np.isfinite(points), axis=1)
            & np.all(np.isfinite(tangents), axis=1)
            & np.isfinite(radii)
            & (radii > 0.0)
            & np.isfinite(tangent_norms)
            & (tangent_norms > 1.0e-8)
        )
        if not bool(np.all(valid)) or count <= 0:
            raise RuntimeError(
                f'centerline result contains {int(count - np.count_nonzero(valid))} invalid sample(s)'
            )
        tangents /= tangent_norms.reshape(-1, 1)
        # A branch may not repeat or reverse arc indices: downstream densification
        # trusts each ordered branch and must never draw a disconnected shortcut.
        for branch_id in np.unique(branches):
            branch_arcs = arcs[branches == int(branch_id)]
            if int(branch_arcs.size) <= 0 or bool(np.any(np.diff(branch_arcs.astype(np.int64)) <= 0)):
                raise RuntimeError(f'branch {int(branch_id)} has non-increasing arc indices')
        details = dict(raster_info)
        if status_path.exists():
            details.update(json.loads(status_path.read_text()))
        details['contract_validated'] = True
        return V14CenterlineSamples(
            points, tangents, radii, branches, arcs, str(backend_name),
            int(endpoint_count), bool(allow_remove), details,
        )

    def _attempt_backend(
        name: str,
        target: object,
        *,
        automatic_removal_allowed: Optional[bool],
    ) -> Optional[V14CenterlineSamples]:
        result_path = work_dir / f'{str(name)}_centerlines.npz'
        status_path = work_dir / f'{str(name)}_status.json'
        # A retained/reused scratch directory must never let a stale successful
        # artifact masquerade as the current child result.
        result_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
        context = mp.get_context('spawn')
        process = context.Process(
            target=target,
            args=(
                str(coarse_path),
                tuple(float(v) for v in raster_info['spacing_tyx']),
                tuple(float(v) for v in raster_info['origin_tyx']),
                int(surface_points),
                str(result_path),
                str(status_path),
            ),
            daemon=False,
        )
        exit_code, process_error = _v14_run_isolated_process(
            process, timeout_seconds=float(timeout_seconds),
        )
        error = str(process_error)
        if not error and int(exit_code or 0) != 0:
            error = f'isolated process exit code {exit_code}'
        if not error and not result_path.exists():
            error = 'isolated process produced no result file'
        samples: Optional[V14CenterlineSamples] = None
        if not error:
            try:
                samples = _load_contract(
                    result_path,
                    status_path,
                    backend_name='embedded_edt_medial_ridges',
                    automatic_removal_allowed=automatic_removal_allowed,
                )
            except Exception as exc:
                error = f'invalid isolated {str(name)} result: {type(exc).__name__}: {exc}'
        if error and status_path.exists():
            try:
                error = str(json.loads(status_path.read_text()).get('error', error))
            except Exception:
                pass
        _cleanup_attempt(result_path, status_path)
        if samples is None:
            errors[str(name)] = str(error or 'unknown backend failure')
        return samples

    samples = _attempt_backend(
        'embedded', _v1401_embedded_worker,
        automatic_removal_allowed=None,
    )
    if samples is not None:
        try:
            samples = _v1401_refine_fullres_center_samples(mask_mm, samples)
        finally:
            if not bool(keep_temp):
                Path(coarse_path).unlink(missing_ok=True)
        print(
            f'Centerline pass {int(pass_index)}: embedded exact-EDT medial ridges, '
            f'samples={int(samples.points_tyx.shape[0])}, tracks='
            f"{int(samples.details.get('tracks', 0))}, "
            f'automatic_removal_allowed={bool(samples.automatic_removal_allowed)}.'
        )
        return samples

    if not bool(keep_temp):
        Path(coarse_path).unlink(missing_ok=True)
    error = str(errors.get('embedded', 'unknown embedded-centerline failure'))
    print(
        f'Warning: embedded centerline failed ({error}); using the bounded '
        'marker-only fallback. Automatic removal remains disabled.'
    )
    fallback = _v14_fallback_centerline_samples(
        mask_mm, extent_xyt=raster_info.get('extent_xyt', _nrrd_empty_segment_extent()),
    )
    return dataclasses_replace(
        fallback,
        details={**dict(raster_info), **dict(fallback.details), 'error': str(error)},
    )

def _v14_subsample_centerline_samples(samples: V14CenterlineSamples) -> V14CenterlineSamples:
    """Densify ordered cells so anomaly spans cover every crossed source slice.

 The historical name is retained to avoid disturbing callers. Radius-adaptive
 spacing still bounds in-plane work, but the t increment is never greater than
 0.75 source voxels. Thus a 100-slice web is inspected on all crossed source
 slices instead of only at sparse backend vertices. If the global safety cap is
 reached, deletion is disabled and the retained samples remain marker-only."""
    input_count = int(samples.points_tyx.shape[0])
    if input_count <= 1:
        return samples
    max_samples = max(1000, _env_int('YOLO_TTA_CENTERLINE_MAX_DENSE_SAMPLES', 50000))
    points_out: List[np.ndarray] = []
    tangents_out: List[np.ndarray] = []
    radii_out: List[float] = []
    branches_out: List[int] = []
    arcs_out: List[int] = []
    truncated = False
    for branch_id in np.unique(samples.branch_ids):
        branch_idx = np.flatnonzero(samples.branch_ids == int(branch_id))
        if int(branch_idx.size) <= 0:
            continue
        order = branch_idx[np.argsort(samples.arc_indices[branch_idx], kind='stable')]
        branch_arc = 0
        if int(order.size) == 1:
            only = int(order[0])
            points_out.append(np.asarray(samples.points_tyx[only], dtype=np.float64))
            tangents_out.append(np.asarray(samples.tangents_tyx[only], dtype=np.float64))
            radii_out.append(float(samples.radii[only]))
            branches_out.append(int(branch_id))
            arcs_out.append(0)
            if len(points_out) >= int(max_samples):
                truncated = True
                break
            continue
        for local_pos in range(max(1, int(order.size) - 1)):
            idx0 = int(order[min(local_pos, int(order.size) - 1)])
            idx1 = int(order[min(local_pos + 1, int(order.size) - 1)])
            point0 = np.asarray(samples.points_tyx[idx0], dtype=np.float64)
            point1 = np.asarray(samples.points_tyx[idx1], dtype=np.float64)
            delta = point1 - point0
            distance = float(np.linalg.norm(delta))
            desired = max(1.0, min(4.0, 0.5 * max(
                float(samples.radii[idx0]), float(samples.radii[idx1]), 0.5,
            )))
            steps = max(
                1,
                int(math.ceil(abs(float(delta[0])) / 0.75)),
                int(math.ceil(float(distance) / float(desired))),
            )
            first_step = 0 if local_pos == 0 else 1
            for step in range(int(first_step), int(steps) + 1):
                alpha = float(step) / float(max(1, steps))
                point = point0 + alpha * delta
                tangent = (
                    (1.0 - alpha) * np.asarray(samples.tangents_tyx[idx0], dtype=np.float64)
                    + alpha * np.asarray(samples.tangents_tyx[idx1], dtype=np.float64)
                )
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm <= 1.0e-8 and distance > 1.0e-8:
                    tangent = delta / distance
                elif tangent_norm > 1.0e-8:
                    tangent = tangent / tangent_norm
                points_out.append(point)
                tangents_out.append(tangent)
                radii_out.append(float(
                    (1.0 - alpha) * float(samples.radii[idx0])
                    + alpha * float(samples.radii[idx1])
                ))
                branches_out.append(int(branch_id))
                arcs_out.append(int(branch_arc))
                branch_arc += 1
                if len(points_out) >= int(max_samples):
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break
    if not points_out:
        return samples
    return V14CenterlineSamples(
        np.asarray(points_out, dtype=np.float64),
        np.asarray(tangents_out, dtype=np.float64),
        np.asarray(radii_out, dtype=np.float64),
        np.asarray(branches_out, dtype=np.int32),
        np.asarray(arcs_out, dtype=np.int32),
        str(samples.backend), int(samples.endpoint_count),
        bool(samples.automatic_removal_allowed and not truncated),
        {
            **dict(samples.details),
            'pre_dense_sample_count': int(input_count),
            'sample_count': int(len(points_out)),
            'dense_sample_cap': int(max_samples),
            'dense_sample_cap_reached': bool(truncated),
            'automatic_removal_allowed': bool(
                samples.automatic_removal_allowed and not truncated
            ),
        },
    )

def _v14_round_and_sample_mask(
    mask_mm: np.ndarray,
    coords_tyx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords_tyx, dtype=np.float64).reshape((-1, 3))
    rounded = np.rint(coords).astype(np.int64, copy=False)
    shape = tuple(int(v) for v in mask_mm.shape)
    valid = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] < int(shape[0]))
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < int(shape[1]))
        & (rounded[:, 2] >= 0) & (rounded[:, 2] < int(shape[2]))
    )
    values = np.zeros((int(rounded.shape[0]),), dtype=bool)
    if np.any(valid):
        active = rounded[valid]
        values[valid] = np.asarray(
            mask_mm[active[:, 0], active[:, 1], active[:, 2]], dtype=np.uint8,
        ) != 0
    return rounded, valid, values

def _v14_snap_center_to_foreground(
    mask_mm: np.ndarray,
    point_tyx: np.ndarray,
    radius: float,
) -> Optional[np.ndarray]:
    point = np.asarray(point_tyx, dtype=np.float64).reshape(3)
    rounded, valid, values = _v14_round_and_sample_mask(mask_mm, point.reshape(1, 3))
    if bool(valid[0]) and bool(values[0]):
        return np.asarray(rounded[0], dtype=np.float64)
    # a block-max center can be several source voxels from the thin
    # foreground voxel that admitted its coarse block (factor 6--8 is common).
    # The search runs only when the rounded center already missed foreground.
    reach = max(1, min(16, int(math.ceil(max(0.5, float(radius)) + 1.0))))
    base = np.rint(point).astype(np.int64)
    t0 = max(0, int(base[0]) - reach)
    t1 = min(int(mask_mm.shape[0]), int(base[0]) + reach + 1)
    y0 = max(0, int(base[1]) - reach)
    y1 = min(int(mask_mm.shape[1]), int(base[1]) + reach + 1)
    x0 = max(0, int(base[2]) - reach)
    x1 = min(int(mask_mm.shape[2]), int(base[2]) + reach + 1)
    local = np.argwhere(np.asarray(mask_mm[t0:t1, y0:y1, x0:x1]) != 0)
    if int(local.shape[0]) <= 0:
        return None
    global_coords = local.astype(np.float64)
    global_coords[:, 0] += float(t0)
    global_coords[:, 1] += float(y0)
    global_coords[:, 2] += float(x0)
    distances = np.sum((global_coords - point.reshape(1, 3)) ** 2, axis=1)
    return global_coords[int(np.argmin(distances))]

def _v1401_refine_fullres_center_samples(
    mask_mm: np.ndarray,
    samples: V14CenterlineSamples,
) -> V14CenterlineSamples:
    """Snap coarse block centers to full-resolution foreground before detection.

 A bounded failed-center count is a deletion safety gate, not a fatal error:
 marker-only analysis can still use the section detector's per-sample snap."""
    count = int(samples.points_tyx.shape[0])
    if count <= 0:
        return samples
    _rounded, valid, values = _v14_round_and_sample_mask(mask_mm, samples.points_tyx)
    missing = np.flatnonzero(~(valid & values))
    points = np.asarray(samples.points_tyx, dtype=np.float64).copy()
    validation_cap = max(256, _env_int('YOLO_TTA_CENTERLINE_FULLRES_SNAP_CAP', 4096))
    failures = 0
    validated = min(int(missing.size), int(validation_cap))
    for sample_index in missing[:int(validated)]:
        snapped = _v14_snap_center_to_foreground(
            mask_mm,
            points[int(sample_index)],
            float(samples.radii[int(sample_index)]),
        )
        if snapped is None:
            failures += 1
        else:
            points[int(sample_index)] = np.asarray(snapped, dtype=np.float64)
    complete = bool(int(missing.size) <= int(validation_cap) and int(failures) == 0)
    details = {
        **dict(samples.details),
        'fullres_center_count': int(count),
        'fullres_centers_initially_on_foreground': int(count - int(missing.size)),
        'fullres_centers_requiring_snap': int(missing.size),
        'fullres_centers_snap_validated': int(validated),
        'fullres_center_snap_failures': int(failures),
        'fullres_center_snap_validation_complete': bool(complete),
        'automatic_removal_allowed': bool(
            samples.automatic_removal_allowed and complete
        ),
    }
    return dataclasses_replace(
        samples,
        points_tyx=points,
        automatic_removal_allowed=bool(samples.automatic_removal_allowed and complete),
        details=details,
    )

def _v14_plane_basis(tangent_tyx: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    normal = np.asarray(tangent_tyx, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1.0e-8:
        return None
    normal = normal / norm
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    first = np.cross(normal, axis)
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1.0e-8:
        return None
    first /= first_norm
    second = np.cross(normal, first)
    second_norm = float(np.linalg.norm(second))
    if second_norm <= 1.0e-8:
        return None
    second /= second_norm
    return normal, first, second

def _v14_unique_foreground_voxels(mask_mm: np.ndarray, coords_tyx: np.ndarray) -> np.ndarray:
    rounded, valid, values = _v14_round_and_sample_mask(mask_mm, coords_tyx)
    keep = valid & values
    if not np.any(keep):
        return np.empty((0, 3), dtype=np.int32)
    return np.unique(np.asarray(rounded[keep], dtype=np.int32), axis=0)

def _v14_bound_evidence_voxels(voxels: np.ndarray) -> np.ndarray:
    """Bound one audit payload without changing its anomaly classification."""
    array = np.asarray(voxels, dtype=np.int32).reshape((-1, 3))
    cap = max(128, _env_int('YOLO_TTA_CENTERLINE_MAX_EVIDENCE_VOXELS', 2048))
    if int(array.shape[0]) <= int(cap):
        return array
    select = np.linspace(0, int(array.shape[0]) - 1, int(cap), dtype=np.int64)
    return np.asarray(array[select], dtype=np.int32)

def _v14_sample_one_normal_section(
    mask_mm: np.ndarray,
    samples: V14CenterlineSamples,
    sample_index: int,
    radius_factor: float,
    plane_half_cap: int,
    capture_payload: bool = True,
) -> Tuple[str, Optional[V14SectionEvidence]]:
    """Return ``(state, evidence)`` where state is anomaly, clean, or unknown.

 Unknown is deliberately distinct from a tested clean section: an invalid
 tangent, clipped plane, failed center snap, or safety cap may never serve as
 a clean longitudinal flank that authorizes deletion."""
    idx = int(sample_index)
    radius = float(samples.radii[idx])
    if not np.isfinite(radius) or radius < 0.5:
        return 'unknown', None
    basis = _v14_plane_basis(samples.tangents_tyx[idx])
    if basis is None:
        return 'unknown', None
    _normal, first, second = basis
    center = _v14_snap_center_to_foreground(mask_mm, samples.points_tyx[idx], radius)
    if center is None:
        return 'unknown', None
    expanded_radius = float(radius_factor) * float(radius)
    if not np.isfinite(expanded_radius) or expanded_radius <= radius:
        return 'unknown', None

    ring_count = max(32, min(2048, int(math.ceil(2.0 * math.pi * expanded_radius))))
    angles = np.linspace(0.0, 2.0 * math.pi, int(ring_count), endpoint=False, dtype=np.float64)
    ring_coords = (
        center.reshape(1, 3)
        + expanded_radius
        * (
            np.cos(angles).reshape(-1, 1) * first.reshape(1, 3)
            + np.sin(angles).reshape(-1, 1) * second.reshape(1, 3)
        )
    )
    ring_rounded, ring_valid, ring_values = _v14_round_and_sample_mask(mask_mm, ring_coords)
    hit_indices = np.flatnonzero(ring_valid & ring_values)
    ring_complete = bool(np.all(ring_valid))
    if int(hit_indices.size) <= 0:
        return ('clean', None) if ring_complete else ('unknown', None)

    # Fast path: check a bounded, angularly distributed set of straight paths in
    # this exact plane. This is the user's proposed line test and avoids the plane
    # raster/connected-component allocation for the common case.
    if int(hit_indices.size) > 24:
        choose = np.linspace(0, int(hit_indices.size) - 1, 24, dtype=np.int64)
        path_hit_indices = hit_indices[choose]
    else:
        path_hit_indices = hit_indices
    evidence_coords: List[np.ndarray] = []
    path_step = 0.75
    path_count = max(2, int(math.ceil(expanded_radius / path_step)) + 1)
    alpha = np.linspace(0.0, 1.0, int(path_count), dtype=np.float64)
    outer_threshold = max(float(radius) + 0.75, 1.10 * float(radius))
    for hit_idx in path_hit_indices:
        endpoint = ring_coords[int(hit_idx)]
        line = center.reshape(1, 3) + alpha.reshape(-1, 1) * (endpoint - center).reshape(1, 3)
        _rounded, valid, values = _v14_round_and_sample_mask(mask_mm, line)
        if bool(np.all(valid & values)):
            distances = alpha * expanded_radius
            outer = line[distances >= outer_threshold]
            if int(outer.shape[0]) > 0:
                if not bool(capture_payload):
                    return 'anomaly', V14SectionEvidence(
                        idx, int(samples.branch_ids[idx]), int(samples.arc_indices[idx]),
                        tuple(float(v) for v in center), float(radius),
                        np.empty((0, 3), dtype=np.int32), False,
                    )
                evidence_coords.append(outer)
    if evidence_coords:
        voxels = _v14_unique_foreground_voxels(mask_mm, np.concatenate(evidence_coords, axis=0))
        if int(voxels.shape[0]) > 0:
            return 'anomaly', V14SectionEvidence(
                idx, int(samples.branch_ids[idx]), int(samples.arc_indices[idx]),
                tuple(float(v) for v in center), float(radius),
                _v14_bound_evidence_voxels(voxels), False,
            )

    # Curved-path fallback: rasterize only the tangent-normal plane, label it in
    # 2D, and require the center's foreground component to reach the expanded
    # circle. No equivalent-radius sphere is sampled anywhere in this test.
    half = int(math.ceil(expanded_radius + 1.5))
    if half > int(plane_half_cap):
        return 'unknown', None
    offsets = np.arange(-int(half), int(half) + 1, dtype=np.float64)
    vv, uu = np.meshgrid(offsets, offsets, indexing='ij')
    plane_coords = (
        center.reshape(1, 1, 3)
        + uu[..., None] * first.reshape(1, 1, 3)
        + vv[..., None] * second.reshape(1, 1, 3)
    )
    _plane_rounded, plane_valid, plane_values = _v14_round_and_sample_mask(
        mask_mm, plane_coords.reshape((-1, 3)),
    )
    plane_complete = bool(np.all(plane_valid))
    plane_fg = (plane_valid & plane_values).reshape(plane_coords.shape[:2]).astype(np.uint8)
    center_rc = (int(half), int(half))
    if int(plane_fg[center_rc]) == 0:
        local_fg = np.argwhere(plane_fg != 0)
        if int(local_fg.shape[0]) <= 0:
            return 'unknown', None
        center_dist2 = np.sum((local_fg - np.asarray(center_rc).reshape(1, 2)) ** 2, axis=1)
        nearest_i = int(np.argmin(center_dist2))
        if float(center_dist2[nearest_i]) > 9.0:
            return 'unknown', None
        center_rc = (int(local_fg[nearest_i, 0]), int(local_fg[nearest_i, 1]))
    label_count, labels = cv2.connectedComponents(plane_fg, connectivity=8, ltype=cv2.CV_32S)
    center_label = int(labels[center_rc])
    if int(label_count) <= 1 or center_label <= 0:
        return ('clean', None) if plane_complete else ('unknown', None)
    radial = np.sqrt(uu * uu + vv * vv)
    annulus = np.abs(radial - expanded_radius) <= 1.25
    center_component = labels == int(center_label)
    if not np.any(center_component & annulus):
        return ('clean', None) if plane_complete else ('unknown', None)
    if not bool(capture_payload):
        return 'anomaly', V14SectionEvidence(
            idx, int(samples.branch_ids[idx]), int(samples.arc_indices[idx]),
            tuple(float(v) for v in center), float(radius),
            np.empty((0, 3), dtype=np.int32), True,
        )
    evidence_grid = center_component & (radial >= outer_threshold)
    evidence_plane_coords = plane_coords[evidence_grid]
    evidence_coordinate_cap = max(
        512,
        4 * _env_int('YOLO_TTA_CENTERLINE_MAX_EVIDENCE_VOXELS', 2048),
    )
    if int(evidence_plane_coords.shape[0]) > int(evidence_coordinate_cap):
        select = np.linspace(
            0, int(evidence_plane_coords.shape[0]) - 1,
            int(evidence_coordinate_cap), dtype=np.int64,
        )
        evidence_plane_coords = evidence_plane_coords[select]
    voxels = _v14_unique_foreground_voxels(mask_mm, evidence_plane_coords)
    if int(voxels.shape[0]) <= 0:
        return 'unknown', None
    return 'anomaly', V14SectionEvidence(
        idx, int(samples.branch_ids[idx]), int(samples.arc_indices[idx]),
        tuple(float(v) for v in center), float(radius),
        _v14_bound_evidence_voxels(voxels), True,
    )

def _v14_detect_normal_plane_evidence(
    mask_mm: np.ndarray,
    samples: V14CenterlineSamples,
    *,
    radius_factor: float,
    workers: int,
    capture_payload: bool = True,
) -> List[V14SectionEvidence]:
    count = int(samples.points_tyx.shape[0])
    if count <= 0:
        return []
    plane_cap = max(32, _env_int('YOLO_TTA_CENTERLINE_PLANE_HALF_CAP', 256))
    worker_count = max(1, min(8, choose_slice_parallel_workers(int(workers), int(count))))

    def _sample(idx: int) -> Tuple[str, Optional[V14SectionEvidence]]:
        return _v14_sample_one_normal_section(
            mask_mm, samples, int(idx), float(radius_factor), int(plane_cap),
            capture_payload=bool(capture_payload),
        )

    print(
        f'v14 centerline: {"testing" if bool(capture_payload) else "classifying"} '
        f'{int(count)} tangent-normal 2D section(s) '
        f'with {int(worker_count)} worker(s), radius factor X={float(radius_factor):g}.'
    )
    if int(worker_count) <= 1:
        results = [_sample(i) for i in range(int(count))]
    else:
        pool = _acquire_parallel_pool(int(worker_count))
        try:
            results = list(pool.map(_sample, range(int(count))))
        finally:
            _release_parallel_pool(int(worker_count), pool)
    tested_indices = [
        int(idx) for idx, (state, _item) in enumerate(results)
        if str(state) in {'clean', 'anomaly'}
    ]
    unknown_indices = [
        int(idx) for idx, (state, _item) in enumerate(results)
        if str(state) == 'unknown'
    ]
    # ``details`` is intentionally a mutable audit dictionary even though the
    # sample arrays are held in a frozen dataclass.
    samples.details['tested_section_sample_indices'] = tested_indices
    samples.details['unknown_section_sample_indices'] = unknown_indices
    return [item for state, item in results if str(state) == 'anomaly' and item is not None]

def _v14_materialize_selected_section_evidence(
    mask_mm: np.ndarray,
    samples: V14CenterlineSamples,
    selected_stubs: Sequence[V14SectionEvidence],
    *,
    radius_factor: float,
    workers: int,
) -> Tuple[List[V14SectionEvidence], Dict[str, object]]:
    """Materialize bounded voxel payloads only for clean-flank anomaly samples."""
    requested = list(selected_stubs)
    sample_cap = max(
        64,
        _env_int('YOLO_TTA_CENTERLINE_MAX_ACTIONABLE_EVIDENCE_SAMPLES', 2048),
    )
    sample_cap_reached = bool(len(requested) > int(sample_cap))
    if bool(sample_cap_reached):
        choose = np.linspace(0, len(requested) - 1, int(sample_cap), dtype=np.int64)
        requested = [requested[int(index)] for index in choose]
    if not requested:
        return [], {
            'requested_actionable_evidence_samples': int(len(selected_stubs)),
            'materialized_actionable_evidence_samples': 0,
            'materialized_actionable_evidence_voxels': 0,
            'actionable_evidence_sample_cap_reached': bool(sample_cap_reached),
            'actionable_evidence_voxel_cap_reached': False,
        }

    plane_cap = max(32, _env_int('YOLO_TTA_CENTERLINE_PLANE_HALF_CAP', 256))
    worker_count = max(1, min(8, choose_slice_parallel_workers(int(workers), len(requested))))

    def _sample(stub: V14SectionEvidence) -> Tuple[str, Optional[V14SectionEvidence]]:
        return _v14_sample_one_normal_section(
            mask_mm, samples, int(stub.sample_index), float(radius_factor), int(plane_cap),
            capture_payload=True,
        )

    print(
        f'v14 centerline: materializing bounded evidence for {len(requested)} '
        f'clean-flank section(s) with {int(worker_count)} worker(s).'
    )
    if int(worker_count) <= 1:
        results = [_sample(stub) for stub in requested]
    else:
        pool = _acquire_parallel_pool(int(worker_count))
        try:
            results = list(pool.map(_sample, requested))
        finally:
            _release_parallel_pool(int(worker_count), pool)

    voxel_cap = max(
        4096,
        _env_int('YOLO_TTA_CENTERLINE_MAX_ACTIONABLE_EVIDENCE_VOXELS', 1_000_000),
    )
    materialized: List[V14SectionEvidence] = []
    retained_voxels = 0
    voxel_cap_reached = False
    for stub, (state, item) in zip(requested, results):
        if str(state) != 'anomaly' or item is None:
            continue
        voxels = np.asarray(item.voxel_tyx, dtype=np.int32).reshape((-1, 3))
        remaining = int(voxel_cap) - int(retained_voxels)
        if int(remaining) <= 0:
            voxel_cap_reached = True
            break
        if int(voxels.shape[0]) > int(remaining):
            voxel_cap_reached = True
            select = np.linspace(0, int(voxels.shape[0]) - 1, int(remaining), dtype=np.int64)
            voxels = np.asarray(voxels[select], dtype=np.int32)
        if int(voxels.shape[0]) <= 0:
            continue
        retained_voxels += int(voxels.shape[0])
        materialized.append(dataclasses_replace(
            item,
            voxel_tyx=voxels,
            event_id=int(stub.event_id),
        ))
    return materialized, {
        'requested_actionable_evidence_samples': int(len(selected_stubs)),
        'materialized_actionable_evidence_samples': int(len(materialized)),
        'materialized_actionable_evidence_voxels': int(retained_voxels),
        'actionable_evidence_sample_cap_reached': bool(sample_cap_reached),
        'actionable_evidence_voxel_cap_reached': bool(voxel_cap_reached),
    }

def _v14_cluster_centerline_events(
    samples: V14CenterlineSamples,
    evidence: Sequence[V14SectionEvidence],
    *,
    minimum_samples: int = 3,
    close_gap: int = 2,
    clean_flank_samples: int = 3,
) -> Tuple[List[V14CenterlineEvent], List[V14SectionEvidence]]:
    evidence_by_sample: Dict[int, V14SectionEvidence] = {
        int(item.sample_index): item for item in evidence
    }
    tested_raw = samples.details.get('tested_section_sample_indices')
    if tested_raw is None:
        # Conservative behavior for callers that did not run the tri-state
        # detector: evidence is tested-positive; all other samples are unknown.
        tested_samples = set(int(v) for v in evidence_by_sample)
    else:
        tested_samples = set(int(v) for v in tested_raw)
    events: List[V14CenterlineEvent] = []
    selected: List[V14SectionEvidence] = []
    next_event_id = 0
    for branch_id in np.unique(samples.branch_ids):
        branch_idx = np.flatnonzero(samples.branch_ids == int(branch_id))
        if int(branch_idx.size) <= 0:
            continue
        order = branch_idx[np.argsort(samples.arc_indices[branch_idx], kind='stable')]
        anomaly = np.asarray(
            [int(idx) in evidence_by_sample for idx in order], dtype=bool,
        )
        tested_clean = np.asarray(
            [int(idx) in tested_samples and int(idx) not in evidence_by_sample for idx in order],
            dtype=bool,
        )
        # Close only tiny longitudinal gaps; there is intentionally no maximum
        # anomaly/event duration, so a webbed run spanning hundreds of slices remains one event.
        closed = anomaly.copy()
        pos = 0
        while pos < int(closed.size):
            if bool(closed[pos]):
                pos += 1
                continue
            start = int(pos)
            while pos < int(closed.size) and not bool(closed[pos]):
                pos += 1
            stop = int(pos)
            if (
                start > 0 and stop < int(closed.size)
                and int(stop - start) <= int(close_gap)
                and bool(closed[start - 1]) and bool(closed[stop])
                and bool(np.all(tested_clean[int(start):int(stop)]))
            ):
                closed[start:stop] = True
        pos = 0
        while pos < int(closed.size):
            if not bool(closed[pos]):
                pos += 1
                continue
            run_start = int(pos)
            while pos < int(closed.size) and bool(closed[pos]):
                pos += 1
            run_stop = int(pos)
            actual = [
                evidence_by_sample[int(order[k])]
                for k in range(int(run_start), int(run_stop))
                if int(order[k]) in evidence_by_sample
            ]
            if len(actual) < int(minimum_samples):
                continue
            clean_left = bool(
                int(run_start) >= int(clean_flank_samples)
                and np.all(tested_clean[int(run_start) - int(clean_flank_samples):int(run_start)])
            )
            clean_right = bool(
                int(run_stop) + int(clean_flank_samples) <= int(anomaly.size)
                and np.all(tested_clean[int(run_stop):int(run_stop) + int(clean_flank_samples)])
            )
            voxel_blocks = [item.voxel_tyx for item in actual if int(item.voxel_tyx.shape[0]) > 0]
            if voxel_blocks:
                all_voxels = np.concatenate(voxel_blocks, axis=0)
                min_t = int(np.min(all_voxels[:, 0]))
                max_t = int(np.max(all_voxels[:, 0]))
            else:
                center_t = np.asarray([item.center_tyx[0] for item in actual], dtype=np.float64)
                min_t = int(math.floor(float(np.min(center_t))))
                max_t = int(math.ceil(float(np.max(center_t))))
            event_selected_indices: List[int] = []
            # The requested failure pattern is bracketed by reliable anatomy.
            # Unbracketed runs remain in event statistics but are too ambiguous
            # to produce a watershed/deletion proposal (volume ends, uncovered
            # branches, and persistently eccentric valid anatomy are common).
            if bool(clean_left and clean_right):
                for item in actual:
                    event_selected_indices.append(len(selected))
                    selected.append(dataclasses_replace(item, event_id=int(next_event_id)))
            events.append(V14CenterlineEvent(
                int(next_event_id), int(branch_id),
                int(samples.arc_indices[int(order[run_start])]),
                int(samples.arc_indices[int(order[run_stop - 1])]),
                tuple(int(v) for v in event_selected_indices),
                bool(clean_left), bool(clean_right),
                int(min_t), int(max_t),
            ))
            next_event_id += 1
    return events, selected

def _v14_dense_centerline_seeds_by_slice(
    mask_mm: np.ndarray,
    samples: V14CenterlineSamples,
) -> Dict[int, List[Tuple[int, int, float]]]:
    seeds: Dict[int, Dict[Tuple[int, int], float]] = {}

    def _add_protection_seeds(point_tyx: np.ndarray, radius: float) -> None:
        """Rasterize a bounded swept tube used only as a do-not-delete guard.

 Anomaly evidence is still obtained exclusively from the tangent-normal
 2D circle/path test. This small 3D neighborhood is deliberately allowed
 only to make protection more conservative for oblique centerlines whose
 central voxel falls on a neighboring source slice."""
        point = np.asarray(point_tyx, dtype=np.float64).reshape(3)
        protect_radius = max(1.0, float(radius))
        t_reach = min(8, int(math.ceil(protect_radius)))
        center_t = int(round(float(point[0])))
        for t_idx in range(int(center_t) - int(t_reach), int(center_t) + int(t_reach) + 1):
            if t_idx < 0 or t_idx >= int(mask_mm.shape[0]):
                continue
            dt = abs(float(t_idx) - float(point[0]))
            inplane = math.sqrt(max(0.0, protect_radius * protect_radius - dt * dt))
            if inplane < 0.5:
                continue
            center_y = int(round(float(point[1])))
            center_x = int(round(float(point[2])))
            selected: Optional[Tuple[int, int]] = None
            if (
                0 <= center_y < int(mask_mm.shape[1])
                and 0 <= center_x < int(mask_mm.shape[2])
                and int(mask_mm[int(t_idx), int(center_y), int(center_x)]) != 0
            ):
                selected = (int(center_y), int(center_x))
            else:
                reach = min(12, max(1, int(math.ceil(inplane))))
                y0 = max(0, int(center_y) - int(reach))
                y1 = min(int(mask_mm.shape[1]), int(center_y) + int(reach) + 1)
                x0 = max(0, int(center_x) - int(reach))
                x1 = min(int(mask_mm.shape[2]), int(center_x) + int(reach) + 1)
                local = np.argwhere(np.asarray(mask_mm[int(t_idx), y0:y1, x0:x1]) != 0)
                if int(local.shape[0]) > 0:
                    global_y = local[:, 0].astype(np.float64) + float(y0)
                    global_x = local[:, 1].astype(np.float64) + float(x0)
                    dist2 = (global_y - float(point[1])) ** 2 + (global_x - float(point[2])) ** 2
                    best = int(np.argmin(dist2))
                    if float(dist2[best]) <= float(inplane * inplane):
                        selected = (int(round(global_y[best])), int(round(global_x[best])))
            if selected is not None:
                per_slice = seeds.setdefault(int(t_idx), {})
                per_slice[selected] = max(
                    float(radius), float(per_slice.get(selected, 0.0)),
                )

    for branch_id in np.unique(samples.branch_ids):
        branch_idx = np.flatnonzero(samples.branch_ids == int(branch_id))
        if int(branch_idx.size) <= 0:
            continue
        order = branch_idx[np.argsort(samples.arc_indices[branch_idx], kind='stable')]
        for local_pos in range(int(order.size)):
            idx0 = int(order[local_pos])
            idx1 = int(order[min(int(order.size) - 1, local_pos + 1)])
            point0 = np.asarray(samples.points_tyx[idx0], dtype=np.float64)
            point1 = np.asarray(samples.points_tyx[idx1], dtype=np.float64)
            radius0 = float(samples.radii[idx0])
            radius1 = float(samples.radii[idx1])
            delta = point1 - point0
            distance = float(np.linalg.norm(delta))
            if distance > max(32.0, 10.0 * max(radius0, radius1, 1.0)):
                steps = 1
            else:
                steps = max(1, int(math.ceil(distance / 0.5)))
            for step in range(int(steps) + 1):
                alpha = float(step) / float(max(1, steps))
                point = point0 + alpha * delta
                radius = (1.0 - alpha) * radius0 + alpha * radius1
                snapped = _v14_snap_center_to_foreground(mask_mm, point, radius)
                if snapped is None:
                    continue
                _add_protection_seeds(snapped, float(radius))
    return {
        int(t_idx): [(int(y), int(x), float(radius)) for (y, x), radius in sorted(per_slice.items())]
        for t_idx, per_slice in seeds.items()
    }

def _v14_watershed_candidate_basin(
    component_u8: np.ndarray,
    good_seeds_yxr: Sequence[Tuple[int, int, float]],
    bad_seeds_yx: Sequence[Tuple[int, int]],
) -> np.ndarray:
    component = np.ascontiguousarray(np.asarray(component_u8) != 0, dtype=np.uint8)
    if not np.any(component):
        return np.zeros_like(component, dtype=np.uint8)
    good = np.zeros_like(component, dtype=np.uint8)
    for y, x, radius in good_seeds_yxr:
        if 0 <= int(y) < int(component.shape[0]) and 0 <= int(x) < int(component.shape[1]):
            seed_radius = max(1, min(8, int(round(0.25 * max(1.0, float(radius))))))
            cv2.circle(good, (int(x), int(y)), int(seed_radius), 1, thickness=-1)
    good &= component
    bad = np.zeros_like(component, dtype=np.uint8)
    for y, x in bad_seeds_yx:
        if 0 <= int(y) < int(component.shape[0]) and 0 <= int(x) < int(component.shape[1]):
            bad[int(y), int(x)] = np.uint8(1)
    if np.any(bad):
        bad = cv2.dilate(bad, np.ones((3, 3), dtype=np.uint8), iterations=1)
    bad &= component
    bad[good != 0] = np.uint8(0)
    fallback = np.ascontiguousarray(bad != 0, dtype=np.uint8)
    if not np.any(bad):
        return fallback
    if not np.any(good):
        # With no trusted centerline seed there is no defensible watershed
        # partition. Mark the complete suspect 2D component so audit-first output
        # is actionable instead of emitting a few seed dots.
        return np.ascontiguousarray(component != 0, dtype=np.uint8)
    dist = cv2.distanceTransform(component, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_dist = float(np.max(dist))
    if max_dist <= 0.0:
        return fallback
    topography = np.uint8(np.clip(255.0 - 254.0 * dist / max_dist, 0.0, 255.0))
    topography[component == 0] = np.uint8(255)
    markers = np.zeros(component.shape, dtype=np.int32)
    markers[good != 0] = np.int32(1)
    markers[bad != 0] = np.int32(2)
    # watershed_ift has no mask argument. Pre-seeding all background as a third
    # region prevents good/bad fronts from taking a shortcut outside the component.
    markers[component == 0] = np.int32(3)
    try:
        watershed = ndi.watershed_ift(
            topography, markers, structure=np.ones((3, 3), dtype=np.uint8),
        )
        artifact = np.ascontiguousarray((watershed == 2) & (component != 0), dtype=np.uint8)
        if np.any(artifact[good != 0]):
            return fallback
        if np.any(artifact):
            return artifact
    except Exception:
        pass
    return fallback

def _v14_submit_sparse_audit_layer(
    *,
    store_dir: Path,
    store_stats: Dict[str, object],
    shape_tyx: Tuple[int, int, int],
    model_name: str,
    mask_kind: str,
    pass_index: int,
    description: str,
) -> 'NrrdLayerRef':
    key = _nrrd_layer_key(
        view_name='global', source='centerline_filter', mask_kind=str(mask_kind),
        pass_index=int(pass_index), stage='',
    )
    extent = _coerce_segment_extent(store_stats.get('segment_extent_ijk')) or _nrrd_empty_segment_extent()
    is_removed_delta = str(mask_kind) == 'removed_components'
    ref = NrrdLayerRef(
        key=str(key),
        name=_nrrd_layer_name(
            view=None, source='centerline_filter', mask_kind=str(mask_kind),
            pass_index=int(pass_index), stage='',
        ),
        path=Path(store_dir),
        shape=tuple(int(v) for v in shape_tyx),
        dtype='uint8', storage_format=CVOL_FORMAT,
        model_name=str(model_name), view_name='global', view_family='global',
        source='centerline_filter', mask_kind=str(mask_kind),
        pass_index=int(pass_index), stage='', description=str(description),
        layer_role='subtractive_delta' if is_removed_delta else 'marker_only',
        recomposition_op='subtract_from_previous_checkpoint' if is_removed_delta else 'none',
        # Max-pool downbinning does not commute with subtraction, so these mirrors are
        # diagnostic-only at low quality. They are nevertheless emitted so every requested
        # downbin contains matching removed-components and watershed-candidate audit layers.
        low_quality_recomposition_op='diagnostic_only' if is_removed_delta else 'none',
        mirror_low_quality=True,
        segment_extent_ijk=extent,
        segment_extent_shape_tyx=tuple(int(v) for v in shape_tyx),
        segment_extent_source='incremental_centerline_audit_cvol_index',
    )
    sink = nrrd_layer_sink()
    if sink is None:
        raise RuntimeError('v14 centerline audit layer requested without an NRRD sink')
    sink.submit_layer(
        ref,
        nrrd_layer_output_suffix(
            view_token='Global', source='centerline_filter', mask_kind=str(mask_kind),
            pass_index=int(pass_index), stage='',
        ),
    )
    return ref

def _v14_component_flank_is_clear(
    mask_mm: np.ndarray,
    component_crop: np.ndarray,
    bbox_yxyx: Tuple[int, int, int, int],
    event: V14CenterlineEvent,
    temporal_context: int,
) -> bool:
    y0, y1, x0, x1 = (int(v) for v in bbox_yxyx)
    context = max(0, int(temporal_context))
    if int(context) <= 0:
        return True
    left_start = int(event.min_t) - int(context)
    left_stop = int(event.min_t)
    right_start = int(event.max_t) + 1
    right_stop = int(event.max_t) + int(context) + 1
    if left_start < 0 or right_stop > int(mask_mm.shape[0]):
        return False
    component = np.asarray(component_crop) != 0
    area = max(1, int(np.count_nonzero(component)))
    # Permit modest source-slice motion while testing continuity. Dilating the
    # candidate support makes the guard conservative: nearby anatomy in *any* of
    # the requested clean slices prevents deletion.
    default_motion = max(2, min(8, int(math.ceil(math.sqrt(float(context))))))
    motion = max(0, _env_int('YOLO_TTA_CENTERLINE_FLANK_MOTION_PX', default_motion))
    expanded_y0 = max(0, int(y0) - int(motion))
    expanded_y1 = min(int(mask_mm.shape[1]), int(y1) + int(motion))
    expanded_x0 = max(0, int(x0) - int(motion))
    expanded_x1 = min(int(mask_mm.shape[2]), int(x1) + int(motion))
    support = np.zeros(
        (int(expanded_y1 - expanded_y0), int(expanded_x1 - expanded_x0)), dtype=np.uint8,
    )
    support[
        int(y0 - expanded_y0):int(y1 - expanded_y0),
        int(x0 - expanded_x0):int(x1 - expanded_x0),
    ] = np.asarray(component, dtype=np.uint8)
    if int(motion) > 0:
        size = 2 * int(motion) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
        support = cv2.dilate(support, kernel, iterations=1)
    for flank_t in list(range(int(left_start), int(left_stop))) + list(range(int(right_start), int(right_stop))):
        flank = np.ascontiguousarray(
            np.asarray(mask_mm[
                int(flank_t), int(expanded_y0):int(expanded_y1), int(expanded_x0):int(expanded_x1)
            ]) != 0,
            dtype=np.uint8,
        )
        overlap = int(np.count_nonzero((support != 0) & (flank != 0)))
        if float(overlap) / float(area) > 0.01:
            return False
    return True

def _v14_apply_component_removal_plan(
    mask_mm: np.ndarray,
    anchors_by_t: Dict[int, List[Tuple[int, int]]],
    *,
    workers: int,
) -> int:
    if not anchors_by_t:
        return 0
    slice_ids = np.asarray(sorted(int(v) for v in anchors_by_t), dtype=np.int64)
    removed_counts = np.zeros((int(slice_ids.size),), dtype=np.int64)

    def _apply(local_index: int) -> None:
        t_idx = int(slice_ids[int(local_index)])
        plane = np.asarray(mask_mm[int(t_idx)], dtype=np.uint8)
        x0, y0, width, height = (int(v) for v in cv2.boundingRect(plane))
        if width <= 0 or height <= 0:
            return
        crop = np.ascontiguousarray(
            plane[int(y0):int(y0) + int(height), int(x0):int(x0) + int(width)] != 0,
            dtype=np.uint8,
        )
        _count, labels = cv2.connectedComponents(crop, connectivity=8, ltype=cv2.CV_32S)
        label_ids: set[int] = set()
        for anchor_y, anchor_x in anchors_by_t[int(t_idx)]:
            local_y = int(anchor_y) - int(y0)
            local_x = int(anchor_x) - int(x0)
            if 0 <= local_y < int(labels.shape[0]) and 0 <= local_x < int(labels.shape[1]):
                label_id = int(labels[local_y, local_x])
                if label_id > 0:
                    label_ids.add(int(label_id))
        if not label_ids:
            return
        remove_mask = np.zeros(labels.shape, dtype=bool)
        for label_id in sorted(label_ids):
            remove_mask |= labels == int(label_id)
        removed_counts[int(local_index)] = np.int64(np.count_nonzero(remove_mask))
        dst_crop = plane[int(y0):int(y0) + int(height), int(x0):int(x0) + int(width)]
        dst_crop[remove_mask] = np.uint8(0)

    parallel_for_indices(
        int(slice_ids.size), _apply,
        max_workers=choose_slice_parallel_workers(int(workers), int(slice_ids.size)),
        desc='v14 centerline: applying safe 2D component removals',
        show_progress=True,
    )
    flush_array(mask_mm)
    return int(np.sum(removed_counts, dtype=np.int64))

def _v14_plan_components_and_write_sparse_audits(
    mask_mm: np.ndarray,
    samples: V14CenterlineSamples,
    events: Sequence[V14CenterlineEvent],
    evidence: Sequence[V14SectionEvidence],
    center_seeds_by_t: Dict[int, List[Tuple[int, int, float]]],
    *,
    temp_dir: Path,
    model_name: str,
    pass_index: int,
    temporal_context: int,
    automatic_removal_enabled: bool,
    workers: int,
    nrrd_layer_refs: List['NrrdLayerRef'],
) -> Dict[str, object]:
    shape = tuple(int(v) for v in mask_mm.shape)
    pass_dir = temp_dir / 'centerline_filter' / f'pass{int(pass_index):02d}'
    removed_store_dir = pass_dir / 'removed_components.cvol'
    watershed_store_dir = pass_dir / 'watershed_candidates.cvol'
    removed_writer = IncrementalRawBBoxMaskStoreWriter(
        shape=shape, store_dir=removed_store_dir, format_name=CVOL_FORMAT,
        desc=f'v14 pass {int(pass_index)} removed 2D components',
        extra_meta={'source': 'centerline_filter', 'pass_index': int(pass_index)},
    )
    watershed_writer = IncrementalRawBBoxMaskStoreWriter(
        shape=shape, store_dir=watershed_store_dir, format_name=CVOL_FORMAT,
        desc=f'v14 pass {int(pass_index)} watershed candidates',
        extra_meta={'source': 'centerline_filter', 'pass_index': int(pass_index)},
    )
    event_by_id = {int(event.event_id): event for event in events}
    evidence_by_t: Dict[int, List[Tuple[V14SectionEvidence, int, int]]] = {}
    for item in evidence:
        for voxel in np.asarray(item.voxel_tyx, dtype=np.int32):
            t_idx, y_idx, x_idx = (int(v) for v in voxel)
            if 0 <= t_idx < int(shape[0]):
                evidence_by_t.setdefault(int(t_idx), []).append((item, int(y_idx), int(x_idx)))
    candidate_slices = sorted(int(v) for v in evidence_by_t)
    anchors_by_t: Dict[int, List[Tuple[int, int]]] = {}
    candidate_components = 0
    protected_components = 0
    deletion_plans = 0
    marker_only_components = 0
    min_component_area = max(1, _env_int('YOLO_TTA_CENTERLINE_MIN_COMPONENT_AREA', 32))
    min_evidence_coverage = min(
        1.0, max(0.0, _env_float('YOLO_TTA_CENTERLINE_MIN_EVIDENCE_COVERAGE', 0.25)),
    )
    evidence_only_markers = bool(
        samples.details.get('section_reliability_guard_triggered', False)
        or samples.details.get('actionable_evidence_cap_reached', False)
    )
    cursor = 0
    try:
        for t_idx in candidate_slices:
            if int(t_idx) > int(cursor):
                removed_writer.consume_empty_range(int(cursor), int(t_idx) - int(cursor))
                watershed_writer.consume_empty_range(int(cursor), int(t_idx) - int(cursor))
            removed_plane = np.zeros((int(shape[1]), int(shape[2])), dtype=np.uint8)
            watershed_plane = np.zeros_like(removed_plane)
            plane = np.asarray(mask_mm[int(t_idx)], dtype=np.uint8)
            x0, y0, width, height = (int(v) for v in cv2.boundingRect(plane))
            if width > 0 and height > 0:
                crop = np.ascontiguousarray(
                    plane[int(y0):int(y0) + int(height), int(x0):int(x0) + int(width)] != 0,
                    dtype=np.uint8,
                )
                count, labels, cc_stats, _centroids = cv2.connectedComponentsWithStats(
                    crop, connectivity=8, ltype=cv2.CV_32S,
                )
                bad_by_label: Dict[int, List[Tuple[V14SectionEvidence, int, int]]] = {}
                for item, global_y, global_x in evidence_by_t[int(t_idx)]:
                    local_y = int(global_y) - int(y0)
                    local_x = int(global_x) - int(x0)
                    if 0 <= local_y < int(labels.shape[0]) and 0 <= local_x < int(labels.shape[1]):
                        label_id = int(labels[local_y, local_x])
                        if label_id > 0:
                            bad_by_label.setdefault(int(label_id), []).append(
                                (item, int(local_y), int(local_x)),
                            )
                good_by_label: Dict[int, List[Tuple[int, int, float]]] = {}
                for global_y, global_x, radius in center_seeds_by_t.get(int(t_idx), []):
                    local_y = int(global_y) - int(y0)
                    local_x = int(global_x) - int(x0)
                    if 0 <= local_y < int(labels.shape[0]) and 0 <= local_x < int(labels.shape[1]):
                        label_id = int(labels[local_y, local_x])
                        if label_id > 0:
                            good_by_label.setdefault(int(label_id), []).append(
                                (int(local_y), int(local_x), float(radius)),
                            )

                for label_id, bad_items in sorted(bad_by_label.items()):
                    if int(label_id) <= 0 or int(label_id) >= int(count):
                        continue
                    candidate_components += 1
                    component = labels == int(label_id)
                    area = int(cc_stats[int(label_id), cv2.CC_STAT_AREA])
                    bad_yx = sorted(set((int(y), int(x)) for _item, y, x in bad_items))
                    good_yxr = good_by_label.get(int(label_id), [])
                    relevant_events = [
                        event_by_id[int(event_id)]
                        for event_id in sorted(set(int(item.event_id) for item, _y, _x in bad_items))
                        if int(event_id) in event_by_id
                    ]
                    protected = bool(good_yxr)
                    if protected:
                        protected_components += 1

                    bad_seed_mask = np.zeros(component.shape, dtype=np.uint8)
                    for bad_y, bad_x in bad_yx:
                        bad_seed_mask[int(bad_y), int(bad_x)] = np.uint8(1)
                    bad_coverage_mask = cv2.dilate(
                        bad_seed_mask, np.ones((5, 5), dtype=np.uint8), iterations=1,
                    )
                    evidence_coverage = float(np.count_nonzero((bad_coverage_mask != 0) & component)) / float(max(1, area))

                    can_remove = bool(
                        automatic_removal_enabled
                        and
                        samples.automatic_removal_allowed
                        and not protected
                        and int(area) >= int(min_component_area)
                        and float(evidence_coverage) >= float(min_evidence_coverage)
                        and bool(relevant_events)
                        and all(bool(event.clean_flanks) for event in relevant_events)
                    )
                    if can_remove:
                        bbox = (
                            int(y0), int(y0) + int(height),
                            int(x0), int(x0) + int(width),
                        )
                        can_remove = all(
                            _v14_component_flank_is_clear(
                                mask_mm, component, bbox, event, int(temporal_context),
                            )
                            for event in relevant_events
                        )

                    if can_remove:
                        removed_crop = removed_plane[
                            int(y0):int(y0) + int(height), int(x0):int(x0) + int(width)
                        ]
                        removed_crop[component] = np.uint8(1)
                        first_y, first_x = bad_yx[0]
                        anchors_by_t.setdefault(int(t_idx), []).append(
                            (int(y0) + int(first_y), int(x0) + int(first_x)),
                        )
                        deletion_plans += 1
                    else:
                        if bool(evidence_only_markers):
                            # A globally unreliable/capped centerline fit may
                            # still provide useful locations, but it may not
                            # expand those seeds into a broad watershed basin.
                            basin = np.ascontiguousarray(
                                (bad_seed_mask != 0) & component,
                                dtype=np.uint8,
                            )
                        else:
                            basin = _v14_watershed_candidate_basin(
                                np.ascontiguousarray(component, dtype=np.uint8),
                                good_yxr, bad_yx,
                            )
                        marker_crop = watershed_plane[
                            int(y0):int(y0) + int(height), int(x0):int(x0) + int(width)
                        ]
                        marker_crop[basin != 0] = np.uint8(1)
                        marker_only_components += 1

            removed_writer.consume(int(t_idx), removed_plane.reshape((1, int(shape[1]), int(shape[2]))))
            watershed_writer.consume(int(t_idx), watershed_plane.reshape((1, int(shape[1]), int(shape[2]))))
            cursor = int(t_idx) + 1
        if int(cursor) < int(shape[0]):
            removed_writer.consume_empty_range(int(cursor), int(shape[0]) - int(cursor))
            watershed_writer.consume_empty_range(int(cursor), int(shape[0]) - int(cursor))
        removed_stats = removed_writer.finalize()
        watershed_stats = watershed_writer.finalize()
    except BaseException:
        removed_writer.discard()
        watershed_writer.discard()
        raise

    removed_ref = _v14_submit_sparse_audit_layer(
        store_dir=removed_store_dir, store_stats=removed_stats, shape_tyx=shape,
        model_name=str(model_name), mask_kind='removed_components',
        pass_index=int(pass_index),
        description=(
            'Whole 2D source-slice components approved for removal by centerline, '
            'clean-flank, evidence-coverage, and centerline-protection guards.'
        ),
    )
    watershed_ref = _v14_submit_sparse_audit_layer(
        store_dir=watershed_store_dir, store_stats=watershed_stats, shape_tyx=shape,
        model_name=str(model_name), mask_kind='watershed_candidates',
        pass_index=int(pass_index),
        description=(
            'Marker-only artifact candidates for suspect components that contain protected '
            'centerline anatomy or do not satisfy every automatic-removal guard. A triggered '
            'backend-reliability/evidence-cap guard records bounded evidence seeds instead of '
            'expanding an unreliable watershed basin.'
        ),
    )
    nrrd_layer_refs.extend([removed_ref, watershed_ref])
    applied_removed_voxels = _v14_apply_component_removal_plan(
        mask_mm, anchors_by_t, workers=int(workers),
    )
    planned_removed_voxels = int(removed_stats.get('foreground_voxels', 0))
    if int(applied_removed_voxels) != int(planned_removed_voxels):
        raise RuntimeError(
            f'v14 centerline removal audit mismatch: planned={planned_removed_voxels}, '
            f'applied={applied_removed_voxels}'
        )
    return {
        'candidate_slices': int(len(candidate_slices)),
        'candidate_components': int(candidate_components),
        'protected_components': int(protected_components),
        'removed_components': int(deletion_plans),
        'removed_voxels': int(applied_removed_voxels),
        'watershed_components': int(marker_only_components),
        'watershed_voxels': int(watershed_stats.get('foreground_voxels', 0)),
        'minimum_component_area': int(min_component_area),
        'minimum_evidence_coverage': float(min_evidence_coverage),
        'marker_mode': (
            'bounded_evidence_seeds' if bool(evidence_only_markers)
            else 'watershed_candidate_basin'
        ),
    }

def apply_v14_centerline_filter_inplace(
    final_union_mm: np.ndarray,
    *,
    model_name: str,
    temp_dir: Path,
    passes: int,
    backend: str,
    radius_factor: float,
    temporal_context: int,
    automatic_removal_enabled: bool,
    surface_max_dim: int,
    surface_points: int,
    timeout_seconds: float,
    workers: int,
    keep_temp: bool,
    nrrd_layer_refs: List['NrrdLayerRef'],
) -> Dict[str, object]:
    """Iteratively filter the post-union volume and preserve every audit stage."""
    requested_passes = max(0, int(passes))
    stats: Dict[str, object] = {
        'enabled': bool(requested_passes > 0 and str(backend).lower() != 'off'),
        'requested_passes': int(requested_passes),
        'requested_backend': str(backend),
        'radius_factor': float(radius_factor),
        'temporal_context': int(temporal_context),
        'automatic_removal_enabled': bool(automatic_removal_enabled),
        'passes': [],
        'stop_reason': 'disabled',
    }
    if not bool(stats['enabled']):
        return stats

    print('\n=== v14 centerline filter: immutable pass-0 checkpoint ===')
    pass0_ref = materialize_nrrd_global_layer(
        final_union_mm,
        model_name=str(model_name), source='centerline_filter', mask_kind='pass00_input',
        pass_index=0, stage='',
        description=(
            'Untouched final source-geometry union before centerline filtering. '
            'This is the exact pass-0 recovery checkpoint.'
        ),
        temp_dir=temp_dir, workers=int(workers), known_has_foreground=True,
        volume_is_immutable=False, keep_temp=bool(keep_temp),
        layer_role='checkpoint', recomposition_op='select',
        low_quality_recomposition_op='select',
    )
    if pass0_ref is not None:
        nrrd_layer_refs.append(pass0_ref)

    pass_records: List[Dict[str, object]] = []
    stop_reason = 'pass_limit_exhausted'
    for pass_index in range(1, int(requested_passes) + 1):
        print(f'\n=== v14 centerline filter pass {int(pass_index)}/{int(requested_passes)} ===')
        pass_started = time.monotonic()
        samples = _v14_extract_centerline_samples(
            final_union_mm,
            temp_dir=temp_dir, pass_index=int(pass_index), backend=str(backend),
            surface_max_dim=int(surface_max_dim), surface_points=int(surface_points),
            timeout_seconds=float(timeout_seconds), workers=int(workers),
            keep_temp=bool(keep_temp),
        )
        samples = _v14_subsample_centerline_samples(samples)
        if int(samples.points_tyx.shape[0]) <= 0:
            pass_record = {
                'pass_index': int(pass_index),
                'backend': str(samples.backend),
                'backend_automatic_removal_allowed': bool(samples.automatic_removal_allowed),
                'endpoint_count': int(samples.endpoint_count),
                'centerline_samples': 0,
                'section_evidence_samples': 0,
                'longitudinal_events': 0,
                'clean_flank_events': 0,
                'candidate_slices': 0,
                'candidate_components': 0,
                'protected_components': 0,
                'removed_components': 0,
                'removed_voxels': 0,
                'watershed_components': 0,
                'watershed_voxels': 0,
                'seconds': round(float(time.monotonic() - pass_started), 3),
            }
            if samples.details.get('error'):
                pass_record['backend_error'] = str(samples.details.get('error'))
            elif samples.details.get('fallback_reason'):
                pass_record['backend_fallback_reason'] = str(samples.details.get('fallback_reason'))
            pass_records.append(pass_record)
            stop_reason = (
                'backend_failed_after_prior_pass'
                if len(pass_records) > 1 else 'backend_unavailable_pass_through'
            )
            print(
                f'v14 centerline pass {int(pass_index)}: backend={samples.backend}; '
                'no centerline samples, preserving the current pre-pass checkpoint without '
                'writing a duplicate result checkpoint.'
            )
            break
        evidence = _v14_detect_normal_plane_evidence(
            final_union_mm, samples,
            radius_factor=float(radius_factor), workers=int(workers),
            capture_payload=False,
        )
        tested_count = int(len(samples.details.get('tested_section_sample_indices', [])))
        anomaly_fraction = (
            float(len(evidence)) / float(tested_count) if int(tested_count) > 0 else 1.0
        )
        maximum_anomaly_fraction = min(
            0.95,
            max(0.05, _env_float('YOLO_TTA_CENTERLINE_MAX_ANOMALY_FRACTION', 0.35)),
        )
        reliability_guard = bool(
            int(tested_count) <= 0
            or float(anomaly_fraction) > float(maximum_anomaly_fraction)
        )
        samples.details['section_anomaly_fraction'] = float(anomaly_fraction)
        samples.details['maximum_section_anomaly_fraction'] = float(maximum_anomaly_fraction)
        samples.details['section_reliability_guard_triggered'] = bool(reliability_guard)
        if bool(reliability_guard and samples.automatic_removal_allowed):
            samples = dataclasses_replace(samples, automatic_removal_allowed=False)
            samples.details['automatic_removal_allowed'] = False
            print(
                f'Warning: v14 centerline pass {int(pass_index)} classified '
                f'{100.0 * float(anomaly_fraction):.1f}% of tested sections as anomalous; '
                'automatic removal is disabled for this pass, while clean-flank '
                'watershed candidates remain available for audit.'
            )
        events, selected_evidence = _v14_cluster_centerline_events(
            samples, evidence, minimum_samples=3, close_gap=2, clean_flank_samples=3,
        )
        selected_evidence, payload_stats = _v14_materialize_selected_section_evidence(
            final_union_mm, samples, selected_evidence,
            radius_factor=float(radius_factor), workers=int(workers),
        )
        payload_bounds: Dict[int, Tuple[int, int]] = {}
        for item in selected_evidence:
            voxels = np.asarray(item.voxel_tyx, dtype=np.int32).reshape((-1, 3))
            if int(voxels.shape[0]) <= 0 or int(item.event_id) < 0:
                continue
            item_min = int(np.min(voxels[:, 0]))
            item_max = int(np.max(voxels[:, 0]))
            previous = payload_bounds.get(int(item.event_id))
            payload_bounds[int(item.event_id)] = (
                min(int(previous[0]), int(item_min)) if previous is not None else int(item_min),
                max(int(previous[1]), int(item_max)) if previous is not None else int(item_max),
            )
        events = [
            dataclasses_replace(
                event,
                min_t=int(payload_bounds[int(event.event_id)][0]),
                max_t=int(payload_bounds[int(event.event_id)][1]),
            )
            if int(event.event_id) in payload_bounds else event
            for event in events
        ]
        payload_cap_reached = bool(
            payload_stats.get('actionable_evidence_sample_cap_reached', False)
            or payload_stats.get('actionable_evidence_voxel_cap_reached', False)
        )
        samples.details.update(payload_stats)
        samples.details['actionable_evidence_cap_reached'] = bool(payload_cap_reached)
        if bool(payload_cap_reached and samples.automatic_removal_allowed):
            samples = dataclasses_replace(samples, automatic_removal_allowed=False)
            samples.details['automatic_removal_allowed'] = False
            print(
                f'Warning: v14 centerline pass {int(pass_index)} reached the bounded '
                'actionable-evidence budget; automatic removal is disabled for this pass.'
            )
        center_seeds = _v14_dense_centerline_seeds_by_slice(final_union_mm, samples)
        component_stats = _v14_plan_components_and_write_sparse_audits(
            final_union_mm, samples, events, selected_evidence, center_seeds,
            temp_dir=temp_dir, model_name=str(model_name), pass_index=int(pass_index),
            temporal_context=int(temporal_context),
            automatic_removal_enabled=bool(automatic_removal_enabled), workers=int(workers),
            nrrd_layer_refs=nrrd_layer_refs,
        )
        result_ref = materialize_nrrd_global_layer(
            final_union_mm,
            model_name=str(model_name), source='centerline_filter', mask_kind='result',
            pass_index=int(pass_index), stage='',
            description=(
                f'Post-union result after v14 centerline filter pass {int(pass_index)}; '
                'watershed-candidate voxels remain present unless their complete 2D component '
                'separately satisfied every automatic-removal guard.'
            ),
            temp_dir=temp_dir, workers=int(workers), known_has_foreground=True,
            volume_is_immutable=False, keep_temp=bool(keep_temp),
            layer_role='checkpoint', recomposition_op='select',
            low_quality_recomposition_op='select',
        )
        if result_ref is not None:
            nrrd_layer_refs.append(result_ref)
        pass_record: Dict[str, object] = {
            'pass_index': int(pass_index),
            'backend': str(samples.backend),
            'backend_automatic_removal_allowed': bool(samples.automatic_removal_allowed),
            'endpoint_count': int(samples.endpoint_count),
            'centerline_samples': int(samples.points_tyx.shape[0]),
            'section_evidence_samples': int(len(evidence)),
            'section_anomaly_fraction': float(anomaly_fraction),
            'section_reliability_guard_triggered': bool(reliability_guard),
            'actionable_evidence_samples': int(len(selected_evidence)),
            'actionable_evidence_voxels': int(
                payload_stats.get('materialized_actionable_evidence_voxels', 0)
            ),
            'actionable_evidence_cap_reached': bool(payload_cap_reached),
            'longitudinal_events': int(len(events)),
            'clean_flank_events': int(sum(1 for event in events if event.clean_flanks)),
            **component_stats,
            'seconds': round(float(time.monotonic() - pass_started), 3),
        }
        if samples.details.get('error'):
            pass_record['backend_error'] = str(samples.details.get('error'))
        elif samples.details.get('fallback_reason'):
            pass_record['backend_fallback_reason'] = str(samples.details.get('fallback_reason'))
        pass_records.append(pass_record)
        print(
            f'v14 centerline pass {int(pass_index)}: backend={samples.backend}, '
            f'events={len(events)}, removed_components={int(component_stats["removed_components"])}, '
            f'removed_voxels={int(component_stats["removed_voxels"])}, '
            f'watershed_voxels={int(component_stats["watershed_voxels"])}.'
        )
        if int(component_stats['removed_voxels']) <= 0:
            stop_reason = (
                'backend_unavailable_pass_through'
                if int(samples.points_tyx.shape[0]) <= 0
                else 'no_safe_removals'
            )
            break
    stats['passes'] = pass_records
    stats['passes_completed'] = int(len(pass_records))
    stats['stop_reason'] = str(stop_reason)
    stats['total_removed_components'] = int(sum(int(item['removed_components']) for item in pass_records))
    stats['total_removed_voxels'] = int(sum(int(item['removed_voxels']) for item in pass_records))
    stats['total_watershed_voxels'] = int(sum(int(item['watershed_voxels']) for item in pass_records))
    return stats
