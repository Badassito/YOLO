"""Completed-view assembly, tile gates, smoothing, and output handoff."""

from __future__ import annotations

import contextlib
import math
import shutil
import threading
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
)
import numpy as np
from ._deps import cv2, ndi, tqdm

from .config import (
    GIB,
)
from .runtime import (
    _interpolation_array_backing_path,
    allocate_workspace_array,
    choose_slice_parallel_workers,
    close_memmap_array,
    close_memmap_array_without_flush,
    copy_workspace_array,
    flush_array,
    interpolate_view_volume_pass_maybe_process,
    parallel_for_indices_chunked,
    parallel_map_in_order,
    release_memfd_owners_under,
    runtime_telemetry,
    runtime_telemetry_phase,
)

# Explicit lower-layer dependencies keep imports one-way.
from .workspace import (
    _env_flag,
    _env_int,
)
from .geometry import (
    ViewInfo,
    coronal_block_cols,
    delayed_native_expansion_enabled,
    is_radial_view,
    is_tilted_radial_view,
    is_tilted_view,
    physical_view_name,
    radial_base_view_name,
    radial_sink_only_projection_supported,
    view_output_token,
    view_processing_min_radius,
    view_processing_search_angle,
)
from .inference import cleanup_view_volume_after_prediction_inplace
from .interpolation import (
    CTILE_FORMAT,
    CVOL_FORMAT,
    DeferredTilePostprocessResult,
    INTERNAL_PACKED_CVOL_FORMAT,
    IncrementalRawBBoxMaskStoreWriter,
    NrrdLayerRef,
    PreparedViewResult,
    RawBBoxMaskStore,
    RawBBoxSlicePayload,
    TileConsolidationResult,
    TileGateResult,
    TileParentGateResult,
    TilePostprocessResult,
    TilePostprocessTask,
    _coerce_segment_extent,
    _drain_volume_to_mmap,
    _encode_bool_mask_slice_payload,
    _nrrd_empty_segment_extent,
    _view_uses_interpolation,
    _write_raw_bbox_payload_store,
    write_raw_bbox_mask_store,
)
from .cuda_d1 import (
    _nrrd_layer_key,
    _nrrd_layer_name,
    _read_binary_volume_slice_crop_bool,
    _volume_has_foreground,
    _volume_shape_tuple,
    close_raw_store_or_memmap_volume,
    raw_bbox_nrrd_layers_enabled,
    subtract_volume_to_raw_bbox_store,
)
from .backprojection import (
    SinkOnlyProjectionResult,
    backproject_radial_volume_to_volume,
    backproject_tilted_volume_to_volume,
)
from .outputs import (
    compute_segment_extent_zyx,
    nrrd_layer_output_suffix,
    nrrd_layer_sink,
    nrrd_live_global_layer_enabled,
)


if TYPE_CHECKING:
    from .finalization import union_volume_into_volume

_FINAL_SOURCE_OUTPUT_SHAPE_TYX: Optional[Tuple[int, int, int]] = None

def set_final_source_output_shape(shape_tyx: Optional[Tuple[int, int, int]]) -> None:
    global _FINAL_SOURCE_OUTPUT_SHAPE_TYX
    _FINAL_SOURCE_OUTPUT_SHAPE_TYX = None if shape_tyx is None else tuple(int(v) for v in shape_tyx)

def final_source_output_shape() -> Optional[Tuple[int, int, int]]:
    return _FINAL_SOURCE_OUTPUT_SHAPE_TYX

def project_view_volume_to_orthogonal_volume(
    view_mask_mm: np.ndarray,
    view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    workers: int = 1,
    prefer_memory: bool = False,
    reserve_bytes: int = 16 * GIB,
    out_shape_tyx: Optional[Tuple[int, int, int]] = None,
    allow_transverse_passthrough: bool = False,
    known_row_occupancy: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    sink_only: bool = False,
) -> np.ndarray | SinkOnlyProjectionResult:
    """Project a Radial or Tilted view into orthogonal geometry.
    
    Eligible transverse inputs may pass through without an extra copy."""
    source_shape = tuple(int(v) for v in np.asarray(view_mask_mm).shape)
    if len(source_shape) != 3:
        raise ValueError(f'{desc}: view layer must be 3D, got {source_shape}')
    plane_h, plane_w = int(source_shape[1]), int(source_shape[2])
    reduced_processing = bool(
        delayed_native_expansion_enabled()
        and (int(plane_h), int(plane_w)) != (int(view.src_h), int(view.src_w))
    )

    # reduced Cartesian stacks are axis-permuted without expansion. Their resulting
    # orthogonal grid is smaller on exactly the two axes represented by the YOLO plane;
    # the NRRD streamer/final union performs the sole restore to output geometry.
    if reduced_processing and view.family != 'radial' and not is_tilted_view(view):
        if physical_view_name(view) == 'transverse':
            t_dim, h_dim, w_dim = int(source_shape[0]), int(plane_h), int(plane_w)
        elif physical_view_name(view) == 'sagittal':
            t_dim, h_dim, w_dim = int(plane_h), int(source_shape[0]), int(plane_w)
        elif physical_view_name(view) == 'coronal':
            t_dim, h_dim, w_dim = int(plane_h), int(plane_w), int(source_shape[0])
        else:  # pragma: no cover
            raise ValueError(f'{desc}: unsupported reduced Cartesian view {view.name!r}')
    else:
        t_dim = int(view.full_t) if int(view.full_t) > 0 else int(view_mask_mm.shape[0])
        h_dim = int(view.full_h) if int(view.full_h) > 0 else int(view.src_h)
        w_dim = int(view.full_w) if int(view.full_w) > 0 else int(view.src_w)

    if view.family == 'radial':
        return backproject_radial_volume_to_volume(
            radial_mask_mm=view_mask_mm,
            radial_view=view,
            out_path=out_path,
            desc=desc,
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
            out_shape_tyx=out_shape_tyx,
            known_row_occupancy=known_row_occupancy,
            known_slice_bboxes=known_slice_bboxes,
            projection_block_callback=projection_block_callback,
            sink_only=bool(sink_only),
        )

    if bool(sink_only):
        raise ValueError(f'{desc}: sink-only projection is supported only for radial views')

    if is_tilted_view(view):
        return backproject_tilted_volume_to_volume(
            tilted_mask_mm=view_mask_mm,
            tilted_view=view,
            out_path=out_path,
            desc=desc,
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
            out_shape_tyx=out_shape_tyx,
        )

    if out_shape_tyx is not None and tuple(int(v) for v in out_shape_tyx) != (t_dim, h_dim, w_dim):
        raise ValueError(
            f'{desc}: Cartesian view projection is a pure axis permutation; '
            f'requested output shape {tuple(out_shape_tyx)} != working {(t_dim, h_dim, w_dim)}'
        )

    if physical_view_name(view) == 'transverse':
        if tuple(int(x) for x in np.asarray(view_mask_mm).shape) != (t_dim, h_dim, w_dim):
            raise ValueError(f'{desc}: transverse layer shape {tuple(view_mask_mm.shape)} != {(t_dim, h_dim, w_dim)}')
        if bool(allow_transverse_passthrough):
            # the transverse projection is an identity copy. When the caller
            # only reads the result synchronously (raw-bbox layer encode), hand back the source
            # volume itself and skip the full-volume copy + teardown. The caller detects the
            # passthrough via np.may_share_memory and must not close/unlink it.
            return np.asarray(view_mask_mm, dtype=np.uint8)
        return copy_workspace_array(
            np.asarray(view_mask_mm, dtype=np.uint8),
            out_path,
            desc=desc,
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
        )

    out = allocate_workspace_array(
        shape=(t_dim, h_dim, w_dim),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    if physical_view_name(view) == 'sagittal':
        src = np.asarray(view_mask_mm)
        if tuple(int(x) for x in src.shape) != (h_dim, t_dim, w_dim):
            raise ValueError(f'{desc}: sagittal layer shape {tuple(src.shape)} != {(h_dim, t_dim, w_dim)}')
        # the per-y scatter (out[:, y,:] = src[y]) swept every destination
        # page once per y (T rows of W bytes at stride H*W). Copy K rows per block so each
        # t-slice window [t, y0:y1,:] is one contiguous K*W write and destination pages
        # are visited once per block instead of once per y.
        blk_rows = int(coronal_block_cols())
        n_row_blocks = (int(h_dim) + blk_rows - 1) // blk_rows

        def _copy_y_block(block_idx: int) -> None:
            y0 = int(block_idx) * blk_rows
            y1 = min(int(h_dim), y0 + blk_rows)
            sub = np.asarray(src[y0:y1])
            for t in range(int(t_dim)):
                out[int(t), y0:y1, :] = sub[:, int(t), :]

        parallel_for_indices_chunked(
            n_row_blocks,
            _copy_y_block,
            max_workers=choose_slice_parallel_workers(int(workers), n_row_blocks),
            desc=desc,
            show_progress=False,
            target_chunks_per_worker=2,
        )
    elif physical_view_name(view) == 'coronal':
        src = np.asarray(view_mask_mm)
        if tuple(int(x) for x in src.shape) != (w_dim, t_dim, h_dim):
            raise ValueError(f'{desc}: coronal layer shape {tuple(src.shape)} != {(w_dim, t_dim, h_dim)}')
        # the old per-x scatter (out[:,:, x] = src[x]) wrote one byte per cache
        # line across the whole (t, Y) extent, ~64x write amplification. Permute K columns at a
        # time through per-t (K, H) -> (H, K) tile transposes: tiles stay cache-resident and
        # both the source reads and destination writes use full cache lines.
        blk_cols = int(coronal_block_cols())
        n_blocks = (int(w_dim) + blk_cols - 1) // blk_cols

        def _copy_x_block(block_idx: int) -> None:
            x0 = int(block_idx) * blk_cols
            x1 = min(int(w_dim), x0 + blk_cols)
            sub = np.asarray(src[x0:x1])
            for t in range(int(t_dim)):
                out[int(t), :, x0:x1] = sub[:, int(t), :].T

        parallel_for_indices_chunked(
            n_blocks,
            _copy_x_block,
            max_workers=choose_slice_parallel_workers(int(workers), n_blocks),
            desc=desc,
            show_progress=False,
            target_chunks_per_worker=2,
        )
    else:  # pragma: no cover
        raise ValueError(f'Unsupported view for orthogonal NRRD projection: {view.name}/{view.family}')

    flush_array(out)
    return out

@runtime_telemetry_phase('projection.materialize')
def materialize_nrrd_view_layer(
    view_volume_mm: np.ndarray,
    *,
    model_name: str,
    view: ViewInfo,
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    interpolation_walk_back_index: int = 0,
    interpolation_candidate_index: int = 0,
    tile_config_id: str = '',
    tile_acceptance: str = '',
    stage: str = '',
    description: str = '',
    temp_dir: Path,
    workers: int = 1,
    known_has_foreground: Optional[bool] = None,
    known_row_occupancy: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    submit_to_sink: bool = True,
    force_path_backed_store: bool = False,
    internal_packbits_store: bool = False,
    emit_empty: bool = False,
) -> Optional[NrrdLayerRef]:
    """Persist a view-derived layer in orthogonal processing geometry for the NRRD writer."""
    if bool(internal_packbits_store) and bool(submit_to_sink):
        raise ValueError('The internal packbits cvol format must not be submitted as NRRD output')
    # callers that already know whether the volume has foreground (e.g. the
    # interpolation pass's added_voxels stat for bridge deltas) skip the per-slice scan.
    if known_has_foreground is not None:
        if not bool(known_has_foreground) and not bool(emit_empty):
            return None
    elif not bool(emit_empty) and not _volume_has_foreground(view_volume_mm):
        return None

    key = _nrrd_layer_key(
        view_name=str(view.name),
        source=str(source),
        mask_kind=str(mask_kind),
        pass_index=int(pass_index),
        tile_config_id=str(tile_config_id),
        tile_acceptance=str(tile_acceptance),
        stage=str(stage),
    )
    layer_dir = temp_dir / 'nrrd_layers' / str(view.name)
    storage_format = 'raw_u8'
    bbox_store_enabled = bool(raw_bbox_nrrd_layers_enabled() or internal_packbits_store)
    bbox_store_format = (
        INTERNAL_PACKED_CVOL_FORMAT
        if bool(internal_packbits_store)
        else CVOL_FORMAT
    )
    if bbox_store_enabled:
        raw_path = temp_dir / 'nrrd_work' / 'projected_layers' / str(view.name) / f'{key}.orthogonal.u8.dat'
        out_path = layer_dir / f'{key}.orthogonal.cvol'
    else:
        raw_path = layer_dir / f'{key}.orthogonal.u8.dat'
        out_path = raw_path

    transient_projection_in_memory = bool(
        bbox_store_enabled and not force_path_backed_store
    )
    # projected radial/tilted layers directly into source geometry. keeps
    # non-radial layers reduced: Cartesian layers are reduced axis permutations and Tilted
    # layers are reduced sheared orthogonal grids. Their sparse stores are therefore built at
    # inference pitch and the NRRD/final-union reader performs the one terminal restore.
    projection_out_shape: Optional[Tuple[int, int, int]] = None
    reduced_view_layer = bool(
        delayed_native_expansion_enabled()
        and tuple(int(v) for v in np.asarray(view_volume_mm).shape[-2:])
        != (int(view.src_h), int(view.src_w))
    )
    if view.family == 'radial' or (is_tilted_view(view) and not reduced_view_layer):
        projection_out_shape = final_source_output_shape()
    incremental_writer: Optional[IncrementalRawBBoxMaskStoreWriter] = None
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None
    incremental_extra_meta = {
        'nrrd_layer_key': key,
        'source_raw_path': str(raw_path),
        'source_raw_workspace': 'in_memory_when_available' if bool(transient_projection_in_memory) else 'disk_backed',
        'projection_payload_fusion': (
            (
                f'{radial_base_view_name(view)}_tilted_radial_composed_sink'
                if is_tilted_radial_view(view)
                else f'{radial_base_view_name(view)}_radial_sink_only'
            )
            if radial_sink_only_projection_supported(view) else 'dense_projection'
        ),
    }
    if bool(bbox_store_enabled) and radial_sink_only_projection_supported(view):
        expected_shape = (
            tuple(int(v) for v in projection_out_shape)
            if projection_out_shape is not None
            else (int(view.src_h), int(view.full_h), int(view.full_w))
        )
        try:
            incremental_writer = IncrementalRawBBoxMaskStoreWriter(
                shape=(int(expected_shape[0]), int(expected_shape[1]), int(expected_shape[2])),
                store_dir=out_path,
                format_name=bbox_store_format,
                desc=f'NRRD layer {key}',
                extra_meta=incremental_extra_meta,
                force_path_backed=bool(force_path_backed_store),
            )
            projection_block_callback = incremental_writer
        except Exception as exc:
            print(
                f'Warning: NRRD layer {key}: incremental radial cvol writer unavailable '
                f'({exc}); the completed projection will use the regular encoder.'
            )
            incremental_writer = None
            projection_block_callback = None

    def _project_layer(
        block_callback: Optional[Callable[[int, np.ndarray], None]],
        *,
        sink_only_mode: bool,
    ) -> np.ndarray | SinkOnlyProjectionResult:
        return project_view_volume_to_orthogonal_volume(
            view_volume_mm,
            view,
            raw_path,
            desc=f'NRRD layer {key}',
            workers=int(workers),
            prefer_memory=bool(transient_projection_in_memory),
            reserve_bytes=32 * GIB,
            out_shape_tyx=projection_out_shape,
            # transverse layers headed for a raw-bbox store are encoded straight
            # from the source volume (identity projection, synchronous encode) — no copy.
            allow_transverse_passthrough=bool(bbox_store_enabled),
            # device-union row occupancy (radial views only; valid for the
            # pre-interpolation layer, which is the only caller that supplies it).
            known_row_occupancy=known_row_occupancy,
            known_slice_bboxes=known_slice_bboxes,
            projection_block_callback=block_callback,
            sink_only=bool(sink_only_mode),
        )

    try:
        projected = _project_layer(
            projection_block_callback,
            sink_only_mode=bool(projection_block_callback is not None),
        )
    except Exception as exc:
        # A projection failure can leave a callback store incomplete. Discard it and retry
        # once without fusion. sink-only delivery is transactional: its callback failures
        # propagate here because no complete dense projected volume exists to recover from.
        if incremental_writer is None:
            raise
        incremental_writer.abort(exc)
        incremental_writer.discard()
        incremental_writer = None
        projection_block_callback = None
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        print(
            f'Warning: NRRD layer {key}: sink-only radial projection failed ({exc}); '
            'discarding the partial store and retrying with the dense fallback.'
        )
        projected = _project_layer(None, sink_only_mode=False)
    projected_sink_only = isinstance(projected, SinkOnlyProjectionResult)
    projected_is_source = bool(
        not projected_sink_only
        and np.may_share_memory(np.asarray(projected), np.asarray(view_volume_mm))
    )
    shape = (
        tuple(int(x) for x in projected.shape)
        if projected_sink_only
        else tuple(int(x) for x in np.asarray(projected).shape)
    )
    if bbox_store_enabled:
        layer_stats: Optional[Dict[str, object]] = None
        if incremental_writer is not None:
            try:
                if tuple(int(v) for v in incremental_writer.shape) != tuple(int(v) for v in shape):
                    raise ValueError(
                        f'incremental shape {incremental_writer.shape} != projected shape {shape}'
                    )
                layer_stats = incremental_writer.finalize()
            except Exception as exc:
                incremental_writer.abort(exc)
                incremental_writer.warn_failed_once(
                    f'NRRD layer {key}: incremental radial cvol finalization failed ({exc})'
                )
                incremental_writer.discard()
                incremental_writer = None
                layer_stats = None
                if projected_sink_only:
                    # Finalization (coverage/index/close) is part of the sink transaction.
                    # If it fails, rebuild once through the authoritative dense path rather
                    # than attempting to encode a SinkOnlyProjectionResult descriptor.
                    print(
                        f'NRRD layer {key}: retrying dense radial projection after '
                        'sink-only finalization failure.'
                    )
                    projected = _project_layer(None, sink_only_mode=False)
                    projected_sink_only = False
                    projected_is_source = bool(
                        np.may_share_memory(np.asarray(projected), np.asarray(view_volume_mm))
                    )
                    shape = tuple(int(x) for x in np.asarray(projected).shape)
        if layer_stats is None:
            if projected_sink_only:
                raise RuntimeError(
                    f'NRRD layer {key}: sink-only projection completed without a finalized store'
                )
            layer_stats = write_raw_bbox_mask_store(
                projected,
                out_path,
                format_name=bbox_store_format,
                desc=f'NRRD layer {key}',
                workers=int(workers),
                extra_meta={
                    'nrrd_layer_key': key,
                    'source_raw_path': 'encoded_direct_from_view_volume' if projected_is_source else str(raw_path),
                    'source_raw_workspace': 'in_memory_when_available' if bool(transient_projection_in_memory) else 'disk_backed',
                },
                force_path_backed=bool(force_path_backed_store),
            )
        segment_extent = _coerce_segment_extent(layer_stats.get('segment_extent_ijk')) or _nrrd_empty_segment_extent()
        segment_extent_source = 'raw_bbox_cvol_index'
        storage_format = bbox_store_format
        if not projected_is_source and not projected_sink_only:
            close_memmap_array(projected)
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
    else:
        if projected_sink_only:
            raise RuntimeError(f'NRRD layer {key}: sink-only result requires raw-bbox storage')
        segment_extent = compute_segment_extent_zyx(projected, workers=int(workers))
        segment_extent_source = 'raw_layer_materialization_scan'
        close_memmap_array(projected)

    layer_ref = NrrdLayerRef(
        key=key,
        name=_nrrd_layer_name(
            view=view,
            source=str(source),
            mask_kind=str(mask_kind),
            pass_index=int(pass_index),
            tile_config_id=str(tile_config_id),
            tile_acceptance=str(tile_acceptance),
            stage=str(stage),
        ),
        path=out_path,
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        dtype='uint8',
        storage_format=storage_format,
        model_name=str(model_name),
        view_name=str(view.name),
        physical_view_name=physical_view_name(view),
        aug_id=str(view.tta_aug_id),
        angle_deg=float(view.tta_angle_deg),
        view_family=str(view.family),
        source=str(source),
        mask_kind=str(mask_kind),
        pass_index=int(pass_index),
        interpolation_walk_back_index=int(interpolation_walk_back_index),
        interpolation_candidate_index=int(interpolation_candidate_index),
        tile_config_id=str(tile_config_id),
        tile_acceptance=str(tile_acceptance),
        stage=str(stage),
        description=str(description),
        segment_extent_ijk=segment_extent,
        segment_extent_shape_tyx=(int(shape[0]), int(shape[1]), int(shape[2])),
        segment_extent_source=segment_extent_source,
    )
    sink = nrrd_layer_sink()
    if sink is not None and bool(submit_to_sink):
        sink.submit_layer(
            layer_ref,
            nrrd_layer_output_suffix(
                view_token=view_output_token(view),
                source=str(source),
                mask_kind=str(mask_kind),
                pass_index=int(pass_index),
                interpolation_walk_back_index=int(interpolation_walk_back_index),
                interpolation_candidate_index=int(interpolation_candidate_index),
                tile_config_id=str(tile_config_id),
                tile_acceptance=str(tile_acceptance),
                stage=str(stage),
            ),
        )
    return layer_ref


def materialize_interpolation_component_nrrd_view_layer(
    component_store_path: Path,
    *,
    added_voxels: int,
    model_name: str,
    view: ViewInfo,
    source: str,
    pass_index: int,
    interpolation_walk_back_index: int,
    interpolation_candidate_index: int,
    tile_config_id: str = '',
    tile_acceptance: str = '',
    stage: str,
    description: str,
    temp_dir: Path,
    workers: int,
    keep_temp: bool,
) -> NrrdLayerRef:
    """Project one sparse interpolation component and submit its deterministic NRRD.

    Nonempty cvol inputs are expanded into only one reusable dense workspace at a time;
    empty combinations bypass projection entirely and reuse their compact all-empty cvol.
    """
    component_store_path = Path(component_store_path)
    store = RawBBoxMaskStore.open(component_store_path, mmap_payload=True)
    combo_stage = (
        f'{str(stage)}_walkback{int(interpolation_walk_back_index):02d}_'
        f'candidate{int(interpolation_candidate_index):02d}'
    )
    if int(added_voxels) <= 0:
        try:
            shape = tuple(int(v) for v in store.shape)
            key = _nrrd_layer_key(
                view_name=str(view.name),
                source=str(source),
                mask_kind='bridge',
                pass_index=int(pass_index),
                tile_config_id=str(tile_config_id),
                tile_acceptance=str(tile_acceptance),
                stage=str(combo_stage),
            )
            layer_ref = NrrdLayerRef(
                key=key,
                name=_nrrd_layer_name(
                    view=view,
                    source=str(source),
                    mask_kind='bridge',
                    pass_index=int(pass_index),
                    tile_config_id=str(tile_config_id),
                    tile_acceptance=str(tile_acceptance),
                    stage=str(combo_stage),
                ),
                path=component_store_path,
                shape=(int(shape[0]), int(shape[1]), int(shape[2])),
                dtype='uint8',
                storage_format=CVOL_FORMAT,
                model_name=str(model_name),
                view_name=str(view.name),
                physical_view_name=physical_view_name(view),
                aug_id=str(view.tta_aug_id),
                angle_deg=float(view.tta_angle_deg),
                view_family=str(view.family),
                source=str(source),
                mask_kind='bridge',
                pass_index=int(pass_index),
                interpolation_walk_back_index=int(interpolation_walk_back_index),
                interpolation_candidate_index=int(interpolation_candidate_index),
                tile_config_id=str(tile_config_id),
                tile_acceptance=str(tile_acceptance),
                stage=str(combo_stage),
                description=str(description),
                segment_extent_ijk=_nrrd_empty_segment_extent(),
                segment_extent_shape_tyx=(int(shape[0]), int(shape[1]), int(shape[2])),
                segment_extent_source='interpolation_empty_component_cvol',
            )
        finally:
            store.close()
        sink = nrrd_layer_sink()
        if sink is not None:
            sink.submit_layer(
                layer_ref,
                nrrd_layer_output_suffix(
                    view_token=view_output_token(view),
                    source=str(source),
                    mask_kind='bridge',
                    pass_index=int(pass_index),
                    interpolation_walk_back_index=int(interpolation_walk_back_index),
                    interpolation_candidate_index=int(interpolation_candidate_index),
                    tile_config_id=str(tile_config_id),
                    tile_acceptance=str(tile_acceptance),
                    stage=str(combo_stage),
                ),
            )
        return layer_ref

    decode_path = (
        Path(temp_dir) / 'nrrd_work' / 'interpolation_component_decode'
        / str(view.name)
        / f'{combo_stage}.u8.dat'
    )
    decoded_mm: Optional[np.ndarray] = None
    try:
        decoded_mm = allocate_workspace_array(
            shape=tuple(int(v) for v in store.shape),
            dtype=np.uint8,
            path=decode_path,
            desc=(
                f'Interpolation component decode {model_name}/{view.name} '
                f'walkback {int(interpolation_walk_back_index)} '
                f'candidate {int(interpolation_candidate_index)}'
            ),
            prefer_memory=False,
            prefer_memfd=False,
            initialize_zero=False,
        )

        def _decode_component_slice(idx: int) -> None:
            store.fill_decoded_slice_into(int(idx), decoded_mm[int(idx)])  # type: ignore[index]

        parallel_for_indices_chunked(
            int(store.shape[0]),
            _decode_component_slice,
            max_workers=choose_slice_parallel_workers(int(workers), int(store.shape[0])),
            desc=f'Interpolation component decode {view.name}',
            show_progress=False,
            target_chunks_per_worker=2,
        )
        flush_array(decoded_mm)
        layer_ref = materialize_nrrd_view_layer(
            decoded_mm,
            model_name=str(model_name),
            view=view,
            source=str(source),
            mask_kind='bridge',
            pass_index=int(pass_index),
            interpolation_walk_back_index=int(interpolation_walk_back_index),
            interpolation_candidate_index=int(interpolation_candidate_index),
            tile_config_id=str(tile_config_id),
            tile_acceptance=str(tile_acceptance),
            stage=str(combo_stage),
            description=str(description),
            temp_dir=Path(temp_dir),
            workers=int(workers),
            known_has_foreground=True,
        )
        if layer_ref is None:  # pragma: no cover - known nonempty cvol contract
            raise RuntimeError(f'Nonempty interpolation component was suppressed: {component_store_path}')
        return layer_ref
    finally:
        store.close()
        close_memmap_array(decoded_mm)
        if not bool(keep_temp):
            try:
                decode_path.unlink(missing_ok=True)
            except Exception:
                pass
            # Nonempty inputs have been synchronously projected into the returned ref's own
            # store, so their view-native cvol can be retired immediately. Empty cvols remain
            # the returned ref backing until the NRRD sink finishes.
            shutil.rmtree(component_store_path, ignore_errors=True)

def materialize_internal_final_view_layer(
    view_volume_mm: np.ndarray,
    *,
    model_name: str,
    view: ViewInfo,
    temp_dir: Path,
    workers: int,
) -> Optional[NrrdLayerRef]:
    """Persist one private, pathname-backed final-view ref for dense retirement.

    This layer is a terminal-union input, not a requested NRRD output.  Forcing pathname
    storage prevents the replacement sparse payload from silently consuming memfd RAM. Its
    bbox crops are row-wise bit-packed; exported CTILE/NRRD component formats remain raw u8.
    """
    return materialize_nrrd_view_layer(
        view_volume_mm,
        model_name=str(model_name),
        view=view,
        source='internal',
        mask_kind='union',
        pass_index=0,
        stage='final_sparse_retention',
        description=(
            'Private final-view raw-bbox layer used to retire the dense canvas; '
            'not submitted as an NRRD output.'
        ),
        temp_dir=Path(temp_dir),
        workers=int(workers),
        submit_to_sink=False,
        force_path_backed_store=True,
        internal_packbits_store=True,
    )

def materialize_nrrd_global_layer(
    volume_mm: np.ndarray,
    *,
    model_name: str,
    source: str,
    mask_kind: str,
    pass_index: int = 0,
    stage: str = '',
    description: str = '',
    temp_dir: Path,
    workers: int = 1,
    known_has_foreground: Optional[bool] = None,
    volume_is_immutable: bool = False,
    keep_temp: bool = False,
    layer_role: str = 'checkpoint',
    recomposition_op: str = 'select',
    low_quality_recomposition_op: str = 'select',
    mirror_low_quality: bool = True,
) -> Optional[NrrdLayerRef]:
    # see materialize_nrrd_view_layer.
    if known_has_foreground is not None:
        if not bool(known_has_foreground):
            return None
    elif not _volume_has_foreground(volume_mm):
        return None
    view_name = 'global'
    key = _nrrd_layer_key(
        view_name=view_name,
        source=str(source),
        mask_kind=str(mask_kind),
        pass_index=int(pass_index),
        stage=str(stage),
    )
    layer_dir = temp_dir / 'nrrd_layers' / view_name
    storage_format = 'raw_u8'
    if raw_bbox_nrrd_layers_enabled():
        raw_path = temp_dir / 'nrrd_work' / 'global_layers' / f'{key}.orthogonal.u8.dat'
        out_path = layer_dir / f'{key}.orthogonal.cvol'
    else:
        raw_path = layer_dir / f'{key}.orthogonal.u8.dat'
        out_path = raw_path

    # IMMUTABLE global layers (the final output — nothing mutates the volume
    # after it) skip the store entirely: the sink streams the live in-RAM volume in ONE
    # pass (segment extent computed on the sink worker), deleting the synchronous
    # encode-to-store pass on this thread AND the store read-back pass in the writer.
    # keep_temp keeps the store path so temp archives still contain the layer.
    sink = nrrd_layer_sink()
    if (
        bool(volume_is_immutable)
        and not bool(keep_temp)
        and nrrd_live_global_layer_enabled()
        and sink is not None
    ):
        source_arr = np.asarray(volume_mm, dtype=np.uint8)
        shape = tuple(int(x) for x in source_arr.shape)
        layer_ref = NrrdLayerRef(
            key=key,
            name=_nrrd_layer_name(
                view=None,
                source=str(source),
                mask_kind=str(mask_kind),
                pass_index=int(pass_index),
                stage=str(stage),
            ),
            path=layer_dir / f'{key}.live_volume',  # placeholder; never created
            shape=(int(shape[0]), int(shape[1]), int(shape[2])),
            dtype='uint8',
            storage_format='live_u8',
            model_name=str(model_name),
            view_name=view_name,
            view_family='global',
            source=str(source),
            mask_kind=str(mask_kind),
            pass_index=int(pass_index),
            stage=str(stage),
            description=str(description),
            layer_role=str(layer_role),
            recomposition_op=str(recomposition_op),
            low_quality_recomposition_op=str(low_quality_recomposition_op),
            mirror_low_quality=bool(mirror_low_quality),
            segment_extent_ijk=None,  # deferred: computed by the sink worker before the header
            segment_extent_shape_tyx=(int(shape[0]), int(shape[1]), int(shape[2])),
            segment_extent_source='deferred_live_volume_scan',
            live_array=source_arr,
        )
        sink.submit_layer(
            layer_ref,
            nrrd_layer_output_suffix(
                view_token='Global',
                source=str(source),
                mask_kind=str(mask_kind),
                pass_index=int(pass_index),
                stage=str(stage),
            ),
        )
        print(
            f'NRRD layer {key}: streaming from the live volume '
            '(v13.3.6 N3; store encode + read-back deleted).'
        )
        return layer_ref

    if raw_bbox_nrrd_layers_enabled():
        # encode the raw-bbox store straight from the source volume. The old
        # copy_workspace_array staged a full copy (~36 GB of traffic per global layer on the
        # serial tail) purely as encoder input; write_raw_bbox_mask_store completes before this
        # function returns, and no caller mutates the volume during the synchronous call.
        source_arr = np.asarray(volume_mm, dtype=np.uint8)
        shape = tuple(int(x) for x in source_arr.shape)
        layer_stats = write_raw_bbox_mask_store(
            source_arr,
            out_path,
            format_name=CVOL_FORMAT,
            desc=f'NRRD layer {key}',
            workers=int(workers),
            extra_meta={
                'nrrd_layer_key': key,
                'source_raw_path': 'encoded_direct_from_source_volume',
                'source_raw_workspace': 'source_volume',
            },
        )
        segment_extent = _coerce_segment_extent(layer_stats.get('segment_extent_ijk')) or _nrrd_empty_segment_extent()
        segment_extent_source = 'raw_bbox_cvol_index'
        storage_format = CVOL_FORMAT
    else:
        copied = copy_workspace_array(
            np.asarray(volume_mm, dtype=np.uint8),
            raw_path,
            desc=f'NRRD layer {key}',
            prefer_memory=False,
            reserve_bytes=32 * GIB,
            workers=int(workers),
        )
        shape = tuple(int(x) for x in np.asarray(copied).shape)
        segment_extent = compute_segment_extent_zyx(copied, workers=int(workers))
        segment_extent_source = 'raw_layer_materialization_scan'
        close_memmap_array(copied)

    layer_ref = NrrdLayerRef(
        key=key,
        name=_nrrd_layer_name(
            view=None,
            source=str(source),
            mask_kind=str(mask_kind),
            pass_index=int(pass_index),
            stage=str(stage),
        ),
        path=out_path,
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        dtype='uint8',
        storage_format=storage_format,
        model_name=str(model_name),
        view_name=view_name,
        view_family='global',
        source=str(source),
        mask_kind=str(mask_kind),
        pass_index=int(pass_index),
        stage=str(stage),
        description=str(description),
        layer_role=str(layer_role),
        recomposition_op=str(recomposition_op),
        low_quality_recomposition_op=str(low_quality_recomposition_op),
        mirror_low_quality=bool(mirror_low_quality),
        segment_extent_ijk=segment_extent,
        segment_extent_shape_tyx=(int(shape[0]), int(shape[1]), int(shape[2])),
        segment_extent_source=segment_extent_source,
    )
    sink = nrrd_layer_sink()
    if sink is not None:
        sink.submit_layer(
            layer_ref,
            nrrd_layer_output_suffix(
                view_token='Global',
                source=str(source),
                mask_kind=str(mask_kind),
                pass_index=int(pass_index),
                stage=str(stage),
            ),
        )
    return layer_ref

def view_interpolation_wrap_axis(view: ViewInfo) -> bool:
    """Return the fixed interpolation boundary policy for one view family.

    Radial and Tilted-Radial frame order is circular and always wraps. Cartesian and
    Tilted-Cartesian stacks are linear and never wrap. v16.4.0 intentionally exposes no
    environment variable or CLI switch capable of changing this geometry contract.
    """
    return bool(is_radial_view(view))

def prepare_view_volume_after_fullframe(
    *,
    model_name: str,
    view: ViewInfo,
    union_mm: np.ndarray,
    confmap_mm: Optional[np.ndarray],
    union_path: Path,
    confmap_path: Optional[Path],
    temp_dir: Path,
    dense_tiling_active: bool,
    min_conf: float,
    min_radius: float,
    interpolate: int,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_passes: int,
    interpolate_min_radius: float,
    interpolation_search_angle: float,
    keep_temp: bool,
    slice_workers: int,
    interpolation_task_workers: int,
    nrrd_layers_enabled: bool = False,
    precleaned_slice_cleanup: bool = False,
    hole_fill_done_on_device: bool = False,
    slice_meta: Optional[Dict[str, object]] = None,
    fuse_radial_component_layers: bool = False,
    parent_mask_ready_callback: Optional[Callable[[str, str, object], None]] = None,
    internal_final_layer_enabled: bool = False,
    preinterpolation_layer_already_published: bool = False,
) -> PreparedViewResult:
    # Local import keeps the package dependency graph acyclic.
    from .finalization import union_volume_into_volume

    baseline_native_volume = union_mm
    d1_delta_only = bool(preinterpolation_layer_already_published)
    d1_additions_mm: Optional[np.ndarray] = None
    d1_additions_path: Optional[Path] = None
    nrrd_layers: List[NrrdLayerRef] = []
    parent_mask_support_mm: Optional[object] = None
    parent_bridge_support_mm: Optional[object] = None
    parent_mask_support_path: Optional[Path] = None
    parent_bridge_support_path: Optional[Path] = None
    fused_radial_components = bool(
        fuse_radial_component_layers
        and nrrd_layers_enabled
        and str(view.family) == 'radial'
        and not bool(dense_tiling_active)
        and not bool(preinterpolation_layer_already_published)
        and not (
            _view_uses_interpolation(view, int(interpolate))
            and int(interpolation_walk_back) > 0
            and int(interpolation_candidates) > 0
        )
    )

    # Device-union slice metadata is aggregated per view by the scheduler.
    # It remains valid through skipped cleanup and per-slice hole filling, which do not change
    # foreground presence/bounds/row occupancy; interpolation bridges invalidate it. It feeds
    # only pre-interpolation consumers and the first interpolation pass's labeling.
    meta_valid = bool(slice_meta) and bool(slice_meta.get('valid', False))
    meta_slice_any: Optional[np.ndarray] = None
    meta_slice_bboxes: Optional[np.ndarray] = None
    meta_row_occupancy: Optional[np.ndarray] = None
    if meta_valid:
        try:
            meta_slice_any = np.asarray(slice_meta['slice_any'], dtype=bool)
            meta_slice_bboxes = np.asarray(slice_meta['slice_bboxes'], dtype=np.int64)
            rows_packed = slice_meta.get('slice_row_any')
            row_count = int(slice_meta.get('slice_row_count', 0) or 0)
            if rows_packed is not None and row_count > 0:
                meta_row_occupancy = np.unpackbits(
                    np.asarray(rows_packed, dtype=np.uint8), axis=1, count=row_count,
                ).astype(bool, copy=False)
            if int(meta_slice_any.shape[0]) != int(view.num_slices):
                raise ValueError('slice metadata length mismatch')
        except Exception:
            meta_valid = False
            meta_slice_any = None
            meta_slice_bboxes = None
            meta_row_occupancy = None

    hole_metadata_valid = bool(
        meta_valid
        and (
            bool(precleaned_slice_cleanup)
            or (float(min_conf) <= 0.0 and float(min_radius) <= 0.0)
        )
    )
    cleanup_view_volume_after_prediction_inplace(
        baseline_native_volume,
        confmap_mm,
        view,
        float(min_conf),
        float(min_radius),
        workers=int(slice_workers),
        precleaned_slice_cleanup=bool(precleaned_slice_cleanup),
        skip_hole_fill=bool(hole_fill_done_on_device),
        known_slice_any=(meta_slice_any if hole_metadata_valid else None),
        known_slice_bboxes=(meta_slice_bboxes if hole_metadata_valid else None),
    )

    close_memmap_array(confmap_mm)
    if confmap_path is not None and not keep_temp:
        try:
            confmap_path.unlink(missing_ok=True)
        except Exception:
            pass

    if bool(d1_delta_only):
        d1_additions_path = (
            temp_dir / 'd1_view_additions' / str(model_name)
            / f'{str(view.name)}.u8.dat'
        )
        d1_additions_mm = allocate_workspace_array(
            shape=tuple(int(v) for v in np.asarray(baseline_native_volume).shape),
            dtype=np.uint8,
            path=d1_additions_path,
            desc=f'D1 delta-only continuation {model_name}/{view.name}',
            prefer_memory=False,
            prefer_memfd=False,
            reserve_bytes=16 * GIB,
            initialize_zero=True,
        )
        runtime_telemetry().add('d1.delta_continuation_volume_bytes', int(np.asarray(d1_additions_mm).nbytes))

    if bool(nrrd_layers_enabled) and not bool(preinterpolation_layer_already_published):
        if not bool(fused_radial_components):
            layer_ref = materialize_nrrd_view_layer(
                baseline_native_volume,
                model_name=str(model_name),
                view=view,
                source='fullframe',
                mask_kind='yolo',
                pass_index=0,
                stage='pre_interpolation',
                description='Cleaned full-frame YOLO mask before interpolation bridges.',
                temp_dir=temp_dir,
                workers=int(slice_workers),
                # the device union already answered the foreground question and
                # (for radial views) the per-row occupancy the backprojection would rescan.
                known_has_foreground=(bool(meta_slice_any.any()) if meta_valid and meta_slice_any is not None else None),
                known_row_occupancy=(
                    np.ascontiguousarray(meta_row_occupancy.any(axis=0))
                    if (meta_valid and meta_row_occupancy is not None and str(view.family) == 'radial')
                    else None
                ),
                known_slice_bboxes=(meta_slice_bboxes if meta_valid else None),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)

    # Per-tile gating always needs the immutable same-angle parent YOLO support, even
    # when NRRD output is disabled. Keep it as a sparse slice-chunked cvol so the new
    # two-stage gate can classify each original tile component before any cross-tile OR.
    if bool(dense_tiling_active):
        parent_mask_support_path = temp_dir / 'tile_support' / model_name / view.name / 'fullframe_yolo_support.cvol'
        write_raw_bbox_mask_store(
            baseline_native_volume,
            parent_mask_support_path,
            format_name=CVOL_FORMAT,
            desc=f'Tile support fullframe YOLO {model_name}/{view.name}',
            workers=int(slice_workers),
            extra_meta={
                'support_kind': 'fullframe_yolo_pre_interpolation',
                'physical_view': physical_view_name(view),
                'tta_aug_id': str(view.tta_aug_id),
                'tta_angle_deg': float(view.tta_angle_deg),
            },
        )
        parent_mask_support_mm = RawBBoxMaskStore.open(parent_mask_support_path, mmap_payload=True)
        # Publish immutable P immediately after cleanup, before any parent interpolation.
        # Tile components may now perform their first gate while the parent bridge planner runs.
        if parent_mask_ready_callback is not None:
            parent_mask_ready_callback(str(model_name), str(view.name), parent_mask_support_mm)


    interpolation_stats: List[Dict[str, object]] = []
    processing_plane_shape = tuple(int(v) for v in np.asarray(baseline_native_volume).shape[-2:])
    effective_interpolate_min_radius = view_processing_min_radius(
        view, float(interpolate_min_radius), processing_plane_shape,
    )
    effective_interpolation_search_angle = view_processing_search_angle(
        view, float(interpolation_search_angle), processing_plane_shape,
    )
    if _view_uses_interpolation(view, int(interpolate)):
        total_passes = int(interpolate_passes)
        for pass_idx in range(1, total_passes + 1):
            # the pass itself writes the exact added-voxel delta
            # (bridge AND NOT pre-merge mask) to this path during its merge step, replacing
            # the old full-volume before-copy + subtract bookkeeping.
            pass_delta_path: Optional[Path] = None
            if bool(d1_delta_only) and not bool(fused_radial_components):
                pass_delta_path = temp_dir / 'nrrd_work' / view.name / f'fullframe_bridge_pass{int(pass_idx):02d}.u8.dat'
            pass_component_dir: Optional[Path] = None
            if (
                bool(nrrd_layers_enabled)
                and not bool(fused_radial_components)
                and int(interpolation_walk_back) > 0
                and int(interpolation_candidates) > 0
            ):
                pass_component_dir = (
                    temp_dir / 'nrrd_work' / view.name
                    / f'fullframe_bridge_pass{int(pass_idx):02d}_components'
                )

            baseline_native_volume, stats_local = interpolate_view_volume_pass_maybe_process(
                mask_mm=baseline_native_volume,
                view=view,
                work_dir=temp_dir / 'interpolation' / model_name / view.name,
                pass_tag=f'pass{pass_idx}',
                max_slice_distance=int(interpolate),
                search_angle_deg=float(effective_interpolation_search_angle),
                interpolation_walk_back=int(interpolation_walk_back),
                interpolation_candidates=int(interpolation_candidates),
                interpolate_min_radius=float(effective_interpolate_min_radius),
                keep_temp=bool(keep_temp),
                prefer_memory=True,
                workers=int(interpolation_task_workers),
                bridge_delta_path=pass_delta_path,
                bridge_component_dir=pass_component_dir,
                # pass 1 labels exactly the flushed volume; later passes see
                # bridge-mutated content, so the metadata is only forwarded for pass 1.
                known_slice_any=(meta_slice_any if (meta_valid and int(pass_idx) == 1) else None),
                known_slice_bboxes=(meta_slice_bboxes if (meta_valid and int(pass_idx) == 1) else None),
            )
            stats_local = dict(stats_local)
            stats_local.update({
                'pass_index': int(pass_idx),
                'model': str(model_name),
                'view': str(view.name),
                'source': 'fullframe',
                'max_slice_distance': int(interpolate),
                'interpolation_walk_back': int(interpolation_walk_back),
                'interpolation_candidates': int(interpolation_candidates),
                'interpolation_search_angle': float(interpolation_search_angle),
                'processing_interpolation_search_angle': float(effective_interpolation_search_angle),
                'processing_interpolate_min_radius': float(effective_interpolate_min_radius),
            })
            interpolation_stats.append(stats_local)

            if pass_delta_path is not None:
                delta_reported = str(stats_local.get('bridge_delta_path', '') or '')
                if delta_reported and Path(delta_reported).exists():
                    bridge_delta_mm = np.memmap(
                        Path(delta_reported),
                        dtype=np.uint8,
                        mode='r',
                        shape=tuple(int(v) for v in np.asarray(baseline_native_volume).shape),
                    )
                    if d1_additions_mm is not None:
                        union_volume_into_volume(
                            d1_additions_mm,
                            bridge_delta_mm,
                            workers=int(slice_workers),
                            desc=(
                                f'D1 delta-only full-frame bridge pass {int(pass_idx)} '
                                f'{model_name}/{view.name}'
                            ),
                        )
                    close_memmap_array(bridge_delta_mm)
                if not bool(keep_temp):
                    try:
                        pass_delta_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            if pass_component_dir is not None:
                component_entries = [
                    dict(entry)
                    for entry in stats_local.get('bridge_component_deltas', [])
                ]
                expected_components = int(interpolation_walk_back) * int(interpolation_candidates)
                if len(component_entries) != int(expected_components):
                    raise RuntimeError(
                        f'{model_name}/{view.name} interpolation pass {int(pass_idx)} returned '
                        f'{len(component_entries)} component delta(s); expected '
                        f'{int(interpolation_walk_back)} x {int(interpolation_candidates)} = '
                        f'{int(expected_components)}'
                    )
                component_entries.sort(key=lambda entry: (
                    int(entry.get('walk_back_index', 0)),
                    int(entry.get('candidate_index', 0)),
                ))
                for component_entry in component_entries:
                    walk_back_index = int(component_entry['walk_back_index'])
                    candidate_index = int(component_entry['candidate_index'])
                    layer_ref = materialize_interpolation_component_nrrd_view_layer(
                        Path(str(component_entry['path'])),
                        added_voxels=int(component_entry.get('added_voxels', 0)),
                        model_name=str(model_name),
                        view=view,
                        source='fullframe',
                        pass_index=int(pass_idx),
                        interpolation_walk_back_index=int(walk_back_index),
                        interpolation_candidate_index=int(candidate_index),
                        stage='interpolation',
                        description=(
                            'Voxels added by this full-frame interpolation pass for '
                            f'walk-back origin {int(walk_back_index)} and candidate '
                            f'{int(candidate_index)} only.'
                        ),
                        temp_dir=temp_dir,
                        workers=int(slice_workers),
                        keep_temp=bool(keep_temp),
                    )
                    nrrd_layers.append(layer_ref)


            if int(stats_local.get('added_voxels', 0)) <= 0:
                break
    else:
        # Direct-union CUDA workers already wrote this completed view into a root file-backed
        # memmap. Re-copying it to a second *.noninterpolated_native file read and rewrote the
        # entire multi-GiB view after inference, often as a low-CPU disk/page-cache tail. Retain
        # the existing backing in place; only anonymous/fallback arrays need a drain copy.
        old_volume = baseline_native_volume
        existing_backing = _interpolation_array_backing_path(old_volume)
        if existing_backing is not None:
            print(
                f'{model_name}/{view.name} non-interpolated native retention (v16.1.3): '
                f'reusing {existing_backing}; no full-volume drain copy.'
            )
        else:
            drained_path = (
                temp_dir / 'view_volumes' / model_name
                / f'{view.name}.noninterpolated_native.u8.dat'
            )
            baseline_native_volume = _drain_volume_to_mmap(
                old_volume,
                drained_path,
                desc=f'{model_name}/{view.name} non-interpolated native drain',
                workers=int(slice_workers),
            )
            if old_volume is not baseline_native_volume:
                close_memmap_array(old_volume)
                if not keep_temp:
                    try:
                        union_path.unlink(missing_ok=True)
                    except Exception:
                        pass

    if bool(fused_radial_components):
        # projection and positive-support restore distribute over binary OR, so the
        # cleaned YOLO mask and every accepted bridge can be kept in the already-mutated
        # baseline and projected once. This removes one source-store write and one complete
        # Radial backprojection per bridge pass on the prioritized angle-variant path.
        fused_added_voxels = sum(
            max(0, int(item.get('added_voxels', 0) or 0))
            for item in interpolation_stats
        )
        has_fused_bridges = int(fused_added_voxels) > 0
        layer_ref = materialize_nrrd_view_layer(
            baseline_native_volume,
            model_name=str(model_name),
            view=view,
            source='fullframe',
            mask_kind=('union' if has_fused_bridges else 'yolo'),
            pass_index=0,
            stage=('post_interpolation' if has_fused_bridges else 'pre_interpolation'),
            description=(
                'Cleaned full-frame YOLO mask fused with all interpolation bridges.'
                if has_fused_bridges
                else 'Cleaned full-frame YOLO mask; interpolation added no bridge voxels.'
            ),
            temp_dir=temp_dir,
            workers=int(slice_workers),
            known_has_foreground=(
                bool(meta_slice_any.any())
                if meta_valid and meta_slice_any is not None else None
            ),
            known_row_occupancy=(
                np.ascontiguousarray(meta_row_occupancy.any(axis=0))
                if (
                    meta_valid and meta_row_occupancy is not None
                    and not _view_uses_interpolation(view, int(interpolate))
                ) else None
            ),
        )
        if layer_ref is not None:
            nrrd_layers.append(layer_ref)
            print(
                f'v13.3.18 (C12): {model_name}/{view.name} emitted one Radial '
                f'{"YOLO+bridge union" if has_fused_bridges else "YOLO"} layer for one '
                f'projection pass ({int(fused_added_voxels)} bridge voxel(s)).'
            )

    if bool(dense_tiling_active) and parent_mask_support_mm is not None:
        parent_bridge_support_path = temp_dir / 'tile_support' / model_name / view.name / 'fullframe_bridge_support.cvol'
        bridge_stats = subtract_volume_to_raw_bbox_store(
            baseline_native_volume,
            parent_mask_support_mm,
            parent_bridge_support_path,
            desc=f'NRRD support fullframe bridges {model_name}/{view.name}',
            workers=int(slice_workers),
            format_name=CVOL_FORMAT,
        )
        if int(bridge_stats.get('foreground_voxels', 0)) <= 0:
            parent_bridge_support_mm = None
            if parent_bridge_support_path is not None:
                try:
                    shutil.rmtree(parent_bridge_support_path, ignore_errors=True)
                except Exception:
                    pass
        else:
            parent_bridge_support_mm = RawBBoxMaskStore.open(parent_bridge_support_path, mmap_payload=True)

    if bool(internal_final_layer_enabled) and not bool(dense_tiling_active):
        internal_ref = materialize_internal_final_view_layer(
            (d1_additions_mm if d1_additions_mm is not None else baseline_native_volume),
            model_name=str(model_name),
            view=view,
            temp_dir=temp_dir,
            workers=int(slice_workers),
        )
        if internal_ref is not None:
            nrrd_layers.append(internal_ref)

    if bool(d1_delta_only):
        if d1_additions_mm is None:
            raise RuntimeError(
                f'{model_name}/{view.name}: D1 continuation did not allocate its additions volume'
            )
        base_bytes = int(np.asarray(baseline_native_volume).nbytes)
        close_memmap_array_without_flush(baseline_native_volume)
        if (
            not bool(keep_temp)
            and union_path is not None
            and not str(union_path).startswith('/proc/')
        ):
            try:
                Path(union_path).unlink(missing_ok=True)
            except Exception:
                pass
        baseline_native_volume = d1_additions_mm
        print(
            f'D1 delta-only continuation ready for {model_name}/{view.name}: '
            f'released {base_bytes / GIB:.2f} GiB materialized base; final dense view now '
            'contains only interpolation/tile additions because the source-space D1 base '
            'was already published.'
        )

    # Return the dense angle variant to the scheduler. When immutable component NRRD
    # references fully cover it, the scheduler may retire this canvas before physical-view
    # finalization; otherwise the legacy dense TTA-collapse path remains authoritative.
    sparse_retire_dense = False

    if sparse_retire_dense:
        retired_bytes = int(np.asarray(baseline_native_volume).nbytes)
        close_memmap_array_without_flush(baseline_native_volume)
        baseline_native_volume = None  # type: ignore[assignment]
        try:
            if not bool(keep_temp) and union_path is not None and not str(union_path).startswith('/proc/'):
                Path(union_path).unlink(missing_ok=True)
        except Exception:
            pass
        print(
            f'v16.1.3 sparse retirement: {model_name}/{view.name} released '
            f'{retired_bytes / GIB:.2f} GiB dense union after cvol materialization.'
        )

    final_view_volume: Optional[np.ndarray] = None
    if sparse_retire_dense:
        final_view_volume = None
    elif bool(dense_tiling_active):
        # The immutable parent-YOLO mask P and parent-bridge mask B now live in separate
        # sparse support stores. Tile gates never read this dense array, so accepted tile
        # masks can be ORed into it after both supports are frozen without a full-volume copy.
        final_view_volume = baseline_native_volume
    elif view.family == 'radial':
        # Keep Radial masks view-native here. Final projection is orientation-aware and may
        # stream directly into destination bands; transverse also has a GPU backprojection path.
        final_view_volume = baseline_native_volume
    elif is_tilted_view(view):
        # Keep Tilted masks view-native here for the final CPU backprojection.
        final_view_volume = baseline_native_volume
    else:
        final_view_volume = baseline_native_volume

    returned_parent_mask_support = parent_mask_support_mm if bool(dense_tiling_active) else None
    returned_parent_bridge_support = parent_bridge_support_mm if bool(dense_tiling_active) else None
    if parent_mask_support_mm is not None and returned_parent_mask_support is None:
        close_raw_store_or_memmap_volume(parent_mask_support_mm, keep_temp=bool(keep_temp))
    if parent_bridge_support_mm is not None and returned_parent_bridge_support is None:
        close_raw_store_or_memmap_volume(parent_bridge_support_mm, keep_temp=bool(keep_temp))

    return PreparedViewResult(
        model_name=str(model_name),
        view_name=str(view.name),
        aug_id=str(view.tta_aug_id),
        angle_deg=float(view.tta_angle_deg),
        native_support_mm=baseline_native_volume,
        final_view_volume_mm=final_view_volume,
        interpolation_stats=interpolation_stats,
        nrrd_layers=nrrd_layers,
        parent_mask_support_mm=returned_parent_mask_support,
        parent_bridge_support_mm=returned_parent_bridge_support,
    )

def gate_tile_components_against_support_inplace(
    tile_mask_mm: np.ndarray,
    support_mm: object,
    *,
    parent_crop: Tuple[int, int, int, int],
    accepted_total_mm: np.ndarray,
    accepted_total_locks: Optional[Sequence[threading.Lock]] = None,
    accepted_category_mm: Optional[np.ndarray] = None,
    accepted_category_locks: Optional[Sequence[threading.Lock]] = None,
    retain_rejected_components: bool,
    workers: int = 1,
    desc: str = 'Tile component support gate',
) -> Dict[str, int]:
    """Gate every 2-D component from one original tile against one immutable support.

    The tile volume is crop-local. Accepted components are ORed into the matching crop of
    the full variant accumulator. With ``retain_rejected_components=True`` the input is
    replaced by whole rejected components for a later bridge-support gate; otherwise it is
    replaced by the accepted mask only. Components from different tiles are never merged
    before either decision.
    """
    tile = np.asarray(tile_mask_mm)
    if tile.ndim != 3:
        raise ValueError(f'{desc}: expected a 3-D tile mask, got {tile.shape}')
    py0, py1, px0, px1 = (int(v) for v in parent_crop)
    if not (0 <= py0 < py1 <= int(accepted_total_mm.shape[1]) and 0 <= px0 < px1 <= int(accepted_total_mm.shape[2])):
        raise ValueError(
            f'{desc}: parent crop {(py0, py1, px0, px1)} is outside '
            f'accumulator shape {tuple(int(v) for v in accepted_total_mm.shape)}'
        )
    expected_local = (int(py1 - py0), int(px1 - px0))
    if tuple(int(v) for v in tile.shape[1:]) != expected_local:
        raise ValueError(
            f'{desc}: crop-local tile plane {tuple(int(v) for v in tile.shape[1:])} '
            f'does not match parent crop {expected_local}'
        )
    support_shape = _volume_shape_tuple(support_mm)
    if int(tile.shape[0]) != int(support_shape[0]):
        raise ValueError(
            f'{desc}: tile/support slice mismatch {int(tile.shape[0])} != {int(support_shape[0])}'
        )
    if tuple(int(v) for v in accepted_total_mm.shape) != tuple(int(v) for v in support_shape):
        raise ValueError(
            f'{desc}: support/accumulator shape mismatch {support_shape} != '
            f'{tuple(int(v) for v in accepted_total_mm.shape)}'
        )
    if accepted_category_mm is not None and tuple(int(v) for v in accepted_category_mm.shape) != tuple(int(v) for v in support_shape):
        raise ValueError(f'{desc}: category accumulator shape mismatch')

    total_lock_count = int(len(accepted_total_locks)) if accepted_total_locks else 0
    category_lock_count = int(len(accepted_category_locks)) if accepted_category_locks else 0

    def _or_crop(
        dst_mm: Optional[np.ndarray],
        locks: Optional[Sequence[threading.Lock]],
        lock_count: int,
        idx: int,
        plane_u8: np.ndarray,
    ) -> None:
        if dst_mm is None or not bool(np.any(plane_u8)):
            return
        lock = locks[int(idx) % int(lock_count)] if locks is not None and lock_count > 0 else None
        ctx = lock if lock is not None else contextlib.nullcontext()
        with ctx:
            dst_view = dst_mm[int(idx), py0:py1, px0:px1]
            np.bitwise_or(dst_view, plane_u8, out=dst_view)

    def _process(idx: int) -> Tuple[int, int, int, int]:
        tile_slice = np.asarray(tile[int(idx)], dtype=np.uint8) > 0
        if not bool(np.any(tile_slice)):
            tile[int(idx)] = np.uint8(0)
            return (0, 0, 0, 0)

        num_labels, labels2d, stats, _centroids = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(tile_slice, dtype=np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        num_labels = int(num_labels)
        if num_labels <= 1:
            tile[int(idx)] = np.uint8(0)
            return (0, 0, 0, 0)

        sizes = np.asarray(stats[:num_labels, cv2.CC_STAT_AREA], dtype=np.int64)
        present = sizes > 0
        present[0] = False
        support_slice = _read_binary_volume_slice_crop_bool(
            support_mm, int(idx), int(py0), int(py1), int(px0), int(px1),
        )
        if tuple(int(v) for v in support_slice.shape) != expected_local:
            raise ValueError(
                f'{desc}: support crop shape {tuple(int(v) for v in support_slice.shape)} '
                f'does not match tile plane {expected_local}'
            )
        hits = np.bincount(labels2d[support_slice], minlength=num_labels)[:num_labels] > 0
        accepted_labels = present & hits
        rejected_labels = present & (~accepted_labels)
        accepted_plane = accepted_labels[labels2d].astype(np.uint8, copy=False)
        rejected_plane = rejected_labels[labels2d].astype(np.uint8, copy=False)

        _or_crop(accepted_total_mm, accepted_total_locks, total_lock_count, int(idx), accepted_plane)
        _or_crop(accepted_category_mm, accepted_category_locks, category_lock_count, int(idx), accepted_plane)
        tile[int(idx)] = rejected_plane if bool(retain_rejected_components) else accepted_plane

        return (
            int(np.count_nonzero(accepted_labels)),
            int(np.count_nonzero(rejected_labels)),
            int(sizes[accepted_labels].sum()) if bool(np.any(accepted_labels)) else 0,
            int(sizes[rejected_labels].sum()) if bool(np.any(rejected_labels)) else 0,
        )

    totals = np.zeros((4,), dtype=np.int64)
    worker_count = choose_slice_parallel_workers(int(workers), max(1, int(tile.shape[0])))
    if worker_count <= 1:
        for idx in range(int(tile.shape[0])):
            totals += np.asarray(_process(int(idx)), dtype=np.int64)
    else:
        for values in parallel_map_in_order(
            _process,
            range(int(tile.shape[0])),
            max_workers=int(worker_count),
            max_pending=max(int(worker_count), int(worker_count) * 2),
        ):
            totals += np.asarray(values, dtype=np.int64)

    return {
        'accepted_components': int(totals[0]),
        'rejected_components': int(totals[1]),
        'accepted_voxels': int(totals[2]),
        'rejected_voxels': int(totals[3]),
    }

def _partition_tile_components_2d(
    tile_plane: np.ndarray,
    support_plane: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Partition one tile-local slice into whole accepted and rejected components."""
    tile_bool = np.asarray(tile_plane, dtype=np.uint8) > 0
    support_bool = np.asarray(support_plane, dtype=np.uint8) > 0
    if tuple(int(v) for v in tile_bool.shape) != tuple(int(v) for v in support_bool.shape):
        raise ValueError(
            f'tile/support plane mismatch {tuple(tile_bool.shape)} != {tuple(support_bool.shape)}'
        )
    if not bool(np.any(tile_bool)):
        zeros = np.zeros(tile_bool.shape, dtype=np.uint8)
        return zeros, zeros.copy(), (0, 0, 0, 0)

    num_labels, labels2d, stats, _centroids = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(tile_bool, dtype=np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    num_labels = int(num_labels)
    if num_labels <= 1:
        zeros = np.zeros(tile_bool.shape, dtype=np.uint8)
        return zeros, zeros.copy(), (0, 0, 0, 0)

    sizes = np.asarray(stats[:num_labels, cv2.CC_STAT_AREA], dtype=np.int64)
    present = sizes > 0
    present[0] = False
    hits = np.bincount(labels2d[support_bool], minlength=num_labels)[:num_labels] > 0
    accepted_labels = present & hits
    rejected_labels = present & (~accepted_labels)
    accepted = accepted_labels[labels2d].astype(np.uint8, copy=False)
    rejected = rejected_labels[labels2d].astype(np.uint8, copy=False)
    return accepted, rejected, (
        int(np.count_nonzero(accepted_labels)),
        int(np.count_nonzero(rejected_labels)),
        int(sizes[accepted_labels].sum()) if bool(np.any(accepted_labels)) else 0,
        int(sizes[rejected_labels].sum()) if bool(np.any(rejected_labels)) else 0,
    )

def _or_sparse_tile_crop_into_parent(
    dst_mm: Optional[np.ndarray],
    locks: Optional[Sequence[threading.Lock]],
    *,
    idx: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    crop_u8: np.ndarray,
) -> None:
    if dst_mm is None or not bool(np.any(crop_u8)):
        return
    lock_count = int(len(locks)) if locks else 0
    lock = locks[int(idx) % int(lock_count)] if locks is not None and lock_count > 0 else None
    ctx = lock if lock is not None else contextlib.nullcontext()
    with ctx:
        dst = dst_mm[int(idx), int(y0):int(y1), int(x0):int(x1)]
        if tuple(int(v) for v in dst.shape) != tuple(int(v) for v in crop_u8.shape):
            raise ValueError(
                f'sparse tile OR destination {tuple(dst.shape)} != crop {tuple(crop_u8.shape)}'
            )
        np.bitwise_or(dst, np.asarray(crop_u8, dtype=np.uint8), out=dst)

def gate_raw_bbox_tile_store_against_parent_mask(
    result: TilePostprocessResult,
    *,
    parent_mask_support_mm: object,
    tile_accumulator_mm: np.ndarray,
    tile_accumulator_locks: Optional[Sequence[threading.Lock]],
    work_dir: Path,
    keep_temp: bool,
    slice_workers: int,
    tile_parent_mask_accumulator_mm: Optional[np.ndarray] = None,
    tile_parent_mask_accumulator_locks: Optional[Sequence[threading.Lock]] = None,
) -> TileParentGateResult:
    """Parent-gate a waiting CTILE directly, emitting only a sparse residual CTILE."""
    store = result.tile_mask_store
    if store is None:
        raise ValueError('Sparse parent tile gate requires tile_mask_store')
    py0, py1, px0, px1 = (int(v) for v in result.parent_crop)
    support_shape = _volume_shape_tuple(parent_mask_support_mm)
    if tuple(int(v) for v in tile_accumulator_mm.shape) != support_shape:
        raise ValueError(
            f'Sparse parent tile gate support/accumulator mismatch '
            f'{support_shape} != {tuple(int(v) for v in tile_accumulator_mm.shape)}'
        )
    expected_shape = (int(support_shape[0]), int(py1 - py0), int(px1 - px0))
    if tuple(int(v) for v in store.shape) != expected_shape:
        raise ValueError(
            f'Sparse parent tile gate store shape {store.shape} != {expected_shape}'
        )
    if tile_parent_mask_accumulator_mm is not None and (
        tuple(int(v) for v in tile_parent_mask_accumulator_mm.shape) != support_shape
    ):
        raise ValueError('Sparse parent tile category accumulator shape mismatch')

    residual_dir = (
        Path(work_dir) / result.model_name / result.view_name / result.config_id /
        f'{result.tile_id}.parent_residual.ctile'
    )
    per_slice = np.zeros((int(store.shape[0]), 4), dtype=np.int64)

    def _encode_residual(idx: int) -> RawBBoxSlicePayload:
        decoded = store.decode_slice_crop(int(idx), dtype=np.uint8)
        if decoded is None:
            return RawBBoxSlicePayload(idx=int(idx), is_empty=True)
        local_y0, local_x0, local_y1, local_x1, tile_crop = decoded
        global_y0 = int(py0) + int(local_y0)
        global_y1 = int(py0) + int(local_y1)
        global_x0 = int(px0) + int(local_x0)
        global_x1 = int(px0) + int(local_x1)
        support_slice = _read_binary_volume_slice_crop_bool(
            parent_mask_support_mm, int(idx),
            int(global_y0), int(global_y1), int(global_x0), int(global_x1),
        )
        accepted, rejected, stats = _partition_tile_components_2d(tile_crop, support_slice)
        per_slice[int(idx)] = np.asarray(stats, dtype=np.int64)
        _or_sparse_tile_crop_into_parent(
            tile_accumulator_mm,
            tile_accumulator_locks,
            idx=int(idx),
            y0=int(global_y0), y1=int(global_y1),
            x0=int(global_x0), x1=int(global_x1),
            crop_u8=accepted,
        )
        _or_sparse_tile_crop_into_parent(
            tile_parent_mask_accumulator_mm,
            tile_parent_mask_accumulator_locks,
            idx=int(idx),
            y0=int(global_y0), y1=int(global_y1),
            x0=int(global_x0), x1=int(global_x1),
            crop_u8=accepted,
        )
        payload = _encode_bool_mask_slice_payload(int(idx), rejected)
        if bool(payload.is_empty):
            return payload
        return dataclasses_replace(
            payload,
            y0=int(payload.y0) + int(local_y0),
            y1=int(payload.y1) + int(local_y0),
            x0=int(payload.x0) + int(local_x0),
            x1=int(payload.x1) + int(local_x0),
        )

    try:
        residual_stats = _write_raw_bbox_payload_store(
            shape=tuple(int(v) for v in store.shape),
            store_dir=residual_dir,
            encode_slice=_encode_residual,
            format_name=CTILE_FORMAT,
            desc=(
                f'Parent-failed tile residual {result.model_name}/'
                f'{result.view_name}/{result.tile_id}'
            ),
            workers=int(slice_workers),
            extra_meta={
                'parent_crop': [int(py0), int(py1), int(px0), int(px1)],
                'tta_aug_id': str(result.aug_id),
                'tta_angle_deg': float(result.angle_deg),
                'source_tile_store': str(store.root),
                'gate': 'parent_mask',
            },
        )
    finally:
        if bool(keep_temp):
            store.close()
        else:
            store.unlink()

    totals = np.sum(per_slice, axis=0, dtype=np.int64)
    gate_stats = {
        'accepted_components': int(totals[0]),
        'rejected_components': int(totals[1]),
        'accepted_voxels': int(totals[2]),
        'rejected_voxels': int(totals[3]),
    }
    residual_result: Optional[TilePostprocessResult] = None
    if int(gate_stats['rejected_voxels']) > 0:
        residual_store = RawBBoxMaskStore.open(residual_dir, mmap_payload=True)
        residual_result = TilePostprocessResult(
            model_name=str(result.model_name),
            view_name=str(result.view_name),
            aug_id=str(result.aug_id),
            angle_deg=float(result.angle_deg),
            config_id=str(result.config_id),
            tile_id=str(result.tile_id),
            parent_crop=tuple(int(v) for v in result.parent_crop),
            tile_mask_mm=None,
            tile_mask_path=residual_dir,
            tile_mask_store=residual_store,
        )
        runtime_telemetry().add(
            'tile.parent_residual_ctile_payload_bytes',
            int(residual_stats.get('raw_payload_bytes', 0)),
        )
    else:
        release_memfd_owners_under(residual_dir)
        shutil.rmtree(residual_dir, ignore_errors=True)

    return TileParentGateResult(
        model_name=str(result.model_name),
        view_name=str(result.view_name),
        aug_id=str(result.aug_id),
        angle_deg=float(result.angle_deg),
        config_id=str(result.config_id),
        tile_id=str(result.tile_id),
        gate_stats=gate_stats,
        residual_result=residual_result,
    )

def gate_raw_bbox_tile_store_against_parent_bridge(
    result: TilePostprocessResult,
    *,
    parent_bridge_support_mm: object,
    tile_accumulator_mm: np.ndarray,
    tile_accumulator_locks: Optional[Sequence[threading.Lock]],
    keep_temp: bool,
    slice_workers: int,
    tile_parent_bridge_accumulator_mm: Optional[np.ndarray] = None,
    tile_parent_bridge_accumulator_locks: Optional[Sequence[threading.Lock]] = None,
) -> TileGateResult:
    """Bridge-gate a sparse parent residual without reconstructing a dense volume."""
    store = result.tile_mask_store
    if store is None:
        raise ValueError('Sparse bridge tile gate requires tile_mask_store')
    py0, py1, px0, px1 = (int(v) for v in result.parent_crop)
    support_shape = _volume_shape_tuple(parent_bridge_support_mm)
    if tuple(int(v) for v in tile_accumulator_mm.shape) != support_shape:
        raise ValueError('Sparse bridge tile gate support/accumulator shape mismatch')
    expected_shape = (int(support_shape[0]), int(py1 - py0), int(px1 - px0))
    if tuple(int(v) for v in store.shape) != expected_shape:
        raise ValueError(
            f'Sparse bridge tile gate store shape {store.shape} != {expected_shape}'
        )
    if tile_parent_bridge_accumulator_mm is not None and (
        tuple(int(v) for v in tile_parent_bridge_accumulator_mm.shape) != support_shape
    ):
        raise ValueError('Sparse bridge tile category accumulator shape mismatch')

    per_slice = np.zeros((int(store.shape[0]), 4), dtype=np.int64)

    def _process(idx: int) -> None:
        decoded = store.decode_slice_crop(int(idx), dtype=np.uint8)
        if decoded is None:
            return
        local_y0, local_x0, local_y1, local_x1, tile_crop = decoded
        global_y0 = int(py0) + int(local_y0)
        global_y1 = int(py0) + int(local_y1)
        global_x0 = int(px0) + int(local_x0)
        global_x1 = int(px0) + int(local_x1)
        support_slice = _read_binary_volume_slice_crop_bool(
            parent_bridge_support_mm, int(idx),
            int(global_y0), int(global_y1), int(global_x0), int(global_x1),
        )
        accepted, _rejected, stats = _partition_tile_components_2d(tile_crop, support_slice)
        per_slice[int(idx)] = np.asarray(stats, dtype=np.int64)
        _or_sparse_tile_crop_into_parent(
            tile_accumulator_mm,
            tile_accumulator_locks,
            idx=int(idx),
            y0=int(global_y0), y1=int(global_y1),
            x0=int(global_x0), x1=int(global_x1),
            crop_u8=accepted,
        )
        _or_sparse_tile_crop_into_parent(
            tile_parent_bridge_accumulator_mm,
            tile_parent_bridge_accumulator_locks,
            idx=int(idx),
            y0=int(global_y0), y1=int(global_y1),
            x0=int(global_x0), x1=int(global_x1),
            crop_u8=accepted,
        )

    try:
        parallel_for_indices_chunked(
            int(store.shape[0]),
            _process,
            max_workers=choose_slice_parallel_workers(
                int(slice_workers), max(1, int(store.shape[0])),
            ),
            desc=(
                f'Sparse parent-bridge tile gate '
                f'{result.model_name}/{result.view_name}/{result.tile_id}'
            ),
            show_progress=False,
            target_chunks_per_worker=2,
        )
    finally:
        if bool(keep_temp):
            store.close()
        else:
            store.unlink()

    totals = np.sum(per_slice, axis=0, dtype=np.int64)
    return TileGateResult(
        model_name=str(result.model_name),
        view_name=str(result.view_name),
        aug_id=str(result.aug_id),
        angle_deg=float(result.angle_deg),
        config_id=str(result.config_id),
        tile_id=str(result.tile_id),
        gate_stats={
            'accepted_components': int(totals[0]),
            'rejected_components': int(totals[1]),
            'accepted_voxels': int(totals[2]),
            'rejected_voxels': int(totals[3]),
        },
    )

def postprocess_tile_volume_after_inference(
    task: TilePostprocessTask,
    *,
    view: ViewInfo,
    min_conf: float,
    min_radius: float,
    keep_temp: bool,
    slice_workers: int,
    sparse_retire_dir: Optional[Path] = None,
) -> Optional[TilePostprocessResult | DeferredTilePostprocessResult]:
    py0, py1, px0, px1 = (int(v) for v in task.parent_crop)
    expected_shape = (
        int(view.num_slices),
        int(py1 - py0),
        int(px1 - px0),
    )
    actual_shape = tuple(int(v) for v in np.asarray(task.tile_mask_mm).shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f'{task.model_name}/{task.view_name}/{task.tile_id}: crop-local tile mask '
            f'shape {actual_shape} does not match parent crop {task.parent_crop} -> {expected_shape}'
        )
    if task.processing_shape is not None and tuple(int(v) for v in task.processing_shape) != expected_shape:
        raise ValueError(
            f'{task.model_name}/{task.view_name}/{task.tile_id}: declared tile processing '
            f'shape {task.processing_shape} does not match {expected_shape}'
        )

    cleanup_view_volume_after_prediction_inplace(
        task.tile_mask_mm,
        task.tile_confmap_mm,
        view,
        float(min_conf),
        float(min_radius),
        workers=int(slice_workers),
        precleaned_slice_cleanup=bool(task.precleaned_slice_cleanup),
        threshold_plane_shape=task.threshold_plane_shape,
    )

    close_memmap_array(task.tile_confmap_mm)
    if task.tile_confmap_path is not None and not keep_temp:
        try:
            task.tile_confmap_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not _volume_has_foreground(task.tile_mask_mm):
        close_memmap_array(task.tile_mask_mm)
        if not keep_temp:
            try:
                task.tile_mask_path.unlink(missing_ok=True)
            except Exception:
                pass
        return None

    dense_result = TilePostprocessResult(
        model_name=str(task.model_name),
        view_name=str(task.view_name),
        aug_id=str(task.aug_id),
        angle_deg=float(task.angle_deg),
        config_id=str(task.config_id),
        tile_id=str(task.tile_id),
        parent_crop=tuple(int(v) for v in task.parent_crop),
        tile_mask_mm=task.tile_mask_mm,
        tile_mask_path=task.tile_mask_path,
    )
    if sparse_retire_dir is not None and not bool(keep_temp):
        # v16.4.3: the GPU-worker dense result is only a cleanup workspace.  Convert the
        # cleaned crop immediately to CTILE and close/unlink the uint8 .dat before parent
        # interpolation or either component gate can delay its retirement.
        return spill_waiting_tile_result_to_raw_store(
            dense_result,
            Path(sparse_retire_dir),
            workers=int(slice_workers),
            keep_original=False,
        )
    return dense_result

def spill_waiting_tile_result_to_raw_store(
    result: TilePostprocessResult,
    temp_dir: Path,
    *,
    workers: int = 1,
    keep_original: bool = False,
) -> DeferredTilePostprocessResult:
    """Spill one crop-local tile or residual without expanding it to the parent canvas."""
    if result.tile_mask_mm is None:
        raise ValueError('Cannot spill a tile result without a dense tile_mask_mm')
    tile_arr = np.asarray(result.tile_mask_mm)
    if tile_arr.ndim != 3:
        raise ValueError(f'Waiting tile spill expects a 3-D mask volume, got {tile_arr.shape}')
    py0, py1, px0, px1 = (int(v) for v in result.parent_crop)
    expected_plane = (int(py1 - py0), int(px1 - px0))
    if tuple(int(v) for v in tile_arr.shape[1:]) != expected_plane:
        raise ValueError(
            f'Waiting tile {result.tile_id}: crop-local shape {tuple(tile_arr.shape[1:])} '
            f'does not match parent crop {result.parent_crop}'
        )

    store_dir = (
        temp_dir / 'waiting_tiles' / result.model_name / result.view_name /
        result.config_id / f'{result.tile_id}.ctile'
    )
    store_stats = write_raw_bbox_mask_store(
        tile_arr,
        store_dir,
        format_name=CTILE_FORMAT,
        desc=(
            f'Waiting crop-local tile raw bbox store '
            f'{result.model_name}/{result.view_name}/{result.config_id}/{result.tile_id}'
        ),
        workers=int(workers),
        extra_meta={
            'waiting_tile_id': str(result.tile_id),
            'parent_crop': [int(py0), int(py1), int(px0), int(px1)],
            'tta_aug_id': str(result.aug_id),
            'tta_angle_deg': float(result.angle_deg),
        },
    )
    runtime_telemetry().add(
        'tile.ctile_logical_dense_bytes_retired',
        int(store_stats.get('logical_raw_uint8_bytes', int(tile_arr.nbytes))),
    )
    runtime_telemetry().add(
        'tile.ctile_payload_bytes', int(store_stats.get('raw_payload_bytes', 0)),
    )

    tile_shape = tuple(int(x) for x in tile_arr.shape)
    close_memmap_array(result.tile_mask_mm)
    if not bool(keep_original) and result.tile_mask_path is not None:
        try:
            Path(result.tile_mask_path).unlink(missing_ok=True)
        except Exception:
            pass

    return DeferredTilePostprocessResult(
        model_name=str(result.model_name),
        view_name=str(result.view_name),
        aug_id=str(result.aug_id),
        angle_deg=float(result.angle_deg),
        config_id=str(result.config_id),
        tile_id=str(result.tile_id),
        parent_crop=tuple(int(v) for v in result.parent_crop),
        tile_mask_path=store_dir,
        tile_shape=(int(tile_shape[0]), int(tile_shape[1]), int(tile_shape[2])),
        storage_format=CTILE_FORMAT,
    )

def load_waiting_tile_result_from_raw_store(waiting: DeferredTilePostprocessResult) -> TilePostprocessResult:
    if str(waiting.storage_format) != CTILE_FORMAT:
        raise ValueError(f'Unsupported waiting tile storage format: {waiting.storage_format}')
    store = RawBBoxMaskStore.open(waiting.tile_mask_path, mmap_payload=True)
    if tuple(int(x) for x in store.shape) != tuple(int(x) for x in waiting.tile_shape):
        store.close()
        raise ValueError(f'Raw tile store shape mismatch: store={store.shape}, expected={waiting.tile_shape}')
    return TilePostprocessResult(
        model_name=str(waiting.model_name),
        view_name=str(waiting.view_name),
        aug_id=str(waiting.aug_id),
        angle_deg=float(waiting.angle_deg),
        config_id=str(waiting.config_id),
        tile_id=str(waiting.tile_id),
        parent_crop=tuple(int(v) for v in waiting.parent_crop),
        tile_mask_mm=None,
        tile_mask_path=waiting.tile_mask_path,
        tile_mask_store=store,
    )

def defer_open_tile_result_store(result: TilePostprocessResult) -> DeferredTilePostprocessResult:
    """Drop an open CTILE mapping while retaining only its lightweight descriptor."""
    store = result.tile_mask_store
    if store is None or result.tile_mask_path is None:
        raise ValueError('Cannot defer a tile result without an open raw tile store')
    tile_shape = tuple(int(v) for v in store.shape)
    store.close()
    result.tile_mask_store = None
    return DeferredTilePostprocessResult(
        model_name=str(result.model_name),
        view_name=str(result.view_name),
        aug_id=str(result.aug_id),
        angle_deg=float(result.angle_deg),
        config_id=str(result.config_id),
        tile_id=str(result.tile_id),
        parent_crop=tuple(int(v) for v in result.parent_crop),
        tile_mask_path=Path(result.tile_mask_path),
        tile_shape=(int(tile_shape[0]), int(tile_shape[1]), int(tile_shape[2])),
        storage_format=CTILE_FORMAT,
    )

def _delete_tile_result_storage(
    result: TilePostprocessResult | DeferredTilePostprocessResult,
    *,
    keep_temp: bool,
) -> None:
    if isinstance(result, DeferredTilePostprocessResult):
        if bool(keep_temp):
            return
        path = Path(result.tile_mask_path)
        release_memfd_owners_under(path)
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception:
            pass
        return
    if result.tile_mask_store is not None:
        if bool(keep_temp):
            result.tile_mask_store.close()
        else:
            result.tile_mask_store.unlink()
        return
    if not bool(keep_temp) and result.tile_mask_path is not None:
        try:
            Path(result.tile_mask_path).unlink(missing_ok=True)
        except Exception:
            pass

def gate_tile_result_against_parent_mask(
    result: TilePostprocessResult | DeferredTilePostprocessResult,
    *,
    parent_mask_support_mm: object,
    tile_accumulator_mm: np.ndarray,
    tile_accumulator_locks: Optional[Sequence[threading.Lock]],
    work_dir: Path,
    keep_temp: bool,
    slice_workers: int,
    tile_parent_mask_accumulator_mm: Optional[np.ndarray] = None,
    tile_parent_mask_accumulator_locks: Optional[Sequence[threading.Lock]] = None,
) -> TileParentGateResult:
    """Accept parent-supported components and retain failed components as one residual.

    Waiting CTILE inputs remain sparse throughout the gate. Fresh worker results are also
    sparse-retired at the cleanup boundary in v16.4.3, so deferred descriptors are opened
    only inside the gate worker and never reconstruct a dense crop-local volume.
    """
    if isinstance(result, DeferredTilePostprocessResult):
        result = load_waiting_tile_result_from_raw_store(result)
    dense_tile = result.tile_mask_mm
    if dense_tile is None:
        if result.tile_mask_store is None:
            raise ValueError('Parent tile gate requires a dense mask or raw tile store')
        return gate_raw_bbox_tile_store_against_parent_mask(
            result,
            parent_mask_support_mm=parent_mask_support_mm,
            tile_accumulator_mm=tile_accumulator_mm,
            tile_accumulator_locks=tile_accumulator_locks,
            work_dir=Path(work_dir),
            keep_temp=bool(keep_temp),
            slice_workers=int(slice_workers),
            tile_parent_mask_accumulator_mm=tile_parent_mask_accumulator_mm,
            tile_parent_mask_accumulator_locks=tile_parent_mask_accumulator_locks,
        )

    gate_stats = gate_tile_components_against_support_inplace(
        dense_tile,
        parent_mask_support_mm,
        parent_crop=tuple(int(v) for v in result.parent_crop),
        accepted_total_mm=tile_accumulator_mm,
        accepted_total_locks=tile_accumulator_locks,
        accepted_category_mm=tile_parent_mask_accumulator_mm,
        accepted_category_locks=tile_parent_mask_accumulator_locks,
        retain_rejected_components=True,
        workers=int(slice_workers),
        desc=f'Parent-YOLO tile gate {result.model_name}/{result.view_name}/{result.tile_id}',
    )

    residual_result: Optional[TilePostprocessResult]
    if int(gate_stats.get('rejected_voxels', 0)) > 0:
        residual_result = TilePostprocessResult(
            model_name=str(result.model_name),
            view_name=str(result.view_name),
            aug_id=str(result.aug_id),
            angle_deg=float(result.angle_deg),
            config_id=str(result.config_id),
            tile_id=str(result.tile_id),
            parent_crop=tuple(int(v) for v in result.parent_crop),
            tile_mask_mm=dense_tile,
            tile_mask_path=result.tile_mask_path,
            tile_mask_store=None,
        )
    else:
        close_memmap_array(dense_tile)
        if result.tile_mask_path is not None and not bool(keep_temp):
            try:
                Path(result.tile_mask_path).unlink(missing_ok=True)
            except Exception:
                pass
        residual_result = None

    return TileParentGateResult(
        model_name=str(result.model_name),
        view_name=str(result.view_name),
        aug_id=str(result.aug_id),
        angle_deg=float(result.angle_deg),
        config_id=str(result.config_id),
        tile_id=str(result.tile_id),
        gate_stats={k: int(v) for k, v in gate_stats.items()},
        residual_result=residual_result,
    )

def gate_tile_residual_against_parent_bridge(
    result: TilePostprocessResult | DeferredTilePostprocessResult,
    *,
    parent_bridge_support_mm: object,
    tile_accumulator_mm: np.ndarray,
    tile_accumulator_locks: Optional[Sequence[threading.Lock]],
    work_dir: Path,
    keep_temp: bool,
    slice_workers: int,
    tile_parent_bridge_accumulator_mm: Optional[np.ndarray] = None,
    tile_parent_bridge_accumulator_locks: Optional[Sequence[threading.Lock]] = None,
) -> TileGateResult:
    """Re-gate only parent-failed components against immutable parent bridge support.

    A deferred sparse residual CTILE is opened inside this gate worker, consumed slice-by-slice,
    and retired directly; it is never expanded back into a dense crop-local volume.
    """
    if isinstance(result, DeferredTilePostprocessResult):
        result = load_waiting_tile_result_from_raw_store(result)
    dense_tile = result.tile_mask_mm
    if dense_tile is None:
        if result.tile_mask_store is None:
            raise ValueError('Bridge tile gate requires a dense residual or raw tile store')
        return gate_raw_bbox_tile_store_against_parent_bridge(
            result,
            parent_bridge_support_mm=parent_bridge_support_mm,
            tile_accumulator_mm=tile_accumulator_mm,
            tile_accumulator_locks=tile_accumulator_locks,
            keep_temp=bool(keep_temp),
            slice_workers=int(slice_workers),
            tile_parent_bridge_accumulator_mm=tile_parent_bridge_accumulator_mm,
            tile_parent_bridge_accumulator_locks=tile_parent_bridge_accumulator_locks,
        )

    try:
        gate_stats = gate_tile_components_against_support_inplace(
            dense_tile,
            parent_bridge_support_mm,
            parent_crop=tuple(int(v) for v in result.parent_crop),
            accepted_total_mm=tile_accumulator_mm,
            accepted_total_locks=tile_accumulator_locks,
            accepted_category_mm=tile_parent_bridge_accumulator_mm,
            accepted_category_locks=tile_parent_bridge_accumulator_locks,
            retain_rejected_components=False,
            workers=int(slice_workers),
            desc=f'Parent-bridge tile gate {result.model_name}/{result.view_name}/{result.tile_id}',
        )
        return TileGateResult(
            model_name=str(result.model_name),
            view_name=str(result.view_name),
            aug_id=str(result.aug_id),
            angle_deg=float(result.angle_deg),
            config_id=str(result.config_id),
            tile_id=str(result.tile_id),
            gate_stats={k: int(v) for k, v in gate_stats.items()},
        )
    finally:
        close_memmap_array(dense_tile)
        if result.tile_mask_path is not None and not bool(keep_temp):
            try:
                Path(result.tile_mask_path).unlink(missing_ok=True)
            except Exception:
                pass

def finalize_consolidated_tile_volume_for_parent(
    *,
    model_name: str,
    view: ViewInfo,
    tile_accumulator_mm: np.ndarray,
    destination_mm: np.ndarray,
    destination_lock: threading.Lock,
    temp_dir: Path,
    interpolate: int,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_passes: int,
    interpolate_min_radius: float,
    interpolation_search_angle: float,
    keep_temp: bool,
    slice_workers: int,
    interpolation_task_workers: int,
    nrrd_layers_enabled: bool = False,
    tile_parent_mask_accumulator_mm: Optional[np.ndarray] = None,
    tile_parent_bridge_accumulator_mm: Optional[np.ndarray] = None,
    internal_final_layer_enabled: bool = False,
    config_id: str = '',
) -> TileConsolidationResult:
    """Interpolate one configuration's consolidated gated tiles, then union them.

 The input accumulator contains the OR of every accepted tile mask for one tile configuration
 of this parent view. Interpolation is performed once per configuration instead of once per tile
 or once across unrelated configurations. When NRRD decomposition is enabled, the accepted YOLO
 tile support is written separately for tiles accepted by parent YOLO masks and by parent
 interpolation bridges; tile interpolation bridges are then exported per configuration/pass."""
    # Local import keeps the package dependency graph acyclic.
    from .finalization import union_volume_into_volume

    interpolation_stats: List[Dict[str, object]] = []
    nrrd_layers: List[NrrdLayerRef] = []
    config_id_norm = str(config_id).strip()
    config_label = config_id_norm or 'consolidated'
    pre_interpolation_stage = (
        f'{config_id_norm}_pre_tile_interpolation'
        if config_id_norm else 'pre_tile_interpolation'
    )
    interpolation_stage = (
        f'{config_id_norm}_tile_interpolation'
        if config_id_norm else 'tile_interpolation'
    )
    tile_plane_shape = tuple(int(v) for v in np.asarray(tile_accumulator_mm).shape[-2:])
    effective_interpolate_min_radius = view_processing_min_radius(
        view, float(interpolate_min_radius), tile_plane_shape,
    )
    effective_interpolation_search_angle = view_processing_search_angle(
        view, float(interpolation_search_angle), tile_plane_shape,
    )

    if not _volume_has_foreground(tile_accumulator_mm):
        if bool(internal_final_layer_enabled):
            internal_ref = materialize_internal_final_view_layer(
                destination_mm,
                model_name=str(model_name),
                view=view,
                temp_dir=temp_dir,
                workers=int(slice_workers),
            )
            if internal_ref is not None:
                nrrd_layers.append(internal_ref)
        return TileConsolidationResult(
            model_name=str(model_name),
            view_name=str(view.name),
            aug_id=str(view.tta_aug_id),
            angle_deg=float(view.tta_angle_deg),
            interpolation_stats=interpolation_stats,
            nrrd_layers=nrrd_layers,
            final_accumulator_mm=tile_accumulator_mm,
        )

    if bool(nrrd_layers_enabled):
        emitted_category = False
        if tile_parent_mask_accumulator_mm is not None:
            layer_ref = materialize_nrrd_view_layer(
                tile_parent_mask_accumulator_mm,
                model_name=str(model_name),
                view=view,
                source='tile',
                mask_kind='yolo',
                pass_index=0,
                tile_config_id=config_id_norm,
                tile_acceptance='parent_mask',
                stage=pre_interpolation_stage,
                description=f'Accepted {config_label} tile YOLO masks whose components intersected parent full-frame YOLO support. Parent-mask support has priority when a component intersects both parent mask and parent bridge.',
                temp_dir=temp_dir,
                workers=int(slice_workers),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)
                emitted_category = True
        if tile_parent_bridge_accumulator_mm is not None:
            layer_ref = materialize_nrrd_view_layer(
                tile_parent_bridge_accumulator_mm,
                model_name=str(model_name),
                view=view,
                source='tile',
                mask_kind='yolo',
                pass_index=0,
                tile_config_id=config_id_norm,
                tile_acceptance='parent_bridge',
                stage=pre_interpolation_stage,
                description=f'Accepted {config_label} tile YOLO masks whose components did not intersect parent YOLO support but did intersect a parent interpolation bridge.',
                temp_dir=temp_dir,
                workers=int(slice_workers),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)
                emitted_category = True
        if not emitted_category:
            layer_ref = materialize_nrrd_view_layer(
                tile_accumulator_mm,
                model_name=str(model_name),
                view=view,
                source='tile',
                mask_kind='yolo',
                pass_index=0,
                tile_config_id=config_id_norm,
                tile_acceptance='parent_support',
                stage=pre_interpolation_stage,
                description=f'Accepted {config_label} tile YOLO masks before tile interpolation. Parent mask/bridge category supports were unavailable, so the category is the total parent support.',
                temp_dir=temp_dir,
                workers=int(slice_workers),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)


    if _view_uses_interpolation(view, int(interpolate)):
        total_passes = int(interpolate_passes)
        for pass_idx in range(1, total_passes + 1):
            # the pass writes the exact added-voxel delta itself,
            # replacing the old full-volume before-copy + subtract bookkeeping.
            pass_component_dir: Optional[Path] = None
            if (
                bool(nrrd_layers_enabled)
                and int(interpolation_walk_back) > 0
                and int(interpolation_candidates) > 0
            ):
                pass_component_dir = (
                    temp_dir / 'nrrd_work' / view.name / config_label /
                    f'tile_bridge_pass{int(pass_idx):02d}_components'
                )

            tile_accumulator_mm, stats_local = interpolate_view_volume_pass_maybe_process(
                mask_mm=tile_accumulator_mm,
                view=view,
                work_dir=(
                    temp_dir / 'tile_interpolation' / str(model_name) /
                    view.name / config_label
                ),
                pass_tag=f'pass{pass_idx}',
                max_slice_distance=int(interpolate),
                search_angle_deg=float(effective_interpolation_search_angle),
                interpolation_walk_back=int(interpolation_walk_back),
                interpolation_candidates=int(interpolation_candidates),
                interpolate_min_radius=float(effective_interpolate_min_radius),
                keep_temp=bool(keep_temp),
                prefer_memory=True,
                workers=int(interpolation_task_workers),
                bridge_component_dir=pass_component_dir,
            )
            stats_local = dict(stats_local)
            stats_local.update({
                'pass_index': int(pass_idx),
                'model': str(model_name),
                'view': f'{view.name}[tiles:{config_label}]',
                'source': 'tile',
                'tile_config_id': config_id_norm,
                'max_slice_distance': int(interpolate),
                'interpolation_walk_back': int(interpolation_walk_back),
                'interpolation_candidates': int(interpolation_candidates),
                'interpolation_search_angle': float(interpolation_search_angle),
                'processing_interpolation_search_angle': float(effective_interpolation_search_angle),
                'processing_interpolate_min_radius': float(effective_interpolate_min_radius),
            })
            interpolation_stats.append(stats_local)

            if pass_component_dir is not None:
                component_entries = [
                    dict(entry)
                    for entry in stats_local.get('bridge_component_deltas', [])
                ]
                expected_components = int(interpolation_walk_back) * int(interpolation_candidates)
                if len(component_entries) != int(expected_components):
                    raise RuntimeError(
                        f'{model_name}/{view.name}/{config_label} interpolation pass '
                        f'{int(pass_idx)} returned {len(component_entries)} component delta(s); '
                        f'expected {int(interpolation_walk_back)} x '
                        f'{int(interpolation_candidates)} = {int(expected_components)}'
                    )
                component_entries.sort(key=lambda entry: (
                    int(entry.get('walk_back_index', 0)),
                    int(entry.get('candidate_index', 0)),
                ))
                for component_entry in component_entries:
                    walk_back_index = int(component_entry['walk_back_index'])
                    candidate_index = int(component_entry['candidate_index'])
                    layer_ref = materialize_interpolation_component_nrrd_view_layer(
                        Path(str(component_entry['path'])),
                        added_voxels=int(component_entry.get('added_voxels', 0)),
                        model_name=str(model_name),
                        view=view,
                        source='tile',
                        pass_index=int(pass_idx),
                        interpolation_walk_back_index=int(walk_back_index),
                        interpolation_candidate_index=int(candidate_index),
                        tile_config_id=config_id_norm,
                        tile_acceptance='consolidated',
                        stage=interpolation_stage,
                        description=(
                            f'Voxels added by this {config_label} tile interpolation pass '
                            f'for walk-back origin {int(walk_back_index)} and candidate '
                            f'{int(candidate_index)} only. Bridges are generated after '
                            'accepted tile masks within this configuration are consolidated, '
                            'so they are not attributed back to parent-mask vs parent-bridge '
                            'acceptance categories.'
                        ),
                        temp_dir=temp_dir,
                        workers=int(slice_workers),
                        keep_temp=bool(keep_temp),
                    )
                    nrrd_layers.append(layer_ref)


            if int(stats_local.get('added_voxels', 0)) <= 0:
                break

    with destination_lock:
        union_volume_into_volume(
            destination_mm,
            tile_accumulator_mm,
            workers=int(slice_workers),
            desc=f'Union consolidated gated tiles {model_name}/{view.name}/{config_label}',
        )

    if bool(internal_final_layer_enabled):
        internal_ref = materialize_internal_final_view_layer(
            destination_mm,
            model_name=str(model_name),
            view=view,
            temp_dir=temp_dir,
            workers=int(slice_workers),
        )
        if internal_ref is not None:
            nrrd_layers.append(internal_ref)

    return TileConsolidationResult(
        model_name=str(model_name),
        view_name=str(view.name),
        aug_id=str(view.tta_aug_id),
        angle_deg=float(view.tta_angle_deg),
        interpolation_stats=interpolation_stats,
        nrrd_layers=nrrd_layers,
        final_accumulator_mm=tile_accumulator_mm,
    )

def finalize_parent_without_tile_contribution_for_sparse_retirement(
    *,
    model_name: str,
    view: ViewInfo,
    destination_mm: np.ndarray,
    destination_lock: threading.Lock,
    temp_dir: Path,
    slice_workers: int,
) -> TileConsolidationResult:
    """Materialize the private final-view ref when every original tile was empty."""
    layers: List[NrrdLayerRef] = []
    with destination_lock:
        internal_ref = materialize_internal_final_view_layer(
            destination_mm,
            model_name=str(model_name),
            view=view,
            temp_dir=temp_dir,
            workers=int(slice_workers),
        )
    if internal_ref is not None:
        layers.append(internal_ref)
    return TileConsolidationResult(
        model_name=str(model_name),
        view_name=str(view.name),
        aug_id=str(view.tta_aug_id),
        angle_deg=float(view.tta_angle_deg),
        interpolation_stats=[],
        nrrd_layers=layers,
        final_accumulator_mm=None,
    )

def gaussian_smoothing_gpu_enabled() -> bool:
    return _env_flag('YOLO_TTA_GAUSSIAN_SMOOTHING_GPU', True)

def gaussian_smoothing_gpu_chunk_mib() -> int:
    return max(256, _env_int('YOLO_TTA_GAUSSIAN_GPU_CHUNK_MIB', 16384))

def _try_get_gpu_gaussian_backend() -> Optional[Tuple[object, object]]:
    try:
        import cupy as cp  # type: ignore
        from cupyx.scipy import ndimage as cpx_ndi  # type: ignore
    except Exception:
        return None
    if not hasattr(cpx_ndi, 'gaussian_filter'):
        return None
    return cp, cpx_ndi

def _gaussian_gpu_core_depth(shape_tyx: Tuple[int, int, int], sigma: float, truncate: float = 4.0) -> Tuple[int, int]:
    z_dim, h_dim, w_dim = (int(shape_tyx[0]), int(shape_tyx[1]), int(shape_tyx[2]))
    halo = max(0, int(math.ceil(float(truncate) * float(sigma))))
    target_bytes = int(gaussian_smoothing_gpu_chunk_mib()) * 1024 * 1024
    bytes_per_slice = max(1, int(h_dim) * int(w_dim) * np.dtype(np.float32).itemsize)
    # cupyx.ndimage may allocate temporaries; budget input/output plus one scratch-equivalent.
    max_read_slices = max(1, int(target_bytes // max(1, bytes_per_slice * 3)))
    core_depth = max(1, int(max_read_slices) - 2 * int(halo))
    return min(max(1, int(core_depth)), max(1, int(z_dim))), int(halo)

def _try_apply_gaussian_smoothing_gpu_chunked_inplace(
    mask_mm: np.ndarray,
    sigma: float,
    passes: int,
    temp_dir: Path,
    *,
    stats: Dict[str, object],
    keep_temp: bool = False,
    workers: int = 1,
    nrrd_layers: Optional[List[NrrdLayerRef]] = None,
    nrrd_model_name: str = 'global',
) -> Optional[Dict[str, object]]:
    """Apply Gaussian smoothing with a CuPy/cupyx chunk+halo implementation.

 Chunks are split along the t/slice axis and include a halo of ceil(truncate*sigma)
 slices on both sides. Each GPU result writes only its core region back to the
 CPU-backed volume, which makes the chunked result match a whole-volume filter for
 the unchunked Y/X axes and removes seams along the chunked axis."""
    if not gaussian_smoothing_gpu_enabled():
        return None
    backend = _try_get_gpu_gaussian_backend()
    if backend is None:
        print('Warning: Gaussian smoothing GPU backend unavailable (CuPy/cupyx.ndimage); falling back to CPU scipy.ndimage.')
        return None
    cp, cpx_ndi = backend

    sigma_f = float(sigma)
    passes_i = int(passes)
    if sigma_f <= 0.0 or passes_i <= 0:
        return stats

    smooth_dir = temp_dir / 'gaussian_smoothing'
    smooth_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(x) for x in np.asarray(mask_mm).shape)
    if len(shape) != 3:
        raise ValueError(f'Gaussian smoothing expects a 3D volume, got shape {shape}')
    num_slices = int(shape[0])
    core_depth, halo = _gaussian_gpu_core_depth(shape, sigma_f, truncate=4.0)
    stats['backend'] = 'cupy_chunked_gpu'
    stats['chunk_core_slices'] = int(core_depth)
    stats['halo_slices'] = int(halo)
    print(
        'Gaussian smoothing GPU backend active: '
        f'shape={shape}, sigma={sigma_f:g}, passes={passes_i}, core_slices={core_depth}, '
        f'halo={halo}, chunk_budget={gaussian_smoothing_gpu_chunk_mib()} MiB'
    )

    for pass_idx in range(1, passes_i + 1):
        print(f'Gaussian smoothing GPU pass {int(pass_idx)}/{int(passes_i)} (sigma={sigma_f:g} voxels)')
        source_path = smooth_dir / f'pass{int(pass_idx):02d}_source.u8.dat'
        source_mm = copy_workspace_array(
            np.asarray(mask_mm, dtype=np.uint8),
            source_path,
            desc=f'Gaussian smoothing GPU pass {int(pass_idx)} source snapshot',
            prefer_memory=False,
            workers=int(workers),
        )
        added_by_slice = np.zeros((num_slices,), dtype=np.int64)
        removed_by_slice = np.zeros((num_slices,), dtype=np.int64)
        try:
            ranges = [(int(z0), int(min(num_slices, z0 + core_depth))) for z0 in range(0, num_slices, core_depth)]
            for z0, z1 in tqdm(ranges, desc=f'Gaussian smoothing GPU pass {int(pass_idx)}: chunked filter'):
                read0 = max(0, int(z0) - int(halo))
                read1 = min(num_slices, int(z1) + int(halo))
                core0 = int(z0) - int(read0)
                core1 = core0 + int(z1 - z0)
                chunk_cpu = np.ascontiguousarray(np.asarray(source_mm[int(read0):int(read1)], dtype=np.float32))
                chunk_gpu = cp.asarray(chunk_cpu)
                smoothed_gpu = cpx_ndi.gaussian_filter(
                    chunk_gpu,
                    sigma=float(sigma_f),
                    mode='constant',
                    cval=0.0,
                    truncate=4.0,
                )
                core_bool = cp.asnumpy(smoothed_gpu[int(core0):int(core1)] >= 0.5).astype(bool, copy=False)
                old_bool = np.asarray(source_mm[int(z0):int(z1)], dtype=bool)
                for local_z in range(int(z1 - z0)):
                    new = core_bool[int(local_z)]
                    old = old_bool[int(local_z)]
                    added_by_slice[int(z0) + int(local_z)] = np.int64(np.count_nonzero(new & (~old)))
                    removed_by_slice[int(z0) + int(local_z)] = np.int64(np.count_nonzero(old & (~new)))
                mask_mm[int(z0):int(z1), :, :] = core_bool.astype(np.uint8, copy=False)
                try:
                    del chunk_gpu, smoothed_gpu, core_bool, chunk_cpu
                    free_all = getattr(cp.get_default_memory_pool(), 'free_all_blocks', None)
                    if callable(free_all):
                        free_all()
                except Exception:
                    pass
        finally:
            close_memmap_array(source_mm)
            if not bool(keep_temp):
                try:
                    source_path.unlink(missing_ok=True)
                except Exception:
                    pass

        flush_array(mask_mm)
        if nrrd_layers is not None:
            layer_ref = materialize_nrrd_global_layer(
                mask_mm,
                model_name=str(nrrd_model_name),
                source='global',
                mask_kind='smoothing_result',
                pass_index=int(pass_idx),
                stage='gaussian_smoothing',
                description='Full-volume result after this GPU Gaussian smoothing pass.',
                temp_dir=temp_dir,
                workers=int(workers),
            )
            if layer_ref is not None:
                nrrd_layers.append(layer_ref)

        added = int(np.sum(added_by_slice, dtype=np.int64))
        removed = int(np.sum(removed_by_slice, dtype=np.int64))
        stats[f'pass{int(pass_idx)}_added_voxels'] = int(added)
        stats[f'pass{int(pass_idx)}_removed_voxels'] = int(removed)
        stats['total_added_voxels'] = int(stats.get('total_added_voxels', 0)) + int(added)
        stats['total_removed_voxels'] = int(stats.get('total_removed_voxels', 0)) + int(removed)
        stats['passes_completed'] = int(pass_idx)

    return stats

def apply_gaussian_smoothing_inplace(
    mask_mm: np.ndarray,
    sigma: float,
    passes: int,
    temp_dir: Path,
    *,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    nrrd_layers: Optional[List[NrrdLayerRef]] = None,
    nrrd_model_name: str = 'global',
) -> Dict[str, object]:
    """Smooth the source-geometry final union and re-threshold at 0.5.

    Smoothing runs after optional 3D void fill and centerline filtering, before
    postprocessing ``keep_objects``. One float32 workspace is reused across passes.
    """
    sigma_f = float(sigma)
    passes_i = int(passes)
    stats: Dict[str, object] = {
        'enabled': 1 if sigma_f > 0.0 and passes_i > 0 else 0,
        'sigma': float(sigma_f),
        'passes_requested': int(max(0, passes_i)),
        'passes_completed': 0,
        'total_added_voxels': 0,
        'total_removed_voxels': 0,
    }
    if sigma_f <= 0.0 or passes_i <= 0:
        return stats

    gpu_stats = _try_apply_gaussian_smoothing_gpu_chunked_inplace(
        mask_mm,
        sigma_f,
        passes_i,
        temp_dir,
        stats=stats,
        keep_temp=bool(keep_temp),
        workers=int(workers),
        nrrd_layers=nrrd_layers,
        nrrd_model_name=str(nrrd_model_name),
    )
    if gpu_stats is not None:
        return gpu_stats

    stats['backend'] = 'cpu_scipy_ndimage'
    smooth_dir = temp_dir / 'gaussian_smoothing'
    smooth_dir.mkdir(parents=True, exist_ok=True)
    work_path = smooth_dir / 'gaussian_float32.dat'
    work_mm = allocate_workspace_array(
        shape=tuple(int(x) for x in np.asarray(mask_mm).shape),
        dtype=np.float32,
        path=work_path,
        desc='Gaussian smoothing float32 workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    num_slices = int(mask_mm.shape[0])
    worker_count = choose_slice_parallel_workers(int(workers), num_slices)

    try:
        for pass_idx in range(1, passes_i + 1):
            print(f'Gaussian smoothing pass {int(pass_idx)}/{int(passes_i)} (sigma={sigma_f:g} voxels)')

            def _copy_to_float(z: int) -> None:
                work_mm[int(z), :, :] = np.asarray(mask_mm[int(z)], dtype=np.float32)

            parallel_for_indices_chunked(
                num_slices,
                _copy_to_float,
                max_workers=worker_count,
                desc=f'Gaussian smoothing pass {int(pass_idx)}: copy binary volume',
                show_progress=True,
                target_chunks_per_worker=2,
            )
            flush_array(work_mm)

            ndi.gaussian_filter(
                input=work_mm,
                sigma=float(sigma_f),
                output=work_mm,
                mode='constant',
                cval=0.0,
                truncate=4.0,
            )
            flush_array(work_mm)

            added_by_slice = np.zeros((num_slices,), dtype=np.int64)
            removed_by_slice = np.zeros((num_slices,), dtype=np.int64)

            def _threshold_slice(z: int) -> None:
                old = np.asarray(mask_mm[int(z)], dtype=bool)
                new = np.asarray(work_mm[int(z)] >= 0.5, dtype=bool)
                added_by_slice[int(z)] = np.int64(np.count_nonzero(new & (~old)))
                removed_by_slice[int(z)] = np.int64(np.count_nonzero(old & (~new)))
                mask_mm[int(z), :, :] = new.astype(np.uint8, copy=False)

            parallel_for_indices_chunked(
                num_slices,
                _threshold_slice,
                max_workers=worker_count,
                desc=f'Gaussian smoothing pass {int(pass_idx)}: threshold smoothed volume',
                show_progress=True,
                target_chunks_per_worker=2,
            )
            flush_array(mask_mm)

            if nrrd_layers is not None:
                layer_ref = materialize_nrrd_global_layer(
                    mask_mm,
                    model_name=str(nrrd_model_name),
                    source='global',
                    mask_kind='smoothing_result',
                    pass_index=int(pass_idx),
                    stage='gaussian_smoothing',
                    description='Full-volume result after this Gaussian smoothing pass.',
                    temp_dir=temp_dir,
                    workers=int(workers),
                )
                if layer_ref is not None:
                    nrrd_layers.append(layer_ref)

            added = int(np.sum(added_by_slice, dtype=np.int64))
            removed = int(np.sum(removed_by_slice, dtype=np.int64))
            stats[f'pass{int(pass_idx)}_added_voxels'] = int(added)
            stats[f'pass{int(pass_idx)}_removed_voxels'] = int(removed)
            stats['total_added_voxels'] = int(stats.get('total_added_voxels', 0)) + int(added)
            stats['total_removed_voxels'] = int(stats.get('total_removed_voxels', 0)) + int(removed)
            stats['passes_completed'] = int(pass_idx)
    finally:
        close_memmap_array(work_mm)
        if not bool(keep_temp):
            try:
                work_path.unlink(missing_ok=True)
            except Exception:
                pass

    return stats
