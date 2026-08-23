"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

import contextlib
import json
import math
import mmap
import os
import shutil
import sys
import threading
from collections import deque
from dataclasses import (
    dataclass,
    field,
    replace as dataclasses_replace,
)
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
)
import numpy as np
from ._deps import _NUMBA_IMPORT_ERROR, _numba, cv2, ndi, tqdm
from .cuda_interpolation import (
    CudaInterpolationRenderer,
    create_cuda_interpolation_renderer,
    gpu_interpolation_required,
)

from .config import (
    GIB,
)
from .runtime import (
    interpolation_process_worker_active,
    runtime_telemetry_phase,
)

# Explicit lower-layer dependencies keep imports one-way.
from .workspace import (
    _env_flag,
    _env_int,
)
from .runtime import (
    _create_memfd_backed_payload_path,
    allocate_workspace_array,
    array_nbytes,
    choose_slice_parallel_workers,
    close_memmap_array,
    close_memmap_array_without_flush,
    copy_workspace_array,
    estimate_interpolation_workspace_bytes,
    flush_array,
    numa_interleave_memory,
    open_raw_store_payload_writer,
    parallel_for_indices,
    parallel_for_indices_chunked,
    parallel_map_in_order,
    parallel_map_unordered,
    raw_store_memfd_enabled,
    release_memfd_owners_under,
    runtime_telemetry,
    should_use_in_memory_workspace,
)
from .geometry import (
    ViewInfo,
    is_tilted_view,
)
from .inference import (
    _cv2_connected_components,
    _fill_holes_2d_opencv,
)


if TYPE_CHECKING:
    from .topology import (
        SliceLocalLabelLUTs,
        SparseSliceLabelStore,
        _local_label_store_dtype,
        build_slice_endpoint_seeds_from_label_volume,
        interpolation_skip_compact_relabel_enabled,
        label_foreground_volume_streaming,
    )
    from .outputs import (
        _resize_sparse_binary_crop_to_output_region,
        _restore_source_indices_for_output_z,
    )

def _keep_center_component_2d(mask2d: np.ndarray) -> np.ndarray:
    # cv2 connectedComponents + the cv2 hole fill replace scipy.ndimage,
    # whose label/fill_holes hold the GIL and serialize the interpolation planner/render
    # thread pools. Semantics are unchanged (8-connected components, 4-connected hole fill).
    mask2d = np.asarray(mask2d, dtype=bool)
    if not mask2d.any():
        return mask2d

    if _planning_kernels_active():
        try:
            # fused keep-center + hole fill in one nogil kernel (two flood
            # fills on the small canvas) instead of cv2 label + per-edge uniques + fill.
            return _numba_keep_center_fill_kernel(np.ascontiguousarray(mask2d))
        except Exception as exc:
            _disable_planning_kernels(exc)

    mask_u8 = np.ascontiguousarray(mask2d.astype(np.uint8, copy=False))
    num_labels, labels2d = _cv2_connected_components(mask_u8, connectivity=8)
    if int(num_labels) <= 2:  # background label 0 + at most one component
        return _fill_holes_2d_opencv(mask2d)

    cy = mask2d.shape[0] // 2
    cx = mask2d.shape[1] // 2
    keep = int(labels2d[cy, cx])
    if keep == 0:
        pts = np.argwhere(labels2d > 0)
        if pts.size == 0:
            return np.zeros_like(mask2d, dtype=bool)
        d2 = (pts[:, 0] - cy) ** 2 + (pts[:, 1] - cx) ** 2
        py, px = pts[int(np.argmin(d2))]
        keep = int(labels2d[py, px])

    kept = (labels2d == keep)
    return _fill_holes_2d_opencv(kept)

def _signed_distance_2d(mask2d: np.ndarray) -> np.ndarray:
    # cv2.distanceTransform with DIST_MASK_PRECISE is the exact euclidean
    # transform (Felzenszwalb), matching scipy.ndimage.distance_transform_edt while releasing
    # the GIL so SDF construction parallelizes across planner threads.
    mask_u8 = np.ascontiguousarray(np.asarray(mask2d, dtype=np.uint8))
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(np.uint8(1) - mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.asarray(inside - outside, dtype=np.float32)

INTERPOLATION_LOCAL_SDF_PAD = 2

@dataclass(frozen=True)
class SliceComponentRecord:
    z: int
    label: int
    component_index: int
    bbox: Tuple[int, int, int, int]  # y0, x0, y1, x1 in full-slice coordinates
    anchor: Tuple[int, int]          # centroid-nearest component pixel in full-slice coordinates
    area: int
    mask_crop: np.ndarray            # bool crop in bbox coordinates

@dataclass(frozen=True)
class CroppedProjectionSDF:
    origin_y: int
    origin_x: int
    sdf: np.ndarray

@dataclass
class SliceComponentTable:
    z: int
    shape: Tuple[int, int]
    components: List[SliceComponentRecord]
    by_label: Dict[int, List[SliceComponentRecord]]

    def find_record_for_point(
        self,
        label: int,
        point_yx: Tuple[int, int],
    ) -> Tuple[Optional[SliceComponentRecord], Optional[Tuple[int, int]]]:
        records = self.by_label.get(int(label), [])
        if not records:
            return None, None

        h, w = int(self.shape[0]), int(self.shape[1])
        py = int(np.clip(int(point_yx[0]), 0, max(0, h - 1)))
        px = int(np.clip(int(point_yx[1]), 0, max(0, w - 1)))

        for record in records:
            y0, x0, y1, x1 = record.bbox
            if y0 <= py < y1 and x0 <= px < x1 and bool(record.mask_crop[py - y0, px - x0]):
                return record, (int(py), int(px))

        best_record: Optional[SliceComponentRecord] = None
        best_point: Optional[Tuple[int, int]] = None
        best_d2: Optional[int] = None
        for record in records:
            nearest = _nearest_point_in_component_record(record, (py, px))
            if nearest is None:
                continue
            d2 = int((int(nearest[0]) - py) ** 2 + (int(nearest[1]) - px) ** 2)
            if best_d2 is None or d2 < best_d2:
                best_d2 = int(d2)
                best_record = record
                best_point = nearest
        return best_record, best_point

    def find_branch_continuation(
        self,
        label: int,
        prev_record: SliceComponentRecord,
        prev_anchor: Tuple[int, int],
    ) -> Tuple[Optional[SliceComponentRecord], Optional[Tuple[int, int]]]:
        records = self.by_label.get(int(label), [])
        if not records:
            return None, None
        if len(records) == 1:
            record = records[0]
            return record, _nearest_point_in_component_record(record, prev_anchor)

        best_record: Optional[SliceComponentRecord] = None
        best_anchor: Optional[Tuple[int, int]] = None
        best_score: Optional[Tuple[int, int, int]] = None
        for record in records:
            anchor = _nearest_point_in_component_record(record, prev_anchor)
            if anchor is None:
                continue
            overlap = _component_record_dilated_overlap_count(prev_record, record)
            d2 = int((int(anchor[0]) - int(prev_anchor[0])) ** 2 + (int(anchor[1]) - int(prev_anchor[1])) ** 2)
            score = (int(overlap), -int(d2), int(record.area))
            if best_score is None or score > best_score:
                best_score = score
                best_record = record
                best_anchor = anchor
        return best_record, best_anchor

class SliceComponentTableCache:
    """Lazy per-slice component table and local projection-SDF cache for interpolation.

 Tables are keyed by slice and store cropped masks/bboxes for each 2D component of
 each 3D object label. Projection candidate search then operates only on the source
 component's bbox expanded by the maximum possible search-angle growth, avoiding
 full-slice SDFs and full-slice boolean projection scans per seed."""

    def __init__(self, labels_real: object, slice_luts: Optional['SliceLocalLabelLUTs'] = None) -> None:
        self.labels_real = labels_real
        # when the label volume holds per-slice LOCAL ids (compact relabel
        # skipped), tables canonicalize each component's id through the per-slice LUT at
        # build time, so every record.label / by_label key is a canonical 3D object id
        # exactly as if the full-volume relabel had run.
        self.slice_luts = slice_luts
        self.z_dim = int(labels_real.shape[0])
        self.shape_yx = (int(labels_real.shape[1]), int(labels_real.shape[2]))
        self._tables: Dict[int, SliceComponentTable] = {}
        self._table_order: deque[int] = deque()
        self._table_cache_lock = threading.Lock()
        table_budget_mib = _env_int('YOLO_TTA_INTERPOLATION_COMPONENT_TABLE_CACHE_MIB', 2048)
        self._table_budget_bytes = (
            -1 if int(table_budget_mib) < 0 else int(table_budget_mib) * (1024 ** 2)
        )
        self._table_live_payload_bytes = 0
        self._table_live_charge_bytes = 0
        self._table_peak_payload_bytes = 0
        self._table_peak_charge_bytes = 0
        self._table_builds = 0
        self._table_evictions = 0
        self._projection_sdfs: Dict[Tuple[int, int, int, int, float, int], CroppedProjectionSDF] = {}
        self._projection_sdf_order: deque[Tuple[int, int, int, int, float, int]] = deque()
        sdf_budget_mib = _env_int('YOLO_TTA_INTERPOLATION_SDF_CACHE_MIB', 512)
        self._projection_sdf_budget_bytes = (
            -1 if int(sdf_budget_mib) < 0 else int(sdf_budget_mib) * (1024 ** 2)
        )
        self._projection_sdf_live_bytes = 0
        self._projection_sdf_live_charge_bytes = 0
        self._projection_sdf_peak_bytes = 0
        self._projection_sdf_peak_charge_bytes = 0
        self._projection_sdf_computations = 0
        self._projection_sdf_evictions = 0
        self._table_locks = [threading.Lock() for _ in range(max(1, self.z_dim))]
        self._sdf_lock = threading.Lock()

    @staticmethod
    def _table_payload_and_charge_bytes(table: SliceComponentTable) -> Tuple[int, int]:
        """Return exact ndarray payload and a conservative cache charge.

        NumPy owns the dominant component-mask payload. The charge additionally reserves
        1 KiB per Python component record and 128 bytes per label bucket so the byte cap
        does not pretend the dataclass/list/dict graph is free.
        """
        payload = int(sum(int(record.mask_crop.nbytes) for record in table.components))
        charge = int(payload) + int(len(table.components)) * 1024 + int(len(table.by_label)) * 128 + 256
        return int(payload), int(charge)

    def _insert_table(self, z: int, table: SliceComponentTable) -> SliceComponentTable:
        payload_bytes, charge_bytes = self._table_payload_and_charge_bytes(table)
        with self._table_cache_lock:
            self._table_builds += 1
            existing = self._tables.get(int(z))
            if existing is not None:
                return existing
            # A zero budget intentionally disables retention. The caller still owns the
            # newly built table for the duration of its operation.
            if int(self._table_budget_bytes) == 0:
                return table
            self._tables[int(z)] = table
            self._table_order.append(int(z))
            self._table_live_payload_bytes += int(payload_bytes)
            self._table_live_charge_bytes += int(charge_bytes)
            self._table_peak_payload_bytes = max(
                int(self._table_peak_payload_bytes), int(self._table_live_payload_bytes)
            )
            self._table_peak_charge_bytes = max(
                int(self._table_peak_charge_bytes), int(self._table_live_charge_bytes)
            )
            while (
                int(self._table_budget_bytes) > 0
                and int(self._table_live_charge_bytes) > int(self._table_budget_bytes)
                and self._table_order
            ):
                victim_z = int(self._table_order.popleft())
                victim = self._tables.pop(victim_z, None)
                if victim is None:
                    continue
                victim_payload, victim_charge = self._table_payload_and_charge_bytes(victim)
                self._table_live_payload_bytes = max(
                    0, int(self._table_live_payload_bytes) - int(victim_payload)
                )
                self._table_live_charge_bytes = max(
                    0, int(self._table_live_charge_bytes) - int(victim_charge)
                )
                self._table_evictions += 1
            return table

    def get(self, z: int) -> SliceComponentTable:
        # Local import keeps the package dependency graph acyclic.
        from .topology import SparseSliceLabelStore

        z_i = int(z)
        if z_i < 0 or z_i >= self.z_dim:
            raise IndexError(z_i)
        cached = self._tables.get(z_i)
        if cached is not None:
            return cached
        with self._table_locks[z_i]:
            cached = self._tables.get(z_i)
            if cached is None:
                if isinstance(self.labels_real, SparseSliceLabelStore):
                    origin_y, origin_x, labels2d = self.labels_real.crop_with_origin(z_i)
                else:
                    origin_y, origin_x = 0, 0
                    labels2d = np.asarray(self.labels_real[z_i])  # type: ignore[index]
                cached = _build_slice_component_table(
                    labels2d,
                    z_i,
                    local_to_canonical=(
                        self.slice_luts.lut_for(z_i) if self.slice_luts is not None else None
                    ),
                    origin_yx=(int(origin_y), int(origin_x)),
                    full_shape_yx=self.shape_yx,
                )
                cached = self._insert_table(z_i, cached)
            return cached

    def prebuild(self, *, workers: int = 1, desc: str = 'Interpolation: per-slice component tables') -> None:
        total = int(self.z_dim)
        if total <= 0:
            return
        # Eagerly building every slice defeats a byte-bounded cache and immediately
        # evicts the early slices before endpoint scanning consumes them. Bounded mode
        # therefore builds lazily in the scan's naturally local z order.
        if int(self._table_budget_bytes) >= 0:
            return
        worker_count = choose_slice_parallel_workers(int(workers), total)

        def _build(z: int) -> int:
            return int(len(self.get(int(z)).components))

        if worker_count <= 1:
            for z in tqdm(range(total), desc=desc):
                _build(int(z))
            return

        pending = max(worker_count, worker_count * 8)
        for _component_count in tqdm(
            parallel_map_unordered(_build, range(total), max_workers=worker_count, max_pending=pending),
            total=total,
            desc=desc,
        ):
            pass

    def find_record_for_point(
        self,
        z: int,
        label: int,
        point_yx: Tuple[int, int],
    ) -> Tuple[Optional[SliceComponentRecord], Optional[Tuple[int, int]]]:
        return self.get(int(z)).find_record_for_point(int(label), point_yx)

    def get_projection_sdf(
        self,
        record: SliceComponentRecord,
        max_slice_distance: int,
        search_angle_deg: float,
    ) -> CroppedProjectionSDF:
        slope = math.tan(math.radians(float(search_angle_deg)))
        growth = max(0.0, float(slope)) * float(max(0, int(max_slice_distance)))
        pad = max(int(INTERPOLATION_LOCAL_SDF_PAD), int(math.ceil(growth)) + int(INTERPOLATION_LOCAL_SDF_PAD))
        key = (
            int(record.z),
            int(record.label),
            int(record.component_index),
            int(max_slice_distance),
            round(float(search_angle_deg), 6),
            int(pad),
        )
        # lock-free read fast path — CPython dict reads are atomic, and 160
        # planner threads were serializing on this lock for cache HITS; only inserts lock.
        cached = self._projection_sdfs.get(key)
        if cached is not None:
            return cached

        h, w = self.shape_yx
        y0, x0, y1, x1 = record.bbox
        crop_y0 = max(0, int(y0) - int(pad))
        crop_x0 = max(0, int(x0) - int(pad))
        crop_y1 = min(int(h), int(y1) + int(pad))
        crop_x1 = min(int(w), int(x1) + int(pad))
        source_crop = np.zeros((int(crop_y1 - crop_y0), int(crop_x1 - crop_x0)), dtype=bool)
        dy0 = int(y0) - int(crop_y0)
        dx0 = int(x0) - int(crop_x0)
        source_crop[dy0:dy0 + int(record.mask_crop.shape[0]), dx0:dx0 + int(record.mask_crop.shape[1])] = record.mask_crop
        cropped = CroppedProjectionSDF(
            origin_y=int(crop_y0),
            origin_x=int(crop_x0),
            sdf=np.ascontiguousarray(_signed_distance_2d(source_crop)),
        )
        with self._sdf_lock:
            self._projection_sdf_computations += 1
            existing = self._projection_sdfs.get(key)
            if existing is not None:
                return existing
            if int(self._projection_sdf_budget_bytes) == 0:
                return cropped
            self._projection_sdfs[key] = cropped
            self._projection_sdf_order.append(key)
            self._projection_sdf_live_bytes += int(cropped.sdf.nbytes)
            self._projection_sdf_live_charge_bytes += int(cropped.sdf.nbytes) + 512
            self._projection_sdf_peak_bytes = max(
                int(self._projection_sdf_peak_bytes), int(self._projection_sdf_live_bytes)
            )
            self._projection_sdf_peak_charge_bytes = max(
                int(self._projection_sdf_peak_charge_bytes),
                int(self._projection_sdf_live_charge_bytes),
            )
            while (
                int(self._projection_sdf_budget_bytes) > 0
                and int(self._projection_sdf_live_charge_bytes) > int(self._projection_sdf_budget_bytes)
                and self._projection_sdf_order
            ):
                victim_key = self._projection_sdf_order.popleft()
                victim = self._projection_sdfs.pop(victim_key, None)
                if victim is None:
                    continue
                self._projection_sdf_live_bytes = max(
                    0, int(self._projection_sdf_live_bytes) - int(victim.sdf.nbytes)
                )
                self._projection_sdf_live_charge_bytes = max(
                    0,
                    int(self._projection_sdf_live_charge_bytes)
                    - int(victim.sdf.nbytes) - 512,
                )
                self._projection_sdf_evictions += 1
            return cropped

    def telemetry(self) -> Dict[str, int]:
        """Snapshot exact cache counts and tracked ndarray payload bytes."""
        with self._table_cache_lock:
            table_stats = {
                'component_table_cache_budget_bytes': int(self._table_budget_bytes),
                'component_table_cache_entries': int(len(self._tables)),
                'component_table_cache_live_payload_bytes': int(self._table_live_payload_bytes),
                'component_table_cache_live_charge_bytes': int(self._table_live_charge_bytes),
                'component_table_cache_peak_payload_bytes': int(self._table_peak_payload_bytes),
                'component_table_cache_peak_charge_bytes': int(self._table_peak_charge_bytes),
                'component_table_builds': int(self._table_builds),
                'component_table_evictions': int(self._table_evictions),
            }
        with self._sdf_lock:
            sdf_stats = {
                'projection_sdf_cache_budget_bytes': int(self._projection_sdf_budget_bytes),
                'projection_sdf_cache_entries': int(len(self._projection_sdfs)),
                'projection_sdf_cache_live_bytes': int(self._projection_sdf_live_bytes),
                'projection_sdf_cache_live_charge_bytes': int(self._projection_sdf_live_charge_bytes),
                'projection_sdf_cache_peak_bytes': int(self._projection_sdf_peak_bytes),
                'projection_sdf_cache_peak_charge_bytes': int(self._projection_sdf_peak_charge_bytes),
                'projection_sdf_computations': int(self._projection_sdf_computations),
                'projection_sdf_evictions': int(self._projection_sdf_evictions),
            }
        table_stats.update(sdf_stats)
        return table_stats

    def clear(self) -> None:
        """Release every cached table/SDF after the final seed batch is rendered."""
        with self._table_cache_lock:
            self._tables.clear()
            self._table_order.clear()
            self._table_live_payload_bytes = 0
            self._table_live_charge_bytes = 0
        with self._sdf_lock:
            self._projection_sdfs.clear()
            self._projection_sdf_order.clear()
            self._projection_sdf_live_bytes = 0
            self._projection_sdf_live_charge_bytes = 0

def _nearest_point_in_component_record(record: SliceComponentRecord, ref_yx: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    y0, x0, _y1, _x1 = record.bbox
    if _planning_kernels_active():
        try:
            by, bx, found = _numba_nearest_true_pixel_kernel(
                record.mask_crop, int(ref_yx[0]) - int(y0), int(ref_yx[1]) - int(x0),
            )
            if int(found) == 0:
                return None
            return int(by) + int(y0), int(bx) + int(x0)
        except Exception as exc:
            _disable_planning_kernels(exc)
    ys, xs = np.nonzero(record.mask_crop)
    if ys.size == 0:
        return None
    gy = ys.astype(np.int64, copy=False) + int(y0)
    gx = xs.astype(np.int64, copy=False) + int(x0)
    d2 = (gy - int(ref_yx[0])) ** 2 + (gx - int(ref_yx[1])) ** 2
    idx = int(np.argmin(d2))
    return int(gy[idx]), int(gx[idx])

def _component_record_dilated_overlap_count(prev_record: SliceComponentRecord, candidate_record: SliceComponentRecord) -> int:
    py0, px0, py1, px1 = prev_record.bbox
    cy0, cx0, cy1, cx1 = candidate_record.bbox
    # The previous component is dilated by one pixel for branch following, so its
    # effective bbox expands by one pixel in every direction.
    iy0 = max(int(py0) - 1, int(cy0))
    ix0 = max(int(px0) - 1, int(cx0))
    iy1 = min(int(py1) + 1, int(cy1))
    ix1 = min(int(px1) + 1, int(cx1))
    if iy0 >= iy1 or ix0 >= ix1:
        return 0

    if _planning_kernels_active():
        try:
            # direct 3x3 neighborhood test — no pad, no cv2.dilate materialization.
            return int(_numba_dilated_overlap_count_kernel(
                prev_record.mask_crop, int(py0), int(px0),
                candidate_record.mask_crop, int(cy0), int(cx0),
                int(iy0), int(ix0), int(iy1), int(ix1),
            ))
        except Exception as exc:
            _disable_planning_kernels(exc)

    padded_prev = np.pad(np.asarray(prev_record.mask_crop, dtype=np.uint8), 1, mode='constant', constant_values=0)
    dilated_prev = cv2.dilate(padded_prev, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool, copy=False)
    prev_origin_y = int(py0) - 1
    prev_origin_x = int(px0) - 1
    prev_block = dilated_prev[iy0 - prev_origin_y:iy1 - prev_origin_y, ix0 - prev_origin_x:ix1 - prev_origin_x]
    cand_block = candidate_record.mask_crop[iy0 - int(cy0):iy1 - int(cy0), ix0 - int(cx0):ix1 - int(cx0)]
    return int(np.count_nonzero(prev_block & cand_block))

def _component_record_mirrored_u(record: SliceComponentRecord, width: int) -> SliceComponentRecord:
    """Mirror a component record along the u (x) axis of its full slice.

 /: the radial angular domain is [0°, 180°) over full-diameter
 frames, so the frame that continues across the 0°/180° wrap is the neighbor frame
 with u REVERSED (u -> width-1-u). Wrap-crossing comparisons mirror one side through
 this helper so overlap/continuation tests run in a common coordinate frame."""
    w = int(width)
    y0, x0, y1, x1 = record.bbox
    return SliceComponentRecord(
        z=int(record.z),
        label=int(record.label),
        component_index=int(record.component_index),
        bbox=(int(y0), int(w - int(x1)), int(y1), int(w - int(x0))),
        anchor=(int(record.anchor[0]), int(w - 1 - int(record.anchor[1]))),
        area=int(record.area),
        mask_crop=record.mask_crop[:, ::-1],
    )

def _build_slice_component_table(
    labels2d: np.ndarray,
    z: int,
    local_to_canonical: Optional[np.ndarray] = None,
    origin_yx: Tuple[int, int] = (0, 0),
    full_shape_yx: Optional[Tuple[int, int]] = None,
) -> SliceComponentTable:
    """Build interpolation component records with one bbox-limited 8-connected pass per slice.
    
    Each 2D component is associated with its 3D label through an anchor pixel."""
    labels_arr = np.asarray(labels2d)
    h, w = (int(labels_arr.shape[0]), int(labels_arr.shape[1]))
    origin_y, origin_x = (int(origin_yx[0]), int(origin_yx[1]))
    table_shape = (
        (int(full_shape_yx[0]), int(full_shape_yx[1]))
        if full_shape_yx is not None else (h, w)
    )
    components: List[SliceComponentRecord] = []
    by_label: Dict[int, List[SliceComponentRecord]] = {}

    rows_any = labels_arr.any(axis=1)
    if not rows_any.any():
        return SliceComponentTable(
            z=int(z), shape=table_shape, components=components, by_label=by_label,
        )
    cols_any = labels_arr.any(axis=0)
    row_idx = np.flatnonzero(rows_any)
    col_idx = np.flatnonzero(cols_any)
    by0 = int(row_idx[0])
    by1 = int(row_idx[-1]) + 1
    bx0 = int(col_idx[0])
    bx1 = int(col_idx[-1]) + 1

    crop_labels = labels_arr[by0:by1, bx0:bx1]
    fg_u8 = (crop_labels > 0).astype(np.uint8, copy=False)
    num_cc, cc, stats, centroids = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(fg_u8), connectivity=8, ltype=cv2.CV_32S,
    )
    for local_lbl in range(1, int(num_cc)):
        area = int(stats[int(local_lbl), cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[int(local_lbl), cv2.CC_STAT_LEFT])
        y = int(stats[int(local_lbl), cv2.CC_STAT_TOP])
        width = int(stats[int(local_lbl), cv2.CC_STAT_WIDTH])
        height = int(stats[int(local_lbl), cv2.CC_STAT_HEIGHT])
        if width <= 0 or height <= 0:
            continue
        comp_crop = np.ascontiguousarray(cc[y:y + height, x:x + width] == int(local_lbl))
        ys, xs = np.nonzero(comp_crop)
        if ys.size <= 0:
            continue
        # Whole component belongs to one 3D label (see docstring): read it at any member pixel.
        label_i = int(crop_labels[y + int(ys[0]), x + int(xs[0])])
        if label_i <= 0:
            continue
        if local_to_canonical is not None:
            # per-slice LOCAL id -> canonical 3D object id (no relabel pass).
            label_i = int(local_to_canonical[label_i])
            if label_i <= 0:
                continue
        centroid_x = float(centroids[int(local_lbl), 0]) - float(x)
        centroid_y = float(centroids[int(local_lbl), 1]) - float(y)
        d2 = (ys.astype(np.float32, copy=False) - centroid_y) ** 2 + (xs.astype(np.float32, copy=False) - centroid_x) ** 2
        anchor_idx = int(np.argmin(d2))
        record = SliceComponentRecord(
            z=int(z),
            label=int(label_i),
            component_index=int(len(components) + 1),
            bbox=(
                int(origin_y + by0 + y), int(origin_x + bx0 + x),
                int(origin_y + by0 + y + height), int(origin_x + bx0 + x + width),
            ),
            anchor=(
                int(origin_y + by0 + y + int(ys[anchor_idx])),
                int(origin_x + bx0 + x + int(xs[anchor_idx])),
            ),
            area=int(area),
            mask_crop=comp_crop,
        )
        components.append(record)
        by_label.setdefault(int(label_i), []).append(record)
    return SliceComponentTable(
        z=int(z), shape=table_shape, components=components, by_label=by_label,
    )

def _component_record_to_local_canvas(record: SliceComponentRecord, anchor_yx: Tuple[int, int], half_width: int) -> np.ndarray:
    size = int(2 * int(half_width) + 1)
    local = np.zeros((size, size), dtype=bool)
    y0, x0, _y1, _x1 = record.bbox
    y_off = int(y0) - int(anchor_yx[0]) + int(half_width)
    x_off = int(x0) - int(anchor_yx[1]) + int(half_width)
    if _planning_kernels_active():
        try:
            _numba_scatter_true_kernel(record.mask_crop, int(y_off), int(x_off), local)
            return local
        except Exception as exc:
            _disable_planning_kernels(exc)
    ys, xs = np.nonzero(record.mask_crop)
    if ys.size == 0:
        return local
    yy = ys.astype(np.int64, copy=False) + int(y_off)
    xx = xs.astype(np.int64, copy=False) + int(x_off)
    valid = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
    local[yy[valid], xx[valid]] = True
    return local

def _local_half_width_for_component_records(
    source_record: SliceComponentRecord,
    source_anchor: Tuple[int, int],
    target_record: SliceComponentRecord,
    target_anchor: Tuple[int, int],
) -> int:
    max_extent = 0.0
    for record, anchor in ((source_record, source_anchor), (target_record, target_anchor)):
        y0, x0, _y1, _x1 = record.bbox
        if _planning_kernels_active():
            try:
                ext = int(_numba_max_abs_extent_kernel(
                    record.mask_crop, int(anchor[0]) - int(y0), int(anchor[1]) - int(x0),
                ))
                if ext >= 0:
                    max_extent = max(max_extent, float(ext))
                continue
            except Exception as exc:
                _disable_planning_kernels(exc)
        ys, xs = np.nonzero(record.mask_crop)
        if ys.size == 0:
            continue
        gy = ys.astype(np.int64, copy=False) + int(y0)
        gx = xs.astype(np.int64, copy=False) + int(x0)
        max_extent = max(
            max_extent,
            float(np.max(np.abs(gy - int(anchor[0])))),
            float(np.max(np.abs(gx - int(anchor[1])))),
        )

    max_extent = max(
        max_extent,
        float(abs(int(target_anchor[0]) - int(source_anchor[0]))),
        float(abs(int(target_anchor[1]) - int(source_anchor[1]))),
    )
    return max(4, int(math.ceil(max_extent)) + 4)

@dataclass(frozen=True)
class SliceEndpointSeed:
    label: int
    point: Tuple[int, int, int]  # (slice, row, col)
    direction_sign: int          # 1 or +1 along the slice axis
    # Cheap scheduler hint captured while the source 2D component is already hot.
    # It never participates in candidate selection or bridge geometry.
    planning_cost: int = 1


def interpolation_seed_schedule_window_factor() -> int:
    """Number of planner-worker waves rebalanced within one slice-local window.

    Zero restores strictly slice-major submission.  A bounded window starts expensive
    component seeds first without globally scattering component-table/SDF cache access.
    """
    return max(0, _env_int('YOLO_TTA_INTERPOLATION_SEED_SCHEDULE_WINDOW_FACTOR', 4))


def _rebalance_slice_major_endpoint_seeds(
    seeds: List[SliceEndpointSeed],
    *,
    plan_workers: int,
    window_factor: Optional[int] = None,
) -> int:
    """Stable largest-cost-first scheduling inside bounded slice-major windows.

    All seeds plan against the same frozen label/mask snapshot and rendered sections are
    reduced with OR, so submission order is not semantic.  Keeping the rebalance bounded
    preserves the component-cache locality of the prior slice-major traversal while moving
    the expensive SDF seeds ahead of cheap work that can fill each worker wave's tail.
    """
    workers = max(1, int(plan_workers))
    factor = (
        interpolation_seed_schedule_window_factor()
        if window_factor is None else max(0, int(window_factor))
    )
    if workers <= 1 or factor <= 0 or len(seeds) <= 1:
        return 0
    window = max(workers, workers * factor)
    for start in range(0, len(seeds), int(window)):
        end = min(len(seeds), int(start) + int(window))
        # Python's sort is stable: equal-cost seeds retain the exact prior slice-major
        # order, which keeps the scheduling policy deterministic across worker counts.
        seeds[start:end] = sorted(
            seeds[start:end],
            key=lambda seed: -max(1, int(getattr(seed, 'planning_cost', 1))),
        )
    return int(window)


@dataclass(frozen=True)
class SliceProjectionCandidate:
    source_label: int
    target_label: int
    source_point: Tuple[int, int, int]
    target_point: Tuple[int, int, int]
    slice_distance: int

def _nearest_point_in_mask(mask2d: np.ndarray, ref_yx: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if _planning_kernels_active():
        try:
            by, bx, found = _numba_nearest_true_pixel_kernel(
                np.asarray(mask2d, dtype=bool), int(ref_yx[0]), int(ref_yx[1]),
            )
            if int(found) == 0:
                return None
            return int(by), int(bx)
        except Exception as exc:
            _disable_planning_kernels(exc)
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return None
    d2 = (ys.astype(np.int64) - int(ref_yx[0])) ** 2 + (xs.astype(np.int64) - int(ref_yx[1])) ** 2
    idx = int(np.argmin(d2))
    return int(ys[idx]), int(xs[idx])

def _component_mask_and_anchor(mask2d: np.ndarray, point_yx: Tuple[int, int]) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    mask2d = np.asarray(mask2d, dtype=bool)
    if not mask2d.any():
        return np.zeros(mask2d.shape, dtype=bool), None

    labels2d, num = ndi.label(mask2d, structure=np.ones((3, 3), dtype=bool))
    if num <= 0:
        return np.zeros(mask2d.shape, dtype=bool), None

    py = int(np.clip(int(point_yx[0]), 0, mask2d.shape[0] - 1))
    px = int(np.clip(int(point_yx[1]), 0, mask2d.shape[1] - 1))
    lbl = int(labels2d[py, px])
    if lbl <= 0:
        nearest = _nearest_point_in_mask(mask2d, (py, px))
        if nearest is None:
            return np.zeros(mask2d.shape, dtype=bool), None
        lbl = int(labels2d[nearest[0], nearest[1]])
        py, px = nearest

    comp = labels2d == lbl
    anchor = _nearest_point_in_mask(comp, (py, px))
    return comp.astype(bool), anchor

def _component_centroid_anchor(mask2d: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return None
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    d2 = (ys.astype(np.float32) - cy) ** 2 + (xs.astype(np.float32) - cx) ** 2
    idx = int(np.argmin(d2))
    return int(ys[idx]), int(xs[idx])

def _component_max_radius(mask2d: np.ndarray) -> float:
    # exact cv2 distance transform (GIL-releasing) replaces scipy EDT.
    mask_u8 = np.ascontiguousarray(np.asarray(mask2d, dtype=np.uint8))
    if not np.any(mask_u8):
        return 0.0
    if bool(np.all(mask_u8)):
        # No background pixel: the inscribed radius is unbounded within the crop; return an
        # upper bound so the caller never rejects on this degenerate case.
        return float(max(mask_u8.shape[0], mask_u8.shape[1]))
    return float(np.max(cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)))

def _follow_branch_component(
    slice_mask: np.ndarray,
    prev_component_mask: np.ndarray,
    prev_anchor: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    slice_mask = np.asarray(slice_mask, dtype=bool)
    if not slice_mask.any():
        return np.zeros(slice_mask.shape, dtype=bool), None

    labels2d, num = ndi.label(slice_mask, structure=np.ones((3, 3), dtype=bool))
    if num <= 0:
        return np.zeros(slice_mask.shape, dtype=bool), None

    if num == 1:
        comp = labels2d == 1
        return comp.astype(bool), _nearest_point_in_mask(comp, prev_anchor)

    dil_prev = ndi.binary_dilation(np.asarray(prev_component_mask, dtype=bool), structure=np.ones((3, 3), dtype=bool))

    best_lbl = 0
    best_score: Optional[Tuple[int, int, int]] = None
    for lbl in range(1, int(num) + 1):
        comp = labels2d == lbl
        overlap = int(np.count_nonzero(comp & dil_prev))
        anchor = _nearest_point_in_mask(comp, prev_anchor)
        if anchor is None:
            continue
        d2 = int((anchor[0] - int(prev_anchor[0])) ** 2 + (anchor[1] - int(prev_anchor[1])) ** 2)
        area = int(np.count_nonzero(comp))
        score = (overlap, -d2, area)
        if best_score is None or score > best_score:
            best_score = score
            best_lbl = int(lbl)

    if best_lbl <= 0:
        return np.zeros(slice_mask.shape, dtype=bool), None

    comp = labels2d == best_lbl
    return comp.astype(bool), _nearest_point_in_mask(comp, prev_anchor)

def _component_to_local_canvas(mask2d: np.ndarray, anchor_yx: Tuple[int, int], half_width: int) -> np.ndarray:
    size = int(2 * half_width + 1)
    local = np.zeros((size, size), dtype=bool)
    y_off = -int(anchor_yx[0]) + int(half_width)
    x_off = -int(anchor_yx[1]) + int(half_width)
    if _planning_kernels_active():
        try:
            _numba_scatter_true_kernel(np.asarray(mask2d, dtype=bool), int(y_off), int(x_off), local)
            return local
        except Exception as exc:
            _disable_planning_kernels(exc)
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return local

    yy = ys.astype(np.int64) + int(y_off)
    xx = xs.astype(np.int64) + int(x_off)
    valid = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
    local[yy[valid], xx[valid]] = True
    return local

def _paste_local_mask_onto_slice(
    dest_slice: np.ndarray,
    local_mask: np.ndarray,
    center_yx: Tuple[float, float],
    *,
    dst_bbox_union: Optional[List[int]] = None,
    paint_value: int = 1,
    binary_destination: bool = True,
) -> int:
    """OR a local mask into one destination slice and update the caller-provided touched bbox."""
    if not np.any(local_mask):
        return 0

    size_y, size_x = local_mask.shape
    hh_y = size_y // 2
    hh_x = size_x // 2
    cy = int(round(float(center_yx[0])))
    cx = int(round(float(center_yx[1])))

    y0 = cy - hh_y
    x0 = cx - hh_x
    y1 = y0 + size_y
    x1 = x0 + size_x

    dy0 = max(0, -y0)
    dx0 = max(0, -x0)
    dy1 = max(0, y1 - dest_slice.shape[0])
    dx1 = max(0, x1 - dest_slice.shape[1])

    src_y0 = dy0
    src_y1 = size_y - dy1
    src_x0 = dx0
    src_x1 = size_x - dx1

    dst_y0 = y0 + dy0
    dst_y1 = y1 - dy1
    dst_x0 = x0 + dx0
    dst_x1 = x1 - dx1

    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return 0

    if dst_bbox_union is not None:
        dst_bbox_union[0] = min(int(dst_bbox_union[0]), int(dst_y0))
        dst_bbox_union[1] = min(int(dst_bbox_union[1]), int(dst_x0))
        dst_bbox_union[2] = max(int(dst_bbox_union[2]), int(dst_y1))
        dst_bbox_union[3] = max(int(dst_bbox_union[3]), int(dst_x1))

    if bool(binary_destination) and int(paint_value) == 1 and _planning_kernels_active():
        try:
            # one nogil pass counts+writes only newly-set pixels (bridge slices
            # hold 0/1) instead of two bool temporaries + a count + an OR store per paste.
            return int(_numba_paste_masked_or_kernel(
                np.asarray(dest_slice), np.asarray(local_mask, dtype=bool),
                int(dst_y0), int(dst_x0),
                int(src_y0), int(src_y1), int(src_x0), int(src_x1),
            ))
        except Exception as exc:
            _disable_planning_kernels(exc)

    patch = np.asarray(local_mask[src_y0:src_y1, src_x0:src_x1], dtype=bool)
    current = np.asarray(dest_slice[dst_y0:dst_y1, dst_x0:dst_x1])
    value = np.asarray(int(paint_value), dtype=current.dtype)
    added = int(np.count_nonzero(patch & ((current & value) == 0)))
    np.bitwise_or(current, value, out=current, where=patch)
    return added

def _local_half_width_for_components(
    source_component: np.ndarray,
    source_anchor: Tuple[int, int],
    target_component: np.ndarray,
    target_anchor: Tuple[int, int],
) -> int:
    max_extent = 0.0
    for comp, anchor in ((source_component, source_anchor), (target_component, target_anchor)):
        ys, xs = np.nonzero(comp)
        if ys.size == 0:
            continue
        max_extent = max(
            max_extent,
            float(np.max(np.abs(ys.astype(np.int64) - int(anchor[0])))),
            float(np.max(np.abs(xs.astype(np.int64) - int(anchor[1])))),
        )

    max_extent = max(
        max_extent,
        float(abs(int(target_anchor[0]) - int(source_anchor[0]))),
        float(abs(int(target_anchor[1]) - int(source_anchor[1]))),
    )
    return max(4, int(math.ceil(max_extent)) + 4)

_NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED = False

_NUMBA_PLANNING_KERNELS_RUNTIME_DISABLED = False

def compiled_topology_kernels_enabled() -> bool:
    """Generic 3D topology/keep_objects kernels, independent of interpolation settings."""
    return bool(_numba is not None and _env_flag('YOLO_TTA_TOPOLOGY_COMPILED_KERNELS', True))

def compiled_interpolation_kernels_enabled() -> bool:
    return bool(_numba is not None and _env_flag('YOLO_TTA_INTERPOLATION_COMPILED_KERNELS', True))

if _numba is not None:
    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_nearest_true_pixel_kernel(mask: np.ndarray, ref_y: int, ref_x: int) -> Tuple[int, int, int]:
        best_y = -1
        best_x = -1
        best_d2 = np.int64(0)
        found = 0
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if not mask[y, x]:
                    continue
                dy = np.int64(y - ref_y)
                dx = np.int64(x - ref_x)
                d2 = dy * dy + dx * dx
                if found == 0 or d2 < best_d2:
                    best_d2 = d2
                    best_y = y
                    best_x = x
                    found = 1
        return best_y, best_x, found

    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_scatter_true_kernel(mask: np.ndarray, y_off: int, x_off: int, out: np.ndarray) -> None:
        size_y = out.shape[0]
        size_x = out.shape[1]
        for y in range(mask.shape[0]):
            yy = y + y_off
            if yy < 0 or yy >= size_y:
                continue
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    xx = x + x_off
                    if 0 <= xx < size_x:
                        out[yy, xx] = True

    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_max_abs_extent_kernel(mask: np.ndarray, ref_y: int, ref_x: int) -> int:
        best = -1
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if not mask[y, x]:
                    continue
                dy = y - ref_y
                if dy < 0:
                    dy = -dy
                dx = x - ref_x
                if dx < 0:
                    dx = -dx
                ext = dy if dy > dx else dx
                if ext > best:
                    best = ext
        return best

    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_dilated_overlap_count_kernel(
        prev_mask: np.ndarray, prev_oy: int, prev_ox: int,
        cand_mask: np.ndarray, cand_oy: int, cand_ox: int,
        iy0: int, ix0: int, iy1: int, ix1: int,
    ) -> int:
        # Count candidate pixels with any prev pixel in their 3x3 neighborhood — the
        # dilate(prev, 3x3) & cand overlap without materializing the dilation.
        ph = prev_mask.shape[0]
        pw = prev_mask.shape[1]
        count = 0
        for gy in range(iy0, iy1):
            cy = gy - cand_oy
            for gx in range(ix0, ix1):
                if not cand_mask[cy, gx - cand_ox]:
                    continue
                hit = False
                for dy in range(-1, 2):
                    py = gy + dy - prev_oy
                    if py < 0 or py >= ph:
                        continue
                    for dx in range(-1, 2):
                        px = gx + dx - prev_ox
                        if 0 <= px < pw and prev_mask[py, px]:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    count += 1
        return count

    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_keep_center_fill_kernel(mask: np.ndarray) -> np.ndarray:
        # keep-center-component (8-connected) + enclosed-hole fill (4-connected background),
        # fused: result = NOT(border-4-connected component of NOT kept). Matches
        # _keep_center_component_2d + _fill_holes_2d_opencv semantics exactly.
        h = mask.shape[0]
        w = mask.shape[1]
        out = np.zeros((h, w), dtype=np.bool_)
        if h <= 0 or w <= 0:
            return out

        cy = h // 2
        cx = w // 2
        seed_y = -1
        seed_x = -1
        if mask[cy, cx]:
            seed_y = cy
            seed_x = cx
        else:
            best_d2 = np.int64(-1)
            for y in range(h):
                for x in range(w):
                    if not mask[y, x]:
                        continue
                    dy = np.int64(y - cy)
                    dx = np.int64(x - cx)
                    d2 = dy * dy + dx * dx
                    if best_d2 < 0 or d2 < best_d2:
                        best_d2 = d2
                        seed_y = y
                        seed_x = x
        if seed_y < 0:
            return out

        # state: 0 unknown, 1 kept (center component), 2 outside background
        state = np.zeros((h, w), dtype=np.uint8)
        stack = np.empty((h * w, 2), dtype=np.int32)
        sp = 0
        state[seed_y, seed_x] = 1
        stack[sp, 0] = seed_y
        stack[sp, 1] = seed_x
        sp += 1
        while sp > 0:
            sp -= 1
            y = int(stack[sp, 0])
            x = int(stack[sp, 1])
            for dy in range(-1, 2):
                ny = y + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    nx = x + dx
                    if nx < 0 or nx >= w:
                        continue
                    if state[ny, nx] == 0 and mask[ny, nx]:
                        state[ny, nx] = 1
                        stack[sp, 0] = ny
                        stack[sp, 1] = nx
                        sp += 1

        # 4-connected flood of NOT-kept from every border pixel -> outside background.
        sp = 0
        for x in range(w):
            if state[0, x] == 0:
                state[0, x] = 2
                stack[sp, 0] = 0
                stack[sp, 1] = x
                sp += 1
            if state[h - 1, x] == 0:
                state[h - 1, x] = 2
                stack[sp, 0] = h - 1
                stack[sp, 1] = x
                sp += 1
        for y in range(h):
            if state[y, 0] == 0:
                state[y, 0] = 2
                stack[sp, 0] = y
                stack[sp, 1] = 0
                sp += 1
            if state[y, w - 1] == 0:
                state[y, w - 1] = 2
                stack[sp, 0] = y
                stack[sp, 1] = w - 1
                sp += 1
        while sp > 0:
            sp -= 1
            y = int(stack[sp, 0])
            x = int(stack[sp, 1])
            if y > 0 and state[y - 1, x] == 0:
                state[y - 1, x] = 2
                stack[sp, 0] = y - 1
                stack[sp, 1] = x
                sp += 1
            if y + 1 < h and state[y + 1, x] == 0:
                state[y + 1, x] = 2
                stack[sp, 0] = y + 1
                stack[sp, 1] = x
                sp += 1
            if x > 0 and state[y, x - 1] == 0:
                state[y, x - 1] = 2
                stack[sp, 0] = y
                stack[sp, 1] = x - 1
                sp += 1
            if x + 1 < w and state[y, x + 1] == 0:
                state[y, x + 1] = 2
                stack[sp, 0] = y
                stack[sp, 1] = x + 1
                sp += 1

        for y in range(h):
            for x in range(w):
                out[y, x] = state[y, x] != 2
        return out

    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_paste_masked_or_kernel(
        dest: np.ndarray, local_mask: np.ndarray,
        dst_y0: int, dst_x0: int,
        src_y0: int, src_y1: int, src_x0: int, src_x1: int,
    ) -> int:
        added = 0
        for sy in range(src_y0, src_y1):
            dy = dst_y0 + (sy - src_y0)
            for sx in range(src_x0, src_x1):
                if not local_mask[sy, sx]:
                    continue
                dx = dst_x0 + (sx - src_x0)
                if dest[dy, dx] == 0:
                    dest[dy, dx] = 1
                    added += 1
        return added
else:
    _numba_nearest_true_pixel_kernel = None
    _numba_scatter_true_kernel = None
    _numba_max_abs_extent_kernel = None
    _numba_dilated_overlap_count_kernel = None
    _numba_keep_center_fill_kernel = None
    _numba_paste_masked_or_kernel = None

def _planning_kernels_active() -> bool:
    return bool(
        _numba_nearest_true_pixel_kernel is not None
        and compiled_interpolation_kernels_enabled()
        and not _NUMBA_PLANNING_KERNELS_RUNTIME_DISABLED
    )

def _disable_planning_kernels(exc: BaseException) -> None:
    global _NUMBA_PLANNING_KERNELS_RUNTIME_DISABLED
    if not _NUMBA_PLANNING_KERNELS_RUNTIME_DISABLED:
        _NUMBA_PLANNING_KERNELS_RUNTIME_DISABLED = True
        print(f'Warning: numba planning kernels failed ({exc}); using python planning helpers for remaining calls in this process.')

def interpolation_projection_numba_max_tracked() -> int:
    # Tiled masks can place hundreds of distinct canonical labels inside one local
    # projection window.  The former 64-entry scratch array silently discarded the compiled
    # result and reran the entire seed in Python whenever that happened.  A 1024-entry
    # workspace is only ~80 KiB across the ten int64 arrays and keeps the common fragmented
    # case on the no-GIL kernel; the overflow path remains exact for unusually dense windows.
    return max(8, _env_int('YOLO_TTA_INTERPOLATION_NUMBA_MAX_TRACKED_CANDIDATES', 1024))

def interpolation_compiled_kernels_status() -> str:
    if _numba is None:
        return f'unavailable ({_NUMBA_IMPORT_ERROR})'
    if not _env_flag('YOLO_TTA_INTERPOLATION_COMPILED_KERNELS', True):
        return 'disabled by YOLO_TTA_INTERPOLATION_COMPILED_KERNELS=0'
    if _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED:
        return 'disabled after runtime compilation/execution failure'
    return 'enabled: numba no-GIL projection-candidate kernel'

def interpolation_planning_backend_name() -> str:
    base = 'cached_per_slice_component_tables_local_sdf_unordered'
    if compiled_interpolation_kernels_enabled() and not _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED:
        base += '+numba_nogil_projection_candidates'
    else:
        base += '+python_projection_candidates'
    if interpolation_process_worker_active():
        base += '+process_isolated_pass'
    return base

if _numba is not None:
    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_find_projection_candidates_kernel(
        labels_real: np.ndarray,
        sparse_flat: np.ndarray,
        sparse_offsets: np.ndarray,
        sparse_bboxes: np.ndarray,
        sparse_mode: bool,
        num_slices_arg: int,
        full_w_arg: int,
        lut_flat: np.ndarray,
        lut_offsets: np.ndarray,
        sdf: np.ndarray,
        crop_y0: int,
        crop_x0: int,
        s0: int,
        source_anchor_y: int,
        source_anchor_x: int,
        seed_label: int,
        direction_sign: int,
        max_steps: int,
        slope: float,
        max_candidates: int,
        wrap_axis: bool,
        search_angle_negative: bool,
        max_tracked: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        out_labels = np.zeros((max_tracked,), dtype=np.int64)
        out_slices = np.zeros((max_tracked,), dtype=np.int64)
        out_ys = np.zeros((max_tracked,), dtype=np.int64)
        out_xs = np.zeros((max_tracked,), dtype=np.int64)
        out_steps = np.zeros((max_tracked,), dtype=np.int64)
        out_d2 = np.zeros((max_tracked,), dtype=np.int64)

        step_labels = np.zeros((max_tracked,), dtype=np.int64)
        step_ys = np.zeros((max_tracked,), dtype=np.int64)
        step_xs = np.zeros((max_tracked,), dtype=np.int64)
        step_d2 = np.zeros((max_tracked,), dtype=np.int64)

        found_count = 0
        num_slices = num_slices_arg
        full_w = full_w_arg
        sdf_h = sdf.shape[0]
        sdf_w = sdf.shape[1]
        overflow = 0
        # when the label volume holds per-slice LOCAL ids (compact relabel
        # skipped), lut_offsets has one entry per slice and reads canonicalize through the
        # concatenated local->canonical table; a shorter array means canonical ids in-raster.
        use_lut = lut_offsets.shape[0] == num_slices

        for step in range(1, max_steps + 1):
            s = s0 + direction_sign * step
            # a projection step that crosses the radial 0°/180° wrap lands
            # in a frame whose u axis is REVERSED relative to the projection cone, so read
            # (and report) the mirrored column there. max_steps <= num_slices-1 caps the
            # walk at a single crossing.
            mirrored = False
            if wrap_axis:
                if s < 0 or s >= num_slices:
                    mirrored = True
                s = s % num_slices
            else:
                if s < 0 or s >= num_slices:
                    break

            threshold = -slope * float(step)
            any_projection = False
            step_count = 0

            for yy in range(sdf_h):
                gy = crop_y0 + yy
                for xx in range(sdf_w):
                    if sdf[yy, xx] < threshold:
                        continue
                    any_projection = True
                    gx = crop_x0 + xx
                    if mirrored:
                        gx_read = full_w - 1 - gx
                    else:
                        gx_read = gx
                    if sparse_mode:
                        sy0 = int(sparse_bboxes[s, 0])
                        sy1 = int(sparse_bboxes[s, 1])
                        sx0 = int(sparse_bboxes[s, 2])
                        sx1 = int(sparse_bboxes[s, 3])
                        if gy < sy0 or gy >= sy1 or gx_read < sx0 or gx_read >= sx1:
                            raw_label = 0
                        else:
                            sparse_index = int(sparse_offsets[s]) + (
                                (gy - sy0) * (sx1 - sx0) + (gx_read - sx0)
                            )
                            raw_label = int(sparse_flat[sparse_index])
                    else:
                        raw_label = int(labels_real[s, gy, gx_read])
                    if raw_label <= 0:
                        continue
                    if use_lut:
                        target_label = int(lut_flat[lut_offsets[s] + raw_label])
                    else:
                        target_label = raw_label
                    if target_label <= 0 or target_label == seed_label:
                        continue

                    already_found = False
                    for prev_idx in range(found_count):
                        if out_labels[prev_idx] == target_label:
                            already_found = True
                            break
                    if already_found:
                        continue

                    step_idx = -1
                    for local_idx in range(step_count):
                        if step_labels[local_idx] == target_label:
                            step_idx = local_idx
                            break

                    dy = gy - source_anchor_y
                    dx = gx - source_anchor_x
                    d2 = dy * dy + dx * dx
                    if step_idx < 0:
                        if step_count >= max_tracked:
                            overflow = 1
                            return out_labels, out_slices, out_ys, out_xs, out_steps, out_d2, found_count, overflow
                        step_labels[step_count] = target_label
                        step_ys[step_count] = gy
                        # Candidate points are reported in the target slice's own (actual)
                        # coordinates; d2 stays in unrolled projection coordinates.
                        step_xs[step_count] = gx_read
                        step_d2[step_count] = d2
                        step_count += 1
                    else:
                        if d2 < step_d2[step_idx]:
                            step_ys[step_idx] = gy
                            step_xs[step_idx] = gx_read
                            step_d2[step_idx] = d2

            if not any_projection:
                if search_angle_negative:
                    break
                continue

            for local_idx in range(step_count):
                if found_count >= max_tracked:
                    overflow = 1
                    return out_labels, out_slices, out_ys, out_xs, out_steps, out_d2, found_count, overflow
                out_labels[found_count] = step_labels[local_idx]
                out_slices[found_count] = s
                out_ys[found_count] = step_ys[local_idx]
                out_xs[found_count] = step_xs[local_idx]
                out_steps[found_count] = step
                out_d2[found_count] = step_d2[local_idx]
                found_count += 1

            if found_count >= max_candidates:
                break

        return out_labels, out_slices, out_ys, out_xs, out_steps, out_d2, found_count, overflow
else:
    _numba_find_projection_candidates_kernel = None

def _find_slice_projection_candidates_numba(
    labels_real: object,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    max_candidates: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
    slice_luts: Optional['SliceLocalLabelLUTs'] = None,
) -> Optional[List[SliceProjectionCandidate]]:
    # Local import keeps the package dependency graph acyclic.
    from .topology import SparseSliceLabelStore

    global _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED
    if (
        _numba_find_projection_candidates_kernel is None
        or not compiled_interpolation_kernels_enabled()
        or _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED
    ):
        return None
    if int(max_slice_distance) <= 0 or int(max_candidates) <= 0:
        return []

    s0, y0, x0 = seed.point
    num_slices = int(labels_real.shape[0])
    if num_slices <= 0:
        return []

    local_cache = (
        component_cache
        if component_cache is not None
        else SliceComponentTableCache(labels_real, slice_luts=slice_luts)
    )
    source_record, source_anchor = local_cache.find_record_for_point(int(s0), int(seed.label), (int(y0), int(x0)))
    if source_record is None or source_anchor is None or int(source_record.area) <= 0:
        return []

    max_steps = min(int(max_slice_distance), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(max_slice_distance)
    if int(max_steps) <= 0:
        return []

    cropped_sdf = local_cache.get_projection_sdf(
        source_record,
        max_slice_distance=int(max_steps),
        search_angle_deg=float(search_angle_deg),
    )
    sdf = np.ascontiguousarray(cropped_sdf.sdf, dtype=np.float32)
    crop_y0 = int(cropped_sdf.origin_y)
    crop_x0 = int(cropped_sdf.origin_x)
    slope = math.tan(math.radians(float(search_angle_deg)))
    max_tracked = max(int(max_candidates), int(interpolation_projection_numba_max_tracked()))

    # pass the per-slice local->canonical LUTs when the label volume holds
    # local ids; a 1-entry offsets sentinel tells the kernel the raster is already canonical.
    if slice_luts is not None:
        lut_flat_arg = slice_luts.lut_flat
        lut_offsets_arg = slice_luts.lut_offsets
    else:
        # 0-length offsets can never equal num_slices (>0 here), so use_lut stays False.
        lut_flat_arg = np.zeros((1,), dtype=np.uint32)
        lut_offsets_arg = np.zeros((0,), dtype=np.int64)
    if isinstance(labels_real, SparseSliceLabelStore):
        # Numba cannot accept the Python store object, so pass its native packed arena and
        # metadata plus a tiny unused dense sentinel. Candidate reads stay wholly compiled.
        dense_arg = np.empty((0, 0, 0), dtype=labels_real.dtype)
        sparse_flat_arg = labels_real.flat
        sparse_offsets_arg = labels_real.offsets
        sparse_bboxes_arg = labels_real.bboxes
        sparse_mode_arg = True
    else:
        dense_arg = labels_real
        sparse_flat_arg = np.empty((0,), dtype=np.uint16)
        sparse_offsets_arg = np.empty((0,), dtype=np.int64)
        sparse_bboxes_arg = np.empty((0, 4), dtype=np.int64)
        sparse_mode_arg = False
    try:
        labels_out, slices_out, ys_out, xs_out, steps_out, d2_out, count, overflow = _numba_find_projection_candidates_kernel(
            dense_arg,
            sparse_flat_arg,
            sparse_offsets_arg,
            sparse_bboxes_arg,
            bool(sparse_mode_arg),
            int(num_slices),
            int(labels_real.shape[2]),
            lut_flat_arg,
            lut_offsets_arg,
            sdf,
            int(crop_y0),
            int(crop_x0),
            int(s0),
            int(source_anchor[0]),
            int(source_anchor[1]),
            int(seed.label),
            int(seed.direction_sign),
            int(max_steps),
            float(slope),
            int(max_candidates),
            bool(wrap_axis),
            bool(float(search_angle_deg) < 0.0),
            int(max_tracked),
        )
    except Exception as exc:
        _NUMBA_PROJECTION_KERNEL_RUNTIME_DISABLED = True
        print(f'Warning: Numba interpolation projection kernel failed ({exc}); using Python candidate search for remaining calls in this process.')
        return None

    if int(overflow) != 0:
        # Preserve exact semantics by falling back to the Python implementation when a single
        # projection step contains more distinct labels than the fixed-size compiled workspace.
        return None
    if int(count) <= 0:
        return []

    rows: List[Tuple[int, int, int, int, int, int]] = []
    for idx in range(int(count)):
        rows.append((
            int(steps_out[int(idx)]),
            int(d2_out[int(idx)]),
            int(labels_out[int(idx)]),
            int(slices_out[int(idx)]),
            int(ys_out[int(idx)]),
            int(xs_out[int(idx)]),
        ))
    rows.sort(key=lambda item: (int(item[0]), int(item[1]), int(item[2])))
    out: List[SliceProjectionCandidate] = []
    for step, _d2, target_label, target_slice, target_y, target_x in rows[: int(max_candidates)]:
        out.append(SliceProjectionCandidate(
            source_label=int(seed.label),
            target_label=int(target_label),
            source_point=(int(s0), int(y0), int(x0)),
            target_point=(int(target_slice), int(target_y), int(target_x)),
            slice_distance=int(step),
        ))
    return out

def _find_slice_projection_candidates_python(
    labels_real: object,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    max_candidates: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
    slice_luts: Optional['SliceLocalLabelLUTs'] = None,
) -> List[SliceProjectionCandidate]:
    # Local import keeps the package dependency graph acyclic.
    from .topology import SparseSliceLabelStore

    if int(max_slice_distance) <= 0 or int(max_candidates) <= 0:
        return []

    s0, y0, x0 = seed.point
    num_slices = int(labels_real.shape[0])
    if num_slices <= 0:
        return []

    local_cache = (
        component_cache
        if component_cache is not None
        else SliceComponentTableCache(labels_real, slice_luts=slice_luts)
    )
    source_record, source_anchor = local_cache.find_record_for_point(int(s0), int(seed.label), (int(y0), int(x0)))
    if source_record is None or source_anchor is None or int(source_record.area) <= 0:
        return []

    max_steps = min(int(max_slice_distance), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(max_slice_distance)
    if int(max_steps) <= 0:
        return []

    cropped_sdf = local_cache.get_projection_sdf(
        source_record,
        max_slice_distance=int(max_steps),
        search_angle_deg=float(search_angle_deg),
    )
    sdf = np.asarray(cropped_sdf.sdf, dtype=np.float32)
    crop_y0 = int(cropped_sdf.origin_y)
    crop_x0 = int(cropped_sdf.origin_x)
    crop_y1 = int(crop_y0 + int(sdf.shape[0]))
    crop_x1 = int(crop_x0 + int(sdf.shape[1]))

    slope = math.tan(math.radians(float(search_angle_deg)))
    full_w = int(labels_real.shape[2])
    # target_label -> (step, unrolled d2, candidate); d2 is kept separately because the
    # candidate's target_point is in the target slice's own (actual) coordinates, which
    # differ from unrolled projection coordinates for wrap-crossing steps.
    found: Dict[int, Tuple[int, int, SliceProjectionCandidate]] = {}

    for step in range(1, int(max_steps) + 1):
        s_raw = int(s0 + int(seed.direction_sign) * step)
        mirrored = False
        if bool(wrap_axis):
            s = int(s_raw % int(num_slices))
            # a step across the radial 0°/180° wrap lands in a frame whose
            # u axis is REVERSED relative to the projection cone. max_steps <= num_slices-1
            # caps the walk at a single crossing.
            mirrored = bool(s_raw < 0 or s_raw >= int(num_slices))
        elif s_raw < 0 or s_raw >= num_slices:
            break
        else:
            s = s_raw

        threshold = -float(slope) * float(step)
        projection = sdf >= threshold
        if not np.any(projection):
            if float(search_angle_deg) < 0.0:
                break
            continue

        if isinstance(labels_real, SparseSliceLabelStore):
            # materialize only the SDF window. A wrap-crossing projection requests
            # the corresponding actual-u window and reverses that small crop back into
            # unrolled projection coordinates.
            if mirrored:
                actual_x0 = int(full_w - crop_x1)
                actual_x1 = int(full_w - crop_x0)
                labels_crop = labels_real.read_window(
                    int(s), crop_y0, crop_y1, actual_x0, actual_x1,
                )[:, ::-1]
            else:
                labels_crop = labels_real.read_window(
                    int(s), crop_y0, crop_y1, crop_x0, crop_x1,
                )
        else:
            slice_view = np.asarray(labels_real[int(s)])
            if mirrored:
                slice_view = slice_view[:, ::-1]
            labels_crop = slice_view[crop_y0:crop_y1, crop_x0:crop_x1]
        if slice_luts is not None:
            # local-id raster — canonicalize just the SDF crop (small gather)
            # instead of relying on a full-volume compact relabel pass.
            labels_crop = slice_luts.lut_for(int(s))[labels_crop]
        overlap = projection & (labels_crop > 0) & (labels_crop != int(seed.label))
        if not np.any(overlap):
            continue

        ys_local, xs_local = np.nonzero(overlap)
        lbls = labels_crop[ys_local, xs_local].astype(np.int64, copy=False)
        for target_label in np.unique(lbls):
            target_label_i = int(target_label)
            if target_label_i <= 0 or target_label_i == int(seed.label) or target_label_i in found:
                continue
            use = lbls == target_label_i
            ys_t = ys_local[use]
            xs_t = xs_local[use]
            if ys_t.size == 0:
                continue
            ys_global = ys_t.astype(np.int64, copy=False) + int(crop_y0)
            xs_global = xs_t.astype(np.int64, copy=False) + int(crop_x0)
            d2 = (ys_global - int(source_anchor[0])) ** 2 + (xs_global - int(source_anchor[1])) ** 2
            idx = int(np.argmin(d2))
            x_actual = int(full_w - 1 - int(xs_global[idx])) if mirrored else int(xs_global[idx])
            found[target_label_i] = (int(step), int(d2[idx]), SliceProjectionCandidate(
                source_label=int(seed.label),
                target_label=target_label_i,
                source_point=(int(s0), int(y0), int(x0)),
                target_point=(int(s), int(ys_global[idx]), x_actual),
                slice_distance=int(step),
            ))

        if len(found) >= int(max_candidates):
            break

    ordered = sorted(
        found.items(),
        key=lambda item: (int(item[1][0]), int(item[1][1]), int(item[0])),
    )
    return [candidate for _label, (_step, _d2, candidate) in ordered[: int(max_candidates)]]

def _find_slice_projection_candidates(
    labels_real: np.ndarray,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    max_candidates: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
    slice_luts: Optional['SliceLocalLabelLUTs'] = None,
) -> List[SliceProjectionCandidate]:
    fast_candidates = _find_slice_projection_candidates_numba(
        labels_real=labels_real,
        seed=seed,
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        max_candidates=int(max_candidates),
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
        slice_luts=slice_luts,
    )
    if fast_candidates is not None:
        return fast_candidates
    return _find_slice_projection_candidates_python(
        labels_real=labels_real,
        seed=seed,
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        max_candidates=int(max_candidates),
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
        slice_luts=slice_luts,
    )

def _collect_walkback_source_points(
    labels_real: np.ndarray,
    label: int,
    start_point: Tuple[int, int, int],
    direction_sign: int,
    walk_back: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
) -> List[Tuple[int, int, int]]:
    if int(walk_back) <= 0:
        return []

    s0, y0, x0 = start_point
    num_slices = int(labels_real.shape[0])

    if component_cache is not None:
        current_record, current_anchor = component_cache.find_record_for_point(int(s0), int(label), (int(y0), int(x0)))
        if current_record is None or current_anchor is None:
            return []
        out: List[Tuple[int, int, int]] = []
        current_slice = int(s0)
        slice_w = int(labels_real.shape[2])
        max_walk = min(int(walk_back), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(walk_back)
        visited_slices = {int(current_slice)}
        for _ in range(int(max_walk)):
            next_slice_raw = int(current_slice - int(direction_sign))
            wrap_crossed = False
            if bool(wrap_axis):
                next_slice = int(next_slice_raw % int(num_slices))
                wrap_crossed = bool(next_slice_raw < 0 or next_slice_raw >= int(num_slices))
                if next_slice in visited_slices:
                    break
            elif next_slice_raw < 0 or next_slice_raw >= num_slices:
                break
            else:
                next_slice = next_slice_raw

            cmp_record = current_record
            cmp_anchor = current_anchor
            if wrap_crossed:
                # continuation across the radial 0°/180° wrap matches at
                # u -> width-1-u; mirror the near side into the far frame's coordinates.
                cmp_record = _component_record_mirrored_u(current_record, slice_w)
                cmp_anchor = (int(current_anchor[0]), int(slice_w - 1 - int(current_anchor[1])))
            next_record, next_anchor = component_cache.get(next_slice).find_branch_continuation(
                int(label),
                cmp_record,
                cmp_anchor,
            )
            if next_record is None or next_anchor is None:
                break
            out.append((int(next_slice), int(next_anchor[0]), int(next_anchor[1])))
            visited_slices.add(int(next_slice))
            current_slice = int(next_slice)
            current_record = next_record
            current_anchor = next_anchor
        return out

    current_component, current_anchor = _component_mask_and_anchor(labels_real[s0] == int(label), (y0, x0))
    if current_anchor is None or not np.any(current_component):
        return []

    out: List[Tuple[int, int, int]] = []
    current_slice = int(s0)
    slice_w = int(labels_real.shape[2])

    max_walk = min(int(walk_back), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(walk_back)
    visited_slices = {int(current_slice)}
    for _ in range(int(max_walk)):
        next_slice_raw = int(current_slice - int(direction_sign))
        wrap_crossed = False
        if bool(wrap_axis):
            next_slice = int(next_slice_raw % int(num_slices))
            wrap_crossed = bool(next_slice_raw < 0 or next_slice_raw >= int(num_slices))
            if next_slice in visited_slices:
                break
        elif next_slice_raw < 0 or next_slice_raw >= num_slices:
            break
        else:
            next_slice = next_slice_raw

        next_slice_mask = labels_real[next_slice] == int(label)
        if not np.any(next_slice_mask):
            break

        cmp_component = current_component
        cmp_anchor = current_anchor
        if wrap_crossed:
            # match across the radial 0°/180° wrap at u -> width-1-u.
            cmp_component = current_component[:, ::-1]
            cmp_anchor = (int(current_anchor[0]), int(slice_w - 1 - int(current_anchor[1])))
        next_component, next_anchor = _follow_branch_component(next_slice_mask, cmp_component, cmp_anchor)
        if next_anchor is None or not np.any(next_component):
            break

        out.append((int(next_slice), int(next_anchor[0]), int(next_anchor[1])))
        visited_slices.add(int(next_slice))
        current_slice = int(next_slice)
        current_component = next_component
        current_anchor = next_anchor

    return out

def interpolation_cache_bridge_sections_enabled() -> bool:
    """Reuse accepted min-radius scan sections during painting.

 ``YOLO_TTA_INTERPOLATION_CACHE_BRIDGE_SECTIONS=0`` restores recomputation."""
    return _env_flag('YOLO_TTA_INTERPOLATION_CACHE_BRIDGE_SECTIONS', True)

def interpolation_fused_bridge_merge_enabled() -> bool:
    """Restrict bridge merge/delta capture to rendered paste bboxes.

 ``YOLO_TTA_INTERPOLATION_FUSED_BRIDGE_MERGE=0`` restores full-slice sweeps."""
    return _env_flag('YOLO_TTA_INTERPOLATION_FUSED_BRIDGE_MERGE', True)

def interpolation_plan_batch_budget_bytes() -> int:
    """Maximum charged bytes retained in the accepted-plan render batch.

    ``YOLO_TTA_INTERPOLATION_PLAN_BATCH_MIB`` defaults to 512 MiB. Values below
    one MiB still retain one complete plan at a time, because a plan cannot be
    subdivided without changing the min-radius acceptance calculation.
    """
    return max(
        1024 ** 2,
        _env_int('YOLO_TTA_INTERPOLATION_PLAN_BATCH_MIB', 512) * (1024 ** 2),
    )

@dataclass(frozen=True)
class SliceBridgeRenderPlan:
    source_label: int
    target_label: int
    source_point: Tuple[int, int, int]
    target_point: Tuple[int, int, int]
    source_anchor: Tuple[int, int]
    # for a bridge crossing the radial 0°/180° wrap, target_anchor and
    # sdf1 are stored in source-side "unrolled" coordinates (u mirrored relative to the
    # target slice's own frame); the painter maps wrapped intermediate slices back.
    target_anchor: Tuple[int, int]
    steps: int
    sign: int
    num_slices: int
    sdf0: np.ndarray
    sdf1: np.ndarray
    # One-based output-decomposition coordinates. To preserve the historical bridge
    # union while emitting exactly ``walk_back * candidates`` layers, index 1 combines
    # the endpoint and first walked-back origin; indices 2..N are progressively earlier
    # source slices. Candidate indices follow nearest-first projection-search ordering.
    interpolation_walk_back_index: int = 1
    interpolation_candidate_index: int = 1
    # the min-radius acceptance scan already constructs every
    # intermediate bool section. Accepted plans retain those exact post-component-filter
    # arrays by step index so painting does not repeat SDF lerp/threshold/component work.
    cached_sections: List[Optional[np.ndarray]] = field(default_factory=list, compare=False, repr=False)

@dataclass
class SliceSeedBridgePlanResult:
    candidate_connections: int = 0
    accepted_connections: int = 0
    default_bridges: int = 0
    walk_back_bridges: int = 0
    skipped_by_min_radius: int = 0
    plans: List[SliceBridgeRenderPlan] = field(default_factory=list)


def interpolation_bridge_component_paths(
    root: Path,
    interpolation_walk_back: int,
    interpolation_candidates: int,
) -> List[Tuple[int, int, Path]]:
    """Return the deterministic walk-back/candidate delta layout for one pass.

    ``interpolation_walk_back`` retains its historical meaning: the number of additional
    source slices before the endpoint. Output index 1 combines the endpoint and first
    additional origin, while indices 2..N hold the remaining origins. Consequently the
    returned cardinality is exactly ``walk_back * candidates`` without changing the
    aggregate bridge union; zero walk-back produces no component stores.
    """
    return [
        (
            int(walk_back_index),
            int(candidate_index),
            Path(root) / (
                f'walkback{int(walk_back_index):02d}_'
                f'candidate{int(candidate_index):02d}.cvol'
            ),
        )
        for walk_back_index in range(1, max(0, int(interpolation_walk_back)) + 1)
        for candidate_index in range(1, max(0, int(interpolation_candidates)) + 1)
    ]

def _slice_bridge_plan_payload_bytes(plan: SliceBridgeRenderPlan) -> int:
    """Exact retained NumPy payload bytes owned by one accepted render plan."""
    total = int(plan.sdf0.nbytes) + int(plan.sdf1.nbytes)
    total += int(sum(int(section.nbytes) for section in plan.cached_sections if section is not None))
    return int(total)

def _slice_bridge_plan_charge_bytes(plan: SliceBridgeRenderPlan) -> int:
    """Payload plus conservative Python-plan/render-schedule overhead."""
    schedule_entries = max(0, int(plan.steps) - 1)
    return int(_slice_bridge_plan_payload_bytes(plan)) + 512 + int(schedule_entries) * 96

def _build_slice_endpoint_seeds(
    labels_real: np.ndarray,
    workers: int = 1,
    wrap_axis: bool = False,
    component_cache: Optional['SliceComponentTableCache'] = None,
) -> Tuple[List[SliceEndpointSeed], int]:
    """Build interpolation endpoint seeds with the per-slice component scan.

 Interpolation no longer uses skeletonization. Each labeled 3D object is scanned slice by
 slice; every 2D connected component is evaluated independently for overlap continuation
 into the previous and next slice. Components without continuation become endpoint seeds in
 the corresponding direction. Radial interpolation can wrap the slice/frame axis so frame 0
 and the final radial frame are considered adjacent."""
    # Local import keeps the package dependency graph acyclic.
    from .topology import build_slice_endpoint_seeds_from_label_volume

    return build_slice_endpoint_seeds_from_label_volume(
        labels_real,
        workers=int(workers),
        desc='Interpolation: endpoint seeds [scan]',
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
    )

def _build_linear_slice_bridge_plan(
    labels_real: np.ndarray,
    source_label: int,
    target_label: int,
    source_point: Tuple[int, int, int],
    target_point: Tuple[int, int, int],
    direction_sign: Optional[int] = None,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
) -> Optional[SliceBridgeRenderPlan]:
    s0, y0, x0 = source_point
    s1, y1, x1 = target_point
    if int(s0) == int(s1):
        return None

    num_slices = int(labels_real.shape[0])
    if bool(wrap_axis):
        sign = 1 if int(direction_sign or 1) >= 0 else -1
        if sign > 0:
            steps = int((int(s1) - int(s0)) % num_slices)
        else:
            steps = int((int(s0) - int(s1)) % num_slices)
    else:
        steps = int(abs(int(s1) - int(s0)))
        sign = 1 if int(s1) > int(s0) else -1
    if steps <= 0:
        return None

    # does the raw source->target walk cross the 0°/180° wrap? If so, the
    # target slice's u axis is REVERSED relative to the source side; build the morph in
    # source-side "unrolled" coordinates by mirroring the target component and anchor.
    raw_end = int(s0) + int(sign) * int(steps)
    wrap_crossed = bool(wrap_axis) and not (0 <= raw_end < num_slices)
    slice_w = int(labels_real.shape[2])

    if component_cache is not None:
        source_record, source_anchor = component_cache.find_record_for_point(int(s0), int(source_label), (int(y0), int(x0)))
        target_record, target_anchor = component_cache.find_record_for_point(int(s1), int(target_label), (int(y1), int(x1)))
        if source_record is None or target_record is None or source_anchor is None or target_anchor is None:
            return None
        if int(source_record.area) <= 0 or int(target_record.area) <= 0:
            return None
        if wrap_crossed:
            target_record = _component_record_mirrored_u(target_record, slice_w)
            target_anchor = (int(target_anchor[0]), int(slice_w - 1 - int(target_anchor[1])))

        half_width = _local_half_width_for_component_records(source_record, source_anchor, target_record, target_anchor)
        source_local = _component_record_to_local_canvas(source_record, source_anchor, half_width)
        target_local = _component_record_to_local_canvas(target_record, target_anchor, half_width)
    else:
        source_component, source_anchor = _component_mask_and_anchor(labels_real[s0] == int(source_label), (y0, x0))
        target_component, target_anchor = _component_mask_and_anchor(labels_real[s1] == int(target_label), (y1, x1))
        if source_anchor is None or target_anchor is None:
            return None
        if not np.any(source_component) or not np.any(target_component):
            return None
        if wrap_crossed:
            target_component = target_component[:, ::-1]
            target_anchor = (int(target_anchor[0]), int(slice_w - 1 - int(target_anchor[1])))

        half_width = _local_half_width_for_components(source_component, source_anchor, target_component, target_anchor)
        source_local = _component_to_local_canvas(source_component, source_anchor, half_width)
        target_local = _component_to_local_canvas(target_component, target_anchor, half_width)

    if not np.any(source_local) or not np.any(target_local):
        return None

    return SliceBridgeRenderPlan(
        source_label=int(source_label),
        target_label=int(target_label),
        source_point=(int(s0), int(y0), int(x0)),
        target_point=(int(s1), int(y1), int(x1)),
        source_anchor=(int(source_anchor[0]), int(source_anchor[1])),
        target_anchor=(int(target_anchor[0]), int(target_anchor[1])),
        steps=int(steps),
        sign=int(sign),
        num_slices=int(num_slices),
        sdf0=np.ascontiguousarray(_signed_distance_2d(source_local)),
        sdf1=np.ascontiguousarray(_signed_distance_2d(target_local)),
    )

def _estimate_linear_slice_bridge_min_radius_from_plan(
    plan: SliceBridgeRenderPlan,
    *,
    reject_at_or_below: float = 0.0,
    cache_sections: bool = False,
) -> float:
    """Estimate the minimum accepted bridge radius from the plan's cached endpoint distance fields."""
    source_local = np.asarray(plan.sdf0 >= 0.0, dtype=bool)
    target_local = np.asarray(plan.sdf1 >= 0.0, dtype=bool)
    if not np.any(source_local) or not np.any(target_local):
        return 0.0

    min_radius = float(min(float(np.max(plan.sdf0)), float(np.max(plan.sdf1))))
    threshold = float(reject_at_or_below)
    if threshold > 0.0 and min_radius <= threshold:
        return float(min_radius)
    section_cache = plan.cached_sections
    if bool(cache_sections):
        section_cache[:] = [None] * (int(plan.steps) + 1)
    for idx in range(1, int(plan.steps)):
        alpha = float(idx) / float(plan.steps)
        section = ((1.0 - alpha) * plan.sdf0 + alpha * plan.sdf1) >= 0.0
        if not np.any(section):
            return 0.0
        section = _keep_center_component_2d(section)
        if bool(cache_sections):
            # The painter only reads this array (wrap handling creates a reversed view),
            # so retaining the exact estimator result is bit-for-bit equivalent.
            section_cache[int(idx)] = section
        min_radius = min(min_radius, _component_max_radius(section))
        if threshold > 0.0 and min_radius <= threshold:
            return float(min_radius)
    return float(min_radius)

def _paint_linear_slice_bridge_plan_onto_slice(
    dest_slice: np.ndarray,
    plan: SliceBridgeRenderPlan,
    step_idx: int,
    *,
    dst_bbox_union: Optional[List[int]] = None,
    paint_value: int = 1,
    binary_destination: bool = True,
) -> int:
    if int(step_idx) <= 0 or int(step_idx) >= int(plan.steps):
        return 0

    alpha = float(step_idx) / float(plan.steps)
    section: Optional[np.ndarray] = None
    if int(step_idx) < len(plan.cached_sections):
        section = plan.cached_sections[int(step_idx)]
    if section is None:
        section = ((1.0 - alpha) * plan.sdf0 + alpha * plan.sdf1) >= 0.0
        if not np.any(section):
            return 0
        section = _keep_center_component_2d(section)
    center_y = (1.0 - alpha) * float(plan.source_anchor[0]) + alpha * float(plan.target_anchor[0])
    center_x = (1.0 - alpha) * float(plan.source_anchor[1]) + alpha * float(plan.target_anchor[1])
    # the morph runs in source-side "unrolled" coordinates. Intermediate
    # slices beyond the radial 0°/180° boundary store the u-mirrored frame, so flip the
    # section (the local canvas is centered, so a column flip mirrors it about its
    # center) and mirror the center's u there. Non-wrap plans never leave [0, num_slices).
    s_raw = int(plan.source_point[0]) + int(plan.sign) * int(step_idx)
    if int(plan.num_slices) > 0 and (s_raw < 0 or s_raw >= int(plan.num_slices)):
        section = section[:, ::-1]
        center_x = float(int(dest_slice.shape[1]) - 1) - center_x
    return _paste_local_mask_onto_slice(
        dest_slice,
        section,
        (center_y, center_x),
        dst_bbox_union=dst_bbox_union,
        paint_value=int(paint_value),
        binary_destination=bool(binary_destination),
    )

def _plan_slice_seed_bridges(
    labels_real: np.ndarray,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_min_radius: float,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
    slice_luts: Optional['SliceLocalLabelLUTs'] = None,
    gpu_renderer: Optional[CudaInterpolationRenderer] = None,
    gpu_required: bool = False,
) -> SliceSeedBridgePlanResult:
    result = SliceSeedBridgePlanResult()

    # Keep the historical bridge set: the endpoint plus N additional source slices.
    # Layer 1 coalesces the endpoint and nearest walked-back origin so the externally
    # visible decomposition is still exactly N x C rather than (N + 1) x C.
    walk_back_count = max(0, int(interpolation_walk_back))

    candidates = _find_slice_projection_candidates(
        labels_real=labels_real,
        seed=seed,
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        max_candidates=int(interpolation_candidates),
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
        slice_luts=slice_luts,
    )
    if not candidates:
        return result

    result.candidate_connections = int(len(candidates))
    source_points = [seed.point] + _collect_walkback_source_points(
        labels_real=labels_real,
        label=int(seed.label),
        start_point=seed.point,
        direction_sign=int(seed.direction_sign),
        walk_back=int(walk_back_count),
        wrap_axis=bool(wrap_axis),
        component_cache=component_cache,
    )

    for candidate_idx, candidate in enumerate(candidates, start=1):
        accepted_this_candidate = False
        for source_index, src_point in enumerate(source_points):
            plan = _build_linear_slice_bridge_plan(
                labels_real=labels_real,
                source_label=int(candidate.source_label),
                target_label=int(candidate.target_label),
                source_point=src_point,
                target_point=candidate.target_point,
                direction_sign=int(seed.direction_sign),
                wrap_axis=bool(wrap_axis),
                component_cache=component_cache,
            )
            if plan is None:
                continue

            if float(interpolate_min_radius) > 0.0:
                bridge_radius: Optional[float] = None
                if gpu_renderer is not None and gpu_renderer.available:
                    try:
                        bridge_radius = gpu_renderer.estimate_min_radius(
                            plan,
                            reject_at_or_below=float(interpolate_min_radius),
                            cache_sections=bool(interpolation_cache_bridge_sections_enabled()),
                            cache_host_sections=False,
                        )
                    except Exception as exc:
                        first_failure = gpu_renderer.disable(exc)
                        if bool(gpu_required):
                            raise RuntimeError(
                                'required CUDA interpolation radius evaluation failed'
                            ) from exc
                        if first_failure:
                            print(
                                f'Warning: CUDA interpolation radius evaluation failed ({exc}); '
                                'using the CPU bridge evaluator for this and remaining plans.'
                            )
                if bridge_radius is None:
                    bridge_radius = _estimate_linear_slice_bridge_min_radius_from_plan(
                        plan,
                        reject_at_or_below=float(interpolate_min_radius),
                        cache_sections=bool(interpolation_cache_bridge_sections_enabled()),
                    )
                if bridge_radius <= float(interpolate_min_radius):
                    result.skipped_by_min_radius += 1
                    if gpu_renderer is not None:
                        # Rejected plans never enter the render batch, so retire their
                        # device SDFs here instead of waiting for LRU pressure.
                        gpu_renderer.release_plans((plan,))
                    continue

            if source_index == 0:
                result.default_bridges += 1
            else:
                result.walk_back_bridges += 1

            if not accepted_this_candidate:
                result.accepted_connections += 1
                accepted_this_candidate = True

            result.plans.append(dataclasses_replace(
                plan,
                interpolation_walk_back_index=(
                    max(1, int(source_index)) if int(walk_back_count) > 0 else 0
                ),
                interpolation_candidate_index=int(candidate_idx),
            ))

    return result

def _build_slice_bridge_render_schedule(
    plans: Sequence[SliceBridgeRenderPlan],
    num_slices: int,
    wrap_axis: bool = False,
) -> List[List[Tuple[int, int]]]:
    schedule: List[List[Tuple[int, int]]] = [[] for _ in range(int(num_slices))]
    for plan_idx, plan in enumerate(plans):
        start_slice = int(plan.source_point[0])
        for step_idx in range(1, int(plan.steps)):
            s = int(start_slice + int(plan.sign) * step_idx)
            if bool(wrap_axis):
                s = int(s % int(num_slices))
                schedule[s].append((int(plan_idx), int(step_idx)))
            elif 0 <= s < int(num_slices):
                schedule[s].append((int(plan_idx), int(step_idx)))
    return schedule

def interpolate_view_volume_pass_inplace(
    mask_mm: np.ndarray,
    work_dir: Path,
    pass_tag: str,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_min_radius: float,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    wrap_axis: bool = False,
    bridge_delta_path: Optional[Path] = None,
    bridge_component_dir: Optional[Path] = None,
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run one interpolation pass in place.
    
    Optional outputs capture exact added voxels and sparse label metadata; accepted bridge sections are reused during painting."""
    # Local import keeps the package dependency graph acyclic.
    from .topology import (
        _local_label_store_dtype,
        interpolation_skip_compact_relabel_enabled,
        label_foreground_volume_streaming,
    )

    if int(max_slice_distance) <= 0:
        return {
            'num_objects': 0,
            'num_endpoints': 0,
            'candidate_connections': 0,
            'accepted_connections': 0,
            'default_bridges': 0,
            'walk_back_bridges': 0,
            'skipped_by_min_radius': 0,
            'added_voxels': 0,
            'skipped': True,
            'wrap_axis': bool(wrap_axis),
            'endpoint_method': 'slice_component_scan',
            'planning_backend': interpolation_planning_backend_name(),
        }

    work_dir.mkdir(parents=True, exist_ok=True)
    component_specs = (
        interpolation_bridge_component_paths(
            Path(bridge_component_dir),
            int(interpolation_walk_back),
            int(interpolation_candidates),
        )
        if bridge_component_dir is not None else []
    )
    component_count = int(len(component_specs))
    if int(component_count) <= 8:
        component_word_dtype = np.dtype(np.uint8)
    elif int(component_count) <= 16:
        component_word_dtype = np.dtype(np.uint16)
    elif int(component_count) <= 32:
        component_word_dtype = np.dtype(np.uint32)
    else:
        component_word_dtype = np.dtype(np.uint64)
    component_bits_per_word = int(component_word_dtype.itemsize * 8)
    component_word_count = (
        (int(component_count) + int(component_bits_per_word) - 1) // int(component_bits_per_word)
        if int(component_count) > 0 else 0
    )
    component_membership_paths = [
        Path(bridge_component_dir) / '_membership' / (
            f'word{int(word_index):02d}.{component_word_dtype.name}.dat'
        )
        for word_index in range(int(component_word_count))
    ]

    def _discard_failed_component_outputs() -> None:
        if bool(keep_temp):
            return
        for membership_path in component_membership_paths:
            try:
                membership_path.unlink(missing_ok=True)
            except Exception:
                pass
        if component_membership_paths:
            try:
                component_membership_paths[0].parent.rmdir()
            except OSError:
                pass
        for _walk_back_index, _candidate_index, component_path in component_specs:
            shutil.rmtree(component_path, ignore_errors=True)
        if bridge_component_dir is not None:
            try:
                Path(bridge_component_dir).rmdir()
            except OSError:
                pass

    component_bit_layout: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for flat_index, (walk_back_index, candidate_index, _component_path) in enumerate(component_specs):
        component_bit_layout[(int(walk_back_index), int(candidate_index))] = (
            int(flat_index // int(component_bits_per_word)),
            int(1 << int(flat_index % int(component_bits_per_word))),
        )
    # Publish a compact empty cvol for every configured combination before any early return.
    # Render scratch canvases are created only after endpoint discovery proves planning work
    # exists, then compacted one at a time before this pass returns.
    try:
        for _walk_back_index, _candidate_index, component_path in component_specs:
            empty_writer = IncrementalRawBBoxMaskStoreWriter(
                shape=tuple(int(v) for v in np.asarray(mask_mm).shape),
                store_dir=component_path,
                format_name=CVOL_FORMAT,
                desc=(
                    f'Empty interpolation component {pass_tag} walkback '
                    f'{int(_walk_back_index)} candidate {int(_candidate_index)}'
                ),
                extra_meta={
                    'interpolation_walk_back_index': int(_walk_back_index),
                    'interpolation_candidate_index': int(_candidate_index),
                    'added_voxels': 0,
                },
                force_path_backed=True,
            )
            empty_writer.consume_empty_range(0, int(mask_mm.shape[0]))
            empty_writer.finalize()
    except BaseException:
        _discard_failed_component_outputs()
        raise

    def _bridge_component_stats(
        added_counts: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> List[Dict[str, object]]:
        counts = added_counts or {}
        return [
            {
                'walk_back_index': int(walk_back_index),
                'candidate_index': int(candidate_index),
                'path': str(component_path),
                'added_voxels': int(counts.get((int(walk_back_index), int(candidate_index)), 0)),
            }
            for walk_back_index, candidate_index, component_path in component_specs
        ]
    # Keep per-slice local ids in the label store and consume them through exported LUTs.
    # Size admission with the actual local dtype so a uint16-capable pass is not forced to disk.
    skip_relabel = interpolation_skip_compact_relabel_enabled()
    estimated_label_dtype = _local_label_store_dtype(compact_relabel=not skip_relabel)
    estimated_bytes = estimate_interpolation_workspace_bytes(
        tuple(int(x) for x in mask_mm.shape), label_dtype=estimated_label_dtype,
    )
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(estimated_bytes, reserve_bytes=reserve_bytes)
    if use_in_memory:
        print(f"Interpolation workspace ({pass_tag}): in-memory ({estimated_bytes / GIB:.1f} GiB estimated)")
    else:
        print(f"Interpolation workspace ({pass_tag}): disk-backed ({estimated_bytes / GIB:.1f} GiB estimated) -> {work_dir}")

    # Component tables + candidate kernels canonicalize at read time, deleting the
    # full-volume compact-relabel read+write pass over the label store.
    label_stats: Dict[str, object] = {}
    label_work_prefix = work_dir / f'{pass_tag}_labels'
    try:
        labels_mm, num_objects, label_paths = label_foreground_volume_streaming(
            mask_mm,
            label_work_prefix,
            prefer_memory=use_in_memory,
            reserve_bytes=reserve_bytes,
            wrap_axis=bool(wrap_axis),
            workers=int(workers),
            compact_relabel=not skip_relabel,
            component_stats_out=label_stats if skip_relabel else None,
            known_slice_any=known_slice_any,
            known_slice_bboxes=known_slice_bboxes,
            sparse_local_labels=bool(skip_relabel),
        )
    except BaseException:
        if not bool(keep_temp):
            for suffix in ('.fg_labels.u16.dat', '.fg_labels.u32.dat'):
                try:
                    label_work_prefix.with_suffix(suffix).unlink(missing_ok=True)
                except Exception:
                    pass
        _discard_failed_component_outputs()
        raise
    slice_luts: Optional[SliceLocalLabelLUTs] = (
        label_stats.get('slice_local_luts') if skip_relabel else None  # type: ignore[assignment]
    )

    if int(num_objects) <= 1:
        del labels_mm
        if not keep_temp:
            for p in label_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        return {
            'num_objects': int(num_objects),
            'num_endpoints': 0,
            'candidate_connections': 0,
            'accepted_connections': 0,
            'default_bridges': 0,
            'walk_back_bridges': 0,
            'skipped_by_min_radius': 0,
            'added_voxels': 0,
            'skipped': int(num_objects) <= 1,
            'wrap_axis': bool(wrap_axis),
            'endpoint_method': 'slice_component_scan',
            'planning_backend': interpolation_planning_backend_name(),
            'bridge_component_deltas': _bridge_component_stats(),
        }

    if skip_relabel and slice_luts is None:
        # Defensive: a local-id raster without LUTs would be misread downstream. This can
        # only happen through an unexpected labeler edit; fail loudly rather than corrupt.
        close_memmap_array(labels_mm)
        if not bool(keep_temp):
            for p in label_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        _discard_failed_component_outputs()
        raise RuntimeError(f'interpolation pass {pass_tag}: local-id label store without slice LUTs')

    component_cache = SliceComponentTableCache(labels_mm, slice_luts=slice_luts)
    worker_count = choose_slice_parallel_workers(int(workers), int(labels_mm.shape[0]))
    try:
        seeds, num_endpoints = _build_slice_endpoint_seeds(
            labels_mm,
            workers=worker_count,
            wrap_axis=bool(wrap_axis),
            component_cache=component_cache,
        )
    except BaseException:
        component_cache.clear()
        close_memmap_array(labels_mm)
        if not bool(keep_temp):
            for p in label_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        _discard_failed_component_outputs()
        raise
    if not seeds:
        del labels_mm
        if not keep_temp:
            for p in label_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        return {
            'num_objects': int(num_objects),
            'num_endpoints': int(num_endpoints),
            'candidate_connections': 0,
            'accepted_connections': 0,
            'default_bridges': 0,
            'walk_back_bridges': 0,
            'skipped_by_min_radius': 0,
            'added_voxels': 0,
            'skipped': False,
            'wrap_axis': bool(wrap_axis),
            'endpoint_method': 'slice_component_scan',
            'planning_backend': interpolation_planning_backend_name(),
            'bridge_component_deltas': _bridge_component_stats(),
        }

    bridge_path: Optional[Path] = None
    bridge_mm: Optional[np.ndarray] = None
    component_membership_mms: List[np.memmap] = []
    pending_membership_mm: Optional[np.memmap] = None
    try:
        if component_specs:
            # One packed membership word replaces both the aggregate bridge canvas and up to
            # 8/16/32/64 per-combination uint8 canvases. Additional words are added only after
            # exhausting all bits in the current machine word.
            bridge_mm = None
        elif use_in_memory:
            bridge_mm = np.zeros(mask_mm.shape, dtype=np.uint8)
        else:
            bridge_path = work_dir / f'{pass_tag}_bridges.u8.dat'
            bridge_mm = np.memmap(bridge_path, dtype=np.uint8, mode='w+', shape=mask_mm.shape)
        if bridge_mm is not None:
            numa_interleave_memory(bridge_mm, desc='Interpolation bridge canvas')  #
        for membership_path in component_membership_paths:
            membership_path.parent.mkdir(parents=True, exist_ok=True)
            pending_membership_mm = np.memmap(
                membership_path,
                dtype=component_word_dtype,
                mode='w+',
                shape=tuple(int(v) for v in np.asarray(mask_mm).shape),
            )
            component_membership_mms.append(pending_membership_mm)
            pending_membership_mm = None
    except BaseException:
        close_memmap_array(pending_membership_mm)
        pending_membership_mm = None
        for membership_mm in component_membership_mms:
            close_memmap_array(membership_mm)
        component_membership_mms.clear()
        close_memmap_array(bridge_mm)
        bridge_mm = None
        component_cache.clear()
        close_memmap_array(labels_mm)
        if not bool(keep_temp):
            for p in label_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            if bridge_path is not None:
                try:
                    bridge_path.unlink(missing_ok=True)
                except Exception:
                    pass
        _discard_failed_component_outputs()
        raise
    component_added_by_slice: Dict[Tuple[int, int], np.ndarray] = {
        key: np.zeros((int(mask_mm.shape[0]),), dtype=np.int64)
        for key in component_bit_layout
    }

    candidate_connections = 0
    accepted_connections = 0
    default_bridges = 0
    walk_back_bridges = 0
    skipped_by_min_radius = 0
    added_voxels = 0
    bridge_delta_written = False
    # interpolation already visits every bridge-delta crop while it is hot.
    # Preserve the exact per-azimuth bounds and per-t row occupancy here so radial
    # backprojection does not rediscover them by strided scans over the dense delta file.
    bridge_delta_slice_bboxes: Optional[np.ndarray] = None
    bridge_delta_rows_by_slice: Optional[np.ndarray] = None
    plan_batch: List[SliceBridgeRenderPlan] = []
    plan_batch_payload_bytes = 0
    plan_batch_charge_bytes = 0
    plan_batch_budget_bytes = int(interpolation_plan_batch_budget_bytes())
    plan_batch_peak_payload_bytes = 0
    plan_batch_peak_charge_bytes = 0
    plan_batch_peak_plans = 0
    plan_batches_rendered = 0
    planned_plan_count = 0
    cache_telemetry: Dict[str, int] = {}
    gpu_renderer: Optional[CudaInterpolationRenderer] = None
    gpu_renderer_status = 'not attempted'
    gpu_renderer_telemetry: Dict[str, object] = {}
    gpu_required = bool(gpu_interpolation_required())
    gpu_render_batches = 0
    gpu_render_fallback_batches = 0
    render_workers = choose_slice_parallel_workers(int(workers), int(mask_mm.shape[0]))
    fused_bridge_merge = bool(interpolation_fused_bridge_merge_enabled())
    rendered_paste_bboxes = (
        np.zeros((int(mask_mm.shape[0]), 4), dtype=np.int64)
        if fused_bridge_merge else None
    )
    scheduled_slice_flags = np.zeros((int(mask_mm.shape[0]),), dtype=bool)

    try:
        gpu_renderer, gpu_renderer_status = create_cuda_interpolation_renderer(
            process_worker=bool(interpolation_process_worker_active()),
        )
        if gpu_renderer is None and bool(gpu_required):
            raise RuntimeError(
                'YOLO_TTA_GPU_INTERPOLATION_REQUIRED=1 but CUDA interpolation is '
                f'unavailable: {gpu_renderer_status}'
            )
        if gpu_renderer is not None:
            print(
                f'CUDA bridge interpolation active on {gpu_renderer_status}: '
                'min-radius morphology and crop-bounded bridge painting run on-device; '
                'YOLO_TTA_GPU_INTERPOLATION=0 disables.'
            )
        # Endpoint discovery returns label-major order. Planning is independent per seed
        # and the renderer only ORs sections, so consume the same seeds slice-major to
        # keep bounded component tables hot instead of repeatedly rebuilding distant z.
        seeds.sort(key=lambda seed: (
            int(seed.point[0]), int(seed.direction_sign), int(seed.label),
            int(seed.point[1]), int(seed.point[2]),
        ))
        plan_workers = choose_slice_parallel_workers(int(workers), len(seeds))
        max_seed_planning_cost = max(
            max(1, int(getattr(seed, 'planning_cost', 1))) for seed in seeds
        )
        seed_schedule_window = _rebalance_slice_major_endpoint_seeds(
            seeds,
            plan_workers=int(plan_workers),
        )

        def _render_plan_batch() -> None:
            nonlocal added_voxels
            nonlocal plan_batch_payload_bytes, plan_batch_charge_bytes
            nonlocal plan_batch_peak_payload_bytes, plan_batch_peak_charge_bytes
            nonlocal plan_batch_peak_plans, plan_batches_rendered
            nonlocal gpu_render_batches, gpu_render_fallback_batches
            if not plan_batch:
                return

            plan_batch_peak_payload_bytes = max(
                int(plan_batch_peak_payload_bytes), int(plan_batch_payload_bytes)
            )
            plan_batch_peak_charge_bytes = max(
                int(plan_batch_peak_charge_bytes), int(plan_batch_charge_bytes)
            )
            plan_batch_peak_plans = max(int(plan_batch_peak_plans), int(len(plan_batch)))

            schedule = _build_slice_bridge_render_schedule(
                plan_batch, int(mask_mm.shape[0]), wrap_axis=bool(wrap_axis),
            )
            batch_slices = [
                int(z) for z in range(int(mask_mm.shape[0])) if schedule[int(z)]
            ]
            if batch_slices:
                scheduled_slice_flags[np.asarray(batch_slices, dtype=np.int64)] = True
            batch_added_counts = np.zeros((len(batch_slices),), dtype=np.int64)

            def _initial_bbox_union(z: int) -> Optional[List[int]]:
                if rendered_paste_bboxes is None:
                    return None
                old_y0, old_x0, old_y1, old_x1 = (
                    int(v) for v in rendered_paste_bboxes[int(z)]
                )
                if old_y0 < old_y1 and old_x0 < old_x1:
                    return [old_y0, old_x0, old_y1, old_x1]
                return [int(mask_mm.shape[1]), int(mask_mm.shape[2]), 0, 0]

            def _extend_bbox_union(
                bbox_union: Optional[List[int]],
                bbox: Optional[Tuple[int, int, int, int]],
            ) -> None:
                if bbox_union is None or bbox is None:
                    return
                bbox_union[0] = min(int(bbox_union[0]), int(bbox[0]))
                bbox_union[1] = min(int(bbox_union[1]), int(bbox[1]))
                bbox_union[2] = max(int(bbox_union[2]), int(bbox[2]))
                bbox_union[3] = max(int(bbox_union[3]), int(bbox[3]))

            def _render_batch_slice_cpu(list_idx: int) -> None:
                z = int(batch_slices[int(list_idx)])
                bbox_union = _initial_bbox_union(int(z))
                local_added = 0
                for plan_idx, step_idx in schedule[int(z)]:
                    plan = plan_batch[int(plan_idx)]
                    component_layout = component_bit_layout.get((
                        int(plan.interpolation_walk_back_index),
                        int(plan.interpolation_candidate_index),
                    ))
                    if component_layout is not None:
                        word_index, bit_value = component_layout
                        local_added += _paint_linear_slice_bridge_plan_onto_slice(
                            component_membership_mms[int(word_index)][int(z)],
                            plan,
                            int(step_idx),
                            dst_bbox_union=bbox_union,
                            paint_value=int(bit_value),
                            binary_destination=False,
                        )
                    elif bridge_mm is not None:
                        local_added += _paint_linear_slice_bridge_plan_onto_slice(
                            bridge_mm[int(z)],
                            plan,
                            int(step_idx),
                            dst_bbox_union=bbox_union,
                        )
                # If CUDA committed earlier slices before a later slice failed, replaying
                # the whole batch is safe because painting is OR-idempotent.  Preserve the
                # already-counted additions and add only the bits CPU replay newly sets.
                batch_added_counts[int(list_idx)] += np.int64(local_added)
                if rendered_paste_bboxes is not None and bbox_union is not None:
                    rendered_paste_bboxes[int(z)] = np.asarray(bbox_union, dtype=np.int64)

            gpu_batch_complete = False
            if gpu_renderer is not None and gpu_renderer.available:
                try:
                    for list_idx, z in enumerate(batch_slices):
                        bbox_union = _initial_bbox_union(int(z))
                        # One destination group per packed membership word (or -1 for
                        # the ordinary binary bridge canvas). Different logical bits in
                        # the same word remain in schedule order and are ORed together.
                        grouped_jobs: Dict[int, List[Tuple[object, int, int]]] = {}
                        for plan_idx, step_idx in schedule[int(z)]:
                            plan = plan_batch[int(plan_idx)]
                            component_layout = component_bit_layout.get((
                                int(plan.interpolation_walk_back_index),
                                int(plan.interpolation_candidate_index),
                            ))
                            if component_layout is None:
                                group_index, paint_value = -1, 1
                            else:
                                group_index, paint_value = component_layout
                            grouped_jobs.setdefault(int(group_index), []).append((
                                plan, int(step_idx), int(paint_value),
                            ))

                        local_added = 0
                        for group_index, jobs in grouped_jobs.items():
                            if int(group_index) < 0:
                                if bridge_mm is None:  # pragma: no cover - invalid layout
                                    raise RuntimeError('missing aggregate bridge canvas')
                                destination = bridge_mm[int(z)]
                            else:
                                destination = component_membership_mms[int(group_index)][int(z)]
                            gpu_result = gpu_renderer.render_slice(destination, jobs)
                            local_added += int(gpu_result.added_voxels)
                            _extend_bbox_union(bbox_union, gpu_result.bbox)
                        batch_added_counts[int(list_idx)] += np.int64(local_added)
                        if rendered_paste_bboxes is not None and bbox_union is not None:
                            rendered_paste_bboxes[int(z)] = np.asarray(
                                bbox_union, dtype=np.int64,
                            )
                    gpu_render_batches += 1
                    gpu_batch_complete = True
                except Exception as exc:
                    first_failure = gpu_renderer.disable(exc)
                    gpu_render_fallback_batches += 1
                    if bool(gpu_required):
                        raise RuntimeError(
                            'required CUDA interpolation bridge rendering failed'
                        ) from exc
                    if first_failure:
                        print(
                            f'Warning: CUDA interpolation bridge rendering failed ({exc}); '
                            'replaying the batch on CPU and keeping CUDA disabled.'
                        )

            if not gpu_batch_complete:
                parallel_for_indices(
                    len(batch_slices),
                    _render_batch_slice_cpu,
                    max_workers=choose_slice_parallel_workers(
                        int(render_workers), max(1, len(batch_slices)),
                    ),
                    desc='Interpolation: render bounded plan batch',
                    show_progress=False,
                )
            added_voxels += int(np.sum(batch_added_counts, dtype=np.int64))
            plan_batches_rendered += 1
            schedule.clear()
            if gpu_renderer is not None:
                try:
                    gpu_renderer.release_plans(plan_batch)
                except Exception:
                    pass
            plan_batch.clear()
            plan_batch_payload_bytes = 0
            plan_batch_charge_bytes = 0

        def _plan_seed(idx: int) -> SliceSeedBridgePlanResult:
            return _plan_slice_seed_bridges(
                labels_real=labels_mm,
                seed=seeds[int(idx)],
                max_slice_distance=int(max_slice_distance),
                search_angle_deg=float(search_angle_deg),
                interpolation_walk_back=int(interpolation_walk_back),
                interpolation_candidates=int(interpolation_candidates),
                interpolate_min_radius=float(interpolate_min_radius),
                wrap_axis=bool(wrap_axis),
                component_cache=component_cache,
                slice_luts=slice_luts,
                gpu_renderer=gpu_renderer,
                gpu_required=bool(gpu_required),
            )

        # A completed future owns its plan arrays until the consumer accepts it.
        # Keep at most one result slot per planner worker instead of retaining a second
        # whole wave behind the explicit byte-capped render batch.
        pending = max(1, plan_workers)
        if plan_workers <= 1:
            iterable = (_plan_seed(int(idx)) for idx in range(len(seeds)))
        else:
            iterable = parallel_map_unordered(
                _plan_seed,
                range(len(seeds)),
                max_workers=plan_workers,
                max_pending=pending,
            )

        for seed_result in tqdm(iterable, total=len(seeds), desc='Interpolation: seed planning'):
            candidate_connections += int(seed_result.candidate_connections)
            accepted_connections += int(seed_result.accepted_connections)
            default_bridges += int(seed_result.default_bridges)
            walk_back_bridges += int(seed_result.walk_back_bridges)
            skipped_by_min_radius += int(seed_result.skipped_by_min_radius)
            if seed_result.plans:
                seed_plans = seed_result.plans
                seed_result.plans = []
                for plan in seed_plans:
                    payload_bytes = int(_slice_bridge_plan_payload_bytes(plan))
                    charge_bytes = int(_slice_bridge_plan_charge_bytes(plan))
                    if (
                        plan_batch
                        and int(plan_batch_charge_bytes) + int(charge_bytes)
                        > int(plan_batch_budget_bytes)
                    ):
                        _render_plan_batch()
                    plan_batch.append(plan)
                    plan_batch_payload_bytes += int(payload_bytes)
                    plan_batch_charge_bytes += int(charge_bytes)
                    planned_plan_count += 1
                    if int(plan_batch_charge_bytes) >= int(plan_batch_budget_bytes):
                        _render_plan_batch()
                del plan
                seed_plans.clear()
        # The loop variable otherwise retains the final seed result (and its section
        # caches) independently of the bounded plan batch.
        del seed_result
        del iterable
        _render_plan_batch()
        seeds.clear()

        cache_telemetry = component_cache.telemetry()
        component_cache.clear()
        print(
            f'Interpolation planner ({pass_tag}): {planned_plan_count:,} plan(s) in '
            f'{plan_batches_rendered:,} bounded batch(es); plan payload peak '
            f'{plan_batch_peak_payload_bytes / GIB:.2f} GiB '
            f'({plan_batch_peak_plans:,} plans, {plan_batch_peak_charge_bytes / GIB:.2f} GiB charged); '
            f'component-table payload peak '
            f'{cache_telemetry.get("component_table_cache_peak_payload_bytes", 0) / GIB:.2f} GiB; '
            f'projection-SDF peak '
            f'{cache_telemetry.get("projection_sdf_cache_peak_bytes", 0) / GIB:.2f} GiB; '
            f'seed schedule '
            f'{"cost-balanced/" + str(seed_schedule_window) if seed_schedule_window else "slice-major"} '
            f'(max cost {max_seed_planning_cost:,}).'
        )

        if planned_plan_count > 0:
            # bridge voxels can only exist on slices the render schedule touched;
            # merge (and delta-capture) only those instead of np.any-scanning every slice of the
            # full bridge volume (a mostly-empty ~volume-sized read per pass per view).
            scheduled_slices = np.flatnonzero(scheduled_slice_flags).astype(np.int64).tolist()

            delta_mm: Optional[np.ndarray] = None
            if bridge_delta_path is not None:
                # the delta is exactly bridge AND NOT pre-merge mask,
                # captured in the same read the merge already performs (no before-copy,
                # no post-pass subtract). 'w+' memmaps start zero-filled.
                Path(bridge_delta_path).parent.mkdir(parents=True, exist_ok=True)
                delta_mm = np.memmap(Path(bridge_delta_path), dtype=np.uint8, mode='w+', shape=mask_mm.shape)
                numa_interleave_memory(delta_mm, desc='Interpolation bridge delta')  #
                bridge_delta_slice_bboxes = np.zeros((int(mask_mm.shape[0]), 4), dtype=np.int64)
                bridge_delta_rows_by_slice = np.zeros(
                    (int(mask_mm.shape[0]), int(mask_mm.shape[1])), dtype=bool,
                )

            def _record_bridge_delta_metadata(
                z: int,
                delta_region: np.ndarray,
                base_y: int,
                base_x: int,
            ) -> None:
                if bridge_delta_slice_bboxes is None or bridge_delta_rows_by_slice is None:
                    return
                rows = np.any(delta_region, axis=1)
                if not bool(np.any(rows)):
                    return
                cols = np.any(delta_region, axis=0)
                row_ids = np.flatnonzero(rows)
                col_ids = np.flatnonzero(cols)
                bridge_delta_slice_bboxes[int(z)] = np.asarray((
                    int(base_y) + int(row_ids[0]),
                    int(base_y) + int(row_ids[-1]) + 1,
                    int(base_x) + int(col_ids[0]),
                    int(base_x) + int(col_ids[-1]) + 1,
                ), dtype=np.int64)
                bridge_delta_rows_by_slice[int(z), int(base_y):int(base_y) + int(rows.size)] = rows

            merged_added_by_slice = np.zeros((int(mask_mm.shape[0]),), dtype=np.int64)

            def _merge_slice(list_idx: int) -> None:
                z = int(scheduled_slices[int(list_idx)])
                if rendered_paste_bboxes is not None:
                    y0, x0, y1, x1 = (int(v) for v in rendered_paste_bboxes[z])
                    if y0 >= y1 or x0 >= x1:
                        return
                else:
                    y0, x0, y1, x1 = 0, 0, int(mask_mm.shape[1]), int(mask_mm.shape[2])

                if component_membership_mms:
                    bridge_region = np.asarray(
                        component_membership_mms[0][z, y0:y1, x0:x1] != 0,
                        dtype=bool,
                    )
                    for membership_mm in component_membership_mms[1:]:
                        bridge_region |= np.asarray(
                            membership_mm[z, y0:y1, x0:x1] != 0,
                            dtype=bool,
                        )
                elif bridge_mm is not None:
                    bridge_region = np.asarray(bridge_mm[z, y0:y1, x0:x1])
                else:  # pragma: no cover - one canvas is always configured
                    return
                if not np.any(bridge_region):
                    return

                mask_region = np.asarray(mask_mm[z, y0:y1, x0:x1])
                base_foreground = np.asarray(mask_region != 0, dtype=bool)
                delta_region = np.where(
                    ~base_foreground, bridge_region, np.uint8(0)
                ).astype(np.uint8, copy=False)
                merged_added_by_slice[int(z)] = np.int64(np.count_nonzero(delta_region))

                if component_membership_mms:
                    # Difference every packed membership bit against the same immutable
                    # pre-pass mask. Combinations may overlap; their OR remains the exact
                    # aggregate delta merged below and supplied to subsequent passes.
                    for membership_mm in component_membership_mms:
                        membership_region = np.asarray(
                            membership_mm[z, y0:y1, x0:x1]
                        )
                        membership_region[base_foreground] = np.asarray(
                            0, dtype=membership_region.dtype
                        )
                    for component_key, (word_index, bit_value) in component_bit_layout.items():
                        membership_region = np.asarray(
                            component_membership_mms[int(word_index)][z, y0:y1, x0:x1]
                        )
                        component_added_by_slice[component_key][int(z)] = np.int64(
                            np.count_nonzero(
                                membership_region
                                & np.asarray(int(bit_value), dtype=membership_region.dtype)
                            )
                        )

                if delta_mm is not None:
                    delta_mm[z, y0:y1, x0:x1] = delta_region
                    _record_bridge_delta_metadata(int(z), delta_region, int(y0), int(x0))
                mask_region |= np.asarray(bridge_region, dtype=np.uint8)

            parallel_for_indices(
                len(scheduled_slices),
                _merge_slice,
                max_workers=choose_slice_parallel_workers(int(render_workers), max(1, len(scheduled_slices))),
                desc='Interpolation: merge bridges',
            )
            added_voxels = int(np.sum(merged_added_by_slice, dtype=np.int64))
            flush_array(mask_mm)
            if delta_mm is not None:
                flush_array(delta_mm)
                close_memmap_array(delta_mm)
                bridge_delta_written = True
    finally:
        pass_failed = sys.exc_info()[0] is not None
        if gpu_renderer is not None:
            try:
                gpu_renderer_telemetry = gpu_renderer.telemetry()
            except Exception:
                gpu_renderer_telemetry = {}
            try:
                gpu_renderer.close()
            except Exception:
                pass
        if isinstance(bridge_mm, np.memmap):
            flush_array(bridge_mm)
        del bridge_mm
        for membership_mm in component_membership_mms:
            flush_array(membership_mm)
            close_memmap_array(membership_mm)
        component_membership_mms.clear()
        try:
            del component_cache
        except Exception:
            pass
        if isinstance(labels_mm, np.memmap):
            flush_array(labels_mm)
        del labels_mm
        if not keep_temp:
            for p in label_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            if bridge_path is not None:
                try:
                    bridge_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if bool(pass_failed):
                for membership_path in component_membership_paths:
                    try:
                        membership_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                for _walk_back_index, _candidate_index, component_path in component_specs:
                    shutil.rmtree(component_path, ignore_errors=True)

    component_added_counts = {
        key: int(np.sum(counts, dtype=np.int64))
        for key, counts in component_added_by_slice.items()
    }
    membership_readers: List[np.memmap] = []
    try:
        membership_readers = [
            np.memmap(
                membership_path,
                dtype=component_word_dtype,
                mode='r',
                shape=tuple(int(v) for v in np.asarray(mask_mm).shape),
            )
            for membership_path in component_membership_paths
        ]
        # Extract each logical bit-plane directly into a bbox-cropped cvol. Empty
        # combinations retain the already-published no-scan cvol from initialization.
        for walk_back_index, candidate_index, component_path in component_specs:
            component_key = (int(walk_back_index), int(candidate_index))
            component_voxel_count = int(component_added_counts.get(component_key, 0))
            if component_voxel_count <= 0:
                continue
            word_index, bit_value = component_bit_layout[component_key]
            membership_reader = membership_readers[int(word_index)]
            bit_scalar = np.asarray(int(bit_value), dtype=component_word_dtype)

            def _encode_component_slice(idx: int) -> RawBBoxSlicePayload:
                return _encode_bool_mask_slice_payload(
                    int(idx),
                    np.bitwise_and(membership_reader[int(idx)], bit_scalar) != 0,
                )

            _write_raw_bbox_payload_store(
                shape=tuple(int(v) for v in np.asarray(mask_mm).shape),
                store_dir=component_path,
                encode_slice=_encode_component_slice,
                format_name=CVOL_FORMAT,
                desc=(
                    f'Interpolation component {pass_tag} walkback '
                    f'{int(walk_back_index)} candidate {int(candidate_index)}'
                ),
                workers=int(workers),
                extra_meta={
                    'interpolation_walk_back_index': int(walk_back_index),
                    'interpolation_candidate_index': int(candidate_index),
                    'added_voxels': int(component_voxel_count),
                    'render_storage': (
                        f'packed_{component_word_dtype.name}_membership_bitplane'
                    ),
                },
                force_path_backed=True,
            )
    except BaseException:
        if not bool(keep_temp):
            for _walk_back_index, _candidate_index, component_path in component_specs:
                shutil.rmtree(component_path, ignore_errors=True)
        raise
    finally:
        for membership_reader in membership_readers:
            close_memmap_array(membership_reader)
        if not bool(keep_temp):
            for membership_path in component_membership_paths:
                try:
                    membership_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if component_membership_paths:
                try:
                    component_membership_paths[0].parent.rmdir()
                except OSError:
                    pass

    result_stats: Dict[str, object] = {
        'num_objects': int(num_objects),
        'num_endpoints': int(num_endpoints),
        'candidate_connections': int(candidate_connections),
        'accepted_connections': int(accepted_connections),
        'default_bridges': int(default_bridges),
        'walk_back_bridges': int(walk_back_bridges),
        'skipped_by_min_radius': int(skipped_by_min_radius),
        'added_voxels': int(added_voxels),
        'skipped': False,
        'wrap_axis': bool(wrap_axis),
        'endpoint_method': 'slice_component_scan',
        'planning_backend': interpolation_planning_backend_name(),
        'compact_relabel_skipped': bool(skip_relabel),  #
        'planner_plan_count': int(planned_plan_count),
        'planner_plan_batches': int(plan_batches_rendered),
        'planner_plan_batch_budget_bytes': int(plan_batch_budget_bytes),
        'planner_plan_batch_peak_payload_bytes': int(plan_batch_peak_payload_bytes),
        'planner_plan_batch_peak_charge_bytes': int(plan_batch_peak_charge_bytes),
        'planner_plan_batch_peak_plans': int(plan_batch_peak_plans),
        'planner_seed_schedule': (
            'cost_balanced_slice_window' if int(seed_schedule_window) > 0 else 'slice_major'
        ),
        'planner_seed_schedule_window': int(seed_schedule_window),
        'planner_seed_max_cost': int(max_seed_planning_cost),
        'bridge_component_count': int(component_count),
        'bridge_component_render_storage': (
            f'packed_{component_word_dtype.name}_membership_bitplanes'
            if int(component_count) > 0 else 'none'
        ),
        'bridge_component_render_word_count': int(component_word_count),
        'bridge_component_render_logical_bytes': int(
            int(component_word_count)
            * array_nbytes(tuple(int(v) for v in np.asarray(mask_mm).shape), component_word_dtype)
        ),
        'interpolation_render_backend': (
            'cuda_cupy_crop_bounded'
            + ('+cpu_fallback' if gpu_renderer_telemetry.get('failed_reason') else '')
            if (
                int(gpu_renderer_telemetry.get('estimated_plans', 0)) > 0
                or int(gpu_renderer_telemetry.get('rendered_slices', 0)) > 0
            )
            else 'cpu_numpy_numba'
        ),
        'gpu_interpolation_status': str(gpu_renderer_status),
        'gpu_interpolation_active': bool(
            int(gpu_renderer_telemetry.get('estimated_plans', 0)) > 0
            or int(gpu_renderer_telemetry.get('rendered_slices', 0)) > 0
        ),
        'gpu_interpolation_required': bool(gpu_required),
        'gpu_interpolation_batches': int(gpu_render_batches),
        'gpu_interpolation_fallback_batches': int(gpu_render_fallback_batches),
        'gpu_interpolation_estimated_plans': int(
            gpu_renderer_telemetry.get('estimated_plans', 0)
        ),
        'gpu_interpolation_estimated_sections': int(
            gpu_renderer_telemetry.get('estimated_sections', 0)
        ),
        'gpu_interpolation_rendered_slices': int(
            gpu_renderer_telemetry.get('rendered_slices', 0)
        ),
        'gpu_interpolation_rendered_sections': int(
            gpu_renderer_telemetry.get('rendered_sections', 0)
        ),
        'gpu_interpolation_host_to_device_bytes': int(
            gpu_renderer_telemetry.get('host_to_device_bytes', 0)
        ),
        'gpu_interpolation_device_to_host_bytes': int(
            gpu_renderer_telemetry.get('device_to_host_bytes', 0)
        ),
        'gpu_interpolation_cache_hits': int(
            gpu_renderer_telemetry.get('cache_hits', 0)
        ),
        'gpu_interpolation_cache_misses': int(
            gpu_renderer_telemetry.get('cache_misses', 0)
        ),
        'gpu_interpolation_cache_peak_bytes': int(
            gpu_renderer_telemetry.get('cache_peak_bytes', 0)
        ),
        'gpu_interpolation_fallback_reason': gpu_renderer_telemetry.get('failed_reason'),
    }
    result_stats.update(cache_telemetry)
    result_stats['bridge_component_deltas'] = _bridge_component_stats(
        component_added_counts
    )
    if bool(bridge_delta_written) and bridge_delta_path is not None:
        result_stats['bridge_delta_path'] = str(bridge_delta_path)
        if bridge_delta_slice_bboxes is not None:
            result_stats['bridge_delta_slice_bboxes'] = np.ascontiguousarray(bridge_delta_slice_bboxes)
        if bridge_delta_rows_by_slice is not None:
            result_stats['bridge_delta_row_occupancy'] = np.ascontiguousarray(
                np.any(bridge_delta_rows_by_slice, axis=0)
            )
    return result_stats

class _ByteAdmissionPool:
    """Weighted transient-memory admission with one emergency oversize lane."""

    def __init__(self, capacity_bytes: int, name: str) -> None:
        self.capacity = max(1, int(capacity_bytes))
        self.name = str(name)
        self.in_use = 0
        self.condition = threading.Condition()

    @contextlib.contextmanager
    def reserve(self, requested_bytes: int, desc: str) -> Iterator[None]:
        requested = max(1, int(requested_bytes))
        charged = min(int(requested), int(self.capacity))
        waited = False
        with self.condition:
            while int(self.in_use) > 0 and int(self.in_use) + int(charged) > int(self.capacity):
                if not waited:
                    print(
                        f'{self.name}: waiting to admit {desc} '
                        f'({requested / GIB:.1f} GiB requested; '
                        f'{self.in_use / GIB:.1f}/{self.capacity / GIB:.1f} GiB active).'
                    )
                    waited = True
                self.condition.wait()
            self.in_use += int(charged)
        try:
            yield
        finally:
            with self.condition:
                self.in_use = max(0, int(self.in_use) - int(charged))
                self.condition.notify_all()

@dataclass
class _DirectUnionBackingLease:
    """Single-owner lifecycle record for one dense direct-union backing.

    The array ownership itself moves into the postprocess closure; this record makes the
    scheduler phase transition explicit and catches double admission, early release, and
    inference work accidentally targeting a postprocess-owned backing.
    """

    key: Tuple[str, str]
    nbytes: int
    phase: str = 'inference'
    owner_count: int = 1

    def transition(self, expected: str, new_phase: str) -> None:
        if self.phase != str(expected) or int(self.owner_count) != 1:
            raise RuntimeError(
                f'direct-union lease {self.key} transition {expected}->{new_phase} '
                f'from phase={self.phase!r}, owners={self.owner_count}'
            )
        self.phase = str(new_phase)

    def release(self, expected: str) -> None:
        if self.phase != str(expected) or int(self.owner_count) != 1:
            raise RuntimeError(
                f'direct-union lease {self.key} release from phase={self.phase!r}, '
                f'owners={self.owner_count}; expected {expected!r}'
            )
        self.owner_count = 0
        self.phase = 'released'

@dataclass
class PreparedViewResult:
    model_name: str
    view_name: str
    aug_id: str
    angle_deg: float
    native_support_mm: Optional[np.ndarray]
    final_view_volume_mm: Optional[np.ndarray]
    interpolation_stats: List[Dict[str, object]]
    nrrd_layers: List[NrrdLayerRef] = field(default_factory=list)
    parent_mask_support_mm: Optional[object] = None
    parent_bridge_support_mm: Optional[object] = None

@dataclass
class TilePostprocessTask:
    model_name: str
    view_name: str
    aug_id: str
    angle_deg: float
    config_id: str
    tile_id: str
    # Fixed parent-processing-grid footprint (py0, py1, px0, px1). The tile mask
    # itself is crop-local and is never expanded into a parent-sized raw canvas.
    parent_crop: Tuple[int, int, int, int]
    tile_mask_mm: np.ndarray
    tile_confmap_mm: Optional[np.ndarray]
    tile_mask_path: Path
    tile_confmap_path: Optional[Path]
    precleaned_slice_cleanup: bool = False
    # Actual crop-local output shape. Tile masks never expand into a parent-sized canvas.
    processing_shape: Optional[Tuple[int, int, int]] = None
    # Full parent processing-plane geometry used only to scale native-space thresholds.
    threshold_plane_shape: Optional[Tuple[int, int]] = None

@dataclass
class TilePostprocessResult:
    model_name: str
    view_name: str
    aug_id: str
    angle_deg: float
    config_id: str
    tile_id: str
    parent_crop: Tuple[int, int, int, int]
    tile_mask_mm: Optional[np.ndarray] = None
    tile_mask_path: Optional[Path] = None
    tile_mask_store: Optional['RawBBoxMaskStore'] = None

@dataclass(frozen=True)
class DeferredTilePostprocessResult:
    model_name: str
    view_name: str
    aug_id: str
    angle_deg: float
    config_id: str
    tile_id: str
    parent_crop: Tuple[int, int, int, int]
    tile_mask_path: Path
    tile_shape: Tuple[int, int, int]
    storage_format: str = 'ctile-mask-v2-raw'

CTILE_FORMAT = 'ctile-mask-v2-raw'

CVOL_FORMAT = 'cvol-mask-v2-raw'

INTERNAL_PACKED_CVOL_FORMAT = 'cvol-mask-v3-packbits-internal'

MASK_STORE_FORMATS = {CTILE_FORMAT, CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT}

def _merge_raw_bbox_extra_meta(
    meta: Dict[str, object], extra_meta: Optional[Dict[str, object]],
) -> None:
    """Merge annotations without allowing callers to rewrite the wire schema."""
    if not extra_meta:
        return
    annotations = dict(extra_meta)
    conflicts = sorted(set(annotations).intersection(meta))
    if conflicts:
        raise ValueError(
            'Raw bbox extra metadata cannot override reserved field(s): '
            + ', '.join(str(key) for key in conflicts)
        )
    meta.update(annotations)

_NRRD_RAW_STORE_CHUNKS_RAM_CACHE: Dict[Path, Tuple[object, int]] = {}

_NRRD_RAW_STORE_CHUNKS_RAM_CACHE_LOCK = threading.Lock()

_NRRD_RAW_STORE_CHUNKS_LOAD_EVENTS: Dict[Path, threading.Event] = {}

def _raw_store_chunks_cache_key(chunks_path: Path) -> Path:
    try:
        return Path(chunks_path).resolve()
    except Exception:
        return Path(chunks_path)

def _close_raw_store_chunks_mapping(payload: object) -> None:
    if not isinstance(payload, mmap.mmap):
        return
    try:
        payload.close()
    except BufferError:
        # A compressor may be dropping its final exported memoryview concurrently. Once
        # that view goes away the mmap object's own finalizer releases the mapping.
        pass

def _acquire_raw_store_chunks_ram_cache(chunks_path: Path) -> Tuple[object, bool]:
    """Return a shared read-only payload buffer and hold one reference on it."""
    key = _raw_store_chunks_cache_key(chunks_path)
    while True:
        with _NRRD_RAW_STORE_CHUNKS_RAM_CACHE_LOCK:
            entry = _NRRD_RAW_STORE_CHUNKS_RAM_CACHE.get(key)
            if entry is not None:
                payload, refs = entry
                _NRRD_RAW_STORE_CHUNKS_RAM_CACHE[key] = (payload, int(refs) + 1)
                return payload, True
            loading = _NRRD_RAW_STORE_CHUNKS_LOAD_EVENTS.get(key)
            if loading is None:
                _NRRD_RAW_STORE_CHUNKS_LOAD_EVENTS[key] = threading.Event()
                break
        # Another thread is mapping this chunks.bin; wait for it so every reader shares
        # exactly one mmap object. If the loader fails, the loop retries here.
        loading.wait()
    try:
        size = int(Path(chunks_path).stat().st_size)
        if size > 0:
            with Path(chunks_path).open('rb') as fh:
                payload: object = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            # mmap cannot map an empty file; empty stores never request a payload view.
            payload = b''
        with _NRRD_RAW_STORE_CHUNKS_RAM_CACHE_LOCK:
            _NRRD_RAW_STORE_CHUNKS_RAM_CACHE[key] = (payload, 1)
        return payload, False
    finally:
        with _NRRD_RAW_STORE_CHUNKS_RAM_CACHE_LOCK:
            load_event = _NRRD_RAW_STORE_CHUNKS_LOAD_EVENTS.pop(key, None)
        if load_event is not None:
            load_event.set()

def _release_raw_store_chunks_ram_cache(chunks_path: Path) -> None:
    key = _raw_store_chunks_cache_key(chunks_path)
    with _NRRD_RAW_STORE_CHUNKS_RAM_CACHE_LOCK:
        entry = _NRRD_RAW_STORE_CHUNKS_RAM_CACHE.get(key)
        if entry is None:
            return
        payload, refs = entry
        if int(refs) <= 1:
            _NRRD_RAW_STORE_CHUNKS_RAM_CACHE.pop(key, None)
            close_payload = payload
        else:
            _NRRD_RAW_STORE_CHUNKS_RAM_CACHE[key] = (payload, int(refs) - 1)
            close_payload = None
    if close_payload is not None:
        _close_raw_store_chunks_mapping(close_payload)

def _invalidate_raw_store_chunks_ram_cache(chunks_path: Path) -> None:
    """Drop a cache entry whose backing file is about to be rewritten (any refcount)."""
    key = _raw_store_chunks_cache_key(chunks_path)
    with _NRRD_RAW_STORE_CHUNKS_RAM_CACHE_LOCK:
        entry = _NRRD_RAW_STORE_CHUNKS_RAM_CACHE.pop(key, None)
    if entry is not None:
        _close_raw_store_chunks_mapping(entry[0])

CTILE_INDEX_DTYPE = np.dtype([
    ('kind', 'u1'),              # 0 = empty/zero slice, 1 = bbox payload (meta names precodec)
    ('reserved', 'u1', (7,)),
    ('offset', '<u8'),
    ('payload_size', '<u8'),
    ('y0', '<u4'),
    ('x0', '<u4'),
    ('y1', '<u4'),
    ('x1', '<u4'),
    ('payload_nbytes', '<u8'),
])

NrrdSegmentExtent = Tuple[int, int, int, int, int, int]

@dataclass(frozen=True)
class NrrdLayerRef:
    """Reference one component-layer backing store and its geometry metadata.
    
    The sink restores the store to output geometry and writes one independent single-layer NRRD."""

    key: str
    name: str
    path: Path
    shape: Tuple[int, int, int]
    dtype: str = 'uint8'
    storage_format: str = 'raw_u8'
    model_name: str = ''
    view_name: str = ''
    physical_view_name: str = ''
    aug_id: str = ''
    angle_deg: float = 0.0
    view_family: str = ''
    source: str = ''  # fullframe, tile, or global
    mask_kind: str = ''  # yolo, bridge, union, smoothing_result
    pass_index: int = 0
    interpolation_walk_back_index: int = 0
    interpolation_candidate_index: int = 0
    tile_config_id: str = ''
    tile_acceptance: str = ''  # parent_mask, parent_bridge, consolidated, or blank
    stage: str = ''
    description: str = ''
    # make recomposition semantics explicit. Legacy view contributors are
    # additive by default; global checkpoints and centerline audit deltas override
    # these values at construction time.
    layer_role: str = 'additive_component'
    recomposition_op: str = 'union'
    low_quality_recomposition_op: str = 'union'
    mirror_low_quality: bool = True
    # Slicer SegmentN_Extent in this layer backing store's own (X,Y,t) index space.
    # Final NRRD packaging maps this extent into the requested output geometry without
    # reopening the layer solely to compute header metadata.
    segment_extent_ijk: Optional[NrrdSegmentExtent] = None
    segment_extent_shape_tyx: Tuple[int, int, int] = (0, 0, 0)
    segment_extent_source: str = ''
    # an IMMUTABLE in-process live volume the sink streams directly (no store
    # encode, no read-back). Only valid within the producing process; ``path`` is then a
    # never-created placeholder. compare=False keeps the frozen dataclass hashless-safe.
    live_array: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

@dataclass(frozen=True)
class NrrdRasterPlan:
    """Full-reference raster plan for one single-layer NRRD."""

    stored_shape_tyx: Tuple[int, int, int]
    segment_extent_xyt: NrrdSegmentExtent
    empty_segment: bool

@dataclass(frozen=True)
class TileParentGateResult:
    model_name: str
    view_name: str
    aug_id: str
    angle_deg: float
    config_id: str
    tile_id: str
    gate_stats: Dict[str, int]
    # Entire original components that failed parent-YOLO support. They remain tile-local
    # and angle-local until the immutable parent bridge support is published.
    residual_result: Optional[TilePostprocessResult] = field(default=None, compare=False)

@dataclass(frozen=True)
class TileGateResult:
    model_name: str
    view_name: str
    aug_id: str
    angle_deg: float
    config_id: str
    tile_id: str
    gate_stats: Dict[str, int]

@dataclass(frozen=True)
class TileConsolidationResult:
    model_name: str
    view_name: str
    aug_id: str
    angle_deg: float
    interpolation_stats: List[Dict[str, object]]
    nrrd_layers: List[NrrdLayerRef] = field(default_factory=list)
    # the interpolation process backend may rebind the consolidated
    # accumulator to a fresh disk memmap; the scheduler must re-point its registry at the
    # volume that actually received the interpolation bridges.
    final_accumulator_mm: Optional[np.ndarray] = None

def _view_uses_interpolation(view: ViewInfo, interpolate: int) -> bool:
    return bool((view.family in ('orthogonal', 'radial') or is_tilted_view(view)) and int(interpolate) > 0)

def _drain_volume_to_mmap(
    volume: np.ndarray,
    path: Path,
    desc: str,
    *,
    workers: int = 1,
) -> np.ndarray:
    drained = copy_workspace_array(
        np.asarray(volume),
        path,
        desc=desc,
        prefer_memory=False,
        workers=int(workers),
    )
    flush_array(drained)
    return drained

def _nrrd_empty_segment_extent() -> NrrdSegmentExtent:
    return (0, -1, 0, -1, 0, -1)

def _coerce_segment_extent(value: object) -> Optional[NrrdSegmentExtent]:
    if value is None:
        return None
    try:
        vals = [int(v) for v in value]  # type: ignore[arg-type]
    except Exception:
        return None
    if len(vals) != 6:
        return None
    return (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]))

def _segment_extent_to_json(extent: Sequence[int]) -> List[int]:
    coerced = _coerce_segment_extent(extent)
    if coerced is None:
        coerced = _nrrd_empty_segment_extent()
    return [int(v) for v in coerced]

@dataclass(frozen=True)
class RawBBoxSlicePayload:
    """One slice payload for a bbox mask store.

 External stores carry raw uint8 crop bytes. Private terminal retention may carry
 row-wise packbits while ``payload_nbytes`` records the decoded logical size."""

    idx: int
    is_empty: bool
    y0: int = 0
    x0: int = 0
    y1: int = 0
    x1: int = 0
    payload_nbytes: int = 0
    payload: bytes = b''
    foreground_voxels: int = 0

class IncrementalRawBBoxMaskStoreWriter:
    """Thread-safe raw-bbox writer fed by completed projection blocks.

 Blocks may arrive in any order. Each nonempty slice reserves an append offset, then
 writes its normalized crop in bounded row slabs with ``pwrite``; neither block-sized nor
 whole-crop payload objects are retained. Finalization is allowed only after every output
 slice has been delivered exactly once, so its index, foreground count, and segment extent
 have the same semantics as:func:`write_raw_bbox_mask_store`."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int, int],
        store_dir: Path,
        format_name: str,
        desc: str,
        extra_meta: Optional[Dict[str, object]] = None,
        force_path_backed: bool = False,
    ) -> None:
        fmt = str(format_name)
        if fmt not in MASK_STORE_FORMATS:
            raise ValueError(f'Unsupported raw bbox mask format: {fmt}')
        shape_i = tuple(int(v) for v in shape)
        if len(shape_i) != 3 or any(int(v) < 0 for v in shape_i):
            raise ValueError(f'{desc}: invalid incremental raw store shape {shape_i}')

        self.shape = (int(shape_i[0]), int(shape_i[1]), int(shape_i[2]))
        self.store_dir = Path(store_dir)
        self.format_name = fmt
        self._packbits_payload = bool(fmt == INTERNAL_PACKED_CVOL_FORMAT)
        self.desc = str(desc)
        self.extra_meta = dict(extra_meta or {})
        self.force_path_backed = bool(force_path_backed)
        self.chunks_path = self.store_dir / 'chunks.bin'
        self.index_path = self.store_dir / 'index.bin'
        self.meta_path = self.store_dir / 'meta.json'
        self.index = np.zeros((int(self.shape[0]),), dtype=CTILE_INDEX_DTYPE)
        # 0=not received, 1=payload write in progress, 2=complete.
        self._slice_state = np.zeros((int(self.shape[0]),), dtype=np.uint8)
        self._lock = threading.RLock()
        self._active_callbacks = 0
        self._next_offset = 0
        self._nonempty_slices = 0
        self._foreground_voxels = 0
        self._min_t, self._max_t = int(self.shape[0]), -1
        self._min_y, self._max_y = int(self.shape[1]), -1
        self._min_x, self._max_x = int(self.shape[2]), -1
        self._failed_reason: Optional[BaseException] = None
        self._warned_failed = False
        self._finalized = False

        _invalidate_raw_store_chunks_ram_cache(self.chunks_path)
        release_memfd_owners_under(self.store_dir)
        if self.store_dir.exists():
            shutil.rmtree(self.store_dir, ignore_errors=True)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._fd: Optional[int]
        if raw_store_memfd_enabled() and not self.force_path_backed:
            try:
                self._fd = _create_memfd_backed_payload_path(
                    self.chunks_path, f'{self.desc} chunks',
                )
                runtime_telemetry().add('cvol.memfd_stores', 1)
            except Exception as exc:
                runtime_telemetry().fallback('cvol.memfd.incremental', exc)
                self._fd = os.open(
                    self.chunks_path,
                    os.O_CREAT | os.O_TRUNC | os.O_RDWR,
                    0o666,
                )
        else:
            self._fd = os.open(
                self.chunks_path,
                os.O_CREAT | os.O_TRUNC | os.O_RDWR,
                0o666,
            )

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._failed_reason is not None

    def warn_failed_once(self, message: str) -> None:
        with self._lock:
            if self._warned_failed:
                return
            self._warned_failed = True
        print(f'Warning: {message}; the completed projected volume will use the regular encoder.')

    def abort(self, reason: BaseException) -> None:
        with self._lock:
            if self._failed_reason is None:
                self._failed_reason = reason

    def __call__(self, z0: int, block: np.ndarray) -> None:
        self.consume(int(z0), block)

    def consume_empty_range(self, z0: int, count: int) -> None:
        """Register a proven-empty contiguous slice range without scanning dense planes."""
        start = int(z0)
        stop = int(start) + max(0, int(count))
        if start < 0 or stop > int(self.shape[0]):
            raise IndexError(f'{self.desc}: empty range [{start}, {stop}) is outside {self.shape[0]} slices')
        with self._lock:
            if self._failed_reason is not None:
                return
            for z_i in range(int(start), int(stop)):
                if int(self._slice_state[z_i]) != 0:
                    raise ValueError(f'{self.desc}: projected slice {z_i} was delivered more than once')
            for z_i in range(int(start), int(stop)):
                self.index[z_i]['kind'] = np.uint8(0)
                self._slice_state[z_i] = np.uint8(2)

    @staticmethod
    def _pwrite_all(fd: int, data: memoryview, offset: int) -> None:
        written = 0
        while int(written) < int(len(data)):
            n = int(os.pwrite(int(fd), data[int(written):], int(offset) + int(written)))
            if n <= 0:
                raise OSError(f'pwrite returned {n} before completing {len(data)} bytes')
            written += int(n)

    def _consume_slice(self, z: int, plane: np.ndarray) -> None:
        z_i = int(z)
        plane_arr = np.ascontiguousarray(np.asarray(plane, dtype=np.uint8))
        # OpenCV performs one compiled scan for both emptiness and the tight bbox.
        # The previous two NumPy reductions read the entire plane twice and created two
        # temporary boolean vectors before the sink normalized/copied the crop.
        x0, y0, bbox_w, bbox_h = (int(v) for v in cv2.boundingRect(plane_arr))
        if int(bbox_w) <= 0 or int(bbox_h) <= 0:
            with self._lock:
                if self._failed_reason is not None:
                    return
                if int(self._slice_state[z_i]) != 0:
                    raise ValueError(f'{self.desc}: projected slice {z_i} was delivered more than once')
                self.index[z_i]['kind'] = np.uint8(0)
                self._slice_state[z_i] = np.uint8(2)
            return

        x1 = int(x0) + int(bbox_w)
        y1 = int(y0) + int(bbox_h)
        payload_nbytes = int(y1 - y0) * int(x1 - x0)
        encoded_row_width = (
            int((int(x1 - x0) + 7) // 8)
            if bool(self._packbits_payload)
            else int(x1 - x0)
        )
        payload_size = int(y1 - y0) * int(encoded_row_width)

        with self._lock:
            if self._failed_reason is not None:
                return
            if int(self._slice_state[z_i]) != 0:
                raise ValueError(f'{self.desc}: projected slice {z_i} was delivered more than once')
            if self._fd is None:
                raise RuntimeError(f'{self.desc}: chunks payload is already closed')
            fd = int(self._fd)
            offset = int(self._next_offset)
            self._next_offset += int(payload_size)
            self._slice_state[z_i] = np.uint8(1)

        foreground_voxels = 0
        row_width = int(x1 - x0)
        rows_per_slab = max(1, int((1 * 1024 * 1024) // max(1, int(row_width))))
        for slab_y0 in range(int(y0), int(y1), int(rows_per_slab)):
            slab_y1 = min(int(y1), int(slab_y0) + int(rows_per_slab))
            # A <=1 MiB normalized slab keeps syscall count bounded without retaining the
            # whole bbox crop. pwrite lets out-of-order callback threads fill reservations.
            slab_u8 = np.ascontiguousarray(
                plane_arr[int(slab_y0):int(slab_y1), int(x0):int(x1)] != 0,
                dtype=np.uint8,
            )
            foreground_voxels += int(np.count_nonzero(slab_u8))
            encoded_slab = (
                np.packbits(slab_u8, axis=1, bitorder='little')
                if bool(self._packbits_payload)
                else slab_u8
            )
            self._pwrite_all(
                fd,
                memoryview(np.ascontiguousarray(encoded_slab)).cast('B'),
                int(offset) + int(slab_y0 - y0) * int(encoded_row_width),
            )

        with self._lock:
            if self._failed_reason is not None:
                return
            rec = self.index[z_i]
            rec['kind'] = np.uint8(1)
            rec['offset'] = np.uint64(offset)
            rec['payload_size'] = np.uint64(payload_size)
            rec['y0'] = np.uint32(y0)
            rec['x0'] = np.uint32(x0)
            rec['y1'] = np.uint32(y1)
            rec['x1'] = np.uint32(x1)
            rec['payload_nbytes'] = np.uint64(payload_nbytes)
            self._slice_state[z_i] = np.uint8(2)
            self._nonempty_slices += 1
            self._foreground_voxels += int(foreground_voxels)
            self._min_t = min(int(self._min_t), z_i)
            self._max_t = max(int(self._max_t), z_i)
            self._min_y = min(int(self._min_y), y0)
            self._max_y = max(int(self._max_y), int(y1) - 1)
            self._min_x = min(int(self._min_x), x0)
            self._max_x = max(int(self._max_x), int(x1) - 1)

    def consume_sparse_slice(
        self,
        z: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        crop: np.ndarray,
    ) -> None:
        """Append one already-bounded slice crop without constructing a dense plane."""
        z_i = int(z)
        y0_i, y1_i, x0_i, x1_i = int(y0), int(y1), int(x0), int(x1)
        if not (0 <= z_i < int(self.shape[0])):
            raise IndexError(f'{self.desc}: sparse slice {z_i} is outside {self.shape[0]} slices')
        if not (
            0 <= y0_i <= y1_i <= int(self.shape[1])
            and 0 <= x0_i <= x1_i <= int(self.shape[2])
        ):
            raise ValueError(
                f'{self.desc}: sparse bbox {(y0_i, y1_i, x0_i, x1_i)} is outside {self.shape}'
            )
        crop_arr = np.ascontiguousarray(np.asarray(crop, dtype=np.uint8))
        if tuple(int(v) for v in crop_arr.shape) != (int(y1_i - y0_i), int(x1_i - x0_i)):
            raise ValueError(
                f'{self.desc}: sparse crop shape {tuple(crop_arr.shape)} does not match '
                f'bbox {(y0_i, y1_i, x0_i, x1_i)}'
            )

        with self._lock:
            if self._failed_reason is not None:
                return
            if self._finalized:
                raise RuntimeError(f'{self.desc}: sparse slice arrived after finalization')
            self._active_callbacks += 1
        try:
            tight_x, tight_y, tight_w, tight_h = (int(v) for v in cv2.boundingRect(crop_arr))
            if int(tight_w) <= 0 or int(tight_h) <= 0:
                self.consume_empty_range(int(z_i), 1)
                return
            x0_t = int(x0_i + tight_x)
            y0_t = int(y0_i + tight_y)
            x1_t = int(x0_t + tight_w)
            y1_t = int(y0_t + tight_h)
            tight_crop = crop_arr[
                int(tight_y):int(tight_y + tight_h),
                int(tight_x):int(tight_x + tight_w),
            ]
            payload_nbytes = int(tight_h) * int(tight_w)
            encoded_row_width = (
                int((int(tight_w) + 7) // 8)
                if bool(self._packbits_payload) else int(tight_w)
            )
            payload_size = int(tight_h) * int(encoded_row_width)

            with self._lock:
                if self._failed_reason is not None:
                    return
                if int(self._slice_state[z_i]) != 0:
                    raise ValueError(f'{self.desc}: projected slice {z_i} was delivered more than once')
                if self._fd is None:
                    raise RuntimeError(f'{self.desc}: chunks payload is already closed')
                fd = int(self._fd)
                offset = int(self._next_offset)
                self._next_offset += int(payload_size)
                self._slice_state[z_i] = np.uint8(1)

            foreground_voxels = 0
            rows_per_slab = max(1, int((1 * 1024 * 1024) // max(1, int(tight_w))))
            for local_y0 in range(0, int(tight_h), int(rows_per_slab)):
                local_y1 = min(int(tight_h), int(local_y0) + int(rows_per_slab))
                slab_u8 = np.ascontiguousarray(
                    tight_crop[int(local_y0):int(local_y1), :] != 0,
                    dtype=np.uint8,
                )
                foreground_voxels += int(np.count_nonzero(slab_u8))
                encoded_slab = (
                    np.packbits(slab_u8, axis=1, bitorder='little')
                    if bool(self._packbits_payload) else slab_u8
                )
                self._pwrite_all(
                    fd,
                    memoryview(np.ascontiguousarray(encoded_slab)).cast('B'),
                    int(offset) + int(local_y0) * int(encoded_row_width),
                )

            with self._lock:
                if self._failed_reason is not None:
                    return
                rec = self.index[z_i]
                rec['kind'] = np.uint8(1)
                rec['offset'] = np.uint64(offset)
                rec['payload_size'] = np.uint64(payload_size)
                rec['y0'] = np.uint32(y0_t)
                rec['x0'] = np.uint32(x0_t)
                rec['y1'] = np.uint32(y1_t)
                rec['x1'] = np.uint32(x1_t)
                rec['payload_nbytes'] = np.uint64(payload_nbytes)
                self._slice_state[z_i] = np.uint8(2)
                self._nonempty_slices += 1
                self._foreground_voxels += int(foreground_voxels)
                self._min_t = min(int(self._min_t), z_i)
                self._max_t = max(int(self._max_t), z_i)
                self._min_y = min(int(self._min_y), y0_t)
                self._max_y = max(int(self._max_y), int(y1_t) - 1)
                self._min_x = min(int(self._min_x), x0_t)
                self._max_x = max(int(self._max_x), int(x1_t) - 1)
        except Exception as exc:
            self.abort(exc)
            raise
        finally:
            with self._lock:
                self._active_callbacks -= 1

    def consume(self, z0: int, block: np.ndarray) -> None:
        block_arr = np.asarray(block)
        if block_arr.ndim != 3:
            raise ValueError(f'{self.desc}: projection callback expected a 3D block, got {block_arr.shape}')
        if tuple(int(v) for v in block_arr.shape[1:]) != tuple(int(v) for v in self.shape[1:]):
            raise ValueError(
                f'{self.desc}: projection callback plane shape {tuple(block_arr.shape[1:])} '
                f'!= {tuple(self.shape[1:])}'
            )
        start = int(z0)
        stop = int(start) + int(block_arr.shape[0])
        if start < 0 or stop > int(self.shape[0]):
            raise ValueError(f'{self.desc}: projection callback block [{start},{stop}) is out of range')

        with self._lock:
            if self._failed_reason is not None:
                return
            if self._finalized:
                raise RuntimeError(f'{self.desc}: projection callback arrived after finalization')
            self._active_callbacks += 1
        try:
            for local_z in range(int(block_arr.shape[0])):
                if self.failed:
                    return
                self._consume_slice(int(start) + int(local_z), block_arr[int(local_z)])
        except Exception as exc:
            self.abort(exc)
            raise
        finally:
            with self._lock:
                self._active_callbacks -= 1

    def finalize(self) -> Dict[str, object]:
        with self._lock:
            if self._failed_reason is not None:
                raise RuntimeError(
                    f'{self.desc}: incremental store was invalidated: {self._failed_reason}'
                ) from self._failed_reason
            if self._active_callbacks != 0:
                raise RuntimeError(f'{self.desc}: finalized with {self._active_callbacks} callback(s) active')
            missing = np.flatnonzero(self._slice_state != np.uint8(2))
            if int(missing.size) > 0:
                raise RuntimeError(
                    f'{self.desc}: incremental store is missing {int(missing.size)} slice(s); '
                    f'first missing={int(missing[0])}'
                )
            if self._finalized:
                raise RuntimeError(f'{self.desc}: incremental store was finalized more than once')
            fd = self._fd
            self._fd = None
            self._finalized = True
            payload_bytes = int(self._next_offset)
            nonempty_slices = int(self._nonempty_slices)
            foreground_voxels = int(self._foreground_voxels)
            extent = (
                _nrrd_empty_segment_extent()
                if int(self._max_t) < 0
                else (
                    int(self._min_x), int(self._max_x),
                    int(self._min_y), int(self._max_y),
                    int(self._min_t), int(self._max_t),
                )
            )
        if fd is not None:
            # Close publishes every pwrite through the host page cache; an fsync here
            # would force a multi-GiB durability write that the transient cvol/NRRD
            # pipeline neither requires nor used in the established encoder.
            os.close(int(fd))

        self.index.tofile(self.index_path)
        raw_logical_bytes = int(array_nbytes(self.shape, np.uint8))
        stats: Dict[str, object] = {
            'nonempty_slices': int(nonempty_slices),
            'empty_slices': int(self.shape[0] - nonempty_slices),
            'foreground_voxels': int(foreground_voxels),
            'logical_raw_uint8_bytes': int(raw_logical_bytes),
            'raw_payload_bytes': int(payload_bytes),
            'index_bytes': int(self.index.nbytes),
            'segment_extent_ijk': _segment_extent_to_json(extent),
            'segment_extent_shape_tyx': [int(v) for v in self.shape],
        }
        meta: Dict[str, object] = {
            'format': self.format_name,
            'shape': [int(v) for v in self.shape],
            'dtype': 'bool',
            'logical_dtype_in_pipeline': 'uint8_0_or_1',
            'chunking': 'slice',
            'precodec': (
                'numpy_packbits_axis_x_little'
                if bool(self._packbits_payload)
                else 'none'
            ),
            'compressor': 'none',
            'bbox_per_chunk': True,
            'zero_chunk_elision': True,
            'index_dtype': 'ctile-index-v2-raw',
            'index_record_bytes': int(CTILE_INDEX_DTYPE.itemsize),
            'payload_shape_encoding': (
                'packbits_rows_ceil_width_div_8_bbox_shape_from_index'
                if bool(self._packbits_payload)
                else 'raw_uint8_bbox_shape_from_index'
            ),
            'description': self.desc,
            'segment_extent_ijk': _segment_extent_to_json(extent),
            'segment_extent_axis_order': (
                'Slicer IJK inclusive extent: minX maxX minY maxY minT maxT '
                'for internal layer order (t,Y,X)'
            ),
            'segment_extent_shape_tyx': [int(v) for v in self.shape],
            'stats': stats,
        }
        _merge_raw_bbox_extra_meta(meta, self.extra_meta)
        self.meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        print(
            f'{self.desc}: incremental raw bbox mask store {self.store_dir} '
            f'(logical_raw={raw_logical_bytes / GIB:.2f} GiB, '
            f'payload={payload_bytes / GIB:.2f} GiB, nonempty_slices={nonempty_slices})'
        )
        return stats

    def discard(self) -> None:
        with self._lock:
            fd = self._fd
            self._fd = None
            if self._failed_reason is None and not self._finalized:
                self._failed_reason = RuntimeError('incremental store discarded')
        if fd is not None:
            try:
                os.close(int(fd))
            except OSError:
                pass
        _invalidate_raw_store_chunks_ram_cache(self.chunks_path)
        release_memfd_owners_under(self.store_dir)
        shutil.rmtree(self.store_dir, ignore_errors=True)

class RawBBoxMaskStore:
    """Read adapter for slice-bbox binary mask volumes.

 Empty slices are elided and nonempty slices are cropped to their nonzero bbox. External
 waiting-tile, NRRD, and support stores retain raw uint8 payloads; private final-view retention
 may use the internal row-wise packbits precodec. Neither form uses LZ4."""

    def __init__(
        self,
        root: Path,
        meta: Dict[str, object],
        index: np.ndarray,
        *,
        chunks_bytes: Optional[object] = None,
        mmap_payload: bool = False,
    ) -> None:
        self.root = Path(root)
        self.meta = dict(meta)
        self.index = np.asarray(index, dtype=CTILE_INDEX_DTYPE)
        format_name = str(self.meta.get('format', ''))
        if format_name not in MASK_STORE_FORMATS:
            raise ValueError(f'{self.root}: unsupported raw mask format {format_name!r}')
        self._packbits_payload = bool(format_name == INTERNAL_PACKED_CVOL_FORMAT)
        precodec = str(self.meta.get('precodec', 'none'))
        expected_precodec = (
            'numpy_packbits_axis_x_little'
            if bool(self._packbits_payload)
            else 'none'
        )
        if precodec != expected_precodec:
            raise ValueError(
                f'{self.root}: format {format_name!r} requires precodec '
                f'{expected_precodec!r}, got {precodec!r}'
            )
        self.chunks_path = self.root / 'chunks.bin'
        # ``chunks_bytes`` is retained as a compatibility keyword, but passes
        # the process-shared read-only mmap instead of a heap-owned bytes copy.
        self._chunks_bytes: Optional[object] = chunks_bytes
        self._chunks_mmap: Optional[mmap.mmap] = None
        # True when open holds a reference on the shared RAM cache entry for this
        # store's chunks.bin (released via _drop_nrrd_raw_store_chunks_ram_cache).
        self._ram_cache_ref_held = False
        shape = self.meta.get('shape')
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f'{self.root}: invalid raw-mask shape metadata: {shape!r}')
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        if int(self.index.shape[0]) != int(self.shape[0]):
            raise ValueError(f'{self.root}: index slice count {int(self.index.shape[0])} != shape[0] {int(self.shape[0])}')
        if bool(mmap_payload) and self._chunks_bytes is None and self.chunks_path.stat().st_size > 0:
            # opens many stores at once. A read-only mapping keeps pages reclaimable by
            # the kernel while avoiding one open/seek/read/close cycle per (layer,z).
            try:
                with self.chunks_path.open('rb') as fh:
                    self._chunks_mmap = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError):
                self._chunks_mmap = None

    @classmethod
    def open(
        cls, root: Path, *, cache_payload_in_ram: bool = False, mmap_payload: bool = False,
    ) -> 'RawBBoxMaskStore':
        root = Path(root)
        meta_path = root / 'meta.json'
        index_path = root / 'index.bin'
        chunks_path = root / 'chunks.bin'
        if not meta_path.exists() or not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f'Incomplete raw mask store: {root}')
        meta = json.loads(meta_path.read_text())
        fmt = str(meta.get('format', ''))
        if fmt not in MASK_STORE_FORMATS:
            raise ValueError(f'{root}: unsupported raw mask format {meta.get("format")!r}')
        shape = meta.get('shape')
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f'{root}: invalid shape metadata {shape!r}')
        index = np.fromfile(index_path, dtype=CTILE_INDEX_DTYPE, count=int(shape[0]))
        chunks_bytes: Optional[object] = None
        ram_cache_ref_held = False
        if bool(cache_payload_in_ram):
            chunks_bytes, was_cached = _acquire_raw_store_chunks_ram_cache(chunks_path)
            ram_cache_ref_held = True
        else:
            was_cached = False
        try:
            if ram_cache_ref_held:
                verb = 'reused shared mmap' if was_cached else 'mapped read-only'
                print(
                    f'Raw bbox mask store {verb} for NRRD streaming: {root} '
                    f'({len(chunks_bytes) / GIB:.3f} GiB chunks.bin)'
                )
            store = cls(
                root, meta, index, chunks_bytes=chunks_bytes,
                mmap_payload=bool(mmap_payload),
            )
        except BaseException:
            # Anything failing after the acquire (a print on a dead stdout pipe, or
            # constructor validation on a truncated index.bin) leaves no store object to
            # carry the reference, so release it here or the multi-GiB payload stays
            # pinned in the cache for the life of the process.
            if ram_cache_ref_held:
                _release_raw_store_chunks_ram_cache(chunks_path)
            raise
        store._ram_cache_ref_held = bool(ram_cache_ref_held)
        return store


    def close(self) -> None:
        self._chunks_bytes = None
        chunks_mmap = self._chunks_mmap
        if chunks_mmap is not None:
            try:
                advice = getattr(mmap, 'MADV_DONTNEED', None)
                if advice is not None:
                    chunks_mmap.madvise(advice)
            except (AttributeError, OSError, ValueError, BufferError):
                pass
        self._chunks_mmap = None
        if chunks_mmap is not None:
            try:
                chunks_mmap.close()
            except BufferError:
                # A just-finished worker may still be dropping a NumPy view. Removing our
                # reference lets the mapping close when that final exported view disappears.
                pass

    def decode_slice_crop(
        self, idx: int, *, dtype: np.dtype | str | type = np.uint8,
    ) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
        """Return only a nonempty slice's decoded bbox crop, or None for an empty slice."""
        idx_i = int(idx)
        z_dim, h, w = self.shape
        if idx_i < 0 or idx_i >= int(z_dim):
            raise IndexError(idx_i)
        rec = self.index[idx_i]
        if int(rec['kind']) == 0:
            return None
        if int(rec['kind']) != 1:
            raise ValueError(f'{self.root}: invalid raw-mask chunk marker {int(rec["kind"])} at slice {idx_i}')

        y0 = int(rec['y0']); x0 = int(rec['x0']); y1 = int(rec['y1']); x1 = int(rec['x1'])
        if not (0 <= y0 < y1 <= int(h) and 0 <= x0 < x1 <= int(w)):
            raise ValueError(f'{self.root}: invalid bbox {(y0, x0, y1, x1)} for shape {(h, w)} at slice {idx_i}')
        payload_size = int(rec['payload_size'])
        payload_nbytes = int(rec['payload_nbytes'])
        if payload_size <= 0 or payload_nbytes <= 0:
            raise ValueError(f'{self.root}: nonempty slice {idx_i} has empty payload metadata')

        start = int(rec['offset'])
        stop = start + int(payload_size)
        if self._chunks_bytes is not None:
            payload: object = memoryview(self._chunks_bytes)[start:stop]
        elif self._chunks_mmap is not None:
            payload = memoryview(self._chunks_mmap)[start:stop]
        else:
            with self.chunks_path.open('rb') as fh:
                fh.seek(start)
                payload = fh.read(int(payload_size))
        if len(payload) != payload_size:  # type: ignore[arg-type]
            raise IOError(f'{self.root}: short read for slice {idx_i}: {len(payload)} != {payload_size}')  # type: ignore[arg-type]
        crop_h = int(y1 - y0)
        crop_w = int(x1 - x0)
        expected = int(crop_h * crop_w)
        if int(payload_nbytes) != expected:
            raise ValueError(
                f'{self.root}: logical bbox byte count mismatch at slice {idx_i}: '
                f'{payload_nbytes} != {expected}'
            )
        if bool(self._packbits_payload):
            packed_w = int((crop_w + 7) // 8)
            encoded_expected = int(crop_h * packed_w)
            if int(payload_size) != int(encoded_expected):
                raise ValueError(
                    f'{self.root}: packed payload byte count mismatch at slice {idx_i}: '
                    f'{payload_size} != {encoded_expected}'
                )
            packed = np.frombuffer(
                payload, dtype=np.uint8, count=int(encoded_expected),
            ).reshape((crop_h, packed_w))
            crop = np.unpackbits(
                packed, axis=1, count=int(crop_w), bitorder='little',
            )
        else:
            if int(payload_size) != int(expected):
                raise ValueError(
                    f'{self.root}: raw payload byte count mismatch at slice {idx_i}: '
                    f'{payload_size} != {expected}'
                )
            crop = np.frombuffer(
                payload, dtype=np.uint8, count=expected,
            ).reshape((crop_h, crop_w))
        target_dtype = np.dtype(dtype)
        if crop.dtype != target_dtype:
            crop = crop.astype(target_dtype, copy=False)
        return y0, x0, y1, x1, crop

    def fill_decoded_slice_into(self, idx: int, out: np.ndarray) -> None:
        """Decode one raw-bbox slice directly into an existing ``(Y,X)`` array."""
        idx_i = int(idx)
        z_dim, h, w = self.shape
        if idx_i < 0 or idx_i >= int(z_dim):
            raise IndexError(idx_i)
        out_arr = np.asarray(out)
        if tuple(int(x) for x in out_arr.shape) != (int(h), int(w)):
            raise ValueError(f'{self.root}: output slice shape {tuple(out_arr.shape)} != {(int(h), int(w))}')
        out_arr.fill(np.uint8(0))
        decoded = self.decode_slice_crop(idx_i, dtype=out_arr.dtype)
        if decoded is None:
            return
        y0, x0, y1, x1, crop = decoded
        out_arr[y0:y1, x0:x1] = crop

    def decode_slice(self, idx: int, *, dtype: np.dtype | str | type = np.uint8) -> np.ndarray:
        idx_i = int(idx)
        z_dim, h, w = self.shape
        if idx_i < 0 or idx_i >= int(z_dim):
            raise IndexError(idx_i)
        out = np.zeros((int(h), int(w)), dtype=np.dtype(dtype))
        self.fill_decoded_slice_into(idx_i, out)
        return out

    @runtime_telemetry_phase('cvol.iter_native_sparse')
    def iter_native_sparse_members(
        self,
        z_start: int,
        z_stop: int,
        *,
        member_bytes: int,
        sparse_consumer: Optional[Callable[[int, int, int, np.ndarray], None]] = None,
    ) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
        """Yield whole-slice gzip-member buffers assembled only from overlapping bbox chunks.
        
        An optional sparse consumer receives each output-coordinate crop while it is already cache-resident."""
        z_dim, h, w = (int(v) for v in self.shape)
        z0 = int(np.clip(int(z_start), 0, z_dim))
        z1 = int(np.clip(int(z_stop), z0, z_dim))
        slice_bytes = int(h) * int(w)
        slices_per_member = max(1, int(member_bytes) // max(1, int(slice_bytes)))

        for first_z in range(int(z0), int(z1), int(slices_per_member)):
            last_z_exclusive = min(int(z1), int(first_z) + int(slices_per_member))
            ln = int(last_z_exclusive - first_z) * int(slice_bytes)
            candidates: List[int] = []
            for z in range(int(first_z), int(last_z_exclusive)):
                rec = self.index[int(z)]
                if int(rec['kind']) == 0:
                    continue
                if int(rec['kind']) != 1:
                    raise ValueError(
                        f'{self.root}: invalid raw-mask chunk marker '
                        f'{int(rec["kind"])} at slice {int(z)}'
                    )
                candidates.append(int(z))

            if not candidates:
                yield int(ln), None
                continue

            # one owned, slice-aligned buffer is handed to the member writer. All
            # large zero-fill/copy operations below are NumPy native loops (no Python row
            # assembly), and the default member writer can retain this allocation directly.
            member = np.zeros(
                (int(last_z_exclusive - first_z), int(h), int(w)),
                dtype=np.uint8,
                order='C',
            )
            for z in candidates:
                decoded = self.decode_slice_crop(int(z), dtype=np.uint8)
                if decoded is None:  # index changed/truncated after classification
                    continue
                y0, x0, y1, x1, crop = decoded
                plane = member[int(z - first_z)]
                np.copyto(
                    plane[int(y0):int(y1), int(x0):int(x1)],
                    np.asarray(crop, dtype=np.uint8),
                    casting='unsafe',
                )
                if sparse_consumer is not None:
                    sparse_consumer(int(z), int(y0), int(x0), np.asarray(crop, dtype=np.uint8))
            yield int(ln), member

    @runtime_telemetry_phase('cvol.iter_restored_sparse')
    def iter_restored_sparse_members(
        self,
        output_shape: Tuple[int, int, int],
        z_start: int,
        z_stop: int,
        *,
        member_bytes: int,
        sparse_consumer: Optional[Callable[[int, int, int, np.ndarray], None]] = None,
    ) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
        """Yield restored cvol payload without constructing intermediate dense planes.

 Temporal restore is resolved from the index and every contributing source bbox is
 OR-scattered directly into its destination member plane. XY restore operates on
 the bbox plus its mathematically required global-coordinate influence halo; it
 never decodes or resizes an ``in_h x in_w`` zero background. Members contain an
 integral number of output slices so gzip, sparse assembly, and mirror tee all share
 one stable unit of work."""
        # Local import keeps the package dependency graph acyclic.
        from .outputs import (
            _resize_sparse_binary_crop_to_output_region,
            _restore_source_indices_for_output_z,
        )

        in_t, in_h, in_w = (int(v) for v in self.shape)
        out_t, out_h, out_w = (int(v) for v in output_shape)
        z0 = int(np.clip(int(z_start), 0, out_t))
        z1 = int(np.clip(int(z_stop), z0, out_t))
        slice_bytes = int(out_h) * int(out_w)
        slices_per_member = max(1, int(member_bytes) // max(1, int(slice_bytes)))

        for first_z in range(int(z0), int(z1), int(slices_per_member)):
            last_z_exclusive = min(int(z1), int(first_z) + int(slices_per_member))
            raw_len = int(last_z_exclusive - first_z) * int(slice_bytes)
            source_lists: List[Tuple[int, List[int]]] = []
            any_nonempty = False
            for out_z in range(int(first_z), int(last_z_exclusive)):
                sources = _restore_source_indices_for_output_z(int(in_t), int(out_t), int(out_z))
                active: List[int] = []
                for src_z in sources:
                    rec = self.index[int(src_z)]
                    kind = int(rec['kind'])
                    if kind == 1:
                        active.append(int(src_z))
                    elif kind != 0:
                        raise ValueError(
                            f'{self.root}: invalid raw-mask chunk marker {kind} '
                            f'at slice {int(src_z)}'
                        )
                if active:
                    any_nonempty = True
                source_lists.append((int(out_z), active))

            if not any_nonempty:
                yield int(raw_len), None
                continue

            member = np.zeros(
                (int(last_z_exclusive - first_z), int(out_h), int(out_w)),
                dtype=np.uint8,
                order='C',
            )
            for out_z, sources in source_lists:
                plane = member[int(out_z - first_z)]
                consumer_y0, consumer_x0 = int(out_h), int(out_w)
                consumer_y1, consumer_x1 = 0, 0
                for src_z in sources:
                    decoded = self.decode_slice_crop(int(src_z), dtype=np.uint8)
                    if decoded is None:
                        continue
                    y0, x0, y1, x1, crop = decoded
                    if int(in_h) == int(out_h) and int(in_w) == int(out_w):
                        out_y0, out_x0 = int(y0), int(x0)
                        restored_crop = np.asarray(crop, dtype=np.uint8)
                    else:
                        restored = _resize_sparse_binary_crop_to_output_region(
                            np.asarray(crop, dtype=np.uint8),
                            source_shape=(int(in_h), int(in_w)),
                            source_bbox=(int(y0), int(x0), int(y1), int(x1)),
                            output_shape=(int(out_h), int(out_w)),
                        )
                        if restored is None:
                            continue
                        out_y0, out_x0, _out_y1, _out_x1, restored_crop = restored
                    dst = plane[
                        int(out_y0):int(out_y0) + int(restored_crop.shape[0]),
                        int(out_x0):int(out_x0) + int(restored_crop.shape[1]),
                    ]
                    np.bitwise_or(dst, restored_crop, out=dst)
                    consumer_y0 = min(int(consumer_y0), int(out_y0))
                    consumer_x0 = min(int(consumer_x0), int(out_x0))
                    consumer_y1 = max(
                        int(consumer_y1), int(out_y0) + int(restored_crop.shape[0]),
                    )
                    consumer_x1 = max(
                        int(consumer_x1), int(out_x0) + int(restored_crop.shape[1]),
                    )
                if sparse_consumer is not None and int(consumer_y1) > int(consumer_y0):
                    # Tee the temporal union once. Resizing individual contributions could
                    # lose two sub-half-LSB INTER_AREA slivers whose union rounds to one.
                    sparse_consumer(
                        int(out_z), int(consumer_y0), int(consumer_x0),
                        plane[
                            int(consumer_y0):int(consumer_y1),
                            int(consumer_x0):int(consumer_x1),
                        ],
                    )
            yield int(raw_len), member


    def unlink(self) -> None:
        self.close()
        if bool(getattr(self, '_ram_cache_ref_held', False)):
            self._ram_cache_ref_held = False
            _release_raw_store_chunks_ram_cache(self.chunks_path)
        _invalidate_raw_store_chunks_ram_cache(self.chunks_path)
        release_memfd_owners_under(self.root)
        shutil.rmtree(self.root, ignore_errors=True)

def materialize_raw_bbox_mask_store_workspace(
    store_path: Path,
    out_path: Path,
    *,
    desc: str,
    workers: int,
) -> np.ndarray:
    """Decode a sparse raw-bbox store into one path-backed uint8 workspace."""
    store = RawBBoxMaskStore.open(Path(store_path), mmap_payload=True)
    workspace: Optional[np.ndarray] = None
    try:
        workspace = allocate_workspace_array(
            shape=tuple(int(v) for v in store.shape),
            dtype=np.uint8,
            path=Path(out_path),
            desc=str(desc),
            prefer_memory=False,
            prefer_memfd=False,
            initialize_zero=False,
        )

        def _decode(idx: int) -> None:
            store.fill_decoded_slice_into(int(idx), workspace[int(idx)])

        parallel_for_indices_chunked(
            int(store.shape[0]),
            _decode,
            max_workers=choose_slice_parallel_workers(int(workers), int(store.shape[0])),
            desc=f'{desc}: sparse decode',
            show_progress=False,
            target_chunks_per_worker=2,
        )
        flush_array(workspace)
        return workspace
    except BaseException:
        close_memmap_array_without_flush(workspace)
        raise
    finally:
        store.close()

def _encode_bool_mask_slice_payload(
    idx: int,
    mask_bool: np.ndarray,
    *,
    packbits_payload: bool = False,
) -> RawBBoxSlicePayload:
    # scan the slice as handed in (uint8 or bool — any nonzero is foreground)
    # instead of casting the whole slice to bool first (a full-slice copy per encoded slice
    # across >200 G voxels per run). The row reduction doubles as the emptiness test, and the
    # binarizing compare+copy happens only inside the nonzero bbox crop.
    mask_arr = np.asarray(mask_bool)
    if mask_arr.size == 0:
        return RawBBoxSlicePayload(idx=int(idx), is_empty=True)

    rows = np.any(mask_arr, axis=1)
    if not np.any(rows):
        return RawBBoxSlicePayload(idx=int(idx), is_empty=True)
    cols = np.any(mask_arr, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(rows.size - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(cols.size - np.argmax(cols[::-1]))
    crop = np.ascontiguousarray(np.asarray(mask_arr[y0:y1, x0:x1]) > 0, dtype=np.uint8)
    encoded = (
        np.packbits(crop, axis=1, bitorder='little')
        if bool(packbits_payload)
        else crop
    )
    payload = np.ascontiguousarray(encoded).tobytes(order='C')
    return RawBBoxSlicePayload(
        idx=int(idx),
        is_empty=False,
        y0=int(y0),
        x0=int(x0),
        y1=int(y1),
        x1=int(x1),
        # payload_nbytes remains the logical decoded bbox size; payload_size in the
        # index records the physical encoded byte count.
        payload_nbytes=int(crop.size),
        payload=payload,
        foreground_voxels=int(np.count_nonzero(crop)),
    )

def _encode_ctile_slice(idx: int, tile_mask_mm: np.ndarray) -> RawBBoxSlicePayload:
    # the encoder handles raw uint8 slices directly; the old `>0` here was
    # another full-slice compare+copy per slice.
    return _encode_bool_mask_slice_payload(int(idx), tile_mask_mm[int(idx)])

def _write_raw_bbox_payload_store(
    *,
    shape: Tuple[int, int, int],
    store_dir: Path,
    encode_slice: Callable[[int], RawBBoxSlicePayload],
    format_name: str,
    desc: str,
    workers: int = 1,
    extra_meta: Optional[Dict[str, object]] = None,
    force_path_backed: bool = False,
) -> Dict[str, object]:
    """Write a slice-chunked bbox binary mask store.

 External CTILE/CVOL formats use raw uint8 bbox payloads. The distinct private
 ``INTERNAL_PACKED_CVOL_FORMAT`` uses row-wise packbits for terminal retention."""
    fmt = str(format_name)
    if fmt not in MASK_STORE_FORMATS:
        raise ValueError(f'Unsupported raw bbox mask format: {fmt}')

    shape_i = (int(shape[0]), int(shape[1]), int(shape[2]))
    if any(v < 0 for v in shape_i):
        raise ValueError(f'{desc}: invalid raw store shape {shape_i}')

    store_dir = Path(store_dir)
    chunks_path_prewrite = store_dir / 'chunks.bin'
    _invalidate_raw_store_chunks_ram_cache(chunks_path_prewrite)
    release_memfd_owners_under(store_dir)
    if store_dir.exists():
        shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = store_dir / 'chunks.bin'
    index_path = store_dir / 'index.bin'
    meta_path = store_dir / 'meta.json'

    index = np.zeros((int(shape_i[0]),), dtype=CTILE_INDEX_DTYPE)
    worker_count = choose_slice_parallel_workers(int(workers), int(shape_i[0]))
    payload_bytes = 0
    nonempty_slices = 0
    foreground_voxels = 0
    min_t, max_t = int(shape_i[0]), -1
    min_y, max_y = int(shape_i[1]), -1
    min_x, max_x = int(shape_i[2]), -1

    if worker_count <= 1:
        iterable: Iterable[RawBBoxSlicePayload] = (encode_slice(idx) for idx in range(int(shape_i[0])))
    else:
        iterable = parallel_map_in_order(
            encode_slice,
            range(int(shape_i[0])),
            max_workers=worker_count,
            max_pending=max(worker_count, worker_count * 2),
        )

    offset = 0
    with open_raw_store_payload_writer(
        chunks_path,
        f'{desc} chunks',
        force_path_backed=bool(force_path_backed),
    ) as chunks_fh:
        for item in iterable:
            idx = int(item.idx)
            if idx < 0 or idx >= int(shape_i[0]):
                raise ValueError(f'{desc}: encoder returned out-of-range slice index {idx}')
            if bool(item.is_empty):
                index[idx]['kind'] = np.uint8(0)
                continue
            payload = bytes(item.payload)
            payload_size = int(len(payload))
            chunks_fh.write(payload)
            index[idx]['kind'] = np.uint8(1)
            index[idx]['offset'] = np.uint64(offset)
            index[idx]['payload_size'] = np.uint64(payload_size)
            index[idx]['y0'] = np.uint32(item.y0)
            index[idx]['x0'] = np.uint32(item.x0)
            index[idx]['y1'] = np.uint32(item.y1)
            index[idx]['x1'] = np.uint32(item.x1)
            index[idx]['payload_nbytes'] = np.uint64(item.payload_nbytes)
            offset += payload_size
            payload_bytes += payload_size
            foreground_voxels += int(item.foreground_voxels)
            nonempty_slices += 1
            min_t = min(int(min_t), int(idx))
            max_t = max(int(max_t), int(idx))
            min_y = min(int(min_y), int(item.y0))
            max_y = max(int(max_y), int(item.y1) - 1)
            min_x = min(int(min_x), int(item.x0))
            max_x = max(int(max_x), int(item.x1) - 1)

    index.tofile(index_path)
    raw_logical_bytes = int(array_nbytes(shape_i, np.uint8))
    segment_extent_ijk = (
        _nrrd_empty_segment_extent()
        if int(max_t) < 0
        else (int(min_x), int(max_x), int(min_y), int(max_y), int(min_t), int(max_t))
    )
    stats: Dict[str, object] = {
        'nonempty_slices': int(nonempty_slices),
        'empty_slices': int(shape_i[0] - nonempty_slices),
        'foreground_voxels': int(foreground_voxels),
        'logical_raw_uint8_bytes': int(raw_logical_bytes),
        'raw_payload_bytes': int(payload_bytes),
        'index_bytes': int(index.nbytes),
        'segment_extent_ijk': _segment_extent_to_json(segment_extent_ijk),
        'segment_extent_shape_tyx': [int(shape_i[0]), int(shape_i[1]), int(shape_i[2])],
    }
    meta = {
        'format': fmt,
        'shape': [int(shape_i[0]), int(shape_i[1]), int(shape_i[2])],
        'dtype': 'bool',
        'logical_dtype_in_pipeline': 'uint8_0_or_1',
        'chunking': 'slice',
        'precodec': (
            'numpy_packbits_axis_x_little'
            if fmt == INTERNAL_PACKED_CVOL_FORMAT
            else 'none'
        ),
        'compressor': 'none',
        'bbox_per_chunk': True,
        'zero_chunk_elision': True,
        'index_dtype': 'ctile-index-v2-raw',
        'index_record_bytes': int(CTILE_INDEX_DTYPE.itemsize),
        'payload_shape_encoding': (
            'packbits_rows_ceil_width_div_8_bbox_shape_from_index'
            if fmt == INTERNAL_PACKED_CVOL_FORMAT
            else 'raw_uint8_bbox_shape_from_index'
        ),
        'description': str(desc),
        'segment_extent_ijk': _segment_extent_to_json(segment_extent_ijk),
        'segment_extent_axis_order': 'Slicer IJK inclusive extent: minX maxX minY maxY minT maxT for internal layer order (t,Y,X)',
        'segment_extent_shape_tyx': [int(shape_i[0]), int(shape_i[1]), int(shape_i[2])],
        'stats': stats,
    }
    _merge_raw_bbox_extra_meta(meta, extra_meta)
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    print(
        f'{desc}: raw bbox mask store {store_dir} '
        f'(logical_raw={raw_logical_bytes / GIB:.2f} GiB, payload={payload_bytes / GIB:.2f} GiB, '
        f'nonempty_slices={nonempty_slices})'
    )
    return stats

def write_raw_bbox_mask_store(
    mask_volume: np.ndarray,
    store_dir: Path,
    *,
    format_name: str = CVOL_FORMAT,
    desc: str = 'Raw bbox mask store',
    workers: int = 1,
    extra_meta: Optional[Dict[str, object]] = None,
    force_path_backed: bool = False,
) -> Dict[str, object]:
    """Write a raw bbox mask store."""
    arr = np.asarray(mask_volume)
    if arr.ndim != 3:
        raise ValueError(f'{desc}: expected 3D mask volume, got shape {arr.shape}')
    shape = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))

    def _encode(idx: int) -> RawBBoxSlicePayload:
        if str(format_name) == INTERNAL_PACKED_CVOL_FORMAT:
            return _encode_bool_mask_slice_payload(
                int(idx), arr[int(idx)], packbits_payload=True,
            )
        return _encode_ctile_slice(int(idx), arr)

    return _write_raw_bbox_payload_store(
        shape=shape,
        store_dir=Path(store_dir),
        encode_slice=_encode,
        format_name=str(format_name),
        desc=str(desc),
        workers=int(workers),
        extra_meta=extra_meta,
        force_path_backed=bool(force_path_backed),
    )
