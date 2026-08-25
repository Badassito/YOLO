"""Slice labeling, union-find, and component metadata."""

from __future__ import annotations

import gc
import math
import os
import threading
import time
import weakref
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
    Sequence,
    TYPE_CHECKING,
    Tuple,
)
import numpy as np
from ._deps import _numba, cv2, tqdm

from .config import (
    GIB,
)
from .runtime import (
    array_nbytes,
    choose_slice_parallel_workers,
    estimate_voidfill_workspace_bytes,
    flush_array,
    interpolation_process_worker_active,
    numa_interleave_memory,
    parallel_for_indices_chunked,
    parallel_map_unordered,
    runtime_telemetry,
    runtime_telemetry_phase,
    should_use_in_memory_workspace,
    workspace_budget_summary,
)

# Explicit lower-layer dependencies keep imports one-way.
from .workspace import (
    _cpu_count,
    _env_flag,
    _env_float,
    _env_int,
)
from .inference import (
    _cv2_connected_components,
    _try_import_cupy_ndimage,
)
from .interpolation import (
    SliceComponentRecord,
    SliceEndpointSeed,
    _component_centroid_anchor,
    compiled_topology_kernels_enabled,
)


if TYPE_CHECKING:
    from .backprojection import (
        _MainProcessGpuStageLease,
        _announce_main_gpu_stage_skip_once,
        _trim_main_process_cuda_device,
        _try_acquire_specific_main_process_gpu_stage,
    )

if _numba is not None:
    @_numba.njit(cache=True, nogil=True)  # type: ignore[misc]
    def _numba_union_find_batch_kernel(
        parent: np.ndarray,   # int64, mutated in place
        rank: np.ndarray,     # int32, mutated in place
        touches: np.ndarray,  # bool, mutated in place
        a_ids: np.ndarray,    # int64
        b_ids: np.ndarray,    # int64
    ) -> None:
        # Identical algorithm to _UnionFind.find/union (path halving + union by rank), applied
        # to a whole batch of pairs in one nogil call.
        for i in range(a_ids.shape[0]):
            ra = a_ids[i]
            while parent[ra] != ra:
                parent[ra] = parent[parent[ra]]
                ra = parent[ra]
            rb = b_ids[i]
            while parent[rb] != rb:
                parent[rb] = parent[parent[rb]]
                rb = parent[rb]
            if ra == rb:
                continue
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if touches[rb]:
                touches[ra] = True
            if rank[ra] == rank[rb]:
                rank[ra] += 1
else:
    _numba_union_find_batch_kernel = None

_NUMBA_UNION_FIND_KERNEL_RUNTIME_DISABLED = False

class _UnionFind:
    """Array-backed disjoint-set structure with batched pair merges and path compression."""

    def __init__(self) -> None:
        cap = 1024
        self.parent = np.arange(cap, dtype=np.int64)
        self.rank = np.zeros(cap, dtype=np.int32)
        self.touches_boundary = np.zeros(cap, dtype=bool)
        self._size = 1

    def _ensure_capacity(self, need: int) -> None:
        cap = int(self.parent.shape[0])
        if int(need) <= cap:
            return
        new_cap = max(int(need), cap * 2)
        new_parent = np.arange(new_cap, dtype=np.int64)
        new_parent[:cap] = self.parent
        new_rank = np.zeros(new_cap, dtype=np.int32)
        new_rank[:cap] = self.rank
        new_touches = np.zeros(new_cap, dtype=bool)
        new_touches[:cap] = self.touches_boundary
        self.parent = new_parent
        self.rank = new_rank
        self.touches_boundary = new_touches

    def new_ids(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0,), dtype=np.uint32)
        start = int(self._size)
        stop = start + int(count)
        if stop >= 2 ** 32:
            raise RuntimeError('3D component id space exceeded uint32 capacity')
        self._ensure_capacity(stop)
        # parent[start:stop] is already arange-initialized (construction/growth), rank/touches zero.
        self._size = stop
        return np.arange(start, stop, dtype=np.uint32)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(self, a: int, b: int) -> int:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return ra

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.touches_boundary[ra] = bool(self.touches_boundary[ra] or self.touches_boundary[rb])
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return int(ra)

    def union_pair_codes(self, codes: np.ndarray) -> None:
        """Merge a batch of uint64 pair codes ((a << 32) | b) in one call."""
        global _NUMBA_UNION_FIND_KERNEL_RUNTIME_DISABLED
        codes_arr = np.asarray(codes, dtype=np.uint64)
        if codes_arr.size <= 0:
            return
        a_ids = (codes_arr >> np.uint64(32)).astype(np.int64, copy=False)
        b_ids = (codes_arr & np.uint64(0xFFFFFFFF)).astype(np.int64, copy=False)
        if (
            _numba_union_find_batch_kernel is not None
            and compiled_topology_kernels_enabled()
            and not _NUMBA_UNION_FIND_KERNEL_RUNTIME_DISABLED
        ):
            try:
                _numba_union_find_batch_kernel(
                    self.parent, self.rank, self.touches_boundary,
                    np.ascontiguousarray(a_ids), np.ascontiguousarray(b_ids),
                )
                return
            except Exception as exc:
                _NUMBA_UNION_FIND_KERNEL_RUNTIME_DISABLED = True
                print(f'Warning: numba union-find batch kernel failed ({exc}); using the python loop.')
        for i in range(int(a_ids.shape[0])):
            self.union(int(a_ids[i]), int(b_ids[i]))

    def mark_boundary(self, x: int) -> None:
        self.touches_boundary[self.find(int(x))] = True

    def root_map(self) -> np.ndarray:
        # vectorized pointer jumping — every id follows its parent chain in
        # O(log depth) whole-array passes (depth is tiny thanks to path halving during unions).
        n = int(self._size)
        p = self.parent[:n].copy()
        while True:
            pp = p[p]
            if np.array_equal(pp, p):
                break
            p = pp
        out = p.astype(np.uint32, copy=False)
        out[0] = np.uint32(0)
        return np.ascontiguousarray(out)

def _adjacent_xy_offsets_for_3d_connectivity(connectivity: int) -> Tuple[Tuple[int, int], ...]:
    """Return XY offsets that connect components across adjacent z-slices.

 Connectivity is interpreted in the standard cubic-neighborhood sense:
 - 6-connected: only face adjacency across z, so the same (y, x) position
 - 18-connected: face/edge adjacency, so same position plus cardinal XY offsets
 - 26-connected: face/edge/corner adjacency, so the full 3x3 XY neighborhood"""
    connectivity_i = int(connectivity)
    if connectivity_i == 6:
        return ((0, 0),)
    if connectivity_i == 18:
        return tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if abs(dy) + abs(dx) <= 1)
    if connectivity_i == 26:
        return tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    raise ValueError('3D connectivity must be one of 6, 18, or 26')

def _slice_connectivity_for_3d_connectivity(connectivity: int) -> int:
    connectivity_i = int(connectivity)
    if connectivity_i == 6:
        return 4
    if connectivity_i in (18, 26):
        return 8
    raise ValueError('3D connectivity must be one of 6, 18, or 26')

def _adjacent_gid_pair_codes(
    prev_gid: np.ndarray,
    curr_gid: np.ndarray,
    xy_offsets: Optional[Sequence[Tuple[int, int]]] = None,
    prev_offset: int = 0,
    curr_offset: int = 0,
) -> np.ndarray:
    """Return unique touching component-id pairs encoded as uint64 values.

 The upper 32 bits contain the previous-slice gid and the lower 32 bits contain the
 current-slice gid. ``prev_offset`` and ``curr_offset`` allow callers to pass per-slice
 local labels and encode global provisional ids without first rewriting the whole volume.
 Pair extraction is independent for each adjacent slice pair, so callers can run this
 function concurrently and then apply union-find merges serially."""
    h, w = prev_gid.shape
    offsets = tuple(xy_offsets) if xy_offsets is not None else _adjacent_xy_offsets_for_3d_connectivity(26)
    code_parts: List[np.ndarray] = []

    for dy, dx in offsets:
        dy_i = int(dy)
        dx_i = int(dx)
        py0 = max(0, -dy_i)
        py1 = min(h, h - dy_i)
        cy0 = max(0, dy_i)
        cy1 = min(h, h + dy_i)
        px0 = max(0, -dx_i)
        px1 = min(w, w - dx_i)
        cx0 = max(0, dx_i)
        cx1 = min(w, w + dx_i)

        if py0 >= py1 or px0 >= px1 or cy0 >= cy1 or cx0 >= cx1:
            continue

        a = prev_gid[py0:py1, px0:px1]
        b = curr_gid[cy0:cy1, cx0:cx1]
        overlap = (a > 0) & (b > 0)
        if not np.any(overlap):
            continue

        a_vals = a[overlap].astype(np.uint64, copy=False)
        b_vals = b[overlap].astype(np.uint64, copy=False)
        if int(prev_offset) != 0:
            a_vals = a_vals + np.uint64(int(prev_offset))
        if int(curr_offset) != 0:
            b_vals = b_vals + np.uint64(int(curr_offset))
        codes = (a_vals << np.uint64(32)) | b_vals
        if codes.size > 0:
            code_parts.append(np.unique(codes))

    if not code_parts:
        return np.zeros((0,), dtype=np.uint64)
    if len(code_parts) == 1:
        return np.asarray(code_parts[0], dtype=np.uint64)
    return np.unique(np.concatenate(code_parts).astype(np.uint64, copy=False))

def _mark_boundary_components_from_local_labels(
    uf: _UnionFind,
    local_to_gid: np.ndarray,
    labels2d: np.ndarray,
    z: int,
    z_max: int,
) -> None:
    if labels2d.size == 0 or local_to_gid.size <= 1:
        return

    if z == 0 or z == z_max:
        for gid in local_to_gid[1:]:
            uf.mark_boundary(int(gid))

    border_vals = np.unique(np.concatenate([
        labels2d[0, :],
        labels2d[-1, :],
        labels2d[:, 0],
        labels2d[:, -1],
    ]))
    border_vals = border_vals[border_vals > 0]
    for local_lbl in border_vals.tolist():
        uf.mark_boundary(int(local_to_gid[int(local_lbl)]))

def fill_3d_voids_inplace_streaming(
    mask_mm: np.ndarray,
    work_prefix: Path,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    connectivity: int = 6,
) -> None:
    """Fill enclosed 3D background components with configurable 6, 18, or 26 connectivity.
    
    The label workspace prefers anonymous memory and falls back to a memmap when admission requires it."""
    env_conn = os.environ.get('YOLO_TTA_VOIDFILL_CONNECTIVITY', '').strip()
    if env_conn:
        try:
            connectivity = int(env_conn)
        except Exception:
            pass
    connectivity = int(connectivity)
    if connectivity not in (6, 18, 26):
        raise ValueError('3D void fill connectivity must be one of 6, 18, or 26')

    z_dim, h, w = mask_mm.shape
    if z_dim <= 0:
        return

    slice_connectivity = _slice_connectivity_for_3d_connectivity(connectivity)
    adjacent_offsets = _adjacent_xy_offsets_for_3d_connectivity(connectivity)

    estimated_bytes = estimate_voidfill_workspace_bytes((z_dim, h, w))
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(estimated_bytes, reserve_bytes=reserve_bytes)
    budget = workspace_budget_summary(estimated_bytes, reserve_bytes=reserve_bytes)
    if use_in_memory:
        print(f"3D void fill workspace: in-memory ({budget}, background connectivity={connectivity})")
        bg_gid_store: np.ndarray = np.zeros((z_dim, h, w), dtype=np.uint32)
        bg_gid_path: Optional[Path] = None
    else:
        print(f"3D void fill workspace: disk-backed ({budget}, background connectivity={connectivity}) -> {work_prefix.parent}")
        bg_gid_path = work_prefix.with_suffix('.bg_gid.u32.dat')
        bg_gid_path.parent.mkdir(parents=True, exist_ok=True)
        bg_gid_store = np.memmap(bg_gid_path, dtype=np.uint32, mode='w+', shape=(z_dim, h, w))
    numa_interleave_memory(bg_gid_store, desc='3D void fill gid store')  #

    uf = _UnionFind()
    prev_gid_slice: Optional[np.ndarray] = None

    for z in tqdm(range(z_dim), desc='3D void fill: slice labeling'):
        bg = (np.asarray(mask_mm[z]) == 0).astype(np.uint8, copy=False)
        num_labels, labels2d = cv2.connectedComponents(bg, connectivity=slice_connectivity, ltype=cv2.CV_32S)
        if int(num_labels) <= 1:
            bg_gid_store[z, :, :] = 0
            prev_gid_slice = None
            continue

        local_to_gid = np.zeros((int(num_labels),), dtype=np.uint32)
        local_to_gid[1:] = uf.new_ids(int(num_labels) - 1)
        gid_slice = local_to_gid[labels2d]
        bg_gid_store[z, :, :] = gid_slice

        _mark_boundary_components_from_local_labels(uf, local_to_gid, labels2d, z=z, z_max=z_dim - 1)

        if prev_gid_slice is not None and np.any(prev_gid_slice) and np.any(gid_slice):
            uf.union_pair_codes(_adjacent_gid_pair_codes(prev_gid_slice, gid_slice, adjacent_offsets))

        prev_gid_slice = np.asarray(gid_slice)

    root_map = uf.root_map()
    touches_by_gid = np.zeros(root_map.shape, dtype=bool)
    for gid in range(1, root_map.shape[0]):
        touches_by_gid[gid] = bool(uf.touches_boundary[int(root_map[gid])])

    for z in tqdm(range(z_dim), desc='3D void fill: apply enclosed components'):
        gid_slice = np.asarray(bg_gid_store[z])
        enclosed = (gid_slice > 0) & (~touches_by_gid[gid_slice])
        if np.any(enclosed):
            mask_mm[z, enclosed] = np.uint8(1)

    flush_array(mask_mm)
    flush_array(bg_gid_store)
    del bg_gid_store
    if bg_gid_path is not None and not keep_temp:
        try:
            bg_gid_path.unlink(missing_ok=True)
        except Exception:
            pass

if _numba is not None:
    @_numba.njit(cache=True, nogil=True, parallel=True)  # type: ignore[misc]
    def _numba_compact_relabel_kernel(labels, lut_flat, lut_offsets, bboxes, counts):  # pragma: no cover - jit
        z_dim = labels.shape[0]
        for z in _numba.prange(z_dim):
            if counts[z] == 0:
                continue
            off = lut_offsets[z]
            y0 = bboxes[z, 0]
            y1 = bboxes[z, 1]
            x0 = bboxes[z, 2]
            x1 = bboxes[z, 3]
            for y in range(y0, y1):
                row = labels[z, y]
                for x in range(x0, x1):
                    v = row[x]
                    if v != 0:
                        row[x] = lut_flat[off + v]

    @_numba.njit(cache=True, nogil=True, parallel=True)  # type: ignore[misc]
    def _numba_keep_lut_apply_kernel(labels, keep_flat, lut_offsets, bboxes, apply_slice, mask_out):  # pragma: no cover - jit
        z_dim = labels.shape[0]
        for z in _numba.prange(z_dim):
            if apply_slice[z] == 0:
                continue
            off = lut_offsets[z]
            y0 = bboxes[z, 0]
            y1 = bboxes[z, 1]
            x0 = bboxes[z, 2]
            x1 = bboxes[z, 3]
            for y in range(y0, y1):
                lrow = labels[z, y]
                mrow = mask_out[z, y]
                for x in range(x0, x1):
                    mrow[x] = keep_flat[off + lrow[x]]

    @_numba.njit(cache=True, nogil=True, parallel=True)  # type: ignore[misc]
    def _numba_sparse_keep_lut_apply_kernel(  # pragma: no cover - jit
        labels_flat, label_offsets, keep_flat, lut_offsets, bboxes, apply_slice, mask_out,
    ):
        """Keep-largest apply directly from the packed per-slice label arena."""
        z_dim = apply_slice.shape[0]
        for z in _numba.prange(z_dim):
            if apply_slice[z] == 0:
                continue
            keep_off = lut_offsets[z]
            label_off = label_offsets[z]
            y0 = bboxes[z, 0]
            y1 = bboxes[z, 1]
            x0 = bboxes[z, 2]
            x1 = bboxes[z, 3]
            crop_w = x1 - x0
            for y in range(y0, y1):
                row_off = label_off + (y - y0) * crop_w
                mrow = mask_out[z, y]
                for x in range(x0, x1):
                    mrow[x] = keep_flat[keep_off + labels_flat[row_off + x - x0]]
else:
    _numba_compact_relabel_kernel = None
    _numba_keep_lut_apply_kernel = None
    _numba_sparse_keep_lut_apply_kernel = None

@dataclass(frozen=True)
class SliceLocalLabelLUTs:
    """Per-slice local-id -> canonical-compact-id lookup tables.

 Exported by label_foreground_volume_streaming when the compact relabel pass is
 skipped: slice z's local label v canonicalizes through
 ``lut_flat[lut_offsets[z] + v]``. Consumers (component tables, projection-candidate
 kernels) resolve ids through these tables instead of the full-volume relabel
 rewriting the full-volume compact/global-id store."""

    lut_flat: np.ndarray       # uint32, concatenated per-slice tables (index 0 of each = 0)
    lut_offsets: np.ndarray    # int64, one entry per slice
    component_counts: np.ndarray  # uint32, per-slice 2D component counts

    def lut_for(self, z: int) -> np.ndarray:
        lo = int(self.lut_offsets[int(z)])
        return self.lut_flat[lo:lo + int(self.component_counts[int(z)]) + 1]

def interpolation_sparse_labels_enabled() -> bool:
    """Retain only each slice's local-label bbox crop.

 The dense uint16 local-id raster was 22.7--37.1 GiB for the prioritized
 views even though every consumer already knew the per-slice foreground
 bbox. Sparse labels are default-on whenever compact relabel is skipped;
 YOLO_TTA_INTERPOLATION_SPARSE_LABELS=0 restores the dense store."""
    return _env_flag('YOLO_TTA_INTERPOLATION_SPARSE_LABELS', True)

class SparseSliceLabelStore:
    """Sparse slice-local label storage with one bbox crop per nonempty slice.
    
    Finalization packs crops into a contiguous typed arena with offsets and bboxes for compiled downstream access."""

    def __init__(self, shape: Sequence[int], dtype: np.dtype | str | type) -> None:
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) != 3:
            raise ValueError(f'SparseSliceLabelStore expects 3D shape, got {self.shape}')
        self.dtype = np.dtype(dtype)
        self.ndim = 3
        self._pending: Optional[List[Optional[Tuple[int, int, int, int, np.ndarray]]]] = [
            None
        ] * int(self.shape[0])
        self.bboxes = np.zeros((int(self.shape[0]), 4), dtype=np.int64)
        self.offsets = np.zeros((int(self.shape[0]) + 1,), dtype=np.int64)
        self.flat = np.empty((0,), dtype=self.dtype)
        self._finalized = False

    @property
    def nbytes(self) -> int:
        return int(self.flat.nbytes) if self._finalized else int(sum(
            0 if item is None else int(item[4].nbytes)
            for item in (self._pending or [])
        ))

    def write_crop(
        self,
        z: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        values: np.ndarray,
    ) -> None:
        if self._finalized:
            raise RuntimeError('SparseSliceLabelStore is finalized')
        z_i = int(z)
        y0_i, y1_i, x0_i, x1_i = int(y0), int(y1), int(x0), int(x1)
        h, w = int(self.shape[1]), int(self.shape[2])
        if not (0 <= z_i < int(self.shape[0])):
            raise IndexError(z_i)
        if y1_i <= y0_i or x1_i <= x0_i:
            self.clear_slice(z_i)
            return
        if not (0 <= y0_i < y1_i <= h and 0 <= x0_i < x1_i <= w):
            raise ValueError(
                f'sparse label crop {(y0_i, y1_i, x0_i, x1_i)} outside {(h, w)}'
            )
        crop = np.ascontiguousarray(values, dtype=self.dtype)
        if tuple(int(v) for v in crop.shape) != (y1_i - y0_i, x1_i - x0_i):
            raise ValueError(
                f'sparse label crop data {tuple(crop.shape)} != '
                f'{(y1_i - y0_i, x1_i - x0_i)}'
            )
        assert self._pending is not None
        self._pending[z_i] = (y0_i, y1_i, x0_i, x1_i, crop)
        self.bboxes[z_i] = np.asarray((y0_i, y1_i, x0_i, x1_i), dtype=np.int64)

    def clear_slice(self, z: int) -> None:
        if self._finalized:
            raise RuntimeError('SparseSliceLabelStore is finalized')
        z_i = int(z)
        assert self._pending is not None
        self._pending[z_i] = None
        self.bboxes[z_i] = np.int64(0)

    def finalize(self) -> None:
        if self._finalized:
            return
        pending = self._pending or []
        sizes = np.zeros((int(self.shape[0]),), dtype=np.int64)
        for z, item in enumerate(pending):
            if item is not None:
                sizes[int(z)] = np.int64(item[4].size)
        if sizes.size:
            self.offsets[1:] = np.cumsum(sizes, dtype=np.int64)
        pending_bytes = int(np.sum(sizes, dtype=np.int64)) * int(self.dtype.itemsize)
        self.flat = np.empty((int(self.offsets[-1]),), dtype=self.dtype)
        for z, item in enumerate(pending):
            if item is None:
                continue
            lo, hi = int(self.offsets[int(z)]), int(self.offsets[int(z) + 1])
            # v17.0.4: release each source crop as soon as it has been copied.  ``flat`` is
            # a lazily-faulted anonymous mapping, so destination pages become resident while
            # the corresponding pending crop is dropped instead of retaining both complete
            # arenas through the entire pack.  The old implementation held roughly 2P live
            # bytes at the end of this loop for a packed payload of P bytes.
            crop = item[4]
            pending[int(z)] = None
            self.flat[lo:hi] = crop.reshape(-1)
            del crop
        self._pending = None
        self._finalized = True
        packed_bytes = int(self.flat.nbytes)
        runtime_telemetry().add('interpolation.sparse_labels.packed_bytes', packed_bytes)
        runtime_telemetry().add(
            'interpolation.sparse_labels.pending_bytes_released_during_pack',
            int(pending_bytes),
        )
        runtime_telemetry().gauge(
            'interpolation.sparse_labels.last_packed_bytes', packed_bytes,
        )

    def crop_with_origin(self, z: int) -> Tuple[int, int, np.ndarray]:
        if not self._finalized:
            self.finalize()
        z_i = int(z)
        y0, y1, x0, x1 = (int(v) for v in self.bboxes[z_i])
        if y1 <= y0 or x1 <= x0:
            return 0, 0, np.empty((0, 0), dtype=self.dtype)
        lo, hi = int(self.offsets[z_i]), int(self.offsets[z_i + 1])
        return y0, x0, self.flat[lo:hi].reshape((y1 - y0, x1 - x0))

    def read_window(self, z: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        y0_i, y1_i, x0_i, x1_i = int(y0), int(y1), int(x0), int(x1)
        out = np.zeros(
            (max(0, y1_i - y0_i), max(0, x1_i - x0_i)), dtype=self.dtype,
        )
        if out.size <= 0:
            return out
        cy0, cx0, crop = self.crop_with_origin(int(z))
        if crop.size <= 0:
            return out
        cy1, cx1 = cy0 + int(crop.shape[0]), cx0 + int(crop.shape[1])
        iy0, iy1 = max(y0_i, cy0), min(y1_i, cy1)
        ix0, ix1 = max(x0_i, cx0), min(x1_i, cx1)
        if iy0 < iy1 and ix0 < ix1:
            out[iy0 - y0_i:iy1 - y0_i, ix0 - x0_i:ix1 - x0_i] = crop[
                iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0
            ]
        return out

    def _normalize_plane_key(self, selector: object, length: int) -> Tuple[int, int]:
        if isinstance(selector, slice):
            start, stop, step = selector.indices(int(length))
            if int(step) != 1:
                raise IndexError('SparseSliceLabelStore only supports unit-stride windows')
            return int(start), int(stop)
        idx = int(selector)  # type: ignore[arg-type]
        if idx < 0:
            idx += int(length)
        if idx < 0 or idx >= int(length):
            raise IndexError(idx)
        return int(idx), int(idx) + 1

    def __getitem__(self, key: object) -> object:
        z_dim, h, w = self.shape
        if isinstance(key, tuple):
            if len(key) != 3:
                raise IndexError('SparseSliceLabelStore expects (z,y,x)')
            z_sel, y_sel, x_sel = key
            if isinstance(z_sel, (int, np.integer)):
                z_i = int(z_sel)
                if z_i < 0:
                    z_i += int(z_dim)
                y0, y1 = self._normalize_plane_key(y_sel, int(h))
                x0, x1 = self._normalize_plane_key(x_sel, int(w))
                window = self.read_window(z_i, y0, y1, x0, x1)
                if not isinstance(y_sel, slice):
                    window = window[0]
                if not isinstance(x_sel, slice):
                    window = window[..., 0]
                return window
            z0, z1 = self._normalize_plane_key(z_sel, int(z_dim))
            y0, y1 = self._normalize_plane_key(y_sel, int(h))
            x0, x1 = self._normalize_plane_key(x_sel, int(w))
            return np.stack(
                [self.read_window(z, y0, y1, x0, x1) for z in range(z0, z1)],
                axis=0,
            )
        if isinstance(key, (int, np.integer)):
            return self.read_window(int(key), 0, int(h), 0, int(w))
        z0, z1 = self._normalize_plane_key(key, int(z_dim))
        return np.stack(
            [self.read_window(z, 0, int(h), 0, int(w)) for z in range(z0, z1)],
            axis=0,
        )

    def __setitem__(self, key: object, value: object) -> None:
        # Used only by the GPU failure reset. Normal writers call write_crop.
        if isinstance(key, tuple) and len(key) == 3 and isinstance(key[0], (int, np.integer)):
            z_i = int(key[0])
            if np.isscalar(value) and int(value) == 0:
                self.clear_slice(z_i)
                return
        raise TypeError('SparseSliceLabelStore writes must use write_crop')

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        dense = np.stack(
            [self.read_window(z, 0, int(self.shape[1]), 0, int(self.shape[2]))
             for z in range(int(self.shape[0]))],
            axis=0,
        )
        if dtype is not None:
            dense = dense.astype(dtype, copy=False)
        if copy:
            dense = dense.copy()
        return dense

def interpolation_skip_compact_relabel_enabled() -> bool:
    """Interpolation consumes per-slice local ids through LUTs by default.

 The compact relabel was a full read+write pass over the label store whose only product
 was canonical ids in the raster; component tables and
 the candidate kernels now canonicalize at read time. Set
 YOLO_TTA_INTERPOLATION_SKIP_COMPACT_RELABEL=0 to restore the relabel pass."""
    return _env_flag('YOLO_TTA_INTERPOLATION_SKIP_COMPACT_RELABEL', True)

def interpolation_local_label_uint16_enabled() -> bool:
    """Use uint16 for stores that contain only slice-local ids.

 Compact relabeling still requires uint32 because canonical/global component ids
 can exceed 65535. Set YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16=0 to restore
 the previous uint32 local-label store."""
    return _env_flag('YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16', True)

def _local_label_store_dtype(compact_relabel: bool) -> np.dtype:
    if not bool(compact_relabel) and interpolation_local_label_uint16_enabled():
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)

class _LocalLabelCapacityError(RuntimeError):
    """Signal local-label dtype exhaustion so the caller can retry with a wider or CPU path."""

def _check_local_label_store_capacity(count: int, dtype: np.dtype, *, z: int) -> None:
    if np.dtype(dtype) == np.dtype(np.uint16) and int(count) > int(np.iinfo(np.uint16).max):
        raise _LocalLabelCapacityError(
            f'Interpolation slice {int(z)} has {int(count)} local components, exceeding the '
            'uint16 local-label capacity (65535). Set '
            'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16=0 and rerun.'
        )

@dataclass(frozen=True)
class BinaryVolumeSliceMetadata:
    """Known per-z foreground support for one live binary volume.

    ``slice_bboxes`` stores ``(y0, y1, x0, x1)`` with exclusive stops. A bbox may be
    an exact foreground bbox or a conservative union of contributor bboxes; either is
    sufficient to restrict connected-component labeling without changing its result.
    """

    slice_any: np.ndarray
    slice_bboxes: np.ndarray
    source: str
    exact: bool = False

_BINARY_VOLUME_SLICE_METADATA_LOCK = threading.RLock()

_BINARY_VOLUME_SLICE_METADATA: Dict[
    int, Tuple[weakref.ReferenceType[object], BinaryVolumeSliceMetadata]
] = {}

def _validate_binary_slice_metadata(
    volume: object,
    slice_any: np.ndarray,
    slice_bboxes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(v) for v in np.asarray(volume).shape)
    if len(shape) != 3:
        raise ValueError(f'binary slice metadata requires a 3D volume, got shape={shape}')
    z_dim, h, w = shape
    any_arr = np.ascontiguousarray(np.asarray(slice_any, dtype=bool).reshape(-1))
    bbox_arr = np.ascontiguousarray(np.asarray(slice_bboxes, dtype=np.int64))
    if tuple(int(v) for v in any_arr.shape) != (int(z_dim),):
        raise ValueError(
            f'binary slice metadata slice_any shape {tuple(any_arr.shape)} != {(int(z_dim),)}'
        )
    if tuple(int(v) for v in bbox_arr.shape) != (int(z_dim), 4):
        raise ValueError(
            f'binary slice metadata slice_bboxes shape {tuple(bbox_arr.shape)} '
            f'!= {(int(z_dim), 4)}'
        )
    # Empty slices must carry an empty bbox. Nonempty bboxes are clipped only after
    # validating their ordering; silently accepting inverted metadata could drop anatomy.
    for z in range(int(z_dim)):
        y0, y1, x0, x1 = (int(v) for v in bbox_arr[int(z)])
        if not bool(any_arr[int(z)]):
            bbox_arr[int(z)] = np.int64(0)
            continue
        if not (0 <= y0 < y1 <= int(h) and 0 <= x0 < x1 <= int(w)):
            raise ValueError(
                f'invalid nonempty binary slice bbox at z={int(z)}: '
                f'{(y0, y1, x0, x1)} for plane {(int(h), int(w))}'
            )
    return any_arr, bbox_arr

def register_binary_volume_slice_metadata(
    volume: object,
    slice_any: np.ndarray,
    slice_bboxes: np.ndarray,
    *,
    source: str,
    exact: bool = False,
) -> BinaryVolumeSliceMetadata:
    """Attach validated foreground-support metadata to a live ndarray by identity."""
    any_arr, bbox_arr = _validate_binary_slice_metadata(volume, slice_any, slice_bboxes)
    metadata = BinaryVolumeSliceMetadata(
        slice_any=any_arr,
        slice_bboxes=bbox_arr,
        source=str(source),
        exact=bool(exact),
    )
    key = int(id(volume))

    def _drop(_ref: weakref.ReferenceType[object], *, _key: int = int(key)) -> None:
        with _BINARY_VOLUME_SLICE_METADATA_LOCK:
            entry = _BINARY_VOLUME_SLICE_METADATA.get(int(_key))
            if entry is not None and entry[0] is _ref:
                _BINARY_VOLUME_SLICE_METADATA.pop(int(_key), None)

    try:
        ref: weakref.ReferenceType[object] = weakref.ref(volume, _drop)
    except TypeError:
        # NumPy arrays and memmaps are weak-referenceable in supported environments.
        # Keep the optimization optional rather than retaining an unbounded strong ref.
        return metadata
    with _BINARY_VOLUME_SLICE_METADATA_LOCK:
        _BINARY_VOLUME_SLICE_METADATA[int(key)] = (ref, metadata)
    return metadata

def binary_volume_slice_metadata(volume: object) -> Optional[BinaryVolumeSliceMetadata]:
    """Return live metadata only when the registry entry still names this exact object."""
    key = int(id(volume))
    with _BINARY_VOLUME_SLICE_METADATA_LOCK:
        entry = _BINARY_VOLUME_SLICE_METADATA.get(int(key))
        if entry is None:
            return None
        target = entry[0]()
        if target is not volume:
            _BINARY_VOLUME_SLICE_METADATA.pop(int(key), None)
            return None
        return entry[1]

def discard_binary_volume_slice_metadata(volume: object) -> None:
    """Invalidate cached support before or after any in-place topology-changing pass."""
    key = int(id(volume))
    with _BINARY_VOLUME_SLICE_METADATA_LOCK:
        entry = _BINARY_VOLUME_SLICE_METADATA.get(int(key))
        if entry is not None and entry[0]() is volume:
            _BINARY_VOLUME_SLICE_METADATA.pop(int(key), None)

def binary_slice_bbox_coverage(
    shape_tyx: Sequence[int],
    slice_any: np.ndarray,
    slice_bboxes: np.ndarray,
) -> Tuple[int, int, float]:
    """Return ``(nonempty_slices, bbox_pixels, bbox_fraction_of_full_volume)``."""
    z_dim, h, w = (int(v) for v in shape_tyx)
    any_arr = np.asarray(slice_any, dtype=bool).reshape((int(z_dim),))
    boxes = np.asarray(slice_bboxes, dtype=np.int64).reshape((int(z_dim), 4))
    heights = np.maximum(np.int64(0), boxes[:, 1] - boxes[:, 0])
    widths = np.maximum(np.int64(0), boxes[:, 3] - boxes[:, 2])
    bbox_pixels = int(np.sum((heights * widths)[any_arr], dtype=np.int64))
    logical_pixels = max(1, int(z_dim) * int(h) * int(w))
    return int(np.count_nonzero(any_arr)), int(bbox_pixels), float(bbox_pixels) / float(logical_pixels)

@runtime_telemetry_phase('topology.slice_metadata_scan')
def scan_binary_volume_slice_metadata(
    volume: np.ndarray,
    *,
    workers: int,
    source: str,
) -> BinaryVolumeSliceMetadata:
    """Compute exact per-slice support as a correctness-preserving metadata fallback."""
    arr = np.asarray(volume)
    if int(arr.ndim) != 3:
        raise ValueError(f'binary slice metadata scan requires 3D input, got {arr.shape}')
    z_dim, _h, _w = (int(v) for v in arr.shape)
    slice_any = np.zeros((int(z_dim),), dtype=bool)
    slice_bboxes = np.zeros((int(z_dim), 4), dtype=np.int64)
    scan_workers = choose_slice_parallel_workers(
        min(max(1, int(workers)), max(1, int(_cpu_count()))), int(z_dim),
    )

    def _scan_z(z: int) -> None:
        z_i = int(z)
        plane = np.ascontiguousarray(np.asarray(arr[z_i], dtype=np.uint8))
        x0, y0, bw, bh = (int(v) for v in cv2.boundingRect(plane))
        if int(bw) <= 0 or int(bh) <= 0:
            return
        slice_any[z_i] = True
        slice_bboxes[z_i] = np.asarray(
            (int(y0), int(y0 + bh), int(x0), int(x0 + bw)), dtype=np.int64,
        )

    parallel_for_indices_chunked(
        int(z_dim),
        _scan_z,
        max_workers=int(scan_workers),
        desc='v16.1.7 topology slice-metadata scan',
        show_progress=False,
        target_chunks_per_worker=2,
    )
    return register_binary_volume_slice_metadata(
        volume,
        slice_any,
        slice_bboxes,
        source=str(source),
        exact=True,
    )

def topology_sparse_cpu_labeling_enabled() -> bool:
    """Prefer crop-bounded CPU CCL over a dense parent-process CUDA first use."""
    return _env_flag('YOLO_TTA_TOPOLOGY_SPARSE_CPU_LABELING', True)

def topology_sparse_cpu_max_coverage() -> float:
    """Maximum bbox coverage at which crop-bounded CPU labeling is selected."""
    return max(
        0.0,
        min(1.0, _env_float('YOLO_TTA_TOPOLOGY_SPARSE_CPU_MAX_COVERAGE', 0.50)),
    )

def gpu_slice_labeling_enabled() -> bool:
    """Run per-slice 2D component labeling on the GPU where CUDA is available."""
    return _env_flag('YOLO_TTA_GPU_SLICE_LABELING', True)

def gpu_slice_labeling_in_children_enabled() -> bool:
    """Allow GPU labeling inside dedicated interpolation child processes.

 Off by default: each child would have to create its own CUDA context (init latency +
 VRAM per child, times up to 6 children) just for labeling. The aux GPU-worker
 interpolation hosts already own a warm context and take the GPU path
 automatically; the main process (keep_objects) does too."""
    return _env_flag('YOLO_TTA_GPU_SLICE_LABELING_IN_CHILDREN', False)

def gpu_slice_labeling_pairs_enabled() -> bool:
    """Emit adjacent-slice local-id pair codes while labels are on CUDA."""
    return _env_flag('YOLO_TTA_GPU_SLICE_LABELING_PAIRS', True)

def topology_slab_slices() -> int:
    """Z-depth of independently resolved 3D topology slabs."""
    return max(4, _env_int('YOLO_TTA_TOPOLOGY_SLAB_SLICES', 32))

def topology_slab_ranges(z_dim: int) -> List[Tuple[int, int]]:
    depth = max(1, int(topology_slab_slices()))
    return [
        (int(z0), min(int(z_dim), int(z0) + int(depth)))
        for z0 in range(0, max(0, int(z_dim)), int(depth))
    ]

def topology_slab_workers(requested_workers: int, slab_count: int) -> int:
    """Bound concurrent local union-find slabs; Numba releases the GIL for pair batches."""
    # The former fixed cap of eight consumed only ~5% of a 160-vCPU allocation during the
    # post-inference keep_objects tail. Use up to half the visible logical CPUs (64 max) while
    # retaining an explicit environment override and one worker per available slab.
    default_workers = min(
        64,
        max(1, int(_cpu_count()) // 2),
        max(1, int(requested_workers)),
        max(1, int(slab_count)),
    )
    resolved = max(1, _env_int('YOLO_TTA_TOPOLOGY_SLAB_WORKERS', int(default_workers)))
    if _numba_union_find_batch_kernel is None or not compiled_topology_kernels_enabled():
        # Python union-find is GIL-bound; several slab threads would only add memory pressure.
        return 1
    return max(1, min(int(resolved), max(1, int(slab_count))))

def _round_robin_topology_gpu_blocks(
    z_dim: int,
    block_slices: int,
    device_indices: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """Assign deterministic contiguous slab/block ranges across all admitted GPUs."""
    devices = tuple(dict.fromkeys(int(v) for v in device_indices))
    if int(z_dim) <= 0 or not devices:
        return []
    depth = max(1, int(block_slices))
    return [
        (int(z0), min(int(z_dim), int(z0) + int(depth)), int(devices[idx % len(devices)]))
        for idx, z0 in enumerate(range(0, int(z_dim), int(depth)))
    ]

_GPU_SLICE_LABELING_CONFIGURED_DEVICES: Optional[Tuple[int, ...]] = None

def configure_gpu_slice_labeling_devices(devices: Sequence[str]) -> None:
    """Remember the CUDA subset selected by ``--device`` for main-process work.

 CUDA worker processes are pinned to one visible device before importing/initializing
 torch, so their logical device count remains authoritative. This hint is primarily for
 the unpinned parent, where every SLURM-visible GPU may otherwise be discoverable even when
 the command selected a smaller subset."""
    configured: List[int] = []
    for device in devices:
        token = str(device).strip().lower()
        if not token.startswith('cuda:'):
            continue
        try:
            idx = int(token.split(':', 1)[1])
        except Exception:
            continue
        if idx >= 0 and idx not in configured:
            configured.append(idx)
    global _GPU_SLICE_LABELING_CONFIGURED_DEVICES
    _GPU_SLICE_LABELING_CONFIGURED_DEVICES = tuple(configured) if configured else None


def gpu_slice_labeling_configured_devices() -> Optional[Tuple[int, ...]]:
    """Return the configured process-local devices for slice labeling."""

    return _GPU_SLICE_LABELING_CONFIGURED_DEVICES

@dataclass
class _GpuSliceLabelBlockResult:
    z0: int
    z1: int
    device_index: int
    first_plane: Optional[object]
    last_plane: Optional[object]
    internal_pair_codes: Dict[int, np.ndarray]

def _gpu_adjacent_local_pair_codes_device(
    cp: object,
    prev_local: object,
    curr_local: object,
    prev_bbox: Sequence[int],
    curr_bbox: Sequence[int],
    h: int,
    w: int,
) -> object:
    """Device equivalent of ``_adjacent_gid_pair_codes`` for local ids.

 The same one-pixel-expanded bbox-intersection window and the same nine XY shifts are
 used as the CPU stage-C path. Local ids are packed into uint64 (previous in the high
 word, current in the low word); global slice offsets are deliberately applied only after
 all blocks finish and the complete component-count prefix sum is known."""
    py0, py1, px0, px1 = (int(v) for v in prev_bbox)
    cy0, cy1, cx0, cx1 = (int(v) for v in curr_bbox)
    ry0 = max(0, max(py0, cy0) - 1)
    ry1 = min(int(h), min(py1, cy1) + 1)
    rx0 = max(0, max(px0, cx0) - 1)
    rx1 = min(int(w), min(px1, cx1) + 1)
    if ry0 >= ry1 or rx0 >= rx1:
        return cp.empty((0,), dtype=cp.uint64)

    prev_crop = prev_local[ry0:ry1, rx0:rx1]
    curr_crop = curr_local[ry0:ry1, rx0:rx1]
    ch = int(ry1 - ry0)
    cw = int(rx1 - rx0)
    code_parts: List[object] = []
    for dy, dx in _adjacent_xy_offsets_for_3d_connectivity(26):
        dy_i = int(dy)
        dx_i = int(dx)
        py0_i = max(0, -dy_i)
        py1_i = min(ch, ch - dy_i)
        cy0_i = max(0, dy_i)
        cy1_i = min(ch, ch + dy_i)
        px0_i = max(0, -dx_i)
        px1_i = min(cw, cw - dx_i)
        cx0_i = max(0, dx_i)
        cx1_i = min(cw, cw + dx_i)
        if py0_i >= py1_i or px0_i >= px1_i:
            continue
        a = prev_crop[py0_i:py1_i, px0_i:px1_i]
        b = curr_crop[cy0_i:cy1_i, cx0_i:cx1_i]
        overlap = (a > 0) & (b > 0)
        a_vals = a[overlap].astype(cp.uint64, copy=False)
        b_vals = b[overlap].astype(cp.uint64, copy=False)
        code_parts.append((a_vals << cp.uint64(32)) | b_vals)
    if not code_parts:
        return cp.empty((0,), dtype=cp.uint64)
    # Exactly one unique reduction per adjacent slice pair. The caller concatenates every
    # pair result in the block for one small D2H transfer rather than one transfer per z.
    return cp.unique(cp.concatenate(code_parts))

_GPU_SLICE_LABELING_ANNOUNCED = False

def _try_label_slices_stage_a_gpu(
    mask_mm: np.ndarray,
    labels_store: object,
    component_counts: np.ndarray,
    slice_bboxes: np.ndarray,
    slice_areas: Optional[List[Optional[np.ndarray]]],
    known_slice_any: Optional[np.ndarray] = None,
    preferred_block_slices: Optional[int] = None,
) -> Tuple[bool, Optional[List[Optional[np.ndarray]]]]:
    """Label independent slice blocks on the selected GPUs.
    
    Produces slice-local labels, area/bbox metadata, and optional adjacent-slice pair codes without a duplicate CPU pass."""
    # Local import keeps the package dependency graph acyclic.
    from .backprojection import (
        _announce_main_gpu_stage_skip_once,
        _try_acquire_specific_main_process_gpu_stage,
    )

    if not gpu_slice_labeling_enabled():
        return False, None
    try:
        import torch  # type: ignore
        if not bool(torch.cuda.is_available()):
            return False, None
        if (
            interpolation_process_worker_active()
            and not gpu_slice_labeling_in_children_enabled()
            and not bool(torch.cuda.is_initialized())
        ):
            # Dedicated interpolation children refuse to CREATE a CUDA context just for
            # labeling (init latency + VRAM, times up to 6 children). The aux GPU-worker
            # hosts enter through the same entry point but already own a live
            # context from inference — they proceed.
            return False, None
        cp_mod = _try_import_cupy_ndimage()
        if cp_mod is None:
            return False, None
        cp, cpx_ndi = cp_mod
    except Exception:
        return False, None

    z_dim, h, w = (int(x) for x in mask_mm.shape)
    if z_dim <= 0 or h < 2 or w < 2:
        return False, None

    # Device leases cover admission, execution, reference teardown and allocator trimming.
    # Returned edge planes and completed futures can retain CuPy blocks after the kernels
    # finish, so those references are explicitly dropped before either allocator is trimmed.
    device_indices: List[int] = []
    stage_leases: Dict[int, _MainProcessGpuStageLease] = {}
    pending: Dict[int, Future] = {}
    previous_result = result = None
    prev_cp = curr_cp = copied_prev = boundary_dev = None

    def _free_admitted_device_pools() -> None:
        # Local import keeps the package dependency graph acyclic.
        from .backprojection import _trim_main_process_cuda_device

        cleanup_indices = sorted({int(v) for v in device_indices} or {int(v) for v in stage_leases})
        for cleanup_dev_idx in cleanup_indices:
            try:
                _trim_main_process_cuda_device(
                    torch,
                    torch.device(f'cuda:{int(cleanup_dev_idx)}'),
                    cupy_module=cp,
                    desc='GPU per-slice labeling cleanup',
                )
            except Exception:
                pass

    def _release_stage_leases() -> None:
        for cleanup_dev_idx, lease in list(stage_leases.items()):
            try:
                lease.release()
            except Exception:
                pass
            stage_leases.pop(int(cleanup_dev_idx), None)

    try:
        visible_count = max(0, int(torch.cuda.device_count()))
        if visible_count <= 0:
            return False, None
        configured = _GPU_SLICE_LABELING_CONFIGURED_DEVICES
        candidate_indices = [
            int(idx) for idx in (configured if configured is not None else tuple(range(visible_count)))
            if 0 <= int(idx) < int(visible_count)
        ]
        if not candidate_indices:
            candidate_indices = list(range(visible_count))

        leased_candidates: List[int] = []
        for dev_idx in candidate_indices:
            lease = _try_acquire_specific_main_process_gpu_stage(
                torch, int(dev_idx), 'GPU per-slice labeling',
            )
            if lease is None:
                continue
            stage_leases[int(dev_idx)] = lease
            leased_candidates.append(int(dev_idx))
        candidate_indices = leased_candidates
        if not candidate_indices:
            _announce_main_gpu_stage_skip_once(
                'gpu-slice-labeling-inference-busy',
                'GPU per-slice labeling skipped while all configured GPUs have active/queued '
                'inference or another output-stage lease; using CPU slice labeling.',
            )
            return False, None

        requested_device_cap = max(
            1,
            _env_int('YOLO_TTA_GPU_SLICE_LABELING_DEVICES', len(candidate_indices)),
        )
        requested_device_cap = min(int(requested_device_cap), len(candidate_indices))
        plane = int(h) * int(w)
        # Adaptive z-block: u8 mask + int32 labels + int32 relabel + label workspace,
        # ~17 B/voxel, plus ~1 GiB headroom for the coordinate transients.
        requested_block = max(4, _env_int('YOLO_TTA_GPU_SLICE_LABELING_BLOCK', int(preferred_block_slices or topology_slab_slices())))
        device_free: List[Tuple[int, int]] = []
        for dev_idx in candidate_indices:
            try:
                device = torch.device(f'cuda:{int(dev_idx)}')
                free_bytes, _total = torch.cuda.mem_get_info(device)
                device_free.append((int(dev_idx), int(free_bytes)))
            except Exception:
                continue
        if not device_free:
            return False, None
        if requested_device_cap == 1:
            # When explicitly restricted to one device, use the freest candidate.
            selected_free = [max(device_free, key=lambda item: int(item[1]))]
        else:
            # Use the freest configured devices, then restore logical-index order so block
            # assignment is stable and visibly round-robin in logs/profilers.
            selected_free = sorted(device_free, key=lambda item: int(item[1]), reverse=True)[:requested_device_cap]
            selected_free.sort(key=lambda item: int(item[0]))

        admitted: List[Tuple[int, int]] = []
        for dev_idx, free_bytes in selected_free:
            candidate_block = int(requested_block)
            while candidate_block > 4 and int(free_bytes) < candidate_block * plane * 17 + GIB:
                candidate_block = max(4, candidate_block // 2)
            if int(free_bytes) >= candidate_block * plane * 17 + GIB:
                admitted.append((int(dev_idx), int(candidate_block)))
        if not admitted:
            return False, None
        device_indices = [int(item[0]) for item in admitted]
        # Fixed block boundaries make round-robin dispatch and the retained-plane boundary
        # contract deterministic. Use the smallest admitted block across selected devices.
        block = min(int(item[1]) for item in admitted)
        provisional_block_count = int(math.ceil(float(z_dim) / float(max(1, int(block)))))
        device_indices = device_indices[:max(1, min(len(device_indices), int(provisional_block_count)))]
        selected_set = {int(v) for v in device_indices}
        for unused_idx, unused_lease in list(stage_leases.items()):
            if int(unused_idx) in selected_set:
                continue
            unused_lease.release()
            stage_leases.pop(int(unused_idx), None)
        block_assignments = _round_robin_topology_gpu_blocks(
            int(z_dim), int(block), device_indices,
        )
        pairs_on_device = bool(gpu_slice_labeling_pairs_enabled())

        global _GPU_SLICE_LABELING_ANNOUNCED
        if not _GPU_SLICE_LABELING_ANNOUNCED:
            _GPU_SLICE_LABELING_ANNOUNCED = True
            print(
                f'GPU per-slice labeling active on '
                f'{", ".join(f"cuda:{idx}" for idx in device_indices)} '
                f'(v16.0.2 slab-aligned D3/D7, block={int(block)}, blocks={len(block_assignments)}, device-pairs={pairs_on_device}; '
                'YOLO_TTA_GPU_SLICE_LABELING=0 disables, '
                'YOLO_TTA_GPU_SLICE_LABELING_DEVICES=1 forces one device).'
            )

        empty_codes = lambda: np.zeros((0,), dtype=np.uint64)

        def _label_block(z0: int, z1: int, dev_idx: int) -> _GpuSliceLabelBlockResult:
            device = torch.device(f'cuda:{int(dev_idx)}')
            with torch.cuda.device(device), cp.cuda.Device(int(dev_idx)):
                structure = cp.zeros((3, 3, 3), dtype=cp.bool_)
                structure[1, :, :] = True  # in-plane 8-connectivity; z disconnected
                bs = int(z1 - z0)
                if known_slice_any is not None and not bool(np.any(known_slice_any[z0:z1])):
                    component_counts[z0:z1] = np.uint32(0)
                    internal = (
                        {int(z): empty_codes() for z in range(int(z0) + 1, int(z1))}
                        if pairs_on_device else {}
                    )
                    return _GpuSliceLabelBlockResult(
                        int(z0), int(z1), int(dev_idx), None, None, internal,
                    )
                host_block = np.ascontiguousarray(np.asarray(mask_mm[z0:z1]))
                mask_t = torch.from_numpy(host_block).to(device=device, non_blocking=False)
                fg_cp = cp.asarray(mask_t) > 0
                labels_cp, num = cpx_ndi.label(fg_cp, structure=structure)
                del fg_cp, mask_t, host_block
                num_i = int(num)
                if num_i <= 0:
                    component_counts[z0:z1] = np.uint32(0)
                    internal = (
                        {int(z): empty_codes() for z in range(int(z0) + 1, int(z1))}
                        if pairs_on_device else {}
                    )
                    return _GpuSliceLabelBlockResult(
                        int(z0), int(z1), int(dev_idx), None, None, internal,
                    )
                try:
                    labels_t = torch.from_dlpack(labels_cp)
                except Exception:
                    labels_t = torch.utils.dlpack.from_dlpack(labels_cp.toDlpack())
                flat = labels_t.reshape(-1)
                nz = flat.nonzero().squeeze(1)
                vals = flat.index_select(0, nz).to(torch.int64)
                zz = torch.div(nz, plane, rounding_mode='floor')
                rem = nz - zz * plane
                yy = torch.div(rem, int(w), rounding_mode='floor')
                xx = rem - yy * int(w)
                big = torch.iinfo(torch.int64).max
                comp_area = torch.bincount(vals, minlength=num_i + 1)
                comp_z = torch.full((num_i + 1,), big, dtype=torch.int64, device=device)
                comp_z.scatter_reduce_(0, vals, zz, reduce='amin', include_self=True)
                comp_y0 = torch.full((num_i + 1,), big, dtype=torch.int64, device=device)
                comp_y0.scatter_reduce_(0, vals, yy, reduce='amin', include_self=True)
                comp_y1 = torch.full((num_i + 1,), -1, dtype=torch.int64, device=device)
                comp_y1.scatter_reduce_(0, vals, yy, reduce='amax', include_self=True)
                comp_x0 = torch.full((num_i + 1,), big, dtype=torch.int64, device=device)
                comp_x0.scatter_reduce_(0, vals, xx, reduce='amin', include_self=True)
                comp_x1 = torch.full((num_i + 1,), -1, dtype=torch.int64, device=device)
                comp_x1.scatter_reduce_(0, vals, xx, reduce='amax', include_self=True)
                del nz, vals, zz, rem, yy, xx
                # Per-slice-contiguous local ids: rank components within their slice after a
                # stable (slice, component-id) ordering.
                comp_ids = torch.arange(1, num_i + 1, device=device, dtype=torch.int64)
                order = torch.argsort(comp_z[1:] * (num_i + 2) + comp_ids)
                sorted_z = comp_z[1:].index_select(0, order)
                counts_t = torch.bincount(sorted_z, minlength=bs)
                starts = torch.zeros((bs,), dtype=torch.int64, device=device)
                if bs > 1:
                    starts[1:] = torch.cumsum(counts_t, dim=0)[:-1]
                local_rank = torch.arange(num_i, device=device, dtype=torch.int64) - starts.index_select(0, sorted_z)
                lut = torch.zeros((num_i + 1,), dtype=torch.int32, device=device)
                lut.index_copy_(0, order + 1, (local_rank + 1).to(torch.int32))
                # Relabel in cupy (int32 fancy indexing avoids a full int64 index copy).
                relabeled_cp = cp.take(cp.asarray(lut), labels_cp)
                del labels_cp, labels_t, flat, lut

                counts_np = counts_t.cpu().numpy()
                starts_np = starts.cpu().numpy()
                y0s = comp_y0[1:].index_select(0, order).cpu().numpy()
                y1s = comp_y1[1:].index_select(0, order).cpu().numpy()
                x0s = comp_x0[1:].index_select(0, order).cpu().numpy()
                x1s = comp_x1[1:].index_select(0, order).cpu().numpy()
                areas_sorted = comp_area[1:].index_select(0, order).cpu().numpy()
                del comp_area, comp_z, comp_y0, comp_y1, comp_x0, comp_x1, order, sorted_z, counts_t, starts, local_rank, comp_ids

                store_dtype = np.dtype(labels_store.dtype)
                block_bboxes = np.zeros((int(bs), 4), dtype=np.int64)
                for zi in range(bs):
                    z = int(z0 + zi)
                    k = int(counts_np[zi])
                    _check_local_label_store_capacity(k, store_dtype, z=z)
                    component_counts[z] = np.uint32(k)
                    if k <= 0:
                        continue
                    seg0 = int(starts_np[zi])
                    seg1 = seg0 + k
                    by0 = int(y0s[seg0:seg1].min())
                    by1 = int(y1s[seg0:seg1].max()) + 1
                    bx0 = int(x0s[seg0:seg1].min())
                    bx1 = int(x1s[seg0:seg1].max()) + 1
                    block_bboxes[zi, :] = (by0, by1, bx0, bx1)
                    slice_bboxes[z, 0] = by0
                    slice_bboxes[z, 1] = by1
                    slice_bboxes[z, 2] = bx0
                    slice_bboxes[z, 3] = bx1
                    if slice_areas is not None:
                        slice_areas[z] = areas_sorted[seg0:seg1].astype(np.int64, copy=True)

                internal_pair_codes: Dict[int, np.ndarray] = {}
                if pairs_on_device:
                    pair_device_arrays: List[object] = []
                    pair_zs: List[int] = []
                    for zi in range(1, int(bs)):
                        z = int(z0 + zi)
                        pair_zs.append(z)
                        if int(counts_np[zi - 1]) <= 0 or int(counts_np[zi]) <= 0:
                            pair_device_arrays.append(cp.empty((0,), dtype=cp.uint64))
                        else:
                            pair_device_arrays.append(_gpu_adjacent_local_pair_codes_device(
                                cp,
                                relabeled_cp[zi - 1],
                                relabeled_cp[zi],
                                block_bboxes[zi - 1],
                                block_bboxes[zi],
                                int(h),
                                int(w),
                            ))
                    pair_lengths = [int(arr.size) for arr in pair_device_arrays]
                    pair_total = int(sum(pair_lengths))
                    if pair_total > 0:
                        pair_flat = cp.asnumpy(cp.concatenate(pair_device_arrays))
                    else:
                        pair_flat = empty_codes()
                    cursor = 0
                    for z, length in zip(pair_zs, pair_lengths):
                        internal_pair_codes[int(z)] = np.ascontiguousarray(
                            pair_flat[cursor:cursor + int(length)], dtype=np.uint64,
                        )
                        cursor += int(length)

                # Retain only the two block-edge planes on their owning device. Internal
                # pairs have already consumed the full block; these small references let
                # the coordinator form the one cross-block pair without rereading labels.
                first_plane = (
                    relabeled_cp[0].copy()
                    if pairs_on_device and int(counts_np[0]) > 0 else None
                )
                last_plane = (
                    relabeled_cp[-1].copy()
                    if pairs_on_device and int(counts_np[-1]) > 0 else None
                )
                # Existing transfer: labels must remain in the local-id store for later
                # component-table/SDF consumers. Pair codes above add only one tiny D2H block.
                # packs the tight device crops (already narrowed to the store
                # dtype) and transfers them once. The normal sparse path therefore never
                # creates the former dense int32 host block (~1 GiB at 64x2048^2).
                if isinstance(labels_store, SparseSliceLabelStore):
                    crop_specs: List[Tuple[int, int, int, int, int, int]] = []
                    crop_device_parts: List[object] = []
                    for zi in range(bs):
                        z = int(z0 + zi)
                        if int(counts_np[zi]) <= 0:
                            continue
                        by0, by1, bx0, bx1 = (int(v) for v in block_bboxes[zi])
                        size = int((by1 - by0) * (bx1 - bx0))
                        crop_specs.append((z, by0, by1, bx0, bx1, size))
                        crop_device_parts.append(
                            relabeled_cp[zi, by0:by1, bx0:bx1]
                            .astype(cp.dtype(store_dtype), copy=False)
                            .reshape(-1)
                        )
                    if crop_device_parts:
                        packed_device = (
                            crop_device_parts[0]
                            if len(crop_device_parts) == 1
                            else cp.concatenate(crop_device_parts)
                        )
                        packed_host = cp.asnumpy(packed_device)
                        del packed_device
                    else:
                        packed_host = np.empty((0,), dtype=store_dtype)
                    del crop_device_parts, relabeled_cp
                    cursor = 0
                    for z, by0, by1, bx0, bx1, size in crop_specs:
                        labels_store.write_crop(
                            z, by0, by1, bx0, bx1,
                            packed_host[cursor:cursor + size].reshape(
                                (by1 - by0, bx1 - bx0),
                            ),
                        )
                        cursor += int(size)
                    del packed_host, crop_specs
                else:
                    relabeled_np = cp.asnumpy(relabeled_cp)
                    del relabeled_cp
                    for zi in range(bs):
                        z = int(z0 + zi)
                        if int(counts_np[zi]) <= 0:
                            continue
                        by0, by1, bx0, bx1 = (int(v) for v in block_bboxes[zi])
                        np.copyto(
                            np.asarray(labels_store[z, by0:by1, bx0:bx1]),
                            relabeled_np[zi, by0:by1, bx0:bx1].astype(
                                store_dtype, copy=False,
                            ),
                        )
                return _GpuSliceLabelBlockResult(
                    int(z0), int(z1), int(dev_idx), first_plane, last_plane,
                    internal_pair_codes,
                )

        precomputed_pairs: Optional[List[Optional[np.ndarray]]] = (
            [None] * int(z_dim) if pairs_on_device else None
        )
        # Keep no more than one block outstanding per GPU. Block i always returns to lane
        # i%N, so independent z blocks are genuinely round-robin without concurrent calls
        # fighting over one device's allocator/default stream.
        lane_count = max(1, len(device_indices))
        previous_result = None
        with ThreadPoolExecutor(max_workers=lane_count, thread_name_prefix='slice-label-gpu') as executor:
            pending = {}
            for block_idx in range(min(lane_count, len(block_assignments))):
                z0, z1, dev_idx = block_assignments[block_idx]
                pending[block_idx] = executor.submit(_label_block, z0, z1, dev_idx)

            for block_idx in range(len(block_assignments)):
                result = pending.pop(block_idx).result()
                if precomputed_pairs is not None:
                    for z, codes in result.internal_pair_codes.items():
                        precomputed_pairs[int(z)] = np.ascontiguousarray(codes, dtype=np.uint64)
                    if previous_result is not None:
                        boundary_z = int(result.z0)
                        if (
                            int(component_counts[boundary_z - 1]) <= 0
                            or int(component_counts[boundary_z]) <= 0
                        ):
                            boundary_codes = empty_codes()
                        else:
                            if previous_result.last_plane is None or result.first_plane is None:
                                raise RuntimeError(
                                    f'missing retained GPU label plane for block boundary z={boundary_z}'
                                )
                            # boundary pair: use the retained previous-block device edge
                            # and current device edge, never re-read either label slice from
                            # the host store. Peer-copy the one previous plane when round-robin
                            # placed the blocks on different GPUs; only non-P2P systems use a
                            # retained-plane host bounce as the correctness fallback.
                            with (
                                torch.cuda.device(torch.device(f'cuda:{int(result.device_index)}')),
                                cp.cuda.Device(int(result.device_index)),
                            ):
                                curr_cp = result.first_plane
                                prev_cp = previous_result.last_plane
                                copied_prev = None
                                if int(previous_result.device_index) != int(result.device_index):
                                    try:
                                        if not bool(cp.cuda.runtime.deviceCanAccessPeer(
                                            int(result.device_index),
                                            int(previous_result.device_index),
                                        )):
                                            raise RuntimeError('CUDA peer access unavailable')
                                        copied_prev = cp.empty(
                                            tuple(int(v) for v in prev_cp.shape), dtype=prev_cp.dtype,
                                        )
                                        cp.cuda.runtime.memcpyPeerAsync(
                                            int(copied_prev.data.ptr),
                                            int(result.device_index),
                                            int(prev_cp.data.ptr),
                                            int(previous_result.device_index),
                                            int(prev_cp.nbytes),
                                            int(cp.cuda.get_current_stream().ptr),
                                        )
                                        prev_cp = copied_prev
                                    except Exception:
                                        with cp.cuda.Device(int(previous_result.device_index)):
                                            prev_host = cp.asnumpy(previous_result.last_plane)
                                        with cp.cuda.Device(int(result.device_index)):
                                            copied_prev = cp.asarray(prev_host)
                                        prev_cp = copied_prev
                                boundary_dev = _gpu_adjacent_local_pair_codes_device(
                                    cp,
                                    prev_cp,
                                    curr_cp,
                                    slice_bboxes[boundary_z - 1],
                                    slice_bboxes[boundary_z],
                                    int(h),
                                    int(w),
                                )
                                boundary_codes = np.ascontiguousarray(
                                    cp.asnumpy(boundary_dev), dtype=np.uint64,
                                )
                                del prev_cp, curr_cp, copied_prev, boundary_dev
                        precomputed_pairs[boundary_z] = boundary_codes
                previous_result = result

                next_idx = int(block_idx + lane_count)
                if next_idx < len(block_assignments):
                    z0, z1, dev_idx = block_assignments[next_idx]
                    pending[next_idx] = executor.submit(_label_block, z0, z1, dev_idx)

        return True, precomputed_pairs
    except _LocalLabelCapacityError:
        # The result is data-dependent, not a GPU backend failure. Re-raise before the
        # generic fallback's full-store zero sweep and duplicate CPU labeling pass.
        raise
    except Exception as exc:
        print(f'Warning: GPU per-slice labeling failed ({exc}); falling back to the CPU labeling stage.')
        # Zero everything the GPU pass may have written so the CPU stage starts from a
        # clean store (the CPU fallback only rewrites each nonempty slice's bbox window).
        for z in range(0, int(z_dim)):
            try:
                labels_store[int(z), :, :] = 0
            except Exception:
                break
        try:
            component_counts[:] = np.uint32(0)
        except Exception:
            pass
        try:
            slice_bboxes[:] = np.int64(0)
        except Exception:
            pass
        if slice_areas is not None:
            for z in range(len(slice_areas)):
                slice_areas[z] = None
        return False, None
    finally:
        # Futures and block-boundary results own the retained device edge planes. Clear
        # them before freeing CuPy/Torch pools so the blocks become driver-reclaimable now.
        pending.clear()
        previous_result = result = None
        prev_cp = curr_cp = copied_prev = boundary_dev = None
        gc.collect()
        _free_admitted_device_pools()
        _release_stage_leases()

def label_foreground_volume_streaming(
    mask_mm: np.ndarray,
    work_prefix: Path,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    wrap_axis: bool = False,
    workers: int = 1,
    compact_relabel: bool = True,
    component_stats_out: Optional[Dict[str, object]] = None,
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    sparse_local_labels: bool = False,
    prefer_crop_bounded_cpu_labeling: bool = False,
) -> Tuple[object, int, List[Path]]:
    """Resolve 3D foreground topology from parallel per-slice labels and slab-local unions.
    
    The caller may request compact global labels or retain slice-local ids with lookup tables for sparse downstream work."""
    z_dim, h, w = (int(v) for v in mask_mm.shape)
    if (known_slice_any is None) != (known_slice_bboxes is None):
        raise ValueError('known_slice_any and known_slice_bboxes must be supplied together')
    if known_slice_any is not None and known_slice_bboxes is not None:
        known_slice_any = np.ascontiguousarray(
            np.asarray(known_slice_any, dtype=bool).reshape((int(z_dim),)),
        )
        known_slice_bboxes = np.ascontiguousarray(
            np.asarray(known_slice_bboxes, dtype=np.int64).reshape((int(z_dim), 4)),
        )
    # when compact relabel is skipped, this raster never holds
    # canonical/global ids — only per-slice local ids 1..k. Halve the dominant
    # workspace with uint16 while retaining uint32 for the compact/global path.
    label_dtype = _local_label_store_dtype(bool(compact_relabel))
    estimated_bytes = estimate_voidfill_workspace_bytes((z_dim, h, w), dtype=label_dtype)
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(estimated_bytes, reserve_bytes=reserve_bytes)

    work_prefix.parent.mkdir(parents=True, exist_ok=True)
    provisional_path = work_prefix.with_suffix(
        '.fg_labels.u16.dat' if label_dtype == np.dtype(np.uint16) else '.fg_labels.u32.dat'
    )
    label_paths: List[Path] = []

    sparse_enabled = bool(
        sparse_local_labels
        and not bool(compact_relabel)
        and interpolation_sparse_labels_enabled()
    )
    if bool(sparse_local_labels) and bool(compact_relabel):
        raise ValueError('sparse_local_labels requires compact_relabel=False')

    budget = workspace_budget_summary(estimated_bytes, reserve_bytes=reserve_bytes)
    if sparse_enabled:
        logical_gib = array_nbytes((z_dim, h, w), label_dtype) / GIB
        print(
            f'3D topology label workspace: sparse per-slice arena '
            f'(dense logical size {logical_gib:.1f} GiB, dtype={label_dtype.name})'
        )
        labels_store: object = SparseSliceLabelStore((z_dim, h, w), label_dtype)
    elif use_in_memory:
        print(f"3D topology label workspace: in-memory ({budget}, dtype={label_dtype.name})")
        labels_store = np.zeros((z_dim, h, w), dtype=label_dtype)
    else:
        print(
            f"3D topology label workspace: disk-backed ({budget}, dtype={label_dtype.name}) "
            f"-> {work_prefix.parent}"
        )
        labels_store = np.memmap(provisional_path, dtype=label_dtype, mode='w+', shape=(z_dim, h, w))
        label_paths = [provisional_path]
    if not sparse_enabled:
        numa_interleave_memory(labels_store, desc='3D topology label store')  #

    if int(z_dim) <= 0:
        flush_array(labels_store)
        return labels_store, 0, label_paths

    worker_count = choose_slice_parallel_workers(int(workers), int(z_dim))
    label_workers = choose_slice_parallel_workers(
        _env_int('YOLO_TTA_INTERPOLATION_LABEL_WORKERS', worker_count),
        int(z_dim),
    )
    compact_workers = choose_slice_parallel_workers(
        _env_int('YOLO_TTA_INTERPOLATION_COMPACT_WORKERS', worker_count),
        int(z_dim),
    )
    pair_workers = choose_slice_parallel_workers(
        _env_int('YOLO_TTA_INTERPOLATION_PAIR_WORKERS', worker_count),
        max(1, int(z_dim) - 1),
    )

    component_counts = np.zeros((int(z_dim),), dtype=np.uint32)
    # per-slice foreground bboxes (y0, y1, x0, x1; exclusive stops) captured
    # during labeling so cross-slice pair extraction only reads the overlapping bbox windows.
    slice_bboxes = np.zeros((int(z_dim), 4), dtype=np.int64)
    collect_stats = component_stats_out is not None
    slice_areas: List[Optional[np.ndarray]] = [None] * int(z_dim) if collect_stats else []

    def _label_slice_local(z: int) -> None:
        z_i = int(z)
        # device-union metadata skips empty slices without touching their
        # pages and restricts the CC call (and the store write) to the known bbox window.
        if known_slice_any is not None and not bool(known_slice_any[z_i]):
            component_counts[z_i] = np.uint32(0)
            return
        ky0 = kx0 = 0
        if known_slice_bboxes is not None:
            ky0, ky1, kx0, kx1 = (int(v) for v in known_slice_bboxes[z_i])
            if ky1 <= ky0 or kx1 <= kx0:
                component_counts[z_i] = np.uint32(0)
                return
            src = np.asarray(mask_mm[z_i, ky0:ky1, kx0:kx1])
        else:
            src = np.asarray(mask_mm[z_i])
        # OpenCV treats ANY nonzero pixel as foreground, so the raw uint8
        # slice labels identically to the old (mask>0) cast — the full-slice compare+copy
        # per slice is deleted (contiguity copy only for bbox crops).
        fg = np.ascontiguousarray(src)
        metadata_guarantees_nonempty = bool(
            known_slice_any is not None and bool(known_slice_any[z_i])
        )
        if fg.size == 0 or (
            not bool(metadata_guarantees_nonempty) and not bool(np.any(fg))
        ):
            component_counts[z_i] = np.uint32(0)
            return

        if collect_stats:
            # per-component areas (and the slice bbox) fall out of the stats —
            # this deletes the caller's full-volume bincount pass.
            num_labels, labels2d, cc_stats, _cents = cv2.connectedComponentsWithStats(
                fg, connectivity=8, ltype=cv2.CV_32S,
            )
            if int(num_labels) <= 1:
                component_counts[z_i] = np.uint32(0)
                return
            tops = cc_stats[1:, cv2.CC_STAT_TOP]
            lefts = cc_stats[1:, cv2.CC_STAT_LEFT]
            slice_bboxes[z_i, 0] = int(tops.min()) + ky0
            slice_bboxes[z_i, 1] = int((tops + cc_stats[1:, cv2.CC_STAT_HEIGHT]).max()) + ky0
            slice_bboxes[z_i, 2] = int(lefts.min()) + kx0
            slice_bboxes[z_i, 3] = int((lefts + cc_stats[1:, cv2.CC_STAT_WIDTH]).max()) + kx0
            slice_areas[z_i] = cc_stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=True)
        else:
            num_labels, labels2d = _cv2_connected_components(fg, connectivity=8)
            if int(num_labels) <= 1:
                component_counts[z_i] = np.uint32(0)
                return

            rows_any = np.flatnonzero(fg.any(axis=1))
            cols_any = np.flatnonzero(fg.any(axis=0))
            slice_bboxes[z_i, 0] = int(rows_any[0]) + ky0
            slice_bboxes[z_i, 1] = int(rows_any[-1]) + 1 + ky0
            slice_bboxes[z_i, 2] = int(cols_any[0]) + kx0
            slice_bboxes[z_i, 3] = int(cols_any[-1]) + 1 + kx0
        local_count = int(num_labels) - 1
        _check_local_label_store_capacity(local_count, label_dtype, z=z_i)
        if sparse_enabled:
            by0, by1, bx0, bx1 = (int(v) for v in slice_bboxes[z_i])
            labels_store.write_crop(  # type: ignore[union-attr]
                z_i, by0, by1, bx0, bx1,
                np.asarray(
                    labels2d[
                        by0 - int(ky0):by1 - int(ky0),
                        bx0 - int(kx0):bx1 - int(kx0),
                    ],
                    dtype=label_dtype,
                ),
            )
        elif known_slice_bboxes is not None:
            labels_store[z_i, ky0:ky0 + int(labels2d.shape[0]), kx0:kx0 + int(labels2d.shape[1])] = (  # type: ignore[index]
                np.asarray(labels2d, dtype=label_dtype)
            )
        else:
            labels_store[z_i, :, :] = np.asarray(labels2d, dtype=label_dtype)  # type: ignore[index]
        component_counts[z_i] = np.uint32(local_count)

    # A source-space D1 union already carries conservative per-z bboxes. For sparse
    # final topology, bounded CPU CCL avoids initializing the parent CuPy context and
    # transferring/scanning the full dense 16+ GiB volume merely to recover the same
    # slice-local labels. Dense or metadata-free callers retain the established GPU path.
    sparse_cpu_selected = False
    support_nonempty = 0
    support_bbox_pixels = 0
    support_bbox_fraction = 1.0
    sparse_cpu_threshold = float(topology_sparse_cpu_max_coverage())
    if (
        bool(prefer_crop_bounded_cpu_labeling)
        and bool(topology_sparse_cpu_labeling_enabled())
        and known_slice_any is not None
        and known_slice_bboxes is not None
    ):
        support_nonempty, support_bbox_pixels, support_bbox_fraction = binary_slice_bbox_coverage(
            (int(z_dim), int(h), int(w)),
            known_slice_any,
            known_slice_bboxes,
        )
        sparse_cpu_selected = bool(float(support_bbox_fraction) <= float(sparse_cpu_threshold))
        if bool(sparse_cpu_selected):
            label_workers = choose_slice_parallel_workers(
                min(int(label_workers), max(1, int(_cpu_count()))), int(z_dim),
            )
            print(
                'v16.1.7 crop-bounded CPU slice labeling selected: '
                f'nonempty_z={int(support_nonempty)}/{int(z_dim)}, '
                f'bbox_coverage={100.0 * float(support_bbox_fraction):.2f}% '
                f'({int(support_bbox_pixels) / GIB:.2f} GiPixels), '
                f'workers={int(label_workers)}. Parent CUDA/CuPy startup and the '
                f'{array_nbytes((int(z_dim), int(h), int(w)), np.uint8) / GIB:.2f} GiB '
                'dense H2D scan are bypassed. '
                'YOLO_TTA_TOPOLOGY_SPARSE_CPU_LABELING=0 restores GPU labeling.',
                flush=True,
            )
        else:
            print(
                'v16.1.7 crop-bounded CPU slice labeling not selected: '
                f'bbox_coverage={100.0 * float(support_bbox_fraction):.2f}% exceeds '
                f'{100.0 * float(sparse_cpu_threshold):.2f}%; retaining the GPU path.',
                flush=True,
            )
    gpu_labeling_requested = bool(gpu_slice_labeling_enabled() and not sparse_cpu_selected)
    print(
        '3D topology slice-label phase: '
        f'shape={int(z_dim)}x{int(h)}x{int(w)}, CPU fallback workers={int(label_workers)}, '
        f'GPU labeling requested={bool(gpu_labeling_requested)}.'
    )
    runtime_telemetry().gauge('pipeline.phase', '3d_topology_slice_label')
    runtime_telemetry().gauge('topology.slice_label.crop_cpu_selected', bool(sparse_cpu_selected))
    runtime_telemetry().gauge('topology.slice_label.support_bbox_fraction', float(support_bbox_fraction))
    label_phase_started = time.perf_counter()
    if bool(gpu_labeling_requested):
        gpu_stage_a_done, gpu_stage_a_pair_codes = _try_label_slices_stage_a_gpu(
            mask_mm,
            labels_store,
            component_counts,
            slice_bboxes,
            slice_areas if collect_stats else None,
            known_slice_any=known_slice_any,
            preferred_block_slices=int(topology_slab_slices()),
        )
    else:
        gpu_stage_a_done, gpu_stage_a_pair_codes = False, None
    if not gpu_stage_a_done:
        parallel_for_indices_chunked(
            int(z_dim),
            _label_slice_local,
            max_workers=label_workers,
            desc='3D topology: 2D slice labeling',
            show_progress=True,
            target_chunks_per_worker=2,
        )
    label_phase_seconds = time.perf_counter() - label_phase_started

    if sparse_enabled:
        labels_store.finalize()  # type: ignore[union-attr]
        payload_gib = int(labels_store.nbytes) / GIB  # type: ignore[union-attr]
        logical_bytes = max(1, array_nbytes((z_dim, h, w), label_dtype))
        pct = 100.0 * float(int(labels_store.nbytes)) / float(logical_bytes)  # type: ignore[union-attr]
        print(
            f'3D topology sparse labels: {payload_gib:.3f} GiB packed '
            f'({pct:.2f}% of dense local-id raster)'
        )

    total_components = int(np.sum(component_counts, dtype=np.uint64))
    if total_components <= 0:
        flush_array(labels_store)
        return labels_store, 0, label_paths
    if total_components >= 2 ** 32:
        raise RuntimeError('3D component id space exceeded uint32 capacity')

    # Slice z's provisional global id range is
    # [slice_offsets[z] + 1, slice_offsets[z] + component_counts[z]]. The label volume remains
    # in local per-slice ids until compact relabel so we avoid a full-volume promotion write pass.
    cumsum = np.cumsum(component_counts.astype(np.uint64, copy=False), dtype=np.uint64)
    slice_offsets = np.zeros((int(z_dim),), dtype=np.uint32)
    if int(z_dim) > 1:
        slice_offsets[1:] = cumsum[:-1].astype(np.uint32, copy=False)

    def _pair_codes_for_z(z: int) -> np.ndarray:
        if gpu_stage_a_pair_codes is not None:
            local_codes = gpu_stage_a_pair_codes[int(z)]
            if local_codes is not None:
                local_codes_u64 = np.asarray(local_codes, dtype=np.uint64)
                if local_codes_u64.size <= 0:
                    return np.zeros((0,), dtype=np.uint64)
                prev_ids = (local_codes_u64 >> np.uint64(32)) + np.uint64(
                    int(slice_offsets[int(z) - 1])
                )
                curr_ids = (local_codes_u64 & np.uint64(0xFFFFFFFF)) + np.uint64(
                    int(slice_offsets[int(z)])
                )
                return np.ascontiguousarray(
                    (prev_ids << np.uint64(32)) | curr_ids,
                    dtype=np.uint64,
                )
        if int(component_counts[int(z) - 1]) == 0 or int(component_counts[int(z)]) == 0:
            return np.zeros((0,), dtype=np.uint64)
        py0, py1, px0, px1 = (int(v) for v in slice_bboxes[int(z) - 1])
        cy0, cy1, cx0, cx1 = (int(v) for v in slice_bboxes[int(z)])
        ry0 = max(0, max(py0, cy0) - 1)
        ry1 = min(int(h), min(py1, cy1) + 1)
        rx0 = max(0, max(px0, cx0) - 1)
        rx1 = min(int(w), min(px1, cx1) + 1)
        if ry0 >= ry1 or rx0 >= rx1:
            return np.zeros((0,), dtype=np.uint64)
        if isinstance(labels_store, SparseSliceLabelStore):
            prev_local_slice = labels_store.read_window(
                int(z) - 1, int(ry0), int(ry1), int(rx0), int(rx1),
            )
            curr_local_slice = labels_store.read_window(
                int(z), int(ry0), int(ry1), int(rx0), int(rx1),
            )
        else:
            prev_local_slice = np.asarray(labels_store[int(z) - 1, ry0:ry1, rx0:rx1])
            curr_local_slice = np.asarray(labels_store[int(z), ry0:ry1, rx0:rx1])
        return _adjacent_gid_pair_codes(
            prev_local_slice,
            curr_local_slice,
            prev_offset=int(slice_offsets[int(z) - 1]),
            curr_offset=int(slice_offsets[int(z)]),
        )

    topology_started = time.perf_counter()
    slab_ranges = topology_slab_ranges(int(z_dim))
    slab_worker_count = topology_slab_workers(
        max(int(workers), int(pair_workers)), len(slab_ranges),
    )
    union_code_cap = max(1, _env_int(
        'YOLO_TTA_TOPOLOGY_UNION_BATCH_CODES', 1_048_576,
    ))
    print(
        '3D topology local-union plan (v16.1.3): '
        f'{len(slab_ranges)} slab(s) x {int(topology_slab_slices())} slices, '
        f'{int(slab_worker_count)} concurrent worker(s), '
        f'compiled_nogil={bool(_numba_union_find_batch_kernel is not None and compiled_topology_kernels_enabled())}.'
    )
    runtime_telemetry().gauge('pipeline.phase', '3d_topology_local_union')
    runtime_telemetry().gauge('topology.slab_workers', int(slab_worker_count))

    def _resolve_topology_slab(slab_idx: int) -> Tuple[int, int, int, int, np.ndarray, int, float, float]:
        z0, z1 = slab_ranges[int(slab_idx)]
        gid_base = int(slice_offsets[int(z0)])
        gid_count = int(np.sum(component_counts[int(z0):int(z1)], dtype=np.uint64))
        if gid_count <= 0:
            return (
                int(slab_idx), int(z0), int(z1), int(gid_base),
                np.zeros((1,), dtype=np.uint32), 0, 0.0, 0.0,
            )

        local_uf = _UnionFind()
        local_uf.new_ids(int(gid_count))
        pending: List[np.ndarray] = []
        pending_count = 0
        pair_seconds = 0.0
        union_seconds = 0.0

        def _flush_local() -> None:
            nonlocal pending_count, union_seconds
            if pending_count <= 0:
                return
            joined = pending[0] if len(pending) == 1 else np.concatenate(pending)
            t_union = time.perf_counter()
            local_uf.union_pair_codes(np.ascontiguousarray(joined, dtype=np.uint64))
            union_seconds += time.perf_counter() - t_union
            pending.clear()
            pending_count = 0

        for z in range(int(z0) + 1, int(z1)):
            t_pair = time.perf_counter()
            global_codes = np.asarray(_pair_codes_for_z(int(z)), dtype=np.uint64)
            pair_seconds += time.perf_counter() - t_pair
            if global_codes.size <= 0:
                continue
            a_local = (global_codes >> np.uint64(32)) - np.uint64(int(gid_base))
            b_local = (global_codes & np.uint64(0xFFFFFFFF)) - np.uint64(int(gid_base))
            local_codes = np.ascontiguousarray(
                (a_local << np.uint64(32)) | b_local, dtype=np.uint64,
            )
            if pending_count > 0 and pending_count + int(local_codes.size) > int(union_code_cap):
                _flush_local()
            if int(local_codes.size) >= int(union_code_cap):
                for code0 in range(0, int(local_codes.size), int(union_code_cap)):
                    t_union = time.perf_counter()
                    local_uf.union_pair_codes(np.ascontiguousarray(
                        local_codes[int(code0):int(code0) + int(union_code_cap)],
                        dtype=np.uint64,
                    ))
                    union_seconds += time.perf_counter() - t_union
            else:
                pending.append(local_codes)
                pending_count += int(local_codes.size)
        _flush_local()

        local_root_map = local_uf.root_map()
        unique_local_roots = np.unique(local_root_map[1:])
        unique_local_roots = unique_local_roots[unique_local_roots > 0]
        root_to_slab_component = np.zeros(local_root_map.shape, dtype=np.uint32)
        root_to_slab_component[unique_local_roots] = np.arange(
            1, int(unique_local_roots.size) + 1, dtype=np.uint32,
        )
        gid_local_to_slab_component = np.ascontiguousarray(
            root_to_slab_component[local_root_map], dtype=np.uint32,
        )
        return (
            int(slab_idx), int(z0), int(z1), int(gid_base),
            gid_local_to_slab_component, int(unique_local_roots.size),
            float(pair_seconds), float(union_seconds),
        )

    slab_results: List[Optional[Tuple[int, int, int, int, np.ndarray, int, float, float]]] = [
        None
    ] * len(slab_ranges)
    if slab_worker_count <= 1:
        for slab_idx in tqdm(
            range(len(slab_ranges)), desc='Topology: local 3D slab unions',
        ):
            result = _resolve_topology_slab(int(slab_idx))
            slab_results[int(result[0])] = result
    else:
        for result_obj in tqdm(
            parallel_map_unordered(
                _resolve_topology_slab,
                range(len(slab_ranges)),
                max_workers=int(slab_worker_count),
                max_pending=max(int(slab_worker_count), int(slab_worker_count) * 2),
            ),
            total=len(slab_ranges),
            desc='Topology: local 3D slab unions',
        ):
            result = result_obj  # type: ignore[assignment]
            slab_results[int(result[0])] = result

    gid_to_slab_component = np.zeros((int(total_components) + 1,), dtype=np.uint32)
    slab_component_count = 0
    slab_pair_seconds = 0.0
    slab_union_seconds = 0.0
    for result_opt in slab_results:
        if result_opt is None:
            raise RuntimeError('Topology slab worker returned no result')
        (
            _slab_idx, _z0, _z1, gid_base, local_map, local_component_count,
            pair_seconds, union_seconds,
        ) = result_opt
        slab_pair_seconds += float(pair_seconds)
        slab_union_seconds += float(union_seconds)
        gid_count = int(local_map.size) - 1
        if gid_count > 0:
            mapped = local_map[1:].astype(np.uint64, copy=False) + np.uint64(
                int(slab_component_count)
            )
            gid_to_slab_component[
                int(gid_base) + 1:int(gid_base) + int(gid_count) + 1
            ] = mapped.astype(np.uint32, copy=False)
        slab_component_count += int(local_component_count)

    if slab_component_count <= 0:
        root_map = np.zeros((int(total_components) + 1,), dtype=np.uint32)
        topology_boundary_seconds = 0.0
        topology_root_seconds = 0.0
    else:
        global_uf = _UnionFind()
        global_uf.new_ids(int(slab_component_count))
        boundary_pending: List[np.ndarray] = []
        boundary_pending_count = 0
        topology_boundary_pair_seconds = 0.0
        topology_boundary_union_seconds = 0.0

        def _flush_boundaries() -> None:
            nonlocal boundary_pending_count, topology_boundary_union_seconds
            if boundary_pending_count <= 0:
                return
            joined = boundary_pending[0] if len(boundary_pending) == 1 else np.concatenate(boundary_pending)
            t_union = time.perf_counter()
            global_uf.union_pair_codes(np.ascontiguousarray(joined, dtype=np.uint64))
            topology_boundary_union_seconds += time.perf_counter() - t_union
            boundary_pending.clear()
            boundary_pending_count = 0

        def _submit_original_boundary_codes(original_codes: np.ndarray) -> None:
            nonlocal boundary_pending_count
            codes = np.asarray(original_codes, dtype=np.uint64)
            if codes.size <= 0:
                return
            a_gid = (codes >> np.uint64(32)).astype(np.int64, copy=False)
            b_gid = (codes & np.uint64(0xFFFFFFFF)).astype(np.int64, copy=False)
            a_nodes = gid_to_slab_component[a_gid].astype(np.uint64, copy=False)
            b_nodes = gid_to_slab_component[b_gid].astype(np.uint64, copy=False)
            valid = (a_nodes > 0) & (b_nodes > 0) & (a_nodes != b_nodes)
            if not bool(np.any(valid)):
                return
            mapped_codes = np.unique(
                (a_nodes[valid] << np.uint64(32)) | b_nodes[valid]
            ).astype(np.uint64, copy=False)
            if boundary_pending_count > 0 and boundary_pending_count + int(mapped_codes.size) > int(union_code_cap):
                _flush_boundaries()
            if int(mapped_codes.size) >= int(union_code_cap):
                for code0 in range(0, int(mapped_codes.size), int(union_code_cap)):
                    t_union = time.perf_counter()
                    global_uf.union_pair_codes(np.ascontiguousarray(
                        mapped_codes[int(code0):int(code0) + int(union_code_cap)],
                        dtype=np.uint64,
                    ))
                    topology_boundary_union_seconds += time.perf_counter() - t_union
            else:
                boundary_pending.append(np.ascontiguousarray(mapped_codes, dtype=np.uint64))
                boundary_pending_count += int(mapped_codes.size)

        for slab_idx in range(1, len(slab_ranges)):
            boundary_z = int(slab_ranges[int(slab_idx)][0])
            t_pair = time.perf_counter()
            boundary_codes = _pair_codes_for_z(int(boundary_z))
            topology_boundary_pair_seconds += time.perf_counter() - t_pair
            _submit_original_boundary_codes(boundary_codes)

        if bool(wrap_axis) and int(z_dim) > 1:
            if int(component_counts[0]) > 0 and int(component_counts[int(z_dim) - 1]) > 0:
                t_pair = time.perf_counter()
                if isinstance(labels_store, SparseSliceLabelStore):
                    ly0, ly1, lx0, lx1 = (int(v) for v in slice_bboxes[int(z_dim) - 1])
                    fy0, fy1, fx0, fx1 = (int(v) for v in slice_bboxes[0])
                    mfx0, mfx1 = int(w - fx1), int(w - fx0)
                    ry0 = max(0, max(ly0, fy0) - 1)
                    ry1 = min(int(h), min(ly1, fy1) + 1)
                    rx0 = max(0, max(lx0, mfx0) - 1)
                    rx1 = min(int(w), min(lx1, mfx1) + 1)
                    if ry0 < ry1 and rx0 < rx1:
                        last_gid_slice = labels_store.read_window(
                            int(z_dim) - 1, ry0, ry1, rx0, rx1,
                        )
                        first_gid_slice = labels_store.read_window(
                            0, ry0, ry1, int(w - rx1), int(w - rx0),
                        )[:, ::-1]
                    else:
                        last_gid_slice = np.empty((0, 0), dtype=label_dtype)
                        first_gid_slice = np.empty((0, 0), dtype=label_dtype)
                else:
                    first_gid_slice = np.asarray(labels_store[0])  # type: ignore[index]
                    last_gid_slice = np.asarray(labels_store[int(z_dim) - 1])  # type: ignore[index]
                    first_gid_slice = first_gid_slice[:, ::-1]
                wrap_codes = _adjacent_gid_pair_codes(
                    last_gid_slice,
                    first_gid_slice,
                    prev_offset=int(slice_offsets[int(z_dim) - 1]),
                    curr_offset=int(slice_offsets[0]),
                )
                topology_boundary_pair_seconds += time.perf_counter() - t_pair
                _submit_original_boundary_codes(wrap_codes)
        _flush_boundaries()
        topology_boundary_seconds = float(
            topology_boundary_pair_seconds + topology_boundary_union_seconds
        )

        t_root = time.perf_counter()
        slab_node_root_map = global_uf.root_map()
        root_map = np.ascontiguousarray(
            slab_node_root_map[gid_to_slab_component], dtype=np.uint32,
        )
        root_map[0] = np.uint32(0)
        topology_root_seconds = time.perf_counter() - t_root

    topology_total_seconds = time.perf_counter() - topology_started
    topology_phase_seconds = {
        'slice_label': float(label_phase_seconds),
        'internal_pair_extraction': float(slab_pair_seconds),
        'local_slab_union': float(slab_union_seconds),
        'boundary_merge': float(topology_boundary_seconds),
        'root_expansion': float(topology_root_seconds),
        'topology_total': float(topology_total_seconds),
    }
    print(
        'v16.0.2 topology slabs: '
        f'{len(slab_ranges)} slab(s) x {int(topology_slab_slices())} slices, '
        f'{int(slab_worker_count)} concurrent local union worker(s), '
        f'{int(total_components)} slice component(s) -> {int(slab_component_count)} slab component(s); '
        f'label={label_phase_seconds:.3f}s, internal_pairs={slab_pair_seconds:.3f}s, '
        f'local_union={slab_union_seconds:.3f}s, boundary/root='
        f'{float(topology_boundary_seconds) + float(topology_root_seconds):.3f}s.'
    )
    if root_map.shape[0] <= 1:
        flush_array(labels_store)
        return labels_store, 0, label_paths

    unique_roots = np.unique(root_map[1:])
    unique_roots = unique_roots[unique_roots > 0]

    compact_root_ids = np.zeros(root_map.shape, dtype=np.uint32)
    compact_root_ids[unique_roots] = np.arange(1, unique_roots.size + 1, dtype=np.uint32)
    gid_to_compact = compact_root_ids[root_map]

    # per-slice local->compact LUTs live in one concatenated table (slice z's
    # local id v maps through lut_flat[lut_offsets[z] + v]), and the relabel touches only each
    # slice's recorded foreground bbox — labels outside it are guaranteed zero.
    # the same table is now ALSO exported through component_stats_out
    # ('slice_local_luts') so callers can consume per-slice local ids directly and skip the
    # full-volume relabel pass entirely.
    lut_started = time.perf_counter()
    lut_sizes = component_counts.astype(np.int64, copy=False) + 1
    lut_offsets = np.zeros((int(z_dim),), dtype=np.int64)
    if int(z_dim) > 1:
        lut_offsets[1:] = np.cumsum(lut_sizes)[:-1]
    lut_flat = np.zeros((int(lut_sizes.sum()),), dtype=np.uint32)
    for z in range(int(z_dim)):
        count = int(component_counts[int(z)])
        if count > 0:
            offset = int(slice_offsets[int(z)])
            lo = int(lut_offsets[int(z)])
            lut_flat[lo + 1:lo + count + 1] = gid_to_compact[offset + 1:offset + count + 1]
    topology_phase_seconds['lut_build'] = float(time.perf_counter() - lut_started)

    if collect_stats:
        area_started = time.perf_counter()
        # per-gid areas -> per-root areas (bincount is one vectorized pass; voxel
        # counts stay far below 2^53 so the float64 accumulation is exact).
        gid_areas = np.zeros((int(total_components) + 1,), dtype=np.int64)
        for z in range(int(z_dim)):
            count = int(component_counts[int(z)])
            if count <= 0:
                continue
            offset = int(slice_offsets[int(z)])
            gid_areas[offset + 1:offset + count + 1] = slice_areas[int(z)]
        root_areas = np.bincount(
            root_map.astype(np.int64, copy=False),
            weights=gid_areas.astype(np.float64, copy=False),
            minlength=int(total_components) + 1,
        ).astype(np.int64)
        topology_phase_seconds['area_reduction'] = float(time.perf_counter() - area_started)
        component_stats_out.update({
            'component_counts': component_counts,
            'slice_offsets': slice_offsets,
            'slice_bboxes': slice_bboxes,
            'root_map': root_map,
            'unique_roots': unique_roots,
            'root_areas': root_areas,
            'total_components': int(total_components),
            'topology_phase_seconds': dict(topology_phase_seconds),
            'topology_slab_count': int(len(slab_ranges)),
            'topology_slab_workers': int(slab_worker_count),
            'topology_slab_components': int(slab_component_count),
            'slice_local_luts': SliceLocalLabelLUTs(
                lut_flat=lut_flat,
                lut_offsets=lut_offsets,
                component_counts=component_counts,
            ),
        })

    if not compact_relabel:
        # the caller consumes per-slice LOCAL ids through root LUTs — skip the
        # full-volume relabel write pass entirely.
        flush_array(labels_store)
        return labels_store, int(unique_roots.size), label_paths

    kernel_done = False
    if compiled_topology_kernels_enabled() and _numba_compact_relabel_kernel is not None:
        try:
            print('3D topology: compact relabel via numba nogil kernel')
            _numba_compact_relabel_kernel(
                np.asarray(labels_store),
                lut_flat,
                lut_offsets,
                np.ascontiguousarray(slice_bboxes),
                component_counts,
            )
            kernel_done = True
        except Exception as exc:
            print(f'3D topology: numba compact relabel unavailable ({exc}); using the thread pool.')

    if not kernel_done:
        compact_tasks: List[Tuple[int, int, int, int, int]] = []
        row_block = max(1, _env_int('YOLO_TTA_INTERPOLATION_COMPACT_RELABEL_ROWS', 256))
        for z in range(int(z_dim)):
            if int(component_counts[int(z)]) <= 0:
                continue
            by0, by1, bx0, bx1 = (int(v) for v in slice_bboxes[int(z)])
            for y0 in range(by0, by1, int(row_block)):
                compact_tasks.append((int(z), int(y0), int(min(by1, y0 + int(row_block))), bx0, bx1))

        def _compact_block(task_idx: int) -> int:
            z, y0, y1, x0, x1 = compact_tasks[int(task_idx)]
            lo = int(lut_offsets[int(z)])
            local_to_compact = lut_flat[lo:lo + int(component_counts[int(z)]) + 1]
            block = np.asarray(labels_store[int(z), int(y0):int(y1), int(x0):int(x1)])
            if np.any(block):
                labels_store[int(z), int(y0):int(y1), int(x0):int(x1)] = local_to_compact[block]
            # an all-zero block is already zero in the store; the old
            # else-branch fill(0) rewrote it needlessly (dirtying pages on memmap-backed stores).
            return int(y1) - int(y0)

        if compact_tasks:
            pending = max(compact_workers, compact_workers * 8)
            if compact_workers <= 1:
                for task_idx in tqdm(range(len(compact_tasks)), desc='3D topology: compact relabel'):
                    _compact_block(int(task_idx))
            else:
                for _rows_done in tqdm(
                    parallel_map_unordered(
                        _compact_block,
                        range(len(compact_tasks)),
                        max_workers=compact_workers,
                        max_pending=pending,
                    ),
                    total=len(compact_tasks),
                    desc='3D topology: compact relabel',
                ):
                    pass

    flush_array(labels_store)
    return labels_store, int(unique_roots.size), label_paths

def build_slice_endpoint_seeds_from_label_volume(
    labels_real: np.ndarray,
    workers: int = 1,
    desc: str = 'Interpolation: endpoint seeds [scan]',
    wrap_axis: bool = False,
    component_cache: Optional['SliceComponentTableCache'] = None,
) -> Tuple[List[SliceEndpointSeed], int]:
    """Fast slice-graph endpoint scan for slice-direction interpolation.

 This avoids per-object 3D voxel skeletonization on large relabeled volumes, which can become
 prohibitively slow when an object's bounding box spans a large fraction of the volume. Endpoints
 are identified from connected components in each slice that do not continue into the previous or
 next slice of the same relabeled object. When wrap_axis is true, slice 0 and the
 final slice are also considered adjacent for Radial interpolation.

 When a component cache is supplied, the scan reuses cached per-slice component tables and tests
 continuation only inside each component's crop. This makes endpoint discovery and later seed
 planning share the same component records instead of relabeling slices again per seed/bridge."""
    z_dim = int(labels_real.shape[0])
    if z_dim <= 0:
        return [], 0

    def _scan_slice_from_cache(z: int) -> List[SliceEndpointSeed]:
        if component_cache is None:
            return []
        z_i = int(z)
        table = component_cache.get(z_i)
        if not table.components:
            return []

        prev_z: Optional[int] = None
        next_z: Optional[int] = None
        prev_wrapped = False
        next_wrapped = False
        if bool(wrap_axis) and z_dim > 1:
            prev_z = int((z_i - 1) % z_dim)
            next_z = int((z_i + 1) % z_dim)
            # continuation across the radial 0°/180° wrap happens at
            # u -> width-1-u, not at the same u.
            prev_wrapped = z_i == 0
            next_wrapped = z_i == (z_dim - 1)
        else:
            if z_i > 0:
                prev_z = int(z_i - 1)
            if (z_i + 1) < z_dim:
                next_z = int(z_i + 1)

        slice_w = int(table.shape[1])

        def _record_continues_in_neighbor(
            record: SliceComponentRecord,
            neighbor_z: Optional[int],
            *,
            mirrored: bool,
        ) -> bool:
            """Test exact adjacent-slice footprint overlap in one bbox read.

            The previous implementation compared a component against every same-label
            component in the adjacent slice.  A tiled label fragmented into K components
            therefore paid O(K²) Python calls per slice.  Direct overlap has a simpler
            equivalent definition: at any pixel in this component, the adjacent canonical
            label equals ``record.label``.  Reading only this component's bbox preserves that
            definition, including local-label LUTs and radial wrap mirroring, without building
            or scanning the neighbor's component list.
            """
            if neighbor_z is None:
                return False
            y0, x0, y1, x1 = (int(v) for v in record.bbox)
            read_x0, read_x1 = int(x0), int(x1)
            if bool(mirrored):
                read_x0, read_x1 = int(slice_w - x1), int(slice_w - x0)

            labels_store = component_cache.labels_real
            if isinstance(labels_store, SparseSliceLabelStore):
                neighbor = labels_store.read_window(
                    int(neighbor_z), int(y0), int(y1), int(read_x0), int(read_x1),
                )
            else:
                neighbor = np.asarray(
                    labels_store[
                        int(neighbor_z), int(y0):int(y1), int(read_x0):int(read_x1)
                    ]
                )
            if bool(mirrored):
                neighbor = neighbor[:, ::-1]
            if component_cache.slice_luts is not None:
                neighbor = component_cache.slice_luts.lut_for(int(neighbor_z))[neighbor]
            return bool(np.any(record.mask_crop & (neighbor == int(record.label))))

        seeds_local: List[SliceEndpointSeed] = []
        for record in table.components:
            y0, x0, y1, x1 = (int(v) for v in record.bbox)
            planning_cost = max(
                1,
                int(record.area),
                max(0, int(y1) - int(y0)) * max(0, int(x1) - int(x0)),
            )
            has_prev = _record_continues_in_neighbor(
                record, prev_z, mirrored=bool(prev_wrapped),
            )
            has_next = _record_continues_in_neighbor(
                record, next_z, mirrored=bool(next_wrapped),
            )

            if not has_prev:
                seeds_local.append(SliceEndpointSeed(
                    label=int(record.label),
                    point=(z_i, int(record.anchor[0]), int(record.anchor[1])),
                    direction_sign=-1,
                    planning_cost=int(planning_cost),
                ))
            if not has_next:
                seeds_local.append(SliceEndpointSeed(
                    label=int(record.label),
                    point=(z_i, int(record.anchor[0]), int(record.anchor[1])),
                    direction_sign=1,
                    planning_cost=int(planning_cost),
                ))
        return seeds_local

    def _scan_slice(z: int) -> List[SliceEndpointSeed]:
        if component_cache is not None:
            return _scan_slice_from_cache(int(z))

        curr_slice = np.asarray(labels_real[int(z)])
        if not np.any(curr_slice):
            return []

        if bool(wrap_axis) and z_dim > 1:
            # the wrap-adjacent frame continues at u -> width-1-u, so
            # mirror the neighbor slice when the adjacency crosses the 0°/180° boundary.
            prev_slice = np.asarray(labels_real[(int(z) - 1) % z_dim])
            if int(z) == 0:
                prev_slice = prev_slice[:, ::-1]
            next_slice = np.asarray(labels_real[(int(z) + 1) % z_dim])
            if int(z) == (z_dim - 1):
                next_slice = next_slice[:, ::-1]
        else:
            prev_slice = np.asarray(labels_real[int(z) - 1]) if int(z) > 0 else None
            next_slice = np.asarray(labels_real[int(z) + 1]) if (int(z) + 1) < z_dim else None

        seeds_local: List[SliceEndpointSeed] = []
        present = np.unique(curr_slice)
        present = present[present > 0]
        if present.size == 0:
            return seeds_local

        for obj_id in present.tolist():
            obj_mask = (curr_slice == int(obj_id)).astype(np.uint8, copy=False)
            num_cc, cc = cv2.connectedComponents(obj_mask, connectivity=8, ltype=cv2.CV_32S)
            if int(num_cc) <= 1:
                continue

            prev_same = (prev_slice == int(obj_id)) if prev_slice is not None else None
            next_same = (next_slice == int(obj_id)) if next_slice is not None else None

            for local_lbl in range(1, int(num_cc)):
                comp = cc == int(local_lbl)
                if not np.any(comp):
                    continue
                anchor = _component_centroid_anchor(comp)
                if anchor is None:
                    continue
                ys, xs = np.nonzero(comp)
                planning_cost = max(
                    1,
                    int(ys.size),
                    (int(ys.max()) - int(ys.min()) + 1)
                    * (int(xs.max()) - int(xs.min()) + 1),
                )

                # endpoint continuation is defined by direct footprint overlap in
                # the adjacent slice; do not dilate/skeletonize the component for endpoint discovery.
                has_prev = bool(prev_same is not None and np.any(comp & prev_same))
                has_next = bool(next_same is not None and np.any(comp & next_same))

                if not has_prev:
                    seeds_local.append(SliceEndpointSeed(
                        label=int(obj_id),
                        point=(int(z), int(anchor[0]), int(anchor[1])),
                        direction_sign=-1,
                        planning_cost=int(planning_cost),
                    ))
                if not has_next:
                    seeds_local.append(SliceEndpointSeed(
                        label=int(obj_id),
                        point=(int(z), int(anchor[0]), int(anchor[1])),
                        direction_sign=1,
                        planning_cost=int(planning_cost),
                    ))

        return seeds_local

    worker_count = choose_slice_parallel_workers(int(workers), z_dim)
    if component_cache is not None:
        component_cache.prebuild(
            workers=worker_count,
            desc=f'{desc}: component tables',
        )

    seeds: List[SliceEndpointSeed] = []

    if worker_count <= 1:
        for z in tqdm(range(z_dim), desc=desc):
            seeds.extend(_scan_slice(int(z)))
    else:
        pending = max(worker_count, worker_count * 8)
        for seeds_local in tqdm(
            parallel_map_unordered(_scan_slice, range(z_dim), max_workers=worker_count, max_pending=pending),
            total=z_dim,
            desc=desc,
        ):
            seeds.extend(seeds_local)

    seeds.sort(key=lambda s: (int(s.label), int(s.point[0]), int(s.direction_sign), int(s.point[1]), int(s.point[2])))
    return seeds, int(len(seeds))
