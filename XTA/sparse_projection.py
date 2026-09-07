"""Exact sparse Azimuthal/tilted-Azimuthal projection into immutable source mask stores.

The legacy projector gathers a discrete Azimuthal ownership map for every stack
frame. This module inverts that same relation and visits only positive input
samples. Input crops, map construction strips and output crops are bounded; no
dense view-native mask or volume-sized coordinate list is materialized.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import threading
import time
from typing import Dict, Iterator, Tuple

import numpy as np
from ._deps import _numba
from .geometry import ViewInfo, is_azimuthal_view, is_tilted_azimuthal_view, azimuthal_base_view_name, azimuthal_plane_shape, azimuthal_source_tilted_view, tilted_frame_center, tilted_stack_axis_length
from .interpolation import INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, RawBBoxSlicePayload, _write_raw_bbox_payload_store

_MAP_CACHE_MAX_BYTES = 256 * 1024 * 1024
_MAP_STRIP_PIXELS = 262144
_INPUT_SLAB_BYTES = 8 * 1024 * 1024
_CACHE: OrderedDict[tuple, '_InverseMap'] = OrderedDict()
_CACHE_BYTES = 0
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _InverseMap:
    key_offsets: np.ndarray
    owners: np.ndarray
    row_offsets: np.ndarray
    frames: np.ndarray
    shear: np.ndarray
    plane_shape: Tuple[int, int]
    working_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int, int]
    processing_width: int
    base_id: int
    tilted: bool
    vertical: bool

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in (
            self.key_offsets, self.owners, self.row_offsets, self.frames, self.shear,
        ))


def clear_sparse_projection_cache() -> None:
    """Release cached map owners; active projector calls retain their own maps."""
    global _CACHE_BYTES
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_BYTES = 0


def sparse_projection_cache_info() -> Dict[str, int]:
    with _CACHE_LOCK:
        return {'entries': len(_CACHE), 'bytes': int(_CACHE_BYTES), 'budget_bytes': int(_MAP_CACHE_MAX_BYTES)}


def _count_keys(keys, counts):
    for key in keys:
        counts[key] += 1


def _prefix_counts(counts):
    offsets = np.empty(len(counts) + 1, dtype=np.uint32)
    total = np.uint64(0)
    offsets[0] = 0
    for index in range(len(counts)):
        total += np.uint64(counts[index])
        if total > np.uint64(0xFFFFFFFF):
            raise ValueError('Sparse projection map exceeds uint32 owner capacity')
        offsets[index + 1] = np.uint32(total)
    return offsets


def _fill_owners(keys, positions, cursors, owners):
    for index in range(len(keys)):
        key = keys[index]
        at = cursors[key]
        owners[at] = positions[index]
        cursors[key] = at + 1


def _scatter_crop(
    crop, azimuth, first_row, first_u, key_offsets, owners, row_offsets, frames,
    shear, plane_w, processing_width, base_id, tilted, vertical,
    work_t, work_h, work_w, out_t, out_h, out_w, packed, bounds, slice_counts,
):
    foreground = contributions = unique = 0
    packed_w = (out_w + 7) // 8
    for row in range(crop.shape[0]):
        source_row = first_row + row
        for col in range(crop.shape[1]):
            if crop[row, col] == 0:
                continue
            foreground += 1
            key = azimuth * processing_width + first_u + col
            for frame_at in range(row_offsets[source_row], row_offsets[source_row + 1]):
                frame = int(frames[frame_at])
                for owner_at in range(key_offsets[key], key_offsets[key + 1]):
                    owner = int(owners[owner_at])
                    v, u = owner // plane_w, owner % plane_w
                    if tilted:
                        stack = int(shear[frame, v if vertical else u])
                        if stack < 0:
                            continue
                        if base_id == 0:
                            t, y, x = stack, v, u
                        elif base_id == 1:
                            t, y, x = v, stack, u
                        else:
                            t, y, x = v, u, stack
                        t = min(t * out_t // work_t, out_t - 1)
                        y = min(y * out_h // work_h, out_h - 1)
                        x = min(x * out_w // work_w, out_w - 1)
                    elif base_id == 0:
                        t, y, x = frame, v, u
                    elif base_id == 1:
                        t, y, x = v, frame, u
                    else:
                        t, y, x = v, u, frame
                    at = (t * out_h + y) * packed_w + (x >> 3)
                    bit = np.uint8(1 << (x & 7))
                    contributions += 1
                    if not (packed[at] & bit):
                        packed[at] |= bit
                        unique += 1
                        slice_counts[t] += 1
                        bounds[t, 0] = min(bounds[t, 0], y)
                        bounds[t, 1] = max(bounds[t, 1], y + 1)
                        bounds[t, 2] = min(bounds[t, 2], x)
                        bounds[t, 3] = max(bounds[t, 3], x + 1)
    return foreground, contributions, unique


if _numba is not None:
    _count_keys = _numba.njit(cache=True, nogil=True)(_count_keys)
    _prefix_counts = _numba.njit(cache=True, nogil=True)(_prefix_counts)
    _fill_owners = _numba.njit(cache=True, nogil=True)(_fill_owners)
    _scatter_crop = _numba.njit(cache=True, nogil=True)(_scatter_crop)


def _map_key_strips(view, plan, grid, plane_shape) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """The existing dense-map equations, evaluated in bounded full-width strips.

    Do not call the legacy map builder here: its global cache retains full maps
    indefinitely. Tests compare these keys to that actual builder's result.
    """
    work_h, work_w = azimuthal_plane_shape(view)
    out_h, out_w = plane_shape
    if not plan:
        return
    radius = float(view.roi_radius)
    if radius <= 0:
        radius = max(1.0, float(view.diameter - 1) / 2.0)
    diameter = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
    angles = np.asarray([float(sample.angle_deg) % 180.0 for sample in plan], dtype=np.float32)
    sources = np.asarray([int(sample.source_index) for sample in plan], dtype=np.int32)
    reverses = np.asarray([bool(sample.reverse_u) for sample in plan], dtype=bool)
    diffs = np.diff(angles.astype(np.float64, copy=False))
    positive = diffs[diffs > 1e-9]
    step = float(np.median(positive)) if positive.size else 180.0 / float(len(plan))
    if len(plan) == 1:
        step = 180.0
    step = max(step, 1e-9)
    strip_rows = max(1, _MAP_STRIP_PIXELS // max(1, out_w))
    for y0 in range(0, out_h, strip_rows):
        y1 = min(out_h, y0 + strip_rows)
        yy, xx = np.indices((y1 - y0, out_w), dtype=np.float32)
        yy += np.float32(y0)
        if (out_h, out_w) != (int(work_h), int(work_w)):
            xx = (xx + np.float32(0.5)) * np.float32(float(work_w) / float(out_w)) - np.float32(0.5)
            yy = (yy + np.float32(0.5)) * np.float32(float(work_h) / float(out_h)) - np.float32(0.5)
        dx, dy = xx - float(view.center_x), yy - float(view.center_y)
        rr = np.sqrt((dx * dx) + (dy * dy)).astype(np.float32, copy=False)
        valid = rr <= radius + 0.5
        theta = np.mod(np.degrees(np.arctan2(dy, dx)).astype(np.float32, copy=False), 180.0).astype(np.float32, copy=False)
        nearest = np.mod(np.rint(theta / step).astype(np.int32, copy=False), len(plan))
        target = angles[nearest]
        cos_t = np.cos(np.deg2rad(target)).astype(np.float32, copy=False)
        sin_t = np.sin(np.deg2rad(target)).astype(np.float32, copy=False)
        signed = dx * cos_t + dy * sin_t
        signed[reverses[nearest]] *= -1.0
        u_float = ((signed + radius) / max(1e-6, 2.0 * radius)) * float(diameter - 1)
        native_u = np.clip(np.rint(u_float).astype(np.int32, copy=False), 0, diameter - 1)
        local = np.flatnonzero(valid.reshape(-1))
        keys = sources[nearest].reshape(-1)[local].astype(np.int64) * int(grid.processing_w)
        keys += grid.native_u_to_processing[native_u.reshape(-1)[local]]
        positions = np.asarray(local + y0 * out_w, dtype=np.uint32)
        yield np.ascontiguousarray(keys), np.ascontiguousarray(positions)


def _make_inverse_map(view, input_shape, output_shape) -> _InverseMap:
    from . import backprojection as projection

    # resolve_azimuthal_processing_grid only inspects shape before constructing its
    # 1-D OpenCV coordinate ramps. This probe has one byte of backing storage.
    probe = np.broadcast_to(np.zeros((1, 1, 1), dtype=np.uint8), input_shape)
    grid = projection.resolve_azimuthal_processing_grid(probe, view)
    plan, _ = projection.build_azimuthal_backprojection_plan(view)
    base_id = {'transverse': 0, 'sagittal': 1, 'coronal': 2}[azimuthal_base_view_name(view)]
    tilted = is_tilted_azimuthal_view(view)
    vertical = str(view.tilt_direction) == 'vertical'
    working_shape = (int(view.full_t), int(view.full_h), int(view.full_w))
    if tilted:
        tilted_source = azimuthal_source_tilted_view(view)
        plane_shape = (int(tilted_source.src_h), int(tilted_source.src_w))
        frame_count = int(tilted_source.num_slices)
        if str(tilted_source.tilt_direction) not in ('vertical', 'horizontal'):
            raise ValueError('Unsupported tilted Azimuthal direction')
        axis_count = plane_shape[0] if vertical else plane_shape[1]
        axis_center = float((axis_count - 1) / 2.0)
        axis = np.arange(axis_count, dtype=np.int32)
        tangent = float(math.tan(math.radians(float(tilted_source.tilt_angle_deg))))
        shear = np.empty((frame_count, axis_count), dtype=np.int32)
        stack_limit = int(tilted_stack_axis_length(tilted_source))
        for frame in range(frame_count):
            # Preserve the reference's NumPy float32/tie-rounding path exactly.
            stack_float = float(tilted_frame_center(tilted_source, frame)) + (
                tangent * (axis.astype(np.float32, copy=False) - axis_center)
            )
            values = np.rint(stack_float).astype(np.int32, copy=False)
            values[(values < 0) | (values >= stack_limit)] = -1
            shear[frame] = values
    else:
        frame_count, plane_shape = projection._azimuthal_output_stack_and_plane_shape(view, output_shape)
        shear = np.empty((0, 0), dtype=np.int32)
    if int(plane_shape[0]) * int(plane_shape[1]) > 0xFFFFFFFF:
        raise ValueError('Sparse projection base plane exceeds uint32 map capacity')
    key_count = int(input_shape[0]) * int(input_shape[2])
    counts = np.zeros(key_count, dtype=np.uint32)
    for keys, _positions in _map_key_strips(view, plan, grid, plane_shape):
        if keys.size and (int(keys.min()) < 0 or int(keys.max()) >= key_count):
            raise ValueError('Azimuthal ownership map references an absent source sample')
        _count_keys(keys, counts)
    offsets = _prefix_counts(counts)
    owners = np.empty(int(offsets[-1]), dtype=np.uint32)
    counts[:] = offsets[:-1]
    for keys, positions in _map_key_strips(view, plan, grid, plane_shape):
        _fill_owners(keys, positions, counts, owners)
    del counts
    row_lists = [[] for _ in range(int(input_shape[1]))]
    for frame in range(frame_count):
        for row in projection._azimuthal_processing_rows_for_output(grid, frame_count, frame):
            row_lists[int(row)].append(frame)
    row_counts = np.asarray([len(items) for items in row_lists], dtype=np.uint32)
    row_offsets = _prefix_counts(row_counts)
    frames = np.asarray([frame for items in row_lists for frame in items], dtype=np.uint32)
    result = _InverseMap(offsets, owners, row_offsets, frames, shear, tuple(plane_shape),
                         working_shape, output_shape, int(input_shape[2]), base_id, tilted, vertical)
    for value in (offsets, owners, row_offsets, frames, shear):
        value.flags.writeable = False
    return result


def _inverse_map(view, input_shape, output_shape):
    global _CACHE_BYTES
    # Include every ViewInfo field rather than silently aliasing different affine
    # policies or rounding geometries. Evicted entries remain safe for active calls.
    key = (view, tuple(input_shape), tuple(output_shape))
    with _CACHE_LOCK:
        found = _CACHE.pop(key, None)
        if found is not None:
            _CACHE[key] = found
            return found, True
        value = _make_inverse_map(view, input_shape, output_shape)
        if value.nbytes <= _MAP_CACHE_MAX_BYTES:
            while _CACHE and _CACHE_BYTES + value.nbytes > _MAP_CACHE_MAX_BYTES:
                _, old = _CACHE.popitem(last=False)
                _CACHE_BYTES -= old.nbytes
            _CACHE[key] = value
            _CACHE_BYTES += value.nbytes
        return value, False


def _input_slabs(store: RawBBoxMaskStore):
    """Yield bounded row slabs, without calling full-slice/full-volume decode."""
    depth, height, width = map(int, store.shape)
    backing = store._chunks_bytes if store._chunks_bytes is not None else store._chunks_mmap
    for z in range(depth):
        rec = store.index[z]
        kind = int(rec['kind'])
        if kind == 0:
            continue
        if kind != 1:
            raise ValueError(f'{store.root}: invalid mask chunk marker {kind}')
        y0, x0, y1, x1 = (int(rec[field]) for field in ('y0', 'x0', 'y1', 'x1'))
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f'{store.root}: invalid sparse projection input bounds')
        rows, cols = y1-y0, x1-x0
        stride = (cols+7)//8 if store._packbits_payload else cols
        if int(rec['payload_nbytes']) != rows*cols or int(rec['payload_size']) != rows*stride:
            raise ValueError(f'{store.root}: input payload size does not match bounds')
        begin = int(rec['offset'])
        rows_per_slab = max(1, _INPUT_SLAB_BYTES // max(1, cols))
        for row0 in range(0, rows, rows_per_slab):
            count_rows = min(rows_per_slab, rows-row0)
            start, count = begin + row0*stride, count_rows*stride
            if backing is not None:
                if start+count > len(backing):
                    raise IOError(f'{store.root}: short input payload')
                data = np.frombuffer(backing, dtype=np.uint8, count=count, offset=start).reshape(count_rows, stride)
            else:
                with store.chunks_path.open('rb') as stream:
                    stream.seek(start)
                    payload = stream.read(count)
                if len(payload) != count:
                    raise IOError(f'{store.root}: short input payload')
                data = np.frombuffer(payload, dtype=np.uint8).reshape(count_rows, stride)
            crop = np.unpackbits(data, axis=1, count=cols, bitorder='little') if store._packbits_payload else data
            yield z, y0+row0, x0, crop, count


def _packed_output_slice(z, packed, bounds, slice_counts):
    y0, y1, x0, x1 = (int(value) for value in bounds[z])
    if y0 >= y1 or x0 >= x1:
        return RawBBoxSlicePayload(idx=int(z), is_empty=True)
    width = x1-x0
    byte0, shift = x0//8, x0 % 8
    count = (width+7)//8
    crop = np.asarray(packed[z, y0:y1, byte0:byte0+count], dtype=np.uint8).copy()
    if shift:
        crop >>= np.uint8(shift)
        following = np.asarray(packed[z, y0:y1, byte0+1:byte0+count+1], dtype=np.uint8)
        crop[:, :following.shape[1]] |= following << np.uint8(8-shift)
    if width % 8:
        crop[:, -1] &= np.uint8((1 << (width % 8))-1)
    return RawBBoxSlicePayload(idx=int(z), is_empty=False, y0=y0, y1=y1, x0=x0, x1=x1,
                              payload_nbytes=(y1-y0)*width, payload=crop.tobytes(),
                              foreground_voxels=int(slice_counts[z]))


def project_azimuthal_sparse_store(
    source: RawBBoxMaskStore | Path,
    view: ViewInfo,
    store_dir: Path,
    *,
    out_shape_tyx: Tuple[int, int, int],
    workers: int = 1,
) -> Dict[str, object]:
    """Write one exact source-space packed CVOL; preserve caller-owned input.

    The result is published only after its writer closes successfully. ``workers``
    bounds independent output-slice encoding; sparse scatter uses one nogil kernel
    so overlapping source bits cannot race. GPU execution is not selected here.
    """
    started = time.perf_counter()
    if not is_azimuthal_view(view):
        raise ValueError('Sparse Azimuthal projection requires a Azimuthal view')
    output_shape = tuple(int(value) for value in out_shape_tyx)
    if len(output_shape) != 3 or min(output_shape) <= 0:
        raise ValueError('Sparse projection requires three positive output dimensions')
    target = Path(store_dir).resolve()
    if target.exists():
        raise FileExistsError(f'Immutable projected store already exists: {target}')
    own_input = not isinstance(source, RawBBoxMaskStore)
    store = RawBBoxMaskStore.open(Path(source), mmap_payload=True) if own_input else source
    packed = None
    try:
        if not all(int(value) > 0 for value in store.shape):
            raise ValueError('Sparse Azimuthal input requires three positive dimensions')
        if int(store.shape[0]) != int(view.num_slices) or int(store.shape[0]) != len(view.azimuths_deg):
            raise ValueError('Sparse Azimuthal depth differs from the view azimuths')
        if (target == store.root.resolve() or target in store.root.resolve().parents
                or store.root.resolve() in target.parents):
            raise ValueError('Projected store must not replace its input')
        if np.any((store.index['kind'] != 0) & (store.index['kind'] != 1)):
            raise ValueError('Invalid mask chunk marker')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'.{target.name}.projection-', dir=target.parent) as temporary:
            temporary = Path(temporary)
            staging = temporary/'projected.cvol'
            try:
                map_seconds = scatter_seconds = 0.0
                cache_hit = False
                map_bytes = input_bytes = foreground = contributions = unique = max_slab = 0
                out_t, out_h, out_w = output_shape
                packed_w = (out_w+7)//8
                bounds = np.tile(np.asarray([out_h, 0, out_w, 0], dtype=np.int32), (out_t, 1))
                slice_counts = np.zeros(out_t, dtype=np.uint64)
                if np.any(store.index['kind'] == 1):
                    map_started = time.perf_counter()
                    mapping, cache_hit = _inverse_map(view, tuple(store.shape), output_shape)
                    map_seconds = time.perf_counter()-map_started
                    map_bytes = mapping.nbytes
                    packed = np.memmap(temporary/'source.bits', mode='w+', dtype=np.uint8,
                                       shape=(out_t, out_h, packed_w))
                    flat = packed.reshape(-1)
                    scatter_started = time.perf_counter()
                    slabs = _input_slabs(store)
                    try:
                        for azimuth, row0, u0, crop, stored_bytes in slabs:
                            try:
                                positive, mapped, newly_set = _scatter_crop(
                                    crop, azimuth, row0, u0, mapping.key_offsets, mapping.owners,
                                    mapping.row_offsets, mapping.frames, mapping.shear,
                                    mapping.plane_shape[1], mapping.processing_width,
                                    mapping.base_id, mapping.tilted, mapping.vertical,
                                    *mapping.working_shape, *output_shape, flat, bounds, slice_counts,
                                )
                                foreground += int(positive)
                                contributions += int(mapped)
                                unique += int(newly_set)
                                input_bytes += int(stored_bytes)
                                max_slab = max(max_slab, int(crop.nbytes))
                            finally:
                                del crop
                    finally:
                        slabs.close()
                    scatter_seconds = time.perf_counter()-scatter_started
                    del flat
                encode_started = time.perf_counter()

                stats = dict(_write_raw_bbox_payload_store(
                    shape=output_shape, store_dir=staging,
                    encode_slice=lambda z: _packed_output_slice(z, packed, bounds, slice_counts),
                    format_name=INTERNAL_PACKED_CVOL_FORMAT,
                    desc=f'Sparse Azimuthal projection {view.name}', workers=int(workers),
                    extra_meta={'projection_payload_fusion': 'sparse_inverse_azimuthal_ownership'},
                ))
                encode_seconds = time.perf_counter()-encode_started
                if int(stats['foreground_voxels']) != unique:
                    raise RuntimeError('Packed projection foreground count differs from encoded store')
                if target.exists():
                    raise FileExistsError(f'Projected store appeared during publication: {target}')
                staging.rename(target)
                return {**stats, 'path': str(target), 'storage_format': INTERNAL_PACKED_CVOL_FORMAT,
                        'shape': output_shape, 'backend': 'cpu_numba' if hasattr(_scatter_crop, 'signatures') else 'cpu_python',
                        'map_cache_hit': cache_hit, 'map_bytes': map_bytes,
                        'input_payload_bytes': input_bytes, 'input_foreground_samples': foreground,
                        'projected_contributions': contributions, 'max_decoded_input_slab_bytes': max_slab,
                        'map_seconds': map_seconds, 'scatter_seconds': scatter_seconds,
                        'encode_seconds': encode_seconds, 'seconds': time.perf_counter()-started}
            finally:
                if packed is not None:
                    packed._mmap.close()
                    packed = None
    finally:
        if own_input:
            store.close()
