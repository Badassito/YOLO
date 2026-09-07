"""Bounded pull projection of radius-swept cylindrical patch trajectories.

This geometry is separate from the legacy Azimuthal diameter-plane projector.
Every source voxel chooses its nearest *global* shell, then reads every periodic
occurrence in this intrinsic patch. Distinct predictions in repeated wraps are
combined by OR rather than silently choosing one copy.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from .geometry import ViewInfo
from .runtime import allocate_workspace_array, close_memmap_array_without_flush

# Coordinate intermediates are bounded independently of source volume size.
_PULL_CHUNK_VOXELS = 128 * 1024


def _nearest_global_shell(radius: np.ndarray, radii: np.ndarray) -> np.ndarray:
    right = np.searchsorted(radii, radius, side='left')
    right = np.clip(right, 0, len(radii) - 1)
    left = np.maximum(right - 1, 0)
    # A midpoint belongs to the inner shell. This also handles exact endpoints.
    return np.where(radius - radii[left] <= radii[right] - radius, left, right)


def _processing_index(native: np.ndarray, native_count: int, processing_count: int) -> np.ndarray:
    if native_count == processing_count:
        return native
    return np.minimum(
        ((native.astype(np.float64) + 0.5) * processing_count / native_count).astype(np.int64),
        processing_count - 1,
    )


def _occurrence_rows(view, source_stack, radii, columns, stack_length):
    """Invert shear at the actual discrete shell sample, independently per wrap.

    Radius/arc quantization changes the in-plane point. Reusing the ideal source
    voxel's inverse-shear height can select zero padding across a source face.
    Height selection is global before patch offsets, so overlapping bands agree
    even at nearest-neighbor ties.
    """
    height = np.asarray(source_stack, dtype=np.float64)
    if bool(view.radial_tilted_source):
        theta = np.remainder((float(view.radial_arc_origin) + columns) / radii, 2.0 * math.pi)
        offset = radii * (np.sin(theta) if view.tilt_direction == 'vertical' else np.cos(theta))
        height = height - math.tan(math.radians(float(view.tilt_angle_deg))) * offset
    global_height = np.rint(np.clip(height, 0.0, float(stack_length - 1))).astype(np.int64)
    return global_height - int(view.radial_height_origin)


def _pull_radial_chunk(
    source: np.ndarray,
    view: ViewInfo,
    radii: np.ndarray,
    output_shape: Tuple[int, int, int],
    z: int,
    first: int,
    stop: int,
) -> np.ndarray:
    """Project one bounded, flattened source-coordinate XY strip."""
    out_t, out_h, out_w = output_shape
    work_t, work_h, work_w = int(view.full_t), int(view.full_h), int(view.full_w)
    flat = np.arange(first, stop, dtype=np.int64)
    wt = (float(z) + 0.5) * work_t / out_t - 0.5
    wy = (flat // out_w + 0.5) * work_h / out_h - 0.5
    wx = (flat % out_w + 0.5) * work_w / out_w - 0.5
    base = str(view.radial_base_view)
    if base == 'transverse':
        stack, py, px, stack_len = wt, wy, wx, work_t
    elif base == 'sagittal':
        stack, py, px, stack_len = wy, np.full(flat.shape, wt), wx, work_h
    elif base == 'coronal':
        stack, py, px, stack_len = wx, np.full(flat.shape, wt), wy, work_w
    else:
        raise ValueError(f'Unsupported Radial base {base!r}')
    dx, dy = px - float(view.center_x), py - float(view.center_y)
    radius = np.hypot(dx, dy)
    source_stack = np.broadcast_to(np.asarray(stack, dtype=np.float64), flat.shape)
    height = source_stack.copy()
    if bool(view.radial_tilted_source):
        direction = str(view.tilt_direction)
        if direction not in ('vertical', 'horizontal'):
            raise ValueError(f'Unsupported Radial tilt direction {direction!r}')
        axis = dy if direction == 'vertical' else dx
        height -= math.tan(math.radians(float(view.tilt_angle_deg))) * axis
    global_shell = _nearest_global_shell(radius, radii)
    shell = global_shell - int(view.radial_shell_start)
    valid = (
        (radius >= float(view.radial_min_radius))
        & (radius <= float(view.radial_max_radius))
        & (height >= 0.0) & (height <= float(stack_len - 1))
        & (shell >= 0) & (shell < source.shape[0])
    )
    result = np.zeros(flat.shape, dtype=np.uint8)
    positions = np.flatnonzero(valid)
    if not positions.size:
        return result
    local_shell = shell[positions]
    selected_radius = radii[global_shell[positions]]
    circumference = 2.0 * math.pi * selected_radius
    theta = np.mod(np.arctan2(dy[positions], dx[positions]), 2.0 * math.pi)
    width = int(view.src_w)

    # When a circumference is <= one native pixel, every native column contains
    # a nearest periodic occurrence. Radius zero similarly represents the axis.
    tiny = circumference <= 1.0
    if np.any(tiny):
        tiny_positions = positions[tiny]
        for column in range(width):
            rows = _occurrence_rows(
                view, source_stack[tiny_positions], selected_radius[tiny], column, stack_len,
            )
            inside = (rows >= 0) & (rows < int(view.src_h))
            if np.any(inside):
                proc_row = _processing_index(rows[inside], int(view.src_h), int(source.shape[1]))
                proc_col = int(_processing_index(np.asarray(column), width, int(source.shape[2])))
                result[tiny_positions[inside]] |= np.asarray(
                    source[local_shell[tiny][inside], proc_row, proc_col] != 0, dtype=np.uint8,
                )
    ordinary = ~tiny
    if not np.any(ordinary):
        return result
    positions = positions[ordinary]
    local_shell = local_shell[ordinary]
    selected_radius = selected_radius[ordinary]
    period = circumference[ordinary]
    arc = theta[ordinary] * selected_radius
    # Include an occurrence just below zero when it rounds onto column zero.
    column_float = np.mod(arc - float(view.radial_arc_origin) + 0.5, period) - 0.5
    while True:
        column = np.rint(column_float).astype(np.int64)
        active = (column >= 0) & (column < width)
        if not np.any(active):
            break
        rows = _occurrence_rows(view, source_stack[positions], selected_radius, column, stack_len)
        active &= (rows >= 0) & (rows < int(view.src_h))
        if np.any(active):
            proc_row = _processing_index(rows[active], int(view.src_h), int(source.shape[1]))
            proc_column = _processing_index(column[active], width, int(source.shape[2]))
            result[positions[active]] |= np.asarray(
                source[local_shell[active], proc_row, proc_column] != 0,
                dtype=np.uint8,
            )
        column_float += period
    return result


def backproject_radial_volume_to_volume(
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 0,
    workers: int = 1,
    out_shape_tyx: Optional[Tuple[int, int, int]] = None,
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    sink_only: bool = False,
):
    """Project one intrinsic Radial patch trajectory into source geometry.

    The output is caller-owned. Dense output uses path-backed storage, while an
    optional sink receives one completed source slice at a time. Neither path
    materializes a whole-volume coordinate map or changes the borrowed input.
    """
    from .geometry import radial_global_radii
    from .backprojection import SinkOnlyProjectionResult, _emit_projection_block_callback

    if str(radial_view.family) != 'radial':
        raise ValueError('Cylindrical shell projection requires a Radial view')
    source = np.asarray(radial_mask_mm)
    if source.ndim != 3 or min(source.shape) <= 0:
        raise ValueError('Radial projection requires a nonempty three-dimensional mask')
    if int(source.shape[0]) != int(radial_view.num_slices):
        raise ValueError('Radial mask depth differs from the shell trajectory')
    radii = np.asarray(radial_global_radii(radial_view), dtype=np.float64)
    start = int(radial_view.radial_shell_start)
    local_radii = np.asarray(radial_view.radial_radii, dtype=np.float64)
    if (
        radii.ndim != 1 or not radii.size or not np.isfinite(radii).all()
        or np.any(radii < 0) or np.any(np.diff(radii) <= 0)
        or start < 0 or start + source.shape[0] > radii.size
        or local_radii.shape != (source.shape[0],)
        or not np.array_equal(local_radii, radii[start:start + source.shape[0]])
    ):
        raise ValueError('Radial trajectory does not match its global radius grid')
    shape = tuple(int(v) for v in (
        out_shape_tyx if out_shape_tyx is not None
        else (radial_view.full_t, radial_view.full_h, radial_view.full_w)
    ))
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Radial projection requires positive source output dimensions')
    if sink_only and projection_block_callback is None:
        raise ValueError('Radial sink-only projection requires a block consumer')
    output = None
    if not sink_only:
        output = allocate_workspace_array(
            shape=shape, dtype=np.uint8, path=Path(out_path), desc=f'{desc} workspace',
            prefer_memory=False, prefer_memfd=False, reserve_bytes=int(reserve_bytes),
            initialize_zero=False,
        )
    try:
        for z in range(shape[0]):
            plane = np.empty(shape[1:], dtype=np.uint8) if output is None else output[z]
            flat_plane = plane.reshape(-1)
            for first in range(0, flat_plane.size, _PULL_CHUNK_VOXELS):
                stop = min(flat_plane.size, first + _PULL_CHUNK_VOXELS)
                flat_plane[first:stop] = _pull_radial_chunk(source, radial_view, radii, shape, z, first, stop)
            _emit_projection_block_callback(
                projection_block_callback, z, plane[None], desc=desc, required=bool(sink_only),
            )
        if output is not None:
            flush = getattr(output, 'flush', None)
            if callable(flush):
                flush()
            return output
        return SinkOnlyProjectionResult(shape)
    except BaseException:
        close_memmap_array_without_flush(output)
        raise
