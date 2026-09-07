"""Periodic cylindrical shell patches with radius as their slice direction.

Arc and height pixels have unit source-voxel spacing. The global radius grid
has gaps <= one voxel and includes both annular endpoints. A patch trajectory
keeps its arc/height origin across radii, so channels and interpolation cannot
cross into a different intrinsic patch. Tiling and in-plane transforms follow
native extraction through the ordinary geometry entry points.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .geometry import ViewInfo


def radius_grid(minimum: float, maximum: float) -> tuple[float, ...]:
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError('--radial_min_radius must be finite and strictly positive')
    if not math.isfinite(maximum) or maximum < minimum:
        raise ValueError(
            f'--radial_min_radius {minimum:g} exceeds the largest cylinder radius '
            f'{maximum:g}; reduce --radial_min_radius or --imgsz'
        )
    count = max(1, int(math.ceil(maximum - minimum)) + 1)
    return tuple(float(r) for r in np.linspace(minimum, maximum, count))


def global_radii(view: 'ViewInfo') -> tuple[float, ...]:
    return radius_grid(float(view.radial_min_radius), float(view.radial_max_radius))


def _height_starts(length: int, size: int) -> tuple[int, ...]:
    if length <= size:
        return (0,)
    # Overlap the last band instead of stretching the axial pixel spacing.
    starts = list(range(0, length - size + 1, size))
    if starts[-1] != length - size:
        starts.append(length - size)
    return tuple(starts)


def build_radial_view_infos(
    t: int, h: int, w: int, *, targets: Sequence[str], min_radius: float | None,
    patch_size: int, tilted_views: Sequence['ViewInfo'],
) -> list['ViewInfo']:
    from .config import AZIMUTHAL_VIEW_TOKENS, _resolve_unique_view_tokens
    from .geometry import ViewInfo, cartesian_view_axis_spec, tilted_base_view_name

    selected = _resolve_unique_view_tokens(
        targets, valid=AZIMUTHAL_VIEW_TOKENS, flag_name='Radial view assembly',
    )
    if not selected:
        return []
    if patch_size <= 0:
        raise ValueError('Radial shell patches require --imgsz > 0')
    if min(t, h, w) <= 0:
        raise ValueError('Radial geometry requires positive source dimensions')
    minimum = float(patch_size) / (4.0 * math.pi) if min_radius is None else float(min_radius)
    result = []
    for target in selected:
        base = target.removeprefix('tilted_')
        spec = cartesian_view_axis_spec(base, t, h, w)
        height = int(spec['num_slices'])
        plane_h, plane_w = int(spec['src_h']), int(spec['src_w'])
        maximum = (min(plane_h, plane_w) - 1) / 2.0
        radii = radius_grid(minimum, maximum)
        step = (maximum - minimum) / (len(radii) - 1) if len(radii) > 1 else 0.0
        if target.startswith('tilted_'):
            sources = [v for v in tilted_views if tilted_base_view_name(v) == base]
            if not sources:
                print(f'Radial target {target!r} skipped: no {base} Tilted variants are enabled. '
                      f'Add --enable_tilted {base}:30:both to generate it.')
                continue
        else:
            sources = [None]
        for source in sources:
            label = str(source.display_name) if source is not None else str(spec['display_name'])
            source_name = str(source.name) if source is not None else base
            for arc_index in range(max(1, int(math.ceil(2.0 * math.pi * maximum / patch_size)))):
                origin = arc_index * patch_size
                first = next(i for i, r in enumerate(radii) if 2.0 * math.pi * r > origin)
                for height_index, height_origin in enumerate(_height_starts(height, patch_size)):
                    name = f'radial_{source_name}_patch_u{arc_index}_h{height_index}'
                    result.append(ViewInfo(
                        name=name, family='radial', summary_family=name,
                        display_name=f'Radial {label} / Shell patch {arc_index},{height_index}',
                        num_slices=len(radii) - first, src_h=patch_size, src_w=patch_size,
                        pad_mode='pad', full_t=t, full_h=h, full_w=w,
                        center_x=(plane_w - 1) / 2.0, center_y=(plane_h - 1) / 2.0,
                        diameter=min(plane_h, plane_w), roi_radius=maximum,
                        horizontal_axis='azimuth', vertical_axis=str(spec['stack_axis']),
                        stack_axis='radius', tilt_base_view=base,
                        tilt_angle_deg=float(source.tilt_angle_deg) if source is not None else 0.0,
                        tilt_direction=str(source.tilt_direction) if source is not None else '',
                        tilt_frame_start=0, tilt_frame_stop=height - 1,
                        radial_base_view=base, radial_tilted_source=source is not None,
                        radial_source_view_name=source_name,
                        radial_request_token=target,
                        radial_min_radius=minimum, radial_max_radius=maximum,
                        radial_step=step, radial_shell_start=first, radial_radii=radii[first:],
                        radial_arc_origin=float(origin), radial_height_origin=height_origin,
                        radial_patch_size=patch_size, radial_patch_index=arc_index,
                        radial_height_index=height_index,
                    ))
    return result


def shell_coordinates(view: 'ViewInfo', index: int, x=None, y=None):
    """Return source (t,y,x,valid) for native shell-patch pixel centers.

    The optional coordinate arrays broadcast, supporting bounded row strips and
    independent geometry oracles. Periodicity belongs to arc coordinates, never
    to the radius/frame index. Height padding is zero; it never wraps.
    """
    if view.family != 'radial' or not 0 <= int(index) < len(view.radial_radii):
        raise ValueError(f'Invalid radial frame {index} for {view.name!r}')
    if x is None:
        x = np.arange(int(view.src_w), dtype=np.float64)[None, :]
    if y is None:
        y = np.arange(int(view.src_h), dtype=np.float64)[:, None]
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    radius = float(view.radial_radii[int(index)])
    theta = np.remainder((float(view.radial_arc_origin) + xx) / radius, 2.0 * math.pi)
    px = float(view.center_x) + radius * np.cos(theta)
    py = float(view.center_y) + radius * np.sin(theta)
    height = float(view.radial_height_origin) + yy
    stack = height
    if bool(view.radial_tilted_source):
        offset = py - float(view.center_y) if view.tilt_direction == 'vertical' else px - float(view.center_x)
        stack = stack + math.tan(math.radians(float(view.tilt_angle_deg))) * offset
    base = str(view.radial_base_view)
    if base == 'transverse':
        tt, sy, sx = stack, py, px
        height_length = view.full_t
    elif base == 'sagittal':
        tt, sy, sx = py, stack, px
        height_length = view.full_h
    elif base == 'coronal':
        tt, sy, sx = py, px, stack
        height_length = view.full_w
    else:
        raise ValueError(f'Unsupported Radial base {base!r}')
    valid = ((xx >= 0) & (xx <= view.src_w - 1) & (yy >= 0) & (yy <= view.src_h - 1)
             & (height >= 0) & (height <= height_length - 1)
             & (tt > -1) & (tt < view.full_t)
             & (sy > -1) & (sy < view.full_h)
             & (sx > -1) & (sx < view.full_w))
    # Keep fractional coordinates outside a source face: valid in-volume taps
    # still contribute to zero-extended trilinear interpolation. Clipping these
    # coordinates would repeat edge intensities; discarding the entire sample
    # would omit source voxels in thin tilted cylinders.
    return tt, sy, sx, valid


def render_shell_frame(volume: np.ndarray, view: 'ViewInfo', index: int, *, categorical=False):
    """Bounded native rendering: trilinear gray8 intensity or nearest categorical."""
    array = np.asarray(volume)
    if array.ndim != 3 or tuple(array.shape) != (view.full_t, view.full_h, view.full_w):
        raise ValueError('Radial source shape does not match physical view geometry')
    out = np.empty((int(view.src_h), int(view.src_w)), dtype=np.uint8)
    for row in range(0, int(view.src_h), 32):
        stop = min(int(view.src_h), row + 32)
        coords = shell_coordinates(view, index, y=np.arange(row, stop)[:, None])
        tt, yy, xx, valid = coords
        if categorical:
            ti, yi, xi = (np.floor(c + 0.5).astype(np.intp) for c in (tt, yy, xx))
            valid &= ((ti >= 0) & (ti < array.shape[0]) & (yi >= 0) & (yi < array.shape[1])
                      & (xi >= 0) & (xi < array.shape[2]))
            ti, yi, xi = (np.clip(a, 0, n - 1) for a, n in zip((ti, yi, xi), array.shape))
            out[row:stop] = np.asarray(valid & (array[ti, yi, xi] != 0), dtype=np.uint8)
            continue
        t0, y0, x0 = (np.floor(c).astype(np.intp) for c in (tt, yy, xx))
        dt, dy, dx = ((c - lower).astype(np.float32) for c, lower in zip((tt, yy, xx), (t0, y0, x0)))
        acc = np.zeros(tt.shape, dtype=np.float32)
        for it in (0, 1):
            wt = (dt if it else np.float32(1) - dt) * ((t0 + it >= 0) & (t0 + it < array.shape[0]))
            ti = np.clip(t0 + it, 0, array.shape[0] - 1)
            for iy in (0, 1):
                wy = (dy if iy else np.float32(1) - dy) * ((y0 + iy >= 0) & (y0 + iy < array.shape[1]))
                yi = np.clip(y0 + iy, 0, array.shape[1] - 1)
                for ix in (0, 1):
                    wx = (dx if ix else np.float32(1) - dx) * ((x0 + ix >= 0) & (x0 + ix < array.shape[2]))
                    xi = np.clip(x0 + ix, 0, array.shape[2] - 1)
                    acc += array[ti, yi, xi].astype(np.float32) * (wt * wy * wx)
        out[row:stop] = np.where(valid, np.clip(np.rint(acc), 0, 255), 0).astype(np.uint8)
    return out
