"""Dense radius stacks on a rotated, six-face QSC atlas.

The unit-sphere chart is independent of the source-volume transform. Faces use
one endpoint-inclusive grid sized for the outer radius; directions and intrinsic
patch origins therefore remain fixed across radius channels and interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .qsc import QSC_FACE_NAMES, qsc_face_intervals, qsc_inverse

if TYPE_CHECKING:
    from .geometry import ViewInfo


@dataclass(frozen=True)
class SphericalSurface:
    """A unit QSC surface with a separate world pose; ellipsoid adapters fit here."""
    center_xyz: tuple[float, float, float]
    rotation_xyz: tuple[float, ...]

    def points(self, directions, radius):
        rotation = np.asarray(self.rotation_xyz, np.float64).reshape(3, 3)
        return np.asarray(self.center_xyz, np.float64) + float(radius) * (np.asarray(directions) @ rotation.T)


def radius_grid(minimum, maximum):
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError('--spherical_min_radius must be finite and strictly positive')
    if not math.isfinite(maximum) or maximum < minimum:
        raise ValueError(f'--spherical_min_radius {minimum:g} exceeds largest inscribed sphere radius {maximum:g}; '
                         'reduce --spherical_min_radius or --imgsz')
    count = max(1, int(math.ceil(maximum - minimum)) + 1)
    return tuple(float(value) for value in np.linspace(minimum, maximum, count))


def _patch_origins(length, size):
    if length <= size:
        return (-((size - length) // 2),)
    starts = list(range(0, length - size + 1, size))
    if starts[-1] != length - size:
        starts.append(length - size)
    return tuple(starts)


def cube_rotation(direction='', angle=0.0):
    """Canonical XYZ rotation independent of the requested Cartesian alias.

    Vertical rotates about +X (Y toward Z); horizontal rotates about -Y
    (X toward Z). Positive angles follow the existing tilted height convention
    at small angles, while the transform stays an orthonormal rotation.
    """
    c, s = math.cos(math.radians(float(angle))), math.sin(math.radians(float(angle)))
    if not direction:
        matrix = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    elif direction == 'vertical':
        matrix = ((1., 0., 0.), (0., c, -s), (0., s, c))
    elif direction == 'horizontal':
        matrix = ((c, 0., -s), (0., 1., 0.), (s, 0., c))
    else:
        raise ValueError(f'Unknown spherical cube tilt direction {direction!r}')
    return tuple(float(value) for row in matrix for value in row)


def build_spherical_view_infos(t: int, h: int, w: int, *, targets: Sequence[str],
                               min_radius: float | None, patch_size: int,
                               tilted_views: Sequence['ViewInfo']) -> list['ViewInfo']:
    from .config import SPHERICAL_VIEW_TOKENS, _resolve_unique_view_tokens
    from .geometry import ViewInfo, tilted_base_view_name

    requested = set(_resolve_unique_view_tokens(targets, valid=SPHERICAL_VIEW_TOKENS,
                                                 flag_name='Spherical view assembly'))
    if not requested:
        return []
    if patch_size <= 0:
        raise ValueError('Spherical QSC patches require --imgsz > 0')
    if min(t, h, w) <= 1:
        raise ValueError('Spherical views require at least two voxel centers on every source axis')
    minimum = float(patch_size) / (4.0 * math.pi) if min_radius is None else float(min_radius)
    maximum = (min(t, h, w) - 1) / 2.0
    radii = radius_grid(minimum, maximum)
    intervals = qsc_face_intervals(maximum)
    origins = _patch_origins(intervals + 1, patch_size)
    groups = {}
    upright = tuple(token for token in SPHERICAL_VIEW_TOKENS if token in requested and not token.startswith('tilted_'))
    if upright:
        groups[('', 0.0)] = set(upright)
    for target in SPHERICAL_VIEW_TOKENS:
        if target not in requested or not target.startswith('tilted_'):
            continue
        base = target.removeprefix('tilted_')
        sources = [view for view in tilted_views if tilted_base_view_name(view) == base]
        if not sources:
            print(f'Spherical target {target!r} skipped: no {base} Tilted variants are enabled. '
                  f'Add --enable_tilted {base}:30:both to generate it.')
        for view in sources:
            key = (str(view.tilt_direction), float(view.tilt_angle_deg))
            groups.setdefault(key, set()).add(target)
    result = []
    step = (maximum - minimum) / (len(radii) - 1) if len(radii) > 1 else 0.0
    for (direction, angle), aliases in sorted(groups.items()):
        token = ('p' if angle >= 0 else 'm') + format(abs(angle), '.12g').replace('.', 'p')
        group = f'{direction}_{token}' if direction else 'upright'
        rotation = cube_rotation(direction, angle)
        provenance = tuple(token for token in SPHERICAL_VIEW_TOKENS if token in aliases)
        for face, face_name in enumerate(QSC_FACE_NAMES):
            for iv, v0 in enumerate(origins):
                for iu, u0 in enumerate(origins):
                    name = f'spherical_{group}_{face_name.lower()}_patch_u{iu}_v{iv}'
                    result.append(ViewInfo(
                        name=name, family='spherical', summary_family=name,
                        display_name=f'Spherical QSC {group} / {face_name} / patch {iu},{iv}',
                        num_slices=len(radii), src_h=patch_size, src_w=patch_size,
                        pad_mode='pad', full_t=t, full_h=h, full_w=w,
                        diameter=min(t, h, w), roi_radius=maximum,
                        horizontal_axis='qsc_u', vertical_axis='qsc_v', stack_axis='radius',
                        tilt_angle_deg=angle, tilt_direction=direction,
                        spherical_face=face, spherical_group=group,
                        spherical_request_tokens=provenance, spherical_tilted_source=bool(direction),
                        spherical_min_radius=minimum, spherical_max_radius=maximum,
                        spherical_step=step, spherical_radii=radii,
                        spherical_face_intervals=intervals, spherical_patch_size=patch_size,
                        spherical_u_origin=u0, spherical_v_origin=v0,
                        spherical_patch_u=iu, spherical_patch_v=iv,
                        spherical_rotation_xyz=rotation,
                    ))
    return result


def face_directions(view: 'ViewInfo', x=None, y=None):
    """Return fixed world-frame unit XYZ directions and native patch validity."""
    if view.family != 'spherical' or view.spherical_face_intervals <= 0:
        raise ValueError('Spherical directions require a valid QSC view')
    xx = np.arange(view.src_w, dtype=np.float64)[None, :] if x is None else np.asarray(x, np.float64)
    yy = np.arange(view.src_h, dtype=np.float64)[:, None] if y is None else np.asarray(y, np.float64)
    xx, yy = np.broadcast_arrays(xx, yy)
    column, row = xx + view.spherical_u_origin, yy + view.spherical_v_origin
    n = int(view.spherical_face_intervals)
    valid = ((xx >= 0) & (xx <= view.src_w - 1) & (yy >= 0) & (yy <= view.src_h - 1)
             & (column >= 0) & (column <= n) & (row >= 0) & (row <= n))
    u, v = -1.0 + 2.0 * np.clip(column, 0, n) / n, 1.0 - 2.0 * np.clip(row, 0, n) / n
    directions = qsc_inverse(int(view.spherical_face), u, v)
    rotation = np.asarray(view.spherical_rotation_xyz, np.float64).reshape(3, 3)
    return directions @ rotation.T, valid


def shell_coordinates(view: 'ViewInfo', index: int, x=None, y=None):
    if not 0 <= int(index) < len(view.spherical_radii):
        raise ValueError(f'Invalid spherical frame {index} for {view.name!r}')
    directions, valid = face_directions(view, x, y)
    radius = float(view.spherical_radii[int(index)])
    center = np.asarray(((view.full_w - 1) / 2., (view.full_h - 1) / 2., (view.full_t - 1) / 2.))
    xyz = center + radius * directions
    xx, yy, tt = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    valid &= (tt > -1) & (tt < view.full_t) & (yy > -1) & (yy < view.full_h) & (xx > -1) & (xx < view.full_w)
    return tt, yy, xx, valid


def render_shell_frame(volume: np.ndarray, view: 'ViewInfo', index: int, *, categorical=False):
    """Zero-extended trilinear gray8 or nearest categorical native QSC input."""
    array = np.asarray(volume)
    if array.ndim != 3 or tuple(array.shape) != (view.full_t, view.full_h, view.full_w):
        raise ValueError('Spherical source shape does not match physical view geometry')
    out = np.empty((int(view.src_h), int(view.src_w)), dtype=np.uint8)
    for row in range(0, int(view.src_h), 32):
        stop = min(int(view.src_h), row + 32)
        tt, yy, xx, valid = shell_coordinates(view, index, y=np.arange(row, stop)[:, None])
        if categorical:
            ti, yi, xi = (np.floor(c + .5).astype(np.intp) for c in (tt, yy, xx))
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
