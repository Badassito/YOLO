"""Optional compiled Spherical pull prototype, independent of the NumPy oracle.

One scalar float64 loop replaces temporary coordinate/tap arrays. The existing
source-grid, shell midpoint, closed-face, global-pixel and padding decisions are
retained. Production selection requires an explicit opt-in flag. It owns only the
requested uint8 output strip, borrows the source, and uses no parallel Numba pool.
"""
from __future__ import annotations

import math

import numpy as np

from ._deps import _numba


_EMPTY_BOXES = np.empty((0, 4), dtype=np.int64)


class SphericalCpuProjectionUnavailable(RuntimeError):
    """Optional compilation failed before any source samples were evaluated."""


def _pull_spherical_f64(source, radii, rotation, boxes, use_boxes,
                        out_t, out_h, out_w, work_t, work_h, work_w,
                        face, intervals, origin_u, origin_v, native_h, native_w,
                        minimum, maximum, z, first, stop,
                        rectangle_width=0, rectangle_x=0, rectangle_y=0):
    result = np.zeros(stop - first, dtype=np.uint8)
    dz = ((float(z) + .5) * work_t / out_t - .5) - (work_t - 1) / 2.0
    source_h, source_w = source.shape[1], source.shape[2]
    for offset in range(stop - first):
        pixel = np.int64(first) + np.int64(offset)
        x, y = pixel % out_w, pixel // out_w
        if rectangle_width:
            x = pixel % rectangle_width + rectangle_x
            y = pixel // rectangle_width + rectangle_y
        dx = ((float(x) + .5) * work_w / out_w - .5) - (work_w - 1) / 2.0
        dy = ((float(y) + .5) * work_h / out_h - .5) - (work_h - 1) / 2.0
        radius = math.sqrt((dx * dx + dy * dy) + dz * dz)
        if radius < minimum or radius > maximum:
            continue
        lx = (dx * rotation[0, 0] + dy * rotation[1, 0]) + dz * rotation[2, 0]
        ly = (dx * rotation[0, 1] + dy * rotation[1, 1]) + dz * rotation[2, 1]
        lz = (dx * rotation[0, 2] + dy * rotation[1, 2]) + dz * rotation[2, 2]
        scale = max(abs(lx), abs(ly), abs(lz))
        if scale == 0.0:
            continue
        lx, ly, lz = lx / scale, ly / scale, lz / scale
        if face == 0:
            normal, right, up = lx, ly, lz
        elif face == 1:
            normal, right, up = ly, -lx, lz
        elif face == 2:
            normal, right, up = -lx, -ly, lz
        elif face == 3:
            normal, right, up = -ly, lx, lz
        elif face == 4:
            normal, right, up = lz, ly, -lx
        else:
            normal, right, up = -lz, ly, lx
        if normal <= 0.0 or normal + 1.7763568394002505e-15 < max(abs(right), abs(up)):
            continue
        norm = math.sqrt((lx * lx + ly * ly) + lz * lz)
        normal, right, up = normal / norm, right / norm, up / norm
        if abs(right) >= abs(up):
            if right >= 0.0:
                area, major, minor = 0, right, up
            else:
                area, major, minor = 2, -right, -up
        elif up >= 0.0:
            area, major, minor = 1, up, -right
        else:
            area, major, minor = 3, -up, right
        theta = math.atan2(minor, major)
        ratio = (12.0 / math.pi) * (theta - math.asin(math.sin(theta) * (1.0 / math.sqrt(2.0))))
        if abs(minor) == major:
            ratio = 1.0 if minor > 0.0 else (-1.0 if minor < 0.0 else 0.0)
        cosine = math.cos(theta)
        d = 1.0 - cosine / math.sqrt(1.0 + cosine * cosine)
        p = math.hypot(right, up) / math.sqrt((1.0 + normal) * d)
        if major == normal:
            p = 1.0
        m = p * ratio
        if area == 0:
            u, v = p, m
        elif area == 1:
            u, v = -m, p
        elif area == 2:
            u, v = -p, -m
        else:
            u, v = m, -p
        u, v = min(1., max(-1., u)), min(1., max(-1., v))
        column = np.int64(np.rint((u + 1.0) * intervals / 2.0)) - origin_u
        row = np.int64(np.rint((1.0 - v) * intervals / 2.0)) - origin_v
        if row < 0 or row >= native_h or column < 0 or column >= native_w:
            continue
        pr = row if native_h == source_h else min(np.int64((float(row) + .5) * source_h / native_h), source_h - 1)
        pc = column if native_w == source_w else min(np.int64((float(column) + .5) * source_w / native_w), source_w - 1)
        lo, hi = 0, len(radii)
        while lo < hi:
            middle = lo + (hi - lo) // 2
            if radii[middle] < radius:
                lo = middle + 1
            else:
                hi = middle
        outer = min(lo, len(radii) - 1)
        inner = max(outer - 1, 0)
        shell = inner if radius - radii[inner] <= radii[outer] - radius else outer
        if use_boxes and (pr < boxes[shell, 0] or pr >= boxes[shell, 1]
                          or pc < boxes[shell, 2] or pc >= boxes[shell, 3]):
            continue
        result[offset] = np.uint8(source[shell, pr, pc] != 0)
    return result


_compiled_pull_spherical_f64 = None
_dispatcher_unavailable_reason = 'Numba is unavailable'
if _numba is not None:
    try:
        _compiled_pull_spherical_f64 = _numba.njit(
            cache=True, nogil=True, fastmath=False)(_pull_spherical_f64)
    except Exception as exc:
        # Enabling Numba's disk cache can fail before compilation starts (for
        # example an installed module without a cache locator). No input has
        # been evaluated; preserve the optional-backend fallback boundary.
        _dispatcher_unavailable_reason = f'Numba initialization failed: {type(exc).__name__}: {exc}'


def _spherical_cpu_kernel_arguments(source, view, radii, rotation, output_shape, z, first, stop, bboxes):
    """Validate borrowed buffers/strip indices before unchecked compiled indexing."""
    source = np.asarray(source)
    if source.ndim != 3 or min(source.shape) <= 0 or source.dtype not in (np.uint8, np.bool_):
        raise ValueError('Compiled Spherical pull requires a nonempty uint8/bool mask')
    radii, rotation = np.asarray(radii, np.float64), np.asarray(rotation, np.float64)
    if radii.shape != (source.shape[0],) or rotation.shape != (3, 3):
        raise ValueError('Compiled Spherical geometry arrays have inconsistent shapes')
    out_t, out_h, out_w = map(int, output_shape)
    z, first, stop = int(z), int(first), int(stop)
    if min(out_t, out_h, out_w) <= 0 or not (0 <= z < out_t and 0 <= first <= stop <= out_h * out_w):
        raise ValueError('Compiled Spherical output strip is outside its source grid')
    boxes = _EMPTY_BOXES if bboxes is None else np.asarray(bboxes)
    if boxes.dtype != np.int64 or boxes.shape != ((0, 4) if bboxes is None else (len(radii), 4)):
        raise ValueError('Compiled Spherical bounding boxes must be int64 shell bounds')
    return (
        source, radii, rotation, boxes, bboxes is not None,
        out_t, out_h, out_w, int(view.full_t), int(view.full_h), int(view.full_w),
        int(view.spherical_face), int(view.spherical_face_intervals),
        int(view.spherical_u_origin), int(view.spherical_v_origin), int(view.src_h), int(view.src_w),
        float(view.spherical_min_radius), float(view.spherical_max_radius), z, first, stop, 0, 0, 0,
    )


def prepare_spherical_chunk_numba(source, view, radii, rotation, output_shape, bboxes=None):
    """Compile the precise buffer signature before any worker can publish output.

    Buffer/metadata errors remain ordinary errors. Only optional availability or
    compiler failures become a safe fallback signal; compilation samples no data.
    """
    if _compiled_pull_spherical_f64 is None:
        raise SphericalCpuProjectionUnavailable(_dispatcher_unavailable_reason)
    arguments = _spherical_cpu_kernel_arguments(source, view, radii, rotation, output_shape, 0, 0, 0, bboxes)
    try:
        signature = tuple(_numba.typeof(value) for value in arguments)
        _compiled_pull_spherical_f64.compile(signature)
    except Exception as exc:
        raise SphericalCpuProjectionUnavailable(f'Numba compilation failed: {type(exc).__name__}: {exc}') from exc
    return pull_spherical_chunk_numba


def pull_spherical_chunk_numba(source, view, radii, rotation, output_shape, z, first, stop, bboxes=None):
    """Evaluate one validated projection strip; never copy the borrowed mask.

    The projection owner validates the complete view/radius/bbox contract before
    dispatch. This helper additionally checks buffer types and strip bounds.
    Runtime numerical/data errors propagate and cannot silently select an oracle.
    """
    if _compiled_pull_spherical_f64 is None:
        raise SphericalCpuProjectionUnavailable(_dispatcher_unavailable_reason)
    arguments = _spherical_cpu_kernel_arguments(source, view, radii, rotation, output_shape, z, first, stop, bboxes)
    return _compiled_pull_spherical_f64(*arguments)


def pull_spherical_rectangle_numba(source, view, radii, rotation, output_shape, z, first, stop,
                                  bboxes=None, *, bounds_yx):
    """Pull a flattened rectangle while retaining global output voxel centers."""
    if _compiled_pull_spherical_f64 is None:
        raise SphericalCpuProjectionUnavailable(_dispatcher_unavailable_reason)
    arguments = _spherical_cpu_kernel_arguments(source, view, radii, rotation, output_shape, z, 0, 0, bboxes)
    y0, y1, x0, x1 = map(int, bounds_yx)
    first, stop = int(first), int(stop)
    if (not (0 <= y0 < y1 <= int(output_shape[1]) and 0 <= x0 < x1 <= int(output_shape[2]))
            or not 0 <= first <= stop <= (y1 - y0) * (x1 - x0)):
        raise ValueError('Compiled Spherical rectangle is outside its source grid')
    # The same fully explicit signature was compiled at admission. Geometry and
    # scalar floating-point arithmetic are shared with the original flat path.
    return _compiled_pull_spherical_f64(*arguments[:-5], first, stop, x1 - x0, x0, y0)


pull_spherical_chunk_numba.rectangle = pull_spherical_rectangle_numba
