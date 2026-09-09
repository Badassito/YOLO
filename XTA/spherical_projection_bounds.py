"""Conservative source-grid bounds for a spherical face's categorical pull.

The face is a closed polyhedral cone: in its local frame n >= abs(right)
and n >= abs(up). A coordinate's extrema on the cone's unit sphere occur
in its interior, on a great-circle edge, or at the intersection of two
edges. Testing the coordinate axis, its orthogonal projections onto each
edge plane, and the pairwise plane intersections therefore gives the exact
full-face directional bounds in real arithmetic. No sampled chart bounds
or foreground-density assumptions are used.

The CUDA/CPU face test admits a few ulps around incident edges. Outward
rounding below includes that tolerance, arithmetic in the near-orthogonal
rotation, and voxel-center restoration. These are broad-phase bounds only:
the existing scalar/CUDA pull retains every final radius, face, patch and
nearest-neighbor test.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .qsc import QSC_FACE_BASES


_ROUNDING_MARGIN = 4096 * np.finfo(np.float64).eps


@dataclass(frozen=True)
class SphericalOutputBounds:
    """Half-open bounds in the caller's restored source Z/Y/X grid."""

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def voxel_count(self):
        return (self.z1 - self.z0) * (self.y1 - self.y0) * (self.x1 - self.x0)

    def block(self, first_z, count):
        first = max(int(first_z), self.z0)
        stop = min(int(first_z) + int(count), self.z1)
        return (first, max(first, stop), self.y0, self.y1, self.x0, self.x1)


def _face_direction_bounds(rotation, face):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    normal, right, up = (rotation @ np.asarray(axis) for axis in QSC_FACE_BASES[int(face)])
    planes = np.asarray((normal - right, normal + right, normal - up, normal + up))
    planes /= np.linalg.norm(planes, axis=1)[:, None]
    candidates = []

    def include(direction):
        length = float(np.linalg.norm(direction))
        if length <= _ROUNDING_MARGIN:
            return
        unit = direction / length
        # A loose feasibility test can only enlarge the extrema candidate set.
        if np.all(planes @ unit >= -_ROUNDING_MARGIN):
            candidates.append(unit)

    for axis in np.eye(3):
        for sign in (-1, 1):
            direction = sign * axis
            include(direction)
            for plane in planes:
                include(direction - np.dot(direction, plane) * plane)
    for first, second in combinations(planes, 2):
        intersection = np.cross(first, second)
        include(intersection)
        include(-intersection)
    if not candidates:
        raise ValueError('Spherical face cone has no finite directional extrema')
    points = np.asarray(candidates)
    return (np.maximum(-1., points.min(axis=0) - _ROUNDING_MARGIN),
            np.minimum(1., points.max(axis=0) + _ROUNDING_MARGIN))


def spherical_output_bounds(view, output_shape, bboxes=None):
    """Enclose every possible contribution without inspecting mask pixels.

    Supplied bounding boxes can prove outer shells empty. The surviving radius
    interval includes both adjacent nearest-shell midpoints, including inward
    midpoint ties. Interior empty shells are conservatively retained.
    """
    shape = tuple(map(int, output_shape))
    work = (int(view.full_t), int(view.full_h), int(view.full_w))
    radii = np.asarray(view.spherical_radii, dtype=np.float64)
    lower, upper = float(view.spherical_min_radius), float(view.spherical_max_radius)
    if bboxes is not None:
        boxes = np.asarray(bboxes)
        nonempty = np.flatnonzero((boxes[:, 1] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 2]))
        if not len(nonempty):
            return SphericalOutputBounds(0, 0, 0, 0, 0, 0)
        first, last = int(nonempty[0]), int(nonempty[-1])
        if first:
            lower = max(lower, float(radii[first - 1] + (radii[first] - radii[first - 1]) * .5))
        if last + 1 < len(radii):
            upper = min(upper, float(radii[last] + (radii[last + 1] - radii[last]) * .5))
    direction_min, direction_max = _face_direction_bounds(view.spherical_rotation_xyz, view.spherical_face)
    # Validated projection rotations are near orthogonal and dimensions fit
    # int32. A 4096-epsilon envelope is much wider than the sum of rounding
    # errors in plane construction, projection, face tolerance and restoration.
    # Scaling by the working coordinate extent also covers center subtraction.
    margin = _ROUNDING_MARGIN * max(1, *work)
    lower, upper = max(0., lower - margin), upper + margin
    limits = []
    for axis, native_count, out_count in zip((2, 1, 0), work, shape):
        products = (lower * direction_min[axis], lower * direction_max[axis],
                    upper * direction_min[axis], upper * direction_max[axis])
        lo = (min(products) - margin + native_count / 2.) * out_count / native_count - .5
        hi = (max(products) + margin + native_count / 2.) * out_count / native_count - .5
        # Enlarge by an additional output voxel beyond floor/ceil. This keeps
        # exact boundary centers despite cancellation next to integer indices.
        start = max(0, min(out_count, math.floor(np.nextafter(lo, -np.inf)) - 1))
        stop = max(start, min(out_count, math.ceil(np.nextafter(hi, np.inf)) + 2))
        limits.extend((start, stop))
    return SphericalOutputBounds(*limits)
