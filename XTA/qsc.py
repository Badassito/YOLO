"""True equal-area Quadrilateralized Spherical Cube coordinates.

These are the O'Neill--Laubscher (1976) equations used by PROJ's ``qsc``,
expressed directly in Cartesian face frames. They are neither gnomonic cube
coordinates nor the approximate COBE projection. References:
https://proj.org/en/stable/operations/projections/qsc.html
https://github.com/OSGeo/PROJ/blob/master/src/projections/qsc.cpp

Directions are XYZ, independently of a volume's array-axis order. A future
ellipsoid adapter can transform coordinates around this unit-sphere interface;
the coverage guarantee below currently applies to spherical voxel coordinates.

Sampling bound
--------------
The inverse map S from one normalized face to the unit sphere is globally
Lipschitz with conservative Euclidean constant L = 5/3. To see this, rotate
each of the four face sectors to u=p>=|v|, set t=v/p, and write

    k=pi*t/12; theta=atan(sin(k)/(cos(k)-1/sqrt(2)))
    D=1-cos(theta)/sqrt(1+cos(theta)**2); H=sqrt(D*(2-p*p*D))
    cos(phi)=1-p*p*D.

Here |t|<=1, 0<=p<=1, D is between 1-1/sqrt(2) and
1-1/sqrt(3), H**2>=1/2, and theta'(t)<0.9. Differentiating gives
D_theta=sin(theta)/(1+cos(theta)**2)**(3/2), |D_theta|<0.385,

    phi_u=(2*D-t*D_theta*theta')/H; phi_v=D_theta*theta'/H
    sin(phi)*theta_u=-t*H*theta'; sin(phi)*theta_v=H*theta'.

The subtracted term in phi_u is nonnegative and smaller than 2*D.
Consequently phi_u**2<1.08, phi_v**2<0.241, and each of the last
two squared derivatives is <0.685. Since the spherical metric is
dphi**2+sin(phi)**2*dtheta**2, the Jacobian's squared Frobenius norm
is <2.691<25/9. Sector rotations preserve this bound, continuity extends
it to their boundaries and the center, and integrating along a face segment
proves the stated global bound. This is an analytic bound, not a grid probe.

An endpoint-inclusive grid of n intervals per face puts any direction within
L*sqrt(2)/n of a sample direction. If consecutive shell radii have gaps <=1
and include both annulus endpoints, any source voxel center of radius rho
has a shell r with |rho-r|<=1/2. Choosing even n>=3*sqrt(r*(r+1/2)) yields

    ||rho*d-r*d_sample||**2
      = (rho-r)**2 + rho*r*||d-d_sample||**2
      <= 1/4 + 50/81 = 281/324 < 1.

Thus every voxel center in that annulus is strictly inside some sample's
trilinear footprint: all three coordinate differences are <1 and its weight
is positive. This does not claim nearest-neighbor categorical coverage, or
coverage after downsampling the proven native input grid.
"""
from __future__ import annotations

import math

import numpy as np


QSC_FACE_NAMES = ('PX', 'PY', 'NX', 'NY', 'PZ', 'NZ')
# Each row contains (normal, image-right, image-up), with right x up = normal.
# Face IDs and orientations agree with PROJ front/right/back/left/top/bottom.
QSC_FACE_BASES = (
    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
    ((0., 1., 0.), (-1., 0., 0.), (0., 0., 1.)),
    ((-1., 0., 0.), (0., -1., 0.), (0., 0., 1.)),
    ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)),
    ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.)),
    ((0., 0., -1.), (0., 1., 0.), (1., 0., 0.)),
)
QSC_INVERSE_LIPSCHITZ = 5.0 / 3.0
_BASES = np.asarray(QSC_FACE_BASES, dtype=np.float64)
_BASES.flags.writeable = False
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _fold(x: np.ndarray, y: np.ndarray):
    """Rotate a square quadrant into its east-facing sector without trig."""
    horizontal = np.abs(x) >= np.abs(y)
    area = np.where(horizontal, np.where(x >= 0, 0, 2), np.where(y >= 0, 1, 3))
    major = np.maximum(np.abs(x), np.abs(y))
    minor = np.select((area == 0, area == 1, area == 2), (y, -x, -y), default=x)
    return area, major, minor


def _unfold(area: np.ndarray, major: np.ndarray, minor: np.ndarray):
    x = np.select((area == 0, area == 1, area == 2), (major, -minor, -major), default=minor)
    y = np.select((area == 0, area == 1, area == 2), (minor, major, -minor), default=-major)
    return x, y


def _project_local(normal: np.ndarray, right: np.ndarray, up: np.ndarray):
    area, major, minor = _fold(right, up)
    theta = np.arctan2(minor, major)
    ratio = (12.0 / math.pi) * (theta - np.arcsin(np.sin(theta) * _INV_SQRT2))
    # A sector diagonal has exactly equal coordinates, including at corners.
    ratio = np.where(np.abs(minor) == major, np.sign(minor), ratio)
    cosine = np.cos(theta)
    d = 1.0 - cosine / np.sqrt(1.0 + cosine * cosine)
    # sqrt(1-normal) = hypot(right,up)/sqrt(1+normal) retains tiny angles.
    p = np.hypot(right, up) / np.sqrt((1.0 + normal) * d)
    p = np.where(major == normal, 1.0, p)
    u, v = _unfold(area, p, p * ratio)
    return np.clip(u, -1.0, 1.0), np.clip(v, -1.0, 1.0)


def _face_ids(face):
    raw_face = np.asarray(face)
    if raw_face.dtype.kind not in 'iu' or np.any((raw_face < 0) | (raw_face >= 6)):
        raise ValueError('QSC face IDs must be integers from 0 through 5')
    return raw_face.astype(np.intp, copy=False)


def qsc_forward(xyz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map nonzero ``(..., 3)`` directions to ``(face, u, v)`` arrays.

    u and v lie in [-1,1], with v increasing toward image-up. Face ownership
    chooses the greatest signed normal component, with the lowest face ID
    winning exact edge/corner ties. Inputs need not be normalized. Nonfinite
    vectors and the origin are rejected; callers handle radius zero explicitly.
    Work and storage are proportional to the supplied batch, with no cache.
    """
    vectors = np.asarray(xyz, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError('QSC directions must have shape (..., 3)')
    if not np.all(np.isfinite(vectors)):
        raise ValueError('QSC directions must be finite')
    scale = np.max(np.abs(vectors), axis=-1)
    if np.any(scale == 0):
        raise ValueError('QSC directions must be nonzero')
    # Scaling before normalization avoids overflow and underflow of the norm.
    vectors = vectors / scale[..., None]
    face = np.argmax(vectors @ _BASES[:, 0, :].T, axis=-1)
    vectors = vectors / np.linalg.norm(vectors, axis=-1)[..., None]
    basis = _BASES[face]
    normal = np.sum(vectors * basis[..., 0, :], axis=-1)
    right = np.sum(vectors * basis[..., 1, :], axis=-1)
    up = np.sum(vectors * basis[..., 2, :], axis=-1)
    u, v = _project_local(normal, right, up)
    return face, u, v


def qsc_forward_face(xyz, face) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(u,v,valid)`` on a specified closed face, with broadcasting.

    Every incident face accepts an exact edge or corner; this permits OR
    publication from all face predictions. Membership allows eight float64
    epsilons relative to the largest vector component, solely to retain ties
    after a floating-point frame rotation. Accepted roundoff excursions are
    clipped to the closed square. Zero vectors and nonmembers return zero
    coordinates with valid=False. Nonfinite or malformed inputs are rejected.
    """
    vectors = np.asarray(xyz, dtype=np.float64)
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError('QSC directions must have shape (..., 3)')
    if not np.all(np.isfinite(vectors)):
        raise ValueError('QSC directions must be finite')
    faces = _face_ids(face)
    shape = np.broadcast_shapes(vectors.shape[:-1], faces.shape)
    vectors = np.broadcast_to(vectors, shape + (3,))
    basis = _BASES[np.broadcast_to(faces, shape)]
    scale = np.max(np.abs(vectors), axis=-1)
    vectors = np.divide(vectors, scale[..., None], out=np.zeros_like(vectors), where=scale[..., None] != 0)
    normal = np.sum(vectors * basis[..., 0, :], axis=-1)
    right = np.sum(vectors * basis[..., 1, :], axis=-1)
    up = np.sum(vectors * basis[..., 2, :], axis=-1)
    tolerance = 8.0 * np.finfo(np.float64).eps
    valid = ((normal > 0) & (normal + tolerance >= np.maximum(np.abs(right), np.abs(up))))
    norm = np.linalg.norm(vectors, axis=-1)
    norm = np.where(valid, norm, 1.0)
    u, v = _project_local(np.where(valid, normal, 1.0) / norm,
                          np.where(valid, right, 0.0) / norm,
                          np.where(valid, up, 0.0) / norm)
    return u, v, valid


def qsc_inverse(face, u, v) -> np.ndarray:
    """Map broadcast ``face,u,v`` arrays to unit ``(..., 3)`` XYZ vectors.

    Face IDs are integers 0..5. Coordinates must be finite and in [-1,1].
    Face centers return their normals exactly; shared edge/corner coordinates
    retain exact component ties for deterministic subsequent ownership.
    """
    faces, uu, vv = np.broadcast_arrays(
        _face_ids(face),
        np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64),
    )
    if not np.all(np.isfinite(uu) & np.isfinite(vv)):
        raise ValueError('QSC face coordinates must be finite')
    if np.any((np.abs(uu) > 1.0) | (np.abs(vv) > 1.0)):
        raise ValueError('QSC face coordinates must be in [-1, 1]')
    area, p, minor = _fold(uu, vv)
    ratio = np.divide(minor, p, out=np.zeros_like(p), where=p != 0)
    tau = (math.pi / 12.0) * ratio
    tangent_numerator = np.sin(tau)
    tangent_denominator = np.cos(tau) - _INV_SQRT2
    theta_norm = np.hypot(tangent_numerator, tangent_denominator)
    cosine = tangent_denominator / theta_norm
    sine = tangent_numerator / theta_norm
    diagonal = np.abs(ratio) == 1.0
    cosine = np.where(diagonal, _INV_SQRT2, cosine)
    sine = np.where(diagonal, np.sign(ratio) * _INV_SQRT2, sine)
    d = 1.0 - cosine / np.sqrt(1.0 + cosine * cosine)
    normal = 1.0 - p * p * d
    # Unlike sqrt(1-normal**2), this preserves arbitrarily small offsets.
    tangent_radius = p * np.sqrt(d * (2.0 - p * p * d))
    major = tangent_radius * cosine
    minor = tangent_radius * sine
    major = np.where(p == 1.0, normal, major)
    minor = np.where((p == 1.0) & diagonal, np.sign(ratio) * normal, minor)
    right, up = _unfold(area, major, minor)
    basis = _BASES[faces]
    return (normal[..., None] * basis[..., 0, :]
            + right[..., None] * basis[..., 1, :]
            + up[..., None] * basis[..., 2, :])


def qsc_face_intervals(radius: float) -> int:
    """Even face-grid interval count proving annular trilinear coverage.

    Use n+1 sample centers per side, including -1 and +1. Shell gaps must
    be <=1 and both annulus endpoints must be present; see the module proof.
    Counts can be computed from a common outer radius to keep face grids fixed.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('QSC sampling radius must be finite and positive')
    # Separate roots avoid overflowing radius*(radius+0.5).
    half_intervals = 1.5 * math.sqrt(radius) * math.sqrt(radius + 0.5)
    if not math.isfinite(half_intervals):
        raise ValueError('QSC sampling radius is too large')
    return 2 * max(1, int(math.ceil(half_intervals)))
