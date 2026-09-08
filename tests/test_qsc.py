"""CPU-only numerical invariants for the true equal-area QSC projection."""
from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from XTA.qsc import (
    QSC_FACE_BASES, QSC_FACE_NAMES, QSC_INVERSE_LIPSCHITZ,
    qsc_face_intervals, qsc_forward, qsc_forward_face, qsc_inverse,
)


class QSCMathTests(unittest.TestCase):
    def test_canonical_frames_and_exact_centers(self):
        bases = np.asarray(QSC_FACE_BASES)
        self.assertEqual(QSC_FACE_NAMES, ('PX', 'PY', 'NX', 'NY', 'PZ', 'NZ'))
        np.testing.assert_array_equal(np.cross(bases[:, 1], bases[:, 2]), bases[:, 0])
        np.testing.assert_array_equal(qsc_inverse(np.arange(6), 0, 0), bases[:, 0])
        face, u, v = qsc_forward(bases[:, 0])
        np.testing.assert_array_equal(face, np.arange(6))
        np.testing.assert_array_equal(u, np.zeros(6))
        np.testing.assert_array_equal(v, np.zeros(6))

    def test_direction_roundtrip_and_scale_invariance(self):
        rng = np.random.default_rng(721)
        directions = rng.normal(size=(11, 37, 3))
        unit = directions / np.linalg.norm(directions, axis=-1)[..., None]
        face, u, v = qsc_forward(directions)
        self.assertTrue(np.all((np.abs(u) <= 1) & (np.abs(v) <= 1)))
        np.testing.assert_allclose(qsc_inverse(face, u, v), unit, atol=8e-16, rtol=0)
        for magnitude in (1e-300, 1e300):
            other_face, other_u, other_v = qsc_forward(directions * magnitude)
            np.testing.assert_array_equal(other_face, face)
            np.testing.assert_allclose(other_u, u, atol=6e-16, rtol=0)
            np.testing.assert_allclose(other_v, v, atol=6e-16, rtol=0)

    def test_face_roundtrip_all_sectors_and_broadcast(self):
        uv = np.linspace(-.99, .99, 43)
        face = np.arange(6)[:, None, None]
        xyz = qsc_inverse(face, uv[None, None, :], uv[None, :, None])
        self.assertEqual(xyz.shape, (6, 43, 43, 3))
        other_face, u, v = qsc_forward(xyz)
        np.testing.assert_array_equal(other_face, np.broadcast_to(face, u.shape))
        np.testing.assert_allclose(u, np.broadcast_to(uv[None, None, :], u.shape), atol=1e-15, rtol=0)
        np.testing.assert_allclose(v, np.broadcast_to(uv[None, :, None], v.shape), atol=1e-15, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=-1), 1, atol=4e-16, rtol=0)

    def test_subresolution_offsets_survive_both_transforms(self):
        offsets = np.array([1e-300, 1e-100, 1e-20, 1e-12, 1e-8])
        face = np.arange(6)[:, None]
        xyz = qsc_inverse(face, offsets, -.3 * offsets)
        actual_face, u, v = qsc_forward(xyz)
        np.testing.assert_array_equal(actual_face, np.broadcast_to(face, u.shape))
        np.testing.assert_allclose(u, np.broadcast_to(offsets, u.shape), rtol=2e-15, atol=0)
        np.testing.assert_allclose(v, np.broadcast_to(-.3 * offsets, v.shape), rtol=3e-15, atol=0)

    def test_exact_edge_and_corner_ties_choose_lowest_face_id(self):
        normals = np.asarray(QSC_FACE_BASES)[:, 0]
        vectors = np.array([v for v in itertools.product((-1., 0., 1.), repeat=3) if any(v)])
        expected = np.argmax(vectors @ normals.T, axis=-1)
        face, u, v = qsc_forward(vectors)
        np.testing.assert_array_equal(face, expected)
        np.testing.assert_allclose(qsc_inverse(face, u, v),
                                   vectors / np.linalg.norm(vectors, axis=-1)[:, None],
                                   atol=2e-16, rtol=0)
        for original in range(6):
            for edge in (-1., 1.):
                uv = np.linspace(-1, 1, 25)
                for uu, vv in ((uv, edge), (edge, uv)):
                    xyz = qsc_inverse(original, uu, vv)
                    face, u, v = qsc_forward(xyz)
                    np.testing.assert_array_equal(face, np.argmax(xyz @ normals.T, axis=-1))
                    np.testing.assert_allclose(qsc_inverse(face, u, v), xyz, atol=6e-16, rtol=0)
                    # Each exact cube edge has at least two equal maximal components.
                    scores = xyz @ normals.T
                    self.assertTrue(np.all(np.sum(scores == np.max(scores, axis=-1)[:, None], axis=-1) >= 2))

    def test_analytic_mid_edges_and_corners(self):
        edge = qsc_inverse(0, np.array([1., -1., 0., 0.]), np.array([0., 0., 1., -1.]))
        expected = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1]]) / math.sqrt(2)
        np.testing.assert_allclose(edge, expected, atol=2e-16, rtol=0)
        corner = qsc_inverse(0, np.array([1., -1., 1., -1.]), np.array([1., 1., -1., -1.]))
        expected = np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1]]) / math.sqrt(3)
        np.testing.assert_allclose(corner, expected, atol=2e-16, rtol=0)

    def test_closed_faces_include_all_incident_edges_and_corners(self):
        directions = np.array([v for v in itertools.product((-1., 0., 1.), repeat=3) if any(v)])
        faces = np.arange(6)[:, None]
        u, v, valid = qsc_forward_face(directions, faces)
        expected_count = np.count_nonzero(directions, axis=-1)
        np.testing.assert_array_equal(np.count_nonzero(valid, axis=0), expected_count)
        expected = np.broadcast_to(directions / np.linalg.norm(directions, axis=-1)[:, None], (6, 26, 3))
        np.testing.assert_allclose(qsc_inverse(faces, u, v)[valid], expected[valid], atol=2e-16, rtol=0)
        np.testing.assert_array_equal(u[~valid], 0)
        np.testing.assert_array_equal(v[~valid], 0)
        _, _, zero_valid = qsc_forward_face([0, 0, 0], np.arange(6))
        self.assertFalse(np.any(zero_valid))

    def test_closed_face_tolerance_only_covers_rotation_roundoff(self):
        rng = np.random.default_rng(973)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        directions = np.array([[1., 1., 0.], [1., 1., 1.], [-1., -1., 1.]])
        recovered = (directions @ rotation.T) @ rotation
        _, _, before = qsc_forward_face(directions, np.arange(6)[:, None])
        _, _, after = qsc_forward_face(recovered, np.arange(6)[:, None])
        np.testing.assert_array_equal(after, before)
        _, _, outside = qsc_forward_face([1., 1. + 1e-8, 0.], np.arange(6))
        np.testing.assert_array_equal(outside, [False, True, False, False, False, False])

    def test_local_area_is_pi_over_six(self):
        # An equal-area face maps a square of area 4 onto sphere area 2*pi/3.
        # Check away from fold boundaries; a gnomonic or COBE approximation
        # cannot pass this uniform differential-area invariant.
        u = np.array([.08, .31, -.73, .81, -.43, -.62, .17])
        v = np.array([.21, -.12, .21, .49, -.57, -.84, -.89])
        epsilon = 1e-6
        for face in range(6):
            du = (qsc_inverse(face, u + epsilon, v) - qsc_inverse(face, u - epsilon, v)) / (2 * epsilon)
            dv = (qsc_inverse(face, u, v + epsilon) - qsc_inverse(face, u, v - epsilon)) / (2 * epsilon)
            area = np.linalg.norm(np.cross(du, dv), axis=-1)
            np.testing.assert_allclose(area, math.pi / 6, atol=2e-10, rtol=0)

    def test_global_lipschitz_bound_across_sectors(self):
        rng = np.random.default_rng(557)
        a = rng.uniform(-1, 1, size=(2000, 2))
        b = rng.uniform(-1, 1, size=(2000, 2))
        a[:4] = [[0, 0], [-1, -1], [-1, 1], [1, 0]]
        b[:4] = [[1e-12, -1e-12], [1, 1], [1, -1], [0, -1]]
        distance = np.linalg.norm(qsc_inverse(0, *a.T) - qsc_inverse(0, *b.T), axis=-1)
        bound = QSC_INVERSE_LIPSCHITZ * np.linalg.norm(a - b, axis=-1)
        self.assertTrue(np.all(distance <= bound))

    def test_interval_count_is_even_monotone_and_meets_proof(self):
        radii = np.geomspace(.0001, 1000, 311)
        counts = np.array([qsc_face_intervals(r) for r in radii])
        self.assertTrue(np.all(counts >= 2))
        self.assertTrue(np.all(counts % 2 == 0))
        self.assertTrue(np.all(np.diff(counts) >= 0))
        self.assertTrue(np.all(counts >= 3 * np.sqrt(radii * (radii + .5))))
        squared_bound = .25 + radii * (radii + .5) * 2 * QSC_INVERSE_LIPSCHITZ**2 / counts**2
        self.assertTrue(np.all(squared_bound <= 281 / 324))

    def test_voxel_centers_have_positive_trilinear_footprint_witness(self):
        # Verify actual emitted lattice coordinates for odd/even and thin
        # volumes, fractional annulus endpoints and per-shell resolutions.
        for shape in ((5, 7, 9), (8, 10, 12), (3, 9, 9), (13, 13, 13)):
            points = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1)
            points -= (np.asarray(shape) - 1) / 2
            points = points.reshape(-1, 3)
            rho = np.linalg.norm(points, axis=-1)
            maximum = (min(shape) - 1) / 2
            for minimum in (.1, .71, maximum):
                mask = (rho >= minimum) & (rho <= maximum)
                points_in_annulus, radii = points[mask], rho[mask]
                shells = np.linspace(minimum, maximum, max(1, math.ceil(maximum - minimum) + 1))
                nearest = np.argmin(np.abs(radii[:, None] - shells), axis=-1)
                face, u, v = qsc_forward(points_in_annulus)
                for index, shell in enumerate(shells):
                    selected = nearest == index
                    n = qsc_face_intervals(shell)
                    sample_u = -1 + 2 * np.rint((u[selected] + 1) * n / 2) / n
                    sample_v = -1 + 2 * np.rint((v[selected] + 1) * n / 2) / n
                    samples = shell * qsc_inverse(face[selected], sample_u, sample_v)
                    delta = np.abs(samples - points_in_annulus[selected])
                    self.assertTrue(np.all(np.sum(delta * delta, axis=-1) <= 281 / 324 + 1e-14))
                    self.assertTrue(np.all(np.prod(1 - delta, axis=-1) > 0))

    def test_invalid_inputs_and_empty_batches(self):
        for vector in ([0, 0, 0], [math.inf, 1, 1], [math.nan, 0, 0], [1, 2], 1):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                qsc_forward(vector)
        for face in (-1, 6, .5, 0., True, '0'):
            with self.subTest(face=face), self.assertRaises(ValueError):
                qsc_inverse(face, 0, 0)
            with self.subTest(forward_face=face), self.assertRaises(ValueError):
                qsc_forward_face([1, 1, 1], face)
        for u, v in ((1.01, 0), (0, -1.01), (math.nan, 0), (0, math.inf)):
            with self.subTest(uv=(u, v)), self.assertRaises(ValueError):
                qsc_inverse(0, u, v)
        for radius in (0, -1, math.nan, math.inf, 1.7e308):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                qsc_face_intervals(radius)
        face, u, v = qsc_forward(np.empty((0, 3)))
        self.assertEqual(qsc_inverse(face, u, v).shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
