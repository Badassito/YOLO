"""Compact QSC broadcast and repeated native-frame validation regressions."""
from __future__ import annotations

from itertools import product
import importlib.util
import unittest
from unittest import mock

import numpy as np

from XTA import qsc, spherical_cuda
from tests.test_spherical_cuda import resident_engine, rotation_xyz, spherical_view


class CompactQSCGeometryTests(unittest.TestCase):
    def test_inverse_compact_faces_are_bitwise_equal_to_dense_face_ids(self):
        coordinates = np.array([-1., -.75, -.01, -1e-300, -0., 0., 1e-300, .01, .75, 1.])
        u, v = coordinates[None, :], coordinates[:, None]
        faces = [*range(6), np.arange(6)[:, None, None], np.array([0, 2, 4])[:, None, None, None]]
        for face in faces:
            shape = np.broadcast_shapes(np.shape(face), u.shape, v.shape)
            compact = qsc.qsc_inverse(face, u, v)
            dense = qsc.qsc_inverse(np.broadcast_to(face, shape), u, v)
            with self.subTest(face_shape=np.shape(face)):
                np.testing.assert_array_equal(compact.view(np.uint64), dense.view(np.uint64))

    def test_forward_compact_faces_preserve_membership_and_exact_coordinates(self):
        rng = np.random.default_rng(20260909)
        vectors = np.concatenate((rng.normal(size=(400, 3)),
                                  np.array(list(product((-1., 0., 1.), repeat=3)))))
        vectors[0:2] *= 1e-300
        vectors[2:4] *= 1e300
        for face in [*range(6), np.arange(6)[:, None], np.array([1, 3, 5])[:, None, None]]:
            shape = np.broadcast_shapes(np.shape(face), vectors.shape[:-1])
            compact = qsc.qsc_forward_face(vectors, face)
            dense = qsc.qsc_forward_face(vectors, np.broadcast_to(face, shape))
            with self.subTest(face_shape=np.shape(face)):
                for actual, expected in zip(compact[:2], dense[:2]):
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                np.testing.assert_array_equal(compact[2], dense[2])


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Torch is optional')
class SphericalRotationReuseTests(unittest.TestCase):
    def test_validation_reuses_values_but_rejects_mutated_rotation(self):
        engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        view = spherical_view(rotation=rotation_xyz())
        with mock.patch.object(np.linalg, 'det', wraps=np.linalg.det) as determinant:
            spherical_cuda._render_contract(engine, view, 0)
            # A different container with the same values shares the validation.
            view.spherical_rotation_xyz = list(view.spherical_rotation_xyz)
            spherical_cuda._render_contract(engine, view, 1)
            self.assertEqual(determinant.call_count, 1)
            view.spherical_rotation_xyz[0] *= 2
            with self.assertRaisesRegex(ValueError, 'proper rigid rotation'):
                spherical_cuda._render_contract(engine, view, 1)
        self.assertEqual(len(engine._spherical_rotation_cache), 1)

    def test_invalid_values_never_enter_cache_and_validations_stay_bounded(self):
        engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        view = spherical_view()
        from XTA.spherical_geometry import cube_rotation
        with mock.patch.object(spherical_cuda, '_ROTATION_CACHE_ENTRIES', 2):
            for angle in (0., 20., 40., 60.):
                view.spherical_rotation_xyz = cube_rotation('vertical', angle)
                spherical_cuda._render_contract(engine, view, 0)
            self.assertEqual(len(engine._spherical_rotation_cache), 2)
            for rotation in ((1.,) * 8, (float('nan'),) * 9, (float('inf'),) * 9,
                             tuple(np.diag((1., 1., -1.)).reshape(-1))):
                view.spherical_rotation_xyz = rotation
                with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                    spherical_cuda._render_contract(engine, view, 0)
            self.assertEqual(len(engine._spherical_rotation_cache), 2)
        spherical_cuda.clear_spherical_render_cache(engine)
        self.assertFalse(engine._spherical_rotation_cache)

    def test_cached_rotation_does_not_bypass_source_or_radius_validation(self):
        engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        view = spherical_view()
        spherical_cuda._render_contract(engine, view, 0)
        for field, value in (('full_t', 10), ('spherical_radii', (-1.,)),
                             ('spherical_face', 8), ('src_w', 0)):
            invalid = spherical_view()
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                spherical_cuda._render_contract(engine, invalid, 0)


if __name__ == '__main__':
    unittest.main()
