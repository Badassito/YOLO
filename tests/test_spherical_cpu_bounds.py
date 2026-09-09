"""CPU broad-phase scheduling keeps the unchanged categorical pull oracle."""
from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

from XTA import spherical_projection as sp
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from XTA.spherical_projection_bounds import spherical_output_bounds


def project(data, view, shape, boxes=None, *, bounded=True):
    return sp._project_spherical_block(
        data, view, np.asarray(view.spherical_radii),
        np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0], boxes,
        spherical_output_bounds(view, shape, boxes) if bounded else None,
    )


class SphericalCpuBoundsTests(unittest.TestCase):
    def test_exact_full_oracle_parity_on_faces_patches_rotations_and_restoration(self):
        built = build_spherical_view_infos(33, 35, 37, targets=('transverse',),
            min_radius=.5, patch_size=29, tilted_views=())
        rng = np.random.default_rng(20260909)
        checked = 0
        for rotation in (None, cube_rotation('vertical', 31), cube_rotation('horizontal', -23)):
            for initial in built:
                view = replace(initial, spherical_rotation_xyz=rotation) if rotation else initial
                for processing in ((29, 29), (7, 11)):
                    data = rng.integers(0, 4, (view.num_slices, *processing), dtype=np.uint8)
                    boxes = np.zeros((view.num_slices, 4), np.int64)
                    boxes[len(boxes) // 3:2 * len(boxes) // 3] = (1, processing[0], 2, processing[1])
                    for shape in ((33, 35, 37), (23, 39, 31)):
                        for metadata in (None, boxes):
                            with self.subTest(face=view.spherical_face, origin=(view.spherical_u_origin,
                                    view.spherical_v_origin), rotation=rotation, shape=shape,
                                    processing=processing, boxes=metadata is not None):
                                np.testing.assert_array_equal(project(data, view, shape, metadata),
                                    project(data, view, shape, metadata, bounded=False))
                            checked += 1
        self.assertGreaterEqual(checked, 144)

    def test_empty_metadata_and_disjoint_slabs_never_call_pull(self):
        shape = (129, 131, 133)
        view = build_spherical_view_infos(*shape, targets=('transverse',),
            min_radius=.5, patch_size=257, tilted_views=())[4]
        data = np.ones((view.num_slices, 5, 7), np.uint8)
        boxes = np.zeros((view.num_slices, 4), np.int64)
        with mock.patch.object(sp, '_pull_spherical_chunk', side_effect=AssertionError('empty pull')):
            self.assertFalse(project(data, view, shape, boxes).any())
            bounds = spherical_output_bounds(view, shape)
            self.assertGreater(bounds.z0, 0)
            block = sp._project_spherical_block(data, view, np.asarray(view.spherical_radii),
                np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, bounds.z0, None, bounds)
            self.assertFalse(block.any())

    def test_contiguous_chunks_keep_budget_global_coordinates_and_midpoint_ties(self):
        shape = (65, 67, 69)
        built = build_spherical_view_infos(*shape, targets=('transverse',),
            min_radius=.5, patch_size=101, tilted_views=())
        for initial in built:
            rotation = np.eye(3)
            rotation[0, 1] = 5e-13
            view = replace(initial, spherical_rotation_xyz=tuple(rotation.reshape(-1)))
            data = np.zeros((view.num_slices, 13, 15), np.uint8)
            data[::2, 1:-1, 2:-1] = 1
            expected = project(data, view, shape, bounded=False)
            visits = []
            original = sp._pull_spherical_chunk

            def pull(*args, **kwargs):
                visits.append((args[5], args[6], args[7]))
                return original(*args, **kwargs)

            with mock.patch.object(sp, '_PULL_CHUNK_VOXELS', 257), \
                    mock.patch.object(sp, '_pull_spherical_chunk', side_effect=pull):
                actual = project(data, view, shape)
            np.testing.assert_array_equal(actual, expected)
            bounds = spherical_output_bounds(view, shape)
            self.assertTrue(visits)
            self.assertTrue(all(bounds.z0 <= z < bounds.z1 and
                bounds.y0 * shape[2] <= first < stop <= bounds.y1 * shape[2] and
                stop - first <= 257 for z, first, stop in visits))
            self.assertLess(sum(stop - first for _, first, stop in visits), np.prod(shape))


if __name__ == '__main__':
    unittest.main()
