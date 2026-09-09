"""Analytic face-cone bounds and ROI CUDA publication retain scalar pull output."""
from __future__ import annotations

from dataclasses import replace
import itertools
import math
import os
import unittest
from unittest import mock

import numpy as np

from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from XTA.spherical_projection import _project_spherical_block
from XTA.spherical_projection_bounds import _face_direction_bounds, spherical_output_bounds
from XTA.spherical_projection_cuda import SphericalCudaProjector


def projected(data, view, shape, boxes=None):
    return _project_spherical_block(data, view, np.asarray(view.spherical_radii),
        np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0], boxes)


class SphericalBoundsTests(unittest.TestCase):
    def test_upright_face_extrema_include_edge_stationary_points_and_corners(self):
        lower, upper = _face_direction_bounds(np.eye(3), 0)
        np.testing.assert_allclose(lower, (1 / math.sqrt(3), -1 / math.sqrt(2), -1 / math.sqrt(2)), atol=2e-12)
        np.testing.assert_allclose(upper, (1, 1 / math.sqrt(2), 1 / math.sqrt(2)), atol=2e-12)
        # A corner-only implementation incorrectly bounds tangent coordinates by 1/sqrt(3).
        self.assertGreater(upper[1], 1 / math.sqrt(3))

    def test_rotated_cones_contain_face_edges_corners_and_interior(self):
        from XTA.qsc import QSC_FACE_BASES
        tilted = np.asarray(cube_rotation('vertical', 31)).reshape(3, 3)
        perturbed = tilted.copy()
        perturbed[0, 1] += 5e-13
        for face, rotation in itertools.product(range(6), (np.eye(3), tilted, perturbed,
                np.asarray(cube_rotation('horizontal', -23)).reshape(3, 3))):
            lower, upper = _face_direction_bounds(rotation, face)
            basis = np.asarray(QSC_FACE_BASES[face])
            for right, up in itertools.product((-1., -.5, 0., .5, 1.), repeat=2):
                local = np.asarray((1., right, up)) @ basis
                # The actual pull forms R.T @ world. Solve that equation also
                # for accepted near-orthogonal matrices, without assuming R^-1=R.T.
                world = np.linalg.solve(rotation.T, local)
                world /= np.linalg.norm(world)
                self.assertTrue(np.all(world >= lower))
                self.assertTrue(np.all(world <= upper))

    def test_bounds_never_drop_scalar_output_across_faces_patches_restored_grids_and_shells(self):
        checked = 0
        for shape, size, minimum in (((17, 19, 21), 15, .5), ((18, 20, 22), 9, 3.),
                                      ((33, 35, 37), 17, 7.)):
            built = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=minimum,
                                               patch_size=size, tilted_views=())
            for rotation in (None, cube_rotation('vertical', 31), cube_rotation('horizontal', -23)):
                for initial in built:
                    view = replace(initial, spherical_rotation_xyz=rotation) if rotation else initial
                    for output in (shape, (shape[0] - 6, shape[1] + 2, shape[2] - 2)):
                        for selected in ('all', 'middle', 'endpoints', 'empty'):
                            data = np.ones((view.num_slices, size, size), np.uint8)
                            boxes = np.tile((0, size, 0, size), (view.num_slices, 1))
                            if selected != 'all':
                                boxes[:] = 0
                                if selected == 'middle':
                                    boxes[len(boxes) // 2] = (0, size, 0, size)
                                elif selected == 'endpoints':
                                    boxes[[0, -1]] = (0, size, 0, size)
                            bounds = spherical_output_bounds(view, output, boxes)
                            result = projected(data, view, output, boxes)
                            coordinates = np.argwhere(result)
                            for axis, lo, hi in ((0, bounds.z0, bounds.z1), (1, bounds.y0, bounds.y1), (2, bounds.x0, bounds.x1)):
                                self.assertTrue(np.all(coordinates[:, axis] >= lo))
                                self.assertTrue(np.all(coordinates[:, axis] < hi))
                            if selected == 'empty':
                                self.assertEqual(bounds.voxel_count, 0)
                            checked += 1
        self.assertGreaterEqual(checked, 600)

    def test_near_orthogonal_rotation_and_incident_midpoint_centers_stay_inside(self):
        built = build_spherical_view_infos(65, 65, 65, targets=('transverse',), min_radius=.5,
                                           patch_size=99, tilted_views=())
        for initial in built:
            rotation = np.eye(3)
            rotation[0, 1] = 5e-13
            view = replace(initial, spherical_rotation_xyz=tuple(rotation.reshape(-1)))
            boxes = np.zeros((view.num_slices, 4), np.int64)
            boxes[1:3] = (0, 99, 0, 99)
            data = np.ones((view.num_slices, 99, 99), np.uint8)
            result = projected(data, view, (65, 65, 65), boxes)
            bounds = spherical_output_bounds(view, (65, 65, 65), boxes)
            cropped = np.zeros_like(result)
            cropped[bounds.z0:bounds.z1, bounds.y0:bounds.y1, bounds.x0:bounds.x1] = \
                result[bounds.z0:bounds.z1, bounds.y0:bounds.y1, bounds.x0:bounds.x1]
            np.testing.assert_array_equal(cropped, result)

    def test_production_grid_pruning_is_bounded_without_volume_allocation(self):
        work = (2911, 3064, 3022)
        shape = (1931, 3064, 3022)
        built = build_spherical_view_infos(*work, targets=('transverse',), min_radius=243.,
                                           patch_size=3072, tilted_views=())
        fractions = []
        for rotation in (None, cube_rotation('vertical', 30), cube_rotation('vertical', -30),
                         cube_rotation('horizontal', 30), cube_rotation('horizontal', -30)):
            for initial in built:
                view = replace(initial, spherical_rotation_xyz=rotation) if rotation else initial
                bounds = spherical_output_bounds(view, shape)
                fractions.append(bounds.voxel_count / math.prod(shape))
        self.assertEqual(len(fractions), 120)
        self.assertLess(max(fractions), .5)
        self.assertLess(sum(fractions) / len(fractions), .3)


@unittest.skipUnless(os.environ.get('XTA_TEST_SPHERICAL_CUDA') == '1', 'requires explicit available-GPU qualification')
class SphericalBoundsCudaTests(unittest.TestCase):
    def test_roi_dense_and_compact_output_match_full_cpu_oracle_with_empty_slabs(self):
        shape = (41, 43, 45)
        output = (31, 45, 43)
        size = 25
        built = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=4.,
                                           patch_size=size, tilted_views=())
        rng = np.random.default_rng(125754)
        for rotation in (None, cube_rotation('vertical', 31), cube_rotation('horizontal', -23)):
            for initial in built:
                view = replace(initial, spherical_rotation_xyz=rotation) if rotation else initial
                data = rng.integers(0, 2, (view.num_slices, size, size), dtype=np.uint8)
                boxes = np.zeros((view.num_slices, 4), np.int64)
                boxes[len(boxes) // 3:2 * len(boxes) // 3] = (1, size - 1, 2, size - 2)
                expected = projected(data, view, output, boxes)
                with SphericalCudaProjector(data, view, output, boxes,
                        block_bytes=output[1] * output[2] * 3, reserve_bytes=0) as projector:
                    actual = np.concatenate([projector.project(first, min(3, output[0] - first))
                                             for first in range(0, output[0], 3)])
                    np.testing.assert_array_equal(actual, expected)
                    self.assertLess(projector.roi_projection_voxels, math.prod(output))
                    for packed in (False, True):
                        for first in range(0, output[0], 3):
                            count = min(3, output[0] - first)
                            encoded = projector.project_encoded(first, count, packed)
                            projector._validate_encoded_preflight(expected[first:first + count], encoded)
                    self.assertGreater(projector.roi_skipped_blocks, 0)

    def test_proven_empty_compact_slabs_do_not_access_cuda_or_stale_buffers(self):
        shape = (41, 43, 45)
        view = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=4.,
                                          patch_size=65, tilted_views=())[0]
        data = np.ones((view.num_slices, 65, 65), np.uint8)
        boxes = np.zeros((view.num_slices, 4), np.int64)
        with SphericalCudaProjector(data, view, shape, boxes, reserve_bytes=0) as projector:
            with mock.patch.object(projector, '_launch_projection', side_effect=AssertionError('empty CUDA launch')), \
                    mock.patch.object(projector, '_encode_current_output', side_effect=AssertionError('empty metadata scan')):
                for packed in (False, True):
                    encoded = projector.project_encoded(7, 5, packed)
                    self.assertEqual([record.z for record in encoded.records], list(range(7, 12)))
                    self.assertFalse(encoded.payload.flags.writeable)
                    self.assertEqual(encoded.payload.size, 0)
                    self.assertTrue(all(record.size == record.foreground == 0 for record in encoded.records))


if __name__ == '__main__':
    unittest.main()
