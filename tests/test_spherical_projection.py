"""CPU source-coordinate and lifetime checks for spherical pull projection."""
from __future__ import annotations

from dataclasses import replace
import contextlib
import io
import math
from pathlib import Path
import threading
import time
import unittest
from unittest import mock

import numpy as np

from XTA.backprojection import SinkOnlyProjectionResult
from XTA import spherical_projection as sp
from XTA.qsc import QSC_FACE_BASES
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation


def shells(*, shape=(7, 9, 11), size=5, minimum=.7, rotation=None):
    views = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=minimum,
                                      patch_size=size, tilted_views=())
    if rotation is not None:
        views = [replace(view, spherical_rotation_xyz=tuple(np.asarray(rotation).reshape(-1))) for view in views]
    return views


def scalar_oracle(data, view, shape):
    """Independent scalar XYZ/PROJ-equation oracle without vectorized helpers."""
    out = np.zeros(shape, np.uint8)
    work = (view.full_t, view.full_h, view.full_w)
    basis = QSC_FACE_BASES[view.spherical_face]
    rotation = np.asarray(view.spherical_rotation_xyz).reshape(3, 3)
    radii = view.spherical_radii
    for position in np.ndindex(shape):
        z, y, x = [(i + .5) * a / b - .5 - (a - 1) / 2 for i, a, b in zip(position, work, shape)]
        distance = math.sqrt(x * x + y * y + z * z)
        if not view.spherical_min_radius <= distance <= view.spherical_max_radius:
            continue
        world = (x, y, z)
        local = [sum(world[i] * rotation[i, j] for i in range(3)) for j in range(3)]
        n, a, b = [sum(local[i] * axis[i] for i in range(3)) for axis in basis]
        tolerance = 8 * np.finfo(np.float64).eps * max(abs(value) for value in local)
        if n <= 0 or n + tolerance < max(abs(a), abs(b)):
            continue
        length = math.sqrt(sum(value * value for value in local))
        n, a, b = n / length, a / length, b / length
        if abs(a) >= abs(b):
            area, major, minor = (0, a, b) if a >= 0 else (2, -a, -b)
        else:
            area, major, minor = (1, b, -a) if b >= 0 else (3, -b, a)
        theta = math.atan2(minor, major)
        tangent_mu = 12 / math.pi * (theta - math.asin(math.sin(theta) / math.sqrt(2)))
        if abs(minor) == major:
            tangent_mu = math.copysign(1, minor) if minor else 0
        cosine = math.cos(theta)
        d = 1 - cosine / math.sqrt(1 + cosine * cosine)
        p = math.hypot(a, b) / math.sqrt((1 + n) * d)
        if major == n:
            p = 1
        m = p * tangent_mu
        u, v = ((p, m), (-m, p), (-p, -m), (m, -p))[area]
        u, v = max(-1, min(1, u)), max(-1, min(1, v))
        col = round((u + 1) * view.spherical_face_intervals / 2) - view.spherical_u_origin
        row = round((1 - v) * view.spherical_face_intervals / 2) - view.spherical_v_origin
        if not 0 <= col < view.src_w or not 0 <= row < view.src_h:
            continue
        shell = min(range(len(radii)), key=lambda i: abs(radii[i] - distance))
        pr = min(int((row + .5) * data.shape[1] / view.src_h), data.shape[1] - 1)
        pc = min(int((col + .5) * data.shape[2] / view.src_w), data.shape[2] - 1)
        out[position] = int(data[shell, pr, pc] != 0)
    return out


class SphericalProjectionTests(unittest.TestCase):
    def setUp(self):
        cpu_only = mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT': '0'})
        cpu_only.start()
        self.addCleanup(cpu_only.stop)
        stdout = contextlib.redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)

    def project(self, data, view, shape=None, **kwargs):
        blocks = []
        result = sp.backproject_spherical_volume_to_volume(
            data, view, Path('unused-spherical-projection.dat'), 'spherical test',
            out_shape_tyx=shape, sink_only=True,
            projection_block_callback=lambda first, block: blocks.append((first, block.copy())), **kwargs,
        )
        self.assertIsInstance(result, SinkOnlyProjectionResult)
        self.assertEqual([z + i for z, block in blocks for i in range(len(block))], list(range(result.shape[0])))
        return np.concatenate([block for _, block in blocks])

    def test_all_ones_union_covers_only_annulus_for_rotated_and_padded_grids(self):
        for shape, size, minimum in (((5, 7, 9), 4, .7), ((6, 8, 10), 5, .1),
                                      ((3, 5, 7), 8, .2), ((7, 7, 7), 5, 3.)):
            for rotation in (None, cube_rotation('vertical', 31), cube_rotation('horizontal', -23)):
                views = shells(shape=shape, size=size, minimum=minimum, rotation=rotation)
                union = np.zeros(shape, np.uint8)
                for view in views:
                    data = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
                    union |= self.project(data, view)
                delta = np.moveaxis(np.indices(shape), 0, -1) - (np.asarray(shape) - 1) / 2
                radius = np.sqrt(np.sum(delta * delta, axis=-1))
                expected = (radius >= minimum) & (radius <= views[0].spherical_max_radius)
                with self.subTest(shape=shape, size=size, rotation=rotation):
                    np.testing.assert_array_equal(union, expected)

    def test_random_masks_match_scalar_oracle_on_restored_processing_grids(self):
        rng = np.random.default_rng(377)
        for rotation in (None, cube_rotation('vertical', 31), cube_rotation('horizontal', -23)):
            views = shells(shape=(5, 7, 9), size=4, minimum=.3, rotation=rotation)
            for view in views[::max(1, len(views) // 7)]:
                for processing_shape in ((4, 4), (3, 2), (7, 6)):
                    data = (rng.random((view.num_slices, *processing_shape)) < .29).astype(np.uint8)
                    for output_shape in ((5, 7, 9), (3, 5, 7), (6, 8, 10)):
                        with self.subTest(face=view.spherical_face, processing=processing_shape,
                                          output=output_shape, rotation=rotation):
                            np.testing.assert_array_equal(self.project(data, view, output_shape),
                                                          scalar_oracle(data, view, output_shape))

    def test_incident_edges_and_corners_keep_every_face_contribution(self):
        views = shells(shape=(7, 7, 7), size=15, minimum=.7)
        for view in views:
            actual = self.project(np.ones((view.num_slices, 15, 15), np.uint8), view)
            self.assertEqual(actual[3, 4, 4], int(view.spherical_face in (0, 1)))
            self.assertEqual(actual[4, 4, 4], int(view.spherical_face in (0, 1, 4)))
            self.assertEqual(actual[3, 3, 3], 0)

    def test_padding_pixels_never_publish_and_overlap_can_contribute(self):
        view = shells(shape=(3, 5, 7), size=8, minimum=.1)[0]
        self.assertLess(view.spherical_u_origin, 0)
        data = np.ones((view.num_slices, 8, 8), np.uint8)
        first = -view.spherical_u_origin
        end = first + view.spherical_face_intervals + 1
        data[:, first:end, first:end] = 0
        self.assertFalse(self.project(data, view).any())
        views = shells(shape=(7, 7, 7), size=7, minimum=.1)
        # Every patch retains its own predictions in overlaps; no patch owner
        # assignment may suppress an otherwise valid nearest global pixel.
        rng = np.random.default_rng(433)
        union = np.zeros((7, 7, 7), np.uint8)
        expected = union.copy()
        for view in views:
            data = (rng.random((view.num_slices, 7, 7)) < .31).astype(np.uint8)
            union |= self.project(data, view)
            expected |= scalar_oracle(data, view, union.shape)
        np.testing.assert_array_equal(union, expected)

    def test_radius_midpoints_choose_inner_and_endpoints_are_closed(self):
        np.testing.assert_array_equal(sp._nearest_global_shell(np.array([.5, 1., 1.5, 2., 2.5, 3.]),
                                                               np.array([.5, 1.5, 2.5, 3.])),
                                      [0, 0, 1, 1, 2, 3])
        view = shells(shape=(7, 7, 7), size=11, minimum=.5)[0]
        view = replace(view, spherical_radii=(.5, 1.5, 2.5, 3.))
        data = np.zeros((4, 11, 11), np.uint8)
        data[0] = 1
        actual = self.project(data, view)
        self.assertEqual(actual[3, 3, 4], 1)
        self.assertEqual(actual[3, 3, 5], 0)
        data[3] = 1
        self.assertEqual(self.project(data, view)[3, 3, 6], 1)

    def test_bounding_boxes_preserve_sparse_masks_and_reject_invalid_metadata(self):
        view = shells(size=5)[0]
        data = np.zeros((view.num_slices, 3, 4), np.uint8)
        data[::2, 1:3, 1:4] = 1
        boxes = np.zeros((view.num_slices, 4), np.int64)
        boxes[::2] = (1, 3, 1, 4)
        np.testing.assert_array_equal(self.project(data, view, known_slice_bboxes=boxes), self.project(data, view))
        for invalid in (boxes.astype(float), boxes[:, :3], np.full_like(boxes, -1), boxes + 100):
            with self.subTest(boxes=invalid), self.assertRaisesRegex(ValueError, 'bounding boxes'):
                self.project(data, view, known_slice_bboxes=invalid)

    def test_tiny_chunks_and_parallel_blocks_remain_ordered_and_bounded(self):
        view = shells(size=15)[0]
        data = np.ones((view.num_slices, 15, 15), np.uint8)
        expected = self.project(data, view)
        sizes = []
        original = sp._pull_spherical_chunk

        def pull(*args, **kwargs):
            sizes.append(args[7] - args[6])
            return original(*args, **kwargs)

        with mock.patch.object(sp, '_PULL_CHUNK_VOXELS', 3), \
                mock.patch.object(sp, '_OUTPUT_BLOCK_BYTES', 99), \
                mock.patch.object(sp, '_cpu_count', return_value=3), \
                mock.patch.object(sp, '_pull_spherical_chunk', side_effect=pull), \
                mock.patch.object(sp, 'allocate_workspace_array', side_effect=AssertionError('sink allocated dense output')):
            actual = self.project(data, view, workers=3)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 3)
        with mock.patch.object(sp, '_INFLIGHT_WORK_BYTES', 1):
            self.assertEqual(sp._spherical_block_schedule(100, 1_000_000, 99)[1], 1)

    def test_sink_failure_joins_borrowed_input_readers(self):
        view = shells(size=15)[0]
        data = np.ones((view.num_slices, 15, 15), np.uint8)
        live = [0, 0]
        lock = threading.Lock()
        barrier = threading.Barrier(3)

        def project(source, view, radii, rotation, shape, first, count, boxes, bounds=None):
            with lock:
                live[0] += 1
                live[1] = max(live[1], live[0])
            try:
                barrier.wait(timeout=3)
                time.sleep(.01 * (first + 1))
                return np.ones((count, shape[1], shape[2]), np.uint8)
            finally:
                with lock:
                    live[0] -= 1

        with mock.patch.object(sp, '_OUTPUT_BLOCK_BYTES', 99), \
                mock.patch.object(sp, '_cpu_count', return_value=3), \
                mock.patch.object(sp, '_project_spherical_block', side_effect=project):
            with self.assertRaisesRegex(RuntimeError, 'sink rejected'):
                sp.backproject_spherical_volume_to_volume(
                    data, view, Path('unused.dat'), 'failure', workers=3, sink_only=True,
                    projection_block_callback=mock.Mock(side_effect=RuntimeError('sink rejected')),
                )
        self.assertEqual(live[0], 0)
        self.assertEqual(live[1], 3)
        self.assertTrue(data.all())

    def test_dense_output_is_caller_owned_and_requests_path_backed_storage(self):
        view = shells(size=15)[0]
        data = np.ones((view.num_slices, 15, 15), np.uint8)
        target = np.empty((7, 9, 11), np.uint8)
        with mock.patch.object(sp, 'allocate_workspace_array', return_value=target) as allocate:
            result = sp.backproject_spherical_volume_to_volume(data, view, Path('output.dat'), 'dense test')
        self.assertIs(result, target)
        self.assertFalse(allocate.call_args.kwargs['prefer_memory'])
        self.assertFalse(allocate.call_args.kwargs['prefer_memfd'])
        np.testing.assert_array_equal(target, self.project(data, view))

    def test_bad_contracts_fail_before_output_allocation(self):
        view = shells(size=15)[0]
        data = np.ones((view.num_slices, 15, 15), np.uint8)
        variants = (replace(view, family='radial'), replace(view, spherical_face=6),
                    replace(view, spherical_face_intervals=0), replace(view, spherical_min_radius=0),
                    replace(view, spherical_radii=(1.,)), replace(view, spherical_u_origin=-15),
                    replace(view, spherical_rotation_xyz=(1.,) * 9))
        with mock.patch.object(sp, 'allocate_workspace_array') as allocate:
            for invalid in variants:
                with self.subTest(view=invalid), self.assertRaises(ValueError):
                    sp.backproject_spherical_volume_to_volume(data, invalid, Path('bad.dat'), 'invalid')
            with self.assertRaisesRegex(ValueError, 'depth'):
                self.project(data[:-1], view)
            with self.assertRaisesRegex(ValueError, 'block consumer'):
                sp.backproject_spherical_volume_to_volume(data, view, Path('bad.dat'), 'invalid', sink_only=True)
            for invalid in ((0, 7, 9), (7, 9)):
                with self.subTest(shape=invalid), self.assertRaises(ValueError):
                    self.project(data, view, shape=invalid)
        allocate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
