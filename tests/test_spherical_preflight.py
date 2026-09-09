"""Bounded runtime sampling fails closed at selected image/ROI/foreground edges."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from XTA.spherical_preflight import spherical_preflight_windows, validate_spherical_preflight_plane
from XTA.spherical_projection_bounds import SphericalOutputBounds


class SphericalMathPreflightTests(unittest.TestCase):
    shape = (1025, 1031)
    bounds = SphericalOutputBounds(2, 7, 71, 971, 53, 999)
    foreground = (311, 812, 223, 876)

    def plane(self):
        return (np.arange(np.prod(self.shape), dtype=np.uint32).reshape(self.shape) % 3 == 0).astype(np.uint8)

    def test_large_plane_cap_determinism_edges_interior_and_tail(self):
        mode, windows = spherical_preflight_windows(self.shape, self.bounds, self.foreground)
        self.assertEqual(mode, 'bounded_windows')
        self.assertEqual((mode, windows), spherical_preflight_windows(self.shape, self.bounds, self.foreground))
        self.assertLessEqual(sum(stop - first for first, stop in windows), 32768)
        self.assertTrue(all(0 <= first < stop <= np.prod(self.shape) for first, stop in windows))
        self.assertTrue(all(stop < following for (_, stop), (following, _) in zip(windows, windows[1:])))
        rows = (0, 1, self.shape[0] - 2, self.shape[0] - 1,
                self.bounds.y0 - 1, self.bounds.y0, self.bounds.y1 - 1, self.bounds.y1,
                self.foreground[0], self.foreground[1] - 1)
        # Mandatory rectangle edges are sampled as their own cross-products.
        points = [(y, x) for y in rows[:4] for x in (0, self.shape[1] - 1)]
        points += [(y, x) for y in rows[4:8] for x in
                   (self.bounds.x0 - 1, self.bounds.x0, self.bounds.x1 - 1, self.bounds.x1)]
        points += [(y, x) for y in rows[8:] for x in (self.foreground[2], self.foreground[3] - 1)]
        points += [(self.shape[0] // 2, self.shape[1] // 2)]
        for y, x in points:
            flat = y * self.shape[1] + x
            self.assertTrue(any(first <= flat < stop for first, stop in windows), (y, x))

    def test_corruption_at_checked_boundaries_and_tail_is_rejected(self):
        expected = self.plane()
        points = ((0, 0), (0, self.shape[1] - 1), (self.shape[0] - 1, 0),
                  (self.shape[0] - 1, self.shape[1] - 1),
                  (self.bounds.y0 - 1, self.bounds.x0), (self.bounds.y1, self.bounds.x1 - 1),
                  (self.foreground[0], self.foreground[2]), (self.foreground[1] - 1, self.foreground[3] - 1))
        for point in points:
            checked = expected.copy()
            checked[point] ^= np.uint8(1)
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, 'math preflight differs.*source Z=17'):
                validate_spherical_preflight_plane(checked, lambda first, stop: expected.reshape(-1)[first:stop],
                    self.bounds, self.foreground, z=17)

    def test_small_planes_and_full_optout_remain_exhaustive_and_chunk_bounded(self):
        mode, windows = spherical_preflight_windows((19, 23), self.bounds)
        self.assertEqual((mode, windows), ('full_small', ((0, 437),)))
        mode, windows = spherical_preflight_windows(self.shape, self.bounds, full=True)
        self.assertEqual(mode, 'full_requested')
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], np.prod(self.shape))
        self.assertTrue(all(stop == following for (_, stop), (following, _) in zip(windows, windows[1:])))
        self.assertLessEqual(max(stop - first for first, stop in windows), 128 * 1024)

    def test_correct_windows_account_exactly_and_invalid_oracle_fails(self):
        expected = self.plane()
        visited = []
        def oracle(first, stop):
            visited.append((first, stop))
            return expected.reshape(-1)[first:stop]
        mode, pixels = validate_spherical_preflight_plane(expected, oracle, self.bounds, self.foreground)
        self.assertEqual(mode, 'bounded_windows')
        self.assertEqual(pixels, sum(stop - first for first, stop in visited))
        self.assertLessEqual(pixels, 32768)
        with self.assertRaisesRegex(ValueError, 'invalid window'):
            validate_spherical_preflight_plane(expected, lambda *_: np.zeros(1, np.uint8), self.bounds)

    def test_empty_and_narrow_rois_are_bounded_without_foreground_scans(self):
        for shape, bounds in (((1, 100003), SphericalOutputBounds(0, 0, 0, 0, 0, 0)),
                              ((100003, 1), SphericalOutputBounds(0, 1, 99999, 100003, 0, 1))):
            mode, windows = spherical_preflight_windows(shape, bounds)
            self.assertEqual(mode, 'bounded_windows')
            self.assertLessEqual(sum(stop - first for first, stop in windows), 32768)
            self.assertEqual(windows[0][0], 0)
            self.assertEqual(windows[-1][1], np.prod(shape))


@unittest.skipUnless(os.environ.get('XTA_TEST_SPHERICAL_CUDA') == '1', 'requires explicit available-GPU qualification')
class SphericalMathPreflightCudaTests(unittest.TestCase):
    def test_production_plane_bounded_checks_preserve_full_offline_oracle_and_codecs(self):
        from XTA import spherical_projection as sp
        from XTA.spherical_geometry import build_spherical_view_infos
        from XTA.spherical_projection_cuda import SphericalCudaProjector
        work = (2911, 3064, 3022)
        shape = (7, 3064, 3022)
        view = build_spherical_view_infos(*work, targets=('transverse',), min_radius=243.,
                                          patch_size=3072, tilted_views=())[0]
        data = np.random.default_rng(142765).integers(0, 2, (view.num_slices, 9, 9), dtype=np.uint8)
        pull = sp._pull_spherical_chunk
        visits = []
        def recorded(*args):
            visits.append(int(args[7]) - int(args[6]))
            return pull(*args)
        with mock.patch.dict(os.environ, {'YOLO_TTA_SPHERICAL_FULL_MATH_PREFLIGHT': '0'}), \
                mock.patch.object(sp, '_pull_spherical_chunk', side_effect=recorded):
            projector = SphericalCudaProjector(data, view, shape, reserve_bytes=0)
        try:
            self.assertEqual(projector.preflight_mode, 'bounded_windows')
            self.assertEqual(sum(visits), projector.preflight_pixels)
            self.assertLessEqual(sum(visits), len(projector.preflight_planes) * 32768)
            self.assertLessEqual(max(visits), 32768)
            for first in (0, shape[0] // 2, shape[0] - 1):
                expected = sp._project_spherical_block(data, view, np.asarray(view.spherical_radii),
                    np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, first, 1)
                np.testing.assert_array_equal(projector.project(first, 1), expected)
                for packed in (False, True):
                    projector._validate_encoded_preflight(expected, projector.project_encoded(first, 1, packed))
        finally:
            projector.close()

    def test_runtime_bounded_and_full_modes_preserve_full_output_parity(self):
        from XTA.spherical_geometry import build_spherical_view_infos
        from XTA.spherical_projection import _project_spherical_block
        from XTA.spherical_projection_cuda import SphericalCudaProjector
        shape = (7, 257, 263)
        view = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=.5,
                                          patch_size=9, tilted_views=())[0]
        data = np.ones((view.num_slices, 9, 9), np.uint8)
        expected = _project_spherical_block(data, view, np.asarray(view.spherical_radii),
            np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0])
        for full in (False, True):
            with self.subTest(full=full), mock.patch.dict(os.environ,
                    {'YOLO_TTA_SPHERICAL_FULL_MATH_PREFLIGHT': str(int(full))}), \
                    SphericalCudaProjector(data, view, shape, reserve_bytes=0) as projector:
                self.assertEqual(projector.preflight_mode, 'full_requested' if full else 'bounded_windows')
                if full:
                    self.assertEqual(projector.preflight_pixels, len(projector.preflight_planes) * shape[1] * shape[2])
                else:
                    self.assertLessEqual(projector.preflight_pixels, len(projector.preflight_planes) * 32768)
                np.testing.assert_array_equal(projector.project(0, shape[0]), expected)

    def test_checked_gpu_corruption_fails_constructor_after_stream_cleanup(self):
        from XTA.spherical_geometry import build_spherical_view_infos
        from XTA.spherical_projection_cuda import SphericalCudaProjector, SphericalCudaProjectionUnavailable
        shape = (7, 257, 263)
        view = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=.5,
                                          patch_size=9, tilted_views=())[0]
        data = np.ones((view.num_slices, 9, 9), np.uint8)
        run = SphericalCudaProjector._run_block
        close = SphericalCudaProjector.close
        closed = []
        def close_checked(projector):
            close(projector)
            closed.append(projector._closed)
        for row, col in ((0, 0), (0, shape[2] - 1), (shape[1] - 1, 0), (shape[1] - 1, shape[2] - 1)):
            def corrupt(projector, first, count):
                result = run(projector, first, count)
                result[0, row, col] ^= np.uint8(1)
                return result
            with self.subTest(point=(row, col)), \
                    mock.patch.object(SphericalCudaProjector, '_run_block', corrupt), \
                    mock.patch.object(SphericalCudaProjector, 'close', close_checked), \
                    self.assertRaisesRegex(SphericalCudaProjectionUnavailable, 'math preflight differs'):
                SphericalCudaProjector(data, view, shape, reserve_bytes=0)
        self.assertEqual(closed, [True] * 4)


if __name__ == '__main__':
    unittest.main()
