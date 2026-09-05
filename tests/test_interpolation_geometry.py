"""CPU numerical contracts for bridge geometry, radius admission, and membership painting."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from XTA import interpolation


CPU_ENV = {
    'YOLO_TTA_GPU_INTERPOLATION': '0',
    'YOLO_TTA_GPU_INTERPOLATION_REQUIRED': '0',
    'YOLO_TTA_GPU_INTERPOLATION_RADIUS': '0',
    'YOLO_TTA_GPU_SLICE_LABELING': '0',
    'YOLO_TTA_TELEMETRY': '0',
    'YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER': '0',
}


def _ellipse(dest, z, cy, cx, ry, rx, value=1):
    yy, xx = np.ogrid[:dest.shape[1], :dest.shape[2]]
    dest[z, ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0] = value


def _paint_cases():
    rng = np.random.default_rng(60213)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        top = 1 << (np.dtype(dtype).itemsize * 8 - 1)
        for bit in (0, 1, 3, top, top | 1):
            for variant in range(6):
                base = rng.integers(0, 128, size=(42, 54), dtype=dtype)
                base[::3, ::4] |= np.asarray(top, dtype=dtype)
                dest = (base, base[::2, ::2], base[::-1, ::-1], base.T, base, base)[variant]
                mask = rng.random((13, 19)) > 0.38
                if variant % 2:
                    mask = mask[::-1, ::-1]
                if variant == 4:
                    mask.fill(False)
                center = ((12.5, 13.5), (0, 1), (41, 53), (-2, 8), (4, 4), (-100, -100))[variant]
                yield (dtype.__name__, bit, variant), dest, mask, center, bit, False
        binary = np.zeros((31, 37), dtype=dtype)
        binary[::2, ::3] = 1
        yield (dtype.__name__, 'binary'), binary, np.ones((9, 13), bool), (14, 15), 1, True
    for dtype in (np.bool_, np.int8, np.int16, np.int32, np.int64):
        dest = np.zeros((18, 22), dtype=dtype)
        dest[::3, ::2] = 1
        yield (dtype.__name__, 'fallback'), dest, np.ones((7, 11), bool), (0, 9), 1, False


class MembershipPainterTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(interpolation._numba_paste_packed_or_kernel, 'signatures'), 'requires Numba')
    def test_unsigned_top_bit_reaches_compiled_painter(self):
        top = 1 << 63
        dest = np.array([[1, top]], dtype=np.uint64)
        with (
            mock.patch.object(interpolation, '_NUMBA_PLANNING_KERNELS_RUNTIME_DISABLED', False),
            mock.patch.object(interpolation, 'compiled_interpolation_kernels_enabled', return_value=True),
            mock.patch.object(interpolation, '_numba_paste_packed_or_kernel',
                              wraps=interpolation._numba_paste_packed_or_kernel) as kernel,
        ):
            added = interpolation._paste_local_mask_onto_slice(
                dest, np.ones((1, 2), bool), (0, 1), paint_value=top, binary_destination=False,
            )
            kernel.assert_called_once()
            self.assertFalse(interpolation._NUMBA_PLANNING_KERNELS_RUNTIME_DISABLED)
        self.assertEqual(added, 1)
        np.testing.assert_array_equal(dest, np.array([[top | 1, top]], dtype=np.uint64))

    def test_membership_or_counts_and_clipping_with_strided_unsigned_arrays(self):
        for name, dest, mask, center, bit, binary in _paint_cases():
            with self.subTest(case=name):
                expected = dest.copy()
                expected_count = 0
                y_origin = round(center[0]) - mask.shape[0] // 2
                x_origin = round(center[1]) - mask.shape[1] // 2
                for my, mx in np.argwhere(mask):
                    y, x = int(my) + y_origin, int(mx) + x_origin
                    if 0 <= y < dest.shape[0] and 0 <= x < dest.shape[1]:
                        old = int(expected[y, x])
                        expected_count += (old & bit) == 0
                        expected[y, x] = old | bit
                bounds = [10000, 10000, 0, 0]
                count = interpolation._paste_local_mask_onto_slice(
                    dest, mask, center, paint_value=bit,
                    binary_destination=binary, dst_bbox_union=bounds,
                )
                self.assertEqual(count, expected_count)
                np.testing.assert_array_equal(dest, expected)
                repeat = interpolation._paste_local_mask_onto_slice(
                    dest, mask, center, paint_value=bit, binary_destination=binary,
                )
                self.assertEqual(repeat, 0 if bit else expected_count)
                if np.any(mask) and y_origin < dest.shape[0] and x_origin < dest.shape[1] and (
                    y_origin + mask.shape[0] > 0 and x_origin + mask.shape[1] > 0
                ):
                    self.assertEqual(bounds, [
                        max(0, y_origin), max(0, x_origin),
                        min(dest.shape[0], y_origin + mask.shape[0]),
                        min(dest.shape[1], x_origin + mask.shape[1]),
                    ])
                else:
                    self.assertEqual(bounds, [10000, 10000, 0, 0])

    def test_numpy_fallback_sets_partly_overlapping_composite_values(self):
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            dest = np.array([[1, 0], [3, 2]], dtype=dtype)
            with mock.patch.object(interpolation, '_planning_kernels_active', return_value=False):
                added = interpolation._paste_local_mask_onto_slice(
                    dest, np.ones((2, 2), bool), (1, 1),
                    paint_value=3, binary_destination=False,
                )
            self.assertEqual(added, 1)
            np.testing.assert_array_equal(dest, 3)


@unittest.skipUnless(getattr(interpolation.cv2, '__file__', None), 'requires real OpenCV and SciPy')
class BridgeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, CPU_ENV))
        old_threads = interpolation.cv2.getNumThreads()
        interpolation.cv2.setNumThreads(1)
        self.addCleanup(interpolation.cv2.setNumThreads, old_threads)

    @staticmethod
    def _exact_radius_and_sections(plan):
        # Independent SciPy topology and EDT oracle for threshold admission.
        from scipy import ndimage

        radius = min(float(plan.sdf0.max()), float(plan.sdf1.max()))
        sections = []
        for step in range(1, plan.steps):
            alpha = step / plan.steps
            mask = ((1 - alpha) * plan.sdf0 + alpha * plan.sdf1) >= 0
            labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
            center = np.array(mask.shape) // 2
            label = labels[tuple(center)]
            if not label and count:
                coords = np.argwhere(labels)
                nearest = coords[np.argmin(np.sum((coords - center) ** 2, axis=1))]
                label = labels[tuple(nearest)]
            kept = ndimage.binary_fill_holes(labels == label) if label else np.zeros_like(mask)
            radius = min(radius, float(ndimage.distance_transform_edt(kept).max()))
            sections.append(kept)
        return radius, sections

    def test_radius_admission_cached_painting_and_mirrored_wrap(self):
        for z0, z1, sign, wrap in ((2, 11, 1, False), (11, 2, -1, False),
                                  (13, 2, 1, True), (2, 13, -1, True)):
            for use_cache in (False, True):
                for shape in ('elongated', 'asymmetric', 'threshold'):
                    labels = np.zeros((16, 55, 89), np.uint16)
                    cy, cx, target_x = 23, 17, 69 if wrap else 19
                    ry, rx = {'elongated': (13, 5), 'asymmetric': (7, 9), 'threshold': (3, 3)}[shape]
                    _ellipse(labels, z0, cy, cx, ry, rx, 1)
                    _ellipse(labels, z1, cy + 2, target_x, ry - 1 if ry > 3 else ry, rx, 2)
                    if shape == 'asymmetric':
                        labels[z0, cy - 3:cy + 2, cx - 8:cx - 4] = 0
                    cache = interpolation.SliceComponentTableCache(labels) if use_cache else None
                    if cache is not None:
                        self.addCleanup(cache.clear)
                    plan = interpolation._build_linear_slice_bridge_plan(
                        labels, 1, 2, (z0, cy, cx), (z1, cy + 2, target_x),
                        direction_sign=sign, wrap_axis=wrap, component_cache=cache,
                    )
                    self.assertIsNotNone(plan)
                    exact, sections = self._exact_radius_and_sections(plan)
                    for threshold in (0., 1., 3., 3.000001, 10.):
                        with self.subTest(z0=z0, z1=z1, cache=use_cache, shape=shape, threshold=threshold):
                            radius = interpolation._estimate_linear_slice_bridge_min_radius_from_plan(
                                plan, reject_at_or_below=threshold, cache_sections=True,
                            )
                            self.assertEqual(radius > threshold, exact > threshold)
                            if threshold == 0:
                                self.assertAlmostEqual(radius, exact, places=5)
                            if radius > threshold:
                                for step, section in enumerate(sections, 1):
                                    np.testing.assert_array_equal(plan.cached_sections[step], section)
                            cached = np.zeros_like(labels, dtype=np.uint8)
                            for step in range(1, plan.steps):
                                z = (z0 + plan.sign * step) % plan.num_slices
                                interpolation._paint_linear_slice_bridge_plan_onto_slice(cached[z], plan, step)
                            plan.cached_sections.clear()
                            recomputed = np.zeros_like(cached)
                            for step in range(1, plan.steps):
                                z = (z0 + plan.sign * step) % plan.num_slices
                                interpolation._paint_linear_slice_bridge_plan_onto_slice(recomputed[z], plan, step)
                            np.testing.assert_array_equal(cached, recomputed)

    def test_tight_canvas_preserves_endpoints_and_ignores_world_translation(self):
        plans = []
        for target_x in (21, 121):
            labels = np.zeros((7, 61, 151), np.uint16)
            _ellipse(labels, 0, 29, 20, 15, 3, 1)
            _ellipse(labels, 6, 31, target_x, 14, 3, 2)
            cache = interpolation.SliceComponentTableCache(labels)
            try:
                for selected_cache in (None, cache):
                    plan = interpolation._build_linear_slice_bridge_plan(
                        labels, 1, 2, (0, 29, 20), (6, 31, target_x),
                        component_cache=selected_cache,
                    )
                    self.assertGreater(plan.sdf0.shape[0], 2 * plan.sdf0.shape[1])
                    for sdf, anchor, z, label in ((plan.sdf0, plan.source_anchor, 0, 1),
                                                   (plan.sdf1, plan.target_anchor, 6, 2)):
                        rendered = np.zeros(labels.shape[1:], np.uint8)
                        interpolation._paste_local_mask_onto_slice(rendered, sdf >= 0, anchor)
                        np.testing.assert_array_equal(rendered, labels[z] == label)
                    plans.append(plan)
            finally:
                cache.clear()
        for plan in plans[1:]:
            np.testing.assert_allclose(plan.sdf0, plans[0].sdf0, atol=1e-6)
            np.testing.assert_allclose(plan.sdf1, plans[0].sdf1, atol=1e-6)

    def test_radius_certificate_avoids_edt_but_defers_near_float_boundary(self):
        labels = np.zeros((6, 25, 25), np.uint16)
        _ellipse(labels, 0, 12, 12, 6, 6, 1)
        _ellipse(labels, 5, 12, 12, 6, 6, 2)
        plan = interpolation._build_linear_slice_bridge_plan(labels, 1, 2, (0, 12, 12), (5, 12, 12))
        with mock.patch.object(interpolation, '_component_max_radius', wraps=interpolation._component_max_radius) as edt:
            certified = interpolation._estimate_linear_slice_bridge_min_radius_from_plan(
                plan, reject_at_or_below=3., cache_sections=True,
            )
            self.assertGreater(certified, 3.)
            edt.assert_not_called()
            self.assertEqual(len(plan.cached_sections), plan.steps + 1)
            interpolation._estimate_linear_slice_bridge_min_radius_from_plan(
                plan, reject_at_or_below=certified - 1e-8,
            )
            self.assertGreater(edt.call_count, 0)

    def test_failed_anchor_certificate_does_not_reject_a_wide_component(self):
        labels = np.zeros((5, 25, 25), np.uint16)
        _ellipse(labels, 0, 12, 12, 6, 6, 1)
        _ellipse(labels, 4, 12, 12, 6, 6, 2)
        plan = interpolation._build_linear_slice_bridge_plan(labels, 1, 2, (0, 12, 6), (4, 12, 6))
        self.assertLessEqual(plan.sdf0[plan.sdf0.shape[0] // 2, plan.sdf0.shape[1] // 2], 3.)
        radius = interpolation._estimate_linear_slice_bridge_min_radius_from_plan(plan, reject_at_or_below=3.)
        self.assertGreater(radius, 3.)

    def test_full_pass_preserves_source_and_exact_decomposed_delta(self):
        tracks = np.zeros((17, 66, 100), np.uint8)
        for z in (2, 3, 9, 10, 15):
            _ellipse(tracks, z, 21, 23 + z // 7, 13, 6)
        for z in (1, 2, 8, 9, 14, 15):
            _ellipse(tracks, z, 48, 70 - z // 7, 6, 9)
        for z in (3, 10):
            _ellipse(tracks, z, 48, 20, 1, 2)
        branched = np.zeros((15, 64, 96), np.uint8)
        for z in (1, 2, 3, 4):
            _ellipse(branched, z, 30, 48, 7, 5)
        for z in (10, 11, 12, 13):
            for x in (40, 56):
                _ellipse(branched, z, 30, x, 6, 5)
        radial = np.zeros((16, 60, 96), np.uint8)
        for z in (1, 2):
            _ellipse(radial, z, 24, 18, 10, 6)
        for z in (12, 13, 14):
            _ellipse(radial, z, 24, 77, 10, 6)
        edge = np.zeros((12, 43, 58), np.uint8)
        for z in (1, 2, 8, 9):
            _ellipse(edge, z, 17, 0, 10, 6)
            _ellipse(edge, z, 34, 48, 4, 8)
        empty = np.zeros((4, 13, 17), np.uint8)
        one = empty.copy()
        one[1:3, 3:9, 4:12] = 1
        # name, mask, radius, walkback, candidates, wrap, components, passes, workers, memory
        cases = (
            ('tracks', tracks, 3., 1, 1, False, True, 2, 1, True),
            ('branched', branched, 3., 2, 3, False, True, 1, 3, True),
            ('no_radius', tracks, 0., 1, 1, False, True, 1, 2, True),
            ('binary_disk', tracks, 3., 0, 1, False, False, 1, 2, False),
            ('radial', radial, 3., 1, 1, True, True, 2, 2, True),
            ('edge', edge, 3., 1, 2, False, True, 1, 2, True),
            ('empty', empty, 3., 1, 1, False, True, 1, 1, True),
            ('one', one, 3., 1, 1, False, True, 1, 1, True),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for name, source, radius, walk, candidates, wrap, components, passes, workers, memory in cases:
                mask = source.copy()
                for pass_idx in range(passes):
                    with self.subTest(case=name, pass_index=pass_idx):
                        before = mask.copy()
                        folder = Path(tmp) / f'{name}_{pass_idx}'
                        delta_path = folder / 'delta.u8.dat'
                        stats = interpolation.interpolate_view_volume_pass_inplace(
                            mask, work_dir=folder, pass_tag='test', max_slice_distance=8,
                            search_angle_deg=30., interpolation_walk_back=walk,
                            interpolation_candidates=candidates, interpolate_min_radius=radius,
                            prefer_memory=memory, reserve_bytes=0, workers=workers, wrap_axis=wrap,
                            bridge_delta_path=delta_path,
                            bridge_component_dir=folder / 'components' if components else None,
                        )
                        delta = ((mask != 0) & (before == 0)).astype(np.uint8)
                        self.assertEqual(int(delta.sum()), stats['added_voxels'])
                        self.assertTrue(np.all(mask[before != 0] != 0))
                        if delta_path.exists():
                            np.testing.assert_array_equal(np.fromfile(delta_path, np.uint8).reshape(mask.shape), delta)
                        planes = []
                        for entry in stats.get('bridge_component_deltas', []):
                            store = interpolation.RawBBoxMaskStore.open(Path(entry['path']))
                            try:
                                plane = np.stack([store.decode_slice(z) for z in range(mask.shape[0])])
                            finally:
                                store.close()
                            np.testing.assert_array_equal(plane & (before != 0), 0)
                            self.assertEqual(int(np.count_nonzero(plane)), entry['added_voxels'])
                            planes.append(plane)
                        if planes:
                            np.testing.assert_array_equal(np.bitwise_or.reduce(planes), delta)
                        if name == 'branched':
                            self.assertGreaterEqual(sum(np.any(plane) for plane in planes), 4)
                            self.assertGreater(sum(np.count_nonzero(plane) for plane in planes), int(delta.sum()))
                        if name == 'radial' and pass_idx == 0:
                            self.assertTrue(np.any(delta[0]) and np.any(delta[-1]))


if __name__ == '__main__':
    unittest.main()
