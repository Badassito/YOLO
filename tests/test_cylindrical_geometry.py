from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from XTA import geometry as g
from XTA.config import resolve_tilted_view_groups
from XTA.cylindrical_geometry import radius_grid


class AzimuthalRenameRegressionTests(unittest.TestCase):
    def test_pixels_match_unmodified_v19_reference(self):
        reference = json.loads((Path(__file__).parent / 'fixtures/azimuthal_v19_reference.json').read_text())
        shape = tuple(reference['source_shape'])
        a = np.indices(shape)
        src = ((a[0] * 47 + a[1] * 19 + a[2] * 7 + 3) % 251).astype(np.uint8)
        mask = (((a[0] * 5 + a[1] * 3 + a[2]) % 7) < 2).astype(np.uint8)
        for raster in (0, 7):
            views = g.get_view_infos(
                *shape, azimuthal_views=('transverse', 'sagittal', 'coronal',
                    'tilted_transverse', 'tilted_sagittal', 'tilted_coronal'),
                azimuthal_azimuth_angles=(60,) * 6,
                tilt_groups=resolve_tilted_view_groups(['transverse,sagittal,coronal:20:both']),
                azimuthal_native_raster=raster,
            )
            actual = {v.name: v for v in views if g.is_azimuthal_view(v)}
            expected = [row for row in reference['cases'] if row['raster'] == raster]
            self.assertEqual(set(actual), {r['name'] for r in expected})
            for row in expected:
                view = actual[row['name']]
                self.assertEqual([view.num_slices, view.src_h, view.src_w], row['shape'])
                for index, hashes in enumerate(row['frames']):
                    with self.subTest(view=view.name, raster=raster, frame=index):
                        intensity = g.get_view_frame_by_index(src, view, index)
                        categorical = g.get_categorical_view_frame_by_index(mask, view, index)
                        self.assertEqual(hashlib.sha256(intensity.tobytes()).hexdigest(), hashes['intensity'])
                        self.assertEqual(hashlib.sha256(categorical.tobytes()).hexdigest(), hashes['categorical'])


class CylindricalGeometryTests(unittest.TestCase):
    def views(self, shape=(9, 11, 13), size=8, targets=('transverse',), minimum=None, tilted=()):
        return [v for v in g.get_view_infos(
            *shape, radial_views=targets, radial_patch_size=size,
            radial_min_radius=minimum, tilt_groups=tilted,
        ) if g.is_radial_view(v)]

    def test_default_two_wraps_and_square_patches(self):
        views = self.views()
        first = views[0]
        self.assertAlmostEqual(first.radial_min_radius, 8 / (4 * math.pi))
        self.assertEqual((first.src_h, first.src_w), (8, 8))
        tokens = [g.view_output_token(v) for v in views]
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertTrue(all('/' not in token and '\\' not in token for token in tokens))
        source = np.arange(9 * 11 * 13, dtype=np.uint8).reshape(9, 11, 13)
        frame = g.get_view_frame_by_index(source, first, 0)
        np.testing.assert_array_equal(frame[:, :4], frame[:, 4:])
        for view in views:
            self.assertTrue(all(0 < d <= 1 for d in np.diff(view.radial_radii)))
            self.assertEqual(view.radial_radii[-1], 5.0)
            self.assertEqual(view.stack_axis, 'radius')
            self.assertFalse(g.is_azimuthal_view(view))
            self.assertFalse(g.is_tilted_view(view))

    def test_arc_remainder_uses_real_periodic_data_and_height_never_wraps(self):
        views = self.views(shape=(3, 11, 13))
        last = views[-1]
        source = np.full((3, 11, 13), 87, dtype=np.uint8)
        frame = g.get_view_frame_by_index(source, last, last.num_slices - 1)
        self.assertTrue(np.all(frame[:3] == 87))
        self.assertTrue(np.all(frame[3:] == 0))
        self.assertGreater(last.radial_arc_origin + 8, 2 * math.pi * last.radial_radii[-1])

    def test_height_bands_cover_without_stretching_and_channels_clamp_radius(self):
        views = self.views(shape=(19, 11, 13))
        self.assertEqual({v.radial_height_origin for v in views}, {0, 8, 11})
        for view in views:
            self.assertEqual(g.channel_view_slice_source(view, -1), (0, False))
            self.assertEqual(g.channel_view_slice_source(view, view.num_slices), (view.num_slices - 1, False))
            self.assertEqual(tuple(g.radial_global_radii(view)[view.radial_shell_start:]), view.radial_radii)

    def test_minimum_validation_and_single_shell(self):
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.views(minimum=value)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.views(minimum=6)
        self.assertEqual(radius_grid(5, 5), (5.0,))
        self.assertTrue(all(v.num_slices == 1 for v in self.views(minimum=5)))

    def test_all_axis_intensity_matches_analytic_linear_field(self):
        shape = (9, 11, 13)
        t, y, x = np.indices(shape)
        source = (t * 7 + y * 3 + x * 2 + 11).astype(np.uint8)
        for view in self.views(shape=shape, targets=('transverse', 'sagittal', 'coronal')):
            for index in (0, view.num_slices - 1):
                tt, yy, xx, valid = g.radial_shell_coordinates(view, index)
                expected = np.where(valid, np.rint(tt * 7 + yy * 3 + xx * 2 + 11), 0).astype(np.uint8)
                actual = g.get_view_frame_by_index(source, view, index)
                self.assertLessEqual(int(np.abs(actual.astype(int) - expected).max()), 1)

    def test_dense_annulus_has_actual_nonzero_source_sampling_weights(self):
        # Enumerate source taps actually visited by every frame, then compare
        # against independent Cartesian voxel-center cylinder membership.
        for shape, size in (((7, 8, 9), 4), ((8, 9, 10), 6), ((9, 11, 13), 8)):
            for base in ('transverse', 'sagittal', 'coronal'):
                views = self.views(shape=shape, size=size, targets=(base,))
                touched = np.zeros(shape, dtype=bool)
                for view in views:
                    for index in range(view.num_slices):
                        tt, yy, xx, valid = g.radial_shell_coordinates(view, index)
                        lower = [np.floor(c).astype(np.intp) for c in (tt, yy, xx)]
                        delta = [c - a for c, a in zip((tt, yy, xx), lower)]
                        for it in (0, 1):
                            for iy in (0, 1):
                                for ix in (0, 1):
                                    weights = [(d if k else 1 - d) for d, k in zip(delta, (it, iy, ix))]
                                    include = valid & (weights[0] * weights[1] * weights[2] > 1e-9)
                                    raw_ids = [a + k for a, k in zip(lower, (it, iy, ix))]
                                    for ids_for_axis, length in zip(raw_ids, shape):
                                        include &= (ids_for_axis >= 0) & (ids_for_axis < length)
                                    ids = [np.clip(a, 0, length - 1) for a, length in zip(raw_ids, shape)]
                                    touched[tuple(a[include] for a in ids)] = True
                t, y, x = np.indices(shape)
                if base == 'transverse':
                    distance = np.hypot(x - (shape[2] - 1) / 2, y - (shape[1] - 1) / 2)
                elif base == 'sagittal':
                    distance = np.hypot(x - (shape[2] - 1) / 2, t - (shape[0] - 1) / 2)
                else:
                    distance = np.hypot(y - (shape[1] - 1) / 2, t - (shape[0] - 1) / 2)
                wanted = (distance >= views[0].radial_min_radius) & (distance <= views[0].radial_max_radius)
                with self.subTest(shape=shape, base=base):
                    self.assertTrue(np.all(touched[wanted]), np.argwhere(wanted & ~touched).tolist())

    def test_thin_tilted_source_retains_partial_taps_and_height_padding_is_zero(self):
        for base, shape in (('transverse', (1, 5, 7)), ('sagittal', (5, 1, 7)), ('coronal', (5, 7, 1))):
            for direction in ('vertical', 'horizontal'):
                px, py = (1, 2) if direction == 'vertical' else (3, 0)
                source_index = ((0, py, px) if base == 'transverse' else
                                (py, 0, px) if base == 'sagittal' else (py, px, 0))
                impulse = np.zeros(shape, np.uint8)
                impulse[source_index] = 255
                views = self.views(shape=shape, size=4, targets=('tilted_' + base,),
                    tilted=resolve_tilted_view_groups([f'{base}:30:{direction}']))
                for sign in (-1, 1):
                    positive = False
                    for view in views:
                        if (view.tilt_angle_deg > 0) != (sign > 0):
                            continue
                        for index in range(view.num_slices):
                            actual = g.get_view_frame_by_index(impulse, view, index)
                            positive |= bool(np.any(actual))
                            self.assertFalse(np.any(actual[1:]), 'height padding sheared back into source')
                    with self.subTest(base=base, direction=direction, sign=sign):
                        self.assertTrue(positive, 'dense shell sampling dropped an annular source voxel')


if __name__ == '__main__':
    unittest.main()
