"""Exact CSR construction across strip, wrap and budget boundaries."""
from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

from XTA import cylindrical_projection as cp, geometry


class RadialPlanConstructionTests(unittest.TestCase):
    def test_single_pass_matches_reference_across_strips_wraps_and_restored_grids(self):
        for base in ('transverse', 'sagittal', 'coronal'):
            for minimum in (.01, .7, 1.):
                views = geometry.get_view_infos(9, 11, 13, cartesian_views=(), radial_views=(base,),
                    radial_min_radius=minimum, radial_patch_size=16)
                for initial in views:
                    for shift in (0., np.nextafter(.5, 0.), np.nextafter(.5, 1.)):
                        view = replace(initial, radial_arc_origin=initial.radial_arc_origin + shift)
                        radii = np.asarray(geometry.radial_global_radii(view))
                        for shape in ((9, 11, 13), (7, 8, 9), (10, 14, 17)):
                            expected = cp._build_radial_plane_plan_reference(view, radii, shape)
                            for chunk in (1, 17, 128):
                                with self.subTest(base=base, minimum=minimum, shift=shift,
                                                  shape=shape, chunk=chunk), \
                                        mock.patch.object(cp, '_PLANE_BUILD_CHUNK_PIXELS', chunk):
                                    actual = cp._build_radial_plane_plan(view, radii, shape)
                                    for name in ('shell_index', 'column_offsets', 'native_columns'):
                                        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
                                        self.assertFalse(getattr(actual, name).flags.writeable)

    def test_occurrence_budget_is_enforced_before_retaining_oversized_strips(self):
        view = geometry.get_view_infos(3, 7, 7, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.01, radial_patch_size=128)[0]
        radii = np.asarray(geometry.radial_global_radii(view))
        expected = cp._build_radial_plane_plan_reference(view, radii, (3, 7, 7))
        with mock.patch.object(cp, '_PLANE_PLAN_MAX_BYTES', expected.nbytes - 1):
            for build in (cp._build_radial_plane_plan, cp._build_radial_plane_plan_reference):
                with self.assertRaises(cp._RadialPlanePlanTooLarge):
                    build(view, radii, (3, 7, 7))


if __name__ == '__main__':
    unittest.main()
