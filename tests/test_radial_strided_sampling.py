import os
import unittest
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA import geometry


class RadialStridedSamplingTests(unittest.TestCase):
    def tearDown(self) -> None:
        with geometry._RADIAL_SAMPLER_CACHE_LOCK:
            geometry._RADIAL_SAMPLER_CACHE.clear()

    def test_sagittal_and_coronal_match_contiguous_reference_without_plane_copy(self) -> None:
        rng = np.random.default_rng(20260830)
        volume = rng.integers(0, 256, size=(19, 23, 29), dtype=np.uint8)
        views = geometry.get_view_infos(
            19,
            23,
            29,
            cartesian_views=(),
            radial_views=("sagittal", "coronal"),
            radial_azimuth_angles=(17.0, 19.0),
            radial_native_raster=0,
        )

        for view in views:
            with self.subTest(base=geometry.radial_base_view_name(view)):
                oriented = geometry.radial_oriented_stack_view(volume, view)
                self.assertFalse(bool(oriented.flags["C_CONTIGUOUS"]))
                sampler = geometry.get_radial_sampler(view, view.azimuths_deg[2])
                contiguous_reference = geometry.extract_radial_slice_frame(
                    np.ascontiguousarray(oriented),
                    sampler,
                    out_rows=int(view.src_h),
                )
                with mock.patch.object(
                    geometry,
                    "_radial_selected_samples_from_strided_block",
                    wraps=geometry._radial_selected_samples_from_strided_block,
                ) as selected_gather:
                    actual = geometry.extract_radial_slice_frame(
                        oriented,
                        sampler,
                        out_rows=int(view.src_h),
                    )

                self.assertGreater(selected_gather.call_count, 0)
                np.testing.assert_array_equal(actual, contiguous_reference)

    def test_strided_gather_size_depends_on_taps_not_plane_area(self) -> None:
        volume = np.arange(13 * 17 * 19, dtype=np.uint16).reshape(13, 17, 19)
        view = geometry.get_view_infos(
            13,
            17,
            19,
            cartesian_views=(),
            radial_views=("sagittal",),
            radial_azimuth_angles=(30.0,),
            radial_native_raster=0,
        )[0]
        oriented = geometry.radial_oriented_stack_view(volume, view)
        sampler = geometry.get_radial_sampler(view, view.azimuths_deg[1])
        block = oriented[:5]
        selected = geometry._radial_selected_samples_from_strided_block(block, sampler)

        tap_count = int(sampler.x_idx.shape[1]) * int(sampler.y_idx.shape[1])
        self.assertEqual(selected.shape, (5, int(sampler.diameter), tap_count))
        self.assertLess(selected.nbytes, np.ascontiguousarray(block).nbytes)

    def test_folded_sagittal_and_coronal_outputs_remain_exact(self) -> None:
        rng = np.random.default_rng(18)
        volume = rng.integers(0, 256, size=(19, 23, 29), dtype=np.uint8)
        views = geometry.get_view_infos(
            19,
            23,
            29,
            cartesian_views=(),
            radial_views=("sagittal", "coronal"),
            radial_azimuth_angles=(23.0, 29.0),
            radial_native_raster=11,
        )

        for view in views:
            with self.subTest(base=geometry.radial_base_view_name(view)):
                oriented = geometry.radial_oriented_stack_view(volume, view)
                sampler = geometry.get_radial_sampler(view, view.azimuths_deg[3])
                expected = geometry.extract_radial_slice_frame(
                    np.ascontiguousarray(oriented),
                    sampler,
                    out_rows=int(view.src_h),
                )
                actual = geometry.extract_radial_slice_frame(
                    oriented,
                    sampler,
                    out_rows=int(view.src_h),
                )
                self.assertEqual(actual.shape, (11, 11))
                np.testing.assert_array_equal(actual, expected)

    def test_radial_sampler_cache_is_bounded(self) -> None:
        view = geometry.get_view_infos(
            13,
            17,
            19,
            cartesian_views=(),
            radial_views=("transverse",),
            radial_azimuth_angles=(1.0,),
            radial_native_raster=0,
        )[0]
        with mock.patch.dict(os.environ, {"YOLO_TTA_RADIAL_SAMPLER_CACHE": "16"}):
            for angle in view.azimuths_deg[:40]:
                geometry.get_radial_sampler(view, angle)

        with geometry._RADIAL_SAMPLER_CACHE_LOCK:
            self.assertLessEqual(len(geometry._RADIAL_SAMPLER_CACHE), 16)


if __name__ == "__main__":
    unittest.main()
