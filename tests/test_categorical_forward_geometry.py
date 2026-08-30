from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA import geometry
from XTA.config import TiltedViewGroup


IDENTITY_2X3 = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def _aug_job(view: geometry.ViewInfo, out_size: int, *, angle_deg: float = 0.0) -> geometry.AugJob:
    aff = geometry.AffineSpec(
        view=str(view.name),
        angle_deg=float(angle_deg),
        src_w=int(view.src_w),
        src_h=int(view.src_h),
        out_size=int(out_size),
        canvas_w=int(view.src_w),
        canvas_h=int(view.src_h),
        pad_size=max(int(view.src_w), int(view.src_h)),
        pad_off_x=0.0,
        pad_off_y=0.0,
        M_out_to_src=IDENTITY_2X3.copy(),
        M_src_to_out=IDENTITY_2X3.copy(),
        M_canvas_to_src=IDENTITY_2X3.copy(),
        M_src_to_canvas=IDENTITY_2X3.copy(),
    )
    return geometry.AugJob(
        aug_id='test',
        angle_deg=float(angle_deg),
        meta_path=Path('test.json'),
        aff=aff,
    )


def _tile_job(view: geometry.ViewInfo, out_size: int) -> geometry.DenseTileJob:
    return geometry.DenseTileJob(
        view=str(view.name),
        aug_id='test',
        config_id='s2_st1',
        tile_id='x0_y0',
        tile_x=0,
        tile_y=0,
        tile_size=2,
        tile_stride=1,
        out_size=int(out_size),
        meta_path=Path('tile.json'),
        M_out_to_src=IDENTITY_2X3.copy(),
        M_src_to_out=IDENTITY_2X3.copy(),
    )


class CategoricalForwardGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        t, y, x = np.indices((5, 6, 7))
        # Include noncanonical foreground values to prove that every public
        # categorical entry point returns canonical uint8 0/1.
        self.mask = np.asarray(((3 * t + 2 * y + x) % 5 == 0) * 9, dtype=np.uint8)

    def assert_binary(self, actual: np.ndarray) -> None:
        self.assertEqual(actual.dtype, np.uint8)
        self.assertTrue(set(np.unique(actual)).issubset({0, 1}))

    def test_native_upright_cartesian_views_are_binary_slices(self) -> None:
        views = geometry.get_view_infos(
            5,
            6,
            7,
            cartesian_views=('transverse', 'sagittal', 'coronal'),
            radial_views=(),
            radial_azimuth_angles=(),
        )
        by_name = {view.name: view for view in views}

        cases = (
            ('transverse', 2, self.mask[2]),
            ('sagittal', 3, self.mask[:, 3, :]),
            ('coronal', 4, self.mask[:, :, 4]),
        )
        for name, frame_idx, expected in cases:
            with self.subTest(name=name):
                actual = geometry.get_categorical_view_frame_by_index(
                    self.mask, by_name[name], frame_idx,
                )
                np.testing.assert_array_equal(actual, np.asarray(expected > 0, dtype=np.uint8))
                self.assert_binary(actual)

    def test_upright_radial_views_use_sampler_nearest_taps_and_nearest_row_fold(self) -> None:
        views = geometry.get_view_infos(
            5,
            6,
            7,
            cartesian_views=(),
            radial_views=('transverse', 'sagittal', 'coronal'),
            radial_azimuth_angles=(45.0, 45.0, 45.0),
            radial_native_raster=3,
        )

        for view in views:
            with self.subTest(base=geometry.radial_base_view_name(view)):
                frame_idx = 1
                sampler = geometry.get_radial_sampler(view, view.azimuths_deg[frame_idx])
                oriented = geometry.radial_oriented_stack_view(self.mask, view)
                rows = geometry._center_aligned_nearest_fold_indices(
                    int(oriented.shape[0]), int(view.src_h),
                )
                expected = oriented[
                    rows[:, None],
                    sampler.nn_y[None, :],
                    sampler.nn_x[None, :],
                ]

                actual = geometry.get_categorical_view_frame_by_index(
                    self.mask, view, frame_idx,
                )
                np.testing.assert_array_equal(actual, np.asarray(expected > 0, dtype=np.uint8))
                self.assert_binary(actual)

    def test_tilted_cartesian_all_bases_delegate_to_mask_mode_renderer(self) -> None:
        group = TiltedViewGroup(
            views=('transverse', 'sagittal', 'coronal'),
            tilt_angles=(20.0,),
            tilt_directions=('vertical',),
        )
        views = geometry.get_view_infos(
            5,
            6,
            7,
            cartesian_views=(),
            radial_views=(),
            radial_azimuth_angles=(),
            tilt_groups=(group,),
        )
        positive_views = [view for view in views if float(view.tilt_angle_deg) > 0.0]
        self.assertEqual(
            {geometry.tilted_base_view_name(view) for view in positive_views},
            {'transverse', 'sagittal', 'coronal'},
        )

        calls: list[dict[str, object]] = []

        def fake_renderer(
            volume_arr: np.ndarray,
            view: geometry.ViewInfo,
            frame_idx: int,
            M_grid_to_src: np.ndarray,
            grid_h: int,
            grid_w: int,
            *,
            mask_mode: bool,
            block_rows: int = 256,
        ) -> np.ndarray:
            calls.append({
                'volume': volume_arr,
                'view': view,
                'frame_idx': frame_idx,
                'matrix': M_grid_to_src,
                'grid_h': grid_h,
                'grid_w': grid_w,
                'mask_mode': mask_mode,
                'block_rows': block_rows,
            })
            return np.full((int(grid_h), int(grid_w)), 7, dtype=np.uint8)

        with mock.patch.object(
            geometry,
            '_render_tilted_array_on_grid',
            side_effect=fake_renderer,
        ):
            for view in positive_views:
                with self.subTest(base=geometry.tilted_base_view_name(view)):
                    job = _aug_job(view, out_size=4)
                    full = geometry.render_categorical_fullframe_for_job(
                        self.mask, view, job, frame_idx=1,
                    )
                    tile = geometry.render_categorical_dense_tile_for_job(
                        self.mask, view, _tile_job(view, out_size=3), frame_idx=1,
                    )
                    np.testing.assert_array_equal(full, np.ones((4, 4), dtype=np.uint8))
                    np.testing.assert_array_equal(tile, np.ones((3, 3), dtype=np.uint8))
                    self.assert_binary(full)
                    self.assert_binary(tile)

        self.assertEqual(len(calls), 6)
        self.assertTrue(all(call['mask_mode'] is True for call in calls))
        self.assertTrue(all(call['volume'] is self.mask for call in calls))

    def test_tilted_radial_all_bases_use_nearest_inplane_and_established_stack_blend(self) -> None:
        group = TiltedViewGroup(
            views=('transverse', 'sagittal', 'coronal'),
            tilt_angles=(25.0,),
            tilt_directions=('horizontal',),
        )
        views = geometry.get_view_infos(
            5,
            6,
            7,
            cartesian_views=(),
            radial_views=('tilted_transverse', 'tilted_sagittal', 'tilted_coronal'),
            radial_azimuth_angles=(60.0, 60.0, 60.0),
            tilt_groups=(group,),
            radial_native_raster=4,
        )
        positive_views = [
            view
            for view in views
            if geometry.is_tilted_radial_view(view) and float(view.tilt_angle_deg) > 0.0
        ]
        self.assertEqual(
            {geometry.radial_base_view_name(view) for view in positive_views},
            {'transverse', 'sagittal', 'coronal'},
        )

        for view in positive_views:
            with self.subTest(base=geometry.radial_base_view_name(view)):
                frame_idx = 1
                sampler = geometry.get_radial_sampler(view, view.azimuths_deg[frame_idx])
                px = sampler.nn_x.astype(np.intp, copy=False)
                py = sampler.nn_y.astype(np.intp, copy=False)
                stack_len = geometry.radial_stack_length(view)
                row_centers = geometry._tilted_radial_row_centers(stack_len, view.src_h)
                offsets = px.astype(np.float32) - np.float32(view.center_x)
                stack_src = row_centers[:, None] + np.float32(
                    np.tan(np.radians(view.tilt_angle_deg))
                ) * offsets[None, :]
                valid = (stack_src >= 0.0) & (stack_src <= float(stack_len - 1))
                s0 = np.clip(np.floor(stack_src).astype(np.int32), 0, stack_len - 1)
                s1 = np.minimum(s0 + 1, stack_len - 1)
                alpha = (stack_src - s0).astype(np.float32)

                base = geometry.radial_base_view_name(view)
                if base == 'transverse':
                    f0 = self.mask[s0, py[None, :], px[None, :]] > 0
                    f1 = self.mask[s1, py[None, :], px[None, :]] > 0
                elif base == 'sagittal':
                    f0 = self.mask[py[None, :], s0, px[None, :]] > 0
                    f1 = self.mask[py[None, :], s1, px[None, :]] > 0
                else:
                    f0 = self.mask[py[None, :], px[None, :], s0] > 0
                    f1 = self.mask[py[None, :], px[None, :], s1] > 0
                blended = f0.astype(np.float32) + alpha * (
                    f1.astype(np.float32) - f0.astype(np.float32)
                )
                expected = np.asarray(valid & (blended >= 0.5), dtype=np.uint8)

                actual = geometry.get_categorical_view_frame_by_index(
                    self.mask, view, frame_idx,
                )
                np.testing.assert_array_equal(actual, expected)
                self.assert_binary(actual)

    def test_tilted_radial_stack_blend_includes_exact_half_foreground(self) -> None:
        view = geometry.ViewInfo(
            name='radial_tilted_transverse_vertical_p30',
            num_slices=1,
            src_h=1,
            src_w=1,
            pad_mode='pad',
            family=geometry.RADIAL_VIEW_FAMILY,
            azimuths_deg=(0.0,),
            diameter=1,
            center_x=0.0,
            center_y=0.0,
            roi_radius=0.0,
            full_t=2,
            full_h=1,
            full_w=1,
            tilt_angle_deg=30.0,
            tilt_direction='vertical',
            tilt_base_view='transverse',
            radial_base_view='transverse',
            radial_tilted_source=True,
        )
        sampler = geometry.get_radial_sampler(view, 0.0)
        mask = np.asarray([[[0]], [[1]]], dtype=np.uint8)

        actual = geometry.extract_tilted_radial_categorical_slice_frame(
            mask, view, sampler, out_rows=1,
        )
        np.testing.assert_array_equal(actual, np.ones((1, 1), dtype=np.uint8))

    def test_fullframe_and_dense_tile_cartesian_affines_are_nearest_and_binary(self) -> None:
        view = geometry.get_view_infos(
            1,
            3,
            3,
            cartesian_views=('transverse',),
            radial_views=(),
            radial_azimuth_angles=(),
        )[0]
        mask = np.asarray([[[0, 4, 0], [9, 0, 0], [0, 0, 5]]], dtype=np.uint8)
        calls: list[dict[str, object]] = []
        nearest = 17
        border_constant = 23

        def fake_warp(
            source: np.ndarray,
            matrix: np.ndarray,
            dsize: tuple[int, int],
            *,
            flags: int,
            borderMode: int,
            borderValue: int,
        ) -> np.ndarray:
            calls.append({
                'source': source.copy(),
                'matrix': matrix,
                'dsize': dsize,
                'flags': flags,
                'borderMode': borderMode,
                'borderValue': borderValue,
            })
            out = np.zeros((int(dsize[1]), int(dsize[0])), dtype=np.uint8)
            h = min(out.shape[0], source.shape[0])
            w = min(out.shape[1], source.shape[1])
            out[:h, :w] = np.asarray(source[:h, :w])
            return out

        job = _aug_job(view, out_size=2, angle_deg=15.0)
        tile_job = _tile_job(view, out_size=2)
        with (
            mock.patch.object(geometry.cv2, 'warpAffine', side_effect=fake_warp),
            mock.patch.object(geometry.cv2, 'INTER_NEAREST', nearest, create=True),
            mock.patch.object(geometry.cv2, 'BORDER_CONSTANT', border_constant, create=True),
        ):
            full = geometry.render_categorical_fullframe_for_job(
                mask, view, job, frame_idx=0,
            )
            tile = geometry.render_categorical_dense_tile_for_job(
                mask, view, tile_job, frame_idx=0,
            )

        self.assertEqual([call['flags'] for call in calls], [nearest, nearest])
        self.assertEqual(
            [call['borderMode'] for call in calls],
            [border_constant, border_constant],
        )
        self.assertTrue(all(set(np.unique(call['source'])).issubset({0, 1}) for call in calls))
        self.assert_binary(full)
        self.assert_binary(tile)


if __name__ == '__main__':
    unittest.main()
