from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from volume_tta import cuda_backend, geometry
from volume_tta.config import resolve_channel_format


class RadialChannelSeamTests(unittest.TestCase):
    @staticmethod
    def _radial_view(num_slices: int = 4) -> geometry.ViewInfo:
        return geometry.ViewInfo(
            name="radial_transverse",
            num_slices=int(num_slices),
            src_h=2,
            src_w=3,
            pad_mode="pad",
            family="radial",
            azimuths_deg=tuple(float(i * 180.0 / num_slices) for i in range(num_slices)),
        )

    @staticmethod
    def _planes(num_slices: int = 4) -> list[np.ndarray]:
        return [
            np.asarray(
                [[10 * index + 0, 10 * index + 1, 10 * index + 2],
                 [10 * index + 3, 10 * index + 4, 10 * index + 5]],
                dtype=np.uint8,
            )
            for index in range(num_slices)
        ]

    def test_source_resolution_tracks_wrap_parity_without_mirroring_cartesian(self) -> None:
        radial = self._radial_view()

        # No-crossing controls retain both the source index and native orientation.
        self.assertEqual(geometry.channel_view_slice_source(radial, 0), (0, False))
        self.assertEqual(geometry.channel_view_slice_source(radial, 3), (3, False))

        # Both directions across the 0/180 seam reverse radial-u.
        self.assertEqual(geometry.channel_view_slice_source(radial, -1), (3, True))
        self.assertEqual(geometry.channel_view_slice_source(radial, 4), (0, True))

        # Arbitrary custom strides can span whole periods: only odd wrap parity mirrors.
        self.assertEqual(geometry.channel_view_slice_source(radial, -5), (3, False))
        self.assertEqual(geometry.channel_view_slice_source(radial, 8), (0, False))

        cartesian = geometry.ViewInfo("transverse", 4, 2, 3, "clamp")
        self.assertEqual(geometry.channel_view_slice_source(cartesian, -1), (0, False))
        self.assertEqual(geometry.channel_view_slice_source(cartesian, 4), (3, False))

    def test_renderer_mirrors_only_width_axis_at_negative_and_positive_seams(self) -> None:
        view = self._radial_view()
        planes = self._planes()
        renderer = geometry.ChannelFormattedFrameRenderer(
            lambda index: planes[int(index)],
            view,
            resolve_channel_format("C3S1"),
            cache_frames=0,
        )

        negative_crossing = renderer(0)
        np.testing.assert_array_equal(negative_crossing[:, :, 0], planes[3][:, ::-1])
        np.testing.assert_array_equal(negative_crossing[:, :, 1], planes[0])
        np.testing.assert_array_equal(negative_crossing[:, :, 2], planes[1])
        # Row order is unchanged: only radial-u (the frame-width axis) reverses.
        np.testing.assert_array_equal(negative_crossing[0, :, 0], planes[3][0, ::-1])
        np.testing.assert_array_equal(negative_crossing[1, :, 0], planes[3][1, ::-1])

        positive_crossing = renderer(3)
        np.testing.assert_array_equal(positive_crossing[:, :, 0], planes[2])
        np.testing.assert_array_equal(positive_crossing[:, :, 1], planes[3])
        np.testing.assert_array_equal(positive_crossing[:, :, 2], planes[0][:, ::-1])

        no_crossing = renderer(1)
        for channel, expected in enumerate(planes[:3]):
            np.testing.assert_array_equal(no_crossing[:, :, channel], expected)

    def test_cache_keeps_mirrored_and_unmirrored_orientation_separate(self) -> None:
        view = self._radial_view(num_slices=1)
        plane = self._planes(num_slices=1)[0]
        calls: list[int] = []

        def render(index: int) -> np.ndarray:
            calls.append(int(index))
            return plane

        renderer = geometry.ChannelFormattedFrameRenderer(
            render,
            view,
            resolve_channel_format("C3S1"),
            cache_frames=4,
        )
        frame = renderer(0)

        np.testing.assert_array_equal(frame[:, :, 0], plane[:, ::-1])
        np.testing.assert_array_equal(frame[:, :, 1], plane)
        np.testing.assert_array_equal(frame[:, :, 2], plane[:, ::-1])
        # The two wrapped channels share their mirrored cache entry, while the center owns
        # a separate unmirrored entry even though all three resolve to source index zero.
        self.assertEqual(calls, [0, 0])

    def test_fullframe_and_tile_factories_request_native_pre_affine_mirror(self) -> None:
        view = self._radial_view()
        planes = self._planes()
        fmt = resolve_channel_format("C3S1")

        fullframe_calls: list[tuple[int, bool]] = []

        def fake_fullframe(*args: object, **kwargs: object) -> np.ndarray:
            index = int(kwargs["frame_idx"])
            mirror_u = bool(kwargs.get("mirror_radial_u", False))
            fullframe_calls.append((index, mirror_u))
            plane = planes[index]
            return np.ascontiguousarray(plane[:, ::-1] if mirror_u else plane)

        with mock.patch.object(
            geometry, "render_fullframe_frame_for_job", side_effect=fake_fullframe,
        ):
            renderer = geometry.make_fullframe_channel_renderer(
                np.zeros((1, 1, 1), dtype=np.uint8),
                view,
                mock.sentinel.fullframe_job,
                channel_format=fmt,
                cache_frames=0,
            )
            rendered = renderer(0)

        self.assertEqual(fullframe_calls, [(3, True), (0, False), (1, False)])
        np.testing.assert_array_equal(rendered[:, :, 0], planes[3][:, ::-1])

        tile_calls: list[tuple[int, bool]] = []

        def fake_tile(*args: object, **kwargs: object) -> np.ndarray:
            index = int(kwargs["frame_idx"])
            mirror_u = bool(kwargs.get("mirror_radial_u", False))
            tile_calls.append((index, mirror_u))
            plane = planes[index]
            return np.ascontiguousarray(plane[:, ::-1] if mirror_u else plane)

        with mock.patch.object(
            geometry, "render_dense_tile_frame_for_job", side_effect=fake_tile,
        ):
            renderer = geometry.make_dense_tile_channel_renderer(
                np.zeros((1, 1, 1), dtype=np.uint8),
                view,
                mock.sentinel.tile_job,
                channel_format=fmt,
                cache_frames=0,
            )
            rendered = renderer(3)

        self.assertEqual(tile_calls, [(2, False), (3, False), (0, True)])
        np.testing.assert_array_equal(rendered[:, :, 2], planes[0][:, ::-1])

    def test_slab_renderer_mirrors_native_plane_before_job_affine(self) -> None:
        view = geometry.ViewInfo(
            name="radial_transverse",
            num_slices=4,
            src_h=3,
            src_w=3,
            pad_mode="pad",
            family="radial",
            azimuths_deg=(0.0, 45.0, 90.0, 135.0),
        )
        planes = np.stack(
            [
                np.arange(9, dtype=np.uint8).reshape(3, 3) + np.uint8(10 * index)
                for index in range(4)
            ],
            axis=0,
        )
        identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        aff = geometry.AffineSpec(
            view=view.name,
            angle_deg=15.0,  # force the affine branch
            src_w=3,
            src_h=3,
            out_size=3,
            canvas_w=3,
            canvas_h=3,
            pad_size=3,
            pad_off_x=0.0,
            pad_off_y=0.0,
            M_out_to_src=identity,
            M_src_to_out=identity,
            M_canvas_to_src=identity,
            M_src_to_canvas=identity,
        )
        job = geometry.AugJob("a15", 15.0, Path("unused.json"), aff)

        warped_inputs: list[np.ndarray] = []

        def fake_warp(plane: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
            warped_inputs.append(np.asarray(plane).copy())
            return np.asarray(plane).copy()

        with mock.patch.object(cuda_backend.cv2, "warpAffine", side_effect=fake_warp):
            renderer = cuda_backend._radial_slab_channel_renderer(
                planes,
                tuple(range(4)),
                view,
                job,
                center_start=0,
                channel_format=resolve_channel_format("C3S1"),
            )
            rendered = renderer(0)

        np.testing.assert_array_equal(rendered[:, :, 0], planes[3][:, ::-1])
        # Four ordinary bank transforms occur at setup; the boundary request then supplies
        # the native-u-mirrored plane as the affine input, rather than flipping its output.
        np.testing.assert_array_equal(warped_inputs[-1], planes[3][:, ::-1])

    def test_cuda_source_coordinate_composition_reverses_only_u_row(self) -> None:
        matrix = np.asarray(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32,
        )
        mirrored = cuda_backend._GpuWorkerRenderEngine._mirror_radial_u_out_to_src(
            matrix, source_width=11,
        )
        np.testing.assert_array_equal(mirrored[0], np.asarray([-2.0, -3.0, 6.0]))
        np.testing.assert_array_equal(mirrored[1], matrix[1])
        np.testing.assert_array_equal(matrix[0], np.asarray([2.0, 3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
