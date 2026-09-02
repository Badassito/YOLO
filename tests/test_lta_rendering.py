from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.smoke_import import install_stubs

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    install_stubs()

from XTA.geometry import expand_views_into_tta_variants, get_view_infos
from XTA.lta_rendering import build_lta_rendered_view, implicit_rgb
from XTA.runtime import close_memmap_array


class LtaRenderingTests(unittest.TestCase):
    def test_implicit_rgb_is_exact_and_rejects_non_uint8(self) -> None:
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        rgb = implicit_rgb(gray)

        self.assertEqual(rgb.shape, (3, 4, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], gray)
        np.testing.assert_array_equal(rgb[:, :, 1], gray)
        np.testing.assert_array_equal(rgb[:, :, 2], gray)
        with self.assertRaisesRegex(ValueError, "uint8"):
            implicit_rgb(gray.astype(np.float32))

    @unittest.skipIf(type(__import__("sys").modules.get("cv2")).__name__ == "_StubModule", "requires OpenCV")
    def test_transverse_identity_render_mask_restore_and_native_projection(self) -> None:
        volume = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
        physical = get_view_infos(
            T=3,
            H=4,
            W=4,
            cartesian_views=("transverse",),
            radial_views=(),
            radial_azimuth_angles=(),
            tilt_groups=(),
        )[0]
        runtime = expand_views_into_tta_variants((physical,), (0.0,))[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            rendered = build_lta_rendered_view(
                volume,
                runtime,
                temp_dir=Path(temp_dir),
                output_size=4,
            )
            frame = rendered.render_frame_rgb(1)
            model_mask = np.zeros((4, 4), dtype=bool)
            model_mask[2, 1] = True
            view_masks = rendered.restore_model_masks_to_view({1: model_mask})
            native = rendered.project_view_masks_to_native(
                view_masks,
                out_path=Path(temp_dir) / "native.u8.dat",
            )
            native_copy = np.array(native, copy=True)
            close_memmap_array(native)

        self.assertEqual(rendered.raster_plan.mode.value, "lta")
        self.assertEqual(frame.shape, (4, 4, 3))
        np.testing.assert_array_equal(frame[:, :, 0], volume[1])
        self.assertEqual(int(view_masks[1, 2, 1]), 1)
        np.testing.assert_array_equal(native_copy, view_masks)


if __name__ == "__main__":
    unittest.main()
