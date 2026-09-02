from __future__ import annotations

import contextlib
import io
import unittest

from XTA.pta_config import (
    build_pta_argparser,
    parse_pta_args,
    resolve_preprocessing_options,
    resolve_pta_device_ids,
    resolve_pta_save_request,
    resolve_tile_requests,
)


class PtaConfigTests(unittest.TestCase):
    def test_no_view_and_no_publication_are_valid(self) -> None:
        config = parse_pta_args(["--input", "dataset"])

        self.assertFalse(config.has_physical_views)
        self.assertEqual(config.save.tokens, ())
        self.assertFalse(config.preprocessing.gaussian_smoothing_enabled)
        self.assertEqual(config.args.imgsz, 0)
        self.assertIsNone(config.args.device)
        self.assertIsNone(config.device_ids)

    def test_device_selects_logical_cuda_subset_without_a_default(self) -> None:
        self.assertEqual(
            resolve_pta_device_ids(["2,0", "cuda:1", "gpu:2"]),
            (2, 0, 1),
        )
        config = parse_pta_args(
            ["--input", "dataset", "--device", "2,0", "cuda:1"]
        )
        self.assertEqual(config.args.device, ["2,0", "cuda:1"])
        self.assertEqual(config.device_ids, (2, 0, 1))

        for invalid in ("cpu", "-1", "cuda:x"):
            with self.subTest(invalid=invalid):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_pta_args(
                            ["--input", "dataset", "--device", invalid]
                        )

    def test_gaussian_is_opt_in_with_tta_defaults(self) -> None:
        disabled = resolve_preprocessing_options(None)
        enabled = resolve_preprocessing_options("gaussian_smoothing")
        two_passes = resolve_preprocessing_options("gaussian_smoothing::2")

        self.assertFalse(disabled.gaussian_smoothing_enabled)
        self.assertEqual((enabled.gaussian_sigma, enabled.gaussian_passes), (3.0, 1))
        self.assertEqual((two_passes.gaussian_sigma, two_passes.gaussian_passes), (3.0, 2))

    def test_one_channel_format_and_pta_save_contract(self) -> None:
        config = parse_pta_args([
            "--input",
            "dataset",
            "--channel_format",
            "C5S2",
            "--save",
            "images",
            "labels,summary",
        ])

        self.assertEqual(config.channel_format.offsets, (-4, -2, 0, 2, 4))
        self.assertEqual(config.save.tokens, ("images", "labels", "summary"))
        with self.assertRaises(ValueError):
            resolve_pta_save_request("images,images")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_pta_args(
                    [
                        "--input",
                        "dataset",
                        "--channel_format",
                        "gray",
                        "--channel_format=C3S1",
                    ]
                )

    def test_grouped_geometry_uses_tta_semantics(self) -> None:
        config = parse_pta_args([
            "--input",
            "dataset",
            "--enable_cartesian",
            "sagittal,transverse",
            "--enable_tilted",
            "coronal:15:horizontal",
            "--enable_radial",
            "transverse:2.5",
            "tilted_coronal:auto",
            "--enable_tile",
            "512:256",
        ])

        self.assertEqual(config.cartesian_views, ("sagittal", "transverse"))
        self.assertEqual(
            [(request.view, request.azimuth_angle) for request in config.radial_requests],
            [("transverse", 2.5), ("tilted_coronal", None)],
        )
        self.assertEqual(config.tilted_groups[0].views, ("coronal",))
        self.assertEqual(config.tiles[0].tile_stride, 256)

    def test_tile_values_are_strict_and_covered(self) -> None:
        self.assertEqual(resolve_tile_requests("64:32")[0].config_id, "s64_st32")
        for invalid in ("64", "64:65", "64.5:32", "0:1", "64:0"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_tile_requests(invalid)

    def test_removed_and_tta_only_flags_are_not_available(self) -> None:
        parser = build_pta_argparser()
        removed = (
            ("--resume",),
            ("--max_pending_frames", "1"),
            ("--render_queue_depth", "1"),
            ("--tile_task_chunk", "1"),
            ("--aug_task_chunk", "1"),
            ("--angle", "0"),
            ("--interpolation_distance", "15"),
            ("--enable_sagittal",),
            ("--azimuth_angle", "2"),
            ("--tile_size", "256"),
        )
        for extra in removed:
            with self.subTest(extra=extra):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["--input", "dataset", *extra])


if __name__ == "__main__":
    unittest.main()
