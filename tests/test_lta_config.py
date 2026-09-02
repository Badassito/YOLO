from __future__ import annotations

import contextlib
import io
import unittest

from XTA.lta_config import (
    build_lta_argparser,
    parse_lta_args,
    resolve_lta_device_ids,
)


class LtaConfigTests(unittest.TestCase):
    REQUIRED = [
        "--input", "target",
        "--output", "published",
        "--model", "sam-bundle",
        "--device", "0",
        "--enable_cartesian", "transverse",
    ]

    def test_complete_public_contract_is_resolved(self) -> None:
        config = parse_lta_args([
            "--input", "target",
            "--output", "published",
            "--temp", "scratch",
            "--model", "sam-bundle",
            "--device", "3,1", "cuda:2", "gpu:1",
            "--exemplar", "positive-a", "positive-b",
            "--enable_cartesian", "sagittal,transverse",
            "--enable_radial", "transverse:2.5", "tilted_coronal:auto",
            "--enable_tilted", "coronal:15:horizontal",
            "--enable_tile", "512:256", "128:64",
            "--angle", "240,0",
            "--sam_execution", "image",
            "--conf", "0.25",
            "--save", "nrrd", "images,labels", "overlay", "voxel_volume", "summary",
            "--postprocessing", "keep_objects:2", "3d_void_fill", "gaussian_smoothing::2",
        ])

        self.assertEqual(config.device_ids, (3, 1, 2))
        self.assertEqual(config.exemplar_dirs, ("positive-a", "positive-b"))
        self.assertEqual(config.cartesian_views, ("sagittal", "transverse"))
        self.assertEqual(
            [(request.view, request.azimuth_angle) for request in config.radial_requests],
            [("transverse", 2.5), ("tilted_coronal", None)],
        )
        self.assertEqual(config.tilted_groups[0].views, ("coronal",))
        self.assertEqual(config.tiles[0].config_id, "s512_st256")
        self.assertEqual(config.angles, (240.0, 0.0))
        self.assertEqual(config.args.sam_execution, "image")
        self.assertEqual(config.args.conf, 0.25)
        self.assertEqual(
            config.save.tokens,
            ("nrrd", "images", "labels", "overlay", "voxel_volume", "summary"),
        )
        self.assertEqual(config.postprocessing.keep_objects, 2)
        self.assertTrue(config.postprocessing.enable_3d_void_fill)
        self.assertTrue(config.postprocessing.gaussian_smoothing_enabled)
        self.assertEqual(config.postprocessing.gaussian_passes, 2)

    def test_defaults_keep_optional_publication_empty(self) -> None:
        config = parse_lta_args(self.REQUIRED)

        self.assertEqual(config.exemplar_dirs, ())
        self.assertEqual(config.angles, (0.0, 120.0, 240.0))
        self.assertEqual(config.args.sam_execution, "video")
        self.assertEqual(config.args.conf, 0.15)
        self.assertEqual(config.save.tokens, ())
        self.assertEqual(config.postprocessing.keep_objects, 0)

    def test_required_flags_and_parent_view_are_strict(self) -> None:
        parser = build_lta_argparser()
        required_without_views = [
            "--input", "target",
            "--output", "published",
            "--model", "sam-bundle",
            "--device", "0",
        ]
        for option, value in (
            ("--input", "target"),
            ("--output", "published"),
            ("--model", "sam-bundle"),
            ("--device", "0"),
        ):
            arguments = list(required_without_views)
            position = arguments.index(option)
            del arguments[position:position + 2]
            with self.subTest(missing=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(arguments)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_lta_args([*required_without_views, "--enable_tile", "128:64"])

    def test_device_confidence_and_exemplars_are_validated(self) -> None:
        self.assertEqual(resolve_lta_device_ids(["2,0", "cuda:1"]), (2, 0, 1))
        for replacement in ("cpu", "-1", "cuda:x"):
            arguments = list(self.REQUIRED)
            arguments[arguments.index("0")] = replacement
            with self.subTest(device=replacement), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_lta_args(arguments)

        for value in ("-0.01", "1.01", "nan", "inf"):
            with self.subTest(conf=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_lta_args([*self.REQUIRED, "--conf", value])

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_lta_args([*self.REQUIRED, "--exemplar", ""])

    def test_v1_intentionally_absent_flags_are_rejected(self) -> None:
        unavailable = (
            ("--prompt", "object"),
            ("--channel_format", "RGB"),
            ("--output_cartesian", "transverse"),
            ("--publish_view", "transverse"),
            ("--tracking_window_frames", "30"),
            ("--window_overlap", "5"),
            ("--multiplex_count", "16"),
            ("--max_num_objects", "128"),
            ("--jpeg_quality", "100"),
        )
        for option, value in unavailable:
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_lta_args([*self.REQUIRED, option, value])

    def test_shared_angle_save_and_postprocessing_grammars_remain_strict(self) -> None:
        invalid_tails = (
            ("--angle", "0,360"),
            ("--save", "binary"),
            ("--save", "images,images"),
            ("--postprocessing", "unknown"),
        )
        for option, value in invalid_tails:
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_lta_args([*self.REQUIRED, option, value])


if __name__ == "__main__":
    unittest.main()
