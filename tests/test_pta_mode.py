from __future__ import annotations

import contextlib
import importlib
import io
import types
import unittest
from unittest import mock

from XTA import pta_runtime


class PtaModeBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pta_mode = importlib.import_module("XTA.pta_mode")

    def test_help_and_unavailable_flags_do_not_import_runtime(self) -> None:
        with mock.patch.object(self.pta_mode.importlib, "import_module") as import_module:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.pta_mode.run(["--help"])
            self.assertEqual(raised.exception.code, 0)
            import_module.assert_not_called()

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.pta_mode.run(["--input", "dataset", "--resume"])
            self.assertEqual(raised.exception.code, 2)
            import_module.assert_not_called()

    def test_resolved_configuration_is_forwarded_to_native_runtime(self) -> None:
        arguments = [
            "--input", "dataset", "--output", "published", "--imgsz", "640",
            "--device", "2,0",
            "--output_format", "jpeg", "--channel_format", "C5S2", "--force",
            "--preprocessing", "gaussian_smoothing:1.5:2",
            "--save", "images", "labels", "nrrd", "overlay", "voxel_volume", "summary",
            "--train_split", "0.7", "--split_method", "slice",
            "--background_percent", "0.25", "--augmentation", "policy.py",
            "--augmentation_ratio", "2.5", "--augmentation_execution", "offline",
            "--offline_augmentation_backend", "gpu", "--gpu_batch_size", "7",
            "--enable_cartesian", "sagittal,transverse",
            "--enable_radial", "transverse:2.5", "tilted_coronal:auto",
            "--enable_tilted", "coronal:15:horizontal",
            "--enable_tile", "512:256", "128:64", "--workers", "4",
            "--frame_workers", "3", "--png_compression", "5",
            "--overlay_tile_writer_limit", "9", "--overlay_workers", "2",
            "--overlay_pending_frames", "8", "--worker_backend", "thread",
            "--pipeline_depth", "1", "--jpeg_decode_backend", "opencv",
            "--jpeg_batch_size", "17", "--jpeg_encode_backend", "opencv",
            "--tiff_encode_backend", "nvtiff",
            "--jpeg_quality", "88", "--no-topology_aware",
        ]
        runtime = types.SimpleNamespace(run=mock.Mock())
        with mock.patch.object(self.pta_mode, "_load_runtime_module", return_value=runtime):
            self.pta_mode.run(arguments)

        runtime.run.assert_called_once()
        config = runtime.run.call_args.args[0]
        self.assertEqual(runtime.run.call_args.kwargs["argv"], arguments)
        options = pta_runtime.build_runtime_options(config)

        expected_direct = {
            "input": "dataset", "output": "published", "imgsz": 640,
            "device": ["2,0"], "device_ids": (2, 0),
            "output_format": "tif", "force": True, "train_split": 0.7,
            "split_method": "slice", "background_percent": 0.25,
            "augmentation": "policy.py", "augmentation_ratio": 2.5,
            "augmentation_execution": "offline", "offline_augmentation_backend": "gpu",
            "gpu_batch_size": 7, "workers": 4, "frame_workers": 3,
            "png_compression": 5, "overlay_tile_writer_limit": 9,
            "overlay_workers": 2, "overlay_pending_frames": 8,
            "worker_backend": "thread", "pipeline_depth": 1,
            "jpeg_decode_backend": "opencv", "jpeg_batch_size": 17,
            "jpeg_encode_backend": "opencv", "jpeg_quality": 88,
            "tiff_encode_backend": "nvtiff",
            "topology_aware": False,
        }
        for name, expected in expected_direct.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(options, name), expected)

        self.assertEqual(options.channel_format, ["C5S2"])
        self.assertEqual(options.gaussian_smoothing, 1.5)
        self.assertEqual(options.gaussian_smoothing_passes, 2)
        self.assertEqual(options.tile_size, ["512", "128"])
        self.assertEqual(options.tile_stride, ["256", "64"])
        self.assertTrue(options.save_images)
        self.assertTrue(options.save_labels)
        self.assertTrue(options.save_nrrd)
        self.assertTrue(options.save_overlay)
        self.assertTrue(options.voxel_volume)
        self.assertTrue(options.save_summary)
        self.assertEqual(options.max_pending_frames, 0)
        self.assertEqual(options.tile_task_chunk, 1)
        self.assertEqual(options.aug_task_chunk, 4)
        self.assertFalse(options.resume)
        self.assertIs(options._v18_config, config)

    def test_output_format_aliases_are_case_insensitive_and_canonical(self) -> None:
        cases = {"PNG": "png", ".pNg": "png", "JPG": "jpg", ".jPeG": "jpg", "TIF": "tif", ".TiFf": "tif"}
        for supplied, canonical in cases.items():
            with self.subTest(supplied=supplied):
                config = self.pta_mode.parse_pta_args(["--input", "dataset", "--output_format", supplied])
                options = pta_runtime.build_runtime_options(config)
                self.assertEqual(config.args.output_format, canonical)
                self.assertEqual(options.output_format, canonical)

    def test_png_compression_is_accepted_for_effective_non_png_formats(self) -> None:
        cases = (
            (["--output_format", "jpeg"], "jpg"),
            (["--output_format", ".TIFF"], "tif"),
            (["--output_format", "png", "--channel_format", "C3S1"], "tif"),
        )
        for format_arguments, effective_format in cases:
            with self.subTest(arguments=format_arguments):
                config = self.pta_mode.parse_pta_args(["--input", "dataset", *format_arguments, "--png_compression", "99"])
                options = pta_runtime.build_runtime_options(config)
                self.assertEqual(options.output_format, effective_format)
                self.assertEqual(options.png_compression, 99)

    def test_absent_preprocessing_tiles_and_save_tokens_disable_outputs(self) -> None:
        options = pta_runtime.build_runtime_options(self.pta_mode.parse_pta_args(["--input", "dataset"]))
        self.assertEqual(options.gaussian_smoothing, 0.0)
        self.assertEqual(options.gaussian_smoothing_passes, 0)
        self.assertEqual(options.tile_size, ["0"])
        self.assertIsNone(options.tile_stride)
        self.assertFalse(options.save_images)
        self.assertFalse(options.save_labels)
        self.assertFalse(options.save_nrrd)
        self.assertFalse(options.save_overlay)
        self.assertFalse(options.voxel_volume)
        self.assertFalse(options.save_summary)


if __name__ == "__main__":
    unittest.main()
