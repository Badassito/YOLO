from __future__ import annotations

import contextlib
import io
import unittest

from volume_tta.config import (
    SAVE_OPTION_TOKENS,
    build_argparser,
    resolve_backend_batches,
    resolve_backend_devices,
    resolve_backend_models,
    resolve_backend_precisions,
    resolve_channel_format,
    resolve_save_request,
    resolve_tta_angles,
)


class ConfigTests(unittest.TestCase):
    def test_v17_1_inference_and_interpolation_defaults(self) -> None:
        args = build_argparser().parse_args([
            '--input', 'input.mkv', '--model', 'gpu:model.engine',
        ])
        self.assertEqual(args.imgsz, 3072)
        self.assertEqual(args.interpolation_walk_back, 1)
        self.assertEqual(args.interpolation_candidates, 1)

    def test_temp_help_requires_explicit_memory_backed_scratch(self) -> None:
        help_text = build_argparser().format_help()
        self.assertIn('/dev/shm', help_text)
        self.assertNotIn('YOLO_TTA_SCRATCH', help_text)

    def test_unknown_processor_and_crop_options_are_rejected(self) -> None:
        parser = build_argparser()
        option_strings = {
            option for action in parser._actions for option in action.option_strings
        }
        unknown_options = [
            "--" + "_".join(("retina", "mask", "processor")),
            "--" + "_".join(("save", "nrrd", "tight", "crop")),
        ]
        help_text = parser.format_help()
        required = ["--input", "input.mkv", "--model", "gpu:model.engine"]
        for underscored in unknown_options:
            for option in (underscored, underscored.replace("_", "-")):
                with self.subTest(option=option):
                    self.assertNotIn(option, option_strings)
                    self.assertNotIn(option, help_text)
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            parser.parse_args([*required, option])

        crop_value = "_".join(("nrrd", "tight", "crop"))
        for value in (crop_value, crop_value.replace("_", "-")):
            with self.subTest(save_value=value):
                self.assertNotIn(value, SAVE_OPTION_TOKENS)
                with self.assertRaises(ValueError):
                    resolve_save_request(value)

    def test_retired_centerline_backend_and_tilt_alias_are_not_registered(self) -> None:
        parser = build_argparser()
        option_strings = {option for action in parser._actions for option in action.option_strings}
        self.assertNotIn('--centerline_filter_backend', option_strings)
        self.assertNotIn('--enable_tilt', option_strings)
        self.assertIn('--enable_tilted', option_strings)
        required = ['--input', 'input.mkv', '--model', 'gpu:model.engine']
        for retired_option, value in (
            ('--centerline_filter_backend', 'off'),
            ('--enable_tilt', 'transverse'),
        ):
            with self.subTest(retired_option=retired_option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args([*required, retired_option, value])

    def test_angles_are_normalized_and_unique(self) -> None:
        self.assertEqual(resolve_tta_angles("-120,0,120"), [240.0, 0.0, 120.0])
        with self.assertRaises(ValueError):
            resolve_tta_angles("0,360")

    def test_channel_layouts(self) -> None:
        self.assertEqual(resolve_channel_format("grey").offsets, (0,))
        self.assertEqual(resolve_channel_format("RGB").offsets, (0, 0, 0))
        custom = resolve_channel_format("c5s2")
        self.assertEqual(custom.token, "C5S2")
        self.assertEqual(custom.offsets, (-4, -2, 0, 2, 4))
        with self.assertRaises(ValueError):
            resolve_channel_format("C4S1")

    def test_current_cpu_cuda_cli_contract_is_preserved(self) -> None:
        devices = resolve_backend_devices(["0,2:cpu"])
        self.assertEqual(devices.gpu_devices, ("cuda:0", "cuda:2"))
        self.assertTrue(devices.cpu)
        models = resolve_backend_models(["cpu:/models/openvino", "gpu:/models/model.engine"])
        self.assertEqual(models.cpu, "/models/openvino")
        self.assertEqual(models.gpu, "/models/model.engine")
        precisions = resolve_backend_precisions(["gpu:fp16", "cpu:bf16"], devices)
        self.assertEqual(precisions.gpu, 16)
        self.assertEqual(precisions.cpu, "bf16")
        batches = resolve_backend_batches(["gpu:4", "cpu:2"], devices)
        self.assertEqual((batches.gpu, batches.cpu), (4, 2))

    def test_unselected_backend_settings_fail_instead_of_being_discarded(self) -> None:
        cpu_only = resolve_backend_devices(["cpu"])
        with self.assertRaisesRegex(ValueError, "selected no GPU backend"):
            resolve_backend_precisions(["gpu:fp16"], cpu_only)
        with self.assertRaisesRegex(ValueError, "selected no GPU backend"):
            resolve_backend_batches(["gpu:8"], cpu_only)

        gpu_only = resolve_backend_devices(["0"])
        with self.assertRaisesRegex(ValueError, "selected no CPU backend"):
            resolve_backend_precisions(["cpu:bf16"], gpu_only)
        with self.assertRaisesRegex(ValueError, "selected no CPU backend"):
            resolve_backend_batches(["cpu:2"], gpu_only)


if __name__ == "__main__":
    unittest.main()
