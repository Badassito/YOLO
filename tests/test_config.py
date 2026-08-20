from __future__ import annotations

import unittest

from volume_tta.config import (
    resolve_backend_batches,
    resolve_backend_devices,
    resolve_backend_models,
    resolve_backend_precisions,
    resolve_channel_format,
    resolve_tta_angles,
)


class ConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
