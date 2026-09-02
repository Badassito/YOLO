from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "v19_lta_gpu_smoke.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("v19_lta_gpu_smoke", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V19LtaGpuSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    def test_zero_based_label_row_is_converted_to_top_left_xywh(self) -> None:
        bounds = (
            0.675478835978836,
            0.3001426240208877,
            0.9722883597883597,
            0.4890078328981722,
        )
        min_x, min_y, max_x, max_y = bounds
        polygon = (
            f"0 {min_x} {min_y} {max_x} {min_y} "
            f"{max_x} {max_y} {min_x} {max_y}"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            label = Path(temp_dir) / "exemplar.txt"
            label.write_text("\n" * 17 + polygon + "\n", encoding="utf-8")

            xywh = self.tool.prompt_xywh_from_label(label, 17)

        expected = (min_x, min_y, max_x - min_x, max_y - min_y)
        for actual, wanted in zip(xywh, expected):
            self.assertAlmostEqual(actual, wanted, places=14)

    def test_f1_composite_has_fixed_left_exemplar_and_right_target_geometry(self) -> None:
        from PIL import Image

        target = Image.new("RGB", (302, 306), (0, 255, 0))
        exemplar = Image.new("RGB", (99, 99), (255, 0, 0))
        frames = [target] * self.tool.LTA_SESSION_FRAMES
        prompt = (0.3, 0.2, 0.15, 0.09)

        composites, mapped, target_rect = self.tool.build_f1_composites(
            frames,
            exemplar,
            prompt,
            canvas_size=99,
        )

        self.assertEqual(len(composites), 30)
        self.assertEqual(composites[0].size, (99, 99))
        self.assertEqual(target_rect, (33, 16, 99, 83))
        expected = (prompt[0] / 3, 1 / 3 + prompt[1] / 3, prompt[2] / 3, prompt[3] / 3)
        for actual, wanted in zip(mapped, expected):
            self.assertAlmostEqual(actual, wanted)
        self.assertEqual(composites[0].getpixel((16, 49)), (255, 0, 0))
        self.assertEqual(composites[0].getpixel((66, 49)), (0, 255, 0))

        prompt_only, _, _ = self.tool.build_f1_composites(
            frames,
            exemplar,
            prompt,
            canvas_size=99,
            exemplar_frame_offset=19,
        )
        self.assertEqual(prompt_only[0].getpixel((16, 49)), (128, 128, 128))
        self.assertEqual(prompt_only[19].getpixel((16, 49)), (255, 0, 0))
        self.assertEqual(prompt_only[29].getpixel((16, 49)), (128, 128, 128))

    def test_constrained_gpu_batches_fail_closed_and_disable_frame_batching(self) -> None:
        model = type(
            "Model",
            (),
            {
                "use_batched_grounding": True,
                "batched_grounding_batch_size": 16,
                "postprocess_batch_size": 16,
            },
        )()
        predictor = type("Predictor", (), {"model": model})()

        settings = self.tool.configure_constrained_gpu_batches(predictor)

        self.assertFalse(model.use_batched_grounding)
        self.assertEqual(model.batched_grounding_batch_size, 1)
        self.assertEqual(model.postprocess_batch_size, 1)
        self.assertEqual(settings["postprocess_batch_size"], 1)
        with self.assertRaisesRegex(RuntimeError, "missing constrained batch controls"):
            self.tool.configure_constrained_gpu_batches(
                type("Predictor", (), {"model": object()})()
            )

    def test_sdpa_fallback_is_ordered_and_restorable(self) -> None:
        calls = []

        def original(backends, *, set_priority=False):
            calls.append(tuple(backends))
            calls.append(bool(set_priority))
            return "context"

        backend = type(
            "Backend",
            (),
            {
                "FLASH_ATTENTION": "flash",
                "EFFICIENT_ATTENTION": "efficient",
                "MATH": "math",
            },
        )
        decoder = type(
            "Decoder",
            (),
            {"sdpa_kernel": staticmethod(original), "SDPBackend": backend},
        )

        restore = self.tool.install_sdpa_fallback(decoder)
        self.assertEqual(decoder.sdpa_kernel("flash-only"), "context")
        self.assertEqual(calls, [("flash", "efficient", "math"), True])
        restore()
        self.assertIs(decoder.sdpa_kernel, original)

    def test_window_and_useful_output_acceptance_fail_before_expensive_work(self) -> None:
        with self.assertRaisesRegex(ValueError, "start-frame"):
            self.tool.validate_smoke_window(-1, 0)
        with self.assertRaisesRegex(ValueError, "prompt-frame"):
            self.tool.validate_smoke_window(360, 390)
        self.tool.validate_smoke_window(360, 379)

        empty_m1 = self.tool.evaluate_case_acceptance("m1", {"prediction_count": 0})
        useful_m1 = self.tool.evaluate_case_acceptance("m1", {"prediction_count": 1})
        empty_f1 = self.tool.evaluate_case_acceptance("f1", {"target_prediction_hits": 0})
        useful_f1 = self.tool.evaluate_case_acceptance("f1", {"target_prediction_hits": 1})
        self.assertFalse(empty_m1["passed"])
        self.assertTrue(useful_m1["passed"])
        self.assertFalse(empty_f1["passed"])
        self.assertTrue(useful_f1["passed"])

    def test_review_diagnostics_write_contact_prompt_overlay_and_active_mask(self) -> None:
        import numpy as np
        from PIL import Image

        frames = [Image.new("RGB", (99, 99), (80, 80, 80)) for _ in range(30)]
        mask = np.zeros((99, 99), dtype=bool)
        mask[20:40, 30:50] = True
        prediction = type(
            "Prediction",
            (),
            {"frame_index": 362, "binary_mask": mask},
        )()
        session = self.tool.SamSessionPlan(
            sequence_id="m1__transverse_smoke",
            session_index=0,
            frame_start=360,
            frame_stop=390,
        )
        prompt = self.tool.SamPromptBox(
            exemplar_id="positive",
            frame_index=379,
            xywh=(0.1, 0.2, 0.3, 0.4),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "review"
            record = self.tool.write_case_diagnostics(
                root,
                case="m1",
                resource=frames,
                predictions=(prediction,),
                session=session,
                prompt=prompt,
            )
            self.assertTrue(Path(record["contact_sheet"]).is_file())
            self.assertTrue((root / "m1" / "frame_0362_mask.png").is_file())
            self.assertTrue((root / "m1" / "frame_0362_overlay.png").is_file())
            self.assertTrue((root / "m1" / "frame_0379_overlay.png").is_file())
            self.assertEqual(record["interesting_frames"], [362, 379])


if __name__ == "__main__":
    unittest.main()
