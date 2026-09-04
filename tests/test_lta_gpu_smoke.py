from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "lta_gpu_smoke.py"


class _FakeDistribution:
    def __init__(self, root: Path, *, direct_text: str | None, version: str = "0.1.0"):
        self.root = Path(root)
        self.direct_text = direct_text
        self.version = version

    def locate_file(self, name: str) -> Path:
        return self.root / name

    def read_text(self, name: str) -> str | None:
        return self.direct_text if name == "direct_url.json" else None


def _load_tool():
    spec = importlib.util.spec_from_file_location("lta_gpu_smoke", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LtaGpuSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    @staticmethod
    def _write_fake_sam_tree(root: Path, *, newline: bytes = b"\n") -> Path:
        package = Path(root) / "sam3"
        (package / "model").mkdir(parents=True)
        (package / "__init__.py").write_bytes(b"VERSION = 'test'" + newline)
        (package / "model" / "runtime.py").write_bytes(
            newline.join((b"def run():", b"    return 1", b""))
        )
        return package

    def test_source_tree_fingerprint_is_newline_stable(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self._write_fake_sam_tree(Path(first), newline=b"\n")
            self._write_fake_sam_tree(Path(second), newline=b"\r\n")
            left = self.tool._canonical_sam_package_fingerprint(
                _FakeDistribution(Path(first), direct_text=None)
            )
            right = self.tool._canonical_sam_package_fingerprint(
                _FakeDistribution(Path(second), direct_text=None)
            )

        self.assertEqual(left[:2], right[:2])
        self.assertEqual(left[1], 2)

    def test_runtime_provenance_accepts_pinned_wheel_without_direct_url(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_fake_sam_tree(root)
            distribution = _FakeDistribution(root, direct_text=None)
            digest, count, package = self.tool._canonical_sam_package_fingerprint(
                distribution
            )
            bpe = root / "bpe.gz"
            bpe.write_bytes(b"pinned bpe")
            with (
                mock.patch.object(
                    self.tool.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
                mock.patch.object(self.tool, "resolve_installed_sam_bpe", return_value=bpe),
                mock.patch.object(
                    self.tool.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(package / "__init__.py")),
                ),
                mock.patch.object(self.tool, "PINNED_SAM_PACKAGE_TREE_SHA256", digest),
                mock.patch.object(self.tool, "PINNED_SAM_PACKAGE_FILE_COUNT", count),
            ):
                result = self.tool.resolve_pinned_sam_runtime_provenance()

        self.assertIsNone(result["git_commit"])
        self.assertEqual(result["pinned_git_commit"], self.tool.PINNED_SAM_COMMIT)
        self.assertEqual(result["provenance_method"], "pinned_installed_source_tree")
        self.assertEqual(result["package_tree_sha256"], digest)
        self.assertFalse(result["direct_url_present"])
        self.assertIsNone(result["source_url"])
        self.assertEqual(result["pinned_source_url"], self.tool.PINNED_SAM_SOURCE_URL)

    def test_runtime_provenance_accepts_exact_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_fake_sam_tree(root)
            distribution = _FakeDistribution(
                root,
                direct_text=json.dumps(
                    {
                        "url": self.tool.PINNED_SAM_SOURCE_URL,
                        "vcs_info": {
                            "commit_id": self.tool.PINNED_SAM_COMMIT,
                            "vcs": "git",
                        },
                    }
                ),
            )
            digest, count, package = self.tool._canonical_sam_package_fingerprint(
                distribution
            )
            bpe = root / "bpe.gz"
            bpe.write_bytes(b"pinned bpe")
            with (
                mock.patch.object(
                    self.tool.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
                mock.patch.object(self.tool, "resolve_installed_sam_bpe", return_value=bpe),
                mock.patch.object(
                    self.tool.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(package / "__init__.py")),
                ),
                mock.patch.object(self.tool, "PINNED_SAM_PACKAGE_TREE_SHA256", digest),
                mock.patch.object(self.tool, "PINNED_SAM_PACKAGE_FILE_COUNT", count),
            ):
                result = self.tool.resolve_pinned_sam_runtime_provenance()

        self.assertEqual(result["git_commit"], self.tool.PINNED_SAM_COMMIT)
        self.assertEqual(result["provenance_method"], "pep610_vcs_plus_source_tree")

    def test_runtime_provenance_rejects_import_shadowing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_fake_sam_tree(root)
            distribution = _FakeDistribution(root, direct_text=None)
            digest, count, _package = self.tool._canonical_sam_package_fingerprint(
                distribution
            )
            shadow = root / "shadow" / "sam3" / "__init__.py"
            shadow.parent.mkdir(parents=True)
            shadow.write_text("# shadow\n", encoding="utf-8")
            with (
                mock.patch.object(
                    self.tool.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
                mock.patch.object(
                    self.tool.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(shadow)),
                ),
                mock.patch.object(self.tool, "PINNED_SAM_PACKAGE_TREE_SHA256", digest),
                mock.patch.object(self.tool, "PINNED_SAM_PACKAGE_FILE_COUNT", count),
            ):
                with self.assertRaisesRegex(RuntimeError, "outside the audited"):
                    self.tool.resolve_pinned_sam_runtime_provenance()

    def test_runtime_provenance_rejects_wrong_distribution_version_first(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            distribution = _FakeDistribution(
                Path(folder), direct_text=None, version="0.2.0"
            )
            with mock.patch.object(
                self.tool.importlib.metadata,
                "distribution",
                return_value=distribution,
            ):
                with self.assertRaisesRegex(RuntimeError, "version='0.2.0'"):
                    self.tool.resolve_pinned_sam_runtime_provenance()

    def test_runtime_provenance_rejects_explicit_wrong_vcs_commit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_fake_sam_tree(root)
            distribution = _FakeDistribution(
                root,
                direct_text=json.dumps(
                    {
                        "url": self.tool.PINNED_SAM_SOURCE_URL,
                        "vcs_info": {"commit_id": "f" * 40, "vcs": "git"},
                    }
                ),
            )
            with mock.patch.object(
                self.tool.importlib.metadata,
                "distribution",
                return_value=distribution,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    self.tool.resolve_pinned_sam_runtime_provenance()

    def test_runtime_provenance_rejects_unpinned_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_fake_sam_tree(root)
            distribution = _FakeDistribution(root, direct_text="")
            _digest, count, package = self.tool._canonical_sam_package_fingerprint(
                distribution
            )
            with (
                mock.patch.object(
                    self.tool.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
                mock.patch.object(
                    self.tool.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(package / "__init__.py")),
                ),
                mock.patch.object(self.tool, "PINNED_SAM_PACKAGE_TREE_SHA256", "0" * 64),
                mock.patch.object(self.tool, "PINNED_SAM_PACKAGE_FILE_COUNT", count),
            ):
                with self.assertRaisesRegex(RuntimeError, "package tree"):
                    self.tool.resolve_pinned_sam_runtime_provenance()

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

    def test_video_discovery_is_sample_neutral_and_supports_explicit_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            only = root / "arbitrary-specimen.mkv"
            only.write_bytes(b"video")
            self.assertEqual(self.tool.find_case_video(root, "direct"), only.resolve())
            only.unlink()
            direct = root / "direct_source.mkv"
            composite = root / "composite_source.mkv"
            direct.write_bytes(b"direct")
            composite.write_bytes(b"composite")
            self.assertEqual(self.tool.find_case_video(root, "direct"), direct.resolve())
            self.assertEqual(
                self.tool.find_case_video(root, "composite"),
                composite.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "unknown smoke case"):
                self.tool.find_case_video(root, "named-sample")

    def test_composite_has_fixed_left_exemplar_and_right_target_geometry(self) -> None:
        from PIL import Image

        target = Image.new("RGB", (302, 306), (0, 255, 0))
        exemplar = Image.new("RGB", (99, 99), (255, 0, 0))
        frames = [target] * self.tool.LTA_SESSION_FRAMES
        prompt = (0.3, 0.2, 0.15, 0.09)

        composites, mapped, target_rect = self.tool.build_composite_frames(
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

        prompt_only, _, _ = self.tool.build_composite_frames(
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

        empty_direct = self.tool.evaluate_case_acceptance("direct", {"prediction_count": 0})
        useful_direct = self.tool.evaluate_case_acceptance("direct", {"prediction_count": 1})
        empty_composite = self.tool.evaluate_case_acceptance(
            "composite", {"target_prediction_hits": 0}
        )
        useful_composite = self.tool.evaluate_case_acceptance(
            "composite", {"target_prediction_hits": 1}
        )
        self.assertFalse(empty_direct["passed"])
        self.assertTrue(useful_direct["passed"])
        self.assertFalse(empty_composite["passed"])
        self.assertTrue(useful_composite["passed"])

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
            sequence_id="lta__transverse_smoke",
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
                case="direct",
                resource=frames,
                predictions=(prediction,),
                session=session,
                prompt=prompt,
            )
            self.assertTrue(Path(record["contact_sheet"]).is_file())
            self.assertTrue((root / "direct" / "frame_0362_mask.png").is_file())
            self.assertTrue((root / "direct" / "frame_0362_overlay.png").is_file())
            self.assertTrue((root / "direct" / "frame_0379_overlay.png").is_file())
            self.assertEqual(record["interesting_frames"], [362, 379])


if __name__ == "__main__":
    unittest.main()
