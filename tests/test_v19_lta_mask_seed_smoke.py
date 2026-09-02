from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "v19_lta_mask_seed_smoke.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("v19_lta_mask_seed_smoke", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Tracker:
    def __init__(
        self,
        *,
        swap_seed_masks: bool = False,
        swap_propagation_masks: bool = False,
        empty_propagation: bool = False,
        anchor_only: bool = False,
    ):
        self.swap_seed_masks = swap_seed_masks
        self.swap_propagation_masks = swap_propagation_masks
        self.empty_propagation = empty_propagation
        self.anchor_only = anchor_only
        self.states = []

    def add_new_masks(self, state, *, frame_idx, obj_ids, masks, reconditioning):
        import torch

        stored = masks.to(torch.bool).clone()
        state["authoritative_masks"] = stored
        state["anchor_frame"] = int(frame_idx)
        self.states.append(state)
        returned = stored.flip(0) if self.swap_seed_masks else stored
        return frame_idx, obj_ids, None, returned[:, None]

    def propagate_in_video_preflight(self, _state, *, run_mem_encoder):
        self.run_mem_encoder = run_mem_encoder

    def propagate_in_video(
        self,
        state,
        *,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
        tqdm_disable,
        run_mem_encoder,
    ):
        import torch

        masks = state["authoritative_masks"][:, None]
        if self.swap_propagation_masks:
            masks = masks.flip(0)
        if self.empty_propagation:
            masks = torch.zeros_like(masks)
        if self.anchor_only and int(start_frame_idx) != int(state["anchor_frame"]):
            masks = torch.zeros_like(masks)
        count = int(masks.shape[0])
        yield start_frame_idx, list(range(count)), None, masks, torch.ones(count)


class _Model:
    def __init__(self, tracker):
        self.tracker = tracker

    def _init_new_sam2_state(self, _inference_state):
        return {}

    def _prepare_backbone_feats(self, inference_state, frame_index, *, reverse):
        inference_state["feature_cache"][frame_index] = ("feature", reverse)


class _Predictor:
    def __init__(self, tracker):
        self.model = _Model(tracker)
        self._all_inference_states = {
            "mask-test": {"state": {"feature_cache": {}}},
        }


class _Measured:
    def __init__(self):
        self.closed = False

    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "mask-test"}
        if request["type"] == "close_session":
            self.closed = True
            return {"success": True}
        raise AssertionError(request)


class V19LtaMaskSeedSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    def test_capacity_rounds_without_silent_32_object_cap(self) -> None:
        resolve = self.tool.resolve_mask_seed_capacity
        self.assertEqual(resolve(1), 16)
        self.assertEqual(resolve(16), 16)
        self.assertEqual(resolve(17), 32)
        self.assertEqual(resolve(32), 32)
        self.assertEqual(resolve(33), 48)
        self.assertEqual(resolve(128), 128)
        with self.assertRaisesRegex(ValueError, r"\[1,128\]"):
            resolve(0)
        with self.assertRaisesRegex(ValueError, r"\[1,128\]"):
            resolve(129)
        with self.assertRaises(TypeError):
            resolve(True)

    def test_multi_mask_round_trip_is_validated_per_object(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("mask metrics require real OpenCV, not import stubs")

        first = np.zeros((8, 8), dtype=bool)
        first[1:4, 1:4] = True
        second = np.zeros((8, 8), dtype=bool)
        second[4:7, 4:7] = True
        measured = _Measured()
        tracker = _Tracker(swap_seed_masks=True)
        with self.assertRaisesRegex(RuntimeError, "changed object mask"):
            self.tool.run_mask_seed_session(
                measured,
                _Predictor(tracker),
                resource=[object(), object()],
                session=self.tool.SamSessionPlan("multi", 0, 0, 2),
                prompt_frame=0,
                ground_truth=first | second,
                seed=None,
                object_masks=(first, second),
                conf=0.15,
                propagation_mode="tracker-only",
            )
        self.assertTrue(measured.closed)
        self.assertEqual(tracker.states[0], {})

    def test_tracker_only_retains_object_identity_and_passes_anchor(self) -> None:
        import cv2
        import json
        import tempfile
        import numpy as np
        from PIL import Image

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("mask metrics require real OpenCV, not import stubs")

        first = np.zeros((8, 8), dtype=bool)
        first[1:4, 1:4] = True
        second = np.zeros((8, 8), dtype=bool)
        second[4:7, 4:7] = True
        measured = _Measured()
        tracker = _Tracker()
        result = self.tool.run_mask_seed_session(
            measured,
            _Predictor(tracker),
            resource=[object(), object()],
            session=self.tool.SamSessionPlan("multi", 0, 0, 2),
            prompt_frame=0,
            ground_truth=first | second,
            seed=None,
            object_masks=(first, second),
            conf=0.15,
            propagation_mode="tracker-only",
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["anchor_integrity_passed"])
        self.assertTrue(result["diagnostic_propagation_gate_passed"])
        self.assertEqual(result["non_anchor_active_frames"], [1])
        self.assertFalse(result["drop_stats_applicable"])
        self.assertEqual(result["anchor_returned_object_ids"], (0, 1))
        self.assertEqual(len(result["propagation"]), 4)
        self.assertEqual(
            {(item.frame_index, item.object_id) for item in result["propagation"]},
            {(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        self.assertEqual(result["propagation_active_frames"], [0, 1])
        self.assertEqual(result["anchor_preview_propagation_iou"], 1.0)
        self.assertTrue(measured.closed)
        self.assertEqual(tracker.states[0], {})
        with tempfile.TemporaryDirectory() as folder:
            artifacts = self.tool.write_strategy_artifacts(
                Path(folder),
                result=result,
                resource=[Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
                ground_truth=first | second,
                session=self.tool.SamSessionPlan("multi", 0, 0, 2),
                prompt_frame=0,
            )
            summary = json.loads(Path(artifacts["summary"]).read_text(encoding="utf-8"))
        self.assertEqual(
            summary["propagation_frames"][0]["returned_object_ids"], [0, 1]
        )
        self.assertEqual(
            summary["propagation_frames"][0]["active_object_ids"], [0, 1]
        )
        self.assertEqual(
            [item["mask_pixels"] for item in summary["propagation_frames"][0]["objects"]],
            [9, 9],
        )
        self.assertTrue(
            all(
                item["mask_path"]
                for item in summary["propagation_frames"][0]["objects"]
            )
        )
        self.assertEqual(summary["propagation_frames"][0]["mask_pixels"], 18)
        self.assertTrue(summary["anchor_integrity_passed"])
        self.assertTrue(summary["diagnostic_propagation_gate_passed"])
        self.assertEqual(summary["anchor_expected_object_ids"], [0, 1])
        self.assertEqual(len(summary["anchor_propagation_object_metrics"]), 2)

    def test_empty_anchor_propagation_fails_acceptance(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("mask metrics require real OpenCV, not import stubs")

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        result = self.tool.run_mask_seed_session(
            _Measured(),
            _Predictor(_Tracker(empty_propagation=True)),
            resource=[object()],
            session=self.tool.SamSessionPlan("empty", 0, 0, 1),
            prompt_frame=0,
            ground_truth=mask,
            seed=None,
            object_masks=(mask,),
            conf=0.15,
            propagation_mode="tracker-only",
        )
        self.assertFalse(result["success"])
        self.assertFalse(result["anchor_integrity_passed"])
        self.assertFalse(result["diagnostic_propagation_gate_passed"])
        self.assertEqual(result["anchor_preview_propagation_iou"], 0.0)
        self.assertEqual(result["propagation_active_frames"], [])

    def test_swapped_anchor_objects_fail_even_when_union_is_exact(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("mask metrics require real OpenCV, not import stubs")

        first = np.zeros((8, 8), dtype=bool)
        first[1:4, 1:4] = True
        second = np.zeros((8, 8), dtype=bool)
        second[4:7, 4:7] = True
        result = self.tool.run_mask_seed_session(
            _Measured(),
            _Predictor(_Tracker(swap_propagation_masks=True)),
            resource=[object()],
            session=self.tool.SamSessionPlan("swapped", 0, 0, 1),
            prompt_frame=0,
            ground_truth=first | second,
            seed=None,
            object_masks=(first, second),
            conf=0.15,
            propagation_mode="tracker-only",
        )
        self.assertEqual(result["anchor_preview_propagation_iou"], 1.0)
        self.assertFalse(result["success"])
        self.assertFalse(result["anchor_integrity_passed"])
        self.assertFalse(result["diagnostic_propagation_gate_passed"])
        self.assertEqual(
            [metrics["iou"] for metrics in result["anchor_propagation_object_metrics"]],
            [0.0, 0.0],
        )

    def test_anchor_only_activity_preserves_integrity_but_fails_diagnostic_gate(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("mask metrics require real OpenCV, not import stubs")

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        result = self.tool.run_mask_seed_session(
            _Measured(),
            _Predictor(_Tracker(anchor_only=True)),
            resource=[object(), object()],
            session=self.tool.SamSessionPlan("anchor-only", 0, 0, 2),
            prompt_frame=0,
            ground_truth=mask,
            seed=None,
            object_masks=(mask,),
            conf=0.15,
            propagation_mode="tracker-only",
        )
        self.assertTrue(result["anchor_integrity_passed"])
        self.assertFalse(result["diagnostic_propagation_gate_passed"])
        self.assertFalse(result["success"])
        self.assertEqual(result["propagation_active_frames"], [0])
        self.assertEqual(result["non_anchor_active_frames"], [])

    def test_cli_defaults_to_tracker_only(self) -> None:
        self.assertEqual(
            self.tool._build_parser().get_default("propagation_mode"),
            "tracker-only",
        )

    def test_merged_propagation_allows_absent_drop_stats(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("mask metrics require real OpenCV, not import stubs")

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        tracker = _Tracker()
        tracker_state = {}

        class Model(_Model):
            def _get_sam2_inference_states_by_obj_ids(self, _state, object_ids):
                self.object_ids = object_ids
                return [tracker_state]

            def _cache_frame_outputs(self, _state, _frame_index, _outputs):
                return None

        class Predictor:
            def __init__(self):
                self.model = Model(tracker)
                self._all_inference_states = {
                    "mask-test": {"state": {"feature_cache": {}}},
                }

        class Measured(_Measured):
            def handle_request(self, request):
                if request["type"] == "add_prompt":
                    return {"outputs": {}}
                return super().handle_request(request)

            def handle_stream_request(self, _request):
                yield {
                    "frame_index": 0,
                    "outputs": {
                        "out_obj_ids": [0],
                        "out_probs": [1.0],
                        "out_binary_masks": mask[None],
                    },
                }

        measured = Measured()
        result = self.tool.run_mask_seed_session(
            measured,
            Predictor(),
            resource=[object()],
            session=self.tool.SamSessionPlan("merged", 0, 0, 1),
            prompt_frame=0,
            ground_truth=mask,
            seed=self.tool.PointClick(0.5, 0.5, True, "seed"),
            object_masks=None,
            conf=0.15,
            propagation_mode="merged",
        )
        self.assertTrue(result["anchor_integrity_passed"])
        self.assertFalse(result["diagnostic_propagation_gate_passed"])
        self.assertFalse(result["drop_stats_applicable"])
        self.assertTrue(measured.closed)
        self.assertEqual(tracker_state, {})

    def test_merged_propagation_rejects_nonzero_drop_stats(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("mask metrics require real OpenCV, not import stubs")

        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        tracker = _Tracker()
        tracker_state = {}

        class Model(_Model):
            def _get_sam2_inference_states_by_obj_ids(self, _state, _object_ids):
                return [tracker_state]

            def _cache_frame_outputs(self, _state, _frame_index, _outputs):
                return None

        class Predictor:
            def __init__(self):
                self.model = Model(tracker)
                self._all_inference_states = {
                    "mask-test": {"state": {"feature_cache": {}}},
                }

        class Measured(_Measured):
            def handle_request(self, request):
                if request["type"] == "add_prompt":
                    return {"outputs": {}}
                return super().handle_request(request)

            def handle_stream_request(self, _request):
                yield {
                    "frame_index": 0,
                    "outputs": {
                        "out_obj_ids": [0],
                        "out_probs": [1.0],
                        "out_binary_masks": mask[None],
                        "frame_stats": {"num_obj_dropped": 1},
                    },
                }

        measured = Measured()
        with self.assertRaisesRegex(RuntimeError, "dropped 1 object"):
            self.tool.run_mask_seed_session(
                measured,
                Predictor(),
                resource=[object()],
                session=self.tool.SamSessionPlan("merged", 0, 0, 1),
                prompt_frame=0,
                ground_truth=mask,
                seed=self.tool.PointClick(0.5, 0.5, True, "seed"),
                object_masks=None,
                conf=0.15,
                propagation_mode="merged",
            )
        self.assertTrue(measured.closed)
        self.assertEqual(tracker_state, {})


if __name__ == "__main__":
    unittest.main()
