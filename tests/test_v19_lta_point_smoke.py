from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "v19_lta_point_smoke.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("v19_lta_point_smoke", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V19LtaPointSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    def test_click_contract_uses_normalized_xy_and_strict_polarity(self) -> None:
        click = self.tool.PointClick(0.5, 0.25, True, "positive")
        self.assertEqual(click.label, 1)
        self.assertEqual(click.pixel_xy(100), (50, 25))
        with self.assertRaisesRegex(ValueError, "normalized"):
            self.tool.PointClick(1.1, 0.25, True, "bad")
        with self.assertRaisesRegex(TypeError, "strict boolean"):
            self.tool.PointClick(0.5, 0.25, 1, "bad")

    def test_both_initial_strategies_keep_positive_and_negative_points_safe(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("point-strategy numerics require real OpenCV, not import stubs")

        polygon = type(
            "Polygon",
            (),
            {
                "points": ((0.25, 0.3), (0.75, 0.3), (0.75, 0.7), (0.25, 0.7)),
                "box_xyxy": (0.25, 0.3, 0.75, 0.7),
            },
        )()
        target = self.tool.rasterize_polygons((polygon,), size=192)
        fov = np.ones_like(target, dtype=bool)
        distance = self.tool.build_distance_strategy(
            polygon,
            target_mask=target,
            all_foreground=target,
            fov_mask=fov,
        )
        centerline = self.tool.build_centerline_strategy(
            polygon,
            target_mask=target,
            all_foreground=target,
            fov_mask=fov,
        )

        self.assertEqual((len(distance), len(centerline)), (6, 3))
        for click in distance + centerline:
            x, y = click.pixel_xy(192)
            self.assertEqual(bool(target[y, x]), click.positive)
        self.tool.validate_clicks(distance)
        self.tool.validate_clicks(centerline)

    def test_metrics_and_error_clicks_distinguish_false_negative_and_positive(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("point-strategy numerics require real OpenCV, not import stubs")

        ground_truth = np.zeros((64, 64), dtype=bool)
        ground_truth[10:30, 10:30] = True
        prediction = np.zeros_like(ground_truth)
        prediction[15:35, 15:35] = True
        metrics = self.tool.mask_metrics(ground_truth, prediction)
        self.assertEqual(metrics["tp"], 225)
        self.assertEqual(metrics["fp"], 175)
        self.assertEqual(metrics["fn"], 175)
        clicks = self.tool.next_error_clicks(
            ground_truth,
            prediction,
            (),
            capacity=2,
            all_foreground=ground_truth,
            fov_mask=np.ones_like(ground_truth),
        )
        self.assertEqual({click.positive for click in clicks}, {False, True})
        for click in clicks:
            x, y = click.pixel_xy(64)
            self.assertEqual(bool(ground_truth[y, x]), click.positive)

    def test_strategy_comparison_requires_material_iou_difference(self) -> None:
        first = {
            "strategy": "distance",
            "success": False,
            "final_clicks": tuple(range(8)),
            "final_metrics": {"iou": 0.80},
        }
        second = {
            "strategy": "centerline",
            "success": False,
            "final_clicks": tuple(range(8)),
            "final_metrics": {"iou": 0.795},
        }
        self.assertEqual(
            self.tool.compare_results((first, second))["winner"],
            "inconclusive",
        )

    def test_partial_propagation_records_drop_stats_as_not_applicable(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("propagation metrics require real OpenCV, not import stubs")

        mask = np.ones((16, 16), dtype=bool)

        class Predictor:
            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "point-test"}
                if request["type"] == "add_prompt":
                    return {
                        "frame_index": request["frame_index"],
                        "outputs": {
                            "out_obj_ids": [0],
                            "out_probs": [1.0],
                            "out_binary_masks": mask[None],
                        },
                    }
                if request["type"] == "close_session":
                    return {"success": True}
                raise AssertionError(request)

            def handle_stream_request(self, _request):
                for frame_index in range(30):
                    yield {
                        "frame_index": frame_index,
                        "outputs": {
                            "out_obj_ids": [0],
                            "out_probs": [1.0],
                            "out_binary_masks": mask[None],
                            "frame_stats": None,
                        },
                    }

        result = self.tool.run_point_strategy(
            Predictor(),
            strategy="partial-propagation",
            resource=[object()] * 30,
            session=self.tool.SamSessionPlan(
                sequence_id="partial-propagation",
                session_index=0,
                frame_start=0,
                frame_stop=30,
            ),
            prompt_frame=0,
            initial_clicks=(self.tool.PointClick(0.5, 0.5, True, "seed"),),
            ground_truth=mask,
            all_foreground=mask,
            fov_mask=mask,
            conf=0.15,
        )
        self.assertFalse(result["drop_stats_applicable"])
        self.assertEqual(result["propagated_revision"]["selection"], "final")

    def test_zero_drop_propagation_covers_the_session(self) -> None:
        import cv2
        import numpy as np

        if getattr(cv2, "__file__", None) is None:
            self.skipTest("propagation metrics require real OpenCV, not import stubs")

        mask = np.ones((16, 16), dtype=bool)

        class Predictor:
            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "point-test"}
                if request["type"] == "add_prompt":
                    return {
                        "frame_index": request["frame_index"],
                        "outputs": {
                            "out_obj_ids": [0],
                            "out_probs": [1.0],
                            "out_binary_masks": mask[None],
                        },
                    }
                if request["type"] == "close_session":
                    return {"success": True}
                raise AssertionError(request)

            def handle_stream_request(self, _request):
                for frame_index in range(30):
                    yield {
                        "frame_index": frame_index,
                        "outputs": {
                            "out_obj_ids": [0],
                            "out_probs": [1.0],
                            "out_binary_masks": mask[None],
                            "frame_stats": {"num_obj_dropped": 0},
                        },
                    }

        result = self.tool.run_point_strategy(
            Predictor(),
            strategy="drop-audit",
            resource=[object()] * 30,
            session=self.tool.SamSessionPlan(
                sequence_id="drop-audit",
                session_index=0,
                frame_start=0,
                frame_stop=30,
            ),
            prompt_frame=0,
            initial_clicks=(self.tool.PointClick(0.5, 0.5, True, "seed"),),
            ground_truth=mask,
            all_foreground=mask,
            fov_mask=mask,
            conf=0.15,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["propagation_response_count"], 30)
        self.assertEqual(len(result["propagation_active_frames"]), 30)
        self.assertFalse(result["drop_stats_applicable"])
        self.assertTrue(result["propagated_revision"]["equals_best"])

    def test_final_revision_provenance_can_differ_from_best_round(self) -> None:
        import numpy as np
        from unittest import mock

        best_mask = np.asarray([[True, True], [True, False]], dtype=bool)
        final_mask = np.asarray([[True, True], [False, False]], dtype=bool)

        class Predictor:
            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "point-test"}
                if request["type"] == "close_session":
                    return {"success": True}
                raise AssertionError(request)

            def handle_stream_request(self, _request):
                for frame_index in range(30):
                    yield {
                        "frame_index": frame_index,
                        "outputs": {
                            "out_obj_ids": [0],
                            "out_probs": [1.0],
                            "out_binary_masks": final_mask[None],
                            "frame_stats": None,
                        },
                    }

        previews = iter(
            (
                (best_mask, None, True),
                (final_mask, None, True),
            )
        )

        def metrics(ground_truth, prediction):
            if int(np.asarray(ground_truth).sum()) == int(final_mask.sum()):
                score = 1.0
            elif int(np.asarray(prediction).sum()) == int(best_mask.sum()):
                score = 0.80
            else:
                score = 0.70
            return {
                "iou": score,
                "dice": score,
                "precision": score,
                "recall": score,
            }

        correction = self.tool.PointClick(0.25, 0.25, False, "regression")
        with (
            mock.patch.object(
                self.tool,
                "_point_preview",
                side_effect=lambda *_args, **_kwargs: next(previews),
            ),
            mock.patch.object(self.tool, "mask_metrics", side_effect=metrics),
            mock.patch.object(self.tool, "next_error_clicks", return_value=(correction,)),
        ):
            result = self.tool.run_point_strategy(
                Predictor(),
                strategy="final-versus-best",
                resource=[object()] * 30,
                session=self.tool.SamSessionPlan(
                    sequence_id="final-versus-best",
                    session_index=0,
                    frame_start=0,
                    frame_stop=30,
                ),
                prompt_frame=0,
                initial_clicks=(self.tool.PointClick(0.5, 0.5, True, "seed"),),
                ground_truth=np.ones((2, 2), dtype=bool),
                all_foreground=np.ones((2, 2), dtype=bool),
                fov_mask=np.ones((2, 2), dtype=bool),
                conf=0.15,
            )

        self.assertEqual(result["best_round"], 0)
        self.assertEqual(result["propagated_revision"]["round"], 1)
        self.assertEqual(result["propagated_revision"]["selection"], "final")
        self.assertFalse(result["propagated_revision"]["equals_best"])

    def test_primary_inference_error_survives_stream_and_session_cleanup_failures(self) -> None:
        import numpy as np
        from unittest import mock

        mask = np.ones((2, 2), dtype=bool)

        class BrokenStream:
            def __init__(self) -> None:
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                raise MemoryError("primary propagation failure")

            def close(self):
                self.closed = True
                raise RuntimeError("stream cleanup failure")

        stream = BrokenStream()

        class Predictor:
            def __init__(self) -> None:
                self.session_close_attempted = False

            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "point-test"}
                if request["type"] == "close_session":
                    self.session_close_attempted = True
                    raise RuntimeError("session cleanup failure")
                raise AssertionError(request)

            def handle_stream_request(self, _request):
                return stream

        predictor = Predictor()
        perfect = {
            "iou": 1.0,
            "dice": 1.0,
            "precision": 1.0,
            "recall": 1.0,
        }
        with (
            mock.patch.object(self.tool, "_point_preview", return_value=(mask, None, True)),
            mock.patch.object(self.tool, "mask_metrics", return_value=perfect),
        ):
            with self.assertRaisesRegex(MemoryError, "primary propagation failure") as raised:
                self.tool.run_point_strategy(
                    predictor,
                    strategy="cleanup",
                    resource=[object()] * 30,
                    session=self.tool.SamSessionPlan(
                        sequence_id="cleanup",
                        session_index=0,
                        frame_start=0,
                        frame_stop=30,
                    ),
                    prompt_frame=0,
                    initial_clicks=(self.tool.PointClick(0.5, 0.5, True, "seed"),),
                    ground_truth=mask,
                    all_foreground=mask,
                    fov_mask=mask,
                    conf=0.15,
                )

        notes = tuple(getattr(raised.exception, "__notes__", ()))
        self.assertTrue(stream.closed)
        self.assertTrue(predictor.session_close_attempted)
        self.assertTrue(any("stream cleanup failure" in note for note in notes))
        self.assertTrue(any("session cleanup failure" in note for note in notes))

    def test_cleanup_runner_executes_all_steps_without_masking_primary(self) -> None:
        calls = []
        primary = MemoryError("primary")

        def first():
            calls.append("first")
            raise RuntimeError("secondary")

        def second():
            calls.append("second")

        self.tool._run_cleanup_steps(primary, (("first", first), ("second", second)))

        self.assertEqual(calls, ["first", "second"])
        self.assertTrue(
            any("secondary" in note for note in getattr(primary, "__notes__", ()))
        )


if __name__ == "__main__":
    unittest.main()
