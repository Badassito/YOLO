from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "v19_lta_tile_smoke.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("v19_lta_tile_smoke", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V19LtaTileSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    @staticmethod
    def polygon(points):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return type(
            "Polygon",
            (),
            {"points": tuple(points), "box_xyxy": (min(xs), min(ys), max(xs), max(ys)), "row_index": 0},
        )()

    def test_object_tile_is_fixed_native_size_and_contains_polygon(self) -> None:
        polygon = self.polygon(((0.6, 0.3), (0.9, 0.3), (0.9, 0.5), (0.6, 0.5)))
        plan = self.tool.plan_object_tile(
            polygon,
            source_width=3024,
            source_height=3064,
        )
        self.assertEqual(plan.xyxy[2] - plan.xyxy[0], 1008)
        self.assertEqual(plan.xyxy[3] - plan.xyxy[1], 1008)
        self.assertGreaterEqual(0.6 * 3024, plan.left)
        self.assertLessEqual(0.9 * 3024, plan.left + plan.size)
        self.assertGreaterEqual(0.3 * 3064, plan.top)
        self.assertLessEqual(0.5 * 3064, plan.top + plan.size)

    def test_boundary_tile_clamps_and_coordinate_transform_round_trips(self) -> None:
        polygon = self.polygon(((0.01, 0.01), (0.1, 0.01), (0.1, 0.1), (0.01, 0.1)))
        plan = self.tool.plan_object_tile(
            polygon,
            source_width=3024,
            source_height=3064,
        )
        self.assertEqual((plan.left, plan.top), (0, 0))
        local = self.tool.transform_polygon_to_tile(polygon, plan)
        for source, transformed in zip(polygon.points, local.points):
            recovered_x = (transformed[0] * plan.size + plan.left) / plan.source_width
            recovered_y = (transformed[1] * plan.size + plan.top) / plan.source_height
            self.assertAlmostEqual(recovered_x, source[0])
            self.assertAlmostEqual(recovered_y, source[1])

    def test_oversized_polygon_is_rejected(self) -> None:
        polygon = self.polygon(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            self.tool.plan_object_tile(
                polygon,
                source_width=3024,
                source_height=3064,
            )

    def test_tile_requires_positive_adequate_source_dimensions(self) -> None:
        polygon = self.polygon(((0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)))
        for source_width, source_height, size in (
            (0, 3064, 1008),
            (3024, -1, 1008),
            (3024, 3064, 0),
            (1007, 3064, 1008),
            (3024, 1007, 1008),
        ):
            with self.subTest(
                source_width=source_width,
                source_height=source_height,
                size=size,
            ):
                with self.assertRaisesRegex(ValueError, "positive|cannot contain"):
                    self.tool.plan_object_tile(
                        polygon,
                        source_width=source_width,
                        source_height=source_height,
                        size=size,
                    )

    def test_blank_yolo_rows_select_by_source_row_not_compact_offset(self) -> None:
        first = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"
        selected = "0 0.6 0.3 0.7 0.3 0.7 0.4 0.6 0.4"
        with tempfile.TemporaryDirectory() as temp_dir:
            label = Path(temp_dir) / "labels.txt"
            label.write_text(f"{first}\n\n{selected}\n", encoding="utf-8")
            _digest, polygons = self.tool.parse_yolo_segmentation_label(label)

        self.assertEqual([polygon.row_index for polygon in polygons], [0, 2])
        polygon = self.tool.select_polygon_row(polygons, 2)
        self.assertIs(polygon, polygons[1])
        plan = self.tool.plan_object_tile(
            polygon,
            source_width=3024,
            source_height=3064,
        )
        local = self.tool.transform_polygon_to_tile(polygon, plan)
        self.assertGreater(local.box_xyxy[0], 0.0)
        self.assertLess(local.box_xyxy[2], 1.0)

    def test_native_decode_rejects_inexact_seek_and_releases_capture(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.released = False

            def isOpened(self) -> bool:
                return True

            def set(self, _property: int, _value: int) -> bool:
                return True

            def get(self, _property: int) -> float:
                return 359.0

            def release(self) -> None:
                self.released = True

        capture = FakeCapture()
        fake_cv2 = type(
            "FakeCv2",
            (),
            {
                "CAP_PROP_POS_FRAMES": 1,
                "VideoCapture": staticmethod(lambda _path: capture),
            },
        )
        plan = self.tool.TilePlan(0, 0, 1008, 3024, 3064)
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            with self.assertRaisesRegex(RuntimeError, "reported frame 359"):
                self.tool.decode_rgb_tile_frames(Path("missing.mkv"), 360, plan)
        self.assertTrue(capture.released)

    def test_box_candidate_tie_break_is_iou_then_score_then_lowest_id(self) -> None:
        import numpy as np

        ground_truth = np.zeros((4, 4), dtype=bool)
        ground_truth[:2, :2] = True
        exact = ground_truth.copy()
        partial = np.zeros_like(ground_truth)
        partial[0, 0] = True
        predictions = (
            SimpleNamespace(
                frame_index=3,
                object_id=0,
                initial_detection_score=0.99,
                binary_mask=partial,
            ),
            SimpleNamespace(
                frame_index=3,
                object_id=8,
                initial_detection_score=0.70,
                binary_mask=exact,
            ),
            SimpleNamespace(
                frame_index=3,
                object_id=2,
                initial_detection_score=0.80,
                binary_mask=exact,
            ),
            SimpleNamespace(
                frame_index=3,
                object_id=1,
                initial_detection_score=0.80,
                binary_mask=exact,
            ),
        )
        session = self.tool.SamSessionPlan(
            sequence_id="tile",
            session_index=0,
            frame_start=0,
            frame_stop=4,
        )
        polygon = self.polygon(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))

        def metrics(_ground_truth, prediction):
            exact_match = bool(np.array_equal(np.asarray(prediction), exact))
            score = 1.0 if exact_match else 0.25
            return {
                "iou": score,
                "dice": score,
                "precision": score,
                "recall": score,
            }

        with (
            mock.patch.object(self.tool, "run_video_session", return_value=predictions),
            mock.patch.object(self.tool, "mask_metrics", side_effect=metrics),
        ):
            result = self.tool.run_box_tile(
                object(),
                resource=[object()] * 4,
                session=session,
                prompt_frame=3,
                polygon=polygon,
                ground_truth=ground_truth,
                conf=0.15,
            )

        self.assertEqual(result["selected_object_id"], 1)
        self.assertTrue(result["acceptance_passed"])

    @staticmethod
    def point_result(
        name: str,
        *,
        final_iou: float,
        best_iou: float,
        passed: bool = True,
        anchor_iou: float | None = 1.0,
        clicks: int = 4,
    ):
        return {
            "strategy": name,
            "success": passed,
            "final_clicks": tuple(range(clicks)),
            "final_metrics": {"iou": final_iou},
            "best_metrics": {"iou": best_iou},
            "propagation_active_frames": tuple(range(30)),
            "anchor_preview_propagation_iou": anchor_iou,
        }

    @staticmethod
    def box_result(*, iou: float, passed: bool = True):
        return {
            "acceptance_passed": passed,
            "anchor_metrics": {"iou": iou},
            "active_frames": tuple(range(30)),
        }

    def test_comparison_uses_final_propagated_state_and_records_click_budgets(self) -> None:
        comparison = self.tool.compare_tile_results(
            self.box_result(iou=0.91),
            (
                self.point_result(
                    "regressed",
                    final_iou=0.90,
                    best_iou=0.99,
                    clicks=8,
                ),
                self.point_result(
                    "stable",
                    final_iou=0.93,
                    best_iou=0.94,
                    clicks=3,
                ),
            ),
        )

        self.assertEqual(comparison["winner"], "stable")
        by_name = {arm["strategy"]: arm for arm in comparison["arms"]}
        self.assertEqual(by_name["regressed"]["final_prompt_iou"], 0.90)
        self.assertEqual(by_name["regressed"]["prompt_budget"]["point_clicks_used"], 8)
        self.assertEqual(by_name["stable"]["prompt_budget"]["max_point_clicks"], 8)
        self.assertEqual(by_name["box"]["prompt_budget"]["box_prompts"], 1)

    def test_comparison_is_inconclusive_for_unaccepted_or_immaterial_results(self) -> None:
        none_accepted = self.tool.compare_tile_results(
            self.box_result(iou=0.99, passed=False),
            (
                self.point_result(
                    "point",
                    final_iou=0.99,
                    best_iou=0.99,
                    passed=False,
                ),
            ),
        )
        self.assertEqual(none_accepted["winner"], "inconclusive")
        self.assertEqual(none_accepted["accepted_strategy_count"], 0)

        equalish = self.tool.compare_tile_results(
            self.box_result(iou=0.910),
            (
                self.point_result(
                    "point",
                    final_iou=0.919,
                    best_iou=0.95,
                ),
            ),
        )
        self.assertEqual(equalish["winner"], "inconclusive")
        self.assertAlmostEqual(equalish["material_iou_delta"], 0.009)

    def test_comparison_rejects_point_arm_without_propagated_anchor_acceptance(self) -> None:
        comparison = self.tool.compare_tile_results(
            self.box_result(iou=0.91, passed=False),
            (
                self.point_result(
                    "missing_anchor",
                    final_iou=0.99,
                    best_iou=0.99,
                    anchor_iou=None,
                ),
            ),
        )
        self.assertEqual(comparison["winner"], "inconclusive")
        self.assertFalse(comparison["arms"][1]["acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
