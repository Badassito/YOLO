from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "lta_tracklet_pair.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("lta_tracklet_pair", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LtaTrackletPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    @staticmethod
    def _mask(
        top: int = 1,
        left: int = 1,
        bottom: int = 4,
        right: int = 4,
        *,
        size: int = 6,
    ) -> np.ndarray:
        mask = np.zeros((size, size), dtype=bool)
        mask[top:bottom, left:right] = True
        return mask

    @staticmethod
    def _prediction(
        frame_index: int,
        object_id: int,
        probability: float | None,
        mask: np.ndarray,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            frame_index=frame_index,
            object_id=object_id,
            frame_tracker_score=probability,
            binary_mask=mask,
        )

    def _tracklet(
        self,
        *,
        anchor: int,
        object_id: int,
        frames: list[int],
        mask: np.ndarray,
        probability: float = 0.9,
        tile_index: int = 0,
    ):
        return self.tool.Tracklet(
            tile_index=tile_index,
            anchor_frame=anchor,
            local_object_id=object_id,
            frames=tuple(
                self.tool.TrackletFrame(
                    frame_index=frame_index,
                    mask=mask,
                    tracker_probability=probability,
                    authoritative=frame_index == anchor,
                )
                for frame_index in frames
            ),
        )

    def test_observation_band_adds_inclusive_halo_and_clamps_to_source(self) -> None:
        self.assertEqual(
            self.tool.plan_observation_band.__module__,
            "XTA.lta_tracklets",
        )
        self.assertEqual(
            self.tool.plan_observation_band(100, 200, frame_count=300, halo_frames=48),
            (52, 249),
        )
        self.assertEqual(
            self.tool.plan_observation_band(10, 290, frame_count=300, halo_frames=48),
            (0, 300),
        )
        self.assertEqual(
            self.tool.plan_observation_band(100, 200, frame_count=300, halo_frames=0),
            (100, 201),
        )
        with self.assertRaisesRegex(ValueError, "ordered"):
            self.tool.plan_observation_band(200, 100, frame_count=300)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.tool.plan_observation_band(100, 200, frame_count=300, halo_frames=-1)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.tool.plan_observation_band(0, 1, frame_count=0)

    def test_pair_resolution_requires_adjacent_tile_visible_anchors(self) -> None:
        anchors = tuple(SimpleNamespace(frame_index=value) for value in (10, 20, 30, 40))
        tile = object()
        visible_masks = {
            10: (self._mask(),),
            20: (self._mask(),),
            30: (self._mask(),),
            40: (),
        }

        with mock.patch.object(
            self.tool,
            "_seed_masks_for_layout",
            side_effect=lambda anchor, _tile, layout: visible_masks[anchor.frame_index]
            if layout == "instances"
            else (),
        ):
            left, left_masks, right, right_masks, visible = self.tool._resolve_pair(
                anchors,
                tile=tile,
                left_anchor_frame=20,
                right_anchor_frame=30,
            )
            self.assertEqual((left.frame_index, right.frame_index), (20, 30))
            self.assertEqual(len(left_masks), 1)
            self.assertEqual(len(right_masks), 1)
            self.assertEqual(visible, (10, 20, 30))

            with self.assertRaisesRegex(ValueError, "not adjacent"):
                self.tool._resolve_pair(
                    anchors,
                    tile=tile,
                    left_anchor_frame=10,
                    right_anchor_frame=30,
                )
            with self.assertRaisesRegex(ValueError, "no positive raster support"):
                self.tool._resolve_pair(
                    anchors,
                    tile=tile,
                    left_anchor_frame=30,
                    right_anchor_frame=40,
                )
            with self.assertRaisesRegex(ValueError, "must precede"):
                self.tool._resolve_pair(
                    anchors,
                    tile=tile,
                    left_anchor_frame=30,
                    right_anchor_frame=20,
                )

    def test_canonical_tracker_probabilities_pass_through_exactly_once(self) -> None:
        authoritative = self._mask()
        propagated = self._mask(2, 2, 5, 5)
        predictions = (
            self._prediction(10, 0, 0.25, propagated),
            self._prediction(11, 0, 0.50, propagated),
            self._prediction(12, 0, 0.75, propagated),
        )

        tracklets, receipt = self.tool.predictions_to_tracklets(
            predictions,
            tile_index=3,
            anchor_frame=11,
            authoritative_masks=(authoritative,),
            frame_start=10,
            frame_stop=13,
        )

        probabilities = [item.tracker_probability for item in tracklets[0].frames]
        self.assertEqual(probabilities, [0.25, 0.50, 0.75])
        self.assertNotAlmostEqual(probabilities[0], 1.0 / (1.0 + math.exp(-0.25)))
        self.assertFalse(hasattr(tracklets[0].frames[0], "tracker_score_logit"))
        self.assertEqual(receipt["source_representation"], "sigmoid_probability")
        self.assertEqual(receipt["tracklet_representation"], "sigmoid_probability")
        self.assertEqual(receipt["transform"], "identity")
        self.assertEqual(receipt["sigmoid_application_count"], 1)
        self.assertFalse(receipt["raw_object_score_logits_persisted"])
        self.assertEqual(receipt["probability_min"], 0.25)
        self.assertEqual(receipt["probability_max"], 0.75)

    def test_conversion_injects_exact_authoritative_masks_per_object(self) -> None:
        authoritative_a = self._mask(1, 1, 3, 3)
        authoritative_b = self._mask(3, 3, 5, 5)
        wrong = np.ones((6, 6), dtype=bool)
        predictions = (
            self._prediction(7, 0, 0.05, wrong),
            self._prediction(7, 1, 0.95, wrong),
            self._prediction(8, 0, 0.80, wrong),
            self._prediction(8, 1, 0.70, wrong),
        )

        tracklets, _receipt = self.tool.predictions_to_tracklets(
            predictions,
            tile_index=2,
            anchor_frame=7,
            authoritative_masks=(authoritative_a, authoritative_b),
            frame_start=7,
            frame_stop=9,
        )

        anchor_a = tracklets[0].frame_map()[7]
        anchor_b = tracklets[1].frame_map()[7]
        self.assertTrue(anchor_a.authoritative)
        self.assertTrue(anchor_b.authoritative)
        self.assertTrue(anchor_a.active, "authoritative geometry must survive a low score")
        self.assertTrue(np.array_equal(anchor_a.mask, authoritative_a))
        self.assertTrue(np.array_equal(anchor_b.mask, authoritative_b))
        self.assertTrue(np.array_equal(tracklets[0].frame_map()[8].mask, wrong))

    def test_conversion_rejects_missing_invalid_duplicate_and_out_of_range_scores(self) -> None:
        mask = self._mask()
        kwargs = {
            "tile_index": 0,
            "anchor_frame": 5,
            "authoritative_masks": (mask,),
            "frame_start": 4,
            "frame_stop": 7,
        }
        duplicate = self._prediction(5, 0, 0.8, mask)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.tool.predictions_to_tracklets((duplicate, duplicate), **kwargs)
        with self.assertRaisesRegex(RuntimeError, "no persisted"):
            self.tool.predictions_to_tracklets(
                (self._prediction(5, 0, None, mask),),
                **kwargs,
            )
        for invalid in (-0.01, 1.01, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                RuntimeError, "probability is invalid"
            ):
                self.tool.predictions_to_tracklets(
                    (self._prediction(5, 0, invalid, mask),),
                    **kwargs,
                )
        with self.assertRaisesRegex(RuntimeError, "outside"):
            self.tool.predictions_to_tracklets(
                (self._prediction(8, 0, 0.8, mask),),
                **kwargs,
            )
        with self.assertRaisesRegex(RuntimeError, "omitted its authoritative frame"):
            self.tool.predictions_to_tracklets(
                (self._prediction(4, 0, 0.8, mask),),
                **kwargs,
            )

    def test_sparse_tracklet_reduction_keeps_policy_skipped_frames_zero(self) -> None:
        mask = self._mask()
        tracklet = self._tracklet(
            anchor=1,
            object_id=0,
            frames=[1, 3],
            mask=mask,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "work.uint8.raw"
            volume = self.tool._open_work_volume(path, (5, 6, 6))
            try:
                self.assertEqual(int(np.count_nonzero(volume)), 0)
                self.tool._or_tracklets_into_volume(
                    volume,
                    (tracklet,),
                    frame_start=0,
                    frame_stop=5,
                )
                self.assertTrue(np.array_equal(volume[1].astype(bool), mask))
                self.assertTrue(np.array_equal(volume[3].astype(bool), mask))
                self.assertEqual(int(np.count_nonzero(volume[0])), 0)
                self.assertEqual(int(np.count_nonzero(volume[2])), 0)
                self.assertEqual(int(np.count_nonzero(volume[4])), 0)
            finally:
                self.tool._close_work_volume(volume)

    def test_hard_positive_restoration_preserves_other_object_foreground(self) -> None:
        first = self._mask(1, 1, 3, 3)
        second = self._mask(3, 3, 5, 5)
        for name, unmatched in (
            ("disjoint", self._mask(0, 4, 2, 6)),
            ("overlapping", self._mask(2, 2, 6, 6)),
        ):
            with self.subTest(name=name):
                volume = np.zeros((4, 6, 6), dtype=np.uint8)
                volume[1] = first
                volume[2] = unmatched

                self.tool._restore_anchor_positives(
                    volume,
                    anchor_frame=12,
                    frame_start=10,
                    masks=(first, second),
                )

                np.testing.assert_array_equal(volume[2], first | second | unmatched)
                np.testing.assert_array_equal(volume[1], first)
                self.assertFalse(volume[0].any())
                self.assertFalse(volume[3].any())

    def test_reconciled_partial_anchor_keeps_overlapping_unmatched_object(self) -> None:
        frames = [10, 11, 12, 13, 14]
        known = self._mask(1, 1, 3, 3)
        unmatched = self._mask(2, 2, 5, 5)
        left = (
            self._tracklet(anchor=10, object_id=0, frames=frames, mask=known),
            self._tracklet(anchor=10, object_id=1, frames=frames, mask=unmatched),
        )
        # The later annotation labels only the first object.  The second object
        # overlaps it but still has independent support outside the label.
        right = (
            self._tracklet(anchor=14, object_id=0, frames=frames, mask=known),
        )
        matching = self.tool.match_tracklets(left, right)
        self.assertEqual(len(matching.matches), 1)
        self.assertEqual(matching.unmatched_left_keys, (left[1].key,))
        composition = self.tool.compose_binary(left, right, matching)
        matched = next(item for item in composition.lineages if item.status == "matched")
        own_anchor = next(item for item in matched.frames if item.frame_index == 14)
        self.assertTrue(own_anchor.authoritative)
        np.testing.assert_array_equal(own_anchor.mask, known)

        volume = np.zeros((len(frames), 6, 6), dtype=np.uint8)
        for frame in composition.union_frames:
            volume[frame.frame_index - 10] = frame.mask
        self.tool._restore_anchor_positives(
            volume, anchor_frame=10, frame_start=10, masks=(known, unmatched)
        )
        self.tool._restore_anchor_positives(
            volume, anchor_frame=14, frame_start=10, masks=(known,)
        )

        for frame in volume:
            np.testing.assert_array_equal(frame, known | unmatched)
        self.assertTrue(np.any(volume[-1].astype(bool) & ~known))

    def test_matching_composition_and_session_receipts_are_strict_json(self) -> None:
        mask = self._mask()
        left = self._tracklet(
            anchor=10,
            object_id=0,
            frames=[10, 11, 12, 13],
            mask=mask,
        )
        right = self._tracklet(
            anchor=20,
            object_id=4,
            frames=[11, 12, 13, 20],
            mask=mask,
        )
        matching = self.tool.match_tracklets(
            (left,),
            (right,),
            frame_start=10,
            frame_stop=21,
        )
        composition = self.tool.compose_binary((left,), (right,), matching)

        matching_receipt = self.tool._matching_receipt(matching)
        composition_receipt = self.tool._composition_receipt(composition)
        json.dumps(matching_receipt, allow_nan=False)
        json.dumps(composition_receipt, allow_nan=False)
        self.assertEqual(len(matching_receipt["matches"]), 1)
        self.assertEqual(matching_receipt["unmatched_left_keys"], [])
        self.assertEqual(matching_receipt["unmatched_right_keys"], [])
        self.assertEqual(
            matching_receipt["assignment_backend"],
            matching.assignment_backend,
        )
        self.assertEqual(composition_receipt["lineages"][0]["status"], "matched")
        self.assertEqual(
            composition_receipt["lineages"][0]["frame_ranges"],
            [[10, 14], [20, 21]],
        )
        self.assertIsNotNone(composition_receipt["lineages"][0]["handoff"])

        result = {
            "anchor_integrity_passed": True,
            "anchor_preview_propagation_iou": 1.0,
            "anchor_propagation_object_metrics": ({"iou": 1.0},),
            "seed_object_metrics": ({"iou": 1.0},),
            "seeded_object_count": 1,
            "propagation_response_count": 4,
            "model_visited_frame_ranges": [[10, 14]],
            "policy_zero_frame_ranges": [[14, 20]],
            "policy_zero_frame_count": 6,
            "empty_frame_limit": 30,
            "empty_frame_termination_receipts": [],
            "frame_tracker_score_semantics": "sigmoid(object_score_logit)",
        }
        session_receipt = self.tool._session_receipt(result, (left,))
        json.dumps(session_receipt, allow_nan=False)
        self.assertEqual(session_receipt["empty_frame_limit"], 30)
        self.assertEqual(session_receipt["policy_zero_frame_ranges"], [[14, 20]])
        self.assertEqual(session_receipt["tracklet_probability_min"], 0.9)
        self.assertEqual(session_receipt["tracklet_probability_max"], 0.9)

    def test_composition_receipt_exposes_sparse_and_authoritative_actual_sources(self) -> None:
        mask = self._mask()
        sparse_left = self._tracklet(
            anchor=10,
            object_id=0,
            frames=[10, 11, 12, 13, 30],
            mask=mask,
        )
        sparse_right = self._tracklet(
            anchor=20,
            object_id=1,
            frames=[5, 11, 12, 13, 20],
            mask=mask,
        )
        sparse_matching = self.tool.match_tracklets((sparse_left,), (sparse_right,))
        sparse_composition = self.tool.compose_binary(
            (sparse_left,),
            (sparse_right,),
            sparse_matching,
        )
        sparse_receipt = self.tool._composition_receipt(sparse_composition)
        sparse_groups = {
            (
                item["source_key"]["local_object_id"],
                item["selection_reason"],
            ): item
            for item in sparse_receipt["lineages"][0][
                "actual_source_selection_ranges"
            ]
        }
        self.assertEqual(sparse_groups[(1, "sparse_fallback")]["frame_ranges"], [[5, 6]])
        self.assertEqual(sparse_groups[(0, "sparse_fallback")]["frame_ranges"], [[30, 31]])

        shared_frames = [5, 6, 7, 10, 11, 12, 13, 20]
        authority_left = self._tracklet(
            anchor=10,
            object_id=2,
            frames=shared_frames,
            mask=mask,
            probability=0.51,
        )
        authority_right = self._tracklet(
            anchor=20,
            object_id=3,
            frames=shared_frames,
            mask=mask,
            probability=0.99,
        )
        authority_matching = self.tool.match_tracklets(
            (authority_left,),
            (authority_right,),
        )
        authority_composition = self.tool.compose_binary(
            (authority_left,),
            (authority_right,),
            authority_matching,
        )
        authority_receipt = self.tool._composition_receipt(authority_composition)
        authority_groups = authority_receipt["lineages"][0][
            "actual_source_selection_ranges"
        ]
        override = next(
            item
            for item in authority_groups
            if item["selection_reason"] == "authoritative_override"
        )
        self.assertEqual(override["source_key"]["local_object_id"], 2)
        self.assertEqual(override["frame_ranges"], [[10, 11]])

    def test_three_to_one_merge_extends_an_accepted_match_with_two_residuals(self) -> None:
        frames = list(range(30, 36))
        large = self._mask(1, 1, 5, 7, size=8)
        small_a = self._mask(5, 1, 6, 7, size=8)
        small_b = self._mask(6, 1, 7, 7, size=8)
        merged = large | small_a | small_b
        left = (
            self._tracklet(anchor=10, object_id=0, frames=frames, mask=large),
            self._tracklet(anchor=10, object_id=1, frames=frames, mask=small_a),
            self._tracklet(anchor=10, object_id=2, frames=frames, mask=small_b),
        )
        right = (self._tracklet(anchor=50, object_id=9, frames=frames, mask=merged),)

        matching = self.tool.match_tracklets(left, right)

        self.assertEqual(len(matching.matches), 1)
        self.assertEqual(matching.matches[0].left_key, left[0].key)
        self.assertEqual(set(matching.unmatched_left_keys), {left[1].key, left[2].key})
        merges = [
            item for item in matching.residual_hypotheses if item.kind == "merge"
        ]
        self.assertEqual(len(merges), 1)
        self.assertEqual(set(merges[0].left_keys), {item.key for item in left})
        self.assertEqual(merges[0].right_keys, (right[0].key,))
        self.assertEqual(merges[0].metrics.iou_3d, 1.0)

        composition = self.tool.compose_binary(left, right, matching)
        self.assertTrue(
            all(np.array_equal(item.mask, merged) for item in composition.union_frames)
        )

    def test_ambiguous_three_to_one_merge_preserves_union_without_a_forced_match(self) -> None:
        frames = list(range(30, 36))
        first = self._mask(1, 1, 7, 3, size=8)
        second = self._mask(1, 3, 7, 5, size=8)
        third = self._mask(1, 5, 7, 7, size=8)
        merged = first | second | third
        left = (
            self._tracklet(anchor=10, object_id=0, frames=frames, mask=first),
            self._tracklet(anchor=10, object_id=1, frames=frames, mask=second),
            self._tracklet(anchor=10, object_id=2, frames=frames, mask=third),
        )
        right = (self._tracklet(anchor=50, object_id=9, frames=frames, mask=merged),)
        config = self.tool.MatchingConfig(
            max_centroid_distance=1.0,
            weight_centroid=0.0,
        )

        matching = self.tool.match_tracklets(left, right, config=config)

        self.assertEqual(matching.matches, ())
        self.assertEqual(set(matching.unmatched_left_keys), {item.key for item in left})
        self.assertEqual(matching.unmatched_right_keys, (right[0].key,))
        merges = [
            item for item in matching.residual_hypotheses if item.kind == "merge"
        ]
        self.assertEqual(len(merges), 1)
        self.assertEqual(set(merges[0].left_keys), {item.key for item in left})
        self.assertEqual(merges[0].metrics.iou_3d, 1.0)

        composition = self.tool.compose_binary(left, right, matching)
        self.assertEqual(len(composition.lineages), 4)
        self.assertTrue(
            all(np.array_equal(item.mask, merged) for item in composition.union_frames)
        )

    def test_false_propagated_anchor_diagnostic_does_not_abort_session(self) -> None:
        authoritative = self._mask(1, 1, 4, 4)
        propagated = self._mask(2, 2, 5, 5)
        prediction = self._prediction(10, 0, 0.8, propagated)
        result = {
            "anchor_integrity_passed": False,
            "anchor_integrity_minimum_iou": 0.99,
            "anchor_preview_propagation_iou": 0.975,
            "anchor_propagation_object_metrics": ({"iou": 0.975},),
            "seed_object_metrics": ({"iou": 1.0},),
            "seeded_object_count": 1,
            "propagation_response_count": 1,
            "model_visited_frame_ranges": ((10, 11),),
            "policy_zero_frame_ranges": ((11, 12),),
            "policy_zero_frame_count": 1,
            "empty_frame_limit": 30,
            "empty_frame_termination_receipts": (),
            "frame_tracker_score_semantics": {
                "source_representation": "logit",
                "stored_representation": "sigmoid_probability",
            },
            "propagation": (prediction,),
        }
        anchor = SimpleNamespace(frame_index=10)

        with mock.patch.object(
            self.tool,
            "run_mask_seed_session",
            return_value=result,
        ) as run:
            tracklets, receipt, conversion = self.tool._run_anchor_session(
                object(),
                object(),
                resource=[object(), object()],
                tile_index=0,
                anchor=anchor,
                masks=(authoritative,),
                frame_start=10,
                frame_stop=12,
                conf=0.15,
                empty_frame_limit=30,
                session_index=0,
            )

        self.assertEqual(run.call_count, 1)
        self.assertFalse(receipt["propagated_anchor_diagnostic"]["passed"])
        self.assertFalse(receipt["propagated_anchor_diagnostic"]["execution_gate"])
        self.assertEqual(
            receipt["propagated_anchor_diagnostic"]["union_iou"],
            0.975,
        )
        self.assertTrue(receipt["seed_injection_invariant"]["passed"])
        self.assertTrue(
            receipt["seed_injection_invariant"][
                "binary_masks_preserved_at_required_iou"
            ]
        )
        self.assertEqual(
            receipt["seed_injection_invariant"][
                "minimum_per_object_and_union_iou"
            ],
            0.999999,
        )
        self.assertEqual(
            receipt["seed_injection_invariant"]["seed_object_metrics"],
            [{"iou": 1.0}],
        )
        self.assertTrue(receipt["seed_injection_invariant"]["execution_gate"])
        self.assertFalse(receipt["anchor_integrity_passed"])
        self.assertEqual(conversion["transform"], "identity")
        self.assertTrue(
            np.array_equal(tracklets[0].frame_map()[10].mask, authoritative)
        )

    def test_parser_requires_explicit_tile_and_anchor_pair_and_keeps_safe_defaults(self) -> None:
        parser = self.tool._build_parser()
        required = [
            "--model",
            "model",
            "--input-root",
            "input",
            "--exemplar-root",
            "exemplars",
            "--output",
            "output",
            "--tile-index",
            "5",
            "--left-anchor-frame",
            "100",
            "--right-anchor-frame",
            "200",
            "--plan-only",
        ]

        args = parser.parse_args(required)

        self.assertEqual(args.tile_index, 5)
        self.assertEqual(args.left_anchor_frame, 100)
        self.assertEqual(args.right_anchor_frame, 200)
        self.assertEqual(args.empty_frame_limit, 30)
        self.assertEqual(args.observation_halo_frames, 48)
        self.assertTrue(args.plan_only)
        without_tile = required[:8] + required[10:]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(without_tile)

    def test_plan_only_serializes_clamped_observation_and_exact_anchor_match_band(self) -> None:
        mask = self._mask(size=4)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            inputs = root / "input"
            exemplars = root / "exemplars"
            output = root / "output"
            for path in (model, inputs, exemplars):
                path.mkdir()
            video_path = inputs / "sample.mkv"
            video_path.write_bytes(b"placeholder")
            left = self.tool.PositiveAnchor(
                encoded_index=11,
                frame_index=10,
                image_path=exemplars / "left.png",
                label_path=exemplars / "left.txt",
                label_sha256="1" * 64,
                polygons=(object(),),
            )
            right = self.tool.PositiveAnchor(
                encoded_index=21,
                frame_index=20,
                image_path=exemplars / "right.png",
                label_path=exemplars / "right.txt",
                label_sha256="2" * 64,
                polygons=(object(),),
            )
            tile = SimpleNamespace(
                left=0,
                top=0,
                size=4,
                xyxy=(0, 0, 4, 4),
                source_width=4,
                source_height=4,
            )
            bundle = SimpleNamespace(
                root=model,
                checkpoint_path=model / "model.pt",
                model_version="sam3.1",
                checkpoint_identity_sha256="3" * 64,
            )
            argv = [
                str(TOOL_PATH),
                "--model",
                str(model),
                "--input-root",
                str(inputs),
                "--exemplar-root",
                str(exemplars),
                "--output",
                str(output),
                "--tile-index",
                "0",
                "--left-anchor-frame",
                "10",
                "--right-anchor-frame",
                "20",
                "--observation-halo-frames",
                "3",
                "--plan-only",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(self.tool, "resolve_local_sam_bundle", return_value=bundle),
                mock.patch.object(self.tool, "find_case_video", return_value=video_path),
                mock.patch.object(
                    self.tool,
                    "probe_video",
                    return_value={
                        "path": str(video_path),
                        "width": 4,
                        "height": 4,
                        "frame_count": 23,
                        "fps": 10.0,
                    },
                ),
                mock.patch.object(
                    self.tool,
                    "discover_positive_anchors",
                    return_value=(left, right),
                ),
                mock.patch.object(
                    self.tool,
                    "discover_known_background_frames",
                    return_value=(),
                ),
                mock.patch.object(self.tool, "plan_tile_grid", return_value=(tile,)),
                mock.patch.object(
                    self.tool,
                    "_resolve_pair",
                    return_value=(left, (mask,), right, (mask,), (10, 20)),
                ),
                mock.patch.object(self.tool, "resolve_mask_seed_capacity", return_value=16),
                mock.patch.object(self.tool, "_sha256_file", return_value="4" * 64),
                mock.patch.object(
                    self.tool,
                    "resolve_pinned_sam_runtime_provenance",
                    side_effect=AssertionError("plan-only touched the runtime"),
                ),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                self.tool.main()

            plan = json.loads((output / "run_plan.json").read_text(encoding="utf-8"))
            console = json.loads(stdout.getvalue())
            self.assertEqual(console["status"], "plan_complete")
            self.assertEqual(plan["observation_frame_range"], [7, 23])
            self.assertEqual(plan["matching_frame_range"], [10, 21])
            self.assertEqual(
                plan["matching_frame_range_policy"],
                "exactly [left_anchor,right_anchor+1)",
            )
            self.assertEqual(plan["empty_frame_limit"], 30)
            self.assertIn("hard-positive OR", plan["authoritative_anchor_policy"])
            self.assertIn("unmatched", plan["seed_integrity_policy"]["output_anchor"])
            self.assertIn(
                "every unmatched proposal",
                plan["diagnostic_product_policies"]["reconciled"],
            )
            self.assertIn(
                "not added to this reference",
                plan["diagnostic_product_policies"]["nearest_anchor_reference"],
            )
            self.assertIn(
                "not added to disagreement",
                plan["diagnostic_product_policies"]["opposing_disagreement"],
            )
            self.assertTrue(
                {
                    "tools/lta_tracklet_pair.py",
                    "XTA/lta_tracklets.py",
                    "XTA/lta_experimental.py",
                    "XTA/lta_tiles.py",
                    "tools/lta_full_volume.py",
                    "tools/lta_gpu_smoke.py",
                    "XTA/lta_sam.py",
                    "XTA/lta_inputs.py",
                }.issubset(set(plan["code_identity_sha256"])),
            )
            self.assertEqual(plan["tracker_probability_contract"]["transform"], "identity")
            self.assertEqual(
                plan["tracker_probability_contract"]["sigmoid_application_count"],
                1,
            )
            self.assertEqual(
                plan["binary_publication_policy"]["mask_source"],
                "tracker video_masks thresholded at > 0",
            )
            self.assertEqual(
                plan["binary_publication_policy"]["tracker_probability_role"],
                "matching and handoff only",
            )


if __name__ == "__main__":
    unittest.main()
