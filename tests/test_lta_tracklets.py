from __future__ import annotations

import unittest
from unittest import mock

import numpy as np


def _load_tool():
    import XTA.lta_tracklets as module

    return module


class LtaTrackletTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    @staticmethod
    def _mask(*rectangles: tuple[int, int, int, int], size: int = 12) -> np.ndarray:
        mask = np.zeros((size, size), dtype=bool)
        for top, left, bottom, right in rectangles:
            mask[top:bottom, left:right] = True
        return mask

    def _tracklet(
        self,
        *,
        anchor: int,
        object_id: int,
        frames: list[int],
        masks: np.ndarray | dict[int, np.ndarray],
        probabilities: float | dict[int, float] = 0.95,
        authoritative_frames: set[int] | None = None,
        tile: int = 0,
    ):
        records = []
        authoritative_frames = (
            {int(anchor)}
            if authoritative_frames is None
            else {int(value) for value in authoritative_frames}
        )
        for frame_index in frames:
            mask = masks[frame_index] if isinstance(masks, dict) else masks
            probability = (
                probabilities[frame_index]
                if isinstance(probabilities, dict)
                else probabilities
            )
            records.append(
                self.tool.TrackletFrame(
                    frame_index=frame_index,
                    mask=mask,
                    tracker_probability=probability,
                    authoritative=frame_index in authoritative_frames,
                )
            )
        return self.tool.Tracklet(
            tile_index=tile,
            anchor_frame=anchor,
            local_object_id=object_id,
            frames=tuple(records),
        )

    def test_module_keeps_numerical_dependencies_lazy(self) -> None:
        self.assertNotIn("np", self.tool.__dict__)
        self.assertNotIn("numpy", self.tool.__dict__)
        self.assertNotIn("scipy", self.tool.__dict__)

    def test_numpy_hungarian_matches_scipy_and_has_deterministic_ties(self) -> None:
        weights = np.asarray(
            (
                (9.0, 2.0, -1.0e9, 1.0),
                (6.0, 4.0, 3.0, -1.0e9),
                (5.0, 8.0, 1.0, 2.0),
                (-1.0e9, 3.0, 7.0, 6.0),
            ),
            dtype=np.float64,
        )
        fallback_rows, fallback_columns = self.tool._hungarian_square_maximize(weights)
        self.assertEqual(fallback_rows.tolist(), [0, 1, 2, 3])
        self.assertEqual(fallback_columns.tolist(), [0, 2, 1, 3])
        self.assertEqual(float(weights[fallback_rows, fallback_columns].sum()), 26.0)
        self.assertTrue(
            np.all(weights[fallback_rows, fallback_columns] > -1.0e9)
        )

        tied = np.zeros((4, 4), dtype=np.float64)
        first_tie = self.tool._hungarian_square_maximize(tied)[1].tolist()
        second_tie = self.tool._hungarian_square_maximize(tied)[1].tolist()
        self.assertEqual(first_tie, [0, 1, 2, 3])
        self.assertEqual(second_tie, first_tie)

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return
        scipy_rows, scipy_columns = linear_sum_assignment(-weights)
        self.assertEqual(
            list(zip(fallback_rows.tolist(), fallback_columns.tolist())),
            list(zip(scipy_rows.tolist(), scipy_columns.tolist())),
        )

    def test_tracklet_is_anchor_scoped_and_owns_an_immutable_boolean_copy(self) -> None:
        source = self._mask((2, 3, 6, 8))
        frame = self.tool.TrackletFrame(101, source, 1.0, authoritative=True)
        source[:] = False
        self.assertTrue(frame.mask.any())
        self.assertEqual(frame.mask.dtype, np.bool_)
        self.assertFalse(frame.mask.flags.writeable)
        self.assertEqual(frame.tracker_probability, 1.0)
        low_probability = self.tool.TrackletFrame(102, frame.mask, 0.20)
        threshold_probability = self.tool.TrackletFrame(103, frame.mask, 0.50)
        self.assertEqual(low_probability.tracker_probability, 0.20)
        self.assertFalse(low_probability.active)
        self.assertFalse(threshold_probability.active)
        tracklet = self.tool.Tracklet(7, 100, 2, (frame,))
        self.assertEqual(
            tracklet.key,
            self.tool.TrackletKey(tile_index=7, anchor_frame=100, local_object_id=2),
        )

        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            self.tool.Tracklet(
                7,
                100,
                2,
                (
                    self.tool.TrackletFrame(102, frame.mask, 1.0),
                    self.tool.TrackletFrame(101, frame.mask, 1.0),
                ),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            self.tool.TrackletFrame(1, frame.mask, float("nan"))
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            self.tool.TrackletFrame(1, frame.mask, 1.01)
        with self.assertRaisesRegex(ValueError, "authoritative.*nonempty"):
            self.tool.TrackletFrame(1, self._mask(), 0.0, authoritative=True)

    def test_overlap_metrics_report_exact_spatial_and_confidence_agreement(self) -> None:
        mask = self._mask((2, 2, 7, 8))
        frames = list(range(140, 145))
        left = self._tracklet(anchor=100, object_id=0, frames=frames, masks=mask)
        right = self._tracklet(anchor=200, object_id=4, frames=frames, masks=mask)

        metrics = self.tool.compute_overlap_metrics(left, right)

        self.assertEqual(metrics.shared_frame_indices, tuple(frames))
        self.assertEqual(metrics.common_frame_count, 5)
        self.assertEqual(metrics.useful_frame_count, 5)
        self.assertEqual(metrics.coactive_frame_count, 5)
        self.assertEqual(metrics.iou_3d, 1.0)
        self.assertEqual(metrics.dice_3d, 1.0)
        self.assertEqual(metrics.frame_iou_mean, 1.0)
        self.assertEqual(metrics.frame_iou_p10, 1.0)
        self.assertEqual(metrics.centroid_distance_median, 0.0)
        self.assertEqual(metrics.area_log_drift_median, 0.0)
        self.assertEqual(metrics.tracker_probability_mean, 0.95)

    def test_union_confidence_comes_only_from_nonempty_geometry_contributors(self) -> None:
        empty = self._mask()
        geometry = self._mask((2, 2, 7, 8))
        high_empty = self._tracklet(
            anchor=100,
            object_id=0,
            frames=[145],
            masks=empty,
            probabilities=0.99,
            authoritative_frames=set(),
        )
        low_geometry = self._tracklet(
            anchor=100,
            object_id=1,
            frames=[145],
            masks=geometry,
            probabilities=0.20,
            authoritative_frames=set(),
        )

        union = self.tool._union_tracklet_frames((high_empty, low_geometry))[145]

        self.assertTrue(np.array_equal(union.mask, geometry))
        self.assertEqual(union.tracker_probability, 0.20)
        self.assertFalse(union.active)

    def test_hungarian_matching_is_one_to_one_and_preserves_unmatched_keys(self) -> None:
        frames = list(range(145, 151))
        upper = self._mask((1, 1, 5, 5))
        lower = self._mask((7, 7, 11, 11))
        unrelated = self._mask((1, 7, 5, 11))
        left = (
            self._tracklet(anchor=100, object_id=0, frames=frames, masks=upper),
            self._tracklet(anchor=100, object_id=1, frames=frames, masks=lower),
        )
        right = (
            self._tracklet(anchor=200, object_id=7, frames=frames, masks=lower),
            self._tracklet(anchor=200, object_id=8, frames=frames, masks=upper),
            self._tracklet(anchor=200, object_id=9, frames=frames, masks=unrelated),
        )

        result = self.tool.match_tracklets(left, right)

        pairs = {(item.left_key.local_object_id, item.right_key.local_object_id) for item in result.matches}
        self.assertEqual(pairs, {(0, 8), (1, 7)})
        self.assertEqual(result.unmatched_left_keys, ())
        self.assertEqual(
            tuple(item.local_object_id for item in result.unmatched_right_keys),
            (9,),
        )
        self.assertTrue(all(item.similarity > 0.9 for item in result.matches))

    def test_matching_falls_back_when_scipy_optimize_cannot_import(self) -> None:
        frames = list(range(145, 151))
        upper = self._mask((1, 1, 5, 5))
        lower = self._mask((7, 7, 11, 11))
        unrelated = self._mask((1, 7, 5, 11))
        left = (
            self._tracklet(anchor=100, object_id=0, frames=frames, masks=upper),
            self._tracklet(anchor=100, object_id=1, frames=frames, masks=lower),
        )
        right = (
            self._tracklet(anchor=200, object_id=7, frames=frames, masks=lower),
            self._tracklet(anchor=200, object_id=8, frames=frames, masks=upper),
            self._tracklet(anchor=200, object_id=9, frames=frames, masks=unrelated),
        )
        normal = self.tool.match_tracklets(left, right)
        real_import = __import__

        def reject_scipy_optimize(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "scipy.optimize":
                raise ModuleNotFoundError("forced missing scipy.optimize")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=reject_scipy_optimize):
            fallback = self.tool.match_tracklets(left, right)

        normal_pairs = {(item.left_key, item.right_key) for item in normal.matches}
        fallback_pairs = {(item.left_key, item.right_key) for item in fallback.matches}
        self.assertEqual(fallback_pairs, normal_pairs)
        self.assertEqual(fallback.unmatched_left_keys, normal.unmatched_left_keys)
        self.assertEqual(fallback.unmatched_right_keys, normal.unmatched_right_keys)
        self.assertEqual(fallback.assignment_backend, "numpy_hungarian_fallback")

    def test_equal_runner_up_candidates_are_left_unmatched_not_forced(self) -> None:
        frames = list(range(145, 151))
        mask = self._mask((2, 2, 8, 8))
        left = (self._tracklet(anchor=100, object_id=0, frames=frames, masks=mask),)
        right = (
            self._tracklet(anchor=200, object_id=1, frames=frames, masks=mask),
            self._tracklet(anchor=200, object_id=2, frames=frames, masks=mask),
        )

        result = self.tool.match_tracklets(left, right)

        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_left_keys, (left[0].key,))
        self.assertEqual(set(result.unmatched_right_keys), {item.key for item in right})
        assessments = [item for item in result.pair_assessments if item.left_key == left[0].key]
        self.assertTrue(
            all(
                "left_runner_up_margin_below_minimum" in item.gate_failures
                for item in assessments
            )
        )

    def test_residual_split_hypothesis_uses_union_overlap_without_forcing_match(self) -> None:
        frames = list(range(145, 151))
        top = self._mask((2, 2, 5, 9))
        bottom = self._mask((5, 2, 8, 9))
        combined = top | bottom
        left = (self._tracklet(anchor=100, object_id=0, frames=frames, masks=combined),)
        right = (
            self._tracklet(anchor=200, object_id=1, frames=frames, masks=top),
            self._tracklet(anchor=200, object_id=2, frames=frames, masks=bottom),
        )
        config = self.tool.MatchingConfig(min_iou_3d=0.75)

        result = self.tool.match_tracklets(left, right, config=config)

        self.assertEqual(result.matches, ())
        splits = [item for item in result.residual_hypotheses if item.kind == "split"]
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0].left_keys, (left[0].key,))
        self.assertEqual(set(splits[0].right_keys), {item.key for item in right})
        self.assertEqual(splits[0].metrics.iou_3d, 1.0)
        self.assertAlmostEqual(splits[0].best_individual_iou_3d, 0.5)
        self.assertAlmostEqual(splits[0].union_iou_gain, 0.5)

    def test_split_hypothesis_can_extend_an_accepted_match_with_a_residual_child(self) -> None:
        frames = list(range(145, 151))
        large_child = self._mask((2, 2, 8, 9))
        small_child = self._mask((8, 2, 10, 9))
        parent = large_child | small_child
        left = (self._tracklet(anchor=100, object_id=0, frames=frames, masks=parent),)
        right = (
            self._tracklet(anchor=200, object_id=1, frames=frames, masks=large_child),
            self._tracklet(anchor=200, object_id=2, frames=frames, masks=small_child),
        )

        result = self.tool.match_tracklets(left, right)

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].right_key, right[0].key)
        self.assertEqual(result.unmatched_right_keys, (right[1].key,))
        splits = [item for item in result.residual_hypotheses if item.kind == "split"]
        self.assertEqual(len(splits), 1)
        self.assertEqual(set(splits[0].right_keys), {item.key for item in right})
        self.assertEqual(splits[0].metrics.iou_3d, 1.0)

    def test_residual_merge_hypothesis_is_symmetric(self) -> None:
        frames = list(range(145, 151))
        left_half = self._mask((2, 2, 9, 5))
        right_half = self._mask((2, 5, 9, 8))
        combined = left_half | right_half
        left = (
            self._tracklet(anchor=100, object_id=0, frames=frames, masks=left_half),
            self._tracklet(anchor=100, object_id=1, frames=frames, masks=right_half),
        )
        right = (self._tracklet(anchor=200, object_id=2, frames=frames, masks=combined),)

        result = self.tool.match_tracklets(
            left,
            right,
            config=self.tool.MatchingConfig(min_iou_3d=0.75),
        )

        merges = [item for item in result.residual_hypotheses if item.kind == "merge"]
        self.assertEqual(len(merges), 1)
        self.assertEqual(set(merges[0].left_keys), {item.key for item in left})
        self.assertEqual(merges[0].right_keys, (right[0].key,))
        self.assertEqual(merges[0].metrics.iou_3d, 1.0)

    def test_merge_hypothesis_can_extend_an_accepted_match_with_a_residual_parent(self) -> None:
        frames = list(range(145, 151))
        large_parent = self._mask((2, 2, 8, 9))
        small_parent = self._mask((8, 2, 10, 9))
        child = large_parent | small_parent
        left = (
            self._tracklet(anchor=100, object_id=0, frames=frames, masks=large_parent),
            self._tracklet(anchor=100, object_id=1, frames=frames, masks=small_parent),
        )
        right = (self._tracklet(anchor=200, object_id=2, frames=frames, masks=child),)

        result = self.tool.match_tracklets(left, right)

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].left_key, left[0].key)
        self.assertEqual(result.unmatched_left_keys, (left[1].key,))
        merges = [item for item in result.residual_hypotheses if item.kind == "merge"]
        self.assertEqual(len(merges), 1)
        self.assertEqual(set(merges[0].left_keys), {item.key for item in left})
        self.assertEqual(merges[0].right_keys, (right[0].key,))
        self.assertEqual(merges[0].metrics.iou_3d, 1.0)

    def test_handoff_uses_one_monotonic_score_weighted_cut(self) -> None:
        frames = list(range(145, 156))
        mask = self._mask((3, 3, 9, 9))
        left_probabilities = {frame: (0.99 if frame <= 150 else 0.55) for frame in frames}
        right_probabilities = {frame: (0.55 if frame <= 150 else 0.99) for frame in frames}
        left = self._tracklet(
            anchor=100,
            object_id=0,
            frames=frames,
            masks=mask,
            probabilities=left_probabilities,
        )
        right = self._tracklet(
            anchor=200,
            object_id=1,
            frames=frames,
            masks=mask,
            probabilities=right_probabilities,
        )

        handoff = self.tool.choose_monotonic_handoff(left, right, midpoint_frame=150.5)

        self.assertEqual(handoff.left_last_shared_frame, 150)
        self.assertEqual(handoff.right_first_shared_frame, 151)
        sources = [item.source for item in handoff.choices]
        first_right = sources.index("right")
        self.assertTrue(all(source == "left" for source in sources[:first_right]))
        self.assertTrue(all(source == "right" for source in sources[first_right:]))

    def test_matched_composition_preserves_both_authoritative_anchor_masks(self) -> None:
        seam = self._mask((3, 3, 9, 9))
        left_anchor_mask = seam | self._mask((2, 2, 3, 3))
        right_anchor_mask = seam | self._mask((9, 9, 10, 10))
        left = self._tracklet(
            anchor=100,
            object_id=0,
            frames=[100, 145, 146, 147],
            masks={100: left_anchor_mask, 145: seam, 146: seam, 147: seam},
        )
        right = self._tracklet(
            anchor=200,
            object_id=1,
            frames=[145, 146, 147, 200],
            masks={145: seam, 146: seam, 147: seam, 200: right_anchor_mask},
        )
        matching = self.tool.match_tracklets((left,), (right,))

        composition = self.tool.compose_binary((left,), (right,), matching)

        self.assertEqual(len(composition.lineages), 1)
        frames = {item.frame_index: item for item in composition.lineages[0].frames}
        self.assertTrue(frames[100].authoritative)
        self.assertTrue(frames[200].authoritative)
        self.assertTrue(np.array_equal(frames[100].mask, left_anchor_mask))
        self.assertTrue(np.array_equal(frames[200].mask, right_anchor_mask))
        union = {item.frame_index: item.mask for item in composition.union_frames}
        self.assertTrue(np.array_equal(union[100], left_anchor_mask))
        self.assertTrue(np.array_equal(union[200], right_anchor_mask))

    def test_authoritative_mask_overrides_a_score_driven_cut(self) -> None:
        seam = self._mask((3, 3, 9, 9))
        left_anchor_mask = seam | self._mask((2, 2, 3, 3))
        right_anchor_mask = seam | self._mask((9, 9, 10, 10))
        frames = [90, 91, 92, 100, 145, 146, 147, 200]
        left_masks = {frame: seam for frame in frames}
        left_masks[100] = left_anchor_mask
        right_masks = {frame: seam for frame in frames}
        right_masks[200] = right_anchor_mask
        left = self._tracklet(
            anchor=100,
            object_id=0,
            frames=frames,
            masks=left_masks,
            probabilities=0.51,
        )
        right = self._tracklet(
            anchor=200,
            object_id=1,
            frames=frames,
            masks=right_masks,
            probabilities=0.99,
        )
        matching = self.tool.match_tracklets((left,), (right,))

        composition = self.tool.compose_binary((left,), (right,), matching)

        lineage = composition.lineages[0]
        by_frame = {item.frame_index: item for item in lineage.frames}
        self.assertLess(lineage.handoff.cut_coordinate, 100)
        self.assertEqual(by_frame[100].source_key, left.key)
        self.assertEqual(by_frame[100].selection_reason, "authoritative_override")
        self.assertTrue(np.array_equal(by_frame[100].mask, left_anchor_mask))
        self.assertEqual(by_frame[200].source_key, right.key)
        self.assertEqual(by_frame[200].selection_reason, "authoritative_cut")
        choices = {item.frame_index: item for item in lineage.handoff.choices}
        self.assertEqual(choices[100].source, "left")
        self.assertEqual(choices[100].selection_reason, "authoritative_override")

    def test_conflicting_dual_authority_fails_closed(self) -> None:
        seam = self._mask((3, 3, 9, 9))
        left_claim = seam | self._mask((2, 2, 3, 3))
        right_claim = seam | self._mask((9, 9, 10, 10))
        frames = [145, 146, 147]
        left = self._tracklet(
            anchor=100,
            object_id=0,
            frames=frames,
            masks={145: seam, 146: left_claim, 147: seam},
            authoritative_frames={146},
        )
        right = self._tracklet(
            anchor=200,
            object_id=1,
            frames=frames,
            masks={145: seam, 146: right_claim, 147: seam},
            authoritative_frames={146},
        )
        matching = self.tool.match_tracklets((left,), (right,))

        with self.assertRaisesRegex(ValueError, "conflicting authoritative masks"):
            self.tool.compose_binary((left,), (right,), matching)

    def test_matched_sparse_fallback_is_explicitly_receipted(self) -> None:
        mask = self._mask((3, 3, 9, 9))
        left = self._tracklet(
            anchor=100,
            object_id=0,
            frames=[100, 145, 146, 147, 205],
            masks=mask,
        )
        right = self._tracklet(
            anchor=200,
            object_id=1,
            frames=[95, 145, 146, 147, 200],
            masks=mask,
        )
        matching = self.tool.match_tracklets((left,), (right,))

        composition = self.tool.compose_binary((left,), (right,), matching)

        lineage = composition.lineages[0]
        by_frame = {item.frame_index: item for item in lineage.frames}
        self.assertEqual(by_frame[95].source_key, right.key)
        self.assertEqual(by_frame[95].selection_reason, "sparse_fallback")
        self.assertEqual(by_frame[205].source_key, left.key)
        self.assertEqual(by_frame[205].selection_reason, "sparse_fallback")
        self.assertEqual(
            [by_frame[index].source_key for index in (95, 100, 145, 200, 205)],
            [right.key, left.key, left.key, right.key, left.key],
        )

    def test_binary_composition_keeps_matched_handoff_and_both_unmatched_tails(self) -> None:
        seam_frames = list(range(145, 151))
        match_mask = self._mask((2, 2, 6, 6))
        left_tail_mask = self._mask((7, 1, 11, 5))
        right_tail_mask = self._mask((1, 7, 5, 11))
        left_match = self._tracklet(
            anchor=100,
            object_id=0,
            frames=seam_frames,
            masks=match_mask,
        )
        right_match = self._tracklet(
            anchor=200,
            object_id=3,
            frames=seam_frames,
            masks=match_mask,
        )
        left_unmatched = self._tracklet(
            anchor=100,
            object_id=1,
            frames=[145, 150, 175, 205],
            masks=left_tail_mask,
        )
        right_unmatched = self._tracklet(
            anchor=200,
            object_id=4,
            frames=[95, 125, 145, 150],
            masks=right_tail_mask,
        )
        left = (left_match, left_unmatched)
        right = (right_match, right_unmatched)
        matching = self.tool.match_tracklets(left, right)

        composed = self.tool.compose_binary(left, right, matching)

        statuses = [item.status for item in composed.lineages]
        self.assertEqual(statuses.count("matched"), 1)
        self.assertEqual(statuses.count("left_unmatched"), 1)
        self.assertEqual(statuses.count("right_unmatched"), 1)
        left_lineage = next(item for item in composed.lineages if item.status == "left_unmatched")
        right_lineage = next(item for item in composed.lineages if item.status == "right_unmatched")
        self.assertEqual([item.frame_index for item in left_lineage.frames], [145, 150, 175, 205])
        self.assertEqual([item.frame_index for item in right_lineage.frames], [95, 125, 145, 150])
        union = {item.frame_index: item for item in composed.union_frames}
        self.assertTrue(np.array_equal(union[205].mask, left_tail_mask))
        self.assertTrue(np.array_equal(union[95].mask, right_tail_mask))
        self.assertTrue(np.array_equal(union[145].mask, match_mask | left_tail_mask | right_tail_mask))

    def test_binary_composition_rejects_overlapping_or_duplicate_matching_keys(self) -> None:
        frames = [145, 146, 147]
        mask = self._mask((2, 2, 8, 8))
        left = (self._tracklet(anchor=100, object_id=0, frames=frames, masks=mask),)
        right = (self._tracklet(anchor=200, object_id=1, frames=frames, masks=mask),)
        valid = self.tool.match_tracklets(left, right)
        self.assertEqual(len(valid.matches), 1)

        overlapping = self.tool.MatchingResult(
            left_anchor_frame=valid.left_anchor_frame,
            right_anchor_frame=valid.right_anchor_frame,
            matches=valid.matches,
            unmatched_left_keys=(left[0].key,),
            unmatched_right_keys=valid.unmatched_right_keys,
            pair_assessments=valid.pair_assessments,
            residual_hypotheses=valid.residual_hypotheses,
        )
        with self.assertRaisesRegex(ValueError, "both matched and unmatched"):
            self.tool.compose_binary(left, right, overlapping)

        duplicate = self.tool.MatchingResult(
            left_anchor_frame=valid.left_anchor_frame,
            right_anchor_frame=valid.right_anchor_frame,
            matches=(valid.matches[0], valid.matches[0]),
            unmatched_left_keys=(),
            unmatched_right_keys=(),
            pair_assessments=valid.pair_assessments,
            residual_hypotheses=valid.residual_hypotheses,
        )
        with self.assertRaisesRegex(ValueError, "repeats a matched left key"):
            self.tool.compose_binary(left, right, duplicate)

    def test_cross_tile_and_mismatched_shape_inputs_are_rejected(self) -> None:
        frames = [1, 2, 3]
        left = self._tracklet(
            anchor=0,
            object_id=0,
            frames=frames,
            masks=self._mask((1, 1, 4, 4)),
            tile=0,
        )
        right_other_tile = self._tracklet(
            anchor=4,
            object_id=0,
            frames=frames,
            masks=self._mask((1, 1, 4, 4)),
            tile=1,
        )
        with self.assertRaisesRegex(ValueError, "one tile"):
            self.tool.match_tracklets((left,), (right_other_tile,))

        right_other_shape = self._tracklet(
            anchor=4,
            object_id=0,
            frames=frames,
            masks=self._mask((1, 1, 4, 4), size=10),
            tile=0,
        )
        with self.assertRaisesRegex(ValueError, "common shape"):
            self.tool.match_tracklets((left,), (right_other_shape,))


if __name__ == "__main__":
    unittest.main()
