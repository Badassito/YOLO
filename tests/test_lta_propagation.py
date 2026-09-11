from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from XTA.lta_propagation import (
    LtaMaskSeed,
    LtaPropagationRequest,
    LtaSeedProvenance,
    partition_mask_seed_sessions,
    read_seed_artifact,
    run_mask_injected_session,
    write_seed_artifact,
)
from XTA.lta_sam import SamFramePrediction, SamSessionPlan
from XTA.lta_tile_tracking import LtaLineageId


def _lineage(name: str = "object") -> LtaLineageId:
    return LtaLineageId(
        volume_id="volume",
        physical_view_id="transverse",
        runtime_view_id="transverse__tta_a0",
        tile_config_id="s8_st6",
        lineage_id=name,
    )


def _donut() -> np.ndarray:
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    mask[3, 3] = False
    return mask


def _fill_center(mask: object) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    result[3, 3] = True
    return result


def _seed(name: str, object_id: int, mask: np.ndarray) -> LtaMaskSeed:
    return LtaMaskSeed(
        lineage=_lineage(name),
        frame_index=0,
        object_id=object_id,
        mask=mask,
    )


class LtaPropagationTests(unittest.TestCase):
    def test_seed_session_partition_separates_logged_overlap_with_first_fit(self) -> None:
        first_mask = np.zeros((1, 6_200), dtype=bool)
        first_mask[0, :5_169] = True
        second_mask = np.zeros_like(first_mask)
        second_mask[0, 24 : 24 + 5_315] = True
        third_mask = np.zeros_like(first_mask)
        third_mask[0, 5_500:5_600] = True
        first = _seed("logged-first", 0, first_mask)
        second = _seed("logged-second", 1, second_mask)
        third = _seed("first-fit", 2, third_mask)

        batches = partition_mask_seed_sessions((first, second, third))

        self.assertEqual(
            tuple(tuple(seed.object_id for seed in batch) for batch in batches),
            ((0, 2), (1,)),
        )
        self.assertIs(batches[0][0], first)
        self.assertIs(batches[0][1], third)
        self.assertIs(batches[1][0], second)
        self.assertEqual(np.count_nonzero(first_mask & second_mask), 5_145)

    def test_seed_session_partition_checks_cumulative_not_pairwise_overlap(self) -> None:
        first_mask = np.zeros((1, 300), dtype=bool)
        first_mask[0, :100] = True
        second_mask = np.zeros_like(first_mask)
        second_mask[0, :3] = True
        second_mask[0, 100:197] = True
        third_mask = np.zeros_like(first_mask)
        third_mask[0, 3:6] = True
        third_mask[0, 197:294] = True
        seeds = (
            _seed("cumulative-first", 0, first_mask),
            _seed("cumulative-second", 1, second_mask),
            _seed("cumulative-third", 2, third_mask),
        )

        batches = partition_mask_seed_sessions(seeds)

        # Each overlap with the first mask is only 3%, but both together would
        # leave it 94% exclusive and must therefore open another session.
        self.assertEqual(
            tuple(tuple(seed.object_id for seed in batch) for batch in batches),
            ((0, 1), (2,)),
        )

    def test_seed_session_partition_uses_transformed_injection_geometry(self) -> None:
        center = np.zeros((7, 7), dtype=bool)
        center[3, 3] = True
        donut = _seed("transform-donut", 0, _donut())
        center_seed = _seed("transform-center", 1, center)

        raw_batches = partition_mask_seed_sessions((donut, center_seed))
        filled_batches = partition_mask_seed_sessions(
            (donut, center_seed),
            mask_transform=_fill_center,
        )

        self.assertEqual(raw_batches, ((donut, center_seed),))
        self.assertEqual(filled_batches, ((donut,), (center_seed,)))
        self.assertFalse(donut.mask[3, 3], "compatibility transform mutated the seed")

    def test_seed_session_partition_enforces_capacity_and_validates_inputs(self) -> None:
        seeds = []
        for index in range(129):
            mask = np.zeros((1, 129), dtype=bool)
            mask[0, index] = True
            seeds.append(_seed(f"capacity-{index}", index, mask))

        batches = partition_mask_seed_sessions(seeds)

        self.assertEqual(tuple(len(batch) for batch in batches), (128, 1))
        self.assertEqual(partition_mask_seed_sessions(()), ())
        with self.assertRaisesRegex(TypeError, "LtaMaskSeed"):
            partition_mask_seed_sessions((object(),))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "same shape"):
            partition_mask_seed_sessions(
                (
                    _seed("shape-a", 0, np.ones((1, 1), dtype=bool)),
                    _seed("shape-b", 1, np.ones((1, 2), dtype=bool)),
                )
            )
        different_frame = LtaMaskSeed(
            lineage=_lineage("different-frame"),
            frame_index=1,
            object_id=1,
            mask=np.ones((1, 1), dtype=bool),
        )
        with self.assertRaisesRegex(ValueError, "same frame"):
            partition_mask_seed_sessions(
                (_seed("frame-zero", 0, np.ones((1, 1), dtype=bool)), different_frame)
            )
        duplicate_lineage = _lineage("duplicate")
        with self.assertRaisesRegex(ValueError, "unique lineages"):
            partition_mask_seed_sessions(
                (
                    LtaMaskSeed(duplicate_lineage, 0, 0, np.array([[True, False]])),
                    LtaMaskSeed(duplicate_lineage, 0, 1, np.array([[False, True]])),
                )
            )
        duplicate_object_ids = (
            _seed("duplicate-id-first", 0, np.array([[True, False]])),
            _seed("duplicate-id-second", 0, np.array([[False, True]])),
        )
        self.assertEqual(
            partition_mask_seed_sessions(duplicate_object_ids),
            (duplicate_object_ids,),
        )
        for capacity in (0, 129):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    partition_mask_seed_sessions((), max_objects=capacity)
        with self.assertRaises(TypeError):
            partition_mask_seed_sessions((), max_objects=True)
        for minimum in (0.0, float("nan"), 1.01):
            with self.subTest(minimum=minimum):
                with self.assertRaises(ValueError):
                    partition_mask_seed_sessions(
                        (), min_exclusive_fraction=minimum
                    )
        with self.assertRaises(TypeError):
            partition_mask_seed_sessions((), min_exclusive_fraction=True)
        with self.assertRaises(TypeError):
            partition_mask_seed_sessions((), mask_transform=object())
        transform_seed = _seed("bad-transform", 0, np.ones((2, 2), dtype=bool))
        invalid_transforms = (
            lambda _mask: np.ones((4,), dtype=bool),
            lambda _mask: np.ones((3, 3), dtype=bool),
            lambda mask: np.zeros_like(mask),
        )
        for transform in invalid_transforms:
            with self.subTest(transform=transform):
                with self.assertRaisesRegex(ValueError, "transformed seed mask"):
                    partition_mask_seed_sessions(
                        (transform_seed,), mask_transform=transform
                    )

    def test_seed_artifact_round_trips_without_pickle_and_detects_mutation(self) -> None:
        seeds = (
            LtaMaskSeed(
                lineage=_lineage("first"),
                frame_index=4,
                object_id=2,
                mask=np.ones((7, 7), dtype=bool),
                provenance=LtaSeedProvenance.SPATIAL_RELAY,
                tracker_probability=0.75,
                source_receipt={"edge": "east"},
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = write_seed_artifact(Path(temp_dir) / "seeds.npz", seeds)
            restored = read_seed_artifact(artifact)
            self.assertEqual(restored[0].lineage, seeds[0].lineage)
            self.assertEqual(restored[0].object_id, 2)
            self.assertEqual(restored[0].source_receipt, {"edge": "east"})
            np.testing.assert_array_equal(restored[0].mask, seeds[0].mask)
            with artifact.path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "digest changed"):
                read_seed_artifact(artifact)

    def test_request_requires_mask_seeds_with_global_tile_identity(self) -> None:
        seed = LtaMaskSeed(
            lineage=_lineage(),
            frame_index=2,
            object_id=7,
            mask=_donut(),
        )
        request = LtaPropagationRequest(
            work_id="work",
            session=SamSessionPlan("sequence", 0, 0, 5),
            prompt_frame=2,
            direction="both",
            seeds=(seed,),
            conf=0.15,
        )

        self.assertEqual(request.seeds[0].lineage.tile_config_id, "s8_st6")
        self.assertFalse(request.seeds[0].mask.flags.writeable)
        with self.assertRaisesRegex(ValueError, "prompt frame"):
            LtaPropagationRequest(
                work_id="bad",
                session=request.session,
                prompt_frame=3,
                direction="forward",
                seeds=(seed,),
                conf=0.15,
            )

    def test_production_adapter_fills_injection_predictions_and_dogfood(self) -> None:
        authoritative = _donut()
        captured: dict[str, object] = {}

        def adapter(_measured, _raw, **kwargs):
            captured.update(kwargs)
            injected = np.asarray(kwargs["object_masks"][0], dtype=bool)
            self.assertTrue(injected[3, 3])
            session = kwargs["session"]
            predictions = tuple(
                SamFramePrediction(
                    sequence_id=session.sequence_id,
                    session_index=session.session_index,
                    frame_index=frame,
                    object_id=0,
                    initial_detection_score=1.0,
                    frame_tracker_score=0.9,
                    binary_mask=_donut(),
                )
                for frame in range(kwargs["prompt_frame"], session.frame_stop)
            )
            return {
                "propagation": predictions,
                "adapter": "fake",
                "seed_roundtrip_policy": "overlap-aware",
                "seed_roundtrip_passed": True,
                "anchor_integrity_passed": True,
            }

        request = LtaPropagationRequest(
            work_id="forward-chain",
            session=SamSessionPlan("sequence", 0, 0, 5),
            prompt_frame=2,
            direction="forward",
            seeds=(
                LtaMaskSeed(
                    lineage=_lineage(),
                    frame_index=2,
                    object_id=9,
                    mask=authoritative,
                    provenance=LtaSeedProvenance.AUTHORITATIVE,
                ),
            ),
            conf=0.15,
        )

        result = run_mask_injected_session(
            object(),
            object(),
            resource=[object()] * 5,
            request=request,
            adapter=adapter,
            fill_mask=_fill_center,
        )

        self.assertFalse(authoritative[3, 3], "authoritative input was mutated")
        self.assertEqual(captured["propagation_direction"], "forward")
        self.assertEqual(captured["seed_roundtrip_policy"], "overlap-aware")
        self.assertEqual(
            {item.prediction.frame_index for item in result.predictions},
            {2, 3, 4},
        )
        self.assertTrue(all(item.prediction.binary_mask[3, 3] for item in result.predictions))
        self.assertEqual({item.prediction.object_id for item in result.predictions}, {9})
        self.assertEqual(len(result.dogfood_seeds), 1)
        self.assertEqual(result.dogfood_seeds[0].frame_index, 4)
        self.assertEqual(
            result.dogfood_seeds[0].provenance,
            LtaSeedProvenance.TEMPORAL_DOGFOOD,
        )
        self.assertTrue(result.dogfood_seeds[0].mask[3, 3])
        self.assertGreater(result.hole_fill_added_pixels, 0)

    def test_backward_session_reinjects_prompt_and_uses_first_frame_boundary(self) -> None:
        def adapter(_measured, _raw, **kwargs):
            session = kwargs["session"]
            return {
                "propagation": tuple(
                    SamFramePrediction(
                        sequence_id=session.sequence_id,
                        session_index=session.session_index,
                        frame_index=frame,
                        object_id=0,
                        initial_detection_score=1.0,
                        frame_tracker_score=0.8,
                        binary_mask=np.ones((7, 7), dtype=bool),
                    )
                    for frame in range(session.frame_start, kwargs["prompt_frame"])
                ),
                "seed_roundtrip_policy": "overlap-aware",
                "seed_roundtrip_passed": True,
                "anchor_integrity_passed": True,
            }

        request = LtaPropagationRequest(
            work_id="backward-chain",
            session=SamSessionPlan("sequence", 1, 10, 15),
            prompt_frame=13,
            direction="backward",
            seeds=(
                LtaMaskSeed(
                    lineage=_lineage("backward"),
                    frame_index=13,
                    object_id=0,
                    mask=np.ones((7, 7), dtype=bool),
                    provenance=LtaSeedProvenance.SPATIAL_RELAY,
                ),
            ),
            conf=0.1,
        )

        result = run_mask_injected_session(
            object(),
            object(),
            resource=[object()] * 5,
            request=request,
            adapter=adapter,
            fill_mask=lambda mask: np.asarray(mask, dtype=bool).copy(),
        )

        self.assertEqual(
            {item.prediction.frame_index for item in result.predictions},
            {10, 11, 12, 13},
        )
        self.assertEqual(result.dogfood_seeds[0].frame_index, 10)

    def test_streaming_matches_retained_union_and_dogfood_without_retaining_masks(self) -> None:
        def adapter(_measured, _raw, **kwargs):
            session = kwargs["session"]
            predictions = tuple(
                SamFramePrediction(
                    sequence_id=session.sequence_id,
                    session_index=session.session_index,
                    frame_index=frame,
                    object_id=0,
                    initial_detection_score=1.0,
                    frame_tracker_score=0.9,
                    binary_mask=_donut(),
                )
                for frame in range(kwargs["prompt_frame"], session.frame_stop)
            )
            callback = kwargs.get("prediction_callback")
            if callback is not None:
                for prediction in predictions:
                    callback(prediction)
            return {
                "propagation": (
                    predictions if kwargs.get("retain_predictions", True) else ()
                ),
                "raw_prediction_count": len(predictions),
                "seed_roundtrip_policy": "overlap-aware",
                "seed_roundtrip_passed": True,
                "anchor_integrity_passed": True,
            }

        request = LtaPropagationRequest(
            work_id="stream-forward",
            session=SamSessionPlan("sequence", 3, 0, 5),
            prompt_frame=2,
            direction="forward",
            seeds=(
                LtaMaskSeed(
                    lineage=_lineage("streamed"),
                    frame_index=2,
                    object_id=4,
                    mask=_donut(),
                ),
            ),
            conf=0.15,
        )
        retained = run_mask_injected_session(
            object(),
            object(),
            resource=[object()] * 5,
            request=request,
            adapter=adapter,
            fill_mask=_fill_center,
        )
        streamed_by_frame: dict[int, np.ndarray] = {}

        def reduce_prediction(item) -> None:
            frame = item.prediction.frame_index
            streamed_by_frame.setdefault(frame, np.zeros((7, 7), dtype=bool))
            streamed_by_frame[frame] |= np.asarray(
                item.prediction.binary_mask,
                dtype=bool,
            )

        streamed = run_mask_injected_session(
            object(),
            object(),
            resource=[object()] * 5,
            request=request,
            adapter=adapter,
            fill_mask=_fill_center,
            prediction_callback=reduce_prediction,
            retain_predictions=False,
        )

        self.assertEqual(streamed.predictions, ())
        self.assertFalse(streamed.adapter_receipt["canonical_predictions_retained"])
        self.assertEqual(streamed.adapter_receipt["canonical_prediction_count"], 3)
        self.assertEqual(streamed.adapter_receipt["canonical_callback_prediction_count"], 3)
        for frame in (2, 3, 4):
            np.testing.assert_array_equal(
                streamed_by_frame[frame],
                retained.frame_union(frame),
            )
        self.assertEqual(len(streamed.dogfood_seeds), len(retained.dogfood_seeds))
        np.testing.assert_array_equal(
            streamed.dogfood_seeds[0].mask,
            retained.dogfood_seeds[0].mask,
        )
        self.assertEqual(
            streamed.hole_fill_added_pixels,
            retained.hole_fill_added_pixels,
        )

    def test_bidirectional_stream_restores_each_prompt_and_preserves_both_boundaries(self) -> None:
        seeds = tuple(
            LtaMaskSeed(
                lineage=_lineage(f"bidirectional-{object_id}"),
                frame_index=12,
                object_id=object_id,
                mask=np.eye(7, dtype=bool) if object_id == 3 else np.fliplr(np.eye(7, dtype=bool)),
            )
            for object_id in (9, 3)
        )
        request = LtaPropagationRequest(
            work_id="bidirectional-boundaries",
            session=SamSessionPlan("sequence", 4, 10, 15),
            prompt_frame=12,
            direction="both",
            seeds=seeds,
            conf=0.15,
        )

        def adapter(_measured, _raw, **kwargs):
            predictions = []
            # Match the tracker bridge: forward owns the prompt, then backward
            # visits the preceding frames in descending order.
            for frame in (12, 13, 14, 11, 10):
                for local_id in (0, 1):
                    mask = np.zeros((7, 7), dtype=bool)
                    mask[local_id, frame - 10] = True
                    if frame == 12:
                        mask[:] = True  # Prompt drift must never enter the union.
                    prediction = SamFramePrediction(
                        sequence_id=request.session.sequence_id,
                        session_index=request.session.session_index,
                        frame_index=frame,
                        object_id=local_id,
                        initial_detection_score=1.0,
                        frame_tracker_score=0.8 if frame < 12 else 0.6,
                        binary_mask=mask,
                    )
                    predictions.append(prediction)
                    if kwargs.get("prediction_callback") is not None:
                        kwargs["prediction_callback"](prediction)
            return {
                "propagation": tuple(predictions) if kwargs.get("retain_predictions", True) else (),
                "seed_roundtrip_policy": "overlap-aware",
                "seed_roundtrip_passed": True,
                "anchor_integrity_passed": False,
            }

        retained = run_mask_injected_session(
            object(), object(), resource=[object()] * 5, request=request,
            adapter=adapter, fill_mask=lambda mask: np.asarray(mask, dtype=bool).copy(),
        )
        emitted = []
        streamed = run_mask_injected_session(
            object(), object(), resource=[object()] * 5, request=request,
            adapter=adapter, fill_mask=lambda mask: np.asarray(mask, dtype=bool).copy(),
            prediction_callback=emitted.append, retain_predictions=False,
        )

        self.assertEqual(streamed.predictions, ())
        self.assertEqual(len(emitted), 10)
        self.assertEqual(
            {(item.prediction.frame_index, item.prediction.object_id) for item in emitted},
            {(frame, object_id) for frame in range(10, 15) for object_id in (3, 9)},
        )
        for seed in seeds:
            prompt = next(
                item for item in emitted
                if item.prediction.frame_index == 12 and item.lineage == seed.lineage
            )
            np.testing.assert_array_equal(prompt.prediction.binary_mask, seed.mask)
            self.assertEqual(prompt.prediction.frame_tracker_score, 1.0)
        for frame in range(10, 15):
            np.testing.assert_array_equal(
                np.logical_or.reduce(tuple(
                    item.prediction.binary_mask for item in emitted
                    if item.prediction.frame_index == frame
                )),
                retained.frame_union(frame),
            )
        self.assertEqual(
            {(seed.frame_index, seed.object_id) for seed in streamed.dogfood_seeds},
            {(frame, object_id) for frame in (10, 14) for object_id in (3, 9)},
        )
        for seed, retained_seed in zip(streamed.dogfood_seeds, retained.dogfood_seeds):
            self.assertEqual(seed.lineage, _lineage(f"bidirectional-{seed.object_id}"))
            self.assertEqual(seed.provenance, LtaSeedProvenance.TEMPORAL_DOGFOOD)
            self.assertEqual(seed.tracker_probability, 0.8 if seed.frame_index == 10 else 0.6)
            expected = np.zeros((7, 7), dtype=bool)
            expected[0 if seed.object_id == 3 else 1, seed.frame_index - 10] = True
            np.testing.assert_array_equal(seed.mask, expected)
            np.testing.assert_array_equal(seed.mask, retained_seed.mask)

    def test_adapter_rejects_foreign_sessions_and_broadcastable_masks_before_publication(self) -> None:
        request = LtaPropagationRequest(
            work_id="prediction-contract",
            session=SamSessionPlan("sequence", 2, 10, 13),
            prompt_frame=11,
            direction="both",
            seeds=(LtaMaskSeed(_lineage(), 11, 9, _donut()),),
            conf=0.15,
        )
        for frame in (10, 11, 12):
            for invalid_field, invalid_value, error in (
                ("sequence_id", "stale-sequence", "different session"),
                ("session_index", 1, "different session"),
                ("binary_mask", np.ones((1, 7), dtype=bool), "mask shape"),
                ("binary_mask", np.ones((7, 1), dtype=bool), "mask shape"),
            ):
                for streaming in (False, True):
                    with self.subTest(frame=frame, field=invalid_field, streaming=streaming):
                        fields = {
                            "sequence_id": request.session.sequence_id,
                            "session_index": request.session.session_index,
                            "frame_index": frame,
                            "object_id": 0,
                            "initial_detection_score": 1.0,
                            "frame_tracker_score": 0.9,
                            "binary_mask": _donut(),
                            invalid_field: invalid_value,
                        }
                        prediction = SamFramePrediction(**fields)

                        def adapter(_measured, _raw, **kwargs):
                            if streaming:
                                kwargs["prediction_callback"](prediction)
                            return {
                                "propagation": () if streaming else (prediction,),
                                "seed_roundtrip_policy": "overlap-aware",
                                "seed_roundtrip_passed": True,
                            }

                        emitted = []
                        with self.assertRaisesRegex(RuntimeError, error):
                            run_mask_injected_session(
                                object(), object(), resource=[object()] * 3, request=request,
                                adapter=adapter,
                                fill_mask=lambda mask: np.asarray(mask, dtype=bool).copy(),
                                prediction_callback=emitted.append if streaming else None,
                                retain_predictions=not streaming,
                            )
                        self.assertEqual(emitted, [])

    def test_hole_filling_must_preserve_injection_and_prediction_geometry(self) -> None:
        request = LtaPropagationRequest(
            work_id="fill-contract",
            session=SamSessionPlan("sequence", 0, 0, 2),
            prompt_frame=0,
            direction="forward",
            seeds=(LtaMaskSeed(_lineage(), 0, 0, _donut()),),
            conf=0.15,
        )
        for corrupt_injection in (True, False):
            with self.subTest(corrupt_injection=corrupt_injection):
                emitted = []

                def fill_mask(mask):
                    array = np.asarray(mask, dtype=bool)
                    if corrupt_injection or array.all():
                        return array[:1, :].copy()
                    return array.copy()

                def adapter(_measured, _raw, **kwargs):
                    self.assertFalse(corrupt_injection)
                    kwargs["prediction_callback"](SamFramePrediction(
                        sequence_id="sequence", session_index=0, frame_index=1,
                        object_id=0, initial_detection_score=1.0, frame_tracker_score=0.9,
                        binary_mask=np.ones((7, 7), dtype=bool),
                    ))
                    return {}  # pragma: no cover - callback must reject changed geometry

                with self.assertRaisesRegex(RuntimeError, "hole filling changed.*shape"):
                    run_mask_injected_session(
                        object(), object(), resource=[object()] * 2, request=request,
                        adapter=adapter, fill_mask=fill_mask,
                        prediction_callback=emitted.append, retain_predictions=False,
                    )
                self.assertEqual(emitted, [])

    def test_streaming_callback_error_runs_adapter_cleanup(self) -> None:
        cleaned: list[bool] = []

        def adapter(_measured, _raw, **kwargs):
            session = kwargs["session"]
            prediction = SamFramePrediction(
                sequence_id=session.sequence_id,
                session_index=session.session_index,
                frame_index=1,
                object_id=0,
                initial_detection_score=1.0,
                frame_tracker_score=0.9,
                binary_mask=np.ones((7, 7), dtype=bool),
            )
            try:
                kwargs["prediction_callback"](prediction)
            finally:
                cleaned.append(True)
            return {"propagation": ()}  # pragma: no cover - callback raises

        request = LtaPropagationRequest(
            work_id="callback-error",
            session=SamSessionPlan("sequence", 0, 0, 2),
            prompt_frame=0,
            direction="forward",
            seeds=(
                LtaMaskSeed(
                    lineage=_lineage("error"),
                    frame_index=0,
                    object_id=0,
                    mask=np.ones((7, 7), dtype=bool),
                ),
            ),
            conf=0.15,
        )

        with self.assertRaisesRegex(RuntimeError, "reducer failed"):
            run_mask_injected_session(
                object(),
                object(),
                resource=[object(), object()],
                request=request,
                adapter=adapter,
                fill_mask=lambda mask: np.asarray(mask, dtype=bool).copy(),
                prediction_callback=lambda _item: (_ for _ in ()).throw(
                    RuntimeError("reducer failed")
                ),
                retain_predictions=False,
            )
        self.assertEqual(cleaned, [True])

    def test_low_confidence_tracker_masks_are_not_streamed_but_prompt_is_authoritative(self) -> None:
        def adapter(_measured, _raw, **kwargs):
            session = kwargs["session"]
            callback = kwargs["prediction_callback"]
            for frame in range(session.frame_start, session.frame_stop):
                callback(
                    SamFramePrediction(
                        sequence_id=session.sequence_id,
                        session_index=session.session_index,
                        frame_index=frame,
                        object_id=0,
                        initial_detection_score=1.0,
                        frame_tracker_score=0.4,
                        binary_mask=np.ones((7, 7), dtype=bool),
                    )
                )
            return {
                "propagation": (),
                "seed_roundtrip_policy": "overlap-aware",
                "seed_roundtrip_passed": True,
                "anchor_integrity_passed": True,
            }

        request = LtaPropagationRequest(
            work_id="confidence-filter",
            session=SamSessionPlan("sequence", 0, 0, 3),
            prompt_frame=0,
            direction="forward",
            seeds=(
                LtaMaskSeed(
                    lineage=_lineage("confidence"),
                    frame_index=0,
                    object_id=0,
                    mask=_donut(),
                ),
            ),
            conf=0.5,
        )
        emitted = []
        result = run_mask_injected_session(
            object(),
            object(),
            resource=[object()] * 3,
            request=request,
            adapter=adapter,
            fill_mask=_fill_center,
            prediction_callback=emitted.append,
            retain_predictions=False,
        )

        self.assertEqual([item.prediction.frame_index for item in emitted], [0])
        self.assertEqual(emitted[0].prediction.frame_tracker_score, 1.0)
        self.assertTrue(emitted[0].prediction.binary_mask[3, 3])
        self.assertEqual(result.dogfood_seeds, ())
        self.assertEqual(
            result.adapter_receipt["production_below_confidence_prediction_count"],
            2,
        )

    def test_dogfood_prompt_retains_carried_probability_for_relay_gating(self) -> None:
        request = LtaPropagationRequest(
            work_id="dogfood-confidence",
            session=SamSessionPlan("sequence", 0, 0, 2),
            prompt_frame=0,
            direction="forward",
            seeds=(
                LtaMaskSeed(
                    lineage=_lineage("dogfood-confidence"),
                    frame_index=0,
                    object_id=0,
                    mask=_donut(),
                    provenance=LtaSeedProvenance.TEMPORAL_DOGFOOD,
                    tracker_probability=0.2,
                ),
            ),
            conf=0.15,
        )
        emitted = []

        run_mask_injected_session(
            object(),
            object(),
            resource=[object(), object()],
            request=request,
            adapter=lambda _measured, _raw, **_kwargs: {
                "propagation": (),
                "seed_roundtrip_policy": "overlap-aware",
                "seed_roundtrip_passed": True,
                "anchor_integrity_passed": True,
            },
            fill_mask=_fill_center,
            prediction_callback=emitted.append,
            retain_predictions=False,
        )

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].prediction.frame_index, 0)
        self.assertEqual(emitted[0].prediction.frame_tracker_score, 0.2)

    def test_seed_roundtrip_failure_is_rejected_before_returned_masks_publish(self) -> None:
        request = LtaPropagationRequest(
            work_id="seed-roundtrip-failure",
            session=SamSessionPlan("sequence", 0, 0, 2),
            prompt_frame=0,
            direction="forward",
            seeds=(
                LtaMaskSeed(
                    lineage=_lineage("seed-roundtrip-failure"),
                    frame_index=0,
                    object_id=0,
                    mask=_donut(),
                ),
            ),
            conf=0.15,
        )
        prediction = SamFramePrediction(
            sequence_id=request.session.sequence_id,
            session_index=request.session.session_index,
            frame_index=1,
            object_id=0,
            initial_detection_score=1.0,
            frame_tracker_score=0.9,
            binary_mask=np.ones((7, 7), dtype=bool),
        )

        for receipt in (
            {
                "propagation": (prediction,),
                "seed_roundtrip_policy": "overlap-aware",
                "seed_roundtrip_passed": False,
                "seed_roundtrip_object_audit": {0: {"iou": 0.98}},
            },
            {"propagation": (prediction,)},
        ):
            with self.subTest(receipt=receipt):
                emitted = []
                with self.assertRaisesRegex(
                    RuntimeError,
                    "seed round-trip contract",
                ):
                    run_mask_injected_session(
                        object(),
                        object(),
                        resource=[object(), object()],
                        request=request,
                        adapter=lambda _measured, _raw, **_kwargs: receipt,
                        fill_mask=_fill_center,
                        prediction_callback=emitted.append,
                        retain_predictions=False,
                    )
                self.assertEqual(emitted, [])

    def test_anchor_integrity_drift_is_diagnostic_and_prompt_is_restored(self) -> None:
        authoritative = np.ones((9, 9), dtype=bool)
        drifted = authoritative.copy()
        drifted.reshape(-1)[66:] = False
        request = LtaPropagationRequest(
            work_id="anchor-integrity-diagnostic",
            session=SamSessionPlan("sequence", 0, 0, 2),
            prompt_frame=0,
            direction="forward",
            seeds=(
                LtaMaskSeed(
                    lineage=_lineage("anchor-integrity-diagnostic"),
                    frame_index=0,
                    object_id=0,
                    mask=authoritative,
                ),
            ),
            conf=0.15,
        )
        prompt_prediction = SamFramePrediction(
            sequence_id=request.session.sequence_id,
            session_index=request.session.session_index,
            frame_index=request.prompt_frame,
            object_id=0,
            initial_detection_score=1.0,
            frame_tracker_score=0.9,
            binary_mask=drifted,
        )
        anchor_metrics = {
            0: {
                "tp": 66,
                "fp": 0,
                "fn": 15,
                "iou": 66 / 81,
                "ground_truth_pixels": 81,
                "prediction_pixels": 66,
            }
        }
        emitted = []

        result = run_mask_injected_session(
            object(),
            object(),
            resource=[object(), object()],
            request=request,
            adapter=lambda _measured, _raw, **_kwargs: {
                "propagation": (prompt_prediction,),
                "seed_roundtrip_policy": "overlap-aware",
                "seed_roundtrip_passed": True,
                "anchor_integrity_passed": False,
                "anchor_integrity_object_metrics": anchor_metrics,
            },
            fill_mask=lambda mask: np.asarray(mask, dtype=bool).copy(),
            prediction_callback=emitted.append,
            retain_predictions=False,
        )

        self.assertEqual(result.predictions, ())
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].prediction.frame_index, request.prompt_frame)
        np.testing.assert_array_equal(
            emitted[0].prediction.binary_mask,
            authoritative,
        )
        self.assertFalse(result.adapter_receipt["anchor_integrity_passed"])
        self.assertEqual(
            result.adapter_receipt["anchor_integrity_object_metrics"],
            anchor_metrics,
        )
        self.assertEqual(
            result.adapter_receipt["anchor_integrity_disposition"],
            "diagnostic_only_authoritative_prompt_restored",
        )


if __name__ == "__main__":
    unittest.main()
