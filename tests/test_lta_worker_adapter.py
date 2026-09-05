from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    install_stubs()

from XTA.lta_propagation import (
    LtaMaskSeed,
    LtaObjectPrediction,
    LtaPropagationResult,
    LtaSeedProvenance,
    read_seed_artifact,
    write_seed_artifact,
)
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from XTA.lta_sam import SamFramePrediction
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_worker_adapter import (
    _relay_observation,
    _write_relay_artifacts,
    execute_worker_task,
)
from XTA.lta_tiles import TilePlan
from XTA.lta_windows import WindowPlan


class LtaWorkerAdapterTests(unittest.TestCase):
    def test_disjoint_overlap_episodes_seed_complete_neighbor_handoffs(self) -> None:
        source_tile = TilePlan(0, 0, 4, 6, 4, row=0, column=0)
        destination_tile = TilePlan(2, 0, 4, 6, 4, row=0, column=1)
        lineage = LtaLineageId(
            "volume",
            "transverse",
            "transverse__tta_a0",
            "object",
            tile_config_id="s4_st2",
        )
        seed = LtaMaskSeed(
            lineage=lineage,
            frame_index=0,
            object_id=0,
            mask=np.ones((4, 4), dtype=bool),
            visited_tile_indices=(0, 1),
        )
        observations: dict[tuple[str, int], dict[str, object]] = {}
        # Deliberately insert out of order. Frames 0-1 and 3-4 are two
        # distinct boundary episodes separated by a destination-free frame.
        for frame_index in (4, 0, 3, 1):
            mask = np.zeros((4, 4), dtype=bool)
            mask[1, 3] = True
            _relay_observation(
                observations,
                prediction=LtaObjectPrediction(
                    lineage=lineage,
                    prediction=SamFramePrediction(
                        sequence_id="sequence",
                        session_index=0,
                        frame_index=frame_index,
                        object_id=0,
                        initial_detection_score=1.0,
                        frame_tracker_score=0.9,
                        binary_mask=mask,
                    ),
                    hole_fill_added_pixels=0,
                    source_provenance=LtaSeedProvenance.AUTHORITATIVE,
                ),
                seed_by_lineage={lineage: seed},
                source_tile=source_tile,
                neighbors=((1, destination_tile),),
                min_pixels=1,
                min_probability=0.5,
            )

        episodes = next(iter(observations.values()))["episodes"]
        self.assertEqual(
            [(int(first[0]), int(last[0])) for first, last in episodes],
            [(0, 1), (3, 4)],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            records = _write_relay_artifacts(
                observations,
                output_dir=Path(temp_dir),
                source_tile_index=0,
                generation=0,
            )
            self.assertEqual(
                {
                    (
                        record["temporal_direction"],
                        record["frame_index"],
                        tuple(record["overlap_episode_frame_range"]),
                    )
                    for record in records
                },
                {
                    ("forward", 0, (0, 2)),
                    ("backward", 1, (0, 2)),
                    ("forward", 3, (3, 5)),
                    ("backward", 4, (3, 5)),
                },
            )
            for record in records:
                relay = read_seed_artifact(
                    record["seed_artifact_path"],
                    expected_sha256=record["seed_artifact_sha256"],
                )[0]
                np.testing.assert_array_equal(np.argwhere(relay.mask), [[1, 1]])

    def test_chain_writes_union_and_neighbor_relay_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_path = root / "cache.raw"
            np.zeros((2, 4, 6), dtype=np.uint8).tofile(cache_path)
            stat = cache_path.stat()
            cache = LtaPhysicalViewCacheRef(
                path=cache_path,
                shape=(2, 4, 6),
                dtype="uint8",
                physical_view_id="transverse",
                identity_sha256="a" * 64,
                size_bytes=48,
                mtime_ns=stat.st_mtime_ns,
            )
            lineage = LtaLineageId(
                "volume",
                "transverse",
                "transverse__tta_a0",
                "object",
                tile_config_id="s4_st2",
            )
            seed = LtaMaskSeed(
                lineage=lineage,
                frame_index=0,
                object_id=0,
                mask=np.ones((4, 4), dtype=bool),
                # Ancestry is audit context, not a permanent no-reentry gate.
                # The exact scheduler relay key owns duplicate suppression.
                visited_tile_indices=(0, 1),
            )
            seed_artifact = write_seed_artifact(root / "seed.npz", (seed,))
            window = WindowPlan(
                branch="center",
                ordinal=0,
                frame_start=0,
                frame_stop=2,
                prompt_frame=0,
                direction="both",
                seed_kind="authoritative",
            )

            def propagate(_measured, _raw, *, request, **kwargs):
                values = []
                for frame in (0, 1):
                    mask = np.zeros((4, 4), dtype=bool)
                    mask[1, 3] = True
                    prediction = SamFramePrediction(
                        sequence_id=request.session.sequence_id,
                        session_index=request.session.session_index,
                        frame_index=frame,
                        object_id=0,
                        initial_detection_score=1.0,
                        frame_tracker_score=0.9,
                        binary_mask=mask,
                    )
                    values.append(
                        LtaObjectPrediction(
                            lineage=lineage,
                            prediction=prediction,
                            hole_fill_added_pixels=0,
                            source_provenance=LtaSeedProvenance.AUTHORITATIVE,
                        )
                    )
                callback = kwargs.get("prediction_callback")
                if callback is not None:
                    for value in values:
                        callback(value)
                return LtaPropagationResult(
                    request=request,
                    predictions=(
                        tuple(values)
                        if kwargs.get("retain_predictions", True)
                        else ()
                    ),
                    dogfood_seeds=(),
                    adapter_receipt={
                        "fake": True,
                        "canonical_prediction_count": len(values),
                        "canonical_predictions_retained": kwargs.get(
                            "retain_predictions",
                            True,
                        ),
                    },
                    hole_fill_added_pixels=0,
                )

            payload = {
                "work_id": "work",
                "sequence_id": "sequence",
                "cache_ref": cache.payload(),
                "tile_index": 0,
                "tile_config_id": "s4_st2",
                "tile": {
                    "left": 0,
                    "top": 0,
                    "size": 4,
                    "source_width": 6,
                    "source_height": 4,
                    "row": 0,
                    "column": 0,
                },
                "neighbors": [
                    {
                        "tile_index": 1,
                        "direction": "east",
                        "overlap_xyxy": [2, 0, 4, 4],
                        "tile": {
                            "left": 2,
                            "top": 0,
                            "size": 4,
                            "source_width": 6,
                            "source_height": 4,
                            "row": 0,
                            "column": 1,
                        },
                    }
                ],
                "seed_artifact_path": str(seed_artifact.path),
                "seed_artifact_sha256": seed_artifact.sha256,
                "windows": [
                    {
                        "branch": window.branch,
                        "ordinal": window.ordinal,
                        "frame_start": window.frame_start,
                        "frame_stop": window.frame_stop,
                        "prompt_frame": window.prompt_frame,
                        "direction": window.direction,
                        "seed_kind": window.seed_kind,
                    }
                ],
                "output_frame_start": 0,
                "output_frame_stop": 2,
                "conf": 0.15,
                "empty_frame_limit": 30,
                "relay_generation": 0,
                "relay_min_pixels": 1,
                "output_dir": str(root / "output"),
            }
            context = types.SimpleNamespace(
                predictor=object(),
                profile={"name": "fake"},
                sam_runtime={"distribution_version": "fake"},
                constrained_batches=None,
            )
            with (
                mock.patch(
                    "XTA.lta_postprocessing.fill_binary_mask_holes_2d",
                    side_effect=lambda mask: np.asarray(mask, dtype=bool).copy(),
                ),
                mock.patch(
                    "XTA.lta_propagation.run_mask_injected_session",
                    side_effect=propagate,
                ),
                mock.patch(
                    "XTA.lta_rendering.render_native_tile_window",
                    return_value=[object(), object()],
                ),
            ):
                output = execute_worker_task(context, "propagation_chain", payload)

            manifest = __import__("json").loads(
                Path(output["artifact_path"]).read_text(encoding="utf-8")
            )
            union = np.fromfile(manifest["union"]["path"], dtype=np.uint8).reshape(2, 4, 4)
            self.assertTrue(union[:, 1, 3].all())
            self.assertEqual(
                int(np.count_nonzero(union)),
                2,
                "fresh w+ union storage must remain zero outside predicted pixels",
            )
            self.assertEqual(len(manifest["relays"]), 2)
            self.assertEqual(manifest["windows"][0]["prediction_count"], 2)
            self.assertEqual(manifest["windows"][0]["retained_prediction_count"], 0)
            self.assertFalse(
                manifest["windows"][0]["adapter"]["canonical_predictions_retained"]
            )
            self.assertEqual(
                manifest["lineage_active_frame_ranges"],
                {lineage.token: [[0, 2]]},
            )
            directions = {record["temporal_direction"] for record in manifest["relays"]}
            self.assertEqual(directions, {"forward", "backward"})
            for record in manifest["relays"]:
                relay = read_seed_artifact(
                    record["seed_artifact_path"],
                    expected_sha256=record["seed_artifact_sha256"],
                )[0]
                self.assertEqual(relay.visited_tile_indices, (0, 1))
                self.assertEqual(relay.relay_generation, 1)
                np.testing.assert_array_equal(
                    np.argwhere(relay.mask),
                    np.asarray([[1, 1]]),
                )

    def test_chain_partitions_nearly_coincident_lineages_into_separate_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_path = root / "cache.raw"
            np.zeros((1, 4, 4), dtype=np.uint8).tofile(cache_path)
            stat = cache_path.stat()
            cache = LtaPhysicalViewCacheRef(
                path=cache_path,
                shape=(1, 4, 4),
                dtype="uint8",
                physical_view_id="transverse",
                identity_sha256="b" * 64,
                size_bytes=16,
                mtime_ns=stat.st_mtime_ns,
            )
            first = np.zeros((4, 4), dtype=bool)
            first[:3, :3] = True
            second = first.copy()
            second[3, 3] = True
            lineages = tuple(
                LtaLineageId(
                    "volume",
                    "transverse",
                    "transverse__tta_a0",
                    f"object-{index}",
                    tile_config_id="s4_st4",
                )
                for index in range(2)
            )
            seeds = tuple(
                LtaMaskSeed(
                    lineage=lineage,
                    frame_index=0,
                    object_id=index,
                    mask=mask,
                    provenance=LtaSeedProvenance.SPATIAL_RELAY,
                )
                for index, (lineage, mask) in enumerate(
                    zip(lineages, (first, second))
                )
            )
            seed_artifact = write_seed_artifact(root / "seed.npz", seeds)
            window = WindowPlan(
                branch="backward",
                ordinal=0,
                frame_start=0,
                frame_stop=1,
                prompt_frame=0,
                direction="backward",
                seed_kind="spatial_relay",
            )
            requests = []

            def propagate(_measured, _raw, *, request, **kwargs):
                requests.append(request)
                values = []
                for seed in request.seeds:
                    item = LtaObjectPrediction(
                        lineage=seed.lineage,
                        prediction=SamFramePrediction(
                            sequence_id=request.session.sequence_id,
                            session_index=request.session.session_index,
                            frame_index=0,
                            object_id=seed.object_id,
                            initial_detection_score=1.0,
                            frame_tracker_score=0.9,
                            binary_mask=seed.mask,
                        ),
                        hole_fill_added_pixels=0,
                        source_provenance=seed.provenance,
                    )
                    values.append(item)
                    kwargs["prediction_callback"](item)
                return LtaPropagationResult(
                    request=request,
                    predictions=(),
                    dogfood_seeds=(),
                    adapter_receipt={
                        "canonical_prediction_count": len(values),
                        "canonical_predictions_retained": False,
                    },
                    hole_fill_added_pixels=0,
                )

            payload = {
                "work_id": "overlap-partition",
                "sequence_id": "sequence",
                "cache_ref": cache.payload(),
                "tile_index": 0,
                "tile_config_id": "s4_st4",
                "tile": {
                    "left": 0,
                    "top": 0,
                    "size": 4,
                    "source_width": 4,
                    "source_height": 4,
                    "row": 0,
                    "column": 0,
                },
                "neighbors": [],
                "seed_artifact_path": str(seed_artifact.path),
                "seed_artifact_sha256": seed_artifact.sha256,
                "windows": [
                    {
                        "branch": window.branch,
                        "ordinal": window.ordinal,
                        "frame_start": window.frame_start,
                        "frame_stop": window.frame_stop,
                        "prompt_frame": window.prompt_frame,
                        "direction": window.direction,
                        "seed_kind": window.seed_kind,
                    }
                ],
                "output_frame_start": 0,
                "output_frame_stop": 1,
                "conf": 0.15,
                "empty_frame_limit": 30,
                "relay_generation": 1,
                "relay_min_pixels": 1,
                "output_dir": str(root / "output"),
            }
            context = types.SimpleNamespace(
                predictor=object(),
                profile={"name": "fake"},
                sam_runtime={"distribution_version": "fake"},
                constrained_batches=None,
            )
            with (
                mock.patch(
                    "XTA.lta_postprocessing.fill_binary_mask_holes_2d",
                    side_effect=lambda mask: np.asarray(mask, dtype=bool).copy(),
                ),
                mock.patch(
                    "XTA.lta_propagation.run_mask_injected_session",
                    side_effect=propagate,
                ),
                mock.patch(
                    "XTA.lta_rendering.render_native_tile_window",
                    return_value=[object()],
                ),
            ):
                output = execute_worker_task(context, "propagation_chain", payload)

            self.assertEqual([len(request.seeds) for request in requests], [1, 1])
            self.assertEqual(
                [request.session.session_index for request in requests],
                [0, 1],
            )
            manifest = __import__("json").loads(
                Path(output["artifact_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["seed_session_partition"]["tracker_session_count"],
                2,
            )
            self.assertEqual(len(manifest["windows"]), 2)
            self.assertEqual(
                [item["seed_partition_index"] for item in manifest["windows"]],
                [0, 1],
            )
            self.assertTrue(
                all(item["seed_partition_count"] == 2 for item in manifest["windows"])
            )
            union = np.fromfile(
                manifest["union"]["path"],
                dtype=np.uint8,
            ).reshape(1, 4, 4)
            np.testing.assert_array_equal(union[0] != 0, first | second)

    def test_converged_dogfood_is_repartitioned_across_two_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_path = root / "cache.raw"
            np.zeros((3, 4, 4), dtype=np.uint8).tofile(cache_path)
            stat = cache_path.stat()
            cache = LtaPhysicalViewCacheRef(
                path=cache_path,
                shape=(3, 4, 4),
                dtype="uint8",
                physical_view_id="transverse",
                identity_sha256="c" * 64,
                size_bytes=48,
                mtime_ns=stat.st_mtime_ns,
            )
            lineages = tuple(
                LtaLineageId(
                    "volume",
                    "transverse",
                    "transverse__tta_a0",
                    f"dogfood-{index}",
                    tile_config_id="s4_st4",
                )
                for index in range(2)
            )
            initial_masks = []
            for row, column in ((0, 0), (3, 3)):
                mask = np.zeros((4, 4), dtype=bool)
                mask[row, column] = True
                initial_masks.append(mask)
            initial_seeds = tuple(
                LtaMaskSeed(
                    lineage=lineage,
                    frame_index=0,
                    object_id=index,
                    mask=mask,
                    provenance=LtaSeedProvenance.AUTHORITATIVE,
                )
                for index, (lineage, mask) in enumerate(
                    zip(lineages, initial_masks)
                )
            )
            seed_artifact = write_seed_artifact(root / "seed.npz", initial_seeds)
            windows = (
                WindowPlan(
                    branch="center",
                    ordinal=0,
                    frame_start=0,
                    frame_stop=2,
                    prompt_frame=0,
                    direction="both",
                    seed_kind="authoritative",
                ),
                WindowPlan(
                    branch="forward",
                    ordinal=1,
                    frame_start=1,
                    frame_stop=3,
                    prompt_frame=1,
                    direction="forward",
                    seed_kind="dogfood",
                ),
            )
            converged = np.zeros((4, 4), dtype=bool)
            converged[:3, :3] = True
            terminal_masks = []
            for row, column in ((0, 1), (2, 3)):
                mask = np.zeros((4, 4), dtype=bool)
                mask[row, column] = True
                terminal_masks.append(mask)
            requests = []
            callback_events: list[tuple[int, str, int, int]] = []

            def propagate(_measured, _raw, *, request, **kwargs):
                requests.append(request)
                predictions = []
                if request.prompt_frame == 0:
                    frames = (0, 1)
                else:
                    frames = (1, 2)
                for seed in request.seeds:
                    lineage_index = lineages.index(seed.lineage)
                    for frame_index in frames:
                        if frame_index == 0:
                            mask = initial_masks[lineage_index]
                        elif frame_index == 1:
                            mask = converged
                        else:
                            mask = terminal_masks[lineage_index]
                        item = LtaObjectPrediction(
                            lineage=seed.lineage,
                            prediction=SamFramePrediction(
                                sequence_id=request.session.sequence_id,
                                session_index=request.session.session_index,
                                frame_index=frame_index,
                                object_id=seed.object_id,
                                initial_detection_score=1.0,
                                frame_tracker_score=0.9,
                                binary_mask=mask,
                            ),
                            hole_fill_added_pixels=0,
                            source_provenance=seed.provenance,
                        )
                        predictions.append(item)
                        callback_events.append(
                            (
                                request.session.session_index,
                                seed.lineage.lineage_id,
                                seed.object_id,
                                frame_index,
                            )
                        )
                        kwargs["prediction_callback"](item)
                dogfood = (
                    tuple(
                        LtaMaskSeed(
                            lineage=seed.lineage,
                            frame_index=1,
                            object_id=seed.object_id,
                            mask=converged,
                            provenance=LtaSeedProvenance.TEMPORAL_DOGFOOD,
                            tracker_probability=0.9,
                        )
                        for seed in request.seeds
                    )
                    if request.prompt_frame == 0
                    else ()
                )
                return LtaPropagationResult(
                    request=request,
                    predictions=(),
                    dogfood_seeds=dogfood,
                    adapter_receipt={
                        "canonical_prediction_count": len(predictions),
                        "canonical_predictions_retained": False,
                    },
                    hole_fill_added_pixels=0,
                )

            payload = {
                "work_id": "dogfood-overlap-repartition",
                "sequence_id": "sequence",
                "cache_ref": cache.payload(),
                "tile_index": 0,
                "tile_config_id": "s4_st4",
                "tile": {
                    "left": 0,
                    "top": 0,
                    "size": 4,
                    "source_width": 4,
                    "source_height": 4,
                    "row": 0,
                    "column": 0,
                },
                "neighbors": [],
                "seed_artifact_path": str(seed_artifact.path),
                "seed_artifact_sha256": seed_artifact.sha256,
                "windows": [
                    {
                        "branch": window.branch,
                        "ordinal": window.ordinal,
                        "frame_start": window.frame_start,
                        "frame_stop": window.frame_stop,
                        "prompt_frame": window.prompt_frame,
                        "direction": window.direction,
                        "seed_kind": window.seed_kind,
                    }
                    for window in windows
                ],
                "output_frame_start": 0,
                "output_frame_stop": 3,
                "conf": 0.15,
                "empty_frame_limit": 30,
                "relay_generation": 0,
                "relay_min_pixels": 1,
                "output_dir": str(root / "output"),
            }
            context = types.SimpleNamespace(
                predictor=object(),
                profile={"name": "fake"},
                sam_runtime={"distribution_version": "fake"},
                constrained_batches=None,
            )
            with (
                mock.patch(
                    "XTA.lta_postprocessing.fill_binary_mask_holes_2d",
                    side_effect=lambda mask: np.asarray(mask, dtype=bool).copy(),
                ),
                mock.patch(
                    "XTA.lta_propagation.run_mask_injected_session",
                    side_effect=propagate,
                ),
                mock.patch(
                    "XTA.lta_rendering.render_native_tile_window",
                    side_effect=lambda _cache, *, frame_start, frame_stop, **_kwargs: (
                        [object()] * (frame_stop - frame_start)
                    ),
                ),
            ):
                output = execute_worker_task(context, "propagation_chain", payload)

            self.assertEqual([len(request.seeds) for request in requests], [2, 1, 1])
            self.assertEqual(
                [request.session.session_index for request in requests],
                [0, 1, 2],
            )
            self.assertEqual(
                [
                    tuple(
                        (seed.lineage.lineage_id, seed.object_id)
                        for seed in request.seeds
                    )
                    for request in requests
                ],
                [
                    (("dogfood-0", 0), ("dogfood-1", 1)),
                    (("dogfood-0", 0),),
                    (("dogfood-1", 1),),
                ],
            )
            self.assertEqual(len(callback_events), len(set(callback_events)))
            manifest = __import__("json").loads(
                Path(output["artifact_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["seed_session_partition"]["planned_window_count"],
                2,
            )
            self.assertEqual(
                manifest["seed_session_partition"]["tracker_session_count"],
                3,
            )
            self.assertEqual(len(manifest["windows"]), 3)
            self.assertEqual(
                [
                    (
                        item["window"]["prompt_frame"],
                        item["seed_partition_index"],
                        item["seed_partition_count"],
                    )
                    for item in manifest["windows"]
                ],
                [(0, 0, 1), (1, 0, 2), (1, 1, 2)],
            )
            self.assertEqual(
                output["metrics"],
                {
                    "foreground_pixels": 13,
                    "relay_count": 0,
                    "window_count": 3,
                    "planned_window_count": 2,
                    "tracker_session_count": 3,
                },
            )
            union = np.fromfile(
                manifest["union"]["path"],
                dtype=np.uint8,
            ).reshape(3, 4, 4)
            np.testing.assert_array_equal(
                union[0] != 0,
                initial_masks[0] | initial_masks[1],
            )
            np.testing.assert_array_equal(union[1] != 0, converged)
            np.testing.assert_array_equal(
                union[2] != 0,
                terminal_masks[0] | terminal_masks[1],
            )


if __name__ == "__main__":
    unittest.main()
