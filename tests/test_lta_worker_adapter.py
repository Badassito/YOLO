from __future__ import annotations

import tempfile
import types
import unittest
from dataclasses import asdict, replace
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
from XTA.lta_relay_episodes import merge_relay_observations, read_relay_observations
from XTA.lta_sam import SamFramePrediction
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_worker_adapter import (
    _mark_completed_coverage,
    _relay_observation,
    _write_relay_artifacts,
    execute_worker_task,
)
from XTA.lta_tiles import TilePlan
from XTA.lta_union_artifacts import read_union_array
from XTA.lta_windows import AnchorDomain, WindowPlan, owned_frame_range, plan_domain_windows


class LtaWorkerAdapterTests(unittest.TestCase):
    def test_visited_coverage_keeps_gaps_and_splits_directions_at_actual_prompt(self):
        lineage = object()
        request = types.SimpleNamespace(
            seeds=(types.SimpleNamespace(lineage=lineage),),
            session=types.SimpleNamespace(frame_start=10, frame_stop=20),
            prompt_frame=14, direction="both", empty_frame_limit=3,
        )
        coverage = mock.Mock()
        _mark_completed_coverage(coverage, request=request, adapter_receipt={
            "model_visited_frame_ranges": [[10, 12], [13, 17], [18, 20]],
        })
        self.assertEqual(coverage.mark_observed.call_args_list, [
            mock.call((lineage,), frame_start=10, frame_stop=12, prompt_frame=11, direction="backward"),
            mock.call((lineage,), frame_start=13, frame_stop=17, prompt_frame=14, direction="both"),
            mock.call((lineage,), frame_start=18, frame_stop=20, prompt_frame=18, direction="forward"),
        ])
        # Forward-only propagation must not claim the part before the prompt.
        coverage.reset_mock()
        request.direction = "forward"
        _mark_completed_coverage(coverage, request=request, adapter_receipt={
            "model_visited_frame_ranges": [[10, 17], [17, 20]],
        })
        self.assertEqual(coverage.mark_observed.call_args_list, [
            mock.call((lineage,), frame_start=14, frame_stop=17, prompt_frame=14, direction="forward"),
            mock.call((lineage,), frame_start=17, frame_stop=20, prompt_frame=17, direction="forward"),
        ])

    def test_invalid_visit_receipts_are_rejected_before_any_coverage_mark(self):
        request = types.SimpleNamespace(
            seeds=(types.SimpleNamespace(lineage=object()),),
            session=types.SimpleNamespace(frame_start=10, frame_stop=20),
            prompt_frame=14, direction="both", empty_frame_limit=3,
        )
        for ranges in (None, [[9, 12]], [[18, 21]], [[12, 12]], [[10, 13], [12, 14]],
                       [[15, 17], [10, 12]], [[True, 12]], [[10.0, 12]], [[10, 12, 13]]):
            coverage = mock.Mock()
            with self.subTest(ranges=ranges), self.assertRaisesRegex(RuntimeError, "model-visited"):
                _mark_completed_coverage(coverage, request=request,
                                         adapter_receipt={"model_visited_frame_ranges": ranges})
            coverage.mark_observed.assert_not_called()

    def test_missing_visit_receipt_is_conservative_when_early_stopping_is_possible(self):
        lineage = object()
        request = types.SimpleNamespace(
            seeds=(types.SimpleNamespace(lineage=lineage),),
            session=types.SimpleNamespace(frame_start=10, frame_stop=20),
            prompt_frame=14, direction="both", empty_frame_limit=3,
        )
        coverage = mock.Mock()
        _mark_completed_coverage(coverage, request=request, adapter_receipt={})
        coverage.mark_observed.assert_called_once_with(
            (lineage,), frame_start=14, frame_stop=15, prompt_frame=14, direction="both")
        for limit in (None, 10, 30):
            coverage.reset_mock()
            request.empty_frame_limit = limit
            _mark_completed_coverage(coverage, request=request, adapter_receipt={})
            coverage.mark_observed.assert_called_once_with(
                (lineage,), frame_start=10, frame_stop=20, prompt_frame=14, direction="both")

    def _coverage_task(self, root):
        cache_path = root / "cache.raw"
        np.zeros((4, 4, 4), dtype=np.uint8).tofile(cache_path)
        stat = cache_path.stat()
        cache = LtaPhysicalViewCacheRef(cache_path, (4, 4, 4), "uint8", "transverse",
                                       "d" * 64, stat.st_size, stat.st_mtime_ns)
        lineage = LtaLineageId("volume", "transverse", "runtime", "coverage-object", "s4_st4")
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        seed = LtaMaskSeed(lineage, 1, 0, mask)
        artifact = write_seed_artifact(root / "seed.npz", (seed,))
        windows = (
            WindowPlan("center", 0, 1, 3, 1, "both", "authoritative"),
            WindowPlan("forward", 1, 2, 4, 2, "forward", "dogfood"),
        )
        payload = {
            "work_id": "coverage-task", "sequence_id": "sequence", "cache_ref": cache.payload(),
            "tile_index": 0, "tile_config_id": "s4_st4", "tile": asdict(TilePlan(0, 0, 4, 4, 4)),
            "neighbors": [], "seed_artifact_path": str(artifact.path), "seed_artifact_sha256": artifact.sha256,
            "windows": [asdict(window) for window in windows], "output_frame_start": 1, "output_frame_stop": 4,
            "conf": 0.15, "empty_frame_limit": 30, "relay_generation": 0,
            "output_dir": str(root / "output"), "worker_trace_root": str(root / "trace"),
        }
        context = types.SimpleNamespace(predictor=object(), profile={"name": "fake"},
                                        sam_runtime={}, constrained_batches=None)
        return payload, seed, context

    def test_coverage_marks_only_completed_context_and_keeps_authoritative_prompt(self):
        from XTA.lta_coverage import LtaCoverageBuilder, LtaCoverageLedger
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload, seed, context = self._coverage_task(root)
            builder = LtaCoverageBuilder((4, 4))
            spy = mock.Mock(wraps=builder)
            returned = []
            def observe(*args, **kwargs):
                self.assertEqual(returned, [True], "observation was marked before the session completed")
                return builder.mark_observed(*args, **kwargs)
            spy.mark_observed.side_effect = observe
            def propagate(_measured, _raw, *, request, prediction_callback, **kwargs):
                for frame in (1, 2):
                    mask = seed.mask if frame == 1 else np.zeros((4, 4), dtype=bool)
                    prediction_callback(LtaObjectPrediction(seed.lineage, SamFramePrediction(
                        request.session.sequence_id, request.session.session_index, frame, 0, 1.0, mask, 0.9,
                    ), 0, LtaSeedProvenance.AUTHORITATIVE))
                returned.append(True)
                return LtaPropagationResult(request=request, predictions=(), dogfood_seeds=(),
                    adapter_receipt={"canonical_prediction_count": 2, "canonical_predictions_retained": False},
                    hole_fill_added_pixels=0)
            with (mock.patch("XTA.lta_coverage.LtaCoverageBuilder", return_value=spy),
                  mock.patch("XTA.lta_propagation.run_mask_injected_session", side_effect=propagate),
                  mock.patch("XTA.lta_rendering.render_native_tile_window", return_value=[object(), object()])):
                result = execute_worker_task(context, "propagation_chain", payload)
            manifest = __import__("json").loads(Path(result["artifact_path"]).read_text())
            spy.mark_observed.assert_called_once_with((seed.lineage,), frame_start=1, frame_stop=3,
                                                     prompt_frame=1, direction="both")
            self.assertEqual(spy.add_prediction.call_count, 2)
            np.testing.assert_array_equal(spy.add_prediction.call_args_list[0].args[2], seed.mask)
            spy.write.assert_called_once_with(root / "output", work_id="coverage-task", tile_index=0,
                                              tile_config_id="s4_st4")
            coverage = manifest["lineage_coverage"]
            self.assertEqual(coverage["work_id"], "coverage-task")
            self.assertEqual(coverage["tile_shape_hw"], [4, 4])
            self.assertEqual(coverage["lineage_count"], 1)
            self.assertEqual(coverage["mask_count"], 1)
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=4) as ledger:
                ledger.ingest(coverage, expected_work_id="coverage-task", tile_index=0,
                              tile_config_id="s4_st4", frame_start=1, frame_stop=3,
                              expected_lineages=(seed.lineage,))
                self.assertEqual(ledger.stats()["masked_frames"], 1)
                self.assertTrue(ledger.can_handoff(seed, tile_index=0, direction="forward"))
                self.assertFalse(ledger.can_handoff(seed, tile_index=0, direction="backward"))
            self.assertEqual(manifest["windows"][1]["status"], "halted_empty_dogfood_boundary")
            self.assertEqual(read_relay_observations(manifest["relay_observation_artifact"]), {})
            trace_rows = [__import__("json").loads(line) for path in (root / "trace").glob("*.jsonl")
                          for line in path.read_text().splitlines()]
            self.assertTrue(any(row.get("phase") == "lineage_coverage_artifact" and
                                row["event"] == "phase_end" for row in trace_rows))

    def test_failed_session_never_marks_observed_or_writes_coverage(self):
        from XTA.lta_coverage import LtaCoverageBuilder
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload, seed, context = self._coverage_task(root)
            spy = mock.Mock(wraps=LtaCoverageBuilder((4, 4)))
            def fail_after_prediction(_measured, _raw, *, request, prediction_callback, **kwargs):
                prediction_callback(LtaObjectPrediction(seed.lineage, SamFramePrediction(
                    request.session.sequence_id, request.session.session_index, 1, 0, 1.0, seed.mask, 0.9,
                ), 0, LtaSeedProvenance.AUTHORITATIVE))
                raise RuntimeError("intentional session failure")
            with (mock.patch("XTA.lta_coverage.LtaCoverageBuilder", return_value=spy),
                  mock.patch("XTA.lta_propagation.run_mask_injected_session", side_effect=fail_after_prediction),
                  mock.patch("XTA.lta_rendering.render_native_tile_window", return_value=[object(), object()])):
                with self.assertRaisesRegex(RuntimeError, "intentional session failure"):
                    execute_worker_task(context, "propagation_chain", payload)
            self.assertEqual(spy.add_prediction.call_count, 1)
            spy.mark_observed.assert_not_called()
            spy.write.assert_not_called()
            self.assertFalse((root / "output" / "manifest.json").exists())

    def test_policy_zero_tail_does_not_create_coverage_over_an_older_positive_mask(self):
        from XTA.lta_coverage import LtaCoverageBuilder, LtaCoverageLedger
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload, seed, context = self._coverage_task(root)
            payload["empty_frame_limit"] = 1
            payload["windows"] = [asdict(WindowPlan("center", 0, 1, 4, 1, "both", "authoritative"))]
            def propagate(_measured, _raw, *, request, prediction_callback, **kwargs):
                for frame in (1, 2):
                    mask = seed.mask if frame == 1 else np.zeros((4, 4), dtype=bool)
                    prediction_callback(LtaObjectPrediction(seed.lineage, SamFramePrediction(
                        request.session.sequence_id, request.session.session_index, frame, 0, 1.0, mask, 0.9,
                    ), 0, LtaSeedProvenance.AUTHORITATIVE))
                return LtaPropagationResult(request=request, predictions=(), dogfood_seeds=(),
                    adapter_receipt={"canonical_prediction_count": 2, "canonical_predictions_retained": False,
                                     "model_visited_frame_ranges": [[1, 3]], "policy_zero_frame_ranges": [[3, 4]]},
                    hole_fill_added_pixels=0)
            with (mock.patch("XTA.lta_propagation.run_mask_injected_session", side_effect=propagate),
                  mock.patch("XTA.lta_rendering.render_native_tile_window", return_value=[object()] * 3)):
                result = execute_worker_task(context, "propagation_chain", payload)
            manifest = __import__("json").loads(Path(result["artifact_path"]).read_text())
            prior = LtaCoverageBuilder((4, 4))
            prior.add_prediction(seed.lineage, 2, seed.mask)
            prior.mark_observed((seed.lineage,), frame_start=2, frame_stop=3, prompt_frame=2, direction="both")
            prior_receipt = prior.write(root / "prior", work_id="older-positive", tile_index=0, tile_config_id="s4_st4")
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=4) as ledger:
                ledger.ingest(prior_receipt, expected_work_id="older-positive", tile_index=0,
                              tile_config_id="s4_st4", frame_start=1, frame_stop=4, expected_lineages=(seed.lineage,))
                ledger.ingest(manifest["lineage_coverage"], expected_work_id="coverage-task", tile_index=0,
                              tile_config_id="s4_st4", frame_start=1, frame_stop=4, expected_lineages=(seed.lineage,))
                self.assertEqual(ledger.stats()["masked_frames"], 2, "prior positive support must remain intact")
                self.assertEqual(ledger.stats()["covered_directional_transitions"], 1)
                self.assertTrue(ledger.can_handoff(seed, tile_index=0, direction="forward"))
                self.assertFalse(ledger.can_handoff(replace(seed, frame_index=2), tile_index=0, direction="forward"),
                                 "policy-zero frame 3 must not create a traversed edge from frame 2")

    def test_window_tasks_match_whole_chain_masks_and_merged_relay_endpoints(self) -> None:
        from XTA.lta_coverage import LtaCoverageLedger
        from XTA.lta_propagation import run_mask_injected_session as run_session

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_path = root / "cache.raw"
            np.zeros((59, 4, 6), dtype=np.uint8).tofile(cache_path)
            stat = cache_path.stat()
            cache = LtaPhysicalViewCacheRef(
                path=cache_path, shape=(59, 4, 6), dtype="uint8",
                physical_view_id="transverse", identity_sha256="a" * 64,
                size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            )
            lineage = LtaLineageId("volume", "transverse", "runtime", "object")
            mask = np.zeros((4, 4), dtype=bool)
            mask[1, 3] = True
            seed = LtaMaskSeed(
                lineage=lineage, frame_index=19, object_id=3,
                mask=mask, visited_tile_indices=(0,),
            )
            seed_artifact = write_seed_artifact(root / "seed.npz", (seed,))
            windows = plan_domain_windows(AnchorDomain(19, 0, 59))
            source_tile = TilePlan(0, 0, 4, 6, 4)
            neighbor = TilePlan(2, 0, 4, 6, 4)
            base_payload = {
                "work_id": "original-chain", "sequence_id": "sequence",
                "cache_ref": cache.payload(), "tile_index": 0,
                "tile": asdict(source_tile),
                "neighbors": [{"tile_index": 1, "tile": asdict(neighbor)}],
                "seed_artifact_path": str(seed_artifact.path),
                "seed_artifact_sha256": seed_artifact.sha256,
                "windows": [asdict(window) for window in windows],
                "output_frame_start": 0, "output_frame_stop": 59,
                "conf": 0.15, "empty_frame_limit": 30, "relay_generation": 0,
                "relay_min_pixels": 1, "relay_min_probability": 0.5,
                "worker_trace_root": str(root / "trace"),
            }
            context = types.SimpleNamespace(
                predictor=object(), profile={"name": "fake"}, sam_runtime={},
                constrained_batches=None, cpu_budget={"torch_intraop_threads": 1},
            )

            def adapter(_measured, _raw, **kwargs):
                session = kwargs["session"]
                for frame in range(session.frame_start, session.frame_stop):
                    for object_id, object_mask in enumerate(kwargs["object_masks"]):
                        kwargs["prediction_callback"](SamFramePrediction(
                            sequence_id=session.sequence_id, session_index=session.session_index,
                            frame_index=frame, object_id=object_id,
                            initial_detection_score=1.0, frame_tracker_score=0.9,
                            binary_mask=object_mask,
                        ))
                return {
                    "propagation": (), "seed_roundtrip_policy": "overlap-aware",
                    "seed_roundtrip_passed": True, "anchor_integrity_passed": True,
                }

            def execute(kind, payload):
                with (
                    mock.patch("XTA.lta_propagation.run_mask_injected_session", side_effect=(
                        lambda measured, raw, **kwargs: run_session(
                            measured, raw, adapter=adapter,
                            fill_mask=lambda value: np.asarray(value, dtype=bool).copy(),
                            **kwargs,
                        )
                    )),
                    mock.patch("XTA.lta_rendering.render_native_tile_window", side_effect=(
                        lambda _cache, *, frame_start, frame_stop, **_kwargs: [object()] * (frame_stop - frame_start)
                    )),
                    mock.patch("numpy.memmap", side_effect=AssertionError("dense full-chain union allocation")),
                ):
                    output = execute_worker_task(context, kind, payload)
                return __import__("json").loads(Path(output["artifact_path"]).read_text())

            legacy = execute("propagation_chain", {**base_payload, "output_dir": str(root / "legacy")})
            union = np.zeros((59, 4, 4), dtype=np.uint8)
            observations = {}
            parent_seeds = {}
            coverage_receipts = []
            # Execute the forward child before the backward child, as separate
            # workers may do, while preserving each child's parent prompt.
            for ordinal in (0, 2, 1):
                window = windows[ordinal]
                start, stop = owned_frame_range(window)
                payload = {
                    **base_payload, "work_id": f"original-chain::window-{ordinal:04d}",
                    "chain_work_id": "original-chain", "windows": [asdict(window)],
                    "output_frame_start": start, "output_frame_stop": stop,
                    "output_dir": str(root / f"window-{ordinal}"),
                }
                if ordinal:
                    boundary = parent_seeds[window.prompt_frame]
                    payload["seed_artifact_path"] = boundary["path"]
                    payload["seed_artifact_sha256"] = boundary["sha256"]
                    child_seeds = read_seed_artifact(boundary["path"], expected_sha256=boundary["sha256"])
                    self.assertTrue(all(value.provenance is LtaSeedProvenance.TEMPORAL_DOGFOOD for value in child_seeds))
                manifest = execute("propagation_window", payload)
                coverage_receipts.append((manifest["lineage_coverage"], payload["work_id"], window))
                self.assertEqual(manifest["chain_work_id"], "original-chain")
                self.assertEqual(manifest["window"], asdict(window))
                self.assertEqual(manifest["output_frame_range"], [start, stop])
                self.assertEqual(manifest["relays"], [])
                self.assertEqual(manifest["cpu_budget"], context.cpu_budget)
                self.assertTrue(all(receipt["wall_seconds"] >= 0 for receipt in manifest["windows"]))
                self.assertLessEqual(manifest["union"]["shape"][0], 30)
                union[start:stop] |= read_union_array(manifest["union"])
                merge_relay_observations(observations, read_relay_observations(manifest["relay_observation_artifact"]))
                for artifact in manifest["dogfood_seed_artifacts"]:
                    read_seed_artifact(artifact["path"], expected_sha256=artifact["sha256"])
                    parent_seeds[artifact["frame_index"]] = artifact
            np.testing.assert_array_equal(union, read_union_array(legacy["union"]))
            self.assertEqual(int(union.sum()), 59)
            with (LtaCoverageLedger(root / "legacy-coverage.sqlite", frame_count=59) as full_coverage,
                  LtaCoverageLedger(root / "window-coverage.sqlite", frame_count=59) as split_coverage):
                full_coverage.ingest(legacy["lineage_coverage"], expected_work_id="original-chain",
                    tile_index=0, tile_config_id=lineage.tile_config_id, frame_start=0, frame_stop=59,
                    expected_lineages=(lineage,))
                for receipt, work_id, window in coverage_receipts:
                    split_coverage.ingest(receipt, expected_work_id=work_id, tile_index=0,
                        tile_config_id=lineage.tile_config_id, frame_start=window.frame_start,
                        frame_stop=window.frame_stop, expected_lineages=(lineage,))
                for field in ("lineages", "masked_frames", "foreground_pixels", "covered_directional_transitions"):
                    self.assertEqual(full_coverage.stats()[field], split_coverage.stats()[field])
                self.assertEqual(split_coverage.stats()["masked_frames"], 59)
            relays = _write_relay_artifacts(
                observations, output_dir=root / "assembled", source_tile_index=0, generation=0,
            )
            self.assertEqual(
                [record["seed_artifact_sha256"] for record in relays],
                [record["seed_artifact_sha256"] for record in legacy["relays"]],
            )
            self.assertEqual([(record["frame_index"], record["temporal_direction"]) for record in relays], [(0, "forward"), (58, "backward")])
            trace_rows = [
                __import__("json").loads(line)
                for path in (root / "trace").glob("*.jsonl")
                for line in path.read_text().splitlines()
            ]
            phases = {row.get("phase") for row in trace_rows if row["event"] == "phase_end"}
            self.assertTrue({"sam_session", "render_window", "union_chunk_pack", "dogfood_seed_artifacts"}.issubset(phases))

    def test_nested_episode_candidates_keep_novel_pixels_at_their_original_frames(self):
        lineage = LtaLineageId("volume", "transverse", "runtime", "nested", "s4_st4")
        outer = np.zeros((4, 4), dtype=bool)
        outer[1, 1] = True
        inner = outer.copy()
        inner[2, 2] = True
        seed = LtaMaskSeed(lineage, 4, 7, outer, visited_tile_indices=(0,))
        def value(frame, mask, probability):
            return frame, np.packbits(mask), mask.shape, probability
        first, last = value(4, outer, 0.7), value(12, outer, 0.7)
        observation = {
            "lineage": lineage, "seed": seed, "destination_index": 1,
            "episodes": [(first, last)],
            "endpoint_candidates": [
                ("backward", last, (4, 13)),
                ("forward", value(6, inner, 0.9), (6, 11)),
                ("backward", value(10, inner, 0.9), (6, 11)),
                ("forward", first, (4, 13)),
            ],
        }
        key = lineage.token, 1
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = _write_relay_artifacts({key: observation}, output_dir=root / "nested",
                                             source_tile_index=0, generation=2)
            self.assertEqual([(r["frame_index"], r["temporal_direction"]) for r in records],
                             [(4, "forward"), (6, "forward"), (10, "backward"), (12, "backward")])
            self.assertEqual([r["overlap_episode_index"] for r in records], [0, 1, 1, 0])
            for record in records:
                frame = record["frame_index"]
                received = read_seed_artifact(record["seed_artifact_path"],
                                              expected_sha256=record["seed_artifact_sha256"])[0]
                expected_mask = inner if frame in (6, 10) else outer
                expected_range = [6, 11] if frame in (6, 10) else [4, 13]
                np.testing.assert_array_equal(received.mask, expected_mask)
                self.assertEqual(received.tracker_probability, 0.9 if frame in (6, 10) else 0.7)
                self.assertEqual(received.frame_index, frame)
                self.assertEqual(record["overlap_episode_frame_range"], expected_range)
                self.assertEqual(received.source_receipt["overlap_episode_frame_range"], expected_range)
                self.assertEqual(received.source_receipt["overlap_episode_index"], record["overlap_episode_index"])
            self.assertEqual(_write_relay_artifacts(
                {key: {**observation, "endpoint_candidates": []}}, output_dir=root / "empty",
                source_tile_index=0, generation=2), [], "empty candidates must not fall back to merged episodes")

    def test_explicit_ordinary_candidates_match_legacy_episode_hash_order(self):
        lineage = LtaLineageId("volume", "transverse", "runtime", "legacy-order", "s4_st4")
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 2] = True
        seed = LtaMaskSeed(lineage, 4, 0, mask)
        def value(frame):
            return frame, np.packbits(mask), mask.shape, 0.9
        episodes = [(value(4), value(12)), (value(20), value(22))]
        observation = {"lineage": lineage, "seed": seed, "destination_index": 1, "episodes": episodes}
        candidates = [(direction, endpoint, (first[0], last[0] + 1))
                      for first, last in episodes for direction, endpoint in (("forward", first), ("backward", last))]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            legacy = _write_relay_artifacts({(lineage.token, 1): observation}, output_dir=root / "legacy",
                                            source_tile_index=0, generation=0)
            explicit = _write_relay_artifacts({(lineage.token, 1): {**observation, "endpoint_candidates": candidates}},
                                              output_dir=root / "explicit", source_tile_index=0, generation=0)
            self.assertEqual([r["seed_artifact_sha256"] for r in explicit],
                             [r["seed_artifact_sha256"] for r in legacy])
            self.assertEqual([Path(r["seed_artifact_path"]).name for r in explicit],
                             [Path(r["seed_artifact_path"]).name for r in legacy])

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
            union = read_union_array(manifest["union"])
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
        from XTA.lta_coverage import LtaCoverageBuilder
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
            coverage_spy = mock.Mock(wraps=LtaCoverageBuilder((4, 4)))
            with (
                mock.patch("XTA.lta_coverage.LtaCoverageBuilder", return_value=coverage_spy),
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
            self.assertEqual(coverage_spy.mark_observed.call_count, 2)
            self.assertEqual(coverage_spy.add_prediction.call_count, 2)
            for index, lineage in enumerate(lineages):
                self.assertEqual(coverage_spy.mark_observed.call_args_list[index], mock.call(
                    (lineage,), frame_start=0, frame_stop=1, prompt_frame=0, direction="backward"))
                call = coverage_spy.add_prediction.call_args_list[index]
                self.assertEqual(call.args[:2], (lineage, 0))
                np.testing.assert_array_equal(call.args[2], (first, second)[index])
            self.assertEqual(manifest["lineage_coverage"]["lineage_count"], 2)
            self.assertEqual(manifest["lineage_coverage"]["mask_count"], 2)
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
            union = read_union_array(manifest["union"])
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
            union = read_union_array(manifest["union"])
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
