from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np

from XTA.lta_coverage import LtaCoverageBuilder, LtaCoverageLedger
from XTA.lta_execution import _plan_relay_generation
from XTA.lta_propagation import LtaMaskSeed, LtaSeedProvenance, read_seed_artifact, write_seed_artifact
from XTA.lta_rendering import reference_existing_physical_view_cache
from XTA.lta_runtime import LtaRuntimeViewPlan, LtaTileGridPlan
from XTA.lta_sam import plan_sam_sessions
from XTA.lta_scheduler import LtaSessionWork, LtaViewAffinityScheduler, LtaViewKey
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_tiles import plan_tile_grid


class LtaRelayAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frames = 8
        self.config_id = "s4_st2"
        grid = LtaTileGridPlan(
            self.config_id, 4, 2,
            plan_tile_grid(source_width=6, source_height=4, tile_size=4, tile_stride=2),
        )
        self.view = LtaRuntimeViewPlan(
            volume_id="volume", physical_view_id="transverse", runtime_view_id="transverse__tta_a0",
            tta_angle_deg=0.0, frame_count=self.frames, frame_height=4, frame_width=6,
            sessions=plan_sam_sessions("sequence", self.frames),
            tile_config_ids=(self.config_id,), tile_grids=(grid,),
        )
        cache_path = self.root / "cache.raw"
        np.zeros((self.frames, 4, 6), dtype=np.uint8).tofile(cache_path)
        self.cache = reference_existing_physical_view_cache(
            cache_path, shape=(self.frames, 4, 6), physical_view_id="transverse",
            source_identity="tiny-relay-admission-fixture",
        )
        self.lineage = self._lineage("covered")
        self.unmatched = self._lineage("unmatched")
        self.base = np.zeros((4, 4), dtype=bool)
        self.base[1, 0] = True
        self.novel = np.zeros((4, 4), dtype=bool)
        self.novel[1, 1] = True
        self.ledger = LtaCoverageLedger(self.root / "coverage.sqlite3", frame_count=self.frames)
        self.addCleanup(self.ledger.close)

    def _lineage(self, name):
        return LtaLineageId(
            "volume", "transverse", "transverse__tta_a0", name, tile_config_id=self.config_id,
        )

    def _ingest(self, name, *, lineage=None, mask=None, direction="forward", prompt=2):
        lineage = self.lineage if lineage is None else lineage
        mask = self.base if mask is None else mask
        builder = LtaCoverageBuilder((4, 4))
        builder.mark_observed(
            (lineage,), frame_start=0, frame_stop=self.frames,
            prompt_frame=prompt, direction=direction,
        )
        builder.add_prediction(lineage, 2, mask)
        receipt = builder.write(
            self.root / "coverage-packets" / name,
            work_id=name, tile_index=1, tile_config_id=self.config_id,
        )
        self.ledger.ingest(
            receipt, expected_work_id=name, tile_index=1, tile_config_id=self.config_id,
            frame_start=0, frame_stop=self.frames, expected_lineages=(lineage,),
        )
        return receipt

    def _relay(self, case, name="relay", *, lineage=None, mask=None, direction="forward", probability=0.85):
        lineage = self.lineage if lineage is None else lineage
        seed = LtaMaskSeed(
            lineage=lineage, frame_index=2, object_id=7,
            mask=self.base if mask is None else mask,
            provenance=LtaSeedProvenance.SPATIAL_RELAY,
            tracker_probability=probability, relay_generation=1,
            visited_tile_indices=(0, 1),
        )
        artifact = write_seed_artifact(self.root / case / "incoming" / f"{name}.npz", (seed,))
        return {
            "lineage": asdict(lineage), "source_tile_index": 0, "destination_tile_index": 1,
            "frame_index": 2, "temporal_direction": direction, "generation": 1,
            "seed_artifact_path": str(artifact.path), "seed_artifact_sha256": artifact.sha256,
        }, seed

    def _plan(self, records, *, case, ledger=None):
        # A fresh revision registry isolates coverage admission from the
        # independent exact-event/revision idempotency mechanism.
        initial = LtaSessionWork(
            work_id="initial", view=LtaViewKey("volume", "transverse"),
            runtime_view_id=self.view.runtime_view_id, session_index=0,
            frame_start=0, frame_stop=self.frames, plan_order=0,
            estimated_cost=128.0, projection_key=self.cache.identity_sha256,
            tile_index=0, tile_config_id=self.config_id,
        )
        scheduler = LtaViewAffinityScheduler((initial,), (0,), max_relay_generation=4)
        scheduler.mark_projection_ready(initial.view, device_id=0)
        claim = scheduler.claim(0)
        self.assertIsNotNone(claim)
        scheduler.complete(claim, "settled initial fixture")
        scheduler.drain_committable()
        audit = {}
        planned = _plan_relay_generation(
            records, generation=1, view_plan=self.view, cache_ref=self.cache,
            scheduler=scheduler, relay_mask_revisions={}, temp_root=self.root / case,
            conf=0.15, empty_frame_limit=30, first_plan_order=1,
            coverage_ledger=self.ledger if ledger is None else ledger,
            admission_audit=audit,
        )
        return planned, audit

    def _planned_seeds(self, planned):
        return tuple(
            seed for chain in planned
            for seed in read_seed_artifact(
                chain.payload["seed_artifact_path"],
                expected_sha256=chain.payload["seed_artifact_sha256"],
            )
        )

    def test_covered_same_lineage_and_forward_edge_consumes_artifact_without_replay(self):
        self._ingest("prior")
        record, seed = self._relay("covered")
        self.assertTrue(self.ledger.can_handoff(seed, tile_index=1, direction="forward"))
        planned, audit = self._plan((record,), case="covered")
        self.assertEqual(planned, ())
        self.assertEqual(audit["spatial_seed_candidates"], 1)
        self.assertEqual(audit["spatial_seed_handoffs"], 1)
        self.assertFalse(Path(record["seed_artifact_path"]).exists())

    def test_one_new_pixel_admits_the_entire_offered_seed(self):
        self._ingest("prior")
        offered = self.base | self.novel
        record, seed = self._relay("novel", mask=offered)
        self.assertFalse(self.ledger.can_handoff(seed, tile_index=1, direction="forward"))
        planned, audit = self._plan((record,), case="novel")
        accepted = self._planned_seeds(planned)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].lineage, self.lineage)
        np.testing.assert_array_equal(accepted[0].mask, offered)
        self.assertEqual(int(accepted[0].mask.sum()), 2)
        self.assertEqual(audit.get("spatial_seed_handoffs", 0), 0)
        self.assertFalse(Path(record["seed_artifact_path"]).exists())

    def test_complementary_arrivals_merge_before_coverage_and_retain_full_support(self):
        self._ingest("prior")
        covered, _ = self._relay("complementary", "covered-fragment", probability=0.7)
        novel, _ = self._relay("complementary", "novel-fragment", mask=self.novel, probability=0.95)
        planned, audit = self._plan((covered, novel), case="complementary")
        accepted = self._planned_seeds(planned)
        self.assertEqual(len(accepted), 1)
        np.testing.assert_array_equal(accepted[0].mask, self.base | self.novel)
        self.assertEqual(accepted[0].tracker_probability, 0.95)
        self.assertEqual(accepted[0].source_receipt["merged_relay_count"], 2)
        self.assertEqual(audit["spatial_seed_candidates"], 1)
        self.assertEqual(audit.get("spatial_seed_handoffs", 0), 0)
        self.assertTrue(all(not Path(record["seed_artifact_path"]).exists() for record in (covered, novel)))

    def test_novel_pixel_inside_existing_crop_is_not_mistaken_for_covered_support(self):
        recorded = np.zeros((4, 4), dtype=bool)
        recorded[0, 0] = recorded[2, 1] = True
        offered = recorded.copy()
        offered[1, 0] = True  # Adds support without enlarging the existing crop.
        self._ingest("prior-crop", mask=recorded)
        record, seed = self._relay("novel-inside-crop", mask=offered)
        self.assertFalse(self.ledger.can_handoff(seed, tile_index=1, direction="forward"))
        planned, _audit = self._plan((record,), case="novel-inside-crop")
        accepted = self._planned_seeds(planned)
        self.assertEqual(len(accepted), 1)
        np.testing.assert_array_equal(accepted[0].mask, offered)

    def test_identical_pixels_from_an_unmatched_lineage_are_admitted(self):
        self._ingest("prior")
        record, seed = self._relay("unmatched", lineage=self.unmatched)
        self.assertFalse(self.ledger.can_handoff(seed, tile_index=1, direction="forward"))
        planned, _audit = self._plan((record,), case="unmatched")
        accepted = self._planned_seeds(planned)
        self.assertEqual(tuple(seed.lineage for seed in accepted), (self.unmatched,))
        np.testing.assert_array_equal(accepted[0].mask, self.base)

    def test_backward_observation_does_not_block_forward_replay(self):
        self._ingest("backward-prior", direction="backward", prompt=6)
        record, seed = self._relay("forward-after-backward")
        self.assertTrue(self.ledger.can_handoff(seed, tile_index=1, direction="backward"))
        self.assertFalse(self.ledger.can_handoff(seed, tile_index=1, direction="forward"))
        planned, audit = self._plan((record,), case="forward-after-backward")
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].payload["windows"][0]["direction"], "forward")
        self.assertEqual(audit.get("spatial_seed_handoffs", 0), 0)

    def test_prior_snapshot_decision_is_unchanged_by_new_main_ledger_coverage(self):
        self._ingest("prior")
        offered = self.base | self.novel
        with self.ledger.snapshot(self.root / "prior.sqlite3") as snapshot:
            before, candidate = self._relay("snapshot-before", mask=offered)
            self.assertFalse(snapshot.can_handoff(candidate, tile_index=1, direction="forward"))
            first, _ = self._plan((before,), case="snapshot-before", ledger=snapshot)
            self.assertEqual(len(first), 1)
            self._ingest("later-complement", mask=self.novel)
            self.assertTrue(self.ledger.can_handoff(candidate, tile_index=1, direction="forward"))
            self.assertFalse(snapshot.can_handoff(candidate, tile_index=1, direction="forward"))
            after, _ = self._relay("snapshot-after", mask=offered)
            second, _ = self._plan((after,), case="snapshot-after", ledger=snapshot)
            self.assertEqual(len(second), 1)
            np.testing.assert_array_equal(self._planned_seeds(second)[0].mask, offered)
            current, _ = self._relay("main-after", mask=offered)
            now, audit = self._plan((current,), case="main-after", ledger=self.ledger)
            self.assertEqual(now, ())
            self.assertEqual(audit["spatial_seed_handoffs"], 1)

    def test_mixed_prompt_group_preserves_unmatched_lineage_after_covered_handoff(self):
        self._ingest("prior")
        covered, _ = self._relay("mixed", "covered")
        unmatched, _ = self._relay("mixed", "unmatched", lineage=self.unmatched)
        planned, audit = self._plan((covered, unmatched), case="mixed")
        accepted = self._planned_seeds(planned)
        self.assertEqual(tuple(seed.lineage for seed in accepted), (self.unmatched,))
        np.testing.assert_array_equal(accepted[0].mask, self.base)
        self.assertEqual(audit["spatial_seed_candidates"], 2)
        self.assertEqual(audit["spatial_seed_handoffs"], 1)
        self.assertTrue(all(not Path(record["seed_artifact_path"]).exists() for record in (covered, unmatched)))

    def test_invalid_artifact_is_rejected_before_covered_candidates_are_dropped(self):
        self._ingest("prior")
        valid, candidate = self._relay("invalid", "valid")
        invalid, _ = self._relay("invalid", "corrupt")
        self.assertTrue(self.ledger.can_handoff(candidate, tile_index=1, direction="forward"))
        checks_before = self.ledger.stats()["handoff_checks"]
        with Path(invalid["seed_artifact_path"]).open("ab") as stream:
            stream.write(b"tampered after receipt")
        with self.assertRaisesRegex(RuntimeError, "relay-seed artifact digest changed"):
            self._plan((valid, invalid), case="invalid")
        self.assertEqual(self.ledger.stats()["handoff_checks"], checks_before)
        self.assertTrue(Path(valid["seed_artifact_path"]).is_file())
        self.assertTrue(Path(invalid["seed_artifact_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
