from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from XTA.lta_coverage import COVERAGE_SCHEMA, LtaCoverageBuilder, LtaCoverageLedger
from XTA.lta_tile_tracking import LtaLineageId


def _lineage(name="object", config="s32_st24"):
    return LtaLineageId("volume", "transverse", "runtime", name, tile_config_id=config)


def _mask(*points, shape=(32, 32)):
    mask = np.zeros(shape, dtype=bool)
    for y, x in points:
        mask[y, x] = True
    return mask


def _seed(lineage, frame, mask):
    return SimpleNamespace(lineage=lineage, frame_index=frame, mask=mask)


def _ingest(ledger, packet, *, start=0, stop=6, lineages=None):
    return ledger.ingest(
        packet, expected_work_id=packet["work_id"], tile_index=packet["tile_index"],
        tile_config_id=packet["tile_config_id"], frame_start=start, frame_stop=stop,
        expected_lineages=lineages,
    )


def _packet(root, work, *, lineage=None, tile=0, shape=(32, 32),
            observations=((0, 6, 0, "forward"),), predictions=()):
    lineage = _lineage() if lineage is None else lineage
    builder = LtaCoverageBuilder(shape)
    for start, stop, prompt, direction in observations:
        builder.mark_observed((lineage,), frame_start=start, frame_stop=stop,
                              prompt_frame=prompt, direction=direction)
    for frame, mask in predictions:
        builder.add_prediction(lineage, frame, mask)
    return builder.write(root / work, work_id=work, tile_index=tile, tile_config_id=lineage.tile_config_id)


class LtaCoverageTests(unittest.TestCase):
    def test_both_directions_split_at_prompt_and_preserve_unobserved_transitions(self):
        lineage = _lineage()
        mask = _mask((4, 5), (4, 6))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packet = _packet(root, "both", observations=((0, 6, 3, "both"),),
                             predictions=tuple((frame, mask) for frame in range(6)))
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=6) as ledger:
                _ingest(ledger, packet, lineages=(lineage,))
                self.assertTrue(ledger.can_handoff(_seed(lineage, 2, mask), tile_index=0, direction="backward"))
                self.assertFalse(ledger.can_handoff(_seed(lineage, 2, mask), tile_index=0, direction="forward"))
                self.assertTrue(ledger.can_handoff(_seed(lineage, 3, mask), tile_index=0, direction="forward"))
                self.assertTrue(ledger.can_handoff(_seed(lineage, 3, mask), tile_index=0, direction="backward"))
                self.assertFalse(ledger.can_handoff(_seed(lineage, 4, mask), tile_index=0, direction="backward"))
                self.assertTrue(ledger.can_handoff(_seed(lineage, 0, mask), tile_index=0, direction="backward"))
                self.assertTrue(ledger.can_handoff(_seed(lineage, 5, mask), tile_index=0, direction="forward"))
                self.assertEqual(ledger.stats()["covered_directional_transitions"], 5)

    def test_current_window_endpoint_cannot_suppress_the_unobserved_next_edge(self):
        lineage = _lineage()
        mask = _mask((2, 3))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _packet(root, "first", observations=((0, 3, 0, "forward"),), predictions=((2, mask),))
            next_packet = _packet(root, "next", observations=((2, 5, 2, "forward"),), predictions=((2, mask),))
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=6) as ledger:
                _ingest(ledger, first, stop=3)
                self.assertFalse(ledger.can_handoff(_seed(lineage, 2, mask), tile_index=0, direction="forward"))
                _ingest(ledger, next_packet, start=2, stop=5)
                self.assertTrue(ledger.can_handoff(_seed(lineage, 2, mask), tile_index=0, direction="forward"))
                self.assertEqual(ledger.stats()["directional_ranges"], 1)

    def test_adjacent_observations_do_not_invent_a_missing_transition(self):
        lineage = _lineage()
        mask = _mask((3, 3))
        for direction in ("forward", "backward"):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                observations = ((0, 2, 0, direction), (2, 4, 2, direction)) if direction == "forward" else ((0, 2, 1, direction), (2, 4, 3, direction))
                packet = _packet(root, "separate", observations=observations,
                                 predictions=((1, mask), (2, mask)))
                bridge = _packet(root, "bridge", observations=((1, 3, 1 if direction == "forward" else 2, direction),))
                with LtaCoverageLedger(root / "coverage.sqlite", frame_count=6) as ledger:
                    _ingest(ledger, packet)
                    frame = 1 if direction == "forward" else 2
                    self.assertFalse(ledger.can_handoff(_seed(lineage, frame, mask), tile_index=0, direction=direction))
                    _ingest(ledger, bridge)
                    self.assertTrue(ledger.can_handoff(_seed(lineage, frame, mask), tile_index=0, direction=direction))

    def test_novel_pixels_disjoint_objects_other_tiles_and_unmatched_ids_remain_admitted(self):
        lineage = _lineage()
        first = _mask((4, 4), (4, 5))
        novel = first | _mask((5, 5))
        disjoint = _mask((20, 20))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packet = _packet(root, "first", predictions=((2, first),))
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=6) as ledger:
                _ingest(ledger, packet)
                self.assertTrue(ledger.can_handoff(_seed(lineage, 2, _mask((4, 5))), tile_index=0, direction="forward"))
                for candidate, identity, tile in (
                    (novel, lineage, 0), (disjoint, lineage, 0),
                    (first, _lineage("another-object"), 0),
                    (first, _lineage(config="another-grid"), 0),
                    (first, replace(lineage, runtime_view_id="another-view"), 0),
                    (first, lineage, 1),
                ):
                    self.assertFalse(ledger.can_handoff(_seed(identity, 2, candidate), tile_index=tile, direction="forward"))
                second = _packet(root, "second", predictions=((2, disjoint), (2, _mask((5, 5)))))
                _ingest(ledger, second)
                self.assertTrue(ledger.can_handoff(_seed(lineage, 2, novel | disjoint), tile_index=0, direction="forward"))
                self.assertEqual(ledger.stats()["masked_frames"], 1)
                self.assertEqual(ledger.stats()["foreground_pixels"], 4)

    def test_snapshot_is_frozen_independent_and_read_only(self):
        lineage = _lineage()
        first, novel = _mask((1, 1)), _mask((2, 2))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packet = _packet(root, "old", predictions=((2, first),))
            later = _packet(root, "later", predictions=((2, novel),))
            with LtaCoverageLedger(root / "main.sqlite", frame_count=6) as ledger:
                _ingest(ledger, packet)
                with ledger.snapshot(root / "frozen.sqlite") as snapshot:
                    before = snapshot.stats()
                    _ingest(ledger, later)
                    self.assertTrue(ledger.can_handoff(_seed(lineage, 2, novel), tile_index=0, direction="forward"))
                    self.assertFalse(snapshot.can_handoff(_seed(lineage, 2, novel), tile_index=0, direction="forward"))
                    self.assertTrue(snapshot.can_handoff(_seed(lineage, 2, first), tile_index=0, direction="forward"))
                    self.assertEqual(snapshot.stats()["packets"], before["packets"])
                    self.assertEqual(snapshot.stats()["foreground_pixels"], 1)
                    self.assertTrue(snapshot.stats()["read_only"])
                    with self.assertRaisesRegex(RuntimeError, "read-only"):
                        _ingest(snapshot, later)
                self.assertTrue(snapshot.closed)
                self.assertTrue(ledger.can_handoff(_seed(lineage, 2, first | novel), tile_index=0, direction="forward"))
                with self.assertRaises(FileExistsError):
                    ledger.snapshot(root / "frozen.sqlite")

    def test_corrupt_late_mask_is_rejected_before_any_database_write(self):
        mask = _mask((1, 1))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packet = _packet(root, "broken", predictions=((0, mask), (1, mask)))
            path = Path(packet["path"])
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            arrays["mask_00000001"][:] = 0
            with path.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            # Simulate a producer bug with a matching outer artifact hash. The
            # second mask must still fail validation before mask zero is stored.
            packet["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            packet["file_size_bytes"] = path.stat().st_size
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=6) as ledger:
                statements = []
                ledger._db.set_trace_callback(statements.append)
                with self.assertRaisesRegex(ValueError, "foreground count"):
                    _ingest(ledger, packet)
                self.assertFalse(any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for sql in statements))
                self.assertEqual(ledger.stats()["packets"], 0)
                self.assertEqual(ledger.stats()["masked_frames"], 0)

    def test_receipt_identity_range_and_expected_lineage_validation_precede_ingestion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packet = _packet(root, "valid", predictions=((0, _mask((1, 1))),))
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=6) as ledger:
                for changes in (
                    {"expected_work_id": "wrong"}, {"tile_index": 1},
                    {"tile_config_id": "wrong"}, {"frame_start": 1},
                    {"expected_lineages": (_lineage("unmatched"),)},
                ):
                    with self.subTest(changes=changes), self.assertRaises((ValueError, RuntimeError)):
                        ledger.ingest(packet, **{
                            "expected_work_id": "valid", "tile_index": 0,
                            "tile_config_id": "s32_st24", "frame_start": 0,
                            "frame_stop": 6, **changes,
                        })
                with Path(packet["path"]).open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaises((ValueError, RuntimeError)):
                    _ingest(ledger, packet)
                self.assertEqual(ledger.stats()["packets"], 0)

    def test_cropped_storage_and_database_page_cache_remain_bounded_and_close_cleanly(self):
        lineage = _lineage(config="s1008_st756")
        builder = LtaCoverageBuilder((1008, 1008))
        builder.mark_observed((lineage,), frame_start=0, frame_stop=30, prompt_frame=0, direction="forward")
        for frame in range(30):
            builder.add_prediction(lineage, frame, _mask((900, 900 + frame), shape=(1008, 1008)))
        builder.add_prediction(lineage, 0, _mask(shape=(1008, 1008)))
        self.assertEqual(builder.stats()["stored_mask_bytes"], 30)
        self.assertEqual(builder.stats()["empty_predictions"], 1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packet = builder.write(root, work_id="cropped", tile_index=0, tile_config_id=lineage.tile_config_id)
            self.assertEqual(packet["schema"], COVERAGE_SCHEMA)
            self.assertLess(packet["file_size_bytes"], 20000)
            with np.load(packet["path"], allow_pickle=False) as archive:
                self.assertTrue(all(archive[name].dtype.kind in "iu" for name in archive.files))
            path = root / "ledger.sqlite"
            ledger = LtaCoverageLedger(path, frame_count=30)
            _ingest(ledger, packet, stop=30)
            self.assertEqual(ledger.stats()["stored_bytes"], 30)
            self.assertEqual(ledger.stats()["masked_frames"], 30)
            self.assertEqual(ledger.stats()["page_cache_limit_bytes"], 4 * 1024 * 1024)
            ledger.close()
            ledger.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                ledger.stats()
            with LtaCoverageLedger(path, frame_count=30) as reopened:
                self.assertEqual(reopened.stats()["packets"], 1)
            with self.assertRaisesRegex(ValueError, "frame count"):
                LtaCoverageLedger(path, frame_count=31)

    def test_empty_candidates_bad_dimensions_and_changed_tile_geometry_are_rejected(self):
        lineage = _lineage()
        builder = LtaCoverageBuilder((32, 32))
        with self.assertRaises(ValueError):
            builder.add_prediction(lineage, 0, np.ones((1, 32), dtype=bool))
        with self.assertRaises(ValueError):
            builder.add_prediction(lineage, 0, np.full((32, 32), 2, dtype=np.uint8))
        with self.assertRaises(ValueError):
            builder.mark_observed((lineage,), frame_start=0, frame_stop=2, prompt_frame=2, direction="forward")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _packet(root, "first", predictions=((1, _mask((1, 1))),))
            changed = _packet(root, "changed", shape=(64, 64), predictions=((1, _mask((1, 1), shape=(64, 64))),))
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=6) as ledger:
                _ingest(ledger, first)
                self.assertFalse(ledger.can_handoff(_seed(lineage, 1, _mask()), tile_index=0, direction="forward"))
                with self.assertRaisesRegex(ValueError, "shape"):
                    ledger.can_handoff(_seed(lineage, 1, np.ones((64, 64), dtype=bool)), tile_index=0, direction="forward")
                with self.assertRaisesRegex(ValueError, "geometry"):
                    _ingest(ledger, changed)
                self.assertEqual(ledger.stats()["packets"], 1)
                with self.assertRaisesRegex(ValueError, "forward or backward"):
                    ledger.can_handoff(_seed(lineage, 1, _mask((1, 1))), tile_index=0, direction="both")

    def test_packet_reingestion_is_idempotent_and_conflicting_work_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            packet = _packet(root, "same", predictions=((1, _mask((1, 1))),))
            builder = LtaCoverageBuilder((32, 32))
            builder.mark_observed((_lineage(),), frame_start=0, frame_stop=6, prompt_frame=0, direction="forward")
            builder.add_prediction(_lineage(), 1, _mask((2, 2)))
            conflict = builder.write(root / "conflict", work_id="same", tile_index=0, tile_config_id="s32_st24")
            with LtaCoverageLedger(root / "coverage.sqlite", frame_count=6) as ledger:
                self.assertFalse(_ingest(ledger, packet)["duplicate"])
                self.assertTrue(_ingest(ledger, packet)["duplicate"])
                with self.assertRaisesRegex(ValueError, "changed its packet digest"):
                    _ingest(ledger, conflict)
                self.assertEqual(ledger.stats()["packets"], 1)
                self.assertEqual(ledger.stats()["foreground_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
