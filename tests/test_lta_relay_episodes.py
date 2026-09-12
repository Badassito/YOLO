from __future__ import annotations

from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from XTA.lta_relay_episodes import (
    merge_relay_observations, merge_relay_observations_across_chains,
    read_relay_observations, write_relay_observations,
)
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_worker_adapter import _write_relay_artifacts


class RelayEpisodeTests(unittest.TestCase):
    def _observation(
        self, intervals, *, point=(1, 1), probability=0.9,
        object_id=3, visited=(0,), lineage=None, destination=1,
    ):
        if lineage is None:
            lineage = LtaLineageId("volume", "transverse", "runtime", "object")
        mask = np.zeros((4, 4), dtype=bool)
        mask[point] = True
        def endpoint(frame):
            return frame, np.packbits(mask.reshape(-1)), (4, 4), probability
        return {(lineage.token, destination): {
            "lineage": lineage, "destination_index": destination,
            "seed": SimpleNamespace(object_id=object_id, visited_tile_indices=visited),
            "episodes": [(endpoint(first), endpoint(last)) for first, last in intervals],
        }}

    def test_window_context_boundaries_do_not_invent_new_spatial_relays(self):
        pieces = (
            self._observation(((5, 11), (14, 34))),
            self._observation(((0, 5),)),
            self._observation(((34, 58),)),
        )
        reference = self._observation(((0, 11), (14, 58)))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = _write_relay_artifacts(
                reference, output_dir=root / "reference", source_tile_index=0, generation=0,
            )
            expected_hashes = [row["seed_artifact_sha256"] for row in expected]
            for ordinal, ordering in enumerate(permutations(pieces)):
                merged = {}
                for index, piece in enumerate(ordering):
                    receipt = write_relay_observations(root / f"piece-{ordinal}-{index}", piece)
                    merge_relay_observations(merged, read_relay_observations(receipt))
                episodes = next(iter(merged.values()))["episodes"]
                self.assertEqual([(first[0], last[0]) for first, last in episodes], [(0, 11), (14, 58)])
                actual = _write_relay_artifacts(
                    merged, output_dir=root / f"merged-{ordinal}",
                    source_tile_index=0, generation=0,
                )
                self.assertEqual([row["seed_artifact_sha256"] for row in actual], expected_hashes)
                self.assertEqual(
                    [(row["temporal_direction"], row["frame_index"]) for row in actual],
                    [("forward", 0), ("backward", 11), ("forward", 14), ("backward", 58)],
                )

    def test_duplicate_episode_endpoints_union_masks_and_keep_maximum_probability(self):
        first = self._observation(((4, 10),), point=(1, 1), probability=0.6)
        second = self._observation(((4, 10),), point=(2, 2), probability=0.95)
        merged = {}
        merge_relay_observations(merged, first)
        merge_relay_observations(merged, second)
        episode = next(iter(merged.values()))["episodes"][0]
        for frame, packed, shape, probability in episode:
            self.assertIn(frame, (4, 10))
            mask = np.unpackbits(packed).reshape(shape)
            self.assertEqual(int(mask.sum()), 2)
            self.assertTrue(mask[1, 1] and mask[2, 2])
            self.assertEqual(probability, 0.95)

    def test_empty_observations_roundtrip_and_digest_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = write_relay_observations(Path(folder), {})
            self.assertEqual(read_relay_observations(receipt), {})
            with Path(receipt["path"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(RuntimeError, "digest changed"):
                read_relay_observations(receipt)

    def test_cross_chain_merging_is_completion_order_invariant_including_artifacts(self):
        lineage = LtaLineageId("volume", "transverse", "runtime", "object", tile_config_id="s4_st2")
        pieces = (
            self._observation(((5, 11), (14, 34)), lineage=lineage, object_id=9, visited=(4, 0), probability=0.6),
            self._observation(((0, 5),), lineage=lineage, object_id=3, visited=(0, 2), probability=0.7),
            self._observation(((34, 58),), lineage=lineage, object_id=7, visited=(8, 0), probability=0.8),
            self._observation(((0, 11), (14, 58)), lineage=lineage, object_id=4, visited=(0, 6), point=(2, 2), probability=0.95),
        )
        expected_hashes = None
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for ordinal, ordering in enumerate(permutations(pieces)):
                merged = {}
                for piece in ordering:
                    merge_relay_observations_across_chains(merged, piece)
                record = merged[(lineage.token, 1)]
                self.assertIs(record["lineage"], lineage)
                self.assertEqual(record["seed"].object_id, 3)
                self.assertEqual(record["seed"].visited_tile_indices, (0, 2, 4, 6, 8))
                self.assertEqual([(first[0], last[0]) for first, last in record["episodes"]], [(0, 11), (14, 58)])
                for episode in record["episodes"]:
                    for _frame, packed, shape, probability in episode:
                        mask = np.unpackbits(packed).reshape(shape)
                        self.assertEqual(int(mask.sum()), 2)
                        self.assertTrue(mask[1, 1] and mask[2, 2])
                        self.assertEqual(probability, 0.95)
                artifacts = _write_relay_artifacts(
                    merged, output_dir=root / str(ordinal), source_tile_index=0, generation=2,
                )
                hashes = [artifact["seed_artifact_sha256"] for artifact in artifacts]
                if expected_hashes is None:
                    expected_hashes = hashes
                self.assertEqual(hashes, expected_hashes)
                self.assertEqual(
                    [(artifact["temporal_direction"], artifact["frame_index"]) for artifact in artifacts],
                    [("forward", 0), ("forward", 5), ("backward", 5),
                     ("backward", 11), ("forward", 14), ("forward", 34),
                     ("backward", 34), ("backward", 58)],
                )
                # A previously combined record must retain its original inner
                # boundaries when it becomes input to another accumulator.
                forwarded = {}
                merge_relay_observations_across_chains(forwarded, merged)
                forwarded_artifacts = _write_relay_artifacts(
                    forwarded, output_dir=root / f"forwarded-{ordinal}",
                    source_tile_index=0, generation=2,
                )
                self.assertEqual([item["seed_artifact_sha256"] for item in forwarded_artifacts], hashes)

    def test_cross_chain_merge_preserves_unmatched_lineages_and_destinations(self):
        first = LtaLineageId("volume", "transverse", "runtime", "first", tile_config_id="s4_st2")
        unmatched = LtaLineageId("volume", "transverse", "runtime", "unmatched", tile_config_id="s4_st2")
        merged = {}
        for piece in (
            self._observation(((2, 8),), lineage=first, object_id=8, visited=(0, 4)),
            self._observation(((2, 8),), lineage=unmatched, object_id=1, visited=(0, 7)),
            self._observation(((9, 12),), lineage=first, object_id=3, visited=(0, 2)),
            self._observation(((2, 8),), lineage=first, object_id=2, visited=(0, 3), destination=2),
        ):
            merge_relay_observations_across_chains(merged, piece)
        self.assertEqual(set(merged), {(first.token, 1), (unmatched.token, 1), (first.token, 2)})
        self.assertEqual([(a[0], b[0]) for a, b in merged[(first.token, 1)]["episodes"]], [(2, 12)])
        self.assertEqual(merged[(first.token, 1)]["seed"].object_id, 3)
        self.assertEqual(merged[(first.token, 1)]["seed"].visited_tile_indices, (0, 2, 4))
        self.assertIs(merged[(unmatched.token, 1)]["lineage"], unmatched)
        self.assertEqual(merged[(unmatched.token, 1)]["seed"].visited_tile_indices, (0, 7))
        self.assertEqual(merged[(first.token, 2)]["seed"].object_id, 2)

    def test_cross_chain_merge_preserves_novel_inner_candidates_at_their_own_frames(self):
        pieces = (
            self._observation(((4, 10),), point=(0, 0), probability=0.6),
            self._observation(((4, 12),), point=(1, 1), probability=0.8, object_id=1),
            self._observation(((6, 10),), point=(2, 2), probability=0.99, object_id=5),
        )
        for ordering in permutations(pieces):
            merged = {}
            for piece in ordering:
                merge_relay_observations_across_chains(merged, piece)
            first, last = next(iter(merged.values()))["episodes"][0]
            self.assertEqual((first[0], last[0]), (4, 12))
            first_mask = np.unpackbits(first[1]).reshape(first[2])
            last_mask = np.unpackbits(last[1]).reshape(last[2])
            self.assertEqual(int(first_mask.sum()), 2)
            self.assertTrue(first_mask[0, 0] and first_mask[1, 1])
            self.assertEqual(int(last_mask.sum()), 1)
            self.assertTrue(last_mask[1, 1])
            self.assertEqual((first[3], last[3]), (0.8, 0.8))
            candidates = {
                (direction, value[0]): (value, bounds)
                for direction, value, bounds in next(iter(merged.values()))["endpoint_candidates"]
            }
            self.assertEqual(set(candidates), {("forward", 4), ("forward", 6), ("backward", 10), ("backward", 12)})
            expected_points = {
                ("forward", 4): {(0, 0), (1, 1)},
                ("forward", 6): {(2, 2)},
                ("backward", 10): {(0, 0), (2, 2)},
                ("backward", 12): {(1, 1)},
            }
            expected_bounds = {
                ("forward", 4): (4, 13), ("forward", 6): (6, 11),
                ("backward", 10): (4, 11), ("backward", 12): (4, 13),
            }
            for event, (value, bounds) in candidates.items():
                self.assertEqual(bounds, expected_bounds[event])
                self.assertEqual(
                    {tuple(point) for point in np.argwhere(np.unpackbits(value[1]).reshape(value[2]))},
                    expected_points[event],
                )
            self.assertEqual(candidates["forward", 6][0][3], 0.99)
            self.assertEqual(candidates["backward", 10][0][3], 0.99)
            merge_relay_observations_across_chains(
                merged, self._observation(((0, 20),), point=(3, 3), probability=0.7),
            )
            retained = {(direction, value[0]): value for direction, value, _bounds in next(iter(merged.values()))["endpoint_candidates"]}
            for event in expected_points:
                self.assertTrue(np.array_equal(retained[event][1], candidates[event][0][1]))

    def test_cross_chain_single_frame_boundary_keeps_complementary_evidence(self):
        first = self._observation(((4, 4),), point=(0, 0), probability=0.6)
        second = self._observation(((4, 4),), point=(1, 1), probability=0.8, object_id=1)
        key = next(iter(first))
        # One frame can be supplied through different endpoint representations;
        # both must contribute to the single physical boundary.
        combined = dict(first[key])
        combined["episodes"] = [(first[key]["episodes"][0][0], second[key]["episodes"][0][1])]
        merged = {}
        merge_relay_observations_across_chains(merged, {key: combined})
        for endpoint in merged[key]["episodes"][0]:
            self.assertEqual(int(np.unpackbits(endpoint[1]).sum()), 2)
            self.assertEqual(endpoint[3], 0.8)
        for _direction, endpoint, bounds in merged[key]["endpoint_candidates"]:
            self.assertEqual(int(np.unpackbits(endpoint[1]).sum()), 2)
            self.assertEqual(endpoint[3], 0.8)
            self.assertEqual(bounds, (4, 5))

    def test_invalid_endpoint_candidates_do_not_mutate_existing_entry(self):
        for corruption in ("direction", "frame_boundary", "bounds", "geometry", "outside_episode"):
            with self.subTest(corruption=corruption):
                merged = {}
                merge_relay_observations_across_chains(merged, self._observation(((5, 10),)))
                key = next(iter(merged))
                original = merged[key]
                incoming = self._observation(((6, 9),), point=(2, 2))
                value = incoming[key]["episodes"][0][0]
                direction, bounds = "forward", (6, 10)
                if corruption == "direction":
                    direction = "both"
                elif corruption == "frame_boundary":
                    bounds = (5, 10)
                elif corruption == "bounds":
                    bounds = (6, 6)
                elif corruption == "geometry":
                    value = (6, np.array([128], dtype=np.uint8), (2, 4), 0.9)
                else:
                    bounds = (6, 20)
                incoming[key]["endpoint_candidates"] = [(direction, value, bounds)]
                with self.assertRaises(ValueError):
                    merge_relay_observations_across_chains(merged, incoming)
                self.assertIs(merged[key], original)

    def test_within_chain_merge_still_rejects_object_id_and_visited_history_changes(self):
        for changed in (
            self._observation(((8, 12),), object_id=9),
            self._observation(((8, 12),), visited=(0, 2)),
        ):
            merged = {}
            merge_relay_observations(merged, self._observation(((2, 8),)))
            key = next(iter(merged))
            original = merged[key]
            with self.assertRaisesRegex(ValueError, "source identity changed"):
                merge_relay_observations(merged, changed)
            self.assertIs(merged[key], original)

    def test_invalid_cross_chain_record_does_not_partially_mutate_existing_entry(self):
        for corruption in ("probability", "geometry", "source_index", "lineage"):
            with self.subTest(corruption=corruption):
                merged = {}
                merge_relay_observations_across_chains(merged, self._observation(((5, 10),)))
                key = next(iter(merged))
                original = merged[key]
                incoming = self._observation(((0, 5), (20, 25)), object_id=1, visited=(0, 7))
                record = incoming[key]
                first, last = record["episodes"][1]
                if corruption == "probability":
                    record["episodes"][1] = (first, (*last[:3], float("nan")))
                elif corruption == "geometry":
                    record["episodes"][1] = (
                        (20, np.array([128], dtype=np.uint8), (2, 4), 0.9),
                        (25, np.array([128], dtype=np.uint8), (2, 4), 0.9),
                    )
                elif corruption == "source_index":
                    record["seed"] = SimpleNamespace(object_id=-1, visited_tile_indices=(0,))
                else:
                    record["lineage"] = LtaLineageId("different", "transverse", "runtime", "object")
                with self.assertRaises(ValueError):
                    merge_relay_observations_across_chains(merged, incoming)
                self.assertIs(merged[key], original)
                self.assertEqual(merged[key]["seed"].object_id, 3)
                self.assertEqual(merged[key]["seed"].visited_tile_indices, (0,))
                self.assertEqual([(a[0], b[0]) for a, b in merged[key]["episodes"]], [(5, 10)])


if __name__ == "__main__":
    unittest.main()
