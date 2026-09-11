from __future__ import annotations

from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from XTA.lta_relay_episodes import (
    merge_relay_observations, read_relay_observations, write_relay_observations,
)
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_worker_adapter import _write_relay_artifacts


class RelayEpisodeTests(unittest.TestCase):
    def _observation(self, intervals, *, point=(1, 1), probability=0.9):
        lineage = LtaLineageId("volume", "transverse", "runtime", "object")
        mask = np.zeros((4, 4), dtype=bool)
        mask[point] = True
        def endpoint(frame):
            return frame, np.packbits(mask.reshape(-1)), (4, 4), probability
        return {(lineage.token, 1): {
            "lineage": lineage, "destination_index": 1,
            "seed": SimpleNamespace(object_id=3, visited_tile_indices=(0,)),
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


if __name__ == "__main__":
    unittest.main()
