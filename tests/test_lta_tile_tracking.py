from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from XTA.lta_tile_tracking import (
    LtaLineageId,
    RelayLedger,
    SpatialRelay,
    cross_tile_overlap,
    merge_inbound_relays,
    plan_spatial_relays,
    rebase_tile_mask,
)
from XTA.lta_tiles import eight_neighbor_graph, plan_tile_grid
from XTA.lta_tracklets import Tracklet, TrackletFrame, TrackletKey


def _mask(*points: tuple[int, int], size: int = 6) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.bool_)
    for row, column in points:
        mask[row, column] = True
    return mask


def _tracklet(
    tile_index: int,
    frames: tuple[tuple[int, np.ndarray, float], ...],
    *,
    object_id: int = 0,
) -> Tracklet:
    return Tracklet(
        tile_index=tile_index,
        anchor_frame=frames[0][0],
        local_object_id=object_id,
        frames=tuple(
            TrackletFrame(
                frame_index=frame_index,
                mask=mask,
                tracker_probability=probability,
            )
            for frame_index, mask, probability in frames
        ),
    )


class LtaTileTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lineage = LtaLineageId(
            volume_id="volume",
            physical_view_id="transverse",
            runtime_view_id="transverse::angle-0",
            lineage_id="object-7",
        )

    def test_eight_neighbor_graph_and_cardinal_diagonal_rebasing(self) -> None:
        tiles = plan_tile_grid(
            source_width=14,
            source_height=14,
            tile_size=6,
            tile_stride=4,
        )
        graph = eight_neighbor_graph(tiles)
        center_index = 4
        center = tiles[center_index]
        edges = graph[center_index]

        self.assertEqual(len(edges), 8)
        self.assertEqual(
            {edge.direction for edge in edges},
            {
                "northwest",
                "north",
                "northeast",
                "west",
                "east",
                "southwest",
                "south",
                "southeast",
            },
        )
        for edge in edges:
            reverse = next(
                candidate
                for candidate in graph[edge.destination_index]
                if candidate.destination_index == center_index
            )
            self.assertEqual(reverse.overlap_xyxy, edge.overlap_xyxy)
            self.assertEqual(reverse.row_delta, -edge.row_delta)
            self.assertEqual(reverse.column_delta, -edge.column_delta)

            global_x, global_y = edge.overlap_xyxy[:2]
            source_mask = _mask(
                (global_y - center.top, global_x - center.left),
                size=center.size,
            )
            destination = tiles[edge.destination_index]
            translated = rebase_tile_mask(source_mask, center, destination)
            self.assertEqual(int(np.count_nonzero(translated)), 1, edge.direction)
            self.assertTrue(
                translated[
                    global_y - destination.top,
                    global_x - destination.left,
                ],
                edge.direction,
            )

    def test_edge_pinned_grid_uses_actual_overlap_rectangles(self) -> None:
        tiles = plan_tile_grid(
            source_width=13,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        graph = eight_neighbor_graph(tiles)

        self.assertEqual([tile.left for tile in tiles], [0, 4, 7])
        self.assertEqual(
            [(edge.destination_index, edge.overlap_xyxy) for edge in graph[0]],
            [(1, (4, 0, 6, 6))],
        )
        self.assertEqual(
            [(edge.destination_index, edge.overlap_xyxy) for edge in graph[1]],
            [(0, (4, 0, 6, 6)), (2, (7, 0, 10, 6))],
        )
        self.assertEqual(
            [(edge.destination_index, edge.overlap_xyxy) for edge in graph[2]],
            [(1, (7, 0, 10, 6))],
        )

    def test_touching_tiles_relay_across_cardinal_and_diagonal_ports(self) -> None:
        tiles = plan_tile_grid(
            source_width=12,
            source_height=12,
            tile_size=6,
            tile_stride=6,
        )
        graph = eight_neighbor_graph(tiles)
        self.assertEqual(
            {edge.direction for edge in graph[0]},
            {"east", "south", "southeast"},
        )
        tracklet = _tracklet(
            0,
            ((3, _mask((2, 5), (5, 3), (5, 5)), 0.9),),
        )

        forward = {
            relay.neighbor_direction: relay
            for relay in plan_spatial_relays(self.lineage, tracklet, tiles)
            if relay.temporal_direction == "forward"
        }

        self.assertTrue(forward["east"].destination_mask[2, 0])
        self.assertTrue(forward["south"].destination_mask[0, 3])
        self.assertTrue(forward["southeast"].destination_mask[0, 0])
        right = _tracklet(
            1,
            ((3, _mask((2, 0)), 0.9),),
        )
        self.assertEqual(cross_tile_overlap(tracklet, right, tiles).iou, 0.5)

    def test_first_entry_relays_backward_and_last_exit_relays_forward(self) -> None:
        tiles = plan_tile_grid(
            source_width=10,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        tracklet = _tracklet(
            0,
            (
                (1, _mask((2, 1)), 0.99),
                (2, _mask((1, 4)), 0.75),
                (4, _mask((3, 5)), 0.90),
            ),
        )

        relays = plan_spatial_relays(self.lineage, tracklet, tiles)

        self.assertEqual(
            [(relay.temporal_direction, relay.frame_index) for relay in relays],
            [("backward", 2), ("forward", 4)],
        )
        backward, forward = relays
        self.assertEqual(backward.neighbor_direction, "east")
        self.assertEqual(backward.overlap_xyxy, (4, 0, 6, 6))
        self.assertEqual(backward.tile_path, (0, 1))
        self.assertEqual(backward.tracker_probability, 0.75)
        self.assertTrue(backward.destination_mask[1, 0])
        self.assertEqual(int(np.count_nonzero(backward.destination_mask)), 1)
        self.assertEqual(forward.tracker_probability, 0.90)
        self.assertTrue(forward.destination_mask[3, 1])
        self.assertEqual(int(np.count_nonzero(forward.destination_mask)), 1)

    def test_two_neighbor_arrivals_merge_into_one_unseeded_destination_seed(self) -> None:
        tiles = plan_tile_grid(
            source_width=14,
            source_height=14,
            tile_size=6,
            tile_stride=4,
        )
        # Row-major indexes: north=1, west=3, center=4.  Each source has one
        # pixel that reaches only the center, and no center tracklet is needed.
        north = _tracklet(1, ((12, _mask((5, 3)), 0.80),), object_id=1)
        west = _tracklet(3, ((12, _mask((3, 5)), 0.95),), object_id=2)
        arrivals = tuple(
            relay
            for relay in (
                *plan_spatial_relays(self.lineage, north, tiles),
                *plan_spatial_relays(self.lineage, west, tiles),
            )
            if relay.destination_tile_index == 4
            and relay.temporal_direction == "forward"
        )
        self.assertEqual(len(arrivals), 2)

        seeds = merge_inbound_relays(arrivals)

        self.assertEqual(len(seeds), 1)
        seed = seeds[0]
        self.assertEqual(seed.destination_tile_index, 4)
        self.assertEqual(seed.frame_index, 12)
        self.assertEqual(seed.temporal_direction, "forward")
        self.assertEqual(seed.source_tile_indices, (1, 3))
        self.assertEqual(seed.generation, 1)
        self.assertEqual(seed.visited_tile_indices, (1, 3, 4))
        self.assertEqual(seed.tracker_probability, 0.95)
        self.assertEqual(len(seed.relay_keys), 2)
        self.assertEqual(int(np.count_nonzero(seed.mask)), 2)
        self.assertTrue(seed.mask[1, 3])
        self.assertTrue(seed.mask[3, 1])
        self.assertFalse(seed.mask.flags.writeable)

    def test_relay_path_suppresses_cycles_and_ping_pong(self) -> None:
        tiles = plan_tile_grid(
            source_width=14,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        middle = _tracklet(
            1,
            ((20, _mask((2, 0), (2, 5)), 0.90),),
        )

        outbound = plan_spatial_relays(
            self.lineage,
            middle,
            tiles,
            generation=1,
            tile_path=(0, 1),
        )

        self.assertEqual({relay.destination_tile_index for relay in outbound}, {2})
        self.assertEqual({relay.tile_path for relay in outbound}, {(0, 1, 2)})
        self.assertEqual(
            {relay.temporal_direction for relay in outbound},
            {"forward", "backward"},
        )
        forward = next(
            relay for relay in outbound if relay.temporal_direction == "forward"
        )
        at_edge = _tracklet(
            2,
            ((forward.frame_index, forward.destination_mask, 0.90),),
        )
        self.assertEqual(
            plan_spatial_relays(
                self.lineage,
                at_edge,
                tiles,
                generation=2,
                tile_path=forward.tile_path,
            ),
            (),
        )

    def test_merged_seed_preserves_ancestry_for_the_next_relay_wave(self) -> None:
        tiles = plan_tile_grid(
            source_width=14,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        source = _tracklet(0, ((5, _mask((2, 5)), 0.9),))
        first_wave = tuple(
            relay
            for relay in plan_spatial_relays(self.lineage, source, tiles)
            if relay.destination_tile_index == 1
            and relay.temporal_direction == "forward"
        )
        seed = merge_inbound_relays(first_wave)[0]
        self.assertEqual(seed.visited_tile_indices, (0, 1))
        self.assertEqual(seed.generation, 1)
        propagated = _tracklet(1, ((6, _mask((2, 0), (2, 5)), 0.9),))

        second_wave = plan_spatial_relays(
            self.lineage,
            propagated,
            tiles,
            generation=seed.generation,
            visited_tile_indices=seed.visited_tile_indices,
        )

        self.assertEqual({relay.destination_tile_index for relay in second_wave}, {2})
        self.assertTrue(
            all(relay.visited_tile_indices == (0, 1, 2) for relay in second_wave)
        )

    def test_relay_ledger_admits_each_idempotency_key_once(self) -> None:
        relay = SpatialRelay(
            lineage=self.lineage,
            source_key=TrackletKey(0, 10, 4),
            source_tile_index=0,
            destination_tile_index=1,
            frame_index=12,
            temporal_direction="forward",
            neighbor_direction="east",
            overlap_xyxy=(4, 0, 6, 6),
            destination_mask=_mask((2, 0)),
            generation=0,
            tile_path=(0, 1),
        )
        retry_with_different_storage = replace(
            relay,
            destination_mask=_mask((3, 1)),
        )
        next_generation = replace(relay, generation=1)
        ledger = RelayLedger()

        self.assertTrue(ledger.admit(relay))
        self.assertIn(retry_with_different_storage, ledger)
        self.assertFalse(ledger.admit(retry_with_different_storage))
        self.assertTrue(ledger.admit(next_generation))
        self.assertEqual(len(ledger), 2)

    def test_cross_tile_overlap_aligns_masks_in_global_coordinates(self) -> None:
        tiles = plan_tile_grid(
            source_width=10,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        left = _tracklet(
            0,
            (
                (10, _mask((1, 4), (1, 5), (0, 0)), 0.90),
                (11, _mask((3, 5), (4, 0)), 0.90),
            ),
        )
        right = _tracklet(
            1,
            (
                (10, _mask((1, 0), (2, 0), (5, 5)), 0.90),
                (11, _mask((3, 1), (0, 5)), 0.90),
            ),
        )

        metrics = cross_tile_overlap(left, right, tiles)

        self.assertEqual(metrics.shared_frame_indices, (10, 11))
        self.assertEqual(metrics.coactive_frame_count, 2)
        self.assertEqual(metrics.intersection_pixels, 2)
        self.assertEqual(metrics.union_pixels, 4)
        self.assertEqual(metrics.iou, 0.5)

    def test_cross_tile_overlap_rejects_non_neighbors(self) -> None:
        tiles = plan_tile_grid(
            source_width=14,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        left = _tracklet(0, ((10, _mask((2, 5)), 0.90),))
        right = _tracklet(2, ((10, _mask((2, 0)), 0.90),))

        with self.assertRaisesRegex(ValueError, "eight-neighbor"):
            cross_tile_overlap(left, right, tiles)

    def test_cross_tile_overlap_does_not_treat_absent_evidence_as_perfect(self) -> None:
        tiles = plan_tile_grid(
            source_width=10,
            source_height=6,
            tile_size=6,
            tile_stride=4,
        )
        left = _tracklet(0, ((10, _mask((2, 1)), 0.9),))
        right = _tracklet(1, ((11, _mask((2, 4)), 0.9),))

        metrics = cross_tile_overlap(left, right, tiles)

        self.assertEqual(metrics.shared_frame_indices, ())
        self.assertEqual(metrics.union_pixels, 0)
        self.assertEqual(metrics.iou, 0.0)


if __name__ == "__main__":
    unittest.main()
