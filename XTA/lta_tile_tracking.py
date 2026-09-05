"""Cross-tile LTA relay planning in global view coordinates.

Spatial relay is deliberately separate from temporal anchor reconciliation.
Objects can leave a tile in forward time or be recovered as entrants by a
backward relay.  Every operation uses actual overlap rectangles from the
eight-neighbor graph, so edge-pinned grids do not rely on tile-index math.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .lta_tiles import TileNeighbor, TilePlan, eight_neighbor_graph
from .lta_tracklets import Tracklet, TrackletFrame, TrackletKey


_NEIGHBOR_DIRECTIONS = frozenset(
    {
        "northwest",
        "north",
        "northeast",
        "west",
        "east",
        "southwest",
        "south",
        "southeast",
    }
)


@dataclass(frozen=True, order=True)
class LtaLineageId:
    """Globally scoped identity for one LTA lineage."""

    volume_id: str
    physical_view_id: str
    runtime_view_id: str
    lineage_id: str
    tile_config_id: str = "fullframe"

    def __post_init__(self) -> None:
        for name in (
            "volume_id",
            "physical_view_id",
            "runtime_view_id",
            "lineage_id",
            "tile_config_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)

    @property
    def token(self) -> str:
        return "::".join(
            (
                self.volume_id,
                self.physical_view_id,
                self.runtime_view_id,
                self.tile_config_id,
                self.lineage_id,
            )
        )


@dataclass(frozen=True)
class SpatialRelay:
    """One immutable seed translated from a source tile into a neighbor."""

    lineage: LtaLineageId
    source_key: TrackletKey
    source_tile_index: int
    destination_tile_index: int
    frame_index: int
    temporal_direction: str
    neighbor_direction: str
    overlap_xyxy: tuple[int, int, int, int]
    destination_mask: Any = field(repr=False, compare=False)
    tracker_probability: float = 1.0
    generation: int = 0
    tile_path: tuple[int, ...] = ()
    visited_tile_indices: tuple[int, ...] = ()
    provenance_kind: str = "spatial_relay"

    def __post_init__(self) -> None:
        import numpy as np

        if self.temporal_direction not in {"forward", "backward"}:
            raise ValueError("temporal_direction must be 'forward' or 'backward'")
        if self.neighbor_direction not in _NEIGHBOR_DIRECTIONS:
            raise ValueError("neighbor_direction must name one of the eight neighboring ports")
        if self.provenance_kind != "spatial_relay":
            raise ValueError("a SpatialRelay must use spatial_relay provenance")
        probability = float(self.tracker_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("tracker_probability must be finite and in [0,1]")
        mask = np.ascontiguousarray(np.asarray(self.destination_mask), dtype=np.bool_).copy()
        if mask.ndim != 2 or not bool(np.any(mask)):
            raise ValueError("a spatial relay requires a nonempty two-dimensional mask")
        mask.setflags(write=False)
        source_index = int(self.source_tile_index)
        destination_index = int(self.destination_tile_index)
        if int(self.source_key.tile_index) != source_index:
            raise ValueError("source_key tile does not match source_tile_index")
        path = tuple(int(value) for value in self.tile_path)
        if not path:
            path = (source_index, destination_index)
        if (
            len(path) < 2
            or path[-2] != source_index
            or path[-1] != destination_index
        ):
            raise ValueError(
                "tile_path must end with the source-to-destination relay edge"
            )
        if len(path) != len(set(path)):
            raise ValueError("tile_path cannot revisit a tile")
        visited = set(int(value) for value in self.visited_tile_indices)
        if not visited:
            visited.update(path)
        if not set(path).issubset(visited):
            raise ValueError("visited_tile_indices must contain the complete tile_path")
        object.__setattr__(self, "source_tile_index", source_index)
        object.__setattr__(self, "destination_tile_index", destination_index)
        frame_index = int(self.frame_index)
        generation = int(self.generation)
        if frame_index < 0 or generation < 0:
            raise ValueError("relay frame_index and generation must be non-negative")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "tracker_probability", probability)
        object.__setattr__(self, "tile_path", path)
        object.__setattr__(self, "visited_tile_indices", tuple(sorted(visited)))
        object.__setattr__(self, "destination_mask", mask)

    @property
    def idempotency_key(self) -> tuple[object, ...]:
        """Stable key used to suppress retries and relay ping-pong."""

        return (
            self.lineage,
            self.source_tile_index,
            self.destination_tile_index,
            self.frame_index,
            self.temporal_direction,
            self.generation,
        )

    @property
    def mask_sha256(self) -> str:
        return hashlib.sha256(memoryview(self.destination_mask).cast("B")).hexdigest()


@dataclass(frozen=True)
class InboundSpatialSeed:
    """A once-per-lineage destination seed, possibly merged from two neighbors."""

    lineage: LtaLineageId
    destination_tile_index: int
    frame_index: int
    temporal_direction: str
    mask: Any = field(repr=False, compare=False)
    source_tile_indices: tuple[int, ...] = ()
    tracker_probability: float = 1.0
    relay_keys: tuple[tuple[object, ...], ...] = ()
    generation: int = 1
    visited_tile_indices: tuple[int, ...] = ()
    provenance_kind: str = "spatial_relay"

    def __post_init__(self) -> None:
        import numpy as np

        if self.temporal_direction not in {"forward", "backward"}:
            raise ValueError("temporal_direction must be 'forward' or 'backward'")
        if self.provenance_kind != "spatial_relay":
            raise ValueError("an inbound seed must use spatial_relay provenance")
        destination = int(self.destination_tile_index)
        frame_index = int(self.frame_index)
        probability = float(self.tracker_probability)
        generation = int(self.generation)
        if destination < 0 or frame_index < 0 or generation < 1:
            raise ValueError(
                "destination/frame indexes must be non-negative and generation positive"
            )
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("tracker_probability must be finite and in [0,1]")
        mask = np.ascontiguousarray(np.asarray(self.mask), dtype=np.bool_).copy()
        if mask.ndim != 2 or not bool(mask.any()):
            raise ValueError("an inbound spatial seed requires a nonempty mask")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "destination_tile_index", destination)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "tracker_probability", probability)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(
            self,
            "source_tile_indices",
            tuple(sorted(set(int(value) for value in self.source_tile_indices))),
        )
        visited = tuple(sorted(set(int(value) for value in self.visited_tile_indices)))
        if destination not in visited or not set(self.source_tile_indices).issubset(visited):
            raise ValueError(
                "visited_tile_indices must retain every source and destination tile"
            )
        object.__setattr__(self, "visited_tile_indices", visited)


@dataclass(frozen=True)
class CrossTileOverlap:
    """Shared-overlap evidence for two neighbor-local tracklets."""

    left_key: TrackletKey
    right_key: TrackletKey
    shared_frame_indices: tuple[int, ...]
    coactive_frame_count: int
    intersection_pixels: int
    union_pixels: int
    iou: float


class RelayLedger:
    """In-memory exactly-once admission for retryable spatial relay work."""

    def __init__(self) -> None:
        self._seen: set[tuple[object, ...]] = set()

    def admit(self, relay: SpatialRelay) -> bool:
        if not isinstance(relay, SpatialRelay):
            raise TypeError("relay must be a SpatialRelay")
        key = relay.idempotency_key
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def __contains__(self, relay: object) -> bool:
        return isinstance(relay, SpatialRelay) and relay.idempotency_key in self._seen

    def __len__(self) -> int:
        return len(self._seen)


def _tile_by_index(tiles: Sequence[TilePlan], index: int) -> TilePlan:
    try:
        return tuple(tiles)[int(index)]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"unknown tile index {index}") from exc


def rebase_tile_mask(mask: Any, source: TilePlan, destination: TilePlan) -> Any:
    """Translate overlap or a touching border into a destination-local mask.

    Overlapping tiles preserve global pixel coordinates.  When stride equals
    tile size, the source border is shifted by one pixel onto the destination
    border so a tracker can continue across a zero-width/height handoff port.
    """

    import numpy as np

    source_mask = np.asarray(mask, dtype=bool)
    if source_mask.shape != (source.height, source.width):
        raise ValueError(
            f"source mask shape {source_mask.shape} does not match tile "
            f"{(source.height, source.width)}"
        )
    sx0, sy0, sx1, sy1 = source.xyxy
    dx0, dy0, dx1, dy1 = destination.xyxy
    gx0, gy0 = max(sx0, dx0), max(sy0, dy0)
    gx1, gy1 = min(sx1, dx1), min(sy1, dy1)
    output = np.zeros((destination.height, destination.width), dtype=np.bool_)
    if gx1 < gx0 or gy1 < gy0:
        return output

    if gx1 > gx0:
        source_x = slice(gx0 - sx0, gx1 - sx0)
        destination_x = slice(gx0 - dx0, gx1 - dx0)
    elif sx1 == dx0:
        source_x = slice(source.width - 1, source.width)
        destination_x = slice(0, 1)
    elif dx1 == sx0:
        source_x = slice(0, 1)
        destination_x = slice(destination.width - 1, destination.width)
    else:
        return output

    if gy1 > gy0:
        source_y = slice(gy0 - sy0, gy1 - sy0)
        destination_y = slice(gy0 - dy0, gy1 - dy0)
    elif sy1 == dy0:
        source_y = slice(source.height - 1, source.height)
        destination_y = slice(0, 1)
    elif dy1 == sy0:
        source_y = slice(0, 1)
        destination_y = slice(destination.height - 1, destination.height)
    else:
        return output

    output[destination_y, destination_x] = source_mask[source_y, source_x]
    return output




def plan_spatial_relays(
    lineage: LtaLineageId,
    tracklet: Tracklet,
    tiles: Sequence[TilePlan],
    *,
    graph: Mapping[int, Sequence[TileNeighbor]] | None = None,
    generation: int = 0,
    tile_path: Sequence[int] | None = None,
    visited_tile_indices: Iterable[int] | None = None,
    min_seed_pixels: int = 1,
) -> tuple[SpatialRelay, ...]:
    """Plan symmetric entry/exit relays from one tracklet to all eight neighbors.

    The last shared active frame seeds forward continuation (an object leaving
    this tile).  The first shared active frame seeds backward continuation (an
    object that entered this tile).  A visited tile path prevents recursive
    relay waves from bouncing indefinitely.
    """

    import numpy as np

    if not isinstance(lineage, LtaLineageId):
        raise TypeError("lineage must be an LtaLineageId")
    if not isinstance(tracklet, Tracklet):
        raise TypeError("tracklet must be a Tracklet")
    minimum = int(min_seed_pixels)
    if minimum < 1:
        raise ValueError("min_seed_pixels must be positive")
    source_index = int(tracklet.tile_index)
    source = _tile_by_index(tiles, source_index)
    if tracklet.mask_shape != (source.height, source.width):
        raise ValueError("tracklet masks do not match their source tile")
    resolved_graph = eight_neighbor_graph(tiles) if graph is None else graph
    path = tuple(int(value) for value in (tile_path or (source_index,)))
    if not path or path[-1] != source_index or len(path) != len(set(path)):
        raise ValueError("tile_path must be a unique path ending at the source tile")
    visited = set(
        int(value)
        for value in (
            path if visited_tile_indices is None else visited_tile_indices
        )
    )
    if source_index not in visited or not set(path).issubset(visited):
        raise ValueError(
            "visited_tile_indices must contain the source and complete tile_path"
        )
    relays: list[SpatialRelay] = []
    for edge in resolved_graph.get(source_index, ()):
        destination_index = int(edge.destination_index)
        if destination_index in visited:
            continue
        destination = _tile_by_index(tiles, destination_index)
        candidates: list[tuple[TrackletFrame, Any]] = []
        for frame in tracklet.frames:
            if not frame.active:
                continue
            translated = rebase_tile_mask(frame.mask, source, destination)
            if int(np.count_nonzero(translated)) >= minimum:
                candidates.append((frame, translated))
        if not candidates:
            continue
        choices = (
            ("backward", candidates[0]),
            ("forward", candidates[-1]),
        )
        for temporal_direction, (frame, translated) in choices:
            relays.append(
                SpatialRelay(
                    lineage=lineage,
                    source_key=tracklet.key,
                    source_tile_index=source_index,
                    destination_tile_index=destination_index,
                    frame_index=frame.frame_index,
                    temporal_direction=temporal_direction,
                    neighbor_direction=edge.direction,
                    overlap_xyxy=edge.overlap_xyxy,
                    destination_mask=translated,
                    tracker_probability=frame.tracker_probability,
                    generation=int(generation),
                    tile_path=path + (destination_index,),
                    visited_tile_indices=tuple(sorted(visited | {destination_index})),
                )
            )
    return tuple(
        sorted(
            relays,
            key=lambda item: (
                item.destination_tile_index,
                item.temporal_direction,
                item.frame_index,
                item.source_key,
            ),
        )
    )


def merge_inbound_relays(relays: Iterable[SpatialRelay]) -> tuple[InboundSpatialSeed, ...]:
    """Union same-lineage arrivals so a destination session is seeded once."""

    import numpy as np

    grouped: dict[tuple[object, ...], list[SpatialRelay]] = {}
    for relay in relays:
        if not isinstance(relay, SpatialRelay):
            raise TypeError("relays must contain only SpatialRelay instances")
        key = (
            relay.lineage,
            relay.destination_tile_index,
            relay.frame_index,
            relay.temporal_direction,
        )
        grouped.setdefault(key, []).append(relay)
    merged: list[InboundSpatialSeed] = []
    for key in sorted(grouped, key=lambda value: (value[0], value[1], value[2], value[3])):
        items = sorted(grouped[key], key=lambda item: item.source_tile_index)
        shape = items[0].destination_mask.shape
        if any(item.destination_mask.shape != shape for item in items):
            raise ValueError("inbound relay masks for one destination must have one shape")
        mask = np.zeros(shape, dtype=np.bool_)
        for item in items:
            mask |= item.destination_mask
        merged.append(
            InboundSpatialSeed(
                lineage=items[0].lineage,
                destination_tile_index=items[0].destination_tile_index,
                frame_index=items[0].frame_index,
                temporal_direction=items[0].temporal_direction,
                mask=mask,
                source_tile_indices=tuple(item.source_tile_index for item in items),
                tracker_probability=max(item.tracker_probability for item in items),
                relay_keys=tuple(item.idempotency_key for item in items),
                generation=max(item.generation for item in items) + 1,
                visited_tile_indices=tuple(
                    sorted(
                        {
                            tile_index
                            for item in items
                            for tile_index in item.visited_tile_indices
                        }
                    )
                ),
            )
        )
    return tuple(merged)


def cross_tile_overlap(
    left: Tracklet,
    right: Tracklet,
    tiles: Sequence[TilePlan],
) -> CrossTileOverlap:
    """Measure neighbor-local tracklets in their shared global rectangle."""

    import numpy as np

    if left.tile_index == right.tile_index:
        raise ValueError("cross-tile overlap requires distinct tiles")
    graph = eight_neighbor_graph(tiles)
    edge = next(
        (
            item
            for item in graph.get(int(left.tile_index), ())
            if item.destination_index == int(right.tile_index)
        ),
        None,
    )
    if edge is None:
        raise ValueError("cross-tile overlap requires eight-neighbor tiles")
    left_tile = _tile_by_index(tiles, left.tile_index)
    right_tile = _tile_by_index(tiles, right.tile_index)
    left_map = left.frame_map()
    right_map = right.frame_map()
    shared = tuple(sorted(set(left_map) & set(right_map)))
    intersection = union = coactive = 0
    for frame_index in shared:
        left_mask = rebase_tile_mask(
            left_map[frame_index].mask,
            left_tile,
            right_tile,
        )
        port = rebase_tile_mask(
            np.ones((left_tile.height, left_tile.width), dtype=np.bool_),
            left_tile,
            right_tile,
        )
        right_mask = np.asarray(right_map[frame_index].mask, dtype=np.bool_) & port
        left_active = bool(left_map[frame_index].active and left_mask.any())
        right_active = bool(right_map[frame_index].active and right_mask.any())
        if not left_active:
            left_mask = np.zeros_like(left_mask)
        if not right_active:
            right_mask = np.zeros_like(right_mask)
        if left_active and right_active:
            coactive += 1
        intersection += int(np.count_nonzero(left_mask & right_mask))
        union += int(np.count_nonzero(left_mask | right_mask))
    return CrossTileOverlap(
        left_key=left.key,
        right_key=right.key,
        shared_frame_indices=shared,
        coactive_frame_count=coactive,
        intersection_pixels=intersection,
        union_pixels=union,
        iou=0.0 if union == 0 else float(intersection / union),
    )


__all__ = (
    "CrossTileOverlap",
    "InboundSpatialSeed",
    "LtaLineageId",
    "RelayLedger",
    "SpatialRelay",
    "cross_tile_overlap",
    "merge_inbound_relays",
    "plan_spatial_relays",
    "rebase_tile_mask",
)
