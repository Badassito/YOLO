"""Compact, verified overlap episodes shared by LTA window tasks.

Window seams are internal tracker context boundaries. Combining their compact
intervals before spatial relay publication preserves the original chain's
first/last endpoint policy instead of inventing new seeds at every seam.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import operator
from pathlib import Path
import tempfile
from typing import Any, Mapping, MutableMapping

from .lta_tile_tracking import LtaLineageId


RELAY_OBSERVATION_SCHEMA = "lta.relay-observations/1"


@dataclass(frozen=True)
class _RelaySource:
    object_id: int
    visited_tile_indices: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_value(value: tuple[object, ...]) -> tuple[object, ...]:
    import math
    import numpy as np

    if len(value) != 4:
        raise ValueError("relay endpoint must contain frame, packed mask, shape, and probability")
    frame, packed, shape, probability = value
    frame = int(frame)
    shape = tuple(int(item) for item in shape)
    probability = float(probability)
    packed = np.asarray(packed)
    if frame < 0 or len(shape) != 2 or any(item < 1 for item in shape):
        raise ValueError("relay endpoint has invalid frame or mask shape")
    if packed.dtype != np.uint8 or packed.ndim != 1 or len(packed) != (shape[0] * shape[1] + 7) // 8:
        raise ValueError("relay endpoint has invalid packed mask storage")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("relay endpoint probability must be finite and in [0,1]")
    if not bool(packed.any()):
        raise ValueError("relay endpoint must contain foreground")
    return frame, packed, shape, probability


def _union_endpoint(left: tuple[object, ...], right: tuple[object, ...]) -> tuple[object, ...]:
    import numpy as np

    if left[0] != right[0] or left[2] != right[2]:
        raise ValueError("relay endpoints must share frame and shape before union")
    return left[0], np.bitwise_or(left[1], right[1]), left[2], max(left[3], right[3])


def _merge_episode_ranges(
    values,
    *,
    shared_geometry: bool = False,
) -> list[tuple[tuple[object, ...], tuple[object, ...]]]:
    """Validate first, then coalesce ranges without mutating endpoint masks."""

    episodes = []
    expected_shape = None
    for first, last in values:
        first, last = _checked_value(first), _checked_value(last)
        if first[0] > last[0] or first[2] != last[2]:
            raise ValueError("relay episode has inconsistent endpoints")
        if shared_geometry:
            if expected_shape is not None and first[2] != expected_shape:
                raise ValueError("one relay destination has inconsistent mask geometry")
            expected_shape = first[2]
            if first[0] == last[0]:
                # A one-frame episode has one boundary; preserve evidence
                # supplied through either endpoint representation.
                first = last = _union_endpoint(first, last)
        episodes.append((first, last))
    episodes.sort(key=lambda item: (item[0][0], item[1][0]))
    merged = []
    for first, last in episodes:
        if not merged or int(first[0]) > int(merged[-1][1][0]) + 1:
            merged.append((first, last))
            continue
        old_first, old_last = merged[-1]
        if first[0] == old_first[0]:
            old_first = _union_endpoint(old_first, first)
        if last[0] == old_last[0]:
            old_last = _union_endpoint(old_last, last)
        elif int(last[0]) > int(old_last[0]):
            old_last = last
        merged[-1] = (old_first, old_last)
    return merged


def merge_relay_observations(
    destination: MutableMapping[tuple[str, int], dict[str, object]],
    incoming: Mapping[tuple[str, int], Mapping[str, object]],
) -> None:
    """Merge contiguous intervals without retaining interior-frame masks."""

    for key, record in incoming.items():
        lineage = record["lineage"]
        tile = int(record["destination_index"])
        seed = record["seed"]
        if not isinstance(lineage, LtaLineageId) or key != (lineage.token, tile) or tile < 0:
            raise ValueError("relay observation key differs from its lineage/destination")
        target = destination.get(key)
        if target is None:
            target = {
                "lineage": lineage, "seed": seed,
                "destination_index": tile, "episodes": [],
            }
        elif (
            target["lineage"] != lineage
            or int(target["seed"].object_id) != int(seed.object_id)
            or tuple(target["seed"].visited_tile_indices) != tuple(seed.visited_tile_indices)
        ):
            raise ValueError("one chain's relay lineage/source identity changed across windows")
        merged = _merge_episode_ranges((*target["episodes"], *record.get("episodes", ())))
        destination[key] = {**target, "episodes": merged}


def _checked_relay_source(seed: object) -> _RelaySource:
    try:
        raw_id = seed.object_id
        raw_visited = tuple(seed.visited_tile_indices)
        if isinstance(raw_id, bool) or any(isinstance(value, bool) for value in raw_visited):
            raise TypeError("boolean source indexes are not integers")
        object_id = operator.index(raw_id)
        visited = tuple(sorted({operator.index(value) for value in raw_visited}))
    except (AttributeError, TypeError) as exc:
        raise ValueError("relay source object and visited tile indexes must be integers") from exc
    if object_id < 0 or any(value < 0 for value in visited):
        raise ValueError("relay source object and visited tile indexes must be nonnegative")
    return _RelaySource(object_id, visited)


def _relay_endpoint_candidates(record: Mapping[str, object]):
    """Retain original chain endpoints even after their intervals coalesce."""

    if "endpoint_candidates" in record:
        try:
            return tuple(record["endpoint_candidates"])
        except TypeError as exc:
            raise ValueError("relay endpoint_candidates must be a sequence") from exc
    candidates = []
    for first, last in record.get("episodes", ()):
        first, last = _checked_value(first), _checked_value(last)
        if first[0] > last[0] or first[2] != last[2]:
            raise ValueError("relay episode has inconsistent endpoints")
        if first[0] == last[0]:
            first = last = _union_endpoint(first, last)
        bounds = (int(first[0]), int(last[0]) + 1)
        candidates.extend((("forward", first, bounds), ("backward", last, bounds)))
    return tuple(candidates)


def _merge_relay_endpoint_candidates(candidates, episodes):
    """Union only candidates with the same direction and physical frame."""

    by_event = {}
    expected_shape = None if not episodes else episodes[0][0][2]
    for candidate in candidates:
        try:
            direction, value, bounds = candidate
            raw_start, raw_stop = bounds
            if isinstance(raw_start, bool) or isinstance(raw_stop, bool):
                raise TypeError("boolean episode bounds are not integers")
            start, stop = operator.index(raw_start), operator.index(raw_stop)
        except (TypeError, ValueError) as exc:
            raise ValueError("relay endpoint candidate must contain direction, value, and integer bounds") from exc
        if direction not in {"forward", "backward"}:
            raise ValueError("relay endpoint candidate direction must be forward or backward")
        value = _checked_value(value)
        frame = int(value[0])
        if not 0 <= start <= frame < stop:
            raise ValueError("relay endpoint candidate frame is outside its episode")
        if (direction == "forward" and start != frame) or (direction == "backward" and stop != frame + 1):
            raise ValueError("relay endpoint candidate must remain at its directional episode boundary")
        if value[2] != expected_shape:
            raise ValueError("relay endpoint candidate geometry differs from its destination")
        if not any(int(first[0]) <= start and stop <= int(last[0]) + 1 for first, last in episodes):
            raise ValueError("relay endpoint candidate episode lies outside the observed intervals")
        key = direction, frame
        previous = by_event.get(key)
        if previous is not None:
            previous_value, previous_bounds = previous
            value = _union_endpoint(previous_value, value)
            if direction == "forward":
                start, stop = frame, max(previous_bounds[1], stop)
            else:
                start, stop = min(previous_bounds[0], start), frame + 1
        by_event[key] = value, (start, stop)
    if episodes and not by_event:
        raise ValueError("nonempty relay observations require endpoint candidates")
    return [
        (direction, by_event[(direction, frame)][0], by_event[(direction, frame)][1])
        for direction, frame in sorted(by_event, key=lambda key: (key[1], key[0] != "forward"))
    ]


def merge_relay_observations_across_chains(
    destination: MutableMapping[tuple[str, int], dict[str, object]],
    incoming: Mapping[tuple[str, int], Mapping[str, object]],
) -> None:
    """Coalesce one lineage's overlap episodes across independently seeded chains.

    The caller must separate generations, tile configurations and source tiles.
    Global lineage identity remains exact; local object IDs are represented by
    their minimum and visited tile histories are unioned. Merged episodes are
    temporal summaries; endpoint_candidates retains every original chain
    boundary for spatial admission, including novel nested/interior evidence.
    Only same-direction, same-frame candidates share mask bits. Each record is
    validated before replacement; earlier valid records in the same call remain
    committed if a later record is invalid.
    """

    for key, record in incoming.items():
        lineage = record["lineage"]
        tile = int(record["destination_index"])
        if not isinstance(lineage, LtaLineageId) or key != (lineage.token, tile) or tile < 0:
            raise ValueError("relay observation key differs from its lineage/destination")
        source = _checked_relay_source(record["seed"])
        target = destination.get(key)
        if target is None:
            target = {"lineage": lineage, "destination_index": tile, "episodes": []}
        else:
            if target["lineage"] != lineage or int(target["destination_index"]) != tile:
                raise ValueError("cross-chain relay merge changed its global lineage/destination")
            previous = _checked_relay_source(target["seed"])
            source = _RelaySource(
                min(previous.object_id, source.object_id),
                tuple(sorted(set(previous.visited_tile_indices) | set(source.visited_tile_indices))),
            )
        merged = _merge_episode_ranges(
            (*target["episodes"], *record.get("episodes", ())), shared_geometry=True,
        )
        candidates = _merge_relay_endpoint_candidates(
            (*_relay_endpoint_candidates(target), *_relay_endpoint_candidates(record)), merged,
        )
        destination[key] = {**target, "seed": source, "episodes": merged,
                            "endpoint_candidates": candidates}


def write_relay_observations(
    output_dir: Path,
    observations: Mapping[tuple[str, int], Mapping[str, object]],
) -> dict[str, object]:
    """Write endpoint bitmaps plus source metadata as one immutable NPZ."""

    import numpy as np

    normalized: dict[tuple[str, int], dict[str, object]] = {}
    merge_relay_observations(normalized, observations)
    arrays = {}
    rows = []
    endpoint_count = 0
    episode_count = 0
    for _key, record in sorted(normalized.items()):
        episodes = []
        for first, last in record["episodes"]:
            endpoints = []
            for frame, packed, shape, probability in (first, last):
                name = f"mask_{endpoint_count:06d}"
                arrays[name] = packed
                endpoints.append({
                    "frame_index": frame, "mask_key": name,
                    "shape_hw": list(shape), "tracker_probability": probability,
                })
                endpoint_count += 1
            episodes.append(endpoints)
            episode_count += 1
        rows.append({
            "lineage": asdict(record["lineage"]),
            "destination_index": record["destination_index"],
            "object_id": int(record["seed"].object_id),
            "visited_tile_indices": list(record["seed"].visited_tile_indices),
            "episodes": episodes,
        })
    metadata = {"schema": RELAY_OBSERVATION_SCHEMA, "observations": rows}
    arrays["metadata"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        dtype=np.uint8,
    )
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "relay_observations.npz"
    descriptor, temporary = tempfile.mkstemp(prefix=".relay_observations.", suffix=".partial", dir=folder)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "schema": RELAY_OBSERVATION_SCHEMA, "path": str(path.resolve()),
        "sha256": _sha256(path), "observation_count": len(rows), "episode_count": episode_count,
    }


def read_relay_observations(receipt: Mapping[str, object]) -> dict[tuple[str, int], dict[str, object]]:
    """Verify and decode compact episodes without expanding tile-sized masks."""

    import numpy as np

    if receipt.get("schema") != RELAY_OBSERVATION_SCHEMA:
        raise ValueError("unsupported relay observation receipt schema")
    path = Path(str(receipt["path"])).resolve(strict=True)
    if _sha256(path) != str(receipt["sha256"]):
        raise RuntimeError("relay observation artifact digest changed")
    result: dict[tuple[str, int], dict[str, object]] = {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(np.asarray(archive["metadata"], dtype=np.uint8).tobytes().decode("utf-8"))
        if metadata.get("schema") != RELAY_OBSERVATION_SCHEMA:
            raise ValueError("unsupported relay observation artifact schema")
        for row in metadata["observations"]:
            lineage = LtaLineageId(**row["lineage"])
            tile = int(row["destination_index"])
            key = lineage.token, tile
            if key in result:
                raise ValueError("relay observation artifact repeats a lineage/destination")
            episodes = []
            for endpoints in row["episodes"]:
                if len(endpoints) != 2:
                    raise ValueError("relay episode must contain exactly two endpoints")
                values = []
                for endpoint in endpoints:
                    values.append(_checked_value((
                        endpoint["frame_index"], np.asarray(archive[endpoint["mask_key"]]).copy(),
                        tuple(endpoint["shape_hw"]), endpoint["tracker_probability"],
                    )))
                episodes.append(tuple(values))
            result[key] = {
                "lineage": lineage, "destination_index": tile,
                "seed": _RelaySource(int(row["object_id"]), tuple(row["visited_tile_indices"])),
                "episodes": episodes,
            }
    normalized: dict[tuple[str, int], dict[str, object]] = {}
    merge_relay_observations(normalized, result)
    if (
        len(normalized) != int(receipt["observation_count"])
        or sum(len(record["episodes"]) for record in normalized.values()) != int(receipt["episode_count"])
    ):
        raise RuntimeError("relay observation artifact counts differ from receipt")
    return normalized


__all__ = (
    "RELAY_OBSERVATION_SCHEMA", "merge_relay_observations", "merge_relay_observations_across_chains",
    "read_relay_observations", "write_relay_observations",
)
