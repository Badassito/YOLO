"""Production authoritative-mask propagation contracts for LTA.

Mask injection is the only production video-conditioning path.  The private
SAM calls remain quarantined in :mod:`XTA.lta_experimental`, while this module
owns stable requests, globally scoped identities, hole-filled predictions,
dogfood boundaries, and hard-positive provenance.
"""

from __future__ import annotations

import math
import operator
import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .lta_experimental import (
    MASK_SEED_MIN_EXCLUSIVE_FRACTION,
    resolve_mask_seed_capacity,
    run_mask_seed_session,
)
from .lta_sam import LTA_MAX_NUM_OBJECTS, SamFramePrediction, SamSessionPlan
from .lta_tile_tracking import LtaLineageId


class LtaSeedProvenance(str, Enum):
    AUTHORITATIVE = "authoritative"
    TEMPORAL_DOGFOOD = "temporal_dogfood"
    SPATIAL_RELAY = "spatial_relay"

    @classmethod
    def coerce(cls, value: "LtaSeedProvenance | str") -> "LtaSeedProvenance":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown LTA seed provenance {value!r}; use {allowed}") from exc


def _nonnegative_index(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(resolved)


def _binary_mask_copy(mask: object, *, name: str) -> Any:
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2 or any(int(value) < 1 for value in array.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional mask")
    output = np.ascontiguousarray(array != 0, dtype=np.bool_).copy()
    if not bool(output.any()):
        raise ValueError(f"{name} must contain foreground")
    output.setflags(write=False)
    return output


def _fill_binary_mask(mask: object) -> Any:
    # Lazy import keeps LTA CLI/config discovery free of OpenCV/SciPy.
    from .lta_postprocessing import fill_binary_mask_holes_2d

    return fill_binary_mask_holes_2d(mask)


@dataclass(frozen=True)
class LtaMaskSeed:
    """One globally identified object mask injected at a tracker frame."""

    lineage: LtaLineageId
    frame_index: int
    object_id: int
    mask: Any = field(repr=False, compare=False)
    provenance: LtaSeedProvenance | str = LtaSeedProvenance.AUTHORITATIVE
    tracker_probability: float = 1.0
    relay_generation: int = 0
    visited_tile_indices: tuple[int, ...] = ()
    source_receipt: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, LtaLineageId):
            raise TypeError("lineage must be an LtaLineageId")
        frame_index = _nonnegative_index(self.frame_index, name="frame_index")
        object_id = _nonnegative_index(self.object_id, name="object_id")
        probability = float(self.tracker_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("tracker_probability must be finite and in [0,1]")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "mask", _binary_mask_copy(self.mask, name="seed mask"))
        object.__setattr__(self, "provenance", LtaSeedProvenance.coerce(self.provenance))
        object.__setattr__(self, "tracker_probability", probability)
        generation = _nonnegative_index(self.relay_generation, name="relay_generation")
        visited = tuple(
            sorted(
                {
                    _nonnegative_index(value, name="visited_tile_index")
                    for value in self.visited_tile_indices
                }
            )
        )
        object.__setattr__(self, "relay_generation", generation)
        object.__setattr__(self, "visited_tile_indices", visited)
        object.__setattr__(self, "source_receipt", dict(self.source_receipt))


def partition_mask_seed_sessions(
    seeds: Sequence[LtaMaskSeed],
    *,
    max_objects: int = LTA_MAX_NUM_OBJECTS,
    min_exclusive_fraction: float = MASK_SEED_MIN_EXCLUSIVE_FRACTION,
    mask_transform: Callable[[object], object] | None = None,
) -> tuple[tuple[LtaMaskSeed, ...], ...]:
    """Partition seeds with deterministic cumulative-overlap first-fit.

    A seed is placed in the first session where every mask, after considering
    the aggregate overlap of the entire tentative session, retains at least
    ``min_exclusive_fraction`` of its foreground as exclusive pixels.  A new
    session is opened otherwise.  Separate SAM sessions avoid the multiplex
    model's shared-pixel suppression and therefore preserve the full masks.
    ``mask_transform`` can mirror injection-time normalization such as hole
    filling; it is applied to private copies solely for compatibility checks.
    Input order is never sorted, and each returned seed is the original object.
    """

    import numpy as np

    capacity = _nonnegative_index(max_objects, name="max_objects")
    if capacity < 1:
        raise ValueError("max_objects must be positive")
    if capacity > LTA_MAX_NUM_OBJECTS:
        raise ValueError(
            f"max_objects must not exceed the {LTA_MAX_NUM_OBJECTS}-object cap"
        )
    if isinstance(min_exclusive_fraction, bool):
        raise TypeError("min_exclusive_fraction must be a real number")
    try:
        minimum = float(min_exclusive_fraction)
    except (TypeError, ValueError) as exc:
        raise TypeError("min_exclusive_fraction must be a real number") from exc
    if not math.isfinite(minimum) or not 0.0 < minimum <= 1.0:
        raise ValueError("min_exclusive_fraction must be finite and in (0,1]")
    if mask_transform is not None and not callable(mask_transform):
        raise TypeError("mask_transform must be callable or None")

    ordered = tuple(seeds)
    if not all(isinstance(seed, LtaMaskSeed) for seed in ordered):
        raise TypeError("seeds must contain only LtaMaskSeed values")
    if not ordered:
        return ()
    if len({seed.frame_index for seed in ordered}) != 1:
        raise ValueError("seed-session inputs must all belong to the same frame")
    lineages = [seed.lineage for seed in ordered]
    if len(lineages) != len(set(lineages)):
        raise ValueError("seed-session inputs must have unique lineages")
    shapes = {tuple(int(value) for value in seed.mask.shape) for seed in ordered}
    if len(shapes) != 1:
        raise ValueError("seed-session masks must all have the same shape")

    masks_list = []
    for index, seed in enumerate(ordered):
        source = np.asarray(seed.mask, dtype=np.bool_)
        transformed = (
            source
            if mask_transform is None
            else np.asarray(mask_transform(source.copy()))
        )
        if transformed.ndim != 2 or any(int(value) < 1 for value in transformed.shape):
            raise ValueError(
                f"transformed seed mask {index} must be nonempty and two-dimensional"
            )
        if transformed.shape != source.shape:
            raise ValueError(
                f"transformed seed mask {index} changed shape from "
                f"{source.shape} to {transformed.shape}"
            )
        transformed = np.ascontiguousarray(transformed != 0, dtype=np.bool_)
        if not bool(transformed.any()):
            raise ValueError(f"transformed seed mask {index} must contain foreground")
        masks_list.append(transformed)
    masks = tuple(masks_list)
    foreground_pixels = tuple(int(np.count_nonzero(mask)) for mask in masks)
    batches: list[list[LtaMaskSeed]] = []
    # Per-pixel owner state is -1 for empty, -2 for shared, or the local index
    # of the sole owner.  This lets each candidate account for cumulative
    # overlap in one mask traversal rather than relying on pairwise IoU tests.
    owner_maps: list[Any] = []
    exclusive_counts: list[list[int]] = []
    batch_foreground_counts: list[list[int]] = []

    def meets_minimum(exclusive: int, foreground: int) -> bool:
        return int(exclusive) >= minimum * int(foreground)

    for seed, mask, foreground in zip(ordered, masks, foreground_pixels):
        placed = False
        for batch_index, batch in enumerate(batches):
            if len(batch) >= capacity:
                continue
            owner_map = owner_maps[batch_index]
            owners_under_candidate = owner_map[mask]
            candidate_exclusive = int(np.count_nonzero(owners_under_candidate == -1))
            if not meets_minimum(candidate_exclusive, foreground):
                continue

            sole_owners = owners_under_candidate[owners_under_candidate >= 0]
            losses = np.bincount(
                sole_owners.astype(np.intp, copy=False),
                minlength=len(batch),
            )
            tentative_exclusive = [
                current - int(losses[index])
                for index, current in enumerate(exclusive_counts[batch_index])
            ]
            if any(
                not meets_minimum(exclusive, member_foreground)
                for exclusive, member_foreground in zip(
                    tentative_exclusive,
                    batch_foreground_counts[batch_index],
                )
            ):
                continue

            local_index = len(batch)
            newly_shared = mask & (owner_map >= 0)
            newly_owned = mask & (owner_map == -1)
            owner_map[newly_shared] = -2
            owner_map[newly_owned] = local_index
            batch.append(seed)
            exclusive_counts[batch_index] = tentative_exclusive + [candidate_exclusive]
            batch_foreground_counts[batch_index].append(foreground)
            placed = True
            break

        if not placed:
            owner_map = np.full(mask.shape, -1, dtype=np.int16)
            owner_map[mask] = 0
            batches.append([seed])
            owner_maps.append(owner_map)
            exclusive_counts.append([foreground])
            batch_foreground_counts.append([foreground])

    return tuple(tuple(batch) for batch in batches)


@dataclass(frozen=True)
class LtaPropagationRequest:
    """One atomic SAM session seeded exclusively by authoritative mask copies."""

    work_id: str
    session: SamSessionPlan
    prompt_frame: int
    direction: str
    seeds: tuple[LtaMaskSeed, ...]
    conf: float
    empty_frame_limit: int | None = 30
    relay_generation: int = 0

    def __post_init__(self) -> None:
        work_id = str(self.work_id).strip()
        if not work_id:
            raise ValueError("work_id must not be empty")
        session = self.session
        if not isinstance(session, SamSessionPlan):
            try:
                session = SamSessionPlan(
                    sequence_id=str(session.sequence_id),
                    session_index=int(session.session_index),
                    frame_start=int(session.frame_start),
                    frame_stop=int(session.frame_stop),
                )
            except Exception as exc:
                raise TypeError("session must satisfy the SamSessionPlan contract") from exc
            object.__setattr__(self, "session", session)
        prompt = _nonnegative_index(self.prompt_frame, name="prompt_frame")
        if prompt not in self.session.frame_indices:
            raise ValueError("prompt_frame must lie inside the session")
        direction = str(self.direction).strip().lower()
        if direction not in {"both", "forward", "backward"}:
            raise ValueError("direction must be 'both', 'forward', or 'backward'")
        seeds = tuple(self.seeds)
        if not seeds or not all(isinstance(item, LtaMaskSeed) for item in seeds):
            raise ValueError("a propagation request requires LtaMaskSeed values")
        if len(seeds) > LTA_MAX_NUM_OBJECTS:
            raise ValueError(
                f"a propagation request exceeds the {LTA_MAX_NUM_OBJECTS}-object cap"
            )
        if any(item.frame_index != prompt for item in seeds):
            raise ValueError("every seed must belong to the prompt frame")
        shapes = {tuple(int(value) for value in item.mask.shape) for item in seeds}
        if len(shapes) != 1:
            raise ValueError("every seed in one request must have the same mask shape")
        object_ids = [item.object_id for item in seeds]
        lineages = [item.lineage for item in seeds]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("seed object_id values must be unique")
        if len(lineages) != len(set(lineages)):
            raise ValueError("seed lineages must be unique")
        conf = float(self.conf)
        if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be finite and in [0,1]")
        limit = self.empty_frame_limit
        if limit is not None:
            limit = _nonnegative_index(limit, name="empty_frame_limit")
            if limit < 1:
                raise ValueError("empty_frame_limit must be positive")
        generation = _nonnegative_index(self.relay_generation, name="relay_generation")
        object.__setattr__(self, "work_id", work_id)
        object.__setattr__(self, "prompt_frame", prompt)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "seeds", tuple(sorted(seeds, key=lambda item: item.object_id)))
        object.__setattr__(self, "conf", conf)
        object.__setattr__(self, "empty_frame_limit", limit)
        object.__setattr__(self, "relay_generation", generation)


@dataclass(frozen=True)
class LtaObjectPrediction:
    """One filled SAM prediction retaining its global lineage identity."""

    lineage: LtaLineageId
    prediction: SamFramePrediction
    hole_fill_added_pixels: int
    source_provenance: LtaSeedProvenance


@dataclass(frozen=True)
class LtaPropagationResult:
    request: LtaPropagationRequest
    predictions: tuple[LtaObjectPrediction, ...]
    dogfood_seeds: tuple[LtaMaskSeed, ...]
    adapter_receipt: Mapping[str, object]
    hole_fill_added_pixels: int

    def frame_union(self, frame_index: int) -> Any:
        import numpy as np

        selected = [
            item.prediction.binary_mask
            for item in self.predictions
            if item.prediction.frame_index == int(frame_index)
        ]
        if not selected:
            shape = tuple(int(value) for value in self.request.seeds[0].mask.shape)
            return np.zeros(shape, dtype=np.bool_)
        return np.logical_or.reduce(tuple(np.asarray(mask, dtype=bool) for mask in selected))


@dataclass(frozen=True)
class LtaSeedArtifact:
    path: Path
    sha256: str
    seed_count: int
    mask_shape: tuple[int, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_seed_artifact(path: str | Path, seeds: Sequence[LtaMaskSeed]) -> LtaSeedArtifact:
    """Atomically persist compact seed masks without pickle."""

    import numpy as np

    ordered = tuple(seeds)
    if not ordered or not all(isinstance(seed, LtaMaskSeed) for seed in ordered):
        raise ValueError("seed artifacts require LtaMaskSeed values")
    shapes = {tuple(int(value) for value in seed.mask.shape) for seed in ordered}
    if len(shapes) != 1:
        raise ValueError("seed artifact masks must share one shape")
    metadata = [
        {
            "lineage": {
                "volume_id": seed.lineage.volume_id,
                "physical_view_id": seed.lineage.physical_view_id,
                "runtime_view_id": seed.lineage.runtime_view_id,
                "tile_config_id": seed.lineage.tile_config_id,
                "lineage_id": seed.lineage.lineage_id,
            },
            "frame_index": seed.frame_index,
            "object_id": seed.object_id,
            "provenance": seed.provenance.value,
            "tracker_probability": seed.tracker_probability,
            "relay_generation": seed.relay_generation,
            "visited_tile_indices": list(seed.visited_tile_indices),
            "source_receipt": dict(seed.source_receipt),
        }
        for seed in ordered
    ]
    destination = Path(path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.partial")
    try:
        with stage.open("wb") as handle:
            np.savez_compressed(
                handle,
                masks=np.stack(
                    [np.asarray(seed.mask, dtype=np.uint8) for seed in ordered]
                ),
                metadata_utf8=np.frombuffer(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8"),
                    dtype=np.uint8,
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, destination)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    shape = next(iter(shapes))
    return LtaSeedArtifact(
        path=destination,
        sha256=_sha256_file(destination),
        seed_count=len(ordered),
        mask_shape=shape,
    )


def read_seed_artifact(
    artifact: LtaSeedArtifact | str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[LtaMaskSeed, ...]:
    """Load and validate an immutable seed artifact."""

    import numpy as np

    path = artifact.path if isinstance(artifact, LtaSeedArtifact) else Path(artifact)
    path = Path(path).resolve(strict=True)
    expected = (
        artifact.sha256
        if isinstance(artifact, LtaSeedArtifact)
        else expected_sha256
    )
    actual = _sha256_file(path)
    if expected is not None and actual != str(expected):
        raise RuntimeError(
            f"LTA seed artifact digest changed: expected={expected}, actual={actual}"
        )
    with np.load(path, allow_pickle=False) as payload:
        masks = np.asarray(payload["masks"], dtype=np.uint8)
        metadata = json.loads(bytes(payload["metadata_utf8"].tolist()).decode("utf-8"))
    if masks.ndim != 3 or len(metadata) != int(masks.shape[0]):
        raise RuntimeError("LTA seed artifact mask/metadata counts disagree")
    seeds = []
    for index, record in enumerate(metadata):
        lineage = record["lineage"]
        seeds.append(
            LtaMaskSeed(
                lineage=LtaLineageId(
                    volume_id=lineage["volume_id"],
                    physical_view_id=lineage["physical_view_id"],
                    runtime_view_id=lineage["runtime_view_id"],
                    tile_config_id=lineage["tile_config_id"],
                    lineage_id=lineage["lineage_id"],
                ),
                frame_index=record["frame_index"],
                object_id=record["object_id"],
                mask=masks[index],
                provenance=record["provenance"],
                tracker_probability=record["tracker_probability"],
                relay_generation=record.get("relay_generation", 0),
                visited_tile_indices=tuple(record.get("visited_tile_indices", ())),
                source_receipt=record.get("source_receipt", {}),
            )
        )
    if isinstance(artifact, LtaSeedArtifact):
        if len(seeds) != artifact.seed_count:
            raise RuntimeError("LTA seed artifact count changed")
        if seeds and tuple(seeds[0].mask.shape) != artifact.mask_shape:
            raise RuntimeError("LTA seed artifact mask shape changed")
    return tuple(seeds)


def _expected_frames(request: LtaPropagationRequest) -> set[int]:
    if request.direction == "both":
        return set(request.session.frame_indices)
    if request.direction == "forward":
        return set(range(request.prompt_frame, request.session.frame_stop))
    return set(range(request.session.frame_start, request.prompt_frame + 1))


def run_mask_injected_session(
    measured_predictor: object,
    raw_predictor: object,
    *,
    resource: list[object],
    request: LtaPropagationRequest,
    adapter: Callable[..., Mapping[str, object]] = run_mask_seed_session,
    fill_mask: Callable[[object], object] = _fill_binary_mask,
    prediction_callback: Callable[[LtaObjectPrediction], None] | None = None,
    retain_predictions: bool = True,
) -> LtaPropagationResult:
    """Execute one production mask-injected session and fill every reusable mask.

    The default remains the diagnostic/tool-friendly retained result.  A
    production reducer can instead supply ``prediction_callback`` and set
    ``retain_predictions=False``; in that mode only boundary dogfood masks,
    authoritative prompt copies, and compact audit state survive the stream.
    """

    import numpy as np

    if not isinstance(request, LtaPropagationRequest):
        raise TypeError("request must be an LtaPropagationRequest")
    if prediction_callback is not None and not callable(prediction_callback):
        raise TypeError("prediction_callback must be callable or None")
    if not isinstance(retain_predictions, bool):
        raise TypeError("retain_predictions must be a bool")
    if len(resource) != request.session.frame_count:
        raise ValueError(
            f"resource frame count {len(resource)} != session {request.session.frame_count}"
        )
    filled_seeds: list[LtaMaskSeed] = []
    seed_added_pixels = 0
    for seed in request.seeds:
        filled = np.ascontiguousarray(fill_mask(seed.mask), dtype=np.bool_)
        if filled.shape != seed.mask.shape:
            raise RuntimeError(
                f"injection hole filling changed seed mask shape from "
                f"{seed.mask.shape} to {filled.shape}"
            )
        added = int(np.count_nonzero(filled & ~np.asarray(seed.mask, dtype=bool)))
        seed_added_pixels += added
        filled_seeds.append(
            LtaMaskSeed(
                lineage=seed.lineage,
                frame_index=seed.frame_index,
                object_id=seed.object_id,
                mask=filled,
                provenance=seed.provenance,
                tracker_probability=seed.tracker_probability,
                relay_generation=seed.relay_generation,
                visited_tile_indices=seed.visited_tile_indices,
                source_receipt={
                    **dict(seed.source_receipt),
                    "injection_hole_fill_added_pixels": added,
                },
            )
        )
    ground_truth = np.logical_or.reduce(tuple(item.mask for item in filled_seeds))
    by_local_id = {index: seed for index, seed in enumerate(filled_seeds)}
    expected_frames = _expected_frames(request)
    boundary_frames = (
        (request.session.frame_start, request.session.frame_stop - 1)
        if request.direction == "both"
        else (
            (request.session.frame_stop - 1,)
            if request.direction == "forward"
            else (request.session.frame_start,)
        )
    )
    boundary_frame_set = set(boundary_frames)
    retained: dict[tuple[int, int], LtaObjectPrediction] = {}
    boundary_predictions: dict[tuple[int, int], LtaObjectPrediction] = {}
    raw_prediction_keys: set[tuple[int, int]] = set()
    prediction_added_pixels = 0
    canonical_prediction_count = 0
    callback_prediction_count = 0
    below_confidence_prediction_count = 0

    def consume_raw_prediction(prediction: object) -> None:
        nonlocal prediction_added_pixels
        nonlocal canonical_prediction_count
        nonlocal callback_prediction_count
        nonlocal below_confidence_prediction_count
        if not isinstance(prediction, SamFramePrediction):
            try:
                prediction = SamFramePrediction(
                    sequence_id=str(prediction.sequence_id),
                    session_index=int(prediction.session_index),
                    frame_index=int(prediction.frame_index),
                    object_id=int(prediction.object_id),
                    initial_detection_score=float(prediction.initial_detection_score),
                    frame_tracker_score=(
                        None
                        if prediction.frame_tracker_score is None
                        else float(prediction.frame_tracker_score)
                    ),
                    binary_mask=prediction.binary_mask,
                )
            except Exception as exc:
                raise RuntimeError(
                    "mask propagation returned a value outside the "
                    "SamFramePrediction contract"
                ) from exc
        if (
            prediction.sequence_id != request.session.sequence_id
            or prediction.session_index != request.session.session_index
        ):
            raise RuntimeError(
                "mask propagation returned a prediction from a different session: "
                f"expected={(request.session.sequence_id, request.session.session_index)!r}, "
                f"actual={(prediction.sequence_id, prediction.session_index)!r}"
            )
        seed = by_local_id.get(int(prediction.object_id))
        if seed is None:
            raise RuntimeError(
                f"mask propagation returned unknown local object {prediction.object_id}"
            )
        key = int(prediction.frame_index), int(prediction.object_id)
        if key in raw_prediction_keys:
            raise RuntimeError(f"duplicate canonical propagation prediction {key}")
        raw_prediction_keys.add(key)
        if key[0] not in expected_frames:
            raise RuntimeError(
                f"propagation returned frames outside its direction: {[key[0]]}"
            )
        raw_mask = np.asarray(prediction.binary_mask, dtype=np.bool_)
        if raw_mask.shape != seed.mask.shape:
            raise RuntimeError(
                f"mask propagation returned mask shape {raw_mask.shape}; "
                f"expected seed mask shape {seed.mask.shape}"
            )
        filled = np.ascontiguousarray(fill_mask(prediction.binary_mask), dtype=np.bool_)
        if filled.shape != raw_mask.shape:
            raise RuntimeError(
                f"prediction hole filling changed mask shape from "
                f"{raw_mask.shape} to {filled.shape}"
            )
        added = int(np.count_nonzero(filled & ~raw_mask))
        prediction_added_pixels += added
        canonical = SamFramePrediction(
            sequence_id=prediction.sequence_id,
            session_index=prediction.session_index,
            frame_index=prediction.frame_index,
            object_id=seed.object_id,
            initial_detection_score=prediction.initial_detection_score,
            frame_tracker_score=prediction.frame_tracker_score,
            binary_mask=filled,
        )
        # Never publish the tracker copy at the hard-positive frame.  It is
        # emitted exactly once below from the filled authoritative seed, so a
        # drifting prompt prediction cannot leak false positives into an
        # irreversible streaming union.
        if canonical.frame_index == request.prompt_frame:
            return
        if canonical.frame_tracker_score is None:
            raise RuntimeError(
                "tracker-only propagation returned no probability for a "
                "non-authoritative prediction"
            )
        if float(canonical.frame_tracker_score) < request.conf:
            below_confidence_prediction_count += 1
            return
        item = LtaObjectPrediction(
            lineage=seed.lineage,
            prediction=canonical,
            hole_fill_added_pixels=added,
            source_provenance=seed.provenance,
        )
        canonical_prediction_count += 1
        if canonical.frame_index in boundary_frame_set:
            boundary_predictions[key] = item
        if retain_predictions:
            retained[key] = item
        if prediction_callback is not None:
            prediction_callback(item)
            callback_prediction_count += 1

    adapter_kwargs: dict[str, object] = {
        "resource": resource,
        "session": request.session,
        "prompt_frame": request.prompt_frame,
        "ground_truth": ground_truth,
        "seed": None,
        "object_masks": tuple(item.mask for item in filled_seeds),
        "conf": request.conf,
        "propagation_mode": "tracker-only",
        "empty_frame_limit": request.empty_frame_limit,
        "propagation_direction": request.direction,
        "seed_roundtrip_policy": "overlap-aware",
    }
    streaming_adapter = prediction_callback is not None or not retain_predictions
    if streaming_adapter:
        adapter_kwargs.update(
            {
                "prediction_callback": consume_raw_prediction,
                "retain_predictions": False,
            }
        )
    result = adapter(
        measured_predictor,
        raw_predictor,
        **adapter_kwargs,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("mask propagation adapter returned a non-mapping result")
    if (
        result.get("seed_roundtrip_policy") != "overlap-aware"
        or result.get("seed_roundtrip_passed") is not True
    ):
        raise RuntimeError(
            "production mask propagation failed its immediate overlap-aware seed "
            "round-trip contract: "
            f"policy={result.get('seed_roundtrip_policy')!r}, "
            f"passed={result.get('seed_roundtrip_passed')!r}, "
            f"objects={result.get('seed_roundtrip_object_audit')!r}"
        )
    # The streaming adapter normally returns no masks.  Consuming any returned
    # values preserves compatibility with adapters that only implement the
    # retained contract, without ever copying the complete tuple into a list.
    for prediction in result.get("propagation", ()):
        consume_raw_prediction(prediction)
    receipt = {
        key: value
        for key, value in result.items()
        if key not in {"propagation", "iterations"}
    }
    result = None

    # The filled injection geometry is authoritative for this session even
    # when a directional tracker call does not revisit the prompt or drifts.
    # Only original annotations receive confidence 1.0; dogfood and spatial
    # relays retain their carried tracker probability so a session boundary
    # cannot promote weak evidence through the spatial-relay gate.
    for seed in filled_seeds:
        prompt_probability = (
            1.0
            if seed.provenance is LtaSeedProvenance.AUTHORITATIVE
            else float(seed.tracker_probability)
        )
        canonical = SamFramePrediction(
            sequence_id=request.session.sequence_id,
            session_index=request.session.session_index,
            frame_index=request.prompt_frame,
            object_id=seed.object_id,
            initial_detection_score=1.0,
            frame_tracker_score=prompt_probability,
            binary_mask=seed.mask,
        )
        key = request.prompt_frame, seed.object_id
        item = LtaObjectPrediction(
            lineage=seed.lineage,
            prediction=canonical,
            hole_fill_added_pixels=int(
                seed.source_receipt.get("injection_hole_fill_added_pixels", 0)
            ),
            source_provenance=seed.provenance,
        )
        canonical_prediction_count += 1
        if request.prompt_frame in boundary_frame_set:
            boundary_predictions[key] = item
        if retain_predictions:
            retained[key] = item
        if prediction_callback is not None:
            prediction_callback(item)
            callback_prediction_count += 1
    ordered = tuple(
        retained[key]
        for key in sorted(retained, key=lambda item: (item[0], item[1]))
    )
    dogfood_values: list[LtaMaskSeed] = []
    for boundary_frame in dict.fromkeys(boundary_frames):
        boundary = tuple(
            boundary_predictions[key]
            for key in sorted(boundary_predictions)
            if key[0] == boundary_frame
            and bool(
                np.asarray(
                    boundary_predictions[key].prediction.binary_mask,
                    dtype=bool,
                ).any()
            )
        )
        dogfood_values.extend(
            LtaMaskSeed(
                lineage=item.lineage,
                frame_index=boundary_frame,
                object_id=item.prediction.object_id,
                mask=fill_mask(item.prediction.binary_mask),
                provenance=LtaSeedProvenance.TEMPORAL_DOGFOOD,
                tracker_probability=(
                    item.prediction.frame_tracker_score
                    if item.prediction.frame_tracker_score is not None
                    else item.prediction.initial_detection_score
                ),
                relay_generation=request.relay_generation,
                visited_tile_indices=next(
                    seed.visited_tile_indices
                    for seed in filled_seeds
                    if seed.lineage == item.lineage
                ),
                source_receipt={
                    "source_work_id": request.work_id,
                    "source_frame": boundary_frame,
                    "source_hole_fill_added_pixels": item.hole_fill_added_pixels,
                },
            )
            for item in boundary
        )
    dogfood = tuple(dogfood_values)
    receipt.update(
        {
            "conditioning": "authoritative_mask_injection",
            "direction": request.direction,
            "seed_count": len(filled_seeds),
            "model_object_capacity": resolve_mask_seed_capacity(len(filled_seeds)),
            "seed_hole_fill_added_pixels": seed_added_pixels,
            "prediction_hole_fill_added_pixels": prediction_added_pixels,
            "injected_prompt_frame": request.prompt_frame,
            "prompt_provenance": [seed.provenance.value for seed in filled_seeds],
            "canonical_prediction_count": canonical_prediction_count,
            "canonical_predictions_retained": retain_predictions,
            "canonical_callback_prediction_count": callback_prediction_count,
            "production_below_confidence_prediction_count": (
                below_confidence_prediction_count
            ),
            "production_confidence_threshold": request.conf,
            "production_confidence_comparison": (
                "frame_tracker_score >= threshold; authoritative prompt reinjected "
                "independently"
            ),
            "anchor_integrity_disposition": (
                "diagnostic_only_authoritative_prompt_restored"
            ),
        }
    )
    return LtaPropagationResult(
        request=request,
        predictions=ordered,
        dogfood_seeds=dogfood,
        adapter_receipt=receipt,
        hole_fill_added_pixels=seed_added_pixels + prediction_added_pixels,
    )


__all__ = (
    "LtaMaskSeed",
    "LtaObjectPrediction",
    "LtaPropagationRequest",
    "LtaPropagationResult",
    "LtaSeedProvenance",
    "LtaSeedArtifact",
    "partition_mask_seed_sessions",
    "read_seed_artifact",
    "run_mask_injected_session",
    "write_seed_artifact",
)
