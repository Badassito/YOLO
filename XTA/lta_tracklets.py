"""CPU-only cross-anchor reconciliation for LTA tracklets.

The SAM tracking sessions assign object ids locally.  This module deliberately
keeps those ids anchor-scoped, measures opposing tracklets in a shared temporal
band, and produces conservative one-to-one identity hypotheses.  It does not
mutate a tracker session and it does not treat an object missing from the next
anchor as evidence that the object ended.

NumPy and SciPy are imported inside numerical entry points so importing this
module remains cheap and does not initialize any accelerator runtime.
"""

from __future__ import annotations

import itertools
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


def _numpy():
    import numpy as np

    return np


def _as_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(result)


@dataclass(frozen=True, order=True)
class TrackletKey:
    """Stable identity of one tracker object within one tile and anchor."""

    tile_index: int
    anchor_frame: int
    local_object_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tile_index",
            _as_nonnegative_int(self.tile_index, name="tile_index"),
        )
        object.__setattr__(
            self,
            "anchor_frame",
            _as_nonnegative_int(self.anchor_frame, name="anchor_frame"),
        )
        object.__setattr__(
            self,
            "local_object_id",
            _as_nonnegative_int(self.local_object_id, name="local_object_id"),
        )


@dataclass(frozen=True)
class TrackletFrame:
    """One binary object proposal and its canonical tracker probability."""

    frame_index: int
    mask: Any = field(repr=False, compare=False)
    tracker_probability: float
    authoritative: bool = False

    def __post_init__(self) -> None:
        np = _numpy()
        frame_index = _as_nonnegative_int(self.frame_index, name="frame_index")
        probability = float(self.tracker_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("tracker_probability must be finite and in [0,1]")
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or any(int(size) < 1 for size in mask.shape):
            raise ValueError("tracklet masks must be nonempty two-dimensional arrays")
        mask = np.ascontiguousarray(mask, dtype=np.bool_).copy()
        authoritative = bool(self.authoritative)
        if authoritative and not bool(np.any(mask)):
            raise ValueError("an authoritative tracklet frame must have a nonempty mask")
        mask.setflags(write=False)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "tracker_probability", probability)
        object.__setattr__(self, "authoritative", authoritative)

    @property
    def active(self) -> bool:
        np = _numpy()
        return bool(
            self.authoritative
            or (self.tracker_probability > 0.5 and np.any(self.mask))
        )


@dataclass(frozen=True)
class Tracklet:
    """A temporally ordered tracker object whose id is scoped to its anchor."""

    tile_index: int
    anchor_frame: int
    local_object_id: int
    frames: tuple[TrackletFrame, ...]

    def __post_init__(self) -> None:
        key = TrackletKey(
            tile_index=self.tile_index,
            anchor_frame=self.anchor_frame,
            local_object_id=self.local_object_id,
        )
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("a tracklet requires at least one frame")
        if not all(isinstance(item, TrackletFrame) for item in frames):
            raise TypeError("tracklet frames must be TrackletFrame instances")
        indexes = tuple(int(item.frame_index) for item in frames)
        if tuple(sorted(indexes)) != indexes or len(set(indexes)) != len(indexes):
            raise ValueError("tracklet frame indexes must be unique and increasing")
        shape = tuple(int(value) for value in frames[0].mask.shape)
        if any(tuple(int(value) for value in item.mask.shape) != shape for item in frames):
            raise ValueError("all masks in a tracklet must have the same shape")
        object.__setattr__(self, "tile_index", key.tile_index)
        object.__setattr__(self, "anchor_frame", key.anchor_frame)
        object.__setattr__(self, "local_object_id", key.local_object_id)
        object.__setattr__(self, "frames", frames)
    @property
    def key(self) -> TrackletKey:
        return TrackletKey(self.tile_index, self.anchor_frame, self.local_object_id)

    @property
    def mask_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.frames[0].mask.shape)

    def frame_map(self) -> dict[int, TrackletFrame]:
        return {int(item.frame_index): item for item in self.frames}


@dataclass(frozen=True)
class OverlapMetrics:
    """Geometry, activity, and confidence agreement in a shared frame band."""

    shared_frame_indices: tuple[int, ...]
    common_frame_count: int
    useful_frame_count: int
    coactive_frame_count: int
    coactive_coverage: float
    voxel_intersection: int
    voxel_union: int
    iou_3d: float
    dice_3d: float
    frame_iou_mean: float | None
    frame_iou_median: float | None
    frame_iou_p10: float | None
    frame_iou_min: float | None
    frame_iou_max: float | None
    centroid_distance_mean: float | None
    centroid_distance_median: float | None
    centroid_distance_p95: float | None
    area_log_drift_mean: float | None
    area_log_drift_median: float | None
    area_log_drift_p95: float | None
    left_tracker_probability_mean: float | None
    right_tracker_probability_mean: float | None
    tracker_probability_mean: float | None
    tracker_probability_min: float | None


@dataclass(frozen=True)
class MatchingConfig:
    """Conservative gates and weights for adjacent-anchor assignment."""

    min_common_frames: int = 3
    min_useful_frames: int = 3
    min_coactive_frames: int = 3
    min_iou_3d: float = 0.20
    min_frame_iou_median: float = 0.15
    min_coactive_coverage: float = 0.50
    max_centroid_distance: float = 0.15
    max_area_log_drift: float = math.log(4.0)
    min_tracker_probability: float = 0.50
    min_similarity: float = 0.45
    min_runner_up_margin: float = 0.08
    weight_iou_3d: float = 0.30
    weight_dice_3d: float = 0.10
    weight_frame_iou: float = 0.15
    weight_coactive_coverage: float = 0.15
    weight_centroid: float = 0.10
    weight_area: float = 0.05
    weight_tracker_probability: float = 0.15
    detect_residual_hypotheses: bool = True
    hypothesis_max_members: int = 3
    hypothesis_candidate_limit: int = 6
    min_union_iou_3d: float = 0.55
    min_union_iou_gain: float = 0.15
    min_union_similarity: float = 0.55

    def __post_init__(self) -> None:
        for name in (
            "min_common_frames",
            "min_useful_frames",
            "min_coactive_frames",
        ):
            value = _as_nonnegative_int(getattr(self, name), name=name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("hypothesis_max_members", "hypothesis_candidate_limit"):
            value = _as_nonnegative_int(getattr(self, name), name=name)
            if value < 2:
                raise ValueError(f"{name} must be at least two")
            object.__setattr__(self, name, value)
        unit_values = (
            "min_iou_3d",
            "min_frame_iou_median",
            "min_coactive_coverage",
            "max_centroid_distance",
            "min_tracker_probability",
            "min_similarity",
            "min_runner_up_margin",
            "min_union_iou_3d",
            "min_union_iou_gain",
            "min_union_similarity",
        )
        for name in unit_values:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
            object.__setattr__(self, name, value)
        drift = float(self.max_area_log_drift)
        if not math.isfinite(drift) or drift < 0.0:
            raise ValueError("max_area_log_drift must be finite and non-negative")
        object.__setattr__(self, "max_area_log_drift", drift)
        weights = (
            "weight_iou_3d",
            "weight_dice_3d",
            "weight_frame_iou",
            "weight_coactive_coverage",
            "weight_centroid",
            "weight_area",
            "weight_tracker_probability",
        )
        total = 0.0
        for name in weights:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
            total += value
        if total <= 0.0:
            raise ValueError("at least one matching weight must be positive")
        object.__setattr__(
            self,
            "detect_residual_hypotheses",
            bool(self.detect_residual_hypotheses),
        )


@dataclass(frozen=True)
class PairAssessment:
    left_key: TrackletKey
    right_key: TrackletKey
    metrics: OverlapMetrics
    similarity: float
    left_runner_up_margin: float
    right_runner_up_margin: float
    gate_failures: tuple[str, ...]
    assignment_candidate: bool


@dataclass(frozen=True)
class TrackletMatch:
    left_key: TrackletKey
    right_key: TrackletKey
    similarity: float
    left_runner_up_margin: float
    right_runner_up_margin: float
    metrics: OverlapMetrics


@dataclass(frozen=True)
class ResidualHypothesis:
    """Audit-only split/merge suggestion; never changes one-to-one matches."""

    kind: str
    left_keys: tuple[TrackletKey, ...]
    right_keys: tuple[TrackletKey, ...]
    metrics: OverlapMetrics
    similarity: float
    best_individual_iou_3d: float
    union_iou_gain: float

    def __post_init__(self) -> None:
        if self.kind not in {"split", "merge"}:
            raise ValueError("residual hypothesis kind must be 'split' or 'merge'")
        if self.kind == "split" and not (
            len(self.left_keys) == 1 and len(self.right_keys) >= 2
        ):
            raise ValueError("a split hypothesis requires one left and multiple right keys")
        if self.kind == "merge" and not (
            len(self.left_keys) >= 2 and len(self.right_keys) == 1
        ):
            raise ValueError("a merge hypothesis requires multiple left and one right key")


@dataclass(frozen=True)
class MatchingResult:
    left_anchor_frame: int
    right_anchor_frame: int
    matches: tuple[TrackletMatch, ...]
    unmatched_left_keys: tuple[TrackletKey, ...]
    unmatched_right_keys: tuple[TrackletKey, ...]
    pair_assessments: tuple[PairAssessment, ...]
    residual_hypotheses: tuple[ResidualHypothesis, ...]
    assignment_backend: str = "unspecified"


@dataclass(frozen=True)
class HandoffConfig:
    midpoint_prior_strength: float = 0.35
    empty_choice_penalty: float = 20.0
    probability_floor: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("midpoint_prior_strength", "empty_choice_penalty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        floor = float(self.probability_floor)
        if not math.isfinite(floor) or not 0.0 < floor < 0.5:
            raise ValueError("probability_floor must be finite and in (0,0.5)")
        object.__setattr__(self, "probability_floor", floor)


@dataclass(frozen=True)
class HandoffFrameChoice:
    frame_index: int
    source: str
    selection_reason: str
    left_probability: float
    right_probability: float
    left_active: bool
    right_active: bool


@dataclass(frozen=True)
class HandoffDecision:
    left_key: TrackletKey
    right_key: TrackletKey
    nominal_midpoint: float
    cut_coordinate: float
    left_last_shared_frame: int | None
    right_first_shared_frame: int | None
    objective: float
    choices: tuple[HandoffFrameChoice, ...]


_COMPOSITION_SELECTION_REASONS = frozenset(
    {
        "monotonic_cut",
        "sparse_fallback",
        "authoritative_cut",
        "authoritative_override",
        "coincident_authoritative_equal",
        "unmatched_verbatim",
    }
)


@dataclass(frozen=True)
class ComposedFrame:
    frame_index: int
    mask: Any = field(repr=False, compare=False)
    source_key: TrackletKey
    tracker_probability: float
    authoritative: bool
    selection_reason: str

    def __post_init__(self) -> None:
        np = _numpy()
        frame_index = _as_nonnegative_int(self.frame_index, name="frame_index")
        mask = np.ascontiguousarray(np.asarray(self.mask), dtype=np.bool_).copy()
        if mask.ndim != 2 or any(int(size) < 1 for size in mask.shape):
            raise ValueError("composed masks must be nonempty two-dimensional arrays")
        mask.setflags(write=False)
        probability = float(self.tracker_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("tracker_probability must be finite and in [0,1]")
        reason = str(self.selection_reason)
        if reason not in _COMPOSITION_SELECTION_REASONS:
            raise ValueError(f"unknown composed-frame selection reason: {reason}")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "tracker_probability", probability)
        object.__setattr__(self, "authoritative", bool(self.authoritative))
        object.__setattr__(self, "selection_reason", reason)


@dataclass(frozen=True)
class ComposedLineage:
    lineage_id: str
    status: str
    member_keys: tuple[TrackletKey, ...]
    frames: tuple[ComposedFrame, ...]
    handoff: HandoffDecision | None = None

    def __post_init__(self) -> None:
        if self.status not in {"matched", "left_unmatched", "right_unmatched"}:
            raise ValueError("unknown composed-lineage status")
        if not self.frames:
            raise ValueError("a composed lineage requires at least one frame")


@dataclass(frozen=True)
class UnionFrame:
    frame_index: int
    mask: Any = field(repr=False, compare=False)
    contributing_lineage_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        np = _numpy()
        frame_index = _as_nonnegative_int(self.frame_index, name="frame_index")
        mask = np.ascontiguousarray(np.asarray(self.mask), dtype=np.bool_).copy()
        if mask.ndim != 2 or any(int(size) < 1 for size in mask.shape):
            raise ValueError("union masks must be nonempty two-dimensional arrays")
        mask.setflags(write=False)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(
            self,
            "contributing_lineage_ids",
            tuple(str(value) for value in self.contributing_lineage_ids),
        )


@dataclass(frozen=True)
class BinaryComposition:
    lineages: tuple[ComposedLineage, ...]
    union_frames: tuple[UnionFrame, ...]


@dataclass(frozen=True)
class _FrameSummary:
    frame: TrackletFrame
    area: int
    centroid_yx: tuple[float, float] | None


def _summarize_frames(frames: Mapping[int, TrackletFrame]) -> dict[int, _FrameSummary]:
    np = _numpy()
    output: dict[int, _FrameSummary] = {}
    for frame_index, item in frames.items():
        rows, columns = np.nonzero(item.mask)
        area = int(rows.size)
        centroid = None
        if area:
            centroid = (float(rows.mean()), float(columns.mean()))
        output[int(frame_index)] = _FrameSummary(item, area, centroid)
    return output


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    np = _numpy()
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _compute_overlap_from_maps(
    left: Mapping[int, TrackletFrame],
    right: Mapping[int, TrackletFrame],
    *,
    frame_start: int | None,
    frame_stop: int | None,
) -> OverlapMetrics:
    np = _numpy()
    shared = sorted(set(left) & set(right))
    if frame_start is not None:
        shared = [value for value in shared if value >= int(frame_start)]
    if frame_stop is not None:
        shared = [value for value in shared if value < int(frame_stop)]
    left_summary = _summarize_frames({value: left[value] for value in shared})
    right_summary = _summarize_frames({value: right[value] for value in shared})
    if shared:
        left_shape = tuple(int(value) for value in left[shared[0]].mask.shape)
        right_shape = tuple(int(value) for value in right[shared[0]].mask.shape)
        if left_shape != right_shape:
            raise ValueError(f"overlap mask shapes differ: {left_shape} != {right_shape}")
        diagonal = math.hypot(*left_shape)
    else:
        diagonal = 1.0

    intersection_total = 0
    area_left_total = 0
    area_right_total = 0
    useful = 0
    coactive = 0
    frame_ious: list[float] = []
    centroid_distances: list[float] = []
    area_drifts: list[float] = []
    left_probabilities: list[float] = []
    right_probabilities: list[float] = []
    joint_probabilities: list[float] = []
    for frame_index in shared:
        left_item = left_summary[frame_index]
        right_item = right_summary[frame_index]
        left_area = int(left_item.area)
        right_area = int(right_item.area)
        intersection = int(np.count_nonzero(left_item.frame.mask & right_item.frame.mask))
        union = left_area + right_area - intersection
        area_left_total += left_area
        area_right_total += right_area
        intersection_total += intersection
        if union <= 0:
            continue
        useful += 1
        frame_ious.append(float(intersection / union))
        left_probability = float(left_item.frame.tracker_probability)
        right_probability = float(right_item.frame.tracker_probability)
        left_probabilities.append(left_probability)
        right_probabilities.append(right_probability)
        joint_probabilities.append(math.sqrt(left_probability * right_probability))
        if left_area and right_area:
            coactive += 1
            left_centroid = left_item.centroid_yx
            right_centroid = right_item.centroid_yx
            if left_centroid is None or right_centroid is None:
                raise RuntimeError("active overlap frame has no centroid")
            distance = math.hypot(
                left_centroid[0] - right_centroid[0],
                left_centroid[1] - right_centroid[1],
            ) / diagonal
            centroid_distances.append(float(distance))
            area_drifts.append(abs(math.log(float(left_area) / float(right_area))))

    union_total = area_left_total + area_right_total - intersection_total
    denominator = area_left_total + area_right_total
    iou_3d = float(intersection_total / union_total) if union_total else 0.0
    dice_3d = float(2 * intersection_total / denominator) if denominator else 0.0
    return OverlapMetrics(
        shared_frame_indices=tuple(shared),
        common_frame_count=len(shared),
        useful_frame_count=useful,
        coactive_frame_count=coactive,
        coactive_coverage=float(coactive / useful) if useful else 0.0,
        voxel_intersection=intersection_total,
        voxel_union=union_total,
        iou_3d=iou_3d,
        dice_3d=dice_3d,
        frame_iou_mean=(float(sum(frame_ious) / len(frame_ious)) if frame_ious else None),
        frame_iou_median=_percentile(frame_ious, 50.0),
        frame_iou_p10=_percentile(frame_ious, 10.0),
        frame_iou_min=(float(min(frame_ious)) if frame_ious else None),
        frame_iou_max=(float(max(frame_ious)) if frame_ious else None),
        centroid_distance_mean=(
            float(sum(centroid_distances) / len(centroid_distances))
            if centroid_distances
            else None
        ),
        centroid_distance_median=_percentile(centroid_distances, 50.0),
        centroid_distance_p95=_percentile(centroid_distances, 95.0),
        area_log_drift_mean=(
            float(sum(area_drifts) / len(area_drifts)) if area_drifts else None
        ),
        area_log_drift_median=_percentile(area_drifts, 50.0),
        area_log_drift_p95=_percentile(area_drifts, 95.0),
        left_tracker_probability_mean=(
            float(sum(left_probabilities) / len(left_probabilities))
            if left_probabilities
            else None
        ),
        right_tracker_probability_mean=(
            float(sum(right_probabilities) / len(right_probabilities))
            if right_probabilities
            else None
        ),
        tracker_probability_mean=(
            float(sum(joint_probabilities) / len(joint_probabilities))
            if joint_probabilities
            else None
        ),
        tracker_probability_min=(
            float(min(joint_probabilities)) if joint_probabilities else None
        ),
    )


def compute_overlap_metrics(
    left: Tracklet,
    right: Tracklet,
    *,
    frame_start: int | None = None,
    frame_stop: int | None = None,
) -> OverlapMetrics:
    """Measure two anchor-scoped tracklets on their common frames."""

    if not isinstance(left, Tracklet) or not isinstance(right, Tracklet):
        raise TypeError("left and right must be Tracklet instances")
    if int(left.tile_index) != int(right.tile_index):
        raise ValueError("cross-tile tracklets cannot be reconciled")
    if frame_start is not None:
        frame_start = _as_nonnegative_int(frame_start, name="frame_start")
    if frame_stop is not None:
        frame_stop = _as_nonnegative_int(frame_stop, name="frame_stop")
    if frame_start is not None and frame_stop is not None and frame_stop <= frame_start:
        raise ValueError("frame_stop must be greater than frame_start")
    return _compute_overlap_from_maps(
        left.frame_map(),
        right.frame_map(),
        frame_start=frame_start,
        frame_stop=frame_stop,
    )


def _metric_similarity(metrics: OverlapMetrics, config: MatchingConfig) -> float:
    centroid = metrics.centroid_distance_median
    area = metrics.area_log_drift_median
    frame_iou = metrics.frame_iou_median
    probability = metrics.tracker_probability_mean
    centroid_similarity = 0.0 if centroid is None else math.exp(-4.0 * centroid)
    area_similarity = 0.0 if area is None else math.exp(-area)
    values = (
        (config.weight_iou_3d, metrics.iou_3d),
        (config.weight_dice_3d, metrics.dice_3d),
        (config.weight_frame_iou, 0.0 if frame_iou is None else frame_iou),
        (config.weight_coactive_coverage, metrics.coactive_coverage),
        (config.weight_centroid, centroid_similarity),
        (config.weight_area, area_similarity),
        (
            config.weight_tracker_probability,
            0.0 if probability is None else probability,
        ),
    )
    weight_total = sum(weight for weight, _value in values)
    return float(sum(weight * value for weight, value in values) / weight_total)


def _geometry_gate_failures(
    metrics: OverlapMetrics,
    config: MatchingConfig,
) -> tuple[str, ...]:
    failures: list[str] = []
    if metrics.common_frame_count < config.min_common_frames:
        failures.append("insufficient_common_frames")
    if metrics.useful_frame_count < config.min_useful_frames:
        failures.append("insufficient_useful_frames")
    if metrics.coactive_frame_count < config.min_coactive_frames:
        failures.append("insufficient_coactive_frames")
    if metrics.iou_3d < config.min_iou_3d:
        failures.append("iou_3d_below_minimum")
    if (
        metrics.frame_iou_median is None
        or metrics.frame_iou_median < config.min_frame_iou_median
    ):
        failures.append("frame_iou_below_minimum")
    if metrics.coactive_coverage < config.min_coactive_coverage:
        failures.append("coactive_coverage_below_minimum")
    if (
        metrics.centroid_distance_median is None
        or metrics.centroid_distance_median > config.max_centroid_distance
    ):
        failures.append("centroid_distance_above_maximum")
    if (
        metrics.area_log_drift_median is None
        or metrics.area_log_drift_median > config.max_area_log_drift
    ):
        failures.append("area_drift_above_maximum")
    if (
        metrics.tracker_probability_mean is None
        or metrics.tracker_probability_mean < config.min_tracker_probability
    ):
        failures.append("tracker_probability_below_minimum")
    return tuple(failures)


def _validate_adjacent_sides(
    left_tracklets: Sequence[Tracklet],
    right_tracklets: Sequence[Tracklet],
) -> tuple[tuple[Tracklet, ...], tuple[Tracklet, ...]]:
    left = tuple(left_tracklets)
    right = tuple(right_tracklets)
    if not left or not right:
        raise ValueError("matching requires nonempty left and right tracklet sets")
    if not all(isinstance(item, Tracklet) for item in (*left, *right)):
        raise TypeError("matching inputs must contain only Tracklet instances")
    tiles = {item.tile_index for item in (*left, *right)}
    left_anchors = {item.anchor_frame for item in left}
    right_anchors = {item.anchor_frame for item in right}
    if len(tiles) != 1:
        raise ValueError("all reconciliation tracklets must belong to one tile")
    if len(left_anchors) != 1 or len(right_anchors) != 1:
        raise ValueError("each reconciliation side must belong to exactly one anchor")
    left_anchor = next(iter(left_anchors))
    right_anchor = next(iter(right_anchors))
    if left_anchor >= right_anchor:
        raise ValueError("left anchor must precede right anchor")
    keys = [item.key for item in (*left, *right)]
    if len(keys) != len(set(keys)):
        raise ValueError("tracklet keys must be unique")
    shapes = {item.mask_shape for item in (*left, *right)}
    if len(shapes) != 1:
        raise ValueError("all reconciliation masks must have one common shape")
    return tuple(sorted(left, key=lambda item: item.key)), tuple(
        sorted(right, key=lambda item: item.key)
    )


def _union_tracklet_frames(tracklets: Sequence[Tracklet]) -> dict[int, TrackletFrame]:
    np = _numpy()
    grouped: dict[int, list[TrackletFrame]] = {}
    for tracklet in tracklets:
        for item in tracklet.frames:
            grouped.setdefault(int(item.frame_index), []).append(item)
    output: dict[int, TrackletFrame] = {}
    for frame_index, items in grouped.items():
        union = np.zeros(items[0].mask.shape, dtype=np.bool_)
        geometric_contributors = [item for item in items if bool(np.any(item.mask))]
        for item in geometric_contributors:
            union |= item.mask
        output[frame_index] = TrackletFrame(
            frame_index=frame_index,
            mask=union,
            # Confidence must come from an object that supplied geometry.  An
            # empty high-confidence member cannot lend its score to another
            # member's low-confidence mask.
            tracker_probability=max(
                (item.tracker_probability for item in geometric_contributors),
                default=0.0,
            ),
            authoritative=any(item.authoritative for item in items),
        )
    return output


def _residual_hypotheses(
    left_by_key: Mapping[TrackletKey, Tracklet],
    right_by_key: Mapping[TrackletKey, Tracklet],
    unmatched_left: Sequence[TrackletKey],
    unmatched_right: Sequence[TrackletKey],
    matches: Sequence[TrackletMatch],
    assessments: Mapping[tuple[TrackletKey, TrackletKey], PairAssessment],
    config: MatchingConfig,
    *,
    frame_start: int | None,
    frame_stop: int | None,
) -> tuple[ResidualHypothesis, ...]:
    if not config.detect_residual_hypotheses:
        return ()
    hypotheses: list[ResidualHypothesis] = []
    matched_right_by_left = {item.left_key: item.right_key for item in matches}
    matched_left_by_right = {item.right_key: item.left_key for item in matches}
    unmatched_left_set = set(unmatched_left)
    unmatched_right_set = set(unmatched_right)

    # An accepted one-to-one edge may be one member of a real split.  Include
    # that accepted counterpart alongside residual right objects for the
    # audit-only union test; the primary assignment remains unchanged.
    for left_key in left_by_key:
        accepted_right = matched_right_by_left.get(left_key)
        right_pool = list(unmatched_right)
        if accepted_right is not None:
            right_pool.append(accepted_right)
        if len(right_pool) < 2:
            continue
        ranked_right = sorted(
            right_pool,
            key=lambda key: assessments[(left_key, key)].similarity,
            reverse=True,
        )[: config.hypothesis_candidate_limit]
        if accepted_right is not None and accepted_right not in ranked_right:
            ranked_right[-1] = accepted_right
        best: ResidualHypothesis | None = None
        for size in range(2, min(config.hypothesis_max_members, len(ranked_right)) + 1):
            for right_keys in itertools.combinations(ranked_right, size):
                if accepted_right is not None and accepted_right not in right_keys:
                    continue
                if not any(key in unmatched_right_set for key in right_keys):
                    continue
                metrics = _compute_overlap_from_maps(
                    left_by_key[left_key].frame_map(),
                    _union_tracklet_frames([right_by_key[key] for key in right_keys]),
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                )
                similarity = _metric_similarity(metrics, config)
                individual = max(
                    assessments[(left_key, key)].metrics.iou_3d for key in right_keys
                )
                gain = float(metrics.iou_3d - individual)
                if (
                    _geometry_gate_failures(metrics, config)
                    or metrics.iou_3d < config.min_union_iou_3d
                    or gain < config.min_union_iou_gain
                    or similarity < config.min_union_similarity
                ):
                    continue
                candidate = ResidualHypothesis(
                    kind="split",
                    left_keys=(left_key,),
                    right_keys=tuple(right_keys),
                    metrics=metrics,
                    similarity=similarity,
                    best_individual_iou_3d=float(individual),
                    union_iou_gain=gain,
                )
                if best is None or (
                    candidate.metrics.iou_3d,
                    candidate.similarity,
                    -len(candidate.right_keys),
                ) > (
                    best.metrics.iou_3d,
                    best.similarity,
                    -len(best.right_keys),
                ):
                    best = candidate
        if best is not None:
            hypotheses.append(best)

    # Symmetrically, an accepted left counterpart may be one member of a merge
    # completed by one or more residual left objects.
    for right_key in right_by_key:
        accepted_left = matched_left_by_right.get(right_key)
        left_pool = list(unmatched_left)
        if accepted_left is not None:
            left_pool.append(accepted_left)
        if len(left_pool) < 2:
            continue
        ranked_left = sorted(
            left_pool,
            key=lambda key: assessments[(key, right_key)].similarity,
            reverse=True,
        )[: config.hypothesis_candidate_limit]
        if accepted_left is not None and accepted_left not in ranked_left:
            ranked_left[-1] = accepted_left
        best = None
        for size in range(2, min(config.hypothesis_max_members, len(ranked_left)) + 1):
            for left_keys in itertools.combinations(ranked_left, size):
                if accepted_left is not None and accepted_left not in left_keys:
                    continue
                if not any(key in unmatched_left_set for key in left_keys):
                    continue
                metrics = _compute_overlap_from_maps(
                    _union_tracklet_frames([left_by_key[key] for key in left_keys]),
                    right_by_key[right_key].frame_map(),
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                )
                similarity = _metric_similarity(metrics, config)
                individual = max(
                    assessments[(key, right_key)].metrics.iou_3d for key in left_keys
                )
                gain = float(metrics.iou_3d - individual)
                if (
                    _geometry_gate_failures(metrics, config)
                    or metrics.iou_3d < config.min_union_iou_3d
                    or gain < config.min_union_iou_gain
                    or similarity < config.min_union_similarity
                ):
                    continue
                candidate = ResidualHypothesis(
                    kind="merge",
                    left_keys=tuple(left_keys),
                    right_keys=(right_key,),
                    metrics=metrics,
                    similarity=similarity,
                    best_individual_iou_3d=float(individual),
                    union_iou_gain=gain,
                )
                if best is None or (
                    candidate.metrics.iou_3d,
                    candidate.similarity,
                    -len(candidate.left_keys),
                ) > (
                    best.metrics.iou_3d,
                    best.similarity,
                    -len(best.left_keys),
                ):
                    best = candidate
        if best is not None:
            hypotheses.append(best)
    return tuple(
        sorted(
            hypotheses,
            key=lambda item: (item.kind, item.left_keys, item.right_keys),
        )
    )


def _hungarian_square_maximize(weights: Any) -> tuple[Any, Any]:
    """Return a deterministic maximum-weight assignment for a square matrix.

    This is the O(n^3) shortest-augmenting-path form of the Hungarian
    algorithm.  Columns are scanned in ascending order and equal reduced costs
    retain their first predecessor, providing deterministic tie resolution.
    Forbidden edges remain ordinary finite low weights so the caller's private
    dummy rows and columns participate exactly as they do under SciPy.
    """

    np = _numpy()
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or int(matrix.shape[0]) != int(matrix.shape[1]):
        raise ValueError("assignment weights must be a square matrix")
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError("assignment weights must all be finite")
    size = int(matrix.shape[0])
    if size == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty.copy()
    costs = float(np.max(matrix)) - matrix
    if not bool(np.all(np.isfinite(costs))):
        raise ValueError("assignment cost conversion overflowed")

    # Potentials and matching use one-based indexes.  p[column] is the row
    # currently assigned to that column; column zero is the augmenting sentinel.
    row_potential = np.zeros(size + 1, dtype=np.float64)
    column_potential = np.zeros(size + 1, dtype=np.float64)
    matched_row = np.zeros(size + 1, dtype=np.int64)
    predecessor_column = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        matched_row[0] = row
        minimum_reduced_cost = np.full(size + 1, np.inf, dtype=np.float64)
        used_column = np.zeros(size + 1, dtype=np.bool_)
        column = 0
        while True:
            used_column[column] = True
            active_row = int(matched_row[column])
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used_column[candidate_column]:
                    continue
                reduced_cost = (
                    costs[active_row - 1, candidate_column - 1]
                    - row_potential[active_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum_reduced_cost[candidate_column]:
                    minimum_reduced_cost[candidate_column] = reduced_cost
                    predecessor_column[candidate_column] = column
                candidate_cost = minimum_reduced_cost[candidate_column]
                if candidate_cost < delta:
                    delta = float(candidate_cost)
                    next_column = candidate_column
            if next_column == 0 or not math.isfinite(delta):
                raise RuntimeError("Hungarian assignment could not find an augmenting edge")
            for candidate_column in range(size + 1):
                if used_column[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                elif candidate_column:
                    minimum_reduced_cost[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = int(predecessor_column[column])
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    row_indexes = np.arange(size, dtype=np.int64)
    column_indexes = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        row = int(matched_row[column])
        if not 1 <= row <= size:
            raise RuntimeError("Hungarian assignment produced an invalid row")
        column_indexes[row - 1] = column - 1
    if len(set(int(value) for value in column_indexes.tolist())) != size:
        raise RuntimeError("Hungarian assignment produced duplicate columns")
    return row_indexes, column_indexes


def _maximum_weight_assignment(weights: Any) -> tuple[Any, Any, str]:
    """Prefer SciPy and fall back only when its optimize API cannot import."""

    np = _numpy()
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or int(matrix.shape[0]) != int(matrix.shape[1]):
        raise ValueError("assignment weights must be a square matrix")
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError("assignment weights must all be finite")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        rows, columns = _hungarian_square_maximize(matrix)
        return rows, columns, "numpy_hungarian_fallback"
    rows, columns = linear_sum_assignment(-matrix)
    return rows, columns, "scipy_linear_sum_assignment"


def match_tracklets(
    left_tracklets: Sequence[Tracklet],
    right_tracklets: Sequence[Tracklet],
    *,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    config: MatchingConfig | None = None,
) -> MatchingResult:
    """Conservatively match adjacent-anchor tracklets with Hungarian assignment."""

    np = _numpy()
    config = MatchingConfig() if config is None else config
    if not isinstance(config, MatchingConfig):
        raise TypeError("config must be a MatchingConfig")
    left, right = _validate_adjacent_sides(left_tracklets, right_tracklets)
    if frame_start is not None:
        frame_start = _as_nonnegative_int(frame_start, name="frame_start")
    if frame_stop is not None:
        frame_stop = _as_nonnegative_int(frame_stop, name="frame_stop")
    if frame_start is not None and frame_stop is not None and frame_stop <= frame_start:
        raise ValueError("frame_stop must be greater than frame_start")

    preliminary: dict[tuple[int, int], tuple[OverlapMetrics, float, tuple[str, ...]]] = {}
    for left_index, left_item in enumerate(left):
        for right_index, right_item in enumerate(right):
            metrics = compute_overlap_metrics(
                left_item,
                right_item,
                frame_start=frame_start,
                frame_stop=frame_stop,
            )
            preliminary[(left_index, right_index)] = (
                metrics,
                _metric_similarity(metrics, config),
                _geometry_gate_failures(metrics, config),
            )

    assessments: list[PairAssessment] = []
    candidate_matrix = np.zeros((len(left), len(right)), dtype=np.bool_)
    similarity_matrix = np.zeros((len(left), len(right)), dtype=np.float64)
    for left_index, left_item in enumerate(left):
        for right_index, right_item in enumerate(right):
            metrics, similarity, geometry_failures = preliminary[(left_index, right_index)]
            row_alternatives = [
                preliminary[(left_index, other)][1]
                for other in range(len(right))
                if other != right_index and not preliminary[(left_index, other)][2]
            ]
            column_alternatives = [
                preliminary[(other, right_index)][1]
                for other in range(len(left))
                if other != left_index and not preliminary[(other, right_index)][2]
            ]
            left_margin = float(similarity - max(row_alternatives, default=0.0))
            right_margin = float(similarity - max(column_alternatives, default=0.0))
            failures = list(geometry_failures)
            if similarity < config.min_similarity:
                failures.append("similarity_below_minimum")
            if left_margin < config.min_runner_up_margin:
                failures.append("left_runner_up_margin_below_minimum")
            if right_margin < config.min_runner_up_margin:
                failures.append("right_runner_up_margin_below_minimum")
            candidate = not failures
            candidate_matrix[left_index, right_index] = candidate
            similarity_matrix[left_index, right_index] = similarity
            assessments.append(
                PairAssessment(
                    left_key=left_item.key,
                    right_key=right_item.key,
                    metrics=metrics,
                    similarity=float(similarity),
                    left_runner_up_margin=left_margin,
                    right_runner_up_margin=right_margin,
                    gate_failures=tuple(failures),
                    assignment_candidate=candidate,
                )
            )

    # Add one private dummy column per left object and one private dummy row per
    # right object.  This lets the assignment leave either side unmatched
    # instead of forcing a poor real edge to consume a useful column.
    size = len(left) + len(right)
    forbidden = -1.0e9
    weights = np.full((size, size), forbidden, dtype=np.float64)
    for left_index in range(len(left)):
        for right_index in range(len(right)):
            if candidate_matrix[left_index, right_index]:
                weights[left_index, right_index] = similarity_matrix[
                    left_index, right_index
                ]
        weights[left_index, len(right) + left_index] = 0.0
    for right_index in range(len(right)):
        weights[len(left) + right_index, right_index] = 0.0
    weights[len(left) :, len(right) :] = 0.0
    row_indexes, column_indexes, assignment_backend = _maximum_weight_assignment(weights)
    assessment_by_indexes = {
        (left_index, right_index): assessments[left_index * len(right) + right_index]
        for left_index in range(len(left))
        for right_index in range(len(right))
    }
    matches: list[TrackletMatch] = []
    for row_index, column_index in zip(row_indexes.tolist(), column_indexes.tolist()):
        if row_index >= len(left) or column_index >= len(right):
            continue
        assessment = assessment_by_indexes[(row_index, column_index)]
        if not assessment.assignment_candidate:
            continue
        matches.append(
            TrackletMatch(
                left_key=assessment.left_key,
                right_key=assessment.right_key,
                similarity=assessment.similarity,
                left_runner_up_margin=assessment.left_runner_up_margin,
                right_runner_up_margin=assessment.right_runner_up_margin,
                metrics=assessment.metrics,
            )
        )
    matches.sort(key=lambda item: (item.left_key, item.right_key))
    matched_left = {item.left_key for item in matches}
    matched_right = {item.right_key for item in matches}
    unmatched_left = tuple(item.key for item in left if item.key not in matched_left)
    unmatched_right = tuple(item.key for item in right if item.key not in matched_right)
    assessment_lookup = {
        (item.left_key, item.right_key): item for item in assessments
    }
    left_by_key = {item.key: item for item in left}
    right_by_key = {item.key: item for item in right}
    hypotheses = _residual_hypotheses(
        left_by_key,
        right_by_key,
        unmatched_left,
        unmatched_right,
        matches,
        assessment_lookup,
        config,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    return MatchingResult(
        left_anchor_frame=left[0].anchor_frame,
        right_anchor_frame=right[0].anchor_frame,
        matches=tuple(matches),
        unmatched_left_keys=unmatched_left,
        unmatched_right_keys=unmatched_right,
        pair_assessments=tuple(assessments),
        residual_hypotheses=hypotheses,
        assignment_backend=assignment_backend,
    )


def _select_composed_source(
    left_frame: TrackletFrame | None,
    right_frame: TrackletFrame | None,
    *,
    prefer_left: bool,
) -> tuple[TrackletFrame, str, str]:
    """Apply hard authority, then the cut preference, then sparse fallback."""

    np = _numpy()
    if left_frame is None and right_frame is None:
        raise RuntimeError("composed lineage frame unexpectedly has no proposal")
    if left_frame is not None and right_frame is not None:
        if left_frame.authoritative and right_frame.authoritative:
            if not bool(np.array_equal(left_frame.mask, right_frame.mask)):
                raise ValueError(
                    "matched tracklets contain conflicting authoritative masks "
                    f"on frame {left_frame.frame_index}"
                )
            selected = left_frame if prefer_left else right_frame
            source = "left" if prefer_left else "right"
            return selected, source, "coincident_authoritative_equal"
        if left_frame.authoritative:
            return (
                left_frame,
                "left",
                "authoritative_cut" if prefer_left else "authoritative_override",
            )
        if right_frame.authoritative:
            return (
                right_frame,
                "right",
                "authoritative_cut" if not prefer_left else "authoritative_override",
            )
        return (
            (left_frame, "left", "monotonic_cut")
            if prefer_left
            else (right_frame, "right", "monotonic_cut")
        )
    if left_frame is not None:
        return (
            left_frame,
            "left",
            "monotonic_cut" if prefer_left else "sparse_fallback",
        )
    if right_frame is None:  # pragma: no cover - guarded at entry
        raise RuntimeError("composed lineage frame unexpectedly has no proposal")
    return (
        right_frame,
        "right",
        "monotonic_cut" if not prefer_left else "sparse_fallback",
    )


def choose_monotonic_handoff(
    left: Tracklet,
    right: Tracklet,
    *,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    midpoint_frame: float | None = None,
    config: HandoffConfig | None = None,
) -> HandoffDecision:
    """Choose a monotonic shared-band preference cut.

    Hard-authoritative choices may override the preference on individual shared
    frames.  Sparse fallback outside the shared band is recorded later on each
    :class:`ComposedFrame`; the cut does not claim globally monotonic provenance.
    """

    if left.tile_index != right.tile_index:
        raise ValueError("handoff tracklets must belong to one tile")
    if left.anchor_frame >= right.anchor_frame:
        raise ValueError("handoff left anchor must precede the right anchor")
    config = HandoffConfig() if config is None else config
    if not isinstance(config, HandoffConfig):
        raise TypeError("config must be a HandoffConfig")
    left_map = left.frame_map()
    right_map = right.frame_map()
    shared = sorted(set(left_map) & set(right_map))
    if frame_start is not None:
        shared = [value for value in shared if value >= int(frame_start)]
    if frame_stop is not None:
        shared = [value for value in shared if value < int(frame_stop)]
    if not shared:
        raise ValueError("handoff tracklets have no shared frames in the requested band")
    midpoint = (
        (float(left.anchor_frame) + float(right.anchor_frame)) / 2.0
        if midpoint_frame is None
        else float(midpoint_frame)
    )
    if not math.isfinite(midpoint):
        raise ValueError("midpoint_frame must be finite")
    floor = config.probability_floor

    left_utilities: list[float] = []
    right_utilities: list[float] = []
    for frame_index in shared:
        left_frame = left_map[frame_index]
        right_frame = right_map[frame_index]
        if left_frame.authoritative != right_frame.authoritative:
            authoritative = left_frame if left_frame.authoritative else right_frame
            fixed_utility = math.log(
                max(floor, authoritative.tracker_probability)
            )
            # Composition must choose this hard mask regardless of the cut, so
            # keep its utility constant across every cut candidate as well.
            left_utilities.append(fixed_utility)
            right_utilities.append(fixed_utility)
            continue
        if not left_frame.active and not right_frame.active:
            left_utilities.append(0.0)
            right_utilities.append(0.0)
            continue
        left_utility = math.log(max(floor, left_frame.tracker_probability))
        right_utility = math.log(max(floor, right_frame.tracker_probability))
        if not left_frame.active and right_frame.active:
            left_utility -= config.empty_choice_penalty
        if not right_frame.active and left_frame.active:
            right_utility -= config.empty_choice_penalty
        left_utilities.append(left_utility)
        right_utilities.append(right_utility)

    left_prefix = [0.0]
    right_prefix = [0.0]
    for value in left_utilities:
        left_prefix.append(left_prefix[-1] + value)
    for value in right_utilities:
        right_prefix.append(right_prefix[-1] + value)
    right_total = right_prefix[-1]
    span = max(1.0, float(shared[-1] - shared[0] + 1))
    candidates: list[tuple[float, float, int]] = []
    # split_index is the number of shared frames sourced from the left.
    for split_index in range(len(shared) + 1):
        if split_index == 0:
            cut_coordinate = float(shared[0]) - 0.5
        elif split_index == len(shared):
            cut_coordinate = float(shared[-1]) + 0.5
        else:
            cut_coordinate = (
                float(shared[split_index - 1]) + float(shared[split_index])
            ) / 2.0
        objective = (
            left_prefix[split_index]
            + right_total
            - right_prefix[split_index]
            - config.midpoint_prior_strength
            * abs(cut_coordinate - midpoint)
            / span
        )
        candidates.append((float(objective), cut_coordinate, split_index))
    objective, cut_coordinate, split_index = max(
        candidates,
        key=lambda item: (
            item[0],
            -abs(item[1] - midpoint),
            -item[2],
        ),
    )
    choices_list: list[HandoffFrameChoice] = []
    for index, frame_index in enumerate(shared):
        left_frame = left_map[frame_index]
        right_frame = right_map[frame_index]
        _selected, source, reason = _select_composed_source(
            left_frame,
            right_frame,
            prefer_left=index < split_index,
        )
        choices_list.append(
            HandoffFrameChoice(
                frame_index=frame_index,
                source=source,
                selection_reason=reason,
                left_probability=left_frame.tracker_probability,
                right_probability=right_frame.tracker_probability,
                left_active=left_frame.active,
                right_active=right_frame.active,
            )
        )
    choices = tuple(choices_list)
    return HandoffDecision(
        left_key=left.key,
        right_key=right.key,
        nominal_midpoint=midpoint,
        cut_coordinate=cut_coordinate,
        left_last_shared_frame=(shared[split_index - 1] if split_index else None),
        right_first_shared_frame=(
            shared[split_index] if split_index < len(shared) else None
        ),
        objective=objective,
        choices=choices,
    )


def _copy_tracklet_lineage(
    tracklet: Tracklet,
    *,
    lineage_id: str,
    status: str,
) -> ComposedLineage:
    return ComposedLineage(
        lineage_id=lineage_id,
        status=status,
        member_keys=(tracklet.key,),
        frames=tuple(
            ComposedFrame(
                frame_index=item.frame_index,
                mask=item.mask,
                source_key=tracklet.key,
                tracker_probability=item.tracker_probability,
                authoritative=item.authoritative,
                selection_reason="unmatched_verbatim",
            )
            for item in tracklet.frames
        ),
    )


def compose_binary(
    left_tracklets: Sequence[Tracklet],
    right_tracklets: Sequence[Tracklet],
    matching: MatchingResult,
    *,
    handoff_config: HandoffConfig | None = None,
) -> BinaryComposition:
    """Compose matched lineages and preserve every unmatched proposal verbatim.

    A matched handoff is a monotonic source *preference* inside its measured
    shared band.  An authoritative mask always wins when both sides propose a
    frame, and a sole available proposal is retained as a sparse fallback even
    when it lies on the nonpreferred side of the cut.  Per-frame receipts make
    those deliberate exceptions explicit.
    """

    np = _numpy()
    left, right = _validate_adjacent_sides(left_tracklets, right_tracklets)
    if (
        matching.left_anchor_frame != left[0].anchor_frame
        or matching.right_anchor_frame != right[0].anchor_frame
    ):
        raise ValueError("matching result belongs to different anchors")
    left_by_key = {item.key: item for item in left}
    right_by_key = {item.key: item for item in right}
    matched_left_keys = tuple(item.left_key for item in matching.matches)
    matched_right_keys = tuple(item.right_key for item in matching.matches)
    unmatched_left_keys = tuple(matching.unmatched_left_keys)
    unmatched_right_keys = tuple(matching.unmatched_right_keys)
    if len(matched_left_keys) != len(set(matched_left_keys)):
        raise ValueError("matching result repeats a matched left key")
    if len(matched_right_keys) != len(set(matched_right_keys)):
        raise ValueError("matching result repeats a matched right key")
    if len(unmatched_left_keys) != len(set(unmatched_left_keys)):
        raise ValueError("matching result repeats an unmatched left key")
    if len(unmatched_right_keys) != len(set(unmatched_right_keys)):
        raise ValueError("matching result repeats an unmatched right key")
    if set(matched_left_keys) & set(unmatched_left_keys):
        raise ValueError("matching result marks a left key as both matched and unmatched")
    if set(matched_right_keys) & set(unmatched_right_keys):
        raise ValueError("matching result marks a right key as both matched and unmatched")
    if set(unmatched_left_keys) | set(matched_left_keys) != set(left_by_key):
        raise ValueError("matching result does not cover the supplied left tracklets")
    if set(unmatched_right_keys) | set(matched_right_keys) != set(right_by_key):
        raise ValueError("matching result does not cover the supplied right tracklets")

    lineages: list[ComposedLineage] = []
    for index, match in enumerate(matching.matches):
        left_item = left_by_key[match.left_key]
        right_item = right_by_key[match.right_key]
        shared = match.metrics.shared_frame_indices
        if not shared:
            raise RuntimeError("a matched pair has no shared handoff frames")
        handoff = choose_monotonic_handoff(
            left_item,
            right_item,
            frame_start=shared[0],
            frame_stop=shared[-1] + 1,
            config=handoff_config,
        )
        left_map = left_item.frame_map()
        right_map = right_item.frame_map()
        frames: list[ComposedFrame] = []
        for frame_index in sorted(set(left_map) | set(right_map)):
            prefer_left = float(frame_index) <= handoff.cut_coordinate
            selected, source, selection_reason = _select_composed_source(
                left_map.get(frame_index),
                right_map.get(frame_index),
                prefer_left=prefer_left,
            )
            selected_key = left_item.key if source == "left" else right_item.key
            frames.append(
                ComposedFrame(
                    frame_index=frame_index,
                    mask=selected.mask,
                    source_key=selected_key,
                    tracker_probability=selected.tracker_probability,
                    authoritative=selected.authoritative,
                    selection_reason=selection_reason,
                )
            )
        lineages.append(
            ComposedLineage(
                lineage_id=f"match_{index:03d}",
                status="matched",
                member_keys=(left_item.key, right_item.key),
                frames=tuple(frames),
                handoff=handoff,
            )
        )
    for index, key in enumerate(matching.unmatched_left_keys):
        lineages.append(
            _copy_tracklet_lineage(
                left_by_key[key],
                lineage_id=f"left_unmatched_{index:03d}",
                status="left_unmatched",
            )
        )
    for index, key in enumerate(matching.unmatched_right_keys):
        lineages.append(
            _copy_tracklet_lineage(
                right_by_key[key],
                lineage_id=f"right_unmatched_{index:03d}",
                status="right_unmatched",
            )
        )

    mask_shape = left[0].mask_shape
    by_frame: dict[int, list[tuple[str, ComposedFrame]]] = {}
    for lineage in lineages:
        for item in lineage.frames:
            by_frame.setdefault(item.frame_index, []).append((lineage.lineage_id, item))
    union_frames: list[UnionFrame] = []
    for frame_index in sorted(by_frame):
        mask = np.zeros(mask_shape, dtype=np.bool_)
        contributors: list[str] = []
        for lineage_id, item in by_frame[frame_index]:
            if bool(np.any(item.mask)):
                mask |= item.mask
                contributors.append(lineage_id)
        union_frames.append(
            UnionFrame(
                frame_index=frame_index,
                mask=mask,
                contributing_lineage_ids=tuple(contributors),
            )
        )
    return BinaryComposition(lineages=tuple(lineages), union_frames=tuple(union_frames))


DEFAULT_OBSERVATION_HALO_FRAMES = 48


def plan_observation_band(
    left_anchor_frame: int,
    right_anchor_frame: int,
    *,
    frame_count: int,
    halo_frames: int = DEFAULT_OBSERVATION_HALO_FRAMES,
) -> tuple[int, int]:
    """Clamp an inclusive halo around an ordered adjacent-anchor pair."""

    left = int(left_anchor_frame)
    right = int(right_anchor_frame)
    count = int(frame_count)
    halo = int(halo_frames)
    if count < 1:
        raise ValueError("frame_count must be positive")
    if halo < 0:
        raise ValueError("halo_frames must be non-negative")
    if not 0 <= left < right < count:
        raise ValueError("anchors must be ordered and inside the source frame range")
    return max(0, left - halo), min(count, right + halo + 1)


def predictions_to_tracklets(
    predictions: Sequence[Any],
    *,
    tile_index: int,
    anchor_frame: int,
    authoritative_masks: Sequence[Any],
    frame_start: int,
    frame_stop: int,
) -> tuple[tuple[Tracklet, ...], dict[str, Any]]:
    """Pass canonical tracker probabilities into anchor-scoped tracklets once."""

    import numpy as np

    masks = tuple(np.asarray(mask, dtype=bool) for mask in authoritative_masks)
    if not masks:
        raise ValueError("tracklet conversion requires authoritative instance masks")
    expected_ids = tuple(range(len(masks)))
    grouped: dict[int, list[Any]] = {object_id: [] for object_id in expected_ids}
    seen: set[tuple[int, int]] = set()
    stored_probabilities: list[float] = []
    for item in predictions:
        frame_index = int(item.frame_index)
        object_id = int(item.object_id)
        identity = frame_index, object_id
        if identity in seen:
            raise RuntimeError(f"duplicate tracker prediction {identity}")
        seen.add(identity)
        if not int(frame_start) <= frame_index < int(frame_stop):
            raise RuntimeError(
                f"prediction frame {frame_index} lies outside [{frame_start},{frame_stop})"
            )
        if object_id not in grouped:
            raise RuntimeError(
                f"tracker returned object id {object_id}; expected {list(expected_ids)}"
            )
        probability = item.frame_tracker_score
        if probability is None:
            raise RuntimeError("tracker-only prediction has no persisted tracker probability")
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"persisted tracker probability is invalid: {probability!r}")
        stored_probabilities.append(probability)
        binary = np.asarray(item.binary_mask, dtype=bool)
        if binary.shape != masks[object_id].shape:
            raise RuntimeError(
                f"tracker mask shape {binary.shape} does not match authoritative "
                f"shape {masks[object_id].shape}"
            )
        # A selected exemplar remains authoritative even if another anchor's
        # propagated hypothesis disagrees at this exact slice.
        if frame_index == int(anchor_frame):
            binary = masks[object_id]
        grouped[object_id].append(
            TrackletFrame(
                frame_index=frame_index,
                mask=binary,
                tracker_probability=probability,
                authoritative=frame_index == int(anchor_frame),
            )
        )
    tracklets = tuple(
        Tracklet(
            tile_index=int(tile_index),
            anchor_frame=int(anchor_frame),
            local_object_id=object_id,
            frames=tuple(sorted(grouped[object_id], key=lambda item: item.frame_index)),
        )
        for object_id in expected_ids
    )
    for tracklet in tracklets:
        if int(anchor_frame) not in tracklet.frame_map():
            raise RuntimeError(f"tracklet {tracklet.key} omitted its authoritative frame")
    receipt = {
        "source_field": "SamFramePrediction.frame_tracker_score",
        "source_representation": "sigmoid_probability",
        "tracklet_field": "TrackletFrame.tracker_probability",
        "tracklet_representation": "sigmoid_probability",
        "transform": "identity",
        "sigmoid_application_count": 1,
        "raw_object_score_logits_persisted": False,
        "prediction_count": len(seen),
        "probability_min": min(stored_probabilities),
        "probability_max": max(stored_probabilities),
    }
    return tracklets, receipt

__all__ = [
    "BinaryComposition",
    "DEFAULT_OBSERVATION_HALO_FRAMES",
    "ComposedFrame",
    "ComposedLineage",
    "HandoffConfig",
    "HandoffDecision",
    "HandoffFrameChoice",
    "MatchingConfig",
    "MatchingResult",
    "OverlapMetrics",
    "PairAssessment",
    "ResidualHypothesis",
    "Tracklet",
    "TrackletFrame",
    "TrackletKey",
    "TrackletMatch",
    "UnionFrame",
    "choose_monotonic_handoff",
    "compose_binary",
    "compute_overlap_metrics",
    "match_tracklets",
    "plan_observation_band",
    "predictions_to_tracklets",
]
