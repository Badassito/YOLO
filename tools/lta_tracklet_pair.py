"""Reconcile one adjacent exemplar pair as anchor-scoped instance tracklets.

This is an isolated diagnostic harness for the experimental SAM 3.1 private
mask-seed path.  It deliberately does not alter the full-volume baseline.  One
native tile and two adjacent, positive exemplar frames are selected explicitly.
The tile interval is decoded once, one SAM predictor is loaded once, and every
authoritative polygon from each anchor is propagated forward and backward in a
separate tracker session on that same predictor/GPU.

The opposing instance tracklets are conservatively matched in a shared band,
joined with a monotonic confidence-weighted handoff, and reduced to binary
review products.  All NRRDs use the source tile/frame offset so they align in
3D Slicer.  SAM's tracker-only bridge persists canonical sigmoid probabilities;
the conversion to ``TrackletFrame`` passes those probabilities through exactly
once and records that identity transform in the run summary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for entry in (ROOT, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from lta_full_volume import (  # noqa: E402
    DEFAULT_NRRD_GZIP_LEVEL,
    DEFAULT_OVERLAY_ALPHA,
    DEFAULT_TILE_SIZE,
    DEFAULT_TILE_STRIDE,
    ExperimentSessionPlan,
    PositiveAnchor,
    _atomic_write_json,
    _resolve_profile,
    _seed_masks_for_layout,
    _sha256_file,
    decode_rgb_tile_range,
    discover_known_background_frames,
    discover_positive_anchors,
    plan_tile_grid,
    probe_video,
    write_full_overlay,
    write_offset_seg_nrrd,
    write_tile_mask_video,
)
from lta_gpu_smoke import (  # noqa: E402
    _MeasuredPredictor,
    configure_constrained_gpu_batches,
    cuda_snapshot,
    find_case_video,
    install_sdpa_fallback,
    resolve_pinned_sam_runtime_provenance,
)
from XTA.lta_experimental import (  # noqa: E402
    resolve_mask_seed_capacity,
    run_mask_seed_session,
)
from XTA.lta_tracklets import (  # noqa: E402
    BinaryComposition,
    DEFAULT_OBSERVATION_HALO_FRAMES,
    HandoffConfig,
    MatchingConfig,
    MatchingResult,
    Tracklet,
    TrackletFrame,
    TrackletKey,
    compose_binary,
    match_tracklets,
    plan_observation_band,
    predictions_to_tracklets,
)
from XTA.lta_sam import (  # noqa: E402
    LTA_MAX_NUM_OBJECTS,
    LTA_MULTIPLEX_COUNT,
    build_local_sam_predictor,
    resolve_local_sam_bundle,
)


EXPERIMENT_SCHEMA = "lta.instance-tracklet-pair/1"
DEFAULT_EMPTY_FRAME_LIMIT = 30


def _jsonable(value: Any) -> Any:
    """Convert summary-only structures to strict JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _jsonable(scalar())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} as run metadata")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _half_open_ranges(values: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    output: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            output.append([start, previous + 1])
            start = value
        previous = value
    output.append([start, previous + 1])
    return output


def _key_record(key: TrackletKey) -> dict[str, int]:
    return {
        "tile_index": int(key.tile_index),
        "anchor_frame": int(key.anchor_frame),
        "local_object_id": int(key.local_object_id),
    }


def _anchor_record(anchor: PositiveAnchor, masks: Sequence[Any]) -> dict[str, Any]:
    import numpy as np

    return {
        "encoded_index": int(anchor.encoded_index),
        "decoded_frame_index": int(anchor.frame_index),
        "image_path": str(anchor.image_path),
        "label_path": str(anchor.label_path),
        "label_sha256": str(anchor.label_sha256),
        "source_polygon_count": len(anchor.polygons),
        "tile_instance_count": len(masks),
        "tile_instance_pixels": [
            int(np.count_nonzero(np.asarray(mask, dtype=bool))) for mask in masks
        ],
    }


def _resolve_pair(
    anchors: Sequence[PositiveAnchor],
    *,
    tile: Any,
    left_anchor_frame: int,
    right_anchor_frame: int,
) -> tuple[PositiveAnchor, tuple[Any, ...], PositiveAnchor, tuple[Any, ...], tuple[int, ...]]:
    """Resolve and verify an adjacent pair among anchors visible on this tile."""

    visible: list[tuple[PositiveAnchor, tuple[Any, ...]]] = []
    for anchor in anchors:
        masks = tuple(_seed_masks_for_layout(anchor, tile, "instances"))
        if masks:
            visible.append((anchor, masks))
    frames = tuple(int(anchor.frame_index) for anchor, _masks in visible)
    requested = (int(left_anchor_frame), int(right_anchor_frame))
    if requested[0] >= requested[1]:
        raise ValueError("--left-anchor-frame must precede --right-anchor-frame")
    by_frame = {int(anchor.frame_index): (anchor, masks) for anchor, masks in visible}
    missing = [frame for frame in requested if frame not in by_frame]
    if missing:
        raise ValueError(
            f"selected decoded anchor frame(s) have no positive raster support on this tile: "
            f"{missing}; visible frames are {list(frames)}"
        )
    left_position = frames.index(requested[0])
    right_position = frames.index(requested[1])
    if right_position != left_position + 1:
        intervening = list(frames[left_position + 1 : right_position])
        raise ValueError(
            "the selected frames are not adjacent positive anchors on this tile; "
            f"intervening frames={intervening}"
        )
    left, left_masks = by_frame[requested[0]]
    right, right_masks = by_frame[requested[1]]
    return left, left_masks, right, right_masks, frames




def _session_receipt(result: Mapping[str, Any], tracklets: Sequence[Tracklet]) -> dict[str, Any]:
    probabilities = [
        frame.tracker_probability for tracklet in tracklets for frame in tracklet.frames
    ]
    propagated_anchor_passed = bool(result["anchor_integrity_passed"])
    return _jsonable(
        {
            # ``run_mask_seed_session`` raises before propagation if the
            # private add_new_masks boundary changes an injected binary mask.
            # Its historical ``anchor_integrity_passed`` field instead tests
            # the tracker's subsequently propagated anchor proposal at the
            # diagnostic IoU threshold.  That second observation is useful
            # evidence, but is not the injection invariant and must not abort
            # this reconciliation experiment.
            "seed_injection_invariant": {
                "passed": True,
                "binary_masks_preserved_at_required_iou": True,
                "minimum_per_object_and_union_iou": 0.999999,
                "seed_object_metrics": result["seed_object_metrics"],
                "seed_union_metrics": result.get("final_metrics"),
                "execution_gate": True,
                "enforcement": (
                    "run_mask_seed_session raises before propagation if private "
                    "per-object or union mask injection IoU is below 0.999999"
                ),
            },
            "propagated_anchor_diagnostic": {
                "passed": propagated_anchor_passed,
                "minimum_iou": result.get("anchor_integrity_minimum_iou"),
                "union_iou": result["anchor_preview_propagation_iou"],
                "per_object_metrics": result["anchor_propagation_object_metrics"],
                "execution_gate": False,
                "policy": (
                    "record disagreement; predictions_to_tracklets and final products "
                    "hard-reinject exact authoritative masks"
                ),
            },
            # Retain the upstream spelling for machine-readable provenance,
            # while the structured fields above define its actual semantics.
            "anchor_integrity_passed": propagated_anchor_passed,
            "anchor_preview_propagation_iou": result["anchor_preview_propagation_iou"],
            "anchor_propagation_object_metrics": result[
                "anchor_propagation_object_metrics"
            ],
            "seed_object_metrics": result["seed_object_metrics"],
            "seeded_object_count": int(result["seeded_object_count"]),
            "propagation_response_count": int(result["propagation_response_count"]),
            "model_visited_frame_ranges": result["model_visited_frame_ranges"],
            "policy_zero_frame_ranges": result["policy_zero_frame_ranges"],
            "policy_zero_frame_count": int(result["policy_zero_frame_count"]),
            "empty_frame_limit": result["empty_frame_limit"],
            "empty_frame_termination_receipts": result[
                "empty_frame_termination_receipts"
            ],
            "frame_tracker_score_semantics": result["frame_tracker_score_semantics"],
            "tracklet_count": len(tracklets),
            "tracklet_frame_counts": [len(tracklet.frames) for tracklet in tracklets],
            "tracklet_frame_ranges": [
                _half_open_ranges(frame.frame_index for frame in tracklet.frames)
                for tracklet in tracklets
            ],
            "tracklet_probability_min": min(probabilities),
            "tracklet_probability_max": max(probabilities),
        }
    )


def _run_anchor_session(
    measured: Any,
    predictor: Any,
    *,
    resource: list[Any],
    tile_index: int,
    anchor: PositiveAnchor,
    masks: Sequence[Any],
    frame_start: int,
    frame_stop: int,
    conf: float,
    empty_frame_limit: int,
    session_index: int,
) -> tuple[tuple[Tracklet, ...], dict[str, Any], dict[str, Any]]:
    import numpy as np

    seed_masks = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    ground_truth = np.logical_or.reduce(seed_masks)
    result = run_mask_seed_session(
        measured,
        predictor,
        resource=resource,
        session=ExperimentSessionPlan(
            sequence_id=(
                f"lta__instance_pair_tile_{int(tile_index):02d}_"
                f"anchor_{int(anchor.frame_index):04d}"
            ),
            session_index=int(session_index),
            frame_start=int(frame_start),
            frame_stop=int(frame_stop),
        ),
        prompt_frame=int(anchor.frame_index),
        ground_truth=ground_truth,
        seed=None,
        object_masks=seed_masks,
        conf=float(conf),
        propagation_mode="tracker-only",
        empty_frame_limit=int(empty_frame_limit),
    )
    tracklets, conversion = predictions_to_tracklets(
        result["propagation"],
        tile_index=int(tile_index),
        anchor_frame=int(anchor.frame_index),
        authoritative_masks=seed_masks,
        frame_start=int(frame_start),
        frame_stop=int(frame_stop),
    )
    receipt = _session_receipt(result, tracklets)
    result = None
    gc.collect()
    return tracklets, receipt, conversion


def _matching_receipt(matching: MatchingResult) -> dict[str, Any]:
    return _jsonable(
        {
            "left_anchor_frame": int(matching.left_anchor_frame),
            "right_anchor_frame": int(matching.right_anchor_frame),
            "matches": matching.matches,
            "unmatched_left_keys": matching.unmatched_left_keys,
            "unmatched_right_keys": matching.unmatched_right_keys,
            "pair_assessments": matching.pair_assessments,
            "residual_hypotheses": matching.residual_hypotheses,
            "assignment_backend": str(matching.assignment_backend),
        }
    )


def _composition_receipt(composition: BinaryComposition) -> dict[str, Any]:
    lineages = []
    for lineage in composition.lineages:
        source_selection_frames: dict[tuple[TrackletKey, str], list[int]] = {}
        for item in lineage.frames:
            identity = item.source_key, str(item.selection_reason)
            source_selection_frames.setdefault(identity, []).append(
                int(item.frame_index)
            )
        source_selection_ranges = [
            {
                "source_key": _key_record(source_key),
                "selection_reason": selection_reason,
                "frame_ranges": _half_open_ranges(frame_indexes),
                "frame_count": len(frame_indexes),
            }
            for (source_key, selection_reason), frame_indexes in sorted(
                source_selection_frames.items(),
                key=lambda item: (item[0][0], item[0][1]),
            )
        ]
        lineages.append(
            {
                "lineage_id": str(lineage.lineage_id),
                "status": str(lineage.status),
                "member_keys": [_key_record(key) for key in lineage.member_keys],
                "frame_ranges": _half_open_ranges(item.frame_index for item in lineage.frames),
                "frame_count": len(lineage.frames),
                "active_frame_ranges": _half_open_ranges(
                    item.frame_index for item in lineage.frames if bool(item.mask.any())
                ),
                "actual_source_selection_ranges": source_selection_ranges,
                "handoff": None if lineage.handoff is None else _jsonable(lineage.handoff),
            }
        )
    return {
        "lineages": lineages,
        "union_frame_ranges": _half_open_ranges(
            item.frame_index for item in composition.union_frames
        ),
        "union_active_frame_ranges": _half_open_ranges(
            item.frame_index for item in composition.union_frames if bool(item.mask.any())
        ),
    }


def _open_work_volume(path: Path, shape: tuple[int, int, int]) -> Any:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    volume = np.memmap(str(path), mode="w+", dtype=np.uint8, shape=shape)
    volume[:] = 0
    return volume


def _close_work_volume(volume: Any) -> None:
    if volume is None:
        return
    volume.flush()
    mmap = getattr(volume, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _or_tracklets_into_volume(
    volume: Any,
    tracklets: Sequence[Tracklet],
    *,
    frame_start: int,
    frame_stop: int,
) -> None:
    import numpy as np

    for tracklet in tracklets:
        for item in tracklet.frames:
            frame_index = int(item.frame_index)
            if not int(frame_start) <= frame_index < int(frame_stop):
                raise RuntimeError(f"tracklet frame {frame_index} lies outside output interval")
            target = volume[frame_index - int(frame_start)]
            np.bitwise_or(
                target,
                np.asarray(item.mask, dtype=np.uint8),
                out=target,
            )


def _force_anchor(
    volume: Any,
    *,
    anchor_frame: int,
    frame_start: int,
    masks: Sequence[Any],
) -> None:
    import numpy as np

    union = np.logical_or.reduce(tuple(np.asarray(mask, dtype=bool) for mask in masks))
    volume[int(anchor_frame) - int(frame_start)] = np.asarray(union, dtype=np.uint8)


def _nrrd_product(
    output: Path,
    *,
    filename: str,
    volume: Any,
    tile: Any,
    frame_start: int,
    full_frame_count: int,
    spacing_xyz: Sequence[float],
    segment_name: str,
    gzip_level: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = output / filename
    receipt = write_offset_seg_nrrd(
        path,
        volume,
        tile=tile,
        frame_start=int(frame_start),
        full_frame_count=int(full_frame_count),
        spacing_xyz=spacing_xyz,
        segment_name=segment_name,
        gzip_level=int(gzip_level),
    )
    receipt["path"] = str(path.relative_to(output)).replace("\\", "/")
    artifact = {
        "kind": "offset_seg_nrrd",
        "path": receipt["path"],
        "sha256": str(receipt["sha256"]),
    }
    return receipt, artifact


def _known_background_validation(
    volume: Any,
    *,
    known_background_frames: Iterable[int],
    frame_start: int,
    frame_stop: int,
) -> dict[str, Any]:
    import numpy as np

    background_frames = tuple(int(value) for value in known_background_frames)
    rows = []
    for frame_index in sorted(
        int(value)
        for value in background_frames
        if int(frame_start) <= int(value) < int(frame_stop)
    ):
        pixels = int(np.count_nonzero(volume[frame_index - int(frame_start)]))
        if pixels:
            rows.append({"frame_index": frame_index, "mask_pixels": pixels})
    return {
        "checked_frame_count": sum(
            int(frame_start) <= int(value) < int(frame_stop)
            for value in background_frames
        ),
        "false_positive_frame_count": len(rows),
        "false_positive_pixels": sum(int(item["mask_pixels"]) for item in rows),
        "false_positive_frames": rows,
        "policy": "validation only; explicit empty labels never force output to zero",
    }


def _artifact_is_valid(output: Path, artifact: Mapping[str, Any]) -> bool:
    relative = Path(str(artifact.get("path", "")))
    if relative.is_absolute() or not relative.parts:
        return False
    candidate = (output / relative).resolve(strict=False)
    try:
        candidate.relative_to(output)
    except ValueError:
        return False
    return (
        candidate.is_file()
        and (
            str(artifact.get("kind", "")) == "ffmpeg_log"
            or candidate.stat().st_size > 0
        )
        and _sha256_file(candidate) == str(artifact.get("sha256", ""))
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True, help="Local SAM 3.1 model bundle")
    parser.add_argument("--input-root", required=True, help="Directory containing the source MKV")
    parser.add_argument("--exemplar-root", required=True, help="Source exemplar pair directory")
    parser.add_argument("--output", required=True, help="New output directory, or existing with --resume")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--profile", choices=("auto", "h100", "egpu"), default="auto")
    parser.add_argument(
        "--tile-index",
        type=int,
        required=True,
        help="One row-major native tile index; there is no parent gate",
    )
    parser.add_argument(
        "--left-anchor-frame",
        type=int,
        required=True,
        help="Earlier decoded, zero-based positive exemplar frame",
    )
    parser.add_argument(
        "--right-anchor-frame",
        type=int,
        required=True,
        help="Later decoded, zero-based positive exemplar frame",
    )
    parser.add_argument(
        "--observation-halo-frames",
        type=int,
        default=DEFAULT_OBSERVATION_HALO_FRAMES,
        help=(
            "Frames before the left and after the right anchor; the resulting shared "
            "interval is clamped to the source"
        ),
    )
    parser.add_argument("--exemplar-index-origin", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument(
        "--empty-frame-limit",
        type=int,
        default=DEFAULT_EMPTY_FRAME_LIMIT,
        help="Stop one direction after this many consecutive all-object empty masks",
    )
    parser.add_argument("--min-common-frames", type=int, default=3)
    parser.add_argument("--min-useful-frames", type=int, default=3)
    parser.add_argument("--min-coactive-frames", type=int, default=3)
    parser.add_argument("--min-iou-3d", type=float, default=0.20)
    parser.add_argument("--min-frame-iou-median", type=float, default=0.15)
    parser.add_argument("--min-coactive-coverage", type=float, default=0.50)
    parser.add_argument("--max-centroid-distance", type=float, default=0.15)
    parser.add_argument("--max-area-log-drift", type=float, default=math.log(4.0))
    parser.add_argument("--min-tracker-probability", type=float, default=0.50)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--min-runner-up-margin", type=float, default=0.08)
    parser.add_argument("--hypothesis-max-members", type=int, default=3)
    parser.add_argument("--hypothesis-candidate-limit", type=int, default=6)
    parser.add_argument("--min-union-iou-3d", type=float, default=0.55)
    parser.add_argument("--min-union-iou-gain", type=float, default=0.15)
    parser.add_argument("--min-union-similarity", type=float, default=0.55)
    parser.add_argument("--midpoint-prior-strength", type=float, default=0.35)
    parser.add_argument("--empty-choice-penalty", type=float, default=20.0)
    parser.add_argument(
        "--voxel-spacing",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(1.0, 1.0, 1.0),
    )
    parser.add_argument("--nrrd-gzip-level", type=int, default=DEFAULT_NRRD_GZIP_LEVEL)
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 0.0 <= float(args.conf) <= 1.0:
        raise ValueError("--conf must be in [0,1]")
    if int(args.empty_frame_limit) < 1:
        raise ValueError("--empty-frame-limit must be >= 1")
    if int(args.observation_halo_frames) < 0:
        raise ValueError("--observation-halo-frames must be non-negative")
    spacing = tuple(float(value) for value in args.voxel_spacing)
    if len(spacing) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in spacing
    ):
        raise ValueError("--voxel-spacing must contain three finite positive values")
    if not 0 <= int(args.nrrd_gzip_level) <= 9:
        raise ValueError("--nrrd-gzip-level must be in [0,9]")
    if not 0.0 < float(args.overlay_alpha) <= 1.0:
        raise ValueError("--overlay-alpha must be in (0,1]")

    matching_config = MatchingConfig(
        min_common_frames=int(args.min_common_frames),
        min_useful_frames=int(args.min_useful_frames),
        min_coactive_frames=int(args.min_coactive_frames),
        min_iou_3d=float(args.min_iou_3d),
        min_frame_iou_median=float(args.min_frame_iou_median),
        min_coactive_coverage=float(args.min_coactive_coverage),
        max_centroid_distance=float(args.max_centroid_distance),
        max_area_log_drift=float(args.max_area_log_drift),
        min_tracker_probability=float(args.min_tracker_probability),
        min_similarity=float(args.min_similarity),
        min_runner_up_margin=float(args.min_runner_up_margin),
        hypothesis_max_members=int(args.hypothesis_max_members),
        hypothesis_candidate_limit=int(args.hypothesis_candidate_limit),
        min_union_iou_3d=float(args.min_union_iou_3d),
        min_union_iou_gain=float(args.min_union_iou_gain),
        min_union_similarity=float(args.min_union_similarity),
    )
    handoff_config = HandoffConfig(
        midpoint_prior_strength=float(args.midpoint_prior_strength),
        empty_choice_penalty=float(args.empty_choice_penalty),
    )

    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists() and not output.is_dir():
        raise ValueError(f"--output is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not (args.resume or args.plan_only):
        raise ValueError(f"--output must be new/empty unless --resume is used: {output}")
    output.mkdir(parents=True, exist_ok=True)

    bundle = resolve_local_sam_bundle(args.model)
    if str(bundle.model_version) != "sam3.1":
        raise ValueError("the instance-tracklet experiment requires SAM 3.1")
    input_root = Path(args.input_root).expanduser().resolve(strict=True)
    exemplar_root = Path(args.exemplar_root).expanduser().resolve(strict=True)
    video_path = find_case_video(input_root, "direct")
    video = probe_video(video_path)
    left_anchor_frame = int(args.left_anchor_frame)
    right_anchor_frame = int(args.right_anchor_frame)
    frame_start, frame_stop = plan_observation_band(
        left_anchor_frame,
        right_anchor_frame,
        frame_count=int(video["frame_count"]),
        halo_frames=int(args.observation_halo_frames),
    )
    # Reconciliation is deliberately evaluated only on the complete interval
    # between these two authoritative observations.  The halo exists to test
    # appearance/disappearance behavior, not to move the identity seam.
    match_frame_start = left_anchor_frame
    match_frame_stop = right_anchor_frame + 1

    anchors = discover_positive_anchors(
        exemplar_root,
        frame_count=int(video["frame_count"]),
        index_origin=int(args.exemplar_index_origin),
    )
    known_background_frames = discover_known_background_frames(
        exemplar_root,
        frame_count=int(video["frame_count"]),
        index_origin=int(args.exemplar_index_origin),
    )
    tiles = plan_tile_grid(
        source_width=int(video["width"]),
        source_height=int(video["height"]),
        tile_size=int(args.tile_size),
        tile_stride=int(args.tile_stride),
    )
    if not 0 <= int(args.tile_index) < len(tiles):
        raise ValueError(f"--tile-index must be in [0,{len(tiles)}); got {args.tile_index}")
    tile_index = int(args.tile_index)
    tile = tiles[tile_index]
    left_anchor, left_masks, right_anchor, right_masks, visible_anchor_frames = _resolve_pair(
        anchors,
        tile=tile,
        left_anchor_frame=left_anchor_frame,
        right_anchor_frame=right_anchor_frame,
    )
    maximum_objects = max(len(left_masks), len(right_masks))
    if maximum_objects > LTA_MAX_NUM_OBJECTS:
        raise ValueError(
            f"one selected anchor requires {maximum_objects} instances, exceeding "
            f"SAM's {LTA_MAX_NUM_OBJECTS}-object experiment cap"
        )
    object_capacity = resolve_mask_seed_capacity(maximum_objects)

    plan: dict[str, Any] = {
        "experiment_schema": EXPERIMENT_SCHEMA,
        "diagnostic_only": True,
        "private_unstable_api": True,
        "baseline_mutation": False,
        "code_identity_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "XTA" / "lta_tracklets.py",
                ROOT / "XTA" / "lta_experimental.py",
                ROOT / "XTA" / "lta_tiles.py",
                TOOLS / "lta_full_volume.py",
                TOOLS / "lta_gpu_smoke.py",
                ROOT / "XTA" / "lta_sam.py",
                ROOT / "XTA" / "lta_inputs.py",
            )
        },
        "video": dict(video),
        "model": {
            "bundle_root": str(bundle.root),
            "checkpoint_path": str(bundle.checkpoint_path),
            "model_version": str(bundle.model_version),
            "checkpoint_identity_sha256": str(bundle.checkpoint_identity_sha256),
        },
        "exemplar_root": str(exemplar_root),
        "tile_index": tile_index,
        "tile_xyxy": list(tile.xyxy),
        "tile_parent_gate": False,
        "visible_positive_anchor_frames": list(visible_anchor_frames),
        "selected_anchors": [
            _anchor_record(left_anchor, left_masks),
            _anchor_record(right_anchor, right_masks),
        ],
        "adjacent_positive_anchors_on_tile": True,
        "observation_frame_range": [frame_start, frame_stop],
        "observation_halo_frames": int(args.observation_halo_frames),
        "observation_range_clamped_to_source": True,
        "matching_frame_range": [match_frame_start, match_frame_stop],
        "matching_frame_range_policy": "exactly [left_anchor,right_anchor+1)",
        "shared_decode_count": 1,
        "tracker_sessions": 2,
        "propagation_directions_per_session": ["forward", "backward"],
        "single_predictor_load": True,
        "same_gpu_for_both_anchors": True,
        "seed_layout": "instances",
        "empty_frame_limit": int(args.empty_frame_limit),
        "empty_termination_scope": "per-anchor session, per direction, all seeded objects",
        "conf": float(args.conf),
        "tracker_probability_contract": {
            "source": "SamFramePrediction.frame_tracker_score",
            "source_representation": "sigmoid_probability",
            "tracklet_representation": "sigmoid_probability",
            "transform": "identity",
            "sigmoid_application_count": 1,
        },
        "binary_publication_policy": {
            "mask_source": "tracker video_masks thresholded at > 0",
            "tracker_probability_role": "matching and handoff only",
            "pinned_tracker_expectation": (
                "the pinned tracker suppresses masks internally when "
                "object_score_logit <= 0"
            ),
        },
        "matching_config": asdict(matching_config),
        "handoff_config": asdict(handoff_config),
        "authoritative_anchor_policy": (
            "each selected anchor slice is overwritten with its exact tile-local label "
            "union after all compositions"
        ),
        "seed_integrity_policy": {
            "private_injection": (
                "authoritative per-object and union masks must retain IoU >=0.999999; "
                "run_mask_seed_session raises before propagation below that threshold"
            ),
            "propagated_anchor": (
                "the >=0.99 IoU check describes the tracker's propagated proposal at "
                "the anchor; it is recorded as a non-gating diagnostic"
            ),
            "output_anchor": (
                "predictions_to_tracklets and final binary volumes hard-reinject the "
                "exact authoritative masks"
            ),
        },
        "unmatched_tracklet_policy": "preserve verbatim in reconciled binary union",
        "residual_split_merge_policy": "audit-only hypotheses; never force identity",
        "profile_requested": str(args.profile),
        "model_object_capacity": int(object_capacity),
        "voxel_spacing_xyz": list(spacing),
        "nrrd_gzip_level": int(args.nrrd_gzip_level),
        "write_overlay": bool(args.write_overlay),
        "overlay_alpha": float(args.overlay_alpha),
        "known_background_frames": [
            int(value)
            for value in known_background_frames
            if frame_start <= int(value) < frame_stop
        ],
        "resume_policy": (
            "a complete matching summary with verified artifact hashes is reused; an "
            "incomplete run restarts both anchor sessions with one new predictor load"
        ),
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    plan_path = output / "run_plan.json"
    if plan_path.is_file():
        prior = json.loads(plan_path.read_text(encoding="utf-8"))
        if str(prior.get("plan_sha256")) != str(plan["plan_sha256"]):
            raise ValueError("existing run_plan.json does not match this invocation")
    else:
        _atomic_write_json(plan_path, _jsonable(plan))
    print(
        json.dumps(
            {
                "status": "plan_complete" if args.plan_only else "execution_planned",
                "plan_path": str(plan_path),
                "plan_sha256": str(plan["plan_sha256"]),
                "tile_index": tile_index,
                "tile_xyxy": list(tile.xyxy),
                "left_anchor_frame": left_anchor_frame,
                "right_anchor_frame": right_anchor_frame,
                "left_instance_count": len(left_masks),
                "right_instance_count": len(right_masks),
                "observation_frame_range": [frame_start, frame_stop],
                "matching_frame_range": [match_frame_start, match_frame_stop],
                "planned_model_frame_visits": 2 * (frame_stop - frame_start),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.plan_only:
        return

    summary_path = output / "summary.json"
    if args.resume and summary_path.is_file():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(completed.get("status")) != "execution_complete":
            raise ValueError("existing summary is not marked execution_complete")
        if str(completed.get("plan_sha256")) != str(plan["plan_sha256"]):
            raise ValueError("completed summary belongs to a different plan")
        artifacts = completed.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not all(isinstance(item, Mapping) for item in artifacts)
        ):
            raise ValueError("completed summary has no auditable artifact list")
        invalid = [item for item in artifacts if not _artifact_is_valid(output, item)]
        if invalid:
            raise ValueError(f"completed summary has invalid artifact receipts: {invalid}")
        print(json.dumps(completed, indent=2, sort_keys=True))
        return

    runtime = resolve_pinned_sam_runtime_provenance()
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for instance-tracklet inference")
    if not 0 <= int(args.device) < int(torch.cuda.device_count()):
        raise ValueError(
            f"--device {args.device} is outside cuda device_count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(int(args.device))
    profile = _resolve_profile(torch, int(args.device), str(args.profile))
    if profile["name"] == "egpu" and frame_stop - frame_start > 48:
        raise ValueError("the constrained eGPU profile limits this uncapped pair test to 48 frames")
    execution_contract = {
        "plan_sha256": str(plan["plan_sha256"]),
        "profile": dict(profile),
        "device_index": int(args.device),
        "construction_device": "meta",
        "compile": False,
        "warm_up": False,
        "async_loading_frames": False,
        "multiplex_count": int(LTA_MULTIPLEX_COUNT),
        "max_num_objects": int(object_capacity),
    }
    execution_sha256 = _canonical_sha256(execution_contract)
    torch.cuda.reset_peak_memory_stats(int(args.device))
    memory_events: list[dict[str, Any]] = [
        cuda_snapshot(torch, int(args.device), "before_builder")
    ]
    predictor = None
    measured = None
    restore_sdpa = None
    constrained = None
    resource = None
    active_error: BaseException | None = None
    left_tracklets: tuple[Tracklet, ...]
    right_tracklets: tuple[Tracklet, ...]
    try:
        print(
            f"Loading SAM 3.1 once for adjacent anchors {left_anchor_frame} and "
            f"{right_anchor_frame} on tile {tile_index} with profile={profile['name']}...",
            file=sys.stderr,
            flush=True,
        )
        predictor = build_local_sam_predictor(
            bundle,
            device_id=int(args.device),
            use_fa3=bool(profile["use_fa3"]),
            use_rope_real=bool(profile["use_rope_real"]),
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            conf=float(args.conf),
            weight_storage=str(profile["weight_storage"]),
            max_num_objects=int(object_capacity),
            construction_device="meta",
        )
        if profile["constrained_batches"]:
            constrained = configure_constrained_gpu_batches(predictor)
        if profile["sdpa_fallback"]:
            restore_sdpa = install_sdpa_fallback()
        measured = _MeasuredPredictor(
            predictor,
            torch,
            int(args.device),
            memory_events,
        )
        memory_events.append(cuda_snapshot(torch, int(args.device), "before_shared_decode"))
        resource = decode_rgb_tile_range(video_path, frame_start, frame_stop, tile)
        memory_events.append(cuda_snapshot(torch, int(args.device), "after_shared_decode"))
        print(
            f"Propagating {len(left_masks)} left-anchor instance(s) in both directions...",
            file=sys.stderr,
            flush=True,
        )
        left_tracklets, left_session, left_conversion = _run_anchor_session(
            measured,
            predictor,
            resource=resource,
            tile_index=tile_index,
            anchor=left_anchor,
            masks=left_masks,
            frame_start=frame_start,
            frame_stop=frame_stop,
            conf=float(args.conf),
            empty_frame_limit=int(args.empty_frame_limit),
            session_index=0,
        )
        memory_events.append(cuda_snapshot(torch, int(args.device), "after_left_anchor"))
        print(
            f"Propagating {len(right_masks)} right-anchor instance(s) in both directions...",
            file=sys.stderr,
            flush=True,
        )
        right_tracklets, right_session, right_conversion = _run_anchor_session(
            measured,
            predictor,
            resource=resource,
            tile_index=tile_index,
            anchor=right_anchor,
            masks=right_masks,
            frame_start=frame_start,
            frame_stop=frame_stop,
            conf=float(args.conf),
            empty_frame_limit=int(args.empty_frame_limit),
            session_index=1,
        )
        memory_events.append(cuda_snapshot(torch, int(args.device), "after_right_anchor"))
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        resource = None
        cleanup_errors: list[tuple[str, Exception]] = []
        if restore_sdpa is not None:
            try:
                restore_sdpa()
            except Exception as exc:
                cleanup_errors.append(("SDPA restoration", exc))
        if predictor is not None:
            shutdown = getattr(predictor, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    cleanup_errors.append(("predictor shutdown", exc))
        predictor = None
        measured = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            cleanup_errors.append(("CUDA garbage collection", exc))
        if active_error is not None:
            add_note = getattr(active_error, "add_note", None)
            if callable(add_note):
                for boundary, cleanup_error in cleanup_errors:
                    add_note(
                        f"SAM {boundary} also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        elif cleanup_errors:
            raise cleanup_errors[0][1]

    matching = match_tracklets(
        left_tracklets,
        right_tracklets,
        frame_start=match_frame_start,
        frame_stop=match_frame_stop,
        config=matching_config,
    )
    composition = compose_binary(
        left_tracklets,
        right_tracklets,
        matching,
        handoff_config=handoff_config,
    )
    matching_record = _matching_receipt(matching)
    composition_record = _composition_receipt(composition)

    depth = frame_stop - frame_start
    shape = (depth, int(tile.size), int(tile.size))
    work_root = output / ".instance_tracklets_work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_paths = {
        name: work_root / f"{name}.uint8.raw"
        for name in ("left", "right", "reconciled", "product")
    }
    left_volume = right_volume = reconciled_volume = product_volume = None
    products: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    mask_video_path = output / "lta_instance_tracklets_tile_mask.mkv"
    try:
        left_volume = _open_work_volume(work_paths["left"], shape)
        right_volume = _open_work_volume(work_paths["right"], shape)
        reconciled_volume = _open_work_volume(work_paths["reconciled"], shape)
        _or_tracklets_into_volume(
            left_volume,
            left_tracklets,
            frame_start=frame_start,
            frame_stop=frame_stop,
        )
        _or_tracklets_into_volume(
            right_volume,
            right_tracklets,
            frame_start=frame_start,
            frame_stop=frame_stop,
        )
        for union_frame in composition.union_frames:
            reconciled_volume[int(union_frame.frame_index) - frame_start] = np.asarray(
                union_frame.mask,
                dtype=np.uint8,
            )
        _force_anchor(
            left_volume,
            anchor_frame=left_anchor_frame,
            frame_start=frame_start,
            masks=left_masks,
        )
        _force_anchor(
            right_volume,
            anchor_frame=right_anchor_frame,
            frame_start=frame_start,
            masks=right_masks,
        )
        _force_anchor(
            reconciled_volume,
            anchor_frame=left_anchor_frame,
            frame_start=frame_start,
            masks=left_masks,
        )
        _force_anchor(
            reconciled_volume,
            anchor_frame=right_anchor_frame,
            frame_start=frame_start,
            masks=right_masks,
        )
        reconciled_volume.flush()
        receipt, artifact = _nrrd_product(
            output,
            filename="lta_instance_tracklets_reconciled.seg.nrrd",
            volume=reconciled_volume,
            tile=tile,
            frame_start=frame_start,
            full_frame_count=int(video["frame_count"]),
            spacing_xyz=spacing,
            segment_name=(
                f"LTA reconciled instance tracklets tile {tile_index}, "
                f"frames {frame_start}-{frame_stop - 1}"
            ),
            gzip_level=int(args.nrrd_gzip_level),
        )
        products["reconciled"] = receipt
        artifacts.append(artifact)

        product_volume = _open_work_volume(work_paths["product"], shape)
        midpoint = (float(left_anchor_frame) + float(right_anchor_frame)) / 2.0
        for offset, frame_index in enumerate(range(frame_start, frame_stop)):
            source = left_volume if float(frame_index) <= midpoint else right_volume
            product_volume[offset] = source[offset]
        _force_anchor(
            product_volume,
            anchor_frame=left_anchor_frame,
            frame_start=frame_start,
            masks=left_masks,
        )
        _force_anchor(
            product_volume,
            anchor_frame=right_anchor_frame,
            frame_start=frame_start,
            masks=right_masks,
        )
        receipt, artifact = _nrrd_product(
            output,
            filename="lta_instance_tracklets_nearest_anchor_reference.seg.nrrd",
            volume=product_volume,
            tile=tile,
            frame_start=frame_start,
            full_frame_count=int(video["frame_count"]),
            spacing_xyz=spacing,
            segment_name=f"LTA nearest-anchor reference tile {tile_index}",
            gzip_level=int(args.nrrd_gzip_level),
        )
        receipt["nominal_midpoint"] = midpoint
        products["nearest_anchor_reference"] = receipt
        artifacts.append(artifact)

        for offset in range(depth):
            np.bitwise_or(left_volume[offset], right_volume[offset], out=product_volume[offset])
        _force_anchor(
            product_volume,
            anchor_frame=left_anchor_frame,
            frame_start=frame_start,
            masks=left_masks,
        )
        _force_anchor(
            product_volume,
            anchor_frame=right_anchor_frame,
            frame_start=frame_start,
            masks=right_masks,
        )
        receipt, artifact = _nrrd_product(
            output,
            filename="lta_instance_tracklets_observation_recall_union.seg.nrrd",
            volume=product_volume,
            tile=tile,
            frame_start=frame_start,
            full_frame_count=int(video["frame_count"]),
            spacing_xyz=spacing,
            segment_name=f"LTA opposing-tracklet recall union tile {tile_index}",
            gzip_level=int(args.nrrd_gzip_level),
        )
        products["observation_recall_union"] = receipt
        artifacts.append(artifact)

        for offset in range(depth):
            np.bitwise_xor(left_volume[offset], right_volume[offset], out=product_volume[offset])
        receipt, artifact = _nrrd_product(
            output,
            filename="lta_instance_tracklets_opposing_disagreement.seg.nrrd",
            volume=product_volume,
            tile=tile,
            frame_start=frame_start,
            full_frame_count=int(video["frame_count"]),
            spacing_xyz=spacing,
            segment_name=f"LTA opposing-tracklet XOR disagreement tile {tile_index}",
            gzip_level=int(args.nrrd_gzip_level),
        )
        products["opposing_disagreement"] = receipt
        artifacts.append(artifact)

        mask_video = write_tile_mask_video(
            mask_video_path,
            reconciled_volume,
            fps=float(video["fps"]),
        )
        mask_video["path"] = str(mask_video_path.relative_to(output)).replace("\\", "/")
        artifacts.append(
            {
                "kind": "lossless_tile_mask_video",
                "path": mask_video["path"],
                "sha256": str(mask_video["sha256"]),
            }
        )
        known_background_validation = _known_background_validation(
            reconciled_volume,
            known_background_frames=known_background_frames,
            frame_start=frame_start,
            frame_stop=frame_stop,
        )
    finally:
        for volume in (product_volume, reconciled_volume, right_volume, left_volume):
            _close_work_volume(volume)
        product_volume = reconciled_volume = right_volume = left_volume = None
        gc.collect()
        for path in work_paths.values():
            path.unlink(missing_ok=True)
        try:
            work_root.rmdir()
        except OSError:
            pass

    overlay = None
    if args.write_overlay:
        overlay_path = output / "lta_instance_tracklets_full_overlay.mkv"
        overlay = write_full_overlay(
            video_path,
            tile_inputs=((tile, mask_video_path),),
            frame_start=frame_start,
            frame_stop=frame_stop,
            output_path=overlay_path,
            alpha=float(args.overlay_alpha),
            known_background_frames=known_background_frames,
        )
        overlay["path"] = str(overlay_path.relative_to(output)).replace("\\", "/")
        overlay_log = Path(str(overlay["ffmpeg_log"]))
        overlay["ffmpeg_log"] = str(overlay_log.relative_to(output)).replace("\\", "/")
        artifacts.extend(
            (
                {
                    "kind": "full_source_overlay",
                    "path": overlay["path"],
                    "sha256": str(overlay["sha256"]),
                },
                {
                    "kind": "ffmpeg_log",
                    "path": overlay["ffmpeg_log"],
                    "sha256": _sha256_file(overlay_log),
                },
            )
        )

    peak_allocated_mib = int(
        torch.cuda.max_memory_allocated(int(args.device)) // (1024 * 1024)
    )
    summary = {
        "status": "execution_complete",
        "experiment_schema": EXPERIMENT_SCHEMA,
        "diagnostic_only": True,
        "private_unstable_api": True,
        "baseline_mutation": False,
        "plan_sha256": str(plan["plan_sha256"]),
        "execution_sha256": execution_sha256,
        "run_plan": str(plan_path),
        "runtime": runtime,
        "device": {
            "index": int(args.device),
            "name": torch.cuda.get_device_name(int(args.device)),
            "capability": list(torch.cuda.get_device_capability(int(args.device))),
            "profile": profile,
        },
        "execution_contract": execution_contract,
        "constrained_batches": constrained,
        "left_anchor_session": left_session,
        "right_anchor_session": right_session,
        "tracker_probability_conversion": {
            "left": left_conversion,
            "right": right_conversion,
        },
        "binary_publication_policy": plan["binary_publication_policy"],
        "matching": matching_record,
        "composition": composition_record,
        "products": products,
        "tile_mask_video": mask_video,
        "overlay": overlay,
        "known_background_validation": known_background_validation,
        "artifacts": artifacts,
        "peak_allocated_mib": peak_allocated_mib,
        "memory_events": memory_events,
        "limitations": [
            "this calibration diagnostic executes one adjacent anchor pair on one tile",
            "residual split/merge findings are audit-only and do not force lineage edges",
            "neighbor-session execution is not invoked here; reusable relay planning lives "
            "in XTA.lta_tile_tracking",
            "raw tracker object-score logits are not persisted; canonical sigmoid "
            "probabilities pass through exactly once",
            "an incomplete --resume reruns both tracker sessions",
        ],
    }
    summary["summary_path"] = str(summary_path)
    _atomic_write_json(summary_path, _jsonable(summary))
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
