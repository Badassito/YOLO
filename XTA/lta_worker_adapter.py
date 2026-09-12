"""Concrete persistent-worker adapter for production LTA propagation chains.

The module is imported only inside an LTA worker after that process narrows
``CUDA_VISIBLE_DEVICES``.  Dense masks never cross multiprocessing queues: each
task reads an immutable seed/cache artifact and atomically publishes a compact
JSON manifest plus file-backed union and relay-seed artifacts.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import operator
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class LtaSamWorkerContext:
    predictor: object
    profile: Mapping[str, object]
    sam_runtime: Mapping[str, object]
    torch_module: object
    restore_sdpa: object | None
    constrained_batches: Mapping[str, object] | None
    cpu_budget: Mapping[str, object] | None = None
    trace_root: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
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
    raise TypeError(f"cannot encode {type(value).__name__} as worker metadata")


def build_worker_predictor(config: Mapping[str, object]) -> LtaSamWorkerContext:
    """Build SAM 3.1 exactly once after worker-local CUDA binding."""

    import torch
    import cv2

    from .lta_cpu import configure_worker_runtime_threads

    cpu_budget = configure_worker_runtime_threads(torch, cv2_module=cv2)

    from .lta_sam import (
        LTA_MAX_NUM_OBJECTS,
        build_local_sam_predictor,
        configure_constrained_gpu_batches,
        install_sdpa_fallback,
        resolve_local_sam_bundle,
        resolve_pinned_sam_runtime_provenance,
        resolve_sam_device_profile,
    )

    if int(torch.cuda.device_count()) != 1:
        raise RuntimeError(
            "an LTA worker must expose exactly one logical CUDA device after binding"
        )
    torch.cuda.set_device(0)
    bundle = resolve_local_sam_bundle(str(config["model_path"]))
    if bundle.model_version != "sam3.1":
        raise ValueError("production mask-injected LTA requires a SAM 3.1 bundle")
    sam_runtime = resolve_pinned_sam_runtime_provenance()
    profile = resolve_sam_device_profile(
        torch,
        0,
        requested=str(config.get("profile", "auto")),
    )
    predictor = build_local_sam_predictor(
        bundle,
        device_id=0,
        use_fa3=bool(profile["use_fa3"]),
        use_rope_real=bool(profile["use_rope_real"]),
        compile=bool(profile["compile"]),
        warm_up=bool(profile["warm_up"]),
        async_loading_frames=bool(profile["async_loading_frames"]),
        conf=float(config.get("conf", 0.15)),
        weight_storage=str(profile["weight_storage"]),
        max_num_objects=LTA_MAX_NUM_OBJECTS,
        construction_device=str(profile["construction_device"]),
    )
    constrained = (
        configure_constrained_gpu_batches(predictor)
        if bool(profile["constrained_batches"])
        else None
    )
    restore_sdpa = install_sdpa_fallback() if bool(profile["sdpa_fallback"]) else None
    return LtaSamWorkerContext(
        predictor=predictor,
        profile=dict(profile),
        sam_runtime=dict(sam_runtime),
        torch_module=torch,
        restore_sdpa=restore_sdpa,
        constrained_batches=constrained,
        cpu_budget=cpu_budget,
        trace_root=(None if config.get("worker_trace_root") is None else str(config["worker_trace_root"])),
    )


def close_worker_predictor(context: LtaSamWorkerContext) -> None:
    """Release the one worker-owned predictor without masking cleanup failures."""

    errors: list[Exception] = []
    if callable(context.restore_sdpa):
        try:
            context.restore_sdpa()
        except Exception as exc:
            errors.append(exc)
    shutdown = getattr(context.predictor, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception as exc:
            errors.append(exc)
    context.predictor = None  # type: ignore[assignment]
    gc.collect()
    try:
        context.torch_module.cuda.empty_cache()
    except Exception as exc:
        errors.append(exc)
    if errors:
        raise errors[0]


def _tile_from_payload(record: Mapping[str, object]):
    from .lta_tiles import TilePlan

    return TilePlan(
        left=int(record["left"]),
        top=int(record["top"]),
        size=int(record["size"]),
        source_width=int(record["source_width"]),
        source_height=int(record["source_height"]),
        row=None if record.get("row") is None else int(record["row"]),
        column=None if record.get("column") is None else int(record["column"]),
    )


def _window_from_payload(record: Mapping[str, object]):
    from .lta_windows import WindowPlan

    return WindowPlan(
        branch=str(record["branch"]),
        ordinal=int(record["ordinal"]),
        frame_start=int(record["frame_start"]),
        frame_stop=int(record["frame_stop"]),
        prompt_frame=int(record["prompt_frame"]),
        direction=str(record["direction"]),
        seed_kind=str(record["seed_kind"]),
    )


def _half_open_frame_ranges(values: Sequence[int]) -> list[list[int]]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append([start, previous + 1])
            start = value
        previous = value
    ranges.append([start, previous + 1])
    return ranges


def _relay_observation(
    observations: dict[tuple[str, int], dict[str, object]],
    *,
    prediction: object,
    seed_by_lineage: Mapping[object, object],
    source_tile: object,
    neighbors: Sequence[tuple[int, object]],
    min_pixels: int,
    min_probability: float,
) -> None:
    import numpy as np

    from .lta_tile_tracking import rebase_tile_mask

    lineage = prediction.lineage
    seed = seed_by_lineage[lineage]
    frame = prediction.prediction
    probability = float(
        frame.frame_tracker_score
        if frame.frame_tracker_score is not None
        else frame.initial_detection_score
    )
    if probability < float(min_probability):
        return
    for destination_index, destination_tile in neighbors:
        source_x0, source_y0, source_x1, source_y1 = source_tile.xyxy
        destination_x0, destination_y0, destination_x1, destination_y1 = (
            destination_tile.xyxy
        )
        overlap_x0 = max(source_x0, destination_x0)
        overlap_y0 = max(source_y0, destination_y0)
        overlap_x1 = min(source_x1, destination_x1)
        overlap_y1 = min(source_y1, destination_y1)
        if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
            source_mask = np.asarray(frame.binary_mask, dtype=np.bool_)
            overlap_pixels = int(
                np.count_nonzero(
                    source_mask[
                        overlap_y0 - source_y0 : overlap_y1 - source_y0,
                        overlap_x0 - source_x0 : overlap_x1 - source_x0,
                    ]
                )
            )
            if overlap_pixels < int(min_pixels):
                continue
        rebased = rebase_tile_mask(frame.binary_mask, source_tile, destination_tile)
        if int(np.count_nonzero(rebased)) < int(min_pixels):
            continue
        key = lineage.token, int(destination_index)
        record = observations.setdefault(
            key,
            {
                "lineage": lineage,
                "seed": seed,
                "destination_index": int(destination_index),
                "episodes": [],
            },
        )
        packed = np.packbits(np.asarray(rebased, dtype=np.bool_).reshape(-1))
        value = (
            int(frame.frame_index),
            np.ascontiguousarray(packed, dtype=np.uint8),
            tuple(int(value) for value in rebased.shape),
            probability,
        )
        _insert_relay_episode(record, value)


def _merge_relay_values(left: tuple[object, ...], right: tuple[object, ...]):
    """Union duplicate overlap evidence for one lineage/destination/frame."""

    import numpy as np

    left_frame, left_mask, left_shape, left_probability = left
    right_frame, right_mask, right_shape, right_probability = right
    if int(left_frame) != int(right_frame) or tuple(left_shape) != tuple(right_shape):
        raise RuntimeError("cannot merge relay observations from different frames or shapes")
    return (
        int(left_frame),
        np.bitwise_or(
            np.asarray(left_mask, dtype=np.uint8),
            np.asarray(right_mask, dtype=np.uint8),
        ),
        tuple(int(value) for value in left_shape),
        max(float(left_probability), float(right_probability)),
    )


def _insert_relay_episode(
    record: dict[str, object],
    value: tuple[object, ...],
) -> None:
    """Retain only endpoints of every contiguous overlap episode.

    Predictions can arrive forward, backward, or through repeated dogfood
    boundaries.  Maintaining sorted disjoint intervals makes insertion order
    irrelevant without retaining a mask for every active frame.
    """

    frame = int(value[0])
    episodes = list(record.get("episodes", ()))
    for index, raw_episode in enumerate(episodes):
        first, last = raw_episode
        first_frame, last_frame = int(first[0]), int(last[0])
        if first_frame <= frame <= last_frame:
            if frame == first_frame:
                first = _merge_relay_values(first, value)
            if frame == last_frame:
                last = first if first_frame == last_frame else _merge_relay_values(last, value)
            episodes[index] = (first, last)
            record["episodes"] = episodes
            return

    left_index = next(
        (index for index, episode in enumerate(episodes) if int(episode[1][0]) == frame - 1),
        None,
    )
    right_index = next(
        (index for index, episode in enumerate(episodes) if int(episode[0][0]) == frame + 1),
        None,
    )
    if left_index is not None and right_index is not None:
        left = episodes[left_index]
        right = episodes[right_index]
        episodes[left_index] = (left[0], right[1])
        del episodes[right_index]
    elif left_index is not None:
        first, _last = episodes[left_index]
        episodes[left_index] = (first, value)
    elif right_index is not None:
        _first, last = episodes[right_index]
        episodes[right_index] = (value, last)
    else:
        episodes.append((value, value))
    episodes.sort(key=lambda episode: int(episode[0][0]))
    record["episodes"] = episodes


def _write_relay_artifacts(
    observations: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    output_dir: Path,
    source_tile_index: int,
    generation: int,
) -> list[dict[str, object]]:
    import numpy as np

    from .lta_propagation import LtaMaskSeed, LtaSeedProvenance, write_seed_artifact

    records: list[dict[str, object]] = []
    relay_dir = Path(output_dir) / "relays"
    artifact_ordinal = 0
    for _key, observation in sorted(observations.items()):
        lineage = observation["lineage"]
        source_seed = observation["seed"]
        destination = int(observation["destination_index"])
        visited = tuple(
            sorted(
                set(source_seed.visited_tile_indices)
                | {int(source_tile_index), destination}
            )
        )
        if "endpoint_candidates" in observation:
            raw_candidates = observation["endpoint_candidates"]
            if not isinstance(raw_candidates, (tuple, list)):
                raise ValueError("relay endpoint candidates must be a list")
            candidates = []
            for candidate in raw_candidates:
                if not isinstance(candidate, (tuple, list)) or len(candidate) != 3:
                    raise ValueError("relay endpoint candidates require direction, value, and range")
                direction, value, raw_range = candidate
                if direction not in {"forward", "backward"}:
                    raise ValueError("relay endpoint candidate direction is invalid")
                if not isinstance(value, (tuple, list)) or len(value) != 4:
                    raise ValueError("relay endpoint candidate value is invalid")
                if not isinstance(raw_range, (tuple, list)) or len(raw_range) != 2:
                    raise ValueError("relay endpoint candidate range is invalid")
                try:
                    if any(isinstance(item, bool) for item in (*raw_range, value[0])):
                        raise TypeError("boolean frame index")
                    first_frame, stop_frame = (int(operator.index(item)) for item in raw_range)
                    candidate_frame = int(operator.index(value[0]))
                except TypeError as exc:
                    raise ValueError("relay endpoint candidate frame indices must be integers") from exc
                if (not 0 <= first_frame <= candidate_frame < stop_frame
                        or direction == "forward" and candidate_frame != first_frame
                        or direction == "backward" and candidate_frame != stop_frame - 1):
                    raise ValueError("relay endpoint candidate frame does not match its episode range")
                candidates.append((direction, value, (first_frame, stop_frame)))
            indices = {value: index for index, value in enumerate(sorted({row[2] for row in candidates}))}
            groups = [
                (indices[frame_range], frame_range, ((direction, value),))
                for direction, value, frame_range in sorted(
                    candidates, key=lambda row: (int(row[1][0]), 0 if row[0] == "forward" else 1, row[2])
                )
            ]
        else:
            # Keep legacy episode and forward/backward ordering byte-identical.
            groups = [
                (index, (int(first[0]), int(last[0]) + 1), (("forward", first), ("backward", last)))
                for index, (first, last) in enumerate(observation.get("episodes", ()))
            ]
        for episode_index, episode_frame_range, endpoints in groups:
            for direction, value in endpoints:
                frame_index, packed_mask, mask_shape, probability = value
                mask = np.unpackbits(
                    np.asarray(packed_mask, dtype=np.uint8),
                    count=int(mask_shape[0]) * int(mask_shape[1]),
                ).reshape(tuple(int(item) for item in mask_shape))
                seed = LtaMaskSeed(
                    lineage=lineage,
                    frame_index=int(frame_index),
                    object_id=int(source_seed.object_id),
                    mask=mask,
                    provenance=LtaSeedProvenance.SPATIAL_RELAY,
                    tracker_probability=float(probability),
                    relay_generation=int(generation) + 1,
                    visited_tile_indices=visited,
                    source_receipt={
                        "source_tile_index": int(source_tile_index),
                        "destination_tile_index": destination,
                        "temporal_direction": direction,
                        "overlap_episode_index": int(episode_index),
                        "overlap_episode_frame_range": list(episode_frame_range),
                        "source_generation": int(generation),
                    },
                )
                artifact = write_seed_artifact(
                    relay_dir
                    / (
                        f"relay_{artifact_ordinal:04d}_episode_{episode_index:04d}_"
                        f"to_{destination:04d}_{direction}_"
                        f"frame_{int(frame_index):06d}.npz"
                    ),
                    (seed,),
                )
                artifact_ordinal += 1
                records.append(
                    {
                        "lineage": {
                            "volume_id": lineage.volume_id,
                            "physical_view_id": lineage.physical_view_id,
                            "runtime_view_id": lineage.runtime_view_id,
                            "tile_config_id": lineage.tile_config_id,
                            "lineage_id": lineage.lineage_id,
                        },
                        "source_tile_index": int(source_tile_index),
                        "destination_tile_index": destination,
                        "frame_index": int(frame_index),
                        "temporal_direction": direction,
                        "overlap_episode_index": int(episode_index),
                        "overlap_episode_frame_range": list(episode_frame_range),
                        "generation": int(generation) + 1,
                        "visited_tile_indices": list(visited),
                        "seed_artifact_path": str(artifact.path),
                        "seed_artifact_sha256": artifact.sha256,
                    }
                )
    return records


def execute_worker_task(
    context: LtaSamWorkerContext,
    kind: str,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    from .lta_telemetry import LtaExecutionTrace

    trace = LtaExecutionTrace(
        payload.get("worker_trace_root", getattr(context, "trace_root", None)),
        f"adapter_gpu{os.environ.get('LTA_EXECUTION_DEVICE_ID', 'unknown')}",
    )
    try:
        return _execute_worker_task_impl(context, kind, payload, trace=trace)
    finally:
        trace.close()


def _mark_completed_coverage(coverage: object, *, request: object, adapter_receipt: Mapping[str, object]) -> None:
    """Record only visited directed intervals, retaining gaps and prompt sides."""
    lineages = tuple(seed.lineage for seed in request.seeds)
    context_start = request.session.frame_start
    context_stop = request.session.frame_stop
    prompt = request.prompt_frame
    if "model_visited_frame_ranges" not in adapter_receipt:
        if request.empty_frame_limit is None or request.empty_frame_limit >= context_stop - context_start:
            coverage.mark_observed(lineages, frame_start=context_start, frame_stop=context_stop,
                                   prompt_frame=prompt, direction=request.direction)
        else:
            # Custom adapters may omit visit receipts. The injected prompt is
            # known, but a short empty-frame limit leaves every edge uncertain.
            coverage.mark_observed(lineages, frame_start=prompt, frame_stop=prompt + 1,
                                   prompt_frame=prompt, direction=request.direction)
        return

    raw_ranges = adapter_receipt["model_visited_frame_ranges"]
    if not isinstance(raw_ranges, (tuple, list)):
        raise RuntimeError("model-visited frame ranges must be an ordered list")
    ranges = []
    previous_stop = context_start
    for pair in raw_ranges:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise RuntimeError("model-visited frame ranges require start/stop pairs")
        try:
            if any(isinstance(value, bool) for value in pair):
                raise TypeError("boolean frame index")
            first, last = (int(operator.index(value)) for value in pair)
        except TypeError as exc:
            raise RuntimeError("model-visited frame range endpoints must be integers") from exc
        if not context_start <= first < last <= context_stop or first < previous_stop:
            raise RuntimeError("model-visited frame ranges must be sorted, nonoverlapping, and inside the session")
        ranges.append((first, last))
        previous_stop = last

    # Validate every range before mutating the coverage builder. Separate
    # intervals stay separate, including adjacent intervals with no shared
    # frame; joining them would invent an unobserved transition.
    for first, last in ranges:
        if request.direction == "both" and first <= prompt < last:
            coverage.mark_observed(lineages, frame_start=first, frame_stop=last,
                                   prompt_frame=prompt, direction="both")
            continue
        if request.direction in {"both", "forward"}:
            forward_start = max(first, prompt)
            if forward_start < last:
                coverage.mark_observed(lineages, frame_start=forward_start, frame_stop=last,
                                       prompt_frame=forward_start, direction="forward")
        if request.direction in {"both", "backward"}:
            backward_stop = min(last, prompt + 1)
            if first < backward_stop:
                coverage.mark_observed(lineages, frame_start=first, frame_stop=backward_stop,
                                       prompt_frame=backward_stop - 1, direction="backward")


def _execute_worker_task_impl(
    context: LtaSamWorkerContext,
    kind: str,
    payload: Mapping[str, object],
    *,
    trace: object,
) -> Mapping[str, object]:
    window_task = str(kind) == "propagation_window"
    if str(kind) not in {"propagation_chain", "propagation_window"}:
        raise ValueError(f"unsupported production LTA worker task kind {kind!r}")

    import numpy as np

    from .lta_coverage import LtaCoverageBuilder
    from .lta_experimental import MASK_SEED_MIN_EXCLUSIVE_FRACTION
    from .lta_postprocessing import fill_binary_mask_holes_2d
    from .lta_propagation import (
        LtaPropagationRequest,
        partition_mask_seed_sessions,
        read_seed_artifact,
        run_mask_injected_session,
        write_seed_artifact,
    )
    from .lta_relay_episodes import write_relay_observations
    from .lta_rendering import LtaPhysicalViewCacheRef, render_native_tile_window
    from .lta_sam import LTA_MAX_NUM_OBJECTS, SamSessionPlan
    from .lta_windows import owned_frame_range
    from .lta_union_artifacts import LtaUnionWriter

    output_dir = Path(str(payload["output_dir"])).resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_ref = LtaPhysicalViewCacheRef.from_payload(payload["cache_ref"])  # type: ignore[arg-type]
    source_tile = _tile_from_payload(payload["tile"])  # type: ignore[arg-type]
    tile_index = int(payload["tile_index"])
    generation = int(payload.get("relay_generation", 0))
    relay_min_pixels = int(payload.get("relay_min_pixels", 16))
    relay_min_probability = float(payload.get("relay_min_probability", 0.5))
    if relay_min_pixels < 1:
        raise ValueError("relay_min_pixels must be positive")
    if not 0.0 <= relay_min_probability <= 1.0:
        raise ValueError("relay_min_probability must be in [0,1]")
    frame_start = int(payload["output_frame_start"])
    frame_stop = int(payload["output_frame_stop"])
    if not 0 <= frame_start < frame_stop <= cache_ref.shape[0]:
        raise ValueError("worker output frame range is outside its view cache")
    windows = tuple(
        _window_from_payload(item)
        for item in payload["windows"]  # type: ignore[union-attr]
    )
    if not windows:
        raise ValueError("a propagation-chain task requires at least one window")
    if window_task:
        if len(windows) != 1:
            raise ValueError("a propagation-window task requires exactly one window")
        if (frame_start, frame_stop) != owned_frame_range(windows[0]):
            raise ValueError("a propagation-window output range must match its owned frames")
        if not str(payload.get("chain_work_id", "")).strip():
            raise ValueError("a propagation-window task requires its original chain_work_id")
    initial_seeds = read_seed_artifact(
        str(payload["seed_artifact_path"]),
        expected_sha256=str(payload["seed_artifact_sha256"]),
    )
    if any(seed.frame_index != windows[0].prompt_frame for seed in initial_seeds):
        raise ValueError("initial seed artifact does not match the first prompt frame")
    if any(tuple(seed.mask.shape) != (source_tile.size, source_tile.size) for seed in initial_seeds):
        raise ValueError("seed mask shape does not match the task tile")
    if not initial_seeds:
        raise ValueError("a propagation task requires at least one mask seed")
    tile_config_id = str(payload.get("tile_config_id", initial_seeds[0].lineage.tile_config_id))
    if any(seed.lineage.tile_config_id != tile_config_id for seed in initial_seeds):
        raise ValueError("seed lineage tile configuration does not match the task")

    neighbors = tuple(
        (int(item["tile_index"]), _tile_from_payload(item["tile"]))
        for item in payload.get("neighbors", ())  # type: ignore[union-attr]
    )
    union_shape = (frame_stop - frame_start, source_tile.size, source_tile.size)
    coverage = LtaCoverageBuilder((source_tile.size, source_tile.size))
    union_writer = LtaUnionWriter(
        output_dir / "union.pack", shape=union_shape, frame_start=frame_start,
    )
    try:
        seed_by_lineage = {seed.lineage: seed for seed in initial_seeds}
        boundary_seeds: dict[int, tuple[object, ...]] = {
            int(initial_seeds[0].frame_index): tuple(initial_seeds)
        }
        outbound_dogfood: dict[int, list[object]] = {}
        observations: dict[tuple[str, int], dict[str, object]] = {}
        active_frames_by_lineage: dict[str, set[int]] = {}
        receipts: list[dict[str, object]] = []
        halted_branches: set[str] = set()
        tracker_session_index = 0
        for window_index, window in enumerate(windows):
            if window.branch in halted_branches:
                continue
            seeds = (
                tuple(initial_seeds)
                if window_index == 0
                else tuple(boundary_seeds.pop(window.prompt_frame, ()))
            )
            if not seeds:
                halted_branches.add(window.branch)
                receipts.append(
                    {
                        "window": _jsonable(asdict(window)),
                        "status": "halted_empty_dogfood_boundary",
                    }
                )
                continue
            # The production wrapper hole-fills immediately before injection.
            # Partition on that same geometry so filling cannot introduce an
            # overlap conflict after the planning-time check.
            with trace.phase("seed_partition", work_id=str(payload["work_id"]), window_index=window_index):
                seed_partitions = partition_mask_seed_sessions(
                    seeds,
                    mask_transform=fill_binary_mask_holes_2d,
                )
            with trace.phase("render_window", work_id=str(payload["work_id"]), window_index=window_index):
                resource = render_native_tile_window(
                    cache_ref,
                    frame_start=window.frame_start,
                    frame_stop=window.frame_stop,
                    tile_xyxy=source_tile.xyxy,
                )
            seed_by_lineage.update({seed.lineage: seed for seed in seeds})
            reduced_prediction_keys: set[tuple[str, int, int]] = set()
            owned_start, owned_stop = owned_frame_range(window)
            window_union = np.zeros(
                (owned_stop - owned_start, source_tile.size, source_tile.size),
                dtype=np.uint8,
            )

            def reduce_prediction(item: Any) -> None:
                prediction = item.prediction
                key = (
                    str(item.lineage.token),
                    int(prediction.frame_index),
                    int(prediction.object_id),
                )
                if key in reduced_prediction_keys:
                    raise RuntimeError(f"worker received duplicate streamed prediction {key}")
                reduced_prediction_keys.add(key)
                coverage.add_prediction(item.lineage, key[1], prediction.binary_mask)
                if bool(np.asarray(prediction.binary_mask, dtype=np.bool_).any()):
                    active_frames_by_lineage.setdefault(key[0], set()).add(key[1])
                if owned_start <= prediction.frame_index < owned_stop:
                    window_union[prediction.frame_index - owned_start] |= np.asarray(
                        prediction.binary_mask,
                        dtype=np.uint8,
                    )
                _relay_observation(
                    observations,
                    prediction=item,
                    seed_by_lineage=seed_by_lineage,
                    source_tile=source_tile,
                    neighbors=neighbors,
                    min_pixels=relay_min_pixels,
                    min_probability=relay_min_probability,
                )

            partition_count = len(seed_partitions)
            for partition_index, partition_seeds in enumerate(seed_partitions):
                current_session_index = tracker_session_index
                tracker_session_index += 1
                request = LtaPropagationRequest(
                    work_id=(
                        f"{payload['work_id']}::window-{window_index:04d}::"
                        f"seed-partition-{partition_index:04d}-of-{partition_count:04d}"
                    ),
                    session=SamSessionPlan(
                        sequence_id=str(payload["sequence_id"]),
                        session_index=current_session_index,
                        frame_start=window.frame_start,
                        frame_stop=window.frame_stop,
                    ),
                    prompt_frame=window.prompt_frame,
                    direction=window.direction,
                    seeds=tuple(partition_seeds),
                    conf=float(payload["conf"]),
                    empty_frame_limit=(
                        None
                        if payload.get("empty_frame_limit") is None
                        else int(payload["empty_frame_limit"])
                    ),
                    relay_generation=generation,
                )
                try:
                    session_started = time.monotonic()
                    with trace.phase(
                        "sam_session", work_id=str(payload["work_id"]),
                        window_index=window_index, partition_index=partition_index,
                        frame_start=window.frame_start, frame_stop=window.frame_stop,
                        prompt_frame=window.prompt_frame, direction=window.direction,
                        seed_count=len(partition_seeds),
                    ):
                        result = run_mask_injected_session(
                            context.predictor,
                            context.predictor,
                            resource=resource,
                            request=request,
                            prediction_callback=reduce_prediction,
                            retain_predictions=False,
                        )
                    session_wall_seconds = time.monotonic() - session_started
                except RuntimeError as exc:
                    seed_context = [
                        {
                            "lineage": seed.lineage.token,
                            "object_id": seed.object_id,
                            "frame_index": seed.frame_index,
                            "provenance": seed.provenance.value,
                            "foreground_pixels": int(np.count_nonzero(seed.mask)),
                            "source_receipt": dict(seed.source_receipt),
                        }
                        for seed in tuple(partition_seeds)[:8]
                    ]
                    raise RuntimeError(
                        f"{exc}; production seed context: work_id={payload['work_id']!r}, "
                        f"tile_index={tile_index}, window={_jsonable(asdict(window))}, "
                        f"seed_partition={partition_index + 1}/{partition_count}, "
                        f"window_seed_count={len(seeds)}, "
                        f"task_seed_artifact={payload['seed_artifact_path']!r}, "
                        f"task_seed_sha256={payload['seed_artifact_sha256']!r}, "
                        f"seed_count={len(partition_seeds)}, "
                        f"first_seeds={_jsonable(seed_context)}"
                    ) from exc
                # Compatibility fallback for injected test/third-party adapters
                # that return retained results without honoring the reducer.  The
                # production implementation takes the streaming branch above.
                for item in result.predictions:
                    key = (
                        str(item.lineage.token),
                        int(item.prediction.frame_index),
                        int(item.prediction.object_id),
                    )
                    if key not in reduced_prediction_keys:
                        reduce_prediction(item)
                # A failed session may have streamed partial predictions, but
                # it must never publish a completed observation interval. The
                # coverage packet itself is written only after every session
                # in this task has completed successfully.
                _mark_completed_coverage(coverage, request=request, adapter_receipt=result.adapter_receipt)
                for seed in result.dogfood_seeds:
                    boundary_seeds.setdefault(seed.frame_index, tuple())
                    boundary_seeds[seed.frame_index] = (
                        *boundary_seeds[seed.frame_index],
                        seed,
                    )
                    seed_by_lineage[seed.lineage] = seed
                    if window_task:
                        outbound_dogfood.setdefault(seed.frame_index, []).append(seed)
                receipts.append(
                    {
                        "window": _jsonable(asdict(window)),
                        "status": "complete",
                        "seed_partition_index": partition_index,
                        "seed_partition_count": partition_count,
                        "seed_count": len(partition_seeds),
                        "wall_seconds": session_wall_seconds,
                        "wall_time_semantics": "host wall time including tracker and synchronous prediction reduction",
                        "prediction_count": int(
                            result.adapter_receipt.get(
                                "canonical_prediction_count",
                                len(reduced_prediction_keys),
                            )
                        ),
                        "retained_prediction_count": len(result.predictions),
                        "dogfood_seed_count": len(result.dogfood_seeds),
                        "hole_fill_added_pixels": result.hole_fill_added_pixels,
                        "adapter": _jsonable(result.adapter_receipt),
                    }
                )
            with trace.phase("union_chunk_pack", work_id=str(payload["work_id"]), window_index=window_index):
                union_writer.append_chunk(owned_start, window_union)
            window_union = None
            resource = None
            gc.collect()
    except BaseException:
        union_writer.abort()
        raise

    with trace.phase("union_finish", work_id=str(payload["work_id"])):
        union_receipt = union_writer.finish()
    foreground_pixels = int(union_receipt["foreground_pixels"])
    dogfood_seed_artifacts = []
    with trace.phase("relay_observation_artifact", work_id=str(payload["work_id"])):
        observation_artifact = write_relay_observations(output_dir, observations)
    if window_task:
        relay_records = []
        with trace.phase("dogfood_seed_artifacts", work_id=str(payload["work_id"])):
            for frame_index, seeds in sorted(outbound_dogfood.items()):
                artifact = write_seed_artifact(
                    output_dir / "dogfood" / f"frame-{frame_index:06d}.npz",
                    tuple(sorted(seeds, key=lambda seed: seed.object_id)),
                )
                dogfood_seed_artifacts.append({
                    "frame_index": int(frame_index), "path": str(artifact.path),
                    "sha256": artifact.sha256, "seed_count": artifact.seed_count,
                })
    else:
        with trace.phase("spatial_relay_artifacts", work_id=str(payload["work_id"])):
            relay_records = _write_relay_artifacts(
                observations,
                output_dir=output_dir,
                source_tile_index=tile_index,
                generation=generation,
            )
    with trace.phase("lineage_coverage_artifact", work_id=str(payload["work_id"])):
        coverage_artifact = coverage.write(
            output_dir, work_id=str(payload["work_id"]), tile_index=tile_index,
            tile_config_id=tile_config_id,
        )
    manifest = {
        "schema": "lta.propagation-chain/1",
        "task_granularity": "window" if window_task else "chain",
        "chain_work_id": str(payload.get("chain_work_id", payload["work_id"])),
        "status": "complete",
        "work_id": str(payload["work_id"]),
        "sequence_id": str(payload["sequence_id"]),
        "tile_index": tile_index,
        "tile": dict(payload["tile"]),  # type: ignore[arg-type]
        "relay_generation": generation,
        "output_frame_range": [frame_start, frame_stop],
        "union": union_receipt,
        "relays": relay_records,
        "dogfood_seed_artifacts": dogfood_seed_artifacts,
        "relay_observation_artifact": observation_artifact,
        "lineage_coverage": _jsonable(coverage_artifact),
        "lineage_active_frame_ranges": {
            lineage: _half_open_frame_ranges(frames)
            for lineage, frames in sorted(active_frames_by_lineage.items())
        },
        "relay_gate": {
            "minimum_overlap_pixels": relay_min_pixels,
            "minimum_tracker_probability": relay_min_probability,
        },
        "seed_session_partition": {
            "strategy": "deterministic_first_fit_aggregate_exclusive_support",
            "minimum_exclusive_fraction": MASK_SEED_MIN_EXCLUSIVE_FRACTION,
            "maximum_objects_per_session": LTA_MAX_NUM_OBJECTS,
            "tracker_session_count": tracker_session_index,
            "planned_window_count": len(windows),
        },
        "windows": receipts,
        "profile": _jsonable(context.profile),
        "sam_runtime": _jsonable(context.sam_runtime),
        "constrained_batches": _jsonable(context.constrained_batches),
        "cpu_budget": _jsonable(getattr(context, "cpu_budget", None)),
        "foreground_pixels": foreground_pixels,
    }
    if window_task:
        manifest["window"] = _jsonable(asdict(windows[0]))
    from .lta_outputs import write_json_atomically

    manifest_path = write_json_atomically(output_dir / "manifest.json", manifest)
    return {
        "artifact_path": str(manifest_path),
        "metrics": {
            "foreground_pixels": manifest["foreground_pixels"],
            "relay_count": len(relay_records),
            # ``window_count`` is the legacy count of concrete tracker-session
            # receipts. A logical planned window can now require more than one
            # overlap-compatible seed partition.
            "window_count": len(receipts),
            "planned_window_count": len(windows),
            "tracker_session_count": tracker_session_index,
        },
        "metadata": {
            "tile_index": tile_index,
            "relay_generation": generation,
            "profile": str(context.profile["name"]),
        },
    }


__all__ = (
    "LtaSamWorkerContext",
    "build_worker_predictor",
    "close_worker_predictor",
    "execute_worker_task",
)
