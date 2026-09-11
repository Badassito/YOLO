"""Production coordinator for authoritative-mask LTA execution.

The first executable production contract is intentionally the qualified native
Transverse path: one angle (0 degrees), one or more overlapping 1008-pixel tile
grids, directly addressable target annotations, persistent one-model-per-GPU
workers, and a breadth-first cross-tile relay fixed point.  Unsupported view
bootstrap modes fail before CUDA workers start rather than falling back to the
inferior box/composite experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Mapping, Sequence

from .lta_cpu import bounded_lta_host_threads, host_native_thread_counts
from .lta_outputs import (
    LtaArtifactReceipt,
    LtaCheckpointArtifact,
    LtaLayerRecord,
    LtaPublicationReceipt,
    LtaRecompositionOp,
    write_complete_lta_manifest,
    write_json_atomically,
)
from .lta_postprocessing import (
    fill_completed_view_holes_2d_inplace,
    fill_merged_seed_mask_holes_2d,
    finalize_lta_native_union,
    write_global_final_output_nrrd,
    write_lta_postprocessing_checkpoint,
)
from .lta_propagation import (
    LtaMaskSeed,
    LtaSeedProvenance,
    partition_mask_seed_sessions,
    read_seed_artifact,
    write_seed_artifact,
)
from .lta_rendering import (
    LtaPhysicalViewCacheRef,
    reference_existing_physical_view_cache,
    union_tile_chunk_into_view,
)
from .lta_runtime import LtaRunPlan, LtaRuntimeViewPlan, LtaTileGridPlan
from .lta_sam import LTA_SESSION_FRAMES, revalidate_local_sam_bundle
from .lta_scheduler import (
    LtaSessionWork,
    LtaSpatialRelayKey,
    LtaViewAffinityScheduler,
    LtaViewKey,
)
from .lta_tile_tracking import LtaLineageId
from .lta_tiles import (
    TilePlan,
    eight_neighbor_graph,
    polygon_intersects_tile,
    rasterize_polygons,
    rasterize_polygons_to_shape,
    transform_polygon_to_tile,
)
from .lta_windows import (
    AnchorDomain,
    WindowPlan,
    owned_frame_range,
    plan_directional_windows,
    plan_domain_windows,
)
from .lta_workers import LtaWorkerInit, LtaWorkerPool, LtaWorkerResult, LtaWorkerTask


PRODUCTION_LTA_TILE_SIZE = 1008
PRODUCTION_LTA_PROFILE = "auto"
PRODUCTION_LTA_RELAY_MIN_PIXELS = 16
PRODUCTION_LTA_RELAY_MIN_PROBABILITY = 0.5
PRODUCTION_LTA_TASK_TIMEOUT_SECONDS = 4 * 60 * 60
_GIB = 1024**3


def _seed_batches(seeds: Sequence[LtaMaskSeed]) -> tuple[tuple[LtaMaskSeed, ...], ...]:
    """Split one prompt into deterministic, overlap-compatible SAM sessions."""

    return partition_mask_seed_sessions(seeds)


def _relay_generation_bound(view_plan: LtaRuntimeViewPlan) -> int:
    """Set a practical cap; remaining monotone growth fails closed at the cap."""

    total_tiles = sum(len(grid.tiles) for grid in view_plan.tile_grids)
    return 2 * int(view_plan.frame_count) * int(total_tiles)


def _storage_capacity_probe(path: Path) -> tuple[Path, int, int]:
    """Resolve an existing filesystem probe without creating the target tree."""

    probe = Path(path).resolve(strict=False)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe, int(probe.stat().st_dev), int(shutil.disk_usage(probe).free)


def _preflight_lta_storage(
    plan: LtaRunPlan,
    view_plan: LtaRuntimeViewPlan,
) -> dict[str, object]:
    """Reserve scratch and persistent NRRD capacity before source decode."""

    voxel_bytes = math.prod(int(value) for value in plan.volumes[0].source_shape_tyx)
    base_uint8_volumes = 4 * voxel_bytes
    filter_workspace = (
        4 * voxel_bytes
        if bool(plan.postprocessing.get("enable_3d_void_fill", False))
        or bool(plan.postprocessing.get("gaussian_smoothing_enabled", False))
        else 0
    )
    keep_replacement = (
        voxel_bytes if int(plan.postprocessing.get("keep_objects", 0)) > 0 else 0
    )
    largest_worker_union = max(
        (
            min(int(view_plan.frame_count), LTA_SESSION_FRAMES) * int(tile.size) * int(tile.size)
            for grid in view_plan.tile_grids
            for tile in grid.tiles
        ),
        default=0,
    )
    # Each completed dense union is reduced and removed before its device is
    # reused; only compact manifests wait for ordered logical commits.
    worker_union_reserve = len(plan.device_ids) * largest_worker_union
    checkpoint_count = (
        1
        + int(bool(plan.postprocessing.get("enable_3d_void_fill", False)))
        + int(bool(plan.postprocessing.get("gaussian_smoothing_enabled", False)))
        + int(int(plan.postprocessing.get("keep_objects", 0)) > 0)
    )
    output_nrrd_count = checkpoint_count + 1
    public_nrrd_reserve = output_nrrd_count * (
        int(math.ceil(voxel_bytes * 1.01)) + 64 * 1024
    )
    scratch_estimated = (
        base_uint8_volumes
        + filter_workspace
        + keep_replacement
        + worker_union_reserve
    )
    scratch_probe, scratch_filesystem, scratch_free = _storage_capacity_probe(plan.temp_root)
    output_probe, output_filesystem, output_free = _storage_capacity_probe(plan.output_root)
    shared_filesystem = scratch_filesystem == output_filesystem

    def reservation(role: str, probe: Path, free: int, estimated: int) -> dict[str, object]:
        headroom = max(1 * _GIB, int(math.ceil(estimated * 0.10)))
        required = estimated + headroom
        if free < required:
            raise RuntimeError(
                f"LTA {role} capacity preflight failed before decode: "
                f"need at least {required / _GIB:.1f} GiB including headroom, "
                f"but {free / _GIB:.1f} GiB is free at {probe}. "
                "Select sufficient storage with --temp and --output. "
                "Relay artifacts, labels, and overlays may require more."
            )
        return {
            "role": role,
            "capacity_probe_path": str(probe),
            "free_bytes_at_preflight": free,
            "estimated_bytes_before_headroom": estimated,
            "headroom": headroom,
            "required_bytes_with_headroom": required,
        }

    if shared_filesystem:
        scratch_reservation = output_reservation = reservation(
            "scratch and output", scratch_probe, min(scratch_free, output_free),
            scratch_estimated + public_nrrd_reserve,
        )
        reservations = [scratch_reservation]
    else:
        scratch_reservation = reservation("scratch", scratch_probe, scratch_free, scratch_estimated)
        output_reservation = reservation("output", output_probe, output_free, public_nrrd_reserve)
        reservations = [scratch_reservation, output_reservation]
    return {
        "scratch_root": str(plan.temp_root),
        "output_root": str(plan.output_root),
        "capacity_probe_path": str(scratch_probe),
        "output_capacity_probe_path": str(output_probe),
        "lifecycle": "scratch_ephemeral_output_persistent",
        "scratch_and_output_share_filesystem": shared_filesystem,
        "filesystem_reservations": reservations,
        "free_bytes_at_preflight": scratch_reservation["free_bytes_at_preflight"],
        "estimated_bytes_before_headroom": scratch_reservation["estimated_bytes_before_headroom"],
        "required_bytes_with_headroom": scratch_reservation["required_bytes_with_headroom"],
        "output_required_bytes_with_headroom": output_reservation["required_bytes_with_headroom"],
        "components": {
            "four_full_uint8_volumes": int(base_uint8_volumes),
            "largest_selected_dense_filter_workspace": int(filter_workspace),
            "keep_objects_replacement": int(keep_replacement),
            "bounded_worker_union_reserve": int(worker_union_reserve),
            "compressed_public_nrrd_upper_bound": int(public_nrrd_reserve),
            "headroom": scratch_reservation["headroom"],
        },
        "maximum_uncommitted_worker_unions": len(plan.device_ids),
        "postprocessing_checkpoint_count": checkpoint_count,
        "output_nrrd_count_including_final": output_nrrd_count,
        "compressed_postprocessing_checkpoints_included": True,
        "compressed_final_nrrd_included": True,
        "relay_artifacts_labels_and_overlays_included": False,
    }


@dataclass(frozen=True)
class _RelayMaskRevision:
    shape_hw: tuple[int, int]
    packed_mask: Any
    revision_sha256: str
    visited_tile_indices: tuple[int, ...]
    tracker_probability: float


def _relay_mask_revision(mask: object) -> tuple[tuple[int, int], object, str]:
    import numpy as np

    binary = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    if binary.ndim != 2 or not bool(binary.any()):
        raise ValueError("a relay mask revision must contain 2D foreground")
    shape = tuple(int(value) for value in binary.shape)
    packed = np.ascontiguousarray(np.packbits(binary.reshape(-1)), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(f"{shape[0]}x{shape[1]}:".encode("ascii"))
    digest.update(memoryview(packed).cast("B"))
    packed.setflags(write=False)
    return shape, packed, digest.hexdigest()


def _accumulate_relay_seed_revision(
    event_key: LtaSpatialRelayKey,
    seed: LtaMaskSeed,
    revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision],
) -> tuple[LtaMaskSeed, str] | None:
    """Admit only new foreground and return the accumulated event seed."""

    import numpy as np

    if event_key.mask_revision_sha256 != "0" * 64:
        raise ValueError("relay event keys must use the zero revision placeholder")
    shape, current_packed, _current_digest = _relay_mask_revision(seed.mask)
    previous = revisions.get(event_key)
    previous_digest = None
    if previous is None:
        accumulated_packed = np.asarray(current_packed, dtype=np.uint8).copy()
        delta_pixels = int(np.unpackbits(accumulated_packed).sum())
        visited = seed.visited_tile_indices
        probability = seed.tracker_probability
    else:
        if previous.shape_hw != shape or previous.packed_mask.shape != current_packed.shape:
            raise RuntimeError("one relay event changed mask shape across generations")
        current_array = np.asarray(current_packed, dtype=np.uint8)
        previous_array = np.asarray(previous.packed_mask, dtype=np.uint8)
        delta = np.bitwise_and(current_array, np.bitwise_not(previous_array))
        delta_pixels = int(np.unpackbits(delta).sum())
        if delta_pixels == 0:
            return None
        accumulated_packed = np.bitwise_or(previous_array, current_array)
        previous_digest = previous.revision_sha256
        visited = tuple(
            sorted(set(previous.visited_tile_indices) | set(seed.visited_tile_indices))
        )
        probability = max(previous.tracker_probability, seed.tracker_probability)

    accumulated_mask = np.unpackbits(
        accumulated_packed,
        count=shape[0] * shape[1],
    ).reshape(shape)
    filled = fill_merged_seed_mask_holes_2d(
        accumulated_mask,
        context={
            "destination_tile_index": event_key.destination_tile_index,
            "temporal_direction": event_key.temporal_direction,
            "frame_index": event_key.frame_index,
            "relay_generation": seed.relay_generation,
            "accumulated_revision": True,
        },
    )
    shape, packed, digest = _relay_mask_revision(filled.mask)
    revisions[event_key] = _RelayMaskRevision(
        shape_hw=shape,
        packed_mask=packed,
        revision_sha256=digest,
        visited_tile_indices=tuple(visited),
        tracker_probability=float(probability),
    )
    accumulated_pixels = int(np.unpackbits(np.asarray(packed, dtype=np.uint8)).sum())
    return (
        LtaMaskSeed(
            lineage=seed.lineage,
            frame_index=seed.frame_index,
            object_id=seed.object_id,
            mask=filled.mask,
            provenance=seed.provenance,
            tracker_probability=float(probability),
            relay_generation=seed.relay_generation,
            visited_tile_indices=tuple(visited),
            source_receipt={
                **dict(seed.source_receipt),
                "relay_mask_revision_sha256": digest,
                "previous_relay_mask_revision_sha256": previous_digest,
                "new_foreground_pixels": delta_pixels,
                "accumulated_foreground_pixels": accumulated_pixels,
                "accumulated_hole_fill": filled.receipt.manifest_record(),
            },
        ),
        digest,
    )


@dataclass(frozen=True)
class LtaProductionResult:
    plan: LtaRunPlan
    manifest_path: Path
    final_nrrd_path: Path
    terminal_union_foreground_voxels: int
    worker_pids: Mapping[int, int]
    relay_generations: int


@dataclass(frozen=True)
class _PlannedChain:
    work: LtaSessionWork
    payload: Mapping[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact_path(
    path: str | Path,
    *,
    expected_sha256: object,
    description: str,
) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"{description} is not a regular file: {resolved}")
    if _sha256_file(resolved) != str(expected_sha256):
        raise RuntimeError(f"{description} digest changed: {resolved}")
    return resolved


def _artifact_ownership_path(path: PurePath) -> PurePath:
    """Compare resolved Windows paths without DOS/UNC namespace spelling differences.

    Only standard extended DOS and UNC prefixes are equivalent here. Device
    namespaces and volume GUID paths retain their spelling and fail closed
    against a normal filesystem root. This never changes the path used for I/O.
    """

    if not isinstance(path, PureWindowsPath):
        return path
    value = str(path)
    if value[:8].casefold() == "\\\\?\\unc\\":
        return PureWindowsPath("\\\\" + value[8:])
    if (
        value.startswith("\\\\?\\") and len(value) >= 7
        and value[4].isascii() and value[4].isalpha() and value[5:7] == ":\\"
    ):
        return PureWindowsPath(value[4:])
    return path


def _unlink_consumed_temp_artifacts(
    paths: Sequence[str | Path],
    *,
    temp_root: Path,
) -> None:
    """Unlink exact, already-consumed files only after validating their scope."""

    root = Path(temp_root).resolve(strict=True)
    ownership_root = _artifact_ownership_path(root)
    resolved: list[Path] = []
    seen: set[PurePath] = set()
    for path in paths:
        candidate = Path(path).resolve(strict=True)
        ownership_candidate = _artifact_ownership_path(candidate)
        try:
            ownership_candidate.relative_to(ownership_root)
        except ValueError as exc:
            raise RuntimeError(
                f"refusing to delete consumed LTA artifact outside temp root: {candidate}"
            ) from exc
        if not candidate.is_file():
            raise RuntimeError(f"consumed LTA artifact is not a regular file: {candidate}")
        if ownership_candidate not in seen:
            seen.add(ownership_candidate)
            resolved.append(candidate)
    for candidate in resolved:
        candidate.unlink()


def _view_cache_manifest_record(cache_ref: LtaPhysicalViewCacheRef) -> dict[str, object]:
    """Describe the ephemeral shared cache without publishing a dead path."""

    return {
        "physical_view_id": cache_ref.physical_view_id,
        "identity_sha256": cache_ref.identity_sha256,
        "shape": list(cache_ref.shape),
        "dtype": cache_ref.dtype,
        "size_bytes": cache_ref.size_bytes,
        "lifecycle": "ephemeral_removed_before_manifest_publication",
    }


def _execution_run_plan_manifest_record(plan: LtaRunPlan) -> dict[str, object]:
    """Retain preflight provenance without publishing its unused work schedule."""

    record = plan.manifest_record()
    record["device_schedule"] = {
        "status": "superseded",
        "superseded_by": "execution.device_schedule",
        "reason": (
            "production execution replans full-view chains for each authoritative anchor "
            "and dynamically admitted spatial relays"
        ),
    }
    return record


def _safe_token(value: object) -> str:
    text = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value)
    ).strip("_")
    return text[:80] or "lta"


def _tile_payload(tile: TilePlan) -> dict[str, object]:
    return {
        "left": tile.left,
        "top": tile.top,
        "size": tile.size,
        "source_width": tile.source_width,
        "source_height": tile.source_height,
        "row": tile.row,
        "column": tile.column,
    }


def _windows_payload(windows: Sequence[object]) -> list[dict[str, object]]:
    return [dict(asdict(window)) for window in windows]


def _require_supported_runtime(plan: LtaRunPlan) -> tuple[object, object, LtaRuntimeViewPlan]:
    if plan.bundle.model_version != "sam3.1":
        raise ValueError("production mask-injected LTA requires SAM 3.1")
    if plan.sam_execution != "video":
        raise ValueError(
            "production LTA uses authoritative mask-injected video propagation; "
            "--sam_execution image is not supported"
        )
    if len(plan.discovery.target_volumes) != 1 or len(plan.volumes) != 1:
        raise ValueError("production LTA currently requires exactly one target volume")
    source = plan.discovery.target_volumes[0]
    volume_plan = plan.volumes[0]
    runtime_views = tuple(volume_plan.runtime_views)
    if any(
        str(getattr(getattr(view, "runtime_view", None), "family", "")) == "spherical"
        for view in runtime_views
    ):
        raise ValueError(
            "Spherical QSC views are available for LTA planning and rendering, but production "
            "mask injection requires qualification for spherical geometry; use "
            "--enable_cartesian transverse --angle 0 for production execution"
        )
    if len(runtime_views) != 1:
        raise ValueError(
            "the qualified production mask-injection path currently requires exactly "
            "one runtime view; use --enable_cartesian transverse --angle 0"
        )
    view_plan = runtime_views[0]
    if view_plan.physical_view_id != "transverse" or abs(view_plan.tta_angle_deg) > 1e-12:
        raise ValueError(
            "non-Transverse and nonzero-angle mask injection requires provisional-volume "
            "bootstrap qualification; use transverse with --angle 0"
        )
    if view_plan.runtime_view is None:
        raise RuntimeError("LTA planning did not retain its runtime ViewInfo")
    if not view_plan.tile_grids:
        raise ValueError(
            "full-resolution production LTA requires --enable_tile 1008:STRIDE"
        )
    if any(grid.tile_size != PRODUCTION_LTA_TILE_SIZE for grid in view_plan.tile_grids):
        raise ValueError(
            f"production mask injection requires {PRODUCTION_LTA_TILE_SIZE}-pixel tiles"
        )
    return source, volume_plan, view_plan


def _aligned_annotations(plan: LtaRunPlan, source: object) -> tuple[object, ...]:
    """Merge native labels with explicitly index-aligned exemplar annotations.

    YOLO polygon coordinates are normalized to the target frame.  The raster
    stored beside an external label is provenance for that annotation, not the
    geometry space used for mask injection; exports commonly resize that
    preview (for example, the qualified 1008-square Full export for a
    3024-by-3064 target).
    """

    from .lta_inputs import AnnotationState, FrameAnnotation, SourceRole, YoloPolygon

    annotations = [item for item in source.annotations if item.polygons]
    seen = {
        (
            int(item.frame_position),
            str(item.label_sha256),
            int(polygon.row_index),
            tuple(polygon.points),
        )
        for item in annotations
        for polygon in item.polygons
    }
    origin = int(plan.exemplar_index_origin)
    for exemplar in plan.discovery.positive_pool:
        if exemplar.source_role is not SourceRole.EXEMPLAR:
            continue
        frame = int(exemplar.encoded_frame_index) - origin
        if not 0 <= frame < int(source.frame_count):
            raise ValueError(
                f"aligned exemplar index {exemplar.encoded_frame_index} maps outside "
                f"target frame range with origin {origin}"
            )
        polygon = YoloPolygon(
            class_id=int(exemplar.class_id),
            row_index=int(exemplar.label_row_index),
            points=tuple(exemplar.polygon),
            box_xyxy=tuple(exemplar.box_xyxy),
            box_cxcywh=tuple(exemplar.box_cxcywh),
            normalized_area=float(exemplar.normalized_area),
        )
        key = (
            frame,
            str(exemplar.label_sha256),
            int(polygon.row_index),
            tuple(polygon.points),
        )
        if key in seen:
            continue
        seen.add(key)
        annotations.append(
            FrameAnnotation(
                encoded_index=int(exemplar.encoded_frame_index),
                frame_position=frame,
                state=AnnotationState.FOREGROUND,
                label_path=Path(exemplar.label_path),
                label_sha256=str(exemplar.label_sha256),
                polygons=(polygon,),
            )
        )
    if not annotations:
        raise ValueError(
            "mask-injected LTA requires directly addressable target labels or "
            "index-aligned --exemplar image/YOLO pairs"
        )
    return tuple(
        sorted(
            annotations,
            key=lambda item: (
                int(item.frame_position),
                str(item.label_path),
                tuple(int(polygon.row_index) for polygon in item.polygons),
            ),
        )
    )


def _aligned_known_background_frames(
    plan: LtaRunPlan,
    source: object,
    positive_annotations: Sequence[object],
) -> tuple[int, ...]:
    """Resolve explicit empty labels as audit-only target-frame assertions."""

    from .lta_inputs import AnnotationState

    frames = {
        int(annotation.frame_position)
        for annotation in source.annotations
        if annotation.state is AnnotationState.KNOWN_BACKGROUND
    }
    origin = int(plan.exemplar_index_origin)
    for volume in plan.discovery.exemplar_volumes:
        for annotation in volume.annotations:
            if annotation.state is not AnnotationState.KNOWN_BACKGROUND:
                continue
            frame = int(annotation.encoded_index) - origin
            if 0 <= frame < int(source.frame_count):
                frames.add(frame)
    positive_frames = {
        int(annotation.frame_position)
        for annotation in positive_annotations
        if annotation.polygons
    }
    return tuple(sorted(frames - positive_frames))


def _known_background_audit(
    terminal_union: object,
    frame_indices: Sequence[int],
) -> dict[str, object]:
    import numpy as np

    volume = np.asarray(terminal_union)
    per_frame = []
    for frame_index in frame_indices:
        pixels = int(np.count_nonzero(volume[int(frame_index)]))
        if pixels:
            per_frame.append(
                {
                    "frame_index": int(frame_index),
                    "predicted_foreground_pixels": pixels,
                }
            )
    return {
        "policy": "audit_only_no_subtraction",
        "known_background_frame_count": len(tuple(frame_indices)),
        "frames_with_predicted_foreground": len(per_frame),
        "predicted_foreground_pixels": sum(
            int(record["predicted_foreground_pixels"]) for record in per_frame
        ),
        "nonempty_frames": per_frame,
    }


def _prepare_output_roots(plan: LtaRunPlan) -> None:
    output = plan.output_root
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"LTA output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan.temp_root.mkdir(parents=True, exist_ok=False)
    print(f"LTA scratch directory created: {plan.temp_root}", flush=True)


def _materialize_source_volume(source: object, *, path: Path):
    import numpy as np

    shape = (int(source.frame_count), int(source.height), int(source.width))
    if source.video_path is not None:
        from .media import decode_video_to_memmap_gray8

        return decode_video_to_memmap_gray8(
            Path(source.video_path),
            Path(path),
            shape[0],
            shape[2],
            shape[1],
            overwrite=True,
            prefer_memory=False,
            reserve_bytes=0,
            strict_frame_count=True,
        )
    media = tuple(sorted(source.media, key=lambda item: int(item.frame_position)))
    if len(media) != shape[0]:
        raise RuntimeError("image-sequence discovery count changed before execution")
    from PIL import Image

    volume = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    try:
        for expected_frame, item in enumerate(media):
            if int(item.frame_position) != expected_frame:
                raise RuntimeError("image-sequence frame positions are not contiguous")
            with Image.open(item.path) as image:
                frame = np.asarray(image.convert("L"), dtype=np.uint8)
            if frame.shape != shape[1:]:
                raise RuntimeError(
                    f"image {item.path} shape {frame.shape} != {shape[1:]}"
                )
            volume[expected_frame] = frame
        volume.flush()
        return volume
    except BaseException:
        mmap_obj = getattr(volume, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
        raise


def _hard_positive_volume(source: object, annotations: Sequence[object], *, path: Path):
    import numpy as np

    shape = (int(source.frame_count), int(source.height), int(source.width))
    volume = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    polygons_by_frame: dict[int, list[object]] = {}
    for annotation in annotations:
        if not annotation.polygons:
            continue
        polygons_by_frame.setdefault(int(annotation.frame_position), []).extend(
            annotation.polygons
        )
    for frame_index, polygons in sorted(polygons_by_frame.items()):
        mask = rasterize_polygons_to_shape(
            polygons,
            height=shape[1],
            width=shape[2],
        )
        volume[frame_index] |= np.asarray(mask, dtype=np.uint8)
    volume.flush()
    return volume


def _source_identity(source: object) -> str:
    if source.video_identity_sha256:
        return str(source.video_identity_sha256)
    digest = hashlib.sha256()
    for item in source.media:
        digest.update(str(item.identity_sha256).encode("ascii"))
    return digest.hexdigest()


def _annotation_tile_owner_key(annotation: object, polygon: object) -> tuple[object, ...]:
    return (
        int(annotation.frame_position),
        str(annotation.label_sha256),
        int(polygon.row_index),
        tuple(polygon.points),
    )


def _authoritative_tile_owners(
    annotations: Sequence[object],
    grid: LtaTileGridPlan,
) -> dict[tuple[object, ...], tuple[int, ...]]:
    """Choose one full-mask owner, or every tile required by a clipped mask."""

    import numpy as np

    owners: dict[tuple[object, ...], tuple[int, ...]] = {}
    for annotation in annotations:
        for polygon in annotation.polygons:
            candidates: list[tuple[int, int, int]] = []
            containing: list[tuple[int, int, int]] = []
            min_x, min_y, max_x, max_y = (
                float(value) for value in polygon.box_xyxy
            )
            for tile_index, tile in enumerate(grid.tiles):
                if not polygon_intersects_tile(polygon, tile):
                    continue
                local = transform_polygon_to_tile(polygon, tile)
                mask = np.asarray(
                    rasterize_polygons((local,), size=tile.size),
                    dtype=np.bool_,
                )
                rows, columns = np.nonzero(mask)
                if not len(rows):
                    continue
                margin = min(
                    int(rows.min()),
                    int(columns.min()),
                    int(tile.size - 1 - rows.max()),
                    int(tile.size - 1 - columns.max()),
                )
                score = (int(mask.sum()), margin, -int(tile_index))
                candidates.append(score)
                if (
                    min_x >= float(tile.left) / float(tile.source_width)
                    and max_x <= float(tile.left + tile.size) / float(tile.source_width)
                    and min_y >= float(tile.top) / float(tile.source_height)
                    and max_y <= float(tile.top + tile.size) / float(tile.source_height)
                ):
                    containing.append(score)
            if not candidates:
                raise RuntimeError(
                    "an authoritative polygon has no raster support in the complete tile grid"
                )
            if containing:
                best = max(containing)
                selected = (-best[2],)
            else:
                # No tile contains the complete polygon. Every clipped local
                # fragment is authoritative and must seed its own tracker;
                # relays cannot reconstruct pixels absent from the owner crop.
                selected = tuple(sorted(-candidate[2] for candidate in candidates))
            owners[_annotation_tile_owner_key(annotation, polygon)] = selected
    return owners


def _seeds_for_tile(
    source: object,
    annotations: Sequence[object],
    view_plan: LtaRuntimeViewPlan,
    grid: LtaTileGridPlan,
    tile_index: int,
    *,
    authoritative_tile_owners: Mapping[tuple[object, ...], Sequence[int]] | None = None,
) -> dict[int, tuple[LtaMaskSeed, ...]]:
    import numpy as np

    tile = grid.tiles[int(tile_index)]
    by_frame: dict[int, list[LtaMaskSeed]] = {}
    for annotation in annotations:
        visible: list[tuple[object, object]] = []
        for polygon in annotation.polygons:
            selected_owners = (
                None
                if authoritative_tile_owners is None
                else authoritative_tile_owners[
                    _annotation_tile_owner_key(annotation, polygon)
                ]
            )
            if selected_owners is not None and int(tile_index) not in {
                int(value) for value in selected_owners
            }:
                continue
            if not polygon_intersects_tile(polygon, tile):
                continue
            local = transform_polygon_to_tile(polygon, tile)
            mask = rasterize_polygons((local,), size=tile.size)
            if bool(np.asarray(mask, dtype=bool).any()):
                visible.append((polygon, mask))
        for polygon, mask in visible:
            object_id = len(by_frame.get(int(annotation.frame_position), ()))
            lineage = LtaLineageId(
                volume_id=str(source.volume_id),
                physical_view_id=view_plan.physical_view_id,
                runtime_view_id=view_plan.runtime_view_id,
                tile_config_id=grid.config_id,
                lineage_id=(
                    f"annotation-{int(annotation.frame_position):06d}-"
                    f"{str(annotation.label_sha256)[:12]}-"
                    f"row-{int(polygon.row_index):04d}"
                ),
            )
            by_frame.setdefault(int(annotation.frame_position), []).append(
                LtaMaskSeed(
                    lineage=lineage,
                    frame_index=int(annotation.frame_position),
                    object_id=object_id,
                    mask=mask,
                    provenance=LtaSeedProvenance.AUTHORITATIVE,
                    tracker_probability=1.0,
                    relay_generation=0,
                    visited_tile_indices=(int(tile_index),),
                    source_receipt={
                        "label_path": (
                            None
                            if annotation.label_path is None
                            else str(annotation.label_path)
                        ),
                        "label_sha256": annotation.label_sha256,
                        "label_row_index": int(polygon.row_index),
                        "tile_index": int(tile_index),
                        "authoritative_tile_owner": True,
                        "tile_owner_policy": (
                            "single_full_mask_owner"
                            if selected_owners is not None and len(tuple(selected_owners)) == 1
                            else "all_required_clipped_fragments"
                        ),
                    },
                )
            )
    return {
        frame: tuple(sorted(seeds, key=lambda seed: seed.object_id))
        for frame, seeds in by_frame.items()
    }


def _neighbors_payload(grid: LtaTileGridPlan, tile_index: int) -> list[dict[str, object]]:
    graph = eight_neighbor_graph(grid.tiles)
    return [
        {
            "tile_index": edge.destination_index,
            "direction": edge.direction,
            "overlap_xyxy": list(edge.overlap_xyxy),
            "tile": _tile_payload(grid.tiles[edge.destination_index]),
        }
        for edge in graph[int(tile_index)]
    ]


def _base_chain_payload(
    *,
    work_id: str,
    view_plan: LtaRuntimeViewPlan,
    grid: LtaTileGridPlan,
    tile_index: int,
    cache_ref: LtaPhysicalViewCacheRef,
    seed_path: Path,
    seed_sha256: str,
    windows: Sequence[object],
    output_frame_start: int,
    output_frame_stop: int,
    conf: float,
    empty_frame_limit: int | None,
    relay_generation: int,
) -> dict[str, object]:
    return {
        "work_id": work_id,
        "sequence_id": f"{view_plan.volume_id}::{view_plan.runtime_view_id}",
        "cache_ref": cache_ref.payload(),
        "tile_index": int(tile_index),
        "tile_config_id": grid.config_id,
        "tile": _tile_payload(grid.tiles[int(tile_index)]),
        "neighbors": _neighbors_payload(grid, int(tile_index)),
        "seed_artifact_path": str(seed_path),
        "seed_artifact_sha256": str(seed_sha256),
        "windows": _windows_payload(windows),
        "output_frame_start": int(output_frame_start),
        "output_frame_stop": int(output_frame_stop),
        "conf": float(conf),
        "empty_frame_limit": empty_frame_limit,
        "relay_generation": int(relay_generation),
        "relay_min_pixels": PRODUCTION_LTA_RELAY_MIN_PIXELS,
        "relay_min_probability": PRODUCTION_LTA_RELAY_MIN_PROBABILITY,
    }


def _plan_initial_chains(
    source: object,
    annotations: Sequence[object],
    view_plan: LtaRuntimeViewPlan,
    cache_ref: LtaPhysicalViewCacheRef,
    *,
    temp_root: Path,
    conf: float,
    empty_frame_limit: int | None,
) -> tuple[
    tuple[_PlannedChain, ...],
    dict[tuple[str, int, int], tuple[LtaMaskSeed, ...]],
]:
    chains: list[_PlannedChain] = []
    seed_inventory: dict[tuple[str, int, int], tuple[LtaMaskSeed, ...]] = {}
    plan_order = 0
    view_key = LtaViewKey(view_plan.volume_id, view_plan.physical_view_id)
    for grid in view_plan.tile_grids:
        authoritative_tile_owners = _authoritative_tile_owners(annotations, grid)
        for tile_index, tile in enumerate(grid.tiles):
            by_frame = _seeds_for_tile(
                source,
                annotations,
                view_plan,
                grid,
                tile_index,
                authoritative_tile_owners=authoritative_tile_owners,
            )
            for frame, seeds in by_frame.items():
                seed_inventory[(grid.config_id, tile_index, frame)] = seeds
            # An unrelated positive annotation is not evidence that an existing
            # lineage ended. Until production integrates cross-anchor identity
            # reconciliation, let every anchor propagate over the full view and
            # union its evidence with the other independently seeded lineages.
            for anchor_frame in sorted(by_frame):
                domain = AnchorDomain(
                    anchor_frame=anchor_frame,
                    frame_start=0,
                    frame_stop=view_plan.frame_count,
                )
                windows = plan_domain_windows(domain)
                batches = _seed_batches(by_frame[domain.anchor_frame])
                batch_count = len(batches)
                for batch_index, batch in enumerate(batches):
                    seeds = tuple(
                        LtaMaskSeed(
                            lineage=seed.lineage,
                            frame_index=seed.frame_index,
                            object_id=object_id,
                            mask=seed.mask,
                            provenance=seed.provenance,
                            tracker_probability=seed.tracker_probability,
                            relay_generation=seed.relay_generation,
                            visited_tile_indices=seed.visited_tile_indices,
                            source_receipt=seed.source_receipt,
                        )
                        for object_id, seed in enumerate(batch)
                    )
                    work_id = (
                        f"{view_plan.volume_id}::{view_plan.runtime_view_id}::"
                        f"{grid.config_id}::tile-{tile_index:04d}::"
                        f"anchor-{domain.anchor_frame:06d}::"
                        f"batch-{batch_index:04d}-of-{batch_count:04d}"
                    )
                    seed_artifact = write_seed_artifact(
                        temp_root
                        / "seeds"
                        / "generation-0000"
                        / f"{hashlib.sha256(work_id.encode()).hexdigest()[:20]}.npz",
                        seeds,
                    )
                    work = LtaSessionWork(
                        work_id=work_id,
                        view=view_key,
                        runtime_view_id=view_plan.runtime_view_id,
                        session_index=plan_order,
                        frame_start=domain.frame_start,
                        frame_stop=domain.frame_stop,
                        plan_order=plan_order,
                        estimated_cost=float(domain.frame_count * tile.size * tile.size),
                        projection_key=cache_ref.identity_sha256,
                        tile_index=tile_index,
                        tile_config_id=grid.config_id,
                        tail_eligible=True,
                        relay_generation=0,
                    )
                    payload = _base_chain_payload(
                        work_id=work_id,
                        view_plan=view_plan,
                        grid=grid,
                        tile_index=tile_index,
                        cache_ref=cache_ref,
                        seed_path=seed_artifact.path,
                        seed_sha256=seed_artifact.sha256,
                        windows=windows,
                        output_frame_start=domain.frame_start,
                        output_frame_stop=domain.frame_stop,
                        conf=conf,
                        empty_frame_limit=empty_frame_limit,
                        relay_generation=0,
                    )
                    chains.append(_PlannedChain(work=work, payload=payload))
                    plan_order += 1
    if not chains:
        raise ValueError("no production LTA tile contains a directly addressable mask seed")
    return tuple(chains), seed_inventory


def _plan_window_tasks(
    chains: Sequence[_PlannedChain],
    *,
    first_plan_order: int = 0,
) -> tuple[_PlannedChain, ...]:
    """Expand fresh tracker resets into a dependency graph of bounded tasks."""

    tasks: list[_PlannedChain] = []
    for chain in chains:
        windows = tuple(WindowPlan(**record) for record in chain.payload["windows"])
        previous_by_branch: dict[str, str] = {}
        root_id = f"{chain.work.work_id}::window-0000"
        for index, window in enumerate(windows):
            work_id = f"{chain.work.work_id}::window-{index:04d}"
            predecessor = None if index == 0 else previous_by_branch.get(window.branch, root_id)
            previous_by_branch[window.branch] = work_id
            start, stop = owned_frame_range(window)
            work = replace(
                chain.work,
                work_id=work_id,
                session_index=first_plan_order + len(tasks),
                plan_order=first_plan_order + len(tasks),
                frame_start=window.frame_start,
                frame_stop=window.frame_stop,
                estimated_cost=chain.work.estimated_cost * window.frame_count / chain.work.frame_count,
                dependency_work_ids=() if predecessor is None else (predecessor,),
            )
            payload = {
                **dict(chain.payload),
                "work_id": work_id,
                "chain_work_id": chain.work.work_id,
                "chain_window_count": len(windows),
                "window_index": index,
                "window": dict(asdict(window)),
                "windows": [dict(asdict(window))],
                "predecessor_work_id": predecessor,
                "output_frame_start": start,
                "output_frame_stop": stop,
            }
            if predecessor is not None:
                payload["seed_artifact_path"] = None
                payload["seed_artifact_sha256"] = None
            tasks.append(_PlannedChain(work=work, payload=payload))
    return tuple(tasks)


@dataclass(frozen=True)
class _SkippedWindow:
    reason: str = "halted_empty_dogfood_boundary"


def _authoritative_seed_audit(
    seed_inventory: Mapping[tuple[str, int, int], Sequence[LtaMaskSeed]],
) -> dict[str, object]:
    tiles_by_lineage: dict[str, set[int]] = {}
    for (_config_id, tile_index, _frame), seeds in seed_inventory.items():
        for seed in seeds:
            tiles_by_lineage.setdefault(seed.lineage.token, set()).add(int(tile_index))
    return {
        "policy": (
            "one strongest fully-containing tile; all clipped fragments when no tile "
            "contains the complete polygon"
        ),
        "unique_lineage_count": len(tiles_by_lineage),
        "direct_tile_seed_count": sum(len(seeds) for seeds in seed_inventory.values()),
        "multi_fragment_lineage_count": sum(
            1 for tiles in tiles_by_lineage.values() if len(tiles) > 1
        ),
        "maximum_fragments_per_lineage": max(
            (len(tiles) for tiles in tiles_by_lineage.values()),
            default=0,
        ),
    }


def _prime_authoritative_relay_destinations(
    scheduler: LtaViewAffinityScheduler,
    view: LtaViewKey,
    seed_inventory: Mapping[tuple[str, int, int], Sequence[LtaMaskSeed]],
    relay_mask_revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision],
) -> None:
    """Prime event masks so identical relays cannot reopen authoritative seeds."""

    for (config_id, tile_index, _frame), seeds in seed_inventory.items():
        for seed in seeds:
            for direction in ("forward", "backward"):
                event_key = LtaSpatialRelayKey(
                    view=view,
                    runtime_view_id=seed.lineage.runtime_view_id,
                    tile_config_id=config_id,
                    lineage_id=seed.lineage.lineage_id,
                    destination_tile_index=tile_index,
                    frame_index=seed.frame_index,
                    temporal_direction=direction,
                )
                shape, packed, revision = _relay_mask_revision(seed.mask)
                relay_mask_revisions[event_key] = _RelayMaskRevision(
                    shape_hw=shape,
                    packed_mask=packed,
                    revision_sha256=revision,
                    visited_tile_indices=seed.visited_tile_indices,
                    tracker_probability=seed.tracker_probability,
                )
                scheduler.admit_spatial_relay(
                    replace(event_key, mask_revision_sha256=revision)
                )


def _allocate_view_union(view_plan: LtaRuntimeViewPlan, path: Path):
    import numpy as np

    shape = (view_plan.frame_count, view_plan.frame_height, view_plan.frame_width)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    destination = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    destination.flush()
    return destination


def _load_chain_manifest(result: LtaWorkerResult) -> dict[str, object]:
    path = Path(result.artifact_path).resolve(strict=True)
    if _sha256_file(path) != result.artifact_sha256:
        raise RuntimeError(f"worker manifest changed after completion: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("work_id") != result.work_id:
        raise RuntimeError("worker manifest does not match its completed work")
    return payload


def _consume_chain_manifest(
    manifest: Mapping[str, object],
    *,
    view_union: object,
) -> tuple[tuple[dict[str, object], ...], Path]:
    from .lta_union_artifacts import reduce_union_artifact_into_view

    tile = manifest["tile"]
    tile_xyxy = (
        int(tile["left"]), int(tile["top"]),
        int(tile["left"]) + int(tile["size"]),
        int(tile["top"]) + int(tile["size"]),
    )
    start, stop = (int(value) for value in manifest["output_frame_range"])
    receipt = reduce_union_artifact_into_view(
        manifest["union"], view_union=view_union, tile_xyxy=tile_xyxy,
        frame_start=start, frame_stop=stop,
    )
    return tuple(dict(item) for item in manifest.get("relays", ())), Path(receipt["path"])


def _new_worker_audit() -> dict[str, object]:
    return {
        "ready_worker_count": 0,
        "chain_count": 0,
        "task_count": 0,
        "skipped_window_count": 0,
        "window_count": 0,
        "planned_window_count": 0,
        "tracker_session_count": 0,
        "complete_window_count": 0,
        "halted_window_count": 0,
        "prediction_count": 0,
        "retained_prediction_count": 0,
        "tracker_feature_only_preparations": 0,
        "tracker_feature_fallback_preparations": 0,
        "dogfood_seed_count": 0,
        "hole_fill_added_pixels": 0,
        "injection_hole_fill_added_pixels": 0,
        "prediction_hole_fill_added_pixels": 0,
        "below_confidence_prediction_count": 0,
        "removed_object_observation_frame_count": 0,
        "removed_object_observation_count": 0,
        "empty_frame_termination_count": 0,
        "relay_artifact_count": 0,
        "foreground_pixels_across_chain_unions": 0,
        "profiles_by_device": {},
        "sam_runtime_by_device": {},
        "constrained_batches_by_device": {},
        "visible_devices_by_device": {},
        "cpu_budgets_by_device": {},
        "cpu_runtime_by_device": {},
        "relay_gate": None,
    }


def _accumulate_worker_ready_audit(
    audit: dict[str, object],
    ready: object,
) -> None:
    device = str(int(ready.execution_device_id))
    metadata = ready.metadata
    if not isinstance(metadata, Mapping):
        raise RuntimeError("worker ready metadata is not a mapping")
    profile = metadata.get("profile")
    sam_runtime = metadata.get("sam_runtime")
    constrained = metadata.get("constrained_batches")
    if not isinstance(profile, Mapping) or not isinstance(sam_runtime, Mapping):
        raise RuntimeError("worker ready event omitted profile or SAM provenance")
    if constrained is not None and not isinstance(constrained, Mapping):
        raise RuntimeError("worker ready constrained-batch metadata is not a mapping")
    profiles = audit["profiles_by_device"]
    runtimes = audit["sam_runtime_by_device"]
    constrained_by_device = audit["constrained_batches_by_device"]
    visible = audit["visible_devices_by_device"]
    assert isinstance(profiles, dict)
    assert isinstance(runtimes, dict)
    assert isinstance(constrained_by_device, dict)
    assert isinstance(visible, dict)
    profiles[device] = json.loads(json.dumps(dict(profile), sort_keys=True))
    runtimes[device] = json.loads(json.dumps(dict(sam_runtime), sort_keys=True))
    constrained_by_device[device] = (
        None
        if constrained is None
        else json.loads(json.dumps(dict(constrained), sort_keys=True))
    )
    visible[device] = str(ready.visible_device)
    for key, target in (("cpu_budget", "cpu_budgets_by_device"), ("cpu_runtime", "cpu_runtime_by_device")):
        if key in metadata:
            if not isinstance(metadata[key], Mapping):
                raise RuntimeError(f"worker {key} metadata is not a mapping")
            audit[target][device] = json.loads(json.dumps(dict(metadata[key]), sort_keys=True))
    audit["ready_worker_count"] = int(audit["ready_worker_count"]) + 1


def _accumulate_worker_audit(
    audit: dict[str, object],
    manifest: Mapping[str, object],
    result: LtaWorkerResult,
) -> None:
    """Retain compact execution evidence before task artifacts are deleted."""

    def add(name: str, value: object) -> None:
        audit[name] = int(audit[name]) + int(value)

    device = str(int(result.execution_device_id))
    profile = manifest.get("profile")
    if not isinstance(profile, Mapping):
        raise RuntimeError("worker manifest omitted its resolved device profile")
    profiles = audit["profiles_by_device"]
    assert isinstance(profiles, dict)
    normalized_profile = dict(profile)
    if device in profiles and profiles[device] != normalized_profile:
        raise RuntimeError(f"worker cuda:{device} changed its resolved profile")
    profiles[device] = normalized_profile

    sam_runtime = manifest.get("sam_runtime")
    if not isinstance(sam_runtime, Mapping):
        raise RuntimeError("worker manifest omitted pinned SAM runtime provenance")
    runtimes = audit["sam_runtime_by_device"]
    assert isinstance(runtimes, dict)
    normalized_runtime = dict(sam_runtime)
    if device in runtimes and runtimes[device] != normalized_runtime:
        raise RuntimeError(f"worker cuda:{device} changed its SAM runtime provenance")
    runtimes[device] = normalized_runtime

    constrained = manifest.get("constrained_batches")
    constrained_by_device = audit["constrained_batches_by_device"]
    assert isinstance(constrained_by_device, dict)
    if constrained is not None and not isinstance(constrained, Mapping):
        raise RuntimeError("worker constrained-batch audit is not a mapping")
    normalized_constrained = None if constrained is None else dict(constrained)
    if device in constrained_by_device and constrained_by_device[device] != normalized_constrained:
        raise RuntimeError(f"worker cuda:{device} changed constrained-batch settings")
    constrained_by_device[device] = normalized_constrained

    relay_gate = manifest.get("relay_gate")
    if not isinstance(relay_gate, Mapping):
        raise RuntimeError("worker manifest omitted its spatial relay gate")
    normalized_gate = dict(relay_gate)
    if audit["relay_gate"] is None:
        audit["relay_gate"] = normalized_gate
    elif audit["relay_gate"] != normalized_gate:
        raise RuntimeError("worker tasks used inconsistent spatial relay gates")

    add("task_count", 1)
    if manifest.get("task_granularity") != "window":
        add("chain_count", 1)
    add("relay_artifact_count", len(tuple(manifest.get("relays", ()))))
    add("foreground_pixels_across_chain_unions", manifest.get("foreground_pixels", 0))
    windows = tuple(manifest.get("windows", ()))
    add("window_count", len(windows))
    partition_audit = manifest.get("seed_session_partition", {})
    if partition_audit is not None and not isinstance(partition_audit, Mapping):
        raise RuntimeError("worker seed-session partition audit is not a mapping")
    partition_audit = dict(partition_audit or {})
    add(
        "planned_window_count",
        partition_audit.get("planned_window_count", len(windows)),
    )
    add(
        "tracker_session_count",
        partition_audit.get("tracker_session_count", len(windows)),
    )
    for window in windows:
        if not isinstance(window, Mapping):
            raise RuntimeError("worker window audit record is not a mapping")
        status = str(window.get("status", ""))
        if status == "complete":
            add("complete_window_count", 1)
        elif status.startswith("halted_"):
            add("halted_window_count", 1)
        else:
            raise RuntimeError(f"worker window has unknown status {status!r}")
        add("prediction_count", window.get("prediction_count", 0))
        retained = int(window.get("retained_prediction_count", 0))
        if retained:
            raise RuntimeError("production worker retained dense prediction masks")
        add("retained_prediction_count", retained)
        add("dogfood_seed_count", window.get("dogfood_seed_count", 0))
        add("hole_fill_added_pixels", window.get("hole_fill_added_pixels", 0))
        adapter = window.get("adapter", {})
        if not isinstance(adapter, Mapping):
            raise RuntimeError("worker window omitted its adapter audit mapping")
        feature_preparation = adapter.get("tracker_feature_preparation", {})
        if not isinstance(feature_preparation, Mapping):
            raise RuntimeError("worker tracker-feature audit is not a mapping")
        add("tracker_feature_only_preparations", feature_preparation.get("feature_only_preparations", 0))
        add("tracker_feature_fallback_preparations", feature_preparation.get("fallback_preparations", 0))
        add(
            "injection_hole_fill_added_pixels",
            adapter.get("seed_hole_fill_added_pixels", 0),
        )
        add(
            "prediction_hole_fill_added_pixels",
            adapter.get("prediction_hole_fill_added_pixels", 0),
        )
        tracker_filter = adapter.get("tracker_confidence_filter", {})
        if not isinstance(tracker_filter, Mapping):
            raise RuntimeError("worker tracker-confidence audit is not a mapping")
        add(
            "below_confidence_prediction_count",
            int(adapter.get("production_below_confidence_prediction_count", 0))
            + int(tracker_filter.get("below_threshold_object_observation_count", 0)),
        )
        removed_observations = tuple(adapter.get("removed_object_observations", ()))
        add("removed_object_observation_frame_count", len(removed_observations))
        add(
            "removed_object_observation_count",
            sum(
                len(tuple(record.get("object_ids", ())))
                for record in removed_observations
                if isinstance(record, Mapping)
            ),
        )
        add(
            "empty_frame_termination_count",
            len(tuple(adapter.get("empty_frame_termination_receipts", ()))),
        )


def _merge_half_open_ranges(
    current: Sequence[Sequence[int]],
    incoming: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    values = sorted(
        (int(item[0]), int(item[1]))
        for item in (*tuple(current), *tuple(incoming))
    )
    merged: list[list[int]] = []
    for start, stop in values:
        if start < 0 or stop <= start:
            raise RuntimeError("worker emitted an invalid active-frame range")
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return tuple((start, stop) for start, stop in merged)


def _accumulate_authoritative_active_coverage(
    coverage: dict[tuple[str, int, str], tuple[tuple[int, int], ...]],
    work: LtaSessionWork,
    manifest: Mapping[str, object],
) -> None:
    if work.relay_generation != 0:
        return
    if work.tile_config_id is None or work.tile_index is None:
        raise RuntimeError("authoritative LTA work omitted its tile identity")
    records = manifest.get("lineage_active_frame_ranges", {})
    if not isinstance(records, Mapping):
        raise RuntimeError("worker lineage active coverage is not a mapping")
    for lineage_token, ranges in records.items():
        key = (str(work.tile_config_id), int(work.tile_index), str(lineage_token))
        coverage[key] = _merge_half_open_ranges(
            coverage.get(key, ()),
            tuple(ranges),  # type: ignore[arg-type]
        )


def _lineage_from_record(record: Mapping[str, object]) -> LtaLineageId:
    return LtaLineageId(
        volume_id=str(record["volume_id"]),
        physical_view_id=str(record["physical_view_id"]),
        runtime_view_id=str(record["runtime_view_id"]),
        tile_config_id=str(record["tile_config_id"]),
        lineage_id=str(record["lineage_id"]),
    )


def _select_and_merge_relay_group(
    records: Sequence[Mapping[str, object]],
    *,
    direction: str,
    destination_tile_index: int,
    generation: int,
) -> LtaMaskSeed:
    import numpy as np

    selected_frame = (
        max(int(record["frame_index"]) for record in records)
        if direction == "forward"
        else min(int(record["frame_index"]) for record in records)
    )
    candidates = [record for record in records if int(record["frame_index"]) == selected_frame]
    source_seeds = [
        read_seed_artifact(
            str(record["seed_artifact_path"]),
            expected_sha256=str(record["seed_artifact_sha256"]),
        )[0]
        for record in candidates
    ]
    masks = tuple(np.asarray(seed.mask, dtype=bool) for seed in source_seeds)
    merged = np.logical_or.reduce(masks)
    filled = fill_merged_seed_mask_holes_2d(
        merged,
        context={
            "destination_tile_index": int(destination_tile_index),
            "temporal_direction": direction,
            "relay_generation": int(generation),
        },
    )
    lineage = source_seeds[0].lineage
    if any(seed.lineage != lineage for seed in source_seeds):
        raise RuntimeError("relay merge group contains multiple lineages")
    visited = tuple(
        sorted(
            {
                tile
                for seed in source_seeds
                for tile in seed.visited_tile_indices
            }
            | {int(destination_tile_index)}
        )
    )
    return LtaMaskSeed(
        lineage=lineage,
        frame_index=selected_frame,
        object_id=0,
        mask=filled.mask,
        provenance=LtaSeedProvenance.SPATIAL_RELAY,
        tracker_probability=max(seed.tracker_probability for seed in source_seeds),
        relay_generation=int(generation),
        visited_tile_indices=visited,
        source_receipt={
            "merged_relay_count": len(source_seeds),
            "hole_fill": filled.receipt.manifest_record(),
            "source_artifacts": [str(record["seed_artifact_path"]) for record in candidates],
        },
    )


def _verified_relay_artifact_paths(
    records: Sequence[Mapping[str, object]],
) -> tuple[Path, ...]:
    return tuple(
        _verified_artifact_path(
            str(record["seed_artifact_path"]),
            expected_sha256=record["seed_artifact_sha256"],
            description="worker relay-seed artifact",
        )
        for record in records
    )


def _plan_relay_generation(
    records: Sequence[Mapping[str, object]],
    *,
    generation: int,
    view_plan: LtaRuntimeViewPlan,
    cache_ref: LtaPhysicalViewCacheRef,
    scheduler: LtaViewAffinityScheduler,
    relay_mask_revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision],
    temp_root: Path,
    conf: float,
    empty_frame_limit: int | None,
    first_plan_order: int,
) -> tuple[_PlannedChain, ...]:
    # Validate every emitted relay, including arrivals that an idempotency key
    # will deliberately discard.  Cleanup below is therefore restricted to
    # exact artifacts whose worker receipts have been proven.
    relay_artifacts = _verified_relay_artifact_paths(records)
    grid_by_id = {grid.config_id: grid for grid in view_plan.tile_grids}
    by_relay_key: dict[LtaSpatialRelayKey, list[Mapping[str, object]]] = {}
    view_key = LtaViewKey(view_plan.volume_id, view_plan.physical_view_id)
    for record in records:
        lineage = _lineage_from_record(record["lineage"])  # type: ignore[arg-type]
        destination_tile_index = int(record["destination_tile_index"])
        frame_index = int(record["frame_index"])
        key = LtaSpatialRelayKey(
            view=view_key,
            runtime_view_id=lineage.runtime_view_id,
            tile_config_id=lineage.tile_config_id,
            lineage_id=lineage.lineage_id,
            destination_tile_index=destination_tile_index,
            frame_index=frame_index,
            temporal_direction=str(record["temporal_direction"]),
        )
        by_relay_key.setdefault(key, []).append(record)

    accepted: list[tuple[LtaSpatialRelayKey, LtaMaskSeed]] = []
    for event_key in sorted(by_relay_key):
        seed = _select_and_merge_relay_group(
            by_relay_key[event_key],
            direction=event_key.temporal_direction,
            destination_tile_index=event_key.destination_tile_index,
            generation=generation,
        )
        accumulated = _accumulate_relay_seed_revision(
            event_key,
            seed,
            relay_mask_revisions,
        )
        if accumulated is None:
            continue
        accumulated_seed, revision = accumulated
        key = replace(event_key, mask_revision_sha256=revision)
        if not scheduler.admit_spatial_relay(key):
            raise RuntimeError("a new monotone relay revision reused an admitted digest")
        accepted.append((key, accumulated_seed))

    # Multiplex lineages that share a destination, prompt, and direction.
    grouped: dict[tuple[str, int, int, str], list[tuple[LtaSpatialRelayKey, LtaMaskSeed]]] = {}
    for key, seed in accepted:
        group_key = (
            key.tile_config_id,
            key.destination_tile_index,
            seed.frame_index,
            key.temporal_direction,
        )
        grouped.setdefault(group_key, []).append((key, seed))

    planned: list[_PlannedChain] = []
    plan_order = int(first_plan_order)
    for (config_id, tile_index, prompt_frame, direction), values in sorted(grouped.items()):
        grid = grid_by_id.get(config_id)
        if grid is None:
            raise RuntimeError(f"relay references unknown tile configuration {config_id}")
        windows = plan_directional_windows(
            frame_start=0,
            frame_stop=view_plan.frame_count,
            prompt_frame=prompt_frame,
            direction=direction,
        )
        work_id_prefix = (
            f"{view_plan.volume_id}::{view_plan.runtime_view_id}::{config_id}::"
            f"tile-{tile_index:04d}::relay-{generation:04d}::{direction}::"
            f"frame-{prompt_frame:06d}"
        )
        output_start = prompt_frame if direction == "forward" else 0
        output_stop = view_plan.frame_count if direction == "forward" else prompt_frame + 1
        tile = grid.tiles[tile_index]
        ordered_values = tuple(sorted(values, key=lambda item: item[0].lineage_id))
        relay_key_by_seed_identity = {
            id(seed): key for key, seed in ordered_values
        }
        if len(relay_key_by_seed_identity) != len(ordered_values):
            raise RuntimeError("one relay seed object was associated with multiple relay keys")
        value_batches = tuple(
            tuple(
                (relay_key_by_seed_identity[id(seed)], seed)
                for seed in seed_batch
            )
            for seed_batch in partition_mask_seed_sessions(
                tuple(seed for _key, seed in ordered_values)
            )
        )
        batch_count = len(value_batches)
        for batch_index, batch in enumerate(value_batches):
            normalized_seeds = tuple(
                LtaMaskSeed(
                    lineage=seed.lineage,
                    frame_index=seed.frame_index,
                    object_id=object_id,
                    mask=seed.mask,
                    provenance=seed.provenance,
                    tracker_probability=seed.tracker_probability,
                    relay_generation=seed.relay_generation,
                    visited_tile_indices=seed.visited_tile_indices,
                    source_receipt=seed.source_receipt,
                )
                for object_id, (_key, seed) in enumerate(batch)
            )
            work_id = (
                f"{work_id_prefix}::"
                f"batch-{batch_index:04d}-of-{batch_count:04d}"
            )
            seed_artifact = write_seed_artifact(
                temp_root
                / "seeds"
                / f"generation-{generation:04d}"
                / f"{hashlib.sha256(work_id.encode()).hexdigest()[:20]}.npz",
                normalized_seeds,
            )
            work = LtaSessionWork(
                work_id=work_id,
                view=view_key,
                runtime_view_id=view_plan.runtime_view_id,
                session_index=plan_order,
                frame_start=output_start,
                frame_stop=output_stop,
                plan_order=plan_order,
                estimated_cost=float((output_stop - output_start) * tile.size * tile.size),
                projection_key=cache_ref.identity_sha256,
                tile_index=tile_index,
                tile_config_id=config_id,
                tail_eligible=True,
                relay_generation=generation,
            )
            payload = _base_chain_payload(
                work_id=work_id,
                view_plan=view_plan,
                grid=grid,
                tile_index=tile_index,
                cache_ref=cache_ref,
                seed_path=seed_artifact.path,
                seed_sha256=seed_artifact.sha256,
                windows=windows,
                output_frame_start=output_start,
                output_frame_stop=output_stop,
                conf=conf,
                empty_frame_limit=empty_frame_limit,
                relay_generation=generation,
            )
            planned.append(_PlannedChain(work=work, payload=payload))
            plan_order += 1
    _unlink_consumed_temp_artifacts(relay_artifacts, temp_root=temp_root)
    return tuple(planned)


def _dispatch_available(
    scheduler: LtaViewAffinityScheduler,
    pool: LtaWorkerPool,
    payload_by_work: Mapping[str, Mapping[str, object]],
    active: dict[str, object],
    active_started: dict[str, float],
    dispatched: list[dict[str, object]],
    *,
    tasks_root: Path,
    trace: object | None = None,
) -> None:
    for device_id in scheduler.device_ids:
        if any(
            int(claim.execution_device_id) == int(device_id)
            for claim in active.values()
        ):
            continue
        claim = scheduler.claim(device_id)
        if claim is None:
            continue
        attempt = uuid.uuid4().hex
        task_dir = (
            tasks_root
            / f"generation-{claim.work.relay_generation:04d}"
            / _safe_token(claim.work.work_id)
            / attempt
        )
        payload = {
            **dict(payload_by_work[claim.work.work_id]),
            "output_dir": str(task_dir),
        }
        task = LtaWorkerTask(
            work_id=claim.work.work_id,
            attempt_token=attempt,
            kind="propagation_window" if "chain_work_id" in payload else "propagation_chain",
            payload=payload,
        )
        try:
            pool.submit(task, execution_device_id=device_id)
        except BaseException:
            scheduler.fail(claim, retry=True)
            raise
        active[claim.work.work_id] = claim
        active_started[claim.work.work_id] = time.monotonic()
        dispatched.append(
            {
                "work_id": claim.work.work_id,
                "chain_work_id": payload.get("chain_work_id", claim.work.work_id),
                "dependency_work_ids": list(claim.work.dependency_work_ids),
                "attempt_token": attempt,
                "volume_id": claim.work.view.volume_id,
                "physical_view_id": claim.work.view.physical_view_id,
                "runtime_view_id": claim.work.runtime_view_id,
                "relay_generation": claim.work.relay_generation,
                "plan_order": claim.work.plan_order,
                "frame_start": claim.work.frame_start,
                "frame_stop": claim.work.frame_stop,
                "tile_config_id": claim.work.tile_config_id,
                "tile_index": claim.work.tile_index,
                "owner_device_id": claim.owner_device_id,
                "execution_device_id": claim.execution_device_id,
                "tail_assist": claim.tail_assist,
            }
        )
        if trace is not None:
            trace.event("window_dispatch", **dispatched[-1])


def _verified_dogfood_artifacts(
    manifest: Mapping[str, object],
    payload: Mapping[str, object],
) -> dict[int, dict[str, object]]:
    """Admit immutable continuation seeds before unblocking a dependent task."""

    records = tuple(manifest.get("dogfood_seed_artifacts", ()))
    if not records:
        return {}
    incoming = read_seed_artifact(
        str(payload["seed_artifact_path"]),
        expected_sha256=str(payload["seed_artifact_sha256"]),
    )
    allowed_lineages = {seed.lineage for seed in incoming}
    expected_shape = tuple(incoming[0].mask.shape)
    result: dict[int, dict[str, object]] = {}
    for raw in records:
        record = dict(raw)
        frame = int(record["frame_index"])
        if frame in result:
            raise RuntimeError("worker emitted duplicate dogfood frame artifacts")
        seeds = read_seed_artifact(str(record["path"]), expected_sha256=str(record["sha256"]))
        if len(seeds) != int(record["seed_count"]):
            raise RuntimeError("worker dogfood artifact seed count changed")
        if any(
            seed.frame_index != frame or seed.lineage not in allowed_lineages
            or tuple(seed.mask.shape) != expected_shape
            or seed.provenance is not LtaSeedProvenance.TEMPORAL_DOGFOOD
            for seed in seeds
        ):
            raise RuntimeError("worker dogfood artifact does not match its parent window")
        result[frame] = record
    return result


def _drive_workers_to_fixed_point(
    *,
    scheduler: LtaViewAffinityScheduler,
    pool: LtaWorkerPool,
    initial: Sequence[_PlannedChain],
    view_plan: LtaRuntimeViewPlan,
    cache_ref: LtaPhysicalViewCacheRef,
    view_union: object,
    relay_mask_revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision],
    temp_root: Path,
    conf: float,
    empty_frame_limit: int | None,
    worker_task_timeout: float,
    trace: object | None = None,
) -> tuple[
    int,
    Mapping[int, int],
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
]:
    if scheduler.helper_queue_order != "head":
        raise ValueError("bounded production dispatch requires helper_queue_order='head'")
    payload_by_work: dict[str, Mapping[str, object]] = {
        chain.work.work_id: chain.payload for chain in initial
    }
    work_by_id: dict[str, LtaSessionWork] = {}
    children: dict[str, list[str]] = {}
    chain_state: dict[str, dict[str, object]] = {}
    windowed = bool(initial and "chain_work_id" in initial[0].payload)

    def register_tasks(tasks: Sequence[_PlannedChain]) -> None:
        for task in tasks:
            work_by_id[task.work.work_id] = task.work
            payload_by_work[task.work.work_id] = dict(task.payload)
            trace_path = getattr(trace, "path", None)
            if trace_path is not None:
                payload_by_work[task.work.work_id]["worker_trace_root"] = str(Path(trace_path).parent)
            chain_id = str(task.payload.get("chain_work_id", task.work.work_id))
            state = chain_state.setdefault(chain_id, {
                "remaining": 0, "observations": {}, "legacy_relays": [],
                "windowed": "chain_work_id" in task.payload,
            })
            state["remaining"] = int(state["remaining"]) + 1
            for predecessor in task.work.dependency_work_ids:
                children.setdefault(predecessor, []).append(task.work.work_id)

    register_tasks(initial)
    active: dict[str, object] = {}
    active_started: dict[str, float] = {}
    reduced_work_ids: set[str] = set()
    dispatched: list[dict[str, object]] = []
    worker_audit = _new_worker_audit()
    worker_audit["task_count"] = 0
    worker_audit["skipped_window_count"] = 0
    for ready in tuple(getattr(pool, "ready_events", ())):
        _accumulate_worker_ready_audit(worker_audit, ready)
    authoritative_active_coverage: dict[
        tuple[str, int, str], tuple[tuple[int, int], ...]
    ] = {}
    relay_records: dict[int, list[dict[str, object]]] = {0: []}
    generation = 0
    next_plan_order = max(chain.work.plan_order for chain in initial) + 1
    view_key = initial[0].work.view
    scheduler.mark_projection_ready(
        view_key,
        device_id=scheduler.owner_for_view(view_key),
    )
    tasks_root = temp_root / "tasks"
    last_progress = time.monotonic()

    def progress(*, force: bool = False) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < 10.0:
            return
        counts = scheduler.queue_counts()
        reason = "waiting_for_window_results" if counts["active"] else "ready_work" if counts["ready"] else "generation_fan_in"
        if trace is not None:
            trace.event("scheduler_state", generation=generation, reason=reason, **counts)
            trace.flush()
            print(f"LTA window DAG: generation={generation} ready={counts['ready']} blocked={counts['blocked']} active={counts['active']} reason={reason}", flush=True)
        last_progress = now

    def skip_branch(work_id: str) -> None:
        pending = [work_id]
        while pending:
            skipped_id = pending.pop()
            scheduler.skip_pending(skipped_id, _SkippedWindow())
            if trace is not None:
                trace.event("window_skipped", work_id=skipped_id, reason="empty_dogfood_boundary")
            pending.extend(children.get(skipped_id, ()))

    def complete_chain_window(work: LtaSessionWork) -> None:
        payload = payload_by_work[work.work_id]
        chain_id = str(payload.get("chain_work_id", work.work_id))
        state = chain_state[chain_id]
        state["remaining"] = int(state["remaining"]) - 1
        if state["remaining"]:
            return
        from .lta_worker_adapter import _write_relay_artifacts
        phase = trace.phase("chain_relay_emission", chain_work_id=chain_id) if trace is not None else nullcontext()
        with phase:
            emitted = _write_relay_artifacts(
                state["observations"],
                output_dir=temp_root / "chain-relays" / hashlib.sha256(chain_id.encode()).hexdigest()[:20],
                source_tile_index=int(work.tile_index), generation=work.relay_generation,
            )
        relay_records.setdefault(work.relay_generation, []).extend((*state["legacy_relays"], *emitted))
        if state["windowed"]:
            worker_audit["chain_count"] = int(worker_audit["chain_count"]) + 1
        worker_audit["relay_artifact_count"] = int(worker_audit["relay_artifact_count"]) + len(emitted)
        if trace is not None:
            trace.event("chain_settled", chain_work_id=chain_id, relay_count=len(emitted))
        del chain_state[chain_id]

    def plan_relays(records, **kwargs):
        phase = trace.phase("relay_planning", generation=kwargs["generation"], relay_record_count=len(records)) if trace is not None else nullcontext()
        with phase:
            return _plan_relay_generation(records, **kwargs)

    while True:
        progress()
        _dispatch_available(
            scheduler,
            pool,
            payload_by_work,
            active,
            active_started,
            dispatched,
            tasks_root=tasks_root,
            trace=trace,
        )
        if active:
            oldest_work_id = min(
                active_started,
                key=lambda work_id: active_started[work_id],
            )
            remaining = (
                float(worker_task_timeout)
                - (time.monotonic() - active_started[oldest_work_id])
            )
            if remaining <= 0.0:
                raise TimeoutError(
                    f"LTA worker task {oldest_work_id!r} exceeded its "
                    f"{float(worker_task_timeout):.1f}-second lease"
                )
            wait_phase = trace.phase("wait_for_window_result", **scheduler.queue_counts()) if trace is not None else nullcontext({})
            with wait_phase as wait_details:
                try:
                    worker_result = pool.wait_result(timeout=min(remaining, 10.0))
                except TimeoutError:
                    worker_result = None
                    wait_details["poll_timeout"] = True
            if worker_result is None:
                progress(force=True)
                if time.monotonic() - active_started[oldest_work_id] >= float(worker_task_timeout):
                    raise TimeoutError(f"LTA worker task {oldest_work_id!r} exceeded its {float(worker_task_timeout):.1f}-second lease")
                continue
            claim = active.pop(worker_result.work_id, None)
            active_started.pop(worker_result.work_id, None)
            if claim is None:
                raise RuntimeError(
                    f"worker returned unclaimed LTA work {worker_result.work_id}"
                )
            # Dense OR into the private view is associative and commutative.
            # Reduce completed work immediately so a slow earlier chain cannot
            # hold every helper idle or retain a full-view file per later task.
            # Audit/relay acknowledgment remains in deterministic plan order.
            phase = trace.phase("window_result_reduction", work_id=worker_result.work_id) if trace is not None else nullcontext()
            with phase:
                manifest = _load_chain_manifest(worker_result)
                parent_payload = payload_by_work[worker_result.work_id]
                if list(manifest["output_frame_range"]) != [parent_payload["output_frame_start"], parent_payload["output_frame_stop"]]:
                    raise RuntimeError("worker result changed its owned window frame range")
                _relays, union_artifact = _consume_chain_manifest(manifest, view_union=view_union)
                outgoing = _verified_dogfood_artifacts(manifest, parent_payload)
            _unlink_consumed_temp_artifacts((union_artifact,), temp_root=temp_root)
            reduced_work_ids.add(worker_result.work_id)
            empty_children: list[str] = []
            used_seed_paths: set[str] = set()
            for child_id in children.get(worker_result.work_id, ()):
                child_payload = dict(payload_by_work[child_id])
                prompt = int(child_payload["window"]["prompt_frame"])
                artifact = outgoing.get(prompt)
                if artifact is None:
                    empty_children.append(child_id)
                    continue
                child_payload["seed_artifact_path"] = str(artifact["path"])
                child_payload["seed_artifact_sha256"] = str(artifact["sha256"])
                payload_by_work[child_id] = child_payload
                used_seed_paths.add(str(artifact["path"]))
            scheduler.complete(claim, worker_result)
            for child_id in empty_children:
                skip_branch(child_id)
            unused_seeds = [str(record["path"]) for record in outgoing.values() if str(record["path"]) not in used_seed_paths]
            _unlink_consumed_temp_artifacts(unused_seeds, temp_root=temp_root)
            # Overlap ordered receipt work with the next GPU session instead
            # of waiting until all committable manifests have been processed.
            _dispatch_available(
                scheduler, pool, payload_by_work, active, active_started,
                dispatched, tasks_root=tasks_root, trace=trace,
            )

        while True:
            commit = scheduler.claim_committable()
            if commit is None:
                break
            consumed_artifacts: list[Path] = []
            try:
                for _work, result in commit.entries:
                    task_payload = payload_by_work[_work.work_id]
                    if isinstance(result, _SkippedWindow):
                        worker_audit["skipped_window_count"] = int(worker_audit["skipped_window_count"]) + 1
                        worker_audit["planned_window_count"] = int(worker_audit["planned_window_count"]) + 1
                        worker_audit["halted_window_count"] = int(worker_audit["halted_window_count"]) + 1
                        complete_chain_window(_work)
                        continue
                    if _work.work_id not in reduced_work_ids:
                        raise RuntimeError("LTA work reached commit before dense union reduction")
                    manifest = _load_chain_manifest(result)
                    if "chain_work_id" in task_payload:
                        manifest = {**manifest, "task_granularity": "window"}
                    _accumulate_worker_audit(worker_audit, manifest, result)
                    _accumulate_authoritative_active_coverage(
                        authoritative_active_coverage,
                        _work,
                        manifest,
                    )
                    chain_id = str(task_payload.get("chain_work_id", _work.work_id))
                    state = chain_state[chain_id]
                    state["legacy_relays"].extend(dict(item) for item in manifest.get("relays", ()))
                    observation_artifact = manifest.get("relay_observation_artifact")
                    if observation_artifact is not None:
                        from .lta_relay_episodes import read_relay_observations, merge_relay_observations
                        phase = trace.phase("relay_episode_merge", work_id=_work.work_id) if trace is not None else nullcontext()
                        with phase:
                            observations = read_relay_observations(observation_artifact)
                            merge_relay_observations(state["observations"], observations)
                        consumed_artifacts.append(Path(observation_artifact["path"]))
                    input_seed = _verified_artifact_path(
                        str(task_payload["seed_artifact_path"]),
                        expected_sha256=task_payload["seed_artifact_sha256"],
                        description="consumed worker input-seed artifact",
                    )
                    consumed_artifacts.extend(
                        (
                            Path(result.artifact_path).resolve(strict=True),
                            input_seed,
                        )
                    )
                    complete_chain_window(_work)
            except BaseException:
                scheduler.fail_commit(commit)
                raise
            scheduler.complete_commit(commit)
            _unlink_consumed_temp_artifacts(
                consumed_artifacts,
                temp_root=temp_root,
            )
            reduced_work_ids.difference_update(work.work_id for work, _result in commit.entries)

        if not scheduler.generation_settled(view_key, generation):
            if not active:
                pool.check_liveness()
                counts = scheduler.queue_counts()
                if counts["ready"] == 0:
                    if trace is not None:
                        trace.event("scheduler_stalled", generation=generation, **counts)
                    raise RuntimeError(
                        f"LTA relay generation {generation} stalled before settlement: "
                        f"no active or ready windows; blocked={counts['blocked']}, "
                        f"completed_awaiting_commit={counts['completed_awaiting_commit']}"
                    )
            continue

        if generation >= scheduler.max_relay_generation:
            overflow = plan_relays(
                relay_records.get(generation, ()),
                generation=generation + 1,
                view_plan=view_plan,
                cache_ref=cache_ref,
                scheduler=scheduler,
                relay_mask_revisions=relay_mask_revisions,
                temp_root=temp_root,
                conf=conf,
                empty_frame_limit=empty_frame_limit,
                first_plan_order=next_plan_order,
            )
            if overflow:
                raise RuntimeError(
                    "LTA spatial relay did not reach a mask fixed point within "
                    f"the safety bound of {scheduler.max_relay_generation} generations"
                )
            scheduler.seal_view(view_key)
            break
        next_generation = generation + 1
        planned = plan_relays(
            relay_records.get(generation, ()),
            generation=next_generation,
            view_plan=view_plan,
            cache_ref=cache_ref,
            scheduler=scheduler,
            relay_mask_revisions=relay_mask_revisions,
            temp_root=temp_root,
            conf=conf,
            empty_frame_limit=empty_frame_limit,
            first_plan_order=next_plan_order,
        )
        if not planned:
            scheduler.seal_view(view_key)
            break
        if windowed:
            planned = _plan_window_tasks(planned, first_plan_order=next_plan_order)
        scheduler.register_generation(
            view_key,
            next_generation,
            (chain.work for chain in planned),
        )
        scheduler.seal_generation(view_key, next_generation)
        register_tasks(planned)
        relay_records[next_generation] = []
        next_plan_order += len(planned)
        generation = next_generation

    if active:
        raise RuntimeError("LTA fixed point sealed with active worker claims")
    worker_audit["authoritative_active_lineage_tile_count"] = len(
        authoritative_active_coverage
    )
    worker_audit["authoritative_active_range_count"] = sum(
        len(ranges) for ranges in authoritative_active_coverage.values()
    )
    worker_audit["window_graph"] = {
        "planned_work_count": len(scheduler.work),
        "dispatched_work_count": len(dispatched),
        "skipped_work_count": int(worker_audit["skipped_window_count"]),
        "dependency_readiness": "verified_window_completion",
        "relay_fan_in": "original_chain_then_generation",
    }
    return (
        generation,
        dict(pool.pids),
        tuple(
            sorted(
                dispatched,
                key=lambda item: (int(item["plan_order"]), str(item["work_id"])),
            )
        ),
        worker_audit,
    )


def _write_small_json_artifact(
    path: Path,
    *,
    name: str,
    payload: Mapping[str, object],
) -> LtaArtifactReceipt:
    destination = write_json_atomically(path, dict(payload))
    return LtaArtifactReceipt(
        name=name,
        path=destination,
        sha256=_sha256_file(destination),
    )


def _write_tree_inventory(
    root: Path,
    *,
    path: Path,
    name: str,
) -> LtaArtifactReceipt:
    tree = Path(root).resolve(strict=True)
    files = [candidate for candidate in sorted(tree.rglob("*")) if candidate.is_file()]
    if not files:
        raise RuntimeError(f"LTA {name} publication produced no files: {tree}")
    return _write_small_json_artifact(
        path,
        name=name,
        payload={
            "root": str(tree),
            "files": [
                {
                    "path": str(candidate.relative_to(tree)).replace("\\", "/"),
                    "size_bytes": int(candidate.stat().st_size),
                    "sha256": _sha256_file(candidate),
                }
                for candidate in files
            ],
        },
    )


def _publish_selected_transverse_outputs(
    *,
    plan: LtaRunPlan,
    source: object,
    volume_plan: object,
    view_plan: LtaRuntimeViewPlan,
    cache_ref: LtaPhysicalViewCacheRef,
    terminal_union: object,
    workers: int,
) -> list[LtaArtifactReceipt]:
    receipts: list[LtaArtifactReceipt] = []
    selected = set(plan.save_tokens)
    if not selected.intersection({"images", "labels", "overlay"}):
        return receipts
    cache = cache_ref.open(mode="r")
    try:
        if "images" in selected:
            from .outputs import write_view_images

            image_root = write_view_images(
                cache,
                view_plan.runtime_view,
                plan.output_root,
                str(volume_plan.stem),
                workers=max(1, int(workers)),
                show_progress=True,
            )
            receipts.append(
                _write_tree_inventory(
                    image_root,
                    path=plan.output_root / "images_manifest.json",
                    name="images",
                )
            )
        if "labels" in selected:
            from .outputs import write_yolo_labels_from_pattern

            label_root = write_yolo_labels_from_pattern(
                terminal_union,
                plan.output_root / "labels" / f"{volume_plan.stem}_%04d.txt",
                workers=max(1, int(workers)),
                show_progress=True,
            )
            receipts.append(
                _write_tree_inventory(
                    label_root,
                    path=plan.output_root / "labels_manifest.json",
                    name="labels",
                )
            )
        if "overlay" in selected:
            from .outputs import write_overlay_video

            overlay_path = plan.output_root / f"{volume_plan.stem}_Overlay.mkv"
            write_overlay_video(
                cache,
                terminal_union,
                overlay_path,
                fps=float(source.fps or 1.0),
                show_progress=True,
                scratch_dir=plan.temp_root / "overlay",
                publication_root=plan.output_root,
            )
            receipts.append(
                LtaArtifactReceipt(
                    name="overlay",
                    path=overlay_path,
                    sha256=_sha256_file(overlay_path),
                )
            )
    finally:
        mmap_obj = getattr(cache, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
    return receipts


@bounded_lta_host_threads
def execute_lta_plan(
    plan: LtaRunPlan,
    *,
    empty_frame_limit: int | None = 30,
    worker_startup_timeout: float = 900.0,
    worker_task_timeout: float = PRODUCTION_LTA_TASK_TIMEOUT_SECONDS,
    workers: int | None = None,
) -> LtaProductionResult:
    """Execute the qualified mask-injected LTA production path."""

    import numpy as np

    if not isinstance(plan, LtaRunPlan):
        raise TypeError("plan must be an LtaRunPlan")
    source, volume_plan, view_plan = _require_supported_runtime(plan)
    if not math.isfinite(float(worker_startup_timeout)) or float(worker_startup_timeout) <= 0.0:
        raise ValueError("worker_startup_timeout must be finite and positive")
    if not math.isfinite(float(worker_task_timeout)) or float(worker_task_timeout) <= 0.0:
        raise ValueError("worker_task_timeout must be finite and positive")
    if empty_frame_limit is not None and (
        isinstance(empty_frame_limit, bool) or int(empty_frame_limit) < 1
    ):
        raise ValueError("empty_frame_limit must be a positive integer or None")
    storage_preflight = _preflight_lta_storage(plan, view_plan)
    _prepare_output_roots(plan)
    from . import __version__
    from .lta_telemetry import LtaExecutionTrace, lta_source_fingerprint
    trace_dir = plan.output_root / "lta_diagnostics"
    trace = LtaExecutionTrace(trace_dir, "coordinator")
    try:
        fingerprint = lta_source_fingerprint()
        write_json_atomically(plan.output_root / "lta_execution_identity.json", {
            "schema": "lta.execution-identity/1",
            "contract": "lta.window_dag/1",
            "pipeline_version": __version__,
            "source_fingerprint": fingerprint,
            "run_id": plan.run_id,
            "requested_devices": list(plan.device_ids),
            "scratch_root": str(plan.temp_root),
            "run_completion_marker": "manifest.json",
        })
    except BaseException:
        trace.close()
        raise
    trace.event("run_contract", version=__version__, contract="lta.window_dag/1", source_fingerprint=fingerprint, devices=list(plan.device_ids), scratch_root=str(plan.temp_root))
    trace.flush()
    print(f"LTA v{__version__}: contract=lta.window_dag/1 max_window_frames=30 devices={list(plan.device_ids)} source_sha256={fingerprint['sha256']} diagnostics={trace_dir}", flush=True)
    from .lta_cpu import resolve_worker_cpu_budget
    parent_cpu_budget = resolve_worker_cpu_budget(len(plan.device_ids))
    effective_cpu_count = int(parent_cpu_budget["effective_cpu_count"])
    worker_count = max(1, min(effective_cpu_count, int(effective_cpu_count if workers is None else workers)))
    trace.event("parent_cpu_budget", budget=parent_cpu_budget, finalization_workers=worker_count, native_thread_counts=host_native_thread_counts())
    source_volume = None
    hard_positive = None
    view_union = None
    native_prediction = None
    finalization = None
    completed = False
    scratch_cleaned = False
    cache_ref: LtaPhysicalViewCacheRef
    worker_pids: Mapping[int, int] = {}
    execution_schedule: tuple[Mapping[str, object], ...] = ()
    worker_audit: Mapping[str, object] = {}
    authoritative_seed_audit: Mapping[str, object] = {}
    known_background_audit: Mapping[str, object] = {}
    relay_generation = 0
    try:
        aligned_annotations = _aligned_annotations(plan, source)
        known_background_frames = _aligned_known_background_frames(
            plan,
            source,
            aligned_annotations,
        )
        source_cache_path = plan.temp_root / "source.gray8.dat"
        with trace.phase("decode_source_cache", shape_tyx=list(volume_plan.source_shape_tyx)):
            source_volume = _materialize_source_volume(source, path=source_cache_path)
        with trace.phase("prepare_authoritative_volume", annotation_count=len(aligned_annotations)):
            hard_positive = _hard_positive_volume(
                source, aligned_annotations, path=plan.temp_root / "hard_positive.uint8.raw",
            )
        source_mmap = getattr(source_volume, "_mmap", None)
        if source_mmap is not None:
            source_mmap.close()
        source_volume = None
        cache_ref = reference_existing_physical_view_cache(
            source_cache_path,
            shape=volume_plan.source_shape_tyx,
            physical_view_id=view_plan.physical_view_id,
            source_identity=_source_identity(source),
        )

        with trace.phase("plan_seed_window_graph") as planning:
            initial, seed_inventory = _plan_initial_chains(
                source, aligned_annotations, view_plan, cache_ref,
                temp_root=plan.temp_root, conf=plan.conf, empty_frame_limit=empty_frame_limit,
            )
            planning["chain_count"] = len(initial)
            initial = _plan_window_tasks(initial)
            initial = tuple(_PlannedChain(task.work, {**task.payload, "worker_trace_root": str(trace_dir)}) for task in initial)
            planning["window_count"] = len(initial)
        authoritative_seed_audit = _authoritative_seed_audit(seed_inventory)
        scheduler = LtaViewAffinityScheduler(
            (chain.work for chain in initial),
            plan.device_ids,
            helper_queue_order="head",
            max_relay_generation=_relay_generation_bound(view_plan),
        )
        relay_mask_revisions: dict[LtaSpatialRelayKey, _RelayMaskRevision] = {}
        _prime_authoritative_relay_destinations(
            scheduler,
            initial[0].work.view,
            seed_inventory,
            relay_mask_revisions,
        )
        view_union = _allocate_view_union(
            view_plan,
            plan.temp_root / "views" / "transverse_union.uint8.raw",
        )
        with trace.phase("worker_pool_startup", devices=list(plan.device_ids)):
            pool = LtaWorkerPool(
                plan.device_ids,
                LtaWorkerInit(
                    adapter_module="XTA.lta_worker_adapter",
                    adapter_factory="build_worker_predictor",
                    adapter_execute="execute_worker_task",
                    adapter_shutdown="close_worker_predictor",
                    adapter_config={
                        "model_path": str(plan.bundle.root), "conf": float(plan.conf),
                        "profile": PRODUCTION_LTA_PROFILE, "trace_dir": str(trace_dir),
                    },
                ),
                startup_timeout=float(worker_startup_timeout),
            )
        worker_error: BaseException | None = None
        try:
            relay_generation, worker_pids, execution_schedule, worker_audit = (
                _drive_workers_to_fixed_point(
                    scheduler=scheduler,
                    pool=pool,
                    initial=initial,
                    view_plan=view_plan,
                    cache_ref=cache_ref,
                    view_union=view_union,
                    relay_mask_revisions=relay_mask_revisions,
                    temp_root=plan.temp_root,
                    conf=plan.conf,
                    empty_frame_limit=empty_frame_limit,
                    worker_task_timeout=float(worker_task_timeout),
                    trace=trace,
                )
            )
            worker_audit = {
                **worker_audit,
                "execution_contract": "lta.window_dag/1",
                "source_fingerprint": fingerprint,
                "diagnostics_root": str(trace_dir),
                "diagnostic_write_errors_before_postprocessing": trace.write_errors,
                "parent_cpu_budget": parent_cpu_budget,
                "parent_finalization_workers": worker_count,
                "parent_native_threads_during_tracking": host_native_thread_counts(),
            }
        except BaseException as exc:
            worker_error = exc
            raise
        finally:
            try:
                forced_devices = pool.shutdown(timeout=120.0, force=True)
            except BaseException as cleanup_error:
                if worker_error is None:
                    raise
                add_note = getattr(worker_error, "add_note", None)
                if callable(add_note):
                    add_note(
                        "LTA worker shutdown also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                forced_devices = ()
            if forced_devices:
                cleanup_error = RuntimeError(
                    "LTA workers required forced termination during shutdown: "
                    f"{list(forced_devices)}"
                )
                if worker_error is None:
                    raise cleanup_error
                add_note = getattr(worker_error, "add_note", None)
                if callable(add_note):
                    add_note(str(cleanup_error))

        view_fill = fill_completed_view_holes_2d_inplace(
            view_union,
            runtime_view_id=view_plan.runtime_view_id,
            workers=worker_count,
        )
        backprojection_claim = scheduler.claim_backprojection(
            scheduler.owner_for_view(initial[0].work.view)
        )
        if backprojection_claim is None:
            raise RuntimeError("settled LTA view was not admitted for backprojection")
        try:
            from .assembly import project_view_volume_to_orthogonal_volume

            native_prediction = project_view_volume_to_orthogonal_volume(
                view_union,
                view_plan.runtime_view,
                plan.temp_root / "native_prediction.uint8.raw",
                "LTA settled Transverse backprojection",
                workers=worker_count,
                prefer_memory=False,
                reserve_bytes=0,
                out_shape_tyx=volume_plan.source_shape_tyx,
                allow_transverse_passthrough=True,
            )
        except BaseException:
            scheduler.fail_backprojection(backprojection_claim)
            raise
        scheduler.complete_backprojection(backprojection_claim)

        prediction_layer = LtaLayerRecord(
            layer_id="transverse_mask_propagation",
            recomposition_op=LtaRecompositionOp.UNION,
            source_role="sam_mask_propagation",
            volume=native_prediction,
            physical_view_id=view_plan.physical_view_id,
            runtime_view_id=view_plan.runtime_view_id,
            tta_angle_deg=view_plan.tta_angle_deg,
            metadata={
                "conditioning": "authoritative_mask_injection",
                "cross_tile_relay": True,
                "relay_generations": relay_generation,
                "completed_view_hole_fill": view_fill.manifest_record(),
            },
        )
        hard_positive_layer = LtaLayerRecord(
            layer_id="authoritative_hard_positives",
            recomposition_op=LtaRecompositionOp.UNION,
            source_role="authoritative_annotation",
            volume=hard_positive,
            physical_view_id="transverse",
            runtime_view_id="source",
            tta_angle_deg=0.0,
            metadata={
                "immutable_source_annotations": True,
                "hole_fill_applied": False,
                "restored_after_terminal_filters": True,
            },
        )
        contributor_layers = (prediction_layer, hard_positive_layer)
        from .topology import configure_gpu_slice_labeling_devices

        configure_gpu_slice_labeling_devices(
            tuple(f"cuda:{device_id}" for device_id in plan.device_ids)
        )
        # Validate identities before the first durable checkpoint. Each finished
        # stage survives a later filter failure; only the final manifest marks
        # the entire run complete.
        from . import __version__
        from .lta_inputs import revalidate_lta_input_identities

        revalidate_lta_input_identities(plan.discovery)
        revalidate_local_sam_bundle(plan.bundle)
        checkpoints: list[LtaCheckpointArtifact] = []

        def write_checkpoint_index(*, sequence_complete: bool) -> LtaArtifactReceipt:
            return _write_small_json_artifact(
                plan.output_root / f"{volume_plan.stem}_Postprocessing_checkpoints.json",
                name="postprocessing_checkpoints",
                payload={
                    "schema": "lta.postprocessing-checkpoints/1",
                    "checkpoint_sequence_complete": bool(sequence_complete),
                    "run_completion_marker": "manifest.json",
                    "run_id": plan.run_id,
                    "pipeline_version": __version__,
                    "command": list(plan.command),
                    "source_identity_sha256": _source_identity(source),
                    "model_identity_sha256": plan.bundle.checkpoint_identity_sha256,
                    "annotations": [
                        {
                            "frame_index": int(annotation.frame_position),
                            "label_sha256": annotation.label_sha256,
                        }
                        for annotation in aligned_annotations
                    ],
                    "requested_postprocessing": dict(plan.postprocessing),
                    "checkpoints": [item.manifest_record() for item in checkpoints],
                },
            )

        def save_postprocessing_checkpoint(
            stage: str,
            volume: object,
            metadata: Mapping[str, object],
        ) -> None:
            checkpoints.append(write_lta_postprocessing_checkpoint(
                plan.output_root,
                stem=str(volume_plan.stem),
                stage=stage,
                volume=volume,
                metadata=metadata,
                model_name="sam3.1",
            ))
            write_checkpoint_index(sequence_complete=False)

        finalization = finalize_lta_native_union(
            contributor_layers,
            workspace_path=plan.temp_root / "terminal_union.uint8.raw",
            temp_dir=plan.temp_root / "postprocessing",
            protected_foreground_layers=(hard_positive_layer,),
            postprocessing=plan.postprocessing,
            prefer_memory=False,
            reserve_bytes=0,
            workers=worker_count,
            checkpoint_callback=save_postprocessing_checkpoint,
        )
        checkpoint_index = write_checkpoint_index(sequence_complete=True)
        known_background_audit = _known_background_audit(
            finalization.terminal_union,
            known_background_frames,
        )

        # Revalidate once more after filtering, before final output/manifest
        # publication. Earlier independently completed checkpoints are retained.
        revalidate_lta_input_identities(plan.discovery)
        revalidate_local_sam_bundle(plan.bundle)

        final_nrrd = write_global_final_output_nrrd(
            plan.output_root,
            stem=str(volume_plan.stem),
            terminal_union=finalization.terminal_union,
            model_name="sam3.1",
        )
        artifact_receipts: list[LtaArtifactReceipt] = [
            *(checkpoint.receipt for checkpoint in checkpoints),
            checkpoint_index,
            final_nrrd.receipt,
        ]
        artifact_receipts.extend(
            _publish_selected_transverse_outputs(
                plan=plan,
                source=source,
                volume_plan=volume_plan,
                view_plan=view_plan,
                cache_ref=cache_ref,
                terminal_union=finalization.terminal_union,
                workers=worker_count,
            )
        )
        if "voxel_volume" in plan.save_tokens:
            artifact_receipts.append(
                _write_small_json_artifact(
                    plan.output_root / f"{volume_plan.stem}_voxel_volume.json",
                    name="voxel_volume",
                    payload={
                        "foreground_voxels": finalization.terminal_union_foreground_voxels,
                        "shape_tyx": list(volume_plan.source_shape_tyx),
                    },
                )
            )
        if "summary" in plan.save_tokens:
            artifact_receipts.append(
                _write_small_json_artifact(
                    plan.output_root / f"{volume_plan.stem}_summary.json",
                    name="summary",
                    payload={
                        "mode": "lta",
                        "conditioning": "authoritative_mask_injection",
                        "devices": list(plan.device_ids),
                        "worker_pids": {str(key): value for key, value in worker_pids.items()},
                        "worker_task_timeout_seconds": float(worker_task_timeout),
                        "relay_generations": relay_generation,
                        "authoritative_tile_seeding": dict(authoritative_seed_audit),
                        "storage_preflight": dict(storage_preflight),
                        "known_background_audit": dict(known_background_audit),
                        "worker_audit": dict(worker_audit),
                        "postprocessing": dict(finalization.postprocessing),
                        "postprocessing_checkpoints": [
                            checkpoint.manifest_record() for checkpoint in checkpoints
                        ],
                        "foreground_voxels": finalization.terminal_union_foreground_voxels,
                        "final_nrrd": final_nrrd.manifest_record(),
                    },
                )
            )

        publication = LtaPublicationReceipt(
            artifacts=tuple(artifact_receipts),
            terminal_union_shape_tyx=volume_plan.source_shape_tyx,
            terminal_union_foreground_voxels=(
                finalization.terminal_union_foreground_voxels
            ),
            source_revalidated=True,
            model_revalidated=True,
            layers_settled=True,
        )

        # A complete manifest is the final commit marker.  Close every scratch
        # owner and prove removal before publishing it, so a cleanup failure
        # cannot leave a successful-looking run behind.
        from .runtime import close_memmap_array

        if not finalization.closed:
            finalization.close()
        for volume in (native_prediction, view_union, hard_positive, source_volume):
            if volume is not None:
                close_memmap_array(volume)
        native_prediction = None
        view_union = None
        hard_positive = None
        source_volume = None
        shutil.rmtree(plan.temp_root, ignore_errors=False)
        scratch_cleaned = True

        manifest_path = write_complete_lta_manifest(
            plan.output_root / "manifest.json",
            version=__version__,
            command=plan.command,
            layers=(*contributor_layers, *checkpoints, final_nrrd.layer),
            publication_receipt=publication,
            payload={
                "run_plan": _execution_run_plan_manifest_record(plan),
                "execution": {
                    "conditioning": "authoritative_mask_injection",
                    "devices": list(plan.device_ids),
                    "worker_pids": {str(key): value for key, value in worker_pids.items()},
                    "worker_task_timeout_seconds": float(worker_task_timeout),
                    "relay_generations": relay_generation,
                    "cross_tile_tracking": "eight_neighbor_fixed_point",
                    "authoritative_tile_seeding": dict(authoritative_seed_audit),
                    "temporal_propagation": {
                        "policy": "full_view_per_anchor_recall_union",
                        "per_anchor_frame_range": [0, view_plan.frame_count],
                        "cross_anchor_identity_reconciliation": False,
                        "output_combination": "union",
                        "missing_at_other_anchors": "does_not_terminate_lineage",
                        "continuation": "nonempty_directional_dogfood_boundaries",
                    },
                    "storage_preflight": dict(storage_preflight),
                    "known_background_audit": dict(known_background_audit),
                    "worker_audit": dict(worker_audit),
                    "view_cache": _view_cache_manifest_record(cache_ref),
                    "completed_view_hole_fill": view_fill.manifest_record(),
                    "postprocessing": dict(finalization.postprocessing),
                    "device_schedule": {
                        "status": "settled",
                        "policy": "physical_view_affinity_with_bounded_head_assist",
                        "helper_queue_order": scheduler.helper_queue_order,
                        "maximum_uncommitted_worker_unions": len(plan.device_ids),
                        "dense_union_reduction_order": "completion",
                        "logical_commit_order": "plan",
                        "coordinator_selected": True,
                        "view_owner_device_id": scheduler.owner_for_view(
                            initial[0].work.view
                        ),
                        "backprojection_device_id": backprojection_claim.owner_device_id,
                        "work_count": len(execution_schedule),
                        "work": [dict(item) for item in execution_schedule],
                    },
                },
                "final_nrrd": final_nrrd.manifest_record(),
                "postprocessing_checkpoints": [
                    checkpoint.manifest_record() for checkpoint in checkpoints
                ],
            },
        )
        result = LtaProductionResult(
            plan=plan,
            manifest_path=manifest_path,
            final_nrrd_path=final_nrrd.receipt.path,
            terminal_union_foreground_voxels=(
                finalization.terminal_union_foreground_voxels
            ),
            worker_pids=dict(worker_pids),
            relay_generations=relay_generation,
        )
        completed = True
        return result
    finally:
        from .runtime import close_memmap_array
        from .topology import configure_gpu_slice_labeling_devices

        trace.event("coordinator_complete" if completed else "coordinator_failed")
        trace.close()
        configure_gpu_slice_labeling_devices(())

        if finalization is not None and not finalization.closed:
            finalization.close()
        for volume in (native_prediction, view_union, hard_positive, source_volume):
            if volume is not None:
                try:
                    close_memmap_array(volume)
                except Exception:
                    if completed:
                        raise
        if completed and not scratch_cleaned:
            shutil.rmtree(plan.temp_root, ignore_errors=False)


__all__ = (
    "LtaProductionResult",
    "PRODUCTION_LTA_PROFILE",
    "PRODUCTION_LTA_RELAY_MIN_PIXELS",
    "PRODUCTION_LTA_RELAY_MIN_PROBABILITY",
    "PRODUCTION_LTA_TASK_TIMEOUT_SECONDS",
    "PRODUCTION_LTA_TILE_SIZE",
    "execute_lta_plan",
)
