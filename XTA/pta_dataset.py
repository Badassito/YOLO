"""Deterministic PTA dataset identity, augmentation, filtering, and split policy.

This module is deliberately independent of :mod:`XTA.pta`.  It owns the
mutable candidate records and pure dataset-policy decisions, while source
geometry, rendering, filesystem publication, and worker process state remain
with the PTA orchestration layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import string
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .pta_augmentation import (
    AugmentationDefinition,
    LoadedAugmentation,
    LoadedGpuAugmentation,
    OfflineAugmentation,
)


AUGMENTATION_TAG_LENGTH = 16
AUGMENTATION_TAG_ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase


class WarningSink(Protocol):
    """Minimum warning interface consumed by dataset policy."""

    def add(self, key: str, msg: str = "") -> None: ...


@dataclass
class OutputCandidate:
    order: int
    volume_name: str
    parent_view_tag: str
    output_tag: str
    item_key: str
    frame_idx: int
    is_tile: bool
    label_enabled: bool
    is_transverse: bool = False
    physical_view_id: str = ""
    presentation_variant_id: str = ""
    geometry_item_id: str = ""
    channel_format: str = "gray"
    channel_kind: str = "gray"
    channel_reverse: bool = False
    channel_offsets: Tuple[int, ...] = (0,)
    source_order: int = -1
    augmentation_index: int = 0
    augmentation_tag: Optional[str] = None
    augmentation_seed: Optional[int] = None
    foreground: bool = True
    keep: bool = True
    split_subset: Optional[str] = None
    tile_size: int = 0
    tile_stride: int = 0
    tile_x: int = 0
    tile_y: int = 0


@dataclass
class BackgroundFilterStats:
    active: bool = False
    classification_performed: bool = False
    skipped_reason: str = ""
    foreground_before: int = 0
    background_before: int = 0
    background_max: Optional[int] = None
    background_retained: int = 0
    dropped: int = 0
    original_background_dropped: int = 0
    augmented_background_dropped: int = 0
    subset_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class SplitStats:
    active: bool = False
    train_split: Optional[float] = None
    split_method: Optional[str] = None
    atomic_units_total: int = 0
    atomic_units_train: int = 0
    atomic_units_val: int = 0
    frames_total: int = 0
    frames_train: int = 0
    frames_val: int = 0
    achieved_train_fraction: float = 0.0
    warning: str = ""


@dataclass
class AugmentationStats:
    configured: bool = False
    active: bool = False
    path: Optional[Path] = None
    content_sha256: str = ""
    export_name: str = ""
    albumentations_version: str = ""
    runtime_backend: str = "none"
    execution_mode: str = "none"
    deferred_policy_path: Optional[Path] = None
    requested_ratio: float = 1.0
    applies_to: str = "none"
    foreground_classification_performed: bool = False
    eligible_originals: int = 0
    planned_augmented_copies: int = 0
    planned_augmented_foreground: int = 0
    planned_augmented_background: int = 0
    retained_augmented_copies: int = 0
    dropped_augmented_background: int = 0
    achieved_ratio: float = 1.0
    final_train_files: int = 0
    final_val_files: int = 0
    final_unsplit_files: int = 0


def output_source_identity_text(
    volume_name: str,
    parent_view_tag: str,
    item_key: str,
    frame_idx: int,
) -> str:
    return "\0".join((str(volume_name), str(parent_view_tag), str(item_key), str(int(frame_idx))))


def candidate_source_identity(cand: OutputCandidate) -> str:
    if cand.physical_view_id and cand.presentation_variant_id and cand.geometry_item_id:
        return "\0".join((
            str(cand.volume_name),
            str(cand.physical_view_id),
            str(cand.presentation_variant_id),
            str(cand.geometry_item_id),
            str(int(cand.frame_idx)),
        ))
    return output_source_identity_text(
        cand.volume_name,
        cand.parent_view_tag,
        cand.item_key,
        int(cand.frame_idx),
    )


def stable_digest_rank(domain: str, parts: Sequence[object]) -> bytes:
    payload = json.dumps(
        [str(domain), *[str(part) for part in parts]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def candidate_background_rank(cand: OutputCandidate) -> Tuple[bytes, int, int]:
    return (
        stable_digest_rank(
            "PTA-v4.0.2-background-retention",
            (candidate_source_identity(cand), int(cand.augmentation_index)),
        ),
        int(cand.source_order),
        int(cand.order),
    )


def background_limit_from_foreground(foreground_count: int, background_percent: float) -> int:
    p = float(background_percent)
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return sys.maxsize
    return int(math.floor((p / (1.0 - p)) * float(max(0, int(foreground_count)))))


def augmentation_digest(
    augmentation: OfflineAugmentation,
    *,
    domain: str,
    source_identity: str,
    copy_index: int,
    nonce: int = 0,
) -> bytes:
    payload = "\0".join((
        "PTA-v4",
        str(domain),
        augmentation.content_sha256,
        str(source_identity),
        str(int(copy_index)),
        str(int(nonce)),
    )).encode("utf-8")
    return hashlib.sha256(payload).digest()


def augmentation_seed_for_identity(
    augmentation: OfflineAugmentation,
    source_identity: str,
    copy_index: int,
) -> int:
    digest = augmentation_digest(
        augmentation,
        domain="seed",
        source_identity=source_identity,
        copy_index=int(copy_index),
    )
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def base62_tag_from_digest(digest: bytes, length: int = AUGMENTATION_TAG_LENGTH) -> str:
    value = int.from_bytes(digest, byteorder="big", signed=False)
    chars: List[str] = []
    radix = len(AUGMENTATION_TAG_ALPHABET)
    for _ in range(int(length)):
        value, remainder = divmod(value, radix)
        chars.append(AUGMENTATION_TAG_ALPHABET[int(remainder)])
    return "".join(reversed(chars))


def deterministic_augmentation_tag(
    augmentation: OfflineAugmentation,
    source_identity: str,
    copy_index: int,
    *,
    used_tags: set[str],
) -> str:
    nonce = 0
    while True:
        digest = augmentation_digest(
            augmentation,
            domain="tag",
            source_identity=source_identity,
            copy_index=int(copy_index),
            nonce=int(nonce),
        )
        tag = base62_tag_from_digest(digest)
        if tag not in used_tags:
            used_tags.add(tag)
            return tag
        nonce += 1


def split_round_half_toward_train(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def plan_augmented_versions(
    base_candidates: Sequence[OutputCandidate],
    *,
    augmentation: Optional[OfflineAugmentation],
    augmentation_definition: Optional[AugmentationDefinition] = None,
    execution_mode: str = "offline",
    augmentation_ratio: float,
    split_active: bool,
    augmented_foregrounds: Mapping[Tuple[str, int], bool],
    require_augmented_foregrounds: bool,
    used_tags: Optional[set[str]] = None,
) -> Tuple[List[OutputCandidate], AugmentationStats]:
    mode = str(execution_mode).strip().lower()
    if mode not in {"deferred", "offline"}:
        raise ValueError(f"Unknown augmentation execution mode: {execution_mode!r}")
    if augmentation_definition is None and augmentation is not None:
        augmentation_definition = AugmentationDefinition(
            augmentation.path,
            augmentation.content_sha256,
            augmentation.export_name,
        )
    stats = AugmentationStats(
        configured=augmentation_definition is not None,
        active=False,
        path=augmentation_definition.path if augmentation_definition is not None else None,
        content_sha256=augmentation_definition.content_sha256 if augmentation_definition is not None else "",
        export_name=augmentation_definition.export_name if augmentation_definition is not None else "",
        albumentations_version=augmentation.albumentations_version if augmentation is not None else "",
        runtime_backend=(
            augmentation.runtime_name
            if isinstance(augmentation, LoadedGpuAugmentation)
            else ("albumentations-cpu" if isinstance(augmentation, LoadedAugmentation) else "none")
        ),
        execution_mode=mode if augmentation_definition is not None else "none",
        requested_ratio=float(augmentation_ratio),
        applies_to=(
            ("training_loader_train" if split_active else "training_loader_all")
            if mode == "deferred"
            else ("train" if split_active else "all")
        ),
        foreground_classification_performed=bool(require_augmented_foregrounds),
    )
    eligible = [
        cand for cand in base_candidates
        if cand.keep and (not split_active or cand.split_subset == "train")
    ]
    stats.eligible_originals = len(eligible)
    if mode == "deferred":
        versions = list(base_candidates)
        for physical_order, cand in enumerate(versions):
            cand.order = int(physical_order)
        return versions, stats
    if augmentation_definition is not None and augmentation is None:
        raise RuntimeError("Offline augmentation execution requires a loaded CPU or GPU policy definition")
    if augmentation is None or float(augmentation_ratio) <= 1.0 or not eligible:
        versions = list(base_candidates)
        for physical_order, cand in enumerate(versions):
            cand.order = int(physical_order)
        return versions, stats

    n_eligible = len(eligible)
    target_total_versions = split_round_half_toward_train(float(augmentation_ratio) * float(n_eligible))
    target_total_versions = max(n_eligible, target_total_versions)
    target_augmented = int(target_total_versions - n_eligible)
    copies_per_source, fractional_remainder = divmod(target_augmented, n_eligible)

    fractional_copy_index = int(copies_per_source) + 1
    ranked = sorted(
        eligible,
        key=lambda cand: (
            0 if bool(augmented_foregrounds.get(
                (candidate_source_identity(cand), int(fractional_copy_index)),
                cand.foreground,
            )) else 1,
            augmentation_digest(
                augmentation,
                domain="fractional-ratio-rank",
                source_identity=candidate_source_identity(cand),
                copy_index=0,
            ),
            candidate_source_identity(cand),
        ),
    )
    extra_source_orders = {int(cand.source_order) for cand in ranked[:int(fractional_remainder)]}
    used_tags = used_tags if used_tags is not None else set()
    versions: List[OutputCandidate] = []
    eligible_source_orders = {int(cand.source_order) for cand in eligible}

    for base in base_candidates:
        base.order = len(versions)
        versions.append(base)
        if int(base.source_order) not in eligible_source_orders:
            continue
        copy_count = int(copies_per_source) + (1 if int(base.source_order) in extra_source_orders else 0)
        identity = candidate_source_identity(base)
        for copy_index in range(1, copy_count + 1):
            tag = deterministic_augmentation_tag(
                augmentation,
                identity,
                int(copy_index),
                used_tags=used_tags,
            )
            seed = augmentation_seed_for_identity(augmentation, identity, int(copy_index))
            foreground_key = (identity, int(copy_index))
            if require_augmented_foregrounds and foreground_key not in augmented_foregrounds:
                raise RuntimeError(
                    f"Missing planned augmented foreground classification for {identity!r}, copy {copy_index}"
                )
            copy_foreground = bool(augmented_foregrounds.get(foreground_key, base.foreground))
            augmented = replace(
                base,
                order=len(versions),
                augmentation_index=int(copy_index),
                augmentation_tag=tag,
                augmentation_seed=int(seed),
                foreground=copy_foreground,
                keep=True,
            )
            versions.append(augmented)

    augmented_versions = [cand for cand in versions if int(cand.augmentation_index) > 0]
    stats.active = bool(augmented_versions)
    stats.planned_augmented_copies = len(augmented_versions)
    stats.planned_augmented_foreground = sum(1 for cand in augmented_versions if cand.foreground)
    stats.planned_augmented_background = sum(1 for cand in augmented_versions if not cand.foreground)
    return versions, stats


def finalize_background_filter_with_augmentations(
    candidates: List[OutputCandidate],
    *,
    base_stats: BackgroundFilterStats,
    background_percent: float,
    labels_available: bool,
    warnings: WarningSink,
) -> BackgroundFilterStats:
    """Apply the final source-balanced background cap after augmentation."""
    del base_stats, warnings
    p = float(background_percent)
    stats = BackgroundFilterStats(
        active=p < 1.0,
        classification_performed=bool(p < 1.0 and labels_available),
    )
    stats.foreground_before = sum(1 for cand in candidates if cand.foreground)
    stats.background_before = sum(1 for cand in candidates if not cand.foreground)

    if p >= 1.0:
        stats.skipped_reason = "--background_percent is 1.0"
        stats.background_retained = stats.background_before
        return stats
    if not labels_available:
        stats.skipped_reason = "label operations are disabled for unlabeled volumes"
        return stats

    subset_names = sorted(
        {cand.split_subset or "all" for cand in candidates},
        key=lambda value: {"train": 0, "val": 1, "all": 2}.get(value, 3),
    )
    total_b_max = 0
    for subset_name in subset_names:
        subset = [cand for cand in candidates if (cand.split_subset or "all") == subset_name]
        original_foreground = [
            cand for cand in subset
            if int(cand.augmentation_index) == 0 and cand.foreground
        ]
        augmented_foreground = [
            cand for cand in subset
            if int(cand.augmentation_index) > 0 and cand.foreground
        ]
        foreground = [*original_foreground, *augmented_foreground]
        for cand in foreground:
            cand.keep = True
        original_background = sorted(
            [cand for cand in subset if int(cand.augmentation_index) == 0 and not cand.foreground],
            key=candidate_background_rank,
        )
        augmented_background = [
            cand for cand in subset
            if int(cand.augmentation_index) > 0 and not cand.foreground
        ]

        b_max = background_limit_from_foreground(len(foreground), p)
        total_b_max += int(b_max)
        original_to_keep = min(int(b_max), len(original_background))
        for position, cand in enumerate(original_background):
            cand.keep = int(position) < original_to_keep

        grouped_by_source: Dict[str, List[OutputCandidate]] = defaultdict(list)
        for cand in augmented_background:
            grouped_by_source[candidate_source_identity(cand)].append(cand)
        breadth_ranked: List[Tuple[Tuple[int, bytes, int, int], OutputCandidate]] = []
        for group in grouped_by_source.values():
            group.sort(key=candidate_background_rank)
            for position, cand in enumerate(group):
                digest, src_order, order = candidate_background_rank(cand)
                breadth_ranked.append(((int(position), digest, int(src_order), int(order)), cand))
        breadth_ranked.sort(key=lambda pair: pair[0])
        remaining = max(0, int(b_max) - original_to_keep)
        augmented_to_keep = min(remaining, len(breadth_ranked))
        for position, (_, cand) in enumerate(breadth_ranked):
            cand.keep = int(position) < augmented_to_keep

        original_dropped = len(original_background) - original_to_keep
        augmented_dropped = len(breadth_ranked) - augmented_to_keep
        stats.original_background_dropped += int(original_dropped)
        stats.augmented_background_dropped += int(augmented_dropped)
        stats.subset_stats[subset_name] = {
            "foreground": len(foreground),
            "original_foreground": len(original_foreground),
            "augmented_foreground": len(augmented_foreground),
            "background_before": len(original_background) + len(breadth_ranked),
            "original_background_max": int(b_max),
            "background_max": int(b_max),
            "original_background_retained": int(original_to_keep),
            "augmented_background_retained": int(augmented_to_keep),
            "background_retained": int(original_to_keep + augmented_to_keep),
        }

    stats.background_max = int(total_b_max)
    stats.background_retained = sum(1 for cand in candidates if cand.keep and not cand.foreground)
    stats.dropped = int(stats.original_background_dropped + stats.augmented_background_dropped)
    for subset_name, subset_stats in stats.subset_stats.items():
        if int(subset_stats["background_retained"]) > int(subset_stats["background_max"]):
            raise RuntimeError(f"Internal error: {subset_name} background filtering exceeded its B_max")
    return stats


def withhold_background_overage_after_flips(
    physical_candidates: Sequence[OutputCandidate],
    *,
    flips_by_subset: Mapping[str, int],
    background_percent: float,
    labels_available: bool,
    background_stats: BackgroundFilterStats,
    warnings: WarningSink,
) -> int:
    """Tighten each background cap before rendering after foreground flips."""
    p = float(background_percent)
    if p >= 1.0 or not labels_available:
        return 0
    if not any(int(value) > 0 for value in flips_by_subset.values()):
        return 0
    total_withheld = 0
    subset_names = sorted(
        {cand.split_subset or "all" for cand in physical_candidates},
        key=lambda value: {"train": 0, "val": 1, "all": 2}.get(value, 3),
    )
    for subset_name in subset_names:
        flips = int(flips_by_subset.get(subset_name, 0))
        if flips <= 0:
            continue
        subset = [
            cand for cand in physical_candidates
            if (cand.split_subset or "all") == subset_name
        ]
        planned_foreground = sum(1 for cand in subset if cand.keep and cand.foreground)
        realized_foreground = max(0, planned_foreground - flips)
        allowed = background_limit_from_foreground(realized_foreground, p)
        retained_original_background = sorted(
            [
                cand for cand in subset
                if cand.keep and not cand.foreground and int(cand.augmentation_index) == 0
            ],
            key=candidate_background_rank,
        )
        grouped_by_source: Dict[str, List[OutputCandidate]] = defaultdict(list)
        for cand in subset:
            if cand.keep and not cand.foreground and int(cand.augmentation_index) > 0:
                grouped_by_source[candidate_source_identity(cand)].append(cand)
        breadth_ranked: List[Tuple[Tuple[int, bytes, int, int], OutputCandidate]] = []
        for group in grouped_by_source.values():
            group.sort(key=candidate_background_rank)
            for position, cand in enumerate(group):
                digest, src_order, order = candidate_background_rank(cand)
                breadth_ranked.append(((int(position), digest, int(src_order), int(order)), cand))
        breadth_ranked.sort(key=lambda pair: pair[0])
        admission_order = [*retained_original_background, *[cand for _, cand in breadth_ranked]]
        excess = max(0, len(admission_order) - int(allowed))
        if excess <= 0:
            continue
        victims = admission_order[len(admission_order) - excess:]
        withheld_original = 0
        withheld_augmented = 0
        for cand in victims:
            cand.keep = False
            if int(cand.augmentation_index) == 0:
                withheld_original += 1
            else:
                withheld_augmented += 1
        total_withheld += int(excess)
        background_stats.background_retained -= int(excess)
        background_stats.dropped += int(excess)
        background_stats.original_background_dropped += int(withheld_original)
        background_stats.augmented_background_dropped += int(withheld_augmented)
        subset_stats = background_stats.subset_stats.get(subset_name)
        if subset_stats is not None:
            subset_stats["background_retained"] = int(subset_stats.get("background_retained", 0)) - int(excess)
            subset_stats["original_background_retained"] = int(subset_stats.get("original_background_retained", 0)) - int(withheld_original)
            subset_stats["augmented_background_retained"] = int(subset_stats.get("augmented_background_retained", 0)) - int(withheld_augmented)
        warnings.add(
            "background_cap_withheld_after_foreground_flips",
            f"{subset_name}: flips={flips}, realized_foreground={realized_foreground}, withheld_backgrounds={excess}",
        )
    return total_withheld


def finalize_augmentation_stats(
    stats: AugmentationStats,
    candidates: Sequence[OutputCandidate],
    *,
    split_active: bool,
) -> AugmentationStats:
    retained_augmented = [
        cand for cand in candidates
        if cand.keep and int(cand.augmentation_index) > 0
    ]
    final_eligible_originals = [
        cand for cand in candidates
        if cand.keep
        and int(cand.augmentation_index) == 0
        and (not split_active or cand.split_subset == "train")
    ]
    stats.eligible_originals = len(final_eligible_originals)
    stats.retained_augmented_copies = len(retained_augmented)
    stats.dropped_augmented_background = max(
        0,
        int(stats.planned_augmented_copies - stats.retained_augmented_copies),
    )
    if stats.eligible_originals > 0:
        stats.achieved_ratio = (
            float(stats.eligible_originals + stats.retained_augmented_copies)
            / float(stats.eligible_originals)
        )
    else:
        stats.achieved_ratio = 1.0
    retained = [cand for cand in candidates if cand.keep]
    if split_active:
        stats.final_train_files = sum(1 for cand in retained if cand.split_subset == "train")
        stats.final_val_files = sum(1 for cand in retained if cand.split_subset == "val")
    else:
        stats.final_unsplit_files = len(retained)
    return stats


def apply_background_filter(
    candidates: List[OutputCandidate],
    *,
    background_percent: float,
    labels_available: bool,
    warnings: WarningSink,
) -> BackgroundFilterStats:
    p = float(background_percent)
    stats = BackgroundFilterStats(
        active=p < 1.0,
        classification_performed=bool(p < 1.0 and labels_available),
    )
    if p >= 1.0:
        stats.skipped_reason = "--background_percent is 1.0"
        stats.foreground_before = sum(1 for cand in candidates if cand.foreground)
        stats.background_before = sum(1 for cand in candidates if not cand.foreground)
        stats.background_retained = stats.background_before
        return stats
    if not labels_available:
        stats.skipped_reason = "label operations are disabled for unlabeled volumes"
        warnings.add(
            "background_filter_disabled_for_unlabeled_volume",
            "--background_percent requires labels because background is defined by the YOLO polygon export",
        )
        return stats
    stats.foreground_before = sum(1 for cand in candidates if cand.foreground)
    stats.background_before = sum(1 for cand in candidates if not cand.foreground)
    subset_names = sorted(
        {cand.split_subset or "all" for cand in candidates},
        key=lambda value: {"train": 0, "val": 1, "all": 2}.get(value, 3),
    )
    total_b_max = 0
    for subset_name in subset_names:
        subset = [cand for cand in candidates if (cand.split_subset or "all") == subset_name]
        foreground = [cand for cand in subset if cand.foreground]
        background = sorted(
            [cand for cand in subset if not cand.foreground],
            key=candidate_background_rank,
        )
        for cand in foreground:
            cand.keep = True
        b_max = background_limit_from_foreground(len(foreground), p)
        total_b_max += int(b_max)
        background_to_keep = min(int(b_max), len(background))
        for position, cand in enumerate(background):
            cand.keep = int(position) < background_to_keep
        dropped = len(background) - background_to_keep
        stats.dropped += int(dropped)
        stats.original_background_dropped += int(dropped)
        stats.background_retained += int(background_to_keep)
        stats.subset_stats[subset_name] = {
            "foreground": len(foreground),
            "background_before": len(background),
            "background_max": int(b_max),
            "original_background_retained": int(background_to_keep),
            "augmented_background_retained": 0,
            "background_retained": int(background_to_keep),
        }
    stats.background_max = int(total_b_max)
    return stats


def candidate_atomic_key(cand: OutputCandidate, method: str) -> Tuple[object, ...]:
    physical_view_id = str(cand.physical_view_id or cand.parent_view_tag)
    if method == "volume":
        return (cand.volume_name,)
    if method == "view":
        return (cand.volume_name, physical_view_id)
    if method == "slice":
        return (cand.volume_name, physical_view_id, int(cand.frame_idx))
    raise ValueError(f"Unsupported split method: {method}")


def apply_dataset_split(
    candidates: List[OutputCandidate],
    *,
    active: bool,
    train_split: Optional[float],
    split_method: Optional[str],
    warnings: WarningSink,
    emit_warnings: bool = True,
) -> SplitStats:
    stats = SplitStats(active=bool(active), train_split=train_split, split_method=split_method)
    retained = [cand for cand in candidates if cand.keep]
    stats.frames_total = len(retained)
    if not active:
        return stats
    if train_split is None or split_method is None:
        raise ValueError("Internal error: active splitting requires train_split and split_method")
    train_f = float(train_split)
    if not (0.0 <= train_f <= 1.0):
        raise ValueError("--train_split must be in [0.0, 1.0]")
    if split_method not in {"volume", "view", "slice"}:
        raise ValueError("--split_method must be one of: volume, view, slice")

    unit_order: List[Tuple[object, ...]] = []
    seen_units: set[Tuple[object, ...]] = set()
    for cand in retained:
        key = candidate_atomic_key(cand, split_method)
        if key not in seen_units:
            unit_order.append(key)
            seen_units.add(key)
    n_units = len(unit_order)
    stats.atomic_units_total = int(n_units)
    target_train = max(
        0,
        min(n_units, split_round_half_toward_train(train_f * float(n_units))),
    )
    ranked_units = sorted(
        unit_order,
        key=lambda key: (
            stable_digest_rank("PTA-v4.0.2-split-unit", key),
            tuple(str(value) for value in key),
        ),
    )
    train_units = set(ranked_units[:target_train])
    stats.atomic_units_train = int(len(train_units))
    stats.atomic_units_val = int(n_units - len(train_units))

    for cand in retained:
        cand.split_subset = "train" if candidate_atomic_key(cand, split_method) in train_units else "val"
    if train_f == 1.0:
        for cand in retained:
            cand.split_subset = "train"
        stats.atomic_units_train = int(n_units)
        stats.atomic_units_val = 0

    stats.frames_train = sum(1 for cand in retained if cand.split_subset == "train")
    stats.frames_val = sum(1 for cand in retained if cand.split_subset == "val")
    stats.achieved_train_fraction = (
        float(stats.frames_train) / float(stats.frames_total)
        if stats.frames_total else 0.0
    )
    if abs(stats.achieved_train_fraction - train_f) > 0.10:
        stats.warning = (
            f"achieved train fraction {stats.achieved_train_fraction:.6f} differs from requested "
            f"--train_split {train_f:.6f} by more than 0.10"
        )
    if train_f < 1.0 and stats.frames_val == 0:
        extra = "val split is empty while --train_split < 1.0"
        stats.warning = f"{stats.warning}; {extra}" if stats.warning else extra
    if stats.warning and emit_warnings:
        warnings.add("split_best_effort_warning", stats.warning)
        print(f"WARNING: {stats.warning}", file=sys.stderr)
    return stats


def refresh_retained_original_split_stats(
    candidates: Sequence[OutputCandidate],
    *,
    stats: SplitStats,
    train_split: Optional[float],
    split_method: Optional[str],
    warnings: WarningSink,
    emit_warnings: bool = True,
) -> SplitStats:
    """Refresh retained-original split statistics without changing assignments."""
    if not stats.active:
        return stats
    if train_split is None or split_method is None:
        raise RuntimeError("Internal error: split reconciliation requires resolved split settings")
    retained_originals = sorted(
        [cand for cand in candidates if cand.keep and int(cand.augmentation_index) == 0],
        key=lambda cand: (int(cand.source_order), int(cand.order)),
    )
    unit_order: List[Tuple[object, ...]] = []
    seen: set[Tuple[object, ...]] = set()
    unit_subset: Dict[Tuple[object, ...], str] = {}
    for cand in retained_originals:
        key = candidate_atomic_key(cand, split_method)
        if key not in seen:
            seen.add(key)
            unit_order.append(key)
        if cand.split_subset not in {"train", "val"}:
            raise RuntimeError(f"Retained original is missing its pre-filter split assignment: {cand}")
        existing = unit_subset.get(key)
        if existing is not None and existing != cand.split_subset:
            raise RuntimeError(f"Internal split conflict for atomic unit {key}: {existing} vs {cand.split_subset}")
        unit_subset[key] = str(cand.split_subset)

    stats.atomic_units_total = len(unit_order)
    stats.atomic_units_train = sum(1 for key in unit_order if unit_subset.get(key) == "train")
    stats.atomic_units_val = int(stats.atomic_units_total - stats.atomic_units_train)
    stats.frames_total = len(retained_originals)
    stats.frames_train = sum(1 for cand in retained_originals if cand.split_subset == "train")
    stats.frames_val = sum(1 for cand in retained_originals if cand.split_subset == "val")
    stats.achieved_train_fraction = (
        float(stats.frames_train) / float(stats.frames_total)
        if stats.frames_total else 0.0
    )
    stats.warning = ""
    if abs(stats.achieved_train_fraction - float(train_split)) > 0.10:
        stats.warning = (
            f"achieved train fraction {stats.achieved_train_fraction:.6f} differs from requested "
            f"--train_split {float(train_split):.6f} by more than 0.10"
        )
    if float(train_split) < 1.0 and stats.frames_val == 0:
        extra = "val split is empty while --train_split < 1.0"
        stats.warning = f"{stats.warning}; {extra}" if stats.warning else extra
    if stats.warning and emit_warnings:
        warnings.add("split_best_effort_warning", stats.warning)
        print(f"WARNING: {stats.warning}", file=sys.stderr)
    return stats


def assign_volume_split_by_stem(specs: Sequence[object], *, train_split: float) -> Dict[str, str]:
    """Assign whole-volume split subsets globally from objects with ``stem``."""
    unit_keys = [(str(getattr(spec, "stem")),) for spec in specs]
    n_units = len(unit_keys)
    target_train = max(
        0,
        min(
            n_units,
            split_round_half_toward_train(float(train_split) * float(n_units)),
        ),
    )
    ranked = sorted(
        unit_keys,
        key=lambda key: (
            stable_digest_rank("PTA-v4.0.2-split-unit", key),
            tuple(str(value) for value in key),
        ),
    )
    train_units = set(ranked[:target_train])
    if float(train_split) == 1.0:
        train_units = set(unit_keys)
    return {key[0]: ("train" if key in train_units else "val") for key in unit_keys}


__all__ = [
    "AUGMENTATION_TAG_ALPHABET",
    "AUGMENTATION_TAG_LENGTH",
    "AugmentationStats",
    "BackgroundFilterStats",
    "OutputCandidate",
    "SplitStats",
    "WarningSink",
    "apply_background_filter",
    "apply_dataset_split",
    "assign_volume_split_by_stem",
    "augmentation_digest",
    "augmentation_seed_for_identity",
    "background_limit_from_foreground",
    "base62_tag_from_digest",
    "candidate_atomic_key",
    "candidate_background_rank",
    "candidate_source_identity",
    "deterministic_augmentation_tag",
    "finalize_augmentation_stats",
    "finalize_background_filter_with_augmentations",
    "output_source_identity_text",
    "plan_augmented_versions",
    "refresh_retained_original_split_stats",
    "split_round_half_toward_train",
    "stable_digest_rank",
    "withhold_background_overage_after_flips",
]
