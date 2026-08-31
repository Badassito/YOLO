from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from tools.smoke_import import install_stubs


ROOT = Path(__file__).resolve().parents[1]


def _module_is_available(name: str) -> bool:
    loaded = sys.modules.get(name)
    if loaded is not None:
        return getattr(loaded, "__spec__", None) is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


if not all(_module_is_available(name) for name in ("cv2", "scipy", "tqdm")):
    install_stubs()

from XTA import pta_dataset
from XTA import pta
from XTA.pta_augmentation import LoadedAugmentation


class RecordingWarnings:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def add(self, key: str, msg: str = "") -> None:
        self.entries.append((str(key), str(msg)))


def _candidate(
    source_order: int,
    *,
    foreground: bool = True,
    physical_view_id: str | None = None,
) -> pta_dataset.OutputCandidate:
    view_id = physical_view_id or f"physical-{source_order}"
    return pta_dataset.OutputCandidate(
        order=int(source_order),
        volume_name="vol",
        parent_view_tag=f"view-{source_order}",
        output_tag=f"view-{source_order}",
        item_key="full",
        frame_idx=int(source_order),
        is_tile=False,
        label_enabled=True,
        physical_view_id=view_id,
        presentation_variant_id="channel:gray",
        geometry_item_id="full",
        source_order=int(source_order),
        foreground=bool(foreground),
    )


class PtaDatasetBoundaryTests(unittest.TestCase):
    COMPAT_NAMES = (
        "AUGMENTATION_TAG_ALPHABET",
        "AUGMENTATION_TAG_LENGTH",
        "AugmentationStats",
        "BackgroundFilterStats",
        "OutputCandidate",
        "SplitStats",
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
    )

    def test_pta_reexports_dataset_owner_objects_by_identity(self) -> None:
        self.assertEqual(
            tuple(pta_dataset.__all__),
            (
                "AUGMENTATION_TAG_ALPHABET",
                "AUGMENTATION_TAG_LENGTH",
                "AugmentationStats",
                "BackgroundFilterStats",
                "OutputCandidate",
                "SplitStats",
                "WarningSink",
                *self.COMPAT_NAMES[6:],
            ),
        )
        for name in self.COMPAT_NAMES:
            with self.subTest(name=name):
                owned = getattr(pta_dataset, name)
                self.assertIs(getattr(pta, name), owned)
                if inspect.isfunction(owned) or inspect.isclass(owned):
                    self.assertEqual(owned.__module__, "XTA.pta_dataset")

    def test_owner_import_does_not_pull_pta_geometry_or_worker_state(self) -> None:
        program = (
            "from tools.smoke_import import install_stubs; install_stubs(); "
            "import sys; import XTA.pta_dataset; "
            "forbidden={'XTA.pta','XTA.geometry','XTA.pta_scheduler','XTA.render_batch'}; "
            "assert not (forbidden & set(sys.modules)), forbidden & set(sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT,
            env={**os.environ, "YOLO_TTA_TELEMETRY": "0"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_identity_digest_and_base62_goldens(self) -> None:
        candidate = _candidate(0)
        self.assertEqual(
            pta_dataset.candidate_source_identity(candidate).split("\0"),
            ["vol", "physical-0", "channel:gray", "full", "0"],
        )
        fallback = replace(
            candidate,
            parent_view_tag="legacy-view",
            item_key="tile:1:2",
            physical_view_id="",
            presentation_variant_id="",
            geometry_item_id="",
        )
        self.assertEqual(
            pta_dataset.candidate_source_identity(fallback).split("\0"),
            ["vol", "legacy-view", "tile:1:2", "0"],
        )
        self.assertEqual(
            pta_dataset.stable_digest_rank(
                "PTA-v4.0.2-split-unit",
                ("vol-a", "view-a"),
            ).hex(),
            "7af878377a28697e9aeb22bb075c540a62ceeab3ea976318cfae6a7c821b19de",
        )
        self.assertEqual(
            pta_dataset.base62_tag_from_digest(bytes.fromhex("00" * 31 + "01")),
            "0000000000000001",
        )

    def test_fractional_ratio_plan_preserves_base_identity_and_golden_order(self) -> None:
        augmentation = LoadedAugmentation(
            path=Path("policy.py"),
            content_sha256="0123456789abcdef" * 4,
            export_name="build_augmentation",
            albumentations_version="2.0",
            pipeline_builder=lambda: None,
        )
        base = [
            _candidate(0, foreground=True),
            _candidate(1, foreground=False),
            _candidate(2, foreground=True),
        ]
        identities = [pta_dataset.candidate_source_identity(candidate) for candidate in base]
        used_tags: set[str] = set()
        versions, stats = pta_dataset.plan_augmented_versions(
            base,
            augmentation=augmentation,
            augmentation_ratio=1.5,
            split_active=False,
            augmented_foregrounds={
                (identities[0], 1): False,
                (identities[1], 1): True,
                (identities[2], 1): True,
            },
            require_augmented_foregrounds=True,
            used_tags=used_tags,
        )

        self.assertIs(versions[0], base[0])
        self.assertIs(versions[1], base[1])
        self.assertIs(versions[3], base[2])
        self.assertEqual(
            [
                (
                    candidate.source_order,
                    candidate.augmentation_index,
                    candidate.augmentation_tag,
                    candidate.augmentation_seed,
                    candidate.foreground,
                    candidate.order,
                )
                for candidate in versions
            ],
            [
                (0, 0, None, None, True, 0),
                (1, 0, None, None, False, 1),
                (1, 1, "sJKlqDVfIaUGJH9D", 2066662270, True, 2),
                (2, 0, None, None, True, 3),
                (2, 1, "W69ytd951wL3myBt", 3290071656, True, 4),
            ],
        )
        self.assertEqual(used_tags, {"sJKlqDVfIaUGJH9D", "W69ytd951wL3myBt"})
        self.assertEqual(stats.eligible_originals, 3)
        self.assertEqual(stats.planned_augmented_copies, 2)
        self.assertEqual(stats.planned_augmented_foreground, 2)
        self.assertEqual(stats.planned_augmented_background, 0)

    def test_background_retention_and_post_flip_withholding_are_deterministic(self) -> None:
        warnings = RecordingWarnings()
        candidates = [
            _candidate(index, foreground=index < 2)
            for index in range(6)
        ]
        stats = pta_dataset.apply_background_filter(
            candidates,
            background_percent=0.5,
            labels_available=True,
            warnings=warnings,
        )
        self.assertEqual(
            [(candidate.source_order, candidate.keep) for candidate in candidates],
            [(0, True), (1, True), (2, False), (3, False), (4, True), (5, True)],
        )
        self.assertEqual(stats.background_max, 2)
        self.assertEqual(stats.background_retained, 2)
        self.assertEqual(stats.dropped, 2)
        self.assertEqual(warnings.entries, [])

        withheld = pta_dataset.withhold_background_overage_after_flips(
            candidates,
            flips_by_subset={"all": 1},
            background_percent=0.5,
            labels_available=True,
            background_stats=stats,
            warnings=warnings,
        )
        self.assertEqual(withheld, 1)
        self.assertFalse(candidates[4].keep)
        self.assertTrue(candidates[5].keep)
        self.assertEqual(stats.background_retained, 1)
        self.assertEqual(stats.dropped, 3)
        self.assertEqual(
            warnings.entries,
            [
                (
                    "background_cap_withheld_after_foreground_flips",
                    "all: flips=1, realized_foreground=1, withheld_backgrounds=1",
                )
            ],
        )

    def test_split_and_global_volume_assignment_goldens(self) -> None:
        warnings = RecordingWarnings()
        candidates = [
            _candidate(index, physical_view_id=view_id)
            for index, view_id in enumerate(("zeta", "alpha", "gamma", "beta"))
        ]
        stats = pta_dataset.apply_dataset_split(
            candidates,
            active=True,
            train_split=0.5,
            split_method="view",
            warnings=warnings,
        )
        self.assertEqual(
            [(candidate.physical_view_id, candidate.split_subset) for candidate in candidates],
            [("zeta", "val"), ("alpha", "train"), ("gamma", "val"), ("beta", "train")],
        )
        self.assertEqual(
            (
                stats.atomic_units_total,
                stats.atomic_units_train,
                stats.atomic_units_val,
                stats.frames_train,
                stats.frames_val,
                stats.achieved_train_fraction,
                stats.warning,
            ),
            (4, 2, 2, 2, 2, 0.5, ""),
        )
        specs = [
            type("Spec", (), {"stem": stem})()
            for stem in ("zeta", "alpha", "gamma", "beta")
        ]
        self.assertEqual(
            pta_dataset.assign_volume_split_by_stem(specs, train_split=0.5),
            {"zeta": "val", "alpha": "train", "gamma": "train", "beta": "val"},
        )


if __name__ == "__main__":
    unittest.main()
