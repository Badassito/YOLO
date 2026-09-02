from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from XTA import lta_inputs


POLYGON = "0 0.2 0.3 0.8 0.3 0.8 0.9 0.2 0.9\n"


def _write(path: Path, payload: bytes | str = b"media") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)
    return path


class LtaInputDiscoveryTests(unittest.TestCase):
    def test_roboflow_export_name_recovers_original_one_based_frame_index(self) -> None:
        image = Path(
            "M1_20_5_1_2026_8bit_RGB_0380_png.rf.aa7c7a78ad83a955e91976bea8a83e06.jpg"
        )
        label = image.with_suffix(".txt")

        self.assertEqual(
            lta_inputs.split_indexed_stem(image),
            ("M1_20_5_1_2026_8bit_RGB", 380),
        )
        self.assertEqual(lta_inputs.split_indexed_stem(label), lta_inputs.split_indexed_stem(image))

    def test_default_video_probe_counts_decoded_frames_not_packets(self) -> None:
        completed = mock.Mock(
            stdout=(
                '{"streams":[{"width":80,"height":64,'
                '"avg_frame_rate":"30/1","r_frame_rate":"30/1",'
                '"nb_read_frames":"17"}]}'
            )
        )
        with (
            mock.patch.object(lta_inputs.shutil, "which", return_value="ffprobe"),
            mock.patch.object(lta_inputs.subprocess, "run", return_value=completed) as run,
        ):
            metadata = lta_inputs.probe_video_with_ffprobe(Path("sample.mp4"))

        command = run.call_args.args[0]
        self.assertIn("-count_frames", command)
        self.assertTrue(any("nb_read_frames" in token for token in command))
        self.assertNotIn("-count_packets", command)
        self.assertEqual(metadata.frame_count, 17)

    def test_direct_image_input_builds_positive_and_warns_when_fully_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _write(root / "sample_0001.png", b"sample-image")
            _write(root / "sample_0001.txt", POLYGON)

            discovery = lta_inputs.discover_lta_inputs(
                image,
                image_probe=lambda _path: (640, 480),
            )

            self.assertEqual(len(discovery.target_volumes), 1)
            volume = discovery.target_volumes[0]
            self.assertEqual(volume.stem, "sample")
            self.assertEqual(volume.kind, "sequence")
            self.assertEqual(volume.encoded_indices, (1,))
            self.assertEqual(volume.volume_class, lta_inputs.VolumeClass.FULLY_LABELED)
            self.assertEqual(volume.annotations[0].state, lta_inputs.AnnotationState.FOREGROUND)
            self.assertEqual((volume.width, volume.height), (640, 480))
            self.assertEqual(len(discovery.positive_pool), 1)
            exemplar = discovery.positive_pool[0]
            self.assertEqual(exemplar.box_xyxy, (0.2, 0.3, 0.8, 0.9))
            self.assertAlmostEqual(exemplar.normalized_area, 0.36)
            self.assertTrue(exemplar.target_preference_capable)
            self.assertEqual(
                [warning.code for warning in discovery.warnings],
                ["fully_labeled_target"],
            )

    def test_multiple_target_stems_preserve_background_unknown_and_mixed_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(3):
                _write(root / f"alpha_{index:04d}.png", f"alpha-{index}".encode())
            _write(root / "alpha_0000.txt", POLYGON)
            _write(root / "alpha_0001.txt", "\n")
            _write(root / "beta_0001.png", b"beta")

            discovery = lta_inputs.discover_lta_inputs(root)
            by_stem = {volume.stem: volume for volume in discovery.target_volumes}

            self.assertEqual(set(by_stem), {"alpha", "beta"})
            alpha = by_stem["alpha"]
            self.assertEqual(alpha.volume_class, lta_inputs.VolumeClass.PARTIALLY_LABELED)
            self.assertEqual(
                tuple(annotation.state for annotation in alpha.annotations),
                (
                    lta_inputs.AnnotationState.FOREGROUND,
                    lta_inputs.AnnotationState.KNOWN_BACKGROUND,
                    lta_inputs.AnnotationState.UNKNOWN,
                ),
            )
            self.assertEqual(alpha.known_background_count, 1)
            self.assertEqual(alpha.unknown_count, 1)
            beta = by_stem["beta"]
            self.assertEqual(beta.volume_class, lta_inputs.VolumeClass.UNLABELED)
            self.assertEqual(beta.annotations[0].state, lta_inputs.AnnotationState.UNKNOWN)
            self.assertIn(
                "partially_labeled_target",
                {warning.code for warning in discovery.warnings},
            )

    def test_indexed_image_stack_rejects_mismatched_probed_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _write(root / "scan_0000.png", b"first")
            second = _write(root / "scan_0001.png", b"second")
            _write(root / "scan_0000.txt", POLYGON)
            sizes = {first.name: (640, 480), second.name: (641, 480)}

            with self.assertRaisesRegex(lta_inputs.LtaInputError, "identical dimensions"):
                lta_inputs.discover_lta_inputs(
                    root,
                    image_probe=lambda path: sizes[path.name],
                )

    def test_fully_labeled_background_target_is_allowed_with_external_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            exemplar = root / "exemplar"
            _write(target / "scan_0000.png", b"scan")
            _write(target / "scan_0000.txt", "")
            _write(exemplar / "reference_0000.png", b"reference")
            _write(exemplar / "reference_0000.txt", POLYGON)

            discovery = lta_inputs.discover_lta_inputs(target, [exemplar])

            self.assertEqual(
                discovery.target_volumes[0].volume_class,
                lta_inputs.VolumeClass.FULLY_LABELED,
            )
            self.assertEqual(
                discovery.target_volumes[0].annotations[0].state,
                lta_inputs.AnnotationState.KNOWN_BACKGROUND,
            )
            self.assertEqual(len(discovery.positive_pool), 1)
            self.assertEqual(
                discovery.positive_pool[0].source_role,
                lta_inputs.SourceRole.EXEMPLAR,
            )
            self.assertEqual(discovery.warnings[0].code, "fully_labeled_target")

    def test_identity_revalidation_detects_selected_image_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = _write(root / "sample_0000.png", b"before")
            _write(root / "sample_0000.txt", POLYGON)
            discovery = lta_inputs.discover_lta_inputs(root)
            image.write_bytes(b"after")

            with self.assertRaisesRegex(RuntimeError, "image changed"):
                lta_inputs.revalidate_lta_input_identities(discovery)

    def test_sparse_zero_based_video_labels_are_partial_and_missing_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "clip.mp4", b"encoded-video")
            _write(root / "clip_0000.txt", POLYGON)
            _write(root / "clip_0002.txt", "")

            discovery = lta_inputs.discover_lta_inputs(
                root,
                video_probe=lambda _path: lta_inputs.VideoMetadata(
                    frame_count=4,
                    width=1920,
                    height=1080,
                    fps=29.97,
                ),
            )

            volume = discovery.target_volumes[0]
            self.assertEqual(volume.kind, "video")
            self.assertEqual(volume.index_origin, 0)
            self.assertEqual(volume.encoded_indices, (0, 1, 2, 3))
            self.assertEqual(volume.volume_class, lta_inputs.VolumeClass.PARTIALLY_LABELED)
            self.assertEqual(
                tuple(annotation.state for annotation in volume.annotations),
                (
                    lta_inputs.AnnotationState.FOREGROUND,
                    lta_inputs.AnnotationState.UNKNOWN,
                    lta_inputs.AnnotationState.KNOWN_BACKGROUND,
                    lta_inputs.AnnotationState.UNKNOWN,
                ),
            )
            self.assertEqual((volume.width, volume.height, volume.fps), (1920, 1080, 29.97))
            self.assertIsNone(volume.video_sha256)
            self.assertIsNotNone(volume.video_identity_sha256)
            self.assertEqual(discovery.positive_pool[0].frame_position, 0)
            self.assertIsNone(discovery.positive_pool[0].media_sha256)

    def test_video_probe_mapping_accepts_num_frames_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "clip.mp4", b"video")
            _write(root / "clip_0000.txt", POLYGON)

            discovery = lta_inputs.discover_lta_inputs(
                root,
                video_probe=lambda _path: {
                    "num_frames": 2,
                    "width": 100,
                    "height": 50,
                    "fps": 6.0,
                },
            )

            self.assertEqual(discovery.target_volumes[0].frame_count, 2)

    def test_sparse_video_origin_must_be_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "clip.mp4", b"video")
            _write(root / "clip_0002.txt", POLYGON)

            with self.assertRaisesRegex(lta_inputs.LtaInputError, "ambiguous"):
                lta_inputs.discover_lta_inputs(
                    root,
                    video_probe=lambda _path: lta_inputs.VideoMetadata(frame_count=5),
                )

    def test_sparse_one_based_video_labels_use_frame_count_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "clip.mp4", b"video")
            _write(root / "clip_0002.txt", POLYGON)
            _write(root / "clip_0005.txt", "")

            discovery = lta_inputs.discover_lta_inputs(
                root,
                video_probe=lambda _path: 5,
            )

            volume = discovery.target_volumes[0]
            self.assertEqual(volume.index_origin, 1)
            self.assertEqual(volume.encoded_indices, (1, 2, 3, 4, 5))
            self.assertEqual(
                volume.annotation_for_index(1).state,
                lta_inputs.AnnotationState.UNKNOWN,
            )
            self.assertEqual(
                volume.annotation_for_index(2).state,
                lta_inputs.AnnotationState.FOREGROUND,
            )
            self.assertEqual(
                volume.annotation_for_index(5).state,
                lta_inputs.AnnotationState.KNOWN_BACKGROUND,
            )

    def test_video_labels_reject_out_of_range_and_unindexed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "clip.mp4", b"video")
            _write(root / "clip_0005.txt", POLYGON)
            with self.assertRaisesRegex(lta_inputs.LtaInputError, "out of range"):
                lta_inputs.discover_lta_inputs(
                    root,
                    video_probe=lambda _path: 4,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "clip.mp4", b"video")
            _write(root / "clip.txt", POLYGON)
            with self.assertRaisesRegex(lta_inputs.LtaInputError, "final '_NNNN'"):
                lta_inputs.discover_lta_inputs(
                    root,
                    video_probe=lambda _path: 4,
                )

    def test_orphans_duplicates_and_mixed_media_encodings_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "orphan_0001.txt", POLYGON)
            _write(root / "other_0001.png", b"other")
            with self.assertRaisesRegex(lta_inputs.LtaInputError, "without matching media"):
                lta_inputs.discover_lta_inputs(root)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "dup_0001.png", b"png")
            _write(root / "dup_0001.jpg", b"jpg")
            _write(root / "dup_0001.txt", POLYGON)
            with self.assertRaisesRegex(lta_inputs.LtaInputError, "Duplicate image"):
                lta_inputs.discover_lta_inputs(root)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "mixed_0000.png", b"image")
            _write(root / "mixed.mp4", b"video")
            _write(root / "mixed_0000.txt", POLYGON)
            with self.assertRaisesRegex(lta_inputs.LtaInputError, "Ambiguous media"):
                lta_inputs.discover_lta_inputs(root, video_probe=lambda _path: 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "dup_0001.png", b"image")
            _write(root / "dup_1.txt", POLYGON)
            _write(root / "dup_0001.txt", POLYGON)
            with self.assertRaisesRegex(lta_inputs.LtaInputError, "Duplicate label"):
                lta_inputs.discover_lta_inputs(root)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "scan_0000.png", b"image")
            _write(root / "scan_0000.txt", POLYGON)
            _write(root / "scan.nrrd", b"NRRD")
            with self.assertRaisesRegex(lta_inputs.LtaInputError, "not NRRD"):
                lta_inputs.discover_lta_inputs(root)

    def test_yolo_parser_rejects_nonzero_malformed_and_degenerate_rows(self) -> None:
        cases = (
            ("1 0 0 1 0 1 1\n", "only YOLO class 0"),
            ("0 0 0 1 0\n", "at least three"),
            ("0 0 0 1 0 2 1\n", "normalized"),
            ("0 0.2 0.2 0.4 0.4 0.6 0.6\n", "Degenerate"),
        )
        for payload, error in cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp_dir:
                path = _write(Path(temp_dir) / "label.txt", payload)
                with self.assertRaisesRegex(lta_inputs.LtaInputError, error):
                    lta_inputs.parse_yolo_segmentation_label(path)

    def test_no_positive_polygon_across_all_roots_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            exemplar = root / "exemplar"
            _write(target / "target_0000.png", b"target")
            _write(target / "target_0000.txt", "")
            _write(exemplar / "exemplar_0000.png", b"exemplar")

            with self.assertRaises(lta_inputs.NoPositiveExemplarError):
                lta_inputs.discover_lta_inputs(target, [exemplar])

            discovery = lta_inputs.discover_lta_inputs(
                target,
                [exemplar],
                require_positive=False,
            )
            self.assertEqual(discovery.positive_pool, ())

    def test_pool_deduplicates_identical_image_box_pairs_preferring_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            exemplar = root / "exemplar"
            _write(target / "target_0000.png", b"same-image")
            _write(target / "target_0000.txt", POLYGON)
            _write(exemplar / "external_0000.png", b"same-image")
            _write(exemplar / "external_0000.txt", POLYGON)

            discovery = lta_inputs.discover_lta_inputs(target, [exemplar])

            self.assertEqual(len(discovery.positive_pool), 1)
            self.assertEqual(
                discovery.positive_pool[0].source_role,
                lta_inputs.SourceRole.TARGET,
            )

    def test_pool_deduplicates_different_polygons_with_the_same_image_and_box(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(root / "sample_0000.png", b"same-image")
            _write(
                root / "sample_0000.txt",
                POLYGON + "0 0.2 0.3 0.8 0.3 0.5 0.6 0.8 0.9 0.2 0.9\n",
            )

            discovery = lta_inputs.discover_lta_inputs(root)

            self.assertEqual(discovery.target_volumes[0].positive_count, 2)
            self.assertEqual(len(discovery.positive_pool), 1)

    def test_session_ranking_is_deterministic_and_exposes_preference_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            exemplar = root / "exemplar"
            for name, payload in (
                ("alpha_0000", b"alpha-0"),
                ("alpha_0001", b"alpha-1"),
                ("beta_0000", b"beta-0"),
            ):
                _write(target / f"{name}.png", payload)
                _write(target / f"{name}.txt", POLYGON)
            _write(exemplar / "external_0000.png", b"external")
            _write(exemplar / "external_0000.txt", POLYGON)
            discovery = lta_inputs.discover_lta_inputs(target, [exemplar])
            alpha = next(volume for volume in discovery.target_volumes if volume.stem == "alpha")
            direct_id = next(
                item.exemplar_id
                for item in discovery.positive_pool
                if item.volume_id == alpha.volume_id and item.encoded_frame_index == 0
            )

            first = lta_inputs.rank_positive_exemplars_for_session(
                discovery.positive_pool,
                target_volume_id=alpha.volume_id,
                directly_addressable_exemplar_ids=(direct_id,),
                seed=17,
            )
            second = lta_inputs.rank_positive_exemplars_for_session(
                discovery.positive_pool,
                target_volume_id=alpha.volume_id,
                directly_addressable_exemplar_ids=(direct_id,),
                seed=17,
            )

            self.assertEqual(first, second)
            self.assertEqual(
                tuple(item.preference_tier for item in first),
                (
                    lta_inputs.ExemplarPreferenceTier.SAME_TARGET_SESSION,
                    lta_inputs.ExemplarPreferenceTier.SAME_TARGET_VOLUME,
                    lta_inputs.ExemplarPreferenceTier.OTHER_TARGET_VOLUME,
                    lta_inputs.ExemplarPreferenceTier.EXTERNAL_EXEMPLAR,
                ),
            )
            self.assertIn("directly addressable", first[0].preference_reason)

    def test_direct_media_file_does_not_absorb_unrelated_sibling_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = _write(root / "selected_0001.png", b"selected")
            _write(root / "selected_0001.txt", POLYGON)
            _write(root / "other_0001.png", b"other")
            _write(root / "orphan_0001.txt", POLYGON)

            discovery = lta_inputs.discover_lta_inputs(selected)

            self.assertEqual(
                tuple(volume.stem for volume in discovery.target_volumes),
                ("selected",),
            )

    def test_direct_indexed_image_matches_logical_index_across_zero_padding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = _write(root / "selected_0001.png", b"selected")
            _write(root / "selected_1.txt", POLYGON)

            discovery = lta_inputs.discover_lta_inputs(selected)

            self.assertEqual(len(discovery.positive_pool), 1)
            self.assertEqual(
                discovery.target_volumes[0].annotations[0].label_path.name,
                "selected_1.txt",
            )


if __name__ == "__main__":
    unittest.main()
