from __future__ import annotations

import tempfile
import unittest
import math
import json
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA import pta
from XTA import pta_publication
from XTA import pta_rendering
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options


class PtaGeometryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        def rotation_matrix(
            center: tuple[float, float], angle_deg: float, scale: float
        ) -> np.ndarray:
            alpha = math.cos(math.radians(float(angle_deg))) * float(scale)
            beta = math.sin(math.radians(float(angle_deg))) * float(scale)
            center_x, center_y = (float(value) for value in center)
            return np.asarray(
                [
                    [
                        alpha,
                        beta,
                        (1.0 - alpha) * center_x - beta * center_y,
                    ],
                    [
                        -beta,
                        alpha,
                        beta * center_x + (1.0 - alpha) * center_y,
                    ],
                ],
                dtype=np.float64,
            )

        self._rotation_patch = mock.patch.object(
            pta.shared_geometry.cv2,
            "getRotationMatrix2D",
            side_effect=rotation_matrix,
        )
        self._rotation_patch.start()

    def tearDown(self) -> None:
        self._rotation_patch.stop()

    @staticmethod
    def _runtime_args(arguments: list[str]):
        config = parse_pta_args(arguments)
        return config, build_runtime_options(config)

    def test_pta_and_tta_share_the_same_compiled_physical_view_objects(self) -> None:
        config = parse_pta_args(
            [
                "--input",
                "dataset",
                "--enable_cartesian",
                "transverse,sagittal",
                "--enable_radial",
                "transverse:60",
                "--enable_tilted",
                "coronal:15:horizontal",
            ]
        )

        adapted, compiled = pta.compile_v18_pta_views(
            t_dim=5,
            h=6,
            w=7,
            config=config,
            radial_native_raster=8,
        )

        self.assertEqual(len(adapted), len(compiled.views))
        self.assertTrue(adapted)
        for pta_view, shared_view in zip(adapted, compiled.views):
            self.assertIs(pta_view.shared_view, shared_view)
            self.assertEqual(pta_view.name, shared_view.name)
            self.assertEqual(pta_view.num_slices, shared_view.num_slices)
        radial = [view for view in adapted if view.family == "radial"]
        self.assertTrue(radial)
        self.assertEqual(radial[0].azimuths_deg, (0.0, 60.0, 120.0))

    def test_fullframe_intensity_and_categorical_outputs_delegate_to_tta(self) -> None:
        config = parse_pta_args(
            ["--input", "dataset", "--enable_cartesian", "transverse"]
        )
        views, _compiled = pta.compile_v18_pta_views(
            t_dim=3,
            h=4,
            w=5,
            config=config,
            radial_native_raster=6,
        )
        view = views[0]
        aff = pta.build_affine(
            view.src_w,
            view.src_h,
            0.0,
            view.pad_mode,
            6,
            shared_view=view.shared_view,
        )
        intensity = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
        categorical = np.asarray(intensity % 3 == 0, dtype=np.uint8)
        expected_image = np.full((6, 6), 37, dtype=np.uint8)
        expected_mask = np.zeros((6, 6), dtype=np.uint8)
        expected_mask[2:4, 1:5] = 1

        with (
            mock.patch.object(
                pta.shared_geometry,
                "render_intensity_frame_on_grid",
                return_value=expected_image,
            ) as image_render,
            mock.patch.object(
                pta.shared_geometry,
                "render_categorical_frame_on_grid",
                return_value=expected_mask,
            ) as mask_render,
        ):
            actual_image = pta._shared_render_full_intensity(
                intensity, view, 1, aff
            )
            actual_mask = pta._shared_render_full_mask(
                categorical, view, 1, aff
            )

        np.testing.assert_array_equal(actual_image, expected_image)
        np.testing.assert_array_equal(actual_mask, expected_mask)
        image_args = image_render.call_args.args
        mask_args = mask_render.call_args.args
        self.assertIs(image_args[0], intensity)
        self.assertIs(image_args[1], view.shared_view)
        self.assertEqual(image_args[2], 1)
        self.assertEqual(image_render.call_args.kwargs["output_height"], 6)
        self.assertEqual(image_render.call_args.kwargs["output_width"], 6)
        self.assertIs(mask_args[0], categorical)
        self.assertIs(mask_args[1], view.shared_view)
        self.assertEqual(mask_args[2], 1)
        self.assertEqual(mask_render.call_args.kwargs["output_height"], 6)
        self.assertEqual(mask_render.call_args.kwargs["output_width"], 6)

    def test_native_size_fullframe_uses_shared_grid_renderers(self) -> None:
        config = parse_pta_args(
            ["--input", "dataset", "--enable_cartesian", "transverse"]
        )
        views, _compiled = pta.compile_v18_pta_views(
            t_dim=3,
            h=4,
            w=5,
            config=config,
            radial_native_raster=0,
        )
        view = views[0]
        aff = pta.build_affine(
            view.src_w,
            view.src_h,
            0.0,
            view.pad_mode,
            0,
            shared_view=view.shared_view,
        )
        expected_image = np.full((4, 5), 12, dtype=np.uint8)
        expected_mask = np.ones((4, 5), dtype=np.uint8)
        with (
            mock.patch.object(
                pta.shared_geometry,
                "render_intensity_frame_on_grid",
                return_value=expected_image,
            ) as image_render,
            mock.patch.object(
                pta.shared_geometry,
                "render_categorical_frame_on_grid",
                return_value=expected_mask,
            ) as mask_render,
        ):
            image = pta._shared_render_full_intensity(
                np.zeros((3, 4, 5), dtype=np.uint8), view, 1, aff
            )
            mask = pta._shared_render_full_mask(
                np.zeros((3, 4, 5), dtype=np.uint8), view, 1, aff
            )

        self.assertTrue(aff.native_output)
        np.testing.assert_array_equal(image, expected_image)
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertEqual(image_render.call_args.kwargs["output_height"], 4)
        self.assertEqual(image_render.call_args.kwargs["output_width"], 5)
        self.assertEqual(mask_render.call_args.kwargs["output_height"], 4)
        self.assertEqual(mask_render.call_args.kwargs["output_width"], 5)

    def test_c1_has_one_direction_and_presentation_does_not_split_anatomy(self) -> None:
        variants = pta.expand_channel_variants(
            pta.resolve_channel_formats(["C1S3"])
        )
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].order_name, "forward")

        forward = pta.OutputCandidate(
            order=0,
            volume_name="sample",
            parent_view_tag="transverse_C3S1_forward",
            output_tag="transverse_C3S1_forward",
            item_key="full",
            frame_idx=2,
            is_tile=False,
            label_enabled=True,
            physical_view_id="transverse",
            presentation_variant_id="channel:C3S1:forward",
            geometry_item_id="full",
        )
        reverse = pta.replace(
            forward,
            order=1,
            parent_view_tag="transverse_C3S1_reverse",
            output_tag="transverse_C3S1_reverse",
            presentation_variant_id="channel:C3S1:reverse",
        )
        self.assertEqual(
            pta.candidate_atomic_key(forward, "view"),
            pta.candidate_atomic_key(reverse, "view"),
        )
        self.assertEqual(
            pta.candidate_atomic_key(forward, "slice"),
            pta.candidate_atomic_key(reverse, "slice"),
        )
        self.assertNotEqual(
            pta.candidate_source_identity(forward),
            pta.candidate_source_identity(reverse),
        )

    def test_tile_frame_source_uses_direct_tta_tile_rasters(self) -> None:
        config = parse_pta_args(
            ["--input", "dataset", "--enable_cartesian", "transverse"]
        )
        views, _compiled = pta.compile_v18_pta_views(
            t_dim=2,
            h=4,
            w=4,
            config=config,
            radial_native_raster=4,
        )
        view = views[0]
        aff = pta.build_affine(
            4, 4, 0.0, view.pad_mode, 4, shared_view=view.shared_view
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = pta.build_render_plan(
                view=view,
                aff=aff,
                tag="transverse",
                out_dir=Path(temp_dir),
                stem="sample",
                tile_configs=(pta.TileConfig(2, 2, "s2_st2"),),
                save_overlay=False,
                imgsz=4,
                label_enabled=True,
                publish_images=False,
                publish_labels=False,
            )

            full_image = np.full((4, 4), 9, dtype=np.uint8)
            full_mask = np.zeros((4, 4), dtype=np.uint8)

            def image_tile(*, tile: pta.RenderTileItem, **kwargs: object) -> np.ndarray:
                return np.full((tile.out_h, tile.out_w), tile.x + tile.y + 1, dtype=np.uint8)

            def mask_tile(
                _mask: np.ndarray,
                _view: object,
                tile_job: object,
                _idx: int,
            ) -> np.ndarray:
                return np.full(
                    (int(tile_job.out_size), int(tile_job.out_size)),
                    (int(tile_job.tile_x) + int(tile_job.tile_y)) % 2,
                    dtype=np.uint8,
                )

            with (
                mock.patch.object(
                    pta_rendering,
                    "_shared_render_full_intensity",
                    return_value=full_image,
                ),
                mock.patch.object(
                    pta_rendering,
                    "_shared_render_full_mask",
                    return_value=full_mask,
                ),
                mock.patch.object(
                    pta_rendering,
                    "render_shared_tile_images",
                    side_effect=image_tile,
                ) as image_tile_render,
                mock.patch.object(
                    pta.shared_geometry,
                    "render_categorical_dense_tile_for_job",
                    side_effect=mask_tile,
                ) as mask_tile_render,
                mock.patch.object(
                    pta_rendering,
                    "extract_padded_tile",
                    side_effect=AssertionError("canvas crop must not run"),
                ),
                mock.patch.object(
                    pta_rendering,
                    "resize_centered",
                    side_effect=AssertionError("second resize must not run"),
                ),
            ):
                source = pta.render_plan_frame_source(
                    volume=np.zeros((2, 4, 4), dtype=np.uint8),
                    mask=np.zeros((2, 4, 4), dtype=np.uint8),
                    plan=plan,
                    idx=0,
                    need_canvas=True,
                )

        self.assertIsNone(source.img_canvas)
        self.assertIsNone(source.mask_canvas)
        self.assertEqual(len(source.tile_arrays), len(plan.tile_layout))
        self.assertEqual(image_tile_render.call_count, len(plan.tile_layout))
        self.assertEqual(mask_tile_render.call_count, len(plan.tile_layout))
        for tile in plan.tile_layout:
            image, mask = source.tile_arrays[tile.tile_tag]
            self.assertEqual(image.shape, (tile.out_h, tile.out_w))
            self.assertEqual(mask.shape, (tile.out_h, tile.out_w))
            self.assertLessEqual(set(np.unique(mask).tolist()), {0, 1})

    def test_radial_custom_channels_use_shared_wrap_and_mirror_addresses(self) -> None:
        config = parse_pta_args(
            ["--input", "dataset", "--enable_radial", "transverse:60"]
        )
        views, _compiled = pta.compile_v18_pta_views(
            t_dim=3,
            h=4,
            w=5,
            config=config,
            radial_native_raster=0,
        )
        view = views[0]
        aff = pta.build_affine(
            view.src_w,
            view.src_h,
            0.0,
            view.pad_mode,
            0,
            shared_view=view.shared_view,
        )
        variant = pta.ChannelVariant(
            format_token="C3S1",
            kind="custom",
            channel_count=3,
            stride=1,
            reverse=False,
            offsets=(-1, 0, 1),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = pta.build_render_plan(
                view=view,
                aff=aff,
                tag="radial_C3S1",
                out_dir=Path(temp_dir),
                stem="sample",
                tile_configs=(),
                save_overlay=False,
                imgsz=0,
                label_enabled=False,
                channel_variant=variant,
                publish_images=False,
                publish_labels=False,
            )

            def render_plane(
                _volume: np.ndarray,
                _view: pta.ViewInfo,
                source_idx: int,
                _aff: pta.AffineSpec,
                *,
                mirror_radial_u: bool = False,
            ) -> np.ndarray:
                value = int(source_idx) + (100 if mirror_radial_u else 0)
                return np.full((_aff.out_h, _aff.out_w), value, dtype=np.uint8)

            with mock.patch.object(
                pta_rendering,
                "_shared_render_full_intensity",
                side_effect=render_plane,
            ):
                stacked, canvas = pta.render_channel_formatted_images(
                    volume=np.zeros((3, 4, 5), dtype=np.uint8),
                    plan=plan,
                    idx=0,
                    need_canvas=False,
                )

        expected_addresses = [
            pta.shared_geometry.channel_view_slice_source(view.shared_view, offset)
            for offset in (-1, 0, 1)
        ]
        expected_values = [index + (100 if mirror else 0) for index, mirror in expected_addresses]
        self.assertIsNone(canvas)
        self.assertEqual(stacked.shape[2], 3)
        self.assertEqual([int(stacked[0, 0, index]) for index in range(3)], expected_values)
        self.assertTrue(expected_addresses[0][1])

    def test_no_selected_views_skips_transverse_invariant(self) -> None:
        source = pta.SourceVolume(
            input_dir=Path("input"),
            stem="sample",
            kind="video",
            image_paths=[],
            video_path=Path("input/sample.mkv"),
            labels_by_frame={},
            segmentation_nrrd_path=None,
            mask_volume=None,
            volume_class="fully_labeled",
            label_source="yolo",
            input_start_index=None,
            encoded_indices=(0, 1, 2),
            volume=np.zeros((3, 2, 2), dtype=np.uint8),
            fps=30.0,
        )
        prep = pta.PreparedVolume(
            src=source,
            source_shape=(3, 2, 2),
            processing_shape=(3, 2, 2),
            effective_volume_class="fully_labeled",
            label_enabled=True,
            annotation_states=(pta.ANNOTATION_FOREGROUND,) * 3,
            save_overlay=False,
            volume_for_render=source.volume,
            mask_for_render=np.ones_like(source.volume),
            views=[],
            plans=[],
            smoothing_stats=[],
            nrrd_paths=[],
            voxel_initial=None,
            voxel_final=None,
            foreground_preservation_stats={
                "input_foreground_transverse_slices": 3,
            },
            v18_mode=True,
        )

        self.assertEqual(
            pta.validate_foreground_transverse_candidate_invariant(
                prep, (), retained_only=False
            ),
            0,
        )
        self.assertEqual(
            prep.foreground_preservation_stats[
                "classified_output_foreground_transverse_slices"
            ],
            0,
        )

    def test_diagnostics_only_positive_volume_skips_dataset_candidate_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            image_path = input_dir / "sample_0001.png"
            label_path = input_dir / "sample_0001.txt"
            image_path.write_bytes(b"image")
            label_path.write_text(
                "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                encoding="utf-8",
            )

            spec = pta.VolumeInputSpec(
                input_dir=input_dir,
                stem="sample",
                kind="sequence",
                image_paths_by_index={1: image_path},
                video_path=None,
                labels_by_index={1: label_path},
                segmentation_nrrd_path=None,
                volume_class="fully_labeled",
                label_source="yolo",
                input_start_index=1,
                encoded_indices=(1,),
            )
            volume = np.zeros((1, 2, 2), dtype=np.uint8)
            mask = np.ones_like(volume)
            source = pta.SourceVolume(
                input_dir=input_dir,
                stem="sample",
                kind="sequence",
                image_paths=[image_path],
                video_path=None,
                labels_by_frame={0: label_path},
                segmentation_nrrd_path=None,
                mask_volume=None,
                volume_class="fully_labeled",
                label_source="yolo",
                input_start_index=1,
                encoded_indices=(1,),
                volume=volume,
                fps=1.0,
            )
            view = pta.ViewInfo(
                name="transverse",
                display_name="Transverse",
                family="transverse",
                num_slices=1,
                src_h=2,
                src_w=2,
                pad_mode="clamp",
                full_t=1,
                full_h=2,
                full_w=2,
            )
            plan = mock.Mock(
                view=view,
                tag="transverse",
                stats={"tag": "transverse", "view": "Transverse"},
            )
            prepared = pta.PreparedVolume(
                src=source,
                source_shape=(1, 2, 2),
                processing_shape=(1, 2, 2),
                effective_volume_class="fully_labeled",
                label_enabled=True,
                annotation_states=(pta.ANNOTATION_FOREGROUND,),
                save_overlay=False,
                volume_for_render=volume,
                mask_for_render=mask,
                views=[view],
                plans=[plan],
                smoothing_stats=[],
                nrrd_paths=[],
                voxel_initial=None,
                voxel_final=None,
                foreground_preservation_stats={
                    "input_foreground_transverse_slices": 1,
                    "classified_output_foreground_transverse_slices": 0,
                    "retained_output_foreground_transverse_slices": 0,
                },
                v18_mode=True,
            )
            arguments = [
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--enable_cartesian",
                "transverse",
                "--save",
                "summary",
                "--worker_backend",
                "thread",
                "--pipeline_depth",
                "1",
                "--workers",
                "1",
                "--frame_workers",
                "1",
            ]
            _config, runtime_options = self._runtime_args(arguments)
            topology = mock.Mock(
                cuda_device_ids=(),
                worker_cpu_order=(),
                allowed_cpus=(0,),
                summary="test topology",
            )

            with (
                mock.patch.object(pta, "discover_topology", return_value=topology),
                mock.patch.object(pta, "discover_volume_specs", return_value=[spec]),
                mock.patch.object(
                    pta,
                    "load_source_volume_from_spec",
                    return_value=source,
                ),
                mock.patch.object(
                    pta,
                    "prepare_loaded_source",
                    return_value=prepared,
                ),
                mock.patch.object(
                    pta,
                    "validate_foreground_transverse_candidate_invariant",
                    wraps=pta.validate_foreground_transverse_candidate_invariant,
                ) as invariant,
                mock.patch.object(
                    pta,
                    "write_pta_summary",
                    return_value=output_dir / "summary.txt",
                ),
                mock.patch.object(
                    pta,
                    "write_v18_pta_manifest",
                    return_value=output_dir / "manifest.json",
                ),
                mock.patch.object(pta, "assert_v18_pta_inputs_unchanged"),
                mock.patch.object(pta, "_cleanup_v18_pta_selected_run_work"),
                mock.patch("builtins.print"),
            ):
                pta.main(args=runtime_options, argv=arguments)

            invariant.assert_not_called()

    def test_encoded_source_gaps_disable_all_3d_views_even_when_fully_labeled(self) -> None:
        config, runtime_options = self._runtime_args(
            [
                "--input",
                "dataset",
                "--enable_cartesian",
                "transverse,sagittal",
                "--enable_radial",
                "transverse:60",
                "--channel_format",
                "C3S1",
            ]
        )
        source = pta.SourceVolume(
            input_dir=Path("dataset"),
            stem="gapped",
            kind="sequence",
            image_paths=[Path("1.png"), Path("3.png"), Path("4.png")],
            video_path=None,
            labels_by_frame={},
            segmentation_nrrd_path=Path("mask.nrrd"),
            mask_volume=np.zeros((3, 2, 4), dtype=np.uint8),
            volume_class="fully_labeled",
            label_source="nrrd",
            input_start_index=1,
            encoded_indices=(1, 3, 4),
            volume=np.arange(3 * 2 * 4, dtype=np.uint8).reshape(3, 2, 4),
            fps=1.0,
        )
        warnings = pta.WarningLog()
        channel_formats = pta.resolve_channel_formats(runtime_options.channel_format)
        channel_variants = pta.expand_channel_variants(channel_formats)

        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = pta.prepare_loaded_source(
                source,
                args=runtime_options,
                warnings=warnings,
                workers=1,
                out_dir=Path(temp_dir),
                tile_configs=(),
                channel_variants=channel_variants,
                requested_tilt_angles=(),
                requested_tilt_directions=(),
                write_side_effects=False,
                allocator=None,
            )

        self.assertIs(prepared.volume_for_render, source.volume)
        self.assertEqual(prepared.processing_shape, source.volume.shape)
        self.assertEqual([view.family for view in prepared.views], ["transverse"])
        self.assertTrue(prepared.plans)
        self.assertTrue(
            all(plan.source_encoded_indices == (1, 3, 4) for plan in prepared.plans)
        )
        self.assertIn("cubic_resize_disabled_for_encoded_gaps", warnings.counts)
        self.assertIn("partial_volume_3d_views_disabled", warnings.counts)
        self.assertIs(prepared.views[0].shared_view, prepared.plans[0].view.shared_view)
        self.assertIs(config, runtime_options._v18_config)

    def test_fresh_output_safety_rejects_input_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            image_path = input_dir / "sample_0001.png"
            image_path.touch()
            spec = pta.VolumeInputSpec(
                input_dir=input_dir,
                stem="sample",
                kind="sequence",
                image_paths_by_index={1: image_path},
                video_path=None,
                labels_by_index={},
                segmentation_nrrd_path=None,
                volume_class="unlabeled",
                label_source="none",
                input_start_index=1,
                encoded_indices=(1,),
            )

            with self.assertRaises(ValueError):
                pta.validate_fresh_output_safety(
                    input_dir,
                    input_dir=input_dir,
                    specs=(spec,),
                )
            with self.assertRaises(ValueError):
                pta.validate_fresh_output_safety(
                    input_dir / "published",
                    input_dir=input_dir,
                    specs=(spec,),
                )
            with self.assertRaises(ValueError):
                pta.validate_fresh_output_safety(
                    root,
                    input_dir=input_dir,
                    specs=(spec,),
                )
            pta.validate_fresh_output_safety(
                root / "published",
                input_dir=input_dir,
                specs=(spec,),
            )

            nonowned = root / "nonowned"
            nonowned.mkdir()
            (nonowned / "unrelated.txt").write_text("keep")
            with self.assertRaises(ValueError):
                pta.validate_fresh_output_safety(
                    nonowned,
                    input_dir=input_dir,
                    specs=(spec,),
                )

            owned = root / "owned"
            owned.mkdir()
            pta.write_v18_output_sentinel(owned)
            (owned / "images").mkdir()
            pta.validate_fresh_output_safety(
                owned,
                input_dir=input_dir,
                specs=(spec,),
            )

    def test_fresh_output_safety_rejects_broad_and_false_owned_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            image_path = input_dir / "sample_0001.png"
            image_path.touch()
            spec = pta.VolumeInputSpec(
                input_dir=input_dir,
                stem="sample",
                kind="sequence",
                image_paths_by_index={1: image_path},
                video_path=None,
                labels_by_index={},
                segmentation_nrrd_path=None,
                volume_class="unlabeled",
                label_source="none",
                input_start_index=1,
                encoded_indices=(1,),
            )

            filesystem_root = Path(Path.cwd().anchor)
            for unsafe in (filesystem_root, Path.cwd()):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaises(ValueError):
                        pta.validate_fresh_output_safety(
                            unsafe,
                            input_dir=input_dir,
                            specs=(spec,),
                        )

            false_owned = root / "false_owned"
            false_owned.mkdir()
            (false_owned / ".pta_v18_output.json").write_text(
                json.dumps(
                    {
                        "schema": "pta.v18.output-directory/1",
                        "output": str(root / "somewhere_else"),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                pta.validate_fresh_output_safety(
                    false_owned,
                    input_dir=input_dir,
                    specs=(spec,),
                )

            owned = root / "owned_again"
            owned.mkdir()
            pta.write_v18_output_sentinel(owned)
            (owned / "manifest.json").write_text("stale", encoding="utf-8")
            pta.validate_fresh_output_safety(
                owned,
                input_dir=input_dir,
                specs=(spec,),
            )

    def test_custom_tiff_capability_is_not_probed_without_image_publication(self) -> None:
        class PreflightReached(RuntimeError):
            pass

        scenarios = (
            # A custom-channel zero-view run has no image frames to encode even
            # when image publication was requested.
            ["--input", "dataset", "--channel_format", "C3S1", "--save", "images"],
            # Labels-only publication must not depend on a TIFF encoder.
            [
                "--input",
                "dataset",
                "--channel_format",
                "C3S1",
                "--enable_cartesian",
                "transverse",
                "--save",
                "labels",
            ],
            # Geometry can be exercised for benchmarking without publishing
            # images or labels.
            [
                "--input",
                "dataset",
                "--channel_format",
                "C3S1",
                "--enable_cartesian",
                "transverse",
            ],
        )
        topology = mock.Mock(
            cuda_device_ids=(),
            worker_cpu_order=(),
            allowed_cpus=(0,),
            summary="test topology",
        )
        for arguments in scenarios:
            with self.subTest(arguments=arguments):
                _config, runtime_options = self._runtime_args(arguments)
                with (
                    mock.patch.object(
                        pta,
                        "ensure_tiff_output_available",
                        side_effect=AssertionError(
                            "TIFF capability must be checked only when an image is encoded"
                        ),
                    ) as capability_probe,
                    mock.patch.object(
                        pta,
                        "discover_topology",
                        return_value=topology,
                    ),
                    mock.patch.object(
                        pta,
                        "discover_volume_specs",
                        side_effect=PreflightReached,
                    ),
                ):
                    with self.assertRaises(PreflightReached):
                        pta.main(args=runtime_options, argv=arguments)
                capability_probe.assert_not_called()

    def test_gaussian_preprocessing_uses_tta_boundary_and_binary_pass_chaining(self) -> None:
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[0, 0, 0] = 1
        first_smoothed = np.asarray(
            [
                [[0.49, 0.50], [0.75, 0.0]],
                [[0.1, 0.2], [0.3, 0.4]],
            ],
            dtype=np.float32,
        )
        second_smoothed = np.asarray(
            [
                [[0.5, 0.49], [0.2, 0.9]],
                [[0.5, 0.0], [0.8, 0.1]],
            ],
            dtype=np.float32,
        )
        inputs: list[np.ndarray] = []
        kwargs_seen: list[dict[str, object]] = []

        def gaussian_filter(input: np.ndarray, **kwargs: object) -> np.ndarray:
            inputs.append(np.asarray(input).copy())
            kwargs_seen.append(dict(kwargs))
            return first_smoothed if len(inputs) == 1 else second_smoothed

        with mock.patch.object(
            pta.ndi,
            "gaussian_filter",
            side_effect=gaussian_filter,
        ):
            stats = pta.apply_gaussian_smoothing(
                mask,
                sigma=1.25,
                passes=2,
                warnings=pta.WarningLog(),
            )

        expected_first_binary = (first_smoothed >= 0.5).astype(np.uint8)
        expected_final = (second_smoothed >= 0.5).astype(np.uint8)
        np.testing.assert_array_equal(inputs[0], np.asarray([[[1, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.float32))
        np.testing.assert_array_equal(inputs[1], expected_first_binary.astype(np.float32))
        np.testing.assert_array_equal(mask, expected_final)
        self.assertEqual(len(stats), 2)
        for kwargs in kwargs_seen:
            self.assertEqual(kwargs["sigma"], 1.25)
            self.assertEqual(kwargs["mode"], "constant")
            self.assertEqual(kwargs["cval"], 0.0)
            self.assertEqual(kwargs["truncate"], 4.0)

    def test_manifest_records_resolved_geometry_sampling_and_output_selection(self) -> None:
        arguments = [
            "--input",
            "dataset",
            "--enable_radial",
            "transverse:60",
            "--channel_format",
            "C3S1",
            "--enable_tile",
            "4:2",
            "--save",
            "images",
            "labels",
            "voxel_volume",
        ]
        config, runtime_options = self._runtime_args(arguments)
        views, _compiled = pta.compile_v18_pta_views(
            t_dim=3,
            h=4,
            w=5,
            config=config,
            radial_native_raster=0,
        )
        channel_variants = pta.expand_channel_variants(
            pta.resolve_channel_formats(runtime_options.channel_format)
        )
        spec = pta.VolumeInputSpec(
            input_dir=Path("dataset"),
            stem="sample",
            kind="video",
            image_paths_by_index={},
            video_path=Path("dataset/sample.mkv"),
            labels_by_index={},
            segmentation_nrrd_path=Path("dataset/sample.nrrd"),
            volume_class="fully_labeled",
            label_source="nrrd",
            input_start_index=None,
            encoded_indices=(0, 1, 2),
        )
        record = pta.VolumeSummaryRecord(
            stem="sample",
            input_kind="video",
            volume_class="fully_labeled",
            effective_volume_class="fully_labeled",
            label_source="nrrd",
            label_enabled=True,
            source_shape=(3, 4, 5),
            processing_shape=(5, 5, 5),
            fps=30.0,
            input_start_index=None,
            encoded_indices=(0, 1, 2),
            annotation_state_counts={"annotated_foreground": 1},
            foreground_preservation_stats={"processed_anchor_repairs": 2},
            voxel_initial=7,
            voxel_final=8,
            smoothing_stats=[],
            nrrd_paths=[],
            views=views,
            tile_configs=[pta.TileConfig(4, 2, "s4_st2")],
            render_stats=[{"forward_geometry": "XTA.geometry"}],
            candidates_total=3,
            candidates_retained=2,
            candidates_written=2,
        )
        augmentation = pta.AugmentationStats(
            configured=True,
            path=Path("policy.py"),
            content_sha256="abc123",
            export_name="build_augmentation",
            runtime_backend="cpu",
            execution_mode="offline",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = pta.write_v18_pta_manifest(
                Path(temp_dir) / "manifest.json",
                args=runtime_options,
                cli_argv=arguments,
                specs=(spec,),
                records=(record,),
                channel_variants=channel_variants,
                tile_configs=(pta.TileConfig(4, 2, "s4_st2"),),
                augmentation_stats=augmentation,
                total_written=2,
            )
            manifest = json.loads(manifest_path.read_text())
            voxel_path = pta.write_v18_voxel_volume_report(
                Path(temp_dir) / "voxel_volume.json",
                (record,),
            )
            voxel = json.loads(voxel_path.read_text())

        self.assertEqual(manifest["pipeline_version"], "18.0.3")
        self.assertEqual(manifest["mode"], "pta")
        self.assertEqual(manifest["resolved_configuration"]["in_plane_variants_deg"], [0.0])
        self.assertEqual(
            [item["direction"] for item in manifest["resolved_configuration"]["channel_variants"]],
            ["forward", "reverse"],
        )
        self.assertEqual(
            manifest["volumes"][0]["views"][0]["azimuth_angles_deg"],
            [0.0, 60.0, 120.0],
        )
        self.assertEqual(manifest["forward_sampling"]["geometry_module"], "XTA.geometry")
        self.assertEqual(manifest["external_augmentation"]["sha256"], "abc123")
        self.assertTrue(
            manifest["external_augmentation"]["outside_shared_builtin_geometry_guarantee"]
        )
        self.assertEqual(manifest["outputs"]["selected"], ["images", "labels", "voxel_volume"])
        self.assertEqual(voxel["units"], "foreground_voxel_count")
        self.assertFalse(voxel["physical_volume"])
        self.assertEqual(voxel["volumes"][0]["initial_foreground_voxel_count"], 7)

    def test_pta_input_identities_detect_image_video_and_label_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "sample_0001.png"
            video = root / "sample.mkv"
            label = root / "sample_0001.txt"
            image.write_bytes(b"image")
            video.write_bytes(b"video")
            label.write_text("0 0.5 0.5\n", encoding="utf-8")
            spec = pta.VolumeInputSpec(
                input_dir=root,
                stem="sample",
                kind="sequence",
                image_paths_by_index={1: image},
                video_path=video,
                labels_by_index={1: label},
                segmentation_nrrd_path=None,
                volume_class="fully_labeled",
                label_source="yolo",
                input_start_index=1,
                encoded_indices=(1,),
            )

            for target in (image, video, label):
                with self.subTest(target=target.name):
                    identities = pta.capture_v18_pta_input_identities((spec,))
                    source = identities[0]
                    self.assertEqual(source["stem"], "sample")
                    self.assertEqual(source["encoded_indices"], [1])
                    expected_stat = target.stat()
                    nested = [
                        source["video"],
                        *source["image_paths"],
                        *source["label_paths"],
                    ]
                    target_identity = next(
                        identity
                        for identity in nested
                        if Path(identity["path"]) == target.resolve()
                    )
                    self.assertEqual(
                        target_identity["size_bytes"], expected_stat.st_size
                    )
                    self.assertEqual(
                        target_identity["modified_time_ns"], expected_stat.st_mtime_ns
                    )
                    self.assertIn("change_or_creation_time_ns", target_identity)
                    self.assertIn("device", target_identity)
                    self.assertIn("file_id", target_identity)

                    pta.assert_v18_pta_inputs_unchanged(identities)
                    target.write_bytes(target.read_bytes() + b"-mutated")
                    with self.assertRaisesRegex(
                        RuntimeError, "changed during execution"
                    ):
                        pta.assert_v18_pta_inputs_unchanged(identities)

    def test_complete_pta_manifest_nests_inputs_execution_and_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            image = input_dir / "sample_0001.png"
            label = input_dir / "sample_0001.txt"
            source_nrrd = input_dir / "sample.nrrd"
            image.write_bytes(b"image")
            label.write_text("0 0.5 0.5\n", encoding="utf-8")
            source_nrrd.write_bytes(b"nrrd")

            arguments = [
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--output_format",
                ".JPEG",
                "--channel_format",
                "C3S1",
                "--worker_backend",
                "thread",
                "--pipeline_depth",
                "1",
                "--save",
                "summary",
                "voxel_volume",
                "nrrd",
            ]
            config, runtime_options = self._runtime_args(arguments)
            spec = pta.VolumeInputSpec(
                input_dir=input_dir,
                stem="sample",
                kind="sequence",
                image_paths_by_index={1: image},
                video_path=None,
                labels_by_index={1: label},
                segmentation_nrrd_path=source_nrrd,
                volume_class="fully_labeled",
                label_source="nrrd",
                input_start_index=1,
                encoded_indices=(1,),
            )
            published_nrrd = output_dir / "nrrd" / "sample.nrrd"
            record = pta.VolumeSummaryRecord(
                stem="sample",
                input_kind="sequence",
                volume_class="fully_labeled",
                effective_volume_class="fully_labeled",
                label_source="nrrd",
                label_enabled=True,
                source_shape=(1, 2, 3),
                processing_shape=(1, 2, 3),
                fps=1.0,
                input_start_index=1,
                encoded_indices=(1,),
                annotation_state_counts={"annotated_foreground": 1},
                foreground_preservation_stats={},
                voxel_initial=1,
                voxel_final=1,
                smoothing_stats=[],
                nrrd_paths=[published_nrrd],
                views=[],
                tile_configs=[],
                render_stats=[],
                candidates_total=0,
                candidates_retained=0,
                candidates_written=0,
            )
            identities = pta.capture_v18_pta_input_identities((spec,))
            summary_path = output_dir / "summary.txt"
            voxel_path = output_dir / "voxel_volume.json"
            dataset_path = output_dir / "dataset.yaml"
            manifest_path = pta.write_v18_pta_manifest(
                output_dir / "manifest.json",
                args=runtime_options,
                cli_argv=arguments,
                specs=(spec,),
                records=(record,),
                channel_variants=pta.expand_channel_variants(
                    pta.resolve_channel_formats(runtime_options.channel_format)
                ),
                tile_configs=(),
                augmentation_stats=pta.AugmentationStats(),
                total_written=0,
                input_identities=identities,
                render_backend="thread",
                workers=7,
                frame_workers=5,
                planning_workers=2,
                topology_summary="cpu-only test topology",
                summary_path=summary_path,
                voxel_report_path=voxel_path,
                dataset_yaml_path=dataset_path,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "complete")
        resolved = manifest["resolved_configuration"]
        self.assertEqual(resolved["requested_output_format"], "jpg")
        self.assertEqual(resolved["effective_output_format"], "tif")
        self.assertEqual(config.requested_output_format, "jpg")
        self.assertEqual(config.effective_output_format, "tif")
        self.assertEqual(
            resolved["execution"],
            {
                "requested_worker_backend": "thread",
                "resolved_render_backend": "thread",
                "workers": 7,
                "frame_workers": 5,
                "planning_workers": 2,
                "pipeline_depth": 1,
                "topology_aware": True,
                "topology_summary": "cpu-only test topology",
            },
        )
        self.assertTrue(manifest["inputs"]["captured_before_execution"])
        source = manifest["inputs"]["artifacts"][0]
        self.assertEqual(source["image_paths"][0]["path"], str(image.resolve()))
        self.assertEqual(source["label_paths"][0]["path"], str(label.resolve()))
        self.assertEqual(
            source["segmentation_nrrd"]["path"], str(source_nrrd.resolve())
        )
        paths = manifest["outputs"]["paths"]
        self.assertEqual(paths["manifest"], str((output_dir / "manifest.json").resolve()))
        self.assertEqual(paths["summary"], str(summary_path.resolve()))
        self.assertEqual(paths["voxel_volume"], str(voxel_path.resolve()))
        self.assertEqual(paths["dataset_yaml"], str(dataset_path.resolve()))
        self.assertEqual(paths["nrrd"], [str(published_nrrd.resolve())])

    def test_main_publishes_summary_and_voxel_before_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            arguments = [
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--save",
                "summary",
                "voxel_volume",
            ]
            _config, runtime_options = self._runtime_args(arguments)
            summary_path = output_dir / "summary.txt"
            voxel_path = output_dir / "voxel_volume.json"
            manifest_path = output_dir / "manifest.json"
            calls: list[str] = []
            topology = mock.Mock(
                cuda_device_ids=(),
                worker_cpu_order=(),
                allowed_cpus=(0,),
                summary="test topology",
            )

            def summary_writer(*args: object, **kwargs: object) -> Path:
                calls.append("summary")
                return summary_path

            def voxel_writer(*args: object, **kwargs: object) -> Path:
                calls.append("voxel")
                return voxel_path

            def unchanged(*args: object, **kwargs: object) -> None:
                calls.append("input_stability")

            def cleanup(*args: object, **kwargs: object) -> None:
                calls.append("cleanup")

            def manifest_writer(*args: object, **kwargs: object) -> Path:
                calls.append("manifest")
                self.assertEqual(kwargs["summary_path"], summary_path)
                self.assertEqual(kwargs["voxel_report_path"], voxel_path)
                return manifest_path

            with (
                mock.patch.object(pta, "discover_topology", return_value=topology),
                mock.patch.object(pta, "discover_volume_specs", return_value=[]),
                mock.patch.object(pta, "write_pta_summary", side_effect=summary_writer),
                mock.patch.object(
                    pta,
                    "write_v18_voxel_volume_report",
                    side_effect=voxel_writer,
                ),
                mock.patch.object(
                    pta,
                    "assert_v18_pta_inputs_unchanged",
                    side_effect=unchanged,
                ),
                mock.patch.object(
                    pta,
                    "_cleanup_v18_pta_selected_run_work",
                    side_effect=cleanup,
                ),
                mock.patch.object(
                    pta,
                    "write_v18_pta_manifest",
                    side_effect=manifest_writer,
                ),
                mock.patch("builtins.print"),
            ):
                pta.main(args=runtime_options, argv=arguments)

        self.assertEqual(
            calls,
            ["summary", "voxel", "cleanup", "input_stability", "manifest"],
        )

    def test_primary_image_and_label_writes_follow_save_tokens(self) -> None:
        candidate = pta.OutputCandidate(
            order=0,
            volume_name="sample",
            parent_view_tag="transverse",
            output_tag="transverse",
            item_key="full",
            frame_idx=0,
            is_tile=False,
            label_enabled=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.object(pta_publication, "write_image") as write_image,
                mock.patch.object(
                    pta_publication, "write_yolo_lines"
                ) as write_labels,
            ):
                outcome = pta.write_selected_candidate_version(
                    cand=candidate,
                    image=np.zeros((2, 2), dtype=np.uint8),
                    mask=np.zeros((2, 2), dtype=np.uint8),
                    out_dir=Path(temp_dir),
                    split_active=False,
                    image_format="png",
                    png_compression=1,
                    jpeg_quality=95,
                    warnings=pta.WarningLog(),
                    augmentation=None,
                    save_images=False,
                    save_labels=False,
                )

        self.assertEqual(outcome, "written")
        write_image.assert_not_called()
        write_labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
