from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from XTA import geometry, pta, pta_rendering, pta_workers
from XTA.lta_config import parse_lta_args
from XTA.lta_execution import _require_supported_runtime
from XTA.lta_rendering import build_lta_rendered_view
from XTA.lta_runtime import build_lta_run_plan
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options
from tests import test_lta_runtime as lta_test_fixtures


class SphericalModeIntegrationTests(unittest.TestCase):
    @staticmethod
    def _pta_views():
        config = parse_pta_args([
            "--input", "dataset", "--enable_spherical", "transverse,sagittal,coronal",
            "--imgsz", "8", "--radial_min_radius", "3.5", "--enable_tile", "4:4",
        ])
        views, compiled = pta.compile_v18_pta_views(
            t_dim=11, h=13, w=15, config=config, azimuthal_native_raster=8,
        )
        return config, views, compiled

    def test_pta_compiles_canonical_cubes_and_retains_manifest_and_workload_identity(self):
        config, views, compiled = self._pta_views()
        single = replace(config, spherical_requests=config.spherical_requests[:1])
        single_views, _ = pta.compile_v18_pta_views(
            t_dim=11, h=13, w=15, config=single, azimuthal_native_raster=8,
        )
        self.assertEqual([view.name for view in views], [view.name for view in single_views])
        self.assertEqual(compiled.spherical_targets, ("transverse", "sagittal", "coronal"))
        for adapted, shared in zip(views, compiled.views):
            self.assertEqual(adapted.family, "spherical")
            self.assertIs(adapted.shared_view, shared)
            self.assertEqual(shared.spherical_min_radius, 8 / (4 * math.pi))
            self.assertEqual(shared.spherical_max_radius, 5.0)
            record = pta._v18_view_manifest_record(adapted)["spherical_geometry"]
            self.assertEqual(record["spherical_request_tokens"], ["transverse", "sagittal", "coronal"])
        plans = [SimpleNamespace(tag=view.name, view=view) for view in views]
        candidates = [SimpleNamespace(parent_view_tag=view.name) for view in views]
        self.assertEqual(pta_workers.projection_phase_summary(plans, candidates), (("spherical", len(views), len(views)),))
        with tempfile.TemporaryDirectory() as temporary:
            path = pta.write_v18_pta_manifest(
                Path(temporary) / "manifest.json", args=build_runtime_options(config), cli_argv=(),
                specs=(), records=(), channel_variants=(), tile_configs=(),
                augmentation_stats=pta.AugmentationStats(), total_written=0, input_identities=(),
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["resolved_configuration"]["spherical_requests"], list(compiled.spherical_targets))
        self.assertIsNone(manifest["resolved_configuration"]["spherical_min_radius"])
        self.assertIn("face_patch", manifest["forward_sampling"]["spherical_channel_boundary"])

    @unittest.skipIf(type(cv2).__name__ == '_StubModule', 'requires real OpenCV numerical runtime')
    def test_pta_cpu_fullframe_tiles_and_channel_directions_share_native_qsc_sampling(self):
        _, views, _ = self._pta_views()
        view = views[len(views) // 2]
        volume = np.arange(11 * 13 * 15, dtype=np.uint16).reshape(11, 13, 15).astype(np.uint8)
        mask = np.asarray(volume % 5 < 2, dtype=np.uint8)
        variants = pta.expand_channel_variants(pta.resolve_channel_formats(["C3S1"]))
        with tempfile.TemporaryDirectory() as temporary:
            for angle in (0.0, 90.0):
                aff = pta.build_affine(8, 8, angle, view.pad_mode, 8, shared_view=view.shared_view)
                for variant in variants:
                    with self.subTest(angle=angle, direction=variant.order_name):
                        plan = pta.build_render_plan(
                            view=view, aff=aff, tag=view.name, out_dir=Path(temporary), stem="sample",
                            tile_configs=(pta.TileConfig(4, 4, "s4_st4"),), save_overlay=False,
                            imgsz=8, label_enabled=True, channel_variant=variant,
                            publish_images=False, publish_labels=False,
                        )
                        source = pta_rendering.render_plan_frame_source(volume=volume, mask=mask, plan=plan, idx=0)
                        self.assertEqual(source.img_full.shape, (8, 8, 3))
                        self.assertEqual(len(source.tile_arrays), 4)
                        self.assertEqual(plan.canonical_plan.metadata["spherical_shell"]["spherical_face"], view.shared_view.spherical_face)
                        for channel, offset in enumerate(variant.offsets):
                            index, mirror = geometry.channel_view_slice_source(view.shared_view, offset)
                            self.assertFalse(mirror)
                            native = geometry.get_view_frame_by_index(volume, view.shared_view, index)
                            expected = cv2.warpAffine(native, aff.M_src_to_out, (8, 8), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            np.testing.assert_array_equal(source.img_full[:, :, channel], expected)
                        expected_mask = geometry.render_categorical_frame_on_grid(
                            mask, view.shared_view, 0, output_height=8, output_width=8,
                            M_src_to_out=aff.M_src_to_out, M_out_to_src=aff.M_out_to_src,
                        )
                        np.testing.assert_array_equal(source.mask_full, expected_mask)
                        for tile in plan.tile_layout:
                            tile_image, tile_mask = source.tile_arrays[tile.tile_tag]
                            self.assertEqual(tile_image.shape, (8, 8, 3))
                            self.assertEqual(tile_mask.shape, (8, 8))
                            self.assertIsNotNone(tile.canonical_plan.tile_layout)
                            self.assertEqual(tile.canonical_plan.metadata["spherical_shell"], plan.canonical_plan.metadata["spherical_shell"])
                            np.testing.assert_array_equal(tile_mask, geometry.render_categorical_dense_tile_for_job(mask, view.shared_view, tile.shared_job, 0))

    @unittest.skipIf(type(cv2).__name__ == '_StubModule', 'requires real OpenCV numerical runtime')
    def test_pta_partial_labels_and_encoded_gaps_remove_spherical_requests_before_compilation(self):
        config = parse_pta_args([
            "--input", "dataset", "--enable_cartesian", "transverse", "--enable_spherical", "transverse",
            "--imgsz", "8",
        ])
        options = build_runtime_options(config)
        for classification, indices in (("partially_labeled", (1, 2, 3)), ("fully_labeled", (1, 3, 4))):
            with self.subTest(classification=classification), tempfile.TemporaryDirectory() as temporary:
                source = pta.SourceVolume(
                    input_dir=Path("dataset"), stem="sample", kind="sequence",
                    image_paths=[Path(f"{index}.png") for index in indices], video_path=None,
                    labels_by_frame={}, segmentation_nrrd_path=Path("mask.nrrd"),
                    mask_volume=np.zeros((3, 2, 4), np.uint8), volume_class=classification,
                    label_source="nrrd", input_start_index=1, encoded_indices=indices,
                    volume=np.zeros((3, 2, 4), np.uint8), fps=1.0,
                )
                warnings = pta.WarningLog()
                prepared = pta.prepare_loaded_source(
                    source, args=options, warnings=warnings, workers=1, out_dir=Path(temporary),
                    tile_configs=(), channel_variants=(pta.DEFAULT_CHANNEL_VARIANT,),
                    write_side_effects=False, allocator=None,
                )
                self.assertEqual([view.family for view in prepared.views], ["transverse"])
                self.assertIn("spherical:transverse", " ".join(warnings.examples["partial_volume_3d_views_disabled"]))

    def test_lta_planning_records_qsc_geometry_but_production_explicitly_rejects_it(self):
        root = Path("spherical-lta-fixture")
        discovery = lta_test_fixtures.LtaRuntimePlanningTests._discovery(root)
        discovery = replace(discovery, target_volumes=(replace(discovery.target_volumes[0], frame_count=181, height=183, width=185, encoded_indices=tuple(range(181))),))
        config = parse_lta_args([
            "--input", str(root), "--output", str(root / "out"), "--model", "sam-bundle", "--device", "0",
            "--enable_spherical", "transverse,sagittal,coronal", "--radial_min_radius", "85", "--angle", "0,90",
        ])
        bundle = SimpleNamespace(root=root, checkpoint_path=root / "sam.pt", model_version="sam3.1", checkpoint_identity_sha256="a" * 64, bpe_path=None)
        with mock.patch("XTA.lta_runtime.resolve_local_sam_bundle", return_value=bundle):
            plan = build_lta_run_plan(config, discovery_fn=mock.Mock(return_value=discovery), run_id="spherical")
        views = plan.volumes[0].runtime_views
        self.assertEqual(len(views), 12)
        self.assertEqual(len(plan.view_assignments), 6)
        self.assertEqual({view.tta_angle_deg for view in views}, {0.0, 90.0})
        for view in views:
            record = view.manifest_record()["spherical_geometry"]
            self.assertEqual(record["spherical_min_radius"], 1008 / (4 * math.pi))
            self.assertEqual(record["spherical_max_radius"], 90.0)
            self.assertEqual(record["spherical_request_tokens"], ["transverse", "sagittal", "coronal"])
            self.assertTrue(view.raster_plan_digest)
        with self.assertRaisesRegex(ValueError, "Spherical QSC.*production"):
            _require_supported_runtime(plan)

    @unittest.skipIf(type(cv2).__name__ == '_StubModule', 'requires real OpenCV numerical runtime')
    def test_lta_low_level_spherical_rgb_renderer_retains_plan_geometry(self):
        _, views, _ = self._pta_views()
        physical = views[0].shared_view
        runtime = geometry.expand_views_into_tta_variants((physical,), (0.0,))[0]
        volume = np.arange(11 * 13 * 15, dtype=np.uint16).reshape(11, 13, 15).astype(np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            rendered = build_lta_rendered_view(volume, runtime, temp_dir=Path(temporary), output_size=8)
            rgb = rendered.render_frame_rgb(0)
        self.assertIn("spherical_shell", rendered.raster_plan.metadata)
        self.assertEqual(rgb.shape, (8, 8, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], geometry.get_view_frame_by_index(volume, physical, 0))
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])


if __name__ == "__main__":
    unittest.main()
