from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from XTA.config import (
    build_argparser,
    parse_radial_min_radius,
    resolve_azimuthal_view_requests,
    resolve_radial_view_requests,
)
from XTA.lta_config import parse_lta_args
from XTA.pta_config import parse_pta_args
from XTA.unification.context import activate_unified_launch
from XTA.unification.sampling import build_forward_raster_plan, forward_sampling_policy
from XTA.unification.tta_manifest import (
    build_tta_run_manifest,
    radial_view_manifest_record,
    radial_view_plan_metadata,
)
from tests.test_run_context_manifest import _View, _Channel


def _radial_view(**overrides):
    values = {
        **asdict(_View()),
        "name": "radial_transverse_p0_h0",
        "family": "radial",
        "summary_family": "radial",
        "display_name": "Radial Transverse",
        "azimuths_deg": (),
        "azimuthal_request_token": "",
        "src_h": 8,
        "src_w": 8,
        "radial_base_view": "transverse",
        "radial_tilted_source": False,
        "radial_source_view_name": "transverse",
        "radial_request_token": "transverse",
        "radial_min_radius": 1.5,
        "radial_max_radius": 3.0,
        "radial_step": 1.0,
        "radial_shell_start": 0,
        "radial_radii": (1.5, 2.5, 3.0),
        "radial_arc_origin": 0.0,
        "radial_height_origin": 0.0,
        "radial_patch_size": 8,
        "radial_patch_index": 0,
        "radial_height_index": 0,
    }
    return SimpleNamespace(**{**values, **overrides})


class CylindricalConfigTests(unittest.TestCase):
    def test_families_have_independent_public_grammars(self):
        args = build_argparser().parse_args([
            "--input", "source.mkv", "--model", "cpu:model.xml",
            "--enable_azimuthal", "transverse:2.5",
            "--enable_radial", "transverse,sagittal", "tilted_coronal",
            "--radial_min_radius", "1.25",
        ])
        self.assertEqual(resolve_azimuthal_view_requests(args.enable_azimuthal)[0].azimuth_angle, 2.5)
        self.assertEqual(
            [request.view for request in resolve_radial_view_requests(args.enable_radial)],
            ["transverse", "sagittal", "tilted_coronal"],
        )
        self.assertEqual(args.radial_min_radius, 1.25)
        for invalid in ("transverse:2.5", "transverse,transverse", "unknown"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_radial_view_requests(invalid)

    def test_minimum_radius_rejects_degenerate_and_nonfinite_values(self):
        self.assertIsNone(parse_radial_min_radius(None))
        self.assertIsNone(parse_radial_min_radius("auto"))
        self.assertEqual(parse_radial_min_radius("0.01"), 0.01)
        for invalid in ("0", "-1", "nan", "inf", "-inf", "bad"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                parse_radial_min_radius(invalid)

    def test_pta_shell_only_requires_a_declared_patch_size(self):
        config = parse_pta_args([
            "--input", "dataset", "--enable_radial", "transverse", "--imgsz", "64",
        ])
        self.assertTrue(config.has_physical_views)
        self.assertEqual([request.view for request in config.radial_requests], ["transverse"])
        self.assertIsNone(config.args.radial_min_radius)
        self.assertEqual(config.tiles, ())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_pta_args(["--input", "dataset", "--enable_radial", "transverse"])

    def test_tilted_shell_requires_corresponding_tilt_parent(self):
        base = ["--input", "dataset", "--imgsz", "64", "--enable_radial", "tilted_coronal"]
        self.assertFalse(parse_pta_args(base).has_physical_views)
        self.assertTrue(parse_pta_args(base + ["--enable_tilted", "coronal:15:vertical"]).has_physical_views)

    def test_lta_can_plan_shell_only_without_claiming_production_support(self):
        config = parse_lta_args([
            "--input", "target", "--output", "out", "--model", "sam-bundle",
            "--device", "0", "--enable_radial", "coronal", "--radial_min_radius", "auto",
        ])
        self.assertTrue(config.has_physical_views)
        self.assertEqual(config.radial_requests[0].view, "coronal")
        self.assertEqual(config.azimuthal_requests, ())

    def test_manifest_separates_angular_views_and_shell_trajectories(self):
        radial = _radial_view()
        with activate_unified_launch(
            version="20.0.0", launcher="xta", mode="tta", mode_arguments=(),
        ) as launch:
            manifest = build_tta_run_manifest(
                launch_context=launch, pipeline_version="20.0.0", resolved_config={},
                artifact_identities={}, source_shape_tyx=(3, 5, 7),
                processing_shape_tyx=(3, 5, 7), fps=1.0,
                physical_views=(_View(), radial), inference_views=(radial,),
                angles=(0.0,), channel_format=_Channel(), tile_configs=(),
                azimuthal_requests=resolve_azimuthal_view_requests("transverse:45"),
                azimuthal_diameters=(7,), azimuthal_azimuth_angles=(45.0,),
                radial_requests=resolve_radial_view_requests("transverse"),
                backend={}, forward_sampling={}, prediction_processing={},
                requested_outputs=(), output_paths={},
            )
        geometry = manifest["geometry"]
        self.assertEqual(manifest["schema"], "xta.v20.run_manifest/1")
        self.assertEqual(len(geometry["azimuthal_groups"][0]["concrete_azimuth_vectors"]), 1)
        trajectories = geometry["radial_groups"][0]["concrete_patch_trajectories"]
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0]["radial_radii"], [1.5, 2.5, 3.0])
        self.assertEqual(trajectories[0]["radial_patch_size"], 8)
        self.assertFalse(trajectories[0]["patch_is_tile"])
        self.assertEqual(trajectories[0]["channel_boundary"], "clamp_radius_within_patch_trajectory")
        self.assertIn("central_core_excluded", trajectories[0]["coverage_domain"])
        self.assertIsNone(radial_view_manifest_record(_View()))
        self.assertEqual(forward_sampling_policy().policy_version, 20)

    def test_shell_plan_metadata_retains_source_center_even_when_radius_range_matches(self):
        original = radial_view_plan_metadata(_radial_view())
        shifted = radial_view_plan_metadata(_radial_view(full_w=9, center_x=4.0))
        self.assertEqual(original['radial_shell']['radial_radii'], shifted['radial_shell']['radial_radii'])
        self.assertNotEqual(original, shifted)
        self.assertEqual(radial_view_plan_metadata(_View()), {})
        common = {
            'mode': 'pta', 'physical_view_id': 'radial_transverse_p0_h0',
            'angle_deg': 0.0, 'channel_token': 'gray', 'channel_kind': 'gray',
            'channel_count': 1, 'channel_stride': 1, 'channel_offsets': (0,),
            'channel_direction': 'ascending', 'output_shape': (8, 8),
        }
        self.assertNotEqual(
            build_forward_raster_plan(**common, metadata=original).digest,
            build_forward_raster_plan(**common, metadata=shifted).digest,
        )


if __name__ == "__main__":
    unittest.main()
