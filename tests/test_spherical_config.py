from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from XTA.config import (
    RADIAL_VIEW_TOKENS,
    SPHERICAL_VIEW_TOKENS,
    build_argparser,
    parse_spherical_min_radius,
    resolve_radial_view_requests,
    resolve_spherical_view_requests,
)
from XTA.lta_config import parse_lta_args
from XTA.pta_config import parse_pta_args
from XTA.unification.context import activate_unified_launch
from XTA.unification.sampling import build_forward_raster_plan, forward_sampling_policy
from XTA.unification.tta_manifest import (
    build_tta_run_manifest,
    spherical_view_manifest_record,
    spherical_view_plan_metadata,
)
from tests.test_run_context_manifest import _Channel, _View


def _spherical_view(**overrides):
    values = {
        **asdict(_View()),
        "name": "spherical_upright_face0_patch_u0_v0",
        "family": "spherical",
        "summary_family": "spherical",
        "display_name": "Spherical Upright / Face 0",
        "azimuths_deg": (),
        "azimuthal_request_token": "",
        "src_h": 8,
        "src_w": 8,
        "full_t": 9,
        "full_h": 11,
        "full_w": 13,
        "spherical_face": 0,
        "spherical_group": "upright",
        "spherical_request_tokens": ("transverse", "sagittal", "coronal"),
        "spherical_tilted_source": False,
        "spherical_min_radius": 1.5,
        "spherical_max_radius": 4.0,
        "spherical_step": 5.0 / 6.0,
        "spherical_radii": (1.5, 7.0 / 3.0, 19.0 / 6.0, 4.0),
        "spherical_face_intervals": 7,
        "spherical_patch_size": 8,
        "spherical_u_origin": 0,
        "spherical_v_origin": 0,
        "spherical_patch_u": 0,
        "spherical_patch_v": 0,
        "spherical_rotation_xyz": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }
    return SimpleNamespace(**{**values, **overrides})


class SphericalConfigTests(unittest.TestCase):
    def test_six_token_grammar_retains_aliases_and_radius_defaults_are_independent(self):
        self.assertEqual(SPHERICAL_VIEW_TOKENS, RADIAL_VIEW_TOKENS)
        args = build_argparser().parse_args([
            "--input", "source.mkv", "--model", "cpu:model.xml",
            "--enable_radial", "sagittal", "--radial_min_radius", "2.5",
            "--enable_spherical", "transverse,sagittal", "coronal",
            "tilted_transverse,tilted_sagittal", "tilted_coronal",
        ])
        self.assertEqual(
            tuple(request.view for request in resolve_spherical_view_requests(args.enable_spherical)),
            SPHERICAL_VIEW_TOKENS,
        )
        self.assertEqual(resolve_radial_view_requests(args.enable_radial)[0].view, "sagittal")
        self.assertEqual(args.radial_min_radius, 2.5)
        self.assertIsNone(args.spherical_min_radius)
        explicit = build_argparser().parse_args([
            "--input", "source.mkv", "--model", "cpu:model.xml",
            "--spherical_min_radius", "1.25",
        ])
        self.assertEqual(explicit.spherical_min_radius, 1.25)
        self.assertIsNone(explicit.radial_min_radius)
        for invalid in ("transverse:15", "transverse,transverse", "unknown"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "enable_spherical"):
                resolve_spherical_view_requests(invalid)

    def test_minimum_radius_accepts_auto_and_rejects_degenerate_values(self):
        self.assertIsNone(parse_spherical_min_radius(None))
        self.assertIsNone(parse_spherical_min_radius(" auto "))
        self.assertEqual(parse_spherical_min_radius("0.01"), 0.01)
        for invalid in ("0", "-1", "nan", "inf", "-inf", "bad"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                argparse.ArgumentTypeError, "spherical_min_radius"
            ):
                parse_spherical_min_radius(invalid)

    def test_pta_spherical_only_requires_a_positive_patch_size(self):
        config = parse_pta_args([
            "--input", "dataset", "--enable_spherical", "coronal", "--imgsz", "64",
            "--radial_min_radius", "10",
        ])
        self.assertTrue(config.has_physical_views)
        self.assertEqual(config.spherical_requests[0].view, "coronal")
        self.assertIsNone(config.args.spherical_min_radius)
        self.assertEqual(config.radial_requests, ())
        for size in ((), ("--imgsz", "0")):
            with self.subTest(size=size), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_pta_args(["--input", "dataset", "--enable_spherical", "transverse", *size])

    def test_tilted_spherical_request_requires_its_enabled_base(self):
        arguments = [
            "--input", "dataset", "--imgsz", "64", "--enable_spherical", "tilted_coronal",
        ]
        self.assertFalse(parse_pta_args(arguments).has_physical_views)
        self.assertTrue(parse_pta_args(arguments + ["--enable_tilted", "coronal:15:vertical"]).has_physical_views)

    def test_lta_retains_independent_spherical_planning_configuration(self):
        arguments = [
            "--input", "target", "--output", "out", "--model", "sam-bundle", "--device", "0",
        ]
        config = parse_lta_args([
            *arguments, "--enable_spherical", "sagittal", "--radial_min_radius", "25",
        ])
        self.assertTrue(config.has_physical_views)
        self.assertEqual(config.spherical_requests[0].view, "sagittal")
        self.assertIsNone(config.args.spherical_min_radius)
        self.assertEqual(config.radial_requests, ())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_lta_args([*arguments, "--enable_spherical", "tilted_transverse"])

    def test_manifest_counts_canonical_groups_once_and_maps_requested_aliases(self):
        upright = _spherical_view()
        second_face = _spherical_view(name="spherical_upright_face1", spherical_face=1)
        tilted = _spherical_view(
            name="spherical_vertical_p15_face0", spherical_group="vertical_p15",
            spherical_request_tokens=("tilted_transverse", "tilted_sagittal"),
            spherical_tilted_source=True, tilt_angle_deg=15.0, tilt_direction="vertical",
        )
        with activate_unified_launch(
            version="21.0.0", launcher="xta", mode="tta", mode_arguments=(),
        ) as launch:
            manifest = build_tta_run_manifest(
                launch_context=launch, pipeline_version="21.0.0", resolved_config={},
                artifact_identities={}, source_shape_tyx=(9, 11, 13),
                processing_shape_tyx=(9, 11, 13), fps=1.0,
                physical_views=(upright, second_face, tilted), inference_views=(upright,),
                angles=(0.0,), channel_format=_Channel(), tile_configs=(),
                azimuthal_requests=(), azimuthal_diameters=(), azimuthal_azimuth_angles=(),
                spherical_requests=resolve_spherical_view_requests(SPHERICAL_VIEW_TOKENS),
                backend={}, forward_sampling={}, prediction_processing={},
                requested_outputs=(), output_paths={},
            )
        geometry = manifest["geometry"]
        groups = geometry["spherical_groups"]
        self.assertEqual([group["group"] for group in groups], ["upright", "vertical_p15"])
        self.assertEqual([len(group["concrete_patch_trajectories"]) for group in groups], [2, 1])
        self.assertEqual(groups[0]["requested_views"], ["transverse", "sagittal", "coronal"])
        aliases = {item["view"]: item["canonical_groups"] for item in geometry["spherical_requests"]}
        self.assertEqual(aliases["coronal"], ["upright"])
        self.assertEqual(aliases["tilted_sagittal"], ["vertical_p15"])
        self.assertEqual(aliases["tilted_coronal"], [])
        self.assertEqual(geometry["radial_groups"], [])
        self.assertEqual(len(geometry["physical_views"]), 3)

    def test_plan_identity_includes_center_rotation_face_patch_and_radius_grid(self):
        original_view = _spherical_view()
        original = spherical_view_plan_metadata(original_view)
        self.assertEqual(original["spherical_shell"]["center_x_y_t"], [6.0, 5.0, 4.0])
        self.assertFalse(original["spherical_shell"]["patch_is_tile"])
        self.assertIn("central_core_excluded", original["spherical_shell"]["coverage_domain"])
        self.assertIsNone(spherical_view_manifest_record(_View()))
        self.assertEqual(spherical_view_plan_metadata(_View()), {})
        common = {
            "mode": "pta", "physical_view_id": original_view.name,
            "angle_deg": 0.0, "channel_token": "gray", "channel_kind": "gray",
            "channel_count": 1, "channel_stride": 1, "channel_offsets": (0,),
            "channel_direction": "ascending", "output_shape": (8, 8),
        }
        original_digest = build_forward_raster_plan(**common, metadata=original).digest
        for changed in (
            {"full_w": 15}, {"spherical_face": 1}, {"spherical_u_origin": 3},
            {"spherical_radii": (1.6, 2.4, 3.2, 4.0)},
            {"spherical_rotation_xyz": (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
        ):
            with self.subTest(changed=changed):
                record = spherical_view_plan_metadata(_spherical_view(**changed))
                self.assertNotEqual(build_forward_raster_plan(**common, metadata=record).digest, original_digest)
        policy = forward_sampling_policy()
        self.assertIn("spherical_qsc_face_patch_extraction", policy.stage_order)


if __name__ == "__main__":
    unittest.main()
