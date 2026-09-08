"""CPU-only spherical request canonicalization and planning invariants."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import unittest

import numpy as np

from XTA.config import resolve_spherical_view_requests, resolve_tilted_view_groups
from XTA.geometry import get_view_infos
from XTA.spherical_geometry import build_spherical_view_infos
from XTA.unification.context import activate_unified_launch
from XTA.unification.runtime import compile_physical_views
from XTA.unification.tta_manifest import build_tta_run_manifest, spherical_view_plan_metadata
from tests.test_run_context_manifest import _Channel


class SphericalPlanningTests(unittest.TestCase):
    @staticmethod
    def _compile(targets, groups=()):
        return compile_physical_views(
            t_dim=11, height=13, width=15, cartesian_views=(), azimuthal_requests=(),
            tilted_groups=resolve_tilted_view_groups(groups),
            spherical_requests=resolve_spherical_view_requests(targets),
            spherical_patch_size=8, radial_min_radius=4.5,
        )

    def test_default_spherical_only_request_emits_one_dense_annular_cube(self):
        compiled = self._compile("transverse")
        self.assertTrue(compiled.views)
        self.assertEqual(compiled.radial_targets, ())
        self.assertEqual(compiled.azimuthal_targets, ())
        self.assertEqual({view.family for view in compiled.views}, {"spherical"})
        self.assertEqual({view.spherical_group for view in compiled.views}, {"upright"})
        self.assertEqual({view.spherical_face for view in compiled.views}, set(range(6)))
        for view in compiled.views:
            radii = np.asarray(view.spherical_radii)
            self.assertEqual(radii[0], 8 / (4 * math.pi))
            self.assertEqual(radii[-1], 5.0)
            self.assertTrue(np.all((np.diff(radii) > 0) & (np.diff(radii) <= 1)))
            self.assertEqual(view.num_slices, len(radii))

    def test_tilted_aliases_dedupe_within_direction_angle_groups_and_ignore_request_order(self):
        targets = ("tilted_transverse", "tilted_sagittal", "tilted_coronal")
        groups = ("transverse,sagittal,coronal:15:both", "coronal:30:horizontal")
        first = [view for view in self._compile(targets, groups).views if view.family == "spherical"]
        second = [view for view in self._compile(tuple(reversed(targets)), tuple(reversed(groups))).views if view.family == "spherical"]
        self.assertEqual(first, second)
        expected = {"horizontal_m15", "horizontal_p15", "horizontal_m30", "horizontal_p30", "vertical_m15", "vertical_p15"}
        counts = Counter(view.spherical_group for view in first)
        self.assertEqual(set(counts), expected)
        one_cube = len(self._compile("transverse").views)
        self.assertEqual(set(counts.values()), {one_cube})
        for view in first:
            self.assertTrue(view.spherical_tilted_source)
            aliases = ("tilted_coronal",) if abs(view.tilt_angle_deg) == 30 else targets
            self.assertEqual(view.spherical_request_tokens, aliases)
        self.assertEqual(
            [spherical_view_plan_metadata(view) for view in first],
            [spherical_view_plan_metadata(view) for view in second],
        )

    def test_internal_zero_angle_tilt_groups_stay_distinct_from_upright(self):
        # The public Tilted grammar requires positive angles. Internal concrete
        # sources may nevertheless have identity rotations, which must retain
        # their direction/group identity rather than merging with the upright cube.
        source = get_view_infos(
            11, 13, 15, cartesian_views=(),
            tilt_groups=resolve_tilted_view_groups("transverse:15:vertical"),
        )[0]
        sources = tuple(
            replace(source, tilt_base_view=base, tilt_direction=direction, tilt_angle_deg=0.0)
            for base in ("transverse", "sagittal", "coronal")
            for direction in ("vertical", "horizontal")
        )
        views = build_spherical_view_infos(
            11, 13, 15, targets=("transverse", "sagittal", "tilted_transverse", "tilted_sagittal", "tilted_coronal"),
            min_radius=None, patch_size=8, tilted_views=sources,
        )
        counts = Counter(view.spherical_group for view in views)
        self.assertEqual(set(counts), {"upright", "horizontal_p0", "vertical_p0"})
        self.assertEqual(set(counts.values()), {len(self._compile("transverse").views)})
        self.assertEqual(len({view.name for view in views}), len(views))
        for view in views:
            np.testing.assert_array_equal(np.asarray(view.spherical_rotation_xyz).reshape(3, 3), np.eye(3))

    def test_real_geometry_manifest_alias_counts_match_canonical_workload(self):
        targets = ("transverse", "sagittal", "tilted_transverse", "tilted_sagittal")
        compiled = self._compile(targets, ("transverse,sagittal:15:vertical",))
        views = tuple(view for view in compiled.views if view.family == "spherical")
        with activate_unified_launch(version="21.0.0", launcher="xta", mode="tta", mode_arguments=()) as launch:
            manifest = build_tta_run_manifest(
                launch_context=launch, pipeline_version="21.0.0", resolved_config={}, artifact_identities={},
                source_shape_tyx=(11, 13, 15), processing_shape_tyx=(11, 13, 15), fps=1.0,
                physical_views=views, inference_views=views, angles=(0.0,), channel_format=_Channel(),
                tile_configs=(), azimuthal_requests=(), azimuthal_diameters=(), azimuthal_azimuth_angles=(),
                spherical_requests=resolve_spherical_view_requests(targets), backend={}, forward_sampling={},
                prediction_processing={}, requested_outputs=(), output_paths={},
            )
        geometry = manifest["geometry"]
        groups = geometry["spherical_groups"]
        self.assertEqual(len(groups), 3)
        self.assertEqual(sum(len(group["concrete_patch_trajectories"]) for group in groups), len(views))
        aliases = {record["view"]: set(record["canonical_groups"]) for record in geometry["spherical_requests"]}
        self.assertEqual(aliases["transverse"], {"upright"})
        self.assertEqual(aliases["sagittal"], {"upright"})
        self.assertEqual(aliases["tilted_transverse"], {"vertical_m15", "vertical_p15"})
        self.assertEqual(aliases["tilted_sagittal"], {"vertical_m15", "vertical_p15"})


if __name__ == "__main__":
    unittest.main()
