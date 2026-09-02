from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from XTA.lta_config import parse_lta_args
from XTA.lta_inputs import (
    AnnotationState,
    FrameAnnotation,
    LtaInputDiscovery,
    LtaVolumeSpec,
    PositiveExemplar,
    SourceRole,
    VolumeClass,
)
from XTA.lta_runtime import LtaPrototypeExecutionPending, build_lta_run_plan, run


class LtaRuntimePlanningTests(unittest.TestCase):
    def _config(self, root: Path, checkpoint: Path, *, execution: str = "video"):
        return parse_lta_args(
            [
                "--input",
                str(root),
                "--output",
                str(root / "out"),
                "--model",
                str(checkpoint),
                "--device",
                "0",
                "--enable_cartesian",
                "transverse",
                "--angle",
                "0,120",
                "--sam_execution",
                execution,
            ]
        )

    @staticmethod
    def _discovery(root: Path) -> LtaInputDiscovery:
        label = root / "sample_0000.txt"
        image = root / "sample_0000.png"
        annotation = FrameAnnotation(
            encoded_index=0,
            frame_position=0,
            state=AnnotationState.FOREGROUND,
            label_path=label,
            label_sha256="label",
            polygons=(),
        )
        volume = LtaVolumeSpec(
            source_role=SourceRole.TARGET,
            source_root=root,
            volume_id="input:sample",
            stem="sample",
            kind="sequence",
            media=(),
            video_path=None,
            video_sha256=None,
            video_identity_sha256=None,
            annotations=(annotation,),
            volume_class=VolumeClass.PARTIALLY_LABELED,
            encoded_indices=tuple(range(65)),
            index_origin=0,
            frame_count=65,
            width=80,
            height=64,
            fps=None,
        )
        exemplar = PositiveExemplar(
            exemplar_id="positive",
            source_role=SourceRole.TARGET,
            source_root=root,
            volume_id=volume.volume_id,
            volume_stem="sample",
            volume_kind="sequence",
            encoded_frame_index=0,
            frame_position=0,
            media_path=image,
            media_sha256="image",
            media_identity_sha256="image-identity",
            label_path=label,
            label_sha256="label",
            label_row_index=0,
            class_id=0,
            polygon=((0.1, 0.1), (0.2, 0.1), (0.2, 0.2)),
            box_xyxy=(0.1, 0.1, 0.2, 0.2),
            box_cxcywh=(0.15, 0.15, 0.1, 0.1),
            normalized_area=0.005,
            bundle_sha256="bundle",
        )
        return LtaInputDiscovery(
            input_path=root,
            target_volumes=(volume,),
            exemplar_roots=(),
            exemplar_volumes=(),
            positive_pool=(exemplar,),
            warnings=(),
        )

    @staticmethod
    def _physical_compiler(**_kwargs):
        return types.SimpleNamespace(
            views=(types.SimpleNamespace(name="transverse", num_slices=65),)
        )

    @staticmethod
    def _variant_expander(physical_views, angles):
        physical = physical_views[0]
        return tuple(
            types.SimpleNamespace(
                physical_view=physical,
                runtime_view=types.SimpleNamespace(
                    name=f"transverse__tta_a{int(angle)}",
                    num_slices=65,
                ),
                in_plane_variant=types.SimpleNamespace(angle_deg=float(angle)),
            )
            for angle in angles
        )

    def test_video_plan_crosses_views_angles_and_fixed_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "sam3.1_multiplex.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = self._config(root, checkpoint)
            discovery = self._discovery(root)

            plan = build_lta_run_plan(
                config,
                discovery_fn=mock.Mock(return_value=discovery),
                physical_compiler=self._physical_compiler,
                variant_expander=self._variant_expander,
                run_id="run-a",
            )

        views = plan.volumes[0].runtime_views
        self.assertEqual([view.tta_angle_deg for view in views], [0.0, 120.0])
        self.assertEqual(
            [(session.frame_start, session.frame_stop) for session in views[0].sessions],
            [(0, 30), (30, 60), (60, 65)],
        )
        self.assertEqual(plan.manifest_record()["prototype_stage"], "preflight_and_geometry_plan")
        self.assertEqual(plan.run_id, "run-a")
        self.assertEqual(plan.temp_root.name, "lta_run-a")
        self.assertTrue(views[0].sessions[0].sequence_id.startswith("input:sample::"))
        self.assertEqual(views[0].encoded_frame_indices, tuple(range(65)))
        self.assertIsNotNone(views[0].raster_plan_digest)
        self.assertEqual(plan.manifest_record()["channel_policy"], "implicit_rgb_v1")

    def test_image_execution_plans_independent_one_frame_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = self._config(root, checkpoint, execution="image")
            plan = build_lta_run_plan(
                config,
                discovery_fn=mock.Mock(return_value=self._discovery(root)),
                physical_compiler=self._physical_compiler,
                variant_expander=self._variant_expander,
                run_id="run-b",
            )

        sessions = plan.volumes[0].runtime_views[0].sessions
        self.assertEqual(len(sessions), 65)
        self.assertTrue(all(session.frame_count == 1 for session in sessions))

    def test_public_run_fails_loudly_before_false_complete_publication(self) -> None:
        fake_plan = types.SimpleNamespace(
            volumes=(),
            discovery=types.SimpleNamespace(positive_pool=()),
            device_ids=(0,),
        )
        with mock.patch("XTA.lta_runtime.build_lta_run_plan", return_value=fake_plan):
            with self.assertRaisesRegex(LtaPrototypeExecutionPending, "no outputs"):
                run(mock.Mock(), argv=[])


if __name__ == "__main__":
    unittest.main()
