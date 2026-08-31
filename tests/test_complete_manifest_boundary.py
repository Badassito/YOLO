from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.smoke_import import install_stubs


install_stubs()

from XTA import pipeline
from XTA import pta
from XTA.pta_runtime import build_runtime_options
from XTA.pta_config import parse_pta_args


class CompleteManifestBoundaryTests(unittest.TestCase):
    def test_tta_finalize_facade_preserves_private_callback_patch_seams(self) -> None:
        manifest_path = Path("manifest.json")
        manifest = {"status": "complete"}
        identities = {"input": {}}
        with (
            mock.patch.object(pipeline, "_cleanup_tta_selected_run_scratch") as cleanup,
            mock.patch.object(
                pipeline,
                "_publish_complete_tta_manifest",
                return_value=manifest_path,
            ) as publish,
        ):
            result = pipeline._finalize_tta_selected_run_and_publish(
                temp_dir=Path("temp"),
                out_dir=Path("output"),
                keep_temp_artifacts=False,
                manifest_path=manifest_path,
                manifest=manifest,
                artifact_identities=identities,
            )

        self.assertEqual(result, manifest_path)
        cleanup.assert_called_once_with(
            temp_dir=Path("temp"), out_dir=Path("output")
        )
        publish.assert_called_once_with(
            path=manifest_path,
            manifest=manifest,
            artifact_identities=identities,
        )

    def test_tta_cleanup_failure_prevents_revalidation_and_complete_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            scratch_dir = output_dir / "temp"
            (scratch_dir / "locked").mkdir(parents=True)
            manifest_path = output_dir / "manifest.json"
            identities = {"input": {"path": "source.mkv"}}

            with (
                mock.patch.object(
                    pipeline.shutil,
                    "rmtree",
                    side_effect=OSError("injected locked scratch directory"),
                ),
                mock.patch.object(
                    pipeline, "assert_tta_artifacts_unchanged"
                ) as revalidate,
                mock.patch.object(pipeline, "write_json_manifest") as atomic_write,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "cleanup failed; refusing a complete manifest"
                ):
                    pipeline._finalize_tta_selected_run_and_publish(
                        temp_dir=scratch_dir,
                        out_dir=output_dir,
                        keep_temp_artifacts=False,
                        manifest_path=manifest_path,
                        manifest={"status": "complete"},
                        artifact_identities=identities,
                    )

            revalidate.assert_not_called()
            atomic_write.assert_not_called()
            self.assertFalse(manifest_path.exists())

    def test_tta_revalidates_immediately_before_atomic_complete_write(self) -> None:
        events: list[str] = []
        identities = {"input": {"path": "source.mkv"}}
        manifest = {"status": "complete", "mode": "tta"}
        manifest_path = Path("manifest.json")

        def revalidate(value: object) -> None:
            self.assertIs(value, identities)
            events.append("revalidate")

        def atomic_write(path: Path, value: object) -> Path:
            self.assertEqual(path, manifest_path)
            self.assertIs(value, manifest)
            events.append("atomic_write")
            return path

        with (
            mock.patch.object(
                pipeline,
                "assert_tta_artifacts_unchanged",
                side_effect=revalidate,
            ),
            mock.patch.object(
                pipeline,
                "write_json_manifest",
                side_effect=atomic_write,
            ),
        ):
            result = pipeline._finalize_tta_selected_run_and_publish(
                temp_dir=Path("unused"),
                out_dir=Path("unused-output"),
                keep_temp_artifacts=True,
                manifest_path=manifest_path,
                manifest=manifest,
                artifact_identities=identities,
            )

        self.assertEqual(result, manifest_path)
        self.assertEqual(events, ["revalidate", "atomic_write"])

    def test_tta_late_identity_failure_prevents_atomic_complete_write(self) -> None:
        with (
            mock.patch.object(
                pipeline,
                "assert_tta_artifacts_unchanged",
                side_effect=RuntimeError("injected late artifact mutation"),
            ),
            mock.patch.object(pipeline, "write_json_manifest") as atomic_write,
        ):
            with self.assertRaisesRegex(RuntimeError, "late artifact mutation"):
                pipeline._finalize_tta_selected_run_and_publish(
                    temp_dir=Path("unused"),
                    out_dir=Path("unused-output"),
                    keep_temp_artifacts=True,
                    manifest_path=Path("manifest.json"),
                    manifest={"status": "complete"},
                    artifact_identities={"input": {}},
                )

        atomic_write.assert_not_called()

    def test_pta_work_cleanup_failure_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / ".v18_work"
            work_dir.mkdir()
            (work_dir / "pending.bin").write_bytes(b"pending")
            with mock.patch.object(
                pta.shutil,
                "rmtree",
                side_effect=OSError("injected locked PTA work directory"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "cleanup failed; refusing a complete manifest"
                ):
                    pta._cleanup_v18_pta_selected_run_work(work_dir)

            self.assertTrue(work_dir.exists())

    def test_pta_main_cleanup_failure_prevents_complete_manifest(self) -> None:
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
            config = parse_pta_args(arguments)
            runtime_options = build_runtime_options(config)
            topology = mock.Mock(
                cuda_device_ids=(),
                worker_cpu_order=(),
                allowed_cpus=(0,),
                summary="test topology",
            )

            with (
                mock.patch.object(pta, "discover_topology", return_value=topology),
                mock.patch.object(pta, "discover_volume_specs", return_value=[]),
                mock.patch.object(
                    pta,
                    "write_pta_summary",
                    return_value=output_dir / "summary.txt",
                ),
                mock.patch.object(
                    pta,
                    "write_v18_voxel_volume_report",
                    return_value=output_dir / "voxel_volume.json",
                ),
                mock.patch.object(
                    pta,
                    "_cleanup_v18_pta_selected_run_work",
                    side_effect=RuntimeError("injected selected-run cleanup failure"),
                ),
                mock.patch.object(
                    pta, "assert_v18_pta_inputs_unchanged"
                ) as revalidate,
                mock.patch.object(pta, "write_v18_pta_manifest") as complete_write,
                mock.patch("builtins.print"),
            ):
                with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
                    pta.main(args=runtime_options, argv=arguments)

            revalidate.assert_not_called()
            complete_write.assert_not_called()
            self.assertFalse((output_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
