from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from XTA import cli


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "GPT-5.6-Sol-Ultra_v18.0.1_SLURM.py"


class CliTests(unittest.TestCase):
    def run_python(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env={**os.environ, "YOLO_TTA_TELEMETRY": "0"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_top_help_and_version_are_dependency_light(self) -> None:
        completed = self.run_python(str(LAUNCHER), "--help")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("--mode {tta,pta}", completed.stdout)
        self.assertIn("tta --help", completed.stdout)

        completed = self.run_python(str(LAUNCHER), "--version")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("18.0.1", completed.stdout)

        for mode in ("tta", "pta"):
            with self.subTest(mode_version=mode):
                completed = self.run_python(
                    str(LAUNCHER), "--mode", mode, "--version"
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)
                self.assertIn("18.0.1", completed.stdout)

        program = (
            "import sys; import XTA.cli; "
            "forbidden={'cv2','scipy','torch','cupy','openvino','ultralytics','jax'}; "
            "loaded={name.split('.')[0] for name in sys.modules}; "
            "assert not (forbidden & loaded), forbidden & loaded"
        )
        completed = self.run_python("-c", program)
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_mode_is_required_and_strict(self) -> None:
        completed = self.run_python(str(LAUNCHER))
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("--mode", completed.stdout)

        completed = self.run_python(str(LAUNCHER), "--mode", "foreign")
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("invalid choice", completed.stdout)

        completed = self.run_python(
            str(LAUNCHER), "--mode", "tta", "--mode", "pta"
        )
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("exactly once", completed.stdout)

    def test_mode_specific_tta_help_uses_existing_parser(self) -> None:
        completed = self.run_python(str(LAUNCHER), "--mode", "tta", "--help")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("YOLO segmentation TTA", completed.stdout)
        self.assertIn("--enable_radial", completed.stdout)
        self.assertIn("default: 3072", completed.stdout)

    def test_run_forwards_only_mode_local_arguments(self) -> None:
        tta_arguments = ["--input", "input.mkv", "--model", "gpu:model.engine"]
        with mock.patch.object(cli, "_run_tta") as run_tta:
            cli.run(["--mode", "tta", *tta_arguments])
        run_tta.assert_called_once_with(tta_arguments)

        pta_arguments = ["--input", "volume.nrrd", "--save", "nrrd"]
        with mock.patch.object(cli, "_run_pta") as run_pta:
            cli.run(["--mode=pta", *pta_arguments])
        run_pta.assert_called_once_with(pta_arguments)

    def test_tta_foreign_flags_are_argparse_errors_before_pipeline_import(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli._run_tta(
                    [
                        "--input",
                        "input.mkv",
                        "--model",
                        "gpu:model.engine",
                        "--pta-only-flag",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --pta-only-flag", stderr.getvalue())

    def test_tta_rejects_repeated_channel_format(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli._run_tta(
                    [
                        "--input",
                        "input.mkv",
                        "--model",
                        "gpu:model.engine",
                        "--channel_format",
                        "gray",
                        "--channel_format=C3S1",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--channel_format may be provided only once", stderr.getvalue())

    def test_tta_delegate_receives_sys_argv_without_mode(self) -> None:
        arguments = ["--input", "input.mkv", "--model", "gpu:model.engine"]
        parser = mock.Mock(spec=argparse.ArgumentParser)
        observed: list[str] = []

        def delegate() -> None:
            observed.extend(sys.argv[1:])

        with mock.patch("XTA.config.build_argparser", return_value=parser):
            with mock.patch("XTA.tta_mode.run", side_effect=delegate):
                cli._run_tta(arguments)

        parser.parse_args.assert_called_once_with(arguments)
        self.assertEqual(observed, arguments)

    def test_pta_module_is_imported_only_by_pta_delegate(self) -> None:
        fake_module = types.ModuleType("XTA.pta_mode")
        fake_run = mock.Mock()
        fake_module.run = fake_run  # type: ignore[attr-defined]

        # Other PTA tests may have imported the real module earlier in the same
        # discovery process. Isolate this lazy-import assertion from suite order.
        previous = sys.modules.pop("XTA.pta_mode", None)
        try:
            self.assertNotIn("XTA.pta_mode", sys.modules)
            with mock.patch.dict(sys.modules, {"XTA.pta_mode": fake_module}):
                cli._run_pta(["--input", "volume.nrrd"])
        finally:
            if previous is not None:
                sys.modules["XTA.pta_mode"] = previous
        fake_run.assert_called_once_with(["--input", "volume.nrrd"])

    def test_pta_delegate_remains_responsible_for_foreign_flag_errors(self) -> None:
        fake_module = types.ModuleType("XTA.pta_mode")

        def fake_run(arguments: list[str]) -> None:
            parser = argparse.ArgumentParser(allow_abbrev=False)
            parser.add_argument("--input", required=True)
            parser.parse_args(arguments)

        fake_module.run = fake_run  # type: ignore[attr-defined]
        stderr = io.StringIO()
        with mock.patch.dict(sys.modules, {"XTA.pta_mode": fake_module}):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    cli._run_pta(["--input", "volume.nrrd", "--tta-only-flag"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --tta-only-flag", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
