from __future__ import annotations

import contextlib
import importlib
import io
import types
import unittest
from unittest import mock


class LtaModeBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lta_mode = importlib.import_module("XTA.lta_mode")

    def test_help_and_invalid_flags_do_not_import_runtime(self) -> None:
        with mock.patch.object(self.lta_mode.importlib, "import_module") as import_module:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.lta_mode.run(["--help"])
            self.assertEqual(raised.exception.code, 0)
            import_module.assert_not_called()

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.lta_mode.run([
                        "--input", "target",
                        "--output", "published",
                        "--model", "sam-bundle",
                        "--device", "0",
                        "--enable_cartesian", "transverse",
                        "--prompt", "object",
                    ])
            self.assertEqual(raised.exception.code, 2)
            import_module.assert_not_called()

    def test_resolved_config_is_forwarded_to_future_runtime(self) -> None:
        arguments = [
            "--input", "target",
            "--output", "published",
            "--model", "sam-bundle",
            "--device", "2,0",
            "--enable_cartesian", "transverse",
            "--save", "images", "labels",
        ]
        runtime = types.SimpleNamespace(run=mock.Mock())
        with mock.patch.object(self.lta_mode, "_load_runtime_module", return_value=runtime):
            self.lta_mode.run(arguments)

        runtime.run.assert_called_once()
        config = runtime.run.call_args.args[0]
        self.assertEqual(runtime.run.call_args.kwargs["argv"], arguments)
        self.assertEqual(config.device_ids, (2, 0))
        self.assertEqual(config.cartesian_views, ("transverse",))
        self.assertEqual(config.save.tokens, ("images", "labels"))

    def test_planning_only_runtime_uses_controlled_nonzero_exit(self) -> None:
        class Pending(RuntimeError):
            pass

        runtime = types.SimpleNamespace(
            run=mock.Mock(side_effect=Pending("execution is not connected")),
            LtaPrototypeExecutionPending=Pending,
        )
        arguments = [
            "--input", "target",
            "--output", "published",
            "--model", "sam-bundle",
            "--device", "0",
            "--enable_cartesian", "transverse",
        ]
        stderr = io.StringIO()
        with (
            mock.patch.object(self.lta_mode, "_load_runtime_module", return_value=runtime),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            self.lta_mode.run(arguments)

        self.assertEqual(raised.exception.code, 3)
        self.assertIn("planning prototype", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
