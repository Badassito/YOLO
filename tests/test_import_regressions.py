from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ImportRegressionTests(unittest.TestCase):
    def run_python(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20,
        )

    def test_topology_can_be_the_first_subsystem_imported(self) -> None:
        completed = self.run_python(str(ROOT / "tools" / "smoke_import.py"), "topology")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("all callable-only dependencies resolved", completed.stdout)
        self.assertIn("all function globals resolved", completed.stdout)

    def test_interpolation_can_be_the_first_subsystem_imported(self) -> None:
        completed = self.run_python(
            str(ROOT / "tools" / "smoke_import.py"), "interpolation"
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("all callable-only dependencies resolved", completed.stdout)
        self.assertIn("all function globals resolved", completed.stdout)

    def test_regression_imports_do_not_initialize_accelerator_runtimes(self) -> None:
        program = textwrap.dedent(
            """
            import importlib
            import sys

            from tools.smoke_import import install_stubs

            install_stubs()
            importlib.import_module("volume_tta." + sys.argv[1])
            forbidden = {"cupy", "jax", "openvino", "torch", "ultralytics"}
            loaded = {name.split(".")[0] for name in sys.modules}
            assert not (forbidden & loaded), forbidden & loaded
            """
        )
        for subsystem in ("topology", "interpolation"):
            with self.subTest(subsystem=subsystem):
                completed = self.run_python("-c", program, subsystem)
                self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_concurrent_cycle_imports_do_not_raise_importlib_deadlock(self) -> None:
        program = textwrap.dedent(
            """
            import importlib
            import sys
            import threading
            import traceback

            from tools.smoke_import import install_stubs

            install_stubs()
            barrier = threading.Barrier(3)
            errors = []

            def load(name):
                try:
                    barrier.wait()
                    importlib.import_module("volume_tta." + name)
                except BaseException:
                    errors.append(traceback.format_exc())

            threads = [
                threading.Thread(target=load, args=(name,), daemon=True)
                for name in sys.argv[1:3]
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(5)

            assert not any(thread.is_alive() for thread in threads), "import threads hung"
            assert not errors, "\\n".join(errors)

            from volume_tta._latebind import unresolved_bindings

            assert not unresolved_bindings(), unresolved_bindings()
            """
        )
        for pair in (("topology", "finalization"), ("interpolation", "cuda_d1")):
            with self.subTest(pair=pair):
                completed = self.run_python("-c", program, *pair)
                self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_registration_during_resolution_is_not_stranded(self) -> None:
        program = textwrap.dedent(
            """
            import sys
            import threading
            import types

            import volume_tta._latebind as latebind

            first_provider = types.ModuleType("race.first_provider")
            second_provider = types.ModuleType("race.second_provider")
            sentinel = object()
            second_provider.second_symbol = sentinel
            sys.modules[first_provider.__name__] = first_provider
            sys.modules[second_provider.__name__] = second_provider

            entered = threading.Event()
            release = threading.Event()
            original_import_module = latebind.importlib.import_module

            def controlled_import(name):
                if name == first_provider.__name__ and not entered.is_set():
                    entered.set()
                    assert release.wait(5), "resolver release timed out"
                return sys.modules.get(name) or original_import_module(name)

            latebind.importlib.import_module = controlled_import
            first_namespace = {}
            second_namespace = {}
            resolver = threading.Thread(
                target=latebind.bind_late_symbols,
                args=(
                    "race.first_consumer",
                    first_namespace,
                    {"first_provider": ("missing_symbol",)},
                ),
                daemon=True,
            )
            resolver.start()
            assert entered.wait(5), "resolver did not enter controlled import"

            latebind.bind_late_symbols(
                "race.second_consumer",
                second_namespace,
                {"second_provider": ("second_symbol",)},
            )
            release.set()
            resolver.join(5)

            assert not resolver.is_alive(), "resolver thread hung"
            assert second_namespace.get("second_symbol") is sentinel, second_namespace
            pending = latebind.unresolved_bindings()
            assert "race.second_consumer" not in pending, pending
            """
        )
        completed = self.run_python("-c", program)
        self.assertEqual(completed.returncode, 0, completed.stdout)


if __name__ == "__main__":
    unittest.main()
