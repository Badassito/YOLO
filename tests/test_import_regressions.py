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
        self.assertIn("eager package import graph is acyclic", completed.stdout)
        self.assertIn("all function globals resolved", completed.stdout)

    def test_interpolation_can_be_the_first_subsystem_imported(self) -> None:
        completed = self.run_python(
            str(ROOT / "tools" / "smoke_import.py"), "interpolation"
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("eager package import graph is acyclic", completed.stdout)
        self.assertIn("all function globals resolved", completed.stdout)

    def test_retina_processor_state_remains_backend_settable(self) -> None:
        program = textwrap.dedent(
            """
            from tools.smoke_import import install_stubs

            install_stubs()
            from volume_tta.inference import (
                cpu_retina_masks_enabled,
                set_retina_mask_processor,
            )

            set_retina_mask_processor("gpu")
            assert not cpu_retina_masks_enabled()
            set_retina_mask_processor("cpu")
            assert cpu_retina_masks_enabled()
            """
        )
        completed = self.run_python("-c", program)
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_automatic_retina_backend_selection_remains_wired(self) -> None:
        pipeline_source = (ROOT / "volume_tta" / "pipeline.py").read_text(encoding="utf-8")
        workers_source = (ROOT / "volume_tta" / "workers.py").read_text(encoding="utf-8")

        self.assertIn(
            "retina_processor = 'gpu' if gpu_worker_process_active else 'cpu'",
            pipeline_source,
        )
        self.assertIn("set_retina_mask_processor(retina_processor)", pipeline_source)
        self.assertIn("'retina_processor': str(retina_processor)", pipeline_source)
        self.assertIn("set_retina_mask_processor('cpu')", workers_source)
        self.assertIn(
            "set_retina_mask_processor(str(init_dict.get('retina_processor', 'cpu')))",
            workers_source,
        )

    def test_regression_imports_do_not_initialize_accelerator_runtimes(self) -> None:
        program = textwrap.dedent(
            """
            import importlib
            import sys

            from tools.smoke_import import install_stubs

            install_stubs()
            importlib.import_module("volume_tta." + sys.argv[1])
            forbidden = {
                "cupy", "jax", "openvino", "torch", "ultralytics",
                "qatzip", "qpl", "accel_config", "dto", "dml",
            }
            loaded = {name.split(".")[0] for name in sys.modules}
            assert not (forbidden & loaded), forbidden & loaded
            native_modules = {
                "volume_tta._qat_codec", "volume_tta._qpl_codec",
                "volume_tta._dsa_copy",
            }
            assert not (native_modules & set(sys.modules)), native_modules & set(sys.modules)
            """
        )
        for subsystem in ("topology", "interpolation", "outputs", "runtime"):
            with self.subTest(subsystem=subsystem):
                completed = self.run_python("-c", program, subsystem)
                self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_concurrent_first_imports_do_not_raise_importlib_deadlock(self) -> None:
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
            """
        )
        for pair in (("topology", "finalization"), ("interpolation", "cuda_d1")):
            with self.subTest(pair=pair):
                completed = self.run_python("-c", program, *pair)
                self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_transitional_import_shims_are_retired(self) -> None:
        package = ROOT / "volume_tta"
        self.assertFalse((package / "_latebind.py").exists())
        self.assertFalse((package / "_stdlib.py").exists())
        for path in package.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("_latebind", source, path.name)
            self.assertNotIn("_stdlib", source, path.name)
            self.assertNotIn("import *", source, path.name)


if __name__ == "__main__":
    unittest.main()
