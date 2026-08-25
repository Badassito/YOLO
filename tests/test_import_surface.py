from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "GPT-5.6-Sol-Ultra_v17.1.4_SLURM.py"


class ImportSurfaceTests(unittest.TestCase):
    def run_python(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env={**os.environ, 'YOLO_TTA_TELEMETRY': '0'},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_config_import_does_not_load_native_or_accelerator_runtimes(self) -> None:
        program = (
            "import sys; import volume_tta.config; "
            "forbidden={'cv2','scipy','torch','cupy','openvino','ultralytics','jax'}; "
            "loaded={name.split('.')[0] for name in sys.modules}; "
            "assert not (forbidden & loaded), forbidden & loaded"
        )
        completed = self.run_python("-c", program)
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_compatibility_launcher_help_is_dependency_light(self) -> None:
        completed = self.run_python(str(WRAPPER), "--help")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("YOLO segmentation TTA", completed.stdout)

    def test_module_version(self) -> None:
        completed = self.run_python("-m", "volume_tta", "--version")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("17.1.4", completed.stdout)

    def test_cycle_safe_full_import_smoke(self) -> None:
        completed = self.run_python(str(ROOT / "tools" / "smoke_import.py"), "pipeline")
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("eager package import graph is acyclic", completed.stdout)
        self.assertIn("all function globals resolved", completed.stdout)
        for module in (
            "intel_compression",
            "intel_dsa",
            "inference_backends",
            "inference_backends.contracts",
            "inference_backends.descriptors",
            "inference_backends.registry",
        ):
            with self.subTest(module=module):
                self.assertIn(f"imported volume_tta.{module}", completed.stdout)

    def test_package_statement_inventory(self) -> None:
        completed = self.run_python(str(ROOT / "tools" / "verify_package_inventory.py"))
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("package inventory verified", completed.stdout)

    def test_no_mutable_global_is_copied_across_subsystems(self) -> None:
        completed = self.run_python(str(ROOT / "tools" / "analyze_package_state.py"))
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertEqual(completed.stdout.strip(), "")

    def test_pipeline_enters_main_through_packaged_dependencies(self) -> None:
        completed = self.run_python(str(ROOT / "tools" / "smoke_main.py"))
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("expected missing-input guard", completed.stdout)


if __name__ == "__main__":
    unittest.main()
