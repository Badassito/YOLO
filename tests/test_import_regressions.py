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
            from XTA.inference import (
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
        pipeline_source = (ROOT / "XTA" / "pipeline.py").read_text(encoding="utf-8")
        workers_source = (ROOT / "XTA" / "workers.py").read_text(encoding="utf-8")

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
            importlib.import_module("XTA." + sys.argv[1])
            forbidden = {
                "cupy", "jax", "openvino", "torch", "ultralytics",
                "qatzip", "qpl", "accel_config", "dto", "dml",
            }
            loaded = {name.split(".")[0] for name in sys.modules}
            assert not (forbidden & loaded), forbidden & loaded
            native_modules = {
                "XTA._qat_codec", "XTA._qpl_codec",
                "XTA._dsa_copy",
            }
            assert not (native_modules & set(sys.modules)), native_modules & set(sys.modules)
            """
        )
        for subsystem in (
            "topology",
            "interpolation",
            "outputs",
            "runtime",
            "intel_compression",
            "intel_dsa",
            "inference_backends",
        ):
            with self.subTest(subsystem=subsystem):
                completed = self.run_python("-c", program, subsystem)
                self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_dependency_stubs_preserve_module_metadata_semantics(self) -> None:
        program = textwrap.dedent(
            """
            import sys

            from tools.smoke_import import install_stubs

            install_stubs()
            for name in ("cv2", "scipy", "scipy.ndimage", "tifffile", "tqdm"):
                module = sys.modules[name]
                assert module.__file__ is None or isinstance(module.__file__, str)
                try:
                    getattr(module, "__missing_stub_metadata__")
                except AttributeError:
                    pass
                else:
                    raise AssertionError(f"{name} fabricated missing dunder metadata")
            """
        )
        completed = self.run_python("-c", program)
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_split_feature_telemetry_reports_packaged_capabilities(self) -> None:
        program = textwrap.dedent(
            """
            import sys

            from tools.smoke_import import install_stubs

            install_stubs()
            from XTA import runtime

            owner_names = {
                "XTA.assembly",
                "XTA.backprojection",
                "XTA.interpolation",
                "XTA.outputs",
            }
            assert not (owner_names & set(sys.modules)), owner_names & set(sys.modules)

            class Capture:
                def __init__(self):
                    self.gauges = {}

                def gauge(self, name, value):
                    self.gauges[name] = value

            capture = Capture()
            runtime._record_runtime_feature_gauges(capture)
            features = capture.gauges["features"]
            expected = {
                "raw_bbox_restored_sparse_members",
                "crop_aware_low_quality_mirror",
                "owned_nrrd_member_transfer",
                "native_projection_callback",
                "native_projected_layer_materializer",
                "native_persistent_trt_ring",
            }
            assert all(features[name] is True for name in expected), features
            assert not (owner_names & set(sys.modules)), owner_names & set(sys.modules)

            from XTA import assembly, backprojection, interpolation, outputs

            assert hasattr(interpolation.RawBBoxMaskStore, "iter_restored_sparse_members")
            assert callable(outputs._resize_sparse_binary_crop_to_output_region)
            assert hasattr(
                outputs._MemberParallelGzipPayloadWriter,
                "write_owned_known_nonzero",
            )
            assert callable(backprojection._emit_projection_block_callback)
            assert callable(assembly.materialize_nrrd_view_layer)
            assert backprojection._RESIDENT_TRT_PIPELINE_CACHE_NATIVE is True
            """
        )
        completed = self.run_python("-c", program)
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
                    importlib.import_module("XTA." + name)
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
        package = ROOT / "XTA"
        self.assertFalse((package / "_latebind.py").exists())
        self.assertFalse((package / "_stdlib.py").exists())
        for path in package.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("_latebind", source, path.name)
            self.assertNotIn("_stdlib", source, path.name)
            self.assertNotIn("import *", source, path.name)


if __name__ == "__main__":
    unittest.main()
