from __future__ import annotations

import gzip
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "qpl_codec.c"


class QplNativeSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_extension_is_hardware_path_only(self) -> None:
        self.assertIn("qpl_get_job_size(qpl_path_hardware", self.source)
        self.assertIn("qpl_init_job(qpl_path_hardware", self.source)
        self.assertIn("job->data_ptr.path != qpl_path_hardware", self.source)
        self.assertNotIn("qpl_get_job_size(qpl_path_auto", self.source)
        self.assertNotIn("qpl_init_job(qpl_path_auto", self.source)
        self.assertNotIn("qpl_get_job_size(qpl_path_software", self.source)
        self.assertNotIn("qpl_init_job(qpl_path_software", self.source)

    def test_each_descriptor_is_a_complete_standard_gzip_member(self) -> None:
        required = (
            "QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_GZIP_MODE",
            "QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY",
            "qpl_get_safe_deflate_compression_buffer_size",
            "VT_QPL_GZIP_OVERHEAD 18U",
            "vt_gzip_member_shape_is_valid",
            "state->job->available_in != 0U",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)

    def test_work_queue_transfer_and_operation_eligibility_are_enforced(self) -> None:
        required = (
            "accfg_wq_get_max_transfer_size",
            "accfg_wq_get_op_config",
            "VT_IAA_COMPRESS_OPCODE 0x43U",
            "max_member_input_bytes",
            "open(path, O_RDWR | O_CLOEXEC)",
            "capacity > transfer_limit",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)

    def test_thread_state_cleanup_gil_release_and_fork_guards_are_present(self) -> None:
        required = (
            "pthread_key_create",
            "pthread_setspecific",
            "qpl_fini_job",
            "Py_BEGIN_ALLOW_THREADS",
            "pthread_atfork",
            "pthread_rwlock_rdlock",
            "pthread_rwlock_wrlock",
            "state->pid != getpid()",
            "g_qpl_inherited_after_fork = 1",
            "inventory_ok = !g_qpl_inherited_after_fork",
            "start accelerator workers with spawn or exec",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.source)

    def test_python_adapter_contract_and_supported_level_are_exported(self) -> None:
        for method in (
            '"capabilities"',
            '"compress_gzip"',
            '"preflight_thread_state"',
            '"stats"',
            '"close_thread_state"',
        ):
            with self.subTest(method=method):
                self.assertIn(method, self.source)
        self.assertIn("supported_levels=(1,)", self.source)
        self.assertIn('PyUnicode_FromString("1.9.0")', self.source)
        self.assertIn('"software_fallback_requests", (unsigned long long)0U', self.source)


@unittest.skipUnless(
    sys.platform.startswith("linux")
    and os.environ.get("VOLUME_TTA_TEST_IAA_HARDWARE", "").strip() == "1",
    "set VOLUME_TTA_TEST_IAA_HARDWARE=1 on a configured IAA host",
)
class QplHardwareSmokeTests(unittest.TestCase):
    def test_hardware_only_gzip_round_trip_and_counters(self) -> None:
        from volume_tta import _qpl_codec

        capabilities = dict(_qpl_codec.capabilities())
        self.assertTrue(capabilities.get("hardware_available"), capabilities)
        self.assertEqual(capabilities.get("execution_path"), "qpl_path_hardware")
        self.assertFalse(capabilities.get("software_fallback_enabled"))
        self.assertEqual(tuple(capabilities.get("supported_levels", ())), (1,))
        self.assertGreater(int(capabilities.get("work_queue_count", 0)), 0)

        seed = bytes(range(251)) + b"volume-tta-qpl-hardware-smoke\x00"
        payload_size = 3 * 1024 * 1024 + 257
        payload = (seed * ((payload_size + len(seed) - 1) // len(seed)))[:payload_size]
        _qpl_codec.stats(reset=True)
        try:
            _qpl_codec.preflight_thread_state(
                1, require_hardware=True, numa_id=None,
            )
            encoded = _qpl_codec.compress_gzip(
                memoryview(payload), 1, require_hardware=True, numa_id=None,
            )
            self.assertEqual(gzip.decompress(encoded), payload)
            counters = dict(_qpl_codec.stats())
            self.assertGreaterEqual(int(counters.get("hardware_requests", 0)), 1)
            self.assertGreaterEqual(int(counters.get("physical_members", 0)), 1)
            self.assertEqual(int(counters.get("software_fallback_requests", -1)), 0)
            self.assertEqual(int(counters.get("failures", -1)), 0)
        finally:
            _qpl_codec.close_thread_state()

    def test_non_hardware_and_high_level_requests_fail_closed(self) -> None:
        from volume_tta import _qpl_codec

        with self.assertRaises(ValueError):
            _qpl_codec.compress_gzip(b"hardware only", 1, require_hardware=False)
        with self.assertRaises(ValueError):
            _qpl_codec.compress_gzip(b"level one only", 3, require_hardware=True)


if __name__ == "__main__":
    unittest.main()
