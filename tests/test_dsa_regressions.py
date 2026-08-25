from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ['YOLO_TTA_TELEMETRY'] = '0'

from tools.smoke_import import install_stubs

install_stubs()

from volume_tta import intel_dsa, runtime

runtime.runtime_telemetry().enabled = False

ROOT = Path(__file__).resolve().parents[1]


class _NativeFailure(RuntimeError):
    def __init__(self, message: str, stats: dict[str, object]) -> None:
        super().__init__(message)
        self.stats = dict(stats)


class _FakeDsaModule:
    def __init__(self, *, fail: bool = False, drain_results: list[bool] | None = None) -> None:
        self.fail = bool(fail)
        self.drain_results = list(drain_results or [])
        self.capability_calls = 0
        self.copy_calls: list[dict[str, object]] = []
        self.events: list[str] = []
        self.close_calls = 0
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def capabilities(self, *, work_queue: str | None = None) -> dict[str, object]:
        self.capability_calls += 1
        return {
            'hardware_available': True,
            'interface': 'idxd-cdev',
            'software_fallback_enabled': False,
            'drain_guaranteed': True,
            'work_queue_type': 'user',
            'work_queue': work_queue or '/dev/dsa/wq0.0',
            'numa_local': True,
            'numa_node': 0,
            'max_transfer_size': 4096,
            'max_inflight': 8,
        }

    def copy(
        self,
        src: object,
        dst: object,
        *,
        work_queue: str,
        max_transfer_size: int,
        max_inflight: int,
        require_hardware: bool = True,
    ) -> dict[str, object]:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.events.append('submit')
            self.copy_calls.append({
                'work_queue': work_queue,
                'max_transfer_size': int(max_transfer_size),
                'max_inflight': int(max_inflight),
                'require_hardware': bool(require_hardware),
            })
            src_bytes = np.frombuffer(memoryview(np.asarray(src)).cast('B'), dtype=np.uint8)
            dst_bytes = np.frombuffer(memoryview(np.asarray(dst)).cast('B'), dtype=np.uint8)
            if self.fail:
                copied = max(1, int(src_bytes.size) // 2)
                dst_bytes[:copied] = src_bytes[:copied]
                self.events.append('partial_write')
                raise _NativeFailure(
                    'forced partial submission failure',
                    {
                        'submitted_descriptors': 1,
                        'hardware_bytes': copied,
                        'partial_failures': 1,
                    },
                )
            np.copyto(dst_bytes, src_bytes)
            self.events.append('complete')
            return {
                'drained': True,
                'hardware_only': True,
                'software_bytes': 0,
                'hardware_bytes': int(src_bytes.size),
                'descriptors': max(1, (int(src_bytes.size) + int(max_transfer_size) - 1) // int(max_transfer_size)),
                'batches': 1,
                'max_inflight': int(max_inflight),
            }
        finally:
            with self.lock:
                self.active -= 1

    def drain(self) -> dict[str, object]:
        self.events.append('drain')
        drained = self.drain_results.pop(0) if self.drain_results else True
        return {'drained': bool(drained)}

    def close(self) -> None:
        self.events.append('close')
        self.close_calls += 1


class DsaNativeSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / 'native' / 'dsa_copy.c').read_text(encoding='utf-8')

    def test_direct_idxd_cdev_abi_and_no_host_copy_fallback(self) -> None:
        self.assertIn('#include <linux/idxd.h>', self.source)
        self.assertIn('_Static_assert(sizeof(struct dsa_hw_desc) == 64', self.source)
        self.assertIn('descriptors[index].opcode = DSA_OPCODE_MEMMOVE', self.source)
        self.assertIn('IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR', self.source)
        self.assertIn('written = write(', self.source)
        self.assertNotIn('memcpy(', self.source)

    def test_completion_storage_lives_through_close_drain(self) -> None:
        core = self.source[self.source.index('vt_core_copy('):]
        self.assertLess(core.index('close(fd)'), core.index('free(descriptors)'))
        self.assertLess(core.index('close(fd)'), core.index('free(completions)'))
        self.assertIn('vt_wait_for_batch(descriptors, completions', core)
        self.assertIn('idxd-cdev-per-open-release', self.source)

    def test_native_lock_and_drain_state_are_pid_guarded(self) -> None:
        self.assertIn('static pid_t native_pid', self.source)
        self.assertIn('vt_ensure_process_state()', self.source)
        self.assertIn('native_pid = current', self.source)
        self.assertIn('last_drained = 1', self.source)
        self.assertIn('pthread_atfork(vt_atfork_prepare, vt_atfork_parent, vt_atfork_child)', self.source)
        self.assertIn('pthread_rwlock_wrlock(&fork_gate)', self.source)
        self.assertIn('pthread_rwlock_rdlock(&fork_gate)', self.source)

    def test_setup_uses_the_owned_optional_source(self) -> None:
        setup_source = (ROOT / 'setup.py').read_text(encoding='utf-8')
        self.assertIn('VOLUME_TTA_BUILD_INTEL', setup_source)
        self.assertIn('native/dsa_copy.c', setup_source)
        self.assertIn('volume_tta._dsa_copy', setup_source)
        self.assertIn('extra_link_args=["-pthread"]', setup_source)


class DsaWorkspaceCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        intel_dsa._reset_for_tests()
        runtime.runtime_telemetry().enabled = False

    def tearDown(self) -> None:
        intel_dsa._reset_for_tests()

    @staticmethod
    def _anonymous_allocator(*, shape, dtype, **_kwargs):
        return np.empty(tuple(shape), dtype=np.dtype(dtype))

    def _dsa_env(self, backend: str = 'auto') -> dict[str, str]:
        return {
            'YOLO_TTA_TELEMETRY': '0',
            'YOLO_TTA_WORKSPACE_COPY_BACKEND': str(backend),
            'YOLO_TTA_DSA_MIN_MIB': '0',
            'YOLO_TTA_DSA_MAX_INFLIGHT': '3',
            'YOLO_TTA_DSA_WQ': '/dev/dsa/wq7.1',
        }

    def test_cpu_default_never_probes_or_imports_native_backend(self) -> None:
        fake = _FakeDsaModule()
        intel_dsa._set_test_module(fake)
        src = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        with mock.patch.dict(os.environ, {'YOLO_TTA_TELEMETRY': '0'}, clear=True), \
                mock.patch.object(runtime, 'allocate_workspace_array', side_effect=self._anonymous_allocator):
            dst = runtime.copy_workspace_array(src, None, 'cpu default', workers=2)
        np.testing.assert_array_equal(dst, src)
        self.assertEqual(fake.capability_calls, 0)
        self.assertEqual(fake.copy_calls, [])

    def test_dsa_seam_covers_scalar_vector_and_multidimensional_cpu_branches(self) -> None:
        fake = _FakeDsaModule()
        intel_dsa._set_test_module(fake)
        sources = (
            np.asarray(17, dtype=np.int32),
            np.arange(11, dtype=np.int16),
            np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        )
        with mock.patch.dict(os.environ, self._dsa_env('dsa'), clear=True), \
                mock.patch.object(runtime, 'allocate_workspace_array', side_effect=self._anonymous_allocator):
            results = [runtime.copy_workspace_array(src, None, 'eligible') for src in sources]
        for result, src in zip(results, sources):
            np.testing.assert_array_equal(result, src)
        self.assertEqual(len(fake.copy_calls), 3)
        self.assertTrue(all(call['require_hardware'] for call in fake.copy_calls))
        self.assertTrue(all(call['max_inflight'] == 3 for call in fake.copy_calls))
        self.assertTrue(all(call['work_queue'] == '/dev/dsa/wq7.1' for call in fake.copy_calls))

    def test_auto_ineligible_copy_uses_exact_cpu_path_without_probe(self) -> None:
        fake = _FakeDsaModule()
        intel_dsa._set_test_module(fake)
        source = np.arange(24, dtype=np.int32).reshape(4, 6).T
        self.assertFalse(source.flags['C_CONTIGUOUS'])
        with mock.patch.dict(os.environ, self._dsa_env('auto'), clear=True), \
                mock.patch.object(runtime, 'allocate_workspace_array', side_effect=self._anonymous_allocator):
            result = runtime.copy_workspace_array(source, None, 'noncontiguous', workers=2)
        np.testing.assert_array_equal(result, source)
        self.assertEqual(fake.capability_calls, 0)
        self.assertEqual(fake.copy_calls, [])

    def test_small_auto_copy_does_not_probe_native_backend(self) -> None:
        fake = _FakeDsaModule()
        intel_dsa._set_test_module(fake)
        env = self._dsa_env('auto')
        env['YOLO_TTA_DSA_MIN_MIB'] = '1'
        src = np.arange(128, dtype=np.uint8)
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(runtime, 'allocate_workspace_array', side_effect=self._anonymous_allocator):
            result = runtime.copy_workspace_array(src, None, 'small')
        np.testing.assert_array_equal(result, src)
        self.assertEqual(fake.capability_calls, 0)

    def test_auto_unavailable_queue_is_an_initial_cpu_copy_not_a_recopy(self) -> None:
        fake = _FakeDsaModule()

        def unavailable(*, work_queue=None):
            fake.capability_calls += 1
            return {
                'hardware_available': False,
                'unavailable_reason': 'no local enabled queue',
            }

        fake.capabilities = unavailable  # type: ignore[method-assign]
        intel_dsa._set_test_module(fake)
        src = np.arange(128, dtype=np.uint8)
        with mock.patch.dict(os.environ, self._dsa_env('auto'), clear=True), \
                mock.patch.object(runtime, 'allocate_workspace_array', side_effect=self._anonymous_allocator), \
                mock.patch.object(
                    runtime,
                    '_copy_workspace_array_cpu',
                    wraps=runtime._copy_workspace_array_cpu,
                ) as cpu_copy:
            result = runtime.copy_workspace_array(src, None, 'unavailable')
        np.testing.assert_array_equal(result, src)
        self.assertEqual(fake.capability_calls, 1)
        self.assertEqual(fake.copy_calls, [])
        self.assertNotIn('drain', fake.events)
        cpu_copy.assert_called_once()

    def test_regular_file_memmap_is_outside_initial_rollout(self) -> None:
        fake = _FakeDsaModule()
        intel_dsa._set_test_module(fake)
        with tempfile.TemporaryDirectory() as temp_dir:
            src_path = Path(temp_dir) / 'source.bin'
            src = np.memmap(src_path, mode='w+', dtype=np.uint8, shape=(256,))
            src[:] = np.arange(256, dtype=np.uint8)
            with mock.patch.dict(os.environ, self._dsa_env('auto'), clear=True), \
                    mock.patch.object(runtime, 'allocate_workspace_array', side_effect=self._anonymous_allocator):
                result = runtime.copy_workspace_array(src, None, 'regular memmap')
            np.testing.assert_array_equal(result, src)
            self.assertEqual(fake.capability_calls, 0)
            with mock.patch.dict(os.environ, self._dsa_env('dsa'), clear=True), \
                    mock.patch.object(runtime, 'allocate_workspace_array', side_effect=self._anonymous_allocator):
                with self.assertRaisesRegex(intel_dsa.IntelDsaIneligible, 'regular_memmap'):
                    runtime.copy_workspace_array(src, None, 'regular memmap')
            runtime.close_memmap_array_without_flush(src)

    def test_overlap_is_rejected_before_native_probe(self) -> None:
        fake = _FakeDsaModule()
        intel_dsa._set_test_module(fake)
        src = np.arange(32, dtype=np.uint8)
        eligibility = intel_dsa.assess_copy_eligibility(src, src, minimum_bytes=0)
        self.assertFalse(eligibility.eligible)
        self.assertIn('overlapping_ranges', eligibility.reasons)

    def test_auto_partial_failure_drains_before_full_cpu_recopy(self) -> None:
        fake = _FakeDsaModule(fail=True, drain_results=[True])
        intel_dsa._set_test_module(fake)
        src = np.arange(101, dtype=np.uint8)
        real_cpu_copy = runtime._copy_workspace_array_cpu

        def checked_cpu_copy(dst, source, **kwargs):
            fake.events.append('cpu_recopy')
            self.assertIn('drain', fake.events)
            self.assertLess(fake.events.index('drain'), fake.events.index('cpu_recopy'))
            return real_cpu_copy(dst, source, **kwargs)

        with mock.patch.dict(os.environ, self._dsa_env('auto'), clear=True), \
                mock.patch.object(runtime, 'allocate_workspace_array', side_effect=self._anonymous_allocator), \
                mock.patch.object(runtime, '_copy_workspace_array_cpu', side_effect=checked_cpu_copy):
            result = runtime.copy_workspace_array(src, None, 'recover')
        np.testing.assert_array_equal(result, src)
        self.assertEqual(fake.events[:4], ['submit', 'partial_write', 'drain', 'cpu_recopy'])

    def test_undrained_auto_failure_quarantines_dst_until_later_drain(self) -> None:
        fake = _FakeDsaModule(fail=True, drain_results=[False, True])
        intel_dsa._set_test_module(fake)
        src = np.arange(64, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            dst_path = Path(temp_dir) / 'owned.bin'

            def mapped_allocator(*, shape, dtype, path, **_kwargs):
                return np.memmap(path, mode='w+', dtype=dtype, shape=shape)

            # Mark the synthetic mapping as tmpfs-eligible so this test reaches submission;
            # ownership/unlink behavior is independent of its real temporary filesystem.
            with mock.patch.dict(os.environ, self._dsa_env('auto'), clear=True), \
                    mock.patch.object(runtime, 'allocate_workspace_array', side_effect=mapped_allocator), \
                    mock.patch.object(intel_dsa, 'classify_array_backing', return_value='tmpfs'), \
                    mock.patch.object(runtime, '_copy_workspace_array_cpu') as cpu_copy:
                with self.assertRaises(intel_dsa.IntelDsaCopyError) as caught:
                    runtime.copy_workspace_array(src, dst_path, 'undrained')
            self.assertFalse(caught.exception.drained)
            cpu_copy.assert_not_called()
            self.assertTrue(dst_path.exists())
            self.assertIn('partial_write', fake.events)
            with self.assertRaisesRegex(intel_dsa.IntelDsaUnavailable, 'quarantined'):
                intel_dsa.get_manager().capabilities()
            # A later runtime boundary proves drain, then executes the deferred
            # close/unlink while the quarantined mapping is still alive.
            intel_dsa.close_manager()
            self.assertFalse(dst_path.exists())

    def test_explicit_dsa_failure_never_recovers_on_cpu(self) -> None:
        fake = _FakeDsaModule(fail=True, drain_results=[True])
        intel_dsa._set_test_module(fake)
        src = np.arange(64, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            dst_path = Path(temp_dir) / 'explicit-owned.bin'

            def mapped_allocator(*, shape, dtype, path, **_kwargs):
                return np.memmap(path, mode='w+', dtype=dtype, shape=shape)

            with mock.patch.dict(os.environ, self._dsa_env('dsa'), clear=True), \
                    mock.patch.object(runtime, 'allocate_workspace_array', side_effect=mapped_allocator), \
                    mock.patch.object(intel_dsa, 'classify_array_backing', return_value='tmpfs'), \
                    mock.patch.object(runtime, '_copy_workspace_array_cpu') as cpu_copy:
                with self.assertRaises(intel_dsa.IntelDsaCopyError) as caught:
                    runtime.copy_workspace_array(src, dst_path, 'explicit')
            self.assertTrue(caught.exception.drained)
            cpu_copy.assert_not_called()
            self.assertFalse(dst_path.exists())

    def test_manager_serializes_native_requests_and_runtime_reset_closes_it(self) -> None:
        fake = _FakeDsaModule()
        original_copy = fake.copy

        def slow_copy(*args, **kwargs):
            time.sleep(0.01)
            return original_copy(*args, **kwargs)

        fake.copy = slow_copy  # type: ignore[method-assign]
        intel_dsa._set_test_module(fake)
        manager = intel_dsa.get_manager()
        capabilities = manager.capabilities()
        sources = [np.arange(256, dtype=np.uint8) for _ in range(3)]
        destinations = [np.empty_like(source) for source in sources]
        errors: list[BaseException] = []

        def run(index: int) -> None:
            try:
                manager.copy(
                    sources[index],
                    destinations[index],
                    capabilities=capabilities,
                    max_inflight=2,
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(index,)) for index in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(fake.max_active, 1)
        runtime.reset_runtime_state_for_new_run()
        self.assertEqual(fake.close_calls, 1)

    def test_invalid_policy_values_fail_before_destination_allocation(self) -> None:
        src = np.arange(4, dtype=np.uint8)
        with mock.patch.dict(
            os.environ,
            {'YOLO_TTA_TELEMETRY': '0', 'YOLO_TTA_WORKSPACE_COPY_BACKEND': 'magic'},
            clear=True,
        ), mock.patch.object(runtime, 'allocate_workspace_array') as allocate:
            with self.assertRaisesRegex(ValueError, 'cpu, auto, or dsa'):
                runtime.copy_workspace_array(src, None, 'invalid')
        allocate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
