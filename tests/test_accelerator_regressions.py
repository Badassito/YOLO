from __future__ import annotations

import contextlib
import gzip
import os
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ['YOLO_TTA_TELEMETRY'] = '0'

from tests.test_output_regressions import outputs
from volume_tta import intel_compression

# Several policy tests deliberately replace the entire environment. Initialize the
# process-local singleton while telemetry is disabled so those temporary clear=True
# contexts cannot make a targeted unit test write into the repository at interpreter exit.
outputs.runtime_telemetry().enabled = False


class _FakeHardwareCodec:
    def __init__(
        self,
        *,
        backend: str = 'qat',
        minimum_input_bytes: int = 1,
        max_concurrency: int = 2,
        corrupt: bool = False,
    ) -> None:
        self.backend = str(backend)
        self.minimum_input_bytes = int(minimum_input_bytes)
        self.max_concurrency = int(max_concurrency)
        self.corrupt = bool(corrupt)
        self.capability_calls = 0
        self.compress_calls: list[dict[str, object]] = []
        self.preflight_threads: set[int] = set()
        self.closed_threads: set[int] = set()
        self.lock = threading.Lock()

    def capabilities(self) -> dict[str, object]:
        self.capability_calls += 1
        return {
            'hardware_available': True,
            'standard_gzip': True,
            'software_fallback_enabled': False,
            'hardware_generation': '2.x',
            'binding_version': 'fake-1',
            'instance_count': self.max_concurrency,
            'max_concurrency': self.max_concurrency,
            'minimum_input_bytes': self.minimum_input_bytes,
            'supported_levels': (1, 3),
        }

    def compress_gzip(
        self,
        buffer: object,
        level: int,
        *,
        require_hardware: bool = True,
        numa_id: int | None = None,
    ) -> bytes:
        payload = bytes(memoryview(buffer).cast('B'))  # type: ignore[arg-type]
        with self.lock:
            self.compress_calls.append({
                'payload': payload,
                'level': int(level),
                'require_hardware': bool(require_hardware),
                'numa_id': numa_id,
                'thread': threading.get_ident(),
            })
        if self.corrupt:
            return b'not-gzip'
        # Deliberately return two concatenated members: QATzip permits this shape.
        split = max(1, len(payload) // 2)
        return gzip.compress(payload[:split], mtime=0) + gzip.compress(payload[split:], mtime=0)

    def preflight_thread_state(
        self,
        level: int,
        *,
        require_hardware: bool = True,
        numa_id: int | None = None,
    ) -> None:
        self.assert_hardware_request(level, require_hardware)
        with self.lock:
            self.preflight_threads.add(threading.get_ident())

    def assert_hardware_request(self, level: int, require_hardware: bool) -> None:
        if not require_hardware or int(level) <= 0:
            raise RuntimeError('fake received a non-hardware request')

    def stats(self, *, reset: bool = False) -> dict[str, object]:
        result = {
            'hardware_requests': len(self.compress_calls),
            'software_fallback_members': 0,
        }
        if reset:
            self.compress_calls.clear()
        return result

    def close_thread_state(self) -> None:
        with self.lock:
            self.closed_threads.add(threading.get_ident())


class IntelCompressionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        outputs.shutdown_nrrd_gzip_executors()
        intel_compression._reset_for_tests()

    def tearDown(self) -> None:
        outputs.shutdown_nrrd_gzip_executors()
        intel_compression._reset_for_tests()

    def test_auto_prefers_qat_and_cpu_is_the_opt_out(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                outputs._nrrd_member_codec_candidates(),
                ('qat', 'libdeflate', 'isal', 'zlib'),
            )
        with mock.patch.dict(
            os.environ, {'YOLO_TTA_NRRD_MEMBER_CODEC': 'cpu'}, clear=True,
        ):
            self.assertEqual(
                outputs._nrrd_member_codec_candidates(),
                ('libdeflate', 'isal', 'zlib'),
            )
        with mock.patch.dict(
            os.environ, {'YOLO_TTA_NRRD_MEMBER_CODEC': 'iaa'}, clear=True,
        ):
            self.assertEqual(outputs._nrrd_member_codec_candidates(), ('iaa',))

    def test_cpu_policy_never_probes_qat(self) -> None:
        fake = _FakeHardwareCodec()
        intel_compression._set_test_module('qat', fake)
        zlib_spec = outputs._nrrd_member_codec_spec('zlib')
        with mock.patch.dict(
            os.environ, {'YOLO_TTA_NRRD_MEMBER_CODEC': 'cpu'}, clear=True,
        ), mock.patch.object(outputs, '_nrrd_member_codec_spec', return_value=zlib_spec) as resolve:
            selected = outputs._require_nrrd_member_codec()
            self.assertEqual(selected[0], 'zlib')
            resolve.assert_called_once_with('libdeflate')
            self.assertEqual(fake.capability_calls, 0)
            self.assertEqual(fake.compress_calls, [])

    def test_explicit_qat_failure_does_not_try_a_cpu_codec(self) -> None:
        attempted: list[str] = []

        def fail(name: str):
            attempted.append(str(name))
            raise RuntimeError('forced QAT failure')

        with mock.patch.dict(
            os.environ, {'YOLO_TTA_NRRD_MEMBER_CODEC': 'qat'}, clear=True,
        ), mock.patch.object(outputs, '_nrrd_member_codec_spec', side_effect=fail):
            self.assertIsNone(outputs._select_nrrd_member_codec())
        self.assertEqual(attempted, ['qat'])

    def test_valid_qat_runs_kat_on_every_bounded_worker(self) -> None:
        fake = _FakeHardwareCodec(max_concurrency=2)
        intel_compression._set_test_module('qat', fake)
        with mock.patch.dict(
            os.environ,
            {
                'YOLO_TTA_NRRD_MEMBER_CODEC': 'qat',
                'YOLO_TTA_NRRD_QAT_WORKERS': '9',
                'YOLO_TTA_NRRD_GZIP_LEVEL': '3',
            },
            clear=True,
        ):
            spec = outputs._require_nrrd_member_codec(expected_input_bytes=4096)
        self.assertEqual(spec[:2], ('qat', 1))
        self.assertEqual(len(fake.preflight_threads), 2)
        self.assertTrue(fake.compress_calls)
        self.assertTrue(all(call['require_hardware'] for call in fake.compress_calls))
        self.assertTrue(all(call['level'] == 1 for call in fake.compress_calls))
        compression_threads = {int(call['thread']) for call in fake.compress_calls}
        self.assertTrue(fake.preflight_threads <= compression_threads)
        outputs.shutdown_nrrd_gzip_executors()
        self.assertEqual(fake.closed_threads, fake.preflight_threads)

    def test_native_preflight_hook_is_followed_by_hardware_round_trip(self) -> None:
        fake = _FakeHardwareCodec(max_concurrency=1)
        intel_compression._set_test_module('qat', fake)
        capabilities = intel_compression.probe_capabilities('qat')
        compressor = intel_compression.create_gzip_compressor(
            'qat', 1, capabilities=capabilities,
        )
        compressor.preflight_thread_state()
        current_thread = threading.get_ident()
        self.assertEqual(fake.preflight_threads, {current_thread})
        self.assertEqual(len(fake.compress_calls), 1)
        self.assertEqual(fake.compress_calls[0]['thread'], current_thread)
        self.assertGreaterEqual(len(fake.compress_calls[0]['payload']), 128 * 1024)

    def test_qat_kat_scales_to_reported_hardware_minimum(self) -> None:
        minimum_input = 64 * 1024
        fake = _FakeHardwareCodec(
            minimum_input_bytes=int(minimum_input), max_concurrency=1,
        )
        intel_compression._set_test_module('qat', fake)
        with mock.patch.dict(
            os.environ,
            {
                'YOLO_TTA_NRRD_MEMBER_CODEC': 'qat',
                'YOLO_TTA_NRRD_GZIP_LEVEL': '3',
            },
            clear=True,
        ):
            spec = outputs._require_nrrd_member_codec(
                expected_input_bytes=1024 * 1024,
            )
        self.assertEqual(spec[:2], ('qat', 1))
        self.assertTrue(fake.compress_calls)
        self.assertTrue(
            all(len(call['payload']) >= int(minimum_input) for call in fake.compress_calls)
        )

    def test_corrupt_qat_falls_through_in_auto_but_forced_qat_fails(self) -> None:
        fake = _FakeHardwareCodec(corrupt=True)
        intel_compression._set_test_module('qat', fake)
        zlib_spec = outputs._nrrd_member_codec_spec('zlib')

        # Preserve the real function without relying on mock's synthetic __wrapped__.
        real_spec = outputs._nrrd_member_codec_spec

        def resolve(name: str):
            return real_spec(name) if str(name) in {'qat', 'zlib'} else (_ for _ in ()).throw(ModuleNotFoundError(name))

        with mock.patch.dict(os.environ, {'YOLO_TTA_NRRD_MEMBER_CODEC': 'auto'}, clear=True), \
                mock.patch.object(outputs, '_nrrd_member_codec_spec', side_effect=resolve):
            selected = outputs._select_nrrd_member_codec()
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 'zlib')  # type: ignore[index]
        with mock.patch.dict(os.environ, {'YOLO_TTA_NRRD_MEMBER_CODEC': 'qat'}, clear=True), \
                mock.patch.object(outputs, '_nrrd_member_codec_spec', side_effect=resolve):
            self.assertIsNone(outputs._select_nrrd_member_codec())

    def test_qat_software_backup_setting_is_rejected(self) -> None:
        fake = _FakeHardwareCodec()
        intel_compression._set_test_module('qat', fake)
        with mock.patch.dict(
            os.environ,
            {
                'YOLO_TTA_NRRD_MEMBER_CODEC': 'qat',
                'YOLO_TTA_NRRD_QAT_SW_FALLBACK': '1',
            },
            clear=True,
        ):
            self.assertIsNone(outputs._select_nrrd_member_codec())
        self.assertEqual(fake.capability_calls, 0)

    def test_unavailable_or_software_fallback_capability_is_rejected(self) -> None:
        no_hardware = _FakeHardwareCodec()
        no_hardware.capabilities = lambda: {  # type: ignore[method-assign]
            'hardware_available': False,
            'unavailable_reason': 'no assigned instance',
        }
        intel_compression._set_test_module('qat', no_hardware)
        with self.assertRaisesRegex(
            intel_compression.IntelCompressionUnavailable, 'no assigned instance',
        ):
            intel_compression.probe_capabilities('qat')

        software = _FakeHardwareCodec()
        original_capabilities = software.capabilities
        software.capabilities = lambda: {  # type: ignore[method-assign]
            **original_capabilities(), 'software_fallback_enabled': True,
        }
        intel_compression._set_test_module('qat', software)
        with self.assertRaisesRegex(
            intel_compression.IntelCompressionUnavailable, 'software fallback enabled',
        ):
            intel_compression.probe_capabilities('qat')

    def test_explicit_kat_failure_preserves_native_status_context(self) -> None:
        fake = _FakeHardwareCodec()

        def fail_with_status(*_args: object, **_kwargs: object) -> bytes:
            raise RuntimeError('QAT status=QZ_FAIL device=0000:6b:00.0')

        fake.compress_gzip = fail_with_status  # type: ignore[method-assign]
        intel_compression._set_test_module('qat', fake)
        sink = outputs.io.StringIO()
        with mock.patch.dict(
            os.environ, {'YOLO_TTA_NRRD_MEMBER_CODEC': 'qat'}, clear=True,
        ), contextlib.redirect_stdout(sink):
            self.assertIsNone(outputs._select_nrrd_member_codec())
        self.assertIn('QAT status=QZ_FAIL device=0000:6b:00.0', sink.getvalue())

    def test_qat_level_must_be_reported_supported(self) -> None:
        fake = _FakeHardwareCodec()
        original_capabilities = fake.capabilities
        fake.capabilities = lambda: {  # type: ignore[method-assign]
            **original_capabilities(), 'supported_levels': (1,),
        }
        intel_compression._set_test_module('qat', fake)
        with mock.patch.dict(
            os.environ,
            {'YOLO_TTA_NRRD_QAT_LEVEL': '5', 'YOLO_TTA_NRRD_GZIP_LEVEL': '3'},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                outputs._nrrd_member_codec_spec('qat')
        with mock.patch.dict(
            os.environ, {'YOLO_TTA_NRRD_GZIP_LEVEL': '6'}, clear=True,
        ):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                outputs._nrrd_member_codec_spec('qat')


class CompressionLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        outputs.shutdown_nrrd_gzip_executors()
        intel_compression._reset_for_tests()

    def tearDown(self) -> None:
        outputs.shutdown_nrrd_gzip_executors()
        intel_compression._reset_for_tests()

    def test_intel_fork_child_replaces_lock_and_discards_module_state(self) -> None:
        fake = _FakeHardwareCodec()
        intel_compression._set_test_module('qat', fake)
        with intel_compression._MODULE_LOCK:
            intel_compression._IMPORT_ERRORS['iaa'] = ('parent import failure', True)
        old_lock = intel_compression._MODULE_LOCK

        intel_compression._after_fork_child()

        self.assertIsNot(intel_compression._MODULE_LOCK, old_lock)
        self.assertEqual(intel_compression.loaded_backend_names(), ())
        self.assertEqual(intel_compression._IMPORT_ERRORS, {})

    def test_outputs_fork_child_discards_pool_without_shutdown(self) -> None:
        class InheritedExecutor:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def shutdown(self, **_kwargs: object) -> None:
                self.shutdown_calls += 1

        inherited = InheritedExecutor()
        old_executor_lock = outputs._NRRD_GZIP_EXECUTOR_LOCK
        old_zero_lock = outputs._NRRD_ZERO_MEMBER_LOCK
        old_test_lock = outputs._NRRD_MEMBER_GZIP_TEST_LOCK
        old_announce_lock = outputs._NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK
        outputs._NRRD_GZIP_EXECUTOR = inherited
        outputs._NRRD_GZIP_EXECUTORS[('qat', 1)] = inherited
        outputs._NRRD_ZERO_MEMBER_CACHE[8] = b'parent member'
        outputs._NRRD_MEMBER_CODEC_SETTING_WARNED = True
        outputs._NRRD_MEMBER_GZIP_OK[('parent', 1)] = True
        outputs._NRRD_MEMBER_GZIP_FAILURE_REASONS[('parent', 1)] = 'failure'
        outputs._NRRD_MEMBER_GZIP_ANNOUNCED = True
        outputs._NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED.add('qat')
        outputs._NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = True

        outputs._after_nrrd_compression_fork_child()

        self.assertEqual(inherited.shutdown_calls, 0)
        self.assertIsNone(outputs._NRRD_GZIP_EXECUTOR)
        self.assertEqual(outputs._NRRD_GZIP_EXECUTORS, {})
        self.assertEqual(outputs._NRRD_ZERO_MEMBER_CACHE, {})
        self.assertEqual(outputs._NRRD_MEMBER_GZIP_OK, {})
        self.assertEqual(outputs._NRRD_MEMBER_GZIP_FAILURE_REASONS, {})
        self.assertEqual(outputs._NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED, set())
        self.assertFalse(outputs._NRRD_MEMBER_CODEC_SETTING_WARNED)
        self.assertFalse(outputs._NRRD_MEMBER_GZIP_ANNOUNCED)
        self.assertFalse(outputs._NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED)
        self.assertIsNot(outputs._NRRD_GZIP_EXECUTOR_LOCK, old_executor_lock)
        self.assertIsNot(outputs._NRRD_ZERO_MEMBER_LOCK, old_zero_lock)
        self.assertIsNot(outputs._NRRD_MEMBER_GZIP_TEST_LOCK, old_test_lock)
        self.assertIsNot(
            outputs._NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK,
            old_announce_lock,
        )

    def test_executor_shutdown_and_diagnostic_failures_are_contained(self) -> None:
        class BrokenExecutor:
            _shutdown = True

            def __init__(self) -> None:
                self.shutdown_calls = 0

            def shutdown(self, **kwargs: object) -> None:
                self.shutdown_calls += 1
                if 'cancel_futures' in kwargs:
                    raise TypeError('legacy executor sentinel')
                raise RuntimeError('shutdown failure sentinel')

        executor = BrokenExecutor()
        with mock.patch.object(
            outputs, 'runtime_telemetry', side_effect=RuntimeError('telemetry sentinel'),
        ), mock.patch('builtins.print', side_effect=OSError('stdout sentinel')):
            outputs._close_nrrd_gzip_executor_entry('qat', 1, executor)
        self.assertEqual(executor.shutdown_calls, 2)


class MemberWriterHardwareTests(unittest.TestCase):
    def setUp(self) -> None:
        outputs.shutdown_nrrd_gzip_executors()

    def tearDown(self) -> None:
        outputs.shutdown_nrrd_gzip_executors()

    def test_concatenated_native_members_and_cached_zeros_round_trip(self) -> None:
        calls: list[bytes] = []

        def compress(payload: object) -> bytes:
            data = bytes(memoryview(payload).cast('B'))  # type: ignore[arg-type]
            calls.append(data)
            split = max(1, len(data) // 2)
            return gzip.compress(data[:split], mtime=0) + gzip.compress(data[split:], mtime=0)

        sink = outputs.io.BytesIO()
        writer = outputs._MemberParallelGzipPayloadWriter(
            sink, chunk_bytes=16, codec_spec=('qat-test', 1, compress),
        )
        writer.write(b'abcdefghijk')
        writer.write_zeros(23)
        writer.write(b'last')
        writer.close()
        self.assertEqual(gzip.decompress(sink.getvalue()), b'abcdefghijk' + bytes(23) + b'last')
        self.assertEqual(calls, [b'abcdefghijk', b'last'])

    def test_hardware_minimum_coalesces_a_tiny_tail(self) -> None:
        seen: list[int] = []

        class Compressor:
            hardware_backend = True
            minimum_input_bytes = 128
            max_concurrency = 1

            def __call__(self, payload: object) -> bytes:
                data = bytes(memoryview(payload).cast('B'))  # type: ignore[arg-type]
                seen.append(len(data))
                return gzip.compress(data, mtime=0)

        payload = bytes(range(100)) * 3
        sink = outputs.io.BytesIO()
        writer = outputs._MemberParallelGzipPayloadWriter(
            sink, chunk_bytes=256, codec_spec=('qat-test', 1, Compressor()),
        )
        writer.write(payload)
        writer.close()
        self.assertEqual(seen, [300])
        self.assertEqual(gzip.decompress(sink.getvalue()), payload)

    def test_hardware_minimum_merges_a_separate_tiny_tail_on_close(self) -> None:
        seen: list[bytes] = []

        class Compressor:
            hardware_backend = True
            minimum_input_bytes = 128
            max_concurrency = 1

            def __call__(self, payload: object) -> bytes:
                data = bytes(memoryview(payload).cast('B'))  # type: ignore[arg-type]
                seen.append(data)
                return gzip.compress(data, mtime=0)

        first = bytes(range(128)) * 2
        tail = b'separate tiny tail'
        sink = outputs.io.BytesIO()
        writer = outputs._MemberParallelGzipPayloadWriter(
            sink, chunk_bytes=256, codec_spec=('qat-test', 1, Compressor()),
        )

        self.assertEqual(writer.write(first), len(first))
        self.assertEqual(writer.write(tail), len(tail))
        writer.close()

        self.assertEqual(seen, [first + tail])
        self.assertEqual(gzip.decompress(sink.getvalue()), first + tail)

    def test_hardware_minimum_merges_tiny_final_data_after_cached_zeros(self) -> None:
        seen: list[bytes] = []

        class Compressor:
            hardware_backend = True
            minimum_input_bytes = 128
            max_concurrency = 1

            def __call__(self, payload: object) -> bytes:
                data = bytes(memoryview(payload).cast('B'))  # type: ignore[arg-type]
                seen.append(data)
                return gzip.compress(data, mtime=0)

        tail = b'final sparse crop'
        sink = outputs.io.BytesIO()
        writer = outputs._MemberParallelGzipPayloadWriter(
            sink, chunk_bytes=256, codec_spec=('qat-test', 1, Compressor()),
        )

        self.assertEqual(writer.write_zeros(300), 300)
        self.assertEqual(writer.write(tail), len(tail))
        writer.close()

        self.assertEqual(seen, [bytes(128) + tail])
        self.assertEqual(gzip.decompress(sink.getvalue()), bytes(300) + tail)

    def test_hardware_minimum_keeps_a_short_zero_gap_with_final_data(self) -> None:
        seen: list[bytes] = []

        class Compressor:
            hardware_backend = True
            minimum_input_bytes = 128
            max_concurrency = 1

            def __call__(self, payload: object) -> bytes:
                data = bytes(memoryview(payload).cast('B'))  # type: ignore[arg-type]
                seen.append(data)
                return gzip.compress(data, mtime=0)

        first = bytes(range(128)) * 2
        tail = b'final sparse crop'
        sink = outputs.io.BytesIO()
        writer = outputs._MemberParallelGzipPayloadWriter(
            sink, chunk_bytes=256, codec_spec=('qat-test', 1, Compressor()),
        )

        writer.write(first)
        writer.write_zeros(64)
        writer.write(tail)
        writer.close()

        self.assertEqual(seen, [first + bytes(64) + tail])
        self.assertEqual(gzip.decompress(sink.getvalue()), first + bytes(64) + tail)

    def test_cached_zeros_do_not_invoke_hardware(self) -> None:
        class Compressor:
            hardware_backend = True
            minimum_input_bytes = 128
            max_concurrency = 1

            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, payload: object) -> bytes:
                self.calls += 1
                return gzip.compress(bytes(memoryview(payload).cast('B')), mtime=0)  # type: ignore[arg-type]

        compressor = Compressor()
        sink = outputs.io.BytesIO()
        writer = outputs._MemberParallelGzipPayloadWriter(
            sink, chunk_bytes=256, codec_spec=('qat-test', 1, compressor),
        )

        writer.write_zeros(300)
        writer.close()

        self.assertEqual(compressor.calls, 0)
        self.assertEqual(gzip.decompress(sink.getvalue()), bytes(300))

    def test_hardware_type_error_is_never_resubmitted(self) -> None:
        class Compressor:
            hardware_backend = True
            minimum_input_bytes = 1
            max_concurrency = 1

            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, _payload: object) -> bytes:
                self.calls += 1
                raise TypeError('native status conversion failed after request')

        compressor = Compressor()
        writer = outputs._MemberParallelGzipPayloadWriter(
            outputs.io.BytesIO(), codec_spec=('qat-test', 1, compressor),
        )

        def finish() -> None:
            writer.write(b'one hardware request')
            writer.close()

        with self.assertRaisesRegex(TypeError, 'native status conversion failed'):
            finish()
        self.assertEqual(compressor.calls, 1)

    def test_failure_settles_other_native_requests_before_returning(self) -> None:
        second_started = threading.Event()
        second_finished = threading.Event()
        lock = threading.Lock()
        calls = 0

        def compress(payload: object) -> bytes:
            nonlocal calls
            with lock:
                calls += 1
                index = calls
            if index == 1:
                self.assertTrue(second_started.wait(5))
                raise RuntimeError('device failure')
            second_started.set()
            time.sleep(0.05)
            second_finished.set()
            return gzip.compress(bytes(memoryview(payload).cast('B')), mtime=0)  # type: ignore[arg-type]

        sink = outputs.io.BytesIO()
        with mock.patch.dict(os.environ, {'YOLO_TTA_NRRD_GZIP_WORKERS': '2'}, clear=False):
            writer = outputs._MemberParallelGzipPayloadWriter(
                sink, chunk_bytes=4, codec_spec=('failure-test', 1, compress),
            )
            def finish_writer() -> None:
                writer.write(b'abcdefgh')
                writer.close()
            with self.assertRaisesRegex(RuntimeError, 'device failure'):
                finish_writer()
        self.assertTrue(second_finished.is_set())
        self.assertEqual(writer._pending, {})


class LogicalNrrdCodecPinningTests(unittest.TestCase):
    @contextlib.contextmanager
    def _recording_writer(
        self,
        file_handle: object,
        *,
        codec_spec: object,
        seen: list[object],
    ):
        seen.append(codec_spec)
        yield file_handle

    def test_z_shards_share_one_preselected_codec(self) -> None:
        sentinel = ('qat', 1, lambda payload: gzip.compress(bytes(payload), mtime=0))
        selected = 0
        opened: list[object] = []

        def select(**_kwargs: object):
            nonlocal selected
            selected += 1
            return sentinel

        @contextlib.contextmanager
        def open_writer(file_handle: object, *, codec_spec: object):
            opened.append(codec_spec)
            yield file_handle

        def payload_writer(_ref: object, _shape: object, writer: object, **kwargs: object) -> None:
            z0 = int(kwargs.get('z_start', 0))
            writer.write(bytes([65 + z0]))

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'pinned.seg.nrrd'
            ref = types.SimpleNamespace(source='global', shape=(2, 1, 1))
            plan = types.SimpleNamespace(
                stored_shape_tyx=(2, 1, 1),
                segment_extent_xyt=(0, 0, 0, 0, 0, 1),
            )
            with mock.patch.object(outputs, '_resolve_live_ref_extent', side_effect=lambda value: value), \
                    mock.patch.object(outputs, '_nrrd_raster_plan', return_value=plan), \
                    mock.patch.object(outputs, 'nrrd_slicer_header', return_value={}), \
                    mock.patch.object(outputs, 'slicer_segmentation_header_fields', return_value={}), \
                    mock.patch.object(outputs, '_nrrd_full_slice_z_chunk', return_value=1), \
                    mock.patch.object(outputs, '_nrrd_zshard_capacity', return_value=2), \
                    mock.patch.object(outputs, '_nrrd_layer_zshard_bands', return_value=([(0, 1), (1, 2)], None)), \
                    mock.patch.object(outputs, '_require_nrrd_member_codec', side_effect=select), \
                    mock.patch.object(outputs, '_open_nrrd_payload_writer', side_effect=open_writer), \
                    mock.patch.object(outputs, '_write_one_decomposed_nrrd_layer_payload', side_effect=payload_writer), \
                    mock.patch.object(outputs, '_write_nrrd_ascii_header', side_effect=lambda fh, **_kwargs: fh.write(b'H')):
                outputs.write_single_layer_nrrd_from_ref(
                    ref, (2, 1, 1), out_path, z_shards=2,
                )
            self.assertTrue(out_path.is_file())
        self.assertEqual(selected, 1)
        self.assertEqual(opened, [sentinel, sentinel])


class AtomicHardwareFailureTests(unittest.TestCase):
    def tearDown(self) -> None:
        outputs.shutdown_nrrd_gzip_executors()

    def test_midstream_hardware_error_preserves_previous_nrrd(self) -> None:
        class FailingCompressor:
            hardware_backend = True
            minimum_input_bytes = 1
            max_concurrency = 1

            def __call__(self, _payload: object) -> bytes:
                raise RuntimeError('QAT device lost after submission')

        def write_payload(
            _ref: object, _shape: object, writer: object, **_kwargs: object,
        ) -> None:
            writer.write(b'native payload')

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'atomic-hardware.seg.nrrd'
            out_path.write_bytes(b'PREVIOUS')
            plan = types.SimpleNamespace(
                stored_shape_tyx=(1, 1, 16),
                segment_extent_xyt=(0, 15, 0, 0, 0, 0),
            )
            spec = ('qat', 1, FailingCompressor())
            with mock.patch.dict(
                os.environ, {'YOLO_TTA_NRRD_GZIP_WORKERS': '1'}, clear=False,
            ), mock.patch.object(
                outputs, '_resolve_live_ref_extent', side_effect=lambda value: value,
            ), mock.patch.object(
                outputs, '_nrrd_raster_plan', return_value=plan,
            ), mock.patch.object(
                outputs, 'nrrd_slicer_header', return_value={},
            ), mock.patch.object(
                outputs, 'slicer_segmentation_header_fields', return_value={},
            ), mock.patch.object(
                outputs, '_nrrd_full_slice_z_chunk', return_value=1,
            ), mock.patch.object(
                outputs, '_require_nrrd_member_codec', return_value=spec,
            ), mock.patch.object(
                outputs, '_write_nrrd_ascii_header',
                side_effect=lambda fh, **_kwargs: fh.write(b'HEADER\n'),
            ), mock.patch.object(
                outputs, '_write_one_decomposed_nrrd_layer_payload',
                side_effect=write_payload,
            ):
                with self.assertRaisesRegex(RuntimeError, 'QAT device lost'):
                    outputs.write_single_layer_nrrd_from_ref(
                        object(), (1, 1, 16), out_path, z_shards=1,
                    )
            self.assertEqual(out_path.read_bytes(), b'PREVIOUS')
            self.assertEqual(list(Path(tmp).glob('.*.assembling')), [])


if __name__ == '__main__':
    unittest.main()
