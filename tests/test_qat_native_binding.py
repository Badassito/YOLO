from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_BUILD_HOST = (
    sys.platform.startswith('linux')
    and platform.machine().strip().lower() in {'x86_64', 'amd64'}
)


class QatNativeSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / 'native' / 'qat_codec.c').read_text(encoding='utf-8')

    def test_hardware_only_configuration_and_standard_gzip_are_explicit(self) -> None:
        self.assertIn('qzInit(&state->session, 0)', self.source)
        self.assertIn('QZ_DISABLE_SOFTWARE_BACKUP', self.source)
        self.assertIn('QZ_DISABLE_SOFTWARE_ONLY_EXECUTION', self.source)
        self.assertIn('is_sensitive_mode = 0', self.source)
        self.assertIn('data_fmt = QZ_DEFLATE_GZIP', self.source)
        self.assertIn('#define VOLUME_TTA_QAT_LEVEL_MAX 8', self.source)

    def test_request_proof_and_lifecycle_guards_are_present(self) -> None:
        self.assertIn('qzMaxCompressedLength', self.source)
        self.assertIn('qzCompressExt(', self.source)
        self.assertIn('source_length != source_expected', self.source)
        self.assertIn('QZ_SW_EXECUTION_MASK', self.source)
        self.assertIn('QZ_TIMEOUT_MASK', self.source)
        self.assertIn('pthread_key_create', self.source)
        self.assertIn('pthread_atfork', self.source)
        self.assertIn('g_forked_child = 1', self.source)
        self.assertIn('Py_BEGIN_ALLOW_THREADS', self.source)

    def test_build_floor_contract_exports_and_telemetry_are_present(self) -> None:
        self.assertIn('QATZIP_API_VERSION < 20500', self.source)
        for method in (
            '"capabilities"',
            '"compress_gzip"',
            '"preflight_thread_state"',
            '"stats"',
            '"close_thread_state"',
        ):
            self.assertIn(method, self.source)
        for key in (
            '"hardware_requests"',
            '"software_fallback_requests"',
            '"sessions_created"',
            '"sessions_closed"',
            '"elapsed_ns"',
            '"post_fork_child"',
        ):
            self.assertIn(key, self.source)


@unittest.skipUnless(SUPPORTED_BUILD_HOST, 'native QATzip binding is Linux x86-64 only')
class QatNativeBindingFakeProviderTests(unittest.TestCase):
    """Compile the production binding against a hardware-free QATzip fake."""

    build_directory: tempfile.TemporaryDirectory[str]
    extension_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not any(shutil.which(name) for name in ('cc', 'gcc', 'clang')):
            raise unittest.SkipTest('a C compiler is required for the native binding test')
        try:
            from setuptools import Distribution, Extension
            from setuptools.command.build_ext import build_ext
        except Exception as exc:  # pragma: no cover - build environment failure
            raise unittest.SkipTest(f'setuptools build_ext is unavailable: {exc}') from exc

        cls.build_directory = tempfile.TemporaryDirectory(
            prefix='volume-tta-qatzip-fake-'
        )
        build_root = Path(cls.build_directory.name)
        extension = Extension(
            'volume_tta._qat_codec',
            sources=[
                str(ROOT / 'native' / 'qat_codec.c'),
                str(ROOT / 'tests' / 'native_qatzip_fake' / 'fake_qatzip.c'),
            ],
            include_dirs=[str(ROOT / 'tests' / 'native_qatzip_fake')],
            extra_compile_args=['-std=c11', '-pthread', '-Wall', '-Wextra', '-Werror'],
            extra_link_args=['-pthread'],
        )
        distribution = Distribution({
            'name': 'volume-tta-qatzip-native-test',
            'ext_modules': [extension],
        })
        command = build_ext(distribution)
        command.ensure_finalized()
        command.build_lib = str(build_root / 'lib')
        command.build_temp = str(build_root / 'temp')
        command.force = True
        command.run()
        cls.extension_path = Path(
            command.get_ext_fullpath('volume_tta._qat_codec')
        ).resolve()
        if not cls.extension_path.is_file():
            raise AssertionError(
                f'native fake build did not create {cls.extension_path}'
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, 'build_directory'):
            cls.build_directory.cleanup()
        super().tearDownClass()

    def run_native(self, body: str, *, mode: str = 'hardware') -> None:
        bootstrap = f'''
            import importlib.util
            import pathlib

            extension_path = pathlib.Path({str(self.extension_path)!r})
            spec = importlib.util.spec_from_file_location(
                'volume_tta._qat_codec', extension_path,
            )
            assert spec is not None and spec.loader is not None
            qat = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qat)
        '''
        environment = dict(os.environ)
        environment['VOLUME_TTA_FAKE_QAT_MODE'] = str(mode)
        source = textwrap.dedent(bootstrap) + '\n' + textwrap.dedent(body)
        completed = subprocess.run(
            [sys.executable, '-c', source],
            cwd=str(ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f'native fake subprocess failed in mode={mode!r}\n'
                f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}'
            ),
        )

    def test_stub_status_still_admits_proven_hardware_session_and_round_trips(self) -> None:
        self.run_native('''
            import gzip

            capabilities = qat.capabilities()
            assert capabilities['hardware_available'] is True
            assert capabilities['standard_gzip'] is True
            assert capabilities['software_fallback_enabled'] is False
            assert capabilities['qz_status_populated'] is False
            assert capabilities['device_count'] is None
            assert capabilities['max_concurrency'] == 1
            assert capabilities['supported_levels'] == tuple(range(1, 9))
            assert capabilities['rejected_ambiguous_levels'] == (9,)
            assert capabilities['gzip_format'] == 'QZ_DEFLATE_GZIP'

            payload = (b'qat-native-binding-round-trip\\x00' * 5000)[:131072]
            encoded = qat.compress_gzip(payload, 1, require_hardware=True)
            assert encoded[:4] == b'\\x1f\\x8b\\x08\\x00'
            assert gzip.decompress(encoded) == payload

            stats = qat.stats()
            assert stats['logical_requests'] == 1
            assert stats['hardware_requests'] == 1
            assert stats['software_fallback_requests'] == 0
            assert stats['input_bytes'] == len(payload)
            assert stats['output_bytes'] == len(encoded)
            assert stats['sessions_created'] == 1
            assert stats['active_sessions'] == 1
            assert isinstance(stats['elapsed_ns'], int) and stats['elapsed_ns'] >= 0
            assert stats['physical_members'] is None
            assert stats['queue_busy_events'] is None

            qat.close_thread_state()
            closed = qat.stats()
            assert closed['sessions_closed'] == 1
            assert closed['active_sessions'] == 0
        ''')

    def test_populated_status_is_reported_without_becoming_required(self) -> None:
        self.run_native('''
            capabilities = qat.capabilities()
            assert capabilities['hardware_available'] is True
            assert capabilities['qz_status_populated'] is True
            assert capabilities['device_count'] == 2
            assert capabilities['deflate_device_count'] == 2
            assert capabilities['max_concurrency'] == 2
        ''', mode='populated_status')
        self.run_native('''
            capabilities = qat.capabilities()
            assert capabilities['hardware_available'] is True
            assert capabilities['qz_status_status'] == -2
            assert capabilities['qz_status_populated'] is False
        ''', mode='status_error')

    def test_no_hardware_and_setup_failures_are_capability_misses(self) -> None:
        self.run_native('''
            capabilities = qat.capabilities()
            assert capabilities['hardware_available'] is False
            assert 'QZ_NOSW_NO_HW' in capabilities['unavailable_reason']
        ''', mode='no_hardware')
        self.run_native('''
            capabilities = qat.capabilities()
            assert capabilities['hardware_available'] is False
            assert 'qzSetupSessionDeflate' in capabilities['unavailable_reason']
            assert 'QZ_NOSW_NO_INST_ATTACH' in capabilities['unavailable_reason']
        ''', mode='setup_error')

    def test_known_ambiguous_level_and_non_hardware_contracts_fail_closed(self) -> None:
        self.run_native('''
            payload = b'x' * 1024

            try:
                qat.compress_gzip(payload, 9, require_hardware=True)
            except ValueError as exc:
                assert '[1, 8]' in str(exc)
            else:
                raise AssertionError('level 9 was not rejected')

            try:
                qat.compress_gzip(payload, 1, require_hardware=False)
            except ValueError as exc:
                assert 'hardware-only' in str(exc)
            else:
                raise AssertionError('require_hardware=False was not rejected')

            try:
                qat.compress_gzip(payload, 1, numa_id=0)
            except NotImplementedError as exc:
                assert 'NUMA' in str(exc)
            else:
                raise AssertionError('unsupported NUMA binding was accepted')

            try:
                qat.compress_gzip(b'too short', 1)
            except ValueError as exc:
                assert '128' in str(exc)
            else:
                raise AssertionError('sub-threshold request was accepted')
        ''')

    def test_request_proof_rejects_software_partial_timeout_and_bad_output(self) -> None:
        cases = {
            'software': ('software execution', 'software_fallback_requests'),
            'partial': ('consumed=', 'partial_consumption_failures'),
            'timeout': ('timeout=1', 'timeouts'),
            'bad_framing': ('framing_invalid=1', 'failures'),
            'extended_gzip': ('framing_invalid=1', 'failures'),
            'lost_session': ('session_hw_status=QZ_NO_INST_ATTACH', 'failures'),
            'bad_bound': ('invalid bound', None),
        }
        for mode, (message, counter) in cases.items():
            with self.subTest(mode=mode):
                counter_assertion = (
                    f"assert qat.stats()[{counter!r}] == 1"
                    if counter is not None else
                    "assert qat.stats()['logical_requests'] == 0"
                )
                self.run_native(f'''
                    payload = b'hardware-proof' * 1024
                    try:
                        qat.compress_gzip(payload, 1)
                    except qat.QATzipError as exc:
                        assert {message!r} in str(exc), str(exc)
                    else:
                        raise AssertionError('invalid native outcome was accepted')
                    {counter_assertion}
                ''', mode=mode)

    def test_thread_exit_runs_tls_session_cleanup(self) -> None:
        self.run_native('''
            import gzip
            import threading

            payload = b'thread-owned-session' * 1024
            failures = []

            def worker():
                try:
                    assert gzip.decompress(qat.compress_gzip(payload, 1)) == payload
                except BaseException as exc:
                    failures.append(exc)

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            assert not failures, failures
            stats = qat.stats()
            assert stats['sessions_created'] == 1
            assert stats['sessions_closed'] == 1
            assert stats['active_sessions'] == 0
        ''')

    def test_post_fork_child_is_fail_closed(self) -> None:
        self.run_native('''
            import os

            assert qat.capabilities()['hardware_available'] is True
            child = os.fork()
            if child == 0:
                capabilities = qat.capabilities()
                ok = (
                    capabilities['hardware_available'] is False
                    and capabilities['post_fork_child'] is True
                    and 'spawn or exec' in capabilities['unavailable_reason']
                )
                os._exit(0 if ok else 7)
            waited, status = os.waitpid(child, 0)
            assert waited == child
            assert os.waitstatus_to_exitcode(status) == 0
        ''')


if __name__ == '__main__':
    unittest.main()
