"""Real-pipe regressions for gray8 decoder cleanup on Windows and POSIX."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest import mock

import numpy as np

from XTA import media


class _Progress:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, frames):
        pass


class Gray8DecodeCleanupTests(unittest.TestCase):
    def _run_decode(self, *, streaming, byte_count=12, exit_code=0, stderr_bytes=256 * 1024):
        """Use the actual host subprocess pipe implementation without FFmpeg/files."""
        spawned = []
        thread_errors = []
        producer_targets = []
        original_popen = subprocess.Popen
        child = (
            'import sys; '
            f'sys.stdout.buffer.write(bytes(range({byte_count}))); sys.stdout.buffer.flush(); '
            f'sys.stderr.buffer.write(b"decode-diagnostic:" + b"x" * {stderr_bytes}); '
            f'sys.stderr.buffer.flush(); sys.exit({exit_code})'
        )

        def spawn_ffmpeg_replacement(command, **kwargs):
            self.assertEqual(command[0], 'ffmpeg')
            self.assertEqual(command[-1], '-')
            process = original_popen([sys.executable, '-B', '-c', child], **kwargs)
            spawned.append((process, process.stdout, process.stderr))
            return process

        result = error = readiness = None
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(mock.patch.object(media, '_require_bin'))
            stack.enter_context(mock.patch.object(media, 'allocate_workspace_array',
                side_effect=lambda *, shape, **kwargs: np.empty(shape, np.uint8)))
            stack.enter_context(mock.patch.object(media, 'tqdm', _Progress))
            stack.enter_context(mock.patch.object(media.subprocess, 'Popen', side_effect=spawn_ffmpeg_replacement))
            stack.enter_context(mock.patch.object(threading, 'excepthook', side_effect=thread_errors.append))
            stack.enter_context(mock.patch.object(media, '_start_streaming_producer',
                side_effect=lambda target, **kwargs: producer_targets.append(target)))
            stack.enter_context(mock.patch.object(media, 'streaming_producers_aborted', return_value=False))
            stack.enter_context(mock.patch.object(media, '_register_streaming_subprocess'))
            unregister = stack.enter_context(mock.patch.object(media, '_unregister_streaming_subprocess'))
            stack.enter_context(mock.patch.dict(media._VOLUME_READINESS_BY_ARRAY_ID, {}, clear=True))
            decoder = media.decode_video_to_memmap_gray8_streaming if streaming else media.decode_video_to_memmap_gray8
            try:
                result = decoder(Path('unused-input.mkv'), Path('unused-output.dat'),
                                 num_frames=2, width=3, height=2, overwrite=True)
                if streaming:
                    readiness = media.volume_readiness(result)
                    producer_targets[0]()
            except BaseException as exc:
                error = exc
            finally:
                for process, stdout, stderr in spawned:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
                    if stdout is not None:
                        stdout.close()
                    if stderr is not None:
                        stderr.close()
            if streaming:
                unregister.assert_called_once_with(spawned[0][0])
        self.assertEqual(thread_errors, [], 'communicate started a reader for the closed stdout pipe')
        self.assertEqual(len(spawned), 1)
        process, stdout, stderr = spawned[0]
        self.assertIsNone(process.stdout, 'Consumed stdout must be detached before communicate')
        self.assertTrue(stdout.closed)
        self.assertTrue(stderr.closed)
        return result, error, readiness, process.returncode

    def test_plain_and_streaming_decode_preserve_gray8_bytes_and_drain_large_stderr(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                result, error, readiness, code = self._run_decode(streaming=streaming)
                self.assertIsNone(error)
                self.assertEqual(code, 0)
                np.testing.assert_array_equal(result, np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
                if readiness is not None:
                    readiness.wait_all()

    def test_nonzero_exit_still_propagates_stderr_diagnostics(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                _, error, readiness, code = self._run_decode(streaming=streaming, exit_code=7)
                self.assertEqual(code, 7)
                if streaming:
                    self.assertIsNone(error)
                    with self.assertRaises(RuntimeError) as caught:
                        readiness.wait_all()
                    failure = caught.exception.__cause__
                else:
                    failure = error
                self.assertIsInstance(failure, RuntimeError)
                self.assertIn('ffmpeg decode failed: decode-diagnostic:', str(failure))

    def test_short_stdout_keeps_eof_failure_and_streaming_root_cause(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                # Small stderr avoids testing the separate pre-EOF pipe-drain
                # policy: this case specifically qualifies cleanup after EOF.
                _, error, readiness, _ = self._run_decode(streaming=streaming, byte_count=5, stderr_bytes=32)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn('Unexpected EOF', str(error))
                if streaming:
                    with self.assertRaises(RuntimeError) as caught:
                        readiness.wait_all()
                    self.assertIs(caught.exception.__cause__, error)


if __name__ == '__main__':
    unittest.main()
