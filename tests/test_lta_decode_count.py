"""Strict source-frame validation with actual pipes and mapped workspaces."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import media


class _Progress:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def update(self, _frames):
        pass


class StrictDecodeCountTests(unittest.TestCase):
    def _decode_mapped_pipe(self, root: Path, *, byte_count: int, exit_code: int):
        processes = []
        mappings = []
        commands = []
        original_popen = subprocess.Popen
        output_path = root / "decoded.u8.raw"
        # Diagnostics precede stdout and exceed ordinary pipe capacity. Strict
        # EOF verification must drain them while reading the decoded pixels.
        child = (
            "import sys; "
            "sys.stderr.buffer.write(b'x' * (300 * 1024) + b'last-decoder-diagnostic'); "
            "sys.stderr.buffer.flush(); "
            f"sys.stdout.buffer.write(bytes([3]) * {byte_count}); "
            "sys.stdout.buffer.flush(); "
            f"sys.exit({exit_code})"
        )

        def allocate(*, shape, dtype, path, **kwargs):
            self.assertFalse(kwargs["reuse_existing"])
            mapping = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
            mappings.append(mapping)
            return mapping

        def spawn(command, **kwargs):
            commands.append(command)
            process = original_popen([sys.executable, "-B", "-c", child], **kwargs)
            processes.append(process)
            return process

        result = failure = None
        try:
            with (
                mock.patch.object(media, "_require_bin"),
                mock.patch.object(media, "allocate_workspace_array", side_effect=allocate),
                mock.patch.object(media.subprocess, "Popen", side_effect=spawn),
                mock.patch.object(media, "tqdm", _Progress),
                redirect_stdout(io.StringIO()),
            ):
                try:
                    result = media.decode_video_to_memmap_gray8(
                        root / "unused.mkv", output_path,
                        num_frames=2, width=3, height=2,
                        overwrite=True, prefer_memory=False, reserve_bytes=0,
                        strict_frame_count=True,
                    )
                except BaseException as error:
                    failure = error
            self.assertEqual(len(mappings), 1)
            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].poll())
            map_index = commands[0].index("-map")
            self.assertEqual(commands[0][map_index + 1], "0:v:0")
            return result, failure, mappings[0], output_path
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)

    def test_count_and_process_failures_close_mapping_while_exception_is_retained(self):
        for name, byte_count, exit_code, message in (
            ("extra", 18, 0, "Frame count mismatch"),
            ("short", 5, 0, "Unexpected EOF"),
            ("nonzero", 12, 7, "ffmpeg decode failed"),
        ):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _result, failure, mapping, path = self._decode_mapped_pipe(
                    root, byte_count=byte_count, exit_code=exit_code,
                )
                try:
                    self.assertIsInstance(failure, RuntimeError)
                    self.assertIn(message, str(failure))
                    self.assertTrue(mapping._mmap.closed)
                    if name == "nonzero":
                        self.assertIn("last-decoder-diagnostic", str(failure))
                    # A retained error traceback must not keep the failed file
                    # mapped and prevent rename/removal on Windows.
                    moved = root / "discarded.raw"
                    path.replace(moved)
                    moved.unlink()
                finally:
                    if not mapping._mmap.closed:
                        mapping._mmap.close()

    def test_exact_count_returns_an_open_mapping_with_unmodified_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            result, failure, mapping, _path = self._decode_mapped_pipe(
                Path(folder), byte_count=12, exit_code=0,
            )
            try:
                self.assertIsNone(failure)
                self.assertIs(result, mapping)
                self.assertFalse(mapping._mmap.closed)
                np.testing.assert_array_equal(result, np.full((2, 2, 3), 3, dtype=np.uint8))
            finally:
                mapping._mmap.close()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for the multi-stream fixture")
    def test_strict_decode_selects_first_video_stream_even_when_second_is_larger(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "two-video-streams.mkv"
            subprocess.run([
                shutil.which("ffmpeg"), "-v", "error", "-nostdin",
                "-filter_threads", "1",
                "-f", "lavfi", "-i", "color=c=white:s=16x16:r=1:d=2",
                "-f", "lavfi", "-i", "color=c=black:s=32x32:r=1:d=2",
                "-map", "0:v:0", "-map", "1:v:0", "-c:v", "ffv1",
                "-pix_fmt", "gray", "-threads", "1", "-y", str(video),
            ], check=True, capture_output=True, timeout=30)
            mapping = None
            try:
                with redirect_stdout(io.StringIO()), mock.patch.object(media, "tqdm", _Progress):
                    mapping = media.decode_video_to_memmap_gray8(
                        video, root / "first-stream.raw", num_frames=2, width=16, height=16,
                        overwrite=True, prefer_memory=False, prefer_memfd=False,
                        reserve_bytes=0, strict_frame_count=True,
                    )
                self.assertEqual(mapping.shape, (2, 16, 16))
                # The first stream is white; FFmpeg's automatic selection would
                # choose the larger, black second stream instead.
                self.assertTrue(bool(np.all(mapping >= 230)))
            finally:
                if mapping is not None:
                    media.close_memmap_array(mapping)


if __name__ == "__main__":
    unittest.main()
