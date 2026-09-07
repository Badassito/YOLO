"""Fatal projection failures discard CPU sink state without releasing GPU owners."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import assembly
from XTA.cylindrical_cuda_projection import RadialCudaProjectionUnsafeFailure
from XTA.geometry import ViewInfo
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter


class FatalProjectionCleanupTests(unittest.TestCase):
    def exercise(self, *, cleanup_failure=None, bbox_store=True, recoverable_first=False):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            view = ViewInfo(
                name='radial_cleanup_probe', family='radial', num_slices=2,
                src_h=4, src_w=5, full_t=3, full_h=4, full_w=5, pad_mode='pad',
            )
            source = np.ones((2, 4, 5), dtype=np.uint8)
            projector = SimpleNamespace(close=mock.Mock())
            lease = SimpleNamespace(release=mock.Mock())
            fatal = RadialCudaProjectionUnsafeFailure('unsettled device stream', projector)
            fatal.stage_lease = lease
            state = {}

            def writer_factory(**kwargs):
                writer = IncrementalRawBBoxMaskStoreWriter(**kwargs)
                state['writer'] = writer
                state['fd'] = writer._fd
                # Exercise the real store/fd ownership on Windows too. Bounding
                # box extraction and POSIX pwrite are unrelated to this cleanup
                # contract; one synchronous payload makes the store partial.
                def consume_partial(z, block):
                    self.assertEqual(z, 0)
                    payload = np.ascontiguousarray(block, dtype=np.uint8).tobytes()
                    self.assertEqual(os.write(writer._fd, payload), len(payload))
                writer.consume = consume_partial
                if cleanup_failure == 'abort':
                    writer.abort = mock.Mock(side_effect=RuntimeError('abort cleanup failed'))
                elif cleanup_failure == 'discard':
                    discard = writer.discard
                    def discard_then_fail():
                        discard()
                        raise RuntimeError('discard cleanup failed')
                    writer.discard = discard_then_fail
                return writer

            def fail_after_partial_output(_source, _view, raw_path, **kwargs):
                state['attempts'] = state.get('attempts', 0) + 1
                state['raw_path'] = Path(raw_path)
                state['raw_path'].parent.mkdir(parents=True, exist_ok=True)
                state['raw_path'].write_bytes(b'partial projection scratch')
                if kwargs['projection_block_callback'] is not None:
                    callback = kwargs['projection_block_callback']
                    self.assertTrue(kwargs['sink_only'])
                    callback(0, np.ones((1, 4, 5), dtype=np.uint8))
                    self.assertGreater(os.fstat(state['fd']).st_size, 0)
                    self.assertTrue(callback.store_dir.is_dir())
                if recoverable_first and state['attempts'] == 1:
                    raise RuntimeError('recoverable sink error')
                raise fatal

            with mock.patch.object(assembly, 'raw_bbox_nrrd_layers_enabled', return_value=bbox_store), \
                    mock.patch.object(assembly, 'delayed_native_expansion_enabled', return_value=False), \
                    mock.patch.object(assembly, 'final_source_output_shape', return_value=(3, 4, 5)), \
                    mock.patch.object(assembly, 'IncrementalRawBBoxMaskStoreWriter', side_effect=writer_factory), \
                    mock.patch.object(assembly, 'project_view_volume_to_orthogonal_volume', side_effect=fail_after_partial_output) as project, \
                    mock.patch.object(assembly, 'write_raw_bbox_mask_store', side_effect=AssertionError('dense retry entered')), \
                    self.assertRaises(RadialCudaProjectionUnsafeFailure) as caught:
                assembly.materialize_nrrd_view_layer(
                    source, model_name='probe', view=view, source='fullframe', mask_kind='yolo',
                    temp_dir=root, known_has_foreground=True, submit_to_sink=False,
                )

            self.assertIs(caught.exception, fatal)
            self.assertIs(caught.exception.projector, projector)
            self.assertIs(caught.exception.stage_lease, lease)
            self.assertEqual(project.call_count, 2 if recoverable_first else 1)
            projector.close.assert_not_called()
            lease.release.assert_not_called()
            np.testing.assert_array_equal(source, 1)
            self.assertFalse(state['raw_path'].exists())
            if bbox_store:
                writer = state['writer']
                self.assertIsNone(writer._fd)
                self.assertFalse(writer.store_dir.exists())
                with self.assertRaises(OSError):
                    os.fstat(state['fd'])
                if cleanup_failure != 'abort' and not recoverable_first:
                    self.assertIs(writer._failed_reason, fatal)
            if cleanup_failure and hasattr(fatal, 'add_note'):
                self.assertTrue(any(f'{cleanup_failure} cleanup failed' in note for note in fatal.__notes__))

    def test_midstream_fatal_closes_partial_store_without_retry_or_gpu_release(self):
        self.exercise()

    def test_cleanup_errors_cannot_replace_fatal_or_skip_other_cpu_cleanup(self):
        for operation in ('abort', 'discard'):
            with self.subTest(operation=operation):
                self.exercise(cleanup_failure=operation)

    def test_dense_partial_scratch_is_removed_without_retry(self):
        self.exercise(bbox_store=False)

    def test_fatal_during_transaction_retry_is_cleaned_and_never_retried_again(self):
        self.exercise(recoverable_first=True)


if __name__ == '__main__':
    unittest.main()
