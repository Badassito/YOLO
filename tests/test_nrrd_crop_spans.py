from __future__ import annotations
import gzip
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from XTA.nrrd_spans import canonical_zero_member, stream_native_crop_spans
from XTA.outputs import _MemberParallelGzipPayloadWriter
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore, CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT


class NrrdCropSpanTests(unittest.TestCase):
    def setUp(self):
        # Aggregate discovery can stub OpenCV. Storage/stream tests use an
        # independent NumPy bbox oracle and also run with real OpenCV separately.
        def bbox(a):
            ys, xs = np.nonzero(a)
            return ((int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
                    if len(xs) else (0, 0, 0, 0))
        patcher = mock.patch('XTA.interpolation.cv2.boundingRect', side_effect=bbox)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_native_spans_preserve_bytes_and_each_sparse_observation(self):
        rng = np.random.default_rng(823)
        a = np.zeros((11, 37, 43), np.uint8)
        a[1, 0:7, :9] = rng.integers(0, 2, (7, 9), dtype=np.uint8)
        a[3:6, 13:16, 7:41] = rng.integers(0, 2, (3, 3, 34), dtype=np.uint8)
        a[8, -3:, -2:] = 1
        for fmt in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
            for first, stop in ((0, 11), (2, 9), (10, 11)):
                with tempfile.TemporaryDirectory() as td:
                    w = IncrementalRawBBoxMaskStoreWriter(shape=a.shape, store_dir=Path(td)/'layer',
                                                         format_name=fmt, desc='span-test')
                    try:
                        w.consume(0, a); w.finalize()
                    except BaseException:
                        w.discard()
                        raise
                    store = RawBBoxMaskStore.open(w.store_dir, mmap_payload=True)
                    sink = io.BytesIO()
                    writer = _MemberParallelGzipPayloadWriter(sink, codec_spec=('zlib', 1, gzip.compress))
                    observations = []
                    def observe(z, y, x, crop):
                        observations.append(z)
                        np.testing.assert_array_equal(crop, a[z, y:y+crop.shape[0], x:x+crop.shape[1]])
                    try:
                        encoded = stream_native_crop_spans(store, writer, first, stop, 86, observe)
                        writer.close()
                    finally:
                        store.close()
                    self.assertEqual(gzip.decompress(sink.getvalue()), a[first:stop].tobytes())
                    self.assertEqual(observations, [z for z in range(first, stop) if a[z].any()])
                    self.assertLess(encoded, a[first:stop].nbytes)

    def test_canonical_zero_members_are_bounded_complete_and_reusable(self):
        for bit in range(21):
            member = canonical_zero_member(1 << bit)
            self.assertEqual(gzip.decompress(member), bytes(1 << bit))
            self.assertIs(canonical_zero_member(1 << bit), member)
        self.assertLessEqual(canonical_zero_member.cache_info().currsize, 21)
        for n in (0, 3, 2**21, -1):
            with self.assertRaises(ValueError):
                canonical_zero_member(n)

    def test_zero_runs_preserve_order_and_propagate_failed_writes(self):
        sink = io.BytesIO()
        w = _MemberParallelGzipPayloadWriter(sink, codec_spec=('zlib', 1, gzip.compress))
        w.write_owned_known_nonzero(b'front')
        w.write_canonical_zeros(1234567)
        w.write_owned_known_nonzero(b'tail')
        w.close()
        self.assertEqual(gzip.decompress(sink.getvalue()), b'front'+bytes(1234567)+b'tail')
        with self.assertRaises(RuntimeError):
            w.write_canonical_zeros(1)
        w = _MemberParallelGzipPayloadWriter(io.BytesIO(), codec_spec=('zlib', 1, gzip.compress))
        with mock.patch.object(w, '_drain', side_effect=OSError('failed output')):
            with self.assertRaisesRegex(OSError, 'failed output'):
                w.write_canonical_zeros(1)
        self.assertTrue(w.closed)
        self.assertFalse(w._pending)

    def test_hardware_minimum_keeps_existing_zero_policy(self):
        codec = mock.Mock(minimum_input_bytes=65536)
        writer = _MemberParallelGzipPayloadWriter(io.BytesIO(), codec_spec=('qat', 1, codec))
        with mock.patch.object(writer, 'write_zeros', return_value=123) as original:
            self.assertEqual(writer.write_canonical_zeros(123), 123)
            original.assert_called_once_with(123)


if __name__ == '__main__':
    unittest.main()
