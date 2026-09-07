"""Device-encoded CVOL publication preserves exact bytes, indices and ownership."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import interpolation as ip


def encode(source, first=0, packed=False):
    records, chunks = [], []
    offset = 0
    for local, plane in enumerate(source):
        ys, xs = np.nonzero(plane)
        if len(ys):
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()+1), int(xs.min()), int(xs.max()+1)
            crop = np.ascontiguousarray(plane[y0:y1, x0:x1] != 0, dtype=np.uint8)
            data = np.packbits(crop, axis=1, bitorder='little') if packed else crop
            data = data.reshape(-1)
        else:
            y0 = y1 = x0 = x1 = 0
            data = np.empty(0, np.uint8)
        records.append(SimpleNamespace(z=first+local, y0=y0, y1=y1, x0=x0, x1=x1,
                                       foreground=len(ys), offset=offset, size=data.size))
        chunks.append(data)
        offset += data.size
    return tuple(records), np.concatenate(chunks)


class EncodedMaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = np.zeros((5, 9, 13), np.uint8)
        self.source[1, 4, 6] = 1
        self.source[2, 1:8:2, 1:12:2] = 1
        self.source[3, 0, 0] = self.source[3, -1, -1] = 1
        self.source[4] = 1
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def writer(self, packed=False):
        writer = ip.IncrementalRawBBoxMaskStoreWriter(shape=self.source.shape,
            store_dir=self.root / ('packed' if packed else 'raw'),
            format_name=ip.INTERNAL_PACKED_CVOL_FORMAT if packed else ip.CVOL_FORMAT, desc='encoded test')
        self.addCleanup(writer.discard)
        return writer

    def test_raw_and_packed_out_of_order_blocks_decode_exactly_without_pixel_rescans(self):
        for packed in (False, True):
            writer = self.writer(packed)
            for first, stop in ((3, 5), (0, 3)):
                records, data = encode(self.source[first:stop], first, packed)
                before = data.copy()
                data.flags.writeable = False
                with mock.patch.object(ip, 'cv2', SimpleNamespace(boundingRect=mock.Mock(side_effect=AssertionError('rescan')))), \
                        mock.patch.object(ip.np, 'count_nonzero', side_effect=AssertionError('recount')), \
                        mock.patch.object(ip.np, 'packbits', side_effect=AssertionError('repack')):
                    writer.consume_encoded_block(first, records, data, packed=packed)
                np.testing.assert_array_equal(data, before)
            stats = writer.finalize()
            self.assertEqual(stats['foreground_voxels'], int(self.source.sum()))
            self.assertEqual(stats['nonempty_slices'], 4)
            self.assertEqual(stats['segment_extent_ijk'], [0, 12, 0, 8, 1, 4])
            self.assertEqual(stats['raw_payload_bytes'], writer.chunks_path.stat().st_size)
            with contextlib.closing(ip.RawBBoxMaskStore.open(writer.store_dir, mmap_payload=True)) as store:
                actual = np.empty_like(self.source)
                for z in range(len(actual)):
                    store.fill_decoded_slice_into(z, actual[z])
                np.testing.assert_array_equal(actual, self.source)

    def test_invalid_packet_cannot_reserve_or_modify_store_state(self):
        writer = self.writer()
        records, data = encode(self.source)
        for name, value in (('z', 99), ('offset', 3), ('size', 100), ('foreground', 1)):
            altered = list(records)
            fields = vars(altered[0]).copy()
            fields[name] = value
            altered[0] = SimpleNamespace(**fields)
            with self.subTest(field=name), self.assertRaises((ValueError, IndexError)):
                writer.consume_encoded_block(0, altered, data, packed=False)
            self.assertEqual(writer._next_offset, 0)
            self.assertFalse(writer._slice_state.any())
        with self.assertRaises(ValueError):
            writer.consume_encoded_block(0, records, data, packed=True)
        with self.assertRaises(ValueError):
            writer.consume_encoded_block(0, records, np.append(data, np.uint8(0)), packed=False)

    def test_packed_padding_and_duplicate_publication_are_rejected(self):
        writer = self.writer(True)
        records, data = encode(self.source, packed=True)
        broken = data.copy()
        broken[records[1].offset] |= np.uint8(128)
        with self.assertRaisesRegex(ValueError, 'padding'):
            writer.consume_encoded_block(0, records, broken, packed=True)
        writer.consume_encoded_block(0, records, data, packed=True)
        with self.assertRaisesRegex(ValueError, 'more than once'):
            writer.consume_encoded_block(0, records, data, packed=True)

    def test_partial_write_aborts_transaction_and_can_be_discarded(self):
        writer = self.writer()
        records, data = encode(self.source)
        fd = writer._fd
        with mock.patch.object(writer, '_pwrite_all', side_effect=OSError('disk error')), \
                self.assertRaisesRegex(OSError, 'disk error'):
            writer.consume_encoded_block(0, records, data, packed=False)
        self.assertTrue(writer.failed)
        self.assertEqual(writer._active_callbacks, 0)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            writer.finalize()
        writer.discard()
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertFalse(writer.store_dir.exists())

    def test_positional_write_fallback_preserves_cursor_and_concurrent_offsets(self):
        path = self.root / 'positional.bin'
        fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
        try:
            os.write(fd, bytes(80))
            os.lseek(fd, 7, os.SEEK_SET)
            with mock.patch.object(ip.os, 'pwrite', None, create=True), ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(ip.IncrementalRawBBoxMaskStoreWriter._pwrite_all,
                    fd, memoryview(bytes([i+1])*20), i*20) for i in range(4)]
                for future in futures:
                    future.result()
            self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 7)
        finally:
            os.close(fd)
        self.assertEqual(path.read_bytes(), b''.join(bytes([i+1])*20 for i in range(4)))


if __name__ == '__main__':
    unittest.main()
