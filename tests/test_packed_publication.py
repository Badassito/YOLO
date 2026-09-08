from __future__ import annotations
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from XTA import packed_publication as packed
from XTA import cuda_d1, geometry
from XTA.interpolation import RawBBoxMaskStore, INTERNAL_PACKED_CVOL_FORMAT


def source_words(volume):
    values = np.packbits(volume.reshape(-1), bitorder='little')
    return np.pad(values, (0, (-len(values)) % 4)).view(np.uint32)


@unittest.skipIf(packed._packed_owner_metadata is None, 'Numba unavailable')
class PackedPublicationTests(unittest.TestCase):
    def test_crop_bits_counts_extents_and_padding_match_dense_reference(self):
        rng = np.random.default_rng(1512)
        for width in (1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127):
            for height in (1, 5, 23):
                volume = (rng.random((7, height, width)) < .17).astype(np.uint8)
                volume[0] = 0
                volume[1] = 1
                words = source_words(volume)
                saved = words.copy()
                result = np.zeros_like(volume)
                for first in (0, 3, 6):
                    records, data = packed.encode_owner_packed_block(words, volume.shape, first, min(3, 7-first))
                    for rec in records:
                        frame = volume[rec.z]
                        self.assertEqual(rec.foreground, int(np.count_nonzero(frame)))
                        if not rec.foreground:
                            self.assertEqual((rec.y0, rec.y1, rec.x0, rec.x1, rec.size), (0, 0, 0, 0, 0))
                            continue
                        ys, xs = np.nonzero(frame)
                        self.assertEqual((rec.y0, rec.y1, rec.x0, rec.x1),
                                         (ys.min(), ys.max()+1, xs.min(), xs.max()+1))
                        rows = data[rec.offset:rec.offset+rec.size].reshape(rec.y1-rec.y0, -1)
                        crop = np.unpackbits(rows, axis=1, bitorder='little')
                        np.testing.assert_array_equal(crop[:, rec.x1-rec.x0:], 0)
                        result[rec.z, rec.y0:rec.y1, rec.x0:rec.x1] = crop[:, :rec.x1-rec.x0]
                np.testing.assert_array_equal(result, volume)
                np.testing.assert_array_equal(words, saved)

    def test_publication_numpy_fallback_and_raw_optout_match(self):
        rng = np.random.default_rng(1513)
        volume = (rng.random((13, 37, 43)) < .12).astype(np.uint8)
        volume[[0, 12]] = 0
        view = geometry.get_view_infos(*volume.shape, cartesian_views=('transverse',))[0]
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('packed', 'fallback', 'raw'):
                with mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': '0' if mode == 'raw' else '1'}), \
                        mock.patch.object(cuda_d1, 'encode_owner_packed_block',
                                          side_effect=RuntimeError('optional compiler unavailable') if mode == 'fallback' else packed.encode_owner_packed_block):
                    result = cuda_d1._d1_finalize_bitset_layer(words=source_words(volume), output_shape=volume.shape,
                              store_dir=Path(directory)/mode, model_name='model', view=view)
                ref = result['d1_layer_ref']
                self.assertEqual(ref.storage_format == INTERNAL_PACKED_CVOL_FORMAT, mode != 'raw')
                store = RawBBoxMaskStore.open(ref.path, mmap_payload=True)
                try:
                    actual = np.stack([store.decode_slice(z) for z in range(len(volume))])
                    np.testing.assert_array_equal(actual, volume)
                    self.assertEqual(result['d1_cvol_stats']['foreground_voxels'], int(volume.sum()))
                finally:
                    store.close()

    def test_invalid_shape_or_bitset_cannot_publish(self):
        for words, shape, first, count in [(np.zeros(1, np.uint8), (1, 2, 3), 0, 1),
                                          (np.zeros(1, np.uint32), (1, 2, 3), 0, 2),
                                          (np.zeros(2, np.uint32), (1, 2, 3), 0, 1)]:
            with self.assertRaises(ValueError):
                packed.encode_owner_packed_block(words, shape, first, count)


if __name__ == '__main__':
    unittest.main()
