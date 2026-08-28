from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from volume_tta import finalization, interpolation


def _bbox(plane: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(plane))
    if int(ys.size) == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    return (x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)


def _raw_ref(path: Path, shape: tuple[int, int, int], key: str) -> interpolation.NrrdLayerRef:
    return interpolation.NrrdLayerRef(
        key=str(key),
        name=str(key),
        path=Path(path),
        shape=tuple(int(value) for value in shape),
        dtype='uint8',
        storage_format='raw_u8',
    )


def _write_cvol(root: Path, source: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    index = np.zeros((int(source.shape[0]),), dtype=interpolation.CTILE_INDEX_DTYPE)
    chunks = bytearray()
    for z_idx, plane in enumerate(np.asarray(source, dtype=np.uint8)):
        x0, y0, width, height = _bbox(plane)
        if int(width) <= 0 or int(height) <= 0:
            continue
        x1 = int(x0 + width)
        y1 = int(y0 + height)
        crop = np.ascontiguousarray(plane[y0:y1, x0:x1], dtype=np.uint8)
        payload = crop.tobytes()
        rec = index[int(z_idx)]
        rec['kind'] = np.uint8(1)
        rec['offset'] = np.uint64(len(chunks))
        rec['payload_size'] = np.uint64(len(payload))
        rec['payload_nbytes'] = np.uint64(crop.size)
        rec['y0'] = np.uint32(y0)
        rec['x0'] = np.uint32(x0)
        rec['y1'] = np.uint32(y1)
        rec['x1'] = np.uint32(x1)
        chunks.extend(payload)
    (root / 'chunks.bin').write_bytes(bytes(chunks))
    index.tofile(root / 'index.bin')
    (root / 'meta.json').write_text(json.dumps({
        'format': interpolation.CVOL_FORMAT,
        'shape': [int(value) for value in source.shape],
        'precodec': 'none',
    }))


class StreamingFinalUnionTests(unittest.TestCase):
    def test_native_bbox_store_unions_only_its_decoded_crop(self) -> None:
        shape = (3, 6, 7)
        source = np.zeros(shape, dtype=np.uint8)
        source[0, 1:3, 2:5] = 1
        source[2, 4:6, 0:2] = 1
        destination = np.zeros(shape, dtype=np.uint8)
        destination[1, 0, 6] = 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'layer.cvol'
            _write_cvol(root, source)
            ref = interpolation.NrrdLayerRef(
                key='native-crop',
                name='native-crop',
                path=root,
                shape=shape,
                dtype='uint8',
                storage_format=interpolation.CVOL_FORMAT,
            )
            singular_destination = destination.copy()
            finalization._union_projected_layer_ref_into_volume(
                ref, singular_destination, workers=2,
            )
            finalization._union_projected_layer_refs_grouped_into_volume(
                (ref,), destination, workers=2,
            )

        expected = source.copy()
        expected[1, 0, 6] = 1
        np.testing.assert_array_equal(singular_destination, expected)
        np.testing.assert_array_equal(destination, expected)

    def test_equal_reduced_geometries_resize_once_per_output_slice(self) -> None:
        source_shape = (2, 3, 4)
        output_shape = (3, 6, 8)
        first = np.zeros(source_shape, dtype=np.uint8)
        second = np.zeros(source_shape, dtype=np.uint8)
        first[0, 0, 1] = 1
        first[1, 2, 3] = 1
        second[0, 1, 2] = 1
        second[1, 0, 0] = 1
        destination = np.zeros(output_shape, dtype=np.uint8)
        destination[1, 5, 7] = 1
        resize_calls: list[tuple[int, int]] = []

        def _resize(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
            resize_calls.append((int(out_h), int(out_w)))
            return np.repeat(np.repeat(np.asarray(plane), 2, axis=0), 2, axis=1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = []
            for key, array in (('first', first), ('second', second)):
                path = root / f'{key}.u8.dat'
                mm = np.memmap(path, dtype=np.uint8, mode='w+', shape=source_shape)
                mm[:] = array
                mm.flush()
                del mm
                refs.append(_raw_ref(path, source_shape, key))

            with (
                mock.patch.object(
                    finalization, '_resize_union_plane_to_out_xy', side_effect=_resize,
                ),
                mock.patch.object(
                    finalization, 'fused_final_restore_geometry_groups_enabled',
                    return_value=True,
                ),
            ):
                finalization._union_projected_layer_refs_grouped_into_volume(
                    refs,
                    destination,
                    workers=1,
                    desc='test grouped restore',
                )

        expected = np.zeros(output_shape, dtype=np.uint8)
        expected[1, 5, 7] = 1
        reduced_union = np.bitwise_or(first, second)
        for out_z in range(output_shape[0]):
            reduced_plane = np.zeros(source_shape[1:], dtype=np.uint8)
            for source_z in finalization._restore_source_indices_for_output_z(
                source_shape[0], output_shape[0], out_z,
            ):
                reduced_plane |= reduced_union[int(source_z)]
            expected[out_z] |= np.repeat(
                np.repeat(reduced_plane, 2, axis=0), 2, axis=1,
            )

        np.testing.assert_array_equal(destination, expected)
        self.assertEqual(resize_calls, [(6, 8), (6, 8), (6, 8)])


if __name__ == '__main__':
    unittest.main()
