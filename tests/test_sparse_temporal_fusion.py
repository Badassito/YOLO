"""Sparse temporal fusion against the production general-restore helper."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from XTA import finalization
from XTA.interpolation import (
    CTILE_INDEX_DTYPE,
    CVOL_FORMAT,
    INTERNAL_PACKED_CVOL_FORMAT,
    NrrdLayerRef,
    RawBBoxMaskStore,
)


def _write_ref(root: Path, name: str, volume: np.ndarray, fmt: str = CVOL_FORMAT) -> NrrdLayerRef:
    """Encode small canonical raw/packed fixtures without dependency stubs."""
    path = root / name
    path.mkdir()
    index = np.zeros((int(volume.shape[0]),), dtype=CTILE_INDEX_DTYPE)
    chunks = bytearray()
    for z, plane in enumerate(volume):
        ys, xs = np.nonzero(plane)
        if not ys.size:
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        crop = np.ascontiguousarray(plane[y0:y1, x0:x1], dtype=np.uint8)
        payload = (
            np.packbits(crop, axis=1, bitorder='little').tobytes()
            if fmt == INTERNAL_PACKED_CVOL_FORMAT else crop.tobytes()
        )
        record = index[int(z)]
        record['kind'] = 1
        record['offset'] = len(chunks)
        record['payload_size'] = len(payload)
        record['payload_nbytes'] = crop.size
        record['y0'], record['x0'], record['y1'], record['x1'] = y0, x0, y1, x1
        chunks.extend(payload)
    index.tofile(path / 'index.bin')
    (path / 'chunks.bin').write_bytes(chunks)
    (path / 'meta.json').write_text(json.dumps({
        'format': fmt,
        'shape': list(volume.shape),
        'dtype': 'bool',
        'logical_dtype_in_pipeline': 'uint8_0_or_1',
        'precodec': 'numpy_packbits_axis_x_little' if fmt == INTERNAL_PACKED_CVOL_FORMAT else 'none',
    }), encoding='utf-8')
    return NrrdLayerRef(
        key=name, name=name, path=path, shape=tuple(volume.shape), storage_format=fmt,
        layer_role='additive_component', recomposition_op='union',
    )


def _random_mask(shape: tuple[int, int, int], seed: int, *, empty: bool = False) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not empty:
        rng = np.random.default_rng(seed)
        mask[:, 2:7, 4:11] = rng.random((shape[0], 5, 7)) > 0.6
        mask[::3] = 0
    return mask


def _reference(refs: list[NrrdLayerRef], seed: np.ndarray, *, workers: int = 1) -> np.ndarray:
    destination = seed.copy()
    finalization._union_projected_layer_refs_with_dense_restore_into_volume(
        refs, destination, workers=workers,
    )
    return destination


class SparseTemporalFusionTests(unittest.TestCase):
    def test_exact_oracle_raw_packed_up_down_multiple_groups_and_native_sources(self) -> None:
        cases = 0
        for source_t, target_t in ((19, 11), (7, 17), (1, 9), (21, 1)):
            for fmt in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
                for grouped in (False, True):
                    for empty in (False, True):
                        with (
                            self.subTest(source_t=source_t, target_t=target_t, format=fmt, grouped=grouped, empty=empty),
                            tempfile.TemporaryDirectory() as folder,
                            contextlib.redirect_stdout(io.StringIO()),
                            mock.patch.object(finalization, 'fused_final_restore_geometry_groups_enabled', return_value=grouped),
                        ):
                            arrays = [
                                _random_mask((source_t, 24, 30), 31, empty=empty),
                                _random_mask((source_t, 24, 30), 51, empty=empty),
                                _random_mask((max(2, source_t + 4), 24, 30), 73, empty=empty),
                                _random_mask((target_t, 24, 30), 41, empty=empty),
                            ]
                            refs = [_write_ref(Path(folder), str(i), array, fmt) for i, array in enumerate(arrays)]
                            seed = _random_mask((target_t, 24, 30), 101)
                            expected = _reference(refs, seed, workers=2)
                            actual = seed.copy()
                            with mock.patch.object(
                                finalization, '_resize_union_plane_to_out_xy',
                                side_effect=AssertionError('selected sparse path normalized a full plane'),
                            ):
                                finalization._union_projected_layer_refs_grouped_into_volume(refs, actual, workers=2)
                            np.testing.assert_array_equal(actual, expected)
                            for ref, array in zip(refs, arrays):
                                store = RawBBoxMaskStore.open(ref.path)
                                try:
                                    np.testing.assert_array_equal(
                                        np.stack([store.decode_slice(z) for z in range(array.shape[0])]), array,
                                    )
                                finally:
                                    store.close()
                            cases += 1
        self.assertEqual(cases, 32)

    def test_xy_resize_generic_sources_and_nonunion_roles_use_general_restore(self) -> None:
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            volume = _random_mask((8, 24, 30), 51)
            ref = _write_ref(root, 'sparse', volume)
            raw = root / 'raw.dat'
            volume.tofile(raw)
            generic = replace(ref, path=raw, storage_format='raw_u8')
            cases = [
                ([ref], np.zeros((5, 48, 60), dtype=np.uint8)),
                ([replace(ref, layer_role='checkpoint', recomposition_op='select')], np.zeros((5, 24, 30), dtype=np.uint8)),
                ([generic], np.zeros((5, 24, 30), dtype=np.uint8)),
                ([ref], np.zeros((8, 24, 30), dtype=np.uint8)),
            ]
            resize_original = finalization._resize_union_plane_to_out_xy

            def resize(plane: np.ndarray, height: int, width: int) -> np.ndarray:
                if plane.shape == (height, width):
                    return resize_original(plane, height, width)
                # The fallback contract is tested even in the import-only suite,
                # where another test module may replace OpenCV with a stub.
                return np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)

            for refs, seed in cases:
                with mock.patch.object(finalization, '_resize_union_plane_to_out_xy', side_effect=resize):
                    expected = _reference(refs, seed)
                    actual = seed.copy()
                    with mock.patch.object(
                        finalization, '_union_projected_layer_refs_with_dense_restore_into_volume',
                        wraps=finalization._union_projected_layer_refs_with_dense_restore_into_volume,
                    ) as fallback:
                        finalization._union_projected_layer_refs_grouped_into_volume(refs, actual)
                    fallback.assert_called_once()
                    np.testing.assert_array_equal(actual, expected)

    def test_noncanonical_metadata_and_payload_replay_is_idempotent(self) -> None:
        for corrupt_metadata in (False, True):
            with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                root = Path(folder)
                ref = _write_ref(root, 'temporal', _random_mask((8, 24, 30), 31))
                native = _write_ref(root, 'native', _random_mask((5, 24, 30), 41))
                if corrupt_metadata:
                    meta = json.loads((ref.path / 'meta.json').read_text(encoding='utf-8'))
                    meta['logical_dtype_in_pipeline'] = 'arbitrary_uint8'
                    (ref.path / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
                else:
                    for source, value in ((ref, 128), (native, 64)):
                        payload = np.memmap(source.path / 'chunks.bin', mode='r+', dtype=np.uint8)
                        payload[np.flatnonzero(payload)[0]] = value
                        payload._mmap.close()
                seed = np.zeros((5, 24, 30), dtype=np.uint8)
                seed[0, 0, 0] = 2
                refs = [ref, native]
                expected = _reference(refs, seed)
                actual = seed.copy()
                with mock.patch.object(
                    finalization, '_union_projected_layer_refs_with_dense_restore_into_volume',
                    wraps=finalization._union_projected_layer_refs_with_dense_restore_into_volume,
                ) as fallback:
                    finalization._union_projected_layer_refs_grouped_into_volume(refs, actual, workers=2)
                fallback.assert_called_once()
                np.testing.assert_array_equal(actual, expected)

    def test_invalid_store_preserves_general_reader_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            ref = _write_ref(Path(folder), 'source', _random_mask((8, 24, 30), 31))
            (ref.path / 'index.bin').write_bytes(b'')
            seed = np.zeros((5, 24, 30), dtype=np.uint8)
            with self.assertRaises(ValueError):
                _reference([ref], seed)
            with mock.patch.object(
                finalization, '_union_projected_layer_refs_with_dense_restore_into_volume',
                wraps=finalization._union_projected_layer_refs_with_dense_restore_into_volume,
            ) as fallback, self.assertRaises(ValueError):
                finalization._union_projected_layer_refs_grouped_into_volume([ref], seed)
            fallback.assert_called_once()
            self.assertTrue((ref.path / 'chunks.bin').exists())

    def test_source_destination_file_alias_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            ref = _write_ref(Path(folder), 'source', np.ones((2, 4, 4), dtype=np.uint8))
            mapped = np.memmap(ref.path / 'chunks.bin', mode='r+', dtype=np.uint8, shape=(2, 4, 4))
            view = np.asarray(mapped)
            try:
                self.assertFalse(finalization._try_union_temporal_sparse_layer_refs_into_volume(
                    [ref], view, workers=1, desc='alias',
                ))
                np.testing.assert_array_equal(mapped, np.ones(mapped.shape, dtype=np.uint8))
            finally:
                mapped._mmap.close()

    def test_destination_layout_dtype_and_writeability_guards(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            ref = _write_ref(Path(folder), 'source', _random_mask((8, 24, 30), 31))
            readonly = np.zeros((5, 24, 30), dtype=np.uint8)
            readonly.flags.writeable = False
            noncontiguous = np.zeros((5, 24, 60), dtype=np.uint8)[:, :, ::2]
            for destination in (readonly, noncontiguous, np.zeros((5, 24, 30), dtype=bool)):
                self.assertFalse(finalization._try_union_temporal_sparse_layer_refs_into_volume(
                    [ref], destination, workers=1, desc='destination guard',
                ))
                self.assertFalse(destination.any())

    def test_empty_refs_do_not_modify_destination(self) -> None:
        destination = np.ones((3, 4, 5), dtype=np.uint8)
        finalization._union_projected_layer_refs_grouped_into_volume([], destination)
        np.testing.assert_array_equal(destination, np.ones(destination.shape, dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
