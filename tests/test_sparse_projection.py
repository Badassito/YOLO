"""Sparse projector equality against actual legacy Azimuthal projection operators."""
from __future__ import annotations

import contextlib
import gc
from dataclasses import replace
import io
import os
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest import mock

import numpy as np
from XTA import backprojection, geometry, sparse_projection
from XTA.config import TiltedViewGroup
from XTA.interpolation import CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store
from XTA.runtime import close_memmap_array


HAS_NATIVE = bool(getattr(geometry.cv2, '__file__', None))


def views(shape=(9, 11, 13), spacing=45.0, native_raster=0):
    all_views = geometry.get_view_infos(*shape, cartesian_views=[],
        azimuthal_views=['transverse', 'sagittal', 'coronal', 'tilted_transverse', 'tilted_sagittal', 'tilted_coronal'],
        azimuthal_azimuth_angles=[spacing]*6,
        tilt_groups=[TiltedViewGroup(('transverse', 'sagittal', 'coronal'), (30.0,), ('vertical', 'horizontal'))],
        azimuthal_native_raster=native_raster)
    return [view for view in all_views if view.family == 'azimuthal']


def read_store(path):
    store = RawBBoxMaskStore.open(path, mmap_payload=True)
    try:
        return np.stack([store.decode_slice(z) for z in range(store.shape[0])])
    finally:
        store.close()


@unittest.skipUnless(HAS_NATIVE, 'Native OpenCV unavailable')
class SparseAzimuthalProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = mock.patch.dict(os.environ, {'YOLO_TTA_GPU_BACKPROJECT': '0', 'YOLO_TTA_TELEMETRY': '0', 'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.gpu = mock.patch.object(backprojection, 'gpu_backproject_enabled', return_value=False)
        self.gpu.start()
        self.addCleanup(self.gpu.stop)
        self.counter = 0

    def compare(self, view, data, target_shape, fmt=CVOL_FORMAT):
        self.counter += 1
        root = self.root/str(self.counter)
        root.mkdir()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            source_path = root/'input.cvol'
            write_raw_bbox_mask_store(data, source_path, format_name=fmt, desc='test input', workers=1)
            oracle = backprojection.backproject_azimuthal_volume_to_volume(data, view, root/'oracle.dat',
                'numerical oracle', prefer_memory=True, reserve_bytes=0, workers=2, out_shape_tyx=target_shape)
            expected = np.asarray(oracle).copy()
            close_memmap_array(oracle)
            source = RawBBoxMaskStore.open(source_path, mmap_payload=True)
            try:
                # The algorithm must consume bounded raw/packed slabs, never expand
                # the view or even ask the adapter to decode a complete source slice.
                with mock.patch.object(RawBBoxMaskStore, 'decode_slice', side_effect=AssertionError('dense source decode')), \
                        mock.patch.object(RawBBoxMaskStore, 'decode_slice_crop', side_effect=AssertionError('whole crop decode')):
                    result = sparse_projection.project_azimuthal_sparse_store(source, view, root/'projected.cvol',
                        out_shape_tyx=target_shape, workers=2)
                # The helper borrows source ownership; it remains readable afterwards.
                self.assertEqual(source.shape, data.shape)
                self.assertIsNotNone(source._chunks_mmap if np.any(data) else source)
            finally:
                source.close()
        actual = read_store(root/'projected.cvol')
        np.testing.assert_array_equal(actual, expected, err_msg=f'{view.name}: {data.shape} -> {target_shape}')
        self.assertEqual(result['foreground_voxels'], int(np.count_nonzero(expected)))
        self.assertEqual(tuple(result['shape']), target_shape)
        self.assertEqual(result['storage_format'], INTERNAL_PACKED_CVOL_FORMAT)
        self.assertTrue(source_path.exists())
        self.assertFalse(list(root.glob('.*.projection-*')))
        return result

    def test_all_azimuthal_bases_tilt_signs_directions_and_temporal_changes(self):
        rng = np.random.default_rng(9903)
        for view in views():
            data = (rng.random((view.num_slices, view.src_h, view.src_w)) < 0.12).astype(np.uint8)
            for shape in ((9, 11, 13), (6, 8, 10), (13, 15, 17)):
                with self.subTest(view=view.name, target=shape):
                    self.compare(view, data, shape)

    def test_reduced_processing_and_native_raster_raw_packed_nonuniform_angles(self):
        rng = np.random.default_rng(234)
        for view in views(shape=(13, 17, 19), native_raster=9):
            # A coarser, nonuniform angle list exercises legacy nearest-angle
            # densification and the 0/180 reversal ownership rules.
            view = replace(view, num_slices=4, azimuths_deg=(1.0, 37.0, 88.0, 151.0))
            for size in (0, 5):
                shape = (4, view.src_h, view.src_w) if size == 0 else (4, size, size)
                data = (rng.random(shape) < 0.2).astype(np.uint8)
                for fmt in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
                    with self.subTest(view=view.name, shape=shape, format=fmt):
                        self.compare(view, data, (8, 15, 13), fmt)

    def test_empty_and_dense_inputs_extents_and_non_byte_aligned_output(self):
        for view in views(shape=(5, 7, 9), spacing=90.0)[::3]:
            shape = (view.num_slices, view.src_h, view.src_w)
            for fill in (0, 1):
                with self.subTest(view=view.name, fill=fill):
                    self.compare(view, np.full(shape, fill, np.uint8), (3, 5, 11), INTERNAL_PACKED_CVOL_FORMAT)

    def test_map_strip_equations_equal_actual_dense_map_and_cache_is_bounded(self):
        for view in views(shape=(9, 11, 13), spacing=37.0)[::2]:
            probe = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
            grid = backprojection.resolve_azimuthal_processing_grid(probe, view)
            plan, _ = backprojection.build_azimuthal_backprojection_plan(view)
            expected = backprojection._azimuthal_dense_map_for_processing(
                backprojection.build_dense_azimuthal_backprojection_map(view, plan, out_shape_hw=(8, 10)), grid)
            with mock.patch.object(sparse_projection, '_MAP_STRIP_PIXELS', 20):
                parts = list(sparse_projection._map_key_strips(view, plan, grid, (8, 10)))
            keys = np.concatenate([pair[0] for pair in parts])
            positions = np.concatenate([pair[1] for pair in parts])
            valid = np.flatnonzero(expected.valid_mask.reshape(-1))
            np.testing.assert_array_equal(positions, valid)
            expected_keys = expected.source_idx_map.reshape(-1)[valid].astype(np.int64)*grid.processing_w + expected.u_idx_map.reshape(-1)[valid]
            np.testing.assert_array_equal(keys, expected_keys)
        sparse_projection.clear_sparse_projection_cache()

    def test_basis_impulses_single_angle_and_degenerate_output_axes(self):
        selected = views(shape=(3, 5, 7), spacing=90.0)
        selected = selected[:3] + selected[3::4]
        for view in selected:
            view = replace(view, num_slices=1, azimuths_deg=(73.0,))
            shape = (1, view.src_h, view.src_w)
            for flat in range(int(np.prod(shape))):
                data = np.zeros(shape, dtype=np.uint8)
                data.reshape(-1)[flat] = 1
                with self.subTest(view=view.name, impulse=flat):
                    self.compare(view, data, (2, 4, 6), INTERNAL_PACKED_CVOL_FORMAT)
            data = np.ones(shape, dtype=np.uint8)
            for output in ((1, 4, 6), (4, 1, 6), (4, 6, 1)):
                with self.subTest(view=view.name, target=output):
                    self.compare(view, data, output)

    def test_concurrent_projection_uses_shared_immutable_maps_and_separate_outputs(self):
        sparse_projection.clear_sparse_projection_cache()
        view = views()[3]
        data = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
        data[:, 1:5, 2:6] = 1
        source_path = self.root/'concurrent.cvol'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            write_raw_bbox_mask_store(data, source_path, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='input')
            oracle = backprojection.backproject_azimuthal_volume_to_volume(data, view, self.root/'expected.dat',
                'oracle', reserve_bytes=0, out_shape_tyx=(7, 8, 11))
            expected = np.asarray(oracle).copy()
            close_memmap_array(oracle)
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda i: sparse_projection.project_azimuthal_sparse_store(
                    source_path, view, self.root/f'parallel-{i}', out_shape_tyx=(7, 8, 11), workers=2), range(4)))
        self.assertEqual(sum(bool(result['map_cache_hit']) for result in results), 3)
        for result in results:
            np.testing.assert_array_equal(read_store(result['path']), expected)

    def test_bounded_input_slabs_and_legacy_cache_are_independent(self):
        view = replace(views()[0], center_x=5.125)
        shape = (view.num_slices, view.src_h, view.src_w)
        before = set(backprojection._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE)
        sparse_projection._inverse_map(view, shape, (7, 8, 11))
        self.assertEqual(set(backprojection._DENSE_AZIMUTHAL_BACKPROJECT_MAP_CACHE), before)
        source_path = self.root/'slabs.cvol'
        with contextlib.redirect_stdout(io.StringIO()):
            write_raw_bbox_mask_store(np.ones(shape, np.uint8), source_path,
                format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='input')
        store = RawBBoxMaskStore.open(source_path, mmap_payload=True)
        try:
            with mock.patch.object(sparse_projection, '_INPUT_SLAB_BYTES', 20):
                rows = [(z, y, x, crop.copy()) for z, y, x, crop, _ in sparse_projection._input_slabs(store)]
            self.assertTrue(all(crop.nbytes <= max(20, shape[2]) for _, _, _, crop in rows))
            self.assertEqual(sum(crop.size for _, _, _, crop in rows), int(np.prod(shape)))
        finally:
            store.close()
        sparse_projection.clear_sparse_projection_cache()
        view = views()[0]
        with mock.patch.object(sparse_projection, '_MAP_CACHE_MAX_BYTES', 1024):
            sparse_projection._inverse_map(view, (view.num_slices, view.src_h, view.src_w), (9, 11, 13))
            self.assertLessEqual(sparse_projection.sparse_projection_cache_info()['bytes'], 1024)
        sparse_projection.clear_sparse_projection_cache()

    def test_input_failure_and_existing_output_do_not_clobber_other_stores(self):
        view = views()[0]
        data = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
        source = self.root/'source.cvol'
        with contextlib.redirect_stdout(io.StringIO()):
            write_raw_bbox_mask_store(data, source, format_name=CVOL_FORMAT, desc='input')
        target = self.root/'existing'
        target.mkdir()
        (target/'keep').write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            sparse_projection.project_azimuthal_sparse_store(source, view, target, out_shape_tyx=(9, 11, 13))
        self.assertEqual((target/'keep').read_bytes(), b'preserve')
        with self.assertRaises(ValueError):
            sparse_projection.project_azimuthal_sparse_store(source, view, source/'child', out_shape_tyx=(9, 11, 13))
        with mock.patch.object(sparse_projection, '_scatter_crop', side_effect=RuntimeError('fault')):
            with self.assertRaisesRegex(RuntimeError, 'fault'):
                sparse_projection.project_azimuthal_sparse_store(source, view, self.root/'failed', out_shape_tyx=(9, 11, 13))
        # Mock's recorded call tuple can keep its borrowed mmap crop alive until
        # its cycle is collected; that external owner is not the projector's leak.
        gc.collect()
        self.assertFalse((self.root/'failed').exists())
        self.assertTrue(source.exists())
        self.assertFalse(list(self.root.glob('.*.projection-*')))


class PackedProjectionFormatTests(unittest.TestCase):
    def test_all_bit_alignments_preserve_crop_payload_and_padding(self):
        rng = np.random.default_rng(419)
        for x0 in range(16):
            for width in range(1, 24-x0):
                source = np.zeros((1, 5, 23), np.uint8)
                source[0, 1:4, x0:x0+width] = rng.integers(0, 2, (3, width), dtype=np.uint8)
                packed = np.packbits(source, axis=2, bitorder='little')
                bounds = np.array([[1, 4, x0, x0+width]], dtype=np.int32)
                counts = np.array([np.count_nonzero(source)], dtype=np.uint64)
                payload = sparse_projection._packed_output_slice(0, packed, bounds, counts)
                encoded = np.frombuffer(payload.payload, dtype=np.uint8).reshape(3, (width+7)//8)
                actual = np.unpackbits(encoded, axis=1, count=width, bitorder='little')
                np.testing.assert_array_equal(actual, source[0, 1:4, x0:x0+width])
                if width % 8:
                    self.assertFalse(np.any(encoded[:, -1] >> np.uint8(width % 8)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
