"""Sparse component publication preserves orthogonal geometry and output lifetime."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import gzip
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import assembly, finalization, outputs
from XTA.geometry import ViewInfo
from XTA.interpolation import CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store


def _view(orientation: str) -> ViewInfo:
    t, h, w = 13, 17, 19
    num, ph, pw = {'transverse': (t, h, w), 'sagittal': (h, t, w), 'coronal': (w, t, h)}[orientation]
    return ViewInfo(name=f'{orientation}__tta_a0', physical_view_name=orientation,
                    num_slices=num, src_h=ph, src_w=pw, pad_mode='clamp',
                    full_t=t, full_h=h, full_w=w, family='orthogonal', tta_aug_id='a0')


def _parameters(root: Path, view: ViewInfo, count: int) -> dict:
    return dict(added_voxels=count, model_name='test', view=view, source='fullframe',
                pass_index=1, interpolation_walk_back_index=1, interpolation_candidate_index=1,
                stage='bridge', description='sparse transpose test', temp_dir=root,
                workers=1, keep_temp=True)


def _read_ref(ref, output_shape):
    source = outputs._open_nrrd_layer_ref(ref)
    try:
        return np.stack([outputs._read_layer_slice_in_output_shape(source, output_shape, z)
                         for z in range(output_shape[0])])
    finally:
        outputs._close_nrrd_layer_source(source)
        outputs._drop_nrrd_raw_store_chunks_ram_cache(source)


def _dense_reference(data, root, view):
    return assembly.materialize_nrrd_view_layer(
        data, model_name='reference', view=view, source='reference', mask_kind='bridge',
        temp_dir=root, workers=1, emit_empty=True, known_has_foreground=bool(np.any(data)),
        submit_to_sink=False,
    )


def _read_nrrd(path):
    header, payload = path.read_bytes().split(b'\n\n', 1)
    sizes_line = next(line for line in header.splitlines() if line.startswith(b'sizes:'))
    sizes = tuple(map(int, sizes_line.split(b':', 1)[1].split()))
    return np.frombuffer(gzip.decompress(payload), dtype=np.uint8).reshape(tuple(reversed(sizes)))


@unittest.skipUnless(getattr(assembly.cv2, '__version__', None), 'requires native image-processing dependencies')
class SparseComponentRetirementTests(unittest.TestCase):
    def setUp(self):
        self.configuration = mock.patch.dict('os.environ', {
            'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1', 'YOLO_TTA_NRRD_MEMBER_CODEC': 'zlib',
        })
        self.configuration.start()
        self.addCleanup(self.configuration.stop)
        self.disabled = mock.patch.object(assembly, '_SPARSE_COMPONENT_NUMBA_DISABLED', False)
        self.disabled.start()
        self.addCleanup(self.disabled.stop)

    @unittest.skipUnless(assembly._numba is not None, 'optional compiled sparse transpose')
    def test_permutation_restore_and_union_match_dense_reference_without_volume_decode(self):
        rng = np.random.default_rng(2007)
        for orientation in ('transverse', 'sagittal', 'coronal'):
            view = _view(orientation)
            for reduced in (False, True):
                shape = (view.num_slices, 7, 9) if reduced else (view.num_slices, view.src_h, view.src_w)
                for fmt in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
                    for density in (0.0, 0.04, 1.0):
                        with self.subTest(orientation=orientation, reduced=reduced, format=fmt, density=density), \
                                tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                            root = Path(directory)
                            data = (rng.random(shape) < density).astype(np.uint8)
                            path = root / 'input.cvol'
                            write_raw_bbox_mask_store(data, path, format_name=fmt, desc='input', workers=1)
                            reference = _dense_reference(data, root / 'reference', view)
                            with mock.patch.object(assembly, 'allocate_workspace_array', side_effect=AssertionError('dense workspace')), \
                                    mock.patch.object(assembly, 'project_view_volume_to_orthogonal_volume', side_effect=AssertionError('dense projection')), \
                                    mock.patch.object(RawBBoxMaskStore, 'decode_slice', side_effect=AssertionError('full slice decode')):
                                actual = assembly.materialize_interpolation_component_nrrd_view_layer(
                                    path, **_parameters(root / 'actual', view, int(data.sum())))
                            permutation = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2), 'coronal': (1, 2, 0)}[orientation]
                            orthogonal = data.transpose(permutation)
                            self.assertEqual(actual.shape, orthogonal.shape)
                            np.testing.assert_array_equal(_read_ref(actual, actual.shape), orthogonal)
                            self.assertEqual(actual.segment_extent_ijk, outputs.compute_segment_extent_zyx(orthogonal))
                            for output_shape in (actual.shape, (23, 29, 31), (9, 11, 13)):
                                expected = _read_ref(reference, output_shape)
                                np.testing.assert_array_equal(_read_ref(actual, output_shape), expected)
                                destination = (rng.random(output_shape) < 0.02).astype(np.uint8)
                                union_expected = destination | expected
                                finalization._union_projected_layer_ref_into_volume(actual, destination, workers=1)
                                np.testing.assert_array_equal(destination, union_expected)
                            self.assertTrue(actual.path.exists())
                            self.assertEqual(actual.storage_format, fmt if orientation == 'transverse' else INTERNAL_PACKED_CVOL_FORMAT)

    @unittest.skipUnless(assembly._numba is not None, 'optional compiled sparse transpose')
    def test_actual_gzip_nrrd_and_low_quality_mirror_match_dense_reference(self):
        for orientation in ('transverse', 'sagittal', 'coronal'):
            for fmt in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
                with self.subTest(orientation=orientation, format=fmt), tempfile.TemporaryDirectory() as directory, \
                        contextlib.redirect_stdout(io.StringIO()):
                    root = Path(directory)
                    view = _view(orientation)
                    data = np.zeros((view.num_slices, 7, 9), dtype=np.uint8)
                    data[1:4, 1:6, 2:5] = 1
                    data[-1, -1, -1] = 1
                    path = root / 'input.cvol'
                    write_raw_bbox_mask_store(data, path, format_name=fmt, desc='input', workers=1)
                    reference = _dense_reference(data, root / 'reference', view)
                    actual = assembly.materialize_interpolation_component_nrrd_view_layer(
                        path, **_parameters(root / 'actual', view, int(data.sum())))
                    for ref, token in ((reference, 'reference'), (actual, 'actual')):
                        outputs.write_layer_nrrd_with_low_quality_mirrors(ref, (23, 29, 31), root / f'{token}.nrrd',
                                                                       [((5, 6, 7), root / f'{token}-lq.nrrd')], z_shards=1)
                    np.testing.assert_array_equal(_read_nrrd(root / 'actual.nrrd'), _read_nrrd(root / 'reference.nrrd'))
                    np.testing.assert_array_equal(_read_nrrd(root / 'actual-lq.nrrd'), _read_nrrd(root / 'reference-lq.nrrd'))

    @unittest.skipUnless(assembly._numba is not None, 'optional compiled sparse transpose')
    def test_keep_temp_false_preserves_returned_backing_and_retires_transposed_source(self):
        for orientation in ('transverse', 'sagittal', 'coronal'):
            with self.subTest(orientation=orientation), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                view = _view(orientation)
                data = np.ones((view.num_slices, 7, 9), dtype=np.uint8)
                path = root / 'input.cvol'
                write_raw_bbox_mask_store(data, path, format_name=CVOL_FORMAT, desc='input')
                kwargs = _parameters(root, view, int(data.sum()))
                kwargs['keep_temp'] = False
                ref = assembly.materialize_interpolation_component_nrrd_view_layer(path, **kwargs)
                self.assertEqual(path.exists(), orientation == 'transverse')
                self.assertTrue(ref.path.is_dir())
                self.assertTrue(np.all(_read_ref(ref, ref.shape)))

    def test_packed_decode_slabs_are_bounded_and_preserve_non_byte_aligned_rows(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            rng = np.random.default_rng(190)
            data = (rng.random((7, 41, 61)) < 0.11).astype(np.uint8)
            path = Path(directory) / 'input.cvol'
            write_raw_bbox_mask_store(data, path, format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='input')
            source = RawBBoxMaskStore.open(path, mmap_payload=True)
            try:
                slabs = list(assembly._iter_sparse_component_decoded_slabs(source, 0, maximum_bytes=130))
                self.assertGreater(len(slabs), 2)
                self.assertLessEqual(max(slab.nbytes for _, _, slab in slabs), 130)
                y0, _, _, _, expected = source.decode_slice_crop(0)
                self.assertEqual(slabs[0][0], y0)
                np.testing.assert_array_equal(np.concatenate([slab for _, _, slab in slabs]), expected)
                del expected, slabs
            finally:
                source.close()

    def test_missing_numba_and_unqualified_geometry_keep_dense_projection(self):
        for overrides in ({'numba_missing': True}, {'angle': 15.0}, {'source': 'tile'}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                view = replace(_view('sagittal'), tta_angle_deg=overrides.get('angle', 0.0))
                data = np.ones((view.num_slices, 7, 9), dtype=np.uint8)
                path = root / 'input.cvol'
                write_raw_bbox_mask_store(data, path, format_name=CVOL_FORMAT, desc='input')
                kwargs = _parameters(root, view, int(data.sum()))
                kwargs['source'] = overrides.get('source', 'fullframe')
                with mock.patch.object(assembly, '_numba_sparse_component_accumulate_bounds',
                                       None if overrides.get('numba_missing') else assembly._numba_sparse_component_accumulate_bounds), \
                        mock.patch.object(assembly, 'allocate_workspace_array', wraps=assembly.allocate_workspace_array) as allocate:
                    result = assembly.materialize_interpolation_component_nrrd_view_layer(path, **kwargs)
                self.assertGreaterEqual(allocate.call_count, 1)
                np.testing.assert_array_equal(_read_ref(result, result.shape), data.transpose(1, 0, 2))

    @unittest.skipUnless(assembly._numba is not None, 'optional compiled sparse transpose')
    def test_failed_scatter_discards_partial_store_and_uses_dense_fallback_once(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            view = _view('coronal')
            data = np.ones((view.num_slices, 7, 9), dtype=np.uint8)
            path = root / 'input.cvol'
            write_raw_bbox_mask_store(data, path, format_name=CVOL_FORMAT, desc='input')
            failures = []

            def fail_scatter(*_args):
                # A Mock would retain the mapped source crop in call_args after
                # this failure, artificially extending the input mmap lifetime.
                failures.append(True)
                raise RuntimeError('JIT failed')

            with mock.patch.object(assembly, '_numba_sparse_component_scatter_crop', new=fail_scatter):
                result = assembly.materialize_interpolation_component_nrrd_view_layer(path, **_parameters(root, view, int(data.sum())))
            self.assertEqual(len(failures), 1)
            self.assertTrue(assembly._SPARSE_COMPONENT_NUMBA_DISABLED)
            np.testing.assert_array_equal(_read_ref(result, result.shape), data.transpose(1, 2, 0))
            self.assertFalse(any(child.name.startswith('.') for child in (root / 'nrrd_layers' / view.name).iterdir()))


if __name__ == '__main__':
    unittest.main()
