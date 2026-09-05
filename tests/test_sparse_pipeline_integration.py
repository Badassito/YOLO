"""Real CPU interpolation, component retirement and source-space fusion integration.

A clean subprocess keeps numerical dependencies independent of the lightweight
stubs used by other contract tests. All production functions run unchanged.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def _run_real_pipeline(output_path: Path) -> None:
    required = ('numpy', 'cv2', 'scipy', 'tifffile', 'tqdm')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        output_path.write_text(json.dumps({'skipped': f'Missing native dependencies: {missing}'}) + '\n')
        return

    import numpy as np
    from XTA import assembly, finalization, interpolation, outputs
    from XTA.geometry import ViewInfo

    def read_ref(ref, shape):
        source = outputs._open_nrrd_layer_ref(ref)
        try:
            return np.stack([outputs._read_layer_slice_in_output_shape(source, shape, z)
                             for z in range(shape[0])])
        finally:
            outputs._close_nrrd_layer_source(source)
            outputs._drop_nrrd_raw_store_chunks_ram_cache(source)

    working_shape = (13, 17, 19)
    rows = []
    for orientation in ('transverse', 'sagittal', 'coronal'):
        t, h, w = working_shape
        num, native_h, native_w = {'transverse': (t, h, w), 'sagittal': (h, t, w), 'coronal': (w, t, h)}[orientation]
        view = ViewInfo(name=f'{orientation}__tta_a0', physical_view_name=orientation,
                        num_slices=num, src_h=native_h, src_w=native_w, pad_mode='clamp',
                        full_t=t, full_h=h, full_w=w, family='orthogonal', tta_aug_id='a0')
        for native_xy in (False, True):
            source_shape = (23, 17, 19) if native_xy else (23, 29, 31)
            assembly.set_final_source_output_shape(source_shape)
            plane_shape = (native_h, native_w) if native_xy else (9, 11)
            for empty in (False, True):
                with tempfile.TemporaryDirectory(prefix='xta-pipeline-integration-') as temporary:
                    root = Path(temporary)
                    initial = np.zeros((num, *plane_shape), dtype=np.uint8)
                    if not empty:
                        initial[1:3, 2:5, 3:6] = 1
                        initial[7:9, 2:5, 3:6] = 1
                    shadow_store = root / 'shadow.cvol'
                    interpolation.write_raw_bbox_mask_store(initial, shadow_store,
                        format_name=interpolation.CVOL_FORMAT, desc='Pipeline-test D1 shadow', workers=1)
                    shadow_path = root / 'view_shadow.u8.dat'
                    shadow = interpolation.materialize_raw_bbox_mask_store_workspace(
                        shadow_store, shadow_path, desc='Pipeline-test shadow materialization', workers=1)
                    shadow_mapping = shadow._mmap
                    present = np.any(initial, axis=(1, 2))
                    boxes = np.zeros((num, 4), dtype=np.int64)
                    for z in np.flatnonzero(present):
                        yy, xx = np.nonzero(initial[z])
                        boxes[z] = (yy.min(), yy.max() + 1, xx.min(), xx.max() + 1)
                    slice_meta = {'valid': True, 'slice_any': present, 'slice_bboxes': boxes}

                    # D1 publishes its source-space base independently. The sentinel
                    # deliberately has no counterpart in the view-native shadow.
                    base = np.zeros(source_shape, dtype=np.uint8)
                    finalization.assemble_view_volumes_into_native_union(
                        base, {orientation: initial}, *working_shape,
                        out_shape_tyx=source_shape, workers=1)
                    base[0, 0, 0] = 1
                    base_path = root / 'published_source_base.cvol'
                    interpolation.write_raw_bbox_mask_store(base, base_path,
                        format_name=interpolation.CVOL_FORMAT, desc='Pipeline-test D1 base', workers=1)
                    base_ref = interpolation.NrrdLayerRef(
                        key='source-base', name='source-base', path=base_path, shape=source_shape,
                        storage_format=interpolation.CVOL_FORMAT, model_name='model',
                        view_name=view.name, physical_view_name=orientation)

                    result = assembly.prepare_view_volume_after_fullframe(
                        model_name='model', view=view, union_mm=shadow, confmap_mm=None,
                        union_path=shadow_path, confmap_path=None, temp_dir=root,
                        dense_tiling_active=False, min_conf=0, min_radius=0,
                        interpolate=6, interpolation_walk_back=1, interpolation_candidates=1,
                        interpolate_passes=2, interpolate_min_radius=0, interpolation_search_angle=15,
                        keep_temp=False, slice_workers=1, interpolation_task_workers=1,
                        nrrd_layers_enabled=True, precleaned_slice_cleanup=True,
                        hole_fill_done_on_device=True, slice_meta=slice_meta,
                        preinterpolation_layer_already_published=True)
                    assert shadow_mapping.closed, 'Original shadow mapping remains live'
                    assert not shadow_path.exists(), 'Retired shadow path survives'
                    assert result.final_view_volume_mm is None, 'A redundant dense additions canvas survived'
                    assert result.native_support_mm is None
                    assert result.nrrd_layers, 'Component output was lost'

                    # Independent reconstruction reads each component as an output
                    # raster and performs a plain NumPy OR with the known source base.
                    expected = base.copy()
                    component_voxels = []
                    for ref in result.nrrd_layers:
                        assert ref.path.exists(), f'Component backing disappeared: {ref.path}'
                        raster = read_ref(ref, source_shape)
                        expected |= raster
                        component_voxels.append(int(np.count_nonzero(raster)))
                        if native_xy:
                            assert tuple(ref.shape[-2:]) == source_shape[-2:]
                            assert int(ref.shape[0]) != source_shape[0]

                    actual = np.zeros(source_shape, dtype=np.uint8)
                    finalization._union_projected_layer_refs_grouped_into_volume(
                        [base_ref, *result.nrrd_layers], actual, workers=1,
                        desc='Pipeline-test grouped refs fusion')
                    np.testing.assert_array_equal(actual, expected)
                    assert actual[0, 0, 0] == 1, 'Independent source base was discarded'
                    added = sum(int(stats.get('added_voxels', 0)) for stats in result.interpolation_stats)
                    if empty:
                        assert added == 0
                        assert len(result.interpolation_stats) == 1
                        assert not any(component_voxels)
                        np.testing.assert_array_equal(actual, base)
                    else:
                        assert added > 0, 'Fixture did not create a real bridge'
                        assert len(result.interpolation_stats) == 2, 'Multipass continuation was not exercised'
                        assert int(np.count_nonzero(actual)) > int(np.count_nonzero(base))
                    assert len(result.nrrd_layers) == len(result.interpolation_stats)
                    assert base_path.exists()
                    assert all(ref.path.exists() for ref in result.nrrd_layers)
                    rows.append(dict(orientation=orientation, empty=empty, native_xy=native_xy,
                                     input_shape=list(initial.shape), source_shape=list(source_shape),
                                     added_voxels=added, passes=len(result.interpolation_stats),
                                     component_voxels=component_voxels,
                                     final_voxels=int(np.count_nonzero(actual))))
    assembly.set_final_source_output_shape(None)
    output_path.write_text(json.dumps({'status': 'passed', 'cases': rows}, indent=2) + '\n')


class SparsePipelineIntegrationTests(unittest.TestCase):
    def test_real_interpolation_retirement_and_fusion_across_cartesian_grids(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='xta-integration-run-') as temporary:
            root = Path(temporary)
            result_path = root / 'result.json'
            environment = os.environ.copy()
            environment.update({
                'CUDA_VISIBLE_DEVICES': '',
                'YOLO_TTA_GPU_INTERPOLATION': '0',
                'YOLO_TTA_GPU_INTERPOLATION_RADIUS': '0',
                'YOLO_TTA_GPU_SLICE_LABELING': '0',
                'YOLO_TTA_GPU_FINAL_FUSION': '0',
                'YOLO_TTA_INTERPOLATION_PROCESS_BACKEND': '0',
                'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1',
                'YOLO_TTA_TELEMETRY': '0',
                'YOLO_TTA_NRRD_MEMBER_CODEC': 'zlib',
                'PYTHONPATH': str(repo),
                'NUMBA_CACHE_DIR': str(root / 'numba-cache'),
            })
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), '--worker', str(result_path)],
                cwd=repo, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', timeout=240)
            self.assertEqual(completed.returncode, 0, completed.stdout[-20000:])
            result = json.loads(result_path.read_text())
            if 'skipped' in result:
                self.skipTest(result['skipped'])
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(len(result['cases']), 12)
            self.assertEqual(sum(case['passes'] == 2 for case in result['cases']), 6)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        _run_real_pipeline(Path(sys.argv[2]))
    else:
        unittest.main()
