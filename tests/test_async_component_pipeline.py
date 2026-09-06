"""Real interpolation -> detached publication -> source fusion across geometries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def run_worker(result_path: Path) -> None:
    from threading import Event
    from unittest.mock import patch
    import numpy as np
    from XTA import assembly, finalization, geometry, interpolation, outputs
    from XTA.config import TiltedViewGroup
    from XTA.projection_queue import ComponentProjectionQueue, settle_prepared_view_components
    from XTA.runtime import close_memmap_array

    shape = (13, 17, 19)
    target = (15, 21, 23)
    all_views = geometry.get_view_infos(*shape, cartesian_views=[],
        radial_views=['transverse', 'sagittal', 'coronal', 'tilted_transverse', 'tilted_sagittal', 'tilted_coronal'],
        radial_azimuth_angles=[15.0] * 6,
        tilt_groups=[TiltedViewGroup(('transverse', 'sagittal', 'coronal'), (30.0,), ('vertical',))])
    # Upright Radial and one tilt sign for each radial/orthogonal base.
    selected = [v for v in all_views if not ('minus' in v.name or 'neg' in v.name)]
    rows = []
    assembly.set_final_source_output_shape(target)
    try:
        for view in selected:
            with tempfile.TemporaryDirectory(prefix='xta-async-native-') as temporary:
                root = Path(temporary)
                data = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
                y, x = view.src_h // 2, view.src_w // 2
                data[1:3, y-2:y+2, x-2:x+2] = 1
                data[7:9, y-2:y+2, x-2:x+2] = 1
                shadow_path = root / 'shadow.dat'
                shadow = np.memmap(shadow_path, mode='w+', dtype=np.uint8, shape=data.shape)
                shadow[:] = data
                mapping = shadow._mmap
                release = Event()
                queue = ComponentProjectionQueue(workers=2, max_pending=8,
                    max_source_bytes=1024**3, max_working_bytes=1024**3)
                expected_layers = []
                submitted_refs = []
                class Sink:
                    def submit_layer(self, ref, suffix):
                        submitted_refs.append((ref, suffix))
                def submit(path, **kwargs):
                    def project():
                        if not release.wait(30):
                            raise RuntimeError('Parent did not detach its publication')
                        source = interpolation.RawBBoxMaskStore.open(path, mmap_payload=True)
                        try:
                            decoded = np.stack([source.decode_slice(z) for z in range(source.shape[0])])
                        finally:
                            source.close()
                        # Use the pre-existing dense operator as the numerical oracle.
                        legacy = assembly.project_view_volume_to_orthogonal_volume(
                            decoded, view, root / f'legacy-{len(expected_layers)}.dat',
                            desc='Legacy projection oracle', workers=1,
                            out_shape_tyx=target)
                        expected = np.asarray(legacy).copy()
                        close_memmap_array(legacy)
                        ref = assembly.materialize_interpolation_component_nrrd_view_layer(path, **kwargs)
                        expected_layers.append((ref, expected))
                        return ref
                    return queue.submit(project, source_bytes=1, working_bytes=1)
                with patch.object(assembly, 'nrrd_layer_sink', return_value=Sink()):
                    try:
                        prepared = assembly.prepare_view_volume_after_fullframe(
                            model_name='model', view=view, union_mm=shadow, confmap_mm=None,
                            union_path=shadow_path, confmap_path=None, temp_dir=root,
                            dense_tiling_active=False, min_conf=0, min_radius=0,
                            interpolate=6, interpolation_walk_back=1, interpolation_candidates=1,
                            interpolate_passes=1, interpolate_min_radius=0, interpolation_search_angle=15,
                            keep_temp=False, slice_workers=1, interpolation_task_workers=1,
                            nrrd_layers_enabled=True, precleaned_slice_cleanup=True,
                            hole_fill_done_on_device=True, preinterpolation_layer_already_published=True,
                            submit_component_projection=submit)
                        assert mapping.closed and not shadow_path.exists()
                        assert prepared.final_view_volume_mm is None
                        assert prepared.pending_component_layers
                        assert not settle_prepared_view_components(prepared)
                        release.set()
                        queue.shutdown()
                        assert settle_prepared_view_components(prepared)
                    finally:
                        release.set()
                        queue.abort()
                        queue.shutdown(cancel_futures=True)
                assert len(submitted_refs) == len(prepared.nrrd_layers) == 1
                assert int(prepared.interpolation_stats[0]['added_voxels']) > 0
                for ref, expected in expected_layers:
                    source = outputs._open_nrrd_layer_ref(ref)
                    try:
                        actual = np.stack([outputs._read_layer_slice_in_output_shape(source, target, z)
                                           for z in range(target[0])])
                    finally:
                        outputs._close_nrrd_layer_source(source)
                        outputs._drop_nrrd_raw_store_chunks_ram_cache(source)
                    np.testing.assert_array_equal(actual, expected, err_msg=view.name)
                base = np.zeros(target, np.uint8)
                base[0, 0, 0] = 1
                base_path = root / 'source-base.cvol'
                interpolation.write_raw_bbox_mask_store(base, base_path, workers=1,
                    format_name=interpolation.CVOL_FORMAT, desc='Independent source base')
                base_ref = interpolation.NrrdLayerRef(key='base', name='base', path=base_path,
                    shape=target, storage_format=interpolation.CVOL_FORMAT, model_name='model', view_name=view.name)
                actual = np.zeros(target, np.uint8)
                finalization._union_projected_layer_refs_grouped_into_volume(
                    [base_ref, *prepared.nrrd_layers], actual, workers=1, desc='Detached component fusion')
                expected_union = base.copy()
                for _, expected in expected_layers:
                    expected_union |= expected
                np.testing.assert_array_equal(actual, expected_union, err_msg=view.name)
                assert actual[0, 0, 0] == 1
                rows.append({'view': view.name, 'layers': len(prepared.nrrd_layers),
                             'added': int(prepared.interpolation_stats[0]['added_voxels'])})
    finally:
        assembly.set_final_source_output_shape(None)
    result_path.write_text(json.dumps({'status': 'passed', 'cases': rows}, indent=2))


class AsyncComponentPipelineTests(unittest.TestCase):
    def test_real_detached_components_and_fusion(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='xta-async-integration-') as temporary:
            result_path = Path(temporary) / 'result.json'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', YOLO_TTA_GPU_INTERPOLATION='0',
                YOLO_TTA_GPU_INTERPOLATION_RADIUS='0', YOLO_TTA_GPU_SLICE_LABELING='0',
                YOLO_TTA_GPU_BACKPROJECT='0', YOLO_TTA_GPU_FINAL_FUSION='0',
                YOLO_TTA_INTERPOLATION_PROCESS_BACKEND='0', YOLO_TTA_DELAY_NATIVE_EXPANSION='1',
                YOLO_TTA_TELEMETRY='0', PYTHONPATH=str(repo))
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(result_path)],
                cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', timeout=240)
            self.assertEqual(completed.returncode, 0, completed.stdout[-18000:])
            result = json.loads(result_path.read_text())
            self.assertEqual(result['status'], 'passed')
            self.assertGreaterEqual(len(result['cases']), 9)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        run_worker(Path(sys.argv[2]))
    else:
        unittest.main()
