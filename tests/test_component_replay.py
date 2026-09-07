"""Persistent capture integrity and CPU projection replay without inference."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from XTA import component_replay, geometry, backprojection
from XTA.interpolation import CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store
from tools import replay_component_projection


class ComponentReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture_root = self.root/'captures'
        self.view = geometry._build_azimuthal_view_info(5, 7, 9, base_view='transverse',
            azimuth_angle=45, azimuthal_native_raster=0, request_token='transverse')
        self.data = np.zeros((self.view.num_slices, self.view.src_h, self.view.src_w), np.uint8)
        self.data[0, 1:4, 1:5] = 1
        self.data[-1, 2, 3:6] = 1
        self.addCleanup(component_replay.configure_component_replay_capture, None)

    def source(self, name='input', fmt=CVOL_FORMAT):
        path = self.root/name
        with contextlib.redirect_stdout(io.StringIO()):
            write_raw_bbox_mask_store(self.data, path, format_name=fmt, desc='capture fixture')
        return path

    def configure(self, **kwargs):
        component_replay.configure_component_replay_capture(self.capture_root, require_persistent=False, **kwargs)

    def capture(self, source, view=None, *, quiet=True):
        with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
            return component_replay.capture_component_projection(source, view=view or self.view,
                out_shape_tyx=(4, 8, 11), added_voxels=int(self.data.sum()),
                layer_metadata={'model': 'test', 'pass_index': 1, 'source': 'fullframe'})

    def test_capture_payload_bytes_and_geometry_survive_source_deletion(self):
        for fmt in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
            source = self.source(fmt, fmt)
            self.capture_root = self.root/f'captures-{fmt}'
            self.configure()
            before = {name: (source/name).read_bytes() for name in ('meta.json', 'index.bin', 'chunks.bin')}
            capture = self.capture(source)
            self.assertIsNotNone(capture)
            shutil.rmtree(source)
            loaded = component_replay.load_component_replay(capture)
            self.assertEqual(loaded.view, self.view)
            self.assertEqual(loaded.output_shape, (4, 8, 11))
            self.assertEqual(loaded.layer_metadata['pass_index'], 1)
            for name, data in before.items():
                self.assertEqual((loaded.source_path/name).read_bytes(), data)
            store = RawBBoxMaskStore.open(loaded.source_path)
            try:
                np.testing.assert_array_equal(np.stack([store.decode_slice(z) for z in range(store.shape[0])]), self.data)
            finally:
                store.close()

    def test_payload_descriptor_and_path_tampering_are_rejected(self):
        self.configure()
        captured = self.capture(self.source())
        path = captured/'input.cvol'/'chunks.bin'
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1])+original[1:])
        with self.assertRaisesRegex(ValueError, 'payload checksum'):
            component_replay.load_component_replay(captured)
        path.write_bytes(original)
        manifest_path = captured/'manifest.json'
        descriptor = json.loads(manifest_path.read_text())
        descriptor['out_shape_tyx'][0] += 1
        manifest_path.write_text(json.dumps(descriptor))
        with self.assertRaisesRegex(ValueError, 'descriptor'):
            component_replay.load_component_replay(captured)
        descriptor['files']['../outside'] = descriptor['files'].pop('input.cvol/meta.json')
        descriptor['descriptor_sha256'] = component_replay._descriptor_digest(descriptor)
        manifest_path.write_text(json.dumps(descriptor))
        with self.assertRaisesRegex(ValueError, 'three local'):
            component_replay.load_component_replay(captured)

    def test_disabled_filter_empty_and_quotas_do_not_copy_unselected_inputs(self):
        component_replay.configure_component_replay_capture(None)
        self.assertIsNone(self.capture(self.root/'missing'))
        self.configure(view_names=['azimuthal_sagittal*'])
        self.assertIsNone(self.capture(self.root/'missing'))
        source = self.source()
        count = sum((source/name).stat().st_size for name in ('meta.json', 'index.bin', 'chunks.bin'))
        self.configure(max_total_bytes=count-1)
        self.assertIsNone(self.capture(source))
        self.assertEqual(component_replay.component_replay_capture_status()['count'], 0)
        self.configure(view_names=['azimuthal_transverse*'], max_captures=1)
        captured = self.capture(source)
        self.assertIsNotNone(captured)
        self.assertIsNone(self.capture(source, replace(self.view, name='azimuthal_transverse_other')))
        # Restarting capture in the same directory does not silently exceed quota.
        self.configure(max_captures=1)
        self.assertIsNone(self.capture(source, replace(self.view, name='another')))

    def test_concurrent_capture_reservations_respect_count_limit(self):
        self.configure(max_captures=1)
        source = self.source()
        variants = [replace(self.view, name=f'view_{i}') for i in range(8)]
        with contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda view: self.capture(source, view, quiet=False), variants))
        self.assertEqual(sum(value is not None for value in results), 1)
        self.assertEqual(component_replay.component_replay_capture_status()['completed'], 1)

    @unittest.skipUnless(bool(getattr(geometry.cv2, '__file__', None)), 'Native OpenCV unavailable')
    def test_real_legacy_and_sparse_cpu_replay_and_streamed_voxel_comparison(self):
        self.configure()
        captured = self.capture(self.source(fmt=INTERNAL_PACKED_CVOL_FORMAT))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.dict(os.environ, {'YOLO_TTA_GPU_BACKPROJECT': '1'}), \
                mock.patch.object(backprojection, '_azimuthal_backproject_gpu_resident', side_effect=AssertionError('unexpected GPU query')), \
                mock.patch.object(backprojection, '_azimuthal_backproject_gpu_streaming', side_effect=AssertionError('unexpected GPU query')):
            legacy = replay_component_projection.execute_backend(captured, self.root/'legacy', 'legacy', 2)
            sparse = replay_component_projection.execute_backend(captured, self.root/'sparse', 'sparse', 2)
        comparison = replay_component_projection.compare_projected_stores(legacy['result_store'], sparse['result_store'])
        self.assertTrue(comparison['exact'])
        self.assertEqual(comparison['changed_voxels'], 0)
        self.assertFalse(legacy['gpu_requested'])
        self.assertFalse(legacy['gpu_used'])
        self.assertGreater(legacy['elapsed_seconds'], 0)
        self.assertGreater(sparse['elapsed_seconds'], 0)
        # Comparison also detects a changed payload rather than trusting stats.
        path = Path(sparse['result_store'])/'chunks.bin'
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        changed = replay_component_projection.compare_projected_stores(legacy['result_store'], sparse['result_store'])
        self.assertFalse(changed['exact'])

    def run_replay_cli_fixture(self, *, keep=False, failure=None, use_default=False):
        self.configure()
        captured = self.capture(self.source())
        output = self.root/'persistent-results'
        scratch = self.root/'local-scratch'
        scratch.mkdir()
        unrelated = scratch/'unrelated'
        unrelated.mkdir()
        (unrelated/'keep.txt').write_text('not owned by replay', encoding='utf-8')
        workspaces = []

        def worker(command, **kwargs):
            workspace = Path(command[command.index('--output') + 1])
            backend = command[command.index('--worker-backend') + 1]
            self.assertTrue(workspace.is_absolute())
            self.assertEqual(workspace.parent.parent, scratch)
            self.assertFalse(workspace.is_relative_to(output))
            self.assertEqual(kwargs['env']['YOLO_TTA_GPU_BACKPROJECT'], '0')
            workspaces.append(workspace)
            workspace.mkdir()
            shutil.copytree(captured/'input.cvol', workspace/'projected.cvol')
            (workspace/'large-workspace-placeholder.u8').write_bytes(b'local projection scratch')
            metrics = dict(requested_backend=backend, elapsed_seconds=1.25,
                           result_store=str(workspace/'projected.cvol'), workspace=str(workspace))
            (workspace/'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
            kwargs['stdout'].write(f'{backend} worker log\n')
            if failure == 'worker' and backend == 'sparse':
                raise subprocess.CalledProcessError(7, command)
            return subprocess.CompletedProcess(command, 0)

        argv = ['replay_component_projection.py', str(captured), '--output', str(output)]
        if not use_default:
            argv.extend(['--scratch', str(scratch)])
        if keep:
            argv.append('--keep-work')
        with mock.patch.object(sys, 'argv', argv), \
             mock.patch.object(replay_component_projection.tempfile, 'gettempdir', return_value=str(scratch)), \
             mock.patch.object(replay_component_projection.subprocess, 'run', side_effect=worker), \
             contextlib.redirect_stdout(io.StringIO()):
            if failure == 'comparison':
                with mock.patch.object(replay_component_projection, 'compare_projected_stores',
                                       side_effect=ValueError('comparison failed')):
                    with self.assertRaisesRegex(ValueError, 'comparison failed'):
                        replay_component_projection.main()
            elif failure == 'worker':
                with self.assertRaises(subprocess.CalledProcessError):
                    replay_component_projection.main()
            else:
                replay_component_projection.main()
        report = json.loads((output/'replay.json').read_text(encoding='utf-8'))
        self.assertEqual(len(workspaces), 2)
        self.assertEqual(workspaces[0].parent, workspaces[1].parent)
        self.assertEqual(Path(report['scratch_root']), workspaces[0].parent)
        self.assertEqual(report['scratch_parent'], str(scratch))
        self.assertEqual(report['backend_workspaces'],
                         {path.name: str(path) for path in workspaces})
        self.assertEqual(report['workspaces_retained'], keep)
        self.assertEqual(Path(report['scratch_root']).exists(), keep)
        self.assertEqual((unrelated/'keep.txt').read_text(encoding='utf-8'), 'not owned by replay')
        self.assertTrue(captured.is_dir())
        for path in workspaces:
            self.assertEqual(path.exists(), keep)
            metrics = output/f'{path.name}.metrics.json'
            self.assertEqual(report['metrics_files'][path.name], str(metrics))
            self.assertEqual(json.loads(metrics.read_text(encoding='utf-8'))['workspace'], str(path))
            self.assertEqual((output/f'{path.name}.log').read_text(encoding='utf-8'),
                             f'{path.name} worker log\n')
        if failure:
            self.assertEqual(report['status'], 'failed')
            self.assertIn('error', report)
            self.assertFalse(report['all_exact'])
        else:
            self.assertEqual(report['status'], 'complete')
            self.assertTrue(report['all_exact'])

    def test_cli_uses_default_local_scratch_and_preserves_metrics_after_cleanup(self):
        self.run_replay_cli_fixture(use_default=True)

    def test_cli_keep_work_records_absolute_retained_locations(self):
        self.run_replay_cli_fixture(keep=True)

    def test_cli_worker_failure_cleans_only_owned_scratch_and_keeps_evidence(self):
        self.run_replay_cli_fixture(failure='worker')

    def test_cli_comparison_failure_also_cleans_workspaces(self):
        self.run_replay_cli_fixture(failure='comparison')

    def test_cli_keep_work_retains_failed_run_for_inspection(self):
        self.run_replay_cli_fixture(keep=True, failure='worker')

    def test_cleanup_refuses_a_directory_outside_the_recorded_scratch_parent(self):
        parent = self.root/'scratch'
        parent.mkdir()
        outside = self.root/'xta-component-replay-outside'
        outside.mkdir()
        (outside/'keep.txt').write_text('keep', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'unowned'):
            replay_component_projection._remove_owned_scratch(outside, parent)
        self.assertEqual((outside/'keep.txt').read_text(encoding='utf-8'), 'keep')

    @unittest.skipUnless(bool(getattr(geometry.cv2, '__file__', None)), 'Native OpenCV unavailable')
    def test_fresh_process_cpu_cli_keeps_results_outside_temporary_workspaces(self):
        self.configure()
        captured = self.capture(self.source(fmt=INTERNAL_PACKED_CVOL_FORMAT))
        output, scratch = self.root/'cli-results', self.root/'cli-scratch'
        command = [sys.executable, '-B', str(Path(replay_component_projection.__file__).resolve()),
                   str(captured), '--output', str(output), '--scratch', str(scratch), '--workers', '1']
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', YOLO_TTA_GPU_BACKPROJECT='0',
                   YOLO_TTA_GPU_INTERPOLATION='0', YOLO_TTA_TELEMETRY='0')
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads((output/'replay.json').read_text(encoding='utf-8'))
        self.assertTrue(report['all_exact'])
        self.assertFalse(report['workspaces_retained'])
        self.assertFalse(Path(report['scratch_root']).exists())
        self.assertEqual(list(scratch.iterdir()), [])
        for backend in ('legacy', 'sparse'):
            self.assertTrue((output/f'{backend}.log').is_file())
            metrics = json.loads((output/f'{backend}.metrics.json').read_text(encoding='utf-8'))
            self.assertFalse(metrics['gpu_used'])
            self.assertEqual(Path(metrics['workspace']).parent, Path(report['scratch_root']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
