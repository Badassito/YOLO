"""CUDA admission, publication and borrowed-buffer lifetime regression checks."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection, cylindrical_projection as cp, geometry


class CylindricalCudaDispatchTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict('os.environ', {'YOLO_TTA_GPU_RADIAL_BACKPROJECT': '1'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.shape = (6, 7, 9)
        self.view = geometry.get_view_infos(
            *self.shape, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.7, radial_patch_size=8,
        )[0]
        self.source = np.ones((self.view.num_slices, self.view.src_h, self.view.src_w), np.uint8)
        self.path = Path(self.temp.name) / 'projection.dat'

    def call(self, callback, **kwargs):
        return cp.backproject_radial_volume_to_volume(
            self.source, self.view, self.path, 'dispatch regression',
            sink_only=True, projection_block_callback=callback, **kwargs,
        )

    def stage(self, project=None, close=None):
        return types.SimpleNamespace(
            max_block_depth=2, device_index=2,
            projector=types.SimpleNamespace(),
            project=project or (lambda z, count: np.full((count, *self.shape[1:]), z // 2, np.uint8)),
            close=close or mock.Mock(),
        )

    def test_busy_gpu_falls_back_to_reference_without_changing_source(self):
        delivered = []
        with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=None), \
                mock.patch.object(cp, '_numba', None):
            self.call(lambda z, block: delivered.append((z, block.copy())))
        self.assertEqual([z for z, _ in delivered], list(range(self.shape[0])))
        self.assertEqual(np.concatenate([b for _, b in delivered]).shape, self.shape)
        np.testing.assert_array_equal(self.source, 1)
        self.assertFalse(self.path.exists())

    def test_cuda_blocks_remain_owned_and_lease_closes_after_last_consumer(self):
        seen = []
        retained = []
        stage = self.stage(close=lambda: seen.append('close'))
        def consume(z, block):
            self.assertNotIn('close', seen)
            retained.append((z, block))
            seen.append(z)
        with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=stage), \
                mock.patch.object(cp, '_project_radial_block', side_effect=AssertionError('CPU entered')):
            self.call(consume)
        self.assertEqual(seen, [0, 2, 4, 'close'])
        for z, block in retained:
            np.testing.assert_array_equal(block, z // 2)

    def test_compact_sink_gets_only_encoded_packets_in_source_order(self):
        for packed in (False, True):
            stage = self.stage(project=mock.Mock(side_effect=AssertionError('dense GPU copy')))
            emitted = []
            def encode(first, count, packed=False):
                return types.SimpleNamespace(first_z=first, packed=packed, records=tuple(range(first,first+count)), payload=np.zeros(0,np.uint8))
            stage.project_encoded = mock.Mock(side_effect=encode)
            sink = mock.Mock(side_effect=AssertionError('dense sink'))
            sink.encoded_slice_format = 'packbits_little' if packed else 'raw_u8'
            sink.consume_encoded_block = lambda first, records, payload, packed=False: emitted.append((first,records,packed))
            with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=stage):
                self.call(sink)
            self.assertEqual(emitted, [(0,(0,1),packed),(2,(2,3),packed),(4,(4,5),packed)])
            stage.project.assert_not_called()
            stage.close.assert_called_once()

    def test_compact_sink_failure_cannot_restart_dense_publication(self):
        stage = self.stage(project=mock.Mock(side_effect=AssertionError('dense GPU retry')))
        stage.project_encoded = lambda first,count,packed=False: types.SimpleNamespace(
            first_z=first,packed=packed,records=tuple(range(first,first+count)),payload=np.zeros(0,np.uint8))
        received = []
        def consume(first, records, payload, packed=False):
            received.append(first)
            if first:
                raise OSError('encoded append failed')
        sink = mock.Mock(side_effect=AssertionError('dense sink retry'))
        sink.encoded_slice_format = 'raw_u8'
        sink.consume_encoded_block = consume
        with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=stage), \
                self.assertRaisesRegex(OSError, 'encoded append failed'):
            self.call(sink)
        self.assertEqual(received,[0,2])
        stage.close.assert_called_once()

    def test_compact_block_count_is_checked_before_publication(self):
        stage = self.stage()
        stage.project_encoded = lambda first,count,packed=False: types.SimpleNamespace(
            first_z=first,packed=packed,records=(),payload=np.zeros(0,np.uint8))
        sink = mock.Mock()
        sink.encoded_slice_format = 'raw_u8'
        with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=stage), \
                self.assertRaisesRegex(RuntimeError, 'identity/format/count'):
            self.call(sink)
        sink.consume_encoded_block.assert_not_called()
        stage.close.assert_called_once()

    def test_unknown_sink_protocol_keeps_dense_callback(self):
        stage = self.stage()
        sink = mock.Mock()
        sink.encoded_slice_format = 'not-a-supported-encoding'
        with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=stage):
            self.call(sink)
        self.assertEqual(sink.call_count,3)
        sink.consume_encoded_block.assert_not_called()

    def test_midstream_gpu_failure_never_restarts_cpu_or_duplicates_callbacks(self):
        delivered = []
        stage = self.stage()
        def project(z, count):
            if z:
                raise RuntimeError('device launch failed')
            return np.zeros((count, *self.shape[1:]), np.uint8)
        stage.project = project
        with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=stage), \
                mock.patch.object(cp, '_project_radial_block', side_effect=AssertionError('CPU entered')), \
                self.assertRaisesRegex(RuntimeError, 'device launch failed'):
            self.call(lambda z, block: delivered.append(z))
        self.assertEqual(delivered, [0])
        stage.close.assert_called_once()

    def test_callback_failure_waits_for_gpu_producer_before_releasing_stage(self):
        running = threading.Event()
        finished = threading.Event()
        seen = []
        def project(z, count):
            if z:
                running.set()
                time.sleep(.08)
                finished.set()
            return np.zeros((count, *self.shape[1:]), np.uint8)
        def consume(z, block):
            self.assertTrue(running.wait(2))
            raise RuntimeError('sink failed')
        def close():
            self.assertTrue(finished.is_set())
            seen.append('closed')
        with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=self.stage(project, close)), \
                self.assertRaisesRegex(RuntimeError, 'sink failed'):
            self.call(consume)
        self.assertEqual(seen, ['closed'])

    def test_output_allocation_failure_releases_admitted_stage(self):
        stage = self.stage()
        with mock.patch.object(cp, '_try_radial_cuda_stage', return_value=stage), \
                mock.patch.object(cp, 'allocate_workspace_array', side_effect=OSError('disk full')), \
                self.assertRaisesRegex(OSError, 'disk full'):
            cp.backproject_radial_volume_to_volume(self.source, self.view, self.path, 'allocation')
        stage.close.assert_called_once()

    def test_wrapper_releases_lease_once_even_if_synchronization_fails(self):
        lease = types.SimpleNamespace(device_index=2, release=mock.Mock())
        projector = self.stage(close=mock.Mock(side_effect=RuntimeError('sync failure')))
        stage = cp._RadialCudaStage(projector, lease)
        with self.assertRaisesRegex(RuntimeError, 'sync failure'):
            stage.close()
        stage.close()
        lease.release.assert_called_once()

    def test_unsettled_stream_quarantines_lease_and_borrowed_owners(self):
        lease = types.SimpleNamespace(device_index=2, release=mock.Mock())
        projector = self.stage()
        failure = cp.RadialCudaProjectionUnsafeFailure('unsettled stream', projector)
        projector.close = mock.Mock(side_effect=failure)
        stage = cp._RadialCudaStage(projector, lease)
        with self.assertRaises(cp.RadialCudaProjectionUnsafeFailure) as caught:
            stage.close()
        self.assertIs(caught.exception.projector, projector)
        self.assertIs(caught.exception.stage_lease, lease)
        self.assertIs(stage.lease, lease)
        lease.release.assert_not_called()

    def test_constructor_failure_releases_admission_before_cpu_fallback(self):
        lease = types.SimpleNamespace(device_index=2, release=mock.Mock())
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        fake_cuda = types.SimpleNamespace(RadialCudaProjector=mock.Mock(side_effect=MemoryError('no VRAM')))
        with mock.patch.dict('sys.modules', {'torch': fake_torch, 'XTA.cylindrical_cuda_projection': fake_cuda}), \
                mock.patch.object(backprojection, '_try_acquire_main_process_gpu_stage', return_value=lease):
            actual = cp._try_radial_cuda_stage(self.source, None, None, self.view, self.shape, None, False)
        self.assertIsNone(actual)
        lease.release.assert_called_once()

    def test_failed_stage_construction_quarantines_an_unsettled_projector(self):
        lease = types.SimpleNamespace(device_index=2, release=mock.Mock())
        projector = self.stage()
        unsafe = cp.RadialCudaProjectionUnsafeFailure('unsettled upload', projector)
        projector.close = mock.Mock(side_effect=unsafe)
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        fake_cuda = types.SimpleNamespace(RadialCudaProjector=mock.Mock(return_value=projector))
        with mock.patch.dict('sys.modules', {'torch': fake_torch, 'XTA.cylindrical_cuda_projection': fake_cuda}), \
                mock.patch.object(backprojection, '_try_acquire_main_process_gpu_stage', return_value=lease), \
                mock.patch.object(cp, '_RadialCudaStage', side_effect=MemoryError('wrapper allocation')), \
                self.assertRaises(cp.RadialCudaProjectionUnsafeFailure) as caught:
            cp._try_radial_cuda_stage(self.source, None, None, self.view, self.shape, None, False)
        self.assertIs(caught.exception.projector, projector)
        self.assertIs(caught.exception.stage_lease, lease)
        lease.release.assert_not_called()

    def test_retirement_barrier_blocks_radial_without_probing_or_allocating(self):
        coordinator = backprojection._MainProcessGpuStageCoordinator()
        coordinator.configure_workers([0, 1])
        coordinator.set_inference_asset_retirement_pending(True)
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device_count=lambda: 2, mem_get_info=mock.Mock()),
        )
        self.assertIsNone(coordinator.try_acquire_stage(torch, 'Radial backprojection'))
        torch.cuda.mem_get_info.assert_not_called()

    def test_radial_uses_inference_priority_even_with_legacy_backprojection_overlap(self):
        with mock.patch.object(backprojection, 'main_process_gpu_stage_inference_priority_enabled', return_value=True), \
                mock.patch.object(backprojection, 'main_process_gpu_stage_inference_overlap_enabled', return_value=False), \
                mock.patch.object(backprojection, 'v1613_d1_backprojection_overlap_enabled', return_value=True):
            coordinator = backprojection._MainProcessGpuStageCoordinator()
            coordinator.configure_workers([0, 1])
            torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(device_count=lambda: 2, mem_get_info=mock.Mock()),
            )
            self.assertIsNone(coordinator.try_acquire_stage(torch, 'Radial source projection layer'))
            torch.cuda.mem_get_info.assert_not_called()


if __name__ == '__main__':
    unittest.main()
