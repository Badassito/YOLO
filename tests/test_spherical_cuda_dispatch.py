"""Spherical CUDA admission, ordered compact publication and lease ownership."""
import contextlib
import io
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection, spherical_projection as sp, spherical_projection_cuda as sc
from XTA.spherical_geometry import build_spherical_view_infos


class SphericalCudaDispatchTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT': '1'})
        patch.start()
        self.addCleanup(patch.stop)
        stdout = contextlib.redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)
        self.shape = (7, 9, 11)
        self.view = build_spherical_view_infos(*self.shape, targets=('transverse',), min_radius=.7,
                                               patch_size=15, tilted_views=())[0]
        self.source = np.ones((self.view.num_slices, 15, 15), np.uint8)

    def call(self, callback):
        return sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused.dat'), 'dispatch',
                                                         sink_only=True, projection_block_callback=callback)

    def stage(self, project=None, close=None):
        return SimpleNamespace(max_block_depth=2, device_index=0, projector=SimpleNamespace(),
                               project=project or (lambda z, n: np.full((n, *self.shape[1:]), z // 2, np.uint8)),
                               close=close or mock.Mock())

    def test_busy_gpu_retains_cpu_progress(self):
        blocks = []
        with mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=None):
            self.call(lambda z, block: blocks.append(block.copy()))
        self.assertEqual(np.concatenate(blocks).shape, self.shape)
        self.assertTrue(self.source.all())

    def test_dense_and_compact_blocks_are_ordered_before_release(self):
        for packed in (None, False, True):
            seen = []
            stage = self.stage(close=lambda: seen.append('close'))
            if packed is None:
                callback = lambda z, block: seen.append(z)
            else:
                stage.project = mock.Mock(side_effect=AssertionError('dense GPU transfer'))
                stage.project_encoded = lambda z, n, packed=False: SimpleNamespace(
                    first_z=z, packed=packed, records=tuple(range(z, z+n)), payload=np.empty(0, np.uint8))
                callback = mock.Mock(side_effect=AssertionError('dense callback'))
                callback.encoded_slice_format = 'packbits_little' if packed else 'raw_u8'
                callback.consume_encoded_block = lambda z, records, payload, packed=False: seen.append(z)
            with mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=stage):
                self.call(callback)
            self.assertEqual(seen, [0, 2, 4, 6, 'close'])

    def test_compact_block_identity_is_checked_before_sink(self):
        stage = self.stage()
        stage.project_encoded = lambda z, n, packed=False: SimpleNamespace(
            first_z=z, packed=packed, records=(), payload=np.empty(0, np.uint8))
        sink = mock.Mock()
        sink.encoded_slice_format = 'raw_u8'
        with mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=stage), \
                self.assertRaisesRegex(RuntimeError, 'identity/format/count'):
            self.call(sink)
        sink.consume_encoded_block.assert_not_called()
        stage.close.assert_called_once()

    def test_midstream_device_failure_cannot_retry_cpu(self):
        seen = []
        def project(first, count):
            if first:
                raise RuntimeError('device launch failed')
            return np.zeros((count, *self.shape[1:]), np.uint8)
        stage = self.stage(project)
        with mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=stage), \
                mock.patch.object(sp, '_project_spherical_block', side_effect=AssertionError('CPU retry')), \
                self.assertRaisesRegex(RuntimeError, 'device launch failed'):
            self.call(lambda z, block: seen.append(z))
        self.assertEqual(seen, [0])
        stage.close.assert_called_once()

    def test_consumer_failure_joins_next_gpu_producer(self):
        started, finished = threading.Event(), threading.Event()
        def project(first, count):
            if first:
                started.set()
                time.sleep(.03)
                finished.set()
            return np.zeros((count, *self.shape[1:]), np.uint8)
        def consume(first, block):
            self.assertTrue(started.wait(2))
            raise RuntimeError('sink rejected')
        stage = self.stage(project, close=lambda: self.assertTrue(finished.is_set()))
        with mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=stage), \
                self.assertRaisesRegex(RuntimeError, 'sink rejected'):
            self.call(consume)

    def test_allocation_failure_closes_admitted_stage(self):
        stage = self.stage()
        with mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=stage), \
                mock.patch.object(sp, 'allocate_workspace_array', side_effect=OSError('disk full')), \
                self.assertRaisesRegex(OSError, 'disk full'):
            sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused.dat'), 'allocation')
        stage.close.assert_called_once()

    def test_preflight_failure_releases_lease_before_cpu_fallback(self):
        lease = SimpleNamespace(device_index=0, release=mock.Mock())
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with mock.patch.dict('sys.modules', {'torch': fake_torch}), \
                mock.patch.object(backprojection, '_try_acquire_main_process_gpu_stage', return_value=lease), \
                mock.patch.object(sc, 'SphericalCudaProjector', side_effect=MemoryError('no VRAM')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(sp._try_spherical_cuda_stage(self.source, self.view, self.shape, None))
        lease.release.assert_called_once()

    def test_failed_fence_quarantines_lease_and_owners(self):
        lease = SimpleNamespace(device_index=0, release=mock.Mock())
        projector = self.stage()
        unsafe = sc.SphericalCudaProjectionUnsafeFailure('unsettled', projector)
        projector.close = mock.Mock(side_effect=unsafe)
        stage = sp._SphericalCudaStage(projector, lease)
        with self.assertRaises(sc.SphericalCudaProjectionUnsafeFailure) as caught:
            stage.close()
        self.assertIs(caught.exception.projector, projector)
        self.assertIs(caught.exception.stage_lease, lease)
        self.assertIs(stage.lease, lease)
        lease.release.assert_not_called()

    def test_normal_stage_inference_priority_applies(self):
        with mock.patch.object(backprojection, 'main_process_gpu_stage_inference_priority_enabled', return_value=True), \
                mock.patch.object(backprojection, 'main_process_gpu_stage_inference_overlap_enabled', return_value=False), \
                mock.patch.object(backprojection, 'v1613_d1_backprojection_overlap_enabled', return_value=True):
            coordinator = backprojection._MainProcessGpuStageCoordinator()
            coordinator.configure_workers([0, 1])
            torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2, mem_get_info=mock.Mock()))
            self.assertIsNone(coordinator.try_acquire_stage(torch, 'Spherical source projection layer'))
            torch.cuda.mem_get_info.assert_not_called()


if __name__ == '__main__':
    unittest.main()
