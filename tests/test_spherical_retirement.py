"""Memory-retiring projection can use idle GPUs without losing native ownership."""
import contextlib
import io
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection as bp, spherical_projection as sp, spherical_projection_cuda as sc
from XTA.spherical_geometry import build_spherical_view_infos
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore, CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT
from tests.test_encoded_mask_store import encode


class SphericalRetirementAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:4,
            mem_get_info=mock.Mock(return_value=(1000,2000))),device=lambda value:value)
        self.coordinator = bp._MainProcessGpuStageCoordinator()
        self.coordinator.configure_workers([0,1,2,3])
        patch=mock.patch.object(bp,'gpu_worker_aux_interpolation_pool',return_value=None)
        patch.start(); self.addCleanup(patch.stop)
        self.purpose='Spherical source projection parent'

    def test_backlog_blocks_and_idle_pressure_opens_without_global_drain(self):
        self.coordinator.set_pending_inference_backlog(True)
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch,self.purpose))
        self.torch.cuda.mem_get_info.assert_not_called()
        self.coordinator.set_pending_inference_backlog(False)
        self.coordinator.begin_inference(0)
        lease=self.coordinator.try_acquire_stage(self.torch,self.purpose)
        self.assertEqual(lease.device_index,1)
        self.assertTrue(self.coordinator.snapshot()['inference_priority_active'])
        lease.release()

    def test_spherical_remains_exclusive_under_overlap_override(self):
        with mock.patch.object(bp,'main_process_gpu_stage_inference_overlap_enabled',return_value=True):
            self.coordinator.set_pending_inference_backlog(False)
            self.coordinator.begin_inference(0)
            self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch,0,self.purpose))
            self.coordinator.finish_inference(0)
            lease=self.coordinator.try_acquire_specific_stage(self.torch,0,self.purpose)
            self.assertIsNotNone(lease)
            self.assertFalse(self.coordinator.can_dispatch_inference(0))
            self.assertFalse(self.coordinator.begin_inference(0))
            self.coordinator.set_pending_inference_backlog(True)
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch,self.purpose))
            lease.release()
            self.assertTrue(self.coordinator.begin_inference(0))

    def test_asset_retirement_and_aux_owner_are_authoritative(self):
        self.coordinator.set_inference_asset_retirement_pending(True)
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch,self.purpose))
        self.coordinator.set_inference_asset_retirement_pending(False)
        aux=SimpleNamespace(revoke_worker=lambda index:False)
        with mock.patch.object(bp,'gpu_worker_aux_interpolation_pool',return_value=aux):
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch,self.purpose))


class SphericalProjectionPromotionTests(unittest.TestCase):
    def setUp(self):
        # Aggregate import tests can stub OpenCV; use the same independent
        # NumPy bbox oracle as the publication-memory and crop-span tests.
        def bbox(a):
            ys, xs = np.nonzero(a)
            return ((int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
                    if len(xs) else (0, 0, 0, 0))
        bbox_patch = mock.patch('XTA.interpolation.cv2.boundingRect', side_effect=bbox)
        bbox_patch.start(); self.addCleanup(bbox_patch.stop)
        patch=mock.patch.dict('os.environ',{'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT':'1'})
        patch.start();self.addCleanup(patch.stop)
        self.output=contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__();self.addCleanup(self.output.__exit__,None,None,None)
        self.view=build_spherical_view_infos(7,9,11,targets=('transverse',),min_radius=.7,patch_size=15,tilted_views=())[0]
        self.source=np.ones((self.view.num_slices,15,15),np.uint8)
        self.shape=(7,9,11)
        self.radii,self.rotation,_,self.boxes=sp._validate_spherical_projection(self.source,self.view,self.shape,None)

    def reference(self,z,count):
        return sp._project_spherical_block(self.source,self.view,self.radii,self.rotation,self.shape,z,count,None)

    def stage(self):
        def encoded(z,count,packed=False):
            records,payload=encode(self.reference(z,count),z,packed)
            return SimpleNamespace(first_z=z,records=records,payload=payload,packed=packed)
        return SimpleNamespace(max_block_depth=2,device_index=0,projector=SimpleNamespace(),
            project=mock.Mock(side_effect=self.reference),project_encoded=mock.Mock(side_effect=encoded),close=mock.Mock())

    def test_cpu_to_cuda_continuation_preserves_raw_packed_and_dense_results(self):
        expected=self.reference(0,self.shape[0])
        for encoding in ('dense','raw','packed'):
            stage=self.stage()
            with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as cleanup:
                writer=None
                target=np.empty(self.shape,np.uint8)
                if encoding!='dense':
                    writer=IncrementalRawBBoxMaskStoreWriter(shape=self.shape,store_dir=Path(directory)/'cvol',
                        format_name=INTERNAL_PACKED_CVOL_FORMAT if encoding=='packed' else CVOL_FORMAT,desc='promotion')
                    cleanup.callback(writer.discard)
                with mock.patch.object(sp,'_try_spherical_cuda_stage',side_effect=[None,stage]) as admit, \
                        mock.patch.object(sp,'_OUTPUT_BLOCK_BYTES',99),mock.patch.object(sp,'_CUDA_RECHECK_SLICES',1), \
                        mock.patch.object(sp,'allocate_workspace_array',return_value=target):
                    result=sp.backproject_spherical_volume_to_volume(self.source,self.view,Path('unused.dat'),'promotion',
                        workers=3,sink_only=writer is not None,projection_block_callback=writer)
                stage.close.assert_called_once()
                self.assertEqual(admit.call_count,2)
                method=stage.project if encoding=='dense' else stage.project_encoded
                self.assertEqual([call.args[0] for call in method.call_args_list],[1,3,5])
                if writer:
                    stats=writer.finalize()
                    with contextlib.closing(RawBBoxMaskStore.open(writer.store_dir,mmap_payload=True)) as store:
                        for z in range(self.shape[0]):store.fill_decoded_slice_into(z,target[z])
                    self.assertEqual(stats['foreground_voxels'],int(expected.sum()))
                else:
                    self.assertIs(result,target)
                np.testing.assert_array_equal(target,expected)

    def test_no_reprobe_after_final_block_and_finite_cpu_only_progress(self):
        for block_bytes,expected_calls in ((10**6,1),(99,7)):
            seen=[]
            with mock.patch.object(sp,'_try_spherical_cuda_stage',return_value=None) as admit, \
                    mock.patch.object(sp,'_OUTPUT_BLOCK_BYTES',block_bytes),mock.patch.object(sp,'_CUDA_RECHECK_SLICES',1):
                sp.backproject_spherical_volume_to_volume(self.source,self.view,Path('unused.dat'),'CPU',
                    sink_only=True,projection_block_callback=lambda z,b:seen.extend(range(z,z+len(b))))
            self.assertEqual(seen,list(range(7)))
            self.assertEqual(admit.call_count,expected_calls)

    def test_cpu_prefetch_finishes_before_promoted_gpu_consumption(self):
        readers=0
        lock=threading.Lock()
        original=sp._project_spherical_block
        def cpu(*args, **kwargs):
            nonlocal readers
            with lock:readers+=1
            try:
                time.sleep(.02)
                return original(*args, **kwargs)
            finally:
                with lock:readers-=1
        stage=self.stage()
        def gpu(z,count):
            self.assertEqual(readers,0)
            return self.reference(z,count)
        stage.project.side_effect=gpu
        with mock.patch.object(sp,'_try_spherical_cuda_stage',side_effect=[None,stage]), \
                mock.patch.object(sp,'_OUTPUT_BLOCK_BYTES',99),mock.patch.object(sp,'_CUDA_RECHECK_SLICES',1), \
                mock.patch.object(sp,'_project_spherical_block',side_effect=cpu):
            sp.backproject_spherical_volume_to_volume(self.source,self.view,Path('unused.dat'),'join',workers=3,
                sink_only=True,projection_block_callback=lambda z,b:None)
        self.assertEqual(readers,0)
        stage.close.assert_called_once()

    def test_failed_promotion_admission_releases_lease_and_preserves_exact_cpu_progress(self):
        expected = self.reference(0, self.shape[0])
        source_before = self.source.copy()
        target = np.empty(self.shape, np.uint8)
        seen = []
        lease = SimpleNamespace(device_index=0, release=mock.Mock())
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

        def consume(z, block):
            seen.extend(range(z, z + len(block)))
            target[z:z + len(block)] = block

        with mock.patch.dict('sys.modules', {'torch': fake_torch}), \
                mock.patch.object(bp, '_try_acquire_main_process_gpu_stage',
                                  side_effect=[None, lease] + [None] * 5) as acquire, \
                mock.patch.object(sc, 'SphericalCudaProjector', side_effect=MemoryError('promotion has no VRAM')) as create, \
                mock.patch.object(sp, '_OUTPUT_BLOCK_BYTES', 99), \
                mock.patch.object(sp, '_CUDA_RECHECK_SLICES', 1):
            sp.backproject_spherical_volume_to_volume(
                self.source, self.view, Path('unused.dat'), 'failed promotion', workers=3,
                sink_only=True, projection_block_callback=consume,
            )
        self.assertEqual(acquire.call_count, 7)
        create.assert_called_once()
        lease.release.assert_called_once()
        self.assertEqual(seen, list(range(self.shape[0])))
        np.testing.assert_array_equal(target, expected)
        np.testing.assert_array_equal(self.source, source_before)

    def test_first_promoted_gpu_sink_failure_joins_producer_and_preserves_exception(self):
        for cleanup_fails in (False, True):
            with self.subTest(cleanup_fails=cleanup_fails):
                producer_started, release_producer, producer_finished = (
                    threading.Event(), threading.Event(), threading.Event()
                )
                events = []
                sink_error = ValueError('first promoted GPU block rejected')
                lease = SimpleNamespace(device_index=0, release=mock.Mock())
                projector = self.stage()
                stage = sp._SphericalCudaStage(projector, lease)
                executor_type = sp.ThreadPoolExecutor

                class JoinSignallingExecutor(executor_type):
                    def __exit__(self, *args):
                        events.append('join')
                        release_producer.set()
                        return super().__exit__(*args)

                def project(z, count):
                    if z == 3:
                        producer_started.set()
                        if not release_producer.wait(2):
                            raise AssertionError('prefetched producer was not joined')
                        producer_finished.set()
                        events.append('producer finished')
                    return self.reference(z, count)

                def consume(z, block):
                    events.append(z)
                    if z == 1:
                        self.assertTrue(producer_started.wait(2))
                        raise sink_error

                def close():
                    self.assertTrue(producer_finished.is_set())
                    events.append('close')
                    if cleanup_fails:
                        raise OSError('secondary cleanup failure')

                projector.project.side_effect = project
                projector.close.side_effect = close
                with mock.patch.object(sp, '_try_spherical_cuda_stage', side_effect=[None, stage]), \
                        mock.patch.object(sp, '_OUTPUT_BLOCK_BYTES', 99), \
                        mock.patch.object(sp, '_CUDA_RECHECK_SLICES', 1), \
                        mock.patch.object(sp, 'ThreadPoolExecutor', JoinSignallingExecutor), \
                        self.assertRaises(ValueError) as caught:
                    sp.backproject_spherical_volume_to_volume(
                        self.source, self.view, Path('unused.dat'), 'promoted sink failure', workers=1,
                        sink_only=True, projection_block_callback=consume,
                    )
                self.assertIs(caught.exception, sink_error)
                self.assertEqual(events, [0, 1, 'join', 'producer finished', 'close'])
                self.assertEqual([call.args[0] for call in projector.project.call_args_list], [1, 3])
                projector.close.assert_called_once()
                lease.release.assert_called_once()
                self.assertIsNone(stage.lease)

    def test_unsafe_promotion_admission_aborts_and_retains_lease_and_projector(self):
        seen = []
        lease = SimpleNamespace(device_index=0, release=mock.Mock())
        projector = self.stage()
        unsafe = sc.SphericalCudaProjectionUnsafeFailure('promotion preflight did not settle', projector)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with mock.patch.dict('sys.modules', {'torch': fake_torch}), \
                mock.patch.object(bp, '_try_acquire_main_process_gpu_stage', side_effect=[None, lease]), \
                mock.patch.object(sc, 'SphericalCudaProjector', side_effect=unsafe), \
                mock.patch.object(sp, '_OUTPUT_BLOCK_BYTES', 99), \
                mock.patch.object(sp, '_CUDA_RECHECK_SLICES', 1), \
                self.assertRaises(sc.SphericalCudaProjectionUnsafeFailure) as caught:
            sp.backproject_spherical_volume_to_volume(
                self.source, self.view, Path('unused.dat'), 'unsafe promotion', workers=3,
                sink_only=True, projection_block_callback=lambda z, block: seen.append(z),
            )
        self.assertIs(caught.exception, unsafe)
        self.assertIs(caught.exception.projector, projector)
        self.assertIs(caught.exception.stage_lease, lease)
        self.assertEqual(seen, [0])
        projector.close.assert_not_called()
        lease.release.assert_not_called()

    def test_unsafe_promoted_cleanup_aborts_and_retains_lease_and_projector(self):
        target = np.empty(self.shape, np.uint8)
        seen = []
        lease = SimpleNamespace(device_index=0, release=mock.Mock())
        projector = self.stage()
        unsafe = sc.SphericalCudaProjectionUnsafeFailure('promoted stream did not settle', projector)
        projector.close.side_effect = unsafe
        stage = sp._SphericalCudaStage(projector, lease)

        def consume(z, block):
            seen.extend(range(z, z + len(block)))
            target[z:z + len(block)] = block

        with mock.patch.object(sp, '_try_spherical_cuda_stage', side_effect=[None, stage]), \
                mock.patch.object(sp, '_OUTPUT_BLOCK_BYTES', 99), \
                mock.patch.object(sp, '_CUDA_RECHECK_SLICES', 1), \
                self.assertRaises(sc.SphericalCudaProjectionUnsafeFailure) as caught:
            sp.backproject_spherical_volume_to_volume(
                self.source, self.view, Path('unused.dat'), 'unsafe promoted cleanup',
                sink_only=True, projection_block_callback=consume,
            )
        self.assertIs(caught.exception, unsafe)
        self.assertIs(caught.exception.projector, projector)
        self.assertIs(caught.exception.stage_lease, lease)
        self.assertIs(stage.lease, lease)
        projector.close.assert_called_once()
        lease.release.assert_not_called()
        self.assertEqual(seen, list(range(self.shape[0])))
        np.testing.assert_array_equal(target, self.reference(0, self.shape[0]))


if __name__=='__main__':unittest.main()
