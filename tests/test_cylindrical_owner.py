"""Exact native cleanup, chunk gather and owner lifetime qualification."""
from dataclasses import replace
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cylindrical_owner as owner, geometry
from XTA.inference import _fill_holes_2d_scipy
from tools.probe_radial_streaming_architecture import decode_words, oracle


class RadialOwnerContractTests(unittest.TestCase):
    def test_active_native_owner_blocks_worker_asset_retirement(self):
        from XTA import workers,cuda_d1
        asset=workers._GpuWorkerInferenceAssets(model=object())
        with mock.patch.object(workers,'active_radial_owners',return_value=(('model','radial'),)), \
                mock.patch.dict(cuda_d1._D1_WORKER_VIEW_STATES,{},clear=True), \
                self.assertRaisesRegex(RuntimeError,'active D1'):
            workers._release_gpu_worker_inference_assets(asset,inference_drained=True,wait_for_publications=lambda:None)
        self.assertFalse(asset.release_started)

    def test_shape_only_target_rejects_host_fallback_and_incomplete_predictions(self):
        target=owner.DeviceOnlyRadialTarget((3,3072,3072))
        self.assertEqual(target.shape,(3,3072,3072))
        with self.assertRaisesRegex(RuntimeError,'materializing'):
            np.asarray(target)
        with self.assertRaisesRegex(RuntimeError,'host-mask fallback'):
            target[0]
        with self.assertRaises(IndexError):target.claim_result(-1)
        target.claim_result(1)
        with self.assertRaisesRegex(RuntimeError,'duplicated'):
            target.claim_result(1)
        with self.assertRaisesRegex(RuntimeError,'every radius'):
            target.require_complete()
        target.claim_result(2);target.claim_result(0);target.require_complete()

    def test_current_workload_admitted_and_dependencies_retain_compatibility(self):
        view = geometry.get_view_infos(7, 9, 11, cartesian_views=(), radial_views=('transverse',),
                                       radial_patch_size=8)[0]
        kwargs = dict(d1_active=True, cpu_workers=False, kind='fullframe', interpolation=0,
                      tiled=False, angle_count=1, min_conf=0, min_radius=0, batch=1, gray=True)
        with mock.patch.dict('os.environ', {'YOLO_TTA_RADIAL_OWNER':'1'}):
            self.assertTrue(owner.radial_owner_eligible(view, **kwargs))
            for key, value in [('d1_active',False), ('cpu_workers',True), ('kind','tile'),
                               ('interpolation',1), ('tiled',True), ('angle_count',2),
                               ('min_conf',.1), ('min_radius',1), ('batch',2), ('gray',False), ('retain_native',True)]:
                self.assertFalse(owner.radial_owner_eligible(view, **{**kwargs,key:value}))
            self.assertFalse(owner.radial_owner_eligible(replace(view, tta_angle_deg=10), **kwargs))
        with mock.patch.dict('os.environ', {'YOLO_TTA_RADIAL_OWNER':'0'}):
            self.assertFalse(owner.radial_owner_eligible(view, **kwargs))

    def test_bucket_partition_preserves_every_owned_plane_position(self):
        shells = np.array([-1,2,0,1,0,-1,2,2],np.int32)
        for build in (owner._bucket_shell_pixels, owner._bucket_shell_pixels_compiled, owner._bucket_shell_pixels_numpy):
            if build is None:
                continue
            offsets, pixels = build(shells,4)
            self.assertEqual(offsets.tolist(), [0,2,3,6,6])
            self.assertEqual(pixels.tolist(), [2,4,3,1,6,7])


@unittest.skipUnless(os.environ.get('XTA_RUN_CUDA_RADIAL_OWNER') == '1', 'explicit native owner CUDA qualification')
class RadialOwnerCudaTests(unittest.TestCase):
    def test_optional_bucket_compilation_failure_keeps_exact_cuda_projection(self):
        import cupy as cp
        view=geometry.get_view_infos(5,7,9,cartesian_views=(),radial_views=('transverse',),
                                    radial_min_radius=.1,radial_patch_size=8)[0]
        source=np.ones((view.num_slices,8,8),np.uint8)
        with mock.patch.object(owner,'_bucket_shell_pixels_compiled',side_effect=RuntimeError('compiler unavailable')):
            active=owner.RadialOwner(view,(8,8),(5,7,9),reserve_bytes=0)
        try:
            active.consume(0,cp.asarray(source))
            np.testing.assert_array_equal(decode_words(active.host_words(),(5,7,9)),oracle(source,view,(5,7,9)))
        finally:active.close()

    def test_worker_consumer_seals_once_and_publishes_real_source_store(self):
        import cupy as cp
        from XTA import cuda_d1, interpolation
        view=geometry.get_view_infos(9,11,13,cartesian_views=(),radial_views=('transverse',),
                                    radial_min_radius=.1,radial_patch_size=8)[0]
        source=np.zeros((view.num_slices,8,8),np.uint8)
        source[:,1:7,1:7]=1;source[:,2:6,2:6]=0
        clean=np.stack([_fill_holes_2d_scipy(x) for x in source]).astype(np.uint8)
        expected=oracle(clean,view,(7,9,11))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'layer.cvol'
            task=dict(projection_contract=owner.RADIAL_OWNER_CONTRACT,result_mode='d1_owner',
                      view=view,model_name='test',d1_output_shape=(7,9,11),d1_store_dir=str(path))
            try:
                for first in range(0,view.num_slices,2):
                    device=cp.asarray(source[first:first+2])
                    target=owner.DeviceOnlyRadialTarget(device.shape)
                    for i in range(len(device)):target.claim_result(i)
                    accumulator=SimpleNamespace(union_dev=device,host_written=False)
                    current=pickle.loads(pickle.dumps({**task,'slice_start':first,'slice_count':len(device)}))
                    result=owner.consume_radial_device_union(current,accumulator,target=target)
                    self.assertIsNone(accumulator.union_dev)
                    self.assertEqual(result['d1_covered_slices'],first+len(device))
                    if first+len(device)<view.num_slices:
                        self.assertNotIn('_publication_future',result)
                        self.assertEqual(len(owner.active_radial_owners()),1)
                self.assertFalse(owner.active_radial_owners())
                published=result['_publication_future'].result(timeout=10)
                self.assertEqual(published['d1_layer_ref'].view_family,'radial')
                store=interpolation.RawBBoxMaskStore.open(path)
                try:
                    np.testing.assert_array_equal(np.stack([store.decode_slice(z) for z in range(7)]),expected)
                finally:store.close()
            finally:
                owner.shutdown_radial_owners();cuda_d1._shutdown_d1_worker_pipeline()

    def test_cleanup_and_chunk_projection_match_native_cpu_oracle(self):
        import cupy as cp
        rng = np.random.default_rng(543)
        checked = 0
        for base in ('transverse','sagittal','coronal'):
            views = geometry.get_view_infos(7,9,11,cartesian_views=(),radial_views=(base,),
                                           radial_min_radius=.01,radial_patch_size=8)
            for initial in (views[0], views[-1]):
                for direction, angle in (('',0),('vertical',-30),('vertical',30),('horizontal',-30),('horizontal',30)):
                    view = replace(initial,radial_tilted_source=bool(direction),tilt_direction=direction,tilt_angle_deg=angle)
                    for shape in ((7,9,11),(5,7,8)):
                        source = (rng.random((view.num_slices,6,7)) > .65).astype(np.uint8)
                        source[:,1:5,1:6] = 1
                        source[:,2:4,2:5] = 0
                        source[::3] = 0
                        expected_clean = np.stack([_fill_holes_2d_scipy(x) for x in source]).astype(np.uint8)
                        expected = oracle(expected_clean,view,shape)
                        active = owner.RadialOwner(view,source.shape[1:],shape,reserve_bytes=0)
                        try:
                            device = cp.asarray(source)
                            for first in reversed(range(0,len(source),2)):
                                active.consume(first,device[first:first+2])
                            np.testing.assert_array_equal(device.get(),expected_clean)
                            np.testing.assert_array_equal(decode_words(active.host_words(),shape),expected)
                            with self.assertRaisesRegex(ValueError,'duplicate'):
                                active.consume(0,device[:1])
                            checked += expected.size
                        finally:
                            active.close()
                        self.assertEqual(active._pool.total_bytes(),0)
        print(f'Native Radial owner exact CUDA output: {checked} voxels.',flush=True)

    def test_cleanup_background_connectivity_border_and_large_roi(self):
        import cupy as cp
        rng=np.random.default_rng(51)
        view=geometry.get_view_infos(7,9,11,cartesian_views=(),radial_views=('transverse',),
                                    radial_min_radius=.1,radial_patch_size=8)[0]
        active=owner.RadialOwner(view,(129,131),(7,9,11),reserve_bytes=0)
        try:
            cases=[np.zeros((129,131),np.uint8),np.ones((129,131),np.uint8),
                   (rng.random((129,131))>.4).astype(np.uint8)]
            ring=np.zeros((129,131),np.uint8);ring[0:128,0:130]=1;ring[1:127,1:129]=0
            cases += [ring,ring.copy()]
            cases[-1][0,1]=0
            for data in cases:
                with cp.cuda.Device(0),cp.cuda.using_allocator(active._pool.malloc),active._stream:
                    device=cp.asarray(data[None]);boxes=active._boxes(device)
                    active._clean(device,boxes);active._fence()
                    np.testing.assert_array_equal(device.get()[0],_fill_holes_2d_scipy(data).astype(np.uint8))
        finally:
            active.close()

    def test_production_size_sparse_background_cleanup(self):
        import cupy as cp
        view=geometry.get_view_infos(7,9,11,cartesian_views=(),radial_views=('transverse',),
                                    radial_min_radius=.1,radial_patch_size=8)[0]
        active=owner.RadialOwner(view,(3072,3072),(7,9,11),reserve_bytes=0)
        try:
            data=np.zeros((3072,3072),np.uint8)
            data[0,:]=data[-1,:]=data[:,0]=data[:,-1]=1
            for open_border in (False,True):
                if open_border:data[0,4]=0
                with cp.cuda.Device(0),cp.cuda.using_allocator(active._pool.malloc),active._stream:
                    device=cp.asarray(data[None]);boxes=active._boxes(device)
                    active._clean(device,boxes);active._fence()
                    np.testing.assert_array_equal(device.get()[0],_fill_holes_2d_scipy(data).astype(np.uint8))
        finally:
            active.close()

    def test_incomplete_and_failed_coverage_cannot_publish(self):
        import cupy as cp
        view=geometry.get_view_infos(5,7,9,cartesian_views=(),radial_views=('transverse',),
                                    radial_min_radius=.1,radial_patch_size=8)[0]
        active=owner.RadialOwner(view,(8,8),(5,7,9),reserve_bytes=0)
        try:
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                active.host_words()
            with mock.patch.object(active,'_clean',side_effect=RuntimeError('cleanup fault')):
                with self.assertRaisesRegex(RuntimeError,'cleanup fault'):
                    active.consume(0,cp.ones((1,8,8),cp.uint8))
            self.assertFalse(active.coverage.any())
            with self.assertRaisesRegex(RuntimeError,'failed'):
                active.host_words()
        finally:
            active.close()


if __name__ == '__main__':
    unittest.main()
