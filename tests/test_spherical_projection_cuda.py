"""Opt-in real CUDA qualification plus CPU-only resource failure contracts.

Set XTA_TEST_SPHERICAL_CUDA=1 and point CUPY_CACHE_DIR/NUMBA_CACHE_DIR at the
task's Scratch directory before running this module on an available GPU.
"""
from __future__ import annotations

import contextlib
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from XTA.spherical_projection import _project_spherical_block
from XTA.spherical_projection_cuda import (
    SphericalCudaProjector, SphericalCudaProjectionUnavailable, SphericalCudaProjectionUnsafeFailure,
)


def views(shape, size, minimum, rotation=None):
    result = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=minimum,
                                        patch_size=size, tilted_views=())
    return [replace(v, spherical_rotation_xyz=rotation) for v in result] if rotation else result


def oracle(data, view, shape, boxes=None):
    return _project_spherical_block(data, view, np.asarray(view.spherical_radii),
                                    np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0], boxes)


class SphericalCudaResourceTests(unittest.TestCase):
    def test_unsettled_close_retains_owners_and_is_fatal(self):
        projector = SphericalCudaProjector.__new__(SphericalCudaProjector)
        projector._lock = threading.RLock()
        projector._closed = False
        projector.device_index = 0
        projector._stream = mock.Mock()
        projector._stream.synchronize.side_effect = RuntimeError('fence failed')
        projector._cp = mock.Mock()
        projector._cp.cuda.Device.return_value = contextlib.nullcontext()
        owner = projector._source_gpu = object()
        with self.assertRaises(SphericalCudaProjectionUnsafeFailure) as caught:
            projector.close()
        self.assertIs(caught.exception.projector, projector)
        self.assertIs(projector._source_gpu, owner)
        self.assertFalse(projector._closed)

    def test_unsupported_input_fails_before_cuda_import(self):
        view = views((5, 7, 9), 4, .7)[0]
        for dtype in (np.bool_, np.float32):
            with self.subTest(dtype=dtype), self.assertRaises(SphericalCudaProjectionUnavailable):
                SphericalCudaProjector(np.ones((view.num_slices, 4, 4), dtype), view, (5, 7, 9))


@unittest.skipUnless(os.environ.get('XTA_TEST_SPHERICAL_CUDA') == '1', 'requires explicit available-GPU qualification')
class SphericalCudaProjectionTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('XTA_SPHERICAL_TEST_ROOT'), 'requires a task Scratch output directory')
    def test_real_retirement_promotes_unpublished_slices_before_global_drain(self):
        from XTA import spherical_projection as sp, backprojection as bp
        from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore, CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT
        shape=(9,11,13)
        view=next(v for v in views(shape,15,.2) if v.spherical_face==5)
        data=np.ones((view.num_slices,15,15),np.uint8)
        expected=oracle(data,view,shape)
        self.assertGreater(int(expected[0].sum()),0)
        root=Path(os.environ['XTA_SPHERICAL_TEST_ROOT']).resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix='spherical-promote-',dir=root) as folder:
            for packed in (False,True):
                coordinator=bp._MainProcessGpuStageCoordinator()
                coordinator.configure_workers([0])
                coordinator.set_inference_priority_active(True)
                coordinator.set_pending_inference_backlog(True)
                coordinator.begin_inference(0)
                writer=IncrementalRawBBoxMaskStoreWriter(shape=shape,store_dir=Path(folder)/str(packed),
                    format_name=INTERNAL_PACKED_CVOL_FORMAT if packed else CVOL_FORMAT,desc='real promotion')
                consume=writer.consume
                encoded=writer.consume_encoded_block
                cpu_starts=[]
                gpu_starts=[]
                def cpu(first,block):
                    cpu_starts.append(first)
                    consume(first,block)
                    coordinator.finish_inference(0)
                    coordinator.set_pending_inference_backlog(False)
                def gpu(first,records,payload,**kwargs):
                    self.assertTrue(coordinator.snapshot()['stage_leases'])
                    gpu_starts.append(first)
                    encoded(first,records,payload,**kwargs)
                try:
                    with mock.patch.object(bp,'_MAIN_PROCESS_GPU_STAGE_COORDINATOR',coordinator), \
                            mock.patch.object(sp,'_OUTPUT_BLOCK_BYTES',shape[1]*shape[2]), \
                            mock.patch.object(sp,'_CUDA_RECHECK_SLICES',1), \
                            mock.patch.object(writer,'consume',side_effect=cpu), \
                            mock.patch.object(writer,'consume_encoded_block',side_effect=gpu):
                        sp.backproject_spherical_volume_to_volume(data,view,Path(folder)/'unused.dat','real promotion',
                            workers=2,sink_only=True,projection_block_callback=writer)
                    self.assertEqual(cpu_starts,[0])
                    self.assertEqual(gpu_starts,[1])
                    self.assertTrue(coordinator.snapshot()['inference_priority_active'])
                    self.assertFalse(coordinator.snapshot()['stage_leases'])
                    writer.finalize()
                    with contextlib.closing(RawBBoxMaskStore.open(writer.store_dir)) as store:
                        np.testing.assert_array_equal(np.stack([store.decode_slice(z) for z in range(shape[0])]),expected)
                finally:
                    writer.discard()

    @unittest.skipUnless(os.environ.get('XTA_SPHERICAL_TEST_ROOT'), 'requires a task Scratch output directory')
    def test_real_compact_dispatch_into_existing_cvol_writer(self):
        from XTA import spherical_projection as sp
        from XTA.interpolation import (IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore,
                                        CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT)
        shape = (9, 11, 13)
        view = views(shape, 15, .2)[0]
        rng = np.random.default_rng(345)
        data = rng.integers(0, 2, size=(view.num_slices, 15, 15), dtype=np.uint8)
        expected = oracle(data, view, shape)
        scratch = Path(os.environ['XTA_SPHERICAL_TEST_ROOT']).resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix='spherical-cvol-', dir=scratch) as folder:
            self.assertEqual(Path(folder).resolve().parent, scratch)
            for packed in (False, True):
                path = Path(folder) / ('packed' if packed else 'raw')
                writer = IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=path,
                    format_name=INTERNAL_PACKED_CVOL_FORMAT if packed else CVOL_FORMAT, desc='spherical CUDA test')
                lease = SimpleNamespace(device_index=0, release=mock.Mock())
                projector = SphericalCudaProjector(data, view, shape, block_bytes=11*13*3, reserve_bytes=0)
                stage = sp._SphericalCudaStage(projector, lease)
                try:
                    with mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=stage), \
                            mock.patch.object(projector, 'project', side_effect=AssertionError('dense D2H')), \
                            mock.patch.object(sp, 'allocate_workspace_array', side_effect=AssertionError('dense output')):
                        sp.backproject_spherical_volume_to_volume(data, view, Path(folder) / 'unused.dat',
                            'compact qualification', sink_only=True, projection_block_callback=writer)
                    writer.finalize()
                    with contextlib.closing(RawBBoxMaskStore.open(path)) as store:
                        actual = np.stack([store.decode_slice(z) for z in range(shape[0])])
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(projector.dense_d2h_bytes, 0)
                    lease.release.assert_called_once()
                finally:
                    stage.close()
                    writer.discard()

    def test_direct_gpu_matches_cpu_random_geometry_matrix(self):
        rng = np.random.default_rng(137)
        cases = 0
        for shape, size, minimum in (((7, 9, 11), 15, .7), ((8, 10, 12), 5, .1),
                                      ((3, 5, 7), 8, .2), ((7, 7, 7), 11, 3.),
                                      ((11, 13, 15), 7, .03)):
            for rotation in (None, cube_rotation('vertical', 31), cube_rotation('horizontal', -23)):
                built = views(shape, size, minimum, rotation)
                # Include every face and differing intrinsic patch origins.
                picked = built if len(built) <= 8 else built[::max(1, len(built) // 8)]
                for view in picked:
                    for ph, pw in ((size, size), (max(1, size - 2), max(1, size - 1))):
                        data = (rng.random((view.num_slices, ph, pw)) < .31).astype(np.uint8)
                        for output_shape in (shape, tuple(n + 1 for n in shape)):
                            expected = oracle(data, view, output_shape)
                            with self.subTest(shape=shape, size=size, face=view.spherical_face,
                                              origin=(view.spherical_u_origin, view.spherical_v_origin),
                                              processing=(ph, pw), output=output_shape, rotation=rotation):
                                with SphericalCudaProjector(data, view, output_shape,
                                        block_bytes=output_shape[1] * output_shape[2] * 2, reserve_bytes=0) as p:
                                    actual = np.concatenate([p.project(z, min(2, output_shape[0] - z))
                                                             for z in range(0, output_shape[0], 2)])
                                    np.testing.assert_array_equal(actual, expected)
                            cases += 1
        self.assertGreaterEqual(cases, 360)

    def test_compact_raw_and_packed_crops_are_exact_without_dense_transfer(self):
        rng = np.random.default_rng(521)
        shape = (9, 11, 13)
        for view in views(shape, 15, .2):
            data = np.zeros((view.num_slices, 7, 9), np.uint8)
            data[::2, 1:6, 1:8] = rng.integers(0, 2, size=data[::2, 1:6, 1:8].shape, dtype=np.uint8)
            boxes = np.zeros((view.num_slices, 4), np.int64)
            boxes[::2] = (1, 6, 1, 8)
            expected = oracle(data, view, shape, boxes)
            with SphericalCudaProjector(data, view, shape, boxes, block_bytes=11*13*3,
                                        upload_bytes=7, reserve_bytes=0) as p:
                self.assertEqual(p.source_layout, 'bbox_u8')
                self.assertEqual(p.source_h2d_bytes, len(boxes[::2]) * 5 * 7)
                for packed in (False, True):
                    for first in range(0, shape[0], 3):
                        count = min(3, shape[0] - first)
                        encoded = p.project_encoded(first, count, packed)
                        self.assertEqual(encoded.first_z, first)
                        self.assertEqual(encoded.packed, packed)
                        self.assertFalse(encoded.payload.flags.writeable)
                        p._validate_encoded_preflight(expected[first:first + count], encoded)
                self.assertEqual(p.dense_d2h_bytes, 0)
                self.assertLess(p.payload_d2h_bytes, 2 * expected.nbytes)

    def test_empty_bbox_upload_and_owned_return_blocks(self):
        view = views((7, 9, 11), 15, .7)[0]
        data = np.zeros((view.num_slices, 15, 15), np.uint8)
        boxes = np.zeros((view.num_slices, 4), np.int64)
        with SphericalCudaProjector(data, view, (7, 9, 11), boxes, reserve_bytes=0) as p:
            self.assertEqual(p.source_h2d_bytes, 0)
            empty = p.project_encoded(0, 7, True)
            self.assertEqual(empty.payload.size, 0)
            self.assertTrue(all(record.size == record.foreground == 0 for record in empty.records))
            first = p.project(0, 2)
            second = p.project(2, 2)
            self.assertFalse(np.shares_memory(first, second))
            first[:] = 1
            self.assertFalse(second.any())


if __name__ == '__main__':
    unittest.main()
