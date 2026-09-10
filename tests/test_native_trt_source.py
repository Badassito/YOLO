"""Opt-in native TensorRT sources retain their generic CUDA input pixels."""
from dataclasses import replace
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cuda_backend as cb, geometry
from XTA.config import resolve_channel_format
from XTA.cylindrical_geometry import build_radial_view_infos
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation


def native_views(shape=(17, 19, 21), size=16):
    radial = build_radial_view_infos(*shape, targets=('transverse', 'sagittal', 'coronal'),
        min_radius=.5, patch_size=size, tilted_views=())
    chosen = [next(v for v in radial if v.radial_base_view == base)
              for base in ('transverse', 'sagittal', 'coronal')]
    chosen.append(replace(chosen[0], name=chosen[0].name + '_tilted',
        radial_tilted_source=True, tilt_direction='vertical', tilt_angle_deg=30.))
    spherical = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=.5,
        patch_size=size, tilted_views=())
    chosen.extend(next(v for v in spherical if v.spherical_face == face) for face in range(6))
    chosen.append(replace(chosen[-1], name=chosen[-1].name + '_rotated',
        spherical_rotation_xyz=cube_rotation('horizontal', -30)))
    return chosen


def make_source(engine, physical, *, tiled=False, channels='gray', fp16=False,
                batch_size=1, num_frames=None, angle=0., size=16):
    view = geometry.expand_views_into_tta_variants((physical,), (angle,))[0]
    job = (SimpleNamespace(M_out_to_src=np.asarray(((.83, .11, 2.), (-.07, .91, -1.)), np.float32))
           if tiled else geometry.build_aug_job_for_variant(view, size, Path('unused')))
    cls = cb.GpuTileRenderedYoloSource if tiled else cb.GpuRenderedYoloSource
    with (mock.patch.object(cb, 'ensure_ultralytics_accepts_in_memory_volume_source'),
          mock.patch.dict('sys.modules', {'ultralytics.data.loaders': None})):
        return cls(engine, view, job, slice_offset=0,
            num_frames=view.num_slices if num_frames is None else num_frames,
            batch_size=batch_size, out_size=size, fp16=fp16, name='native-trt-source',
            channel_format=resolve_channel_format(channels))


class NativeTrtSourceContracts(unittest.TestCase):
    def test_optin_is_independent_of_geometry_and_requires_resident_batch_one(self):
        views = native_views()
        for family in ('radial', 'spherical'):
            view = next(v for v in views if v.family == family)
            for tiled in (False, True):
                for enabled, resident, batch, count in (
                    ('0', True, 1, 3), ('1', True, 1, 3),
                    ('1', False, 1, 3), ('1', True, 2, 3), ('1', True, 1, 0),
                ):
                    engine = SimpleNamespace(_mode='resident' if resident else 'stream')
                    # This checks admission only; numerical affine/render checks
                    # below use real OpenCV in their separate CUDA test process.
                    with (mock.patch.dict(os.environ, {'YOLO_TTA_NATIVE_TRT_RING': enabled,
                                                       'YOLO_TTA_FAST_GEOMETRY': '1'}),
                          mock.patch.object(geometry, 'build_aug_job_for_variant',
                                            return_value=SimpleNamespace(aff=None))):
                        source = make_source(engine, view, tiled=tiled, batch_size=batch, num_frames=count)
                    with self.subTest(family=family, tiled=tiled, enabled=enabled,
                                      resident=resident, batch=batch, count=count):
                        self.assertEqual(source.resident_ring_supported,
                                         enabled == '1' and resident and batch == 1 and count > 0)
                        if not source.resident_ring_supported:
                            with self.assertRaisesRegex(RuntimeError, 'resident.*ring is unsupported'):
                                source.prepare_direct_ring()

    def test_native_flag_defaults_off_and_does_not_enable_legacy_fused_geometry(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(cb.native_trt_ring_enabled())
        engine = object.__new__(cb._GpuWorkerRenderEngine)
        with mock.patch.dict(os.environ, {'YOLO_TTA_NATIVE_TRT_RING': '1'}):
            for view in native_views():
                self.assertEqual(cb._fused_preflight_family(view), '')
                self.assertFalse(engine._try_fused_render_into_ring_slot(None, view, None, 0, 16))


@unittest.skipUnless(os.environ.get('XTA_TEST_NATIVE_TRT_SOURCE_CUDA') == '1',
                     'requires explicit available-GPU source qualification')
class NativeTrtSourceCudaTests(unittest.TestCase):
    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        self.torch = torch
        self.addCleanup(torch.cuda.synchronize)
        flags = mock.patch.dict(os.environ, {'YOLO_TTA_NATIVE_TRT_RING': '1'})
        flags.start(); self.addCleanup(flags.stop)
        self.volume = np.random.default_rng(27018).integers(0, 256, (11, 19, 21), np.uint8)
        self.views = native_views()

    def engine(self):
        from tests.test_cylindrical_cuda import resident_engine
        engine = resident_engine(self.volume, 'cuda:0', logical_t=17)
        engine._stream = self.torch.cuda.Stream(device=engine.device)
        engine._stream.wait_stream(self.torch.cuda.current_stream(engine.device))
        return engine

    def reference(self, source, index, dtype):
        # The production source obeys --quantize before BasePredictor casts to
        # the actual engine binding dtype. These can intentionally disagree.
        fp16 = bool(source.fp16)
        if isinstance(source, cb.GpuTileRenderedYoloSource):
            tensor, ready = source.engine.render_tile_batch(source.view, source.tile_affine, (index,),
                out_size=source.out_size, fp16=fp16, channel_format=source.channel_format)
        else:
            tensor, ready = source.engine.render_fullframe_batch(source.view, source.job, (index,),
                source.out_size, fp16, channel_format=source.channel_format)
        ready.synchronize()
        return tensor.to(dtype=dtype)

    def test_slots_match_generic_render_bytes_for_native_geometry_channels_and_binding_dtypes(self):
        torch = self.torch
        cases = frames = 0
        for fast in ('0', '1'):
            with mock.patch.dict(os.environ, {'YOLO_TTA_FAST_GEOMETRY': fast,
                                              'YOLO_TTA_GPU_SPHERICAL_FP32': fast}):
                engine = self.engine()
                for vi, physical in enumerate(self.views):
                    if fast == '1' and physical.family != 'spherical':
                        continue
                    for tiled in (False, True):
                        for channels in ('gray', 'RGB', 'C3S1', 'C5S2'):
                            for dtype in (torch.float16, torch.float32):
                                source = make_source(engine, physical, tiled=tiled, channels=channels,
                                    fp16=dtype == torch.float32, angle=0. if vi % 2 else 37.)
                                with self.subTest(view=physical.name, tiled=tiled, channels=channels,
                                                  dtype=dtype, fast=fast):
                                    # Native inputs must not borrow fused Azimuthal/Tilted geometry.
                                    with (mock.patch.object(engine, 'validate_fused_ring_renderer',
                                              side_effect=AssertionError('legacy fused validation')),
                                          mock.patch.object(engine, 'capture_fused_ring_renderer',
                                              side_effect=AssertionError('legacy fused graph'))):
                                        slots = source.prepare_direct_ring(input_dtype=dtype)
                                    pointers = [slot.input.data_ptr() for slot in slots]
                                    self.assertNotEqual(*pointers)
                                    self.assertTrue(all(slot.input.dtype == dtype for slot in slots))
                                    for index in range(source.nf):
                                        local, slot = source.next_direct_slot()
                                        slot.render_done.synchronize()
                                        self.assertEqual((local, slot.absolute_index), (index, index))
                                        self.assertEqual(slot.input.data_ptr(), pointers[index & 1])
                                        expected = self.reference(source, index, dtype)
                                        self.assertTrue(torch.equal(slot.input.view(torch.uint8),
                                                                    expected.view(torch.uint8)))
                                        frames += 1
                                    self.assertIsNone(source.next_direct_slot())
                                    self.assertEqual(source._direct_count, source.nf)
                                    if physical.family == 'spherical':
                                        self.assertEqual(engine._spherical_sampler_mode,
                                            'fp32_virtual_cube' if fast == '1' else 'reference_fp64')
                                    source.reset_direct_ring()
                                    self.assertIsNone(source._direct_ring)
                                    self.assertEqual((source._direct_count, source.count), (0, 0))
                                    source.close()
                                    cases += 1
                engine._stream.synchronize()
        print(f'Native TRT source parity: {cases} configurations, {frames} frames, byte-identical inputs.', flush=True)

    def test_slot_reuse_waits_for_inference_before_overwriting_input(self):
        torch = self.torch
        engine = self.engine()
        source = make_source(engine, self.views[-1], channels='C3S1')
        slots = source.prepare_direct_ring(torch.float32)
        _, slot = source.next_direct_slot()
        slot.render_done.synchronize()
        expected = slot.input.clone()
        expected_ready = torch.cuda.Event(); expected_ready.record()
        consumed = torch.empty_like(expected)
        entered, release = threading.Event(), threading.Event()
        timeouts = []
        from XTA.geometry import _cupy_external_stream
        import cupy
        def gate(_):
            entered.set()
            if not release.wait(10):
                timeouts.append(True)
        # Warm the copy and current stream before putting a host gate on the
        # simulated TensorRT stream; the consumer owns this slot until infer_done.
        consumed.copy_(expected); torch.cuda.synchronize()
        external = _cupy_external_stream(cupy, slot.infer_stream)
        external.launch_host_func(gate, None)
        try:
            self.assertTrue(entered.wait(5))
            with torch.cuda.stream(slot.infer_stream):
                slot.infer_stream.wait_event(expected_ready)
                consumed.copy_(slot.input)
                slot.infer_done.record(slot.infer_stream)
            slot.infer_valid = True
            source.next_direct_slot()  # Uses independent slot1.
            source.next_direct_slot()  # Slot0 must wait for the outstanding read.
        finally:
            release.set()
            engine._stream.synchronize()
            slot.infer_stream.synchronize()
        self.assertFalse(timeouts)
        self.assertTrue(torch.equal(consumed, expected))
        self.assertTrue(torch.equal(slot.input, self.reference(source, 2, torch.float32)))

    def test_fp16_quantize_rounds_before_fp32_binding_and_refreshes_on_slot_reuse(self):
        torch = self.torch
        engine = self.engine()
        source = make_source(engine, self.views[-1], channels='C3S1', fp16=True)
        slots = source.prepare_direct_ring(torch.float32)
        pointers = [slot.input.data_ptr() for slot in slots]
        _, slot = source.next_direct_slot()
        slot.render_done.synchronize()
        expected = self.reference(source, 0, torch.float32)
        self.assertTrue(torch.equal(slot.input, expected))
        source.fp16 = False
        unrounded = self.reference(source, 0, torch.float32)
        self.assertFalse(torch.equal(expected, unrounded), 'fixture must expose FP16 intermediate rounding')
        refreshed = source.prepare_direct_ring(torch.float32)
        self.assertEqual([item.input.data_ptr() for item in refreshed], pointers)
        self.assertTrue(all(not item.native_preprocess_fp16 for item in refreshed))
        _, slot = source.next_direct_slot()
        slot.render_done.synchronize()
        self.assertTrue(torch.equal(slot.input, unrounded))

    def test_preconsumption_reset_does_not_recycle_input_with_a_queued_render_write(self):
        torch = self.torch
        engine = self.engine()
        source = make_source(engine, self.views[0])
        source.prepare_direct_ring(torch.float32)
        pointer = source._direct_ring[0].input.data_ptr()
        # Warm allocator/fill machinery before blocking the producer stream.
        warm = [torch.empty((1, 1, 16, 16), device='cuda').fill_(3) for _ in range(32)]
        torch.cuda.current_stream().synchronize()
        del warm
        entered, release = threading.Event(), threading.Event()
        timeouts = []
        from XTA.geometry import _cupy_external_stream
        import cupy
        def gate(_):
            entered.set()
            if not release.wait(10):
                timeouts.append(True)
        _cupy_external_stream(cupy, engine._stream).launch_host_func(gate, None)
        churn = []
        try:
            self.assertTrue(entered.wait(5))
            source.next_direct_slot()
            source.reset_direct_ring()
            self.assertIsNone(source._direct_ring)
            for _ in range(32):
                item = torch.empty((1, 1, 16, 16), device='cuda').fill_(7)
                self.assertNotEqual(item.data_ptr(), pointer)
                churn.append(item)
        finally:
            release.set()
            engine._stream.synchronize()
        self.assertFalse(timeouts)
        torch.cuda.synchronize()
        self.assertTrue(all(bool(torch.all(item == 7)) for item in churn))

    def test_failed_native_probe_can_reset_to_the_first_generic_frame(self):
        engine = self.engine()
        source = make_source(engine, self.views[-1], channels='C3S1')
        original = engine._render_fullframe_frame
        failure = RuntimeError('reject second context plane before inference')
        calls = 0
        def render(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise failure
            return original(*args, **kwargs)
        with mock.patch.object(engine, '_render_fullframe_frame', side_effect=render):
            with self.assertRaises(RuntimeError) as caught:
                source.prepare_direct_ring(self.torch.float32)
        self.assertIs(caught.exception, failure)
        source.reset_direct_ring()
        _, batch, _ = next(source)
        batch._tta_gpu_ready_event.synchronize()
        expected = self.reference(source, 0, self.torch.float32)
        self.assertTrue(self.torch.equal(batch._tta_gpu_tensor, expected))
        self.assertEqual(source.count, 1)
        self.assertEqual(source._direct_count, 0)


if __name__ == '__main__':
    unittest.main()
