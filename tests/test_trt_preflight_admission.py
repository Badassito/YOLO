"""Admission and ordering checks with no Torch/CUDA initialization."""
from __future__ import annotations

from contextlib import ExitStack
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection, cuda_backend


class TensorRTPreflightAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.order = []
        self.fp16, self.fp32 = object(), object()
        self.stack.enter_context(mock.patch.dict(sys.modules, {
            'torch': SimpleNamespace(float16=self.fp16, float32=self.fp32),
        }))
        self.stack.enter_context(mock.patch.object(backprojection, 'resident_trt_ring_enabled', return_value=True))
        self.decline = self.stack.enter_context(mock.patch.object(backprojection, '_resident_trt_pipeline_decline'))
        self.specs = ({'view': 'azimuthal', 'job': 'a0'}, {'view': 'azimuthal', 'job': 'a45'})
        self.stack.enter_context(mock.patch.object(backprojection, 'gpu_worker_fused_preflight_specs', return_value=self.specs))
        self.layout = self.stack.enter_context(mock.patch.object(
            backprojection, '_trt_binding_layout_for_backend',
            side_effect=lambda *args: self.record('layout', (['images'], 'images', [], {'images': 0})),
        ))
        self.dtype = self.stack.enter_context(mock.patch.object(
            backprojection, '_torch_dtype_for_trt_binding',
            side_effect=lambda *args: self.record('dtype', self.fp16),
        ))
        self.acquire = self.stack.enter_context(mock.patch.object(
            backprojection, '_resident_trt_pipeline_acquire', side_effect=self.stop_at_acquire,
        ))
        self.stack.enter_context(mock.patch('builtins.print'))
        self.stack.enter_context(mock.patch.object(backprojection, '_RESIDENT_TRT_RING_FALLBACK_WARNED', False))
        self.channels = 1
        self.trt_engine = SimpleNamespace(
            create_execution_context=mock.Mock(side_effect=AssertionError('context borrowed too early')),
            get_tensor_shape=lambda name: self.record('shape', (1, self.channels, 8, 8)),
        )
        self.backend = SimpleNamespace(model=self.trt_engine, dynamic=False)
        self.predictor = SimpleNamespace(model=self.backend)
        self.preflight = mock.Mock(side_effect=lambda *args, **kwargs: self.record('preflight', None))
        self.render_engine = SimpleNamespace(device='cuda:0', _mode='resident', run_startup_fused_preflight=self.preflight)
        self.source = self.make_source(cuda_backend.GpuRenderedYoloSource)
        self.cfg = SimpleNamespace(batch=1, input_channels=1, channel_token='gray', conf=0.25, quantize=32)
        self.device_union = SimpleNamespace(
            union_dev=SimpleNamespace(shape=(2, 8, 8), device='cuda:0'), conf_dev=None,
        )

    def record(self, label, value):
        self.order.append(label)
        return value

    def stop_at_acquire(self, *args, **kwargs):
        self.order.append('acquire')
        raise backprojection._ResidentTensorRTRingFatalError('acquire boundary reached')

    def make_source(self, cls):
        source = object.__new__(cls)
        source.engine = self.render_engine
        source.channel_count = self.channels
        source.bs, source.nf = 1, 2
        source.resident_ring_supported = True
        source.reset_direct_ring = mock.Mock()
        return source

    def run_admission(self):
        return backprojection._try_resident_trt_ring_accumulate(
            self.predictor, self.source, self.cfg,
            num_frames=2, out_size=8, native_h=8, native_w=8,
            M_out_to_native=np.asarray(((1, 0, 0), (0, 1, 0)), dtype=np.float32),
            device_union=self.device_union,
        )

    def test_static_single_channel_trt_validates_every_spec_before_acquisition(self):
        with self.assertRaisesRegex(backprojection._ResidentTensorRTRingFatalError, 'acquire boundary'):
            self.run_admission()
        self.assertEqual(self.order, ['layout', 'dtype', 'shape', 'preflight', 'acquire'])
        self.preflight.assert_called_once_with(self.specs, out_size=8, fp16=True)
        self.trt_engine.create_execution_context.assert_not_called()

    def test_preflight_failure_stays_fatal_and_never_borrows_bindings(self):
        self.preflight.side_effect = backprojection._ResidentTensorRTRingFatalError('numerical preflight refusal')
        with self.assertRaisesRegex(backprojection._ResidentTensorRTRingFatalError, 'numerical preflight refusal'):
            self.run_admission()
        self.preflight.assert_called_once()
        self.acquire.assert_not_called()
        self.decline.assert_not_called()
        self.trt_engine.create_execution_context.assert_not_called()

    def test_pt_backend_and_uninitialized_predictor_do_not_preflight(self):
        for predictor in (None, SimpleNamespace(model=SimpleNamespace(model=object(), dynamic=False))):
            self.predictor = predictor
            self.assertIsNone(self.run_admission())
        self.preflight.assert_not_called()
        self.acquire.assert_not_called()
        # An initially absent predictor must not mark the volume validated or
        # suppress the check when a real TensorRT backend is later admitted.
        self.predictor = SimpleNamespace(model=self.backend)
        with self.assertRaisesRegex(backprojection._ResidentTensorRTRingFatalError, 'acquire boundary'):
            self.run_admission()
        self.preflight.assert_called_once()

    def test_dynamic_trt_does_not_preflight(self):
        self.backend.dynamic = True
        self.assertIsNone(self.run_admission())
        self.preflight.assert_not_called()
        self.acquire.assert_not_called()

    def test_batch_greater_than_one_does_not_preflight(self):
        self.cfg.batch = self.source.bs = 2
        self.assertIsNone(self.run_admission())
        self.preflight.assert_not_called()
        self.acquire.assert_not_called()

    def test_radial_source_does_not_preflight(self):
        self.source.resident_ring_supported = False
        self.assertIsNone(self.run_admission())
        self.preflight.assert_not_called()
        self.acquire.assert_not_called()

    def test_multichannel_trt_keeps_generic_renderer_without_fused_preflight(self):
        self.channels = self.source.channel_count = self.cfg.input_channels = 3
        self.cfg.channel_token = 'C3S1'
        with self.assertRaisesRegex(backprojection._ResidentTensorRTRingFatalError, 'acquire boundary'):
            self.run_admission()
        self.preflight.assert_not_called()
        self.acquire.assert_called_once()

    def test_tile_trt_keeps_generic_renderer_without_fused_preflight(self):
        self.source = self.make_source(cuda_backend.GpuTileRenderedYoloSource)
        with self.assertRaisesRegex(backprojection._ResidentTensorRTRingFatalError, 'acquire boundary'):
            self.run_admission()
        self.preflight.assert_not_called()
        self.acquire.assert_called_once()

    def test_invalid_binding_shape_cannot_run_preflight(self):
        self.trt_engine.get_tensor_shape = lambda name: (1, 1, 16, 16)
        self.assertIsNone(self.run_admission())
        self.preflight.assert_not_called()
        self.acquire.assert_not_called()

    def test_invalid_binding_dtype_cannot_run_preflight(self):
        self.dtype.side_effect = None
        self.dtype.return_value = object()
        self.assertIsNone(self.run_admission())
        self.preflight.assert_not_called()
        self.acquire.assert_not_called()


if __name__ == '__main__':
    unittest.main()
