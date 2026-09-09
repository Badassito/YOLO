"""Borrowed TensorRT context handoff across generic shell tasks, without CUDA."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection as b, cuda_backend


class Tensor:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class Context:
    def __init__(self):
        self.addresses = {}
        self.fail_name = None

    def set_tensor_address(self, name, address):
        if name == self.fail_name:
            return False
        self.addresses[name] = address
        return True


class RingInterleaveTests(unittest.TestCase):
    generic_family = 'radial'

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(b._RESIDENT_TRT_PIPELINE_CACHE, {}, clear=True))
        self.stack.enter_context(mock.patch.object(b, 'resident_trt_ring_enabled', return_value=True))
        self.stack.enter_context(mock.patch.object(b, 'resident_trt_pipeline_persistence_enabled', return_value=True))
        self.engine = SimpleNamespace(create_execution_context=mock.Mock())
        self.context = Context()
        self.backend = SimpleNamespace(
            model=self.engine, context=self.context,
            bindings={name: SimpleNamespace(data=Tensor(ptr))
                      for name, ptr in (('images', 101), ('head', 102), ('proto', 103))},
        )
        self.render_engine = SimpleNamespace(_volume_key=('source-a', (9, 10, 11), 9))
        self.executor = object.__new__(b._ResidentTensorRTRingExecutor)
        ex = self.executor
        ex.backend, ex.engine, ex.device = self.backend, self.engine, 'cuda:0'
        ex._closed = ex._generic_suspended = False
        ex._borrowed_graph_recapture_needed = False
        ex._borrowed_context = self.context
        ex.binding_names = ['images', 'head', 'proto']
        ex._borrowed_ring_addresses = {'images': 11, 'head': 12, 'proto': 13}
        ex._restore_tensor_addresses = {'images': 101, 'head': 102, 'proto': 103}
        self.context.addresses.update(ex._borrowed_ring_addresses)
        ex.torch = SimpleNamespace(cuda=SimpleNamespace(
            synchronize=mock.Mock(), device=lambda *_: nullcontext(), empty_cache=lambda: None,
        ))
        ex.kernels = SimpleNamespace(cp=None)
        ex.slots = [SimpleNamespace(
            infer_stream=SimpleNamespace(synchronize=mock.Mock()),
            post_stream=SimpleNamespace(synchronize=mock.Mock()),
            _cupy_refs={}, _render_cupy_refs={}, infer_graph=object(), post_graph=object(),
            render_graph=object(), post_valid=False,
        ) for _ in range(2)]
        ex.reset_for_task = mock.Mock()
        ex.configure_slice_bbox_collection = mock.Mock()
        ex.reconfigure_destination = mock.Mock()
        self.recaptured_graph = object()
        ex._recapture_borrowed_inference_graph = mock.Mock(
            side_effect=lambda: setattr(ex.slots[0], 'infer_graph', self.recaptured_graph),
        )
        self.arguments = dict(input_dtype='float16', input_channels=1, out_size=8,
            native_h=8, native_w=8, M_out_to_native=np.eye(2, 3, dtype=np.float32),
            track_conf=False, confidence_threshold=.5, collect_slice_bboxes=False,
            dynamic_unit_descriptors=True)
        signature_args = {k: v for k, v in self.arguments.items() if k != 'collect_slice_bboxes'}
        signature = b._resident_trt_pipeline_signature(self.backend, **signature_args)
        b._RESIDENT_TRT_PIPELINE_CACHE[id(self.backend)] = {
            'backend': self.backend, 'signature': signature, 'executor': ex,
            'in_use': False, 'last_used': 0,
            'render_engine': self.render_engine, 'render_volume_key': self.render_engine._volume_key,
        }
        self.generic_source = self.source(self.generic_family)

    def source(self, family):
        source = object.__new__(cuda_backend.GpuRenderedYoloSource)
        source.view = SimpleNamespace(family=family)
        source.engine = self.render_engine
        source.resident_ring_supported = family not in ('radial', 'spherical')
        source.prepare_direct_ring = mock.Mock(side_effect=lambda **_: self.executor.slots)
        source._direct_ring = None
        return source

    def bypass(self, source=None):
        return b._try_resident_trt_ring_accumulate(
            SimpleNamespace(model=self.backend), self.generic_source if source is None else source, SimpleNamespace(),
            num_frames=2, out_size=8, native_h=8, native_w=8,
            M_out_to_native=np.eye(2, 3, dtype=np.float32), device_union=object(),
        )

    def test_generic_shell_restores_backend_then_next_eligible_source_hits_retained_ring(self):
        ex = self.executor
        graphs = [(slot.infer_graph, slot.render_graph) for slot in ex.slots]
        self.assertIsNone(self.bypass())
        self.assertFalse(ex._closed)
        self.assertTrue(ex._generic_suspended)
        self.assertIsNone(ex.slots[0].infer_graph)
        self.assertIs(ex.slots[1].infer_graph, graphs[1][0])
        self.assertEqual(self.context.addresses, {'images': 101, 'head': 102, 'proto': 103})
        for slot in ex.slots:
            slot.infer_stream.synchronize.assert_called_once()
            slot.post_stream.synchronize.assert_called_once()
        # Generic inference can replace its tensor storage and context addresses.
        self.backend.bindings['head'].data.pointer = 202
        self.context.addresses.update(images=999, head=202)
        source = self.source('azimuthal')
        actual, hit = b._resident_trt_pipeline_acquire(self.backend, source, **self.arguments)
        self.assertTrue(hit)
        self.assertIs(actual, ex)
        self.assertFalse(ex._generic_suspended)
        ex.torch.cuda.synchronize.assert_called_once_with('cuda:0')
        self.assertEqual(self.context.addresses, ex._borrowed_ring_addresses)
        self.assertEqual(ex._restore_tensor_addresses['head'], 202)
        self.assertIs(ex.slots[0].infer_graph, self.recaptured_graph)
        self.assertIs(ex.slots[1].infer_graph, graphs[1][0])
        self.assertEqual([slot.render_graph for slot in ex.slots], [pair[1] for pair in graphs])
        ex._recapture_borrowed_inference_graph.assert_called_once()
        self.engine.create_execution_context.assert_not_called()
        b._resident_trt_pipeline_release(self.backend, ex, source)
        self.assertIsNone(self.bypass())
        self.assertEqual(self.context.addresses['head'], 202)

    def test_source_volume_change_preserves_hard_invalidation(self):
        self.render_engine._volume_key = ('different-source', (9, 10, 11), 9)
        self.assertIsNone(self.bypass())
        self.assertTrue(self.executor._closed)
        self.assertFalse(b._RESIDENT_TRT_PIPELINE_CACHE)

    def test_backend_context_replacement_preserves_hard_invalidation(self):
        self.backend.context = Context()
        self.assertIsNone(self.bypass())
        self.assertTrue(self.executor._closed)
        self.assertFalse(b._RESIDENT_TRT_PIPELINE_CACHE)
        self.assertFalse(self.backend.context.addresses)

    def test_graph_recapture_warms_before_capture_and_keeps_independent_slot(self):
        ex = self.executor
        ex.slots[0].input = SimpleNamespace(zero_=mock.Mock())
        independent = ex.slots[1].infer_graph
        events = []
        graph = object()
        ex.torch.cuda.stream = lambda stream: nullcontext()
        ex.torch.cuda.CUDAGraph = lambda: graph
        ex._execute_context = mock.Mock(side_effect=lambda slot: events.append('execute'))
        def capture(*args):
            events.append('capture')
            return nullcontext()
        with mock.patch.object(b, '_cuda_graph_capture_context', side_effect=capture):
            b._ResidentTensorRTRingExecutor._recapture_borrowed_inference_graph(ex)
        self.assertEqual(events, ['execute', 'capture', 'execute'])
        self.assertIs(ex.slots[0].infer_graph, graph)
        self.assertIs(ex.slots[1].infer_graph, independent)
        self.assertEqual(ex.infer_graph_count, 2)

    def test_model_replacement_preserves_hard_invalidation(self):
        self.backend.model = SimpleNamespace(create_execution_context=mock.Mock())
        self.assertIsNone(self.bypass())
        self.assertTrue(self.executor._closed)
        self.assertFalse(b._RESIDENT_TRT_PIPELINE_CACHE)

    def test_changed_precision_signature_closes_old_executor_and_builds_new(self):
        self.assertIsNone(self.bypass())
        source = self.source('azimuthal')
        source.prepare_direct_ring = mock.Mock(return_value=[object(), object()])
        replacement = SimpleNamespace()
        with mock.patch.object(b, '_ResidentTensorRTRingExecutor', return_value=replacement):
            actual, hit = b._resident_trt_pipeline_acquire(
                self.backend, source, **{**self.arguments, 'input_dtype': 'float32'},
            )
        self.assertFalse(hit)
        self.assertIs(actual, replacement)
        self.assertTrue(self.executor._closed)

    def test_busy_executor_is_not_closed_or_returned_to_generic_inference(self):
        b._RESIDENT_TRT_PIPELINE_CACHE[id(self.backend)]['in_use'] = True
        with self.assertRaisesRegex(b._ResidentTensorRTRingFatalError, 'active'):
            self.bypass()
        self.assertFalse(self.executor._closed)
        self.assertIs(b._RESIDENT_TRT_PIPELINE_CACHE[id(self.backend)]['executor'], self.executor)

    def test_binding_restore_failure_stays_fatal_and_invalidates(self):
        self.context.fail_name = 'head'
        with self.assertRaisesRegex(b._ResidentTensorRTRingFatalError, 'restore'):
            self.bypass()
        self.assertTrue(self.executor._closed)
        self.assertFalse(b._RESIDENT_TRT_PIPELINE_CACHE)

    def test_generic_health_failure_cannot_resume_cached_execution(self):
        self.assertIsNone(self.bypass())
        self.executor.torch.cuda.synchronize.side_effect = RuntimeError('generic work failed')
        with self.assertRaisesRegex(b._ResidentTensorRTRingFatalError, 'drain'):
            b._resident_trt_pipeline_acquire(self.backend, self.source('azimuthal'), **self.arguments)
        self.assertTrue(self.executor._closed)
        self.assertFalse(b._RESIDENT_TRT_PIPELINE_CACHE)

    def test_close_while_suspended_restores_current_generic_tensor_addresses(self):
        self.assertIsNone(self.bypass())
        self.backend.bindings['proto'].data.pointer = 303
        self.executor.close()
        self.assertEqual(self.context.addresses, {'images': 101, 'head': 102, 'proto': 303})

    def test_other_unsupported_family_keeps_existing_teardown(self):
        source = self.source('azimuthal')
        source.resident_ring_supported = False
        self.assertIsNone(self.bypass(source))
        self.assertTrue(self.executor._closed)

    def test_disabled_persistence_does_not_soft_retain_shell_cache(self):
        with mock.patch.object(b, 'resident_trt_pipeline_persistence_enabled', return_value=False):
            self.assertIsNone(self.bypass())
        self.assertTrue(self.executor._closed)


class SphericalRingInterleaveTests(RingInterleaveTests):
    """Run the same context lifetime and failure guards for Spherical bypasses."""

    generic_family = 'spherical'

    def test_cartesian_spherical_cartesian_retains_contexts_and_restores_bindings(self):
        cartesian = self.source('orthogonal')
        before, first_hit = b._resident_trt_pipeline_acquire(
            self.backend, cartesian, **self.arguments,
        )
        self.assertTrue(first_hit)
        b._resident_trt_pipeline_release(self.backend, before, cartesian)
        self.assertIsNone(self.bypass())
        self.generic_source.prepare_direct_ring.assert_not_called()
        self.assertTrue(before._generic_suspended)
        self.assertEqual(self.context.addresses, {'images': 101, 'head': 102, 'proto': 103})

        # Generic spherical inference may replace both output tensors and bindings.
        self.backend.bindings['head'].data.pointer = 202
        self.backend.bindings['proto'].data.pointer = 303
        self.context.addresses.update(images=999, head=202, proto=303)
        after, second_hit = b._resident_trt_pipeline_acquire(
            self.backend, cartesian, **self.arguments,
        )
        self.assertTrue(second_hit)
        self.assertIs(after, before)
        self.assertFalse(after._generic_suspended)
        self.assertEqual(self.context.addresses, after._borrowed_ring_addresses)
        self.assertEqual(after._restore_tensor_addresses, {'images': 101, 'head': 202, 'proto': 303})
        after._recapture_borrowed_inference_graph.assert_called_once()
        self.engine.create_execution_context.assert_not_called()
        b._resident_trt_pipeline_release(self.backend, after, cartesian)
        after.close()
        self.assertEqual(self.context.addresses, {'images': 101, 'head': 202, 'proto': 303})


if __name__ == '__main__':
    unittest.main()
