"""CPU ownership/lifecycle checks plus an explicitly opt-in CUDA residency smoke."""
from __future__ import annotations

import contextlib
import gc
import io
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np

from XTA import backprojection, cuda_backend, cuda_d1, inference, workers


class _Owner:
    def __init__(self, nbytes=64):
        self.nbytes = nbytes


class _Stream:
    def __init__(self, events, label='render_fence', fail=False):
        self.events, self.label, self.fail = events, label, fail

    def synchronize(self):
        self.events.append(self.label)
        if self.fail:
            raise RuntimeError('stream fence failed')


class _Cuda:
    def __init__(self, events, initialized=True):
        self.events, self.initialized = events, initialized
        self.samples = 0

    def is_initialized(self):
        return self.initialized

    def _check(self):
        if not self.initialized:
            raise AssertionError('A physical GPU API was queried without an initialized context')

    def mem_get_info(self, device):
        self._check()
        self.samples += 1
        return (1000 if self.samples == 1 else 9000), 10000

    def memory_allocated(self, device):
        self._check()
        return 128

    def memory_reserved(self, device):
        self._check()
        return 256

    def synchronize(self, device=0):
        self._check()
        self.events.append('device_fence')

    def device(self, device):
        return contextlib.nullcontext()

    def empty_cache(self):
        self._check()
        self.events.append('torch_trim')

    def ipc_collect(self):
        self._check()
        self.events.append('ipc_trim')


def _renderer(events, host_source=None, fail=False):
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine._stream = _Stream(events, fail=fail)
    engine._radial_texture_lock = threading.RLock()
    engine._inference_assets_released = False
    engine._volume_mm = host_source
    source = _Owner(4096)
    image = _Owner(4096)
    engine._volume_gpu = source
    engine._volume_flat = source
    engine._fused_volume_ref = source
    engine._radial_texture_ref = SimpleNamespace(
        source_ref=source, texture=_Owner(), descriptor=_Owner(), resource=image,
        cuda_array=image, channel=_Owner(), nbytes=4096,
    )
    for name in ('_native_t_map_cache', '_native_plane_cache', '_native_u8_plane_cache',
                 '_fold_cache', '_tilted_plans', '_fused_radial_taps'):
        setattr(engine, name, {'buffer': _Owner()})
    for name in ('_fused_preflight_validated_families', '_fused_graph_rejected_keys', '_fused_validated_keys'):
        setattr(engine, name, {'entry'})
    engine._standalone_render_meta = _Owner()
    engine._standalone_render_meta_ref = engine._standalone_render_meta
    engine._fused_preflight_volume_key = 'source'
    engine._radial_texture_admitted = True
    engine._resident_runtime_disabled = False
    engine._mode = 'resident'
    return engine, (weakref.ref(source), weakref.ref(image))


def _model(events):
    weight, binding = _Owner(1024), _Owner(512)
    source = SimpleNamespace(close=lambda: events.append('source_closed'))
    predictor = SimpleNamespace(dataset=source, model=binding, results=_Owner(), batch=_Owner())
    model = SimpleNamespace(model=weight, predictor=predictor, _tta_predict_state='ready')
    return model, (weakref.ref(weight), weakref.ref(binding))


class GpuAssetRetirementTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(cuda_d1._D1_WORKER_VIEW_STATES, {}, clear=True))
        self.stack.enter_context(mock.patch.dict(backprojection._RESIDENT_TRT_PIPELINE_CACHE, {}, clear=True))
        self.stack.enter_context(mock.patch.dict(inference._AFFINE_GRID_CACHE, {}, clear=True))

    def test_renderer_releases_all_linear_texture_aliases_and_preserves_host_input(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.u8'
            host = np.memmap(path, dtype=np.uint8, mode='w+', shape=(2, 3, 4))
            host[:] = 7
            engine, references = _renderer(events, host)
            namespace = engine._radial_texture_ref
            try:
                stats = engine.release_inference_assets()
                gc.collect()
                self.assertEqual(events, ['render_fence'])
                self.assertTrue(all(reference() is None for reference in references))
                self.assertIsNone(namespace.source_ref)
                self.assertIsNone(namespace.cuda_array)
                self.assertIsNone(namespace.texture)
                self.assertEqual(stats['source_bytes'], 4096)
                self.assertEqual(stats['texture_bytes'], 4096)
                self.assertTrue(stats['host_source_preserved'])
                self.assertFalse(host._mmap.closed)
                self.assertTrue(path.exists())
                np.testing.assert_array_equal(host, np.full(host.shape, 7, np.uint8))
                self.assertTrue(engine.release_inference_assets()['already_released'])
                with self.assertRaisesRegex(RuntimeError, 'retired'):
                    engine.ensure_volume(str(path), host.shape)
                with self.assertRaisesRegex(RuntimeError, 'retired'):
                    engine.ensure_volume_array(host)
            finally:
                host._mmap.close()

    def test_renderer_failed_fence_keeps_owners_intact(self):
        events = []
        engine, references = _renderer(events, fail=True)
        with self.assertRaisesRegex(RuntimeError, 'fence failed'):
            engine.release_inference_assets()
        self.assertTrue(all(reference() is not None for reference in references))
        self.assertFalse(engine._inference_assets_released)
        self.assertEqual(engine._mode, 'resident')

    def test_real_ring_teardown_drops_graph_buffers_and_restores_borrowed_bindings(self):
        events = []
        executor = object.__new__(backprojection._ResidentTensorRTRingExecutor)
        executor.device = SimpleNamespace(index=0)
        executor.torch = SimpleNamespace(cuda=_Cuda(events))
        executor.kernels = SimpleNamespace(cp=None)
        executor._closed = False
        executor._restore_tensor_addresses = {'images': 1234}
        executor._borrowed_context = SimpleNamespace(
            set_tensor_address=lambda name, address: events.append(('restore', name, address)) or True)
        owners = [_Owner() for _ in range(5)]
        references = [weakref.ref(owner) for owner in owners]
        slot = SimpleNamespace(infer_stream=_Stream(events, 'infer_fence'), post_stream=_Stream(events, 'post_fence'),
                               infer_graph=owners[0], post_graph=owners[1], render_graph=owners[2],
                               _cupy_refs={'input': owners[3]}, _render_cupy_refs={'source': owners[4]})
        del owners
        executor.slots = [slot]
        source_alias = executor.slots
        backend = object()
        backprojection._RESIDENT_TRT_PIPELINE_CACHE[id(backend)] = {
            'backend': backend, 'executor': executor, 'in_use': False}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(backprojection._release_resident_trt_pipeline_cache(), 1)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(source_alias, [])
        self.assertEqual(backprojection._RESIDENT_TRT_PIPELINE_CACHE, {})
        self.assertLess(events.index('infer_fence'), events.index(('restore', 'images', 1234)))
        self.assertIsNone(slot.input)
        self.assertIsNone(slot.context)

    def test_busy_ring_rejection_has_no_destructive_side_effect(self):
        closer = mock.Mock()
        backprojection._RESIDENT_TRT_PIPELINE_CACHE[1] = {'executor': SimpleNamespace(close=closer), 'in_use': True}
        with self.assertRaisesRegex(RuntimeError, 'in use'):
            backprojection._release_resident_trt_pipeline_cache()
        closer.assert_not_called()
        self.assertEqual(len(backprojection._RESIDENT_TRT_PIPELINE_CACHE), 1)

    def _worker_context(self, *, initialized=True, fail_fence=False):
        events = []
        model, model_refs = _model(events)
        engine, renderer_refs = _renderer(events, fail=fail_fence)
        assets = workers._GpuWorkerInferenceAssets(model)
        cuda = _Cuda(events, initialized=initialized)
        self.stack.enter_context(mock.patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda), 'cupy': None}))
        self.stack.enter_context(mock.patch.object(workers, '_worker_gpu_render_engine', return_value=engine))
        self.stack.enter_context(mock.patch.object(workers, '_gpu_union_retirement_manager', return_value=None))
        self.stack.enter_context(mock.patch.object(workers, '_shutdown_gpu_union_retirement_manager',
                                                 side_effect=lambda: events.append('lanes_closed')))
        self.stack.enter_context(mock.patch.object(workers, '_shutdown_d1_worker_pipeline',
                                                 side_effect=lambda: events.append('d1_closed')))
        return assets, engine, model, model_refs+renderer_refs, events, cuda

    def test_worker_release_drains_first_then_drops_model_and_reports_memory(self):
        assets, engine, model, references, events, cuda = self._worker_context()
        grid = _Owner()
        grid_ref = weakref.ref(grid)
        inference._AFFINE_GRID_CACHE['grid'] = grid
        del grid
        result = workers._release_gpu_worker_inference_assets(
            assets, inference_drained=True, wait_for_publications=lambda: events.append('publications_drained'))
        gc.collect()
        self.assertEqual(events[0], 'publications_drained')
        self.assertLess(events.index('device_fence'), events.index('lanes_closed'))
        self.assertLess(events.index('render_fence'), events.index('lanes_closed'))
        self.assertTrue(all(reference() is None for reference in references))
        self.assertIsNone(grid_ref())
        self.assertIsNone(assets.model)
        self.assertIsNone(model.predictor)
        self.assertTrue(result['released'])
        self.assertEqual(result['driver_free_delta_bytes'], 8000)
        repeated = workers._release_gpu_worker_inference_assets(
            assets, inference_drained=True, wait_for_publications=lambda: self.fail('idempotent release waited again'))
        self.assertTrue(repeated['already_released'])
        self.assertEqual(cuda.samples, 2)

    def test_refuses_unproven_drain_active_d1_and_busy_lanes_with_model_intact(self):
        assets, engine, model, references, events, cuda = self._worker_context()
        wait = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'authoritative'):
            workers._release_gpu_worker_inference_assets(assets, inference_drained=False, wait_for_publications=wait)
        wait.assert_not_called()
        cuda_d1._D1_WORKER_VIEW_STATES[('model', 'view')] = object()
        with self.assertRaisesRegex(RuntimeError, 'active D1'):
            workers._release_gpu_worker_inference_assets(assets, inference_drained=True, wait_for_publications=wait)
        cuda_d1._D1_WORKER_VIEW_STATES.clear()
        with mock.patch.object(workers, '_gpu_union_retirement_manager',
                               return_value=SimpleNamespace(lanes=[SimpleNamespace(_active=True)], capacity=1)):
            with self.assertRaisesRegex(RuntimeError, 'lane is active'):
                workers._release_gpu_worker_inference_assets(assets, inference_drained=True, wait_for_publications=wait)
        self.assertIs(assets.model, model)
        self.assertTrue(all(reference() is not None for reference in references))
        self.assertFalse(assets.release_started)
        self.assertTrue(assets.release_stats['assets_intact'])
        self.assertEqual(cuda.samples, 0)

    def test_preflight_stream_failure_preserves_inference_owners(self):
        assets, engine, model, references, events, cuda = self._worker_context(fail_fence=True)
        with self.assertRaisesRegex(RuntimeError, 'fence failed'):
            workers._release_gpu_worker_inference_assets(assets, inference_drained=True, wait_for_publications=lambda: None)
        self.assertIs(assets.model, model)
        self.assertFalse(assets.release_started)
        self.assertTrue(assets.release_stats['assets_intact'])
        self.assertTrue(all(reference() is not None for reference in references))

    def test_no_gpu_queries_when_context_was_not_initialized(self):
        assets, engine, model, references, events, cuda = self._worker_context(initialized=False)
        result = workers._release_gpu_worker_inference_assets(
            assets, inference_drained=True, wait_for_publications=lambda: None)
        self.assertFalse(result['memory_before']['available'])
        self.assertFalse(result['memory_after']['available'])
        self.assertNotIn('device_fence', events)
        self.assertNotIn('torch_trim', events)

    def test_worker_protocol_uses_logical_gpu_ack_and_keeps_auxiliary_service_alive(self):
        assets, engine, model, references, events, cuda = self._worker_context(initialized=False)
        incoming, outgoing = queue.Queue(), queue.Queue()
        for task in (
            {'task_id': -10, 'task_type': 'control', 'op': 'release_inference_assets', 'inference_drained': True},
            {'task_id': -11, 'task_type': 'control', 'op': 'release_inference_assets', 'inference_drained': True},
            {'task_id': 20, 'task_type': 'interpolation_pass', 'aux_kwargs': {}},
            {'task_id': 21, 'task_type': 'inference', 'slice_count': 1}, None,
        ):
            incoming.put(task)
        init = {'imgsz': 16, 'conf': 0.5, 'batch': 1, 'quantize': 16, 'cpu_workers': 1}
        replacements = {
            'configure_pipeline_modes': None, 'initialize_runtime_observability': None,
            'set_retina_mask_processor': None, 'set_gpu_worker_fused_preflight_specs': None,
            'set_angle_variant_gpu_fastpath': None, 'ensure_yolo_ready_for_predict': None,
            'validate_yolo_model_input_channels': None, 'require_channel_aware_yolo_preprocess_patch': None,
            'ensure_cpu_retina_mask_predictor_patch': None, '_init_gpu_union_retirement_manager': None,
            '_init_worker_gpu_render_engine': engine, 'load_ultralytics_model': model,
            'cpu_retina_masks_enabled': True, 'd1_owner_pipeline_enabled': False,
            '_interpolation_process_entry': {'auxiliary_ok': True},
        }
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
            stack.enter_context(mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,7'}))
            for name, result in replacements.items():
                stack.enter_context(mock.patch.object(workers, name, return_value=result))
            prediction = stack.enter_context(mock.patch.object(workers, 'run_prediction_volume_in_worker'))
            workers._gpu_inference_worker_main(1, 'model.engine', init, incoming, outgoing)
            prediction.assert_not_called()
        messages = []
        while not outgoing.empty(): messages.append(outgoing.get())
        acknowledgments = [message for message in messages if message['type'] == 'inference_assets_released']
        self.assertEqual(len(acknowledgments), 2)
        self.assertTrue(all(message['ok'] and message['gpu_index'] == 1 for message in acknowledgments))
        self.assertTrue(acknowledgments[1]['stats']['already_released'])
        auxiliary = next(message for message in messages if message['type'] == 'aux_result')
        self.assertTrue(auxiliary['ok'])
        self.assertEqual(auxiliary['stats'], {'auxiliary_ok': True})
        late = next(message for message in messages if message.get('task_id') == 21)
        self.assertFalse(late['ok'])
        self.assertIn('after the terminal', late['error'])


@unittest.skipUnless(os.environ.get('XTA_RUN_CUDA_ASSET_RETIREMENT_SMOKE') == '1', 'explicit CUDA smoke only')
class CudaAssetRetirementSmokeTests(unittest.TestCase):
    def test_actual_resident_source_and_texture_owners_release_without_deleting_input(self):
        import torch
        import cupy as cp
        if not torch.cuda.is_available():
            self.skipTest('CUDA is unavailable')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.u8'
            original = np.arange(32*128*128, dtype=np.uint8).reshape(32,128,128)
            original.tofile(path)
            engine = cuda_backend._GpuWorkerRenderEngine('cuda:0')
            try:
                mode = engine.ensure_volume(str(path), original.shape, require_radial_texture=True)
                self.assertEqual(mode, 'resident')
                texture = engine._ensure_radial_texture(SimpleNamespace(cp=cp))
                source_ref = weakref.ref(engine._volume_gpu)
                stats = engine.release_inference_assets()
                gc.collect()
                torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.synchronize()
                self.assertIsNone(source_ref())
                self.assertIsNone(texture.cuda_array)
                self.assertEqual(stats['source_bytes'], original.nbytes)
                self.assertEqual(stats['texture_bytes'], original.nbytes)
                self.assertFalse(engine._volume_mm._mmap.closed)
                np.testing.assert_array_equal(np.fromfile(path,dtype=np.uint8).reshape(original.shape), original)
            finally:
                if engine._volume_mm is not None:
                    engine._volume_mm._mmap.close()


if __name__ == '__main__':
    unittest.main()
