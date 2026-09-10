"""Native ring policy, no-replay guards and real CUDA mask-post parity."""
from __future__ import annotations

import os
import sys
import ast
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection as b, inference


class NativeRingPolicyTests(unittest.TestCase):
    def test_native_policy_is_separate_from_legacy_even_with_same_rasters(self):
        for family in ('radial', 'spherical', 'orthogonal', 'tilted', 'azimuthal'):
            policy = b._resident_trt_post_policy(SimpleNamespace(view=SimpleNamespace(family=family)))
            self.assertEqual(policy, 'native_mask' if family in ('radial', 'spherical') else 'legacy_proto')
        args = dict(input_channels=1, out_size=32, native_h=32, native_w=32,
                    M_out_to_native=np.eye(2, 3), track_conf=False, confidence_threshold=.5,
                    dynamic_unit_descriptors=True)
        backend = SimpleNamespace(model=SimpleNamespace(create_execution_context=lambda: None))
        self.assertNotEqual(
            b._resident_trt_pipeline_signature(backend, 'float32', post_policy='native_mask', **args),
            b._resident_trt_pipeline_signature(backend, 'float32', post_policy='legacy_proto', **args),
        )

    def test_same_native_destination_retains_static_post_graphs_without_warmup(self):
        ex = object.__new__(b._ResidentTensorRTRingExecutor)
        ex.post_policy = 'native_mask'
        ex.native_h, ex.native_w = 43, 67
        ex.dynamic_unit_descriptors = False
        matrix = np.asarray(((.9, .1, 1.25), (0, 1, -.5)), np.float32)
        ex.default_descriptor = SimpleNamespace(M_out_to_native=matrix)
        graphs = [object(), object()]
        ex.slots = [SimpleNamespace(post_graph=graph) for graph in graphs]
        ex.synchronize = mock.Mock(side_effect=AssertionError('unchanged destination drained'))
        ex._refresh_native_post_graphs = mock.Mock(side_effect=AssertionError('unchanged graph rebuilt'))
        ex.reconfigure_destination(native_h=43, native_w=67, M_out_to_native=matrix.copy(),
                                   dynamic_unit_descriptors=False)
        self.assertEqual([slot.post_graph for slot in ex.slots], graphs)

    def test_native_cache_hit_passes_static_descriptor_policy(self):
        from tests.test_trt_ring_interleave import RingInterleaveTests
        fixture = RingInterleaveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        args = {**fixture.arguments, 'post_policy': 'native_mask', 'dynamic_unit_descriptors': False}
        signature_args = {key: value for key, value in args.items() if key != 'collect_slice_bboxes'}
        b._RESIDENT_TRT_PIPELINE_CACHE[id(fixture.backend)]['signature'] = b._resident_trt_pipeline_signature(
            fixture.backend, **signature_args)
        actual, hit = b._resident_trt_pipeline_acquire(fixture.backend, fixture.source('spherical'), **args)
        self.assertTrue(hit)
        self.assertIs(actual, fixture.executor)
        self.assertFalse(actual.reconfigure_destination.call_args.kwargs['dynamic_unit_descriptors'])

    def test_policy_transitions_reuse_inference_contexts_bindings_and_graphs(self):
        from tests.test_trt_ring_interleave import RingInterleaveTests
        fixture = RingInterleaveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        ex = fixture.executor
        ex.reconfigure_post_policy = mock.Mock()
        graphs = [slot.infer_graph for slot in ex.slots]
        addresses = dict(fixture.context.addresses)
        with mock.patch.object(b, '_ResidentTensorRTRingExecutor',
                               side_effect=AssertionError('post policy rebuilt TensorRT contexts')):
            for policy in ('native_mask', 'legacy_proto', 'native_mask'):
                source = fixture.source('spherical' if policy == 'native_mask' else 'azimuthal')
                args = {**fixture.arguments, 'post_policy': policy, 'dynamic_unit_descriptors': False}
                actual, hit = b._resident_trt_pipeline_acquire(fixture.backend, source, **args)
                self.assertIs(actual, ex)
                self.assertTrue(hit)
                self.assertEqual(ex.reconfigure_post_policy.call_args.kwargs['post_policy'], policy)
                self.assertEqual([slot.infer_graph for slot in ex.slots], graphs)
                self.assertEqual(fixture.context.addresses, addresses)
                self.assertFalse(ex._closed)
                b._resident_trt_pipeline_release(fixture.backend, ex, source)
        self.assertEqual(ex.reconfigure_post_policy.call_count, 3)
        fixture.engine.create_execution_context.assert_not_called()

    def test_policy_change_cannot_release_private_post_buffers_before_stream_drain(self):
        ex = object.__new__(b._ResidentTensorRTRingExecutor)
        ex.post_policy = 'native_mask'
        ex.synchronize = mock.Mock()
        refs, graph = {'live': object()}, object()
        ex.slots = [SimpleNamespace(post_stream=SimpleNamespace(
            synchronize=mock.Mock(side_effect=RuntimeError('pending post stream failed'))),
            _cupy_refs=refs, post_graph=graph)]
        with self.assertRaisesRegex(RuntimeError, 'pending post stream failed'):
            ex.reconfigure_post_policy(post_policy='legacy_proto', track_conf=False,
                confidence_threshold=.5, collect_slice_bboxes=False, native_h=8, native_w=8,
                M_out_to_native=np.eye(2, 3), dynamic_unit_descriptors=False)
        self.assertEqual(ex.post_policy, 'native_mask')
        self.assertIs(ex.slots[0]._cupy_refs, refs)
        self.assertTrue(refs)
        self.assertIs(ex.slots[0].post_graph, graph)

    def test_failed_post_contract_change_invalidates_cached_executor_before_source_consumption(self):
        from tests.test_trt_ring_interleave import RingInterleaveTests
        fixture = RingInterleaveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        ex = fixture.executor
        ex.reconfigure_post_policy = mock.Mock(side_effect=RuntimeError('new post contract refused'))
        ex.close = mock.Mock()
        source = fixture.source('spherical')
        with self.assertRaisesRegex(RuntimeError, 'new post contract refused'):
            b._resident_trt_pipeline_acquire(fixture.backend, source,
                **{**fixture.arguments, 'post_policy': 'native_mask'})
        self.assertNotIn(id(fixture.backend), b._RESIDENT_TRT_PIPELINE_CACHE)
        ex.close.assert_called_once()
        source.prepare_direct_ring.assert_not_called()
        self.assertIsNone(source._direct_ring)

    def test_fp32_packed_launch_preserves_count_capacity_and_skips_proto_closing(self):
        fp32, fp16 = object(), object()
        kernels = SimpleNamespace(cp=object(), compact_f32_tiled=mock.Mock(),
                                  union_f32_f32_tiled=mock.Mock(), proto_threshold_signed=mock.Mock())
        ex = object.__new__(b._ResidentTensorRTRingExecutor)
        ex.torch = SimpleNamespace(float16=fp16, float32=fp32)
        ex.kernels = kernels
        ex.post_policy = 'native_mask'
        ex.collect_slice_bboxes = True
        ex.confidence_threshold = .5
        ex.out_size, ex.native_h, ex.native_w = 65, 43, 67
        # Even if a malformed legacy state says closing is active, native dispatch
        # must never call any proto morphology or fused-affine kernel.
        ex.proto_hole_treatment_active = True
        ex._launch_native_mask_post = mock.Mock()
        refs = {key: object() for key in ('head', 'proto', 'indices', 'count', 'native_bbox',
                'compact_coeff', 'compact_proto_boxes', 'compact_confs', 'max_logit')}
        slot = SimpleNamespace(head=SimpleNamespace(shape=(1, 37, 101), dtype=fp32),
                               proto=SimpleNamespace(shape=(1, 32, 5, 66), dtype=fp32),
                               compact_count=mock.Mock(), post_stream=object(), _cupy_refs=refs)
        with mock.patch.object(b, '_cupy_external_stream', return_value='stream'):
            ex._launch_post(slot)
        self.assertEqual(kernels.compact_f32_tiled.call_args.args[2][1], 101)
        self.assertEqual(kernels.union_f32_f32_tiled.call_args.args[:2], ((3, 2), (32, 4)))
        ex._launch_native_mask_post.assert_called_once_with(slot, 'stream', refs['native_bbox'])
        kernels.proto_threshold_signed.assert_not_called()

    def _admission_fixture(self):
        from tests.test_trt_preflight_admission import TensorRTPreflightAdmissionTests
        fixture = TensorRTPreflightAdmissionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.source.view = SimpleNamespace(family='spherical')
        return fixture

    def test_native_admission_selects_native_policy_before_context_acquisition(self):
        fixture = self._admission_fixture()
        with self.assertRaisesRegex(b._ResidentTensorRTRingFatalError, 'acquire boundary'):
            fixture.run_admission()
        fixture.preflight.assert_not_called()
        self.assertEqual(fixture.acquire.call_args.kwargs['post_policy'], 'native_mask')

    def test_native_setup_failure_can_fallback_without_consuming_source(self):
        fixture = self._admission_fixture()
        fixture.acquire.side_effect = RuntimeError('setup rejected')
        fixture.source.next_direct_slot = mock.Mock()
        self.assertIsNone(fixture.run_admission())
        fixture.source.next_direct_slot.assert_not_called()
        fixture.source.reset_direct_ring.assert_called_once()
        self.assertFalse(getattr(fixture.source, '_native_trt_data_consumed', False))

    def test_native_execution_failure_is_fatal_and_cannot_replay_forward_passes(self):
        fixture = self._admission_fixture()
        sys.modules['torch'].zeros = mock.Mock(return_value=object())
        sys.modules['torch'].int32 = object()
        fixture.device_union.device = 'cuda:0'
        ex = SimpleNamespace(enqueue_inference=mock.Mock(side_effect=RuntimeError('enqueue failed')),
                             slots=[SimpleNamespace(head=SimpleNamespace(dtype='float32'),
                                                    proto=SimpleNamespace(dtype='float32'))])
        fixture.acquire.side_effect = None
        fixture.acquire.return_value = (ex, False)
        fixture.source.next_direct_slot = mock.Mock(return_value=(0, object()))
        with mock.patch.object(b, '_resident_trt_pipeline_invalidate') as invalidate:
            with self.assertRaisesRegex(b._ResidentTensorRTRingFatalError, 'replaying.*forbidden'):
                fixture.run_admission()
        fixture.source.next_direct_slot.assert_called_once()
        self.assertTrue(fixture.source._native_trt_data_consumed)
        fixture.source.reset_direct_ring.assert_not_called()
        invalidate.assert_called_once()

    def test_native_release_failure_is_fatal_after_all_forwards_while_legacy_is_unchanged(self):
        class Counts:
            def sum(self, **kwargs):
                return SimpleNamespace(item=lambda: 0)
            def __gt__(self, other):
                return self
        for family in ('spherical', 'radial', 'orthogonal'):
            fixture = self._admission_fixture()
            fixture.source.view = SimpleNamespace(family=family)
            sys.modules['torch'].zeros = mock.Mock(return_value=Counts())
            sys.modules['torch'].int32, sys.modules['torch'].int64 = object(), object()
            fixture.device_union.device = 'cuda:0'
            slots = [SimpleNamespace(head=SimpleNamespace(dtype='float32'),
                                     proto=SimpleNamespace(dtype='float32')) for _ in range(2)]
            ex = SimpleNamespace(slots=slots, enqueue_inference=mock.Mock(),
                                 enqueue_postprocess=mock.Mock(), synchronize=mock.Mock())
            fixture.acquire.side_effect = None
            fixture.acquire.return_value = (ex, False)
            fixture.source.next_direct_slot = mock.Mock(side_effect=list(enumerate(slots)))
            failure = RuntimeError('cached executor release rejected')
            with (self.subTest(family=family),
                  mock.patch.object(b, '_resident_trt_pipeline_release', side_effect=failure),
                  mock.patch.object(b, '_resident_trt_pipeline_invalidate') as invalidate,
                  self.assertRaises(RuntimeError) as caught):
                fixture.run_admission()
            self.assertEqual(ex.enqueue_inference.call_count, 2)
            self.assertEqual(ex.enqueue_postprocess.call_count, 2)
            fixture.source.reset_direct_ring.assert_not_called()
            if family != 'orthogonal':
                self.assertIsInstance(caught.exception, b._ResidentTensorRTRingFatalError)
                self.assertIn('replaying', str(caught.exception))
                self.assertTrue(fixture.source._native_trt_data_consumed)
                invalidate.assert_called_once()
            else:
                self.assertIs(caught.exception, failure)
                invalidate.assert_not_called()

    def test_worker_cannot_restart_native_source_after_post_ring_failure(self):
        from XTA import cuda_backend, workers
        tree = ast.parse(inspect.getsource(workers.run_prediction_volume_in_worker))
        retry = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
            and len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
            and isinstance(node.body[0].value, ast.Call)
            and isinstance(node.body[0].value.func, ast.Name) and node.body[0].value.func.id == '_predict')
        program = compile(ast.fix_missing_locations(ast.Module(body=[retry], type_ignores=[])),
                          '<worker-native-no-replay>', 'exec')
        for cls in (cuda_backend.GpuRenderedYoloSource, cuda_backend.GpuTileRenderedYoloSource):
            for consumed in (False, True):
                source = object.__new__(cls)
                source._native_trt_data_consumed = consumed
                source.close = mock.Mock()
                engine = SimpleNamespace(disable_resident_after_runtime_failure=mock.Mock())
                original_error = RuntimeError('post-ring native cleanup failed')
                predict = mock.Mock(side_effect=original_error)
                mask = np.ones((2, 3, 4), np.uint8)
                namespace = dict(_predict=predict, source=source, task={'result_mode': 'file', 'job_id': 'one'},
                    _ResidentTensorRTRingFatalError=b._ResidentTensorRTRingFatalError,
                    GpuRenderedYoloSource=cuda_backend.GpuRenderedYoloSource,
                    GpuTileRenderedYoloSource=cuda_backend.GpuTileRenderedYoloSource,
                    view=SimpleNamespace(name='spherical'), gpu_engine=engine, np=np,
                    result_mask=mask, result_conf=None, azimuthal_padding_mask=None,
                    azimuthal_padding_conf=None, retry_with_cpu_render=False)
                with self.subTest(source=cls.__name__, consumed=consumed), mock.patch('builtins.print'):
                    if consumed:
                        with self.assertRaises(b._ResidentTensorRTRingFatalError) as caught:
                            exec(program, namespace)
                        self.assertIs(caught.exception.__cause__, original_error)
                        self.assertFalse(namespace['retry_with_cpu_render'])
                        self.assertTrue(mask.all())
                        source.close.assert_not_called()
                        engine.disable_resident_after_runtime_failure.assert_not_called()
                    else:
                        exec(program, namespace)
                        self.assertTrue(namespace['retry_with_cpu_render'])
                        self.assertFalse(mask.any())
                predict.assert_called_once()


@unittest.skipUnless(os.environ.get('XTA_TEST_NATIVE_TRT_POST_CUDA') == '1', 'native ring CUDA opt-in')
class NativeRingCudaPostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA unavailable')
        cls.torch = torch
        cls.kernels = inference._resident_mask_kernels()
        if cls.kernels is None:
            raise unittest.SkipTest('CuPy/NVRTC unavailable')

    def _executor(self, head, proto, matrix, out_size, native_shape, track_conf=True):
        torch = self.torch
        ex = object.__new__(b._ResidentTensorRTRingExecutor)
        ex.torch, ex.kernels = torch, self.kernels
        ex.post_policy = 'native_mask'
        ex.native_h, ex.native_w = native_shape
        ex.out_size = out_size
        ex.track_conf = track_conf
        ex.collect_slice_bboxes = True
        ex.confidence_threshold = .5
        ex.proto_hole_treatment_active = False
        ex.dynamic_unit_descriptors = True
        ex.default_descriptor = inference.ResidentRingUnitDescriptor(
            0, 0, *native_shape, np.asarray(matrix, np.float32))
        ex.identity_native_warp, ex.native_to_out = ex._descriptor_warp(ex.default_descriptor)
        slot = SimpleNamespace(head=head.unsqueeze(0), proto=proto.unsqueeze(0),
                               post_stream=torch.cuda.Stream(), slot_id=0,
                               post_valid=False, infer_valid=False)
        ex._allocate_post_buffers(slot)
        ex._set_slot_unit_descriptor(slot, ex.default_descriptor)
        return ex, slot

    def test_native_post_matches_direct_tensors_counts_confidence_and_bboxes(self):
        torch = self.torch
        generator = torch.Generator(device='cuda').manual_seed(20260909)
        matrices = (
            (np.eye(2, 3, dtype=np.float32), 64, (64, 64)),
            (np.asarray(((.82, .11, -1.7), (-.09, .73, 2.25)), np.float32), 65, (49, 57)),
            (np.asarray(((1.13, 0, .5), (0, .91, -.5)), np.float32), 65, (67, 73)),
        )
        dtypes = ((torch.float32, torch.float32), (torch.float16, torch.float16),
                  (torch.float32, torch.float16), (torch.float16, torch.float32))
        with mock.patch.dict(os.environ, {'YOLO_TTA_DIRECT_TILED_PROTO_UNION': '1'}), \
                mock.patch.object(inference, 'gpu_flatten_conf_tracking_enabled', return_value=True):
            for head_dtype, proto_dtype in dtypes:
                for retained in (0, 1, 17):
                    for matrix, out_size, native_shape in matrices:
                        with self.subTest(head=head_dtype, proto=proto_dtype, retained=retained,
                                          out_size=out_size, native=native_shape):
                            anchors = 23
                            head = torch.zeros((37, anchors), device='cuda', dtype=head_dtype)
                            head[:2] = (torch.rand((2, anchors), generator=generator, device='cuda') * out_size).to(head_dtype)
                            head[2:4] = (torch.rand((2, anchors), generator=generator, device='cuda') * out_size * 1.4).to(head_dtype)
                            head[4, :retained] = .75
                            head[5:] = torch.randn((32, anchors), generator=generator, device='cuda').to(head_dtype)
                            proto = torch.randn((32, 15, 18), generator=generator, device='cuda').to(proto_dtype)
                            ex, slot = self._executor(head, proto, matrix, out_size, native_shape)
                            torch.cuda.synchronize()
                            with torch.cuda.stream(slot.post_stream):
                                ex._launch_post(slot)
                            slot.post_stream.synchronize()
                            im = torch.empty((1, 1, out_size, out_size), device='cuda')
                            reference = inference._build_direct_device_compacted_payload(
                                head, proto, im, .5, min_conf_applied=False)
                            self.assertIsNotNone(reference)
                            expected, conf = inference._torch_warp_planes_to_native(
                                [reference.union_gpu, reference.conf_gpu], matrix, out_size, *native_shape)
                            expected = expected > .5
                            expected_conf = torch.where(expected, (conf.clamp(0., 1.) * 255.).round(), 0.).to(torch.uint8)
                            torch.cuda.synchronize()
                            self.assertTrue(torch.equal(slot.native_union, expected.to(torch.uint8)))
                            self.assertTrue(torch.equal(slot.native_conf, expected_conf))
                            self.assertEqual(slot.compact_count.item(), retained)
                            self.assertEqual(reference.instance_count_device.item(), retained)
                            points = expected.nonzero()
                            expected_box = ([int(points[:, 0].min()), int(points[:, 0].max()) + 1,
                                             int(points[:, 1].min()), int(points[:, 1].max()) + 1]
                                            if points.numel() else [native_shape[0], 0, native_shape[1], 0])
                            self.assertEqual(slot.native_bbox.tolist(), expected_box)

    def test_thin_sign_boundary_requires_threshold_before_nearest_warp(self):
        torch = self.torch
        matrix = np.asarray(((1, 0, .49), (0, 1, -.49)), np.float32)
        head = torch.zeros((37, 1), device='cuda')
        proto = torch.zeros((32, 2, 2), device='cuda')
        ex, slot = self._executor(head, proto, matrix, 8, (8, 8), track_conf=False)
        # An asymmetric sign boundary exposes the legacy fused-affine ordering.
        slot.max_logit.copy_(torch.tensor([[-.11, .89], [-.11, .89]], device='cuda'))
        torch.cuda.synchronize()
        with torch.cuda.stream(slot.post_stream):
            ex._launch_native_mask_post(slot, inference._cupy_external_stream(
                self.kernels.cp, slot.post_stream), np.uintp(0))
        slot.post_stream.synchronize()
        import torch.nn.functional as F
        network = (F.interpolate(slot.max_logit[None, None], size=(8, 8),
                                mode='bilinear', align_corners=False)[0, 0] > 0).float()
        expected = inference._torch_warp_planes_to_native([network], matrix, 8, 8, 8)[0].to(torch.uint8)
        fused = torch.empty_like(slot.native_union)
        self.kernels.upsample_quantize_affine(((1, 1)), (32, 8),
            (self.kernels.cp.asarray(slot.max_logit), np.uintp(0), np.int32(2), np.int32(2),
             np.int32(8), np.int32(8), np.int32(8), np.int32(8), *slot.native_to_out,
             self.kernels.cp.asarray(fused), np.uintp(0), np.uintp(0)),
            stream=inference._cupy_external_stream(self.kernels.cp, torch.cuda.current_stream()))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(slot.native_union, expected))
        self.assertFalse(torch.equal(fused, expected), 'fixture must distinguish legacy fused ordering')

    def test_native_post_graph_replays_new_outputs_and_survives_identical_destination(self):
        torch = self.torch
        head = torch.zeros((37, 5), device='cuda')
        head[:4] = torch.tensor([[16.], [16.], [32.], [32.]], device='cuda')
        head[5] = 1.
        proto = torch.zeros((32, 8, 8), device='cuda')
        proto[0, :, 3:5] = 1.
        matrix = np.asarray(((1, 0, .49), (0, 1, -.49)), np.float32)
        ex, slot = self._executor(head, proto, matrix, 32, (32, 32), track_conf=False)
        ex.slots = [slot]
        ex.dynamic_unit_descriptors = False
        ex.post_graph_count = 0
        torch.cuda.synchronize()
        with mock.patch.object(b, 'resident_trt_cuda_graphs_enabled', return_value=True):
            ex._refresh_native_post_graphs()
        self.assertEqual(ex.post_graph_count, 1)

        graph = slot.post_graph
        ex.reconfigure_destination(native_h=32, native_w=32, M_out_to_native=matrix.copy(),
                                   dynamic_unit_descriptors=False)
        self.assertIs(slot.post_graph, graph)
        for retained in (5, 0, 1):
            head[4].zero_()
            head[4, :retained] = .75
            slot.post_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(slot.post_stream):
                graph.replay()
            slot.post_stream.synchronize()
            self.assertEqual(slot.compact_count.item(), retained)
            self.assertEqual(bool(slot.native_union.any()), bool(retained))
            reference = inference._build_direct_device_compacted_payload(
                head, proto, torch.empty((1, 1, 32, 32), device='cuda'), .5, min_conf_applied=False)
            expected = inference._torch_warp_planes_to_native([reference.union_gpu], matrix, 32, 32, 32)[0]
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(slot.native_union, expected.to(torch.uint8)))
        changed_matrix = np.asarray(((.91, .1, -1.25), (0, .86, 2.5)), np.float32)
        with mock.patch.object(b, 'resident_trt_cuda_graphs_enabled', return_value=True):
            ex.reconfigure_destination(native_h=29, native_w=35, M_out_to_native=changed_matrix,
                                       dynamic_unit_descriptors=False)
        self.assertIsNot(slot.post_graph, graph)
        self.assertEqual(ex.post_graph_count, 1)
        with torch.cuda.stream(slot.post_stream):
            slot.post_graph.replay()
        slot.post_stream.synchronize()
        expected = inference._torch_warp_planes_to_native([reference.union_gpu], changed_matrix, 32, 29, 35)[0]
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(slot.native_union, expected.to(torch.uint8)))
        graph = slot.post_graph
        ex.configure_slice_bbox_collection(False)
        self.assertIsNone(slot.post_graph)
        with mock.patch.object(b, 'resident_trt_cuda_graphs_enabled', return_value=True):
            ex.reconfigure_destination(native_h=29, native_w=35, M_out_to_native=changed_matrix,
                                       dynamic_unit_descriptors=False)
        self.assertIsNotNone(slot.post_graph)
        self.assertIsNot(slot.post_graph, graph)
        self.assertEqual(ex.post_graph_count, 1)


    def test_native_legacy_native_switch_preserves_inference_owners_and_morphology(self):
        torch = self.torch
        head = torch.zeros((37, 1), device='cuda')
        head[:4, 0] = torch.tensor([16., 16., 32., 32.], device='cuda')
        head[4, 0] = .75
        head[5, 0] = 1.
        proto = torch.zeros((32, 8, 8), device='cuda')
        proto[0].fill_(2.)
        proto[0, 3:5, 3:5] = -2.
        matrix = np.eye(2, 3, dtype=np.float32)
        ex, slot = self._executor(head, proto, matrix, 32, (32, 32), track_conf=False)
        ex.slots = [slot]
        slot.context, slot.infer_graph = object(), object()
        slot.input = torch.empty((1, 1, 32, 32), device='cuda')
        slot.render_meta, slot._render_cupy_refs = object(), {'renderer': object()}
        preserved = (slot.context, slot.infer_graph, slot.input.data_ptr(),
                     slot.head.data_ptr(), slot.proto.data_ptr(), slot.render_meta, slot._render_cupy_refs)
        ex._execute_context = mock.Mock(side_effect=AssertionError('policy switch executed inference'))
        torch.cuda.synchronize()
        masks = []
        with mock.patch.object(b, 'proto_hole_treatment_mode', return_value='close'), \
                mock.patch.object(b, 'proto_hole_treatment_radius', return_value=1), \
                mock.patch.object(b, 'resident_trt_cuda_graphs_enabled', return_value=True):
            for policy in ('native_mask', 'legacy_proto', 'native_mask'):
                ex.reconfigure_post_policy(post_policy=policy, track_conf=False,
                    confidence_threshold=.5, collect_slice_bboxes=True, native_h=32, native_w=32,
                    M_out_to_native=matrix, dynamic_unit_descriptors=False)
                self.assertEqual(ex.proto_hole_treatment_active, policy == 'legacy_proto')
                self.assertEqual(slot.proto_tmp is not None, policy == 'legacy_proto')
                self.assertEqual(slot.compact_coeff is not None, policy == 'native_mask')
                self.assertEqual(ex.dynamic_unit_descriptors, policy == 'legacy_proto')
                if slot.post_graph is not None:
                    with torch.cuda.stream(slot.post_stream):
                        slot.post_graph.replay()
                    slot.post_stream.synchronize()
                masks.append(slot.native_union.clone())
                torch.cuda.synchronize()
                self.assertEqual((slot.context, slot.infer_graph, slot.input.data_ptr(),
                    slot.head.data_ptr(), slot.proto.data_ptr(), slot.render_meta, slot._render_cupy_refs), preserved)
                self.assertEqual(slot.compact_count.item(), 1)
        ex._execute_context.assert_not_called()
        self.assertTrue(torch.equal(masks[0], masks[2]))
        self.assertFalse(torch.equal(masks[0], masks[1]))
        self.assertTrue(bool(masks[1].all()), 'legacy closing should fill the enclosed prototype hole')
        # Every former post-only cache field may now change without replacing
        # inference buffers. Exercise thresholds, confidence, mode and radius.
        for policy, track_conf, threshold, mode, radius, expected_count in (
            ('legacy_proto', True, .9, 'close', 1, 0),
            ('legacy_proto', True, .5, 'close', 1, 1),
            ('legacy_proto', False, .5, 'off', 2, 1),
            ('legacy_proto', False, .5, 'close', 0, 1),
            ('legacy_proto', False, .5, 'close', 2, 1),
            ('native_mask', False, .5, 'close', 2, 1),
        ):
            with self.subTest(policy=policy, confidence=track_conf, threshold=threshold, mode=mode, radius=radius), \
                    mock.patch.object(b, 'proto_hole_treatment_mode', return_value=mode), \
                    mock.patch.object(b, 'proto_hole_treatment_radius', return_value=radius), \
                    mock.patch.object(b, 'resident_trt_cuda_graphs_enabled', return_value=True):
                ex.reconfigure_post_policy(post_policy=policy, track_conf=track_conf,
                    confidence_threshold=threshold, collect_slice_bboxes=True, native_h=32, native_w=32,
                    M_out_to_native=matrix, dynamic_unit_descriptors=False)
                self.assertEqual(ex.proto_hole_radius, radius)
                self.assertEqual(ex.proto_hole_treatment_active,
                                 policy == 'legacy_proto' and not track_conf and mode == 'close' and radius > 0)
                self.assertEqual(slot.native_conf is not None, track_conf)
                self.assertEqual(slot.conf_proto is not None, track_conf)
                self.assertEqual('native_conf' in slot._cupy_refs, track_conf)
                self.assertEqual('conf_proto' in slot._cupy_refs, track_conf)
                self.assertEqual(slot.compact_count.item(), expected_count)
                if track_conf:
                    # Foreground interpolation can reach a proto cell whose own
                    # logit is negative; its nearest confidence remains zero.
                    with mock.patch.object(inference, 'gpu_flatten_conf_tracking_enabled', return_value=True):
                        reference = inference._build_direct_device_compacted_payload(
                            head, proto, torch.empty((1, 1, 32, 32), device='cuda'),
                            threshold, min_conf_applied=False)
                    expected_conf = torch.where(
                        slot.native_union != 0, (reference.conf_gpu.clamp(0., 1.) * 255.).round(), 0,
                    ).to(torch.uint8)
                    self.assertTrue(torch.equal(slot.native_conf, expected_conf))
                self.assertEqual((slot.context, slot.infer_graph, slot.input.data_ptr(),
                    slot.head.data_ptr(), slot.proto.data_ptr(), slot.render_meta, slot._render_cupy_refs), preserved)
        ex._execute_context.assert_not_called()

if __name__ == '__main__':
    unittest.main()
