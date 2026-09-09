"""Generic inference reuses the qualified packed FP16 union without changing masks."""
from contextlib import ExitStack, redirect_stdout
import io
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import inference


class DirectTiledDispatchTests(unittest.TestCase):
    def test_layout_diagnostic_names_precise_guards_once_and_retains_no_tensors(self):
        torch_mod = SimpleNamespace(float16='float16')
        cases = (
            ('float16', 'float16', (32, 5, 66), True, True, 'tiled_f16', 'eligible'),
            ('float32', 'float16', (32, 5, 66), True, True, 'scalar', 'head_dtype_not_float16'),
            ('float16', 'float32', (32, 5, 66), True, True, 'scalar', 'proto_dtype_not_float16'),
            ('float16', 'float16', (16, 5, 66), True, True, 'scalar', 'prototype_channels_not_32'),
            ('float16', 'float16', (32, 5, 65), True, True, 'scalar', 'prototype_width_not_even'),
            ('float16', 'float16', (32, 5, 66), False, True, 'scalar', 'tiled_option_disabled'),
            ('float16', 'float16', (32, 5, 66), True, False, 'scalar', 'tiled_disabled_after_workspace_failure'),
            ('float16', 'float16', (32, 5, 66), True, True, 'scalar_workspace_fallback', 'tiled_workspace_allocation_failed'),
        )
        with mock.patch.object(inference, '_DIRECT_COMPACTION_LAYOUTS', set()):
            for hd, pd, shape, enabled, allowed, kernel, reason in cases:
                output = io.StringIO()
                head = SimpleNamespace(shape=(5 + shape[0], 257), dtype=hd)
                proto = SimpleNamespace(shape=shape, dtype=pd)
                with self.subTest(reason=reason), redirect_stdout(output):
                    for _ in range(3):
                        inference._announce_direct_compaction_layout(torch_mod, head, proto,
                            enabled=enabled, allow_tiled=allowed, kernel_name=kernel)
                lines = output.getvalue().splitlines()
                self.assertEqual(len(lines), 1)
                self.assertIn(f'head_shape={head.shape}, head_dtype={hd}', lines[0])
                self.assertIn(f'proto_shape={shape}, proto_dtype={pd}', lines[0])
                self.assertIn(f'tiled_option={enabled}, selected={kernel}, reason={reason}', lines[0])
            self.assertEqual(len(inference._DIRECT_COMPACTION_LAYOUTS), len(cases))
            self.assertTrue(all(isinstance(key[1], str) and isinstance(key[3], str)
                                for key in inference._DIRECT_COMPACTION_LAYOUTS))

    def test_layout_diagnostic_has_a_process_wide_bound(self):
        output = io.StringIO()
        with mock.patch.object(inference, '_DIRECT_COMPACTION_LAYOUTS', set()), redirect_stdout(output):
            for anchors in range(1, 81):
                inference._announce_direct_compaction_layout(SimpleNamespace(float16='float16'),
                    SimpleNamespace(shape=(37, anchors), dtype='float16'),
                    SimpleNamespace(shape=(32, 3, 4), dtype='float16'),
                    enabled=True, allow_tiled=True, kernel_name='tiled_f16')
            self.assertEqual(len(inference._DIRECT_COMPACTION_LAYOUTS), 64)
        self.assertEqual(len(output.getvalue().splitlines()), 64)

    def test_layout_selection_owner_retention_and_bounded_allocation_fallback(self):
        import torch
        for dtype, channels, width, enabled, oom, expected in (
            (torch.float16, 32, 66, True, False, 'tiled_f16'),
            (torch.float16, 32, 65, True, False, 'scalar'),
            (torch.float16, 16, 66, True, False, 'scalar'),
            (torch.float32, 32, 66, True, False, 'scalar'),
            (torch.float16, 32, 66, False, False, 'scalar'),
            (torch.float16, 32, 66, True, True, 'scalar_workspace_fallback'),
        ):
            with self.subTest(dtype=dtype, channels=channels, width=width, enabled=enabled, oom=oom):
                head = torch.zeros((5 + channels, 5), dtype=dtype)
                head[4] = .75
                proto = torch.ones((channels, 5, width), dtype=dtype)
                image = torch.zeros((1, 1, 15, width * 3))
                compact = mock.Mock(side_effect=lambda grid, block, args, **kw: args[4].fill_(5))
                tiled_compact = mock.Mock(side_effect=lambda grid, block, args, **kw: args[4].fill_(5))
                union = mock.Mock(side_effect=lambda grid, block, args, **kw: args[10].fill_(2))
                tiled_union = mock.Mock(side_effect=lambda grid, block, args, **kw: args[8].fill_(2))
                kernels = SimpleNamespace(cp=SimpleNamespace(asarray=lambda value: value),
                    compact_f16=compact, compact_f32=compact, compact_f16_tiled=tiled_compact,
                    union_f16_f16=union, union_f32_f32=union, union_f16_f16_tiled=tiled_union)
                original_empty = torch.empty

                def empty(shape, **kwargs):
                    if oom and shape == (5, channels):
                        raise torch.OutOfMemoryError('bounded optional workspace unavailable')
                    return original_empty(shape, **kwargs)

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.dict(os.environ, {'YOLO_TTA_DIRECT_TILED_PROTO_UNION': str(int(enabled))}))
                    stack.enter_context(mock.patch.object(inference, '_resident_mask_kernels', return_value=kernels))
                    stack.enter_context(mock.patch.object(inference, '_cupy_external_stream', return_value=None))
                    stack.enter_context(mock.patch.object(inference, 'gpu_flatten_conf_tracking_enabled', return_value=False))
                    stack.enter_context(mock.patch.object(inference, 'angle_variant_gpu_fastpath', return_value=None))
                    stack.enter_context(mock.patch.object(torch.cuda, 'current_stream', return_value=None))
                    stack.enter_context(mock.patch.object(torch.cuda, 'Event', return_value=SimpleNamespace(record=lambda _: None)))
                    stack.enter_context(mock.patch.object(torch, 'empty', side_effect=empty))
                    payload = inference._build_direct_device_compacted_payload(head, proto, image, .5)
                self.assertIsNotNone(payload)
                self.assertEqual(payload.compaction_kernel, expected)
                self.assertEqual(int(payload.instance_count_device[0]), 5)
                self.assertTrue(bool(payload.union_gpu.all()))
                if expected == 'tiled_f16':
                    compact.assert_not_called(); union.assert_not_called()
                    tiled_compact.assert_called_once(); tiled_union.assert_called_once()
                    self.assertEqual(tiled_union.call_args.args[:2], ((2, 2), (32, 4)))
                    owners = payload.device_refs[-6:-3]
                    self.assertEqual([tuple(owner.shape) for owner in owners], [(5, 32), (5, 4), (5,)])
                    self.assertEqual([owner.dtype for owner in owners], [torch.float16, torch.float32, torch.float32])
                else:
                    compact.assert_called_once(); union.assert_called_once()
                    tiled_compact.assert_not_called(); tiled_union.assert_not_called()

    def test_generic_loop_reports_one_aggregate_kernel_summary(self):
        import torch
        head = torch.zeros((1, 37, 5))
        proto = torch.zeros((1, 32, 3, 4))
        image = torch.zeros((1, 1, 12, 16))
        backend = mock.Mock(return_value=(head, proto))
        backend.names = {0: 'foreground'}
        predictor = SimpleNamespace(model=backend, preprocess=lambda value: value)
        source = [([], image, []) for _ in range(3)]
        payloads = [inference.GpuFlattenedRetinaPayload(None, None, compaction_kernel=name)
                    for name in ('tiled_f16', 'scalar_workspace_fallback', 'scalar')]
        output = io.StringIO()
        with mock.patch.object(inference, '_build_direct_device_compacted_payload', side_effect=payloads) as build, \
                mock.patch.object(inference, 'angle_variant_gpu_fastpath', return_value=None), \
                redirect_stdout(output):
            actual = list(inference._direct_predict_stream(predictor, source, SimpleNamespace(conf=.5), 'summary-test'))
        self.assertEqual(len(actual), 3)
        lines = [line for line in output.getvalue().splitlines() if line.startswith('Direct compaction summary-test:')]
        self.assertEqual(len(lines), 1)
        self.assertIn('tiled_f16=1, scalar=1, scalar_workspace_fallback=1, synchronized=0, failed_probes=0', lines[0])
        self.assertEqual([call.kwargs['allow_tiled'] for call in build.call_args_list], [True, True, False])


@unittest.skipUnless(os.environ.get('XTA_TEST_DIRECT_TILED_CUDA') == '1', 'explicit CUDA functional qualification only')
class DirectTiledCudaParityTests(unittest.TestCase):
    def test_scalar_and_tiled_logits_masks_confidence_and_counts_are_exact(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        self.assertIsNotNone(inference._resident_mask_kernels())
        generator = torch.Generator(device='cuda').manual_seed(142754)
        for ph, pw, anchors, retained in ((3, 2, 1, 0), (5, 66, 7, 1), (9, 130, 67, 5), (13, 34, 257, 257)):
            for want_conf in (False, True):
                with self.subTest(shape=(ph, pw), retained=retained, confidence=want_conf):
                    ih, iw = ph * 3 + 1, pw * 3 + 1
                    head = torch.randn((37, anchors), generator=generator, device='cuda', dtype=torch.float16)
                    head[0] = torch.linspace(-2, iw + 2, anchors, device='cuda', dtype=torch.float16)
                    head[1] = torch.linspace(ih + 2, -2, anchors, device='cuda', dtype=torch.float16)
                    head[2:4].abs_().mul_(max(ih, iw) * .8)
                    head[4].fill_(.25)
                    head[4, :retained] = .5
                    proto = torch.randn((32, ph, pw), generator=generator, device='cuda', dtype=torch.float16)
                    image = torch.empty((1, 1, ih, iw), device='cuda', dtype=torch.float16)
                    payloads = []
                    for enabled in (False, True):
                        with mock.patch.dict(os.environ, {'YOLO_TTA_DIRECT_TILED_PROTO_UNION': str(int(enabled))}), \
                                mock.patch.object(inference, 'gpu_flatten_conf_tracking_enabled', return_value=want_conf), \
                                mock.patch.object(inference, 'angle_variant_gpu_fastpath', return_value=None):
                            payload = inference._build_direct_device_compacted_payload(head, proto, image, .5)
                        self.assertIsNotNone(payload)
                        payloads.append(payload)
                    torch.cuda.synchronize()
                    scalar, tiled = payloads
                    self.assertEqual((scalar.compaction_kernel, tiled.compaction_kernel), ('scalar', 'tiled_f16'))
                    for left, right in ((scalar.device_refs[4], tiled.device_refs[4]),
                                        (scalar.union_gpu, tiled.union_gpu),
                                        (scalar.instance_count_device, tiled.instance_count_device)):
                        self.assertTrue(torch.equal(left, right))
                    self.assertEqual(int(tiled.instance_count_device.item()), retained)
                    if want_conf:
                        self.assertTrue(torch.equal(scalar.device_refs[5], tiled.device_refs[5]))
                        self.assertTrue(torch.equal(scalar.conf_gpu, tiled.conf_gpu))


if __name__ == '__main__':
    unittest.main()
