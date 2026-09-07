"""CPU contracts plus explicit, opt-in CUDA shell/inference qualification."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cuda_backend, cuda_d1, geometry, inference, pta_workers, runtime, workers
from XTA.inference_backends import (
    PipelineExtent, ResultContract, TaskRequirements,
    cuda_local_capabilities, openvino_local_capabilities,
)
from XTA.cylindrical_geometry import build_radial_view_infos, render_shell_frame
from XTA.config import resolve_channel_format


def shell_views(shape=(11, 13, 15), size=8):
    return build_radial_view_infos(
        *shape, targets=('transverse', 'sagittal', 'coronal'),
        min_radius=0.63, patch_size=size, tilted_views=(),
    )


def resident_engine(volume, device='cpu', logical_t=None):
    import torch
    import torch.nn.functional as functional
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine.torch, engine.F = torch, functional
    engine.device = torch.device(device)
    engine._volume_gpu = torch.as_tensor(volume, dtype=torch.uint8, device=engine.device)
    engine._volume_flat = engine._volume_gpu.reshape(-1)
    engine._logical_t = int(logical_t or volume.shape[0])
    engine._native_t_map_cache = {}
    engine._native_plane_cache = OrderedDict()
    engine._native_u8_plane_cache = OrderedDict()
    engine._mode = 'resident'
    engine._fused_volume_ref = None
    engine._azimuthal_texture_lock = threading.RLock()
    if str(device).startswith('cuda'):
        engine._stream = torch.cuda.current_stream(engine.device)
    return engine


def assert_thin_shell_boundaries(test_case, device='cpu'):
    """An independent point oracle plus every one-voxel stack orientation."""
    volume = np.zeros((1, 5, 7), dtype=np.uint8)
    volume[0, 2, 1] = 255
    view = next(v for v in build_radial_view_infos(
        *volume.shape, targets=('transverse',), min_radius=0.63, patch_size=4,
        tilted_views=(),
    ) if v.radial_arc_origin == 4.0)
    view = replace(view, radial_tilted_source=True, tilt_direction='vertical', tilt_angle_deg=30.0)
    actual = resident_engine(volume, device)._render_native_plane(view, view.num_slices - 1).cpu().numpy()
    # Pixel (height=0, arc=6), radius=2 lands fractionally beyond t=0;
    # its trilinear footprint still includes source voxel (0,2,1).
    px, py = 3.0 + 2.0 * math.cos(3.0), 2.0 + 2.0 * math.sin(3.0)
    tt = math.tan(math.radians(30.0)) * (py - 2.0)
    expected = round(255.0 * (1.0 - abs(tt)) * (1.0 - abs(py - 2.0)) * (1.0 - abs(px - 1.0)))
    test_case.assertGreater(expected, 0)
    test_case.assertEqual(int(actual[0, 2]), expected)
    test_case.assertEqual(int(np.count_nonzero(actual[1:])), 0)
    for base, shape in (('transverse', (1, 5, 7)), ('sagittal', (5, 1, 7)), ('coronal', (5, 7, 1))):
        volume = np.full(shape, 255, dtype=np.uint8)
        engine = resident_engine(volume, device)
        for view in build_radial_view_infos(
            *shape, targets=(base,), min_radius=0.63, patch_size=4, tilted_views=(),
        ):
            for direction in ('vertical', 'horizontal'):
                tilted = replace(view, radial_tilted_source=True, tilt_direction=direction, tilt_angle_deg=30.0)
                for index in sorted({0, tilted.num_slices - 1}):
                    with test_case.subTest(base=base, patch=view.name, direction=direction, index=index):
                        actual = engine._render_native_plane(tilted, index).cpu().numpy().astype(np.uint8)
                        np.testing.assert_array_equal(actual, render_shell_frame(volume, tilted, index))
                        test_case.assertEqual(int(np.count_nonzero(actual[1:])), 0,
                                              'padded height rows must remain zero after tilt')


class RadialGpuRoutingContracts(unittest.TestCase):
    def test_reference_fallback_is_explicit_logged_once_and_stops_repeated_kernel_failures(self):
        view = shell_views()[0]
        for enabled in (False, True):
            engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
            engine._volume_gpu = SimpleNamespace(is_cuda=True)
            result = object()
            engine._render_radial_native_resident_cuda = mock.Mock(side_effect=RuntimeError('test kernel rejection'))
            engine._render_radial_native_resident_torch = mock.Mock(return_value=result)
            with (self.subTest(kernel_enabled=enabled),
                  mock.patch.object(cuda_backend, 'radial_native_kernel_enabled', return_value=enabled),
                  mock.patch('builtins.print') as announce):
                self.assertIs(engine._render_radial_native_resident(view, 0), result)
                self.assertIs(engine._render_radial_native_resident(view, 1), result)
                self.assertEqual(engine._render_radial_native_resident_cuda.call_count, int(enabled))
                self.assertEqual(engine._render_radial_native_resident_torch.call_count, 2)
                announce.assert_called_once()
                self.assertTrue(announce.call_args.kwargs['flush'])

    def test_cpu_and_cuda_capabilities_allow_shell_patches_and_reject_unknown_families(self):
        view = shell_views()[0]
        self.assertTrue(runtime.cpu_inference_supports_view(view))
        self.assertFalse(runtime.cpu_inference_supports_view(replace(view, family='unknown')))
        self.assertFalse(runtime.cpu_inference_supports_view(replace(view, family='azimuthal')))
        requirements = TaskRequirements(
            task_kind='fullframe', view_family='radial', pipeline_extent=PipelineExtent.INFER_ONLY,
            acceptable_results=frozenset({ResultContract.TASK_ARTIFACT}), model_io_contract='yolo-seg-raw-v1',
        )
        for backend in (cuda_local_capabilities(), openvino_local_capabilities()):
            self.assertTrue(backend.supports(requirements))
            self.assertFalse(backend.supports(replace(requirements, view_family='unknown')))

    def test_shell_declines_legacy_fused_and_d1_before_device_access(self):
        view = shell_views()[0]
        engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
        self.assertEqual(cuda_backend._fused_preflight_family(view), '')
        self.assertFalse(engine._try_fused_render_into_ring_slot(None, view, None, 0, 8))
        with self.assertRaisesRegex(ValueError, 'Radial shells'):
            cuda_d1._d1_view_family_ids(view)
        with self.assertRaisesRegex(ValueError, 'Radial shell tasks'):
            workers.run_prediction_volume_in_worker(None, None, {
                'view': view, 'job': None, 'kind': 'fullframe', 'result_mode': 'd1_owner',
            })

    def test_shell_fullframe_and_tile_ring_preparation_reject_without_allocations(self):
        for source_class in (cuda_backend.GpuRenderedYoloSource, cuda_backend.GpuTileRenderedYoloSource):
            source = object.__new__(source_class)
            source.resident_ring_supported = False
            with self.subTest(source=source_class.__name__), self.assertRaisesRegex(RuntimeError, 'Radial shell'):
                source.prepare_direct_ring()

    def test_shell_channels_clamp_radius_without_azimuthal_mirror(self):
        view = shell_views()[0]
        self.assertEqual(geometry.channel_view_slice_source(view, -3), (0, False))
        self.assertEqual(geometry.channel_view_slice_source(view, view.num_slices + 5), (view.num_slices - 1, False))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Torch is optional')
class RadialTorchNumericalTests(unittest.TestCase):
    def test_native_kernel_launch_uses_scalar_metadata_and_no_host_coordinate_maps(self):
        volume = np.zeros((11, 13, 15), dtype=np.uint8)
        engine = resident_engine(volume)
        engine._stream = object()
        source_ref = object()
        engine._fused_cupy_volume = mock.Mock(return_value=source_ref)
        calls = []
        def kernel(grid, block, args, *, stream):
            calls.append((grid, block, args, stream))
            args[1].zero_()
        kernels = SimpleNamespace(cp=SimpleNamespace(asarray=lambda tensor: tensor), radial_native_f32=kernel)
        view = replace(shell_views()[0], radial_tilted_source=True,
                       tilt_angle_deg=30.0, tilt_direction='vertical')
        with (mock.patch.object(cuda_backend, '_radial_native_kernels', return_value=kernels),
              mock.patch.object(cuda_backend, '_cupy_external_stream', return_value='render-stream'),
              mock.patch.object(cuda_backend, 'radial_shell_coordinates', side_effect=AssertionError('host coordinate map')),
              mock.patch('builtins.print')):
            out = engine._render_radial_native_resident_cuda(view, 0)
        self.assertEqual(tuple(out.shape), (8, 8))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ((1, 1), (32, 8)))
        self.assertIs(calls[0][2][0], source_ref)
        self.assertTrue(all(isinstance(arg, np.generic) for arg in calls[0][2][2:]))
        self.assertEqual(len(calls[0][2]), 16)
        self.assertEqual(calls[0][3], 'render-stream')

    def test_thin_tilted_source_retains_boundary_taps_without_height_padding_leaks(self):
        assert_thin_shell_boundaries(self)

    def test_pta_radial_dispatch_uses_native_patches_then_tile_affine_and_clamped_channels(self):
        import torch
        volume = np.random.default_rng(12).integers(0, 256, (11, 13, 15), dtype=np.uint8)
        engine = resident_engine(volume)
        engine._stream = object()
        engine.ensure_volume_array = mock.Mock(return_value='resident')
        view = shell_views(volume.shape)[0]
        identity = np.asarray(((1, 0, 0), (0, 1, 0)), dtype=np.float32)
        shift = np.asarray(((1, 0, 2), (0, 1, 1)), dtype=np.float32)
        plan = SimpleNamespace(
            view=SimpleNamespace(shared_view=view),
            aff=SimpleNamespace(out_h=8, out_w=8, M_out_to_src=identity),
            channel_variant=SimpleNamespace(kind='custom', offsets=(-1, 0, 1)),
            source_encoded_indices=(),
            tile_layout=(SimpleNamespace(tile_tag='tile', out_h=8, out_w=8,
                                         shared_job=SimpleNamespace(M_out_to_src=shift)),),
        )
        runtime_state = {'torch': torch, 'azimuthal_renderer': engine,
                         'azimuthal_render_lock': threading.Lock(), 'azimuthal_texture_required': False}
        with (mock.patch.object(pta_workers, '_require_pta_canonical_plan'),
              mock.patch.object(torch.cuda, 'stream', return_value=nullcontext()),
              mock.patch.object(torch.cuda, 'Event', return_value=SimpleNamespace(record=lambda stream: None)),
              mock.patch('builtins.print')):
            for item in ('full', 'tile'):
                image, event = pta_workers._gpu_projected_item_image(runtime_state, volume, plan, 0, item)
                self.assertEqual(tuple(image.shape), (8, 8, 3))
                for channel, index in enumerate((0, 0, 1)):
                    native = render_shell_frame(volume, view, index)
                    expected = native
                    if item == 'tile':
                        expected = np.zeros_like(native)
                        expected[:7, :6] = native[1:, 2:]
                    np.testing.assert_array_equal(image[:, :, channel].numpy(), expected)
        self.assertTrue(runtime_state['radial_renderer_announced'])
        self.assertNotIn('cartesian_renderer_announced', runtime_state)

    def test_shared_coordinates_match_cpu_for_axes_seams_and_tilted_height_padding(self):
        volume = np.random.default_rng(42).integers(0, 256, (11, 13, 15), dtype=np.uint8)
        engine = resident_engine(volume)
        views = shell_views(volume.shape)
        for original in views:
            for direction in ('', 'vertical', 'horizontal'):
                view = replace(original, radial_tilted_source=bool(direction),
                               tilt_direction=direction, tilt_angle_deg=31.0)
                for index in sorted({0, view.num_slices - 1}):
                    with self.subTest(view=view.name, direction=direction, radius=index):
                        actual = engine._render_native_plane(view, index).numpy().astype(np.uint8)
                        expected = render_shell_frame(volume, view, index)
                        np.testing.assert_array_equal(actual, expected)

    def test_deferred_native_t_resize_quantizes_logical_voxels_before_shell_sampling(self):
        volume = np.random.default_rng(11).integers(0, 256, (5, 13, 15), dtype=np.uint8)
        engine = resident_engine(volume, logical_t=11)
        import torch
        logical = engine._resample_native_t_axis(engine._volume_gpu).to(torch.uint8).numpy()
        for view in shell_views(logical.shape):
            view = replace(view, radial_tilted_source=True, tilt_direction='horizontal', tilt_angle_deg=23.0)
            index = view.num_slices - 1
            with self.subTest(view=view.name):
                actual = engine._render_native_plane(view, index).numpy().astype(np.uint8)
                np.testing.assert_array_equal(actual, render_shell_frame(logical, view, index))

    def test_compaction_launch_supplies_complete_kernel_contract_for_both_head_dtypes(self):
        import torch
        for dtype in (torch.float16, torch.float32):
            calls = []
            def compact(grid, block, args, *, stream):
                head, anchors, threshold, indices, count, height, width, bbox = args
                calls.append((int(height), int(width), int(bbox)))
                chosen = torch.nonzero(head[4] >= float(threshold)).flatten()
                indices[:len(chosen)] = chosen.to(torch.int32)
                count[0] = len(chosen)
            def union(grid, block, args, *, stream):
                args[10].fill_(8.0 if int(args[3][0]) else -16.0)
            kernels = SimpleNamespace(cp=SimpleNamespace(asarray=lambda x: x),
                compact_f16=compact, compact_f32=compact,
                union_f16_f32=union, union_f32_f32=union)
            head = torch.zeros((7, 3), dtype=dtype)
            head[4] = torch.tensor([0.1, 0.5, 0.75], dtype=dtype)
            proto = torch.ones((2, 4, 6), dtype=torch.float32)
            image = torch.zeros((1, 1, 8, 12))
            with self.subTest(dtype=dtype), ExitStack() as stack:
                stack.enter_context(mock.patch.object(inference, '_resident_mask_kernels', return_value=kernels))
                stack.enter_context(mock.patch.object(inference, '_cupy_external_stream', return_value=None))
                stack.enter_context(mock.patch.object(inference, 'gpu_flatten_conf_tracking_enabled', return_value=False))
                stack.enter_context(mock.patch.object(inference, 'angle_variant_gpu_fastpath', return_value=None))
                stack.enter_context(mock.patch.object(torch.cuda, 'current_stream', return_value=None))
                stack.enter_context(mock.patch.object(torch.cuda, 'Event', return_value=SimpleNamespace(record=lambda stream: None)))
                for threshold, expected_count in ((0.3, 2), (0.95, 0)):
                    payload = inference._build_direct_device_compacted_payload(head, proto, image, threshold)
                    self.assertIsNotNone(payload)
                    self.assertEqual(int(payload.instance_count_device[0]), expected_count)
                    self.assertEqual(int(payload.union_gpu.count_nonzero()), 96 if expected_count else 0)
                self.assertEqual(calls, [(8, 12, 0), (8, 12, 0)])


@unittest.skipUnless(os.environ.get('XTA_RUN_CYLINDRICAL_CUDA_SMOKE') == '1', 'explicit CUDA smoke only')
class RadialCudaSmokeTests(unittest.TestCase):
    def test_actual_cuda_native_kernel_deferred_t_and_large_periodic_patch(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        kernels = cuda_backend._radial_native_kernels()
        self.assertIsNotNone(kernels, cuda_backend._RADIAL_NATIVE_KERNELS_ERROR)
        launch = mock.Mock(wraps=kernels.radial_native_f32)
        instrumented = SimpleNamespace(cp=kernels.cp, radial_native_f32=launch)
        max_abs, differing, compared = 0.0, 0, 0
        with (mock.patch.object(cuda_backend, '_radial_native_kernels', return_value=instrumented),
              mock.patch.object(cuda_backend, 'radial_shell_coordinates', side_effect=AssertionError('host coordinate map')),
              mock.patch.object(cuda_backend._GpuWorkerRenderEngine, '_render_radial_native_resident_torch',
                                side_effect=AssertionError('unexpected Torch fallback'))):
            source = np.random.default_rng(5).integers(0, 256, (5, 14, 18), dtype=np.uint8)
            engine = resident_engine(source, 'cuda:0', logical_t=11)
            logical = engine._resample_native_t_axis(engine._volume_gpu).to(torch.uint8).cpu().numpy()
            for view in shell_views(logical.shape):
                for direction in ('vertical', 'horizontal'):
                    view = replace(view, radial_tilted_source=True, tilt_direction=direction, tilt_angle_deg=-30.0)
                    for index in sorted({0, view.num_slices - 1}):
                        before = launch.call_count
                        actual = engine._render_native_plane(view, index).cpu().numpy()
                        self.assertEqual(launch.call_count, before + 1)
                        expected = render_shell_frame(logical, view, index)
                        delta = np.abs(actual - expected.astype(np.float32))
                        max_abs = max(max_abs, float(delta.max()))
                        differing += int(np.count_nonzero(delta))
                        compared += int(actual.size)
                        self.assertLessEqual(float(delta.max()), 1.0)
            source = np.random.default_rng(6).integers(0, 256, (5, 17, 19), dtype=np.uint8)
            view = build_radial_view_infos(*source.shape, targets=('transverse',),
                min_radius=0.63, patch_size=3072, tilted_views=())[0]
            engine = resident_engine(source, 'cuda:0')
            for index in (0, view.num_slices - 1):
                before = launch.call_count
                actual = engine._render_native_plane(view, index).cpu().numpy()
                self.assertEqual(launch.call_count, before + 1)
                # Compare every active-height pixel and every padded pixel,
                # while bounding the independent CPU oracle to the active rows.
                expected_active = render_shell_frame(source, replace(view, src_h=source.shape[0]), index)
                delta = np.abs(actual[:source.shape[0]] - expected_active.astype(np.float32))
                max_abs = max(max_abs, float(delta.max()))
                differing += int(np.count_nonzero(delta))
                compared += int(actual.size)
                self.assertLessEqual(float(delta.max()), 1.0)
                self.assertEqual(int(np.count_nonzero(actual[source.shape[0]:])), 0)
        print(f'Radial native CUDA parity: max_abs_gray8={max_abs:g}, '
              f'differing_pixels={differing}/{compared}, kernel_launches={launch.call_count}.', flush=True)

    def test_actual_cuda_thin_tilted_source_zero_extension_and_height_padding(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        assert_thin_shell_boundaries(self, 'cuda:0')

    def test_actual_cuda_native_shell_channels_and_tile_affines(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        volume = np.random.default_rng(23).integers(0, 256, (11, 13, 15), dtype=np.uint8)
        engine = cuda_backend._GpuWorkerRenderEngine('cuda:0')
        try:
            self.assertEqual(engine.ensure_volume_array(volume, identity='radial-cuda-smoke'), 'resident')
            selected = [v for v in shell_views(volume.shape) if v.radial_patch_index in (0, 2)]
            for original in selected:
                for direction in ('', 'vertical', 'horizontal'):
                    view = replace(original, radial_tilted_source=bool(direction),
                                   tilt_direction=direction, tilt_angle_deg=31.0)
                    for index in sorted({0, view.num_slices - 1}):
                        with torch.cuda.stream(engine._stream):
                            actual = engine._render_native_plane(view, index)
                        engine._stream.synchronize()
                        expected = render_shell_frame(volume, view, index)
                        np.testing.assert_array_equal(actual.cpu().numpy().astype(np.uint8), expected)
            view = selected[0]
            affine = geometry.build_affine(view.name, 8, 8, 8, 0.0, 'pad')
            job = geometry.AugJob('a0', 0.0, Path('unused.json'), affine)
            channel_format = resolve_channel_format('C3S1')
            batch, event = engine.render_fullframe_batch(view, job, [0, view.num_slices - 1],
                out_size=8, fp16=False, channel_format=channel_format)
            event.synchronize()
            for b, center in enumerate((0, view.num_slices - 1)):
                for channel, offset in enumerate((-1, 0, 1)):
                    index = min(max(center + offset, 0), view.num_slices - 1)
                    np.testing.assert_array_equal(np.rint(batch[b, channel].cpu().numpy() * 255).astype(np.uint8),
                                                  render_shell_frame(volume, view, index))
            # An integer translation checks actual affine/tile ordering without
            # inheriting OpenCV versus Torch subpixel coefficient differences.
            matrix = np.asarray(((1, 0, 2), (0, 1, 1)), dtype=np.float32)
            tile, event = engine.render_tile_batch(view, matrix, [0], out_size=8, fp16=False)
            event.synchronize()
            native = render_shell_frame(volume, view, 0)
            expected = np.zeros_like(native)
            expected[:7, :6] = native[1:, 2:]
            np.testing.assert_array_equal(np.rint(tile[0, 0].cpu().numpy() * 255).astype(np.uint8), expected)
        finally:
            engine.release_inference_assets()

    def test_actual_generic_compaction_fp16_fp32_confidence_and_empty_outputs(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        self.assertIsNotNone(inference._resident_mask_kernels(), 'CuPy/NVRTC kernels must compile')
        for head_dtype in (torch.float16, torch.float32):
            for proto_dtype in (torch.float16, torch.float32):
                head = torch.zeros((7, 3), dtype=head_dtype, device='cuda:0')
                head[:4] = torch.tensor([[6], [4], [12], [8]], dtype=head_dtype, device='cuda:0')
                head[4] = torch.tensor([0.1, 0.5, 0.75], dtype=head_dtype, device='cuda:0')
                head[5:] = 4.0
                proto = torch.ones((2, 4, 6), dtype=proto_dtype, device='cuda:0')
                image = torch.zeros((1, 1, 8, 12), device='cuda:0')
                with self.subTest(head=head_dtype, proto=proto_dtype):
                    for threshold, expected_count in ((0.3, 2), (0.95, 0)):
                        payload = inference._build_direct_device_compacted_payload(head, proto, image, threshold)
                        self.assertIsNotNone(payload, 'Generic CUDA compaction must not silently decline')
                        payload.ready_event.synchronize()
                        self.assertEqual(int(payload.instance_count_device.item()), expected_count)
                        self.assertEqual(int(payload.union_gpu.count_nonzero().item()), 96 if expected_count else 0)


if __name__ == '__main__':
    unittest.main()
