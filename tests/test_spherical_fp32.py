"""Typed Spherical plans and opt-in FP32 virtual-cube dispatch qualification."""
from itertools import product
import io
import os
import threading
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import spherical_cuda as sc
from tests.test_spherical_cuda import resident_engine, spherical_view, rotation_xyz


class SphericalFp32PlanTests(unittest.TestCase):
    def setUp(self):
        self.engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        self.view = spherical_view()
        self.key = sc._render_contract(self.engine, self.view, 0)[2]

    def test_typed_keys_and_casts_keep_reference_plans_independent(self):
        blocks64 = list(sc._direction_blocks(self.engine, self.key))
        blocks32 = list(sc._direction_blocks(self.engine, self.key, dtype=np.float32))
        self.assertEqual(len(self.engine._spherical_direction_cache), 2)
        self.assertEqual(self.engine._spherical_direction_cache_bytes, 7 * 7 * (25 + 13))
        self.assertEqual(blocks64[0][1].dtype, self.engine.torch.float64)
        self.assertEqual(blocks32[0][1].dtype, self.engine.torch.float32)
        np.testing.assert_array_equal(blocks32[0][1].numpy(), blocks64[0][1].numpy().astype(np.float32))
        np.testing.assert_array_equal(blocks32[0][2].numpy(), blocks64[0][2].numpy())
        self.assertEqual({key[-1] for key in self.engine._spherical_direction_cache}, {'<f4', '<f8'})
        with self.assertRaisesRegex(ValueError, 'only float32 or float64'):
            list(sc._direction_blocks(self.engine, self.key, dtype=np.float16))
        sc.clear_spherical_render_cache(self.engine)
        self.assertFalse(self.engine._spherical_direction_cache)
        self.assertEqual(self.engine._spherical_direction_cache_bytes, 0)

    def test_two_fp32_plans_fit_budget_and_mixed_precision_evicts_by_actual_bytes(self):
        budget = 7 * 7 * 13 * 2
        with mock.patch.object(sc, '_DIRECTION_CACHE_BYTES', budget):
            first = list(sc._direction_blocks(self.engine, self.key, dtype=np.float32))
            other_key = sc._render_contract(self.engine, spherical_view(face=1), 0)[2]
            list(sc._direction_blocks(self.engine, other_key, dtype=np.float32))
            self.assertEqual(len(self.engine._spherical_direction_cache), 2)
            self.assertIs(list(sc._direction_blocks(self.engine, self.key, dtype=np.float32))[0][1], first[0][1])
            list(sc._direction_blocks(self.engine, self.key, dtype=np.float64))
            self.assertEqual(len(self.engine._spherical_direction_cache), 1)
            self.assertEqual(self.engine._spherical_direction_cache_bytes, 7 * 7 * 25)

    def test_device_cache_oom_uses_typed_strips_without_partial_entry(self):
        engine = self.engine
        view = spherical_view(size=70, intervals=66)
        key = sc._render_contract(engine, view, 0)[2]
        original = engine.torch.as_tensor
        def upload(host, **kwargs):
            if host.shape == (70, 70, 3):
                raise engine.torch.OutOfMemoryError('whole-plan device allocation failed')
            return original(host, **kwargs)
        with mock.patch.object(engine.torch, 'as_tensor', side_effect=upload):
            blocks = list(sc._direction_blocks(engine, key, dtype=np.float32))
        self.assertEqual([row for row, _, _ in blocks], [0, 64])
        self.assertTrue(all(rays.dtype == engine.torch.float32 for _, rays, _ in blocks))
        self.assertFalse(engine._spherical_direction_cache)
        self.assertEqual(engine._spherical_direction_cache_bytes, 0)
        self.assertEqual(sc.spherical_direction_cache_stats(engine)['cache_device_fallbacks'], 1)

    def test_fp32_launch_uses_float_rays_radius_and_render_stream(self):
        self.engine._stream = object()
        self.engine._fused_cupy_volume = mock.Mock(return_value='source')
        kernel = mock.Mock(side_effect=lambda grid, block, args, **kwargs: args[3].fill_(127))
        kernels = SimpleNamespace(cp=SimpleNamespace(asarray=lambda value: value), kernel=kernel)
        with mock.patch.object(sc, '_spherical_fp32_kernels', return_value=kernels), \
                mock.patch.dict(os.environ, {'YOLO_TTA_GPU_SPHERICAL_FP32': '1'}), \
                mock.patch('XTA.geometry._cupy_external_stream', return_value='render-stream'):
            output = sc._render_spherical_fp32_cuda(self.engine, self.view, 1)
        args = kernel.call_args.args[2]
        self.assertEqual(args[1].dtype, self.engine.torch.float32)
        self.assertIsInstance(args[-1], np.float32)
        self.assertEqual(kernel.call_args.kwargs['stream'], 'render-stream')
        self.assertTrue(bool((output == 127).all()))


class SphericalFp32DispatchTests(unittest.TestCase):
    def test_guards_failure_and_reference_controls_select_actual_mode(self):
        for requested, native_enabled, shape, fail, expected in (
            (False, True, (9, 11, 13), False, 'reference_fp64'),
            (True, True, (9, 11, 13), False, 'fp32_virtual_cube'),
            (True, True, (4097, 11, 13), False, 'reference_fp64'),
            (True, True, (9, 11, 13), True, 'reference_fp64'),
            (True, False, (9, 11, 13), False, 'reference_torch'),
        ):
            engine = SimpleNamespace(_volume_gpu=SimpleNamespace(is_cuda=True, shape=shape), _logical_t=9)
            result = object()
            with self.subTest(requested=requested, native=native_enabled, shape=shape, fail=fail), \
                    mock.patch.object(sc, 'spherical_fp32_requested', return_value=requested), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL': str(int(native_enabled))}), \
                    mock.patch.object(sc, '_render_spherical_fp32_cuda', side_effect=RuntimeError('rejected') if fail else None, return_value=result) as fast, \
                    mock.patch.object(sc, '_render_spherical_native_cuda', return_value=result) as reference, \
                    mock.patch.object(sc, '_render_spherical_native_torch', return_value=result) as torch_reference, \
                    redirect_stdout(io.StringIO()) as output:
                for _ in range(2):
                    self.assertIs(sc.render_spherical_native_resident(engine, None, 0), result)
                self.assertEqual(engine._spherical_sampler_mode, expected)
                if fail:
                    self.assertEqual(fast.call_count, 1)
                    self.assertEqual(reference.call_count, 2)
                    self.assertIn('using the FP64 reference sampler', output.getvalue())
                elif expected == 'fp32_virtual_cube':
                    self.assertEqual(fast.call_count, 2)
                    reference.assert_not_called()
                elif expected == 'reference_torch':
                    self.assertEqual(torch_reference.call_count, 2)
                    fast.assert_not_called()
                else:
                    fast.assert_not_called()


@unittest.skipUnless(os.environ.get('XTA_TEST_SPHERICAL_FP32_CUDA') == '1', 'explicit CUDA qualification only')
class SphericalFp32CudaTests(unittest.TestCase):
    def setUp(self):
        self.flags = mock.patch.dict(os.environ, {'YOLO_TTA_GPU_SPHERICAL_FP32': '1'})
        self.flags.start()
        self.addCleanup(self.flags.stop)

    @unittest.skipUnless(os.environ.get('XTA_TEST_SPHERICAL_LARGE_SOURCE') == '1',
                         'explicit source-address qualification above 4 GiB only')
    def test_source_offsets_above_four_gib_remain_uint64(self):
        import torch
        from XTA.cuda_backend import _GpuWorkerRenderEngine
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        shape = (512, 3072, 3072)
        need = int(np.prod(shape, dtype=np.int64))
        free, _total = torch.cuda.mem_get_info(0)
        if free < need + 2 * 1024**3:
            self.skipTest('large-source qualification requires 4.5 GiB plus 2 GiB headroom')
        engine = object.__new__(_GpuWorkerRenderEngine)
        engine.torch = torch
        engine.device = torch.device('cuda:0')
        engine._stream = torch.cuda.Stream()
        engine._logical_t = 3072
        engine._fused_volume_ref = None
        engine._azimuthal_texture_lock = threading.RLock()
        engine._volume_gpu = None
        reference = actual = None
        minimum_offset = 501 * 3072 * 3072 + 1504 * 3072 + 1504
        self.assertGreater(minimum_offset, 2**32)
        self.assertEqual(need, 4_831_838_208)
        # PZ face center lies at pixel(2304,2304). Its logical T3008.5
        # addresses native slice501; every nonzero source byte is above2**32.
        view = spherical_view((3072, 3072, 3072), face=4, size=3072,
                              intervals=4608, origin=(0, 0), radii=(1473.,))
        try:
            with torch.cuda.stream(engine._stream):
                engine._volume_gpu = torch.zeros(shape, dtype=torch.uint8, device=engine.device)
                engine._volume_gpu[501, 1504:1568, 1504:1568] = 255
                reference = sc._render_spherical_native_cuda(engine, view, 0)
                actual = sc._render_spherical_fp32_cuda(engine, view, 0)
            engine._stream.synchronize()
            self.assertGreater(float(reference[2304, 2304]), 200.)
            self.assertGreater(float(actual[2304, 2304]), 200.)
            self.assertGreater(int(actual.count_nonzero()), 0)
            maximum_error = float((actual - reference).abs().max())
            self.assertLessEqual(maximum_error, 2.)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            print(f'Spherical uint64 source addressing: source_bytes={need}, '
                  f'minimum_nonzero_offset={minimum_offset}, foreground={int(actual.count_nonzero())}, '
                  f'center_gray8={float(actual[2304, 2304])}, max_abs_gray8={maximum_error}.', flush=True)
        finally:
            engine._stream.synchronize()
            sc.clear_spherical_render_cache(engine)
            reference = actual = None
            engine._fused_volume_ref = None
            engine._volume_gpu = None
            torch.cuda.empty_cache()

    def test_blocked_reference_reader_survives_typed_cache_eviction(self):
        import torch
        from tests.test_cylindrical_cuda import resident_engine as gpu_engine
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        engine = gpu_engine(np.random.default_rng(916).integers(0, 256, (17, 33, 35), np.uint8),
                            'cuda:0', logical_t=25)
        engine._stream = torch.cuda.Stream()
        view = spherical_view((25, 33, 35), size=32, intervals=30, radii=(4.,))
        key = sc._render_contract(engine, view, 0)[2]
        # All allocations/uploads and first-use compilation finish before the
        # gate. A CUDA host callback then blocks the reader until explicit release.
        expected = sc._render_spherical_native_cuda(engine, view, 0)
        engine._stream.synchronize()
        cached_dirs, cached_valid, reference_bytes = next(iter(engine._spherical_direction_cache.values()))
        dirs_pointer, valid_pointer = cached_dirs.data_ptr(), cached_valid.data_ptr()
        del cached_dirs, cached_valid
        prepared_fp32 = sc._build_cached_directions(engine, key, dtype=np.float32)
        # Warm allocator capacity and dtype-specific fill kernels before the
        # callback: CUDA lazy module loading can otherwise synchronize the device.
        warmed = [
            (torch.empty((32, 32, 3), dtype=torch.float64, device='cuda:0').fill_(0),
             torch.empty((32, 32), dtype=torch.bool, device='cuda:0').fill_(False))
            for _ in range(32)
        ]
        torch.cuda.current_stream().synchronize()
        del warmed
        entered, release = threading.Event(), threading.Event()
        timed_out = []
        def gate(_argument):
            entered.set()
            if not release.wait(10):
                timed_out.append(True)
        kernels = sc._spherical_kernels()
        from XTA.geometry import _cupy_external_stream
        external = _cupy_external_stream(kernels.cp, engine._stream)
        external.launch_host_func(gate, None)
        churn = []
        ticks = [('before render', time.perf_counter())]
        try:
            self.assertTrue(entered.wait(5), 'render-stream gate did not start')
            actual = sc._render_spherical_native_cuda(engine, view, 0)
            ticks.append(('after render', time.perf_counter()))
            with mock.patch.object(sc, '_DIRECTION_CACHE_BYTES', reference_bytes), \
                    mock.patch.object(sc, '_build_cached_directions', return_value=prepared_fp32):
                list(sc._direction_blocks(engine, key, dtype=np.float32))
            ticks.append(('after eviction', time.perf_counter()))
            self.assertEqual([entry[-1] for entry in engine._spherical_direction_cache], ['<f4'])
            for _ in range(32):
                directions = torch.empty((32, 32, 3), dtype=torch.float64, device='cuda:0').fill_(0)
                valid = torch.empty((32, 32), dtype=torch.bool, device='cuda:0').fill_(False)
                self.assertNotEqual(directions.data_ptr(), dirs_pointer)
                self.assertNotEqual(valid.data_ptr(), valid_pointer)
                churn.extend((directions, valid))
            ticks.append(('after churn', time.perf_counter()))
        finally:
            release.set()
            engine._stream.synchronize()
        self.assertFalse(timed_out, f'the test relied on the gate timeout: {[(name, at - ticks[0][1]) for name, at in ticks]}')
        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

    def test_production_patch_two_fp32_cache_entries_and_reference_control(self):
        import torch
        from tests.test_cylindrical_cuda import resident_engine as gpu_engine
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        engine = gpu_engine(np.full((17, 33, 35), 211, np.uint8), 'cuda:0', logical_t=25)
        view = spherical_view((25, 33, 35), size=3072, intervals=3000, origin=(-20, -20), radii=(4.,))
        key = sc._render_contract(engine, view, 0)[2]
        other = spherical_view((25, 33, 35), face=1, size=3072, intervals=3000, origin=(-20, -20), radii=(4.,))
        other_key = sc._render_contract(engine, other, 0)[2]
        with torch.cuda.stream(engine._stream):
            list(sc._direction_blocks(engine, key, dtype=np.float32))
            list(sc._direction_blocks(engine, other_key, dtype=np.float32))
            self.assertEqual(len(engine._spherical_direction_cache), 2)
            self.assertEqual(engine._spherical_direction_cache_bytes, 2 * 3072**2 * 13)
            self.assertLessEqual(engine._spherical_direction_cache_bytes, 256 * 1024**2)
            actual_bytes = sum(directions.nbytes + valid.nbytes for directions, valid, _ in
                               engine._spherical_direction_cache.values())
            self.assertEqual(actual_bytes, engine._spherical_direction_cache_bytes)
            with mock.patch.dict(os.environ, {'YOLO_TTA_FAST_GEOMETRY': '1', 'YOLO_TTA_GPU_SPHERICAL_FP32': '0'}):
                strict = sc.render_spherical_native_resident(engine, view, 0)
            reference = sc._render_spherical_native_cuda(engine, view, 0)
            engine._stream.synchronize()
            self.assertEqual(engine._spherical_sampler_mode, 'reference_fp64')
            self.assertTrue(torch.equal(strict.view(torch.uint8), reference.view(torch.uint8)))
            self.assertEqual(len(engine._spherical_direction_cache), 1)
            self.assertEqual(next(iter(engine._spherical_direction_cache))[-1], '<f8')
            fast = sc.render_spherical_native_resident(engine, view, 0)
            engine._stream.synchronize()
            self.assertEqual(engine._spherical_sampler_mode, 'fp32_virtual_cube')
            self.assertLessEqual(float((fast - strict).abs().max()), 2.)
        sc.clear_spherical_render_cache(engine)
        self.assertFalse(engine._spherical_direction_cache)
        self.assertEqual(engine._spherical_direction_cache_bytes, 0)

    def test_gray8_error_padding_and_stream_eviction_across_faces_and_t_ratios(self):
        import torch
        from tests.test_cylindrical_cuda import resident_engine as gpu_engine
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        generator = np.random.default_rng(914)
        for nt, lt in ((17, 25), (25, 25), (25, 17), (1, 9)):
            volume = generator.integers(0, 256, (nt, 33, 35), np.uint8)
            engine = gpu_engine(volume, 'cuda:0', logical_t=lt)
            engine._stream = torch.cuda.Stream()
            for face, rotation in product(range(6), (np.eye(3), rotation_xyz())):
                view = spherical_view((lt, 33, 35), face=face, rotation=rotation,
                                      size=40, intervals=34, origin=(-2, -2), radii=(.63, 4., 8.))
                for index in range(3):
                    with self.subTest(nt=nt, lt=lt, face=face, radius=index):
                        with torch.cuda.stream(engine._stream):
                            reference = sc._render_spherical_native_cuda(engine, view, index)
                        # Exercise allocation on the default stream and eviction
                        # immediately after launch, while CuPy readers can remain live.
                        actual = sc._render_spherical_fp32_cuda(engine, view, index)
                        sc.clear_spherical_render_cache(engine)
                        engine._stream.synchronize()
                        delta = (actual - reference).abs()
                        self.assertLessEqual(float(delta.max()), 2.)
                        self.assertEqual(int(actual[:2].count_nonzero()), 0)
                        self.assertEqual(int(actual[:, :2].count_nonzero()), 0)
                        self.assertTrue(bool(torch.isfinite(actual).all()))


if __name__ == '__main__':
    unittest.main()
