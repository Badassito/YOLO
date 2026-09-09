"""Exact Radial column-hoisting dispatch, fallback and optional CUDA parity."""
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cuda_backend as cb
from tests.test_cylindrical_cuda import resident_engine, shell_views


class RadialColumnDispatchTests(unittest.TestCase):
    def test_output_stream_is_recorded_before_scalar_or_column_launches(self):
        import torch
        for enabled in (False, True):
            engine = resident_engine(np.zeros((11, 13, 15), np.uint8))
            engine._stream = object()
            order = []
            output = SimpleNamespace(is_cuda=True,
                record_stream=lambda stream: order.append(('output_stream', stream)))
            geometry = SimpleNamespace(is_cuda=True,
                record_stream=lambda stream: order.append(('geometry_stream', stream)))
            def allocate(shape, **kwargs):
                return geometry if kwargs['dtype'] == torch.float64 else output
            def scalar(grid, block, args, **kwargs):
                self.assertIs(args[1], output)
                order.append(('scalar', kwargs['stream']))
            def columns(grid, block, args, **kwargs):
                self.assertIs(args[0], geometry)
                order.append(('columns', kwargs['stream']))
            def sample(grid, block, args, **kwargs):
                self.assertIs(args[2], output)
                order.append(('sample', kwargs['stream']))
            kernels = SimpleNamespace(cp=SimpleNamespace(asarray=lambda value: value),
                radial_native_f32=scalar, radial_columns_f64=columns, radial_native_columns_f32=sample)
            with self.subTest(columns=enabled), \
                    mock.patch.object(cb, '_radial_native_kernels', return_value=kernels), \
                    mock.patch.object(cb, '_cupy_external_stream', return_value=engine._stream), \
                    mock.patch.object(cb, 'radial_column_geometry_for_shape', return_value=enabled), \
                    mock.patch.object(torch, 'empty', side_effect=allocate), mock.patch('builtins.print'):
                self.assertIs(engine._render_radial_native_resident_cuda(shell_views()[0], 0), output)
            expected = ['output_stream', 'geometry_stream', 'columns', 'sample'] if enabled else ['output_stream', 'scalar']
            self.assertEqual([name for name, _ in order], expected)
            self.assertTrue(all(stream is engine._stream for _, stream in order))

    def test_bundle_size_guard_uses_active_rows_and_explicit_override(self):
        with mock.patch.dict(os.environ, {'YOLO_TTA_FAST_GEOMETRY': '1'}, clear=True):
            self.assertTrue(cb.radial_column_geometry_for_shape(3072, 3072, 0, 2911))
            self.assertFalse(cb.radial_column_geometry_for_shape(3072, 3072, 0, 64))
            self.assertFalse(cb.radial_column_geometry_for_shape(3072, 3072, 2800, 2911))
            self.assertFalse(cb.radial_column_geometry_for_shape(64, 64, 0, 64))
            with mock.patch.dict(os.environ, {'YOLO_TTA_GPU_RADIAL_COLUMN_GEOMETRY': '1'}):
                self.assertTrue(cb.radial_column_geometry_for_shape(64, 64, 0, 64))
            with mock.patch.dict(os.environ, {'YOLO_TTA_GPU_RADIAL_COLUMN_GEOMETRY': '0'}):
                self.assertFalse(cb.radial_column_geometry_for_shape(3072, 3072, 0, 2911))

    def test_opt_in_launches_bounded_fp64_columns_before_sampling(self):
        import torch
        for enabled in (False, True):
            engine = resident_engine(np.zeros((11, 13, 15), np.uint8))
            engine._stream = object()
            order = []
            scalar = mock.Mock(side_effect=lambda grid, block, args, **kw: (order.append('scalar'), args[1].zero_()))
            columns = mock.Mock(side_effect=lambda *args, **kw: order.append('columns'))
            sample = mock.Mock(side_effect=lambda grid, block, args, **kw: (order.append('sample'), args[2].zero_()))
            kernels = SimpleNamespace(cp=SimpleNamespace(asarray=lambda value: value),
                radial_native_f32=scalar, radial_columns_f64=columns, radial_native_columns_f32=sample)
            with mock.patch.object(cb, '_radial_native_kernels', return_value=kernels), \
                    mock.patch.object(cb, '_cupy_external_stream', return_value='render'), \
                    mock.patch.object(cb, 'radial_column_geometry_for_shape', return_value=enabled), \
                    mock.patch('builtins.print'):
                result = engine._render_radial_native_resident_cuda(shell_views()[0], 0)
            self.assertFalse(result.any())
            self.assertEqual(order, ['columns', 'sample'] if enabled else ['scalar'])
            if enabled:
                table = columns.call_args.args[2][0]
                self.assertEqual(tuple(table.shape), (3, 8))
                self.assertEqual(table.dtype, torch.float64)
                self.assertIs(sample.call_args.args[2][1], table)
                self.assertEqual(columns.call_args.kwargs['stream'], sample.call_args.kwargs['stream'])

    def test_workspace_oom_retains_scalar_cuda_and_disables_repeated_probes(self):
        import torch
        engine = resident_engine(np.zeros((11, 13, 15), np.uint8))
        engine._stream = object()
        scalar = mock.Mock(side_effect=lambda grid, block, args, **kw: args[1].zero_())
        kernels = SimpleNamespace(cp=SimpleNamespace(asarray=lambda value: value), radial_native_f32=scalar)
        empty = torch.empty
        attempts = []
        def allocate(shape, **kwargs):
            if kwargs['dtype'] == torch.float64:
                attempts.append(shape)
                raise torch.OutOfMemoryError('optional column table')
            return empty(shape, **kwargs)
        with mock.patch.object(cb, '_radial_native_kernels', return_value=kernels), \
                mock.patch.object(cb, '_cupy_external_stream', return_value='render'), \
                mock.patch.object(cb, 'radial_column_geometry_for_shape', return_value=True), \
                mock.patch.object(torch, 'empty', side_effect=allocate), mock.patch('builtins.print'):
            for index in (0, 1):
                engine._render_radial_native_resident_cuda(shell_views()[0], index)
        self.assertEqual(attempts, [(3, 8)])
        self.assertEqual(scalar.call_count, 2)
        self.assertTrue(engine._radial_column_geometry_disabled)


@unittest.skipUnless(os.environ.get('XTA_TEST_RADIAL_COLUMNS_CUDA') == '1', 'explicit CUDA qualification only')
class RadialColumnCudaTests(unittest.TestCase):
    def test_scalar_and_columns_are_byte_exact_across_geometry_and_streams(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        rng = np.random.default_rng(20260909)
        for native_t, logical_t in ((11, 11), (5, 11), (17, 11), (1, 11)):
            source = rng.integers(0, 256, (native_t, 13, 15), dtype=np.uint8)
            engine = resident_engine(source, 'cuda:0', logical_t=logical_t)
            engine._stream = torch.cuda.Stream()
            views = shell_views((logical_t, 13, 15), size=17)
            for base in ('transverse', 'sagittal', 'coronal'):
                view = next(view for view in views if view.radial_base_view == base)
                for direction, angle in (('vertical', 0), ('vertical', 30), ('horizontal', -30)):
                    view = replace(view, radial_tilted_source=bool(angle), tilt_direction=direction,
                                   tilt_angle_deg=angle, radial_arc_origin=-3.25)
                    for index in sorted({0, view.num_slices // 2, view.num_slices - 1}):
                        with self.subTest(native_t=native_t, base=base, direction=direction, angle=angle, index=index):
                            results = []
                            for enabled in (False, True):
                                with mock.patch.object(cb, 'radial_column_geometry_for_shape', return_value=enabled):
                                    # Deliberately allocate outside the render stream; scratch
                                    # record_stream must protect the asynchronous CuPy reads.
                                    results.append(engine._render_radial_native_resident_cuda(view, index))
                            engine._stream.synchronize()
                            self.assertTrue(torch.equal(results[0].view(torch.uint8), results[1].view(torch.uint8)))


if __name__ == '__main__':
    unittest.main()
