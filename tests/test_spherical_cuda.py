"""CPU-only numerical qualification of the resident QSC native-frame contract."""
from __future__ import annotations

from collections import OrderedDict
import importlib.util
from itertools import product
import math
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import spherical_cuda
from XTA.qsc import qsc_inverse


def spherical_view(shape=(9, 11, 13), *, face=0, radii=(0., .63, 2.5, 7.),
                   size=7, intervals=6, origin=(0, 0), rotation=None):
    return SimpleNamespace(
        family='spherical', full_t=shape[0], full_h=shape[1], full_w=shape[2],
        src_h=size, src_w=size, spherical_face=face, spherical_radii=radii,
        spherical_face_intervals=intervals, spherical_u_origin=origin[0],
        spherical_v_origin=origin[1],
        spherical_rotation_xyz=tuple(np.eye(3).reshape(-1) if rotation is None else np.asarray(rotation).reshape(-1)),
    )


def resident_engine(volume, logical_t=None):
    import torch
    from XTA.cuda_backend import _GpuWorkerRenderEngine
    engine = object.__new__(_GpuWorkerRenderEngine)
    engine.torch, engine.device = torch, torch.device('cpu')
    engine._volume_gpu = torch.from_numpy(np.ascontiguousarray(volume))
    engine._volume_flat = engine._volume_gpu.reshape(-1)
    engine._logical_t = int(logical_t or volume.shape[0])
    engine._native_t_map_cache = {}
    engine._native_plane_cache = OrderedDict()
    engine._native_u8_plane_cache = OrderedDict()
    return engine


def logical_cube(volume, depth):
    """Materialize the uint8 logical volume independently of the GPU helpers."""
    out = np.empty((depth, *volume.shape[1:]), np.uint8)
    for t in range(depth):
        coordinate = (t + .5) * (len(volume) / depth) - .5
        low = min(len(volume) - 1, max(0, math.floor(coordinate)))
        high = min(len(volume) - 1, low + 1)
        alpha = np.float32(np.clip(coordinate - low, 0., 1.))
        first, second = volume[low].astype(np.float32), volume[high].astype(np.float32)
        out[t] = np.rint(first + alpha * (second - first)).clip(0, 255).astype(np.uint8)
    return out


def scalar_oracle(volume, view, index):
    """Slow point/tap oracle; does not use renderer coordinate or sampling helpers."""
    radius = view.spherical_radii[index]
    rotation = np.asarray(view.spherical_rotation_xyz).reshape(3, 3)
    result = np.zeros((view.src_h, view.src_w), np.uint8)
    shape = np.asarray(volume.shape)
    for row, column in product(range(view.src_h), range(view.src_w)):
        u_index, v_index = view.spherical_u_origin + column, view.spherical_v_origin + row
        if not (0 <= u_index <= view.spherical_face_intervals and 0 <= v_index <= view.spherical_face_intervals):
            continue
        direction = qsc_inverse(view.spherical_face,
                                -1. + 2. * u_index / view.spherical_face_intervals,
                                1. - 2. * v_index / view.spherical_face_intervals) @ rotation.T
        coordinate = (shape - 1) * .5 + radius * direction[::-1]
        lower = np.floor(coordinate).astype(np.int64)
        fraction = (coordinate - lower).astype(np.float32)
        value = np.float32(0)
        for offset in product((0, 1), repeat=3):
            tap = lower + offset
            if np.any(tap < 0) or np.any(tap >= shape):
                continue
            weights = np.where(offset, fraction, np.float32(1.) - fraction).astype(np.float32)
            weight = np.float32(np.float32(weights[0] * weights[1]) * weights[2])
            value = np.float32(value + np.float32(np.float32(volume[tuple(tap)]) * weight))
        result[row, column] = np.uint8(np.rint(value).clip(0, 255))
    return result


def rotation_xyz():
    a, b = np.deg2rad((27., -43.))
    rz = np.asarray(((np.cos(a), -np.sin(a), 0), (np.sin(a), np.cos(a), 0), (0, 0, 1)))
    ry = np.asarray(((np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))))
    return rz @ ry


@unittest.skipUnless(os.environ.get('XTA_TEST_SPHERICAL_CUDA') == '1', 'explicit available-GPU qualification only')
class SphericalHardwareRendererTests(unittest.TestCase):
    def test_native_cuda_matches_materialized_reference_with_deferred_t(self):
        import torch
        from XTA.cuda_backend import _GpuWorkerRenderEngine
        from XTA.spherical_geometry import render_shell_frame
        volume=np.random.default_rng(209).integers(0,256,(11,13,15),dtype=np.uint8)
        engine=_GpuWorkerRenderEngine('cuda:0')
        self.assertEqual(engine.ensure_volume_array(volume), 'resident')
        try:
            for logical_t in (11,17):
                engine._logical_t=logical_t
                engine._native_t_map_cache.clear()
                materialized=logical_cube(volume,logical_t)
                for face,rotation in product(range(6),(np.eye(3),rotation_xyz())):
                    view=spherical_view((logical_t,13,15),face=face,size=25,intervals=20,
                        origin=(-2,-2),rotation=rotation,radii=(.63,3.25,5.))
                    for index in range(3):
                        with torch.cuda.stream(engine._stream):
                            actual=spherical_cuda.render_spherical_native_resident(engine,view,index)
                        engine._stream.synchronize()
                        np.testing.assert_array_equal(actual.cpu().numpy(),render_shell_frame(materialized,view,index))
            self.assertTrue(engine._spherical_native_kernel_announced)
            self.assertFalse(getattr(engine,'_spherical_native_kernel_disabled',False))
        finally:
            engine._stream.synchronize()
            spherical_cuda.clear_spherical_render_cache(engine)
            engine._volume_gpu=engine._volume_flat=None
            engine._fused_volume_ref=None


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Torch is optional')
class SphericalResidentNumericalTests(unittest.TestCase):
    def test_native_cpu_renderer_matches_across_different_row_block_sizes(self):
        from XTA.spherical_geometry import render_shell_frame
        volume = np.random.default_rng(84).integers(0, 256, (35, 37, 39), dtype=np.uint8)
        engine = resident_engine(volume)
        for face, rotation in product(range(6), (np.eye(3), rotation_xyz())):
            view = spherical_view(volume.shape, face=face, rotation=rotation, size=70,
                                  intervals=64, origin=(-2, -2), radii=(.63, 8.1, 17.))
            for index in (0, 1, 2):
                with self.subTest(face=face, radius=index):
                    actual = spherical_cuda.render_spherical_native_resident(engine, view, index).numpy()
                    np.testing.assert_array_equal(actual, render_shell_frame(volume, view, index))

    def test_all_faces_radii_and_rigid_rotations_match_scalar_gray8_oracle(self):
        volume = np.random.default_rng(29).integers(0, 256, (9, 11, 13), dtype=np.uint8)
        engine = resident_engine(volume)
        for face, rotation in product(range(6), (np.eye(3), rotation_xyz())):
            view = spherical_view(volume.shape, face=face, rotation=rotation)
            for index in range(len(view.spherical_radii)):
                with self.subTest(face=face, rotation=tuple(rotation.reshape(-1)), radius=index):
                    actual = spherical_cuda.render_spherical_native_resident(engine, view, index)
                    self.assertEqual(actual.dtype, engine.torch.float32)
                    np.testing.assert_array_equal(actual.numpy(), scalar_oracle(volume, view, index))

    def test_deferred_native_t_matches_materialized_gray8_cube(self):
        # Odd endpoints create half-value lerps, so omitting the intermediate
        # gray8 conversion measurably changes spherical samples.
        volume = np.random.default_rng(105).integers(0, 256, (3, 7, 9), dtype=np.uint8)
        cube = logical_cube(volume, 8)
        deferred, eager = resident_engine(volume, 8), resident_engine(cube)
        for face in range(6):
            view = spherical_view(cube.shape, face=face, rotation=rotation_xyz(), radii=(.63, 2.4, 5.))
            for index in range(3):
                with self.subTest(face=face, radius=index):
                    first = spherical_cuda.render_spherical_native_resident(deferred, view, index).numpy()
                    second = spherical_cuda.render_spherical_native_resident(eager, view, index).numpy()
                    np.testing.assert_array_equal(first, second)
                    np.testing.assert_array_equal(first, scalar_oracle(cube, view, index))

    def test_padding_stays_zero_and_endpoint_rows_remain_valid(self):
        volume = np.full((13, 13, 13), 211, np.uint8)
        engine = resident_engine(volume)
        view = spherical_view(volume.shape, size=9, origin=(-1, -1), radii=(.63,), rotation=rotation_xyz())
        actual = spherical_cuda.render_spherical_native_resident(engine, view, 0).numpy()
        expected = np.zeros((9, 9), np.float32)
        expected[1:8, 1:8] = 211
        np.testing.assert_array_equal(actual, expected)

    def test_one_voxel_axes_keep_fractional_zero_border_contributions(self):
        for shape in ((1, 5, 7), (5, 1, 7), (5, 7, 1), (1, 1, 1)):
            volume = np.full(shape, 255, np.uint8)
            engine = resident_engine(volume)
            for face in range(6):
                view = spherical_view(shape, face=face, radii=(.63, 4.), rotation=rotation_xyz())
                for index in range(2):
                    actual = spherical_cuda.render_spherical_native_resident(engine, view, index).numpy()
                    with self.subTest(shape=shape, face=face, radius=index):
                        np.testing.assert_array_equal(actual, scalar_oracle(volume, view, index))
                        if index == 0:
                            self.assertGreater(np.count_nonzero(actual), 0)
                            self.assertTrue(np.any((actual > 0) & (actual < 255)))

    def test_all_six_axis_centers_and_rotated_axis_have_correct_array_order(self):
        t, y, x = np.indices((7, 7, 7))
        volume = (t * 20 + y * 5 + x).astype(np.uint8)
        engine = resident_engine(volume)
        offsets = ((0, 0, 1), (0, 1, 0), (0, 0, -1), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
        for face, offset in enumerate(offsets):
            view = spherical_view(volume.shape, face=face, radii=(1.,))
            actual = spherical_cuda.render_spherical_native_resident(engine, view, 0).numpy()
            self.assertEqual(actual[3, 3], volume[tuple(3 + np.asarray(offset))])
        # Rotate the positive X cube face onto world positive Z.
        view = spherical_view(volume.shape, face=0, radii=(1.,), rotation=((0, 0, -1), (0, 1, 0), (1, 0, 0)))
        self.assertEqual(spherical_cuda.render_spherical_native_resident(engine, view, 0)[3, 3].item(), volume[4, 3, 3])

    def test_direction_cache_reuses_radii_evicts_and_clears(self):
        volume = np.zeros((9, 11, 13), np.uint8)
        engine = resident_engine(volume)
        view = spherical_view(volume.shape)
        with mock.patch.object(spherical_cuda, '_build_direction_block', wraps=spherical_cuda._build_direction_block) as build:
            spherical_cuda.render_spherical_native_resident(engine, view, 0)
            spherical_cuda.render_spherical_native_resident(engine, view, 1)
            self.assertEqual(build.call_count, 1)
        self.assertEqual(len(engine._spherical_direction_cache), 1)
        size = view.src_h * view.src_w * 25
        with mock.patch.object(spherical_cuda, '_DIRECTION_CACHE_BYTES', size):
            other = spherical_view(volume.shape, face=1)
            spherical_cuda.render_spherical_native_resident(engine, other, 0)
        self.assertEqual(len(engine._spherical_direction_cache), 1)
        self.assertEqual(engine._spherical_direction_cache_bytes, size)
        spherical_cuda.clear_spherical_render_cache(engine)
        self.assertFalse(engine._spherical_direction_cache)
        self.assertEqual(engine._spherical_direction_cache_bytes, 0)

    def test_oversized_direction_plans_stream_bounded_strips(self):
        volume = np.random.default_rng(2).integers(0, 256, (9, 11, 13), dtype=np.uint8)
        engine = resident_engine(volume)
        view = spherical_view(volume.shape, radii=(2.4,))
        with (mock.patch.object(spherical_cuda, '_DIRECTION_CACHE_BYTES', 7 * 25 * 2),
              mock.patch.object(spherical_cuda, '_build_direction_block', wraps=spherical_cuda._build_direction_block) as build):
            actual = spherical_cuda.render_spherical_native_resident(engine, view, 0).numpy()
        self.assertEqual(build.call_count, 4)
        self.assertTrue(all(call.args[3] - call.args[2] <= 2 for call in build.call_args_list))
        self.assertFalse(engine._spherical_direction_cache)
        np.testing.assert_array_equal(actual, scalar_oracle(volume, view, 0))

    def test_cuda_launch_consumes_cached_directions_and_scalar_radius(self):
        engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        engine._stream = object()
        engine._fused_cupy_volume = mock.Mock(return_value=object())
        view = spherical_view()
        calls = []
        def kernel(grid, block, args, *, stream):
            calls.append((grid, block, args, stream))
            args[3].zero_()
        kernels = SimpleNamespace(cp=SimpleNamespace(asarray=lambda value: value), kernel=kernel)
        with (mock.patch.object(spherical_cuda, '_spherical_kernels', return_value=kernels),
              mock.patch('XTA.geometry._cupy_external_stream', return_value='render-stream')):
            for index in (0, 1):
                spherical_cuda._render_spherical_native_cuda(engine, view, index)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][:2], ((1,), (256,)))
        self.assertIs(calls[0][2][1], calls[1][2][1])
        self.assertIs(calls[0][2][2], calls[1][2][2])
        self.assertEqual(calls[0][2][-1], np.float64(0))
        self.assertEqual(calls[1][2][-1], np.float64(.63))
        self.assertEqual(calls[0][3], 'render-stream')

    def test_contract_rejects_bad_geometry_before_sampling(self):
        engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
        for field, value in (('family', 'radial'), ('spherical_face', 6), ('spherical_face_intervals', 0),
                             ('spherical_radii', (-1.,)), ('full_t', 10),
                             ('spherical_rotation_xyz', tuple((np.eye(3) * 2).reshape(-1)))):
            view = spherical_view()
            setattr(view, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                spherical_cuda.render_spherical_native_resident(engine, view, 0)
        with self.assertRaises(ValueError):
            spherical_cuda.render_spherical_native_resident(engine, spherical_view(), -1)

    def test_optional_kernel_failure_uses_reference_without_repeated_attempts(self):
        engine = SimpleNamespace(_volume_gpu=SimpleNamespace(is_cuda=True))
        result = object()
        with (mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL': '1'}),
              mock.patch.object(spherical_cuda, '_render_spherical_native_cuda', side_effect=RuntimeError('unavailable')) as native,
              mock.patch.object(spherical_cuda, '_render_spherical_native_torch', return_value=result) as reference,
              mock.patch('builtins.print') as announce):
            self.assertIs(spherical_cuda.render_spherical_native_resident(engine, spherical_view(), 0), result)
            self.assertIs(spherical_cuda.render_spherical_native_resident(engine, spherical_view(), 1), result)
        native.assert_called_once()
        self.assertEqual(reference.call_count, 2)
        announce.assert_called_once()
        self.assertIn('resident Torch reference', announce.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
