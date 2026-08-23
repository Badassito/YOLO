from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

# Preserve real numerical modules on CI/developer hosts.  The repository's smoke
# stubs are only needed by stripped-down environments such as the bundled test
# interpreter used here; installing them unconditionally would make later tests
# mistake a capable host for one without OpenCV/SciPy.
def _module_is_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


if not all(_module_is_available(name) for name in ('cv2', 'scipy', 'tifffile', 'tqdm')):
    install_stubs()

from volume_tta import cuda_interpolation, interpolation, topology
from volume_tta.cuda_interpolation import (
    CudaInterpolationRenderer,
    _slice_job_placement,
)


class _NumpyNdimage:
    """Small CPU implementation of the CuPyX primitives used by unit fixtures."""

    @staticmethod
    def label(value: object, structure: object = None):
        mask = np.asarray(value, dtype=bool)
        labels = np.zeros(mask.shape, dtype=np.int32)
        if structure is None:
            offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
        else:
            footprint = np.asarray(structure, dtype=bool)
            cy, cx = footprint.shape[0] // 2, footprint.shape[1] // 2
            offsets = tuple(
                (int(y - cy), int(x - cx))
                for y, x in np.argwhere(footprint)
                if int(y) != int(cy) or int(x) != int(cx)
            )
        component = 0
        height, width = mask.shape
        for seed_y, seed_x in np.argwhere(mask):
            sy, sx = int(seed_y), int(seed_x)
            if int(labels[sy, sx]) != 0:
                continue
            component += 1
            labels[sy, sx] = int(component)
            stack = [(sy, sx)]
            while stack:
                y, x = stack.pop()
                for dy, dx in offsets:
                    yy, xx = int(y + dy), int(x + dx)
                    if (
                        0 <= yy < height and 0 <= xx < width
                        and bool(mask[yy, xx]) and int(labels[yy, xx]) == 0
                    ):
                        labels[yy, xx] = int(component)
                        stack.append((yy, xx))
        return labels, int(component)

    @staticmethod
    def binary_fill_holes(value: object) -> np.ndarray:
        mask = np.asarray(value, dtype=bool)
        height, width = mask.shape
        outside = np.zeros(mask.shape, dtype=bool)
        stack: list[tuple[int, int]] = []
        for x in range(width):
            stack.extend(((0, x), (height - 1, x)))
        for y in range(height):
            stack.extend(((y, 0), (y, width - 1)))
        while stack:
            y, x = stack.pop()
            if (
                y < 0 or y >= height or x < 0 or x >= width
                or bool(mask[y, x]) or bool(outside[y, x])
            ):
                continue
            outside[y, x] = True
            stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
        return ~outside

    @staticmethod
    def distance_transform_edt(
        value: object,
        return_indices: bool = False,
        float64_distances: bool = True,
    ) -> np.ndarray:
        if return_indices:
            raise AssertionError('fixture does not request EDT indices')
        mask = np.asarray(value, dtype=bool)
        background = np.argwhere(~mask)
        dtype = np.float64 if bool(float64_distances) else np.float32
        out = np.zeros(mask.shape, dtype=dtype)
        if int(background.size) <= 0:
            out[mask] = np.inf
            return out
        for y, x in np.argwhere(mask):
            delta = background.astype(np.float64) - np.asarray((y, x), dtype=np.float64)
            out[int(y), int(x)] = math.sqrt(float(np.min(np.sum(delta * delta, axis=1))))
        return out


class _NumpyDeviceRuntime:
    xp = np
    ndi = _NumpyNdimage()
    device_index = 0

    def __init__(self, *, fail_copy_to_host: bool = False) -> None:
        self.fail_copy_to_host = bool(fail_copy_to_host)
        self.freed = False

    @staticmethod
    def activate() -> object:
        return contextlib.nullcontext()

    @staticmethod
    def to_device(value: np.ndarray) -> np.ndarray:
        return np.array(value, copy=True)

    def to_host(self, value: object) -> np.ndarray:
        if self.fail_copy_to_host:
            raise RuntimeError('injected D2H failure')
        return np.array(value, copy=True)

    @staticmethod
    def mem_info() -> tuple[int, int]:
        return 8 * 1024 ** 3, 16 * 1024 ** 3

    @staticmethod
    def synchronize() -> None:
        return None

    def free_cached_memory(self) -> None:
        self.freed = True


def _plan(
    sdf: np.ndarray,
    *,
    source_z: int = 0,
    sign: int = 1,
    num_slices: int = 5,
    source_anchor: tuple[int, int] = (3, 3),
    target_anchor: tuple[int, int] = (3, 3),
    steps: int = 2,
    cached_sections: list[object] | None = None,
) -> interpolation.SliceBridgeRenderPlan:
    return interpolation.SliceBridgeRenderPlan(
        source_label=1,
        target_label=2,
        source_point=(int(source_z), int(source_anchor[0]), int(source_anchor[1])),
        target_point=(
            int((source_z + sign * steps) % num_slices),
            int(target_anchor[0]),
            int(target_anchor[1]),
        ),
        source_anchor=source_anchor,
        target_anchor=target_anchor,
        steps=int(steps),
        sign=int(sign),
        num_slices=int(num_slices),
        sdf0=np.ascontiguousarray(sdf, dtype=np.float32),
        sdf1=np.ascontiguousarray(sdf, dtype=np.float32),
        cached_sections=(list(cached_sections) if cached_sections is not None else []),
    )


class CudaInterpolationRendererContractTests(unittest.TestCase):
    def test_factory_reclaims_torch_cache_before_cupy_admission(self) -> None:
        events: list[str] = []

        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def is_initialized() -> bool:
                return True

            @staticmethod
            def current_device() -> int:
                return 2

            @staticmethod
            def synchronize(device: int) -> None:
                events.append(f'synchronize:{int(device)}')

            @staticmethod
            def empty_cache() -> None:
                events.append('empty_cache')

        class _FakeRuntime:
            @staticmethod
            def mem_info() -> tuple[int, int]:
                events.append('mem_info')
                return 4 * 1024 ** 3, 8 * 1024 ** 3

        class _FakeRenderer:
            reserve_bytes = 1024 ** 3
            runtime = _FakeRuntime()

            def __init__(self, device_index: int) -> None:
                events.append(f'construct:{int(device_index)}')

            @staticmethod
            def preflight() -> None:
                events.append('preflight')

        fake_torch = mock.Mock(cuda=_FakeCuda())
        with (
            mock.patch.dict(
                os.environ,
                {
                    'YOLO_TTA_GPU_INTERPOLATION': '1',
                    'YOLO_TTA_GPU_INTERPOLATION_CREATE_CONTEXT': '0',
                },
                clear=False,
            ),
            mock.patch.dict(sys.modules, {'torch': fake_torch}),
            mock.patch.object(
                cuda_interpolation, 'CudaInterpolationRenderer', _FakeRenderer,
            ),
        ):
            renderer, status = cuda_interpolation.create_cuda_interpolation_renderer(
                process_worker=True,
            )

        self.assertIsInstance(renderer, _FakeRenderer)
        self.assertEqual(status, 'cuda:2 CuPy/CuPyX')
        self.assertEqual(
            events,
            ['synchronize:2', 'empty_cache', 'construct:2', 'mem_info', 'preflight'],
        )

    def test_uncached_radius_scan_and_render_match_expected_morphology(self) -> None:
        sdf = np.full((5, 5), -1.0, dtype=np.float32)
        sdf[1:4, 1:4] = np.float32(1.0)
        sdf[2, 2] = np.float32(-1.0)  # filled as an enclosed 4-connected hole
        plan = _plan(sdf)
        runtime = _NumpyDeviceRuntime()
        renderer = CudaInterpolationRenderer(
            runtime=runtime, cache_bytes=4 * 1024 ** 2, reserve_bytes=0,
        )

        radius = renderer.estimate_min_radius(
            plan, reject_at_or_below=0.5, cache_sections=False,
        )
        self.assertEqual(radius, 1.0)
        destination = np.zeros((7, 7), dtype=np.uint8)
        result = renderer.render_slice(destination, [(plan, 1, 1)])

        expected = np.zeros_like(destination)
        expected[2:5, 2:5] = np.uint8(1)
        np.testing.assert_array_equal(destination, expected)
        self.assertEqual(result.added_voxels, 9)
        self.assertEqual(result.rendered_sections, 1)
        self.assertEqual(result.bbox, (1, 1, 6, 6))
        telemetry = renderer.telemetry()
        self.assertEqual(int(telemetry['estimated_plans']), 1)
        self.assertEqual(int(telemetry['estimated_sections']), 1)
        self.assertEqual(int(telemetry['rendered_slices']), 1)
        self.assertGreater(int(telemetry['host_to_device_bytes']), 0)
        self.assertGreater(int(telemetry['device_to_host_bytes']), 0)
        renderer.close()
        self.assertTrue(runtime.freed)

    def test_center_component_tie_uses_row_major_nearest_pixel(self) -> None:
        sdf = np.full((5, 5), -1.0, dtype=np.float32)
        sdf[0, 2] = np.float32(1.0)
        sdf[2, 0] = np.float32(1.0)
        plan = _plan(sdf)
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        destination = np.zeros((7, 7), dtype=np.uint8)
        result = renderer.render_slice(destination, [(plan, 1, 1)])

        expected = np.zeros_like(destination)
        expected[1, 3] = np.uint8(1)
        np.testing.assert_array_equal(destination, expected)
        self.assertEqual(result.added_voxels, 1)

    def test_packed_bits_and_edge_clipping_are_preserved(self) -> None:
        section = np.ones((3, 3), dtype=bool)
        sdf = np.ones((3, 3), dtype=np.float32)
        plan = _plan(
            sdf,
            source_anchor=(0, 0),
            target_anchor=(0, 0),
            cached_sections=[None, section, None],
        )
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        destination = np.zeros((4, 4), dtype=np.uint16)
        result = renderer.render_slice(destination, [(plan, 1, 1), (plan, 1, 2)])

        expected = np.zeros_like(destination)
        expected[0:2, 0:2] = np.uint16(3)
        np.testing.assert_array_equal(destination, expected)
        self.assertEqual(result.added_voxels, 8)
        self.assertEqual(result.bbox, (0, 0, 2, 2))

    def test_wrap_crossing_mirrors_the_destination_u_coordinate(self) -> None:
        section = np.zeros((3, 3), dtype=bool)
        section[1, 0] = True
        plan = _plan(
            np.ones((3, 3), dtype=np.float32),
            source_z=0,
            sign=-1,
            source_anchor=(2, 1),
            target_anchor=(2, 1),
            cached_sections=[None, section, None],
        )
        placement = _slice_job_placement(plan, 1, (5, 7))
        self.assertIsNotNone(placement)
        self.assertTrue(placement.mirrored)  # type: ignore[union-attr]
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        destination = np.zeros((5, 7), dtype=np.uint8)
        renderer.render_slice(destination, [(plan, 1, 1)])

        expected = np.zeros_like(destination)
        # Local x=0 becomes x=2 when the section flips; center u=1 mirrors to u=5.
        expected[2, 6] = np.uint8(1)
        np.testing.assert_array_equal(destination, expected)

    def test_d2h_failure_leaves_host_destination_unchanged(self) -> None:
        section = np.ones((3, 3), dtype=bool)
        plan = _plan(
            np.ones((3, 3), dtype=np.float32),
            cached_sections=[None, section, None],
        )
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(fail_copy_to_host=True),
            cache_bytes=0,
            reserve_bytes=0,
        )
        destination = np.zeros((7, 7), dtype=np.uint8)
        before = destination.copy()
        with self.assertRaisesRegex(RuntimeError, 'injected D2H failure'):
            renderer.render_slice(destination, [(plan, 1, 1)])
        np.testing.assert_array_equal(destination, before)

    def test_cache_validates_host_owner_identity_for_id_derived_keys(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=1024, reserve_bytes=0,
        )
        first_owners = (np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))
        second_owners = (np.full(1, 2, dtype=np.float32), np.full(1, 3, dtype=np.float32))
        key = ('sdf', 101, 202)
        renderer._cache_put(key, 'stale-device-value', 8, first_owners)
        self.assertIsNone(renderer._cache_get(key, second_owners))
        self.assertEqual(int(renderer.telemetry()['cache_live_bytes']), 0)

    def test_section_scan_obeys_cache_budget_before_plan_completion(self) -> None:
        plan = _plan(np.ones((5, 5), dtype=np.float32), steps=8)
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=60, reserve_bytes=0,
        )
        renderer.estimate_min_radius(plan, cache_sections=True)
        telemetry = renderer.telemetry()
        self.assertLessEqual(int(telemetry['cache_peak_bytes']), 60)
        self.assertLessEqual(int(telemetry['cache_live_bytes']), 60)
        self.assertEqual(sum(section is not None for section in plan.cached_sections), 7)

    def test_device_only_section_cache_avoids_d2h_and_recomputes_after_eviction(self) -> None:
        plan = _plan(np.ones((5, 5), dtype=np.float32), steps=8)
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=60, reserve_bytes=0,
        )
        renderer.estimate_min_radius(
            plan,
            cache_sections=True,
            cache_host_sections=False,
        )
        after_scan = renderer.telemetry()
        self.assertEqual(int(after_scan['device_to_host_bytes']), 0)
        self.assertTrue(all(section is None for section in plan.cached_sections))
        self.assertLessEqual(int(after_scan['cache_peak_bytes']), 60)

        # Step 1 was evicted by the later sections. Rendering must reconstruct it
        # from the SDFs without relying on a host section cache.
        destination = np.zeros((9, 9), dtype=np.uint8)
        result = renderer.render_slice(destination, [(plan, 1, 1)])
        self.assertEqual(result.added_voxels, 25)
        self.assertEqual(int(np.count_nonzero(destination)), 25)
        after_render = renderer.telemetry()
        self.assertGreater(
            int(after_render['cache_misses']), int(after_scan['cache_misses']),
        )
        self.assertGreater(int(after_render['device_to_host_bytes']), 0)

    @unittest.skipUnless(
        os.environ.get('VOLUME_TTA_TEST_CUDA', '').strip() == '1',
        'set VOLUME_TTA_TEST_CUDA=1 on an NVIDIA CUDA host',
    )
    def test_real_cupy_backend_matches_the_numpy_device_oracle(self) -> None:
        sdf = np.full((7, 7), -2.0, dtype=np.float32)
        sdf[1:6, 1:6] = np.float32(2.0)
        sdf[3, 3] = np.float32(-2.0)
        gpu_plan = _plan(sdf, steps=3)
        oracle_plan = _plan(sdf, steps=3)
        gpu = CudaInterpolationRenderer(cache_bytes=32 * 1024 ** 2, reserve_bytes=0)
        oracle = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=32 * 1024 ** 2, reserve_bytes=0,
        )
        try:
            gpu.preflight()
            gpu_radius = gpu.estimate_min_radius(
                gpu_plan, reject_at_or_below=0.5, cache_sections=True,
            )
            oracle_radius = oracle.estimate_min_radius(
                oracle_plan, reject_at_or_below=0.5, cache_sections=True,
            )
            self.assertAlmostEqual(gpu_radius, oracle_radius, places=5)
            gpu_destination = np.zeros((11, 11), dtype=np.uint8)
            oracle_destination = np.zeros_like(gpu_destination)
            gpu.render_slice(gpu_destination, [(gpu_plan, 1, 1), (gpu_plan, 2, 1)])
            oracle.render_slice(
                oracle_destination, [(oracle_plan, 1, 1), (oracle_plan, 2, 1)],
            )
            np.testing.assert_array_equal(gpu_destination, oracle_destination)
        finally:
            gpu.close()
            oracle.close()


class CudaInterpolationPassRoutingTests(unittest.TestCase):
    @staticmethod
    def _pass_fixture(*, steps: int = 2):
        num_slices = int(steps) + 1
        mask = np.zeros((num_slices, 7, 7), dtype=np.uint8)
        mask[0, 3, 3] = np.uint8(1)
        mask[int(steps), 3, 3] = np.uint8(1)
        labels = np.zeros_like(mask, dtype=np.uint16)
        labels[0, 3, 3] = np.uint16(1)
        labels[int(steps), 3, 3] = np.uint16(2)
        seed = interpolation.SliceEndpointSeed(1, (0, 3, 3), 1)
        section = np.ones((3, 3), dtype=bool)
        plan = _plan(
            np.ones((3, 3), dtype=np.float32),
            source_anchor=(3, 3),
            target_anchor=(3, 3),
            steps=int(steps),
            num_slices=int(num_slices),
            cached_sections=[None] + [section] * (int(steps) - 1) + [None],
        )
        plan_result = interpolation.SliceSeedBridgePlanResult(
            candidate_connections=1,
            accepted_connections=1,
            default_bridges=1,
            plans=[plan],
        )
        return mask, labels, seed, plan_result

    def _run_pass(
        self,
        renderer: object,
        *,
        required: bool = False,
        steps: int = 2,
    ):
        mask, labels, seed, plan_result = self._pass_fixture(steps=int(steps))
        temp_context = tempfile.TemporaryDirectory()
        self.addCleanup(temp_context.cleanup)
        with (
            mock.patch.object(
                topology, 'interpolation_skip_compact_relabel_enabled', return_value=False,
            ),
            mock.patch.object(
                topology, 'label_foreground_volume_streaming',
                return_value=(labels, 2, []),
            ),
            mock.patch.object(
                interpolation, '_build_slice_endpoint_seeds', return_value=([seed], 1),
            ),
            mock.patch.object(
                interpolation, '_plan_slice_seed_bridges', return_value=plan_result,
            ),
            mock.patch.object(
                interpolation, 'should_use_in_memory_workspace', return_value=True,
            ),
            mock.patch.object(
                interpolation, 'create_cuda_interpolation_renderer',
                return_value=(renderer, 'test cuda:0'),
            ),
            mock.patch.dict(
                os.environ,
                {'YOLO_TTA_GPU_INTERPOLATION_REQUIRED': '1' if required else '0'},
            ),
        ):
            stats = interpolation.interpolate_view_volume_pass_inplace(
                mask_mm=mask,
                work_dir=Path(temp_context.name) / 'work',
                pass_tag='gpu-routing',
                max_slice_distance=2,
                search_angle_deg=15.0,
                interpolation_walk_back=0,
                interpolation_candidates=1,
                interpolate_min_radius=0.0,
                workers=1,
            )
        return mask, stats

    def test_required_mode_rejects_cpu_admission_fallback(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'CUDA interpolation is unavailable'):
            self._run_pass(None, required=True)

    def test_pass_reports_real_cuda_computation_backend(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        mask, stats = self._run_pass(renderer)
        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertEqual(stats['interpolation_render_backend'], 'cuda_cupy_crop_bounded')
        self.assertTrue(stats['gpu_interpolation_active'])
        self.assertEqual(int(stats['gpu_interpolation_batches']), 1)
        self.assertEqual(int(stats['gpu_interpolation_rendered_sections']), 1)

    def test_device_failure_replays_the_batch_on_cpu(self) -> None:
        class _FailingRenderer:
            def __init__(self) -> None:
                self.available = True
                self.reason = None

            def render_slice(self, *_args: object, **_kwargs: object) -> object:
                raise RuntimeError('injected device render failure')

            def disable(self, exc: BaseException) -> bool:
                self.available = False
                self.reason = f'{type(exc).__name__}: {exc}'
                return True

            @staticmethod
            def release_plans(_plans: object) -> None:
                return None

            def telemetry(self) -> dict[str, object]:
                return {
                    'estimated_plans': 0,
                    'rendered_slices': 0,
                    'failed_reason': self.reason,
                }

            @staticmethod
            def close() -> None:
                return None

        mask, stats = self._run_pass(_FailingRenderer())
        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertEqual(stats['interpolation_render_backend'], 'cpu_numpy_numba')
        self.assertFalse(stats['gpu_interpolation_active'])
        self.assertEqual(int(stats['gpu_interpolation_fallback_batches']), 1)
        self.assertIn('injected device render failure', str(stats['gpu_interpolation_fallback_reason']))

    def test_failure_after_one_gpu_commit_replays_the_whole_batch_safely(self) -> None:
        inner = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )

        class _FailAfterFirstCommit:
            def __init__(self) -> None:
                self.available = True
                self.calls = 0

            def render_slice(self, *args: object, **kwargs: object):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('failure after first committed slice')
                return inner.render_slice(*args, **kwargs)

            def disable(self, exc: BaseException) -> bool:
                self.available = False
                return inner.disable(exc)

            @staticmethod
            def release_plans(plans: object) -> None:
                inner.release_plans(plans)  # type: ignore[arg-type]

            @staticmethod
            def telemetry() -> dict[str, object]:
                return inner.telemetry()

            @staticmethod
            def close() -> None:
                inner.close()

        mask, stats = self._run_pass(_FailAfterFirstCommit(), steps=3)
        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertEqual(int(np.count_nonzero(mask[2])), 9)
        self.assertEqual(
            stats['interpolation_render_backend'],
            'cuda_cupy_crop_bounded+cpu_fallback',
        )
        self.assertTrue(stats['gpu_interpolation_active'])
        self.assertEqual(int(stats['gpu_interpolation_rendered_slices']), 1)
        self.assertEqual(int(stats['gpu_interpolation_fallback_batches']), 1)


if __name__ == '__main__':
    unittest.main()
