from __future__ import annotations

import contextlib
import gc
import importlib.util
import math
import os
import sys
import tempfile
import threading
import unittest
import weakref
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

from XTA import cuda_interpolation, interpolation, runtime as runtime_helpers, topology
from XTA.cuda_interpolation import (
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
        self._next_stream = 0

    @staticmethod
    def activate() -> object:
        return contextlib.nullcontext()

    def create_stream(self) -> object:
        self._next_stream += 1
        return int(self._next_stream)

    @staticmethod
    def activate_stream(_stream: object) -> object:
        return contextlib.nullcontext()

    @staticmethod
    def to_device(value: np.ndarray) -> np.ndarray:
        return np.array(value, copy=True)

    def to_host(self, value: object) -> np.ndarray:
        if self.fail_copy_to_host:
            raise RuntimeError('injected D2H failure')
        return np.array(value, copy=True)

    def to_host_async(self, value: object, _stream: object) -> np.ndarray:
        return self.to_host(value)

    @staticmethod
    def record_completion(_stream: object) -> object:
        return contextlib.nullcontext()

    @staticmethod
    def make_stream_wait(_stream: object, _event: object) -> None:
        return None

    @staticmethod
    def wait_completion(_event: object) -> None:
        return None

    @staticmethod
    def synchronize_stream(_stream: object) -> None:
        return None

    @staticmethod
    def mem_info() -> tuple[int, int]:
        return 8 * 1024 ** 3, 16 * 1024 ** 3

    @staticmethod
    def synchronize() -> None:
        return None

    def free_cached_memory(self) -> None:
        self.freed = True


class _DeferredCompletion:
    def __init__(self, *, fail: bool = False) -> None:
        self.release = threading.Event()
        self.wait_started = threading.Event()
        self.fail = bool(fail)

    def synchronize(self) -> None:
        self.wait_started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError('timed out waiting for injected stream completion')
        if self.fail:
            raise RuntimeError('injected asynchronous D2H failure')


class _DeferredNumpyRuntime(_NumpyDeviceRuntime):
    def __init__(self, *, fail_completion: bool = False) -> None:
        super().__init__()
        self.fail_completion = bool(fail_completion)
        self.created_streams: list[object] = []
        self.completions: list[_DeferredCompletion] = []
        self._completion_condition = threading.Condition()

    def create_stream(self) -> object:
        stream = super().create_stream()
        with self._completion_condition:
            self.created_streams.append(stream)
        return stream

    def record_completion(self, _stream: object) -> object:
        completion = _DeferredCompletion(fail=self.fail_completion)
        with self._completion_condition:
            self.completions.append(completion)
            self._completion_condition.notify_all()
        return completion

    @staticmethod
    def wait_completion(event: object) -> None:
        event.synchronize()  # type: ignore[attr-defined]

    def wait_for_completions(self, count: int) -> list[_DeferredCompletion]:
        with self._completion_condition:
            ready = self._completion_condition.wait_for(
                lambda: len(self.completions) >= int(count), timeout=5.0,
            )
            if not ready:
                raise AssertionError(
                    f'expected {int(count)} queued completion(s), got {len(self.completions)}'
                )
            return list(self.completions)


class _EdtFailingNdimage(_NumpyNdimage):
    @staticmethod
    def distance_transform_edt(
        value: object,
        return_indices: bool = False,
        float64_distances: bool = True,
    ) -> np.ndarray:
        raise RuntimeError('injected EDT failure')


class _EdtFailingRuntime(_NumpyDeviceRuntime):
    ndi = _EdtFailingNdimage()


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


class CudaInterpolationConfigurationTests(unittest.TestCase):
    def test_radius_offload_is_opt_in_while_cuda_rendering_remains_opt_out(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(cuda_interpolation.gpu_interpolation_enabled())
            self.assertFalse(cuda_interpolation.gpu_interpolation_radius_enabled())
            self.assertTrue(
                cuda_interpolation.gpu_interpolation_render_autotune_enabled()
            )
            self.assertEqual(cuda_interpolation.gpu_interpolation_stream_count(), 4)

        with mock.patch.dict(
            os.environ,
            {
                'YOLO_TTA_GPU_INTERPOLATION': '1',
                'YOLO_TTA_GPU_INTERPOLATION_RADIUS': '1',
            },
            clear=True,
        ):
            self.assertTrue(cuda_interpolation.gpu_interpolation_enabled())
            self.assertTrue(cuda_interpolation.gpu_interpolation_radius_enabled())

        with mock.patch.dict(
            os.environ,
            {'YOLO_TTA_GPU_INTERPOLATION_STREAMS': '7'},
            clear=True,
        ):
            self.assertEqual(cuda_interpolation.gpu_interpolation_stream_count(), 7)

        with mock.patch.dict(
            os.environ,
            {
                'YOLO_TTA_GPU_INTERPOLATION': '0',
                'YOLO_TTA_GPU_INTERPOLATION_RADIUS': '1',
            },
            clear=True,
        ):
            self.assertFalse(cuda_interpolation.gpu_interpolation_enabled())
            self.assertTrue(cuda_interpolation.gpu_interpolation_radius_enabled())

        with mock.patch.dict(
            os.environ,
            {'YOLO_TTA_GPU_INTERPOLATION_RENDER_AUTOTUNE': '0'},
            clear=True,
        ):
            self.assertFalse(
                cuda_interpolation.gpu_interpolation_render_autotune_enabled()
            )


class CudaInterpolationRendererContractTests(unittest.TestCase):
    def test_async_cupy_copy_targets_retained_pinned_host_storage(self) -> None:
        allocations: list[bytearray] = []
        calls: list[tuple[object, np.ndarray, object, bool]] = []

        class _FakeCuda:
            @staticmethod
            def alloc_pinned_memory(size: int) -> bytearray:
                allocation = bytearray(int(size))
                allocations.append(allocation)
                return allocation

        class _FakeXp:
            cuda = _FakeCuda()

            @staticmethod
            def asnumpy(
                value: object, *, out: np.ndarray, stream: object, blocking: bool,
            ) -> np.ndarray:
                calls.append((value, out, stream, bool(blocking)))
                out[...] = np.arange(out.size, dtype=out.dtype).reshape(out.shape)
                return out

        runtime = object.__new__(cuda_interpolation._CupyInterpolationRuntime)
        runtime.xp = _FakeXp()
        source = mock.Mock(shape=(2, 3), dtype=np.dtype(np.int16))
        stream = object()

        result = runtime.to_host_async(source, stream)

        self.assertEqual(len(allocations), 1)
        self.assertEqual(len(allocations[0]), 12)
        self.assertIs(calls[0][0], source)
        self.assertIs(calls[0][1], result)
        self.assertIs(calls[0][2], stream)
        self.assertFalse(calls[0][3])
        np.testing.assert_array_equal(
            result, np.arange(6, dtype=np.int16).reshape(2, 3),
        )

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
            radius_failed_reason = None

            def __init__(self, device_index: int) -> None:
                events.append(f'construct:{int(device_index)}')

            @staticmethod
            def preflight(*, check_radius: bool = True) -> None:
                events.append(f'preflight:{bool(check_radius)}')

        fake_torch = mock.Mock(cuda=_FakeCuda())
        for check_radius in (False, True):
            with self.subTest(check_radius=check_radius):
                events.clear()
                with (
                    mock.patch.dict(
                        os.environ,
                        {
                            'YOLO_TTA_GPU_INTERPOLATION': '1',
                            'YOLO_TTA_GPU_INTERPOLATION_CREATE_CONTEXT': '0',
                            'YOLO_TTA_GPU_INTERPOLATION_RADIUS': (
                                '1' if check_radius else '0'
                            ),
                        },
                        clear=True,
                    ),
                    mock.patch.dict(sys.modules, {'torch': fake_torch}),
                    mock.patch.object(
                        cuda_interpolation, 'CudaInterpolationRenderer', _FakeRenderer,
                    ),
                ):
                    renderer, status = (
                        cuda_interpolation.create_cuda_interpolation_renderer(
                            process_worker=True,
                        )
                    )

                self.assertIsInstance(renderer, _FakeRenderer)
                self.assertEqual(status, 'cuda:2 CuPy/CuPyX')
                self.assertEqual(
                    events,
                    [
                        'synchronize:2', 'empty_cache', 'construct:2', 'mem_info',
                        f'preflight:{check_radius}',
                    ],
                )

    def test_factory_keeps_rendering_when_only_radius_preflight_fails(self) -> None:
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
                return 0

            @staticmethod
            def synchronize(_device: int) -> None:
                return None

            @staticmethod
            def empty_cache() -> None:
                return None

        class _FakeRuntime:
            @staticmethod
            def mem_info() -> tuple[int, int]:
                return 4 * 1024 ** 3, 8 * 1024 ** 3

        class _SplitPreflightRenderer:
            reserve_bytes = 1024 ** 3
            runtime = _FakeRuntime()
            visible_device_token = None

            def __init__(self, device_index: int) -> None:
                self.device_index = int(device_index)
                self.radius_failed_reason: str | None = None

            def preflight(self, *, check_radius: bool = True) -> None:
                events.append(f'preflight:{bool(check_radius)}')
                if bool(check_radius):
                    raise RuntimeError('injected EDT failure')

            def disable_radius(self, exc: BaseException) -> bool:
                events.append('disable-radius')
                self.radius_failed_reason = f'{type(exc).__name__}: {exc}'
                return True

            @staticmethod
            def close() -> None:
                events.append('close')

        fake_torch = mock.Mock(cuda=_FakeCuda())
        with (
            mock.patch.dict(
                os.environ,
                {
                    'YOLO_TTA_GPU_INTERPOLATION': '1',
                    'YOLO_TTA_GPU_INTERPOLATION_CREATE_CONTEXT': '0',
                    'YOLO_TTA_GPU_INTERPOLATION_RADIUS': '1',
                },
                clear=True,
            ),
            mock.patch.dict(sys.modules, {'torch': fake_torch}),
            mock.patch.object(
                cuda_interpolation,
                'CudaInterpolationRenderer',
                _SplitPreflightRenderer,
            ),
        ):
            renderer, status = cuda_interpolation.create_cuda_interpolation_renderer(
                process_worker=True,
            )

        self.assertIsInstance(renderer, _SplitPreflightRenderer)
        self.assertEqual(
            status,
            'cuda:0 CuPy/CuPyX; radius unavailable '
            '(RuntimeError: injected EDT failure)',
        )
        self.assertEqual(
            events,
            ['preflight:True', 'disable-radius', 'preflight:False'],
        )

    def test_preflight_skips_edt_when_cuda_radius_is_disabled(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_EdtFailingRuntime(), cache_bytes=0, reserve_bytes=0,
        )

        renderer.preflight(check_radius=False)
        self.assertTrue(renderer.available)
        with self.assertRaisesRegex(RuntimeError, 'injected EDT failure'):
            renderer.preflight(check_radius=True)

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

    def test_distinct_destinations_enqueue_while_another_d2h_is_pending(self) -> None:
        section = np.ones((3, 3), dtype=bool)
        plan = _plan(
            np.ones((3, 3), dtype=np.float32),
            cached_sections=[None, section, None],
        )
        runtime = _DeferredNumpyRuntime()
        renderer = CudaInterpolationRenderer(
            runtime=runtime, cache_bytes=0, reserve_bytes=0, stream_count=2,
        )
        destinations = [
            np.zeros((7, 7), dtype=np.uint8),
            np.zeros((7, 7), dtype=np.uint8),
        ]
        results: list[object] = []
        errors: list[BaseException] = []

        def _render(index: int) -> None:
            try:
                results.append(renderer.render_slice(
                    destinations[int(index)], [(plan, 1, 1)],
                ))
            except BaseException as exc:  # pragma: no cover - assertion reports detail
                errors.append(exc)

        first = threading.Thread(target=_render, args=(0,))
        second = threading.Thread(target=_render, args=(1,))
        first.start()
        completions = runtime.wait_for_completions(1)
        self.assertTrue(completions[0].wait_started.wait(timeout=5.0))
        first_before_completion = destinations[0].copy()

        second.start()
        completions = runtime.wait_for_completions(2)
        self.assertTrue(completions[1].wait_started.wait(timeout=5.0))
        # The first thread is blocked outside the renderer lock. The second therefore
        # acquired a distinct lease stream and enqueued its own D2H before either commit.
        self.assertEqual(runtime.created_streams, [1, 2])
        np.testing.assert_array_equal(destinations[0], first_before_completion)
        np.testing.assert_array_equal(destinations[1], np.zeros_like(destinations[1]))

        for completion in completions:
            completion.release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(int(np.count_nonzero(destinations[0])), 9)
        self.assertEqual(int(np.count_nonzero(destinations[1])), 9)
        telemetry = renderer.telemetry()
        self.assertEqual(int(telemetry['stream_peak']), 2)
        self.assertEqual(int(telemetry['max_streams']), 2)

    def test_uncached_morphology_executes_outside_the_renderer_lock(self) -> None:
        plans = [
            _plan(np.ones((5, 5), dtype=np.float32), cached_sections=[]),
            _plan(np.ones((5, 5), dtype=np.float32), cached_sections=[]),
        ]
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
            stream_count=2,
        )
        original_keep = renderer._keep_center_component_and_fill
        guard = threading.Lock()
        two_active = threading.Event()
        active = 0
        peak = 0

        def _concurrent_keep(section: object, stream: object = None) -> object:
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(int(peak), int(active))
                if active >= 2:
                    two_active.set()
            try:
                if not two_active.wait(timeout=5.0):
                    raise AssertionError('uncached CUDA morphology remained lock-serialized')
                return original_keep(section, stream=stream)
            finally:
                with guard:
                    active -= 1

        destinations = [
            np.zeros((9, 9), dtype=np.uint8),
            np.zeros((9, 9), dtype=np.uint8),
        ]
        errors: list[BaseException] = []

        def _render(index: int) -> None:
            try:
                renderer.render_slice(
                    destinations[int(index)], [(plans[int(index)], 1, 1)],
                )
            except BaseException as exc:  # pragma: no cover - assertion reports detail
                errors.append(exc)

        with mock.patch.object(
            renderer, '_keep_center_component_and_fill', side_effect=_concurrent_keep,
        ):
            threads = [threading.Thread(target=_render, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(peak, 2)
        self.assertTrue(all(int(np.count_nonzero(dst)) == 25 for dst in destinations))

    def test_inflight_cache_value_survives_cross_stream_eviction(self) -> None:
        plan = _plan(
            np.ones((3, 3), dtype=np.float32),
            cached_sections=[None, None, None],
        )
        runtime = _DeferredNumpyRuntime()
        renderer = CudaInterpolationRenderer(
            runtime=runtime, cache_bytes=1024, reserve_bytes=0, stream_count=2,
        )
        cached_device_section = np.ones((3, 3), dtype=bool)
        cached_ref = weakref.ref(cached_device_section)
        renderer._cache_put(
            ('section', id(plan.cached_sections), 1),
            cached_device_section,
            int(cached_device_section.nbytes),
            (plan.cached_sections,),
        )
        del cached_device_section

        destination = np.zeros((7, 7), dtype=np.uint8)
        errors: list[BaseException] = []

        def _render() -> None:
            try:
                renderer.render_slice(destination, [(plan, 1, 1)])
            except BaseException as exc:  # pragma: no cover - assertion reports detail
                errors.append(exc)

        thread = threading.Thread(target=_render)
        thread.start()
        completion = runtime.wait_for_completions(1)[0]
        self.assertTrue(completion.wait_started.wait(timeout=5.0))

        renderer.release_plans((plan,))
        gc.collect()
        self.assertIsNotNone(
            cached_ref(), 'in-flight stream lost its device cache owner after eviction',
        )

        completion.release.set()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(int(np.count_nonzero(destination)), 9)
        gc.collect()
        self.assertIsNone(cached_ref())

    def test_async_completion_failure_preserves_host_transaction(self) -> None:
        section = np.ones((3, 3), dtype=bool)
        plan = _plan(
            np.ones((3, 3), dtype=np.float32),
            cached_sections=[None, section, None],
        )
        runtime = _DeferredNumpyRuntime(fail_completion=True)
        renderer = CudaInterpolationRenderer(
            runtime=runtime, cache_bytes=0, reserve_bytes=0, stream_count=1,
        )
        destination = np.zeros((7, 7), dtype=np.uint8)
        before = destination.copy()
        errors: list[BaseException] = []

        def _render() -> None:
            try:
                renderer.render_slice(destination, [(plan, 1, 1)])
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=_render)
        thread.start()
        completion = runtime.wait_for_completions(1)[0]
        self.assertTrue(completion.wait_started.wait(timeout=5.0))
        np.testing.assert_array_equal(destination, before)
        completion.release.set()
        thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), 'injected asynchronous D2H failure')
        np.testing.assert_array_equal(destination, before)

    def test_radius_failure_does_not_disable_gpu_rendering(self) -> None:
        section = np.ones((3, 3), dtype=bool)
        plan = _plan(
            np.ones((3, 3), dtype=np.float32),
            cached_sections=[None, section, None],
        )
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )

        self.assertTrue(renderer.disable_radius(RuntimeError('injected radius failure')))
        self.assertFalse(renderer.radius_available)
        self.assertTrue(renderer.available)

        destination = np.zeros((7, 7), dtype=np.uint8)
        result = renderer.render_slice(destination, [(plan, 1, 1)])
        self.assertEqual(result.added_voxels, 9)
        self.assertEqual(int(np.count_nonzero(destination)), 9)
        telemetry = renderer.telemetry()
        self.assertIsNone(telemetry['failed_reason'])
        self.assertIn('injected radius failure', str(telemetry['radius_failed_reason']))

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

    def test_radius_planners_execute_on_independent_stream_leases(self) -> None:
        plans = [
            _plan(np.ones((5, 5), dtype=np.float32), steps=2),
            _plan(np.ones((5, 5), dtype=np.float32), steps=2),
        ]
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
            stream_count=2,
        )
        original_section_radius = renderer._section_radius
        guard = threading.Lock()
        two_active = threading.Event()
        active = 0
        peak = 0

        def _concurrent_radius(section: object) -> float:
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(int(peak), int(active))
                if active >= 2:
                    two_active.set()
            try:
                if not two_active.wait(timeout=5.0):
                    raise AssertionError('CUDA radius evaluation remained renderer-serialized')
                return original_section_radius(section)
            finally:
                with guard:
                    active -= 1

        results: list[float] = []
        errors: list[BaseException] = []

        def _estimate(index: int) -> None:
            try:
                results.append(renderer.estimate_min_radius(plans[int(index)]))
            except BaseException as exc:  # pragma: no cover - assertion reports detail
                errors.append(exc)

        with mock.patch.object(renderer, '_section_radius', side_effect=_concurrent_radius):
            threads = [
                threading.Thread(target=_estimate, args=(index,)) for index in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(peak, 2)
        self.assertEqual(int(renderer.telemetry()['stream_peak']), 2)

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
        os.environ.get('XTA_TEST_CUDA', '').strip() == '1',
        'set XTA_TEST_CUDA=1 on an NVIDIA CUDA host',
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

    @staticmethod
    def _accept_radius_on_cpu(
        plan: interpolation.SliceBridgeRenderPlan,
        **_kwargs: object,
    ) -> float:
        plan.cached_sections[:] = (
            [None]
            + [np.ones(plan.sdf0.shape, dtype=bool) for _ in range(int(plan.steps) - 1)]
            + [None]
        )
        return 1.0

    @staticmethod
    def _clock(*ticks: float) -> mock.Mock:
        clock = mock.Mock()
        clock.perf_counter.side_effect = [float(tick) for tick in ticks]
        return clock

    @staticmethod
    def _distinct_single_pixel_plans() -> list[interpolation.SliceBridgeRenderPlan]:
        section = np.ones((1, 1), dtype=bool)
        sdf = np.ones((1, 1), dtype=np.float32)
        return [
            _plan(
                sdf,
                source_anchor=(anchor, anchor),
                target_anchor=(anchor, anchor),
                cached_sections=[None, section.copy(), None],
            )
            for anchor in (1, 5)
        ]

    def _run_pass(
        self,
        renderer: object,
        *,
        required: bool = False,
        steps: int = 2,
        interpolate_min_radius: float = 0.0,
        radius_enabled: bool | None = None,
        render_autotune: bool | None = False,
        exercise_radius_routing: bool = False,
        seed_count: int = 1,
        workers: int = 1,
        planned_plans: list[interpolation.SliceBridgeRenderPlan] | None = None,
        plan_batch_budget_bytes: int | None = None,
    ):
        mask, labels, seed, plan_result = self._pass_fixture(steps=int(steps))
        plan = plan_result.plans[0]
        render_plans = (
            list(planned_plans)
            if planned_plans is not None
            else [plan for _ in range(max(1, int(seed_count)))]
        )
        if not render_plans:
            raise ValueError('planned_plans must contain at least one plan')
        seeds = [seed for _ in render_plans]
        candidate = interpolation.SliceProjectionCandidate(
            source_label=1,
            target_label=2,
            source_point=seed.point,
            target_point=plan.target_point,
            slice_distance=int(steps),
        )
        plan_seed_patch = (
            contextlib.nullcontext()
            if bool(exercise_radius_routing)
            else mock.patch.object(
                interpolation,
                '_plan_slice_seed_bridges',
                side_effect=[
                    interpolation.SliceSeedBridgePlanResult(
                        candidate_connections=1,
                        accepted_connections=1,
                        default_bridges=1,
                        plans=[render_plan],
                    )
                    for render_plan in render_plans
                ],
            )
        )
        candidate_patch = (
            mock.patch.object(
                interpolation, '_find_slice_projection_candidates',
                return_value=[candidate],
            )
            if bool(exercise_radius_routing)
            else contextlib.nullcontext()
        )
        walkback_patch = (
            mock.patch.object(
                interpolation, '_collect_walkback_source_points', return_value=[],
            )
            if bool(exercise_radius_routing)
            else contextlib.nullcontext()
        )
        build_plan_patch = (
            mock.patch.object(
                interpolation, '_build_linear_slice_bridge_plan', return_value=plan,
            )
            if bool(exercise_radius_routing)
            else contextlib.nullcontext()
        )
        env = {
            'YOLO_TTA_GPU_INTERPOLATION': '1',
            'YOLO_TTA_GPU_INTERPOLATION_REQUIRED': '1' if required else '0',
            'YOLO_TTA_INTERPOLATION_CACHE_BRIDGE_SECTIONS': '1',
            # CUDA routing tests should not depend on an optional Numba installation
            # or pay a first-call JIT cost inside the CPU side of the autotune probe.
            'YOLO_TTA_INTERPOLATION_COMPILED_KERNELS': '0',
        }
        if radius_enabled is not None:
            env['YOLO_TTA_GPU_INTERPOLATION_RADIUS'] = (
                '1' if bool(radius_enabled) else '0'
            )
        if render_autotune is not None:
            env['YOLO_TTA_GPU_INTERPOLATION_RENDER_AUTOTUNE'] = (
                '1' if bool(render_autotune) else '0'
            )
        plan_budget_patch = (
            mock.patch.object(
                interpolation,
                'interpolation_plan_batch_budget_bytes',
                return_value=int(plan_batch_budget_bytes),
            )
            if plan_batch_budget_bytes is not None
            else contextlib.nullcontext()
        )
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
                interpolation, '_build_slice_endpoint_seeds', return_value=(seeds, 1),
            ),
            plan_seed_patch,
            candidate_patch,
            walkback_patch,
            build_plan_patch,
            plan_budget_patch,
            mock.patch.object(
                interpolation, 'should_use_in_memory_workspace', return_value=True,
            ),
            mock.patch.object(
                interpolation, 'create_cuda_interpolation_renderer',
                return_value=(renderer, 'test cuda:0'),
            ),
            mock.patch.dict(os.environ, env, clear=True),
        ):
            stats = interpolation.interpolate_view_volume_pass_inplace(
                mask_mm=mask,
                work_dir=Path(temp_context.name) / 'work',
                pass_tag='gpu-routing',
                max_slice_distance=2,
                search_angle_deg=15.0,
                interpolation_walk_back=0,
                interpolation_candidates=1,
                interpolate_min_radius=float(interpolate_min_radius),
                workers=int(workers),
            )
        return mask, stats

    def test_required_mode_rejects_cpu_admission_fallback(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'CUDA interpolation is unavailable'):
            self._run_pass(None, required=True)

    def test_optional_cpu_admission_reports_cpu_selected(self) -> None:
        mask, stats = self._run_pass(
            None,
            required=False,
            render_autotune=None,
        )

        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertEqual(stats['interpolation_render_backend'], 'cpu_numpy_numba')
        self.assertFalse(stats['gpu_interpolation_active'])
        self.assertTrue(stats['gpu_interpolation_render_autotune_enabled'])
        self.assertIs(stats['gpu_interpolation_render_selected'], False)
        self.assertEqual(int(stats['gpu_interpolation_batches']), 0)

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

    def test_painting_dispatches_disjoint_slices_up_to_renderer_stream_bound(self) -> None:
        inner = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
            stream_count=2,
        )

        class _ConcurrentRenderer:
            available = True
            max_streams = 2

            def __init__(self) -> None:
                self._guard = threading.Lock()
                self._two_active = threading.Event()
                self.active = 0
                self.peak = 0

            def render_slice(self, *args: object, **kwargs: object):
                with self._guard:
                    self.active += 1
                    self.peak = max(int(self.peak), int(self.active))
                    if self.active >= 2:
                        self._two_active.set()
                try:
                    if not self._two_active.wait(timeout=5.0):
                        raise AssertionError('CUDA slice painting remained sequential')
                    return inner.render_slice(*args, **kwargs)
                finally:
                    with self._guard:
                        self.active -= 1

            @staticmethod
            def disable(exc: BaseException) -> bool:
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

        renderer = _ConcurrentRenderer()

        class _ProgressFixture:
            def __init__(self, iterable: object = None, **_kwargs: object) -> None:
                self.iterable = iterable

            def __iter__(self):
                return iter(self.iterable)  # type: ignore[arg-type]

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            @staticmethod
            def update(_count: int) -> None:
                return None

        with mock.patch.object(runtime_helpers, 'tqdm', _ProgressFixture):
            mask, stats = self._run_pass(renderer, steps=4, workers=4)

        self.assertEqual(renderer.peak, 2)
        self.assertEqual(int(np.count_nonzero(mask[1:4])), 27)
        self.assertEqual(int(stats['gpu_interpolation_rendered_sections']), 3)

    def test_render_autotune_selects_cpu_after_a_successful_gpu_probe(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        first_plan, second_plan = self._distinct_single_pixel_plans()

        # Calls are planner start; probe CPU start/end; probe GPU start/end;
        # second-batch CPU start/end; planner end. The 1s/2s comparison forces CPU.
        with mock.patch.object(
            interpolation,
            'time',
            self._clock(0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0),
        ):
            mask, stats = self._run_pass(
                renderer,
                render_autotune=True,
                planned_plans=[first_plan, second_plan],
                plan_batch_budget_bytes=1,
            )

        self.assertEqual(int(np.count_nonzero(mask[1])), 2)
        self.assertEqual(int(mask[1, 1, 1]), 1)
        # This pixel belongs only to the post-probe batch. Its presence proves the
        # selected CPU backend continued rendering instead of silently dropping work.
        self.assertEqual(int(mask[1, 5, 5]), 1)
        self.assertEqual(int(stats['planner_plan_batches']), 2)
        self.assertTrue(stats['gpu_interpolation_render_autotune_enabled'])
        self.assertFalse(stats['gpu_interpolation_render_selected'])
        self.assertEqual(
            stats['interpolation_render_backend'],
            'cuda_cupy_probe+cpu_numpy_numba',
        )
        self.assertEqual(int(stats['gpu_interpolation_batches']), 1)
        self.assertEqual(int(stats['gpu_interpolation_rendered_sections']), 1)
        self.assertAlmostEqual(
            float(stats['gpu_interpolation_render_probe_gpu_seconds']), 2.0,
        )
        self.assertAlmostEqual(
            float(stats['gpu_interpolation_render_probe_cpu_seconds']), 1.0,
        )
        self.assertAlmostEqual(float(stats['planner_cpu_render_wall_seconds']), 2.0)

    def test_render_autotune_keeps_gpu_after_a_clear_gpu_win(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        first_plan, second_plan = self._distinct_single_pixel_plans()

        # CPU is measured first on the real empty destination; the following
        # 0.5s/1s GPU/CPU comparison is comfortably beyond the 5% win threshold.
        with mock.patch.object(
            interpolation,
            'time',
            self._clock(0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0),
        ):
            mask, stats = self._run_pass(
                renderer,
                render_autotune=True,
                planned_plans=[first_plan, second_plan],
                plan_batch_budget_bytes=1,
            )

        self.assertEqual(int(np.count_nonzero(mask[1])), 2)
        self.assertEqual(int(mask[1, 1, 1]), 1)
        self.assertEqual(int(mask[1, 5, 5]), 1)
        self.assertEqual(int(stats['planner_plan_batches']), 2)
        self.assertTrue(stats['gpu_interpolation_render_autotune_enabled'])
        self.assertTrue(stats['gpu_interpolation_render_selected'])
        self.assertEqual(
            stats['interpolation_render_backend'],
            'cuda_cupy_crop_bounded',
        )
        self.assertEqual(int(stats['gpu_interpolation_batches']), 2)
        self.assertEqual(int(stats['gpu_interpolation_rendered_sections']), 2)
        self.assertAlmostEqual(
            float(stats['gpu_interpolation_render_probe_gpu_seconds']), 0.5,
        )
        self.assertAlmostEqual(
            float(stats['gpu_interpolation_render_probe_cpu_seconds']), 1.0,
        )
        self.assertAlmostEqual(float(stats['planner_cpu_render_wall_seconds']), 1.0)

    def test_default_hybrid_uses_cpu_radius_and_gpu_rendering(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        with (
            mock.patch.object(
                interpolation,
                '_estimate_linear_slice_bridge_min_radius_from_plan',
                side_effect=self._accept_radius_on_cpu,
            ) as cpu_estimate,
            mock.patch.object(
                renderer,
                'estimate_min_radius',
                wraps=renderer.estimate_min_radius,
            ) as gpu_estimate,
            mock.patch.object(
                interpolation,
                'time',
                self._clock(0.0, 1.0, 2.0, 3.0, 3.5, 4.0),
            ),
        ):
            mask, stats = self._run_pass(
                renderer,
                interpolate_min_radius=0.5,
                radius_enabled=None,
                render_autotune=None,
                exercise_radius_routing=True,
            )

        self.assertEqual(cpu_estimate.call_count, 1)
        gpu_estimate.assert_not_called()
        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertEqual(stats['interpolation_render_backend'], 'cuda_cupy_crop_bounded')
        self.assertEqual(stats['interpolation_radius_backend'], 'cpu_numpy_numba')
        self.assertTrue(stats['gpu_interpolation_active'])
        self.assertTrue(stats['gpu_interpolation_render_autotune_enabled'])
        self.assertTrue(stats['gpu_interpolation_render_selected'])
        self.assertFalse(stats['gpu_interpolation_radius_enabled'])
        self.assertFalse(stats['gpu_interpolation_radius_active'])
        self.assertEqual(int(stats['gpu_interpolation_estimated_plans']), 0)
        self.assertEqual(int(stats['gpu_interpolation_rendered_sections']), 1)

    def test_required_mode_allows_the_intentional_default_cpu_radius(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        with mock.patch.object(
            interpolation,
            '_estimate_linear_slice_bridge_min_radius_from_plan',
            side_effect=self._accept_radius_on_cpu,
        ) as cpu_estimate:
            mask, stats = self._run_pass(
                renderer,
                required=True,
                interpolate_min_radius=0.5,
                radius_enabled=None,
                exercise_radius_routing=True,
            )

        self.assertEqual(cpu_estimate.call_count, 1)
        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertTrue(stats['gpu_interpolation_required'])
        self.assertFalse(stats['gpu_interpolation_radius_enabled'])
        self.assertEqual(stats['interpolation_radius_backend'], 'cpu_numpy_numba')
        self.assertEqual(int(stats['gpu_interpolation_rendered_sections']), 1)

    def test_required_mode_bypasses_render_autotune_and_cpu_replay(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        first_plan, second_plan = self._distinct_single_pixel_plans()

        # Required mode must never enter either the compatibility probe replay or a
        # post-probe CPU route, even when the autotune environment switch is enabled.
        with mock.patch.object(
            interpolation,
            '_paint_linear_slice_bridge_plan_onto_slice',
            side_effect=AssertionError('required mode attempted CPU rendering'),
        ) as cpu_paint:
            mask, stats = self._run_pass(
                renderer,
                required=True,
                render_autotune=True,
                planned_plans=[first_plan, second_plan],
                plan_batch_budget_bytes=1,
            )

        cpu_paint.assert_not_called()
        self.assertEqual(int(np.count_nonzero(mask[1])), 2)
        self.assertFalse(stats['gpu_interpolation_render_autotune_enabled'])
        self.assertTrue(stats['gpu_interpolation_render_selected'])
        self.assertEqual(int(stats['gpu_interpolation_batches']), 2)
        self.assertEqual(float(stats['planner_cpu_render_wall_seconds']), 0.0)

    def test_radius_env_override_restores_cuda_radius_evaluation(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        with mock.patch.object(
            interpolation,
            '_estimate_linear_slice_bridge_min_radius_from_plan',
            side_effect=self._accept_radius_on_cpu,
        ) as cpu_estimate:
            mask, stats = self._run_pass(
                renderer,
                interpolate_min_radius=0.5,
                radius_enabled=True,
                exercise_radius_routing=True,
            )

        cpu_estimate.assert_not_called()
        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertEqual(stats['interpolation_radius_backend'], 'cuda_cupy')
        self.assertTrue(stats['gpu_interpolation_radius_enabled'])
        self.assertTrue(stats['gpu_interpolation_radius_active'])
        self.assertEqual(int(stats['gpu_interpolation_estimated_plans']), 1)
        self.assertGreater(int(stats['gpu_interpolation_estimated_sections']), 0)
        self.assertEqual(int(stats['gpu_interpolation_radius_fallback_plans']), 0)

    def test_radius_failure_falls_back_to_cpu_without_disabling_gpu_render(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=1024, reserve_bytes=0,
        )

        def _fail_after_caching_stale_section(
            plan: interpolation.SliceBridgeRenderPlan,
            **_kwargs: object,
        ) -> float:
            stale_section = np.zeros(plan.sdf0.shape, dtype=bool)
            renderer._cache_put(
                ('section', id(plan.cached_sections), 1),
                stale_section,
                int(stale_section.nbytes),
                (plan.cached_sections,),
            )
            raise RuntimeError('injected device radius failure')

        with (
            mock.patch.object(
                renderer,
                'estimate_min_radius',
                side_effect=_fail_after_caching_stale_section,
            ),
            mock.patch.object(
                interpolation,
                '_estimate_linear_slice_bridge_min_radius_from_plan',
                side_effect=self._accept_radius_on_cpu,
            ) as cpu_estimate,
        ):
            mask, stats = self._run_pass(
                renderer,
                interpolate_min_radius=0.5,
                radius_enabled=True,
                exercise_radius_routing=True,
            )

        self.assertEqual(cpu_estimate.call_count, 1)
        # The stale all-false device section must be discarded before the CPU retry
        # publishes its authoritative all-true host section for GPU painting.
        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertEqual(stats['interpolation_render_backend'], 'cuda_cupy_crop_bounded')
        self.assertEqual(stats['interpolation_radius_backend'], 'cuda_cupy+cpu_fallback')
        self.assertTrue(stats['gpu_interpolation_active'])
        self.assertFalse(stats['gpu_interpolation_radius_active'])
        self.assertEqual(int(stats['gpu_interpolation_radius_fallback_plans']), 1)
        self.assertIn(
            'injected device radius failure',
            str(stats['gpu_interpolation_radius_fallback_reason']),
        )
        self.assertEqual(int(stats['gpu_interpolation_rendered_sections']), 1)
        self.assertIsNone(stats['gpu_interpolation_fallback_reason'])

    def test_required_mode_makes_opted_in_cuda_radius_failure_fatal(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        with (
            mock.patch.object(
                renderer,
                'estimate_min_radius',
                side_effect=RuntimeError('injected required radius failure'),
            ),
            self.assertRaisesRegex(
                RuntimeError, 'required CUDA interpolation radius evaluation failed',
            ),
        ):
            self._run_pass(
                renderer,
                required=True,
                interpolate_min_radius=0.5,
                radius_enabled=True,
                exercise_radius_routing=True,
            )

    def test_required_mode_rejects_radius_disabled_during_preflight(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        renderer.disable_radius(RuntimeError('injected radius preflight failure'))

        with self.assertRaisesRegex(
            RuntimeError, 'CUDA radius evaluation is unavailable',
        ):
            self._run_pass(
                renderer,
                required=True,
                interpolate_min_radius=0.5,
                radius_enabled=True,
                exercise_radius_routing=True,
            )

    def test_required_mode_ignores_disabled_radius_when_filter_is_inactive(self) -> None:
        renderer = CudaInterpolationRenderer(
            runtime=_NumpyDeviceRuntime(), cache_bytes=0, reserve_bytes=0,
        )
        renderer.disable_radius(RuntimeError('injected unused radius failure'))

        with mock.patch.object(
            renderer,
            'estimate_min_radius',
            wraps=renderer.estimate_min_radius,
        ) as gpu_estimate:
            mask, stats = self._run_pass(
                renderer,
                required=True,
                interpolate_min_radius=0.0,
                radius_enabled=True,
                exercise_radius_routing=True,
            )

        gpu_estimate.assert_not_called()
        self.assertEqual(int(np.count_nonzero(mask[1])), 9)
        self.assertTrue(stats['gpu_interpolation_required'])
        self.assertTrue(stats['gpu_interpolation_radius_enabled'])
        self.assertFalse(stats['gpu_interpolation_radius_active'])
        self.assertEqual(stats['interpolation_radius_backend'], 'disabled')
        self.assertEqual(stats['interpolation_render_backend'], 'cuda_cupy_crop_bounded')
        self.assertTrue(stats['gpu_interpolation_render_selected'])
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

        with (
            mock.patch.object(
                interpolation,
                '_paint_linear_slice_bridge_plan_onto_slice',
                side_effect=AssertionError('required render failure replayed on CPU'),
            ) as cpu_paint,
            self.assertRaisesRegex(
                RuntimeError,
                'required CUDA interpolation bridge rendering failed',
            ),
        ):
            self._run_pass(_FailingRenderer(), required=True)
        cpu_paint.assert_not_called()

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
