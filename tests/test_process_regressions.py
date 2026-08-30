from __future__ import annotations

import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from XTA import assembly, interpolation, media, pipeline, runtime


class _DetachHandle:
    def __init__(self, fd: int) -> None:
        self.fd = int(fd)

    def detach(self) -> int:
        return int(self.fd)


class _BrokenDetachHandle:
    def detach(self) -> int:
        raise RuntimeError('detach failure sentinel')


class _SubmitFailureExecutor:
    def submit(self, _function: object, **kwargs: object) -> object:
        stage = np.memmap(
            Path(str(kwargs['mask_path'])),
            dtype=np.dtype(str(kwargs['mask_dtype'])),
            mode='r+',
            shape=tuple(int(value) for value in kwargs['mask_shape']),
        )
        stage[:] = np.uint8(9)
        stage.flush()
        runtime.close_memmap_array(stage)
        raise RuntimeError('submit failure sentinel')


class _SuccessfulExecutor:
    def submit(self, _function: object, **kwargs: object) -> object:
        stage = np.memmap(
            Path(str(kwargs['mask_path'])),
            dtype=np.dtype(str(kwargs['mask_dtype'])),
            mode='r+',
            shape=tuple(int(value) for value in kwargs['mask_shape']),
        )
        stage[:] = np.uint8(9)
        stage.flush()
        runtime.close_memmap_array(stage)

        class _Future:
            @staticmethod
            def result() -> dict[str, object]:
                return {'worker_completed': True}

        return _Future()


class _FailingAuxPool:
    @staticmethod
    def try_submit(_kwargs: object) -> object:
        return object()

    @staticmethod
    def wait(_handle: object) -> dict[str, object]:
        raise RuntimeError('aux wait failure sentinel')


class ProcessRuntimeRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        runtime.set_interpolation_process_executor(None, 0)
        runtime.set_gpu_worker_aux_interpolation_pool(None)
        media.abort_streaming_producers('test teardown')
        media.wait_for_streaming_producers(timeout=5.0)
        media.reset_streaming_state_for_new_run()

    def test_fork_start_method_is_rejected(self) -> None:
        with mock.patch.dict(
            os.environ,
            {'YOLO_TTA_INTERPOLATION_PROCESS_START_METHOD': 'fork'},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, 'spawn or forkserver'):
                runtime.interpolation_process_start_method()

    def test_preflight_surfaces_pickle_failure_synchronously(self) -> None:
        with self.assertRaisesRegex(TypeError, 'not serializable'):
            runtime.preflight_multiprocessing_payload({'callback': lambda: None})

    def test_partial_memfd_materialization_closes_detached_descriptors(self) -> None:
        first_fd, first_writer = os.pipe()
        os.close(first_writer)
        task = {
            'result_mask_fd': _DetachHandle(first_fd),
            'result_mask_fd_key': 'first',
            'result_conf_fd': _BrokenDetachHandle(),
            'result_conf_fd_key': 'second',
        }
        with self.assertRaisesRegex(RuntimeError, 'detach failure sentinel'):
            runtime._materialize_worker_task_memfd_paths(task, {})
        with self.assertRaises(OSError):
            os.fstat(first_fd)

    def _run_interpolation_case(self, executor: object) -> tuple[np.memmap, np.ndarray, dict[str, object], Path]:
        temp_context = tempfile.TemporaryDirectory()
        self.addCleanup(temp_context.cleanup)
        work_dir = Path(temp_context.name)
        backing = work_dir / 'input.u8.dat'
        original = np.memmap(backing, dtype=np.uint8, mode='w+', shape=(2, 2, 2))
        original[:] = np.uint8(1)
        original.flush()

        def _fallback(*, mask_mm: np.ndarray, **_kwargs: object) -> dict[str, object]:
            np.testing.assert_array_equal(np.asarray(mask_mm), np.ones((2, 2, 2), dtype=np.uint8))
            mask_mm[:] = np.uint8(7)
            return {'fallback_saw_clean_input': True}

        runtime.set_interpolation_process_executor(executor, 1)  # type: ignore[arg-type]
        with (
            mock.patch.dict(
                os.environ,
                {
                    'YOLO_TTA_INTERPOLATION_PROCESS_BACKEND': '1',
                    'YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK': '1',
                },
                clear=False,
            ),
            mock.patch.object(assembly, 'view_interpolation_wrap_axis', return_value=False),
            mock.patch.object(interpolation, 'interpolate_view_volume_pass_inplace', side_effect=_fallback),
        ):
            result, stats = runtime.interpolate_view_volume_pass_maybe_process(
                original,
                types.SimpleNamespace(name='test-view'),
                work_dir,
                'fault-injection',
                3,
                15.0,
                1,
                2,
                0.0,
                keep_temp=False,
                workers=1,
            )
        return original, result, stats, work_dir

    def test_submit_failure_fallback_uses_clean_input_and_preserves_identity(self) -> None:
        original, result, stats, work_dir = self._run_interpolation_case(
            _SubmitFailureExecutor(),
        )
        try:
            self.assertIs(result, original)
            np.testing.assert_array_equal(np.asarray(result), np.full((2, 2, 2), 7, dtype=np.uint8))
            self.assertTrue(stats['fallback_saw_clean_input'])
            self.assertEqual(stats['process_backend'], 'fallback_in_process_after_worker_failure')
            self.assertEqual(list(work_dir.glob('*fallback-stage*')), [])
        finally:
            runtime.close_memmap_array(original)

    def test_successful_transaction_commits_back_to_original_mapping(self) -> None:
        temp_context = tempfile.TemporaryDirectory()
        self.addCleanup(temp_context.cleanup)
        work_dir = Path(temp_context.name)
        backing = work_dir / 'input.u8.dat'
        original = np.memmap(backing, dtype=np.uint8, mode='w+', shape=(2, 2, 2))
        original[:] = np.uint8(1)
        original.flush()
        runtime.set_interpolation_process_executor(_SuccessfulExecutor(), 1)  # type: ignore[arg-type]
        with (
            mock.patch.dict(
                os.environ,
                {
                    'YOLO_TTA_INTERPOLATION_PROCESS_BACKEND': '1',
                    'YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK': '1',
                },
                clear=False,
            ),
            mock.patch.object(assembly, 'view_interpolation_wrap_axis', return_value=False),
            mock.patch.object(
                interpolation,
                'interpolate_view_volume_pass_inplace',
                side_effect=AssertionError('fallback must not run after success'),
            ),
        ):
            result, stats = runtime.interpolate_view_volume_pass_maybe_process(
                original,
                types.SimpleNamespace(name='test-view'),
                work_dir,
                'success-injection',
                3,
                15.0,
                1,
                2,
                0.0,
                keep_temp=False,
                workers=1,
            )
        try:
            self.assertIs(result, original)
            np.testing.assert_array_equal(np.asarray(original), np.full((2, 2, 2), 9, dtype=np.uint8))
            self.assertTrue(stats['worker_completed'])
            self.assertEqual(list(work_dir.glob('*fallback-stage*')), [])
        finally:
            runtime.close_memmap_array(original)

    def test_aux_failure_fallback_retains_fallback_backend_telemetry(self) -> None:
        runtime.set_gpu_worker_aux_interpolation_pool(_FailingAuxPool())  # type: ignore[arg-type]
        original, result, stats, _work_dir = self._run_interpolation_case(
            _SuccessfulExecutor(),
        )
        try:
            self.assertIs(result, original)
            np.testing.assert_array_equal(
                np.asarray(result), np.full((2, 2, 2), 7, dtype=np.uint8),
            )
            self.assertTrue(stats['fallback_saw_clean_input'])
            self.assertEqual(
                stats['process_backend'], 'fallback_in_process_after_aux_failure',
            )
        finally:
            runtime.close_memmap_array(original)

    def test_aux_queue_pickle_failure_rolls_back_pending_lease(self) -> None:
        class _Queue:
            def __init__(self) -> None:
                self.items: list[object] = []

            def put(self, item: object) -> None:
                self.items.append(item)

        task_queue = _Queue()
        pool = runtime._GpuWorkerAuxInterpolationPool({0: task_queue})
        self.assertTrue(pool.enable_worker(0))
        handle = pool.try_submit({'unpickleable': lambda: None})
        self.assertIsNone(handle)
        self.assertEqual(pool.outstanding(), 0)
        self.assertEqual(task_queue.items, [])


class PipelineLifecycleRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        media.abort_streaming_producers('test teardown')
        media.wait_for_streaming_producers(timeout=5.0)
        media.reset_streaming_state_for_new_run()

    def test_failed_run_cleans_registered_resources_and_next_run_resets_abort(self) -> None:
        class _Executor:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def shutdown(self, **_kwargs: object) -> None:
                self.shutdown_calls += 1

        class _Queue:
            def __init__(self) -> None:
                self.cancel_calls = 0
                self.close_calls = 0
                self.join_calls = 0

            def cancel_join_thread(self) -> None:
                self.cancel_calls += 1

            def close(self) -> None:
                self.close_calls += 1

            def join_thread(self) -> None:
                self.join_calls += 1

        executor = _Executor()
        process_queue = _Queue()
        calls = 0

        def _implementation() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                pipeline._run_resources().track_executor(executor)
                pipeline._run_resources().track_queue(process_queue)
                raise RuntimeError('pipeline failure sentinel')
            self.assertFalse(media.streaming_producers_aborted())

        with mock.patch.object(pipeline, '_main_impl', side_effect=_implementation):
            with self.assertRaisesRegex(RuntimeError, 'pipeline failure sentinel'):
                pipeline.main()
            self.assertTrue(media.streaming_producers_aborted())
            pipeline.main()
        self.assertEqual(calls, 2)
        self.assertGreaterEqual(executor.shutdown_calls, 1)
        self.assertEqual(process_queue.cancel_calls, 1)
        self.assertEqual(process_queue.close_calls, 1)
        self.assertEqual(process_queue.join_calls, 0)

    def test_compressor_shutdown_failure_does_not_strand_pipeline_lock(self) -> None:
        implementation_calls = 0
        telemetry = mock.Mock()

        def _implementation() -> None:
            nonlocal implementation_calls
            implementation_calls += 1

        with (
            mock.patch.object(pipeline, '_main_impl', side_effect=_implementation),
            mock.patch.object(
                pipeline, 'shutdown_nrrd_gzip_executors',
                side_effect=RuntimeError('compressor shutdown sentinel'),
            ),
            mock.patch.object(pipeline, 'runtime_telemetry', return_value=telemetry),
            mock.patch('builtins.print'),
        ):
            pipeline.main()
            pipeline.main()

        self.assertEqual(implementation_calls, 2)
        self.assertEqual(telemetry.fallback.call_count, 2)

    def test_later_thread_pool_constructor_failure_closes_earlier_pool(self) -> None:
        class _Executor:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def shutdown(self, **_kwargs: object) -> None:
                self.shutdown_calls += 1

        first_executor = _Executor()
        constructor_calls = 0

        def _construct(**_kwargs: object) -> object:
            nonlocal constructor_calls
            constructor_calls += 1
            if constructor_calls == 1:
                return first_executor
            raise RuntimeError('later pool constructor failure sentinel')

        def _implementation() -> None:
            pipeline._create_tracked_thread_pool(
                max_workers=1,
                thread_name_prefix='first-test-pool',
            )
            pipeline._create_tracked_thread_pool(
                max_workers=1,
                thread_name_prefix='second-test-pool',
            )

        with (
            mock.patch.object(pipeline, 'ThreadPoolExecutor', side_effect=_construct),
            mock.patch.object(pipeline, '_main_impl', side_effect=_implementation),
        ):
            with self.assertRaisesRegex(
                RuntimeError, 'later pool constructor failure sentinel',
            ):
                pipeline.main()

        self.assertEqual(constructor_calls, 2)
        self.assertEqual(first_executor.shutdown_calls, 1)

    def test_streaming_reset_refuses_live_producer(self) -> None:
        release = threading.Event()
        media._start_streaming_producer(lambda: release.wait(), name='test-live-producer')
        try:
            with self.assertRaisesRegex(RuntimeError, 'prior producers remain active'):
                media.reset_streaming_state_for_new_run()
        finally:
            release.set()
        self.assertTrue(media.wait_for_streaming_producers(timeout=5.0))
        media.reset_streaming_state_for_new_run()
        self.assertFalse(media.streaming_producers_aborted())


if __name__ == '__main__':
    unittest.main()
