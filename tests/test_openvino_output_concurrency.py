from __future__ import annotations

import threading
import unittest
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from XTA import workers


class _FakeOutputTensor:
    def __init__(self, value: int) -> None:
        self.data = np.asarray([int(value)], dtype=np.float32)


class _FakeInferRequest:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def get_output_tensor(self, _index: int) -> _FakeOutputTensor:
        return _FakeOutputTensor(self.value)


class _FakeAsyncInferQueue:
    def __init__(self) -> None:
        self.callback = None
        self.pending: list[object] = []
        self.wait_called = False

    def set_callback(self, callback: object) -> None:
        self.callback = callback

    def start_async(self, _inputs: object, *, userdata: object) -> None:
        self.pending.append(userdata)

    def wait_all(self) -> None:
        self.wait_called = True
        assert self.callback is not None
        for value, userdata in enumerate(self.pending):
            self.callback(_FakeInferRequest(value), userdata)


def _runner(request_count: int = 2) -> workers._OpenVinoCpuSegmenter:
    runner = workers._OpenVinoCpuSegmenter.__new__(workers._OpenVinoCpuSegmenter)
    runner.request_count = int(request_count)
    runner.batch = 1
    runner.infer_queue = _FakeAsyncInferQueue()
    runner.output_ports = (object(),)
    runner.expected_class_count = None
    runner.input_name = 'images'
    runner.resolved_precision = 'fp32'
    runner.input_element_type = 'f32'
    runner.model_int8_quantized = False
    runner._prepare_input = lambda images: np.zeros(  # type: ignore[method-assign]
        (len(images), 1, 1, 1), dtype=np.float32,
    )
    return runner


def _source(batch_count: int) -> list[tuple[None, list[np.ndarray], None]]:
    image = np.zeros((1, 1, 1), dtype=np.uint8)
    return [(None, [image], None) for _ in range(int(batch_count))]


def _run(
    runner: workers._OpenVinoCpuSegmenter,
    source: object,
    *,
    num_frames: int,
) -> tuple[dict[str, object], np.ndarray]:
    destination = np.zeros((int(num_frames), 1, 1), dtype=np.uint8)
    stats = runner.infer_source_to_union(
        source,
        num_frames=int(num_frames),
        out_size=1,
        conf_threshold=0.1,
        view_union_mm=destination,
        view_confmap_mm=None,
        M_out_to_native=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
        ),
        native_h=1,
        native_w=1,
        min_conf=0.0,
        min_radius=0.0,
    )
    return stats, destination


class OpenVinoOutputConcurrencyTests(unittest.TestCase):
    def test_completed_requests_are_postprocessed_concurrently(self) -> None:
        runner = _runner(request_count=2)
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()
        both_active = threading.Event()

        def fake_payloads(*_args: object, **_kwargs: object) -> list[object]:
            nonlocal active, maximum_active
            with active_lock:
                active += 1
                maximum_active = max(int(maximum_active), int(active))
                if active == 2:
                    both_active.set()
            try:
                if not both_active.wait(timeout=2.0):
                    raise AssertionError('OpenVINO output postprocessing remained serialized')
                return [object()]
            finally:
                with active_lock:
                    active -= 1

        def fake_process(
            frame_index: int,
            _payload: object,
            _out_size: int,
            target_union: np.ndarray,
            _target_conf: object,
            _affine: np.ndarray,
            _native_h: int,
            _native_w: int,
            *,
            slice_lock: object,
        ) -> tuple[int, int]:
            self.assertIsNone(slice_lock)
            target_union[int(frame_index), 0, 0] = np.uint8(1)
            return int(frame_index) + 1, 1

        with (
            mock.patch.object(
                workers, '_openvino_cpu_payloads_from_outputs', side_effect=fake_payloads,
            ),
            mock.patch.object(
                workers, '_process_cpu_retina_prediction_frame', side_effect=fake_process,
            ),
            mock.patch.object(
                workers, '_cleanup_prediction_slice_inplace', return_value=True,
            ),
        ):
            stats, destination = _run(runner, _source(2), num_frames=2)

        self.assertEqual(maximum_active, 2)
        self.assertEqual(stats['prediction_count'], 3)
        self.assertEqual(stats['frames_with_predictions'], 2)
        np.testing.assert_array_equal(destination[:, 0, 0], np.asarray([1, 1]))

    def test_concurrent_failures_report_the_earliest_submitted_batch(self) -> None:
        runner = _runner(request_count=2)
        both_active = threading.Barrier(2, timeout=2.0)
        later_failed = threading.Event()

        def fake_payloads(*_args: object, **_kwargs: object) -> list[object]:
            both_active.wait()
            return [object()]

        def fake_process(frame_index: int, *_args: object, **_kwargs: object) -> tuple[int, int]:
            if int(frame_index) == 1:
                later_failed.set()
                raise ValueError('later batch failed first')
            if not later_failed.wait(timeout=2.0):
                raise AssertionError('later batch never reached postprocessing')
            raise ValueError('earliest submitted batch failed')

        with (
            mock.patch.object(
                workers, '_openvino_cpu_payloads_from_outputs', side_effect=fake_payloads,
            ),
            mock.patch.object(
                workers, '_process_cpu_retina_prediction_frame', side_effect=fake_process,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, 'earliest submitted batch failed',
            ) as raised:
                _run(runner, _source(2), num_frames=2)

        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertFalse(any(
            thread.name.startswith('openvino-output-consumer-')
            for thread in threading.enumerate()
        ))

    def test_source_failure_still_drains_submitted_inference(self) -> None:
        runner = _runner(request_count=2)
        destination = np.zeros((2, 1, 1), dtype=np.uint8)

        def failing_source():
            yield _source(1)[0]
            raise LookupError('source failed')

        def fake_process(
            frame_index: int,
            _payload: object,
            _out_size: int,
            target_union: np.ndarray,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[int, int]:
            target_union[int(frame_index), 0, 0] = np.uint8(1)
            return 1, 1

        with (
            mock.patch.object(
                workers, '_openvino_cpu_payloads_from_outputs', return_value=[object()],
            ),
            mock.patch.object(
                workers, '_process_cpu_retina_prediction_frame', side_effect=fake_process,
            ),
            mock.patch.object(
                workers, '_cleanup_prediction_slice_inplace', return_value=True,
            ),
        ):
            with self.assertRaisesRegex(LookupError, 'source failed'):
                runner.infer_source_to_union(
                    failing_source(),
                    num_frames=2,
                    out_size=1,
                    conf_threshold=0.1,
                    view_union_mm=destination,
                    view_confmap_mm=None,
                    M_out_to_native=np.asarray(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
                    ),
                    native_h=1,
                    native_w=1,
                    min_conf=0.0,
                    min_radius=0.0,
                )

        self.assertTrue(runner.infer_queue.wait_called)
        self.assertEqual(int(destination[0, 0, 0]), 1)
        self.assertFalse(any(
            thread.name.startswith('openvino-output-consumer-')
            for thread in threading.enumerate()
        ))


if __name__ == '__main__':
    unittest.main()
