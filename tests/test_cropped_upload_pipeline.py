"""Delayed-DMA ownership/order checks for bounded source-crop upload."""
from __future__ import annotations

import math
import os
import contextlib
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cylindrical_cuda_projection as cuda


class _DelayedUpload:
    """Read the pinned bytes only when an event/stream fence actually completes."""
    def __init__(self, capacity, *, failure=None):
        self.stage = np.zeros(capacity, np.uint8)
        self.device = None
        self.jobs, self.log = [], []
        self.completed = 0
        self.failure = failure
        self.created_events = 0
        self.pin_pointer, self.device_pointer = 10_000, 100_000
        self.stream = SimpleNamespace(ptr=7, synchronize=self.synchronize)
        self.cp = SimpleNamespace(uint8=np.uint8, empty=self.empty,
            cuda=SimpleNamespace(Event=self.event, runtime=SimpleNamespace(
                memcpyHostToDevice=1, memcpyAsync=self.enqueue)))

    def empty(self, size, dtype):
        self.device = np.zeros(size, dtype=dtype)
        return SimpleNamespace(data=SimpleNamespace(ptr=self.device_pointer), storage=self.device)

    def enqueue(self, destination, source, count, kind, stream):
        assert kind == 1 and stream == 7
        self.jobs.append((destination - self.device_pointer, source - self.pin_pointer, count))
        self.log.append(('copy', *self.jobs[-1]))
        if self.failure == 'enqueue' and len(self.jobs) == 2:
            raise RuntimeError('uncertain enqueue')

    def drain(self, stop):
        while self.completed < stop:
            destination, source, count = self.jobs[self.completed]
            self.device[destination:destination + count] = self.stage[source:source + count]
            self.completed += 1

    def synchronize(self):
        self.log.append(('stream_fence',))
        if self.failure == 'stream':
            raise RuntimeError('failed stream fence')
        self.drain(len(self.jobs))

    def event(self, *, disable_timing):
        assert disable_timing
        self.created_events += 1
        if self.failure == 'event_create' or (self.failure == 'event_second_create' and self.created_events == 2):
            raise RuntimeError('optional event unavailable')
        owner = self
        class Event:
            stop = 0
            def record(self, stream):
                assert stream is owner.stream
                self.stop = len(owner.jobs)
                owner.log.append(('record', self.stop))
                if owner.failure == 'event_record':
                    raise RuntimeError('failed event record')
            def synchronize(self):
                owner.log.append(('lane_wait', self.stop))
                if owner.failure == 'event_wait':
                    raise RuntimeError('failed lane fence')
                if owner.failure == 'unsafe_event_wait':
                    return
                owner.drain(self.stop)
        return Event()

    def pack(self, source, boxes, offsets, first, destination):
        if not destination.size:
            return
        if self.failure == 'pack' and first:
            raise RuntimeError('pack failed after previous enqueue')
        start = destination.ctypes.data - self.stage.ctypes.data
        end = start + destination.size
        for _, pending, count in self.jobs[self.completed:]:
            if start < pending + count and pending < end:
                raise AssertionError('pinned upload stage overwritten before its DMA completed')
        self.log.append(('pack', first, start, int(destination.size)))
        cuda._pack_radial_source_block(source, boxes, offsets, first, destination)


def _fixture(capacity, *, failure=None, empty=False):
    source = np.arange(5 * 7 * 11, dtype=np.uint16).astype(np.uint8).reshape(5, 7, 11)
    boxes = np.asarray(((1, 5, 2, 9), (0, 0, 0, 0), (2, 7, 0, 11),
                        (0, 3, 4, 10), (0, 0, 0, 0)), np.int64)
    if empty:
        boxes[:] = 0
    expected = np.concatenate([source[z, y0:y1, x0:x1].ravel()
                               for z, (y0, y1, x0, x1) in enumerate(boxes)])
    sizes = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    offsets = np.r_[np.uint64(0), np.cumsum(sizes, dtype=np.uint64)]
    transfer = _DelayedUpload(capacity, failure=failure)
    projector = object.__new__(cuda.RadialCudaProjector)
    projector._cp, projector._stream = transfer.cp, transfer.stream
    projector._events = {'existing_kernel_event': object()}
    projector._upload_stage = transfer.stage
    projector._upload_pin = SimpleNamespace(ptr=transfer.pin_pointer)
    projector.source_h2d_bytes = expected.size
    projector.source_pack_seconds = 0.
    projector.contract = SimpleNamespace(arrays={'bboxes': boxes, 'source_offsets': offsets})
    return projector, transfer, source, expected


class CroppedUploadPipelineTests(unittest.TestCase):
    def test_exact_crops_empty_shells_and_split_rows_with_fixed_pinned_budget(self):
        for capacity in (1, 2, 3, 7, 37, 100, 101, 4096):
            for enabled in ('0', '1'):
                with self.subTest(capacity=capacity, enabled=enabled):
                    p, transfer, source, expected = _fixture(capacity)
                    pin, stage = p._upload_pin, p._upload_stage
                    with mock.patch.object(cuda, '_pack_radial_source_block_compiled', transfer.pack), \
                            mock.patch.dict(os.environ, {'YOLO_TTA_CROPPED_UPLOAD_PIPELINE': enabled}):
                        p._upload_cropped_source(source)
                    np.testing.assert_array_equal(transfer.device[:expected.size], expected)
                    self.assertIs(p._upload_pin, pin)
                    self.assertIs(p._upload_stage, stage)
                    self.assertEqual(p.source_upload_stage_bytes, capacity)
                    pipelined = enabled == '1' and capacity >= 2 and expected.size > capacity
                    self.assertEqual(p.source_upload_pipeline, pipelined)
                    self.assertEqual(p.source_upload_copy_count, math.ceil(expected.size / (capacity // 2 if pipelined else capacity)))
                    self.assertEqual(p.source_upload_stream_fences, 1 if pipelined else p.source_upload_copy_count)
                    self.assertEqual(p.source_upload_lane_waits, max(0, p.source_upload_copy_count - 2) if pipelined else 0)
                    self.assertEqual(list(p._events), ['existing_kernel_event'])
                    if pipelined:
                        first_wait = next(i for i, event in enumerate(transfer.log) if event[0] == 'lane_wait')
                        self.assertEqual(sum(event[0] == 'copy' for event in transfer.log[:first_wait]), 2)

    def test_failures_keep_event_pinned_and_device_owners_for_constructor_cleanup(self):
        for failure in ('pack', 'enqueue', 'event_record', 'event_wait', 'stream'):
            p, transfer, source, _ = _fixture(7, failure=failure)
            pin, stage = p._upload_pin, p._upload_stage
            with self.subTest(failure=failure), \
                    mock.patch.object(cuda, '_pack_radial_source_block_compiled', transfer.pack), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_CROPPED_UPLOAD_PIPELINE': '1'}), \
                    self.assertRaises(RuntimeError):
                p._upload_cropped_source(source)
            self.assertIs(p._upload_pin, pin)
            self.assertIs(p._upload_stage, stage)
            self.assertIs(p._source_gpu.storage, transfer.device)
            self.assertIn('source_upload_lane_0', p._events)
            self.assertIn('source_upload_lane_1', p._events)

    def test_missing_event_fence_negative_control_overwrites_pending_dma(self):
        p, transfer, source, _ = _fixture(7, failure='unsafe_event_wait')
        with mock.patch.object(cuda, '_pack_radial_source_block_compiled', transfer.pack), \
                mock.patch.dict(os.environ, {'YOLO_TTA_CROPPED_UPLOAD_PIPELINE': '1'}), \
                self.assertRaisesRegex(AssertionError, 'overwritten before its DMA'):
            p._upload_cropped_source(source)

    def test_long_empty_shell_runs_keep_compiled_packing_offsets_exact(self):
        p, transfer, source, _ = _fixture(7)
        expanded = np.zeros((103, *source.shape[1:]), np.uint8)
        expanded[97:] = source[:1]
        boxes = np.zeros((103, 4), np.int64)
        boxes[98] = (1, 5, 2, 9)
        boxes[101] = (2, 7, 0, 11)
        sizes = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
        offsets = np.r_[np.uint64(0), np.cumsum(sizes, dtype=np.uint64)]
        p.contract.arrays = {'bboxes': boxes, 'source_offsets': offsets}
        expected = np.concatenate([expanded[z, y0:y1, x0:x1].ravel()
                                   for z, (y0, y1, x0, x1) in enumerate(boxes)])
        p.source_h2d_bytes = expected.size
        with mock.patch.object(cuda, '_pack_radial_source_block_compiled', transfer.pack), \
                mock.patch.dict(os.environ, {'YOLO_TTA_CROPPED_UPLOAD_PIPELINE': '1'}):
            p._upload_cropped_source(expanded)
        np.testing.assert_array_equal(transfer.device, expected)

    def test_event_unavailability_falls_back_before_any_partial_delivery(self):
        for failure in ('event_create', 'event_second_create'):
            p, transfer, source, expected = _fixture(7, failure=failure)
            with self.subTest(failure=failure), \
                    mock.patch.object(cuda, '_pack_radial_source_block_compiled', transfer.pack), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_CROPPED_UPLOAD_PIPELINE': '1'}):
                p._upload_cropped_source(source)
            self.assertFalse(p.source_upload_pipeline)
            self.assertEqual(p.source_upload_copy_count, math.ceil(expected.size / 7))
            np.testing.assert_array_equal(transfer.device, expected)

    def test_unsafe_constructor_close_quarantines_pinned_device_and_event_owners(self):
        from XTA.spherical_projection_cuda import SphericalCudaProjector, SphericalCudaProjectionUnsafeFailure
        for cls, error in ((cuda.RadialCudaProjector, cuda.RadialCudaProjectionUnsafeFailure),
                           (SphericalCudaProjector, SphericalCudaProjectionUnsafeFailure)):
            p, transfer, source, _ = _fixture(7, failure='stream')
            p._lock, p._closed, p.device_index = threading.RLock(), False, 0
            p._cp.cuda.Device = lambda _: contextlib.nullcontext()
            with self.subTest(projector=cls.__name__), \
                    mock.patch.object(cuda, '_pack_radial_source_block_compiled', transfer.pack), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_CROPPED_UPLOAD_PIPELINE': '1'}):
                with self.assertRaisesRegex(RuntimeError, 'failed stream fence'):
                    p._upload_cropped_source(source)
                owners = (p._upload_pin, p._upload_stage, p._source_gpu, dict(p._events))
                with self.assertRaises(error) as caught:
                    cls.close(p)
                self.assertIs(caught.exception.projector, p)
                self.assertFalse(p._closed)
                self.assertIs(p._upload_pin, owners[0])
                self.assertIs(p._upload_stage, owners[1])
                self.assertIs(p._source_gpu, owners[2])
                self.assertEqual(p._events, owners[3])

    def test_optional_packer_fallback_and_empty_payload_preserve_existing_routes(self):
        for packer in (None, mock.Mock(side_effect=RuntimeError('compiler unavailable'))):
            p, transfer, source, expected = _fixture(7)
            with mock.patch.object(cuda, '_pack_radial_source_block_compiled', packer):
                p._upload_cropped_source(source)
            self.assertFalse(p.source_upload_pipeline)
            self.assertEqual(p.source_pack_backend, 'numpy')
            np.testing.assert_array_equal(transfer.device, expected)
        p, transfer, source, expected = _fixture(7, empty=True)
        with mock.patch.object(cuda, '_pack_radial_source_block_compiled', transfer.pack):
            p._upload_cropped_source(source)
        self.assertFalse(p.source_upload_pipeline)
        self.assertEqual(p.source_upload_copy_count, 0)
        self.assertEqual(transfer.device.size, 1)


if __name__ == '__main__':
    unittest.main()
