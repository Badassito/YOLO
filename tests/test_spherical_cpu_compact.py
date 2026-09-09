"""Exact compact CPU projection, bounded indexing and transactional publication."""
from concurrent.futures import CancelledError
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import spherical_projection as sp
from XTA import spherical_projection_cpu as compiled
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from XTA.spherical_projection_bounds import SphericalOutputBounds


def decode(block, shape):
    result = np.zeros((len(block.records), *shape[1:]), np.uint8)
    cursor = 0
    for index, record in enumerate(block.records):
        assert record.z == block.first_z + index
        assert record.offset == cursor
        cursor += record.size
        if not record.foreground:
            assert (record.y0, record.y1, record.x0, record.x1, record.size) == (0, 0, 0, 0, 0)
            continue
        h, w = record.y1 - record.y0, record.x1 - record.x0
        payload = block.payload[record.offset:cursor]
        if block.packed:
            data = np.unpackbits(payload.reshape(h, (w + 7) // 8), axis=1, bitorder='little')
            assert not data[:, w:].any()
            crop = data[:, :w]
        else:
            crop = payload.reshape(h, w)
            assert np.all(crop <= 1)
        assert np.count_nonzero(crop) == record.foreground
        assert all(np.any(edge) for edge in (crop[0], crop[-1], crop[:, 0], crop[:, -1]))
        result[index, record.y0:record.y1, record.x0:record.x1] = crop
    assert cursor == block.payload.size
    return result


class EncodedSink:
    def __init__(self, shape, packed=False, on_publish=None):
        self.shape = shape
        self.encoded_slice_format = 'packbits_little' if packed else 'raw_u8'
        self.result = np.zeros(shape, np.uint8)
        self.seen = []
        self.kinds = []
        self.on_publish = on_publish
        self.abort = mock.Mock()

    def _accept(self, first, count, kind):
        assert first == len(self.seen)
        if self.on_publish:
            self.on_publish(first, count)
        self.seen.extend(range(first, first + count))
        self.kinds.append(kind)

    def __call__(self, first, block):
        self._accept(first, len(block), 'dense')
        self.result[first:first + len(block)] = block

    def consume_empty_range(self, first, count):
        self._accept(first, count, 'empty')

    def consume_encoded_block(self, first, records, payload, *, packed):
        self._accept(first, len(records), 'encoded')
        block = sp.RadialEncodedBlock(first, tuple(records), payload, packed)
        self.result[first:first + len(records)] = decode(block, self.shape)


class SphericalCpuCompactTests(unittest.TestCase):
    def setUp(self):
        self.shape = (17, 19, 21)
        self.views = build_spherical_view_infos(*self.shape, targets=('transverse',),
            min_radius=.5, patch_size=15, tilted_views=())
        self.view = self.views[0]
        self.source = np.ones((self.view.num_slices, 7, 11), np.uint8)
        self.radii, self.rotation, _, _ = sp._validate_spherical_projection(
            self.source, self.view, self.shape, None)

    def test_numpy_raw_and_packbits_match_unbounded_oracle_across_geometry(self):
        self._check_geometry(None)

    @unittest.skipUnless(hasattr(compiled._compiled_pull_spherical_f64, 'signatures'), 'Numba is optional')
    def test_rectangular_compiled_raw_and_packbits_match_unbounded_oracle(self):
        compiled.prepare_spherical_chunk_numba(self.source, self.view, self.radii, self.rotation, self.shape)
        self._check_geometry(compiled.pull_spherical_chunk_numba)

    def _check_geometry(self, pull):
        rng = np.random.default_rng(7701)
        cases = 0
        for rotation in (None, cube_rotation('vertical', 31), cube_rotation('horizontal', -23)):
            for initial in self.views:
                view = replace(initial, spherical_rotation_xyz=rotation) if rotation else initial
                source = rng.integers(0, 4, (view.num_slices, 7, 11), dtype=np.uint8)
                source[:, 3, :] = 1
                source[:, :, 5] = 1
                source.flags.writeable = False
                boxes = np.tile((0, 7, 0, 11), (view.num_slices, 1)).astype(np.int64)
                boxes[::3] = 0
                for shape in (self.shape, (13, 23, 17)):
                    args = (source, view, np.asarray(view.spherical_radii),
                            np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape)
                    expected = sp._project_spherical_block(*args, 0, shape[0], boxes)
                    for packed in (False, True):
                        actual = sp._project_spherical_encoded_block(*args, 0, shape[0], boxes,
                                                                    cpu_pull=pull, packed=packed)
                        with self.subTest(face=view.spherical_face, rotation=rotation, shape=shape, packed=packed):
                            np.testing.assert_array_equal(decode(actual, shape), expected)
                            self.assertLessEqual(actual.scan_voxels, actual.pull_voxels)
                            self.assertLessEqual(actual.pull_voxels, np.prod(shape))
                        cases += 1
        self.assertGreaterEqual(cases, 72)

    def test_empty_bounds_never_allocate_dense_or_read_and_preserve_all_z_records(self):
        boxes = np.zeros((self.view.num_slices, 4), np.int64)
        original_empty = np.empty
        def bounded_empty(shape, *args, **kwargs):
            self.assertEqual(shape, 0)
            return original_empty(shape, *args, **kwargs)
        with (mock.patch.object(sp.np, 'empty', side_effect=bounded_empty),
              mock.patch.object(sp, '_pull_spherical_chunk', side_effect=AssertionError('read'))):
            block = sp._project_spherical_encoded_block(self.source, self.view, self.radii,
                self.rotation, self.shape, 3, 9, boxes)
        self.assertEqual([record.z for record in block.records], list(range(3, 12)))
        self.assertEqual((block.payload.size, block.pull_voxels, block.scan_voxels), (0, 0, 0))
        self.assertFalse(decode(block, self.shape).any())

    def test_pre_cancelled_compact_block_never_allocates_or_reads(self):
        cancel = threading.Event()
        cancel.set()
        with (mock.patch.object(sp.np, 'empty', side_effect=AssertionError('allocated')),
              mock.patch.object(sp, '_pull_spherical_chunk', side_effect=AssertionError('read')),
              self.assertRaises(CancelledError)):
            sp._project_spherical_encoded_block(self.source, self.view, self.radii,
                self.rotation, self.shape, 0, 1, cancel_event=cancel)

    def test_compact_depth_caps_wire_metadata_and_accounts_for_transient_storage(self):
        depth, workers = sp._spherical_block_schedule(10_000_000, 1, 32, compact=True)
        self.assertEqual(depth, sp._MAX_ENCODED_SLICES)
        self.assertGreaterEqual(workers, 1)
        with mock.patch.object(sp, '_INFLIGHT_WORK_BYTES', 1024):
            self.assertEqual(sp._spherical_block_schedule(10_000_000, 1, 32, compact=True)[1], 1)

    def test_compiled_schedule_preserves_legacy_concurrency_and_caps_large_workspaces(self):
        with mock.patch.object(sp, '_cpu_count', return_value=64):
            self.assertEqual(sp._spherical_block_schedule(1931, 3064 * 3022, 64), (1, 4))
            self.assertEqual(sp._spherical_block_schedule(1931, 3064 * 3022, 64,
                                                         compact=True, compiled=True), (1, 4))
            for plane_bytes in (1, 1000, 3072**2, 25_000_000, 70_000_000, 300_000_000):
                for compiled_mode in (False, True):
                    depth, workers = sp._spherical_block_schedule(10_000, plane_bytes, 64,
                        compact=True, compiled=compiled_mode)
                    legacy_workers = sp._spherical_block_schedule(10_000, plane_bytes, 64)[1]
                    self.assertLessEqual(workers, legacy_workers)
                    scratch = min(plane_bytes, sp._PULL_CHUNK_VOXELS) * (
                        sp._COMPILED_CHUNK_BYTES_PER_VOXEL if compiled_mode else sp._CHUNK_BYTES_PER_VOXEL)
                    memory = plane_bytes * (2 * depth + 1) + scratch + depth * sp._CPU_ENCODED_SLICE_BYTES
                    # One oversized block must still progress; no second worker
                    # may be admitted beyond the configured in-flight budget.
                    if workers > 1:
                        self.assertLessEqual(memory * workers, sp._INFLIGHT_WORK_BYTES)

    def test_raw_crops_normalize_bool_and_255_source_to_binary(self):
        for source in (self.source.astype(bool), self.source * np.uint8(255)):
            block = sp._project_spherical_encoded_block(source, self.view, self.radii,
                self.rotation, self.shape, 0, self.shape[0])
            self.assertTrue(block.payload.size)
            self.assertEqual(set(np.unique(block.payload)), {0, 1})
            expected = sp._project_spherical_block(source, self.view, self.radii, self.rotation,
                                                  self.shape, 0, self.shape[0])
            np.testing.assert_array_equal(decode(block, self.shape), expected)

    def test_malformed_empty_metadata_aborts_before_empty_publication(self):
        for field, value in (('z', 9), ('y1', 1), ('x0', 1), ('foreground', 1), ('offset', 1),
                             ('size', 1), ('z', 0.0), ('offset', 0.0)):
            sink = EncodedSink(self.shape)
            records = [sp.RadialEncodedSlice(z, 0, 0, 0, 0, 0, 0, 0) for z in range(self.shape[0])]
            records[0] = replace(records[0], **{field: value})
            block = sp.SphericalCpuEncodedBlock(0, tuple(records), np.empty(0, np.uint8))
            with (self.subTest(field=field),
                  mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=None),
                  mock.patch.object(sp, '_select_spherical_cpu_pull', return_value=None),
                  mock.patch.object(sp, '_project_spherical_encoded_block', return_value=block),
                  redirect_stdout(io.StringIO()),
                  self.assertRaisesRegex(RuntimeError, 'empty block metadata')):
                sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'), 'invalid empty',
                    sink_only=True, projection_block_callback=sink)
            sink.abort.assert_called_once()
            self.assertFalse(sink.seen)

    def test_producer_failure_after_publication_aborts_without_replay(self):
        original = sp._project_spherical_encoded_block
        failure = ValueError('failed second block')
        def project(*args, **kwargs):
            if args[5]:
                raise failure
            return original(*args, **kwargs)
        sink = EncodedSink(self.shape)
        with (mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=None),
              mock.patch.object(sp, 'spherical_cuda_backproject_enabled', return_value=False),
              mock.patch.object(sp, '_select_spherical_cpu_pull', return_value=None),
              mock.patch.object(sp, '_spherical_block_schedule', return_value=(1, 1)),
              mock.patch.object(sp, '_project_spherical_encoded_block', side_effect=project),
              redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as caught):
            sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'), 'producer failure',
                sink_only=True, projection_block_callback=sink)
        self.assertIs(caught.exception, failure)
        sink.abort.assert_called_once_with(failure)
        self.assertEqual(sink.seen, [0])

    @unittest.skipUnless(hasattr(compiled._compiled_pull_spherical_f64, 'signatures'), 'Numba is optional')
    def test_rectangle_global_origins_row_boundaries_and_invalid_indices(self):
        view = self.views[-1]
        radii = np.asarray(view.spherical_radii)
        rotation = np.asarray(view.spherical_rotation_xyz).reshape(3, 3)
        source = np.zeros((view.num_slices, 7, 11), np.uint8)
        source[:, 2:5, 5] = 1
        source[:, 3, 3:9] = 1
        shape, z, bounds = (19, 23, 29), 5, (2, 18, 4, 25)
        expected = sp._project_spherical_block(source, view, radii, rotation, shape, z, 1)[0, 2:18, 4:25].ravel()
        args = (source, view, radii, rotation, shape, z)
        pieces = [compiled.pull_spherical_rectangle_numba(*args, first, min(first + 37, len(expected)),
                  bounds_yx=bounds) for first in range(0, len(expected), 37)]
        np.testing.assert_array_equal(np.concatenate(pieces), expected)
        for bad in ((-1, 18, 4, 25), (2, 24, 4, 25), (2, 18, 25, 25), (2, 18, 4, 30)):
            with self.subTest(bounds=bad), self.assertRaisesRegex(ValueError, 'rectangle'):
                compiled.pull_spherical_rectangle_numba(*args, 0, 1, bounds_yx=bad)
        with self.assertRaisesRegex(ValueError, 'rectangle'):
            compiled.pull_spherical_rectangle_numba(*args, 0, len(expected) + 1, bounds_yx=bounds)

    def test_rectangular_chunks_cancel_before_more_than_one_chunk_and_skip_x(self):
        cancel = threading.Event()
        visits = []
        def rectangle(*args, **kwargs):
            visits.append((args[6], args[7], kwargs['bounds_yx']))
            cancel.set()
            return np.ones(args[7] - args[6], np.uint8)
        def forbidden(*args):
            raise AssertionError('full width pull')
        forbidden.rectangle = rectangle
        bounds = SphericalOutputBounds(4, 5, 3, 13, 7, 11)
        with mock.patch.object(sp, '_PULL_CHUNK_VOXELS', 17), self.assertRaises(CancelledError):
            sp._project_spherical_encoded_block(self.source, self.view, self.radii, self.rotation,
                self.shape, 4, 1, output_bounds=bounds, cpu_pull=forbidden, cancel_event=cancel)
        self.assertEqual(visits, [(0, 17, (3, 13, 7, 11))])

    def test_real_dispatch_empty_protocol_off_switch_and_generic_compatibility(self):
        expected = sp._project_spherical_block(self.source, self.view, self.radii, self.rotation,
                                                self.shape, 0, self.shape[0])
        for compact in ('0', '1'):
            sink = EncodedSink(self.shape, packed=True)
            with (mock.patch.dict(os.environ, {'YOLO_TTA_CPU_SPHERICAL_COMPACT': compact,
                                               'YOLO_TTA_CPU_SPHERICAL_COMPILED': '0'}),
                  mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=None),
                  mock.patch.object(sp, 'spherical_cuda_backproject_enabled', return_value=False),
                  mock.patch.object(sp, '_spherical_block_schedule', return_value=(2, 2)),
                  redirect_stdout(io.StringIO())):
                sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'), 'compact',
                    workers=2, sink_only=True, projection_block_callback=sink)
            np.testing.assert_array_equal(sink.result, expected)
            self.assertEqual(sink.seen, list(range(self.shape[0])))
            if compact == '1':
                self.assertNotIn('dense', sink.kinds)
                self.assertIn('encoded', sink.kinds)
            else:
                self.assertEqual(set(sink.kinds), {'dense'})
        empty = EncodedSink(self.shape)
        with (mock.patch.object(sp, '_try_spherical_cuda_stage', return_value=None),
              mock.patch.object(sp, 'spherical_cuda_backproject_enabled', return_value=False),
              mock.patch.object(sp, '_select_spherical_cpu_pull', return_value=None),
              redirect_stdout(io.StringIO())):
            sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'), 'empty',
                sink_only=True, projection_block_callback=empty,
                known_slice_bboxes=np.zeros((self.view.num_slices, 4), np.int64))
        self.assertEqual(set(empty.kinds), {'empty'})
        self.assertEqual(empty.seen, list(range(self.shape[0])))

    def test_encoded_failure_aborts_once_and_joins_readers_without_replay(self):
        self._check_handoff(True)

    def test_encoded_cpu_to_gpu_promotion_is_exactly_once_and_joins_readers(self):
        self._check_handoff(False)

    def _check_handoff(self, fail_sink):
        shape = self.shape
        original = sp._project_spherical_encoded_block
        original_pull = sp._pull_spherical_chunk
        bounds = SphericalOutputBounds(0, shape[0], 0, shape[1], 0, shape[2])
        expected = sp._project_spherical_block(self.source, self.view, self.radii, self.rotation,
                                              shape, 0, shape[0])
        lock, ready = threading.Lock(), threading.Event()
        started, events, calls = set(), [], {1: [], 2: []}
        live = 0
        failure = ValueError('partial encoded publication failed')
        def project(*args, **kwargs):
            nonlocal live
            with lock:
                live += 1
                events.append(kwargs['cancel_event'])
            try:
                return original(*args, **kwargs)
            finally:
                with lock:
                    live -= 1
        def pull(*args, **kwargs):
            z, first = args[5:7]
            if z in calls:
                with lock:
                    calls[z].append(first)
                    started.add(z)
                    if len(started) == 2:
                        ready.set()
                    cancel = events[0]
                if not cancel.wait(5):
                    raise AssertionError('reader was not cancelled')
            return original_pull(*args, **kwargs)
        def consume(first, count):
            if first == 0:
                self.assertTrue(ready.wait(5))
                if fail_sink:
                    raise failure
        sink = EncodedSink(shape, packed=True, on_publish=consume)
        def gpu(first, count, packed):
            self.assertEqual(live, 0)
            return original(self.source, self.view, self.radii, self.rotation, shape, first, count,
                            output_bounds=bounds, cpu_pull=original_pull, packed=packed)
        stage = SimpleNamespace(device_index=0, max_block_depth=2, projector=SimpleNamespace(),
            project_encoded=mock.Mock(side_effect=gpu), close=mock.Mock())
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(mock.patch.object(sp, '_try_spherical_cuda_stage', side_effect=[None, stage]))
            stack.enter_context(mock.patch.object(sp, 'spherical_cuda_backproject_enabled', return_value=True))
            stack.enter_context(mock.patch.object(sp, '_select_spherical_cpu_pull', return_value=None))
            stack.enter_context(mock.patch.object(sp, 'spherical_output_bounds', return_value=bounds))
            stack.enter_context(mock.patch.object(sp, '_spherical_block_schedule', return_value=(1, 3)))
            stack.enter_context(mock.patch.object(sp, '_PULL_CHUNK_VOXELS', 11))
            stack.enter_context(mock.patch.object(sp, '_CUDA_RECHECK_SLICES', 1))
            stack.enter_context(mock.patch.object(sp, '_project_spherical_encoded_block', side_effect=project))
            stack.enter_context(mock.patch.object(sp, '_pull_spherical_chunk', side_effect=pull))
            if fail_sink:
                with self.assertRaises(ValueError) as caught:
                    sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'),
                        'failure', workers=3, sink_only=True, projection_block_callback=sink)
                self.assertIs(caught.exception, failure)
            else:
                sp.backproject_spherical_volume_to_volume(self.source, self.view, Path('unused'),
                    'promotion', workers=3, sink_only=True, projection_block_callback=sink)
        self.assertEqual(live, 0)
        self.assertEqual(calls, {1: [0], 2: [0]})
        if fail_sink:
            sink.abort.assert_called_once_with(failure)
            stage.project_encoded.assert_not_called()
            self.assertFalse(sink.seen)
        else:
            sink.abort.assert_not_called()
            self.assertEqual(sink.seen, list(range(shape[0])))
            np.testing.assert_array_equal(sink.result, expected)
            stage.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
