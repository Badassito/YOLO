"""CPU contracts and explicitly enabled CUDA pull-projection parity."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import os
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import cylindrical_projection as reference
from XTA import cylindrical_cuda_projection as cuda
from XTA.cylindrical_geometry import build_radial_view_infos
from XTA.geometry import ViewInfo, radial_global_radii


def case(base='transverse', shape=(5, 7, 9), size=4, processing=None, seed=4):
    views = build_radial_view_infos(*shape, targets=(base,), min_radius=.7,
                                    patch_size=size, tilted_views=())
    view = views[-1]
    source = np.random.default_rng(seed).integers(0, 2,
        (view.num_slices, *(processing or (size, size))), dtype=np.uint8)
    source *= np.uint8(255)
    return source, view


def contract(source, view, shape):
    radii = np.asarray(radial_global_radii(view), dtype=np.float64)
    plan = reference._build_radial_plane_plan(view, radii, shape)
    metadata = reference._radial_projection_metadata(view, source.shape, shape, plan)
    boxes = np.zeros((source.shape[0], 4), dtype=np.int64)
    for shell, plane in enumerate(source):
        yy, xx = np.nonzero(plane)
        if len(yy):
            boxes[shell] = (yy.min(), yy.max() + 1, xx.min(), xx.max() + 1)
    return plan, metadata, boxes


def numpy_oracle(source, view, shape):
    radii = np.asarray(radial_global_radii(view), dtype=np.float64)
    return np.stack([reference._pull_radial_chunk(source, view, radii, shape, z, 0,
        shape[1] * shape[2]).reshape(shape[1:]) for z in range(shape[0])])


def assert_encoded(test_case, encoded, expected):
    restored = np.zeros_like(expected)
    test_case.assertTrue(encoded.payload.flags.owndata)
    test_case.assertTrue(encoded.payload.flags.c_contiguous)
    test_case.assertFalse(encoded.payload.flags.writeable)
    cursor = 0
    for local, record in enumerate(encoded.records):
        test_case.assertEqual(record.z, encoded.first_z + local)
        test_case.assertEqual(record.offset, cursor)
        test_case.assertEqual(record.foreground, int(np.count_nonzero(expected[local])))
        yy, xx = np.nonzero(expected[local])
        bounds = (int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1) if len(yy) else (0, 0, 0, 0)
        test_case.assertEqual((record.y0, record.y1, record.x0, record.x1), bounds)
        if record.foreground:
            height, width = record.y1 - record.y0, record.x1 - record.x0
            row_bytes = (width + 7) // 8 if encoded.packed else width
            test_case.assertEqual(record.size, height * row_bytes)
            data = encoded.payload[cursor:cursor + record.size].reshape(height, row_bytes)
            if encoded.packed:
                if width % 8:
                    test_case.assertFalse(np.any(data[:, -1] & np.uint8(255 ^ ((1 << (width % 8)) - 1))))
                data = np.unpackbits(data, axis=1, count=width, bitorder='little')
            else:
                test_case.assertTrue(np.all(data <= 1))
            restored[local, record.y0:record.y1, record.x0:record.x1] = data
        else:
            test_case.assertEqual(record.size, 0)
        cursor += record.size
    test_case.assertEqual(cursor, encoded.payload.size)
    np.testing.assert_array_equal(restored, expected)


class RadialCudaProjectionContractTests(unittest.TestCase):
    def test_encoded_metadata_layout_prefix_empties_and_fail_closed_bounds(self):
        self.assertEqual(cuda._CROP_METADATA_DTYPE.itemsize, 24)
        self.assertEqual(cuda._CROP_METADATA_DTYPE.fields['foreground'][1], 16)
        metadata = np.zeros(3, cuda._CROP_METADATA_DTYPE)
        metadata[0] = (2, 5, 1, 10, 8)
        metadata[1] = (99, 0, 99, 0, 0)  # CUDA's unoccupied bbox sentinel
        metadata[2] = (0, 1, 0, 1, 1)
        records, offsets, total, largest = cuda._encoded_records(5, metadata, True, (7, 11), 64)
        self.assertEqual(offsets.tolist(), [0, 6, 6, 7])
        self.assertEqual((total, largest), (7, 6))
        self.assertEqual(records[1], cuda.RadialEncodedSlice(6, 0, 0, 0, 0, 0, 6, 0))
        with self.assertRaises(FrozenInstanceError):
            records[0].size = 123
        with self.assertRaisesRegex(RuntimeError, 'output buffer'):
            cuda._encoded_records(0, metadata, False, (7, 11), 20)
        metadata[0]['foreground'] = 28
        with self.assertRaisesRegex(RuntimeError, 'foreground count'):
            cuda._encoded_records(0, metadata, False, (7, 11), 64)

    def setUp(self):
        self.source, self.view = case()
        self.shape = (5, 7, 9)
        self.plan, self.metadata, self.boxes = contract(self.source, self.view, self.shape)

    def validate(self, **changes):
        values = dict(source=self.source, plan=self.plan, metadata=self.metadata,
                      view=self.view, output_shape=self.shape, bboxes=self.boxes,
                      use_bboxes=True, block_bytes=128)
        values.update(changes)
        return cuda._validate_projection_contract(**values)

    def test_output_budget_and_addresses_validate_without_cuda_import(self):
        actual = self.validate()
        self.assertEqual(actual.max_block_depth, 2)
        self.assertEqual(actual.block_bytes, 126)
        self.assertEqual(actual.geometry_bytes, sum(a.nbytes for a in actual.arrays.values()))
        with self.assertRaises(cuda.RadialCudaProjectionUnavailable):
            self.validate(block_bytes=62)
        with self.assertRaisesRegex(cuda.RadialCudaProjectionUnavailable, 'grid limit'):
            self.validate(output_shape=(5, 524281, 1), block_bytes=64 * 1024**2)

    def test_unsupported_source_and_invalid_gathers_fail_before_cuda(self):
        for source in (self.source.astype(np.float32), self.source[:, :, ::-1]):
            with self.assertRaises(cuda.RadialCudaProjectionUnavailable):
                self.validate(source=source)
        bad = self.plan.shell_index.copy()
        bad[0] = self.source.shape[0]
        with self.assertRaisesRegex(ValueError, 'gather addresses'):
            self.validate(plan=replace(self.plan, shell_index=bad))
        offsets = self.plan.column_offsets.copy()
        offsets[-1] += 1
        with self.assertRaisesRegex(ValueError, 'gather addresses'):
            self.validate(plan=replace(self.plan, column_offsets=offsets))
        boxes = self.boxes.copy()
        boxes[0, 1] = self.source.shape[1] + 1
        with self.assertRaisesRegex(ValueError, 'bounding boxes'):
            self.validate(bboxes=boxes)

    def test_unsafe_fence_keeps_owners_and_bypasses_exception_fallback(self):
        projector = object.__new__(cuda.RadialCudaProjector)
        projector._lock = threading.RLock()
        projector._closed = False
        projector.device_index = 0
        projector._cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: mock.MagicMock()))
        projector._stream = SimpleNamespace(synchronize=mock.Mock(side_effect=RuntimeError('failed fence')))
        owner = projector._source_gpu = object()
        with self.assertRaises(cuda.RadialCudaProjectionUnsafeFailure) as caught:
            projector.close()
        self.assertIs(caught.exception.projector, projector)
        self.assertIs(projector._source_gpu, owner)
        self.assertFalse(projector._closed)
        self.assertNotIsInstance(caught.exception, Exception)


@unittest.skipUnless(os.environ.get('XTA_RUN_CUDA_RADIAL_PROJECTION') == '1', 'explicit CUDA projection qualification only')
class RadialCudaProjectionParityTests(unittest.TestCase):
    def test_encoded_raw_packed_metadata_and_owned_threaded_payloads(self):
        for base in ('transverse', 'sagittal', 'coronal'):
            source, initial = case(base, processing=(3, 2))
            for direction in ('', 'vertical', 'horizontal'):
                view = replace(initial, radial_tilted_source=bool(direction),
                               tilt_direction=direction, tilt_angle_deg=23.0)
                shape = (4, 6, 8)
                plan, metadata, boxes = contract(source, view, shape)
                expected = numpy_oracle(source, view, shape)
                with cuda.RadialCudaProjector(source, plan, metadata, view, shape, boxes, True, 0,
                        block_bytes=2 * shape[1] * shape[2], reserve_bytes=0) as projector:
                    returned = []
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        for packed in (False, True):
                            for z in range(0, shape[0], 2):
                                block = executor.submit(projector.project_encoded, z, 2, packed).result()
                                assert_encoded(self, block, expected[z:z + 2])
                                returned.append(block)
                    self.assertTrue(all(not np.shares_memory(a.payload, b.payload)
                        for i, a in enumerate(returned) for b in returned[i + 1:]))
                    self.assertEqual(projector.dense_d2h_bytes, 0)
                    self.assertEqual(projector.payload_d2h_bytes, sum(b.payload.nbytes for b in returned))
                    self.assertEqual(projector.metadata_d2h_bytes, 2 * shape[0] * 24)
                    self.assertTrue(all(getattr(projector, name) >= 0 for name in
                        ('kernel_seconds', 'metadata_seconds', 'pack_seconds', 'd2h_seconds')))

    def test_encoded_postkernels_tight_edges_partial_warps_and_tail_bits(self):
        source, view = case(shape=(5, 17, 35))
        shape = (5, 17, 35)
        plan, metadata, boxes = contract(source, view, shape)
        with cuda.RadialCudaProjector(source, plan, metadata, view, shape, boxes, False, 0,
                block_bytes=2 * shape[1] * shape[2], reserve_bytes=0) as projector:
            for width in (1, 7, 8, 9, 15, 17, 35):
                dense = np.zeros((2, *shape[1:]), np.uint8)
                dense[0, 0, 0] = 1
                dense[0, -1, width - 1] = 1
                dense[0, 4:10, :width] = 1
                with projector._cp.cuda.Device(0), projector._stream:
                    projector._output_gpu[:2].set(dense, stream=projector._stream)
                    for packed in (False, True):
                        encoded = projector._encode_current_output(0, 2, packed)
                        assert_encoded(self, encoded, dense)
            self.assertEqual(projector.dense_d2h_bytes, 0)

    def test_axes_tilts_processing_maps_bboxes_threads_and_owned_blocks(self):
        checked = 0
        for base in ('transverse', 'sagittal', 'coronal'):
            for source_shape in ((5, 7, 9), (11, 7, 9)):
                for processing in (None, (3, 2)):
                    source, initial = case(base, source_shape, processing=processing)
                    untouched = source.copy()
                    for direction in ('', 'vertical', 'horizontal'):
                        view = replace(initial, radial_tilted_source=bool(direction),
                                       tilt_direction=direction, tilt_angle_deg=-23.0)
                        shape = tuple(max(1, v - 1) for v in source_shape)
                        plan, metadata, boxes = contract(source, view, shape)
                        expected = numpy_oracle(source, view, shape)
                        for use_boxes in (False, True):
                            with self.subTest(base=base, shape=source_shape, processing=processing,
                                              direction=direction, bboxes=use_boxes):
                                projector = cuda.RadialCudaProjector(source, plan, metadata, view,
                                    shape, boxes, use_boxes, 0, block_bytes=2 * shape[1] * shape[2],
                                    upload_bytes=37, reserve_bytes=0)
                                pool = projector._pool
                                try:
                                    with ThreadPoolExecutor(max_workers=1) as executor:
                                        blocks = [executor.submit(projector.project, z,
                                            min(projector.max_block_depth, shape[0] - z)).result()
                                            for z in range(0, shape[0], projector.max_block_depth)]
                                    self.assertTrue(all(b.flags.owndata for b in blocks))
                                    self.assertTrue(all(not np.shares_memory(a, b)
                                        for i, a in enumerate(blocks) for b in blocks[i + 1:]))
                                    actual = np.concatenate(blocks)
                                    np.testing.assert_array_equal(actual, expected)
                                    checked += actual.size
                                    with self.assertRaises(ValueError):
                                        projector.project(0, projector.max_block_depth + 1)
                                finally:
                                    projector.close()
                                self.assertEqual(pool.used_bytes(), 0)
                                self.assertEqual(pool.total_bytes(), 0)
                                projector.close()
                                with self.assertRaises(RuntimeError):
                                    projector.project(0, 1)
                    np.testing.assert_array_equal(source, untouched)
        print(f'Radial CUDA pull exact parity: {checked} voxels checked across axes/tilts/processing maps.', flush=True)

    def test_tiny_periods_repeated_wraps_and_empty_masks(self):
        for minimum in (.1, 1.0):
            view = build_radial_view_infos(3, 7, 7, targets=('transverse',), min_radius=minimum,
                                            patch_size=12, tilted_views=())[0]
            source = np.zeros((view.num_slices, 12, 12), np.uint8)
            source[:, :3, -1] = 255
            source[0, 1, 6] = 1
            plan, metadata, boxes = contract(source, view, (3, 7, 7))
            for data in (source, np.zeros_like(source)):
                with cuda.RadialCudaProjector(data, plan, metadata, view, (3, 7, 7),
                                               boxes, False, 0, reserve_bytes=0) as projector:
                    np.testing.assert_array_equal(projector.project(0, 3), numpy_oracle(data, view, (3, 7, 7)))

    def test_double_half_even_ties_and_adjacent_values(self):
        source = np.asarray([255, 0, 1, 255], dtype=np.uint8).reshape(1, 4, 1)
        view = ViewInfo('ties', 1, 4, 1, 'pad', family='radial', radial_base_view='transverse',
                        full_t=4, full_h=1, full_w=1)
        plan = reference.RadialPlanePlan(0, (1, 1), np.asarray([0], np.int32),
            np.asarray([0, 1], np.uint32), np.asarray([0], np.int32))
        centers = np.asarray([-.001, 0., np.nextafter(.5, 0), .5, np.nextafter(.5, 1),
                              1.5, 2.5, 3., np.nextafter(3., 4)], np.float64)
        metadata = (centers, np.zeros(1, np.float64), np.zeros((1, 1), np.float64),
                    np.arange(4, dtype=np.int32), np.zeros(1, np.int32), 4, True)
        with cuda.RadialCudaProjector(source, plan, metadata, view, (9, 1, 1),
                                       None, False, 0, reserve_bytes=0) as projector:
            np.testing.assert_array_equal(projector.project(0, 9).reshape(-1), [0, 1, 1, 1, 0, 1, 1, 1, 0])


if __name__ == '__main__':
    unittest.main()
