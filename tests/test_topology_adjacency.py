"""Adjacent-slice pair contracts, independent oracle, spills, and ownership."""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import unittest
from unittest import mock

import numpy as np
from XTA import topology

HAS_COMPILED_ADJACENCY = getattr(topology._numba_adjacency_scan_kernel, 'signatures', None) is not None

def oracle(previous, current, offsets=None, previous_offset=0, current_offset=0):
    """Coordinate enumeration independent of NumPy slicing and the candidate hash."""
    offsets = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) if offsets is None else tuple(offsets)
    height, width = previous.shape
    codes = set()
    for y in range(height):
        for x in range(width):
            a = int(previous[y, x])
            if a <= 0:
                continue
            for dy, dx in offsets:
                cy, cx = y + int(dy), x + int(dx)
                if not (0 <= cy < height and 0 <= cx < width):
                    continue
                b = int(current[cy, cx])
                if b <= 0:
                    continue
                aa, bb = a + previous_offset, b + current_offset
                assert 0 < aa <= 0xFFFFFFFF and 0 < bb <= 0xFFFFFFFF
                codes.add((aa << 32) | bb)
    return np.array(sorted(codes), dtype=np.uint64)


class TopologyAdjacencyTests(unittest.TestCase):
    def assert_pairs(self, previous, current, offsets=None, previous_offset=0, current_offset=0,
                     *, initial_capacity=None, max_capacity=None):
        before_previous, before_current = previous.copy(), current.copy()
        expected = oracle(previous, current, offsets, previous_offset, current_offset)
        reference = topology._adjacent_gid_pair_codes_numpy(
            previous, current, offsets, previous_offset, current_offset,
        )
        np.testing.assert_array_equal(reference, expected)
        kwargs = dict(xy_offsets=offsets, prev_offset=previous_offset, curr_offset=current_offset)
        if initial_capacity is None:
            actual = topology._adjacent_gid_pair_codes(previous, current, **kwargs)
        else:
            actual = topology._compiled_adjacent_gid_pair_codes(previous, current, initial_capacity=initial_capacity,
                                                   max_capacity=max_capacity, **kwargs)
        self.assertEqual(actual.dtype, np.dtype(np.uint64))
        self.assertEqual(actual.ndim, 1)
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(previous, before_previous)
        np.testing.assert_array_equal(current, before_current)
        return actual

    def test_each_neighbor_direction_and_6_18_26_connectivity(self):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                previous = np.zeros((5, 7), np.uint16)
                current = np.zeros_like(previous)
                previous[2, 3] = 11
                current[2 + dy, 3 + dx] = 19
                for connectivity in (6, 18, 26):
                    with self.subTest(dy=dy, dx=dx, connectivity=connectivity):
                        offsets = topology._adjacent_xy_offsets_for_3d_connectivity(connectivity)
                        self.assert_pairs(previous, current, offsets, 65500, 190000)

    def test_empty_narrow_and_asymmetric_offset_boundaries(self):
        rng = np.random.default_rng(1051)
        for shape in ((0, 0), (0, 7), (5, 0), (1, 1), (1, 13), (11, 1), (2, 3)):
            for offsets in (None, (), ((1, -1),), ((-2, 3), (2, -3), (0, 0)),
                            ((0, 0), (0, 0), (100, -100), (-100, 100))):
                with self.subTest(shape=shape, offsets=offsets):
                    previous = rng.integers(0, 8, shape, dtype=np.uint16)
                    current = rng.integers(0, 9, shape, dtype=np.uint16)
                    self.assert_pairs(previous, current, offsets)

    def test_random_directional_offsets_against_independent_oracle(self):
        rng = np.random.default_rng(5029)
        for case in range(80):
            shape = (int(rng.integers(1, 19)), int(rng.integers(1, 23)))
            previous = rng.integers(0, 34, shape, dtype=np.uint16)
            current = rng.integers(0, 49, shape, dtype=np.uint16)
            previous[rng.random(shape) < 0.75] = 0
            current[rng.random(shape) < 0.65] = 0
            offsets = tuple((int(rng.integers(-4, 5)), int(rng.integers(-4, 5))) for _ in range(int(rng.integers(0, 12))))
            with self.subTest(case=case):
                self.assert_pairs(previous, current, offsets, 500000 + case * 1000, 900000 + case * 1000)

    def test_uint32_limits_and_uint64_all_ones_are_real_keys(self):
        maximum = 0xFFFFFFFF
        for dtype, local_maximum in ((np.uint16, 65535), (np.uint32, maximum)):
            previous = np.array([[local_maximum, 1, 0], [0, 2, local_maximum]], dtype=dtype)
            current = np.array([[local_maximum, 2, 0], [3, 0, 1]], dtype=dtype)
            offset = maximum - local_maximum
            with self.subTest(dtype=dtype):
                codes = self.assert_pairs(previous, current, previous_offset=offset, current_offset=offset)
                self.assertIn(np.uint64(0xFFFFFFFFFFFFFFFF), codes)
        previous = np.array([[1, 0]], dtype=np.uint16)
        current = np.array([[0, 1]], dtype=np.uint16)
        self.assert_pairs(previous, current, ((0, 1),), maximum - 1, maximum - 1)

    def test_layouts_readonly_negative_strides_and_signed_background(self):
        rng = np.random.default_rng(93991)
        full_previous = rng.integers(-3, 13, (26, 30), dtype=np.int32)
        full_current = rng.integers(-2, 17, (26, 30), dtype=np.int32)
        variants = [
            (full_previous[::2, ::2], full_current[::2, ::2]),
            (full_previous[::-1, ::-1], full_current[::-1, ::-1]),
            (full_previous.T, full_current.T),
            (np.asfortranarray(full_previous), np.asfortranarray(full_current)),
            (full_previous, full_current[:, ::-1]),  # mirrored Azimuthal first-slice seam
        ]
        for index, (previous, current) in enumerate(variants):
            previous.flags.writeable = False
            current.flags.writeable = False
            with self.subTest(layout=index):
                self.assert_pairs(previous, current, previous_offset=500, current_offset=1200)

    @unittest.skipUnless(HAS_COMPILED_ADJACENCY, "Native Numba unavailable")
    def test_forced_hash_spills_preserve_all_codes_and_resume_positions(self):
        rng = np.random.default_rng(51773)
        previous = np.arange(1, 256, dtype=np.uint32).reshape(15, 17)
        current = previous[::-1, ::-1].copy()
        for initial, maximum in ((8, 8), (8, 32), (16, 128)):
            for offsets in (None, ((0, 0),), ((1, -1), (0, 0), (-1, 1))):
                with self.subTest(initial=initial, maximum=maximum, offsets=offsets):
                    self.assert_pairs(previous, current, offsets, 33000, 2**31,
                                      initial_capacity=initial, max_capacity=maximum)
        # Repeated same key around transitions must neither loop indefinitely nor
        # confuse an occupied slot with a cursor that has already been committed.
        repeated_previous = rng.integers(1, 5, (21, 23), dtype=np.uint16)
        repeated_current = rng.integers(1, 6, (21, 23), dtype=np.uint16)
        self.assert_pairs(repeated_previous, repeated_current, initial_capacity=8, max_capacity=8)

    def test_unsupported_numeric_dtype_uses_original_contract(self):
        for dtype in (np.uint8, np.int64, np.float32, np.dtype('>u2')):
            previous = np.array([[1, 0, 2], [0, 3, 0]], dtype=dtype)
            current = np.array([[0, 4, 0], [5, 0, 6]], dtype=dtype)
            kwargs = dict(xy_offsets=((0, 0), (1, -1)), prev_offset=17, curr_offset=29)
            with self.subTest(dtype=dtype):
                np.testing.assert_array_equal(topology._adjacent_gid_pair_codes(previous, current, **kwargs), topology._adjacent_gid_pair_codes_numpy(previous, current, **kwargs))

    @unittest.skipUnless(HAS_COMPILED_ADJACENCY, "Native Numba unavailable")
    def test_concurrent_calls_do_not_share_mutable_hash_or_result_ownership(self):
        rng = np.random.default_rng(2281)
        previous = rng.integers(0, 12, (71, 83), dtype=np.uint16)
        current = rng.integers(0, 14, previous.shape, dtype=np.uint16)
        previous.flags.writeable = False
        current.flags.writeable = False
        variants = [(previous, current), (previous[:, ::-1], current[:, ::-1]),
                    (previous.T, current.T)]
        # Warm every dtype/layout signature before launching threads; this checks
        # ownership and native concurrent calls rather than measuring JIT behavior.
        expected = [self.assert_pairs(a, b, previous_offset=700, current_offset=1900) for a, b in variants]

        def call(index):
            a, b = variants[index % len(variants)]
            return topology._adjacent_gid_pair_codes(a, b, prev_offset=700, curr_offset=1900)

        with ThreadPoolExecutor(max_workers=8) as pool:
            returned = list(pool.map(call, range(48)))
        for index, result in enumerate(returned):
            np.testing.assert_array_equal(result, expected[index % len(expected)])
        for index in range(1, len(returned)):
            self.assertFalse(np.shares_memory(returned[0], returned[index]))
        returned[0].fill(0)
        for index in range(1, len(returned)):
            np.testing.assert_array_equal(returned[index], expected[index % len(expected)])

    def test_disabled_or_unavailable_numba_uses_reference_without_native_attempt(self):
        previous = np.array([[0, 1], [2, 0]], dtype=np.uint16)
        current = np.array([[3, 0], [0, 4]], dtype=np.uint16)
        expected = oracle(previous, current)
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable), mock.patch.object(
                topology, '_compiled_adjacent_gid_pair_codes', side_effect=AssertionError('native attempt'),
            ), mock.patch.object(
                topology, '_numba_adjacency_scan_kernel', None if unavailable else object(),
            ), mock.patch.object(topology, 'compiled_topology_kernels_enabled', return_value=unavailable):
                np.testing.assert_array_equal(topology._adjacent_gid_pair_codes(previous, current), expected)

    def test_native_failure_replays_reference_once_and_preserves_iterable_offsets(self):
        previous = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.uint16)
        current = np.array([[0, 4, 0], [5, 0, 6]], dtype=np.uint16)
        offsets = ((0, 1), (1, -1), (-1, 0))
        expected = oracle(previous, current, offsets, 41, 77)
        telemetry = mock.Mock()
        with mock.patch.object(topology, '_NUMBA_ADJACENCY_RUNTIME_DISABLED', False), \
                mock.patch.object(topology, '_numba_adjacency_scan_kernel', object()), \
                mock.patch.object(topology, 'compiled_topology_kernels_enabled', return_value=True), \
                mock.patch.object(topology, 'runtime_telemetry', return_value=telemetry), \
                mock.patch.object(topology, '_compiled_adjacent_gid_pair_codes', side_effect=RuntimeError('native unavailable')) as native, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            for _ in range(2):
                actual = topology._adjacent_gid_pair_codes(previous, current, iter(offsets), 41, 77)
                np.testing.assert_array_equal(actual, expected)
            self.assertTrue(topology._NUMBA_ADJACENCY_RUNTIME_DISABLED)
        self.assertEqual(native.call_count, 1)
        self.assertEqual(output.getvalue().count('compiled adjacency failed'), 1)
        telemetry.fallback.assert_called_once()


if __name__ == '__main__':
    unittest.main()
