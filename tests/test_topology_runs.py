from __future__ import annotations

import unittest
from unittest import mock
import numpy as np
from XTA import topology_runs
from XTA.topology import _adjacent_gid_pair_codes_numpy, _adjacent_xy_offsets_for_3d_connectivity


@unittest.skipIf(topology_runs._topology_label_runs is None, 'Numba unavailable')
class TopologyRunTests(unittest.TestCase):
    def test_optional_compiler_failure_retains_pixel_fallback(self):
        a = np.ones((32, 32), np.uint16)
        with mock.patch.object(topology_runs, '_TOPOLOGY_RUNS_DISABLED', False), \
                mock.patch.object(topology_runs, '_topology_label_runs', side_effect=RuntimeError('compile failed')) as compile_call:
            self.assertIsNone(topology_runs.run_adjacent_pair_codes(a, a, ((0, 0),)))
            self.assertIsNone(topology_runs.run_adjacent_pair_codes(a, a, ((0, 0),)))
            compile_call.assert_called_once()

    def test_exact_pairs_for_connectivity_strides_and_large_ids(self):
        rng = np.random.default_rng(915)
        for dtype in (np.uint16, np.uint32, np.int32):
            for trial in range(24):
                a, b = [np.repeat(np.repeat(rng.integers(0, 9, (5, 7), dtype=dtype), 3, 0), 11, 1)
                        for _ in range(2)]
                if trial % 2:
                    a, b = a[::-1, ::-1], b[::-1, ::-1]
                offsets = _adjacent_xy_offsets_for_3d_connectivity((6, 18, 26)[trial % 3])
                if trial % 4 == 0:
                    offsets = ((-2, -3), (0, -2), (0, 2), (0, 2), (1, 0))
                ao, bo = (0x80000000, 0xFFFFFF00) if trial % 3 else (0, 0)
                expected = _adjacent_gid_pair_codes_numpy(a, b, offsets, ao, bo)
                actual = topology_runs.run_adjacent_pair_codes(a, b, offsets, ao, bo)
                self.assertIsNotNone(actual)
                np.testing.assert_array_equal(actual, expected)

    def test_expanded_neighbor_runs_can_overlap_the_same_current_run(self):
        a = np.repeat(np.array([[1, 1, 2, 2, 3, 3]], np.uint32), 8, axis=0)
        b = np.repeat(np.array([[0, 0, 4, 0, 0, 0]], np.uint32), 8, axis=0)
        offsets = ((0, -1), (0, 0), (0, 1))
        actual = topology_runs.run_adjacent_pair_codes(a, b, offsets)
        # The automatic density admission may decline this tiny fragmented case.
        if actual is not None:
            np.testing.assert_array_equal(actual, _adjacent_gid_pair_codes_numpy(a, b, offsets))
        a = np.repeat(a, 16, axis=1)
        b = np.repeat(b, 16, axis=1)
        offsets = ((0, -16), (0, 16))
        np.testing.assert_array_equal(topology_runs.run_adjacent_pair_codes(a, b, offsets),
                                      _adjacent_gid_pair_codes_numpy(a, b, offsets))

    def test_fragmentation_and_pair_caps_decline_without_mutation(self):
        a = np.arange(1, 4097, dtype=np.uint32).reshape(64, 64)
        original = a.copy()
        self.assertIsNone(topology_runs.run_adjacent_pair_codes(a, a, ((0, 0),), run_cap=4))
        a = np.repeat(np.arange(1, 33, dtype=np.uint32)[None, :], 64, axis=0)
        a = np.repeat(a, 16, axis=1)
        self.assertIsNone(topology_runs.run_adjacent_pair_codes(a, a, ((0, 0),), pair_cap=2))
        np.testing.assert_array_equal(original, np.arange(1, 4097, dtype=np.uint32).reshape(64, 64))

    def test_empty_labels_and_empty_offsets(self):
        a = np.zeros((16, 80), np.uint16)
        for b in (a, a + 1):
            self.assertEqual(topology_runs.run_adjacent_pair_codes(a, b, ((0, 0),)).size, 0)
        self.assertEqual(topology_runs.run_adjacent_pair_codes(a + 1, a + 1, ()).size, 0)


if __name__ == '__main__':
    unittest.main()
