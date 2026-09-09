"""Counterexamples where global IoU hides thin-structure geometry failures."""
from functools import wraps
import os
from pathlib import Path
import subprocess
import sys
import unittest


_CHILD_FLAG = 'XTA_GEOMETRY_QUALITY_TEST_CHILD'
_ISOLATED_CHILD = os.environ.get(_CHILD_FLAG) == '1'

# Import-only suites deliberately replace SciPy globally. Keep these numerical
# cases in fresh interpreters instead of reloading native modules or accepting
# stubbed calculations. Missing real dependencies remain a test failure.
if _ISOLATED_CHILD:
    import numpy as np
    from scipy import ndimage as ndi

    from tools.geometry_quality_metrics import compare_binary_masks, phantom_labels


def isolated_numerical_case(function):
    @wraps(function)
    def run(self):
        if _ISOLATED_CHILD:
            return function(self)
        environment = dict(os.environ)
        environment[_CHILD_FLAG] = '1'
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            environment[name] = '2'
        test = f'tests.test_geometry_quality_metrics.GeometryQualityMetricsTests.{function.__name__}'
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', test],
            cwd=Path(__file__).resolve().parents[1], env=environment,
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return run


class GeometryQualityMetricsTests(unittest.TestCase):
    @isolated_numerical_case
    def test_missing_tiny_rod_is_reported_despite_excellent_global_iou(self):
        ref = np.zeros((64, 64, 64), bool)
        ref[5:50, 5:50, 5:50] = True
        ref[10:31, 58, 58] = True
        got = ref.copy()
        got[10:31, 58, 58] = False
        result = compare_binary_masks(ref, got)
        self.assertGreater(result['iou'], .999)
        self.assertEqual(result['missed_thin_components'], 1)
        missed = next(c for c in result['components'] if not c['survives_within_one_voxel'])
        self.assertEqual(missed['centerline_proxy_recall'], 0.)

    @isolated_numerical_case
    def test_one_voxel_shift_survives_even_when_rod_iou_is_zero(self):
        ref, got = np.zeros((25, 25, 25), bool), np.zeros((25, 25, 25), bool)
        ref[3:22, 10, 10] = True
        got[3:22, 11, 11] = True
        result = compare_binary_masks(ref, got)
        self.assertEqual(result['iou'], 0.)
        self.assertEqual(result['missed_components'], 0)
        self.assertEqual(result['components'][0]['centerline_proxy_recall'], 1.)
        self.assertAlmostEqual(result['surface_distance_max'], np.sqrt(2))

    @isolated_numerical_case
    def test_rod_break_and_truncation_are_not_hidden_by_component_survival(self):
        ref = np.zeros((35, 17, 17), bool)
        ref[2:33, 8, 8] = True
        cut = ref.copy()
        cut[17, 8, 8] = False
        result = compare_binary_masks(ref, cut)
        self.assertEqual(result['missed_components'], 0)
        self.assertEqual(result['split_reference_components'], [1])
        self.assertTrue(result['components'][0]['matching_ambiguous'])
        truncated = ref.copy()
        truncated[17:, 8, 8] = False
        result = compare_binary_masks(ref, truncated)
        self.assertEqual(result['candidate_components'], 1)
        self.assertLess(result['components'][0]['centerline_proxy_recall'], .6)

    @isolated_numerical_case
    def test_new_bridge_is_a_definite_merge_but_tolerance_only_matching_is_ambiguous(self):
        ref = np.zeros((25, 25, 25), bool)
        ref[3:22, 10, 8] = ref[3:22, 10, 10] = True
        bridge = ref.copy()
        bridge[12, 10, 9] = True
        result = compare_binary_masks(ref, bridge)
        self.assertEqual(result['merged_candidate_components'], [1])
        # A single rod between the originals is consistent with either a shift
        # plus deletion or a merger. The metric must not invent correspondence.
        middle = np.zeros_like(ref)
        middle[3:22, 10, 9] = True
        result = compare_binary_masks(ref, middle)
        self.assertEqual(result['merged_candidate_components'], [])
        self.assertEqual(result['possible_merge_candidate_components'], [1])
        self.assertEqual(result['missed_components'], 0)
        self.assertTrue(all(c['matching_ambiguous'] for c in result['components']))

    @isolated_numerical_case
    def test_26_connected_diagonal_rod_and_phantom_object_labels(self):
        objects = [
            {'kind': 'rod', 'start': (2, 2, 2), 'stop': (8, 8, 8), 'width': 1},
            {'kind': 'sheet', 'start': (15, 2, 2), 'stop': (17, 9, 10)},
            {'kind': 'ring', 'center': (10, 22, 22), 'axis': 0, 'radius': 5, 'width': 1},
        ]
        labels = phantom_labels((33, 33, 33), objects)
        np.testing.assert_array_equal(np.unique(labels), [0, 1, 2, 3])
        result = compare_binary_masks(labels, labels)
        self.assertEqual(result['reference_components'], 3)
        self.assertEqual(result['surface_distance_max'], 0.)
        self.assertTrue(all(c['centerline_proxy_recall'] == 1. for c in result['components']))

    @isolated_numerical_case
    def test_known_ring_hole_opens_while_global_component_count_stays_one(self):
        labels = phantom_labels((25, 25, 25), [
            {'kind': 'ring', 'center': (12, 12, 12), 'axis': 0, 'radius': 6, 'width': 1},
        ])
        ref = labels != 0
        cut = ref.copy()
        cut[12, 12, 18] = False
        result = compare_binary_masks(ref, cut)
        self.assertEqual(result['reference_components'], 1)
        self.assertEqual(result['candidate_components'], 1)
        self.assertEqual(result['components'][0]['centerline_proxy_recall'], 1.)
        # A known planar loop gets an explicit hole assertion. Component count
        # and centerline coverage alone cannot certify ring topology.
        hole = lambda plane: ndi.binary_fill_holes(plane) & ~plane
        self.assertTrue(hole(ref[12]).any())
        self.assertFalse(hole(cut[12]).any())

    @isolated_numerical_case
    def test_axis_rods_have_the_requested_one_two_and_three_voxel_widths(self):
        for width in (1, 2, 3):
            labels = phantom_labels((17, 17, 17), [
                {'kind': 'rod', 'start': (4, 8, 8), 'stop': (10, 8, 8), 'width': width},
            ])
            self.assertEqual(np.count_nonzero(labels), (7 + width - 1) * width ** 2)
            result = compare_binary_masks(labels, labels)
            self.assertEqual(result['reference_components'], 1)
            self.assertTrue(result['components'][0]['thin'])

    @isolated_numerical_case
    def test_empty_inputs_and_extra_island_remain_well_defined(self):
        empty = np.zeros((9, 9, 9), bool)
        self.assertEqual(compare_binary_masks(empty, empty)['iou'], 1.)
        got = empty.copy()
        got[4, 4, 4] = True
        result = compare_binary_masks(empty, got)
        self.assertEqual(result['unmatched_candidate_components'], [1])
        self.assertTrue(np.isinf(result['surface_distance_max']))
        self.assertEqual(compare_binary_masks(got, empty)['missed_thin_components'], 1)
        with self.assertRaises(ValueError):
            compare_binary_masks(np.zeros((129, 1, 1)), np.zeros((129, 1, 1)))
        with self.assertRaises(ValueError):
            compare_binary_masks(empty, np.zeros((2, 2, 2)))


if __name__ == '__main__':
    unittest.main()
