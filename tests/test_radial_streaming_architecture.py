"""Exact decomposition and coverage checks for the proposed shell-owner boundary."""
import unittest
import subprocess
import sys

import numpy as np

from XTA import geometry
from tools.probe_radial_streaming_architecture import ShellChunkPullReference, decode_words, oracle, run_matrix


class RadialStreamingArchitectureTests(unittest.TestCase):
    def test_cleanup_requires_the_completed_shell_union(self):
        # Control-only tests can import inference with native dependency stubs.
        # Isolate this numerical law so it always exercises real SciPy.
        code = """
import numpy as np
from XTA.inference import _fill_holes_2d_scipy as fill
ring = np.zeros((7, 7), np.uint8)
ring[1, 1:6] = ring[5, 1:6] = 1
ring[1:6, 1] = ring[1:6, 5] = 1
a = ring.copy(); a[1, 3] = 0
b = np.zeros_like(a); b[1, 3] = 1
assert fill(a | b)[3, 3]
assert not (fill(a) | fill(b))[3, 3]
"""
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)

    def test_all_axes_tilts_periods_and_grid_restores(self):
        report = run_matrix()
        self.assertTrue(report['all_exact'])
        self.assertGreater(report['case_count'], 100)

    def test_impulse_basis_matches_original_pull(self):
        for base in ('transverse', 'sagittal', 'coronal'):
            view = geometry.get_view_infos(3, 5, 7, cartesian_views=(), radial_views=(base,),
                radial_min_radius=.01, radial_patch_size=4)[0]
            shape = (3, 5, 7)
            source = np.zeros((view.num_slices, 4, 4), np.uint8)
            for pixel in range(source.size):
                source.flat[pixel] = 1
                owner = ShellChunkPullReference(view, (4, 4), shape)
                for shell in range(view.num_slices):
                    owner.consume(shell, source[shell])
                np.testing.assert_array_equal(decode_words(owner.seal(), shape), oracle(source, view, shape))
                source.flat[pixel] = 0

    def test_empty_foreign_duplicate_and_missing_shell_coverage(self):
        view = geometry.get_view_infos(5, 7, 9, cartesian_views=(), radial_views=('transverse',),
            radial_min_radius=.1, radial_patch_size=4)[0]
        owner = ShellChunkPullReference(view, (4, 4), (5, 7, 9), expected_shells=(0,))
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            owner.seal()
        with self.assertRaisesRegex(ValueError, 'different owner'):
            owner.consume(1, np.zeros((4, 4), np.uint8))
        owner.consume(0, np.zeros((4, 4), np.uint8))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            owner.consume(0, np.zeros((4, 4), np.uint8))
        self.assertFalse(owner.seal().flags.writeable)
        with self.assertRaisesRegex(RuntimeError, 'sealed'):
            owner.consume(0, np.zeros((4, 4), np.uint8))


if __name__ == '__main__':
    unittest.main()
