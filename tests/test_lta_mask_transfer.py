from __future__ import annotations

import unittest
import importlib.util
from unittest import mock

import numpy as np

from XTA.lta_postprocessing import fill_binary_mask_holes_2d


def available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@unittest.skipUnless(available('cv2') and available('scipy'), 'requires real OpenCV and SciPy')
class LtaCroppedHoleFillTests(unittest.TestCase):
    def test_matches_independent_full_plane_flood_fill_on_edges_and_disconnected_masks(self):
        from scipy.ndimage import binary_fill_holes
        rng = np.random.default_rng(319)
        masks = []
        for top, left in ((0, 0), (0, 23), (19, 0), (19, 23), (7, 11)):
            mask = np.zeros((32, 40), dtype=bool)
            mask[top:top + 13, left:left + 17] = True
            mask[top + 2:top + 11, left + 2:left + 15] = False
            masks.append(mask)
            open_ring = mask.copy()
            open_ring[top:top + 3, left + 8] = False
            masks.append(open_ring)
        masks.extend((np.zeros((32, 40), bool), np.ones((32, 40), bool)))
        masks.append(masks[0] | masks[6])
        for density in (0.02, 0.3, 0.7, 0.98):
            masks.extend(rng.random((10, 32, 40)) < density)
        for index, mask in enumerate(masks):
            with self.subTest(index=index):
                original = mask.copy()
                actual = fill_binary_mask_holes_2d(mask)
                np.testing.assert_array_equal(actual, binary_fill_holes(mask))
                np.testing.assert_array_equal(mask, original)
                self.assertFalse(np.shares_memory(actual, mask))


@unittest.skipUnless(available('torch'), 'requires real Torch')
class LtaMaskTransferTests(unittest.TestCase):
    def test_batched_transfer_preserves_threshold_geometry_and_independent_ownership(self):
        import torch
        from XTA.lta_experimental import _binary_mask_batch_to_host
        values = torch.tensor([
            [[[-1.0, 0.0, 2.0, float('nan')]]],
            [[[3.0, -2.0, 0.1, 0.0]]],
        ])
        calls = []
        original_cpu = torch.Tensor.cpu

        def cpu(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, 'cpu', new=cpu):
            masks = _binary_mask_batch_to_host(values, expected_count=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual([mask.shape for mask in masks], [(1, 4), (1, 4)])
        np.testing.assert_array_equal(masks[0], [[False, False, True, False]])
        np.testing.assert_array_equal(masks[1], [[True, False, True, False]])
        self.assertTrue(all(mask.flags.owndata for mask in masks))
        self.assertFalse(np.shares_memory(masks[0], masks[1]))
        values.fill_(0)
        self.assertTrue(masks[0][0, 2])

    def test_score_batch_preserves_exact_sigmoid_and_sentinel_with_one_host_transfer(self):
        import torch
        from XTA.lta_experimental import _sigmoid_tracker_score_logits
        scores = torch.tensor([-1e4, -9999.0, -20.0, 0.0, 10.0], dtype=torch.float32)
        expected = torch.sigmoid(scores.to(torch.float64)).tolist()
        expected[0] = None
        calls = []
        original_cpu = torch.Tensor.cpu

        def cpu(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        with (
            mock.patch.object(torch.Tensor, 'cpu', new=cpu),
            mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar synchronization')),
        ):
            actual = _sigmoid_tracker_score_logits(scores, expected_count=5, torch_module=torch)
        self.assertEqual(actual, tuple(expected))
        self.assertEqual(len(calls), 1)
        for invalid in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, 'non-finite'):
                _sigmoid_tracker_score_logits(
                    torch.tensor([invalid]), expected_count=1, torch_module=torch,
                )


if __name__ == '__main__':
    unittest.main()
