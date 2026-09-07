from __future__ import annotations

import unittest

import numpy as np

from XTA.inference import _split_segmentation_backend_outputs


class SegmentationBackendOutputLayouts(unittest.TestCase):
    def setUp(self):
        self.head = np.zeros((1, 37, 340), dtype=np.float32)
        self.proto = np.zeros((1, 32, 32, 32), dtype=np.float32)
        self.coefficients = np.zeros((1, 32, 340), dtype=np.float32)

    def assert_pair(self, outputs):
        result = _split_segmentation_backend_outputs(outputs)
        self.assertIsNotNone(result)
        self.assertIs(result[0], self.head)
        self.assertIs(result[1], self.proto)

    def test_exported_tensor_rt_pair_stays_unchanged(self):
        self.assert_pair((self.head, self.proto))
        self.assert_pair([self.head, self.proto])

    def test_legacy_pytorch_auxiliary_tuple_stays_unchanged(self):
        features = [np.zeros((1, 192, 16, 16), dtype=np.float32)]
        self.assert_pair((self.head, (features, self.coefficients, self.proto)))
        self.assert_pair([self.head, [features, self.coefficients, self.proto]])

    def test_current_segment_inference_pair_and_autobackend_wrapper(self):
        auxiliary = {
            'boxes': np.zeros((1, 64, 340), dtype=np.float32),
            'scores': np.zeros((1, 1, 340), dtype=np.float32),
            'feats': [np.zeros((1, 192, 16, 16), dtype=np.float32)],
            'mask_coefficient': self.coefficients,
            'proto': self.proto,
        }
        self.assert_pair(((self.head, self.proto), auxiliary))
        self.assert_pair([(self.head, self.proto), auxiliary])

    def test_auxiliary_tensors_are_not_guessed_as_inference_outputs(self):
        # The inference pair is authoritative even if the auxiliary dictionary
        # contains another shape-compatible tensor at an unrelated key.
        auxiliary = {'proto': np.ones_like(self.proto), 'head': np.ones_like(self.head)}
        self.assert_pair(((self.head, self.proto), auxiliary))
        for outputs in (
            {'head': self.head, 'proto': self.proto},
            ({'head': self.head}, {'proto': self.proto}),
            ((self.head, self.proto), {'unrelated': self.proto}),
            ([self.head, self.proto], {'proto': self.proto}),
            ((self.head, self.proto, self.coefficients), {'proto': self.proto}),
            ((self.proto, self.head), {'proto': self.proto}),
            ((self.head, self.proto[0]), {'proto': self.proto}),
            (), (self.head,), (self.head, ()), None,
        ):
            with self.subTest(kind=type(outputs).__name__):
                self.assertIsNone(_split_segmentation_backend_outputs(outputs))


if __name__ == '__main__':
    unittest.main()
