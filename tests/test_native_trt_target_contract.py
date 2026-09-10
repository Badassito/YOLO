"""The native ring must fulfill the same radius lease as generic prediction."""
from types import SimpleNamespace
import unittest
import numpy as np

from XTA.cylindrical_owner import DeviceOnlyRadialTarget
from XTA.inference import _claim_specialized_prediction_targets


class NativeTargetContractTests(unittest.TestCase):
    def test_completed_device_rows_claim_every_radius_without_host_access(self):
        target = DeviceOnlyRadialTarget((8, 31, 33))
        _claim_specialized_prediction_targets(target, SimpleNamespace(written=np.ones(8, bool)), 8)
        target.require_complete()
        with self.assertRaisesRegex(RuntimeError, 'duplicated'):
            _claim_specialized_prediction_targets(target, SimpleNamespace(written=np.ones(8, bool)), 8)

    def test_incomplete_or_wrong_depth_cannot_partially_claim_the_target(self):
        for written, count in (([True, False, True], 3), ([True, True], 3), ([True]*4, 4)):
            target = DeviceOnlyRadialTarget((3, 7, 9))
            with self.assertRaises(RuntimeError):
                _claim_specialized_prediction_targets(target, SimpleNamespace(written=written), count)
            self.assertFalse(target._received.any())

    def test_plain_dense_targets_need_no_coverage_side_channel(self):
        target = np.zeros((3, 7, 9), np.uint8)
        _claim_specialized_prediction_targets(target, None, 3)
        self.assertFalse(target.any())


if __name__ == '__main__':
    unittest.main()
