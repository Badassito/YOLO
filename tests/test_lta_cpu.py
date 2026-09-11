from __future__ import annotations

import os
import types
import sys
import unittest
from unittest import mock

from XTA.lta_cpu import (
    bind_worker_cpu_environment, configure_worker_runtime_threads, resolve_worker_cpu_budget,
    bounded_lta_host_threads,
)


class LtaCpuBudgetTests(unittest.TestCase):
    def test_wide_affinity_does_not_override_slurm_allocation(self):
        budget = resolve_worker_cpu_budget(4, affinity_count=128, cpu_count=256, environ={
            "SLURM_CPUS_PER_TASK": "16", "SLURM_CPUS_ON_NODE": "128",
            "SLURM_JOB_CPUS_PER_NODE": "128(x2)",
        })
        self.assertEqual(budget["effective_cpu_count"], 16)
        self.assertEqual(budget["threads_per_worker"], 3)
        self.assertLessEqual(4 * budget["threads_per_worker"] + 1, 16)

    def test_affinity_and_explicit_lower_thread_limit_both_apply(self):
        budget = resolve_worker_cpu_budget(4, affinity_count=8, cpu_count=128, environ={
            "SLURM_CPUS_PER_TASK": "64", "OMP_NUM_THREADS": "1",
        })
        self.assertEqual(budget["effective_cpu_count"], 8)
        self.assertEqual(budget["threads_per_worker"], 1)
        self.assertEqual(resolve_worker_cpu_budget(4, cpu_count=128, environ={})["threads_per_worker"], 4)

    def test_insufficient_cpu_allocation_is_reported_without_zero_thread_pools(self):
        budget = resolve_worker_cpu_budget(4, cpu_count=128, environ={"SLURM_CPUS_PER_TASK": "1"})
        self.assertTrue(budget["minimum_worker_threads_exceed_allocation"])
        self.assertEqual(budget["threads_per_worker"], 1)

    def test_runtime_settings_report_actual_limits(self):
        torch = types.SimpleNamespace(
            set_num_threads=mock.Mock(), get_num_threads=lambda: 2,
            set_num_interop_threads=mock.Mock(), get_num_interop_threads=lambda: 1,
        )
        cv2 = types.SimpleNamespace(setNumThreads=mock.Mock(), getNumThreads=lambda: 2)
        with mock.patch.dict(os.environ, {}, clear=True):
            bind_worker_cpu_environment({"threads_per_worker": 2})
            receipt = configure_worker_runtime_threads(torch, cv2)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")
            self.assertEqual(os.environ["OMP_WAIT_POLICY"], "PASSIVE")
            self.assertEqual(receipt["torch_num_threads"], 2)
            self.assertEqual(receipt["opencv_num_threads"], 2)
            torch.set_num_threads.assert_called_once_with(2)
            cv2.setNumThreads.assert_called_once_with(2)

    def test_host_native_pools_are_bounded_and_restored_after_failure(self):
        counts = {"cv2": 64, "torch": 128}
        cv2 = types.SimpleNamespace(
            getNumThreads=lambda: counts['cv2'],
            setNumThreads=lambda value: counts.update(cv2=value),
        )
        torch = types.SimpleNamespace(
            get_num_threads=lambda: counts['torch'],
            set_num_threads=lambda value: counts.update(torch=value),
        )
        @bounded_lta_host_threads
        def failed_run():
            self.assertEqual(counts, {"cv2": 1, "torch": 1})
            raise ValueError('failed run')
        before = dict(os.environ)
        with mock.patch.dict(sys.modules, {'cv2': cv2, 'torch': torch}):
            with self.assertRaisesRegex(ValueError, 'failed run'):
                failed_run()
        self.assertEqual(counts, {"cv2": 64, "torch": 128})
        self.assertEqual(dict(os.environ), before)


if __name__ == '__main__':
    unittest.main()
