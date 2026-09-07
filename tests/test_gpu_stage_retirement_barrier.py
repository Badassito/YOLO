"""GPU stage admission remains fenced through terminal inference-asset ACKs."""
from __future__ import annotations

import ast
from pathlib import Path
import types
import unittest
from unittest import mock

from XTA import backprojection, pipeline


class GpuStageRetirementBarrierTests(unittest.TestCase):
    def setUp(self):
        self.priority = mock.patch.object(backprojection, 'main_process_gpu_stage_inference_priority_enabled', return_value=True)
        self.overlap = mock.patch.object(backprojection, 'main_process_gpu_stage_inference_overlap_enabled', return_value=False)
        self.fallback = mock.patch.object(backprojection, 'v1613_d1_backprojection_overlap_enabled', return_value=True)
        self.aux = mock.patch.object(backprojection, 'gpu_worker_aux_interpolation_pool', return_value=None)
        for patch in (self.priority, self.overlap, self.fallback, self.aux):
            patch.start()
            self.addCleanup(patch.stop)
        self.torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device_count=mock.Mock(return_value=4),
                                       mem_get_info=mock.Mock(return_value=(1000, 2000))),
            device=lambda token: token,
        )
        self.coordinator = backprojection._MainProcessGpuStageCoordinator()
        self.coordinator.configure_workers([0, 2])

    def test_non_d1_overlap_remains_available_before_the_retirement_boundary(self):
        self.assertTrue(self.coordinator.snapshot()['inference_priority_active'])
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'Azimuthal backprojection')
        self.assertIsNotNone(lease)
        lease.release()

    def test_pending_retirement_blocks_selected_devices_under_every_overlap_policy(self):
        for priority in (False, True):
            for overlap in (False, True):
                with self.subTest(priority=priority, overlap=overlap), \
                        mock.patch.object(backprojection, 'main_process_gpu_stage_inference_priority_enabled', return_value=priority), \
                        mock.patch.object(backprojection, 'main_process_gpu_stage_inference_overlap_enabled', return_value=overlap):
                    self.coordinator.configure_workers([0, 2])
                    self.coordinator.set_inference_asset_retirement_pending(True)
                    for purpose in ('Azimuthal backprojection', 'NRRD mirror downbin', 'other output'):
                        for device in (0, 2):
                            self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, device, purpose))
                        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, purpose))
        # In particular, a blocked generic selection must not measure free HBM
        # and commit to a fallback based on the workers' old allocation footprint.
        self.torch.cuda.mem_get_info.assert_not_called()

    def test_pending_flag_does_not_block_other_devices_or_revoke_existing_leases(self):
        prior = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'Azimuthal backprojection')
        self.coordinator.set_inference_asset_retirement_pending(True)
        self.assertEqual(self.coordinator.snapshot()['stage_leases'], {0: 'Azimuthal backprojection'})
        unrelated = self.coordinator.try_acquire_specific_stage(self.torch, 1, 'NRRD output')
        self.assertIsNotNone(unrelated)
        unrelated.release()
        prior.release()
        self.assertEqual(self.coordinator.snapshot()['stage_leases'], {})
        self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 0, 'Azimuthal backprojection'))

    def test_ack_clear_wakes_waiters_and_configure_reset_remove_stale_state(self):
        wake = mock.Mock()
        self.coordinator.set_wake_callback(wake)
        self.coordinator.set_inference_asset_retirement_pending(True)
        self.coordinator.set_inference_asset_retirement_pending(True)
        self.assertEqual(wake.call_count, 1)
        self.coordinator.set_inference_priority_active(False)
        self.coordinator.set_inference_asset_retirement_pending(False)
        self.assertEqual(wake.call_count, 3)
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 2, 'NRRD output')
        self.assertIsNotNone(lease)
        lease.release()
        self.coordinator.set_inference_asset_retirement_pending(True)
        self.coordinator.configure_workers([0, 2])
        self.assertFalse(self.coordinator.snapshot()['inference_asset_retirement_pending'])
        self.coordinator.set_inference_asset_retirement_pending(True)
        self.coordinator.reset()
        self.assertFalse(self.coordinator.snapshot()['inference_asset_retirement_pending'])

    def callback(self, request, *, collected=2):
        source = Path(pipeline.__file__).read_text(encoding='utf-8')
        function = next(node for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.FunctionDef) and node.name == '_announce_process_inference_drain_if_complete')
        namespace = dict(vars(pipeline))
        namespace.update({
            'scheduler_state': types.SimpleNamespace(gpu_worker_results_collected=collected,
                gpu_worker_total_tasks=2, gpu_inference_drain_announced=True),
            'gpu_worker_process_active': True,
            'scheduler': types.SimpleNamespace(request_gpu_inference_asset_release=request),
            '_restore_parent_post_inference_affinity': lambda: None,
            '_set_main_process_gpu_asset_retirement_pending': self.coordinator.set_inference_asset_retirement_pending,
            '_set_main_process_gpu_inference_priority_active': self.coordinator.set_inference_priority_active,
        })
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<pipeline-drain-callback>', 'exec'), namespace)
        return namespace[function.name]

    def test_real_pipeline_callback_seals_before_controls_and_clears_after_priority(self):
        answers = iter((False, True))
        observed = []

        def request():
            observed.append(self.coordinator.snapshot())
            self.assertTrue(observed[-1]['inference_asset_retirement_pending'])
            return next(answers)

        callback = self.callback(request)
        callback()
        self.assertTrue(self.coordinator.snapshot()['inference_asset_retirement_pending'])
        self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 0, 'Azimuthal backprojection'))
        priority_before_clear = []
        original_clear = self.coordinator.set_inference_asset_retirement_pending

        def clear(active):
            if not active:
                priority_before_clear.append(self.coordinator.snapshot()['inference_priority_active'])
            original_clear(active)

        with mock.patch.object(self.coordinator, 'set_inference_asset_retirement_pending', side_effect=clear):
            callback = self.callback(request)
            callback()
        self.assertEqual(priority_before_clear, [False])
        self.assertFalse(self.coordinator.snapshot()['inference_asset_retirement_pending'])
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'Azimuthal backprojection')
        self.assertIsNotNone(lease)
        lease.release()

    def test_early_or_failed_pipeline_handoff_cannot_open_stage_admission(self):
        request = mock.Mock(return_value=False)
        self.callback(request, collected=1)()
        request.assert_not_called()
        self.assertFalse(self.coordinator.snapshot()['inference_asset_retirement_pending'])
        with self.assertRaisesRegex(RuntimeError, 'queue failed'):
            self.callback(mock.Mock(side_effect=RuntimeError('queue failed')))()
        self.assertTrue(self.coordinator.snapshot()['inference_asset_retirement_pending'])
        self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 0, 'Azimuthal backprojection'))


if __name__ == '__main__':
    unittest.main()
