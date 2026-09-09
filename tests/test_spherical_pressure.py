"""One demanded retirement lane can drain without stopping other inference."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import backprojection as bp
from XTA.geometry import ViewInfo
from tests.test_tta_scheduler_boundary import _scheduler, _state


class SphericalPressureTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = bp._MainProcessGpuStageCoordinator()
        self.coordinator.configure_workers([0, 1, 2, 3])
        self.coordinator.set_pending_inference_backlog(True)
        self.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4,
            mem_get_info=mock.Mock(return_value=(1000, 2000))), device=lambda value: value)
        self.purpose = 'Spherical source projection parent'
        for worker in range(4):
            self.coordinator.begin_inference(worker)
            self.coordinator.begin_inference(worker)
        patch = mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=None)
        patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.dict('os.environ', {
            'YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT': '1',
            'YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT': '1',
        })
        patch.start(); self.addCleanup(patch.stop)

    def request(self):
        return self.coordinator.try_acquire_stage(self.torch, self.purpose)

    def test_one_worker_drains_then_projects_with_other_family_backlog(self):
        self.assertIsNone(self.request())
        self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))
        self.coordinator.set_spherical_retirement_pressure(True)
        self.assertIsNone(self.request())
        self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 0)
        self.assertFalse(self.coordinator.can_dispatch_inference(0))
        self.assertFalse(self.coordinator.begin_inference(0))
        for worker in (1, 2, 3):
            self.assertTrue(self.coordinator.can_dispatch_inference(worker))
            self.assertTrue(self.coordinator.begin_inference(worker))
        self.coordinator.finish_inference(0)
        self.assertIsNone(self.request())
        self.coordinator.finish_inference(0)
        lease = self.request()
        self.assertEqual(lease.device_index, 0)
        self.assertTrue(self.coordinator.snapshot()['pending_inference_backlog'])
        self.assertEqual(self.coordinator.snapshot()['spherical_retirement_pressure_acquisitions'], 1)
        self.assertEqual(self.coordinator.snapshot()['spherical_retirement_aged_acquisitions'], 0)
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, self.purpose + ' second'))
        self.assertFalse(self.coordinator.can_dispatch_inference(0))
        self.coordinator.cancel_spherical_retirement_request(self.purpose + ' second')
        lease.release()
        self.assertTrue(self.coordinator.can_dispatch_inference(0))

    def test_fifo_refresh_and_two_projection_handoff_avoid_a_second_worker_drain(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        names = [self.purpose + name for name in (' oldest', ' middle', ' newest')]
        for purpose in names:
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, purpose))
        # Refreshing the oldest reader's TTL does not move it to the FIFO tail.
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, names[0]))
        self.coordinator.finish_inference(0)
        self.coordinator.finish_inference(0)
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, names[2]))
        first = self.coordinator.try_acquire_stage(self.torch, names[0])
        self.assertEqual(first.device_index, 0)
        first.release()
        self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 0)
        self.assertFalse(self.coordinator.begin_inference(0))
        self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in (1, 2, 3)))
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, names[2]))
        second = self.coordinator.try_acquire_stage(self.torch, names[1])
        self.assertEqual(second.device_index, 0)
        self.assertEqual(self.coordinator.snapshot()['spherical_retirement_handoffs'], 1)
        second.release()
        self.assertTrue(self.coordinator.can_dispatch_inference(0))
        # The next demand must drain another worker; a fast polling projector
        # cannot immediately claim the same idle GPU for a third retirement.
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, names[2]))
        self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 1)
        self.assertTrue(self.coordinator.begin_inference(0))

    def test_waiting_demand_can_arrive_and_refresh_while_first_projection_is_live(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        with mock.patch.object(bp.time, 'monotonic', return_value=10.):
            self.assertIsNone(self.request())
            self.coordinator.finish_inference(0)
            self.coordinator.finish_inference(0)
            first = self.request()
            waiting = self.purpose + ' waiting'
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
        with mock.patch.object(bp.time, 'monotonic', return_value=35.):
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
        with mock.patch.object(bp.time, 'monotonic', return_value=45.):
            first.release()
            second = self.coordinator.try_acquire_stage(self.torch, waiting)
            self.assertEqual(second.device_index, 0)
            second.release()

    def test_expired_or_cancelled_handoff_never_parks_an_idle_worker(self):
        for cancel in (False, True):
            self.coordinator.configure_workers([0, 1, 2, 3])
            self.coordinator.set_pending_inference_backlog(True)
            self.coordinator.set_spherical_retirement_pressure(True)
            with mock.patch.object(bp.time, 'monotonic', return_value=10.):
                first = self.request()
                waiting = self.purpose + ' waiting'
                self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
                first.release()
                self.assertFalse(self.coordinator.can_dispatch_inference(first.device_index))
                if cancel:
                    self.coordinator.cancel_spherical_retirement_request(waiting)
                    self.assertTrue(self.coordinator.can_dispatch_inference(first.device_index))
            with mock.patch.object(bp.time, 'monotonic', return_value=41.):
                self.assertTrue(self.coordinator.can_dispatch_inference(first.device_index))
                self.assertEqual(self.coordinator.snapshot()['spherical_retirement_request_count'], 0)

    def test_fifo_expiry_advances_to_next_compatible_live_reader(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        with mock.patch.object(bp.time, 'monotonic', return_value=10.):
            self.assertIsNone(self.request())
        waiting = self.purpose + ' later'
        with mock.patch.object(bp.time, 'monotonic', return_value=20.):
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
        with mock.patch.object(bp.time, 'monotonic', return_value=41.):
            self.coordinator.finish_inference(0)
            self.coordinator.finish_inference(0)
            lease = self.coordinator.try_acquire_stage(self.torch, waiting)
            self.assertEqual(lease.device_index, 0)
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_request_count'], 0)
            lease.release()

    def test_pressure_relief_terminal_fence_and_failure_prevent_handoff(self):
        for reason in ('pressure', 'terminal', 'failed'):
            self.coordinator.configure_workers([0, 1, 2, 3])
            self.coordinator.set_pending_inference_backlog(True)
            self.coordinator.set_spherical_retirement_pressure(True)
            first = self.request()
            waiting = self.purpose + ' waiting'
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
            if reason == 'pressure':
                self.coordinator.set_spherical_retirement_pressure(False)
            elif reason == 'terminal':
                self.coordinator.set_inference_asset_retirement_pending(True)
            else:
                self.coordinator.cancel_spherical_retirement_request(self.purpose, failed=True)
            first.release()
            self.assertTrue(self.coordinator.can_dispatch_inference(first.device_index))
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
            self.assertIsNone(self.coordinator.snapshot()['spherical_retirement_reserved_device'])

    def test_specific_device_waiter_does_not_receive_an_incompatible_handoff(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        self.assertIsNone(self.request())
        self.coordinator.finish_inference(0)
        self.coordinator.finish_inference(0)
        first = self.request()
        specific = self.purpose + ' only cuda1'
        self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 1, specific))
        first.release()
        self.assertTrue(self.coordinator.can_dispatch_inference(0))
        self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 1, specific))
        self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 1)
        self.coordinator.finish_inference(1)
        self.coordinator.finish_inference(1)
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 1, specific)
        self.assertEqual(lease.device_index, 1)
        lease.release()

    def test_sparse_worker_ids_and_pressure_oscillation_keep_burst_cap(self):
        self.coordinator.configure_workers([0, 2])
        self.coordinator.set_pending_inference_backlog(True)
        self.coordinator.set_spherical_retirement_pressure(True)
        first = self.request()
        waiting = self.purpose + ' waiting'
        self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 0, waiting))
        first.release()
        second = self.coordinator.try_acquire_specific_stage(self.torch, 0, waiting)
        self.assertEqual(second.device_index, 0)
        second.release()
        self.coordinator.set_spherical_retirement_pressure(False)
        self.coordinator.set_spherical_retirement_pressure(True)
        third = self.purpose + ' third'
        self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 0, third))
        self.assertTrue(self.coordinator.begin_inference(0))
        self.coordinator.finish_inference(0)
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 0, third)
        self.assertEqual(lease.device_index, 0)
        lease.release()

    def test_new_inference_backlog_counts_an_existing_opportunistic_projection(self):
        self.coordinator.configure_workers([0])
        self.coordinator.set_spherical_retirement_pressure(True)
        first = self.request()  # No backlog: ordinary idle-GPU admission.
        self.coordinator.set_pending_inference_backlog(True)
        second_name, third_name = self.purpose + ' second', self.purpose + ' third'
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, second_name))
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, third_name))
        first.release()
        second = self.coordinator.try_acquire_stage(self.torch, second_name)
        self.assertEqual(second.device_index, 0)
        second.release()
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, third_name))
        self.assertTrue(self.coordinator.begin_inference(0))

    def test_handoff_still_requires_auxiliary_owner_revocation(self):
        self.coordinator.configure_workers([0])
        self.coordinator.set_pending_inference_backlog(True)
        self.coordinator.set_spherical_retirement_pressure(True)
        first = self.request()
        waiting = self.purpose + ' waiting'
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
        first.release()
        self.torch.cuda.mem_get_info.reset_mock()
        auxiliary = SimpleNamespace(revoke_worker=mock.Mock(return_value=False))
        with mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=auxiliary):
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
        self.torch.cuda.mem_get_info.assert_not_called()
        self.assertFalse(self.coordinator.snapshot()['stage_leases'])
        self.coordinator.cancel_spherical_retirement_request(waiting)
        self.assertTrue(self.coordinator.can_dispatch_inference(0))

    def test_slow_fifo_reader_loses_handoff_after_two_seconds_without_reparking_gpu(self):
        self.coordinator.configure_workers([0])
        self.coordinator.set_pending_inference_backlog(True)
        self.coordinator.set_spherical_retirement_pressure(True)
        waiting, newcomer = self.purpose + ' waiting', self.purpose + ' newcomer'
        with mock.patch.object(bp.time, 'monotonic', return_value=10.):
            first = self.request()
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
            first.release()
        with mock.patch.object(bp.time, 'monotonic', return_value=11.999):
            self.assertFalse(self.coordinator.can_dispatch_inference(0))
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, newcomer))
        with mock.patch.object(bp.time, 'monotonic', return_value=12.):
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, newcomer))
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
            snapshot = self.coordinator.snapshot()
            self.assertIsNone(snapshot['spherical_retirement_reserved_device'])
            self.assertEqual(snapshot['spherical_retirement_handoff_expirations'], 1)
            self.assertEqual(snapshot['spherical_retirement_handoffs'], 0)
            self.assertEqual(snapshot['spherical_retirement_request_count'], 2)
            self.assertTrue(self.coordinator.begin_inference(0))
            self.coordinator.finish_inference(0)
            # The deadline yielded the GPU, not the old reader's FIFO position.
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, newcomer))
            second = self.coordinator.try_acquire_stage(self.torch, waiting)
            self.assertEqual(second.device_index, 0)
            self.coordinator.cancel_spherical_retirement_request(newcomer)
            second.release()

    def test_normal_worker_drain_keeps_thirty_second_demand_expiry(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        with mock.patch.object(bp.time, 'monotonic', return_value=10.):
            self.assertIsNone(self.request())
        with mock.patch.object(bp.time, 'monotonic', return_value=12.):
            self.assertFalse(self.coordinator.can_dispatch_inference(0))
        with mock.patch.object(bp.time, 'monotonic', return_value=39.999):
            self.assertFalse(self.coordinator.can_dispatch_inference(0))
        with mock.patch.object(bp.time, 'monotonic', return_value=40.):
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_handoff_expirations'], 0)

    def test_cancel_expiry_and_pressure_relief_restore_dispatch(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        with mock.patch.object(bp.time, 'monotonic', return_value=10.):
            self.assertIsNone(self.request())
            self.assertFalse(self.coordinator.can_dispatch_inference(0))
        with mock.patch.object(bp.time, 'monotonic', return_value=41.):
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
            self.assertIsNone(self.request())
            self.coordinator.cancel_spherical_retirement_request(self.purpose)
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
            self.assertIsNone(self.request())
            self.coordinator.set_spherical_retirement_pressure(False)
            self.assertTrue(self.coordinator.can_dispatch_inference(0))

    def test_failed_admission_cooldown_never_parks_worker(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        with mock.patch.object(bp.time, 'monotonic', return_value=10.):
            self.assertIsNone(self.request())
            self.coordinator.cancel_spherical_retirement_request(self.purpose, failed=True)
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
            self.assertIsNone(self.request())
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
        with mock.patch.object(bp.time, 'monotonic', return_value=21.):
            self.assertIsNone(self.request())
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 1)
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
            self.assertFalse(self.coordinator.can_dispatch_inference(1))

    def test_only_live_demand_reserves_and_terminal_retirement_wins(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))
        self.assertIsNone(self.request())
        self.coordinator.set_inference_asset_retirement_pending(True)
        self.assertIsNone(self.request())
        self.assertIsNone(self.coordinator.snapshot()['spherical_retirement_reserved_device'])
        self.coordinator.set_inference_asset_retirement_pending(False)
        with mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT': '0'}):
            self.assertIsNone(self.request())
            self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))

    def test_specific_request_does_not_steal_a_draining_worker(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        self.assertIsNone(self.request())
        self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 1, self.purpose + ' other'))
        self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 0)
        self.assertTrue(self.coordinator.can_dispatch_inference(1))

    def test_equal_load_pressure_turns_rotate_over_all_zero_based_workers(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        chosen = []
        for _ in range(8):
            self.assertIsNone(self.request())
            chosen.append(self.coordinator.snapshot()['spherical_retirement_reserved_device'])
            self.coordinator.cancel_spherical_retirement_request(self.purpose)
        self.assertEqual(chosen, [0, 1, 2, 3, 0, 1, 2, 3])
        self.coordinator.reset()
        self.coordinator.configure_workers([0, 2])
        self.coordinator.set_pending_inference_backlog(True)
        self.coordinator.set_spherical_retirement_pressure(True)
        for worker in (0, 2):
            self.coordinator.begin_inference(worker)
        chosen = []
        for _ in range(4):
            self.assertIsNone(self.request())
            chosen.append(self.coordinator.snapshot()['spherical_retirement_reserved_device'])
            self.coordinator.cancel_spherical_retirement_request(self.purpose)
        self.assertEqual(chosen, [0, 2, 0, 2])

    def test_failed_driver_query_releases_reservation_and_cools_down(self):
        self.coordinator.set_spherical_retirement_pressure(True)
        self.coordinator.finish_inference(0)
        self.coordinator.finish_inference(0)
        self.torch.cuda.mem_get_info.side_effect = RuntimeError('driver unavailable')
        self.assertIsNone(self.request())
        self.assertTrue(self.coordinator.can_dispatch_inference(0))
        self.assertIsNone(self.request())
        self.assertTrue(self.coordinator.can_dispatch_inference(0))

    def test_failed_wake_callback_cannot_interrupt_request_cleanup(self):
        self.coordinator.set_wake_callback(mock.Mock(side_effect=RuntimeError('closed notifier')))
        self.coordinator.set_spherical_retirement_pressure(True)
        self.assertIsNone(self.request())
        self.coordinator.cancel_spherical_retirement_request(self.purpose)
        self.assertTrue(self.coordinator.can_dispatch_inference(0))

    def test_continuous_below_pressure_demand_ages_into_one_exclusive_drain(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            self.assertIsNone(self.request())
            clock.return_value = 39.999
            self.assertIsNone(self.request())  # Refresh liveness, preserve age.
            self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))
            clock.return_value = 40.
            with mock.patch.object(bp, 'main_process_gpu_stage_inference_overlap_enabled', return_value=True):
                self.assertIsNone(self.request())
                self.assertFalse(self.coordinator.can_dispatch_inference(0))
                self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in (1, 2, 3)))
                self.torch.cuda.mem_get_info.assert_not_called()
                self.coordinator.finish_inference(0)
                self.assertIsNone(self.request())  # Queued inference remains exclusive.
                self.coordinator.finish_inference(0)
                lease = self.request()
                self.assertEqual(lease.device_index, 0)
                self.assertFalse(self.coordinator.snapshot()['spherical_retirement_pressure'])
                self.assertEqual(self.coordinator.snapshot()['spherical_retirement_aged_acquisitions'], 1)
                self.assertEqual(self.coordinator.snapshot()['spherical_retirement_pressure_acquisitions'], 0)
                self.assertFalse(self.coordinator.begin_inference(0))
                lease.release()
                self.assertTrue(self.coordinator.begin_inference(0))

    def test_aged_fifo_handoff_keeps_two_turn_cap_and_inference_fairness(self):
        names = [self.purpose + name for name in (' oldest', ' middle', ' newest')]
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            for name in names:
                self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, name))
            clock.return_value = 35.
            for name in reversed(names):
                self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, name))
            clock.return_value = 40.
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, names[2]))
            self.coordinator.finish_inference(0)
            self.coordinator.finish_inference(0)
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, names[2]))
            first = self.coordinator.try_acquire_stage(self.torch, names[0])
            first.release()
            second = self.coordinator.try_acquire_stage(self.torch, names[1])
            self.assertEqual(second.device_index, 0)
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_handoffs'], 1)
            second.release()
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, names[2]))
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 1)
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_aged_acquisitions'], 2)
            self.assertTrue(self.coordinator.begin_inference(0))

    def test_expired_then_cancelled_below_pressure_demand_must_age_again(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            self.assertIsNone(self.request())
            clock.return_value = 40.
            self.assertIsNone(self.request())  # Expired at the boundary; new age starts now.
            self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))
            clock.return_value = 69.
            self.assertIsNone(self.request())
            self.coordinator.cancel_spherical_retirement_request(self.purpose)
            clock.return_value = 70.
            self.assertIsNone(self.request())
            self.assertIsNone(self.coordinator.snapshot()['spherical_retirement_reserved_device'])
            clock.return_value = 99.
            self.assertIsNone(self.request())
            clock.return_value = 100.
            self.assertIsNone(self.request())
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 0)

    def test_aged_admission_failure_resets_age_and_retains_cooldown(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            self.assertIsNone(self.request())
            clock.return_value = 35.
            self.assertIsNone(self.request())
            clock.return_value = 40.
            self.assertIsNone(self.request())
            self.coordinator.cancel_spherical_retirement_request(self.purpose, failed=True)
            self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))
            clock.return_value = 49.
            self.assertIsNone(self.request())
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_request_count'], 0)
            clock.return_value = 50.
            self.assertIsNone(self.request())
            clock.return_value = 79.
            self.assertIsNone(self.request())
            self.assertIsNone(self.coordinator.snapshot()['spherical_retirement_reserved_device'])
            clock.return_value = 80.
            self.assertIsNone(self.request())
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], 1)

    def test_aged_handoff_requires_an_already_aged_compatible_reader(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            self.assertIsNone(self.request())
            clock.return_value = 35.
            self.assertIsNone(self.request())
            waiting = self.purpose + ' fresh'
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
            clock.return_value = 40.
            self.assertIsNone(self.request())
            self.coordinator.finish_inference(0)
            self.coordinator.finish_inference(0)
            first = self.request()
            first.release()
            self.assertTrue(self.coordinator.begin_inference(0))
            self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, waiting))
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_handoffs'], 0)

    def test_pressure_relief_preserves_aged_but_not_fresh_reservation(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            self.coordinator.set_spherical_retirement_pressure(True)
            self.assertIsNone(self.request())
            self.coordinator.set_spherical_retirement_pressure(False)
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
            clock.return_value = 35.
            self.assertIsNone(self.request())
            clock.return_value = 40.
            self.coordinator.set_spherical_retirement_pressure(True)
            self.assertIsNone(self.request())
            reserved = self.coordinator.snapshot()['spherical_retirement_reserved_device']
            self.coordinator.set_spherical_retirement_pressure(False)
            self.assertFalse(self.coordinator.can_dispatch_inference(reserved))
            self.assertEqual(self.coordinator.snapshot()['spherical_retirement_reserved_device'], reserved)

    def test_terminal_fence_and_auxiliary_owner_still_block_aged_admission(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            self.assertIsNone(self.request())
            clock.return_value = 35.
            self.assertIsNone(self.request())
            clock.return_value = 40.
            self.assertIsNone(self.request())
            self.coordinator.set_inference_asset_retirement_pending(True)
            self.coordinator.finish_inference(0)
            self.coordinator.finish_inference(0)
            self.assertIsNone(self.request())
            self.assertIsNone(self.coordinator.snapshot()['spherical_retirement_reserved_device'])
            self.coordinator.set_inference_asset_retirement_pending(False)
            auxiliary = SimpleNamespace(revoke_worker=mock.Mock(return_value=False))
            with mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=auxiliary):
                self.assertIsNone(self.request())
            self.torch.cuda.mem_get_info.assert_not_called()
            self.assertFalse(self.coordinator.snapshot()['stage_leases'])
            lease = self.request()
            self.assertEqual(lease.device_index, 0)
            lease.release()

    def test_age_optout_preserves_pressure_and_pressure_optout_disables_both(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            with mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT': '0'}):
                self.assertIsNone(self.request())
                for now in (35., 40., 60.):
                    clock.return_value = now
                    self.assertIsNone(self.request())
                    self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))
                self.coordinator.set_spherical_retirement_pressure(True)
                self.assertIsNone(self.request())
                self.assertFalse(self.coordinator.can_dispatch_inference(0))
            with mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT': '0'}):
                self.assertIsNone(self.request())
                self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))
                self.coordinator.set_spherical_retirement_pressure(False)
                self.assertIsNone(self.request())
                self.assertTrue(all(self.coordinator.can_dispatch_inference(i) for i in range(4)))


class SphericalSchedulerPressureTests(unittest.TestCase):
    def test_completed_canvas_pressure_is_published_with_other_family_backlog(self):
        state = _state()
        state.gpu_worker_tasks_by_id[0] = dict(task_id=0, kind='fullframe', model_name='model',
            view=ViewInfo(name='radial', family='radial', num_slices=8, src_h=8, src_w=8, pad_mode='clamp'),
            result_mode='d1_owner', gpu_eligible=True)
        state.gpu_worker_pending_task_ids.append(0)
        backlog, pressure = [], []
        with tempfile.TemporaryDirectory() as directory:
            scheduler = _scheduler(Path(directory), state=state,
                input_overrides={'direct_union_total_dense_byte_limit': 256},
                operation_overrides={'_set_main_process_gpu_pending_inference': backlog.append,
                    '_set_main_process_gpu_spherical_retirement_pressure': pressure.append})
            state.direct_union_postprocess_bytes[('model', 'spherical')] = 191
            scheduler.publish_gpu_worker_admissible_backlog()
            self.assertTrue(backlog[-1]); self.assertFalse(pressure[-1])
            state.direct_union_postprocess_bytes[('model', 'spherical')] = 192
            scheduler.publish_gpu_worker_admissible_backlog()
            self.assertTrue(backlog[-1]); self.assertTrue(pressure[-1])
            # One completed production canvas crosses below the arm threshold;
            # pressure persists so live FIFO demand can use the bounded burst.
            for retained in (191, 180, 161):
                state.direct_union_postprocess_bytes[('model', 'spherical')] = retained
                scheduler.publish_gpu_worker_admissible_backlog()
                self.assertTrue(pressure[-1])
            state.direct_union_postprocess_bytes[('model', 'spherical')] = 160
            scheduler.publish_gpu_worker_admissible_backlog()
            self.assertFalse(pressure[-1])
            state.direct_union_postprocess_bytes[('model', 'spherical')] = 191
            scheduler.publish_gpu_worker_admissible_backlog()
            self.assertFalse(pressure[-1])
            state.direct_union_postprocess_bytes.clear()
            scheduler.publish_gpu_worker_admissible_backlog()
            self.assertFalse(pressure[-1])


if __name__ == '__main__':
    unittest.main()
