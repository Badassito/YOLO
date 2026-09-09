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
        patch = mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT': '1'})
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
        self.assertIsNone(self.coordinator.try_acquire_stage(self.torch, self.purpose + ' second'))
        self.assertFalse(self.coordinator.can_dispatch_inference(0))
        lease.release()
        self.assertTrue(self.coordinator.can_dispatch_inference(0))

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
            state.direct_union_postprocess_bytes.clear()
            scheduler.publish_gpu_worker_admissible_backlog()
            self.assertFalse(pressure[-1])


if __name__ == '__main__':
    unittest.main()
