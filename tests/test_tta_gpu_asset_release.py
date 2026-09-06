"""Post-inference GPU control accounting, ACK barriers and auxiliary admission."""
from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_tta_scheduler_boundary import _bind_callbacks, _scheduler, _state


class _AuxPool:
    def __init__(self):
        self.enabled = set()
        self.failed = []

    def revoke_worker(self, worker):
        self.enabled.discard(worker)
        return True

    def enable_worker(self, worker, *, allow_full_cpu_affinity=False):
        if not allow_full_cpu_affinity:
            raise AssertionError('Post-inference leases must restore worker affinity')
        self.enabled.add(worker)

    def outstanding(self):
        return 0

    def mark_failed(self, reason):
        self.failed.append(reason)


def _ready_state():
    state = _state()
    state.gpu_task_queues = {0: queue.Queue(), 2: queue.Queue()}
    state.gpu_worker_total_tasks = 2
    state.gpu_worker_results_collected = 2
    state.gpu_worker_dispatched_tasks = 2
    state.gpu_worker_dispatched_by_id = {0: 1, 2: 1}
    state.gpu_worker_results_by_id = {0: 1, 2: 1}
    state.gpu_worker_compute_completed_by_id = {0: 1, 2: 1}
    state.gpu_frames_completed_total = 16
    return state


def _ack(worker, command, *, ok=True, intact=False):
    return {'type': 'inference_assets_released', 'op': 'release_inference_assets',
            'gpu_index': worker, 'task_id': command['task_id'], 'ok': ok,
            'stats': {'assets_intact': intact, 'phase': 'validate_drain' if intact else 'release'},
            'error': '' if ok else 'retained active cache'}


class GpuAssetReleaseSchedulerTests(unittest.TestCase):
    def test_control_ids_ack_barrier_aux_leases_and_inference_counts(self):
        state = _ready_state(); aux = _AuxPool(); priority = [True]
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = _scheduler(Path(tmp), state=state, operation_overrides={
                'gpu_worker_aux_interpolation_pool': lambda: aux,
            })
            callback = mock.Mock(side_effect=lambda: priority.__setitem__(0, not scheduler.request_gpu_inference_asset_release()))
            _bind_callbacks(scheduler, announce=callback)
            previous_control = scheduler._next_d1_group_control_task_id()
            before = (state.gpu_worker_results_collected, dict(state.gpu_worker_results_by_id),
                      dict(state.gpu_worker_compute_completed_by_id), state.gpu_frames_completed_total)
            scheduler.refresh_gpu_aux_interpolation_leases()
            self.assertEqual(aux.enabled, set())
            self.assertFalse(scheduler.request_gpu_inference_asset_release())
            commands = {worker: task_queue.get_nowait() for worker, task_queue in state.gpu_task_queues.items()}
            self.assertEqual([commands[w]['task_id'] for w in (0, 2)], [previous_control-1, previous_control-2])
            self.assertTrue(all(command['inference_drained'] for command in commands.values()))
            self.assertTrue(all(command['task_type'] == 'control' for command in commands.values()))
            self.assertTrue(scheduler.process_inference_outstanding())
            self.assertIn('gpu_inference_asset_release_pending', scheduler.process_quiescence_issues())
            self.assertFalse(scheduler.request_gpu_inference_asset_release())
            self.assertTrue(all(task_queue.empty() for task_queue in state.gpu_task_queues.values()))

            scheduler.process_one_worker_result(_ack(2, commands[2]))
            self.assertEqual(aux.enabled, {2})
            self.assertTrue(priority[0])
            callback.assert_not_called()
            scheduler.process_one_worker_result(_ack(0, commands[0], ok=False, intact=True))
            self.assertEqual(aux.enabled, {0, 2})
            self.assertFalse(priority[0])
            callback.assert_called_once()
            self.assertTrue(scheduler.gpu_inference_asset_release_complete())
            self.assertFalse(scheduler.process_inference_outstanding())
            self.assertNotIn('gpu_inference_asset_release_pending', scheduler.process_quiescence_issues())
            self.assertFalse(state.gpu_inference_asset_release_results_by_worker[0]['ok'])
            after = (state.gpu_worker_results_collected, dict(state.gpu_worker_results_by_id),
                     dict(state.gpu_worker_compute_completed_by_id), state.gpu_frames_completed_total)
            self.assertEqual(after, before)
            scheduler.process_one_worker_result(_ack(2, commands[2]))
            callback.assert_called_once()

    def test_no_command_before_authoritative_drain_and_owner_release(self):
        mutations = [
            lambda s: setattr(s, 'gpu_worker_results_collected', 1),
            lambda s: s.gpu_worker_pending_task_ids.append(7),
            lambda s: s.d1_owner_by_parent.update({('model', 'view'): 0}),
            lambda s: s.d1_active_parent_by_worker.update({0: ('model', 'view')}),
            lambda s: s.d1_groups_by_parent.update({('model', 'view'): object()}),
            lambda s: s.d1_group_parent_by_id.update({'group': ('model', 'view')}),
            lambda s: s.gpu_worker_compute_completed_by_id.update({0: 0}),
            lambda s: (s.cpu_task_queues.update({0: queue.Queue()}), s.cpu_worker_dispatched_by_id.update({0: 1})),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for mutation in mutations:
                state = _ready_state(); mutation(state)
                scheduler = _scheduler(Path(tmp), state=state)
                self.assertFalse(scheduler.request_gpu_inference_asset_release())
                self.assertFalse(state.gpu_inference_asset_release_requested)
                self.assertTrue(all(task_queue.empty() for task_queue in state.gpu_task_queues.values()))
            state = _ready_state(); state.gpu_worker_results_collected = 3
            with self.assertRaisesRegex(RuntimeError, 'exceeded'):
                _scheduler(Path(tmp), state=state).request_gpu_inference_asset_release()

    def test_empty_cpu_only_and_unregistered_task_runs_do_not_retire_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for queues in ({}, {0: queue.Queue()}):
                state = _state(); state.gpu_task_queues = queues
                scheduler = _scheduler(Path(tmp), state=state)
                self.assertTrue(scheduler.request_gpu_inference_asset_release())
                self.assertTrue(scheduler.gpu_inference_asset_release_complete())
                self.assertFalse(state.gpu_inference_asset_release_requested)
                self.assertTrue(all(task_queue.empty() for task_queue in queues.values()))

    def test_bad_and_unrequested_acks_never_enter_inference_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = _ready_state(); scheduler = _scheduler(Path(tmp), state=state)
            _bind_callbacks(scheduler)
            with self.assertRaisesRegex(RuntimeError, 'Unrequested'):
                scheduler.process_one_worker_result(_ack(0, {'task_id': -1}))
            scheduler.request_gpu_inference_asset_release()
            command = state.gpu_task_queues[0].get_nowait()
            mutations = [dict(gpu_index=7), dict(task_id=-999), dict(task_id=0),
                         dict(op='other'), dict(worker_kind='cpu')]
            pending = dict(state.gpu_inference_asset_release_pending_by_worker)
            for mutation in mutations:
                message = _ack(0, command); message.update(mutation)
                with self.assertRaises(RuntimeError):scheduler.process_one_worker_result(message)
                self.assertEqual(state.gpu_worker_results_collected, 2)
                self.assertEqual(state.gpu_inference_asset_release_pending_by_worker, pending)

    def test_partial_release_failure_is_fatal_and_leaves_barrier_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = _ready_state(); scheduler = _scheduler(Path(tmp), state=state)
            _full, _tile, announce, _affinity = _bind_callbacks(scheduler)
            scheduler.request_gpu_inference_asset_release()
            command = state.gpu_task_queues[0].get_nowait()
            with self.assertRaisesRegex(RuntimeError, 'failed during inference-asset release'):
                scheduler.process_one_worker_result(_ack(0, command, ok=False, intact=False))
            self.assertIn(0, state.gpu_inference_asset_release_pending_by_worker)
            self.assertNotIn(0, state.gpu_inference_asset_release_results_by_worker)
            self.assertFalse(scheduler.gpu_inference_asset_release_complete())
            self.assertTrue(scheduler.process_inference_outstanding())
            announce.assert_not_called()

    def test_fence_failure_is_fatal_even_when_assets_are_still_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = _ready_state(); scheduler = _scheduler(Path(tmp), state=state)
            _full, _tile, announce, _affinity = _bind_callbacks(scheduler)
            scheduler.request_gpu_inference_asset_release()
            command = state.gpu_task_queues[0].get_nowait()
            message = _ack(0, command, ok=False, intact=True)
            message['stats']['phase'] = 'fence_inference_streams'
            with self.assertRaisesRegex(RuntimeError, 'failed during inference-asset release'):
                scheduler.process_one_worker_result(message)
            self.assertIn(0, state.gpu_inference_asset_release_pending_by_worker)
            announce.assert_not_called()

    def test_queue_failure_cannot_complete_or_reissue_the_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = _ready_state()
            state.gpu_task_queues[2] = mock.Mock()
            state.gpu_task_queues[2].put.side_effect = OSError('closed task queue')
            scheduler = _scheduler(Path(tmp), state=state)
            with self.assertRaisesRegex(OSError, 'closed task queue'):
                scheduler.request_gpu_inference_asset_release()
            self.assertTrue(state.gpu_inference_asset_release_requested)
            self.assertEqual(set(state.gpu_inference_asset_release_pending_by_worker), {0, 2})
            self.assertFalse(scheduler.request_gpu_inference_asset_release())
            state.gpu_task_queues[2].put.assert_called_once()

    def test_worker_death_during_ack_wait_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = _ready_state()
            state.gpu_worker_processes = [mock.Mock(name='dead-gpu')]
            state.gpu_worker_processes[0].is_alive.return_value = False
            scheduler = _scheduler(Path(tmp), state=state)
            _bind_callbacks(scheduler)
            scheduler.request_gpu_inference_asset_release()
            with self.assertRaisesRegex(RuntimeError, 'inference-asset release acknowledgement'):
                scheduler.check_inference_workers_alive()

    def test_late_inference_cannot_be_dispatched_to_retiring_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = _ready_state(); scheduler = _scheduler(Path(tmp), state=state)
            scheduler.request_gpu_inference_asset_release()
            state.gpu_worker_pending_task_ids.append(7)
            with self.assertRaisesRegex(RuntimeError, 'New inference was queued'):
                scheduler.dispatch_gpu_worker_inference_window()


if __name__ == '__main__':
    unittest.main()
