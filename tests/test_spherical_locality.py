"""Spherical cache locality changes placement without acquiring extra leases."""
from dataclasses import replace
from pathlib import Path
import queue
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


def locality_scheduler(*, parents=8, leases=6, workers=4, view_limit=4, enabled=True,
                       byte_limit=128 * 1024**3, blocked_workers=()):
    """Use real admission/selection/dispatch with descriptor-only fake workspaces."""
    state = _state()
    for worker in range(workers):
        state.gpu_task_queues[worker] = queue.Queue()
    task_id = 0
    for parent in range(parents):
        view = replace(_view(), name=f'spherical_parent_{parent}', family='spherical',
                       num_slices=leases * 57, src_h=3072, src_w=3072)
        key = ('model', view.name)
        state.fullframe_remaining[key] = leases
        state.fullframe_task_ids_by_parent[key] = []
        for lease in range(leases):
            state.gpu_worker_tasks_by_id[task_id] = {
                'task_id': task_id, 'kind': 'fullframe', 'model_name': 'model',
                'view': view, 'slice_start': lease * 57, 'slice_count': 57,
                'processing_shape': (view.num_slices, view.src_h, view.src_w),
                'result_mode': 'direct_union', 'gpu_eligible': True,
                'disable_runtime_split': True,
            }
            state.gpu_worker_pending_task_ids.append(task_id)
            state.fullframe_task_ids_by_parent[key].append(task_id)
            task_id += 1
    state.gpu_worker_total_tasks = task_id
    state.gpu_worker_next_dynamic_task_id = task_id
    holder = {}

    def admit(model, view):
        parent = (model, view.name)
        if parent not in state.direct_union_inference_views:
            first = state.gpu_worker_tasks_by_id[state.fullframe_task_ids_by_parent[parent][0]]
            state.direct_union_inference_views.add(parent)
            state.direct_union_inference_bytes[parent] = holder['scheduler'].direct_union_task_bytes(first)
            state.direct_union_backing_leases[parent] = SimpleNamespace(phase='inference')
            state.baseline_union_paths[parent] = Path('synthetic-spherical-mask.dat')
            state.baseline_confmap_paths[parent] = None

    scheduler = _scheduler(
        Path('.'), state=state,
        input_overrides={
            'gpu_device_count': workers, 'direct_union_inference_view_limit': view_limit,
            'direct_union_inference_byte_limit': byte_limit,
            'direct_union_total_dense_byte_limit': 256 * 1024**3,
            'ensure_baseline_workspaces': admit,
        },
        operation_overrides={
            '_env_int': lambda name, default: int(enabled) if name == 'YOLO_TTA_GPU_SPHERICAL_LOCALITY' else default,
            '_main_process_gpu_stage_can_dispatch_inference': lambda worker: worker not in blocked_workers,
            'gpu_worker_default_seconds_per_frame': lambda _view: .016,
        },
    )
    holder['scheduler'] = scheduler
    return scheduler


def complete_synthetic_task(scheduler, worker, task):
    """Publish one complete fake result and release its existing inference credit."""
    state = scheduler.state
    task_id = task['task_id']
    state.gpu_worker_compute_completed_by_id[worker] = state.gpu_worker_compute_completed_by_id.get(worker, 0) + 1
    seconds = state.gpu_worker_task_predicted_seconds_by_id.pop(task_id)
    state.gpu_worker_predicted_load_by_id[worker] = max(0., state.gpu_worker_predicted_load_by_id[worker] - seconds)
    parent = scheduler.gpu_worker_fullframe_parent_key(task)
    state.fullframe_remaining[parent] -= 1
    if state.fullframe_remaining[parent] == 0:
        state.direct_union_inference_views.remove(parent)
        state.direct_union_inference_bytes.pop(parent)
        state.direct_union_backing_leases.pop(parent)


class SphericalLocalityTests(unittest.TestCase):
    def test_four_workers_start_distinct_parents_with_existing_admission_limits(self):
        scheduler = locality_scheduler()
        scheduler.dispatch_gpu_worker_inference_window()
        parents = []
        for worker, work in scheduler.state.gpu_task_queues.items():
            self.assertEqual(work.qsize(), 2)
            first, second = work.get_nowait(), work.get_nowait()
            self.assertEqual(first['view'].name, second['view'].name)
            parents.append(first['view'].name)
        self.assertEqual(len(set(parents)), 4)
        self.assertEqual(len(scheduler.state.direct_union_inference_views), 4)
        self.assertLessEqual(sum(scheduler.state.direct_union_inference_bytes.values()),
                             scheduler.inputs.direct_union_inference_byte_limit)

    def test_disabled_gate_preserves_shared_parent_dispatch(self):
        scheduler = locality_scheduler(enabled=False)
        scheduler.dispatch_gpu_worker_inference_window()
        names = [work.get_nowait()['view'].name for work in scheduler.state.gpu_task_queues.values()]
        self.assertEqual(len(set(names)), 1)

    def test_full_run_reuses_plans_and_covers_every_lease_exactly_once(self):
        scheduler = locality_scheduler(parents=12, leases=8)
        state = scheduler.state
        completed, plan_switches, cache = [], 0, {}
        while state.gpu_worker_pending_task_ids or any(not work.empty() for work in state.gpu_task_queues.values()):
            scheduler.dispatch_gpu_worker_inference_window()
            progress = False
            for worker, work in state.gpu_task_queues.items():
                if work.empty():
                    continue
                task = work.get_nowait()
                parent = scheduler.gpu_worker_fullframe_parent_key(task)
                plan_switches += int(cache.get(worker) != parent)
                cache[worker] = parent
                completed.append(task['task_id'])
                complete_synthetic_task(scheduler, worker, task)
                progress = True
            self.assertTrue(progress, 'admissible work was stranded')
            self.assertLessEqual(len(state.direct_union_inference_views), 4)
            self.assertLessEqual(len(state.spherical_render_parent_by_worker), 4)
        self.assertEqual(sorted(completed), list(range(96)))
        self.assertEqual(plan_switches, 12)
        self.assertFalse(state.direct_union_inference_views)
        self.assertTrue(all(remaining == 0 for remaining in state.fullframe_remaining.values()))

    def test_view_and_byte_pressure_fall_back_to_sharing_without_idle_workers(self):
        for options in ({'view_limit': 1}, {'byte_limit': 1}):
            scheduler = locality_scheduler(leases=22, **options)
            scheduler.dispatch_gpu_worker_inference_window()
            with self.subTest(options=options):
                self.assertEqual(len(scheduler.state.direct_union_inference_views), 1)
                self.assertTrue(all(work.qsize() == 2 for work in scheduler.state.gpu_task_queues.values()))
                self.assertEqual(len(set(scheduler.state.spherical_render_parent_by_worker.values())), 1)

    def test_fewer_parents_than_workers_share_the_tail(self):
        scheduler = locality_scheduler(parents=2)
        scheduler.dispatch_gpu_worker_inference_window()
        self.assertTrue(all(work.qsize() == 2 for work in scheduler.state.gpu_task_queues.values()))
        self.assertEqual(len(scheduler.state.direct_union_inference_views), 2)

    def test_preferred_parent_last_lease_keeps_unlock_priority_over_warm_work(self):
        for enabled in (False, True):
            scheduler = locality_scheduler(parents=2, leases=2, workers=2, enabled=enabled)
            parent = ('model', 'spherical_parent_0')
            other = ('model', 'spherical_parent_1')
            scheduler.state.gpu_worker_pending_task_ids.remove(1)
            scheduler.state.fullframe_remaining[parent] = 1
            scheduler.state.spherical_render_parent_by_worker.update({0: parent, 1: other})
            # The preferred parent's old worker is busy/retiring. Reusing the
            # other worker's plan must not hide a lease that unlocks parent RAM.
            selected = scheduler.pop_gpu_worker_pending_task_id(parent, [1])
            with self.subTest(enabled=enabled):
                self.assertEqual(selected, (0, [1]))

    def test_preferred_parent_unlock_keeps_all_original_feasible_workers(self):
        scheduler = locality_scheduler(parents=2, leases=2, workers=2)
        parent = ('model', 'spherical_parent_0')
        scheduler.state.gpu_worker_pending_task_ids.remove(1)
        scheduler.state.spherical_render_parent_by_worker[0] = parent
        selected = scheduler.pop_gpu_worker_pending_task_id(parent, [0, 1])
        self.assertEqual(selected, (0, [0, 1]))

    def test_retiring_or_d1_owned_worker_never_receives_locality_work(self):
        for blocked, d1 in (((0,), False), ((), True)):
            scheduler = locality_scheduler(parents=1, leases=10, blocked_workers=blocked)
            scheduler.state.spherical_render_parent_by_worker[0] = ('model', 'spherical_parent_0')
            if d1:
                scheduler.inputs = replace(scheduler.inputs, v1613_d1_owner_active=True)
                scheduler.state.d1_active_parent_by_worker[0] = ('model', 'cartesian_owned')
            scheduler.dispatch_gpu_worker_inference_window()
            with self.subTest(blocked=blocked, d1=d1):
                self.assertTrue(scheduler.state.gpu_task_queues[0].empty())
                self.assertTrue(all(scheduler.state.gpu_task_queues[worker].qsize() == 2 for worker in (1, 2, 3)))

    def test_hybrid_and_non_spherical_selection_is_unchanged(self):
        scheduler = locality_scheduler(parents=1, leases=2)
        task0, task1 = scheduler.state.gpu_worker_tasks_by_id.values()
        task0['hybrid_cpu_eligible_origin'] = True
        task1['view'] = replace(task1['view'], family='radial')
        pool, feasible = [(0, 0), (1, 1)], {0: [0, 1], 1: [2, 3]}
        after_pool, after_feasible = scheduler.prefer_spherical_locality(pool, feasible)
        self.assertIs(after_pool, pool)
        self.assertIs(after_feasible, feasible)

    def test_uncached_oversized_patches_keep_original_worker_selection(self):
        scheduler = locality_scheduler(parents=1, leases=2)
        for task in scheduler.state.gpu_worker_tasks_by_id.values():
            task['view'] = replace(task['view'], src_h=4096, src_w=4096)
        pool, feasible = [(0, 0), (1, 1)], {0: [0, 1], 1: [2, 3]}
        after_pool, after_feasible = scheduler.prefer_spherical_locality(pool, feasible)
        self.assertIs(after_pool, pool)
        self.assertIs(after_feasible, feasible)

    def test_failed_queue_put_cannot_publish_a_cache_hint(self):
        scheduler = locality_scheduler(parents=1, workers=1)
        scheduler.state.gpu_task_queues[0] = mock.Mock()
        scheduler.state.gpu_task_queues[0].put.side_effect = RuntimeError('queue unavailable')
        with self.assertRaisesRegex(RuntimeError, 'queue unavailable'):
            scheduler.dispatch_gpu_worker_inference_window()
        self.assertFalse(scheduler.state.spherical_render_parent_by_worker)
        self.assertEqual(scheduler.state.gpu_worker_dispatched_tasks, 0)
        self.assertEqual(len(scheduler.state.gpu_worker_pending_task_ids), 6)

    def test_runtime_split_inherits_parent_affinity_and_exact_ranges(self):
        scheduler = locality_scheduler(parents=4, leases=1)
        for task in scheduler.state.gpu_worker_tasks_by_id.values():
            task['slice_count'] = 128
            task['disable_runtime_split'] = False
        scheduler.dispatch_gpu_worker_inference_window()
        for worker, work in scheduler.state.gpu_task_queues.items():
            self.assertEqual(work.qsize(), 2)
            first, second = work.get_nowait(), work.get_nowait()
            self.assertEqual(first['view'].name, second['view'].name)
            self.assertEqual(first['slice_start'], 0)
            self.assertEqual(first['slice_count'], second['slice_start'])
            self.assertEqual(first['slice_count'] + second['slice_count'], 128)


if __name__ == '__main__':
    unittest.main()
