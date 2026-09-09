"""Exercise the real scheduler with descriptor-only Spherical work and fake GPUs.

Reports cache-plan changes, coverage, and bounded admission for the production
120-parent/22-lease pattern. Simulated walltime is an explicit cost model, not a
hardware benchmark; no model, CUDA allocation, or native canvas is constructed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import heapq
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_spherical_locality import locality_scheduler, complete_synthetic_task


def simulate(*, enabled, parents=120, speeds=(1., 1., 1., 1.), view_limit=4,
             seconds_per_frame=.01, plan_seconds=.909911):
    scheduler = locality_scheduler(parents=parents, leases=22, workers=len(speeds),
                                   enabled=enabled, view_limit=view_limit)
    state = scheduler.state
    for parent, indexed in state.fullframe_task_ids_by_parent.items():
        first = state.gpu_worker_tasks_by_id[indexed[0]]
        view = replace(first['view'], num_slices=1212)
        for task_id in indexed:
            task = state.gpu_worker_tasks_by_id[task_id]
            task['view'] = view
            task['slice_count'] = min(57, 1212 - task['slice_start'])
            task['processing_shape'] = (1212, 3072, 3072)
    events, running, last, completed = [], set(), {}, []
    now, misses, frames, max_active, max_bytes, max_queue = 0., 0, 0, 0, 0, 0
    started = time.perf_counter()

    def dispatch_and_start(preferred_parent=None):
        nonlocal misses, frames, max_active, max_bytes, max_queue
        scheduler.dispatch_gpu_worker_inference_window(preferred_parent)
        max_active = max(max_active, len(state.direct_union_inference_views))
        max_bytes = max(max_bytes, sum(state.direct_union_inference_bytes.values()))
        max_queue = max(max_queue, *(scheduler.gpu_worker_inflight(worker) for worker in state.gpu_task_queues))
        for worker, work in state.gpu_task_queues.items():
            if worker in running or work.empty():
                continue
            task = work.get_nowait()
            parent = scheduler.gpu_worker_fullframe_parent_key(task)
            cold = last.get(worker) != parent
            misses += int(cold)
            frames += task['slice_count']
            last[worker] = parent
            seconds = (seconds_per_frame * task['slice_count'] + plan_seconds * cold) / speeds[worker]
            heapq.heappush(events, (now + seconds, worker, task['task_id']))
            running.add(worker)

    dispatch_and_start()
    while events:
        now, worker, task_id = heapq.heappop(events)
        running.remove(worker)
        completed.append(task_id)
        task = state.gpu_worker_tasks_by_id[task_id]
        complete_synthetic_task(scheduler, worker, task)
        dispatch_and_start(scheduler.gpu_worker_fullframe_parent_key(task))
    expected = set(state.gpu_worker_tasks_by_id)
    if set(completed) != expected or len(completed) != len(expected) or state.gpu_worker_pending_task_ids:
        raise AssertionError('Synthetic dispatch lost, duplicated, or stranded inference work')
    return {
        'enabled': enabled, 'parents': parents, 'workers': len(speeds), 'worker_speed_factors': speeds,
        'view_limit': view_limit, 'tasks_completed': len(completed), 'frames': frames,
        'plan_misses': misses, 'max_active_parents': max_active, 'max_inference_canvas_bytes': max_bytes,
        'max_inflight_tasks_per_worker': max_queue, 'final_hint_entries': len(state.spherical_render_parent_by_worker),
        'simulated_seconds': now, 'host_simulation_seconds': time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parents', type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.parents <= 120:
        parser.error('parents must be from 1 through 120')
    results = {
        'description': 'Real scheduler, synthetic equal 1212-frame Spherical parents; no CUDA or mask data.',
        'seconds_per_frame_assumption': .01,
        'plan_seconds_assumption': .909911,
        'plan_seconds_source': '142812 mean 0.883487s build plus 0.026424s upload',
        'limitations': 'Cost-model walltime is not a cluster prediction; fake results publish immediately and retire canvases immediately.',
        'cases': {},
    }
    for name, options in {
        'equal_workers': {},
        'unequal_workers': {'speeds': (.85, 1., 1.1, 1.2)},
        'one_parent_budget': {'view_limit': 1},
    }.items():
        baseline = simulate(enabled=False, parents=args.parents, **options)
        locality = simulate(enabled=True, parents=args.parents, **options)
        results['cases'][name] = {'baseline': baseline, 'locality': locality,
                                 'avoided_plan_builds': baseline['plan_misses'] - locality['plan_misses'],
                                 'simulated_seconds_saved': baseline['simulated_seconds'] - locality['simulated_seconds']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
