"""Join optional task traces into host queue/compute/publication timelines.

These boundaries include Python, rendering, inference submission, required waits,
and result handling. They are not CUDA-event or GPU-kernel durations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


PHASES = {
    'queue_seconds': ('scheduler_dispatch', 'worker_dequeue'),
    'worker_setup_seconds': ('worker_dequeue', 'worker_compute_start'),
    'compute_host_seconds': ('worker_compute_start', 'worker_compute_done'),
    'publication_lag_seconds': ('worker_compute_done', 'worker_publication_done'),
    'result_transport_seconds': ('worker_publication_done', 'scheduler_result_received'),
    'scheduler_wait_seconds': ('scheduler_result_received', 'scheduler_result_handled'),
    'task_elapsed_seconds': ('scheduler_dispatch', 'scheduler_result_received'),
}
TASK_EVENTS = {name for pair in PHASES.values() for name in pair} | {
    'worker_error', 'scheduler_worker_error', 'scheduler_dispatch_error', 'scheduler_compute_released',
}


def read_events(paths):
    """Read batches and reject conflicting event identities; tolerate a torn final line."""
    events, seen, warnings = [], {}, []
    files = sorted({file for path in map(Path, paths) for file in
                    (path.glob('telemetry-*.jsonl') if path.is_dir() else (path,))})
    for path in files:
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append(f'{path}:{line_number}: incomplete or invalid JSON line')
                    continue
                for event in payload.get('events', ()):
                    identity = (event.get('trace_session'), event.get('sequence'))
                    if identity[0] is None or identity[1] is None:
                        raise ValueError(f'{path}:{line_number}: event lacks a session/sequence identity')
                    if identity in seen:
                        if seen[identity] != event:
                            raise ValueError(f'Conflicting trace event {identity}')
                        continue
                    seen[identity] = event
                    events.append(event)
    return events, warnings


def analyze_events(events):
    groups, controls = defaultdict(list), 0
    for event in events:
        name = str(event.get('event', ''))
        task_id = event.get('task_id')
        if (name not in TASK_EVENTS or not isinstance(task_id, int) or isinstance(task_id, bool)
                or task_id < 0 or event.get('replayed')):
            controls += 1
            continue
        groups[(str(event.get('run_id', '')), task_id, str(event.get('device', '')))].append(event)
    tasks = []
    for (run_id, task_id, device), task_events in groups.items():
        stages = defaultdict(list)
        for event in task_events:
            stages[str(event['event'])].append(event)
        identity = next((event for event in task_events if event.get('view')), task_events[0])
        row = {'run_id': run_id, 'task_id': task_id, 'device': device,
               'view': identity.get('view', ''), 'family': identity.get('family', ''),
               'kind': identity.get('kind', ''), 'stage_counts': {key: len(value) for key, value in stages.items()},
               'errors': [event for event in task_events if 'error' in str(event['event'])],
               'timing_issues': []}
        for metric, (first, last) in PHASES.items():
            row[metric] = None
            if len(stages[first]) != 1 or len(stages[last]) != 1:
                row['timing_issues'].append(f'{metric}: missing or repeated boundary')
                continue
            start, stop = stages[first][0], stages[last][0]
            if start.get('hostname') != stop.get('hostname'):
                row['timing_issues'].append(f'{metric}: different hosts; monotonic clocks cannot be joined')
                continue
            value = (int(stop['monotonic_ns']) - int(start['monotonic_ns'])) / 1e9
            row[metric] = value
            if value < 0:
                row['timing_issues'].append(f'{metric}: negative boundary interval')
        tasks.append(row)
    summary = {}
    for metric in PHASES:
        values = sorted(row[metric] for row in tasks if row[metric] is not None and row[metric] >= 0)
        summary[metric] = ({'count': len(values), 'sum': sum(values), 'median': statistics.median(values),
                            'p95': values[min(len(values) - 1, int(.95 * (len(values) - 1)))], 'max': values[-1]}
                           if values else {'count': 0})
    tasks.sort(key=lambda row: (row['run_id'], str(row['task_id']), row['device']))
    return {'schema': 'xta.pipeline-task-trace.v1',
            'interpretation': 'Host task boundary intervals, not GPU kernel durations; summed worker intervals overlap.',
            'task_count': len(tasks), 'control_or_unassigned_events': controls,
            'tasks_with_timing_issues': sum(bool(row['timing_issues']) for row in tasks),
            'summary': summary, 'tasks': tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path, help='Telemetry JSONL files or per-run telemetry directories')
    parser.add_argument('--output', type=Path, help='Write joined JSON to a task-specific Scratch path')
    args = parser.parse_args()
    events, warnings = read_events(args.paths)
    result = analyze_events(events)
    result['warnings'] = warnings
    result['event_count'] = len(events)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
        print(f'{len(events)} events; {result["task_count"]} tasks; '
              f'{result["tasks_with_timing_issues"]} incomplete/ambiguous task timelines. {args.output}')
    else:
        print(text)


if __name__ == '__main__':
    main()
