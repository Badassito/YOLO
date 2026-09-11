"""Summarize full-run LTA host phases and unresolved waits from diagnostic JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.lta_outputs import write_json_atomically


def summarize(directory: Path) -> dict[str, object]:
    streams = []
    partial_lines = 0
    for path in sorted(Path(directory).glob('*.jsonl')):
        phases = {}
        open_phases = {}
        events = 0
        first_ns = last_ns = None
        metadata = {}
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    partial_lines += 1
                    continue
                if record.get('schema') != 'lta.host-phase/1':
                    continue
                events += 1
                now = record.get('monotonic_ns')
                if isinstance(now, int):
                    first_ns = now if first_ns is None else min(first_ns, now)
                    last_ns = now if last_ns is None else max(last_ns, now)
                event = record.get('event')
                if event in ('process_start', 'process_ready', 'run_start'):
                    metadata[event] = record
                key = str(record.get('span_id', (record.get('phase'), record.get('work_id'), record.get('thread_id'))))
                if event == 'phase_start':
                    open_phases[key] = record
                elif event == 'phase_end':
                    open_phases.pop(key, None)
                    phase = str(record.get('phase'))
                    stats = phases.setdefault(phase, {'calls': 0, 'wall_seconds': 0.0, 'maximum_seconds': 0.0, 'failures': 0})
                    seconds = max(0.0, float(record.get('wall_seconds', 0)))
                    stats['calls'] += 1
                    stats['wall_seconds'] += seconds
                    stats['maximum_seconds'] = max(stats['maximum_seconds'], seconds)
                    stats['failures'] += int(record.get('status') == 'failed')
        streams.append({'path': str(path), 'events': events, 'metadata': metadata,
                        'observed_host_seconds': 0.0 if first_ns is None else (last_ns - first_ns) / 1e9,
                        'phases': phases, 'unfinished_phases': list(open_phases.values())})
    return {'schema': 'lta.host-phase-summary/1', 'streams': streams, 'incomplete_json_lines': partial_lines,
            'timing_semantics': 'Host wall time; nested phases overlap and must not be summed as CUDA utilization.',
            'unfinished_semantics': 'An unfinished phase can be currently running, buffered, interrupted, or killed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    if args.output is not None:
        write_json_atomically(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
