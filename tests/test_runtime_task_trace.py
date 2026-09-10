"""Bounded concurrent host tracing, numeric counters, and cross-process joins."""
import ast
from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import json
import math
import os
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import runtime
from tools.analyze_pipeline_trace import analyze_events, read_events


class RuntimeTaskTraceTests(unittest.TestCase):
    def telemetry(self, directory):
        with mock.patch.dict(os.environ, {'YOLO_TTA_TELEMETRY_DIR': str(directory),
            'YOLO_TTA_TELEMETRY_PATH': 'must-not-use-shared-path.jsonl',
            'YOLO_TTA_TASK_TRACE': '1', 'YOLO_TTA_TELEMETRY': '0'}):
            return runtime.RuntimeTelemetry()

    def test_fractional_seconds_large_integer_counts_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            for value in (.125, np.float64(.25), '0.5'):
                telemetry.add('seconds', value)
            for value in (2**63 + 9, np.uint64(2**63 + 11), '9223372036854775821'):
                telemetry.add('bytes', value)
            for value in (float('nan'), float('inf'), -float('inf'), 'NaN', 'garbage'):
                telemetry.add('seconds', value)
            telemetry.add('huge', 1e308)
            telemetry.add('huge', 1e308)  # Do not overflow an accumulated float to infinity.
            counters = telemetry.snapshot()['counters']
            self.assertEqual(counters['seconds'], .875)
            self.assertEqual(counters['bytes'], 3 * 2**63 + 33)
            self.assertIsInstance(counters['bytes'], int)
            self.assertTrue(math.isfinite(counters['huge']))

    def test_disabled_helper_never_initializes_observability(self):
        with mock.patch.dict(os.environ, {'YOLO_TTA_TASK_TRACE': '0'}), \
                mock.patch.object(runtime, 'runtime_telemetry', side_effect=AssertionError('initialized')):
            runtime.runtime_trace_event('worker_compute_start', task={'task_id': 7})

    def test_directory_precedence_creates_distinct_process_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(runtime.os, 'getpid', return_value=111):
                first = self.telemetry(directory)
            with mock.patch.object(runtime.os, 'getpid', return_value=222):
                second = self.telemetry(directory)
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(first.path.parent, Path(directory))
            self.assertIn('111', first.path.name)
            self.assertIn('222', second.path.name)

    def test_concurrent_flushes_emit_each_event_once_with_bounded_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 7
            gate = threading.Barrier(8)
            def emit(worker):
                gate.wait()
                for index in range(40):
                    telemetry.trace_event('worker_compute_start', task_id=worker * 40 + index,
                                          device=f'cuda:{worker}', family='spherical')
                    if index % 3 == 0:
                        telemetry.flush()
                    self.assertLessEqual(len(telemetry.snapshot()['events']), 7)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(emit, range(8)))
            telemetry.snapshot()  # Inspection must not consume a partial batch.
            telemetry.flush(final=True)
            telemetry.flush(final=True)
            batches = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
            events = [event for batch in batches for event in batch['events']]
            self.assertEqual(len(events), 320)
            self.assertEqual({event['task_id'] for event in events}, set(range(320)))
            self.assertEqual([event['sequence'] for event in events], list(range(1, 321)))
            self.assertTrue(all(len(batch['events']) <= 7 for batch in batches))
            self.assertTrue(all(isinstance(event['wall_time_ns'], int) for event in events))
            self.assertTrue(all(isinstance(event['monotonic_ns'], int) for event in events))
            loaded, warnings = read_events([telemetry.path, telemetry.path])
            self.assertEqual(len(loaded), 320)
            self.assertFalse(warnings)

    def test_event_identity_extracts_no_source_buffers_and_captures_device_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            task = dict(task_id=9, kind='fullframe', view=SimpleNamespace(name='QSC', family='spherical'),
                        source=np.zeros((128, 128), np.uint8), slice_start=11, slice_count=3)
            with mock.patch.dict(os.environ, {'YOLO_TTA_TASK_TRACE': '1', 'CUDA_VISIBLE_DEVICES': '3,7'}), \
                    mock.patch.object(runtime, 'runtime_telemetry', return_value=telemetry):
                runtime.runtime_trace_event('worker_dequeue', task=task, device='cuda:1',
                    accidental_array=task['source'], accidental_container={'pixels': task['source']})
            event = telemetry.snapshot()['events'][0]
            self.assertEqual((event['task_id'], event['view'], event['family']), (9, 'QSC', 'spherical'))
            self.assertEqual(event['cuda_visible_devices'], '3,7')
            self.assertNotIn('source', event)
            self.assertNotIn('accidental_array', event)
            self.assertNotIn('accidental_container', event)

    def test_write_failure_disables_trace_without_unbounded_growth_or_lost_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = self.telemetry(directory)
            telemetry._trace_batch_limit = 2
            with mock.patch.object(Path, 'open', side_effect=OSError('full output device')), \
                    mock.patch('builtins.print'):
                for index in range(20):
                    telemetry.trace_event('worker_dequeue', task_id=index)
            self.assertFalse(telemetry.enabled)
            self.assertEqual(len(telemetry.snapshot()['events']), 2)

    def test_out_of_order_deferred_results_keep_their_own_task_identity(self):
        from XTA import workers
        tree = ast.parse(inspect.getsource(workers._gpu_inference_worker_main))
        names = {'_publish_deferred', '_schedule_deferred_publication'}
        definitions = [node for node in tree.body[0].body if isinstance(node, ast.FunctionDef) and node.name in names]
        program = ast.Module(body=ast.parse('from __future__ import annotations').body + definitions, type_ignores=[])
        events, outgoing = [], queue.Queue()
        namespace = dict(Future=Future, threading=threading, pending_publications=set(),
            publication_condition=threading.Condition(), result_queue=outgoing, gpu_index=2,
            runtime_trace_event=lambda event, **fields: events.append((event, fields)))
        exec(compile(ast.fix_missing_locations(program), '<worker-deferred-trace>', 'exec'), namespace)
        first, second = Future(), Future()
        one = SimpleNamespace(flush_future=first, finish=lambda: first.result())
        two = SimpleNamespace(flush_future=second, finish=lambda: second.result())
        namespace['_schedule_deferred_publication'](7, one, {'view': 'first', 'family': 'radial'})
        namespace['_schedule_deferred_publication'](8, two, {'view': 'second', 'family': 'spherical'})
        second.set_result({'done': True})
        first.set_exception(RuntimeError('failed first publication'))
        self.assertEqual([(name, fields['task_id'], fields['view']) for name, fields in events],
                         [('worker_publication_done', 8, 'second'), ('worker_error', 7, 'first')])
        self.assertEqual([outgoing.get()['task_id'], outgoing.get()['task_id']], [8, 7])
        self.assertFalse(namespace['pending_publications'])

    def test_scheduler_dispatch_receipt_and_compute_credit_have_distinct_events(self):
        from XTA import tta_scheduler
        from tests.test_tta_scheduler_boundary import _scheduler, _state, _view, _bind_callbacks
        state, events = _state(), []
        task = dict(task_id=3, view=_view(), kind='fullframe', model_name='model',
                    slice_start=0, slice_count=3, result_mode='file', gpu_eligible=True)
        state.gpu_worker_tasks_by_id[3] = task
        state.gpu_worker_pending_task_ids.append(3)
        state.gpu_task_queues[1] = queue.Queue()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(tta_scheduler, 'runtime_trace_event',
                    side_effect=lambda event, **fields: events.append((event, fields))):
            scheduler = _scheduler(Path(directory), state=state)
            _bind_callbacks(scheduler)
            scheduler.dispatch_gpu_worker_inference_window()
            dispatched = state.gpu_task_queues[1].get_nowait()
            self.assertEqual(dispatched['task_id'], 3)
            state.gpu_result_queue = queue.Queue()
            state.gpu_result_queue.put(dict(type='compute_released', task_id=3, gpu_index=1, ok=True, stats={}))
            state.gpu_result_queue.put(dict(type='result', task_id=3, gpu_index=1, ok=True, stats={}))
            scheduler.drain_process_inference_results()
        self.assertEqual([event for event, _ in events],
            ['scheduler_dispatch', 'scheduler_compute_released', 'scheduler_message_handled',
             'scheduler_result_received', 'scheduler_result_handled'])
        self.assertTrue(all(fields['device'] == 'cuda:1' for _, fields in events))
        self.assertEqual(state.gpu_worker_results_collected, 1)


class PipelineTraceJoinTests(unittest.TestCase):
    def events(self):
        names = ('scheduler_dispatch', 'worker_dequeue', 'worker_compute_start',
                 'worker_compute_done', 'worker_publication_done', 'scheduler_result_received',
                 'scheduler_result_handled')
        times = (0, 2, 3, 8, 11, 12, 14)
        return [dict(event=name, task_id=3, device='cuda:1', family='spherical', view='face',
                     hostname='node', pid=10 if index in (0, 5, 6) else 20,
                     monotonic_ns=int(stamp * 1e9), wall_time_ns=int((1000 + stamp) * 1e9),
                     run_id='job', trace_session=f'process{10 if index in (0, 5, 6) else 20}', sequence=index)
                for index, (name, stamp) in enumerate(zip(names, times))]

    def test_joins_out_of_order_cross_process_events_and_separates_controls(self):
        events = self.events()
        events.append(dict(event='scheduler_control_received', task_id=-3, device='cuda:1'))
        events.append(dict(events[-2], replayed=True))  # Held D1 result: handling only, not another receipt.
        result = analyze_events(reversed(events))
        self.assertEqual(result['task_count'], 1)
        self.assertEqual(result['control_or_unassigned_events'], 2)
        row = result['tasks'][0]
        self.assertEqual([row[name] for name in ('queue_seconds', 'worker_setup_seconds',
            'compute_host_seconds', 'publication_lag_seconds', 'result_transport_seconds', 'task_elapsed_seconds')],
            [2., 1., 5., 3., 1., 12.])
        self.assertFalse(row['timing_issues'])
        self.assertEqual(row['scheduler_wait_seconds'], 2.)

    def test_missing_repeated_cross_host_and_negative_boundaries_are_explicit(self):
        events = self.events()
        events[1]['hostname'] = 'other node'
        events[4]['monotonic_ns'] = int(7e9)
        events.append(dict(events[0]))
        row = analyze_events(events)['tasks'][0]
        self.assertIsNone(row['queue_seconds'])
        self.assertIsNone(row['worker_setup_seconds'])
        self.assertEqual(row['publication_lag_seconds'], -1.)
        self.assertTrue(row['timing_issues'])

    def test_negative_ids_and_non_task_lifecycle_events_do_not_make_task_rows(self):
        events = self.events() + [
            dict(event='scheduler_message_handled', task_id=-1, device='cuda:0'),
            dict(event='scheduler_worker_error', task_id=-1, device='cuda:0'),
            dict(event='gpu_stage_acquired', task_id=50, device='cuda:0'),
            dict(event='worker_dequeue', task_id=None, device='cuda:0'),
        ]
        result = analyze_events(events)
        self.assertEqual(result['task_count'], 1)
        self.assertEqual(result['control_or_unassigned_events'], 4)
        self.assertFalse(result['tasks'][0]['timing_issues'])

    def test_file_reader_deduplicates_repeated_batches_and_flags_torn_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'telemetry-test.jsonl'
            batch = json.dumps({'events': self.events()})
            path.write_text(batch + '\n' + batch + '\n' + '{"events":[')
            events, warnings = read_events([path])
            self.assertEqual(len(events), 7)
            self.assertEqual(len(warnings), 1)


if __name__ == '__main__':
    unittest.main()
