"""Production-shaped native parent admission without allocating production data."""
from __future__ import annotations

import ast
import contextlib
import io
import os
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.test_tta_scheduler_boundary import _bind_callbacks, _scheduler, _state
from tests.test_terminal_component_refs import _function
from XTA import pipeline, publication_memory, runtime, workers
from XTA.geometry import ViewInfo
from XTA.interpolation import _DirectUnionBackingLease

GIB = 1024 ** 3
SHAPE = (1212, 3072, 3072)
PARENT_BYTES = 1212 * 3072 * 3072


def native_tasks(count=120, mode='direct_union'):
    tasks = []
    for parent in range(count):
        view = ViewInfo(name=f'spherical_{parent}__tta_a0', family='spherical',
                        num_slices=1212, src_h=3072, src_w=3072, pad_mode='clamp')
        for chunk in range(4):
            tasks.append(dict(task_id=len(tasks), kind='fullframe', model_name='model',
                              view=view, result_mode=mode, processing_shape=SHAPE,
                              slice_start=chunk * 303, slice_count=303,
                              union_num_slices=1212, gpu_eligible=True,
                              disable_runtime_split=True))
    return tasks


class SphericalHostAdmissionTests(unittest.TestCase):
    def test_d1_activation_preserves_explicit_shared_union_capability(self):
        tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == 'v1613_d1_owner_active'
                     and any(isinstance(child, ast.Constant) and isinstance(child.value, str)
                             and 'fast bundle active:' in child.value for child in ast.walk(node)))
        code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                       '<D1 activation>', 'exec')
        for enabled in (False, True):
            env = dict(v1613_d1_owner_active=True, gpu_worker_direct_union_active=enabled,
                       legacy_d1_model_eligible=True)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, env)
            self.assertIs(env['gpu_worker_direct_union_active'], enabled)

    def test_120_native_parents_finish_on_four_workers_with_retirement_backpressure(self):
        state = _state()
        tasks = native_tasks()
        for task in tasks:
            index = task['task_id']
            key = ('model', task['view'].name)
            state.gpu_worker_tasks_by_id[index] = task
            state.fullframe_task_ids_by_parent.setdefault(key, []).append(index)
            state.fullframe_remaining[key] = state.fullframe_remaining.get(key, 0) + 1
        state.gpu_worker_pending_task_ids.extend(range(len(tasks)))
        state.gpu_worker_total_tasks = len(tasks)
        state.gpu_worker_next_dynamic_task_id = len(tasks)
        state.gpu_task_queues.update({worker: queue.Queue() for worker in range(4)})
        admissions, retired, used_workers, backlog = [], [], set(), []
        peak_inference = peak_bytes = 0

        def ensure(model, view):
            key = (model, view.name)
            if key in state.direct_union_backing_leases:
                return
            admissions.append(key)
            state.baseline_union_paths[key] = Path('metadata-only') / view.name
            state.direct_union_backing_leases[key] = _DirectUnionBackingLease(key, PARENT_BYTES)
            state.direct_union_inference_views.add(key)
            state.direct_union_inference_bytes[key] = PARENT_BYTES

        def complete(task, stats):
            key = ('model', task['view'].name)
            state.fullframe_remaining[key] -= 1
            if state.fullframe_remaining[key] == 0:
                lease = state.direct_union_backing_leases[key]
                lease.transition('inference', 'postprocess')
                state.direct_union_inference_views.remove(key)
                state.direct_union_inference_bytes.pop(key)
                state.direct_union_postprocess_views.add(key)
                state.direct_union_postprocess_bytes[key] = lease.nbytes

        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            scheduler = _scheduler(Path(td), state=state,
                input_overrides=dict(gpu_device_count=4, imgsz=3072, v1613_d1_owner_active=True,
                    ensure_baseline_workspaces=ensure, direct_union_inference_view_limit=4,
                    direct_union_inference_byte_limit=128 * GIB,
                    direct_union_total_dense_byte_limit=256 * GIB),
                operation_overrides={'_set_main_process_gpu_pending_inference': backlog.append})
            _bind_callbacks(scheduler, fullframe=mock.Mock(side_effect=complete))
            scheduler.dispatch_gpu_worker_inference_window()
            self.assertTrue(all(not q.empty() for q in state.gpu_task_queues.values()))
            # Active-parent priority fills the first four workers from one trajectory.
            self.assertEqual({q.queue[0]['view'].name for q in state.gpu_task_queues.values()},
                             {tasks[0]['view'].name})
            pressure_stops = 0
            while state.gpu_worker_pending_task_ids or any(not q.empty() for q in state.gpu_task_queues.values()):
                progressed = False
                for worker, q in state.gpu_task_queues.items():
                    if q.empty():
                        continue
                    task = q.get_nowait()
                    used_workers.add(worker)
                    scheduler.process_one_worker_result(dict(type='result', gpu_index=worker,
                        task_id=task['task_id'], ok=True, stats={}))
                    progressed = True
                    peak_inference = max(peak_inference, len(state.direct_union_inference_views))
                    live = sum(state.direct_union_inference_bytes.values()) + sum(state.direct_union_postprocess_bytes.values())
                    peak_bytes = max(peak_bytes, live)
                    self.assertLessEqual(len(state.direct_union_inference_views), 4)
                    self.assertLessEqual(live, 256 * GIB)
                if not progressed:
                    pressure_stops += 1
                    self.assertTrue(state.direct_union_postprocess_views)
                    self.assertFalse(state.direct_union_inference_views)
                    self.assertFalse(backlog[-1])
                    # Finished inference does not free the native backing. Retirement
                    # returns credit and the same scheduler can immediately dispatch.
                    for key in list(state.direct_union_postprocess_views):
                        lease = state.direct_union_backing_leases.pop(key)
                        lease.release('postprocess')
                        state.direct_union_postprocess_views.remove(key)
                        state.direct_union_postprocess_bytes.pop(key)
                        retired.append(key)
                    scheduler.dispatch_gpu_worker_inference_window()
                    self.assertTrue(any(not q.empty() for q in state.gpu_task_queues.values()))
            for key in list(state.direct_union_postprocess_views):
                state.direct_union_backing_leases.pop(key).release('postprocess')
                state.direct_union_postprocess_views.remove(key)
                state.direct_union_postprocess_bytes.pop(key)
                retired.append(key)
        self.assertGreater(pressure_stops, 0)
        self.assertGreater(peak_bytes, 128 * GIB)
        self.assertLessEqual(peak_inference, 4)
        self.assertEqual(used_workers, set(range(4)))
        self.assertEqual(state.gpu_worker_results_collected, 480)
        self.assertEqual(len(set(admissions)), 120)
        self.assertEqual(set(retired), set(admissions))
        self.assertFalse(state.direct_union_backing_leases)
        self.assertTrue(all(value == 0 for value in state.fullframe_remaining.values()))
        self.assertEqual(scheduler.process_quiescence_issues(), {})

    def test_native_dense_window_reduces_publication_budget_without_spherical_grants(self):
        reserve = publication_memory.native_fullframe_dense_reserve(native_tasks(), total_dense_limit=256 * GIB)
        self.assertEqual(reserve, 256 * GIB)
        shape = (1931, 3064, 3022)
        plain = publication_memory.retained_payload_plan([shape] * 50, 975 * GIB, 4, 12, 256 * 1024 ** 2)
        bounded = publication_memory.retained_payload_plan([shape] * 50, 975 * GIB, 4, 12, 256 * 1024 ** 2,
                                                            native_dense_reserve_bytes=reserve)
        self.assertEqual(len(bounded['grants']), 50)
        self.assertEqual(bounded['reserve_bytes'] - plain['reserve_bytes'], reserve)
        self.assertEqual(plain['budget_bytes'] - bounded['budget_bytes'], reserve // 2)

    def test_explicit_file_optout_allows_small_work_and_refuses_unbounded_native_parents(self):
        with self.assertRaisesRegex(RuntimeError, 'File-mode.*1278.3 GiB'):
            publication_memory.native_fullframe_dense_reserve(native_tasks(mode='file'), total_dense_limit=256 * GIB)
        self.assertEqual(publication_memory.native_fullframe_dense_reserve(native_tasks(2, 'file'),
                                                                          total_dense_limit=256 * GIB), 2 * PARENT_BYTES)
        # A single oversized geometry retains the established emergency lane.
        self.assertEqual(publication_memory.native_fullframe_dense_reserve(native_tasks(1, 'file'),
                                                                          total_dense_limit=1), PARENT_BYTES)

    def test_hybrid_parents_reserve_potential_native_backings_but_d1_does_not(self):
        tasks = native_tasks(2, runtime.HYBRID_DEFERRED_RESULT_MODE) + native_tasks(1, 'd1_owner')
        self.assertEqual(publication_memory.native_fullframe_dense_reserve(tasks,
                                                                          total_dense_limit=256 * GIB), 2 * PARENT_BYTES)


class SharedNativeWorkspaceTests(unittest.TestCase):
    """Execute actual parent allocation and worker open/close blocks on tiny data."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = _state()
        self.view = ViewInfo(name='spherical_test__tta_a0', family='spherical',
                             num_slices=4, src_h=5, src_w=6, pad_mode='clamp')
        self.shape = (4, 5, 6)
        self.namespace = dict(vars(pipeline))
        self.namespace.update(temp_dir=self.root, worker_direct_union_active=True,
            args=SimpleNamespace(imgsz=8, min_conf=1., interpolation_distance=0), dense_tiling_active=False,
            nrrd_layers_needed=False, baseline_union_by_model_view={},
            baseline_confmap_by_model_view={}, baseline_slice_locks_by_model_view={},
            baseline_union_paths=self.state.baseline_union_paths,
            baseline_confmap_paths=self.state.baseline_confmap_paths,
            direct_union_backing_leases=self.state.direct_union_backing_leases,
            direct_union_inference_views=self.state.direct_union_inference_views,
            direct_union_postprocess_views=self.state.direct_union_postprocess_views,
            direct_union_inference_bytes=self.state.direct_union_inference_bytes,
            direct_union_postprocess_bytes=self.state.direct_union_postprocess_bytes,
            view_processing_volume_shape=lambda *_args: self.shape)
        self.ensure = _function(Path(pipeline.__file__).read_text(encoding='utf-8'),
                                '_ensure_baseline_workspaces', self.namespace)
        tree = ast.parse(Path(workers.__file__).read_text(encoding='utf-8'))
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == 'run_prediction_volume_in_worker')
        outer = next(node for node in ast.walk(fn) if isinstance(node, ast.Try) and node.finalbody
                     and isinstance(node.body[0], ast.If)
                     and "task.get('result_mode', 'file')" in ast.unparse(node.body[0].test)
                     and any(isinstance(child, ast.Name) and child.id == 'DeviceOnlyRadialTarget'
                             for child in ast.walk(node.body[0])))
        self.worker_open = compile(ast.fix_missing_locations(ast.Module(body=[outer.body[0]], type_ignores=[])),
                                   '<worker shared union open>', 'exec')
        self.worker_close = compile(ast.fix_missing_locations(ast.Module(body=outer.finalbody, type_ignores=[])),
                                    '<worker shared union close>', 'exec')

    def close_parent(self):
        for name in ('baseline_union_by_model_view', 'baseline_confmap_by_model_view'):
            for value in self.namespace[name].values():
                runtime.close_memmap_array_without_flush(value)

    def run_shared_worker_windows(self, memfd):
        self.addCleanup(self.close_parent)
        with mock.patch.object(runtime, 'should_use_in_memory_workspace', return_value=memfd), \
                mock.patch.object(runtime, 'memfd_workspace_enabled', return_value=memfd), \
                contextlib.redirect_stdout(io.StringIO()):
            scheduler = _scheduler(self.root, state=self.state,
                input_overrides={'ensure_baseline_workspaces': self.ensure, 'min_conf': 1.})
            for start, fail in ((0, False), (2, True)):
                task = dict(kind='fullframe', result_mode='direct_union', model_name='model',
                            view=self.view, processing_shape=self.shape, union_num_slices=4)
                scheduler.activate_direct_union_task(task)
                env = dict(vars(workers))
                env.update(task=task, slice_count=2, slice_offset=start, processing_h=5, processing_w=6,
                    result_shape=(2, 5, 6), result_mask=None, result_conf=None, result_mask_full=None,
                    result_conf_full=None, azimuthal_padding_mask=None, azimuthal_padding_conf=None,
                    source_mm=None, source=None)
                try:
                    exec(self.worker_open, env)
                    env['result_mask'][:] = start + 1
                    env['result_conf'][:] = start + 2
                    if fail:
                        raise RuntimeError('inference failed after native write')
                except RuntimeError as error:
                    self.assertEqual(str(error), 'inference failed after native write')
                finally:
                    exec(self.worker_close, env)
                self.assertTrue(env['result_mask_full']._mmap.closed)
                self.assertTrue(env['result_conf_full']._mmap.closed)
                key = ('model', self.view.name)
                parent = self.namespace['baseline_union_by_model_view'][key]
                self.assertFalse(parent._mmap.closed)
                self.assertEqual(self.state.direct_union_backing_leases[key].phase, 'inference')
            expected = np.ones(self.shape, np.uint8)
            expected[2:] = 3
            np.testing.assert_array_equal(parent, expected)
            self.assertEqual(self.state.direct_union_inference_bytes[key], 2 * int(np.prod(self.shape)))
            if memfd:
                self.assertTrue(str(self.state.baseline_union_paths[key]).startswith('/proc/'))
            else:
                self.assertTrue(self.state.baseline_union_paths[key].is_file())

    def test_shared_path_windows_and_worker_failure_preserve_parent_owner(self):
        self.run_shared_worker_windows(False)

    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux memfd unavailable')
    def test_shared_memfd_windows_and_worker_failure_preserve_parent_owner(self):
        self.run_shared_worker_windows(True)

    def test_partial_parent_allocation_failure_closes_mapping_before_any_lease_exists(self):
        allocations = []

        def allocate(**kwargs):
            if allocations:
                raise RuntimeError('confidence allocation failed')
            value = runtime.allocate_workspace_array(**kwargs)
            allocations.append(value)
            return value

        self.namespace['allocate_workspace_array'] = allocate
        with mock.patch.object(runtime, 'should_use_in_memory_workspace', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, 'confidence allocation failed'):
            self.ensure('model', self.view)
        self.assertTrue(allocations[0]._mmap.closed)
        self.assertFalse(self.state.direct_union_backing_leases)
        self.assertFalse(self.state.baseline_union_paths)
        self.assertFalse(list(self.root.rglob('*.dat')))


if __name__ == '__main__':
    unittest.main()
