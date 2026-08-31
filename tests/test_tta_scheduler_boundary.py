from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tools.smoke_import import install_stubs


install_stubs()

from XTA.geometry import ViewInfo
from XTA.tta_scheduler import (
    TtaScheduler,
    TtaSchedulerCallbacks,
    TtaSchedulerInputs,
    TtaSchedulerOperations,
    TtaSchedulerState,
)


class _Telemetry:
    def add(self, *_args: object, **_kwargs: object) -> None:
        return None

    def gauge(self, *_args: object, **_kwargs: object) -> None:
        return None


def _view() -> ViewInfo:
    return ViewInfo(
        name="transverse__tta_a0",
        num_slices=16,
        src_h=8,
        src_w=8,
        pad_mode="clamp",
        physical_view_name="transverse",
        tta_aug_id="a0",
    )


def _state() -> TtaSchedulerState:
    return TtaSchedulerState(
        baseline_union_paths={},
        baseline_confmap_paths={},
        parent_mask_support_by_model={"model": {}},
        fullframe_remaining={},
        direct_union_inference_views=set(),
        direct_union_postprocess_views=set(),
        direct_union_inference_bytes={},
        direct_union_postprocess_bytes={},
        direct_union_backing_leases={},
    )


def _inputs(temp_dir: Path, **overrides: object) -> TtaSchedulerInputs:
    values = dict(
        batch=1,
        conf=0.25,
        cpu_batch=1,
        cpu_precision="fp32",
        gpu_batch=1,
        gpu_quantize=None,
        imgsz=8,
        interpolation_distance=0,
        min_conf=0.0,
        min_radius=0.0,
        keep_temp_artifacts=False,
        dense_tiling_active=False,
        nrrd_layers_needed=False,
        direct_union_sparse_retirement_active=True,
        direct_union_inference_view_limit=2,
        direct_union_inference_byte_limit=1 << 30,
        direct_union_total_dense_byte_limit=1 << 31,
        gpu_worker_tile_dense_result_limit=1 << 30,
        gpu_worker_tile_dense_result_task_limit=4,
        gpu_worker_process_active=True,
        inference_worker_process_active=True,
        gpu_device_count=1,
        v1613_d1_owner_active=False,
        gpu_worker_result_dir=temp_dir / "gpu_worker_results",
        ensure_baseline_workspaces=lambda _model, _view: None,
        gib=1024 ** 3,
        hybrid_deferred_result_mode="hybrid_deferred",
    )
    values.update(overrides)
    return TtaSchedulerInputs(**values)


def _operations(**overrides: object) -> TtaSchedulerOperations:
    telemetry = _Telemetry()
    values = dict(
        _attach_memfd_transfers_to_task=lambda _task: None,
        _env_float=lambda _name, default: float(default),
        _env_int=lambda _name, default: int(default),
        _main_process_gpu_stage_begin_inference=lambda _worker: True,
        _main_process_gpu_stage_can_dispatch_inference=lambda _worker: True,
        _main_process_gpu_stage_finish_inference=lambda _worker: None,
        _memfd_owner_key_from_array=lambda _array: None,
        _memmap_backing_path=lambda _array: None,
        _path_is_relative_to=lambda _path, _root: False,
        _sanitize_filesystem_token=lambda value: str(value),
        _set_main_process_gpu_pending_inference=lambda _active: None,
        _set_main_process_gpu_stage_wake_callback=lambda _callback: None,
        set_gpu_worker_aux_interpolation_pool=lambda _pool: None,
        allocate_workspace_array=lambda **_kwargs: None,
        array_nbytes=lambda shape, _dtype: int(__import__("math").prod(shape)),
        available_anon_work_bytes=lambda: 1 << 40,
        close_memmap_array_without_flush=lambda _array: None,
        cpu_inference_task_priority=lambda _task: 0,
        cpu_worker_default_seconds_per_frame=lambda _view: 0.25,
        cpu_worker_max_lease_slices=lambda: 64,
        cpu_worker_min_lease_slices=lambda: 8,
        cpu_worker_target_lease_seconds=lambda: 1.0,
        gpu_worker_aux_interpolation_pool=lambda: None,
        gpu_worker_default_seconds_per_frame=lambda _view: 0.1,
        gpu_worker_max_lease_slices=lambda: 64,
        gpu_worker_min_lease_slices=lambda: 8,
        gpu_worker_tail_split_point=lambda *_args: None,
        gpu_worker_target_lease_seconds=lambda: 1.0,
        gpu_worker_task_cost_key=lambda task: (str(task.get("kind")),),
        hybrid_gpu_stealback_enabled=lambda: False,
        hybrid_gpu_stealback_eta_ratio=lambda: 1.0,
        hybrid_gpu_stealback_max_fraction=lambda: 0.5,
        hybrid_gpu_stealback_min_cpu_samples=lambda: 1,
        hybrid_gpu_stealback_min_lead_seconds=lambda: 0.0,
        memfd_workspace_enabled=lambda: False,
        preflight_multiprocessing_payload=lambda _payload: None,
        radial_batch_padding_count=lambda *_args, **_kwargs: 0,
        radial_batch_padding_mirror_groups=lambda *_args, **_kwargs: (),
        runtime_telemetry=lambda: telemetry,
        tile_dense_worker_result_warn_seconds=lambda: 0.0,
        view_processing_volume_shape=lambda _view, size: (1, int(size), int(size)),
        workspace_anon_cap_bytes=lambda: 0,
    )
    values.update(overrides)
    return TtaSchedulerOperations(**values)


def _scheduler(
    temp_dir: Path,
    *,
    state: TtaSchedulerState | None = None,
    input_overrides: dict[str, object] | None = None,
    operation_overrides: dict[str, object] | None = None,
) -> TtaScheduler:
    return TtaScheduler(
        inputs=_inputs(temp_dir, **(input_overrides or {})),
        state=state or _state(),
        operations=_operations(**(operation_overrides or {})),
    )


def _bind_callbacks(
    scheduler: TtaScheduler,
    *,
    fullframe: object | None = None,
    tile: object | None = None,
    announce: object | None = None,
    affinity: object | None = None,
) -> tuple[mock.Mock, mock.Mock, mock.Mock, mock.Mock]:
    callbacks = tuple(
        value if isinstance(value, mock.Mock) else mock.Mock()
        for value in (fullframe, tile, announce, affinity)
    )
    scheduler.bind_result_callbacks(TtaSchedulerCallbacks(
        handle_fullframe_worker_result=callbacks[0],
        handle_tile_worker_result=callbacks[1],
        announce_process_inference_drain_if_complete=callbacks[2],
        check_parent_affinity=callbacks[3],
    ))
    return callbacks  # type: ignore[return-value]


class TtaSchedulerBoundaryTests(unittest.TestCase):
    def test_state_preserves_shared_container_identity_and_result_copies_stats(self) -> None:
        state = _state()
        baseline_paths = state.baseline_union_paths
        d1_refs = state.d1_layer_ref_by_parent
        state.gpu_worker_dispatched_by_id[0] = 2
        state.gpu_worker_results_by_id[0] = 1
        state.gpu_worker_total_tasks = 2
        state.gpu_worker_results_collected = 1
        state.gpu_worker_tile_dense_result_workspaces[17] = (None, None)
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(Path(temp_dir), state=state)
            result = scheduler.result()

        self.assertIs(scheduler.state.baseline_union_paths, baseline_paths)
        self.assertIs(result.artifacts.d1_layer_ref_by_parent, d1_refs)
        state.gpu_worker_dispatched_by_id[0] = 9
        self.assertEqual(result.gpu_dispatched_by_worker, {0: 2})
        self.assertEqual(result.quiescence_issues["remaining_results"], 1)
        self.assertEqual(result.quiescence_issues["tile_ownership_task_ids"], [17])

    def test_bind_callbacks_and_result_transport_are_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(Path(temp_dir))
            callbacks = TtaSchedulerCallbacks(
                handle_fullframe_worker_result=mock.Mock(),
                handle_tile_worker_result=mock.Mock(),
                announce_process_inference_drain_if_complete=mock.Mock(),
                check_parent_affinity=mock.Mock(),
            )
            scheduler.bind_result_callbacks(callbacks)
            with self.assertRaisesRegex(RuntimeError, "already bound"):
                scheduler.bind_result_callbacks(callbacks)
            scheduler.configure_result_transport(
                push_drain_active=False,
                track_thread=mock.Mock(),
            )
            with self.assertRaisesRegex(RuntimeError, "already configured"):
                scheduler.configure_result_transport(
                    push_drain_active=False,
                    track_thread=mock.Mock(),
                )

    def test_tile_dispatch_failure_rolls_back_reservation_before_requeue(self) -> None:
        state = _state()
        state.gpu_task_queues[0] = mock.Mock()
        state.gpu_worker_dispatched_by_id[0] = 0
        state.gpu_worker_compute_completed_by_id[0] = 0
        task = {
            "task_id": 7,
            "kind": "tile",
            "model_name": "model",
            "view": _view(),
            "slice_start": 0,
            "slice_count": 2,
            "prediction_batch": 1,
            "processing_shape": (2, 3, 4),
            "result_mask_path": "mask.dat",
            "result_conf_path": None,
            "gpu_eligible": True,
        }
        state.gpu_worker_tasks_by_id[7] = task
        state.gpu_worker_pending_task_ids.append(7)
        finish_stage = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir),
                state=state,
                operation_overrides={
                    "_main_process_gpu_stage_finish_inference": finish_stage,
                },
            )
            scheduler.prepare_tile_dense_result_workspaces = mock.Mock(
                side_effect=RuntimeError("workspace failure")
            )
            with self.assertRaisesRegex(RuntimeError, "workspace failure"):
                scheduler.dispatch_gpu_worker_inference_window()

        self.assertEqual(list(state.gpu_worker_pending_task_ids), [7])
        self.assertFalse(state.gpu_worker_tile_dense_result_reservations)
        self.assertEqual(state.gpu_worker_tile_dense_result_bytes_reserved, 0)
        finish_stage.assert_called_once_with(0)

    def test_d1_claim_release_and_dynamic_split_update_authoritative_state(self) -> None:
        state = _state()
        view = _view()
        parent = ("model", view.name)
        d1_task = {
            "task_id": 0,
            "kind": "fullframe",
            "model_name": "model",
            "view": view,
            "result_mode": "d1_owner",
        }
        split_task = {
            "task_id": 1,
            "kind": "fullframe",
            "model_name": "model",
            "view": view,
            "result_mode": "file",
            "slice_start": 0,
            "slice_count": 100,
            "render_workers": 4,
        }
        state.gpu_worker_tasks_by_id.update({0: d1_task, 1: split_task})
        state.gpu_worker_total_tasks = 2
        state.gpu_worker_next_dynamic_task_id = 2
        state.fullframe_remaining[parent] = 2
        state.fullframe_task_ids_by_parent[parent] = [0, 1]
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir),
                state=state,
                input_overrides={"v1613_d1_owner_active": True},
            )
            self.assertTrue(scheduler.claim_d1_owner(d1_task, 0))
            self.assertEqual(state.d1_owner_by_parent[parent], 0)
            scheduler.release_d1_owner_if_complete(
                d1_task, 0, {"d1_view_complete": True}
            )
            self.assertFalse(state.d1_owner_by_parent)
            self.assertFalse(state.d1_active_parent_by_worker)

            # Runtime splitting is independent of D1 for a file-result lease.
            scheduler.split_gpu_worker_task_to_runtime_target(1)

        self.assertEqual(state.gpu_worker_total_tasks, 3)
        self.assertEqual(state.gpu_worker_next_dynamic_task_id, 3)
        self.assertIn(2, state.gpu_worker_tasks_by_id)
        self.assertIn(2, state.gpu_worker_pending_task_ids)
        self.assertEqual(state.fullframe_remaining[parent], 3)

    def test_cpu_and_gpu_results_update_state_before_reentrant_callbacks(self) -> None:
        state = _state()
        view = _view()
        state.gpu_worker_total_tasks = 2
        state.cpu_worker_results_by_id[0] = 0
        state.gpu_worker_results_by_id[0] = 0
        state.gpu_worker_tasks_by_id[10] = {
            "task_id": 10,
            "kind": "fullframe",
            "model_name": "model",
            "view": view,
            "slice_count": 3,
            "result_mode": "file",
        }
        state.gpu_worker_tasks_by_id[11] = {
            "task_id": 11,
            "kind": "fullframe",
            "model_name": "model",
            "view": view,
            "slice_count": 5,
            "result_mode": "file",
        }
        observed: list[tuple[int, int]] = []

        def on_fullframe(_task: object, _stats: object) -> None:
            observed.append(
                (
                    state.gpu_worker_results_collected,
                    sum(state.cpu_worker_results_by_id.values())
                    + sum(state.gpu_worker_results_by_id.values()),
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(Path(temp_dir), state=state)
            scheduler.dispatch_inference_windows = mock.Mock()
            scheduler.refresh_gpu_aux_interpolation_leases = mock.Mock()
            fullframe, _tile, announce, _affinity = _bind_callbacks(
                scheduler, fullframe=mock.Mock(side_effect=on_fullframe)
            )
            scheduler.process_one_worker_result({
                "type": "result",
                "worker_kind": "cpu",
                "cpu_index": 0,
                "task_id": 10,
                "ok": True,
                "stats": {},
            })
            scheduler.process_one_worker_result({
                "type": "result",
                "worker_kind": "gpu",
                "gpu_index": 0,
                "task_id": 11,
                "ok": True,
                "stats": {},
            })

        self.assertEqual(observed, [(1, 1), (2, 2)])
        self.assertEqual(state.cpu_frames_completed_total, 3)
        self.assertEqual(state.gpu_frames_completed_total, 5)
        self.assertEqual(fullframe.call_count, 2)
        self.assertEqual(announce.call_count, 2)

    def test_poll_and_push_transport_each_have_one_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            poll_state = _state()
            poll_state.gpu_result_queue = queue.Queue()
            poll_scheduler = _scheduler(Path(temp_dir), state=poll_state)
            poll_scheduler.process_one_worker_result = mock.Mock()
            poll_state.gpu_result_queue.put({"type": "ready", "worker_kind": "gpu"})
            poll_scheduler.configure_result_transport(
                push_drain_active=False,
                track_thread=mock.Mock(),
            )
            poll_scheduler.drain_process_inference_results()
            poll_scheduler.process_one_worker_result.assert_called_once()

            push_state = _state()
            push_state.gpu_result_queue = queue.Queue()
            push_scheduler = _scheduler(Path(temp_dir), state=push_state)
            push_scheduler.process_one_worker_result = mock.Mock()
            tracked: list[threading.Thread] = []

            def track(thread: threading.Thread, _stop: threading.Event) -> None:
                tracked.append(thread)

            push_scheduler.configure_result_transport(
                push_drain_active=True,
                track_thread=track,
            )
            push_state.gpu_result_queue.put(
                {"type": "ready", "worker_kind": "gpu"}
            )
            self.assertTrue(push_state.scheduler_wake.wait(timeout=2.0))
            push_scheduler.drain_process_inference_results()
            push_state.push_drain_stop.set()
            tracked[0].join(timeout=2.0)

        push_scheduler.process_one_worker_result.assert_called_once()
        self.assertFalse(tracked[0].is_alive())

    def test_dead_worker_and_shutdown_fail_aux_waiters_before_reap(self) -> None:
        events: list[str] = []

        class _Aux:
            def outstanding(self) -> int:
                return 1

            def mark_failed(self, _reason: str) -> None:
                events.append("aux_failed")

        class _Process:
            name = "dead"
            pid = 1
            exitcode = 2

            def is_alive(self) -> bool:
                return False

            def join(self, **_kwargs: object) -> None:
                events.append("reaped")

        class _TaskQueue:
            def put(self, value: object) -> None:
                if value is None:
                    events.append("sentinel")

        aux = _Aux()
        state = _state()
        state.gpu_worker_total_tasks = 1
        state.gpu_worker_processes.append(_Process())
        state.gpu_task_queues[0] = _TaskQueue()
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir),
                state=state,
                operation_overrides={
                    "gpu_worker_aux_interpolation_pool": lambda: aux,
                    "set_gpu_worker_aux_interpolation_pool": (
                        lambda _pool: events.append("aux_unset")
                    ),
                },
            )
            _bind_callbacks(scheduler)
            with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                scheduler.check_inference_workers_alive()
            events.clear()
            scheduler.shutdown_inference_worker_processes()

        self.assertEqual(events[:3], ["aux_failed", "aux_unset", "sentinel"])
        self.assertIn("reaped", events)


if __name__ == "__main__":
    unittest.main()
