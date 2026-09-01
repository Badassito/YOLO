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
    @staticmethod
    def _d1_seed_tasks(
        state: TtaSchedulerState,
        *, ranges: tuple[tuple[int, int], ...],
        view: ViewInfo | None = None,
        shadow: bool = False,
        model_name: str = "model",
        task_id_start: int = 0,
    ) -> tuple[tuple[str, str], list[dict[str, object]]]:
        view_obj = view or _view()
        parent = (str(model_name), view_obj.name)
        tasks: list[dict[str, object]] = []
        for task_id, (start, stop) in enumerate(ranges, start=int(task_id_start)):
            task: dict[str, object] = {
                "task_id": int(task_id),
                "kind": "fullframe",
                "model_name": str(model_name),
                "view": view_obj,
                "result_mode": "d1_owner",
                "slice_start": int(start),
                "slice_count": int(stop - start),
                "gpu_eligible": True,
            }
            if shadow:
                task["d1_view_shadow_store_dir"] = "shadow"
            state.gpu_worker_tasks_by_id[int(task_id)] = task
            tasks.append(task)
        state.fullframe_task_ids_by_parent[parent] = [
            int(task["task_id"]) for task in tasks
        ]
        state.fullframe_remaining[parent] = len(tasks)
        return parent, tasks

    def test_d1_group_planner_assigns_every_seed_deterministically(self) -> None:
        state = _state()
        parent, tasks = self._d1_seed_tasks(
            state, ranges=((0, 2), (2, 6), (6, 9), (9, 12), (12, 14), (14, 16)),
        )
        state.gpu_worker_predicted_load_by_id.update({0: 2.0, 1: 0.0, 2: 1.0, 3: 0.0})
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir),
                state=state,
                input_overrides={"v1613_d1_owner_active": True, "gpu_device_count": 4},
                operation_overrides={
                    "_env_int": lambda name, default: (
                        1 if name == "YOLO_TTA_V1803_D1_OWNER_GROUPS" else
                        4 if name == "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE" else
                        int(default)
                    ),
                },
            )
            group = scheduler.ensure_d1_parent_group(tasks[0], (0, 1, 2, 3))

        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group.parent, parent)
        self.assertEqual(group.participants, (1, 3, 2, 0))
        self.assertEqual(set(group.task_worker_by_id), set(range(6)))
        owned_ranges = sorted(
            range_pair
            for ranges in group.expected_ranges_by_worker.values()
            for range_pair in ranges
        )
        self.assertEqual(owned_ranges, [(0, 2), (2, 6), (6, 9), (9, 12), (12, 14), (14, 16)])
        for task in tasks:
            assigned = int(group.task_worker_by_id[int(task["task_id"])])
            self.assertTrue(task["disable_runtime_split"])
            self.assertEqual(task["d1_group_worker_id"], assigned)
            self.assertEqual(
                scheduler.d1_feasible_workers(task, (0, 1, 2, 3)), [assigned]
            )

    def test_central_d1_planner_reserves_one_heavy_parent_atomically(self) -> None:
        state = _state()
        light_parent, light_tasks = self._d1_seed_tasks(
            state,
            ranges=((0, 4), (4, 8), (8, 12), (12, 16)),
            task_id_start=0,
        )
        heavy_view = ViewInfo(
            name="sagittal__tta_a0",
            num_slices=32,
            src_h=8,
            src_w=8,
            pad_mode="clamp",
            physical_view_name="sagittal",
            tta_aug_id="a0",
        )
        heavy_parent, heavy_tasks = self._d1_seed_tasks(
            state,
            ranges=((0, 8), (8, 16), (16, 24), (24, 32)),
            view=heavy_view,
            task_id_start=4,
        )
        state.gpu_worker_pending_task_ids.extend(range(8))
        state.gpu_task_queues.update({worker: mock.Mock() for worker in range(4)})
        state.gpu_worker_dispatched_by_id.update({worker: 0 for worker in range(4)})
        state.gpu_worker_compute_completed_by_id.update({worker: 0 for worker in range(4)})
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir), state=state,
                input_overrides={"v1613_d1_owner_active": True, "gpu_device_count": 4},
                operation_overrides={
                    "_env_int": lambda name, default: (
                        1 if name == "YOLO_TTA_V1803_D1_OWNER_GROUPS" else
                        2 if name == "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE" else
                        int(default)
                    ),
                },
            )

            group = scheduler.plan_next_d1_parent_group()
            repeated = scheduler.plan_next_d1_parent_group()

            self.assertIs(group, repeated)
            assert group is not None
            self.assertEqual(group.parent, heavy_parent)
            self.assertEqual(group.participants, (0, 1))
            self.assertEqual(state.d1_owner_by_parent[heavy_parent], 0)
            self.assertEqual(
                state.d1_active_parent_by_worker,
                {0: heavy_parent, 1: heavy_parent},
            )
            self.assertEqual(len(state.d1_groups_by_parent), 1)
            self.assertEqual(
                scheduler.d1_feasible_workers(light_tasks[0], (0, 1, 2, 3)),
                [2, 3],
            )
            self.assertIsNone(
                scheduler.ensure_d1_parent_group(light_tasks[0], (0, 1, 2, 3))
            )
            self.assertNotIn(light_parent, state.d1_owner_by_parent)
            for task in heavy_tasks:
                assigned = int(task["d1_group_worker_id"])
                self.assertEqual(
                    scheduler.d1_feasible_workers(task, (0, 1, 2, 3)),
                    [assigned],
                )

    def test_d1_group_size_clamps_and_size_one_preserves_single_owner(self) -> None:
        state = _state()
        _parent, tasks = self._d1_seed_tasks(
            state, ranges=((0, 4), (4, 8), (8, 12), (12, 16)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir),
                state=state,
                input_overrides={"v1613_d1_owner_active": True, "gpu_device_count": 8},
                operation_overrides={
                    "_env_int": lambda name, default: (
                        1 if name == "YOLO_TTA_V1803_D1_OWNER_GROUPS" else
                        8 if name == "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE" else
                        int(default)
                    ),
                },
            )
            group = scheduler.ensure_d1_parent_group(tasks[0], (2, 4, 6))
        assert group is not None
        self.assertEqual(len(group.participants), 3)

        state_one = _state()
        _parent_one, tasks_one = self._d1_seed_tasks(
            state_one, ranges=((0, 8), (8, 16)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler_one = _scheduler(
                Path(temp_dir),
                state=state_one,
                input_overrides={"v1613_d1_owner_active": True, "gpu_device_count": 4},
                operation_overrides={
                    "_env_int": lambda name, default: (
                        1 if name == "YOLO_TTA_V1803_D1_OWNER_GROUPS" else
                        1 if name == "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE" else
                        int(default)
                    ),
                },
            )
            self.assertIsNone(
                scheduler_one.ensure_d1_parent_group(tasks_one[0], (0, 1, 2, 3))
            )
            self.assertEqual(
                scheduler_one.d1_feasible_workers(tasks_one[0], (0, 1, 2, 3)),
                [0, 1, 2, 3],
            )

    def test_d1_group_rejects_gap_and_view_shadow_without_mutating_tasks(self) -> None:
        for ranges, shadow in (
            (((0, 4), (5, 16)), False),
            (((0, 8), (8, 16)), True),
        ):
            with self.subTest(ranges=ranges, shadow=shadow):
                state = _state()
                parent, tasks = self._d1_seed_tasks(
                    state, ranges=ranges, shadow=shadow,
                )
                with tempfile.TemporaryDirectory() as temp_dir:
                    scheduler = _scheduler(
                        Path(temp_dir), state=state,
                        input_overrides={
                            "v1613_d1_owner_active": True, "gpu_device_count": 4,
                        },
                        operation_overrides={
                            "_env_int": lambda name, default: (
                                1 if name == "YOLO_TTA_V1803_D1_OWNER_GROUPS" else
                                int(default)
                            ),
                        },
                    )
                    self.assertIsNone(
                        scheduler.ensure_d1_parent_group(tasks[0], (0, 1, 2, 3))
                    )
                self.assertIn(parent, state.d1_group_fallback_parents)
                self.assertTrue(all("d1_group_id" not in task for task in tasks))

    def test_d1_group_claims_all_participants_until_explicit_release(self) -> None:
        state = _state()
        parent, tasks = self._d1_seed_tasks(
            state, ranges=((0, 4), (4, 8), (8, 12), (12, 16)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir), state=state,
                input_overrides={"v1613_d1_owner_active": True, "gpu_device_count": 4},
                operation_overrides={
                    "_env_int": lambda name, default: (
                        1 if name == "YOLO_TTA_V1803_D1_OWNER_GROUPS" else
                        2 if name == "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE" else
                        int(default)
                    ),
                },
            )
            group = scheduler.ensure_d1_parent_group(tasks[0], (0, 1, 2, 3))
            assert group is not None
            self.assertEqual(set(state.d1_active_parent_by_worker), set(group.participants))
            first_by_worker: dict[int, dict[str, object]] = {}
            for task in tasks:
                first_by_worker.setdefault(int(task["d1_group_worker_id"]), task)
            for worker, task in first_by_worker.items():
                self.assertTrue(scheduler.claim_d1_owner(task, worker))
                scheduler.release_d1_owner_if_complete(
                    task, worker, {"d1_view_complete": True}
                )
            self.assertEqual(set(state.d1_active_parent_by_worker), set(group.participants))
            self.assertEqual(state.d1_owner_by_parent[parent], group.leader_worker_id)

    def test_d1_group_never_rebinds_after_single_candidate_commits_one_owner(self) -> None:
        state = _state()
        parent, tasks = self._d1_seed_tasks(
            state, ranges=((0, 4), (4, 8), (8, 12), (12, 16)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir), state=state,
                input_overrides={"v1613_d1_owner_active": True, "gpu_device_count": 4},
                operation_overrides={
                    "_env_int": lambda name, default: (
                        1 if name == "YOLO_TTA_V1803_D1_OWNER_GROUPS" else
                        4 if name == "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE" else
                        int(default)
                    ),
                },
            )
            # First refill has only one idle slot, so it crosses the dispatch boundary with
            # the established one-owner contract.
            self.assertEqual(scheduler.d1_feasible_workers(tasks[0], (2,)), [2])
            self.assertTrue(scheduler.claim_d1_owner(tasks[0], 2))

            # A later refill exposes all devices. It must retain worker 2 affinity and must
            # not retrofit already-indexed descriptors into a group.
            self.assertEqual(
                scheduler.d1_feasible_workers(tasks[1], (0, 1, 2, 3)), [2]
            )
            self.assertIsNone(
                scheduler.ensure_d1_parent_group(tasks[1], (0, 1, 2, 3))
            )

        self.assertNotIn(parent, state.d1_groups_by_parent)
        self.assertIn(parent, state.d1_group_fallback_parents)
        self.assertTrue(all("d1_group_id" not in task for task in tasks))

    def test_d1_group_dispatch_reservations_do_not_overlap_other_parents(self) -> None:
        """Exercise the real dispatch scan, not just the admission helper."""
        for group_size in (2, 4):
            with self.subTest(group_size=group_size):
                state = _state()
                light_parent, _light_tasks = self._d1_seed_tasks(
                    state,
                    ranges=((0, 4), (4, 8), (8, 12), (12, 16)),
                    task_id_start=0,
                )
                heavy_view = ViewInfo(
                    name="sagittal__tta_a0",
                    num_slices=32,
                    src_h=8,
                    src_w=8,
                    pad_mode="clamp",
                    physical_view_name="sagittal",
                    tta_aug_id="a0",
                )
                heavy_parent, _heavy_tasks = self._d1_seed_tasks(
                    state,
                    ranges=tuple((start, start + 4) for start in range(0, 32, 4)),
                    view=heavy_view,
                    task_id_start=4,
                )
                state.gpu_worker_pending_task_ids.extend(range(12))
                state.gpu_worker_total_tasks = 12
                state.gpu_worker_next_dynamic_task_id = 12
                state.gpu_task_queues.update(
                    {worker: queue.Queue() for worker in range(4)}
                )
                state.gpu_worker_dispatched_by_id.update(
                    {worker: 0 for worker in range(4)}
                )
                state.gpu_worker_compute_completed_by_id.update(
                    {worker: 0 for worker in range(4)}
                )

                def env_int(name: str, default: int) -> int:
                    return {
                        "YOLO_TTA_V1803_D1_OWNER_GROUPS": 1,
                        "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE": group_size,
                        "YOLO_TTA_GPU_WORKER_DISPATCH_WINDOW_PER_GPU": 1,
                    }.get(name, int(default))

                with tempfile.TemporaryDirectory() as temp_dir:
                    scheduler = _scheduler(
                        Path(temp_dir),
                        state=state,
                        input_overrides={
                            "v1613_d1_owner_active": True,
                            "gpu_device_count": 4,
                        },
                        operation_overrides={"_env_int": env_int},
                    )
                    scheduler.dispatch_gpu_worker_inference_window()

                self.assertEqual(len(state.d1_groups_by_parent), 1)
                group = state.d1_groups_by_parent[heavy_parent]
                self.assertEqual(len(group.participants), group_size)
                self.assertEqual(group.claimed_workers, set(group.participants))
                dispatched: dict[int, dict[str, object]] = {}
                for worker, task_queue in state.gpu_task_queues.items():
                    while not task_queue.empty():
                        task = task_queue.get_nowait()
                        task_id = int(task["task_id"])
                        self.assertNotIn(task_id, dispatched)
                        dispatched[task_id] = task
                        parent = scheduler.d1_task_parent_key(task)
                        if worker in group.participants:
                            self.assertEqual(parent, heavy_parent)
                            self.assertEqual(task["d1_group_worker_id"], worker)
                        else:
                            self.assertNotEqual(parent, heavy_parent)
                            if parent == light_parent:
                                self.assertNotIn(worker, group.participants)

                self.assertEqual(len(dispatched), group_size + (group_size == 2))
                for worker in group.participants:
                    self.assertEqual(
                        state.d1_active_parent_by_worker[worker], heavy_parent
                    )
                owners_by_worker = list(state.d1_active_parent_by_worker)
                self.assertEqual(len(owners_by_worker), len(set(owners_by_worker)))

    def test_d1_group_actual_dispatch_reduction_and_release_reaches_quiescence(self) -> None:
        """A CPU-only worker simulation catches reservation and refill deadlocks."""
        for group_size in (2, 4):
            with self.subTest(group_size=group_size):
                state = _state()
                parent, tasks = self._d1_seed_tasks(
                    state,
                    ranges=tuple(
                        (start, start + 2)
                        for start in range(0, 4 * group_size, 2)
                    ),
                    view=ViewInfo(
                        name=f"coronal__tta_group_{group_size}",
                        num_slices=4 * group_size,
                        src_h=8,
                        src_w=8,
                        pad_mode="clamp",
                        physical_view_name="coronal",
                        tta_aug_id="a0",
                    ),
                )
                state.gpu_worker_pending_task_ids.extend(
                    int(task["task_id"]) for task in tasks
                )
                state.gpu_worker_total_tasks = len(tasks)
                state.gpu_worker_next_dynamic_task_id = len(tasks)
                state.gpu_task_queues.update(
                    {worker: queue.Queue() for worker in range(4)}
                )
                state.gpu_worker_dispatched_by_id.update(
                    {worker: 0 for worker in range(4)}
                )
                state.gpu_worker_compute_completed_by_id.update(
                    {worker: 0 for worker in range(4)}
                )

                def env_int(name: str, default: int) -> int:
                    return {
                        "YOLO_TTA_V1803_D1_OWNER_GROUPS": 1,
                        "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE": group_size,
                        "YOLO_TTA_GPU_WORKER_DISPATCH_WINDOW_PER_GPU": 1,
                    }.get(name, int(default))

                def complete_fullframe(
                    task: dict[str, object], _stats: dict[str, object]
                ) -> None:
                    task_parent = scheduler.gpu_worker_fullframe_parent_key(task)
                    assert task_parent is not None
                    self.assertGreater(state.fullframe_remaining[task_parent], 0)
                    state.fullframe_remaining[task_parent] -= 1

                with tempfile.TemporaryDirectory() as temp_dir:
                    scheduler = _scheduler(
                        Path(temp_dir),
                        state=state,
                        input_overrides={
                            "v1613_d1_owner_active": True,
                            "gpu_device_count": 4,
                        },
                        operation_overrides={"_env_int": env_int},
                    )
                    _bind_callbacks(
                        scheduler,
                        fullframe=mock.Mock(side_effect=complete_fullframe),
                    )
                    scheduler.dispatch_gpu_worker_inference_window()
                    group = state.d1_groups_by_parent[parent]
                    self.assertEqual(len(group.participants), group_size)

                    ordinary_task_ids: set[int] = set()
                    release_tokens: set[str] = set()
                    reduction_count = 0
                    for _step in range(100):
                        made_progress = False
                        for worker, task_queue in state.gpu_task_queues.items():
                            if task_queue.empty():
                                continue
                            made_progress = True
                            dispatch = task_queue.get_nowait()
                            task_type = str(dispatch.get("task_type", ""))
                            if task_type == "d1_group_release":
                                release_tokens.add(
                                    str(dispatch["d1_group_lease_token"])
                                )
                                scheduler.process_one_worker_result({
                                    "type": "d1_group_released",
                                    "gpu_index": worker,
                                    "group_id": str(dispatch["d1_group_id"]),
                                    "lease_token": str(
                                        dispatch["d1_group_lease_token"]
                                    ),
                                    "ok": True,
                                    "allocation_released": (
                                        worker != group.leader_worker_id
                                    ),
                                })
                                continue
                            if task_type == "d1_group_reduce":
                                reduction_count += 1
                                artifacts = tuple(dispatch["d1_group_artifacts"])
                                scheduler.process_one_worker_result({
                                    "type": "d1_group_reduced",
                                    "group_id": str(dispatch["d1_group_id"]),
                                    "ok": True,
                                    "stats": {
                                        "d1_group_ack_tokens": tuple(
                                            str(artifact["lease_token"])
                                            for artifact in artifacts
                                        ),
                                        "d1_group_reduction_transport": (
                                            "cuda_ipc_nvlink"
                                        ),
                                        "d1_bitset_words": 128,
                                        "d1_group_peer_or_seconds": 0.02,
                                        "d1_group_d2h_seconds": 0.03,
                                    },
                                })
                                continue
                            self.assertFalse(task_type)
                            task_id = int(dispatch["task_id"])
                            self.assertNotIn(task_id, ordinary_task_ids)
                            ordinary_task_ids.add(task_id)
                            live_group = state.d1_groups_by_parent[parent]
                            self.assertEqual(
                                int(dispatch["d1_group_worker_id"]), worker
                            )
                            self.assertEqual(
                                state.d1_active_parent_by_worker[worker], parent
                            )
                            worker_task_ids = sorted(
                                task_id_i
                                for task_id_i, assigned_worker
                                in live_group.task_worker_by_id.items()
                                if assigned_worker == worker
                            )
                            stats: dict[str, object] = {
                                "worker_compute_seconds": 0.01,
                            }
                            if task_id == worker_task_ids[-1]:
                                stats["d1_group_partial_artifact"] = {
                                    "group_id": live_group.group_id,
                                    "participant_rank": (
                                        live_group.participants.index(worker)
                                    ),
                                    "participant_worker_id": worker,
                                    "covered_ranges": (
                                        live_group.expected_ranges_by_worker[worker]
                                    ),
                                    "lease_token": (
                                        f"lease-{group_size}-{worker}"
                                    ),
                                    "transport": "cuda_ipc",
                                }
                            scheduler.process_one_worker_result({
                                "type": "result",
                                "worker_kind": "gpu",
                                "gpu_index": worker,
                                "task_id": task_id,
                                "ok": True,
                                "stats": stats,
                            })
                        if (
                            not state.gpu_worker_pending_task_ids
                            and not state.d1_groups_by_parent
                            and state.gpu_worker_results_collected
                            == state.gpu_worker_total_tasks
                            and all(
                                task_queue.empty()
                                for task_queue in state.gpu_task_queues.values()
                            )
                        ):
                            break
                        self.assertTrue(made_progress, "D1 group dispatch deadlocked")
                    else:
                        self.fail("D1 group lifecycle did not quiesce within 100 steps")

                    result = scheduler.result()
                    group_summary = scheduler.d1_group_summary()

                self.assertEqual(ordinary_task_ids, set(range(len(tasks))))
                self.assertEqual(reduction_count, 1)
                self.assertEqual(
                    release_tokens,
                    {f"lease-{group_size}-{worker}" for worker in group.participants},
                )
                self.assertEqual(state.fullframe_remaining[parent], 0)
                self.assertEqual(result.gpu_worker_results_collected, len(tasks))
                self.assertEqual(result.quiescence_issues, {})
                self.assertEqual(
                    {
                        key: int(group_summary[key])
                        for key in ("admitted", "reduced", "released", "host_fallbacks")
                    },
                    {"admitted": 1, "reduced": 1, "released": 1, "host_fallbacks": 0},
                )
                self.assertEqual(
                    int(group_summary["peer_read_bytes"]),
                    128 * 4 * (group_size - 1),
                )
                self.assertAlmostEqual(float(group_summary["peer_or_seconds"]), 0.02)
                self.assertAlmostEqual(float(group_summary["d2h_seconds"]), 0.03)

    def test_d1_group_release_ack_barrier_is_quiescence_visible_and_strict(self) -> None:
        state = _state()
        parent, tasks = self._d1_seed_tasks(
            state, ranges=((0, 4), (4, 8), (8, 12), (12, 16)),
        )
        state.gpu_task_queues.update(
            {worker: queue.Queue() for worker in range(2)}
        )
        state.gpu_worker_dispatched_by_id.update({0: 0, 1: 0})
        state.gpu_worker_compute_completed_by_id.update({0: 0, 1: 0})

        def env_int(name: str, default: int) -> int:
            return {
                "YOLO_TTA_V1803_D1_OWNER_GROUPS": 1,
                "YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE": 2,
            }.get(name, int(default))

        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = _scheduler(
                Path(temp_dir),
                state=state,
                input_overrides={
                    "v1613_d1_owner_active": True,
                    "gpu_device_count": 2,
                },
                operation_overrides={"_env_int": env_int},
            )
            group = scheduler.ensure_d1_parent_group(tasks[0], (0, 1))
            assert group is not None
            for worker in group.participants:
                group.partial_artifacts_by_worker[worker] = {
                    "lease_token": f"strict-lease-{worker}",
                    "transport": "cuda_ipc",
                }
            group.held_result = {
                "type": "result",
                "worker_kind": "gpu",
                "gpu_index": group.leader_worker_id,
                "task_id": int(tasks[-1]["task_id"]),
                "ok": True,
                "stats": {},
            }
            scheduler.process_one_worker_result({
                "type": "d1_group_reduced",
                "group_id": group.group_id,
                "ok": True,
                "stats": {
                    "d1_group_ack_tokens": tuple(
                        f"strict-lease-{worker}" for worker in group.participants
                    ),
                },
            })

            releases = {
                worker: state.gpu_task_queues[worker].get_nowait()
                for worker in group.participants
            }
            first, second = group.participants
            first_release = releases[first]
            first_ack = {
                "type": "d1_group_released",
                "gpu_index": first,
                "group_id": group.group_id,
                "lease_token": first_release["d1_group_lease_token"],
                "ok": True,
                "allocation_released": True,
            }
            scheduler.process_one_worker_result(first_ack)

            issues = scheduler.process_quiescence_issues()
            self.assertIn("d1_groups", issues)
            group_issue = issues["d1_groups"][group.group_id]
            self.assertEqual(group_issue["release_pending"], [second])
            self.assertEqual(group_issue["release_acked"], [first])
            self.assertTrue(group_issue["held_final_result"])
            self.assertIs(state.d1_groups_by_parent[parent], group)
            self.assertEqual(
                state.d1_active_parent_by_worker,
                {first: parent, second: parent},
            )

            with self.assertRaisesRegex(RuntimeError, "duplicate/unrequested release"):
                scheduler.process_one_worker_result(dict(first_ack))

            second_release = releases[second]
            with self.assertRaisesRegex(RuntimeError, "failed to release its partial"):
                scheduler.process_one_worker_result({
                    "type": "d1_group_released",
                    "gpu_index": second,
                    "group_id": group.group_id,
                    "lease_token": second_release["d1_group_lease_token"],
                    "ok": False,
                    "allocation_released": False,
                    "error": "simulated cudaFree failure",
                })

        self.assertEqual(group.release_pending_workers, {second})
        self.assertEqual(group.release_ack_workers, {first})
        self.assertIn(parent, state.d1_groups_by_parent)

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
