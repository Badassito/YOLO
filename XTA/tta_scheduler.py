"""Stateful process-inference scheduling for the TTA pipeline.

The pipeline still owns run preparation, worker construction, completed-view assembly, and
the outer failure boundary.  This module owns the mutable process-inference control plane:
tile-result admission, hybrid/D1 ownership, CPU/GPU lease selection and dispatch, and the
worker accounting state consumed by result draining.  Lower layers remain unaware of this
high-level scheduler.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import weakref
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .geometry import ViewInfo


@dataclass(frozen=True)
class TtaSchedulerOperations:
    """Lower-layer operations injected from :mod:`XTA.pipeline`.

    Injection preserves the established pipeline monkeypatch seams and makes this owner
    independently testable without importing or initializing accelerator frameworks.
    """

    _attach_memfd_transfers_to_task: Callable[[Dict[str, object]], object]
    _env_float: Callable[[str, float], float]
    _env_int: Callable[[str, int], int]
    _main_process_gpu_stage_begin_inference: Callable[[int], bool]
    _main_process_gpu_stage_can_dispatch_inference: Callable[[int], bool]
    _main_process_gpu_stage_finish_inference: Callable[[int], object]
    _memfd_owner_key_from_array: Callable[[object], object]
    _memmap_backing_path: Callable[[object], object]
    _path_is_relative_to: Callable[[Path, Path], bool]
    _sanitize_filesystem_token: Callable[[str], str]
    _set_main_process_gpu_pending_inference: Callable[[bool], object]
    _set_main_process_gpu_stage_wake_callback: Callable[[object], object]
    set_gpu_worker_aux_interpolation_pool: Callable[[object], None]
    allocate_workspace_array: Callable[..., np.ndarray]
    array_nbytes: Callable[[Sequence[int], object], int]
    available_anon_work_bytes: Callable[[], int]
    close_memmap_array_without_flush: Callable[[object], None]
    cpu_inference_task_priority: Callable[[Dict[str, object]], object]
    cpu_worker_default_seconds_per_frame: Callable[[object], float]
    cpu_worker_max_lease_slices: Callable[[], int]
    cpu_worker_min_lease_slices: Callable[[], int]
    cpu_worker_target_lease_seconds: Callable[[], float]
    gpu_worker_aux_interpolation_pool: Callable[[], object]
    gpu_worker_default_seconds_per_frame: Callable[[object], float]
    gpu_worker_max_lease_slices: Callable[[], int]
    gpu_worker_min_lease_slices: Callable[[], int]
    gpu_worker_tail_split_point: Callable[..., object]
    gpu_worker_target_lease_seconds: Callable[[], float]
    gpu_worker_task_cost_key: Callable[[Dict[str, object]], Tuple[object, ...]]
    hybrid_gpu_stealback_enabled: Callable[[], bool]
    hybrid_gpu_stealback_eta_ratio: Callable[[], float]
    hybrid_gpu_stealback_max_fraction: Callable[[], float]
    hybrid_gpu_stealback_min_cpu_samples: Callable[[], int]
    hybrid_gpu_stealback_min_lead_seconds: Callable[[], float]
    memfd_workspace_enabled: Callable[[], bool]
    preflight_multiprocessing_payload: Callable[[object], object]
    radial_batch_padding_count: Callable[..., int]
    radial_batch_padding_mirror_groups: Callable[..., Sequence[object]]
    runtime_telemetry: Callable[[], object]
    tile_dense_worker_result_warn_seconds: Callable[[], float]
    view_processing_volume_shape: Callable[..., Sequence[int]]
    workspace_anon_cap_bytes: Callable[[], int]


@dataclass(frozen=True)
class TtaSchedulerInputs:
    """Run-constant process-inference configuration and synchronous callbacks."""

    batch: int
    conf: float
    cpu_batch: int
    cpu_precision: str
    gpu_batch: int
    gpu_quantize: object
    imgsz: int
    interpolation_distance: int
    min_conf: float
    min_radius: float
    keep_temp_artifacts: bool
    dense_tiling_active: bool
    nrrd_layers_needed: bool
    direct_union_sparse_retirement_active: bool
    direct_union_inference_view_limit: int
    direct_union_inference_byte_limit: int
    direct_union_total_dense_byte_limit: int
    gpu_worker_tile_dense_result_limit: int
    gpu_worker_tile_dense_result_task_limit: int
    gpu_worker_process_active: bool
    inference_worker_process_active: bool
    gpu_device_count: int
    v1613_d1_owner_active: bool
    gpu_worker_result_dir: Path
    ensure_baseline_workspaces: Callable[[str, ViewInfo], None]
    gib: int
    hybrid_deferred_result_mode: str


@dataclass(frozen=True)
class TtaSchedulerCallbacks:
    """Main-thread callbacks into the still-inline view/tile completion owner."""

    handle_fullframe_worker_result: Callable[[Dict[str, object], Dict[str, object]], None]
    handle_tile_worker_result: Callable[[Dict[str, object], Dict[str, object]], None]
    announce_process_inference_drain_if_complete: Callable[[], None]
    check_parent_affinity: Callable[[], None]


@dataclass
class D1ParentGroup:
    """Scheduler-owned transaction for one multi-worker D1 parent."""

    parent: Tuple[str, str]
    group_id: str
    participants: Tuple[int, ...]
    leader_worker_id: int
    task_worker_by_id: Dict[int, int]
    expected_ranges_by_worker: Dict[int, Tuple[Tuple[int, int], ...]]
    claimed_workers: set[int] = field(default_factory=set)
    completed_workers: set[int] = field(default_factory=set)
    partial_artifacts_by_worker: Dict[int, Dict[str, object]] = field(
        default_factory=dict
    )
    host_artifacts_by_worker: Dict[int, Dict[str, object]] = field(
        default_factory=dict
    )
    reduction_dispatched: bool = False
    host_fallback_requested: bool = False
    held_result: Optional[Dict[str, object]] = None
    release_pending_workers: set[int] = field(default_factory=set)
    release_ack_workers: set[int] = field(default_factory=set)
    admitted_at: float = field(default_factory=time.perf_counter)
    reduction_started_at: Optional[float] = None
    release_requested_at: Optional[float] = None


@dataclass
class TtaSchedulerState:
    """Sole mutable owner of process scheduler counters, registries, and queues."""

    # Registries shared synchronously with completed-view assembly.
    baseline_union_paths: Dict[Tuple[str, str], Path]
    baseline_confmap_paths: Dict[Tuple[str, str], Optional[Path]]
    parent_mask_support_by_model: Dict[str, Dict[str, object]]
    fullframe_remaining: Dict[Tuple[str, str], int]
    direct_union_inference_views: set[Tuple[str, str]]
    direct_union_postprocess_views: set[Tuple[str, str]]
    direct_union_inference_bytes: Dict[Tuple[str, str], int]
    direct_union_postprocess_bytes: Dict[Tuple[str, str], int]
    direct_union_backing_leases: Dict[Tuple[str, str], object]

    cpu_task_queues: Dict[int, object] = field(default_factory=dict)
    gpu_task_queues: Dict[int, object] = field(default_factory=dict)
    cpu_worker_processes: List[object] = field(default_factory=list)
    gpu_worker_processes: List[object] = field(default_factory=list)
    cpu_worker_dispatched_by_id: Dict[int, int] = field(default_factory=dict)
    cpu_worker_results_by_id: Dict[int, int] = field(default_factory=dict)
    cpu_worker_seconds_per_frame_ewma: Dict[Tuple[object, ...], float] = field(
        default_factory=dict
    )
    cpu_worker_predicted_load_by_id: Dict[int, float] = field(default_factory=dict)
    cpu_worker_task_predicted_seconds_by_id: Dict[int, float] = field(
        default_factory=dict
    )
    cpu_worker_ready_details_by_id: Dict[int, Dict[str, object]] = field(
        default_factory=dict
    )
    cpu_worker_dispatch_cursor: int = 0

    gpu_result_queue: object = None
    gpu_worker_tasks_by_id: Dict[int, Dict[str, object]] = field(default_factory=dict)
    gpu_worker_results_collected: int = 0
    gpu_worker_total_tasks: int = 0
    gpu_worker_dispatched_tasks: int = 0
    gpu_worker_dispatched_by_id: Dict[int, int] = field(default_factory=dict)
    gpu_worker_results_by_id: Dict[int, int] = field(default_factory=dict)
    gpu_worker_compute_completed_by_id: Dict[int, int] = field(default_factory=dict)
    gpu_worker_compute_released_task_ids: set[int] = field(default_factory=set)
    gpu_worker_seconds_per_frame_ewma: Dict[Tuple[object, ...], float] = field(
        default_factory=dict
    )
    gpu_worker_predicted_load_by_id: Dict[int, float] = field(default_factory=dict)
    gpu_worker_task_predicted_seconds_by_id: Dict[int, float] = field(
        default_factory=dict
    )
    gpu_worker_dispatch_cursor: int = 0
    gpu_worker_next_dynamic_task_id: int = 0
    gpu_worker_pending_task_ids: deque[int] = field(default_factory=deque)

    gpu_worker_tile_dense_result_bytes_reserved: int = 0
    gpu_worker_tile_dense_result_memfd_bytes_reserved: int = 0
    gpu_worker_tile_dense_result_max_retention_seconds: float = 0.0
    gpu_worker_tile_dense_result_reservations: Dict[int, int] = field(
        default_factory=dict
    )
    gpu_worker_tile_dense_result_memfd_reservations: Dict[int, int] = field(
        default_factory=dict
    )
    gpu_worker_tile_dense_result_reserved_at: Dict[int, float] = field(
        default_factory=dict
    )
    gpu_worker_tile_dense_result_workspaces: Dict[
        int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    ] = field(default_factory=dict)
    gpu_worker_tile_task_id_by_key: Dict[Tuple[str, str, str], int] = field(
        default_factory=dict
    )
    gpu_worker_tile_pending_result_ids_by_task: Dict[int, set[str]] = field(
        default_factory=dict
    )

    gpu_inference_drained_at: Optional[float] = None
    gpu_inference_drain_announced: bool = False
    d1_owner_by_parent: Dict[Tuple[str, str], int] = field(default_factory=dict)
    d1_active_parent_by_worker: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    d1_layer_ref_by_parent: Dict[Tuple[str, str], object] = field(default_factory=dict)
    d1_view_shadow_path_by_parent: Dict[Tuple[str, str], Path] = field(
        default_factory=dict
    )
    d1_groups_by_parent: Dict[Tuple[str, str], D1ParentGroup] = field(
        default_factory=dict
    )
    d1_group_parent_by_id: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    d1_group_fallback_parents: set[Tuple[str, str]] = field(default_factory=set)
    d1_group_next_control_task_id: int = -1
    d1_group_parents_admitted_total: int = 0
    d1_group_parents_reduced_total: int = 0
    d1_group_parents_released_total: int = 0
    d1_group_host_fallbacks_total: int = 0
    d1_group_peer_read_bytes_total: int = 0
    d1_group_reduction_seconds_total: float = 0.0
    d1_group_peer_or_seconds_total: float = 0.0
    d1_group_d2h_seconds_total: float = 0.0
    d1_group_release_seconds_total: float = 0.0

    hybrid_view_mode_by_parent: Dict[Tuple[str, str], str] = field(default_factory=dict)
    fullframe_task_ids_by_parent: Dict[Tuple[str, str], List[int]] = field(
        default_factory=dict
    )
    hybrid_cpu_reserved_parents: List[Tuple[str, str]] = field(default_factory=list)
    hybrid_cpu_reserved_parent_set: set[Tuple[str, str]] = field(default_factory=set)
    hybrid_cpu_reservation_rank_by_parent: Dict[Tuple[str, str], int] = field(
        default_factory=dict
    )
    gpu_worker_cpu_assist_inflight_task_ids: set[int] = field(default_factory=set)
    gpu_worker_cpu_assist_completed_task_ids: set[int] = field(default_factory=set)
    hybrid_stealback_announced_parents: set[Tuple[str, str]] = field(
        default_factory=set
    )
    hybrid_cpu_idle_reason_counts: Counter[str] = field(default_factory=Counter)
    hybrid_cpu_idle_reason_last: str = ""
    hybrid_cpu_idle_active: bool = False
    hybrid_cpu_idle_since: Optional[float] = None
    gpu_frames_completed_total: int = 0
    cpu_frames_completed_total: int = 0
    hybrid_gpu_frames_completed_total: int = 0
    hybrid_cpu_frames_completed_total: int = 0
    hybrid_view_frames_by_backend: Dict[Tuple[str, str], Counter[str]] = field(
        default_factory=dict
    )
    hybrid_view_tasks_by_backend: Dict[Tuple[str, str], Counter[str]] = field(
        default_factory=dict
    )
    gpu_worker_seed_task_count: int = 0
    result_transport_configured: bool = False
    push_drain_active: bool = False
    scheduler_wake: threading.Event = field(default_factory=threading.Event)
    push_drain_stop: threading.Event = field(default_factory=threading.Event)
    pushed_worker_results: deque[Dict[str, object]] = field(default_factory=deque)
    wake_hooked_futures: weakref.WeakSet[Future] = field(default_factory=weakref.WeakSet)


@dataclass(frozen=True)
class TtaSchedulerArtifacts:
    """Identity-preserving artifact registries handed to the post-scheduler tail."""

    d1_layer_ref_by_parent: Mapping[Tuple[str, str], object]
    d1_view_shadow_path_by_parent: Mapping[Tuple[str, str], Path]


@dataclass(frozen=True)
class TtaSchedulerResult:
    """Immutable process-scheduler telemetry and artifact handoff."""

    gpu_inference_drained_at: Optional[float]
    gpu_worker_results_collected: int
    gpu_worker_total_tasks: int
    gpu_frames_completed_total: int
    cpu_frames_completed_total: int
    hybrid_gpu_frames_completed_total: int
    hybrid_cpu_frames_completed_total: int
    gpu_dispatched_by_worker: Mapping[int, int]
    gpu_completed_by_worker: Mapping[int, int]
    cpu_dispatched_by_worker: Mapping[int, int]
    cpu_completed_by_worker: Mapping[int, int]
    cpu_worker_ready_details: Mapping[int, Mapping[str, object]]
    hybrid_view_mode_by_parent: Mapping[Tuple[str, str], str]
    hybrid_view_frames_by_backend: Mapping[Tuple[str, str], Mapping[str, int]]
    hybrid_view_tasks_by_backend: Mapping[Tuple[str, str], Mapping[str, int]]
    quiescence_issues: Mapping[str, object]
    artifacts: TtaSchedulerArtifacts


class TtaScheduler:
    """Own process-inference admission, ownership policy, and dispatch state."""

    def __init__(
        self,
        *,
        inputs: TtaSchedulerInputs,
        state: TtaSchedulerState,
        operations: TtaSchedulerOperations,
    ) -> None:
        self.inputs = inputs
        self.state = state
        self.operations = operations
        self._callbacks: Optional[TtaSchedulerCallbacks] = None

    def bind_result_callbacks(self, callbacks: TtaSchedulerCallbacks) -> None:
        """Bind the main-thread completion callbacks exactly once before result drain."""

        if self._callbacks is not None:
            raise RuntimeError("TTA scheduler result callbacks are already bound")
        self._callbacks = callbacks

    def _result_callbacks(self) -> TtaSchedulerCallbacks:
        callbacks = self._callbacks
        if callbacks is None:
            raise RuntimeError("TTA scheduler result callbacks are not bound")
        return callbacks

    # Dispatch methods are defined below.

    def tile_task_radial_padding_count(self, task: Dict[str, object]) -> int:
        if str(task.get('kind', '')) != 'tile':
            return 0
        view_obj = task.get('view')
        if not isinstance(view_obj, ViewInfo):
            return 0
        return self.operations.radial_batch_padding_count(
            view_obj,
            int(task.get('slice_count', view_obj.num_slices)),
            max(1, int(task.get('prediction_batch', self.inputs.batch))),
            slice_offset=int(task.get('slice_start', 0)),
        )

    def tile_dense_result_task_bytes(self, task: Dict[str, object]) -> int:
        if str(task.get('kind', '')) != 'tile':
            return 0
        shape = tuple(int(v) for v in task.get('processing_shape', ()))
        if len(shape) != 3:
            return 0
        planes = 2 if task.get('result_conf_path') else 1
        main_bytes = int(self.operations.array_nbytes(shape, np.uint8)) * int(planes)
        padding_count = int(self.tile_task_radial_padding_count(task))
        if padding_count <= 0:
            return int(main_bytes)
        view_obj = task.get('view')
        assert isinstance(view_obj, ViewInfo)
        group_count = len(self.operations.radial_batch_padding_mirror_groups(
            view_obj,
            int(task.get('slice_count', view_obj.num_slices)),
            max(1, int(task.get('prediction_batch', self.inputs.batch))),
            slice_offset=int(task.get('slice_start', 0)),
        ))
        plane_bytes = int(shape[1]) * int(shape[2]) * int(np.dtype(np.uint8).itemsize)
        compact_padding_bytes = int(padding_count) * int(plane_bytes) * int(planes)
        grouped_result_bytes = int(group_count) * int(main_bytes)
        return int(main_bytes + compact_padding_bytes + grouped_result_bytes)

    def tile_dense_result_task_admissible(self, task: Dict[str, object]) -> bool:
        if bool(self.inputs.keep_temp_artifacts) or str(task.get('kind', '')) != 'tile':
            return True
        task_id = int(task.get('task_id', -1))
        if task_id in self.state.gpu_worker_tile_dense_result_reservations:
            return True
        need = int(self.tile_dense_result_task_bytes(task))
        if need <= 0:
            return True
        # Always admit one oversized tile when the live set is empty; otherwise a valid
        # geometry whose crop exceeds the configured budget could deadlock forever.
        if not self.state.gpu_worker_tile_dense_result_reservations:
            return True
        if len(self.state.gpu_worker_tile_dense_result_reservations) >= int(
            self.inputs.gpu_worker_tile_dense_result_task_limit
        ):
            return False
        return bool(
            int(self.state.gpu_worker_tile_dense_result_bytes_reserved) + int(need)
            <= int(self.inputs.gpu_worker_tile_dense_result_limit)
        )

    def tile_parent_mask_ready_for_task(self, task: Dict[str, object]) -> bool:
        if str(task.get('kind', '')) != 'tile':
            return False
        view_obj = task.get('view')
        view_name = getattr(view_obj, 'name', None)
        if view_name is None:
            return False
        return str(view_name) in self.state.parent_mask_support_by_model.get(
            str(task.get('model_name', '')), {}
        )

    def inference_storage_priority_rank(self, task: Dict[str, object]) -> int:
        """Prefer parent work, then immediately gateable tiles, then early tiles."""
        if str(task.get('kind', '')) != 'tile':
            return 0
        return 1 if self.tile_parent_mask_ready_for_task(task) else 2

    def reserve_tile_dense_result_task(self, task: Dict[str, object]) -> bool:
        if bool(self.inputs.keep_temp_artifacts) or str(task.get('kind', '')) != 'tile':
            return False
        task_id = int(task.get('task_id', -1))
        if task_id < 0 or task_id in self.state.gpu_worker_tile_dense_result_reservations:
            return False
        if not self.tile_dense_result_task_admissible(task):
            raise RuntimeError(
                f'array-backed tile-result admission raced its {self.inputs.gpu_worker_tile_dense_result_limit / self.inputs.gib:.1f} GiB budget'
            )
        need = int(self.tile_dense_result_task_bytes(task))
        self.state.gpu_worker_tile_dense_result_reservations[task_id] = int(need)
        self.state.gpu_worker_tile_dense_result_reserved_at[task_id] = float(time.monotonic())
        self.state.gpu_worker_tile_dense_result_bytes_reserved += int(need)
        self.operations.runtime_telemetry().gauge(
            'tile.dense_worker_result_bytes_reserved',
            int(self.state.gpu_worker_tile_dense_result_bytes_reserved),
        )
        self.operations.runtime_telemetry().gauge(
            'tile.dense_worker_result_tasks_reserved',
            int(len(self.state.gpu_worker_tile_dense_result_reservations)),
        )
        return True

    def prepare_tile_dense_result_workspaces(self, task: Dict[str, object]) -> None:
        """Allocate one task's array-backed tile result in shared RAM when possible.

        The parent owns each mapping and transfers duplicate memfd descriptors to the selected
        CUDA worker. Pathname files remain a bounded fallback when cgroup/RAM headroom is too
        small. A logical memfd reservation ledger covers not-yet-faulted pages, preventing a
        burst of dispatches from all passing the same stale memory-headroom snapshot. Holding
        the parent mapping until sparse retirement also prevents asynchronous D2H publication
        from outliving its backing object.
        """
        if bool(self.inputs.keep_temp_artifacts) or str(task.get('kind', '')) != 'tile':
            return
        task_id = int(task.get('task_id', -1))
        if task_id < 0:
            raise ValueError('tile result workspace requires a nonnegative task_id')
        if task_id in self.state.gpu_worker_tile_dense_result_workspaces:
            return
        shape = tuple(int(v) for v in task.get('processing_shape', ()))
        if len(shape) != 3:
            raise ValueError(f'tile task {task_id} has invalid processing_shape={shape}')
        task.setdefault('result_mask_fallback_path', str(task['result_mask_path']))
        task.setdefault('result_conf_fallback_path', task.get('result_conf_path'))
        original_mask_path = Path(str(task['result_mask_fallback_path']))
        original_conf_path = (
            Path(str(task['result_conf_fallback_path']))
            if task.get('result_conf_fallback_path') else None
        )
        mask_mm: Optional[np.ndarray] = None
        conf_mm: Optional[np.ndarray] = None
        total_need = int(self.tile_dense_result_task_bytes(task))
        anon_cap = int(self.operations.workspace_anon_cap_bytes())
        projected_memfd = int(self.state.gpu_worker_tile_dense_result_memfd_bytes_reserved) + int(total_need)
        prefer_shared_memfd = bool(
            self.operations.memfd_workspace_enabled()
            and int(self.operations.available_anon_work_bytes()) >= int(projected_memfd) + 16 * self.inputs.gib
            and (int(anon_cap) <= 0 or int(projected_memfd) <= int(anon_cap))
        )
        try:
            mask_mm = self.operations.allocate_workspace_array(
                shape=shape,
                dtype=np.uint8,
                path=original_mask_path,
                desc=f'GPU-worker tile result mask task {task_id}',
                prefer_memory=False,
                prefer_memfd=bool(prefer_shared_memfd),
                reserve_bytes=16 * self.inputs.gib,
                initialize_zero=True,
            )
            mask_backing = self.operations._memmap_backing_path(mask_mm)
            if mask_backing is None:
                raise RuntimeError(f'tile task {task_id} mask workspace has no reopenable backing')
            task['result_mask_path'] = str(mask_backing)

            if original_conf_path is not None:
                conf_mm = self.operations.allocate_workspace_array(
                    shape=shape,
                    dtype=np.uint8,
                    path=original_conf_path,
                    desc=f'GPU-worker tile result confidence task {task_id}',
                    prefer_memory=False,
                    prefer_memfd=bool(prefer_shared_memfd),
                    reserve_bytes=16 * self.inputs.gib,
                    initialize_zero=True,
                )
                conf_backing = self.operations._memmap_backing_path(conf_mm)
                if conf_backing is None:
                    raise RuntimeError(
                        f'tile task {task_id} confidence workspace has no reopenable backing'
                    )
                task['result_conf_path'] = str(conf_backing)

            task['result_workspace_preallocated'] = True
            self.state.gpu_worker_tile_dense_result_workspaces[task_id] = (mask_mm, conf_mm)
            actual_memfd_bytes = sum(
                int(np.asarray(mm).nbytes)
                for mm in (mask_mm, conf_mm)
                if mm is not None and self.operations._memfd_owner_key_from_array(mm) is not None
            )
            if int(actual_memfd_bytes) > 0:
                self.state.gpu_worker_tile_dense_result_memfd_reservations[task_id] = int(actual_memfd_bytes)
                self.state.gpu_worker_tile_dense_result_memfd_bytes_reserved += int(actual_memfd_bytes)
                self.operations.runtime_telemetry().add(
                    'tile.dense_worker_result_memfd_bytes', int(actual_memfd_bytes),
                )
                self.operations.runtime_telemetry().gauge(
                    'tile.dense_worker_result_memfd_bytes_reserved',
                    int(self.state.gpu_worker_tile_dense_result_memfd_bytes_reserved),
                )
            path_bytes = max(0, int(total_need) - int(actual_memfd_bytes))
            if int(path_bytes) > 0:
                self.operations.runtime_telemetry().add(
                    'tile.dense_worker_result_path_fallback_bytes', int(path_bytes),
                )
        except BaseException:
            task['result_mask_path'] = str(original_mask_path)
            task['result_conf_path'] = (
                str(original_conf_path) if original_conf_path is not None else None
            )
            task.pop('result_workspace_preallocated', None)
            for mm in (conf_mm, mask_mm):
                if mm is None:
                    continue
                backing = self.operations._memmap_backing_path(mm)
                is_memfd = self.operations._memfd_owner_key_from_array(mm) is not None
                try:
                    self.operations.close_memmap_array_without_flush(mm)
                except Exception:
                    pass
                if not is_memfd and backing is not None:
                    try:
                        Path(backing).unlink(missing_ok=True)
                    except Exception:
                        pass
            raise

    def release_tile_dense_result_task_id(self,
        task_id: int, *, reason: str = '', refill: bool = True,
    ) -> bool:
        task_id_i = int(task_id)
        self.state.gpu_worker_tile_pending_result_ids_by_task.pop(task_id_i, None)
        for result_key, mapped_task_id in list(self.state.gpu_worker_tile_task_id_by_key.items()):
            if int(mapped_task_id) == int(task_id_i):
                self.state.gpu_worker_tile_task_id_by_key.pop(result_key, None)
        released = self.state.gpu_worker_tile_dense_result_reservations.pop(task_id_i, None)
        released_memfd = self.state.gpu_worker_tile_dense_result_memfd_reservations.pop(task_id_i, None)
        reserved_at = self.state.gpu_worker_tile_dense_result_reserved_at.pop(task_id_i, None)
        workspaces = self.state.gpu_worker_tile_dense_result_workspaces.pop(task_id_i, None)
        task_obj = self.state.gpu_worker_tasks_by_id.get(task_id_i)
        cleanup_paths: set[Path] = set()
        if isinstance(task_obj, dict):
            for field_name in (
                'result_mask_path', 'result_conf_path',
                'result_mask_fallback_path', 'result_conf_fallback_path',
            ):
                raw_path = task_obj.get(field_name)
                if raw_path:
                    try:
                        cleanup_paths.add(Path(str(raw_path)))
                    except Exception:
                        pass
        if workspaces is not None:
            for mm in workspaces:
                if mm is None:
                    continue
                backing = self.operations._memmap_backing_path(mm)
                if backing is not None:
                    cleanup_paths.add(Path(backing))
                is_memfd = self.operations._memfd_owner_key_from_array(mm) is not None
                try:
                    self.operations.close_memmap_array_without_flush(mm)
                except Exception:
                    pass
                if not is_memfd and not bool(self.inputs.keep_temp_artifacts) and backing is not None:
                    try:
                        Path(backing).unlink(missing_ok=True)
                    except Exception:
                        pass
        if not bool(self.inputs.keep_temp_artifacts):
            # Also remove a fallback pathname that may have been created before a memfd
            # handoff or survived a worker-side error. The guard keeps cleanup confined to
            # this run's non-resumable GPU result directory.
            for cleanup_path in cleanup_paths:
                try:
                    if (
                        cleanup_path.suffix.lower() == '.dat'
                        and self.operations._path_is_relative_to(cleanup_path, self.inputs.gpu_worker_result_dir)
                    ):
                        cleanup_path.unlink(missing_ok=True)
                except Exception:
                    pass
        if isinstance(task_obj, dict):
            if task_obj.get('result_mask_fallback_path') is not None:
                task_obj['result_mask_path'] = task_obj.get('result_mask_fallback_path')
            task_obj['result_conf_path'] = task_obj.get('result_conf_fallback_path')
            task_obj.pop('result_workspace_preallocated', None)
        if released_memfd is not None:
            self.state.gpu_worker_tile_dense_result_memfd_bytes_reserved = max(
                0,
                int(self.state.gpu_worker_tile_dense_result_memfd_bytes_reserved) - int(released_memfd),
            )
            self.operations.runtime_telemetry().gauge(
                'tile.dense_worker_result_memfd_bytes_reserved',
                int(self.state.gpu_worker_tile_dense_result_memfd_bytes_reserved),
            )
        if (
            released is None
            and released_memfd is None
            and reserved_at is None
            and workspaces is None
        ):
            return False
        if reserved_at is not None:
            retention_seconds = max(0.0, float(time.monotonic()) - float(reserved_at))
            self.state.gpu_worker_tile_dense_result_max_retention_seconds = max(
                float(self.state.gpu_worker_tile_dense_result_max_retention_seconds),
                float(retention_seconds),
            )
            self.operations.runtime_telemetry().add(
                'tile.dense_worker_result_retention_seconds_total',
                float(retention_seconds),
            )
            self.operations.runtime_telemetry().add('tile.dense_worker_result_retirements', 1)
            self.operations.runtime_telemetry().gauge(
                'tile.dense_worker_result_last_retention_seconds',
                float(retention_seconds),
            )
            self.operations.runtime_telemetry().gauge(
                'tile.dense_worker_result_max_retention_seconds',
                float(self.state.gpu_worker_tile_dense_result_max_retention_seconds),
            )
            if reason:
                self.operations.runtime_telemetry().add(
                    f'tile.dense_worker_result_retired_reason.'
                    f'{self.operations._sanitize_filesystem_token(reason)}',
                    1,
                )
            warn_seconds = float(self.operations.tile_dense_worker_result_warn_seconds())
            if warn_seconds > 0.0 and retention_seconds >= warn_seconds:
                print(
                    f'Warning: array-backed tile worker result task {task_id_i} remained live for '
                    f'{retention_seconds:.1f}s before {reason or "retirement"}; '
                    'the backing has now been closed and deleted. '
                    'YOLO_TTA_TILE_DENSE_RESULT_WARN_SECONDS adjusts this diagnostic.'
                )
        if released is not None:
            self.state.gpu_worker_tile_dense_result_bytes_reserved = max(
                0, int(self.state.gpu_worker_tile_dense_result_bytes_reserved) - int(released),
            )
            self.operations.runtime_telemetry().add('tile.dense_worker_result_bytes_retired', int(released))
            self.operations.runtime_telemetry().gauge(
                'tile.dense_worker_result_bytes_reserved',
                int(self.state.gpu_worker_tile_dense_result_bytes_reserved),
            )
            self.operations.runtime_telemetry().gauge(
                'tile.dense_worker_result_tasks_reserved',
                int(len(self.state.gpu_worker_tile_dense_result_reservations)),
            )
        if bool(refill) and self.state.gpu_worker_pending_task_ids:
            self.dispatch_inference_windows()
        return True

    def release_tile_dense_result_for_key(self,
        model_name_s: str, view_name_s: str, tile_id_s: str, *, reason: str = '',
    ) -> bool:
        key = (str(model_name_s), str(view_name_s), str(tile_id_s))
        task_id = self.state.gpu_worker_tile_task_id_by_key.get(key)
        if task_id is None:
            return False
        pending_ids = self.state.gpu_worker_tile_pending_result_ids_by_task.get(int(task_id))
        if pending_ids is not None:
            pending_ids.discard(str(tile_id_s))
            self.state.gpu_worker_tile_task_id_by_key.pop(key, None)
            if pending_ids:
                return False
            self.state.gpu_worker_tile_pending_result_ids_by_task.pop(int(task_id), None)
        did_release = self.release_tile_dense_result_task_id(
            int(task_id), reason=str(reason), refill=True,
        )
        return bool(did_release)

    def gpu_worker_task_seconds(self, task: Dict[str, object]) -> float:
        view_obj = task.get('view')
        count = max(1, int(task.get('slice_count', 1)))
        key = self.operations.gpu_worker_task_cost_key(task)
        sec_per_frame = self.state.gpu_worker_seconds_per_frame_ewma.get(key)
        if sec_per_frame is None:
            sec_per_frame = (
                self.operations.gpu_worker_default_seconds_per_frame(view_obj)
                if isinstance(view_obj, ViewInfo) else 0.05
            )
        return max(1e-4, float(sec_per_frame) * float(count))

    def update_gpu_worker_cost(self, task: Dict[str, object], stats: Dict[str, object]) -> None:
        elapsed = float(stats.get('worker_compute_seconds', 0.0) or 0.0)
        count = max(1, int(task.get('slice_count', 1)))
        units = max(1, int(count))
        if elapsed <= 0.0:
            return
        observed = max(1e-5, float(elapsed) / float(units))
        key = self.operations.gpu_worker_task_cost_key(task)
        prior = self.state.gpu_worker_seconds_per_frame_ewma.get(key)
        alpha = min(0.8, max(0.05, self.operations._env_float('YOLO_TTA_GPU_WORKER_COST_EWMA_ALPHA', 0.30)))
        self.state.gpu_worker_seconds_per_frame_ewma[key] = (
            observed if prior is None else (1.0 - alpha) * float(prior) + alpha * observed
        )

    def split_gpu_worker_task_to_runtime_target(self, task_id: int) -> int:
        """Repeatedly split the selected full-frame lease to the current measured target."""
        current_id = int(task_id)
        task = self.state.gpu_worker_tasks_by_id[current_id]
        if str(task.get('kind', '')) != 'fullframe' or bool(task.get('disable_runtime_split', False)):
            return current_id
        min_slices = max(1, int(self.operations.gpu_worker_min_lease_slices()))
        align = max(1, int(self.inputs.gpu_batch))
        while True:
            count = int(task.get('slice_count', 0))
            if count < 2 * min_slices:
                break
            view_obj = task.get('view')
            key = self.operations.gpu_worker_task_cost_key(task)
            sec_per_frame = self.state.gpu_worker_seconds_per_frame_ewma.get(key)
            if sec_per_frame is None:
                sec_per_frame = (
                    self.operations.gpu_worker_default_seconds_per_frame(view_obj)
                    if isinstance(view_obj, ViewInfo) else 0.05
                )
            target_count = int(round(self.operations.gpu_worker_target_lease_seconds() / max(1e-5, float(sec_per_frame))))
            target_count = max(min_slices, min(self.operations.gpu_worker_max_lease_slices(), target_count))
            target_count = max(align, (int(target_count) // align) * align)
            if count <= max(2 * min_slices - 1, int(math.ceil(target_count * 1.25))):
                break
            start = int(task.get('slice_start', 0))
            stop = int(start + count)
            midpoint = min(stop - min_slices, start + max(min_slices, target_count))
            midpoint = start + ((midpoint - start) // align) * align
            if midpoint <= start or stop - midpoint < min_slices:
                break
            child_id = int(self.state.gpu_worker_next_dynamic_task_id)
            self.state.gpu_worker_next_dynamic_task_id += 1
            child = dict(task)
            task['slice_count'] = int(midpoint - start)
            task['render_workers'] = max(1, min(int(task.get('render_workers', 1)), int(midpoint - start)))
            child['task_id'] = child_id
            child['slice_start'] = int(midpoint)
            child['slice_count'] = int(stop - midpoint)
            child['render_workers'] = max(1, min(int(child.get('render_workers', 1)), int(stop - midpoint)))
            child['runtime_split_parent_task_id'] = current_id
            task['runtime_split_child_task_id'] = child_id
            if str(child.get('result_mode', 'file')) != 'direct_union':
                for field_name in ('result_mask_path', 'result_conf_path'):
                    raw_path = child.get(field_name)
                    if raw_path:
                        path_obj = Path(str(raw_path))
                        child[field_name] = str(path_obj.with_name(f'{path_obj.name}.rt{child_id}'))
            self.state.gpu_worker_tasks_by_id[child_id] = child
            self.state.gpu_worker_pending_task_ids.append(child_id)
            self.state.gpu_worker_total_tasks += 1
            parent_key = self.gpu_worker_fullframe_parent_key(task)
            if parent_key is not None:
                self.state.fullframe_remaining[parent_key] = int(self.state.fullframe_remaining.get(parent_key, 0)) + 1
                self.state.fullframe_task_ids_by_parent.setdefault(parent_key, []).append(int(child_id))
            self.operations.runtime_telemetry().add('scheduler.runtime_lease_splits', 1)
            # The selected front lease is now target-sized; leave the remainder central so
            # C3 can place it on the least-loaded eligible worker/owner.
            break
        return current_id

    def gpu_worker_fullframe_parent_key(self, task: Dict[str, object]) -> Optional[Tuple[str, str]]:
        if str(task.get('kind', '')) != 'fullframe':
            return None
        view_obj = task.get('view')
        view_name = getattr(view_obj, 'name', None)
        if view_name is None:
            return None
        return (str(task.get('model_name', '')), str(view_name))

    def hybrid_task_parent_key(self,
        task: Dict[str, object],
    ) -> Optional[Tuple[str, str]]:
        if not bool(task.get('hybrid_cpu_eligible_origin', False)):
            return None
        return self.gpu_worker_fullframe_parent_key(task)

    def hybrid_parent_state(self, parent: Optional[Tuple[str, str]]) -> str:
        if parent is None:
            return 'not_hybrid'
        return str(self.state.hybrid_view_mode_by_parent.get(parent, 'unclaimed'))

    def active_cpu_shared_parent(self) -> Optional[Tuple[str, str]]:
        active = [
            parent for parent, mode in self.state.hybrid_view_mode_by_parent.items()
            if str(mode) == 'direct_union'
            and int(self.state.fullframe_remaining.get(parent, 0)) > 0
        ]
        if len(active) > 1:
            raise RuntimeError(
                f'hybrid scheduler opened more than one CPU direct-union view: {active}'
            )
        return active[0] if active else None

    def hybrid_parent_is_cpu_reserved(self,
        parent: Optional[Tuple[str, str]],
    ) -> bool:
        return bool(parent is not None and parent in self.state.hybrid_cpu_reserved_parent_set)

    def next_cpu_reserved_parent(self) -> Optional[Tuple[str, str]]:
        """Return the first unfinished reserved parent that OpenVINO may still own."""
        active = self.active_cpu_shared_parent()
        if active is not None:
            return active
        for parent in self.state.hybrid_cpu_reserved_parents:
            if int(self.state.fullframe_remaining.get(parent, 0)) <= 0:
                continue
            state = self.hybrid_parent_state(parent)
            if state in {'unclaimed', 'direct_union'}:
                return parent
        return None

    def hybrid_task_is_active_cpu_assist(self,
        task: Dict[str, object],
        active_parent: Optional[Tuple[str, str]] = None,
    ) -> bool:
        parent = self.hybrid_task_parent_key(task)
        active = self.active_cpu_shared_parent() if active_parent is None else active_parent
        return bool(
            parent is not None
            and active is not None
            and parent == active
            and str(task.get('result_mode', 'file')) == 'direct_union'
        )

    def hybrid_task_is_gpu_mandatory(self, task: Dict[str, object]) -> bool:
        """True for ordinary GPU work and unreserved hybrid views assigned to CUDA D1."""
        parent = self.hybrid_task_parent_key(task)
        if parent is None:
            return True
        state = self.hybrid_parent_state(parent)
        if state == 'd1_owner':
            return True
        if state == 'direct_union':
            return False
        if state == 'unclaimed':
            return not self.hybrid_parent_is_cpu_reserved(parent)
        return True

    def set_hybrid_cpu_idle_reason(self, reason: str) -> None:
        """Record OpenVINO idle-state transitions without repeating one reason every lease."""
        normalized = str(reason).strip()
        now = float(time.monotonic())
        if not normalized:
            if self.state.hybrid_cpu_idle_active:
                elapsed = max(0.0, now - float(self.state.hybrid_cpu_idle_since or now))
                self.operations.runtime_telemetry().add('hybrid.cpu_idle_seconds', float(elapsed))
            self.state.hybrid_cpu_idle_active = False
            self.state.hybrid_cpu_idle_since = None
            return
        if self.state.hybrid_cpu_idle_active and normalized == self.state.hybrid_cpu_idle_reason_last:
            return
        if self.state.hybrid_cpu_idle_active:
            elapsed = max(0.0, now - float(self.state.hybrid_cpu_idle_since or now))
            self.operations.runtime_telemetry().add('hybrid.cpu_idle_seconds', float(elapsed))
        self.state.hybrid_cpu_idle_active = True
        self.state.hybrid_cpu_idle_since = now
        if normalized != self.state.hybrid_cpu_idle_reason_last:
            self.state.hybrid_cpu_idle_reason_last = normalized
            self.state.hybrid_cpu_idle_reason_counts[normalized] += 1
            self.operations.runtime_telemetry().add(
                f'hybrid.cpu_idle_reason.{self.operations._sanitize_filesystem_token(normalized)}', 1,
            )
            print(f'[hybrid] OpenVINO idle: {normalized}.')

    def describe_hybrid_cpu_idle_reason(self) -> str:
        active = self.active_cpu_shared_parent()
        pending_ids = [int(value) for value in self.state.gpu_worker_pending_task_ids]
        if active is not None:
            pending_active = [
                task_id for task_id in pending_ids
                if self.hybrid_task_parent_key(self.state.gpu_worker_tasks_by_id[int(task_id)]) == active
                and bool(self.state.gpu_worker_tasks_by_id[int(task_id)].get('cpu_eligible', False))
                and str(self.state.gpu_worker_tasks_by_id[int(task_id)].get('result_mode', 'file')) == 'direct_union'
            ]
            if pending_active:
                return (
                    f'active reserved view {active[0]}/{active[1]} has no currently '
                    'admissible CPU lease'
                )
            cpu_inflight = sum(self.cpu_worker_inflight(worker_id) for worker_id in self.state.cpu_task_queues)
            gpu_assist = sum(
                1 for task_id in self.state.gpu_worker_cpu_assist_inflight_task_ids
                if self.hybrid_task_parent_key(self.state.gpu_worker_tasks_by_id.get(int(task_id), {})) == active
            )
            return (
                f'waiting for active reserved view {active[0]}/{active[1]} to drain '
                f'({cpu_inflight} CPU lease(s), {gpu_assist} CUDA assist lease(s) in flight)'
            )
        next_parent = self.next_cpu_reserved_parent()
        if next_parent is not None:
            next_pending = [
                task_id for task_id in pending_ids
                if self.hybrid_task_parent_key(self.state.gpu_worker_tasks_by_id[int(task_id)]) == next_parent
                and bool(self.state.gpu_worker_tasks_by_id[int(task_id)].get('cpu_eligible', False))
            ]
            if next_pending:
                return (
                    f'next reserved view {next_parent[0]}/{next_parent[1]} is waiting '
                    'for direct-union admission'
                )
            return (
                f'next reserved view {next_parent[0]}/{next_parent[1]} has no central '
                'CPU-claimable lease'
            )
        if any(
            bool(self.state.gpu_worker_tasks_by_id[int(task_id)].get('hybrid_cpu_eligible_origin', False))
            for task_id in pending_ids
        ):
            return (
                'CPU reservation sequence exhausted; remaining CPU-compatible views are '
                'assigned to CUDA D1'
            )
        if pending_ids:
            return 'no CPU-compatible task remains in the central inference queue'
        return 'central inference queue is empty'

    def record_backend_frame_completion(self,
        task: Dict[str, object], backend: str,
    ) -> None:
        backend_name = str(backend).strip().lower()
        count = max(0, int(task.get('slice_count', 0)))
        if backend_name == 'cpu':
            self.state.cpu_frames_completed_total += int(count)
        else:
            self.state.gpu_frames_completed_total += int(count)
        parent = self.hybrid_task_parent_key(task)
        if parent is None:
            return
        if backend_name == 'cpu':
            self.state.hybrid_cpu_frames_completed_total += int(count)
        else:
            self.state.hybrid_gpu_frames_completed_total += int(count)
        holder = self.state.hybrid_view_frames_by_backend.get(parent)
        if holder is None:
            holder = Counter()
            self.state.hybrid_view_frames_by_backend[parent] = holder
        holder[backend_name] += int(count)
        task_holder = self.state.hybrid_view_tasks_by_backend.get(parent)
        if task_holder is None:
            task_holder = Counter()
            self.state.hybrid_view_tasks_by_backend[parent] = task_holder
        task_holder[backend_name] += 1

    def commit_hybrid_fullframe_mode(self,
        task: Dict[str, object], requested_mode: str, *, backend_label: str,
    ) -> str:
        """Commit every lease of one CPU-eligible view to one result contract."""
        requested = str(requested_mode)
        if requested not in {'d1_owner', 'direct_union'}:
            raise ValueError(f'invalid hybrid result mode {requested!r}')
        parent = self.hybrid_task_parent_key(task)
        if parent is None:
            return str(task.get('result_mode', 'file'))
        existing = self.state.hybrid_view_mode_by_parent.get(parent)
        if existing is not None:
            if str(existing) != requested:
                raise RuntimeError(
                    f'hybrid view {parent} was already committed to {existing}, '
                    f'cannot recommit it to {requested}'
                )
            return str(existing)
        if str(task.get('result_mode', 'file')) != self.inputs.hybrid_deferred_result_mode:
            raise RuntimeError(
                f'hybrid view {parent} reached first claim with result_mode='
                f'{task.get("result_mode")!r}'
            )
        task_ids = list(self.state.fullframe_task_ids_by_parent.get(parent, ()))
        if not task_ids:
            raise RuntimeError(f'hybrid view {parent} has no indexed full-frame tasks')
        if requested == 'direct_union' and not self.hybrid_parent_is_cpu_reserved(parent):
            raise RuntimeError(
                f'OpenVINO attempted to claim unreserved hybrid view {parent}; '
                'only the ordered CPU reservation sequence may open dense unions'
            )
        self.state.hybrid_view_mode_by_parent[parent] = requested
        changed = 0
        for indexed_task_id in task_ids:
            candidate = self.state.gpu_worker_tasks_by_id[int(indexed_task_id)]
            candidate_mode = str(candidate.get('result_mode', 'file'))
            if candidate_mode == self.inputs.hybrid_deferred_result_mode:
                candidate['result_mode'] = requested
                candidate['hybrid_committed_backend'] = str(backend_label)
                candidate['device_hole_fill'] = False
                if requested == 'd1_owner':
                    candidate['cpu_eligible'] = False
                changed += 1
            elif candidate_mode != requested:
                raise RuntimeError(
                    f'hybrid view {parent} contains mixed result contracts: '
                    f'{candidate_mode} vs {requested}'
                )
        self.operations.runtime_telemetry().add(f'hybrid.view_commits.{requested}', 1)
        reservation_note = (
            f', CPU reservation #{self.state.hybrid_cpu_reservation_rank_by_parent[parent] + 1}'
            if parent in self.state.hybrid_cpu_reservation_rank_by_parent else
            ', unreserved CUDA view'
        )
        print(
            f'[hybrid] first claim committed {parent[0]}/{parent[1]} to {requested} '
            f'via {backend_label}{reservation_note}; {changed} lease descriptor(s) updated.'
        )
        return requested

    def hybrid_gpu_selection_rank(self, task: Dict[str, object]) -> int:
        d1_parent = self.d1_task_parent_key(task)
        if d1_parent is not None and d1_parent in self.state.d1_groups_by_parent:
            return 0
        parent = self.hybrid_task_parent_key(task)
        mode = str(task.get('result_mode', 'file'))
        if parent is not None and mode == 'direct_union':
            return 1
        if mode == 'd1_owner' and self.d1_task_parent_key(task) in self.state.d1_owner_by_parent:
            return 2
        if parent is None:
            return 3
        if mode == 'd1_owner':
            return 4
        if mode == self.inputs.hybrid_deferred_result_mode and not self.hybrid_parent_is_cpu_reserved(parent):
            return 5
        if mode == self.inputs.hybrid_deferred_result_mode:
            return 6
        return 7

    def hybrid_gpu_stealback_quota(self,
        mandatory_gpu_pairs: Sequence[Tuple[int, int]],
        active_cpu_pairs: Sequence[Tuple[int, int]],
    ) -> int:
        """Return a proportional concurrent-task quota for the active CPU-owned view.

        Unopened reserved views are excluded from the CPU ETA. Unreserved hybrid views are
        mandatory CUDA D1 work. Only central leases from the one active direct-union view
        are assistable. The quota is expressed as live assist tasks, which maps much more
        closely to GPU-worker equivalents than multiplying by each worker's publication
        overlap depth.
        """
        active_parent = self.active_cpu_shared_parent()
        if (
            not self.operations.hybrid_gpu_stealback_enabled()
            or active_parent is None
            or not active_cpu_pairs
            or not self.state.cpu_task_queues
            or not self.state.gpu_task_queues
        ):
            return 0
        active_pairs = [
            (int(position), int(task_id))
            for position, task_id in active_cpu_pairs
            if self.hybrid_task_parent_key(self.state.gpu_worker_tasks_by_id[int(task_id)]) == active_parent
            and str(self.state.gpu_worker_tasks_by_id[int(task_id)].get('result_mode', 'file')) == 'direct_union'
        ]
        if not active_pairs:
            return 0

        gpu_workers = max(1, len(self.state.gpu_task_queues))
        cpu_workers = max(1, len(self.state.cpu_task_queues))
        cpu_committed = float(sum(
            float(predicted)
            for task_id, predicted in self.state.cpu_worker_task_predicted_seconds_by_id.items()
            if self.hybrid_task_parent_key(self.state.gpu_worker_tasks_by_id.get(int(task_id), {})) == active_parent
        ))
        cpu_pending = float(sum(
            self.cpu_worker_task_seconds(self.state.gpu_worker_tasks_by_id[int(task_id)])
            for _position, task_id in active_pairs
        ))
        cpu_work = max(0.0, float(cpu_committed) + float(cpu_pending))
        cpu_eta = float(cpu_work) / float(cpu_workers)

        # Count the complete central mandatory backlog, not only tasks feasible on the
        # particular free worker subset used by this one dispatch iteration. Otherwise
        # owner-affined D1 work can disappear from the horizon and trigger premature assist.
        gpu_committed = float(sum(
            float(predicted)
            for task_id, predicted in self.state.gpu_worker_task_predicted_seconds_by_id.items()
            if self.hybrid_task_is_gpu_mandatory(self.state.gpu_worker_tasks_by_id.get(int(task_id), {}))
        ))
        gpu_pending = float(sum(
            self.gpu_worker_task_seconds(self.state.gpu_worker_tasks_by_id[int(task_id)])
            for task_id in list(self.state.gpu_worker_pending_task_ids)
            if bool(self.state.gpu_worker_tasks_by_id[int(task_id)].get('gpu_eligible', self.inputs.gpu_worker_process_active))
            and self.hybrid_task_is_gpu_mandatory(self.state.gpu_worker_tasks_by_id[int(task_id)])
        ))
        gpu_mandatory_work = max(0.0, float(gpu_committed) + float(gpu_pending))
        gpu_horizon = float(gpu_mandatory_work) / float(gpu_workers)

        completed_cpu_samples = int(
            self.state.hybrid_view_tasks_by_backend.get(active_parent, Counter()).get('cpu', 0)
        )
        minimum_samples = int(self.operations.hybrid_gpu_stealback_min_cpu_samples())
        if gpu_mandatory_work > 0.0 and completed_cpu_samples < minimum_samples:
            self.operations.runtime_telemetry().gauge(
                'hybrid.gpu_assist_waiting_for_cpu_samples',
                {
                    'parent': f'{active_parent[0]}/{active_parent[1]}',
                    'completed': int(completed_cpu_samples),
                    'required': int(minimum_samples),
                },
            )
            return 0

        ratio = float(self.operations.hybrid_gpu_stealback_eta_ratio())
        threshold = (
            float(gpu_horizon) * float(ratio)
            + float(self.operations.hybrid_gpu_stealback_min_lead_seconds())
        )
        self.operations.runtime_telemetry().gauge('hybrid.active_cpu_eta_seconds', float(cpu_eta))
        self.operations.runtime_telemetry().gauge('hybrid.mandatory_gpu_eta_seconds', float(gpu_horizon))
        self.operations.runtime_telemetry().gauge('hybrid.active_cpu_pending_seconds', float(cpu_pending))
        self.operations.runtime_telemetry().gauge('hybrid.active_cpu_committed_seconds', float(cpu_committed))
        self.operations.runtime_telemetry().gauge('hybrid.active_cpu_samples', int(completed_cpu_samples))
        if cpu_eta <= threshold or cpu_pending <= 0.0:
            self.operations.runtime_telemetry().gauge('hybrid.gpu_assist_task_quota', 0)
            return 0

        # Transfer only the still-central CPU seconds needed to make the active view finish
        # near the mandatory-GPU horizon. Already-running OpenVINO leases are non-preemptive.
        target_cpu_eta = max(0.0, float(threshold))
        excess_cpu_seconds = min(
            float(cpu_pending),
            max(0.0, float(cpu_work) - float(target_cpu_eta) * float(cpu_workers)),
        )
        if excess_cpu_seconds <= 0.0:
            return 0
        active_gpu_seconds = float(sum(
            self.gpu_worker_task_seconds(self.state.gpu_worker_tasks_by_id[int(task_id)])
            for _position, task_id in active_pairs
        ))
        gpu_per_cpu_second = (
            float(active_gpu_seconds) / float(cpu_pending)
            if cpu_pending > 0.0 else 1.0
        )
        gpu_seconds_needed = max(
            0.0, float(excess_cpu_seconds) * max(1e-6, float(gpu_per_cpu_second)),
        )

        max_fraction = float(self.operations.hybrid_gpu_stealback_max_fraction())
        if max_fraction <= 0.0:
            return 0
        # The fraction cap protects mandatory GPU throughput during early assistance. Once
        # that backlog is empty, every otherwise-idle GPU may drain the active CPU view.
        max_assist_tasks = (
            int(gpu_workers)
            if gpu_mandatory_work <= 0.0 else
            max(1, min(
                int(gpu_workers),
                int(math.floor(float(gpu_workers) * float(max_fraction) + 1e-9)),
            ))
        )
        capacity_per_gpu = max(
            float(gpu_horizon),
            float(self.operations.gpu_worker_target_lease_seconds()),
            1e-3,
        )
        required_gpu_workers = max(
            1,
            int(math.ceil(float(gpu_seconds_needed) / float(capacity_per_gpu))),
        )
        quota = max(1, min(int(max_assist_tasks), int(required_gpu_workers)))
        self.operations.runtime_telemetry().gauge('hybrid.gpu_assist_task_quota', int(quota))
        self.operations.runtime_telemetry().gauge('hybrid.gpu_assist_seconds_needed', float(gpu_seconds_needed))
        if active_parent not in self.state.hybrid_stealback_announced_parents:
            self.state.hybrid_stealback_announced_parents.add(active_parent)
            print(
                'v17.0.3 active-view ETA GPU assist enabled for '
                f'{active_parent[0]}/{active_parent[1]}: active CPU ETA={cpu_eta:.1f}s, '
                f'mandatory-GPU ETA={gpu_horizon:.1f}s, '
                f'estimated GPU assist={gpu_seconds_needed:.1f}s, '
                f'CUDA assist quota={quota}/{gpu_workers} concurrent task(s), '
                f'CPU samples={completed_cpu_samples}.'
            )
        return int(quota)

    def d1_task_parent_key(self, task: Dict[str, object]) -> Optional[Tuple[str, str]]:
        if not bool(self.inputs.v1613_d1_owner_active):
            return None
        if str(task.get('result_mode', 'file')) != 'd1_owner':
            return None
        return self.gpu_worker_fullframe_parent_key(task)

    def d1_owner_groups_requested(self) -> bool:
        """Keep the experiment dark unless the v18.0.3 gate is explicitly nonzero."""
        return bool(
            self.inputs.v1613_d1_owner_active
            and int(self.inputs.gpu_device_count) > 1
            and self.operations._env_int('YOLO_TTA_V1803_D1_OWNER_GROUPS', 0) != 0
        )

    def d1_owner_group_size_limit(self) -> int:
        return max(
            1,
            min(
                8,
                int(self.inputs.gpu_device_count),
                self.operations._env_int(
                    'YOLO_TTA_V1803_D1_OWNER_GROUP_SIZE',
                    int(self.inputs.gpu_device_count),
                ),
            ),
        )

    def d1_owner_group_mode_active(self) -> bool:
        """Return whether the requested gate can form a real multi-worker group."""
        return bool(
            self.d1_owner_groups_requested()
            and int(self.d1_owner_group_size_limit()) >= 2
        )

    @staticmethod
    def _coalesce_d1_ranges(
        ranges: Sequence[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], ...]:
        merged: List[Tuple[int, int]] = []
        for start, stop in sorted((int(a), int(b)) for a, b in ranges):
            if start < 0 or stop <= start:
                raise ValueError(f'invalid D1 task range [{start},{stop})')
            if merged and start < merged[-1][1]:
                raise ValueError(
                    f'overlapping D1 task ranges {merged[-1]} and {(start, stop)}'
                )
            if merged and start == merged[-1][1]:
                merged[-1] = (merged[-1][0], stop)
            else:
                merged.append((start, stop))
        return tuple(merged)

    def _reject_d1_group_parent(
        self, parent: Tuple[str, str], reason: str,
    ) -> None:
        if parent in self.state.d1_group_fallback_parents:
            return
        self.state.d1_group_fallback_parents.add(parent)
        self.operations.runtime_telemetry().add('d1.group.admission_fallbacks', 1)
        print(
            f'v18.0.3 D1 owner group not admitted for {parent[0]}/{parent[1]} '
            f'({reason}); preserving the one-owner path.'
        )

    def ensure_d1_parent_group(
        self,
        task: Dict[str, object],
        candidate_workers: Sequence[int],
    ) -> Optional[D1ParentGroup]:
        """Atomically bind all existing parent leases to deterministic participants."""
        parent = self.d1_task_parent_key(task)
        if parent is None or not self.d1_owner_group_mode_active():
            return None
        existing = self.state.d1_groups_by_parent.get(parent)
        if existing is not None:
            return existing
        if parent in self.state.d1_group_fallback_parents:
            return None
        # v18.0.3 intentionally runs at most one multi-owner parent at a time.  This
        # preserves ordinary view-level parallelism on nonparticipants and, critically,
        # prevents a feasibility scan from reserving the same not-yet-claimed worker for
        # multiple parents.
        if self.state.d1_groups_by_parent:
            return None
        if parent in self.state.d1_owner_by_parent:
            # A one-owner task may already be queued/running because an earlier refill had
            # fewer than two idle candidates. Never mutate the remaining descriptors into a
            # group after that external dispatch boundary has been crossed.
            self._reject_d1_group_parent(parent, 'one-owner dispatch already committed')
            return None
        view = task.get('view')
        if not isinstance(view, ViewInfo):
            self._reject_d1_group_parent(parent, 'missing ViewInfo')
            return None
        task_ids = sorted(
            (int(value) for value in self.state.fullframe_task_ids_by_parent.get(parent, ())),
            key=lambda task_id: (
                int(self.state.gpu_worker_tasks_by_id[task_id].get('slice_start', 0)),
                int(task_id),
            ),
        )
        descriptors = [
            self.state.gpu_worker_tasks_by_id[int(task_id)] for task_id in task_ids
        ]
        if len(descriptors) < 2:
            self._reject_d1_group_parent(parent, 'fewer than two seed leases')
            return None
        if any(
            str(descriptor.get('result_mode', 'file')) != 'd1_owner'
            for descriptor in descriptors
        ):
            self._reject_d1_group_parent(parent, 'parent result contract is not fully committed')
            return None
        if any(
            str(descriptor.get('d1_view_shadow_store_dir', '') or '').strip()
            for descriptor in descriptors
        ):
            self._reject_d1_group_parent(
                parent, 'view-shadow output still requires one process-local writer',
            )
            return None
        task_ranges = [
            (
                int(descriptor.get('slice_start', 0)),
                int(descriptor.get('slice_start', 0))
                + int(descriptor.get('slice_count', 0)),
            )
            for descriptor in descriptors
        ]
        try:
            complete_ranges = self._coalesce_d1_ranges(task_ranges)
        except ValueError as exc:
            self._reject_d1_group_parent(parent, str(exc))
            return None
        if complete_ranges != ((0, int(view.num_slices)),):
            self._reject_d1_group_parent(
                parent,
                f'seed coverage {complete_ranges} is not exactly [0,{int(view.num_slices)})',
            )
            return None

        available = sorted({int(value) for value in candidate_workers})
        available = [
            worker for worker in available
            if (
                worker not in self.state.d1_active_parent_by_worker
                and self.gpu_worker_inflight(worker) == 0
            )
        ]
        participant_count = min(
            int(self.d1_owner_group_size_limit()), len(available), len(descriptors),
        )
        if participant_count < 2:
            # This is a transient dispatch condition rather than a permanent parent defect.
            # Leave it unmarked so a later refill with more idle workers may admit a group.
            return None
        ranked_workers = sorted(
            available,
            key=lambda worker: (
                float(self.state.gpu_worker_predicted_load_by_id.get(worker, 0.0)),
                int(self.gpu_worker_inflight(worker)),
                int(worker),
            ),
        )
        participants = tuple(ranked_workers[:participant_count])
        assigned_slices = {int(worker): 0 for worker in participants}
        task_worker_by_id: Dict[int, int] = {}
        ranges_by_worker: Dict[int, List[Tuple[int, int]]] = {
            int(worker): [] for worker in participants
        }
        for task_id, range_pair in zip(task_ids, task_ranges):
            worker = min(
                participants,
                key=lambda candidate: (
                    int(assigned_slices[int(candidate)]),
                    participants.index(int(candidate)),
                ),
            )
            task_worker_by_id[int(task_id)] = int(worker)
            ranges_by_worker[int(worker)].append(range_pair)
            assigned_slices[int(worker)] += int(range_pair[1] - range_pair[0])
        expected_ranges_by_worker = {
            int(worker): self._coalesce_d1_ranges(ranges)
            for worker, ranges in ranges_by_worker.items()
        }
        if any(not ranges for ranges in expected_ranges_by_worker.values()):
            raise RuntimeError('D1 group planner produced an empty participant')
        group_id = 'd1g-' + self.operations._sanitize_filesystem_token(
            f'{parent[0]}--{parent[1]}--task{int(task_ids[0])}'
        )
        group = D1ParentGroup(
            parent=parent,
            group_id=str(group_id),
            participants=tuple(int(value) for value in participants),
            leader_worker_id=int(participants[0]),
            task_worker_by_id=task_worker_by_id,
            expected_ranges_by_worker=expected_ranges_by_worker,
        )
        # Reservation is part of admission, not first dispatch.  Without this atomic
        # transition, another parent inspected during the same central-queue scan can
        # reuse a participant before its first lease reaches ``claim_d1_owner``.
        for worker in participants:
            reserved_parent = self.state.d1_active_parent_by_worker.get(int(worker))
            if reserved_parent is not None:
                raise RuntimeError(
                    f'D1 group planner selected already-reserved worker {worker}: '
                    f'{reserved_parent}'
                )
        fallback_root = (
            Path(self.inputs.gpu_worker_result_dir)
            / 'd1_group_fallback'
            / str(group.group_id)
        )
        for task_id in task_ids:
            descriptor = self.state.gpu_worker_tasks_by_id[int(task_id)]
            worker = int(task_worker_by_id[int(task_id)])
            rank = int(participants.index(worker))
            descriptor.update({
                'disable_runtime_split': True,
                'd1_group_id': str(group.group_id),
                'd1_group_rank': int(rank),
                'd1_group_size': int(len(participants)),
                'd1_group_worker_id': int(worker),
                'd1_group_leader_worker_id': int(group.leader_worker_id),
                'd1_group_participants': tuple(int(value) for value in participants),
                'd1_group_expected_ranges': tuple(
                    expected_ranges_by_worker[int(worker)]
                ),
                'd1_group_fallback_path': str(
                    fallback_root / f'participant_{rank:02d}.u32'
                ),
            })
        self.state.d1_groups_by_parent[parent] = group
        self.state.d1_group_parent_by_id[str(group.group_id)] = parent
        self.state.d1_group_parents_admitted_total += 1
        self.state.d1_owner_by_parent[parent] = int(group.leader_worker_id)
        for worker in participants:
            self.state.d1_active_parent_by_worker[int(worker)] = parent
        self.operations.runtime_telemetry().add('d1.group.parents_admitted', 1)
        self.operations.runtime_telemetry().gauge(
            'd1.group.last_participant_count', int(len(participants)),
        )
        print(
            f'v18.0.3 D1 owner group admitted for {parent[0]}/{parent[1]}: '
            f'workers={participants}, leases={len(task_ids)}, '
            f'assigned_slices={assigned_slices}.'
        )
        return group

    def plan_next_d1_parent_group(
        self,
        candidate_workers: Optional[Sequence[int]] = None,
    ) -> Optional[D1ParentGroup]:
        """Reserve one deterministic heavy D1 parent before any one-owner claim.

        Planning is centralized at a dispatch boundary.  ``d1_feasible_workers`` is
        deliberately read-only with respect to group creation because it is called while
        scanning the complete pending queue.
        """
        if not self.d1_owner_group_mode_active():
            return None
        if self.state.d1_groups_by_parent:
            return next(iter(self.state.d1_groups_by_parent.values()))
        workers = tuple(sorted({
            int(value) for value in (
                candidate_workers
                if candidate_workers is not None
                else tuple(self.state.gpu_task_queues)
            )
        }))
        idle_workers = tuple(
            worker for worker in workers
            if (
                worker not in self.state.d1_active_parent_by_worker
                and self.gpu_worker_inflight(worker) == 0
            )
        )
        if min(int(self.d1_owner_group_size_limit()), len(idle_workers)) < 2:
            self.operations.runtime_telemetry().gauge(
                'd1.group.waiting_idle_workers', int(len(idle_workers)),
            )
            return None

        pending_set = {int(value) for value in self.state.gpu_worker_pending_task_ids}
        ranked: List[Tuple[float, str, str, Tuple[str, str], Dict[str, object]]] = []
        for parent, indexed_ids in self.state.fullframe_task_ids_by_parent.items():
            if (
                parent in self.state.d1_owner_by_parent
                or parent in self.state.d1_group_fallback_parents
            ):
                continue
            descriptors = [
                self.state.gpu_worker_tasks_by_id[int(task_id)]
                for task_id in indexed_ids
                if int(task_id) in pending_set
            ]
            if not descriptors:
                continue
            first = descriptors[0]
            if self.d1_task_parent_key(first) != parent:
                continue
            predicted = float(sum(self.gpu_worker_task_seconds(task) for task in descriptors))
            ranked.append((
                -predicted, str(parent[0]), str(parent[1]), parent, first,
            ))
        for _negative_work, _model, _view, _parent, first in sorted(ranked):
            group = self.ensure_d1_parent_group(first, idle_workers)
            if group is not None:
                self.operations.runtime_telemetry().gauge(
                    'd1.group.active_planned_parents', 1,
                )
                return group
        return None

    def d1_feasible_workers(self,
        task: Dict[str, object], candidate_workers: Sequence[int],
    ) -> List[int]:
        workers = [int(value) for value in candidate_workers]
        if not bool(self.inputs.v1613_d1_owner_active):
            return workers
        parent = self.d1_task_parent_key(task)
        if parent is None:
            return [
                worker for worker in workers
                if int(worker) not in self.state.d1_active_parent_by_worker
            ]
        group = self.state.d1_groups_by_parent.get(parent)
        if group is None:
            owner = self.state.d1_owner_by_parent.get(parent)
            if owner is not None:
                return [int(owner)] if int(owner) in workers else []
        if group is not None:
            task_id = int(task.get('task_id', -1))
            assigned = group.task_worker_by_id.get(task_id)
            if assigned is None:
                raise RuntimeError(
                    f'D1 group {group.group_id} has no assignment for task {task_id}'
                )
            return [int(assigned)] if int(assigned) in workers else []
        owner = self.state.d1_owner_by_parent.get(parent)
        if owner is not None:
            return [int(owner)] if int(owner) in workers else []
        return [
            worker for worker in workers
            if int(worker) not in self.state.d1_active_parent_by_worker
        ]

    def claim_d1_owner(self, task: Dict[str, object], worker_id: int) -> bool:
        parent = self.d1_task_parent_key(task)
        if parent is None:
            return False
        worker = int(worker_id)
        group = self.state.d1_groups_by_parent.get(parent)
        if group is not None:
            assigned = group.task_worker_by_id.get(int(task.get('task_id', -1)))
            if assigned != worker:
                raise RuntimeError(
                    f'D1 group {group.group_id} task {task.get("task_id")} is bound to '
                    f'worker {assigned}, not {worker}'
                )
            active = self.state.d1_active_parent_by_worker.get(worker)
            if active != parent:
                raise RuntimeError(
                    f'D1 group worker {worker} lost its admission reservation for {parent}; '
                    f'active={active}'
                )
            newly_claimed = worker not in group.claimed_workers
            group.claimed_workers.add(worker)
            if newly_claimed:
                self.operations.runtime_telemetry().add('d1.group.participant_claims', 1)
            return bool(newly_claimed)
        owner = self.state.d1_owner_by_parent.get(parent)
        active = self.state.d1_active_parent_by_worker.get(worker)
        if owner is not None:
            if int(owner) != worker or active != parent:
                raise RuntimeError(
                    f'D1 owner registry mismatch for {parent}: owner={owner}, '
                    f'worker={worker}, active={active}'
                )
            return False
        if active is not None:
            raise RuntimeError(
                f'D1 worker {worker} cannot claim {parent}; it still owns {active}'
            )
        self.state.d1_owner_by_parent[parent] = worker
        self.state.d1_active_parent_by_worker[worker] = parent
        self.operations.runtime_telemetry().add('d1.owner_claims', 1)
        return True

    def rollback_d1_owner_claim(
        self, task: Dict[str, object], worker_id: int,
    ) -> None:
        parent = self.d1_task_parent_key(task)
        if parent is None:
            return
        worker = int(worker_id)
        group = self.state.d1_groups_by_parent.get(parent)
        if group is not None:
            group.claimed_workers.discard(worker)
            # Admission reserved every participant atomically.  A queue-put rollback
            # removes only the claim; the reservation must survive for the retried lease.
            return
        self.state.d1_owner_by_parent.pop(parent, None)
        self.state.d1_active_parent_by_worker.pop(worker, None)

    def release_d1_owner_if_complete(self,
        task: Dict[str, object], worker_id: int, stats: Dict[str, object],
    ) -> None:
        if not bool(stats.get('d1_view_complete', False)):
            return
        parent = self.d1_task_parent_key(task)
        if parent is None:
            return
        worker = int(worker_id)
        group = self.state.d1_groups_by_parent.get(parent)
        if group is not None:
            # Participant allocations remain live through reduction and are released only
            # after the final layer publication acknowledges every lease token.
            return
        owner = self.state.d1_owner_by_parent.get(parent)
        # A deferred publication sends compute_released first and the final result later.
        # The second notification is intentionally idempotent.
        if owner is None:
            return
        if int(owner) != worker or self.state.d1_active_parent_by_worker.get(worker) != parent:
            raise RuntimeError(
                f'D1 completion registry mismatch for {parent}: owner={owner}, '
                f'worker={worker}, active={self.state.d1_active_parent_by_worker.get(worker)}'
            )
        self.state.d1_owner_by_parent.pop(parent, None)
        self.state.d1_active_parent_by_worker.pop(worker, None)
        self.operations.runtime_telemetry().add('d1.owner_releases', 1)

    def split_one_gpu_worker_dispatch_tail(self,
        issue_slots: int,
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> bool:
        """Split one still-central full-frame lease only when issue reaches the tail."""
        if bool(self.inputs.v1613_d1_owner_active):
            return False
        if int(self.inputs.gpu_device_count) <= 1 or int(issue_slots) <= 0:
            return False
        pending_ids = [int(v) for v in self.state.gpu_worker_pending_task_ids]
        # This is the actual central-queue tail, not a per-view construction-time guess:
        # every pending descriptor would otherwise be issued by this refill.
        if not pending_ids or len(pending_ids) > int(issue_slots):
            return False
        parent_pending_counts: Dict[Tuple[str, str], int] = {}
        for task_id in pending_ids:
            parent_key = self.gpu_worker_fullframe_parent_key(self.state.gpu_worker_tasks_by_id[int(task_id)])
            if parent_key is not None:
                parent_pending_counts[parent_key] = int(parent_pending_counts.get(parent_key, 0)) + 1
        eligible: List[Tuple[int, int, int, int, int]] = []
        for position, task_id in enumerate(pending_ids):
            task = self.state.gpu_worker_tasks_by_id[int(task_id)]
            if str(task.get('kind', '')) != 'fullframe' or bool(task.get('tail_adapted', False)):
                continue
            midpoint = self.operations.gpu_worker_tail_split_point(
                int(task.get('slice_start', 0)),
                int(task.get('slice_count', 0)),
                int(self.inputs.gpu_batch),
            )
            if midpoint is None:
                continue
            parent_key = self.gpu_worker_fullframe_parent_key(task)
            remaining = int(self.state.fullframe_remaining.get(parent_key, 2 ** 31 - 1)) if parent_key is not None else 2 ** 31 - 1
            # Preserve the lease most likely to unlock parent postprocessing. In
            # particular, splitting preferred_parent here would turn its sole
            # pending lease into two and make _pop_gpu_worker_pending_task_id skip it.
            eligible.append((
                1 if parent_key == preferred_parent else 0,
                1 if parent_key is not None and int(parent_pending_counts.get(parent_key, 0)) == 1 else 0,
                -int(remaining), int(position), int(midpoint),
            ))
        if not eligible:
            return False
        _preferred, _sole_pending, _negative_remaining, position, midpoint = min(eligible)
        original_id = int(pending_ids[int(position)])
        original = self.state.gpu_worker_tasks_by_id[original_id]
        original_start = int(original.get('slice_start', 0))
        original_stop = int(original_start) + int(original.get('slice_count', 0))
        child_id = int(self.state.gpu_worker_next_dynamic_task_id)
        self.state.gpu_worker_next_dynamic_task_id += 1
        child = dict(original)
        original['slice_count'] = int(midpoint - original_start)
        original['render_workers'] = max(
            1, min(int(original.get('render_workers', 1)), int(midpoint - original_start)),
        )
        original['tail_adapted'] = True
        original['tail_child_task_id'] = int(child_id)
        child['task_id'] = int(child_id)
        child['slice_start'] = int(midpoint)
        child['slice_count'] = int(original_stop - midpoint)
        child['render_workers'] = max(
            1, min(int(child.get('render_workers', 1)), int(original_stop - midpoint)),
        )
        child['tail_adapted'] = True
        child['tail_parent_task_id'] = int(original_id)
        if str(child.get('result_mode', 'file')) != 'direct_union':
            for field_name in ('result_mask_path', 'result_conf_path'):
                raw_path = child.get(field_name)
                if raw_path:
                    path_obj = Path(str(raw_path))
                    child[field_name] = str(path_obj.with_name(
                        f'{path_obj.name}.tail{int(child_id)}'
                    ))
        self.state.gpu_worker_tasks_by_id[int(child_id)] = child
        pending_ids.insert(int(position) + 1, int(child_id))
        self.state.gpu_worker_pending_task_ids.clear()
        self.state.gpu_worker_pending_task_ids.extend(pending_ids)
        self.state.gpu_worker_total_tasks += 1
        parent_key = self.gpu_worker_fullframe_parent_key(original)
        if parent_key is not None:
            self.state.fullframe_remaining[parent_key] = int(self.state.fullframe_remaining.get(parent_key, 0)) + 1
            self.state.fullframe_task_ids_by_parent.setdefault(parent_key, []).append(int(child_id))
        print(
            f'v13.3.18 (C11): dispatch tail split task {original_id} '
            f'[{original_start}:{original_stop}] -> [{original_start}:{midpoint}] + '
            f'[{midpoint}:{original_stop}] as task {child_id}.'
        )
        return True

    def direct_union_task_key(self, task: Dict[str, object]) -> Optional[Tuple[str, str]]:
        if str(task.get('kind', '')) != 'fullframe' or str(task.get('result_mode', 'file')) != 'direct_union':
            return None
        view_obj = task.get('view')
        if view_obj is None:
            return None
        return (str(task.get('model_name', '')), str(getattr(view_obj, 'name', '')))

    def direct_union_task_bytes(self, task: Dict[str, object]) -> int:
        shape = tuple(int(v) for v in task.get('processing_shape', ()))
        if len(shape) != 3:
            view_obj = task['view']
            shape = self.operations.view_processing_volume_shape(view_obj, int(task.get('out_size', self.inputs.imgsz)))
        dense_volume_count = 2 if float(self.inputs.min_conf) > 0.0 else 1
        if bool(self.inputs.dense_tiling_active):
            dense_volume_count += 1
            if bool(self.inputs.nrrd_layers_needed):
                dense_volume_count += 2
        return int(self.operations.array_nbytes(shape, np.uint8)) * int(dense_volume_count)

    def direct_union_task_admissible(self, task: Dict[str, object]) -> bool:
        key = self.direct_union_task_key(task)
        if key is None or not self.inputs.direct_union_sparse_retirement_active:
            return True
        if key in self.state.direct_union_inference_views:
            lease = self.state.direct_union_backing_leases.get(key)
            if lease is None or lease.phase != 'inference':
                raise RuntimeError(f'direct-union inference registry is inconsistent for {key}')
            return True
        if key in self.state.direct_union_postprocess_views:
            # A task for a view whose final chunk already handed ownership to postprocess is
            # a scheduler lifecycle error; never write into a buffer now read by CPU/NRRD work.
            raise RuntimeError(f'inference task targeted postprocess-owned direct union {key}')
        if len(self.state.direct_union_inference_views) >= int(self.inputs.direct_union_inference_view_limit):
            return False
        need = int(self.direct_union_task_bytes(task))
        inference_active = int(sum(self.state.direct_union_inference_bytes.values()))
        postprocess_active = int(sum(self.state.direct_union_postprocess_bytes.values()))
        total_active = int(inference_active + postprocess_active)
        inference_ok = bool(
            not self.state.direct_union_inference_views
            or int(inference_active) + int(need) <= int(self.inputs.direct_union_inference_byte_limit)
        )
        total_ok = bool(
            not self.state.direct_union_backing_leases
            or int(total_active) + int(need) <= int(self.inputs.direct_union_total_dense_byte_limit)
        )
        return bool(inference_ok and total_ok)

    def activate_direct_union_task(self, task: Dict[str, object]) -> None:
        key = self.direct_union_task_key(task)
        if key is None:
            return
        view_obj = task['view']
        self.inputs.ensure_baseline_workspaces(str(key[0]), view_obj)
        task['result_mask_path'] = str(self.state.baseline_union_paths[key])
        conf_path = self.state.baseline_confmap_paths.get(key)
        task['result_conf_path'] = str(conf_path) if conf_path is not None else None

    def pop_gpu_worker_pending_task_id(self,
        preferred_parent: Optional[Tuple[str, str]] = None,
        candidate_workers: Optional[Sequence[int]] = None,
    ) -> Optional[Tuple[int, List[int]]]:
        """Pick an admissible GPU task and the worker subset allowed to own it.

        Unreserved hybrid parents are ordinary mandatory CUDA D1 work. Future CPU-reserved
        parents remain protected. The one active direct-union parent enters the candidate
        pool only when its active-view ETA quota has an open CUDA-assist slot.
        """
        pending_ids = [int(v) for v in self.state.gpu_worker_pending_task_ids]
        candidates = [int(v) for v in (candidate_workers or tuple(self.state.gpu_task_queues))]
        feasible_by_id: Dict[int, List[int]] = {}
        eligible: List[Tuple[int, int]] = []
        for position, task_id in enumerate(pending_ids):
            task = self.state.gpu_worker_tasks_by_id[int(task_id)]
            if not bool(task.get('gpu_eligible', self.inputs.gpu_worker_process_active)):
                continue
            if not self.direct_union_task_admissible(task):
                continue
            if not self.tile_dense_result_task_admissible(task):
                continue
            feasible = self.d1_feasible_workers(task, candidates)
            if not feasible:
                continue
            feasible_by_id[int(task_id)] = feasible
            eligible.append((int(position), int(task_id)))
        if not eligible:
            return None

        active_parent = self.active_cpu_shared_parent()
        mandatory_gpu: List[Tuple[int, int]] = []
        active_cpu_assist: List[Tuple[int, int]] = []
        for pair in eligible:
            task = self.state.gpu_worker_tasks_by_id[int(pair[1])]
            if self.hybrid_task_is_active_cpu_assist(task, active_parent):
                active_cpu_assist.append(pair)
            elif self.hybrid_task_is_gpu_mandatory(task):
                mandatory_gpu.append(pair)

        assist_ids: set[int] = set()
        quota = self.hybrid_gpu_stealback_quota(mandatory_gpu, active_cpu_assist)
        assist_slots_open = max(
            0,
            int(quota) - int(len(self.state.gpu_worker_cpu_assist_inflight_task_ids)),
        )
        if assist_slots_open > 0 and active_cpu_assist:
            selected_pool = list(active_cpu_assist)
            assist_ids = {int(task_id) for _position, task_id in active_cpu_assist}
        elif mandatory_gpu:
            selected_pool = list(mandatory_gpu)
        else:
            return None

        parent_pending_counts: Dict[Tuple[str, str], int] = {}
        for _position, task_id in selected_pool:
            parent_key = self.gpu_worker_fullframe_parent_key(self.state.gpu_worker_tasks_by_id[int(task_id)])
            if parent_key is not None:
                parent_pending_counts[parent_key] = int(parent_pending_counts.get(parent_key, 0)) + 1
        unlock_candidates: List[Tuple[int, int, int, int, int]] = []
        for position, task_id in selected_pool:
            task = self.state.gpu_worker_tasks_by_id[int(task_id)]
            parent_key = self.gpu_worker_fullframe_parent_key(task)
            if parent_key is None or int(parent_pending_counts.get(parent_key, 0)) != 1:
                continue
            unlock_candidates.append((
                self.hybrid_gpu_selection_rank(task),
                0 if parent_key == preferred_parent else 1,
                int(self.state.fullframe_remaining.get(parent_key, 2 ** 31 - 1)),
                int(position),
                int(task_id),
            ))
        if unlock_candidates:
            _hybrid_rank, _preferred, _remaining, _position, selected_id = min(unlock_candidates)
        else:
            parent_seconds: Dict[Optional[Tuple[str, str]], float] = {}
            for _position_i, task_id_i in selected_pool:
                task_i = self.state.gpu_worker_tasks_by_id[int(task_id_i)]
                parent_i = self.gpu_worker_fullframe_parent_key(task_i)
                parent_seconds[parent_i] = float(parent_seconds.get(parent_i, 0.0)) + self.gpu_worker_task_seconds(task_i)
            selected_id = min(
                selected_pool,
                key=lambda pair: (
                    self.hybrid_gpu_selection_rank(self.state.gpu_worker_tasks_by_id[int(pair[1])]),
                    0 if self.d1_task_parent_key(self.state.gpu_worker_tasks_by_id[int(pair[1])]) in self.state.d1_owner_by_parent else 1,
                    0 if (self.direct_union_task_key(self.state.gpu_worker_tasks_by_id[int(pair[1])]) in self.state.direct_union_inference_views) else 1,
                    self.inference_storage_priority_rank(self.state.gpu_worker_tasks_by_id[int(pair[1])]),
                    -float(parent_seconds.get(self.gpu_worker_fullframe_parent_key(self.state.gpu_worker_tasks_by_id[int(pair[1])]), 0.0)),
                    -float(self.gpu_worker_task_seconds(self.state.gpu_worker_tasks_by_id[int(pair[1])])),
                    int(pair[0]),
                ),
            )[1]
        selected_task = self.state.gpu_worker_tasks_by_id[int(selected_id)]
        selected_task['hybrid_gpu_assist_dispatch'] = bool(int(selected_id) in assist_ids)
        self.state.gpu_worker_pending_task_ids.remove(int(selected_id))
        return int(selected_id), list(feasible_by_id[int(selected_id)])

    def publish_gpu_worker_admissible_backlog(self) -> None:
        """Keep D1 from winning a device while dispatchable inference remains central."""
        admissible = False
        for pending_task_id in list(self.state.gpu_worker_pending_task_ids):
            pending_task = self.state.gpu_worker_tasks_by_id[int(pending_task_id)]
            gpu_policy_eligible = bool(
                self.hybrid_task_is_gpu_mandatory(pending_task)
                or self.hybrid_task_is_active_cpu_assist(pending_task)
            )
            if (
                gpu_policy_eligible
                and self.direct_union_task_admissible(pending_task)
                and self.tile_dense_result_task_admissible(pending_task)
            ):
                admissible = True
                break
        self.operations._set_main_process_gpu_pending_inference(bool(admissible))

    def gpu_worker_inflight(self, worker_id: int) -> int:
        worker = int(worker_id)
        return max(
            0,
            int(self.state.gpu_worker_dispatched_by_id.get(worker, 0))
            - int(self.state.gpu_worker_compute_completed_by_id.get(worker, 0)),
        )

    def refresh_gpu_aux_interpolation_leases(self) -> None:
        """Lease warm CUDA worker interpreters only after global inference drain."""
        aux_pool = self.operations.gpu_worker_aux_interpolation_pool()
        worker_ids = sorted(int(worker_id) for worker_id in self.state.gpu_task_queues)
        if aux_pool is None or not worker_ids:
            return
        inference_tail_drained = bool(
            int(self.state.gpu_worker_total_tasks) > 0
            and int(self.state.gpu_worker_results_collected) >= int(self.state.gpu_worker_total_tasks)
        )
        if not inference_tail_drained:
            for worker_id in worker_ids:
                aux_pool.revoke_worker(worker_id)
            return
        for worker_id in worker_ids:
            if (
                self.gpu_worker_inflight(worker_id) == 0
                and self.operations._main_process_gpu_stage_can_dispatch_inference(worker_id)
            ):
                # Feeder exclusivity ends at global drain; post-inference interpolation may
                # reclaim the worker's full inherited allocation.
                aux_pool.enable_worker(worker_id, allow_full_cpu_affinity=True)
            else:
                aux_pool.revoke_worker(worker_id)

    def dispatch_gpu_worker_inference_window(self,
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Issue bounded, targeted worker leases with D1 owner affinity."""
        worker_ids = sorted(int(worker_id) for worker_id in self.state.gpu_task_queues)
        if not worker_ids:
            self.operations._set_main_process_gpu_pending_inference(False)
            return
        self.publish_gpu_worker_admissible_backlog()
        per_gpu = (
            max(1, min(4, self.operations._env_int('YOLO_TTA_GPU_WORKER_DISPATCH_WINDOW_PER_GPU', 2)))
            if bool(self.inputs.v1613_d1_owner_active) else max(
                1, self.operations._env_int('YOLO_TTA_GPU_WORKER_DISPATCH_WINDOW_PER_GPU', 2),
            )
        )
        aux_pool = self.operations.gpu_worker_aux_interpolation_pool()
        while self.state.gpu_worker_pending_task_ids:
            candidates: List[int] = []
            for worker_id in worker_ids:
                if self.gpu_worker_inflight(worker_id) >= int(per_gpu):
                    continue
                if not self.operations._main_process_gpu_stage_can_dispatch_inference(worker_id):
                    continue
                if aux_pool is not None and not aux_pool.revoke_worker(worker_id):
                    continue
                candidates.append(int(worker_id))
            if not candidates:
                break
            # Group admission must precede the pending-task feasibility scan and the first
            # one-owner claim.  It is repeated after each completed group so the same
            # controlled one-group-at-a-time experiment can cover the full inference run.
            self.plan_next_d1_parent_group(candidates)
            issue_slots = sum(
                max(0, int(per_gpu) - self.gpu_worker_inflight(worker_id))
                for worker_id in candidates
            )
            self.split_one_gpu_worker_dispatch_tail(int(issue_slots), preferred_parent)
            selected = self.pop_gpu_worker_pending_task_id(preferred_parent, candidates)
            if selected is None:
                break
            task_id, _precommit_feasible_workers = selected
            task_to_dispatch = self.state.gpu_worker_tasks_by_id[int(task_id)]
            cpu_assist_dispatch = bool(
                task_to_dispatch.get('hybrid_gpu_assist_dispatch', False)
            )
            if (
                bool(task_to_dispatch.get('hybrid_cpu_eligible_origin', False))
                and str(task_to_dispatch.get('result_mode', 'file')) == self.inputs.hybrid_deferred_result_mode
            ):
                self.commit_hybrid_fullframe_mode(
                    task_to_dispatch, 'd1_owner', backend_label='CUDA',
                )
            task_id = self.split_gpu_worker_task_to_runtime_target(int(task_id))
            task_to_dispatch = self.state.gpu_worker_tasks_by_id[int(task_id)]
            if str(task_to_dispatch.get('result_mode', 'file')) == self.inputs.hybrid_deferred_result_mode:
                raise RuntimeError('GPU dispatch retained an unresolved hybrid result contract')
            feasible_workers = self.d1_feasible_workers(task_to_dispatch, candidates)
            if not feasible_workers:
                self.state.gpu_worker_pending_task_ids.appendleft(int(task_id))
                break
            start_rank = int(self.state.gpu_worker_dispatch_cursor) % len(worker_ids)
            position = {
                worker_id: (worker_ids.index(worker_id) - start_rank) % len(worker_ids)
                for worker_id in feasible_workers
            }
            predicted_seconds = float(self.gpu_worker_task_seconds(task_to_dispatch))
            worker_id = min(
                feasible_workers,
                key=lambda value: (
                    float(self.state.gpu_worker_predicted_load_by_id.get(int(value), 0.0)) + predicted_seconds,
                    self.gpu_worker_inflight(value),
                    position[value],
                ),
            )
            self.state.gpu_worker_dispatch_cursor = (worker_ids.index(worker_id) + 1) % len(worker_ids)
            if not self.operations._main_process_gpu_stage_begin_inference(worker_id):
                self.state.gpu_worker_pending_task_ids.appendleft(int(task_id))
                continue
            owner_claimed = False
            tile_storage_reserved = False
            try:
                owner_claimed = self.claim_d1_owner(task_to_dispatch, int(worker_id))
                self.activate_direct_union_task(task_to_dispatch)
                tile_storage_reserved = self.reserve_tile_dense_result_task(task_to_dispatch)
                if tile_storage_reserved:
                    self.prepare_tile_dense_result_workspaces(task_to_dispatch)
                dispatch_task = dict(task_to_dispatch)
                dispatch_task.pop('hybrid_gpu_assist_dispatch', None)
                self.operations._attach_memfd_transfers_to_task(dispatch_task)
                self.operations.preflight_multiprocessing_payload(dispatch_task)
                self.state.gpu_task_queues[int(worker_id)].put(dispatch_task)
            except BaseException:
                if tile_storage_reserved:
                    self.release_tile_dense_result_task_id(
                        int(task_id), reason='dispatch failure', refill=False,
                    )
                if owner_claimed:
                    self.rollback_d1_owner_claim(task_to_dispatch, int(worker_id))
                self.state.gpu_worker_pending_task_ids.appendleft(int(task_id))
                self.operations._main_process_gpu_stage_finish_inference(worker_id)
                raise
            task_to_dispatch.pop('hybrid_gpu_assist_dispatch', None)
            if cpu_assist_dispatch:
                task_to_dispatch['hybrid_gpu_assist_dispatched'] = True
                self.state.gpu_worker_cpu_assist_inflight_task_ids.add(int(task_id))
                self.operations.runtime_telemetry().add('hybrid.gpu_assist_tasks_dispatched', 1)
                self.operations.runtime_telemetry().add(
                    'hybrid.gpu_assist_frames_dispatched',
                    int(task_to_dispatch.get('slice_count', 0)),
                )
            self.state.gpu_worker_dispatched_tasks += 1
            self.state.gpu_worker_dispatched_by_id[int(worker_id)] = int(
                self.state.gpu_worker_dispatched_by_id.get(int(worker_id), 0)
            ) + 1
            self.state.gpu_worker_task_predicted_seconds_by_id[int(task_id)] = float(predicted_seconds)
            self.state.gpu_worker_predicted_load_by_id[int(worker_id)] = float(
                self.state.gpu_worker_predicted_load_by_id.get(int(worker_id), 0.0)
            ) + float(predicted_seconds)
        self.publish_gpu_worker_admissible_backlog()
        self.refresh_gpu_aux_interpolation_leases()

    def cpu_worker_task_seconds(self, task: Dict[str, object]) -> float:
        view_obj = task.get('view')
        count = max(1, int(task.get('slice_count', 1)))
        key = ('cpu',) + tuple(self.operations.gpu_worker_task_cost_key(task))
        sec_per_frame = self.state.cpu_worker_seconds_per_frame_ewma.get(key)
        if sec_per_frame is None:
            sec_per_frame = (
                self.operations.cpu_worker_default_seconds_per_frame(view_obj)
                if isinstance(view_obj, ViewInfo) else 0.25
            )
        return max(1e-4, float(sec_per_frame) * float(count))

    def update_cpu_worker_cost(self, task: Dict[str, object], stats: Dict[str, object]) -> None:
        elapsed = float(stats.get('worker_compute_seconds', 0.0) or 0.0)
        count = max(1, int(task.get('slice_count', 1)))
        if elapsed <= 0.0:
            return
        observed = max(1e-5, float(elapsed) / float(count))
        key = ('cpu',) + tuple(self.operations.gpu_worker_task_cost_key(task))
        prior = self.state.cpu_worker_seconds_per_frame_ewma.get(key)
        alpha = min(0.8, max(0.05, self.operations._env_float('YOLO_TTA_CPU_WORKER_COST_EWMA_ALPHA', 0.30)))
        self.state.cpu_worker_seconds_per_frame_ewma[key] = (
            observed if prior is None else (1.0 - alpha) * float(prior) + alpha * observed
        )

    def split_cpu_worker_task_to_runtime_target(self, task_id: int) -> int:
        """Split an oversized seed only when OpenVINO actually claims it."""
        current_id = int(task_id)
        task = self.state.gpu_worker_tasks_by_id[current_id]
        if str(task.get('kind', '')) != 'fullframe' or bool(task.get('disable_runtime_split', False)):
            return current_id
        count = int(task.get('slice_count', 0))
        if count <= 1:
            return current_id
        view_obj = task.get('view')
        key = ('cpu',) + tuple(self.operations.gpu_worker_task_cost_key(task))
        sec_per_frame = self.state.cpu_worker_seconds_per_frame_ewma.get(key)
        if sec_per_frame is None:
            sec_per_frame = (
                self.operations.cpu_worker_default_seconds_per_frame(view_obj)
                if isinstance(view_obj, ViewInfo) else 0.25
            )
        align = max(1, int(self.inputs.cpu_batch))
        target_count = int(round(self.operations.cpu_worker_target_lease_seconds() / max(1e-5, float(sec_per_frame))))
        target_count = max(
            self.operations.cpu_worker_min_lease_slices(),
            min(self.operations.cpu_worker_max_lease_slices(), int(target_count)),
        )
        target_count = max(align, int(math.ceil(float(target_count) / float(align))) * align)
        if count <= target_count:
            return current_id
        remainder = int(count - target_count)
        # Do not manufacture a tiny tail merely to hit the target exactly. The supplied
        # workload's 57- and 44-frame GPU seeds therefore remain intact for OpenVINO,
        # filling its 18/20 asynchronous request pools without recreating 7/10-slice tasks.
        min_useful_remainder = max(
            self.operations.cpu_worker_min_lease_slices(),
            int(math.ceil(float(target_count) * 0.50)),
        )
        if remainder < int(min_useful_remainder):
            return current_id
        start = int(task.get('slice_start', 0))
        stop = int(start + count)
        split_at = int(start + target_count)
        child_id = int(self.state.gpu_worker_next_dynamic_task_id)
        self.state.gpu_worker_next_dynamic_task_id += 1
        child = dict(task)
        task['slice_count'] = int(split_at - start)
        task['render_workers'] = max(
            1, min(int(task.get('render_workers', 1)), int(split_at - start)),
        )
        child['task_id'] = int(child_id)
        child['slice_start'] = int(split_at)
        child['slice_count'] = int(stop - split_at)
        child['render_workers'] = max(
            1, min(int(child.get('render_workers', 1)), int(stop - split_at)),
        )
        child['cpu_runtime_split_parent_task_id'] = int(current_id)
        task['cpu_runtime_split_child_task_id'] = int(child_id)
        if str(child.get('result_mode', 'file')) != 'direct_union':
            for field_name in ('result_mask_path', 'result_conf_path'):
                raw_path = child.get(field_name)
                if raw_path:
                    path_obj = Path(str(raw_path))
                    child[field_name] = str(path_obj.with_name(f'{path_obj.name}.cpu{child_id}'))
        self.state.gpu_worker_tasks_by_id[int(child_id)] = child
        self.state.gpu_worker_pending_task_ids.append(int(child_id))
        self.state.gpu_worker_total_tasks += 1
        parent_key = self.gpu_worker_fullframe_parent_key(task)
        if parent_key is not None:
            self.state.fullframe_remaining[parent_key] = int(self.state.fullframe_remaining.get(parent_key, 0)) + 1
            self.state.fullframe_task_ids_by_parent.setdefault(parent_key, []).append(int(child_id))
        self.operations.runtime_telemetry().add('scheduler.cpu_claim_lease_splits', 1)
        return current_id

    def cpu_worker_inflight(self, worker_id: int) -> int:
        worker = int(worker_id)
        return max(
            0,
            int(self.state.cpu_worker_dispatched_by_id.get(worker, 0))
            - int(self.state.cpu_worker_results_by_id.get(worker, 0)),
        )

    def pop_cpu_worker_pending_task_id(self,
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> Optional[int]:
        """Select work from the active or next ordered CPU reservation only."""
        active_cpu_parent = self.active_cpu_shared_parent()
        next_reserved_parent = self.next_cpu_reserved_parent()
        reservation_policy_active = bool(self.state.hybrid_cpu_reserved_parents)
        eligible: List[Tuple[int, int]] = []
        for position, task_id in enumerate(list(self.state.gpu_worker_pending_task_ids)):
            task = self.state.gpu_worker_tasks_by_id[int(task_id)]
            if not bool(task.get('cpu_eligible', False)):
                continue
            if str(task.get('result_mode', 'file')) == 'd1_owner':
                continue
            hybrid_parent = self.hybrid_task_parent_key(task)
            hybrid_state = self.hybrid_parent_state(hybrid_parent)
            if hybrid_parent is not None:
                if hybrid_state == 'd1_owner':
                    continue
                if active_cpu_parent is not None:
                    if hybrid_parent != active_cpu_parent or hybrid_state != 'direct_union':
                        continue
                else:
                    if not reservation_policy_active or next_reserved_parent is None:
                        continue
                    if hybrid_parent != next_reserved_parent:
                        continue
                    if hybrid_state not in {'unclaimed', 'direct_union'}:
                        continue
            elif active_cpu_parent is not None or next_reserved_parent is not None:
                continue
            if not self.direct_union_task_admissible(task):
                continue
            if not self.tile_dense_result_task_admissible(task):
                continue
            eligible.append((int(position), int(task_id)))
        if not eligible:
            return None
        selected = min(
            eligible,
            key=lambda pair: (
                0 if self.hybrid_task_parent_key(
                    self.state.gpu_worker_tasks_by_id[int(pair[1])]
                ) == active_cpu_parent and active_cpu_parent is not None else 1,
                self.state.hybrid_cpu_reservation_rank_by_parent.get(
                    self.hybrid_task_parent_key(self.state.gpu_worker_tasks_by_id[int(pair[1])]),
                    2 ** 31 - 1,
                ),
                0 if self.gpu_worker_fullframe_parent_key(
                    self.state.gpu_worker_tasks_by_id[int(pair[1])]
                ) == preferred_parent else 1,
                self.operations.cpu_inference_task_priority(self.state.gpu_worker_tasks_by_id[int(pair[1])]),
                0 if self.direct_union_task_key(
                    self.state.gpu_worker_tasks_by_id[int(pair[1])]
                ) in self.state.direct_union_inference_views else 1,
                self.inference_storage_priority_rank(self.state.gpu_worker_tasks_by_id[int(pair[1])]),
                -int(self.state.gpu_worker_tasks_by_id[int(pair[1])].get('slice_count', 0)),
                int(pair[0]),
            ),
        )[1]
        self.state.gpu_worker_pending_task_ids.remove(int(selected))
        return int(selected)

    def dispatch_cpu_worker_inference_window(self,
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Issue one claim-time-sized range per socket from the CPU-opened view."""
        worker_ids = sorted(int(worker_id) for worker_id in self.state.cpu_task_queues)
        if not worker_ids:
            return
        while self.state.gpu_worker_pending_task_ids:
            available = [
                worker_id for worker_id in worker_ids
                if self.cpu_worker_inflight(worker_id) < 1
            ]
            if not available:
                self.set_hybrid_cpu_idle_reason('')
                break
            task_id = self.pop_cpu_worker_pending_task_id(preferred_parent)
            if task_id is None:
                self.set_hybrid_cpu_idle_reason(self.describe_hybrid_cpu_idle_reason())
                break
            task = self.state.gpu_worker_tasks_by_id[int(task_id)]
            if (
                bool(task.get('hybrid_cpu_eligible_origin', False))
                and str(task.get('result_mode', 'file')) == self.inputs.hybrid_deferred_result_mode
            ):
                self.commit_hybrid_fullframe_mode(
                    task, 'direct_union', backend_label='OpenVINO',
                )
            task = self.state.gpu_worker_tasks_by_id[int(task_id)]
            if str(task.get('result_mode', 'file')) == self.inputs.hybrid_deferred_result_mode:
                raise RuntimeError('CPU dispatch retained an unresolved hybrid result contract')
            if not self.direct_union_task_admissible(task):
                self.state.gpu_worker_pending_task_ids.appendleft(int(task_id))
                self.set_hybrid_cpu_idle_reason(self.describe_hybrid_cpu_idle_reason())
                break
            task_id = self.split_cpu_worker_task_to_runtime_target(int(task_id))
            task = self.state.gpu_worker_tasks_by_id[int(task_id)]
            start_rank = int(self.state.cpu_worker_dispatch_cursor) % len(worker_ids)
            position = {
                worker_id: (worker_ids.index(worker_id) - start_rank) % len(worker_ids)
                for worker_id in available
            }
            predicted_seconds = float(self.cpu_worker_task_seconds(task))
            worker_id = min(
                available,
                key=lambda value: (
                    float(self.state.cpu_worker_predicted_load_by_id.get(int(value), 0.0))
                    + predicted_seconds,
                    position[value],
                ),
            )
            self.state.cpu_worker_dispatch_cursor = (worker_ids.index(worker_id) + 1) % len(worker_ids)
            tile_storage_reserved = False
            try:
                self.activate_direct_union_task(task)
                tile_storage_reserved = self.reserve_tile_dense_result_task(task)
                if tile_storage_reserved:
                    self.prepare_tile_dense_result_workspaces(task)
                dispatch_task = dict(task)
                self.operations._attach_memfd_transfers_to_task(dispatch_task)
                self.operations.preflight_multiprocessing_payload(dispatch_task)
                self.state.cpu_task_queues[int(worker_id)].put(dispatch_task)
            except BaseException:
                if tile_storage_reserved:
                    self.release_tile_dense_result_task_id(
                        int(task_id), reason='CPU dispatch failure', refill=False,
                    )
                self.state.gpu_worker_pending_task_ids.appendleft(int(task_id))
                raise
            self.state.cpu_worker_dispatched_by_id[int(worker_id)] = int(
                self.state.cpu_worker_dispatched_by_id.get(int(worker_id), 0)
            ) + 1
            self.state.cpu_worker_task_predicted_seconds_by_id[int(task_id)] = float(predicted_seconds)
            self.state.cpu_worker_predicted_load_by_id[int(worker_id)] = float(
                self.state.cpu_worker_predicted_load_by_id.get(int(worker_id), 0.0)
            ) + float(predicted_seconds)
            self.set_hybrid_cpu_idle_reason('')

    def dispatch_inference_windows(self,
        preferred_parent: Optional[Tuple[str, str]] = None,
    ) -> None:
        # Give each idle socket-local OpenVINO worker one claim-time-sized range from the
        # active or next ordered CPU reservation. CUDA fills every remaining slot with
        # mandatory GPU work and assists only the active direct-union view when its measured
        # ETA exceeds the mandatory-GPU horizon.
        # All ownership transitions and range claims run on this main thread.
        self.dispatch_cpu_worker_inference_window(preferred_parent)
        self.dispatch_gpu_worker_inference_window(preferred_parent)

    def _next_d1_group_control_task_id(self) -> int:
        task_id = int(self.state.d1_group_next_control_task_id)
        self.state.d1_group_next_control_task_id = int(task_id - 1)
        return int(task_id)

    def _d1_group_from_id(self, group_id: str) -> D1ParentGroup:
        parent = self.state.d1_group_parent_by_id.get(str(group_id))
        if parent is None:
            raise RuntimeError(f'unknown D1 owner group {group_id!r}')
        group = self.state.d1_groups_by_parent.get(parent)
        if group is None or str(group.group_id) != str(group_id):
            raise RuntimeError(f'inconsistent D1 owner group registry for {group_id!r}')
        return group

    def _register_d1_group_partial(
        self,
        group: D1ParentGroup,
        worker_id: int,
        payload: Mapping[str, object],
    ) -> None:
        worker = int(worker_id)
        if worker not in group.participants:
            raise RuntimeError(
                f'D1 group {group.group_id} received partial from foreign worker {worker}'
            )
        if str(payload.get('group_id', '')) != str(group.group_id):
            raise RuntimeError(
                f'D1 partial group {payload.get("group_id")!r} != {group.group_id!r}'
            )
        if int(payload.get('participant_worker_id', -1)) != worker:
            raise RuntimeError(
                f'D1 partial worker metadata {payload.get("participant_worker_id")} != {worker}'
            )
        expected_rank = int(group.participants.index(worker))
        if int(payload.get('participant_rank', -1)) != expected_rank:
            raise RuntimeError(
                f'D1 partial rank {payload.get("participant_rank")} != {expected_rank}'
            )
        expected_ranges = tuple(group.expected_ranges_by_worker[worker])
        actual_ranges = tuple(
            (int(raw[0]), int(raw[1]))
            for raw in payload.get('covered_ranges', ())  # type: ignore[union-attr]
        )
        if actual_ranges != expected_ranges:
            raise RuntimeError(
                f'D1 partial coverage {actual_ranges} != assigned {expected_ranges}'
            )
        prior = group.partial_artifacts_by_worker.get(worker)
        if prior is not None:
            if dict(prior) == dict(payload):
                return
            raise RuntimeError(
                f'D1 group {group.group_id} worker {worker} exported two partial leases'
            )
        group.partial_artifacts_by_worker[worker] = dict(payload)
        group.completed_workers.add(worker)
        self.operations.runtime_telemetry().add('d1.group.partials_ready', 1)

    def _dispatch_d1_group_reduction(
        self, group: D1ParentGroup, *, force_host: bool,
    ) -> None:
        artifacts_by_worker = (
            group.host_artifacts_by_worker if force_host
            else group.partial_artifacts_by_worker
        )
        if set(artifacts_by_worker) != set(group.participants):
            raise RuntimeError(
                f'D1 group {group.group_id} reduction is missing participant artifacts: '
                f'have={sorted(artifacts_by_worker)}, expected={group.participants}'
            )
        if group.reduction_dispatched:
            return
        control = {
            'task_type': 'd1_group_reduce',
            'task_id': self._next_d1_group_control_task_id(),
            'd1_group_id': str(group.group_id),
            'd1_group_leader_worker_id': int(group.leader_worker_id),
            'd1_group_force_host': bool(force_host),
            'd1_group_artifacts': tuple(
                dict(artifacts_by_worker[int(worker)])
                for worker in group.participants
            ),
        }
        self.operations.preflight_multiprocessing_payload(control)
        group.reduction_dispatched = True
        group.reduction_started_at = time.perf_counter()
        try:
            self.state.gpu_task_queues[int(group.leader_worker_id)].put(control)
        except BaseException:
            group.reduction_dispatched = False
            group.reduction_started_at = None
            raise
        self.operations.runtime_telemetry().add('d1.group.reductions_dispatched', 1)

    def _request_d1_group_host_fallback(
        self, group: D1ParentGroup, error: str,
    ) -> None:
        if group.host_fallback_requested:
            raise RuntimeError(
                f'D1 group {group.group_id} host fallback failed after IPC reduction: {error}'
            )
        group.host_fallback_requested = True
        group.reduction_dispatched = False
        group.reduction_started_at = None
        self.state.d1_group_host_fallbacks_total += 1
        self.operations.runtime_telemetry().add('d1.group.host_fallbacks', 1)
        print(
            f'Warning: D1 group {group.group_id} NVLink reduction failed ({error}); '
            'materializing retained participant bitsets for deterministic D2H reduction.'
        )
        fallback_root = (
            Path(self.inputs.gpu_worker_result_dir)
            / 'd1_group_fallback'
            / str(group.group_id)
        )
        for worker in group.participants:
            partial = group.partial_artifacts_by_worker[int(worker)]
            if str(partial.get('transport', '')) == 'host_path':
                group.host_artifacts_by_worker[int(worker)] = dict(partial)
                continue
            control = {
                'task_type': 'd1_group_export_host',
                'task_id': self._next_d1_group_control_task_id(),
                'd1_group_id': str(group.group_id),
                'd1_group_lease_token': str(partial.get('lease_token', '')),
                'd1_group_fallback_path': str(
                    fallback_root
                    / f'participant_{group.participants.index(int(worker)):02d}.u32'
                ),
            }
            self.operations.preflight_multiprocessing_payload(control)
            self.state.gpu_task_queues[int(worker)].put(control)
        if set(group.host_artifacts_by_worker) == set(group.participants):
            self._dispatch_d1_group_reduction(group, force_host=True)

    def _request_d1_group_release(self, group: D1ParentGroup) -> None:
        if group.release_pending_workers or group.release_ack_workers:
            raise RuntimeError(f'D1 group {group.group_id} release was requested twice')
        group.release_pending_workers.update(int(worker) for worker in group.participants)
        group.release_requested_at = time.perf_counter()
        for worker in group.participants:
            artifact = (
                group.host_artifacts_by_worker.get(int(worker))
                or group.partial_artifacts_by_worker.get(int(worker))
            )
            if artifact is None:
                raise RuntimeError(
                    f'D1 group {group.group_id} has no lease for worker {worker}'
                )
            control = {
                'task_type': 'd1_group_release',
                'task_id': self._next_d1_group_control_task_id(),
                'd1_group_id': str(group.group_id),
                'd1_group_lease_token': str(artifact.get('lease_token', '')),
            }
            self.operations.preflight_multiprocessing_payload(control)
            self.state.gpu_task_queues[int(worker)].put(control)

    def _complete_d1_group_release(self, group: D1ParentGroup) -> Dict[str, object]:
        if group.release_pending_workers:
            raise RuntimeError(
                f'D1 group {group.group_id} still awaits release acknowledgements from '
                f'{sorted(group.release_pending_workers)}'
            )
        if set(group.release_ack_workers) != set(group.participants):
            raise RuntimeError(
                f'D1 group {group.group_id} release acknowledgements are incomplete: '
                f'{sorted(group.release_ack_workers)}'
            )
        held = group.held_result
        if held is None:
            raise RuntimeError(
                f'D1 group {group.group_id} released without a held final result'
            )
        group.held_result = None
        release_seconds = max(
            0.0,
            time.perf_counter() - float(
                group.release_requested_at
                if group.release_requested_at is not None else time.perf_counter()
            ),
        )
        for worker in group.participants:
            active = self.state.d1_active_parent_by_worker.get(int(worker))
            if active != group.parent:
                raise RuntimeError(
                    f'D1 group {group.group_id} release changed worker {worker} owner: '
                    f'{active}'
                )
            self.state.d1_active_parent_by_worker.pop(int(worker), None)
        self.state.d1_owner_by_parent.pop(group.parent, None)
        self.state.d1_groups_by_parent.pop(group.parent, None)
        self.state.d1_group_parent_by_id.pop(str(group.group_id), None)
        self.state.d1_group_parents_released_total += 1
        self.state.d1_group_release_seconds_total += float(release_seconds)
        self.operations.runtime_telemetry().add('d1.group.parents_released', 1)
        self.operations.runtime_telemetry().add(
            'd1.group.partial_release_acks', int(len(group.release_ack_workers)),
        )
        self.operations.runtime_telemetry().gauge('d1.group.active_planned_parents', 0)
        print(
            f'v18.0.3 D1 owner group released for '
            f'{group.parent[0]}/{group.parent[1]}: '
            f'acks={len(group.release_ack_workers)}/{len(group.participants)}, '
            f'release={release_seconds:.3f}s.'
        )
        return dict(held)

    def _process_d1_group_control_result(self, msg: Dict[str, object]) -> None:
        mtype = str(msg.get('type'))
        group = self._d1_group_from_id(str(msg.get('group_id', '')))
        if mtype == 'd1_group_released':
            worker = int(msg.get('gpu_index', -1))
            if worker not in group.participants:
                raise RuntimeError(
                    f'D1 group {group.group_id} received a release ack from foreign '
                    f'worker {worker}'
                )
            if worker not in group.release_pending_workers:
                raise RuntimeError(
                    f'D1 group {group.group_id} received a duplicate/unrequested release '
                    f'ack from worker {worker}'
                )
            artifact = (
                group.host_artifacts_by_worker.get(worker)
                or group.partial_artifacts_by_worker.get(worker)
            )
            expected_token = str((artifact or {}).get('lease_token', ''))
            if str(msg.get('lease_token', '')) != expected_token:
                raise RuntimeError(
                    f'D1 group {group.group_id} release token changed for worker {worker}'
                )
            if not bool(msg.get('ok')):
                raise RuntimeError(
                    f'D1 group {group.group_id} worker {worker} failed to release its '
                    f'partial: {msg.get("error")}\n{msg.get("traceback")}'
                )
            if (
                worker != int(group.leader_worker_id)
                and not bool(msg.get('allocation_released', False))
            ):
                raise RuntimeError(
                    f'D1 group {group.group_id} nonleader worker {worker} acknowledged '
                    'release without finding its live partial allocation'
                )
            group.release_pending_workers.remove(worker)
            group.release_ack_workers.add(worker)
            if not group.release_pending_workers:
                held = self._complete_d1_group_release(group)
                # Re-enter only after every participant confirms release.  This is the
                # dispatch boundary that may reserve the same workers for the next group.
                self.process_one_worker_result(held)
            return
        if mtype == 'd1_group_host_exported':
            worker = int(msg.get('gpu_index', -1))
            if not bool(msg.get('ok')):
                raise RuntimeError(
                    f'D1 group {group.group_id} participant {worker} D2H export failed: '
                    f'{msg.get("error")}\n{msg.get("traceback")}'
                )
            payload = dict(msg.get('artifact') or {})
            original = group.partial_artifacts_by_worker.get(worker)
            if original is None:
                raise RuntimeError(
                    f'D1 group {group.group_id} received host artifact from unknown worker {worker}'
                )
            if (
                str(payload.get('transport', '')) != 'host_path'
                or str(payload.get('lease_token', '')) != str(original.get('lease_token', ''))
            ):
                raise RuntimeError(
                    f'D1 group {group.group_id} host artifact changed its lease identity'
                )
            group.host_artifacts_by_worker[worker] = payload
            if set(group.host_artifacts_by_worker) == set(group.participants):
                self._dispatch_d1_group_reduction(group, force_host=True)
            return
        if mtype != 'd1_group_reduced':
            raise RuntimeError(f'unsupported D1 group control result {mtype!r}')
        group.reduction_dispatched = False
        if not bool(msg.get('ok')):
            if bool(msg.get('fallback_allowed', False)):
                self._request_d1_group_host_fallback(
                    group, str(msg.get('error') or 'unknown CUDA-IPC error'),
                )
                return
            raise RuntimeError(
                f'D1 group {group.group_id} reduction/publication failed: '
                f'{msg.get("error")}\n{msg.get("traceback")}'
            )
        if group.held_result is None:
            raise RuntimeError(
                f'D1 group {group.group_id} completed without a held final logical result'
            )
        reduction_stats = dict(msg.get('stats') or {})
        transaction_seconds = max(
            0.0,
            time.perf_counter() - float(
                group.reduction_started_at
                if group.reduction_started_at is not None else time.perf_counter()
            ),
        )
        lifetime_seconds = max(0.0, time.perf_counter() - float(group.admitted_at))
        expected_tokens = {
            str(payload.get('lease_token', ''))
            for payload in group.partial_artifacts_by_worker.values()
        }
        actual_tokens = {
            str(value) for value in reduction_stats.get('d1_group_ack_tokens', ())
        }
        if actual_tokens != expected_tokens:
            raise RuntimeError(
                f'D1 group {group.group_id} acknowledgement tokens do not match exports'
            )
        held = dict(group.held_result)
        held_stats = dict(held.get('stats') or {})
        held_stats.update(reduction_stats)
        held_stats['d1_view_complete'] = True
        held['stats'] = held_stats
        transport = str(reduction_stats.get("d1_group_reduction_transport", "unknown"))
        bitset_words = int(reduction_stats.get("d1_bitset_words", 0) or 0)
        peer_or_seconds = float(reduction_stats.get('d1_group_peer_or_seconds', 0.0) or 0.0)
        d2h_seconds = float(reduction_stats.get('d1_group_d2h_seconds', 0.0) or 0.0)
        peer_read_bytes = (
            int(bitset_words) * 4 * max(0, len(group.participants) - 1)
            if transport == 'cuda_ipc_nvlink' else 0
        )
        self.state.d1_group_parents_reduced_total += 1
        self.state.d1_group_peer_read_bytes_total += int(peer_read_bytes)
        self.state.d1_group_reduction_seconds_total += float(transaction_seconds)
        self.state.d1_group_peer_or_seconds_total += float(peer_or_seconds)
        self.state.d1_group_d2h_seconds_total += float(d2h_seconds)
        print(
            f'v18.0.3 D1 owner group reduced for '
            f'{group.parent[0]}/{group.parent[1]}: '
            f'participants={len(group.participants)}, '
            f'transport={transport}, words={bitset_words or "unknown"}, '
            f'peer_read={int(peer_read_bytes) / float(self.inputs.gib):.3f}GiB, '
            f'peer_or={peer_or_seconds:.3f}s, d2h={d2h_seconds:.3f}s, '
            f'transaction={transaction_seconds:.3f}s, to_reduced={lifetime_seconds:.3f}s.'
        )
        self.operations.runtime_telemetry().add('d1.group.transaction_seconds', transaction_seconds)
        self.operations.runtime_telemetry().add('d1.group.peer_or_seconds', peer_or_seconds)
        self.operations.runtime_telemetry().add('d1.group.d2h_seconds', d2h_seconds)
        self.operations.runtime_telemetry().add('d1.group.lifetime_seconds', lifetime_seconds)
        group.held_result = held
        # The ordinary result stays held until all release controls are acknowledged.
        # Compute credit was already released and remains idempotent when it re-enters.
        self._request_d1_group_release(group)

    def d1_group_summary(self) -> Dict[str, int | float]:
        return {
            'admitted': int(self.state.d1_group_parents_admitted_total),
            'reduced': int(self.state.d1_group_parents_reduced_total),
            'released': int(self.state.d1_group_parents_released_total),
            'host_fallbacks': int(self.state.d1_group_host_fallbacks_total),
            'peer_read_bytes': int(self.state.d1_group_peer_read_bytes_total),
            'transaction_seconds': float(self.state.d1_group_reduction_seconds_total),
            'peer_or_seconds': float(self.state.d1_group_peer_or_seconds_total),
            'd2h_seconds': float(self.state.d1_group_d2h_seconds_total),
            'release_seconds': float(self.state.d1_group_release_seconds_total),
        }

    def process_one_worker_result(self, msg: Dict[str, object]) -> None:
        mtype = str(msg.get('type'))
        worker_kind = str(msg.get('worker_kind', 'gpu')).strip().lower()
        if mtype == 'ready':
            if worker_kind == 'cpu':
                ready_cpu_index = int(msg.get('cpu_index', -1))
                if ready_cpu_index not in self.state.cpu_worker_ready_details_by_id:
                    ready_details = {
                        'precision': str(msg.get('precision')),
                        'requests': int(msg.get('requests', 0) or 0),
                        'threads': int(msg.get('threads', 0) or 0),
                        'input_element_type': str(msg.get('input_element_type')),
                        'model_int8_quantized': bool(msg.get('model_int8_quantized')),
                        'class_count': msg.get('class_count'),
                        'amx_tile': bool(msg.get('amx_tile')),
                        'amx_bf16': bool(msg.get('amx_bf16')),
                        'amx_int8': bool(msg.get('amx_int8')),
                        'openvino_capabilities': list(msg.get('openvino_capabilities') or ()),
                        'model_xml': str(msg.get('model_xml')),
                    }
                    self.state.cpu_worker_ready_details_by_id[ready_cpu_index] = ready_details
                    self.operations.runtime_telemetry().gauge(
                        f'inference.cpu_instance.{ready_cpu_index}.ready', ready_details,
                    )
                print(
                    f"OpenVINO worker ready: instance {msg.get('cpu_index')} "
                    f"(pid {msg.get('pid')}, precision={msg.get('precision')}, "
                    f"requests={msg.get('requests')}, threads={msg.get('threads')}, "
                    f"input={msg.get('input_element_type')}, INT8-export={msg.get('model_int8_quantized')}, "
                    f"classes={msg.get('class_count')}, "
                    f"AMX(tile/bf16/int8)={msg.get('amx_tile')}/{msg.get('amx_bf16')}/{msg.get('amx_int8')}, "
                    f"OpenVINO capabilities={msg.get('openvino_capabilities')}, "
                    f"model={msg.get('model_xml')})."
                )
            else:
                print(f"GPU worker ready: device cuda:{msg.get('gpu_index')} (pid {msg.get('pid')}).")
            return
        if mtype == 'fatal':
            backend_label = (
                f"OpenVINO CPU instance {msg.get('cpu_index')}"
                if worker_kind == 'cpu' else f"GPU device {msg.get('gpu_index')}"
            )
            raise RuntimeError(
                f"{backend_label} failed to initialize: "
                f"{msg.get('error')}\n{msg.get('traceback')}"
            )
        if mtype in {
            'd1_group_reduced', 'd1_group_host_exported', 'd1_group_released',
        }:
            self._process_d1_group_control_result(msg)
            return
        if worker_kind == 'cpu':
            worker_id = int(msg.get('cpu_index', -1))
            task_id = int(msg.get('task_id', -1))
            predicted = float(self.state.cpu_worker_task_predicted_seconds_by_id.pop(task_id, 0.0))
            self.state.cpu_worker_predicted_load_by_id[worker_id] = max(
                0.0,
                float(self.state.cpu_worker_predicted_load_by_id.get(worker_id, 0.0)) - predicted,
            )
            self.state.cpu_worker_results_by_id[worker_id] = int(
                self.state.cpu_worker_results_by_id.get(worker_id, 0)
            ) + 1
            self.state.gpu_worker_results_collected += 1
            if not bool(msg.get('ok')):
                raise RuntimeError(
                    f"OpenVINO worker task {task_id} failed on CPU instance {worker_id}: "
                    f"{msg.get('error')}\n{msg.get('traceback')}"
                )
            task = self.state.gpu_worker_tasks_by_id[int(task_id)]
            stats = dict(msg.get('stats') or {})
            self.update_cpu_worker_cost(task, stats)
            self.record_backend_frame_completion(task, 'cpu')
            self.operations.runtime_telemetry().add('inference.cpu_tasks_completed', 1)
            self.operations.runtime_telemetry().add(
                'inference.cpu_frames_completed', int(task.get('slice_count', 0)),
            )
            # Apply the result first so the final lease can close this reservation and make
            # the next reserved parent visible before the just-freed OpenVINO worker refills.
            if str(task['kind']) == 'fullframe':
                self._result_callbacks().handle_fullframe_worker_result(task, stats)
            else:
                self._result_callbacks().handle_tile_worker_result(task, stats)
            self.dispatch_inference_windows(self.gpu_worker_fullframe_parent_key(task))
            self._result_callbacks().announce_process_inference_drain_if_complete()
            return

        if mtype == 'compute_released':
            worker_id = int(msg.get('gpu_index', -1))
            task_id = int(msg.get('task_id', -1))
            if task_id not in self.state.gpu_worker_compute_released_task_ids:
                self.state.gpu_worker_compute_released_task_ids.add(task_id)
                self.operations._main_process_gpu_stage_finish_inference(worker_id)
                self.state.gpu_worker_compute_completed_by_id[worker_id] = int(
                    self.state.gpu_worker_compute_completed_by_id.get(worker_id, 0)
                ) + 1
                predicted = float(self.state.gpu_worker_task_predicted_seconds_by_id.pop(task_id, 0.0))
                self.state.gpu_worker_predicted_load_by_id[worker_id] = max(
                    0.0, float(self.state.gpu_worker_predicted_load_by_id.get(worker_id, 0.0)) - predicted,
                )
                task_for_cost = self.state.gpu_worker_tasks_by_id.get(task_id)
                if isinstance(task_for_cost, dict):
                    release_stats = dict(msg.get('stats') or {})
                    self.update_gpu_worker_cost(task_for_cost, release_stats)
                    self.release_d1_owner_if_complete(task_for_cost, worker_id, release_stats)
            self.state.gpu_worker_cpu_assist_inflight_task_ids.discard(int(task_id))
            # Refill immediately; final result publication may still be copying several
            # GiB over PCIe or committing metadata on a retirement lane.
            task = self.state.gpu_worker_tasks_by_id.get(task_id)
            preferred = self.gpu_worker_fullframe_parent_key(task) if isinstance(task, dict) else None
            self.dispatch_inference_windows(preferred)
            self.refresh_gpu_aux_interpolation_leases()
            return
        if mtype == 'aux_result':
            # Route the targeted worker result to its waiting interpolation caller.  Once the
            # lease is free, newly ready inference can immediately revoke it and dispatch.
            worker_id = int(msg.get('gpu_index', -1))
            aux_pool = self.operations.gpu_worker_aux_interpolation_pool()
            if aux_pool is not None:
                aux_pool.complete(
                    int(msg.get('task_id', -1)),
                    worker_id,
                    bool(msg.get('ok')),
                    msg.get('stats') if isinstance(msg.get('stats'), dict) else None,
                    str(msg.get('error') or '') + ('\n' + str(msg.get('traceback')) if msg.get('traceback') else ''),
                )
            self.dispatch_inference_windows()
            self.refresh_gpu_aux_interpolation_leases()
            return
        worker_id = int(msg.get('gpu_index', -1))
        task_id = int(msg.get('task_id', -1))
        if task_id not in self.state.gpu_worker_compute_released_task_ids:
            self.state.gpu_worker_compute_released_task_ids.add(task_id)
            self.operations._main_process_gpu_stage_finish_inference(worker_id)
            self.state.gpu_worker_compute_completed_by_id[worker_id] = int(
                self.state.gpu_worker_compute_completed_by_id.get(worker_id, 0)
            ) + 1
            predicted = float(self.state.gpu_worker_task_predicted_seconds_by_id.pop(task_id, 0.0))
            self.state.gpu_worker_predicted_load_by_id[worker_id] = max(
                0.0, float(self.state.gpu_worker_predicted_load_by_id.get(worker_id, 0.0)) - predicted,
            )
            task_for_cost = self.state.gpu_worker_tasks_by_id.get(task_id)
            if isinstance(task_for_cost, dict):
                self.update_gpu_worker_cost(task_for_cost, dict(msg.get('stats') or {}))
        self.state.gpu_worker_cpu_assist_inflight_task_ids.discard(int(task_id))
        if not bool(msg.get('ok')):
            raise RuntimeError(
                f"GPU worker task {msg.get('task_id')} failed on device {msg.get('gpu_index')}: "
                f"{msg.get('error')}\n{msg.get('traceback')}"
            )
        task = self.state.gpu_worker_tasks_by_id[int(task_id)]
        stats = dict(msg.get('stats') or {})
        parent = self.d1_task_parent_key(task)
        group = self.state.d1_groups_by_parent.get(parent) if parent is not None else None
        partial_payload = stats.get('d1_group_partial_artifact')
        if group is not None and isinstance(partial_payload, Mapping):
            self._register_d1_group_partial(group, worker_id, partial_payload)
        if (
            group is not None
            and int(self.state.fullframe_remaining.get(group.parent, 0)) == 1
            and not bool(stats.get('d1_view_complete', False))
        ):
            if group.held_result is not None:
                raise RuntimeError(
                    f'D1 group {group.group_id} attempted to hold two final results'
                )
            if set(group.partial_artifacts_by_worker) != set(group.participants):
                raise RuntimeError(
                    f'D1 group {group.group_id} reached its final logical result before all '
                    f'participants exported: have={sorted(group.partial_artifacts_by_worker)}, '
                    f'expected={group.participants}'
                )
            held = dict(msg)
            held['stats'] = stats
            group.held_result = held
            if any(
                str(payload.get('transport', '')) != 'cuda_ipc'
                for payload in group.partial_artifacts_by_worker.values()
            ):
                self._request_d1_group_host_fallback(
                    group, 'one or more participants selected D2H during IPC export',
                )
            else:
                self._dispatch_d1_group_reduction(group, force_host=False)
            self.dispatch_inference_windows(group.parent)
            self.refresh_gpu_aux_interpolation_leases()
            return
        self.state.gpu_worker_results_collected += 1
        self.state.gpu_worker_results_by_id[worker_id] = int(
            self.state.gpu_worker_results_by_id.get(worker_id, 0)
        ) + 1
        self.record_backend_frame_completion(task, 'gpu')
        if bool(task.get('hybrid_gpu_assist_dispatched', False)):
            self.state.gpu_worker_cpu_assist_completed_task_ids.add(int(task_id))
            self.operations.runtime_telemetry().add('hybrid.gpu_assist_tasks_completed', 1)
            self.operations.runtime_telemetry().add(
                'hybrid.gpu_assist_frames_completed', int(task.get('slice_count', 0)),
            )
        self.release_d1_owner_if_complete(task, worker_id, stats)
        # Refill before scheduler-side memmap union/postprocess so a worker does not wait
        # behind CPU handling of the result that just freed its window slot. Prefer the
        # current parent when its last unissued lease can unlock postprocessing.
        self.dispatch_inference_windows(self.gpu_worker_fullframe_parent_key(task))
        if str(task['kind']) == 'fullframe':
            self._result_callbacks().handle_fullframe_worker_result(task, stats)
        else:
            self._result_callbacks().handle_tile_worker_result(task, stats)
        # A GPU may publish the final assisted direct-union lease. Refill once more after
        # result handling so OpenVINO can immediately open the next reservation.
        self.dispatch_inference_windows(self.gpu_worker_fullframe_parent_key(task))
        self._result_callbacks().announce_process_inference_drain_if_complete()
        self.refresh_gpu_aux_interpolation_leases()

    def configure_result_transport(
        self,
        *,
        push_drain_active: bool,
        track_thread: Callable[[threading.Thread, threading.Event], object],
    ) -> None:
        """Configure the sole process-result queue consumer for this run."""

        state = self.state
        if state.result_transport_configured:
            raise RuntimeError("TTA scheduler result transport is already configured")
        state.result_transport_configured = True
        state.push_drain_active = bool(push_drain_active)
        self.operations._set_main_process_gpu_stage_wake_callback(
            state.scheduler_wake.set
        )
        if not state.push_drain_active:
            return
        push_drain_thread = threading.Thread(
            target=self.push_drain_pump,
            name="inference-result-push-drain",
            daemon=True,
        )
        track_thread(push_drain_thread, state.push_drain_stop)
        push_drain_thread.start()
        print(
            "Scheduler push drain active (v13.3.8 G1; results handled the instant they "
            "arrive; YOLO_TTA_SCHEDULER_PUSH_DRAIN=0 restores polling)."
        )

    def wake_scheduler(self, _future: object = None) -> None:
        self.state.scheduler_wake.set()

    def hook_scheduler_wake(self, futures_list: Sequence[Future]) -> None:
        """Wake safely even when a future completes while callbacks are attached."""

        for future in futures_list:
            if future not in self.state.wake_hooked_futures:
                self.state.wake_hooked_futures.add(future)
                future.add_done_callback(self.wake_scheduler)

    def push_drain_pump(self) -> None:
        """Transport-only daemon loop; handlers remain on the scheduler thread."""

        state = self.state
        while not state.push_drain_stop.is_set():
            try:
                message = state.gpu_result_queue.get(timeout=0.5)  # type: ignore[attr-defined]
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            state.pushed_worker_results.append(message)
            state.scheduler_wake.set()

    def drain_process_inference_results(self) -> None:
        state = self.state
        if state.gpu_result_queue is None:
            return
        if state.push_drain_active:
            while state.pushed_worker_results:
                self.process_one_worker_result(state.pushed_worker_results.popleft())
            return
        while True:
            try:
                message = state.gpu_result_queue.get_nowait()  # type: ignore[attr-defined]
            except queue.Empty:
                break
            self.process_one_worker_result(message)

    def wait_for_one_process_result(self, timeout: float) -> None:
        state = self.state
        if state.gpu_result_queue is None:
            return
        if state.push_drain_active:
            if not state.pushed_worker_results:
                state.scheduler_wake.wait(timeout=float(timeout))
                state.scheduler_wake.clear()
            self.drain_process_inference_results()
            return
        try:
            message = state.gpu_result_queue.get(timeout=float(timeout))  # type: ignore[attr-defined]
        except queue.Empty:
            return
        self.process_one_worker_result(message)

    def process_inference_outstanding(self) -> bool:
        return bool(
            self.inputs.inference_worker_process_active
            and self.state.gpu_worker_results_collected
            < self.state.gpu_worker_total_tasks
        )

    def check_inference_workers_alive(self) -> None:
        """Fail fast when a selected backend process exits before global drain."""

        self._result_callbacks().check_parent_affinity()
        aux_pool = self.operations.gpu_worker_aux_interpolation_pool()
        aux_outstanding = int(aux_pool.outstanding()) if aux_pool is not None else 0
        if not self.process_inference_outstanding() and aux_outstanding <= 0:
            return
        remaining = max(
            0,
            int(
                self.state.gpu_worker_total_tasks
                - self.state.gpu_worker_results_collected
            ),
        )
        for backend_label, processes in (
            ("GPU", self.state.gpu_worker_processes),
            ("OpenVINO CPU", self.state.cpu_worker_processes),
        ):
            for process in processes:
                if process.is_alive():
                    continue
                reason = (
                    f"{backend_label} worker {getattr(process, 'name', '?')} exited "
                    f"unexpectedly (exitcode={getattr(process, 'exitcode', None)}) with "
                    f"{remaining} inference result(s) and {aux_outstanding} GPU-worker "
                    "auxiliary interpolation pass(es) still outstanding."
                )
                if aux_pool is not None:
                    aux_pool.mark_failed(reason)
                raise RuntimeError(reason)

    @staticmethod
    def reap_inference_worker_process(process: object, backend_label: str) -> None:
        try:
            process.join(timeout=30)  # type: ignore[attr-defined]
        except Exception:
            pass
        if not process.is_alive():  # type: ignore[attr-defined]
            return
        try:
            process.terminate()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            process.join(timeout=10)  # type: ignore[attr-defined]
        except Exception:
            pass
        if not process.is_alive() or not hasattr(process, "kill"):  # type: ignore[attr-defined]
            return
        try:
            process.kill()  # type: ignore[attr-defined]
            process.join(timeout=30)  # type: ignore[attr-defined]
        except Exception:
            pass
        if process.is_alive():  # type: ignore[attr-defined]
            print(
                f"WARNING: {backend_label} worker pid={getattr(process, 'pid', '?')} "
                "survived SIGKILL after 30s and is being abandoned. It is most likely "
                "stuck in D state; this node may need manual cleanup before it accepts "
                "another job.",
                flush=True,
            )

    def shutdown_inference_worker_processes(self) -> None:
        state = self.state
        processes = list(state.gpu_worker_processes) + list(state.cpu_worker_processes)
        if not processes:
            return
        # Fail auxiliary waiters before stopping workers so shutdown cannot strand a parent
        # postprocess future. This ordering is part of the process-owner contract.
        aux_pool = self.operations.gpu_worker_aux_interpolation_pool()
        if aux_pool is not None:
            aux_pool.mark_failed("Inference worker processes are shutting down")
            self.operations.set_gpu_worker_aux_interpolation_pool(None)
        for task_queue in list(state.gpu_task_queues.values()) + list(
            state.cpu_task_queues.values()
        ):
            try:
                task_queue.put(None)
            except Exception:
                pass
        for process in state.gpu_worker_processes:
            self.reap_inference_worker_process(process, "GPU")
        for process in state.cpu_worker_processes:
            self.reap_inference_worker_process(process, "OpenVINO CPU")

    def process_quiescence_issues(self) -> Dict[str, object]:
        """Return unresolved process-owner state that forbids scheduler completion."""

        state = self.state
        issues: Dict[str, object] = {}
        if state.gpu_worker_pending_task_ids:
            issues["pending_task_ids"] = list(state.gpu_worker_pending_task_ids)
        remaining_results = max(
            0, int(state.gpu_worker_total_tasks - state.gpu_worker_results_collected)
        )
        if remaining_results:
            issues["remaining_results"] = int(remaining_results)
        gpu_inflight = sum(
            max(
                0,
                int(state.gpu_worker_dispatched_by_id.get(worker_id, 0))
                - int(state.gpu_worker_compute_completed_by_id.get(worker_id, 0)),
            )
            for worker_id in state.gpu_task_queues
        )
        if gpu_inflight:
            issues["gpu_compute_inflight"] = int(gpu_inflight)
        cpu_inflight = sum(
            max(
                0,
                int(state.cpu_worker_dispatched_by_id.get(worker_id, 0))
                - int(state.cpu_worker_results_by_id.get(worker_id, 0)),
            )
            for worker_id in state.cpu_task_queues
        )
        if cpu_inflight:
            issues["cpu_inflight"] = int(cpu_inflight)
        if state.gpu_worker_tile_dense_result_reservations:
            issues["tile_dense_reservations"] = sorted(
                state.gpu_worker_tile_dense_result_reservations
            )
        if state.gpu_worker_tile_dense_result_memfd_reservations:
            issues["tile_memfd_reservations"] = sorted(
                state.gpu_worker_tile_dense_result_memfd_reservations
            )
        tile_ownership_task_ids = sorted(
            set(state.gpu_worker_tile_dense_result_reservations)
            | set(state.gpu_worker_tile_dense_result_memfd_reservations)
            | set(state.gpu_worker_tile_dense_result_reserved_at)
            | set(state.gpu_worker_tile_dense_result_workspaces)
            | set(state.gpu_worker_tile_pending_result_ids_by_task)
        )
        if tile_ownership_task_ids:
            issues["tile_ownership_task_ids"] = tile_ownership_task_ids
        if state.gpu_worker_tile_task_id_by_key:
            issues["tile_result_task_keys"] = sorted(
                state.gpu_worker_tile_task_id_by_key
            )
        if state.d1_owner_by_parent or state.d1_active_parent_by_worker:
            issues["d1_owners"] = {
                "by_parent": dict(state.d1_owner_by_parent),
                "by_worker": dict(state.d1_active_parent_by_worker),
            }
        if state.d1_groups_by_parent:
            issues['d1_groups'] = {
                str(group.group_id): {
                    'parent': tuple(group.parent),
                    'participants': tuple(group.participants),
                    'claimed': sorted(group.claimed_workers),
                    'completed': sorted(group.completed_workers),
                    'exported': sorted(group.partial_artifacts_by_worker),
                    'host_exported': sorted(group.host_artifacts_by_worker),
                    'reduction_dispatched': bool(group.reduction_dispatched),
                    'host_fallback_requested': bool(group.host_fallback_requested),
                    'held_final_result': bool(group.held_result is not None),
                    'release_pending': sorted(group.release_pending_workers),
                    'release_acked': sorted(group.release_ack_workers),
                }
                for group in state.d1_groups_by_parent.values()
            }
        if not (
            int(state.d1_group_parents_admitted_total)
            == int(state.d1_group_parents_reduced_total)
            == int(state.d1_group_parents_released_total)
        ):
            issues['d1_group_totals'] = self.d1_group_summary()
        if state.pushed_worker_results:
            issues["pushed_messages"] = int(len(state.pushed_worker_results))
        return issues

    def result(self) -> TtaSchedulerResult:
        state = self.state
        return TtaSchedulerResult(
            gpu_inference_drained_at=state.gpu_inference_drained_at,
            gpu_worker_results_collected=int(state.gpu_worker_results_collected),
            gpu_worker_total_tasks=int(state.gpu_worker_total_tasks),
            gpu_frames_completed_total=int(state.gpu_frames_completed_total),
            cpu_frames_completed_total=int(state.cpu_frames_completed_total),
            hybrid_gpu_frames_completed_total=int(
                state.hybrid_gpu_frames_completed_total
            ),
            hybrid_cpu_frames_completed_total=int(
                state.hybrid_cpu_frames_completed_total
            ),
            gpu_dispatched_by_worker=dict(state.gpu_worker_dispatched_by_id),
            gpu_completed_by_worker=dict(state.gpu_worker_results_by_id),
            cpu_dispatched_by_worker=dict(state.cpu_worker_dispatched_by_id),
            cpu_completed_by_worker=dict(state.cpu_worker_results_by_id),
            cpu_worker_ready_details={
                int(key): dict(value)
                for key, value in state.cpu_worker_ready_details_by_id.items()
            },
            hybrid_view_mode_by_parent=dict(state.hybrid_view_mode_by_parent),
            hybrid_view_frames_by_backend={
                key: dict(value)
                for key, value in state.hybrid_view_frames_by_backend.items()
            },
            hybrid_view_tasks_by_backend={
                key: dict(value)
                for key, value in state.hybrid_view_tasks_by_backend.items()
            },
            quiescence_issues=self.process_quiescence_issues(),
            artifacts=TtaSchedulerArtifacts(
                d1_layer_ref_by_parent=state.d1_layer_ref_by_parent,
                d1_view_shadow_path_by_parent=state.d1_view_shadow_path_by_parent,
            ),
        )


__all__ = [
    "TtaScheduler",
    "D1ParentGroup",
    "TtaSchedulerArtifacts",
    "TtaSchedulerCallbacks",
    "TtaSchedulerInputs",
    "TtaSchedulerOperations",
    "TtaSchedulerResult",
    "TtaSchedulerState",
]
