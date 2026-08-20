"""Versioned, transport-neutral backend and scheduler contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable


BackendId: TypeAlias = str
TargetId: TypeAlias = str
TaskId: TypeAlias = str
LeaseId: TypeAlias = str


class DispatchSemantics(str, Enum):
    """How one scheduler-visible execution target consumes a lease."""

    INDEPENDENT = "independent"
    COLLECTIVE = "collective"


class PipelineExtent(str, Enum):
    """Largest contiguous pipeline region a backend can execute."""

    INFER_ONLY = "infer_only"
    PROJECT_INFER_FILTER = "project_infer_filter"
    PROJECT_INFER_FILTER_BACKPROJECT = "project_infer_filter_backproject"


class ResultContract(str, Enum):
    """Ownership and visibility of a completed lease result."""

    TASK_ARTIFACT = "task_artifact"
    SHARED_DISJOINT_UNION = "shared_disjoint_union"
    SOURCE_SPACE_LAYER = "source_space_layer"
    SHARDED_ARTIFACT = "sharded_artifact"


class LeaseShapeMode(str, Enum):
    """How a backend accepts work-window shapes."""

    ADAPTIVE = "adaptive"
    FIXED_BUCKETS = "fixed_buckets"
    WHOLE_TASK = "whole_task"


@dataclass(frozen=True)
class ArtifactRef:
    """Serializable reference to a large payload; never the payload itself."""

    uri: str
    format: str
    shape: tuple[int, ...]
    dtype: str
    mutable: bool = False

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("artifact URI must not be empty")
        if not self.format:
            raise ValueError("artifact format must not be empty")
        if any(int(value) < 0 for value in self.shape):
            raise ValueError(f"artifact shape must be non-negative: {self.shape!r}")

    @property
    def scheme(self) -> str:
        prefix, separator, _ = self.uri.partition(":")
        return prefix.lower() if separator else "path"


@dataclass(frozen=True)
class ModelArtifactRef:
    uri: str
    format: str
    semantic_model_id: str
    io_contract: str
    source_weights_digest: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("uri", self.uri),
            ("format", self.format),
            ("semantic_model_id", self.semantic_model_id),
            ("io_contract", self.io_contract),
        ):
            if not value:
                raise ValueError(f"model artifact {label} must not be empty")


@dataclass(frozen=True)
class LeasePolicy:
    shape_mode: LeaseShapeMode
    batch_alignment: int = 1
    min_slices: int = 1
    max_slices: int = 1
    fixed_buckets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if int(self.batch_alignment) < 1:
            raise ValueError("batch_alignment must be positive")
        if int(self.min_slices) < 1 or int(self.max_slices) < int(self.min_slices):
            raise ValueError("slice bounds must satisfy 1 <= min_slices <= max_slices")
        buckets = tuple(int(value) for value in self.fixed_buckets)
        if any(value < int(self.min_slices) or value > int(self.max_slices) for value in buckets):
            raise ValueError("fixed lease buckets must remain inside the slice bounds")
        if self.shape_mode is LeaseShapeMode.FIXED_BUCKETS and not buckets:
            raise ValueError("FIXED_BUCKETS requires at least one compiled bucket")


@dataclass(frozen=True)
class BackendCapabilities:
    task_kinds: frozenset[str]
    view_families: frozenset[str]
    pipeline_extents: frozenset[PipelineExtent]
    result_contracts: frozenset[ResultContract]
    artifact_schemes: frozenset[str]
    auxiliary_task_types: frozenset[str]
    model_io_contracts: frozenset[str]
    lease_policy: LeasePolicy

    def supports(self, requirements: "TaskRequirements") -> bool:
        return (
            requirements.task_kind in self.task_kinds
            and requirements.view_family in self.view_families
            and requirements.pipeline_extent in self.pipeline_extents
            and bool(requirements.acceptable_results & self.result_contracts)
            and requirements.model_io_contract in self.model_io_contracts
        )


@dataclass(frozen=True)
class ExecutionTarget:
    """One scheduler-visible target, possibly backed by a multi-host collective."""

    target_id: TargetId
    backend_id: BackendId
    semantics: DispatchSemantics
    host_count: int = 1
    world_size: int = 1
    coordinator_rank: int = 0
    host_arches: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_id or not self.backend_id:
            raise ValueError("target_id and backend_id must not be empty")
        if int(self.host_count) < 1 or int(self.world_size) < 1:
            raise ValueError("host_count and world_size must be positive")
        if not 0 <= int(self.coordinator_rank) < int(self.world_size):
            raise ValueError("coordinator_rank must address one member of the target")
        if self.host_arches and len(self.host_arches) not in {1, int(self.host_count)}:
            raise ValueError("host_arches must contain one shared value or one value per host")


@dataclass(frozen=True)
class TaskRequirements:
    task_kind: str
    view_family: str
    pipeline_extent: PipelineExtent
    acceptable_results: frozenset[ResultContract]
    model_io_contract: str

    def __post_init__(self) -> None:
        if not self.task_kind or not self.view_family or not self.model_io_contract:
            raise ValueError("task kind, view family and model I/O contract are required")
        if not self.acceptable_results:
            raise ValueError("a task must accept at least one result contract")


@dataclass(frozen=True)
class TaskEnvelope:
    schema_version: int
    run_id: str
    task_id: TaskId
    model_id: str
    requirements: TaskRequirements
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if int(self.schema_version) < 1:
            raise ValueError("schema_version must be positive")
        if not self.run_id or not self.task_id or not self.model_id:
            raise ValueError("run_id, task_id and model_id are required")


@dataclass(frozen=True)
class DispatchLease:
    schema_version: int
    lease_id: LeaseId
    task_id: TaskId
    attempt: int
    backend_id: BackendId
    target_id: TargetId
    slice_start: int
    logical_slice_count: int
    execution_slice_count: int
    result_contract: ResultContract
    inputs: Mapping[str, ArtifactRef] = field(default_factory=dict)
    outputs: Mapping[str, ArtifactRef] = field(default_factory=dict)
    backend_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.schema_version) < 1 or int(self.attempt) < 0:
            raise ValueError("schema_version must be positive and attempt non-negative")
        if not self.lease_id or not self.task_id or not self.backend_id or not self.target_id:
            raise ValueError("lease, task, backend and target identifiers are required")
        if int(self.slice_start) < 0 or int(self.logical_slice_count) < 1:
            raise ValueError("slice_start must be non-negative and logical count positive")
        if int(self.execution_slice_count) < int(self.logical_slice_count):
            raise ValueError("execution_slice_count cannot be smaller than logical_slice_count")


class WorkerEventType(str, Enum):
    READY = "ready"
    EXECUTION_RELEASED = "execution_released"
    COMPLETED = "completed"
    AUX_COMPLETED = "aux_completed"
    FAILED = "failed"
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True)
class WorkerEvent:
    schema_version: int
    event_type: WorkerEventType
    backend_id: BackendId
    target_id: TargetId
    worker_id: str
    task_id: TaskId | None = None
    lease_id: LeaseId | None = None
    attempt: int = 0
    outputs: Mapping[str, ArtifactRef] = field(default_factory=dict)
    stats: Mapping[str, object] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if int(self.schema_version) < 1 or int(self.attempt) < 0:
            raise ValueError("schema_version must be positive and attempt non-negative")
        if not self.backend_id or not self.target_id or not self.worker_id:
            raise ValueError("backend, target and worker identifiers are required")


@runtime_checkable
class InferenceBackend(Protocol):
    """Control-plane adapter; numerical execution stays backend-owned."""

    backend_id: BackendId
    capabilities: BackendCapabilities

    def start(self, event_sink: Callable[[WorkerEvent], None]) -> None: ...

    def targets(self) -> Sequence[ExecutionTarget]: ...

    def supports(self, requirements: TaskRequirements) -> bool: ...

    def prepare(self, task: TaskEnvelope, target_id: TargetId) -> DispatchLease: ...

    def submit(self, lease: DispatchLease) -> None: ...

    def healthy(self) -> bool: ...

    def shutdown(self) -> None: ...
