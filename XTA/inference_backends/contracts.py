"""Versioned, transport-neutral backend and scheduler contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable


BackendId: TypeAlias = str
TargetId: TypeAlias = str
TaskId: TypeAlias = str
LeaseId: TypeAlias = str


def _contract_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    """Return one transport-safe integer without silently truncating floats or booleans."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not a boolean")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be an integer: {value!r}")
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be an integer: {value!r}") from exc
    if resolved < int(minimum):
        raise ValueError(f"{field_name} must be >= {int(minimum)}")
    return resolved


def _enum_value(enum_type: type[Enum], value: object, field_name: str) -> Enum:
    """Canonicalize enum values arriving from JSON/RPC transports."""

    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"unsupported {field_name} {value!r}; expected one of: {choices}") from exc


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
        shape = tuple(_contract_int(value, "artifact shape", minimum=0) for value in self.shape)
        object.__setattr__(self, "shape", shape)

    @property
    def scheme(self) -> str:
        prefix, separator, _ = self.uri.partition(":")
        if separator and len(prefix) == 1 and prefix.isalpha():
            # Drive-qualified Windows paths are paths, not one-letter URI schemes.
            return "path"
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
        shape_mode = _enum_value(LeaseShapeMode, self.shape_mode, "lease shape mode")
        alignment = _contract_int(self.batch_alignment, "batch_alignment", minimum=1)
        min_slices = _contract_int(self.min_slices, "min_slices", minimum=1)
        max_slices = _contract_int(self.max_slices, "max_slices", minimum=1)
        buckets = tuple(
            _contract_int(value, "fixed lease bucket", minimum=1)
            for value in self.fixed_buckets
        )
        object.__setattr__(self, "shape_mode", shape_mode)
        object.__setattr__(self, "batch_alignment", alignment)
        object.__setattr__(self, "min_slices", min_slices)
        object.__setattr__(self, "max_slices", max_slices)
        object.__setattr__(self, "fixed_buckets", buckets)
        if max_slices < min_slices:
            raise ValueError("slice bounds must satisfy 1 <= min_slices <= max_slices")
        if any(value < min_slices or value > max_slices for value in buckets):
            raise ValueError("fixed lease buckets must remain inside the slice bounds")
        if tuple(sorted(set(buckets))) != buckets:
            raise ValueError("fixed lease buckets must be unique and strictly increasing")
        if any(value % alignment for value in buckets):
            raise ValueError("fixed lease buckets must satisfy batch_alignment")
        if shape_mode is LeaseShapeMode.FIXED_BUCKETS and not buckets:
            raise ValueError("FIXED_BUCKETS requires at least one compiled bucket")
        if shape_mode is LeaseShapeMode.FIXED_BUCKETS and buckets[-1] != max_slices:
            raise ValueError("the largest fixed lease bucket must equal max_slices")
        if shape_mode is not LeaseShapeMode.FIXED_BUCKETS and buckets:
            raise ValueError("fixed lease buckets require FIXED_BUCKETS shape mode")


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_kinds", frozenset(str(v) for v in self.task_kinds))
        object.__setattr__(self, "view_families", frozenset(str(v) for v in self.view_families))
        object.__setattr__(
            self,
            "pipeline_extents",
            frozenset(_enum_value(PipelineExtent, v, "pipeline extent") for v in self.pipeline_extents),
        )
        object.__setattr__(
            self,
            "result_contracts",
            frozenset(_enum_value(ResultContract, v, "result contract") for v in self.result_contracts),
        )
        object.__setattr__(
            self, "artifact_schemes", frozenset(str(v).strip().lower() for v in self.artifact_schemes)
        )
        object.__setattr__(
            self, "auxiliary_task_types", frozenset(str(v) for v in self.auxiliary_task_types)
        )
        object.__setattr__(
            self, "model_io_contracts", frozenset(str(v) for v in self.model_io_contracts)
        )

    def supports(self, requirements: "TaskRequirements") -> bool:
        return (
            requirements.task_kind in self.task_kinds
            and requirements.view_family in self.view_families
            and requirements.pipeline_extent in self.pipeline_extents
            and bool(requirements.acceptable_results & self.result_contracts)
            and requirements.model_io_contract in self.model_io_contracts
            and requirements.required_artifact_schemes.issubset(self.artifact_schemes)
            and (
                requirements.auxiliary_task_type is None
                or requirements.auxiliary_task_type in self.auxiliary_task_types
            )
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
        semantics = _enum_value(DispatchSemantics, self.semantics, "dispatch semantics")
        host_count = _contract_int(self.host_count, "host_count", minimum=1)
        world_size = _contract_int(self.world_size, "world_size", minimum=1)
        coordinator_rank = _contract_int(self.coordinator_rank, "coordinator_rank", minimum=0)
        object.__setattr__(self, "semantics", semantics)
        object.__setattr__(self, "host_count", host_count)
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "coordinator_rank", coordinator_rank)
        object.__setattr__(self, "host_arches", tuple(str(v) for v in self.host_arches))
        if not 0 <= coordinator_rank < world_size:
            raise ValueError("coordinator_rank must address one member of the target")
        if self.host_arches and len(self.host_arches) not in {1, host_count}:
            raise ValueError("host_arches must contain one shared value or one value per host")
        if semantics is DispatchSemantics.INDEPENDENT and (host_count != 1 or world_size != 1):
            raise ValueError("INDEPENDENT targets must describe exactly one host and one rank")


@dataclass(frozen=True)
class TaskRequirements:
    task_kind: str
    view_family: str
    pipeline_extent: PipelineExtent
    acceptable_results: frozenset[ResultContract]
    model_io_contract: str
    required_artifact_schemes: frozenset[str] = frozenset()
    auxiliary_task_type: str | None = None

    def __post_init__(self) -> None:
        if not self.task_kind or not self.view_family or not self.model_io_contract:
            raise ValueError("task kind, view family and model I/O contract are required")
        if not self.acceptable_results:
            raise ValueError("a task must accept at least one result contract")
        object.__setattr__(
            self, "pipeline_extent", _enum_value(PipelineExtent, self.pipeline_extent, "pipeline extent")
        )
        object.__setattr__(
            self,
            "acceptable_results",
            frozenset(_enum_value(ResultContract, v, "result contract") for v in self.acceptable_results),
        )
        object.__setattr__(
            self,
            "required_artifact_schemes",
            frozenset(str(v).strip().lower() for v in self.required_artifact_schemes),
        )
        if self.auxiliary_task_type is not None and not str(self.auxiliary_task_type).strip():
            raise ValueError("auxiliary_task_type must be non-empty when supplied")
        if self.auxiliary_task_type is not None:
            object.__setattr__(self, "auxiliary_task_type", str(self.auxiliary_task_type))


@dataclass(frozen=True)
class TaskEnvelope:
    schema_version: int
    run_id: str
    task_id: TaskId
    model_id: str
    requirements: TaskRequirements
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version", _contract_int(self.schema_version, "schema_version", minimum=1)
        )
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
        schema_version = _contract_int(self.schema_version, "schema_version", minimum=1)
        attempt = _contract_int(self.attempt, "attempt", minimum=0)
        slice_start = _contract_int(self.slice_start, "slice_start", minimum=0)
        logical_count = _contract_int(self.logical_slice_count, "logical_slice_count", minimum=1)
        execution_count = _contract_int(self.execution_slice_count, "execution_slice_count", minimum=1)
        result_contract = _enum_value(ResultContract, self.result_contract, "result contract")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "slice_start", slice_start)
        object.__setattr__(self, "logical_slice_count", logical_count)
        object.__setattr__(self, "execution_slice_count", execution_count)
        object.__setattr__(self, "result_contract", result_contract)
        if not self.lease_id or not self.task_id or not self.backend_id or not self.target_id:
            raise ValueError("lease, task, backend and target identifiers are required")
        if execution_count < logical_count:
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
        object.__setattr__(
            self, "schema_version", _contract_int(self.schema_version, "schema_version", minimum=1)
        )
        object.__setattr__(self, "attempt", _contract_int(self.attempt, "attempt", minimum=0))
        object.__setattr__(
            self, "event_type", _enum_value(WorkerEventType, self.event_type, "worker event type")
        )
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
