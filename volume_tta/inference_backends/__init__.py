"""Backend control-plane contracts.

Numerical implementations remain in the extracted CUDA/OpenVINO modules for this
behavior-preserving refactor.  New backends should enter through these contracts rather
than adding another CPU/GPU branch to the pipeline orchestrator.
"""

from .contracts import (
    ArtifactRef,
    BackendCapabilities,
    BackendId,
    DispatchLease,
    DispatchSemantics,
    ExecutionTarget,
    InferenceBackend,
    LeasePolicy,
    LeaseShapeMode,
    ModelArtifactRef,
    PipelineExtent,
    ResultContract,
    TaskEnvelope,
    TaskRequirements,
    WorkerEvent,
    WorkerEventType,
)
from .descriptors import (
    cuda_local_capabilities,
    cuda_local_target,
    openvino_local_capabilities,
    openvino_local_target,
)
from .registry import BackendRegistry


__all__ = (
    "ArtifactRef",
    "BackendCapabilities",
    "BackendId",
    "BackendRegistry",
    "DispatchLease",
    "DispatchSemantics",
    "ExecutionTarget",
    "InferenceBackend",
    "LeasePolicy",
    "LeaseShapeMode",
    "ModelArtifactRef",
    "PipelineExtent",
    "ResultContract",
    "TaskEnvelope",
    "TaskRequirements",
    "WorkerEvent",
    "WorkerEventType",
    "cuda_local_capabilities",
    "cuda_local_target",
    "openvino_local_capabilities",
    "openvino_local_target",
)
