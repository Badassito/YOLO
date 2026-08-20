"""Dependency-free capability descriptions for the current local backends."""

from __future__ import annotations

from .contracts import (
    BackendCapabilities,
    DispatchSemantics,
    ExecutionTarget,
    LeasePolicy,
    LeaseShapeMode,
    PipelineExtent,
    ResultContract,
)


def cuda_local_target(index: int, *, host_arch: str = "") -> ExecutionTarget:
    """Describe one current CUDA worker; CUDA scale-out is intentionally not modeled."""

    logical_index = int(index)
    if logical_index < 0:
        raise ValueError("CUDA target index must be non-negative")
    return ExecutionTarget(
        target_id=f"cuda/{logical_index}",
        backend_id="cuda",
        semantics=DispatchSemantics.INDEPENDENT,
        host_count=1,
        world_size=1,
        coordinator_rank=0,
        host_arches=(str(host_arch),) if str(host_arch) else (),
        metadata={"logical_device_index": logical_index},
    )


def openvino_local_target(index: int, *, host_arch: str = "") -> ExecutionTarget:
    """Describe one current socket-local OpenVINO worker process."""

    instance_index = int(index)
    if instance_index < 0:
        raise ValueError("OpenVINO target index must be non-negative")
    return ExecutionTarget(
        target_id=f"openvino/{instance_index}",
        backend_id="openvino",
        semantics=DispatchSemantics.INDEPENDENT,
        host_count=1,
        world_size=1,
        coordinator_rank=0,
        host_arches=(str(host_arch),) if str(host_arch) else (),
        metadata={"instance_index": instance_index},
    )


def cuda_local_capabilities() -> BackendCapabilities:
    """Capabilities of the existing single-host, independently scheduled CUDA workers."""

    return BackendCapabilities(
        task_kinds=frozenset({"fullframe", "tile"}),
        view_families=frozenset({"orthogonal", "tilted", "radial", "tilted_radial"}),
        pipeline_extents=frozenset(
            {
                PipelineExtent.INFER_ONLY,
                PipelineExtent.PROJECT_INFER_FILTER,
                PipelineExtent.PROJECT_INFER_FILTER_BACKPROJECT,
            }
        ),
        result_contracts=frozenset(
            {
                ResultContract.TASK_ARTIFACT,
                ResultContract.SHARED_DISJOINT_UNION,
                ResultContract.SOURCE_SPACE_LAYER,
            }
        ),
        artifact_schemes=frozenset({"path", "file", "memfd"}),
        auxiliary_task_types=frozenset({"interpolation_pass"}),
        model_io_contracts=frozenset({"yolo-seg-raw-v1"}),
        lease_policy=LeasePolicy(
            shape_mode=LeaseShapeMode.ADAPTIVE,
            min_slices=1,
            max_slices=2**31 - 1,
        ),
    )


def openvino_local_capabilities() -> BackendCapabilities:
    """Capabilities of the existing socket-local OpenVINO workers."""

    return BackendCapabilities(
        task_kinds=frozenset({"fullframe", "tile"}),
        view_families=frozenset({"orthogonal", "tilted"}),
        pipeline_extents=frozenset({PipelineExtent.INFER_ONLY}),
        result_contracts=frozenset(
            {ResultContract.TASK_ARTIFACT, ResultContract.SHARED_DISJOINT_UNION}
        ),
        artifact_schemes=frozenset({"path", "file", "memfd"}),
        auxiliary_task_types=frozenset(),
        model_io_contracts=frozenset({"yolo-seg-raw-v1"}),
        lease_policy=LeasePolicy(
            shape_mode=LeaseShapeMode.ADAPTIVE,
            min_slices=1,
            max_slices=2**31 - 1,
        ),
    )
