from __future__ import annotations

import pickle
import unittest

from XTA.inference_backends import (
    ArtifactRef,
    BackendRegistry,
    DispatchLease,
    DispatchSemantics,
    ExecutionTarget,
    LeasePolicy,
    LeaseShapeMode,
    PipelineExtent,
    ResultContract,
    TaskRequirements,
    cuda_local_capabilities,
    cuda_local_target,
    openvino_local_capabilities,
)


class BackendContractTests(unittest.TestCase):
    def test_collective_is_one_scheduler_target(self) -> None:
        target = ExecutionTarget(
            target_id="tpu/slice-a",
            backend_id="tpu",
            semantics=DispatchSemantics.COLLECTIVE,
            host_count=4,
            world_size=16,
            coordinator_rank=0,
            host_arches=("aarch64",),
        )
        restored = pickle.loads(pickle.dumps(target))
        self.assertEqual(restored, target)
        self.assertEqual(restored.target_id, "tpu/slice-a")
        self.assertEqual(restored.world_size, 16)

    def test_fixed_bucket_lease_distinguishes_logical_and_execution_shapes(self) -> None:
        policy = LeasePolicy(
            shape_mode=LeaseShapeMode.FIXED_BUCKETS,
            min_slices=8,
            max_slices=64,
            fixed_buckets=(8, 16, 32, 64),
        )
        self.assertEqual(policy.fixed_buckets[-1], 64)
        lease = DispatchLease(
            schema_version=1,
            lease_id="lease-1",
            task_id="task-1",
            attempt=0,
            backend_id="future-collective",
            target_id="future-collective/slice-1",
            slice_start=0,
            logical_slice_count=23,
            execution_slice_count=32,
            result_contract=ResultContract.SHARDED_ARTIFACT,
            inputs={
                "volume": ArtifactRef(
                    uri="gs://bucket/run/input",
                    format="volume-u8",
                    shape=(100, 200, 200),
                    dtype="uint8",
                )
            },
        )
        self.assertEqual(lease.logical_slice_count, 23)
        self.assertEqual(lease.execution_slice_count, 32)

    def test_current_backend_capabilities_remain_explicit(self) -> None:
        radial = TaskRequirements(
            task_kind="fullframe",
            view_family="radial",
            pipeline_extent=PipelineExtent.INFER_ONLY,
            acceptable_results=frozenset({ResultContract.TASK_ARTIFACT}),
            model_io_contract="yolo-seg-raw-v1",
        )
        self.assertTrue(cuda_local_capabilities().supports(radial))
        self.assertFalse(openvino_local_capabilities().supports(radial))
        incompatible_model = TaskRequirements(
            task_kind="fullframe",
            view_family="orthogonal",
            pipeline_extent=PipelineExtent.INFER_ONLY,
            acceptable_results=frozenset({ResultContract.TASK_ARTIFACT}),
            model_io_contract="unknown-model-contract",
        )
        self.assertFalse(cuda_local_capabilities().supports(incompatible_model))
        cuda_target = cuda_local_target(3, host_arch="x86_64")
        self.assertEqual(cuda_target.semantics, DispatchSemantics.INDEPENDENT)
        self.assertEqual((cuda_target.host_count, cuda_target.world_size), (1, 1))

        orthogonal = TaskRequirements(
            task_kind="fullframe",
            view_family="orthogonal",
            pipeline_extent="infer_only",
            acceptable_results=frozenset({"task_artifact"}),
            model_io_contract="yolo-seg-raw-v1",
            required_artifact_schemes=frozenset({"memfd"}),
        )
        self.assertTrue(cuda_local_capabilities().supports(orthogonal))
        self.assertTrue(openvino_local_capabilities().supports(orthogonal))

    def test_transport_values_are_canonicalized_and_validated(self) -> None:
        policy = LeasePolicy(
            shape_mode="fixed_buckets",
            batch_alignment="8",
            min_slices="8",
            max_slices="32",
            fixed_buckets=(8, 16, 32),
        )
        self.assertIs(policy.shape_mode, LeaseShapeMode.FIXED_BUCKETS)
        self.assertEqual(policy.batch_alignment, 8)
        with self.assertRaisesRegex(ValueError, "requires at least one compiled bucket"):
            LeasePolicy(shape_mode="fixed_buckets", fixed_buckets=())
        with self.assertRaisesRegex(ValueError, "largest fixed lease bucket"):
            LeasePolicy(
                shape_mode="fixed_buckets",
                min_slices=1,
                max_slices=64,
                fixed_buckets=(8,),
            )
        with self.assertRaisesRegex(ValueError, "unsupported lease shape mode"):
            LeasePolicy(shape_mode="elastic")

    def test_capability_checks_include_transport_and_auxiliary_requirements(self) -> None:
        remote = TaskRequirements(
            task_kind="fullframe",
            view_family="orthogonal",
            pipeline_extent=PipelineExtent.INFER_ONLY,
            acceptable_results=frozenset({ResultContract.TASK_ARTIFACT}),
            model_io_contract="yolo-seg-raw-v1",
            required_artifact_schemes=frozenset({"gs"}),
        )
        self.assertFalse(cuda_local_capabilities().supports(remote))
        auxiliary = TaskRequirements(
            task_kind="fullframe",
            view_family="orthogonal",
            pipeline_extent=PipelineExtent.INFER_ONLY,
            acceptable_results=frozenset({ResultContract.TASK_ARTIFACT}),
            model_io_contract="yolo-seg-raw-v1",
            required_artifact_schemes=frozenset({"path"}),
            auxiliary_task_type="interpolation_pass",
        )
        self.assertTrue(cuda_local_capabilities().supports(auxiliary))
        self.assertFalse(openvino_local_capabilities().supports(auxiliary))

        tilted_radial = TaskRequirements(
            task_kind="fullframe",
            view_family="tilted_radial",
            pipeline_extent=PipelineExtent.INFER_ONLY,
            acceptable_results=frozenset({ResultContract.TASK_ARTIFACT}),
            model_io_contract="yolo-seg-raw-v1",
        )
        self.assertTrue(cuda_local_capabilities().supports(tilted_radial))
        self.assertFalse(openvino_local_capabilities().supports(tilted_radial))

    def test_windows_artifact_paths_are_not_uri_schemes(self) -> None:
        ref = ArtifactRef(
            uri=r"C:\data\volume.dat",
            format="volume-u8",
            shape=(1, 2, 3),
            dtype="uint8",
        )
        self.assertEqual(ref.scheme, "path")

    def test_independent_targets_cannot_claim_collective_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one host and one rank"):
            ExecutionTarget(
                target_id="cuda/bad",
                backend_id="cuda",
                semantics="independent",
                host_count=2,
                world_size=2,
            )

    def test_execution_count_cannot_underflow_logical_count(self) -> None:
        with self.assertRaises(ValueError):
            DispatchLease(
                schema_version=1,
                lease_id="lease-1",
                task_id="task-1",
                attempt=0,
                backend_id="backend",
                target_id="target",
                slice_start=0,
                logical_slice_count=9,
                execution_slice_count=8,
                result_contract=ResultContract.TASK_ARTIFACT,
            )

    def test_unknown_backend_fails_closed(self) -> None:
        registry = BackendRegistry()
        with self.assertRaisesRegex(KeyError, "unsupported inference backend"):
            registry.require("tpu")


if __name__ == "__main__":
    unittest.main()
