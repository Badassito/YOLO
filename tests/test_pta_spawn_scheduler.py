from __future__ import annotations

import dataclasses
from contextlib import nullcontext
import multiprocessing
import importlib.util
import os
import pickle
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


def _module_is_available(name: str) -> bool:
    loaded = sys.modules.get(name)
    if loaded is not None:
        return getattr(loaded, "__spec__", None) is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAS_NUMERICAL_RUNTIME = all(
    _module_is_available(name) for name in ("cv2", "scipy", "tqdm")
)
if not HAS_NUMERICAL_RUNTIME:
    install_stubs()

from XTA import pta
from XTA import pta_scheduler
from XTA import pta_workers


class PtaSpawnSchedulerTests(unittest.TestCase):
    def test_ready_gpu_work_flushes_at_one_full_batch(self) -> None:
        self.assertFalse(
            pta_scheduler.should_flush_ready_gpu_work(
                ready_candidates=63,
                effective_candidate_limit=64,
                producer_drained=False,
            )
        )
        self.assertTrue(
            pta_scheduler.should_flush_ready_gpu_work(
                ready_candidates=64,
                effective_candidate_limit=64,
                producer_drained=False,
            )
        )
        self.assertTrue(
            pta_scheduler.should_flush_ready_gpu_work(
                ready_candidates=1,
                effective_candidate_limit=64,
                producer_drained=True,
            )
        )

    def _install_static(
        self,
        out_dir: Path,
        *,
        augmentation: pta.OfflineAugmentation | None = None,
    ) -> None:
        pta.set_worker_static_context(
            out_dir=out_dir,
            split_active=False,
            image_format="png",
            png_compression=1,
            jpeg_quality=95,
            jpeg_encode_backend="opencv",
            gpu_batch_size=2,
            gpu_render_threads=1,
            gpu_device_ids=(),
            augmentation=augmentation,
            save_images=False,
            save_labels=False,
        )

    def test_auto_and_explicit_process_select_spawn_capable_backend(self) -> None:
        self.assertEqual(pta.resolve_render_backend("auto"), "process")
        self.assertEqual(pta.resolve_render_backend("process"), "process")
        self.assertEqual(pta.resolve_render_backend("thread"), "thread")
        self.assertEqual(multiprocessing.get_context("spawn").get_start_method(), "spawn")

        with tempfile.TemporaryDirectory() as temp_dir:
            self._install_static(Path(temp_dir) / "output")
            pool = pta.PersistentRenderPool(backend="thread", workers=1)
            try:
                self.assertEqual(pool.start_method, "thread")
                self.assertIsNone(pool._pool)
                self.assertIsNotNone(pool._executor)
            finally:
                pool.close()

    def test_spawn_payload_replaces_loaded_cpu_policy_with_definition_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = Path(temp_dir) / "policy.py"
            loaded = pta.LoadedAugmentation(
                path=policy_path,
                content_sha256="a" * 64,
                export_name="build_augmentation",
                albumentations_version="test",
                pipeline_builder=lambda: object(),
            )
            self._install_static(Path(temp_dir) / "output", augmentation=loaded)
            payload = pta._spawn_worker_static_payload()

        definition = payload["augmentation_definition"]
        self.assertIsInstance(definition, dict)
        self.assertEqual(definition["path"], str(policy_path))
        self.assertEqual(definition["content_sha256"], "a" * 64)
        self.assertNotIn("augmentation", payload)
        # This is the contract passed through multiprocessing spawn.  The
        # loaded object above intentionally contains an unpicklable lambda.
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def test_spawn_initializer_revalidates_then_reloads_cpu_policy_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = (Path(temp_dir) / "policy.py").resolve()
            policy_path.write_text("def build_augmentation():\n    return None\n", encoding="utf-8")
            digest = pta.hashlib.sha256(policy_path.read_bytes()).hexdigest()
            inspected = pta.AugmentationDefinition(
                path=policy_path,
                content_sha256=digest,
                export_name="build_augmentation",
            )
            loaded = pta.LoadedAugmentation(
                path=policy_path,
                content_sha256=digest,
                export_name="build_augmentation",
                albumentations_version="test",
                pipeline_builder=lambda: object(),
            )
            payload = {
                "schema": "pta.v18.spawn-worker-static/1",
                "out_dir": str(Path(temp_dir) / "output"),
                "split_active": False,
                "image_format": "png",
                "png_compression": 1,
                "jpeg_quality": 95,
                "jpeg_encode_backend": "opencv",
                "gpu_batch_size": 2,
                "gpu_render_threads": 1,
                "gpu_device_ids": (),
                "augmentation_definition": {
                    "kind": "cpu",
                    "path": str(policy_path),
                    "content_sha256": digest,
                    "export_name": "build_augmentation",
                },
                "save_images": False,
                "save_labels": False,
            }
            with (
                mock.patch.object(
                    pta_workers, "inspect_augmentation_definition", return_value=inspected
                ) as inspect_definition,
                mock.patch.object(
                    pta_workers, "load_augmentation_definition", return_value=loaded
                ) as load_definition,
            ):
                pta._initialize_spawned_worker_static_context(payload)

        inspect_definition.assert_called_once_with(str(policy_path))
        load_definition.assert_called_once_with(str(policy_path))
        self.assertIs(pta._WORKER_STATIC["augmentation"], loaded)
        self.assertEqual(pta._WORKER_STATIC["out_dir"], Path(temp_dir) / "output")

    def test_spawn_initializer_rejects_policy_mutation_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = (Path(temp_dir) / "policy.py").resolve()
            payload = {
                "schema": "pta.v18.spawn-worker-static/1",
                "out_dir": str(Path(temp_dir) / "output"),
                "split_active": False,
                "image_format": "png",
                "png_compression": 1,
                "jpeg_quality": 95,
                "jpeg_encode_backend": "opencv",
                "gpu_batch_size": 2,
                "gpu_render_threads": 1,
                "augmentation_definition": {
                    "kind": "cpu",
                    "path": str(policy_path),
                    "content_sha256": "a" * 64,
                    "export_name": "build_augmentation",
                },
                "save_images": False,
                "save_labels": False,
            }
            changed = pta.AugmentationDefinition(
                path=policy_path,
                content_sha256="b" * 64,
                export_name="build_augmentation",
            )
            with (
                mock.patch.object(
                    pta_workers, "inspect_augmentation_definition", return_value=changed
                ),
                mock.patch.object(
                    pta_workers, "load_augmentation_definition"
                ) as load_definition,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed after parent preflight"):
                    pta._initialize_spawned_worker_static_context(payload)
        load_definition.assert_not_called()

    def test_success_manifest_gate_rejects_policy_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_path = (Path(temp_dir) / "policy.py").resolve()
            policy_path.write_text(
                "def build_augmentation():\n    return None\n",
                encoding="utf-8",
            )
            initial = pta.inspect_augmentation_definition(str(policy_path))
            pta.assert_augmentation_definition_unchanged(initial)

            policy_path.write_text(
                "def build_augmentation():\n    return object()\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "changed during execution"):
                pta.assert_augmentation_definition_unchanged(initial)

    def test_warning_summary_is_byte_stable_across_completion_order(self) -> None:
        payloads = (
            (
                {"render_warning": 8},
                {"render_warning": [f"task-{index:02d}" for index in range(8)]},
            ),
            (
                {"render_warning": 8},
                {"render_warning": [f"task-{index:02d}" for index in range(8, 16)]},
            ),
        )

        def merged_warnings(order: tuple[int, ...]) -> pta.WarningLog:
            warnings = pta.WarningLog()
            for payload_index in order:
                counts, examples = payloads[payload_index]
                pta._merge_worker_warning_payload(warnings, counts, examples)
            return warnings

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "command": "pta deterministic warning test",
                "input_dir": root / "input",
                "out_dir": root / "output",
                "specs": (),
                "volume_records": (),
                "tile_configs": (),
                "channel_formats": (
                    pta.ChannelFormat(
                        token="gray", kind="gray", channel_count=1, stride=1,
                    ),
                ),
                "channel_variants": (),
                "requested_output_format": "png",
                "background_stats": pta.BackgroundFilterStats(),
                "split_stats": pta.SplitStats(),
                "augmentation_stats": pta.AugmentationStats(),
                "dataset_yaml_path": None,
                "workers": 2,
                "frame_workers": 1,
                "planning_workers": 1,
                "topology_summary": "test topology",
                "jpeg_decode_backend": "opencv",
                "jpeg_batch_size": 1,
                "jpeg_encode_backend": "opencv",
                "jpeg_quality": 95,
                "gpu_batch_size": 1,
                "image_format": "png",
                "png_compression": 1,
            }
            first = pta.write_pta_summary(
                root / "first-summary.txt",
                warnings=merged_warnings((0, 1)),
                **common,
            ).read_bytes()
            second = pta.write_pta_summary(
                root / "second-summary.txt",
                warnings=merged_warnings((1, 0)),
                **common,
            ).read_bytes()

        self.assertEqual(first, second)
        serialized = first.decode("utf-8")
        self.assertIn("render_warning: 16", serialized)
        self.assertIn("example: task-00", serialized)
        self.assertIn("example: task-11", serialized)
        self.assertNotIn("example: task-12", serialized)

    @staticmethod
    def _candidate(order: int) -> pta.OutputCandidate:
        return pta.OutputCandidate(
            order=order,
            volume_name="volume",
            parent_view_tag="transverse",
            output_tag=f"item-{order}",
            item_key=f"item-{order}",
            frame_idx=0,
            is_tile=bool(order),
            label_enabled=False,
        )

    def test_gpu_worker_layout_uses_one_cuda_owner_per_device(self) -> None:
        self.assertEqual(
            pta_scheduler.resolve_gpu_worker_layout(
                worker_budget=32,
                requested_frame_workers=0,
                gpu_count=1,
            ),
            (1, 16),
        )
        self.assertEqual(
            pta_scheduler.resolve_gpu_worker_layout(
                worker_budget=32,
                requested_frame_workers=4,
                gpu_count=1,
            ),
            (1, 4),
        )
        self.assertEqual(
            pta_scheduler.resolve_gpu_worker_layout(
                worker_budget=64,
                requested_frame_workers=32,
                gpu_count=1,
            ),
            (1, 32),
        )
        self.assertEqual(
            pta_scheduler.resolve_gpu_worker_layout(
                worker_budget=64,
                requested_frame_workers=0,
                gpu_count=4,
            ),
            (4, 16),
        )

    def test_multi_source_batches_preserve_candidate_order(self) -> None:
        candidates = [self._candidate(index) for index in range(7)]

        def item(start: int, stop: int) -> pta._GpuItemWork:
            return pta._GpuItemWork(
                candidates=tuple(candidates[start:stop]),
                image=np.zeros((8, 8), dtype=np.uint8),
                mask=np.zeros((8, 8), dtype=np.uint8),
                output_size=(8, 8),
                channel_kind="gray",
                context=f"{start}:{stop}",
            )

        batches = list(
            pta_scheduler.iter_compatible_work_batches(
                (item(0, 3), item(3, 6), item(6, 7)),
                candidate_limit=4,
            )
        )
        flattened = [
            candidate.order
            for batch in batches
            for work in batch
            for candidate in work.candidates
        ]
        self.assertEqual([sum(len(work.candidates) for work in batch) for batch in batches], [3, 4])
        self.assertEqual(flattened, list(range(7)))

    def test_projection_phase_summary_reports_family_boundaries(self) -> None:
        views = {
            "cart": types.SimpleNamespace(kind="cart"),
            "radial": types.SimpleNamespace(kind="radial"),
            "tilted": types.SimpleNamespace(kind="tilted"),
            "tilted_radial": types.SimpleNamespace(kind="tilted_radial"),
        }
        plans = tuple(
            types.SimpleNamespace(
                tag=tag,
                view=types.SimpleNamespace(shared_view=view),
            )
            for tag, view in views.items()
        )
        candidates = tuple(
            dataclasses.replace(self._candidate(index), parent_view_tag=tag)
            for index, tag in enumerate(
                ("cart", "cart", "radial", "radial", "radial", "tilted", "tilted_radial")
            )
        )
        with (
            mock.patch.object(
                pta.shared_geometry,
                "is_radial_view",
                side_effect=lambda view: view.kind in {"radial", "tilted_radial"},
            ),
            mock.patch.object(
                pta.shared_geometry,
                "is_tilted_radial_view",
                side_effect=lambda view: view.kind == "tilted_radial",
            ),
            mock.patch.object(
                pta.shared_geometry,
                "is_tilted_view",
                side_effect=lambda view: view.kind in {"tilted", "tilted_radial"},
            ),
        ):
            summary = pta.projection_phase_summary(plans, candidates)

        self.assertEqual(
            summary,
            (
                ("cartesian", 2, 2),
                ("upright-radial", 3, 5),
                ("tilted-cartesian", 1, 6),
                ("tilted-radial", 1, 7),
            ),
        )

    def test_host_and_cuda_sources_are_never_mixed_in_one_policy_batch(self) -> None:
        class DeviceImage:
            is_cuda = True
            shape = (8, 8)

        host = pta._GpuItemWork(
            candidates=(self._candidate(0),),
            image=np.zeros((8, 8), dtype=np.uint8),
            mask=np.zeros((8, 8), dtype=np.uint8),
            output_size=(8, 8),
            channel_kind="gray",
            context="host",
        )
        device = dataclasses.replace(
            host,
            candidates=(self._candidate(1),),
            image=DeviceImage(),
            context="cuda",
        )

        batches = list(
            pta_scheduler.iter_compatible_work_batches(
                (host, device),
                candidate_limit=2,
            )
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual([batch[0].context for batch in batches], ["host", "cuda"])

    def test_original_gpu_batch_is_identity_fast_path_eligible(self) -> None:
        candidates = tuple(self._candidate(index) for index in range(2))
        work = (
            pta._GpuItemWork(
                candidates=(candidates[0],),
                image=np.zeros((8, 8), dtype=np.uint8),
                mask=np.zeros((8, 8), dtype=np.uint8),
                output_size=(8, 8),
                channel_kind="gray",
                context="first",
            ),
            pta._GpuItemWork(
                candidates=(candidates[1],),
                image=np.zeros((8, 8), dtype=np.uint8),
                mask=np.zeros((8, 8), dtype=np.uint8),
                output_size=(8, 8),
                channel_kind="gray",
                context="second",
            ),
        )

        self.assertTrue(pta._gpu_identity_fast_path_eligible(work, ((None,), (None,))))
        self.assertFalse(pta._gpu_identity_fast_path_eligible(work, ((None,), (123,))))
        resized = (work[0], dataclasses.replace(work[1], output_size=(4, 4)))
        self.assertFalse(pta._gpu_identity_fast_path_eligible(resized, ((None,), (None,))))

        malformed_gray = (
            dataclasses.replace(work[0], image=np.zeros((8, 8, 1), dtype=np.uint8)),
        )
        self.assertFalse(pta._gpu_identity_fast_path_eligible(malformed_gray, ((None,),)))
        valid_rgb = dataclasses.replace(
            work[0],
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            channel_kind="rgb",
        )
        invalid_rgb = dataclasses.replace(
            valid_rgb,
            image=np.zeros((8, 8, 4), dtype=np.uint8),
        )
        self.assertTrue(pta._gpu_identity_fast_path_eligible((valid_rgb,), ((None,),)))
        self.assertFalse(pta._gpu_identity_fast_path_eligible((invalid_rgb,), ((None,),)))
        self.assertFalse(
            pta._gpu_identity_fast_path_eligible((work[0], valid_rgb), ((None,), (None,)))
        )
        valid_custom = dataclasses.replace(
            work[0],
            image=np.zeros((8, 8, 5), dtype=np.uint8),
            channel_kind="custom",
            channel_count=5,
        )
        self.assertTrue(
            pta._gpu_identity_fast_path_eligible((valid_custom,), ((None,),))
        )
        self.assertFalse(
            pta._gpu_identity_fast_path_eligible(
                (dataclasses.replace(valid_custom, channel_count=7),),
                ((None,),),
            )
        )

    def test_unlabeled_identity_batch_does_not_materialize_or_upload_masks(self) -> None:
        dtype_token = object()
        from_numpy_shapes: list[tuple[int, ...]] = []
        zero_shapes: list[tuple[int, ...]] = []

        class DeviceTensor:
            def __init__(self, shape: tuple[int, ...]):
                self.shape = shape

            def contiguous(self):
                return self

            def expand(self, *shape: int):
                self.shape = tuple(shape)
                return self

        class HostTensor:
            def __init__(self, shape: tuple[int, ...]):
                self.shape = shape

            def pin_memory(self):
                return self

            def to(self, _device: str, *, non_blocking: bool):
                self.non_blocking = non_blocking
                return DeviceTensor(self.shape)

        class ExplodingMask:
            def __array__(self, *_args, **_kwargs):
                raise AssertionError("unlabeled masks must not be materialized")

        def from_numpy(array: np.ndarray) -> HostTensor:
            from_numpy_shapes.append(tuple(int(x) for x in array.shape))
            return HostTensor(from_numpy_shapes[-1])

        def zeros(shape: tuple[int, ...], *, device: str, dtype: object) -> DeviceTensor:
            self.assertEqual(device, "cuda:0")
            self.assertIs(dtype, fake_torch.uint8)
            zero_shapes.append(tuple(int(x) for x in shape))
            return DeviceTensor(zero_shapes[-1])

        fake_torch = types.SimpleNamespace(
            uint8=dtype_token,
            from_numpy=from_numpy,
            zeros=zeros,
        )
        work = tuple(
            pta._GpuItemWork(
                candidates=(self._candidate(index),),
                image=np.full((8, 8), index, dtype=np.uint8),
                mask=ExplodingMask(),
                output_size=(8, 8),
                channel_kind="gray",
                context=str(index),
            )
            for index in range(2)
        )

        images, masks = pta._apply_gpu_identity_batch_many(
            {"torch": fake_torch, "device_id": 0},
            work,
        )

        self.assertEqual(from_numpy_shapes, [(2, 1, 8, 8)])
        self.assertEqual(zero_shapes, [(1, 1, 1)])
        self.assertEqual(images.shape, (2, 1, 8, 8))
        self.assertEqual(masks.shape, (2, 8, 8))

    def test_cuda_projected_identity_sources_stay_on_device(self) -> None:
        dtype = object()
        stack_shapes: list[tuple[tuple[int, ...], ...]] = []

        class DeviceTensor:
            is_cuda = True

            def __init__(self, shape: tuple[int, ...]):
                self.shape = shape
                self.dtype = dtype
                self.device = "cuda:0"

            def unsqueeze(self, axis: int):
                self.assert_axis = axis
                return DeviceTensor((1,) + self.shape)

            def permute(self, *axes: int):
                return DeviceTensor(tuple(self.shape[index] for index in axes))

            def contiguous(self):
                return self

            def expand(self, *shape: int):
                return DeviceTensor(tuple(shape))

        def stack(values: list[DeviceTensor], *, dim: int) -> DeviceTensor:
            self.assertEqual(dim, 0)
            stack_shapes.append(tuple(value.shape for value in values))
            return DeviceTensor((len(values),) + values[0].shape)

        fake_torch = types.SimpleNamespace(
            uint8=dtype,
            device=lambda value: value,
            stack=stack,
            zeros=lambda shape, **_kwargs: DeviceTensor(tuple(shape)),
        )
        work = tuple(
            pta._GpuItemWork(
                candidates=(self._candidate(index),),
                image=DeviceTensor((8, 8)),
                mask=np.zeros((8, 8), dtype=np.uint8),
                output_size=(8, 8),
                channel_kind="gray",
                context=str(index),
            )
            for index in range(2)
        )

        images, masks = pta._apply_gpu_identity_batch_many(
            {"torch": fake_torch, "device_id": 0},
            work,
        )

        self.assertEqual(stack_shapes, [((1, 8, 8), (1, 8, 8))])
        self.assertEqual(images.shape, (2, 1, 8, 8))
        self.assertEqual(masks.shape, (2, 8, 8))

    def test_augmented_policy_boundary_materializes_cuda_source_once(self) -> None:
        expected = np.arange(64, dtype=np.uint8).reshape(8, 8)

        class DeviceTensor:
            is_cuda = True

            def detach(self):
                return self

            def to(self, device: str):
                self.assert_device = device
                return self

            def numpy(self):
                return expected

        actual = pta._gpu_policy_host_image(DeviceTensor())
        self.assertTrue(bool(actual.flags["C_CONTIGUOUS"]))
        np.testing.assert_array_equal(actual, expected)

    def test_cuda_capable_policy_receives_projected_sources_without_host_copy(self) -> None:
        class DeviceTensor:
            is_cuda = True

            def detach(self):
                raise AssertionError("CUDA-capable policy must not trigger D2H")

        images = (DeviceTensor(), DeviceTensor())
        work = tuple(
            pta._GpuItemWork(
                candidates=(self._candidate(index),),
                image=image,
                mask=np.zeros((8, 8), dtype=np.uint8),
                output_size=(8, 8),
                channel_kind="gray",
                context=str(index),
            )
            for index, image in enumerate(images)
        )
        policy = types.SimpleNamespace(supports_cuda_sources=True)

        selected, zero_copy = pta._gpu_policy_source_images(policy, work)

        self.assertTrue(zero_copy)
        self.assertEqual(selected, images)

    def test_policy_stream_waits_for_async_projection_events(self) -> None:
        waits: list[object] = []
        records: list[object] = []
        consumer = types.SimpleNamespace(wait_event=lambda event: waits.append(event))

        class DeviceTensor:
            is_cuda = True

            def record_stream(self, stream: object):
                records.append(stream)

        first_event = object()
        second_event = object()
        work = tuple(
            pta._GpuItemWork(
                candidates=(self._candidate(index),),
                image=DeviceTensor(),
                mask=np.zeros((8, 8), dtype=np.uint8),
                output_size=(8, 8),
                channel_kind="gray",
                context=str(index),
                ready_event=event,
            )
            for index, event in enumerate((first_event, second_event))
        )
        runtime = {
            "device_id": 0,
            "torch": types.SimpleNamespace(
                cuda=types.SimpleNamespace(current_stream=lambda **_kwargs: consumer),
            ),
        }

        pta._wait_for_gpu_work_ready(runtime, work)

        self.assertEqual(waits, [first_event, second_event])
        self.assertEqual(records, [consumer, consumer])

    def test_unlabeled_gpu_work_does_not_reslice_blank_mask(self) -> None:
        candidate = self._candidate(0)
        plan = types.SimpleNamespace(tile_layout=(), tag="transverse")
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        with (
            mock.patch.object(
                pta_workers,
                "render_channel_formatted_images",
                return_value=(image, None),
            ),
            mock.patch.object(
                pta_workers,
                "render_plan_frame_mask_source",
                side_effect=AssertionError("blank source mask must not be resliced"),
            ),
        ):
            rendered = pta._render_gpu_item_group(
                np.zeros((2, 2, 2), dtype=np.uint8),
                np.zeros((2, 2, 2), dtype=np.uint8),
                plan,
                0,
                (("full", (candidate,)),),
            )

        self.assertEqual(len(rendered), 1)
        np.testing.assert_array_equal(rendered[0].image, image)
        np.testing.assert_array_equal(rendered[0].mask, np.zeros((8, 8), dtype=np.uint8))

    def test_gpu_projected_view_route_bypasses_cpu_intensity_renderer(self) -> None:
        candidate = self._candidate(0)
        plan = types.SimpleNamespace(tile_layout=(), tag="radial_sagittal")
        projected = np.arange(64, dtype=np.uint8).reshape(8, 8)
        with (
            mock.patch.object(
                pta_workers,
                "_gpu_projected_item_image",
                return_value=(projected, None),
            ) as gpu_renderer,
            mock.patch.object(
                pta_workers,
                "render_channel_formatted_images",
                side_effect=AssertionError("CPU intensity projection must be bypassed"),
            ),
            mock.patch.object(
                pta_workers,
                "render_plan_frame_mask_source",
                side_effect=AssertionError("unlabeled masks must not be rendered"),
            ),
        ):
            rendered = pta._render_gpu_item_group(
                np.zeros((2, 2, 2), dtype=np.uint8),
                np.zeros((2, 2, 2), dtype=np.uint8),
                plan,
                0,
                (("full", (candidate,)),),
                {"device_id": 0},
            )

        gpu_renderer.assert_called_once()
        self.assertIs(rendered[0].image, projected)
        self.assertEqual(rendered[0].output_size, (8, 8))
        np.testing.assert_array_equal(rendered[0].mask, np.zeros((8, 8), dtype=np.uint8))

    def test_tilted_cartesian_projection_uses_resident_cuda_renderer(self) -> None:
        dtype_token = object()

        class DeviceTensor:
            is_cuda = True
            shape = (8, 8)
            ndim = 2
            device = "cuda:0"
            dtype = dtype_token

            def round(self):
                return self

            def clamp_(self, *_args: float):
                return self

            def to(self, _dtype: object):
                return self

            def contiguous(self):
                return self

        renderer = types.SimpleNamespace(
            ensure_volume_array=mock.Mock(return_value="resident"),
            render_tilted_grid_resident=mock.Mock(return_value=DeviceTensor()),
            _stream=types.SimpleNamespace(synchronize=mock.Mock()),
        )
        class Event:
            def record(self, stream: object):
                self.stream = stream
        shared_view = object()
        identity = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32,
        )
        plan = types.SimpleNamespace(
            view=types.SimpleNamespace(shared_view=shared_view),
            channel_variant=pta.ChannelVariant("gray", "gray", 1, 1, False, (0,)),
            aff=types.SimpleNamespace(out_h=8, out_w=8, M_out_to_src=identity),
            tile_layout=(),
            source_encoded_indices=(),
        )
        runtime = {
            "torch": types.SimpleNamespace(
                uint8=dtype_token,
                cuda=types.SimpleNamespace(
                    stream=lambda _stream: nullcontext(),
                    Event=Event,
                ),
            ),
            "radial_renderer": renderer,
            "radial_render_lock": threading.Lock(),
            "cuda_projection_disabled_families": set(),
            "radial_texture_required": True,
        }
        with (
            mock.patch.object(pta.shared_geometry, "is_radial_view", return_value=False),
            mock.patch.object(pta.shared_geometry, "is_tilted_view", return_value=True),
            mock.patch.object(pta_workers, "_require_pta_canonical_plan"),
            mock.patch("builtins.print"),
        ):
            actual, ready_event = pta._gpu_projected_item_image(
                runtime,
                np.zeros((2, 2, 2), dtype=np.uint8),
                plan,
                3,
                "full",
            )

        self.assertIsInstance(actual, DeviceTensor)
        self.assertIs(ready_event.stream, renderer._stream)
        renderer.render_tilted_grid_resident.assert_called_once_with(
            shared_view, identity, 3, 8, 8,
        )

    def test_orthogonal_cartesian_projection_uses_resident_cuda_renderer(self) -> None:
        dtype_token = object()

        class DeviceTensor:
            is_cuda = True
            shape = (8, 8)
            ndim = 2
            device = "cuda:0"
            dtype = dtype_token

            def round(self):
                return self

            def clamp_(self, *_args: float):
                return self

            def to(self, _dtype: object):
                return self

            def contiguous(self):
                return self

        class Event:
            def record(self, stream: object):
                self.stream = stream

        projected = DeviceTensor()
        renderer = types.SimpleNamespace(
            ensure_volume_array=mock.Mock(return_value="resident"),
            render_cartesian_grid_resident=mock.Mock(return_value=projected),
            _stream=types.SimpleNamespace(),
        )
        shared_view = object()
        identity = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32,
        )
        plan = types.SimpleNamespace(
            view=types.SimpleNamespace(shared_view=shared_view),
            channel_variant=pta.ChannelVariant("gray", "gray", 1, 1, False, (0,)),
            aff=types.SimpleNamespace(out_h=8, out_w=8, M_out_to_src=identity),
            tile_layout=(),
            source_encoded_indices=(),
        )
        runtime = {
            "torch": types.SimpleNamespace(
                uint8=dtype_token,
                cuda=types.SimpleNamespace(
                    stream=lambda _stream: nullcontext(),
                    Event=Event,
                ),
            ),
            "radial_renderer": renderer,
            "radial_render_lock": threading.Lock(),
            "cuda_projection_disabled_families": set(),
            "radial_texture_required": True,
        }
        with (
            mock.patch.object(pta.shared_geometry, "is_radial_view", return_value=False),
            mock.patch.object(pta.shared_geometry, "is_tilted_view", return_value=False),
            mock.patch.object(
                pta.shared_geometry,
                "physical_view_name",
                return_value="sagittal",
            ),
            mock.patch.object(pta_workers, "_require_pta_canonical_plan"),
            mock.patch("builtins.print"),
        ):
            actual, ready_event = pta._gpu_projected_item_image(
                runtime,
                np.zeros((2, 2, 2), dtype=np.uint8),
                plan,
                3,
                "full",
            )

        self.assertIs(actual, projected)
        self.assertIs(ready_event.stream, renderer._stream)
        renderer.render_cartesian_grid_resident.assert_called_once_with(
            shared_view, identity, 3, 8, 8,
        )

    def test_gpu_oom_split_preserves_order_and_sources(self) -> None:
        candidates = [self._candidate(index) for index in range(6)]
        first = pta._GpuItemWork(
            candidates=tuple(candidates[:4]),
            image=np.zeros((4, 4), dtype=np.uint8),
            mask=np.zeros((4, 4), dtype=np.uint8),
            output_size=(4, 4),
            channel_kind="gray",
            context="first",
        )
        second = pta._GpuItemWork(
            candidates=tuple(candidates[4:]),
            image=np.zeros((4, 4), dtype=np.uint8),
            mask=np.zeros((4, 4), dtype=np.uint8),
            output_size=(4, 4),
            channel_kind="gray",
            context="second",
        )
        left, right = pta_scheduler.split_work_batch((first, second))
        flattened = [
            candidate.order
            for batch in (left, right)
            for work in batch
            for candidate in work.candidates
        ]
        self.assertEqual(flattened, list(range(6)))
        self.assertEqual(sum(len(work.candidates) for work in left), 3)
        self.assertEqual(sum(len(work.candidates) for work in right), 3)

    def test_vram_budget_caps_large_requested_batch(self) -> None:
        fake_cuda = types.SimpleNamespace(mem_get_info=lambda _device: (12 * pta.GIB, 16 * pta.GIB))
        runtime = {"torch": types.SimpleNamespace(cuda=fake_cuda), "device_id": 0}
        work = pta._GpuItemWork(
            candidates=tuple(self._candidate(index) for index in range(144)),
            image=np.zeros((1, 1), dtype=np.uint8),
            mask=np.zeros((1, 1), dtype=np.uint8),
            output_size=(3072, 3072),
            channel_kind="gray",
            context="large",
        )
        effective = pta_scheduler.gpu_memory_candidate_limit(runtime, (work,), requested_limit=144)
        self.assertGreaterEqual(effective, 1)
        self.assertLess(effective, 144)

    def test_multi_source_publisher_rejects_wrong_spatial_shape(self) -> None:
        dtype = object()

        class FakeTensor:
            is_cuda = True

            def __init__(self, shape: tuple[int, ...]):
                self.shape = shape
                self.ndim = len(shape)
                self.dtype = dtype
                self.device = "cuda:0"

        candidate = self._candidate(0)
        runtime = {
            "torch": types.SimpleNamespace(uint8=dtype, device=lambda value: value),
            "device_id": 0,
        }
        with self.assertRaisesRegex(ValueError, "wrong channel/spatial shape"):
            pta._publish_gpu_policy_batch(
                runtime=runtime,
                batch_images=FakeTensor((1, 1, 7, 8)),
                batch_masks=FakeTensor((1, 8, 8)),
                candidates=(candidate,),
                output_size=(8, 8),
                channel_kind="gray",
                local_warnings=pta.WarningLog(),
            )

    def test_multi_source_publisher_rejects_foreign_cuda_device(self) -> None:
        dtype = object()

        class FakeTensor:
            is_cuda = True

            def __init__(self, shape: tuple[int, ...], device: str):
                self.shape = shape
                self.ndim = len(shape)
                self.dtype = dtype
                self.device = device

        runtime = {
            "torch": types.SimpleNamespace(uint8=dtype, device=lambda value: value),
            "device_id": 0,
        }
        with self.assertRaisesRegex(ValueError, "assigned device"):
            pta._publish_gpu_policy_batch(
                runtime=runtime,
                batch_images=FakeTensor((1, 1, 8, 8), "cuda:0"),
                batch_masks=FakeTensor((1, 8, 8), "cuda:1"),
                candidates=(self._candidate(0),),
                output_size=(8, 8),
                channel_kind="gray",
                local_warnings=pta.WarningLog(),
            )

    def test_unlabeled_publisher_skips_mask_reduction(self) -> None:
        dtype = object()

        class FakeTensor:
            is_cuda = True

            def __init__(self, shape: tuple[int, ...]):
                self.shape = shape
                self.ndim = len(shape)
                self.dtype = dtype
                self.device = "cuda:0"

        class NoReduceMask(FakeTensor):
            def reshape(self, *_args):
                raise AssertionError("unlabeled masks must not be reduced")

        runtime = {
            "torch": types.SimpleNamespace(uint8=dtype, device=lambda value: value),
            "device_id": 0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            self._install_static(Path(temp_dir) / "output")
            written, flips = pta._publish_gpu_policy_batch(
                runtime=runtime,
                batch_images=FakeTensor((1, 1, 8, 8)),
                batch_masks=NoReduceMask((1, 8, 8)),
                candidates=(self._candidate(0),),
                output_size=(8, 8),
                channel_kind="gray",
                local_warnings=pta.WarningLog(),
            )

        self.assertEqual(written, 1)
        self.assertEqual(flips, {})

    def test_custom_cuda_batch_writes_multipage_nvtiff_without_host_copy(self) -> None:
        writes: list[tuple[Path, object, int]] = []

        class Sample:
            def contiguous(self):
                return self

        class Selected:
            device = "cuda:0"

            def __getitem__(self, index: int):
                self.last_index = index
                return Sample()

            def detach(self):
                raise AssertionError("nvTIFF path must not copy image pages to the host")

        class Images:
            device = "cuda:0"

            def index_select(self, _axis: int, _indices: object):
                return Selected()

        class NvTiff:
            def write_multipage_lzw(
                self,
                path: Path,
                pages: object,
                *,
                cuda_stream: int,
            ) -> None:
                writes.append((path, pages, cuda_stream))

        fake_stream = types.SimpleNamespace(cuda_stream=1234)
        fake_torch = types.SimpleNamespace(
            as_tensor=lambda values, **_kwargs: tuple(values),
            cuda=types.SimpleNamespace(current_stream=lambda **_kwargs: fake_stream),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._install_static(root)
            pta._WORKER_STATIC["tiff_encode_backend"] = "nvtiff"
            paths = (root / "first.tif", root / "second.tif")
            fallback = pta._write_gpu_image_batch(
                runtime={
                    "torch": fake_torch,
                    "device_id": 0,
                    "nvtiff_encoder": NvTiff(),
                },
                images_nchw=Images(),
                indices=(0, 1),
                paths=paths,
                channel_kind="custom",
                image_format="tif",
                png_compression=1,
                jpeg_quality=95,
            )

        self.assertIsNone(fallback)
        self.assertEqual([entry[0] for entry in writes], list(paths))
        self.assertEqual([entry[2] for entry in writes], [1234, 1234])

    def test_nvjpeg_codestreams_publish_atomically_after_validation(self) -> None:
        payloads = (b"\xff\xd8first\xff\xd9", b"\xff\xd8second\xff\xd9")
        order: list[str] = []

        class Encoder:
            def encode(self, images: object, **_kwargs: object):
                order.append("encode")
                self.images = images
                return list(payloads)

        stream = types.SimpleNamespace(
            cuda_stream=123,
            synchronize=lambda: order.append("stream_sync"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            finals = (root / "a.jpg", root / "b.jpg")
            for path in finals:
                path.write_bytes(b"OLD")
            pta._write_nvjpeg_batch_atomically(
                encoder=Encoder(),
                images=(object(), object()),
                final_paths=finals,
                params=object(),
                cuda_stream=stream,
                synchronize_device=lambda: order.append("device_sync"),
            )

            self.assertEqual(tuple(path.read_bytes() for path in finals), payloads)
            self.assertEqual(list(root.glob(".*.nvjpeg.*.jpg")), [])
        self.assertEqual(order, ["encode", "stream_sync", "device_sync"])

    def test_nvjpeg_empty_codestream_preserves_existing_outputs(self) -> None:
        class Encoder:
            def encode(self, _images: object, **_kwargs: object):
                return [b"\xff\xd8good\xff\xd9", b""]

        stream = types.SimpleNamespace(cuda_stream=1, synchronize=lambda: None)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            finals = (root / "a.jpg", root / "b.jpg")
            for path in finals:
                path.write_bytes(b"OLD")
            with self.assertRaisesRegex(RuntimeError, "empty/truncated JPEG CodeStream"):
                pta._write_nvjpeg_batch_atomically(
                    encoder=Encoder(),
                    images=(object(), object()),
                    final_paths=finals,
                    params=object(),
                    cuda_stream=stream,
                )

            self.assertEqual(tuple(path.read_bytes() for path in finals), (b"OLD", b"OLD"))
            self.assertEqual(list(root.glob(".*.nvjpeg.*.jpg")), [])

    def test_nvjpeg_partial_none_result_publishes_nothing(self) -> None:
        class Encoder:
            def encode(self, _images: object, **_kwargs: object):
                return [b"\xff\xd8good\xff\xd9", None]

        stream = types.SimpleNamespace(cuda_stream=1, synchronize=lambda: None)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            finals = (root / "a.jpg", root / "b.jpg")
            for path in finals:
                path.write_bytes(b"OLD")
            with self.assertRaisesRegex(RuntimeError, "failed JPEG batch index 1"):
                pta._write_nvjpeg_batch_atomically(
                    encoder=Encoder(),
                    images=(object(), object()),
                    final_paths=finals,
                    params=object(),
                    cuda_stream=stream,
                )
            self.assertEqual(tuple(path.read_bytes() for path in finals), (b"OLD", b"OLD"))

    def test_nvjpeg_auto_empty_codestream_disables_encoder_and_falls_back(self) -> None:
        class Tensor:
            device = "cuda:0"

            def __init__(self, array: np.ndarray):
                self.array = np.asarray(array)
                self.shape = self.array.shape
                self.ndim = self.array.ndim

            def index_select(self, _axis: int, _indices: object):
                return self

            def permute(self, *axes: int):
                return Tensor(np.transpose(self.array, axes))

            def contiguous(self):
                return self

            def __getitem__(self, index: int):
                return Tensor(self.array[index])

            def detach(self):
                return self

            def to(self, _device: str):
                return self

            def numpy(self):
                return self.array

        class Encoder:
            def encode(self, images: object, **_kwargs: object):
                return [b"" for _image in images]

        stream = types.SimpleNamespace(cuda_stream=1, synchronize=lambda: None)
        fake_torch = types.SimpleNamespace(
            as_tensor=lambda values, **_kwargs: tuple(values),
            cuda=types.SimpleNamespace(
                current_stream=lambda **_kwargs: stream,
                synchronize=lambda _device: None,
            ),
        )
        fake_nvimgcodec = types.SimpleNamespace(
            QualityType=types.SimpleNamespace(QUALITY=1),
            ColorSpec=types.SimpleNamespace(GRAY=1, SRGB=2),
            ChromaSubsampling=types.SimpleNamespace(CSS_GRAY=1),
            EncodeParams=lambda **kwargs: kwargs,
            as_images=lambda samples, **_kwargs: samples,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._install_static(root)
            pta._WORKER_STATIC["image_format"] = "jpg"
            pta._WORKER_STATIC["jpeg_encode_backend"] = "auto"
            path = root / "fallback.jpg"
            runtime = {
                "torch": fake_torch,
                "device_id": 0,
                "encoder": Encoder(),
                "nvimgcodec": fake_nvimgcodec,
            }
            with mock.patch.object(
                pta_workers,
                "write_image",
                side_effect=lambda output, *_args, **_kwargs: output.write_bytes(
                    b"\xff\xd8fallback\xff\xd9"
                ),
            ):
                note = pta._write_gpu_image_batch(
                    runtime=runtime,
                    images_nchw=Tensor(np.arange(64, dtype=np.uint8).reshape(1, 1, 8, 8)),
                    indices=(0,),
                    paths=(path,),
                    channel_kind="gray",
                    image_format="jpg",
                    png_compression=1,
                    jpeg_quality=95,
                )

            self.assertIsNotNone(note)
            self.assertIsNone(runtime["encoder"])
            self.assertIsNone(runtime["nvimgcodec"])
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(path.read_bytes(), b"\xff\xd8fallback\xff\xd9")

    def test_final_image_tree_rejects_zero_or_missing_publications(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"\xff\xd8a\xff\xd9")
            (images / "b.jpg").write_bytes(b"\xff\xd8b\xff\xd9")
            record = pta.verify_published_image_tree(
                root,
                expected_count=2,
                image_format="jpg",
            )
            self.assertEqual(record["verified_image_count"], 2)
            self.assertGreater(int(record["verified_total_bytes"]), 0)

            (images / "empty.jpg").write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
                pta.verify_published_image_tree(
                    root,
                    expected_count=3,
                    image_format="jpg",
                )
            (images / "empty.jpg").unlink()
            with self.assertRaisesRegex(RuntimeError, "expected=3, verified=2"):
                pta.verify_published_image_tree(
                    root,
                    expected_count=3,
                    image_format="jpg",
                )

    @unittest.skipUnless(
        os.environ.get("XTA_RUN_NVJPEG_INTEGRATION", "0") == "1",
        "set XTA_RUN_NVJPEG_INTEGRATION=1 on an nvImageCodec CUDA host",
    )
    def test_nvimgcodec_09_gray_and_rgb_codestream_publication(self) -> None:
        try:
            import torch  # type: ignore
            from nvidia import nvimgcodec  # type: ignore
        except Exception as exc:  # pragma: no cover - opt-in hardware gate
            self.skipTest(f"CUDA/nvImageCodec unavailable: {exc}")
        if not bool(torch.cuda.is_available()):
            self.skipTest("CUDA unavailable")
        device_id = int(torch.cuda.current_device())
        encoder = nvimgcodec.Encoder(
            device_id=device_id,
            max_num_cpu_threads=2,
            backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)],
            options=":num_cuda_streams=2",
        )
        checker = (
            (torch.arange(32, device=f"cuda:{device_id}")[:, None]
             + torch.arange(32, device=f"cuda:{device_id}")[None, :])
            % 2
        ).to(torch.uint8).mul_(255)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._install_static(root)
            pta._WORKER_STATIC["image_format"] = "jpg"
            pta._WORKER_STATIC["jpeg_encode_backend"] = "nvjpeg"
            runtime = {
                "torch": torch,
                "device_id": device_id,
                "encoder": encoder,
                "nvimgcodec": nvimgcodec,
            }
            gray_paths = (root / "gray-a.jpg", root / "gray-b.jpg")
            gray_batch = torch.stack((checker, 255 - checker), dim=0).unsqueeze(1)
            self.assertIsNone(
                pta._write_gpu_image_batch(
                    runtime=runtime,
                    images_nchw=gray_batch,
                    indices=(0, 1),
                    paths=gray_paths,
                    channel_kind="gray",
                    image_format="jpg",
                    png_compression=1,
                    jpeg_quality=100,
                )
            )
            rgb_path = root / "rgb.jpg"
            rgb_batch = torch.stack(
                (checker, torch.flip(checker, dims=(0,)), 255 - checker), dim=0,
            ).unsqueeze(0)
            self.assertIsNone(
                pta._write_gpu_image_batch(
                    runtime=runtime,
                    images_nchw=rgb_batch,
                    indices=(0,),
                    paths=(rgb_path,),
                    channel_kind="rgb",
                    image_format="jpg",
                    png_compression=1,
                    jpeg_quality=100,
                )
            )
            for path, expected_shape in (
                (gray_paths[0], (32, 32)),
                (gray_paths[1], (32, 32)),
                (rgb_path, (32, 32, 3)),
            ):
                self.assertGreater(path.stat().st_size, 0)
                decoded = pta.cv2.imread(str(path), pta.cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(decoded)
                self.assertEqual(tuple(decoded.shape), expected_shape)
                self.assertGreater(int(np.ptp(decoded)), 0)
            self.assertEqual(list(root.glob(".*.nvjpeg.*.jpg")), [])

    @unittest.skipUnless(
        HAS_NUMERICAL_RUNTIME,
        "requires the production OpenCV/SciPy/tqdm runtime in spawned workers",
    )
    def test_real_tiny_render_task_executes_in_spawned_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._install_static(root / "output")

            volume_block = pta.SharedBlock(8)
            mask_block = pta.SharedBlock(8)
            volume = volume_block.ndarray((2, 2, 2), np.uint8)
            mask = mask_block.ndarray((2, 2, 2), np.uint8)
            volume[...] = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
            mask[...] = 0

            view = pta.ViewInfo(
                name="transverse",
                display_name="Transverse",
                family="transverse",
                num_slices=2,
                src_h=2,
                src_w=2,
                pad_mode="clamp",
                full_t=2,
                full_h=2,
                full_w=2,
            )
            identity = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
            )
            affine = pta.AffineSpec(
                angle_deg=0.0,
                src_w=2,
                src_h=2,
                canvas_w=2,
                canvas_h=2,
                out_w=2,
                out_h=2,
                pad_off_x=0.0,
                pad_off_y=0.0,
                M_src_to_canvas=identity,
                M_canvas_to_src=identity,
                M_src_to_out=identity,
                M_out_to_src=identity,
            )
            plan = pta.build_render_plan(
                view=view,
                aff=affine,
                tag="transverse",
                out_dir=root / "output",
                stem="tiny",
                tile_configs=(),
                save_overlay=False,
                imgsz=2,
                label_enabled=False,
                publish_images=False,
                publish_labels=False,
            )
            candidate = pta.OutputCandidate(
                order=0,
                volume_name="tiny",
                parent_view_tag="transverse",
                output_tag="transverse",
                item_key="full",
                frame_idx=0,
                is_tile=False,
                label_enabled=False,
            )
            task = pta.FrameRenderTask(
                plan_idx=0,
                frame_idx=0,
                items=(("full", (candidate,)),),
            )
            source = pta.SourceVolume(
                input_dir=root,
                stem="tiny",
                kind="sequence",
                image_paths=[],
                video_path=None,
                labels_by_frame={},
                segmentation_nrrd_path=None,
                mask_volume=mask,
                volume_class="unlabeled",
                label_source="none",
                input_start_index=0,
                encoded_indices=(0, 1),
                volume=volume,
                fps=30.0,
            )
            prepared = pta.PreparedVolume(
                src=source,
                source_shape=(2, 2, 2),
                processing_shape=(2, 2, 2),
                effective_volume_class="unlabeled",
                label_enabled=False,
                annotation_states=(pta.ANNOTATION_UNANNOTATED,) * 2,
                save_overlay=False,
                volume_for_render=volume,
                mask_for_render=mask,
                views=[view],
                plans=[plan],
                smoothing_stats=[],
                nrrd_paths=[],
                voxel_initial=None,
                voxel_final=None,
                foreground_preservation_stats={},
                volume_render_block=volume_block,
                mask_render_block=mask_block,
                shm_blocks=[volume_block, mask_block],
            )

            pool: pta.PersistentRenderPool | None = None
            handle: pta.RenderPhaseHandle | None = None
            try:
                pool = pta.PersistentRenderPool(backend="process", workers=1)
                self.assertEqual(pool.start_method, "spawn")
                handle = pool.install_phase(gen=1, prep=prepared, tasks=(task,))
                self.assertEqual(pool.submit_phase(handle, meta=(1, "A")), 1)
                meta, result, error = pool.results.get(timeout=30.0)
                self.assertEqual(meta, (1, "A"))
                if error is not None:
                    raise error
                self.assertIsNotNone(result)
                written, flips, warning_counts, warning_examples = result
                self.assertEqual(written, 1)
                self.assertEqual(flips, {})
                self.assertEqual(warning_counts, {})
                self.assertEqual(warning_examples, {})
            finally:
                if pool is not None:
                    pool.close(terminate=False)
                if handle is not None and handle.payload_block is not None:
                    handle.payload_block.release()
                volume = np.empty((0,), dtype=np.uint8)
                mask = np.empty((0,), dtype=np.uint8)
                prepared.volume_for_render = volume
                prepared.mask_for_render = mask
                source.volume = volume
                source.mask_volume = None
                volume_block.release()
                mask_block.release()

    @unittest.skipIf(pta.gpu_fork_render_backend_available(), "host supports the GPU fork exception")
    def test_gpu_policy_process_path_remains_explicitly_fork_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gpu = pta.LoadedGpuAugmentation(
                path=Path(temp_dir) / "gpu_policy.py",
                content_sha256="b" * 64,
                export_name="build_gpu_augmentation",
                runtime_name="test",
                policy_builder=lambda **_: object(),
            )
            self._install_static(Path(temp_dir) / "output", augmentation=gpu)
            with self.assertRaisesRegex(RuntimeError, "fork-capable host"):
                pta.PersistentRenderPool(
                    backend="process",
                    workers=1,
                    gpu_device_ids=(0,),
                )


if __name__ == "__main__":
    unittest.main()
