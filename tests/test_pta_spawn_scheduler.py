from __future__ import annotations

import multiprocessing
import importlib.util
import pickle
import sys
import tempfile
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


class PtaSpawnSchedulerTests(unittest.TestCase):
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
                    pta, "inspect_augmentation_definition", return_value=inspected
                ) as inspect_definition,
                mock.patch.object(
                    pta, "load_augmentation_definition", return_value=loaded
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
                    pta, "inspect_augmentation_definition", return_value=changed
                ),
                mock.patch.object(pta, "load_augmentation_definition") as load_definition,
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
            (1, 8),
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
                requested_frame_workers=0,
                gpu_count=4,
            ),
            (4, 8),
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

        candidate = self._candidate(0)
        runtime = {"torch": types.SimpleNamespace(uint8=dtype)}
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
