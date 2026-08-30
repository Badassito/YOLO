from __future__ import annotations

import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA import geometry, workers
from XTA.inference import PredictConfig
from XTA.render_batch import RenderBatch
from XTA.unification.sampling import (
    raster_plan_from_spawn_spec,
    raster_plan_spawn_spec,
)


IDENTITY = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


class _BatchCapture:
    def __init__(self) -> None:
        self.batches: list[RenderBatch] = []

    def __call__(self, batch: RenderBatch) -> None:
        self.batches.append(batch)


class _OpenVinoRunnerProbe:
    def __init__(self) -> None:
        self.source: object | None = None

    def infer_source_to_union(self, source: object, **_kwargs: object) -> dict[str, object]:
        self.source = source
        next(source)  # type: ignore[arg-type]
        return {"prediction_count": 0, "frames_with_predictions": 0}


class WorkerRasterPlanProvenanceTests(unittest.TestCase):
    @staticmethod
    def _view_and_job(root: Path) -> tuple[geometry.ViewInfo, geometry.AugJob]:
        view = geometry.ViewInfo(
            name="transverse__tta_a0",
            num_slices=2,
            src_h=2,
            src_w=2,
            pad_mode="clamp",
            full_t=2,
            full_h=2,
            full_w=2,
            physical_view_name="transverse",
            tta_aug_id="a0",
            tta_angle_deg=0.0,
        )
        job = geometry.AugJob(
            aug_id="a0",
            angle_deg=0.0,
            meta_path=root / "a0.json",
            aff=geometry.AffineSpec(
                view=view.name,
                angle_deg=0.0,
                src_w=2,
                src_h=2,
                out_size=2,
                canvas_w=2,
                canvas_h=2,
                pad_size=2,
                pad_off_x=0.0,
                pad_off_y=0.0,
                M_out_to_src=IDENTITY.copy(),
                M_src_to_out=IDENTITY.copy(),
                M_canvas_to_src=IDENTITY.copy(),
                M_src_to_canvas=IDENTITY.copy(),
            ),
        )
        return view, job

    @classmethod
    def _task(
        cls,
        root: Path,
        *,
        with_image_sink: bool = True,
    ) -> tuple[dict[str, object], object]:
        view, job = cls._view_and_job(root)
        plan = geometry.build_fullframe_raster_plan(view, job)
        task: dict[str, object] = {
            "task_id": 7,
            "kind": "fullframe",
            "view": view,
            "job": job,
            "job_id": job.aug_id,
            "out_size": 2,
            "channel_format": "gray",
            "raster_plan_spec": raster_plan_spawn_spec(plan),
            "canonical_image_spec": ({"probe": True} if with_image_sink else None),
            "M_out_to_src": IDENTITY.copy(),
            "M_out_to_processing": IDENTITY.copy(),
            "processing_shape": (2, 2, 2),
            "threshold_plane_shape": (2, 2),
            "slice_start": 0,
            "slice_count": 2,
            "source_volume_path": root / "source.u8.dat",
            "source_shape": (2, 2, 2),
            "source_dtype": "uint8",
            "result_mask_path": root / "result.mask.u8.dat",
            "result_conf_path": None,
            "result_mode": "file",
            "render_workers": 1,
            "prefetch_frames": 2,
            "postprocess_workers": 1,
            "streaming_cleanup_enabled": False,
            "streaming_cleanup_min_conf": 0.0,
            "streaming_cleanup_min_radius": 0.0,
            "device_hole_fill": False,
        }
        return task, plan

    @staticmethod
    def _renderer(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        def render(index: int) -> np.ndarray:
            return np.full((2, 2), 10 + int(index), dtype=np.uint8)

        return render

    @staticmethod
    def _assert_bound_batch(
        test: unittest.TestCase,
        capture: _BatchCapture,
        expected_plan: object,
    ) -> None:
        test.assertEqual(len(capture.batches), 1)
        batch = capture.batches[0]
        test.assertIsNotNone(batch.raster_plan)
        test.assertEqual(batch.raster_plan.digest, expected_plan.digest)  # type: ignore[attr-defined]
        test.assertTrue(batch.items)
        for item in batch.items:
            test.assertIsNotNone(item.request)
            test.assertEqual(
                item.request.plan.digest, expected_plan.digest  # type: ignore[union-attr,attr-defined]
            )

    def test_spawn_spec_is_primitive_pickle_safe_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task, plan = self._task(Path(temp_dir), with_image_sink=False)
        spec = raster_plan_spawn_spec(plan)
        restored_spec = pickle.loads(pickle.dumps(spec))
        restored = raster_plan_from_spawn_spec(restored_spec)
        self.assertEqual(restored.digest, plan.digest)
        self.assertEqual(restored.canonical_record(), plan.canonical_record())
        restored_task = pickle.loads(pickle.dumps(task))
        task_plan = workers._canonical_raster_plan_for_task(restored_task)
        self.assertIsNotNone(task_plan)
        self.assertEqual(task_plan.digest, plan.digest)

        tampered = copy.deepcopy(spec)
        tampered["plan"]["output_shape"] = [3, 3]
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            raster_plan_from_spawn_spec(tampered)

        policy_drift = copy.deepcopy(spec)
        policy_drift["sampling_policy_digest"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "sampling-policy drift"):
            raster_plan_from_spawn_spec(policy_drift)

    def test_eager_materialized_ref_retains_plan_and_request_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            view, job = self._view_and_job(root)
            plan = geometry.build_fullframe_raster_plan(view, job)
            capture = _BatchCapture()
            ref = geometry.PredictionVolumeRef(
                array=np.arange(8, dtype=np.uint8).reshape(2, 2, 2),
                path=None,
                name="eager-materialized",
                view_name=view.name,
                job_id=job.aug_id,
                kind="fullframe",
                channel_format="gray",  # type: ignore[arg-type]
                view=view,
                render_batch_sink=capture,
                raster_plan=plan,
            )
            cfg = PredictConfig(
                imgsz=2,
                conf=0.25,
                device="cuda:0",
                quantize=None,
                batch=2,
                input_channels=1,
                channel_token="gray",
            )
            with (
                mock.patch.object(
                    geometry,
                    "ensure_ultralytics_accepts_in_memory_volume_source",
                ),
                mock.patch.object(
                    geometry,
                    "maybe_wrap_source_with_gpu_input_staging",
                    side_effect=lambda source, _cfg, _label: source,
                ),
            ):
                staged = geometry.maybe_eager_stage_prediction_ref_on_gpu(ref, cfg)
                self.assertIs(staged, ref)
                self.assertIsNotNone(ref.source)
                next(ref.source)  # type: ignore[arg-type]

        self._assert_bound_batch(self, capture, plan)

    def test_openvino_worker_cpu_source_reconstructs_and_binds_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task, plan = self._task(root)
            cfg = PredictConfig(
                imgsz=2,
                conf=0.25,
                device="cpu",
                quantize=None,
                batch=2,
                input_channels=1,
                channel_token="gray",
            )
            runner = _OpenVinoRunnerProbe()
            capture = _BatchCapture()
            source_volume = np.zeros((2, 2, 2), dtype=np.uint8)
            with (
                mock.patch.object(workers, "cpu_inference_supports_view", return_value=True),
                mock.patch.object(
                    workers, "open_existing_gray_memmap", return_value=source_volume
                ),
                mock.patch.object(
                    workers, "_worker_render_callable", side_effect=self._renderer
                ),
                mock.patch.object(
                    workers,
                    "_canonical_image_sink_for_task",
                    return_value=capture,
                ),
            ):
                workers.run_prediction_volume_in_openvino_worker(runner, cfg, task)

        self.assertIsNotNone(runner.source)
        self.assertEqual(runner.source.raster_plan.digest, plan.digest)  # type: ignore[union-attr]
        self._assert_bound_batch(self, capture, plan)

    def test_cuda_worker_cpu_fallback_reconstructs_and_binds_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task, plan = self._task(root)
            cfg = PredictConfig(
                imgsz=2,
                conf=0.25,
                device="cuda:0",
                quantize=None,
                batch=2,
                input_channels=1,
                channel_token="gray",
            )
            capture = _BatchCapture()
            sources: list[object] = []
            source_volume = np.zeros((2, 2, 2), dtype=np.uint8)

            def predict(_model: object, source: object, **_kwargs: object) -> dict[str, object]:
                sources.append(source)
                next(source)  # type: ignore[arg-type]
                return {"prediction_count": 0, "frames_with_predictions": 0}

            with (
                mock.patch.object(workers, "_worker_gpu_render_engine", return_value=None),
                mock.patch.object(
                    workers, "open_existing_gray_memmap", return_value=source_volume
                ),
                mock.patch.object(
                    workers, "_worker_render_callable", side_effect=self._renderer
                ),
                mock.patch.object(
                    workers,
                    "_canonical_image_sink_for_task",
                    return_value=capture,
                ),
                mock.patch.object(
                    workers, "predict_source_and_accumulate", side_effect=predict
                ),
                mock.patch.object(
                    workers, "gpu_union_flush_overlap_enabled", return_value=False
                ),
            ):
                stats = workers.run_prediction_volume_in_worker(object(), cfg, task)

        self.assertIsInstance(stats, dict)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].raster_plan.digest, plan.digest)  # type: ignore[attr-defined]
        self._assert_bound_batch(self, capture, plan)

    def test_worker_image_sink_without_plan_metadata_fails_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            task, _plan = self._task(Path(temp_dir))
        task.pop("raster_plan_spec")
        with self.assertRaisesRegex(RuntimeError, "requires raster_plan_spec provenance"):
            workers._canonical_raster_plan_for_task(task)

    def test_tile_worker_task_reconstructs_the_exact_tile_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task, _full_plan = self._task(root)
            view = task["view"]
            tile = geometry.DenseTileJob(
                view=view.name,  # type: ignore[union-attr]
                aug_id="a0",
                config_id="s2_st1",
                tile_id="s2_st1_a0_x0000_y0000",
                tile_x=0,
                tile_y=0,
                tile_size=2,
                tile_stride=1,
                out_size=2,
                meta_path=root / "tiles.jsonl",
                M_out_to_src=IDENTITY.copy(),
                M_src_to_out=IDENTITY.copy(),
                parent_crop=(0, 2, 0, 2),
                M_out_to_crop=IDENTITY.copy(),
            )
            plan = geometry.build_dense_tile_raster_plan(view, tile)  # type: ignore[arg-type]
            task.update(
                {
                    "kind": "tile",
                    "job": tile,
                    "job_id": tile.tile_id,
                    "raster_plan_spec": raster_plan_spawn_spec(plan),
                    "parent_crop": tile.parent_crop,
                }
            )
            restored_task = pickle.loads(pickle.dumps(task))
            restored = workers._canonical_raster_plan_for_task(restored_task)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.digest, plan.digest)

            restored_task["job"] = geometry.DenseTileJob(
                **{**tile.__dict__, "tile_x": 1}
            )
            with self.assertRaisesRegex(RuntimeError, "does not match its render task"):
                workers._canonical_raster_plan_for_task(restored_task)


if __name__ == "__main__":
    unittest.main()
