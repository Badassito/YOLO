from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA.geometry import DenseTileJob, PredictionVolumeRef, ViewInfo
from XTA.tta_prediction import (
    PredictionSourceCoordinator,
    PredictionSourceOperations,
    ViewFrameCache,
)


def _view() -> ViewInfo:
    return ViewInfo(
        name="transverse__tta_a0",
        num_slices=2,
        src_h=3,
        src_w=4,
        pad_mode="clamp",
        physical_view_name="transverse",
        tta_aug_id="a0",
    )


class TtaPredictionOwnershipTests(unittest.TestCase):
    def test_view_frame_cache_builds_one_shared_physical_view_array(self) -> None:
        built = np.zeros((2, 3, 4), dtype=np.uint8)
        cache_policy = mock.Mock(return_value=True)
        wait_for_volume = mock.Mock()
        build_cache = mock.Mock(return_value=built)
        source = object()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ViewFrameCache(
                dense_tiling_active=True,
                volume_rgb=source,
                temp_dir=Path(temp_dir),
                augmentation_workers=3,
                cache_policy=cache_policy,
                wait_for_volume=wait_for_volume,
                build_cache=build_cache,
            )
            first = cache.get(_view())
            second = cache.get(_view())

        self.assertIs(first, built)
        self.assertIs(second, built)
        self.assertEqual(build_cache.call_count, 1)
        wait_for_volume.assert_called_once_with(source)
        self.assertEqual(set(cache.arrays), {"transverse"})
        self.assertEqual(set(cache.paths), {"transverse"})

    def test_completed_tile_source_is_given_to_only_the_first_model(self) -> None:
        view = _view()
        tile = DenseTileJob(
            view=view.name,
            aug_id="a0",
            config_id="tile8_stride4",
            tile_id="tile_0000",
            tile_x=0,
            tile_y=0,
            tile_size=8,
            tile_stride=4,
            out_size=8,
            meta_path=Path("tile.json"),
            M_out_to_src=np.eye(2, 3, dtype=np.float32),
            M_src_to_out=np.eye(2, 3, dtype=np.float32),
        )
        prediction_ref = PredictionVolumeRef(
            array=np.zeros((2, 3, 4), dtype=np.uint8),
            path=None,
            name="tile source",
            view_name=view.name,
            job_id=tile.tile_id,
            kind="tile",
            view=view,
        )
        has_gpu_staging = mock.Mock(return_value=False)
        eager_stage = mock.Mock(return_value=prediction_ref)
        pred_cfg = object()
        coordinator = PredictionSourceCoordinator(
            initial_build_jobs=(),
            prediction_volume_executor=mock.Mock(),
            prediction_render_executor=None,
            prediction_volume_queue_slots=1,
            per_prediction_volume_workers=1,
            eager_gpu_input_staging_ahead_sources=1,
            queued_streaming_cpu_warmup_sources=0,
            gpu_worker_process_active=False,
            cpu_worker_process_active=False,
            temp_dir=Path("temp"),
            input_path=Path("input.mkv"),
            canonical_images_stage_root=None,
            volume_rgb=object(),
            channel_format=SimpleNamespace(channel_count=1),
            args=SimpleNamespace(batch=1, gpu_batch=1, cpu_batch=1),
            pred_cfg=pred_cfg,
            yolo_models=(("first", None), ("second", None)),
            keep_temp_artifacts=False,
            get_view_frame_cache=lambda _view: None,
            operations=PredictionSourceOperations(
                prediction_ref_has_gpu_input_staging=has_gpu_staging,
                eager_stage_prediction_ref=eager_stage,
            ),
        )
        completed: Future = Future()
        completed.set_result(prediction_ref)
        coordinator.prediction_volume_futures[completed] = ("tile", view, tile)
        coordinator.pending_prediction_volume_futures.add(completed)

        coordinator.drain_completed_prediction_volume_futures()

        self.assertEqual(len(coordinator.ready_tile_infer), 2)
        first = coordinator.ready_tile_infer.popleft()
        second = coordinator.ready_tile_infer.popleft()
        self.assertEqual(first[:3], ("first", view, tile))
        self.assertIs(first[3], prediction_ref)
        self.assertEqual(second[:3], ("second", view, tile))
        self.assertIsNone(second[3])
        self.assertFalse(coordinator.pending_prediction_volume_futures)
        self.assertFalse(coordinator.prediction_volume_futures)
        eager_stage.assert_called_once_with(prediction_ref, pred_cfg)


if __name__ == "__main__":
    unittest.main()
