from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from XTA import geometry, outputs
from XTA.render_batch import RenderBatch


class _CaptureThenSave:
    def __init__(self, sink: outputs.CanonicalRenderImageSink) -> None:
        self.sink = sink
        self.batches: list[RenderBatch] = []

    def __call__(self, batch: RenderBatch) -> None:
        self.batches.append(batch)
        self.sink(batch)


class CanonicalRenderBatchImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.writer_arrays: list[np.ndarray] = []

    def _fake_imwrite(self, path: str, frame: np.ndarray) -> bool:
        encoded = np.ascontiguousarray(frame)
        self.writer_arrays.append(encoded.copy())
        Path(path).write_bytes(encoded.tobytes())
        return True

    def test_streaming_nonzero_aug_saves_the_same_model_bound_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = geometry.ViewInfo('transverse__tta_a37', 2, 3, 4, 'clamp')
            identity = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
            )
            aug_job = geometry.AugJob(
                aug_id='a37',
                angle_deg=37.0,
                meta_path=root / 'a37.json',
                aff=geometry.AffineSpec(
                    view=view.name,
                    angle_deg=37.0,
                    src_w=4,
                    src_h=3,
                    out_size=4,
                    canvas_w=4,
                    canvas_h=3,
                    pad_size=4,
                    pad_off_x=0.0,
                    pad_off_y=0.0,
                    M_out_to_src=identity,
                    M_src_to_out=identity,
                    M_canvas_to_src=identity,
                    M_src_to_canvas=identity,
                ),
            )
            sink = outputs.CanonicalRenderImageSink(
                stage_root=root,
                stem='scan',
                model_name='shared',
                view_name=view.name,
                kind='fullframe',
                aug_id=aug_job.aug_id,
                channel_count=1,
                backend='inprocess_cpu',
            )
            tee = _CaptureThenSave(sink)
            raster_plan = geometry.build_fullframe_raster_plan(view, aug_job)

            def render(center: int) -> np.ndarray:
                return np.arange(12, dtype=np.uint8).reshape(3, 4) + np.uint8(40 * center)

            source = geometry.StreamingYoloVolumeSource(
                render,
                num_frames=2,
                name='nonzero-angle',
                batch_size=2,
                out_size=None,
                render_workers=1,
                prefetch_frames=2,
                autostart=False,
                view=view,
                render_batch_sink=tee,
                raster_plan=raster_plan,
            )
            try:
                with mock.patch.object(
                    outputs.cv2, 'imwrite', side_effect=self._fake_imwrite, create=True,
                ):
                    _paths, model_images, _info = next(source)
            finally:
                source.close()

            self.assertEqual(len(tee.batches), 1)
            batch = tee.batches[0]
            self.assertIs(batch.frames, model_images)
            self.assertIs(batch.items[0].frame, model_images[0])
            self.assertEqual(batch.raster_plan.digest, raster_plan.digest)
            self.assertEqual(batch.items[0].request.plan.digest, raster_plan.digest)
            self.assertEqual(batch.items[0].request.data_role.value, 'intensity')
            saved = sorted(root.rglob('*.png'))
            self.assertEqual(len(saved), 2)
            self.assertIn('aug-a37', saved[0].name)
            self.assertIn('a37', saved[0].as_posix())
            np.testing.assert_array_equal(self.writer_arrays[0], model_images[0][:, :, 0])
            np.testing.assert_array_equal(self.writer_arrays[1], model_images[1][:, :, 0])
            self.assertEqual(saved[0].read_bytes(), model_images[0][:, :, 0].tobytes())
            self.assertEqual(saved[1].read_bytes(), model_images[1][:, :, 0].tobytes())

    def test_materialized_dense_tile_saves_real_centers_and_not_tail_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = geometry.ViewInfo('coronal__tta_am12p5', 2, 2, 3, 'clamp')
            identity = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
            )
            tile_job = geometry.DenseTileJob(
                view=view.name,
                aug_id='am12p5',
                config_id='s256_st128',
                tile_id='s256_st128_am12p5_x0012_y0034',
                tile_x=12,
                tile_y=34,
                tile_size=256,
                tile_stride=128,
                out_size=3,
                meta_path=root / 'tiles.jsonl',
                M_out_to_src=identity,
                M_src_to_out=identity,
            )
            sink = outputs.CanonicalRenderImageSink(
                stage_root=root,
                stem='scan',
                model_name='gpu_model',
                view_name=view.name,
                kind='tile',
                aug_id=tile_job.aug_id,
                config_id=tile_job.config_id,
                tile_id=tile_job.tile_id,
                channel_count=1,
                backend='openvino_cpu',
            )
            tee = _CaptureThenSave(sink)
            raster_plan = geometry.build_dense_tile_raster_plan(view, tile_job)
            materialized = np.asarray(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[11, 12, 13], [14, 15, 16]],
                ],
                dtype=np.uint8,
            )
            source = geometry.InMemoryYoloVolumeSource(
                materialized,
                name='tile-materialized',
                batch_size=4,
                view=view,
                render_batch_sink=tee,
                raster_plan=raster_plan,
            )

            with mock.patch.object(
                outputs.cv2, 'imwrite', side_effect=self._fake_imwrite, create=True,
            ):
                _paths, model_images, _info = next(source)

            self.assertEqual(len(model_images), 4)
            self.assertEqual(len(tee.batches), 1)
            batch = tee.batches[0]
            self.assertIs(batch.frames, model_images)
            self.assertIs(batch.items[1].frame, model_images[1])
            self.assertEqual(batch.raster_plan.digest, raster_plan.digest)
            self.assertIsNotNone(batch.items[1].request)
            self.assertTrue(batch.items[2].synthetic_padding)
            self.assertTrue(batch.items[3].synthetic_padding)
            self.assertIsNone(batch.items[2].request)
            saved = sorted(root.rglob('*.png'))
            self.assertEqual(len(saved), 2)
            self.assertIn('config-s256_st128', saved[0].name)
            self.assertIn('tile-s256_st128_am12p5_x0012_y0034', saved[0].name)
            np.testing.assert_array_equal(self.writer_arrays[0], model_images[0][:, :, 0])
            np.testing.assert_array_equal(self.writer_arrays[1], model_images[1][:, :, 0])
            self.assertEqual(saved[0].read_bytes(), model_images[0][:, :, 0].tobytes())
            self.assertEqual(saved[1].read_bytes(), model_images[1][:, :, 0].tobytes())

    def test_radial_seam_extension_is_model_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = geometry.ViewInfo(
                'radial_transverse__tta_a15', 3, 2, 4, 'pad', family='radial',
                azimuths_deg=(0.0, 60.0, 120.0),
            )
            sink = outputs.CanonicalRenderImageSink(
                stage_root=root,
                stem='scan',
                model_name='shared',
                view_name=view.name,
                kind='fullframe',
                aug_id='a15',
                channel_count=1,
            )
            planes = np.arange(24, dtype=np.uint8).reshape(3, 2, 4)
            source = geometry.InMemoryYoloVolumeSource(
                planes,
                name='radial',
                batch_size=4,
                view=view,
                render_batch_sink=sink,
            )

            with mock.patch.object(
                outputs.cv2, 'imwrite', side_effect=self._fake_imwrite, create=True,
            ):
                _paths, model_images, _info = next(source)

            self.assertEqual(len(model_images), 4)
            self.assertEqual(len(list(root.rglob('*.png'))), 3)


if __name__ == '__main__':
    unittest.main()
