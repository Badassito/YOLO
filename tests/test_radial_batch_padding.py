from __future__ import annotations

import unittest
from unittest import mock
import types

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from XTA import cuda_backend, geometry, inference


class RadialBatchPaddingTests(unittest.TestCase):
    @staticmethod
    def _radial_view(num_slices: int = 3) -> geometry.ViewInfo:
        return geometry.ViewInfo(
            name='radial_transverse',
            num_slices=int(num_slices),
            src_h=2,
            src_w=4,
            pad_mode='pad',
            family='radial',
            azimuths_deg=tuple(
                float(index * 180.0 / num_slices) for index in range(num_slices)
            ),
        )

    @staticmethod
    def _planes(num_slices: int = 3) -> np.ndarray:
        return np.stack([
            np.asarray(
                [
                    [10 * index + 0, 10 * index + 1, 10 * index + 2, 10 * index + 3],
                    [10 * index + 4, 10 * index + 5, 10 * index + 6, 10 * index + 7],
                ],
                dtype=np.uint8,
            )
            for index in range(int(num_slices))
        ])

    def test_cartesian_partial_batch_still_repeats_last_frame_and_discards_specs(self) -> None:
        view = geometry.ViewInfo('transverse', 3, 2, 4, 'clamp')
        planes = self._planes()
        source = geometry.InMemoryYoloVolumeSource(
            planes,
            name='cartesian',
            batch_size=10,
            view=view,
        )

        _paths, images, info = next(source)

        self.assertEqual(source.synthetic_count, 7)
        self.assertEqual(source.radial_padding_count, 0)
        self.assertEqual(len(images), 10)
        for result_index in range(3, 10):
            self.assertIsNone(source.result_frame_spec(result_index))
            np.testing.assert_array_equal(images[result_index][:, :, 0], planes[2])
            self.assertIn('repeats real slice 3/3', info[result_index])

    def test_one_wrap_radial_source_mirrors_input_and_inverse_accumulates_to_slice_zero(self) -> None:
        view = self._radial_view()
        planes = self._planes()
        source = geometry.InMemoryYoloVolumeSource(
            planes,
            name='radial',
            batch_size=4,
            view=view,
        )
        _paths, images, _info = next(source)
        spec = source.result_frame_spec(3)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(spec.is_radial_padding)
        self.assertTrue(spec.mirror_radial_u)
        self.assertEqual(spec.global_destination_index, 0)
        np.testing.assert_array_equal(images[3][:, :, 0], planes[0][:, ::-1])

        destination = np.zeros((3, 4, 4), dtype=np.uint8)
        identity = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
        )
        target, _conf, target_index, affine, count_stats = (
            inference._prediction_accumulation_target(
                spec,
                view_union_mm=destination,
                view_confmap_mm=None,
                radial_padding_union_mm=None,
                radial_padding_confmap_mm=None,
                M_out_to_native=identity,
                native_w=4,
            )
        )
        self.assertIs(target, destination)
        self.assertEqual(target_index, 0)
        self.assertFalse(count_stats)
        np.testing.assert_array_equal(
            affine,
            np.asarray([[-1.0, 0.0, 3.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )

        def fake_warp(
            plane: np.ndarray, matrix: np.ndarray, *, dsize: tuple[int, int], **_kwargs: object,
        ) -> np.ndarray:
            self.assertEqual(tuple(dsize), (4, 4))
            return (
                np.asarray(plane)[:, ::-1].copy()
                if float(np.asarray(matrix)[0, 0]) < 0.0 else np.asarray(plane).copy()
            )

        mask = np.zeros((1, 4, 4), dtype=np.uint8)
        mask[0, 0, 0] = np.uint8(1)
        with mock.patch.object(inference.cv2, 'warpAffine', side_effect=fake_warp):
            inference._process_prediction_frame(
                idx=target_index,
                masks_np=mask,
                confs_np=None,
                out_size=4,
                view_union_mm=target,
                view_confmap_mm=None,
                M_out_to_native=affine,
                native_h=4,
                native_w=4,
            )
        self.assertEqual(int(destination[0, 0, 3]), 1)
        self.assertEqual(int(np.count_nonzero(destination[1:])), 0)

    def test_padding_only_prediction_uses_aux_sink_without_corrupting_logical_stats(self) -> None:
        view = self._radial_view()
        source = geometry.InMemoryYoloVolumeSource(
            self._planes(),
            name='padding-only',
            batch_size=4,
            view=view,
        )
        empty_result = type('Result', (), {'masks': None})()
        padding_mask = np.zeros((1, 4, 4), dtype=np.uint8)
        padding_mask[0, 1, 0] = np.uint8(1)
        padding_result = type('Result', (), {
            'masks': type('Masks', (), {'data': padding_mask})(),
            'boxes': None,
        })()

        class FakeModel:
            def predict(self, **_kwargs: object):
                return iter((empty_result, empty_result, empty_result, padding_result))

        destination = np.zeros((3, 4, 4), dtype=np.uint8)
        padding_destination = np.zeros((1, 4, 4), dtype=np.uint8)
        identity = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
        )

        def fake_warp(
            plane: np.ndarray, matrix: np.ndarray, *, dsize: tuple[int, int], **_kwargs: object,
        ) -> np.ndarray:
            self.assertEqual(tuple(dsize), (4, 4))
            return (
                np.asarray(plane)[:, ::-1].copy()
                if float(np.asarray(matrix)[0, 0]) < 0.0 else np.asarray(plane).copy()
            )

        with (
            mock.patch.object(inference, 'ensure_yolo_ready_for_predict'),
            mock.patch.object(inference, 'validate_yolo_model_input_channels'),
            mock.patch.object(inference, 'require_channel_aware_yolo_preprocess_patch'),
            mock.patch.object(inference, 'cpu_retina_masks_enabled', return_value=False),
            mock.patch.object(inference, 'ensure_gpu_retina_proto_union_predictor_patch'),
            mock.patch.object(inference, '_direct_predict_applicable', return_value=False),
            mock.patch.object(inference, '_try_create_device_union_accumulator', return_value=None),
            mock.patch.object(inference, 'gpu_retina_flatten_enabled', return_value=False),
            mock.patch.object(inference, 'prediction_hot_path_flush_enabled', return_value=False),
            mock.patch.object(inference.cv2, 'warpAffine', side_effect=fake_warp),
        ):
            stats = inference.predict_source_and_accumulate(
                FakeModel(),
                source,
                source_label='padding-only',
                num_frames=3,
                out_size=4,
                cfg=inference.PredictConfig(
                    imgsz=4,
                    conf=0.1,
                    device='cpu',
                    quantize=None,
                    batch=4,
                    input_channels=1,
                    channel_token='gray',
                ),
                view_union_mm=destination,
                view_confmap_mm=None,
                M_out_to_native=identity,
                native_h=4,
                native_w=4,
                postprocess_workers=2,
                radial_padding_union_mm=padding_destination,
                radial_padding_confmap_mm=None,
            )

        self.assertEqual(stats['prediction_count'], 0)
        self.assertEqual(stats['frames_with_predictions'], 0)
        self.assertEqual(stats['radial_padding_processed'], 1)
        self.assertEqual(int(np.count_nonzero(destination)), 0)
        self.assertEqual(int(padding_destination[0, 1, 3]), 1)

    def test_multiwrap_batch_preserves_exact_destinations_and_crossing_parity(self) -> None:
        view = self._radial_view()
        specs = geometry.radial_batch_padding_frame_specs(
            view,
            num_frames=3,
            batch_size=10,
        )

        self.assertEqual(len(specs), 7)
        self.assertEqual(
            [spec.global_destination_index for spec in specs],
            [0, 1, 2, 0, 1, 2, 0],
        )
        self.assertEqual(
            [spec.mirror_radial_u for spec in specs],
            [True, True, True, False, False, False, True],
        )
        self.assertEqual(
            geometry.radial_batch_padding_mirror_groups(view, 3, 10),
            (True, False),
        )

        # The second crossing is unoriented like the original view: it must retain the
        # base output affine instead of being inverse-mirrored again.
        even_spec = specs[3]
        identity = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
        )
        _target, _conf, target_index, affine, count_stats = (
            inference._prediction_accumulation_target(
                even_spec,
                view_union_mm=np.zeros((3, 2, 4), dtype=np.uint8),
                view_confmap_mm=None,
                radial_padding_union_mm=None,
                radial_padding_confmap_mm=None,
                M_out_to_native=identity,
                native_w=4,
            )
        )
        self.assertEqual(target_index, 0)
        self.assertFalse(count_stats)
        np.testing.assert_array_equal(affine, identity)

    def test_multiwrap_in_memory_compatibility_frames_follow_parity(self) -> None:
        view = self._radial_view()
        planes = self._planes()
        source = geometry.InMemoryYoloVolumeSource(
            planes,
            name='radial-multiwrap',
            batch_size=10,
            view=view,
        )

        _paths, images, _info = next(source)

        np.testing.assert_array_equal(images[3][:, :, 0], planes[0][:, ::-1])
        np.testing.assert_array_equal(images[4][:, :, 0], planes[1][:, ::-1])
        np.testing.assert_array_equal(images[5][:, :, 0], planes[2][:, ::-1])
        np.testing.assert_array_equal(images[6][:, :, 0], planes[0])
        np.testing.assert_array_equal(images[7][:, :, 0], planes[1])
        np.testing.assert_array_equal(images[8][:, :, 0], planes[2])
        np.testing.assert_array_equal(images[9][:, :, 0], planes[0][:, ::-1])

    def test_streaming_source_renders_every_wrapped_center_instead_of_repeating_tail(self) -> None:
        view = self._radial_view()
        planes = self._planes()
        requested_centers: list[int] = []

        def render(center: int) -> np.ndarray:
            requested_centers.append(int(center))
            source_index, mirror_u = geometry.channel_view_slice_source(view, int(center))
            plane = planes[int(source_index)]
            return np.ascontiguousarray(plane[:, ::-1] if mirror_u else plane)

        source = geometry.StreamingYoloVolumeSource(
            render,
            num_frames=3,
            name='streaming-radial-multiwrap',
            batch_size=10,
            out_size=None,
            render_workers=1,
            prefetch_frames=10,
            autostart=False,
            view=view,
        )
        try:
            _paths, images, _info = next(source)
        finally:
            source.close()

        self.assertEqual(requested_centers, list(range(10)))
        np.testing.assert_array_equal(images[3][:, :, 0], planes[0][:, ::-1])
        np.testing.assert_array_equal(images[6][:, :, 0], planes[0])
        np.testing.assert_array_equal(images[9][:, :, 0], planes[0][:, ::-1])

    def test_resident_fullframe_and_tile_sources_forward_logical_multiwrap_indices(self) -> None:
        view = self._radial_view()

        class FakeEngine:
            def __init__(self) -> None:
                self.fullframe_indices: list[int] = []
                self.tile_indices: list[int] = []

            def render_fullframe_batch(
                self, _view: object, _job: object, indices: list[int],
                _out_size: int, _fp16: bool, **_kwargs: object,
            ) -> tuple[object, None]:
                self.fullframe_indices.extend(int(value) for value in indices)
                return object(), None

            def render_tile_batch(
                self, _view: object, _affine: object, indices: list[int],
                **_kwargs: object,
            ) -> tuple[object, None]:
                self.tile_indices.extend(int(value) for value in indices)
                return object(), None

            def clear_native_plane_cache(self) -> None:
                return None

        engine = FakeEngine()
        tile_job = types.SimpleNamespace(
            M_out_to_src=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32,
            )
        )
        with mock.patch.object(
            cuda_backend, 'ensure_ultralytics_accepts_in_memory_volume_source',
        ):
            full_source = cuda_backend.GpuRenderedYoloSource(
                engine,
                view,
                types.SimpleNamespace(),
                slice_offset=0,
                num_frames=3,
                batch_size=10,
                out_size=4,
                fp16=False,
                name='gpu-full-radial-multiwrap',
            )
            tile_source = cuda_backend.GpuTileRenderedYoloSource(
                engine,
                view,
                tile_job,
                slice_offset=0,
                num_frames=3,
                batch_size=10,
                out_size=4,
                fp16=False,
                name='gpu-tile-radial-multiwrap',
            )

        next(full_source)
        next(tile_source)

        self.assertEqual(engine.fullframe_indices, list(range(10)))
        self.assertEqual(engine.tile_indices, list(range(10)))
        self.assertEqual(
            [
                full_source.result_frame_spec(index).mirror_radial_u
                for index in range(3, 10)
            ],
            [True, True, True, False, False, False, True],
        )
        self.assertEqual(
            [
                tile_source.result_frame_spec(index).global_destination_index
                for index in range(3, 10)
            ],
            [0, 1, 2, 0, 1, 2, 0],
        )

    def test_tile_contract_keeps_parity_crops_and_empty_completion_ids_separate(self) -> None:
        view = self._radial_view()
        result_ids = geometry.radial_batch_padding_tile_result_ids(
            'tile-7', view, 3, 10,
        )

        self.assertEqual(
            result_ids,
            (
                'tile-7',
                'tile-7__radial_batch_seam_mirrored',
                'tile-7__radial_batch_seam_unmirrored',
            ),
        )
        # Completion IDs are geometry-derived, not foreground-derived; both auxiliary IDs
        # remain owed even when either parity group produces an empty mask.
        self.assertEqual(len(set(result_ids)), 3)
        original_crop = (1, 5, 2, 6)
        self.assertEqual(
            geometry.mirrored_radial_parent_crop(original_crop, parent_width=10),
            (1, 5, 4, 8),
        )
        self.assertEqual(original_crop, (1, 5, 2, 6))


if __name__ == '__main__':
    unittest.main()
