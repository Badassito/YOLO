from __future__ import annotations

import contextlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


def _native_dependency_stubs() -> dict[str, types.ModuleType]:
    """Return the minimal import surface used by these output-control tests."""
    cv2 = types.ModuleType('cv2')
    for name, value in {
        'INTER_AREA': 1,
        'INTER_LINEAR': 2,
        'INTER_NEAREST': 3,
        'COLOR_GRAY2RGB': 4,
        'RETR_EXTERNAL': 5,
        'CHAIN_APPROX_SIMPLE': 6,
        'CV_32S': 7,
    }.items():
        setattr(cv2, name, value)
    scipy = types.ModuleType('scipy')
    ndimage = types.ModuleType('scipy.ndimage')
    scipy.ndimage = ndimage
    tifffile = types.ModuleType('tifffile')
    tqdm_module = types.ModuleType('tqdm')
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    return {
        'cv2': cv2,
        'scipy': scipy,
        'scipy.ndimage': ndimage,
        'tifffile': tifffile,
        'tqdm': tqdm_module,
    }


if 'XTA.outputs' in sys.modules:
    outputs = sys.modules['XTA.outputs']
else:
    # Deliberately use stubs even when native wheels happen to be installed. This keeps
    # the regression suite honest: none of its control-flow assertions depends on cv2,
    # SciPy, tifffile, CUDA, or their platform loaders.
    with mock.patch.dict(sys.modules, _native_dependency_stubs(), clear=False):
        from XTA import outputs


def _serial_indices(total: int, function: object, **_kwargs: object) -> None:
    for index in range(int(total)):
        function(index)  # type: ignore[operator]


@contextlib.contextmanager
def _payload_writer(file_handle: object, **_kwargs: object):
    yield file_handle


class SummaryTests(unittest.TestCase):
    def test_summary_contains_run_results_without_specification_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = outputs.write_summary_file(
                root / 'sample_Summary.txt',
                command='xta --input sample.mkv',
                input_path=Path('sample.mkv'),
                out_dir=root,
                scratch_dir=root / 'temp',
                source_shape_x_y_t=(8, 8, 4),
                volume_shape=(4, 8, 8),
                fps=24.0,
                model_paths=['gpu:model.engine'],
                view_names=['transverse (4 frames)'],
                view_prediction_stats={'transverse': 4},
                interpolation_stats=[],
                enable_3d_void_fill=False,
                gaussian_smoothing_stats=None,
                keep_objects_stats=None,
                voxel_volume=None,
                final_paths={'binary': root / 'sample_binary.mkv'},
                augmentation_workers=1,
                slice_postprocess_workers=1,
                interpolation_workers=1,
                output_workers=1,
            )

            text = summary_path.read_text()
            self.assertIn('View statistics:', text)
            self.assertIn('Final outputs:', text)
            self.assertNotIn('Specification notes:', text)


class AtomicNrrdTests(unittest.TestCase):
    def _nrrd_patches(self) -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(outputs, '_resolve_live_ref_extent', side_effect=lambda ref: ref))
        stack.enter_context(mock.patch.object(
            outputs,
            '_nrrd_raster_plan',
            return_value=types.SimpleNamespace(
                stored_shape_tyx=(1, 1, 1),
                segment_extent_xyt=(0, 0, 0, 0, 0, 0),
            ),
        ))
        stack.enter_context(mock.patch.object(outputs, 'nrrd_slicer_header', return_value={}))
        stack.enter_context(mock.patch.object(outputs, 'slicer_segmentation_header_fields', return_value={}))
        stack.enter_context(mock.patch.object(outputs, '_nrrd_full_slice_z_chunk', return_value=1))
        stack.enter_context(mock.patch.object(outputs, '_nrrd_zshard_capacity', return_value=1))
        stack.enter_context(mock.patch.object(
            outputs,
            '_write_nrrd_ascii_header',
            side_effect=lambda file_handle, **_kwargs: file_handle.write(b'HEADER\n'),
        ))
        stack.enter_context(mock.patch.object(outputs, '_open_nrrd_payload_writer', side_effect=_payload_writer))
        return stack

    def test_single_shard_failure_preserves_previous_nrrd_and_removes_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'layer.seg.nrrd'
            out_path.write_bytes(b'PREVIOUS-COMPLETE-NRRD')

            def fail_after_partial_write(_ref: object, _shape: object, writer: object, **_kwargs: object) -> None:
                writer.write(b'PARTIAL')
                raise RuntimeError('compression failed')

            with self._nrrd_patches(), mock.patch.object(
                outputs,
                '_write_one_decomposed_nrrd_layer_payload',
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(RuntimeError, 'compression failed'):
                    outputs.write_single_layer_nrrd_from_ref(
                        object(),
                        (1, 1, 1),
                        out_path,
                        segment_name='layer',
                        segment_color=(1.0, 0.0, 0.0),
                        z_shards=1,
                    )

            self.assertEqual(out_path.read_bytes(), b'PREVIOUS-COMPLETE-NRRD')
            self.assertEqual(list(Path(tmp).glob('.*.assembling')), [])

    def test_single_shard_success_atomically_replaces_previous_nrrd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / 'layer.seg.nrrd'
            out_path.write_bytes(b'OLD')

            def write_payload(_ref: object, _shape: object, writer: object, **_kwargs: object) -> None:
                writer.write(b'PAYLOAD')

            with self._nrrd_patches(), mock.patch.object(
                outputs,
                '_write_one_decomposed_nrrd_layer_payload',
                side_effect=write_payload,
            ):
                outputs.write_single_layer_nrrd_from_ref(
                    object(),
                    (1, 1, 1),
                    out_path,
                    segment_name='layer',
                    segment_color=(1.0, 0.0, 0.0),
                    z_shards=1,
                )

            self.assertEqual(out_path.read_bytes(), b'HEADER\nPAYLOAD')
            self.assertEqual(list(Path(tmp).glob('.*.assembling')), [])

    def test_manifest_serialization_failure_preserves_previous_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / 'run_nrrd_manifest.json'
            manifest_path.write_text('{"generation":"previous"}')
            with mock.patch.object(outputs.json, 'dumps', side_effect=TypeError('not serializable')):
                with self.assertRaisesRegex(TypeError, 'not serializable'):
                    outputs._write_json_atomically(manifest_path, {'generation': 'new'})
            self.assertEqual(manifest_path.read_text(), '{"generation":"previous"}')
            self.assertEqual(list(Path(tmp).glob('.*.assembling')), [])


class NrrdSinkConstructionTests(unittest.TestCase):
    def test_low_quality_setup_failure_happens_before_executor_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked_root = root / 'not-a-directory'
            blocked_root.write_text('file blocks nested directory creation')
            spec = types.SimpleNamespace(token='half')

            with mock.patch.object(outputs, 'ThreadPoolExecutor') as executor_type:
                with self.assertRaises(OSError):
                    outputs.NrrdLayerSink(
                        nrrd_dir=root / 'nrrd',
                        stem='sample',
                        output_shape_tyx=(1, 1, 1),
                        max_workers=2,
                        low_quality_specs=[spec],
                        low_quality_root=blocked_root,
                    )

            executor_type.assert_not_called()

    def test_missing_low_quality_root_fails_before_executor_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(outputs, 'ThreadPoolExecutor') as executor_type:
                with self.assertRaisesRegex(RuntimeError, 'require a low_quality_root'):
                    outputs.NrrdLayerSink(
                        nrrd_dir=Path(tmp) / 'nrrd',
                        stem='sample',
                        output_shape_tyx=(1, 1, 1),
                        max_workers=2,
                        low_quality_specs=[types.SimpleNamespace(token='half')],
                    )

            executor_type.assert_not_called()


class SequencePublicationTests(unittest.TestCase):
    @unittest.skipIf(sys.platform.startswith('win'), 'POSIX filenames are case-sensitive')
    def test_cleanup_matches_uppercase_pattern_extension_on_posix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pattern = root / 'sample_%04d.TXT'
            stale_tail = root / 'sample_0002.TXT'
            stale_tail.write_text('stale')

            def write_label(_mask: np.ndarray, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('new')

            with mock.patch.object(outputs, 'parallel_for_indices', side_effect=_serial_indices), mock.patch.object(
                outputs, '_write_label_file_from_mask', side_effect=write_label,
            ):
                outputs.write_yolo_labels_from_pattern(
                    np.ones((1, 1, 1), dtype=np.uint8),
                    pattern,
                    show_progress=False,
                )

            self.assertEqual((root / 'sample_0001.TXT').read_text(), 'new')
            self.assertFalse(stale_tail.exists())

    @unittest.skipIf(sys.platform.startswith('win'), 'POSIX filenames are case-sensitive')
    def test_cleanup_preserves_case_distinct_posix_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pattern = root / 'Sample_%04d.txt'
            distinct = root / 'sample_0002.txt'
            distinct.write_text('different-sequence')

            def write_label(_mask: np.ndarray, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('new')

            with mock.patch.object(outputs, 'parallel_for_indices', side_effect=_serial_indices), mock.patch.object(
                outputs, '_write_label_file_from_mask', side_effect=write_label,
            ):
                outputs.write_yolo_labels_from_pattern(
                    np.ones((1, 1, 1), dtype=np.uint8),
                    pattern,
                    show_progress=False,
                )

            self.assertEqual((root / 'Sample_0001.txt').read_text(), 'new')
            self.assertEqual(distinct.read_text(), 'different-sequence')

    def test_label_rerun_removes_stale_tail_but_not_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pattern = root / 'sample_%04d.txt'
            for index in range(1, 5):
                (root / f'sample_{index:04d}.txt').write_text(f'old-{index}')
            unrelated = root / 'another_0009.txt'
            unrelated.write_text('keep')

            def write_label(mask: np.ndarray, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'new-{int(mask.flat[0])}')

            masks = np.array([[[1]], [[2]]], dtype=np.uint8)
            with mock.patch.object(outputs, 'parallel_for_indices', side_effect=_serial_indices), mock.patch.object(
                outputs, '_write_label_file_from_mask', side_effect=write_label,
            ):
                outputs.write_yolo_labels_from_pattern(masks, pattern, show_progress=False)

            self.assertEqual((root / 'sample_0001.txt').read_text(), 'new-1')
            self.assertEqual((root / 'sample_0002.txt').read_text(), 'new-2')
            self.assertFalse((root / 'sample_0003.txt').exists())
            self.assertFalse((root / 'sample_0004.txt').exists())
            self.assertEqual(unrelated.read_text(), 'keep')

    def test_failed_sequence_render_preserves_previous_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pattern = root / 'sample_%04d.txt'
            previous = {
                root / 'sample_0001.txt': 'old-1',
                root / 'sample_0002.txt': 'old-2',
            }
            for path, content in previous.items():
                path.write_text(content)

            def fail_second(mask: np.ndarray, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('staged')
                if int(mask.flat[0]) == 2:
                    raise RuntimeError('frame failed')

            masks = np.array([[[1]], [[2]]], dtype=np.uint8)
            with mock.patch.object(outputs, 'parallel_for_indices', side_effect=_serial_indices), mock.patch.object(
                outputs, '_write_label_file_from_mask', side_effect=fail_second,
            ):
                with self.assertRaisesRegex(RuntimeError, 'frame failed'):
                    outputs.write_yolo_labels_from_pattern(masks, pattern, show_progress=False)

            for path, content in previous.items():
                self.assertEqual(path.read_text(), content)
            self.assertEqual([p for p in root.iterdir() if p.name.startswith('.sample')], [])

    def test_tiff_rerun_removes_old_extension_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pattern = root / 'binary_%04d.tiff'
            (root / 'binary_0001.tif').write_bytes(b'old-extension')
            (root / 'binary_0002.tiff').write_bytes(b'stale-tail')

            def write_tiff(_mask: np.ndarray, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'new-tiff')

            with mock.patch.object(outputs, 'parallel_for_indices', side_effect=_serial_indices), mock.patch.object(
                outputs, '_write_binary_tiff_frame', side_effect=write_tiff,
            ):
                outputs.write_binary_tiff_sequence_from_pattern(
                    np.ones((1, 1, 1), dtype=np.uint8),
                    pattern,
                    show_progress=False,
                )

            self.assertEqual((root / 'binary_0001.tiff').read_bytes(), b'new-tiff')
            self.assertFalse((root / 'binary_0001.tif').exists())
            self.assertFalse((root / 'binary_0002.tiff').exists())

    def test_view_image_rerun_removes_previous_format_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view_dir = root / 'images' / 'axial'
            view_dir.mkdir(parents=True)
            (view_dir / 'sample_axial_0001.tif').write_bytes(b'old-format')
            (view_dir / 'sample_axial_0002.png').write_bytes(b'stale-tail')
            fmt = types.SimpleNamespace(channel_count=1, token='gray')
            view = types.SimpleNamespace(name='axial', num_slices=1)

            class Renderer:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    pass

                def __call__(self, _index: int) -> np.ndarray:
                    return np.zeros((1, 1, 1), dtype=np.uint8)

            def imwrite(path: str, _frame: np.ndarray) -> bool:
                Path(path).write_bytes(b'new-png')
                return True

            with mock.patch.object(outputs, 'resolve_channel_format', return_value=fmt), mock.patch.object(
                outputs, 'ChannelFormattedFrameRenderer', Renderer,
            ), mock.patch.object(outputs, 'parallel_for_indices', side_effect=_serial_indices), mock.patch.object(
                outputs.cv2, 'imwrite', side_effect=imwrite, create=True,
            ):
                outputs.write_view_images(
                    np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    view,
                    root,
                    'sample',
                    channel_format=fmt,
                    show_progress=False,
                )

            self.assertEqual((view_dir / 'sample_axial_0001.png').read_bytes(), b'new-png')
            self.assertFalse((view_dir / 'sample_axial_0001.tif').exists())
            self.assertFalse((view_dir / 'sample_axial_0002.png').exists())


class GrayDownbinCorrectnessTests(unittest.TestCase):
    def test_gray_gpu_path_declines_before_touching_cuda(self) -> None:
        class ForbiddenTorch:
            @property
            def cuda(self) -> object:
                raise AssertionError('gray path must not touch CUDA')

        result = outputs._try_gpu_downbin_volume_on_device(
            np.zeros((1, 2, 3), dtype=np.uint8),
            np.zeros((1, 1, 2), dtype=np.uint8),
            'gray',
            torch=ForbiddenTorch(),
            F=object(),
            device=object(),
        )
        self.assertFalse(result)

    def test_gray_dispatch_declines_before_importing_torch(self) -> None:
        with mock.patch.object(outputs, 'low_quality_gpu_downbin_enabled', return_value=True), mock.patch.dict(
            sys.modules, {'torch': None}, clear=False,
        ):
            self.assertFalse(outputs._try_gpu_downbin_volume(
                np.zeros((1, 2, 3), dtype=np.uint8),
                np.zeros((1, 1, 2), dtype=np.uint8),
                'gray',
            ))


if __name__ == '__main__':
    unittest.main()
