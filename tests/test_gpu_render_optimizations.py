from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from volume_tta import backprojection, cuda_backend, cuda_d1, inference
from volume_tta.geometry import AffineSpec, AugJob


def make_job(name: str, angle_deg: float, matrix: np.ndarray) -> AugJob:
    affine = AffineSpec(
        view=name,
        angle_deg=float(angle_deg),
        src_w=8,
        src_h=8,
        out_size=8,
        canvas_w=8,
        canvas_h=8,
        pad_size=8,
        pad_off_x=0.0,
        pad_off_y=0.0,
        M_out_to_src=np.asarray(matrix, dtype=np.float32).reshape(2, 3),
        M_src_to_out=np.asarray(matrix, dtype=np.float32).reshape(2, 3),
        M_canvas_to_src=np.asarray(matrix, dtype=np.float32).reshape(2, 3),
        M_src_to_canvas=np.asarray(matrix, dtype=np.float32).reshape(2, 3),
    )
    return AugJob(
        aug_id=f'a{angle_deg:g}',
        angle_deg=float(angle_deg),
        meta_path=Path(f'{name}.json'),
        aff=affine,
    )


class FusedRendererOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        )
        self.rotate_120 = np.asarray(
            ((-0.5, 0.8660254, 3.25), (-0.8660254, -0.5, 7.5)),
            dtype=np.float32,
        )
        self.rotate_240 = np.asarray(
            ((-0.5, -0.8660254, 6.75), (0.8660254, -0.5, 1.5)),
            dtype=np.float32,
        )

    def test_preflight_specs_cover_each_distinct_family_affine(self) -> None:
        views = [
            SimpleNamespace(name='radial_a0', num_slices=9, preflight_family='radial'),
            SimpleNamespace(name='radial_a120', num_slices=9, preflight_family='radial'),
            SimpleNamespace(name='radial_a240', num_slices=9, preflight_family='radial'),
            SimpleNamespace(name='radial_a120_duplicate', num_slices=9, preflight_family='radial'),
            SimpleNamespace(name='tilted_a120', num_slices=7, preflight_family='tilted'),
        ]
        jobs = {
            'radial_a0': [make_job('radial_a0', 0.0, self.identity)],
            'radial_a120': [make_job('radial_a120', 120.0, self.rotate_120)],
            'radial_a240': [make_job('radial_a240', 240.0, self.rotate_240)],
            'radial_a120_duplicate': [
                make_job('radial_a120_duplicate', 120.0, self.rotate_120),
            ],
            'tilted_a120': [make_job('tilted_a120', 120.0, self.rotate_120)],
        }

        with mock.patch.object(
            cuda_backend,
            '_fused_preflight_family',
            side_effect=lambda view: view.preflight_family,
        ):
            specs = cuda_backend.build_fused_renderer_preflight_specs(views, jobs)

        self.assertEqual(
            [(spec['view'].preflight_family, spec['job'].angle_deg) for spec in specs],
            [
                ('radial', 0.0),
                ('radial', 120.0),
                ('radial', 240.0),
                ('tilted', 120.0),
            ],
        )
        self.assertEqual([spec['frame_index'] for spec in specs], [4, 4, 4, 3])

    def test_startup_preflight_executes_every_distinct_affine_spec(self) -> None:
        class FakeView:
            def __init__(self, name: str) -> None:
                self.name = name
                self.preflight_family = 'radial'

        views = [FakeView('radial_a0'), FakeView('radial_a120')]
        jobs = [
            make_job('radial_a0', 0.0, self.identity),
            make_job('radial_a120', 120.0, self.rotate_120),
        ]
        specs = [
            {'view': view, 'job': job, 'frame_index': 3}
            for view, job in zip(views, jobs)
        ]
        engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
        engine._mode = 'resident'
        engine._volume_key = ('volume', (4, 4, 4), 1)
        engine._fused_preflight_volume_key = None
        engine.device = object()
        engine.torch = SimpleNamespace(
            float16=object(),
            float32=object(),
            int32=object(),
            empty=mock.Mock(return_value=object()),
        )
        engine.validate_fused_ring_renderer = mock.Mock()

        with (
            mock.patch.object(cuda_backend, 'ViewInfo', FakeView),
            mock.patch.object(
                cuda_backend,
                '_fused_preflight_family',
                side_effect=lambda view: view.preflight_family,
            ),
            mock.patch.object(
                cuda_backend,
                'fused_renderer_preflight_enabled',
                return_value=True,
            ),
        ):
            engine.run_startup_fused_preflight(specs, out_size=8, fp16=True)

        self.assertEqual(engine.validate_fused_ring_renderer.call_count, 2)
        self.assertTrue(all(
            call.kwargs == {'compare_reference': True}
            for call in engine.validate_fused_ring_renderer.call_args_list
        ))
        self.assertEqual(engine._fused_preflight_volume_key, engine._volume_key)

    def test_nonzero_angle_renders_directly_into_single_channel_ring(self) -> None:
        job = make_job('radial_a120', 120.0, self.rotate_120)
        view = SimpleNamespace(name='radial_a120')
        render_done = mock.Mock()
        slot = SimpleNamespace(
            input=SimpleNamespace(shape=(1, 1, 8, 8)),
            infer_valid=False,
            render_done=render_done,
        )
        engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
        engine.torch = SimpleNamespace(
            cuda=SimpleNamespace(stream=lambda _stream: nullcontext()),
        )
        engine._stream = object()
        engine._try_fused_render_into_ring_slot = mock.Mock(return_value=True)
        engine._render_fullframe_frame = mock.Mock(
            side_effect=AssertionError('nonzero fused render fell back to Torch'),
        )

        engine.render_fullframe_into_ring_slot(
            slot,
            view,
            job,
            frame_index=3,
            out_size=8,
        )

        engine._try_fused_render_into_ring_slot.assert_called_once_with(
            slot,
            view,
            job.aff,
            3,
            8,
        )
        render_done.record.assert_called_once_with(engine._stream)

    def test_nonzero_angle_direct_ring_validates_and_captures_graphs(self) -> None:
        fp16 = object()
        fp32 = object()
        slots = [
            SimpleNamespace(
                input=SimpleNamespace(dtype=fp16, shape=(1, 1, 8, 8)),
                render_graph=None,
                render_graph_key=None,
                render_expected_key=None,
            )
            for _ in range(2)
        ]
        engine = SimpleNamespace(
            torch=SimpleNamespace(float16=fp16, float32=fp32),
            _mode='resident',
            _fused_renderer_key=mock.Mock(
                side_effect=lambda slot, *_args: ('affine', id(slot)),
            ),
            validate_fused_ring_renderer=mock.Mock(),
            capture_fused_ring_renderer=mock.Mock(),
            render_fullframe_into_ring_slot=mock.Mock(
                side_effect=AssertionError('nonzero fused ring used generic setup'),
            ),
        )
        source = object.__new__(cuda_backend.GpuRenderedYoloSource)
        source.engine = engine
        source.view = SimpleNamespace(name='radial_a120', family='radial')
        source.job = make_job('radial_a120', 120.0, self.rotate_120)
        source.slice_offset = 0
        source.bs = 1
        source.nf = 4
        source.channel_count = 1
        source.out_size = 8
        source.fp16 = True
        source._direct_ring = slots

        prepared = source.prepare_direct_ring(input_dtype=fp16)

        self.assertIs(prepared, slots)
        engine.validate_fused_ring_renderer.assert_called_once()
        self.assertEqual(engine.capture_fused_ring_renderer.call_count, 2)
        self.assertTrue(all(slot.render_expected_key is not None for slot in slots))

    def test_direct_radial_and_tilted_pixel_kernels_use_2d_coordinates(self) -> None:
        kernel_source = inspect.getsource(cuda_backend._fused_direct_render_kernels)
        self.assertNotIn('int oy = q / ow', kernel_source)
        self.assertIn(
            'int ox = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;',
            kernel_source,
        )
        launch_source = inspect.getsource(
            cuda_backend._GpuWorkerRenderEngine._try_fused_radial_into_slot,
        )
        self.assertIn('render_block = (32, 8)', launch_source)
        self.assertIn('render_grid, render_block', launch_source)


class ProtoUnionOptimizationTests(unittest.TestCase):
    def test_tiled_fp16_union_eligibility_is_narrow_and_alignment_safe(self) -> None:
        fp16 = object()
        fp32 = object()
        torch_mod = SimpleNamespace(float16=fp16)
        head = SimpleNamespace(dtype=fp16)

        self.assertTrue(inference._tiled_f16_proto_union_applicable(
            torch_mod,
            head,
            SimpleNamespace(dtype=fp16, shape=(1, 32, 768, 768)),
        ))
        self.assertFalse(inference._tiled_f16_proto_union_applicable(
            torch_mod,
            head,
            SimpleNamespace(dtype=fp16, shape=(1, 32, 768, 767)),
        ))
        self.assertFalse(inference._tiled_f16_proto_union_applicable(
            torch_mod,
            head,
            SimpleNamespace(dtype=fp16, shape=(1, 16, 768, 768)),
        ))
        self.assertFalse(inference._tiled_f16_proto_union_applicable(
            torch_mod,
            head,
            SimpleNamespace(dtype=fp32, shape=(1, 32, 768, 768)),
        ))

    def test_resident_fp16_32_mask_post_uses_packed_tiled_union(self) -> None:
        class FakeKernel:
            def __init__(self) -> None:
                self.calls = []

            def __call__(self, grid, block, args, *, stream) -> None:
                self.calls.append((grid, block, args, stream))

        dtype_f16 = object()
        compact_tiled = FakeKernel()
        union_tiled = FakeKernel()
        compact_fallback = FakeKernel()
        union_fallback = FakeKernel()
        quantize = FakeKernel()
        executor = object.__new__(backprojection._ResidentTensorRTRingExecutor)
        executor.torch = SimpleNamespace(float16=dtype_f16)
        executor.kernels = SimpleNamespace(
            cp=object(),
            compact_f16_tiled=compact_tiled,
            union_f16_f16_tiled=union_tiled,
            compact_f16=compact_fallback,
            union_f16_f16=union_fallback,
            upsample_quantize=quantize,
        )
        executor.confidence_threshold = 0.25
        executor.out_size = 65
        executor.native_h = 65
        executor.native_w = 65
        executor.proto_hole_treatment_active = False
        executor.collect_slice_bboxes = False
        head = SimpleNamespace(shape=(1, 37, 4), dtype=dtype_f16)
        proto = SimpleNamespace(shape=(1, 32, 5, 66), dtype=dtype_f16)
        compact_count = mock.Mock()
        refs = {
            'head': object(),
            'proto': object(),
            'indices': object(),
            'count': object(),
            'compact_coeff': object(),
            'compact_proto_boxes': object(),
            'compact_confs': object(),
            'max_logit': object(),
            'native_union': object(),
            'native_bbox': object(),
        }
        slot = SimpleNamespace(
            slot_id=0,
            head=head,
            proto=proto,
            compact_count=compact_count,
            post_stream=object(),
            unit_descriptor=object(),
            identity_native_warp=True,
            _cupy_refs=refs,
        )

        with mock.patch.object(
            backprojection, '_cupy_external_stream', return_value='external-stream',
        ):
            executor._launch_post(slot)

        compact_count.zero_.assert_called_once_with()
        self.assertEqual(len(compact_tiled.calls), 1)
        self.assertEqual(compact_tiled.calls[0][0:2], ((1,), (256,)))
        self.assertEqual(compact_tiled.calls[0][2][-3:], (
            refs['compact_coeff'], refs['compact_proto_boxes'], refs['compact_confs'],
        ))
        self.assertEqual(len(union_tiled.calls), 1)
        self.assertEqual(union_tiled.calls[0][0:2], ((2, 2), (32, 4)))
        self.assertEqual(union_tiled.calls[0][2][0:5], (
            refs['proto'], refs['compact_coeff'], refs['compact_proto_boxes'],
            refs['compact_confs'], refs['count'],
        ))
        self.assertEqual(compact_fallback.calls, [])
        self.assertEqual(union_fallback.calls, [])

    def test_resident_kernel_bundle_contains_packed_tiled_sources(self) -> None:
        source = inspect.getsource(inference._resident_mask_kernels)
        self.assertIn('compact_f16_tiled', source)
        self.assertIn('union_f16_f16_tiled', source)
        self.assertIn('half2 packed = *reinterpret_cast<const half2*>', source)


class D1BackprojectionOptimizationTests(unittest.TestCase):
    def test_quantizer_metadata_readback_normalizes_empty_slice_without_mask_scan(self) -> None:
        class FakeDeviceArray:
            def __init__(self, values: np.ndarray) -> None:
                self.values = values

            def cpu(self) -> 'FakeDeviceArray':
                return self

            def numpy(self) -> np.ndarray:
                return self.values

        synchronize = mock.Mock()
        accumulator = object.__new__(inference._DeviceUnionAccumulator)
        accumulator.torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=synchronize),
        )
        accumulator.device = object()
        # Deliberately expose only shape: the D1 metadata path must not reduce/read the mask.
        accumulator.union_dev = SimpleNamespace(shape=(3, 10, 20))
        accumulator.host_written = False
        accumulator.written = np.ones((3,), dtype=bool)
        accumulator.slice_bboxes_written = np.ones((3,), dtype=bool)
        accumulator.slice_bboxes_dev = FakeDeviceArray(np.asarray(
            (
                (10, 0, 20, 0),
                (2, 7, 3, 11),
                (0, 10, 0, 20),
            ),
            dtype=np.int32,
        ))

        metadata = accumulator.compute_d1_slice_metadata(synchronize_device=False)

        self.assertIsInstance(metadata, dict)
        np.testing.assert_array_equal(
            metadata['slice_any'], np.asarray((False, True, True), dtype=bool),
        )
        np.testing.assert_array_equal(
            metadata['slice_bboxes'],
            np.asarray(((0, 0, 0, 0), (2, 7, 3, 11), (0, 10, 0, 20))),
        )
        synchronize.assert_not_called()

    def test_quantizer_metadata_rejects_incomplete_or_malformed_bboxes(self) -> None:
        class FakeDeviceArray:
            def cpu(self) -> 'FakeDeviceArray':
                return self

            def numpy(self) -> np.ndarray:
                return np.asarray(((0, 0, 0, 0),), dtype=np.int32)

        accumulator = object.__new__(inference._DeviceUnionAccumulator)
        accumulator.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=mock.Mock()))
        accumulator.device = object()
        accumulator.union_dev = SimpleNamespace(shape=(1, 10, 20))
        accumulator.host_written = False
        accumulator.written = np.ones((1,), dtype=bool)
        accumulator.slice_bboxes_written = np.ones((1,), dtype=bool)
        accumulator.slice_bboxes_dev = FakeDeviceArray()

        self.assertIsNone(
            accumulator.compute_d1_slice_metadata(synchronize_device=False),
        )
        accumulator.slice_bboxes_written[0] = False
        self.assertIsNone(
            accumulator.compute_d1_slice_metadata(synchronize_device=False),
        )

    def test_resident_quantizer_uses_2d_launch_and_initializes_bbox(self) -> None:
        class FakeKernel:
            def __init__(self) -> None:
                self.calls = []

            def __call__(self, grid, block, args, *, stream) -> None:
                self.calls.append((grid, block, args, stream))

        dtype_f16 = object()
        compact = FakeKernel()
        union = FakeKernel()
        quantize = FakeKernel()
        executor = object.__new__(backprojection._ResidentTensorRTRingExecutor)
        executor.torch = SimpleNamespace(float16=dtype_f16)
        executor.kernels = SimpleNamespace(
            cp=object(),
            compact_f16=compact,
            union_f16_f16=union,
            upsample_quantize=quantize,
        )
        executor.confidence_threshold = 0.25
        executor.out_size = 65
        executor.native_h = 65
        executor.native_w = 65
        executor.proto_hole_treatment_active = False
        executor.collect_slice_bboxes = True
        head = SimpleNamespace(shape=(1, 6, 4), dtype=dtype_f16)
        proto = SimpleNamespace(shape=(1, 1, 2, 2), dtype=dtype_f16)
        compact_count = mock.Mock()
        slot = SimpleNamespace(
            slot_id=0,
            head=head,
            proto=proto,
            compact_count=compact_count,
            post_stream=object(),
            unit_descriptor=object(),
            identity_native_warp=True,
            _cupy_refs={
                'head': object(),
                'proto': object(),
                'indices': object(),
                'count': object(),
                'max_logit': object(),
                'native_union': object(),
                'native_bbox': object(),
            },
        )

        with mock.patch.object(
            backprojection, '_cupy_external_stream', return_value='external-stream',
        ):
            executor._launch_post(slot)

        compact_count.zero_.assert_called_once_with()
        self.assertIs(compact.calls[0][2][-1], slot._cupy_refs['native_bbox'])
        self.assertEqual(quantize.calls[0][0:2], ((3, 9), (32, 8)))
        self.assertIs(quantize.calls[0][2][-1], slot._cupy_refs['native_bbox'])

    def test_d1_kernel_warp_aggregates_output_word_atomics(self) -> None:
        source = inspect.getsource(cuda_d1._d1_backproject_kernels)
        self.assertIn('__match_any_sync', source)
        self.assertIn('d1_warp_aggregated_atomic_or(output_bits, word, bit, warp_bits)', source)

    def test_bbox_launch_plan_batches_similar_slices_and_buckets_large_ones(self) -> None:
        slice_any = np.asarray((True, True, True, False), dtype=bool)
        slice_bboxes = np.asarray(
            (
                (0, 1, 0, 100),
                (0, 1, 0, 200),
                (0, 1, 0, 300),
                (0, 0, 0, 0),
            ),
            dtype=np.int64,
        )

        specs, groups, scanned = cuda_d1._d1_prepare_bbox_launch_plan(
            slice_any,
            slice_bboxes,
            (4, 2, 400),
        )

        np.testing.assert_array_equal(
            specs,
            np.asarray(
                (
                    (0, 0, 0, 1, 100),
                    (1, 0, 0, 1, 200),
                    (2, 0, 0, 1, 300),
                ),
                dtype=np.int32,
            ),
        )
        self.assertEqual(groups, [(1, 0, 2), (2, 2, 1)])
        self.assertEqual(scanned, 600)

    def test_bbox_launch_plan_rejects_nonempty_slice_with_empty_clamped_bbox(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'nonempty.*empty bbox'):
            cuda_d1._d1_prepare_bbox_launch_plan(
                np.asarray((True,), dtype=bool),
                np.asarray(((5, 8, 2, 2),), dtype=np.int64),
                (1, 4, 4),
            )


if __name__ == '__main__':
    unittest.main()
