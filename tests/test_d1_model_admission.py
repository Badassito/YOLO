"""Keep generic GPU models out of the resident TensorRT D1 result contract."""
from __future__ import annotations

import ast
from contextlib import ExitStack
from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection, inference, pipeline
from tests.test_spherical_runtime import _assigned, compiled_views


class _UnconsumedResults:
    def __iter__(self):
        raise AssertionError('A required D1 ring must never fall through to generic results')


class LegacyD1ModelAdmissionTests(unittest.TestCase):
    def test_model_format_gates_the_production_legacy_and_hybrid_d1_branches(self):
        tree = ast.parse(inspect.getsource(pipeline._main_impl))
        hybrid = next(node for node in ast.walk(tree) if _assigned(node, 'hybrid_deferred'))
        routing = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                       and node.test.id == 'radial_owner'
                       and any(_assigned(statement, 'result_mode', 'd1_owner') for statement in node.body))
        program = compile(ast.fix_missing_locations(ast.Module(body=[hybrid, routing], type_ignores=[])), '<result-routing>', 'exec')
        view = compiled_views()[0]
        for path in ('best.pt', 'best.onnx', 'best.engine', 'best.ENGINE', None):
            eligible = pipeline._legacy_d1_model_supported(path)
            self.assertEqual(eligible, path in ('best.engine', 'best.ENGINE'))
            for family in ('orthogonal', 'tilted', 'azimuthal', 'spherical', 'radial'):
                for cpu in (False, True):
                    env = dict(view=replace(view, family=family), kind='fullframe',
                               v1613_d1_owner_active=True, legacy_d1_model_eligible=eligible,
                               worker_direct_union_active=cpu, cpu_eligible=cpu, gpu_eligible=True,
                               azimuthal_parent_requires_seam_union=False, radial_owner=False,
                               gpu_worker_result_dir=Path('unused'), prefix='probe', chunk_idx=0,
                               args=SimpleNamespace(min_conf=0.),
                               HYBRID_DEFERRED_RESULT_MODE=pipeline.HYBRID_DEFERRED_RESULT_MODE)
                    exec(program, env)
                    expected = ('direct_union' if cpu else 'file')
                    if eligible and family in ('orthogonal', 'tilted', 'azimuthal'):
                        expected = pipeline.HYBRID_DEFERRED_RESULT_MODE if cpu else 'd1_owner'
                    with self.subTest(path=path, family=family, cpu=cpu):
                        self.assertEqual(env['result_mode'], expected)
            # The native Radial owner has its own projection/cleanup contract;
            # it remains available even for a generic PyTorch model.
            env.update(view=replace(view, family='radial'), radial_owner=True)
            exec(program, env)
            self.assertEqual(env['result_mode'], 'd1_owner')

    def _predict(self, ring_stats, *, direct=True):
        results = _UnconsumedResults()
        model = SimpleNamespace(predict=mock.Mock(return_value=results))
        accumulator = SimpleNamespace(host_written=False,
            synchronize_for_retirement=mock.Mock(),
            take_device_prediction_stats=mock.Mock(return_value=(0, 0)))
        consumer = mock.Mock(return_value={'d1_view_complete': True})
        with ExitStack() as stack:
            for name in ('ensure_yolo_ready_for_predict', 'validate_yolo_model_input_channels',
                         'ensure_gpu_retina_proto_union_predictor_patch'):
                stack.enter_context(mock.patch.object(inference, name))
            stack.enter_context(mock.patch.object(inference, '_source_prediction_channel_count', return_value=1))
            stack.enter_context(mock.patch.object(inference, 'maybe_wrap_source_with_gpu_input_staging',
                                                   side_effect=lambda source, *args: source))
            stack.enter_context(mock.patch.object(inference, 'cpu_retina_masks_enabled', return_value=False))
            stack.enter_context(mock.patch.object(inference, '_direct_predict_applicable', return_value=direct))
            stack.enter_context(mock.patch.object(inference, '_ensure_predictor_for_direct_predict', return_value=object()))
            stack.enter_context(mock.patch.object(inference, '_direct_predict_stream', return_value=results))
            stack.enter_context(mock.patch.object(inference, '_try_create_device_union_accumulator', return_value=accumulator))
            stack.enter_context(mock.patch.object(inference, 'gpu_retina_flatten_enabled', return_value=False))
            stack.enter_context(mock.patch.object(backprojection, '_try_resident_trt_ring_accumulate', return_value=ring_stats))
            stack.enter_context(mock.patch('builtins.print'))
            stats = inference.predict_source_and_accumulate(
                model, object(), source_label='tilted-d1-regression', num_frames=2, out_size=4,
                cfg=inference.PredictConfig(imgsz=4, conf=.1, device='cuda:0', quantize=None),
                view_union_mm=np.zeros((1, 1, 1), np.uint8), view_confmap_mm=None,
                M_out_to_native=np.eye(2, 3, dtype=np.float32), native_h=4, native_w=4,
                require_device_union=True, require_proto_hole_treatment=True,
                device_union_consumer=consumer,
            )
        return stats, consumer, accumulator

    def test_declined_or_unavailable_ring_fails_before_any_generic_prediction(self):
        for direct in (False, True):
            with self.subTest(direct=direct), self.assertRaisesRegex(RuntimeError, 'D1 requires the resident TensorRT ring'):
                self._predict(None, direct=direct)

    def test_valid_resident_d1_ring_still_publishes_through_its_device_consumer(self):
        stats, consumer, accumulator = self._predict({
            'prediction_count': 3, 'frames_with_predictions': 2, 'proto_hole_treated_frames': 2,
        })
        self.assertEqual(stats['prediction_count'], 3)
        self.assertEqual(stats['frames_with_predictions'], 2)
        self.assertTrue(stats['d1_view_complete'])
        consumer.assert_called_once_with(accumulator)
        accumulator.synchronize_for_retirement.assert_called_once_with(None)
        accumulator.take_device_prediction_stats.assert_not_called()


if __name__ == '__main__':
    unittest.main()
