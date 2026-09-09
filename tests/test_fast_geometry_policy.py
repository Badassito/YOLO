"""Fast geometry requests cannot silently change strict or categorical plans."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import unittest
from unittest import mock

from XTA import geometry_quality as quality
from XTA.unification.contracts import DataRole
from XTA.unification.sampling import (
    build_forward_raster_plan, forward_sampling_execution_record,
    forward_sampling_policy, raster_plan_from_spawn_spec, raster_plan_spawn_spec,
    require_forward_sampling,
)


_FLAGS = ('YOLO_TTA_FAST_GEOMETRY', 'YOLO_TTA_CPU_SPHERICAL_COMPILED',
          'YOLO_TTA_GPU_RADIAL_COLUMN_GEOMETRY', 'YOLO_TTA_GPU_SPHERICAL_FP32')
_STRICT_DIGEST = 'f6aaea4226a3f2862b89e073034014bf4bad4d8f54a102ffef996f88363e44ec'


class FastGeometryPolicyTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {})
        patch.start()
        self.addCleanup(patch.stop)
        for name in _FLAGS:
            os.environ.pop(name, None)

    @staticmethod
    def plan():
        return build_forward_raster_plan(
            mode='tta', physical_view_id='spherical_fixture', angle_deg=0.,
            channel_token='gray', channel_kind='gray', channel_count=1,
            channel_stride=1, channel_offsets=(0,), channel_direction='ascending',
            output_shape=(32, 32), metadata={'family': 'spherical'},
        )

    def test_default_strict_policy_digest_and_object_identity_survive_mode_changes(self):
        strict = forward_sampling_policy()
        self.assertEqual(strict.digest, _STRICT_DIGEST)
        self.assertEqual(strict.policy_version, 21)
        self.assertFalse(quality.fast_geometry_enabled())
        self.assertFalse(quality.spherical_fp32_requested())
        self.assertFalse(quality.spherical_cpu_compiled_requested())
        self.assertFalse(quality.radial_columns_requested())
        os.environ['YOLO_TTA_FAST_GEOMETRY'] = '1'
        fast = forward_sampling_policy()
        self.assertEqual(fast.policy_version, 22)
        self.assertIs(fast, forward_sampling_policy())
        self.assertNotEqual(strict.digest, fast.digest)
        os.environ['YOLO_TTA_GPU_SPHERICAL_FP32'] = '0'
        self.assertIs(strict, forward_sampling_policy())
        self.assertTrue(quality.spherical_cpu_compiled_requested())
        self.assertTrue(quality.radial_columns_requested())

    def test_local_flags_override_global_without_changing_other_requests(self):
        os.environ['YOLO_TTA_FAST_GEOMETRY'] = 'yes'
        self.assertTrue(quality.spherical_fp32_requested())
        self.assertTrue(quality.spherical_cpu_compiled_requested())
        self.assertTrue(quality.radial_columns_requested())
        for local, request in zip(_FLAGS[1:], (quality.spherical_cpu_compiled_requested,
                                               quality.radial_columns_requested,
                                               quality.spherical_fp32_requested)):
            for value in ('0', 'false', 'OFF', ''):
                os.environ[local] = value
                self.assertFalse(request())
            os.environ.pop(local)
        os.environ['YOLO_TTA_FAST_GEOMETRY'] = '0'
        os.environ['YOLO_TTA_GPU_SPHERICAL_FP32'] = 'true'
        self.assertTrue(quality.spherical_fp32_requested())
        self.assertFalse(quality.spherical_cpu_compiled_requested())
        self.assertFalse(quality.radial_columns_requested())

    def test_shape_guard_keeps_native_and_logical_axes_inside_qualified_range(self):
        self.assertEqual(quality.SPHERICAL_FP32_MAX_AXIS, 4096)
        self.assertTrue(quality.spherical_fp32_shape_eligible((1931, 2911, 3064, 3022)))
        self.assertTrue(quality.spherical_fp32_shape_eligible((1, 4096, 4096, 4096)))
        for shape in ((), (4097, 2, 2), (0, 2, 2), (-1, 2, 2), (4.5, 2, 2), None):
            self.assertFalse(quality.spherical_fp32_shape_eligible(shape))

    def test_fast_backend_tolerance_is_spherical_only_and_categorical_stays_exact(self):
        strict = forward_sampling_policy()
        categorical = require_forward_sampling('cpu', DataRole.CATEGORICAL_GROUND_TRUTH)
        with self.assertRaises(KeyError):
            require_forward_sampling(quality.SPHERICAL_FP32_BACKEND, 'intensity')
        os.environ['YOLO_TTA_FAST_GEOMETRY'] = '1'
        fast = forward_sampling_policy()
        self.assertEqual(require_forward_sampling('cpu', 'categorical_ground_truth'), categorical)
        self.assertTrue(categorical.exact)
        self.assertEqual(categorical.absolute_tolerance, 0.)
        self.assertEqual(strict.role_kernels, fast.role_kernels)
        self.assertEqual(strict.role_boundaries, fast.role_boundaries)
        self.assertEqual(require_forward_sampling('cuda', 'intensity').absolute_tolerance, 1.)
        self.assertEqual(require_forward_sampling(quality.SPHERICAL_FP32_BACKEND, 'intensity').absolute_tolerance, 2.)
        for backend in ('cuda', quality.SPHERICAL_FP32_BACKEND):
            with self.assertRaises(KeyError):
                require_forward_sampling(backend, 'categorical_ground_truth')

    def test_requested_mode_changes_plan_identity_and_spawn_drift_fails_closed(self):
        strict = self.plan()
        os.environ['YOLO_TTA_FAST_GEOMETRY'] = '1'
        fast = self.plan()
        self.assertNotEqual(strict.digest, fast.digest)
        self.assertEqual(strict.output_shape, fast.output_shape)
        self.assertEqual(strict.metadata, fast.metadata)
        spec = pickle.loads(pickle.dumps(raster_plan_spawn_spec(fast)))
        self.assertTrue(spec['geometry_quality_requests']['gpu_spherical_fp32_requested'])
        self.assertFalse(spec['geometry_quality_requests']['native_t_fusion'])
        self.assertEqual(raster_plan_from_spawn_spec(spec).digest, fast.digest)
        os.environ['YOLO_TTA_GPU_SPHERICAL_FP32'] = '0'
        with self.assertRaisesRegex(RuntimeError, 'sampling-policy drift'):
            raster_plan_from_spawn_spec(spec)

    def test_execution_metadata_distinguishes_requested_capability_from_dispatch(self):
        os.environ['YOLO_TTA_FAST_GEOMETRY'] = '1'
        record = forward_sampling_execution_record((('cuda', 'intensity'), ('cpu', 'categorical_ground_truth')))
        regular = next(x for x in record['selected_implementations'] if x['backend'] == 'cuda')
        self.assertEqual(regular['absolute_tolerance'], 1.)
        conditional = record['conditional_implementations'][0]
        self.assertEqual(conditional['family'], 'spherical')
        self.assertEqual(conditional['absolute_tolerance'], 2.)
        self.assertEqual(conditional['fallback_absolute_tolerance'], 1.)
        self.assertTrue(conditional['requested_not_actual_dispatch'])
        self.assertIn('4096', conditional['requires'])
        os.environ['YOLO_TTA_GPU_SPHERICAL_FP32'] = '0'
        record = forward_sampling_execution_record((('cuda', 'intensity'),))
        self.assertNotIn('conditional_implementations', record)
        self.assertEqual(record['policy_digest'], _STRICT_DIGEST)

    def test_fresh_spawn_process_observes_resolved_flags_and_same_policy_digest(self):
        os.environ['YOLO_TTA_FAST_GEOMETRY'] = '1'
        os.environ['YOLO_TTA_CPU_SPHERICAL_COMPILED'] = '0'
        script = (
            'import json; from XTA.geometry_quality import geometry_quality_request_record; '
            'from XTA.unification.sampling import forward_sampling_policy; '
            'print(json.dumps([geometry_quality_request_record(), forward_sampling_policy().digest]))'
        )
        result = subprocess.run([sys.executable, '-B', '-c', script],
            cwd=Path(__file__).resolve().parents[1], env=dict(os.environ),
            capture_output=True, text=True, check=True)
        requests, digest = json.loads(result.stdout)
        self.assertEqual(requests, quality.geometry_quality_request_record())
        self.assertFalse(requests['cpu_spherical_compiled_requested'])
        self.assertTrue(requests['gpu_radial_column_geometry_requested'])
        self.assertTrue(requests['gpu_spherical_fp32_requested'])
        self.assertEqual(digest, forward_sampling_policy().digest)


if __name__ == '__main__':
    unittest.main()
