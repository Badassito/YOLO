"""CPU-only checks for spherical planning, native task routing, and CUDA dispatch."""
from __future__ import annotations

import ast
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
import inspect
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import config, cuda_backend, cuda_d1, geometry, pipeline, workers
from XTA.spherical_geometry import render_shell_frame
from XTA.unification.runtime import compile_physical_views


def compiled_views(targets=('transverse',)):
    return compile_physical_views(
        t_dim=11, height=13, width=15, cartesian_views=(), azimuthal_requests=(),
        tilted_groups=(), spherical_requests=config.resolve_spherical_view_requests(targets),
        spherical_min_radius=.63, spherical_patch_size=8,
    ).views


def _assigned(node, name, value=None):
    return (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            and (value is None or isinstance(node.value, ast.Constant) and node.value.value == value))


class SphericalSchedulerContracts(unittest.TestCase):
    def test_request_aliases_do_not_multiply_canonical_workload(self):
        single = compiled_views()
        aliases = compiled_views(('transverse', 'sagittal', 'coronal'))
        self.assertEqual(len(single), len(aliases))
        records = pipeline._spherical_workload_groups(aliases)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['group'], 'upright')
        self.assertEqual(records[0]['faces'], list(range(6)))
        self.assertEqual(records[0]['request_tokens'], ['transverse', 'sagittal', 'coronal'])
        self.assertEqual(records[0]['patch_trajectories'], len(aliases))
        self.assertEqual(records[0]['frames_per_angle'], sum(view.num_slices for view in aliases))
        variants = geometry.expand_views_into_tta_variants(aliases, (0., 90.))
        self.assertEqual(sum(view.num_slices for view in variants), 2 * records[0]['frames_per_angle'])
        self.assertEqual(len({view.name for view in variants}), len(variants))

    def test_cube_rotations_receive_separate_workload_records(self):
        views = compiled_views()
        rotated = tuple(replace(view, spherical_group='vertical_p30', spherical_request_tokens=('tilted_transverse',)) for view in views)
        records = pipeline._spherical_workload_groups((*views, *rotated))
        self.assertEqual([record['group'] for record in records], ['upright', 'vertical_p30'])
        self.assertEqual([record['patch_trajectories'] for record in records], [len(views), len(views)])

    def test_early_request_resolution_rejects_nonpositive_patch_size(self):
        # Execute the production early-resolution block without opening models,
        # decoding sources, or initializing the pipeline's worker infrastructure.
        tree = ast.parse(inspect.getsource(pipeline._main_impl))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                     and any(_assigned(statement, 'spherical_requests') for statement in node.body))
        program = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), '<view-request-resolution>', 'exec')
        for size in (0, -8, 8):
            env = {**vars(pipeline), 'parser': SimpleNamespace(error=lambda error: (_ for _ in ()).throw(ValueError(error))),
                   'args': SimpleNamespace(enable_cartesian=None, enable_tilted=None, enable_azimuthal=None,
                                           enable_radial=None, enable_spherical=['transverse'], enable_tile=None, imgsz=size)}
            if size <= 0:
                with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'imgsz > 0'):
                    exec(program, env)
            else:
                exec(program, env)
                self.assertEqual(env['spherical_requests'][0].view, 'transverse')

    def test_production_result_mode_branch_keeps_spherical_native_across_backend_flags(self):
        # Isolate the actual scheduler branch rather than reproducing its
        # eligibility equations in a test-only model of the scheduler.
        tree = ast.parse(inspect.getsource(pipeline._main_impl))
        hybrid = next(node for node in ast.walk(tree) if _assigned(node, 'hybrid_deferred'))
        routing = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                       and node.test.id == 'radial_owner'
                       and any(_assigned(statement, 'result_mode', 'd1_owner') for statement in node.body))
        program = compile(ast.fix_missing_locations(ast.Module(body=[hybrid, routing], type_ignores=[])), '<result-routing>', 'exec')
        view = compiled_views()[0]
        for cpu, d1, direct in product((False, True), repeat=3):
            env = dict(view=view, kind='fullframe', v1613_d1_owner_active=d1,
                       legacy_d1_model_eligible=True,
                       worker_direct_union_active=direct, cpu_eligible=cpu, gpu_eligible=True,
                       azimuthal_parent_requires_seam_union=False, radial_owner=False,
                       gpu_worker_result_dir=Path('unused'), prefix='probe', chunk_idx=0,
                       args=SimpleNamespace(min_conf=0.),
                       HYBRID_DEFERRED_RESULT_MODE=pipeline.HYBRID_DEFERRED_RESULT_MODE)
            exec(program, env)
            with self.subTest(cpu=cpu, d1=d1, direct=direct):
                self.assertEqual(env['result_mode'], 'direct_union' if direct else 'file')
                self.assertFalse(env['hybrid_deferred'])
        for family, cpu, expected in (('orthogonal', False, 'd1_owner'),
                                      ('orthogonal', True, pipeline.HYBRID_DEFERRED_RESULT_MODE),
                                      ('radial', False, 'direct_union')):
            env.update(view=replace(view, family=family), cpu_eligible=cpu,
                       v1613_d1_owner_active=True, worker_direct_union_active=True)
            exec(program, env)
            self.assertEqual(env['result_mode'], expected)
        env.update(radial_owner=True)
        exec(program, env)
        self.assertEqual(env['result_mode'], 'd1_owner')

    def test_d1_and_worker_guards_reject_spherical_before_device_access(self):
        view = compiled_views()[0]
        with self.assertRaisesRegex(ValueError, 'Spherical QSC'):
            cuda_d1._d1_view_family_ids(view)
        with self.assertRaisesRegex(ValueError, 'Spherical QSC'):
            cuda_d1._d1_get_or_create_state({'view': view}, None)
        for worker in (workers.run_prediction_volume_in_worker, workers.run_prediction_volume_in_openvino_worker):
            with self.subTest(worker=worker.__name__), self.assertRaisesRegex(ValueError, 'Spherical QSC'):
                worker(None, None, dict(view=view, job=None, kind='fullframe', result_mode='d1_owner'))

    def test_legacy_fused_and_ring_paths_decline_spherical(self):
        view = compiled_views()[0]
        engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
        self.assertEqual(cuda_backend._fused_preflight_family(view), '')
        self.assertFalse(engine._try_fused_render_into_ring_slot(None, view, None, 0, 8))
        self.assertFalse(engine._try_fused_tilted_into_slot(None, view, None, 0, 8))
        with self.assertRaisesRegex(ValueError, 'spherical'):
            engine._render_cartesian_native_u8_cached(view, 0)
        for cls in (cuda_backend.GpuRenderedYoloSource, cuda_backend.GpuTileRenderedYoloSource):
            with (mock.patch.object(cuda_backend, 'ensure_ultralytics_accepts_in_memory_volume_source'),
                  mock.patch.dict('os.environ', {'YOLO_TTA_NATIVE_TRT_RING': '0'}),
                  mock.patch.dict('sys.modules', {'ultralytics.data.loaders': None}),
                  self.subTest(source=cls.__name__)):
                job = SimpleNamespace(M_out_to_src=np.eye(2, 3, dtype=np.float32))
                source = cls(engine, view, job, slice_offset=0, num_frames=view.num_slices,
                             batch_size=1, out_size=8, fp16=False, name='spherical-test')
                self.assertFalse(source.resident_ring_supported)
                with self.assertRaisesRegex(RuntimeError, 'Spherical QSC'):
                    source.prepare_direct_ring()

    def test_engine_dispatch_calls_spherical_renderer_and_does_not_enter_radial(self):
        engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
        view = compiled_views()[0]
        expected = object()
        with (mock.patch.object(cuda_backend, 'render_spherical_native_resident', return_value=expected) as render,
              mock.patch.object(engine, '_render_radial_native_resident', side_effect=AssertionError('Radial renderer'))):
            self.assertIs(engine._render_native_plane(view, 2), expected)
        render.assert_called_once_with(engine, view, 2)

    def test_renderer_retirement_clears_spherical_cache_only_after_successful_fence(self):
        from tests.test_gpu_asset_retirement import _renderer
        for fail in (False, True):
            engine, _ = _renderer([], fail=fail)
            engine._spherical_direction_cache = {'spherical': object()}
            engine._spherical_direction_cache_bytes = 235_929_600
            if fail:
                with self.assertRaises(Exception):
                    engine.release_inference_assets()
                self.assertEqual(len(engine._spherical_direction_cache), 1)
                self.assertEqual(engine._spherical_direction_cache_bytes, 235_929_600)
            else:
                stats = engine.release_inference_assets()
                self.assertEqual(stats['cache_entries'], 7)
                self.assertFalse(engine._spherical_direction_cache)
                self.assertEqual(engine._spherical_direction_cache_bytes, 0)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Torch is optional')
class SphericalCudaBatchCpuTests(unittest.TestCase):
    @unittest.skipIf(type(geometry.cv2).__name__ == '_StubModule', 'requires real OpenCV numerical runtime')
    def test_generic_fullframe_and_tile_batches_keep_radius_channels_and_angle_affines(self):
        import torch
        import torch.nn.functional as functional
        from tests.test_spherical_cuda import resident_engine
        volume = np.random.default_rng(12).integers(0, 256, (11, 13, 15), dtype=np.uint8)
        engine = resident_engine(volume)
        engine.F, engine._stream = functional, object()
        physical = compiled_views()[0]
        fmt = config.resolve_channel_format('C3S1')
        matrix = np.asarray(((1, 0, 1), (0, 1, 2)), np.float32)
        with (mock.patch.object(torch.cuda, 'stream', return_value=nullcontext()),
              mock.patch.object(torch.cuda, 'Event', return_value=SimpleNamespace(record=lambda stream: None))):
            for angle in (0., 90.):
                view = geometry.expand_views_into_tta_variants((physical,), (angle,))[0]
                job = geometry.build_aug_job_for_variant(view, 8, Path('unused'))
                batch, _ = engine.render_fullframe_batch(view, job, (0,), 8, False, channel_format=fmt)
                for channel, index in enumerate((0, 0, 1)):
                    native = render_shell_frame(volume, view, index)
                    expected = np.rot90(native, int(angle // 90)).copy()
                    np.testing.assert_allclose(batch[0, channel].numpy() * 255., expected, rtol=0, atol=2e-4)
            batch, _ = engine.render_tile_batch(physical, matrix, (0,), out_size=8, fp16=False, channel_format=fmt)
            for channel, index in enumerate((0, 0, 1)):
                native = render_shell_frame(volume, physical, index)
                expected = np.zeros((8, 8), np.uint8)
                expected[:6, :7] = native[2:, 1:]
                np.testing.assert_allclose(batch[0, channel].numpy() * 255., expected, rtol=0, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
