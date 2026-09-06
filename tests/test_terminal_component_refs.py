"""D1 component retirement preserves independently published base and all additions."""
from __future__ import annotations

import ast
import contextlib
from concurrent.futures import Future
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from XTA import assembly, finalization, pipeline
from XTA.geometry import ViewInfo
from XTA.interpolation import NrrdLayerRef
from XTA.projection_queue import settle_prepared_view_components


def _function(source: str, name: str, namespace: dict) -> object:
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, '<terminal-ref-contract>', 'exec'), namespace)
    return namespace[name]


class DenseAllocationSelected(Exception):
    pass


class TerminalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(assembly.__file__).read_text(encoding='utf-8')

    def run_prepare(self, *, physical='transverse', passes=2,
                    interpolate=3, walk=2, candidates=2, malformed=None,
                    stop_on_alloc=False, empty=False, defer_components=False, **overrides):
        volume = np.zeros((4, 5, 6), dtype=np.uint8)
        volume[0, 1, 1] = 1
        initial = volume.copy()
        view = ViewInfo(name=f'{physical}__tta_0', num_slices=4, src_h=5, src_w=6,
                        pad_mode='clamp', physical_view_name=physical, tta_aug_id='0')
        if 'family' in overrides:
            view = ViewInfo(**{**view.__dict__, 'family': overrides.pop('family')})
        if 'angle' in overrides:
            view = ViewInfo(**{**view.__dict__, 'tta_angle_deg': overrides.pop('angle')})
        ns = dict(vars(assembly))
        captures = SimpleNamespace(allocations=[], calls=[], layers=[], closed=[], telemetry={}, component_volumes={}, pending=[])
        telemetry = SimpleNamespace(add=lambda key, value: captures.telemetry.__setitem__(key, captures.telemetry.get(key, 0) + value))

        def allocate(**kw):
            captures.allocations.append(kw)
            if stop_on_alloc:
                raise DenseAllocationSelected()
            return np.zeros(kw['shape'], dtype=kw['dtype'])

        def fake_interpolate(**kw):
            captures.calls.append(kw)
            index = len(captures.calls)
            delta = np.zeros_like(kw['mask_mm'])
            if not empty:
                delta[min(index, 3), 2, 2] = 1
            delta &= (kw['mask_mm'] == 0)
            kw['mask_mm'] |= delta
            stats = {'added_voxels': int(delta.sum()), 'bridge_component_deltas': []}
            if kw['bridge_delta_path'] is not None:
                path = kw['bridge_delta_path']
                path.parent.mkdir(parents=True, exist_ok=True)
                mm = np.memmap(path, shape=delta.shape, dtype=np.uint8, mode='w+')
                mm[:] = delta
                mm.flush()
                del mm
                stats['bridge_delta_path'] = str(path)
            if kw['bridge_component_dir'] is not None:
                for wi in range(1, walk + 1):
                    for ci in range(1, candidates + 1):
                        path = kw['bridge_component_dir'] / f'{wi}_{ci}.cvol'
                        # Deliberate overlap tests union semantics, not summation.
                        component = delta.copy() if wi == 1 else np.zeros_like(delta)
                        captures.component_volumes[str(path)] = component
                        stats['bridge_component_deltas'].append({
                            'path': str(path), 'walk_back_index': wi, 'candidate_index': ci,
                            'added_voxels': int(component.sum()),
                        })
                if malformed == 'duplicate':
                    stats['bridge_component_deltas'][-1] = dict(stats['bridge_component_deltas'][0])
                elif malformed == 'missing':
                    stats['bridge_component_deltas'].pop()
                elif malformed == 'counts':
                    for entry in stats['bridge_component_deltas']:
                        entry['added_voxels'] = 0
            return kw['mask_mm'], stats

        def materialize_component(path, **kw):
            data = captures.component_volumes[str(path)]
            ref = NrrdLayerRef(key=f'component-{len(captures.layers)}', name='component', path=path,
                               shape=data.shape, model_name='model', view_name=view.name,
                               physical_view_name=physical, live_array=data.copy())
            captures.layers.append(ref)
            return ref

        def defer_component(path, **kw):
            future = Future()
            captures.pending.append((future, materialize_component(path, **kw)))
            return future

        ns.update({
            'allocate_workspace_array': allocate,
            'cleanup_view_volume_after_prediction_inplace': lambda *a, **kw: None,
            'close_memmap_array': assembly.close_memmap_array,
            'close_memmap_array_without_flush': captures.closed.append,
            'interpolate_view_volume_pass_maybe_process': fake_interpolate,
            'materialize_interpolation_component_nrrd_view_layer': materialize_component,
            'materialize_nrrd_view_layer': lambda *a, **kw: None,
            'materialize_internal_final_view_layer': lambda *a, **kw: None,
            'runtime_telemetry': lambda: telemetry,
        })
        prepare = _function(self.source, 'prepare_view_volume_after_fullframe', ns)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            finalization, 'union_volume_into_volume', side_effect=lambda dst, src, **kw: np.bitwise_or(dst, src, out=dst),
        ), contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            union_path = root / 'shadow.dat'
            union_path.touch()
            args = dict(model_name='model', view=view, union_mm=volume, confmap_mm=None,
                        union_path=union_path, confmap_path=None, temp_dir=root,
                        dense_tiling_active=False, min_conf=0, min_radius=0,
                        interpolate=interpolate, interpolation_walk_back=walk,
                        interpolation_candidates=candidates, interpolate_passes=passes,
                        interpolate_min_radius=0, interpolation_search_angle=0,
                        keep_temp=False, slice_workers=1, interpolation_task_workers=1,
                        nrrd_layers_enabled=True, precleaned_slice_cleanup=True,
                        hole_fill_done_on_device=True, preinterpolation_layer_already_published=True)
            if defer_components:
                args['submit_component_projection'] = defer_component
            args.update(overrides)
            result = prepare(**args)
            captures.original_path_exists = union_path.exists()
        return result, captures, initial, volume

    def test_all_cartesian_views_finish_without_dense_additions_and_cover_base_plus_multipass(self):
        for physical in ('transverse', 'sagittal', 'coronal'):
            with self.subTest(physical=physical):
                result, seen, initial, mutated = self.run_prepare(physical=physical)
                self.assertIsNone(result.native_support_mm)
                self.assertIsNone(result.final_view_volume_mm)
                self.assertFalse(seen.allocations)
                self.assertEqual(len(seen.calls), 2)
                self.assertTrue(all(call['bridge_delta_path'] is None for call in seen.calls))
                fused = initial.copy()
                for layer in result.nrrd_layers:
                    fused |= layer.live_array
                np.testing.assert_array_equal(fused, mutated)
                self.assertEqual(len(result.nrrd_layers), 8)
                self.assertEqual(seen.telemetry['d1.additions_allocation_avoided_bytes'], initial.nbytes)
                self.assertFalse(seen.original_path_exists)
                self.assertIs(seen.closed[-1], mutated)

    def test_debug_retention_keeps_dense_additions_with_matching_components(self):
        control, seen, initial, mutated = self.run_prepare(keep_temp=True)
        self.assertEqual(len(seen.allocations), 1)
        self.assertTrue(all(call['bridge_delta_path'] is not None for call in seen.calls))
        np.testing.assert_array_equal(initial | control.final_view_volume_mm, mutated)
        treatment, _, _, _ = self.run_prepare()
        self.assertEqual(len(control.nrrd_layers), len(treatment.nrrd_layers))
        for baseline, optimized in zip(control.nrrd_layers, treatment.nrrd_layers):
            np.testing.assert_array_equal(baseline.live_array, optimized.live_array)

    def test_no_interpolation_returns_only_already_published_base_without_drain_copy(self):
        result, seen, _, _ = self.run_prepare(interpolate=0)
        self.assertIsNone(result.final_view_volume_mm)
        self.assertFalse(result.nrrd_layers)
        self.assertFalse(seen.calls)
        self.assertFalse(seen.allocations)

    def test_zero_passes_returns_published_base_and_allocates_nothing(self):
        result, seen, _, _ = self.run_prepare(passes=0)
        self.assertIsNone(result.final_view_volume_mm)
        self.assertFalse(result.nrrd_layers)
        self.assertFalse(seen.allocations)

    def test_active_interpolation_with_no_bridges_preserves_all_empty_component_outputs(self):
        result, seen, initial, mutated = self.run_prepare(empty=True)
        self.assertIsNone(result.final_view_volume_mm)
        self.assertEqual(len(seen.calls), 1)
        self.assertEqual(len(result.nrrd_layers), 4)
        self.assertTrue(all(not np.any(ref.live_array) for ref in result.nrrd_layers))
        np.testing.assert_array_equal(initial, mutated)
        self.assertFalse(seen.allocations)

    def test_unqualified_or_debug_modes_keep_dense_fallback(self):
        for overrides in (
            {'walk': 0}, {'candidates': 0}, {'nrrd_layers_enabled': False},
            {'keep_temp': True}, {'dense_tiling_active': True},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(DenseAllocationSelected):
                self.run_prepare(stop_on_alloc=True, **overrides)

    def test_other_geometries_retire_dense_continuation_when_components_are_complete(self):
        for overrides in ({'family': 'radial'}, {'family': 'tilted'}, {'angle': 45.0}):
            with self.subTest(overrides=overrides):
                result, seen, initial, mutated = self.run_prepare(**overrides)
                self.assertFalse(seen.allocations)
                self.assertIsNone(result.final_view_volume_mm)
                fused = initial.copy()
                for ref in result.nrrd_layers:
                    fused |= ref.live_array
                np.testing.assert_array_equal(fused, mutated)

    def test_parent_retires_shadow_before_detached_component_publication_finishes(self):
        result, seen, initial, mutated = self.run_prepare(family='radial', defer_components=True)
        self.assertFalse(seen.original_path_exists)
        self.assertFalse(seen.allocations)
        self.assertIsNone(result.final_view_volume_mm)
        self.assertEqual(result.nrrd_layers, [])
        self.assertEqual(len(result.pending_component_layers), 8)
        self.assertFalse(settle_prepared_view_components(result))
        for future, ref in reversed(seen.pending):
            future.set_result(ref)
        self.assertTrue(settle_prepared_view_components(result))
        fused = initial.copy()
        for ref in result.nrrd_layers:
            fused |= ref.live_array
        np.testing.assert_array_equal(fused, mutated)
        self.assertEqual(result.nrrd_layers, seen.layers)

    def test_non_d1_prepare_does_not_claim_source_base_was_published(self):
        result, seen, _, mutated = self.run_prepare(preinterpolation_layer_already_published=False)
        self.assertIs(result.final_view_volume_mm, mutated)
        self.assertFalse(seen.telemetry)

    def test_missing_duplicate_or_underreported_component_coverage_fails_before_retirement(self):
        for malformed in ('missing', 'duplicate', 'counts'):
            with self.subTest(malformed=malformed), self.assertRaises(RuntimeError):
                self.run_prepare(malformed=malformed)

    def test_scheduler_detaches_ref_only_group_with_source_base_once(self):
        prepared, _, initial, expected = self.run_prepare()
        view = ViewInfo(name=prepared.view_name, num_slices=4, src_h=5, src_w=6, pad_mode='clamp')
        base = NrrdLayerRef(key='d1-base', name='base', path=Path('base.cvol'), shape=initial.shape,
                           model_name='model', view_name=view.name, live_array=initial)
        namespace = dict(vars(pipeline))
        namespace.update({
            'nrrd_layer_refs': [base, *prepared.nrrd_layers, base],
            'native_view_support_by_model': {'model': {}},
            'view_volumes_by_model': {'model': {}},
            'radial_native_output_by_model': {'model': {}},
            'tilted_native_output_by_model': {'model': {}},
            'd1_layer_ref_by_parent': {('model', view.name): base},
            'close_memmap_array_without_flush': mock.Mock(),
        })
        detach = _function(Path(pipeline.__file__).read_text(encoding='utf-8'), '_detach_terminal_physical_group_inputs', namespace)
        dense, refs = detach('model', [view])
        self.assertEqual(dense, ())
        self.assertEqual(sum(ref.key == 'd1-base' for ref in refs), 1)
        fused = np.zeros_like(initial)
        for ref in refs:
            fused |= ref.live_array
        np.testing.assert_array_equal(fused, expected)
        namespace['close_memmap_array_without_flush'].assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
