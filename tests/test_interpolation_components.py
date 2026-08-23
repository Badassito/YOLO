from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from volume_tta import assembly, interpolation, outputs, topology


class InterpolationComponentDecompositionTests(unittest.TestCase):
    @staticmethod
    def _render_plan() -> interpolation.SliceBridgeRenderPlan:
        sdf = np.ones((3, 3), dtype=np.float32)
        return interpolation.SliceBridgeRenderPlan(
            source_label=1,
            target_label=2,
            source_point=(1, 2, 2),
            target_point=(3, 2, 2),
            source_anchor=(2, 2),
            target_anchor=(2, 2),
            steps=2,
            sign=1,
            num_slices=5,
            sdf0=sdf,
            sdf1=sdf,
        )

    @staticmethod
    def _render_plan_at(
        *,
        source_z: int,
        target_z: int,
        center: tuple[int, int],
        walk_back_index: int,
        candidate_index: int,
        num_slices: int,
    ) -> interpolation.SliceBridgeRenderPlan:
        section = np.ones((3, 3), dtype=bool)
        sdf = np.ones((3, 3), dtype=np.float32)
        return interpolation.SliceBridgeRenderPlan(
            source_label=1,
            target_label=2,
            source_point=(source_z, center[0], center[1]),
            target_point=(target_z, center[0], center[1]),
            source_anchor=center,
            target_anchor=center,
            steps=target_z - source_z,
            sign=1,
            num_slices=num_slices,
            sdf0=sdf,
            sdf1=sdf,
            interpolation_walk_back_index=walk_back_index,
            interpolation_candidate_index=candidate_index,
            cached_sections=[None, section, None],
        )

    @staticmethod
    def _decode_component(entry: dict[str, object]) -> np.ndarray:
        store = interpolation.RawBBoxMaskStore.open(Path(str(entry['path'])))
        try:
            return np.stack([
                store.decode_slice(idx) for idx in range(int(store.shape[0]))
            ])
        finally:
            store.close()

    def test_component_paths_are_exact_cartesian_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = interpolation.interpolation_bridge_component_paths(
                Path(tmp), 3, 2,
            )
        self.assertEqual(
            [(walk, candidate, path.name) for walk, candidate, path in specs],
            [
                (1, 1, 'walkback01_candidate01.cvol'),
                (1, 2, 'walkback01_candidate02.cvol'),
                (2, 1, 'walkback02_candidate01.cvol'),
                (2, 2, 'walkback02_candidate02.cvol'),
                (3, 1, 'walkback03_candidate01.cvol'),
                (3, 2, 'walkback03_candidate02.cvol'),
            ],
        )
        self.assertEqual(
            interpolation.interpolation_bridge_component_paths(Path(tmp), 0, 5),
            [],
        )

    def test_walk_back_counts_additional_origins_without_expanding_component_product(self) -> None:
        labels = np.zeros((5, 5, 5), dtype=np.uint16)
        seed = interpolation.SliceEndpointSeed(
            label=1,
            point=(1, 2, 2),
            direction_sign=1,
        )
        candidates = [
            interpolation.SliceProjectionCandidate(
                source_label=1,
                target_label=target,
                source_point=seed.point,
                target_point=(3 + idx, 2, 2),
                slice_distance=2 + idx,
            )
            for idx, target in enumerate((2, 3))
        ]
        source_points = [(0, 2, 2), (0, 2, 1)]
        plan = self._render_plan()

        with (
            mock.patch.object(
                interpolation,
                '_find_slice_projection_candidates',
                return_value=candidates,
            ),
            mock.patch.object(
                interpolation,
                '_collect_walkback_source_points',
                return_value=source_points,
            ) as collect,
            mock.patch.object(
                interpolation,
                '_build_linear_slice_bridge_plan',
                return_value=plan,
            ),
        ):
            result = interpolation._plan_slice_seed_bridges(
                labels_real=labels,
                seed=seed,
                max_slice_distance=4,
                search_angle_deg=15.0,
                interpolation_walk_back=2,
                interpolation_candidates=2,
                interpolate_min_radius=0.0,
            )

        self.assertEqual(collect.call_args.kwargs['walk_back'], 2)
        self.assertEqual(
            [
                (p.interpolation_walk_back_index, p.interpolation_candidate_index)
                for p in result.plans
            ],
            [(1, 1), (1, 1), (2, 1), (1, 2), (1, 2), (2, 2)],
        )
        self.assertEqual(result.default_bridges, 2)
        self.assertEqual(result.walk_back_bridges, 4)

    def test_zero_walk_back_retains_endpoint_bridge_without_component_coordinate(self) -> None:
        seed = interpolation.SliceEndpointSeed(
            label=1,
            point=(0, 0, 0),
            direction_sign=1,
        )
        candidate = interpolation.SliceProjectionCandidate(
            source_label=1,
            target_label=2,
            source_point=seed.point,
            target_point=(1, 0, 0),
            slice_distance=1,
        )
        plan = self._render_plan()
        with (
            mock.patch.object(
                interpolation,
                '_find_slice_projection_candidates',
                return_value=[candidate],
            ) as find,
            mock.patch.object(
                interpolation,
                '_collect_walkback_source_points',
                return_value=[],
            ) as collect,
            mock.patch.object(
                interpolation,
                '_build_linear_slice_bridge_plan',
                return_value=plan,
            ),
        ):
            result = interpolation._plan_slice_seed_bridges(
                labels_real=np.zeros((2, 2, 2), dtype=np.uint16),
                seed=seed,
                max_slice_distance=1,
                search_angle_deg=15.0,
                interpolation_walk_back=0,
                interpolation_candidates=3,
                interpolate_min_radius=0.0,
            )
        find.assert_called_once()
        self.assertEqual(collect.call_args.kwargs['walk_back'], 0)
        self.assertEqual(len(result.plans), 1)
        self.assertEqual(result.plans[0].interpolation_walk_back_index, 0)
        self.assertEqual(result.plans[0].interpolation_candidate_index, 1)
        self.assertEqual(result.default_bridges, 1)
        self.assertEqual(result.walk_back_bridges, 0)

    def test_packed_membership_painter_preserves_overlapping_bits(self) -> None:
        dest = np.zeros((3, 3), dtype=np.uint8)
        local = np.ones((3, 3), dtype=bool)
        interpolation._paste_local_mask_onto_slice(
            dest,
            local,
            (1.0, 1.0),
            paint_value=1,
            binary_destination=False,
        )
        interpolation._paste_local_mask_onto_slice(
            dest,
            local,
            (1.0, 1.0),
            paint_value=2,
            binary_destination=False,
        )
        np.testing.assert_array_equal(dest, np.full((3, 3), 3, dtype=np.uint8))

    def test_rendered_passes_emit_cartesian_components_and_next_pass_consumes_union(self) -> None:
        source = np.zeros((5, 11, 11), dtype=np.uint8)
        source[0, 5, 5] = np.uint8(1)
        source[4, 5, 5] = np.uint8(1)
        mask = source.copy()
        labels = np.zeros_like(mask, dtype=np.uint16)
        labels[0, 5, 5] = np.uint16(1)
        labels[4, 5, 5] = np.uint16(2)
        component_coordinates = [(walk, candidate) for walk in (1, 2) for candidate in (1, 2)]
        plan_coordinates_and_centers = [
            # The endpoint and first additional origin intentionally share layer 1.
            (1, 1, (2, 2)),
            (1, 1, (2, 5)),
            (2, 1, (2, 8)),
            (1, 2, (8, 2)),
            (1, 2, (8, 5)),
            (2, 2, (8, 8)),
        ]
        pass1_plans = [
            self._render_plan_at(
                source_z=0,
                target_z=2,
                center=center,
                walk_back_index=walk,
                candidate_index=candidate,
                num_slices=mask.shape[0],
            )
            for walk, candidate, center in plan_coordinates_and_centers
        ]
        # These sources are on the slice created by pass one. Keeping the painter real
        # while stubbing only discovery/planning makes the second pass exercise the same
        # pre-pass-mask differencing and in-place union path as production.
        pass2_plans = [
            self._render_plan_at(
                source_z=1,
                target_z=3,
                center=center,
                walk_back_index=walk,
                candidate_index=candidate,
                num_slices=mask.shape[0],
            )
            for walk, candidate, center in plan_coordinates_and_centers
        ]
        observed_label_inputs: list[np.ndarray] = []

        def _label_current_union(mask_arg: np.ndarray, *_args: object, **_kwargs: object):
            observed_label_inputs.append(np.array(mask_arg, copy=True))
            return labels.copy(), 2, []

        seed1 = interpolation.SliceEndpointSeed(label=1, point=(0, 2, 2), direction_sign=1)
        seed2 = interpolation.SliceEndpointSeed(label=1, point=(1, 2, 2), direction_sign=1)
        plan_results = [
            interpolation.SliceSeedBridgePlanResult(
                candidate_connections=2,
                accepted_connections=2,
                default_bridges=2,
                walk_back_bridges=4,
                plans=pass1_plans,
            ),
            interpolation.SliceSeedBridgePlanResult(
                candidate_connections=2,
                accepted_connections=2,
                default_bridges=2,
                walk_back_bridges=4,
                plans=pass2_plans,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch.object(
                    topology,
                    'interpolation_skip_compact_relabel_enabled',
                    return_value=False,
                ),
                mock.patch.object(
                    topology,
                    'label_foreground_volume_streaming',
                    side_effect=_label_current_union,
                ),
                mock.patch.object(
                    interpolation,
                    '_build_slice_endpoint_seeds',
                    side_effect=[([seed1], 1), ([seed2], 1)],
                ),
                mock.patch.object(
                    interpolation,
                    '_plan_slice_seed_bridges',
                    side_effect=plan_results,
                ),
                mock.patch.object(
                    interpolation,
                    'should_use_in_memory_workspace',
                    return_value=True,
                ),
            ):
                pass1_stats = interpolation.interpolate_view_volume_pass_inplace(
                    mask_mm=mask,
                    work_dir=root / 'work1',
                    pass_tag='pass1',
                    max_slice_distance=2,
                    search_angle_deg=15.0,
                    interpolation_walk_back=2,
                    interpolation_candidates=2,
                    interpolate_min_radius=0.0,
                    keep_temp=False,
                    prefer_memory=True,
                    workers=1,
                    bridge_component_dir=root / 'components1',
                )
                after_pass1 = mask.copy()
                pass2_stats = interpolation.interpolate_view_volume_pass_inplace(
                    mask_mm=mask,
                    work_dir=root / 'work2',
                    pass_tag='pass2',
                    max_slice_distance=2,
                    search_angle_deg=15.0,
                    interpolation_walk_back=2,
                    interpolation_candidates=2,
                    interpolate_min_radius=0.0,
                    keep_temp=False,
                    prefer_memory=True,
                    workers=1,
                    bridge_component_dir=root / 'components2',
                )

            np.testing.assert_array_equal(observed_label_inputs[0], source)
            np.testing.assert_array_equal(observed_label_inputs[1], after_pass1)
            for pass_index, (stats, pre_pass_mask, post_pass_mask) in enumerate((
                (pass1_stats, source, after_pass1),
                (pass2_stats, after_pass1, mask),
            ), start=1):
                entries = list(stats['bridge_component_deltas'])
                self.assertEqual(len(entries), 4)
                self.assertEqual(
                    [(int(entry['walk_back_index']), int(entry['candidate_index'])) for entry in entries],
                    component_coordinates,
                )
                self.assertEqual(int(stats['bridge_component_count']), 4)
                self.assertEqual(int(stats['bridge_component_render_word_count']), 1)
                self.assertEqual(
                    stats['bridge_component_render_storage'],
                    'packed_uint8_membership_bitplanes',
                )
                component_union = np.zeros_like(mask)
                expected_component_counts = {
                    (1, 1): 18,
                    (1, 2): 18,
                    (2, 1): 9,
                    (2, 2): 9,
                }
                for entry in entries:
                    entry_key = (
                        int(entry['walk_back_index']),
                        int(entry['candidate_index']),
                    )
                    self.assertEqual(
                        int(entry['added_voxels']),
                        expected_component_counts[entry_key],
                    )
                    np.bitwise_or(
                        component_union,
                        self._decode_component(entry),
                        out=component_union,
                    )
                expected_delta = np.where(pre_pass_mask == 0, post_pass_mask, np.uint8(0))
                np.testing.assert_array_equal(
                    component_union,
                    expected_delta,
                    err_msg=f'pass {pass_index} component union must reconstruct its aggregate delta',
                )
                np.testing.assert_array_equal(post_pass_mask, pre_pass_mask | component_union)
                self.assertEqual(int(stats['added_voxels']), int(np.count_nonzero(expected_delta)))
                self.assertFalse((root / f'components{pass_index}' / '_membership').exists())

    def test_labeling_failure_retires_all_component_outputs(self) -> None:
        mask = np.zeros((3, 5, 5), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component_dir = root / 'components'
            with (
                mock.patch.object(
                    topology,
                    'interpolation_skip_compact_relabel_enabled',
                    return_value=False,
                ),
                mock.patch.object(
                    topology,
                    'label_foreground_volume_streaming',
                    side_effect=RuntimeError('injected labeling failure'),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, 'injected labeling failure'):
                    interpolation.interpolate_view_volume_pass_inplace(
                        mask_mm=mask,
                        work_dir=root / 'work',
                        pass_tag='pass1',
                        max_slice_distance=2,
                        search_angle_deg=15.0,
                        interpolation_walk_back=2,
                        interpolation_candidates=2,
                        interpolate_min_radius=0.0,
                        keep_temp=False,
                        prefer_memory=True,
                        workers=1,
                        bridge_component_dir=component_dir,
                    )

            self.assertFalse(component_dir.exists())
            self.assertEqual(list(root.rglob('*.cvol')), [])

    def test_second_membership_allocation_failure_closes_and_retires_first_word(self) -> None:
        class _SilentEmptyWriter:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def consume_empty_range(self, _z0: int, _count: int) -> None:
                pass

            def finalize(self) -> None:
                pass

        class _FailSecondMembershipMemmap:
            def __init__(self) -> None:
                self.membership_calls = 0

            def __getattr__(self, name: str) -> object:
                return getattr(np, name)

            def memmap(self, filename: object, *args: object, **kwargs: object) -> np.memmap:
                if '_membership' in str(filename):
                    self.membership_calls += 1
                    if self.membership_calls == 2:
                        raise OSError('injected second membership allocation failure')
                return np.memmap(filename, *args, **kwargs)

        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        labels = np.zeros_like(mask, dtype=np.uint16)
        labels[0, 1, 1] = np.uint16(1)
        labels[2, 1, 1] = np.uint16(2)
        seed = interpolation.SliceEndpointSeed(
            label=1,
            point=(0, 1, 1),
            direction_sign=1,
        )
        memmap_proxy = _FailSecondMembershipMemmap()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            component_dir = root / 'components'
            with (
                mock.patch.object(
                    topology,
                    'interpolation_skip_compact_relabel_enabled',
                    return_value=False,
                ),
                mock.patch.object(
                    topology,
                    'label_foreground_volume_streaming',
                    return_value=(labels, 2, []),
                ),
                mock.patch.object(
                    interpolation,
                    '_build_slice_endpoint_seeds',
                    return_value=([seed], 1),
                ),
                mock.patch.object(
                    interpolation,
                    'should_use_in_memory_workspace',
                    return_value=True,
                ),
                mock.patch.object(
                    interpolation,
                    'IncrementalRawBBoxMaskStoreWriter',
                    _SilentEmptyWriter,
                ),
                mock.patch.object(interpolation, 'np', memmap_proxy),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    'injected second membership allocation failure',
                ):
                    # 65 logical components require two uint64 membership words.
                    interpolation.interpolate_view_volume_pass_inplace(
                        mask_mm=mask,
                        work_dir=root / 'work',
                        pass_tag='pass1',
                        max_slice_distance=2,
                        search_angle_deg=15.0,
                        interpolation_walk_back=13,
                        interpolation_candidates=5,
                        interpolate_min_radius=0.0,
                        keep_temp=False,
                        prefer_memory=True,
                        workers=1,
                        bridge_component_dir=component_dir,
                    )

            self.assertEqual(memmap_proxy.membership_calls, 2)
            self.assertFalse(component_dir.exists())
            self.assertEqual(list(root.rglob('word*.dat')), [])
            self.assertEqual(list(root.rglob('*.cvol')), [])

    def test_bridge_suffix_names_both_combination_coordinates(self) -> None:
        self.assertEqual(
            outputs.nrrd_layer_output_suffix(
                view_token='Transverse_R000p000',
                source='fullframe',
                mask_kind='bridge',
                pass_index=2,
                interpolation_walk_back_index=3,
                interpolation_candidate_index=4,
            ),
            'Transverse_R000p000_fullframe_bridge_pass02_walkback03_candidate04',
        )
        self.assertEqual(
            outputs.nrrd_layer_output_suffix(
                view_token='Transverse_R000p000',
                source='tile',
                mask_kind='bridge',
                pass_index=1,
                interpolation_walk_back_index=1,
                interpolation_candidate_index=2,
                tile_config_id='s3072_st1536',
            ),
            'Transverse_R000p000_tile_s3072_st1536_bridge_pass01_walkback01_candidate02',
        )

    def test_empty_combination_is_submitted_instead_of_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store_path = root / 'empty.cvol'
            writer = interpolation.IncrementalRawBBoxMaskStoreWriter(
                shape=(3, 5, 5),
                store_dir=store_path,
                format_name=interpolation.CVOL_FORMAT,
                desc='empty test component',
                force_path_backed=True,
            )
            writer.consume_empty_range(0, 3)
            writer.finalize()
            submitted: list[tuple[object, str]] = []
            sink = types.SimpleNamespace(
                submit_layer=lambda ref, suffix: submitted.append((ref, suffix)),
            )
            view = types.SimpleNamespace(
                name='transverse__tta_r000p000',
                family='orthogonal',
                tta_aug_id='r000p000',
                tta_angle_deg=0.0,
            )
            with (
                mock.patch.object(assembly, 'nrrd_layer_sink', return_value=sink),
                mock.patch.object(assembly, 'physical_view_name', return_value='transverse'),
                mock.patch.object(assembly, 'view_output_token', return_value='Transverse_R000p000'),
                mock.patch.object(assembly, '_nrrd_layer_key', return_value='empty-key'),
                mock.patch.object(assembly, '_nrrd_layer_name', return_value='empty-name'),
            ):
                ref = assembly.materialize_interpolation_component_nrrd_view_layer(
                    store_path,
                    added_voxels=0,
                    model_name='model',
                    view=view,
                    source='fullframe',
                    pass_index=1,
                    interpolation_walk_back_index=2,
                    interpolation_candidate_index=3,
                    stage='interpolation',
                    description='empty combination',
                    temp_dir=root,
                    workers=1,
                    keep_temp=False,
                )

        self.assertEqual(ref.path, store_path)
        self.assertEqual(ref.interpolation_walk_back_index, 2)
        self.assertEqual(ref.interpolation_candidate_index, 3)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(
            submitted[0][1],
            'Transverse_R000p000_fullframe_bridge_pass01_walkback02_candidate03',
        )


if __name__ == '__main__':
    unittest.main()
