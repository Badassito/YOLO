from __future__ import annotations

import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

install_stubs()

from XTA import assembly, finalization, geometry


class MultipleTileConfigurationTests(unittest.TestCase):
    def test_two_tile_configs_are_interpolated_and_namespaced_as_separate_sets(self) -> None:
        configs = geometry.resolve_tile_configs(['2048:1024', '1536:768'])
        self.assertEqual(
            [config.config_id for config in configs],
            ['s2048_st1024', 's1536_st768'],
        )

        first = np.zeros((1, 2, 2), dtype=np.uint8)
        first[0, 0, 0] = np.uint8(255)
        second = np.zeros((1, 2, 2), dtype=np.uint8)
        second[0, 1, 1] = np.uint8(255)
        destination = np.zeros_like(first)
        view = types.SimpleNamespace(
            name='transverse__tta_r000p000',
            family='orthogonal',
            tta_aug_id='r000p000',
            tta_angle_deg=0.0,
        )

        interpolation_inputs: list[set[tuple[int, int, int]]] = []
        interpolation_work_dirs: list[Path] = []
        nrrd_stages: list[str] = []
        nrrd_config_ids: list[str] = []

        def interpolate_one_set(*, mask_mm: np.ndarray, work_dir: Path, **_kwargs: object):
            interpolation_inputs.append({
                tuple(int(value) for value in index)
                for index in np.argwhere(np.asarray(mask_mm) > 0)
            })
            interpolation_work_dirs.append(Path(work_dir))
            component_dir = Path(_kwargs['bridge_component_dir'])
            return mask_mm, {
                'added_voxels': 0,
                'bridge_component_deltas': [{
                    'walk_back_index': 1,
                    'candidate_index': 1,
                    'path': str(component_dir / 'walkback01_candidate01.cvol'),
                    'added_voxels': 0,
                }],
            }

        def record_layer(_volume: np.ndarray, **kwargs: object) -> object:
            nrrd_stages.append(str(kwargs['stage']))
            nrrd_config_ids.append(str(kwargs['tile_config_id']))
            return types.SimpleNamespace(stage=str(kwargs['stage']))

        def union_into(destination_mm: np.ndarray, source_mm: np.ndarray, **_kwargs: object) -> None:
            np.bitwise_or(destination_mm, source_mm, out=destination_mm)

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                assembly,
                'interpolate_view_volume_pass_maybe_process',
                side_effect=interpolate_one_set,
            ),
            mock.patch.object(assembly, 'materialize_nrrd_view_layer', side_effect=record_layer),
            mock.patch.object(
                assembly,
                'materialize_interpolation_component_nrrd_view_layer',
                side_effect=record_layer,
            ),
            mock.patch.object(finalization, 'union_volume_into_volume', side_effect=union_into),
            mock.patch.object(assembly, 'view_processing_min_radius', return_value=0.0),
            mock.patch.object(assembly, 'view_processing_search_angle', return_value=15.0),
        ):
            results = []
            for config, accumulator in zip(configs, (first, second)):
                results.append(assembly.finalize_consolidated_tile_volume_for_parent(
                    model_name='model',
                    view=view,
                    tile_accumulator_mm=accumulator,
                    destination_mm=destination,
                    destination_lock=threading.Lock(),
                    temp_dir=Path(tmp),
                    interpolate=1,
                    interpolation_walk_back=1,
                    interpolation_candidates=1,
                    interpolate_passes=1,
                    interpolate_min_radius=0.0,
                    interpolation_search_angle=15.0,
                    keep_temp=False,
                    slice_workers=1,
                    interpolation_task_workers=1,
                    nrrd_layers_enabled=True,
                    tile_parent_mask_accumulator_mm=accumulator,
                    config_id=config.config_id,
                ))

        self.assertEqual(
            interpolation_inputs,
            [{(0, 0, 0)}, {(0, 1, 1)}],
            'tile configurations must reach interpolation independently, before final union',
        )
        self.assertEqual(
            [path.name for path in interpolation_work_dirs],
            ['s2048_st1024', 's1536_st768'],
        )
        self.assertEqual(
            nrrd_stages,
            [
                's2048_st1024_pre_tile_interpolation',
                's2048_st1024_tile_interpolation',
                's1536_st768_pre_tile_interpolation',
                's1536_st768_tile_interpolation',
            ],
        )
        self.assertEqual(
            nrrd_config_ids,
            ['s2048_st1024', 's2048_st1024', 's1536_st768', 's1536_st768'],
        )
        self.assertEqual(
            {
                assembly.nrrd_layer_output_suffix(
                    view_token='Transverse_R000p000',
                    source='tile',
                    mask_kind='yolo',
                    tile_config_id=config.config_id,
                    tile_acceptance='parent_mask',
                )
                for config in configs
            },
            {
                'Transverse_R000p000_tile_s2048_st1024_yolo_parent_mask',
                'Transverse_R000p000_tile_s1536_st768_yolo_parent_mask',
            },
        )
        self.assertEqual(
            [result.interpolation_stats[0]['tile_config_id'] for result in results],
            ['s2048_st1024', 's1536_st768'],
        )
        np.testing.assert_array_equal(
            destination,
            np.array([[[255, 0], [0, 255]]], dtype=np.uint8),
        )


if __name__ == '__main__':
    unittest.main()
