from __future__ import annotations

import os
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


def _module_is_available(name: str) -> bool:
    loaded = sys.modules.get(str(name))
    if loaded is not None:
        return getattr(loaded, '__spec__', None) is not None
    try:
        return importlib.util.find_spec(str(name)) is not None
    except (ImportError, ValueError):
        return False


HAS_NUMERICAL_RUNTIME = all(
    _module_is_available(name) for name in ('cv2', 'scipy', 'tqdm')
)
if not HAS_NUMERICAL_RUNTIME:
    install_stubs()

from volume_tta import interpolation, topology


class InterpolationSeedSchedulingTests(unittest.TestCase):
    @staticmethod
    def _seed(z: int, cost: int) -> interpolation.SliceEndpointSeed:
        return interpolation.SliceEndpointSeed(
            label=int(z + 1),
            point=(int(z), 0, 0),
            direction_sign=1,
            planning_cost=int(cost),
        )

    def test_cost_balancing_is_stable_and_bounded_to_slice_windows(self) -> None:
        seeds = [
            self._seed(0, 1), self._seed(1, 10), self._seed(2, 5), self._seed(3, 10),
            self._seed(4, 2), self._seed(5, 9), self._seed(6, 3), self._seed(7, 100),
        ]

        window = interpolation._rebalance_slice_major_endpoint_seeds(
            seeds,
            plan_workers=2,
            window_factor=2,
        )

        self.assertEqual(window, 4)
        self.assertEqual([seed.point[0] for seed in seeds], [1, 3, 2, 0, 7, 5, 6, 4])

    def test_zero_window_factor_restores_slice_major_submission(self) -> None:
        seeds = [self._seed(0, 1), self._seed(1, 100)]
        original = list(seeds)
        window = interpolation._rebalance_slice_major_endpoint_seeds(
            seeds,
            plan_workers=2,
            window_factor=0,
        )
        self.assertEqual(window, 0)
        self.assertEqual(seeds, original)

    def test_endpoint_scan_captures_component_bbox_cost(self) -> None:
        labels = np.zeros((3, 12, 12), dtype=np.uint16)
        record = interpolation.SliceComponentRecord(
            z=1,
            label=1,
            component_index=1,
            bbox=(2, 4, 5, 8),
            anchor=(3, 5),
            area=12,
            mask_crop=np.ones((3, 4), dtype=bool),
        )
        tables = {
            0: interpolation.SliceComponentTable(0, (12, 12), [], {}),
            1: interpolation.SliceComponentTable(1, (12, 12), [record], {1: [record]}),
            2: interpolation.SliceComponentTable(2, (12, 12), [], {}),
        }

        class _Cache:
            labels_real = labels
            slice_luts = None

            @staticmethod
            def get(z: int) -> interpolation.SliceComponentTable:
                return tables[int(z)]

            @staticmethod
            def prebuild(**_kwargs: object) -> None:
                return None

        cache = _Cache()
        seeds, endpoints = topology.build_slice_endpoint_seeds_from_label_volume(
            labels,
            workers=1,
            component_cache=cache,  # type: ignore[arg-type]
        )
        self.assertEqual(endpoints, 2)
        self.assertEqual(len(seeds), 2)
        self.assertEqual({int(seed.planning_cost) for seed in seeds}, {12})

    @unittest.skipUnless(HAS_NUMERICAL_RUNTIME, 'requires OpenCV/SciPy numerical runtime')
    def test_cost_balancing_preserves_interpolation_output(self) -> None:
        source = np.zeros((8, 48, 48), dtype=np.uint8)
        source[0, 5:10, 5:10] = np.uint8(1)
        source[4, 6:13, 6:13] = np.uint8(1)
        source[1, 31:34, 6:9] = np.uint8(1)
        source[6, 31:36, 8:13] = np.uint8(1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def _run(factor: int, name: str) -> tuple[np.ndarray, dict[str, object]]:
                mask = source.copy()
                with mock.patch.dict(
                    os.environ,
                    {'YOLO_TTA_INTERPOLATION_SEED_SCHEDULE_WINDOW_FACTOR': str(int(factor))},
                    clear=False,
                ):
                    stats = interpolation.interpolate_view_volume_pass_inplace(
                        mask_mm=mask,
                        work_dir=root / name,
                        pass_tag='pass1',
                        max_slice_distance=7,
                        search_angle_deg=45.0,
                        interpolation_walk_back=1,
                        interpolation_candidates=2,
                        interpolate_min_radius=0.0,
                        prefer_memory=True,
                        reserve_bytes=0,
                        workers=4,
                        wrap_axis=False,
                    )
                return mask, dict(stats)

            slice_major, old_stats = _run(0, 'slice-major')
            cost_balanced, new_stats = _run(4, 'cost-balanced')

        np.testing.assert_array_equal(cost_balanced, slice_major)
        self.assertGreater(int(old_stats['accepted_connections']), 0)
        self.assertGreater(int(old_stats['added_voxels']), 0)
        for key in (
            'num_objects', 'num_endpoints', 'candidate_connections',
            'accepted_connections', 'default_bridges', 'walk_back_bridges',
            'skipped_by_min_radius', 'added_voxels',
        ):
            self.assertEqual(new_stats[key], old_stats[key], key)
        self.assertEqual(old_stats['planner_seed_schedule'], 'slice_major')
        self.assertEqual(new_stats['planner_seed_schedule'], 'cost_balanced_slice_window')


if __name__ == '__main__':
    unittest.main()
