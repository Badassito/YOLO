from __future__ import annotations

import itertools
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA import assembly
from XTA import pta
from XTA.gaussian import binary_gaussian_pass


def _constant_zero_gaussian(
    input: np.ndarray,
    *,
    sigma: float,
    output: np.ndarray | None = None,
    mode: str,
    cval: float,
    truncate: float,
) -> np.ndarray:
    """Tiny independent reference backend for environments without SciPy."""
    if mode != "constant" or float(cval) != 0.0 or float(truncate) != 4.0:
        raise AssertionError((mode, cval, truncate))
    source = np.asarray(input, dtype=np.float32).copy()
    radius = int(round(float(truncate) * float(sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    result = np.zeros(source.shape, dtype=np.float32)
    shape = tuple(int(value) for value in source.shape)
    for target in itertools.product(*(range(size) for size in shape)):
        total = 0.0
        for offsets in itertools.product(range(-radius, radius + 1), repeat=3):
            source_index = tuple(int(target[axis] + offsets[axis]) for axis in range(3))
            if not all(0 <= source_index[axis] < shape[axis] for axis in range(3)):
                continue
            weight = math.prod(float(kernel[offset + radius]) for offset in offsets)
            total += float(source[source_index]) * weight
        result[target] = np.float32(total)
    if output is not None:
        output[...] = result
        return output
    return result


def _serial_runner(_phase: str, count: int, operation: object) -> None:
    for index in range(int(count)):
        operation(index)  # type: ignore[operator]


class SharedGaussianPrimitiveTests(unittest.TestCase):
    def test_threshold_is_inclusive_at_exactly_half(self) -> None:
        probabilities = np.asarray(
            [[[0.49999997, 0.5], [0.50000006, 0.0]]],
            dtype=np.float32,
        )

        def fixed_filter(input: np.ndarray, **kwargs: object) -> np.ndarray:
            self.assertEqual(kwargs["mode"], "constant")
            self.assertEqual(kwargs["cval"], 0.0)
            self.assertEqual(kwargs["truncate"], 4.0)
            return probabilities.copy()

        actual = binary_gaussian_pass(
            np.zeros(probabilities.shape, dtype=np.uint8),
            sigma=1.25,
            gaussian_filter=fixed_filter,
        )
        np.testing.assert_array_equal(
            actual,
            np.asarray([[[0, 1], [1, 0]]], dtype=np.uint8),
        )

    def test_pta_tta_exact_equality_at_zero_boundary_across_passes(self) -> None:
        source = np.ones((5, 5, 5), dtype=np.uint8)
        sigma = 0.8

        expected_first = (
            _constant_zero_gaussian(
                source.astype(np.float32),
                sigma=sigma,
                mode="constant",
                cval=0.0,
                truncate=4.0,
            )
            >= 0.5
        ).astype(np.uint8)
        expected_second = (
            _constant_zero_gaussian(
                expected_first.astype(np.float32),
                sigma=sigma,
                mode="constant",
                cval=0.0,
                truncate=4.0,
            )
            >= 0.5
        ).astype(np.uint8)
        self.assertEqual(int(expected_first[0, 0, 0]), 0)
        self.assertEqual(int(expected_first[2, 2, 2]), 1)

        filter_inputs: list[np.ndarray] = []

        def recording_filter(input: np.ndarray, **kwargs: object) -> np.ndarray:
            filter_inputs.append(np.asarray(input, dtype=np.float32).copy())
            return _constant_zero_gaussian(input, **kwargs)  # type: ignore[arg-type]

        pta_mask = source.copy()
        tta_mask = source.copy()
        with (
            mock.patch.object(pta.ndi, "gaussian_filter", side_effect=recording_filter),
            mock.patch.object(
                assembly.ndi, "gaussian_filter", side_effect=recording_filter
            ),
            mock.patch.object(assembly, "gaussian_smoothing_gpu_enabled", return_value=False),
            mock.patch.object(
                assembly,
                "allocate_workspace_array",
                side_effect=lambda **kwargs: np.empty(kwargs["shape"], dtype=kwargs["dtype"]),
            ),
            mock.patch.object(assembly, "close_memmap_array"),
            mock.patch.object(assembly, "flush_array"),
            mock.patch.object(assembly, "choose_slice_parallel_workers", return_value=1),
            mock.patch.object(
                assembly,
                "parallel_for_indices_chunked",
                side_effect=lambda total, function, **_kwargs: [
                    function(index) for index in range(int(total))
                ],
            ),
            tempfile.TemporaryDirectory() as tmp,
        ):
            pta_stats = pta.apply_gaussian_smoothing(
                pta_mask,
                sigma=sigma,
                passes=2,
                warnings=pta.WarningLog(),
            )
            tta_stats = assembly.apply_gaussian_smoothing_inplace(
                tta_mask,
                sigma=sigma,
                passes=2,
                temp_dir=Path(tmp),
                reserve_bytes=0,
                workers=1,
            )

        np.testing.assert_array_equal(pta_mask, expected_second)
        np.testing.assert_array_equal(tta_mask, expected_second)
        np.testing.assert_array_equal(pta_mask, tta_mask)
        self.assertEqual(len(pta_stats), 2)
        self.assertEqual(tta_stats["passes_completed"], 2)
        self.assertEqual(tta_stats["backend"], "cpu_scipy_ndimage")

        self.assertEqual(len(filter_inputs), 4)
        np.testing.assert_array_equal(filter_inputs[0], source.astype(np.float32))
        np.testing.assert_array_equal(filter_inputs[1], expected_first.astype(np.float32))
        np.testing.assert_array_equal(filter_inputs[2], source.astype(np.float32))
        np.testing.assert_array_equal(filter_inputs[3], expected_first.astype(np.float32))

    def test_workspace_alias_preserves_per_slice_change_observation(self) -> None:
        source = np.ones((3, 3, 3), dtype=np.uint8)
        workspace = np.empty(source.shape, dtype=np.float32)
        changes: list[tuple[int, int, int]] = []

        def observe(z: int, old: np.ndarray, new: np.ndarray) -> None:
            changes.append(
                (
                    int(z),
                    int(np.count_nonzero(new & ~old)),
                    int(np.count_nonzero(old & ~new)),
                )
            )

        binary_gaussian_pass(
            source,
            sigma=0.8,
            gaussian_filter=_constant_zero_gaussian,
            float_workspace=workspace,
            destination=source,
            slice_runner=_serial_runner,
            observe_slice=observe,
        )

        self.assertEqual([item[0] for item in changes], [0, 1, 2])
        self.assertGreater(sum(item[2] for item in changes), 0)
        self.assertEqual(sum(item[1] for item in changes), 0)


if __name__ == "__main__":
    unittest.main()
