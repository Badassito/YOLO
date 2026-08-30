from __future__ import annotations

import multiprocessing as mp
import queue
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA import media


def _spawn_categorical_resize(result_queue: object, out_path: str) -> None:
    """Exercise the public helper from a freshly spawned interpreter."""
    try:
        source = np.asarray(
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        result = media.resize_categorical_volume_to_processing_cube_uint8(
            source,
            (3, 3, 3),
            Path(out_path),
            workers=2,
            prefer_memory=False,
            reserve_bytes=0,
        )
        result_queue.put(("ok", str(result.dtype), np.asarray(result).tolist()))
    except BaseException:
        result_queue.put(("error", traceback.format_exc(), None))


class CategoricalProcessingCubeResizeTests(unittest.TestCase):
    @staticmethod
    def _nearest_indices(in_count: int, out_count: int) -> np.ndarray:
        if in_count <= 1 or out_count <= 1:
            return np.zeros((out_count,), dtype=np.intp)
        return np.asarray(
            [
                int(round(float(index) * float(in_count - 1) / float(out_count - 1)))
                for index in range(out_count)
            ],
            dtype=np.intp,
        )

    def test_shape_match_returns_the_input_without_allocating(self) -> None:
        source = np.asarray(
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        with mock.patch.object(media, "allocate_workspace_array") as allocate:
            result = media.resize_categorical_volume_to_processing_cube_uint8(
                source,
                source.shape,
                Path("unused.uint8"),
            )

        self.assertIs(result, source)
        allocate.assert_not_called()

    def test_endpoint_ties_follow_the_established_tta_rounding_rule(self) -> None:
        # 4 -> 7 lands on .5 at output coordinates 1, 3, and 5.  TTA's
        # endpoint-aligned convention uses Python round (ties to even).
        np.testing.assert_array_equal(
            media._endpoint_aligned_nearest_source_indices(4, 7),
            np.asarray([0, 0, 1, 2, 2, 2, 3], dtype=np.intp),
        )
        np.testing.assert_array_equal(
            media._endpoint_aligned_nearest_source_indices(1, 4),
            np.asarray([0, 0, 0, 0], dtype=np.intp),
        )
        np.testing.assert_array_equal(
            media._endpoint_aligned_nearest_source_indices(4, 1),
            np.asarray([0], dtype=np.intp),
        )

    def test_all_axes_use_endpoint_aligned_nearest_and_output_is_binary_uint8(self) -> None:
        source = np.zeros((4, 4, 5), dtype=np.uint8)
        for t_idx in range(source.shape[0]):
            for y_idx in range(source.shape[1]):
                for x_idx in range(source.shape[2]):
                    if (11 * t_idx + 5 * y_idx + 3 * x_idx) % 7 in {0, 2, 5}:
                        source[t_idx, y_idx, x_idx] = np.uint8(255)

        out_shape = (7, 7, 8)
        allocated = np.empty(out_shape, dtype=np.uint8)
        out_path = Path("categorical_cube.uint8")
        with (
            mock.patch.object(
                media,
                "allocate_workspace_array",
                return_value=allocated,
            ) as allocate,
            mock.patch.object(
                media.cv2,
                "resize",
                side_effect=AssertionError("categorical resize must not call cv2.resize"),
            ),
        ):
            result = media.resize_categorical_volume_to_processing_cube_uint8(
                source,
                out_shape,
                out_path,
                workers=4,
                prefer_memory=False,
                reserve_bytes=1234,
            )

        source_t = self._nearest_indices(source.shape[0], out_shape[0])
        source_y = self._nearest_indices(source.shape[1], out_shape[1])
        source_x = self._nearest_indices(source.shape[2], out_shape[2])
        expected = (
            source[np.ix_(source_t, source_y, source_x)] != 0
        ).astype(np.uint8)

        self.assertIs(result, allocated)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.dtype(np.uint8))
        self.assertLessEqual(set(np.unique(result).tolist()), {0, 1})
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[-1], expected[-1])
        allocate.assert_called_once_with(
            shape=out_shape,
            dtype=np.uint8,
            path=out_path,
            desc="Categorical processing-cube volume (endpoint-aligned nearest)",
            prefer_memory=False,
            reserve_bytes=1234,
            initialize_zero=False,
        )

    def test_public_helper_executes_under_spawn(self) -> None:
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                process = context.Process(
                    target=_spawn_categorical_resize,
                    args=(result_queue, str(Path(temp_dir) / "spawn_cube.uint8")),
                )
                process.start()
                try:
                    status, detail, values = result_queue.get(timeout=45.0)
                except queue.Empty:
                    process.terminate()
                    process.join(timeout=10.0)
                    self.fail("spawned categorical resize returned no result")
                process.join(timeout=45.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10.0)
                    self.fail("spawned categorical resize did not exit")
        finally:
            result_queue.close()
            result_queue.join_thread()

        self.assertEqual(process.exitcode, 0)
        self.assertEqual(status, "ok", detail)
        self.assertEqual(detail, "uint8")
        spawned = np.asarray(values, dtype=np.uint8)
        self.assertEqual(spawned.shape, (3, 3, 3))
        self.assertLessEqual(set(np.unique(spawned).tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
