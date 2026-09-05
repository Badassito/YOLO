from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    install_stubs()

from XTA.lta_outputs import LtaLayerRecord, compose_terminal_union
from XTA import lta_postprocessing as subject


def _reference_fill_4_connected(mask: np.ndarray) -> np.ndarray:
    """Tiny independent outside-background flood fill for unit tests."""

    binary = np.asarray(mask, dtype=bool).copy()
    height, width = binary.shape
    outside = np.zeros(binary.shape, dtype=bool)
    pending: deque[tuple[int, int]] = deque()
    for x in range(width):
        pending.append((0, x))
        pending.append((height - 1, x))
    for y in range(height):
        pending.append((y, 0))
        pending.append((y, width - 1))
    while pending:
        y, x = pending.popleft()
        if outside[y, x] or binary[y, x]:
            continue
        outside[y, x] = True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                pending.append((ny, nx))
    return binary | ((~binary) & (~outside))


def _fake_operations(
    events: list[str],
    closed: list[object],
    *,
    gpu_result: object = None,
    gaussian_error: Exception | None = None,
) -> subject.LtaFinalizationOperations:
    def allocate(**kwargs: object) -> np.ndarray:
        events.append("allocate")
        output = np.empty(kwargs["shape"], dtype=kwargs["dtype"])  # type: ignore[arg-type]
        if bool(kwargs.get("initialize_zero", False)):
            output.fill(0)
        return output

    def flush(_volume: object, **_kwargs: object) -> None:
        events.append("flush")

    def close(volume: object) -> None:
        closed.append(volume)

    def void(volume: np.ndarray, _path: Path, **_kwargs: object) -> None:
        events.append("void")
        volume[0, 0, 0] = 1

    def gaussian(
        volume: np.ndarray,
        _sigma: float,
        _passes: int,
        _temp: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        events.append("gaussian")
        if gaussian_error is not None:
            raise gaussian_error
        volume[0, 0, 1] = 1
        return {"enabled": 1, "passes_completed": 1, "backend": "fake"}

    def gpu_keep(
        _volume: np.ndarray,
        _keep: int,
        _temp: Path,
        **_kwargs: object,
    ) -> object:
        events.append("gpu_keep")
        return gpu_result

    def cpu_keep(
        volume: np.ndarray,
        _keep: int,
        _temp: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        events.append("cpu_keep")
        volume[0, 0, 2] = 1
        return {"enabled": 1, "kept_objects": 1}

    return subject.LtaFinalizationOperations(
        allocate_workspace_array=allocate,
        flush_array=flush,
        close_volume=close,
        fill_3d_voids=void,
        apply_gaussian_smoothing=gaussian,
        try_gpu_keep_objects=gpu_keep,
        apply_cpu_keep_objects=cpu_keep,
    )


class LtaMaskHoleFillTests(unittest.TestCase):
    def test_non_mutating_fill_uses_four_connected_outside_background(self) -> None:
        source = np.zeros((7, 7), dtype=np.uint8)
        source[1:6, 1] = 1
        source[1:6, 5] = 1
        source[1, 1:6] = 1
        source[5, 1:6] = 1
        # A separate channel connects this background pixel to the image edge.
        source[0:4, 0] = 0
        source_before = source.copy()

        actual = subject.fill_binary_mask_holes_2d(
            source,
            operation=_reference_fill_4_connected,
        )

        np.testing.assert_array_equal(source, source_before)
        self.assertTrue(actual.dtype == np.bool_)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertTrue(bool(actual[3, 3]))
        self.assertFalse(bool(actual[0, 0]))

    def test_fill_rejects_an_operation_that_removes_foreground(self) -> None:
        source = np.ones((3, 3), dtype=bool)
        with self.assertRaisesRegex(ValueError, "removed foreground"):
            subject.fill_binary_mask_holes_2d(
                source,
                operation=lambda value: np.zeros_like(value),
            )

    def test_prediction_and_dogfood_helpers_keep_object_order_and_provenance(self) -> None:
        donut = np.ones((5, 5), dtype=bool)
        donut[2, 2] = False
        point = np.zeros((5, 5), dtype=bool)
        point[0, 0] = True
        donut_before = donut.copy()

        prediction = subject.fill_prediction_mask_holes_2d(
            donut,
            context={"frame_index": 11, "object_id": 4},
            operation=_reference_fill_4_connected,
        )
        dogfood = subject.fill_dogfood_seed_masks_holes_2d(
            (donut, point),
            context={"source_frame": 11, "direction": "forward"},
            operation=_reference_fill_4_connected,
        )

        np.testing.assert_array_equal(donut, donut_before)
        self.assertTrue(bool(prediction.mask[2, 2]))
        self.assertEqual(prediction.receipt.provenance_kind, "sam_prediction")
        self.assertEqual(prediction.receipt.added_pixels, 1)
        self.assertEqual(prediction.receipt.context["object_id"], 4)
        self.assertEqual([item.mask_index for item in dogfood.receipts], [0, 1])
        self.assertEqual(
            [item.provenance_kind for item in dogfood.receipts],
            ["temporal_dogfood", "temporal_dogfood"],
        )
        self.assertTrue(bool(dogfood.masks[0][2, 2]))
        np.testing.assert_array_equal(dogfood.masks[1], point)

    def test_merged_spatial_seed_is_filled_after_union(self) -> None:
        merged = np.ones((5, 5), dtype=bool)
        merged[2, 2] = False
        result = subject.fill_merged_seed_mask_holes_2d(
            merged,
            context={"source_tile_indices": [1, 3], "destination_tile_index": 4},
            operation=_reference_fill_4_connected,
        )

        self.assertTrue(bool(result.mask[2, 2]))
        self.assertEqual(result.receipt.provenance_kind, "merged_spatial_relay")
        self.assertEqual(result.receipt.context["source_tile_indices"], [1, 3])

    def test_completed_view_wrapper_mutates_uint8_and_reports_added_pixels(self) -> None:
        volume = np.zeros((2, 5, 5), dtype=np.uint8)
        volume[:, 1:4, 1] = 1
        volume[:, 1:4, 3] = 1
        volume[:, 1, 1:4] = 1
        volume[:, 3, 1:4] = 1
        calls: list[dict[str, object]] = []

        def fill_view(target: np.ndarray, **kwargs: object) -> None:
            calls.append(dict(kwargs))
            for index in range(int(target.shape[0])):
                target[index] = _reference_fill_4_connected(target[index])

        receipt = subject.fill_completed_view_holes_2d_inplace(
            volume,
            runtime_view_id="transverse__tta_a0",
            workers=3,
            operation=fill_view,
        )

        self.assertTrue(bool(volume[:, 2, 2].all()))
        self.assertEqual(receipt.added_pixels, 2)
        self.assertEqual(receipt.shape_tyx, (2, 5, 5))
        self.assertEqual(calls[0]["workers"], 3)
        self.assertIn("transverse__tta_a0", str(calls[0]["desc"]))
        with self.assertRaisesRegex(TypeError, "uint8"):
            subject.fill_completed_view_holes_2d_inplace(
                volume.astype(bool),
                runtime_view_id="bad",
                operation=fill_view,
            )


class LtaNativeFinalizationTests(unittest.TestCase):
    def test_scalable_composition_matches_role_aware_reference(self) -> None:
        first = np.zeros((2, 3, 4), dtype=np.int16)
        first[0, 1, 1] = 7
        added = np.zeros_like(first)
        added[1, 1, 2] = -2
        removed = np.zeros_like(first)
        removed[0, 1, 1] = 1
        checkpoint = np.zeros_like(first)
        checkpoint[0, 0, 0] = 5
        layers = (
            LtaLayerRecord("first", "union", "sam", first),
            LtaLayerRecord("added", "union", "sam", added),
            LtaLayerRecord("removed", "subtract_from_previous_checkpoint", "audit", removed),
            LtaLayerRecord("ignored", "none", "diagnostic", np.ones((1, 1, 1))),
            LtaLayerRecord("checkpoint", "select", "checkpoint", checkpoint),
        )
        expected = compose_terminal_union(layers)
        events: list[str] = []
        closed: list[object] = []

        with tempfile.TemporaryDirectory() as folder:
            result = subject.finalize_lta_native_union(
                layers,
                workspace_path=Path(folder) / "union.u8.dat",
                temp_dir=Path(folder),
                workers=1,
                operations=_fake_operations(events, closed),
            )
            np.testing.assert_array_equal(result.terminal_union, expected)
            self.assertEqual(result.composition_layer_ids, ("first", "added", "removed", "checkpoint"))
            self.assertEqual(result.postprocessing["execution_order"], [])
            result.close()

        self.assertEqual(len(closed), 1)
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            result.close()

    def test_terminal_filters_run_void_gaussian_gpu_then_cpu_fallback(self) -> None:
        source = np.zeros((1, 2, 4), dtype=np.uint8)
        source[0, 1, 3] = 1
        events: list[str] = []
        closed: list[object] = []
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
            os.environ,
            {"YOLO_TTA_VOIDFILL_CONNECTIVITY": "18"},
        ):
            result = subject.finalize_lta_native_union(
                (LtaLayerRecord("prediction", "union", "sam", source),),
                workspace_path=Path(folder) / "union.u8.dat",
                temp_dir=Path(folder),
                postprocessing={
                    "enable_3d_void_fill": True,
                    "gaussian_smoothing_enabled": True,
                    "gaussian_sigma": 1.5,
                    "gaussian_passes": 2,
                    "keep_objects": 1,
                },
                operations=_fake_operations(events, closed),
            )

            self.assertEqual(
                [event for event in events if event in {"void", "gaussian", "gpu_keep", "cpu_keep"}],
                ["void", "gaussian", "gpu_keep", "cpu_keep"],
            )
            self.assertEqual(
                result.postprocessing["execution_order"],
                ["3d_void_fill", "gaussian_smoothing", "keep_objects"],
            )
            self.assertEqual(result.postprocessing["keep_objects"]["backend"], "cpu")
            self.assertEqual(
                result.postprocessing["void_fill"]["background_connectivity"],
                18,
            )
            self.assertEqual(result.terminal_union_foreground_voxels, 4)
            result.close()

    def test_gpu_keep_replacement_transfers_ownership_without_cpu_mutation(self) -> None:
        source = np.ones((1, 2, 3), dtype=np.uint8)
        replacement = np.zeros_like(source)
        replacement[0, 0, 0] = 1
        gpu_result = SimpleNamespace(
            volume=replacement,
            stats={"enabled": 1, "kept_objects": 1},
            candidate_path=Path("candidate.u8.dat"),
        )
        events: list[str] = []
        closed: list[object] = []
        with tempfile.TemporaryDirectory() as folder:
            result = subject.finalize_lta_native_union(
                (LtaLayerRecord("prediction", "union", "sam", source),),
                workspace_path=Path(folder) / "union.u8.dat",
                temp_dir=Path(folder),
                postprocessing={"keep_objects": 1},
                operations=_fake_operations(events, closed, gpu_result=gpu_result),
            )

            self.assertIs(result.terminal_union, replacement)
            self.assertEqual(events.count("cpu_keep"), 0)
            self.assertEqual(result.postprocessing["keep_objects"]["backend"], "multi_gpu")
            self.assertEqual(len(closed), 1)
            self.assertIsNot(closed[0], replacement)
            result.close()

        self.assertEqual(len(closed), 2)
        self.assertIs(closed[-1], replacement)

    def test_destructive_filter_cannot_remove_protected_hard_positive(self) -> None:
        prediction = np.ones((1, 2, 3), dtype=np.uint8)
        hard_positive = np.zeros_like(prediction)
        hard_positive[0, 1, 2] = 1
        replacement = np.zeros_like(prediction)
        replacement[0, 0, 0] = 1
        gpu_result = SimpleNamespace(
            volume=replacement,
            stats={"enabled": 1, "kept_objects": 1},
            candidate_path=Path("candidate.u8.dat"),
        )
        prediction_layer = LtaLayerRecord("prediction", "union", "sam", prediction)
        hard_positive_layer = LtaLayerRecord(
            "authoritative_hard_positives",
            "union",
            "authoritative_annotation",
            hard_positive,
        )
        events: list[str] = []
        closed: list[object] = []

        with tempfile.TemporaryDirectory() as folder:
            result = subject.finalize_lta_native_union(
                (prediction_layer, hard_positive_layer),
                workspace_path=Path(folder) / "union.u8.dat",
                temp_dir=Path(folder),
                protected_foreground_layers=(hard_positive_layer,),
                postprocessing={"keep_objects": 1},
                operations=_fake_operations(events, closed, gpu_result=gpu_result),
            )

            self.assertTrue(bool(result.terminal_union[0, 0, 0]))
            self.assertTrue(bool(result.terminal_union[0, 1, 2]))
            self.assertEqual(
                result.postprocessing["execution_order"],
                ["keep_objects", "restore_protected_foreground"],
            )
            self.assertEqual(
                result.postprocessing["protected_foreground"],
                {
                    "layer_ids": ["authoritative_hard_positives"],
                    "restore_stage": "after_terminal_filters",
                    "applied": True,
                },
            )
            result.close()

    def test_filter_failure_closes_the_current_authoritative_union(self) -> None:
        source = np.ones((1, 2, 3), dtype=np.uint8)
        events: list[str] = []
        closed: list[object] = []
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "gaussian failed"):
                subject.finalize_lta_native_union(
                    (LtaLayerRecord("prediction", "union", "sam", source),),
                    workspace_path=Path(folder) / "union.u8.dat",
                    temp_dir=Path(folder),
                    postprocessing={"gaussian_smoothing_enabled": True},
                    operations=_fake_operations(
                        events,
                        closed,
                        gaussian_error=RuntimeError("gaussian failed"),
                    ),
                )

        self.assertEqual(len(closed), 1)

    def test_detach_transfers_volume_and_prevents_result_cleanup(self) -> None:
        source = np.ones((1, 1, 2), dtype=np.uint8)
        events: list[str] = []
        closed: list[object] = []
        with tempfile.TemporaryDirectory() as folder:
            result = subject.finalize_lta_native_union(
                (LtaLayerRecord("prediction", "union", "sam", source),),
                workspace_path=Path(folder) / "union.u8.dat",
                temp_dir=Path(folder),
                operations=_fake_operations(events, closed),
            )
            detached = result.detach_terminal_union()

        self.assertIs(detached, result.terminal_union)
        self.assertTrue(result.closed)
        self.assertEqual(closed, [])
        with self.assertRaisesRegex(RuntimeError, "already closed or detached"):
            result.detach_terminal_union()


class LtaFinalNrrdTests(unittest.TestCase):
    def test_default_writer_serializes_an_empty_global_checkpoint(self) -> None:
        volume = np.zeros((2, 3, 4), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            artifact = subject.write_global_final_output_nrrd(
                Path(folder),
                stem="empty",
                terminal_union=volume,
            )
            header_bytes, compressed = artifact.receipt.path.read_bytes().split(b"\n\n", 1)

        header = header_bytes.decode("ascii")
        self.assertIn("sizes: 4 3 2", header)
        self.assertIn("Segment0_Extent:=0 -1 0 -1 0 -1", header)
        self.assertEqual(gzip.decompress(compressed), volume.tobytes(order="C"))

    def test_empty_terminal_union_is_written_and_linked_to_its_receipt(self) -> None:
        volume = np.zeros((2, 3, 4), dtype=np.uint8)
        observed: dict[str, object] = {}

        def writer(ref: object, shape: tuple[int, int, int], path: Path, **kwargs: object) -> Path:
            observed["shape"] = shape
            observed["live_array"] = getattr(ref, "live_array")
            observed["segment_name"] = kwargs["segment_name"]
            path.write_bytes(np.asarray(getattr(ref, "live_array"), dtype=np.uint8).tobytes())
            return path

        with tempfile.TemporaryDirectory() as folder:
            artifact = subject.write_global_final_output_nrrd(
                Path(folder),
                stem="sample",
                terminal_union=volume,
                writer=writer,
            )
            record = artifact.manifest_record()
            validated = artifact.receipt.validate()
            leftovers = list(Path(folder).glob("*.assembling"))

        self.assertEqual(artifact.receipt.path.name, "sample_Global_final_output.seg.nrrd")
        self.assertEqual(observed["shape"], (2, 3, 4))
        self.assertIs(observed["live_array"], volume)
        self.assertEqual(observed["segment_name"], "sample_Global_final_output")
        self.assertTrue(artifact.layer.metadata["empty_union"])
        self.assertEqual(artifact.layer.metadata["artifact_name"], artifact.receipt.name)
        self.assertEqual(record["layer_id"], artifact.layer.layer_id)
        self.assertEqual(record["sha256"], validated["sha256"])
        self.assertEqual(leftovers, [])

    def test_writer_failure_preserves_prior_public_nrrd_and_cleans_stage(self) -> None:
        volume = np.ones((1, 2, 2), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = root / "sample_Global_final_output.seg.nrrd"
            destination.write_bytes(b"previous")

            def failing_writer(
                _ref: object,
                _shape: tuple[int, int, int],
                path: Path,
                **_kwargs: object,
            ) -> Path:
                path.write_bytes(b"incomplete")
                raise RuntimeError("writer failed")

            with self.assertRaisesRegex(RuntimeError, "writer failed"):
                subject.write_global_final_output_nrrd(
                    root,
                    stem="sample",
                    terminal_union=volume,
                    writer=failing_writer,
                )

            self.assertEqual(destination.read_bytes(), b"previous")
            self.assertEqual(list(root.glob("*.assembling")), [])


if __name__ == "__main__":
    unittest.main()
