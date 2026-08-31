from __future__ import annotations

import unittest
from unittest import mock

from tools.smoke_import import install_stubs


install_stubs()

from XTA import pipeline
from XTA.tta_outputs import (
    TtaOutputArtifacts,
    TtaOutputInputs,
    TtaOutputOperations,
)


class _Artifact:
    def __init__(self, name: str) -> None:
        self.name = str(name)

    def __repr__(self) -> str:
        return self.name


def _name(value: object) -> str:
    return "None" if value is None else str(getattr(value, "name", value))


def _operations(events: list[str], *, fail_on_close: int = 0) -> TtaOutputOperations:
    close_calls = 0

    def close(value: object) -> None:
        nonlocal close_calls
        close_calls += 1
        events.append(f"close:{_name(value)}")
        if int(fail_on_close) == int(close_calls):
            raise RuntimeError("injected close failure")

    return TtaOutputOperations(
        close_memmap_array=close,
        close_raw_store_or_memmap_volume=lambda value, *, keep_temp: events.append(
            f"raw:{_name(value)}:{bool(keep_temp)}"
        ),
        archive_or_delete_binary_volume_storage=(
            lambda value, *, keep_temp, workers, desc: events.append(
                f"archive:{_name(value)}:{bool(keep_temp)}:{int(workers)}:{desc}"
            )
        ),
        unload_yolo_model=lambda value: events.append(f"unload:{_name(value)}"),
        trim_cuda_memory=lambda: events.append("trim"),
        collect_garbage=lambda: events.append("gc"),
    )


def _artifacts(*, same_mask: bool = False, same_source: bool = False) -> TtaOutputArtifacts:
    final_union = _Artifact("final_union")
    input_volume = _Artifact("input")
    return TtaOutputArtifacts(
        final_output_mask_mm=final_union if same_mask else _Artifact("final_mask"),
        final_union_mm=final_union,
        native_view_support_by_model={"model": {"native": _Artifact("native")}},
        radial_native_output_by_model={"model": {"radial": _Artifact("radial")}},
        tilted_native_output_by_model={"model": {"tilted": _Artifact("tilted")}},
        view_volumes_by_model={"model": {"view": _Artifact("view")}},
        parent_mask_support_by_model={"model": {"mask": _Artifact("parent_mask")}},
        parent_bridge_support_by_model={
            "model": {"bridge": _Artifact("parent_bridge")}
        },
        tile_accumulator_by_set={
            ("model", "view", "tile"): _Artifact("tile_union")
        },
        tile_parent_mask_accumulator_by_set={
            ("model", "view", "tile"): _Artifact("tile_mask")
        },
        tile_parent_bridge_accumulator_by_set={
            ("model", "view", "tile"): _Artifact("tile_bridge")
        },
        baseline_union_by_model_view={
            ("model", "view"): _Artifact("baseline_union")
        },
        baseline_confmap_by_model_view={
            ("model", "view"): _Artifact("baseline_conf")
        },
        yolo_models=(("empty", None), ("model", _Artifact("yolo"))),
        volume_rgb=input_volume if same_source else _Artifact("processing"),
        input_volume_rgb=input_volume,
    )


class TtaOutputOwnershipTests(unittest.TestCase):
    def test_owner_preserves_exact_teardown_order_and_clear_semantics(self) -> None:
        events: list[str] = []
        artifacts = _artifacts()
        result = artifacts.close(
            inputs=TtaOutputInputs(
                keep_temp_artifacts=False,
                tile_slice_workers=7,
            ),
            operations=_operations(events),
        )

        self.assertEqual(events, [
            "close:final_mask",
            "close:final_union",
            "close:native",
            "close:radial",
            "close:tilted",
            "close:view",
            "raw:parent_mask:False",
            "raw:parent_bridge:False",
            "archive:tile_union:False:7:remaining consolidated tile accumulator",
            "archive:tile_mask:False:7:remaining parent-mask tile category accumulator",
            "archive:tile_bridge:False:7:remaining parent-bridge tile category accumulator",
            "close:baseline_union",
            "close:baseline_conf",
            "unload:yolo",
            "close:processing",
            "close:input",
            "trim",
            "gc",
        ])
        self.assertTrue(artifacts.closed)
        self.assertEqual(result.close_memmap_calls, 10)
        self.assertEqual(result.retired_tile_accumulators, 3)
        self.assertEqual(result.unloaded_models, 1)
        self.assertTrue(result.processing_volume_was_distinct)
        self.assertEqual(artifacts.native_view_support_by_model, {"model": {}})
        self.assertEqual(artifacts.radial_native_output_by_model, {"model": {}})
        self.assertEqual(artifacts.tilted_native_output_by_model, {"model": {}})
        self.assertEqual(artifacts.view_volumes_by_model, {"model": {}})
        self.assertEqual(artifacts.parent_mask_support_by_model, {"model": {}})
        self.assertEqual(artifacts.parent_bridge_support_by_model, {"model": {}})
        self.assertFalse(artifacts.tile_accumulator_by_set)
        self.assertFalse(artifacts.tile_parent_mask_accumulator_by_set)
        self.assertFalse(artifacts.tile_parent_bridge_accumulator_by_set)
        self.assertTrue(artifacts.baseline_union_by_model_view)
        self.assertTrue(artifacts.baseline_confmap_by_model_view)

    def test_identical_final_and_source_arrays_close_once_per_owned_handle(self) -> None:
        events: list[str] = []
        artifacts = _artifacts(same_mask=True, same_source=True)
        result = artifacts.close(
            inputs=TtaOutputInputs(True, 1),
            operations=_operations(events),
        )

        self.assertEqual(events.count("close:final_union"), 1)
        self.assertEqual(events.count("close:input"), 1)
        self.assertNotIn("close:processing", events)
        self.assertFalse(result.processing_volume_was_distinct)
        self.assertEqual(result.close_memmap_calls, 8)

    def test_partial_failure_is_fail_closed_and_cannot_double_close(self) -> None:
        events: list[str] = []
        artifacts = _artifacts()
        with self.assertRaisesRegex(RuntimeError, "injected close failure"):
            artifacts.close(
                inputs=TtaOutputInputs(False, 1),
                operations=_operations(events, fail_on_close=2),
            )

        self.assertEqual(events, ["close:final_mask", "close:final_union"])
        self.assertFalse(artifacts.closed)
        with self.assertRaisesRegex(RuntimeError, "already entered teardown"):
            artifacts.close(
                inputs=TtaOutputInputs(False, 1),
                operations=_operations(events),
            )
        self.assertEqual(events, ["close:final_mask", "close:final_union"])

    def test_pipeline_facade_injects_current_operations_then_finalizes(self) -> None:
        artifacts = _artifacts(same_mask=True, same_source=True)
        after_close = mock.Mock()
        with (
            mock.patch.object(pipeline, "close_memmap_array") as close,
            mock.patch.object(pipeline, "close_raw_store_or_memmap_volume") as close_raw,
            mock.patch.object(
                pipeline, "archive_or_delete_binary_volume_storage"
            ) as archive,
            mock.patch.object(pipeline, "unload_yolo_model") as unload,
            mock.patch.object(pipeline, "trim_cuda_memory") as trim,
            mock.patch.object(pipeline.gc, "collect") as collect,
        ):
            result = pipeline._close_tta_output_artifacts(
                artifacts,
                keep_temp_artifacts=False,
                tile_slice_workers=3,
                after_close=after_close,
            )

        self.assertTrue(result.close_memmap_calls > 0)
        self.assertTrue(close.called)
        self.assertTrue(close_raw.called)
        self.assertTrue(archive.called)
        unload.assert_called_once()
        trim.assert_called_once()
        collect.assert_called_once()
        after_close.assert_called_once_with()

    def test_close_failure_prevents_complete_publication_callback(self) -> None:
        artifacts = _artifacts()
        publish_complete = mock.Mock()
        with mock.patch.object(
            pipeline,
            "close_memmap_array",
            side_effect=RuntimeError("locked output artifact"),
        ):
            with self.assertRaisesRegex(RuntimeError, "locked output artifact"):
                pipeline._close_tta_output_artifacts(
                    artifacts,
                    keep_temp_artifacts=False,
                    tile_slice_workers=1,
                    after_close=publish_complete,
                )

        publish_complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
