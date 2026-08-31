"""Single-owner teardown handoff for settled TTA output artifacts.

The global assembly, output scheduling, and manifest construction phases still live in the
pipeline because they actively reshape and cross-reference view/tile registries.  This module
establishes the prerequisite for their later extraction: once those phases settle, every live
array, registry, model, and source-volume handle is transferred by identity to one fail-closed
owner and retired in the exact pre-refactor order before complete-manifest publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TtaOutputInputs:
    """Run-constant teardown policy."""

    keep_temp_artifacts: bool
    tile_slice_workers: int


@dataclass(frozen=True)
class TtaOutputOperations:
    """Injected teardown operations preserving pipeline monkeypatch seams."""

    close_memmap_array: Callable[[object], None]
    close_raw_store_or_memmap_volume: Callable[..., None]
    archive_or_delete_binary_volume_storage: Callable[..., None]
    unload_yolo_model: Callable[[object], None]
    trim_cuda_memory: Callable[[], None]
    collect_garbage: Callable[[], object]


@dataclass(frozen=True)
class TtaOutputResult:
    """Immutable confirmation of the completed ownership transaction."""

    close_memmap_calls: int
    retired_tile_accumulators: int
    unloaded_models: int
    processing_volume_was_distinct: bool


@dataclass
class TtaOutputArtifacts:
    """Identity-preserving owner of every artifact live at final output teardown."""

    final_output_mask_mm: np.ndarray
    final_union_mm: np.ndarray
    native_view_support_by_model: Dict[str, Dict[str, np.ndarray]]
    radial_native_output_by_model: Dict[str, Dict[str, np.ndarray]]
    tilted_native_output_by_model: Dict[str, Dict[str, np.ndarray]]
    view_volumes_by_model: Dict[str, Dict[str, np.ndarray]]
    parent_mask_support_by_model: Dict[str, Dict[str, object]]
    parent_bridge_support_by_model: Dict[str, Dict[str, object]]
    tile_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray]
    tile_parent_mask_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray]
    tile_parent_bridge_accumulator_by_set: Dict[Tuple[str, str, str], np.ndarray]
    baseline_union_by_model_view: Mapping[Tuple[str, str], np.ndarray]
    baseline_confmap_by_model_view: Mapping[
        Tuple[str, str], Optional[np.ndarray]
    ]
    yolo_models: Sequence[Tuple[str, Optional[object]]]
    volume_rgb: object
    input_volume_rgb: object
    _close_started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def close(
        self,
        *,
        inputs: TtaOutputInputs,
        operations: TtaOutputOperations,
    ) -> TtaOutputResult:
        """Retire all transferred artifacts exactly once in publication-safe order."""

        if self._close_started:
            raise RuntimeError("TTA output artifacts have already entered teardown")
        self._close_started = True

        close_memmap_calls = 0
        retired_tile_accumulators = 0
        unloaded_models = 0
        processing_volume_was_distinct = bool(
            self.volume_rgb is not self.input_volume_rgb
        )

        if self.final_output_mask_mm is not self.final_union_mm:
            operations.close_memmap_array(self.final_output_mask_mm)
            close_memmap_calls += 1
        operations.close_memmap_array(self.final_union_mm)
        close_memmap_calls += 1

        for model_support in self.native_view_support_by_model.values():
            for volume in model_support.values():
                operations.close_memmap_array(volume)
                close_memmap_calls += 1
            model_support.clear()
        for model_views in self.radial_native_output_by_model.values():
            for volume in model_views.values():
                operations.close_memmap_array(volume)
                close_memmap_calls += 1
            model_views.clear()
        for model_views in self.tilted_native_output_by_model.values():
            for volume in model_views.values():
                operations.close_memmap_array(volume)
                close_memmap_calls += 1
            model_views.clear()
        for model_views in self.view_volumes_by_model.values():
            for volume in model_views.values():
                operations.close_memmap_array(volume)
                close_memmap_calls += 1
            model_views.clear()

        for model_support in self.parent_mask_support_by_model.values():
            for support in model_support.values():
                operations.close_raw_store_or_memmap_volume(
                    support,
                    keep_temp=bool(inputs.keep_temp_artifacts),
                )
            model_support.clear()
        for model_support in self.parent_bridge_support_by_model.values():
            for support in model_support.values():
                operations.close_raw_store_or_memmap_volume(
                    support,
                    keep_temp=bool(inputs.keep_temp_artifacts),
                )
            model_support.clear()

        for accumulator in self.tile_accumulator_by_set.values():
            operations.archive_or_delete_binary_volume_storage(
                accumulator,
                keep_temp=bool(inputs.keep_temp_artifacts),
                workers=int(inputs.tile_slice_workers),
                desc="remaining consolidated tile accumulator",
            )
            retired_tile_accumulators += 1
        self.tile_accumulator_by_set.clear()
        for accumulator in self.tile_parent_mask_accumulator_by_set.values():
            operations.archive_or_delete_binary_volume_storage(
                accumulator,
                keep_temp=bool(inputs.keep_temp_artifacts),
                workers=int(inputs.tile_slice_workers),
                desc="remaining parent-mask tile category accumulator",
            )
            retired_tile_accumulators += 1
        self.tile_parent_mask_accumulator_by_set.clear()
        for accumulator in self.tile_parent_bridge_accumulator_by_set.values():
            operations.archive_or_delete_binary_volume_storage(
                accumulator,
                keep_temp=bool(inputs.keep_temp_artifacts),
                workers=int(inputs.tile_slice_workers),
                desc="remaining parent-bridge tile category accumulator",
            )
            retired_tile_accumulators += 1
        self.tile_parent_bridge_accumulator_by_set.clear()

        for volume in self.baseline_union_by_model_view.values():
            operations.close_memmap_array(volume)
            close_memmap_calls += 1
        for volume in self.baseline_confmap_by_model_view.values():
            operations.close_memmap_array(volume)
            close_memmap_calls += 1
        for _model_name, model in self.yolo_models:
            if model is not None:
                operations.unload_yolo_model(model)
                unloaded_models += 1
        if processing_volume_was_distinct:
            operations.close_memmap_array(self.volume_rgb)
            close_memmap_calls += 1
        operations.close_memmap_array(self.input_volume_rgb)
        close_memmap_calls += 1
        operations.trim_cuda_memory()
        operations.collect_garbage()

        self._closed = True
        return TtaOutputResult(
            close_memmap_calls=int(close_memmap_calls),
            retired_tile_accumulators=int(retired_tile_accumulators),
            unloaded_models=int(unloaded_models),
            processing_volume_was_distinct=bool(processing_volume_was_distinct),
        )


__all__ = [
    "TtaOutputArtifacts",
    "TtaOutputInputs",
    "TtaOutputOperations",
    "TtaOutputResult",
]
