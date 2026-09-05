"""LTA mask normalization and terminal-volume finalization.

The module keeps three operations deliberately distinct:

* two-dimensional prediction/seed hole filling happens before a mask can be
  reused as temporal or spatial dogfood;
* completed runtime-view volumes receive the same final slice-local hole fill
  used by TTA before native backprojection;
* optional three-dimensional postprocessing applies only to the settled native
  terminal union.

Heavy TTA implementation modules are imported only when the default operations
are resolved.  This keeps the LTA command/configuration import surface light and
lets CPU tests inject small deterministic operations.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .lta_outputs import (
    LtaArtifactReceipt,
    LtaLayerRecord,
    LtaRecompositionOp,
)


_DEFAULT_RESERVE_BYTES = 16 * 1024**3
_HOLE_FILL_ALGORITHM = "enclosed_2d_background_components"
_HOLE_FILL_BACKGROUND_CONNECTIVITY = 4


def _resolved_void_fill_connectivity() -> int:
    raw = os.environ.get("YOLO_TTA_VOIDFILL_CONNECTIVITY", "").strip()
    connectivity = 6
    if raw:
        try:
            connectivity = int(raw)
        except Exception:
            connectivity = 6
    if connectivity not in (6, 18, 26):
        raise ValueError("3D void fill connectivity must be one of 6, 18, or 26")
    return connectivity


@dataclass(frozen=True)
class LtaMaskHoleFillReceipt:
    """Provenance for one non-mutating two-dimensional hole fill."""

    provenance_kind: str
    shape_hw: tuple[int, int]
    input_foreground_pixels: int
    output_foreground_pixels: int
    added_pixels: int
    mask_index: Optional[int] = None
    context: Mapping[str, object] = field(default_factory=dict)
    algorithm: str = _HOLE_FILL_ALGORITHM
    background_connectivity: int = _HOLE_FILL_BACKGROUND_CONNECTIVITY

    def __post_init__(self) -> None:
        kind = str(self.provenance_kind).strip()
        shape = tuple(int(value) for value in self.shape_hw)
        before = int(self.input_foreground_pixels)
        after = int(self.output_foreground_pixels)
        added = int(self.added_pixels)
        if not kind:
            raise ValueError("hole-fill provenance_kind must not be empty")
        if len(shape) != 2 or any(value < 1 for value in shape):
            raise ValueError("hole-fill shape_hw must contain two positive dimensions")
        if before < 0 or after < before or added != after - before:
            raise ValueError("hole-fill foreground counts are inconsistent")
        if int(self.background_connectivity) != 4:
            raise ValueError("LTA 2D hole fill requires 4-connected background")
        object.__setattr__(self, "provenance_kind", kind)
        object.__setattr__(self, "shape_hw", shape)
        object.__setattr__(self, "input_foreground_pixels", before)
        object.__setattr__(self, "output_foreground_pixels", after)
        object.__setattr__(self, "added_pixels", added)
        object.__setattr__(self, "context", dict(self.context))
        if self.mask_index is not None:
            if isinstance(self.mask_index, bool) or int(self.mask_index) < 0:
                raise ValueError("mask_index must be non-negative when supplied")
            object.__setattr__(self, "mask_index", int(self.mask_index))

    @property
    def changed(self) -> bool:
        return bool(self.added_pixels)

    def manifest_record(self) -> dict[str, object]:
        return {
            "provenance_kind": self.provenance_kind,
            "shape_hw": list(self.shape_hw),
            "input_foreground_pixels": self.input_foreground_pixels,
            "output_foreground_pixels": self.output_foreground_pixels,
            "added_pixels": self.added_pixels,
            "changed": self.changed,
            "mask_index": self.mask_index,
            "algorithm": self.algorithm,
            "background_connectivity": self.background_connectivity,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class LtaFilledMask:
    """One filled mask paired with its immutable provenance receipt."""

    mask: np.ndarray = field(repr=False, compare=False)
    receipt: LtaMaskHoleFillReceipt

    def __post_init__(self) -> None:
        mask = np.ascontiguousarray(np.asarray(self.mask), dtype=np.bool_).copy()
        if mask.ndim != 2 or tuple(int(value) for value in mask.shape) != self.receipt.shape_hw:
            raise ValueError("filled mask shape does not match its receipt")
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class LtaFilledMaskBatch:
    """A stable object-order dogfood batch and per-mask provenance."""

    masks: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    receipts: tuple[LtaMaskHoleFillReceipt, ...]

    def __post_init__(self) -> None:
        masks = tuple(
            np.ascontiguousarray(np.asarray(mask), dtype=np.bool_).copy()
            for mask in self.masks
        )
        receipts = tuple(self.receipts)
        if not masks:
            raise ValueError("a filled dogfood batch requires at least one mask")
        if len(masks) != len(receipts):
            raise ValueError("filled dogfood masks and receipts must have equal length")
        for mask, receipt in zip(masks, receipts):
            if mask.ndim != 2 or tuple(int(value) for value in mask.shape) != receipt.shape_hw:
                raise ValueError("filled dogfood mask shape does not match its receipt")
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "receipts", receipts)

    def manifest_records(self) -> tuple[dict[str, object], ...]:
        return tuple(receipt.manifest_record() for receipt in self.receipts)


@dataclass(frozen=True)
class LtaCompletedViewFillReceipt:
    """Summary of the in-place post-stitch runtime-view hole fill."""

    runtime_view_id: str
    shape_tyx: tuple[int, int, int]
    input_foreground_pixels: int
    output_foreground_pixels: int
    added_pixels: int
    algorithm: str = _HOLE_FILL_ALGORITHM
    background_connectivity: int = _HOLE_FILL_BACKGROUND_CONNECTIVITY

    def manifest_record(self) -> dict[str, object]:
        return {
            "runtime_view_id": self.runtime_view_id,
            "shape_tyx": list(self.shape_tyx),
            "input_foreground_pixels": int(self.input_foreground_pixels),
            "output_foreground_pixels": int(self.output_foreground_pixels),
            "added_pixels": int(self.added_pixels),
            "changed": bool(self.added_pixels),
            "algorithm": self.algorithm,
            "background_connectivity": int(self.background_connectivity),
        }


def _default_binary_hole_fill(mask_bool: np.ndarray) -> np.ndarray:
    # Function-local import preserves the dependency-light LTA import boundary.
    from .inference import _fill_holes_2d_opencv

    return np.asarray(_fill_holes_2d_opencv(mask_bool), dtype=np.bool_)


def fill_binary_mask_holes_2d(
    mask: object,
    *,
    operation: Optional[Callable[[np.ndarray], object]] = None,
) -> np.ndarray:
    """Return a filled contiguous bool copy without mutating ``mask``.

    Enclosed background uses the established TTA/SciPy-compatible 4-connected
    definition.  The injected operation receives its own copy so even a
    mutating implementation cannot alter the caller's authoritative mask.
    """

    source = np.asarray(mask)
    if source.ndim != 2 or any(int(value) < 1 for value in source.shape):
        raise ValueError(f"LTA hole fill requires a nonempty 2D mask; got {source.shape}")
    binary = np.ascontiguousarray(source != 0, dtype=np.bool_)
    fill = _default_binary_hole_fill if operation is None else operation
    filled = np.ascontiguousarray(
        np.asarray(fill(binary.copy()), dtype=np.bool_),
        dtype=np.bool_,
    )
    if filled.shape != binary.shape:
        raise ValueError(
            f"LTA hole fill changed mask shape from {binary.shape} to {filled.shape}"
        )
    if bool(np.any(binary & ~filled)):
        raise ValueError("LTA hole fill removed foreground pixels")
    return filled.copy()


def _fill_mask_with_receipt(
    mask: object,
    *,
    provenance_kind: str,
    mask_index: Optional[int] = None,
    context: Optional[Mapping[str, object]] = None,
    operation: Optional[Callable[[np.ndarray], object]] = None,
) -> LtaFilledMask:
    source = np.asarray(mask)
    filled = fill_binary_mask_holes_2d(source, operation=operation)
    before = int(np.count_nonzero(source))
    after = int(np.count_nonzero(filled))
    receipt = LtaMaskHoleFillReceipt(
        provenance_kind=str(provenance_kind),
        shape_hw=tuple(int(value) for value in filled.shape),
        input_foreground_pixels=before,
        output_foreground_pixels=after,
        added_pixels=after - before,
        mask_index=mask_index,
        context=dict(context or {}),
    )
    return LtaFilledMask(mask=filled, receipt=receipt)


def fill_prediction_mask_holes_2d(
    mask: object,
    *,
    context: Optional[Mapping[str, object]] = None,
    operation: Optional[Callable[[np.ndarray], object]] = None,
) -> LtaFilledMask:
    """Fill one SAM prediction before storage, tracklets, or relay planning."""

    return _fill_mask_with_receipt(
        mask,
        provenance_kind="sam_prediction",
        context=context,
        operation=operation,
    )


def fill_dogfood_seed_masks_holes_2d(
    masks: Sequence[object],
    *,
    provenance_kind: str = "temporal_dogfood",
    context: Optional[Mapping[str, object]] = None,
    operation: Optional[Callable[[np.ndarray], object]] = None,
) -> LtaFilledMaskBatch:
    """Fill every object mask immediately before a dogfood SAM injection."""

    fill = _default_binary_hole_fill if operation is None else operation
    filled = tuple(
        _fill_mask_with_receipt(
            mask,
            provenance_kind=provenance_kind,
            mask_index=index,
            context=context,
            operation=fill,
        )
        for index, mask in enumerate(tuple(masks))
    )
    return LtaFilledMaskBatch(
        masks=tuple(item.mask for item in filled),
        receipts=tuple(item.receipt for item in filled),
    )


def fill_merged_seed_mask_holes_2d(
    mask: object,
    *,
    context: Optional[Mapping[str, object]] = None,
    operation: Optional[Callable[[np.ndarray], object]] = None,
) -> LtaFilledMask:
    """Fill a merged spatial seed after union and before SAM injection."""

    return _fill_mask_with_receipt(
        mask,
        provenance_kind="merged_spatial_relay",
        context=context,
        operation=operation,
    )


def fill_completed_view_holes_2d_inplace(
    mask_volume_u8: np.ndarray,
    *,
    runtime_view_id: str,
    workers: int = 1,
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    operation: Optional[Callable[..., object]] = None,
) -> LtaCompletedViewFillReceipt:
    """Apply TTA's completed-view 2D fill before native backprojection."""

    view_id = str(runtime_view_id).strip()
    if not view_id:
        raise ValueError("runtime_view_id must not be empty")
    volume_array = np.asarray(mask_volume_u8)
    if volume_array.ndim != 3 or any(int(value) < 1 for value in volume_array.shape):
        raise ValueError(
            f"completed LTA view must be nonempty 3D TYX; got {volume_array.shape}"
        )
    if volume_array.dtype != np.dtype(np.uint8):
        raise TypeError(
            "completed-view hole fill requires mutable uint8 storage; "
            f"got {volume_array.dtype}"
        )
    if not bool(volume_array.flags.writeable):
        raise ValueError("completed-view hole fill requires writable storage")
    if operation is None:
        from .inference import fill_view_volume_holes_2d_inplace

        operation = fill_view_volume_holes_2d_inplace
    before = int(np.count_nonzero(volume_array))
    operation(
        mask_volume_u8,
        workers=max(1, int(workers)),
        desc=f"LTA 2D hole fill ({view_id})",
        known_slice_any=known_slice_any,
        known_slice_bboxes=known_slice_bboxes,
    )
    after = int(np.count_nonzero(volume_array))
    if after < before:
        raise RuntimeError("completed-view hole fill removed foreground pixels")
    return LtaCompletedViewFillReceipt(
        runtime_view_id=view_id,
        shape_tyx=tuple(int(value) for value in volume_array.shape),
        input_foreground_pixels=before,
        output_foreground_pixels=after,
        added_pixels=after - before,
    )


@dataclass(frozen=True)
class LtaFinalizationOperations:
    """Injectable operations used by the terminal-volume transaction."""

    allocate_workspace_array: Callable[..., np.ndarray]
    close_volume: Callable[[object], object]
    fill_3d_voids: Callable[..., object]
    apply_gaussian_smoothing: Callable[..., Mapping[str, object]]
    try_gpu_keep_objects: Callable[..., object]
    apply_cpu_keep_objects: Callable[..., Mapping[str, object]]

    @classmethod
    def defaults(cls) -> "LtaFinalizationOperations":
        """Resolve established TTA implementations only when execution begins."""

        from .assembly import apply_gaussian_smoothing_inplace
        from .cuda_finalization import try_apply_keep_largest_objects_multi_gpu
        from .finalization import apply_keep_largest_objects_inplace
        from .runtime import allocate_workspace_array, close_memmap_array
        from .topology import fill_3d_voids_inplace_streaming

        return cls(
            allocate_workspace_array=allocate_workspace_array,
            close_volume=close_memmap_array,
            fill_3d_voids=fill_3d_voids_inplace_streaming,
            apply_gaussian_smoothing=apply_gaussian_smoothing_inplace,
            try_gpu_keep_objects=try_apply_keep_largest_objects_multi_gpu,
            apply_cpu_keep_objects=apply_keep_largest_objects_inplace,
        )


@dataclass(frozen=True)
class LtaResolvedPostprocessing:
    """Validated terminal postprocessing values accepted from config or a plan."""

    keep_objects: int = 0
    enable_3d_void_fill: bool = False
    gaussian_smoothing_enabled: bool = False
    gaussian_sigma: float = 3.0
    gaussian_passes: int = 1

    @classmethod
    def coerce(cls, value: object | None) -> "LtaResolvedPostprocessing":
        if value is None:
            return cls()

        def _get(name: str, default: object) -> object:
            if isinstance(value, Mapping):
                return value.get(name, default)
            return getattr(value, name, default)

        keep_objects = int(_get("keep_objects", 0))
        enable_void = bool(_get("enable_3d_void_fill", False))
        gaussian_enabled = bool(_get("gaussian_smoothing_enabled", False))
        gaussian_sigma = float(_get("gaussian_sigma", 3.0))
        gaussian_passes = int(_get("gaussian_passes", 1))
        if keep_objects < 0:
            raise ValueError("keep_objects must be non-negative")
        if not math.isfinite(gaussian_sigma) or gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be finite and positive")
        if gaussian_passes < 1:
            raise ValueError("gaussian_passes must be positive")
        return cls(
            keep_objects=keep_objects,
            enable_3d_void_fill=enable_void,
            gaussian_smoothing_enabled=gaussian_enabled,
            gaussian_sigma=gaussian_sigma,
            gaussian_passes=gaussian_passes,
        )

    def manifest_record(self) -> dict[str, object]:
        return {
            "keep_objects": int(self.keep_objects),
            "enable_3d_void_fill": bool(self.enable_3d_void_fill),
            "gaussian_smoothing_enabled": bool(self.gaussian_smoothing_enabled),
            "gaussian_sigma": float(self.gaussian_sigma),
            "gaussian_passes": int(self.gaussian_passes),
        }


@dataclass
class LtaFinalizationResult:
    """Single owner of the settled postprocessed native terminal union."""

    terminal_union: np.ndarray = field(repr=False)
    requested_workspace_path: Path
    composition_layer_ids: tuple[str, ...]
    postprocessing: Mapping[str, object]
    terminal_union_foreground_voxels: int
    _close_volume: Callable[[object], object] = field(repr=False, compare=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def detach_terminal_union(self) -> np.ndarray:
        """Transfer the only volume ownership out of this result."""

        if self._closed:
            raise RuntimeError("LTA finalization result is already closed or detached")
        self._closed = True
        return self.terminal_union

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("LTA finalization result is already closed or detached")
        self._close_volume(self.terminal_union)
        self._closed = True

    def __enter__(self) -> "LtaFinalizationResult":
        if self._closed:
            raise RuntimeError("cannot enter a closed LTA finalization result")
        return self

    def __exit__(self, *_args: object) -> None:
        if not self._closed:
            self.close()


def _parallel_apply_layer(
    destination: np.ndarray,
    source: object,
    operation: LtaRecompositionOp,
    *,
    workers: int,
    layer_id: str,
) -> None:
    source_array = np.asarray(source)
    slice_count = int(destination.shape[0])

    def _apply(index: int) -> None:
        incoming = np.asarray(source_array[int(index)]) != 0
        output = destination[int(index)]
        if operation is LtaRecompositionOp.SELECT:
            output[...] = incoming
        elif operation is LtaRecompositionOp.UNION:
            np.bitwise_or(output, incoming, out=output)
        elif operation is LtaRecompositionOp.SUBTRACT_FROM_PREVIOUS_CHECKPOINT:
            output[incoming] = np.uint8(0)
        else:  # pragma: no cover - caller filters NONE and enum is exhaustive
            raise AssertionError(operation)

    if int(workers) <= 1 or slice_count <= 1:
        for index in range(slice_count):
            _apply(index)
        return
    from .runtime import choose_slice_parallel_workers, parallel_for_indices

    parallel_for_indices(
        slice_count,
        _apply,
        max_workers=choose_slice_parallel_workers(int(workers), slice_count),
        desc=f"LTA terminal union: {layer_id}",
        show_progress=False,
    )


def finalize_lta_native_union(
    layers: Sequence[LtaLayerRecord],
    *,
    workspace_path: Path,
    temp_dir: Path,
    protected_foreground_layers: Sequence[LtaLayerRecord] = (),
    postprocessing: object | None = None,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = _DEFAULT_RESERVE_BYTES,
    workers: int = 1,
    operations: Optional[LtaFinalizationOperations] = None,
) -> LtaFinalizationResult:
    """Compose native layers scalably and apply the exact TTA filter order.

    Layer payloads are consumed one TYX slice at a time, so no full-size binary
    normalization copy is created.  On success, the returned result is the sole
    owner of the final volume.  Any GPU ``keep_objects`` replacement is committed
    before the prior union is closed.  Protected foreground (for example exact
    authoritative labels) is restored after every destructive filter so it
    remains a hard-positive invariant of the published terminal union.
    """

    layer_values = tuple(layers)
    seen_ids: set[str] = set()
    expected_shape: Optional[tuple[int, int, int]] = None
    composable: list[tuple[LtaLayerRecord, LtaRecompositionOp]] = []
    initialized = False
    for layer in layer_values:
        if not isinstance(layer, LtaLayerRecord):
            raise TypeError("layers must contain LtaLayerRecord values")
        if layer.layer_id in seen_ids:
            raise ValueError(f"duplicate LTA layer_id {layer.layer_id!r}")
        seen_ids.add(layer.layer_id)
        operation = LtaRecompositionOp.coerce(layer.recomposition_op)
        if operation is LtaRecompositionOp.NONE:
            continue
        shape = tuple(int(value) for value in np.asarray(layer.volume).shape)
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError(
                f"LTA layer {layer.layer_id!r} must have positive 3D TYX shape; got {shape}"
            )
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"LTA layer {layer.layer_id!r} shape {shape} does not match {expected_shape}"
            )
        if operation is LtaRecompositionOp.SUBTRACT_FROM_PREVIOUS_CHECKPOINT and not initialized:
            raise ValueError(
                f"LTA layer {layer.layer_id!r} cannot subtract before a union/checkpoint"
            )
        initialized = True
        composable.append((layer, operation))
    if expected_shape is None or not composable:
        raise ValueError("at least one composable LTA layer is required")

    protected = tuple(protected_foreground_layers)
    protected_ids: set[str] = set()
    for layer in protected:
        if not isinstance(layer, LtaLayerRecord):
            raise TypeError("protected_foreground_layers must contain LtaLayerRecord values")
        if layer.layer_id in protected_ids:
            raise ValueError(f"duplicate protected LTA layer_id {layer.layer_id!r}")
        protected_ids.add(layer.layer_id)
        shape = tuple(int(value) for value in np.asarray(layer.volume).shape)
        if shape != expected_shape:
            raise ValueError(
                f"protected LTA layer {layer.layer_id!r} shape {shape} "
                f"does not match {expected_shape}"
            )

    resolved = LtaResolvedPostprocessing.coerce(postprocessing)
    ops = LtaFinalizationOperations.defaults() if operations is None else operations
    destination_path = Path(workspace_path)
    scratch = Path(temp_dir)
    terminal_union = ops.allocate_workspace_array(
        shape=expected_shape,
        dtype=np.uint8,
        path=destination_path,
        desc="LTA native terminal union",
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=True,
    )
    terminal_array = np.asarray(terminal_union)
    if terminal_array.dtype != np.dtype(np.uint8) or tuple(terminal_array.shape) != expected_shape:
        try:
            ops.close_volume(terminal_union)
        finally:
            raise RuntimeError("LTA terminal-union allocator returned incompatible storage")
    post_stats: dict[str, object] = {
        "requested": resolved.manifest_record(),
        "execution_order": [],
        "void_fill": None,
        "gaussian_smoothing": None,
        "keep_objects": None,
        "protected_foreground": {
            "layer_ids": [layer.layer_id for layer in protected],
            "restore_stage": "after_terminal_filters",
            "applied": bool(protected),
        },
    }
    active_volume: np.ndarray = terminal_union
    try:
        for layer, operation in composable:
            _parallel_apply_layer(
                active_volume,
                layer.volume,
                operation,
                workers=max(1, int(workers)),
                layer_id=layer.layer_id,
            )

        if resolved.enable_3d_void_fill:
            void_dir = scratch / "final_global_void_fill"
            void_dir.mkdir(parents=True, exist_ok=True)
            void_connectivity = _resolved_void_fill_connectivity()
            ops.fill_3d_voids(
                active_volume,
                void_dir / "final_union",
                keep_temp=bool(keep_temp),
                prefer_memory=bool(prefer_memory),
                reserve_bytes=int(reserve_bytes),
                connectivity=void_connectivity,
            )
            post_stats["execution_order"].append("3d_void_fill")  # type: ignore[union-attr]
            post_stats["void_fill"] = {
                "enabled": True,
                "background_connectivity": void_connectivity,
            }

        if resolved.gaussian_smoothing_enabled:
            gaussian_stats = ops.apply_gaussian_smoothing(
                active_volume,
                float(resolved.gaussian_sigma),
                int(resolved.gaussian_passes),
                scratch,
                keep_temp=bool(keep_temp),
                prefer_memory=bool(prefer_memory),
                reserve_bytes=int(reserve_bytes),
                workers=max(1, int(workers)),
                nrrd_layers=None,
                nrrd_model_name="lta",
            )
            post_stats["execution_order"].append("gaussian_smoothing")  # type: ignore[union-attr]
            post_stats["gaussian_smoothing"] = dict(gaussian_stats)

        if resolved.keep_objects > 0:
            gpu_result = ops.try_gpu_keep_objects(
                active_volume,
                int(resolved.keep_objects),
                scratch,
                keep_temp=bool(keep_temp),
            )
            if gpu_result is None:
                keep_stats = ops.apply_cpu_keep_objects(
                    active_volume,
                    int(resolved.keep_objects),
                    scratch,
                    keep_temp=bool(keep_temp),
                    prefer_memory=bool(prefer_memory),
                    reserve_bytes=int(reserve_bytes),
                    workers=max(1, int(workers)),
                )
                keep_record = dict(keep_stats)
                keep_record.setdefault("backend", "cpu")
            else:
                replacement = getattr(gpu_result, "volume", None)
                replacement_array = np.asarray(replacement)
                if (
                    replacement is None
                    or replacement_array.dtype != np.dtype(np.uint8)
                    or tuple(int(value) for value in replacement_array.shape) != expected_shape
                ):
                    if replacement is not None and replacement is not active_volume:
                        ops.close_volume(replacement)
                    raise RuntimeError("GPU keep_objects returned incompatible replacement storage")
                keep_record = dict(getattr(gpu_result, "stats", {}))
                keep_record.setdefault("backend", "multi_gpu")
                if replacement is not active_volume:
                    prior = active_volume
                    active_volume = replacement
                    ops.close_volume(prior)
            post_stats["execution_order"].append("keep_objects")  # type: ignore[union-attr]
            post_stats["keep_objects"] = keep_record

        if protected:
            for layer in protected:
                _parallel_apply_layer(
                    active_volume,
                    layer.volume,
                    LtaRecompositionOp.UNION,
                    workers=max(1, int(workers)),
                    layer_id=f"protected:{layer.layer_id}",
                )
            post_stats["execution_order"].append(  # type: ignore[union-attr]
                "restore_protected_foreground"
            )

        foreground = int(np.count_nonzero(np.asarray(active_volume)))
        return LtaFinalizationResult(
            terminal_union=active_volume,
            requested_workspace_path=destination_path,
            composition_layer_ids=tuple(layer.layer_id for layer, _operation in composable),
            postprocessing=post_stats,
            terminal_union_foreground_voxels=foreground,
            _close_volume=ops.close_volume,
        )
    except BaseException:
        ops.close_volume(active_volume)
        raise


@dataclass(frozen=True)
class LtaFinalNrrdArtifact:
    """Explicit join between the final checkpoint layer and its receipt."""

    layer: LtaLayerRecord
    receipt: LtaArtifactReceipt

    def manifest_record(self) -> dict[str, object]:
        return {
            "layer_id": self.layer.layer_id,
            "artifact_name": self.receipt.name,
            "path": str(self.receipt.path),
            "sha256": self.receipt.sha256,
            "recomposition_op": self.layer.recomposition_op.value,
            "source_role": self.layer.source_role,
            "shape_tyx": [int(value) for value in np.asarray(self.layer.volume).shape],
        }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_global_final_output_nrrd(
    output_dir: Path,
    *,
    stem: str,
    terminal_union: np.ndarray,
    model_name: str = "sam3.1",
    layer_id: str = "global_final_output",
    writer: Optional[Callable[..., Path]] = None,
) -> LtaFinalNrrdArtifact:
    """Atomically write the always-representable final union, including all-zero volumes."""

    safe_stem = str(stem).strip()
    resolved_layer_id = str(layer_id).strip()
    if not safe_stem or not resolved_layer_id:
        raise ValueError("NRRD stem and layer_id must not be empty")
    volume = np.asarray(terminal_union)
    shape = tuple(int(value) for value in volume.shape)
    if volume.ndim != 3 or any(value < 1 for value in shape):
        raise ValueError(f"final LTA NRRD requires positive 3D TYX shape; got {shape}")
    if volume.dtype != np.dtype(np.uint8):
        raise TypeError(f"final LTA NRRD requires uint8 storage; got {volume.dtype}")
    if writer is None:
        from .outputs import write_single_layer_nrrd_from_ref

        writer = write_single_layer_nrrd_from_ref
    from .interpolation import NrrdLayerRef

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{safe_stem}_Global_final_output.seg.nrrd"
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".assembling",
        dir=str(destination_dir),
    )
    os.close(descriptor)
    stage_path = Path(stage_name)
    stage_path.unlink(missing_ok=True)
    segment_name = f"{safe_stem}_Global_final_output"
    ref = NrrdLayerRef(
        key=resolved_layer_id,
        name=segment_name,
        path=stage_path.with_suffix(".live_volume"),
        shape=shape,
        dtype="uint8",
        storage_format="live_u8",
        model_name=str(model_name),
        view_name="global",
        view_family="global",
        source="global",
        mask_kind="union",
        stage="final_output_after_all_postprocessing",
        description=(
            "Final recall-oriented LTA binary union after all selected terminal "
            "postprocessing."
        ),
        layer_role="checkpoint",
        recomposition_op="select",
        low_quality_recomposition_op="select",
        mirror_low_quality=False,
        segment_extent_ijk=None,
        segment_extent_shape_tyx=shape,
        segment_extent_source="deferred_live_volume_scan",
        live_array=volume,
    )
    try:
        writer(
            ref,
            shape,
            stage_path,
            segment_name=segment_name,
        )
        if not stage_path.is_file():
            raise RuntimeError("final LTA NRRD writer did not create its staging file")
        with stage_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(stage_path, destination)
    except BaseException:
        stage_path.unlink(missing_ok=True)
        raise

    digest = _sha256_path(destination)
    artifact_name = f"{resolved_layer_id}.nrrd"
    receipt = LtaArtifactReceipt(
        name=artifact_name,
        path=destination,
        sha256=digest,
    )
    layer = LtaLayerRecord(
        layer_id=resolved_layer_id,
        recomposition_op=LtaRecompositionOp.SELECT,
        source_role="global_final_output",
        volume=volume,
        metadata={
            "artifact_name": artifact_name,
            "artifact_path": str(destination),
            "artifact_sha256": digest,
            "empty_union": not bool(np.any(volume)),
            "stage": "final_output_after_all_postprocessing",
        },
    )
    return LtaFinalNrrdArtifact(layer=layer, receipt=receipt)


__all__ = (
    "LtaCompletedViewFillReceipt",
    "LtaFilledMask",
    "LtaFilledMaskBatch",
    "LtaFinalNrrdArtifact",
    "LtaFinalizationOperations",
    "LtaFinalizationResult",
    "LtaMaskHoleFillReceipt",
    "LtaResolvedPostprocessing",
    "fill_binary_mask_holes_2d",
    "fill_completed_view_holes_2d_inplace",
    "fill_dogfood_seed_masks_holes_2d",
    "fill_merged_seed_mask_holes_2d",
    "fill_prediction_mask_holes_2d",
    "finalize_lta_native_union",
    "write_global_final_output_nrrd",
)
