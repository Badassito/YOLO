"""Role-aware output primitives for the v19 LTA prototype.

The model and geometry layers produce native-space binary component volumes.
This module owns their TTA-compatible recomposition operations and the final
atomic manifest publication.  NumPy is imported only when volume composition is
actually requested so CLI/help imports remain lightweight.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence


LTA_MANIFEST_SCHEMA = "xta.lta.v19.manifest.1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LtaArtifactReceipt:
    """One settled artifact proven before complete-manifest publication."""

    name: str
    path: Path
    sha256: str

    def validate(self) -> dict[str, object]:
        name = str(self.name).strip()
        path = Path(self.path).resolve(strict=True)
        expected = str(self.sha256).strip().lower()
        if not name:
            raise ValueError("artifact receipt name must not be empty")
        if not path.is_file():
            raise ValueError(f"settled LTA artifact is not a file: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"LTA artifact digest changed for {name}: expected={expected}, actual={actual}"
            )
        return {"name": name, "path": str(path), "sha256": actual}


@dataclass(frozen=True)
class LtaPublicationReceipt:
    """Fail-closed evidence required to claim one LTA publication complete."""

    artifacts: Sequence[LtaArtifactReceipt]
    terminal_union_shape_tyx: tuple[int, int, int]
    terminal_union_foreground_voxels: int
    source_revalidated: bool
    model_revalidated: bool
    layers_settled: bool

    def validate(self) -> dict[str, object]:
        shape = tuple(int(value) for value in self.terminal_union_shape_tyx)
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError("terminal_union_shape_tyx must contain three positive dimensions")
        foreground = int(self.terminal_union_foreground_voxels)
        if foreground < 0 or foreground > shape[0] * shape[1] * shape[2]:
            raise ValueError("terminal union foreground count is outside its shape")
        if not self.source_revalidated or not self.model_revalidated or not self.layers_settled:
            raise ValueError(
                "complete LTA publication requires revalidated source/model and settled layers"
            )
        artifact_records = [artifact.validate() for artifact in tuple(self.artifacts)]
        names = [str(record["name"]) for record in artifact_records]
        if len(set(names)) != len(names):
            raise ValueError("complete LTA publication contains duplicate artifact names")
        return {
            "source_revalidated": True,
            "model_revalidated": True,
            "layers_settled": True,
            "terminal_union_shape_tyx": list(shape),
            "terminal_union_foreground_voxels": foreground,
            "artifacts": artifact_records,
        }


class LtaRecompositionOp(str, Enum):
    """Operation used by TTA/LTA NRRD layer manifests."""

    UNION = "union"
    SELECT = "select"
    SUBTRACT_FROM_PREVIOUS_CHECKPOINT = "subtract_from_previous_checkpoint"
    NONE = "none"

    @classmethod
    def coerce(cls, value: "LtaRecompositionOp | str") -> "LtaRecompositionOp":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unsupported LTA recomposition operation {value!r}; use one of: {allowed}"
            ) from exc


@dataclass(frozen=True)
class LtaLayerRecord:
    """One native-space role layer and its recomposition provenance."""

    layer_id: str
    recomposition_op: LtaRecompositionOp | str
    source_role: str
    volume: object
    physical_view_id: Optional[str] = None
    runtime_view_id: Optional[str] = None
    tta_angle_deg: Optional[float] = None
    interpolation_pass: int = 0
    interpolation_candidate: int = 0
    interpolation_walkback: int = 0
    tile_config_id: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        layer_id = str(self.layer_id).strip()
        source_role = str(self.source_role).strip()
        if not layer_id:
            raise ValueError("layer_id must not be empty")
        if not source_role:
            raise ValueError("source_role must not be empty")
        if self.volume is None:
            raise ValueError("layer volume must not be None")
        for field_name in (
            "interpolation_pass",
            "interpolation_candidate",
            "interpolation_walkback",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{field_name} must be >= 0")
            object.__setattr__(self, field_name, int(value))
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "source_role", source_role)
        object.__setattr__(
            self, "recomposition_op", LtaRecompositionOp.coerce(self.recomposition_op)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def manifest_record(self, *, include_volume_shape: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "layer_id": self.layer_id,
            "recomposition_op": self.recomposition_op.value,
            "source_role": self.source_role,
            "physical_view_id": self.physical_view_id,
            "runtime_view_id": self.runtime_view_id,
            "tta_angle_deg": self.tta_angle_deg,
            "interpolation_pass": int(self.interpolation_pass),
            "interpolation_candidate": int(self.interpolation_candidate),
            "interpolation_walkback": int(self.interpolation_walkback),
            "tile_config_id": self.tile_config_id,
            "metadata": dict(self.metadata),
        }
        if include_volume_shape:
            shape = getattr(self.volume, "shape", None)
            record["shape_tyx"] = (
                None if shape is None else [int(value) for value in tuple(shape)]
            )
        return record


def compose_terminal_union(layers: Sequence[LtaLayerRecord]) -> object:
    """Compose ordered binary layers into the always-created terminal union.

    ``union`` ORs an additive layer, ``select`` replaces the current complete
    checkpoint, ``subtract_from_previous_checkpoint`` clears the supplied mask,
    and ``none`` is diagnostic-only.  The returned volume is a new contiguous
    uint8 array and never aliases a component layer.
    """

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - required project dependency
        raise RuntimeError("NumPy is required to compose LTA role layers") from exc

    current = None
    expected_shape = None
    seen_ids: set[str] = set()
    for layer in tuple(layers):
        if not isinstance(layer, LtaLayerRecord):
            raise TypeError("layers must contain LtaLayerRecord values")
        if layer.layer_id in seen_ids:
            raise ValueError(f"duplicate LTA layer_id {layer.layer_id!r}")
        seen_ids.add(layer.layer_id)
        op = LtaRecompositionOp.coerce(layer.recomposition_op)
        if op is LtaRecompositionOp.NONE:
            continue
        raw = np.asarray(layer.volume)
        if raw.ndim != 3:
            raise ValueError(
                f"LTA layer {layer.layer_id!r} must have 3D (t,Y,X) shape; got {raw.shape}"
            )
        binary = np.ascontiguousarray(raw != 0, dtype=np.uint8)
        shape = tuple(int(value) for value in binary.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"LTA layer {layer.layer_id!r} shape {shape} does not match {expected_shape}"
            )

        if op is LtaRecompositionOp.SELECT:
            current = binary.copy()
        elif op is LtaRecompositionOp.UNION:
            if current is None:
                current = np.zeros(shape, dtype=np.uint8)
            np.bitwise_or(current, binary, out=current)
        elif op is LtaRecompositionOp.SUBTRACT_FROM_PREVIOUS_CHECKPOINT:
            if current is None:
                raise ValueError(
                    f"LTA layer {layer.layer_id!r} cannot subtract before a union/checkpoint"
                )
            current[binary != 0] = np.uint8(0)
        else:  # pragma: no cover - enum exhaustiveness
            raise AssertionError(op)

    if current is None:
        raise ValueError("at least one composable LTA layer is required")
    return np.ascontiguousarray(current, dtype=np.uint8)


def write_json_atomically(path: str | Path, payload: object) -> Path:
    """Write one JSON artifact completely before its public replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".assembling",
        dir=str(destination.parent),
    )
    stage_path = Path(stage_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage_path, destination)
    except BaseException:
        try:
            stage_path.unlink(missing_ok=True)
        finally:
            raise
    return destination


def write_complete_lta_manifest(
    path: str | Path,
    *,
    version: str,
    command: Sequence[str],
    layers: Sequence[LtaLayerRecord],
    publication_receipt: LtaPublicationReceipt,
    payload: Mapping[str, object],
) -> Path:
    """Publish the mandatory complete LTA manifest last."""

    if not isinstance(publication_receipt, LtaPublicationReceipt):
        raise TypeError("publication_receipt must be an LtaPublicationReceipt")
    integrity = publication_receipt.validate()
    layer_records = [layer.manifest_record() for layer in layers]
    manifest = {
        "schema": LTA_MANIFEST_SCHEMA,
        "status": "complete",
        "mode": "lta",
        "pipeline_version": str(version),
        "command": [str(value) for value in command],
        "layers": layer_records,
        "publication_integrity": integrity,
        **dict(payload),
    }
    # Fixed ownership fields cannot be replaced by caller payload.
    manifest.update(
        {
            "schema": LTA_MANIFEST_SCHEMA,
            "status": "complete",
            "mode": "lta",
            "pipeline_version": str(version),
            "command": [str(value) for value in command],
            "layers": layer_records,
            "publication_integrity": integrity,
        }
    )
    return write_json_atomically(path, manifest)


__all__ = (
    "LTA_MANIFEST_SCHEMA",
    "LtaArtifactReceipt",
    "LtaLayerRecord",
    "LtaPublicationReceipt",
    "LtaRecompositionOp",
    "compose_terminal_union",
    "write_complete_lta_manifest",
    "write_json_atomically",
)
