"""Dependency-light v18 contracts shared by PTA and TTA planning.

This module intentionally contains no NumPy, OpenCV, model-runtime, filesystem,
or scheduler imports.  The contracts describe render work; execution backends
remain responsible for turning that work into arrays or model inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence, Tuple


class PipelineMode(str, Enum):
    """Top-level v18 workflow selected by ``--mode``."""

    TTA = "tta"
    PTA = "pta"

    @classmethod
    def coerce(cls, value: "PipelineMode | str") -> "PipelineMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unsupported pipeline mode {value!r}; use 'tta' or 'pta'") from exc


class DataRole(str, Enum):
    """Semantic role whose sampling rules are declared by a raster plan."""

    INTENSITY = "intensity"
    CATEGORICAL_GROUND_TRUTH = "categorical_ground_truth"
    PREDICTION = "prediction"
    PRESENTATION = "presentation"

    @classmethod
    def coerce(cls, value: "DataRole | str") -> "DataRole":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(role.value for role in cls)
            raise ValueError(f"unsupported data role {value!r}; use one of: {allowed}") from exc


class _FrozenMetadata(Mapping[str, Any]):
    """Small immutable mapping that remains pickle-safe for spawn workers."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, items: Sequence[tuple[str, Any]]) -> None:
        self._items = tuple(items)
        self._lookup = dict(self._items)

    def __getitem__(self, key: str) -> Any:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self) -> tuple[object, tuple[Tuple[tuple[str, Any], ...]]]:
        return (type(self), (self._items,))


def _require_nonempty(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _require_int(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    resolved = int(value)
    if minimum is not None and resolved < int(minimum):
        raise ValueError(f"{field_name} must be >= {int(minimum)}")
    return resolved


def _normalize_angle(angle_deg: object) -> float:
    angle = float(angle_deg)
    if not math.isfinite(angle):
        raise ValueError(f"in-plane angle must be finite; got {angle_deg!r}")
    normalized = float(angle % 360.0)
    if math.isclose(normalized, 0.0, rel_tol=0.0, abs_tol=1e-9) or math.isclose(
        normalized, 360.0, rel_tol=0.0, abs_tol=1e-9
    ):
        return 0.0
    return normalized


def _format_angle_variant_id(angle_deg: float) -> str:
    token = f"{float(angle_deg):g}".replace("-", "m").replace(".", "p")
    return f"a{token}"


@dataclass(frozen=True)
class InPlaneVariant:
    """One normalized 2-D rotation applied after physical-view extraction."""

    angle_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "angle_deg", _normalize_angle(self.angle_deg))

    @property
    def variant_id(self) -> str:
        return _format_angle_variant_id(self.angle_deg)


@dataclass(frozen=True)
class ChannelLayout:
    """One canonical TTA channel format before mode-owned direction expansion."""

    token: str
    kind: str
    channel_count: int
    stride: int
    offsets: Tuple[int, ...]

    def __post_init__(self) -> None:
        token = _require_nonempty(self.token, "channel token")
        kind = _require_nonempty(self.kind, "channel kind").lower()
        channel_count = _require_int(self.channel_count, "channel_count", minimum=1)
        stride = _require_int(self.stride, "channel stride", minimum=1)
        offsets = tuple(_require_int(offset, "channel offset") for offset in self.offsets)
        if len(offsets) != channel_count:
            raise ValueError(
                f"channel_count={channel_count} does not match {len(offsets)} offset(s)"
            )
        if kind not in {"gray", "rgb", "custom"}:
            raise ValueError(f"unsupported channel kind {self.kind!r}")
        if kind == "custom" and channel_count % 2 == 0:
            raise ValueError("custom channel layouts require an odd channel_count")
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "channel_count", channel_count)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "offsets", offsets)


@dataclass(frozen=True)
class ChannelVariant:
    """A mode-expanded channel order for one canonical layout."""

    layout: ChannelLayout
    direction: str = "ascending"
    offsets: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.layout, ChannelLayout):
            raise TypeError("layout must be a ChannelLayout")
        direction = str(self.direction).strip().lower()
        if direction not in {"ascending", "reversed"}:
            raise ValueError("channel direction must be 'ascending' or 'reversed'")
        expected = (
            tuple(self.layout.offsets)
            if direction == "ascending"
            else tuple(reversed(self.layout.offsets))
        )
        offsets = (
            expected
            if not self.offsets
            else tuple(_require_int(value, "channel variant offset") for value in self.offsets)
        )
        if offsets != expected:
            raise ValueError(
                f"{direction} channel variant offsets must be {expected}, got {offsets}"
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "offsets", offsets)

    @property
    def variant_id(self) -> str:
        if self.direction == "ascending":
            return self.layout.token
        return f"{self.layout.token}_reversed"

    @property
    def is_reversed(self) -> bool:
        return self.direction == "reversed"


@dataclass(frozen=True)
class TileLayout:
    """One strict TTA-style square tile layout."""

    tile_size: int
    tile_stride: int

    def __post_init__(self) -> None:
        size = _require_int(self.tile_size, "tile_size", minimum=1)
        stride = _require_int(self.tile_stride, "tile_stride", minimum=1)
        if stride > size:
            raise ValueError("tile_stride must be <= tile_size")
        object.__setattr__(self, "tile_size", size)
        object.__setattr__(self, "tile_stride", stride)

    @property
    def layout_id(self) -> str:
        return f"tile_{self.tile_size}_{self.tile_stride}"


@dataclass(frozen=True)
class FrameAddress:
    """Resolved source frame plus the radial odd-wrap reflection state."""

    index: int
    mirror_u: bool = False

    def __post_init__(self) -> None:
        index = _require_int(self.index, "frame index", minimum=0)
        if not isinstance(self.mirror_u, bool):
            raise TypeError("mirror_u must be a boolean")
        object.__setattr__(self, "index", index)


@dataclass(frozen=True)
class BackendSamplingImplementation:
    """One backend's declared implementation of a policy for selected roles."""

    backend: str
    implementation: str
    roles: Tuple[DataRole, ...] = tuple(DataRole)
    exact: bool = False
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        backend = _require_nonempty(self.backend, "backend").lower()
        implementation = _require_nonempty(self.implementation, "implementation")
        roles = tuple(DataRole.coerce(role) for role in self.roles)
        if not roles:
            raise ValueError("backend implementation must declare at least one data role")
        if len(set(roles)) != len(roles):
            raise ValueError(f"backend {backend!r} declares a data role more than once")
        absolute_tolerance = float(self.absolute_tolerance)
        relative_tolerance = float(self.relative_tolerance)
        for value, name in (
            (absolute_tolerance, "absolute_tolerance"),
            (relative_tolerance, "relative_tolerance"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "implementation", implementation)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "exact", bool(self.exact))
        object.__setattr__(self, "absolute_tolerance", absolute_tolerance)
        object.__setattr__(self, "relative_tolerance", relative_tolerance)


def _normalize_role_rules(
    values: Sequence[tuple[DataRole | str, str]],
    field_name: str,
) -> Tuple[tuple[DataRole, str], ...]:
    normalized = tuple(
        (DataRole.coerce(role), _require_nonempty(value, field_name))
        for role, value in values
    )
    roles = tuple(role for role, _ in normalized)
    if len(set(roles)) != len(roles):
        raise ValueError(f"{field_name} declares a data role more than once")
    if not normalized:
        raise ValueError(f"{field_name} must declare at least one data role")
    return normalized


@dataclass(frozen=True)
class ForwardSamplingPolicy:
    """Canonical forward-render sampling semantics and backend declarations.

    This is deliberately distinct from TTA prediction interpolation/walk-back
    configuration.  PTA consumes this policy but exposes no interpolation flags.
    """

    policy_id: str
    coordinate_convention: str
    stage_order: Tuple[str, ...]
    role_kernels: Tuple[tuple[DataRole, str], ...]
    role_boundaries: Tuple[tuple[DataRole, str], ...]
    backend_implementations: Tuple[BackendSamplingImplementation, ...]
    policy_version: int = 1

    def __post_init__(self) -> None:
        policy_id = _require_nonempty(self.policy_id, "policy_id")
        coordinate_convention = _require_nonempty(
            self.coordinate_convention, "coordinate_convention"
        )
        stage_order = tuple(_require_nonempty(stage, "sampling stage") for stage in self.stage_order)
        if not stage_order:
            raise ValueError("stage_order must contain at least one stage")
        if len(set(stage_order)) != len(stage_order):
            raise ValueError("stage_order must not contain duplicate stages")
        role_kernels = _normalize_role_rules(self.role_kernels, "role_kernels")
        role_boundaries = _normalize_role_rules(self.role_boundaries, "role_boundaries")
        implementations = tuple(self.backend_implementations)
        if not implementations:
            raise ValueError("at least one backend sampling implementation is required")
        for implementation in implementations:
            if not isinstance(implementation, BackendSamplingImplementation):
                raise TypeError(
                    "backend_implementations must contain BackendSamplingImplementation values"
                )
        claimed: set[tuple[str, DataRole]] = set()
        for implementation in implementations:
            for role in implementation.roles:
                key = (implementation.backend, role)
                if key in claimed:
                    raise ValueError(
                        f"backend {implementation.backend!r} has multiple implementations "
                        f"for role {role.value!r}"
                    )
                claimed.add(key)
        policy_version = _require_int(self.policy_version, "policy_version", minimum=1)
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "coordinate_convention", coordinate_convention)
        object.__setattr__(self, "stage_order", stage_order)
        object.__setattr__(self, "role_kernels", role_kernels)
        object.__setattr__(self, "role_boundaries", role_boundaries)
        object.__setattr__(self, "backend_implementations", implementations)
        object.__setattr__(self, "policy_version", policy_version)

    def kernel_for(self, role: DataRole | str) -> str:
        resolved = DataRole.coerce(role)
        return dict(self.role_kernels)[resolved]

    def boundary_for(self, role: DataRole | str) -> str:
        resolved = DataRole.coerce(role)
        return dict(self.role_boundaries)[resolved]

    def implementation_for(
        self,
        backend: str,
        role: DataRole | str,
    ) -> BackendSamplingImplementation:
        backend_name = str(backend).strip().lower()
        resolved_role = DataRole.coerce(role)
        for implementation in self.backend_implementations:
            if implementation.backend == backend_name and resolved_role in implementation.roles:
                return implementation
        raise KeyError(
            f"sampling policy {self.policy_id!r} has no {backend_name!r} implementation "
            f"for role {resolved_role.value!r}"
        )

    def canonical_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "coordinate_convention": self.coordinate_convention,
            "stage_order": list(self.stage_order),
            "role_kernels": {
                role.value: kernel for role, kernel in sorted(self.role_kernels, key=lambda item: item[0].value)
            },
            "role_boundaries": {
                role.value: boundary
                for role, boundary in sorted(self.role_boundaries, key=lambda item: item[0].value)
            },
            "backend_implementations": [
                {
                    "backend": implementation.backend,
                    "implementation": implementation.implementation,
                    "roles": sorted(role.value for role in implementation.roles),
                    "exact": implementation.exact,
                    "absolute_tolerance": implementation.absolute_tolerance,
                    "relative_tolerance": implementation.relative_tolerance,
                }
                for implementation in sorted(
                    self.backend_implementations,
                    key=lambda item: (
                        item.backend,
                        item.implementation,
                        tuple(sorted(role.value for role in item.roles)),
                        item.exact,
                        item.absolute_tolerance,
                        item.relative_tolerance,
                    ),
                )
            ],
        }

    @property
    def digest(self) -> str:
        """Stable identity of the policy declaration used by plans and manifests."""

        return _digest_record(self.canonical_record())


def _freeze_metadata(value: Any, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Enum):
        return _freeze_metadata(value.value, path=path)
    if isinstance(value, Mapping):
        frozen: list[tuple[str, Any]] = []
        for key in sorted(value, key=lambda candidate: str(candidate)):
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings; got {key!r}")
            frozen.append((key, _freeze_metadata(value[key], path=f"{path}.{key}")))
        return _FrozenMetadata(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} contains unsupported value {value!r}; metadata must be JSON-compatible"
    )


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _digest_record(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RasterPlan:
    """Immutable, digest-addressed description of one two-dimensional raster."""

    mode: PipelineMode
    physical_view_id: str
    in_plane_variant: InPlaneVariant
    channel_variant: ChannelVariant
    sampling_policy: ForwardSamplingPolicy
    output_shape: Tuple[int, int]
    tile_layout: TileLayout | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "v18.raster-plan.1"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        mode = PipelineMode.coerce(self.mode)
        physical_view_id = _require_nonempty(self.physical_view_id, "physical_view_id")
        if not isinstance(self.in_plane_variant, InPlaneVariant):
            raise TypeError("in_plane_variant must be an InPlaneVariant")
        if not isinstance(self.channel_variant, ChannelVariant):
            raise TypeError("channel_variant must be a ChannelVariant")
        if not isinstance(self.sampling_policy, ForwardSamplingPolicy):
            raise TypeError("sampling_policy must be a ForwardSamplingPolicy")
        output_shape = tuple(self.output_shape)
        if len(output_shape) != 2:
            raise ValueError("output_shape must contain exactly (height, width)")
        output_shape = (
            _require_int(output_shape[0], "output height", minimum=1),
            _require_int(output_shape[1], "output width", minimum=1),
        )
        if self.tile_layout is not None and not isinstance(self.tile_layout, TileLayout):
            raise TypeError("tile_layout must be a TileLayout or None")
        metadata = _freeze_metadata(dict(self.metadata))
        schema_version = _require_nonempty(self.schema_version, "schema_version")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "physical_view_id", physical_view_id)
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "digest", _digest_record(self.canonical_record()))

    def canonical_record(self) -> dict[str, Any]:
        tile = self.tile_layout
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "physical_view_id": self.physical_view_id,
            "in_plane_variant": {
                "variant_id": self.in_plane_variant.variant_id,
                "angle_deg": self.in_plane_variant.angle_deg,
            },
            "channel_variant": {
                "layout": {
                    "token": self.channel_variant.layout.token,
                    "kind": self.channel_variant.layout.kind,
                    "channel_count": self.channel_variant.layout.channel_count,
                    "stride": self.channel_variant.layout.stride,
                    "offsets": list(self.channel_variant.layout.offsets),
                },
                "direction": self.channel_variant.direction,
                "offsets": list(self.channel_variant.offsets),
            },
            "sampling_policy": self.sampling_policy.canonical_record(),
            "output_shape": list(self.output_shape),
            "tile_layout": (
                None
                if tile is None
                else {"tile_size": tile.tile_size, "tile_stride": tile.tile_stride}
            ),
            "metadata": _thaw_metadata(self.metadata),
        }


@dataclass(frozen=True)
class RenderItem:
    """One role-specific frame request backed by a canonical raster plan."""

    plan: RasterPlan
    data_role: DataRole
    frame_address: FrameAddress
    metadata: Mapping[str, Any] = field(default_factory=dict)
    item_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, RasterPlan):
            raise TypeError("plan must be a RasterPlan")
        role = DataRole.coerce(self.data_role)
        if not isinstance(self.frame_address, FrameAddress):
            raise TypeError("frame_address must be a FrameAddress")
        metadata = _freeze_metadata(dict(self.metadata), path="render_item.metadata")
        record = {
            "plan_digest": self.plan.digest,
            "data_role": role.value,
            "frame_address": {
                "index": self.frame_address.index,
                "mirror_u": self.frame_address.mirror_u,
            },
            "metadata": _thaw_metadata(metadata),
        }
        object.__setattr__(self, "data_role", role)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "item_id", _digest_record(record))


@dataclass(frozen=True)
class RenderRequestBatch:
    """A bounded collection of logical render items; empty PTA batches are valid."""

    mode: PipelineMode
    items: Tuple[RenderItem, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    batch_id: str = field(init=False)

    def __post_init__(self) -> None:
        mode = PipelineMode.coerce(self.mode)
        items = tuple(self.items)
        for item in items:
            if not isinstance(item, RenderItem):
                raise TypeError("items must contain RenderItem values")
            if item.plan.mode is not mode:
                raise ValueError(
                    f"render item mode {item.plan.mode.value!r} does not match batch mode {mode.value!r}"
                )
        item_ids = tuple(item.item_id for item in items)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError(
                "a RenderRequestBatch must not contain duplicate logical render items"
            )
        metadata = _freeze_metadata(dict(self.metadata), path="render_batch.metadata")
        record = {
            "mode": mode.value,
            "item_ids": list(item_ids),
            "metadata": _thaw_metadata(metadata),
        }
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "batch_id", _digest_record(record))


__all__ = (
    "BackendSamplingImplementation",
    "ChannelLayout",
    "ChannelVariant",
    "DataRole",
    "ForwardSamplingPolicy",
    "FrameAddress",
    "InPlaneVariant",
    "PipelineMode",
    "RasterPlan",
    "RenderRequestBatch",
    "RenderItem",
    "TileLayout",
)
