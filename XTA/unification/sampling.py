"""Canonical v18 forward-sampling policy, plan factory, and execution binding.

The declarations in this module are consumed by production PTA and TTA render
paths.  They describe source-to-raster sampling only; TTA prediction
interpolation, backprojection, and fusion are deliberately outside this policy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

from .contracts import (
    BackendSamplingImplementation,
    ChannelLayout,
    ChannelVariant,
    DataRole,
    ForwardSamplingPolicy,
    InPlaneVariant,
    PipelineMode,
    RasterPlan,
    TileLayout,
)


@lru_cache(maxsize=1)
def forward_sampling_policy() -> ForwardSamplingPolicy:
    """Return the single policy object used by v18 built-in geometry."""

    return ForwardSamplingPolicy(
        policy_id="xta.forward_sampling",
        policy_version=18,
        coordinate_convention=(
            "gray8_t_y_x_frame_index; destination-pixel-center to source; "
            "radial [0,180) index wrap with odd-crossing radial-u mirror"
        ),
        stage_order=(
            "physical_view_extraction",
            "in_plane_affine_and_output_resize",
            "channel_addressing",
            "dense_tile_transform",
        ),
        role_kernels=(
            (
                DataRole.INTENSITY,
                "backend-registered TTA intensity sampler",
            ),
            (
                DataRole.CATEGORICAL_GROUND_TRUTH,
                "nearest categorical taps; tilted stack blend threshold >=0.5",
            ),
        ),
        role_boundaries=(
            (
                DataRole.INTENSITY,
                "TTA family-specific boundary; radial seam index-wrap+mirror-u",
            ),
            (
                DataRole.CATEGORICAL_GROUND_TRUTH,
                "same coordinate boundary with categorical outside value zero",
            ),
        ),
        backend_implementations=(
            BackendSamplingImplementation(
                backend="cpu",
                implementation=(
                    "XTA.geometry.render_intensity_frame_on_grid@v18; "
                    "OpenCV linear affine/resize and TTA hardware-linear radial tables"
                ),
                roles=(DataRole.INTENSITY,),
                exact=True,
            ),
            BackendSamplingImplementation(
                backend="cpu",
                implementation=(
                    "XTA.geometry.render_categorical_frame_on_grid@v18; "
                    "nearest categorical sampling"
                ),
                roles=(DataRole.CATEGORICAL_GROUND_TRUTH,),
                exact=True,
            ),
            BackendSamplingImplementation(
                backend="cuda",
                implementation=(
                    "XTA.cuda_backend TTA resident/direct-ring forward renderer@v18"
                ),
                roles=(DataRole.INTENSITY,),
                exact=False,
                absolute_tolerance=1.0,
                relative_tolerance=0.0,
            ),
        ),
    )


def require_forward_sampling(
    backend: str,
    role: DataRole | str,
) -> BackendSamplingImplementation:
    """Resolve a registered implementation or fail instead of substituting one."""

    return forward_sampling_policy().implementation_for(backend, role)


def forward_sampling_execution_record(
    bindings: Sequence[tuple[str, DataRole | str]],
) -> dict[str, Any]:
    """Serialize the policy and exact backend-role implementations selected."""

    policy = forward_sampling_policy()
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for backend, role in bindings:
        resolved_role = DataRole.coerce(role)
        implementation = require_forward_sampling(backend, resolved_role)
        key = (implementation.backend, resolved_role.value)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "backend": implementation.backend,
                "data_role": resolved_role.value,
                "implementation": implementation.implementation,
                "exact_same_backend_cross_mode": bool(implementation.exact),
                "absolute_tolerance": float(implementation.absolute_tolerance),
                "relative_tolerance": float(implementation.relative_tolerance),
            }
        )
    return {
        "policy": policy.canonical_record(),
        "policy_digest": policy.digest,
        "selected_implementations": selected,
    }


def build_forward_raster_plan(
    *,
    mode: PipelineMode | str,
    physical_view_id: str,
    angle_deg: float,
    channel_token: str,
    channel_kind: str,
    channel_count: int,
    channel_stride: int,
    channel_offsets: Sequence[int],
    channel_direction: str,
    output_shape: tuple[int, int],
    tile_size: int | None = None,
    tile_stride: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RasterPlan:
    """Construct the canonical digest-addressed plan used by a runtime job."""

    resolved_offsets = tuple(int(value) for value in channel_offsets)
    direction = str(channel_direction).strip().lower()
    if direction in {"forward", "ascending"}:
        direction = "ascending"
    elif direction in {"reverse", "reversed"}:
        direction = "reversed"
    layout_offsets = (
        resolved_offsets
        if direction == "ascending"
        else tuple(reversed(resolved_offsets))
    )
    layout = ChannelLayout(
        token=str(channel_token),
        kind=str(channel_kind),
        channel_count=int(channel_count),
        stride=int(channel_stride),
        offsets=layout_offsets,
    )
    variant = ChannelVariant(
        layout=layout,
        direction=direction,
        offsets=resolved_offsets,
    )
    if (tile_size is None) != (tile_stride is None):
        raise ValueError("tile_size and tile_stride must either both be set or both be absent")
    tile = (
        None
        if tile_size is None
        else TileLayout(int(tile_size), int(tile_stride))
    )
    return RasterPlan(
        mode=PipelineMode.coerce(mode),
        physical_view_id=str(physical_view_id),
        in_plane_variant=InPlaneVariant(float(angle_deg)),
        channel_variant=variant,
        sampling_policy=forward_sampling_policy(),
        output_shape=(int(output_shape[0]), int(output_shape[1])),
        tile_layout=tile,
        metadata=dict(metadata or {}),
    )


_RASTER_PLAN_SPAWN_SPEC_SCHEMA = "v18.raster-plan.spawn-spec.1"


def raster_plan_spawn_spec(plan: RasterPlan) -> dict[str, Any]:
    """Return a primitive-only, spawn-safe description of one raster plan.

    Worker task dictionaries cross a multiprocessing ``spawn`` boundary.  The
    canonical record keeps that payload independent of class-pickling details,
    while the two digests make worker-side reconstruction fail closed if either
    the task metadata or the active sampling policy drifts.
    """

    if not isinstance(plan, RasterPlan):
        raise TypeError("raster_plan_spawn_spec requires a RasterPlan")
    return {
        "schema_version": _RASTER_PLAN_SPAWN_SPEC_SCHEMA,
        "plan_digest": str(plan.digest),
        "sampling_policy_digest": str(plan.sampling_policy.digest),
        "plan": plan.canonical_record(),
    }


def raster_plan_from_spawn_spec(spec: Mapping[str, Any]) -> RasterPlan:
    """Reconstruct and verify a canonical plan inside a spawned worker."""

    if not isinstance(spec, Mapping):
        raise TypeError("raster plan spawn spec must be a mapping")
    if str(spec.get("schema_version", "")) != _RASTER_PLAN_SPAWN_SPEC_SCHEMA:
        raise ValueError(
            "unsupported raster plan spawn spec schema: "
            f"{spec.get('schema_version')!r}"
        )
    record = spec.get("plan")
    if not isinstance(record, Mapping):
        raise ValueError("raster plan spawn spec is missing its canonical plan record")

    policy = forward_sampling_policy()
    expected_policy_digest = str(spec.get("sampling_policy_digest", ""))
    if expected_policy_digest != str(policy.digest):
        raise RuntimeError(
            "spawned worker sampling-policy drift: "
            f"task={expected_policy_digest!r}, current={policy.digest!r}"
        )
    if record.get("sampling_policy") != policy.canonical_record():
        raise RuntimeError(
            "spawned worker canonical sampling-policy record does not match the active policy"
        )

    try:
        in_plane = record["in_plane_variant"]
        channel = record["channel_variant"]
        layout = channel["layout"]
        output_shape = tuple(int(value) for value in record["output_shape"])
        tile_record = record.get("tile_layout")
        if tile_record is not None and not isinstance(tile_record, Mapping):
            raise TypeError("tile_layout must be a mapping or null")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        rebuilt = build_forward_raster_plan(
            mode=str(record["mode"]),
            physical_view_id=str(record["physical_view_id"]),
            angle_deg=float(in_plane["angle_deg"]),
            channel_token=str(layout["token"]),
            channel_kind=str(layout["kind"]),
            channel_count=int(layout["channel_count"]),
            channel_stride=int(layout["stride"]),
            channel_offsets=tuple(int(value) for value in channel["offsets"]),
            channel_direction=str(channel["direction"]),
            output_shape=(int(output_shape[0]), int(output_shape[1])),
            tile_size=(
                None if tile_record is None else int(tile_record["tile_size"])
            ),
            tile_stride=(
                None if tile_record is None else int(tile_record["tile_stride"])
            ),
            metadata=dict(metadata),
        )
    except Exception as exc:
        raise ValueError("invalid canonical raster plan in spawn spec") from exc

    if str(record.get("schema_version", "")) != str(rebuilt.schema_version):
        raise RuntimeError(
            "spawned worker raster-plan schema drift: "
            f"task={record.get('schema_version')!r}, current={rebuilt.schema_version!r}"
        )
    if rebuilt.canonical_record() != dict(record):
        raise RuntimeError(
            "spawned worker raster-plan reconstruction does not match the task record"
        )
    expected_plan_digest = str(spec.get("plan_digest", ""))
    if str(rebuilt.digest) != expected_plan_digest:
        raise RuntimeError(
            "spawned worker raster-plan digest mismatch: "
            f"task={expected_plan_digest!r}, rebuilt={rebuilt.digest!r}"
        )
    return rebuilt


__all__ = (
    "build_forward_raster_plan",
    "forward_sampling_execution_record",
    "forward_sampling_policy",
    "raster_plan_from_spawn_spec",
    "raster_plan_spawn_spec",
    "require_forward_sampling",
)
