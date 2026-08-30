"""Mode-aware in-plane variant expansion for v18 planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, Tuple, TypeVar

from XTA.config import resolve_tta_angles

from .contracts import InPlaneVariant, PipelineMode


PhysicalViewT = TypeVar("PhysicalViewT")


@dataclass(frozen=True)
class ResolvedViewVariant(Generic[PhysicalViewT]):
    """Physical view, declarative angle, and mode-compatible runtime view."""

    physical_view: PhysicalViewT
    in_plane_variant: InPlaneVariant
    runtime_view: PhysicalViewT


def resolve_in_plane_variants(
    mode: PipelineMode | str,
    angles: Sequence[str] | str | float | int | None = None,
) -> Tuple[InPlaneVariant, ...]:
    """Resolve TTA angles or PTA's non-configurable internal identity variant."""

    resolved_mode = PipelineMode.coerce(mode)
    if resolved_mode is PipelineMode.PTA:
        if angles is not None:
            raise ValueError("--angle is invalid in PTA mode; PTA uses an internal 0-degree identity")
        return (InPlaneVariant(0.0),)
    return tuple(InPlaneVariant(angle) for angle in resolve_tta_angles(angles))


def expand_view_variants(
    mode: PipelineMode | str,
    physical_views: Sequence[PhysicalViewT],
    angles: Sequence[str] | str | float | int | None = None,
) -> Tuple[ResolvedViewVariant[PhysicalViewT], ...]:
    """Cross physical views with mode-owned in-plane variants.

    PTA uses an identity runtime view for every selected physical view.  An empty
    PTA physical-view sequence therefore yields zero variants even though its
    internal angle sequence contains the identity.  TTA delegates runtime-view
    construction to the existing, authoritative expansion implementation.
    """

    resolved_mode = PipelineMode.coerce(mode)
    physical = tuple(physical_views)
    variants = resolve_in_plane_variants(resolved_mode, angles)
    if not physical:
        return ()

    pairs = tuple((view, variant) for view in physical for variant in variants)
    if resolved_mode is PipelineMode.PTA:
        return tuple(
            ResolvedViewVariant(
                physical_view=view,
                in_plane_variant=variant,
                runtime_view=view,
            )
            for view, variant in pairs
        )

    # Lazy import preserves a dependency-light unification package import while
    # retaining the existing TTA ViewInfo naming and metadata behavior verbatim.
    from XTA.geometry import expand_views_into_tta_variants

    runtime_views = tuple(
        expand_views_into_tta_variants(
            physical,
            tuple(variant.angle_deg for variant in variants),
        )
    )
    if len(runtime_views) != len(pairs):
        raise RuntimeError(
            "existing TTA expansion returned an unexpected number of runtime variants"
        )
    return tuple(
        ResolvedViewVariant(
            physical_view=view,
            in_plane_variant=variant,
            runtime_view=runtime_view,
        )
        for (view, variant), runtime_view in zip(pairs, runtime_views)
    )


__all__ = (
    "ResolvedViewVariant",
    "expand_view_variants",
    "resolve_in_plane_variants",
)
