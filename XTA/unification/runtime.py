"""Runtime bridge from v18 mode configuration to canonical TTA geometry.

This module is intentionally not imported by :mod:`XTA.unification`'s
dependency-light public surface.  It owns the one heavy physical-view compiler
used by both workflows after their mode-specific arguments have been resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from XTA.config import RadialViewRequest, TiltedViewGroup
from XTA.geometry import ViewInfo, get_view_infos, radial_target_diameter
from XTA.media import resolve_radial_azimuth_angles


@dataclass(frozen=True)
class CompiledPhysicalViews:
    """Resolved physical views plus the exact radial geometry they contain."""

    views: Tuple[ViewInfo, ...]
    radial_targets: Tuple[str, ...]
    radial_diameters: Tuple[int, ...]
    radial_azimuth_angles: Tuple[float, ...]


def compile_physical_views(
    *,
    t_dim: int,
    height: int,
    width: int,
    cartesian_views: Sequence[str],
    radial_requests: Sequence[RadialViewRequest],
    tilted_groups: Sequence[TiltedViewGroup],
    radial_native_raster: int = 0,
) -> CompiledPhysicalViews:
    """Compile grouped view requests through the authoritative TTA geometry.

    The exact generated radial angle vectors live on the returned ``ViewInfo``
    objects.  The paired spacing and diameter tuples are retained separately so
    both modes can record the same planning facts in their manifests.
    """

    radial_targets = tuple(str(request.view) for request in radial_requests)
    radial_diameters = tuple(
        int(radial_target_diameter(target, int(t_dim), int(height), int(width)))
        for target in radial_targets
    )
    radial_angles = tuple(
        float(value)
        for value in resolve_radial_azimuth_angles(
            tuple(radial_requests),
            diameters=radial_diameters,
        )
    )
    views = tuple(
        get_view_infos(
            T=int(t_dim),
            H=int(height),
            W=int(width),
            cartesian_views=tuple(str(value) for value in cartesian_views),
            radial_views=radial_targets,
            radial_azimuth_angles=radial_angles,
            tilt_groups=tuple(tilted_groups),
            radial_native_raster=int(radial_native_raster),
        )
    )
    return CompiledPhysicalViews(
        views=views,
        radial_targets=radial_targets,
        radial_diameters=radial_diameters,
        radial_azimuth_angles=radial_angles,
    )


__all__ = ("CompiledPhysicalViews", "compile_physical_views")
