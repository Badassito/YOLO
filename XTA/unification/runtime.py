"""Runtime bridge from v18 mode configuration to canonical TTA geometry.

This module is intentionally not imported by :mod:`XTA.unification`'s
dependency-light public surface.  It owns the one heavy physical-view compiler
used by both workflows after their mode-specific arguments have been resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from XTA.config import AzimuthalViewRequest, RadialViewRequest, SphericalViewRequest, TiltedViewGroup
from XTA.geometry import ViewInfo, get_view_infos, azimuthal_target_diameter
from XTA.media import resolve_azimuthal_azimuth_angles


@dataclass(frozen=True)
class CompiledPhysicalViews:
    """Resolved physical views plus the exact azimuthal geometry they contain."""

    views: Tuple[ViewInfo, ...]
    azimuthal_targets: Tuple[str, ...]
    azimuthal_diameters: Tuple[int, ...]
    azimuthal_azimuth_angles: Tuple[float, ...]
    radial_targets: Tuple[str, ...] = ()
    spherical_targets: Tuple[str, ...] = ()


def compile_physical_views(
    *,
    t_dim: int,
    height: int,
    width: int,
    cartesian_views: Sequence[str],
    azimuthal_requests: Sequence[AzimuthalViewRequest],
    tilted_groups: Sequence[TiltedViewGroup],
    azimuthal_native_raster: int = 0,
    radial_requests: Sequence[RadialViewRequest] = (),
    radial_min_radius: float | None = None,
    radial_patch_size: int = 3072,
    spherical_requests: Sequence[SphericalViewRequest] = (),
    spherical_min_radius: float | None = None,
    spherical_patch_size: int = 0,
) -> CompiledPhysicalViews:
    """Compile grouped view requests through the authoritative TTA geometry.

    The exact generated azimuthal angle vectors live on the returned ``ViewInfo``
    objects.  The paired spacing and diameter tuples are retained separately so
    both modes can record the same planning facts in their manifests.
    """

    azimuthal_targets = tuple(str(request.view) for request in azimuthal_requests)
    radial_targets = tuple(str(request.view) for request in radial_requests)
    spherical_targets = tuple(str(request.view) for request in spherical_requests)
    azimuthal_diameters = tuple(
        int(azimuthal_target_diameter(target, int(t_dim), int(height), int(width)))
        for target in azimuthal_targets
    )
    azimuthal_angles = tuple(
        float(value)
        for value in resolve_azimuthal_azimuth_angles(
            tuple(azimuthal_requests),
            diameters=azimuthal_diameters,
        )
    )
    views = tuple(
        get_view_infos(
            T=int(t_dim),
            H=int(height),
            W=int(width),
            cartesian_views=tuple(str(value) for value in cartesian_views),
            azimuthal_views=azimuthal_targets,
            azimuthal_azimuth_angles=azimuthal_angles,
            tilt_groups=tuple(tilted_groups),
            azimuthal_native_raster=int(azimuthal_native_raster),
            radial_views=radial_targets,
            radial_min_radius=radial_min_radius,
            radial_patch_size=int(radial_patch_size),
            spherical_views=spherical_targets,
            spherical_min_radius=spherical_min_radius,
            spherical_patch_size=int(spherical_patch_size),
        )
    )
    return CompiledPhysicalViews(
        views=views,
        azimuthal_targets=azimuthal_targets,
        azimuthal_diameters=azimuthal_diameters,
        azimuthal_azimuth_angles=azimuthal_angles,
        radial_targets=radial_targets,
        spherical_targets=spherical_targets,
    )


__all__ = ("CompiledPhysicalViews", "compile_physical_views")
