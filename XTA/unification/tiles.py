"""Dependency-light grouped tile grammar shared by v18 PTA and TTA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class ResolvedTileGroup:
    tile_size: int
    tile_stride: int
    config_id: str


def _group_values(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else list(values)
    groups: list[str] = []
    for value in raw_values:
        groups.extend(
            token for token in re.split(r"\s+", str(value).strip()) if token
        )
    return groups


def resolve_tile_groups(
    values: Sequence[str] | str | None,
) -> Tuple[ResolvedTileGroup, ...]:
    """Resolve strict ``TILE_SIZE:TILE_STRIDE`` groups once for both modes."""

    groups: list[ResolvedTileGroup] = []
    seen: set[str] = set()
    for raw_group in _group_values(values):
        slots = [slot.strip() for slot in str(raw_group).split(":")]
        if len(slots) != 2 or not all(slots):
            raise ValueError(
                f"--enable_tile group {raw_group!r} requires TILE_SIZE:TILE_STRIDE"
            )
        if any("," in slot or re.search(r"\s", slot) for slot in slots):
            raise ValueError(
                f"--enable_tile group {raw_group!r} accepts one TILE_SIZE and "
                "TILE_STRIDE; use spaces to separate additional groups"
            )
        try:
            tile_size = int(slots[0])
            tile_stride = int(slots[1])
        except Exception as exc:
            raise ValueError(
                f"--enable_tile group {raw_group!r} requires strict integer values"
            ) from exc
        if tile_size <= 0:
            raise ValueError("--enable_tile requires TILE_SIZE > 0")
        if tile_stride <= 0:
            raise ValueError("--enable_tile requires TILE_STRIDE > 0")
        if tile_stride > tile_size:
            raise ValueError("--enable_tile requires TILE_STRIDE <= TILE_SIZE")
        config_id = f"s{tile_size}_st{tile_stride}"
        if config_id in seen:
            raise ValueError(
                f"--enable_tile contains duplicate group {tile_size}:{tile_stride}"
            )
        seen.add(config_id)
        groups.append(ResolvedTileGroup(tile_size, tile_stride, config_id))
    return tuple(groups)


__all__ = ("ResolvedTileGroup", "resolve_tile_groups")
