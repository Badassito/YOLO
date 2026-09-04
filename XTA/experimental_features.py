"""Dependency-light configuration for opt-in experimental execution paths.

This module deliberately depends only on the standard library so runtime telemetry,
schedulers, and CUDA implementations can share one environment contract without
importing accelerator frameworks or higher pipeline layers.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Mapping, Optional


GIB = 1024 ** 3

D1_OWNER_GROUPS_ENV = "YOLO_TTA_D1_OWNER_GROUPS"
D1_OWNER_GROUP_SIZE_ENV = "YOLO_TTA_D1_OWNER_GROUP_SIZE"
GPU_RESIDENT_TAIL_ENV = "YOLO_TTA_GPU_RESIDENT_TAIL"
GPU_RESIDENT_TAIL_REQUIRED_ENV = "YOLO_TTA_GPU_RESIDENT_TAIL_REQUIRED"
GPU_TAIL_RESERVE_GIB_ENV = "YOLO_TTA_GPU_TAIL_RESERVE_GIB"
GPU_TAIL_BLOCK_SLICES_ENV = "YOLO_TTA_GPU_TAIL_BLOCK_SLICES"

EXPERIMENTAL_FEATURE_ENV_NAMES = (
    D1_OWNER_GROUPS_ENV,
    D1_OWNER_GROUP_SIZE_ENV,
    GPU_RESIDENT_TAIL_ENV,
    GPU_RESIDENT_TAIL_REQUIRED_ENV,
    GPU_TAIL_RESERVE_GIB_ENV,
    GPU_TAIL_BLOCK_SLICES_ENV,
)

_FALSE_VALUES = frozenset(("", "0", "false", "no", "off", "disabled"))


def _env_flag(
    name: str,
    default: bool = False,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    source = os.environ if environ is None else environ
    raw = source.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in _FALSE_VALUES


def _env_int(
    name: str,
    default: int,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    source = os.environ if environ is None else environ
    raw = str(source.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_float(
    name: str,
    default: float,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> float:
    source = os.environ if environ is None else environ
    raw = str(source.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def d1_owner_groups_requested(
    *,
    environ: Optional[Mapping[str, str]] = None,
    read_int: Optional[Callable[[str, int], int]] = None,
) -> bool:
    """Return whether multi-device D1 ownership was explicitly requested."""

    if read_int is not None:
        return bool(int(read_int(D1_OWNER_GROUPS_ENV, 0)) != 0)
    return _env_flag(D1_OWNER_GROUPS_ENV, False, environ=environ)


def d1_owner_group_size_limit(
    *,
    device_count: Optional[int] = None,
    environ: Optional[Mapping[str, str]] = None,
    read_int: Optional[Callable[[str, int], int]] = None,
) -> int:
    """Return the configured D1 participant ceiling, clamped to available devices."""

    configured = (
        int(read_int(D1_OWNER_GROUP_SIZE_ENV, 8))
        if read_int is not None
        else _env_int(D1_OWNER_GROUP_SIZE_ENV, 8, environ=environ)
    )
    limit = max(1, min(8, int(configured)))
    if device_count is not None:
        limit = min(limit, max(1, int(device_count)))
    return int(limit)


def gpu_resident_tail_enabled(
    *, environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether the default-off GPU-resident finalization path is enabled."""

    return _env_flag(GPU_RESIDENT_TAIL_ENV, False, environ=environ)


def gpu_resident_tail_required(
    *, environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether GPU-resident-tail admission or execution failure is fatal."""

    return _env_flag(GPU_RESIDENT_TAIL_REQUIRED_ENV, False, environ=environ)


def gpu_tail_reserve_bytes(
    *, environ: Optional[Mapping[str, str]] = None,
) -> int:
    reserve_gib = max(
        1.0,
        _env_float(GPU_TAIL_RESERVE_GIB_ENV, 8.0, environ=environ),
    )
    return int(reserve_gib * GIB)


def gpu_tail_block_slices(
    *, environ: Optional[Mapping[str, str]] = None,
) -> int:
    return max(4, _env_int(GPU_TAIL_BLOCK_SLICES_ENV, 32, environ=environ))


@dataclass(frozen=True)
class ExperimentalFeatureSnapshot:
    """One coherent read of the experimental feature environment."""

    d1_owner_groups_requested: bool
    d1_owner_group_size_limit: int
    gpu_resident_tail_enabled: bool
    gpu_resident_tail_required: bool
    gpu_tail_reserve_bytes: int
    gpu_tail_block_slices: int

    def as_dict(self) -> Dict[str, object]:
        return dict(asdict(self))

    def requested_dict(self) -> Dict[str, bool]:
        return {
            "d1_owner_groups": bool(self.d1_owner_groups_requested),
            "gpu_resident_tail": bool(self.gpu_resident_tail_enabled),
            "gpu_resident_tail_required": bool(self.gpu_resident_tail_required),
        }


def experimental_features_snapshot(
    *,
    environ: Optional[Mapping[str, str]] = None,
    device_count: Optional[int] = None,
) -> ExperimentalFeatureSnapshot:
    """Read and normalize every experimental feature setting exactly once."""

    return ExperimentalFeatureSnapshot(
        d1_owner_groups_requested=d1_owner_groups_requested(environ=environ),
        d1_owner_group_size_limit=d1_owner_group_size_limit(
            device_count=device_count,
            environ=environ,
        ),
        gpu_resident_tail_enabled=gpu_resident_tail_enabled(environ=environ),
        gpu_resident_tail_required=gpu_resident_tail_required(environ=environ),
        gpu_tail_reserve_bytes=gpu_tail_reserve_bytes(environ=environ),
        gpu_tail_block_slices=gpu_tail_block_slices(environ=environ),
    )


__all__ = [
    "D1_OWNER_GROUPS_ENV",
    "D1_OWNER_GROUP_SIZE_ENV",
    "EXPERIMENTAL_FEATURE_ENV_NAMES",
    "ExperimentalFeatureSnapshot",
    "GPU_RESIDENT_TAIL_ENV",
    "GPU_RESIDENT_TAIL_REQUIRED_ENV",
    "GPU_TAIL_BLOCK_SLICES_ENV",
    "GPU_TAIL_RESERVE_GIB_ENV",
    "d1_owner_group_size_limit",
    "d1_owner_groups_requested",
    "experimental_features_snapshot",
    "gpu_resident_tail_enabled",
    "gpu_resident_tail_required",
    "gpu_tail_block_slices",
    "gpu_tail_reserve_bytes",
]
