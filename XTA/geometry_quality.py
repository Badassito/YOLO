"""Dependency-light opt-in geometry controls shared by parent and workers.

Local controls override the global bundle. These are requests, not evidence of
backend availability or successful dispatch; each backend keeps its own guards.
"""
from __future__ import annotations

import operator
import os


SPHERICAL_FP32_MAX_AXIS = 4096
SPHERICAL_FP32_BACKEND = 'cuda_spherical_fp32_virtual_gray8'


def _flag(name, default=False):
    value = os.environ.get(name)
    return bool(default) if value is None else value.strip().lower() not in ('', '0', 'false', 'no', 'off')


def fast_geometry_enabled():
    return _flag('YOLO_TTA_FAST_GEOMETRY')


def spherical_fp32_requested():
    return _flag('YOLO_TTA_GPU_SPHERICAL_FP32', fast_geometry_enabled())


def spherical_cpu_compiled_requested():
    return _flag('YOLO_TTA_CPU_SPHERICAL_COMPILED', fast_geometry_enabled())


def radial_columns_requested():
    return _flag('YOLO_TTA_GPU_RADIAL_COLUMN_GEOMETRY', fast_geometry_enabled())


def spherical_fp32_shape_eligible(shape):
    try:
        axes = tuple(operator.index(axis) for axis in shape)
    except (TypeError, ValueError):
        return False
    return bool(axes) and all(0 < axis <= SPHERICAL_FP32_MAX_AXIS for axis in axes)


def geometry_quality_request_record():
    """Resolved, spawn-visible requests without exposing unrelated environment."""
    spherical = spherical_fp32_requested()
    return {
        'fast_geometry_requested': fast_geometry_enabled(),
        'cpu_spherical_compiled_requested': spherical_cpu_compiled_requested(),
        'gpu_radial_column_geometry_requested': radial_columns_requested(),
        'gpu_spherical_fp32_requested': spherical,
        'spherical_sampler_requested': 'fp32_fma_virtual_gray8' if spherical else 'reference_fp64',
        'spherical_fp32_max_axis': SPHERICAL_FP32_MAX_AXIS,
        'native_t_fusion': False,
        'categorical_sampling': 'unchanged_nearest',
        'requested_not_actual_dispatch': True,
    }
