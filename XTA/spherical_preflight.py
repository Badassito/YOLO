"""Bounded deterministic CPU checks for a production spherical CUDA plane.

Offline CUDA qualification still compares entire outputs. Runtime checks use
the unchanged scalar pull on contiguous windows so validating a GPU does not
reserve it while millions of CPU QSC evaluations run. Image/ROI edges have
priority, followed by GPU foreground bounds and regular interior windows.
"""
from __future__ import annotations

import numpy as np


_FULL_PLANE_PIXELS = 64 * 1024
_PROBE_PIXELS = 32 * 1024
_WINDOW_PIXELS = 128


def _grid(first, stop, count):
    if stop <= first:
        return ()
    return tuple(sorted(set(int(round(value)) for value in np.linspace(first, stop - 1, count))))


def _edge_points(first, stop, limit):
    return tuple(sorted(set(max(0, min(limit - 1, value)) for value in
        (first - 1, first, first + 1, (first + stop - 1) // 2, stop - 2, stop - 1, stop))))


def spherical_preflight_windows(shape, bounds, foreground_bounds=None, *, full=False):
    """Return a mode and disjoint half-open flat windows; large planes cap at 32K."""
    height, width = map(int, shape)
    if min(height, width) <= 0:
        raise ValueError('Spherical math preflight needs a positive plane shape')
    if full or height * width <= _FULL_PLANE_PIXELS:
        # Full diagnostics preserve the original bounded CPU chunk size.
        return ('full_requested' if full else 'full_small', tuple(
            (first, min(height * width, first + 128 * 1024))
            for first in range(0, height * width, 128 * 1024)))
    windows = []

    def add(row, column):
        row = max(0, min(height - 1, int(row)))
        column = max(0, min(width - 1, int(column)))
        length = min(width, _WINDOW_PIXELS)
        x0 = max(0, min(width - length, column - length // 2))
        first = row * width + x0
        candidate = sorted((*windows, (first, first + length)))
        merged = []
        for start, stop in candidate:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            else:
                merged.append((start, stop))
        if sum(stop - start for start, stop in merged) <= _PROBE_PIXELS:
            windows[:] = merged

    def cross(rows, columns):
        for row in rows:
            for column in columns:
                add(row, column)

    # These mandatory sets include first/last rows and columns, both sides of
    # the conservative ROI, and the ROI center. Even their unmerged maximum is
    # below the budget; only later interior additions can be declined.
    cross(_edge_points(0, height, height), _edge_points(0, width, width))
    cross(_edge_points(bounds.y0, bounds.y1, height), _edge_points(bounds.x0, bounds.x1, width))
    if foreground_bounds is not None:
        y0, y1, x0, x1 = map(int, foreground_bounds)
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError('Spherical math preflight received invalid GPU foreground bounds')
        cross(_edge_points(y0, y1, height), _edge_points(x0, x1, width))
        cross(_grid(y0, y1, 5), _grid(x0, x1, 5))
    cross(_grid(0, height, 9), _grid(0, width, 9))
    cross(_grid(bounds.y0, bounds.y1, 7), _grid(bounds.x0, bounds.x1, 7))
    return 'bounded_windows', tuple(windows)


def validate_spherical_preflight_plane(plane, oracle, bounds, foreground_bounds=None, *, z=0, full=False):
    """Compare every selected window; any mismatch aborts CUDA admission."""
    checked = np.asarray(plane)
    if checked.ndim != 2 or checked.dtype != np.uint8:
        raise ValueError('Spherical math preflight requires a uint8 plane')
    mode, windows = spherical_preflight_windows(checked.shape, bounds, foreground_bounds, full=full)
    flattened = checked.reshape(-1)
    compared = 0
    for first, stop in windows:
        expected = np.asarray(oracle(first, stop))
        actual = flattened[first:stop]
        if expected.shape != actual.shape or expected.dtype != np.uint8:
            raise ValueError('Spherical CPU preflight oracle returned an invalid window')
        if not np.array_equal(actual, expected):
            different = np.flatnonzero(actual != expected)
            position = first + int(different[0])
            raise ValueError(
                f'Spherical CUDA math preflight differs from CPU at {len(different)} checked voxels '
                f'on source Z={z}, first (Y,X)={divmod(position, checked.shape[1])}, '
                f'window=[{first}:{stop}), mode={mode}')
        compared += stop - first
    return mode, compared
