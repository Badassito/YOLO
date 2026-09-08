"""Bounded run-interval adjacency for coherent slice labels.

Every run represents equal nonzero labels on one row. Intersecting shifted runs
finds the same touching label pairs as the pixel oracle. Fragmented inputs exceed
an explicit run/pair cap and return to the existing bounded pixel hash.
"""
from __future__ import annotations

import numpy as np
from ._deps import _numba

_TOPOLOGY_RUNS_DISABLED = False


if _numba is not None:
    @_numba.njit(cache=True, nogil=True)
    def _topology_label_runs(plane, capacity):
        runs = np.empty((capacity, 3), np.uint32)
        rows = np.empty(plane.shape[0] + 1, np.int64)
        count = 0
        for y in range(plane.shape[0]):
            rows[y] = count
            x = 0
            while x < plane.shape[1]:
                value = plane[y, x]
                if value <= 0:
                    x += 1
                    continue
                first = x
                x += 1
                while x < plane.shape[1] and plane[y, x] == value:
                    x += 1
                if count == capacity:
                    return runs, rows, -1
                runs[count, 0] = first
                runs[count, 1] = x
                runs[count, 2] = value
                count += 1
        rows[plane.shape[0]] = count
        return runs, rows, count

    @_numba.njit(cache=True, nogil=True)
    def _topology_run_intersections(a, arows, b, brows, groups, ao, bo, capacity):
        codes = np.empty(capacity, np.uint64)
        count = 0
        last = np.uint64(0)
        height = len(arows) - 1
        for group in groups:
            dy, dx0, dx1 = group
            for y in range(height):
                cy = y + dy
                if cy < 0 or cy >= height:
                    continue
                first_b = brows[cy]
                stop_b = brows[cy + 1]
                for i in range(arows[y], arows[y + 1]):
                    left = np.int64(a[i, 0]) + dx0
                    right = np.int64(a[i, 1]) + dx1
                    while first_b < stop_b and np.int64(b[first_b, 1]) <= left:
                        first_b += 1
                    j = first_b
                    while j < stop_b and np.int64(b[j, 0]) < right:
                        code = ((np.uint64(a[i, 2]) + ao) << np.uint64(32)) | (np.uint64(b[j, 2]) + bo)
                        if code != last:
                            if count == capacity:
                                return codes, -1
                            codes[count] = code
                            count += 1
                            last = code
                        j += 1
        return codes, count
else:
    _topology_label_runs = _topology_run_intersections = None


def _run_adjacent_pair_codes(prev, curr, offsets, prev_offset=0, curr_offset=0, *,
                             run_cap=65536, pair_cap=262144):
    """Return exact sorted pairs, or None when the bounded run path is unsuitable."""
    if _topology_label_runs is None or min(prev.shape, default=0) <= 0:
        return None
    # On fragmented images abandon the run scan early, before allocating a dense
    # run representation. At most 1.5 MiB of runs plus 2 MiB of pair scratch.
    capacity = min(int(run_cap), max(16, int(prev.size) // 8))
    if capacity <= 0 or int(pair_cap) <= 0:
        return None
    a, ar, ac = _topology_label_runs(prev, capacity)
    if ac < 0:
        return None
    b, br, bc = _topology_label_runs(curr, capacity)
    if bc < 0:
        return None
    if not ac or not bc:
        return np.empty(0, np.uint64)
    groups = []
    for dy, dx in sorted(set(tuple(map(int, v)) for v in offsets)):
        if groups and groups[-1][0] == dy and groups[-1][2] + 1 == dx:
            groups[-1][2] = dx
        else:
            groups.append([dy, dx, dx])
    codes, count = _topology_run_intersections(
        a, ar, b, br, np.asarray(groups, np.int64).reshape(-1, 3),
        np.uint64(prev_offset), np.uint64(curr_offset), np.int64(pair_cap),
    )
    return None if count < 0 else np.unique(codes[:count])


def run_adjacent_pair_codes(prev, curr, offsets, prev_offset=0, curr_offset=0, *,
                            run_cap=65536, pair_cap=262144):
    """A failed optional compiler falls back before any topology mutation."""
    global _TOPOLOGY_RUNS_DISABLED
    if _TOPOLOGY_RUNS_DISABLED:
        return None
    try:
        return _run_adjacent_pair_codes(prev, curr, offsets, prev_offset, curr_offset,
                                       run_cap=run_cap, pair_cap=pair_cap)
    except Exception as exc:
        _TOPOLOGY_RUNS_DISABLED = True
        print(f'Run adjacency unavailable; retaining pixel adjacency: {exc}', flush=True)
        return None
