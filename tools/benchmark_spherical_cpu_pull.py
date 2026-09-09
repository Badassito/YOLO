"""CPU-only compiled-vs-NumPy pull qualification at native production coordinates.

The source is a read-only broadcast of one native-sized mask plane across all
shells. This preserves full-resolution label boundaries without allocating the
11 GiB native mask. Timings measure synthetic compute, not production walltime.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '2'
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage

from XTA import spherical_projection as reference
from XTA import spherical_projection_cpu as candidate
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from XTA.spherical_projection_bounds import spherical_output_bounds


def render_plane(pull, source, view, shape, z, boxes):
    result = np.zeros(shape[1:], np.uint8)
    bounds = spherical_output_bounds(view, shape, boxes)
    if not bounds.z0 <= z < bounds.z1:
        return result
    radii = np.asarray(view.spherical_radii, np.float64)
    rotation = np.asarray(view.spherical_rotation_xyz, np.float64).reshape(3, 3)
    plane = result.reshape(-1)
    stop_pixel = bounds.y1 * shape[2]
    for first in range(bounds.y0 * shape[2], stop_pixel, 128 * 1024):
        stop = min(first + 128 * 1024, stop_pixel)
        plane[first:stop] = pull(source, view, radii, rotation, shape, z, first, stop, boxes)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--all-rotations', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('repeats must be from 1 through 10')
    shape, working, size = (1931, 3064, 3022), (2911, 3064, 3022), 3072
    built = build_spherical_view_infos(*working, targets=('transverse',),
                                      min_radius=None, patch_size=size, tilted_views=())
    rng = np.random.default_rng(142828)
    source_plane = (rng.random((size, size), dtype=np.float32) < .06).astype(np.uint8)
    source_plane[::37, :] = 1
    source_plane[:, ::41] = 1
    source_plane[np.arange(size), np.arange(size)] = 1
    source_plane[np.arange(size), size - 1 - np.arange(size)] = 1
    source = np.broadcast_to(source_plane, (built[0].num_slices, size, size))
    rotations = (('upright', cube_rotation()), ('vertical31', cube_rotation('vertical', 31)),
                 ('horizontal-23', cube_rotation('horizontal', -23)))
    warm_view = built[0]
    started = time.perf_counter()
    candidate.pull_spherical_chunk_numba(source, warm_view, np.asarray(warm_view.spherical_radii),
        np.asarray(warm_view.spherical_rotation_xyz).reshape(3, 3), shape, shape[0] // 2, 0, 1)
    compile_seconds = time.perf_counter() - started
    rows = []
    for face in range(6):
        initial = next(view for view in (built if face % 2 == 0 else built[::-1]) if view.spherical_face == face)
        for rotation_name, rotation in (rotations if args.all_rotations else (rotations[face % 3],)):
            view = replace(initial, spherical_rotation_xyz=rotation)
            from XTA.qsc import QSC_FACE_BASES
            normal = np.asarray(rotation).reshape(3, 3) @ np.asarray(QSC_FACE_BASES[face][0])
            radius = (view.spherical_min_radius + view.spherical_max_radius) / 2
            z = int(np.clip(np.rint((working[0] / 2 + radius * normal[2]) * shape[0] / working[0] - .5), 0, shape[0] - 1))
            boxes = None
            if face >= 3:
                boxes = np.zeros((view.num_slices, 4), np.int64)
                boxes[len(boxes) // 5:4 * len(boxes) // 5] = (1, size - 1, 2, size - 2)
            calls = {'numpy': reference._pull_spherical_chunk, 'numba_f64': candidate.pull_spherical_chunk_numba}
            outputs, timings = {}, {'numpy': [], 'numba_f64': []}
            for repeat in range(args.repeats):
                for name in (('numpy', 'numba_f64') if repeat % 2 == 0 else ('numba_f64', 'numpy')):
                    started = time.perf_counter()
                    outputs[name] = render_plane(calls[name], source, view, shape, z, boxes)
                    timings[name].append(time.perf_counter() - started)
            expected, actual = outputs['numpy'], outputs['numba_f64']
            different = actual != expected
            difference_count = int(np.count_nonzero(different))
            boundary_shift = 0.0
            if difference_count:
                # Distance to the nearest voxel of the opposite output's same
                # categorical value; this reports spatial movement, not intensity.
                for mask, other in ((actual, expected), (expected, actual)):
                    changed_positive = (mask != 0) & different
                    if np.any(changed_positive):
                        distance = ndimage.distance_transform_edt(other == 0)
                        boundary_shift = max(boundary_shift, float(distance[changed_positive].max()))
            peaks = {}
            for name in calls:
                tracemalloc.start()
                render_plane(calls[name], source, view, shape, z, boxes)
                _, peaks[name] = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            row = {'face': face, 'rotation': rotation_name, 'z': z, 'bbox_metadata': boxes is not None,
                   'output_pixels': int(expected.size), 'reference_foreground': int(np.count_nonzero(expected)),
                   'candidate_foreground': int(np.count_nonzero(actual)), 'different_voxels': difference_count,
                   'max_changed_label_distance_pixels': boundary_shift, 'timings_seconds': timings,
                   'speedup': statistics.median(timings['numpy']) / statistics.median(timings['numba_f64']),
                   'peak_traced_bytes': peaks,
                   'numpy_sha256': hashlib.sha256(expected).hexdigest(),
                   'numba_sha256': hashlib.sha256(actual).hexdigest()}
            rows.append(row)
            print(json.dumps({key: row[key] for key in ('face', 'rotation', 'different_voxels', 'speedup')}), flush=True)
    report = {
        'working_shape': working, 'output_shape': shape, 'native_patch': [size, size],
        'shells': built[0].num_slices, 'source_allocated_bytes': source_plane.nbytes,
        'source_logical_bytes': source.nbytes, 'thread_limit': 2, 'parallel_numba': False,
        'compile_or_cache_load_seconds': compile_seconds, 'repeats': args.repeats,
        'numerics': 'float64, fastmath=False, unchanged nearest categorical rules',
        'limitations': 'Broadcast mask reuses one plane across radii; timings isolate synthetic CPU computation, not full production mask bandwidth or pipeline walltime.',
        'source_sha256': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
        'cases': rows,
        'total_different_voxels': sum(row['different_voxels'] for row in rows),
        'total_output_pixels': sum(row['output_pixels'] for row in rows),
        'median_case_speedup': statistics.median(row['speedup'] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('total_different_voxels', 'total_output_pixels', 'median_case_speedup')}))


if __name__ == '__main__':
    main()
