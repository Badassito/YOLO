"""Bounded CPU-only A/B: analytic Z/Y scheduling against the full pull oracle.

Uses a small processing mask and sampled output slabs, never a full native
canvas. Timings are isolated synthetic CPU evidence, not end-to-end speedups.
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

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
os.environ['YOLO_TTA_GPU_SPHERICAL_BACKPROJECT'] = '0'

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import spherical_projection as sp
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from XTA.spherical_projection_bounds import spherical_output_bounds


def triple(value):
    result = tuple(map(int, value.split(',')))
    if len(result) != 3 or min(result) <= 0:
        raise argparse.ArgumentTypeError('requires three positive comma-separated dimensions')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--working-shape', type=triple, default=(513, 547, 571))
    parser.add_argument('--output-shape', type=triple, default=(389, 563, 557))
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--slab-depth', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeats, args.slab_depth, args.patch_size) <= 0:
        parser.error('repeat, slab depth and patch size must be positive')
    shape = args.output_shape
    built = build_spherical_view_infos(*args.working_shape, targets=('transverse',),
        min_radius=args.patch_size / (4 * np.pi), patch_size=args.patch_size, tilted_views=())
    # One patch per face, including alternating corners on overlapping grids.
    views = [next(v for v in (built if face % 2 == 0 else built[::-1])
                  if v.spherical_face == face) for face in range(6)]
    starts = sorted(set((0, shape[0] // 4, shape[0] // 2, 3 * shape[0] // 4, shape[0] - 1)))
    rng = np.random.default_rng(20260909)
    rows = []
    for rotation_name, rotation in (('upright', None), ('vertical31', cube_rotation('vertical', 31)),
                                    ('horizontal-23', cube_rotation('horizontal', -23))):
        for initial in views:
            view = replace(initial, spherical_rotation_xyz=rotation) if rotation else initial
            source = rng.integers(0, 2, (view.num_slices, 31, 37), dtype=np.uint8)
            radii = np.asarray(view.spherical_radii)
            matrix = np.asarray(view.spherical_rotation_xyz).reshape(3, 3)
            boxes = np.zeros((view.num_slices, 4), np.int64)
            boxes[len(boxes) // 4:3 * len(boxes) // 4] = (1, 30, 2, 35)
            for case, metadata in (('all_shells', None), ('middle_shells', boxes)):
                plan_started = time.perf_counter()
                bounds = spherical_output_bounds(view, shape, metadata)
                planning_seconds = time.perf_counter() - plan_started
                timings = {'full': [], 'bounded': []}
                reference_hash = None
                for repetition in range(args.repeats):
                    # Alternate A/B order so the first path does not always pay
                    # for the same cold source/cache effects.
                    for mode in (('full', 'bounded') if repetition % 2 == 0 else ('bounded', 'full')):
                        digest = hashlib.sha256()
                        started = time.perf_counter()
                        for first in starts:
                            block = sp._project_spherical_block(source, view, radii, matrix, shape,
                                first, min(args.slab_depth, shape[0] - first), metadata,
                                bounds if mode == 'bounded' else None)
                            digest.update(block)
                        elapsed = time.perf_counter() - started
                        timings[mode].append(elapsed)
                        actual_hash = digest.hexdigest()
                        if reference_hash is None:
                            reference_hash = actual_hash
                        if actual_hash != reference_hash:
                            raise AssertionError(f'{rotation_name}/{view.spherical_face}/{case}: exact output mismatch')
                full = statistics.median(timings['full'])
                bounded = statistics.median(timings['bounded'])
                full_voxels = sum(min(args.slab_depth, shape[0] - first) for first in starts) * shape[1] * shape[2]
                bounded_voxels = sum(max(0, min(first + args.slab_depth, shape[0], bounds.z1) -
                    max(first, bounds.z0)) for first in starts) * (bounds.y1 - bounds.y0) * shape[2]
                row = {'rotation': rotation_name, 'face': view.spherical_face, 'case': case,
                    'planning_seconds': planning_seconds, 'timings': timings,
                    'median_full_seconds': full, 'median_bounded_seconds': bounded,
                    'speedup': full / bounded, 'full_pull_voxels': full_voxels,
                    'bounded_pull_voxels': bounded_voxels, 'output_sha256': reference_hash}
                rows.append(row)
                print(f'{rotation_name} face={view.spherical_face} {case}: '
                      f'{full:.4f} -> {bounded:.4f}s ({full / bounded:.2f}x), exact SHA256 parity', flush=True)
    report = {'working_shape': args.working_shape, 'output_shape': shape,
        'patch_size': args.patch_size, 'sampled_z_starts': starts, 'slab_depth': args.slab_depth,
        'repeats': args.repeats, 'projection_sha256': hashlib.sha256(Path(sp.__file__).read_bytes()).hexdigest(),
        'limits': 'CPU-only synthetic sampled slabs; reduced processing masks; no inference, GPU, or end-to-end timing.',
        'cases': rows, 'total_full_seconds': sum(r['median_full_seconds'] for r in rows),
        'total_bounded_seconds': sum(r['median_bounded_seconds'] for r in rows)}
    report['aggregate_speedup'] = report['total_full_seconds'] / report['total_bounded_seconds']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Aggregate: {report["aggregate_speedup"]:.2f}x; {args.output}', flush=True)


if __name__ == '__main__':
    main()
