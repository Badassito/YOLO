"""Check thin-object round trips for isolated relaxed Spherical samplers.

A threshold oracle replaces inference here so sampling effects are visible.
This is a geometry qualification, not a segmentation-accuracy benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from XTA import spherical_projection as sp
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from tools.benchmark_spherical_sampling import kernels, render, directions_for
from tools.geometry_quality_metrics import compare_binary_masks, phantom_labels


def compact(metrics):
    result = {k: v for k, v in metrics.items() if k != 'components'}
    result['minimum_component_recall'] = min((x['recall_within_one_voxel'] for x in metrics['components']), default=1.)
    result['minimum_centerline_proxy_recall'] = min((x['centerline_proxy_recall'] for x in metrics['components']), default=1.)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error('Use a task Scratch output path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR', str(args.output.parent/'cupy-cache'))
    import cupy as cp
    compiled = kernels(cp)
    native, working = (17, 65, 67), (25, 65, 67)
    base = build_spherical_view_infos(*working, targets=('transverse',), min_radius=1., patch_size=64, tilted_views=())
    specs = {
        'rods': [{'kind': 'rod', 'start': (8, 27, 24), 'stop': (8, 27, 42), 'width': 1},
                 {'kind': 'rod', 'start': (9, 35, 24), 'stop': (9, 35, 42), 'width': 2}],
        'parallel_gap': [{'kind': 'rod', 'start': (8, 29, 25), 'stop': (8, 29, 41), 'width': 1},
                         {'kind': 'rod', 'start': (8, 32, 25), 'stop': (8, 32, 41), 'width': 1}],
        'sheet': [{'kind': 'sheet', 'start': (8, 24, 25), 'stop': (9, 41, 42)}],
        'ring': [{'kind': 'ring', 'center': (8, 32, 33), 'axis': 0, 'radius': 8, 'width': 1}],
    }
    z, y, x = np.ogrid[:native[0], :native[1], :native[2]]
    dz = (z + .5) * working[0] / native[0] - .5 - (working[0]-1)/2
    radius = np.sqrt(dz*dz + (y-32)**2 + (x-33)**2)
    roi = (radius >= 1) & (radius <= base[0].spherical_max_radius)
    rows = []
    report = {'scope': __doc__, 'native_shape': native, 'working_shape': working,
              'sampler_frames_per_case': sum(v.num_slices for v in base), 'cases': rows}
    for label, objects in specs.items():
        truth = (phantom_labels(native, objects) != 0) & roi
        source = cp.asarray(truth.astype(np.uint8) * 255)
        for pose, rotation in (('upright', None), ('vertical30', cube_rotation('vertical', 30))):
            views = [replace(v, spherical_rotation_xyz=rotation) for v in base] if rotation else base
            masks = {mode: {threshold: np.zeros(native, np.uint8) for threshold in (64, 128, 192)} for mode in compiled}
            count = {mode: 0 for mode in compiled}
            image_errors = {mode: {'max_abs': 0., 'changed': 0} for mode in compiled}
            for view in views:
                rays = directions_for(cp, view)
                images = {mode: [] for mode in compiled}
                for shell_radius in view.spherical_radii:
                    reference = render(cp, compiled, 'reference', source, view, shell_radius, rays).get()
                    for mode in compiled:
                        image = reference if mode == 'reference' else render(cp, compiled, mode, source, view, shell_radius, rays).get()
                        difference = np.abs(image-reference)
                        image_errors[mode]['max_abs'] = max(image_errors[mode]['max_abs'], float(difference.max()))
                        image_errors[mode]['changed'] += int(np.count_nonzero(difference))
                        images[mode].append(image)
                        count[mode] += 1
                for mode in compiled:
                    frames = np.stack(images[mode])
                    for threshold in masks[mode]:
                        categorical = (frames >= threshold).astype(np.uint8)
                        masks[mode][threshold] |= sp._project_spherical_block(categorical, view,
                            np.asarray(view.spherical_radii), np.asarray(view.spherical_rotation_xyz).reshape(3, 3),
                            native, 0, native[0])
            assert len(set(count.values())) == 1
            for threshold in masks['reference']:
                reference = masks['reference'][threshold]
                reference_xy = np.any(reference, axis=0)
                reference_hole = bool(np.any(ndi.binary_fill_holes(reference_xy) & ~reference_xy))
                for mode in compiled:
                    candidate = masks[mode][threshold]
                    xy = np.any(candidate, axis=0)
                    row = {'fixture': label, 'pose': pose, 'threshold': threshold, 'variant': mode,
                           'frames': count[mode], 'image_errors': image_errors[mode],
                           'vs_strict': compact(compare_binary_masks(reference, candidate)),
                           'vs_source_truth': compact(compare_binary_masks(truth, candidate)),
                           'reference_xy_hole': reference_hole,
                           'candidate_xy_hole': bool(np.any(ndi.binary_fill_holes(xy) & ~xy))}
                    rows.append(row)
            print(f'{label}/{pose}: all variants used {count["reference"]} frames.', flush=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
