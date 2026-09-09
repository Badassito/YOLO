"""Small 3D geometry phantoms and feature-wise binary quality checks.

All coordinates are source-array (Z,Y,X) voxel centers. Matching uses a one-voxel
Chebyshev neighborhood; surface distances use Euclidean source-voxel units.
The ridge metric is a distance-transform centerline proxy, not a skeleton.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


_NEIGHBORS = np.ones((3, 3, 3), bool)


def _shape(shape):
    shape = tuple(map(int, shape))
    if len(shape) != 3 or min(shape) < 1 or max(shape) > 128:
        raise ValueError('quality fixtures require three dimensions between 1 and 128')
    return shape


def compare_binary_masks(reference, candidate):
    """Report component loss/splits/merges and displacement without hiding thin loss.

    A tolerant match alone is not proof of object identity. Exact-overlap merges
    are distinguished from possible merges inferred only within the tolerance.
    Per-component coverage catches truncation even when the component survives.
    Nonzero labels count as foreground; callers check categorical alphabets.
    """
    ref, got = np.asarray(reference), np.asarray(candidate)
    _shape(ref.shape)
    if got.shape != ref.shape:
        raise ValueError('reference and candidate shapes must match')
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError('binary quality inputs must contain finite values')
    ref, got = ref != 0, got != 0
    rl, nr = ndi.label(ref, _NEIGHBORS)
    gl, ng = ndi.label(got, _NEIGHBORS)
    near = ndi.binary_dilation(got, _NEIGHBORS)
    radius = ndi.distance_transform_edt(np.pad(ref, 1))[1:-1, 1:-1, 1:-1]
    ridge = ref & (radius == ndi.maximum_filter(radius, size=3, mode='constant'))
    exact_owners, near_owners, components = {}, {}, []
    for label, box in enumerate(ndi.find_objects(rl), 1):
        roi = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1)) for s, n in zip(box, ref.shape))
        own = rl[roi] == label
        exact = np.unique(gl[roi][own & got[roi]])
        matches = np.unique(gl[roi][ndi.binary_dilation(own, _NEIGHBORS) & got[roi]])
        exact, matches = list(map(int, exact)), list(map(int, matches))
        for target in exact:
            exact_owners.setdefault(target, []).append(label)
        for target in matches:
            near_owners.setdefault(target, []).append(label)
        core = own & ridge[roi]
        components.append({
            'reference_component': label, 'voxels': int(own.sum()),
            'thin': bool(radius[roi][own].max() <= 2.),
            'survives_within_one_voxel': bool(matches),
            'recall_within_one_voxel': float(near[roi][own].mean()),
            'centerline_proxy_recall': float(near[roi][core].mean()),
            'exact_candidate_components': exact, 'near_candidate_components': matches,
        })
    for component in components:
        matches = component['near_candidate_components']
        component['matching_ambiguous'] = (len(matches) > 1 or
            any(len(near_owners[target]) > 1 for target in matches))
    # Bidirectional distances also expose extra disconnected material and shape
    # expansion that reference-only coverage would otherwise count as success.
    distances = []
    for first, second in ((ref, got), (got, ref)):
        surface = first & ~ndi.binary_erosion(first, _NEIGHBORS, border_value=0)
        other = second & ~ndi.binary_erosion(second, _NEIGHBORS, border_value=0)
        if surface.any():
            distances.append(ndi.distance_transform_edt(~other)[surface]
                             if other.any() else np.full(int(surface.sum()), np.inf))
    distance = np.concatenate(distances) if distances else np.zeros(1)
    exact_merges = sorted(k for k, owners in exact_owners.items() if len(owners) > 1)
    possible_merges = sorted(k for k, owners in near_owners.items()
                             if len(owners) > 1 and k not in exact_merges)
    union = np.count_nonzero(ref | got)
    return {
        'iou': float(np.count_nonzero(ref & got) / union) if union else 1.,
        'reference_voxels': int(ref.sum()), 'candidate_voxels': int(got.sum()),
        'connectivity': 26, 'matching_tolerance_linf_voxels': 1,
        'reference_components': int(nr), 'candidate_components': int(ng),
        'missed_components': sum(not c['survives_within_one_voxel'] for c in components),
        'missed_thin_components': sum(c['thin'] and not c['survives_within_one_voxel'] for c in components),
        'split_reference_components': [c['reference_component'] for c in components
                                       if len(c['exact_candidate_components']) > 1],
        'merged_candidate_components': exact_merges,
        'possible_merge_candidate_components': possible_merges,
        'unmatched_candidate_components': sorted(set(range(1, ng + 1)) - near_owners.keys()),
        'surface_distance_p95': float(np.percentile(distance, 95)) if np.isfinite(distance).all() else float('inf'),
        'surface_distance_max': float(distance.max()),
        'components': components,
    }


def phantom_labels(shape, objects):
    """Rasterize disjoint objects, assigning label i+1 to objects[i].

    Rod: kind/start/stop/width; sheet: kind/start/stop (half-open box);
    ring: kind/center/axis/radius/width. Width is 1, 2 or 3 voxels.
    Callers place these at native-T knots, QSC seams/corners or other test sites.
    """
    shape = _shape(shape)
    labels = np.zeros(shape, np.int32)
    grid = np.ogrid[tuple(slice(0, n) for n in shape)]
    for label, spec in enumerate(objects, 1):
        mask = np.zeros(shape, bool)
        width = int(spec.get('width', 1))
        if width not in (1, 2, 3):
            raise ValueError('phantom widths must be 1, 2 or 3 voxels')
        if spec['kind'] == 'rod':
            first, last = np.asarray(spec['start'], float), np.asarray(spec['stop'], float)
            points = np.rint(np.linspace(first, last, 2 * int(np.ceil(np.max(abs(last - first)))) + 1)).astype(int)
            if points.shape[1:] != (3,) or np.any(points < 0) or np.any(points >= shape):
                raise ValueError('rod endpoints must lie inside the source shape')
            mask[tuple(points.T)] = True
            mask = ndi.maximum_filter(mask, size=width, mode='constant')
        elif spec['kind'] == 'sheet':
            first, last = tuple(spec['start']), tuple(spec['stop'])
            if len(first) != 3 or len(last) != 3 or any(not 0 <= a < b <= n for a, b, n in zip(first, last, shape)):
                raise ValueError('sheet bounds must be a nonempty source-space box')
            mask[tuple(slice(a, b) for a, b in zip(first, last))] = True
        elif spec['kind'] == 'ring':
            center, axis, radius = tuple(spec['center']), int(spec['axis']), float(spec['radius'])
            if axis not in (0, 1, 2) or len(center) != 3 or not np.isfinite(radius) or radius <= 0:
                raise ValueError('ring needs a source center, axis 0/1/2 and positive radius')
            a, b = [k for k in range(3) if k != axis]
            radial = np.sqrt((grid[a] - center[a]) ** 2 + (grid[b] - center[b]) ** 2)
            mask = (radial - radius) ** 2 + (grid[axis] - center[axis]) ** 2 <= (width / 2.) ** 2
        else:
            raise ValueError('unknown phantom kind')
        if not mask.any() or np.any(labels[mask]):
            raise ValueError('phantom objects must be nonempty and must not overlap')
        labels[mask] = label
    return labels
