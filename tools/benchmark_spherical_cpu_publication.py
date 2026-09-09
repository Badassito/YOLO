"""Compare exact dense/compact Spherical CPU pulls through the real mask writer.

Production coordinates are retained while the source broadcasts one mask plane
across radii and only a small, explicitly reported output slice set is evaluated.
This measures compute plus publication; it does not predict cluster walltime.
The baseline producer and compiled pull are loaded from a retained source release.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from unittest import mock

for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_variable] = '2'
os.environ['YOLO_TTA_GPU_SPHERICAL_BACKPROJECT'] = '0'
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from XTA import spherical_projection as candidate
from XTA.interpolation import CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore
from XTA.qsc import QSC_FACE_BASES
from XTA.spherical_geometry import build_spherical_view_infos, cube_rotation
from XTA.spherical_projection_bounds import spherical_output_bounds
from XTA.spherical_projection_cpu import prepare_spherical_chunk_numba


OUTPUT_SHAPE = (1931, 3064, 3022)
WORKING_SHAPE = (2911, 3064, 3022)
PATCH_SIZE = 3072


def load_baseline(root, stem):
    path = root / 'XTA' / f'{stem}.py'
    name = f'XTA._publication_benchmark_baseline_{stem}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_cases(samples, small=False):
    shape = (47, 83, 79) if small else OUTPUT_SHAPE
    working = (73, 83, 79) if small else WORKING_SHAPE
    patch = 96 if small else PATCH_SIZE
    views = build_spherical_view_infos(*working, targets=('transverse',), min_radius=None,
                                       patch_size=patch, tilted_views=())
    configurations = (
        ('known_empty', 0, 'upright', 0, 'empty'),
        ('upright_sparse', 0, 'upright', 0, 'sparse'),
        ('upright_pz', 4, 'upright', 0, 'wide'),
        ('tilted_sparse', 2, 'vertical', -30, 'sparse'),
        ('tilted_wide', 4, 'horizontal', 31, 'wide'),
    )
    rng = np.random.default_rng(142868)
    for name, face, axis, angle, pattern in configurations:
        view = next(v for v in views if v.spherical_face == face
                    and 0 <= v.spherical_face_intervals / 2 - v.spherical_u_origin < patch
                    and 0 <= v.spherical_face_intervals / 2 - v.spherical_v_origin < patch)
        rotation = cube_rotation() if axis == 'upright' else cube_rotation(axis, angle)
        view = replace(view, spherical_rotation_xyz=rotation)
        plane = np.zeros((patch, patch), np.uint8)
        boxes = np.zeros((view.num_slices, 4), np.int64)
        if pattern == 'sparse':
            cy = int(round(view.spherical_face_intervals / 2 - view.spherical_v_origin))
            cx = int(round(view.spherical_face_intervals / 2 - view.spherical_u_origin))
            half = max(3, patch // 32)
            y0, y1 = max(0, cy - half), min(patch, cy + half + 1)
            x0, x1 = max(0, cx - half), min(patch, cx + half + 1)
            plane[y0:y1, x0:x1] = 1
            plane[y0:y1:5, x0:x1:7] = 0
            boxes[:] = (y0, y1, x0, x1)
        elif pattern == 'wide':
            plane[:] = rng.random((patch, patch), dtype=np.float32) < .06
            plane[::37] = 1
            plane[:, ::41] = 1
            boxes[:] = (0, patch, 0, patch)
        source = np.broadcast_to(plane, (view.num_slices, patch, patch))
        bounds = spherical_output_bounds(view, shape, boxes)
        normal = np.asarray(rotation).reshape(3, 3) @ np.asarray(QSC_FACE_BASES[face][0])
        radius = .65 * float(view.spherical_max_radius)
        center = int(np.clip(round((working[0] / 2 + radius * normal[2]) * shape[0] / working[0] - .5),
                             0, shape[0] - 1))
        # Mix useful face slices with early/late analytic empties. This is a
        # bounded coordinate sample, not a claim about production occupancy.
        zs = {0, shape[0] - 1}
        for delta in range(-(samples // 2), samples // 2 + 1):
            zs.add(int(np.clip(center + delta, 0, shape[0] - 1)))
        for boundary in (bounds.z0 - 1, bounds.z0, bounds.z1 - 1, bounds.z1):
            if 0 <= boundary < shape[0]:
                zs.add(boundary)
        yield name, view, source, boxes, bounds, tuple(sorted(zs)), shape, working


def scan_instrumentation(stack, counters):
    """Measure separate diagnostic passes; never wrap timed benchmark runs."""
    for owner, attribute, key in ((cv2, 'boundingRect', 'bounding_rect'),
                                  (np, 'any', 'any'),
                                  (np, 'count_nonzero', 'count_nonzero'),
                                  (np, 'packbits', 'packbits')):
        original = getattr(owner, attribute)

        def wrapped(array, *args, _original=original, _key=key, **kwargs):
            counters[f'{_key}_calls'] = counters.get(f'{_key}_calls', 0) + 1
            counters[f'{_key}_input_bytes'] = counters.get(f'{_key}_input_bytes', 0) + np.asarray(array).nbytes
            return _original(array, *args, **kwargs)

        stack.enter_context(mock.patch.object(owner, attribute, wrapped))


def run_once(mode, producer, pull, case, directory, packed, *, instrument=False):
    name, view, source, boxes, bounds, zs, shape, _working = case
    radii = np.asarray(view.spherical_radii, np.float64)
    rotation = np.asarray(view.spherical_rotation_xyz, np.float64).reshape(3, 3)
    if directory.exists():
        raise FileExistsError(f'Refusing to replace existing benchmark store: {directory}')
    setup_started = time.perf_counter()
    writer = IncrementalRawBBoxMaskStoreWriter(
        shape=shape, store_dir=directory,
        format_name=INTERNAL_PACKED_CVOL_FORMAT if packed else CVOL_FORMAT,
        desc=f'{name} {mode}',
    )
    setup_seconds = time.perf_counter() - setup_started
    compute_seconds = publish_seconds = 0.0
    counters = {'pull_calls': 0, 'pull_evaluated_pixels': 0, 'pull_result_bytes': 0,
                'producer_dense_output_bytes': 0, 'producer_encoded_payload_bytes': 0}

    def measured_pull(*args, **kwargs):
        result = pull(*args, **kwargs)
        counters['pull_calls'] += 1
        counters['pull_evaluated_pixels'] += int(args[7]) - int(args[6])
        counters['pull_result_bytes'] += result.nbytes
        return result

    if callable(getattr(pull, 'rectangle', None)):
        def measured_rectangle(*args, **kwargs):
            result = pull.rectangle(*args, **kwargs)
            counters['pull_calls'] += 1
            counters['pull_evaluated_pixels'] += int(args[7]) - int(args[6])
            counters['pull_result_bytes'] += result.nbytes
            return result
        measured_pull.rectangle = measured_rectangle

    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            if instrument:
                scan_instrumentation(stack, counters)
                tracemalloc.start()
            next_z = 0
            for z in zs:
                # Unselected coordinates are intentionally absent from the
                # benchmark. Mark them empty identically in both stores.
                if z > next_z:
                    writer.consume_empty_range(next_z, z - next_z)
                tick = time.perf_counter()
                kwargs = {'packed': packed} if mode == 'compact' else {}
                block = producer(source, view, radii, rotation, shape, z, 1, boxes, bounds,
                                 cpu_pull=measured_pull if instrument else pull, **kwargs)
                compute_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                if mode == 'compact':
                    counters['producer_encoded_payload_bytes'] += block.payload.nbytes
                    if not any(record.foreground for record in block.records):
                        writer.consume_empty_range(z, len(block.records))
                    else:
                        writer.consume_encoded_block(z, block.records, block.payload, packed=packed)
                else:
                    counters['producer_dense_output_bytes'] += block.nbytes
                    writer.consume(z, block)
                publish_seconds += time.perf_counter() - tick
                next_z = z + 1
                del block
            if next_z < shape[0]:
                writer.consume_empty_range(next_z, shape[0] - next_z)
            tick = time.perf_counter()
            stats = writer.finalize()
            finalize_seconds = time.perf_counter() - tick
            peak_bytes = tracemalloc.get_traced_memory()[1] if instrument else None
    except BaseException as exc:
        writer.abort(exc)
        if writer._fd is not None:
            os.close(writer._fd)
            writer._fd = None
        raise
    finally:
        if instrument:
            tracemalloc.stop()
    return {
        'mode': mode, 'compute_seconds': compute_seconds, 'publish_seconds': publish_seconds,
        'compute_publish_seconds': compute_seconds + publish_seconds,
        'total_seconds': time.perf_counter() - started, 'finalize_seconds': finalize_seconds,
        'writer_setup_seconds': setup_seconds, 'peak_traced_bytes': peak_bytes,
        'counters': counters if instrument else None, 'store_stats': stats,
        'store': str(directory),
    }


def compare_stores(reference, actual, zs):
    left, right = RawBBoxMaskStore.open(Path(reference)), RawBBoxMaskStore.open(Path(actual))
    differing = 0
    hashes = {}
    try:
        index_equal = np.array_equal(left.index, right.index)
        # Independent store decoding checks geometry, tight boxes and payload
        # padding in addition to producer-array equivalence.
        for z in zs:
            a, b = left.decode_slice(z), right.decode_slice(z)
            differing += int(np.count_nonzero(a != b))
            hashes[str(z)] = hashlib.sha256(a).hexdigest()
        payload_equal = (Path(reference, 'chunks.bin').read_bytes()
                         == Path(actual, 'chunks.bin').read_bytes())
        stats_equal = left.meta['stats'] == right.meta['stats']
    finally:
        left.close()
        right.close()
    if differing or not index_equal or not payload_equal or not stats_equal:
        raise AssertionError(f'Dense/compact mismatch: voxels={differing}, index={index_equal}, '
                             f'payload={payload_equal}, stats={stats_equal}')
    return {'different_voxels': differing, 'index_records_equal': bool(index_equal),
            'payload_bytes_equal': payload_equal, 'store_stats_equal': stats_equal,
            'decoded_slice_sha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--samples', type=int, default=6, help='central sample count; boundary slices are added')
    parser.add_argument('--formats', nargs='+', choices=('raw', 'packed'), default=['raw', 'packed'])
    parser.add_argument('--small', action='store_true', help='quick functional check before production coordinates')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error('Generated evidence belongs outside the repository, in task Scratch')
    if not 1 <= args.repeats <= 10 or not 2 <= args.samples <= 32:
        parser.error('repeats must be 1..10 and samples 2..32')
    cv2.setNumThreads(1)
    baseline = load_baseline(args.baseline_root.resolve(), 'spherical_projection')
    baseline_cpu = load_baseline(args.baseline_root.resolve(), 'spherical_projection_cpu')
    encoded = getattr(candidate, '_project_spherical_encoded_block', None)
    if not callable(encoded):
        parser.error('candidate compact producer is unavailable')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output.parent / (args.output.stem + '-stores')
    if artifact_dir.exists():
        parser.error(f'benchmark artifact directory already exists: {artifact_dir}')
    artifact_dir.mkdir()
    rows = []
    compile_started = time.perf_counter()
    first = next(build_cases(args.samples, args.small))
    _, view, source, boxes, _, _, shape, _ = first
    radii = np.asarray(view.spherical_radii, np.float64)
    rotation = np.asarray(view.spherical_rotation_xyz, np.float64).reshape(3, 3)
    baseline_pull = baseline_cpu.prepare_spherical_chunk_numba(source, view, radii, rotation, shape, boxes)
    compact_pull = prepare_spherical_chunk_numba(source, view, radii, rotation, shape, boxes)
    compile_seconds = time.perf_counter() - compile_started
    for case in build_cases(args.samples, args.small):
        name, view, source, boxes, bounds, zs, shape, working = case
        for format_name in args.formats:
            packed = format_name == 'packed'
            runs = {'dense': [], 'compact': []}
            diagnostics = {}
            for repeat in range(args.repeats + 1):
                instrument = repeat == args.repeats
                for mode in (('dense', 'compact') if repeat % 2 == 0 else ('compact', 'dense')):
                    producer = baseline._project_spherical_block if mode == 'dense' else encoded
                    pull = baseline_pull if mode == 'dense' else compact_pull
                    directory = artifact_dir / f'{name}-{format_name}-{mode}-{repeat}'
                    result = run_once(mode, producer, pull, case, directory, packed, instrument=instrument)
                    if instrument:
                        diagnostics[mode] = result
                    else:
                        runs[mode].append(result)
                verification = compare_stores(
                    (diagnostics if instrument else {key: value[-1] for key, value in runs.items()})['dense']['store'],
                    (diagnostics if instrument else {key: value[-1] for key, value in runs.items()})['compact']['store'], zs)
            dense_time = statistics.median(item['compute_publish_seconds'] for item in runs['dense'])
            compact_time = statistics.median(item['compute_publish_seconds'] for item in runs['compact'])
            row = {'case': name, 'format': format_name, 'output_shape': shape, 'working_shape': working,
                   'view': view.name, 'face': view.spherical_face, 'rotation': view.spherical_rotation_xyz,
                   'selected_z': zs, 'analytic_bounds': vars(bounds), 'source_logical_bytes': source.nbytes,
                   'source_allocated_bytes': int(source.shape[1] * source.shape[2]),
                   'runs': runs, 'diagnostics': diagnostics, 'verification': verification,
                   'compute_publish_speedup': dense_time / max(compact_time, 1e-12)}
            rows.append(row)
            print(json.dumps({'case': name, 'format': format_name, 'speedup': row['compute_publish_speedup'],
                              'different_voxels': verification['different_voxels']}), flush=True)
    report = {
        'baseline_root': str(args.baseline_root.resolve()), 'thread_limit': 2, 'opencv_threads': 1,
        'compiled_pull_parallel': False, 'repeats': args.repeats, 'small': args.small,
        'compile_or_cache_load_seconds': compile_seconds, 'cases': rows,
        'baseline_projection_sha256': hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
        'candidate_projection_sha256': hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
        'candidate_cpu_sha256': hashlib.sha256(
            Path(candidate.__file__).with_name('spherical_projection_cpu.py').read_bytes()).hexdigest(),
        'baseline_cpu_sha256': hashlib.sha256(Path(baseline_cpu.__file__).read_bytes()).hexdigest(),
        'limitations': ('Synthetic mask broadcast across radii and bounded output slice set; unsampled slices '
                        'marked empty in both stores. Single producer/writer, disk-backed host page cache, '
                        'no fsync. No production pipeline or H100 walltime inference. Separate instrumented '
                        'passes measure Python-visible allocation peaks and scan inputs; tracemalloc does not '
                        'include every allocator and no measured overhead is inserted in timing passes.'),
        'total_different_voxels': sum(row['verification']['different_voxels'] for row in rows),
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': len(rows), 'total_different_voxels': report['total_different_voxels']}), flush=True)


if __name__ == '__main__':
    main()
