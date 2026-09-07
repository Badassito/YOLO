"""Production-grid CPU slab comparison; no GPU, model or full output allocation.

The temporary source has the real C-contiguous mask strides and logical size.
Only deterministic sparse rectangles are written. Remaining bytes are zero in
the new path-backed file. Reported times are isolated synthetic slab evidence,
not a cluster qualification. An optional bounded full-output stream measures the
compiled projector with a checksum sink. Run a separate invocation per case.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cylindrical_projection as cp, geometry


def main():
    # This remains the CPU qualification tool even when CUDA is available.
    os.environ['YOLO_TTA_GPU_RADIAL_BACKPROJECT'] = '0'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', choices=('transverse', 'sagittal', 'coronal'), default='transverse')
    parser.add_argument('--tilt', type=float, default=0.)
    parser.add_argument('--direction', choices=('vertical', 'horizontal'), default='vertical')
    parser.add_argument('--patch', type=int, default=0)
    parser.add_argument('--workers', default='1,4')
    parser.add_argument('--z-starts', default='0,960,1927')
    parser.add_argument('--slab-depth', type=int, default=4)
    parser.add_argument('--full-compiled-stream', action='store_true',
                        help='Also stream the entire native output through a bounded SHA256 sink using the largest requested worker count')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output_shape = (1931, 3064, 3022)
    view = next(v for v in geometry.get_view_infos(
        2911, 3064, 3022, cartesian_views=(), radial_views=(args.base,),
        radial_patch_size=3072,
    ) if v.radial_patch_index == args.patch)
    if args.tilt:
        view = replace(view, radial_tilted_source=True, tilt_angle_deg=args.tilt, tilt_direction=args.direction)
    source_shape = (view.num_slices, view.src_h, view.src_w)
    samples = sorted(set(z for start in map(int, args.z_starts.split(','))
                         for z in range(start, min(output_shape[0], start + args.slab_depth))))
    if not samples or min(samples) < 0:
        raise ValueError('Requested source-Z slabs are outside the production output')
    report = {
        'projection_source_sha256': hashlib.sha256(Path(cp.__file__).read_bytes()).hexdigest(),
        'case': {'base': args.base, 'tilt': args.tilt, 'direction': args.direction, 'patch': args.patch},
        'working_shape': [2911, 3064, 3022], 'output_shape': list(output_shape),
        'source_shape': list(source_shape), 'source_logical_bytes': math.prod(source_shape),
        'sampled_output_z': samples, 'sampled_output_voxels': len(samples) * output_shape[1] * output_shape[2],
        'limits': 'Synthetic sparse rectangles in a full-stride path-backed source; page cache and sampled-Z locality differ from production masks. No GPU, NRRD writer, full-volume timing, or SLURM allocation was exercised.',
    }
    radii = np.asarray(geometry.radial_global_radii(view))
    cp.clear_radial_plane_plan_cache()
    t0 = time.perf_counter()
    plan, _ = cp._radial_plane_plan(view, radii, output_shape)
    report['plan_seconds'] = time.perf_counter() - t0
    report['plan_bytes'] = plan.nbytes
    report['plan_occurrences'] = int(plan.native_columns.size)
    print(f'Plan ready: {plan.nbytes / 2**20:.2f} MiB in {report["plan_seconds"]:.3f}s', flush=True)
    metadata = cp._radial_projection_metadata(view, source_shape, output_shape, plan)
    centers, ideal, sampled, row_map, column_map, stack_length, vertical = metadata
    report['metadata_bytes'] = sum(getattr(x, 'nbytes', 0) for x in metadata)
    temporary = tempfile.TemporaryDirectory(prefix='xta-radial-slab-benchmark-')
    workspace = Path(temporary.name).resolve()
    expected_parent = Path(tempfile.gettempdir()).resolve()
    if workspace.parent != expected_parent:
        raise RuntimeError('Benchmark temporary workspace resolved outside its expected parent')
    source = None
    try:
        source = np.memmap(workspace / 'source.u8.dat', mode='w+', dtype=np.uint8, shape=source_shape)
        # Real C strides, with sparse nonempty rectangles in every shell. Include
        # low/middle/high native rows rather than a single favorable axial band.
        bounds = np.zeros((source_shape[0], 4), np.int64)
        anchors = sorted(set((0, max(0, min(source_shape[1] - 24, stack_length // 2 - 12)),
                              max(0, min(source_shape[1] - 24, stack_length - 24)))))
        for shell in range(source_shape[0]):
            x0 = (shell * 29 + 337) % (source_shape[2] - 128)
            for y0 in anchors:
                source[shell, y0:y0 + 24, x0:x0 + 128] = 1
            bounds[shell] = (0, source_shape[1], x0, x0 + 128)
        source.flush()
        report['source_strides'] = list(source.strides)
        arguments = (
            source, plan.shell_index, plan.column_offsets, plan.native_columns, sampled,
            row_map, column_map, centers, ideal, stack_length, view.radial_height_origin,
            view.src_h, plan.base_id, vertical, plan.plane_shape[1], output_shape[1], output_shape[2],
        )
        t0 = time.perf_counter()
        cp._project_radial_block(*arguments, 0, 0, bounds, False)
        report['jit_warmup_seconds'] = time.perf_counter() - t0

        def reference(z):
            plane = np.empty(output_shape[1:], np.uint8)
            flat = plane.reshape(-1)
            for first in range(0, flat.size, cp._PULL_CHUNK_VOXELS):
                stop = min(flat.size, first + cp._PULL_CHUNK_VOXELS)
                flat[first:stop] = cp._pull_radial_chunk(source, view, radii, output_shape, z, first, stop)
            return plane

        expected = {}
        foreground = {}
        controls = []
        t0 = time.perf_counter()
        for z in samples:
            value = reference(z)
            expected[z] = hashlib.sha256(value.tobytes()).hexdigest()
            foreground[z] = int(np.count_nonzero(value))
            del value
        controls.append(time.perf_counter() - t0)
        print(f'Reference before: {controls[-1]:.3f}s', flush=True)
        results = []
        for use_bounds in (False, True):
            for workers in map(int, args.workers.split(',')):
                def project(z):
                    return z, cp._project_radial_block(*arguments, z, 1, bounds, use_bounds)
                t0 = time.perf_counter()
                mismatches = []
                # Submit at most one output plane per worker. Consume each before
                # scheduling another wave; no list of full slab arrays is retained.
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for first in range(0, len(samples), workers):
                        futures = [pool.submit(project, z) for z in samples[first:first + workers]]
                        for future in futures:
                            z, value = future.result()
                            if hashlib.sha256(value.tobytes()).hexdigest() != expected[z]:
                                mismatches.append(z)
                            if int(np.count_nonzero(value)) != foreground[z]:
                                mismatches.append(z)
                            del value
                        del futures
                seconds = time.perf_counter() - t0
                result = {'workers': workers, 'source_bounds': use_bounds, 'seconds_including_sha256': seconds,
                          'exact': not mismatches, 'mismatched_output_z': mismatches}
                results.append(result)
                print(f'Compiled workers={workers}, bounds={use_bounds}: {seconds:.3f}s, exact={not mismatches}', flush=True)
                if mismatches:
                    raise RuntimeError(f'Production-grid numerical mismatch: {mismatches}')
        t0 = time.perf_counter()
        for z in samples:
            value = reference(z)
            if hashlib.sha256(value.tobytes()).hexdigest() != expected[z]:
                raise RuntimeError('Repeated reference changed')
            if int(np.count_nonzero(value)) != foreground[z]:
                raise RuntimeError('Repeated reference foreground changed')
            del value
        controls.append(time.perf_counter() - t0)
        print(f'Reference after: {controls[-1]:.3f}s', flush=True)
        report['reference_seconds_including_sha256'] = controls
        report['compiled'] = results
        for result in results:
            result['speedup_vs_fastest_reference'] = min(controls) / result['seconds_including_sha256']
        report['all_exact'] = all(x['exact'] for x in results)
        report['max_output_inflight_bytes'] = max(map(int, args.workers.split(','))) * output_shape[1] * output_shape[2]
        report['decoded_output_sha256'] = expected
        report['foreground_voxels_by_z'] = foreground
        if args.full_compiled_stream:
            report['limits'] = ('Synthetic sparse rectangles in a full-stride path-backed source; '
                'page cache and sample locality differ from production masks. Full compiled output '
                'streaming is measured below, with independent reference comparisons on sampled slices only. '
                'No GPU, model inference, NRRD writer or SLURM allocation was exercised.')
            state = {'next_z': 0, 'bytes': 0, 'foreground': 0, 'sample_mismatches': []}
            digest = hashlib.sha256()
            def consume(z0, block):
                if z0 != state['next_z']:
                    raise RuntimeError('Full projection output is out of order or duplicated')
                for dz, plane in enumerate(block):
                    raw = memoryview(np.ascontiguousarray(plane)).cast('B')
                    digest.update(raw)
                    z = z0 + dz
                    if z in expected and hashlib.sha256(raw).hexdigest() != expected[z]:
                        state['sample_mismatches'].append(z)
                    state['bytes'] += len(raw)
                    state['foreground'] += int(np.count_nonzero(plane))
                state['next_z'] += len(block)
            t0 = time.perf_counter()
            cp.backproject_radial_volume_to_volume(
                source, view, workspace / 'unused-output.dat', 'full benchmark stream',
                workers=max(map(int, args.workers.split(','))), out_shape_tyx=output_shape,
                known_slice_bboxes=bounds, sink_only=True, projection_block_callback=consume,
            )
            full_seconds = time.perf_counter() - t0
            if state['next_z'] != output_shape[0] or state['bytes'] != math.prod(output_shape) or state['sample_mismatches']:
                raise RuntimeError(f'Full projection coverage/checksum failure: {state}')
            report['full_compiled_stream'] = {
                **state, 'seconds_including_sha256': full_seconds,
                'sha256': digest.hexdigest(), 'workers': max(map(int, args.workers.split(','))),
                'limits': 'Every output voxel was streamed; only sampled slices were compared with the independent reference. NRRD encoding and model inference were not measured.',
            }
            print(f'Full compiled stream: {full_seconds:.3f}s, {state["bytes"]} bytes, sampled slices exact', flush=True)
    finally:
        if source is not None:
            source._mmap.close()
        # Only remove the unique disposable workspace checked above.
        if workspace.parent != expected_parent or not workspace.name.startswith('xta-radial-slab-benchmark-'):
            raise RuntimeError('Refusing cleanup outside owned benchmark workspace')
        temporary.cleanup()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
