"""Compare exact Radial plans and dense/cropped CUDA input on production strides.

Writes evidence and temporary source files only under the required --output-dir.
This is a synthetic component benchmark, not a model/SLURM walltime prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cylindrical_projection as projection, cylindrical_cuda_projection as cuda, geometry
from tools.benchmark_radial_cuda_projection import WORKING_SHAPE, OUTPUT_SHAPE


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def heatsoak(seconds, device):
    import cupy as cp
    with cp.cuda.Device(device):
        a = cp.full((4096, 4096), .001, cp.float32)
        b = cp.empty_like(a)
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            cp.matmul(a, a, out=b)
            cp.cuda.get_current_stream().synchronize()
        del a, b
        cp.get_default_memory_pool().free_all_blocks()
    return time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--base', choices=('transverse', 'sagittal', 'coronal'), default='transverse')
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--full-stream', action='store_true')
    args = parser.parse_args()
    if args.heat_seconds < 0 or args.device < 0:
        parser.error('heat seconds and device must be nonnegative')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {'working_shape': WORKING_SHAPE, 'output_shape': OUTPUT_SHAPE, 'plans': [], 'gpu_runs': [],
              'limits': 'Synthetic rectangular masks; no inference, NRRD compression or cluster concurrency. '
                        'Only --full-stream compares all CUDA output voxels; NumPy checks sampled slabs.'}

    def save():
        (root / 'setup-benchmark.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    for base in ('transverse', 'sagittal', 'coronal'):
        views = geometry.get_view_infos(*WORKING_SHAPE, cartesian_views=(), radial_views=(base,),
                                        radial_patch_size=3072)
        for view in views:
            radii = np.asarray(geometry.radial_global_radii(view))
            row = {'view': view.name, 'runs': []}
            expected = None
            # Alternating order avoids comparing one cold build to one warm build.
            for name in ('reference', 'single_pass', 'single_pass', 'reference'):
                build = (projection._build_radial_plane_plan_reference if name == 'reference'
                         else projection._build_radial_plane_plan)
                started = time.perf_counter()
                plan = build(view, radii, OUTPUT_SHAPE)
                elapsed = time.perf_counter() - started
                checksums = [digest(getattr(plan, field)) for field in
                             ('shell_index', 'column_offsets', 'native_columns')]
                if expected is not None and expected != checksums:
                    raise RuntimeError(f'Plan differs from reference: {view.name}')
                expected = checksums
                row['runs'].append({'builder': name, 'seconds': elapsed, 'plan_bytes': plan.nbytes})
                del plan
            row['sha256'] = expected
            report['plans'].append(row)
            save()
            print(f'Plan {view.name}: {row["runs"]}', flush=True)
    if not args.gpu:
        return
    report['heatsoak_seconds'] = heatsoak(args.heat_seconds, args.device)
    import cupy as cp
    report['gpu'] = str(cp.cuda.runtime.getDeviceProperties(args.device)['name'])
    view = geometry.get_view_infos(*WORKING_SHAPE, cartesian_views=(), radial_views=(args.base,),
                                  radial_patch_size=3072)[0]
    view = replace(view, radial_tilted_source=True, tilt_direction='horizontal', tilt_angle_deg=-30.)
    source_shape = (view.num_slices, view.src_h, view.src_w)
    radii = np.asarray(geometry.radial_global_radii(view))
    plan = projection._build_radial_plane_plan(view, radii, OUTPUT_SHAPE)
    metadata = projection._radial_projection_metadata(view, source_shape, OUTPUT_SHAPE, plan)
    sample_z = sorted(set((0, OUTPUT_SHAPE[0] // 2, OUTPUT_SHAPE[0] - 1)))
    report['source_shape'] = source_shape
    report['fixture'] = ('Each shell except multiples of seven has a 256x768 rectangle with value 255; '
        'row=(shell*13)%(native_height-256), col=(shell*29)%(native_width-768). Others are empty.')
    with tempfile.TemporaryDirectory(prefix='source-', dir=root) as temporary:
        if Path(temporary).resolve().parent != root:
            raise RuntimeError('Benchmark source directory escaped the requested output directory')
        source = np.memmap(Path(temporary) / 'mask.u8', mode='w+', dtype=np.uint8, shape=source_shape)
        try:
            boxes = np.zeros((source_shape[0], 4), np.int64)
            for shell in range(source_shape[0]):
                if shell % 7:
                    y = (shell * 13) % (source_shape[1] - 256)
                    x = (shell * 29) % (source_shape[2] - 768)
                    source[shell, y:y + 256, x:x + 768] = 255
                    boxes[shell] = (y, y + 256, x, x + 768)
            source.flush()
            report['source_strides'] = source.strides
            # Touch every source byte before both representations, so skipped
            # page faults on a sparse file cannot masquerade as a transfer speedup.
            started = time.perf_counter()
            report['source_sum'] = int(source.sum(dtype=np.uint64))
            report['host_cache_warm_seconds'] = time.perf_counter() - started
            expected = {}
            for z in sample_z:
                plane = np.empty(OUTPUT_SHAPE[1:], np.uint8)
                flat = plane.reshape(-1)
                for first in range(0, flat.size, projection._PULL_CHUNK_VOXELS):
                    stop = min(flat.size, first + projection._PULL_CHUNK_VOXELS)
                    flat[first:stop] = projection._pull_radial_chunk(source, view, radii, OUTPUT_SHAPE, z, first, stop)
                expected[z] = digest(plane)
            all_hashes = []
            for use_boxes in (False, True, True, False):
                # Repeat host warming independently of the measured constructor.
                if int(source.sum(dtype=np.uint64)) != report['source_sum']:
                    raise RuntimeError('Source changed during benchmark')
                with cuda.RadialCudaProjector(source, plan, metadata, view, OUTPUT_SHAPE,
                                               boxes, use_boxes, args.device) as projector:
                    row = {name: getattr(projector, name) for name in
                        ('source_layout', 'source_bytes', 'source_h2d_bytes', 'source_upload_seconds',
                         'geometry_upload_seconds', 'preflight_seconds', 'constructor_seconds',
                         'required_device_bytes')}
                    output_hash = hashlib.sha256()
                    byte_count = foreground = 0
                    started = time.perf_counter()
                    starts = (range(0, OUTPUT_SHAPE[0], projector.max_block_depth)
                              if args.full_stream else sample_z)
                    for z in starts:
                        count = min(projector.max_block_depth, OUTPUT_SHAPE[0] - z) if args.full_stream else 1
                        block = projector.project(z, count)
                        output_hash.update(memoryview(block).cast('B'))
                        byte_count += block.nbytes
                        foreground += int(np.count_nonzero(block))
                        for local, plane in enumerate(block):
                            if z + local in expected and digest(plane) != expected[z + local]:
                                raise RuntimeError(f'NumPy reference mismatch at z={z + local}')
                    row.update(project_and_checksum_seconds=time.perf_counter() - started,
                               sha256=output_hash.hexdigest(), output_bytes=byte_count, foreground=foreground)
                    all_hashes.append(row['sha256'])
                    report['gpu_runs'].append(row)
                    save()
                    print(f'CUDA {row}', flush=True)
            if len(set(all_hashes)) != 1:
                raise RuntimeError('Dense/cropped CUDA streams differ')
            report['all_exact'] = True
            save()
        finally:
            source._mmap.close()


if __name__ == '__main__':
    main()
