"""Qualify actual CUDA Radial projection on production-grid synthetic masks.

Uses working (2911,3064,3022), output (1931,3064,3022), and genuine C-strided
3072-square mask frames in a unique temporary file. CPU reference slices are
compared with the CUDA class; --full-stream also exercises the public dispatcher
through a bounded checksum sink, requiring CUDA with no hidden CPU fallback.
No model inference or NRRD/video writing is measured. The GPU is used only when
this program is run; --help does not initialize CUDA.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cylindrical_projection as cp, geometry, backprojection


WORKING_SHAPE = (2911, 3064, 3022)
OUTPUT_SHAPE = (1931, 3064, 3022)


def _save(report, path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def _sample_blocks(indices, maximum):
    first = previous = indices[0]
    for z in indices[1:]:
        if z != previous + 1 or z - first >= maximum:
            yield first, previous - first + 1
            first = z
        previous = z
    yield first, previous - first + 1


def _stage_stats(projector):
    names = ('source_bytes', 'source_layout', 'source_h2d_bytes', 'geometry_bytes', 'output_buffer_bytes', 'required_device_bytes',
             'reserve_bytes', 'device_index', 'max_block_depth', 'source_upload_seconds',
             'source_pack_seconds', 'source_pack_backend', 'contract_validation_seconds',
             'module_setup_seconds', 'buffer_setup_seconds', 'constructor_seconds',
             'geometry_upload_seconds', 'preflight_seconds', 'cuda_graph_setup_seconds',
             'cuda_graph_enabled', 'cuda_graph_note')
    result = {name: getattr(projector, name, None) for name in names}
    result['max_owned_host_output_bytes'] = 2 * int(projector.max_block_depth) * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]
    return result


@contextmanager
def _measure_uploads(projector_type, source_shape, measurements):
    """Time the actual synchronized upload method without replacing its work."""
    original = projector_type._upload_array
    original_cropped = projector_type._upload_cropped_source
    measurements.update(observed_source_upload_seconds=0.0, observed_geometry_upload_seconds=0.0,
        source_upload_scope='Destination allocation, host byte normalization/staging and synchronized H2D; excludes module compilation and kernel/D2H preflight')
    def timed(projector, host):
        started = time.perf_counter()
        try:
            return original(projector, host)
        finally:
            key = ('observed_source_upload_seconds' if tuple(host.shape) == tuple(source_shape)
                   else 'observed_geometry_upload_seconds')
            measurements[key] += time.perf_counter() - started
    def timed_cropped(projector, source):
        started = time.perf_counter()
        try:
            return original_cropped(projector, source)
        finally:
            measurements['observed_source_upload_seconds'] += time.perf_counter() - started
    with mock.patch.object(projector_type, '_upload_array', timed), \
            mock.patch.object(projector_type, '_upload_cropped_source', timed_cropped):
        yield


def _check_block(block, count):
    if (not isinstance(block, np.ndarray) or block.dtype != np.uint8
            or block.shape != (count, *OUTPUT_SHAPE[1:]) or not block.flags.c_contiguous):
        raise RuntimeError('CUDA returned an unexpected host mask representation')
    if int(block.max(initial=0)) > 1:
        raise RuntimeError('CUDA returned a nonbinary mask')


def _gpu_samples(projector, indices, expected, foreground):
    started = time.perf_counter()
    project_seconds = 0.0
    previous = previous_digest = None
    mismatches = []
    for first, count in _sample_blocks(indices, int(projector.max_block_depth)):
        t0 = time.perf_counter()
        block = projector.project(first, count)
        project_seconds += time.perf_counter() - t0
        _check_block(block, count)
        if previous is not None and hashlib.sha256(memoryview(previous).cast('B')).hexdigest() != previous_digest:
            raise RuntimeError('A later CUDA call overwrote an earlier returned host block')
        for dz, plane in enumerate(block):
            z = first + dz
            if (hashlib.sha256(memoryview(plane).cast('B')).hexdigest() != expected[z]
                    or int(np.count_nonzero(plane)) != foreground[z]):
                mismatches.append(z)
        previous = block
        previous_digest = hashlib.sha256(memoryview(block).cast('B')).hexdigest()
    return {'seconds_including_checks': time.perf_counter() - started,
            'project_seconds_including_d2h': project_seconds,
            'exact': not mismatches, 'mismatched_output_z': mismatches,
            'host_block_ownership_checked': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', choices=('transverse', 'sagittal', 'coronal'), default='transverse')
    parser.add_argument('--tilt', type=float, default=0.)
    parser.add_argument('--direction', choices=('vertical', 'horizontal'), default='vertical')
    parser.add_argument('--patch', type=int, default=0)
    parser.add_argument('--device', type=int, default=0, help='Logical CUDA device within the current visible allocation')
    parser.add_argument('--z-starts', default='0,960,1927')
    parser.add_argument('--slab-depth', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--full-stream', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.tilt) or abs(args.tilt) > 45:
        parser.error('--tilt must be finite and within [-45,45]')
    if args.slab_depth < 1 or args.repeats < 1 or args.device < 0:
        parser.error('slab depth/repeats must be positive and device nonnegative')
    starts = [int(z) for z in args.z_starts.split(',')]
    if not starts or any(z < 0 or z >= OUTPUT_SHAPE[0] for z in starts):
        parser.error('--z-starts must select actual source-output slices')
    indices = sorted(set(z for first in starts for z in range(first, min(OUTPUT_SHAPE[0], first + args.slab_depth))))
    views = geometry.get_view_infos(*WORKING_SHAPE, cartesian_views=(), radial_views=(args.base,), radial_patch_size=3072)
    matches = [v for v in views if v.radial_patch_index == args.patch]
    if not matches:
        parser.error(f'Patch {args.patch} does not exist for {args.base}')
    view = matches[0]
    if args.tilt:
        view = replace(view, radial_tilted_source=True, tilt_angle_deg=args.tilt, tilt_direction=args.direction)
    source_shape = (view.num_slices, view.src_h, view.src_w)
    os.environ['YOLO_TTA_GPU_RADIAL_BACKPROJECT'] = '1'
    from XTA import cylindrical_cuda_projection as cuda_projection
    from XTA.cylindrical_cuda_projection import RadialCudaProjector, RadialCudaProjectionUnsafeFailure
    import torch
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError('Requested CUDA device is unavailable; qualification will not fall back to CPU')
    report = {
        'schema': 'xta-radial-cuda-qualification-v1', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'case': {'base': args.base, 'tilt': args.tilt, 'direction': args.direction, 'patch': args.patch},
        'working_shape': list(WORKING_SHAPE), 'output_shape': list(OUTPUT_SHAPE),
        'source_shape': list(source_shape), 'source_logical_bytes': math.prod(source_shape),
        'sampled_output_z': indices, 'sampled_output_voxels': len(indices) * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2],
        'device': {'logical_index': args.device, 'name': torch.cuda.get_device_name(args.device),
                   'visible_count': torch.cuda.device_count(), 'torch_version': torch.__version__},
        'source_code_sha256': {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (cp.__file__, cuda_projection.__file__,
                         Path(geometry.__file__).with_name('cylindrical_geometry.py'), __file__)},
        'fixture_recipe': 'v1: zero-extended new C-strided uint8 file; every shell contains 24x128 rectangles at low/middle/high logical-height rows; column origin=(shell*29+337)%(3072-128)',
        'limits': 'Synthetic sparse masks, not captured model outputs. Source strides and production geometry are real; foreground distribution, filesystem/page-cache locality, and GPU differ from an H100/SLURM production run. Sampled CUDA outputs are independently compared with the NumPy pull oracle. Full-stream mode hashes all output bytes but compares only sampled slices. No inference, NRRD/CVOL encoding, overlays, or whole-pipeline throughput is measured.',
    }
    _save(report, args.output)
    radii = np.asarray(geometry.radial_global_radii(view))
    cp.clear_radial_plane_plan_cache()
    t0 = time.perf_counter()
    plan, _ = cp._radial_plane_plan(view, radii, OUTPUT_SHAPE)
    report['plan_seconds'] = time.perf_counter() - t0
    report['plan_bytes'] = plan.nbytes
    report['plan_occurrences'] = int(plan.native_columns.size)
    metadata = cp._radial_projection_metadata(view, source_shape, OUTPUT_SHAPE, plan)
    report['metadata_bytes'] = sum(getattr(x, 'nbytes', 0) for x in metadata)
    print(f'Production plane plan: {plan.nbytes / 2**20:.2f} MiB, {report["plan_seconds"]:.3f}s', flush=True)
    expected_parent = Path(tempfile.gettempdir()).resolve()
    workspace = Path(tempfile.mkdtemp(prefix='xta-radial-cuda-benchmark-')).resolve()
    if workspace.parent != expected_parent:
        raise RuntimeError('Temporary fixture directory is outside its expected parent')
    report['temporary_workspace'] = str(workspace)
    source = projector = None
    safe_cleanup = True
    try:
        source = np.memmap(workspace / 'source.u8.dat', mode='w+', dtype=np.uint8, shape=source_shape)
        if source.strides != (source_shape[1] * source_shape[2], source_shape[2], 1):
            raise RuntimeError('Fixture does not have genuine 3072-mask C strides')
        stack_length = int(metadata[5])
        anchors = sorted(set((0, max(0, min(source_shape[1] - 24, stack_length // 2 - 12)),
                              max(0, min(source_shape[1] - 24, stack_length - 24)))))
        bounds = np.zeros((source_shape[0], 4), np.int64)
        for shell in range(source_shape[0]):
            x0 = (shell * 29 + 337) % (source_shape[2] - 128)
            for y0 in anchors:
                source[shell, y0:y0 + 24, x0:x0 + 128] = 1
            bounds[shell] = (0, source_shape[1], x0, x0 + 128)
        source.flush()
        report['source_strides'] = list(source.strides)
        report['fixture_row_origins'] = anchors
        expected, foreground = {}, {}
        def reference(z):
            plane = np.empty(OUTPUT_SHAPE[1:], np.uint8)
            flat = plane.reshape(-1)
            for first in range(0, flat.size, cp._PULL_CHUNK_VOXELS):
                stop = min(flat.size, first + cp._PULL_CHUNK_VOXELS)
                flat[first:stop] = cp._pull_radial_chunk(source, view, radii, OUTPUT_SHAPE, z, first, stop)
            return plane
        t0 = time.perf_counter()
        for z in indices:
            plane = reference(z)
            expected[z] = hashlib.sha256(memoryview(plane).cast('B')).hexdigest()
            foreground[z] = int(np.count_nonzero(plane))
            del plane
        controls = [time.perf_counter() - t0]
        if not any(foreground.values()):
            raise RuntimeError('All selected reference slices are empty; choose slices that exercise foreground gathers')
        report['decoded_reference_sha256'] = expected
        report['foreground_voxels_by_z'] = foreground
        print(f'CPU reference before: {controls[0]:.3f}s', flush=True)
        t0 = time.perf_counter()
        upload_measurements = {}
        with _measure_uploads(RadialCudaProjector, source_shape, upload_measurements):
            projector = RadialCudaProjector(source, plan, metadata, view, OUTPUT_SHAPE, bounds, True, args.device)
        report['cuda_constructor_upload_preflight_seconds'] = time.perf_counter() - t0
        report['cuda_stage'] = {**_stage_stats(projector), **upload_measurements}
        print(f'CUDA source ready: constructor/upload/preflight {report["cuda_constructor_upload_preflight_seconds"]:.3f}s', flush=True)
        results = []
        report['cuda_samples'] = results
        _save(report, args.output)
        for repeat in range(args.repeats):
            measured = _gpu_samples(projector, indices, expected, foreground)
            measured['repeat'] = repeat
            results.append(measured)
            _save(report, args.output)
            print(f'CUDA sample run {repeat}: {measured["seconds_including_checks"]:.3f}s, exact={measured["exact"]}', flush=True)
            if not measured['exact']:
                raise RuntimeError(f'CUDA output differs at source-Z slices {measured["mismatched_output_z"]}')
        projector.close()
        projector = None
        t0 = time.perf_counter()
        for z in indices:
            plane = reference(z)
            if (hashlib.sha256(memoryview(plane).cast('B')).hexdigest() != expected[z]
                    or int(np.count_nonzero(plane)) != foreground[z]):
                raise RuntimeError('Repeated CPU oracle changed')
            del plane
        controls.append(time.perf_counter() - t0)
        report['cpu_reference_seconds_including_checks'] = controls
        report['cuda_samples'] = results
        for measured in results:
            measured['speedup_excluding_constructor_vs_fastest_reference'] = min(controls) / measured['seconds_including_checks']
        _save(report, args.output)
        if args.full_stream:
            # This is a standalone process with no inference. Configure exactly
            # the explicitly chosen visible device, then test the real dispatcher.
            backprojection._configure_main_process_gpu_stage_workers([args.device])
            backprojection._set_main_process_gpu_inference_priority_active(False)
            backprojection._set_main_process_gpu_asset_retirement_pending(False)
            backprojection._set_main_process_gpu_pending_inference(False)
            original_admit = cp._try_radial_cuda_stage
            admissions = []
            def require_cuda(*values, **options):
                start = time.perf_counter()
                uploads = {}
                with _measure_uploads(RadialCudaProjector, source_shape, uploads):
                    stage = original_admit(*values, **options)
                if stage is None:
                    raise RuntimeError('Public dispatcher declined CUDA; hidden CPU fallback is forbidden')
                if not isinstance(stage.projector, RadialCudaProjector) or stage.device_index != args.device:
                    stage.close()
                    raise RuntimeError('Public dispatcher did not use the requested actual CUDA projector')
                admissions.append({'admission_upload_preflight_seconds': time.perf_counter() - start,
                                   **_stage_stats(stage.projector), **uploads})
                return stage
            state = {'next_z': 0, 'bytes': 0, 'foreground': 0, 'sample_mismatches': []}
            digest = hashlib.sha256()
            def consume(first, block):
                _check_block(block, len(block))
                if first != state['next_z']:
                    raise RuntimeError('Public CUDA stream duplicated, omitted, or reordered source-Z slices')
                for dz, plane in enumerate(block):
                    raw = memoryview(plane).cast('B')
                    digest.update(raw)
                    z = first + dz
                    if z in expected and hashlib.sha256(raw).hexdigest() != expected[z]:
                        state['sample_mismatches'].append(z)
                    state['bytes'] += len(raw)
                    state['foreground'] += int(np.count_nonzero(plane))
                state['next_z'] += len(block)
            t0 = time.perf_counter()
            with (mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=require_cuda),
                  mock.patch.object(cp, '_project_radial_block', side_effect=AssertionError('CPU compiled fallback entered')),
                  mock.patch.object(cp, '_pull_radial_chunk', side_effect=AssertionError('CPU reference fallback entered'))):
                cp.backproject_radial_volume_to_volume(
                    source, view, workspace / 'must-not-exist-output.dat', 'CUDA full-stream qualification',
                    workers=1, out_shape_tyx=OUTPUT_SHAPE, known_slice_bboxes=bounds,
                    sink_only=True, projection_block_callback=consume,
                )
            if (len(admissions) != 1 or state['next_z'] != OUTPUT_SHAPE[0]
                    or state['bytes'] != math.prod(OUTPUT_SHAPE) or state['sample_mismatches']
                    or (workspace / 'must-not-exist-output.dat').exists()):
                raise RuntimeError(f'Public CUDA full-stream qualification failed: {state}')
            report['full_cuda_stream'] = {**state, 'seconds_including_upload_preflight_and_checks': time.perf_counter() - t0,
                'sha256': digest.hexdigest(), 'backend': 'cuda_factored', 'admissions': admissions,
                'limits': 'All output bytes were streamed through the public CUDA dispatcher; only sampled slices were independently compared with the CPU oracle. The sink hashes/counts and does not encode NRRDs.'}
            print(f'Public full CUDA stream: {report["full_cuda_stream"]["seconds_including_upload_preflight_and_checks"]:.3f}s, samples exact', flush=True)
        report['status'] = 'complete'
        report['all_sample_comparisons_exact'] = True
    except RadialCudaProjectionUnsafeFailure as exc:
        safe_cleanup = False
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      cleanup_deferred='CUDA stream ownership is uncertain; preserve source fixture until this process has exited')
        raise
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if projector is not None:
            try:
                projector.close()
            except BaseException as exc:
                safe_cleanup = False
                report.update(status='failed', cleanup_error=f'{type(exc).__name__}: {exc}',
                              cleanup_deferred='CUDA stream could not be settled; source fixture retained until process exit')
        if safe_cleanup:
            if source is not None:
                source._mmap.close()
            if workspace.parent != expected_parent or not workspace.name.startswith('xta-radial-cuda-benchmark-'):
                raise RuntimeError('Refusing cleanup outside the unique checked benchmark workspace')
            shutil.rmtree(workspace)
            report['fixture_removed'] = True
        else:
            report['fixture_removed'] = False
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        _save(report, args.output)
        print(json.dumps(report, indent=2), flush=True)
        if not safe_cleanup:
            raise RuntimeError('CUDA qualification cleanup was not safe; fixture retained as recorded')


if __name__ == '__main__':
    main()
