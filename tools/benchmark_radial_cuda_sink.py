"""Compare dense and compact CUDA output through the real incremental CVOL writer.

Defaults to full production-grid transverse projection. The source is a unique
11 GiB C-strided synthetic uint8 file; no reduced/broadcast source is substituted.
GPU upload, real callback writes and finalization are timed. Afterwards each CVOL
is decoded through a bounded full-slice buffer to SHA256 and foreground counts.
No model inference, NRRD/gzip encoding, or cluster throughput is measured.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from XTA import cylindrical_projection as cp, geometry, backprojection, interpolation
from tools.benchmark_radial_cuda_projection import WORKING_SHAPE, OUTPUT_SHAPE, _measure_uploads, _stage_stats, _sample_blocks


RECIPE = 'v1: zero-extended new C-strided uint8 file; every shell contains 24x128 rectangles at low/middle/high logical-height rows; column origin=(shell*29+337)%(3072-128)'
EVENTS = ('kernel_seconds', 'metadata_seconds', 'pack_seconds', 'd2h_seconds', 'cuda_graph_seconds')
TRANSFERS = ('metadata_d2h_bytes', 'payload_d2h_bytes', 'dense_d2h_bytes')


class DenseSink:
    """Intentionally lacks encoded_slice_format and consume_encoded_block."""
    def __init__(self, writer):
        self.writer = writer
        self.next_z = 0
        self.sink_seconds = 0.0
        self.dense_calls = self.encoded_calls = self.placeholder_slices = 0

    def __call__(self, first, block):
        if first != self.next_z:
            raise RuntimeError('Dense callback reordered or duplicated source slices')
        started = time.perf_counter()
        try:
            self.writer(first, block)
        finally:
            self.sink_seconds += time.perf_counter() - started
        self.next_z += len(block)
        self.dense_calls += 1

    def consume_empty_range(self, first, count):
        if first != self.next_z:
            raise RuntimeError('Placeholder range reordered source slices')
        started = time.perf_counter()
        try:
            self.writer.consume_empty_range(first, count)
        finally:
            self.sink_seconds += time.perf_counter() - started
        self.next_z += count
        self.placeholder_slices += count


class CompactSink(DenseSink):
    @property
    def encoded_slice_format(self):
        return self.writer.encoded_slice_format

    def __call__(self, first, block):
        raise RuntimeError('Compact mode unexpectedly received a dense host raster')

    def consume_encoded_block(self, first, records, payload, *, packed):
        if first != self.next_z:
            raise RuntimeError('Encoded callback reordered or duplicated source slices')
        started = time.perf_counter()
        try:
            self.writer.consume_encoded_block(first, records, payload, packed=packed)
        finally:
            self.sink_seconds += time.perf_counter() - started
        self.next_z += len(records)
        self.encoded_calls += 1


def metrics(projector):
    # Missing counters are an incompatible backend, never silently reported zero.
    return {name: getattr(projector, name) for name in EVENTS + TRANSFERS + ('cuda_graph_blocks', 'empty_encoded_blocks')}


def save(report, path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def decode_store(path):
    started = time.perf_counter()
    store = interpolation.RawBBoxMaskStore.open(path, mmap_payload=True)
    try:
        if tuple(store.shape) != OUTPUT_SHAPE:
            raise RuntimeError('CVOL output geometry differs from production geometry')
        plane = np.empty(OUTPUT_SHAPE[1:], np.uint8)
        digest = hashlib.sha256()
        foreground = byte_count = 0
        for z in range(OUTPUT_SHAPE[0]):
            store.fill_decoded_slice_into(z, plane)
            if int(plane.max(initial=0)) > 1:
                raise RuntimeError('Decoded CVOL contains nonbinary values')
            digest.update(memoryview(plane).cast('B'))
            foreground += int(np.count_nonzero(plane))
            byte_count += int(plane.nbytes)
        return {'sha256': digest.hexdigest(), 'foreground': foreground, 'bytes': byte_count,
                'seconds': time.perf_counter() - started, 'shape': list(OUTPUT_SHAPE)}
    finally:
        store.close()


def prior_targets(args, report):
    paths = list(args.prior_json or ())
    if not paths and report['scope'] == 'full' and report['case'] == {
        'base': 'transverse', 'tilt': 0.0, 'direction': 'vertical', 'patch': 0,
    }:
        candidate = ROOT / 'build/v20-cuda/transverse-full.json'
        if candidate.is_file():
            paths.append(candidate)
    targets, skipped = [], []
    for path in paths:
        path = Path(path).resolve()
        if args.output is not None and path == args.output.resolve():
            raise ValueError('Benchmark output must not overwrite its prior comparison artifact')
        raw = path.read_bytes()
        prior = json.loads(raw)
        if (report['scope'] != 'full' or prior.get('case') != report['case']
                or prior.get('working_shape') != list(WORKING_SHAPE)
                or prior.get('output_shape') != list(OUTPUT_SHAPE)
                or prior.get('source_shape') != report['source_shape']
                or prior.get('fixture_recipe') != RECIPE):
            skipped.append({'path': str(path), 'reason': 'geometry/case/fixture identity is not established as equal'})
            continue
        entries = [prior.get('full_cuda_stream'), prior.get('full_compiled_stream')]
        entries += [run.get('decoded') for run in prior.get('runs', []) if prior.get('scope') == 'full']
        for entry in entries:
            if entry and entry.get('sha256') and entry.get('bytes') == math.prod(OUTPUT_SHAPE):
                targets.append({'path': str(path), 'artifact_sha256': hashlib.sha256(raw).hexdigest(),
                                'sha256': entry['sha256'], 'foreground': entry['foreground']})
    return targets, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dense', 'compact', 'both'), default='both')
    parser.add_argument('--format', choices=('raw', 'packed'), default='raw')
    parser.add_argument('--base', choices=('transverse', 'sagittal', 'coronal'), default='transverse')
    parser.add_argument('--tilt', type=float, default=0.)
    parser.add_argument('--direction', choices=('vertical', 'horizontal'), default='vertical')
    parser.add_argument('--patch', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full-stream', action='store_true', help='Stream all source-Z slices (the default)')
    group.add_argument('--z-starts', help='Selected slabs only; unselected output slices are explicitly recorded empty')
    parser.add_argument('--slab-depth', type=int, default=4)
    parser.add_argument('--prior-json', type=Path, action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.device < 0 or args.slab_depth < 1 or not math.isfinite(args.tilt) or abs(args.tilt) > 45:
        parser.error('Device must be nonnegative, slab depth positive, and tilt within [-45,45]')
    selected = None
    if args.z_starts:
        starts = [int(value) for value in args.z_starts.split(',')]
        if any(z < 0 or z >= OUTPUT_SHAPE[0] for z in starts):
            parser.error('Selected slabs must be inside the production output')
        selected = sorted(set(z for start in starts for z in range(start, min(OUTPUT_SHAPE[0], start + args.slab_depth))))
    views = geometry.get_view_infos(*WORKING_SHAPE, cartesian_views=(), radial_views=(args.base,), radial_patch_size=3072)
    found = [view for view in views if view.radial_patch_index == args.patch]
    if not found:
        parser.error('Requested intrinsic patch does not exist')
    view = found[0]
    if args.tilt:
        view = replace(view, radial_tilted_source=True, tilt_angle_deg=args.tilt, tilt_direction=args.direction)
    source_shape = (view.num_slices, view.src_h, view.src_w)
    os.environ['YOLO_TTA_GPU_RADIAL_BACKPROJECT'] = '1'
    from XTA import cylindrical_cuda_projection as gpu
    import torch
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError('Requested CUDA device is unavailable; CPU fallback is forbidden')
    report = {
        'schema': 'xta-radial-cuda-cvol-sink-v1', 'status': 'running', 'scope': 'selected' if selected else 'full',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'case': {'base': args.base, 'tilt': args.tilt, 'direction': args.direction, 'patch': args.patch},
        'format': args.format, 'working_shape': list(WORKING_SHAPE), 'output_shape': list(OUTPUT_SHAPE),
        'source_shape': list(source_shape), 'source_logical_bytes': math.prod(source_shape),
        'source_strides': [source_shape[1] * source_shape[2], source_shape[2], 1],
        'fixture_recipe': RECIPE, 'selected_output_z': selected,
        'device': {'logical_index': args.device, 'name': torch.cuda.get_device_name(args.device)},
        'source_code_sha256': {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (cp.__file__, gpu.__file__, interpolation.__file__,
                         Path(geometry.__file__).with_name('cylindrical_geometry.py'), __file__)},
        'limits': 'Production geometry and real 3072 C-strided synthetic masks; real incremental CVOL writes, not a checksum-only sink. GPU upload/projection/D2H/callback/finalization are timed; full CVOL decoding is separately timed afterwards. No NRRD/gzip encoding, video/overlay generation, model inference or complete SLURM workload is measured. Sparse fixture and page-cache behavior differ from production. Selected-slab mode deliberately records other slices empty and cannot qualify the full projection.',
        'runs': [],
    }
    targets, skipped = prior_targets(args, report)
    report['prior_comparison_targets'], report['skipped_prior_comparisons'] = targets, skipped
    save(report, args.output)
    radii = np.asarray(geometry.radial_global_radii(view))
    cp.clear_radial_plane_plan_cache()
    plan_started = time.perf_counter()
    plan, _ = cp._radial_plane_plan(view, radii, OUTPUT_SHAPE)
    metadata = cp._radial_projection_metadata(view, source_shape, OUTPUT_SHAPE, plan)
    report['host_plan_setup_seconds'] = time.perf_counter() - plan_started
    report['plan_bytes'] = plan.nbytes
    expected_parent = Path(tempfile.gettempdir()).resolve()
    workspace = Path(tempfile.mkdtemp(prefix='xta-radial-cuda-sink-')).resolve()
    if workspace.parent != expected_parent:
        raise RuntimeError('Unexpected temporary fixture parent')
    source = active_writer = active_stage = None
    safe_cleanup = True
    report['temporary_workspace'] = str(workspace)
    try:
        source = np.memmap(workspace / 'source.u8.dat', mode='w+', dtype=np.uint8, shape=source_shape)
        if list(source.strides) != report['source_strides']:
            raise RuntimeError('Fixture is not truly C-strided')
        length = int(metadata[5])
        anchors = sorted(set((0, max(0, min(source_shape[1] - 24, length // 2 - 12)),
                              max(0, min(source_shape[1] - 24, length - 24)))))
        bounds = np.zeros((source_shape[0], 4), np.int64)
        for shell in range(source_shape[0]):
            x0 = (shell * 29 + 337) % (source_shape[2] - 128)
            for y0 in anchors:
                source[shell, y0:y0 + 24, x0:x0 + 128] = 1
            bounds[shell] = (0, source_shape[1], x0, x0 + 128)
        source.flush()
        report['fixture_row_origins'] = anchors
        backprojection._configure_main_process_gpu_stage_workers([args.device])
        backprojection._set_main_process_gpu_inference_priority_active(False)
        backprojection._set_main_process_gpu_asset_retirement_pending(False)
        backprojection._set_main_process_gpu_pending_inference(False)
        modes = ('dense', 'compact') if args.mode == 'both' else (args.mode,)
        fmt = interpolation.CVOL_FORMAT if args.format == 'raw' else interpolation.INTERNAL_PACKED_CVOL_FORMAT
        original_admit = cp._try_radial_cuda_stage
        for ordinal, mode in enumerate(modes):
            store_path = workspace / f'{ordinal}-{mode}-{args.format}.cvol'
            active_writer = interpolation.IncrementalRawBBoxMaskStoreWriter(
                shape=OUTPUT_SHAPE, store_dir=store_path, format_name=fmt, desc=f'{mode} CUDA real CVOL sink',
            )
            sink = DenseSink(active_writer) if mode == 'dense' else CompactSink(active_writer)
            admitted = []
            def require_cuda(*values, **options):
                started = time.perf_counter()
                uploads = {}
                with _measure_uploads(gpu.RadialCudaProjector, source_shape, uploads):
                    stage = original_admit(*values, **options)
                if stage is None:
                    raise RuntimeError('Actual CUDA admission failed; CPU fallback is forbidden')
                if stage.device_index != args.device or not isinstance(stage.projector, gpu.RadialCudaProjector):
                    stage.close()
                    raise RuntimeError('Wrong CUDA device or projector admitted')
                admitted.append({'stage': stage, 'before': metrics(stage.projector),
                    'admission_upload_preflight_seconds': time.perf_counter() - started,
                    **_stage_stats(stage.projector), **uploads})
                return stage
            started = time.perf_counter()
            if selected is None:
                with (mock.patch.object(cp, '_try_radial_cuda_stage', side_effect=require_cuda),
                      mock.patch.object(cp, '_project_radial_block', side_effect=AssertionError('CPU fallback entered')),
                      mock.patch.object(cp, '_pull_radial_chunk', side_effect=AssertionError('CPU fallback entered'))):
                    cp.backproject_radial_volume_to_volume(
                        source, view, workspace / 'must-not-exist.dat', f'{mode} CUDA sink benchmark',
                        out_shape_tyx=OUTPUT_SHAPE, known_slice_bboxes=bounds, workers=1,
                        sink_only=True, projection_block_callback=sink,
                    )
            else:
                active_stage = require_cuda(source, plan, metadata, view, OUTPUT_SHAPE, bounds, True)
                for first, count in _sample_blocks(selected, active_stage.max_block_depth):
                    if first > sink.next_z:
                        sink.consume_empty_range(sink.next_z, first - sink.next_z)
                    if mode == 'dense':
                        block = active_stage.project(first, count)
                        sink(first, block)
                    else:
                        block = active_stage.project_encoded(first, count, packed=args.format == 'packed')
                        sink.consume_encoded_block(block.first_z, block.records, block.payload, packed=block.packed)
                    del block
                if sink.next_z < OUTPUT_SHAPE[0]:
                    sink.consume_empty_range(sink.next_z, OUTPUT_SHAPE[0] - sink.next_z)
                active_stage.close()
                active_stage = None
            projection_seconds = time.perf_counter() - started
            if (len(admitted) != 1 or sink.next_z != OUTPUT_SHAPE[0]
                    or (mode == 'dense' and (not sink.dense_calls or sink.encoded_calls))
                    or (mode == 'compact' and (not sink.encoded_calls or sink.dense_calls))):
                raise RuntimeError('Requested dense/compact CUDA path did not execute exactly once')
            final_started = time.perf_counter()
            stats = active_writer.finalize()
            finalize_seconds = time.perf_counter() - final_started
            active_writer = None
            admission = admitted[0]
            after = metrics(admission['stage'].projector)
            device_metrics = {name: after[name] - admission['before'][name] for name in EVENTS + TRANSFERS}
            result = {'mode': mode, 'format': args.format, 'backend': 'cuda_factored' if mode == 'dense' else 'cuda_factored_compact',
                'projection_upload_sink_cleanup_seconds': projection_seconds,
                'sink_callback_seconds': sink.sink_seconds, 'finalize_seconds': finalize_seconds,
                'total_timed_seconds': projection_seconds + finalize_seconds,
                'dense_callbacks': sink.dense_calls, 'encoded_callbacks': sink.encoded_calls,
                'placeholder_slices': sink.placeholder_slices,
                'cuda_events_and_transfer_deltas': device_metrics,
                'actual_total_d2h_bytes': sum(int(device_metrics[name]) for name in TRANSFERS),
                'admission': {k: v for k, v in admission.items() if k not in ('stage', 'before')},
                'cvol_stats': stats,
                'cvol_total_file_bytes': sum(p.stat().st_size for p in store_path.iterdir() if p.is_file())}
            print(f'{mode} real sink: {result["total_timed_seconds"]:.3f}s, '
                  f'sink={sink.sink_seconds:.3f}s, D2H={result["actual_total_d2h_bytes"]:,} bytes, '
                  f'CVOL={stats["raw_payload_bytes"]:,} bytes', flush=True)
            report['runs'].append(result)
            save(report, args.output)
            result['decoded'] = decode_store(store_path)
            if (result['decoded']['bytes'] != math.prod(OUTPUT_SHAPE)
                    or result['decoded']['foreground'] != stats['foreground_voxels']):
                raise RuntimeError('Decoded CVOL length/count differs from published GPU metadata')
            result['prior_matches'] = [result['decoded']['sha256'] == t['sha256']
                and result['decoded']['foreground'] == t['foreground'] for t in targets]
            if targets and not all(result['prior_matches']):
                raise RuntimeError('Decoded CVOL differs from an independently recorded matching fixture')
            save(report, args.output)
            admitted.clear()
        if len(report['runs']) == 2:
            left, right = report['runs']
            report['dense_compact_decoded_equal'] = (left['decoded']['sha256'] == right['decoded']['sha256']
                and left['decoded']['foreground'] == right['decoded']['foreground'])
            if not report['dense_compact_decoded_equal']:
                raise RuntimeError('Dense and compact CVOL decoded outputs differ')
            report['dense_to_compact_total_time_ratio'] = left['total_timed_seconds'] / right['total_timed_seconds']
            report['dense_to_compact_d2h_ratio'] = left['actual_total_d2h_bytes'] / max(1, right['actual_total_d2h_bytes'])
        report['independent_comparison_available'] = bool(targets or len(report['runs']) == 2)
        report['status'] = 'complete'
    except gpu.RadialCudaProjectionUnsafeFailure as exc:
        safe_cleanup = False
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      cleanup_deferred='CUDA ownership uncertain; retain fixture until process exit')
        raise
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if active_writer is not None:
            active_writer.abort(exc)
        raise
    finally:
        if active_stage is not None:
            try:
                active_stage.close()
            except BaseException as exc:
                safe_cleanup = False
                report.update(status='failed', cleanup_error=f'{type(exc).__name__}: {exc}')
        if safe_cleanup:
            if active_writer is not None:
                active_writer.discard()
            if source is not None:
                source._mmap.close()
            if workspace.parent != expected_parent or not workspace.name.startswith('xta-radial-cuda-sink-'):
                raise RuntimeError('Refusing cleanup outside checked owned benchmark directory')
            shutil.rmtree(workspace)
            report['temporary_files_removed'] = True
        else:
            report['temporary_files_removed'] = False
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(report, args.output)
        print(json.dumps(report, indent=2), flush=True)
        if not safe_cleanup:
            raise RuntimeError('CUDA cleanup was unsafe; fixture retained as recorded')


if __name__ == '__main__':
    main()
