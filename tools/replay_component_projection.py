"""Replay a captured view-native component without rerunning model inference.

Each backend runs in a fresh process. Default runs are CPU-only; CUDA is queried
only with --cuda-reference. Numerical comparison reads one output slice at a
time. Both projection workspaces use one local temporary root; logs and metrics
persist separately under --output. Workspaces are deleted unless requested.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


@contextlib.contextmanager
def _gpu_reference_tracking(enabled):
    from XTA import backprojection
    old_env = os.environ.get('YOLO_TTA_GPU_BACKPROJECT')
    os.environ['YOLO_TTA_GPU_BACKPROJECT'] = '1' if enabled else '0'
    selected = []
    originals = {}
    for name in ('_radial_backproject_gpu_resident', '_radial_backproject_gpu_streaming'):
        original = getattr(backprojection, name)
        originals[name] = original

        def wrap(function, label):
            @functools.wraps(function)
            def call(*args, **kwargs):
                result = function(*args, **kwargs)
                if result:
                    selected.append(label)
                return result
            return call

        setattr(backprojection, name, wrap(original, name))
    try:
        yield selected
    finally:
        for name, original in originals.items():
            setattr(backprojection, name, original)
        if old_env is None:
            os.environ.pop('YOLO_TTA_GPU_BACKPROJECT', None)
        else:
            os.environ['YOLO_TTA_GPU_BACKPROJECT'] = old_env


def execute_backend(capture, work_dir, backend, workers):
    import numpy as np
    from XTA.component_replay import load_component_replay
    from XTA.interpolation import CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter, materialize_raw_bbox_mask_store_workspace
    from XTA.runtime import close_memmap_array_without_flush
    from XTA import backprojection, sparse_projection

    if backend not in ('legacy', 'sparse', 'legacy_cuda'):
        raise ValueError(f'Unknown replay backend: {backend}')
    replay = load_component_replay(capture)
    if replay.view.family != 'radial':
        raise ValueError('This replay tool currently compares Radial and tilted-Radial components')
    work_dir = Path(work_dir).resolve()
    if work_dir.exists():
        raise FileExistsError(f'Replay workspace already exists: {work_dir}')
    work_dir.mkdir(parents=True)
    destination = work_dir/'projected.cvol'
    try:
        import psutil
        process = psutil.Process()
        resident = lambda: int(process.memory_info().rss)
    except ImportError:
        resident = lambda: None
    rss_before = resident()
    memory = {'peak': rss_before}
    stop = threading.Event()

    def sample():
        while not stop.wait(0.02):
            value = resident()
            if value is not None:
                memory['peak'] = max(memory['peak'] or 0, value)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    decoded = None
    gpu_paths = []
    started = time.perf_counter()
    try:
        if backend == 'sparse':
            stats = sparse_projection.project_radial_sparse_store(replay.source_path, replay.view, destination,
                out_shape_tyx=replay.output_shape, workers=int(workers))
            decode_seconds = 0.0
        else:
            # Subclass only the syscall adapter on Windows; the reference's mask
            # projection, bounding boxes, packing and writer implementation remain
            # unchanged. Production SLURM uses its ordinary pwrite implementation.
            class ReplayWriter(IncrementalRawBBoxMaskStoreWriter):
                _windows_lock = threading.Lock()

                @staticmethod
                def _pwrite_all(fd, data, offset):
                    if hasattr(os, 'pwrite'):
                        return IncrementalRawBBoxMaskStoreWriter._pwrite_all(fd, data, offset)
                    with ReplayWriter._windows_lock:
                        os.lseek(fd, offset, os.SEEK_SET)
                        written = 0
                        while written < len(data):
                            count = os.write(fd, data[written:])
                            if count <= 0:
                                raise OSError('Short replay writer write')
                            written += count

            decode_started = time.perf_counter()
            decoded = materialize_raw_bbox_mask_store_workspace(replay.source_path, work_dir/'decoded.u8',
                desc='Legacy component input decode', workers=int(workers))
            decode_seconds = time.perf_counter()-decode_started
            writer = ReplayWriter(shape=replay.output_shape, store_dir=destination,
                                  format_name=CVOL_FORMAT, desc='Legacy component replay')
            try:
                with _gpu_reference_tracking(backend == 'legacy_cuda') as selected:
                    backprojection.backproject_radial_volume_to_volume(
                        decoded, replay.view, work_dir/'projected-work.u8', 'Legacy component replay',
                        prefer_memory=True, reserve_bytes=32*1024**3, workers=int(workers),
                        out_shape_tyx=replay.output_shape, projection_block_callback=writer, sink_only=True)
                    gpu_paths = list(selected)
                stats = dict(writer.finalize())
                stats['storage_format'] = CVOL_FORMAT
            except BaseException as exc:
                writer.abort(exc)
                writer.discard()
                raise
    finally:
        if decoded is not None:
            close_memmap_array_without_flush(decoded)
        stop.set()
        sampler.join()
        now = resident()
        if now is not None:
            memory['peak'] = max(memory['peak'] or 0, now)
    elapsed = time.perf_counter()-started
    effective = stats.get('backend', 'legacy_cuda' if gpu_paths else 'legacy_cpu')
    result = {'requested_backend': backend, 'effective_backend': effective,
              'gpu_requested': backend == 'legacy_cuda', 'gpu_used': bool(gpu_paths), 'gpu_paths': gpu_paths,
              'elapsed_seconds': elapsed, 'dense_input_decode_seconds': decode_seconds,
              'rss_before_bytes': rss_before, 'peak_rss_bytes': memory['peak'],
              'peak_rss_increase_bytes': (memory['peak']-rss_before) if rss_before is not None else None,
              'shape': replay.output_shape, 'workspace': str(work_dir),
              'result_store': str(destination), 'stats': stats,
              'timing_scope': 'fresh-process projection through closed store publication; checksum validation and comparison excluded; initialization/JIT may be included'}
    (work_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def compare_projected_stores(reference, candidate):
    """Read at most two source-z planes; do not allocate a full output volume."""
    import numpy as np
    from XTA.interpolation import RawBBoxMaskStore
    a = RawBBoxMaskStore.open(Path(reference), mmap_payload=True)
    b = None
    try:
        b = RawBBoxMaskStore.open(Path(candidate), mmap_payload=True)
        if a.shape != b.shape:
            raise ValueError('Projected store geometries differ')
        digest_a, digest_b = hashlib.sha256(), hashlib.sha256()
        changed = foreground_a = foreground_b = 0
        first_changed = None
        for z in range(a.shape[0]):
            left, right = a.decode_slice(z), b.decode_slice(z)
            digest_a.update(memoryview(np.ascontiguousarray(left)))
            digest_b.update(memoryview(np.ascontiguousarray(right)))
            differences = int(np.count_nonzero(left != right))
            changed += differences
            if differences and first_changed is None:
                first_changed = z
            foreground_a += int(np.count_nonzero(left))
            foreground_b += int(np.count_nonzero(right))
        return {'shape': a.shape, 'changed_voxels': changed, 'first_changed_z': first_changed,
                'reference_foreground': foreground_a, 'candidate_foreground': foreground_b,
                'reference_sha256': digest_a.hexdigest(), 'candidate_sha256': digest_b.hexdigest(),
                'exact': changed == 0}
    finally:
        a.close()
        if b is not None:
            b.close()


def _remove_owned_scratch(root, parent):
    """Remove only the exact temporary root created by this replay invocation."""
    root, parent = Path(root), Path(parent)
    if root.resolve() != root or root.parent != parent or not root.name.startswith('xta-component-replay-'):
        raise RuntimeError(f'Refusing to remove an unowned replay scratch directory: {root}')
    shutil.rmtree(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path, help='Captured directory or its manifest.json')
    parser.add_argument('--output', required=True, type=Path, help='Fresh persistent directory for metrics and logs')
    parser.add_argument('--scratch', type=Path, default=Path(tempfile.gettempdir()),
                        help='Local workspace parent; defaults to the system temporary directory')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cuda-reference', action='store_true')
    parser.add_argument('--keep-work', action='store_true', help='Retain the recorded scratch root and backend workspaces')
    parser.add_argument('--worker-backend', choices=('legacy', 'sparse', 'legacy_cuda'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error('Workers must be positive')
    if args.worker_backend:
        if args.worker_backend == 'legacy_cuda' and not args.cuda_reference:
            parser.error('CUDA reference requires --cuda-reference')
        execute_backend(args.capture, args.output, args.worker_backend, args.workers)
        return
    from XTA.component_replay import load_component_replay
    replay = load_component_replay(args.capture)
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f'Use a fresh replay output directory: {root}')
    root.mkdir(parents=True)
    backends = ['legacy', 'sparse'] + (['legacy_cuda'] if args.cuda_reference else [])
    runs = []
    report = {'schema': 'xta-component-projection-replay-result.v1', 'capture': str(replay.root),
              'capture_descriptor_sha256': replay.descriptor['descriptor_sha256'], 'view': replay.view.name,
              'status': 'running', 'all_exact': False, 'runs': runs, 'comparisons': {},
              'scratch_parent': str(args.scratch.resolve()), 'scratch_root': None,
              'backend_workspaces': {}, 'metrics_files': {}, 'workspaces_retained': False,
              'limits': 'Component replay timing is not end-to-end TTA timing. Legacy and sparse store encodings differ; decoded voxels are compared.'}
    scratch_root = None
    failure = None

    def save_report():
        (root/'replay.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')

    try:
        scratch_parent = args.scratch.resolve()
        scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch_root = Path(tempfile.mkdtemp(prefix='xta-component-replay-', dir=scratch_parent)).resolve()
        report['scratch_root'] = str(scratch_root)
        report['backend_workspaces'] = {backend: str(scratch_root/backend) for backend in backends}
        save_report()
        for backend in backends:
            workspace = scratch_root/backend
            command = [sys.executable, '-B', str(Path(__file__).resolve()), str(replay.root), '--output', str(workspace),
                       '--workers', str(args.workers), '--worker-backend', backend]
            if backend == 'legacy_cuda':
                command.append('--cuda-reference')
            env = dict(os.environ)
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            env['YOLO_TTA_TELEMETRY'] = '0'
            if backend != 'legacy_cuda':
                env['YOLO_TTA_GPU_BACKPROJECT'] = '0'
                env['YOLO_TTA_GPU_INTERPOLATION'] = '0'
            with (root/f'{backend}.log').open('w', encoding='utf-8') as log:
                try:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
                finally:
                    # Persist every completed metrics file even if a worker exits
                    # unsuccessfully afterward. Large projection files stay local.
                    metrics = workspace/'metrics.json'
                    if metrics.is_file():
                        persistent_metrics = root/f'{backend}.metrics.json'
                        shutil.copyfile(metrics, persistent_metrics)
                        report['metrics_files'][backend] = str(persistent_metrics)
            runs.append(json.loads((root/f'{backend}.metrics.json').read_text(encoding='utf-8')))
            save_report()
        comparisons = {run['requested_backend']: compare_projected_stores(runs[0]['result_store'], run['result_store'])
                       for run in runs[1:]}
        report['comparisons'] = comparisons
        report['all_exact'] = all(value['exact'] for value in comparisons.values())
        report['status'] = 'complete' if report['all_exact'] else 'mismatch'
    except BaseException as exc:
        failure = exc
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        cleanup_error = None
        if scratch_root is not None:
            if not args.keep_work:
                try:
                    _remove_owned_scratch(scratch_root, scratch_parent)
                except Exception as exc:
                    cleanup_error = exc
                    report['cleanup_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            report['workspaces_retained'] = scratch_root.exists()
        save_report()
        if cleanup_error is not None and failure is None:
            raise cleanup_error
    print(json.dumps({'all_exact': report['all_exact'], 'report': str(root/'replay.json'),
                      'seconds': {run['requested_backend']: run['elapsed_seconds'] for run in runs}}, indent=2))
    if not report['all_exact']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
