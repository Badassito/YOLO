"""Compare complete local TTA runs through generic and native TensorRT paths.

Every mode uses the same engine, views, forward-pass plan and output settings.
Small local fixtures qualify integration; they do not predict H100 job walltime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qualify_spherical_optimization import decoded_layers, write_json


def run_case(args, label, ordinal, source_root=None):
    source_root = Path(source_root or ROOT).resolve(strict=True)
    if not (source_root / 'XTA' / '__init__.py').is_file():
        raise ValueError('Source root must contain the XTA package')
    native = label != 'generic'
    output = args.output_dir / f'{ordinal:02d}-{label}'
    output.mkdir(exist_ok=False)
    config = output / 'ultralytics-config'
    config.mkdir()
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
        PYTHONPATH=os.pathsep.join([str(source_root), *(str(p) for p in args.dependency_path)]),
        YOLO_CONFIG_DIR=str(config), YOLO_AUTOINSTALL='false',
        YOLO_TTA_NATIVE_TRT_RING=str(int(native)),
        YOLO_TTA_FAST_GEOMETRY='1', YOLO_TTA_TASK_TRACE='1', YOLO_TTA_TELEMETRY='1',
        YOLO_TTA_TELEMETRY_DIR=str(output / 'telemetry'),
        YOLO_TTA_TELEMETRY_SAMPLE_SECONDS='1', YOLO_TTA_TELEMETRY_FLUSH_SECONDS='2',
        NUMBA_CACHE_DIR=str(args.cache_dir / 'numba-cache'),
        CUPY_CACHE_DIR=str(args.cache_dir / 'cupy-cache'),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        SLURM_CPUS_PER_TASK=str(args.workers), YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND='0')
    environment.pop('YOLO_TTA_TELEMETRY_PATH', None)
    command = [sys.executable, '-B', '-u', '-m', 'XTA', '--mode', 'tta',
        '--input', str(args.input), '--model', 'gpu:' + str(args.engine), '--device', '0',
        '--output', str(output / 'outputs'), '--temp', str(output / 'runtime'),
        '--imgsz', str(args.imgsz), '--batch', 'gpu:1', '--quantize', 'gpu:fp16',
        '--angle', '0', '--channel_format', 'grey', '--conf', str(args.conf),
        '--min_conf', '0', '--min_radius', '0', '--interpolation_distance', '0',
        '--enable_cartesian', 'transverse', 'sagittal', 'coronal',
        '--enable_tilted', 'transverse:30:vertical', '--enable_azimuthal', 'transverse',
        '--enable_radial', 'transverse', 'sagittal', 'coronal', 'tilted_transverse',
        '--enable_spherical', 'transverse', 'tilted_transverse',
        '--radial_min_radius', str(args.minimum), '--spherical_min_radius', str(args.minimum),
        '--save', 'nrrd', 'summary']
    write_json(output / 'invocation.json', {'argv': command, 'mode': label, 'source_root': str(source_root),
        'environment': {k: v for k, v in environment.items()
                        if k.startswith(('YOLO_', 'NUMBA_', 'CUPY_', 'SLURM_'))
                        or k in ('PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}})
    started = time.perf_counter()
    with (output / 'pipeline.log').open('w', encoding='utf-8') as handle:
        result = subprocess.run(command, cwd=source_root, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    process_seconds = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f'{label} pipeline failed ({result.returncode}); see {output / "pipeline.log"}')
    log = (output / 'pipeline.log').read_text(encoding='utf-8', errors='replace')
    if '\nDone.' not in log:
        raise RuntimeError('Pipeline did not complete output publication')
    layers = decoded_layers(output / 'outputs')
    row = {'mode': label, 'source_root': str(source_root), 'process_seconds': process_seconds,
           'pipeline_seconds': float(re.findall(r'End-to-end pipeline walltime: ([\d.]+)s', log)[-1]),
           'model_frames': int(re.search(r'(\d+) total model frame\(s\)', log)[1]),
           'layers': layers, 'output': str(output),
           'native_route_lines': [line for line in log.splitlines() if 'Native-mask TensorRT ring active' in line]}
    if native and not all(any(f'for {family}:' in line for line in row['native_route_lines'])
                                     for family in ('radial', 'spherical')):
        raise RuntimeError('Native TensorRT ring did not activate for both native families')
    write_json(output / 'result.json', row)
    print(f'{label}: {row["pipeline_seconds"]:.1f}s, {row["model_frames"]} frames, {len(layers)} layers', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dependency-path', type=Path, action='append', default=[])
    parser.add_argument('--cache-dir', type=Path, help='Reuse already-warmed compilation caches in Scratch')
    parser.add_argument('--baseline-root', type=Path,
                        help='Compare this source tree with a baseline, with native TensorRT enabled in both')
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--minimum', type=float, default=121.)
    parser.add_argument('--conf', type=float, default=.00001)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--runs', type=int, choices=(2, 4), default=4)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = (args.cache_dir or args.output_dir).resolve()
    args.input, args.engine = args.input.resolve(strict=True), args.engine.resolve(strict=True)
    args.dependency_path = [path.resolve(strict=True) for path in args.dependency_path]
    if (args.output_dir.is_relative_to(ROOT) or args.cache_dir.is_relative_to(ROOT)
            or args.workers < 1 or args.heat_seconds < 0):
        parser.error('Use task Scratch, positive workers and nonnegative heat time')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.heat_seconds:
        from tools.benchmark_radial_setup import heatsoak
        print(f'Heatsoaking GPU for {args.heat_seconds:g} seconds.', flush=True)
        heatsoak(args.heat_seconds, 0)
    report = {'scope': 'Real complete local TTA integration; not a full-volume H100 throughput estimate.',
              'runs': [], 'all_masks_exact': None, 'all_frame_counts_equal': None}
    if args.baseline_root:
        baseline = args.baseline_root.resolve(strict=True)
        modes = [('baseline', baseline), ('candidate', ROOT)]
        if args.runs == 4:
            modes += [('candidate', ROOT), ('baseline', baseline)]
    else:
        labels = ('generic', 'native') if args.runs == 2 else ('generic', 'native', 'native', 'generic')
        modes = [(label, ROOT) for label in labels]
    for ordinal, (label, source_root) in enumerate(modes, 1):
        row = run_case(args, label, ordinal, source_root)
        report['runs'].append(row)
        write_json(args.output_dir / 'qualification.json', report)
    first = report['runs'][0]
    report['all_frame_counts_equal'] = all(r['model_frames'] == first['model_frames'] for r in report['runs'])
    report['all_masks_exact'] = all(r['layers'] == first['layers'] for r in report['runs'])
    write_json(args.output_dir / 'qualification.json', report)
    if not report['all_frame_counts_equal'] or not report['all_masks_exact']:
        raise RuntimeError('Native ring changed forward-pass count or decoded masks/spatial grids')


if __name__ == '__main__':
    main()
