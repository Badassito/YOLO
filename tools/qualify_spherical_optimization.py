"""Run bounded real-model Spherical regressions and compare decoded NRRD layers.

An optional baseline source tree enables an alternating A/B/B/A comparison.
This small workstation workload is useful for parity and local timing, not an
estimate of cluster throughput. All outputs and runtime files belong in Scratch.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def normalize(value):
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, (list, tuple)):
        return [normalize(x) for x in value]
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def decoded_layers(output):
    """Read the pipeline's inline uint8 NRRDs without an optional PTA dependency."""
    result = {}
    spatial_keys = ('space', 'space directions', 'space origin', 'sizes',
                    'kinds', 'space units', 'measurement frame', 'centers')
    for path in sorted((output / 'nrrd').glob('*.nrrd')):
        raw = path.read_bytes()
        boundary = re.search(b'\r?\n\r?\n', raw)
        if boundary is None or not raw.startswith(b'NRRD'):
            raise ValueError(f'Missing NRRD header: {path}')
        header = {}
        for line in raw[:boundary.start()].decode('ascii').splitlines():
            if line.startswith('#') or ':' not in line:
                continue
            key, value = line.split(':', 1)
            header[key.strip()] = value.strip()
        if header.get('type') not in ('unsigned char', 'uchar', 'uint8', 'uint8_t'):
            raise ValueError(f'Qualification expects uint8 NRRD: {path}')
        if 'data file' in header or 'byte skip' in header or 'line skip' in header:
            raise ValueError(f'Qualification expects inline NRRD payload: {path}')
        sizes = tuple(map(int, header['sizes'].split()))
        if len(sizes) != int(header['dimension']) or min(sizes) <= 0:
            raise ValueError(f'Invalid NRRD sizes: {path}')
        payload = raw[boundary.end():]
        if header.get('encoding') in ('gzip', 'gz'):
            payload = gzip.decompress(payload)
        elif header.get('encoding') != 'raw':
            raise ValueError(f'Unsupported NRRD encoding: {path}')
        data = np.frombuffer(payload, dtype=np.uint8).reshape(sizes[::-1])
        result[path.name] = {
            'shape': list(data.shape), 'dtype': str(data.dtype),
            'sha256': hashlib.sha256(memoryview(data).cast('B')).hexdigest(),
            'foreground': int(np.count_nonzero(data)),
            'spatial_header': {key: normalize(header[key]) for key in spatial_keys if key in header},
        }
    if not result or not any(item['foreground'] for item in result.values()):
        raise RuntimeError('Qualification requires nonempty decoded NRRD output')
    return result


def run_case(source, output, args):
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-u', '-m', 'XTA', '--mode', 'tta',
               '--input', str(args.input), '--model', 'gpu:' + str(args.model),
               '--device', str(args.device), '--output', str(output / 'outputs'),
               '--temp', str(output / 'runtime'), '--imgsz', str(args.imgsz),
               '--batch', 'gpu:1', '--angle', '0',
               '--enable_tilted', 'transverse:30:vertical',
               '--enable_spherical', 'transverse', 'sagittal', 'coronal', 'tilted_transverse',
               '--spherical_min_radius', str(args.minimum),
               '--interpolation_distance', '0', '--conf', '.00001',
               '--min_conf', '0', '--min_radius', '0', '--save', 'nrrd', 'summary']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(source)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['NUMBA_CACHE_DIR'] = str(args.output_dir / 'numba-cache')
    env['CUPY_CACHE_DIR'] = str(args.output_dir / 'cupy-cache')
    env['YOLO_TTA_TELEMETRY'] = '0'
    env['YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND'] = '0'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[name] = '2'
    env['SLURM_CPUS_PER_TASK'] = str(args.workers)
    write_json(output / 'invocation.json', {'source': str(source), 'argv': command,
               'environment': {key: value for key, value in env.items()
                               if key.startswith(('YOLO_TTA_', 'XTA_', 'NUMBA_', 'CUPY_', 'SLURM_'))
                               or key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}})
    started = time.perf_counter()
    with (output / 'pipeline.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=source, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f'Pipeline exit {completed.returncode}; see {output / "pipeline.log"}')
    text = (output / 'pipeline.log').read_text(encoding='utf-8', errors='replace')
    timing = re.findall(r'End-to-end pipeline walltime: ([0-9.]+)s', text)
    if not timing or '\nDone.' not in text:
        raise RuntimeError('Pipeline did not report completed output')
    layers = decoded_layers(output / 'outputs')
    result = {'source': str(source), 'process_seconds': elapsed,
              'pipeline_seconds': float(timing[-1]), 'layers': layers,
              'compaction_lines': [line for line in text.splitlines()
                                   if 'Direct compaction layout' in line],
              'provenance_lines': [line for line in text.splitlines()
                                   if line.startswith('Execution provenance:')]}
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--minimum', type=float, default=121.)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.source_root = args.source_root.resolve(strict=True)
    args.input, args.model = args.input.resolve(strict=True), args.model.resolve(strict=True)
    if args.output_dir.is_relative_to(ROOT):
        parser.error('Generated evidence must be outside the repository, in task Scratch')
    if args.heat_seconds < 0 or args.workers < 1 or args.device < 0:
        parser.error('Heat/device must be nonnegative and workers positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('NUMBA_CACHE_DIR', 'CUPY_CACHE_DIR'):
        os.environ[name] = str(args.output_dir / name.lower().replace('_dir', ''))
    report = {'scope': 'Local real-model parity and timing; not cluster throughput or accuracy.',
              'runs': [], 'all_decoded_layers_equal': None}
    if args.heat_seconds:
        from tools.benchmark_radial_setup import heatsoak
        print(f'Heatsoaking GPU for {args.heat_seconds:g} seconds.', flush=True)
        report['heatsoak_seconds'] = heatsoak(args.heat_seconds, args.device)
    plan = [('candidate', args.source_root)]
    if args.baseline_root:
        baseline = args.baseline_root.resolve(strict=True)
        plan = [('baseline', baseline), ('candidate', args.source_root),
                ('candidate', args.source_root), ('baseline', baseline)]
    expected = None
    for index, (name, source) in enumerate(plan):
        print(f'Running {index + 1}/{len(plan)}: {name}', flush=True)
        result = run_case(source, args.output_dir / f'{index + 1:02d}-{name}', args)
        if expected is None:
            expected = result['layers']
        elif result['layers'] != expected:
            report['all_decoded_layers_equal'] = False
            report['runs'].append({'name': name, **result})
            write_json(args.output_dir / 'qualification.json', report)
            raise RuntimeError(f'Decoded layers or spatial headers differ in {name}')
        report['runs'].append({'name': name, **result})
        write_json(args.output_dir / 'qualification.json', report)
        print(f'{name}: {result["pipeline_seconds"]:.1f}s; {len(result["layers"])} decoded layers.', flush=True)
    report['all_decoded_layers_equal'] = True if len(plan) > 1 else None
    write_json(args.output_dir / 'qualification.json', report)


if __name__ == '__main__':
    main()
