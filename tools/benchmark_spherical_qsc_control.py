"""Compare QSC host construction and frame validation against saved source files.

CPU only; writes measurements to the supplied Scratch JSON path. Before editing,
save qsc.py and spherical_cuda.py as qsc_baseline.py and spherical_cuda_baseline.py
in --baseline-dir. No reference source is imported from Git or an external URL.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from XTA import qsc, spherical_cuda
from tests.test_spherical_cuda import resident_engine, rotation_xyz, spherical_view


def load_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(function, repeats, calls=1):
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(calls):
            function()
        samples.append((time.perf_counter() - started) / calls)
    gc.collect()
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'median_seconds': statistics.median(samples), 'samples_seconds': samples,
            'peak_traced_bytes': peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--columns', type=int, default=3072)
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=9)
    args = parser.parse_args()
    if min(args.columns, args.rows, args.repeats) <= 0:
        parser.error('rows, columns and repeats must be positive')
    if args.rows * args.columns > 1_000_000:
        parser.error('use at most one million pixels to keep qualification memory bounded')
    reference_qsc = load_reference(args.baseline_dir / 'qsc_baseline.py', 'XTA._qsc_benchmark_reference')
    reference_render = load_reference(args.baseline_dir / 'spherical_cuda_baseline.py',
                                      'XTA._spherical_benchmark_reference')
    reference_render.qsc_inverse = reference_qsc.qsc_inverse
    u = np.linspace(-1., 1., args.columns)[None, :]
    v = np.linspace(1., -1., args.rows)[:, None]
    vectors = np.random.default_rng(141).normal(size=(args.rows, args.columns, 3))
    for face in range(6):
        np.testing.assert_array_equal(qsc.qsc_inverse(face, u, v).view(np.uint64),
                                      reference_qsc.qsc_inverse(face, u, v).view(np.uint64))
        for actual, expected in zip(qsc.qsc_forward_face(vectors, face),
                                    reference_qsc.qsc_forward_face(vectors, face)):
            np.testing.assert_array_equal(actual, expected)
    view = spherical_view(size=args.columns, intervals=args.columns - 2,
                          origin=(-2, -3), face=4, rotation=rotation_xyz())
    engine = resident_engine(np.zeros((9, 11, 13), np.uint8))
    _, _, key = spherical_cuda._render_contract(engine, view, 0)
    for actual, expected in zip(spherical_cuda._build_host_direction_block(key, 0, args.rows),
                                reference_render._build_host_direction_block(key, 0, args.rows)):
        np.testing.assert_array_equal(actual, expected)
    cases = {
        'inverse_scalar_face': (lambda: reference_qsc.qsc_inverse(4, u, v),
                                lambda: qsc.qsc_inverse(4, u, v), 1),
        'forward_scalar_face': (lambda: reference_qsc.qsc_forward_face(vectors, 4),
                                lambda: qsc.qsc_forward_face(vectors, 4), 1),
        'native_host_direction_strip': (
            lambda: reference_render._build_host_direction_block(key, 0, args.rows),
            lambda: spherical_cuda._build_host_direction_block(key, 0, args.rows), 1),
        'native_render_contract_repeated': (
            lambda: reference_render._render_contract(engine, view, 0),
            lambda: spherical_cuda._render_contract(engine, view, 0), 1000),
    }
    results = {'shape': [args.rows, args.columns], 'repeats': args.repeats,
               'numpy_version': np.__version__, 'device': 'cpu', 'thread_limit': 2,
               'parity': 'exact baseline equality, all six faces', 'cases': {}}
    for name, (before, after, calls) in cases.items():
        baseline = measure(before, args.repeats, calls)
        candidate = measure(after, args.repeats, calls)
        results['cases'][name] = {
            'baseline': baseline, 'candidate': candidate,
            'speedup': baseline['median_seconds'] / candidate['median_seconds'],
            'peak_reduction_bytes': baseline['peak_traced_bytes'] - candidate['peak_traced_bytes'],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
