"""Measure real CPU reader shutdown at a controlled GPU-handoff boundary.

Uses three speculative production-plane readers and a small processing mask.
No GPU is accessed and no cluster walltime is predicted. Both paths join every
reader; the candidate cancels abandoned work between 128K-coordinate chunks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from unittest import mock

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from XTA import spherical_projection as sp
from XTA.spherical_geometry import build_spherical_view_infos


def trial(view, source, shape, cooperative, pause_fraction):
    ready, gate = threading.Event(), threading.Event()
    cancel = threading.Event() if cooperative else None
    lock = threading.Lock()
    readers = chunks = cancelled = entered = 0
    original = sp._pull_spherical_chunk
    radii = np.asarray(view.spherical_radii)
    rotation = np.asarray(view.spherical_rotation_xyz).reshape(3, 3)
    chunk_count = (shape[1] * shape[2] + sp._PULL_CHUNK_VOXELS - 1) // sp._PULL_CHUNK_VOXELS
    pause_index = min(chunk_count - 2, int(chunk_count * pause_fraction))
    pause_at = pause_index * sp._PULL_CHUNK_VOXELS

    def pull(*args, **kwargs):
        nonlocal chunks, entered
        if args[6] == pause_at:
            with lock:
                entered += 1
                if entered == 3:
                    ready.set()
            if not gate.wait(30):
                raise RuntimeError('Controlled handoff was not released')
        with lock:
            chunks += 1
        return original(*args, **kwargs)

    def project(first, count):
        nonlocal readers, cancelled
        if first == 0:
            return np.zeros((1, shape[1], shape[2]), np.uint8)
        with lock:
            readers += 1
        try:
            return sp._project_spherical_block(source, view, radii, rotation, shape,
                shape[0] // 2 + first, count, cancel_event=cancel)
        except CancelledError:
            with lock:
                cancelled += 1
            raise
        finally:
            with lock:
                readers -= 1

    with (mock.patch.object(sp, '_pull_spherical_chunk', side_effect=pull),
          mock.patch.object(sp, '_spherical_block_schedule', return_value=(1, 4))):
        blocks = sp._ordered_spherical_blocks(project, 4, shape[1] * shape[2], 4,
                                             cancel_event=cancel)
        try:
            first, block = next(blocks)
            assert first == 0 and block.shape == (1, shape[1], shape[2])
            if not ready.wait(30):
                raise RuntimeError('Three speculative readers did not enter their first chunk')
            started = time.perf_counter()
            if cancel is not None:
                cancel.set()
            gate.set()
            blocks.close()
            elapsed = time.perf_counter() - started
        finally:
            gate.set()
            blocks.close()
    assert readers == 0
    if cooperative:
        assert cancelled == 3 and chunks == 3 * (pause_index + 1)
    else:
        assert cancelled == 0 and chunks == 3 * ((shape[1] * shape[2] + sp._PULL_CHUNK_VOXELS - 1) // sp._PULL_CHUNK_VOXELS)
    return {'cooperative': cooperative, 'drain_seconds': elapsed,
            'pause_fraction': pause_fraction, 'pause_chunk': pause_index,
            'completed_pull_chunks': chunks, 'cancelled_blocks': cancelled,
            'readers_remaining_before_gpu_consumption': readers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT) or args.repeats < 1:
        parser.error('Use positive repeats and a task Scratch output path')
    shape = (1931, 3064, 3022)
    view = build_spherical_view_infos(2911, 3064, 3022, targets=('transverse',),
        min_radius=3072 / (4 * np.pi), patch_size=3072, tilted_views=())[0]
    source = np.ones((view.num_slices, 31, 37), np.uint8)
    rows = []
    medians = {}
    for fraction in (0., .5, .9):
        for repeat in range(args.repeats):
            for cooperative in ((False, True) if repeat % 2 == 0 else (True, False)):
                row = trial(view, source, shape, cooperative, fraction)
                row['repeat'] = repeat
                rows.append(row)
                print(json.dumps(row), flush=True)
        medians[str(fraction)] = {
            name: statistics.median(row['drain_seconds'] for row in rows
                                    if row['cooperative'] == enabled and row['pause_fraction'] == fraction)
            for name, enabled in (('full_reader_join', False), ('chunk_cancel_join', True))}
    report = {'scope': __doc__, 'shape': shape, 'source_shape': source.shape,
              'trials': rows, 'medians_seconds': medians}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
