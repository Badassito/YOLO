"""Compare exact adjacency implementations on coherent and fragmented label planes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import os
import gzip
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA.topology import _compiled_adjacent_gid_pair_codes, _adjacent_gid_pair_codes_numpy
from XTA.topology_runs import run_adjacent_pair_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=3072)
    parser.add_argument('--nrrd', type=Path, help='Optional saved binary volume for additional real slice pairs')
    args = parser.parse_args()
    n = args.size
    y, x = np.ogrid[:n, :n]
    a = np.zeros((n, n), np.uint16)
    a[(x - n/2)**2 + (y - n/2)**2 < (n/3)**2] = 1
    b = np.roll(a, 2, axis=1)
    cases = [('solid', a, b)]
    a = ((x // 64 + y // 64 * (n // 64 + 1)) % 1000 + 1).astype(np.uint16)
    a[(x % 64 < 4) | (y % 64 < 4)] = 0
    cases.append(('patches', a, np.roll(a, 3, axis=1)))
    rng = np.random.default_rng(103)
    a = rng.integers(0, 100, (n, n), dtype=np.uint16)
    cases.append(('fragmented', a, np.roll(a, 1, axis=1)))
    if args.nrrd:
        import cv2
        with args.nrrd.open('rb') as stream:
            header = {}
            while line := stream.readline():
                if line in (b'\n', b'\r\n'):
                    break
                if b':' in line and not line.startswith(b'#'):
                    key, value = line.decode('ascii').strip().split(':', 1)
                    header[key] = value.strip()
            if header.get('encoding') != 'gzip' or header.get('type') not in ('unsigned char', 'uint8', 'uchar'):
                raise ValueError('Benchmark accepts attached gzip uint8 NRRDs')
            shape = tuple(reversed(tuple(map(int, header['sizes'].split()))))
            with gzip.GzipFile(fileobj=stream) as decoded:
                volume = np.frombuffer(decoded.read(), np.uint8).reshape(shape)
        if volume.ndim != 3:
            raise ValueError('Expected a 3D binary NRRD')
        active = np.flatnonzero(np.any(volume != 0, axis=(1, 2)))
        selected = np.unique(np.linspace(1, max(1, len(active)-1), 12).astype(int))
        for j in selected:
            z = int(active[j])
            a = cv2.connectedComponents((volume[z-1] != 0).astype(np.uint8), connectivity=8)[1]
            b = cv2.connectedComponents((volume[z] != 0).astype(np.uint8), connectivity=8)[1]
            # Match the production adjacent-pair overlap window, including its
            # one-pixel neighborhood halo. This is still an LQ qualification.
            ax, ay, aw, ah = cv2.boundingRect((a != 0).astype(np.uint8))
            bx, by, bw, bh = cv2.boundingRect((b != 0).astype(np.uint8))
            y0, y1 = max(0, max(ay, by)-1), min(a.shape[0], min(ay+ah, by+bh)+1)
            x0, x1 = max(0, max(ax, bx)-1), min(a.shape[1], min(ax+aw, bx+bw)+1)
            if y0 >= y1 or x0 >= x1:
                continue
            a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
            cases.append((f'real_z{z}', a, b))
    offsets = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    report = []
    for name, a, b in cases:
        # Compile/cache each signature before component timing.
        with mock.patch.dict(os.environ, {'YOLO_TTA_TOPOLOGY_RUN_ADJACENCY': '0'}):
            expected = _compiled_adjacent_gid_pair_codes(a, b, offsets)
        candidate = run_adjacent_pair_codes(a, b, offsets) if a.size >= 262144 else None
        if candidate is not None:
            np.testing.assert_array_equal(candidate, expected)
        times = {'pixel_hash': [], 'runs_with_fallback': []}
        for trial in range(4):
            for key in (list(times) if trial % 2 == 0 else list(reversed(times))):
                with mock.patch.dict(os.environ, {'YOLO_TTA_TOPOLOGY_RUN_ADJACENCY': '0' if key == 'pixel_hash' else '1'}):
                    started = time.perf_counter()
                    result = _compiled_adjacent_gid_pair_codes(a, b, offsets)
                    elapsed = time.perf_counter() - started
                np.testing.assert_array_equal(result, expected)
                times[key].append(elapsed)
        row = dict(case=name, shape=list(a.shape), pairs=int(expected.size),
                   run_admitted=candidate is not None, seconds=times,
                   medians={k: statistics.median(v) for k, v in times.items()})
        report.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
