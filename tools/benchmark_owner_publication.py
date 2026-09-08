"""Compare raw and packed source publication through native and mirror NRRDs."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cuda_d1, geometry, outputs
from XTA.packed_publication import encode_owner_packed_block


def nrrd_payload_digest(path):
    with path.open('rb') as stream:
        header = []
        while True:
            line = stream.readline()
            if line in (b'\n', b'\r\n'):
                break
            if not line:
                raise IOError('Truncated NRRD header')
            header.append(line)
        checksum, size = hashlib.sha256(), 0
        with gzip.GzipFile(fileobj=stream) as decoded:
            while block := decoded.read(4*1024**2):
                checksum.update(block)
                size += len(block)
    return dict(sha256=checksum.hexdigest(), bytes=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=(96, 3064, 3022))
    parser.add_argument('--trials', type=int, default=2)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    shape = tuple(args.shape)
    if min(shape) <= 0 or args.trials <= 0:
        parser.error('Shape and trials must be positive')
    words = np.zeros((int(np.prod(shape)) + 31)//32, np.uint32)
    y, x = np.ogrid[:shape[1], :shape[2]]
    # 32-slice blocks start/end on uint32 boundaries even for odd plane widths.
    for first in range(0, shape[0], 32):
        depth = min(32, shape[0]-first)
        block = np.zeros((depth, *shape[1:]), np.uint8)
        for i in range(depth):
            z = (first+i-(shape[0]-1)/2) / max(1, shape[0]*.42)
            block[i] = ((x-shape[2]*.51)**2/(shape[2]*.32)**2 +
                        (y-shape[1]*.49)**2/(shape[1]*.27)**2 + z*z < 1)
        packed = np.packbits(block.reshape(-1), bitorder='little')
        packed = np.pad(packed, (0, (-len(packed))%4)).view(np.uint32)
        offset = first*shape[1]*shape[2]//32
        words[offset:offset+len(packed)] = packed
    encode_owner_packed_block(words, shape, 0, min(4, shape[0]))
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    mirror = tuple(max(1, round(s*.2)) for s in shape)
    report = []
    baseline = None
    for trial in range(args.trials):
        for mode in (('raw', 'packed') if trial % 2 == 0 else ('packed', 'raw')):
            with tempfile.TemporaryDirectory(prefix=mode+'-', dir=root) as directory:
                work = Path(directory)
                copied = words.copy()  # producer fixture, excluded from publication timing
                with mock.patch.dict(os.environ, {'YOLO_TTA_PACKED_OWNER_PUBLICATION': '1' if mode=='packed' else '0',
                                                   'YOLO_TTA_NRRD_CROP_ROW_SPANS': '1' if mode=='packed' else '0',
                                                   'YOLO_TTA_NRRD_GPU_MIRROR_TEE': '0'}):
                    started = time.perf_counter()
                    result = cuda_d1._d1_finalize_bitset_layer(words=copied, output_shape=shape,
                              store_dir=work/'layer', model_name='fixture', view=view)
                    publication_s = time.perf_counter()-started
                    started = time.perf_counter()
                    outputs.write_layer_nrrd_with_low_quality_mirrors(result['d1_layer_ref'], shape,
                              work/'native.nrrd', [(mirror, work/'mirror.nrrd')])
                    nrrd_s = time.perf_counter()-started
                hashes = {name: nrrd_payload_digest(work/f'{name}.nrrd') for name in ('native','mirror')}
                if baseline is None:
                    baseline = hashes
                if hashes != baseline:
                    raise AssertionError('Decoded publication parity failed')
                row = dict(trial=trial, mode=mode, publication_seconds=publication_s,
                           nrrd_seconds=nrrd_s, combined_seconds=publication_s+nrrd_s,
                           payload_bytes=result['d1_cvol_stats']['raw_payload_bytes'], hashes=hashes,
                           nrrd_bytes={name: (work/f'{name}.nrrd').stat().st_size for name in ('native','mirror')})
                report.append(row)
                print(json.dumps(row), flush=True)
    (root/'publication-benchmark.json').write_text(json.dumps(dict(shape=shape, trials=report), indent=2)+'\n')


if __name__ == '__main__':
    main()
