"""Compare Radial host setup while actual XTA NRRD writers run concurrently.

Uses production mask strides and deterministic rectangles, not model predictions.
The required output directory owns every temporary file and retained measurement.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cylindrical_projection as projection, cylindrical_cuda_projection as cuda, geometry
from XTA import interpolation, outputs
from tools.benchmark_radial_cuda_projection import WORKING_SHAPE, OUTPUT_SHAPE
from tools.benchmark_radial_setup import heatsoak


def digest(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


class OutputLoad:
    def __init__(self, root, workers, *, mirrors=False, shape=(128, 512, 512)):
        self.root, self.workers = root, workers
        self.stop = threading.Event()
        self.active = [threading.Event() for _ in range(workers)]
        self.counts = [0] * workers
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers else None
        self.futures = []
        self.mirrors = mirrors
        self.mirror_shape = tuple(max(1, round(n * .2)) for n in shape)
        plane = np.zeros(shape[1:], np.uint8)
        if shape[1:] == (512, 512):
            plane[31:487:3, 23:479:5] = 1
        else:
            y0, x0 = (shape[1] - min(1024, shape[1])) // 2, (shape[2] - min(1536, shape[2])) // 2
            plane[y0:y0 + 1024:3, x0:x0 + 1536:5] = 1
        def repeated_digest(frame, count):
            result = hashlib.sha256()
            for _ in range(count):
                result.update(memoryview(frame).cast('B'))
            return result.hexdigest()
        self.expected = repeated_digest(plane, shape[0])
        self.expected_mirror = repeated_digest(outputs._resize_binary_mask_frame_to_output_shape(
            plane, *self.mirror_shape[1:]), self.mirror_shape[0]) if mirrors else None
        store_path = root / 'load-input.cvol'
        writer = interpolation.IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=store_path,
            format_name=interpolation.CVOL_FORMAT, desc='NRRD load fixture')
        try:
            for z in range(shape[0]):
                writer(z, plane[None])
            stats = writer.finalize()
        except BaseException:
            writer.discard()
            raise
        self.ref = interpolation.NrrdLayerRef(key='load', name='load', path=store_path, shape=shape,
            storage_format=interpolation.CVOL_FORMAT, segment_extent_ijk=tuple(stats['segment_extent_ijk']),
            segment_extent_shape_tyx=shape)
        self.thread_index = threading.local()
        self.original_payload = outputs._write_one_decomposed_nrrd_layer_payload
        def observed(*args, **kwargs):
            index = getattr(self.thread_index, 'index', None)
            if index is not None:
                self.active[index].set()
            return self.original_payload(*args, **kwargs)
        outputs._write_one_decomposed_nrrd_layer_payload = observed

    def run(self, index):
        self.thread_index.index = index
        while not self.stop.is_set():
            if self.mirrors:
                outputs.write_layer_nrrd_with_low_quality_mirrors(self.ref, self.ref.shape,
                    self.root / f'load-{index}.seg.nrrd',
                    [(self.mirror_shape, self.root / f'load-{index}-mirror.seg.nrrd')], z_shards=1)
            else:
                outputs.write_single_layer_nrrd_from_ref(self.ref, self.ref.shape,
                    self.root / f'load-{index}.seg.nrrd', z_shards=1)
            self.counts[index] += 1

    def start(self):
        for i in range(self.workers):
            self.futures.append(self.pool.submit(self.run, i))
        for index, event in enumerate(self.active):
            if not event.wait(30):
                if self.futures[index].done():
                    self.futures[index].result()
                raise RuntimeError('NRRD load did not enter its actual payload writer')

    def close(self):
        self.stop.set()
        if self.pool:
            self.pool.shutdown(wait=True)
        outputs._write_one_decomposed_nrrd_layer_payload = self.original_payload
        if not self.futures:
            return
        for future in self.futures:
            future.result()
        for i in range(self.workers):
            for suffix, expected in (('', self.expected), ('-mirror', self.expected_mirror)):
                if expected is None:
                    continue
                path = self.root / f'load-{i}{suffix}.seg.nrrd'
                with path.open('rb') as stream:
                    while stream.readline().strip():
                        pass
                    with gzip.GzipFile(fileobj=stream) as decoded:
                        actual = hashlib.sha256()
                        while block := decoded.read(8 * 1024**2):
                            actual.update(block)
                if actual.hexdigest() != expected:
                    raise RuntimeError(f'Background NRRD differs from its source fixture: {path.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--load-workers', type=int, default=8)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.load_workers < 0 or args.heat_seconds < 0 or args.device < 0:
        parser.error('Worker count, heat seconds and device must be nonnegative')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_TTA_NRRD_MEMBER_CODEC'] = 'cpu'
    os.environ['YOLO_TTA_TELEMETRY'] = '0'
    report = {'working_shape': WORKING_SHAPE, 'output_shape': OUTPUT_SHAPE,
        'load_workers': args.load_workers, 'runs': [], 'limits':
        'Synthetic rectangles and one local GPU with repeated full NRRD writes. '
        'No cluster, model inference, multi-GPU scheduling or low-quality mirrors. '
        'CUDA output equality is checked on three source slices, not a full volume.'}
    report['source_sha256'] = {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
        for name in ('XTA/cylindrical_projection.py', 'XTA/cylindrical_cuda_projection.py')}
    metadata_budget = projection._RADIAL_METADATA_CHUNK_VALUES
    with tempfile.TemporaryDirectory(prefix='host-contention-', dir=root) as temporary:
        workspace = Path(temporary).resolve()
        if workspace.parent != root:
            raise RuntimeError('Benchmark temporary directory escaped its output directory')
        view = geometry.get_view_infos(*WORKING_SHAPE, cartesian_views=(), radial_views=('transverse',),
                                       radial_patch_size=3072)[0]
        view = replace(view, radial_tilted_source=True, tilt_direction='horizontal', tilt_angle_deg=-30.)
        shape = (view.num_slices, view.src_h, view.src_w)
        source = np.memmap(workspace / 'source.u8', mode='w+', dtype=np.uint8, shape=shape)
        load = None
        try:
            boxes = np.zeros((shape[0], 4), np.int64)
            for shell in range(shape[0]):
                if shell % 7:
                    y, x = (shell * 13) % (shape[1] - 768), (shell * 29) % (shape[2] - 2048)
                    source[shell, y:y + 768, x:x + 2048] = 255
                    boxes[shell] = (y, y + 768, x, x + 2048)
            source.flush()
            report['source_sum'] = int(source.sum(dtype=np.uint64))
            report['source_shape'], report['source_strides'] = shape, source.strides
            radii = np.asarray(geometry.radial_global_radii(view))
            plan = projection._build_radial_plane_plan(view, radii, OUTPUT_SHAPE)
            metadata = projection._radial_projection_metadata(view, shape, OUTPUT_SHAPE, plan)
            expected_metadata = [digest(a) for a in metadata[:5]]
            reference = {}
            for z in (0, OUTPUT_SHAPE[0] // 2, OUTPUT_SHAPE[0] - 1):
                plane = np.empty(OUTPUT_SHAPE[1:], np.uint8)
                flat = plane.reshape(-1)
                for first in range(0, flat.size, projection._PULL_CHUNK_VOXELS):
                    stop = min(flat.size, first + projection._PULL_CHUNK_VOXELS)
                    flat[first:stop] = projection._pull_radial_chunk(source, view, radii, OUTPUT_SHAPE, z, first, stop)
                reference[z] = digest(plane)
            # Warm module compilation, Numba, upload and first CUDA calls outside
            # measured runs. Both upload routes still use the same source pages.
            with cuda.RadialCudaProjector(source, plan, metadata, view, OUTPUT_SHAPE,
                                           boxes, True, args.device) as warm:
                if warm.source_pack_backend != 'numba_nogil':
                    raise RuntimeError('Compiled crop packer was not qualified')
            report['heatsoak_seconds'] = heatsoak(args.heat_seconds, args.device)
            load = OutputLoad(workspace, args.load_workers)
            load.start()
            for mode in ('per_shell_numpy', 'batched_nogil', 'batched_nogil', 'per_shell_numpy'):
                before = sum(load.counts)
                started = time.perf_counter()
                with mock.patch.object(projection, '_RADIAL_METADATA_CHUNK_VALUES',
                        view.src_w if mode == 'per_shell_numpy' else metadata_budget):
                    metadata = projection._radial_projection_metadata(view, shape, OUTPUT_SHAPE, plan)
                metadata_seconds = time.perf_counter() - started
                if [digest(a) for a in metadata[:5]] != expected_metadata:
                    raise RuntimeError('Batched metadata differs from per-shell NumPy')
                packer = None if mode == 'per_shell_numpy' else cuda._pack_radial_source_block_compiled
                with mock.patch.object(cuda, '_pack_radial_source_block_compiled', packer), \
                        cuda.RadialCudaProjector(source, plan, metadata, view, OUTPUT_SHAPE,
                                                boxes, True, args.device) as projector:
                    row = {name: getattr(projector, name) for name in (
                        'source_h2d_bytes', 'source_upload_seconds', 'source_pack_seconds',
                        'source_pack_backend', 'contract_validation_seconds', 'module_setup_seconds',
                        'buffer_setup_seconds', 'geometry_upload_seconds', 'preflight_seconds', 'constructor_seconds')}
                    for z, expected in reference.items():
                        if digest(projector.project(z, 1)[0]) != expected:
                            raise RuntimeError(f'CUDA source sample differs at z={z}')
                row.update(mode=mode, metadata_host_seconds=metadata_seconds,
                    nrrds_completed_during_run=sum(load.counts) - before)
                report['runs'].append(row)
                (root / 'host-contention.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                print(row, flush=True)
            load.close()
            report['nrrds_completed'] = sum(load.counts)
            load = None
            report['all_exact'] = True
            (root / 'host-contention.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        finally:
            try:
                if load is not None:
                    load.close()
            finally:
                source._mmap.close()


if __name__ == '__main__':
    main()
