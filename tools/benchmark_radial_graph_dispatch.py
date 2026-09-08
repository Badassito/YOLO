"""Full Radial CUDA/CVOL comparison under full-width NRRD and 0.20 mirror load."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cylindrical_projection as projection, cylindrical_cuda_projection as cuda, geometry, interpolation
from tools.benchmark_radial_cuda_projection import WORKING_SHAPE, OUTPUT_SHAPE
from tools.benchmark_radial_cuda_sink import decode_store
from tools.benchmark_radial_host_contention import OutputLoad, digest
from tools.benchmark_radial_setup import heatsoak


def fill_ellipsoid_source(source, view):
    """Sample a bounded spatial object into the existing intrinsic patch grid.

    Its projected extent is near the logged 1–1.6 GB crop payloads. Repeated
    angular occurrences see the same object instead of unrelated rectangles
    that spread support across most of the output plane.
    """
    boxes = np.zeros((source.shape[0], 4), np.int64)
    columns = np.arange(view.src_w, dtype=np.float64)
    tangent = math.tan(math.radians(view.tilt_angle_deg))
    for shell, radius in enumerate(view.radial_radii):
        theta = np.remainder((view.radial_arc_origin + columns) / radius, 2.0 * math.pi)
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        cross = ((x - 500.) / 600.)**2 + (y / 600.)**2
        reach = 1000. * np.sqrt(np.maximum(0., 1. - cross))
        center = (WORKING_SHAPE[0] - 1) * .5 - tangent * x - view.radial_height_origin
        starts = np.clip(np.ceil(center - reach), 0, source.shape[1]).astype(np.int64)
        stops = np.clip(np.floor(center + reach) + 1, 0, source.shape[1]).astype(np.int64)
        good = (cross <= 1.) & (stops > starts)
        if not np.any(good):
            continue
        starts[~good] = stops[~good] = 0
        x0, x1 = int(np.flatnonzero(good)[0]), int(np.flatnonzero(good)[-1]) + 1
        y0, y1 = int(starts[good].min()), int(stops[good].max())
        rows = np.arange(y0, y1)[:, None]
        mask = (rows >= starts[None, x0:x1]) & (rows < stops[None, x0:x1])
        source[shell, y0:y1, x0:x1] = mask.astype(np.uint8) * np.uint8(255)
        boxes[shell] = (y0, y1, x0, x1)
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--load-workers', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--fixture', choices=('rectangles', 'ellipsoid'), default='ellipsoid')
    args = parser.parse_args()
    if min(args.load_workers, args.device, args.heat_seconds) < 0:
        parser.error('Workers, device and heat seconds must be nonnegative')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_TTA_NRRD_MEMBER_CODEC'] = 'cpu'
    os.environ['YOLO_TTA_TELEMETRY'] = '0'
    report = {'working_shape': WORKING_SHAPE, 'output_shape': OUTPUT_SHAPE, 'runs': [],
        'background_shape': (64, OUTPUT_SHAPE[1], OUTPUT_SHAPE[2]),
        'load_workers': args.load_workers, 'background_mirrors': .2,
        'fixture': args.fixture,
        'candidate': 'graph_and_empty_block_skip',
        'limits': 'Synthetic masks and one local GPU. Real ordered compact CVOL publication and '
                  'full-width 64-slice NRRD layers with 0.20 mirrors run concurrently. '
                  'No inference, four-GPU scheduling or full production-layer depth is reproduced.'}
    report['source_sha256'] = {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
        for name in ('XTA/cylindrical_projection.py', 'XTA/cylindrical_cuda_projection.py',
                     'tools/benchmark_radial_host_contention.py', 'tools/benchmark_radial_graph_dispatch.py')}
    def save():
        (root / 'graph-dispatch.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='graph-dispatch-', dir=root) as temporary:
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
            if args.fixture == 'ellipsoid':
                boxes = fill_ellipsoid_source(source, view)
            else:
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
            reference = {}
            for z in (0, OUTPUT_SHAPE[0] // 2, OUTPUT_SHAPE[0] - 1):
                plane = np.empty(OUTPUT_SHAPE[1:], np.uint8)
                flat = plane.reshape(-1)
                for first in range(0, flat.size, projection._PULL_CHUNK_VOXELS):
                    stop = min(flat.size, first + projection._PULL_CHUNK_VOXELS)
                    flat[first:stop] = projection._pull_radial_chunk(source, view, radii, OUTPUT_SHAPE, z, first, stop)
                reference[z] = digest(plane)
            with cuda.RadialCudaProjector(source, plan, metadata, view, OUTPUT_SHAPE, boxes, True,
                                           args.device, use_graphs=True) as warm:
                if not warm.cuda_graph_enabled:
                    raise RuntimeError(f'CUDA graph preflight was unavailable: {warm.cuda_graph_note}')
                warm.project_encoded(0, warm.max_block_depth)
            load = OutputLoad(workspace, args.load_workers, mirrors=True, shape=report['background_shape'])
            report['heatsoak_seconds'] = heatsoak(args.heat_seconds, args.device)
            load.start()
            for ordinal, use_graphs in enumerate((False, True, True, False)):
                path = workspace / f'projection-{ordinal}.cvol'
                writer = interpolation.IncrementalRawBBoxMaskStoreWriter(shape=OUTPUT_SHAPE,
                    store_dir=path, format_name=interpolation.CVOL_FORMAT, desc='Radial dispatch benchmark')
                before = sum(load.counts)
                try:
                    with cuda.RadialCudaProjector(source, plan, metadata, view, OUTPUT_SHAPE,
                            boxes, True, args.device, use_graphs=use_graphs, skip_empty_blocks=use_graphs) as projector:
                        if projector.cuda_graph_enabled != use_graphs:
                            raise RuntimeError(f'CUDA graph mode was not honored: {projector.cuda_graph_note}')
                        row = {'use_graphs': use_graphs, 'constructor_seconds': projector.constructor_seconds,
                               'source_h2d_bytes': projector.source_h2d_bytes}
                        started = time.perf_counter()
                        sink_seconds = 0.0
                        next_z = 0
                        blocks = projection._ordered_radial_cuda_blocks(projector, OUTPUT_SHAPE[0], False)
                        try:
                            for first, block in blocks:
                                if first != next_z:
                                    raise RuntimeError('Output delivery was reordered or incomplete')
                                sink_started = time.perf_counter()
                                writer.consume_encoded_block(first, block.records, block.payload, packed=False)
                                sink_seconds += time.perf_counter() - sink_started
                                next_z += len(block.records)
                        finally:
                            blocks.close()
                        writer.finalize()
                        if next_z != OUTPUT_SHAPE[0]:
                            raise RuntimeError('Output stream did not cover the requested volume')
                        row.update(projection_and_sink_seconds=time.perf_counter() - started,
                                   sink_seconds=sink_seconds, nrrd_layers_completed=sum(load.counts) - before)
                        for name in ('cuda_graph_blocks', 'cuda_graph_seconds', 'empty_encoded_blocks', 'kernel_seconds', 'metadata_seconds',
                                     'pack_seconds', 'd2h_seconds', 'metadata_d2h_bytes', 'payload_d2h_bytes'):
                            row[name] = getattr(projector, name)
                        if use_graphs and projector.cuda_graph_blocks == 0:
                            raise RuntimeError('No CUDA graph was replayed')
                    report['runs'].append(row)
                    save()
                    print(row, flush=True)
                except BaseException:
                    writer.discard()
                    raise
            load.close()
            report['background_full_layers'] = sum(load.counts)
            report['background_mirror_layers'] = sum(load.counts)
            load = None
            # Decode all four full output volumes after stopping load, so
            # validation does not interfere with the next measured projection.
            for ordinal, row in enumerate(report['runs']):
                path = workspace / f'projection-{ordinal}.cvol'
                row['decoded'] = decode_store(path)
                store = interpolation.RawBBoxMaskStore.open(path)
                try:
                    for z, expected in reference.items():
                        if digest(store.decode_slice(z)) != expected:
                            raise RuntimeError(f'Independent reference mismatch at z={z}')
                finally:
                    store.close()
                save()
            if len({r['decoded']['sha256'] for r in report['runs']}) != 1:
                raise RuntimeError('Full direct/graph decoded output streams differ')
            report['all_exact'] = True
            save()
        finally:
            try:
                if load is not None:
                    load.close()
            finally:
                source._mmap.close()


if __name__ == '__main__':
    main()
