"""ABBA source-crop uploads on production plane strides, plus projector parity.

The generated uint8 source is a Scratch memmap, not a resident 11 GiB volume.
Only crop pages are initialized. Timings measure this upload component and must
not be interpreted as full-pipeline/H100 predictions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cylindrical_cuda_projection as cuda, cylindrical_projection as radial
from XTA.geometry import radial_global_radii
from XTA.cylindrical_geometry import build_radial_view_infos
from XTA.spherical_geometry import build_spherical_view_infos
from XTA.spherical_projection_cuda import SphericalCudaProjector
from XTA.spherical_projection import _project_spherical_block
from tools.benchmark_radial_setup import heatsoak


@contextmanager
def pipeline_mode(enabled):
    key = 'YOLO_TTA_CROPPED_UPLOAD_PIPELINE'
    old = os.environ.get(key)
    os.environ[key] = '1' if enabled else '0'
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def slice_boxes(source):
    boxes = np.zeros((source.shape[0], 4), np.int64)
    for z, plane in enumerate(source):
        yy, xx = np.nonzero(plane)
        if yy.size:
            boxes[z] = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    return boxes


def packed_offsets(boxes):
    sizes = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    return np.r_[np.uint64(0), np.cumsum(sizes, dtype=np.uint64)]


def upload_metrics(projector):
    names = ('source_upload_pipeline', 'source_upload_stage_bytes', 'source_upload_copy_count',
             'source_upload_stream_fences', 'source_upload_lane_waits', 'source_upload_lane_wait_seconds',
             'source_pack_seconds', 'source_pack_backend', 'source_upload_seconds', 'constructor_seconds')
    return {name: getattr(projector, name, None) for name in names}


def qualify_projectors(device):
    shape, size = (11, 17, 19), 16
    rv = build_radial_view_infos(*shape, targets=('transverse',), min_radius=.5,
                                patch_size=size, tilted_views=())[0]
    sv = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=.5,
                                    patch_size=size, tilted_views=())[0]
    records = []
    for view in (rv, sv):
        source = np.zeros((view.num_slices, size, size), np.uint8)
        source[:, 1:15, 2:14] = 255
        source[::3] = 0
        boxes = slice_boxes(source)
        packed = np.concatenate([source[z, y0:y1, x0:x1].ravel()
                                 for z, (y0, y1, x0, x1) in enumerate(boxes)])
        if view.family == 'radial':
            radii = np.asarray(radial_global_radii(view), np.float64)
            plan = radial._build_radial_plane_plan(view, radii, shape)
            metadata = radial._radial_projection_metadata(view, source.shape, shape, plan)
            expected = np.stack([radial._pull_radial_chunk(source, view, radii, shape, z, 0,
                shape[1] * shape[2]).reshape(shape[1:]) for z in range(shape[0])])
            make = lambda: cuda.RadialCudaProjector(source, plan, metadata, view, shape,
                boxes, True, device, upload_bytes=37, reserve_bytes=0)
        else:
            expected = _project_spherical_block(source, view, np.asarray(view.spherical_radii),
                np.asarray(view.spherical_rotation_xyz).reshape(3, 3), shape, 0, shape[0], boxes)
            make = lambda: SphericalCudaProjector(source, view, shape, boxes,
                device_index=device, upload_bytes=37, reserve_bytes=0)
        for enabled in (False, True):
            with pipeline_mode(enabled), make() as projector:
                np.testing.assert_array_equal(projector._source_gpu.get()[:packed.size], packed)
                np.testing.assert_array_equal(projector.project(0, shape[0]), expected)
                assert bool(projector.source_upload_pipeline) == enabled
                records.append({'family': view.family, 'requested': enabled,
                                'exact_source_and_projection': True, **upload_metrics(projector)})
    return records


def run_upload(cp, source, boxes, offsets, stage_bytes, device, enabled, expected_sha):
    projector = object.__new__(cuda.RadialCudaProjector)
    projector._cp = cp
    projector._events = {}
    projector.source_h2d_bytes = int(offsets[-1])
    projector.source_pack_seconds = 0.
    projector.contract = SimpleNamespace(arrays={'bboxes': boxes, 'source_offsets': offsets})
    with cp.cuda.Device(device):
        stream = projector._stream = cp.cuda.Stream(non_blocking=True)
        pool, pin_pool = cp.cuda.MemoryPool(), cp.cuda.PinnedMemoryPool()
        with cp.cuda.using_allocator(pool.malloc), stream:
            projector._upload_pin = pin_pool.malloc(stage_bytes)
            projector._upload_stage = np.frombuffer(projector._upload_pin, np.uint8, count=stage_bytes)
            try:
                with pipeline_mode(enabled):
                    started = time.perf_counter()
                    projector._upload_cropped_source(source)
                    elapsed = time.perf_counter() - started
                # Verification is outside the timer, bounded to one stage-size host chunk.
                digest = hashlib.sha256()
                for first in range(0, projector.source_h2d_bytes, stage_bytes):
                    digest.update(projector._source_gpu[first:first + stage_bytes].get().tobytes())
                assert digest.hexdigest() == expected_sha
                assert bool(projector.source_upload_pipeline) == enabled
                return {'requested': enabled, 'upload_seconds': elapsed, 'packed_sha256': digest.hexdigest(),
                        'source_h2d_bytes': projector.source_h2d_bytes, **upload_metrics(projector)}
            finally:
                # If this fence fails, none of the following owner release occurs.
                stream.synchronize()
                projector._events.clear()
                projector._source_gpu = projector._upload_stage = projector._upload_pin = None
                pool.free_all_blocks()
                pin_pool.free_all_blocks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--shells', type=int, default=192)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.shells < 64 or args.shells > 256 or args.rounds < 1 or args.heat_seconds < 0 or args.device < 0:
        parser.error('shells must be 64..256, rounds >= 1, heat seconds >= 0, device >= 0')
    root = args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    if root == repository or repository in root.parents:
        parser.error('generated upload evidence must be outside the repository, in task-specific Scratch')
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'upload-source-{os.getpid()}.u8'
    if path.exists():
        raise FileExistsError(path)
    import cupy as cp
    source = np.memmap(path, mode='w+', dtype=np.uint8, shape=(args.shells, 3072, 3072))
    boxes = np.zeros((args.shells, 4), np.int64)
    digest = hashlib.sha256()
    report = {'scope': 'source pack+H2D component, not end-to-end prediction', 'device': args.device,
              'shape': list(source.shape), 'source_logical_bytes': source.nbytes,
              'pinned_stage_bytes': 64 * 1024**2, 'heat_seconds': args.heat_seconds,
              'source_files': {'cylindrical_cuda_projection.py': hashlib.sha256(
                  Path(cuda.__file__).read_bytes()).hexdigest()}, 'runs': []}
    try:
        for shell in range(args.shells):
            if shell % 9 == 0:
                continue
            height, width = 1024 + (shell % 7) * 64, 1024 + (shell % 5) * 128
            y0, x0 = 100 + shell * 37 % 800, 80 + shell * 19 % 1200
            boxes[shell] = y0, y0 + height, x0, x0 + width
            crop = source[shell, y0:y0 + height, x0:x0 + width]
            crop[:] = ((np.arange(width, dtype=np.uint32) + shell) % 251).astype(np.uint8)[None, :]
            digest.update(np.ascontiguousarray(crop).tobytes())
        source.flush()
        offsets = packed_offsets(boxes)
        expected_sha = digest.hexdigest()
        assert int(offsets[-1]) > report['pinned_stage_bytes']
        if cuda._pack_radial_source_block_compiled is None:
            raise RuntimeError('compiled source packer is required for this qualification')
        cuda._pack_radial_source_block_compiled(np.asarray(source), boxes, offsets, 0, np.empty(0, np.uint8))
        if args.heat_seconds:
            heatsoak(args.heat_seconds, args.device)
        report['projector_parity'] = qualify_projectors(args.device)
        # Warm both upload variants and page mappings before the timed ABBA rounds.
        for enabled in (False, True):
            run_upload(cp, np.asarray(source), boxes, offsets, report['pinned_stage_bytes'],
                       args.device, enabled, expected_sha)
        for _ in range(args.rounds):
            for enabled in (False, True, True, False):
                result = run_upload(cp, np.asarray(source), boxes, offsets, report['pinned_stage_bytes'],
                                    args.device, enabled, expected_sha)
                report['runs'].append(result)
                print(json.dumps(result), flush=True)
                (root / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
        report['median_seconds'] = {str(enabled): statistics.median(
            r['upload_seconds'] for r in report['runs'] if r['requested'] == enabled) for enabled in (False, True)}
        report['speedup'] = report['median_seconds']['False'] / report['median_seconds']['True']
        (root / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
        print(json.dumps({'median_seconds': report['median_seconds'], 'speedup': report['speedup']}), flush=True)
    finally:
        source._mmap.close()
        path.unlink()


if __name__ == '__main__':
    main()
