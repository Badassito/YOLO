"""Functionally qualify uncropped spherical CUDA source addresses above 4 GiB.

This is a correctness check, not a benchmark. The only foreground is a 64x64
stamp in shell 501 of a (512,3072,3072) uint8 mask. Three complete source planes
are compared with the bounded CPU reference. The temporary mask is removed
after a successful CUDA fence; a JSON record is retained in --output-dir.

Example:
  python tools/qualify_spherical_large_address.py --output-dir PATH_TO_SCRATCH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('--output-dir must be outside the source repository')
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'tmp'
    temporary.mkdir(exist_ok=True)
    for name in ('cupy', 'numba'):
        (output / name).mkdir(exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR', str(output / 'cupy'))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(output / 'numba'))

    import numpy as np
    from XTA.qsc import qsc_inverse
    from XTA.spherical_geometry import build_spherical_view_infos
    from XTA.spherical_projection import _project_spherical_block
    from XTA.spherical_projection_cuda import SphericalCudaProjector, SphericalCudaProjectionUnsafeFailure

    token = uuid.uuid4().hex[:12]
    mask_path = temporary / f'spherical-large-address-{token}.u8'
    report_path = output / f'large-address-qualification-{token}.json'
    source_shape, shape = (512, 3072, 3072), (1025, 1025, 1025)
    source_bytes = int(np.prod(source_shape, dtype=np.int64))
    stamp_shell = 501
    view = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=1.,
                                      patch_size=3072, tilted_views=())[0]
    center_row = view.spherical_face_intervals // 2 - view.spherical_v_origin - 128
    center_col = view.spherical_face_intervals // 2 - view.spherical_u_origin + 96
    y0, y1, x0, x1 = center_row - 32, center_row + 32, center_col - 32, center_col + 32
    first_address = (stamp_shell * source_shape[1] + y0) * source_shape[2] + x0
    last_address = (stamp_shell * source_shape[1] + y1 - 1) * source_shape[2] + x1 - 1
    if not 2**32 < first_address <= last_address < source_bytes:
        raise AssertionError('The qualification stamp must lie wholly beyond the 32-bit address boundary')
    u = -1 + 2 * (center_col + view.spherical_u_origin) / view.spherical_face_intervals
    v = 1 - 2 * (center_row + view.spherical_v_origin) / view.spherical_face_intervals
    direction = qsc_inverse(view.spherical_face, u, v)
    sample_xyz = (np.asarray(shape[::-1]) - 1) / 2 + view.spherical_radii[stamp_shell] * direction
    middle_z = int(np.rint(sample_xyz[2]))
    planes = (middle_z - 3, middle_z, middle_z + 3)
    record = {
        'qualification': 'spherical_uncropped_cuda_64bit_source_address',
        'purpose': 'functional_correctness_only',
        'passed': False,
        'source_shape': source_shape,
        'source_bytes': source_bytes,
        'source_mask': str(mask_path),
        'stamp_shell_index': stamp_shell,
        'stamp_radius': view.spherical_radii[stamp_shell],
        'stamp_bounds_yxyx': [y0, y1, x0, x1],
        'foreground_byte_address_first': first_address,
        'foreground_byte_address_last': last_address,
        'output_shape': shape,
        'qsc_face': view.spherical_face,
        'qsc_face_intervals': view.spherical_face_intervals,
        'stamp_center_source_xyz': sample_xyz.tolist(),
        'slices': [],
    }
    source = None
    settled = True
    try:
        print(f'Preparing uncropped uint8 source: {source_shape}, {source_bytes} bytes.', flush=True)
        source = np.memmap(mask_path, mode='w+', dtype=np.uint8, shape=source_shape)
        # Explicit bounded initialization avoids assumptions about new mmap pages.
        flat = source.reshape(-1)
        for first in range(0, source_bytes, 64 * 1024**2):
            flat[first:first + 64 * 1024**2] = 0
        del flat
        source[stamp_shell, y0:y1, x0:x1] = 1
        source.flush()
        print(f'Only foreground addresses: {first_address} through {last_address}; all exceed 2**32.', flush=True)
        radii = np.asarray(view.spherical_radii)
        rotation = np.asarray(view.spherical_rotation_xyz).reshape(3, 3)
        with SphericalCudaProjector(source, view, shape, bboxes=None, device_index=args.device,
                                    block_bytes=shape[1] * shape[2]) as projector:
            if projector.source_layout != 'dense_u8' or projector.source_h2d_bytes != source_bytes:
                raise AssertionError('Qualification must upload the complete uncropped >4 GiB source')
            record['source_layout'] = projector.source_layout
            record['source_h2d_bytes'] = projector.source_h2d_bytes
            record['max_output_block_depth'] = projector.max_block_depth
            record['device'] = args.device
            properties = projector._cp.cuda.runtime.getDeviceProperties(args.device)
            name = properties['name']
            record['device_name'] = name.decode() if isinstance(name, bytes) else str(name)
            for z in planes:
                expected = _project_spherical_block(source, view, radii, rotation, shape, z, 1)
                actual = projector.project(z, 1)
                mismatches = int(np.count_nonzero(actual != expected))
                foreground = int(np.count_nonzero(actual))
                if mismatches or foreground == 0:
                    raise AssertionError(f'Source Z={z}: {mismatches} mismatches, {foreground} foreground voxels')
                compact = []
                for packed in (False, True):
                    encoded = projector.project_encoded(z, 1, packed=packed)
                    projector._validate_encoded_preflight(expected, encoded)
                    compact.append({'packed': packed, 'payload_bytes': encoded.payload.size})
                record['slices'].append({
                    'z': z, 'pixels_checked': expected.size, 'mismatches': mismatches,
                    'foreground_voxels': foreground,
                    'sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                    'compact_outputs': compact,
                })
                print(f'Source Z={z}: all {expected.size} voxels match; {foreground} foreground voxels from >4 GiB addresses.', flush=True)
            record['preflight_planes'] = projector.preflight_planes
        record['passed'] = True
    except SphericalCudaProjectionUnsafeFailure as exc:
        settled = False
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # A failed fence preserves all potentially borrowed owners and evidence.
        if settled and source is not None:
            source._mmap.close()
            source = None
        if settled and mask_path.exists():
            resolved = mask_path.resolve(strict=True)
            if resolved.parent != temporary.resolve(strict=True):
                raise RuntimeError('Temporary mask cleanup target escaped its task directory')
            resolved.unlink()
        record['cuda_stream_settled'] = settled
        record['temporary_source_removed'] = not mask_path.exists()
        report_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print(f'Correctness record: {report_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
