"""Compare exact scalar and column-hoisted Radial CUDA renderers.

The small native-T source keeps memory bounded; logical geometry and output
patches can still match production sizes. This isolates renderer performance,
not TensorRT throughput or cluster walltime. Outputs are written only to the
explicit output directory; reserve the GPU before running this benchmark.
"""
from dataclasses import replace
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cuda_backend as cb
from XTA.cylindrical_geometry import build_radial_view_infos
from tools.benchmark_radial_setup import heatsoak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--patch-sizes', type=int, nargs='+', default=[64, 768, 3072])
    parser.add_argument('--spatial-size', type=int, default=1536)
    parser.add_argument('--native-depth', type=int, default=64)
    args = parser.parse_args()
    if args.heat_seconds < 0 or min(args.patch_sizes + [args.spatial_size, args.native_depth]) <= 0:
        parser.error('sizes must be positive and heat seconds nonnegative')
    if args.native_depth * args.spatial_size**2 > 1024**3:
        parser.error('native source must remain within the 1 GiB qualification budget')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(2)
    engine = object.__new__(cb._GpuWorkerRenderEngine)
    engine.torch = torch
    engine.device = torch.device('cuda:0')
    engine._stream = torch.cuda.Stream()
    engine._logical_t = args.spatial_size
    engine._volume_gpu = torch.randint(0, 256, (args.native_depth, args.spatial_size, args.spatial_size),
                                      dtype=torch.uint8, device=engine.device)
    engine._fused_volume_ref = None
    engine._azimuthal_texture_lock = threading.RLock()
    torch.cuda.synchronize()
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'native_shape': list(engine._volume_gpu.shape), 'logical_shape': [args.spatial_size] * 3,
              'scope': 'renderer calls including allocation and two launches versus one; no model inference',
              'heat_seconds': heatsoak(args.heat_seconds, 0) if args.heat_seconds else 0, 'cases': []}

    def render(view, index, enabled):
        os.environ['YOLO_TTA_GPU_RADIAL_COLUMN_GEOMETRY'] = str(int(enabled))
        return engine._render_radial_native_resident_cuda(view, index)

    def measure(view, index, enabled, repeats):
        with torch.cuda.stream(engine._stream):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            engine._stream.synchronize()
            started = time.perf_counter()
            begin.record(engine._stream)
            for _ in range(repeats):
                render(view, index, enabled)
            end.record(engine._stream)
            end.synchronize()
            return begin.elapsed_time(end) / repeats, (time.perf_counter() - started) * 1000 / repeats

    for size in args.patch_sizes:
        views = build_radial_view_infos(*([args.spatial_size] * 3),
            targets=('transverse', 'sagittal', 'coronal'), min_radius=121., patch_size=size, tilted_views=())
        for base in ('transverse', 'sagittal', 'coronal'):
            original = next(view for view in views if view.radial_base_view == base)
            for direction, angle in (('vertical', 0.), ('horizontal', -30.)):
                view = replace(original, radial_tilted_source=bool(angle), tilt_direction=direction, tilt_angle_deg=angle)
                for index in sorted({0, view.num_slices - 1}):
                    with torch.cuda.stream(engine._stream):
                        scalar = render(view, index, False)
                        columns = render(view, index, True)
                        engine._stream.synchronize()
                        if not torch.equal(scalar.view(torch.uint8), columns.view(torch.uint8)):
                            raise AssertionError(f'Radial column parity failed: {base}/{size}/{angle}/{index}')
                    warm_ms = max(measure(view, index, enabled, 1)[0] for enabled in (False, True))
                    repeats = max(1, min(30, int(20 / max(warm_ms, .001))))
                    samples = [[], []]
                    for round_index in range(7):
                        for enabled in ((False, True) if round_index % 2 == 0 else (True, False)):
                            samples[int(enabled)].append(measure(view, index, enabled, repeats))
                    medians = [statistics.median(pair[0] for pair in values) for values in samples]
                    row = {'patch': size, 'base': base, 'direction': direction, 'tilt': angle,
                           'radius': float(view.radial_radii[index]), 'exact_bytes': True,
                           'scalar_ms': medians[0], 'columns_ms': medians[1],
                           'speedup': medians[0] / medians[1], 'repeats': repeats, 'samples_ms': samples}
                    report['cases'].append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != 'samples_ms'}), flush=True)
                    (args.output_dir / 'radial-columns.json').write_text(
                        json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
