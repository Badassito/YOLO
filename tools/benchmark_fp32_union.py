"""Isolated direct FP32 packed-kernel qualification; run only with the GPU reserved."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from XTA import inference
from tools.benchmark_radial_setup import heatsoak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heat-seconds', type=float, default=60)
    args = parser.parse_args()
    if args.heat_seconds < 0:
        parser.error('heat seconds must be nonnegative')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.heat_seconds:
        heatsoak(args.heat_seconds, 0)
    kernels = inference._resident_mask_kernels()
    assert kernels is not None
    cp = kernels.cp
    external = inference._cupy_external_stream(cp, torch.cuda.current_stream())
    gen = torch.Generator(device='cuda').manual_seed(20260909)
    report = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'heat_seconds': args.heat_seconds, 'scope': 'count reset + compaction + union, CUDA events',
              'implementation': 'FP32 packed coefficients/boxes and lazy per-pixel prototype registers',
              'cases': []}
    for ph in (160, 384, 768):
        pw = ph
        ih = iw = ph * 4
        anchors = sum((ih // stride) * (iw // stride) for stride in (8, 16, 32))
        head = torch.zeros((37, anchors), device='cuda')
        proto = torch.randn((32, ph, pw), generator=gen, device='cuda')
        coeff = torch.empty((anchors, 32), device='cuda')
        boxes = torch.empty((anchors, 4), device='cuda')
        confs = torch.empty((anchors,), device='cuda')
        indices = [torch.empty((anchors,), device='cuda', dtype=torch.int32) for _ in range(2)]
        counts = [torch.empty((1,), device='cuda', dtype=torch.int32) for _ in range(2)]
        logits = [torch.empty((ph, pw), device='cuda') for _ in range(2)]
        confidence = [torch.empty((ph, pw), device='cuda') for _ in range(2)]
        # Owners remain live for every CuPy view until both streams are synchronized.
        ch, cp_proto, cc, cb, cf = map(cp.asarray, (head, proto, coeff, boxes, confs))
        ci, cn, cl, cq = [[cp.asarray(v) for v in row] for row in (indices, counts, logits, confidence)]

        def launch(mode):
            counts[mode].zero_()
            common = (ch, np.int32(anchors), np.float32(.5), ci[mode], cn[mode],
                      np.int32(ih), np.int32(iw), np.uintp(0))
            if mode:
                kernels.compact_f32_tiled(((anchors + 255) // 256,), (256,),
                    common + (np.int32(32), np.int32(ph), np.int32(pw), np.int32(ih), np.int32(iw), cc, cb, cf),
                    stream=external)
                kernels.union_f32_f32_tiled(((pw + 31) // 32, (ph + 3) // 4), (32, 4),
                    (cp_proto, cc, cb, cf, cn[mode], np.int32(32), np.int32(ph), np.int32(pw), cl[mode], cq[mode]),
                    stream=external)
            else:
                kernels.compact_f32(((anchors + 255) // 256,), (256,), common, stream=external)
                kernels.union_f32_f32(((ph * pw + 255) // 256,), (256,),
                    (ch, cp_proto, ci[mode], cn[mode], np.int32(anchors), np.int32(32), np.int32(ph),
                     np.int32(pw), np.int32(ih), np.int32(iw), cl[mode], cq[mode]), stream=external)

        def measure(mode, repeats):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                launch(mode)
            end.record()
            end.synchronize()
            return begin.elapsed_time(end) / repeats

        for retained in (0, 1, 8, 32, 128, 512):
            for coverage in ((0.,) if retained == 0 else (.1, 1.)):
                head.zero_()
                if retained:
                    head[:2, :retained] = torch.rand((2, retained), generator=gen, device='cuda') * ih
                    if coverage == 1.:
                        head[:2, :retained].fill_(ih * .5)
                    head[2:4, :retained] = ih * coverage ** .5
                    head[4, :retained] = .5 + torch.rand((retained,), generator=gen, device='cuda') * .5
                    head[5:, :retained] = torch.randn((32, retained), generator=gen, device='cuda')
                for _ in range(3):
                    launch(0); launch(1)
                torch.cuda.synchronize()
                assert all(torch.equal(a[0].view(torch.uint8), a[1].view(torch.uint8))
                           for a in (counts, logits, confidence)), (ph, retained, coverage)
                assert counts[0].item() == retained
                repeats = max(1, min(50, int(20 / max(measure(0, 1), measure(1, 1)))))
                timings = [[], []]
                for round_index in range(7):
                    for mode in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                        timings[mode].append(measure(mode, repeats))
                medians = list(map(statistics.median, timings))
                item = {'proto': [ph, pw], 'anchors': anchors, 'retained': retained,
                        'box_area_fraction': coverage, 'exact_logits_confidence_count': True,
                        'scalar_ms': medians[0], 'tiled_ms': medians[1],
                        'speedup': medians[0] / medians[1], 'repeats': repeats, 'samples_ms': timings}
                report['cases'].append(item)
                print(json.dumps({k: v for k, v in item.items() if k != 'samples_ms'}), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        torch.cuda.synchronize()
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
