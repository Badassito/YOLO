"""Qualify an experimental BF16 mask union against the shipping FP32 CUDA kernel.

Run only while the GPU is reserved. --capture accepts NPZ files containing head
[37,anchors], proto [32,ph,pw], and input_hw [height,width], all from one actual
backend forward pass before compaction. Only --output-dir receives artifacts.
The measurements exclude inference and must not predict multi-GPU walltime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as functional

from tools.bf16_proto_union import PreparedUnion


def synthetic_cases(production_size=False):
    rng = np.random.default_rng(20260912)
    for count in (0, 1, 16, 65, 257):
        for coverage in ((0.,) if count == 0 else (.05, 1.)):
            ph, pw = (768, 768) if production_size else (63, 66)
            anchors = 193536 if production_size else max(300, count + 3)
            head = np.zeros((37, anchors), np.float32)
            proto = rng.normal(size=(32, ph, pw)).astype(np.float32)
            if count:
                head[0, :count] = rng.uniform(0, pw * 4, count)
                head[1, :count] = rng.uniform(0, ph * 4, count)
                head[2, :count] = pw * 4 * coverage**.5
                head[3, :count] = ph * 4 * coverage**.5
                if coverage == 1:
                    head[0, :count] = pw * 2
                    head[1, :count] = ph * 2
                head[4, :count] = rng.uniform(.5, 1, count)
                head[4, count - 1] = .5  # equality at the admission threshold
                head[5:, :count] = rng.normal(size=(32, count))
            head[4, count] = np.nextafter(np.float32(.5), np.float32(0))
            head[4, count + 1] = np.nan  # must not enter compaction
            yield f'synthetic_{ph}x{pw}_n{count}_coverage{coverage}', head, proto, (ph * 4, pw * 4), count
    if not production_size:
        # Exact BF16 inputs isolate control/crop/count/default semantics from
        # numerical approximation. Includes >one detection tile and NaN boxes.
        ph, pw, anchors = 13, 34, 73
        head = np.zeros((37, anchors), np.float32)
        proto = rng.integers(-2, 3, size=(32, ph, pw)).astype(np.float32)
        head[0:2] = 16
        head[2:4] = 30
        head[4] = np.linspace(.49, .99, anchors, dtype=np.float32)
        head[5:] = rng.integers(-2, 3, size=(32, anchors))
        head[:4, -1] = np.nan
        yield 'exact_integer_semantics', head, proto, (52, 136), int((head[4] >= .5).sum())


def load_capture(path):
    with np.load(path, allow_pickle=False) as data:
        head, proto = np.asarray(data['head']), np.asarray(data['proto'])
        if head.ndim == 3 and head.shape[0] == 1:
            head = head[0]
        if proto.ndim == 4 and proto.shape[0] == 1:
            proto = proto[0]
        if head.dtype != np.float32 or proto.dtype != np.float32:
            raise TypeError(f'{path}: actual FP32 backend outputs are required')
        return (path.stem, np.ascontiguousarray(head), np.ascontiguousarray(proto),
                tuple(map(int, data['input_hw'])), int((head[4] >= np.float32(.5)).sum()))


def mask_metrics(reference, candidate):
    from scipy import ndimage

    reference, candidate = reference.astype(bool), candidate.astype(bool)
    intersection = int((reference & candidate).sum())
    union = int((reference | candidate).sum())
    structure = np.ones((3, 3), dtype=bool)
    rlabels, rn = ndimage.label(reference, structure)
    clabels, cn = ndimage.label(candidate, structure)
    rpresent = np.unique(rlabels[candidate])
    cpresent = np.unique(clabels[reference])
    return {'changed_pixels': int((reference != candidate).sum()),
            'added_pixels': int((candidate & ~reference).sum()),
            'removed_pixels': int((reference & ~candidate).sum()),
            'iou': intersection / union if union else 1.,
            'reference_components_8': int(rn), 'candidate_components_8': int(cn),
            'reference_components_with_no_overlap': int(rn - (rpresent > 0).sum()),
            'candidate_components_with_no_overlap': int(cn - (cpresent > 0).sum()),
            'removed_farther_than_one_pixel': int((reference & ~ndimage.binary_dilation(candidate, structure)).sum()),
            'added_farther_than_one_pixel': int((candidate & ~ndimage.binary_dilation(reference, structure)).sum())}


def quality(prepared):
    prepared.run_reference()
    prepared.run_bf16()
    prepared.synchronize()
    ref, ref_conf = (value.cpu().numpy() for value in prepared.reference)
    cand, cand_conf = (value.cpu().numpy() for value in prepared.candidate)
    delta = np.abs(ref.astype(np.float64) - cand.astype(np.float64))
    rows = {'count': int(prepared.count.item()),
            'max_absolute_logit_delta': float(delta.max()),
            'mean_absolute_logit_delta': float(delta.mean()),
            'p99_absolute_logit_delta': float(np.quantile(delta, .99)),
            'changed_confidence_pixels': int((ref_conf != cand_conf).sum()),
            'max_absolute_confidence_delta': float(np.abs(ref_conf - cand_conf).max()),
            'proto_mask': mask_metrics(ref > 0, cand > 0)}
    # Threshold remains after the same FP32 bilinear upsample used in production.
    resized = [functional.interpolate(value[None, None], size=(prepared.ih, prepared.iw),
                                      mode='bilinear', align_corners=False)[0, 0] > 0
               for value in (prepared.reference[0], prepared.candidate[0])]
    rows['upsampled_mask'] = mask_metrics(*(value.cpu().numpy() for value in resized))
    return rows


def event_ms(function, repeats):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def timings(prepared, rounds):
    operations = {'reference_total': prepared.run_reference, 'bf16_total': prepared.run_bf16,
                  'compaction': prepared.compact, 'bf16_conversion': prepared.convert,
                  'reference_union': prepared.union_reference, 'bf16_union': prepared.union_bf16}
    for fn in operations.values():
        fn()
    prepared.synchronize()
    slowest = max(event_ms(fn, 1) for fn in operations.values())
    repeats = max(1, min(30, int(15 / max(slowest, .01))))
    samples = {name: [] for name in operations}
    for index in range(rounds):
        order = list(operations) if index % 2 == 0 else list(reversed(operations))
        for name in order:
            samples[name].append(event_ms(operations[name], repeats))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {'median_ms': medians, 'samples_ms': samples, 'repeats': repeats,
            'total_speedup_including_conversion': medians['reference_total'] / medians['bf16_total']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capture', type=Path, action='append', default=[])
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--production-size', action='store_true')
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--quality-only', action='store_true')
    parser.add_argument('--pixels', type=int, default=64)
    parser.add_argument('--det-tile', type=int, default=16)
    args = parser.parse_args()
    if args.heat_seconds < 0 or args.rounds < 1:
        parser.error('Heat time must be nonnegative and rounds positive')
    if not args.capture and not args.synthetic:
        parser.error('Choose --capture or --synthetic')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    import triton
    report = {'scope': 'Experimental C32 BF16 inputs/FP32 MMA accumulation; no shipping changes. '
                      'CUDA-event component measurements exclude model inference, pipeline contention, '
                      'snapshot copies, and allocations; not an H100 walltime estimate.',
              'limitations': 'Phase timings are separate replays and must not be added. Quality covers '
                             'prototype and bilinearly upsampled network masks; native projection, '
                             'morphology, final unions, and anatomical feature survival are not evaluated.',
              'torch': torch.__version__, 'triton': triton.__version__,
              'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
              'heat_seconds_requested': args.heat_seconds, 'cases': [],
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ('bf16_proto_union.py', 'qualify_bf16_proto_union.py')}}
    output = args.output_dir / 'qualification.json'
    def save():
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    if not args.quality_only and args.heat_seconds:
        # 128 MiB scratch, below the reserved-window memory budget.
        a = torch.full((4096, 4096), .001, device='cuda')
        b = torch.empty_like(a)
        start = time.perf_counter()
        while time.perf_counter() - start < args.heat_seconds:
            torch.mm(a, a, out=b)
            torch.cuda.synchronize()
        report['heat_seconds_actual'] = time.perf_counter() - start
        del a, b
        torch.cuda.empty_cache()
    cases = ((load_capture(path), path) for path in args.capture)
    import itertools
    if args.synthetic:
        cases = itertools.chain(cases, ((row, None) for row in synthetic_cases(args.production_size)))
    for (name, head, proto, input_hw, expected_count), path in cases:
        prepared = PreparedUnion(torch.from_numpy(head).to('cuda'), torch.from_numpy(proto).to('cuda'),
                                 input_hw, pixels=args.pixels, det_tile=args.det_tile)
        result = quality(prepared)
        if expected_count is not None and result['count'] != expected_count:
            raise AssertionError(f'{name}: wrong admitted count {result["count"]} != {expected_count}')
        if name == 'exact_integer_semantics':
            assert result['max_absolute_logit_delta'] == 0
            assert result['changed_confidence_pixels'] == 0
        receipt = prepared.tensorcore_receipt()
        if not receipt['bf16_mma_present']:
            raise RuntimeError('No actual BF16 MMA instruction in compiled PTX')
        row = {'name': name, 'head_shape': list(head.shape), 'proto_shape': list(proto.shape),
               'input_hw': input_hw, 'quality': result, 'compiler': receipt,
               'extra_bf16_workspace_bytes': prepared.extra_bf16_workspace_bytes,
               'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path else None}
        if not args.quality_only:
            row['timing'] = timings(prepared, args.rounds)
        report['cases'].append(row)
        save()
        print(json.dumps(row), flush=True)
        if len(report['cases']) == 1:
            (args.output_dir / 'bf16_union.ptx').write_text(prepared.compiled_kernel.asm['ptx'], encoding='utf-8')
        prepared.synchronize()
        del prepared
        torch.cuda.empty_cache()
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
