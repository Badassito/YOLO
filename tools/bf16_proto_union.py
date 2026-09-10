"""Experimental C32 BF16 TensorCore union; never imported by shipping XTA.

The FP32 production compactor owns confidence, boxes, and the device-side count.
Only coefficients and prototypes are rounded to BF16. Each Triton program holds
a fixed 16-detection by 64-pixel dot tile, immediately crops and reduces it, and
loops through *all* retained detections. No detection-by-image workspace exists.
This changes logits near zero and is a qualification prototype, not a backend.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _convert_coefficients(source, count, output, ANCHORS: tl.constexpr,
                          BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(count)
    value = tl.load(source + offset, (offset < ANCHORS * 32) & (offset < n * 32), 0)
    tl.store(output + offset, value.to(tl.bfloat16), offset < ANCHORS * 32)


@triton.jit
def _bf16_union(proto, coefficients, boxes, confidences, count,
                logits_output, confidence_output,
                PH: tl.constexpr, PW: tl.constexpr,
                PIXELS: tl.constexpr, DET_TILE: tl.constexpr):
    pixel = tl.program_id(0) * PIXELS + tl.arange(0, PIXELS)
    channel = tl.arange(0, 32)
    detection = tl.arange(0, DET_TILE)
    x, y = pixel % PW, pixel // PW
    n = tl.load(count)
    best = tl.full((PIXELS,), -6.0, tl.float32)
    best_confidence = tl.full((PIXELS,), 0.0, tl.float32)
    # Reused across all detection tiles. The prototypes are already BF16 here;
    # input conversion is deliberately timed separately AND in the full path.
    prototype_values = tl.load(proto + channel[:, None] * (PH * PW) + pixel[None, :],
                               pixel[None, :] < PH * PW, 0)
    for start in range(0, n, DET_TILE):
        d = start + detection
        coefficient_values = tl.load(coefficients + d[:, None] * 32 + channel[None, :],
                                     d[:, None] < n, 0)
        dot = tl.dot(coefficient_values, prototype_values, out_dtype=tl.float32)
        x1 = tl.load(boxes + d * 4, d < n, 1.0)
        y1 = tl.load(boxes + d * 4 + 1, d < n, 1.0)
        x2 = tl.load(boxes + d * 4 + 2, d < n, 0.0)
        y2 = tl.load(boxes + d * 4 + 3, d < n, 0.0)
        score = tl.load(confidences + d, d < n, 0.0)
        # Match the FP32 kernel's rejection predicate, including NaN boxes.
        rejected = ((x[None, :].to(tl.float32) < x1[:, None]) |
                    (x[None, :].to(tl.float32) >= x2[:, None]) |
                    (y[None, :].to(tl.float32) < y1[:, None]) |
                    (y[None, :].to(tl.float32) >= y2[:, None]))
        inside = (~rejected) & (d[:, None] < n) & (pixel[None, :] < PH * PW)
        # fmaxf ignores a NaN candidate. Zero-positive confidence is evaluated
        # for every cropped detection, independent of which logit wins the max.
        candidate = tl.where(inside & (dot == dot), dot, float('-inf'))
        best = tl.maximum(best, tl.max(candidate, axis=0))
        positive_score = tl.where(inside & (dot > 0.0), score[:, None], 0.0)
        best_confidence = tl.maximum(best_confidence, tl.max(positive_score, axis=0))
    tl.store(logits_output + pixel, best, pixel < PH * PW)
    tl.store(confidence_output + pixel, best_confidence, pixel < PH * PW)


class PreparedUnion:
    """Reusable buffers for one immutable, FP32, captured backend-output pair.

    All launches use the construction stream. Call ``synchronize`` before
    releasing this object or modifying its input owners. The harness deliberately
    does not attach this experimental implementation to the production pipeline.
    """

    def __init__(self, head, proto, input_hw, threshold=0.5, *, pixels=64, det_tile=16):
        from XTA import inference

        if (head.ndim != 2 or head.shape[0] != 37 or proto.ndim != 3 or
                proto.shape[0] != 32 or min(proto.shape[1:]) <= 0):
            raise ValueError('Expected head[37,anchors] and proto[32,ph,pw]')
        if head.shape[1] <= 0 or head.shape[1] > (2**31 - 1) // 32:
            raise ValueError('Invalid anchor extent for int32 production compaction')
        if head.dtype != torch.float32 or proto.dtype != torch.float32:
            raise TypeError('Capture must retain the actual FP32 head and prototypes')
        if not head.is_cuda or proto.device != head.device:
            raise ValueError('Head and prototypes must reside on the same CUDA device')
        if not head.is_contiguous() or not proto.is_contiguous():
            raise ValueError('Use contiguous immutable capture tensors')
        if pixels not in (32, 64, 128) or det_tile not in (16, 32):
            raise ValueError('Only qualified fixed-size MMA tile candidates are supported')
        if not math.isfinite(float(threshold)):
            raise ValueError('Confidence threshold must be finite')
        self.ih, self.iw = map(int, input_hw)
        if min(self.ih, self.iw) <= 0:
            raise ValueError('Input height and width must be positive')
        self.head, self.proto = head, proto
        self.anchors, self.ph, self.pw = int(head.shape[1]), int(proto.shape[1]), int(proto.shape[2])
        self.threshold, self.pixels, self.det_tile = float(threshold), pixels, det_tile
        self.stream = torch.cuda.current_stream(head.device)
        self.kernels = inference._resident_mask_kernels()
        if self.kernels is None:
            raise RuntimeError('Production FP32 reference kernel unavailable')
        self.cp = self.kernels.cp
        self.external = inference._cupy_external_stream(self.cp, self.stream)
        device = head.device
        self.indices = torch.empty(self.anchors, dtype=torch.int32, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.coefficients = torch.empty((self.anchors, 32), device=device)
        self.boxes = torch.empty((self.anchors, 4), device=device)
        self.confidences = torch.empty(self.anchors, device=device)
        self.coefficients_bf16 = torch.empty_like(self.coefficients, dtype=torch.bfloat16)
        self.proto_bf16 = torch.empty_like(proto, dtype=torch.bfloat16)
        self.reference = [torch.empty((self.ph, self.pw), device=device) for _ in range(2)]
        self.candidate = [torch.empty_like(value) for value in self.reference]
        values = [head, proto, self.indices, self.count, self.coefficients, self.boxes,
                  self.confidences, *self.reference]
        self.cupy_owners = [self.cp.asarray(value) for value in values]
        self.compiled_kernel = None

    def _check_stream(self):
        if torch.cuda.current_stream(self.head.device) != self.stream:
            raise RuntimeError('Experimental buffers must remain on their construction stream')

    def compact(self):
        self._check_stream()
        head, _, indices, count, coeff, boxes, confs, _, _ = self.cupy_owners
        self.count.zero_()
        self.kernels.compact_f32_tiled(
            ((self.anchors + 255) // 256,), (256,),
            (head, np.int32(self.anchors), np.float32(self.threshold), indices, count,
             np.int32(self.ih), np.int32(self.iw), np.uintp(0), np.int32(32),
             np.int32(self.ph), np.int32(self.pw), np.int32(self.ih), np.int32(self.iw),
             coeff, boxes, confs), stream=self.external)

    def union_reference(self):
        self._check_stream()
        _, proto, _, count, coeff, boxes, confs, logits, confidence = self.cupy_owners
        self.kernels.union_f32_f32_tiled(
            ((self.pw + 31) // 32, (self.ph + 3) // 4), (32, 4),
            (proto, coeff, boxes, confs, count, np.int32(32), np.int32(self.ph),
             np.int32(self.pw), logits, confidence), stream=self.external)
        return self.reference

    def convert(self):
        self._check_stream()
        self.proto_bf16.copy_(self.proto)
        _convert_coefficients[(triton.cdiv(self.anchors * 32, 1024),)](
            self.coefficients, self.count, self.coefficients_bf16,
            ANCHORS=self.anchors, BLOCK=1024)

    def union_bf16(self):
        self._check_stream()
        self.compiled_kernel = _bf16_union[(triton.cdiv(self.ph * self.pw, self.pixels),)](
            self.proto_bf16, self.coefficients_bf16, self.boxes, self.confidences,
            self.count, *self.candidate, PH=self.ph, PW=self.pw,
            PIXELS=self.pixels, DET_TILE=self.det_tile, num_warps=4)
        return self.candidate

    def run_reference(self):
        self.compact()
        return self.union_reference()

    def run_bf16(self):
        self.compact()
        self.convert()
        return self.union_bf16()

    def synchronize(self):
        self.stream.synchronize()

    @property
    def extra_bf16_workspace_bytes(self):
        return (self.coefficients_bf16.numel() + self.proto_bf16.numel()) * 2

    def tensorcore_receipt(self):
        if self.compiled_kernel is None:
            raise RuntimeError('Launch the candidate before requesting a compiler receipt')
        ptx = self.compiled_kernel.asm['ptx']
        instructions = sorted(set(line.strip() for line in ptx.splitlines()
                                  if 'mma.' in line and 'bf16' in line))
        return {'bf16_mma_present': bool(instructions), 'bf16_mma_instructions': instructions,
                'registers': self.compiled_kernel.n_regs,
                'shared_memory_bytes': self.compiled_kernel.metadata.shared,
                'pixels_per_program': self.pixels, 'detections_per_dot': self.det_tile,
                'maximum_intermediate_dot_elements': self.pixels * self.det_tile}
