"""Unprivileged v18.0.3 HGX smoke test for the resident keep backend.

This test needs no model or production data.  It builds deterministic components that
cross every contiguous GPU Z boundary, compares the opt-in GPU result byte-for-byte with
the established CPU implementation, and prints one JSON record suitable for indirect
cluster qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=0, help="visible GPUs to use; 0 uses all")
    parser.add_argument("--height", type=int, default=129)
    parser.add_argument("--width", type=int, default=131)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    for z in range(int(array.shape[0])):
        digest.update(np.ascontiguousarray(array[int(z)]).view(np.uint8))
    return digest.hexdigest()


def _fixture(z_dim: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(z_dim), int(height), int(width)), dtype=np.uint8)
    # Largest object spans every shard through face connectivity.
    mask[:, 7:18, 9:24] = np.uint8(1)
    # Second object spans every shard diagonally; 26-connectivity must retain it.
    for z in range(int(z_dim)):
        y = 38 + (z % 3)
        x = 45 + (z % 3)
        mask[z, y:y + 7, x:x + 7] = np.uint8(1)
    # Smaller independent objects exercise the top-N decision and dropped slices.
    mask[1:max(2, int(z_dim) // 3), 70:75, 72:77] = np.uint8(1)
    mask[max(1, int(z_dim) // 2):max(2, int(z_dim) // 2 + 2), 92:95, 99:102] = np.uint8(1)
    return mask


def main() -> None:
    args = _parser().parse_args()
    os.environ["YOLO_TTA_V1803_GPU_RESIDENT_TAIL"] = "1"
    os.environ["YOLO_TTA_V1803_GPU_RESIDENT_TAIL_REQUIRED"] = "1"

    from XTA.cuda_finalization import (
        artifact_plan_from_host,
        pack_binary_rows,
        try_apply_keep_largest_objects_multi_gpu,
        unpack_binary_rows,
    )

    if bool(args.plan_only):
        selected = max(1, int(args.gpus) if int(args.gpus) > 0 else 4)
        z_dim = max(16, selected * 8)
        fixture = _fixture(z_dim, int(args.height), int(args.width))
        packed = pack_binary_rows(fixture)
        restored = unpack_binary_rows(packed, int(args.width))
        artifact = artifact_plan_from_host(fixture.shape, tuple(range(selected)))
        artifact.validate()
        print(json.dumps({
            "status": "plan_only_ok",
            "gpus": int(selected),
            "shape": list(fixture.shape),
            "packed_bytes": int(packed.nbytes),
            "roundtrip_equal": bool(np.array_equal(fixture, restored)),
            "partitions": [[int(s.z0), int(s.z1)] for s in artifact.shards],
        }, sort_keys=True))
        return

    import torch  # type: ignore

    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable")
    visible = int(torch.cuda.device_count())
    selected = visible if int(args.gpus) <= 0 else min(visible, int(args.gpus))
    if selected <= 0:
        raise RuntimeError("no visible CUDA GPU was selected")
    devices = tuple(range(int(selected)))
    from XTA.topology import configure_gpu_slice_labeling_devices

    configure_gpu_slice_labeling_devices(tuple(f"cuda:{index}" for index in devices))
    z_dim = max(16, int(selected) * 8)
    fixture = _fixture(z_dim, int(args.height), int(args.width))

    from XTA.finalization import apply_keep_largest_objects_inplace

    with tempfile.TemporaryDirectory(prefix="xta-v1803-hgx-") as raw_tmp:
        root = Path(raw_tmp)
        gpu_input = np.memmap(root / "gpu_input.u8.dat", dtype=np.uint8, mode="w+", shape=fixture.shape)
        cpu_input = np.memmap(root / "cpu_input.u8.dat", dtype=np.uint8, mode="w+", shape=fixture.shape)
        gpu_input[:] = fixture
        cpu_input[:] = fixture
        gpu_input.flush(); cpu_input.flush()

        gpu_started = time.perf_counter()
        gpu_result = try_apply_keep_largest_objects_multi_gpu(
            gpu_input, int(args.keep), root / "gpu", keep_temp=True,
        )
        gpu_seconds = max(0.0, time.perf_counter() - gpu_started)
        if gpu_result is None:
            raise RuntimeError("required GPU resident-tail path returned no result")

        cpu_started = time.perf_counter()
        cpu_stats = apply_keep_largest_objects_inplace(
            cpu_input, int(args.keep), root / "cpu", keep_temp=True,
            prefer_memory=True, workers=max(1, os.cpu_count() or 1),
        )
        cpu_seconds = max(0.0, time.perf_counter() - cpu_started)
        equal = bool(np.array_equal(np.asarray(gpu_result.volume), np.asarray(cpu_input)))
        record = {
            "status": "ok" if equal else "mismatch",
            "visible_gpus": int(visible),
            "selected_gpus": int(selected),
            "shape": list(fixture.shape),
            "keep": int(args.keep),
            "equal": bool(equal),
            "gpu_sha256": _sha256(np.asarray(gpu_result.volume)),
            "cpu_sha256": _sha256(np.asarray(cpu_input)),
            "gpu_seconds": float(gpu_seconds),
            "cpu_seconds": float(cpu_seconds),
            "gpu_stats": dict(gpu_result.stats),
            "cpu_stats": dict(cpu_stats),
        }
        print(json.dumps(record, sort_keys=True))
        if not equal:
            raise RuntimeError("v18.0.3 GPU/CPU keep_objects byte mismatch")


if __name__ == "__main__":
    main()
