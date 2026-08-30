"""PTA scheduling policy independent of geometry and publication.

This module owns CUDA-owner layout, shape-compatible device work packing,
free-VRAM admission, and deterministic allocation-failure splitting. Keeping
these policies outside the dataset engine makes scheduling independently
testable and avoids coupling it to source discovery or output formats.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple, TypeVar


GIB = 1024 ** 3
WorkT = TypeVar("WorkT")


def resolve_gpu_worker_layout(
    *,
    worker_budget: int,
    requested_frame_workers: int,
    gpu_count: int,
) -> Tuple[int, int]:
    """Return ``(CUDA owners, CPU render threads per owner)``.

    A CUDA context is owned by exactly one persistent process per visible
    device. The frame-worker request controls bounded CPU preparation rather
    than creating competing contexts on one device.
    """

    devices = max(1, int(gpu_count))
    cpu_budget = (
        int(requested_frame_workers)
        if int(requested_frame_workers) > 0
        else max(devices, int(worker_budget))
    )
    per_owner_budget = int(math.ceil(float(cpu_budget) / float(devices)))
    # Auto mode leaves CPU capacity for volume planning and input prefetch.
    # An explicit frame-worker budget is authoritative up to a conservative
    # per-owner ceiling for bounded host memory.
    per_owner_cap = 32 if int(requested_frame_workers) > 0 else 16
    threads_per_owner = max(1, min(per_owner_cap, per_owner_budget))
    return devices, threads_per_owner


def iter_compatible_work_batches(
    work: Sequence[WorkT],
    *,
    candidate_limit: int,
) -> Iterator[Tuple[WorkT, ...]]:
    """Pack shape-compatible source items into bounded policy calls."""

    limit = max(1, int(candidate_limit))
    grouped: Dict[Tuple[object, ...], List[WorkT]] = defaultdict(list)
    for item in work:
        key = (
            tuple(int(x) for x in item.image.shape),  # type: ignore[attr-defined]
            tuple(int(x) for x in item.mask.shape),  # type: ignore[attr-defined]
            tuple(int(x) for x in item.output_size),  # type: ignore[attr-defined]
            str(item.channel_kind),  # type: ignore[attr-defined]
            int(getattr(item, "channel_count", 1)),
            bool(getattr(item.image, "is_cuda", False)),  # type: ignore[attr-defined]
        )
        candidates = tuple(item.candidates)  # type: ignore[attr-defined]
        for start in range(0, len(candidates), limit):
            grouped[key].append(
                replace(item, candidates=candidates[start:start + limit])
            )

    for items in grouped.values():
        pending: List[WorkT] = []
        pending_count = 0
        for item in items:
            item_count = len(item.candidates)  # type: ignore[attr-defined]
            if pending and pending_count + item_count > limit:
                yield tuple(pending)
                pending = []
                pending_count = 0
            pending.append(item)
            pending_count += item_count
        if pending:
            yield tuple(pending)


def gpu_memory_candidate_limit(
    runtime: Mapping[str, object],
    work: Sequence[WorkT],
    *,
    requested_limit: int,
) -> int:
    """Cap a policy call using current free VRAM and output tensor geometry."""

    requested = max(1, int(requested_limit))
    if not work:
        return requested
    max_pixels = max(
        1,
        max(
            int(item.output_size[0]) * int(item.output_size[1])  # type: ignore[attr-defined]
            for item in work
        ),
    )
    # The policy materializes source/candidate floats, inverse-coordinate and
    # sampling grids, warped images/masks, and intensity/noise buffers.
    bytes_per_pixel = max(
        112 + 48 * max(1, int(getattr(item, "channel_count", 1)))
        for item in work
    )
    try:
        torch = runtime["torch"]
        free_bytes, _total_bytes = torch.cuda.mem_get_info(  # type: ignore[attr-defined]
            int(runtime["device_id"])
        )
        usable_bytes = max(0, int(free_bytes) - (2 * GIB))
        memory_limit = int(
            (float(usable_bytes) * 0.45) / float(max_pixels * bytes_per_pixel)
        )
    except Exception:
        return requested
    return max(1, min(requested, memory_limit))


def should_flush_ready_gpu_work(
    *,
    ready_candidates: int,
    effective_candidate_limit: int,
    producer_drained: bool,
) -> bool:
    """Launch the first full device batch while CPU producers continue.

    Waiting for two complete device batches defeats producer/consumer overlap
    when an outer task itself contains only two batches: CUDA starts only
    after all CPU rendering has finished. One full batch is sufficient; the
    bounded producer window can fill the following batch during the policy
    and encoder call.
    """

    return bool(
        bool(producer_drained)
        or int(ready_candidates) >= max(1, int(effective_candidate_limit))
    )


def split_work_batch(batch: Sequence[WorkT]) -> Tuple[Tuple[WorkT, ...], Tuple[WorkT, ...]]:
    """Split a failed policy batch in candidate order for deterministic retry."""

    total = sum(len(item.candidates) for item in batch)  # type: ignore[attr-defined]
    if total < 2:
        raise ValueError("A one-candidate GPU work batch cannot be split")
    left_target = max(1, total // 2)
    left: List[WorkT] = []
    right: List[WorkT] = []
    remaining_left = left_target
    for item in batch:
        candidates = tuple(item.candidates)  # type: ignore[attr-defined]
        take = min(len(candidates), remaining_left)
        if take:
            left.append(replace(item, candidates=candidates[:take]))
            remaining_left -= take
        if take < len(candidates):
            right.append(replace(item, candidates=candidates[take:]))
    if not left or not right:
        raise RuntimeError("Internal error while splitting a GPU work batch")
    return tuple(left), tuple(right)


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    """Recognize allocation failures without importing a CUDA framework."""

    message = f"{type(exc).__name__}: {exc}".lower()
    return (
        "outofmemory" in message
        or "out of memory" in message
        or "memory allocation" in message
    )


__all__ = [
    "gpu_memory_candidate_limit",
    "is_cuda_out_of_memory",
    "iter_compatible_work_batches",
    "resolve_gpu_worker_layout",
    "should_flush_ready_gpu_work",
    "split_work_batch",
]
