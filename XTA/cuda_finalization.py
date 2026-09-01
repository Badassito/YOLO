"""Experimental v18.0.3 multi-GPU binary-tail primitives.

The production path remains host-authoritative unless explicitly enabled.  This
module is dependency-light at import time: CuPy/Torch are imported only inside
the physical-GPU entry point, while partitioning, bit packing, graph reduction,
and artifact contracts remain CPU-testable.

The first hardware path is intentionally transactional.  It uploads the settled
host final union into contiguous Z shards, keeps the local label rasters resident
while the CPU resolves only compact component equivalences, writes a separate
candidate volume, and returns that candidate only after every device succeeds.
Future D1-owner groups and interpolation cohorts can produce the same
``DistributedBinaryArtifact`` without the initial host upload.
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .workspace import _cpu_count, _env_flag, _env_float, _env_int


GIB = 1024 ** 3
ROW_WORD_BITS = 32


def _flush_array(array: object) -> None:
    flush = getattr(array, "flush", None)
    if callable(flush):
        flush()


def _close_array(array: object) -> None:
    try:
        from .runtime import close_memmap_array

        close_memmap_array(array)
    except Exception:
        mmap_obj = getattr(array, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass


def v1803_gpu_resident_tail_enabled() -> bool:
    """Return whether the qualified, default-off v18.0.3 resident-tail experiment is enabled."""

    return _env_flag("YOLO_TTA_V1803_GPU_RESIDENT_TAIL", False)


def v1803_gpu_resident_tail_required() -> bool:
    """Make resident-tail admission/execution failure fatal instead of falling back."""

    return _env_flag("YOLO_TTA_V1803_GPU_RESIDENT_TAIL_REQUIRED", False)


def v1803_gpu_tail_reserve_bytes() -> int:
    return int(max(1.0, _env_float("YOLO_TTA_V1803_GPU_TAIL_RESERVE_GIB", 8.0)) * GIB)


def v1803_gpu_tail_block_slices() -> int:
    return max(4, _env_int("YOLO_TTA_V1803_GPU_TAIL_BLOCK_SLICES", 32))


def row_word_count(width: int) -> int:
    return int((max(0, int(width)) + ROW_WORD_BITS - 1) // ROW_WORD_BITS)


def row_tail_mask(width: int) -> np.uint32:
    remainder = int(width) % ROW_WORD_BITS
    if remainder == 0:
        return np.uint32(0xFFFFFFFF)
    return np.uint32((1 << remainder) - 1)


@dataclass(frozen=True)
class ZPartition:
    rank: int
    z0: int
    z1: int

    @property
    def slices(self) -> int:
        return max(0, int(self.z1) - int(self.z0))


def contiguous_z_partitions(z_dim: int, device_count: int) -> Tuple[ZPartition, ...]:
    """Return deterministic, gap-free contiguous Z ownership for 1..N devices."""

    z = max(0, int(z_dim))
    count = max(1, min(max(1, int(device_count)), max(1, z)))
    q, r = divmod(z, count)
    out: List[ZPartition] = []
    cursor = 0
    for rank in range(count):
        length = int(q + (1 if rank < r else 0))
        out.append(ZPartition(rank=int(rank), z0=int(cursor), z1=int(cursor + length)))
        cursor += int(length)
    if int(cursor) != int(z):  # pragma: no cover - defensive arithmetic invariant
        raise RuntimeError(f"Z partition coverage {cursor} != {z}")
    return tuple(out)


def pack_binary_rows(mask: np.ndarray) -> np.ndarray:
    """Pack a binary ``(..., W)`` array into little-bit-order row-padded uint32 words."""

    source = np.asarray(mask)
    if source.ndim < 1:
        raise ValueError("binary row pack requires at least one dimension")
    width = int(source.shape[-1])
    words = row_word_count(width)
    flat_rows = np.ascontiguousarray(source != 0).reshape(-1, width)
    packed_u8 = np.packbits(flat_rows, axis=1, bitorder="little")
    padded_bytes = int(words) * 4
    if int(packed_u8.shape[1]) < padded_bytes:
        padded = np.zeros((int(packed_u8.shape[0]), padded_bytes), dtype=np.uint8)
        padded[:, : int(packed_u8.shape[1])] = packed_u8
        packed_u8 = padded
    packed = np.ascontiguousarray(packed_u8).view(np.uint32).reshape(
        tuple(int(v) for v in source.shape[:-1]) + (int(words),)
    )
    if int(words) > 0:
        packed[..., -1] &= row_tail_mask(width)
    return packed


def unpack_binary_rows(words: np.ndarray, width: int) -> np.ndarray:
    """Unpack row-padded uint32 words to a uint8 binary array of exact width."""

    packed = np.ascontiguousarray(np.asarray(words, dtype=np.uint32))
    expected_words = row_word_count(int(width))
    if packed.ndim < 1 or int(packed.shape[-1]) != int(expected_words):
        raise ValueError(
            f"packed row word count {packed.shape if packed.ndim else ()} != {expected_words}"
        )
    byte_rows = packed.reshape(-1, expected_words).view(np.uint8)
    unpacked = np.unpackbits(byte_rows, axis=1, bitorder="little")[:, : int(width)]
    return np.ascontiguousarray(
        unpacked.reshape(tuple(int(v) for v in packed.shape[:-1]) + (int(width),)),
        dtype=np.uint8,
    )


@dataclass
class DistributedBinaryShard:
    device_index: int
    z0: int
    z1: int
    layout: str
    storage: object = field(repr=False, compare=False)
    ready_event: Optional[object] = field(default=None, repr=False, compare=False)


@dataclass
class DistributedBinaryArtifact:
    """One logical binary volume with explicit device-local Z ownership."""

    shape_tyx: Tuple[int, int, int]
    shards: Tuple[DistributedBinaryShard, ...]
    layout: str
    source: str
    slice_any: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    slice_bboxes: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    transaction_token: str = ""

    def validate(self) -> None:
        z_dim, h, w = (int(v) for v in self.shape_tyx)
        if min(z_dim, h, w) <= 0:
            raise ValueError(f"invalid distributed binary shape {self.shape_tyx}")
        cursor = 0
        devices: set[int] = set()
        for shard in sorted(self.shards, key=lambda item: int(item.z0)):
            if int(shard.z0) != int(cursor) or int(shard.z1) <= int(shard.z0):
                raise ValueError(
                    f"non-contiguous distributed shard [{shard.z0},{shard.z1}) at {cursor}"
                )
            if int(shard.device_index) in devices:
                raise ValueError(f"duplicate distributed shard device {shard.device_index}")
            devices.add(int(shard.device_index))
            cursor = int(shard.z1)
        if int(cursor) != int(z_dim):
            raise ValueError(f"distributed shard coverage {cursor} != {z_dim}")


def artifact_plan_from_host(
    shape_tyx: Sequence[int], devices: Sequence[int], *, source: str = "host_uint8",
) -> DistributedBinaryArtifact:
    shape = tuple(int(v) for v in shape_tyx)
    if len(shape) != 3:
        raise ValueError(f"distributed binary artifact requires 3D shape, got {shape}")
    unique_devices = tuple(dict.fromkeys(int(v) for v in devices))
    if not unique_devices:
        raise ValueError("distributed binary artifact requires at least one device")
    partitions = contiguous_z_partitions(int(shape[0]), len(unique_devices))
    artifact = DistributedBinaryArtifact(
        shape_tyx=(int(shape[0]), int(shape[1]), int(shape[2])),
        shards=tuple(
            DistributedBinaryShard(
                device_index=int(unique_devices[int(part.rank)]),
                z0=int(part.z0), z1=int(part.z1), layout="planned", storage=None,
            )
            for part in partitions
        ),
        layout="planned",
        source=str(source),
    )
    artifact.validate()
    return artifact


@dataclass(frozen=True)
class KeepGraphShard:
    component_areas: np.ndarray
    internal_pair_codes: np.ndarray

    @property
    def component_count(self) -> int:
        return max(0, int(np.asarray(self.component_areas).size) - 1)


@dataclass(frozen=True)
class CrossShardPairCodes:
    left_rank: int
    right_rank: int
    pair_codes: np.ndarray


@dataclass(frozen=True)
class KeepGraphDecision:
    num_objects: int
    kept_objects: int
    removed_objects: int
    removed_voxels: int
    kept_voxels: int
    keep_by_shard_local_id: Tuple[np.ndarray, ...]
    global_root_map: np.ndarray = field(repr=False, compare=False)
    root_areas: np.ndarray = field(repr=False, compare=False)


def _offset_pair_codes(
    codes: np.ndarray,
    high_offset: int,
    low_offset: int,
    *,
    high_count: Optional[int] = None,
    low_count: Optional[int] = None,
    context: str = "component pair",
) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint64).reshape(-1)
    if int(values.size) <= 0:
        return np.zeros((0,), dtype=np.uint64)
    high_local = values >> np.uint64(32)
    low_local = values & np.uint64(0xFFFFFFFF)
    if high_count is not None and (
        bool(np.any(high_local == 0))
        or bool(np.any(high_local > np.uint64(int(high_count))))
    ):
        raise ValueError(
            f"{context} high ids must be in 1..{int(high_count)}"
        )
    if low_count is not None and (
        bool(np.any(low_local == 0))
        or bool(np.any(low_local > np.uint64(int(low_count))))
    ):
        raise ValueError(
            f"{context} low ids must be in 1..{int(low_count)}"
        )
    high = high_local + np.uint64(int(high_offset))
    low = low_local + np.uint64(int(low_offset))
    return np.ascontiguousarray((high << np.uint64(32)) | low, dtype=np.uint64)


def resolve_keep_graph(
    shards: Sequence[KeepGraphShard],
    cross_shard_pairs: Sequence[CrossShardPairCodes],
    keep_objects: int,
) -> KeepGraphDecision:
    """Resolve global component roots/areas from compact per-shard graph metadata."""

    from .topology import _UnionFind

    keep_n = max(0, int(keep_objects))
    counts = [int(shard.component_count) for shard in shards]
    for rank, shard in enumerate(shards):
        areas = np.asarray(shard.component_areas)
        if int(areas.ndim) != 1 or int(areas.size) < 1:
            raise ValueError(f"shard {rank} component areas must include background id 0")
        if int(areas[0]) != 0 or bool(np.any(areas < 0)):
            raise ValueError(f"shard {rank} component areas must be nonnegative with area[0]=0")
    offsets: List[int] = []
    cursor = 0
    for count in counts:
        offsets.append(int(cursor))
        cursor += int(count)
    total_components = int(cursor)
    if total_components <= 0:
        return KeepGraphDecision(
            num_objects=0, kept_objects=0, removed_objects=0, removed_voxels=0,
            kept_voxels=0,
            keep_by_shard_local_id=tuple(
                np.zeros((int(count) + 1,), dtype=bool) for count in counts
            ),
            global_root_map=np.zeros((1,), dtype=np.uint32),
            root_areas=np.zeros((1,), dtype=np.int64),
        )
    if total_components >= 2 ** 32:
        raise RuntimeError("v18.0.3 GPU-tail component id space exceeded uint32")

    uf = _UnionFind()
    uf.new_ids(int(total_components))
    for rank, shard in enumerate(shards):
        codes = _offset_pair_codes(
            np.asarray(shard.internal_pair_codes, dtype=np.uint64),
            int(offsets[rank]), int(offsets[rank]),
            high_count=int(counts[rank]), low_count=int(counts[rank]),
            context=f"shard {rank} internal pair",
        )
        if int(codes.size) > 0:
            uf.union_pair_codes(codes)
    for boundary in cross_shard_pairs:
        left = int(boundary.left_rank)
        right = int(boundary.right_rank)
        if not (0 <= left < len(shards) and 0 <= right < len(shards)):
            raise IndexError(f"invalid cross-shard ranks {left}->{right}")
        codes = _offset_pair_codes(
            np.asarray(boundary.pair_codes, dtype=np.uint64),
            int(offsets[left]), int(offsets[right]),
            high_count=int(counts[left]), low_count=int(counts[right]),
            context=f"cross-shard pair {left}->{right}",
        )
        if int(codes.size) > 0:
            uf.union_pair_codes(codes)

    root_map = np.ascontiguousarray(uf.root_map(), dtype=np.uint32)
    global_areas = np.zeros((int(total_components) + 1,), dtype=np.int64)
    for rank, shard in enumerate(shards):
        local = np.asarray(shard.component_areas, dtype=np.int64).reshape(-1)
        if int(local.size) != int(counts[rank]) + 1:
            raise ValueError(
                f"shard {rank} area length {local.size} != {counts[rank] + 1}"
            )
        if counts[rank] > 0:
            start = int(offsets[rank]) + 1
            global_areas[start:start + int(counts[rank])] = local[1:]
    root_areas = np.bincount(
        root_map.astype(np.int64, copy=False),
        weights=global_areas.astype(np.float64, copy=False),
        minlength=int(total_components) + 1,
    ).astype(np.int64)
    unique_roots = np.unique(root_map[1:])
    unique_roots = unique_roots[unique_roots > 0]
    order = np.argsort(root_areas[unique_roots])[::-1]
    keep_roots = unique_roots[order[: int(keep_n)]]
    keep_root_flag = np.zeros((int(total_components) + 1,), dtype=bool)
    keep_root_flag[keep_roots] = True
    gid_keep = keep_root_flag[root_map]
    gid_keep[0] = False
    per_shard: List[np.ndarray] = []
    for rank, count in enumerate(counts):
        local_keep = np.zeros((int(count) + 1,), dtype=bool)
        if count > 0:
            start = int(offsets[rank]) + 1
            local_keep[1:] = gid_keep[start:start + int(count)]
        per_shard.append(local_keep)
    kept_area = int(np.sum(root_areas[keep_roots], dtype=np.int64))
    total_area = int(np.sum(root_areas[unique_roots], dtype=np.int64))
    return KeepGraphDecision(
        num_objects=int(unique_roots.size),
        kept_objects=int(min(int(keep_n), int(unique_roots.size))),
        removed_objects=max(0, int(unique_roots.size) - int(keep_n)),
        removed_voxels=max(0, int(total_area) - int(kept_area)),
        kept_voxels=int(kept_area),
        keep_by_shard_local_id=tuple(per_shard),
        global_root_map=root_map,
        root_areas=root_areas,
    )


@dataclass
class GpuTailKeepResult:
    volume: np.ndarray
    stats: Dict[str, int | float]
    candidate_path: Optional[Path]
    used_gpu: bool


@dataclass
class _CudaKeepShard:
    rank: int
    device_index: int
    z0: int
    z1: int
    mask_dev: object = field(repr=False)
    labels_dev: object = field(repr=False)
    component_areas: np.ndarray = field(repr=False)
    internal_pair_codes: np.ndarray = field(repr=False)
    setup_seconds: float = 0.0
    label_seconds: float = 0.0
    pair_seconds: float = 0.0
    ccl_blocks: int = 0
    pair_boundaries: int = 0
    eligible_pair_boundaries: int = 0
    block_slices: int = 0


def _configured_tail_devices(torch_mod: object) -> Tuple[int, ...]:
    try:
        from .topology import gpu_slice_labeling_configured_devices

        configured = gpu_slice_labeling_configured_devices()
    except Exception:
        configured = None
    visible = max(0, int(torch_mod.cuda.device_count()))
    values = tuple(
        int(value) for value in (
            configured if configured is not None else tuple(range(visible))
        )
        if 0 <= int(value) < int(visible)
    )
    return tuple(dict.fromkeys(values))


def _gpu_tail_candidate_path(temp_dir: Path) -> Path:
    return Path(temp_dir) / "final_union_gpu_keep_candidate.u8.dat"


def try_apply_keep_largest_objects_multi_gpu(
    mask_mm: np.ndarray,
    keep_objects: int,
    temp_dir: Path,
    *,
    keep_temp: bool = False,
) -> Optional[GpuTailKeepResult]:
    """Try the v18.0.3 resident multi-GPU keep path.

    ``None`` means the caller must execute the established CPU implementation.
    The input is never mutated.  When REQUIRED is set, admission/execution errors
    propagate instead of returning ``None``.
    """

    enabled = bool(v1803_gpu_resident_tail_enabled())
    required = bool(v1803_gpu_resident_tail_required())
    if required and not enabled:
        raise RuntimeError(
            "YOLO_TTA_V1803_GPU_RESIDENT_TAIL_REQUIRED=1 requires "
            "YOLO_TTA_V1803_GPU_RESIDENT_TAIL=1"
        )
    if not enabled:
        return None
    keep_n = int(keep_objects)
    if keep_n <= 0:
        return None
    source = np.asarray(mask_mm)
    print(
        'v18.0.3 GPU-resident keep_objects requested: '
        f'shape={tuple(int(value) for value in source.shape)}, keep={keep_n}, '
        f'required={required}.'
    )
    if source.ndim != 3 or source.dtype != np.dtype(np.uint8):
        exc = ValueError(
            f"v18.0.3 GPU tail requires uint8 3D mask, got {source.shape}/{source.dtype}"
        )
        if v1803_gpu_resident_tail_required():
            raise exc
        print(f"Warning: {exc}; using CPU keep_objects.")
        return None

    candidate: Optional[np.memmap] = None
    candidate_path = _gpu_tail_candidate_path(Path(temp_dir))
    cuda_shards: List[_CudaKeepShard] = []
    # Explicitly cleared before allocator trimming.  Completed Future objects retain
    # their results (and failed Futures retain worker tracebacks), while a CuPy slice
    # view retains its complete base allocation.  Leaving either local alive through
    # ``free_all_blocks`` can strand the full label volume in CuPy's pool after return.
    futures: List[object] = []
    prev_remote = None
    copied_prev = None
    prev_host = None
    pair_dev = None
    started = time.perf_counter()
    try:
        import torch  # type: ignore
        import cupy as cp  # type: ignore
        import cupyx.scipy.ndimage as cpx_ndi  # type: ignore
        import_seconds = max(0.0, time.perf_counter() - started)

        discovery_started = time.perf_counter()
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("CUDA is unavailable")
        devices = _configured_tail_devices(torch)
        if not devices:
            raise RuntimeError("no configured CUDA devices are visible")
        plan = artifact_plan_from_host(source.shape, devices)
        discovery_seconds = max(0.0, time.perf_counter() - discovery_started)
        runtime_init_seconds = max(0.0, time.perf_counter() - started)

        metadata_started = time.perf_counter()
        from .topology import (
            binary_volume_slice_metadata,
            scan_binary_volume_slice_metadata,
        )

        support = binary_volume_slice_metadata(mask_mm)
        if support is None:
            # Reuse the established exact, affinity-aware parallel scanner.  A serial
            # np.any + per-nonempty-plane row/column rescan cost roughly 130 seconds on
            # the 1931x3064x3022 qualification volume despite only ~21 seconds of actual
            # four-GPU work.
            support = scan_binary_volume_slice_metadata(
                mask_mm,
                workers=int(_cpu_count()),
                source="v18.0.3 GPU-resident keep_objects exact fallback scan",
            )
        metadata_seconds = max(0.0, time.perf_counter() - metadata_started)
        known_any = np.ascontiguousarray(
            np.asarray(support.slice_any, dtype=bool),
        )
        known_bboxes = np.ascontiguousarray(
            np.asarray(support.slice_bboxes, dtype=np.int64),
        )

        from .topology import _gpu_adjacent_local_pair_codes_device

        def _process_shard(rank: int, shard_spec: DistributedBinaryShard) -> _CudaKeepShard:
            device_index = int(shard_spec.device_index)
            z0, z1 = int(shard_spec.z0), int(shard_spec.z1)
            local_shape = (int(z1 - z0), int(source.shape[1]), int(source.shape[2]))
            mask_host = None
            mask_dev = None
            labels_dev = None
            structure = None
            raw_labels = None
            areas = None
            pair_dev_local = None
            succeeded = False
            try:
                setup_started = time.perf_counter()
                with torch.cuda.device(int(device_index)), cp.cuda.Device(int(device_index)):
                    free_bytes, _total_bytes = torch.cuda.mem_get_info(int(device_index))
                    persistent_estimate = int(np.prod(local_shape, dtype=np.int64)) * 5
                    if int(free_bytes) < int(persistent_estimate) + int(v1803_gpu_tail_reserve_bytes()):
                        raise RuntimeError(
                            f"cuda:{device_index} needs approximately "
                            f"{persistent_estimate / GIB:.2f} GiB persistent + "
                            f"{v1803_gpu_tail_reserve_bytes() / GIB:.2f} GiB reserve; "
                            f"only {int(free_bytes) / GIB:.2f} GiB is free"
                        )
                    # CuPy host-to-device copies may be asynchronous.  Keep the exact
                    # contiguous host object alive until the labeling stream is settled.
                    mask_host = np.ascontiguousarray(source[z0:z1])
                    mask_dev = cp.asarray(mask_host, dtype=cp.uint8)
                    # uint32 is intentional: pair codes reserve all 32 bits for a local id.
                    # int32 would silently turn ids >= 2**31 negative and drop them from
                    # connectivity extraction before the advertised uint32 limit.
                    labels_dev = cp.empty(local_shape, dtype=cp.uint32)
                    setup_seconds = max(0.0, time.perf_counter() - setup_started)
                    label_started = time.perf_counter()
                    # Resolve exact 26-connectivity inside each memory-bounded 3-D block.
                    # The prior disconnected-Z structure created independent 2-D slice ids,
                    # then rebuilt all 3-D edges through one unique/synchronizing pair pass
                    # per adjacent slice.  Only block and shard boundaries need explicit
                    # equivalence pairs when the block CCL itself is 26-connected.
                    structure = cp.zeros((3, 3, 3), dtype=cp.bool_)
                    structure[:, :, :] = True
                    block = min(int(local_shape[0]), int(v1803_gpu_tail_block_slices()))
                    plane_pixels = int(local_shape[1]) * int(local_shape[2])
                    while block > 4:
                        free_now, _total_now = torch.cuda.mem_get_info(int(device_index))
                        if int(free_now) >= int(block) * int(plane_pixels) * 17 + int(v1803_gpu_tail_reserve_bytes()):
                            break
                        block = max(4, int(block) // 2)
                    free_now, _total_now = torch.cuda.mem_get_info(int(device_index))
                    scratch_need = int(block) * int(plane_pixels) * 17
                    if int(free_now) < int(scratch_need) + int(v1803_gpu_tail_reserve_bytes()):
                        raise RuntimeError(
                            f"cuda:{device_index} cannot admit even a {block}-slice 3-D CCL "
                            f"block ({scratch_need / GIB:.2f} GiB scratch + "
                            f"{v1803_gpu_tail_reserve_bytes() / GIB:.2f} GiB reserve); "
                            f"only {int(free_now) / GIB:.2f} GiB is free"
                        )
                    component_offset = 0
                    area_parts: List[np.ndarray] = []
                    block_boundaries: List[int] = []
                    ccl_blocks = 0
                    for local0 in range(0, int(local_shape[0]), int(block)):
                        local1 = min(int(local_shape[0]), int(local0) + int(block))
                        if int(local0) > 0:
                            block_boundaries.append(int(local0))
                        ccl_blocks += 1
                        raw_labels, raw_count = cpx_ndi.label(
                            mask_dev[int(local0):int(local1)] > 0,
                            structure=structure,
                        )
                        count = int(raw_count)
                        if count <= 0:
                            labels_dev[int(local0):int(local1)].fill(cp.uint32(0))
                            raw_labels = None
                            continue
                        if int(component_offset) + int(count) >= 2 ** 32:
                            raise RuntimeError("one GPU-tail shard exceeded uint32 local component ids")
                        areas = cp.bincount(raw_labels.reshape(-1), minlength=int(count) + 1)[1:]
                        area_parts.append(np.ascontiguousarray(cp.asnumpy(areas), dtype=np.int64))
                        labels_dev[int(local0):int(local1)] = cp.where(
                            raw_labels > 0,
                            raw_labels.astype(cp.uint64, copy=False) + int(component_offset),
                            cp.uint64(0),
                        ).astype(cp.uint32)
                        component_offset += int(count)
                        raw_labels = None
                        areas = None
                    cp.cuda.get_current_stream().synchronize()
                    mask_host = None
                    mask_dev = None
                    structure = None
                    label_seconds = max(0.0, time.perf_counter() - label_started)
                    component_areas = np.zeros((int(component_offset) + 1,), dtype=np.int64)
                    cursor = 1
                    for values in area_parts:
                        component_areas[int(cursor):int(cursor) + int(values.size)] = values
                        cursor += int(values.size)

                    pair_started = time.perf_counter()
                    pair_parts: List[np.ndarray] = []
                    eligible_pair_boundaries = 0
                    for local_z in block_boundaries:
                        global_z = int(z0 + local_z)
                        if not bool(known_any[int(global_z) - 1]) or not bool(known_any[int(global_z)]):
                            continue
                        eligible_pair_boundaries += 1
                        pair_dev_local = _gpu_adjacent_local_pair_codes_device(
                            cp,
                            labels_dev[int(local_z) - 1], labels_dev[int(local_z)],
                            known_bboxes[int(global_z) - 1], known_bboxes[int(global_z)],
                            int(local_shape[1]), int(local_shape[2]),
                        )
                        if int(pair_dev_local.size) > 0:
                            pair_parts.append(np.ascontiguousarray(cp.asnumpy(pair_dev_local), dtype=np.uint64))
                        pair_dev_local = None
                    internal_pairs = (
                        np.unique(np.concatenate(pair_parts)).astype(np.uint64, copy=False)
                        if pair_parts else np.zeros((0,), dtype=np.uint64)
                    )
                    pair_seconds = max(0.0, time.perf_counter() - pair_started)
                    result = _CudaKeepShard(
                        rank=int(rank), device_index=int(device_index), z0=int(z0), z1=int(z1),
                        mask_dev=None, labels_dev=labels_dev,
                        component_areas=component_areas,
                        internal_pair_codes=np.ascontiguousarray(internal_pairs, dtype=np.uint64),
                        setup_seconds=float(setup_seconds),
                        label_seconds=float(label_seconds), pair_seconds=float(pair_seconds),
                        ccl_blocks=int(ccl_blocks),
                        pair_boundaries=int(len(block_boundaries)),
                        eligible_pair_boundaries=int(eligible_pair_boundaries),
                        block_slices=int(block),
                    )
                    labels_dev = None  # ownership transferred to ``result``
                    succeeded = True
                    return result
            finally:
                # A failed Future retains this frame through its traceback.  Null every
                # device owner here so an admission/CCL error cannot pin an entire shard.
                mask_host = None
                mask_dev = None
                labels_dev = None
                structure = None
                raw_labels = None
                areas = None
                pair_dev_local = None
                # On success, the returned shard now owns the only live label allocation;
                # a full-process collection here only serializes the four GPU threads.
                # Failed Futures retain tracebacks, so collect only after an exceptional
                # path has had every local device reference explicitly nulled.
                if not succeeded:
                    gc.collect()

        shard_stage_started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=len(plan.shards), thread_name_prefix="v1803-gpu-tail-label",
        ) as executor:
            futures = [
                executor.submit(_process_shard, int(rank), shard)
                for rank, shard in enumerate(plan.shards)
            ]
            for future in futures:
                cuda_shards.append(future.result())
        futures.clear()
        cuda_shards.sort(key=lambda item: int(item.rank))
        shard_stage_seconds = max(0.0, time.perf_counter() - shard_stage_started)

        cross_pairs: List[CrossShardPairCodes] = []
        peer_bytes = 0
        host_bounces = 0
        peer_started = time.perf_counter()
        for left, right in zip(cuda_shards, cuda_shards[1:]):
            boundary_z = int(right.z0)
            if not bool(known_any[int(boundary_z) - 1]) or not bool(known_any[int(boundary_z)]):
                continue
            with cp.cuda.Device(int(right.device_index)):
                prev_remote = left.labels_dev[-1]
                try:
                    if not bool(cp.cuda.runtime.deviceCanAccessPeer(
                        int(right.device_index), int(left.device_index),
                    )):
                        raise RuntimeError("CUDA peer access unavailable")
                    # CuPy's cross-device transfer establishes peer access before the
                    # copy.  A raw cudaMemcpyPeerAsync without deviceEnablePeerAccess can
                    # legally stage through host memory even when deviceCanAccessPeer=1.
                    copied_prev = cp.asarray(prev_remote)
                    peer_bytes += int(prev_remote.nbytes)
                except Exception:
                    with cp.cuda.Device(int(left.device_index)):
                        prev_host = cp.asnumpy(prev_remote)
                    with cp.cuda.Device(int(right.device_index)):
                        copied_prev = cp.asarray(prev_host)
                    host_bounces += 1
                pair_dev = _gpu_adjacent_local_pair_codes_device(
                    cp, copied_prev, right.labels_dev[0],
                    known_bboxes[int(boundary_z) - 1], known_bboxes[int(boundary_z)],
                    int(source.shape[1]), int(source.shape[2]),
                )
                pair_host = np.ascontiguousarray(cp.asnumpy(pair_dev), dtype=np.uint64)
                cross_pairs.append(CrossShardPairCodes(
                    left_rank=int(left.rank), right_rank=int(right.rank), pair_codes=pair_host,
                ))
                pair_dev = None
                copied_prev = None
                prev_remote = None
                prev_host = None
        peer_seconds = max(0.0, time.perf_counter() - peer_started)

        graph_started = time.perf_counter()
        decision = resolve_keep_graph(
            [
                KeepGraphShard(
                    component_areas=shard.component_areas,
                    internal_pair_codes=shard.internal_pair_codes,
                )
                for shard in cuda_shards
            ],
            cross_pairs,
            int(keep_n),
        )
        graph_seconds = max(0.0, time.perf_counter() - graph_started)
        # The CPU reference currently resolves equal-area cutoff ties through its
        # provisional root-id ordering. CuPy/block/shard label ids are allowed to differ,
        # so the only reference-safe choice is to fall back when a tie straddles the
        # keep/drop boundary. Ties wholly inside either side do not affect the voxel set.
        unique_roots = np.unique(np.asarray(decision.global_root_map)[1:])
        unique_roots = unique_roots[unique_roots > 0]
        if 0 < int(keep_n) < int(unique_roots.size):
            ranked_areas = np.sort(
                np.asarray(decision.root_areas, dtype=np.int64)[unique_roots]
            )[::-1]
            if int(ranked_areas[int(keep_n) - 1]) == int(ranked_areas[int(keep_n)]):
                raise RuntimeError(
                    'equal-area component tie crosses the keep_objects cutoff; '
                    'the CPU reference must resolve its established root-id tie break'
                )
        if int(decision.num_objects) <= int(keep_n):
            return GpuTailKeepResult(
                volume=mask_mm,
                stats={
                    "enabled": 1, "gpu_resident_tail": 1,
                    "num_objects": int(decision.num_objects),
                    "kept_objects": int(decision.num_objects), "removed_objects": 0,
                    "removed_voxels": 0, "kept_voxels": int(decision.kept_voxels),
                    "gpu_count": int(len(cuda_shards)),
                    "runtime_init_seconds": float(runtime_init_seconds),
                    "import_seconds": float(import_seconds),
                    "discovery_seconds": float(discovery_seconds),
                    "setup_seconds": float(max((s.setup_seconds for s in cuda_shards), default=0.0)),
                    "shard_stage_seconds": float(shard_stage_seconds),
                    "ccl_blocks": int(sum(s.ccl_blocks for s in cuda_shards)),
                    "pair_boundaries": int(sum(s.pair_boundaries for s in cuda_shards)),
                    "eligible_pair_boundaries": int(sum(s.eligible_pair_boundaries for s in cuda_shards)),
                    "block_slices_min": int(min((s.block_slices for s in cuda_shards), default=0)),
                    "block_slices_max": int(max((s.block_slices for s in cuda_shards), default=0)),
                    "label_seconds": float(max((s.label_seconds for s in cuda_shards), default=0.0)),
                    "metadata_seconds": float(metadata_seconds),
                    "pair_extraction_seconds": float(max((s.pair_seconds for s in cuda_shards), default=0.0)),
                    "boundary_merge_seconds": float(peer_seconds),
                    "root_expansion_seconds": float(graph_seconds),
                    "apply_seconds": 0.0,
                    "total_seconds": float(time.perf_counter() - started),
                    "peer_bytes": int(peer_bytes), "peer_host_bounces": int(host_bounces),
                },
                candidate_path=None,
                used_gpu=True,
            )

        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate = np.memmap(
            candidate_path, dtype=np.uint8, mode="w+", shape=tuple(int(v) for v in source.shape),
        )
        slice_any = np.zeros((int(source.shape[0]),), dtype=bool)
        slice_bboxes = np.zeros((int(source.shape[0]), 4), dtype=np.int64)
        apply_started = time.perf_counter()

        def _apply_shard(shard: _CudaKeepShard) -> None:
            with torch.cuda.device(int(shard.device_index)), cp.cuda.Device(int(shard.device_index)):
                # Retain the host owner until the same stream's first D2H synchronization;
                # CuPy may otherwise outlive an anonymous async H2D source temporary.
                keep_host = np.ascontiguousarray(
                    decision.keep_by_shard_local_id[int(shard.rank)], dtype=np.uint8,
                )
                keep_dev = cp.asarray(keep_host)
                chunk = max(1, min(32, int(shard.z1 - shard.z0)))
                for local0 in range(0, int(shard.z1 - shard.z0), int(chunk)):
                    local1 = min(int(shard.z1 - shard.z0), int(local0) + int(chunk))
                    out_dev = keep_dev[shard.labels_dev[int(local0):int(local1)]].astype(
                        cp.uint8, copy=False,
                    )
                    out_host = np.ascontiguousarray(cp.asnumpy(out_dev), dtype=np.uint8)
                    global0 = int(shard.z0 + local0)
                    global1 = int(shard.z0 + local1)
                    np.copyto(candidate[int(global0):int(global1)], out_host)
                    rows = np.any(out_host != 0, axis=2)
                    cols = np.any(out_host != 0, axis=1)
                    any_local = np.any(rows, axis=1)
                    slice_any[int(global0):int(global1)] = any_local
                    for offset in np.flatnonzero(any_local):
                        z_global = int(global0 + int(offset))
                        row_ids = np.flatnonzero(rows[int(offset)])
                        col_ids = np.flatnonzero(cols[int(offset)])
                        slice_bboxes[z_global] = np.asarray(
                            (int(row_ids[0]), int(row_ids[-1]) + 1,
                             int(col_ids[0]), int(col_ids[-1]) + 1),
                            dtype=np.int64,
                        )
                    del out_dev, out_host

        with ThreadPoolExecutor(
            max_workers=len(cuda_shards), thread_name_prefix="v1803-gpu-tail-apply",
        ) as executor:
            list(executor.map(_apply_shard, cuda_shards))
        _flush_array(candidate)
        apply_seconds = max(0.0, time.perf_counter() - apply_started)
        try:
            from .topology import register_binary_volume_slice_metadata

            register_binary_volume_slice_metadata(
                candidate, slice_any, slice_bboxes,
                source="v18.0.3 GPU-resident multi-GPU keep_objects", exact=True,
            )
        except Exception:
            pass
        result = GpuTailKeepResult(
            volume=candidate,
            stats={
                "enabled": 1, "gpu_resident_tail": 1,
                "num_objects": int(decision.num_objects),
                "kept_objects": int(decision.kept_objects),
                "removed_objects": int(decision.removed_objects),
                "removed_voxels": int(decision.removed_voxels),
                "kept_voxels": int(decision.kept_voxels),
                "gpu_count": int(len(cuda_shards)),
                "runtime_init_seconds": float(runtime_init_seconds),
                "import_seconds": float(import_seconds),
                "discovery_seconds": float(discovery_seconds),
                "setup_seconds": float(max((s.setup_seconds for s in cuda_shards), default=0.0)),
                "shard_stage_seconds": float(shard_stage_seconds),
                "ccl_blocks": int(sum(s.ccl_blocks for s in cuda_shards)),
                "pair_boundaries": int(sum(s.pair_boundaries for s in cuda_shards)),
                "eligible_pair_boundaries": int(sum(s.eligible_pair_boundaries for s in cuda_shards)),
                "block_slices_min": int(min((s.block_slices for s in cuda_shards), default=0)),
                "block_slices_max": int(max((s.block_slices for s in cuda_shards), default=0)),
                "label_seconds": float(max((s.label_seconds for s in cuda_shards), default=0.0)),
                "metadata_seconds": float(metadata_seconds),
                "pair_extraction_seconds": float(max((s.pair_seconds for s in cuda_shards), default=0.0)),
                "boundary_merge_seconds": float(peer_seconds),
                "root_expansion_seconds": float(graph_seconds),
                "apply_seconds": float(apply_seconds),
                "total_seconds": float(time.perf_counter() - started),
                "peer_bytes": int(peer_bytes), "peer_host_bounces": int(host_bounces),
            },
            candidate_path=Path(candidate_path),
            used_gpu=True,
        )
        candidate = None
        return result
    except Exception as exc:
        if candidate is not None:
            _close_array(candidate)
            candidate = None
        if not bool(keep_temp):
            try:
                candidate_path.unlink(missing_ok=True)
            except Exception:
                pass
        if v1803_gpu_resident_tail_required():
            raise RuntimeError("v18.0.3 GPU-resident tail failed") from exc
        print(
            "Warning: v18.0.3 GPU-resident tail unavailable/failed "
            f"({type(exc).__name__}: {exc}); using CPU keep_objects."
        )
        return None
    finally:
        futures.clear()
        prev_remote = None
        copied_prev = None
        prev_host = None
        pair_dev = None
        for shard in cuda_shards:
            try:
                shard.mask_dev = None
                shard.labels_dev = None
            except Exception:
                pass
        cuda_shards.clear()
        gc.collect()
        try:
            import cupy as cp  # type: ignore

            for device_index in tuple(locals().get("devices", ())):
                try:
                    with cp.cuda.Device(int(device_index)):
                        cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
        except Exception:
            pass


__all__ = [
    "CrossShardPairCodes",
    "DistributedBinaryArtifact",
    "DistributedBinaryShard",
    "GpuTailKeepResult",
    "KeepGraphDecision",
    "KeepGraphShard",
    "ZPartition",
    "artifact_plan_from_host",
    "contiguous_z_partitions",
    "pack_binary_rows",
    "resolve_keep_graph",
    "row_tail_mask",
    "row_word_count",
    "try_apply_keep_largest_objects_multi_gpu",
    "unpack_binary_rows",
    "v1803_gpu_resident_tail_enabled",
    "v1803_gpu_resident_tail_required",
]
