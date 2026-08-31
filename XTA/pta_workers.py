"""Persistent PTA render-worker process state, tasks, and pool ownership.

Normal CPU process workers use spawn plus named shared-memory payloads.  The
external GPU augmentation path retains its documented fork-only exception.
This module is the canonical multiprocessing owner and never imports
:mod:`XTA.pta`.
"""

from __future__ import annotations

import atexit
import importlib
import multiprocessing
import os
import pickle
import queue
import threading
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

try:
    from tqdm import tqdm  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tqdm is required: pip install tqdm") from exc

from . import geometry as shared_geometry
from .pta_augmentation import (
    LoadedAugmentation,
    LoadedGpuAugmentation,
    OfflineAugmentation,
    inspect_augmentation_definition,
    load_augmentation_definition,
)
from .pta_dataset import OutputCandidate
from .pta_publication import (
    _write_nvjpeg_batch_atomically,
    candidate_output_paths,
    ensure_output_parent_once,
    mask_to_yolo_lines,
    parse_output_image_format,
    write_image,
    write_selected_candidate_version,
    write_yolo_lines,
)
from .pta_rendering import (
    RenderFrameSource,
    RenderPlan,
    encoded_channel_source_positions,
    extract_padded_tile,
    render_channel_formatted_images,
    render_plan_frame_mask_source,
    render_plan_frame_source,
    render_shared_tile_images,
    resize_centered,
    _require_pta_canonical_plan,
)
from .pta_scheduler import (
    gpu_memory_candidate_limit as _gpu_memory_candidate_limit,
    is_cuda_out_of_memory as _is_cuda_out_of_memory,
    iter_compatible_work_batches as _gpu_multi_source_work_batches,
    should_flush_ready_gpu_work as _should_flush_ready_gpu_work,
    split_work_batch as _split_gpu_work_batch,
)
from .unification.contracts import RasterPlan
from .workspace import radial_source_mode as _radial_source_mode


class _StemmedSource(Protocol):
    stem: str


class PreparedRenderVolume(Protocol):
    """Structural parent-side input consumed by :meth:`install_phase`."""

    src: _StemmedSource
    plans: Sequence[RenderPlan]
    volume_for_render: np.ndarray
    mask_for_render: np.ndarray
    volume_render_block: Optional["SharedBlock"]
    mask_render_block: Optional["SharedBlock"]


_WARNING_EXAMPLE_LIMIT = 12


def _retain_deterministic_warning_examples(
    existing: List[str], candidates: Iterable[str],
) -> None:
    """Keep the lexicographically first examples, independent of arrival order."""

    existing.extend(str(candidate) for candidate in candidates)
    existing.sort()
    del existing[_WARNING_EXAMPLE_LIMIT:]

@dataclass
class WarningLog:
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    examples: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, key: str, msg: str = "") -> None:
        with self.lock:
            self.counts[str(key)] += 1
            if msg:
                _retain_deterministic_warning_examples(
                    self.examples[str(key)], (str(msg),),
                )

    def merge_from(self, other: "WarningLog", *, include_keys: Optional[set[str]] = None, exclude_keys: Optional[set[str]] = None) -> None:
        include = include_keys
        exclude = exclude_keys or set()
        with self.lock:
            for key, count in other.counts.items():
                if include is not None and key not in include:
                    continue
                if key in exclude:
                    continue
                self.counts[str(key)] += int(count)
                _retain_deterministic_warning_examples(
                    self.examples[str(key)], other.examples.get(key, ()),
                )

    def summary_lines(self) -> List[str]:
        with self.lock:
            counts = {str(key): int(count) for key, count in self.counts.items()}
            examples = {
                str(key): tuple(sorted(str(example) for example in values))
                for key, values in self.examples.items()
            }
        lines: List[str] = ["Warnings:"]
        if not counts:
            lines.append("  none")
            return lines
        for key in sorted(counts):
            lines.append(f"  {key}: {counts[key]}")
            for ex in examples.get(key, ())[:_WARNING_EXAMPLE_LIMIT]:
                lines.append(f"    example: {ex}")
        return lines

def bind_current_thread_to_cpus(cpus: Sequence[int]) -> bool:
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, {int(x) for x in cpus})
        return True
    except Exception:
        return False

try:
    from multiprocessing import shared_memory as _shared_memory  # type: ignore
except Exception:  # pragma: no cover
    _shared_memory = None  # type: ignore[assignment]


class SharedBlock:
    """One named POSIX shared-memory block owned by the parent process.

    Volumes, masks, and per-phase task payloads travel to the persistent
    render pool through these blocks instead of per-volume COW forks.  The
    parent creates and unlinks; workers attach untracked (see
    _attach_shm_untracked) so the resource tracker does not reap or warn
    about blocks it does not own.
    """

    def __init__(self, nbytes: int):
        if _shared_memory is None:
            raise RuntimeError("multiprocessing.shared_memory is unavailable on this platform")
        self.nbytes = max(1, int(nbytes))
        self.shm = _shared_memory.SharedMemory(create=True, size=self.nbytes)
        self.released = False

    @property
    def name(self) -> str:
        return str(self.shm.name)

    def ndarray(self, shape: Tuple[int, ...], dtype: object = np.uint8) -> np.ndarray:
        return np.ndarray(tuple(int(x) for x in shape), dtype=np.dtype(dtype), buffer=self.shm.buf)

    def release(self) -> None:
        """Close the parent mapping (best effort) and unlink the name.

        A still-exported buffer (live ndarray view) makes close() raise
        BufferError; unlink alone is then sufficient - the memory is freed
        once every mapping in every process is gone.
        """
        if self.released:
            return
        self.released = True
        try:
            self.shm.close()
        except BufferError:
            pass
        except Exception:
            pass
        try:
            self.shm.unlink()
        except Exception:
            pass


def _attach_shm_untracked(name: str):
    """Attach an existing shared-memory block without resource_tracker ownership."""
    try:
        return _shared_memory.SharedMemory(name=str(name), track=False)  # type: ignore[call-arg]
    except TypeError:
        # Python < 3.13 has no track parameter; unregister manually so the
        # per-process resource tracker does not unlink parent-owned blocks.
        shm = _shared_memory.SharedMemory(name=str(name))
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(getattr(shm, "_name", "/" + str(name).lstrip("/")), "shared_memory")
        except Exception:
            pass
        return shm


ArrayAllocator = Callable[[Tuple[int, ...]], Tuple[np.ndarray, Optional[SharedBlock]]]


def make_volume_allocator(use_shared: bool) -> ArrayAllocator:
    """Return an allocator for zero-initialized uint8 volume-sized arrays.

    The process backend allocates through SharedBlock so loaders write volumes
    directly into pool-visible memory; the thread backend and direct callers
    get ordinary arrays.  Shared pages arrive zero-filled, so both branches
    satisfy callers that rely on zeros (e.g. label rasterization).
    """

    def _alloc(shape: Tuple[int, ...]) -> Tuple[np.ndarray, Optional[SharedBlock]]:
        dims = tuple(int(x) for x in shape)
        if not use_shared:
            return np.zeros(dims, dtype=np.uint8), None
        block = SharedBlock(int(np.prod(dims, dtype=np.int64)))
        return block.ndarray(dims, np.uint8), block

    return _alloc


def ensure_shared_uint8(arr: np.ndarray, block: Optional[SharedBlock], allocator: Optional[ArrayAllocator]) -> Tuple[np.ndarray, Optional[SharedBlock]]:
    """Return a contiguous uint8 array backed by a SharedBlock when required.

    Pass-through when the array is already block-backed or no shared allocator
    is in play; otherwise copy once into a fresh block (cubic-resize and NRRD
    paths produce ordinary arrays).
    """
    if block is not None or allocator is None:
        return np.ascontiguousarray(arr, dtype=np.uint8), block
    out, new_block = allocator(tuple(int(x) for x in arr.shape))
    if new_block is None:
        return np.ascontiguousarray(arr, dtype=np.uint8), None
    np.copyto(out, np.asarray(arr, dtype=np.uint8))
    return out, new_block

_WORKER_STATIC: Dict[str, object] = {}

# CUDA state is deliberately child-local and lazy.  The GPU-policy exception
# forks the persistent pool before source decode or any CUDA context exists.
_WORKER_GPU_DEVICE_ID: Optional[int] = None
_WORKER_GPU_RUNTIME: Optional[Dict[str, object]] = None
_WORKER_GPU_CODEC_WARNING_EMITTED = False
_WORKER_GPU_BATCH_CAP_WARNING_EMITTED = False


_THREAD_AFFINITY_LOCK = threading.Lock()
_THREAD_AFFINITY_NEXT = 0


def _render_worker_initializer(
    worker_cpu_order: Sequence[int] = (),
    gpu_device_ids: Sequence[int] = (),
    gpu_cpu_sets: Sequence[Sequence[int]] = (),
    worker_static_payload: Optional[Mapping[str, object]] = None,
) -> None:
    global _WORKER_GPU_DEVICE_ID, _WORKER_GPU_RUNTIME
    global _WORKER_GPU_CODEC_WARNING_EMITTED, _WORKER_GPU_BATCH_CAP_WARNING_EMITTED
    if worker_static_payload is not None:
        _initialize_spawned_worker_static_context(worker_static_payload)
    _WORKER_GPU_DEVICE_ID = None
    _WORKER_GPU_RUNTIME = None
    _WORKER_GPU_CODEC_WARNING_EMITTED = False
    _WORKER_GPU_BATCH_CAP_WARNING_EMITTED = False
    # A module can be imported and used by more than one pool over the life of
    # a spawned interpreter in tests/embedders.  Never let an earlier pool's
    # shared-memory attachments leak into a later worker contract.
    _WORKER_PAYLOAD_CACHE.clear()
    _WORKER_VOLUME_IDENTITY_BY_POINTER.clear()
    for state in _WORKER_GEN_CACHE.values():
        for shm in state.get("shms", ()):  # type: ignore[union-attr]
            try:
                shm.close()
            except Exception:
                pass
    _WORKER_GEN_CACHE.clear()
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    identity = getattr(multiprocessing.current_process(), "_identity", ())
    worker_index = max(0, int(identity[0]) - 1) if identity else 0
    if gpu_device_ids:
        rank = int(worker_index) % len(gpu_device_ids)
        _WORKER_GPU_DEVICE_ID = int(gpu_device_ids[rank])
        _WORKER_GPU_RUNTIME = None
        local_cpus = tuple(int(x) for x in gpu_cpu_sets[rank]) if rank < len(gpu_cpu_sets) else tuple()
        if local_cpus:
            bind_current_thread_to_cpus(local_cpus)
        elif worker_cpu_order:
            bind_current_thread_to_cpus((int(worker_cpu_order[rank % len(worker_cpu_order)]),))
    elif worker_cpu_order:
        bind_current_thread_to_cpus((int(worker_cpu_order[worker_index % len(worker_cpu_order)]),))


def _render_thread_initializer(worker_cpu_order: Sequence[int] = ()) -> None:
    global _THREAD_AFFINITY_NEXT
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    if not worker_cpu_order:
        return
    with _THREAD_AFFINITY_LOCK:
        worker_index = int(_THREAD_AFFINITY_NEXT)
        _THREAD_AFFINITY_NEXT += 1
    bind_current_thread_to_cpus((int(worker_cpu_order[worker_index % len(worker_cpu_order)]),))


def set_worker_static_context(
    *,
    out_dir: Path,
    split_active: bool,
    image_format: str,
    png_compression: int,
    jpeg_quality: int,
    jpeg_encode_backend: str,
    gpu_batch_size: int,
    gpu_render_threads: int,
    gpu_device_ids: Sequence[int],
    augmentation: Optional[OfflineAugmentation],
    save_images: bool = True,
    save_labels: bool = True,
    tiff_encode_backend: str = "auto",
) -> None:
    """Install run-constant worker state.  Must run before the pool is created."""
    _WORKER_STATIC.clear()
    _WORKER_STATIC.update({
        "out_dir": out_dir,
        "split_active": bool(split_active),
        "image_format": str(image_format),
        "png_compression": int(png_compression),
        "jpeg_quality": int(jpeg_quality),
        "jpeg_encode_backend": str(jpeg_encode_backend),
        "tiff_encode_backend": str(tiff_encode_backend),
        "gpu_batch_size": int(gpu_batch_size),
        "gpu_render_threads": max(1, int(gpu_render_threads)),
        "gpu_device_ids": tuple(int(x) for x in gpu_device_ids),
        "augmentation": augmentation,
        "save_images": bool(save_images),
        "save_labels": bool(save_labels),
    })


def _spawn_worker_static_payload() -> Dict[str, object]:
    """Return a picklable worker contract for the normal spawn process path.

    Loaded Albumentations pipelines contain thread locals, closures, and
    imported module objects, so they must never be pickled.  A spawned child
    instead receives the already-validated definition identity and reloads it
    from the authoritative file.  The content digest is checked before import
    so a policy changed between parent preflight and pool startup is rejected.
    """

    if not _WORKER_STATIC:
        raise RuntimeError("PTA render worker static context has not been initialized")
    augmentation = _WORKER_STATIC.get("augmentation")
    if isinstance(augmentation, LoadedGpuAugmentation):
        raise RuntimeError(
            "GPU offline augmentation retains its explicit fork-only worker path; "
            "it cannot be serialized into the normal spawn worker contract"
        )
    augmentation_definition: Optional[Dict[str, str]] = None
    if isinstance(augmentation, LoadedAugmentation):
        augmentation_definition = {
            "kind": "cpu",
            "path": str(augmentation.path),
            "content_sha256": str(augmentation.content_sha256),
            "export_name": str(augmentation.export_name),
        }
    elif augmentation is not None:
        raise TypeError(
            "Normal PTA process workers support no augmentation or a LoadedAugmentation; "
            f"got {type(augmentation).__name__}"
        )
    return {
        "schema": "pta.v18.spawn-worker-static/1",
        "out_dir": str(Path(_WORKER_STATIC["out_dir"])),
        "split_active": bool(_WORKER_STATIC["split_active"]),
        "image_format": str(_WORKER_STATIC["image_format"]),
        "png_compression": int(_WORKER_STATIC["png_compression"]),
        "jpeg_quality": int(_WORKER_STATIC["jpeg_quality"]),
        "jpeg_encode_backend": str(_WORKER_STATIC["jpeg_encode_backend"]),
        "tiff_encode_backend": str(_WORKER_STATIC.get("tiff_encode_backend", "auto")),
        "gpu_batch_size": int(_WORKER_STATIC["gpu_batch_size"]),
        "gpu_render_threads": int(_WORKER_STATIC.get("gpu_render_threads", 1)),
        "gpu_device_ids": (),
        "augmentation_definition": augmentation_definition,
        "save_images": bool(_WORKER_STATIC.get("save_images", True)),
        "save_labels": bool(_WORKER_STATIC.get("save_labels", True)),
    }


def _initialize_spawned_worker_static_context(payload: Mapping[str, object]) -> None:
    """Install and validate one explicit static contract in a spawned child."""

    if str(payload.get("schema", "")) != "pta.v18.spawn-worker-static/1":
        raise RuntimeError(
            "Unsupported PTA spawn worker static schema: "
            f"{payload.get('schema')!r}"
        )
    augmentation: Optional[OfflineAugmentation] = None
    definition_payload = payload.get("augmentation_definition")
    if definition_payload is not None:
        if not isinstance(definition_payload, Mapping):
            raise TypeError("PTA spawn augmentation definition must be a mapping")
        if str(definition_payload.get("kind", "")) != "cpu":
            raise RuntimeError("Only CPU augmentation definitions are valid in PTA spawn workers")
        path = Path(str(definition_payload["path"])).expanduser().resolve()
        expected_sha256 = str(definition_payload["content_sha256"])
        expected_export = str(definition_payload["export_name"])
        inspected = inspect_augmentation_definition(str(path))
        if inspected.content_sha256 != expected_sha256 or inspected.export_name != expected_export:
            raise RuntimeError(
                "PTA CPU augmentation policy changed after parent preflight; refusing to "
                f"import it in a spawned worker: {path}"
            )
        augmentation = load_augmentation_definition(str(path))
        if (
            augmentation.content_sha256 != expected_sha256
            or augmentation.export_name != expected_export
        ):
            raise RuntimeError(
                "PTA CPU augmentation identity changed while loading it in a spawned worker: "
                f"{path}"
            )

    _WORKER_STATIC.clear()
    _WORKER_STATIC.update({
        "out_dir": Path(str(payload["out_dir"])),
        "split_active": bool(payload["split_active"]),
        "image_format": str(payload["image_format"]),
        "png_compression": int(payload["png_compression"]),
        "jpeg_quality": int(payload["jpeg_quality"]),
        "jpeg_encode_backend": str(payload["jpeg_encode_backend"]),
        "tiff_encode_backend": str(payload.get("tiff_encode_backend", "auto")),
        "gpu_batch_size": int(payload["gpu_batch_size"]),
        "gpu_render_threads": max(1, int(payload.get("gpu_render_threads", 1))),
        "gpu_device_ids": (),
        "augmentation": augmentation,
        "save_images": bool(payload.get("save_images", True)),
        "save_labels": bool(payload.get("save_labels", True)),
    })


@dataclass(frozen=True)
class FrameRenderTask:
    """Every retained item/version derived from one rendered source frame."""
    plan_idx: int
    frame_idx: int
    items: Tuple[Tuple[str, Tuple[OutputCandidate, ...]], ...]  # (item_key, candidates)


@dataclass(frozen=True)
class GpuFrameBatchTask:
    """Several frame renders coalesced into one device-feeding work unit."""

    frames: Tuple[FrameRenderTask, ...]


def _frame_render_candidate_count(task: FrameRenderTask) -> int:
    return sum(len(candidates) for _item_key, candidates in task.items)


def batch_gpu_frame_tasks(
    tasks: Sequence[FrameRenderTask],
    *,
    candidate_limit: int,
) -> List[GpuFrameBatchTask]:
    """Pack frame tasks without exceeding the requested device batch target.

    A frame remains atomic so its shared render is never repeated. Oversized
    frames form a one-frame batch and are microbatched by the execution path.
    """

    limit = max(1, int(candidate_limit))
    batches: List[GpuFrameBatchTask] = []
    pending: List[FrameRenderTask] = []
    pending_count = 0
    for task in tasks:
        task_count = _frame_render_candidate_count(task)
        if pending and pending_count + task_count > limit:
            batches.append(GpuFrameBatchTask(tuple(pending)))
            pending = []
            pending_count = 0
        pending.append(task)
        pending_count += task_count
        if pending_count >= limit:
            batches.append(GpuFrameBatchTask(tuple(pending)))
            pending = []
            pending_count = 0
    if pending:
        batches.append(GpuFrameBatchTask(tuple(pending)))
    return batches


def build_phase_render_tasks(
    plans: Sequence[RenderPlan],
    candidates: Sequence[OutputCandidate],
    *,
    aug_chunk: int,
    gpu_batch_size: int = 0,
) -> List[object]:
    """Group all work by source frame so a reslice/canvas is produced once."""
    _ = aug_chunk  # retained as a deprecated CLI/API compatibility parameter
    plan_index_by_tag = {plan.tag: int(i) for i, plan in enumerate(plans)}
    grouped: Dict[Tuple[int, int], Dict[str, List[OutputCandidate]]] = defaultdict(lambda: defaultdict(list))
    for cand in candidates:
        plan_idx = plan_index_by_tag[cand.parent_view_tag]
        grouped[(plan_idx, int(cand.frame_idx))][str(cand.item_key)].append(cand)
    tasks: List[FrameRenderTask] = []
    for frame_key in sorted(grouped.keys()):
        items = tuple(
            (str(item_key), tuple(sorted(cands, key=lambda c: (int(c.augmentation_index), int(c.order)))))
            for item_key, cands in sorted(grouped[frame_key].items())
        )
        tasks.append(FrameRenderTask(int(frame_key[0]), int(frame_key[1]), items))
    # Longest-processing-time-first submission without changing candidate IDs.
    ordered = sorted(
        tasks,
        key=lambda task: (
            -sum(1 for _, cands in task.items for cand in cands if int(cand.augmentation_index) > 0),
            -sum(len(cands) for _, cands in task.items),
            int(task.plan_idx),
            int(task.frame_idx),
        ),
    )
    if int(gpu_batch_size) > 0:
        return list(
            batch_gpu_frame_tasks(
                ordered,
                # Keep two device batches of source frames available so CPU
                # geometry can run ahead while the GPU consumes the first.
                candidate_limit=max(1, int(gpu_batch_size)) * 2,
            )
        )
    return list(ordered)


def projection_phase_summary(
    plans: Sequence[RenderPlan],
    candidates: Sequence[OutputCandidate],
) -> Tuple[Tuple[str, int, int], ...]:
    """Return ordered ``(family, count, cumulative_end)`` projection phases."""

    counts_by_tag: Dict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts_by_tag[str(candidate.parent_view_tag)] += 1
    family_counts: Dict[str, int] = {}
    family_order: List[str] = []
    for plan in plans:
        count = int(counts_by_tag.get(str(plan.tag), 0))
        if count <= 0:
            continue
        shared_view = plan.view.shared_view
        if shared_view is not None and shared_geometry.is_radial_view(shared_view):
            family = (
                "tilted-radial"
                if shared_geometry.is_tilted_radial_view(shared_view)
                else "upright-radial"
            )
        elif shared_view is not None and shared_geometry.is_tilted_view(shared_view):
            family = "tilted-cartesian"
        else:
            family = "cartesian"
        if family not in family_counts:
            family_counts[family] = 0
            family_order.append(family)
        family_counts[family] += count
    cumulative = 0
    summary: List[Tuple[str, int, int]] = []
    for family in family_order:
        count = int(family_counts[family])
        cumulative += count
        summary.append((family, count, cumulative))
    return tuple(summary)


def _derive_item_arrays(source: RenderFrameSource, plan: RenderPlan, item_key: str) -> Tuple[np.ndarray, np.ndarray]:
    if item_key == "full":
        return source.img_full, source.mask_full
    shared_arrays = source.tile_arrays.get(str(item_key))
    if shared_arrays is not None:
        return shared_arrays
    if source.img_canvas is None or source.mask_canvas is None:
        raise RuntimeError(f"Tile output requested without a rendered canvas for {plan.tag}")
    for tile in plan.tile_layout:
        if tile.tile_tag == item_key:
            tile_img = extract_padded_tile(source.img_canvas, tile.x, tile.y, tile.cfg.tile_size)
            tile_mask = extract_padded_tile(source.mask_canvas, tile.x, tile.y, tile.cfg.tile_size)
            tile_img_out = resize_centered(tile_img, tile.out_w, tile.out_h, cv2.INTER_LINEAR)
            tile_mask_out = resize_centered(tile_mask, tile.out_w, tile.out_h, cv2.INTER_NEAREST)
            return tile_img_out, tile_mask_out
    raise RuntimeError(f"Unknown tile item {item_key!r} for plan {plan.tag}")


def _derive_gpu_item_source(
    source: RenderFrameSource,
    plan: RenderPlan,
    item_key: str,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Return an unresized ROI plus the required output (H,W).

    Full-frame render geometry remains on CPU, but tile crop/resize moves to
    the GPU policy.  In-bounds tile extraction is a NumPy view (M2); padding is
    materialized only for a genuine edge tile.
    """
    if item_key == "full":
        return source.img_full, source.mask_full, (
            int(source.img_full.shape[0]),
            int(source.img_full.shape[1]),
        )
    shared_arrays = source.tile_arrays.get(str(item_key))
    if shared_arrays is not None:
        image, mask = shared_arrays
        return image, mask, (int(image.shape[0]), int(image.shape[1]))
    if source.img_canvas is None or source.mask_canvas is None:
        raise RuntimeError(f"Tile output requested without a rendered canvas for {plan.tag}")
    for tile in plan.tile_layout:
        if tile.tile_tag == item_key:
            tile_img = extract_padded_tile(source.img_canvas, tile.x, tile.y, tile.cfg.tile_size)
            tile_mask = extract_padded_tile(source.mask_canvas, tile.x, tile.y, tile.cfg.tile_size)
            return tile_img, tile_mask, (int(tile.out_h), int(tile.out_w))
    raise RuntimeError(f"Unknown tile item {item_key!r} for plan {plan.tag}")


def _gpu_runtime_for_worker() -> Dict[str, object]:
    """Initialize one policy and optional nvJPEG encoder per persistent rank."""
    global _WORKER_GPU_RUNTIME
    if _WORKER_GPU_RUNTIME is not None:
        return _WORKER_GPU_RUNTIME
    augmentation = _WORKER_STATIC.get("augmentation")
    if not isinstance(augmentation, LoadedGpuAugmentation):
        raise RuntimeError("Internal error: GPU runtime requested without a GPU augmentation definition")
    if _WORKER_GPU_DEVICE_ID is None:
        raise RuntimeError("Internal error: GPU render worker has no assigned CUDA device")
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "GPU offline augmentation requires a CUDA-enabled PyTorch installation"
        ) from exc
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("PyTorch reports CUDA unavailable inside the GPU render rank")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    device_id = int(_WORKER_GPU_DEVICE_ID)
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    policy = augmentation.build_for_device(
        device=device,
        batch_size=int(_WORKER_STATIC["gpu_batch_size"]),
    )

    radial_renderer = None
    radial_renderer_error = ""
    try:
        cuda_backend = importlib.import_module(".cuda_backend", package=__package__)
        radial_renderer = cuda_backend._GpuWorkerRenderEngine(device)
    except Exception as exc:
        radial_renderer_error = f"{type(exc).__name__}: {exc}"

    encoder = None
    nvimgcodec = None
    codec_error = ""
    nvtiff_encoder = None
    nvtiff_error = ""
    image_format = parse_output_image_format(str(_WORKER_STATIC["image_format"]))
    codec_request = str(_WORKER_STATIC["jpeg_encode_backend"])
    if image_format == "jpg" and codec_request in {"auto", "nvjpeg"}:
        try:
            nvimgcodec = importlib.import_module("nvidia.nvimgcodec")
            # nvJPEG's CUDA encoder still performs part of its work on the
            # CPU.  Restricting it to GPU_ONLY makes the framework reject
            # otherwise valid CUDA-resident samples as unprocessable.
            nvjpeg_backend = nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)
            encoder_lanes = max(
                1,
                min(4, int(_WORKER_STATIC.get("gpu_render_threads", 1))),
            )
            encoder = nvimgcodec.Encoder(
                device_id=device_id,
                max_num_cpu_threads=encoder_lanes,
                backends=[nvjpeg_backend],
                # Global nvImageCodec options require a leading colon. Several
                # bounded lanes allow a JPEG batch to use the otherwise-idle
                # CPU/GPU encode capacity without creating another CUDA owner.
                options=f":num_cuda_streams={encoder_lanes}",
            )
        except Exception as exc:
            if codec_request == "nvjpeg":
                raise RuntimeError(
                    "--jpeg_encode_backend nvjpeg requires a working "
                    "nvidia-nvimgcodec-cu12 or nvidia-nvimgcodec-cu13 installation "
                    "matching the active CUDA major version"
                ) from exc
            encoder = None
            nvimgcodec = None
            codec_error = f"{type(exc).__name__}: {exc}"
    tiff_codec_request = str(_WORKER_STATIC.get("tiff_encode_backend", "auto"))
    if image_format == "tif" and tiff_codec_request in {"auto", "nvtiff"}:
        try:
            nvtiff_module = importlib.import_module(".nvtiff_backend", package=__package__)
            cuda_version = str(getattr(torch.version, "cuda", "") or "")
            cuda_major = int(cuda_version.split(".", 1)[0]) if cuda_version else None
            nvtiff_encoder = nvtiff_module.NvTiffBackend(
                device_id,
                cuda_major=cuda_major,
            )
            atexit.register(nvtiff_encoder.close)
        except Exception as exc:
            if tiff_codec_request == "nvtiff":
                raise RuntimeError(
                    "--tiff_encode_backend nvtiff requires nvTIFF 0.8 or newer; install "
                    "nvidia-nvtiff-cu13 for CUDA 13 or nvidia-nvtiff-cu12 for CUDA 12"
                ) from exc
            nvtiff_error = f"{type(exc).__name__}: {exc}"
    _WORKER_GPU_RUNTIME = {
        "torch": torch,
        "policy": policy,
        "device_id": device_id,
        "encoder": encoder,
        "nvimgcodec": nvimgcodec,
        "codec_error": codec_error,
        "nvtiff_encoder": nvtiff_encoder,
        "nvtiff_error": nvtiff_error,
        "radial_renderer": radial_renderer,
        "radial_renderer_error": radial_renderer_error,
        "radial_render_lock": threading.Lock(),
        "radial_renderer_announced": False,
        "radial_renderer_manifest_announced": False,
        "cartesian_renderer_announced": False,
        "cartesian_renderer_manifest_announced": False,
        "tilted_renderer_announced": False,
        "tilted_renderer_manifest_announced": False,
        "radial_renderer_fallback_announced": False,
        "cartesian_renderer_fallback_announced": False,
        "tilted_renderer_fallback_announced": False,
        "cuda_projection_disabled_families": set(),
    }
    return _WORKER_GPU_RUNTIME


def _nvjpeg_encode_params(nvimgcodec: object, *, quality: int, channel_kind: str) -> object:
    kwargs: Dict[str, object] = {
        "quality_type": nvimgcodec.QualityType.QUALITY,  # type: ignore[attr-defined]
        "quality_value": int(quality),
    }
    if str(channel_kind) == "gray":
        kwargs["color_spec"] = nvimgcodec.ColorSpec.GRAY  # type: ignore[attr-defined]
        kwargs["chroma_subsampling"] = nvimgcodec.ChromaSubsampling.CSS_GRAY  # type: ignore[attr-defined]
    elif str(channel_kind) == "rgb":
        kwargs["color_spec"] = nvimgcodec.ColorSpec.SRGB  # type: ignore[attr-defined]
    return nvimgcodec.EncodeParams(**kwargs)  # type: ignore[attr-defined]


def _nvjpeg_samples_nhwc(images_nchw: object, *, channel_kind: str) -> object:
    """Convert a uint8 NCHW batch to nvImageCodec's interleaved HWC layout.

    nvImageCodec represents even a one-channel image as a three-dimensional
    HxWx1 buffer.  Dropping the channel axis for grayscale produces HxW
    tensors that can be wrapped but cannot be encoded by nvJPEG.
    """
    if int(images_nchw.ndim) != 4:  # type: ignore[attr-defined]
        raise ValueError(f"GPU JPEG input must be NCHW, got shape={tuple(images_nchw.shape)}")  # type: ignore[attr-defined]
    channels = int(images_nchw.shape[1])  # type: ignore[attr-defined]
    if str(channel_kind) == "gray":
        if channels != 1:
            raise ValueError(f"Grayscale GPU JPEG input must have one channel, got {channels}")
    elif str(channel_kind) == "rgb":
        if channels != 3:
            raise ValueError(f"RGB GPU JPEG input must have three channels, got {channels}")
    else:
        raise ValueError("GPU JPEG encoding supports gray or RGB channel formats")
    return images_nchw.permute(0, 2, 3, 1).contiguous()  # type: ignore[attr-defined]


def _nvjpeg_wrap_samples(
    nvimgcodec: object,
    samples: Sequence[object],
    *,
    channel_kind: str,
    cuda_stream: int,
) -> Sequence[object]:
    """Wrap external CUDA buffers with explicit format metadata when supported."""
    as_images = getattr(nvimgcodec, "as_images", None)
    if not callable(as_images):
        return samples

    kwargs: Dict[str, object] = {"cuda_stream": int(cuda_stream)}
    sample_format_type = getattr(nvimgcodec, "SampleFormat", None)
    color_spec_type = getattr(nvimgcodec, "ColorSpec", None)
    if str(channel_kind) == "gray":
        sample_format = getattr(sample_format_type, "I_Y", None)
        color_spec = getattr(color_spec_type, "GRAY", None)
    elif str(channel_kind) == "rgb":
        sample_format = getattr(sample_format_type, "I_RGB", None)
        color_spec = getattr(color_spec_type, "SRGB", None)
    else:
        raise ValueError("GPU JPEG encoding supports gray or RGB channel formats")
    if sample_format is not None:
        kwargs["sample_format"] = sample_format
    if color_spec is not None:
        kwargs["color_spec"] = color_spec

    try:
        return as_images(samples, **kwargs)
    except TypeError:
        # Older nvImageCodec builds exposed as_images before adding the
        # sample_format/color_spec keyword overrides.  The HWC shapes supplied
        # above still allow those builds to infer I_Y or I_RGB correctly.
        return as_images(samples, cuda_stream=int(cuda_stream))


def _write_gpu_image_batch(
    *,
    runtime: Mapping[str, object],
    images_nchw: object,
    indices: Sequence[int],
    paths: Sequence[Path],
    channel_kind: str,
    image_format: str,
    png_compression: int,
    jpeg_quality: int,
) -> Optional[str]:
    """Encode CUDA images with nvJPEG/nvTIFF, or return an auto-fallback note."""
    if not indices:
        return None
    if len(indices) != len(paths):
        raise RuntimeError(f"GPU image/path batch mismatch: images={len(indices)}, paths={len(paths)}")
    torch = runtime["torch"]
    index_tensor = torch.as_tensor(list(int(x) for x in indices), device=images_nchw.device)  # type: ignore[attr-defined]
    selected = images_nchw.index_select(0, index_tensor)  # type: ignore[attr-defined]
    for path in paths:
        ensure_output_parent_once(path)

    encoder = runtime.get("encoder")
    nvimgcodec = runtime.get("nvimgcodec")
    codec_request = str(_WORKER_STATIC["jpeg_encode_backend"])
    tiff_codec_request = str(_WORKER_STATIC.get("tiff_encode_backend", "auto"))
    fallback_note: Optional[str] = None
    parsed_format = parse_output_image_format(image_format)
    if parsed_format == "tif" and str(channel_kind) == "custom":
        nvtiff_encoder = runtime.get("nvtiff_encoder")
        if nvtiff_encoder is not None:
            try:
                stream = torch.cuda.current_stream(device=int(runtime["device_id"]))  # type: ignore[attr-defined]
                for sample_index, path in enumerate(paths):
                    nvtiff_encoder.write_multipage_lzw(  # type: ignore[union-attr]
                        path,
                        selected[int(sample_index)].contiguous(),
                        cuda_stream=int(stream.cuda_stream),
                    )
                return None
            except Exception as exc:
                if tiff_codec_request == "nvtiff":
                    raise RuntimeError("nvTIFF multipage encoding failed") from exc
                fallback_note = f"{type(exc).__name__}: {exc}"
                if isinstance(runtime, dict):
                    try:
                        nvtiff_encoder.close()
                    except Exception:
                        pass
                    runtime["nvtiff_encoder"] = None
                    runtime["nvtiff_error"] = fallback_note
        elif tiff_codec_request == "nvtiff":
            raise RuntimeError(
                "nvTIFF was explicitly requested but no encoder is active: "
                f"{runtime.get('nvtiff_error') or 'no diagnostic'}"
            )
        elif runtime.get("nvtiff_error"):
            fallback_note = str(runtime["nvtiff_error"])
    if parsed_format == "jpg" and encoder is not None and nvimgcodec is not None:
        try:
            stream = torch.cuda.current_stream(device=int(runtime["device_id"]))  # type: ignore[attr-defined]
            samples_tensor = _nvjpeg_samples_nhwc(selected, channel_kind=str(channel_kind))
            samples = [samples_tensor[index] for index in range(int(samples_tensor.shape[0]))]
            wrapped = _nvjpeg_wrap_samples(
                nvimgcodec,
                samples,
                channel_kind=str(channel_kind),
                cuda_stream=int(stream.cuda_stream),
            )
            _write_nvjpeg_batch_atomically(
                encoder=encoder,
                images=wrapped,
                final_paths=paths,
                params=_nvjpeg_encode_params(
                    nvimgcodec,
                    quality=int(jpeg_quality),
                    channel_kind=str(channel_kind),
                ),
                cuda_stream=stream,
                synchronize_device=(
                    lambda: torch.cuda.synchronize(int(runtime["device_id"]))
                ),
            )
            return None
        except Exception as exc:
            if codec_request == "nvjpeg":
                raise RuntimeError("nvJPEG batch encoding failed") from exc
            fallback_note = f"{type(exc).__name__}: {exc}"
            if isinstance(runtime, dict):
                runtime["encoder"] = None
                runtime["nvimgcodec"] = None
                runtime["codec_error"] = fallback_note

    host = selected.detach().to("cpu").numpy()
    for batch_index, path in enumerate(paths):
        sample = host[batch_index]
        if str(channel_kind) == "gray":
            sample_out = np.ascontiguousarray(sample[0])
        elif str(channel_kind) == "rgb":
            sample_out = np.ascontiguousarray(np.transpose(sample, (1, 2, 0)))
        elif str(channel_kind) == "custom":
            sample_out = np.ascontiguousarray(np.transpose(sample, (1, 2, 0)))
        else:
            raise ValueError("GPU augmentation CPU-encode fallback received an unknown channel kind")
        write_image(
            path,
            sample_out,
            int(png_compression),
            channel_kind=str(channel_kind),
            jpeg_quality=int(jpeg_quality),
        )
    return fallback_note


@dataclass(frozen=True)
class _GpuItemWork:
    """One rendered source item and the candidate versions derived from it."""

    candidates: Tuple[OutputCandidate, ...]
    image: object
    mask: np.ndarray
    output_size: Tuple[int, int]
    channel_kind: str
    context: str
    channel_count: int = 1
    ready_event: Optional[object] = None


def _gpu_identity_fast_path_eligible(
    work: Sequence[_GpuItemWork],
    seeds: Sequence[Sequence[Optional[int]]],
) -> bool:
    """Return whether originals can bypass an external identity grid warp."""

    if not work or len(work) != len(seeds):
        return False
    channel_kind = str(work[0].channel_kind)
    if channel_kind not in {"gray", "rgb", "custom"}:
        return False
    for item, item_seeds in zip(work, seeds):
        if len(item.candidates) != len(item_seeds) or any(seed is not None for seed in item_seeds):
            return False
        image_shape = tuple(int(x) for x in item.image.shape)
        if str(item.channel_kind) != channel_kind:
            return False
        if channel_kind == "gray" and len(image_shape) != 2:
            return False
        if channel_kind == "rgb" and (len(image_shape) != 3 or image_shape[2] != 3):
            return False
        if channel_kind == "custom" and (
            len(image_shape) != 3
            or image_shape[2] != int(item.channel_count)
            or int(item.channel_count) < 1
        ):
            return False
        if tuple(image_shape[:2]) != tuple(int(x) for x in item.output_size):
            return False
        if tuple(int(x) for x in item.mask.shape) != tuple(image_shape[:2]):
            return False
    return True


def _apply_gpu_identity_batch_many(
    runtime: Mapping[str, object],
    work: Sequence[_GpuItemWork],
) -> Tuple[object, object]:
    """Upload already-rendered originals without building or sampling a grid."""

    torch = runtime["torch"]
    channel_kind = str(work[0].channel_kind)
    device = f"cuda:{int(runtime['device_id'])}"
    cuda_images = [bool(getattr(item.image, "is_cuda", False)) for item in work]
    if any(cuda_images) and not all(cuda_images):
        raise ValueError("GPU identity publication cannot mix host and CUDA source images")
    if all(cuda_images):
        expected_device = torch.device(device)
        image_planes = []
        for item in work:
            image = item.image
            if getattr(image, "dtype", None) != torch.uint8:
                raise TypeError("GPU-rendered identity sources must be torch.uint8")
            if torch.device(getattr(image, "device", None)) != expected_device:
                raise ValueError(
                    "GPU-rendered identity source is on the wrong device: "
                    f"expected {expected_device}, got {getattr(image, 'device', None)}"
                )
            if channel_kind == "gray":
                image_planes.append(image.unsqueeze(0))
            elif channel_kind in {"rgb", "custom"}:
                image_planes.append(image.permute(2, 0, 1))
            else:
                raise ValueError(
                    f"GPU identity publication does not support channel kind {channel_kind!r}"
                )
        image_sources = torch.stack(image_planes, dim=0).contiguous()
    else:
        image_arrays = [
            np.ascontiguousarray(np.asarray(item.image), dtype=np.uint8)
            for item in work
        ]
        if channel_kind == "gray":
            image_nchw = np.stack(image_arrays, axis=0)[:, None, :, :]
        elif channel_kind in {"rgb", "custom"}:
            image_nchw = np.transpose(np.stack(image_arrays, axis=0), (0, 3, 1, 2))
        else:
            raise ValueError(
                f"GPU identity publication does not support channel kind {channel_kind!r}"
            )
        image_host = torch.from_numpy(np.ascontiguousarray(image_nchw)).pin_memory()
        image_sources = image_host.to(device, non_blocking=True)
    source_mask_required = any(
        bool(candidate.label_enabled)
        for item in work
        for candidate in item.candidates
    )
    total_candidates = sum(len(item.candidates) for item in work)
    if source_mask_required:
        mask_arrays = [
            np.ascontiguousarray((np.asarray(item.mask) > 0).astype(np.uint8))
            for item in work
        ]
        mask_nhw = np.stack(mask_arrays, axis=0)
        mask_host = torch.from_numpy(np.ascontiguousarray(mask_nhw)).pin_memory()
        mask_sources = mask_host.to(device, non_blocking=True)
    else:
        # The publisher does not inspect mask pixels for an unlabeled batch.
        # Retain the NHW contract as a zero-strided device view without
        # allocating or transferring one full blank mask per candidate.
        out_h, out_w = (int(value) for value in work[0].output_size)
        mask_sources = torch.zeros(
            (1, 1, 1),
            device=device,
            dtype=torch.uint8,
        ).expand(total_candidates, out_h, out_w)

    if all(len(item.candidates) == 1 for item in work):
        return image_sources.contiguous(), mask_sources

    source_indices = torch.as_tensor(
        [
            source_index
            for source_index, item in enumerate(work)
            for _candidate in item.candidates
        ],
        device=device,
        dtype=torch.int64,
    )
    return (
        image_sources.index_select(0, source_indices).contiguous(),
        (
            mask_sources.index_select(0, source_indices).contiguous()
            if source_mask_required
            else mask_sources
        ),
    )


def _zero_mask_view(height: int, width: int) -> np.ndarray:
    """Return a shape-correct, zero-strided unlabeled mask without a raster allocation."""

    return np.broadcast_to(
        np.zeros((1, 1), dtype=np.uint8),
        (int(height), int(width)),
    )


def _gpu_policy_host_image(image: object) -> np.ndarray:
    """Return the legacy writable NumPy policy input for a CUDA-rendered source."""

    if bool(getattr(image, "is_cuda", False)):
        image = image.detach().to("cpu").numpy()  # type: ignore[union-attr]
    return np.ascontiguousarray(np.asarray(image), dtype=np.uint8)


def _gpu_policy_source_images(
    policy: object,
    batch: Sequence[_GpuItemWork],
) -> Tuple[Tuple[object, ...], bool]:
    """Choose the zero-copy CUDA policy contract when every source supports it."""

    cuda_sources = bool(batch) and all(
        bool(getattr(item.image, "is_cuda", False)) for item in batch
    )
    if cuda_sources and bool(getattr(policy, "supports_cuda_sources", False)):
        return tuple(item.image for item in batch), True
    return tuple(_gpu_policy_host_image(item.image) for item in batch), False


def _wait_for_gpu_work_ready(
    runtime: Mapping[str, object],
    batch: Sequence[_GpuItemWork],
) -> None:
    """Order policy/publication after asynchronous CUDA projection events."""

    events: List[object] = []
    seen: set[int] = set()
    for item in batch:
        event = item.ready_event
        if event is None or id(event) in seen:
            continue
        seen.add(id(event))
        events.append(event)
    if not events:
        return
    torch = runtime["torch"]
    consumer_stream = torch.cuda.current_stream(device=int(runtime["device_id"]))
    for event in events:
        consumer_stream.wait_event(event)
    for item in batch:
        image = item.image
        record_stream = getattr(image, "record_stream", None)
        if bool(getattr(image, "is_cuda", False)) and callable(record_stream):
            record_stream(consumer_stream)


def _gpu_projected_item_image(
    runtime: Mapping[str, object],
    volume: np.ndarray,
    plan: RenderPlan,
    frame_idx: int,
    item_key: str,
) -> Optional[Tuple[object, object]]:
    """Render one canonical Cartesian/Radial/Tilted intensity item on CUDA."""

    shared_view = plan.view.shared_view
    radial_view = bool(
        shared_view is not None and shared_geometry.is_radial_view(shared_view)
    )
    tilted_view = bool(
        shared_view is not None
        and shared_geometry.is_tilted_view(shared_view)
        and not radial_view
    )
    cartesian_view = bool(
        shared_view is not None
        and not radial_view
        and not tilted_view
        and shared_geometry.physical_view_name(shared_view)
        in {"transverse", "sagittal", "coronal"}
    )
    if (
        shared_view is None
        or not (cartesian_view or radial_view or tilted_view)
        or str(plan.channel_variant.kind) not in {"gray", "rgb", "custom"}
    ):
        return None
    gate_name = (
        "YOLO_TTA_PTA_GPU_RADIAL_RENDER"
        if radial_view
        else (
            "YOLO_TTA_PTA_GPU_TILTED_RENDER"
            if tilted_view
            else "YOLO_TTA_PTA_GPU_CARTESIAN_RENDER"
        )
    )
    if os.environ.get(gate_name, "1").strip().lower() in {
        "0", "false", "no", "off", "disabled",
    }:
        return None
    if radial_view and _radial_source_mode() != "texture_linear":
        return None
    family_key = (
        "radial" if radial_view else ("tilted" if tilted_view else "cartesian")
    )
    if family_key in runtime.get("cuda_projection_disabled_families", set()):
        return None
    renderer = runtime.get("radial_renderer")
    lock = runtime.get("radial_render_lock")
    if renderer is None or lock is None:
        fallback_key = f"{family_key}_renderer_fallback_announced"
        if isinstance(runtime, dict) and not bool(runtime.get(fallback_key)):
            runtime[fallback_key] = True
            print(
                "Warning: PTA resident CUDA view renderer is unavailable "
                f"({runtime.get('radial_renderer_error') or 'no diagnostic'}); using CPU projection."
            )
        return None
    if str(item_key) == "full":
        _require_pta_canonical_plan(
            plan,
            "intensity",
            backend="cuda",
        )
        output_height = int(plan.aff.out_h)
        output_width = int(plan.aff.out_w)
        output_to_source = np.asarray(plan.aff.M_out_to_src, dtype=np.float32)
    else:
        tile = next(
            (candidate for candidate in plan.tile_layout if str(candidate.tile_tag) == str(item_key)),
            None,
        )
        if tile is None or tile.shared_job is None:
            return None
        _require_pta_canonical_plan(
            plan,
            "intensity",
            tile=tile,
            backend="cuda",
        )
        output_height = int(tile.out_h)
        output_width = int(tile.out_w)
        output_to_source = np.asarray(tile.shared_job.M_out_to_src, dtype=np.float32)

    try:
        with lock:  # type: ignore[attr-defined]
            volume_identity = _WORKER_VOLUME_IDENTITY_BY_POINTER.get(
                int(volume.__array_interface__["data"][0]),
                f"array:{int(volume.__array_interface__['data'][0])}",
            )
            mode = renderer.ensure_volume_array(  # type: ignore[union-attr]
                volume,
                identity=volume_identity,
                require_radial_texture=bool(
                    runtime.get("radial_texture_required", True)
                ),
            )
            if str(mode) != "resident":
                return None
            torch = runtime["torch"]
            channel_kind = str(plan.channel_variant.kind)
            if channel_kind == "custom":
                if plan.source_encoded_indices:
                    source_positions, missing = encoded_channel_source_positions(
                        plan.source_encoded_indices,
                        int(frame_idx),
                        plan.channel_variant.offsets,
                    )
                    if source_positions is None:
                        return None
                    addresses = tuple((int(position), False) for position in source_positions)
                else:
                    addresses = tuple(
                        shared_geometry.channel_view_slice_source(
                            shared_view,
                            int(frame_idx) + int(offset),
                        )
                        for offset in plan.channel_variant.offsets
                    )
            else:
                addresses = ((int(frame_idx), False),)

            native_by_address: Dict[Tuple[int, bool], object] = {}
            with torch.cuda.stream(renderer._stream):  # type: ignore[union-attr]
                for source_index, mirror_u in dict.fromkeys(addresses):
                    if radial_view:
                        if shared_geometry.is_tilted_radial_view(shared_view):
                            native = renderer._render_tilted_radial_native_resident(  # type: ignore[union-attr]
                                shared_view,
                                int(source_index),
                            )
                        else:
                            # Upright Radial uses TTA's allocation-free
                            # hardware-linear texture projector, stopping at
                            # the native uint8 boundary before the PTA affine.
                            native = renderer._render_radial_native_resident(  # type: ignore[union-attr]
                                shared_view,
                                int(source_index),
                            )
                        native_u8 = native.round().clamp_(0.0, 255.0).to(torch.uint8)
                        if bool(mirror_u):
                            native_u8 = torch.flip(native_u8, dims=(1,))
                        rendered_u8 = renderer.warp_native_uint8_frame(  # type: ignore[union-attr]
                            native_u8.contiguous(), output_to_source,
                            int(output_height), int(output_width),
                        )
                    elif tilted_view:
                        if bool(mirror_u):
                            raise RuntimeError("Tilted Cartesian addressing cannot mirror radial-u")
                        rendered = renderer.render_tilted_grid_resident(  # type: ignore[union-attr]
                            shared_view,
                            output_to_source,
                            int(source_index),
                            int(output_height),
                            int(output_width),
                        )
                        rendered_u8 = rendered.round().clamp_(0.0, 255.0).to(
                            torch.uint8
                        ).contiguous()
                    else:
                        if bool(mirror_u):
                            raise RuntimeError("Cartesian addressing cannot mirror radial-u")
                        rendered_u8 = renderer.render_cartesian_grid_resident(  # type: ignore[union-attr]
                            shared_view,
                            output_to_source,
                            int(source_index),
                            int(output_height),
                            int(output_width),
                        )
                    native_by_address[(int(source_index), bool(mirror_u))] = rendered_u8
                first_native = native_by_address[addresses[0]]
                if channel_kind == "gray":
                    image = first_native
                elif channel_kind == "rgb":
                    image = first_native.unsqueeze(2).expand(-1, -1, 3).contiguous()
                else:
                    image = torch.stack(
                        [native_by_address[address] for address in addresses],
                        dim=2,
                    ).contiguous()
                ready_event = torch.cuda.Event()
                ready_event.record(renderer._stream)  # type: ignore[union-attr]
            announced_key = (
                "radial_renderer_announced"
                if radial_view
                else (
                    "tilted_renderer_announced"
                    if tilted_view
                    else "cartesian_renderer_announced"
                )
            )
            if isinstance(runtime, dict) and not bool(runtime.get(announced_key)):
                runtime[announced_key] = True
                family_label = (
                    "Radial"
                    if radial_view
                    else ("Tilted Cartesian" if tilted_view else "Cartesian")
                )
                print(
                    f"PTA {family_label} intensity projection: TTA resident CUDA renderer active; "
                    "projected uint8 sources remain on device through CUDA policy/publication."
                )
            return image, ready_event
    except Exception as exc:
        if isinstance(runtime, dict):
            disabled = runtime.setdefault("cuda_projection_disabled_families", set())
            disabled.add(family_key)
            fallback_key = f"{family_key}_renderer_fallback_announced"
            if not bool(runtime.get(fallback_key)):
                runtime[fallback_key] = True
                print(
                    "Warning: PTA CUDA view projection failed "
                    f"({type(exc).__name__}: {exc}); using CPU projection for this worker."
                )
        return None


def _render_gpu_frame_items(
    volume: np.ndarray,
    mask: np.ndarray,
    plans: Sequence[RenderPlan],
    frame_task: FrameRenderTask,
) -> Tuple[_GpuItemWork, ...]:
    """Render one frame's source items without touching CUDA state."""

    plan = plans[int(frame_task.plan_idx)]
    source = render_plan_frame_source(
        volume=volume,
        mask=mask,
        plan=plan,
        idx=int(frame_task.frame_idx),
        need_canvas=any(item_key != "full" for item_key, _ in frame_task.items),
    )
    rendered: List[_GpuItemWork] = []
    for item_key, candidates in frame_task.items:
        item_image, item_mask, output_size = _derive_gpu_item_source(
            source,
            plan,
            str(item_key),
        )
        rendered.append(
            _GpuItemWork(
                candidates=tuple(candidates),
                image=item_image,
                mask=item_mask,
                output_size=tuple(int(x) for x in output_size),
                channel_kind=str(candidates[0].channel_kind),
                context=f"{plan.tag}/{item_key}/frame={int(frame_task.frame_idx) + 1:04d}",
                channel_count=int(plan.channel_variant.channel_count),
            )
        )
    return tuple(rendered)


def _render_gpu_item_group(
    volume: np.ndarray,
    mask: np.ndarray,
    plan: RenderPlan,
    frame_idx: int,
    items: Sequence[Tuple[str, Tuple[OutputCandidate, ...]]],
    runtime: Optional[Mapping[str, object]] = None,
) -> Tuple[_GpuItemWork, ...]:
    """Render an independently schedulable full/tile item group on the CPU."""

    tile_by_tag = {str(tile.tile_tag): tile for tile in plan.tile_layout}
    plan_channel_count = max(
        1,
        int(getattr(getattr(plan, "channel_variant", None), "channel_count", 1)),
    )
    source_mask_required = any(
        bool(candidate.label_enabled)
        for _item_key, candidates in items
        for candidate in candidates
    )
    if runtime is not None and len(items) == 1:
        item_key, candidates = items[0]
        gpu_projection = _gpu_projected_item_image(
            runtime,
            volume,
            plan,
            int(frame_idx),
            str(item_key),
        )
        if gpu_projection is not None:
            gpu_image, gpu_ready_event = gpu_projection
            if str(item_key) == "full":
                if source_mask_required:
                    item_mask, _mask_canvas = render_plan_frame_mask_source(
                        mask=mask,
                        plan=plan,
                        idx=int(frame_idx),
                        need_canvas=False,
                    )
                else:
                    item_mask = _zero_mask_view(
                        int(gpu_image.shape[0]),
                        int(gpu_image.shape[1]),
                    )
                output_size = (
                    int(gpu_image.shape[0]),
                    int(gpu_image.shape[1]),
                )
            else:
                tile = next(
                    (
                        candidate
                        for candidate in plan.tile_layout
                        if str(candidate.tile_tag) == str(item_key)
                    ),
                    None,
                )
                if tile is None or tile.shared_job is None:
                    raise RuntimeError(
                        f"CUDA-rendered PTA tile {plan.tag}/{item_key} has no canonical tile job"
                    )
                item_mask = (
                    shared_geometry.render_categorical_dense_tile_for_job(
                        mask,
                        plan.view.shared_view,
                        tile.shared_job,
                        int(frame_idx),
                    )
                    if source_mask_required
                    else _zero_mask_view(int(tile.out_h), int(tile.out_w))
                )
                output_size = (int(tile.out_h), int(tile.out_w))
            return (
                _GpuItemWork(
                    candidates=tuple(candidates),
                    image=gpu_image,
                    mask=(
                        np.ascontiguousarray((np.asarray(item_mask) > 0).astype(np.uint8))
                        if source_mask_required
                        else item_mask
                    ),
                    output_size=output_size,
                    channel_kind=str(candidates[0].channel_kind),
                    context=f"{plan.tag}/{item_key}/frame={int(frame_idx) + 1:04d}",
                    channel_count=plan_channel_count,
                    ready_event=gpu_ready_event,
                ),
            )
    if len(items) == 1 and str(items[0][0]) != "full":
        item_key, candidates = items[0]
        tile = tile_by_tag[str(item_key)]
        if tile.shared_job is not None:
            assert plan.view.shared_view is not None
            tile_image = render_shared_tile_images(
                volume=volume,
                plan=plan,
                tile=tile,
                idx=int(frame_idx),
            )
            tile_mask = (
                shared_geometry.render_categorical_dense_tile_for_job(
                    mask,
                    plan.view.shared_view,
                    tile.shared_job,
                    int(frame_idx),
                )
                if source_mask_required
                else _zero_mask_view(int(tile_image.shape[0]), int(tile_image.shape[1]))
            )
            return (
                _GpuItemWork(
                    candidates=tuple(candidates),
                    image=np.ascontiguousarray(tile_image, dtype=np.uint8),
                    mask=(
                        np.ascontiguousarray((np.asarray(tile_mask) > 0).astype(np.uint8))
                        if source_mask_required
                        else tile_mask
                    ),
                    output_size=(int(tile.out_h), int(tile.out_w)),
                    channel_kind=str(candidates[0].channel_kind),
                    context=f"{plan.tag}/{item_key}/frame={int(frame_idx) + 1:04d}",
                    channel_count=plan_channel_count,
                ),
            )

    need_canvas = any(str(item_key) != "full" for item_key, _ in items)
    image_full, image_canvas = render_channel_formatted_images(
        volume=volume,
        plan=plan,
        idx=int(frame_idx),
        need_canvas=need_canvas,
    )
    if source_mask_required:
        mask_full, mask_canvas = render_plan_frame_mask_source(
            mask=mask,
            plan=plan,
            idx=int(frame_idx),
            need_canvas=need_canvas,
        )
    else:
        mask_full = _zero_mask_view(int(image_full.shape[0]), int(image_full.shape[1]))
        mask_canvas = (
            _zero_mask_view(int(image_canvas.shape[0]), int(image_canvas.shape[1]))
            if image_canvas is not None
            else None
        )
    rendered: List[_GpuItemWork] = []
    for item_key, candidates in items:
        if str(item_key) == "full":
            item_image = image_full
            item_mask = mask_full
            output_size = (int(image_full.shape[0]), int(image_full.shape[1]))
        else:
            tile = tile_by_tag[str(item_key)]
            if image_canvas is None or mask_canvas is None:
                raise RuntimeError(f"Tile output requested without a rendered canvas for {plan.tag}")
            item_image = extract_padded_tile(
                image_canvas,
                tile.x,
                tile.y,
                tile.cfg.tile_size,
            )
            item_mask = extract_padded_tile(
                mask_canvas,
                tile.x,
                tile.y,
                tile.cfg.tile_size,
            )
            output_size = (int(tile.out_h), int(tile.out_w))
        rendered.append(
            _GpuItemWork(
                candidates=tuple(candidates),
                image=item_image,
                mask=item_mask,
                output_size=output_size,
                channel_kind=str(candidates[0].channel_kind),
                context=f"{plan.tag}/{item_key}/frame={int(frame_idx) + 1:04d}",
                channel_count=plan_channel_count,
            )
        )
    return tuple(rendered)


def _publish_gpu_policy_batch(
    *,
    runtime: Mapping[str, object],
    batch_images: object,
    batch_masks: object,
    candidates: Sequence[OutputCandidate],
    output_size: Tuple[int, int],
    channel_kind: str,
    local_warnings: WarningLog,
    channel_count: Optional[int] = None,
) -> Tuple[int, Dict[str, int]]:
    """Validate and publish one flat CUDA policy result batch."""

    global _WORKER_GPU_CODEC_WARNING_EMITTED
    torch = runtime["torch"]
    static = _WORKER_STATIC
    expected = len(candidates)
    if not bool(getattr(batch_images, "is_cuda", False)) or not bool(getattr(batch_masks, "is_cuda", False)):
        raise TypeError("GPU policy outputs must remain CUDA tensors")
    if getattr(batch_images, "dtype", None) != torch.uint8 or getattr(batch_masks, "dtype", None) != torch.uint8:
        raise TypeError("GPU policy outputs must be torch.uint8")
    expected_device = torch.device(f"cuda:{int(runtime['device_id'])}")
    image_device = torch.device(getattr(batch_images, "device", None))
    mask_device = torch.device(getattr(batch_masks, "device", None))
    if image_device != expected_device or mask_device != expected_device:
        raise ValueError(
            "GPU policy outputs must remain on their assigned device: "
            f"expected {expected_device}, got images={image_device}, masks={mask_device}"
        )
    if int(batch_images.ndim) != 4 or int(batch_images.shape[0]) != expected:
        raise ValueError(f"GPU image batch must be NCHW with N={expected}, got {tuple(batch_images.shape)}")
    expected_h, expected_w = int(output_size[0]), int(output_size[1])
    expected_channels = int(
        channel_count
        if channel_count is not None
        else (1 if str(channel_kind) == "gray" else 3)
    )
    if str(channel_kind) == "gray" and expected_channels != 1:
        raise ValueError(f"Gray GPU output requires one channel, got {expected_channels}")
    if str(channel_kind) == "rgb" and expected_channels != 3:
        raise ValueError(f"RGB GPU output requires three channels, got {expected_channels}")
    if str(channel_kind) == "custom" and expected_channels < 1:
        raise ValueError("Custom GPU output requires at least one channel")
    expected_image_shape = (expected, expected_channels, expected_h, expected_w)
    if tuple(int(x) for x in batch_images.shape) != expected_image_shape:
        raise ValueError(
            "GPU image batch has the wrong channel/spatial shape: "
            f"expected {expected_image_shape}, got {tuple(batch_images.shape)}"
        )
    if int(batch_masks.ndim) == 4 and int(batch_masks.shape[1]) == 1:
        batch_masks = batch_masks[:, 0]
    if int(batch_masks.ndim) != 3 or int(batch_masks.shape[0]) != expected:
        raise ValueError(f"GPU mask batch must be NHW with N={expected}, got {tuple(batch_masks.shape)}")
    expected_mask_shape = (expected, expected_h, expected_w)
    if tuple(int(x) for x in batch_masks.shape) != expected_mask_shape:
        raise ValueError(
            "GPU mask batch has the wrong spatial shape: "
            f"expected {expected_mask_shape}, got {tuple(batch_masks.shape)}"
        )

    out_dir: Path = static["out_dir"]  # type: ignore[assignment]
    split_active = bool(static["split_active"])
    image_format = str(static["image_format"])
    save_images = bool(static.get("save_images", True))
    save_labels = bool(static.get("save_labels", True))
    # A device-side reduction is enough for background/flip decisions. Full
    # masks cross PCIe only when labels were explicitly requested.
    mask_semantics_required = any(
        bool(candidate.label_enabled) or not bool(candidate.foreground)
        for candidate in candidates
    )
    if mask_semantics_required:
        mask_nonempty = (
            batch_masks.reshape(expected, -1)
            .any(dim=1)
            .detach()
            .to("cpu")
            .tolist()
        )
    else:
        # Unlabeled candidates carry no mask-dependent keep/drop semantics.
        # Avoid launching a reduction over the expanded zero-mask view.
        mask_nonempty = [False] * expected
    label_indices = [
        index
        for index, candidate in enumerate(candidates)
        if save_labels and bool(candidate.label_enabled)
    ]
    host_label_masks: Dict[int, np.ndarray] = {}
    if label_indices:
        device_indices = torch.as_tensor(
            label_indices,
            device=batch_masks.device,
            dtype=torch.int64,
        )
        selected_masks = batch_masks.index_select(0, device_indices).detach().to("cpu").numpy()
        host_label_masks = {
            int(candidate_index): np.ascontiguousarray((selected_masks[offset] > 0).astype(np.uint8))
            for offset, candidate_index in enumerate(label_indices)
        }
    keep_indices: List[int] = []
    image_paths: List[Path] = []
    label_payloads: List[Tuple[Path, List[str]]] = []
    flips_by_subset: Dict[str, int] = {}
    for local_index, cand in enumerate(candidates):
        is_nonempty = bool(mask_nonempty[local_index])
        if not bool(cand.foreground) and is_nonempty:
            raise RuntimeError(
                f"GPU augmentation synthesized a mask for known-background candidate {cand}"
            )
        img_path, lbl_path = candidate_output_paths(
            out_dir,
            cand,
            split_active=split_active,
            image_format=image_format,
        )
        label_lines: Optional[List[str]] = None
        if save_labels and cand.label_enabled and lbl_path is not None:
            mask_out = host_label_masks[int(local_index)]
            label_context = (
                f"{cand.volume_name} {cand.output_tag} "
                f"frame {int(cand.frame_idx) + 1:04d}"
            )
            label_lines = mask_to_yolo_lines(
                mask_out,
                warnings=local_warnings,
                context=label_context,
                known_empty=not bool(cand.foreground),
            )
            if int(cand.augmentation_index) > 0 and bool(cand.foreground) and not label_lines:
                local_warnings.add(
                    "augmented_foreground_flip_dropped",
                    f"{cand.volume_name}/{cand.output_tag}/frame={int(cand.frame_idx) + 1:04d}/tag={cand.augmentation_tag}",
                )
                subset_key = cand.split_subset or "all"
                flips_by_subset[subset_key] = int(flips_by_subset.get(subset_key, 0)) + 1
                continue
        elif (
            int(cand.augmentation_index) > 0
            and bool(cand.foreground)
            and bool(cand.label_enabled)
            and not is_nonempty
        ):
            local_warnings.add(
                "augmented_foreground_flip_dropped",
                f"{cand.volume_name}/{cand.output_tag}/frame={int(cand.frame_idx) + 1:04d}/tag={cand.augmentation_tag}",
            )
            subset_key = cand.split_subset or "all"
            flips_by_subset[subset_key] = int(flips_by_subset.get(subset_key, 0)) + 1
            continue
        keep_indices.append(int(local_index))
        if save_images:
            image_paths.append(img_path)
        if save_labels and label_lines is not None and lbl_path is not None:
            label_payloads.append((lbl_path, label_lines))

    fallback_note = None
    if save_images and keep_indices:
        fallback_note = _write_gpu_image_batch(
            runtime=runtime,
            images_nchw=batch_images,
            indices=keep_indices,
            paths=image_paths,
            channel_kind=str(channel_kind),
            image_format=image_format,
            png_compression=int(static["png_compression"]),
            jpeg_quality=int(static["jpeg_quality"]),
        )
    if fallback_note and not _WORKER_GPU_CODEC_WARNING_EMITTED:
        local_warnings.add(
            (
                "nvtiff_encode_fallback_to_opencv"
                if parse_output_image_format(image_format) == "tif"
                and str(channel_kind) == "custom"
                else "nvjpeg_encode_fallback_to_opencv"
            ),
            fallback_note,
        )
        _WORKER_GPU_CODEC_WARNING_EMITTED = True
    for lbl_path, label_lines in label_payloads:
        write_yolo_lines(label_lines, lbl_path)
    return len(keep_indices), flips_by_subset


def execute_gpu_frame_batch_task(
    volume: np.ndarray,
    mask: np.ndarray,
    plans: Sequence[RenderPlan],
    task: GpuFrameBatchTask,
) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
    """Overlap bounded CPU frame rendering with memory-safe GPU policy calls."""

    global _WORKER_GPU_BATCH_CAP_WARNING_EMITTED
    runtime = _gpu_runtime_for_worker()
    if isinstance(runtime, dict):
        runtime["radial_texture_required"] = any(
            plan.view.shared_view is not None
            and shared_geometry.is_radial_view(plan.view.shared_view)
            for plan in plans
        )
    policy = runtime["policy"]
    apply_batch_many = getattr(policy, "apply_batch_many", None)
    if not callable(apply_batch_many):
        written = 0
        flips_by_subset: Dict[str, int] = {}
        warnings = WarningLog()
        warnings.add(
            "gpu_policy_single_source_fallback",
            "External GPU policy has no apply_batch_many(); device batches cannot span source items",
        )
        for frame_task in task.frames:
            part_written, part_flips, counts, examples = execute_gpu_render_task(
                volume,
                mask,
                plans,
                frame_task,
            )
            written += int(part_written)
            for subset, count in part_flips.items():
                flips_by_subset[subset] = int(flips_by_subset.get(subset, 0)) + int(count)
            _merge_worker_warning_payload(warnings, counts, examples)
        return (
            written,
            flips_by_subset,
            {str(key): int(count) for key, count in warnings.counts.items()},
            {str(key): [str(x) for x in values] for key, values in warnings.examples.items()},
        )

    local_warnings = WarningLog()
    codec_error = str(runtime.get("codec_error") or "")
    if codec_error:
        local_warnings.add("nvjpeg_encode_fallback_to_opencv", codec_error)
    total_written = 0
    total_flips: Dict[str, int] = {}
    requested_batch_size = max(1, int(_WORKER_STATIC["gpu_batch_size"]))

    def _consume_rendered_work(work: Sequence[_GpuItemWork]) -> None:
        global _WORKER_GPU_BATCH_CAP_WARNING_EMITTED
        nonlocal total_written
        effective_batch_size = _gpu_memory_candidate_limit(
            runtime,
            work,
            requested_limit=requested_batch_size,
        )
        if (
            effective_batch_size < requested_batch_size
            and not _WORKER_GPU_BATCH_CAP_WARNING_EMITTED
        ):
            local_warnings.add(
                "gpu_batch_size_vram_capped",
                f"requested={requested_batch_size}, effective={effective_batch_size}",
            )
            _WORKER_GPU_BATCH_CAP_WARNING_EMITTED = True

        for initial_batch in _gpu_multi_source_work_batches(
            work,
            candidate_limit=effective_batch_size,
        ):
            pending_batches: List[Tuple[_GpuItemWork, ...]] = [tuple(initial_batch)]
            while pending_batches:
                batch = pending_batches.pop(0)
                flat_candidates = tuple(
                    candidate
                    for item in batch
                    for candidate in item.candidates
                )
                if any(
                    int(candidate.augmentation_index) > 0
                    and (candidate.augmentation_seed is None or not candidate.augmentation_tag)
                    for candidate in flat_candidates
                ):
                    raise RuntimeError("GPU augmented candidate is missing its seed or tag")
                seeds = tuple(
                    tuple(
                        None
                        if int(candidate.augmentation_index) == 0
                        else int(candidate.augmentation_seed)  # type: ignore[arg-type]
                        for candidate in item.candidates
                    )
                    for item in batch
                )
                try:
                    _wait_for_gpu_work_ready(runtime, batch)
                    if _gpu_identity_fast_path_eligible(batch, seeds):
                        result = _apply_gpu_identity_batch_many(runtime, batch)
                    else:
                        policy_images, zero_copy_cuda_sources = _gpu_policy_source_images(
                            policy,
                            batch,
                        )
                        if (
                            any(bool(getattr(item.image, "is_cuda", False)) for item in batch)
                            and not zero_copy_cuda_sources
                            and not bool(runtime.get("cuda_source_policy_fallback_announced"))
                        ):
                            local_warnings.add(
                                "gpu_policy_cuda_source_fallback",
                                "external policy lacks supports_cuda_sources; projected images cross CUDA->CPU->CUDA",
                            )
                            if isinstance(runtime, dict):
                                runtime["cuda_source_policy_fallback_announced"] = True
                        result = apply_batch_many(
                            images=policy_images,
                            # API-v2 policies historically receive writable,
                            # contiguous masks.  Preserve that contract here;
                            # the internal originals-only fast path can safely
                            # retain zero-strided blank views end to end.
                            masks=tuple(
                                np.ascontiguousarray(item.mask, dtype=np.uint8)
                                for item in batch
                            ),
                            seeds=seeds,
                            output_size=batch[0].output_size,
                        )
                except Exception as exc:
                    if _is_cuda_out_of_memory(exc) and len(flat_candidates) > 1:
                        try:
                            runtime["torch"].cuda.empty_cache()
                        except Exception:
                            pass
                        left, right = _split_gpu_work_batch(batch)
                        pending_batches[0:0] = [left, right]
                        local_warnings.add(
                            "gpu_batch_oom_reduced",
                            f"failed_candidates={len(flat_candidates)}, retry_candidates="
                            f"{sum(len(item.candidates) for item in left)}/"
                            f"{sum(len(item.candidates) for item in right)}",
                        )
                        continue
                    contexts = ", ".join(item.context for item in batch[:3])
                    raise RuntimeError(
                        f"GPU multi-source policy failed for {len(batch)} item(s) "
                        f"({contexts}): {type(exc).__name__}: {exc}"
                    ) from exc
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise TypeError("GPU policy apply_batch_many() must return (images_nchw, masks_nhw)")
                batch_written, batch_flips = _publish_gpu_policy_batch(
                    runtime=runtime,
                    batch_images=result[0],
                    batch_masks=result[1],
                    candidates=flat_candidates,
                    output_size=batch[0].output_size,
                    channel_kind=batch[0].channel_kind,
                    channel_count=int(batch[0].channel_count),
                    local_warnings=local_warnings,
                )
                total_written += int(batch_written)
                for subset, count in batch_flips.items():
                    total_flips[subset] = int(total_flips.get(subset, 0)) + int(count)

    render_jobs: List[
        Tuple[RenderPlan, int, Tuple[Tuple[str, Tuple[OutputCandidate, ...]], ...]]
    ] = []
    for frame_task in task.frames:
        plan = plans[int(frame_task.plan_idx)]
        tile_by_tag = {str(tile.tile_tag): tile for tile in plan.tile_layout}
        canvas_group: List[Tuple[str, Tuple[OutputCandidate, ...]]] = []
        for item_key, candidates in frame_task.items:
            if str(item_key) == "full":
                canvas_group.append((str(item_key), tuple(candidates)))
                continue
            tile = tile_by_tag[str(item_key)]
            if tile.shared_job is None:
                canvas_group.append((str(item_key), tuple(candidates)))
            else:
                render_jobs.append(
                    (plan, int(frame_task.frame_idx), ((str(item_key), tuple(candidates)),))
                )
        if canvas_group:
            render_jobs.append((plan, int(frame_task.frame_idx), tuple(canvas_group)))

    render_threads = min(
        len(render_jobs),
        max(1, int(_WORKER_STATIC.get("gpu_render_threads", 1))),
    )
    # Keep at most two CPU jobs per thread live. Completed results remain in
    # this bounded window while CUDA is busy, providing overlap without an
    # unbounded host-memory queue.
    max_in_flight = max(1, render_threads * 2)
    next_job = 0
    ready: List[_GpuItemWork] = []
    with ThreadPoolExecutor(
        max_workers=max(1, render_threads),
        thread_name_prefix="pta-gpu-render",
    ) as executor:
        pending: Dict[Future, int] = {}

        def _fill_render_window() -> None:
            nonlocal next_job
            while next_job < len(render_jobs) and len(pending) < max_in_flight:
                plan, frame_idx, items = render_jobs[next_job]
                future = executor.submit(
                    _render_gpu_item_group,
                    volume,
                    mask,
                    plan,
                    frame_idx,
                    items,
                    runtime,
                )
                pending[future] = int(next_job)
                next_job += 1

        _fill_render_window()
        while pending:
            done, _not_done = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: pending[item]):
                pending.pop(future)
                ready.extend(future.result())
            _fill_render_window()
            effective = _gpu_memory_candidate_limit(
                runtime,
                ready,
                requested_limit=requested_batch_size,
            )
            ready_candidates = sum(len(item.candidates) for item in ready)
            if _should_flush_ready_gpu_work(
                ready_candidates=ready_candidates,
                effective_candidate_limit=effective,
                producer_drained=(next_job >= len(render_jobs) and not pending),
            ):
                _consume_rendered_work(tuple(ready))
                ready.clear()

    if (
        bool(runtime.get("cartesian_renderer_announced"))
        and not bool(runtime.get("cartesian_renderer_manifest_announced"))
    ):
        local_warnings.add(
            "pta_cuda_cartesian_projection_active",
            "TTA resident Cartesian intensity projector; categorical masks retain CPU nearest sampling",
        )
        if isinstance(runtime, dict):
            runtime["cartesian_renderer_manifest_announced"] = True
    if (
        bool(runtime.get("radial_renderer_announced"))
        and not bool(runtime.get("radial_renderer_manifest_announced"))
    ):
        local_warnings.add(
            "pta_cuda_radial_projection_active",
            "TTA resident CUDA intensity projector; categorical masks retain CPU nearest sampling",
        )
        if isinstance(runtime, dict):
            runtime["radial_renderer_manifest_announced"] = True
    if (
        bool(runtime.get("tilted_renderer_announced"))
        and not bool(runtime.get("tilted_renderer_manifest_announced"))
    ):
        local_warnings.add(
            "pta_cuda_tilted_projection_active",
            "TTA fused Tilted Cartesian intensity projector; categorical masks retain CPU nearest sampling",
        )
        if isinstance(runtime, dict):
            runtime["tilted_renderer_manifest_announced"] = True
    return (
        total_written,
        total_flips,
        {str(key): int(count) for key, count in local_warnings.counts.items()},
        {str(key): [str(x) for x in values] for key, values in local_warnings.examples.items()},
    )


def execute_gpu_render_task(
    volume: np.ndarray,
    mask: np.ndarray,
    plans: Sequence[RenderPlan],
    task: object,
) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
    """Execute one grouped source-frame task on a persistent GPU rank."""
    global _WORKER_GPU_CODEC_WARNING_EMITTED
    if isinstance(task, GpuFrameBatchTask):
        return execute_gpu_frame_batch_task(volume, mask, plans, task)
    if not isinstance(task, FrameRenderTask):
        raise RuntimeError(f"Unsupported GPU render task type: {type(task).__name__}")
    runtime = _gpu_runtime_for_worker()
    torch = runtime["torch"]
    policy = runtime["policy"]
    static = _WORKER_STATIC
    out_dir: Path = static["out_dir"]  # type: ignore[assignment]
    split_active = bool(static["split_active"])
    image_format = str(static["image_format"])
    png_compression = int(static["png_compression"])
    jpeg_quality = int(static["jpeg_quality"])
    save_images = bool(static.get("save_images", True))
    save_labels = bool(static.get("save_labels", True))
    gpu_batch_size = max(1, int(static["gpu_batch_size"]))
    local_warnings = WarningLog()
    codec_error = str(runtime.get("codec_error") or "")
    if codec_error and not _WORKER_GPU_CODEC_WARNING_EMITTED:
        local_warnings.add(
            "nvjpeg_encode_fallback_to_opencv",
            codec_error,
        )
        _WORKER_GPU_CODEC_WARNING_EMITTED = True

    written = 0
    flips_by_subset: Dict[str, int] = {}
    plan = plans[int(task.plan_idx)]
    source = render_plan_frame_source(
        volume=volume,
        mask=mask,
        plan=plan,
        idx=int(task.frame_idx),
        need_canvas=any(item_key != "full" for item_key, _ in task.items),
    )
    for item_key, candidates in task.items:
        item_image, item_mask, output_size = _derive_gpu_item_source(source, plan, str(item_key))
        for chunk_start in range(0, len(candidates), gpu_batch_size):
            chunk = list(candidates[chunk_start:chunk_start + gpu_batch_size])
            seeds: List[Optional[int]] = []
            for cand in chunk:
                if int(cand.augmentation_index) == 0:
                    seeds.append(None)
                elif cand.augmentation_seed is None or not cand.augmentation_tag:
                    raise RuntimeError(f"GPU augmented candidate is missing its seed or tag: {cand}")
                else:
                    seeds.append(int(cand.augmentation_seed))
            try:
                batch_result = policy.apply_batch(
                    image=item_image,
                    mask=item_mask,
                    seeds=seeds,
                    output_size=tuple(int(x) for x in output_size),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"GPU policy failed for {plan.tag}/{item_key}/frame={int(task.frame_idx)+1:04d}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(batch_result, (tuple, list)) or len(batch_result) != 2:
                raise TypeError("GPU policy apply_batch() must return (images_nchw, masks_nhw)")
            batch_images, batch_masks = batch_result
            if not bool(getattr(batch_images, "is_cuda", False)) or not bool(getattr(batch_masks, "is_cuda", False)):
                raise TypeError("GPU policy outputs must remain CUDA tensors")
            if getattr(batch_images, "dtype", None) != torch.uint8 or getattr(batch_masks, "dtype", None) != torch.uint8:
                raise TypeError("GPU policy outputs must be torch.uint8")
            if int(batch_images.ndim) != 4 or int(batch_images.shape[0]) != len(chunk):
                raise ValueError(f"GPU image batch must be NCHW with N={len(chunk)}, got {tuple(batch_images.shape)}")
            if int(batch_masks.ndim) == 4 and int(batch_masks.shape[1]) == 1:
                batch_masks = batch_masks[:, 0]
            if int(batch_masks.ndim) != 3 or int(batch_masks.shape[0]) != len(chunk):
                raise ValueError(f"GPU mask batch must be NHW with N={len(chunk)}, got {tuple(batch_masks.shape)}")
            if tuple(int(x) for x in batch_images.shape[-2:]) != tuple(int(x) for x in output_size):
                raise ValueError(
                    f"GPU image output size {tuple(batch_images.shape[-2:])} != requested {output_size}"
                )
            if tuple(int(x) for x in batch_masks.shape[-2:]) != tuple(int(x) for x in output_size):
                raise ValueError(
                    f"GPU mask output size {tuple(batch_masks.shape[-2:])} != requested {output_size}"
                )

            host_masks = batch_masks.detach().to("cpu").numpy()
            keep_indices: List[int] = []
            image_paths: List[Path] = []
            label_payloads: List[Tuple[Path, List[str]]] = []
            for local_index, cand in enumerate(chunk):
                mask_out = np.ascontiguousarray((host_masks[local_index] > 0).astype(np.uint8))
                if not bool(cand.foreground) and np.any(mask_out):
                    raise RuntimeError(
                        f"GPU augmentation synthesized a mask for known-background candidate {cand}"
                    )
                img_path, lbl_path = candidate_output_paths(
                    out_dir,
                    cand,
                    split_active=split_active,
                    image_format=image_format,
                )
                label_lines: Optional[List[str]] = None
                if cand.label_enabled and lbl_path is not None:
                    label_context = f"{cand.volume_name} {cand.output_tag} frame {int(cand.frame_idx)+1:04d}"
                    label_lines = mask_to_yolo_lines(
                        mask_out,
                        warnings=local_warnings,
                        context=label_context,
                        known_empty=not bool(cand.foreground),
                    )
                    if int(cand.augmentation_index) > 0 and bool(cand.foreground) and not label_lines:
                        local_warnings.add(
                            "augmented_foreground_flip_dropped",
                            f"{cand.volume_name}/{cand.output_tag}/frame={int(cand.frame_idx)+1:04d}/tag={cand.augmentation_tag}",
                        )
                        subset_key = cand.split_subset or "all"
                        flips_by_subset[subset_key] = int(flips_by_subset.get(subset_key, 0)) + 1
                        continue
                keep_indices.append(int(local_index))
                if save_images:
                    image_paths.append(img_path)
                if save_labels and label_lines is not None and lbl_path is not None:
                    label_payloads.append((lbl_path, label_lines))

            fallback_note = None
            if save_images:
                fallback_note = _write_gpu_image_batch(
                    runtime=runtime,
                    images_nchw=batch_images,
                    indices=keep_indices,
                    paths=image_paths,
                    channel_kind=str(chunk[0].channel_kind),
                    image_format=image_format,
                    png_compression=png_compression,
                    jpeg_quality=jpeg_quality,
                )
            if fallback_note and not _WORKER_GPU_CODEC_WARNING_EMITTED:
                local_warnings.add(
                    (
                        "nvtiff_encode_fallback_to_opencv"
                        if parse_output_image_format(image_format) == "tif"
                        and str(chunk[0].channel_kind) == "custom"
                        else "nvjpeg_encode_fallback_to_opencv"
                    ),
                    fallback_note,
                )
                _WORKER_GPU_CODEC_WARNING_EMITTED = True
            for lbl_path, label_lines in label_payloads:
                write_yolo_lines(label_lines, lbl_path)
            written += len(keep_indices)

    warn_counts = {str(key): int(count) for key, count in local_warnings.counts.items()}
    warn_examples = {str(key): [str(x) for x in examples] for key, examples in local_warnings.examples.items()}
    return written, flips_by_subset, warn_counts, warn_examples


def execute_render_task(volume: np.ndarray, mask: np.ndarray, plans: Sequence[RenderPlan], task: object) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
    """Execute one source-frame task.

    Runs inside a persistent spawned process worker (normal), the explicit
    fork-only GPU-policy worker, or a thread; run-constant state comes from
    _WORKER_STATIC.  Returns the written count, flip-dropped counts keyed by
    split subset (for the realized-cap withhold), and the worker-local warning
    payload for the parent to merge.
    """
    static = _WORKER_STATIC
    if isinstance(static.get("augmentation"), LoadedGpuAugmentation):
        return execute_gpu_render_task(volume, mask, plans, task)
    augmentation: Optional[OfflineAugmentation] = static["augmentation"]  # type: ignore[assignment]
    out_dir: Path = static["out_dir"]  # type: ignore[assignment]
    split_active = bool(static["split_active"])
    image_format = str(static["image_format"])
    png_compression = int(static["png_compression"])
    jpeg_quality = int(static["jpeg_quality"])
    local_warnings = WarningLog()
    written = 0
    flips_by_subset: Dict[str, int] = {}

    def _write_versions(
        candidates: Sequence[OutputCandidate],
        item_image: np.ndarray,
        item_mask: np.ndarray,
        canonical_plan: Optional[RasterPlan],
    ) -> None:
        nonlocal written
        # Item arrays are task-local (warp/resize outputs), so replays reuse
        # them directly under the single-private-copy contract.
        for cand in candidates:
            outcome = write_selected_candidate_version(
                cand=cand,
                image=item_image,
                mask=item_mask,
                out_dir=out_dir,
                split_active=split_active,
                image_format=image_format,
                png_compression=png_compression,
                jpeg_quality=jpeg_quality,
                warnings=local_warnings,
                augmentation=augmentation if isinstance(augmentation, LoadedAugmentation) else None,
                inputs_are_private=True,
                save_images=bool(static.get("save_images", True)),
                save_labels=bool(static.get("save_labels", True)),
                canonical_plan=canonical_plan,
            )
            if outcome == "written":
                written += 1
            else:
                subset_key = cand.split_subset or "all"
                flips_by_subset[subset_key] = int(flips_by_subset.get(subset_key, 0)) + 1

    if isinstance(task, FrameRenderTask):
        plan = plans[int(task.plan_idx)]
        source = render_plan_frame_source(
            volume=volume, mask=mask, plan=plan, idx=int(task.frame_idx),
            need_canvas=any(item_key != "full" for item_key, _ in task.items),
        )
        for item_key, candidates in task.items:
            item_image, item_mask = _derive_item_arrays(source, plan, str(item_key))
            if str(item_key) == "full":
                item_plan = plan.canonical_plan
            else:
                item_plan = next(
                    (
                        tile.canonical_plan
                        for tile in plan.tile_layout
                        if str(tile.tile_tag) == str(item_key)
                    ),
                    None,
                )
            if plan.view.shared_view is not None and item_plan is None:
                raise RuntimeError(
                    f"shared PTA item {plan.tag}/{item_key} has no canonical RasterPlan"
                )
            _write_versions(candidates, item_image, item_mask, item_plan)
    else:
        raise RuntimeError(f"Unknown render task type: {type(task).__name__}")

    warn_counts = {str(key): int(count) for key, count in local_warnings.counts.items()}
    warn_examples = {str(key): [str(x) for x in examples] for key, examples in local_warnings.examples.items()}
    return written, flips_by_subset, warn_counts, warn_examples


# Worker-side caches for the process backend: per-phase task payloads and
# per-generation volume/mask attachments.  A worker can legitimately hold two
# generations at once (volume k phase B interleaves with volume k+1 phase A),
# so eviction keeps the newest two.
_WORKER_PAYLOAD_CACHE: Dict[str, Dict[str, object]] = {}
_WORKER_GEN_CACHE: Dict[int, Dict[str, object]] = {}
_WORKER_VOLUME_IDENTITY_BY_POINTER: Dict[int, str] = {}


def _worker_load_payload(payload_name: str, payload_nbytes: int) -> Dict[str, object]:
    cached = _WORKER_PAYLOAD_CACHE.get(payload_name)
    if cached is not None:
        return cached
    shm = _attach_shm_untracked(payload_name)
    try:
        payload: Dict[str, object] = pickle.loads(bytes(shm.buf[:int(payload_nbytes)]))
    finally:
        try:
            shm.close()
        except Exception:
            pass
    _WORKER_PAYLOAD_CACHE[payload_name] = payload
    gen = int(payload["gen"])  # type: ignore[arg-type]
    for stale_name in [name for name, p in _WORKER_PAYLOAD_CACHE.items() if int(p["gen"]) < gen - 1]:  # type: ignore[arg-type]
        _WORKER_PAYLOAD_CACHE.pop(stale_name, None)
    return payload


def _worker_volume_arrays(payload: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    gen = int(payload["gen"])  # type: ignore[arg-type]
    state = _WORKER_GEN_CACHE.get(gen)
    if state is None:
        volume_shm = _attach_shm_untracked(str(payload["volume_shm"]))
        mask_shm = _attach_shm_untracked(str(payload["mask_shm"]))
        volume = np.ndarray(tuple(int(x) for x in payload["volume_shape"]), dtype=np.uint8, buffer=volume_shm.buf)  # type: ignore[arg-type]
        mask = np.ndarray(tuple(int(x) for x in payload["mask_shape"]), dtype=np.uint8, buffer=mask_shm.buf)  # type: ignore[arg-type]
        state = {"volume": volume, "mask": mask, "shms": [volume_shm, mask_shm]}
        _WORKER_VOLUME_IDENTITY_BY_POINTER[
            int(volume.__array_interface__["data"][0])
        ] = f"shm:{payload['volume_shm']}"
        _WORKER_GEN_CACHE[gen] = state
        for stale_gen in [g for g in _WORKER_GEN_CACHE if int(g) < gen - 1]:
            stale = _WORKER_GEN_CACHE.pop(stale_gen)
            stale_volume = stale.get("volume")
            if isinstance(stale_volume, np.ndarray):
                _WORKER_VOLUME_IDENTITY_BY_POINTER.pop(
                    int(stale_volume.__array_interface__["data"][0]),
                    None,
                )
            stale["volume"] = None
            stale["mask"] = None
            for stale_shm in stale["shms"]:  # type: ignore[union-attr]
                try:
                    stale_shm.close()
                except Exception:
                    pass
    return state["volume"], state["mask"]  # type: ignore[return-value]


def _render_task_entry(job: Tuple[str, int, int]) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
    """Process-pool entry: resolve shared-memory payload + volume, run the task."""
    payload_name, payload_nbytes, task_idx = (str(job[0]), int(job[1]), int(job[2]))
    payload = _worker_load_payload(payload_name, payload_nbytes)
    volume, mask = _worker_volume_arrays(payload)
    tasks: List[object] = payload["tasks"]  # type: ignore[assignment]
    plans: List[RenderPlan] = payload["plans"]  # type: ignore[assignment]
    return execute_render_task(volume, mask, plans, tasks[task_idx])


def _render_task_entry_thread(payload: Mapping[str, object], task_idx: int) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
    """Thread-backend entry: the payload holds direct array references."""
    tasks: List[object] = payload["tasks"]  # type: ignore[assignment]
    plans: List[RenderPlan] = payload["plans"]  # type: ignore[assignment]
    return execute_render_task(payload["volume"], payload["mask"], plans, tasks[int(task_idx)])  # type: ignore[arg-type]


def _merge_worker_warning_payload(target: WarningLog, counts: Mapping[str, int], examples: Mapping[str, Sequence[str]]) -> None:
    with target.lock:
        for key, count in counts.items():
            target.counts[str(key)] += int(count)
        for key, exs in examples.items():
            _retain_deterministic_warning_examples(
                target.examples[str(key)], exs,
            )


def resolve_render_backend(requested: str) -> str:
    req = str(requested or "auto").strip().lower()
    if req not in {"auto", "process", "thread"}:
        raise ValueError("--worker_backend must be one of: auto, process, thread")
    if req == "thread":
        return "thread"
    try:
        multiprocessing.get_context("spawn")
    except Exception as exc:
        if req == "process":
            raise RuntimeError(
                "--worker_backend process requires multiprocessing spawn support"
            ) from exc
        return "thread"
    else:
        return "process"


def gpu_fork_render_backend_available() -> bool:
    """Whether the explicit PTA GPU-policy fork exception can be created."""

    if not hasattr(os, "fork"):
        return False
    try:
        multiprocessing.get_context("fork")
    except Exception:
        return False
    return True


@dataclass
class RenderPhaseHandle:
    """One installed phase: its task list plus the pool-visible payload."""
    payload_name: str
    payload_nbytes: int
    payload_block: Optional[SharedBlock]
    tasks: List[object]
    payload: Optional[Dict[str, object]]  # thread backend: direct reference


@dataclass
class VolumeRenderProgress:
    """Parent-side accounting for one volume across both render phases."""
    stem: str
    warnings: WarningLog
    pbar: Optional[tqdm] = None
    pending_a: int = 0
    pending_b: int = 0
    written: int = 0
    flips_by_subset: Dict[str, int] = field(default_factory=dict)


class PersistentRenderPool:
    """v18: one render pool for the entire run.

    Normal process rendering uses module-importable ``spawn`` workers, matching
    TTA's process-safety convention and avoiding inherited runtime/thread state.
    The active external GPU augmentation path is the sole fork-only exception.
    Volumes reach either process path through named shared-memory blocks;
    per-phase task payloads travel the same way.  The thread backend keeps one
    ThreadPoolExecutor and shares parent arrays directly.  Results arrive on a
    queue as (meta, result, exception) tuples so the parent can interleave
    phases of adjacent volumes (G5).
    """

    def __init__(
        self,
        *,
        backend: str,
        workers: int,
        worker_cpu_order: Sequence[int] = (),
        gpu_device_ids: Sequence[int] = (),
        gpu_cpu_sets: Sequence[Sequence[int]] = (),
    ):
        self.backend = str(backend)
        self.workers = max(1, int(workers))
        self.results: "queue.Queue[Tuple[Tuple[object, ...], Optional[tuple], Optional[BaseException]]]" = queue.Queue()
        self._pool = None
        self._executor: Optional[ThreadPoolExecutor] = None
        affinity_order = tuple(int(x) for x in worker_cpu_order)
        if self.backend == "process":
            gpu_ids = tuple(int(x) for x in gpu_device_ids)
            gpu_sets = tuple(tuple(int(cpu) for cpu in cpus) for cpus in gpu_cpu_sets)
            if gpu_ids:
                if not gpu_fork_render_backend_available():
                    raise RuntimeError(
                        "PTA GPU offline augmentation currently requires a fork-capable host; "
                        "normal CPU process rendering uses spawn"
                    )
                process_ctx = multiprocessing.get_context("fork")
                worker_static_payload: Optional[Mapping[str, object]] = None
                self.start_method = "fork"
            else:
                process_ctx = multiprocessing.get_context("spawn")
                worker_static_payload = _spawn_worker_static_payload()
                self.start_method = "spawn"
            self._pool = process_ctx.Pool(
                processes=self.workers,
                initializer=_render_worker_initializer,
                initargs=(
                    affinity_order,
                    gpu_ids,
                    gpu_sets,
                    worker_static_payload,
                ),
            )
        else:
            self.start_method = "thread"
            self._executor = ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="pretrain-render",
                initializer=_render_thread_initializer,
                initargs=(affinity_order,),
            )

    def install_phase(self, *, gen: int, prep: PreparedRenderVolume, tasks: Sequence[object]) -> RenderPhaseHandle:
        if self.backend != "process":
            payload: Dict[str, object] = {
                "gen": int(gen),
                "plans": list(prep.plans),
                "tasks": list(tasks),
                "volume": prep.volume_for_render,
                "mask": prep.mask_for_render,
            }
            return RenderPhaseHandle("", 0, None, list(tasks), payload)
        if prep.volume_render_block is None or prep.mask_render_block is None:
            raise RuntimeError(f"{prep.src.stem}: process render backend requires shared-memory volume/mask blocks")
        payload = {
            "gen": int(gen),
            "volume_shm": prep.volume_render_block.name,
            "volume_shape": tuple(int(x) for x in prep.volume_for_render.shape),
            "mask_shm": prep.mask_render_block.name,
            "mask_shape": tuple(int(x) for x in prep.mask_for_render.shape),
            "plans": list(prep.plans),
            "tasks": list(tasks),
        }
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        block = SharedBlock(len(blob))
        block.shm.buf[: len(blob)] = blob
        return RenderPhaseHandle(block.name, len(blob), block, list(tasks), None)

    def submit_phase(self, handle: RenderPhaseHandle, *, meta: Tuple[object, ...]) -> int:
        """Submit every task of a phase; completion lands on self.results."""
        results = self.results
        if self._pool is not None:
            for task_idx in range(len(handle.tasks)):
                self._pool.apply_async(
                    _render_task_entry,
                    ((handle.payload_name, int(handle.payload_nbytes), int(task_idx)),),
                    callback=(lambda result, _meta=meta: results.put((_meta, result, None))),
                    error_callback=(lambda exc, _meta=meta: results.put((_meta, None, exc))),
                )
        else:
            assert handle.payload is not None and self._executor is not None

            def _on_done(fut: Future, _meta: Tuple[object, ...] = meta) -> None:
                exc = fut.exception()
                if exc is not None:
                    results.put((_meta, None, exc))
                else:
                    results.put((_meta, fut.result(), None))

            for task_idx in range(len(handle.tasks)):
                self._executor.submit(_render_task_entry_thread, handle.payload, int(task_idx)).add_done_callback(_on_done)
        return len(handle.tasks)

    def close(self, *, terminate: bool = False) -> None:
        if self._pool is not None:
            if terminate:
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
            self._pool = None
        if self._executor is not None:
            self._executor.shutdown(wait=not terminate, cancel_futures=terminate)
            self._executor = None


def drain_render_results(pool: PersistentRenderPool, progress_by_gen: Mapping[int, VolumeRenderProgress], *, until: Callable[[], bool]) -> None:
    """Consume pool results (from any in-flight volume) until `until` holds."""
    while not until():
        try:
            meta, result, exc = pool.results.get(timeout=0.05)
        except queue.Empty:
            continue
        gen, phase = int(meta[0]), str(meta[1])
        progress = progress_by_gen[gen]
        if exc is not None:
            raise RuntimeError(f"Render task failed for {progress.stem} (phase {phase})") from exc
        assert result is not None
        written, flips_by_subset, warn_counts, warn_examples = result
        progress.written += int(written)
        task_flips = 0
        for subset_key, count in flips_by_subset.items():
            progress.flips_by_subset[str(subset_key)] = int(progress.flips_by_subset.get(str(subset_key), 0)) + int(count)
            task_flips += int(count)
        _merge_worker_warning_payload(progress.warnings, warn_counts, warn_examples)
        if phase == "A":
            progress.pending_a -= 1
        else:
            progress.pending_b -= 1
        if progress.pbar is not None:
            progress.pbar.update(int(written) + task_flips)

__all__ = [
    "ArrayAllocator",
    "FrameRenderTask",
    "GpuFrameBatchTask",
    "PersistentRenderPool",
    "PreparedRenderVolume",
    "RenderPhaseHandle",
    "SharedBlock",
    "VolumeRenderProgress",
    "WarningLog",
    "batch_gpu_frame_tasks",
    "bind_current_thread_to_cpus",
    "build_phase_render_tasks",
    "drain_render_results",
    "ensure_shared_uint8",
    "execute_gpu_frame_batch_task",
    "execute_gpu_render_task",
    "execute_render_task",
    "gpu_fork_render_backend_available",
    "make_volume_allocator",
    "projection_phase_summary",
    "resolve_render_backend",
    "set_worker_static_context",
]
