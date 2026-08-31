#!/usr/bin/env python3
# The implementation has an ordinary unversioned package identity; release
# constituent provenance is recorded outside the release package.
"""Unified v18 pretraining augmentation and dataset-publication engine.

The public interface is defined by :mod:`XTA.pta_config`. Built-in physical
views, frame addressing, channel assembly, tiles, and categorical rendering
use the same :mod:`XTA.geometry` implementation as TTA. PTA owns source
discovery, label eligibility, dataset splitting, augmentation, and publication.

Generate view-augmented, single-class YOLO segmentation training data from
fully labeled, partially labeled, or unlabeled transverse volumes.

Core behavior:
  - Loads the full source image volume and, when labels exist, the rasterized
    binary mask volume in RAM before reslicing.
  - Classifies inputs as Fully Labeled, Partially Labeled, or Unlabeled per the
    inherited volume rules.
  - Applies Gaussian 3D mask smoothing when enabled for fully labeled volumes.
  - Cubic-resizes complete 3D volumes before view extraction.
  - Emits single-channel gray, triplicated RGB, or custom neighboring-slice
    channel stacks for active transverse, sagittal, coronal, radial,
    tilted-transverse, and tile variants.  Arbitrary-channel stacks are stored
    as multi-page TIFF with one grayscale page per channel; labels always come
    from the center slice.
  - Optionally loads one external CPU Albumentations or GPU Python policy and
    emits a deterministic, split-aware number of augmented versions of each
    retained output image/label pair.
  - Supports optional NRRD mask snapshots, FFV1 MKV blue-overlay videos, and
    voxel counts for labeled inputs.

Runtime architecture:
  - One persistent process pool serves the full run. CPU workers use spawn and
    named shared-memory volumes; GPU-policy workers use the isolated CUDA
    process path required by the external factory contract.
  - Source-frame rendering is shared by each full-frame/tile item and all of
    its deterministic augmentation copies.
  - GPU policies may implement the v2 multi-source batch contract so work from
    several source frames fills a device batch. Older single-source policies
    remain supported through a compatibility fallback.
  - Volume loading and planning overlap bounded rendering of the preceding
    volume when the configured memory budget permits it.

Dependencies:
  pip install opencv-python numpy scipy tqdm
  pip install albumentations  # needed only for --augmentation_execution offline
  pip install torch  # CUDA build; needed by the supplied GPU policy
  pip install nvidia-nvimgcodec-cu13[all]  # CUDA 13; use the cu12 package on CUDA 12
  pip install nvidia-nvtiff-cu13  # optional lossless multipage TIFF GPU encoder
  pip install pynrrd     # needed for NRRD input or --save_nrrd
System:
  ffmpeg + ffprobe on PATH for video input and overlay output.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("SciPy is required: pip install scipy") from exc

try:
    from tqdm import tqdm  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tqdm is required: pip install tqdm") from exc

# v18 routes built-in physical-view planning and forward rendering through the
# same implementation used by TTA.  PTA-owned discovery, eligibility, splitting,
# augmentation, and dataset publication remain in this module.
from . import geometry as shared_geometry
from . import media as shared_media
from .config import SCRIPT_BASENAME, SCRIPT_VERSION
from .gaussian import binary_gaussian_pass
from .unification.manifest import write_json_manifest
from .unification.contracts import RasterPlan
from .pta_augmentation import (
    AugmentationDefinition,
    LoadedAugmentation,
    LoadedGpuAugmentation,
    OfflineAugmentation,
    _augmented_image_to_uint8,
    _augmented_mask_to_binary,
    _load_external_python_module,
    apply_augmentation_pair,
    assert_augmentation_definition_unchanged,
    assert_augmentation_did_not_synthesize_mask,
    inspect_augmentation_definition,
    load_augmentation_definition,
    load_gpu_augmentation_definition,
    load_offline_augmentation_definition,
    validate_seedable_augmentation_pipeline,
)
from .pta_dataset import (
    AUGMENTATION_TAG_ALPHABET,
    AUGMENTATION_TAG_LENGTH,
    AugmentationStats,
    BackgroundFilterStats,
    OutputCandidate,
    SplitStats,
    apply_background_filter,
    apply_dataset_split,
    assign_volume_split_by_stem,
    augmentation_digest,
    augmentation_seed_for_identity,
    background_limit_from_foreground,
    base62_tag_from_digest,
    candidate_atomic_key,
    candidate_background_rank,
    candidate_source_identity,
    deterministic_augmentation_tag,
    finalize_augmentation_stats,
    finalize_background_filter_with_augmentations,
    output_source_identity_text,
    plan_augmented_versions,
    refresh_retained_original_split_stats,
    split_round_half_toward_train,
    stable_digest_rank,
    withhold_background_overage_after_flips,
)
from .pta_publication import (
    OUTPUT_IMAGE_FORMATS,
    PtaDatasetImageSink,
    _CREATED_OUTPUT_DIRS,
    _CREATED_OUTPUT_DIRS_LOCK,
    _private_image_stage_path,
    _validate_jpeg_file,
    _validate_nonempty_regular_file,
    _write_nvjpeg_batch_atomically,
    candidate_output_paths,
    ensure_output_parent_once,
    ensure_tiff_output_available,
    mask_to_yolo_lines,
    output_image_suffix,
    parse_output_image_format,
    publish_pta_candidate_image_batch,
    verify_published_image_tree,
    write_image,
    write_image_gray,
    write_label_from_mask,
    write_selected_candidate_version,
    write_yolo_lines,
)
from .pta_rendering import (
    DEFAULT_CHANNEL_VARIANT,
    RADIAL_LANCZOS_A,
    AffineSpec,
    ChannelFormat,
    ChannelVariant,
    RadialSampler,
    RenderFrameSource,
    RenderPlan,
    RenderTileItem,
    TileConfig,
    TiltPlan,
    ViewInfo,
    _RADIAL_CACHE,
    _RADIAL_CACHE_LOCK,
    _TILT_PLAN_CACHE,
    _TILT_PLAN_CACHE_LOCK,
    _require_pta_canonical_plan,
    _shared_aug_job_for_affine,
    _shared_render_canvas_intensity,
    _shared_render_canvas_mask,
    _shared_render_full_intensity,
    _shared_render_full_mask,
    edge_clamped_view_index,
    encoded_channel_source_positions,
    extract_padded_tile,
    get_native_view_frame,
    get_native_view_image,
    get_native_view_mask,
    get_radial_sampler,
    get_tilt_plan,
    lanczos_kernel,
    lanczos_offsets,
    normalize_lanczos_weight_rows,
    radial_extract_lanczos,
    render_channel_formatted_images,
    render_full_and_optional_canvas,
    render_image_full_and_optional_canvas,
    render_plan_frame_mask_source,
    render_plan_frame_source,
    render_shared_tile_images,
    render_tilted_on_grid,
    resize_centered,
    tilt_plan_key,
    warp_image,
    warp_mask_only,
    warp_pair,
)
from .pta_workers import (
    ArrayAllocator,
    FrameRenderTask,
    GpuFrameBatchTask,
    PersistentRenderPool,
    PreparedRenderVolume,
    RenderPhaseHandle,
    SharedBlock,
    VolumeRenderProgress,
    WarningLog,
    _GpuItemWork,
    _THREAD_AFFINITY_LOCK,
    _WORKER_GEN_CACHE,
    _WORKER_PAYLOAD_CACHE,
    _WORKER_STATIC,
    _WORKER_VOLUME_IDENTITY_BY_POINTER,
    _apply_gpu_identity_batch_many,
    _attach_shm_untracked,
    _derive_gpu_item_source,
    _derive_item_arrays,
    _frame_render_candidate_count,
    _gpu_identity_fast_path_eligible,
    _gpu_policy_host_image,
    _gpu_policy_source_images,
    _gpu_projected_item_image,
    _gpu_runtime_for_worker,
    _initialize_spawned_worker_static_context,
    _merge_worker_warning_payload,
    _nvjpeg_encode_params,
    _nvjpeg_samples_nhwc,
    _nvjpeg_wrap_samples,
    _publish_gpu_policy_batch,
    _render_gpu_frame_items,
    _render_gpu_item_group,
    _render_task_entry,
    _render_task_entry_thread,
    _render_thread_initializer,
    _render_worker_initializer,
    _spawn_worker_static_payload,
    _wait_for_gpu_work_ready,
    _worker_load_payload,
    _worker_volume_arrays,
    _write_gpu_image_batch,
    _zero_mask_view,
    batch_gpu_frame_tasks,
    bind_current_thread_to_cpus,
    build_phase_render_tasks,
    drain_render_results,
    ensure_shared_uint8,
    execute_gpu_frame_batch_task,
    execute_gpu_render_task,
    execute_render_task,
    gpu_fork_render_backend_available,
    make_volume_allocator,
    projection_phase_summary,
    resolve_render_backend,
    set_worker_static_context,
)
from .pta_scheduler import (
    gpu_memory_candidate_limit as _gpu_memory_candidate_limit,
    is_cuda_out_of_memory as _is_cuda_out_of_memory,
    iter_compatible_work_batches as _gpu_multi_source_work_batches,
    resolve_gpu_worker_layout,
    should_flush_ready_gpu_work as _should_flush_ready_gpu_work,
    split_work_batch as _split_gpu_work_batch,
)
from .unification.channels import resolve_channel_variants as resolve_v18_channel_variants
from .workspace import radial_source_mode as _radial_source_mode
from .unification.runtime import compile_physical_views
from .unification.sampling import (
    build_forward_raster_plan,
    forward_sampling_execution_record,
    forward_sampling_policy,
    require_forward_sampling,
)


GIB = 1024 ** 3
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_EXTS = {".mkv", ".mp4", ".mov", ".avi", ".mpg", ".mpeg", ".m4v", ".webm"}
NRRD_EXTS = {".nrrd", ".nhdr"}
PIPELINE_SPEC_VERSION = f"v{SCRIPT_VERSION}"
NRRD_SPACE = "left-posterior-superior"
NRRD_AXIS_ORDER_NOTE = "internal mask (t,Y,X) is exported as Slicer spatial axes (X,Y,t)"

ANNOTATION_UNANNOTATED = 0
ANNOTATION_BACKGROUND = 1
ANNOTATION_FOREGROUND = 2
ANNOTATION_STATE_NAMES = {
    ANNOTATION_UNANNOTATED: "unannotated",
    ANNOTATION_BACKGROUND: "annotated_background",
    ANNOTATION_FOREGROUND: "annotated_foreground",
}


# ---------------------------------------------------------------------------
# Warnings / small helpers
# ---------------------------------------------------------------------------



def _require_bin(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found on PATH: {name}")


def visible_cpu_count() -> int:
    # Process affinity is the authoritative limit inside SLURM/cgroups.  Broad
    # SLURM variables can describe the node rather than this process and used
    # to cause substantial oversubscription on shared allocations.
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, len(affinity))
    except Exception:
        pass
    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            m = re.search(r"\d+", raw)
            if m:
                return max(1, int(m.group(0)))
    return max(1, int(os.cpu_count() or 1))


def choose_workers(requested: int) -> int:
    if int(requested) > 0:
        return max(1, int(requested))
    return max(1, visible_cpu_count())


def probe_gpu_offline_runtime(*, require_nvjpeg: bool, expected_device_count: int) -> str:
    """Fail before output cleanup when the requested CUDA runtime is unusable.

    The probe runs out-of-process so the parent remains CUDA-uninitialized and
    can safely fork one persistent rank per GPU afterwards.
    """
    statements = [
        "import torch",
        "nvimgcodec_version = 'not-requested'",
        "assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'",
        "count = int(torch.cuda.device_count())",
        "assert count > 0, 'torch.cuda.device_count() is zero'",
        f"assert count == {int(expected_device_count)}, "
        f"f'topology discovered {int(expected_device_count)} CUDA-visible GPU(s), but PyTorch sees {{count}}'",
    ]
    if require_nvjpeg:
        statements.extend([
            "from nvidia import nvimgcodec",
            "assert hasattr(nvimgcodec, 'Encoder'), 'nvImageCodec Encoder API is unavailable'",
            "assert hasattr(nvimgcodec, 'as_images'), 'nvImageCodec as_images API is unavailable; install >=0.7'",
            "assert hasattr(nvimgcodec.BackendKind, 'HYBRID_CPU_GPU'), "
            "'nvImageCodec HYBRID_CPU_GPU backend is unavailable'",
            "nvimgcodec_version = getattr(nvimgcodec, '__version__', 'unknown')",
        ])
    statements.append(
        "print(f'torch={torch.__version__}; torch_cuda={torch.version.cuda}; "
        "visible_gpus={count}; nvimgcodec={nvimgcodec_version}')"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "; ".join(statements)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except Exception as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        requirement = "CUDA PyTorch plus nvidia-nvimgcodec" if require_nvjpeg else "CUDA PyTorch"
        raise RuntimeError(
            f"GPU offline runtime probe failed; required={requirement}"
            + (f"; detail={detail}" if detail else "")
        ) from exc
    return proc.stdout.strip()


@dataclass(frozen=True)
class TopologyPlan:
    allowed_cpus: Tuple[int, ...]
    cuda_device_ids: Tuple[int, ...]       # CUDA-visible (logical) ids
    gpu_physical_ids: Tuple[str, ...]
    gpu_cpu_sets: Tuple[Tuple[int, ...], ...]
    worker_cpu_order: Tuple[int, ...]
    discovery: str

    @property
    def summary(self) -> str:
        if not self.cuda_device_ids:
            return f"{self.discovery}; allocated GPUs=none; CPUs={list(self.allowed_cpus)}"
        mappings = ", ".join(
            f"cuda:{logical}/physical:{physical}->cpus:{list(cpus)}"
            for logical, physical, cpus in zip(self.cuda_device_ids, self.gpu_physical_ids, self.gpu_cpu_sets)
        )
        return f"{self.discovery}; {mappings}"


def parse_linux_cpu_list(text: str) -> Tuple[int, ...]:
    values: set[int] = set()
    for token in str(text).strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                lo, hi = hi, lo
            values.update(range(lo, hi + 1))
        else:
            values.add(int(token))
    return tuple(sorted(values))


def _allowed_cpu_tuple() -> Tuple[int, ...]:
    try:
        values = tuple(sorted(int(x) for x in os.sched_getaffinity(0)))
        if values:
            return values
    except Exception:
        pass
    return tuple(range(visible_cpu_count()))


def _gpu_records_from_nvidia_smi() -> List[Dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:
        return []
    records: List[Dict[str, str]] = []
    for line in proc.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3:
            records.append({"index": fields[0], "uuid": fields[1], "pci": fields[2]})
    return records


def _visible_gpu_records(records: Sequence[Mapping[str, str]]) -> List[Mapping[str, str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return list(records)
    tokens = [part.strip() for part in raw.split(",") if part.strip() and part.strip() != "-1"]
    selected: List[Mapping[str, str]] = []
    for token in tokens:
        match = next(
            (
                rec for rec in records
                if token == str(rec.get("index", ""))
                or str(rec.get("uuid", "")).startswith(token)
                or token.startswith(str(rec.get("uuid", "")))
            ),
            None,
        )
        if match is not None:
            selected.append(match)
        else:
            # MIG identifiers and scheduler aliases do not always map to the
            # parent GPU query.  Retain the logical slot with unknown locality.
            selected.append({"index": token, "uuid": token, "pci": ""})
    return selected


def _cpus_for_pci_device(pci_bus_id: str, allowed: Sequence[int]) -> Tuple[int, ...]:
    pci = str(pci_bus_id).strip().lower()
    candidates = [pci]
    if len(pci) > 12 and pci.startswith("00000000:"):
        candidates.append("0000:" + pci.split(":", 1)[1])
    numa_node: Optional[int] = None
    for candidate in candidates:
        path = Path("/sys/bus/pci/devices") / candidate / "numa_node"
        try:
            numa_node = int(path.read_text().strip())
            break
        except Exception:
            continue
    if numa_node is None or numa_node < 0:
        return ()
    try:
        node_cpus = parse_linux_cpu_list((Path("/sys/devices/system/node") / f"node{numa_node}" / "cpulist").read_text())
    except Exception:
        return ()
    allowed_set = set(int(x) for x in allowed)
    return tuple(cpu for cpu in node_cpus if cpu in allowed_set)


def _partition_gpu_cpu_sets(raw_sets: Sequence[Sequence[int]], allowed: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    """Give GPUs on the same NUMA node disjoint subsets of allocated CPUs."""
    fallback = tuple(int(x) for x in allowed)
    normalized = [tuple(sorted(set(int(x) for x in cpus))) or fallback for cpus in raw_sets]
    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for index, cpus in enumerate(normalized):
        groups[cpus].append(index)
    result: List[Tuple[int, ...]] = [tuple() for _ in normalized]
    for cpus, indices in groups.items():
        for rank, index in enumerate(indices):
            assigned = tuple(cpus[rank::len(indices)])
            result[index] = assigned or cpus
    return tuple(result)


def discover_topology(*, enabled: bool, warnings: WarningLog) -> TopologyPlan:
    allowed = _allowed_cpu_tuple()
    # CUDA-visible device discovery is required even when NUMA binding is
    # disabled; --no-topology_aware suppresses locality decisions, not GPUs.
    records = _gpu_records_from_nvidia_smi()
    visible_records = _visible_gpu_records(records)
    if not visible_records:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        tokens = [part.strip() for part in raw.split(",") if part.strip() and part.strip() != "-1"]
        visible_records = [{"index": token, "uuid": token, "pci": ""} for token in tokens]
    logical_ids = tuple(range(len(visible_records)))
    physical_ids = tuple(str(rec.get("index") or rec.get("uuid") or "unknown") for rec in visible_records)
    raw_cpu_sets = (
        tuple(_cpus_for_pci_device(str(rec.get("pci", "")), allowed) for rec in visible_records)
        if enabled
        else tuple(tuple() for _ in visible_records)
    )
    gpu_cpu_sets = (
        _partition_gpu_cpu_sets(raw_cpu_sets, allowed)
        if enabled and visible_records
        else tuple(tuple() for _ in visible_records)
    )
    if enabled and visible_records and not any(raw_cpu_sets):
        warnings.add("topology_locality_fallback", "GPU PCI/NUMA locality unavailable; allocated CPUs were partitioned evenly")
    worker_order: List[int] = []
    if gpu_cpu_sets and any(gpu_cpu_sets):
        for offset in range(max(len(cpus) for cpus in gpu_cpu_sets)):
            for cpus in gpu_cpu_sets:
                if offset < len(cpus):
                    worker_order.append(int(cpus[offset]))
    else:
        worker_order.extend(int(x) for x in allowed)
    return TopologyPlan(
        allowed_cpus=allowed,
        cuda_device_ids=logical_ids,
        gpu_physical_ids=physical_ids,
        gpu_cpu_sets=gpu_cpu_sets,
        worker_cpu_order=tuple(worker_order),
        discovery="topology-aware" if enabled else "topology discovery disabled",
    )




def available_memory_budget_bytes() -> Optional[int]:
    candidates: List[int] = []
    cgroup_pairs = [
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    ]
    for limit_path, used_path in cgroup_pairs:
        try:
            limit_text = limit_path.read_text().strip()
            if limit_text != "max":
                limit = int(limit_text)
                used = int(used_path.read_text().strip())
                if 0 < limit < (1 << 60):
                    candidates.append(max(0, limit - used))
        except Exception:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except Exception:
        pass
    positive = [value for value in candidates if value > 0]
    return min(positive) if positive else None


def probe_image_dimensions(path: Path) -> Tuple[int, int]:
    if shutil.which("ffprobe") is not None:
        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            width_text, height_text = proc.stdout.strip().split("x", 1)
            return int(height_text), int(width_text)
        except Exception:
            pass
    gray = _read_gray_image_opencv(path)
    return int(gray.shape[0]), int(gray.shape[1])


def estimate_spec_resident_bytes(spec: object) -> Optional[int]:
    """Conservative source+cube image/mask resident estimate for depth-2 guard."""
    try:
        t_dim = int(getattr(spec, "frame_count_hint"))
        if str(getattr(spec, "kind")) == "sequence":
            image_map = getattr(spec, "image_paths_by_index")
            first_path = image_map[sorted(image_map)[0]]
            h, w = probe_image_dimensions(first_path)
        else:
            video_path = getattr(spec, "video_path")
            info = ffprobe_info(video_path)
            h, w = int(info["height"]), int(info["width"])
        target = cubic_target_shape((t_dim, h, w))
        source_pair = 2 * int(t_dim) * int(h) * int(w)
        processed_pair = 2 * int(target[0]) * int(target[1]) * int(target[2])
        # Allow for resize/filter scratch and worker-local frame buffers.
        return int(math.ceil(1.35 * float(source_pair + processed_pair)))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared-memory volume blocks
# ---------------------------------------------------------------------------



def parallel_for_indices(
    count: int,
    func: Callable[[int], None],
    *,
    workers: int,
    desc: str,
    show_progress: bool = True,
    worker_cpu_order: Sequence[int] = (),
) -> None:
    total = max(0, int(count))
    if total <= 0:
        return
    nworkers = max(1, min(int(workers), total))
    if nworkers <= 1:
        iterable: Iterable[int] = tqdm(range(total), desc=desc) if show_progress else range(total)
        for idx in iterable:
            func(int(idx))
        return
    affinity_queue: "queue.SimpleQueue[int]" = queue.SimpleQueue()
    affinity_values = tuple(int(x) for x in worker_cpu_order)
    for worker_index in range(nworkers):
        if affinity_values:
            affinity_queue.put(int(affinity_values[worker_index % len(affinity_values)]))

    def _initializer() -> None:
        if affinity_values:
            bind_current_thread_to_cpus((affinity_queue.get(),))

    with ThreadPoolExecutor(max_workers=nworkers, initializer=_initializer) as executor:
        futures = [executor.submit(func, int(i)) for i in range(total)]
        if show_progress:
            with tqdm(total=total, desc=desc) as pbar:
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)
        else:
            for fut in as_completed(futures):
                fut.result()


def parse_token_list(values: Sequence[str] | str | None) -> List[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else [str(v) for v in values]
    out: List[str] = []
    for raw in raw_values:
        raw = str(raw).strip()
        if not raw:
            continue
        out.extend([p for p in re.split(r"[,\s]+", raw) if p])
    return out


def parse_float_list(values: Sequence[str] | str | None, default: Optional[Sequence[float]] = None) -> List[float]:
    tokens = parse_token_list(values)
    if not tokens and default is not None:
        return [float(x) for x in default]
    return [float(x) for x in tokens]


def parse_int_list(values: Sequence[str] | str | int | None) -> List[int]:
    if values is None:
        return []
    if isinstance(values, int):
        return [int(values)]
    return [int(float(x)) for x in parse_token_list(values)]




def format_float_token(v: float) -> str:
    s = f"{float(v):g}".replace("-", "m").replace("+", "p").replace(".", "p")
    return s


def format_signed_token(v: float) -> str:
    sign = "p" if float(v) >= 0 else "m"
    return sign + f"{abs(float(v)):g}".replace(".", "p")


def natural_index_key(path: Path) -> Tuple[str, int, str]:
    stem = path.stem
    m = re.search(r"(.*?)(\d+)$", stem)
    if m:
        return (m.group(1), int(m.group(2)), path.name)
    return (stem, -1, path.name)


def split_stem_index(path: Path) -> Tuple[str, Optional[int]]:
    m = re.match(r"^(.*?)(?:_)?(\d+)$", path.stem)
    if not m:
        return path.stem, None
    base = m.group(1).rstrip("_")
    return base, int(m.group(2))


def to_gray8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape}")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(np.clip(np.rint(arr.astype(np.float32) / 257.0), 0, 255).astype(np.uint8))
    amin = float(np.nanmin(arr)) if arr.size else 0.0
    amax = float(np.nanmax(arr)) if arr.size else 0.0
    if not math.isfinite(amin) or not math.isfinite(amax) or amax <= amin:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    scaled = (arr.astype(np.float32) - amin) * (255.0 / (amax - amin))
    return np.ascontiguousarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



_CUSTOM_CHANNEL_FORMAT_RE = re.compile(r"^C([1-9]\d*)S([1-9]\d*)$", re.IGNORECASE)


def resolve_channel_formats(values: Sequence[str] | str | None) -> List[ChannelFormat]:
    """Validate and canonicalize ``--channel_format`` values.

    Gray/grey and RGB may be mixed freely with custom formats.  When multiple
    custom C...S... values are supplied, their C value must match while S may
    vary, e.g. C5S1 C5S2.
    """
    tokens = parse_token_list(values or ["gray"])
    if not tokens:
        tokens = ["gray"]
    formats: List[ChannelFormat] = []
    seen: set[str] = set()
    custom_channel_count: Optional[int] = None
    for raw in tokens:
        normalized = str(raw).strip()
        lowered = normalized.lower()
        if lowered in {"gray", "grey"}:
            fmt = ChannelFormat("gray", "gray", 1, 1)
        elif lowered == "rgb":
            fmt = ChannelFormat("RGB", "rgb", 3, 1)
        else:
            match = _CUSTOM_CHANNEL_FORMAT_RE.fullmatch(normalized)
            if match is None:
                raise ValueError(
                    f"Unsupported --channel_format value {raw!r}; use gray/grey, RGB, or C{{odd}}S{{stride>=1}}"
                )
            channel_count = int(match.group(1))
            stride = int(match.group(2))
            if channel_count % 2 == 0:
                raise ValueError(
                    f"--channel_format {raw!r} is invalid: C must be odd, got C={channel_count}"
                )
            if stride < 1:
                raise ValueError(
                    f"--channel_format {raw!r} is invalid: S must be an integer >= 1, got S={stride}"
                )
            if custom_channel_count is None:
                custom_channel_count = channel_count
            elif channel_count != custom_channel_count:
                raise ValueError(
                    "All custom --channel_format values must use the same C value; "
                    f"got C={custom_channel_count} and C={channel_count}"
                )
            fmt = ChannelFormat(f"C{channel_count}S{stride}", "custom", channel_count, stride)
        if fmt.token not in seen:
            formats.append(fmt)
            seen.add(fmt.token)
    return formats


def expand_channel_variants(formats: Sequence[ChannelFormat]) -> List[ChannelVariant]:
    variants: List[ChannelVariant] = []
    for fmt in formats:
        for shared_variant in resolve_v18_channel_variants("pta", str(fmt.token)):
            variants.append(ChannelVariant(
                str(shared_variant.layout.token),
                str(shared_variant.layout.kind),
                int(shared_variant.layout.channel_count),
                int(shared_variant.layout.stride),
                bool(shared_variant.is_reversed),
                tuple(int(value) for value in shared_variant.offsets),
            ))
    return variants


def make_channel_tag(base_tag: str, variant: ChannelVariant) -> str:
    # Preserve v4 filenames for the default single-channel output.  Additional
    # formats receive explicit suffixes so multiple requested formats cannot
    # collide within one run.
    if variant.kind == "gray":
        return str(base_tag)
    return f"{base_tag}_{variant.tag_token}"


def resolve_tilt_angles(values: Sequence[str] | str | None) -> List[float]:
    out: List[float] = []
    seen: set[float] = set()
    for a in parse_float_list(values, default=[0.0]):
        if float(a) == 0.0:
            continue
        mag = abs(float(a))
        if not (0.0 < mag <= 45.0):
            raise ValueError("--tilt_angle values must be greater than 0 and at most 45")
        key = round(mag, 8)
        if key not in seen:
            out.append(mag)
            seen.add(key)
    return out


def resolve_tilt_directions(values: Sequence[str] | str | None) -> List[str]:
    raw = [str(x).lower() for x in parse_token_list(values or ["vertical"])]
    out: List[str] = []
    for tok in raw:
        if tok == "both":
            for expanded in ("vertical", "horizontal"):
                if expanded not in out:
                    out.append(expanded)
            continue
        if tok not in {"vertical", "horizontal"}:
            raise ValueError("--tilt_direction must contain vertical, horizontal, or both")
        if tok not in out:
            out.append(tok)
    return out or ["vertical"]


def resolve_tile_configs(tile_sizes_raw: Sequence[str] | str | int | None, tile_strides_raw: Sequence[str] | str | int | None) -> List[TileConfig]:
    sizes = parse_int_list(tile_sizes_raw)
    if not sizes:
        sizes = [0]
    if len(sizes) == 1 and sizes[0] == 0:
        strides = parse_int_list(tile_strides_raw)
        if strides and any(int(s) != 0 for s in strides):
            raise ValueError("--tile_stride must be omitted or 0 when --tile_size is 0")
        return []
    if any(int(s) <= 0 for s in sizes):
        raise ValueError("All --tile_size values must be > 0 when tiling is enabled")

    strides = parse_int_list(tile_strides_raw)
    if not strides:
        strides = list(sizes)
    if len(strides) != len(sizes):
        raise ValueError("--tile_size and --tile_stride must have the same number of values")
    configs: List[TileConfig] = []
    seen: set[str] = set()
    for size, stride in zip(sizes, strides):
        if int(stride) <= 0:
            raise ValueError("--tile_stride must be > 0 when --tile_size is active")
        cid = f"s{int(size)}_st{int(stride)}"
        if cid in seen:
            raise ValueError(f"Duplicate tile config: {cid}")
        configs.append(TileConfig(int(size), int(stride), cid))
        seen.add(cid)
    # v3 canonical order: tile variants sorted by (tile_size ascending, tile_stride ascending).
    return sorted(configs, key=lambda c: (int(c.tile_size), int(c.tile_stride), c.config_id))


# ---------------------------------------------------------------------------
# Input volume and label rasterization
# ---------------------------------------------------------------------------

@dataclass
class IndexedFileReport:
    paths_by_index: Dict[int, Path]
    bases_by_index: Dict[int, str]
    duplicate_indices: Dict[int, List[Path]]
    unindexed_paths: List[Path]
    indices: List[int]


@dataclass
class SourceVolume:
    input_dir: Path
    stem: str
    kind: str
    image_paths: List[Path]
    video_path: Optional[Path]
    labels_by_frame: Dict[int, Path]
    segmentation_nrrd_path: Optional[Path]
    mask_volume: Optional[np.ndarray]
    volume_class: str  # fully_labeled, partially_labeled, unlabeled
    label_source: str  # yolo, nrrd, none
    input_start_index: Optional[int]
    encoded_indices: Tuple[int, ...]
    volume: np.ndarray  # shape (t,Y,X), uint8
    fps: float
    # Parsed before image decode for label-first planning.  Presence with an
    # empty tuple means an explicit background label file.
    yolo_polygons_by_frame: Dict[int, Tuple[Tuple[Tuple[float, float], ...], ...]] = field(default_factory=dict)
    volume_block: Optional[SharedBlock] = None  # backing block when the process pool is active




def ffprobe_info(video_path: Path) -> Dict[str, object]:
    _require_bin("ffprobe")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-of", "json", str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    if not info.get("streams"):
        raise RuntimeError(f"No video stream found in {video_path}")
    st = info["streams"][0]

    def parse_ratio(s: str) -> float:
        if not s or s == "0/0":
            return 0.0
        n, d = s.split("/")
        return float(n) / float(d) if int(d) else 0.0

    width = int(st["width"])
    height = int(st["height"])
    fps = parse_ratio(str(st.get("avg_frame_rate", "0/0"))) or parse_ratio(str(st.get("r_frame_rate", "0/0"))) or 30.0
    nf = st.get("nb_frames")
    if nf is None or str(nf) in {"", "N/A"}:
        cmd2 = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets", "-of", "json", str(video_path),
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
        info2 = json.loads(proc2.stdout)
        nf = info2["streams"][0].get("nb_read_packets")
    if nf is None or str(nf) in {"", "N/A"}:
        raise RuntimeError(f"Could not determine frame count for {video_path}")
    return {"width": width, "height": height, "fps": fps, "num_frames": int(nf)}


def decode_video_gray8_to_memory(video_path: Path, *, warnings: WarningLog, allocator: Optional[ArrayAllocator] = None) -> Tuple[np.ndarray, float, Optional[SharedBlock]]:
    _require_bin("ffmpeg")
    info = ffprobe_info(video_path)
    w = int(info["width"])
    h = int(info["height"])
    t = int(info["num_frames"])
    fps = float(info["fps"])
    if allocator is not None:
        arr, block = allocator((t, h, w))
    else:
        arr = np.empty((t, h, w), dtype=np.uint8)
        block = None
    frame_bytes = int(w) * int(h)
    chunk_frames = max(1, min(128, (256 * 1024 * 1024) // max(1, frame_bytes)))
    raw = memoryview(arr).cast("B")
    cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        with tqdm(total=t, desc="Decoding video to in-memory gray8 volume") as pbar:
            for start in range(0, t, chunk_frames):
                nframes = min(chunk_frames, t - start)
                need = nframes * frame_bytes
                offset = start * frame_bytes
                view = raw[offset:offset + need]
                filled = 0
                while filled < need:
                    nread = proc.stdout.readinto(view[filled:])
                    if nread is None or nread <= 0:
                        raise RuntimeError(f"Unexpected EOF while decoding {video_path} near frame {start + 1}")
                    filled += int(nread)
                pbar.update(nframes)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        _, err = proc.communicate()
    if proc.returncode not in (0, None):
        msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
        raise RuntimeError(f"ffmpeg decode failed: {msg}")
    warnings.add("input_volume_loaded_in_memory", f"{arr.shape}, {arr.nbytes / GIB:.2f} GiB")
    return arr, fps, block


def analyze_indexed_files(paths: Sequence[Path], *, kind: str) -> IndexedFileReport:
    paths_by_index: Dict[int, Path] = {}
    bases_by_index: Dict[int, str] = {}
    duplicate_indices: Dict[int, List[Path]] = {}
    unindexed_paths: List[Path] = []
    buckets: Dict[int, List[Path]] = defaultdict(list)
    bases: Dict[int, str] = {}
    for path in paths:
        base, idx = split_stem_index(path)
        if idx is None:
            unindexed_paths.append(path)
            continue
        buckets[int(idx)].append(path)
        bases[int(idx)] = base
    for idx, bucket in buckets.items():
        if len(bucket) > 1:
            duplicate_indices[int(idx)] = list(bucket)
        else:
            paths_by_index[int(idx)] = bucket[0]
            bases_by_index[int(idx)] = bases[int(idx)]
    return IndexedFileReport(
        paths_by_index=paths_by_index,
        bases_by_index=bases_by_index,
        duplicate_indices=duplicate_indices,
        unindexed_paths=unindexed_paths,
        indices=sorted(int(x) for x in buckets.keys()),
    )


def contiguous_start_0_or_1(indices: Sequence[int]) -> Optional[int]:
    idxs = sorted(int(x) for x in indices)
    n = len(idxs)
    if n == 0:
        return None
    for start in (0, 1):
        if idxs == list(range(start, start + n)):
            return int(start)
    return None


def describe_index_problem(indices: Sequence[int]) -> str:
    idxs = sorted(int(x) for x in indices)
    if not idxs:
        return "no indexed files"
    if len(idxs) <= 24:
        return f"indices={idxs}"
    return f"count={len(idxs)}, first={idxs[:8]}, last={idxs[-8:]}"


YoloPolygon = Tuple[Tuple[float, float], ...]
YoloPolygons = Tuple[YoloPolygon, ...]


def read_yolo_polygons_normalized(path: Path, *, warnings: WarningLog) -> YoloPolygons:
    """Parse one YOLO segmentation file without allocating an image mask."""
    try:
        text = Path(path).read_text().strip()
    except Exception as exc:
        warnings.add("label_read_error", f"{path}: {exc}")
        return tuple()
    if not text:
        return tuple()
    polygons: List[YoloPolygon] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            warnings.add("invalid_label_line", f"{path.name}:{line_no}")
            continue
        try:
            coords = [float(value) for value in parts[1:]]
        except Exception:
            warnings.add("invalid_label_coordinate", f"{path.name}:{line_no}")
            continue
        points: List[Tuple[float, float]] = []
        for x_n, y_n in zip(coords[0::2], coords[1::2]):
            if not math.isfinite(x_n) or not math.isfinite(y_n):
                points = []
                break
            points.append((min(1.0, max(0.0, float(x_n))), min(1.0, max(0.0, float(y_n)))))
        if len(points) < 3 or len(set(points)) < 3 or _polygon_area_xy(points) <= 1e-12:
            warnings.add("degenerate_input_polygon", f"{path.name}:{line_no}")
            continue
        polygons.append(tuple(points))
    return tuple(polygons)


def validate_no_unindexed_or_duplicate(report: IndexedFileReport, *, kind: str) -> None:
    if report.unindexed_paths:
        examples = ", ".join(p.name for p in report.unindexed_paths[:8])
        raise ValueError(f"{kind} files must contain a numeric frame index in the filename; unindexed examples: {examples}")
    if report.duplicate_indices:
        examples = ", ".join(f"{idx}:" + "/".join(p.name for p in paths[:3]) for idx, paths in list(report.duplicate_indices.items())[:8])
        raise ValueError(f"Duplicate {kind} frame indices are invalid: {examples}")


def _read_gray_image_opencv(path: Path) -> np.ndarray:
    # Request luminance/gray at decode time.  This avoids materializing the
    # source RGB array and the subsequent cvtColor pass for the target command.
    img = cv2.imread(str(path), int(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return to_gray8(img)


def _nvimgcodec_decoder_for_device(nvimgcodec: object, *, device_id: int, cpu_threads: int, batch_size: int) -> object:
    backends: List[object] = []
    backend_type = getattr(nvimgcodec, "Backend", None)
    backend_kind = getattr(nvimgcodec, "BackendKind", None)
    if backend_type is not None and backend_kind is not None:
        for name in ("HW_GPU_ONLY", "GPU_ONLY", "HYBRID_CPU_GPU"):
            kind = getattr(backend_kind, name, None)
            if kind is not None:
                try:
                    backends.append(backend_type(kind))
                except TypeError:
                    try:
                        backends.append(backend_type(kind=kind))
                    except Exception:
                        pass
    options = (
        f":num_cuda_streams=4 "
        f"nvjpeg_cuda_decoder:hybrid_huffman_threshold=0 "
        f"nvjpeg_cuda_decoder:preallocate_buffers=1 "
        f"nvjpeg_hw_decoder:preallocate_batch_size={max(1, int(batch_size))}"
    )
    kwargs: Dict[str, object] = {
        "device_id": int(device_id),
        "max_num_cpu_threads": max(1, int(cpu_threads)),
        "options": options,
    }
    if backends:
        kwargs["backends"] = backends
    return getattr(nvimgcodec, "Decoder")(**kwargs)


def _nvimgcodec_gray_params(nvimgcodec: object) -> object:
    params_type = getattr(nvimgcodec, "DecodeParams")
    color_spec_type = getattr(nvimgcodec, "ColorSpec", None)
    gray = getattr(color_spec_type, "GRAY", None) if color_spec_type is not None else None
    if gray is not None:
        return params_type(color_spec=gray, allow_any_depth=False)
    return params_type(allow_any_depth=False)


def _decoded_nvimgcodec_batch_to_arrays(decoded: object, expected: int) -> List[np.ndarray]:
    if isinstance(decoded, (list, tuple)):
        items = list(decoded)
    else:
        try:
            items = list(decoded)  # nvImageCodec batch objects are iterable
        except TypeError:
            host = decoded.cpu() if callable(getattr(decoded, "cpu", None)) else decoded
            batch = np.asarray(host)
            if batch.ndim >= 3 and int(batch.shape[0]) == int(expected):
                items = [batch[i] for i in range(int(expected))]
            else:
                items = [host]
    arrays: List[np.ndarray] = []
    for item in items:
        host = item.cpu() if callable(getattr(item, "cpu", None)) else item
        arr = np.asarray(host)
        if arr.ndim == 3 and int(arr.shape[-1]) == 1:
            arr = arr[:, :, 0]
        arrays.append(to_gray8(arr))
    if len(arrays) != int(expected):
        raise RuntimeError(f"nvImageCodec returned {len(arrays)} image(s) for a batch of {expected}")
    return arrays


def _decode_jpeg_sequence_nvimgcodec(
    images: Sequence[Path],
    volume: np.ndarray,
    *,
    device_ids: Sequence[int],
    cpu_sets: Sequence[Sequence[int]],
    batch_size: int,
    workers: int,
) -> None:
    try:
        from nvidia import nvimgcodec  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "nvJPEG input decoding requires NVIDIA nvImageCodec; install "
            "nvidia-nvimgcodec-cu12[all] or nvidia-nvimgcodec-cu13[all] "
            "to match the active CUDA major version"
        ) from exc
    if not device_ids:
        raise RuntimeError("nvJPEG input decoding requested but no CUDA-visible GPU was discovered")
    batch = max(1, int(batch_size))
    batches = [list(range(start, min(len(images), start + batch))) for start in range(0, len(images), batch)]
    per_device: List[List[List[int]]] = [[] for _ in device_ids]
    for batch_index, indices in enumerate(batches):
        per_device[batch_index % len(device_ids)].append(indices)

    def _decode_device(slot: int) -> None:
        cpus = tuple(int(x) for x in (cpu_sets[slot] if slot < len(cpu_sets) else ()))
        bind_current_thread_to_cpus(cpus)
        cpu_threads = max(1, min(len(cpus) if cpus else int(workers), 8))
        decoder = _nvimgcodec_decoder_for_device(
            nvimgcodec,
            device_id=int(device_ids[slot]),
            cpu_threads=cpu_threads,
            batch_size=batch,
        )
        params = _nvimgcodec_gray_params(nvimgcodec)
        for indices in per_device[slot]:
            paths = [str(images[i]) for i in indices]
            decoded = decoder.read(paths, params=params)
            arrays = _decoded_nvimgcodec_batch_to_arrays(decoded, len(indices))
            for frame_index, gray in zip(indices, arrays):
                if gray.shape != tuple(int(x) for x in volume.shape[1:]):
                    raise ValueError(
                        f"Image dimension mismatch: {images[frame_index]} has {gray.shape}, "
                        f"expected {tuple(volume.shape[1:])}"
                    )
                volume[frame_index] = gray

    with ThreadPoolExecutor(max_workers=len(device_ids), thread_name_prefix="nvjpeg-decode") as executor:
        futures = [executor.submit(_decode_device, slot) for slot in range(len(device_ids)) if per_device[slot]]
        for future in futures:
            future.result()


def load_image_sequence_to_memory(
    images: Sequence[Path],
    *,
    warnings: WarningLog,
    workers: int,
    allocator: Optional[ArrayAllocator] = None,
    jpeg_decode_backend: str = "auto",
    jpeg_batch_size: int = 64,
    jpeg_device_ids: Sequence[int] = (),
    jpeg_cpu_sets: Sequence[Sequence[int]] = (),
    load_cpu_order: Sequence[int] = (),
) -> Tuple[np.ndarray, Optional[SharedBlock]]:
    if not images:
        raise ValueError("No image sequence frames supplied")
    first_gray = _read_gray_image_opencv(images[0])
    h, w = first_gray.shape
    t = len(images)
    if allocator is not None:
        volume, block = allocator((t, h, w))
    else:
        volume = np.empty((t, h, w), dtype=np.uint8)
        block = None
    volume[0] = first_gray

    backend = str(jpeg_decode_backend).strip().lower()
    all_jpeg = all(path.suffix.lower() in {".jpg", ".jpeg"} for path in images)
    if backend == "nvjpeg" and not all_jpeg:
        raise ValueError("--jpeg_decode_backend nvjpeg requires every sequence frame to be .jpg/.jpeg")
    try_nvjpeg = all_jpeg and backend in {"auto", "nvjpeg"} and bool(jpeg_device_ids)
    if try_nvjpeg:
        try:
            _decode_jpeg_sequence_nvimgcodec(
                images,
                volume,
                device_ids=jpeg_device_ids,
                cpu_sets=jpeg_cpu_sets,
                batch_size=int(jpeg_batch_size),
                workers=max(1, int(workers)),
            )
            warnings.add(
                "jpeg_sequence_decoded_with_nvjpeg",
                f"frames={len(images)}, GPUs={list(jpeg_device_ids)}, batch_size={int(jpeg_batch_size)}",
            )
            warnings.add("input_volume_loaded_in_memory", f"{volume.shape}, {volume.nbytes / GIB:.2f} GiB")
            return volume, block
        except Exception as exc:
            if backend == "nvjpeg":
                raise
            warnings.add("nvjpeg_fallback_to_opencv", f"{type(exc).__name__}: {exc}")
    elif backend == "nvjpeg":
        raise RuntimeError("--jpeg_decode_backend nvjpeg requires at least one discovered CUDA-visible GPU")

    def _read_image(i: int) -> None:
        if i == 0:
            return
        gray = _read_gray_image_opencv(images[i])
        if gray.shape != (h, w):
            raise ValueError(f"Image dimension mismatch: {images[i]} has {gray.shape}, expected {(h, w)}")
        volume[i] = gray

    parallel_for_indices(
        t,
        _read_image,
        workers=workers,
        desc="Reading image sequence into in-memory gray8 volume",
        worker_cpu_order=load_cpu_order,
    )
    warnings.add("image_sequence_decoded_direct_grayscale", f"frames={len(images)}, backend=opencv")
    warnings.add("input_volume_loaded_in_memory", f"{volume.shape}, {volume.nbytes / GIB:.2f} GiB")
    return volume, block


def nrrd_header_to_internal_permutation(header: Mapping[str, object]) -> Optional[Tuple[int, int, int]]:
    """Infer the array-axis permutation required for internal ``(t,Y,X)``.

    NRRD ``space directions`` contains one physical direction vector per data
    axis.  For an axis-aligned spatial volume, map the data axes representing
    physical Z, Y, and X into the pipeline's t/Y/X order.  This disambiguates
    cubic volumes, for which shape alone cannot distinguish XYZ from TYX.
    """
    raw_directions = header.get("space directions")
    if raw_directions is not None:
        try:
            directions = np.asarray(raw_directions, dtype=np.float64)
            if directions.shape == (3, 3) and np.all(np.isfinite(directions)):
                magnitudes = np.abs(directions)
                dominant = tuple(int(x) for x in np.argmax(magnitudes, axis=1).tolist())
                dominant_values = magnitudes[np.arange(3), np.asarray(dominant, dtype=np.int64)]
                if set(dominant) == {0, 1, 2} and np.all(dominant_values > 0.0):
                    # Physical coordinates are X=0, Y=1, Z/t=2.
                    return (
                        int(dominant.index(2)),
                        int(dominant.index(1)),
                        int(dominant.index(0)),
                    )
        except Exception:
            pass

    # Recognize snapshots written by this script even if a downstream tool
    # removes the space-directions field while preserving content metadata.
    content = str(header.get("content", ""))
    if "exported_axes=(X,Y,t)" in content:
        return (2, 1, 0)
    return None


def load_nrrd_mask_for_volume(nrrd_path: Path, expected_shape_tyx: Tuple[int, int, int], *, warnings: WarningLog) -> np.ndarray:
    try:
        import nrrd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pynrrd is required for NRRD segmentation input: pip install pynrrd") from exc
    data, header = nrrd.read(str(nrrd_path))
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"NRRD segmentation must be 3D, got shape {arr.shape} from {nrrd_path}")
    t_dim, h, w = (int(expected_shape_tyx[0]), int(expected_shape_tyx[1]), int(expected_shape_tyx[2]))
    shape = tuple(int(x) for x in arr.shape)
    candidates: List[Tuple[str, Tuple[int, int, int]]] = []
    for name, source_shape, permutation in (
        ("already_tyx", (t_dim, h, w), (0, 1, 2)),
        ("xyz_to_tyx", (w, h, t_dim), (2, 1, 0)),
        ("yxt_to_tyx", (h, w, t_dim), (2, 0, 1)),
        ("txy_to_tyx", (t_dim, w, h), (0, 2, 1)),
    ):
        if shape == source_shape:
            candidates.append((name, permutation))
    if not candidates:
        raise ValueError(
            f"NRRD segmentation shape {shape} does not match image volume shape (t,Y,X)=({t_dim},{h},{w}) "
            f"or recognized spatial permutations (X,Y,t)=({w},{h},{t_dim}) / (Y,X,t)=({h},{w},{t_dim})"
        )

    header_permutation = nrrd_header_to_internal_permutation(header)
    if header_permutation is not None:
        internal_shape = tuple(shape[int(axis)] for axis in header_permutation)
        if internal_shape != (t_dim, h, w):
            raise ValueError(
                f"NRRD header axis directions imply internal shape {internal_shape}, but image volume shape is "
                f"(t,Y,X)=({t_dim},{h},{w}) for {nrrd_path}"
            )
        permutation = tuple(int(x) for x in header_permutation)
        orientation = "header_axes_" + "".join(str(x) for x in permutation) + "_to_tyx"
    else:
        unique_permutations = {tuple(permutation) for _, permutation in candidates}
        if len(unique_permutations) != 1:
            matches = ", ".join(name for name, _ in candidates)
            raise ValueError(
                f"NRRD axis order is ambiguous from shape {shape} ({matches}) and the header has no usable "
                f"space directions/content marker: {nrrd_path}. Add axis-aware NRRD metadata before import."
            )
        orientation, permutation = candidates[0]

    internal = arr if permutation == (0, 1, 2) else np.transpose(arr, permutation)
    mask = np.ascontiguousarray((np.asarray(internal) > 0).astype(np.uint8))
    warnings.add("segmentation_nrrd_loaded_in_memory", f"{nrrd_path.name}: {mask.shape}, {mask.nbytes / GIB:.2f} GiB, orientation={orientation}")
    return mask


def most_common_stem(paths: Sequence[Path]) -> str:
    if not paths:
        return "Volume"
    bases = [split_stem_index(p)[0] for p in paths]
    return Counter(bases).most_common(1)[0][0] or paths[0].stem


def classify_sequence_input(
    *,
    image_report: IndexedFileReport,
    label_report: Optional[IndexedFileReport],
    has_nrrd: bool,
    num_frames: int,
    warnings: WarningLog,
) -> Tuple[str, Optional[int]]:
    del num_frames
    image_start = contiguous_start_0_or_1(image_report.indices)
    if label_report is None and not has_nrrd:
        if image_start is None:
            warnings.add("unlabeled_sequence_noncontiguous_indices", describe_index_problem(image_report.indices))
        return "unlabeled", image_start
    if label_report is not None:
        image_set = set(int(x) for x in image_report.indices)
        label_set = set(int(x) for x in label_report.indices)
        orphan_labels = sorted(label_set - image_set)
        if orphan_labels:
            raise ValueError(
                "Every YOLO label index must have a matching image index; "
                f"orphan label indices={orphan_labels[:24]}, "
                f"image {describe_index_problem(image_report.indices)}, "
                f"label {describe_index_problem(label_report.indices)}"
            )
        if image_start is not None and label_set == image_set:
            return "fully_labeled", image_start
        missing_labels = sorted(image_set - label_set)
        reasons: List[str] = []
        if image_start is None:
            reasons.append("image indices are not contiguous from 0 or 1")
        if missing_labels:
            reasons.append(f"{len(missing_labels)} image slice(s) have no label file")
        warnings.add("partial_volume_detected", "; ".join(reasons) or "partial YOLO label coverage")
        return "partially_labeled", image_start
    if has_nrrd:
        if image_start is not None:
            return "fully_labeled", image_start
        warnings.add("partial_volume_detected", "NRRD segmentation exists, but image indices are not contiguous from 0 or 1")
        return "partially_labeled", image_start
    raise AssertionError("unreachable")


def labels_by_frame_from_matching_indices(image_indices: Sequence[int], labels_by_index: Dict[int, Path]) -> Dict[int, Path]:
    """Map only image positions that have an explicit YOLO label file."""
    return {
        frame_i0: labels_by_index[int(encoded_idx)]
        for frame_i0, encoded_idx in enumerate(sorted(int(x) for x in image_indices))
        if int(encoded_idx) in labels_by_index
    }


def labels_by_frame_from_video_indices(label_report: IndexedFileReport, frame_count: int) -> Tuple[Dict[int, Path], int]:
    start = contiguous_start_0_or_1(label_report.indices)
    if start is None:
        raise ValueError(f"Video label indices must be contiguous and start at 0 or 1; {describe_index_problem(label_report.indices)}")
    expected = list(range(start, start + int(frame_count)))
    got = sorted(int(x) for x in label_report.indices)
    if got != expected:
        raise ValueError(
            f"Video labels must map exactly to all video frame indices. Expected {expected[:3]}...{expected[-3:]} "
            f"for {frame_count} frames with start {start}; got {describe_index_problem(got)}"
        )
    return {int(idx) - int(start): label_report.paths_by_index[int(idx)] for idx in got}, int(start)


def collect_input(input_arg: str, *, warnings: WarningLog, workers: int) -> SourceVolume:
    input_path = Path(input_arg).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not input_path.is_dir():
        raise ValueError("--input must be a directory containing an accepted v3.0.0_SLURM input")

    files = [p for p in input_path.iterdir() if p.is_file()]
    videos = sorted([p for p in files if p.suffix.lower() in VIDEO_EXTS])
    images_raw = sorted([p for p in files if p.suffix.lower() in IMAGE_EXTS], key=natural_index_key)
    labels = sorted([p for p in files if p.suffix.lower() == ".txt"], key=natural_index_key)
    nrrds = sorted([p for p in files if p.suffix.lower() in NRRD_EXTS])

    if videos and images_raw:
        raise ValueError("Input directory must contain either one video or one image sequence, not both")
    if len(videos) > 1:
        raise ValueError(f"Expected at most one video file, found {len(videos)}")
    if len(nrrds) > 1:
        raise ValueError(f"Expected at most one NRRD segmentation file, found {len(nrrds)}")
    if labels and nrrds:
        raise ValueError("Provide either YOLO txt labels or one NRRD segmentation, not both")
    if not videos and not images_raw:
        raise ValueError("No video or image sequence found in input directory")

    label_report: Optional[IndexedFileReport] = None
    if labels:
        label_report = analyze_indexed_files(labels, kind="label")
        validate_no_unindexed_or_duplicate(label_report, kind="label")

    if videos:
        video_path = videos[0]
        stem = video_path.stem
        volume, fps, _volume_block = decode_video_gray8_to_memory(video_path, warnings=warnings)
        frame_count = int(volume.shape[0])
        labels_by_frame: Dict[int, Path] = {}
        mask_volume: Optional[np.ndarray] = None
        input_start_index: Optional[int] = None
        if label_report is not None:
            if len(label_report.indices) != frame_count:
                raise ValueError(f"Video Fully Labeled Volume requires label count to equal frame count; frames={frame_count}, labels={len(label_report.indices)}")
            labels_by_frame, input_start_index = labels_by_frame_from_video_indices(label_report, frame_count)
            volume_class = "fully_labeled"
            label_source = "yolo"
        elif nrrds:
            mask_volume = load_nrrd_mask_for_volume(nrrds[0], tuple(volume.shape), warnings=warnings)
            volume_class = "fully_labeled"
            label_source = "nrrd"
        else:
            volume_class = "unlabeled"
            label_source = "none"
        return SourceVolume(
            input_dir=input_path,
            stem=stem,
            kind="video",
            image_paths=[],
            video_path=video_path,
            labels_by_frame=labels_by_frame,
            segmentation_nrrd_path=nrrds[0] if nrrds else None,
            mask_volume=mask_volume,
            volume_class=volume_class,
            label_source=label_source,
            input_start_index=input_start_index,
            encoded_indices=tuple(range(int(input_start_index or 0), int(input_start_index or 0) + frame_count)) if label_report is not None else tuple(range(frame_count)),
            volume=volume,
            fps=fps,
        )

    image_report = analyze_indexed_files(images_raw, kind="image")
    validate_no_unindexed_or_duplicate(image_report, kind="image")
    if not image_report.indices:
        raise ValueError("Image sequence files must contain numeric frame indices")
    ordered_images = [image_report.paths_by_index[int(idx)] for idx in sorted(image_report.indices)]
    stem = most_common_stem(ordered_images)
    volume, _volume_block = load_image_sequence_to_memory(ordered_images, warnings=warnings, workers=workers)
    volume_class, input_start_index = classify_sequence_input(
        image_report=image_report,
        label_report=label_report,
        has_nrrd=bool(nrrds),
        num_frames=int(volume.shape[0]),
        warnings=warnings,
    )
    labels_by_frame: Dict[int, Path] = {}
    mask_volume: Optional[np.ndarray] = None
    label_source = "none"
    if label_report is not None:
        labels_by_frame = labels_by_frame_from_matching_indices(image_report.indices, label_report.paths_by_index)
        label_source = "yolo"
    elif nrrds:
        mask_volume = load_nrrd_mask_for_volume(nrrds[0], tuple(volume.shape), warnings=warnings)
        label_source = "nrrd"

    return SourceVolume(
        input_dir=input_path,
        stem=stem,
        kind="sequence",
        image_paths=ordered_images,
        video_path=None,
        labels_by_frame=labels_by_frame,
        segmentation_nrrd_path=nrrds[0] if nrrds else None,
        mask_volume=mask_volume,
        volume_class=volume_class,
        label_source=label_source,
        input_start_index=input_start_index,
        encoded_indices=tuple(sorted(int(x) for x in image_report.indices)),
        volume=volume,
        fps=1.0,
    )


def rasterize_yolo_labels(src: SourceVolume, *, warnings: WarningLog, workers: int, allocator: Optional[ArrayAllocator] = None) -> Tuple[np.ndarray, Optional[SharedBlock]]:
    volume = src.volume
    t, h, w = volume.shape
    if allocator is not None:
        mask, mask_block = allocator((t, h, w))  # shared pages arrive zero-filled
    else:
        mask = np.zeros((t, h, w), dtype=np.uint8)
        mask_block = None
    if src.label_source != "yolo":
        raise ValueError("rasterize_yolo_labels requires a SourceVolume with YOLO txt labels")

    def _rasterize_frame(frame_i0: int) -> None:
        label_path = src.labels_by_frame.get(int(frame_i0))
        frame_1 = int(frame_i0) + 1
        if label_path is None:
            # An absent label file is an intentional unannotated state
            # for partially labeled image sequences, not an empty/background label.
            return
        if not label_path.exists():
            warnings.add("missing_label", f"frame {frame_1:04d}: expected {label_path}")
            return
        polygons = src.yolo_polygons_by_frame.get(int(frame_i0))
        if polygons is None:
            # Support SourceVolume objects created by direct helper APIs rather
            # than discover_volume_specs().
            polygons = read_yolo_polygons_normalized(label_path, warnings=warnings)
        if not polygons:
            return
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        for polygon in polygons:
            pts: List[List[int]] = []
            for x_n, y_n in polygon:
                # YOLO coordinates are normalized by image width/height.  This
                # is the inverse of mask_to_yolo_lines() below; multiplying by
                # (w-1)/(h-1) caused a systematic one-pixel inward drift.
                x = int(round(float(x_n) * float(w)))
                y = int(round(float(y_n) * float(h)))
                pts.append([max(0, min(w - 1, x)), max(0, min(h - 1, y))])
            if len(pts) < 3:
                continue
            poly = np.asarray(pts, dtype=np.int32)
            cv2.fillPoly(frame_mask, [poly], 1)
        mask[frame_i0] = frame_mask

    parallel_for_indices(t, _rasterize_frame, workers=workers, desc="Rasterizing YOLO labels into in-memory mask volume")
    warnings.add("mask_volume_loaded_in_memory", f"{mask.shape}, {mask.nbytes / GIB:.2f} GiB")
    return mask, mask_block


def derive_annotation_states(src: SourceVolume, mask_u8: np.ndarray) -> Tuple[int, ...]:
    """Return one explicit three-state annotation value per loaded image slice.

    A YOLO file that exists but rasterizes empty is annotated background.  A
    YOLO file with a nonempty rasterized mask is annotated foreground.  An image
    with no YOLO file is unannotated and may be used only as image-channel
    context in an unforced partial volume.  NRRD masks annotate every frame.
    """
    mask = np.asarray(mask_u8, dtype=np.uint8)
    if mask.ndim != 3 or int(mask.shape[0]) != int(src.volume.shape[0]):
        raise ValueError(
            f"Annotation-state derivation requires matching 3D image/mask frames: "
            f"image={src.volume.shape}, mask={mask.shape}"
        )
    states = np.full((int(mask.shape[0]),), ANNOTATION_UNANNOTATED, dtype=np.uint8)
    if src.label_source == "nrrd":
        states[:] = ANNOTATION_BACKGROUND
        states[np.any(mask > 0, axis=(1, 2))] = ANNOTATION_FOREGROUND
    elif src.label_source == "yolo":
        for frame_i0, label_path in src.labels_by_frame.items():
            idx = int(frame_i0)
            if idx < 0 or idx >= int(mask.shape[0]) or not Path(label_path).exists():
                continue
            polygons = src.yolo_polygons_by_frame.get(idx)
            is_foreground = bool(polygons) if polygons is not None else bool(np.any(mask[idx] > 0))
            states[idx] = ANNOTATION_FOREGROUND if is_foreground else ANNOTATION_BACKGROUND
    return tuple(int(x) for x in states.tolist())


def annotation_state_counts(states: Sequence[int]) -> Dict[str, int]:
    counts = Counter(int(x) for x in states)
    return {
        "annotated_foreground": int(counts.get(ANNOTATION_FOREGROUND, 0)),
        "annotated_background": int(counts.get(ANNOTATION_BACKGROUND, 0)),
        "unannotated": int(counts.get(ANNOTATION_UNANNOTATED, 0)),
    }


ForegroundAnchor = Tuple[int, int, int]  # source (t, y, x)


def collect_foreground_slice_anchors(
    src: SourceVolume,
    mask_u8: np.ndarray,
    annotation_states: Sequence[int],
    *,
    warnings: WarningLog,
) -> Tuple[Tuple[ForegroundAnchor, ...], int]:
    """Keep one compact mask anchor for every foreground input slice."""
    mask = np.asarray(mask_u8, dtype=np.uint8)
    anchors: List[ForegroundAnchor] = []
    seeded = 0
    for frame_idx, state in enumerate(annotation_states):
        if int(state) != ANNOTATION_FOREGROUND:
            continue
        frame = mask[int(frame_idx)]
        flat_index = int(np.argmax(frame)) if frame.size else 0
        if frame.size and int(frame.reshape(-1)[flat_index]) > 0:
            y, x = divmod(flat_index, int(frame.shape[1]))
        else:
            polygons = src.yolo_polygons_by_frame.get(int(frame_idx), tuple())
            if not polygons:
                raise RuntimeError(
                    f"Foreground input slice {frame_idx + 1} in {src.stem} has no mask pixel or polygon anchor"
                )
            x_n, y_n = polygons[0][0]
            x = max(0, min(int(frame.shape[1]) - 1, int(round(float(x_n) * float(frame.shape[1])))))
            y = max(0, min(int(frame.shape[0]) - 1, int(round(float(y_n) * float(frame.shape[0])))))
            frame[int(y), int(x)] = 1
            seeded += 1
        anchors.append((int(frame_idx), int(y), int(x)))
    if seeded:
        warnings.add(
            "foreground_anchor_seeded_from_polygon",
            f"{src.stem}: seeded {seeded} foreground slice anchor(s) whose rasterized polygon was empty",
        )
    return tuple(anchors), int(seeded)


def _map_anchor_axis(index: int, source_length: int, output_length: int) -> int:
    if int(source_length) <= 1 or int(output_length) <= 1:
        return 0
    mapped = int(round(float(index) * float(output_length - 1) / float(source_length - 1)))
    return max(0, min(int(output_length) - 1, mapped))


def reassert_foreground_slice_anchors(
    mask_u8: np.ndarray,
    anchors: Sequence[ForegroundAnchor],
    *,
    source_shape: Tuple[int, int, int],
    patch_radius: int,
) -> Tuple[int, int]:
    """Map one anchor per input foreground slice into a transformed mask.

    The depth mapping is injective whenever output depth is at least source
    depth.  Reasserting an anchor only when its small mapped neighborhood is
    empty prevents Gaussian thresholding or interpolation from deleting an
    input foreground slice, without retaining any background candidate.
    """
    mask = np.asarray(mask_u8, dtype=np.uint8)
    if mask.ndim != 3:
        raise ValueError(f"Foreground anchor preservation requires a 3D mask, got {mask.shape}")
    source_t, source_h, source_w = (int(value) for value in source_shape)
    output_t, output_h, output_w = (int(value) for value in mask.shape)
    mapped_depths: set[int] = set()
    repaired = 0
    radius = max(0, int(patch_radius))
    for source_z, source_y, source_x in anchors:
        z = _map_anchor_axis(int(source_z), source_t, output_t)
        y = _map_anchor_axis(int(source_y), source_h, output_h)
        x = _map_anchor_axis(int(source_x), source_w, output_w)
        mapped_depths.add(int(z))
        y0, y1 = max(0, y - radius), min(output_h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(output_w, x + radius + 1)
        region = mask[z, y0:y1, x0:x1]
        if not np.all(region > 0):
            region[...] = 1
            repaired += 1
    if len(mapped_depths) < len(anchors):
        raise RuntimeError(
            "Foreground transverse preservation failed: multiple foreground input slices "
            f"collapsed onto one output slice (input={len(anchors)}, output={len(mapped_depths)})"
        )
    return int(len(mapped_depths)), int(repaired)

# ---------------------------------------------------------------------------
# NRRD and overlay/video helpers
# ---------------------------------------------------------------------------

def nrrd_mask_to_slicer_xyz(mask_u8: np.ndarray) -> np.ndarray:
    """Return a Slicer-oriented NRRD payload from the internal mask layout.

    The pipeline keeps volumes as ``(t, Y, X)`` / ``(Z, row, column)`` so frame
    operations and overlays match the source Transverse image sequence.  3D
    Slicer expects a spatial volume whose axes are ordered as ``(X, Y, Z)`` in
    the NRRD payload.  Writing the internal array directly makes Slicer treat
    frame number as the first spatial axis, which presents as rotated/flipped
    Transverse slices plus swapped Sagittal/Coronal slice families.
    """
    arr = np.asarray(mask_u8, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"NRRD export expects a 3D mask volume with shape (t,Y,X), got {arr.shape}")
    # Internal: (t, Y, X).  NRRD/Slicer spatial payload: (X, Y, t).
    # Use Fortran-contiguous storage and explicitly request pynrrd's Fortran
    # index order so the first NumPy dimension remains the first NRRD axis.
    return np.asfortranarray(np.transpose(arr, (2, 1, 0)))


def nrrd_slicer_header(mask_shape_zyx: Tuple[int, int, int]) -> Dict[str, object]:
    t_dim, h, w = (int(mask_shape_zyx[0]), int(mask_shape_zyx[1]), int(mask_shape_zyx[2]))
    return {
        "space": NRRD_SPACE,
        "kinds": ["domain", "domain", "domain"],
        "space directions": np.eye(3, dtype=np.float64),
        "space origin": np.zeros((3,), dtype=np.float64),
        "content": f"binary segmentation mask; source_shape_tyx=({t_dim},{h},{w}); exported_axes=(X,Y,t)",
    }


def write_nrrd(mask_u8: np.ndarray, path: Path) -> Path:
    try:
        import inspect
        import nrrd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pynrrd is required for --save_nrrd: pip install pynrrd") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = nrrd_mask_to_slicer_xyz(mask_u8)
    header = nrrd_slicer_header(tuple(np.asarray(mask_u8).shape))
    kwargs: Dict[str, object] = {"header": header}
    try:
        if "index_order" in inspect.signature(nrrd.write).parameters:
            kwargs["index_order"] = "F"
    except Exception:
        pass
    nrrd.write(str(path), payload, **kwargs)
    return path


def ffmpeg_ffv1_rgb_writer(out_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    _require_bin("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{int(width)}x{int(height)}", "-r", f"{float(fps)}",
        "-i", "-", "-an", "-c:v", "ffv1", "-level", "3", "-slices", "30", "-threads", "1",
        "-pix_fmt", "yuv444p", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    return proc


def close_ffmpeg_writer(proc: subprocess.Popen) -> None:
    if proc.stdin is not None and not proc.stdin.closed:
        try:
            proc.stdin.close()
        except Exception:
            pass
    proc.stdin = None  # type: ignore[attr-defined]
    _, err = proc.communicate()
    if proc.returncode not in (0, None):
        msg = err.decode("utf-8", errors="ignore") if isinstance(err, (bytes, bytearray)) else str(err)
        raise RuntimeError(f"ffmpeg writer failed: {msg}")


def overlay_rgb(frame_gray: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Create an RGB diagnostic overlay from the center image channel."""
    image = np.asarray(frame_gray, dtype=np.uint8)
    if image.ndim == 2:
        g = image
    elif image.ndim == 3 and int(image.shape[2]) >= 1:
        g = image[:, :, int(image.shape[2]) // 2]
    else:
        raise ValueError(f"Overlay image must be HxW or HxWxC, got shape={image.shape}")
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    m = np.asarray(mask_u8, dtype=bool)
    if np.any(m):
        blue = np.array([0, 0, 255], dtype=np.uint16)
        rgb[m] = ((rgb[m].astype(np.uint16) + blue) // 2).astype(np.uint8)
    return np.ascontiguousarray(rgb)


# ---------------------------------------------------------------------------
# Gaussian smoothing and cubic volume preprocessing
# ---------------------------------------------------------------------------

def apply_gaussian_smoothing(
    mask_u8: np.ndarray,
    *,
    sigma: float,
    passes: int,
    warnings: WarningLog,
    after_pass: Optional[Callable[[int, np.ndarray], None]] = None,
) -> List[Dict[str, int | float]]:
    """Apply v3.0.0 Gaussian smoothing to the in-memory binary 3D mask.

    The mask remains binary after every pass by thresholding the smoothed probability
    volume at 0.5.  The operation is intentionally whole-volume because the v3.0.0
    spec requires the full image and mask tensors to be resident before reslicing.
    """
    stats: List[Dict[str, int | float]] = []
    sigma_f = float(sigma)
    pass_count = int(passes)
    if sigma_f <= 0.0 or pass_count <= 0:
        warnings.add("gaussian_smoothing_disabled", f"sigma={sigma_f:g}, passes={pass_count}")
        return stats
    for pass_idx in range(1, pass_count + 1):
        before = np.asarray(mask_u8, dtype=np.uint8)
        before_count = int(np.count_nonzero(before))
        print(f"Gaussian smoothing pass {pass_idx}/{pass_count}: sigma={sigma_f:g} voxels")
        # v18 deliberately uses the TTA numerical primitive: isotropic voxel
        # sigma, constant-zero boundary, truncate=4, and threshold after every
        # pass so the following pass starts from the previous binary result.
        after = binary_gaussian_pass(
            before,
            sigma=sigma_f,
            gaussian_filter=ndi.gaussian_filter,
        )
        after_count = int(np.count_nonzero(after))
        mask_u8[:] = after
        stats.append({
            "pass_index": int(pass_idx),
            "sigma": float(sigma_f),
            "foreground_before": int(before_count),
            "foreground_after": int(after_count),
            "delta_voxels": int(after_count - before_count),
        })
        if after_pass is not None:
            after_pass(int(pass_idx), mask_u8)
    return stats


def cubic_target_shape(shape_tyx: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return target (t,Y,X) dimensions whose smaller axes are within 5% of the longest."""
    t_dim, h, w = (int(shape_tyx[0]), int(shape_tyx[1]), int(shape_tyx[2]))
    longest = max(t_dim, h, w)
    min_allowed = int(math.ceil(float(longest) * 0.95))
    return tuple(max(int(dim), min_allowed) for dim in (t_dim, h, w))  # type: ignore[return-value]


def resize_image_volume_to_shape(volume_u8: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    src_shape = tuple(int(x) for x in volume_u8.shape)
    target = tuple(int(x) for x in target_shape)
    if src_shape == target:
        return np.ascontiguousarray(volume_u8, dtype=np.uint8)
    zoom = tuple(float(t) / float(s) for t, s in zip(target, src_shape))
    print(f"Cubic resizing image volume: {src_shape} -> {target} (zoom={tuple(round(z, 6) for z in zoom)})")
    resized = ndi.zoom(volume_u8.astype(np.float32, copy=False), zoom=zoom, order=1, mode="nearest", prefilter=False)
    return np.ascontiguousarray(np.clip(np.rint(resized), 0, 255).astype(np.uint8))


def resize_mask_volume_to_shape(mask_u8: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    src_shape = tuple(int(x) for x in mask_u8.shape)
    target = tuple(int(x) for x in target_shape)
    if src_shape == target:
        return np.ascontiguousarray(mask_u8, dtype=np.uint8)
    zoom = tuple(float(t) / float(s) for t, s in zip(target, src_shape))
    print(f"Cubic resizing mask volume: {src_shape} -> {target} (zoom={tuple(round(z, 6) for z in zoom)})")
    resized = ndi.zoom(mask_u8.astype(np.float32, copy=False), zoom=zoom, order=1, mode="nearest", prefilter=False)
    return np.ascontiguousarray((resized >= 0.5).astype(np.uint8))


def resize_to_approximately_cube(
    volume_u8: np.ndarray,
    mask_u8: np.ndarray,
    *,
    warnings: WarningLog,
    work_dir: Optional[Path] = None,
    workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    source_shape = tuple(int(x) for x in volume_u8.shape)
    target = shared_media.compute_cube_resize_shape(*source_shape)
    if source_shape == target:
        warnings.add("cubic_resize_skipped", f"source_shape={source_shape} already within 5% cube tolerance")
        return np.ascontiguousarray(volume_u8, dtype=np.uint8), np.ascontiguousarray(mask_u8, dtype=np.uint8), target
    if tuple(int(x) for x in mask_u8.shape) != source_shape:
        raise ValueError(f"Image and mask volumes must have identical shape before cubic resize: image={source_shape}, mask={mask_u8.shape}")
    resolved_work_dir = Path(work_dir) if work_dir is not None else Path.cwd() / ".v18_work"
    resolved_work_dir.mkdir(parents=True, exist_ok=True)
    resized_volume = shared_media.resize_volume_to_processing_cube_gray8(
        np.asarray(volume_u8, dtype=np.uint8),
        target,
        resolved_work_dir / "intensity_cube.uint8",
        workers=max(1, int(workers)),
        prefer_memory=True,
        reserve_bytes=0,
    )
    resized_mask = shared_media.resize_categorical_volume_to_processing_cube_uint8(
        np.asarray(mask_u8, dtype=np.uint8),
        target,
        resolved_work_dir / "categorical_cube.uint8",
        workers=max(1, int(workers)),
        prefer_memory=True,
        reserve_bytes=0,
    )
    warnings.add("cubic_resize_applied", f"{source_shape} -> {target}; smaller axes are within 5% of longest axis")
    return resized_volume, resized_mask, target


# ---------------------------------------------------------------------------
# Geometry: views, radial sampling, affine transforms, tilted sampling
# ---------------------------------------------------------------------------



def build_radial_azimuths(azimuth_angle: float) -> List[float]:
    if float(azimuth_angle) <= 0.0:
        return []
    out: List[float] = []
    a = 0.0
    step = float(azimuth_angle)
    while a < 180.0 - 1e-9:
        out.append(float(a))
        a += step
    return out or [0.0]


def default_radial_azimuth_angle(w: int, h: int) -> float:
    """Coverage default from the v3.0.0 specification: 360 / (pi * D) degrees."""
    diameter = max(1, int(min(int(w), int(h))))
    return float(360.0 / (math.pi * float(diameter)))


def resolve_radial_settings(enable_radial: bool, azimuth_angle_arg: Optional[float], *, w: int, h: int, warnings: WarningLog) -> Tuple[bool, float]:
    """Resolve v3.0.0 radial activation and azimuth spacing."""
    if azimuth_angle_arg is None:
        if bool(enable_radial):
            return True, default_radial_azimuth_angle(int(w), int(h))
        return False, 0.0
    spacing = float(azimuth_angle_arg)
    if spacing < 0.0:
        raise ValueError("--azimuth_angle must be >= 0")
    if spacing == 0.0:
        if bool(enable_radial):
            warnings.add("radial_disabled_by_zero_azimuth_angle", "--enable_radial was set, but --azimuth_angle 0 disables radial views")
        return False, 0.0
    if not bool(enable_radial):
        warnings.add("azimuth_angle_ignored_without_enable_radial", "--azimuth_angle > 0 was supplied without --enable_radial")
        return False, 0.0
    return True, spacing


def build_views(t_dim: int, h: int, w: int, *, enable_sagittal: bool, enable_coronal: bool, enable_radial: bool, azimuth_angle: float, tilt_angles: Sequence[float], tilt_directions: Sequence[str]) -> List[ViewInfo]:
    """Build active views in the v3 canonical order.

    Canonical order matters for background filtering and splitting: Transverse,
    Sagittal, Coronal, Radial, then Tilted Transverse variants sorted by
    direction (horizontal before vertical) and signed angle ascending.
    """
    views: List[ViewInfo] = [
        ViewInfo("transverse", "Transverse", "transverse", int(t_dim), int(h), int(w), "clamp", int(t_dim), int(h), int(w))
    ]
    if enable_sagittal:
        views.append(ViewInfo("sagittal", "Sagittal", "sagittal", int(h), int(t_dim), int(w), "pad", int(t_dim), int(h), int(w)))
    if enable_coronal:
        views.append(ViewInfo("coronal", "Coronal", "coronal", int(w), int(t_dim), int(h), "pad", int(t_dim), int(h), int(w)))
    if bool(enable_radial) and float(azimuth_angle) > 0.0:
        azimuths = tuple(build_radial_azimuths(float(azimuth_angle)))
        diameter = int(min(w, h))
        views.append(ViewInfo(
            name="radial",
            display_name="Radial",
            family="radial",
            num_slices=len(azimuths),
            src_h=int(t_dim),
            src_w=int(diameter),
            pad_mode="pad",
            full_t=int(t_dim),
            full_h=int(h),
            full_w=int(w),
            azimuths_deg=azimuths,
            diameter=diameter,
            center_x=float((w - 1) / 2.0),
            center_y=float((h - 1) / 2.0),
            roi_radius=float(max(0, (diameter - 1) / 2.0)),
        ))

    direction_order = {"horizontal": 0, "vertical": 1}
    normalized_dirs = sorted({str(d).lower() for d in tilt_directions if str(d).lower() in direction_order}, key=lambda d: direction_order[d])
    signed_angles = sorted({round(float(sign) * float(a), 8) for a in tilt_angles for sign in (-1.0, +1.0)})
    for direction in normalized_dirs:
        for signed in signed_angles:
            if abs(float(signed)) < 1e-12:
                continue
            views.append(ViewInfo(
                name=f"tilted_transverse_{direction}_{format_signed_token(float(signed))}",
                display_name=f"TiltedTransverse_{direction}_{format_signed_token(float(signed))}",
                family="tilted_transverse",
                num_slices=int(t_dim),
                src_h=int(h),
                src_w=int(w),
                pad_mode="clamp",
                full_t=int(t_dim),
                full_h=int(h),
                full_w=int(w),
                tilt_angle_deg=float(signed),
                tilt_direction=direction,
            ))
    return views


def adapt_shared_view(view: shared_geometry.ViewInfo) -> ViewInfo:
    """Expose a canonical TTA view to the unchanged PTA planner vocabulary."""

    physical_name = shared_geometry.physical_view_name(view)
    if shared_geometry.is_radial_view(view):
        runtime_family = "radial"
    elif shared_geometry.is_tilted_view(view):
        # The old token described only transverse tilts.  It is retained solely
        # for PTA summary/filter call sites; ``shared_view`` owns the base axes.
        runtime_family = "tilted_transverse"
    else:
        runtime_family = str(physical_name)
    return ViewInfo(
        name=str(view.name),
        display_name=str(shared_geometry.view_output_token(view)),
        family=runtime_family,
        num_slices=int(view.num_slices),
        src_h=int(view.src_h),
        src_w=int(view.src_w),
        pad_mode=str(view.pad_mode),
        full_t=int(view.full_t),
        full_h=int(view.full_h),
        full_w=int(view.full_w),
        azimuths_deg=tuple(float(value) for value in view.azimuths_deg),
        diameter=int(view.diameter),
        center_x=float(view.center_x),
        center_y=float(view.center_y),
        roi_radius=float(view.roi_radius),
        tilt_angle_deg=float(view.tilt_angle_deg),
        tilt_direction=str(view.tilt_direction),
        shared_view=view,
    )


def compile_v18_pta_views(
    *,
    t_dim: int,
    h: int,
    w: int,
    config: object,
    radial_native_raster: int,
) -> Tuple[List[ViewInfo], object]:
    """Compile PTA grouped requests with the exact shared TTA view compiler."""

    compiled = compile_physical_views(
        t_dim=int(t_dim),
        height=int(h),
        width=int(w),
        cartesian_views=tuple(getattr(config, "cartesian_views")),
        radial_requests=tuple(getattr(config, "radial_requests")),
        tilted_groups=tuple(getattr(config, "tilted_groups")),
        radial_native_raster=int(radial_native_raster),
    )
    return [adapt_shared_view(view) for view in compiled.views], compiled

def rotated_bbox_size(w: int, h: int, angle_deg: float) -> Tuple[int, int]:
    theta = math.radians(float(angle_deg) % 360.0)
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    bw = int(math.ceil(float(w) * c + float(h) * s))
    bh = int(math.ceil(float(w) * s + float(h) * c))
    return max(1, bw), max(1, bh)


def affine2x3_to_3x3(M: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, :3] = np.asarray(M, dtype=np.float64)
    return out


def center_scale_matrix(src_w: int, src_h: int, out_w: int, out_h: int) -> np.ndarray:
    cx_s = (int(src_w) - 1) / 2.0
    cy_s = (int(src_h) - 1) / 2.0
    cx_o = (int(out_w) - 1) / 2.0
    cy_o = (int(out_h) - 1) / 2.0
    sx = float(out_w) / float(src_w)
    sy = float(out_h) / float(src_h)
    return np.array([[sx, 0.0, cx_o - sx * cx_s], [0.0, sy, cy_o - sy * cy_s], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_affine(
    src_w: int,
    src_h: int,
    angle_deg: float,
    pad_mode: str,
    imgsz: int,
    *,
    shared_view: Optional[shared_geometry.ViewInfo] = None,
) -> AffineSpec:
    if shared_view is not None:
        # TTA owns padding, rotation, center alignment, and square scaling.
        # PTA's imgsz=0 remains a native-raster publication default; the
        # canonical canvas matrices are still retained for grouped tiles.
        native_output = int(imgsz) == 0
        shared_out_size = int(imgsz) if int(imgsz) > 0 else max(int(src_w), int(src_h), 1)
        shared_affine = shared_geometry.build_affine(
            view=str(shared_view.name),
            src_w=int(src_w),
            src_h=int(src_h),
            out_size=int(shared_out_size),
            angle_deg=float(angle_deg),
            pad_mode=str(pad_mode),
        )
        if native_output:
            identity = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
            )
            out_w, out_h = int(src_w), int(src_h)
            src_to_out = identity
            out_to_src = identity
        else:
            out_w = out_h = int(shared_affine.out_size)
            src_to_out = np.asarray(shared_affine.M_src_to_out, dtype=np.float32)
            out_to_src = np.asarray(shared_affine.M_out_to_src, dtype=np.float32)
        return AffineSpec(
            angle_deg=float(angle_deg),
            src_w=int(src_w),
            src_h=int(src_h),
            canvas_w=int(shared_affine.canvas_w),
            canvas_h=int(shared_affine.canvas_h),
            out_w=int(out_w),
            out_h=int(out_h),
            pad_off_x=float(shared_affine.pad_off_x),
            pad_off_y=float(shared_affine.pad_off_y),
            M_src_to_canvas=np.asarray(shared_affine.M_src_to_canvas, dtype=np.float32),
            M_canvas_to_src=np.asarray(shared_affine.M_canvas_to_src, dtype=np.float32),
            M_src_to_out=src_to_out,
            M_out_to_src=out_to_src,
            shared_affine=shared_affine,
            native_output=bool(native_output),
        )
    if pad_mode not in {"clamp", "pad"}:
        raise ValueError("pad_mode must be clamp or pad")
    if pad_mode == "pad":
        canvas_w, canvas_h = rotated_bbox_size(int(src_w), int(src_h), float(angle_deg))
    else:
        canvas_w, canvas_h = int(src_w), int(src_h)
    off_x = (canvas_w - int(src_w)) / 2.0
    off_y = (canvas_h - int(src_h)) / 2.0
    cx = (canvas_w - 1) / 2.0
    cy = (canvas_h - 1) / 2.0
    M_pad = np.array([[1.0, 0.0, off_x], [0.0, 1.0, off_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    M_rot = affine2x3_to_3x3(cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0))
    M_src_to_canvas3 = M_rot @ M_pad
    M_canvas_to_src3 = np.linalg.inv(M_src_to_canvas3)
    if int(imgsz) > 0:
        out_w = out_h = int(imgsz)
        M_scale = center_scale_matrix(canvas_w, canvas_h, out_w, out_h)
    else:
        out_w, out_h = int(canvas_w), int(canvas_h)
        M_scale = np.eye(3, dtype=np.float64)
    M_src_to_out3 = M_scale @ M_src_to_canvas3
    M_out_to_src3 = np.linalg.inv(M_src_to_out3)
    return AffineSpec(
        float(angle_deg), int(src_w), int(src_h), int(canvas_w), int(canvas_h), int(out_w), int(out_h), float(off_x), float(off_y),
        M_src_to_canvas3[:2, :].astype(np.float32), M_canvas_to_src3[:2, :].astype(np.float32),
        M_src_to_out3[:2, :].astype(np.float32), M_out_to_src3[:2, :].astype(np.float32),
    )




def dense_tile_positions(length: int, tile_size: int, stride: int) -> List[int]:
    length = int(length)
    tile_size = int(tile_size)
    stride = int(stride)
    last = max(0, length - tile_size)
    starts = list(range(0, last + 1, stride)) if last > 0 else [0]
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return [int(x) for x in starts]


# ---------------------------------------------------------------------------
# YOLO polygon export
# ---------------------------------------------------------------------------


def frame_path_from_pattern(pattern: str, idx_1: int) -> Path:
    return Path(pattern % int(idx_1))


def angle_tag(angle: float) -> str:
    return f"a{format_float_token(float(angle))}"


def make_tag(view: ViewInfo, angle: float = 0.0) -> str:
    # v3.0.0 removes rotation-angle augmentation flags.  The angle parameter is
    # retained internally only so the existing affine/render scheduler can emit
    # the canonical unrotated view without changing its queue structure.
    if abs(float(angle)) < 1e-9:
        return view.display_name
    return f"{view.display_name}_{angle_tag(angle)}"


def render_variant(
    *,
    volume: np.ndarray,
    mask: np.ndarray,
    view: ViewInfo,
    aff: AffineSpec,
    tag: str,
    out_dir: Path,
    stem: str,
    fps: float,
    tile_configs: Sequence[TileConfig],
    save_overlay: bool,
    overlay_tile_writer_limit: int,
    png_compression: int,
    imgsz: int,
    warnings: WarningLog,
    workers: int,
    image_format: str = "png",
) -> Dict[str, object]:
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    overlays_dir = out_dir / "overlays"
    image_suffix = output_image_suffix(image_format)
    img_pattern = str(images_dir / f"{stem}_{tag}_%04d{image_suffix}")
    lbl_pattern = str(labels_dir / f"{stem}_{tag}_%04d.txt")
    overlay_path = overlays_dir / f"{stem}_{tag}_Overlay.mkv" if save_overlay else None
    stats: Dict[str, object] = {
        "tag": tag,
        "view": view.name,
        "angle": aff.angle_deg,
        "frames": int(view.num_slices),
        "image_format": parse_output_image_format(image_format),
        "full_output_size": [int(aff.out_w), int(aff.out_h)],
        "tiles": [],
    }

    tile_layout: List[Tuple[TileConfig, int, int, str, int, int, str, str, Optional[Path]]] = []
    for cfg in tile_configs:
        xs = dense_tile_positions(aff.canvas_w, cfg.tile_size, cfg.tile_stride)
        ys = dense_tile_positions(aff.canvas_h, cfg.tile_size, cfg.tile_stride)
        out_side = int(imgsz) if int(imgsz) > 0 else int(cfg.tile_size)
        for y in ys:
            for x in xs:
                tile_tag = f"{tag}_tile_{cfg.config_id}_x{int(x):04d}_y{int(y):04d}"
                tile_img_pattern = str(images_dir / f"{stem}_{tile_tag}_%04d{image_suffix}")
                tile_lbl_pattern = str(labels_dir / f"{stem}_{tile_tag}_%04d.txt")
                tile_overlay = overlays_dir / f"{stem}_{tile_tag}_Overlay.mkv" if save_overlay else None
                tile_layout.append((cfg, int(x), int(y), tile_tag, int(out_side), int(out_side), tile_img_pattern, tile_lbl_pattern, tile_overlay))
    stats["tiles"] = [
        {"tag": item[3], "tile_size": item[0].tile_size, "tile_stride": item[0].tile_stride, "x": item[1], "y": item[2], "output_size": [item[4], item[5]]}
        for item in tile_layout
    ]
    need_canvas = bool(tile_layout)

    def _process_frame(idx: int) -> None:
        img_full, mask_full, img_canvas, mask_canvas = render_full_and_optional_canvas(volume, mask, view, int(idx), aff, need_canvas)
        write_image_gray(frame_path_from_pattern(img_pattern, int(idx) + 1), img_full, png_compression)
        write_label_from_mask(mask_full, frame_path_from_pattern(lbl_pattern, int(idx) + 1), warnings=warnings, context=f"{tag} frame {int(idx)+1:04d}")
        if need_canvas:
            assert img_canvas is not None and mask_canvas is not None
            for cfg, x, y, tile_tag, out_w, out_h, tile_img_pattern, tile_lbl_pattern, _tile_overlay in tile_layout:
                tile_img = extract_padded_tile(img_canvas, x, y, cfg.tile_size)
                tile_mask = extract_padded_tile(mask_canvas, x, y, cfg.tile_size)
                tile_img_out = resize_centered(tile_img, out_w, out_h, cv2.INTER_LINEAR)
                tile_mask_out = resize_centered(tile_mask, out_w, out_h, cv2.INTER_NEAREST)
                write_image_gray(frame_path_from_pattern(tile_img_pattern, int(idx) + 1), tile_img_out, png_compression)
                write_label_from_mask(tile_mask_out, frame_path_from_pattern(tile_lbl_pattern, int(idx) + 1), warnings=warnings, context=f"{tile_tag} frame {int(idx)+1:04d}")

    if not save_overlay:
        parallel_for_indices(int(view.num_slices), _process_frame, workers=workers, desc=f"Rendering {tag}")
        return stats

    full_writer = ffmpeg_ffv1_rgb_writer(overlay_path, aff.out_w, aff.out_h, fps) if overlay_path is not None else None
    tile_writers: Dict[str, subprocess.Popen] = {}
    immediate_tile_layout = tile_layout
    deferred_tile_layout: List[Tuple[TileConfig, int, int, str, int, int, str, str, Optional[Path]]] = []
    if tile_layout and int(overlay_tile_writer_limit) >= len(tile_layout):
        for _cfg, _x, _y, tile_tag, out_w, out_h, _img_pat, _lbl_pat, tile_overlay in tile_layout:
            if tile_overlay is not None:
                tile_writers[tile_tag] = ffmpeg_ffv1_rgb_writer(tile_overlay, out_w, out_h, fps)
    elif tile_layout:
        # Avoid holding thousands of ffmpeg processes open simultaneously.  Images/labels and the
        # full-frame overlay are written in the primary pass; tile overlays are generated below in
        # bounded batches by recomputing the rotated canvas frames.
        immediate_tile_layout = []
        deferred_tile_layout = list(tile_layout)
        warnings.add("tile_overlay_batched", f"{tag}: {len(tile_layout)} tile overlays generated in bounded batches")

    try:
        for idx in tqdm(range(int(view.num_slices)), desc=f"Rendering {tag} with overlays"):
            img_full, mask_full, img_canvas, mask_canvas = render_full_and_optional_canvas(volume, mask, view, int(idx), aff, need_canvas)
            write_image_gray(frame_path_from_pattern(img_pattern, int(idx) + 1), img_full, png_compression)
            write_label_from_mask(mask_full, frame_path_from_pattern(lbl_pattern, int(idx) + 1), warnings=warnings, context=f"{tag} frame {int(idx)+1:04d}")
            if full_writer is not None and full_writer.stdin is not None:
                full_writer.stdin.write(overlay_rgb(img_full, mask_full).tobytes())
            if need_canvas:
                assert img_canvas is not None and mask_canvas is not None
                for cfg, x, y, tile_tag, out_w, out_h, tile_img_pattern, tile_lbl_pattern, _tile_overlay in tile_layout:
                    tile_img = extract_padded_tile(img_canvas, x, y, cfg.tile_size)
                    tile_mask = extract_padded_tile(mask_canvas, x, y, cfg.tile_size)
                    tile_img_out = resize_centered(tile_img, out_w, out_h, cv2.INTER_LINEAR)
                    tile_mask_out = resize_centered(tile_mask, out_w, out_h, cv2.INTER_NEAREST)
                    write_image_gray(frame_path_from_pattern(tile_img_pattern, int(idx) + 1), tile_img_out, png_compression)
                    write_label_from_mask(tile_mask_out, frame_path_from_pattern(tile_lbl_pattern, int(idx) + 1), warnings=warnings, context=f"{tile_tag} frame {int(idx)+1:04d}")
                    writer = tile_writers.get(tile_tag) if immediate_tile_layout else None
                    if writer is not None and writer.stdin is not None:
                        writer.stdin.write(overlay_rgb(tile_img_out, tile_mask_out).tobytes())
    finally:
        if full_writer is not None:
            close_ffmpeg_writer(full_writer)
        for writer in tile_writers.values():
            close_ffmpeg_writer(writer)

    if deferred_tile_layout:
        batch_size = max(1, int(overlay_tile_writer_limit))
        for batch_start in range(0, len(deferred_tile_layout), batch_size):
            batch = deferred_tile_layout[batch_start:batch_start + batch_size]
            batch_writers: Dict[str, subprocess.Popen] = {}
            try:
                for _cfg, _x, _y, tile_tag, out_w, out_h, _img_pat, _lbl_pat, tile_overlay in batch:
                    if tile_overlay is not None:
                        batch_writers[tile_tag] = ffmpeg_ffv1_rgb_writer(tile_overlay, out_w, out_h, fps)
                batch_no = (batch_start // batch_size) + 1
                batch_total = int(math.ceil(len(deferred_tile_layout) / float(batch_size)))
                for idx in tqdm(range(int(view.num_slices)), desc=f"Rendering tile overlays {tag} batch {batch_no}/{batch_total}"):
                    _img_full, _mask_full, img_canvas, mask_canvas = render_full_and_optional_canvas(volume, mask, view, int(idx), aff, True)
                    assert img_canvas is not None and mask_canvas is not None
                    for cfg, x, y, tile_tag, out_w, out_h, _tile_img_pattern, _tile_lbl_pattern, _tile_overlay in batch:
                        writer = batch_writers.get(tile_tag)
                        if writer is None or writer.stdin is None:
                            continue
                        tile_img = extract_padded_tile(img_canvas, x, y, cfg.tile_size)
                        tile_mask = extract_padded_tile(mask_canvas, x, y, cfg.tile_size)
                        tile_img_out = resize_centered(tile_img, out_w, out_h, cv2.INTER_LINEAR)
                        tile_mask_out = resize_centered(tile_mask, out_w, out_h, cv2.INTER_NEAREST)
                        writer.stdin.write(overlay_rgb(tile_img_out, tile_mask_out).tobytes())
            finally:
                for writer in batch_writers.values():
                    close_ffmpeg_writer(writer)

    return stats


# ---------------------------------------------------------------------------
# Throughput-first global render scheduler
# ---------------------------------------------------------------------------



def build_render_plan(
    *,
    view: ViewInfo,
    aff: AffineSpec,
    tag: str,
    out_dir: Path,
    stem: str,
    tile_configs: Sequence[TileConfig],
    save_overlay: bool,
    imgsz: int,
    label_enabled: bool,
    image_format: str = "png",
    channel_variant: ChannelVariant = DEFAULT_CHANNEL_VARIANT,
    eligible_frame_indices: Optional[Sequence[int]] = None,
    source_encoded_indices: Sequence[int] = (),
    publish_images: bool = True,
    publish_labels: bool = True,
) -> RenderPlan:
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    overlays_dir = out_dir / "overlays"
    if publish_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    if label_enabled and publish_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    image_suffix = output_image_suffix(image_format)
    img_pattern = str(images_dir / f"{stem}_{tag}_%04d{image_suffix}")
    lbl_pattern = str(labels_dir / f"{stem}_{tag}_%04d.txt")
    overlay_path = overlays_dir / f"{stem}_{tag}_Overlay.mkv" if save_overlay else None

    shared_aug_job: Optional[shared_geometry.AugJob] = None
    if view.shared_view is not None and aff.shared_affine is not None:
        shared_aug_job = shared_geometry.AugJob(
            aug_id="a0",
            angle_deg=0.0,
            meta_path=out_dir / ".v18_geometry" / f"{stem}_{tag}.meta.json",
            aff=aff.shared_affine,
        )
    canonical_plan: Optional[RasterPlan] = None
    if view.shared_view is not None:
        canonical_plan = build_forward_raster_plan(
            mode="pta",
            physical_view_id=shared_geometry.physical_view_name(view.shared_view),
            angle_deg=float(aff.angle_deg),
            channel_token=str(channel_variant.format_token),
            channel_kind=str(channel_variant.kind),
            channel_count=int(channel_variant.channel_count),
            channel_stride=int(channel_variant.stride),
            channel_offsets=tuple(int(value) for value in channel_variant.offsets),
            channel_direction=str(channel_variant.order_name),
            output_shape=(int(aff.out_h), int(aff.out_w)),
            metadata={
                "runtime_view_id": str(view.name),
                "runtime_job_id": str(tag),
                "runtime_kind": "fullframe",
            },
        )

    tile_items: List[RenderTileItem] = []
    for cfg in tile_configs:
        out_side = int(imgsz) if int(imgsz) > 0 else int(cfg.tile_size)
        if view.shared_view is not None and shared_aug_job is not None:
            shared_jobs: Sequence[Optional[shared_geometry.DenseTileJob]] = tuple(
                shared_geometry.build_dense_tile_jobs_for_aug(
                    view.shared_view,
                    shared_aug_job,
                    shared_geometry.TileConfig(
                        tile_size=int(cfg.tile_size),
                        tile_stride=int(cfg.tile_stride),
                        config_id=str(cfg.config_id),
                    ),
                    out_size=int(out_side),
                    temp_dir=out_dir / ".v18_geometry",
                )
            )
        else:
            shared_jobs = tuple(
                None
                for _y in dense_tile_positions(aff.canvas_h, cfg.tile_size, cfg.tile_stride)
                for _x in dense_tile_positions(aff.canvas_w, cfg.tile_size, cfg.tile_stride)
            )
        if shared_jobs and shared_jobs[0] is not None:
            positions = (
                (int(job.tile_x), int(job.tile_y), job)
                for job in shared_jobs
                if job is not None
            )
        else:
            positions = (
                (int(x), int(y), None)
                for y in dense_tile_positions(aff.canvas_h, cfg.tile_size, cfg.tile_stride)
                for x in dense_tile_positions(aff.canvas_w, cfg.tile_size, cfg.tile_stride)
            )
        for x, y, shared_job in positions:
            tile_tag = f"{tag}_tile_{cfg.config_id}_x{int(x):04d}_y{int(y):04d}"
            canonical_tile_plan: Optional[RasterPlan] = None
            if view.shared_view is not None:
                canonical_tile_plan = build_forward_raster_plan(
                    mode="pta",
                    physical_view_id=shared_geometry.physical_view_name(
                        view.shared_view
                    ),
                    angle_deg=float(aff.angle_deg),
                    channel_token=str(channel_variant.format_token),
                    channel_kind=str(channel_variant.kind),
                    channel_count=int(channel_variant.channel_count),
                    channel_stride=int(channel_variant.stride),
                    channel_offsets=tuple(
                        int(value) for value in channel_variant.offsets
                    ),
                    channel_direction=str(channel_variant.order_name),
                    output_shape=(int(out_side), int(out_side)),
                    tile_size=int(cfg.tile_size),
                    tile_stride=int(cfg.tile_stride),
                    metadata={
                        "runtime_view_id": str(view.name),
                        "runtime_job_id": str(tile_tag),
                        "runtime_kind": "tile",
                        "tile_config_id": str(cfg.config_id),
                        "tile_x": int(x),
                        "tile_y": int(y),
                    },
                )
            tile_items.append(RenderTileItem(
                cfg=cfg,
                x=int(x),
                y=int(y),
                tile_tag=tile_tag,
                out_w=int(out_side),
                out_h=int(out_side),
                img_pattern=str(images_dir / f"{stem}_{tile_tag}_%04d{image_suffix}"),
                lbl_pattern=str(labels_dir / f"{stem}_{tile_tag}_%04d.txt"),
                overlay_path=(overlays_dir / f"{stem}_{tile_tag}_Overlay.mkv" if save_overlay else None),
                label_enabled=bool(label_enabled),
                channel_kind=str(channel_variant.kind),
                shared_job=shared_job,
                canonical_plan=canonical_tile_plan,
                publish_images=bool(publish_images),
                publish_labels=bool(publish_labels),
            ))

    normalized_eligible: Optional[Tuple[int, ...]]
    if eligible_frame_indices is None:
        normalized_eligible = None
        eligible_count = int(view.num_slices)
    else:
        normalized_eligible = tuple(sorted({int(x) for x in eligible_frame_indices}))
        if any(x < 0 or x >= int(view.num_slices) for x in normalized_eligible):
            raise ValueError(
                f"Eligible center index is outside view {view.name} bounds 0..{int(view.num_slices)-1}: "
                f"{normalized_eligible[:24]}"
            )
        eligible_count = len(normalized_eligible)
    normalized_encoded = tuple(int(x) for x in source_encoded_indices)
    if normalized_encoded and len(normalized_encoded) != int(view.num_slices):
        raise ValueError(
            f"Encoded-index mapping for {view.name} has {len(normalized_encoded)} entries, "
            f"expected {int(view.num_slices)}"
        )

    stats: Dict[str, object] = {
        "tag": tag,
        "view": view.name,
        "angle": float(aff.angle_deg),
        "frames": int(eligible_count),
        "view_frames": int(view.num_slices),
        "skipped_center_frames": int(view.num_slices) - int(eligible_count),
        "image_format": parse_output_image_format(image_format),
        "channel_format": str(channel_variant.format_token),
        "channel_kind": str(channel_variant.kind),
        "channel_count": int(channel_variant.channel_count),
        "channel_stride": int(channel_variant.stride),
        "channel_order": str(channel_variant.order_name),
        "channel_offsets": [int(x) for x in channel_variant.offsets],
        "full_output_size": [int(aff.out_w), int(aff.out_h)],
        "tiles": [
            {"tag": item.tile_tag, "tile_size": int(item.cfg.tile_size), "tile_stride": int(item.cfg.tile_stride), "x": int(item.x), "y": int(item.y), "output_size": [int(item.out_w), int(item.out_h)]}
            for item in tile_items
        ],
        "scheduler": "global_source_frame_and_output_frame_parallel",
        "forward_geometry": "XTA.geometry",
        "forward_backend": "runtime_selected_cpu_or_cuda_tta_canonical",
        "sampling_policy_digest": forward_sampling_policy().digest,
        "canonical_plan_digest": (
            None if canonical_plan is None else str(canonical_plan.digest)
        ),
        "canonical_tile_plan_digests": [
            str(item.canonical_plan.digest)
            for item in tile_items
            if item.canonical_plan is not None
        ],
        "label_enabled": bool(label_enabled),
        "publish_images": bool(publish_images),
        "publish_labels": bool(publish_labels),
    }
    return RenderPlan(
        view=view,
        aff=aff,
        tag=tag,
        img_pattern=img_pattern,
        lbl_pattern=lbl_pattern,
        overlay_path=overlay_path,
        label_enabled=bool(label_enabled),
        tile_layout=tuple(tile_items),
        stats=stats,
        channel_variant=channel_variant,
        eligible_frame_indices=normalized_eligible,
        source_encoded_indices=normalized_encoded,
        shared_aug_job=shared_aug_job,
        canonical_plan=canonical_plan,
        publish_images=bool(publish_images),
        publish_labels=bool(publish_labels),
    )


def render_plan_frame_indices(plan: RenderPlan) -> Sequence[int]:
    if plan.eligible_frame_indices is None:
        return range(int(plan.view.num_slices))
    return plan.eligible_frame_indices


def iter_render_source_frame_jobs_round_robin(plans: Sequence[RenderPlan]) -> Iterator[Tuple[int, int]]:
    """Yield only eligible source-frame jobs, frame-major across variants.

    Keep the original bounded-memory iteration pattern: eligibility is stored
    per plan rather than materializing every (plan, frame) job in a global map.
    """
    if not plans:
        return
    eligible_sets: List[Optional[set[int]]] = [
        None if plan.eligible_frame_indices is None else set(int(x) for x in plan.eligible_frame_indices)
        for plan in plans
    ]
    max_frames = max(int(plan.view.num_slices) for plan in plans)
    for frame_idx in range(max_frames):
        for plan_idx, plan in enumerate(plans):
            if frame_idx >= int(plan.view.num_slices):
                continue
            eligible = eligible_sets[plan_idx]
            if eligible is None or int(frame_idx) in eligible:
                yield int(plan_idx), int(frame_idx)




def write_full_render_output_from_source(*, plan: RenderPlan, idx: int, source: RenderFrameSource, png_compression: int, warnings: WarningLog) -> None:
    idx_1 = int(idx) + 1
    if plan.publish_images:
        write_image(
            frame_path_from_pattern(plan.img_pattern, idx_1),
            source.img_full,
            int(png_compression),
            channel_kind=plan.channel_variant.kind,
        )
    if plan.label_enabled and plan.publish_labels:
        write_label_from_mask(
            source.mask_full,
            frame_path_from_pattern(plan.lbl_pattern, idx_1),
            warnings=warnings,
            context=f"{plan.tag} frame {idx_1:04d}",
        )


def write_tile_render_output_from_source(*, tile: RenderTileItem, idx: int, source: RenderFrameSource, png_compression: int, warnings: WarningLog) -> None:
    idx_1 = int(idx) + 1
    shared_arrays = source.tile_arrays.get(str(tile.tile_tag))
    if shared_arrays is not None:
        tile_img_out, tile_mask_out = shared_arrays
    else:
        if source.img_canvas is None or source.mask_canvas is None:
            raise RuntimeError(f"Tile output requested without a rendered canvas for {tile.tile_tag}")
        tile_img = extract_padded_tile(source.img_canvas, tile.x, tile.y, tile.cfg.tile_size)
        tile_mask = extract_padded_tile(source.mask_canvas, tile.x, tile.y, tile.cfg.tile_size)
        tile_img_out = resize_centered(tile_img, tile.out_w, tile.out_h, cv2.INTER_LINEAR)
        tile_mask_out = resize_centered(tile_mask, tile.out_w, tile.out_h, cv2.INTER_NEAREST)
    if tile.publish_images:
        write_image(
            frame_path_from_pattern(tile.img_pattern, idx_1),
            tile_img_out,
            int(png_compression),
            channel_kind=tile.channel_kind,
        )
    if tile.label_enabled and tile.publish_labels:
        write_label_from_mask(
            tile_mask_out,
            frame_path_from_pattern(tile.lbl_pattern, idx_1),
            warnings=warnings,
            context=f"{tile.tile_tag} frame {idx_1:04d}",
        )


def write_tile_render_outputs_from_source(*, tiles: Sequence[RenderTileItem], idx: int, source: RenderFrameSource, png_compression: int, warnings: WarningLog) -> None:
    """Write one scheduled tile-frame task, optionally containing a tile chunk.

    ``--tile_task_chunk 1`` exposes every tile frame as its own independent task. Larger
    chunks are an escape hatch if filesystem/open-file overhead becomes dominant.
    """
    for tile in tiles:
        write_tile_render_output_from_source(tile=tile, idx=int(idx), source=source, png_compression=int(png_compression), warnings=warnings)


def render_primary_outputs_global(*, volume: np.ndarray, mask: np.ndarray, plans: Sequence[RenderPlan], png_compression: int, warnings: WarningLog, workers: int, max_pending: int, tile_task_chunk: int = 1) -> None:
    """Render images/labels with global source-frame and output-frame queues.

    Source-frame jobs do the expensive reslice once for each (view/angle, frame). Full-frame
    writes and tile writes are then separate output jobs sharing that source frame/canvas.
    """
    total_sources = int(sum(len(render_plan_frame_indices(plan)) for plan in plans))
    tile_chunk = max(1, int(tile_task_chunk))
    total_outputs = int(sum(len(render_plan_frame_indices(plan)) * (1 + len(plan.tile_layout)) for plan in plans))
    scheduled_output_tasks = int(sum(
        len(render_plan_frame_indices(plan))
        * (1 + int(math.ceil(len(plan.tile_layout) / float(tile_chunk))) if plan.tile_layout else 1)
        for plan in plans
    ))
    if total_sources <= 0 or total_outputs <= 0:
        return

    worker_budget = max(1, int(workers))
    source_workers = max(1, min(worker_budget, total_sources))
    output_workers = max(1, min(worker_budget, total_outputs))
    requested_window = int(max_pending) if int(max_pending) > 0 else 0
    source_pending_limit = max(source_workers, requested_window if requested_window > 0 else max(512, source_workers * 4))
    source_pending_limit = min(max(1, source_pending_limit), total_sources)
    output_pending_limit = requested_window * 2 if requested_window > 0 else max(1024, output_workers * 8)
    max_dependents = max(1 + (int(math.ceil(len(plan.tile_layout) / float(tile_chunk))) if plan.tile_layout else 0) for plan in plans)
    output_pending_limit = max(output_workers, max_dependents, int(output_pending_limit))

    print(
        f"Global frame scheduler: source_frame_tasks={total_sources}, output_frames={total_outputs}, "
        f"scheduled_output_tasks={scheduled_output_tasks}, tile_task_chunk={tile_chunk}, "
        f"source_workers={source_workers}, output_workers={output_workers}, "
        f"source_window={source_pending_limit}, output_window={output_pending_limit}"
    )

    source_iter = iter(iter_render_source_frame_jobs_round_robin(plans))
    pending_sources: Dict[Future, Tuple[int, int]] = {}
    pending_outputs: set[Future] = set()
    pending_output_units: Dict[Future, int] = {}
    exhausted_sources = False

    def _submit_more_sources(source_executor: ThreadPoolExecutor) -> None:
        nonlocal exhausted_sources
        while not exhausted_sources and len(pending_sources) < source_pending_limit:
            try:
                plan_idx, frame_idx = next(source_iter)
            except StopIteration:
                exhausted_sources = True
                break
            plan = plans[int(plan_idx)]
            fut = source_executor.submit(render_plan_frame_source, volume=volume, mask=mask, plan=plan, idx=int(frame_idx))
            pending_sources[fut] = (int(plan_idx), int(frame_idx))

    def _drain_outputs(pbar: tqdm, *, block: bool) -> None:
        if not pending_outputs:
            return
        if block:
            done, _ = wait(pending_outputs, return_when=FIRST_COMPLETED)
        else:
            done = {fut for fut in pending_outputs if fut.done()}
        for fut in done:
            pending_outputs.remove(fut)
            fut.result()
            pbar.update(int(pending_output_units.pop(fut, 1)))

    def _submit_output_future(fut: Future, pbar: tqdm, *, units: int = 1) -> None:
        while len(pending_outputs) >= output_pending_limit:
            _drain_outputs(pbar, block=True)
        pending_outputs.add(fut)
        pending_output_units[fut] = max(1, int(units))

    with ThreadPoolExecutor(max_workers=source_workers, thread_name_prefix="pretrain-source-frame") as source_executor, \
         ThreadPoolExecutor(max_workers=output_workers, thread_name_prefix="pretrain-output-frame") as output_executor, \
         tqdm(total=total_outputs, desc=f"Rendering images/labels globally ({len(plans)} variants)") as pbar:
        _submit_more_sources(source_executor)
        while pending_sources:
            done_sources, _ = wait(pending_sources, return_when=FIRST_COMPLETED)
            for source_future in done_sources:
                plan_idx, frame_idx = pending_sources.pop(source_future)
                source = source_future.result()
                plan = plans[int(plan_idx)]

                _submit_output_future(
                    output_executor.submit(
                        write_full_render_output_from_source,
                        plan=plan,
                        idx=int(frame_idx),
                        source=source,
                        png_compression=int(png_compression),
                        warnings=warnings,
                    ),
                    pbar,
                )
                for tile_start in range(0, len(plan.tile_layout), tile_chunk):
                    tile_chunk_items = plan.tile_layout[tile_start:tile_start + tile_chunk]
                    _submit_output_future(
                        output_executor.submit(
                            write_tile_render_outputs_from_source,
                            tiles=tile_chunk_items,
                            idx=int(frame_idx),
                            source=source,
                            png_compression=int(png_compression),
                            warnings=warnings,
                        ),
                        pbar,
                        units=len(tile_chunk_items),
                    )
                _drain_outputs(pbar, block=False)
            _submit_more_sources(source_executor)
        while pending_outputs:
            _drain_outputs(pbar, block=True)


def parallel_map_ordered_limited(count: int, func: Callable[[int], object], *, workers: int, max_pending: int, desc: str) -> Iterator[object]:
    """Compute frames concurrently but yield results in input order for video writers."""
    total = max(0, int(count))
    if total <= 0:
        return
    nworkers = max(1, min(int(workers), total))
    if nworkers <= 1:
        for idx in tqdm(range(total), desc=desc):
            yield func(int(idx))
        return

    requested_pending = int(max_pending) if int(max_pending) > 0 else nworkers
    pending_limit = max(1, min(requested_pending, total))
    with ThreadPoolExecutor(max_workers=nworkers, thread_name_prefix="pretrain-overlay-frame") as executor:
        queue: List[Future] = []
        next_idx = 0
        while next_idx < total and len(queue) < pending_limit:
            queue.append(executor.submit(func, int(next_idx)))
            next_idx += 1
        with tqdm(total=total, desc=desc) as pbar:
            while queue:
                fut = queue.pop(0)
                result = fut.result()
                pbar.update(1)
                if next_idx < total:
                    queue.append(executor.submit(func, int(next_idx)))
                    next_idx += 1
                yield result


def write_overlay_for_plan(*, volume: np.ndarray, mask: np.ndarray, plan: RenderPlan, fps: float, overlay_tile_writer_limit: int, frame_workers: int, overlay_pending_frames: int, warnings: WarningLog) -> None:
    if plan.overlay_path is None:
        return
    frame_indices = tuple(int(x) for x in render_plan_frame_indices(plan))
    if not frame_indices:
        return
    frame_worker_count = max(1, int(frame_workers))
    pending = int(overlay_pending_frames) if int(overlay_pending_frames) > 0 else min(64, max(frame_worker_count, frame_worker_count * 2))
    writer = ffmpeg_ffv1_rgb_writer(plan.overlay_path, plan.aff.out_w, plan.aff.out_h, fps)
    try:
        assert writer.stdin is not None
        def _render_full(order_idx: int) -> bytes:
            frame_idx = int(frame_indices[int(order_idx)])
            img_full, mask_full, _img_canvas, _mask_canvas = render_full_and_optional_canvas(volume, mask, plan.view, frame_idx, plan.aff, False)
            return overlay_rgb(img_full, mask_full).tobytes()
        for payload in parallel_map_ordered_limited(len(frame_indices), _render_full, workers=frame_worker_count, max_pending=pending, desc=f"Overlay frames {plan.tag}"):
            writer.stdin.write(payload)
    finally:
        close_ffmpeg_writer(writer)

    if not plan.tile_layout:
        return
    batch_size = max(1, int(overlay_tile_writer_limit))
    if len(plan.tile_layout) > batch_size:
        warnings.add("tile_overlay_batched", f"{plan.tag}: {len(plan.tile_layout)} tile overlays generated in batches of {batch_size}")
    for batch_start in range(0, len(plan.tile_layout), batch_size):
        batch = plan.tile_layout[batch_start:batch_start + batch_size]
        batch_no = (batch_start // batch_size) + 1
        batch_total = int(math.ceil(len(plan.tile_layout) / float(batch_size)))
        writers: Dict[str, subprocess.Popen] = {}
        try:
            for tile in batch:
                if tile.overlay_path is not None:
                    writers[tile.tile_tag] = ffmpeg_ffv1_rgb_writer(tile.overlay_path, tile.out_w, tile.out_h, fps)
            if not writers:
                continue
            def _render_tiles(order_idx: int) -> List[Tuple[str, bytes]]:
                frame_idx = int(frame_indices[int(order_idx)])
                payloads: List[Tuple[str, bytes]] = []
                shared_batch = all(tile.shared_job is not None for tile in batch)
                img_canvas: Optional[np.ndarray] = None
                mask_canvas: Optional[np.ndarray] = None
                if not shared_batch:
                    _img_full, _mask_full, img_canvas, mask_canvas = render_full_and_optional_canvas(
                        volume, mask, plan.view, frame_idx, plan.aff, True
                    )
                    assert img_canvas is not None and mask_canvas is not None
                for tile in batch:
                    if tile.tile_tag not in writers:
                        continue
                    if tile.shared_job is not None:
                        if plan.view.shared_view is None:
                            raise RuntimeError("shared overlay tile has no canonical view")
                        tile_img_out = render_shared_tile_images(
                            volume=volume,
                            plan=plan,
                            tile=tile,
                            idx=frame_idx,
                        )
                        tile_mask_out = shared_geometry.render_categorical_dense_tile_for_job(
                            mask,
                            plan.view.shared_view,
                            tile.shared_job,
                            frame_idx,
                        )
                    else:
                        assert img_canvas is not None and mask_canvas is not None
                        tile_img = extract_padded_tile(img_canvas, tile.x, tile.y, tile.cfg.tile_size)
                        tile_mask = extract_padded_tile(mask_canvas, tile.x, tile.y, tile.cfg.tile_size)
                        tile_img_out = resize_centered(tile_img, tile.out_w, tile.out_h, cv2.INTER_LINEAR)
                        tile_mask_out = resize_centered(tile_mask, tile.out_w, tile.out_h, cv2.INTER_NEAREST)
                    payloads.append((tile.tile_tag, overlay_rgb(tile_img_out, tile_mask_out).tobytes()))
                return payloads
            for frame_payloads in parallel_map_ordered_limited(len(frame_indices), _render_tiles, workers=frame_worker_count, max_pending=pending, desc=f"Tile overlays {plan.tag} batch {batch_no}/{batch_total}"):
                for tile_tag, payload in frame_payloads:
                    proc = writers.get(tile_tag)
                    if proc is not None and proc.stdin is not None:
                        proc.stdin.write(payload)
        finally:
            for proc in writers.values():
                close_ffmpeg_writer(proc)


def write_overlays_global(*, volume: np.ndarray, mask: np.ndarray, plans: Sequence[RenderPlan], fps: float, overlay_tile_writer_limit: int, workers: int, overlay_workers: int, overlay_pending_frames: int, warnings: WarningLog) -> None:
    overlay_plans = [
        plan for plan in plans
        if plan.overlay_path is not None and len(render_plan_frame_indices(plan)) > 0
    ]
    if not overlay_plans:
        return
    total_workers = max(1, int(workers))
    default_overlay_workers = max(1, min(len(overlay_plans), max(1, total_workers // 4), 16))
    nworkers = int(overlay_workers) if int(overlay_workers) > 0 else default_overlay_workers
    # Each concurrent overlay plan can enter its tile-writing phase at the same
    # time.  Keep the plan count within the global tile-writer budget so the
    # per-plan minimum of one writer cannot make the advertised cap leak.
    nworkers = max(1, min(nworkers, len(overlay_plans), int(overlay_tile_writer_limit)))
    per_overlay_frame_workers = max(1, total_workers // max(1, nworkers))
    per_plan_tile_writer_limit = max(1, int(overlay_tile_writer_limit) // max(1, nworkers))
    pending = int(overlay_pending_frames) if int(overlay_pending_frames) > 0 else min(64, max(per_overlay_frame_workers, per_overlay_frame_workers * 2))
    warnings.add("overlay_generation_scheduled", f"overlay_plans={len(overlay_plans)}, overlay_workers={nworkers}, per_overlay_frame_workers={per_overlay_frame_workers}, global_tile_writer_limit={int(overlay_tile_writer_limit)}, per_plan_tile_writer_limit={per_plan_tile_writer_limit}, pending_frames={pending}")
    print(f"Overlay scheduler: variants={len(overlay_plans)}, concurrent_overlay_writers={nworkers}, per-overlay frame workers={per_overlay_frame_workers}, global tile writers<={int(overlay_tile_writer_limit)}, pending_frames={pending}")

    def _write(plan: RenderPlan) -> None:
        write_overlay_for_plan(volume=volume, mask=mask, plan=plan, fps=fps, overlay_tile_writer_limit=int(per_plan_tile_writer_limit), frame_workers=per_overlay_frame_workers, overlay_pending_frames=pending, warnings=warnings)

    with ThreadPoolExecutor(max_workers=nworkers, thread_name_prefix="pretrain-overlay-writer") as executor:
        futures = [executor.submit(_write, plan) for plan in overlay_plans]
        with tqdm(total=len(futures), desc=f"Overlay videos ({len(overlay_plans)} variants)") as pbar:
            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)


def render_all_variants_throughput(*, volume: np.ndarray, mask: np.ndarray, views: Sequence[ViewInfo], angles: Sequence[float], out_dir: Path, stem: str, fps: float, tile_configs: Sequence[TileConfig], save_overlay: bool, overlay_tile_writer_limit: int, png_compression: int, imgsz: int, warnings: WarningLog, workers: int, render_queue_depth: int, overlay_workers: int, overlay_pending_frames: int, tile_task_chunk: int = 1, label_enabled: bool = True, image_format: str = "png") -> List[Dict[str, object]]:
    plans: List[RenderPlan] = []
    for view in views:
        for angle in angles:
            aff = build_affine(view.src_w, view.src_h, float(angle), view.pad_mode, int(imgsz))
            tag = make_tag(view, float(angle))
            plan = build_render_plan(view=view, aff=aff, tag=tag, out_dir=out_dir, stem=stem, tile_configs=tile_configs, save_overlay=bool(save_overlay), imgsz=int(imgsz), label_enabled=bool(label_enabled), image_format=image_format)
            plans.append(plan)

    total_source_frames = int(sum(len(render_plan_frame_indices(plan)) for plan in plans))
    total_output_frames = int(sum(len(render_plan_frame_indices(plan)) * (1 + len(plan.tile_layout)) for plan in plans))
    total_tile_sets = int(sum(len(plan.tile_layout) for plan in plans))
    tile_chunk = max(1, int(tile_task_chunk))
    scheduled_output_tasks = int(sum(
        len(render_plan_frame_indices(plan))
        * (1 + int(math.ceil(len(plan.tile_layout) / float(tile_chunk))) if plan.tile_layout else 1)
        for plan in plans
    ))
    queue_depth = int(render_queue_depth) if int(render_queue_depth) > 0 else max(512, max(1, int(workers)) * 4)

    print("\n=== Throughput-first render scheduler ===")
    print(f"Variants: {len(plans)}; source-frame jobs: {total_source_frames}; output-frame jobs: {total_output_frames}; scheduled output tasks: {scheduled_output_tasks}; tile streams: {total_tile_sets}; workers: {int(workers)}; pending source-frame window: {queue_depth}; tile_task_chunk={tile_chunk}")
    for plan in plans:
        print(f"  queued {plan.tag}: eligible_frames={len(render_plan_frame_indices(plan))}/{plan.view.num_slices}, canvas={plan.aff.canvas_w}x{plan.aff.canvas_h}, output={plan.aff.out_w}x{plan.aff.out_h}, tiles={len(plan.tile_layout)}")
        plan.stats["tile_task_chunk"] = int(tile_chunk)
        plan.stats["source_frame_workers"] = int(workers)
        plan.stats["output_frame_workers"] = int(workers)

    warnings.add("throughput_scheduler", f"variants={len(plans)}, source_frame_jobs={total_source_frames}, output_frame_jobs={total_output_frames}, scheduled_output_tasks={scheduled_output_tasks}, frame_workers={int(workers)}, queue_depth={queue_depth}, tile_streams={total_tile_sets}, tile_task_chunk={tile_chunk}")
    render_primary_outputs_global(volume=volume, mask=mask, plans=plans, png_compression=int(png_compression), warnings=warnings, workers=max(1, int(workers)), max_pending=int(queue_depth), tile_task_chunk=int(tile_chunk))

    if bool(save_overlay):
        write_overlays_global(volume=volume, mask=mask, plans=plans, fps=float(fps), overlay_tile_writer_limit=int(overlay_tile_writer_limit), workers=max(1, int(workers)), overlay_workers=int(overlay_workers), overlay_pending_frames=int(overlay_pending_frames), warnings=warnings)

    return [dict(plan.stats) for plan in plans]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    path: Path,
    *,
    command: str,
    src: SourceVolume,
    out_dir: Path,
    source_shape: Tuple[int, int, int],
    processing_shape: Tuple[int, int, int],
    effective_volume_class: str,
    label_enabled: bool,
    views: Sequence[ViewInfo],
    tile_configs: Sequence[TileConfig],
    smoothing_stats: Sequence[Dict[str, int | float]],
    nrrd_paths: Sequence[Path],
    voxel_initial: Optional[int],
    voxel_final: Optional[int],
    render_stats: Sequence[Dict[str, object]],
    warnings: WarningLog,
    workers: int,
) -> Path:
    t, h, w = source_shape
    pt, ph, pw = processing_shape
    lines: List[str] = []
    lines.append(f"Specification version: {PIPELINE_SPEC_VERSION}")
    lines.append(f"Command: {command}")
    lines.append(f"Input directory: {src.input_dir}")
    if src.video_path is not None:
        lines.append(f"Input video: {src.video_path}")
    if src.segmentation_nrrd_path is not None:
        lines.append(f"Input NRRD segmentation: {src.segmentation_nrrd_path}")
    lines.append(f"Output directory: {out_dir}")
    lines.append(f"Source dimensions (X, Y, t) before cubic resizing: ({int(w)}, {int(h)}, {int(t)})")
    lines.append(f"Processing dimensions (X, Y, t): ({int(pw)}, {int(ph)}, {int(pt)})")
    lines.append(f"Source frame count: {int(t)}")
    lines.append(f"Processing frame count: {int(pt)}")
    lines.append(f"Input kind: {src.kind}")
    lines.append(f"Detected volume class: {src.volume_class}")
    lines.append(f"Effective volume class: {effective_volume_class}")
    lines.append(f"Label source: {src.label_source}")
    lines.append(f"Label operations enabled: {bool(label_enabled)}")
    if src.input_start_index is not None:
        lines.append(f"Detected input start index: {int(src.input_start_index)}")
    if src.encoded_indices:
        encoded = list(src.encoded_indices)
        if len(encoded) <= 24:
            lines.append(f"Encoded input indices: {encoded}")
        else:
            lines.append(f"Encoded input indices: count={len(encoded)}, first={encoded[:8]}, last={encoded[-8:]}")
    lines.append(f"FPS for overlay videos: {src.fps}")
    lines.append(f"Workers: {int(workers)} (0 defaults to the process CPU-affinity count)")
    lines.append(
        "Shared forward sampling: TTA hardware-linear radial/intensity policy and "
        "TTA affine stage; categorical ground truth uses nearest sampling with the "
        "TTA tilted-stack threshold"
    )
    lines.append(f"NRRD export layout: {NRRD_AXIS_ORDER_NOTE}; space={NRRD_SPACE}; space_directions=identity")
    lines.append("Rotation-angle augmentation: removed in v3.0.0_SLURM")
    lines.append("Active views:")
    for v in views:
        extra = ""
        if v.family == "radial":
            spacing = float(v.azimuths_deg[1] - v.azimuths_deg[0]) if len(v.azimuths_deg) > 1 else 0.0
            extra = (
                f", azimuth frames={len(v.azimuths_deg)}, azimuth_step={spacing:g}, "
                f"diameter={v.diameter}, image_sampling={shared_geometry.RADIAL_FILTER_MODE}, "
                "categorical_sampling=nearest"
            )
        if v.family == "tilted_transverse":
            extra = f", direction={v.tilt_direction}, signed_tilt={v.tilt_angle_deg:g}"
        lines.append(f"  {v.display_name}: frames={int(v.num_slices)}, source_plane=({int(v.src_w)}x{int(v.src_h)}){extra}")
    if tile_configs:
        lines.append("Tile configurations:")
        for cfg in tile_configs:
            lines.append(f"  {cfg.config_id}: tile_size={cfg.tile_size}, tile_stride={cfg.tile_stride}")
    else:
        lines.append("Tile configurations: disabled")

    if voxel_initial is not None or voxel_final is not None:
        lines.append("")
        lines.append("--voxel_volume:")
        if voxel_initial is not None:
            lines.append(f"  initial_rasterized_transverse_mask: {int(voxel_initial)}")
        if voxel_final is not None:
            lines.append(f"  final_mask_after_gaussian_smoothing: {int(voxel_final)}")

    lines.append("")
    if smoothing_stats:
        lines.append("Gaussian smoothing:")
        for st in smoothing_stats:
            lines.append(
                f"  pass {int(st.get('pass_index', 0))}: sigma={float(st.get('sigma', 0.0)):g}, "
                f"foreground_before={int(st.get('foreground_before', 0))}, "
                f"foreground_after={int(st.get('foreground_after', 0))}, "
                f"delta_voxels={int(st.get('delta_voxels', 0))}"
            )
    else:
        lines.append("Gaussian smoothing: disabled, not requested, or unavailable for this input class")

    if nrrd_paths:
        lines.append("")
        lines.append("NRRD outputs:")
        for p in nrrd_paths:
            lines.append(f"  {p}")

    lines.append("")
    lines.append("Rendered output sets:")
    for st in render_stats:
        lines.append(
            f"  {st.get('tag')}: view={st.get('view')}, frames={st.get('frames')}, "
            f"full_output_size={st.get('full_output_size')}, tiles={len(st.get('tiles', []))}, "
            f"label_enabled={st.get('label_enabled')}"
        )

    lines.append("")
    lines.extend(warnings.summary_lines())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path

# ---------------------------------------------------------------------------
# Multi-volume planning, filtering, splitting, augmentation, and rendering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolumeInputSpec:
    input_dir: Path
    stem: str
    kind: str  # sequence or video
    image_paths_by_index: Dict[int, Path]
    video_path: Optional[Path]
    labels_by_index: Dict[int, Path]
    segmentation_nrrd_path: Optional[Path]
    volume_class: str  # fully_labeled, partially_labeled, unlabeled
    label_source: str  # yolo, nrrd, none
    input_start_index: Optional[int]
    encoded_indices: Tuple[int, ...]
    frame_count_hint: Optional[int] = None
    fps_hint: Optional[float] = None
    yolo_polygons_by_index: Dict[int, YoloPolygons] = field(default_factory=dict)


@dataclass
class PreparedVolume:
    src: SourceVolume
    source_shape: Tuple[int, int, int]
    processing_shape: Tuple[int, int, int]
    effective_volume_class: str
    label_enabled: bool
    # One source-frame state before smoothing/resizing: explicit foreground,
    # explicit empty/background, or no YOLO label file.
    annotation_states: Tuple[int, ...]
    save_overlay: bool
    volume_for_render: np.ndarray
    mask_for_render: np.ndarray
    views: List[ViewInfo]
    plans: List[RenderPlan]
    smoothing_stats: List[Dict[str, int | float]]
    nrrd_paths: List[Path]
    voxel_initial: Optional[int]
    voxel_final: Optional[int]
    foreground_preservation_stats: Dict[str, int]
    # Blocks backing volume_for_render/mask_for_render for the
    # process pool, plus every block this volume owns (released at finalize).
    volume_render_block: Optional[SharedBlock] = None
    mask_render_block: Optional[SharedBlock] = None
    shm_blocks: List[SharedBlock] = field(default_factory=list)
    v18_mode: bool = False


@dataclass
class VolumeSummaryRecord:
    stem: str
    input_kind: str
    volume_class: str
    effective_volume_class: str
    label_source: str
    label_enabled: bool
    source_shape: Tuple[int, int, int]
    processing_shape: Tuple[int, int, int]
    fps: float
    input_start_index: Optional[int]
    encoded_indices: Tuple[int, ...]
    annotation_state_counts: Dict[str, int]
    foreground_preservation_stats: Dict[str, int]
    voxel_initial: Optional[int]
    voxel_final: Optional[int]
    smoothing_stats: List[Dict[str, int | float]]
    nrrd_paths: List[Path]
    views: List[ViewInfo]
    tile_configs: List[TileConfig]
    render_stats: List[Dict[str, object]]
    candidates_total: int = 0
    candidates_retained: int = 0
    augmented_candidates_planned: int = 0
    augmented_candidates_retained: int = 0
    candidates_written: int = 0


def _v18_view_manifest_record(view: ViewInfo) -> Dict[str, object]:
    shared = view.shared_view
    if shared is None:
        return {
            "physical_view_id": str(view.name),
            "family": str(view.family),
            "num_slices": int(view.num_slices),
        }
    return {
        "physical_view_id": str(shared_geometry.physical_view_name(shared)),
        "runtime_view_id": str(shared.name),
        "family": str(shared.family),
        "summary_family": str(shared.summary_family),
        "base_view": str(
            shared.radial_base_view or shared.tilt_base_view or shared_geometry.physical_view_name(shared)
        ),
        "num_slices": int(shared.num_slices),
        "source_raster_h_w": [int(shared.src_h), int(shared.src_w)],
        "pad_mode": str(shared.pad_mode),
        "azimuth_angles_deg": [float(value) for value in shared.azimuths_deg],
        "tilt_angle_deg": float(shared.tilt_angle_deg),
        "tilt_direction": str(shared.tilt_direction),
        "radial_tilted_source": bool(shared.radial_tilted_source),
    }


_V18_IDENTITY_STAT_FIELDS = (
    "size_bytes",
    "modified_time_ns",
    "change_or_creation_time_ns",
    "device",
    "file_id",
)


def _v18_input_file_identity(
    path_value: Optional[Path],
    *,
    allow_missing: bool = False,
) -> Optional[Dict[str, object]]:
    if path_value is None:
        return None
    resolved = Path(path_value).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        if allow_missing:
            return {"path": str(resolved), "exists": False}
        raise
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
        "change_or_creation_time_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "file_id": int(stat.st_ino),
    }


def capture_v18_pta_input_identities(
    specs: Sequence[VolumeInputSpec],
    *,
    allow_missing: bool = False,
) -> List[Dict[str, object]]:
    """Snapshot every discovered PTA source artifact before dataset generation."""

    records: List[Dict[str, object]] = []
    for spec in specs:
        records.append({
            "stem": str(spec.stem),
            "kind": str(spec.kind),
            "volume_class": str(spec.volume_class),
            "label_source": str(spec.label_source),
            "video": _v18_input_file_identity(
                spec.video_path, allow_missing=allow_missing
            ),
            "image_paths": [
                _v18_input_file_identity(path_value, allow_missing=allow_missing)
                for _, path_value in sorted(spec.image_paths_by_index.items())
            ],
            "label_paths": [
                _v18_input_file_identity(path_value, allow_missing=allow_missing)
                for _, path_value in sorted(spec.labels_by_index.items())
            ],
            "segmentation_nrrd": _v18_input_file_identity(
                spec.segmentation_nrrd_path,
                allow_missing=allow_missing,
            ),
            "encoded_indices": [int(value) for value in spec.encoded_indices],
        })
    return records


def assert_v18_pta_inputs_unchanged(
    records: Sequence[Mapping[str, object]],
) -> None:
    """Refuse a successful manifest when any discovered source changed mid-run."""

    for source in records:
        identities: List[Mapping[str, object]] = []
        for key in ("video", "segmentation_nrrd"):
            identity = source.get(key)
            if isinstance(identity, Mapping):
                identities.append(identity)
        for key in ("image_paths", "label_paths"):
            for identity in source.get(key, ()) or ():
                if isinstance(identity, Mapping):
                    identities.append(identity)
        for identity in identities:
            try:
                current = _v18_input_file_identity(Path(str(identity["path"])))
            except OSError as exc:
                raise RuntimeError(
                    "PTA input artifact changed during execution; refusing a complete "
                    f"manifest: path={identity.get('path')}, unavailable={exc}"
                ) from exc
            assert current is not None
            changed = [
                field
                for field in _V18_IDENTITY_STAT_FIELDS
                if int(current[field]) != int(identity[field])
            ]
            if changed:
                raise RuntimeError(
                    "PTA input artifact changed during execution; refusing a complete "
                    f"manifest: path={identity['path']}, fields={changed}"
                )


def write_v18_pta_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    cli_argv: Sequence[str],
    specs: Sequence[VolumeInputSpec],
    records: Sequence[VolumeSummaryRecord],
    channel_variants: Sequence[ChannelVariant],
    tile_configs: Sequence[TileConfig],
    augmentation_stats: AugmentationStats,
    total_written: int,
    input_identities: Optional[Sequence[Mapping[str, object]]] = None,
    render_backend: str = "cpu",
    workers: int = 1,
    frame_workers: int = 1,
    planning_workers: int = 1,
    topology_summary: str = "",
    summary_path: Optional[Path] = None,
    voxel_report_path: Optional[Path] = None,
    dataset_yaml_path: Optional[Path] = None,
    publication_integrity: Optional[Mapping[str, object]] = None,
) -> Path:
    """Write the mandatory v18 reproducibility and geometry manifest."""

    config = getattr(args, "_v18_config", None)
    if config is None:
        raise ValueError("v18 PTA manifest requires a resolved PtaConfig")

    source_records = (
        [dict(record) for record in input_identities]
        if input_identities is not None
        else capture_v18_pta_input_identities(specs, allow_missing=True)
    )
    volume_records = [
        {
            "stem": str(record.stem),
            "source_shape_t_y_x": [int(value) for value in record.source_shape],
            "processing_shape_t_y_x": [int(value) for value in record.processing_shape],
            "effective_volume_class": str(record.effective_volume_class),
            "views": [_v18_view_manifest_record(view) for view in record.views],
            "render_plans": [dict(stats) for stats in record.render_stats],
            "gaussian_preprocessing": [dict(stats) for stats in record.smoothing_stats],
            "foreground_anchor_repair": dict(record.foreground_preservation_stats),
            "candidates_total": int(record.candidates_total),
            "candidates_retained": int(record.candidates_retained),
            "candidates_processed": int(record.candidates_written),
        }
        for record in records
    ]
    cuda_intensity_selected = "cuda" in str(
        augmentation_stats.runtime_backend
    ).lower()
    sampling_bindings: Tuple[Tuple[str, str], ...] = (
        (
            (("cuda", "intensity"), ("cpu", "intensity"))
            if cuda_intensity_selected
            else (("cpu", "intensity"),)
        )
        + (("cpu", "categorical_ground_truth"),)
    )
    manifest = {
        "schema": "pta-tta.v18.manifest.1",
        "status": "complete",
        "pipeline_version": SCRIPT_VERSION,
        "mode": "pta",
        "command": [SCRIPT_BASENAME, "--mode", "pta", *list(cli_argv)],
        "determinism_contract": "same v18 version, command, and input identities",
        "coordinate_contract": {
            "source": "gray8_t_y_x",
            "t_axis": "frame_index",
            "physical_spacing": None,
        },
        "resolved_configuration": {
            "accepted_mode_arguments": dict(vars(config.args)),
            "imgsz": int(config.args.imgsz),
            "requested_output_format": str(config.requested_output_format),
            "effective_output_format": str(config.effective_output_format),
            "cartesian_views": list(config.cartesian_views),
            "radial_requests": [
                {
                    "view": str(request.view),
                    "requested_azimuth_angle_deg": (
                        None if request.azimuth_angle is None else float(request.azimuth_angle)
                    ),
                }
                for request in config.radial_requests
            ],
            "tilted_groups": [
                {
                    "views": list(group.views),
                    "tilt_angles_deg": [float(value) for value in group.tilt_angles],
                    "tilt_directions": list(group.tilt_directions),
                }
                for group in config.tilted_groups
            ],
            "in_plane_variants_deg": [0.0],
            "channel_variants": [
                {
                    "format": str(variant.format_token),
                    "direction": str(variant.order_name),
                    "offsets": [int(value) for value in variant.offsets],
                }
                for variant in channel_variants
            ],
            "tiles": [
                {
                    "tile_size": int(tile.tile_size),
                    "tile_stride": int(tile.tile_stride),
                    "config_id": str(tile.config_id),
                }
                for tile in tile_configs
            ],
            "preprocessing": {
                "gaussian_smoothing": bool(config.preprocessing.gaussian_smoothing_enabled),
                "sigma": float(config.preprocessing.gaussian_sigma),
                "passes": int(config.preprocessing.gaussian_passes),
                "boundary": "constant_zero",
                "truncate": 4.0,
                "threshold_each_pass": 0.5,
            },
            "save": list(config.save.tokens),
            "execution": {
                "requested_worker_backend": str(config.args.worker_backend),
                "resolved_render_backend": str(render_backend),
                "workers": int(workers),
                "frame_workers": int(frame_workers),
                "planning_workers": int(planning_workers),
                "pipeline_depth": int(config.args.pipeline_depth),
                "topology_aware": bool(config.args.topology_aware),
                "topology_summary": str(topology_summary),
            },
        },
        "forward_sampling": {
            **forward_sampling_execution_record(
                sampling_bindings
            ),
            "geometry_module": "XTA.geometry",
            "backend": (
                "cuda_intensity_with_cpu_fallback;cpu_categorical"
                if cuda_intensity_selected
                else "cpu"
            ),
            "intensity_radial_filter": str(shared_geometry.RADIAL_FILTER_MODE),
            "intensity_affine_filter": (
                "cuda_grid_sample_bilinear_or_opencv_inter_linear_fallback"
                if cuda_intensity_selected
                else "opencv_inter_linear"
            ),
            "categorical_filter": "nearest_with_tilt_stack_threshold_0.5",
            "radial_channel_boundary": "index_wrap_with_odd_crossing_mirror_u",
            "prediction_interpolation": "not_applicable_to_pta",
        },
        "external_augmentation": {
            "configured": bool(augmentation_stats.configured),
            "path": str(augmentation_stats.path) if augmentation_stats.path is not None else None,
            "sha256": str(augmentation_stats.content_sha256),
            "export": str(augmentation_stats.export_name),
            "execution": str(augmentation_stats.execution_mode),
            "backend": str(augmentation_stats.runtime_backend),
            "outside_shared_builtin_geometry_guarantee": bool(augmentation_stats.configured),
        },
        "inputs": {
            "captured_before_execution": input_identities is not None,
            "artifacts": source_records,
        },
        "volumes": volume_records,
        "outputs": {
            "selected": list(config.save.tokens),
            "paths": {
                "root": str(path.parent.resolve()),
                "manifest": str(path.resolve()),
                "summary": (
                    None if summary_path is None else str(summary_path.resolve())
                ),
                "voxel_volume": (
                    None
                    if voxel_report_path is None
                    else str(voxel_report_path.resolve())
                ),
                "dataset_yaml": (
                    None
                    if dataset_yaml_path is None
                    else str(dataset_yaml_path.resolve())
                ),
                "nrrd": [
                    str(nrrd_path.resolve())
                    for record in records
                    for nrrd_path in record.nrrd_paths
                ],
            },
            "dataset_candidates_processed": int(total_written),
            "image_publication_integrity": dict(publication_integrity or {}),
            "zero_view_success": not any(record.views for record in records),
        },
    }
    return write_json_manifest(path, manifest)


def write_v18_voxel_volume_report(
    path: Path,
    records: Sequence[VolumeSummaryRecord],
) -> Path:
    payload = {
        "schema": "pta.v18.foreground-voxel-count.1",
        "units": "foreground_voxel_count",
        "physical_volume": False,
        "conversion": "users convert voxel counts to their desired units externally",
        "volumes": [
            {
                "stem": str(record.stem),
                "target": "categorical_ground_truth",
                "initial_stage": "rasterized_source_ground_truth_before_preprocessing",
                "initial_foreground_voxel_count": record.voxel_initial,
                "final_stage": "after_gaussian_preprocessing_before_cube_resize_and_anchor_repair",
                "final_foreground_voxel_count": record.voxel_final,
                "available": record.voxel_initial is not None,
            }
            for record in records
        ],
    }
    return write_json_manifest(path, payload)


def split_final_underscore_index(path: Path) -> Tuple[str, Optional[int]]:
    """Return (volume_prefix, numeric_index) for v3 final `_NNNN` suffix names."""
    m = re.match(r"^(.*)_(\d+)$", path.stem)
    if not m:
        return path.stem, None
    return m.group(1), int(m.group(2))


def group_indexed_files_by_volume(paths: Sequence[Path], *, kind: str) -> Dict[str, Dict[int, Path]]:
    groups: Dict[str, Dict[int, Path]] = defaultdict(dict)
    duplicates: Dict[Tuple[str, int], List[Path]] = defaultdict(list)
    unindexed: List[Path] = []
    for path in paths:
        stem, idx = split_final_underscore_index(path)
        if idx is None:
            unindexed.append(path)
            continue
        key = (stem, int(idx))
        if int(idx) in groups[stem]:
            duplicates[key].append(groups[stem][int(idx)])
            duplicates[key].append(path)
        else:
            groups[stem][int(idx)] = path
    if unindexed:
        examples = ", ".join(p.name for p in sorted(unindexed)[:12])
        raise ValueError(f"{kind} files must use the final '_NNNN' index suffix; unindexed examples: {examples}")
    if duplicates:
        examples = ", ".join(f"{stem}_{idx}:" + "/".join(p.name for p in paths[:3]) for (stem, idx), paths in list(duplicates.items())[:12])
        raise ValueError(f"Duplicate {kind} frame indices are invalid: {examples}")
    return {stem: dict(idx_map) for stem, idx_map in groups.items()}


def map_sequence_labels_by_frame(image_indices: Sequence[int], labels_by_index: Dict[int, Path]) -> Dict[int, Path]:
    """Map explicit labels while preserving every image as channel context."""
    return {
        frame_i0: labels_by_index[int(encoded_idx)]
        for frame_i0, encoded_idx in enumerate(sorted(int(x) for x in image_indices))
        if int(encoded_idx) in labels_by_index
    }


def effective_class(volume_class: str, *, force: bool) -> str:
    if volume_class == "partially_labeled" and bool(force):
        return "fully_labeled"
    return volume_class


def classify_sequence_spec(image_indices: Sequence[int], labels_by_index: Dict[int, Path], has_nrrd: bool) -> Tuple[str, Optional[int]]:
    image_idxs = sorted(int(x) for x in image_indices)
    image_start = contiguous_start_0_or_1(image_idxs)
    if labels_by_index:
        label_idxs = sorted(int(x) for x in labels_by_index.keys())
        image_set = set(image_idxs)
        label_set = set(label_idxs)
        orphan_labels = sorted(label_set - image_set)
        if orphan_labels:
            raise ValueError(
                "Every YOLO label index must have a matching image index; "
                f"orphan label indices={orphan_labels[:24]}, "
                f"image {describe_index_problem(image_idxs)}, label {describe_index_problem(label_idxs)}"
            )
        if image_start is not None and label_set == image_set:
            return "fully_labeled", image_start
        return "partially_labeled", image_start
    if has_nrrd:
        if image_start is not None:
            return "fully_labeled", image_start
        return "partially_labeled", image_start
    return "unlabeled", image_start


def discover_volume_specs(input_arg: str, *, force: bool, warnings: WarningLog) -> List[VolumeInputSpec]:
    input_path = Path(input_arg).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not input_path.is_dir():
        raise ValueError("--input must be a directory containing supported PTA volume inputs")

    files = [p for p in input_path.iterdir() if p.is_file()]
    images_raw = sorted([p for p in files if p.suffix.lower() in IMAGE_EXTS], key=natural_index_key)
    labels_raw = sorted([p for p in files if p.suffix.lower() == ".txt"], key=natural_index_key)
    videos_raw = sorted([p for p in files if p.suffix.lower() in VIDEO_EXTS])
    nrrds_raw = sorted([p for p in files if p.suffix.lower() in NRRD_EXTS])

    image_groups = group_indexed_files_by_volume(images_raw, kind="image") if images_raw else {}
    label_groups = group_indexed_files_by_volume(labels_raw, kind="label") if labels_raw else {}

    videos_by_stem: Dict[str, Path] = {}
    for video in videos_raw:
        if video.stem in videos_by_stem:
            raise ValueError(f"Multiple video encodings found for volume {video.stem}: {videos_by_stem[video.stem].name}, {video.name}")
        videos_by_stem[video.stem] = video

    nrrds_by_stem: Dict[str, Path] = {}
    for nrrd_path in nrrds_raw:
        if nrrd_path.stem in nrrds_by_stem:
            raise ValueError(f"Multiple NRRD segmentations found for volume {nrrd_path.stem}")
        nrrds_by_stem[nrrd_path.stem] = nrrd_path

    candidate_stems = set(image_groups.keys()) | set(videos_by_stem.keys())
    orphan_label_stems = sorted(set(label_groups.keys()) - candidate_stems)
    orphan_nrrd_stems = sorted(set(nrrds_by_stem.keys()) - candidate_stems)
    if orphan_label_stems:
        raise ValueError(f"Label files exist without matching image sequence/video volume(s): {orphan_label_stems[:12]}")
    if orphan_nrrd_stems:
        raise ValueError(f"NRRD files exist without matching image sequence/video volume(s): {orphan_nrrd_stems[:12]}")
    if not candidate_stems:
        raise ValueError("No image sequence or video volumes found in input directory")

    specs: List[VolumeInputSpec] = []
    for stem in sorted(candidate_stems):
        image_map = dict(image_groups.get(stem, {}))
        video_path = videos_by_stem.get(stem)
        labels_by_index = dict(label_groups.get(stem, {}))
        # H1: parse compact polygon metadata before any source image is decoded.
        yolo_polygons_by_index = {
            int(index): read_yolo_polygons_normalized(path, warnings=warnings)
            for index, path in labels_by_index.items()
        }
        nrrd_path = nrrds_by_stem.get(stem)
        if image_map and video_path is not None:
            raise ValueError(f"Volume {stem} has both an image sequence and a video; choose one encoding per volume")
        if labels_by_index and nrrd_path is not None:
            raise ValueError(f"Volume {stem} has both YOLO labels and an NRRD segmentation; choose one label source per volume")

        if video_path is not None:
            info = ffprobe_info(video_path)
            frame_count = int(info["num_frames"])
            fps = float(info["fps"])
            input_start_index: Optional[int] = None
            encoded_indices: Tuple[int, ...]
            if labels_by_index:
                label_indices = sorted(int(x) for x in labels_by_index.keys())
                input_start_index = contiguous_start_0_or_1(label_indices)
                if input_start_index is None:
                    raise ValueError(f"Video labels for {stem} must be contiguous and start at 0 or 1; {describe_index_problem(label_indices)}")
                expected = list(range(int(input_start_index), int(input_start_index) + frame_count))
                if label_indices != expected:
                    raise ValueError(
                        f"Video Fully Labeled Volume {stem} requires label count and indices to map exactly to all video frames; "
                        f"frames={frame_count}, labels={describe_index_problem(label_indices)}"
                    )
                volume_class = "fully_labeled"
                label_source = "yolo"
                encoded_indices = tuple(label_indices)
            elif nrrd_path is not None:
                volume_class = "fully_labeled"
                label_source = "nrrd"
                encoded_indices = tuple(range(frame_count))
            else:
                volume_class = "unlabeled"
                label_source = "none"
                encoded_indices = tuple(range(frame_count))
            specs.append(VolumeInputSpec(
                input_dir=input_path,
                stem=stem,
                kind="video",
                image_paths_by_index={},
                video_path=video_path,
                labels_by_index=labels_by_index,
                segmentation_nrrd_path=nrrd_path,
                volume_class=volume_class,
                label_source=label_source,
                input_start_index=input_start_index,
                encoded_indices=encoded_indices,
                frame_count_hint=frame_count,
                fps_hint=fps,
                yolo_polygons_by_index=yolo_polygons_by_index,
            ))
            continue

        image_indices = sorted(int(x) for x in image_map.keys())
        if not image_indices:
            raise ValueError(f"Volume {stem} has no indexed image frames")
        volume_class, input_start_index = classify_sequence_spec(image_indices, labels_by_index, nrrd_path is not None)
        if volume_class == "partially_labeled":
            reasons: List[str] = []
            if input_start_index is None:
                reasons.append("image indices are not contiguous from 0 or 1")
            if labels_by_index:
                unlabeled_count = len(set(image_indices) - set(int(x) for x in labels_by_index.keys()))
                if unlabeled_count:
                    reasons.append(f"{unlabeled_count} image slice(s) have no YOLO label file")
            warnings.add("partial_volume_detected", f"{stem}: " + ("; ".join(reasons) or "partial annotation coverage"))
        if volume_class == "unlabeled" and input_start_index is None:
            warnings.add("unlabeled_sequence_noncontiguous_indices", f"{stem}: {describe_index_problem(image_indices)}")
        label_source = "yolo" if labels_by_index else ("nrrd" if nrrd_path is not None else "none")
        specs.append(VolumeInputSpec(
            input_dir=input_path,
            stem=stem,
            kind="sequence",
            image_paths_by_index=image_map,
            video_path=None,
            labels_by_index=labels_by_index,
            segmentation_nrrd_path=nrrd_path,
            volume_class=volume_class,
            label_source=label_source,
            input_start_index=input_start_index,
            encoded_indices=tuple(image_indices),
            frame_count_hint=len(image_indices),
            fps_hint=1.0,
            yolo_polygons_by_index=yolo_polygons_by_index,
        ))

    raw_classes = {spec.volume_class for spec in specs}
    resolved_classes = {effective_class(spec.volume_class, force=force) for spec in specs}
    if len(resolved_classes) != 1:
        detail = ", ".join(f"{spec.stem}:{spec.volume_class}" for spec in specs)
        raise ValueError(
            "All volumes in the input directory must resolve to the same volume type within a single run. "
            f"Resolved classes={sorted(resolved_classes)}; volumes={detail}"
        )
    if len(raw_classes) != 1 and bool(force):
        warnings.add("mixed_raw_volume_classes_resolved_by_force", ", ".join(f"{spec.stem}:{spec.volume_class}" for spec in specs))
    return specs


def load_source_volume_from_spec(
    spec: VolumeInputSpec,
    *,
    warnings: WarningLog,
    workers: int,
    allocator: Optional[ArrayAllocator] = None,
    jpeg_decode_backend: str = "auto",
    jpeg_batch_size: int = 64,
    jpeg_device_ids: Sequence[int] = (),
    jpeg_cpu_sets: Sequence[Sequence[int]] = (),
    load_cpu_order: Sequence[int] = (),
) -> SourceVolume:
    if spec.kind == "video":
        if spec.video_path is None:
            raise RuntimeError(f"Internal error: video spec for {spec.stem} has no video path")
        volume, fps, volume_block = decode_video_gray8_to_memory(spec.video_path, warnings=warnings, allocator=allocator)
        frame_count = int(volume.shape[0])
        labels_by_frame: Dict[int, Path] = {}
        polygons_by_frame: Dict[int, YoloPolygons] = {}
        mask_volume: Optional[np.ndarray] = None
        if spec.label_source == "yolo":
            if spec.input_start_index is None:
                raise RuntimeError(f"Internal error: video labels for {spec.stem} have no start index")
            labels_by_frame = {int(idx) - int(spec.input_start_index): path for idx, path in spec.labels_by_index.items()}
            polygons_by_frame = {
                int(idx) - int(spec.input_start_index): spec.yolo_polygons_by_index.get(int(idx), tuple())
                for idx in spec.labels_by_index
            }
            if sorted(labels_by_frame.keys()) != list(range(frame_count)):
                raise ValueError(f"Video label frame mapping for {spec.stem} does not cover every decoded frame")
        elif spec.label_source == "nrrd":
            if spec.segmentation_nrrd_path is None:
                raise RuntimeError(f"Internal error: NRRD label source missing for {spec.stem}")
            mask_volume = load_nrrd_mask_for_volume(spec.segmentation_nrrd_path, tuple(volume.shape), warnings=warnings)
        return SourceVolume(
            input_dir=spec.input_dir,
            stem=spec.stem,
            kind="video",
            image_paths=[],
            video_path=spec.video_path,
            labels_by_frame=labels_by_frame,
            segmentation_nrrd_path=spec.segmentation_nrrd_path,
            mask_volume=mask_volume,
            volume_class=spec.volume_class,
            label_source=spec.label_source,
            input_start_index=spec.input_start_index,
            encoded_indices=spec.encoded_indices,
            volume=volume,
            fps=fps,
            yolo_polygons_by_frame=polygons_by_frame,
            volume_block=volume_block,
        )

    ordered_indices = sorted(int(x) for x in spec.image_paths_by_index.keys())
    ordered_images = [spec.image_paths_by_index[int(idx)] for idx in ordered_indices]
    volume, volume_block = load_image_sequence_to_memory(
        ordered_images,
        warnings=warnings,
        workers=workers,
        allocator=allocator,
        jpeg_decode_backend=jpeg_decode_backend,
        jpeg_batch_size=int(jpeg_batch_size),
        jpeg_device_ids=jpeg_device_ids,
        jpeg_cpu_sets=jpeg_cpu_sets,
        load_cpu_order=load_cpu_order,
    )
    labels_by_frame: Dict[int, Path] = {}
    polygons_by_frame: Dict[int, YoloPolygons] = {}
    mask_volume: Optional[np.ndarray] = None
    if spec.label_source == "yolo":
        labels_by_frame = map_sequence_labels_by_frame(ordered_indices, spec.labels_by_index)
        polygons_by_frame = {
            frame_i0: spec.yolo_polygons_by_index.get(int(encoded_idx), tuple())
            for frame_i0, encoded_idx in enumerate(ordered_indices)
            if int(encoded_idx) in spec.labels_by_index
        }
    elif spec.label_source == "nrrd":
        if spec.segmentation_nrrd_path is None:
            raise RuntimeError(f"Internal error: NRRD label source missing for {spec.stem}")
        mask_volume = load_nrrd_mask_for_volume(spec.segmentation_nrrd_path, tuple(volume.shape), warnings=warnings)
    return SourceVolume(
        input_dir=spec.input_dir,
        stem=spec.stem,
        kind="sequence",
        image_paths=ordered_images,
        video_path=None,
        labels_by_frame=labels_by_frame,
        segmentation_nrrd_path=spec.segmentation_nrrd_path,
        mask_volume=mask_volume,
        volume_class=spec.volume_class,
        label_source=spec.label_source,
        input_start_index=spec.input_start_index,
        encoded_indices=spec.encoded_indices,
        volume=volume,
        fps=1.0,
        yolo_polygons_by_frame=polygons_by_frame,
        volume_block=volume_block,
    )


def build_plans_for_views(
    *,
    views: Sequence[ViewInfo],
    out_dir: Path,
    stem: str,
    tile_configs: Sequence[TileConfig],
    save_overlay: bool,
    imgsz: int,
    label_enabled: bool,
    image_format: str = "png",
    channel_variants: Sequence[ChannelVariant] = (DEFAULT_CHANNEL_VARIANT,),
    center_eligible_frame_indices: Optional[Sequence[int]] = None,
    transverse_source_encoded_indices: Sequence[int] = (),
    warnings: Optional[WarningLog] = None,
    publish_images: bool = True,
    publish_labels: bool = True,
) -> List[RenderPlan]:
    """Build plans with partial-center and continuity guards.

    ``center_eligible_frame_indices`` applies only to the native transverse view
    of an unforced Partially Labeled YOLO sequence.  Frames with no label file
    remain resident as image context but cannot become output centers.

    ``transverse_source_encoded_indices`` preserves the original numeric slice
    coordinates for custom C...S... stacks.  Requested indices inside the true
    [minimum, maximum] bounds must exist; requests outside those bounds retain
    edge clamping at the actual volume boundary.
    """
    plans: List[RenderPlan] = []
    normalized_base_centers = (
        None
        if center_eligible_frame_indices is None
        else tuple(sorted({int(x) for x in center_eligible_frame_indices}))
    )
    normalized_encoded = tuple(int(x) for x in transverse_source_encoded_indices)
    if normalized_encoded and any(
        normalized_encoded[i] >= normalized_encoded[i + 1]
        for i in range(len(normalized_encoded) - 1)
    ):
        raise ValueError(
            f"{stem}: transverse encoded image indices must be strictly increasing; "
            f"got {normalized_encoded[:24]}"
        )
    warned_discontinuity_formats: set[str] = set()

    for view in views:
        aff = build_affine(
            view.src_w,
            view.src_h,
            0.0,
            view.pad_mode,
            int(imgsz),
            shared_view=view.shared_view,
        )
        base_tag = make_tag(view, 0.0)
        is_transverse = view.family == "transverse"
        if is_transverse and normalized_base_centers is not None:
            if any(x < 0 or x >= int(view.num_slices) for x in normalized_base_centers):
                raise ValueError(
                    f"{stem}: annotated center index is outside transverse bounds "
                    f"0..{int(view.num_slices)-1}: {normalized_base_centers[:24]}"
                )
        if is_transverse and normalized_encoded and len(normalized_encoded) != int(view.num_slices):
            raise ValueError(
                f"{stem}: transverse encoded-index mapping has {len(normalized_encoded)} entries, "
                f"expected {int(view.num_slices)}"
            )

        for channel_variant in channel_variants:
            tag = make_channel_tag(base_tag, channel_variant)
            eligible: Optional[Tuple[int, ...]] = (
                normalized_base_centers if is_transverse else None
            )
            source_indices_for_plan: Tuple[int, ...] = ()
            skipped_discontinuous: List[Tuple[int, Tuple[int, ...]]] = []

            if is_transverse and channel_variant.kind == "custom" and normalized_encoded:
                source_indices_for_plan = normalized_encoded
                candidate_centers: Sequence[int] = (
                    range(int(view.num_slices)) if eligible is None else eligible
                )
                continuous_centers: List[int] = []
                for frame_idx in candidate_centers:
                    positions, missing = encoded_channel_source_positions(
                        normalized_encoded,
                        int(frame_idx),
                        channel_variant.offsets,
                    )
                    if positions is None:
                        skipped_discontinuous.append((int(frame_idx), tuple(int(x) for x in missing)))
                    else:
                        continuous_centers.append(int(frame_idx))
                eligible = tuple(continuous_centers)

                warning_key = str(channel_variant.format_token)
                if skipped_discontinuous and warning_key not in warned_discontinuity_formats:
                    warned_discontinuity_formats.add(warning_key)
                    examples = [
                        f"center={int(normalized_encoded[frame_idx])}, missing={list(missing)}"
                        for frame_idx, missing in skipped_discontinuous[:8]
                    ]
                    msg = (
                        f"{stem} {channel_variant.format_token}: skipped "
                        f"{len(skipped_discontinuous)} annotated center slice(s) because required "
                        f"in-volume channel indices are missing across a discontinuity; "
                        f"examples: {'; '.join(examples)}. Requests beyond actual volume edges "
                        f"remain edge-clamped."
                    )
                    if warnings is not None:
                        warnings.add("channel_stack_discontinuity_centers_skipped", msg)
                    print(f"WARNING: {msg}", file=sys.stderr)

            plan = build_render_plan(
                view=view,
                aff=aff,
                tag=tag,
                out_dir=out_dir,
                stem=stem,
                tile_configs=tile_configs,
                save_overlay=bool(save_overlay),
                imgsz=int(imgsz),
                label_enabled=bool(label_enabled),
                image_format=image_format,
                channel_variant=channel_variant,
                eligible_frame_indices=eligible,
                source_encoded_indices=source_indices_for_plan,
                publish_images=bool(publish_images),
                publish_labels=bool(publish_labels),
            )
            base_count = (
                len(normalized_base_centers)
                if is_transverse and normalized_base_centers is not None
                else int(view.num_slices)
            )
            final_count = len(render_plan_frame_indices(plan))
            plan.stats["annotated_center_frames"] = int(base_count)
            plan.stats["skipped_unannotated_centers"] = (
                int(view.num_slices) - int(base_count)
                if is_transverse and normalized_base_centers is not None
                else 0
            )
            plan.stats["skipped_discontinuous_centers"] = int(len(skipped_discontinuous))
            plan.stats["eligible_center_frames"] = int(final_count)
            plans.append(plan)
    return plans


def prepare_loaded_source(
    src: SourceVolume,
    *,
    args: argparse.Namespace,
    warnings: WarningLog,
    workers: int,
    out_dir: Path,
    tile_configs: Sequence[TileConfig],
    channel_variants: Sequence[ChannelVariant],
    requested_tilt_angles: Sequence[float],
    requested_tilt_directions: Sequence[str],
    write_side_effects: bool,
    allocator: Optional[ArrayAllocator] = None,
) -> PreparedVolume:
    source_shape = tuple(int(x) for x in src.volume.shape)
    effective_volume_class = effective_class(src.volume_class, force=bool(args.force))
    source_encoded_gaps = bool(
        src.kind == "sequence"
        and len(src.encoded_indices) > 1
        and any(
            int(src.encoded_indices[index + 1]) != int(src.encoded_indices[index]) + 1
            for index in range(len(src.encoded_indices) - 1)
        )
    )
    if src.volume_class == "partially_labeled" and bool(args.force):
        warnings.add(
            "partial_volume_forced_as_fully_labeled",
            f"{src.stem}: --force supplied; complete contiguous volumes may use 3D views, "
            "but encoded slice gaps remain ineligible",
        )

    label_enabled = src.label_source in {"yolo", "nrrd"} and src.volume_class != "unlabeled"
    save_overlay = bool(args.save_overlay)
    if save_overlay and not label_enabled:
        warnings.add("overlay_disabled_for_unlabeled_volume", f"{src.stem}: --save overlay requires labels or an NRRD segmentation")
        save_overlay = False
    if bool(args.voxel_volume) and not label_enabled:
        warnings.add("voxel_volume_disabled_for_unlabeled_volume", f"{src.stem}: --save voxel_volume requires labels or an NRRD segmentation")
    if bool(args.save_nrrd) and not label_enabled:
        warnings.add("save_nrrd_disabled_for_unlabeled_volume", f"{src.stem}: --save nrrd requires labels or an NRRD segmentation")

    mask_block: Optional[SharedBlock] = None
    if src.label_source == "yolo":
        print(f"Rasterizing YOLO labels for {src.stem} (t,Y,X)={source_shape}")
        mask, mask_block = rasterize_yolo_labels(src, warnings=warnings, workers=workers, allocator=allocator)
    elif src.label_source == "nrrd":
        if src.mask_volume is None:
            raise RuntimeError("Internal error: SourceVolume label_source is nrrd but mask_volume is missing")
        mask = np.ascontiguousarray(src.mask_volume, dtype=np.uint8)
        warnings.add("mask_volume_loaded_in_memory", f"{src.stem}: {mask.shape}, {mask.nbytes / GIB:.2f} GiB")
    else:
        if allocator is not None:
            mask, mask_block = allocator(tuple(source_shape))
        else:
            mask = np.zeros_like(src.volume, dtype=np.uint8)
        warnings.add("label_operations_disabled_for_unlabeled_volume", f"{src.stem}: blank in-memory mask is used only to satisfy image reslicing code paths")

    if tuple(int(x) for x in mask.shape) != source_shape:
        raise ValueError(f"Image and mask volumes must match before preprocessing for {src.stem}: image={source_shape}, mask={mask.shape}")

    annotation_states = derive_annotation_states(src, mask)
    state_counts = annotation_state_counts(annotation_states)
    unannotated_count = int(state_counts["unannotated"])
    if (
        src.volume_class == "partially_labeled"
        and src.label_source == "yolo"
        and unannotated_count > 0
        and effective_volume_class == "partially_labeled"
    ):
        message = (
            f"{src.stem}: {unannotated_count} image slice(s) have no matching YOLO label file. "
            "They may be used as non-center image channels, but they will never be emitted as "
            "foreground/background center samples."
        )
        warnings.add("unannotated_center_slices_excluded", message)
        print(f"WARNING: {message}", file=sys.stderr)
    elif (
        src.volume_class == "partially_labeled"
        and bool(args.force)
        and unannotated_count > 0
    ):
        warnings.add(
            "unannotated_state_overridden_by_force",
            f"{src.stem}: --force treats {unannotated_count} unannotated slice(s) as mask-empty during full-volume transforms",
        )

    foreground_anchors, source_anchor_seeds = collect_foreground_slice_anchors(
        src,
        mask,
        annotation_states,
        warnings=warnings,
    ) if label_enabled else (tuple(), 0)
    foreground_preservation_stats: Dict[str, int] = {
        "input_foreground_transverse_slices": int(len(foreground_anchors)),
        "source_polygon_anchor_seeds": int(source_anchor_seeds),
        "smoothing_anchor_repairs": 0,
        "processed_anchor_repairs": 0,
        "guaranteed_output_foreground_transverse_slices": int(len(foreground_anchors)),
        "classified_output_foreground_transverse_slices": 0,
        "retained_output_foreground_transverse_slices": 0,
    }

    voxel_initial = int(np.count_nonzero(mask)) if (label_enabled and bool(args.voxel_volume)) else None

    nrrd_paths: List[Path] = []
    if write_side_effects and label_enabled and bool(args.save_nrrd):
        nrrd_paths.append(write_nrrd(mask, out_dir / "nrrds" / f"{src.stem}_Pass_0.nrrd"))

    smoothing_stats: List[Dict[str, int | float]] = []
    gaussian_requested = float(args.gaussian_smoothing) > 0.0 and int(args.gaussian_smoothing_passes) > 0
    if (
        label_enabled
        and effective_volume_class == "fully_labeled"
        and gaussian_requested
        and not source_encoded_gaps
    ):
        def _save_gaussian_pass_nrrd(pass_idx: int, current_mask: np.ndarray) -> None:
            if write_side_effects and bool(args.save_nrrd):
                nrrd_paths.append(write_nrrd(current_mask, out_dir / "nrrds" / f"{src.stem}_Pass_{int(pass_idx)}.nrrd"))

        smoothing_stats = apply_gaussian_smoothing(
            mask,
            sigma=float(args.gaussian_smoothing),
            passes=int(args.gaussian_smoothing_passes),
            warnings=warnings,
            after_pass=_save_gaussian_pass_nrrd,
        )
    elif label_enabled and source_encoded_gaps and gaussian_requested:
        warnings.add(
            "gaussian_smoothing_disabled_for_encoded_gaps",
            f"{src.stem}: numeric source-index gaps do not define adjacent 3D samples",
        )
    elif label_enabled and effective_volume_class == "partially_labeled" and gaussian_requested:
        warnings.add("gaussian_smoothing_disabled_for_partial_volume", f"{src.stem}: Partial volumes disable 3D label transformations unless --force is supplied")
    elif not label_enabled and gaussian_requested:
        warnings.add("gaussian_smoothing_disabled_for_unlabeled_volume", f"{src.stem}: Unlabeled volumes disable label operations")

    voxel_final = int(np.count_nonzero(mask)) if (label_enabled and bool(args.voxel_volume)) else None

    cube_resize_eligible = bool(
        effective_volume_class in {"fully_labeled", "unlabeled"}
        and not source_encoded_gaps
    )
    if cube_resize_eligible:
        volume_for_render, mask_for_render, processing_shape = resize_to_approximately_cube(
            src.volume,
            mask,
            warnings=warnings,
            work_dir=out_dir / ".v18_work" / src.stem,
            workers=max(1, int(workers)),
        )
    else:
        volume_for_render = src.volume
        mask_for_render = mask
        processing_shape = source_shape
        if source_encoded_gaps:
            warnings.add(
                "cubic_resize_disabled_for_encoded_gaps",
                f"{src.stem}: numeric source-index gaps retain native depth; only explicitly "
                "enabled transverse/tiled-transverse outputs are eligible",
            )
        else:
            warnings.add(
                "cubic_resize_disabled_for_partial_volume",
                f"{src.stem}: partial volumes retain native depth and only explicitly enabled "
                "transverse/tiled-transverse outputs are eligible unless --force is supplied",
            )

    if cube_resize_eligible and int(processing_shape[0]) < int(source_shape[0]):
        raise RuntimeError(
            f"C1 preservation invariant failed for {src.stem}: processed transverse depth "
            f"{processing_shape[0]} is smaller than input depth {source_shape[0]}"
        )

    output_patch_radius = 0
    if int(args.imgsz) > 0:
        output_patch_radius = int(math.ceil(max(
            float(processing_shape[1]) / float(args.imgsz),
            float(processing_shape[2]) / float(args.imgsz),
        )))
    guaranteed_foreground_slices, processed_repairs = reassert_foreground_slice_anchors(
        mask_for_render,
        foreground_anchors,
        source_shape=source_shape,
        patch_radius=output_patch_radius,
    )
    foreground_preservation_stats["processed_anchor_repairs"] = int(processed_repairs)
    foreground_preservation_stats["guaranteed_output_foreground_transverse_slices"] = int(guaranteed_foreground_slices)
    if int(guaranteed_foreground_slices) < int(len(foreground_anchors)):
        raise RuntimeError(
            f"Foreground transverse invariant failed for {src.stem}: input={len(foreground_anchors)}, "
            f"guaranteed output={guaranteed_foreground_slices}"
        )

    # The render arrays must live in shared memory for the
    # persistent process pool.  Pass-through when the loader already allocated
    # them there (no resize); copy once into fresh blocks otherwise.
    volume_for_render, volume_render_block = ensure_shared_uint8(
        volume_for_render, src.volume_block if volume_for_render is src.volume else None, allocator)
    mask_for_render, mask_render_block = ensure_shared_uint8(
        mask_for_render, mask_block if mask_for_render is mask else None, allocator)
    owned_blocks: List[SharedBlock] = []
    for candidate_block in (src.volume_block, mask_block, volume_render_block, mask_render_block):
        if candidate_block is not None and all(candidate_block is not existing for existing in owned_blocks):
            owned_blocks.append(candidate_block)

    t_dim, h, w = (int(processing_shape[0]), int(processing_shape[1]), int(processing_shape[2]))

    v18_config = getattr(args, "_v18_config", None)
    if v18_config is None:
        raise RuntimeError("PTA volume preparation requires a resolved v18 configuration")
    encoded_gaps = bool(source_encoded_gaps)
    native_only = bool(effective_volume_class == "partially_labeled" or encoded_gaps)
    effective_config = v18_config
    if native_only:
        allowed_cartesian = tuple(
            name for name in v18_config.cartesian_views if str(name) == "transverse"
        )
        disabled = [
            *(f"cartesian:{name}" for name in v18_config.cartesian_views if str(name) != "transverse"),
            *(f"radial:{request.view}" for request in v18_config.radial_requests),
            *(f"tilted:{','.join(group.views)}" for group in v18_config.tilted_groups),
        ]
        if disabled:
            reason = (
                "unforced partial-label eligibility"
                if effective_volume_class == "partially_labeled"
                else "encoded slice gaps that cannot define adjacent 3D samples"
            )
            warnings.add(
                "partial_volume_3d_views_disabled",
                f"{src.stem}: {reason}; disabled {', '.join(disabled)}",
            )
        effective_config = replace(
            v18_config,
            cartesian_views=allowed_cartesian,
            radial_requests=(),
            tilted_groups=(),
        )
    views, compiled_views = compile_v18_pta_views(
        t_dim=int(t_dim),
        h=int(h),
        w=int(w),
        config=effective_config,
        radial_native_raster=int(args.imgsz),
    )
    for request, diameter, spacing in zip(
        effective_config.radial_requests,
        compiled_views.radial_diameters,
        compiled_views.radial_azimuth_angles,
    ):
        print(
            f"{src.stem}: shared TTA radial {request.view}, diameter={int(diameter)}, "
            f"azimuth_angle={float(spacing):g} deg, hardware-linear intensity / nearest categorical"
        )
    native_partial_encoded_indices: Tuple[int, ...] = ()
    if (
        src.kind == "sequence"
        and len(src.encoded_indices) == int(t_dim)
    ):
        # Preserve encoded slice coordinates whenever preprocessing retained the
        # depth axis.  This covers partial labels and fully labeled sequences
        # with numeric gaps, including --force.
        native_partial_encoded_indices = tuple(int(x) for x in src.encoded_indices)
    elif (
        src.kind == "sequence"
        and src.volume_class == "partially_labeled"
        and any(variant.kind == "custom" for variant in channel_variants)
    ):
        continuity_message = (
            f"{src.stem}: custom-channel encoded-index continuity cannot be applied after "
            f"depth resizing changed {len(src.encoded_indices)} source slices to {int(t_dim)} "
            "processing slices; this requires --force on a partial volume"
        )
        warnings.add("custom_channel_continuity_unavailable_after_forced_depth_resize", continuity_message)
        print(f"WARNING: {continuity_message}", file=sys.stderr)

    plans = build_plans_for_views(
        views=views,
        out_dir=out_dir,
        stem=src.stem,
        tile_configs=tile_configs,
        save_overlay=save_overlay,
        imgsz=int(args.imgsz),
        label_enabled=bool(label_enabled),
        image_format=str(args.output_format),
        channel_variants=channel_variants,
        center_eligible_frame_indices=(
            tuple(
                idx for idx, state in enumerate(annotation_states)
                if int(state) != ANNOTATION_UNANNOTATED
            )
            if (
                src.volume_class == "partially_labeled"
                and effective_volume_class == "partially_labeled"
                and src.label_source == "yolo"
            )
            else None
        ),
        transverse_source_encoded_indices=native_partial_encoded_indices,
        warnings=warnings,
        publish_images=bool(getattr(args, "save_images", True)),
        publish_labels=bool(getattr(args, "save_labels", True)),
    )

    return PreparedVolume(
        src=src,
        source_shape=source_shape,
        processing_shape=processing_shape,
        effective_volume_class=effective_volume_class,
        label_enabled=bool(label_enabled),
        annotation_states=tuple(int(x) for x in annotation_states),
        save_overlay=bool(save_overlay),
        volume_for_render=volume_for_render,
        mask_for_render=mask_for_render,
        views=list(views),
        plans=list(plans),
        smoothing_stats=smoothing_stats,
        nrrd_paths=nrrd_paths,
        voxel_initial=voxel_initial,
        voxel_final=voxel_final,
        foreground_preservation_stats=foreground_preservation_stats,
        volume_render_block=volume_render_block,
        mask_render_block=mask_render_block,
        shm_blocks=owned_blocks,
        v18_mode=True,
    )


def prepare_volume(
    spec: VolumeInputSpec,
    *,
    args: argparse.Namespace,
    warnings: WarningLog,
    workers: int,
    out_dir: Path,
    tile_configs: Sequence[TileConfig],
    channel_variants: Sequence[ChannelVariant],
    requested_tilt_angles: Sequence[float],
    requested_tilt_directions: Sequence[str],
    write_side_effects: bool,
) -> PreparedVolume:
    """Compatibility wrapper for callers that do not use raw-volume prefetch."""
    src = load_source_volume_from_spec(spec, warnings=warnings, workers=workers)
    return prepare_loaded_source(
        src,
        args=args,
        warnings=warnings,
        workers=workers,
        out_dir=out_dir,
        tile_configs=tile_configs,
        channel_variants=channel_variants,
        requested_tilt_angles=requested_tilt_angles,
        requested_tilt_directions=requested_tilt_directions,
        write_side_effects=write_side_effects,
    )


MASK_POLYGON_FAST_FOREGROUND_MIN_PIXELS = 16


def mask_has_yolo_polygon(mask01: np.ndarray) -> bool:
    m = (np.asarray(mask01, dtype=np.uint8) > 0).astype(np.uint8)
    nonzero = int(np.count_nonzero(m))
    if nonzero == 0:
        return False
    # Masks with a healthy pixel count always survive the contour
    # export; the full findContours/approxPolyDP check is reserved for the
    # degenerate gray zone where a 1-2 px line could still be demoted.
    if nonzero >= MASK_POLYGON_FAST_FOREGROUND_MIN_PIXELS:
        return True
    contours, hierarchy = cv2.findContours(m * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    hierarchy_arr = [None] * len(contours) if hierarchy is None else list(hierarchy[0])
    for cnt, hrow in zip(contours, hierarchy_arr):
        if hrow is not None and int(hrow[3]) >= 0:
            continue
        if cnt is None or len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        if approx is not None and len(approx) >= 3:
            return True
    return False




def _source_frames_for_processed_transverse(prep: PreparedVolume, frame_idx: int) -> range:
    source_t = int(prep.source_shape[0])
    processed_t = int(prep.processing_shape[0])
    if source_t <= 0 or processed_t <= 0:
        return range(0)
    if source_t <= 1 or processed_t <= 1:
        center = 0.0
    else:
        # scipy.ndimage.zoom(grid_mode=False) maps the first/last output
        # centers to the first/last input centers.
        center = float(frame_idx) * float(source_t - 1) / float(processed_t - 1)
    # Cubic resize uses linear z interpolation.  Gaussian support is normally
    # truncated at four sigma; multiplying by passes is intentionally
    # conservative and can retain a few extra backgrounds but cannot discard
    # a foreground that the dense transformed mask would create.
    smooth_halo = 0
    for stats in prep.smoothing_stats:
        smooth_halo += int(math.ceil(4.0 * max(0.0, float(stats.get("sigma", 0.0)))))
    radius = float(smooth_halo) + (0.5 if source_t != processed_t else 0.0)
    lo = max(0, int(math.ceil(center - radius - 1e-9)))
    hi = min(source_t - 1, int(math.floor(center + radius + 1e-9)))
    return range(lo, hi + 1)


def _transverse_polygons_for_processed_frame(prep: PreparedVolume, frame_idx: int) -> YoloPolygons:
    polygons: List[YoloPolygon] = []
    for source_frame in _source_frames_for_processed_transverse(prep, int(frame_idx)):
        polygons.extend(prep.src.yolo_polygons_by_frame.get(int(source_frame), tuple()))
    return tuple(polygons)


def _polygon_area_xy(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(0.5 * sum(
        float(points[i][0]) * float(points[(i + 1) % len(points)][1])
        - float(points[(i + 1) % len(points)][0]) * float(points[i][1])
        for i in range(len(points))
    ))


def _clip_polygon_to_rect(
    points: Sequence[Tuple[float, float]],
    rect: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman clip against an axis-aligned rectangle."""
    x0, y0, x1, y1 = (float(value) for value in rect)
    output = [(float(x), float(y)) for x, y in points]

    def clip_edge(
        vertices: List[Tuple[float, float]],
        inside: Callable[[Tuple[float, float]], bool],
        intersect: Callable[[Tuple[float, float], Tuple[float, float]], Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if not vertices:
            return []
        result: List[Tuple[float, float]] = []
        previous = vertices[-1]
        previous_inside = inside(previous)
        for current in vertices:
            current_inside = inside(current)
            if current_inside != previous_inside:
                result.append(intersect(previous, current))
            if current_inside:
                result.append(current)
            previous, previous_inside = current, current_inside
        return result

    def at_x(a: Tuple[float, float], b: Tuple[float, float], x: float) -> Tuple[float, float]:
        delta = float(b[0] - a[0])
        ratio = 0.0 if abs(delta) < 1e-12 else float(x - a[0]) / delta
        return float(x), float(a[1] + ratio * (b[1] - a[1]))

    def at_y(a: Tuple[float, float], b: Tuple[float, float], y: float) -> Tuple[float, float]:
        delta = float(b[1] - a[1])
        ratio = 0.0 if abs(delta) < 1e-12 else float(y - a[1]) / delta
        return float(a[0] + ratio * (b[0] - a[0])), float(y)

    output = clip_edge(output, lambda p: p[0] >= x0, lambda a, b: at_x(a, b, x0))
    output = clip_edge(output, lambda p: p[0] <= x1, lambda a, b: at_x(a, b, x1))
    output = clip_edge(output, lambda p: p[1] >= y0, lambda a, b: at_y(a, b, y0))
    output = clip_edge(output, lambda p: p[1] <= y1, lambda a, b: at_y(a, b, y1))
    return output


def _polygon_intersects_canvas_tile(
    polygons: Sequence[YoloPolygon],
    *,
    plan: RenderPlan,
    tile: RenderTileItem,
    xy_halo_source: float,
) -> bool:
    if not polygons:
        return False
    matrix = np.asarray(plan.aff.M_src_to_canvas, dtype=np.float64)
    src_w = float(plan.view.src_w)
    src_h = float(plan.view.src_h)
    source_halo = max(0.0, float(xy_halo_source))
    halo_x = source_halo * float(np.linalg.norm(matrix[0, :2]))
    halo_y = source_halo * float(np.linalg.norm(matrix[1, :2]))
    rect = (
        float(tile.x) - halo_x,
        float(tile.y) - halo_y,
        float(tile.x + tile.cfg.tile_size) + halo_x,
        float(tile.y + tile.cfg.tile_size) + halo_y,
    )
    for polygon in polygons:
        points = np.asarray([(x * src_w, y * src_h, 1.0) for x, y in polygon], dtype=np.float64)
        if points.size == 0:
            continue
        canvas = points @ matrix.T
        clipped = _clip_polygon_to_rect(
            [(float(point[0]), float(point[1])) for point in canvas],
            rect,
        )
        # Explicit area threshold avoids classifying a corner/edge touch as a
        # foreground polygon that the YOLO writer could never emit.
        if len(clipped) >= 3 and _polygon_area_xy(clipped) >= 0.5:
            return True
    return False


def classify_original_foregrounds_for_volume(
    prep: PreparedVolume,
    *,
    workers: int,
    warnings: Optional[WarningLog] = None,
) -> Dict[Tuple[str, int, str, int], bool]:
    """Classify copy-0 (original) frames/tiles as foreground/background.

    The Albumentations pipeline is never executed here. Augmented
    copies of a background original are background by the no-mask-synthesis
    invariant enforced at render time; copies of a foreground original are
    presumed foreground for budgeting and reconciled when the render pass
    replays them (a copy that renders background is dropped).
    """
    foregrounds: Dict[Tuple[str, int, str, int], bool] = {}
    if not prep.label_enabled:
        return foregrounds
    jobs = list(iter_render_source_frame_jobs_round_robin(prep.plans))
    if not jobs:
        return foregrounds
    merge_lock = threading.Lock()

    geometry_jobs: List[Tuple[int, int]] = []
    dense_jobs: List[Tuple[int, int]] = []
    # Once Gaussian smoothing has changed mask support, polygon-only geometry
    # is no longer authoritative for the background cap.  Use the transformed
    # dense mask in that case so --background_percent retains its documented behavior.
    use_yolo_geometry = (
        prep.src.label_source == "yolo"
        and bool(prep.src.yolo_polygons_by_frame or prep.src.labels_by_frame)
        and not prep.smoothing_stats
    )
    if prep.src.label_source == "yolo" and prep.smoothing_stats and warnings is not None:
        warnings.add(
            "label_geometry_fallback_after_smoothing",
            f"{prep.src.stem}: transformed dense-mask classification preserves the background cap after Gaussian smoothing",
        )
    for plan_idx, idx in jobs:
        plan = prep.plans[int(plan_idx)]
        if use_yolo_geometry and plan.view.family == "transverse":
            geometry_jobs.append((int(plan_idx), int(idx)))
        else:
            dense_jobs.append((int(plan_idx), int(idx)))

    xy_halo = sum(
        float(math.ceil(4.0 * max(0.0, float(stats.get("sigma", 0.0)))))
        for stats in prep.smoothing_stats
    )
    for plan_idx, idx in geometry_jobs:
        plan = prep.plans[int(plan_idx)]
        polygons = _transverse_polygons_for_processed_frame(prep, int(idx))
        foregrounds[(plan.tag, int(idx), "full", 0)] = bool(polygons)
        for tile in plan.tile_layout:
            foregrounds[(plan.tag, int(idx), tile.tile_tag, 0)] = _polygon_intersects_canvas_tile(
                polygons,
                plan=plan,
                tile=tile,
                xy_halo_source=xy_halo,
            )
    if geometry_jobs and warnings is not None:
        warnings.add(
            "label_geometry_foreground_planner",
            f"{prep.src.stem}: {len(geometry_jobs)} transverse plan/frame job(s) classified from predecoded YOLO polygons; "
            f"dense fallback jobs={len(dense_jobs)}",
        )

    def _classify_frame(job_idx: int) -> None:
        plan_idx, idx = dense_jobs[int(job_idx)]
        plan = prep.plans[int(plan_idx)]
        canvas_tiles = tuple(tile for tile in plan.tile_layout if tile.shared_job is None)
        mask_full, mask_canvas = render_plan_frame_mask_source(
            mask=prep.mask_for_render,
            plan=plan,
            idx=int(idx),
            need_canvas=bool(canvas_tiles),
        )
        local: Dict[Tuple[str, int, str, int], bool] = {
            (plan.tag, int(idx), "full", 0): bool(mask_has_yolo_polygon(mask_full)),
        }
        if plan.tile_layout:
            for tile in plan.tile_layout:
                if tile.shared_job is not None and plan.view.shared_view is not None:
                    tile_mask_out = shared_geometry.render_categorical_dense_tile_for_job(
                        prep.mask_for_render,
                        plan.view.shared_view,
                        tile.shared_job,
                        int(idx),
                    )
                else:
                    if mask_canvas is None:
                        raise RuntimeError(
                            f"Tile classification requested without a mask canvas for {plan.tag}"
                        )
                    tile_mask = extract_padded_tile(mask_canvas, tile.x, tile.y, tile.cfg.tile_size)
                    tile_mask_out = resize_centered(tile_mask, tile.out_w, tile.out_h, cv2.INTER_NEAREST)
                local[(plan.tag, int(idx), tile.tile_tag, 0)] = bool(mask_has_yolo_polygon(tile_mask_out))
        with merge_lock:
            foregrounds.update(local)

    parallel_for_indices(
        len(dense_jobs),
        _classify_frame,
        workers=max(1, int(workers)),
        desc=f"Classifying original frames/tiles {prep.src.stem}",
    )
    return foregrounds


def enumerate_candidates_for_volume(prep: PreparedVolume, foregrounds: Optional[Dict[Tuple[str, int, str, int], bool]] = None) -> List[OutputCandidate]:
    fg_map = foregrounds or {}
    candidates: List[OutputCandidate] = []

    def _fg(plan_tag: str, frame_idx: int, item_key: str) -> bool:
        if not prep.label_enabled:
            return True
        return bool(fg_map.get((plan_tag, int(frame_idx), item_key, 0), True))

    def _physical_view_id(plan: RenderPlan) -> str:
        if plan.view.shared_view is not None:
            return str(shared_geometry.physical_view_name(plan.view.shared_view))
        return str(plan.view.name)

    def _presentation_variant_id(plan: RenderPlan) -> str:
        return (
            f"channel:{plan.channel_variant.format_token}:"
            f"{plan.channel_variant.order_name}"
        )

    # Canonical order part 1: full-frame views in v3 view order.
    # Iterate only explicitly eligible centers.
    for plan in prep.plans:
        for frame_idx in render_plan_frame_indices(plan):
            candidates.append(OutputCandidate(
                order=-1,
                volume_name=prep.src.stem,
                parent_view_tag=plan.tag,
                output_tag=plan.tag,
                item_key="full",
                frame_idx=int(frame_idx),
                is_tile=False,
                label_enabled=bool(prep.label_enabled),
                is_transverse=plan.view.family == "transverse",
                physical_view_id=_physical_view_id(plan),
                presentation_variant_id=_presentation_variant_id(plan),
                geometry_item_id="full",
                channel_format=str(plan.channel_variant.format_token),
                channel_kind=str(plan.channel_variant.kind),
                channel_reverse=bool(plan.channel_variant.reverse),
                channel_offsets=tuple(int(x) for x in plan.channel_variant.offsets),
                foreground=_fg(plan.tag, frame_idx, "full"),
            ))

    # Canonical order part 2: tile variants of each view sorted by tile config;
    # inside each frame, tiles are row-major because build_render_plan creates
    # tile_layout as y-then-x for each canonical tile config.
    for plan in prep.plans:
        if not plan.tile_layout:
            continue
        by_cfg: Dict[str, List[RenderTileItem]] = defaultdict(list)
        for tile in plan.tile_layout:
            by_cfg[tile.cfg.config_id].append(tile)
        cfg_order = (
            list(dict.fromkeys(tile.cfg.config_id for tile in plan.tile_layout))
            if plan.view.shared_view is not None
            else sorted(
                by_cfg.keys(),
                key=lambda cid: (
                    by_cfg[cid][0].cfg.tile_size,
                    by_cfg[cid][0].cfg.tile_stride,
                    cid,
                ),
            )
        )
        for cfg_id in cfg_order:
            tiles = sorted(by_cfg[cfg_id], key=lambda t: (int(t.y), int(t.x), t.tile_tag))
            for frame_idx in render_plan_frame_indices(plan):
                for tile in tiles:
                    candidates.append(OutputCandidate(
                        order=-1,
                        volume_name=prep.src.stem,
                        parent_view_tag=plan.tag,
                        output_tag=tile.tile_tag,
                        item_key=tile.tile_tag,
                        frame_idx=int(frame_idx),
                        is_tile=True,
                        label_enabled=bool(prep.label_enabled),
                        is_transverse=plan.view.family == "transverse",
                        physical_view_id=_physical_view_id(plan),
                        presentation_variant_id=_presentation_variant_id(plan),
                        geometry_item_id=(
                            f"tile:{int(tile.cfg.tile_size)}:{int(tile.cfg.tile_stride)}:"
                            f"{int(tile.x)}:{int(tile.y)}"
                        ),
                        channel_format=str(plan.channel_variant.format_token),
                        channel_kind=str(plan.channel_variant.kind),
                        channel_reverse=bool(plan.channel_variant.reverse),
                        channel_offsets=tuple(int(x) for x in plan.channel_variant.offsets),
                        foreground=_fg(plan.tag, frame_idx, tile.tile_tag),
                        tile_size=int(tile.cfg.tile_size),
                        tile_stride=int(tile.cfg.tile_stride),
                        tile_x=int(tile.x),
                        tile_y=int(tile.y),
                    ))
    return candidates


def validate_foreground_transverse_candidate_invariant(
    prep: PreparedVolume,
    candidates: Sequence[OutputCandidate],
    *,
    retained_only: bool,
) -> int:
    """Verify full-frame foreground transverse outputs never shrink in count."""
    input_count = int(prep.foreground_preservation_stats.get("input_foreground_transverse_slices", 0))
    # Transverse is no longer privileged in v18.  A zero-view PTA run, or a
    # run containing only non-transverse views, is valid and intentionally has
    # no transverse candidate against which this preservation invariant
    # can be evaluated.
    if bool(prep.v18_mode) and not any(
        plan.view.family == "transverse" for plan in prep.plans
    ):
        key = (
            "retained_output_foreground_transverse_slices"
            if retained_only
            else "classified_output_foreground_transverse_slices"
        )
        prep.foreground_preservation_stats[key] = 0
        return 0
    output_frames = {
        int(cand.frame_idx)
        for cand in candidates
        if cand.is_transverse
        and not cand.is_tile
        and int(cand.augmentation_index) == 0
        and cand.foreground
        and (cand.keep or not retained_only)
    }
    output_count = len(output_frames)
    key = (
        "retained_output_foreground_transverse_slices"
        if retained_only
        else "classified_output_foreground_transverse_slices"
    )
    prep.foreground_preservation_stats[key] = int(output_count)
    if output_count < input_count:
        phase = "retained" if retained_only else "classified"
        raise RuntimeError(
            f"Foreground transverse invariant failed for {prep.src.stem}: "
            f"input foreground slices={input_count}, {phase} output foreground slices={output_count}"
        )
    return int(output_count)


def trim_background_overage_after_flips(
    physical_candidates: Sequence[OutputCandidate],
    *,
    flips_by_subset: Mapping[str, int],
    background_percent: float,
    labels_available: bool,
    out_dir: Path,
    split_active: bool,
    image_format: str,
    background_stats: BackgroundFilterStats,
    warnings: WarningLog,
    images_selected: bool = True,
    labels_selected: bool = True,
) -> int:
    """Re-tighten the per-subset background cap after render-time flips.

    B_max is budgeted from the PRESUMED foreground count, so foreground copies
    that flipped to background (and were dropped) shrink the realized
    foreground below plan, and the already-written backgrounds can exceed the
    --background_percent maximum.  Delete the lowest-priority written
    backgrounds - reversing the admission order, so augmented breadth-first
    admissions go first and unique-source originals go last - until the
    realized cap holds.  Deterministic: admission order and flips are both
    seed-determined.
    """
    p = float(background_percent)
    if p >= 1.0 or not labels_available:
        return 0
    if not any(int(x) > 0 for x in flips_by_subset.values()):
        return 0
    total_deleted = 0
    subset_names = sorted({cand.split_subset or "all" for cand in physical_candidates}, key=lambda x: {"train": 0, "val": 1, "all": 2}.get(x, 3))
    for subset_name in subset_names:
        flips = int(flips_by_subset.get(subset_name, 0))
        if flips <= 0:
            continue
        subset = [c for c in physical_candidates if (c.split_subset or "all") == subset_name]
        planned_foreground = sum(1 for c in subset if c.keep and c.foreground)
        realized_foreground = max(0, planned_foreground - flips)
        allowed = background_limit_from_foreground(realized_foreground, p)
        retained_original_background = sorted(
            [c for c in subset if c.keep and not c.foreground and int(c.augmentation_index) == 0],
            key=candidate_background_rank,
        )
        grouped_by_source: Dict[str, List[OutputCandidate]] = defaultdict(list)
        for cand in subset:
            if cand.keep and not cand.foreground and int(cand.augmentation_index) > 0:
                grouped_by_source[candidate_source_identity(cand)].append(cand)
        breadth_ranked: List[Tuple[Tuple[int, bytes, int, int], OutputCandidate]] = []
        for group in grouped_by_source.values():
            group.sort(key=candidate_background_rank)
            for position, cand in enumerate(group):
                digest, src_order, order = candidate_background_rank(cand)
                breadth_ranked.append(((int(position), digest, int(src_order), int(order)), cand))
        breadth_ranked.sort(key=lambda pair: pair[0])
        admission_order = [*retained_original_background, *[cand for _, cand in breadth_ranked]]
        excess = max(0, len(admission_order) - int(allowed))
        if excess <= 0:
            continue
        victims = admission_order[len(admission_order) - excess:]
        deleted_original = 0
        deleted_augmented = 0
        for cand in victims:
            img_path, lbl_path = candidate_output_paths(out_dir, cand, split_active=split_active, image_format=image_format)
            if bool(images_selected):
                _validate_nonempty_regular_file(
                    img_path,
                    context="background-overage trim expected image",
                )
                img_path.unlink()
            if bool(labels_selected) and cand.label_enabled and lbl_path is not None:
                if not lbl_path.is_file():
                    raise RuntimeError(
                        f"background-overage trim expected label is missing: {lbl_path}"
                    )
                lbl_path.unlink()
            cand.keep = False
            if int(cand.augmentation_index) == 0:
                deleted_original += 1
            else:
                deleted_augmented += 1
        total_deleted += int(excess)
        background_stats.background_retained -= int(excess)
        background_stats.dropped += int(excess)
        background_stats.original_background_dropped += int(deleted_original)
        background_stats.augmented_background_dropped += int(deleted_augmented)
        subset_stats = background_stats.subset_stats.get(subset_name)
        if subset_stats is not None:
            subset_stats["background_retained"] = int(subset_stats.get("background_retained", 0)) - int(excess)
            subset_stats["original_background_retained"] = int(subset_stats.get("original_background_retained", 0)) - int(deleted_original)
            subset_stats["augmented_background_retained"] = int(subset_stats.get("augmented_background_retained", 0)) - int(deleted_augmented)
        warnings.add(
            "background_cap_trimmed_after_foreground_flips",
            f"{subset_name}: flips={flips}, realized_foreground={realized_foreground}, deleted_backgrounds={excess}",
        )
    return total_deleted




_GENERATED_OUTPUT_DIR_NAMES = (
    "images",
    "labels",
    "overlays",
    "nrrds",
    "augmentation",
    ".volume_done",
    ".v18_work",
)
_GENERATED_OUTPUT_FILE_NAMES = (
    "dataset.yaml",
    "summary.txt",
    "manifest.json",
    "voxel_volume.json",
)
_V18_OUTPUT_SENTINEL_NAME = ".pta_v18_output.json"


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(parent).resolve(strict=False))
        return True
    except ValueError:
        return False


def validate_fresh_output_safety(
    out_dir: Path,
    *,
    input_dir: Path,
    specs: Sequence[VolumeInputSpec],
    augmentation_path: Optional[Path] = None,
) -> None:
    """Refuse fresh-publication cleanup that could overlap authoritative inputs."""

    output = Path(out_dir).resolve(strict=False)
    source_root = Path(input_dir).resolve(strict=False)
    filesystem_root = Path(output.anchor).resolve(strict=False)
    user_home = Path.home().resolve(strict=False)
    working_directory = Path.cwd().resolve(strict=False)
    if output == filesystem_root:
        raise ValueError(f"PTA output may not be a filesystem/drive root: {output}")
    if output == user_home or _path_is_within(user_home, output):
        raise ValueError(
            "PTA output may not be the user home directory or one of its ancestors: "
            f"{output}"
        )
    if output == working_directory or _path_is_within(working_directory, output):
        raise ValueError(
            "PTA output may not be the active workspace directory or one of its ancestors: "
            f"workspace={working_directory}, output={output}"
        )
    if (
        output == source_root
        or _path_is_within(output, source_root)
        or _path_is_within(source_root, output)
    ):
        raise ValueError(
            "PTA output and input directories must be disjoint for a fresh v18 publication; "
            f"input={source_root}, output={output}"
        )

    sentinel = output / _V18_OUTPUT_SENTINEL_NAME
    if output.exists():
        existing_children = tuple(output.iterdir())
        if existing_children and not sentinel.is_file():
            raise ValueError(
                "PTA refuses to clean a nonempty output directory that is not owned by "
                f"v18 (missing {sentinel.name}): {output}"
            )
        if sentinel.is_file():
            try:
                sentinel_payload = json.loads(sentinel.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"Invalid PTA v18 output sentinel: {sentinel}") from exc
            if (
                sentinel_payload.get("schema") != "pta.v18.output-directory/1"
                or Path(str(sentinel_payload.get("output", ""))).resolve(strict=False)
                != output
            ):
                raise ValueError(
                    f"PTA v18 output sentinel does not own this exact directory: {sentinel}"
                )

    protected: List[Path] = [source_root]
    for spec in specs:
        protected.extend(Path(value).resolve(strict=False) for value in spec.image_paths_by_index.values())
        protected.extend(Path(value).resolve(strict=False) for value in spec.labels_by_index.values())
        if spec.video_path is not None:
            protected.append(Path(spec.video_path).resolve(strict=False))
        if spec.segmentation_nrrd_path is not None:
            protected.append(Path(spec.segmentation_nrrd_path).resolve(strict=False))
    if augmentation_path is not None:
        protected.append(Path(augmentation_path).resolve(strict=False))

    cleanup_targets = [output / name for name in _GENERATED_OUTPUT_DIR_NAMES]
    cleanup_targets.extend(output / name for name in _GENERATED_OUTPUT_FILE_NAMES)
    for target in cleanup_targets:
        junction_test = getattr(target, "is_junction", None)
        if target.is_symlink() or (
            callable(junction_test) and bool(junction_test())
        ):
            raise ValueError(
                "PTA refuses to clean a generated-output target that is a symlink or "
                f"junction: {target}"
            )
        target_resolved = target.resolve(strict=False)
        for source in protected:
            if source == target_resolved or _path_is_within(source, target_resolved):
                raise ValueError(
                    "PTA fresh-publication cleanup target overlaps an input: "
                    f"target={target_resolved}, input={source}"
                )


def write_v18_output_sentinel(out_dir: Path) -> Path:
    output = Path(out_dir).resolve(strict=False)
    return write_json_manifest(
        output / _V18_OUTPUT_SENTINEL_NAME,
        {
            "schema": "pta.v18.output-directory/1",
            "output": str(output),
            "ownership": "only named v18 PTA generated targets may be cleaned",
        },
    )


def clean_generated_output_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with _CREATED_OUTPUT_DIRS_LOCK:
        _CREATED_OUTPUT_DIRS.clear()
    for name in _GENERATED_OUTPUT_DIR_NAMES:
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in _GENERATED_OUTPUT_FILE_NAMES:
        path = out_dir / name
        if path.exists():
            path.unlink()


def _cleanup_v18_pta_selected_run_work(work_dir: Path) -> None:
    """Strictly remove PTA scratch before committing a complete manifest."""

    work_dir = Path(work_dir)
    if not work_dir.exists() and not work_dir.is_symlink():
        return
    gc.collect()
    try:
        junction_test = getattr(work_dir, "is_junction", None)
        if work_dir.is_symlink() or (
            callable(junction_test) and bool(junction_test())
        ):
            raise RuntimeError(
                f"PTA selected-run work path became a symlink or junction: {work_dir}"
            )
        shutil.rmtree(work_dir)
        if work_dir.exists() or work_dir.is_symlink():
            raise RuntimeError(f"PTA selected-run work path remains after cleanup: {work_dir}")
    except Exception as exc:
        raise RuntimeError(
            "PTA selected-run cleanup failed; refusing a complete manifest: "
            f"{work_dir}"
        ) from exc


def write_deferred_augmentation_bundle(
    out_dir: Path,
    *,
    definition: AugmentationDefinition,
    requested_ratio: float,
    split_active: bool,
) -> Path:
    """Persist one self-contained external policy plus loader-facing metadata."""
    bundle_dir = out_dir / "augmentation"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    policy_path = bundle_dir / definition.path.name
    shutil.copy2(definition.path, policy_path)
    copied_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if copied_sha256 != definition.content_sha256:
        raise RuntimeError(f"Deferred augmentation policy copy failed SHA-256 verification: {policy_path}")
    manifest = {
        "schema": "pta.deferred-augmentation.v2",
        "pipeline_spec_version": PIPELINE_SPEC_VERSION,
        "execution": "training_loader",
        "loader_hook_required": True,
        "stock_ultralytics_consumes_policy_automatically": False,
        "policy_file": policy_path.name,
        "policy_sha256": copied_sha256,
        "policy_export": definition.export_name,
        "requested_total_versions_ratio": float(requested_ratio),
        "ratio_semantics": "expected training-time presentations per retained original; no physical augmented copies",
        "applies_to": "train" if split_active else "all",
        "distributed_seed_contract": "derive per-sample/per-epoch/per-rank seeds in the training loader",
        "single_file_gpu_contract": {
            "offline_export": "build_gpu_augmentation(device, batch_size)",
            "offline_status": "executed when --augmentation_execution offline",
            "deferred_status": "copied for a custom training-loader hook; stock Ultralytics does not import it",
        },
    }
    manifest_path = bundle_dir / "deferred_policy.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def create_output_dirs(
    out_dir: Path,
    *,
    split_active: bool,
    train_split: Optional[float],
    labels_available: bool,
    publish_images: bool = True,
    publish_labels: bool = True,
) -> None:
    if split_active:
        if publish_images:
            (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        if labels_available and publish_labels:
            (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        if train_split is None or float(train_split) < 1.0:
            if publish_images:
                (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
            if labels_available and publish_labels:
                (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    else:
        if publish_images:
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
        if labels_available and publish_labels:
            (out_dir / "labels").mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(out_dir: Path, *, train_split: float, channels: int) -> Path:
    channel_count = int(channels)
    if channel_count < 1:
        raise ValueError(f"Dataset channel count must be positive, got {channels}")
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
    ]
    if float(train_split) < 1.0:
        lines.append("val: images/val")
    else:
        # Stock Ultralytics requires both train and val keys in every detection/
        # segmentation dataset YAML. Reuse the training images for the explicit
        # 100%-train smoke-test case rather than referencing a nonexistent or
        # empty validation directory.
        lines.append("val: images/train")
    lines.extend([
        f"channels: {channel_count}",
        "nc: 1",
        "names:",
        "  0: object",
    ])
    path = out_dir / "dataset.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def render_stats_from_plans(plans: Sequence[RenderPlan], retained_candidates: Sequence[OutputCandidate]) -> List[Dict[str, object]]:
    retained_by_tag = Counter(c.parent_view_tag for c in retained_candidates if c.keep)
    retained_by_output_tag = Counter(c.output_tag for c in retained_candidates if c.keep)
    stats: List[Dict[str, object]] = []
    for plan in plans:
        st = dict(plan.stats)
        st["retained_full_or_tile_candidates_for_parent_view"] = int(retained_by_tag.get(plan.tag, 0))
        st["retained_full_frame_candidates"] = int(retained_by_output_tag.get(plan.tag, 0))
        stats.append(st)
    return stats


def write_pta_summary(
    path: Path,
    *,
    command: str,
    input_dir: Path,
    out_dir: Path,
    specs: Sequence[VolumeInputSpec],
    volume_records: Sequence[VolumeSummaryRecord],
    tile_configs: Sequence[TileConfig],
    channel_formats: Sequence[ChannelFormat],
    channel_variants: Sequence[ChannelVariant],
    requested_output_format: str,
    background_stats: BackgroundFilterStats,
    split_stats: SplitStats,
    augmentation_stats: AugmentationStats,
    dataset_yaml_path: Optional[Path],
    warnings: WarningLog,
    workers: int,
    frame_workers: int,
    planning_workers: int,
    topology_summary: str,
    jpeg_decode_backend: str,
    jpeg_batch_size: int,
    jpeg_encode_backend: str,
    jpeg_quality: int,
    gpu_batch_size: int,
    image_format: str,
    png_compression: int,
    tiff_encode_backend: str = "auto",
) -> Path:
    normalized_image_format = parse_output_image_format(image_format)
    lines: List[str] = []
    lines.append(f"Specification version: {PIPELINE_SPEC_VERSION}")
    lines.append(f"Command: {command}")
    lines.append(f"Input directory: {input_dir}")
    lines.append(f"Output directory: {out_dir}")
    lines.append(f"Volumes discovered: {len(specs)}")
    lines.append(f"Workers: {int(workers)} (0 defaults to the process CPU-affinity count)")
    lines.append(f"Frame workers: {int(frame_workers)}")
    lines.append(f"Planning/prefetch worker budget during pipelining: {int(planning_workers)}")
    lines.append(f"Topology plan: {topology_summary}")
    lines.append(f"JPEG input decoder request: {jpeg_decode_backend}; batch_size={int(jpeg_batch_size)}")
    lines.append(
        f"JPEG output encoder request: {jpeg_encode_backend}; quality={int(jpeg_quality)}; "
        f"GPU augmentation batch_size={int(gpu_batch_size)}"
    )
    lines.append(f"Requested output image format: {parse_output_image_format(requested_output_format)}")
    lines.append(f"Output image format: {normalized_image_format}")
    lines.append(f"Multipage TIFF output encoder request: {tiff_encode_backend}")
    if normalized_image_format == "png":
        lines.append(f"Image encoding: PNG (lossless; compression level {int(png_compression)})")
    elif normalized_image_format == "jpg":
        lines.append(
            f"Image encoding: JPEG (quality {int(jpeg_quality)}; intrinsically lossy; "
            f"requested backend {jpeg_encode_backend})"
        )
    else:
        lines.append("Image encoding: TIFF uint8 (lossless; custom channels stored as one grayscale page per channel for stock Ultralytics)")
    lines.append("Requested channel formats: " + ", ".join(fmt.token for fmt in channel_formats))
    lines.append("Expanded channel output sets:")
    for variant in channel_variants:
        lines.append(
            f"  {variant.tag_token}: kind={variant.kind}, channels={int(variant.channel_count)}, "
            f"stride={int(variant.stride)}, order={variant.order_name}, offsets={list(variant.offsets)}"
        )
    lines.append("Channel-stack label policy: every output uses the center slice N mask/YOLO label")
    lines.append("Channel-stack boundary policy: requests beyond the actual minimum/maximum image index are edge-clamped")
    lines.append("Channel-stack discontinuity policy: a custom C...S... center is skipped when any required in-volume encoded image index is absent")
    lines.append("Stock Ultralytics multispectral compatibility target: version 8.3.112 or newer")
    lines.append(
        "Shared forward sampling: TTA hardware-linear radial/intensity policy and "
        "TTA affine stage; categorical ground truth uses nearest sampling with the "
        "TTA tilted-stack threshold"
    )
    lines.append(f"NRRD export layout: {NRRD_AXIS_ORDER_NOTE}; space={NRRD_SPACE}; space_directions=identity")
    lines.append("Built-in PTA in-plane variant: fixed internally at 0 degrees in v18; --angle is not a PTA flag")
    lines.append("Implementation notes/conflicts:")
    lines.append("  - --force resolves Partially Labeled volumes to Fully Labeled for uniform type checking; without --force, mixed raw volume classes are rejected before processing.")
    lines.append("  - For unlabeled volumes, --background_percent is disabled because v3 defines background by the YOLO label export while label operations are excluded for unlabeled annotation outputs.")
    lines.append("  - The pipeline emits frame index %04d as the one-based source-view frame index; filtered-out frames are omitted rather than compact-renumbered.")
    lines.append("  - --channel_format gray/grey emits one channel; RGB triplicates slice N; C{odd}S{stride} maps neighboring view slices into channels and adds a reverse-order output set.")
    lines.append("  - Labels and foreground/background classification always use the center slice N, independent of C, S, or forward/reverse image-channel order.")
    lines.append("  - A YOLO label file may cover only a subset of image indices. These inputs are Partially Labeled: labeled-empty centers are annotated background, labeled-nonempty centers are annotated foreground, and images without a label file are context-only/unannotated and are never emitted as centers.")
    lines.append("  - Native partial-volume C...S... channels resolve offsets by encoded image index. Missing indices inside the actual min/max bounds skip the center with a warning; only requests outside those bounds edge-clamp.")
    lines.append("  - Any custom C...S... channel format forces multi-page TIFF for the run; each grayscale page is one channel, matching stock Ultralytics decoding.")
    lines.append("  - Default gray outputs use compact filename tags; RGB/custom sets append _RGB, _C...S..., or _C...S..._reverse.")
    if augmentation_stats.execution_mode == "deferred":
        lines.append("  - --augmentation_ratio is deferred training-loader replay metadata; the generator writes originals only and requires a loader hook to consume augmentation/deferred_policy.json.")
        lines.append("  - With an active split, deferred external augmentation metadata applies only to train originals; validation files remain original-only.")
    else:
        lines.append("  - In offline mode, --augmentation_ratio is the target physical version count including the original; augmented copies alone receive deterministic 16-character alphanumeric suffixes.")
        lines.append("  - With an active split, offline external augmentations apply only to retained train originals; validation files remain original-only.")
    lines.append("  - Split units are assigned by stable SHA-256 rank before background filtering; train/val enforce independent background caps so canonical validation-tail backgrounds are not preferentially removed.")
    lines.append("  - Within each subset, every unique-source (original) background is admitted up to the full B_max before any augmented duplicate; remaining capacity is filled breadth-first across source identities.")
    lines.append("  - --background_percent is enforced per volume as a maximum; no dataset-wide classification barrier exists and outputs stream from the first volume onward.")
    lines.append("  - Complete labeled volumes preserve at least the input count of foreground transverse slices through smoothing/cubic resize; --background_percent continues to govern background retention without a C1 override.")
    lines.append("  - Foreground classification runs on copy-0 originals only. Transverse YOLO candidates use predecoded polygon/ROI geometry; NRRD and general resliced views retain mask-only rendering. Offline augmented copies inherit their source class for budgeting.")
    lines.append("  - Source-frame scheduling is grouped and deterministic, while built-in view, affine, tile, and categorical rendering are delegated to the shared TTA geometry module.")
    lines.append("  - CPU policies use the persistent CPU pool. A build_gpu_augmentation policy uses one persistent process per CUDA-visible GPU; tile resize and deterministic replay batches execute on that GPU (H2), while fused geometry/pointwise behavior is owned by the external policy (M1).")
    lines.append("  - JPEG GPU-policy outputs are batch encoded directly from CUDA tensors through nvImageCodec/nvJPEG when requested and available; explicit nvjpeg is fail-fast while auto records any OpenCV fallback.")
    lines.append("  - Render/augment/encode work runs on one persistent pool per run (--worker_backend); all retained full/tile items and offline replays for a (plan, source frame) share one canonical render. Legacy queue/chunk tuning flags are not accepted by the v18 PTA interface.")
    lines.append("  - Topology-aware mode maps CUDA-visible GPUs to observed PCI/NUMA-local allocated CPUs and binds persistent render workers plus nvJPEG decode threads; it falls back to an allocation-safe CPU partition when locality is unavailable.")
    lines.append("  - Overlay videos use the center image channel and remain original source-view diagnostics; they do not include filtered/split-specific augmented copies.")
    if dataset_yaml_path is not None:
        lines.append(f"Dataset YAML: {dataset_yaml_path}")

    lines.append("")
    lines.append("Volume source dimensions before cubic resizing:")
    for rec in volume_records:
        t, h, w = rec.source_shape
        pt, ph, pw = rec.processing_shape
        lines.append(f"  {rec.stem}:")
        lines.append(f"    input_kind: {rec.input_kind}")
        lines.append(f"    detected_volume_class: {rec.volume_class}")
        lines.append(f"    effective_volume_class: {rec.effective_volume_class}")
        lines.append(f"    label_source: {rec.label_source}")
        lines.append(f"    label_operations_enabled: {bool(rec.label_enabled)}")
        lines.append(f"    source_dimensions_X_Y_t: ({int(w)}, {int(h)}, {int(t)})")
        lines.append(f"    processing_dimensions_X_Y_t: ({int(pw)}, {int(ph)}, {int(pt)})")
        lines.append(f"    fps_for_overlay_videos: {rec.fps}")
        if rec.input_start_index is not None:
            lines.append(f"    detected_input_start_index: {int(rec.input_start_index)}")
        if rec.encoded_indices:
            encoded = list(rec.encoded_indices)
            if len(encoded) <= 24:
                lines.append(f"    encoded_input_indices: {encoded}")
            else:
                lines.append(f"    encoded_input_indices: count={len(encoded)}, first={encoded[:8]}, last={encoded[-8:]}")
        lines.append("    source_annotation_states_before_transforms:")
        lines.append(f"      annotated_foreground: {int(rec.annotation_state_counts.get('annotated_foreground', 0))}")
        lines.append(f"      annotated_background: {int(rec.annotation_state_counts.get('annotated_background', 0))}")
        lines.append(f"      unannotated: {int(rec.annotation_state_counts.get('unannotated', 0))}")
        lines.append("    foreground_transverse_preservation:")
        lines.append(f"      input_foreground_slices: {int(rec.foreground_preservation_stats.get('input_foreground_transverse_slices', 0))}")
        lines.append(f"      guaranteed_after_preprocessing: {int(rec.foreground_preservation_stats.get('guaranteed_output_foreground_transverse_slices', 0))}")
        lines.append(f"      classified_output_slices: {int(rec.foreground_preservation_stats.get('classified_output_foreground_transverse_slices', 0))}")
        lines.append(f"      retained_output_slices: {int(rec.foreground_preservation_stats.get('retained_output_foreground_transverse_slices', 0))}")
        lines.append(f"      source_polygon_anchor_seeds: {int(rec.foreground_preservation_stats.get('source_polygon_anchor_seeds', 0))}")
        lines.append(f"      smoothing_anchor_repairs: {int(rec.foreground_preservation_stats.get('smoothing_anchor_repairs', 0))}")
        lines.append(f"      processed_anchor_repairs: {int(rec.foreground_preservation_stats.get('processed_anchor_repairs', 0))}")
        lines.append(f"    original_candidates_total_before_filtering: {int(rec.candidates_total)}")
        lines.append(f"    original_candidates_retained_after_filtering: {int(rec.candidates_retained)}")
        lines.append(f"    augmented_candidates_planned: {int(rec.augmented_candidates_planned)}")
        lines.append(f"    augmented_candidates_retained: {int(rec.augmented_candidates_retained)}")
        lines.append(f"    candidate_versions_processed_original_plus_augmented: {int(rec.candidates_written)}")
        if rec.voxel_initial is not None or rec.voxel_final is not None:
            lines.append("    voxel_volume:")
            if rec.voxel_initial is not None:
                lines.append(f"      initial_rasterized_transverse_mask: {int(rec.voxel_initial)}")
            if rec.voxel_final is not None:
                lines.append(f"      final_mask_after_gaussian_smoothing: {int(rec.voxel_final)}")
        if rec.smoothing_stats:
            lines.append("    gaussian_smoothing:")
            for st in rec.smoothing_stats:
                lines.append(
                    f"      pass {int(st.get('pass_index', 0))}: sigma={float(st.get('sigma', 0.0)):g}, "
                    f"foreground_before={int(st.get('foreground_before', 0))}, "
                    f"foreground_after={int(st.get('foreground_after', 0))}, "
                    f"delta_voxels={int(st.get('delta_voxels', 0))}"
                )
        else:
            lines.append("    gaussian_smoothing: disabled, not requested, or unavailable for this input class")
        if rec.nrrd_paths:
            lines.append("    nrrd_outputs:")
            for pth in rec.nrrd_paths:
                lines.append(f"      {pth}")
        lines.append("    active_views:")
        for v in rec.views:
            extra = ""
            if v.family == "radial":
                spacing = float(v.azimuths_deg[1] - v.azimuths_deg[0]) if len(v.azimuths_deg) > 1 else 0.0
                extra = (
                    f", azimuth_frames={len(v.azimuths_deg)}, azimuth_step={spacing:g}, "
                    f"diameter={v.diameter}, image_sampling={shared_geometry.RADIAL_FILTER_MODE}, "
                    "categorical_sampling=nearest"
                )
            if v.family == "tilted_transverse":
                extra = f", direction={v.tilt_direction}, signed_tilt={v.tilt_angle_deg:g}"
            lines.append(f"      {v.display_name}: frames={int(v.num_slices)}, source_plane=({int(v.src_w)}x{int(v.src_h)}){extra}")
        lines.append("    rendered_output_sets:")
        for st in rec.render_stats:
            lines.append(
                f"      {st.get('tag')}: view={st.get('view')}, eligible_center_frames={st.get('eligible_center_frames', st.get('frames'))}, "
                f"view_frames={st.get('view_frames')}, skipped_center_frames={st.get('skipped_center_frames')}, "
                f"annotated_center_frames={st.get('annotated_center_frames', st.get('view_frames'))}, "
                f"unannotated_centers_excluded={st.get('skipped_unannotated_centers', 0)}, "
                f"discontinuous_centers_excluded={st.get('skipped_discontinuous_centers', 0)}, "
                f"channel_format={st.get('channel_format')}, channel_order={st.get('channel_order')}, "
                f"channel_offsets={st.get('channel_offsets')}, "
                f"full_output_size={st.get('full_output_size')}, tiles={len(st.get('tiles', []))}, "
                f"label_enabled={st.get('label_enabled')}, "
                f"retained_parent_candidates={st.get('retained_full_or_tile_candidates_for_parent_view', 0)}"
            )

    lines.append("")
    if tile_configs:
        lines.append("Tile configurations:")
        for cfg in tile_configs:
            lines.append(f"  {cfg.config_id}: tile_size={cfg.tile_size}, tile_stride={cfg.tile_stride}")
    else:
        lines.append("Tile configurations: disabled")

    lines.append("")
    lines.append("Background filtering:")
    lines.append(f"  active: {bool(background_stats.active and not background_stats.skipped_reason)}")
    lines.append(f"  foreground_background_classification_performed: {bool(background_stats.classification_performed)}")
    if background_stats.skipped_reason:
        lines.append(f"  skipped_reason: {background_stats.skipped_reason}")
    if background_stats.classification_performed:
        lines.append(f"  foreground_count_F: {int(background_stats.foreground_before)}")
        lines.append(f"  background_count_B_before_filtering: {int(background_stats.background_before)}")
        if background_stats.background_max is not None:
            lines.append(f"  B_max: {int(background_stats.background_max)}")
        lines.append(f"  background_count_retained: {int(background_stats.background_retained)}")
        lines.append(f"  dropped_background_frames: {int(background_stats.dropped)}")
        lines.append(f"  dropped_original_background_frames: {int(background_stats.original_background_dropped)}")
        lines.append(f"  dropped_augmented_background_frames: {int(background_stats.augmented_background_dropped)}")
        if background_stats.subset_stats:
            lines.append("  subset_quotas:")
            for subset_name in ("train", "val", "all"):
                subset = background_stats.subset_stats.get(subset_name)
                if subset is None:
                    continue
                lines.append(
                    f"    {subset_name}: foreground={int(subset.get('foreground', 0))}, "
                    f"original_foreground={int(subset.get('original_foreground', subset.get('foreground', 0)))}, "
                    f"augmented_foreground={int(subset.get('augmented_foreground', 0))}, "
                    f"background_before={int(subset.get('background_before', 0))}, "
                    f"original_B_max={int(subset.get('original_background_max', subset.get('background_max', 0)))}, "
                    f"B_max={int(subset.get('background_max', 0))}, "
                    f"original_background_retained={int(subset.get('original_background_retained', 0))}, "
                    f"augmented_background_retained={int(subset.get('augmented_background_retained', 0))}, "
                    f"background_retained={int(subset.get('background_retained', 0))}"
                )
    else:
        lines.append("  foreground/background counts: not classified because filtering was inactive or labels were unavailable")

    lines.append("")
    lines.append("Dataset splitting:")
    lines.append(f"  active: {bool(split_stats.active)}")
    if split_stats.active:
        lines.append(f"  --train_split: {float(split_stats.train_split if split_stats.train_split is not None else 0.0):g}")
        lines.append(f"  --split_method: {split_stats.split_method}")
        lines.append("  assignment_order: stable SHA-256 rank of the atomic unit (not canonical prefix order)")
        lines.append(f"  atomic_unit_count_total: {int(split_stats.atomic_units_total)}")
        lines.append(f"  atomic_unit_count_train: {int(split_stats.atomic_units_train)}")
        lines.append(f"  atomic_unit_count_val: {int(split_stats.atomic_units_val)}")
        lines.append(f"  retained_original_frame_count_total_before_augmentation: {int(split_stats.frames_total)}")
        lines.append(f"  retained_original_frame_count_train_before_augmentation: {int(split_stats.frames_train)}")
        lines.append(f"  retained_original_frame_count_val_before_augmentation: {int(split_stats.frames_val)}")
        lines.append(f"  achieved_original_train_fraction_before_augmentation: {float(split_stats.achieved_train_fraction):.6f}")
        if split_stats.warning:
            lines.append(f"  best_effort_warning: {split_stats.warning}")

    lines.append("")
    lines.append("External augmentation policy:")
    lines.append(f"  configured: {bool(augmentation_stats.configured)}")
    lines.append(f"  execution_mode: {augmentation_stats.execution_mode}")
    lines.append(f"  runtime_backend: {augmentation_stats.runtime_backend}")
    lines.append(f"  active_copies_planned: {bool(augmentation_stats.active)}")
    lines.append(f"  requested_total_versions_ratio: {float(augmentation_stats.requested_ratio):g}")
    lines.append(f"  applies_to: {augmentation_stats.applies_to}")
    if augmentation_stats.path is not None:
        lines.append(f"  definition_file: {augmentation_stats.path}")
        lines.append(f"  definition_sha256: {augmentation_stats.content_sha256}")
        lines.append(f"  definition_export: {augmentation_stats.export_name}")
        if augmentation_stats.albumentations_version:
            lines.append(f"  albumentations_version: {augmentation_stats.albumentations_version}")
    if augmentation_stats.deferred_policy_path is not None:
        lines.append(f"  deferred_policy_manifest: {augmentation_stats.deferred_policy_path}")
        lines.append("  loader_hook_required: true (stock Ultralytics does not consume this policy automatically)")
    lines.append(f"  eligible_retained_originals: {int(augmentation_stats.eligible_originals)}")
    lines.append(f"  planned_augmented_copies: {int(augmentation_stats.planned_augmented_copies)}")
    if augmentation_stats.execution_mode == "offline":
        lines.append("  augmented_copy_classification: presumed from copy-0 class; foreground flips are reconciled and dropped at render time")
    lines.append(f"  planned_augmented_foreground_presumed: {int(augmentation_stats.planned_augmented_foreground)}")
    lines.append(f"  planned_augmented_background_presumed: {int(augmentation_stats.planned_augmented_background)}")
    lines.append(f"  dropped_augmented_background: {int(augmentation_stats.dropped_augmented_background)}")
    lines.append(f"  retained_augmented_copies: {int(augmentation_stats.retained_augmented_copies)}")
    lines.append(f"  physical_versions_ratio_after_filtering: {float(augmentation_stats.achieved_ratio):.6f}")
    if split_stats.active:
        lines.append(f"  final_train_dataset_samples: {int(augmentation_stats.final_train_files)}")
        lines.append(f"  final_val_dataset_samples: {int(augmentation_stats.final_val_files)}")
    else:
        lines.append(f"  final_unsplit_dataset_samples: {int(augmentation_stats.final_unsplit_files)}")
    if augmentation_stats.execution_mode == "offline":
        lines.append("  naming: original filename unchanged; augmented filename appends _[0-9A-Za-z]{16}")
        lines.append("  determinism: SHA-256 definition-content/source/copy identity -> base62 tag and per-copy seed")
    elif augmentation_stats.execution_mode == "deferred":
        lines.append("  deferred_semantics: requested ratio is an expected replay/sampler weight; generator emits no augmented suffix files")

    lines.append("")
    lines.extend(warnings.summary_lines())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def resolve_split_cli(args: argparse.Namespace, argv: Sequence[str]) -> Tuple[bool, Optional[float], Optional[str], bool, bool]:
    train_explicit = any(tok == "--train_split" or tok.startswith("--train_split=") for tok in argv)
    method_explicit = any(tok == "--split_method" or tok.startswith("--split_method=") for tok in argv)
    split_active = bool(train_explicit or method_explicit)
    train_split = args.train_split
    split_method = args.split_method
    if split_active:
        if train_split is None:
            train_split = 0.8
        if split_method is None:
            split_method = "view"
    return split_active, train_split, split_method, train_explicit, method_explicit


def _accumulate_background_stats(total: BackgroundFilterStats, part: BackgroundFilterStats) -> None:
    total.foreground_before += int(part.foreground_before)
    total.background_before += int(part.background_before)
    if part.background_max is not None:
        total.background_max = int(total.background_max or 0) + int(part.background_max)
    total.background_retained += int(part.background_retained)
    total.dropped += int(part.dropped)
    total.original_background_dropped += int(part.original_background_dropped)
    total.augmented_background_dropped += int(part.augmented_background_dropped)
    if part.skipped_reason and not total.skipped_reason:
        total.skipped_reason = part.skipped_reason
    for subset_name, subset in part.subset_stats.items():
        agg = total.subset_stats.setdefault(subset_name, {})
        for key, value in subset.items():
            agg[key] = int(agg.get(key, 0)) + int(value)


def _accumulate_augmentation_stats(total: AugmentationStats, part: AugmentationStats) -> None:
    total.active = bool(total.active or part.active)
    total.eligible_originals += int(part.eligible_originals)
    total.planned_augmented_copies += int(part.planned_augmented_copies)
    total.planned_augmented_foreground += int(part.planned_augmented_foreground)
    total.planned_augmented_background += int(part.planned_augmented_background)
    total.retained_augmented_copies += int(part.retained_augmented_copies)
    total.dropped_augmented_background += int(part.dropped_augmented_background)
    total.final_train_files += int(part.final_train_files)
    total.final_val_files += int(part.final_val_files)
    total.final_unsplit_files += int(part.final_unsplit_files)


def _accumulate_split_stats(total: SplitStats, part: SplitStats) -> None:
    total.atomic_units_total += int(part.atomic_units_total)
    total.atomic_units_train += int(part.atomic_units_train)
    total.atomic_units_val += int(part.atomic_units_val)
    total.frames_total += int(part.frames_total)
    total.frames_train += int(part.frames_train)
    total.frames_val += int(part.frames_val)


def main(
    args: argparse.Namespace | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    cli_argv = list(argv) if argv is not None else list(sys.argv[1:])
    if args is None:
        raise TypeError(
            "PTA requires a resolved runtime configuration; launch through "
            f"{SCRIPT_BASENAME} --mode pta"
        )
    warnings = WarningLog()
    workers = choose_workers(int(args.workers))
    frame_workers = max(1, int(args.frame_workers) if int(args.frame_workers) > 0 else int(workers))
    gpu_render_threads = 1
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    channel_formats = resolve_channel_formats(args.channel_format)
    channel_variants = expand_channel_variants(channel_formats)
    requested_output_format = str(
        getattr(args, "_v18_requested_output_format", args.output_format)
    )
    custom_channel_formats = [fmt for fmt in channel_formats if fmt.kind == "custom"]
    if custom_channel_formats:
        args.output_format = "tif"
        forced_formats = ", ".join(fmt.token for fmt in custom_channel_formats)
        force_message = (
            f"Custom channel format(s) {forced_formats} require TIFF; "
            f"requested --output_format={requested_output_format}, effective --output_format=tif"
        )
        warnings.add("custom_channel_format_forces_tiff", force_message)
        print(f"WARNING: {force_message}", file=sys.stderr)
    if int(args.imgsz) < 0:
        raise ValueError("--imgsz must be >= 0")
    if args.azimuth_angle is not None and float(args.azimuth_angle) < 0.0:
        raise ValueError("--azimuth_angle must be >= 0")
    if float(args.gaussian_smoothing) < 0.0:
        raise ValueError("--gaussian_smoothing must be >= 0; use 0 to disable")
    if int(args.gaussian_smoothing_passes) < 0:
        raise ValueError("--gaussian_smoothing_passes must be >= 0")
    if str(args.output_format) == "png" and not (0 <= int(args.png_compression) <= 9):
        raise ValueError("--png_compression must be between 0 and 9")
    if int(args.frame_workers) < 0:
        raise ValueError("--frame_workers must be >= 0")
    if int(args.max_pending_frames) < 0:
        raise ValueError("--max_pending_frames must be >= 0")
    if int(args.tile_task_chunk) <= 0:
        raise ValueError("--tile_task_chunk must be >= 1")
    if int(args.aug_task_chunk) <= 0:
        raise ValueError("--aug_task_chunk must be >= 1")
    if int(args.overlay_workers) < 0:
        raise ValueError("--overlay_workers must be >= 0")
    if int(args.overlay_pending_frames) < 0:
        raise ValueError("--overlay_pending_frames must be >= 0")
    if int(args.overlay_tile_writer_limit) <= 0:
        raise ValueError("--overlay_tile_writer_limit must be > 0")
    if int(args.jpeg_batch_size) <= 0:
        raise ValueError("--jpeg_batch_size must be > 0")
    if int(args.gpu_batch_size) <= 0:
        raise ValueError("--gpu_batch_size must be > 0")
    if not (1 <= int(args.jpeg_quality) <= 100):
        raise ValueError("--jpeg_quality must be between 1 and 100")
    if not (0.0 <= float(args.background_percent) <= 1.0):
        raise ValueError("--background_percent must be in [0.0, 1.0]")
    if not math.isfinite(float(args.augmentation_ratio)) or float(args.augmentation_ratio) < 1.0:
        raise ValueError("--augmentation_ratio must be finite and >= 1.0; the ratio includes the original version")
    if float(args.augmentation_ratio) > 1.0 and not args.augmentation:
        raise ValueError("--augmentation_ratio > 1.0 requires --augmentation PATH")
    render_backend = resolve_render_backend(str(args.worker_backend))

    split_active, train_split, split_method, _train_explicit, _method_explicit = resolve_split_cli(args, cli_argv)
    if split_active and train_split is not None and not (0.0 <= float(train_split) <= 1.0):
        raise ValueError("--train_split must be in [0.0, 1.0]")

    dataset_channels: Optional[int] = None
    if split_active:
        split_channel_counts = {int(fmt.channel_count) for fmt in channel_formats}
        if len(split_channel_counts) != 1:
            detail = ", ".join(f"{fmt.token}={int(fmt.channel_count)}" for fmt in channel_formats)
            raise ValueError(
                "A split-generated stock Ultralytics dataset cannot mix image channel counts because "
                f"dataset.yaml declares one 'channels' value; requested formats: {detail}. "
                "Run each channel count into a separate output dataset."
            )
        dataset_channels = next(iter(split_channel_counts))

    topology = discover_topology(enabled=bool(args.topology_aware), warnings=warnings)
    requested_tilt_angles = resolve_tilt_angles(args.tilt_angle)
    requested_tilt_directions = resolve_tilt_directions(args.tilt_direction)
    v18_config = getattr(args, "_v18_config", None)
    if v18_config is not None:
        tile_configs = [
            TileConfig(
                int(tile.tile_size),
                int(tile.tile_stride),
                str(tile.config_id),
            )
            for tile in v18_config.tiles
        ]
    else:
        tile_configs = resolve_tile_configs(args.tile_size, args.tile_stride)
    augmentation_definition = inspect_augmentation_definition(args.augmentation) if args.augmentation else None
    augmentation = (
        load_offline_augmentation_definition(args.augmentation)
        if args.augmentation and str(args.augmentation_execution) == "offline"
        else None
    )
    gpu_offline_active = isinstance(augmentation, LoadedGpuAugmentation)
    requested_offline_backend = str(args.offline_augmentation_backend)
    if requested_offline_backend != "auto" and augmentation is None:
        raise ValueError(
            "--offline_augmentation_backend cpu/gpu requires --augmentation PATH "
            "and --augmentation_execution offline"
        )
    if requested_offline_backend == "gpu" and augmentation is not None and not gpu_offline_active:
        raise ValueError(
            "--offline_augmentation_backend gpu requires an external build_gpu_augmentation export"
        )
    if requested_offline_backend == "cpu" and gpu_offline_active:
        raise ValueError(
            "--offline_augmentation_backend cpu cannot execute build_gpu_augmentation; "
            "use auto/gpu or a CPU Albumentations policy"
        )
    if requested_offline_backend != "auto" and str(args.augmentation_execution) != "offline":
        raise ValueError("--offline_augmentation_backend applies only with --augmentation_execution offline")
    if str(args.jpeg_encode_backend) == "nvjpeg" and str(args.output_format) != "jpg":
        raise ValueError("--jpeg_encode_backend nvjpeg requires --output_format jpg/jpeg")
    tiff_encode_backend = str(getattr(args, "tiff_encode_backend", "auto"))
    custom_channel_output = any(
        str(variant.kind) == "custom" for variant in channel_variants
    )
    if tiff_encode_backend == "nvtiff" and not custom_channel_output:
        raise ValueError(
            "--tiff_encode_backend nvtiff currently applies to custom C...S... multipage TIFF output"
        )
    gpu_runtime_probe = ""
    if gpu_offline_active:
        if not topology.cuda_device_ids:
            raise RuntimeError(
                "build_gpu_augmentation was selected but no CUDA-visible GPU was discovered; "
                "check the Slurm GPU allocation and CUDA_VISIBLE_DEVICES"
            )
        if render_backend != "process" or str(args.worker_backend) == "thread":
            raise ValueError("GPU offline augmentation requires the persistent process backend")
        if not gpu_fork_render_backend_available():
            raise RuntimeError(
                "PTA build_gpu_augmentation offline execution currently retains its fork-only "
                "worker constraint and is unavailable on this host; CPU/no-augmentation process "
                "rendering uses spawn"
            )
        gpu_runtime_probe = probe_gpu_offline_runtime(
            require_nvjpeg=str(args.jpeg_encode_backend) == "nvjpeg",
            expected_device_count=len(topology.cuda_device_ids),
        )
        if custom_channel_output and tiff_encode_backend in {"auto", "nvtiff"}:
            nvtiff_module = importlib.import_module(".nvtiff_backend", package=__package__)
            cuda_version_match = re.search(r"torch_cuda=(\d+)", gpu_runtime_probe)
            capability = nvtiff_module.probe_nvtiff(
                cuda_major=(
                    int(cuda_version_match.group(1))
                    if cuda_version_match is not None
                    else None
                )
            )
            if not bool(capability.available) and tiff_encode_backend == "nvtiff":
                raise RuntimeError(
                    "--tiff_encode_backend nvtiff requires nvTIFF 0.8 or newer; "
                    "install nvidia-nvtiff-cu13 for CUDA 13 or nvidia-nvtiff-cu12 for CUDA 12; "
                    f"probe={capability.diagnostic}"
                )
            if bool(capability.available):
                version = ".".join(str(value) for value in capability.version)
                gpu_runtime_probe += f"; nvtiff={version}"
            else:
                gpu_runtime_probe += "; nvtiff=unavailable(auto->opencv)"
        gpu_count = len(topology.cuda_device_ids)
        frame_workers, gpu_render_threads = resolve_gpu_worker_layout(
            worker_budget=int(workers),
            requested_frame_workers=int(args.frame_workers),
            gpu_count=gpu_count,
        )
    elif str(args.jpeg_encode_backend) == "nvjpeg":
        raise ValueError(
            "nvJPEG output encoding is integrated with build_gpu_augmentation; "
            "select the GPU policy or use --jpeg_encode_backend opencv"
        )
    elif tiff_encode_backend == "nvtiff":
        raise ValueError(
            "nvTIFF multipage encoding is integrated with build_gpu_augmentation; "
            "select the GPU offline policy or use --tiff_encode_backend opencv"
        )
    if augmentation_definition is not None and float(args.augmentation_ratio) == 1.0:
        warnings.add(
            "augmentation_configured_without_copies",
            "--augmentation_ratio=1.0 retains only originals; use a value >1.0 to emit augmented copies",
        )
    if int(args.pipeline_depth) >= 2:
        if int(args.frame_workers) <= 0 and not gpu_offline_active:
            frame_workers = max(1, min(workers, int(math.ceil(float(workers) * 0.75))))
        render_cpu_budget = int(frame_workers) * (
            int(gpu_render_threads) if gpu_offline_active else 1
        )
        planning_workers = max(1, int(workers) - min(int(workers), render_cpu_budget))
    else:
        planning_workers = int(workers)
        render_cpu_budget = int(frame_workers) * (
            int(gpu_render_threads) if gpu_offline_active else 1
        )
    render_cpu_order = tuple(topology.worker_cpu_order) if bool(args.topology_aware) else tuple()
    render_cpu_set = {
        int(render_cpu_order[index % len(render_cpu_order)])
        for index in range(int(render_cpu_budget))
    } if render_cpu_order else set()
    planning_cpu_order = tuple(
        cpu for cpu in topology.allowed_cpus if int(cpu) not in render_cpu_set
    ) or tuple(topology.allowed_cpus)
    io_workers = max(1, min(16, visible_cpu_count(), planning_workers))
    runtime_label = (
        "gpu"
        if gpu_offline_active
        else ("cpu" if isinstance(augmentation, LoadedAugmentation) else "none")
    )
    print(
        f"Visible/allocated CPUs: {visible_cpu_count()}; worker budget: {workers}; frame workers: {frame_workers}; "
        f"GPU CPU-render threads/owner: {gpu_render_threads if gpu_offline_active else 'N/A'}; "
        f"planning workers: {planning_workers}; "
        f"raw prefetch I/O workers: {io_workers}; render backend: {render_backend}; "
        f"offline augmentation backend={runtime_label}"
    )
    print(f"Topology plan: {topology.summary}")
    if gpu_runtime_probe:
        print(f"GPU runtime probe: {gpu_runtime_probe}")
    print(
        "Channel output variants: "
        + ", ".join(
            f"{variant.tag_token}[offsets={list(variant.offsets)}]" for variant in channel_variants
        )
        + f"; output_format={args.output_format}"
    )
    specs = discover_volume_specs(args.input, force=bool(args.force), warnings=warnings)
    input_dir = Path(args.input).expanduser().resolve()
    default_output = Path.cwd() / input_dir.name
    out_dir = Path(args.output).expanduser().resolve() if args.output else default_output
    done_dir = out_dir / ".volume_done"
    v18_active = getattr(args, "_v18_config", None) is not None
    v18_input_identities: Optional[List[Dict[str, object]]] = None
    if v18_active:
        v18_input_identities = capture_v18_pta_input_identities(specs)
        validate_fresh_output_safety(
            out_dir,
            input_dir=input_dir,
            specs=specs,
            augmentation_path=(
                augmentation_definition.path
                if augmentation_definition is not None
                else None
            ),
        )
    if bool(args.resume) and not v18_active:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        clean_generated_output_dirs(out_dir)
        if v18_active:
            write_v18_output_sentinel(out_dir)

    labels_available = all(spec.label_source in {"yolo", "nrrd"} and spec.volume_class != "unlabeled" for spec in specs)
    create_output_dirs(
        out_dir,
        split_active=bool(split_active),
        train_split=train_split,
        labels_available=bool(labels_available),
        publish_images=bool(getattr(args, "save_images", True)),
        publish_labels=bool(getattr(args, "save_labels", True)),
    )
    deferred_policy_path: Optional[Path] = None
    if augmentation_definition is not None and str(args.augmentation_execution) == "deferred":
        deferred_policy_path = write_deferred_augmentation_bundle(
            out_dir,
            definition=augmentation_definition,
            requested_ratio=float(args.augmentation_ratio),
            split_active=bool(split_active),
        )
        print(
            f"Deferred augmentation: generator will write originals only; policy manifest={deferred_policy_path}. "
            "A training-loader hook is required to consume it."
        )

    completed_stems: set[str] = set()
    if bool(args.resume) and not v18_active and done_dir.exists():
        completed_stems = {p.stem for p in done_dir.glob("*.json")}
    pending_specs = [spec for spec in specs if spec.stem not in completed_stems]
    effective_pipeline_depth = int(args.pipeline_depth)
    if effective_pipeline_depth >= 2 and len(pending_specs) > 1:
        available_bytes = available_memory_budget_bytes()
        estimates = [estimate_spec_resident_bytes(spec) for spec in pending_specs]
        adjacent_pairs = [
            int(estimates[i]) + int(estimates[i + 1])
            for i in range(len(estimates) - 1)
            if estimates[i] is not None and estimates[i + 1] is not None
        ]
        worst_pair = max(adjacent_pairs) if adjacent_pairs else None
        if available_bytes is not None and worst_pair is not None and int(worst_pair) > int(0.70 * available_bytes):
            effective_pipeline_depth = 1
            message = (
                f"requested depth=2 estimated adjacent resident/scratch={worst_pair / GIB:.1f} GiB "
                f"exceeds 70% of currently available/cgroup memory={available_bytes / GIB:.1f} GiB; using depth=1"
            )
            warnings.add("pipeline_depth_reduced_for_memory", message)
            print(f"WARNING: {message}", file=sys.stderr)
    skipped_completed = len(specs) - len(pending_specs)
    if skipped_completed:
        print(f"Resume: skipping {skipped_completed} completed volume(s); {len(pending_specs)} remaining")
        warnings.add("resume_skipped_completed_volumes", f"{skipped_completed} volume(s) had {done_dir.name} markers")

    print(f"Discovered {len(specs)} volume(s): " + ", ".join(f"{s.stem}:{s.volume_class}/{s.kind}/{s.label_source}" for s in specs))
    print(
        "Pipelined pass: classify eligible copy-0 originals (YOLO geometry where possible), budget --background_percent per volume, "
        "then render every retained item/version in one grouped source-frame phase while the next volume plans"
    )

    volume_split_by_stem: Optional[Dict[str, str]] = None
    if split_active and split_method == "volume":
        assert train_split is not None
        volume_split_by_stem = assign_volume_split_by_stem(specs, train_split=float(train_split))

    background_filter_requested = float(args.background_percent) < 1.0
    dataset_publication_selected = bool(
        getattr(args, "save_images", True) or getattr(args, "save_labels", True)
    )
    total_background_stats = BackgroundFilterStats(
        active=background_filter_requested,
        classification_performed=bool(
            background_filter_requested and labels_available and dataset_publication_selected
        ),
    )
    if background_filter_requested and not dataset_publication_selected:
        total_background_stats.skipped_reason = "image/label publication is not selected"
    elif background_filter_requested and not labels_available:
        total_background_stats.skipped_reason = "label operations are disabled for unlabeled volumes"
    total_split_stats = SplitStats(active=bool(split_active), train_split=train_split, split_method=split_method)
    total_augmentation_stats = AugmentationStats(
        configured=augmentation_definition is not None,
        active=False,
        path=augmentation_definition.path if augmentation_definition is not None else None,
        content_sha256=augmentation_definition.content_sha256 if augmentation_definition is not None else "",
        export_name=augmentation_definition.export_name if augmentation_definition is not None else "",
        albumentations_version=augmentation.albumentations_version if augmentation is not None else "",
        runtime_backend=(
            augmentation.runtime_name
            if isinstance(augmentation, LoadedGpuAugmentation)
            else ("albumentations-cpu" if isinstance(augmentation, LoadedAugmentation) else "none")
        ),
        execution_mode=str(args.augmentation_execution) if augmentation_definition is not None else "none",
        deferred_policy_path=deferred_policy_path,
        requested_ratio=float(args.augmentation_ratio),
        applies_to=(
            ("training_loader_train" if split_active else "training_loader_all")
            if str(args.augmentation_execution) == "deferred"
            else ("train" if split_active else "all")
        ),
        foreground_classification_performed=False,
    )
    volume_records: List[VolumeSummaryRecord] = []
    used_augmentation_tags: set[str] = set()
    order_counter = 0
    total_written = 0
    total_flip_dropped = 0

    total_background_withheld = 0
    if pending_specs:
        # v18 (U-25): install run-constant parent/thread state, then create the
        # one persistent pool before any volume is resident.  Normal process
        # children receive a serialized static contract through spawn; the
        # external GPU policy is the documented fork-inheritance exception.
        set_worker_static_context(
            out_dir=out_dir,
            split_active=bool(split_active),
            image_format=str(args.output_format),
            png_compression=int(args.png_compression),
            jpeg_quality=int(args.jpeg_quality),
            jpeg_encode_backend=str(args.jpeg_encode_backend),
            tiff_encode_backend=str(args.tiff_encode_backend),
            gpu_batch_size=int(args.gpu_batch_size),
            gpu_render_threads=int(gpu_render_threads),
            gpu_device_ids=topology.cuda_device_ids if gpu_offline_active else (),
            augmentation=augmentation,
            save_images=bool(getattr(args, "save_images", True)),
            save_labels=bool(getattr(args, "save_labels", True)),
        )
        render_pool = PersistentRenderPool(
            backend=render_backend,
            workers=frame_workers,
            worker_cpu_order=render_cpu_order,
            gpu_device_ids=topology.cuda_device_ids if gpu_offline_active else (),
            gpu_cpu_sets=topology.gpu_cpu_sets if gpu_offline_active else (),
        )
        use_shared_volumes = render_backend == "process"
        volume_allocator: Optional[ArrayAllocator] = make_volume_allocator(True) if use_shared_volumes else None
        pipeline_enabled = int(effective_pipeline_depth) >= 2 and len(pending_specs) > 1
        if pipeline_enabled and bool(args.topology_aware) and planning_cpu_order:
            if bind_current_thread_to_cpus(planning_cpu_order):
                print(f"Topology plan: parent planning/prefetch threads restricted to CPUs {list(planning_cpu_order)}")
        print(
            f"Persistent render pool: backend={render_backend}, "
            f"start_method={render_pool.start_method}, "
            f"ranks={frame_workers}{' (one CUDA owner per visible GPU)' if gpu_offline_active else ''}, "
            f"gpu_cpu_render_threads={int(gpu_render_threads) if gpu_offline_active else 'N/A'}, "
            f"shared_memory_volumes={'on' if use_shared_volumes else 'off'}, source_frame_grouping=on, "
            f"pipeline_depth={int(effective_pipeline_depth)}, "
            f"jpeg_output_backend={str(args.jpeg_encode_backend)}, "
            f"tiff_output_backend={str(getattr(args, 'tiff_encode_backend', 'auto'))}, "
            f"gpu_batch_size={int(args.gpu_batch_size)}"
        )
        if len(pending_specs) > 1:
            warnings.add(
                "bounded_raw_volume_prefetch",
                f"depth=1, io_workers={io_workers}; raw source loads are decoupled from pool startup",
            )

        load_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pretrain-raw-prefetch")
        pending_loads: Dict[int, Tuple[Future, WarningLog]] = {}
        states: Dict[int, Dict[str, object]] = {}
        progress_by_gen: Dict[int, VolumeRenderProgress] = {}

        def _release_source_volume(src: SourceVolume) -> None:
            """Release a prefetched source that never reached a render state."""
            block = src.volume_block
            src.volume = np.empty((0,), dtype=np.uint8)
            src.mask_volume = None
            src.volume_block = None
            if block is not None:
                block.release()

        def _release_render_state(state: Dict[str, object]) -> None:
            """Release every parent-owned shared block in one render state."""
            progress = state.get("progress")
            if isinstance(progress, VolumeRenderProgress) and progress.pbar is not None:
                progress.pbar.close()
                progress.pbar = None

            blocks: List[SharedBlock] = []
            handles = state.get("handles")
            if isinstance(handles, list):
                for handle in handles:
                    if isinstance(handle, RenderPhaseHandle) and handle.payload_block is not None:
                        blocks.append(handle.payload_block)

            prep_obj = state.get("prep")
            if isinstance(prep_obj, PreparedVolume):
                blocks.extend(prep_obj.shm_blocks)
                prep_obj.src.volume = np.empty((0,), dtype=np.uint8)
                prep_obj.src.mask_volume = None
                prep_obj.src.volume_block = None
                prep_obj.volume_for_render = np.empty((0,), dtype=np.uint8)
                prep_obj.mask_for_render = np.empty((0,), dtype=np.uint8)
                prep_obj.shm_blocks = []

            gen = state.get("gen")
            if gen is not None:
                progress_by_gen.pop(int(gen), None)
            state["prep"] = None
            state["handles"] = []

            released_ids: set[int] = set()
            for block in blocks:
                block_id = id(block)
                if block_id not in released_ids:
                    block.release()
                    released_ids.add(block_id)

        def _submit_load(spec_index: int) -> None:
            if 0 <= spec_index < len(pending_specs) and spec_index not in pending_loads:
                load_warnings = WarningLog()
                future = load_executor.submit(
                    load_source_volume_from_spec,
                    pending_specs[spec_index],
                    warnings=load_warnings,
                    workers=io_workers,
                    allocator=volume_allocator,
                    jpeg_decode_backend=str(args.jpeg_decode_backend),
                    jpeg_batch_size=int(args.jpeg_batch_size),
                    jpeg_device_ids=topology.cuda_device_ids,
                    jpeg_cpu_sets=topology.gpu_cpu_sets,
                    load_cpu_order=planning_cpu_order if bool(args.topology_aware) else (),
                )
                pending_loads[spec_index] = (future, load_warnings)

        def _plan_volume(spec_index: int) -> Dict[str, object]:
            """Load-join + classify + budget + plan one volume (parent side).

            The pipeline can run this for volume k+1 while volume k renders on
            the pool.  Planning still executes in strict volume order, so
            order_counter assignment and augmentation tags are unchanged.
            """
            nonlocal order_counter
            future, vol_warnings = pending_loads.pop(spec_index)
            current_src = future.result()
            _submit_load(spec_index + 1)  # depth-1 prefetch, no fork constraint (G2)
            spec = pending_specs[spec_index]
            prep = prepare_loaded_source(
                current_src,
                args=args,
                warnings=vol_warnings,
                workers=planning_workers,
                out_dir=out_dir,
                tile_configs=tile_configs,
                channel_variants=channel_variants,
                requested_tilt_angles=requested_tilt_angles,
                requested_tilt_directions=requested_tilt_directions,
                write_side_effects=True,
                allocator=volume_allocator,
            )

            foregrounds: Optional[Dict[Tuple[str, int, str, int], bool]] = None
            if dataset_publication_selected:
                # Partial volumes are always classified on their eligible
                # annotated centers so annotated foreground/background remains
                # distinct even when --background_percent=1.0. Offline copies
                # also need source truth when the background cap is unrestricted.
                if prep.label_enabled and (
                    background_filter_requested
                    or prep.src.volume_class == "partially_labeled"
                    or (
                        str(args.augmentation_execution) == "offline"
                        and float(args.augmentation_ratio) > 1.0
                    )
                ):
                    foregrounds = classify_original_foregrounds_for_volume(
                        prep,
                        workers=planning_workers,
                        warnings=vol_warnings,
                    )
                cands = enumerate_candidates_for_volume(prep, foregrounds)
                validate_foreground_transverse_candidate_invariant(
                    prep,
                    cands,
                    retained_only=False,
                )
            else:
                # Diagnostics/manifests do not require dataset candidate
                # classification or augmentation planning work.
                cands = []
            for cand in cands:
                cand.order = order_counter
                cand.source_order = order_counter
                order_counter += 1

            subset_for_volume: Optional[str] = None
            if split_active:
                if volume_split_by_stem is not None:
                    subset_for_volume = volume_split_by_stem.get(spec.stem, "train")
                    for cand in cands:
                        cand.split_subset = subset_for_volume
                    vol_split_stats = SplitStats(active=True, train_split=train_split, split_method=split_method)
                else:
                    # view/slice atomic units never cross volumes, so the
                    # digest-ranked assignment is applied within each
                    # volume; per-volume rounding replaces the earlier
                    # dataset-wide rounding of the train quota.
                    vol_split_stats = apply_dataset_split(
                        cands,
                        active=True,
                        train_split=train_split,
                        split_method=split_method,
                        warnings=vol_warnings,
                        emit_warnings=False,
                    )
            else:
                vol_split_stats = SplitStats(active=False, train_split=train_split, split_method=split_method)

            base_background_stats = apply_background_filter(
                cands,
                background_percent=float(args.background_percent),
                labels_available=bool(labels_available),
                warnings=vol_warnings,
            )
            physical, vol_augmentation_stats = plan_augmented_versions(
                cands,
                augmentation=augmentation,
                augmentation_definition=augmentation_definition,
                execution_mode=str(args.augmentation_execution),
                augmentation_ratio=float(args.augmentation_ratio),
                split_active=bool(split_active),
                augmented_foregrounds={},
                require_augmented_foregrounds=False,
                used_tags=used_augmentation_tags,
            )
            vol_background_stats = finalize_background_filter_with_augmentations(
                physical,
                base_stats=base_background_stats,
                background_percent=float(args.background_percent),
                labels_available=bool(labels_available),
                warnings=vol_warnings,
            )
            # Diagnostics-only runs intentionally skip dataset candidate
            # enumeration, leaving ``physical`` empty even when the source has
            # foreground.  The retained-candidate invariant applies only when
            # images or labels were selected for publication.
            if dataset_publication_selected:
                validate_foreground_transverse_candidate_invariant(
                    prep,
                    physical,
                    retained_only=True,
                )
            if split_active:
                if volume_split_by_stem is not None:
                    retained_originals = [c for c in physical if c.keep and int(c.augmentation_index) == 0]
                    is_train = subset_for_volume == "train"
                    vol_split_stats.atomic_units_total = 1
                    vol_split_stats.atomic_units_train = 1 if is_train else 0
                    vol_split_stats.atomic_units_val = 0 if is_train else 1
                    vol_split_stats.frames_total = len(retained_originals)
                    vol_split_stats.frames_train = len(retained_originals) if is_train else 0
                    vol_split_stats.frames_val = 0 if is_train else len(retained_originals)
                else:
                    vol_split_stats = refresh_retained_original_split_stats(
                        physical,
                        stats=vol_split_stats,
                        train_split=train_split,
                        split_method=split_method,
                        warnings=vol_warnings,
                        emit_warnings=False,
                    )
            return {
                "spec": spec,
                "gen": int(spec_index),
                "warnings": vol_warnings,
                "prep": prep,
                "cands": cands,
                "physical": physical,
                "split_stats": vol_split_stats,
                "background_stats": vol_background_stats,
                "augmentation_stats": vol_augmentation_stats,
                "handles": [],
                "progress": None,
            }

        def _submit_phase_a(state: Dict[str, object]) -> None:
            """Enqueue all retained candidates in one source-frame-grouped phase."""
            prep: PreparedVolume = state["prep"]  # type: ignore[assignment]
            physical: List[OutputCandidate] = state["physical"]  # type: ignore[assignment]
            gen = int(state["gen"])  # type: ignore[arg-type]
            progress = VolumeRenderProgress(stem=prep.src.stem, warnings=state["warnings"])  # type: ignore[arg-type]
            state["progress"] = progress
            progress_by_gen[gen] = progress
            volume_publication_selected = bool(
                bool(getattr(args, "save_images", True))
                or (
                    bool(getattr(args, "save_labels", True))
                    and bool(prep.label_enabled)
                )
            )
            if not volume_publication_selected:
                print(
                    f"{prep.src.stem}: skipping dataset render because neither "
                    "images nor available labels was selected"
                )
                return
            retained = [c for c in physical if c.keep]
            if not retained:
                print(f"{prep.src.stem}: no surviving candidates to render")
                return
            print(f"{prep.src.stem}: ordered projection phases:")
            for family, family_count, cumulative_end in projection_phase_summary(
                prep.plans,
                retained,
            ):
                cuda_family = family in {
                    "cartesian", "upright-radial", "tilted-cartesian", "tilted-radial",
                }
                backend_note = (
                    "resident CUDA when admitted; CPU fallback"
                    if gpu_offline_active and cuda_family
                    else "CPU"
                )
                print(
                    f"  {family}: candidates={family_count}, "
                    f"cumulative_end={cumulative_end}, projection={backend_note}"
                )
            tasks = build_phase_render_tasks(
                prep.plans,
                retained,
                aug_chunk=int(args.aug_task_chunk),
                gpu_batch_size=(int(args.gpu_batch_size) if gpu_offline_active else 0),
            )
            task_description = (
                f"{len(tasks)} coalesced GPU work unit(s), up to "
                f"{max(1, int(args.gpu_batch_size)) * 2} candidates each"
                if gpu_offline_active
                else f"{len(tasks)} grouped source-frame task(s)"
            )
            print(
                f"{prep.src.stem}: rendering retained candidates={len(retained)} "
                f"in {task_description}, backend={render_backend}, workers={frame_workers}"
            )
            publication_kinds: List[str] = []
            if bool(getattr(args, "save_images", True)):
                publication_kinds.append("images")
            if bool(getattr(args, "save_labels", True)) and bool(prep.label_enabled):
                publication_kinds.append("labels")
            progress.pbar = tqdm(
                total=len(retained),
                desc=f"Rendering retained {'/'.join(publication_kinds)} {prep.src.stem}",
            )
            if tasks:
                handle = render_pool.install_phase(gen=gen, prep=prep, tasks=tasks)
                state["handles"].append(handle)  # type: ignore[union-attr]
                progress.pending_a = render_pool.submit_phase(handle, meta=(gen, "A"))

        def _submit_phase_b(state: Dict[str, object]) -> int:
            """C3: no second render phase; any rare offline flip overage is trimmed."""
            return 0

        def _finalize_volume(state: Dict[str, object], withheld: int) -> None:
            nonlocal total_written, total_flip_dropped, total_background_withheld
            prep: PreparedVolume = state["prep"]  # type: ignore[assignment]
            spec: VolumeInputSpec = state["spec"]  # type: ignore[assignment]
            physical: List[OutputCandidate] = state["physical"]  # type: ignore[assignment]
            cands: List[OutputCandidate] = state["cands"]  # type: ignore[assignment]
            progress: VolumeRenderProgress = state["progress"]  # type: ignore[assignment]
            vol_warnings: WarningLog = state["warnings"]  # type: ignore[assignment]
            vol_background_stats: BackgroundFilterStats = state["background_stats"]  # type: ignore[assignment]
            if progress.pbar is not None:
                progress.pbar.close()
            flips_by_subset = dict(progress.flips_by_subset)
            flip_dropped = sum(int(x) for x in flips_by_subset.values())
            if flip_dropped:
                print(f"{prep.src.stem}: dropped {flip_dropped} presumed-foreground augmented copy/copies that rendered background")
            if withheld:
                print(f"{spec.stem}: withheld {withheld} background(s) pre-render to honor the realized --background_percent cap after flips")
            # Offline foreground flips are rare.  C3 deliberately favors one
            # source render over a second background phase; trim only the
            # deterministic overage after observing those flips.
            trimmed = trim_background_overage_after_flips(
                physical,
                flips_by_subset=flips_by_subset,
                background_percent=float(args.background_percent),
                labels_available=bool(labels_available),
                out_dir=out_dir,
                split_active=bool(split_active),
                image_format=str(args.output_format),
                background_stats=vol_background_stats,
                warnings=vol_warnings,
                images_selected=bool(getattr(args, "save_images", True)),
                labels_selected=bool(getattr(args, "save_labels", True)),
            )
            if trimmed:
                print(f"{spec.stem}: trimmed {trimmed} written background(s) to honor the realized --background_percent cap after flips")
            written_effective = int(progress.written) - int(trimmed)
            # Recompute retention-derived stats after flip drops and the
            # realized-cap withhold so the summary reflects the final file set.
            vol_augmentation_stats = finalize_augmentation_stats(
                state["augmentation_stats"],  # type: ignore[arg-type]
                physical,
                split_active=bool(split_active),
            )
            vol_augmentation_stats.retained_augmented_copies = max(
                0, int(vol_augmentation_stats.retained_augmented_copies) - int(flip_dropped)
            )
            vol_augmentation_stats.dropped_augmented_background = max(
                0,
                int(vol_augmentation_stats.planned_augmented_copies)
                - int(vol_augmentation_stats.retained_augmented_copies),
            )
            total_written += int(written_effective)
            total_flip_dropped += int(flip_dropped)
            total_background_withheld += int(withheld)

            if prep.save_overlay:
                write_overlays_global(
                    volume=prep.volume_for_render,
                    mask=prep.mask_for_render,
                    plans=prep.plans,
                    fps=float(prep.src.fps),
                    overlay_tile_writer_limit=int(args.overlay_tile_writer_limit),
                    workers=max(1, int(frame_workers)),
                    overlay_workers=int(args.overlay_workers),
                    overlay_pending_frames=int(args.overlay_pending_frames),
                    warnings=vol_warnings,
                )

            retained = [c for c in physical if c.keep]
            rec = VolumeSummaryRecord(
                stem=prep.src.stem,
                input_kind=prep.src.kind,
                volume_class=prep.src.volume_class,
                effective_volume_class=prep.effective_volume_class,
                label_source=prep.src.label_source,
                label_enabled=prep.label_enabled,
                source_shape=prep.source_shape,
                processing_shape=prep.processing_shape,
                fps=float(prep.src.fps),
                input_start_index=prep.src.input_start_index,
                encoded_indices=prep.src.encoded_indices,
                annotation_state_counts=annotation_state_counts(prep.annotation_states),
                foreground_preservation_stats=dict(prep.foreground_preservation_stats),
                voxel_initial=prep.voxel_initial,
                voxel_final=prep.voxel_final,
                smoothing_stats=list(prep.smoothing_stats),
                nrrd_paths=list(prep.nrrd_paths),
                views=list(prep.views),
                tile_configs=list(tile_configs),
                render_stats=render_stats_from_plans(prep.plans, retained),
                candidates_total=len(cands),
                candidates_retained=sum(1 for c in cands if c.keep),
                augmented_candidates_planned=sum(1 for c in physical if int(c.augmentation_index) > 0),
                augmented_candidates_retained=sum(1 for c in physical if int(c.augmentation_index) > 0 and c.keep),
                candidates_written=int(written_effective),
            )
            volume_records.append(rec)

            _accumulate_background_stats(total_background_stats, vol_background_stats)
            _accumulate_augmentation_stats(total_augmentation_stats, vol_augmentation_stats)
            if split_active:
                _accumulate_split_stats(total_split_stats, state["split_stats"])  # type: ignore[arg-type]
            warnings.merge_from(vol_warnings)

            if not v18_active:
                done_dir.mkdir(parents=True, exist_ok=True)
                marker = {
                    "stem": str(spec.stem),
                    "pipeline_spec_version": PIPELINE_SPEC_VERSION,
                    "candidates_total": len(cands),
                    "candidates_retained": int(len(retained)),
                    "candidates_written": int(written_effective),
                    "flip_dropped": int(flip_dropped),
                    "background_withheld_after_flips": int(withheld),
                    "background_trimmed_after_flips": int(trimmed),
                    "completed_at_unix": float(time.time()),
                }
                (done_dir / f"{spec.stem}.json").write_text(json.dumps(marker, indent=2) + "\n")
            completion_metric = "processed" if v18_active else "written"
            print(
                f"{spec.stem}: volume complete; {completion_metric}={written_effective}, "
                f"flip_dropped={flip_dropped}, background_withheld={withheld}, "
                f"background_trimmed={trimmed}"
            )

            # G2: workers have finished with this volume.  Drop parent-side
            # array views, close the mappings, and unlink their names.
            _release_render_state(state)

        load_executor_stopped = False
        try:
            _submit_load(0)
            states[0] = _plan_volume(0)
            _submit_phase_a(states[0])
            for spec_index in range(len(pending_specs)):
                state = states[spec_index]
                progress: VolumeRenderProgress = state["progress"]  # type: ignore[assignment]
                if pipeline_enabled and spec_index + 1 < len(pending_specs):
                    # G5: plan volume k+1 on the parent while volume k renders.
                    states[spec_index + 1] = _plan_volume(spec_index + 1)
                drain_render_results(render_pool, progress_by_gen, until=lambda p=progress: p.pending_a == 0)
                withheld = _submit_phase_b(state)
                if pipeline_enabled and spec_index + 1 in states:
                    # Enqueue the next volume immediately after the current
                    # grouped phase so the persistent pool queue stays warm.
                    _submit_phase_a(states[spec_index + 1])
                drain_render_results(render_pool, progress_by_gen, until=lambda p=progress: p.pending_b == 0)
                _finalize_volume(state, withheld)
                del states[spec_index]
                if not pipeline_enabled and spec_index + 1 < len(pending_specs):
                    states[spec_index + 1] = _plan_volume(spec_index + 1)
                    _submit_phase_a(states[spec_index + 1])
        except BaseException:
            # Stop workers before unlinking blocks they may still be reading.
            # Also join the one prefetch thread so a late allocation cannot
            # escape cleanup after the exception has propagated.
            try:
                render_pool.close(terminate=True)
            except Exception:
                pass
            for state in list(states.values()):
                _release_render_state(state)
            states.clear()
            load_executor.shutdown(wait=True, cancel_futures=True)
            load_executor_stopped = True
            for future, _load_warnings in list(pending_loads.values()):
                if future.cancelled():
                    continue
                try:
                    loaded = future.result()
                except BaseException:
                    continue
                if isinstance(loaded, SourceVolume):
                    _release_source_volume(loaded)
            pending_loads.clear()
            raise
        else:
            render_pool.close()
        finally:
            if not load_executor_stopped:
                load_executor.shutdown(wait=False, cancel_futures=True)
    else:
        if v18_active:
            print("No render items were scheduled; zero-view PTA completed successfully")
        else:
            print("All volumes already have completion markers; nothing to render")

    if total_augmentation_stats.eligible_originals > 0:
        total_augmentation_stats.achieved_ratio = (
            float(total_augmentation_stats.eligible_originals + total_augmentation_stats.retained_augmented_copies)
            / float(total_augmentation_stats.eligible_originals)
        )
    else:
        total_augmentation_stats.achieved_ratio = 1.0
    if split_active and total_split_stats.frames_total > 0:
        assert train_split is not None
        total_split_stats.achieved_train_fraction = float(total_split_stats.frames_train) / float(total_split_stats.frames_total)
        if abs(total_split_stats.achieved_train_fraction - float(train_split)) > 0.10:
            total_split_stats.warning = (
                f"achieved train fraction {total_split_stats.achieved_train_fraction:.6f} differs from requested "
                f"--train_split {float(train_split):.6f} by more than 0.10"
            )
        if float(train_split) < 1.0 and total_split_stats.frames_val == 0:
            extra = "val split is empty while --train_split < 1.0"
            total_split_stats.warning = f"{total_split_stats.warning}; {extra}" if total_split_stats.warning else extra
        if total_split_stats.warning:
            warnings.add("split_best_effort_warning", total_split_stats.warning)
            print(f"WARNING: {total_split_stats.warning}", file=sys.stderr)

    dataset_yaml_path: Optional[Path] = None
    if (
        split_active
        and bool(getattr(args, "save_images", True))
        and bool(getattr(args, "save_labels", True))
    ):
        assert train_split is not None and dataset_channels is not None
        dataset_yaml_path = write_dataset_yaml(
            out_dir,
            train_split=float(train_split),
            channels=int(dataset_channels),
        )

    publication_integrity: Dict[str, object] = {
        "selected": bool(getattr(args, "save_images", True)),
        "verified_image_count": 0,
        "verified_total_bytes": 0,
    }
    if v18_active and bool(getattr(args, "save_images", True)):
        publication_integrity = verify_published_image_tree(
            out_dir,
            expected_count=int(total_written),
            image_format=str(args.output_format),
        )
        print(
            "PTA image publication verified: "
            f"count={publication_integrity['verified_image_count']}, "
            f"bytes={publication_integrity['verified_total_bytes']}"
        )

    summary_path: Optional[Path] = None
    if bool(getattr(args, "save_summary", True)):
        summary_command = (
            shlex.join(
                [
                    SCRIPT_BASENAME,
                    "--mode",
                    "pta",
                    *(str(x) for x in cli_argv),
                ]
            )
            if v18_active
            else shlex.join([str(sys.argv[0]), *(str(x) for x in cli_argv)])
        )
        summary_path = write_pta_summary(
            out_dir / "summary.txt",
            command=summary_command,
            input_dir=input_dir,
            out_dir=out_dir,
            specs=specs,
            volume_records=volume_records,
            tile_configs=tile_configs,
            channel_formats=channel_formats,
            channel_variants=channel_variants,
            requested_output_format=requested_output_format,
            background_stats=total_background_stats,
            split_stats=total_split_stats,
            augmentation_stats=total_augmentation_stats,
            dataset_yaml_path=dataset_yaml_path,
            warnings=warnings,
            workers=workers,
            frame_workers=frame_workers,
            planning_workers=planning_workers,
            topology_summary=topology.summary,
            jpeg_decode_backend=str(args.jpeg_decode_backend),
            jpeg_batch_size=int(args.jpeg_batch_size),
            jpeg_encode_backend=str(args.jpeg_encode_backend),
            jpeg_quality=int(args.jpeg_quality),
            gpu_batch_size=int(args.gpu_batch_size),
            image_format=str(args.output_format),
            png_compression=int(args.png_compression),
            tiff_encode_backend=str(getattr(args, "tiff_encode_backend", "auto")),
        )

    manifest_path: Optional[Path] = None
    voxel_report_path: Optional[Path] = None
    if v18_active:
        if bool(getattr(args, "voxel_volume", False)):
            voxel_report_path = write_v18_voxel_volume_report(
                out_dir / "voxel_volume.json",
                volume_records,
            )
        _cleanup_v18_pta_selected_run_work(out_dir / ".v18_work")
        if v18_input_identities is None:  # pragma: no cover - launch invariant
            raise RuntimeError("v18 PTA input identities were not captured")
        assert_v18_pta_inputs_unchanged(v18_input_identities)
        assert_augmentation_definition_unchanged(augmentation_definition)
        # This is deliberately the final selected artifact: a complete manifest
        # can never describe a run whose summary/voxel publication or cleanup failed.
        manifest_path = write_v18_pta_manifest(
            out_dir / "manifest.json",
            args=args,
            cli_argv=cli_argv,
            specs=specs,
            records=volume_records,
            channel_variants=channel_variants,
            tile_configs=tile_configs,
            augmentation_stats=total_augmentation_stats,
            total_written=int(total_written),
            input_identities=v18_input_identities,
            render_backend=str(render_backend),
            workers=int(workers),
            frame_workers=int(frame_workers),
            planning_workers=int(planning_workers),
            topology_summary=str(topology.summary),
            summary_path=summary_path,
            voxel_report_path=voxel_report_path,
            dataset_yaml_path=dataset_yaml_path,
            publication_integrity=publication_integrity,
        )

    print("\nDone.")
    print(f"Output directory: {out_dir}")
    if v18_active:
        print(
            f"Candidate versions processed this run: {total_written} "
            f"(flip-dropped augmented copies: {total_flip_dropped}, "
            f"backgrounds withheld pre-render: {total_background_withheld})"
        )
        print(
            "PTA publication tokens: "
            + (", ".join(getattr(args._v18_config.save, "tokens", ())) or "none")
        )
    else:
        print(
            f"Dataset samples written this run: {total_written} "
            f"(flip-dropped augmented copies: {total_flip_dropped}, "
            f"backgrounds withheld pre-render: {total_background_withheld})"
        )
    if dataset_yaml_path is not None:
        print(f"Dataset YAML: {dataset_yaml_path}")
    if summary_path is not None:
        print(f"Summary: {summary_path}")
    if manifest_path is not None:
        print(f"Manifest: {manifest_path}")
    if voxel_report_path is not None:
        print(f"Voxel-count report: {voxel_report_path}")


if __name__ == "__main__":
    main()
