"""Lazy CUDA acceleration for bridge interpolation.

The production interpolation planner deliberately remains host-side: it owns sparse
label stores and Python component records that are cheap to traverse on the CPU.  This
module accelerates the dense work inside each accepted bridge plan (SDF blending,
component filtering, hole filling, radius estimation and crop-bounded painting).

No CUDA package is imported at module-import time.  A renderer is admitted only inside
an already-warm CUDA worker by default, which avoids creating one CUDA context per
dedicated interpolation child.  Every device result is copied into a temporary host crop
before the caller commits it to the bridge canvas, so the CPU fallback can safely replay
a failed batch.
"""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .workspace import _env_flag, _env_int


MIB = 1024 ** 2


def gpu_interpolation_enabled() -> bool:
    """Whether CUDA bridge interpolation should be attempted."""

    return _env_flag("YOLO_TTA_GPU_INTERPOLATION", True)


def gpu_interpolation_required() -> bool:
    """Whether inability to use CUDA is fatal instead of a CPU fallback."""

    return _env_flag("YOLO_TTA_GPU_INTERPOLATION_REQUIRED", False)


def gpu_interpolation_create_context_enabled() -> bool:
    """Allow a dedicated interpolation child to create a new CUDA context."""

    return _env_flag("YOLO_TTA_GPU_INTERPOLATION_CREATE_CONTEXT", False)


def gpu_interpolation_main_process_enabled() -> bool:
    """Allow CUDA interpolation in the main process rather than a leased worker."""

    return _env_flag("YOLO_TTA_GPU_INTERPOLATION_MAIN_PROCESS", False)


def gpu_interpolation_cache_bytes() -> int:
    """Maximum live device bytes retained for plan SDFs and accepted sections."""

    return max(0, _env_int("YOLO_TTA_GPU_INTERPOLATION_CACHE_MIB", 1024)) * MIB


def gpu_interpolation_reserve_bytes() -> int:
    """Free-VRAM headroom required at admission and withheld from the live cache."""

    return max(0, _env_int("YOLO_TTA_GPU_INTERPOLATION_RESERVE_MIB", 1024)) * MIB


class GpuInterpolationUnavailable(RuntimeError):
    """Raised when a renderer was disabled or cannot execute a requested operation."""


@dataclass(frozen=True)
class GpuInterpolationSliceResult:
    added_voxels: int
    rendered_sections: int
    bbox: Optional[Tuple[int, int, int, int]]


@dataclass(frozen=True)
class _SlicePlacement:
    src_y0: int
    src_y1: int
    src_x0: int
    src_x1: int
    dst_y0: int
    dst_y1: int
    dst_x0: int
    dst_x1: int
    mirrored: bool


@dataclass(frozen=True)
class _DeviceCacheEntry:
    value: object
    nbytes: int
    # Keep the host owners alive for as long as an id-derived cache key is live.
    # Without this, a rejected plan can die before eviction and CPython may reuse
    # its ndarray/list id for an unrelated plan, producing a stale device hit.
    owners: Tuple[object, ...]


def _slice_job_placement(
    plan: object,
    step_idx: int,
    destination_shape: Sequence[int],
) -> Optional[_SlicePlacement]:
    """Return the exact clipped placement used by the legacy CPU painter."""

    steps = int(getattr(plan, "steps"))
    if int(step_idx) <= 0 or int(step_idx) >= steps:
        return None
    height, width = int(destination_shape[0]), int(destination_shape[1])
    sdf0 = np.asarray(getattr(plan, "sdf0"))
    size_y, size_x = int(sdf0.shape[0]), int(sdf0.shape[1])
    alpha = float(step_idx) / float(steps)
    source_anchor = getattr(plan, "source_anchor")
    target_anchor = getattr(plan, "target_anchor")
    center_y = (
        (1.0 - alpha) * float(source_anchor[0])
        + alpha * float(target_anchor[0])
    )
    center_x = (
        (1.0 - alpha) * float(source_anchor[1])
        + alpha * float(target_anchor[1])
    )
    source_point = getattr(plan, "source_point")
    s_raw = int(source_point[0]) + int(getattr(plan, "sign")) * int(step_idx)
    num_slices = int(getattr(plan, "num_slices"))
    mirrored = bool(num_slices > 0 and (s_raw < 0 or s_raw >= num_slices))
    if mirrored:
        center_x = float(width - 1) - center_x

    cy = int(round(float(center_y)))
    cx = int(round(float(center_x)))
    y0 = int(cy - size_y // 2)
    x0 = int(cx - size_x // 2)
    y1 = int(y0 + size_y)
    x1 = int(x0 + size_x)
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    src_y1 = int(size_y - max(0, y1 - height))
    src_x1 = int(size_x - max(0, x1 - width))
    dst_y0 = int(y0 + src_y0)
    dst_x0 = int(x0 + src_x0)
    dst_y1 = int(y1 - max(0, y1 - height))
    dst_x1 = int(x1 - max(0, x1 - width))
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return None
    return _SlicePlacement(
        src_y0=int(src_y0), src_y1=int(src_y1),
        src_x0=int(src_x0), src_x1=int(src_x1),
        dst_y0=int(dst_y0), dst_y1=int(dst_y1),
        dst_x0=int(dst_x0), dst_x1=int(dst_x1),
        mirrored=bool(mirrored),
    )


def _scalar_int(value: object) -> int:
    item = getattr(value, "item", None)
    return int(item() if callable(item) else value)


def _scalar_float(value: object) -> float:
    item = getattr(value, "item", None)
    return float(item() if callable(item) else value)


class _CupyInterpolationRuntime:
    """Small adapter that keeps the renderer testable without CUDA hardware."""

    def __init__(self, device_index: int) -> None:
        import cupy as cp  # type: ignore
        import cupyx.scipy.ndimage as cpx_ndi  # type: ignore

        self.xp = cp
        self.ndi = cpx_ndi
        self.device_index = int(device_index)
        self.device = cp.cuda.Device(int(device_index))

    def activate(self) -> object:
        return self.device

    def to_device(self, value: np.ndarray) -> object:
        return self.xp.asarray(value)

    def to_host(self, value: object) -> np.ndarray:
        # cp.asnumpy has been a blocking copy since the oldest supported CuPy releases.
        return np.asarray(self.xp.asnumpy(value))

    def mem_info(self) -> Tuple[int, int]:
        # Construction alone does not make cp.cuda.Device current on this thread.
        # Query inside the selected context so multi-GPU admission is device-correct.
        with self.device:
            free_bytes, total_bytes = self.xp.cuda.runtime.memGetInfo()
        return int(free_bytes), int(total_bytes)

    def synchronize(self) -> None:
        self.xp.cuda.get_current_stream().synchronize()

    def free_cached_memory(self) -> None:
        self.synchronize()
        self.xp.get_default_memory_pool().free_all_blocks()
        pinned_pool = getattr(self.xp, "get_default_pinned_memory_pool", None)
        if callable(pinned_pool):
            pinned_pool().free_all_blocks()


class CudaInterpolationRenderer:
    """Crop-bounded CuPy renderer/evaluator for host-planned interpolation bridges."""

    def __init__(
        self,
        device_index: int = 0,
        *,
        runtime: Optional[object] = None,
        cache_bytes: Optional[int] = None,
        reserve_bytes: Optional[int] = None,
    ) -> None:
        self.runtime = (
            runtime if runtime is not None
            else _CupyInterpolationRuntime(int(device_index))
        )
        self.xp = getattr(self.runtime, "xp")
        self.ndi = getattr(self.runtime, "ndi")
        self.device_index = int(getattr(self.runtime, "device_index", device_index))
        self.reserve_bytes = int(
            gpu_interpolation_reserve_bytes()
            if reserve_bytes is None else max(0, int(reserve_bytes))
        )
        requested_cache = int(
            gpu_interpolation_cache_bytes()
            if cache_bytes is None else max(0, int(cache_bytes))
        )
        try:
            free_bytes, _total_bytes = self.runtime.mem_info()
        except Exception:
            free_bytes = int(requested_cache + self.reserve_bytes)
        self.cache_budget_bytes = int(min(
            int(requested_cache), max(0, int(free_bytes) - int(self.reserve_bytes)),
        ))
        self._cache: "OrderedDict[Tuple[object, ...], _DeviceCacheEntry]" = OrderedDict()
        self._cache_live_bytes = 0
        self._lock = threading.RLock()
        self._failed_reason: Optional[str] = None
        self._metrics: Dict[str, int] = {
            "estimated_plans": 0,
            "estimated_sections": 0,
            "rendered_slices": 0,
            "rendered_sections": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_peak_bytes": 0,
        }
        self._structure8 = None

    @property
    def available(self) -> bool:
        return self._failed_reason is None

    @property
    def failed_reason(self) -> Optional[str]:
        return self._failed_reason

    def disable(self, exc: BaseException) -> bool:
        """Disable future CUDA work; return True only for the first failure."""

        reason = f"{type(exc).__name__}: {exc}"
        with self._lock:
            first = self._failed_reason is None
            if first:
                self._failed_reason = reason
                self._clear_cache()
            return bool(first)

    def _ensure_available(self) -> None:
        if self._failed_reason is not None:
            raise GpuInterpolationUnavailable(self._failed_reason)

    def _cache_get(
        self,
        key: Tuple[object, ...],
        owners: Sequence[object] = (),
    ) -> Optional[object]:
        entry = self._cache.pop(key, None)
        if entry is None:
            self._metrics["cache_misses"] += 1
            return None
        if owners and (
            len(entry.owners) != len(owners)
            or any(cached is not current for cached, current in zip(entry.owners, owners))
        ):
            self._cache_live_bytes = max(
                0, self._cache_live_bytes - int(entry.nbytes),
            )
            self._metrics["cache_misses"] += 1
            return None
        self._cache[key] = entry
        self._metrics["cache_hits"] += 1
        return entry.value

    def _cache_put(
        self,
        key: Tuple[object, ...],
        value: object,
        nbytes: int,
        owners: Sequence[object] = (),
    ) -> None:
        size = max(0, int(nbytes))
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_live_bytes = max(0, self._cache_live_bytes - int(old.nbytes))
        if self.cache_budget_bytes <= 0 or size > self.cache_budget_bytes:
            return
        while self._cache and self._cache_live_bytes + size > self.cache_budget_bytes:
            _old_key, old_entry = self._cache.popitem(last=False)
            self._cache_live_bytes = max(
                0, self._cache_live_bytes - int(old_entry.nbytes),
            )
        self._cache[key] = _DeviceCacheEntry(
            value=value,
            nbytes=int(size),
            owners=tuple(owners),
        )
        self._cache_live_bytes += int(size)
        self._metrics["cache_peak_bytes"] = max(
            int(self._metrics["cache_peak_bytes"]), int(self._cache_live_bytes),
        )

    def _clear_cache(self) -> None:
        self._cache.clear()
        self._cache_live_bytes = 0

    def _device_array(self, value: np.ndarray) -> object:
        contiguous = np.ascontiguousarray(value)
        result = self.runtime.to_device(contiguous)
        self._metrics["host_to_device_bytes"] += int(contiguous.nbytes)
        return result

    def _device_sdfs(self, plan: object) -> Tuple[object, object]:
        sdf0_owner = getattr(plan, "sdf0")
        sdf1_owner = getattr(plan, "sdf1")
        sdf0_host = np.asarray(sdf0_owner, dtype=np.float32)
        sdf1_host = np.asarray(sdf1_owner, dtype=np.float32)
        key = ("sdf", id(sdf0_owner), id(sdf1_owner))
        cached = self._cache_get(key, (sdf0_owner, sdf1_owner))
        if cached is not None:
            return cached  # type: ignore[return-value]
        sdf0 = self._device_array(sdf0_host)
        sdf1 = self._device_array(sdf1_host)
        pair = (sdf0, sdf1)
        self._cache_put(
            key,
            pair,
            int(sdf0_host.nbytes + sdf1_host.nbytes),
            (sdf0_owner, sdf1_owner),
        )
        return pair

    def _blend_section(self, sdf0: object, sdf1: object, alpha: float) -> object:
        # Separate multiply/add ufunc launches avoid device FMA changing a zero-boundary
        # decision relative to NumPy's float32 expression.
        left = self.xp.multiply(sdf0, np.float32(1.0 - float(alpha)))
        right = self.xp.multiply(sdf1, np.float32(float(alpha)))
        return self.xp.add(left, right) >= np.float32(0.0)

    def _keep_center_component_and_fill(self, section: object) -> Optional[object]:
        if self._structure8 is None:
            self._structure8 = self.xp.ones((3, 3), dtype=self.xp.bool_)
        labels, num = self.ndi.label(section, structure=self._structure8)
        component_count = _scalar_int(num)
        if component_count <= 0:
            return None
        if component_count == 1:
            kept = section
        else:
            height, width = int(section.shape[0]), int(section.shape[1])
            cy, cx = int(height // 2), int(width // 2)
            keep_label = _scalar_int(labels[cy, cx])
            if keep_label <= 0:
                indices = self.xp.flatnonzero(section)
                if int(indices.size) <= 0:
                    return None
                ys = indices // int(width)
                xs = indices - ys * int(width)
                d2 = (ys - int(cy)) ** 2 + (xs - int(cx)) ** 2
                nearest = _scalar_int(indices[_scalar_int(self.xp.argmin(d2))])
                keep_label = _scalar_int(labels.reshape(-1)[int(nearest)])
            kept = labels == int(keep_label)
        return self.ndi.binary_fill_holes(kept)

    def _distance_transform(self, section: object) -> object:
        try:
            return self.ndi.distance_transform_edt(
                section, return_indices=False, float64_distances=False,
            )
        except TypeError:
            # CuPy versions predating the float32 extension still provide the exact EDT.
            return self.ndi.distance_transform_edt(section, return_indices=False)

    def _section_radius(self, section: object) -> float:
        if bool(_scalar_int(self.xp.all(section))):
            return float(max(int(section.shape[0]), int(section.shape[1])))
        distances = self._distance_transform(section)
        return _scalar_float(self.xp.max(distances))

    def preflight(self) -> None:
        """Exercise every required CuPyX primitive before any host canvas can change."""

        with self._lock, self.runtime.activate():
            self._ensure_available()
            probe = np.zeros((5, 5), dtype=bool)
            probe[1:4, 1:4] = True
            probe[2, 2] = False
            device_probe = self._device_array(probe)
            kept = self._keep_center_component_and_fill(device_probe)
            if kept is None:
                raise RuntimeError("CUDA interpolation preflight produced an empty component")
            radius = self._section_radius(kept)
            host = self.runtime.to_host(kept)
            self._metrics["device_to_host_bytes"] += int(host.nbytes)
            if host.shape != probe.shape or not bool(host[2, 2]) or radius <= 0.0:
                raise RuntimeError("CUDA interpolation preflight failed its morphology check")
            self._clear_cache()
            for key in self._metrics:
                self._metrics[key] = 0

    def estimate_min_radius(
        self,
        plan: object,
        *,
        reject_at_or_below: float = 0.0,
        cache_sections: bool = False,
        cache_host_sections: bool = True,
    ) -> float:
        """GPU equivalent of the CPU accepted-bridge radius scan.

        Production keeps accepted sections in the bounded device cache without copying
        every bool canvas over PCIe.  Tests and compatibility callers can also request
        host copies; CPU recovery can always reconstruct a missing section from the SDFs.
        """

        with self._lock, self.runtime.activate():
            self._ensure_available()
            sdf0_host = np.asarray(getattr(plan, "sdf0"), dtype=np.float32)
            sdf1_host = np.asarray(getattr(plan, "sdf1"), dtype=np.float32)
            if not bool(np.any(sdf0_host >= 0.0)) or not bool(np.any(sdf1_host >= 0.0)):
                return 0.0
            min_radius = float(min(float(np.max(sdf0_host)), float(np.max(sdf1_host))))
            threshold = float(reject_at_or_below)
            if threshold > 0.0 and min_radius <= threshold:
                return float(min_radius)

            section_cache = getattr(plan, "cached_sections")
            if bool(cache_sections):
                section_cache[:] = [None] * (int(getattr(plan, "steps")) + 1)
            sdf0, sdf1 = self._device_sdfs(plan)
            self._metrics["estimated_plans"] += 1
            for idx in range(1, int(getattr(plan, "steps"))):
                alpha = float(idx) / float(getattr(plan, "steps"))
                section = self._blend_section(sdf0, sdf1, alpha)
                section = self._keep_center_component_and_fill(section)
                if section is None:
                    return 0.0
                if bool(cache_sections):
                    if bool(cache_host_sections):
                        host_section = np.ascontiguousarray(
                            self.runtime.to_host(section), dtype=bool,
                        )
                        self._metrics["device_to_host_bytes"] += int(host_section.nbytes)
                        section_cache[int(idx)] = host_section
                    # Insert as the scan proceeds so the LRU budget also bounds a
                    # long plan; the planner releases all entries if it rejects it.
                    self._cache_put(
                        ("section", id(section_cache), int(idx)),
                        section,
                        int(getattr(section, "nbytes")),
                        (section_cache,),
                    )
                min_radius = min(float(min_radius), float(self._section_radius(section)))
                self._metrics["estimated_sections"] += 1
                if threshold > 0.0 and min_radius <= threshold:
                    return float(min_radius)

            return float(min_radius)

    def _device_section(self, plan: object, step_idx: int) -> Optional[object]:
        section_cache = getattr(plan, "cached_sections")
        if int(step_idx) < len(section_cache):
            key = ("section", id(section_cache), int(step_idx))
            cached_device = self._cache_get(key, (section_cache,))
            if cached_device is not None:
                return cached_device
            cached_host = section_cache[int(step_idx)]
            if cached_host is not None:
                section = self._device_array(np.asarray(cached_host, dtype=bool))
                self._cache_put(
                    key,
                    section,
                    int(np.asarray(cached_host).nbytes),
                    (section_cache,),
                )
                return section
        sdf0, sdf1 = self._device_sdfs(plan)
        alpha = float(step_idx) / float(getattr(plan, "steps"))
        return self._keep_center_component_and_fill(
            self._blend_section(sdf0, sdf1, alpha)
        )

    def render_slice(
        self,
        destination: np.ndarray,
        jobs: Sequence[Tuple[object, int, int]],
    ) -> GpuInterpolationSliceResult:
        """Render jobs into one host slice, committing only after a successful D2H copy."""

        destination_array = np.asarray(destination)
        if destination_array.ndim != 2:
            raise ValueError("CUDA interpolation destination must be two-dimensional")
        if destination_array.dtype.kind not in "ui":
            raise TypeError("CUDA interpolation destination must have an integer dtype")

        placements: List[Tuple[object, int, int, _SlicePlacement]] = []
        for plan, step_idx, paint_value in jobs:
            placement = _slice_job_placement(plan, int(step_idx), destination_array.shape)
            if placement is not None:
                placements.append((plan, int(step_idx), int(paint_value), placement))
        if not placements:
            return GpuInterpolationSliceResult(0, 0, None)

        crop_y0 = min(item[3].dst_y0 for item in placements)
        crop_y1 = max(item[3].dst_y1 for item in placements)
        crop_x0 = min(item[3].dst_x0 for item in placements)
        crop_x1 = max(item[3].dst_x1 for item in placements)

        with self._lock, self.runtime.activate():
            self._ensure_available()
            initial_host = np.ascontiguousarray(
                destination_array[crop_y0:crop_y1, crop_x0:crop_x1]
            )
            device_crop = self._device_array(initial_host)
            added_voxels = 0
            rendered_sections = 0
            actual_bbox: Optional[List[int]] = None
            for plan, step_idx, paint_value, placement in placements:
                section = self._device_section(plan, int(step_idx))
                if section is None:
                    continue
                if bool(placement.mirrored):
                    section = section[:, ::-1]
                patch = section[
                    int(placement.src_y0):int(placement.src_y1),
                    int(placement.src_x0):int(placement.src_x1),
                ]
                if not bool(_scalar_int(self.xp.any(patch))):
                    continue
                local_y0 = int(placement.dst_y0 - crop_y0)
                local_y1 = int(placement.dst_y1 - crop_y0)
                local_x0 = int(placement.dst_x0 - crop_x0)
                local_x1 = int(placement.dst_x1 - crop_x0)
                current = device_crop[local_y0:local_y1, local_x0:local_x1]
                value = self.xp.asarray(int(paint_value), dtype=current.dtype)
                missing = patch & ((self.xp.bitwise_and(current, value)) == 0)
                added_voxels += _scalar_int(self.xp.count_nonzero(missing))
                painted = patch.astype(current.dtype, copy=False) * value
                self.xp.bitwise_or(current, painted, out=current)
                rendered_sections += 1
                if actual_bbox is None:
                    actual_bbox = [
                        int(placement.dst_y0), int(placement.dst_x0),
                        int(placement.dst_y1), int(placement.dst_x1),
                    ]
                else:
                    actual_bbox[0] = min(actual_bbox[0], int(placement.dst_y0))
                    actual_bbox[1] = min(actual_bbox[1], int(placement.dst_x0))
                    actual_bbox[2] = max(actual_bbox[2], int(placement.dst_y1))
                    actual_bbox[3] = max(actual_bbox[3], int(placement.dst_x1))

            # This is the transaction boundary: no host destination changes until every
            # device operation and the complete blocking D2H copy have succeeded.
            rendered_host = np.ascontiguousarray(self.runtime.to_host(device_crop))
            self._metrics["device_to_host_bytes"] += int(rendered_host.nbytes)
            np.copyto(
                destination_array[crop_y0:crop_y1, crop_x0:crop_x1],
                rendered_host,
                casting="no",
            )
            self._metrics["rendered_slices"] += 1
            self._metrics["rendered_sections"] += int(rendered_sections)
            result_bbox = (
                (
                    int(actual_bbox[0]), int(actual_bbox[1]),
                    int(actual_bbox[2]), int(actual_bbox[3]),
                )
                if actual_bbox is not None else None
            )
            return GpuInterpolationSliceResult(
                added_voxels=int(added_voxels),
                rendered_sections=int(rendered_sections),
                bbox=result_bbox,
            )

    def release_plans(self, plans: Iterable[object]) -> None:
        """Release device cache entries owned by a completed host plan batch."""

        sdf_ids = set()
        section_ids = set()
        for plan in plans:
            sdf_ids.add((id(getattr(plan, "sdf0")), id(getattr(plan, "sdf1"))))
            section_ids.add(id(getattr(plan, "cached_sections")))
        with self._lock:
            for key in list(self._cache):
                remove = (
                    key[0] == "sdf" and (int(key[1]), int(key[2])) in sdf_ids
                ) or (
                    key[0] == "section" and int(key[1]) in section_ids
                )
                if not remove:
                    continue
                entry = self._cache.pop(key)
                self._cache_live_bytes = max(
                    0, self._cache_live_bytes - int(entry.nbytes),
                )

    def telemetry(self) -> Dict[str, object]:
        with self._lock:
            result: Dict[str, object] = dict(self._metrics)
            result.update({
                "device_index": int(self.device_index),
                "cache_budget_bytes": int(self.cache_budget_bytes),
                "cache_live_bytes": int(self._cache_live_bytes),
                "failed_reason": self._failed_reason,
            })
            return result

    def close(self) -> None:
        with self._lock:
            self._clear_cache()
            self._structure8 = None
            try:
                with self.runtime.activate():
                    self.runtime.free_cached_memory()
            except Exception:
                pass


def create_cuda_interpolation_renderer(
    *,
    process_worker: bool,
) -> Tuple[Optional[CudaInterpolationRenderer], str]:
    """Admit a CUDA renderer without importing accelerator runtimes eagerly."""

    if not gpu_interpolation_enabled():
        return None, "disabled by YOLO_TTA_GPU_INTERPOLATION=0"
    if not bool(process_worker) and not gpu_interpolation_main_process_enabled():
        return None, "main-process CUDA interpolation is disabled"
    if (
        not gpu_interpolation_create_context_enabled()
        and "torch" not in sys.modules
    ):
        return None, "no warm CUDA runtime in this interpolation worker"
    try:
        import torch  # type: ignore
    except Exception as exc:
        return None, f"PyTorch unavailable ({type(exc).__name__}: {exc})"
    renderer: Optional[CudaInterpolationRenderer] = None
    try:
        if not bool(torch.cuda.is_available()):
            return None, "CUDA is unavailable"
        initialized = bool(torch.cuda.is_initialized())
        if not initialized and not gpu_interpolation_create_context_enabled():
            return None, "no warm CUDA context in this interpolation worker"
        device_index = int(torch.cuda.current_device())
        # Auxiliary interpolation leases are admitted only after inference has drained.
        # Return PyTorch's now-unused cached blocks before CuPy evaluates free VRAM;
        # live model tensors remain allocated and are unaffected by empty_cache().
        try:
            torch.cuda.synchronize(int(device_index))
        except TypeError:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        renderer = CudaInterpolationRenderer(device_index=int(device_index))
        free_bytes, _total_bytes = renderer.runtime.mem_info()
        if int(free_bytes) <= int(renderer.reserve_bytes):
            renderer.close()
            return None, (
                f"only {int(free_bytes) / MIB:.0f} MiB VRAM free; "
                f"{int(renderer.reserve_bytes) / MIB:.0f} MiB is reserved"
            )
        renderer.preflight()
        return renderer, f"cuda:{int(device_index)} CuPy/CuPyX"
    except Exception as exc:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        return None, f"CUDA preflight failed ({type(exc).__name__}: {exc})"


__all__ = (
    "CudaInterpolationRenderer",
    "GpuInterpolationSliceResult",
    "GpuInterpolationUnavailable",
    "create_cuda_interpolation_renderer",
    "gpu_interpolation_create_context_enabled",
    "gpu_interpolation_enabled",
    "gpu_interpolation_main_process_enabled",
    "gpu_interpolation_required",
)
