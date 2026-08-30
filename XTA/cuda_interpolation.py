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

import os
import sys
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .workspace import _env_flag, _env_int


MIB = 1024 ** 2


def gpu_interpolation_enabled() -> bool:
    """Whether CUDA bridge interpolation should be attempted."""

    return _env_flag("YOLO_TTA_GPU_INTERPOLATION", True)


def gpu_interpolation_radius_enabled() -> bool:
    """Whether the fine-grained min-radius scan should run on CUDA.

    The v17.1.1 implementation serialized every planner thread through one CUDA
    renderer and was substantially slower than the parallel CPU evaluator on a
    production four-GPU workload. The renderer now gives opted-in planners bounded
    independent streams, but the per-plan path remains an explicit experiment until
    that change has production profiling. Painting stays enabled by the master switch.
    """

    return _env_flag("YOLO_TTA_GPU_INTERPOLATION_RADIUS", False)


def gpu_interpolation_render_autotune_enabled() -> bool:
    """Probe one representative batch and retain GPU painting only when faster."""

    return _env_flag("YOLO_TTA_GPU_INTERPOLATION_RENDER_AUTOTUNE", True)


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


def gpu_interpolation_stream_count() -> int:
    """Maximum concurrent CUDA streams owned by one interpolation lease."""

    return max(1, _env_int("YOLO_TTA_GPU_INTERPOLATION_STREAMS", 4))


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
    # Entries may become visible while their creating stream is still in flight.
    # A consumer on another stream inserts this dependency before using ``value``.
    ready_event: Optional[object] = None


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

    def create_stream(self) -> object:
        with self.device:
            return self.xp.cuda.Stream(non_blocking=True)

    @staticmethod
    def activate_stream(stream: object) -> object:
        return stream

    def to_device(self, value: np.ndarray) -> object:
        return self.xp.asarray(value)

    def to_host(self, value: object) -> np.ndarray:
        # cp.asnumpy has been a blocking copy since the oldest supported CuPy releases.
        return np.asarray(self.xp.asnumpy(value))

    def to_host_async(self, value: object, stream: object) -> np.ndarray:
        # CuPy 13 exposes ``blocking=False``, but an implicitly allocated NumPy result is
        # not documented as page-locked.  Supply pinned storage explicitly so the CUDA
        # runtime can enqueue a genuine asynchronous D2H transfer.  ``numpy.ndarray``
        # retains the buffer object as its base; the renderer, in turn, retains this array
        # and does not inspect it until ``wait_completion`` succeeds.
        shape = tuple(int(size) for size in getattr(value, "shape"))
        dtype = np.dtype(getattr(value, "dtype"))
        pinned = self.xp.cuda.alloc_pinned_memory(
            int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        )
        host = np.ndarray(shape, dtype=dtype, buffer=pinned, order="C")
        return np.asarray(self.xp.asnumpy(
            value, out=host, stream=stream, blocking=False,
        ))

    @staticmethod
    def record_completion(stream: object) -> object:
        return stream.record()

    @staticmethod
    def make_stream_wait(stream: object, event: object) -> None:
        stream.wait_event(event)

    @staticmethod
    def wait_completion(event: object) -> None:
        event.synchronize()

    @staticmethod
    def synchronize_stream(stream: object) -> None:
        stream.synchronize()

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
        stream_count: Optional[int] = None,
    ) -> None:
        self.runtime = (
            runtime if runtime is not None
            else _CupyInterpolationRuntime(int(device_index))
        )
        self.xp = getattr(self.runtime, "xp")
        self.ndi = getattr(self.runtime, "ndi")
        self.device_index = int(getattr(self.runtime, "device_index", device_index))
        visible_tokens = [
            token.strip()
            for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if token.strip()
        ]
        self.visible_device_token = (
            visible_tokens[int(self.device_index)]
            if 0 <= int(self.device_index) < len(visible_tokens)
            else None
        )
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
        self._cache_generation = 0
        self._lock = threading.RLock()
        self.max_streams = max(
            1,
            int(
                gpu_interpolation_stream_count()
                if stream_count is None else stream_count
            ),
        )
        # Streams are created lazily inside the selected device context. The pool is
        # renderer-local because one renderer is scoped to one scheduler GPU lease.
        self._stream_condition = threading.Condition(threading.Lock())
        self._idle_streams: List[object] = []
        self._created_streams = 0
        self._leased_streams = 0
        self._closing = False
        self._stream_local = threading.local()
        self._destination_lock_guard = threading.Lock()
        self._destination_locks: Dict[
            Tuple[int, Tuple[int, ...], Tuple[int, ...]], Tuple[threading.Lock, int]
        ] = {}
        self._failed_reason: Optional[str] = None
        self._radius_failed_reason: Optional[str] = None
        self._metrics: Dict[str, object] = {
            "estimated_plans": 0,
            "estimated_sections": 0,
            "rendered_slices": 0,
            "rendered_sections": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "sdf_host_to_device_bytes": 0,
            "section_host_to_device_bytes": 0,
            "destination_host_to_device_bytes": 0,
            "destination_device_to_host_bytes": 0,
            "metrics_device_to_host_bytes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "cache_evicted_bytes": 0,
            "cache_peak_bytes": 0,
            "render_crop_pixels": 0,
            "render_patch_pixels": 0,
            "radius_lock_wait_seconds": 0.0,
            "radius_execution_seconds": 0.0,
            "render_lock_wait_seconds": 0.0,
            "render_execution_seconds": 0.0,
            "stream_leases": 0,
            "stream_peak": 0,
        }
        self._structure8 = None
        self._structure8_ready: Optional[object] = None

    @property
    def available(self) -> bool:
        return self._failed_reason is None

    @property
    def failed_reason(self) -> Optional[str]:
        return self._failed_reason

    @property
    def radius_available(self) -> bool:
        return self._failed_reason is None and self._radius_failed_reason is None

    @property
    def radius_failed_reason(self) -> Optional[str]:
        return self._radius_failed_reason

    @contextmanager
    def _lease_stream(self) -> Iterator[object]:
        """Borrow one renderer-local stream until its final event has settled."""

        create_new = False
        stream: object
        with self._stream_condition:
            while (
                not self._idle_streams
                and self._created_streams >= int(self.max_streams)
                and not self._closing
            ):
                self._stream_condition.wait()
            if self._closing:
                raise GpuInterpolationUnavailable("CUDA interpolation renderer is closed")
            if self._idle_streams:
                stream = self._idle_streams.pop()
            else:
                self._created_streams += 1
                create_new = True
                stream = None
            self._leased_streams += 1
            leased_now = int(self._leased_streams)

        if create_new:
            try:
                create_stream = getattr(self.runtime, "create_stream", None)
                if callable(create_stream):
                    with self.runtime.activate():
                        stream = create_stream()
            except BaseException:
                with self._stream_condition:
                    self._created_streams = max(0, self._created_streams - 1)
                    self._leased_streams = max(0, self._leased_streams - 1)
                    self._stream_condition.notify_all()
                raise

        with self._lock:
            self._metrics["stream_leases"] = int(self._metrics["stream_leases"]) + 1
            self._metrics["stream_peak"] = max(
                int(self._metrics["stream_peak"]), int(leased_now),
            )

        active_error = False
        previous_keepalive = getattr(self._stream_local, "keepalive", None)
        previous_cache_generation = getattr(
            self._stream_local, "cache_generation", None,
        )
        self._stream_local.keepalive = []
        with self._lock:
            self._stream_local.cache_generation = int(self._cache_generation)
        try:
            activate_stream = getattr(self.runtime, "activate_stream", None)
            stream_context = (
                activate_stream(stream)
                if callable(activate_stream) else nullcontext()
            )
            with self.runtime.activate(), stream_context:
                yield stream
        except BaseException:
            active_error = True
            raise
        finally:
            # A failed enqueue may still have work using temporary arrays. Quiesce the
            # borrowed stream before those owners die or the stream is reused.
            synchronize_stream = getattr(self.runtime, "synchronize_stream", None)
            synchronize_error: Optional[BaseException] = None
            if callable(synchronize_stream) and stream is not None:
                try:
                    with self.runtime.activate():
                        synchronize_stream(stream)
                except BaseException as exc:
                    synchronize_error = exc
            if previous_keepalive is None:
                try:
                    del self._stream_local.keepalive
                except AttributeError:
                    pass
            else:
                self._stream_local.keepalive = previous_keepalive
            if previous_cache_generation is None:
                try:
                    del self._stream_local.cache_generation
                except AttributeError:
                    pass
            else:
                self._stream_local.cache_generation = previous_cache_generation
            with self._stream_condition:
                self._leased_streams = max(0, self._leased_streams - 1)
                if self._closing or active_error or synchronize_error is not None:
                    self._created_streams = max(0, self._created_streams - 1)
                else:
                    self._idle_streams.append(stream)
                self._stream_condition.notify_all()
            if synchronize_error is not None and not active_error:
                raise synchronize_error

    def _retain_for_stream_lease(self, *values: object) -> None:
        """Keep device arrays/events alive until this thread's stream is quiescent."""

        keepalive = getattr(self._stream_local, "keepalive", None)
        if keepalive is not None:
            keepalive.extend(value for value in values if value is not None)

    @contextmanager
    def _lease_destination(self, destination: np.ndarray) -> Iterator[None]:
        """Serialize only calls targeting the same host view transaction."""

        array = np.asarray(destination)
        key = (
            int(array.__array_interface__["data"][0]),
            tuple(int(v) for v in array.shape),
            tuple(int(v) for v in array.strides),
        )
        with self._destination_lock_guard:
            entry = self._destination_locks.get(key)
            if entry is None:
                destination_lock = threading.Lock()
                users = 0
            else:
                destination_lock, users = entry
            self._destination_locks[key] = (destination_lock, int(users) + 1)
        destination_lock.acquire()
        try:
            yield
        finally:
            destination_lock.release()
            with self._destination_lock_guard:
                current = self._destination_locks.get(key)
                if current is not None and current[0] is destination_lock:
                    remaining = int(current[1]) - 1
                    if remaining <= 0:
                        self._destination_locks.pop(key, None)
                    else:
                        self._destination_locks[key] = (
                            destination_lock, int(remaining),
                        )

    def _record_completion(self, stream: object) -> Optional[object]:
        record_completion = getattr(self.runtime, "record_completion", None)
        if not callable(record_completion):
            return None
        return record_completion(stream)

    def _make_stream_wait(self, stream: object, event: Optional[object]) -> None:
        if event is None:
            return
        make_stream_wait = getattr(self.runtime, "make_stream_wait", None)
        if callable(make_stream_wait):
            make_stream_wait(stream, event)

    def _wait_completion(self, event: Optional[object]) -> None:
        if event is None:
            return
        wait_completion = getattr(self.runtime, "wait_completion", None)
        if callable(wait_completion):
            wait_completion(event)

    def _to_host_async(self, value: object, stream: object) -> np.ndarray:
        to_host_async = getattr(self.runtime, "to_host_async", None)
        if callable(to_host_async):
            return np.asarray(to_host_async(value, stream))
        # Compatibility for injected runtimes predating the stream contract. Production
        # always takes the asynchronous CuPy branch above.
        return np.asarray(self.runtime.to_host(value))

    def disable(self, exc: BaseException) -> bool:
        """Disable future CUDA work; return True only for the first failure."""

        reason = f"{type(exc).__name__}: {exc}"
        with self._lock:
            first = self._failed_reason is None
            if first:
                self._failed_reason = reason
                self._cache_generation += 1
                self._clear_cache()
            return bool(first)

    def disable_radius(self, exc: BaseException) -> bool:
        """Disable only CUDA radius evaluation while preserving GPU painting."""

        reason = f"{type(exc).__name__}: {exc}"
        with self._lock:
            first = self._radius_failed_reason is None
            if first:
                self._radius_failed_reason = reason
                # A failed scan may have left valid but incomplete plan entries.  The
                # CPU retry becomes authoritative, so force painting to reload those
                # exact host sections instead of reusing a partial device cache.
                self._cache_generation += 1
                self._clear_cache()
            return bool(first)

    def _ensure_available(self) -> None:
        if self._failed_reason is not None:
            raise GpuInterpolationUnavailable(self._failed_reason)

    def _ensure_radius_available(self) -> None:
        self._ensure_available()
        if self._radius_failed_reason is not None:
            raise GpuInterpolationUnavailable(self._radius_failed_reason)

    def _cache_get(
        self,
        key: Tuple[object, ...],
        owners: Sequence[object] = (),
        stream: object = None,
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
        self._retain_for_stream_lease(entry.value, entry.ready_event)
        if stream is not None:
            self._make_stream_wait(stream, entry.ready_event)
        return entry.value

    def _cache_put(
        self,
        key: Tuple[object, ...],
        value: object,
        nbytes: int,
        owners: Sequence[object] = (),
        stream: object = None,
    ) -> None:
        size = max(0, int(nbytes))
        self._retain_for_stream_lease(value)
        lease_generation = getattr(self._stream_local, "cache_generation", None)
        if (
            lease_generation is not None
            and int(lease_generation) != int(self._cache_generation)
        ):
            return
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
            self._metrics["cache_evictions"] = int(
                self._metrics["cache_evictions"]
            ) + 1
            self._metrics["cache_evicted_bytes"] = int(
                self._metrics["cache_evicted_bytes"]
            ) + int(old_entry.nbytes)
        ready_event = (
            self._record_completion(stream) if stream is not None else None
        )
        self._retain_for_stream_lease(ready_event)
        self._cache[key] = _DeviceCacheEntry(
            value=value,
            nbytes=int(size),
            owners=tuple(owners),
            ready_event=ready_event,
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
        with self._lock:
            self._metrics["host_to_device_bytes"] += int(contiguous.nbytes)
        return result

    def _device_sdfs(
        self, plan: object, stream: object = None,
    ) -> Tuple[object, object]:
        sdf0_owner = getattr(plan, "sdf0")
        sdf1_owner = getattr(plan, "sdf1")
        sdf0_host = np.asarray(sdf0_owner, dtype=np.float32)
        sdf1_host = np.asarray(sdf1_owner, dtype=np.float32)
        key = ("sdf", id(sdf0_owner), id(sdf1_owner))
        with self._lock:
            cached = self._cache_get(
                key, (sdf0_owner, sdf1_owner), stream=stream,
            )
            if cached is not None:
                return cached  # type: ignore[return-value]

        # Pageable H2D staging can block the submitting host thread. Keep it outside the
        # renderer metadata lock so independent stream leases can upload/compute in parallel.
        # A same-key race may duplicate one upload; the second cache check below selects the
        # already-published entry and the lease keepalive safely retires the redundant pair.
        sdf0 = self._device_array(sdf0_host)
        sdf1 = self._device_array(sdf1_host)
        pair = (sdf0, sdf1)
        self._retain_for_stream_lease(pair)
        with self._lock:
            self._metrics["sdf_host_to_device_bytes"] = int(
                self._metrics["sdf_host_to_device_bytes"]
            ) + int(sdf0_host.nbytes + sdf1_host.nbytes)
            if key in self._cache:
                raced = self._cache_get(
                    key, (sdf0_owner, sdf1_owner), stream=stream,
                )
                if raced is not None:
                    return raced  # type: ignore[return-value]
            self._cache_put(
                key,
                pair,
                int(sdf0_host.nbytes + sdf1_host.nbytes),
                (sdf0_owner, sdf1_owner),
                stream=stream,
            )
            return pair

    def _blend_section(self, sdf0: object, sdf1: object, alpha: float) -> object:
        # Separate multiply/add ufunc launches avoid device FMA changing a zero-boundary
        # decision relative to NumPy's float32 expression.
        left = self.xp.multiply(sdf0, np.float32(1.0 - float(alpha)))
        right = self.xp.multiply(sdf1, np.float32(float(alpha)))
        return self.xp.add(left, right) >= np.float32(0.0)

    def _keep_center_component_and_fill(
        self, section: object, stream: object = None,
    ) -> Optional[object]:
        with self._lock:
            if self._structure8 is None:
                self._structure8 = self.xp.ones((3, 3), dtype=self.xp.bool_)
                self._structure8_ready = (
                    self._record_completion(stream) if stream is not None else None
                )
            structure8 = self._structure8
            structure8_ready = self._structure8_ready
        if stream is not None:
            self._make_stream_wait(stream, structure8_ready)
        labels, num = self.ndi.label(section, structure=structure8)
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

    def preflight(self, *, check_radius: bool = True) -> None:
        """Exercise requested CuPyX primitives before any host canvas can change."""

        with self._lock, self.runtime.activate():
            self._ensure_available()
            probe = np.zeros((5, 5), dtype=bool)
            probe[1:4, 1:4] = True
            probe[2, 2] = False
            device_probe = self._device_array(probe)
            kept = self._keep_center_component_and_fill(device_probe)
            if kept is None:
                raise RuntimeError("CUDA interpolation preflight produced an empty component")
            radius = self._section_radius(kept) if bool(check_radius) else 1.0
            host = self.runtime.to_host(kept)
            self._metrics["device_to_host_bytes"] += int(host.nbytes)
            if host.shape != probe.shape or not bool(host[2, 2]) or radius <= 0.0:
                raise RuntimeError("CUDA interpolation preflight failed its morphology check")
            self._clear_cache()
            for key in self._metrics:
                self._metrics[key] = (
                    0.0 if str(key).endswith("_seconds") else 0
                )

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

        queued_at = time.perf_counter()
        acquired_at: Optional[float] = None
        try:
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

            with self._lease_stream() as stream:
                with self._lock:
                    acquired_at = time.perf_counter()
                    self._metrics["radius_lock_wait_seconds"] = float(
                        self._metrics["radius_lock_wait_seconds"]
                    ) + max(0.0, acquired_at - queued_at)
                    self._ensure_radius_available()
                # Shared renderer metadata/cache mutations lock themselves. H2D staging,
                # CuPyX execution, scalar completion reads, and D2H waits are stream-local.
                sdf0, sdf1 = self._device_sdfs(plan, stream=stream)
                with self._lock:
                    self._metrics["estimated_plans"] = int(
                        self._metrics["estimated_plans"]
                    ) + 1
                for idx in range(1, int(getattr(plan, "steps"))):
                    alpha = float(idx) / float(getattr(plan, "steps"))
                    section = self._blend_section(sdf0, sdf1, alpha)
                    section = self._keep_center_component_and_fill(
                        section, stream=stream,
                    )
                    if section is None:
                        return 0.0
                    if bool(cache_sections):
                        if bool(cache_host_sections):
                            pending_host_section = self._to_host_async(section, stream)
                            host_completion = self._record_completion(stream)
                            self._wait_completion(host_completion)
                            host_section = np.ascontiguousarray(
                                pending_host_section, dtype=bool,
                            )
                            with self._lock:
                                self._metrics["device_to_host_bytes"] = int(
                                    self._metrics["device_to_host_bytes"]
                                ) + int(host_section.nbytes)
                            section_cache[int(idx)] = host_section
                        # Insert as the scan proceeds so the LRU budget also bounds a
                        # long plan; the planner releases all entries if it rejects it.
                        with self._lock:
                            self._cache_put(
                                ("section", id(section_cache), int(idx)),
                                section,
                                int(getattr(section, "nbytes")),
                                (section_cache,),
                                stream=stream,
                            )
                    min_radius = min(
                        float(min_radius), float(self._section_radius(section)),
                    )
                    with self._lock:
                        self._metrics["estimated_sections"] = int(
                            self._metrics["estimated_sections"]
                        ) + 1
                    if threshold > 0.0 and min_radius <= threshold:
                        return float(min_radius)

                return float(min_radius)
        finally:
            if acquired_at is not None:
                with self._lock:
                    self._metrics["radius_execution_seconds"] = float(
                        self._metrics["radius_execution_seconds"]
                    ) + max(0.0, time.perf_counter() - acquired_at)

    def _device_section(
        self, plan: object, step_idx: int, stream: object = None,
    ) -> Optional[object]:
        section_cache = getattr(plan, "cached_sections")
        if int(step_idx) < len(section_cache):
            key = ("section", id(section_cache), int(step_idx))
            with self._lock:
                cached_device = self._cache_get(
                    key, (section_cache,), stream=stream,
                )
                cached_host = section_cache[int(step_idx)]
            if cached_device is not None:
                return cached_device
            if cached_host is not None:
                # CPU-radius mode retains the exact accepted section on the host.
                # A plan-step is painted once, so putting this one-shot upload into
                # the device LRU only evicts reusable SDF entries before later jobs.
                host_section = np.asarray(cached_host, dtype=bool)
                section = self._device_array(host_section)
                with self._lock:
                    self._metrics["section_host_to_device_bytes"] = int(
                        self._metrics["section_host_to_device_bytes"]
                    ) + int(host_section.nbytes)
                self._retain_for_stream_lease(section)
                return section
        sdf0, sdf1 = self._device_sdfs(plan, stream=stream)
        alpha = float(step_idx) / float(getattr(plan, "steps"))
        section = self._keep_center_component_and_fill(
            self._blend_section(sdf0, sdf1, alpha), stream=stream,
        )
        self._retain_for_stream_lease(section)
        return section

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

        queued_at = time.perf_counter()
        acquired_at: Optional[float] = None
        try:
            # Separate destination groups can overlap. Calls that target the same host
            # view retain the old lost-update protection without monopolizing the GPU.
            with self._lease_destination(destination_array), self._lease_stream() as stream:
                with self._lock:
                    acquired_at = time.perf_counter()
                    self._metrics["render_lock_wait_seconds"] = float(
                        self._metrics["render_lock_wait_seconds"]
                    ) + max(0.0, acquired_at - queued_at)
                    self._ensure_available()

                initial_host = np.ascontiguousarray(
                    destination_array[crop_y0:crop_y1, crop_x0:crop_x1]
                )
                device_crop = self._device_array(initial_host)
                with self._lock:
                    self._metrics["destination_host_to_device_bytes"] = int(
                        self._metrics["destination_host_to_device_bytes"]
                    ) + int(initial_host.nbytes)
                    self._metrics["render_crop_pixels"] = int(
                        self._metrics["render_crop_pixels"]
                    ) + int(initial_host.size)

                added_counts_device: List[object] = []
                nonempty_flags_device: List[object] = []
                stat_placements: List[_SlicePlacement] = []
                patch_pixels = 0
                # Keep every section owner alive until the stream completion event;
                # another concurrent stream may evict its global LRU entry meanwhile.
                section_keepalive: List[object] = []
                for plan, step_idx, paint_value, placement in placements:
                    section = self._device_section(
                        plan, int(step_idx), stream=stream,
                    )
                    if section is None:
                        continue
                    section_keepalive.append(section)
                    if bool(placement.mirrored):
                        section = section[:, ::-1]
                    patch = section[
                        int(placement.src_y0):int(placement.src_y1),
                        int(placement.src_x0):int(placement.src_x1),
                    ]
                    local_y0 = int(placement.dst_y0 - crop_y0)
                    local_y1 = int(placement.dst_y1 - crop_y0)
                    local_x0 = int(placement.dst_x0 - crop_x0)
                    local_x1 = int(placement.dst_x1 - crop_x0)
                    current = device_crop[local_y0:local_y1, local_x0:local_x1]
                    value = self.xp.asarray(int(paint_value), dtype=current.dtype)
                    missing = patch & ((self.xp.bitwise_and(current, value)) == 0)
                    # Queue reductions and fetch all tiny scalars together after every
                    # section has been painted. v17.1.1 called .item() twice per job,
                    # forcing thousands of device synchronizations in a typical slice.
                    nonempty_flags_device.append(self.xp.any(patch))
                    added_counts_device.append(self.xp.count_nonzero(missing))
                    patch_pixels += int(patch.size)
                    stat_placements.append(placement)
                    painted = patch.astype(current.dtype, copy=False) * value
                    self.xp.bitwise_or(current, painted, out=current)

                with self._lock:
                    self._metrics["render_patch_pixels"] = int(
                        self._metrics["render_patch_pixels"]
                    ) + int(patch_pixels)

                if added_counts_device:
                    counts_device = self.xp.stack(tuple(added_counts_device))
                    flags_device = self.xp.stack(tuple(nonempty_flags_device)).astype(
                        self.xp.int64, copy=False,
                    )
                    stats_device = self.xp.stack((counts_device, flags_device), axis=1)
                    pending_stats_host: Optional[np.ndarray] = self._to_host_async(
                        stats_device, stream,
                    )
                else:
                    pending_stats_host = None

                pending_rendered_host = self._to_host_async(device_crop, stream)
                completion = self._record_completion(stream)

                # The renderer metadata/cache lock is intentionally not held here. The
                # event covers every kernel and both asynchronous D2H transfers queued on
                # this lease stream, so other destination groups can enqueue and run.
                self._wait_completion(completion)
                stats_host = (
                    np.ascontiguousarray(pending_stats_host, dtype=np.int64)
                    if pending_stats_host is not None
                    else np.zeros((0, 2), dtype=np.int64)
                )
                rendered_host = np.ascontiguousarray(pending_rendered_host)

                added_voxels = int(np.sum(stats_host[:, 0], dtype=np.int64))
                nonempty_flags = np.asarray(stats_host[:, 1] != 0, dtype=bool)
                rendered_sections = int(np.count_nonzero(nonempty_flags))
                actual_bbox: Optional[List[int]] = None
                for placement, nonempty in zip(stat_placements, nonempty_flags):
                    if not bool(nonempty):
                        continue
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

                # This is the transaction boundary: the destination remains untouched
                # until the completion event proves that the entire D2H copy succeeded.
                np.copyto(
                    destination_array[crop_y0:crop_y1, crop_x0:crop_x1],
                    rendered_host,
                    casting="no",
                )
                with self._lock:
                    self._metrics["device_to_host_bytes"] = int(
                        self._metrics["device_to_host_bytes"]
                    ) + int(rendered_host.nbytes + stats_host.nbytes)
                    self._metrics["destination_device_to_host_bytes"] = int(
                        self._metrics["destination_device_to_host_bytes"]
                    ) + int(rendered_host.nbytes)
                    self._metrics["metrics_device_to_host_bytes"] = int(
                        self._metrics["metrics_device_to_host_bytes"]
                    ) + int(stats_host.nbytes)
                    self._metrics["rendered_slices"] = int(
                        self._metrics["rendered_slices"]
                    ) + 1
                    self._metrics["rendered_sections"] = int(
                        self._metrics["rendered_sections"]
                    ) + int(rendered_sections)
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
        finally:
            if acquired_at is not None:
                with self._lock:
                    self._metrics["render_execution_seconds"] = float(
                        self._metrics["render_execution_seconds"]
                    ) + max(0.0, time.perf_counter() - acquired_at)

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
                "visible_device_token": self.visible_device_token,
                "cache_budget_bytes": int(self.cache_budget_bytes),
                "cache_live_bytes": int(self._cache_live_bytes),
                "max_streams": int(self.max_streams),
                "failed_reason": self._failed_reason,
                "radius_failed_reason": self._radius_failed_reason,
            })
            return result

    def close(self) -> None:
        with self._stream_condition:
            if self._closing:
                while self._leased_streams > 0:
                    self._stream_condition.wait()
                return
            self._closing = True
            self._stream_condition.notify_all()
            while self._leased_streams > 0:
                self._stream_condition.wait()
            self._idle_streams.clear()
            self._created_streams = 0
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
        radius_requested = bool(gpu_interpolation_radius_enabled())
        try:
            renderer.preflight(check_radius=bool(radius_requested))
        except Exception as exc:
            if not bool(radius_requested):
                raise
            renderer.disable_radius(exc)
            # EDT/radius support is an optional capability. Re-run the render
            # morphology preflight so a radius-only failure cannot prevent the
            # default GPU painting path from being admitted.
            renderer.preflight(check_radius=False)
        status = f"cuda:{int(device_index)} CuPy/CuPyX"
        visible_token = getattr(renderer, "visible_device_token", None)
        if visible_token is not None:
            status += f" [CUDA_VISIBLE_DEVICES={visible_token}]"
        if renderer.radius_failed_reason:
            status += f"; radius unavailable ({renderer.radius_failed_reason})"
        return renderer, status
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
    "gpu_interpolation_radius_enabled",
    "gpu_interpolation_render_autotune_enabled",
    "gpu_interpolation_required",
    "gpu_interpolation_stream_count",
)
