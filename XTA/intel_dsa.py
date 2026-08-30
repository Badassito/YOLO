"""Lazy, hardware-only Intel DSA workspace-copy control plane.

Linux x86-64 builds may include the project-owned ``XTA._dsa_copy``
extension from ``native/dsa_copy.c`` without adding a userspace DSA library.
Other builds remain usable without it. The extension submits MEMMOVE descriptors through the kernel
``idxd`` user-work-queue character device; this module deliberately rejects
bindings which advertise software fallback or cannot prove a synchronous drain.

The native operation is synchronous.  If it fails after submitting any
descriptors, :class:`DsaManager` calls ``drain()`` and reports whether the native
binding explicitly proved that no device write remains possible.  Callers may
only perform a CPU recovery copy when ``drained`` is true.
"""

from __future__ import annotations

import atexit
import importlib
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence

import numpy as np


class IntelDsaError(RuntimeError):
    """Base error for the optional DSA copy backend."""


class IntelDsaUnavailable(IntelDsaError):
    """No validated, hardware-only DSA work queue is available."""

    def __init__(self, message: str, *, module_missing: bool = False) -> None:
        super().__init__(message)
        self.module_missing = bool(module_missing)


class IntelDsaIneligible(IntelDsaError):
    """A copy is structurally unsafe or outside the initial DSA rollout."""

    def __init__(self, reasons: Sequence[str]) -> None:
        normalized = tuple(str(reason) for reason in reasons if str(reason))
        self.reasons = normalized or ('unknown',)
        super().__init__('DSA-ineligible workspace copy: ' + ', '.join(self.reasons))


class IntelDsaCopyError(IntelDsaError):
    """A native request failed after its submission state was considered.

    ``drained`` is deliberately not inferred from a native exception returning.
    It becomes true only when the binding's explicit ``drain()`` acknowledgement
    says that every submitted descriptor is terminal.
    """

    def __init__(
        self,
        message: str,
        *,
        drained: bool,
        stats: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.drained = bool(drained)
        self.stats = dict(stats or {})


class NativeDsaModule(Protocol):
    """Interface required from ``XTA._dsa_copy``."""

    def capabilities(self, *, work_queue: Optional[str] = None) -> Mapping[str, object]: ...

    def copy(
        self,
        src: object,
        dst: object,
        *,
        work_queue: str,
        max_transfer_size: int,
        max_inflight: int,
        require_hardware: bool = True,
    ) -> Mapping[str, object]: ...

    def drain(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CopyEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    source_backing: str
    destination_backing: str
    nbytes: int


_NATIVE_MODULE_NAME = 'XTA._dsa_copy'
_STATE_LOCK = threading.RLock()
_LOADED_MODULE: Optional[NativeDsaModule] = None
_TEST_MODULE: Optional[NativeDsaModule] = None
_IMPORT_ERROR: Optional[tuple[str, bool]] = None
_MANAGER: Optional['DsaManager'] = None
_ATEXIT_REGISTERED = False


def _validate_native_module(module: object) -> NativeDsaModule:
    for symbol in ('capabilities', 'copy', 'drain', 'close'):
        if not callable(getattr(module, symbol, None)):
            raise IntelDsaUnavailable(
                f'DSA companion module does not implement required {symbol}()'
            )
    return module  # type: ignore[return-value]


def _load_native_module() -> NativeDsaModule:
    global _LOADED_MODULE, _IMPORT_ERROR
    with _STATE_LOCK:
        if _TEST_MODULE is not None:
            return _validate_native_module(_TEST_MODULE)
        if _LOADED_MODULE is not None:
            return _LOADED_MODULE
        if _IMPORT_ERROR is not None:
            message, missing = _IMPORT_ERROR
            raise IntelDsaUnavailable(message, module_missing=missing)
    try:
        imported = importlib.import_module(_NATIVE_MODULE_NAME)
    except ModuleNotFoundError as exc:
        missing_self = str(getattr(exc, 'name', '')) == _NATIVE_MODULE_NAME
        message = (
            f'DSA companion module is not installed ({_NATIVE_MODULE_NAME})'
            if missing_self else
            f'DSA companion module dependency is missing: {exc}'
        )
        with _STATE_LOCK:
            _IMPORT_ERROR = (message, missing_self)
        raise IntelDsaUnavailable(message, module_missing=missing_self) from exc
    except Exception as exc:
        message = f'DSA companion module failed to import: {type(exc).__name__}: {exc}'
        with _STATE_LOCK:
            _IMPORT_ERROR = (message, False)
        raise IntelDsaUnavailable(message) from exc
    validated = _validate_native_module(imported)
    with _STATE_LOCK:
        _LOADED_MODULE = validated
    return validated


def platform_supported() -> bool:
    """Return whether Linux DSA is usable (test fakes are an intentional exception)."""
    with _STATE_LOCK:
        testing = _TEST_MODULE is not None
    return bool(sys.platform.startswith('linux') or testing)


def requested_backend() -> str:
    value = os.environ.get('YOLO_TTA_WORKSPACE_COPY_BACKEND', 'cpu').strip().lower()
    if value not in {'cpu', 'auto', 'dsa'}:
        raise ValueError(
            'YOLO_TTA_WORKSPACE_COPY_BACKEND must be one of cpu, auto, or dsa; '
            f'got {value!r}'
        )
    return value


def minimum_copy_bytes() -> int:
    raw = os.environ.get('YOLO_TTA_DSA_MIN_MIB', '64').strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f'YOLO_TTA_DSA_MIN_MIB must be a non-negative number; got {raw!r}') from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f'YOLO_TTA_DSA_MIN_MIB must be a non-negative finite number; got {raw!r}')
    return int(value * 1024 * 1024)


def requested_work_queue() -> Optional[str]:
    value = os.environ.get('YOLO_TTA_DSA_WQ', '').strip()
    return value or None


def requested_max_inflight() -> int:
    raw = os.environ.get('YOLO_TTA_DSA_MAX_INFLIGHT', '32').strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'YOLO_TTA_DSA_MAX_INFLIGHT must be a positive integer; got {raw!r}') from exc
    if value <= 0:
        raise ValueError(f'YOLO_TTA_DSA_MAX_INFLIGHT must be a positive integer; got {raw!r}')
    # Bound descriptor storage and the time spent draining one submitted batch.
    return min(int(value), 4096)


def _array_chain(arr: object) -> list[object]:
    result: list[object] = []
    current: object | None = arr
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result.append(current)
        current = getattr(current, 'base', None)
    return result


def _decode_mount_path(value: str) -> str:
    return re.sub(
        r'\\([0-7]{3})',
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _filesystem_type(path: Path) -> Optional[str]:
    """Resolve Linux mount type using the process mount namespace."""
    try:
        candidate = str(path.resolve(strict=False))
        best_length = -1
        best_type: Optional[str] = None
        for line in Path('/proc/self/mountinfo').read_text(encoding='utf-8').splitlines():
            left, separator, right = line.partition(' - ')
            if not separator:
                continue
            fields = left.split()
            right_fields = right.split()
            if len(fields) < 5 or not right_fields:
                continue
            mount_point = _decode_mount_path(fields[4])
            try:
                belongs = os.path.commonpath((candidate, mount_point)) == mount_point
            except (ValueError, OSError):
                belongs = False
            if belongs and len(mount_point) > best_length:
                best_length = len(mount_point)
                best_type = str(right_fields[0])
        return best_type
    except Exception:
        return None


def classify_array_backing(arr: object) -> str:
    """Classify only backing types admitted by the first DSA rollout."""
    chain = _array_chain(arr)
    if any(
        getattr(item, '_workspace_memfd_owner_key', None)
        or getattr(item, '_workspace_memfd_path', None)
        for item in chain
    ):
        return 'memfd'
    root_memmap = next((item for item in chain if isinstance(item, np.memmap)), None)
    if root_memmap is not None:
        filename = getattr(root_memmap, 'filename', None)
        if filename:
            filename_text = str(filename)
            if filename_text.startswith('/memfd:') or filename_text.startswith('memfd:'):
                return 'memfd'
            try:
                link_target = os.readlink(filename_text)
            except (OSError, ValueError):
                link_target = ''
            if link_target.startswith('/memfd:') or link_target.startswith('memfd:'):
                return 'memfd'
            fs_type = _filesystem_type(Path(filename_text))
            if fs_type in {'tmpfs', 'ramfs'}:
                return 'tmpfs'
        return 'regular_memmap'
    # NumPy-owned allocations, and ndarrays backed by anonymous mmap objects, are
    # RAM. Unknown foreign exporters are kept out of the initial rollout.
    if isinstance(arr, np.ndarray):
        terminal = chain[-1] if chain else arr
        if terminal is arr or isinstance(terminal, (np.ndarray, memoryview)):
            return 'anonymous'
        if terminal.__class__.__module__ == 'mmap':
            return 'anonymous'
    return 'unknown'


def _byte_range(arr: np.ndarray) -> tuple[int, int]:
    pointer = int(arr.__array_interface__['data'][0])
    return pointer, pointer + int(arr.nbytes)


def assess_copy_eligibility(
    src: object,
    dst: object,
    *,
    minimum_bytes: int,
) -> CopyEligibility:
    reasons: list[str] = []
    try:
        src_arr = np.asarray(src)
        dst_arr = np.asarray(dst)
    except Exception:
        return CopyEligibility(False, ('not_numpy_compatible',), 'unknown', 'unknown', 0)
    nbytes = int(src_arr.nbytes)
    source_backing = classify_array_backing(src)
    destination_backing = classify_array_backing(dst)
    if not platform_supported():
        reasons.append('unsupported_platform')
    if src_arr.shape != dst_arr.shape:
        reasons.append('shape_mismatch')
    if src_arr.dtype != dst_arr.dtype:
        reasons.append('dtype_mismatch')
    if int(dst_arr.nbytes) != nbytes:
        reasons.append('byte_count_mismatch')
    if not bool(src_arr.flags['C_CONTIGUOUS']):
        reasons.append('source_noncontiguous')
    if not bool(dst_arr.flags['C_CONTIGUOUS']):
        reasons.append('destination_noncontiguous')
    try:
        memoryview(src_arr)
    except (BufferError, TypeError, ValueError):
        reasons.append('source_unreadable')
    try:
        if memoryview(dst_arr).readonly:
            reasons.append('destination_readonly')
    except (BufferError, TypeError, ValueError):
        reasons.append('destination_unwritable')
    if nbytes == 0 or nbytes < int(minimum_bytes):
        reasons.append('below_minimum_size')
    if nbytes > 0 and int(dst_arr.nbytes) > 0:
        try:
            src_lo, src_hi = _byte_range(src_arr)
            dst_lo, dst_hi = _byte_range(dst_arr)
            if src_lo < dst_hi and dst_lo < src_hi:
                reasons.append('overlapping_ranges')
        except Exception:
            reasons.append('range_unknown')
    allowed_backings = {'anonymous', 'memfd', 'tmpfs'}
    if source_backing not in allowed_backings:
        reasons.append(f'source_backing_{source_backing}')
    if destination_backing not in allowed_backings:
        reasons.append(f'destination_backing_{destination_backing}')
    return CopyEligibility(
        eligible=not reasons,
        reasons=tuple(reasons),
        source_backing=source_backing,
        destination_backing=destination_backing,
        nbytes=nbytes,
    )


def _capability_bool(
    capabilities: Mapping[str, object],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = capabilities.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _validated_capabilities(raw: Mapping[str, object]) -> Dict[str, object]:
    capabilities = dict(raw)
    if not _capability_bool(capabilities, 'hardware_available'):
        reason = capabilities.get('unavailable_reason', 'no enabled user work queue')
        raise IntelDsaUnavailable(f'DSA unavailable: {reason}')
    if str(capabilities.get('interface', '')) != 'idxd-cdev':
        raise IntelDsaUnavailable('DSA binding is not using the required idxd-cdev interface')
    if _capability_bool(capabilities, 'software_fallback_enabled'):
        raise IntelDsaUnavailable('DSA binding advertises software fallback; hardware-only mode is required')
    if not _capability_bool(capabilities, 'drain_guaranteed'):
        raise IntelDsaUnavailable('DSA binding cannot prove deterministic descriptor drain')
    if str(capabilities.get('work_queue_type', '')) != 'user':
        raise IntelDsaUnavailable('DSA work queue is not a user work queue')
    if not _capability_bool(capabilities, 'numa_local'):
        raise IntelDsaUnavailable('DSA work queue is not local to the calling CPU NUMA node')
    work_queue = str(capabilities.get('work_queue', '')).strip()
    if not work_queue:
        raise IntelDsaUnavailable('DSA binding returned no work-queue identity')
    try:
        max_transfer_size = int(capabilities.get('max_transfer_size', 0))
    except Exception as exc:
        raise IntelDsaUnavailable('DSA binding returned an invalid maximum transfer size') from exc
    if max_transfer_size <= 0:
        raise IntelDsaUnavailable('DSA work queue has no positive maximum transfer size')
    capabilities['work_queue'] = work_queue
    capabilities['max_transfer_size'] = max_transfer_size
    return capabilities


class DsaManager:
    """Thread-safe, process-local owner of one optional native DSA binding."""

    def __init__(self, module: NativeDsaModule) -> None:
        self.module = module
        self.pid = int(os.getpid())
        self.lock = threading.RLock()
        self.closed = False
        self.poisoned = False
        self._quarantined: Dict[
            tuple[int, int],
            tuple[object, object, Optional[Callable[[], None]]],
        ] = {}

    def _check_process(self, *, allow_poisoned: bool = False) -> None:
        if int(os.getpid()) != self.pid:
            raise IntelDsaUnavailable('DSA manager cannot be reused after fork')
        if self.closed:
            raise IntelDsaUnavailable('DSA manager is closed')
        if self.poisoned and not bool(allow_poisoned):
            raise IntelDsaUnavailable(
                'DSA manager is quarantined after an unproven drain; close it before reuse'
            )

    def capabilities(self, *, work_queue: Optional[str] = None) -> Dict[str, object]:
        with self.lock:
            self._check_process()
            try:
                raw = self.module.capabilities(work_queue=work_queue)
                return _validated_capabilities(raw)
            except IntelDsaUnavailable:
                raise
            except Exception as exc:
                raise IntelDsaUnavailable(
                    f'DSA capability probe failed: {type(exc).__name__}: {exc}'
                ) from exc

    def _drain_after_failure(self) -> tuple[bool, Dict[str, object], Optional[BaseException]]:
        try:
            raw = self.module.drain()
            result = dict(raw)
            if not _capability_bool(result, 'drained'):
                raise RuntimeError('native drain did not acknowledge drained=true')
            return True, result, None
        except BaseException as exc:
            return False, {}, exc

    def copy(
        self,
        src: object,
        dst: object,
        *,
        capabilities: Mapping[str, object],
        max_inflight: int,
        failure_cleanup: Optional[Callable[[], None]] = None,
    ) -> Dict[str, object]:
        with self.lock:
            self._check_process()
            expected_bytes = int(np.asarray(src).nbytes)
            native_stats: Dict[str, object] = {}
            try:
                raw = self.module.copy(
                    src,
                    dst,
                    work_queue=str(capabilities['work_queue']),
                    max_transfer_size=int(capabilities['max_transfer_size']),
                    max_inflight=int(max_inflight),
                    require_hardware=True,
                )
                native_stats = dict(raw)
                if not _capability_bool(native_stats, 'drained'):
                    raise RuntimeError('native copy did not prove drained=true')
                if not _capability_bool(native_stats, 'hardware_only'):
                    raise RuntimeError('native copy did not prove hardware-only execution')
                if int(native_stats.get('software_bytes', 0)) != 0:
                    raise RuntimeError('native copy reported software-executed bytes')
                if int(native_stats.get('hardware_bytes', -1)) != expected_bytes:
                    raise RuntimeError(
                        'native copy hardware byte count does not match the complete input'
                    )
                return native_stats
            except BaseException as exc:
                exception_stats = getattr(exc, 'stats', None)
                if isinstance(exception_stats, Mapping):
                    native_stats.update(dict(exception_stats))
                drained, drain_stats, drain_error = self._drain_after_failure()
                native_stats.update(drain_stats)
                detail = f'{type(exc).__name__}: {exc}'
                if drain_error is not None:
                    detail += f'; drain failed: {type(drain_error).__name__}: {drain_error}'
                if not drained:
                    # A broken native binding may still have a DMA write targeting these
                    # objects. Keep them (and deferred invalidation) alive until a later
                    # close attempt explicitly proves drain.
                    self._quarantined[(id(src), id(dst))] = (
                        src,
                        dst,
                        failure_cleanup,
                    )
                    self.poisoned = True
                raise IntelDsaCopyError(
                    f'DSA hardware copy failed: {detail}',
                    drained=drained,
                    stats=native_stats,
                ) from exc

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self._check_process(allow_poisoned=True)
            drained, _stats, drain_error = self._drain_after_failure()
            close_error: Optional[BaseException] = None
            try:
                self.module.close()
            except BaseException as exc:
                close_error = exc
            if not drained or drain_error is not None or close_error is not None:
                detail = drain_error or close_error or RuntimeError('native DSA drain was not acknowledged')
                raise IntelDsaError(f'failed to close DSA manager cleanly: {detail}') from detail
            self.closed = True
            self.poisoned = False
            quarantined = list(self._quarantined.values())
            self._quarantined.clear()
            first_cleanup_error: Optional[BaseException] = None
            for _src, _dst, cleanup in quarantined:
                if cleanup is None:
                    continue
                try:
                    cleanup()
                except BaseException as exc:
                    if first_cleanup_error is None:
                        first_cleanup_error = exc
            if first_cleanup_error is not None:
                raise IntelDsaError(
                    f'DSA drained but quarantined workspace cleanup failed: {first_cleanup_error}'
                ) from first_cleanup_error


def get_manager() -> DsaManager:
    global _MANAGER, _ATEXIT_REGISTERED
    with _STATE_LOCK:
        manager = _MANAGER
        if manager is not None and manager.pid == int(os.getpid()) and not manager.closed:
            return manager
        manager = DsaManager(_load_native_module())
        _MANAGER = manager
        if not _ATEXIT_REGISTERED:
            atexit.register(_close_manager_at_exit)
            _ATEXIT_REGISTERED = True
        return manager


def close_manager() -> None:
    global _MANAGER
    with _STATE_LOCK:
        manager = _MANAGER
        if manager is not None:
            # Retain the manager and its quarantined buffer references if close cannot
            # prove drain. Dropping them could unmap an active DMA target.
            manager.close()
            _MANAGER = None


def _close_manager_at_exit() -> None:
    """Best-effort shutdown without an unraisable interpreter-exit exception."""
    try:
        close_manager()
    except BaseException:
        pass


def _set_test_module(module: Optional[NativeDsaModule]) -> None:
    """Install a fake native binding for hardware-independent regression tests."""
    global _TEST_MODULE, _LOADED_MODULE, _IMPORT_ERROR, _MANAGER
    close_error: Optional[BaseException] = None
    with _STATE_LOCK:
        manager = _MANAGER
        if manager is not None:
            try:
                manager.close()
            except BaseException as exc:
                close_error = exc
            else:
                _MANAGER = None
        if close_error is not None:
            # Do not replace a binding while its old DMA targets remain quarantined.
            raise close_error
        _TEST_MODULE = None if module is None else _validate_native_module(module)
        _LOADED_MODULE = None
        _IMPORT_ERROR = None


def _reset_for_tests() -> None:
    _set_test_module(None)


def loaded() -> bool:
    """Report cached state without importing or probing the companion module."""
    with _STATE_LOCK:
        return bool(_LOADED_MODULE is not None or _TEST_MODULE is not None or _MANAGER is not None)


def _after_fork_child() -> None:
    global _STATE_LOCK, _LOADED_MODULE, _TEST_MODULE, _IMPORT_ERROR, _MANAGER
    _STATE_LOCK = threading.RLock()
    _LOADED_MODULE = None
    _TEST_MODULE = None
    _IMPORT_ERROR = None
    _MANAGER = None


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork_child)


__all__ = (
    'CopyEligibility',
    'DsaManager',
    'IntelDsaCopyError',
    'IntelDsaError',
    'IntelDsaIneligible',
    'IntelDsaUnavailable',
    'assess_copy_eligibility',
    'classify_array_backing',
    'close_manager',
    'get_manager',
    'loaded',
    'minimum_copy_bytes',
    'platform_supported',
    'requested_backend',
    'requested_max_inflight',
    'requested_work_queue',
)
