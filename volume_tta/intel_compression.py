"""Lazy control plane for optional Intel hardware gzip backends.

Linux x86-64 source builds conditionally compile the project-owned
``volume_tta._qat_codec`` and ``volume_tta._qpl_codec`` extensions when suitable
QATzip/QPL development libraries are available. Other builds remain usable
without them. This module is the only place that imports either extension, and
only after NRRD output policy selects that backend.

The native interface and build requirements are documented in ``native/README.md``.
In particular,
``compress_gzip(..., require_hardware=True)`` must raise unless the entire input
was consumed by hardware with no software execution.  Its bytes may contain one
or more concatenated, complete RFC-1952 members.
"""

from __future__ import annotations

import gzip
import importlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol


class IntelCompressionError(RuntimeError):
    """Base error for an optional Intel compression backend."""


class IntelCompressionUnavailable(IntelCompressionError):
    """The requested extension or a usable hardware execution path is absent."""

    def __init__(self, message: str, *, module_missing: bool = False) -> None:
        super().__init__(message)
        self.module_missing = bool(module_missing)


class NativeCompressionModule(Protocol):
    """Structural contract implemented by QATzip/QPL companion extensions."""

    def capabilities(self) -> Mapping[str, object]: ...

    def compress_gzip(
        self,
        buffer: object,
        level: int,
        *,
        require_hardware: bool = True,
        numa_id: Optional[int] = None,
    ) -> bytes: ...

    def stats(self, *, reset: bool = False) -> Mapping[str, object]: ...

    def close_thread_state(self) -> None: ...


_MODULE_NAMES = {
    'qat': 'volume_tta._qat_codec',
    'iaa': 'volume_tta._qpl_codec',
}

_MODULE_LOCK = threading.RLock()
_LOADED_MODULES: Dict[str, NativeCompressionModule] = {}
_TEST_MODULES: Dict[str, NativeCompressionModule] = {}
_IMPORT_ERRORS: Dict[str, tuple[str, bool]] = {}


def _normalize_backend(backend: object) -> str:
    name = str(backend).strip().lower()
    if name not in _MODULE_NAMES:
        raise ValueError(f'Unsupported Intel compression backend {backend!r}')
    return name


def _validate_native_module(name: str, module: object) -> NativeCompressionModule:
    for symbol in ('capabilities', 'compress_gzip', 'stats', 'close_thread_state'):
        if not callable(getattr(module, symbol, None)):
            raise IntelCompressionUnavailable(
                f'{name} companion module does not implement required {symbol}()'
            )
    return module  # type: ignore[return-value]


def _load_native_module(backend: object) -> NativeCompressionModule:
    """Import one optional extension on demand; successful imports are cached."""
    name = _normalize_backend(backend)
    with _MODULE_LOCK:
        override = _TEST_MODULES.get(name)
        if override is not None:
            return _validate_native_module(name, override)
        cached = _LOADED_MODULES.get(name)
        if cached is not None:
            return cached
        cached_error = _IMPORT_ERRORS.get(name)
        if cached_error is not None:
            message, module_missing = cached_error
            raise IntelCompressionUnavailable(
                str(message), module_missing=bool(module_missing),
            )
    module_name = _MODULE_NAMES[name]
    try:
        imported = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only the companion module itself being absent is a quiet auto-mode miss.
        # A missing dependency imported *by* an installed extension is actionable.
        missing_self = str(getattr(exc, 'name', '')) == str(module_name)
        message = (
            f'{name} companion module is not installed ({module_name})'
            if missing_self else
            f'{name} companion module dependency is missing: {exc}'
        )
        with _MODULE_LOCK:
            _IMPORT_ERRORS[name] = (str(message), bool(missing_self))
        raise IntelCompressionUnavailable(
            message, module_missing=bool(missing_self),
        ) from exc
    except Exception as exc:
        message = f'{name} companion module failed to import: {type(exc).__name__}: {exc}'
        with _MODULE_LOCK:
            _IMPORT_ERRORS[name] = (str(message), False)
        raise IntelCompressionUnavailable(message) from exc
    validated = _validate_native_module(name, imported)
    with _MODULE_LOCK:
        _LOADED_MODULES[name] = validated
    return validated


def _capability_bool(
    capabilities: Mapping[str, object],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = capabilities.get(str(key), bool(default))
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def probe_capabilities(backend: object) -> Dict[str, object]:
    """Return and validate a fresh hardware capability snapshot."""
    name = _normalize_backend(backend)
    module = _load_native_module(name)
    try:
        raw = module.capabilities()
        capabilities = dict(raw)
    except Exception as exc:
        raise IntelCompressionUnavailable(
            f'{name} hardware capability probe failed: {type(exc).__name__}: {exc}'
        ) from exc
    if not _capability_bool(capabilities, 'hardware_available'):
        reason = str(capabilities.get('unavailable_reason', 'no usable hardware path'))
        raise IntelCompressionUnavailable(f'{name} unavailable: {reason}')
    if not _capability_bool(capabilities, 'standard_gzip', default=True):
        raise IntelCompressionUnavailable(
            f'{name} binding does not advertise standard RFC-1952 gzip output'
        )
    if _capability_bool(capabilities, 'software_fallback_enabled'):
        raise IntelCompressionUnavailable(
            f'{name} binding reports software fallback enabled; hardware-only mode is required'
        )
    return capabilities


def _identity_tuple(name: str, capabilities: Mapping[str, object]) -> tuple[object, ...]:
    """Stable-enough KAT identity without retaining arbitrary native objects."""
    keys = (
        'binding_version',
        'library_version',
        'driver_version',
        'hardware_generation',
        'device_count',
        'instance_count',
        'work_queue_count',
        'device_identity',
    )
    return (str(name),) + tuple(repr(capabilities.get(key, None)) for key in keys)


@dataclass
class NativeGzipCompressor:
    """Callable adapter around one hardware-only companion extension."""

    backend: str
    level: int
    module: NativeCompressionModule
    capabilities: Dict[str, object]
    numa_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.backend = _normalize_backend(self.backend)
        self.level = int(self.level)
        self.hardware_backend = True
        advertised = self.capabilities.get(
            'max_concurrency', self.capabilities.get('instance_count', 1)
        )
        try:
            self.max_concurrency = max(1, int(advertised))
        except Exception:
            self.max_concurrency = 1
        try:
            self.minimum_input_bytes = max(
                1, int(self.capabilities.get('minimum_input_bytes', 1))
            )
        except Exception:
            self.minimum_input_bytes = 1
        self.cache_key = (
            self.backend,
            int(self.level),
            _identity_tuple(self.backend, self.capabilities),
        )
        self._stats_lock = threading.Lock()
        self._input_bytes = 0
        self._output_bytes = 0
        self._requests = 0
        self._failures = 0
        self._elapsed_ns = 0

    def __call__(self, payload: object) -> bytes:
        mv = payload if isinstance(payload, memoryview) else memoryview(payload)  # type: ignore[arg-type]
        mv = mv.cast('B')
        if not bool(mv.contiguous):
            raise IntelCompressionError(
                f'{self.backend} requires a contiguous one-dimensional input buffer'
            )
        if len(mv) < int(self.minimum_input_bytes):
            raise IntelCompressionError(
                f'{self.backend} hardware minimum input is {self.minimum_input_bytes} bytes; '
                f'received {len(mv)} bytes'
            )
        started = time.monotonic_ns()
        failed = False
        try:
            encoded = self.module.compress_gzip(
                mv,
                int(self.level),
                require_hardware=True,
                numa_id=self.numa_id,
            )
            result = bytes(encoded)
            if len(result) < 18 or result[:3] != b'\x1f\x8b\x08':
                raise IntelCompressionError(
                    f'{self.backend} returned invalid standard gzip framing'
                )
            return result
        except Exception:
            failed = True
            raise
        finally:
            elapsed = int(time.monotonic_ns() - started)
            with self._stats_lock:
                self._input_bytes += int(len(mv))
                self._requests += 1
                self._failures += int(failed)
                self._elapsed_ns += elapsed
                if not failed and 'result' in locals():
                    self._output_bytes += int(len(result))

    def preflight_thread_state(self) -> None:
        """Initialize and prove one executor thread's hardware session/job."""
        native_preflight = getattr(self.module, 'preflight_thread_state', None)
        if callable(native_preflight):
            native_preflight(
                int(self.level), require_hardware=True, numa_id=self.numa_id,
            )
        # A native hook may initialize thread-local state, but its ``None`` return cannot
        # prove RFC-1952 framing or a completed hardware request to this control plane.
        # Always follow it with one deterministic hardware-only gzip round trip on the
        # same executor thread. This also supplies preflight for the minimum interface,
        # which intentionally does not require a dedicated native hook.
        size = max(int(self.minimum_input_bytes), 128 * 1024)
        seed = b'volume-tta-intel-hardware-preflight\x00'
        payload = (seed * ((int(size) + len(seed) - 1) // len(seed)))[:int(size)]
        encoded = self(payload)
        if gzip.decompress(encoded) != payload:
            raise IntelCompressionError(
                f'{self.backend} thread preflight gzip round trip failed'
            )

    def local_stats(self) -> Dict[str, object]:
        with self._stats_lock:
            return {
                'input_bytes': int(self._input_bytes),
                'output_bytes': int(self._output_bytes),
                'logical_requests': int(self._requests),
                'failures': int(self._failures),
                'elapsed_seconds': float(self._elapsed_ns) / 1e9,
            }


def create_gzip_compressor(
    backend: object,
    level: int,
    *,
    numa_id: Optional[int] = None,
    capabilities: Optional[Mapping[str, object]] = None,
) -> NativeGzipCompressor:
    """Construct a fresh callable after proving a hardware-only path exists."""
    name = _normalize_backend(backend)
    resolved_capabilities = (
        dict(capabilities) if capabilities is not None else probe_capabilities(name)
    )
    module = _load_native_module(name)
    return NativeGzipCompressor(
        backend=name,
        level=int(level),
        module=module,
        capabilities=resolved_capabilities,
        numa_id=(None if numa_id is None else int(numa_id)),
    )


def native_stats(backend: object, *, reset: bool = False) -> Dict[str, object]:
    """Read native counters without importing a backend that has never been used."""
    name = _normalize_backend(backend)
    with _MODULE_LOCK:
        module = _TEST_MODULES.get(name) or _LOADED_MODULES.get(name)
    if module is None:
        return {}
    try:
        return dict(module.stats(reset=bool(reset)))
    except Exception as exc:
        raise IntelCompressionError(
            f'{name} native stats failed: {type(exc).__name__}: {exc}'
        ) from exc


def close_current_thread_state() -> None:
    """Close every already-loaded native session/job owned by the calling thread."""
    with _MODULE_LOCK:
        modules = list({id(module): module for module in (
            list(_LOADED_MODULES.values()) + list(_TEST_MODULES.values())
        )}.values())
    first_error: Optional[BaseException] = None
    for module in modules:
        try:
            module.close_thread_state()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise IntelCompressionError(
            f'failed to close Intel compression thread state: {first_error}'
        ) from first_error


def _set_test_module(backend: object, module: Optional[NativeCompressionModule]) -> None:
    """Install or remove a fake module for hardware-independent unit tests."""
    name = _normalize_backend(backend)
    with _MODULE_LOCK:
        _LOADED_MODULES.pop(name, None)
        _IMPORT_ERRORS.pop(name, None)
        if module is None:
            _TEST_MODULES.pop(name, None)
        else:
            _TEST_MODULES[name] = _validate_native_module(name, module)


def _reset_for_tests() -> None:
    with _MODULE_LOCK:
        _TEST_MODULES.clear()
        _LOADED_MODULES.clear()
        _IMPORT_ERRORS.clear()


def loaded_backend_names() -> tuple[str, ...]:
    """Names already imported in this process; this function never probes hardware."""
    with _MODULE_LOCK:
        return tuple(sorted(set(_LOADED_MODULES) | set(_TEST_MODULES)))


def _after_fork_child() -> None:
    """Discard Python caches; native bindings must also PID-guard their TLS state."""
    global _MODULE_LOCK
    # Another parent thread may have owned this lock when fork() cloned the process.
    # That owner does not exist in the child, so acquiring the inherited lock can
    # deadlock forever. Install a child-local lock before touching any cache, and do
    # not retain test/native objects whose process-local state was created pre-fork.
    _MODULE_LOCK = threading.RLock()
    _TEST_MODULES.clear()
    _LOADED_MODULES.clear()
    _IMPORT_ERRORS.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork_child)


__all__ = (
    'IntelCompressionError',
    'IntelCompressionUnavailable',
    'NativeCompressionModule',
    'NativeGzipCompressor',
    'close_current_thread_state',
    'create_gzip_compressor',
    'loaded_backend_names',
    'native_stats',
    'probe_capabilities',
)
