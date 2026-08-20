"""Cycle-safe linker for callable-only dependencies in the physical split.

The original implementation was one Python module, so callable bodies could freely
refer to definitions declared thousands of lines later.  During the behavior-preserving
physical split those references form legitimate subsystem cycles.  Eager dependencies
(base classes, decorators, defaults and module initializers) remain normal imports;
callable-only names are registered here and resolved after their provider has defined
them.

This module starts no threads, processes, telemetry or accelerator runtimes.
"""

from __future__ import annotations

import importlib
import sys
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import MutableMapping


@dataclass
class _BindingRequest:
    namespace: MutableMapping[str, object]
    pending: dict[str, set[str]] = field(default_factory=dict)


_LOCK = threading.RLock()
_REQUESTS: dict[str, _BindingRequest] = {}
_RESOLVING = False
_REQUEST_GENERATION = 0


def _provider_module(consumer: str, provider: str) -> str:
    package = consumer.rpartition(".")[0]
    if not package:
        raise ImportError(f"Late-bound module {consumer!r} has no package")
    return f"{package}.{provider}"


def _resolve_registered() -> None:
    global _RESOLVING
    with _LOCK:
        if _RESOLVING:
            return
        _RESOLVING = True
    try:
        while True:
            progress = False
            with _LOCK:
                generation = _REQUEST_GENERATION
                snapshot = list(_REQUESTS.items())
            for consumer, request in snapshot:
                for provider, names in list(request.pending.items()):
                    qualified = _provider_module(consumer, provider)
                    module: ModuleType
                    try:
                        module = importlib.import_module(qualified)
                    except (ImportError, RuntimeError) as exc:
                        # A normal failure from inside a fully initialized provider is a
                        # real failure; only partially initialized cycles are deferred.
                        #
                        # Concurrent imports add one more cycle shape: importlib detects
                        # that two threads own opposing module locks and raises its private
                        # ``_DeadlockError`` (a RuntimeError) in one of them.  The provider
                        # is already present in sys.modules in that case.  Deferring its
                        # callable-only names lets this module finish and release its lock,
                        # after which the active resolver's next pass can bind them.
                        partial = sys.modules.get(qualified)
                        is_import_cycle = isinstance(exc, ImportError)
                        is_thread_cycle = type(exc).__name__ == "_DeadlockError"
                        if partial is None or not (is_import_cycle or is_thread_cycle):
                            raise
                        module = partial
                    resolved = {name for name in names if hasattr(module, name)}
                    if resolved:
                        request.namespace.update(
                            {name: getattr(module, name) for name in resolved}
                        )
                        names.difference_update(resolved)
                        progress = True
                    if not names:
                        request.pending.pop(provider, None)
                if not request.pending:
                    with _LOCK:
                        _REQUESTS.pop(consumer, None)
            if not progress:
                with _LOCK:
                    # A module imported by another thread can register a request while
                    # this resolver is walking its snapshot.  Do not strand that request
                    # merely because the old snapshot made no progress.  The stable
                    # generation check and resolver hand-off happen under the same lock,
                    # so a registration either belongs to this pass or starts a new one.
                    if generation != _REQUEST_GENERATION:
                        continue
                    _RESOLVING = False
                    return
    finally:
        with _LOCK:
            _RESOLVING = False


def bind_late_symbols(
    consumer: str,
    namespace: MutableMapping[str, object],
    dependencies: dict[str, tuple[str, ...]],
) -> None:
    """Register and resolve names used only after module initialization."""

    global _REQUEST_GENERATION
    request = _BindingRequest(
        namespace=namespace,
        pending={provider: set(names) for provider, names in dependencies.items()},
    )
    with _LOCK:
        _REQUESTS[consumer] = request
        _REQUEST_GENERATION += 1
    _resolve_registered()


def unresolved_bindings() -> dict[str, dict[str, tuple[str, ...]]]:
    """Return unresolved bindings for diagnostics and import-smoke tests."""

    _resolve_registered()
    with _LOCK:
        return {
            consumer: {
                provider: tuple(sorted(names))
                for provider, names in request.pending.items()
            }
            for consumer, request in _REQUESTS.items()
        }


__all__ = ("bind_late_symbols", "unresolved_bindings")
