"""Explicit backend registry with fail-closed lookup semantics."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import BackendId, InferenceBackend, TaskRequirements


class BackendRegistry:
    def __init__(self, backends: Iterable[InferenceBackend] = ()) -> None:
        self._backends: dict[BackendId, InferenceBackend] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: InferenceBackend) -> None:
        backend_id = str(backend.backend_id).strip().lower()
        if not backend_id:
            raise ValueError("backend_id must not be empty")
        if backend_id in self._backends:
            raise ValueError(f"backend {backend_id!r} is already registered")
        self._backends[backend_id] = backend

    def require(self, backend_id: BackendId) -> InferenceBackend:
        normalized = str(backend_id).strip().lower()
        try:
            return self._backends[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._backends)) or "none"
            raise KeyError(
                f"unsupported inference backend {backend_id!r}; registered backends: {available}"
            ) from exc

    def eligible(self, requirements: TaskRequirements) -> tuple[InferenceBackend, ...]:
        return tuple(
            backend
            for backend in self._backends.values()
            if backend.supports(requirements)
        )

    def ids(self) -> tuple[BackendId, ...]:
        return tuple(sorted(self._backends))


__all__ = ("BackendRegistry",)
