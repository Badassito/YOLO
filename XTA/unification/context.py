"""Process-local marker for execution through the strict unified launcher."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass(frozen=True)
class UnifiedLaunchContext:
    version: str
    launcher: str
    mode: str
    mode_arguments: tuple[str, ...]

    @property
    def command(self) -> tuple[str, ...]:
        return (
            self.launcher,
            "--mode",
            self.mode,
            *self.mode_arguments,
        )


_ACTIVE_UNIFIED_LAUNCH: ContextVar[UnifiedLaunchContext | None] = ContextVar(
    "XTA_active_unified_launch",
    default=None,
)


def current_unified_launch() -> UnifiedLaunchContext | None:
    return _ACTIVE_UNIFIED_LAUNCH.get()


@contextlib.contextmanager
def activate_unified_launch(
    *,
    version: str,
    launcher: str,
    mode: str,
    mode_arguments: Sequence[str],
) -> Iterator[UnifiedLaunchContext]:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"tta", "pta", "lta"}:
        raise ValueError(f"unsupported unified launch mode {mode!r}")
    context = UnifiedLaunchContext(
        version=str(version),
        launcher=str(launcher),
        mode=resolved_mode,
        mode_arguments=tuple(str(value) for value in mode_arguments),
    )
    token = _ACTIVE_UNIFIED_LAUNCH.set(context)
    try:
        yield context
    finally:
        _ACTIVE_UNIFIED_LAUNCH.reset(token)


__all__ = (
    "UnifiedLaunchContext",
    "activate_unified_launch",
    "current_unified_launch",
)
