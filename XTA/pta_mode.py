"""Dependency-light entry point for the unified PTA runtime."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType

from .pta_config import parse_pta_args


def _load_runtime_module() -> ModuleType:
    """Load numerical and publication dependencies only after CLI validation."""

    return importlib.import_module(".pta_runtime", package=__package__)


def run(argv: Sequence[str] | None = None) -> None:
    """Validate PTA arguments and execute the native runtime."""

    arguments = None if argv is None else [str(value) for value in argv]
    config = parse_pta_args(arguments)
    _load_runtime_module().run(config, argv=arguments)


__all__ = ["run"]
