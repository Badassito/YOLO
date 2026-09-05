"""Dependency-light entry point for the LTA runtime boundary."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType

from .lta_config import parse_lta_args


def _load_runtime_module() -> ModuleType:
    """Import the production SAM runtime only after mode-local CLI validation."""

    return importlib.import_module(".lta_runtime", package=__package__)


def run(argv: Sequence[str] | None = None) -> None:
    """Validate LTA arguments and enter the runtime boundary."""

    arguments = None if argv is None else [str(value) for value in argv]
    config = parse_lta_args(arguments)
    runtime = _load_runtime_module()
    runtime.run(config, argv=arguments)


__all__ = ["run"]
