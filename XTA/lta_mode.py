"""Dependency-light entry point for the v19 LTA runtime boundary."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from types import ModuleType

from .lta_config import parse_lta_args


def _load_runtime_module() -> ModuleType:
    """Import the future SAM runtime only after mode-local CLI validation."""

    return importlib.import_module(".lta_runtime", package=__package__)


def run(argv: Sequence[str] | None = None) -> None:
    """Validate LTA arguments and enter the future runtime boundary."""

    arguments = None if argv is None else [str(value) for value in argv]
    config = parse_lta_args(arguments)
    runtime = _load_runtime_module()
    try:
        runtime.run(config, argv=arguments)
    except Exception as exc:
        pending_type = getattr(runtime, "LtaPrototypeExecutionPending", None)
        if isinstance(pending_type, type) and isinstance(exc, pending_type):
            print(f"LTA v19 planning prototype: {exc}", file=sys.stderr)
            raise SystemExit(3) from None
        raise


__all__ = ["run"]
