"""Deterministic, atomic JSON support for v18 run manifests."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Mapping


def json_compatible(value: Any, *, path: str = "payload") -> Any:
    """Convert common configuration values without importing numerical runtimes."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return json_compatible(value.value, path=path)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_compatible(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string mapping key {key!r}")
            converted[key] = json_compatible(item, path=f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            json_compatible(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        converted = [json_compatible(item, path=f"{path}[]") for item in value]
        try:
            return sorted(converted)
        except TypeError as exc:
            raise TypeError(f"{path} contains a non-deterministically sortable set") from exc

    # NumPy scalar values provide item(), but importing NumPy merely to identify
    # them would make launcher/manifest discovery unnecessarily heavy.
    item_method = getattr(value, "item", None)
    value_module = str(getattr(type(value), "__module__", ""))
    if value_module.startswith("numpy") and callable(item_method):
        return json_compatible(item_method(), path=path)
    raise TypeError(f"{path} contains unsupported value {value!r}")


def write_json_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write sorted JSON through a same-directory atomic replacement."""

    if not isinstance(payload, Mapping):
        raise TypeError("manifest payload must be a mapping")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = json_compatible(payload)
    serialized = json.dumps(
        normalized,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


__all__ = ("json_compatible", "write_json_manifest")
