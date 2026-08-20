"""Import the refactored package with lightweight native-dependency stubs.

The development runtime does not ship OpenCV/SciPy.  These stubs exercise Python import
order, circular-dependency safety, and eager module initialization without pretending to
validate numerical kernels.
"""

from __future__ import annotations

import importlib
import builtins
import dis
import inspect
import os
import sys
import types
from types import CodeType
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str) -> object:
        def unavailable(*args: object, **kwargs: object) -> object:
            raise RuntimeError(f"stubbed dependency attribute called: {self.__name__}.{name}")

        return unavailable


def install_stubs() -> None:
    cv2 = _StubModule("cv2")
    scipy = _StubModule("scipy")
    scipy.__path__ = []  # type: ignore[attr-defined]
    ndimage = _StubModule("scipy.ndimage")
    scipy.ndimage = ndimage  # type: ignore[attr-defined]
    tifffile = _StubModule("tifffile")
    tqdm_module = _StubModule("tqdm")
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    modules = {
        "cv2": cv2,
        "scipy": scipy,
        "scipy.ndimage": ndimage,
        "tifffile": tifffile,
        "tqdm": tqdm_module,
    }
    if os.environ.get("VOLUME_TTA_SMOKE_NUMBA", "").strip() == "1":
        numba = _StubModule("numba")

        def njit(*args: object, **kwargs: object) -> object:
            def decorate(function: object) -> object:
                return function

            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return decorate

        numba.njit = njit  # type: ignore[attr-defined]
        numba.prange = range  # type: ignore[attr-defined]
        modules["numba"] = numba
    sys.modules.update(modules)


def _code_objects(code: CodeType) -> list[CodeType]:
    nested = [code]
    for value in code.co_consts:
        if isinstance(value, CodeType):
            nested.extend(_code_objects(value))
    return nested


def unresolved_function_globals(module: types.ModuleType) -> set[str]:
    missing: set[str] = set()
    builtin_names = set(vars(builtins))
    for value in vars(module).values():
        functions: list[object] = []
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            functions.append(value)
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            functions.extend(
                member
                for member in vars(value).values()
                if inspect.isfunction(member) and member.__module__ == module.__name__
            )
        for function in functions:
            function_globals = function.__globals__
            for code in _code_objects(function.__code__):
                for instruction in dis.get_instructions(code):
                    if instruction.opname != "LOAD_GLOBAL":
                        continue
                    name = str(instruction.argval)
                    if name not in function_globals and name not in builtin_names:
                        missing.add(name)
    return missing


def main() -> None:
    install_stubs()
    default_modules = (
        "config",
        "workspace",
        "runtime",
        "media",
        "geometry",
        "inference",
        "cuda_backend",
        "workers",
        "topology",
        "backprojection",
        "finalization",
        "interpolation",
        "cuda_d1",
        "assembly",
        "outputs",
        "pipeline",
    )
    requested = [str(name).strip() for name in sys.argv[1:] if str(name).strip()]
    unknown = sorted(set(requested) - set(default_modules))
    if unknown:
        raise ValueError(f"unknown subsystem module(s): {', '.join(unknown)}")
    modules = tuple(requested) + tuple(name for name in default_modules if name not in requested)
    loaded_modules = []
    for name in modules:
        loaded_modules.append(importlib.import_module(f"volume_tta.{name}"))
        print(f"imported volume_tta.{name}")
    from volume_tta._latebind import unresolved_bindings

    unresolved = unresolved_bindings()
    if unresolved:
        raise RuntimeError(f"unresolved callable dependencies: {unresolved}")
    print("all callable-only dependencies resolved")
    missing_globals = {
        module.__name__: sorted(unresolved_function_globals(module))
        for module in loaded_modules
        if unresolved_function_globals(module)
    }
    if missing_globals:
        raise RuntimeError(f"unresolved function globals: {missing_globals}")
    print("all function globals resolved")


if __name__ == "__main__":
    main()
