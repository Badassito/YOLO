"""Optional ctypes binding for lossless GPU multi-page TIFF publication.

The NVIDIA nvTIFF wheel contains native binaries rather than a direct Python
API.  nvImageCodec exposes single-image TIFF encoding, but does not expose the
native ``nvtiffEncodeParamsSetInputs`` operation needed to combine several
device images into one TIFF IFD chain.  This module binds only that small,
stable nvTIFF 0.8 encode surface.

No CUDA runtime is imported here.  Callers retain ownership of CUDA image
buffers until a write returns.  The backend's bound CUDA stream and device
context must remain alive/current until :meth:`NvTiffBackend.close` returns.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.metadata
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


MINIMUM_NVTIFF_VERSION = (0, 8, 0)
NVTIFF_LIBRARY_ENV = "XTA_NVTIFF_LIBRARY"

_MAX_NUM_SAMPLES = 16
_UINT32_MAX = (1 << 32) - 1
_POINTER_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1

# CUDA library_types.h
_MAJOR_VERSION = 0
_MINOR_VERSION = 1
_PATCH_LEVEL = 2

# nvtiff.h / TIFF tag values.
_NVTIFF_STATUS_SUCCESS = 0
_NVTIFF_IMAGETYPE_PAGE = 0x2
_NVTIFF_COMPRESSION_LZW = 5
_NVTIFF_PHOTOMETRIC_MINISBLACK = 1
_NVTIFF_PLANARCONFIG_CONTIG = 1
_NVTIFF_SAMPLEFORMAT_UINT = 1
_NVTIFF_BIG_TIFF = 1

# LZW can expand incompressible input.  Switching well below regular TIFF's
# 4-GiB offset ceiling leaves room for worst-case codes, IFDs, and strip tables.
_BIGTIFF_RAW_INPUT_THRESHOLD = 2 * 1024 * 1024 * 1024

_STATUS_NAMES = {
    0: "NVTIFF_STATUS_SUCCESS",
    1: "NVTIFF_STATUS_NOT_INITIALIZED",
    2: "NVTIFF_STATUS_INVALID_PARAMETER",
    3: "NVTIFF_STATUS_BAD_TIFF",
    4: "NVTIFF_STATUS_TIFF_NOT_SUPPORTED",
    5: "NVTIFF_STATUS_ALLOCATOR_FAILURE",
    6: "NVTIFF_STATUS_EXECUTION_FAILED",
    7: "NVTIFF_STATUS_ARCH_MISMATCH",
    8: "NVTIFF_STATUS_INTERNAL_ERROR",
    9: "NVTIFF_STATUS_NVCOMP_NOT_FOUND",
    10: "NVTIFF_STATUS_NVJPEG_NOT_FOUND",
    11: "NVTIFF_STATUS_TAG_NOT_FOUND",
    12: "NVTIFF_STATUS_PARAMETER_OUT_OF_BOUNDS",
    13: "NVTIFF_STATUS_NVJPEG2K_NOT_FOUND",
    14: "NVTIFF_STATUS_BATCH_INCOMPATIBLE",
}


class NvTiffError(RuntimeError):
    """Base exception raised by the optional nvTIFF backend."""


class NvTiffUnavailableError(NvTiffError):
    """nvTIFF could not be found or does not meet the required ABI version."""

    def __init__(self, message: str, *, attempts: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.attempts = tuple(str(value) for value in attempts)


class NvTiffCallError(NvTiffError):
    """One nvTIFF C API operation returned a non-success status."""

    def __init__(self, operation: str, status: int) -> None:
        self.operation = str(operation)
        self.status = int(status)
        self.status_name = _STATUS_NAMES.get(self.status, "NVTIFF_STATUS_UNKNOWN")
        super().__init__(
            f"{self.operation} failed with {self.status_name} ({self.status})"
        )


@dataclass(frozen=True)
class NvTiffCapability:
    """Non-throwing, user-displayable result from :func:`probe_nvtiff`."""

    available: bool
    version: tuple[int, int, int] | None
    library_path: str | None
    diagnostic: str
    attempts: tuple[str, ...] = ()


# These definitions mirror the public nvTIFF 0.8 C ABI.  C enums have int ABI
# and opaque handles/CUDA streams are pointers.
_DeviceMallocAsync = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_void_p,
)
_DeviceFreeAsync = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
)
_PinnedMallocAsync = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_void_p,
)
_PinnedFreeAsync = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
)


class _NvTiffDeviceAllocator(ctypes.Structure):
    _fields_ = [
        ("device_malloc", _DeviceMallocAsync),
        ("device_free", _DeviceFreeAsync),
        ("device_ctx", ctypes.c_void_p),
    ]


class _NvTiffPinnedAllocator(ctypes.Structure):
    _fields_ = [
        ("pinned_malloc", _PinnedMallocAsync),
        ("pinned_free", _PinnedFreeAsync),
        ("pinned_ctx", ctypes.c_void_p),
    ]


class _NvTiffImageInfo(ctypes.Structure):
    _fields_ = [
        ("image_type", ctypes.c_uint32),
        ("image_width", ctypes.c_uint32),
        ("image_height", ctypes.c_uint32),
        ("compression", ctypes.c_int),
        ("photometric_int", ctypes.c_int),
        ("planar_config", ctypes.c_int),
        ("samples_per_pixel", ctypes.c_uint16),
        ("bits_per_pixel", ctypes.c_uint16),
        ("bits_per_sample", ctypes.c_uint16 * _MAX_NUM_SAMPLES),
        ("sample_format", ctypes.c_int * _MAX_NUM_SAMPLES),
    ]


_Uint8DevicePointer = ctypes.POINTER(ctypes.c_uint8)
_EncodeParamsHandle = ctypes.c_void_p
_EncoderHandle = ctypes.c_void_p
_CudaStream = ctypes.c_void_p


_FUNCTION_SIGNATURES = {
    "nvtiffGetProperty": (
        (ctypes.c_int, ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int,
    ),
    "nvtiffEncoderCreate": (
        (
            ctypes.POINTER(_EncoderHandle),
            ctypes.POINTER(_NvTiffDeviceAllocator),
            ctypes.POINTER(_NvTiffPinnedAllocator),
            _CudaStream,
        ),
        ctypes.c_int,
    ),
    "nvtiffEncoderDestroy": (
        (_EncoderHandle, _CudaStream),
        ctypes.c_int,
    ),
    "nvtiffEncodeParamsCreate": (
        (ctypes.POINTER(_EncodeParamsHandle),),
        ctypes.c_int,
    ),
    "nvtiffEncodeParamsDestroy": (
        (_EncodeParamsHandle, _CudaStream),
        ctypes.c_int,
    ),
    "nvtiffEncodeParamsSetImageInfo": (
        (_EncodeParamsHandle, ctypes.POINTER(_NvTiffImageInfo)),
        ctypes.c_int,
    ),
    "nvtiffEncodeParamsSetTiffVariant": (
        (_EncodeParamsHandle, ctypes.c_int),
        ctypes.c_int,
    ),
    "nvtiffEncodeParamsSetInputs": (
        (
            _EncodeParamsHandle,
            ctypes.POINTER(_Uint8DevicePointer),
            ctypes.c_uint32,
        ),
        ctypes.c_int,
    ),
    "nvtiffEncode": (
        (
            _EncoderHandle,
            ctypes.POINTER(_EncodeParamsHandle),
            ctypes.c_uint32,
            _CudaStream,
        ),
        ctypes.c_int,
    ),
    "nvtiffEncodeFinalize": (
        (
            _EncoderHandle,
            ctypes.POINTER(_EncodeParamsHandle),
            ctypes.c_uint32,
            _CudaStream,
        ),
        ctypes.c_int,
    ),
    "nvtiffWriteTiffFile": (
        (
            _EncoderHandle,
            ctypes.POINTER(_EncodeParamsHandle),
            ctypes.c_uint32,
            ctypes.c_char_p,
            _CudaStream,
        ),
        ctypes.c_int,
    ),
}


def _status_value(value: object) -> int:
    raw = getattr(value, "value", value)
    return int(raw)  # type: ignore[arg-type]


class _NvTiffCAPI:
    def __init__(self, library: object) -> None:
        self.library = library
        missing: list[str] = []
        for name, (argument_types, result_type) in _FUNCTION_SIGNATURES.items():
            try:
                function = getattr(library, name)
            except AttributeError:
                missing.append(name)
                continue
            function.argtypes = list(argument_types)
            function.restype = result_type
            setattr(self, name, function)
        if missing:
            raise NvTiffUnavailableError(
                "nvTIFF library is missing required 0.8 symbols: "
                + ", ".join(missing)
            )

    def checked(self, operation: str, *arguments: object) -> None:
        status = _status_value(getattr(self, operation)(*arguments))
        if status != _NVTIFF_STATUS_SUCCESS:
            raise NvTiffCallError(operation, status)

    def version(self) -> tuple[int, int, int]:
        components: list[int] = []
        for property_type in (_MAJOR_VERSION, _MINOR_VERSION, _PATCH_LEVEL):
            value = ctypes.c_int(-1)
            self.checked(
                "nvtiffGetProperty",
                ctypes.c_int(property_type),
                ctypes.byref(value),
            )
            components.append(int(value.value))
        return (components[0], components[1], components[2])


_WHEEL_DISTRIBUTIONS_BY_CUDA_MAJOR = {
    13: ("nvidia-nvtiff-cu13",),
    12: ("nvidia-nvtiff-cu12", "nvidia-nvtiff-tegra-cu12"),
}


def _looks_like_nvtiff_library(path: os.PathLike[str] | str) -> bool:
    name = Path(os.fspath(path)).name.lower()
    if sys.platform.startswith("win"):
        return name.endswith(".dll") and "nvtiff" in name
    return name.startswith("libnvtiff.so") or name == "libnvtiff.dylib"


def _wheel_candidates_for_distributions(
    distribution_names: Sequence[str],
) -> list[str]:
    candidates: list[str] = []
    for distribution_name in distribution_names:
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for relative in distribution.files or ():
            if not _looks_like_nvtiff_library(str(relative)):
                continue
            candidate = Path(distribution.locate_file(relative))
            if candidate.is_file():
                candidates.append(str(candidate))
    return candidates


def _wheel_library_candidates(cuda_major: int | None = None) -> Iterator[str]:
    if cuda_major is not None:
        yield from _wheel_candidates_for_distributions(
            _WHEEL_DISTRIBUTIONS_BY_CUDA_MAJOR.get(cuda_major, ())
        )
        return

    candidates_by_major = {
        major: _wheel_candidates_for_distributions(distribution_names)
        for major, distribution_names in _WHEEL_DISTRIBUTIONS_BY_CUDA_MAJOR.items()
    }
    installed_majors = [
        major for major, candidates in candidates_by_major.items() if candidates
    ]
    if len(installed_majors) > 1:
        raise NvTiffUnavailableError(
            "multiple CUDA-major nvTIFF wheels are installed "
            f"({sorted(installed_majors)}); pass cuda_major explicitly so the "
            "backend cannot bind an incompatible build"
        )
    if installed_majors:
        yield from candidates_by_major[installed_majors[0]]


def _default_library_names() -> tuple[str, ...]:
    if sys.platform.startswith("win"):
        return ("nvtiff64_0.dll", "nvtiff_0.dll", "nvtiff.dll")
    if sys.platform.startswith("linux"):
        return ("libnvtiff.so.0", "libnvtiff.so")
    return ("libnvtiff.dylib",)


def _library_candidates(
    library_path: os.PathLike[str] | str | None = None,
    *,
    cuda_major: int | None = None,
) -> tuple[str, ...]:
    explicit = library_path
    if explicit is None:
        environment_value = os.environ.get(NVTIFF_LIBRARY_ENV, "").strip()
        explicit = environment_value or None
    if explicit is not None:
        return (os.fspath(explicit),)

    candidates: list[str] = list(_wheel_library_candidates(cuda_major))
    discovered = ctypes.util.find_library("nvtiff")
    if discovered:
        candidates.append(str(discovered))
    candidates.extend(_default_library_names())

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = (
            os.path.normcase(os.path.abspath(candidate))
            if os.path.isabs(candidate)
            else candidate
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


def _load_cdll(candidate: str) -> object:
    if sys.platform.startswith("win") and os.path.isabs(candidate):
        add_directory = getattr(os, "add_dll_directory", None)
        if callable(add_directory):
            with add_directory(str(Path(candidate).parent)):
                return ctypes.CDLL(candidate)
    return ctypes.CDLL(candidate)


def _format_version(version: Sequence[int]) -> str:
    return ".".join(str(int(component)) for component in version)


def _load_binding(
    library_path: os.PathLike[str] | str | None = None,
    *,
    cuda_major: int | None = None,
) -> tuple[_NvTiffCAPI, tuple[int, int, int], str, tuple[str, ...]]:
    attempts: list[str] = []
    try:
        candidates = _library_candidates(library_path, cuda_major=cuda_major)
    except Exception as exc:
        raise NvTiffUnavailableError(
            f"nvTIFF discovery failed safely: {type(exc).__name__}: {exc}"
        ) from exc
    for candidate in candidates:
        try:
            raw_library = _load_cdll(candidate)
            api = _NvTiffCAPI(raw_library)
            version = api.version()
            if version < MINIMUM_NVTIFF_VERSION:
                attempts.append(
                    f"{candidate}: version {_format_version(version)} is older than "
                    f"required {_format_version(MINIMUM_NVTIFF_VERSION)}"
                )
                continue
            return api, version, str(candidate), tuple(attempts)
        except Exception as exc:
            attempts.append(f"{candidate}: {exc}")

    requirement = _format_version(MINIMUM_NVTIFF_VERSION)
    detail = "; ".join(attempts) if attempts else "no library candidates were found"
    raise NvTiffUnavailableError(
        f"nvTIFF >= {requirement} is unavailable ({detail}). Install the matching "
        "nvidia-nvtiff-cu12 or nvidia-nvtiff-cu13 wheel, or set "
        f"{NVTIFF_LIBRARY_ENV} to the native library path.",
        attempts=attempts,
    )


def _coerce_pointer(value: object, *, name: str, allow_zero: bool) -> int:
    raw = getattr(value, "value", value)
    try:
        pointer = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer CUDA pointer") from exc
    minimum = 0 if allow_zero else 1
    if pointer < minimum or pointer > _POINTER_MAX:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} native pointer value")
    return pointer


def _coerce_dimension(value: object, *, name: str) -> int:
    try:
        dimension = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if dimension < 1 or dimension > _UINT32_MAX:
        raise ValueError(f"{name} must be in [1, {_UINT32_MAX}]")
    return dimension


def _coerce_device_id(value: object) -> int:
    try:
        device_id = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("device_id must be a non-negative integer") from exc
    if device_id < 0 or device_id > (1 << 31) - 1:
        raise ValueError("device_id must be a non-negative 32-bit integer")
    return device_id


def _coerce_cuda_major(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        cuda_major = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("cuda_major must be an integer such as 12 or 13") from exc
    if cuda_major < 1 or cuda_major > 99:
        raise ValueError("cuda_major must be in [1, 99]")
    return cuda_major


def _image_info(*, page_count: int, height: int, width: int) -> _NvTiffImageInfo:
    info = _NvTiffImageInfo()
    info.image_type = _NVTIFF_IMAGETYPE_PAGE if page_count > 1 else 0
    info.image_width = width
    info.image_height = height
    info.compression = _NVTIFF_COMPRESSION_LZW
    info.photometric_int = _NVTIFF_PHOTOMETRIC_MINISBLACK
    info.planar_config = _NVTIFF_PLANARCONFIG_CONTIG
    info.samples_per_pixel = 1
    info.bits_per_pixel = 8
    info.bits_per_sample[0] = 8
    info.sample_format[0] = _NVTIFF_SAMPLEFORMAT_UINT
    return info


def _temporary_output_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.nvtiff.tmp"
    )


class NvTiffBackend:
    """One-device, one-stream persistent nvTIFF encoder session.

    The caller must make ``device_id`` current before the first write and keep
    it current for every later call.  This binding intentionally avoids a CUDA
    runtime dependency, so it can validate identity but cannot call
    ``cudaSetDevice`` itself.  The bound stream must also remain valid through
    :meth:`close`.  Use the backend as a context manager or call ``close``
    explicitly.
    """

    def __init__(
        self,
        device_id: int,
        library_path: os.PathLike[str] | str | None = None,
        *,
        cuda_major: int | None = None,
        _library: object | None = None,
    ) -> None:
        self.device_id = _coerce_device_id(device_id)
        self.cuda_major = _coerce_cuda_major(cuda_major)
        if _library is None:
            api, version, resolved_path, attempts = _load_binding(
                library_path, cuda_major=self.cuda_major
            )
        else:
            api = _NvTiffCAPI(_library)
            version = api.version()
            if version < MINIMUM_NVTIFF_VERSION:
                raise NvTiffUnavailableError(
                    f"nvTIFF {_format_version(version)} is older than required "
                    f"{_format_version(MINIMUM_NVTIFF_VERSION)}"
                )
            resolved_path = os.fspath(library_path or "<injected nvTIFF library>")
            attempts = ()
        self._api = api
        self.version = version
        self.library_path = str(resolved_path)
        self.load_attempts = tuple(attempts)
        self._lock = threading.RLock()
        self._encoder = _EncoderHandle()
        self._bound_stream: int | None = None
        self._closed = False

    @property
    def diagnostic(self) -> str:
        return (
            f"nvTIFF {_format_version(self.version)} available at "
            f"{self.library_path}"
        )

    def __enter__(self) -> "NvTiffBackend":
        with self._lock:
            if self._closed:
                raise NvTiffError("nvTIFF backend is closed")
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"nvTIFF context cleanup also failed: {cleanup_error}")
        return False

    def close(self) -> None:
        """Destroy the persistent encoder; safe to call more than once."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._encoder.value:
                return
            encoder = self._encoder
            self._encoder = _EncoderHandle()
            stream = _CudaStream(int(self._bound_stream or 0))
            self._api.checked("nvtiffEncoderDestroy", encoder, stream)

    def _bind_stream(self, cuda_stream: int) -> None:
        if self._closed:
            raise NvTiffError("nvTIFF backend is closed")
        if self._bound_stream is None:
            self._bound_stream = int(cuda_stream)
        elif int(cuda_stream) != self._bound_stream:
            raise ValueError(
                "NvTiffBackend is bound to CUDA stream "
                f"{self._bound_stream}; create one backend per stream instead of "
                f"reusing it with stream {int(cuda_stream)}"
            )

    def _ensure_encoder(self, stream: _CudaStream) -> None:
        if self._encoder.value:
            return
        candidate = _EncoderHandle()
        try:
            self._api.checked(
                "nvtiffEncoderCreate",
                ctypes.byref(candidate),
                None,
                None,
                stream,
            )
            if not candidate.value:
                raise NvTiffError(
                    "nvtiffEncoderCreate succeeded without returning a handle"
                )
        except BaseException as primary_error:
            if candidate.value:
                try:
                    self._api.checked("nvtiffEncoderDestroy", candidate, stream)
                except BaseException as cleanup_error:
                    add_note = getattr(primary_error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "nvTIFF failed to destroy the handle returned by a failed "
                            f"encoder creation: {cleanup_error}"
                        )
            raise
        self._encoder = candidate

    def _discard_encoder(
        self,
        stream: _CudaStream,
        cleanup_errors: list[BaseException],
    ) -> None:
        if not self._encoder.value:
            return
        encoder = self._encoder
        self._encoder = _EncoderHandle()
        try:
            self._api.checked("nvtiffEncoderDestroy", encoder, stream)
        except BaseException as exc:
            cleanup_errors.append(exc)

    def write_multipage_lzw_from_device_pointers(
        self,
        path: os.PathLike[str] | str,
        device_pointers: Sequence[object],
        *,
        height: int,
        width: int,
        cuda_stream: object,
    ) -> Path:
        """Write one lossless TIFF whose pages are CUDA-resident gray8 images.

        ``device_pointers`` order becomes TIFF page order.  Every pointer must
        reference at least ``height * width`` tightly packed bytes on the CUDA
        device associated with ``cuda_stream``.

        This is an unsafe low-level escape hatch: without linking the CUDA
        runtime, ctypes cannot prove allocation, span, or device ownership.
        Prefer :meth:`write_multipage_lzw` when a tensor object is available.
        """

        pointers = tuple(
            _coerce_pointer(value, name=f"device_pointers[{index}]", allow_zero=False)
            for index, value in enumerate(device_pointers)
        )
        if not pointers:
            raise ValueError("device_pointers must contain at least one CUDA page")
        if len(pointers) > _UINT32_MAX:
            raise ValueError(f"page count must not exceed {_UINT32_MAX}")
        image_height = _coerce_dimension(height, name="height")
        image_width = _coerce_dimension(width, name="width")
        stream_value = _coerce_pointer(cuda_stream, name="cuda_stream", allow_zero=True)

        destination = Path(path)
        if not destination.name:
            raise ValueError("path must name a TIFF output file")
        if destination.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError("nvTIFF output path must end in .tif or .tiff")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = _temporary_output_path(destination)

        try:
            self._write_native(
                temporary_path,
                pointers,
                height=image_height,
                width=image_width,
                cuda_stream=stream_value,
            )
            try:
                size = int(temporary_path.stat().st_size)
                with temporary_path.open("rb") as handle:
                    header = handle.read(4)
            except OSError as exc:
                raise NvTiffError(
                    f"nvTIFF did not create a readable staged file: {temporary_path}"
                ) from exc
            if size <= 0 or header not in {
                b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+",
            }:
                raise NvTiffError(
                    f"nvTIFF produced an empty or invalid staged TIFF: "
                    f"path={temporary_path}, size={size}, header={header!r}"
                )
            os.replace(temporary_path, destination)
        except BaseException:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return destination

    def _write_native(
        self,
        path: Path,
        pointers: tuple[int, ...],
        *,
        height: int,
        width: int,
        cuda_stream: int,
    ) -> None:
        with self._lock:
            self._bind_stream(cuda_stream)
            api = self._api
            stream = _CudaStream(cuda_stream)
            params = _EncodeParamsHandle()
            cleanup_errors: list[BaseException] = []
            succeeded = False

            typed_pointers = (_Uint8DevicePointer * len(pointers))(
                *(
                    ctypes.cast(ctypes.c_void_p(pointer), _Uint8DevicePointer)
                    for pointer in pointers
                )
            )
            info = _image_info(page_count=len(pointers), height=height, width=width)

            try:
                self._ensure_encoder(stream)
                api.checked("nvtiffEncodeParamsCreate", ctypes.byref(params))
                if not params.value:
                    raise NvTiffError(
                        "nvtiffEncodeParamsCreate succeeded without returning a handle"
                    )

                raw_input_bytes = len(pointers) * height * width
                if raw_input_bytes >= _BIGTIFF_RAW_INPUT_THRESHOLD:
                    api.checked(
                        "nvtiffEncodeParamsSetTiffVariant",
                        params,
                        ctypes.c_int(_NVTIFF_BIG_TIFF),
                    )
                api.checked(
                    "nvtiffEncodeParamsSetImageInfo",
                    params,
                    ctypes.byref(info),
                )
                api.checked(
                    "nvtiffEncodeParamsSetInputs",
                    params,
                    typed_pointers,
                    ctypes.c_uint32(len(pointers)),
                )
                params_array = (_EncodeParamsHandle * 1)(params.value)
                api.checked(
                    "nvtiffEncode",
                    self._encoder,
                    params_array,
                    ctypes.c_uint32(1),
                    stream,
                )
                api.checked(
                    "nvtiffEncodeFinalize",
                    self._encoder,
                    params_array,
                    ctypes.c_uint32(1),
                    stream,
                )
                api.checked(
                    "nvtiffWriteTiffFile",
                    self._encoder,
                    params_array,
                    ctypes.c_uint32(1),
                    os.fsencode(path),
                    stream,
                )
                succeeded = True
            finally:
                active_error = sys.exc_info()[1]
                if params.value:
                    try:
                        api.checked("nvtiffEncodeParamsDestroy", params, stream)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                if active_error is not None or cleanup_errors or not succeeded:
                    self._discard_encoder(stream, cleanup_errors)
                if cleanup_errors:
                    cleanup_message = "; ".join(str(exc) for exc in cleanup_errors)
                    if active_error is not None:
                        add_note = getattr(active_error, "add_note", None)
                        if callable(add_note):
                            add_note(f"nvTIFF cleanup also failed: {cleanup_message}")
                    else:
                        raise NvTiffError(
                            f"nvTIFF cleanup failed: {cleanup_message}"
                        ) from cleanup_errors[0]

    def write_multipage_lzw(
        self,
        path: os.PathLike[str] | str,
        pages_nhw: object,
        *,
        cuda_stream: object | None = None,
    ) -> Path:
        """Validate a contiguous CUDA uint8 ``(N,H,W)`` tensor and write it."""

        detected_device = _cuda_pages_device_id(pages_nhw)
        if detected_device is not None and detected_device != self.device_id:
            raise ValueError(
                f"pages_nhw is on CUDA device {detected_device}, but this "
                f"NvTiffBackend is bound to device {self.device_id}"
            )
        producer_known, producer_stream = _cuda_pages_producer_stream(pages_nhw)
        if cuda_stream is None:
            if not producer_known:
                raise ValueError(
                    "cuda_stream is required when pages_nhw does not expose CUDA "
                    "Array Interface v3 producer-stream metadata"
                )
            resolved_stream = 0 if producer_stream is None else producer_stream
        else:
            resolved_stream = _coerce_pointer(
                cuda_stream, name="cuda_stream", allow_zero=True
            )
            if (
                producer_known
                and producer_stream is not None
                and resolved_stream != producer_stream
            ):
                raise ValueError(
                    "cuda_stream does not match the CUDA Array Interface v3 "
                    f"producer stream ({resolved_stream} != {producer_stream}); "
                    "encode on the producer stream so writes are ordered"
                )
        pointers, height, width = _cuda_pages_device_pointers(pages_nhw)
        return self.write_multipage_lzw_from_device_pointers(
            path,
            pointers,
            height=height,
            width=width,
            cuda_stream=resolved_stream,
        )


def _shape_tuple(pages: object) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in getattr(pages, "shape"))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TypeError("pages_nhw must expose an integer shape") from exc


def _dtype_is_uint8(dtype: object) -> bool:
    token = str(dtype).strip().lower()
    return token in {
        "uint8",
        "torch.uint8",
        "cupy.uint8",
        "|u1",
        "<u1",
        ">u1",
    }


def _cuda_array_interface(pages: object) -> dict[str, object] | None:
    cuda_interface = getattr(pages, "__cuda_array_interface__", None)
    if callable(cuda_interface):
        cuda_interface = cuda_interface()
    if cuda_interface is not None and not isinstance(cuda_interface, dict):
        raise TypeError("__cuda_array_interface__ must be a mapping")
    return cuda_interface


def _cuda_pages_producer_stream(
    pages: object,
) -> tuple[bool, int | None]:
    """Return ``(known, stream)``; known ``None`` means no sync is needed."""

    cuda_interface = _cuda_array_interface(pages)
    if cuda_interface is None:
        return (False, None)
    try:
        version = int(cuda_interface.get("version", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("CUDA Array Interface version must be an integer") from exc
    if version < 3 or "stream" not in cuda_interface:
        return (False, None)
    producer_stream = cuda_interface["stream"]
    if producer_stream is None:
        # CAI defines None as requiring no producer/consumer synchronization.
        return (True, None)
    stream = _coerce_pointer(
        producer_stream,
        name="__cuda_array_interface__ producer stream",
        allow_zero=True,
    )
    if stream == 0:
        raise ValueError(
            "CUDA Array Interface v3 forbids producer stream 0 because it is "
            "ambiguous; use None, 1, 2, or a real cudaStream_t handle"
        )
    return (True, stream)


def _cuda_pages_device_id(pages: object) -> int | None:
    device = getattr(pages, "device", None)
    device_type = str(getattr(device, "type", "")).strip().lower()
    if device_type == "cuda":
        index = getattr(device, "index", None)
        if index is not None:
            return _coerce_device_id(index)
    device_id = getattr(device, "id", None)
    if device_id is not None and (
        device_type == "cuda" or "cuda" in type(device).__module__.lower()
    ):
        return _coerce_device_id(device_id)
    device_token = str(device).strip().lower()
    if device_token.startswith("cuda:"):
        return _coerce_device_id(device_token.partition(":")[2])

    dlpack_device = getattr(pages, "__dlpack_device__", None)
    if callable(dlpack_device):
        result = dlpack_device()
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("__dlpack_device__() must return (device_type, device_id)")
        dlpack_type = int(result[0])
        if dlpack_type in {2, 13}:  # kDLCUDA, kDLCUDAManaged
            return _coerce_device_id(result[1])
    return None


def _cuda_pages_device_pointers(
    pages: object,
) -> tuple[tuple[int, ...], int, int]:
    shape = _shape_tuple(pages)
    if len(shape) != 3:
        raise ValueError(f"CUDA TIFF pages must have shape (N,H,W), got {shape}")
    page_count = _coerce_dimension(shape[0], name="page count")
    height = _coerce_dimension(shape[1], name="height")
    width = _coerce_dimension(shape[2], name="width")

    cuda_interface = _cuda_array_interface(pages)
    if cuda_interface is not None and "shape" in cuda_interface:
        try:
            interface_shape = tuple(
                int(value)
                for value in cuda_interface["shape"]  # type: ignore[union-attr]
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("CUDA Array Interface shape must contain integers") from exc
        if interface_shape != shape:
            raise ValueError(
                f"pages_nhw shape {shape} disagrees with CUDA Array Interface "
                f"shape {interface_shape}"
            )

    dtype = getattr(pages, "dtype", None)
    interface_typestr = cuda_interface.get("typestr") if cuda_interface else None
    if not (_dtype_is_uint8(dtype) or _dtype_is_uint8(interface_typestr)):
        raise TypeError(f"CUDA TIFF pages must use uint8, got dtype={dtype!r}")

    is_cuda = bool(getattr(pages, "is_cuda", False))
    device = getattr(pages, "device", None)
    is_cuda = is_cuda or str(getattr(device, "type", device)).lower().startswith("cuda")
    is_cuda = is_cuda or cuda_interface is not None
    if not is_cuda:
        raise ValueError("pages_nhw must reside in CUDA device memory")

    contiguous_method = getattr(pages, "is_contiguous", None)
    if callable(contiguous_method) and not bool(contiguous_method()):
        raise ValueError("pages_nhw must be C-contiguous")
    stride_method = getattr(pages, "stride", None)
    if callable(stride_method):
        strides = tuple(int(value) for value in stride_method())
        expected = (height * width, width, 1)
        if strides != expected:
            raise ValueError(
                f"pages_nhw must have contiguous element strides {expected}, got {strides}"
            )
    elif cuda_interface is not None:
        byte_strides = cuda_interface.get("strides")
        expected = (height * width, width, 1)
        if byte_strides is not None and tuple(int(value) for value in byte_strides) != expected:
            raise ValueError(
                f"pages_nhw must have contiguous byte strides {expected}, got {byte_strides}"
            )
    elif not callable(contiguous_method):
        raise TypeError("pages_nhw must expose is_contiguous(), stride(), or CUDA strides")

    pointer_method = getattr(pages, "data_ptr", None)
    if callable(pointer_method):
        base_pointer = _coerce_pointer(
            pointer_method(), name="pages_nhw.data_ptr()", allow_zero=False
        )
    elif cuda_interface is not None:
        data = cuda_interface.get("data")
        if not isinstance(data, (tuple, list)) or not data:
            raise TypeError("__cuda_array_interface__['data'] must contain a pointer")
        base_pointer = _coerce_pointer(
            data[0], name="__cuda_array_interface__ data pointer", allow_zero=False
        )
    else:
        raise TypeError("pages_nhw must expose data_ptr() or __cuda_array_interface__")

    page_bytes = height * width
    final_pointer = base_pointer + (page_count - 1) * page_bytes
    if final_pointer > _POINTER_MAX:
        raise ValueError("CUDA page pointers overflow the native pointer range")
    return (
        tuple(base_pointer + index * page_bytes for index in range(page_count)),
        height,
        width,
    )


def probe_nvtiff(
    library_path: os.PathLike[str] | str | None = None,
    *,
    cuda_major: int | None = None,
) -> NvTiffCapability:
    """Return nvTIFF availability without printing or raising expected load errors."""

    try:
        backend = NvTiffBackend(
            device_id=0,
            library_path=library_path,
            cuda_major=cuda_major,
        )
    except NvTiffUnavailableError as exc:
        return NvTiffCapability(
            available=False,
            version=None,
            library_path=os.fspath(library_path) if library_path is not None else None,
            diagnostic=str(exc),
            attempts=exc.attempts,
        )
    capability = NvTiffCapability(
        available=True,
        version=backend.version,
        library_path=backend.library_path,
        diagnostic=backend.diagnostic,
        attempts=backend.load_attempts,
    )
    backend.close()
    return capability


def write_multipage_lzw_from_device_pointers(
    path: os.PathLike[str] | str,
    device_pointers: Sequence[object],
    *,
    device_id: int,
    height: int,
    width: int,
    cuda_stream: object,
    cuda_major: int | None = None,
    backend: NvTiffBackend | None = None,
) -> Path:
    """Unsafe raw-pointer convenience wrapper.

    Pointer allocation, span, and CUDA-device validity cannot be proven without
    the CUDA runtime.  Prefer :func:`write_multipage_lzw` for tensor objects.
    The caller must already have ``device_id`` current.
    """

    resolved_device = _coerce_device_id(device_id)
    if backend is not None:
        if backend.device_id != resolved_device:
            raise ValueError(
                f"backend device {backend.device_id} does not match device_id "
                f"{resolved_device}"
            )
        return backend.write_multipage_lzw_from_device_pointers(
            path,
            device_pointers,
            height=height,
            width=width,
            cuda_stream=cuda_stream,
        )
    with NvTiffBackend(resolved_device, cuda_major=cuda_major) as active_backend:
        return active_backend.write_multipage_lzw_from_device_pointers(
            path,
            device_pointers,
            height=height,
            width=width,
            cuda_stream=cuda_stream,
        )


def write_multipage_lzw(
    path: os.PathLike[str] | str,
    pages_nhw: object,
    *,
    device_id: int | None = None,
    cuda_stream: object | None = None,
    cuda_major: int | None = None,
    backend: NvTiffBackend | None = None,
) -> Path:
    """Convenience wrapper for a contiguous CUDA uint8 ``(N,H,W)`` tensor.

    The tensor's CUDA device must already be current in this thread.  This
    module validates device identity but deliberately does not load the CUDA
    runtime to change or query the current context.
    """

    detected_device = _cuda_pages_device_id(pages_nhw)
    if device_id is not None:
        resolved_device = _coerce_device_id(device_id)
    elif detected_device is not None:
        resolved_device = detected_device
    elif backend is not None:
        resolved_device = backend.device_id
    else:
        resolved_device = None
    if resolved_device is None:
        raise ValueError(
            "device_id is required when pages_nhw does not expose CUDA device identity"
        )
    if detected_device is not None and detected_device != resolved_device:
        raise ValueError(
            f"pages_nhw is on CUDA device {detected_device}, not requested device "
            f"{resolved_device}"
        )
    if backend is not None:
        if backend.device_id != resolved_device:
            raise ValueError(
                f"backend device {backend.device_id} does not match pages device "
                f"{resolved_device}"
            )
        return backend.write_multipage_lzw(
            path,
            pages_nhw,
            cuda_stream=cuda_stream,
        )
    with NvTiffBackend(resolved_device, cuda_major=cuda_major) as active_backend:
        return active_backend.write_multipage_lzw(
            path,
            pages_nhw,
            cuda_stream=cuda_stream,
        )


__all__ = [
    "MINIMUM_NVTIFF_VERSION",
    "NVTIFF_LIBRARY_ENV",
    "NvTiffBackend",
    "NvTiffCallError",
    "NvTiffCapability",
    "NvTiffError",
    "NvTiffUnavailableError",
    "probe_nvtiff",
    "write_multipage_lzw",
    "write_multipage_lzw_from_device_pointers",
]
