"""Required and optional third-party runtime dependencies.

Only numerical/runtime modules import this module. Configuration and backend-control
contracts remain dependency-light so CLI discovery and orchestration do not initialize
native image-processing or accelerator libraries.
"""

from __future__ import annotations

from typing import Optional

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV (cv2) is required: pip install opencv-python") from exc

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("SciPy is required: pip install scipy") from exc

try:
    import tifffile  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tifffile is required: pip install tifffile") from exc

try:
    from tqdm import tqdm  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("tqdm is required: pip install tqdm") from exc

try:
    import numba as _numba  # type: ignore
except Exception as exc:  # pragma: no cover - optional acceleration
    _numba = None  # type: ignore[assignment]
    _NUMBA_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _NUMBA_IMPORT_ERROR = None


__all__ = (
    "cv2",
    "ndi",
    "tifffile",
    "tqdm",
    "_numba",
    "_NUMBA_IMPORT_ERROR",
)
