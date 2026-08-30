"""Shared binary-volume Gaussian smoothing semantics.

This module deliberately contains only the numerical operation shared by PTA
preprocessing and TTA postprocessing.  Dataset eligibility, foreground-anchor
repair, statistics, temporary storage, and output publication remain with their
respective callers.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


GAUSSIAN_BOUNDARY_MODE = "constant"
GAUSSIAN_BOUNDARY_VALUE = 0.0
GAUSSIAN_TRUNCATE = 4.0
GAUSSIAN_BINARY_THRESHOLD = 0.5

SliceObserver = Callable[[int, Any, Any], None]
SliceRunner = Callable[[str, int, Callable[[int], None]], None]


def binary_gaussian_pass(
    source: Any,
    *,
    sigma: float,
    gaussian_filter: Callable[..., Any],
    array_module: Any = np,
    float_workspace: Optional[Any] = None,
    destination: Optional[Any] = None,
    slice_runner: Optional[SliceRunner] = None,
    observe_slice: Optional[SliceObserver] = None,
) -> Any:
    """Apply one canonical Gaussian pass to a binary 3-D volume.

    The source is converted to float32, filtered isotropically with a
    constant-zero boundary and a four-sigma kernel truncation, then thresholded
    at 0.5.  The returned/destination volume is uint8 binary data and is suitable
    as the source of the next pass.

    ``float_workspace`` and ``slice_runner`` let TTA retain its bounded-memory,
    parallel copy/commit orchestration.  ``array_module`` lets its chunked CUDA
    path use the identical operation with CuPy.  These hooks do not alter the
    numerical contract.
    """
    sigma_f = float(sigma)
    if not np.isfinite(sigma_f) or sigma_f <= 0.0:
        raise ValueError(f"Gaussian sigma must be finite and > 0, got {sigma!r}")

    shape = tuple(int(value) for value in source.shape)
    if len(shape) != 3:
        raise ValueError(f"Binary Gaussian smoothing expects a 3-D volume, got shape {shape}")

    xp = array_module
    if float_workspace is None:
        filter_input = xp.asarray(source, dtype=xp.float32)
        filtered = gaussian_filter(
            input=filter_input,
            sigma=sigma_f,
            mode=GAUSSIAN_BOUNDARY_MODE,
            cval=GAUSSIAN_BOUNDARY_VALUE,
            truncate=GAUSSIAN_TRUNCATE,
        )
    else:
        workspace_shape = tuple(int(value) for value in float_workspace.shape)
        if workspace_shape != shape:
            raise ValueError(
                "Gaussian float workspace shape does not match source: "
                f"source={shape}, workspace={workspace_shape}"
            )

        def _copy_slice(z: int) -> None:
            float_workspace[int(z), :, :] = xp.asarray(source[int(z)], dtype=xp.float32)

        if slice_runner is None:
            float_workspace[...] = xp.asarray(source, dtype=xp.float32)
        else:
            slice_runner("copy", int(shape[0]), _copy_slice)

        gaussian_filter(
            input=float_workspace,
            sigma=sigma_f,
            output=float_workspace,
            mode=GAUSSIAN_BOUNDARY_MODE,
            cval=GAUSSIAN_BOUNDARY_VALUE,
            truncate=GAUSSIAN_TRUNCATE,
        )
        filtered = float_workspace

    if destination is None:
        return xp.asarray(filtered >= GAUSSIAN_BINARY_THRESHOLD, dtype=xp.uint8)

    destination_shape = tuple(int(value) for value in destination.shape)
    if destination_shape != shape:
        raise ValueError(
            "Gaussian destination shape does not match source: "
            f"source={shape}, destination={destination_shape}"
        )

    if slice_runner is None and observe_slice is None:
        destination[...] = xp.asarray(filtered >= GAUSSIAN_BINARY_THRESHOLD, dtype=xp.uint8)
        return destination

    def _commit_slice(z: int) -> None:
        old_binary = xp.asarray(source[int(z)], dtype=bool)
        new_binary = xp.asarray(
            filtered[int(z)] >= GAUSSIAN_BINARY_THRESHOLD,
            dtype=bool,
        )
        if observe_slice is not None:
            observe_slice(int(z), old_binary, new_binary)
        destination[int(z), :, :] = xp.asarray(new_binary, dtype=xp.uint8)

    if slice_runner is None:
        for z in range(int(shape[0])):
            _commit_slice(int(z))
    else:
        slice_runner("threshold", int(shape[0]), _commit_slice)
    return destination


__all__ = (
    "GAUSSIAN_BINARY_THRESHOLD",
    "GAUSSIAN_BOUNDARY_MODE",
    "GAUSSIAN_BOUNDARY_VALUE",
    "GAUSSIAN_TRUNCATE",
    "binary_gaussian_pass",
)
