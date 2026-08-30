"""Canonical model-input batches and fan-out contracts.

Geometry sources create one :class:`RenderBatch` and hand the same frame objects to
both the model-facing loader and optional observers.  Backend-only layout, dtype, and
normalization conversions may happen after this boundary; geometric re-rendering may
not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .unification.contracts import RasterPlan, RenderItem


@dataclass(frozen=True)
class RenderBatchItem:
    """One slot in a canonical render batch."""

    result_index: int
    center_index: Optional[int]
    synthetic_padding: bool
    radial_padding: bool
    frame: np.ndarray
    request: Optional[RenderItem] = None


@dataclass(frozen=True)
class RenderBatch:
    """One geometry render shared verbatim by model and artifact sinks.

    ``frames`` is deliberately a mutable-list reference: ``model_payload`` returns that
    exact object, and every item's ``frame`` is the corresponding object in the list.
    ``device_tensor`` is reserved for backends whose canonical render remains resident
    on an accelerator; CPU sources leave it unset.
    """

    paths: List[str]
    frames: List[np.ndarray]
    info: List[str]
    items: Tuple[RenderBatchItem, ...]
    device_tensor: Optional[object] = None
    raster_plan: Optional[RasterPlan] = None

    def __post_init__(self) -> None:
        if not (len(self.paths) == len(self.frames) == len(self.info) == len(self.items)):
            raise ValueError(
                "canonical RenderBatch paths, frames, info, and items must have equal lengths"
            )
        for index, item in enumerate(self.items):
            if item.frame is not self.frames[index]:
                raise ValueError(
                    "RenderBatchItem.frame must be the exact model-bound frame object"
                )
            if item.request is not None:
                if self.raster_plan is None:
                    raise ValueError("logical render requests require a batch raster_plan")
                if item.request.plan.digest != self.raster_plan.digest:
                    raise ValueError("render request plan does not match batch raster_plan")

    def model_payload(self) -> Tuple[List[str], List[np.ndarray], List[str]]:
        """Return the original loader objects without copying or re-rendering."""
        return self.paths, self.frames, self.info


RenderBatchSink = Callable[[RenderBatch], None]
