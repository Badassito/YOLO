"""Production-geometry rendering seam for the v19 LTA prototype.

This CPU/reference path proves that canonical XTA view/TTA rendering and native
backprojection can surround a SAM mask without using YOLO.  CUDA tensor capture
can replace only :meth:`LtaRenderedView.render_frame_rgb` later while retaining
the same raster-plan and restoration contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from ._deps import cv2
from .assembly import project_view_volume_to_orthogonal_volume
from .config import resolve_channel_format
from .geometry import (
    AugJob,
    ViewInfo,
    build_aug_job_for_variant,
    physical_view_name,
    render_fullframe_frame_for_job,
)
from .unification.contracts import RasterPlan
from .unification.sampling import build_forward_raster_plan


def implicit_rgb(frame: object) -> np.ndarray:
    """Convert one canonical gray/RGB uint8 raster to contiguous RGB."""

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise ValueError(f"LTA canonical intensity frame must be uint8; got {array.dtype}")
    if array.ndim == 2:
        return np.ascontiguousarray(np.repeat(array[:, :, None], 3, axis=2))
    if array.ndim == 3 and int(array.shape[2]) == 3:
        return np.ascontiguousarray(array)
    raise ValueError(f"LTA canonical intensity frame must be HxW or HxWx3; got {array.shape}")


@dataclass(frozen=True)
class LtaRenderedView:
    """One runtime view/angle using XTA's authoritative affine renderer."""

    volume_u8: np.ndarray
    view: ViewInfo
    aug_job: AugJob
    raster_plan: RasterPlan
    source_shape_tyx: Tuple[int, int, int]

    def __post_init__(self) -> None:
        volume = np.asarray(self.volume_u8)
        if volume.dtype != np.uint8 or volume.ndim != 3:
            raise ValueError("LTA source volume must be uint8 with (t,Y,X) shape")
        if tuple(int(value) for value in volume.shape) != tuple(self.source_shape_tyx):
            raise ValueError("LTA source volume shape does not match source_shape_tyx")
        if not isinstance(self.view, ViewInfo):
            raise TypeError("view must be a ViewInfo")
        if not isinstance(self.aug_job, AugJob):
            raise TypeError("aug_job must be an AugJob")
        if not isinstance(self.raster_plan, RasterPlan) or self.raster_plan.mode.value != "lta":
            raise TypeError("raster_plan must be an LTA RasterPlan")

    @property
    def frame_count(self) -> int:
        return int(self.view.num_slices)

    @property
    def model_shape_hw(self) -> Tuple[int, int]:
        return tuple(int(value) for value in self.raster_plan.output_shape)

    def render_frame_rgb(self, frame_index: int) -> np.ndarray:
        if int(frame_index) < 0 or int(frame_index) >= self.frame_count:
            raise IndexError(frame_index)
        frame = render_fullframe_frame_for_job(
            volume_rgb=self.volume_u8,
            view=self.view,
            job=self.aug_job,
            frame_idx=int(frame_index),
        )
        rgb = implicit_rgb(frame)
        if tuple(int(value) for value in rgb.shape[:2]) != self.model_shape_hw:
            raise RuntimeError(
                f"canonical renderer returned {rgb.shape[:2]}, expected {self.model_shape_hw}"
            )
        return rgb

    def restore_model_masks_to_view(
        self,
        masks_by_frame: Mapping[int, object],
    ) -> np.ndarray:
        """Invert only the in-plane model affine into one view-native volume."""

        restored = np.zeros(
            (self.frame_count, int(self.view.src_h), int(self.view.src_w)),
            dtype=np.uint8,
        )
        expected_model_shape = self.model_shape_hw
        for raw_index, raw_mask in masks_by_frame.items():
            frame_index = int(raw_index)
            if frame_index < 0 or frame_index >= self.frame_count:
                raise IndexError(frame_index)
            mask = np.asarray(raw_mask)
            if mask.dtype != np.bool_ or tuple(int(value) for value in mask.shape) != expected_model_shape:
                raise ValueError(
                    f"SAM mask for frame {frame_index} must be bool {expected_model_shape}; "
                    f"got {mask.dtype}/{mask.shape}"
                )
            native = cv2.warpAffine(
                np.asarray(mask, dtype=np.uint8),
                np.asarray(self.aug_job.aff.M_out_to_src, dtype=np.float32),
                dsize=(int(self.view.src_w), int(self.view.src_h)),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            restored[frame_index] = np.asarray(native != 0, dtype=np.uint8)
        return restored

    def project_view_masks_to_native(
        self,
        view_masks: np.ndarray,
        *,
        out_path: Path,
        workers: int = 1,
        prefer_memory: bool = True,
    ) -> np.ndarray:
        """Backproject one settled runtime-view mask volume to source TYX."""

        masks = np.asarray(view_masks)
        expected = (self.frame_count, int(self.view.src_h), int(self.view.src_w))
        if masks.dtype != np.uint8 or tuple(int(value) for value in masks.shape) != expected:
            raise ValueError(f"view mask volume must be uint8 {expected}; got {masks.dtype}/{masks.shape}")
        projected = project_view_volume_to_orthogonal_volume(
            masks,
            self.view,
            Path(out_path),
            desc=f"LTA native projection {self.view.name}",
            workers=max(1, int(workers)),
            prefer_memory=bool(prefer_memory),
            out_shape_tyx=self.source_shape_tyx,
            allow_transverse_passthrough=False,
        )
        if not isinstance(projected, np.ndarray):
            raise RuntimeError("LTA reference projection unexpectedly returned sink-only output")
        result = np.asarray(projected, dtype=np.uint8)
        if tuple(int(value) for value in result.shape) != self.source_shape_tyx:
            raise RuntimeError(
                f"LTA native projection returned {result.shape}, expected {self.source_shape_tyx}"
            )
        return result


def build_lta_rendered_view(
    volume_u8: np.ndarray,
    view: ViewInfo,
    *,
    temp_dir: Path,
    output_size: int = 1008,
) -> LtaRenderedView:
    """Bind one expanded LTA runtime view to canonical geometry and RGB policy."""

    volume = np.asarray(volume_u8)
    if volume.dtype != np.uint8 or volume.ndim != 3:
        raise ValueError("LTA source volume must be uint8 with (t,Y,X) shape")
    job = build_aug_job_for_variant(view, int(output_size), Path(temp_dir))
    fmt = resolve_channel_format("RGB")
    raster_plan = build_forward_raster_plan(
        mode="lta",
        physical_view_id=physical_view_name(view),
        angle_deg=float(job.angle_deg),
        channel_token=fmt.token,
        channel_kind=fmt.kind,
        channel_count=fmt.channel_count,
        channel_stride=fmt.stride,
        channel_offsets=fmt.offsets,
        channel_direction="ascending",
        output_shape=(int(output_size), int(output_size)),
        metadata={
            "runtime_view_id": str(view.name),
            "runtime_job_id": str(job.aug_id),
            "runtime_kind": "fullframe_sam",
            "channel_policy": "implicit_rgb_v1",
            "source_shape_tyx": [int(value) for value in volume.shape],
        },
    )
    return LtaRenderedView(
        volume_u8=volume,
        view=view,
        aug_job=job,
        raster_plan=raster_plan,
        source_shape_tyx=tuple(int(value) for value in volume.shape),
    )


__all__ = (
    "LtaRenderedView",
    "build_lta_rendered_view",
    "implicit_rgb",
)
