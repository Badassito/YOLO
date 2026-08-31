"""PTA render-plan values and CPU geometry used by parent and worker processes.

This module is deliberately independent of :mod:`XTA.pta`.  Every class
serialized in a render-phase payload and every CPU rendering primitive used by
spawned workers has a canonical owner here.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

from . import geometry as shared_geometry
from .unification.contracts import RasterPlan
from .unification.sampling import forward_sampling_policy, require_forward_sampling


RADIAL_LANCZOS_A = 3


@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    tile_stride: int
    config_id: str


@dataclass(frozen=True)
class ChannelFormat:
    """One user-requested output channel format before direction expansion."""

    token: str
    kind: str  # gray, rgb, custom
    channel_count: int
    stride: int


@dataclass(frozen=True)
class ChannelVariant:
    """One physical output set, including custom forward/reverse ordering."""

    format_token: str
    kind: str  # gray, rgb, custom
    channel_count: int
    stride: int
    reverse: bool
    offsets: Tuple[int, ...]

    @property
    def tag_token(self) -> str:
        if self.kind == "custom" and self.reverse:
            return f"{self.format_token}_reverse"
        return self.format_token

    @property
    def order_name(self) -> str:
        return "reverse" if self.reverse else "forward"


DEFAULT_CHANNEL_VARIANT = ChannelVariant(
    format_token="gray",
    kind="gray",
    channel_count=1,
    stride=1,
    reverse=False,
    offsets=(0,),
)


@dataclass(frozen=True)
class ViewInfo:
    name: str
    display_name: str
    family: str  # transverse, sagittal, coronal, radial, tilted_transverse
    num_slices: int
    src_h: int
    src_w: int
    pad_mode: str  # clamp or pad
    full_t: int
    full_h: int
    full_w: int
    azimuths_deg: Tuple[float, ...] = ()
    diameter: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    roi_radius: float = 0.0
    tilt_angle_deg: float = 0.0
    tilt_direction: str = ""
    # Canonical v18 physical view.  Legacy-compatible scalar fields above are
    # presentation/planner adapters only; all built-in pixels come from this
    # shared TTA geometry object when present.
    shared_view: Optional[shared_geometry.ViewInfo] = None


@dataclass(frozen=True)
class AffineSpec:
    angle_deg: float
    src_w: int
    src_h: int
    canvas_w: int
    canvas_h: int
    out_w: int
    out_h: int
    pad_off_x: float
    pad_off_y: float
    M_src_to_canvas: np.ndarray
    M_canvas_to_src: np.ndarray
    M_src_to_out: np.ndarray
    M_out_to_src: np.ndarray
    # TTA's exact affine/canvas object.  ``native_output`` represents PTA's
    # mode-specific imgsz=0 default while retaining the shared physical view.
    shared_affine: Optional[shared_geometry.AffineSpec] = None
    native_output: bool = False


@dataclass(frozen=True)
class RadialSampler:
    angle_deg: float
    diameter: int
    lanczos_a: int
    x_idx: np.ndarray
    y_idx: np.ndarray
    x_w: np.ndarray
    y_w: np.ndarray
    nn_x: np.ndarray
    nn_y: np.ndarray


_RADIAL_CACHE: Dict[Tuple[int, int, int, float, int], RadialSampler] = {}
_RADIAL_CACHE_LOCK = threading.Lock()


def lanczos_offsets(a: int) -> np.ndarray:
    """Offsets for a floor-centered 2a-tap Lanczos kernel.

    For radial reslicing this returns six taps for Lanczos-3:
    [-2, -1, 0, 1, 2, 3].
    """
    radius = max(1, int(a))
    return np.arange(-(radius - 1), radius + 1, dtype=np.int32)


def lanczos_kernel(x: np.ndarray, a: int = RADIAL_LANCZOS_A) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.sinc(x) * np.sinc(x / float(a))
    out[np.abs(x) >= float(a)] = 0.0
    return out.astype(np.float32, copy=False)


def normalize_lanczos_weight_rows(weights: np.ndarray, *, axis_name: str) -> np.ndarray:
    """Normalize valid Lanczos taps so a constant field has exactly unit gain."""
    values64 = np.asarray(weights, dtype=np.float64)
    sums = np.sum(values64, axis=1, keepdims=True, dtype=np.float64)
    invalid = ~np.isfinite(sums) | (np.abs(sums) <= 1e-12)
    if np.any(invalid):
        rows = np.flatnonzero(invalid[:, 0])[:12].tolist()
        raise RuntimeError(f"Invalid radial Lanczos {axis_name}-weight sums at sample rows {rows}")
    return np.ascontiguousarray((values64 / sums).astype(np.float32))


def get_radial_sampler(view: ViewInfo, angle_deg: float) -> RadialSampler:
    lanczos_a = int(RADIAL_LANCZOS_A)
    key = (int(view.full_w), int(view.full_h), int(view.diameter), round(float(angle_deg), 6), lanczos_a)
    with _RADIAL_CACHE_LOCK:
        cached = _RADIAL_CACHE.get(key)
    if cached is not None:
        return cached
    diameter = int(view.diameter)
    coords = np.linspace(-float(view.roi_radius), float(view.roi_radius), diameter, dtype=np.float32)
    theta = math.radians(float(angle_deg))
    xs = np.asarray(float(view.center_x) + coords * math.cos(theta), dtype=np.float32)
    ys = np.asarray(float(view.center_y) + coords * math.sin(theta), dtype=np.float32)
    offsets = lanczos_offsets(lanczos_a)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x_idx_raw = x0[:, None] + offsets[None, :]
    y_idx_raw = y0[:, None] + offsets[None, :]
    x_w = lanczos_kernel(xs[:, None] - x_idx_raw, a=lanczos_a)
    y_w = lanczos_kernel(ys[:, None] - y_idx_raw, a=lanczos_a)
    x_valid = (x_idx_raw >= 0) & (x_idx_raw < int(view.full_w))
    y_valid = (y_idx_raw >= 0) & (y_idx_raw < int(view.full_h))
    x_w *= x_valid.astype(np.float32)
    y_w *= y_valid.astype(np.float32)
    # The finite Lanczos tap sum varies with subpixel phase, and validity
    # clipping changes it further at the ROI boundary.  Normalize after
    # clipping to prevent radial brightness gain/loss and endpoint ringing
    # from shifting the DC level.
    x_w = normalize_lanczos_weight_rows(x_w, axis_name="x")
    y_w = normalize_lanczos_weight_rows(y_w, axis_name="y")
    sampler = RadialSampler(
        float(angle_deg), diameter, lanczos_a,
        np.clip(x_idx_raw, 0, int(view.full_w) - 1).astype(np.int32),
        np.clip(y_idx_raw, 0, int(view.full_h) - 1).astype(np.int32),
        x_w.astype(np.float32), y_w.astype(np.float32),
        np.clip(np.rint(xs).astype(np.int32), 0, int(view.full_w) - 1),
        np.clip(np.rint(ys).astype(np.int32), 0, int(view.full_h) - 1),
    )
    with _RADIAL_CACHE_LOCK:
        existing = _RADIAL_CACHE.get(key)
        if existing is not None:
            return existing
        _RADIAL_CACHE[key] = sampler
        return sampler


def radial_extract_lanczos(volume: np.ndarray, sampler: RadialSampler, *, binary_mask: bool = False) -> np.ndarray:
    if binary_mask:
        # Categorical masks must not receive an oscillatory reconstruction
        # kernel.  The precomputed nearest coordinates preserve labels and are
        # substantially faster than Lanczos plus thresholding.
        sampled = np.asarray(volume)[:, sampler.nn_y, sampler.nn_x]
        return np.ascontiguousarray((sampled > 0).astype(np.uint8))
    t_dim = int(volume.shape[0])
    out_f = np.empty((t_dim, int(sampler.diameter)), dtype=np.float32)
    x_w = np.asarray(sampler.x_w, dtype=np.float32)[None, :, :]
    y_w = np.asarray(sampler.y_w, dtype=np.float32)
    kernel_taps = max(1, int(sampler.x_idx.shape[1]))
    bytes_per_frame = max(1, int(sampler.diameter) * kernel_taps * np.dtype(np.float32).itemsize)
    block_frames = max(1, min(256, (256 * 1024 * 1024) // bytes_per_frame))
    for start in range(0, t_dim, block_frames):
        stop = min(t_dim, start + block_frames)
        block = np.asarray(volume[start:stop])
        acc = np.zeros((stop - start, sampler.diameter), dtype=np.float32)
        for yi in range(sampler.y_idx.shape[1]):
            samples = block[:, sampler.y_idx[:, yi][:, None], sampler.x_idx].astype(np.float32, copy=False)
            row = np.sum(samples * x_w, axis=2)
            acc += row * y_w[:, yi][None, :]
        out_f[start:stop] = acc
    return np.ascontiguousarray(np.clip(np.rint(out_f), 0.0, 255.0).astype(np.uint8))


def get_native_view_image(volume: np.ndarray, view: ViewInfo, idx: int) -> np.ndarray:
    i = int(idx)
    if view.family == "transverse":
        return np.asarray(volume[i])
    if view.family == "sagittal":
        return np.ascontiguousarray(volume[:, i, :])
    if view.family == "coronal":
        return np.ascontiguousarray(volume[:, :, i])
    if view.family == "radial":
        sampler = get_radial_sampler(view, float(view.azimuths_deg[i]))
        return radial_extract_lanczos(volume, sampler, binary_mask=False)
    raise ValueError(f"Native image frame for {view.family} requires a dedicated renderer")


def get_native_view_frame(volume: np.ndarray, mask: np.ndarray, view: ViewInfo, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    i = int(idx)
    image = get_native_view_image(volume, view, i)
    if view.family == "transverse":
        return image, np.asarray(mask[i])
    if view.family == "sagittal":
        return image, np.ascontiguousarray(mask[:, i, :])
    if view.family == "coronal":
        return image, np.ascontiguousarray(mask[:, :, i])
    if view.family == "radial":
        sampler = get_radial_sampler(view, float(view.azimuths_deg[i]))
        return image, radial_extract_lanczos(mask, sampler, binary_mask=True)
    raise ValueError(f"Native frame for {view.family} requires a dedicated renderer")


@dataclass(frozen=True)
class TiltPlan:
    x_idx: np.ndarray
    y_idx: np.ndarray
    valid_xy: np.ndarray
    axis_offset: np.ndarray


_TILT_PLAN_CACHE: Dict[Tuple[str, int, int, int, int, Tuple[float, ...]], TiltPlan] = {}
_TILT_PLAN_CACHE_LOCK = threading.Lock()


def tilt_plan_key(view: ViewInfo, M_grid_to_src: np.ndarray, grid_h: int, grid_w: int) -> Tuple[str, int, int, int, int, Tuple[float, ...]]:
    mat = tuple(round(float(x), 6) for x in np.asarray(M_grid_to_src, dtype=np.float32).reshape(-1).tolist())
    # The expensive XY grid does not depend on tilt magnitude or sign; those
    # enter only through tan(angle) in render_tilted_on_grid().  Keying on the
    # view name retained one ~117 MiB plan per angle at 3072^2.
    return (
        str(view.tilt_direction),
        int(view.full_w),
        int(view.full_h),
        int(grid_h),
        int(grid_w),
        mat,
    )


def get_tilt_plan(view: ViewInfo, M_grid_to_src: np.ndarray, grid_h: int, grid_w: int) -> TiltPlan:
    key = tilt_plan_key(view, M_grid_to_src, grid_h, grid_w)
    # A plan can exceed 100 MiB.  Build it under a cache-miss lock so dozens of
    # frame threads cannot allocate the same coordinate grids simultaneously.
    with _TILT_PLAN_CACHE_LOCK:
        cached = _TILT_PLAN_CACHE.get(key)
        if cached is not None:
            return cached
        yy, xx = np.indices((int(grid_h), int(grid_w)), dtype=np.float32)
        M = np.asarray(M_grid_to_src, dtype=np.float32)
        src_x = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]
        src_y = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]
        x_nn = np.rint(src_x).astype(np.int32)
        y_nn = np.rint(src_y).astype(np.int32)
        valid = (x_nn >= 0) & (x_nn < int(view.full_w)) & (y_nn >= 0) & (y_nn < int(view.full_h))
        if view.tilt_direction == "vertical":
            axis_offset = src_y - float((view.full_h - 1) / 2.0)
        elif view.tilt_direction == "horizontal":
            axis_offset = src_x - float((view.full_w - 1) / 2.0)
        else:
            raise ValueError(f"Unsupported tilt direction: {view.tilt_direction}")
        plan = TiltPlan(
            np.clip(x_nn, 0, int(view.full_w) - 1).astype(np.int32),
            np.clip(y_nn, 0, int(view.full_h) - 1).astype(np.int32),
            valid.astype(bool),
            np.asarray(axis_offset, dtype=np.float32),
        )
        _TILT_PLAN_CACHE[key] = plan
        return plan


def render_tilted_on_grid(volume: np.ndarray, view: ViewInfo, frame_idx: int, M_grid_to_src: np.ndarray, grid_h: int, grid_w: int, *, binary_mask: bool, block_rows: int = 256) -> np.ndarray:
    plan = get_tilt_plan(view, M_grid_to_src, int(grid_h), int(grid_w))
    tan_alpha = float(math.tan(math.radians(float(view.tilt_angle_deg))))
    t_dim = int(volume.shape[0])
    out = np.zeros((int(grid_h), int(grid_w)), dtype=np.uint8)
    frame_center = float(frame_idx)
    for y0 in range(0, int(grid_h), int(block_rows)):
        y1 = min(int(grid_h), y0 + int(block_rows))
        valid_xy = plan.valid_xy[y0:y1]
        if not np.any(valid_xy):
            continue
        t_src = frame_center + tan_alpha * plan.axis_offset[y0:y1]
        valid = valid_xy & (t_src >= 0.0) & (t_src <= float(t_dim - 1))
        if not np.any(valid):
            continue
        t0 = np.floor(t_src).astype(np.int32)
        t1 = np.clip(t0 + 1, 0, t_dim - 1).astype(np.int32)
        t0 = np.clip(t0, 0, t_dim - 1).astype(np.int32)
        alpha = (t_src - t0).astype(np.float32)
        ys = plan.y_idx[y0:y1]
        xs = plan.x_idx[y0:y1]
        f0 = volume[t0, ys, xs].astype(np.float32)
        f1 = volume[t1, ys, xs].astype(np.float32)
        blend = ((1.0 - alpha) * f0) + (alpha * f1)
        block = out[y0:y1]
        vals = (blend >= 0.5).astype(np.uint8) if binary_mask else np.clip(np.rint(blend), 0.0, 255.0).astype(np.uint8)
        block[valid] = vals[valid]
    return out


def warp_pair(native_img: np.ndarray, native_mask: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.warpAffine(
        np.ascontiguousarray(native_img), M, dsize=(int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    msk = cv2.warpAffine(
        np.ascontiguousarray(native_mask), M, dsize=(int(out_w), int(out_h)),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.ascontiguousarray(img), np.ascontiguousarray((msk > 0).astype(np.uint8))


def warp_image(native_img: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    image = cv2.warpAffine(
        np.ascontiguousarray(native_img), M, dsize=(int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.ascontiguousarray(image, dtype=np.uint8)


def _shared_aug_job_for_affine(
    view: ViewInfo,
    aff: AffineSpec,
) -> shared_geometry.AugJob:
    if view.shared_view is None or aff.shared_affine is None:
        raise ValueError("shared affine rendering requires a canonical v18 view and affine")
    return shared_geometry.AugJob(
        aug_id="a0",
        angle_deg=float(aff.angle_deg),
        meta_path=Path(f"{view.name}_a0.meta.json"),
        aff=aff.shared_affine,
    )


def _shared_render_full_intensity(
    volume: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
    *,
    mirror_radial_u: bool = False,
) -> np.ndarray:
    if view.shared_view is None:
        raise ValueError("shared intensity rendering requires a canonical v18 view")
    return np.ascontiguousarray(
        shared_geometry.render_intensity_frame_on_grid(
            volume,
            view.shared_view,
            int(idx),
            M_src_to_out=aff.M_src_to_out,
            M_out_to_src=aff.M_out_to_src,
            output_height=int(aff.out_h),
            output_width=int(aff.out_w),
            mirror_radial_u=bool(mirror_radial_u),
        ),
        dtype=np.uint8,
    )


def _shared_render_canvas_intensity(
    volume: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
    *,
    mirror_radial_u: bool = False,
) -> np.ndarray:
    if view.shared_view is None or aff.shared_affine is None:
        raise ValueError("shared canvas rendering requires a canonical v18 view and affine")
    if shared_geometry.is_tilted_view(view.shared_view):
        return np.ascontiguousarray(
            shared_geometry.render_tilted_canvas_frame(
                volume, view.shared_view, int(idx), aff.shared_affine
            ),
            dtype=np.uint8,
        )
    native = np.ascontiguousarray(
        shared_geometry.get_view_frame_by_index(volume, view.shared_view, int(idx))
    )
    if bool(mirror_radial_u):
        if not shared_geometry.is_radial_view(view.shared_view):
            raise ValueError("radial-u mirroring requested for a non-radial view")
        native = np.ascontiguousarray(native[:, ::-1])
    return warp_image(
        native,
        aff.shared_affine.M_src_to_canvas,
        int(aff.shared_affine.canvas_w),
        int(aff.shared_affine.canvas_h),
    )


def _shared_render_full_mask(
    mask: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
    *,
    mirror_radial_u: bool = False,
) -> np.ndarray:
    if view.shared_view is None:
        raise ValueError("shared categorical rendering requires a canonical v18 view")
    return np.ascontiguousarray(
        shared_geometry.render_categorical_frame_on_grid(
            mask,
            view.shared_view,
            int(idx),
            M_src_to_out=aff.M_src_to_out,
            M_out_to_src=aff.M_out_to_src,
            output_height=int(aff.out_h),
            output_width=int(aff.out_w),
            mirror_radial_u=bool(mirror_radial_u),
        ),
        dtype=np.uint8,
    )


def _shared_render_canvas_mask(
    mask: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
) -> np.ndarray:
    if view.shared_view is None or aff.shared_affine is None:
        raise ValueError("shared categorical canvas requires a canonical v18 view and affine")
    if shared_geometry.is_tilted_view(view.shared_view):
        rendered = shared_geometry._render_tilted_array_on_grid(
            mask,
            view.shared_view,
            int(idx),
            aff.shared_affine.M_canvas_to_src,
            int(aff.shared_affine.canvas_h),
            int(aff.shared_affine.canvas_w),
            mask_mode=True,
        )
        return np.ascontiguousarray((np.asarray(rendered) > 0).astype(np.uint8))
    native = shared_geometry.get_categorical_view_frame_by_index(
        mask, view.shared_view, int(idx)
    )
    return warp_mask_only(
        native,
        aff.shared_affine.M_src_to_canvas,
        int(aff.shared_affine.canvas_w),
        int(aff.shared_affine.canvas_h),
    )


def render_image_full_and_optional_canvas(
    volume: np.ndarray,
    view: ViewInfo,
    idx: int,
    aff: AffineSpec,
    need_canvas: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Render only the image side of one source-view slice."""
    if view.shared_view is not None:
        image_full = _shared_render_full_intensity(volume, view, int(idx), aff)
        image_canvas = (
            _shared_render_canvas_intensity(volume, view, int(idx), aff)
            if bool(need_canvas)
            else None
        )
        return image_full, image_canvas
    shared_full_canvas = bool(
        need_canvas
        and int(aff.out_w) == int(aff.canvas_w)
        and int(aff.out_h) == int(aff.canvas_h)
        and np.array_equal(aff.M_out_to_src, aff.M_canvas_to_src)
    )
    if view.family == "tilted_transverse":
        image_full = render_tilted_on_grid(
            volume, view, int(idx), aff.M_out_to_src, aff.out_h, aff.out_w, binary_mask=False,
        )
        if shared_full_canvas:
            return image_full, image_full
        if need_canvas:
            image_canvas = render_tilted_on_grid(
                volume, view, int(idx), aff.M_canvas_to_src, aff.canvas_h, aff.canvas_w, binary_mask=False,
            )
            return image_full, image_canvas
        return image_full, None

    native_image = get_native_view_image(volume, view, int(idx))
    image_full = warp_image(native_image, aff.M_src_to_out, aff.out_w, aff.out_h)
    if shared_full_canvas:
        return image_full, image_full
    if need_canvas:
        return image_full, warp_image(native_image, aff.M_src_to_canvas, aff.canvas_w, aff.canvas_h)
    return image_full, None


def render_full_and_optional_canvas(volume: np.ndarray, mask: np.ndarray, view: ViewInfo, idx: int, aff: AffineSpec, need_canvas: bool) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if view.shared_view is not None:
        image_full = _shared_render_full_intensity(volume, view, int(idx), aff)
        mask_full = _shared_render_full_mask(mask, view, int(idx), aff)
        if need_canvas:
            return (
                image_full,
                mask_full,
                _shared_render_canvas_intensity(volume, view, int(idx), aff),
                _shared_render_canvas_mask(mask, view, int(idx), aff),
            )
        return image_full, mask_full, None, None
    shared_full_canvas = bool(
        need_canvas
        and int(aff.out_w) == int(aff.canvas_w)
        and int(aff.out_h) == int(aff.canvas_h)
        and np.array_equal(aff.M_out_to_src, aff.M_canvas_to_src)
    )
    if view.family == "tilted_transverse":
        img_full = render_tilted_on_grid(volume, view, idx, aff.M_out_to_src, aff.out_h, aff.out_w, binary_mask=False)
        mask_full = render_tilted_on_grid(mask, view, idx, aff.M_out_to_src, aff.out_h, aff.out_w, binary_mask=True)
        img_canvas: Optional[np.ndarray] = None
        mask_canvas: Optional[np.ndarray] = None
        if shared_full_canvas:
            img_canvas, mask_canvas = img_full, mask_full
        elif need_canvas:
            img_canvas = render_tilted_on_grid(volume, view, idx, aff.M_canvas_to_src, aff.canvas_h, aff.canvas_w, binary_mask=False)
            mask_canvas = render_tilted_on_grid(mask, view, idx, aff.M_canvas_to_src, aff.canvas_h, aff.canvas_w, binary_mask=True)
        return img_full, mask_full, img_canvas, mask_canvas

    img_native, mask_native = get_native_view_frame(volume, mask, view, idx)
    img_full, mask_full = warp_pair(img_native, mask_native, aff.M_src_to_out, aff.out_w, aff.out_h)
    img_canvas = mask_canvas = None
    if shared_full_canvas:
        img_canvas, mask_canvas = img_full, mask_full
    elif need_canvas:
        img_canvas, mask_canvas = warp_pair(img_native, mask_native, aff.M_src_to_canvas, aff.canvas_w, aff.canvas_h)
    return img_full, mask_full, img_canvas, mask_canvas


def resize_centered(frame: np.ndarray, out_w: int, out_h: int, interpolation: int) -> np.ndarray:
    if int(frame.shape[1]) == int(out_w) and int(frame.shape[0]) == int(out_h):
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(cv2.resize(
        np.asarray(frame),
        dsize=(int(out_w), int(out_h)),
        interpolation=int(interpolation),
    ))


def extract_padded_tile(frame: np.ndarray, x0: int, y0: int, tile_size: int) -> np.ndarray:
    arr = np.asarray(frame)
    if (
        int(x0) >= 0
        and int(y0) >= 0
        and int(x0) + int(tile_size) <= int(arr.shape[1])
        and int(y0) + int(tile_size) <= int(arr.shape[0])
    ):
        # cv2.resize accepts non-contiguous ROI views, avoiding a tile-sized
        # zero-fill and copy for every normal dense_tile_positions() result.
        return arr[int(y0):int(y0) + int(tile_size), int(x0):int(x0) + int(tile_size), ...]
    tile_shape = (int(tile_size), int(tile_size)) + tuple(int(x) for x in arr.shape[2:])
    tile = np.zeros(tile_shape, dtype=arr.dtype)
    sx0 = max(0, int(x0))
    sy0 = max(0, int(y0))
    sx1 = min(int(frame.shape[1]), int(x0) + int(tile_size))
    sy1 = min(int(frame.shape[0]), int(y0) + int(tile_size))
    if sx0 >= sx1 or sy0 >= sy1:
        return tile
    dx0 = sx0 - int(x0)
    dy0 = sy0 - int(y0)
    tile[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0), ...] = arr[sy0:sy1, sx0:sx1, ...]
    return tile


@dataclass(frozen=True)
class RenderTileItem:
    cfg: TileConfig
    x: int
    y: int
    tile_tag: str
    out_w: int
    out_h: int
    img_pattern: str
    lbl_pattern: str
    overlay_path: Optional[Path]
    label_enabled: bool
    channel_kind: str = "gray"
    shared_job: Optional[shared_geometry.DenseTileJob] = None
    canonical_plan: Optional[RasterPlan] = None
    publish_images: bool = True
    publish_labels: bool = True


@dataclass(frozen=True)
class RenderPlan:
    view: ViewInfo
    aff: AffineSpec
    tag: str
    img_pattern: str
    lbl_pattern: str
    overlay_path: Optional[Path]
    label_enabled: bool
    tile_layout: Tuple[RenderTileItem, ...]
    stats: Dict[str, object]
    channel_variant: ChannelVariant = DEFAULT_CHANNEL_VARIANT
    # None means every view frame is eligible.  A tuple is used for partial
    # volumes so unannotated centers and discontinuous C...S... centers never
    # enter classification, splitting, augmentation, or rendering.
    eligible_frame_indices: Optional[Tuple[int, ...]] = None
    # Native encoded image indices for a partial transverse sequence.  Custom
    # channels resolve offsets against these values rather than compact array
    # positions, preserving gaps and true-volume edge clamping.
    source_encoded_indices: Tuple[int, ...] = ()
    shared_aug_job: Optional[shared_geometry.AugJob] = None
    canonical_plan: Optional[RasterPlan] = None
    publish_images: bool = True
    publish_labels: bool = True


@dataclass
class RenderFrameSource:
    img_full: np.ndarray
    mask_full: np.ndarray
    img_canvas: Optional[np.ndarray]
    mask_canvas: Optional[np.ndarray]
    # Canonical direct-to-output tile rasters.  These prevent PTA from
    # reintroducing a canvas-crop-resize interpolation layer after TTA's
    # collapsed tile transform.
    tile_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def edge_clamped_view_index(view: ViewInfo, index: int) -> int:
    return max(0, min(int(view.num_slices) - 1, int(index)))


def encoded_channel_source_positions(
    encoded_indices: Sequence[int],
    center_position: int,
    offsets: Sequence[int],
) -> Tuple[Optional[Tuple[int, ...]], Tuple[int, ...]]:
    """Resolve custom-channel offsets against original encoded image indices.

    Requests below the minimum or above the maximum encoded image index are
    true-volume boundary requests and clamp to the nearest edge.  A requested
    index *inside* those bounds must exist exactly; otherwise the center crosses
    a discontinuity and is ineligible.
    """
    encoded = tuple(int(x) for x in encoded_indices)
    if not encoded:
        raise ValueError("encoded_channel_source_positions requires at least one encoded index")
    if any(encoded[i] >= encoded[i + 1] for i in range(len(encoded) - 1)):
        raise ValueError(f"Encoded image indices must be strictly increasing, got {encoded[:24]}")
    center_pos = int(center_position)
    if center_pos < 0 or center_pos >= len(encoded):
        raise IndexError(f"Center position {center_pos} is outside encoded index mapping of length {len(encoded)}")
    lower = int(encoded[0])
    upper = int(encoded[-1])
    center_encoded = int(encoded[center_pos])
    positions: List[int] = []
    missing: List[int] = []
    for offset in offsets:
        requested = center_encoded + int(offset)
        resolved = lower if requested < lower else (upper if requested > upper else requested)
        position = bisect_left(encoded, int(resolved))
        if position >= len(encoded) or int(encoded[position]) != int(resolved):
            missing.append(int(resolved))
        else:
            positions.append(int(position))
    if missing:
        return None, tuple(sorted(set(int(x) for x in missing)))
    return tuple(positions), ()


def _require_pta_canonical_plan(
    plan: RenderPlan,
    role: str,
    *,
    tile: Optional[RenderTileItem] = None,
    backend: str = "cpu",
) -> None:
    """Mechanically bind shared PTA work to one registered sampling backend."""

    if plan.view.shared_view is None:
        return
    canonical = tile.canonical_plan if tile is not None else plan.canonical_plan
    if canonical is None:
        raise RuntimeError(
            f"shared PTA geometry {plan.tag!r} has no canonical RasterPlan"
        )
    current_policy = forward_sampling_policy()
    if canonical.sampling_policy.digest != current_policy.digest:
        raise RuntimeError(
            f"PTA RasterPlan policy drift for {plan.tag!r}: "
            f"plan={canonical.sampling_policy.digest}, current={current_policy.digest}"
        )
    require_forward_sampling(str(backend), role)


def render_channel_formatted_images(
    *,
    volume: np.ndarray,
    plan: RenderPlan,
    idx: int,
    need_canvas: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Render one center slice N into the plan's requested channel layout."""
    _require_pta_canonical_plan(plan, "intensity")
    variant = plan.channel_variant
    center_idx = int(idx)

    if variant.kind in {"gray", "rgb"}:
        image_full, image_canvas = render_image_full_and_optional_canvas(
            volume,
            plan.view,
            center_idx,
            plan.aff,
            bool(need_canvas),
        )
        if variant.kind == "gray":
            return image_full, image_canvas
        rgb_full = np.ascontiguousarray(np.repeat(image_full[:, :, None], 3, axis=2))
        rgb_canvas = (
            np.ascontiguousarray(np.repeat(image_canvas[:, :, None], 3, axis=2))
            if image_canvas is not None
            else None
        )
        return rgb_full, rgb_canvas

    if plan.source_encoded_indices:
        source_positions, missing = encoded_channel_source_positions(
            plan.source_encoded_indices,
            center_idx,
            variant.offsets,
        )
        if source_positions is None:
            center_encoded = int(plan.source_encoded_indices[center_idx])
            raise RuntimeError(
                f"Ineligible discontinuous channel center reached rendering for {plan.tag}: "
                f"center encoded index={center_encoded}, missing required index/indices={list(missing)}"
            )
        source_addresses = tuple((int(position), False) for position in source_positions)
    else:
        if plan.view.shared_view is not None:
            source_addresses = tuple(
                shared_geometry.channel_view_slice_source(
                    plan.view.shared_view, center_idx + int(offset)
                )
                for offset in variant.offsets
            )
        else:
            source_addresses = tuple(
                (edge_clamped_view_index(plan.view, center_idx + int(offset)), False)
                for offset in variant.offsets
            )

    full_planes: List[np.ndarray] = []
    canvas_planes: List[np.ndarray] = []
    for source_idx, mirror_u in source_addresses:
        if plan.view.shared_view is not None:
            image_full = _shared_render_full_intensity(
                volume,
                plan.view,
                int(source_idx),
                plan.aff,
                mirror_radial_u=bool(mirror_u),
            )
            image_canvas = (
                _shared_render_canvas_intensity(
                    volume,
                    plan.view,
                    int(source_idx),
                    plan.aff,
                    mirror_radial_u=bool(mirror_u),
                )
                if need_canvas
                else None
            )
        else:
            image_full, image_canvas = render_image_full_and_optional_canvas(
                volume,
                plan.view,
                int(source_idx),
                plan.aff,
                bool(need_canvas),
            )
        full_planes.append(image_full)
        if need_canvas:
            if image_canvas is None:
                raise RuntimeError(f"Channel stack requested a missing image canvas for {plan.tag}")
            canvas_planes.append(image_canvas)
    stacked_full = np.ascontiguousarray(np.stack(full_planes, axis=2), dtype=np.uint8)
    stacked_canvas = (
        np.ascontiguousarray(np.stack(canvas_planes, axis=2), dtype=np.uint8)
        if need_canvas
        else None
    )
    return stacked_full, stacked_canvas


def render_shared_tile_images(
    *,
    volume: np.ndarray,
    plan: RenderPlan,
    tile: RenderTileItem,
    idx: int,
) -> np.ndarray:
    """Render one PTA tile through TTA's collapsed direct-to-output transform."""

    _require_pta_canonical_plan(plan, "intensity", tile=tile)
    if plan.view.shared_view is None or tile.shared_job is None:
        raise ValueError("shared tile rendering requires canonical view and tile jobs")
    variant = plan.channel_variant
    center_idx = int(idx)

    def _plane(source_idx: int, mirror_u: bool = False) -> np.ndarray:
        return np.ascontiguousarray(
            shared_geometry.render_dense_tile_frame_for_job(
                volume,
                plan.view.shared_view,
                tile.shared_job,
                int(source_idx),
                mirror_radial_u=bool(mirror_u),
            ),
            dtype=np.uint8,
        )

    if variant.kind == "gray":
        source_idx, mirror_u = shared_geometry.channel_view_slice_source(
            plan.view.shared_view, center_idx
        )
        return _plane(source_idx, mirror_u)
    if variant.kind == "rgb":
        source_idx, mirror_u = shared_geometry.channel_view_slice_source(
            plan.view.shared_view, center_idx
        )
        plane = _plane(source_idx, mirror_u)
        return np.ascontiguousarray(np.repeat(plane[:, :, None], 3, axis=2))

    if plan.source_encoded_indices:
        source_positions, missing = encoded_channel_source_positions(
            plan.source_encoded_indices,
            center_idx,
            variant.offsets,
        )
        if source_positions is None:
            center_encoded = int(plan.source_encoded_indices[center_idx])
            raise RuntimeError(
                f"Ineligible discontinuous tile-channel center reached rendering for {plan.tag}: "
                f"center encoded index={center_encoded}, missing required index/indices={list(missing)}"
            )
        addresses = tuple((int(position), False) for position in source_positions)
    else:
        addresses = tuple(
            shared_geometry.channel_view_slice_source(
                plan.view.shared_view, center_idx + int(offset)
            )
            for offset in variant.offsets
        )
    return np.ascontiguousarray(
        np.stack([_plane(source_idx, mirror_u) for source_idx, mirror_u in addresses], axis=2),
        dtype=np.uint8,
    )


def render_plan_frame_source(*, volume: np.ndarray, mask: np.ndarray, plan: RenderPlan, idx: int, need_canvas: Optional[bool] = None) -> RenderFrameSource:
    """Render one center slice and its requested channel context.

    Tasks that only feed the full-frame item pass
    ``need_canvas=False`` so no canvas is rendered for them.

    Image channels may come from neighboring view slices, but both
    full-frame and tile labels always use the center slice ``idx``.  Native
    partial sequences resolve custom-channel neighbors by encoded image index.
    """
    if need_canvas is None:
        need_canvas = bool(plan.tile_layout)
    shared_tiles = tuple(tile for tile in plan.tile_layout if tile.shared_job is not None)
    canvas_required = bool(need_canvas) and len(shared_tiles) != len(plan.tile_layout)
    img_full, img_canvas = render_channel_formatted_images(
        volume=volume,
        plan=plan,
        idx=int(idx),
        need_canvas=bool(canvas_required),
    )
    mask_full, mask_canvas = render_plan_frame_mask_source(
        mask=mask,
        plan=plan,
        idx=int(idx),
        need_canvas=bool(canvas_required),
    )
    tile_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tile in shared_tiles:
        assert plan.view.shared_view is not None and tile.shared_job is not None
        tile_image = render_shared_tile_images(
            volume=volume,
            plan=plan,
            tile=tile,
            idx=int(idx),
        )
        tile_mask = shared_geometry.render_categorical_dense_tile_for_job(
            mask,
            plan.view.shared_view,
            tile.shared_job,
            int(idx),
        )
        tile_arrays[str(tile.tile_tag)] = (
            np.ascontiguousarray(tile_image, dtype=np.uint8),
            np.ascontiguousarray((np.asarray(tile_mask) > 0).astype(np.uint8)),
        )
    return RenderFrameSource(
        img_full=img_full,
        mask_full=mask_full,
        img_canvas=img_canvas,
        mask_canvas=mask_canvas,
        tile_arrays=tile_arrays,
    )


def get_native_view_mask(mask: np.ndarray, view: ViewInfo, idx: int) -> np.ndarray:
    i = int(idx)
    if view.family == "transverse":
        return np.asarray(mask[i])
    if view.family == "sagittal":
        return np.ascontiguousarray(mask[:, i, :])
    if view.family == "coronal":
        return np.ascontiguousarray(mask[:, :, i])
    if view.family == "radial":
        sampler = get_radial_sampler(view, float(view.azimuths_deg[i]))
        return radial_extract_lanczos(mask, sampler, binary_mask=True)
    raise ValueError(f"Native mask frame for {view.family} requires a dedicated renderer")


def warp_mask_only(native_mask: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    msk = cv2.warpAffine(
        np.ascontiguousarray(native_mask), M, dsize=(int(out_w), int(out_h)),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.ascontiguousarray((msk > 0).astype(np.uint8))


def render_plan_frame_mask_source(*, mask: np.ndarray, plan: RenderPlan, idx: int, need_canvas: Optional[bool] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Render only the mask side of one view frame.

    Copy-0 foreground classification never needs image pixels, so
    the planning pass skips the image slice/warp entirely.
    """
    _require_pta_canonical_plan(plan, "categorical_ground_truth")
    view = plan.view
    aff = plan.aff
    if need_canvas is None:
        need_canvas = bool(plan.tile_layout)
    need_canvas = bool(need_canvas)
    if view.shared_view is not None:
        mask_full = _shared_render_full_mask(mask, view, int(idx), aff)
        return (
            mask_full,
            _shared_render_canvas_mask(mask, view, int(idx), aff)
            if need_canvas
            else None,
        )
    shared_full_canvas = bool(
        need_canvas
        and int(aff.out_w) == int(aff.canvas_w)
        and int(aff.out_h) == int(aff.canvas_h)
        and np.array_equal(aff.M_out_to_src, aff.M_canvas_to_src)
    )
    if view.family == "tilted_transverse":
        mask_full = render_tilted_on_grid(mask, view, int(idx), aff.M_out_to_src, aff.out_h, aff.out_w, binary_mask=True)
        if shared_full_canvas:
            return mask_full, mask_full
        if need_canvas:
            return mask_full, render_tilted_on_grid(mask, view, int(idx), aff.M_canvas_to_src, aff.canvas_h, aff.canvas_w, binary_mask=True)
        return mask_full, None
    native = get_native_view_mask(mask, view, int(idx))
    mask_full = warp_mask_only(native, aff.M_src_to_out, aff.out_w, aff.out_h)
    if shared_full_canvas:
        return mask_full, mask_full
    if need_canvas:
        return mask_full, warp_mask_only(native, aff.M_src_to_canvas, aff.canvas_w, aff.canvas_h)
    return mask_full, None


__all__ = [
    "AffineSpec",
    "ChannelFormat",
    "ChannelVariant",
    "DEFAULT_CHANNEL_VARIANT",
    "RenderFrameSource",
    "RenderPlan",
    "RenderTileItem",
    "TileConfig",
    "ViewInfo",
    "edge_clamped_view_index",
    "encoded_channel_source_positions",
    "extract_padded_tile",
    "render_channel_formatted_images",
    "render_image_full_and_optional_canvas",
    "render_plan_frame_mask_source",
    "render_plan_frame_source",
    "render_shared_tile_images",
    "resize_centered",
]
