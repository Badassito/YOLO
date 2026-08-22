"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np
from ._deps import cv2

from .config import (
    ChannelFormat,
    DEFAULT_CHANNEL_FORMAT,
    GIB,
)

@dataclass(frozen=True)
class AffineSpec:
    view: str
    angle_deg: float
    src_w: int
    src_h: int
    out_size: int
    canvas_w: int
    canvas_h: int
    pad_size: int
    pad_off_x: float
    pad_off_y: float
    M_out_to_src: np.ndarray      # 2x3 float32
    M_src_to_out: np.ndarray      # 2x3 float32
    M_canvas_to_src: np.ndarray   # 2x3 float32, pre-scale augmented canvas -> native view
    M_src_to_canvas: np.ndarray   # 2x3 float32, native view -> pre-scale augmented canvas

@dataclass(frozen=True)
class AugJob:
    aug_id: str
    angle_deg: float
    meta_path: Path
    aff: AffineSpec

def _expanded_pad_size(w: int, h: int, angle_deg: float) -> int:
    """Square canvas size P to fit a WxH rectangle rotated by angle_deg."""
    theta = math.radians(angle_deg % 360.0)
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    w_rot = w * c + h * s
    h_rot = w * s + h * c
    P = int(math.ceil(max(w_rot, h_rot)))
    return max(P, max(w, h))

def _affine2x3_to_3x3(M: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float64)
    out[:2, :3] = np.asarray(M, dtype=np.float64)
    return out

def _format_angle_aug_id(angle_deg: float) -> str:
    token = f'{float(angle_deg):g}'.replace('-', 'm').replace('.', 'p')
    return f'a{token}'

def build_affine(
    view: str,
    src_w: int,
    src_h: int,
    out_size: int,
    angle_deg: float,
    pad_mode: str,
) -> AffineSpec:
    """Build one source-to-model affine for padding, rotation, and scaling.
    
    Cartesian and Tilted views clamp to their source canvas; Radial views expand onto a black-padded canvas."""
    if pad_mode not in ("clamp", "pad"):
        raise ValueError("pad_mode must be 'clamp' or 'pad'")

    if pad_mode == "pad":
        canvas_w = canvas_h = _expanded_pad_size(src_w, src_h, angle_deg)
    else:
        canvas_w = src_w
        canvas_h = src_h

    off_x = (canvas_w - src_w) / 2.0
    off_y = (canvas_h - src_h) / 2.0

    cx_canvas = (canvas_w - 1) / 2.0
    cy_canvas = (canvas_h - 1) / 2.0
    cx_out = (out_size - 1) / 2.0
    cy_out = (out_size - 1) / 2.0

    M_pad = np.array(
        [
            [1.0, 0.0, off_x],
            [0.0, 1.0, off_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    M_rot = _affine2x3_to_3x3(cv2.getRotationMatrix2D((cx_canvas, cy_canvas), angle_deg, 1.0))

    sx = out_size / float(canvas_w)
    sy = out_size / float(canvas_h)
    M_scale = np.array(
        [
            [sx, 0.0, cx_out - sx * cx_canvas],
            [0.0, sy, cy_out - sy * cy_canvas],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    M_src_to_canvas3 = M_rot @ M_pad
    M_canvas_to_src3 = np.linalg.inv(M_src_to_canvas3)
    M_src_to_out3 = M_scale @ M_src_to_canvas3
    M_out_to_src3 = np.linalg.inv(M_src_to_out3)

    return AffineSpec(
        view=view,
        angle_deg=float(angle_deg),
        src_w=src_w,
        src_h=src_h,
        out_size=out_size,
        canvas_w=int(canvas_w),
        canvas_h=int(canvas_h),
        pad_size=int(max(canvas_w, canvas_h)),
        pad_off_x=float(off_x),
        pad_off_y=float(off_y),
        M_out_to_src=M_out_to_src3[:2, :3].astype(np.float32),
        M_src_to_out=M_src_to_out3[:2, :3].astype(np.float32),
        M_canvas_to_src=M_canvas_to_src3[:2, :3].astype(np.float32),
        M_src_to_canvas=M_src_to_canvas3[:2, :3].astype(np.float32),
    )

def delayed_native_expansion_enabled() -> bool:
    """Keep capable view masks at inference resolution until final projection.

 ``YOLO_TTA_DELAY_NATIVE_EXPANSION=0`` restores the behavior in which every
 prediction is expanded to ``view.src_h x view.src_w`` before accumulation."""
    return _env_flag('YOLO_TTA_DELAY_NATIVE_EXPANSION', True)

def view_uses_inference_processing_grid(view: 'ViewInfo', out_size: int) -> bool:
    size = int(out_size)
    if not delayed_native_expansion_enabled() or size <= 0:
        return False
    if size * size >= int(view.src_h) * int(view.src_w):
        return False
    if is_radial_view(view) and (size > int(view.src_h) or size > int(view.src_w)):
        # Radial terminal projection consumes compact monotone row/diameter lookups.  Do
        # not turn either axis into an expansion, which could introduce unmapped processing
        # rows between adjacent native coordinates and invalidate bounded row-range staging.
        return False
    return True

def view_processing_plane_shape(view: 'ViewInfo', out_size: int) -> Tuple[int, int]:
    if view_uses_inference_processing_grid(view, int(out_size)):
        return int(out_size), int(out_size)
    return int(view.src_h), int(view.src_w)

def view_processing_volume_shape(view: 'ViewInfo', out_size: int) -> Tuple[int, int, int]:
    ph, pw = view_processing_plane_shape(view, int(out_size))
    return int(view.num_slices), int(ph), int(pw)

def output_to_view_processing_affine(
    view: 'ViewInfo',
    M_out_to_native: np.ndarray,
    out_size: int,
) -> np.ndarray:
    """Map one YOLO output raster into the view's accumulation grid.

    Delayed views compose output-to-native with native-to-canonical; angle zero reduces
    to identity within tolerance. Radial interpolation remains in this canonical raster,
    and terminal backprojection maps native radial row/diameter coordinates into it.
    """
    M_native = np.asarray(M_out_to_native, dtype=np.float64).reshape(2, 3)
    if not view_uses_inference_processing_grid(view, int(out_size)):
        return M_native.astype(np.float32, copy=False)
    canonical = build_affine(
        view=str(view.name),
        src_w=int(view.src_w),
        src_h=int(view.src_h),
        out_size=int(out_size),
        angle_deg=0.0,
        pad_mode=str(view.pad_mode),
    )
    composed = _affine2x3_to_3x3(canonical.M_src_to_out) @ _affine2x3_to_3x3(M_native)
    out = composed[:2, :3].astype(np.float32)
    # Avoid tiny inversion noise defeating the identity resident-ring specialization.
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if np.allclose(out, identity, rtol=0.0, atol=2e-5):
        return identity
    return out

def tile_crop_border_pixels() -> int:
    """Safety ring added around every computed dense-tile crop window."""
    return max(0, _env_int('YOLO_TTA_TILE_CROP_BORDER', 2))

def tile_parent_crop_window(
    view: 'ViewInfo',
    M_out_to_src: np.ndarray,
    out_size: int,
    *,
    border: Optional[int] = None,
    force_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """Return ``((py0, py1, px0, px1), M_out_to_crop)`` for one dense-tile output raster.

 A dense tile's ``M_out_to_src`` is frame-invariant, so the region of the parent
 processing grid the tile can ever write is a fixed rectangle: the axis-aligned hull of
 the four output-raster corners pushed through ``output_to_view_processing_affine``.
 Everything downstream of the warp (device union, D2H, cleanup, hole fill, staging OR)
 only has to cover that rectangle -- outside it the warp writes zeros by construction
 (``BORDER_CONSTANT`` / ``padding_mode='zeros'``).

 ``M_out_to_crop`` is the same affine expressed in crop-local pixel coordinates, so
 ``cv2.warpAffine(plane, M_out_to_crop, dsize=(px1 - px0, py1 - py0))`` produces exactly
 the sub-rectangle ``warp_to_full_grid[py0:py1, px0:px1]`` would have produced.

 ``force_shape`` pins the window to a common (h, w) -- tiles clamped at a canvas edge
 would otherwise yield slightly smaller rectangles, and a uniform shape lets every tile
 of one configuration share a single result volume. The window is slid (never shrunk)
 to stay inside the grid, so the tile's true footprint is always covered."""
    ph, pw = view_processing_plane_shape(view, int(out_size))
    M_proc = np.asarray(
        output_to_view_processing_affine(view, np.asarray(M_out_to_src, dtype=np.float32), int(out_size)),
        dtype=np.float64,
    ).reshape(2, 3)

    pad = int(tile_crop_border_pixels() if border is None else border)
    n = float(int(out_size))
    # Output-raster corners in cv2 pixel-center convention, as homogeneous (x, y, 1) columns.
    corners = np.array(
        [
            [-0.5, n - 0.5, -0.5, n - 0.5],
            [-0.5, -0.5, n - 0.5, n - 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    dst = M_proc @ corners  # (2, 4) as (x, y) in parent-processing pixels
    px0 = int(math.floor(float(np.min(dst[0])))) - pad
    px1 = int(math.ceil(float(np.max(dst[0])))) + 1 + pad
    py0 = int(math.floor(float(np.min(dst[1])))) - pad
    py1 = int(math.ceil(float(np.max(dst[1])))) + 1 + pad

    if force_shape is not None:
        want_h = max(1, min(int(ph), int(force_shape[0])))
        want_w = max(1, min(int(pw), int(force_shape[1])))
    else:
        want_h = max(1, min(int(ph), int(py1) - int(py0)))
        want_w = max(1, min(int(pw), int(px1) - int(px0)))

    # Slide the window inside the grid without shrinking it, keeping the footprint covered.
    py0 = max(0, min(int(py0), int(ph) - int(want_h)))
    px0 = max(0, min(int(px0), int(pw) - int(want_w)))
    py1 = int(py0) + int(want_h)
    px1 = int(px0) + int(want_w)

    M_crop = M_proc.copy()
    M_crop[0, 2] -= float(px0)
    M_crop[1, 2] -= float(py0)
    return (int(py0), int(py1), int(px0), int(px1)), M_crop.astype(np.float32)

def tile_jobs_uniform_crop_shape(
    view: 'ViewInfo',
    tile_jobs: 'Sequence[DenseTileJob]',
    out_size: int,
) -> Tuple[int, int]:
    """Largest crop window across one tile configuration, so all tiles share a result shape."""
    ph, pw = view_processing_plane_shape(view, int(out_size))
    best_h = 1
    best_w = 1
    for job in tile_jobs:
        (y0, y1, x0, x1), _ = tile_parent_crop_window(view, job.M_out_to_src, int(out_size))
        best_h = max(int(best_h), int(y1) - int(y0))
        best_w = max(int(best_w), int(x1) - int(x0))
    return int(min(int(ph), best_h)), int(min(int(pw), best_w))

def view_processing_min_radius(
    view: 'ViewInfo',
    min_radius: float,
    plane_shape: Sequence[int],
) -> float:
    """Convert a native-view radius threshold to the current processing raster.

 The canonical square can scale the two native axes differently. A native circular
 component becomes an ellipse whose EDT radius is governed by the smaller axis scale,
 so the minimum scale preserves the former keep/drop boundary most closely."""
    radius = float(min_radius)
    if radius <= 0.0:
        return max(0.0, radius)
    try:
        ph, pw = int(plane_shape[-2]), int(plane_shape[-1])
    except Exception:
        return max(0.0, radius)
    if ph == int(view.src_h) and pw == int(view.src_w):
        return radius
    sy = float(ph) / float(max(1, int(view.src_h)))
    sx = float(pw) / float(max(1, int(view.src_w)))
    return radius * min(float(sy), float(sx))

def view_processing_search_angle(
    view: 'ViewInfo',
    angle_deg: float,
    plane_shape: Sequence[int],
) -> float:
    """Approximate the same interpolation search cone on a reduced in-plane raster."""
    angle = float(angle_deg)
    if angle == 0.0:
        return angle
    try:
        ph, pw = int(plane_shape[-2]), int(plane_shape[-1])
    except Exception:
        return angle
    if ph == int(view.src_h) and pw == int(view.src_w):
        return angle
    scale = min(
        float(ph) / float(max(1, int(view.src_h))),
        float(pw) / float(max(1, int(view.src_w))),
    )
    return math.degrees(math.atan(math.tan(math.radians(angle)) * float(scale)))

TILTED_VIEW_FAMILY = 'tilted'

RADIAL_VIEW_FAMILY = 'radial'

@dataclass(frozen=True)
class ViewInfo:
    name: str
    num_slices: int
    src_h: int
    src_w: int
    pad_mode: str  # 'clamp' or 'pad'
    family: str = 'orthogonal'
    summary_family: str = 'transverse'
    display_name: str = 'Transverse'
    azimuths_deg: Tuple[float, ...] = ()
    diameter: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    roi_radius: float = 0.0
    full_t: int = 0
    full_h: int = 0
    full_w: int = 0
    tilt_angle_deg: float = 0.0
    tilt_direction: str = ''
    tilt_frame_start: int = 0
    tilt_frame_stop: int = 0
    tilt_base_view: str = ''
    horizontal_axis: str = ''
    vertical_axis: str = ''
    stack_axis: str = ''
    # Radial orientation/source metadata. ``radial_base_view`` names the
    # Cartesian coordinate system whose in-plane circle is transformed. A tilted
    # Radial view carries the selected concrete Tilted variant in the remaining fields.
    radial_base_view: str = ''
    radial_tilted_source: bool = False
    radial_source_view_name: str = ''
    radial_request_token: str = ''
    # v16.4.0 TTA identity. ``name`` is the unique runtime variant name;
    # ``physical_view_name`` retains the underlying projection geometry name.
    physical_view_name: str = ''
    tta_aug_id: str = ''
    tta_angle_deg: float = 0.0

def physical_view_name(view: ViewInfo) -> str:
    return str(view.physical_view_name or view.name)

def is_tta_view_variant(view: ViewInfo) -> bool:
    return bool(str(view.tta_aug_id))

def tta_view_variant_name(view_name: str, aug_id: str) -> str:
    return f'{str(view_name)}__tta_{_sanitize_filesystem_token(aug_id)}'

def expand_views_into_tta_variants(
    views: Sequence[ViewInfo],
    angles: Sequence[float],
) -> List[ViewInfo]:
    """Create one independent runtime ``ViewInfo`` per physical-view/TTA-angle pair."""
    variants: List[ViewInfo] = []
    seen_names: set[str] = set()
    for physical in views:
        base_name = physical_view_name(physical)
        for angle in angles:
            aug_id = _format_angle_aug_id(float(angle))
            variant_name = tta_view_variant_name(base_name, aug_id)
            if variant_name in seen_names:
                raise ValueError(f'duplicate TTA view variant name: {variant_name}')
            seen_names.add(variant_name)
            variants.append(dataclasses_replace(
                physical,
                name=str(variant_name),
                summary_family=(
                    f'{str(physical.summary_family)}__tta_{_sanitize_filesystem_token(aug_id)}'
                ),
                display_name=f'{str(physical.display_name)} / TTA {float(angle):g}°',
                physical_view_name=str(base_name),
                tta_aug_id=str(aug_id),
                tta_angle_deg=float(angle),
            ))
    return variants

def is_tilted_view(view: ViewInfo) -> bool:
    """Return True for a concrete member of the Tilted view family."""
    return str(view.family) == TILTED_VIEW_FAMILY

def is_radial_view(view: ViewInfo) -> bool:
    return str(view.family) == RADIAL_VIEW_FAMILY

def is_tilted_radial_view(view: ViewInfo) -> bool:
    return bool(is_radial_view(view) and bool(view.radial_tilted_source))

def tilted_base_view_name(view: ViewInfo) -> str:
    if str(view.tilt_base_view):
        return str(view.tilt_base_view)
    return str(view.name)

def radial_base_view_name(view: ViewInfo) -> str:
    if not is_radial_view(view):
        raise ValueError(f'Radial base requested for non-radial view {view.name!r}')
    base = str(view.radial_base_view or view.tilt_base_view or 'transverse').strip().lower()
    if base not in CARTESIAN_VIEW_TOKENS:
        raise ValueError(f'Unsupported Radial base view: {base!r}')
    return base

def radial_target_base_view(token: str) -> str:
    target = str(token).strip().lower()
    base = target[len('tilted_'):] if target.startswith('tilted_') else target
    if base not in CARTESIAN_VIEW_TOKENS:
        raise ValueError(f'Unsupported Radial target {token!r}')
    return base

def cartesian_view_axis_spec(base_view: str, T: int, H: int, W: int) -> Dict[str, object]:
    base = str(base_view).lower()
    if base == 'transverse':
        return {
            'name': 'transverse',
            'display_name': 'Transverse',
            'num_slices': int(T),
            'src_h': int(H),
            'src_w': int(W),
            'summary_family': 'transverse',
            'horizontal_axis': 'x',
            'vertical_axis': 'y',
            'stack_axis': 't',
        }
    if base == 'sagittal':
        return {
            'name': 'sagittal',
            'display_name': 'Sagittal',
            'num_slices': int(H),
            'src_h': int(T),
            'src_w': int(W),
            'summary_family': 'sagittal',
            'horizontal_axis': 'x',
            'vertical_axis': 't',
            'stack_axis': 'y',
        }
    if base == 'coronal':
        return {
            'name': 'coronal',
            'display_name': 'Coronal',
            'num_slices': int(W),
            'src_h': int(T),
            'src_w': int(H),
            'summary_family': 'coronal',
            'horizontal_axis': 'y',
            'vertical_axis': 't',
            'stack_axis': 'x',
        }
    raise ValueError(f'Unsupported Cartesian base view: {base_view}')

def radial_plane_shape(view: ViewInfo) -> Tuple[int, int]:
    """Physical working-grid (plane_h, plane_w) containing this Radial circle."""
    spec = cartesian_view_axis_spec(
        radial_base_view_name(view), int(view.full_t), int(view.full_h), int(view.full_w),
    )
    return int(spec['src_h']), int(spec['src_w'])

def radial_stack_length(view: ViewInfo) -> int:
    spec = cartesian_view_axis_spec(
        radial_base_view_name(view), int(view.full_t), int(view.full_h), int(view.full_w),
    )
    return int(spec['num_slices'])

def radial_target_diameter(token: str, T: int, H: int, W: int) -> int:
    base = radial_target_base_view(str(token))
    spec = cartesian_view_axis_spec(base, int(T), int(H), int(W))
    return int(min(int(spec['src_h']), int(spec['src_w'])))

def tilted_radial_resident_gpu_render_enabled() -> bool:
    """Render tilted-Radial frames directly from a resident CUDA source volume."""
    return _env_flag('YOLO_TTA_GPU_TILTED_RADIAL_RENDER', True)

def radial_resident_gpu_render_supported(view: ViewInfo) -> bool:
    """Resident rendering supports upright Radial plus tilted-Radial transforms."""
    if not is_radial_view(view):
        return False
    if not is_tilted_radial_view(view):
        return True
    return bool(tilted_radial_resident_gpu_render_enabled())

def radial_streaming_gpu_render_supported(view: ViewInfo) -> bool:
    """Return whether logical-stack streaming prerender supports this upright Radial view."""
    return bool(
        is_radial_view(view)
        and not is_tilted_radial_view(view)
        and radial_base_view_name(view) in CARTESIAN_VIEW_TOKENS
    )

def radial_fused_render_supported(view: ViewInfo) -> bool:
    """Direct-to-binding support for every upright and enabled tilted-Radial base."""
    return bool(
        (
            is_radial_view(view)
            and not is_tilted_radial_view(view)
            and radial_base_view_name(view) in CARTESIAN_VIEW_TOKENS
        )
        or (
            is_tilted_radial_view(view)
            and tilted_radial_resident_gpu_render_enabled()
        )
    )

def radial_sink_only_projection_supported(view: ViewInfo) -> bool:
    """Block-sink projection supports every upright and tilted Radial orientation."""
    return bool(
        is_radial_view(view)
        and radial_base_view_name(view) in CARTESIAN_VIEW_TOKENS
    )

def tilted_stack_axis_length(view: ViewInfo) -> int:
    axis = str(view.stack_axis or cartesian_view_axis_spec(
        tilted_base_view_name(view), view.full_t, view.full_h, view.full_w,
    )['stack_axis'])
    if axis == 't':
        return int(view.full_t)
    if axis == 'y':
        return int(view.full_h)
    if axis == 'x':
        return int(view.full_w)
    raise ValueError(f'Unsupported Tilted View stacking axis: {axis}')

def build_radial_azimuths(azimuth_angle: float) -> List[float]:
    if float(azimuth_angle) <= 0.0:
        return []
    out: List[float] = []
    a = 0.0
    step = float(azimuth_angle)
    while a < 180.0 - 1e-9:
        out.append(float(a))
        a += step
    if not out:
        out.append(0.0)
    return out

def tilted_frame_center(view: ViewInfo, frame_idx: int) -> int:
    return int(view.tilt_frame_start) + int(frame_idx)

def _format_signed_angle_token(angle_deg: float) -> str:
    sign = 'p' if float(angle_deg) >= 0.0 else 'm'
    mag = f'{abs(float(angle_deg)):g}'.replace('.', 'p')
    return f'{sign}{mag}'

def _format_signed_angle_label(angle_deg: float) -> str:
    sign = '+' if float(angle_deg) >= 0.0 else '-'
    return f'{sign}{abs(float(angle_deg)):g}°'

def pretty_view_name(view: ViewInfo) -> str:
    return str(view.display_name)

def view_output_token(view: ViewInfo) -> str:
    if is_tilted_view(view):
        base = str(tilted_base_view_name(view)).capitalize()
        direction = str(view.tilt_direction or 'vertical').capitalize()
        token = f'Tilted{base}_{direction}_{_format_signed_angle_token(float(view.tilt_angle_deg))}'
    elif is_radial_view(view):
        base = str(radial_base_view_name(view)).capitalize()
        if is_tilted_radial_view(view):
            direction = str(view.tilt_direction or 'vertical').capitalize()
            token = (
                f'RadialTilted{base}_{direction}_'
                f'{_format_signed_angle_token(float(view.tilt_angle_deg))}'
            )
        else:
            token = 'Transverse_Radial' if base == 'Transverse' else f'Radial{base}'
    else:
        token = str(view.display_name).split(' / TTA ', 1)[0].replace(' ', '_')
    if is_tta_view_variant(view):
        token = f'{token}_TTA_{_sanitize_filesystem_token(view.tta_aug_id)}'
    return token

def _build_cartesian_view(base_view: str, T: int, H: int, W: int) -> ViewInfo:
    spec = cartesian_view_axis_spec(str(base_view), int(T), int(H), int(W))
    return ViewInfo(
        name=str(spec['name']),
        num_slices=int(spec['num_slices']),
        src_h=int(spec['src_h']),
        src_w=int(spec['src_w']),
        pad_mode='clamp',
        family='orthogonal',
        summary_family=str(spec['summary_family']),
        display_name=str(spec['display_name']),
        full_t=int(T),
        full_h=int(H),
        full_w=int(W),
        tilt_base_view=str(base_view),
        horizontal_axis=str(spec['horizontal_axis']),
        vertical_axis=str(spec['vertical_axis']),
        stack_axis=str(spec['stack_axis']),
    )

def _build_tilted_view_infos(
    T: int,
    H: int,
    W: int,
    *,
    tilt_views: Sequence[str],
    tilt_angles: Sequence[float],
    tilt_directions: Sequence[str],
) -> List[ViewInfo]:
    out: List[ViewInfo] = []
    angles = [float(a) for a in tilt_angles if float(a) > 0.0]
    for base_view in tilt_views:
        spec = cartesian_view_axis_spec(str(base_view), int(T), int(H), int(W))
        for tilt_direction in tilt_directions:
            for tilt_angle in angles:
                for sign in (+1.0, -1.0):
                    signed_angle = float(sign * tilt_angle)
                    token = _format_signed_angle_token(signed_angle)
                    base_label = str(spec['display_name'])
                    direction_label = str(tilt_direction).capitalize()
                    out.append(ViewInfo(
                        name=f'tilted_{base_view}_{tilt_direction}_{token}',
                        num_slices=int(spec['num_slices']),
                        src_h=int(spec['src_h']),
                        src_w=int(spec['src_w']),
                        pad_mode='clamp',
                        family=TILTED_VIEW_FAMILY,
                        summary_family=f'tilted_{base_view}_{tilt_direction}_{token}',
                        display_name=(
                            f'Tilted {base_label} {direction_label} '
                            f'{_format_signed_angle_label(signed_angle)}'
                        ),
                        full_t=int(T),
                        full_h=int(H),
                        full_w=int(W),
                        tilt_angle_deg=signed_angle,
                        tilt_direction=str(tilt_direction),
                        tilt_frame_start=0,
                        tilt_frame_stop=int(spec['num_slices']) - 1,
                        tilt_base_view=str(base_view),
                        horizontal_axis=str(spec['horizontal_axis']),
                        vertical_axis=str(spec['vertical_axis']),
                        stack_axis=str(spec['stack_axis']),
                    ))
    return out

def _build_tilted_view_infos_from_groups(
    T: int,
    H: int,
    W: int,
    *,
    tilt_groups: Sequence[TiltedViewGroup],
) -> List[ViewInfo]:
    """Expand structured requests through the unchanged Tilted geometry builder."""
    out: List[ViewInfo] = []
    seen_names: set[str] = set()
    for group in tilt_groups:
        built = _build_tilted_view_infos(
            int(T),
            int(H),
            int(W),
            tilt_views=group.views,
            tilt_angles=group.tilt_angles,
            tilt_directions=group.tilt_directions,
        )
        for view in built:
            if str(view.name) in seen_names:
                continue
            seen_names.add(str(view.name))
            out.append(view)
    return out

def _build_radial_view_info(
    T: int,
    H: int,
    W: int,
    *,
    base_view: str,
    azimuth_angle: float,
    radial_native_raster: int,
    request_token: str,
    tilted_source: Optional[ViewInfo] = None,
) -> ViewInfo:
    spec = cartesian_view_axis_spec(str(base_view), int(T), int(H), int(W))
    plane_h = int(spec['src_h'])
    plane_w = int(spec['src_w'])
    stack_len = int(spec['num_slices'])
    diameter = int(min(plane_w, plane_h))
    roi_radius = float(max(0.0, (diameter - 1) / 2.0))
    raster = int(radial_native_raster)
    radial_rows = int(min(stack_len, raster)) if raster > 0 else int(stack_len)
    radial_u = int(min(diameter, raster)) if raster > 0 else int(diameter)
    azimuths = tuple(build_radial_azimuths(float(azimuth_angle)))

    if tilted_source is None:
        name = f'radial_{base_view}'
        display_name = (
            'Transverse Radial'
            if str(base_view) == 'transverse'
            else f'Radial {str(spec["display_name"])}'
        )
        tilt_angle = 0.0
        tilt_direction = ''
        source_name = ''
        radial_tilted = False
    else:
        name = f'radial_{tilted_source.name}'
        display_name = f'Radial {pretty_view_name(tilted_source)}'
        tilt_angle = float(tilted_source.tilt_angle_deg)
        tilt_direction = str(tilted_source.tilt_direction)
        source_name = str(tilted_source.name)
        radial_tilted = True

    return ViewInfo(
        name=name,
        num_slices=len(azimuths),
        src_h=int(radial_rows),
        src_w=int(radial_u),
        pad_mode='pad',
        family=RADIAL_VIEW_FAMILY,
        summary_family=name,
        display_name=display_name,
        full_t=int(T),
        full_h=int(H),
        full_w=int(W),
        azimuths_deg=azimuths,
        diameter=int(diameter),
        center_x=float((plane_w - 1) / 2.0),
        center_y=float((plane_h - 1) / 2.0),
        roi_radius=float(roi_radius),
        tilt_angle_deg=float(tilt_angle),
        tilt_direction=str(tilt_direction),
        tilt_frame_start=0,
        tilt_frame_stop=int(stack_len) - 1,
        tilt_base_view=str(base_view),
        horizontal_axis='r',
        vertical_axis=str(spec['stack_axis']),
        stack_axis='azimuth',
        radial_base_view=str(base_view),
        radial_tilted_source=bool(radial_tilted),
        radial_source_view_name=str(source_name),
        radial_request_token=str(request_token),
    )

def radial_source_tilted_view(view: ViewInfo) -> ViewInfo:
    """Reconstruct the concrete Tilted view underlying a tilted Radial transform."""
    if not is_tilted_radial_view(view):
        raise ValueError(f'{view.name!r} is not a tilted Radial view')
    base = radial_base_view_name(view)
    spec = cartesian_view_axis_spec(base, int(view.full_t), int(view.full_h), int(view.full_w))
    token = _format_signed_angle_token(float(view.tilt_angle_deg))
    name = str(view.radial_source_view_name or f'tilted_{base}_{view.tilt_direction}_{token}')
    return ViewInfo(
        name=name,
        num_slices=int(spec['num_slices']),
        src_h=int(spec['src_h']),
        src_w=int(spec['src_w']),
        pad_mode='clamp',
        family=TILTED_VIEW_FAMILY,
        summary_family=name,
        display_name=(
            f'Tilted {str(spec["display_name"])} {str(view.tilt_direction).capitalize()} '
            f'{_format_signed_angle_label(float(view.tilt_angle_deg))}'
        ),
        full_t=int(view.full_t),
        full_h=int(view.full_h),
        full_w=int(view.full_w),
        tilt_angle_deg=float(view.tilt_angle_deg),
        tilt_direction=str(view.tilt_direction),
        tilt_frame_start=int(view.tilt_frame_start),
        tilt_frame_stop=int(view.tilt_frame_stop),
        tilt_base_view=base,
        horizontal_axis=str(spec['horizontal_axis']),
        vertical_axis=str(spec['vertical_axis']),
        stack_axis=str(spec['stack_axis']),
    )

def get_view_infos(
    T: int,
    H: int,
    W: int,
    *,
    cartesian_views: Optional[Sequence[str]] = None,
    radial_views: Optional[Sequence[str]] = None,
    radial_azimuth_angles: Optional[Sequence[float]] = None,
    tilt_groups: Optional[Sequence[TiltedViewGroup]] = None,
    radial_native_raster: int = 0,
) -> List[ViewInfo]:
    """Build the complete view set without changing any view-family geometry."""
    enabled_cartesian = resolve_cartesian_views(cartesian_views)
    enabled_radial = _resolve_unique_view_tokens(
        radial_views,
        valid=RADIAL_VIEW_TOKENS,
        flag_name='Radial view assembly',
    )
    resolved_tilt_groups = list(tilt_groups or ())
    azimuths = [float(v) for v in (radial_azimuth_angles or [])]
    if len(azimuths) != len(enabled_radial):
        raise ValueError(
            f'get_view_infos received {len(azimuths)} radial azimuth value(s) for '
            f'{len(enabled_radial)} Radial target(s)'
        )
    if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in azimuths):
        raise ValueError('get_view_infos requires one finite positive azimuth spacing per Radial target')

    orthogonal = [
        _build_cartesian_view(base, int(T), int(H), int(W))
        for base in enabled_cartesian
    ]
    tilted = _build_tilted_view_infos_from_groups(
        int(T),
        int(H),
        int(W),
        tilt_groups=resolved_tilt_groups,
    )

    radial_cartesian: List[ViewInfo] = []
    radial_tilted: List[ViewInfo] = []
    for target, angle in zip(enabled_radial, azimuths):
        base = radial_target_base_view(target)
        if not str(target).startswith('tilted_'):
            radial_cartesian.append(_build_radial_view_info(
                int(T), int(H), int(W),
                base_view=base,
                azimuth_angle=float(angle),
                radial_native_raster=int(radial_native_raster),
                request_token=str(target),
                tilted_source=None,
            ))
            continue

        matching = [v for v in tilted if tilted_base_view_name(v) == base]
        if not matching:
            print(
                f'Radial target {target!r} skipped: no {base} Tilted variants are enabled. '
                f'Add --enable_tilted {base}:30:both (or another valid group) to generate it.'
            )
            continue
        for source_view in matching:
            radial_tilted.append(_build_radial_view_info(
                int(T), int(H), int(W),
                base_view=base,
                azimuth_angle=float(angle),
                radial_native_raster=int(radial_native_raster),
                request_token=str(target),
                tilted_source=source_view,
            ))

    # Scheduling order is unchanged: upright Cartesian, upright Radial, concrete Tilted,
    # then Radial transforms of concrete Tilted variants.
    return orthogonal + radial_cartesian + tilted + radial_tilted

def orthogonal_views_only(views: Sequence[ViewInfo]) -> List[ViewInfo]:
    return [v for v in views if v.family == 'orthogonal']

@dataclass(frozen=True)
class RadialSampler:
    angle_deg: float
    diameter: int
    x_idx: np.ndarray
    y_idx: np.ndarray
    x_w: np.ndarray
    y_w: np.ndarray
    nn_x: np.ndarray
    nn_y: np.ndarray

_RADIAL_SAMPLER_CACHE: Dict[Tuple[object, ...], RadialSampler] = {}

RADIAL_FILTER_MODE = 'hardware_linear'

RADIAL_FILTER_LABEL = 'pure hardware-linear sampling'

RADIAL_FILTER_TAP_COUNT = 2

def _radial_filter_offsets() -> np.ndarray:
    """Integer support offsets around floor(sample) for the active radial filter."""
    return np.asarray((0, 1), dtype=np.int32)

def _radial_filter_kernel(x: np.ndarray) -> np.ndarray:
    """Two-tap triangle kernel used by the CPU/streaming hardware-linear fallback."""
    return np.maximum(
        np.float32(0.0),
        np.float32(1.0) - np.abs(np.asarray(x, dtype=np.float32)),
    ).astype(np.float32, copy=False)

def get_radial_sampler(view: ViewInfo, angle_deg: float) -> RadialSampler:
    if not is_radial_view(view):
        raise ValueError('Radial sampler requested for a non-radial view')

    # The circle lives in the selected Cartesian/Tilted projected plane rather than
    # unconditionally in global XY. The active reconstruction filter is therefore
    # evaluated in that base plane's (horizontal, vertical) coordinate system.
    plane_h, plane_w = radial_plane_shape(view)
    n_u = int(view.src_w) if int(view.src_w) > 0 else int(view.diameter)
    key = (
        RADIAL_FILTER_MODE, radial_base_view_name(view), int(plane_h), int(plane_w),
        int(view.diameter), int(n_u), round(float(view.center_x), 6),
        round(float(view.center_y), 6), round(float(view.roi_radius), 6),
        round(float(angle_deg), 6),
    )
    cached = _RADIAL_SAMPLER_CACHE.get(key)
    if cached is not None:
        return cached

    diameter = int(n_u)
    coords = np.linspace(-float(view.roi_radius), float(view.roi_radius), diameter, dtype=np.float32)
    theta = math.radians(float(angle_deg))
    xs = np.asarray(float(view.center_x) + coords * math.cos(theta), dtype=np.float32)
    ys = np.asarray(float(view.center_y) + coords * math.sin(theta), dtype=np.float32)

    offsets = _radial_filter_offsets()
    x0 = np.floor(xs).astype(np.int32, copy=False)
    y0 = np.floor(ys).astype(np.int32, copy=False)
    x_idx_raw = x0[:, None] + offsets[None, :]
    y_idx_raw = y0[:, None] + offsets[None, :]
    x_w = _radial_filter_kernel(xs[:, None] - x_idx_raw)
    y_w = _radial_filter_kernel(ys[:, None] - y_idx_raw)

    # Preserve the established boundary contract: invalid taps are removed and the
    # remaining one-dimensional weights are renormalized independently on each axis.
    x_valid = (x_idx_raw >= 0) & (x_idx_raw < int(plane_w))
    y_valid = (y_idx_raw >= 0) & (y_idx_raw < int(plane_h))
    x_w *= x_valid.astype(np.float32, copy=False)
    y_w *= y_valid.astype(np.float32, copy=False)
    x_w_sum = np.sum(x_w, axis=1, keepdims=True)
    y_w_sum = np.sum(y_w, axis=1, keepdims=True)
    np.divide(x_w, x_w_sum, out=x_w, where=np.abs(x_w_sum) > 1e-6)
    np.divide(y_w, y_w_sum, out=y_w, where=np.abs(y_w_sum) > 1e-6)

    x_idx = np.clip(x_idx_raw, 0, int(plane_w) - 1).astype(np.int32, copy=False)
    y_idx = np.clip(y_idx_raw, 0, int(plane_h) - 1).astype(np.int32, copy=False)
    sampler = RadialSampler(
        angle_deg=float(angle_deg), diameter=diameter,
        x_idx=x_idx, y_idx=y_idx,
        x_w=x_w.astype(np.float32, copy=False),
        y_w=y_w.astype(np.float32, copy=False),
        nn_x=np.clip(np.rint(xs).astype(np.int32, copy=False), 0, int(plane_w) - 1),
        nn_y=np.clip(np.rint(ys).astype(np.int32, copy=False), 0, int(plane_h) - 1),
    )
    _RADIAL_SAMPLER_CACHE[key] = sampler
    return sampler

def choose_radial_exact_block_frames(diameter: int, target_bytes: int = 256 * 1024 * 1024) -> int:
    env = os.environ.get('YOLO_TTA_RADIAL_EXACT_BLOCK_FRAMES', '').strip()
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass
    bytes_per_frame = max(
        1,
        int(diameter) * int(RADIAL_FILTER_TAP_COUNT) * np.dtype(np.float32).itemsize,
    )
    block = max(1, int(target_bytes // bytes_per_frame))
    return max(1, min(256, block))

def _radial_sampler_flat_taps(sampler: RadialSampler, image_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened active-filter tap indices and separable weights for one radial line."""
    u_len = int(sampler.diameter)
    flat_idx = (
        sampler.y_idx.astype(np.int64)[:, :, None] * int(image_w)
        + sampler.x_idx.astype(np.int64)[:, None, :]
    ).reshape(u_len, -1)
    w2d = (
        sampler.y_w.astype(np.float32)[:, :, None]
        * sampler.x_w.astype(np.float32)[:, None, :]
    ).reshape(u_len, -1)
    return flat_idx, w2d

def radial_oriented_stack_view(volume_rgb: np.ndarray, view: ViewInfo) -> np.ndarray:
    """Return a logical ``(stack, plane_v, plane_u)`` view for a Cartesian Radial base."""
    arr = np.asarray(volume_rgb)
    if arr.ndim != 3:
        raise ValueError(f'Radial source volume must be 3D, got {arr.shape}')
    base = radial_base_view_name(view)
    if base == 'transverse':
        return arr
    if base == 'sagittal':
        # stack Y; in-plane axes (t, X)
        return np.transpose(arr, (1, 0, 2))
    if base == 'coronal':
        # stack X; in-plane axes (t, Y)
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f'Unsupported Radial base: {base}')

def extract_radial_slice_frame(
    volume_rgb: np.ndarray,
    sampler: RadialSampler,
    out_rows: Optional[int] = None,
) -> np.ndarray:
    """Extract one diameter frame from a logical ``(stack, plane_v, plane_u)`` volume."""
    stack_dim = int(volume_rgb.shape[0])
    u_len = int(sampler.diameter)
    rows = int(out_rows) if out_rows is not None and int(out_rows) > 0 else stack_dim
    fold_stack = int(rows) != int(stack_dim)

    image_w = int(volume_rgb.shape[2])
    flat_idx, w2d = _radial_sampler_flat_taps(sampler, image_w)
    block_frames = choose_radial_exact_block_frames(u_len)

    plane_len = int(volume_rgb.shape[1]) * image_w
    proj = np.empty((stack_dim, u_len), dtype=np.float32) if fold_stack else None
    out = None if fold_stack else np.empty((stack_dim, u_len), dtype=np.uint8)

    for start_idx in range(0, stack_dim, block_frames):
        stop_idx = min(stack_dim, start_idx + block_frames)
        # Sagittal/Coronal orientation views are strided. Materialize only this bounded
        # stack block contiguously so the active-filter gather remains vectorized.
        block2d = np.ascontiguousarray(volume_rgb[start_idx:stop_idx]).reshape(
            stop_idx - start_idx, plane_len,
        )
        samples = block2d[:, flat_idx].astype(np.float32, copy=False)
        acc = np.einsum('tuk,uk->tu', samples, w2d)
        if fold_stack:
            proj[start_idx:stop_idx, :] = acc
        else:
            out[start_idx:stop_idx, :] = np.clip(np.rint(acc), 0.0, 255.0).astype(np.uint8)

    if not fold_stack:
        return out

    # Center-aligned linear reduction of the selected Radial base's stack axis.
    rf = (np.arange(rows, dtype=np.float64) + 0.5) * (float(stack_dim) / float(rows)) - 0.5
    r0 = np.clip(np.floor(rf).astype(np.int64), 0, stack_dim - 1)
    r1 = np.minimum(r0 + 1, stack_dim - 1)
    alpha = np.clip(rf - r0, 0.0, 1.0).astype(np.float32)[:, None]
    folded = proj[r0] * (np.float32(1.0) - alpha) + proj[r1] * alpha
    return np.clip(np.rint(folded), 0.0, 255.0).astype(np.uint8)

def _tilted_radial_row_centers(stack_len: int, rows: int) -> np.ndarray:
    if int(rows) == int(stack_len):
        return np.arange(int(stack_len), dtype=np.float32)
    coords = (np.arange(int(rows), dtype=np.float64) + 0.5) * (
        float(stack_len) / float(rows)
    ) - 0.5
    return np.clip(coords, 0.0, float(max(0, int(stack_len) - 1))).astype(np.float32)

def extract_tilted_radial_slice_frame(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    sampler: RadialSampler,
    out_rows: Optional[int] = None,
) -> np.ndarray:
    """Render one Radial transform of a concrete Tilted view directly from the volume.

 The output is circular in the Tilted projected plane. For every diameter tap, the
 same signed stacking-axis shear used by the underlying Tilted view is applied before
 a two-tap stack interpolation. The active in-plane reconstruction taps are accumulated without
 materializing thousands of full Tilted frames."""
    if not is_tilted_radial_view(view):
        raise ValueError('Tilted Radial renderer requires a tilted Radial view')
    arr = np.asarray(volume_rgb)
    if arr.ndim != 3:
        raise ValueError(f'Tilted Radial source volume must be 3D, got {arr.shape}')

    base = radial_base_view_name(view)
    stack_len = int(radial_stack_length(view))
    rows = int(out_rows) if out_rows is not None and int(out_rows) > 0 else stack_len
    row_centers = _tilted_radial_row_centers(stack_len, rows)
    u_len = int(sampler.diameter)
    tap_count = int(sampler.x_idx.shape[1]) * int(sampler.y_idx.shape[1])

    x_taps = np.broadcast_to(
        sampler.x_idx[:, None, :],
        (u_len, int(sampler.y_idx.shape[1]), int(sampler.x_idx.shape[1])),
    ).reshape(u_len, tap_count)
    y_taps = np.broadcast_to(
        sampler.y_idx[:, :, None],
        (u_len, int(sampler.y_idx.shape[1]), int(sampler.x_idx.shape[1])),
    ).reshape(u_len, tap_count)
    weights = (
        sampler.y_w[:, :, None].astype(np.float32, copy=False)
        * sampler.x_w[:, None, :].astype(np.float32, copy=False)
    ).reshape(u_len, tap_count)

    tan_alpha = np.float32(math.tan(math.radians(float(view.tilt_angle_deg))))
    if str(view.tilt_direction) == 'vertical':
        tap_offsets = y_taps.astype(np.float32, copy=False) - np.float32(view.center_y)
    elif str(view.tilt_direction) == 'horizontal':
        tap_offsets = x_taps.astype(np.float32, copy=False) - np.float32(view.center_x)
    else:
        raise ValueError(f'Unsupported Tilted Radial direction: {view.tilt_direction!r}')

    out = np.empty((rows, u_len), dtype=np.uint8)
    block_rows = max(1, _env_int('YOLO_TTA_TILTED_RADIAL_ROW_BLOCK', 16))
    for row0 in range(0, rows, block_rows):
        row1 = min(rows, row0 + block_rows)
        centers = row_centers[row0:row1, None]
        acc = np.zeros((row1 - row0, u_len), dtype=np.float32)
        for tap_idx in range(tap_count):
            weight = weights[:, tap_idx]
            if not np.any(weight):
                continue
            px = x_taps[:, tap_idx]
            py = y_taps[:, tap_idx]
            stack_src = centers + tan_alpha * tap_offsets[:, tap_idx][None, :]
            valid = (stack_src >= 0.0) & (stack_src <= float(stack_len - 1))
            if not np.any(valid):
                continue
            s0 = np.clip(np.floor(stack_src).astype(np.int32), 0, stack_len - 1)
            s1 = np.minimum(s0 + 1, stack_len - 1)
            alpha = (stack_src - s0).astype(np.float32, copy=False)

            if base == 'transverse':
                f0 = arr[s0, py[None, :], px[None, :]].astype(np.float32, copy=False)
                f1 = arr[s1, py[None, :], px[None, :]].astype(np.float32, copy=False)
            elif base == 'sagittal':
                f0 = arr[py[None, :], s0, px[None, :]].astype(np.float32, copy=False)
                f1 = arr[py[None, :], s1, px[None, :]].astype(np.float32, copy=False)
            elif base == 'coronal':
                f0 = arr[py[None, :], px[None, :], s0].astype(np.float32, copy=False)
                f1 = arr[py[None, :], px[None, :], s1].astype(np.float32, copy=False)
            else:  # pragma: no cover
                raise ValueError(f'Unsupported Tilted Radial base: {base}')

            values = f0 + alpha * (f1 - f0)
            values *= valid.astype(np.float32, copy=False)
            acc += values * weight[None, :]
        out[row0:row1] = np.clip(np.rint(acc), 0.0, 255.0).astype(np.uint8)
    return out

def _cupy_external_stream(cp: object, torch_stream: object) -> object:
    """Bridge a Torch CUDA stream into CuPy without the CuPy 14 ExternalStream warning.

 CuPy 14 prefers ``Stream.from_external`` and Torch streams implement the CUDA stream
 protocol in current builds. Older CuPy/Torch combinations retain the pointer-based
 ExternalStream fallback so this compatibility cleanup cannot disable an acceleration path."""
    stream_cls = getattr(getattr(cp, 'cuda', None), 'Stream', None)
    factory = getattr(stream_cls, 'from_external', None)
    if callable(factory):
        for candidate in (torch_stream, int(getattr(torch_stream, 'cuda_stream'))):
            try:
                return factory(candidate)
            except Exception:
                continue
    return cp.cuda.ExternalStream(int(getattr(torch_stream, 'cuda_stream')))

def write_aug_job_meta(
    job: AugJob,
    view: ViewInfo,
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
) -> None:
    fmt = resolve_channel_format(channel_format)
    job.meta_path.parent.mkdir(parents=True, exist_ok=True)
    job.meta_path.write_text(
        json.dumps(
            {
                'view': view.name,
                'physical_view': physical_view_name(view),
                'family': view.family,
                'num_slices': int(view.num_slices),
                'full_t': int(view.full_t),
                'angle_deg': float(job.angle_deg),
                'src_w': view.src_w,
                'src_h': view.src_h,
                'tilt_angle_deg': float(view.tilt_angle_deg),
                'tilt_direction': str(view.tilt_direction),
                'tilt_base_view': str(view.tilt_base_view),
                'horizontal_axis': str(view.horizontal_axis),
                'vertical_axis': str(view.vertical_axis),
                'stack_axis': str(view.stack_axis),
                'radial_base_view': str(view.radial_base_view),
                'radial_tilted_source': bool(view.radial_tilted_source),
                'radial_source_view_name': str(view.radial_source_view_name),
                'radial_request_token': str(view.radial_request_token),
                'tilt_frame_start': int(view.tilt_frame_start),
                'tilt_frame_stop': int(view.tilt_frame_stop),
                'out_size': int(job.aff.out_size),
                'pad_mode': view.pad_mode,
                'canvas_w': int(job.aff.canvas_w),
                'canvas_h': int(job.aff.canvas_h),
                'pad_size': int(job.aff.pad_size),
                'pad_off_x': float(job.aff.pad_off_x),
                'pad_off_y': float(job.aff.pad_off_y),
                'M_out_to_src': job.aff.M_out_to_src.tolist(),
                'M_src_to_out': job.aff.M_src_to_out.tolist(),
                'channel_format': fmt.token,
                'model_input_channels': int(fmt.channel_count),
                'channel_stride': int(fmt.stride),
                'channel_offsets': [int(v) for v in fmt.offsets],
                'channel_boundary_policy': 'radial_wrap_mirror_u_cartesian_edge_clamp',
                'prediction_slice_policy': 'center_N_only',
            },
            indent=2,
        )
    )

def build_aug_job_for_variant(
    view: ViewInfo,
    out_size: int,
    temp_dir: Path,
) -> AugJob:
    """Build the sole augmentation owned by one v16.4.0 TTA view variant."""
    if not is_tta_view_variant(view):
        raise ValueError(
            f'v16.4.0 augmentation construction requires a TTA view variant; got {view.name!r}'
        )
    angle = float(view.tta_angle_deg)
    aug_id = str(view.tta_aug_id)
    expected_aug_id = _format_angle_aug_id(angle)
    if aug_id != expected_aug_id:
        raise ValueError(
            f'TTA variant {view.name!r} has aug_id={aug_id!r}, expected {expected_aug_id!r} '
            f'for angle {angle:g}'
        )
    aug_dir = temp_dir / 'aug' / view.name
    aff = build_affine(
        view=view.name,
        src_w=view.src_w,
        src_h=view.src_h,
        out_size=out_size,
        angle_deg=angle,
        pad_mode=view.pad_mode,
    )
    return AugJob(
        aug_id=aug_id,
        angle_deg=angle,
        meta_path=aug_dir / f'{view.name}_{aug_id}.meta.json',
        aff=aff,
    )

def iter_aug_jobs_round_robin(
    views: Sequence[ViewInfo],
    aug_jobs_by_view: Dict[str, Sequence[AugJob]],
) -> Iterator[Tuple[ViewInfo, AugJob]]:
    """Yield augmentation jobs one round per view so later view families start earlier.

 The prior FIFO submission order rendered every augmentation for early views before later
 views even entered the render queue. That could leave later families, notably tilted
 transverse variants, sitting behind a long tail of earlier canvas/tile work even while the GPU
 had no ready inference job. Round-robin submission starts one job per active view first, then
 the second job per view, and so on."""
    queues: Dict[str, deque[AugJob]] = {
        str(view.name): deque(aug_jobs_by_view.get(view.name, ()))
        for view in views
    }
    while True:
        emitted = False
        for view in views:
            q = queues[str(view.name)]
            if not q:
                continue
            emitted = True
            yield view, q.popleft()
        if not emitted:
            break

@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    tile_stride: int
    config_id: str

def resolve_tile_configs(
    values: Sequence[str] | str | None,
) -> List[TileConfig]:
    """Resolve structured ``TILE_SIZE:TILE_STRIDE`` groups."""
    configs: List[TileConfig] = []
    seen: set[str] = set()
    for raw_group in _structured_group_values(values, flag_name='--enable_tile'):
        size_slot, stride_slot = _split_structured_group(
            raw_group,
            slot_count=2,
            flag_name='--enable_tile',
        )
        if not size_slot or not stride_slot:
            raise ValueError(
                f'--enable_tile group {raw_group!r} requires both '
                'TILE_SIZE and TILE_STRIDE'
            )
        if len(_parse_comma_slot(size_slot)) != 1 or len(_parse_comma_slot(stride_slot)) != 1:
            raise ValueError(
                f'--enable_tile group {raw_group!r} accepts one TILE_SIZE and one '
                'TILE_STRIDE; use spaces to separate additional groups'
            )
        try:
            tile_size = int(size_slot)
            tile_stride = int(stride_slot)
        except Exception as exc:
            raise ValueError(
                f'--enable_tile group {raw_group!r} requires integer '
                'TILE_SIZE:TILE_STRIDE values'
            ) from exc
        if int(tile_size) <= 0:
            raise ValueError(
                f'--enable_tile group {raw_group!r} requires TILE_SIZE > 0'
            )
        if int(tile_stride) <= 0:
            raise ValueError(
                f'--enable_tile group {raw_group!r} requires TILE_STRIDE > 0'
            )
        if int(tile_stride) > int(tile_size):
            raise ValueError(
                f'--enable_tile group {raw_group!r} requires TILE_STRIDE <= TILE_SIZE'
            )
        config_id = f's{int(tile_size)}_st{int(tile_stride)}'
        if config_id in seen:
            raise ValueError(
                f'--enable_tile contains duplicate group {int(tile_size)}:{int(tile_stride)}'
            )
        seen.add(config_id)
        configs.append(TileConfig(
            tile_size=int(tile_size),
            tile_stride=int(tile_stride),
            config_id=config_id,
        ))
    return configs

@dataclass(frozen=True)
class DenseTileJob:
    view: str
    aug_id: str
    config_id: str
    tile_id: str
    tile_x: int
    tile_y: int
    tile_size: int
    tile_stride: int
    out_size: int
    meta_path: Path
    M_out_to_src: np.ndarray
    M_src_to_out: np.ndarray
    # the tile's fixed footprint in the parent processing grid, as (py0, py1, px0, px1),
    # plus the same output->parent affine rebased into crop-local pixels. Every per-tile
    # artifact is sized to this window instead of the whole parent grid. Populated by
    # build_dense_tile_jobs_for_aug; the zero default only survives in synthetic jobs.
    parent_crop: Tuple[int, int, int, int] = (0, 0, 0, 0)
    M_out_to_crop: Optional[np.ndarray] = None

def channel_view_slice_source(view: ViewInfo, index: int) -> Tuple[int, bool]:
    """Resolve one contextual channel source as ``(slice_index, mirror_u)``.

    Radial frames cover the unoriented angular domain ``[0, 180)``. Crossing either
    end therefore reuses the modulo-resolved frame with its radial ``u`` coordinate
    reversed. Multiple-period custom strides retain the reversal only after an odd
    number of seam crossings. Cartesian view families continue to edge-clamp.
    """
    count = max(1, int(view.num_slices))
    requested = int(index)
    if is_radial_view(view):
        wraps, source_idx = divmod(requested, int(count))
        return int(source_idx), bool(int(wraps) % 2)
    return max(0, min(int(count) - 1, requested)), False


def channel_view_slice_index(view: ViewInfo, index: int) -> int:
    """Resolve one contextual channel index under the view family's boundary policy."""
    return int(channel_view_slice_source(view, index)[0])

class ChannelFormattedFrameRenderer:
    """Build one model input from independently rendered view planes.

 The bounded single-flight cache reuses transformed neighboring planes across
 overlapping 2.5D centers without materializing a complete channel-formatted
 volume. It also deduplicates repeated boundary-resolved source/orientation pairs
 within one frame."""

    def __init__(
        self,
        plane_renderer: Callable[[int], np.ndarray],
        view: ViewInfo,
        channel_format: ChannelFormat,
        *,
        cache_frames: Optional[int] = None,
        mirrored_plane_renderer: Optional[Callable[[int], np.ndarray]] = None,
    ) -> None:
        self.plane_renderer = plane_renderer
        self.mirrored_plane_renderer = mirrored_plane_renderer
        self.view = view
        self.channel_format = resolve_channel_format(channel_format)
        if cache_frames is None:
            default_cache = 0 if self.channel_format.kind == 'gray' else max(
                8, min(32, 2 * int(self.channel_format.channel_count) + 2)
            )
            cache_frames = _env_int('YOLO_TTA_CHANNEL_PLANE_CACHE_FRAMES', default_cache)
        self.cache_frames = max(0, int(cache_frames))
        self._cache: 'OrderedDict[Tuple[int, bool], Future]' = OrderedDict()
        self._lock = threading.Lock()

    def _render_plane(self, source_idx: int, mirror_u: bool = False) -> np.ndarray:
        if bool(mirror_u) and self.mirrored_plane_renderer is not None:
            rendered = self.mirrored_plane_renderer(int(source_idx))
        else:
            rendered = self.plane_renderer(int(source_idx))
        plane = np.asarray(rendered, dtype=np.uint8)
        if plane.ndim == 3 and int(plane.shape[2]) == 1:
            plane = plane[:, :, 0]
        if plane.ndim != 2:
            raise ValueError(
                f'Channel plane renderer returned {plane.shape} for {self.view.name} '
                f'slice {int(source_idx)}; expected HxW grayscale'
            )
        if bool(mirror_u) and self.mirrored_plane_renderer is None:
            plane = plane[:, ::-1]
        return np.ascontiguousarray(plane, dtype=np.uint8)

    def _get_plane(self, source_idx: int, mirror_u: bool = False) -> np.ndarray:
        key = (int(source_idx), bool(mirror_u))
        if self.cache_frames <= 0:
            return self._render_plane(*key)

        owner = False
        with self._lock:
            future = self._cache.get(key)
            if future is None:
                future = Future()
                self._cache[key] = future
                owner = True
            else:
                self._cache.move_to_end(key)

        if owner:
            try:
                future.set_result(self._render_plane(*key))
            except BaseException as exc:
                future.set_exception(exc)
                with self._lock:
                    self._cache.pop(key, None)
                raise

        plane = future.result()
        with self._lock:
            # A different center can evict this completed entry after
            # future.result wakes us but before this lock is reacquired.
            # The local plane remains valid even when the LRU no longer owns it.
            if self._cache.get(key) is future:
                self._cache.move_to_end(key, last=True)
            while len(self._cache) > int(self.cache_frames):
                removed = False
                for old_key, old_future in list(self._cache.items()):
                    if old_key != key and old_future.done():
                        self._cache.pop(old_key, None)
                        removed = True
                        break
                if not removed:
                    break
        return plane

    def __call__(self, center_idx: int) -> np.ndarray:
        fmt = self.channel_format
        center = int(center_idx)
        if fmt.kind == 'gray':
            source_idx, mirror_u = channel_view_slice_source(self.view, center)
            return self._get_plane(source_idx, mirror_u)

        if fmt.kind == 'rgb':
            source_idx, mirror_u = channel_view_slice_source(self.view, center)
            plane = self._get_plane(source_idx, mirror_u)
            return np.ascontiguousarray(np.repeat(plane[:, :, None], 3, axis=2))

        # Deduplicate repeated boundary-resolved indices inside this stack even when the
        # shared cache is disabled.
        local_planes: Dict[Tuple[int, bool], np.ndarray] = {}
        ordered_planes: List[np.ndarray] = []
        for offset in fmt.offsets:
            source = channel_view_slice_source(self.view, center + int(offset))
            plane = local_planes.get(source)
            if plane is None:
                plane = self._get_plane(*source)
                local_planes[source] = plane
            ordered_planes.append(plane)
        return np.ascontiguousarray(np.stack(ordered_planes, axis=2), dtype=np.uint8)

@dataclass
class PredictionVolumeRef:
    """One active YOLO input job.
    
    ``array`` holds a dense materialized volume; ``source`` holds the default bounded streaming renderer."""

    array: Optional[np.ndarray]
    path: Optional[Path]
    name: str
    view_name: str
    job_id: str
    kind: str = 'fullframe'
    source: Optional[object] = None
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT

def close_prediction_volume_ref(ref: Optional[PredictionVolumeRef], *, keep_temp: bool = False) -> None:
    """Release one prediction source and remove any fallback backing file."""
    if ref is None:
        return

    close_fn = getattr(getattr(ref, 'source', None), 'close', None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception as exc:
            print(f'Warning: failed to close streaming prediction source {ref.name} ({exc})')

    arr = getattr(ref, 'array', None)
    path = getattr(ref, 'path', None)
    if arr is not None:
        if bool(keep_temp):
            close_memmap_array(arr)
        else:
            close_memmap_array_without_flush(arr)
    if not bool(keep_temp) and path is not None:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass

class ChannelFormattedYoloBatch(list):
    """Image-list marker that preserves the requested H×W×C channel order."""

    def __init__(self, *, channel_count: int) -> None:
        super().__init__()
        self._tta_channel_count = max(1, int(channel_count))

class InMemoryYoloVolumeSource:
    """Ultralytics-compatible in-memory source that streams model-input batches.

 Ultralytics' public Python API accepts in-memory numpy inputs, but its built-in
 ``LoadPilAndNumpy`` loader treats a list of arrays as one large batch. This
 loader lets the predictor consume a 3-D gray volume or 4-D H×W×C volume
 incrementally with ``stream=True`` and the requested ``--batch``. The final
 batch repeats the final complete channel-formatted center frame so fixed-batch
 engines always receive the same batch size. Downstream accumulation discards
 synthetic padded results."""

    def __init__(
        self,
        volume_gray: np.ndarray,
        name: str,
        batch_size: int = 1,
        max_frames: Optional[int] = None,
        channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    ) -> None:
        if volume_gray is None:
            raise ValueError('InMemoryYoloVolumeSource requires a volume array')
        self.channel_format = resolve_channel_format(channel_format)
        self.channel_count = int(self.channel_format.channel_count)
        self.volume_gray = np.asarray(volume_gray)
        if self.volume_gray.ndim == 4:
            if int(self.volume_gray.shape[3]) != int(self.channel_count):
                raise ValueError(
                    f'Prediction volume channel count {int(self.volume_gray.shape[3])} does not match '
                    f'--channel_format {self.channel_format.token} ({int(self.channel_count)}); '
                    f'got shape {self.volume_gray.shape}'
                )
        elif self.volume_gray.ndim != 3:
            raise ValueError(f'Prediction volume must have shape (N,H,W) or (N,H,W,C); got {self.volume_gray.shape}')
        elif int(self.channel_count) != 1:
            raise ValueError(
                f'Prediction volume shape {self.volume_gray.shape} is one-channel but '
                f'--channel_format {self.channel_format.token} requires {int(self.channel_count)} channels'
            )
        self.name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'in_memory_volume'
        self.nf = int(self.volume_gray.shape[0])
        if max_frames is not None:
            self.nf = max(0, min(self.nf, int(max_frames)))
        self.bs = max(1, int(batch_size))
        self.yield_nf = int(math.ceil(float(self.nf) / float(self.bs)) * self.bs) if self.nf > 0 else 0
        self.synthetic_count = max(0, int(self.yield_nf) - int(self.nf))
        self.mode = 'image'
        self.count = 0
        try:
            from ultralytics.data.loaders import SourceTypes  # type: ignore
            self.source_type = SourceTypes(stream=False, screenshot=False, from_img=True, tensor=False)
        except Exception:
            # Older/forked Ultralytics builds only require these attributes.
            self.source_type = argparse.Namespace(stream=False, screenshot=False, from_img=True, tensor=False)

    def __iter__(self) -> 'InMemoryYoloVolumeSource':
        self.count = 0
        return self

    def __len__(self) -> int:
        return int(math.ceil(float(self.nf) / float(self.bs))) if self.nf > 0 else 0

    @staticmethod
    def _frame_to_model_channels(frame: np.ndarray, channel_count: int) -> np.ndarray:
        """Return one prediction frame as contiguous H×W×C uint8."""
        frame_u8 = np.asarray(frame, dtype=np.uint8)
        if frame_u8.ndim == 2:
            if int(channel_count) != 1:
                raise ValueError(
                    f'Expected {int(channel_count)} model-input channels, got a 2-D frame'
                )
            return np.ascontiguousarray(frame_u8[:, :, None])
        if frame_u8.ndim == 3 and int(frame_u8.shape[2]) == int(channel_count):
            return np.ascontiguousarray(frame_u8)
        raise ValueError(
            f'Unsupported prediction frame shape {frame_u8.shape}; expected HxWx{int(channel_count)}'
        )

    def __next__(self) -> Tuple[List[str], List[np.ndarray], List[str]]:
        if self.count >= self.yield_nf:
            raise StopIteration
        if self.nf <= 0:
            raise StopIteration
        start = int(self.count)
        stop = min(int(self.yield_nf), start + int(self.bs))
        self.count = int(stop)
        paths: List[str] = []
        imgs: List[np.ndarray] = ChannelFormattedYoloBatch(channel_count=int(self.channel_count))
        info: List[str] = []
        last_real_idx = max(0, int(self.nf) - 1)
        for idx in range(start, stop):
            real_idx = int(idx) if int(idx) < int(self.nf) else int(last_real_idx)
            synthetic = int(idx) >= int(self.nf)
            suffix = '_synthetic' if synthetic else ''
            paths.append(f'{self.name}_{idx + 1:06d}{suffix}.png')
            imgs.append(self._frame_to_model_channels(
                self.volume_gray[int(real_idx)], int(self.channel_count)
            ))
            if synthetic:
                info.append(f'in-memory {self.name} synthetic padded slice {idx + 1}/{self.yield_nf} repeats real slice {real_idx + 1}/{self.nf}: ')
            else:
                info.append(f'in-memory {self.name} slice {idx + 1}/{self.nf}: ')
        return paths, imgs, info

class StreamingYoloVolumeSource:
    """Ultralytics-compatible source that renders prediction centers just ahead of YOLO.

 The legacy path first materialized a complete ``(N,imgsz,imgsz)`` uint8
 array for every full-frame/tile job. This source keeps only a bounded window
 of rendered futures alive: the first batch can enter YOLO as soon as those
 frames are ready, while CPU workers continue rendering later slices behind the
 GPU stream."""

    def __init__(
        self,
        renderer: Callable[[int], np.ndarray],
        *,
        num_frames: int,
        name: str,
        batch_size: int = 1,
        max_frames: Optional[int] = None,
        out_size: Optional[int] = None,
        render_workers: int = 1,
        prefetch_frames: Optional[int] = None,
        autostart: bool = True,
        shared_executor: Optional[ThreadPoolExecutor] = None,
        channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
    ) -> None:
        self.renderer = renderer
        self.channel_format = resolve_channel_format(channel_format)
        self.channel_count = int(self.channel_format.channel_count)
        self.name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_') or 'streaming_volume'
        self.nf = max(0, int(num_frames))
        if max_frames is not None:
            self.nf = max(0, min(self.nf, int(max_frames)))
        self.out_size = None if out_size is None else int(out_size)
        self.bs = max(1, int(batch_size))
        self.yield_nf = int(math.ceil(float(self.nf) / float(self.bs)) * self.bs) if self.nf > 0 else 0
        self.synthetic_count = max(0, int(self.yield_nf) - int(self.nf))
        self.render_workers = max(1, min(int(render_workers), max(1, int(self.nf))))
        default_prefetch = streaming_prediction_source_prefetch_frames(self.bs)
        self.prefetch_frames = max(self.bs, int(prefetch_frames if prefetch_frames is not None else default_prefetch))
        self.mode = 'image'
        self.count = 0
        self._next_submit = 0
        self._futures: Dict[int, Future] = {}
        self._external_executor = shared_executor
        self._executor: Optional[ThreadPoolExecutor] = shared_executor
        self._owns_executor = shared_executor is None
        self._lock = threading.Lock()
        self._closed = False
        try:
            from ultralytics.data.loaders import SourceTypes  # type: ignore
            self.source_type = SourceTypes(stream=False, screenshot=False, from_img=True, tensor=False)
        except Exception:
            self.source_type = argparse.Namespace(stream=False, screenshot=False, from_img=True, tensor=False)
        if bool(autostart):
            self.start()

    def __len__(self) -> int:
        return int(math.ceil(float(self.nf) / float(self.bs))) if self.nf > 0 else 0

    def __iter__(self) -> 'StreamingYoloVolumeSource':
        self.start()
        self.count = 0
        self._fill_prefetch_locked(target_index=0)
        return self

    @staticmethod
    def _frame_to_model_channels(frame: np.ndarray, channel_count: int) -> np.ndarray:
        return InMemoryYoloVolumeSource._frame_to_model_channels(frame, int(channel_count))

    def _ensure_executor_locked(self) -> ThreadPoolExecutor:
        if self._executor is not None:
            return self._executor
        if self._external_executor is not None:
            self._executor = self._external_executor
            self._owns_executor = False
            return self._executor
        self._executor = ThreadPoolExecutor(
            max_workers=int(self.render_workers),
            thread_name_prefix=f'yolo-render-{self.name[:24]}',
        )
        self._owns_executor = True
        return self._executor

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError(f'Streaming YOLO source {self.name} is already closed')
            self._ensure_executor_locked()
            self._fill_prefetch_locked(target_index=0)

    def close(self) -> None:
        executor: Optional[ThreadPoolExecutor]
        futures_to_cancel: List[Future]
        owns_executor: bool
        with self._lock:
            self._closed = True
            executor = self._executor
            owns_executor = bool(self._owns_executor)
            futures_to_cancel = list(self._futures.values())
            self._executor = None
            self._futures.clear()
        for fut in futures_to_cancel:
            try:
                fut.cancel()
            except Exception:
                pass
        if executor is not None and owns_executor:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)

    def _submit_locked(self, idx: int) -> None:
        idx_i = int(idx)
        if idx_i in self._futures or idx_i >= int(self.nf):
            return
        executor = self._ensure_executor_locked()
        self._futures[idx_i] = executor.submit(self._render_one, idx_i)
        self._next_submit = max(int(self._next_submit), idx_i + 1)

    def _fill_prefetch_locked(self, target_index: int) -> None:
        if self._closed or int(self.nf) <= 0:
            return
        submit_stop = min(int(self.nf), max(int(self._next_submit), int(target_index) + int(self.prefetch_frames)))
        while int(self._next_submit) < int(submit_stop):
            self._submit_locked(int(self._next_submit))

    def _ensure_submitted(self, stop_exclusive: int) -> None:
        with self._lock:
            self._fill_prefetch_locked(target_index=max(0, int(stop_exclusive)))
            while int(self._next_submit) < min(int(self.nf), int(stop_exclusive)):
                self._submit_locked(int(self._next_submit))

    def _render_one(self, idx: int) -> np.ndarray:
        frame = np.asarray(self.renderer(int(idx)), dtype=np.uint8)
        formatted = self._frame_to_model_channels(frame, int(self.channel_count))
        if self.out_size is not None and (
            int(formatted.shape[0]) != int(self.out_size)
            or int(formatted.shape[1]) != int(self.out_size)
        ):
            raise ValueError(
                f'{self.name}: renderer returned {formatted.shape}, expected '
                f'({int(self.out_size)}, {int(self.out_size)}, {int(self.channel_count)})'
            )
        return formatted

    def _get_real_frame(self, idx: int) -> np.ndarray:
        idx_i = int(np.clip(int(idx), 0, max(0, int(self.nf) - 1)))
        self._ensure_submitted(idx_i + 1)
        with self._lock:
            fut = self._futures.get(idx_i)
        if fut is None:
            # Should only happen if close raced with iteration; render synchronously so the
            # YOLO stream can still finish deterministically.
            return self._render_one(idx_i)
        frame = fut.result()
        with self._lock:
            self._futures.pop(idx_i, None)
        return frame

    def __next__(self) -> Tuple[List[str], List[np.ndarray], List[str]]:
        if self.count >= self.yield_nf or self.nf <= 0:
            self.close()
            raise StopIteration
        self.start()
        start = int(self.count)
        stop = min(int(self.yield_nf), start + int(self.bs))
        # Ensure the actual batch is ready or rendering before waiting for ordered frames.
        self._ensure_submitted(min(int(stop), int(self.nf)))
        paths: List[str] = []
        imgs: List[np.ndarray] = ChannelFormattedYoloBatch(channel_count=int(self.channel_count))
        info: List[str] = []
        last_real_idx = max(0, int(self.nf) - 1)
        for idx in range(start, stop):
            real_idx = int(idx) if int(idx) < int(self.nf) else int(last_real_idx)
            synthetic = int(idx) >= int(self.nf)
            suffix = '_synthetic' if synthetic else ''
            frame = self._get_real_frame(real_idx)
            paths.append(f'{self.name}_{idx + 1:06d}{suffix}.png')
            imgs.append(self._frame_to_model_channels(frame, int(self.channel_count)))
            if synthetic:
                info.append(f'streaming {self.name} synthetic padded slice {idx + 1}/{self.yield_nf} repeats real slice {real_idx + 1}/{self.nf}: ')
            else:
                info.append(f'streaming {self.name} slice {idx + 1}/{self.nf}: ')
        self.count = int(stop)
        with self._lock:
            self._fill_prefetch_locked(target_index=int(self.count))
        return paths, imgs, info

class GpuPrefetchedYoloBatch(list):
    """List-like orig-image batch whose YOLO tensor is already staged in GPU VRAM.

 Ultralytics still receives a list of H×W×C uint8 images for result-shape bookkeeping,
 while the patched preprocess path consumes ``_tta_gpu_tensor`` directly. The tensor is
 BCHW, normalized to [0,1], and already on the requested CUDA device. A CUDA event is
 attached when staging used a non-default copy stream so the predictor stream can wait
 without forcing a CPU synchronization."""

    def __init__(
        self,
        frames: Sequence[np.ndarray],
        *,
        gpu_tensor: object,
        ready_event: Optional[object] = None,
        source_label: str = '',
        cpu_tensor_ref: Optional[object] = None,
    ) -> None:
        super().__init__(frames)
        self._tta_gpu_tensor = gpu_tensor
        self._tta_gpu_ready_event = ready_event
        self._tta_source_label = str(source_label)
        self._tta_cpu_tensor_ref = cpu_tensor_ref
        self._tta_normalized = True

    def wait_ready(self) -> object:
        tensor = self._tta_gpu_tensor
        event = self._tta_gpu_ready_event
        if event is not None:
            try:
                import torch  # type: ignore
                current_stream = torch.cuda.current_stream(device=tensor.device)  # type: ignore[attr-defined]
                current_stream.wait_event(event)
                record_stream = getattr(tensor, 'record_stream', None)
                if callable(record_stream):
                    record_stream(current_stream)
            except Exception:
                try:
                    synchronize = getattr(event, 'synchronize', None)
                    if callable(synchronize):
                        synchronize()
                    self._tta_cpu_tensor_ref = None
                except Exception:
                    pass
        return tensor

class GpuPrefetchingYoloSource:
    """Wrap a YOLO source and keep a bounded queue of preprocessed CUDA input batches.

 This queue is intentionally separate from Ultralytics' ``--batch``. ``--batch`` remains the
 number of slices in each inference call; ``queue_batches`` controls how many complete batches
 are staged in VRAM ahead of the predictor loop."""

    def __init__(
        self,
        base_source: object,
        *,
        cfg: 'PredictConfig',
        source_label: str,
        queue_batches: int = 8,
        pin_memory: bool = True,
        staging_reservation_bytes: int = 0,
    ) -> None:
        self.base_source = base_source
        self.cfg = cfg
        self.source_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(source_label)).strip('_') or 'gpu_prefetch_source'
        self.queue_batches = max(1, int(queue_batches))
        self.pin_memory = bool(pin_memory)
        self.mode = getattr(base_source, 'mode', 'image')
        self.channel_format = resolve_channel_format(
            getattr(base_source, 'channel_format', DEFAULT_CHANNEL_FORMAT)
        )
        self.channel_count = int(getattr(base_source, 'channel_count', self.channel_format.channel_count))
        if int(self.channel_count) != int(cfg.input_channels):
            raise ValueError(
                f'GPU input staging source has {int(self.channel_count)} channel(s), but '
                f'PredictConfig requires {int(cfg.input_channels)}'
            )
        self.count = 0
        self.bs = max(1, int(getattr(base_source, 'bs', getattr(cfg, 'batch', 1))))
        self.nf = int(getattr(base_source, 'nf', 0) or 0)
        self.yield_nf = int(getattr(base_source, 'yield_nf', 0) or 0)
        self.synthetic_count = int(getattr(base_source, 'synthetic_count', max(0, self.yield_nf - self.nf)))
        self.source_type = getattr(base_source, 'source_type', argparse.Namespace(stream=False, screenshot=False, from_img=True, tensor=False))
        self._queue: 'queue.Queue[object]' = queue.Queue(maxsize=self.queue_batches)
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False
        self._sentinel = object()
        self._copy_stream: Optional[object] = None
        self._device_str = canonical_single_device(str(cfg.device))
        self._dtype_name = 'float16' if quantize_uses_fp16(cfg.quantize) and str(self._device_str).startswith('cuda') else 'float32'
        # ring of reusable PINNED u8 BCHW staging buffers ([tensor, reuse_event]
        # pairs). The producer loop is single-threaded, so a small ring with per-buffer H2D
        # completion events is race-free.
        self._pinned_ring: Optional[List[List[object]]] = None
        self._pinned_ring_next = 0
        # VRAM admitted for this source by gpu_input_staging_preflight_reserve;
        # handed back to the ledger as staged batches materialize, remainder released on close.
        self._staging_reservation_remaining = max(0, int(staging_reservation_bytes))

    def _consume_staging_reservation(self, staged_bytes: int) -> None:
        step_cap = max(0, int(staged_bytes))
        if step_cap <= 0:
            return
        with _GPU_STAGING_RESERVATION_LOCK:
            remaining = int(self._staging_reservation_remaining)
            step = min(remaining, step_cap)
            if step <= 0:
                return
            self._staging_reservation_remaining = remaining - step
            key = str(self._device_str)
            ledger = int(_GPU_STAGING_RESERVED_BYTES.get(key, 0)) - step
            if ledger > 0:
                _GPU_STAGING_RESERVED_BYTES[key] = ledger
            else:
                _GPU_STAGING_RESERVED_BYTES.pop(key, None)

    def __len__(self) -> int:
        try:
            return int(len(self.base_source))  # type: ignore[arg-type]
        except Exception:
            if self.yield_nf > 0 and self.bs > 0:
                return int(math.ceil(float(self.yield_nf) / float(self.bs)))
            if self.nf > 0 and self.bs > 0:
                return int(math.ceil(float(self.nf) / float(self.bs)))
            return 0

    def __iter__(self) -> 'GpuPrefetchingYoloSource':
        self.start()
        return self

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name=f'yolo-gpu-stage-{self.source_label[:32]}',
            daemon=True,
        )
        self._producer_thread.start()

    def close(self) -> None:
        self._closed = True
        self._stop_event.set()
        close_fn = getattr(self.base_source, 'close', None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        thread = self._producer_thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass
        # drop staged batches that were still queued when iteration
        # stopped; abandoned GpuPrefetchedYoloBatch items otherwise keep their CUDA
        # tensors (and pinned CPU tensors) alive for the life of the process.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        # release any admission reservation this source never materialized.
        try:
            self._consume_staging_reservation(int(self._staging_reservation_remaining))
        except Exception:
            pass

    @staticmethod
    def _normalize_model_channel_frames(
        imgs: Sequence[np.ndarray],
        channel_count: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return contiguous H×W×C frames and their result-bookkeeping images."""
        frames: List[np.ndarray] = []
        for item in imgs:
            frames.append(InMemoryYoloVolumeSource._frame_to_model_channels(
                np.asarray(item), int(channel_count)
            ))
        shapes = {tuple(int(v) for v in frame.shape[:2]) for frame in frames}
        if len(shapes) != 1:
            raise ValueError(f'GPU input staging batch contains mixed shapes: {sorted(shapes)}')
        return frames, frames

    @staticmethod
    def _stack_model_channel_cpu_tensor(
        imgs: Sequence[np.ndarray],
        channel_count: int,
    ) -> Tuple[object, List[np.ndarray]]:
        frames, cpu_frames = GpuPrefetchingYoloSource._normalize_model_channel_frames(
            imgs, int(channel_count)
        )
        import torch  # type: ignore
        batch = np.moveaxis(np.stack(frames, axis=0), -1, 1)  # BHWC -> BCHW
        return torch.from_numpy(np.ascontiguousarray(batch)), cpu_frames

    def _acquire_pinned_staging_entry(
        self,
        torch_mod: object,
        b: int,
        c: int,
        h: int,
        w: int,
    ) -> List[object]:
        """Return the next reusable pinned-u8 entry after its prior H2D completes."""
        want = (max(int(self.bs), int(b)), int(c), int(h), int(w))
        ring = self._pinned_ring
        if ring is None or tuple(int(x) for x in ring[0][0].shape) != want:
            ring = [
                [torch_mod.empty(want, dtype=torch_mod.uint8, pin_memory=True), None]
                for _ in range(3)
            ]
            self._pinned_ring = ring
            self._pinned_ring_next = 0
        entry = ring[self._pinned_ring_next]
        self._pinned_ring_next = (self._pinned_ring_next + 1) % len(ring)
        evt = entry[1]
        if evt is not None:
            evt.synchronize()  # type: ignore[attr-defined]
            entry[1] = None
        return entry

    def _stage_batch(self, batch: object) -> object:
        paths, imgs, info = batch  # type: ignore[misc]
        import torch  # type: ignore
        if not str(self._device_str).startswith('cuda'):
            return batch
        if not bool(torch.cuda.is_available()):
            return batch

        target_device = torch.device(self._device_str)
        target_dtype = torch.float16 if self._dtype_name == 'float16' else torch.float32
        ready_event: Optional[object] = None
        gpu_tensor: Optional[object] = None
        cpu_tensor_ref: Optional[object] = None
        frames, cpu_frames = self._normalize_model_channel_frames(
            list(imgs), int(self.channel_count)
        )

        if bool(self.pin_memory):
            try:
                # frames go straight into a pooled PINNED u8 BCHW buffer (the
                # np.stack + pin_memory double copy is gone), the wire carries uint8 (half
                # the bytes of fp16), and the dtype cast + /255 normalization run ON DEVICE on
                # the copy stream. The old mixed.to(device, dtype) materialized the cast on
                # the CPU into an UNPINNED temp — which also degraded non_blocking to a
                # pageable, producer-blocking copy.
                b = len(frames)
                h, w, c = (int(v) for v in frames[0].shape)
                entry = self._acquire_pinned_staging_entry(torch, b, c, h, w)
                pin_view = entry[0][:b]
                pin_np = pin_view.numpy()
                for i, frame in enumerate(frames):
                    np.copyto(pin_np[i], np.moveaxis(frame, -1, 0))
                if self._copy_stream is None:
                    self._copy_stream = torch.cuda.Stream(device=target_device)
                with torch.cuda.stream(self._copy_stream):
                    gpu_u8 = pin_view.to(device=target_device, non_blocking=True)
                    reuse_event = torch.cuda.Event()
                    reuse_event.record(self._copy_stream)
                    entry[1] = reuse_event  # buffer reusable once the u8 H2D completes
                    gpu_tensor = gpu_u8.to(target_dtype)
                    gpu_tensor.div_(255.0)
                    ready_event = torch.cuda.Event()
                    ready_event.record(self._copy_stream)
            except Exception:
                gpu_tensor = None
                ready_event = None

        if gpu_tensor is None:
            # Fallback (pinning unavailable or the pooled path failed): legacy stack + mixed
            # .to; still removes the RAM->GPU copy from the predictor hot path.
            cpu_tensor, cpu_frames = self._stack_model_channel_cpu_tensor(
                list(imgs), int(self.channel_count)
            )
            if bool(self.pin_memory):
                try:
                    cpu_tensor = cpu_tensor.pin_memory()
                except Exception:
                    pass
            try:
                if self._copy_stream is None:
                    self._copy_stream = torch.cuda.Stream(device=target_device)
                with torch.cuda.stream(self._copy_stream):
                    gpu_tensor = cpu_tensor.to(device=target_device, dtype=target_dtype, non_blocking=True)
                    gpu_tensor.div_(255.0)
                    ready_event = torch.cuda.Event()
                    ready_event.record(self._copy_stream)
            except Exception:
                gpu_tensor = cpu_tensor.to(device=target_device, dtype=target_dtype, non_blocking=False)
                gpu_tensor.div_(255.0)
                ready_event = None
            cpu_tensor_ref = cpu_tensor

        gpu_batch = GpuPrefetchedYoloBatch(
            cpu_frames,
            gpu_tensor=gpu_tensor,
            ready_event=ready_event,
            source_label=self.source_label,
            cpu_tensor_ref=cpu_tensor_ref,
        )
        # This batch's VRAM is now visible to mem_get_info; stop double-counting it in
        # the admission ledger.
        try:
            self._consume_staging_reservation(int(gpu_tensor.numel()) * int(gpu_tensor.element_size()))
        except Exception:
            pass
        return paths, gpu_batch, info

    def _put_queue_item(self, item: object) -> None:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def _producer_loop(self) -> None:
        try:
            for batch in self.base_source:  # type: ignore[operator]
                if self._stop_event.is_set():
                    break
                self._put_queue_item(self._stage_batch(batch))
            self._put_queue_item(self._sentinel)
        except BaseException as exc:
            self._put_queue_item(exc)
            self._put_queue_item(self._sentinel)

    def __next__(self) -> Tuple[List[str], object, List[str]]:
        self.start()
        while True:
            if self._closed and self._queue.empty():
                raise StopIteration
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is self._sentinel:
                self.close()
                raise StopIteration
            if isinstance(item, BaseException):
                self.close()
                raise item
            self.count += 1
            return item  # type: ignore[return-value]

def gpu_input_staging_enabled(cfg: 'PredictConfig') -> bool:
    """Return True when a small queue of YOLO input batches should be staged in VRAM."""
    if not _env_flag('YOLO_TTA_GPU_INPUT_STAGING', True):
        return False
    target = canonical_single_device(str(cfg.device))
    if not str(target).startswith('cuda'):
        return False
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def gpu_input_staging_queue_batches() -> int:
    return max(0, _env_int('YOLO_TTA_GPU_INPUT_STAGING_BATCHES', 8))

def gpu_input_staging_pin_memory_enabled() -> bool:
    return _env_flag('YOLO_TTA_GPU_INPUT_STAGING_PIN_MEMORY', True)

def gpu_input_staging_min_free_after_bytes() -> int:
    return int(max(0.0, _env_float('YOLO_TTA_GPU_INPUT_STAGING_MIN_FREE_AFTER_GIB', 2.0)) * GIB)

def _source_prediction_out_size(source: object) -> Optional[int]:
    out_size = getattr(source, 'out_size', None)
    if out_size is not None:
        try:
            return int(out_size)
        except Exception:
            pass
    volume = getattr(source, 'volume_gray', None)
    if volume is not None:
        try:
            arr = np.asarray(volume)
            if arr.ndim >= 3:
                return int(arr.shape[1])
        except Exception:
            pass
    return None

def _source_prediction_channel_count(source: object, cfg: Optional['PredictConfig'] = None) -> int:
    value = getattr(source, 'channel_count', None)
    if value is not None:
        try:
            return max(1, int(value))
        except Exception:
            pass
    volume = getattr(source, 'volume_gray', None)
    if volume is not None:
        try:
            arr = np.asarray(volume)
            return int(arr.shape[3]) if arr.ndim == 4 else 1
        except Exception:
            pass
    if cfg is not None:
        return max(1, int(getattr(cfg, 'input_channels', 1)))
    return 1

_GPU_STAGING_RESERVATION_LOCK = threading.Lock()

_GPU_STAGING_RESERVED_BYTES: Dict[str, int] = {}

def _release_gpu_staging_reservation(device_str: str, release_bytes: int) -> None:
    release = max(0, int(release_bytes))
    if release <= 0:
        return
    key = str(device_str)
    with _GPU_STAGING_RESERVATION_LOCK:
        remaining = int(_GPU_STAGING_RESERVED_BYTES.get(key, 0)) - release
        if remaining > 0:
            _GPU_STAGING_RESERVED_BYTES[key] = remaining
        else:
            _GPU_STAGING_RESERVED_BYTES.pop(key, None)

def gpu_input_staging_preflight_reserve(source: object, cfg: 'PredictConfig', queue_batches: int, source_label: str) -> Optional[int]:
    """Admit one source into CUDA input staging, reserving its VRAM need in the ledger.

 Returns the reserved byte count (0 when the need cannot be estimated) or None when
 staging must be skipped for this source."""
    out_size = _source_prediction_out_size(source)
    if out_size is None or int(out_size) <= 0:
        return 0
    try:
        import torch  # type: ignore
        device_str = canonical_single_device(str(cfg.device))
        device = torch.device(device_str)
        try:
            with torch.cuda.device(device):
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
        except Exception:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
        dtype_bytes = 2 if quantize_uses_fp16(cfg.quantize) else 4
        frame_slots = int(max(1, queue_batches)) * int(max(1, cfg.batch))
        channel_count = _source_prediction_channel_count(source, cfg)
        need_bytes = (
            int(frame_slots)
            * int(channel_count)
            * int(out_size)
            * int(out_size)
            * int(dtype_bytes)
        )
        reserve_bytes = int(gpu_input_staging_min_free_after_bytes())
        with _GPU_STAGING_RESERVATION_LOCK:
            already_reserved = int(_GPU_STAGING_RESERVED_BYTES.get(str(device_str), 0))
            admitted = int(free_bytes) - already_reserved >= int(need_bytes) + int(reserve_bytes)
            if admitted:
                _GPU_STAGING_RESERVED_BYTES[str(device_str)] = already_reserved + int(need_bytes)
        if not admitted:
            print(
                f'CUDA input staging skipped for {source_label}: need≈{need_bytes / GIB:.2f} GiB '
                f'for {int(queue_batches)} queued batch(es) at --batch={int(cfg.batch)}, imgsz={int(out_size)}, '
                f'channels={int(channel_count)}, dtype_bytes={int(dtype_bytes)}; '
                f'free={int(free_bytes) / GIB:.2f} GiB, '
                f'pending_reservations={already_reserved / GIB:.2f} GiB, '
                f'reserve={int(reserve_bytes) / GIB:.2f} GiB.'
            )
            return None
        return int(need_bytes)
    except Exception as exc:
        print(f'CUDA input staging preflight failed for {source_label} ({exc}); using CPU source path.')
        return None

def maybe_wrap_source_with_gpu_input_staging(source: object, cfg: 'PredictConfig', source_label: str) -> object:
    if isinstance(source, GpuPrefetchingYoloSource):
        return source
    # GPU-rendered sources already produce device-resident normalized batches;
    # CPU-side staging would be a pointless host round trip.
    if isinstance(source, (GpuRenderedYoloSource, GpuTileRenderedYoloSource)):
        return source
    if not gpu_input_staging_enabled(cfg):
        return source
    queue_batches = gpu_input_staging_queue_batches()
    if int(queue_batches) <= 0:
        return source
    staging_reservation = gpu_input_staging_preflight_reserve(source, cfg, int(queue_batches), str(source_label))
    if staging_reservation is None:
        return source
    try:
        ensure_ultralytics_accepts_in_memory_volume_source()
        print(
            f'CUDA input staging active for {source_label}: '
            f'queue_batches={int(queue_batches)} ({int(queue_batches) * max(1, int(cfg.batch))} frame slots), '
            f'--batch={int(cfg.batch)}, device={canonical_single_device(str(cfg.device))}, '
            f'channels={int(_source_prediction_channel_count(source, cfg))}, '
            f'dtype={"float16" if quantize_uses_fp16(cfg.quantize) else "float32"}'
        )
        return GpuPrefetchingYoloSource(
            source,
            cfg=cfg,
            source_label=str(source_label),
            queue_batches=int(queue_batches),
            pin_memory=gpu_input_staging_pin_memory_enabled(),
            staging_reservation_bytes=int(staging_reservation),
        )
    except BaseException:
        # No wrapper owns the reservation yet — credit it back or the phantom pending
        # bytes silently veto every later staging preflight on this device.
        _release_gpu_staging_reservation(canonical_single_device(str(cfg.device)), int(staging_reservation))
        raise

def gpu_input_staging_ahead_sources(default_queue_slots: int) -> int:
    """Resolve how many queued prediction sources may pre-stage CUDA batches.
    
    This source count is independent of the per-source batch queue and is still bounded by VRAM admission."""
    raw = os.environ.get('YOLO_TTA_GPU_INPUT_STAGING_AHEAD_SOURCES', '').strip()
    if raw:
        try:
            return max(0, int(raw))
        except Exception:
            return 0
    default_ahead = max(1, _env_int('YOLO_TTA_GPU_INPUT_STAGING_AHEAD_DEFAULT', max(1, _cpu_count())))
    return max(0, min(max(1, int(default_queue_slots)), int(default_ahead)))

def _prediction_ref_has_gpu_input_staging(ref: PredictionVolumeRef) -> bool:
    return isinstance(getattr(ref, 'source', None), GpuPrefetchingYoloSource)

def maybe_eager_stage_prediction_ref_on_gpu(prediction_ref: PredictionVolumeRef, cfg: 'PredictConfig') -> PredictionVolumeRef:
    """Wrap and start one queued prediction source's CUDA input staging queue.

 The ordinary predict path also wraps sources, but only after they become the
 active source. Eager staging lets queued sources render and copy their first
 batches into VRAM while the GPU is still inferencing the previous source."""
    if _prediction_ref_has_gpu_input_staging(prediction_ref):
        source = prediction_ref.source
    else:
        source = getattr(prediction_ref, 'source', None)
        if source is None:
            if prediction_ref.array is None:
                return prediction_ref
            source = make_in_memory_yolo_source(
                prediction_ref.array,
                prediction_ref.name,
                batch_size=max(1, int(cfg.batch)),
                max_frames=None,
                channel_format=resolve_channel_format(prediction_ref.channel_format),
            )
        wrapped = maybe_wrap_source_with_gpu_input_staging(source, cfg, prediction_ref.name)
        prediction_ref.source = wrapped
        source = wrapped
    start_fn = getattr(source, 'start', None)
    if callable(start_fn):
        try:
            start_fn()
        except Exception as exc:
            print(f'Warning: eager CUDA input staging could not start for {prediction_ref.name} ({exc}); source will stage on demand.')
    return prediction_ref

def streaming_prediction_sources_enabled() -> bool:
    """Return True when YOLO input slices should be rendered lazily instead of prebuilt."""
    return _env_flag('YOLO_TTA_STREAMING_PREDICTION_SOURCES', True)

def streaming_prediction_source_autostart_enabled() -> bool:
    """Return whether each new streaming source starts prefetch immediately."""
    return _env_flag('YOLO_TTA_STREAMING_SOURCE_AUTOSTART', False)

def queued_streaming_source_cpu_warmup_slots(default_queue_slots: int) -> int:
    """Resolve how many ready queued streaming sources may pre-render on CPU.

 This keeps source creation unbounded without allowing every future tile/view to enqueue
 a full prefetch window at once. The active source can still use the full shared render pool;
 a bounded number of upcoming sources are kept warm to overlap CPU rendering with inference."""
    raw = os.environ.get('YOLO_TTA_STREAMING_SOURCE_WARMUP_SOURCES', '').strip()
    if raw:
        try:
            return max(0, int(raw))
        except Exception:
            return 0
    default_slots = max(1, _env_int('YOLO_TTA_STREAMING_SOURCE_WARMUP_DEFAULT', max(2, min(8, _cpu_count()))))
    return max(0, min(max(1, int(default_queue_slots)), int(default_slots)))

def streaming_prediction_source_prefetch_frames(batch_size: int) -> int:
    explicit = _env_int('YOLO_TTA_STREAMING_SOURCE_PREFETCH_FRAMES', 0)
    if explicit > 0:
        return max(1, int(explicit))
    batches = max(1, _env_int('YOLO_TTA_STREAMING_SOURCE_PREFETCH_BATCHES', 32))
    max_frames = max(1, _env_int('YOLO_TTA_STREAMING_SOURCE_PREFETCH_MAX_FRAMES', 2048))
    return max(1, min(int(max_frames), max(int(batch_size), int(batch_size) * int(batches))))

def streaming_prediction_source_workers(default_workers: int, num_frames: int) -> int:
    """Resolve the render-worker budget for one streaming source, capped by its frame count."""
    full_cpu_default = max(1, _env_int('YOLO_TTA_STREAMING_SOURCE_DEFAULT_WORKERS', max(1, _cpu_count())))
    min_workers = max(1, _env_int('YOLO_TTA_STREAMING_SOURCE_MIN_WORKERS', full_cpu_default))
    default_resolved = max(1, int(default_workers), int(min_workers), int(full_cpu_default))
    requested = _env_int('YOLO_TTA_STREAMING_SOURCE_WORKERS', int(default_resolved))
    return choose_slice_parallel_workers(max(1, int(requested)), max(1, int(num_frames)))

def shared_streaming_render_pool_enabled() -> bool:
    """Return True when all streaming YOLO sources should share one CPU render pool."""
    return _env_flag('YOLO_TTA_SHARED_STREAMING_RENDER_POOL', True)

def resolve_prediction_render_workers(default_workers: int, max_frames: int) -> int:
    requested = _env_int('YOLO_TTA_PREDICTION_RENDER_WORKERS', int(default_workers))
    return choose_slice_parallel_workers(max(1, int(requested)), max(1, int(max_frames)))

def resolve_prediction_source_queue_slots(total_tasks: int, *, streaming_sources: bool = True) -> int:
    """Resolve how many future prediction sources may be constructed.
    
    Streaming refs default to all tasks; dense materialization uses a CPU-scaled cap unless explicitly overridden."""
    total = max(1, int(total_tasks))
    raw = os.environ.get('YOLO_TTA_PREDICTION_VOLUME_QUEUE_SLOTS', '').strip()
    if raw:
        try:
            requested = int(raw)
        except Exception:
            requested = 0
        if int(requested) <= 0:
            return int(total)
        return max(1, min(int(total), int(requested)))
    if bool(streaming_sources):
        return int(total)
    default_slots = max(8, min(int(total), max(1, int(_cpu_count()))))
    return int(default_slots)

def ensure_ultralytics_accepts_in_memory_volume_source() -> None:
    """Register the in-memory volume loader with Ultralytics' source checker."""
    try:
        import ultralytics.data.build as ultralytics_build  # type: ignore
    except Exception as exc:  # pragma: no cover - ultralytics is imported lazily on SLURM
        raise RuntimeError(f'Unable to import ultralytics.data.build for in-memory prediction source registration: {exc}') from exc

    loaders = getattr(ultralytics_build, 'LOADERS', ())
    try:
        loaders_tuple = tuple(loaders)
    except Exception:
        loaders_tuple = ()
    additions: List[object] = []
    for loader_cls in (
        InMemoryYoloVolumeSource, StreamingYoloVolumeSource, GpuPrefetchingYoloSource,
        GpuRenderedYoloSource, GpuTileRenderedYoloSource,
    ):
        if loader_cls not in loaders_tuple:
            additions.append(loader_cls)
    if additions:
        setattr(ultralytics_build, 'LOADERS', loaders_tuple + tuple(additions))

def make_in_memory_yolo_source(
    volume_gray: np.ndarray,
    name: str,
    *,
    batch_size: int = 1,
    max_frames: Optional[int] = None,
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
) -> InMemoryYoloVolumeSource:
    ensure_ultralytics_accepts_in_memory_volume_source()
    return InMemoryYoloVolumeSource(
        volume_gray,
        name=name,
        batch_size=max(1, int(batch_size)),
        max_frames=max_frames,
        channel_format=resolve_channel_format(channel_format),
    )

def make_prediction_ref_yolo_source(
    prediction_volume: PredictionVolumeRef,
    *,
    batch_size: int = 1,
    max_frames: Optional[int] = None,
) -> object:
    ensure_ultralytics_accepts_in_memory_volume_source()
    source = getattr(prediction_volume, 'source', None)
    if source is not None:
        return source
    if prediction_volume.array is None:
        raise ValueError(f'Prediction input {prediction_volume.name} has neither source nor array')
    return make_in_memory_yolo_source(
        prediction_volume.array,
        prediction_volume.name,
        batch_size=max(1, int(batch_size)),
        max_frames=max_frames,
        channel_format=resolve_channel_format(prediction_volume.channel_format),
    )

def _center_preserving_scale_matrix(src_w: int, src_h: int, out_w: int, out_h: int) -> np.ndarray:
    cx_src = (int(src_w) - 1) / 2.0
    cy_src = (int(src_h) - 1) / 2.0
    cx_out = (int(out_w) - 1) / 2.0
    cy_out = (int(out_h) - 1) / 2.0
    sx = float(out_w) / float(src_w)
    sy = float(out_h) / float(src_h)
    return np.array(
        [
            [sx, 0.0, cx_out - sx * cx_src],
            [0.0, sy, cy_out - sy * cy_src],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

def dense_tile_positions(length: int, tile_size: int, stride: int) -> List[int]:
    length = int(length)
    tile_size = int(tile_size)
    stride = int(stride)
    if tile_size <= 0:
        return []
    if stride <= 0:
        raise ValueError('tile stride must be > 0 when dense tiling is enabled')

    last = max(0, length - tile_size)
    starts = list(range(0, last + 1, stride)) if last > 0 else [0]
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return [int(x) for x in starts]

def build_dense_tile_jobs_for_aug(
    view: ViewInfo,
    aug_job: AugJob,
    tile_cfg: TileConfig,
    out_size: int,
    temp_dir: Path,
) -> List[DenseTileJob]:
    # one consolidated diagnostic sidecar per (view, configuration) instead of one
    # indented JSON file per tile. See write_dense_tile_job_meta.
    tile_meta_path = temp_dir / 'tiles' / view.name / tile_cfg.config_id / 'tiles.jsonl'
    xs = dense_tile_positions(int(aug_job.aff.canvas_w), int(tile_cfg.tile_size), int(tile_cfg.tile_stride))
    ys = dense_tile_positions(int(aug_job.aff.canvas_h), int(tile_cfg.tile_size), int(tile_cfg.tile_stride))

    M_src_to_canvas3 = _affine2x3_to_3x3(aug_job.aff.M_src_to_canvas)
    # Depends only on tile_size/out_size, so it is the same matrix for every tile of this
    # configuration — build it once instead of once per tile.
    M_scale = _center_preserving_scale_matrix(
        int(tile_cfg.tile_size), int(tile_cfg.tile_size), int(out_size), int(out_size),
    )

    tile_specs: List[Tuple[str, int, int]] = []
    forward_mats: List[np.ndarray] = []
    for tile_y in ys:
        for tile_x in xs:
            tile_id = f'{tile_cfg.config_id}_{aug_job.aug_id}_x{int(tile_x):04d}_y{int(tile_y):04d}'
            M_crop = np.array(
                [
                    [1.0, 0.0, -float(tile_x)],
                    [0.0, 1.0, -float(tile_y)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            forward_mats.append(M_scale @ M_crop @ M_src_to_canvas3)
            tile_specs.append((tile_id, int(tile_x), int(tile_y)))

    if not tile_specs:
        return []

    # one batched inversion over the whole (N, 3, 3) stack rather than a np.linalg.inv
    # call per tile.
    M_src_to_out_stack = np.stack(forward_mats, axis=0)
    M_out_to_src_stack = np.linalg.inv(M_src_to_out_stack)

    jobs: List[DenseTileJob] = [
        DenseTileJob(
            view=view.name,
            aug_id=aug_job.aug_id,
            config_id=tile_cfg.config_id,
            tile_id=tile_id,
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            tile_size=int(tile_cfg.tile_size),
            tile_stride=int(tile_cfg.tile_stride),
            out_size=int(out_size),
            meta_path=tile_meta_path,
            M_out_to_src=M_out_to_src_stack[i][:2, :3].astype(np.float32),
            M_src_to_out=M_src_to_out_stack[i][:2, :3].astype(np.float32),
        )
        for i, (tile_id, tile_x, tile_y) in enumerate(tile_specs)
    ]

    # v16.4.0 gates every tile independently, so each tile keeps only its own minimal
    # parent-grid footprint. The retired grouped/configuration-canvas path required a
    # configuration-wide uniform crop and no longer exists.
    if jobs:
        resolved: List[DenseTileJob] = []
        for job in jobs:
            window, M_out_to_crop = tile_parent_crop_window(
                view, job.M_out_to_src, int(out_size),
            )
            resolved.append(dataclasses_replace(job, parent_crop=window, M_out_to_crop=M_out_to_crop))
        jobs = resolved
    return jobs

_TILE_META_STREAM_LOCK = threading.Lock()

_TILE_META_STREAM_WRITTEN: Dict[str, set[str]] = {}

def write_dense_tile_job_meta(
    job: DenseTileJob,
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
) -> None:
    """Append one unique tile record to the per-view/configuration JSONL sidecar.
    
    A process-local lock and tile-id set make repeated call sites idempotent."""
    fmt = resolve_channel_format(channel_format)
    path = Path(job.meta_path)
    key = str(path)
    record = json.dumps(
        {
            'view': job.view,
            'physical_view': str(job.view).split('__tta_', 1)[0],
            'aug_id': job.aug_id,
            'config_id': job.config_id,
            'tile_id': job.tile_id,
            'tile_x': int(job.tile_x),
            'tile_y': int(job.tile_y),
            'tile_size': int(job.tile_size),
            'tile_stride': int(job.tile_stride),
            'out_size': int(job.out_size),
            'M_out_to_src': job.M_out_to_src.tolist(),
            'M_src_to_out': job.M_src_to_out.tolist(),
            'channel_format': fmt.token,
            'model_input_channels': int(fmt.channel_count),
            'channel_stride': int(fmt.stride),
            'channel_offsets': [int(v) for v in fmt.offsets],
            'channel_boundary_policy': 'radial_wrap_mirror_u_cartesian_edge_clamp',
            'prediction_slice_policy': 'center_N_only',
        },
        separators=(',', ':'),
    )
    with _TILE_META_STREAM_LOCK:
        written = _TILE_META_STREAM_WRITTEN.get(key)
        if written is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Drop any stream left behind by an earlier run so the sidecar describes this one.
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            written = set()
            _TILE_META_STREAM_WRITTEN[key] = written
        tile_key = str(job.tile_id)
        if tile_key in written:
            return
        written.add(tile_key)
        try:
            with path.open('a', encoding='utf-8') as handle:
                handle.write(record + '\n')
        except Exception as exc:
            print(f'Warning: could not append tile metadata for {job.tile_id} to {path} ({exc})')

@dataclass(frozen=True)
class TiltedRenderPlan:
    x_idx: np.ndarray
    y_idx: np.ndarray
    valid_xy: np.ndarray
    axis_offset: np.ndarray
    # frame-invariant hoists. The tilted frame center is always an integer, so
    # floor/frac of the sheared stack coordinate, the flat in-plane gather index, and each
    # pixel's valid frame-center interval [c_lo, c_hi] are all constant across frames — only a
    # scalar integer offset varies per frame. row_* fields describe the separable fast path
    # (per-row constant stack coordinate + contiguous in-plane row reads); flat/2D fields back
    # the generic single-flat-gather path. Exactly one of the two families is populated.
    stack_stride: int = 0
    row_c_lo: Optional[np.ndarray] = None      # int32 (H,): min valid frame center per row (sentinel +INT when row empty)
    row_c_hi: Optional[np.ndarray] = None      # int32 (H,): max valid frame center per row (sentinel -INT)
    row_fast: bool = False
    row_sb0: Optional[np.ndarray] = None       # int32 (H,): per-row floor(tan*offset)
    row_alpha: Optional[np.ndarray] = None     # float32 (H,): per-row frac(tan*offset)
    row_v: Optional[np.ndarray] = None         # int32 (H,): per-row fixed in-plane index
    row_u0: Optional[np.ndarray] = None        # int32 (H,): first valid output column
    row_u1: Optional[np.ndarray] = None        # int32 (H,): one past last valid output column
    row_su0: Optional[np.ndarray] = None       # int32 (H,): source u of the first valid column
    flat_base: Optional[np.ndarray] = None     # int64 (H,W): in-plane flat index + sb0*stack_stride
    alpha2d: Optional[np.ndarray] = None       # float32 (H,W)
    om_alpha2d: Optional[np.ndarray] = None    # float32 (H,W): 1 - alpha (kept so the lerp matches the legacy fp32 ops bit-for-bit)
    c_lo2d: Optional[np.ndarray] = None        # int32 (H,W)
    c_hi2d: Optional[np.ndarray] = None        # int32 (H,W)

_TILTED_RENDER_PLAN_CACHE: 'OrderedDict[Tuple[str, int, int, Tuple[float, ...]], TiltedRenderPlan]' = OrderedDict()

_TILTED_RENDER_PLAN_CACHE_BYTES = 0

_TILTED_RENDER_PLAN_CACHE_LOCK = threading.Lock()

_TILTED_RENDER_PLAN_BUILDS_IN_FLIGHT: Dict[Tuple[str, int, int, Tuple[float, ...]], threading.Event] = {}

def tilted_render_plan_cache_max_bytes() -> int:
    return int(max(0.0, _env_float('YOLO_TTA_TILTED_PLAN_CACHE_GIB', 4.0)) * GIB)

def _tilted_render_plan_nbytes(plan: TiltedRenderPlan) -> int:
    total = int(plan.x_idx.nbytes) + int(plan.y_idx.nbytes) + int(plan.valid_xy.nbytes) + int(plan.axis_offset.nbytes)
    # the hoisted frame-invariant arrays count against the LRU byte budget too.
    for extra in (
        plan.row_c_lo, plan.row_c_hi, plan.row_sb0, plan.row_alpha, plan.row_v,
        plan.row_u0, plan.row_u1, plan.row_su0,
        plan.flat_base, plan.alpha2d, plan.om_alpha2d, plan.c_lo2d, plan.c_hi2d,
    ):
        if extra is not None:
            total += int(extra.nbytes)
    return total

def _tilted_plan_cache_key(view: ViewInfo, M_grid_to_src: np.ndarray, grid_h: int, grid_w: int) -> Tuple[str, int, int, Tuple[float, ...]]:
    mat = tuple(round(float(x), 6) for x in np.asarray(M_grid_to_src, dtype=np.float32).reshape(-1).tolist())
    return (str(view.name), int(grid_h), int(grid_w), mat)

def get_tilted_render_plan(
    view: ViewInfo,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> TiltedRenderPlan:
    if not is_tilted_view(view):
        raise ValueError('Tilted render plan requested for a non-tilted view')

    key = _tilted_plan_cache_key(view, M_grid_to_src, int(grid_h), int(grid_w))
    while True:
        with _TILTED_RENDER_PLAN_CACHE_LOCK:
            cached = _TILTED_RENDER_PLAN_CACHE.get(key)
            if cached is not None:
                _TILTED_RENDER_PLAN_CACHE.move_to_end(key)
                return cached
            in_flight = _TILTED_RENDER_PLAN_BUILDS_IN_FLIGHT.get(key)
            if in_flight is None:
                _TILTED_RENDER_PLAN_BUILDS_IN_FLIGHT[key] = threading.Event()
                break
        # Another thread is building this exact plan; wait instead of duplicating the
        # multi-hundred-MB temporaries. If the builder fails, the loop retries here.
        in_flight.wait()
    try:
        plan = _build_tilted_render_plan(view, M_grid_to_src, int(grid_h), int(grid_w))
        global _TILTED_RENDER_PLAN_CACHE_BYTES
        with _TILTED_RENDER_PLAN_CACHE_LOCK:
            _TILTED_RENDER_PLAN_CACHE[key] = plan
            _TILTED_RENDER_PLAN_CACHE.move_to_end(key)
            _TILTED_RENDER_PLAN_CACHE_BYTES += _tilted_render_plan_nbytes(plan)
            budget = tilted_render_plan_cache_max_bytes()
            # Evict LRU entries down to the byte budget, always keeping the entry just
            # inserted (a single oversized plan must still be usable).
            while _TILTED_RENDER_PLAN_CACHE_BYTES > budget and len(_TILTED_RENDER_PLAN_CACHE) > 1:
                evict_key, evicted = _TILTED_RENDER_PLAN_CACHE.popitem(last=False)
                _TILTED_RENDER_PLAN_CACHE_BYTES -= _tilted_render_plan_nbytes(evicted)
                # Visible signal: frequent evictions mean the active working set exceeds
                # the budget and plans are being rebuilt — raise YOLO_TTA_TILTED_PLAN_CACHE_GIB.
                print(
                    f'Tilted render plan cache evicted {evict_key[0]} '
                    f'{int(evict_key[1])}x{int(evict_key[2])} '
                    f'({_tilted_render_plan_nbytes(evicted) / GIB:.2f} GiB freed, '
                    f'{_TILTED_RENDER_PLAN_CACHE_BYTES / GIB:.2f} GiB cached).'
                )
        return plan
    finally:
        with _TILTED_RENDER_PLAN_CACHE_LOCK:
            done_event = _TILTED_RENDER_PLAN_BUILDS_IN_FLIGHT.pop(key, None)
        if done_event is not None:
            done_event.set()

def _build_tilted_render_plan(
    view: ViewInfo,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> TiltedRenderPlan:
    yy, xx = np.indices((int(grid_h), int(grid_w)), dtype=np.float32)
    M = np.asarray(M_grid_to_src, dtype=np.float32)
    src_x = (M[0, 0] * xx) + (M[0, 1] * yy) + M[0, 2]
    src_y = (M[1, 0] * xx) + (M[1, 1] * yy) + M[1, 2]

    x_nn = np.rint(src_x).astype(np.int32, copy=False)
    y_nn = np.rint(src_y).astype(np.int32, copy=False)
    valid_xy = (
        (x_nn >= 0) & (x_nn < int(view.src_w)) &
        (y_nn >= 0) & (y_nn < int(view.src_h))
    )
    x_idx = np.clip(x_nn, 0, int(view.src_w) - 1).astype(np.int32, copy=False)
    y_idx = np.clip(y_nn, 0, int(view.src_h) - 1).astype(np.int32, copy=False)

    if str(view.tilt_direction) == 'vertical':
        axis_offset = src_y - float((int(view.src_h) - 1) / 2.0)
    elif str(view.tilt_direction) == 'horizontal':
        axis_offset = src_x - float((int(view.src_w) - 1) / 2.0)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported tilt direction: {view.tilt_direction}')

    valid_xy = valid_xy.astype(bool, copy=False)
    axis_offset = np.asarray(axis_offset, dtype=np.float32)

    # hoist every frame-invariant term. The frame center c is an integer
    # (tilted_frame_center), so with b = tan(tilt)*axis_offset (fp32):
    # stack coordinate = c + b, floor = c + floor(b), frac = b - floor(b),
    # valid <=> c >= -floor(b) and c <= (stack_len-1) - floor(b) - [frac > 0].
    base_view = tilted_base_view_name(view)
    stack_len = int(tilted_stack_axis_length(view))
    full_h = int(view.full_h)
    full_w = int(view.full_w)
    grid_h_i = int(grid_h)
    grid_w_i = int(grid_w)
    tan_alpha = np.float32(math.tan(math.radians(float(view.tilt_angle_deg))))
    b = axis_offset * tan_alpha
    sb0 = np.floor(b).astype(np.int32, copy=False)
    alpha = (b - sb0).astype(np.float32, copy=False)
    last = np.int32(max(0, stack_len - 1))
    c_lo2d = (-sb0).astype(np.int32, copy=False)
    c_hi2d = (last - sb0 - (alpha > 0)).astype(np.int32, copy=False)
    pos_sent = np.int32(2 ** 31 - 1)
    neg_sent = np.int32(-(2 ** 31) + 1)
    row_c_lo = np.where(valid_xy, c_lo2d, pos_sent).min(axis=1).astype(np.int32, copy=False)
    row_c_hi = np.where(valid_xy, c_hi2d, neg_sent).max(axis=1).astype(np.int32, copy=False)

    # Separable fast path: per-row constant stack coordinate AND contiguous in-plane row reads
    # (base u axis is the volume's fastest axis, i.e. transverse/sagittal bases).
    row_fast = False
    row_sb0 = row_alpha = row_v = row_u0 = row_u1 = row_su0 = None
    if base_view in ('transverse', 'sagittal') and stack_len > 0 and grid_w_i > 1:
        rows_const = (
            bool(np.all(sb0 == sb0[:, :1]))
            and bool(np.all(alpha == alpha[:, :1]))
            and bool(np.all(y_idx == y_idx[:, :1]))
        )
        if rows_const:
            counts = valid_xy.sum(axis=1)
            first = np.argmax(valid_xy, axis=1)
            last_col = grid_w_i - 1 - np.argmax(valid_xy[:, ::-1], axis=1)
            contiguous = bool(np.all((counts == 0) | (counts == (last_col - first + 1))))
            if contiguous:
                pair_valid = valid_xy[:, 1:] & valid_xy[:, :-1]
                consecutive = bool(np.all((np.diff(x_idx, axis=1) == 1) | ~pair_valid))
                if consecutive:
                    row_fast = True
                    row_sb0 = np.ascontiguousarray(sb0[:, 0])
                    row_alpha = np.ascontiguousarray(alpha[:, 0])
                    row_v = np.ascontiguousarray(y_idx[:, 0])
                    row_u0 = first.astype(np.int32, copy=False)
                    row_u1 = np.where(counts > 0, last_col + 1, first).astype(np.int32, copy=False)
                    row_su0 = x_idx[np.arange(grid_h_i), np.minimum(first, grid_w_i - 1)].astype(np.int32, copy=False)

    flat_base = alpha2d = om_alpha2d = None
    if base_view == 'transverse':
        stack_stride = full_h * full_w
        inplane = y_idx.astype(np.int64) * np.int64(full_w) + x_idx
    elif base_view == 'sagittal':
        stack_stride = full_w
        inplane = y_idx.astype(np.int64) * np.int64(full_h * full_w) + x_idx
    elif base_view == 'coronal':
        stack_stride = 1
        inplane = y_idx.astype(np.int64) * np.int64(full_h * full_w) + x_idx.astype(np.int64) * np.int64(full_w)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported Tilted View base: {base_view}')
    if not row_fast:
        flat_base = inplane + sb0.astype(np.int64) * np.int64(stack_stride)
        alpha2d = alpha
        om_alpha2d = (np.float32(1.0) - alpha).astype(np.float32, copy=False)

    return TiltedRenderPlan(
        x_idx=x_idx,
        y_idx=y_idx,
        valid_xy=valid_xy,
        axis_offset=axis_offset,
        stack_stride=int(stack_stride),
        row_c_lo=row_c_lo,
        row_c_hi=row_c_hi,
        row_fast=bool(row_fast),
        row_sb0=row_sb0,
        row_alpha=row_alpha,
        row_v=row_v,
        row_u0=row_u0,
        row_u1=row_u1,
        row_su0=row_su0,
        flat_base=flat_base,
        alpha2d=alpha2d,
        om_alpha2d=om_alpha2d,
        c_lo2d=None if row_fast else c_lo2d,
        c_hi2d=None if row_fast else c_hi2d,
    )

_TILTED_RENDER_TLS = threading.local()

def _tilted_tls_buffer(name: str, count: int, dtype: object) -> np.ndarray:
    """Reusable per-render-thread flat scratch buffer (grow-only)."""
    bufs = getattr(_TILTED_RENDER_TLS, 'bufs', None)
    if bufs is None:
        bufs = {}
        _TILTED_RENDER_TLS.bufs = bufs
    key = (str(name), np.dtype(dtype).str)
    cur = bufs.get(key)
    if cur is None or int(cur.size) < int(count):
        cur = np.empty((int(count),), dtype=np.dtype(dtype))
        bufs[key] = cur
    return cur[: int(count)]

def _render_tilted_array_on_grid(
    volume_arr: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
    *,
    mask_mode: bool,
    block_rows: int = 256,
) -> np.ndarray:
    """Render one Tilted frame on an arbitrary output grid.

    A cached plan hoists frame-invariant geometry; row-contiguous and flattened-gather paths avoid repeated setup."""
    if not is_tilted_view(view):
        raise ValueError('Tilted rendering requested for a non-tilted view')

    if (
        not bool(mask_mode)
        and tilted_inplane_linear_enabled()
        and not _tilted_grid_is_identity(M_grid_to_src, int(grid_h), int(grid_w), view)
    ):
        # v16.1.8 forward-pass in-plane interpolation: render the exact integer-grid
        # native frame, then apply the requested grid->src affine bilinearly (the same
        # warp Cartesian views use). Mask projections keep the nearest plan path.
        native = _render_tilted_array_on_grid(
            volume_arr, view, int(frame_idx), _TILTED_IDENTITY_M,
            int(view.src_h), int(view.src_w),
            mask_mode=False, block_rows=int(block_rows),
        )
        return cv2.warpAffine(
            native,
            np.asarray(M_grid_to_src, dtype=np.float32).reshape(2, 3),
            (int(grid_w), int(grid_h)),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # A tilted output frame can sample a sheared range of the base stack. Wait for
    # the streaming preprocessing producer to finish before using these views.
    wait_for_volume_ready(volume_arr)

    plan = get_tilted_render_plan(view, M_grid_to_src, int(grid_h), int(grid_w))
    stack_len = int(tilted_stack_axis_length(view))
    out = np.zeros((int(grid_h), int(grid_w)), dtype=np.uint8)
    if stack_len <= 0:
        return out

    vol = np.asarray(volume_arr)
    expected_shape = (int(view.full_t), int(view.full_h), int(view.full_w))
    hoisted_usable = (
        int(plan.stack_stride) > 0
        and plan.row_c_lo is not None
        and tuple(int(x) for x in vol.shape) == expected_shape
        and (bool(plan.row_fast) or (plan.flat_base is not None and bool(vol.flags['C_CONTIGUOUS'])))
    )
    if not hoisted_usable:
        return _render_tilted_array_on_grid_reference(
            volume_arr, view, int(frame_idx), M_grid_to_src, int(grid_h), int(grid_w),
            mask_mode=bool(mask_mode), block_rows=int(block_rows),
        )

    c = int(tilted_frame_center(view, int(frame_idx)))
    band = np.flatnonzero((plan.row_c_lo <= c) & (c <= plan.row_c_hi))
    if band.size == 0:
        return out
    last = int(stack_len - 1)

    if bool(plan.row_fast):
        sagittal = tilted_base_view_name(view) == 'sagittal'
        row_a = _tilted_tls_buffer('rowfast_a', int(grid_w), np.float32)
        row_b = _tilted_tls_buffer('rowfast_b', int(grid_w), np.float32)
        for r in band:
            r_i = int(r)
            u0 = int(plan.row_u0[r_i])
            u1 = int(plan.row_u1[r_i])
            n = u1 - u0
            if n <= 0:
                continue
            s0 = int(plan.row_sb0[r_i]) + c  # band guarantees 0 <= s0 <= last (alpha-aware)
            s1 = s0 + 1 if s0 < last else last
            a = float(plan.row_alpha[r_i])
            v = int(plan.row_v[r_i])
            su0 = int(plan.row_su0[r_i])
            if sagittal:
                f0 = vol[v, s0, su0:su0 + n]
                f1 = vol[v, s1, su0:su0 + n]
            else:
                f0 = vol[s0, v, su0:su0 + n]
                f1 = vol[s1, v, su0:su0 + n]
            orow = out[r_i, u0:u1]
            if a == 0.0:
                if bool(mask_mode):
                    np.greater_equal(f0, 1, out=orow)
                else:
                    orow[:] = f0
            else:
                va = row_a[:n]
                vb = row_b[:n]
                np.multiply(f0, np.float32(1.0 - a), out=va)
                np.multiply(f1, np.float32(a), out=vb)
                np.add(va, vb, out=va)
                if bool(mask_mode):
                    np.greater_equal(va, 0.5, out=orow)
                else:
                    np.rint(va, out=va)
                    np.clip(va, 0.0, 255.0, out=va)
                    np.copyto(orow, va, casting='unsafe')
        return out

    r0 = int(band[0])
    r1 = int(band[-1]) + 1
    nrows = r1 - r0
    n = nrows * int(grid_w)
    shape2 = (nrows, int(grid_w))
    idx = _tilted_tls_buffer('flat_idx', n, np.int64).reshape(shape2)
    f0 = _tilted_tls_buffer('flat_f0', n, np.uint8).reshape(shape2)
    f1 = _tilted_tls_buffer('flat_f1', n, np.uint8).reshape(shape2)
    val = _tilted_tls_buffer('flat_val', n, np.float32).reshape(shape2)
    val2 = _tilted_tls_buffer('flat_val2', n, np.float32).reshape(shape2)
    valid = _tilted_tls_buffer('flat_valid', n, np.bool_).reshape(shape2)
    valid2 = _tilted_tls_buffer('flat_valid2', n, np.bool_).reshape(shape2)

    vol_flat = vol.reshape(-1)
    stride = np.int64(plan.stack_stride)
    np.add(plan.flat_base[r0:r1], np.int64(c) * stride, out=idx)
    # mode='clip' bounds out-of-volume indices; those pixels are zeroed by the validity merge.
    np.take(vol_flat, idx, mode='clip', out=f0)
    np.add(idx, stride, out=idx)
    # Where s0 == last (only valid with alpha == 0) this reads the clipped neighbor, but its
    # lerp weight is exactly 0, so the value never contributes.
    np.take(vol_flat, idx, mode='clip', out=f1)

    np.less_equal(plan.c_lo2d[r0:r1], c, out=valid)
    np.greater_equal(plan.c_hi2d[r0:r1], c, out=valid2)
    np.logical_and(valid, valid2, out=valid)
    np.logical_and(valid, plan.valid_xy[r0:r1], out=valid)

    # Same fp32 expression as the reference path: (1-alpha)*f0 + alpha*f1.
    np.multiply(f0, plan.om_alpha2d[r0:r1], out=val)
    np.multiply(f1, plan.alpha2d[r0:r1], out=val2)
    np.add(val, val2, out=val)

    if bool(mask_mode):
        np.greater_equal(val, 0.5, out=valid2)
        np.logical_and(valid2, valid, out=valid2)
        out[r0:r1, :] = valid2
    else:
        np.rint(val, out=val)
        np.clip(val, 0.0, 255.0, out=val)
        rendered = f0  # reuse the u8 buffer for the quantized output
        np.copyto(rendered, val, casting='unsafe')
        np.multiply(rendered, valid, out=rendered)
        out[r0:r1, :] = rendered
    return out

def _render_tilted_array_on_grid_reference(
    volume_arr: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
    *,
    mask_mode: bool,
    block_rows: int = 256,
) -> np.ndarray:
    """Legacy exact tilted render (per-block double fancy gather); fallback/reference."""
    plan = get_tilted_render_plan(view, M_grid_to_src, int(grid_h), int(grid_w))
    tan_alpha = float(math.tan(math.radians(float(view.tilt_angle_deg))))
    stack_len = int(tilted_stack_axis_length(view))
    base_view = tilted_base_view_name(view)
    out = np.zeros((int(grid_h), int(grid_w)), dtype=np.uint8)
    if stack_len <= 0:
        return out

    for y0 in range(0, int(grid_h), int(block_rows)):
        y1 = min(int(grid_h), y0 + int(block_rows))
        valid_xy = np.asarray(plan.valid_xy[y0:y1], dtype=bool)
        if not np.any(valid_xy):
            continue

        frame_center = float(tilted_frame_center(view, int(frame_idx)))
        stack_src = frame_center + tan_alpha * np.asarray(plan.axis_offset[y0:y1], dtype=np.float32)
        valid = valid_xy & (stack_src >= 0.0) & (stack_src <= float(stack_len - 1))
        if not np.any(valid):
            continue

        s0 = np.floor(stack_src).astype(np.int32, copy=False)
        s1 = np.clip(s0 + 1, 0, stack_len - 1).astype(np.int32, copy=False)
        s0 = np.clip(s0, 0, stack_len - 1).astype(np.int32, copy=False)
        alpha = (stack_src - s0).astype(np.float32, copy=False)

        u_idx = np.asarray(plan.x_idx[y0:y1], dtype=np.int32)  # base-view horizontal axis
        v_idx = np.asarray(plan.y_idx[y0:y1], dtype=np.int32)  # base-view vertical axis
        if base_view == 'transverse':
            f0 = np.asarray(volume_arr[s0, v_idx, u_idx], dtype=np.float32)
            f1 = np.asarray(volume_arr[s1, v_idx, u_idx], dtype=np.float32)
        elif base_view == 'sagittal':
            # Sagittal in-plane axes are (X, t); stacking axis is Y.
            f0 = np.asarray(volume_arr[v_idx, s0, u_idx], dtype=np.float32)
            f1 = np.asarray(volume_arr[v_idx, s1, u_idx], dtype=np.float32)
        elif base_view == 'coronal':
            # Coronal in-plane axes are (Y, t); stacking axis is X.
            f0 = np.asarray(volume_arr[v_idx, u_idx, s0], dtype=np.float32)
            f1 = np.asarray(volume_arr[v_idx, u_idx, s1], dtype=np.float32)
        else:  # pragma: no cover
            raise ValueError(f'Unsupported Tilted View base: {base_view}')

        values = ((1.0 - alpha) * f0) + (alpha * f1)
        if bool(mask_mode):
            rendered = (values >= 0.5).astype(np.uint8, copy=False)
        else:
            rendered = np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)
        out_block = out[y0:y1]
        out_block[valid] = rendered[valid]

    return out

def render_tilted_frame_on_grid(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    M_grid_to_src: np.ndarray,
    grid_h: int,
    grid_w: int,
    block_rows: int = 256,
) -> np.ndarray:
    return _render_tilted_array_on_grid(
        volume_rgb,
        view,
        int(frame_idx),
        M_grid_to_src,
        int(grid_h),
        int(grid_w),
        mask_mode=False,
        block_rows=int(block_rows),
    )

def render_tilted_canvas_frame(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    frame_idx: int,
    aff: AffineSpec,
    block_rows: int = 256,
) -> np.ndarray:
    return render_tilted_frame_on_grid(
        volume_rgb=volume_rgb,
        view=view,
        frame_idx=int(frame_idx),
        M_grid_to_src=aff.M_canvas_to_src,
        grid_h=int(aff.canvas_h),
        grid_w=int(aff.canvas_w),
        block_rows=int(block_rows),
    )

_TILTED_NATIVE_AFFINE_CACHE: Dict[Tuple[str, int, int], AffineSpec] = {}

def get_tilted_native_affine(view: ViewInfo) -> AffineSpec:
    if not is_tilted_view(view):
        raise ValueError('Tilted native affine requested for a non-tilted view')
    key = (str(view.name), int(view.src_w), int(view.src_h))
    cached = _TILTED_NATIVE_AFFINE_CACHE.get(key)
    if cached is not None:
        return cached
    aff = build_affine(
        view=view.name,
        src_w=int(view.src_w),
        src_h=int(view.src_h),
        out_size=max(int(view.src_w), int(view.src_h), 1),
        angle_deg=0.0,
        pad_mode='clamp',
    )
    _TILTED_NATIVE_AFFINE_CACHE[key] = aff
    return aff

def render_tilted_native_frame(volume_rgb: np.ndarray, view: ViewInfo, frame_idx: int) -> np.ndarray:
    aff = get_tilted_native_affine(view)
    return render_tilted_canvas_frame(volume_rgb, view, int(frame_idx), aff)

def render_fullframe_frame_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    frame_idx: int,
    view_frames: Optional[np.ndarray] = None,
    *,
    mirror_radial_u: bool = False,
) -> np.ndarray:
    if is_tilted_view(view):
        return render_tilted_frame_on_grid(
            volume_rgb=volume_rgb,
            view=view,
            frame_idx=int(frame_idx),
            M_grid_to_src=job.aff.M_out_to_src,
            grid_h=int(job.aff.out_size),
            grid_w=int(job.aff.out_size),
        )

    native_frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(frame_idx), view_frames=view_frames))
    if bool(mirror_radial_u):
        if not is_radial_view(view):
            raise ValueError('radial-u mirroring requested for a non-Radial view')
        native_frame = np.ascontiguousarray(native_frame[:, ::-1])
    # identity fast path — a rotation-free job whose native frame already
    # matches the model raster (e.g. every imgsz-folded radial frame at --angle 0) needs no warp.
    aff = job.aff
    if (
        int(aff.src_w) == int(aff.out_size)
        and int(aff.src_h) == int(aff.out_size)
        and int(aff.canvas_w) == int(aff.src_w)
        and int(aff.canvas_h) == int(aff.src_h)
        and float(aff.angle_deg) % 360.0 == 0.0
    ):
        # Defensive copy only when the frame aliases the source volume (e.g. a transverse slice).
        return native_frame.copy() if native_frame.base is not None else native_frame
    return cv2.warpAffine(
        native_frame,
        aff.M_src_to_out,
        dsize=(int(aff.out_size), int(aff.out_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

def render_dense_tile_frame_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    tile_job: DenseTileJob,
    frame_idx: int,
    view_frames: Optional[np.ndarray] = None,
    *,
    mirror_radial_u: bool = False,
) -> np.ndarray:
    """Render one dense-tile inference range directly from the native view volume.

 This replaces the legacy canvas-video -> crop -> scale FFmpeg path with the
 same transform collapsed into one in-memory reslice. ``tile_job.M_src_to_out``
 maps native view coordinates directly to the tile's ``--imgsz`` inference
 raster; for Tilted Views the inverse grid-to-native transform is passed into
 the tilted sampler so the stacking-axis shear, in-plane augmentation, crop,
 and scale are sampled in one pass."""
    if is_tilted_view(view):
        return render_tilted_frame_on_grid(
            volume_rgb=volume_rgb,
            view=view,
            frame_idx=int(frame_idx),
            M_grid_to_src=tile_job.M_out_to_src,
            grid_h=int(tile_job.out_size),
            grid_w=int(tile_job.out_size),
        )

    native_frame = np.ascontiguousarray(get_view_frame_by_index(volume_rgb, view, int(frame_idx), view_frames=view_frames))
    if bool(mirror_radial_u):
        if not is_radial_view(view):
            raise ValueError('radial-u mirroring requested for a non-Radial view')
        native_frame = np.ascontiguousarray(native_frame[:, ::-1])
    return cv2.warpAffine(
        native_frame,
        tile_job.M_src_to_out,
        dsize=(int(tile_job.out_size), int(tile_job.out_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

def make_fullframe_channel_renderer(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    *,
    channel_format: ChannelFormat,
    view_frames: Optional[np.ndarray] = None,
    cache_frames: Optional[int] = None,
) -> ChannelFormattedFrameRenderer:
    """Create a center-indexed full-frame renderer for gray/RGB/2.5D inputs."""
    def _render_plane(source_idx: int) -> np.ndarray:
        return render_fullframe_frame_for_job(
            volume_rgb=volume_rgb,
            view=view,
            job=job,
            frame_idx=int(source_idx),
            view_frames=view_frames,
        )

    def _render_mirrored_plane(source_idx: int) -> np.ndarray:
        return render_fullframe_frame_for_job(
            volume_rgb=volume_rgb,
            view=view,
            job=job,
            frame_idx=int(source_idx),
            view_frames=view_frames,
            mirror_radial_u=True,
        )

    return ChannelFormattedFrameRenderer(
        _render_plane,
        view,
        resolve_channel_format(channel_format),
        cache_frames=cache_frames,
        mirrored_plane_renderer=_render_mirrored_plane if is_radial_view(view) else None,
    )

def make_dense_tile_channel_renderer(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    tile_job: DenseTileJob,
    *,
    channel_format: ChannelFormat,
    view_frames: Optional[np.ndarray] = None,
    cache_frames: Optional[int] = None,
) -> ChannelFormattedFrameRenderer:
    """Create a center-indexed dense-tile renderer for gray/RGB/2.5D inputs."""
    def _render_plane(source_idx: int) -> np.ndarray:
        return render_dense_tile_frame_for_job(
            volume_rgb=volume_rgb,
            view=view,
            tile_job=tile_job,
            frame_idx=int(source_idx),
            view_frames=view_frames,
        )

    def _render_mirrored_plane(source_idx: int) -> np.ndarray:
        return render_dense_tile_frame_for_job(
            volume_rgb=volume_rgb,
            view=view,
            tile_job=tile_job,
            frame_idx=int(source_idx),
            view_frames=view_frames,
            mirror_radial_u=True,
        )

    return ChannelFormattedFrameRenderer(
        _render_plane,
        view,
        resolve_channel_format(channel_format),
        cache_frames=cache_frames,
        mirrored_plane_renderer=_render_mirrored_plane if is_radial_view(view) else None,
    )

def _materialize_prediction_volume_from_renderer(
    *,
    num_slices: int,
    out_size: int,
    out_path: Path,
    desc: str,
    renderer: Callable[[int], np.ndarray],
    workers: int = 1,
    show_progress: bool = True,
    view_name: str = '',
    job_id: str = '',
    kind: str = 'fullframe',
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
) -> PredictionVolumeRef:
    """Return a streaming render-backed prediction source by default.
    
    Disabling streaming materializes the complete model-sized volume in a workspace array."""
    fmt = resolve_channel_format(channel_format)
    stream_name = str(desc).replace('Materializing in-memory ', 'Streaming render-backed ')
    if streaming_prediction_sources_enabled():
        stream_workers = streaming_prediction_source_workers(max(1, int(workers)), max(1, int(num_slices)))
        stream_prefetch = streaming_prediction_source_prefetch_frames(inference_batch_size())
        if bool(show_progress):
            print(
                f'{stream_name}: streaming source active '
                f'(frames={int(num_slices)}, out_size={int(out_size)}, '
                f'workers={stream_workers}, prefetch_frames={stream_prefetch}, '
                f'autostart={bool(streaming_prediction_source_autostart_enabled())})'
            )
        source = StreamingYoloVolumeSource(
            renderer,
            num_frames=int(num_slices),
            name=stream_name,
            batch_size=inference_batch_size(),
            out_size=int(out_size),
            render_workers=stream_workers,
            prefetch_frames=stream_prefetch,
            autostart=streaming_prediction_source_autostart_enabled(),
            channel_format=fmt,
        )
        return PredictionVolumeRef(
            array=None,
            path=None,
            name=stream_name,
            view_name=str(view_name),
            job_id=str(job_id),
            kind=str(kind),
            source=source,
            channel_format=fmt,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_shape = (
        (int(num_slices), int(out_size), int(out_size))
        if int(fmt.channel_count) == 1
        else (int(num_slices), int(out_size), int(out_size), int(fmt.channel_count))
    )
    pred_volume = allocate_workspace_array(
        shape=prediction_shape,
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=True,
        reserve_bytes=32 * GIB,
        initialize_zero=False,
    )
    worker_count = choose_slice_parallel_workers(int(workers), int(num_slices))
    chunk_size = choose_parallel_chunk_size(int(num_slices), worker_count, target_chunks_per_worker=2, min_chunk_size=1)

    def _render(idx: int) -> None:
        frame = np.asarray(renderer(int(idx)), dtype=np.uint8)
        if int(fmt.channel_count) == 1 and frame.ndim == 3:
            frame = frame[:, :, 0]
        pred_volume[int(idx)] = np.ascontiguousarray(frame, dtype=np.uint8)

    parallel_for_indices_chunked(
        int(num_slices),
        _render,
        max_workers=worker_count,
        desc=desc,
        show_progress=bool(show_progress),
        chunk_size=chunk_size,
    )
    if prediction_volume_build_flush_enabled():
        flush_array(pred_volume)
    return PredictionVolumeRef(
        array=pred_volume,
        path=out_path if isinstance(pred_volume, np.memmap) else None,
        name=str(desc),
        view_name=str(view_name),
        job_id=str(job_id),
        kind=str(kind),
        channel_format=fmt,
    )

def materialize_fullframe_prediction_volume_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    job: AugJob,
    *,
    out_path: Path,
    view_frames: Optional[np.ndarray] = None,
    workers: int = 1,
    show_progress: bool = True,
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
) -> PredictionVolumeRef:
    """Create the model-sized full-frame prediction volume for one view/angle in RAM."""
    # Refresh reusable scratch metadata so changing --channel_format cannot
    # leave a gray-era sidecar attached to a multichannel prediction source.
    write_aug_job_meta(job, view, channel_format)

    fmt = resolve_channel_format(channel_format)
    renderer = make_fullframe_channel_renderer(
        volume_rgb,
        view,
        job,
        channel_format=fmt,
        view_frames=view_frames,
    )

    return _materialize_prediction_volume_from_renderer(
        num_slices=int(view.num_slices),
        out_size=int(job.aff.out_size),
        out_path=out_path,
        desc=f'Materializing in-memory full-frame prediction volume {view.name}/{job.aug_id}',
        renderer=renderer,
        workers=max(1, int(workers)),
        show_progress=bool(show_progress),
        view_name=str(view.name),
        job_id=str(job.aug_id),
        kind='fullframe',
        channel_format=fmt,
    )

def materialize_dense_tile_prediction_volume_for_job(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    tile_job: DenseTileJob,
    *,
    out_path: Path,
    view_frames: Optional[np.ndarray] = None,
    workers: int = 1,
    show_progress: bool = True,
    channel_format: ChannelFormat = DEFAULT_CHANNEL_FORMAT,
) -> PredictionVolumeRef:
    """Create one dense-tile prediction range as an in-memory YOLO volume."""
    write_dense_tile_job_meta(tile_job, channel_format)

    fmt = resolve_channel_format(channel_format)
    renderer = make_dense_tile_channel_renderer(
        volume_rgb,
        view,
        tile_job,
        channel_format=fmt,
        view_frames=view_frames,
    )

    return _materialize_prediction_volume_from_renderer(
        num_slices=int(view.num_slices),
        out_size=int(tile_job.out_size),
        out_path=out_path,
        desc=f'Materializing in-memory tile prediction volume {view.name}/{tile_job.tile_id}',
        renderer=renderer,
        workers=max(1, int(workers)),
        show_progress=bool(show_progress),
        view_name=str(view.name),
        job_id=str(tile_job.tile_id),
        kind='tile',
        channel_format=fmt,
    )

def should_cache_view_frames(view: ViewInfo, dense_tiling_active: bool) -> bool:
    """Return whether an opt-in dense-tile Radial frame cache should be prebuilt.

    Streaming remains the default to avoid a full-view time-to-first-prediction barrier.
    """
    return bool(_env_flag('YOLO_TTA_PREBUILD_VIEW_FRAME_CACHES', False)) and bool(dense_tiling_active) and view.family == 'radial'

def build_view_frame_cache(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 64 * GIB,
    workers: int = 1,
) -> np.ndarray:
    """Materialize native single-channel frames for a view into a reusable cache volume.

 This is used primarily for the radial view when dense tiling is enabled so later tile video
 generation no longer recomputes the same radial slices for every tile location."""
    cache_mm = allocate_workspace_array(
        shape=(int(view.num_slices), int(view.src_h), int(view.src_w)),
        dtype=np.uint8,
        path=out_path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=False,
    )

    worker_count = choose_slice_parallel_workers(int(workers), int(view.num_slices))
    print(
        f"Preparing reusable native single-channel frame cache for view '{view.name}' over {int(view.num_slices)} slice(s) "
        f"with {int(worker_count)} worker thread(s)"
    )

    def _build(idx: int) -> None:
        cache_mm[int(idx), :, :] = np.ascontiguousarray(
            get_view_frame_by_index(volume_rgb, view, int(idx), view_frames=None)
        )

    parallel_for_indices(
        int(view.num_slices),
        _build,
        max_workers=worker_count,
        desc=f'Caching {view.name} native frames',
        show_progress=False,
    )
    flush_array(cache_mm)
    return cache_mm

_CORONAL_BLOCK_CACHE: 'OrderedDict[Tuple[int, Tuple[int, int, int], int], np.ndarray]' = OrderedDict()

_CORONAL_BLOCK_CACHE_LOCK = threading.Lock()

_CORONAL_BLOCK_BUILDS_IN_FLIGHT: Dict[Tuple[int, Tuple[int, int, int], int], threading.Event] = {}

def coronal_block_cols() -> int:
    """Columns per cached coronal block (64 x 1 B = one full cache line)."""
    return max(8, _env_int('YOLO_TTA_CORONAL_BLOCK_COLS', 64))

def coronal_block_cache_blocks() -> int:
    """Coronal block LRU capacity, in blocks (each K*T*H bytes)."""
    return max(1, _env_int('YOLO_TTA_CORONAL_BLOCK_CACHE', 2))

def _coronal_block_cache_key(volume: np.ndarray, block_idx: int) -> Tuple[int, Tuple[int, int, int], int]:
    ptr = int(volume.__array_interface__['data'][0])
    return (ptr, tuple(int(x) for x in volume.shape), int(block_idx))

def _build_coronal_block(volume: np.ndarray, x0: int, x1: int) -> np.ndarray:
    T, H, _W = (int(v) for v in volume.shape)
    blk = np.empty((int(x1) - int(x0), T, H), dtype=volume.dtype)
    # Per-t (H, K) -> (K, H) tile transpose: the tile (~H*K bytes) stays cache-resident, source
    # reads use full cache lines (K consecutive bytes per row), and destination rows are
    # contiguous. All copies are numpy strided loops (GIL-releasing).
    for t in range(T):
        blk[:, t, :] = volume[t, :, int(x0):int(x1)].T
    return blk

def _coronal_frame_from_block_cache(volume: np.ndarray, x: int) -> np.ndarray:
    K = int(coronal_block_cols())
    W = int(volume.shape[2])
    block_idx = int(x) // K
    key = _coronal_block_cache_key(volume, block_idx)
    while True:
        with _CORONAL_BLOCK_CACHE_LOCK:
            cached = _CORONAL_BLOCK_CACHE.get(key)
            if cached is not None:
                _CORONAL_BLOCK_CACHE.move_to_end(key)
                return cached[int(x) - block_idx * K]
            in_flight = _CORONAL_BLOCK_BUILDS_IN_FLIGHT.get(key)
            if in_flight is None:
                _CORONAL_BLOCK_BUILDS_IN_FLIGHT[key] = threading.Event()
                break
        in_flight.wait()
    try:
        x0 = block_idx * K
        blk = _build_coronal_block(volume, x0, min(W, x0 + K))
        with _CORONAL_BLOCK_CACHE_LOCK:
            _CORONAL_BLOCK_CACHE[key] = blk
            _CORONAL_BLOCK_CACHE.move_to_end(key)
            while len(_CORONAL_BLOCK_CACHE) > coronal_block_cache_blocks():
                _CORONAL_BLOCK_CACHE.popitem(last=False)
        # Returned frames are views of blk; an evicted block stays alive while any frame view
        # still references it, so eviction can never invalidate an in-flight consumer.
        return blk[int(x) - x0]
    finally:
        with _CORONAL_BLOCK_CACHE_LOCK:
            done = _CORONAL_BLOCK_BUILDS_IN_FLIGHT.pop(key, None)
        if done is not None:
            done.set()

def get_view_frame_by_index(
    volume_rgb: np.ndarray,
    view: ViewInfo,
    index: int,
    view_frames: Optional[np.ndarray] = None,
) -> np.ndarray:
    if view_frames is not None:
        return np.asarray(view_frames[int(index)])

    T, H, W = volume_rgb.shape

    if physical_view_name(view) == 'transverse':
        wait_for_volume_slice_ready(volume_rgb, int(index))
        return np.asarray(volume_rgb[int(index)])
    if physical_view_name(view) == 'sagittal':
        wait_for_volume_ready(volume_rgb)
        return np.ascontiguousarray(volume_rgb[:, int(index), :])
    if physical_view_name(view) == 'coronal':
        wait_for_volume_ready(volume_rgb)
        # serve from K-column transposed blocks instead of a strided gather.
        if volume_rgb.ndim == 3 and bool(volume_rgb.flags['C_CONTIGUOUS']):
            return _coronal_frame_from_block_cache(volume_rgb, int(index))
        return np.ascontiguousarray(volume_rgb[:, :, int(index)])
    if is_radial_view(view):
        wait_for_volume_ready(volume_rgb)
        angle_deg = float(view.azimuths_deg[int(index)])
        sampler = get_radial_sampler(view, angle_deg)
        if is_tilted_radial_view(view):
            return np.ascontiguousarray(
                extract_tilted_radial_slice_frame(
                    volume_rgb, view, sampler, out_rows=int(view.src_h),
                )
            )
        oriented = radial_oriented_stack_view(volume_rgb, view)
        return np.ascontiguousarray(
            extract_radial_slice_frame(oriented, sampler, out_rows=int(view.src_h))
        )
    if is_tilted_view(view):
        wait_for_volume_ready(volume_rgb)
        return render_tilted_native_frame(volume_rgb, view, int(index))

    raise ValueError(f'Unknown view: {view.name}')


# Late imports keep callable-only dependency cycles import-safe.
from ._latebind import bind_late_symbols as _bind_late_symbols

_bind_late_symbols(
    __name__,
    globals(),
    {
        "config": (
            "CARTESIAN_VIEW_TOKENS",
            "ChannelFormat",
            "DEFAULT_CHANNEL_FORMAT",
            "GIB",
            "RADIAL_VIEW_TOKENS",
            "TiltedViewGroup",
            "_parse_comma_slot",
            "_resolve_unique_view_tokens",
            "_split_structured_group",
            "_structured_group_values",
            "quantize_uses_fp16",
            "resolve_cartesian_views",
            "resolve_channel_format",
        ),
        "cuda_backend": (
            "GpuRenderedYoloSource",
            "GpuTileRenderedYoloSource",
        ),
        "inference": (
            "canonical_single_device",
            "inference_batch_size",
        ),
        "media": (
            "wait_for_volume_ready",
            "wait_for_volume_slice_ready",
        ),
        "runtime": (
            "_sanitize_filesystem_token",
            "allocate_workspace_array",
            "choose_parallel_chunk_size",
            "choose_slice_parallel_workers",
            "close_memmap_array",
            "close_memmap_array_without_flush",
            "flush_array",
            "parallel_for_indices",
            "parallel_for_indices_chunked",
            "prediction_volume_build_flush_enabled",
        ),
        "workspace": (
            "_TILTED_IDENTITY_M",
            "_cpu_count",
            "_env_flag",
            "_env_float",
            "_env_int",
            "_tilted_grid_is_identity",
            "tilted_inplane_linear_enabled",
        ),
    },
)
