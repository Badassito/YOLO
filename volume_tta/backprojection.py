"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np
from ._deps import _numba, cv2, tqdm

from .config import (
    GIB,
)
from .geometry import (
    ViewInfo,
)
from .runtime import (
    runtime_telemetry_phase,
)


# These announcements belong to the resident TensorRT executor implemented in this
# module. Keeping them process-local avoids copied mutable state across module seams.
_RESIDENT_TRT_RING_ANNOUNCED = False
_RESIDENT_TRT_RING_FALLBACK_WARNED = False

@dataclass(frozen=True)
class RadialBackprojectionSample:
    angle_deg: float
    source_index: int
    reverse_u: bool = False

@dataclass(frozen=True)
class DenseRadialBackprojectionMap:
    valid_mask: np.ndarray
    source_idx_map: np.ndarray
    u_idx_map: np.ndarray

@dataclass(frozen=True)
class RadialProcessingGrid:
    """Map native Radial row/diameter coordinates into the stored processing raster."""

    processing_h: int
    processing_w: int
    native_h: int
    native_w: int
    native_row_to_processing: np.ndarray
    native_u_to_processing: np.ndarray
    reduced: bool = False

def resolve_radial_processing_grid(
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
) -> RadialProcessingGrid:
    """Resolve the exact angle-zero native-to-processing mapping for one Radial layer.

    Radial angle/frame order is unchanged.  Only the per-frame ``(row, diameter)`` raster
    may be canonicalized to ``imgsz``.  The angle-zero affine is axis aligned, so compact
    one-dimensional row and diameter lookup tables carry the transform through wraparound
    interpolation and terminal Cartesian backprojection without expanding a native Radial
    volume first.
    """
    src = np.asarray(radial_mask_mm)
    if src.ndim != 3:
        raise ValueError(f'Radial processing grid expects a 3D layer, got {src.shape}')
    processing_h, processing_w = int(src.shape[1]), int(src.shape[2])
    native_h, native_w = int(radial_view.src_h), int(radial_view.src_w)
    if min(processing_h, processing_w, native_h, native_w) <= 0:
        raise ValueError(
            f'Radial processing grid has invalid processing/native geometry '
            f'{processing_h}x{processing_w} / {native_h}x{native_w}'
        )
    if (processing_h, processing_w) == (native_h, native_w):
        return RadialProcessingGrid(
            processing_h=processing_h,
            processing_w=processing_w,
            native_h=native_h,
            native_w=native_w,
            native_row_to_processing=np.arange(native_h, dtype=np.int32),
            native_u_to_processing=np.arange(native_w, dtype=np.int32),
            reduced=False,
        )
    if processing_h != processing_w:
        raise ValueError(
            f'Reduced Radial processing raster must be square, got '
            f'{processing_h}x{processing_w}'
        )
    if not view_uses_inference_processing_grid(radial_view, processing_w):
        raise ValueError(
            f'Radial layer {src.shape} differs from native '
            f'({native_h},{native_w}) without delayed-expansion capability'
        )
    expected = view_processing_plane_shape(radial_view, processing_w)
    if tuple(int(v) for v in expected) != (processing_h, processing_w):
        raise ValueError(
            f'Radial processing raster {processing_h}x{processing_w} does not match '
            f'canonical geometry {expected}'
        )
    canonical = build_affine(
        view=str(radial_view.name),
        src_w=native_w,
        src_h=native_h,
        out_size=processing_w,
        angle_deg=0.0,
        pad_mode=str(radial_view.pad_mode),
    )
    matrix = np.asarray(canonical.M_src_to_out, dtype=np.float64).reshape(2, 3)
    if not bool(np.isfinite(matrix).all()) or abs(float(matrix[0, 1])) > 1e-6 or abs(float(matrix[1, 0])) > 1e-6:
        raise ValueError('Radial canonical affine is non-finite or not axis aligned')
    # Ask OpenCV to transform one-dimensional coordinate ramps with the same affine
    # and nearest-neighbour implementation used by native expansion.  Re-implementing the
    # rounding algebra is not exact at fixed-point half-pixel boundaries.
    out_to_native = np.asarray(canonical.M_out_to_src, dtype=np.float32).reshape(2, 3)
    row_source = np.arange(processing_h, dtype=np.float32).reshape(processing_h, 1)
    row_affine = np.array(
        [[1.0, 0.0, 0.0], [0.0, out_to_native[1, 1], out_to_native[1, 2]]],
        dtype=np.float32,
    )
    row_values = cv2.warpAffine(
        row_source,
        row_affine,
        dsize=(1, native_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1.0,
    ).reshape(-1)
    u_source = np.arange(processing_w, dtype=np.float32).reshape(1, processing_w)
    u_affine = np.array(
        [[out_to_native[0, 0], 0.0, out_to_native[0, 2]], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    u_values = cv2.warpAffine(
        u_source,
        u_affine,
        dsize=(native_w, 1),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1.0,
    ).reshape(-1)
    if (
        not bool(np.isfinite(row_values).all())
        or not bool(np.isfinite(u_values).all())
        or bool(np.any(row_values < 0.0))
        or bool(np.any(row_values >= float(processing_h)))
        or bool(np.any(u_values < 0.0))
        or bool(np.any(u_values >= float(processing_w)))
    ):
        raise ValueError('Radial canonical affine maps native content outside the processing raster')
    row_map = np.ascontiguousarray(row_values.astype(np.int32, copy=False))
    u_map = np.ascontiguousarray(u_values.astype(np.int32, copy=False))
    return RadialProcessingGrid(
        processing_h=processing_h,
        processing_w=processing_w,
        native_h=native_h,
        native_w=native_w,
        native_row_to_processing=np.ascontiguousarray(row_map),
        native_u_to_processing=np.ascontiguousarray(u_map),
        reduced=True,
    )

def _radial_processing_rows_for_output(
    grid: RadialProcessingGrid,
    output_rows: int,
    output_index: int,
) -> np.ndarray:
    native_rows = _radial_source_rows_for_output(
        int(grid.native_h), int(output_rows), int(output_index),
    )
    mapped = grid.native_row_to_processing[np.fromiter(native_rows, dtype=np.int32)]
    return np.unique(np.asarray(mapped, dtype=np.int32))

def _radial_dense_map_for_processing(
    dense_map: DenseRadialBackprojectionMap,
    grid: RadialProcessingGrid,
) -> DenseRadialBackprojectionMap:
    native_u = np.asarray(dense_map.u_idx_map, dtype=np.int32)
    if native_u.size and (int(native_u.min()) < 0 or int(native_u.max()) >= int(grid.native_w)):
        raise ValueError('Radial backprojection diameter map exceeds native Radial width')
    return DenseRadialBackprojectionMap(
        valid_mask=np.asarray(dense_map.valid_mask, dtype=bool),
        source_idx_map=np.asarray(dense_map.source_idx_map, dtype=np.int32),
        u_idx_map=np.ascontiguousarray(grid.native_u_to_processing[native_u]),
    )

_DENSE_RADIAL_BACKPROJECT_MAP_CACHE: Dict[Tuple[object, ...], DenseRadialBackprojectionMap] = {}

def radial_full_coverage_angle_deg(diameter: int) -> float:
    """Return the radial spacing that gives approximately 1x ROI edge coverage."""
    diameter_i = max(1, int(diameter))
    return float(360.0 / (math.pi * float(diameter_i)))

def _radial_view_nominal_spacing_deg(radial_view: ViewInfo) -> float:
    az = list(float(a) for a in radial_view.azimuths_deg)
    if len(az) >= 2:
        return float(abs(az[1] - az[0]))
    return 180.0

def _nearest_radial_source_for_backprojection(
    target_angle_deg: float,
    source_angles_deg: np.ndarray,
) -> Tuple[int, bool]:
    """Nearest source plane over the unoriented [0, 180) radial diameter domain.

 When the nearest match crosses the 0/180 boundary, the same diameter plane is reused with
 the radial coordinate reversed so source raster coordinates still map to the target plane."""
    if source_angles_deg.size <= 0:
        raise ValueError('No radial source angles are available for backprojection')

    raw = float(target_angle_deg) - source_angles_deg.astype(np.float64, copy=False)
    wrapped = ((raw + 90.0) % 180.0) - 90.0
    idx = int(np.argmin(np.abs(wrapped)))
    reverse_u = bool(abs(float(raw[idx])) > 90.0)
    return idx, reverse_u

def build_radial_backprojection_plan(radial_view: ViewInfo) -> Tuple[List[RadialBackprojectionSample], Dict[str, float]]:
    """Build the angular plan used to backproject a radial view into Cartesian space.

 If the user-requested radial spacing is coarser than the full-coverage spacing,
 backprojection is densified to the full-coverage spacing and each dense angle samples the
 nearest completed radial prediction frame. This keeps Radial masks view-native through
 postprocessing/interpolation while preventing sparse spoke-like Cartesian backprojections."""
    source_angles = np.asarray([float(a) for a in radial_view.azimuths_deg], dtype=np.float64)
    if source_angles.size <= 0:
        return [], {
            'provided_spacing_deg': 0.0,
            'coverage_spacing_deg': radial_full_coverage_angle_deg(int(radial_view.diameter)),
            'effective_spacing_deg': 0.0,
            'source_frames': 0.0,
            'backprojection_angles': 0.0,
            'densified': 0.0,
        }

    provided_spacing = _radial_view_nominal_spacing_deg(radial_view)
    coverage_spacing = radial_full_coverage_angle_deg(int(radial_view.diameter))
    # The specification's guarantee formula is a maximum safe angular spacing. A smaller
    # user spacing is already dense enough; a larger spacing is densified during backprojection.
    effective_spacing = min(float(provided_spacing), float(coverage_spacing))

    if float(provided_spacing) <= float(coverage_spacing) * (1.0 + 1e-9):
        samples = [
            RadialBackprojectionSample(angle_deg=float(angle), source_index=int(idx), reverse_u=False)
            for idx, angle in enumerate(source_angles.tolist())
        ]
        densified = False
    else:
        dense_angles = build_radial_azimuths(float(effective_spacing))
        samples = []
        for dense_angle in dense_angles:
            source_idx, reverse_u = _nearest_radial_source_for_backprojection(float(dense_angle), source_angles)
            samples.append(RadialBackprojectionSample(
                angle_deg=float(dense_angle),
                source_index=int(source_idx),
                reverse_u=bool(reverse_u),
            ))
        densified = True

    return samples, {
        'provided_spacing_deg': float(provided_spacing),
        'coverage_spacing_deg': float(coverage_spacing),
        'effective_spacing_deg': float(effective_spacing),
        'source_frames': float(source_angles.size),
        'backprojection_angles': float(len(samples)),
        'densified': 1.0 if bool(densified) else 0.0,
    }

def _radial_plan_signature(plan: Sequence[RadialBackprojectionSample]) -> Tuple[Tuple[float, int, bool], ...]:
    return tuple((round(float(s.angle_deg), 6), int(s.source_index), bool(s.reverse_u)) for s in plan)

def build_dense_radial_backprojection_map(
    radial_view: ViewInfo,
    plan: Sequence[RadialBackprojectionSample],
    *,
    out_shape_hw: Optional[Tuple[int, int]] = None,
) -> DenseRadialBackprojectionMap:
    """Map every selected-base-plane pixel to a Radial frame and diameter coordinate.

 ``out_shape_hw`` is expressed in the selected Radial base plane: ``(Y,X)`` for
 transverse, ``(t,X)`` for sagittal, and ``(t,Y)`` for coronal. Building the map
 directly on the final plane is what turns the working-cube circle into the required
 ellipse after the source t axis is restored."""
    work_plane_h, work_plane_w = radial_plane_shape(radial_view)
    if out_shape_hw is None:
        out_h, out_w = int(work_plane_h), int(work_plane_w)
    else:
        out_h, out_w = (int(out_shape_hw[0]), int(out_shape_hw[1]))
    if not plan:
        return DenseRadialBackprojectionMap(
            valid_mask=np.zeros((out_h, out_w), dtype=bool),
            source_idx_map=np.zeros((out_h, out_w), dtype=np.int32),
            u_idx_map=np.zeros((out_h, out_w), dtype=np.int32),
        )

    u_len = int(radial_view.src_w) if int(radial_view.src_w) > 0 else int(radial_view.diameter)
    key = (
        radial_base_view_name(radial_view), bool(is_tilted_radial_view(radial_view)),
        int(out_h), int(out_w), int(work_plane_h), int(work_plane_w),
        int(radial_view.diameter), int(u_len),
        round(float(radial_view.center_x), 6), round(float(radial_view.center_y), 6),
        round(float(radial_view.roi_radius), 6), _radial_plan_signature(plan),
    )
    cached = _DENSE_RADIAL_BACKPROJECT_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    diameter = int(u_len)
    radius = float(radial_view.roi_radius)
    if radius <= 0.0:
        radius = max(1.0, float(radial_view.diameter - 1) / 2.0)

    plan_angles = np.asarray([float(s.angle_deg) % 180.0 for s in plan], dtype=np.float32)
    plan_sources = np.asarray([int(s.source_index) for s in plan], dtype=np.int32)
    plan_reverses = np.asarray([bool(s.reverse_u) for s in plan], dtype=bool)
    n_plan = int(plan_angles.size)
    if n_plan <= 0:
        raise ValueError('Dense radial backprojection requires at least one angular plan sample')

    if n_plan >= 2:
        diffs = np.diff(plan_angles.astype(np.float64, copy=False))
        positive_diffs = diffs[diffs > 1e-9]
        step = float(np.median(positive_diffs)) if positive_diffs.size else 180.0 / float(n_plan)
    else:
        step = 180.0
    step = max(float(step), 1e-9)

    yy, xx = np.indices((out_h, out_w), dtype=np.float32)
    if (out_h, out_w) != (int(work_plane_h), int(work_plane_w)):
        xx = (xx + np.float32(0.5)) * np.float32(float(work_plane_w) / float(out_w)) - np.float32(0.5)
        yy = (yy + np.float32(0.5)) * np.float32(float(work_plane_h) / float(out_h)) - np.float32(0.5)
    dx = xx - float(radial_view.center_x)
    dy = yy - float(radial_view.center_y)
    rr = np.sqrt((dx * dx) + (dy * dy)).astype(np.float32, copy=False)
    valid = rr <= (float(radius) + 0.5)

    theta = np.degrees(np.arctan2(dy, dx)).astype(np.float32, copy=False)
    theta = np.mod(theta, 180.0).astype(np.float32, copy=False)
    nearest_plan_idx = np.mod(np.rint(theta / float(step)).astype(np.int32, copy=False), n_plan)

    target_angles = plan_angles[nearest_plan_idx]
    cos_t = np.cos(np.deg2rad(target_angles)).astype(np.float32, copy=False)
    sin_t = np.sin(np.deg2rad(target_angles)).astype(np.float32, copy=False)
    signed_r = (dx * cos_t) + (dy * sin_t)
    signed_r[plan_reverses[nearest_plan_idx]] *= -1.0

    u_float = ((signed_r + float(radius)) / max(1e-6, 2.0 * float(radius))) * float(diameter - 1)
    u_idx = np.clip(np.rint(u_float).astype(np.int32, copy=False), 0, diameter - 1)
    source_idx = plan_sources[nearest_plan_idx].astype(np.int32, copy=False)

    source_idx[~valid] = 0
    u_idx[~valid] = 0

    dense_map = DenseRadialBackprojectionMap(
        valid_mask=np.ascontiguousarray(valid),
        source_idx_map=np.ascontiguousarray(source_idx),
        u_idx_map=np.ascontiguousarray(u_idx),
    )
    _DENSE_RADIAL_BACKPROJECT_MAP_CACHE[key] = dense_map
    return dense_map

def main_process_gpu_stage_inference_overlap_enabled() -> bool:
    """Allow main-process GPU output stages to overlap worker inference on one device."""
    return _env_flag('YOLO_TTA_MAIN_GPU_STAGE_INFERENCE_OVERLAP', False)

def main_process_gpu_stage_inference_priority_enabled() -> bool:
    """Reserve worker GPUs for inference until the global inference queue is permanently drained.

    The old coordinator blocked an output stage only while a task was already queued or running
    on that exact device. A long NRRD mirror/backprojection stage could therefore win the small
    result-publication/refill race and strand that GPU while inference work still existed.
    """
    return _env_flag('YOLO_TTA_MAIN_GPU_STAGE_INFERENCE_PRIORITY', True)

class _MainProcessGpuStageLease:
    """Exclusive main-process lease for one logical CUDA device."""

    def __init__(self, coordinator: '_MainProcessGpuStageCoordinator', device_index: int, purpose: str) -> None:
        self._coordinator = coordinator
        self.device_index = int(device_index)
        self.purpose = str(purpose)
        self._released = False

    def torch_device(self, torch_mod: object) -> object:
        return torch_mod.device(f'cuda:{int(self.device_index)}')

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._coordinator.release_stage(self.device_index, self.purpose)

    def __enter__(self) -> '_MainProcessGpuStageLease':
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover - final safety net
        try:
            self.release()
        except Exception:
            pass

class _MainProcessGpuStageCoordinator:
    """Coordinate device ownership across independent CUDA allocator processes.

    A main-process output stage and an inference worker cannot safely admit against the
    same free-memory snapshot: either process may allocate immediately after the other
    samples it. Dispatch accounting therefore includes both queued and running worker tasks.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker_devices: set[int] = set()
        self._inference_inflight: Counter[int] = Counter()
        self._stage_leases: Dict[int, str] = {}
        self._inference_priority_active = False
        self._pending_inference_backlog = False
        self._wake_callback: Optional[Callable[[], None]] = None

    def configure_workers(self, worker_devices: Sequence[int]) -> None:
        with self._lock:
            self._worker_devices = {int(v) for v in worker_devices}
            self._inference_inflight.clear()
            self._stage_leases.clear()
            self._pending_inference_backlog = False
            self._inference_priority_active = bool(
                self._worker_devices and main_process_gpu_stage_inference_priority_enabled()
            )

    def set_pending_inference_backlog(self, active: bool) -> None:
        """Publish whether at least one central inference lease is dispatch-admissible."""
        callback: Optional[Callable[[], None]] = None
        with self._lock:
            changed = bool(self._pending_inference_backlog) != bool(active)
            self._pending_inference_backlog = bool(active)
            callback = self._wake_callback if changed else None
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def set_inference_priority_active(self, active: bool) -> None:
        callback: Optional[Callable[[], None]] = None
        with self._lock:
            self._inference_priority_active = bool(
                active and self._worker_devices and main_process_gpu_stage_inference_priority_enabled()
            )
            callback = self._wake_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def _priority_blocks_stage_locked(self, device_index: int, purpose: str) -> bool:
        purpose_l = str(purpose).strip().lower()
        # D1 is view-granular in v16.1.3: a completed view may run its Radial/Tilted
        # backprojection on a worker GPU whose compute credit is currently idle, while
        # other devices continue rendering/inference. Other main-process GPU stages retain
        # inference-first ownership until the global queue drains.
        if (
            v1613_d1_backprojection_overlap_enabled()
            and 'backprojection' in purpose_l
            and not bool(self._pending_inference_backlog)
        ):
            return False
        return bool(
            self._inference_priority_active
            and not main_process_gpu_stage_inference_overlap_enabled()
            and int(device_index) in self._worker_devices
        )

    def set_wake_callback(self, callback: Optional[Callable[[], None]]) -> None:
        with self._lock:
            self._wake_callback = callback

    def reset(self) -> None:
        with self._lock:
            self._worker_devices.clear()
            self._inference_inflight.clear()
            self._stage_leases.clear()
            self._inference_priority_active = False
            self._pending_inference_backlog = False
            self._wake_callback = None

    def can_dispatch_inference(self, device_index: int) -> bool:
        if main_process_gpu_stage_inference_overlap_enabled():
            return True
        with self._lock:
            return int(device_index) not in self._stage_leases

    def begin_inference(self, device_index: int) -> bool:
        device = int(device_index)
        with self._lock:
            if (
                not main_process_gpu_stage_inference_overlap_enabled()
                and device in self._stage_leases
            ):
                return False
            self._inference_inflight[device] += 1
            return True

    def finish_inference(self, device_index: int) -> None:
        device = int(device_index)
        with self._lock:
            remaining = int(self._inference_inflight.get(device, 0)) - 1
            if remaining > 0:
                self._inference_inflight[device] = remaining
            else:
                self._inference_inflight.pop(device, None)

    def try_acquire_specific_stage(
        self,
        torch_mod: object,
        device_index: int,
        purpose: str,
    ) -> Optional[_MainProcessGpuStageLease]:
        device = int(device_index)
        try:
            count = int(torch_mod.cuda.device_count())
        except Exception:
            return None
        if device < 0 or device >= count:
            return None
        with self._lock:
            overlap = bool(main_process_gpu_stage_inference_overlap_enabled())
            if device in self._stage_leases:
                return None
            if self._priority_blocks_stage_locked(device, purpose):
                return None
            if not overlap and int(self._inference_inflight.get(device, 0)) > 0:
                return None
            aux_pool = gpu_worker_aux_interpolation_pool()
            if aux_pool is not None and not bool(aux_pool.revoke_worker(device)):
                return None
            self._stage_leases[device] = str(purpose)
            return _MainProcessGpuStageLease(self, device, str(purpose))

    def try_acquire_stage(self, torch_mod: object, purpose: str) -> Optional[_MainProcessGpuStageLease]:
        try:
            count = int(torch_mod.cuda.device_count())
        except Exception:
            return None
        if count <= 0:
            return None
        with self._lock:
            configured = sorted(
                int(idx) for idx in self._worker_devices
                if 0 <= int(idx) < int(count)
            )
            eligible_devices = configured if configured else list(range(count))
            explicit = os.environ.get('YOLO_TTA_GPU_BACKPROJECT_DEVICE', '').strip()
            if explicit:
                try:
                    requested = int(explicit)
                except Exception:
                    return None
                candidates = [requested] if requested in eligible_devices else []
            else:
                candidates = list(eligible_devices)
            if not candidates:
                return None

            overlap = bool(main_process_gpu_stage_inference_overlap_enabled())
            available = [
                idx for idx in candidates
                if idx not in self._stage_leases
                and not self._priority_blocks_stage_locked(idx, purpose)
                and (overlap or int(self._inference_inflight.get(idx, 0)) == 0)
            ]
            if not available:
                return None
            # An idle worker may have been offered to the auxiliary interpolation pool.
            # Revoke that offer before assigning its physical GPU to a main-process stage;
            # a running auxiliary pass remains authoritative and makes the device ineligible.
            aux_pool = gpu_worker_aux_interpolation_pool()
            if aux_pool is not None:
                available = [
                    idx for idx in available
                    if bool(aux_pool.revoke_worker(int(idx)))
                ]
            if not available:
                return None
            best_index: Optional[int] = None
            best_free = -1
            for idx in available:
                try:
                    free_bytes, _total = torch_mod.cuda.mem_get_info(
                        torch_mod.device(f'cuda:{int(idx)}')
                    )
                except Exception:
                    continue
                if int(free_bytes) > int(best_free):
                    best_free = int(free_bytes)
                    best_index = int(idx)
            if best_index is None:
                return None
            self._stage_leases[int(best_index)] = str(purpose)
            return _MainProcessGpuStageLease(self, int(best_index), str(purpose))

    def release_stage(self, device_index: int, purpose: str) -> None:
        callback: Optional[Callable[[], None]] = None
        with self._lock:
            current = self._stage_leases.get(int(device_index))
            if current == str(purpose):
                self._stage_leases.pop(int(device_index), None)
            callback = self._wake_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                'worker_devices': sorted(self._worker_devices),
                'inference_inflight': dict(self._inference_inflight),
                'stage_leases': dict(self._stage_leases),
                'inference_priority_active': bool(self._inference_priority_active),
                'pending_inference_backlog': bool(self._pending_inference_backlog),
            }

_MAIN_PROCESS_GPU_STAGE_COORDINATOR = _MainProcessGpuStageCoordinator()

_MAIN_GPU_STAGE_SKIP_WARNED: set[str] = set()

_MAIN_GPU_STAGE_SKIP_WARNED_LOCK = threading.Lock()

def _configure_main_process_gpu_stage_workers(worker_devices: Sequence[int]) -> None:
    _MAIN_PROCESS_GPU_STAGE_COORDINATOR.configure_workers(worker_devices)

def _reset_main_process_gpu_stage_coordinator() -> None:
    _MAIN_PROCESS_GPU_STAGE_COORDINATOR.reset()

def _set_main_process_gpu_inference_priority_active(active: bool) -> None:
    _MAIN_PROCESS_GPU_STAGE_COORDINATOR.set_inference_priority_active(bool(active))

def _set_main_process_gpu_pending_inference(active: bool) -> None:
    _MAIN_PROCESS_GPU_STAGE_COORDINATOR.set_pending_inference_backlog(bool(active))

def _set_main_process_gpu_stage_wake_callback(callback: Optional[Callable[[], None]]) -> None:
    _MAIN_PROCESS_GPU_STAGE_COORDINATOR.set_wake_callback(callback)

def _main_process_gpu_stage_can_dispatch_inference(device_index: int) -> bool:
    return _MAIN_PROCESS_GPU_STAGE_COORDINATOR.can_dispatch_inference(int(device_index))

def _main_process_gpu_stage_begin_inference(device_index: int) -> bool:
    return _MAIN_PROCESS_GPU_STAGE_COORDINATOR.begin_inference(int(device_index))

def _main_process_gpu_stage_finish_inference(device_index: int) -> None:
    _MAIN_PROCESS_GPU_STAGE_COORDINATOR.finish_inference(int(device_index))

def _try_acquire_main_process_gpu_stage(
    torch_mod: object,
    purpose: str,
) -> Optional[_MainProcessGpuStageLease]:
    return _MAIN_PROCESS_GPU_STAGE_COORDINATOR.try_acquire_stage(torch_mod, str(purpose))

def _try_acquire_specific_main_process_gpu_stage(
    torch_mod: object,
    device_index: int,
    purpose: str,
) -> Optional[_MainProcessGpuStageLease]:
    return _MAIN_PROCESS_GPU_STAGE_COORDINATOR.try_acquire_specific_stage(
        torch_mod, int(device_index), str(purpose),
    )

def _announce_main_gpu_stage_skip_once(key: str, message: str) -> None:
    with _MAIN_GPU_STAGE_SKIP_WARNED_LOCK:
        if str(key) in _MAIN_GPU_STAGE_SKIP_WARNED:
            return
        _MAIN_GPU_STAGE_SKIP_WARNED.add(str(key))
    print(str(message))

def _trim_main_process_cuda_device(
    torch_mod: object,
    device: object,
    *,
    cupy_module: Optional[object] = None,
    desc: str = 'main-process GPU stage',
) -> None:
    """Return completed stage allocations to the driver for worker-process reuse."""
    before_reserved: Optional[int] = None
    before_free: Optional[int] = None
    after_reserved: Optional[int] = None
    after_free: Optional[int] = None
    try:
        before_reserved = int(torch_mod.cuda.memory_reserved(device))
    except Exception:
        pass
    try:
        before_free = int(torch_mod.cuda.mem_get_info(device)[0])
    except Exception:
        pass
    try:
        torch_mod.cuda.synchronize(device)
    except Exception:
        pass
    gc.collect()
    if cupy_module is not None:
        try:
            dev_index = getattr(device, 'index', None)
            if dev_index is None:
                dev_index = int(torch_mod.cuda.current_device())
            with cupy_module.cuda.Device(int(dev_index)):
                cupy_module.get_default_memory_pool().free_all_blocks()
                cupy_module.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    try:
        with torch_mod.cuda.device(device):
            torch_mod.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    try:
        after_reserved = int(torch_mod.cuda.memory_reserved(device))
    except Exception:
        pass
    try:
        after_free = int(torch_mod.cuda.mem_get_info(device)[0])
    except Exception:
        pass
    released = (
        max(0, int(before_reserved) - int(after_reserved))
        if before_reserved is not None and after_reserved is not None else 0
    )
    free_gain = (
        max(0, int(after_free) - int(before_free))
        if before_free is not None and after_free is not None else 0
    )
    if (
        released >= 256 * 1024 * 1024
        or free_gain >= 256 * 1024 * 1024
        or _env_flag('YOLO_TTA_GPU_MEMORY_RELEASE_LOG', False)
    ):
        dev_index = getattr(device, 'index', None)
        dev_label = f'cuda:{int(dev_index)}' if dev_index is not None else str(device)
        reserved_text = (
            f'{before_reserved / GIB:.2f}->{after_reserved / GIB:.2f} GiB'
            if before_reserved is not None and after_reserved is not None else 'unavailable'
        )
        free_text = (
            f'{before_free / GIB:.2f}->{after_free / GIB:.2f} GiB'
            if before_free is not None and after_free is not None else 'unavailable'
        )
        print(
            f'{desc}: released main-process CUDA cache on {dev_label}; '
            f'reserved={reserved_text}, driver_free={free_text}.'
        )

def _radial_row_occupancy(radial_mask_mm: np.ndarray, desc: str) -> np.ndarray:
    """Compute a per-stack-row foreground bitmap over an azimuth-major Radial volume."""
    n_az, work_rows, _u_len = (int(x) for x in radial_mask_mm.shape)
    row_any = np.zeros((work_rows,), dtype=bool)
    if n_az <= 0 or work_rows <= 0:
        return row_any
    lock = threading.Lock()
    scan_workers = max(1, min(16, _cpu_count(), n_az))
    az_chunk = max(1, int(math.ceil(float(n_az) / float(scan_workers * 4))))

    def _scan_range(range_idx: int) -> None:
        a0 = int(range_idx) * az_chunk
        a1 = min(n_az, a0 + az_chunk)
        local = np.zeros((work_rows,), dtype=bool)
        for a in range(a0, a1):
            plane = np.asarray(radial_mask_mm[a])
            np.logical_or(local, plane.any(axis=1), out=local)
        with lock:
            np.logical_or(row_any, local, out=row_any)

    num_ranges = int(math.ceil(float(n_az) / float(az_chunk)))
    parallel_for_indices(
        num_ranges,
        _scan_range,
        max_workers=scan_workers,
        desc=f'{desc} [row occupancy]',
        show_progress=False,
    )
    return row_any

def gpu_backproject_enabled() -> bool:
    """Stream radial backprojection through the GPU when torch + CUDA exist."""
    return _env_flag('YOLO_TTA_GPU_BACKPROJECT', True)

def _validated_radial_slice_bboxes(
    value: Optional[np.ndarray],
    n_az: int,
    work_t: int,
    u_len: int,
) -> Optional[np.ndarray]:
    """Validate interpolation-carried ``(t0,t1,u0,u1)`` bounds for radial planes."""
    if value is None:
        return None
    try:
        bboxes = np.asarray(value, dtype=np.int64)
    except Exception:
        return None
    if tuple(int(v) for v in bboxes.shape) != (int(n_az), 4):
        return None
    valid = (
        (bboxes[:, 0] >= 0) & (bboxes[:, 0] <= bboxes[:, 1]) & (bboxes[:, 1] <= int(work_t))
        & (bboxes[:, 2] >= 0) & (bboxes[:, 2] <= bboxes[:, 3]) & (bboxes[:, 3] <= int(u_len))
    )
    if not bool(np.all(valid)):
        return None
    return np.ascontiguousarray(bboxes)

def _trt_engine_from_autobackend(backend: object) -> Optional[object]:
    """Find the ICudaEngine without relying on one Ultralytics release's attribute names."""
    candidates = [
        getattr(backend, 'model', None),
        getattr(backend, 'trt_engine', None),
        getattr(backend, '_engine', None),
        getattr(getattr(backend, 'context', None), 'engine', None),
    ]
    for candidate in candidates:
        if candidate is None or isinstance(candidate, (bool, str, Path)):
            continue
        if callable(getattr(candidate, 'create_execution_context', None)):
            return candidate
    return None

def _torch_dtype_for_trt_binding(backend: object, engine: object, name: str, torch_mod: object) -> Optional[object]:
    """Resolve a binding dtype, preferring AutoBackend's already-validated torch buffers."""
    try:
        binding = getattr(backend, 'bindings', {}).get(str(name))
        data = getattr(binding, 'data', None)
        if data is not None and getattr(data, 'dtype', None) is not None:
            return data.dtype
    except Exception:
        pass
    try:
        import tensorrt as trt  # type: ignore
        dtype_obj = (
            engine.get_tensor_dtype(str(name))
            if callable(getattr(engine, 'get_tensor_dtype', None))
            else engine.get_binding_dtype(int(engine.get_binding_index(str(name))))
        )
        np_dtype = np.dtype(trt.nptype(dtype_obj))
        mapping = {
            np.dtype(np.float16): torch_mod.float16,
            np.dtype(np.float32): torch_mod.float32,
            np.dtype(np.int8): torch_mod.int8,
            np.dtype(np.int32): torch_mod.int32,
            np.dtype(np.int64): torch_mod.int64,
            np.dtype(np.bool_): torch_mod.bool,
        }
        return mapping.get(np_dtype)
    except Exception:
        return None

def _trt_binding_layout_for_backend(
    backend: object, engine: object,
) -> Tuple[List[str], str, List[str], Dict[str, int]]:
    """Resolve TensorRT I/O names without allocating a resident execution context.

 The ring input dtype must be known before its static-address slots are allocated, so
 layout discovery cannot live solely in ``_ResidentTensorRTRingExecutor.__init__``."""
    names: List[str] = []
    indices: Dict[str, int] = {}
    if callable(getattr(engine, 'get_tensor_name', None)):
        count = int(getattr(engine, 'num_io_tensors'))
        for i in range(count):
            name = str(engine.get_tensor_name(i))
            names.append(name)
            indices[name] = int(i)
    elif callable(getattr(engine, 'get_binding_name', None)):
        count = int(getattr(engine, 'num_bindings'))
        for i in range(count):
            name = str(engine.get_binding_name(i))
            names.append(name)
            indices[name] = int(i)
    else:
        raise RuntimeError('unknown TensorRT binding API')

    backend_outputs = [str(x) for x in (getattr(backend, 'output_names', None) or [])]
    input_name = str(getattr(backend, 'input_name', '') or '')
    if input_name not in names:
        input_candidates: List[str] = []
        for name in names:
            try:
                if callable(getattr(engine, 'binding_is_input', None)):
                    is_input = bool(engine.binding_is_input(indices[name]))
                else:
                    is_input = 'INPUT' in str(engine.get_tensor_mode(name)).upper()
                if is_input:
                    input_candidates.append(name)
            except Exception:
                pass
        if len(input_candidates) != 1:
            input_candidates = [name for name in names if name not in set(backend_outputs)]
        if len(input_candidates) != 1:
            raise RuntimeError(f'expected one TensorRT image input, found {input_candidates}')
        input_name = input_candidates[0]
    output_names = [name for name in backend_outputs if name in names]
    if not output_names:
        output_names = [name for name in names if name != input_name]
    if set(output_names) | {input_name} != set(names):
        raise RuntimeError('TensorRT engine has auxiliary bindings unsupported by the resident ring')
    return names, input_name, output_names, indices

class _ResidentTensorRTRingFatalError(RuntimeError):
    """Ring teardown could not prove the borrowed backend/buffers safe for fallback."""

class _ResidentTensorRTRingExecutor:
    """Two independent TensorRT contexts with static input/output addresses.

 This adapter intentionally uses TensorRT's public execution-context API instead of
 swapping ``AutoBackend.context`` in place. If the engine, binding metadata, async
 API, or fixed shapes cannot be proved safe, construction fails before source
 consumption and the caller retains the ordinary direct-predict path."""

    @runtime_telemetry_phase('trt_ring.init')
    def __init__(
        self,
        backend: object,
        slots: Sequence[_ResidentGpuPipelineSlot],
        *,
        input_channels: int,
        out_size: int,
        native_h: int,
        native_w: int,
        M_out_to_native: np.ndarray,
        track_conf: bool,
        confidence_threshold: float,
        dynamic_unit_descriptors: bool = False,
    ) -> None:
        import torch  # type: ignore
        self.torch = torch
        self._closed = False
        self.backend = backend
        self.device = slots[0].input.device if slots else torch.device('cuda:0')
        self.engine = _trt_engine_from_autobackend(backend)
        if self.engine is None:
            raise RuntimeError('AutoBackend does not expose a TensorRT ICudaEngine')
        if len(slots) != 2:
            raise RuntimeError('resident TensorRT ring requires exactly two slots')
        self.slots = list(slots)
        self.input_channels = int(input_channels)
        self.out_size = int(out_size)
        self.native_h = int(native_h)
        self.native_w = int(native_w)
        self.dynamic_unit_descriptors = bool(dynamic_unit_descriptors)
        if (
            self.input_channels <= 0 or self.out_size <= 0
            or self.native_h <= 0 or self.native_w <= 0
        ):
            raise RuntimeError(
                f'invalid resident ring geometry C={self.input_channels}, network={self.out_size}, '
                f'native={self.native_h}x{self.native_w}'
            )
        expected_input_shape = (
            1, self.input_channels, self.out_size, self.out_size,
        )
        slot_shapes = [
            tuple(int(x) for x in slot.input.shape) for slot in self.slots
        ]
        if any(shape != expected_input_shape for shape in slot_shapes):
            raise RuntimeError(
                f'resident ring slot shapes {slot_shapes} do not match '
                f'expected TensorRT input {expected_input_shape}'
            )
        self.default_descriptor = ResidentRingUnitDescriptor(
            unit_index=0,
            destination_index=0,
            native_h=int(self.native_h),
            native_w=int(self.native_w),
            M_out_to_native=np.asarray(M_out_to_native, dtype=np.float32).reshape(2, 3),
        )
        self.identity_native_warp, self.native_to_out = self._descriptor_warp(
            self.default_descriptor,
        )
        self.track_conf = bool(track_conf)
        self.confidence_threshold = float(confidence_threshold)
        requested_proto_mode = proto_hole_treatment_mode()
        # Confidence-based component cleanup expects an untouched confidence/mask relation;
        # the prioritized D1 command does not allocate a confidence map.
        self.proto_hole_treatment = (
            requested_proto_mode if requested_proto_mode == 'close' and not self.track_conf else 'off'
        )
        self.proto_hole_radius = int(proto_hole_treatment_radius())
        self.proto_hole_treatment_active = bool(
            self.proto_hole_treatment == 'close' and self.proto_hole_radius > 0
        )
        self.kernels = _resident_mask_kernels()
        if self.kernels is None:
            raise RuntimeError('CUDA confidence-compaction kernel is unavailable')

        self.binding_names, self.input_name, self.output_names, self.binding_indices = self._binding_layout()
        if len(self.output_names) < 2:
            raise RuntimeError(f'TensorRT segmentation engine exposes only {len(self.output_names)} outputs')
        input_dtype = _torch_dtype_for_trt_binding(backend, self.engine, self.input_name, torch)
        if input_dtype not in (torch.float16, torch.float32):
            raise RuntimeError(
                f'resident ring kernels do not support TensorRT input dtype {input_dtype}'
            )
        if any(slot.input.dtype != input_dtype for slot in self.slots):
            raise RuntimeError(
                f'TensorRT input dtype {input_dtype} does not match resident ring slot dtypes '
                f'{[slot.input.dtype for slot in self.slots]}'
            )

        # AutoBackend already owns one context. Borrow it plus ONE new context so the
        # worker has two total (not three profile-sized context allocations). tensor
        # addresses are restored before returning control to AutoBackend.
        original_context = getattr(backend, 'context', None)
        if original_context is None:
            raise RuntimeError('AutoBackend does not expose its primary TensorRT context')
        second_context = self.engine.create_execution_context()
        if second_context is None:
            raise RuntimeError('TensorRT could not allocate a second independent execution context')
        contexts: List[object] = [original_context, second_context]
        if contexts[0] is contexts[1]:
            raise RuntimeError('TensorRT returned an aliased execution context')

        self._borrowed_context = original_context
        self._restore_tensor_addresses: Dict[str, int] = {}
        if callable(getattr(original_context, 'set_tensor_address', None)):
            for name in self.binding_names:
                binding = getattr(backend, 'bindings', {}).get(str(name))
                data = getattr(binding, 'data', None)
                if data is None or not callable(getattr(data, 'data_ptr', None)):
                    raise RuntimeError(
                        f'cannot safely restore AutoBackend TensorRT address for {name!r}'
                    )
                self._restore_tensor_addresses[name] = int(data.data_ptr())

        self.infer_graph_count = 0
        self.post_graph_count = 0
        try:
            for slot, context in zip(self.slots, contexts):
                slot.context = context
                output_tensors = self._configure_context_and_buffers(slot)
                split = _split_segmentation_backend_outputs(output_tensors)
                if split is None:
                    # Output order is not guaranteed across TensorRT exporter versions; identify
                    # the detection head/proto by rank, then preserve the direct-loop contract.
                    rank3 = [x for x in output_tensors if int(getattr(x, 'ndim', 0)) == 3]
                    rank4 = [x for x in output_tensors if int(getattr(x, 'ndim', 0)) == 4]
                    if len(rank3) != 1 or len(rank4) != 1:
                        raise RuntimeError('TensorRT outputs cannot be identified as head + proto')
                    split = (rank3[0], rank4[0])
                slot.head, slot.proto = split
                if slot.head.dtype not in (torch.float16, torch.float32):
                    raise RuntimeError(
                        f'resident ring kernels do not support TensorRT head dtype {slot.head.dtype}'
                    )
                if slot.proto.dtype not in (torch.float16, torch.float32):
                    raise RuntimeError(
                        f'resident ring kernels do not support TensorRT proto dtype {slot.proto.dtype}'
                    )
                self._allocate_post_buffers(slot)
                self._set_slot_unit_descriptor(slot, self.default_descriptor)

            # Shape/layout invariants for the user's single-class segmentation engines.
            for slot in self.slots:
                head, proto = slot.head, slot.proto
                if int(head.shape[0]) != 1 or int(proto.shape[0]) != 1:
                    raise RuntimeError('resident TensorRT ring is fixed to batch 1')
                nm = int(proto.shape[1])
                if int(head.shape[1]) != 5 + nm:
                    raise RuntimeError(
                        f'resident TensorRT ring requires one class (head={tuple(head.shape)}, proto={tuple(proto.shape)})'
                    )
                if tuple(int(x) for x in head.shape) != tuple(int(x) for x in self.slots[0].head.shape):
                    raise RuntimeError('TensorRT ring contexts returned different head shapes')
                if tuple(int(x) for x in proto.shape) != tuple(int(x) for x in self.slots[0].proto.shape):
                    raise RuntimeError('TensorRT ring contexts returned different proto shapes')

            # Prove that both contexts may be in flight on distinct streams before the
            # source is consumed. Dynamic-profile restrictions and plugin stream-safety
            # failures surface at stream synchronization and cleanly reject this fast path.
            self._validate_concurrent_contexts()
            # Runtime warmups are mandatory even when CUDA graphs are explicitly disabled:
            # the post warmup is the pre-consumption proof that the compaction/union/affine
            # kernels accept this engine's actual head/proto dtypes and layouts.
            self._capture_static_graphs(
                capture_graphs=bool(resident_trt_cuda_graphs_enabled()),
            )
        except Exception:
            self.close()
            raise

    def _binding_layout(self) -> Tuple[List[str], str, List[str], Dict[str, int]]:
        return _trt_binding_layout_for_backend(self.backend, self.engine)

    def _descriptor_warp(
        self,
        descriptor: ResidentRingUnitDescriptor,
    ) -> Tuple[bool, Tuple[np.float32, ...]]:
        if (
            int(descriptor.native_h) != int(self.native_h)
            or int(descriptor.native_w) != int(self.native_w)
        ):
            raise RuntimeError(
                f'resident ring unit {int(descriptor.unit_index)} destination geometry '
                f'{int(descriptor.native_h)}x{int(descriptor.native_w)} does not match '
                f'static slot geometry {int(self.native_h)}x{int(self.native_w)}'
            )
        try:
            out_to_native = np.asarray(
                descriptor.M_out_to_native, dtype=np.float64,
            ).reshape(2, 3)
            native_to_out3 = np.linalg.inv(_affine2x3_to_3x3(out_to_native))
        except Exception as exc:
            raise RuntimeError(
                f'resident ring unit {int(descriptor.unit_index)} affine is singular or malformed'
            ) from exc
        if not bool(np.isfinite(native_to_out3).all()):
            raise RuntimeError(
                f'resident ring unit {int(descriptor.unit_index)} affine is non-finite'
            )
        identity = bool(
            self.native_h == self.out_size
            and self.native_w == self.out_size
            and _warp_matrix_is_identity(out_to_native)
        )
        inverse = tuple(np.float32(x) for x in native_to_out3[:2, :3].reshape(-1))
        return identity, inverse

    def _set_slot_unit_descriptor(
        self,
        slot: _ResidentGpuPipelineSlot,
        descriptor: ResidentRingUnitDescriptor,
    ) -> None:
        identity, inverse = self._descriptor_warp(descriptor)
        slot.unit_descriptor = descriptor
        slot.identity_native_warp = bool(identity)
        slot.native_to_out = inverse
        if self.dynamic_unit_descriptors:
            # Post graph scalar arguments are captured by value.  Dynamic tile/unit affines
            # retain the static TensorRT graphs but launch the compact post kernel normally.
            slot.post_graph = None

    def _set_input_shape(self, context: object, slot: _ResidentGpuPipelineSlot) -> None:
        shape = tuple(int(x) for x in slot.input.shape)
        if callable(getattr(context, 'set_input_shape', None)):
            ok = context.set_input_shape(self.input_name, shape)
            if ok is False:
                raise RuntimeError(f'TensorRT rejected static input shape {shape}')
        elif callable(getattr(context, 'set_binding_shape', None)):
            ok = context.set_binding_shape(self.binding_indices[self.input_name], shape)
            if ok is False:
                raise RuntimeError(f'TensorRT rejected static binding shape {shape}')

    def _context_shape(self, context: object, name: str) -> Tuple[int, ...]:
        if callable(getattr(context, 'get_tensor_shape', None)):
            shape = tuple(int(x) for x in context.get_tensor_shape(name))
        else:
            shape = tuple(int(x) for x in context.get_binding_shape(self.binding_indices[name]))
        if not shape or any(int(x) <= 0 for x in shape):
            raise RuntimeError(f'TensorRT binding {name!r} has unresolved shape {shape}')
        return shape

    def _configure_context_and_buffers(self, slot: _ResidentGpuPipelineSlot) -> List[object]:
        torch = self.torch
        context = slot.context
        self._set_input_shape(context, slot)
        outputs: Dict[str, object] = {}
        for name in self.output_names:
            dtype = _torch_dtype_for_trt_binding(self.backend, self.engine, name, torch)
            if dtype is None:
                raise RuntimeError(f'cannot resolve TensorRT dtype for {name!r}')
            outputs[name] = torch.empty(
                self._context_shape(context, name), dtype=dtype, device=slot.input.device,
            )

        if callable(getattr(context, 'set_tensor_address', None)) and callable(getattr(context, 'execute_async_v3', None)):
            for name in self.binding_names:
                tensor = slot.input if name == self.input_name else outputs[name]
                ok = context.set_tensor_address(name, int(tensor.data_ptr()))
                if ok is False:
                    raise RuntimeError(f'TensorRT rejected static address for {name!r}')
            slot.binding_addresses = None
        elif callable(getattr(context, 'execute_async_v2', None)):
            addresses = [0] * len(self.binding_names)
            for name in self.binding_names:
                tensor = slot.input if name == self.input_name else outputs[name]
                addresses[self.binding_indices[name]] = int(tensor.data_ptr())
            if any(int(x) == 0 for x in addresses):
                raise RuntimeError('TensorRT v2 binding address table is incomplete')
            slot.binding_addresses = addresses
        else:
            raise RuntimeError('TensorRT context has no asynchronous enqueue API')
        return [outputs[name] for name in self.output_names]

    def _allocate_post_buffers(self, slot: _ResidentGpuPipelineSlot) -> None:
        torch = self.torch
        head, proto = slot.head, slot.proto
        anchors = int(head.shape[2])
        ph, pw = int(proto.shape[2]), int(proto.shape[3])
        device = head.device
        slot.compact_indices = torch.empty((anchors,), dtype=torch.int32, device=device)
        slot.compact_count = torch.zeros((1,), dtype=torch.int32, device=device)
        slot.max_logit = torch.empty((ph, pw), dtype=torch.float32, device=device)
        slot.proto_tmp = (
            torch.empty((ph, pw), dtype=torch.float32, device=device)
            if self.proto_hole_treatment_active else None
        )
        slot.conf_proto = (
            torch.empty((ph, pw), dtype=torch.float32, device=device) if self.track_conf else None
        )
        self._allocate_native_output_buffers(slot)
        cp = self.kernels.cp
        refs = {
            'head': cp.asarray(head[0]),
            'proto': cp.asarray(proto[0]),
            'indices': cp.asarray(slot.compact_indices),
            'count': cp.asarray(slot.compact_count),
            'max_logit': cp.asarray(slot.max_logit),
            'native_union': cp.asarray(slot.native_union),
        }
        if slot.proto_tmp is not None:
            refs['proto_tmp'] = cp.asarray(slot.proto_tmp)
        if slot.conf_proto is not None:
            refs['conf_proto'] = cp.asarray(slot.conf_proto)
        if slot.native_conf is not None:
            refs['native_conf'] = cp.asarray(slot.native_conf)
        slot._cupy_refs = refs

    def _allocate_native_output_buffers(self, slot: _ResidentGpuPipelineSlot) -> None:
        torch = self.torch
        device = slot.head.device
        slot.native_union = torch.empty(
            (int(self.native_h), int(self.native_w)), dtype=torch.uint8, device=device,
        )
        slot.native_conf = (
            torch.empty((int(self.native_h), int(self.native_w)), dtype=torch.uint8, device=device)
            if self.track_conf else None
        )

    def reconfigure_destination(
        self,
        *,
        native_h: int,
        native_w: int,
        M_out_to_native: np.ndarray,
        dynamic_unit_descriptors: bool,
    ) -> None:
        """Retarget postprocessing while retaining TensorRT contexts and bindings."""
        new_h, new_w = int(native_h), int(native_w)
        matrix = np.asarray(M_out_to_native, dtype=np.float32).reshape(2, 3)
        same_geometry = bool(new_h == int(self.native_h) and new_w == int(self.native_w))
        same_matrix = bool(np.array_equal(matrix, self.default_descriptor.M_out_to_native))
        effective_dynamic = True  # post scalar arguments must remain task-dynamic.
        if same_geometry and same_matrix and bool(self.dynamic_unit_descriptors) == effective_dynamic:
            return
        self.synchronize()
        self.native_h = int(new_h)
        self.native_w = int(new_w)
        self.dynamic_unit_descriptors = bool(effective_dynamic)
        self.default_descriptor = ResidentRingUnitDescriptor(
            unit_index=0,
            destination_index=0,
            native_h=int(new_h),
            native_w=int(new_w),
            M_out_to_native=matrix,
        )
        self.identity_native_warp, self.native_to_out = self._descriptor_warp(self.default_descriptor)
        for slot in self.slots:
            # Infer graphs and TRT output bindings are geometry-independent.  Only the
            # post destination tensors/CuPy views and captured post graph are replaced.
            slot.post_graph = None
            slot._cupy_refs.pop('native_union', None)
            slot._cupy_refs.pop('native_conf', None)
            slot.native_union = None
            slot.native_conf = None
            self._allocate_native_output_buffers(slot)
            cp = self.kernels.cp
            slot._cupy_refs['native_union'] = cp.asarray(slot.native_union)
            if slot.native_conf is not None:
                slot._cupy_refs['native_conf'] = cp.asarray(slot.native_conf)
            self._set_slot_unit_descriptor(slot, self.default_descriptor)
            slot.infer_valid = False
            slot.post_valid = False
        runtime_telemetry().add('trt_ring.destination_reconfigurations', 1)

    def _execute_context(self, slot: _ResidentGpuPipelineSlot) -> None:
        context = slot.context
        handle = int(slot.infer_stream.cuda_stream)
        if slot.binding_addresses is None:
            try:
                ok = context.execute_async_v3(stream_handle=handle)
            except TypeError:
                ok = context.execute_async_v3(handle)
        else:
            try:
                ok = context.execute_async_v2(bindings=slot.binding_addresses, stream_handle=handle)
            except TypeError:
                ok = context.execute_async_v2(slot.binding_addresses, handle)
        if ok is False:
            raise RuntimeError(f'TensorRT enqueue failed for ring slot {slot.slot_id}')

    def _launch_post(self, slot: _ResidentGpuPipelineSlot) -> None:
        cp = self.kernels.cp
        refs = slot._cupy_refs
        head, proto = slot.head, slot.proto
        anchors = int(head.shape[2])
        masks = int(proto.shape[1])
        ph, pw = int(proto.shape[2]), int(proto.shape[3])
        external = _cupy_external_stream(cp, slot.post_stream)
        # _launch_post is always entered under torch.cuda.stream(post_stream), so this
        # clear and the external CuPy launches share the same capture-safe CUDA stream.
        slot.compact_count.zero_()
        compact = self.kernels.compact_f16 if head.dtype == self.torch.float16 else self.kernels.compact_f32
        compact(
            ((anchors + 255) // 256,), (256,),
            (refs['head'], np.int32(anchors), np.float32(self.confidence_threshold), refs['indices'], refs['count']),
            stream=external,
        )
        htag = 'f16' if head.dtype == self.torch.float16 else 'f32'
        ptag = 'f16' if proto.dtype == self.torch.float16 else 'f32'
        union_kernel = getattr(self.kernels, f'union_{htag}_{ptag}')
        conf_proto_arg = refs.get('conf_proto', np.uintp(0))
        pixels = ph * pw
        union_kernel(
            ((pixels + 255) // 256,), (256,),
            (
                refs['head'], refs['proto'], refs['indices'], refs['count'],
                np.int32(anchors), np.int32(masks), np.int32(ph), np.int32(pw),
                np.int32(self.out_size), np.int32(self.out_size), refs['max_logit'], conf_proto_arg,
            ),
            stream=external,
        )
        if self.proto_hole_treatment_active:
            self.kernels.proto_threshold_signed(
                ((pixels + 255) // 256,), (256,),
                (refs['max_logit'], np.int32(ph), np.int32(pw), refs['proto_tmp']),
                stream=external,
            )
            self.kernels.proto_dilate_signed(
                ((pixels + 255) // 256,), (256,),
                (refs['proto_tmp'], np.int32(ph), np.int32(pw),
                 np.int32(self.proto_hole_radius), refs['max_logit']),
                stream=external,
            )
            self.kernels.proto_erode_signed(
                ((pixels + 255) // 256,), (256,),
                (refs['max_logit'], np.int32(ph), np.int32(pw),
                 np.int32(self.proto_hole_radius), refs['proto_tmp']),
                stream=external,
            )
        post_logit_arg = refs['proto_tmp'] if self.proto_hole_treatment_active else refs['max_logit']
        if slot.unit_descriptor is None:
            raise RuntimeError(f'resident ring slot {slot.slot_id} has no unit descriptor')
        out_pixels = self.native_h * self.native_w
        if slot.identity_native_warp:
            # Preserve the original radial/identity specialization byte-for-byte: proto
            # bilinear upsample to out_size, threshold, and nearest confidence sampling.
            self.kernels.upsample_quantize(
                ((out_pixels + 255) // 256,), (256,),
                (
                    post_logit_arg, conf_proto_arg, np.int32(ph), np.int32(pw),
                    np.int32(self.out_size), np.int32(self.out_size), refs['native_union'],
                    refs.get('native_conf', np.uintp(0)),
                ),
                stream=external,
            )
        else:
            self.kernels.upsample_quantize_affine(
                ((out_pixels + 255) // 256,), (256,),
                (
                    post_logit_arg, conf_proto_arg, np.int32(ph), np.int32(pw),
                    np.int32(self.out_size), np.int32(self.out_size),
                    np.int32(self.native_h), np.int32(self.native_w),
                    *slot.native_to_out,
                    refs['native_union'], refs.get('native_conf', np.uintp(0)),
                ),
                stream=external,
            )

    def _validate_concurrent_contexts(self) -> None:
        """Enqueue both contexts concurrently before admitting the specialized source."""
        torch = self.torch
        for slot in self.slots:
            with torch.cuda.stream(slot.infer_stream):
                slot.input.zero_()
                self._execute_context(slot)
        # Synchronize only after both enqueues. TensorRT reports several profile/plugin
        # incompatibilities asynchronously, so checking the enqueue return alone is not
        # sufficient evidence that the two-context schedule is valid.
        first_error: Optional[BaseException] = None
        for slot in self.slots:
            try:
                slot.infer_stream.synchronize()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise RuntimeError('resident TensorRT concurrent-context validation failed') from first_error

    def _capture_static_graphs(self, *, capture_graphs: bool = True) -> None:
        torch = self.torch
        # Runtime warmup is a capability test; graph capture itself is opportunistic.
        # Never confuse a failed TensorRT/CuPy launch with an unsupported CUDAGraph API:
        # the former must reject the ring before the source is consumed.
        for slot in self.slots:
            try:
                with torch.cuda.stream(slot.infer_stream):
                    slot.input.zero_()
                    self._execute_context(slot)
                slot.infer_stream.synchronize()
            except BaseException as exc:
                raise RuntimeError(
                    f'resident TensorRT infer warmup failed for ring slot {slot.slot_id}'
                ) from exc
            if bool(capture_graphs):
                try:
                    graph = torch.cuda.CUDAGraph()
                    with _cuda_graph_capture_context(torch, graph, slot.infer_stream):
                        self._execute_context(slot)
                    slot.infer_stream.synchronize()
                    slot.infer_graph = graph
                    self.infer_graph_count += 1
                except Exception:
                    slot.infer_graph = None
                    try:
                        slot.infer_stream.synchronize()
                    except BaseException as sync_exc:
                        raise _ResidentTensorRTRingFatalError(
                            f'failed to recover TensorRT infer stream after graph capture '
                            f'for ring slot {slot.slot_id}'
                        ) from sync_exc
            try:
                with torch.cuda.stream(slot.post_stream):
                    self._launch_post(slot)
                slot.post_stream.synchronize()
            except BaseException as exc:
                raise RuntimeError(
                    f'resident TensorRT post-kernel warmup failed for ring slot {slot.slot_id}'
                ) from exc
            if bool(capture_graphs) and not self.dynamic_unit_descriptors:
                try:
                    graph = torch.cuda.CUDAGraph()
                    with _cuda_graph_capture_context(torch, graph, slot.post_stream):
                        self._launch_post(slot)
                    slot.post_stream.synchronize()
                    slot.post_graph = graph
                    self.post_graph_count += 1
                except Exception:
                    slot.post_graph = None
                    try:
                        slot.post_stream.synchronize()
                    except BaseException as sync_exc:
                        raise _ResidentTensorRTRingFatalError(
                            f'failed to recover post stream after graph capture '
                            f'for ring slot {slot.slot_id}'
                        ) from sync_exc

    def enqueue_inference(self, slot: _ResidentGpuPipelineSlot) -> None:
        torch = self.torch
        with torch.cuda.stream(slot.infer_stream):
            slot.infer_stream.wait_event(slot.render_done)
            if bool(slot.post_valid):
                # The preceding frame's mask kernel still reads this context's static
                # output buffers. Do not let the next enqueue overwrite them early.
                slot.infer_stream.wait_event(slot.post_done)
            if slot.infer_graph is not None:
                slot.infer_graph.replay()
            else:
                self._execute_context(slot)
            slot.infer_done.record(slot.infer_stream)
        slot.infer_valid = True

    def enqueue_postprocess(
        self,
        slot: _ResidentGpuPipelineSlot,
        *,
        descriptor: ResidentRingUnitDescriptor,
        device_union: '_DeviceUnionAccumulator',
        frame_counts_dev: object,
    ) -> None:
        self._set_slot_unit_descriptor(slot, descriptor)
        unit_index = int(descriptor.unit_index)
        destination_index = int(descriptor.destination_index)
        torch = self.torch
        device_union.register_producer_stream(slot.post_stream)
        with torch.cuda.stream(slot.post_stream):
            slot.post_stream.wait_event(slot.infer_done)
            if slot.post_graph is not None:
                slot.post_graph.replay()
            else:
                self._launch_post(slot)
            # The static slot output is copied into the task-resident destination before
            # the slot is released. No payload/result object or per-frame Future exists.
            device_union.union_dev[destination_index].copy_(slot.native_union, non_blocking=True)
            if device_union.conf_dev is not None and slot.native_conf is not None:
                device_union.conf_dev[destination_index].copy_(slot.native_conf, non_blocking=True)
            frame_counts_dev[unit_index].copy_(slot.compact_count[0], non_blocking=True)
            slot.post_done.record(slot.post_stream)
        device_union.written[destination_index] = True
        slot.post_valid = True

    def synchronize(self) -> None:
        for slot in self.slots:
            if bool(slot.post_valid):
                slot.post_done.synchronize()

    def reset_for_task(self) -> None:
        """Drain one task and retain contexts, bindings, buffers, and captured graphs."""
        if bool(getattr(self, '_closed', False)):
            raise RuntimeError('resident TensorRT executor is already closed')
        self.synchronize()
        for slot in self.slots:
            slot.infer_valid = False
            slot.post_valid = False
            slot.frame_index = -1
            slot.absolute_index = -1
            slot.synthetic = False
            self._set_slot_unit_descriptor(slot, self.default_descriptor)

    def close(self) -> None:
        """Drain ring streams, then restore borrowed AutoBackend binding addresses.

 A failed drain or partial address restore is fatal: generic fallback must never run
 while a context can still write ring buffers or the borrowed backend points at storage
 this method is about to release."""
        if bool(getattr(self, '_closed', False)):
            return
        self._closed = True
        drain_error: Optional[BaseException] = None
        for slot in getattr(self, 'slots', []):
            for stream in (getattr(slot, 'infer_stream', None), getattr(slot, 'post_stream', None)):
                if stream is None:
                    continue
                try:
                    stream.synchronize()
                except BaseException as exc:
                    if drain_error is None:
                        drain_error = exc
        context = getattr(self, '_borrowed_context', None)
        restore = getattr(self, '_restore_tensor_addresses', {})
        restore_error: Optional[BaseException] = None
        if context is not None and restore:
            for name, address in restore.items():
                try:
                    ok = context.set_tensor_address(str(name), int(address))
                    if ok is False:
                        raise RuntimeError(f'TensorRT rejected restored address for {name!r}')
                except BaseException as exc:
                    if restore_error is None:
                        restore_error = exc
        # Release graph/context/output references promptly; the task-level device union
        # remains live for metadata, hole fill, and its single D2H drain.
        for slot in getattr(self, 'slots', []):
            slot.infer_graph = None
            slot.post_graph = None
            slot.render_graph = None
            slot.render_graph_key = None
            slot.render_expected_key = None
            slot.context = None
            slot.binding_addresses = None
            slot.head = None
            slot.proto = None
            slot.compact_indices = None
            slot.compact_count = None
            slot.max_logit = None
            slot.proto_tmp = None
            slot.conf_proto = None
            slot.native_union = None
            slot.native_conf = None
            slot.unit_descriptor = None
            slot.native_to_out = tuple()
            slot._cupy_refs.clear()
            slot._render_cupy_refs.clear()
            slot.input = None
            slot.render_meta = None
            slot.render_done = None
            slot.infer_done = None
            slot.post_done = None
            slot.infer_stream = None
            slot.post_stream = None
        # The replacement TensorRT context is allocated outside PyTorch's caching
        # allocator. Return the now-unowned ring storage to the driver before that
        # context is created instead of merely leaving it reusable by PyTorch.
        try:
            self.slots.clear()
        except Exception:
            pass
        self._borrowed_context = None
        self._restore_tensor_addresses = {}
        gc.collect()
        try:
            cp = getattr(getattr(self, 'kernels', None), 'cp', None)
            if cp is not None:
                dev_idx = int(getattr(self.device, 'index', 0) or 0)
                with cp.cuda.Device(dev_idx):
                    cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()
        except Exception:
            pass
        if restore_error is not None:
            raise _ResidentTensorRTRingFatalError(
                'failed to restore every AutoBackend TensorRT binding address; fallback is unsafe'
            ) from restore_error
        if drain_error is not None:
            raise _ResidentTensorRTRingFatalError(
                'failed to drain every resident TensorRT ring stream; fallback is unsafe'
            ) from drain_error

_RESIDENT_TRT_PIPELINE_CACHE_LOCK = threading.RLock()

_RESIDENT_TRT_PIPELINE_CACHE: Dict[int, Dict[str, object]] = {}

_RESIDENT_TRT_PIPELINE_CACHE_NATIVE = True

_RESIDENT_TRT_RING_CACHE_HIT_ANNOUNCED = False

def resident_trt_pipeline_persistence_enabled() -> bool:
    return _env_flag('YOLO_TTA_PERSISTENT_TRT_RING', True)

def _resident_trt_pipeline_signature(
    backend: object,
    input_dtype: object,
    *,
    input_channels: int,
    out_size: int,
    native_h: int,
    native_w: int,
    M_out_to_native: np.ndarray,
    track_conf: bool,
    confidence_threshold: float,
    dynamic_unit_descriptors: bool,
) -> Tuple[object, ...]:
    engine = _trt_engine_from_autobackend(backend)
    # TensorRT bindings depend on engine/profile/input shape, not on the destination
    # mask raster.  Native geometry and affine are handled by the post kernel and are
    # reconfigured in place on cache hits.  Keeping them in this key made every view/
    # orientation transition destroy and recreate a 668 MiB execution context.
    _ = (native_h, native_w, M_out_to_native, dynamic_unit_descriptors)
    return (
        id(engine), str(input_dtype), int(input_channels), int(out_size),
        bool(track_conf), float(confidence_threshold),
        str(proto_hole_treatment_mode()), int(proto_hole_treatment_radius()),
    )

def _resident_trt_pipeline_invalidate(backend: object, reason: Optional[BaseException] = None) -> None:
    entry: Optional[Dict[str, object]] = None
    with _RESIDENT_TRT_PIPELINE_CACHE_LOCK:
        entry = _RESIDENT_TRT_PIPELINE_CACHE.pop(id(backend), None)
    if entry is None:
        return
    executor = entry.get('executor')
    try:
        close = getattr(executor, 'close', None)
        if callable(close):
            close()
    except _ResidentTensorRTRingFatalError:
        raise
    except Exception as exc:
        if reason is None:
            print(f'Warning: resident TensorRT pipeline teardown failed ({exc}).')

def _resident_trt_pipeline_decline(backend: Optional[object]) -> None:
    """Restore AutoBackend before any non-resident fallback uses its borrowed context."""
    if backend is not None:
        _resident_trt_pipeline_invalidate(backend)

def _resident_trt_pipeline_acquire(
    backend: object,
    source: object,
    *,
    input_dtype: object,
    input_channels: int,
    out_size: int,
    native_h: int,
    native_w: int,
    M_out_to_native: np.ndarray,
    track_conf: bool,
    confidence_threshold: float,
    dynamic_unit_descriptors: bool,
) -> Tuple['_ResidentTensorRTRingExecutor', bool]:
    """Acquire the actual executor, not merely its two render slots."""
    persist = bool(resident_trt_pipeline_persistence_enabled())
    if not persist:
        _resident_trt_pipeline_decline(backend)
    signature = _resident_trt_pipeline_signature(
        backend, input_dtype,
        input_channels=int(input_channels), out_size=int(out_size),
        native_h=int(native_h), native_w=int(native_w),
        M_out_to_native=M_out_to_native, track_conf=bool(track_conf),
        confidence_threshold=float(confidence_threshold),
        dynamic_unit_descriptors=bool(dynamic_unit_descriptors),
    )
    stale: Optional[Dict[str, object]] = None
    with _RESIDENT_TRT_PIPELINE_CACHE_LOCK:
        entry = _RESIDENT_TRT_PIPELINE_CACHE.get(id(backend))
        if persist and entry is not None and entry.get('signature') == signature:
            if bool(entry.get('in_use', False)):
                raise RuntimeError('resident TensorRT executor cache re-entered concurrently')
            executor = entry['executor']
            entry['in_use'] = True
            entry['last_used'] = time.monotonic()
        else:
            if entry is not None:
                stale = _RESIDENT_TRT_PIPELINE_CACHE.pop(id(backend), None)
            executor = None
    if stale is not None:
        close = getattr(stale.get('executor'), 'close', None)
        if callable(close):
            close()

    if executor is not None:
        try:
            executor.reset_for_task()
            executor.reconfigure_destination(
                native_h=int(native_h),
                native_w=int(native_w),
                M_out_to_native=np.asarray(M_out_to_native, dtype=np.float32),
                dynamic_unit_descriptors=True,
            )
            source._direct_ring = executor.slots
            slots = source.prepare_direct_ring(input_dtype=input_dtype)
            if len(slots) != 2 or any(a is not b for a, b in zip(slots, executor.slots)):
                raise RuntimeError('cached TensorRT executor did not retain its static ring slots')
            return executor, True
        except Exception as exc:
            source._direct_ring = None
            _resident_trt_pipeline_invalidate(backend, exc)
            raise

    slots = source.prepare_direct_ring(input_dtype=input_dtype)
    if slots is None or len(slots) != 2:
        raise RuntimeError('resident source did not provide two static ring slots')
    executor = _ResidentTensorRTRingExecutor(
        backend, slots, input_channels=int(input_channels), out_size=int(out_size),
        native_h=int(native_h), native_w=int(native_w),
        M_out_to_native=np.asarray(M_out_to_native, dtype=np.float32),
        track_conf=bool(track_conf), confidence_threshold=float(confidence_threshold),
        # Persistent executors use runtime post descriptors so destination geometry
        # can change without invalidating TensorRT execution contexts.
        dynamic_unit_descriptors=True,
    )
    if persist:
        with _RESIDENT_TRT_PIPELINE_CACHE_LOCK:
            _RESIDENT_TRT_PIPELINE_CACHE[id(backend)] = {
                'backend': backend, 'signature': signature, 'executor': executor,
                'in_use': True, 'last_used': time.monotonic(),
            }
    else:
        executor._resident_trt_transient = True
    return executor, False

def _resident_trt_pipeline_release(
    backend: object,
    executor: '_ResidentTensorRTRingExecutor',
    source: Optional[object] = None,
) -> None:
    try:
        executor.reset_for_task()
    except Exception as exc:
        _resident_trt_pipeline_invalidate(backend, exc)
        raise
    with _RESIDENT_TRT_PIPELINE_CACHE_LOCK:
        entry = _RESIDENT_TRT_PIPELINE_CACHE.get(id(backend))
        if entry is None or entry.get('executor') is not executor:
            if bool(getattr(executor, '_resident_trt_transient', False)):
                if source is not None:
                    source._direct_ring = None
                executor.close()
                return
            raise RuntimeError('resident TensorRT executor disappeared before release')
        entry['in_use'] = False
        entry['last_used'] = time.monotonic()
    if source is not None:
        source._direct_ring = None

def _shutdown_resident_trt_pipeline_cache() -> None:
    with _RESIDENT_TRT_PIPELINE_CACHE_LOCK:
        entries = list(_RESIDENT_TRT_PIPELINE_CACHE.values())
        _RESIDENT_TRT_PIPELINE_CACHE.clear()
    for entry in entries:
        try:
            close = getattr(entry.get('executor'), 'close', None)
            if callable(close):
                close()
        except Exception as exc:
            print(f'Warning: resident TensorRT executor shutdown failed ({exc}).')

def _try_resident_trt_ring_accumulate(
    predictor: object,
    source: object,
    cfg: 'PredictConfig',
    *,
    num_frames: int,
    out_size: int,
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    device_union: Optional['_DeviceUnionAccumulator'],
    preunion_min_conf: Optional[float] = None,
) -> Optional[Dict[str, int]]:
    """Run a batch-1 resident source through the persistent two-context TensorRT ring.

    Every v16.4.0 source owns one task-wide post affine. Full-frame and tile tasks
    therefore share the same static destination geometry across all ring slots.
    """
    global _RESIDENT_TRT_RING_ANNOUNCED, _RESIDENT_TRT_RING_FALLBACK_WARNED
    global _RESIDENT_TRT_RING_CACHE_HIT_ANNOUNCED
    backend = getattr(predictor, 'model', None)

    def _decline() -> Optional[Dict[str, int]]:
        _resident_trt_pipeline_decline(backend)
        return None

    if not resident_trt_ring_enabled() or device_union is None:
        return _decline()
    if not isinstance(source, (GpuRenderedYoloSource, GpuTileRenderedYoloSource)):
        return _decline()
    source_channels = int(getattr(source, 'channel_count', 1))
    if source_channels != int(cfg.input_channels):
        try:
            source.reset_direct_ring()
        finally:
            _resident_trt_pipeline_decline(backend)
        raise ModelInputChannelMismatchError(
            f'Resident TensorRT source has C={source_channels}, but '
            f'--channel_format {cfg.channel_token} requires C={int(cfg.input_channels)}'
        )

    matrix = np.asarray(M_out_to_native, dtype=np.float32).reshape(2, 3)
    if not bool(np.isfinite(matrix).all()):
        return _decline()
    identity = bool(
        int(native_h) == int(out_size)
        and int(native_w) == int(out_size)
        and _warp_matrix_is_identity(matrix)
    )
    descriptors = [
        ResidentRingUnitDescriptor(
            unit_index=int(unit_index),
            destination_index=int(unit_index),
            native_h=int(native_h),
            native_w=int(native_w),
            M_out_to_native=matrix,
        )
        for unit_index in range(int(num_frames))
    ]
    dynamic_descriptors = False

    try:
        union_shape = tuple(int(x) for x in device_union.union_dev.shape)
        expected_shape = (int(num_frames), int(native_h), int(native_w))
        if union_shape != expected_shape:
            return _decline()
        if (
            device_union.conf_dev is not None
            and tuple(int(x) for x in device_union.conf_dev.shape) != expected_shape
        ):
            return _decline()
        if str(device_union.union_dev.device) != str(source.engine.device):
            return _decline()
    except Exception:
        return _decline()

    if (
        int(cfg.batch) != 1
        or int(getattr(source, 'bs', 0)) != 1
        or int(getattr(source, 'nf', -1)) != int(num_frames)
        or str(getattr(getattr(source, 'engine', None), '_mode', '')) != 'resident'
        or (not identity and not resident_trt_native_warp_enabled())
    ):
        return _decline()
    trt_engine = None if backend is None else _trt_engine_from_autobackend(backend)
    if backend is None or trt_engine is None or bool(getattr(backend, 'dynamic', False)):
        return _decline()
    prepare = getattr(source, 'prepare_direct_ring', None)
    next_slot = getattr(source, 'next_direct_slot', None)
    if not callable(prepare) or not callable(next_slot):
        return _decline()

    executor: Optional[_ResidentTensorRTRingExecutor] = None
    cache_hit = False
    try:
        import torch  # type: ignore
        _names, input_name, _outputs, _indices = _trt_binding_layout_for_backend(backend, trt_engine)
        input_dtype = _torch_dtype_for_trt_binding(backend, trt_engine, input_name, torch)
        if input_dtype not in (torch.float16, torch.float32):
            raise RuntimeError(
                f'resident ring cannot allocate unsupported TensorRT input dtype {input_dtype}'
            )
        binding_shape = tuple(int(value) for value in (
            trt_engine.get_tensor_shape(str(input_name))
            if callable(getattr(trt_engine, 'get_tensor_shape', None))
            else trt_engine.get_binding_shape(int(_indices[input_name]))
        ))
        expected_binding_shape = (1, int(source_channels), int(out_size), int(out_size))
        if len(binding_shape) != 4 or binding_shape != expected_binding_shape:
            raise RuntimeError(
                f'resident TensorRT ring requires fixed input {expected_binding_shape}, '
                f'but {input_name!r} is {binding_shape}'
            )
        threshold = (
            float(cfg.conf)
            if preunion_min_conf is None
            else max(float(cfg.conf), float(preunion_min_conf))
        )
        executor, cache_hit = _resident_trt_pipeline_acquire(
            backend,
            source,
            input_dtype=input_dtype,
            input_channels=int(source_channels),
            out_size=int(out_size),
            native_h=int(native_h),
            native_w=int(native_w),
            M_out_to_native=matrix,
            track_conf=device_union.conf_dev is not None,
            confidence_threshold=float(threshold),
            dynamic_unit_descriptors=bool(dynamic_descriptors),
        )
    except _ResidentTensorRTRingFatalError:
        raise
    except ModelInputChannelMismatchError:
        try:
            source.reset_direct_ring()
        finally:
            _resident_trt_pipeline_decline(backend)
        raise
    except Exception as exc:
        try:
            source.reset_direct_ring()
        except Exception:
            pass
        _resident_trt_pipeline_decline(backend)
        if not _RESIDENT_TRT_RING_FALLBACK_WARNED:
            _RESIDENT_TRT_RING_FALLBACK_WARNED = True
            print(
                f'Resident TensorRT ring unavailable ({exc}); using the direct predict loop. '
                'YOLO_TTA_RESIDENT_TRT_RING=0 suppresses this capability probe.'
            )
        return None

    pending: 'deque[Tuple[int, _ResidentGpuPipelineSlot]]' = deque()
    try:
        frame_counts_dev = torch.zeros(
            (int(num_frames),), dtype=torch.int32, device=device_union.device,
        )
        for _ in range(min(2, int(num_frames))):
            item = next_slot()
            if item is None:
                break
            unit_index, slot = item
            executor.enqueue_inference(slot)
            pending.append((int(unit_index), slot))
        submitted = len(pending)
        while pending:
            unit_index, slot = pending.popleft()
            executor.enqueue_postprocess(
                slot,
                descriptor=descriptors[int(unit_index)],
                device_union=device_union,
                frame_counts_dev=frame_counts_dev,
            )
            if submitted < int(num_frames):
                item = next_slot()
                if item is None:
                    raise RuntimeError(
                        f'resident TensorRT ring source ended at {submitted}/{int(num_frames)} units'
                    )
                next_index, next_pipeline_slot = item
                executor.enqueue_inference(next_pipeline_slot)
                pending.append((int(next_index), next_pipeline_slot))
                submitted += 1
        executor.synchronize()
        prediction_count = int(frame_counts_dev.sum(dtype=torch.int64).item())
        frames_with_predictions = int((frame_counts_dev > 0).sum(dtype=torch.int64).item())
    except Exception as exc:
        source._direct_ring = None
        _resident_trt_pipeline_invalidate(backend, exc)
        raise
    else:
        _resident_trt_pipeline_release(backend, executor, source)

    if cache_hit and not _RESIDENT_TRT_RING_CACHE_HIT_ANNOUNCED:
        _RESIDENT_TRT_RING_CACHE_HIT_ANNOUNCED = True
        print(
            'Resident TensorRT executor persistence active: contexts, bindings, output '
            'buffers, streams, and compatible CUDA graphs are reused across worker tasks.'
        )
    if not _RESIDENT_TRT_RING_ANNOUNCED:
        _RESIDENT_TRT_RING_ANNOUNCED = True
        post_mode = (
            'identity' if executor.identity_native_warp else 'fused-native-affine'
        )
        print(
            'Resident TensorRT batch-1 ring active: two independent execution contexts, '
            f'two static C={int(source_channels)} input/output slots, low-priority render, '
            'high-priority inference, separate mask/union streams, '
            f'post-warp={post_mode}; proto-hole={executor.proto_hole_treatment}'
            f'(radius={executor.proto_hole_radius if executor.proto_hole_treatment_active else 0}); '
            f'CUDA graphs captured infer={executor.infer_graph_count}/2 '
            f'post={executor.post_graph_count}/2.'
        )
    return {
        'prediction_count': int(prediction_count),
        'frames_with_predictions': int(frames_with_predictions),
        'proto_hole_treated_frames': (
            int(num_frames) if bool(executor.proto_hole_treatment_active) else 0
        ),
    }

def _stage_radial_t_major_block(
    radial_mask_mm: np.ndarray,
    v0: int,
    v1: int,
    out_t_major: np.ndarray,
    *,
    slice_bboxes: Optional[np.ndarray] = None,
    stage_pool: Optional[ThreadPoolExecutor] = None,
    stage_workers: int = 1,
) -> np.ndarray:
    """Stage one contiguous Radial row interval as ``(t, azimuth, u)``.

    Dense sources are transposed one bounded block at a time. Sparse bridge deltas use
    interpolation-carried bboxes and copy only intersecting crop rows. The returned view
    contains exactly ``max(0, v1 - v0)`` clipped rows.
    """
    n_az, work_t, u_len = (int(v) for v in radial_mask_mm.shape)
    lo = int(np.clip(int(v0), 0, work_t))
    hi = int(np.clip(int(v1), lo, work_t))
    count = int(hi - lo)
    dest = np.asarray(out_t_major)[:count, :n_az, :u_len]
    if count <= 0:
        return dest
    if slice_bboxes is None:
        np.copyto(
            dest,
            np.transpose(np.asarray(radial_mask_mm[:, lo:hi, :], dtype=np.uint8), (1, 0, 2)),
        )
        return dest

    # A bridge delta is sparse by azimuth plane. Clearing one bounded t-major staging block
    # is much cheaper than reading all of the source volume's implicit zero regions.
    dest.fill(np.uint8(0))
    workers_i = max(1, min(int(stage_workers), int(n_az)))

    def _stage_az_range(worker_idx: int) -> None:
        a0 = int(worker_idx) * int(math.ceil(float(n_az) / float(workers_i)))
        a1 = min(int(n_az), a0 + int(math.ceil(float(n_az) / float(workers_i))))
        for az in range(a0, a1):
            by0, by1, bx0, bx1 = (int(v) for v in slice_bboxes[int(az)])
            sy0 = max(int(lo), int(by0))
            sy1 = min(int(hi), int(by1))
            if sy0 >= sy1 or bx0 >= bx1:
                continue
            dest[sy0 - lo:sy1 - lo, int(az), bx0:bx1] = np.asarray(
                radial_mask_mm[int(az), sy0:sy1, bx0:bx1], dtype=np.uint8,
            )

    if stage_pool is not None and workers_i > 1:
        list(stage_pool.map(_stage_az_range, range(workers_i)))
    else:
        for worker_idx in range(workers_i):
            _stage_az_range(int(worker_idx))
    return dest

def _abort_projection_block_callback(
    callback: Optional[Callable[[int, np.ndarray], None]],
    reason: BaseException,
) -> None:
    """Invalidate a stateful projection callback without changing projection semantics."""
    if callback is None:
        return
    abort = getattr(callback, 'abort', None)
    if callable(abort):
        try:
            abort(reason)
        except Exception:
            pass

@runtime_telemetry_phase('projection.callback')
def _emit_projection_block_callback(
    callback: Optional[Callable[[int, np.ndarray], None]],
    z0: int,
    block: np.ndarray,
    *,
    desc: str,
    required: bool = False,
) -> None:
    """Deliver one completed projection block to an incremental sink.
    
    Dense projections treat sink failure as best effort; sink-only projections require success and abort transactionally on failure."""
    if callback is None:
        if bool(required):
            raise RuntimeError(f'{desc}: sink-only projection requires a block consumer')
        return
    try:
        callback(int(z0), np.asarray(block))
    except Exception as exc:
        _abort_projection_block_callback(callback, exc)
        if bool(required):
            raise
        warn = getattr(callback, 'warn_failed_once', None)
        if callable(warn):
            try:
                warn(f'{desc}: incremental raw-bbox callback failed ({exc})')
            except Exception:
                pass

def _emit_projection_empty_range(
    callback: Optional[Callable[[int, np.ndarray], None]],
    z0: int,
    count: int,
    plane_shape: Tuple[int, int],
    *,
    desc: str,
    required: bool = False,
) -> None:
    """Deliver a proven-empty range without constructing dense planes when supported."""
    count_i = max(0, int(count))
    if count_i <= 0:
        return
    if callback is None:
        if bool(required):
            raise RuntimeError(f'{desc}: sink-only projection requires an empty-range consumer')
        return
    empty_consumer = getattr(callback, 'consume_empty_range', None)
    if callable(empty_consumer):
        try:
            empty_consumer(int(z0), int(count_i))
            return
        except Exception as exc:
            _abort_projection_block_callback(callback, exc)
            if bool(required):
                raise
            warn = getattr(callback, 'warn_failed_once', None)
            if callable(warn):
                try:
                    warn(f'{desc}: incremental empty-range callback failed ({exc})')
                except Exception:
                    pass
            return
    # Compatibility callbacks that predate consume_empty_range still receive a normal
    # block. This allocation is bounded by one projection chunk, never the full raster.
    _emit_projection_block_callback(
        callback,
        int(z0),
        np.zeros((int(count_i), int(plane_shape[0]), int(plane_shape[1])), dtype=np.uint8),
        desc=desc,
        required=bool(required),
    )

@dataclass(frozen=True)
class SinkOnlyProjectionResult:
    """Successful projection whose complete representation lives in its block sink."""

    shape: Tuple[int, int, int]

_RADIAL_RESIDENT_BACKPROJECT_KERNEL: Optional[object] = None

_RADIAL_RESIDENT_BACKPROJECT_KERNEL_FAILED = False

_RADIAL_RESIDENT_BACKPROJECT_KERNEL_ERROR: Optional[str] = None

def gpu_resident_radial_backproject_enabled() -> bool:
    return _env_flag('YOLO_TTA_GPU_BACKPROJECT_RESIDENT', True)

def fused_angle_variant_radial_component_layer_enabled() -> bool:
    """Project one post-interpolation Radial union in the angle-variant fast path."""
    return _env_flag('YOLO_TTA_FUSED_ANGLE_VARIANT_RADIAL_LAYER', True)

def _radial_resident_backproject_kernel() -> Optional[object]:
    global _RADIAL_RESIDENT_BACKPROJECT_KERNEL
    global _RADIAL_RESIDENT_BACKPROJECT_KERNEL_FAILED, _RADIAL_RESIDENT_BACKPROJECT_KERNEL_ERROR
    if _RADIAL_RESIDENT_BACKPROJECT_KERNEL is not None:
        return _RADIAL_RESIDENT_BACKPROJECT_KERNEL
    if _RADIAL_RESIDENT_BACKPROJECT_KERNEL_FAILED:
        return None
    try:
        import cupy as cp  # type: ignore
        code = r'''
        extern "C" __global__ void radial_backproject_azmajor_u8(
            const unsigned char* radial, int n_az, int work_t, int u_len,
            const int* source_idx, const int* u_idx, int plane_px,
            int output_t, int output_t0, int output_count, unsigned char* output) {
          long long q0 = (long long)blockDim.x * (long long)blockIdx.x + (long long)threadIdx.x;
          long long stride = (long long)blockDim.x * (long long)gridDim.x;
          long long total = (long long)output_count * (long long)plane_px;
          for (long long q = q0; q < total; q += stride) {
            int local_t = (int)(q / (long long)plane_px);
            int p = (int)(q - (long long)local_t * (long long)plane_px);
            int az = source_idx[p];
            int u = u_idx[p];
            if (az < 0 || az >= n_az || u < 0 || u >= u_len) {
              output[q] = 0;
              continue;
            }
            int t = output_t0 + local_t;
            int v0 = (int)(((long long)t * (long long)work_t) / (long long)output_t);
            int v1 = (int)((((long long)(t + 1) * (long long)work_t) + output_t - 1) / (long long)output_t);
            if (v0 < 0) v0 = 0;
            if (v1 > work_t) v1 = work_t;
            if (v1 <= v0) v1 = v0 + 1 <= work_t ? v0 + 1 : work_t;
            unsigned char value = 0;
            long long az_base = (long long)az * (long long)work_t * (long long)u_len;
            for (int v = v0; v < v1; ++v) {
              unsigned char candidate = radial[az_base + (long long)v * (long long)u_len + (long long)u];
              value = candidate > value ? candidate : value;
            }
            output[q] = value;
          }
        }

        extern "C" __global__ void radial_backproject_tmajor_u8(
            const unsigned char* radial, int n_az, int work_t, int u_len,
            const int* source_idx, const int* u_idx, int plane_px,
            int output_t, int output_t0, int output_count, unsigned char* output) {
          long long q0 = (long long)blockDim.x * (long long)blockIdx.x + (long long)threadIdx.x;
          long long stride = (long long)blockDim.x * (long long)gridDim.x;
          long long total = (long long)output_count * (long long)plane_px;
          for (long long q = q0; q < total; q += stride) {
            int local_t = (int)(q / (long long)plane_px);
            int p = (int)(q - (long long)local_t * (long long)plane_px);
            int az = source_idx[p];
            int u = u_idx[p];
            if (az < 0 || az >= n_az || u < 0 || u >= u_len) {
              output[q] = 0;
              continue;
            }
            int t = output_t0 + local_t;
            int v0 = (int)(((long long)t * (long long)work_t) / (long long)output_t);
            int v1 = (int)((((long long)(t + 1) * (long long)work_t) + output_t - 1) / (long long)output_t);
            if (v0 < 0) v0 = 0;
            if (v1 > work_t) v1 = work_t;
            if (v1 <= v0) v1 = v0 + 1 <= work_t ? v0 + 1 : work_t;
            unsigned char value = 0;
            for (int v = v0; v < v1; ++v) {
              long long index = ((long long)v * (long long)n_az + (long long)az)
                              * (long long)u_len + (long long)u;
              unsigned char candidate = radial[index];
              value = candidate > value ? candidate : value;
            }
            output[q] = value;
          }
        }

        extern "C" __global__ void radial_azmajor_to_tmajor_u8(
            const unsigned char* source, int n_az, int work_t, int u_len,
            unsigned char* target) {
          long long q0 = (long long)blockDim.x * (long long)blockIdx.x + (long long)threadIdx.x;
          long long stride = (long long)blockDim.x * (long long)gridDim.x;
          long long total = (long long)n_az * (long long)work_t * (long long)u_len;
          for (long long q = q0; q < total; q += stride) {
            int u = (int)(q % (long long)u_len);
            long long av = q / (long long)u_len;
            int v = (int)(av % (long long)work_t);
            int az = (int)(av / (long long)work_t);
            target[((long long)v * (long long)n_az + (long long)az) * (long long)u_len + (long long)u]
                = source[q];
          }
        }
        '''
        module = cp.RawModule(code=code, options=('--std=c++14',))
        compile_fn = getattr(module, 'compile', None)
        if callable(compile_fn):
            compile_fn()
        kernel_azmajor = module.get_function('radial_backproject_azmajor_u8')
        _RADIAL_RESIDENT_BACKPROJECT_KERNEL = argparse.Namespace(
            cp=cp,
            module=module,
            kernel=kernel_azmajor,  # compatibility alias
            kernel_azmajor=kernel_azmajor,
            kernel_tmajor=module.get_function('radial_backproject_tmajor_u8'),
            transpose=module.get_function('radial_azmajor_to_tmajor_u8'),
        )
        return _RADIAL_RESIDENT_BACKPROJECT_KERNEL
    except Exception as exc:
        _RADIAL_RESIDENT_BACKPROJECT_KERNEL_FAILED = True
        _RADIAL_RESIDENT_BACKPROJECT_KERNEL_ERROR = f'{type(exc).__name__}: {exc}'
        print(
            'Warning: resident Radial backprojection kernel unavailable '
            f'({_RADIAL_RESIDENT_BACKPROJECT_KERNEL_ERROR}); using the streaming GPU path.'
        )
        return None

def _radial_backproject_gpu_resident(
    radial_mask_mm: np.ndarray,
    vol_mm: Optional[np.ndarray],
    valid_mask: np.ndarray,
    source_idx_map: np.ndarray,
    u_idx_map: np.ndarray,
    v_range_for_t: Callable[[int], Tuple[int, int]],
    t_dim: int,
    out_h: int,
    out_w: int,
    desc: str,
    known_row_occupancy: Optional[np.ndarray] = None,
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    sink_only: bool = False,
) -> bool:
    """Run full-resident Radial backprojection under an inference-exclusive GPU lease."""
    if not gpu_resident_radial_backproject_enabled():
        return False
    try:
        import torch  # type: ignore
        if not bool(torch.cuda.is_available()):
            return False
    except Exception:
        return False
    lease = _try_acquire_main_process_gpu_stage(
        torch, f'{desc} full-resident Radial backprojection',
    )
    if lease is None:
        _announce_main_gpu_stage_skip_once(
            'radial-resident-inference-busy',
            f'{desc}: full-resident GPU backprojection skipped because every eligible GPU '
            'has active/queued inference or another main-process GPU stage; trying the bounded fallback.',
        )
        return False
    try:
        dev = lease.torch_device(torch)
        # NVRTC/module initialization is itself a CUDA allocation. Keep it inside the
        # same exclusive interval as the large source upload.
        with torch.cuda.device(dev):
            kernels = _radial_resident_backproject_kernel()
        if kernels is None:
            return False
        return _radial_backproject_gpu_resident_on_device(
            radial_mask_mm,
            vol_mm,
            valid_mask,
            source_idx_map,
            u_idx_map,
            v_range_for_t,
            int(t_dim),
            int(out_h),
            int(out_w),
            str(desc),
            known_row_occupancy=known_row_occupancy,
            projection_block_callback=projection_block_callback,
            sink_only=bool(sink_only),
            kernels=kernels,
            torch=torch,
            dev=dev,
        )
    finally:
        lease.release()

def _radial_backproject_gpu_resident_on_device(
    radial_mask_mm: np.ndarray,
    vol_mm: Optional[np.ndarray],
    valid_mask: np.ndarray,
    source_idx_map: np.ndarray,
    u_idx_map: np.ndarray,
    v_range_for_t: Callable[[int], Tuple[int, int]],
    t_dim: int,
    out_h: int,
    out_w: int,
    desc: str,
    known_row_occupancy: Optional[np.ndarray] = None,
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    sink_only: bool = False,
    *,
    kernels: object,
    torch: object,
    dev: object,
) -> bool:
    """Backproject a fully resident Radial volume with overlapped upload and host commit."""
    n_az, work_t, u_len = (int(v) for v in radial_mask_mm.shape)
    plane_px = int(out_h) * int(out_w)
    t_chunk = max(1, _env_int('YOLO_TTA_GPU_BACKPROJECT_RESIDENT_T_CHUNK', 64))
    requested_pipeline_slots = max(1, min(3, _env_int(
        'YOLO_TTA_GPU_BACKPROJECT_RESIDENT_PIPELINE_SLOTS', 2,
    )))
    reserve = int(max(0.0, _env_float('YOLO_TTA_GPU_BACKPROJECT_RESIDENT_RESERVE_GIB', 4.0)) * GIB)
    source_bytes = int(radial_mask_mm.nbytes)
    map_bytes = int(plane_px) * 2 * np.dtype(np.int32).itemsize
    output_bytes = int(t_chunk) * int(plane_px)
    base_need = int(
        source_bytes + map_bytes + int(requested_pipeline_slots) * output_bytes
        + 256 * 1024 * 1024
    )
    try:
        free_bytes, _total = torch.cuda.mem_get_info(dev)
    except Exception:
        return False
    while t_chunk > 8 and int(free_bytes) < int(base_need) + int(reserve):
        t_chunk //= 2
        output_bytes = int(t_chunk) * int(plane_px)
        base_need = int(
            source_bytes + map_bytes + int(requested_pipeline_slots) * output_bytes
            + 256 * 1024 * 1024
        )
    if int(free_bytes) < int(base_need) + int(reserve):
        print(
            f'{desc}: full-resident GPU backprojection skipped (need≈{base_need / GIB:.1f} GiB '
            f'+ {reserve / GIB:.1f} GiB reserve, free={int(free_bytes) / GIB:.1f} GiB).'
        )
        return False

    if known_row_occupancy is not None and int(np.asarray(known_row_occupancy).shape[0]) == int(work_t):
        row_any = np.asarray(known_row_occupancy, dtype=bool)
    else:
        row_any = _radial_row_occupancy(radial_mask_mm, desc)
    source_flat = np.where(
        np.asarray(valid_mask, dtype=bool), np.asarray(source_idx_map, dtype=np.int32), np.int32(-1),
    ).reshape(-1)
    u_flat = np.where(
        np.asarray(valid_mask, dtype=bool), np.asarray(u_idx_map, dtype=np.int32), np.int32(0),
    ).reshape(-1)
    source_flat = np.ascontiguousarray(source_flat, dtype=np.int32)
    u_flat = np.ascontiguousarray(u_flat, dtype=np.int32)

    upload_target = max(64 * 1024 * 1024, _env_int(
        'YOLO_TTA_GPU_BACKPROJECT_RESIDENT_UPLOAD_MIB', 512,
    ) * 1024 * 1024)
    bytes_per_azimuth = max(1, int(work_t) * int(u_len))
    az_chunk = max(1, min(int(n_az), int(upload_target) // int(bytes_per_azimuth)))
    cp = kernels.cp
    dev_index = getattr(dev, 'index', None)
    if dev_index is None:
        try:
            dev_index = int(torch.cuda.current_device())
        except Exception:
            dev_index = 0
    compute_stream = None
    copy_stream = None
    phase_start = time.perf_counter()
    upload_seconds = 0.0
    transpose_seconds = 0.0
    projection_seconds = 0.0
    commit_cpu_seconds = 0.0
    radial_dev = tmajor_dev = source_dev = u_dev = None
    compute_tensor = compute_kernel = external = None
    cp_source = cp_u = cp_azmajor = cp_tmajor = cp_radial = None
    compute_done = copy_done = event = prior = block = None
    out_dev_slots: List[object] = []
    pin_src_slots: List[object] = []
    pin_out_slots: List[object] = []
    cp_out_slots: List[object] = []
    pin_src_arrays: List[np.ndarray] = []
    pin_out_arrays: List[np.ndarray] = []
    upload_events: List[Optional[object]] = []
    commit_pool: Optional[ThreadPoolExecutor] = None
    commit_futures: List[Optional[Future]] = []
    layout = 'azimuth-major'
    try:
        compute_stream = torch.cuda.Stream(device=dev)
        copy_stream = torch.cuda.Stream(device=dev)
        # Torch and CuPy keep independent current-device state. Pin both explicitly so
        # the late-tail scheduler may choose any visible GPU, not only cuda:0.
        with torch.cuda.device(dev), cp.cuda.Device(int(dev_index)), torch.cuda.stream(compute_stream):
            radial_dev = torch.empty((n_az, work_t, u_len), dtype=torch.uint8, device=dev)

            requested_upload_slots = max(1, min(3, _env_int(
                'YOLO_TTA_GPU_BACKPROJECT_RESIDENT_UPLOAD_SLOTS', 2,
            )))
            for _ in range(int(requested_upload_slots)):
                try:
                    pin_src_slots.append(torch.empty(
                        (az_chunk, work_t, u_len), dtype=torch.uint8, pin_memory=True,
                    ))
                except Exception:
                    break
            if not pin_src_slots:
                pin_src_slots.append(torch.empty(
                    (az_chunk, work_t, u_len), dtype=torch.uint8, pin_memory=True,
                ))
            pin_src_arrays = [slot.numpy() for slot in pin_src_slots]
            upload_events = [None] * len(pin_src_slots)
            upload_t0 = time.perf_counter()
            for chunk_idx, a0 in enumerate(tqdm(
                range(0, n_az, az_chunk), desc=f'{desc} [resident H2D]',
            )):
                slot_idx = int(chunk_idx) % len(pin_src_slots)
                prior = upload_events[slot_idx]
                if prior is not None:
                    prior.synchronize()
                a1 = min(n_az, a0 + az_chunk)
                count = int(a1 - a0)
                np.copyto(
                    pin_src_arrays[slot_idx][:count],
                    np.asarray(radial_mask_mm[a0:a1], dtype=np.uint8),
                )
                radial_dev[a0:a1].copy_(pin_src_slots[slot_idx][:count], non_blocking=True)
                event = torch.cuda.Event(blocking=False)
                event.record(compute_stream)
                upload_events[slot_idx] = event
            compute_stream.synchronize()
            upload_seconds = time.perf_counter() - upload_t0

            source_dev = torch.from_numpy(source_flat).to(dev, non_blocking=False)
            u_dev = torch.from_numpy(u_flat).to(dev, non_blocking=False)
            external = _cupy_external_stream(cp, compute_stream)
            cp_source = cp.asarray(source_dev)
            cp_u = cp.asarray(u_dev)

            layout = 'azimuth-major'
            compute_kernel = kernels.kernel_azmajor
            compute_tensor = radial_dev
            tmajor_peak = int(
                2 * source_bytes + map_bytes
                + int(requested_pipeline_slots) * output_bytes
                + 256 * 1024 * 1024
            )
            if (
                _env_flag('YOLO_TTA_GPU_BACKPROJECT_RESIDENT_TMAJOR', True)
                and int(free_bytes) >= int(tmajor_peak) + int(reserve)
            ):
                transpose_t0 = time.perf_counter()
                try:
                    tmajor_dev = torch.empty((work_t, n_az, u_len), dtype=torch.uint8, device=dev)
                    cp_azmajor = cp.asarray(radial_dev)
                    cp_tmajor = cp.asarray(tmajor_dev)
                    total_source = int(n_az) * int(work_t) * int(u_len)
                    transpose_blocks = max(1, min(
                        (int(total_source) + 255) // 256,
                        _env_int('YOLO_TTA_GPU_BACKPROJECT_RESIDENT_CUDA_BLOCKS', 65535),
                    ))
                    kernels.transpose(
                        (int(transpose_blocks),), (256,),
                        (
                            cp_azmajor, np.int32(n_az), np.int32(work_t), np.int32(u_len),
                            cp_tmajor,
                        ),
                        stream=external,
                    )
                    compute_stream.synchronize()
                    compute_tensor = tmajor_dev
                    compute_kernel = kernels.kernel_tmajor
                    layout = 't-major'
                except Exception as exc:
                    tmajor_dev = None
                    compute_tensor = radial_dev
                    compute_kernel = kernels.kernel_azmajor
                    print(f'{desc}: t-major resident transpose unavailable ({exc}); using azimuth-major layout.')
                transpose_seconds = time.perf_counter() - transpose_t0

            cp_radial = cp.asarray(compute_tensor)
            # allocate output device and pinned-host slots as pairs. A slot is not
            # reused until its D2H event and CPU sink commit have both completed; another
            # slot can therefore run the next projection while copy/packing proceeds.
            for _ in range(int(requested_pipeline_slots)):
                try:
                    out_dev_slots.append(torch.empty(
                        (t_chunk, plane_px), dtype=torch.uint8, device=dev,
                    ))
                    pin_out_slots.append(torch.empty(
                        (t_chunk, plane_px), dtype=torch.uint8, pin_memory=True,
                    ))
                except Exception:
                    if len(out_dev_slots) > len(pin_out_slots):
                        out_dev_slots.pop()
                    break
            if not out_dev_slots:
                out_dev_slots.append(torch.empty(
                    (t_chunk, plane_px), dtype=torch.uint8, device=dev,
                ))
                pin_out_slots.append(torch.empty(
                    (t_chunk, plane_px), dtype=torch.uint8, pin_memory=True,
                ))
            cp_out_slots = [cp.asarray(slot) for slot in out_dev_slots]
            pin_out_arrays = [slot.numpy() for slot in pin_out_slots]
            commit_futures = [None] * len(pin_out_slots)
            commit_pool = _acquire_parallel_pool(len(pin_out_slots))

            def _commit_block(t0_i: int, t1_i: int, block_i: np.ndarray) -> float:
                started = time.perf_counter()
                if vol_mm is not None:
                    np.copyto(np.asarray(vol_mm[int(t0_i):int(t1_i)]), block_i)
                _emit_projection_block_callback(
                    projection_block_callback,
                    int(t0_i),
                    block_i,
                    desc=desc,
                    required=bool(sink_only),
                )
                return float(time.perf_counter() - started)

            def _settle_commit(slot_idx: int) -> None:
                nonlocal commit_cpu_seconds
                fut = commit_futures[int(slot_idx)]
                if fut is not None:
                    commit_cpu_seconds += float(fut.result())
                    commit_futures[int(slot_idx)] = None

            def _commit_after_copy(
                copy_done: object,
                t0_i: int,
                t1_i: int,
                block_i: np.ndarray,
            ) -> float:
                copy_done.synchronize()
                return _commit_block(int(t0_i), int(t1_i), block_i)

            projection_t0 = time.perf_counter()
            scheduled_idx = 0
            for _chunk_idx, t0 in enumerate(tqdm(
                range(0, int(t_dim), int(t_chunk)), desc=f'{desc} [resident gpu]',
            )):
                t1 = min(int(t_dim), int(t0) + int(t_chunk))
                v0 = v_range_for_t(int(t0))[0]
                v1 = v_range_for_t(int(t1 - 1))[1]
                if not bool(np.any(row_any[int(v0):int(v1)])):
                    _emit_projection_empty_range(
                        projection_block_callback,
                        int(t0),
                        int(t1 - t0),
                        (int(out_h), int(out_w)),
                        desc=desc,
                        required=bool(sink_only),
                    )
                    continue
                slot_idx = int(scheduled_idx) % len(pin_out_slots)
                scheduled_idx += 1
                _settle_commit(slot_idx)
                count = int(t1 - t0)
                total = int(count) * int(plane_px)
                launch_blocks = max(1, min(
                    (int(total) + 255) // 256,
                    _env_int('YOLO_TTA_GPU_BACKPROJECT_RESIDENT_CUDA_BLOCKS', 65535),
                ))
                compute_kernel(
                    (int(launch_blocks),), (256,),
                    (
                        cp_radial, np.int32(n_az), np.int32(work_t), np.int32(u_len),
                        cp_source, cp_u, np.int32(plane_px), np.int32(t_dim),
                        np.int32(t0), np.int32(count), cp_out_slots[slot_idx],
                    ),
                    stream=external,
                )
                compute_done = torch.cuda.Event(blocking=False)
                compute_done.record(compute_stream)
                copy_stream.wait_event(compute_done)
                with torch.cuda.stream(copy_stream):
                    pin_out_slots[slot_idx][:count].copy_(
                        out_dev_slots[slot_idx][:count], non_blocking=True,
                    )
                    copy_done = torch.cuda.Event(blocking=False)
                    copy_done.record(copy_stream)
                block = pin_out_arrays[slot_idx][:count].reshape((count, int(out_h), int(out_w)))
                commit_futures[slot_idx] = commit_pool.submit(
                    _commit_after_copy, copy_done, int(t0), int(t1), block,
                )
            for slot_idx in range(len(commit_futures)):
                _settle_commit(slot_idx)
            compute_stream.synchronize()
            copy_stream.synchronize()
            projection_seconds = time.perf_counter() - projection_t0

        total_seconds = time.perf_counter() - phase_start
        print(
            f'{desc}: full-resident GPU backprojection complete; source={source_bytes / GIB:.1f} GiB, '
            f'layout={layout}, upload_slots={len(pin_src_slots)}, pipeline_slots={len(pin_out_slots)}, '
            f'output={"sink-only" if bool(sink_only) else "dense+sink"}, '
            f'active_rows={int(np.count_nonzero(row_any))}/{int(work_t)}, '
            f'upload={upload_seconds:.2f}s, transpose={transpose_seconds:.2f}s, '
            f'project+commit wall={projection_seconds:.2f}s, commit CPU sum={commit_cpu_seconds:.2f}s, '
            f'total={total_seconds:.2f}s.'
        )
        return True
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_error: Optional[BaseException] = None
        for owned_stream in (compute_stream, copy_stream):
            if owned_stream is None:
                continue
            try:
                owned_stream.synchronize()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if commit_pool is not None:
            for fut in commit_futures:
                if fut is None:
                    continue
                try:
                    fut.result()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            try:
                _release_parallel_pool(max(1, len(pin_out_slots)), commit_pool)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        # Drop every Torch owner, CuPy zero-copy alias, event, stream and NumPy view first.
        # Trimming either allocator while one of these references is live leaves the large
        # source/output blocks reserved until a later allocation fails.
        commit_futures.clear()
        upload_events.clear()
        cp_out_slots.clear()
        pin_src_arrays.clear()
        pin_out_arrays.clear()
        out_dev_slots.clear()
        pin_src_slots.clear()
        pin_out_slots.clear()
        cp_radial = cp_azmajor = cp_tmajor = cp_source = cp_u = None
        compute_tensor = compute_kernel = external = None
        compute_done = copy_done = event = prior = block = None
        radial_dev = tmajor_dev = source_dev = u_dev = None
        commit_pool = None
        fut = owned_stream = None
        compute_stream = copy_stream = None
        gc.collect()

        _trim_main_process_cuda_device(
            torch,
            dev,
            cupy_module=cp,
            desc=f'{desc} full-resident backprojection cleanup',
        )
        if cleanup_error is not None and not bool(active_exception):
            raise cleanup_error

def _radial_backproject_gpu_streaming(
    radial_mask_mm: np.ndarray,
    vol_mm: Optional[np.ndarray],
    valid_pos: np.ndarray,
    flat_src: np.ndarray,
    v_range_for_t: Callable[[int], Tuple[int, int]],
    t_dim: int,
    out_h: int,
    out_w: int,
    desc: str,
    known_row_occupancy: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    sink_only: bool = False,
) -> bool:
    """Run bounded Radial backprojection under an inference-exclusive GPU lease."""
    try:
        import torch  # type: ignore
        if not bool(torch.cuda.is_available()):
            return False
    except Exception:
        return False
    lease = _try_acquire_main_process_gpu_stage(
        torch, f'{desc} streaming Radial backprojection',
    )
    if lease is None:
        _announce_main_gpu_stage_skip_once(
            'radial-streaming-inference-busy',
            f'{desc}: streaming GPU backprojection skipped because every eligible GPU '
            'has active/queued inference or another main-process GPU stage; using the CPU path.',
        )
        return False
    try:
        return _radial_backproject_gpu_streaming_on_device(
            radial_mask_mm,
            vol_mm,
            valid_pos,
            flat_src,
            v_range_for_t,
            int(t_dim),
            int(out_h),
            int(out_w),
            str(desc),
            known_row_occupancy=known_row_occupancy,
            known_slice_bboxes=known_slice_bboxes,
            projection_block_callback=projection_block_callback,
            sink_only=bool(sink_only),
            torch=torch,
            dev=lease.torch_device(torch),
        )
    finally:
        lease.release()

def _radial_backproject_gpu_streaming_on_device(
    radial_mask_mm: np.ndarray,
    vol_mm: Optional[np.ndarray],
    valid_pos: np.ndarray,
    flat_src: np.ndarray,
    v_range_for_t: Callable[[int], Tuple[int, int]],
    t_dim: int,
    out_h: int,
    out_w: int,
    desc: str,
    known_row_occupancy: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    sink_only: bool = False,
    *,
    torch: object,
    dev: object,
) -> bool:
    """Backproject a Radial volume through a bounded streaming GPU working set."""
    n_az, work_t, u_len = (int(x) for x in radial_mask_mm.shape)
    plane_px = int(out_h) * int(out_w)
    # A larger projection quantum amortizes map gathers and the t-major staging transpose.
    t_chunk = max(1, _env_int('YOLO_TTA_GPU_BACKPROJECT_T_CHUNK', 128))
    cross_bytes = int(n_az) * int(u_len)
    try:
        free_bytes, _total = torch.cuda.mem_get_info(dev)
    except Exception:
        return False

    def _projection_need(chunk_planes: int) -> Tuple[int, int]:
        rows = int(math.ceil(float(chunk_planes) * float(work_t) / float(max(1, t_dim)))) + 2
        need_bytes = (
            int(valid_pos.nbytes) + int(flat_src.nbytes)
            + rows * cross_bytes
            + rows * int(flat_src.shape[0])
            + int(chunk_planes) * int(flat_src.shape[0])
            + 2 * int(chunk_planes) * plane_px
            + 512 * 1024 * 1024
        )
        return int(rows), int(need_bytes)

    max_rows, need = _projection_need(int(t_chunk))
    while int(t_chunk) > 16 and int(free_bytes) < int(need) + 1 * GIB:
        t_chunk = max(16, int(t_chunk) // 2)
        max_rows, need = _projection_need(int(t_chunk))
    if int(free_bytes) < int(need) + 1 * GIB:
        print(
            f'{desc}: GPU backprojection skipped (needs ~{need / GIB:.1f} GiB + 1 GiB headroom, '
            f'{free_bytes / GIB:.1f} GiB free); using the CPU path.'
        )
        return False

    if known_row_occupancy is not None and int(np.asarray(known_row_occupancy).shape[0]) == int(work_t):
        row_any = np.asarray(known_row_occupancy, dtype=bool)
    else:
        row_any = _radial_row_occupancy(radial_mask_mm, desc)
    slice_bboxes = _validated_radial_slice_bboxes(
        known_slice_bboxes, int(n_az), int(work_t), int(u_len),
    )
    stage_workers = max(1, min(8, _cpu_count()))
    stage_pool = _acquire_parallel_pool(stage_workers)

    stream = None
    valid_pos_t = flat_src_t = None
    pin_rows = pin_out = None
    pin_rows_np = pin_out_np = None
    out_dev = reduced_dev = None
    rows_dev = gathered = None
    pair_row_t = pair_out_t = pair_values = reduced = sub = None
    cleanup_error: Optional[BaseException] = None
    active_exception = False
    try:
        stream = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(stream):
            valid_pos_t = torch.from_numpy(np.ascontiguousarray(valid_pos)).to(dev)
            flat_src_t = torch.from_numpy(np.ascontiguousarray(flat_src)).to(dev)
            pin_rows = torch.empty((max_rows, n_az, u_len), dtype=torch.uint8, pin_memory=True)
            pin_rows_np = pin_rows.numpy()
            pin_out = torch.empty((t_chunk, plane_px), dtype=torch.uint8, pin_memory=True)
            pin_out_np = pin_out.numpy()
            out_dev = torch.empty((t_chunk, plane_px), dtype=torch.uint8, device=dev)
            reduced_dev = torch.empty(
                (t_chunk, int(valid_pos.shape[0])), dtype=torch.uint8, device=dev,
            )
            scatter_reduce_supported = True
            for t0 in tqdm(range(0, int(t_dim), t_chunk), desc=f'{desc} [gpu]'):
                rows_dev = gathered = None
                pair_row_t = pair_out_t = pair_values = reduced = sub = None
                t1 = min(int(t_dim), t0 + t_chunk)
                v0 = v_range_for_t(int(t0))[0]
                v1 = v_range_for_t(int(t1 - 1))[1]
                v_count = int(v1 - v0)
                if not bool(np.any(row_any[int(v0):int(v1)])):
                    _emit_projection_empty_range(
                        projection_block_callback,
                        int(t0),
                        int(t1 - t0),
                        (int(out_h), int(out_w)),
                        desc=desc,
                        required=bool(sink_only),
                    )
                    continue
                if int(v_count) > int(pin_rows.shape[0]):  # pragma: no cover - conservative bound
                    raise RuntimeError(f'{desc}: covering rows {v_count} exceed staging bound {pin_rows.shape[0]}')

                _stage_radial_t_major_block(
                    radial_mask_mm,
                    int(v0),
                    int(v1),
                    pin_rows_np,
                    slice_bboxes=slice_bboxes,
                    stage_pool=stage_pool,
                    stage_workers=stage_workers,
                )
                rows_dev = pin_rows[:v_count].to(dev, non_blocking=True)
                gathered = rows_dev.reshape(v_count, -1)[:, flat_src_t]
                out_dev.zero_()
                t_count = int(t1 - t0)
                if int(work_t) == int(t_dim):
                    out_dev[:t_count].index_copy_(1, valid_pos_t, gathered[:t_count])
                else:
                    pair_out: List[int] = []
                    pair_row: List[int] = []
                    for t in range(int(t0), int(t1)):
                        vs, ve = v_range_for_t(int(t))
                        for v in range(int(vs), int(ve)):
                            pair_out.append(int(t - t0))
                            pair_row.append(int(v - v0))
                    if pair_row and scatter_reduce_supported:
                        try:
                            pair_row_t = torch.as_tensor(pair_row, dtype=torch.int64, device=dev)
                            pair_out_t = torch.as_tensor(pair_out, dtype=torch.int64, device=dev)
                            pair_values = gathered.index_select(0, pair_row_t)
                            reduced = reduced_dev[:t_count]
                            reduced.zero_()
                            reduced.scatter_reduce_(
                                0,
                                pair_out_t[:, None].expand(-1, int(gathered.shape[1])),
                                pair_values,
                                reduce='amax',
                                include_self=True,
                            )
                            out_dev[:t_count].index_copy_(1, valid_pos_t, reduced)
                        except (AttributeError, RuntimeError, TypeError):
                            scatter_reduce_supported = False
                    if pair_row and not scatter_reduce_supported:
                        for local_t in range(t_count):
                            idxs = [i for i, target in enumerate(pair_out) if int(target) == int(local_t)]
                            if not idxs:
                                continue
                            row_ids = [int(pair_row[i]) for i in idxs]
                            sub = gathered[row_ids[0]] if len(row_ids) == 1 else gathered[row_ids].amax(dim=0)
                            out_dev[local_t].index_put_((valid_pos_t,), sub)
                pin_out[: t1 - t0].copy_(out_dev[: t1 - t0], non_blocking=True)
                stream.synchronize()
                if vol_mm is not None:
                    np.copyto(
                        np.asarray(vol_mm[t0:t1]).reshape(t1 - t0, -1),
                        pin_out_np[: t1 - t0],
                    )
                _emit_projection_block_callback(
                    projection_block_callback,
                    int(t0),
                    pin_out_np[: t1 - t0].reshape((int(t1 - t0), int(out_h), int(out_w))),
                    desc=desc,
                    required=bool(sink_only),
                )
            stream.synchronize()
    except BaseException:
        active_exception = True
        raise
    finally:
        try:
            if stream is not None:
                stream.synchronize()
        except BaseException as exc:
            cleanup_error = exc
        _release_parallel_pool(stage_workers, stage_pool)
        # Numpy views keep their pinned Torch owners alive; clear them before the tensors.
        pin_rows_np = pin_out_np = None
        rows_dev = gathered = None
        pair_row_t = pair_out_t = pair_values = reduced = sub = None
        valid_pos_t = flat_src_t = None
        out_dev = reduced_dev = None
        pin_rows = pin_out = None
        stream = None
        _trim_main_process_cuda_device(
            torch,
            dev,
            desc=f'{desc} streaming backprojection cleanup',
        )
        if cleanup_error is not None and not bool(active_exception):
            raise cleanup_error
    print(
        f'{desc}: GPU streaming backprojection complete ({t_dim} output slices; '
        f'output={"sink-only" if bool(sink_only) else "dense+sink"}).'
    )
    return True

def _log_radial_backprojection_densification(
    desc: str,
    plan_stats: Dict[str, float],
) -> None:
    """Report when a coarse Radial source is densified for continuous backprojection."""
    if not bool(plan_stats.get('densified', 0.0)):
        return
    print(
        f"{desc}: densifying radial backprojection for continuity "
        f"from {int(plan_stats['source_frames'])} source frame(s) at "
        f"{float(plan_stats['provided_spacing_deg']):.6g}° spacing to "
        f"{int(plan_stats['backprojection_angles'])} backprojection angle(s) at "
        f"{float(plan_stats['effective_spacing_deg']):.6g}° spacing "
        f"(full-coverage threshold {float(plan_stats['coverage_spacing_deg']):.6g}°)"
    )

def _radial_output_stack_and_plane_shape(
    radial_view: ViewInfo,
    out_shape_tyx: Tuple[int, int, int],
) -> Tuple[int, Tuple[int, int]]:
    out_t, out_h, out_w = (int(v) for v in out_shape_tyx)
    base = radial_base_view_name(radial_view)
    if base == 'transverse':
        return int(out_t), (int(out_h), int(out_w))
    if base == 'sagittal':
        return int(out_h), (int(out_t), int(out_w))
    if base == 'coronal':
        return int(out_w), (int(out_t), int(out_h))
    raise ValueError(f'Unsupported Radial base: {base}')

def _radial_source_rows_for_output(
    source_rows: int,
    output_rows: int,
    output_index: int,
) -> range:
    src = max(1, int(source_rows))
    out = max(1, int(output_rows))
    idx = int(output_index)
    if src >= out:
        start = int(math.floor(float(idx) * float(src) / float(out)))
        stop = int(math.ceil(float(idx + 1) * float(src) / float(out)))
        start = int(np.clip(start, 0, src - 1))
        stop = int(np.clip(max(start + 1, stop), 1, src))
        return range(start, stop)
    coord = (float(idx) + 0.5) * (float(src) / float(out)) - 0.5
    nearest = int(np.clip(int(round(coord)), 0, src - 1))
    return range(nearest, nearest + 1)

def _radial_backproject_plane(
    radial_mask_mm: np.ndarray,
    dense_map: DenseRadialBackprojectionMap,
    source_rows: Sequence[int],
    *,
    plane_row_slice: Optional[slice] = None,
) -> np.ndarray:
    src = np.asarray(radial_mask_mm)
    rows = list(int(v) for v in source_rows)
    row_sel = slice(None) if plane_row_slice is None else plane_row_slice
    valid = np.asarray(dense_map.valid_mask, dtype=bool)[row_sel]
    if not rows:
        return np.zeros(valid.shape, dtype=np.uint8)
    if len(rows) == 1:
        cross = np.asarray(src[:, rows[0], :], dtype=np.uint8)
    else:
        cross = np.bitwise_or.reduce(np.asarray(src[:, rows, :], dtype=np.uint8), axis=1)
    out = np.zeros(valid.shape, dtype=np.uint8)
    if not np.any(cross) or not np.any(valid):
        return out
    source_idx = np.asarray(dense_map.source_idx_map, dtype=np.int32)[row_sel]
    u_idx = np.asarray(dense_map.u_idx_map, dtype=np.int32)[row_sel]
    out[valid] = cross[
        source_idx[valid],
        u_idx[valid],
    ]
    return out

def _backproject_cartesian_radial_generic(
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool,
    reserve_bytes: int,
    workers: int,
    out_shape_tyx: Optional[Tuple[int, int, int]],
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]],
    sink_only: bool,
) -> np.ndarray | SinkOnlyProjectionResult:
    """Orientation-aware dense or sink-only backprojection for upright Radial views."""
    target_shape = (
        (int(radial_view.full_t), int(radial_view.full_h), int(radial_view.full_w))
        if out_shape_tyx is None else tuple(int(v) for v in out_shape_tyx)
    )
    if bool(sink_only) and projection_block_callback is None:
        raise ValueError(f'{desc}: sink_only=True requires projection_block_callback')
    stack_len, plane_shape = _radial_output_stack_and_plane_shape(radial_view, target_shape)
    plan, plan_stats = build_radial_backprojection_plan(radial_view)
    _log_radial_backprojection_densification(desc, plan_stats)
    base = radial_base_view_name(radial_view)
    if base not in ('sagittal', 'coronal'):
        raise ValueError(f'Generic Cartesian Radial branch received base {base!r}')

    out: Optional[np.ndarray] = None
    if not bool(sink_only):
        out = allocate_workspace_array(
            shape=target_shape,
            dtype=np.uint8,
            path=out_path,
            desc=f'{desc} workspace',
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
    else:
        print(
            f'{desc}: orientation-aware sink-only Radial projection active; skipping '
            f'{array_nbytes(target_shape, np.uint8) / GIB:.2f} GiB dense workspace.'
        )

    if not plan:
        _emit_projection_empty_range(
            projection_block_callback,
            0,
            int(target_shape[0]),
            (int(target_shape[1]), int(target_shape[2])),
            desc=desc,
            required=bool(sink_only),
        )
        if out is not None:
            flush_array(out)
            return out
        return SinkOnlyProjectionResult(tuple(int(v) for v in target_shape))

    grid = resolve_radial_processing_grid(radial_mask_mm, radial_view)
    dense_map = _radial_dense_map_for_processing(build_dense_radial_backprojection_map(
        radial_view, plan, out_shape_hw=plane_shape,
    ), grid)
    worker_count = choose_slice_parallel_workers(int(workers), int(stack_len))

    def _rows(stack_idx: int) -> np.ndarray:
        return _radial_processing_rows_for_output(grid, int(stack_len), int(stack_idx))

    if out is not None:
        def _build(stack_idx: int) -> None:
            plane = _radial_backproject_plane(
                radial_mask_mm, dense_map, _rows(int(stack_idx)),
            )
            if base == 'sagittal':
                out[:, int(stack_idx), :] = plane
            else:
                out[:, :, int(stack_idx)] = plane

        parallel_for_indices_chunked(
            int(stack_len),
            _build,
            max_workers=worker_count,
            desc=desc,
            show_progress=True,
            target_chunks_per_worker=4,
        )
        flush_array(out)
        if projection_block_callback is not None:
            block = max(1, _env_int('YOLO_TTA_PROJECTION_CALLBACK_BLOCK', 64))
            for t0 in range(0, int(target_shape[0]), block):
                t1 = min(int(target_shape[0]), t0 + block)
                _emit_projection_block_callback(
                    projection_block_callback, int(t0), np.asarray(out[t0:t1]),
                    desc=desc, required=False,
                )
        return out

    # The sink owns t-major bands.  Slice the orientation-specific dense circle map by
    # output-t band and scatter each stack plane directly into that band.  No full projected
    # volume, transposed copy, or second source-space scan is created.
    t_dim, out_h, out_w = (int(v) for v in target_shape)
    t_block = max(1, _env_int('YOLO_TTA_PROJECTION_CALLBACK_BLOCK', 64))
    for t0 in range(0, t_dim, t_block):
        t1 = min(t_dim, t0 + t_block)
        block_out = np.zeros((t1 - t0, out_h, out_w), dtype=np.uint8)

        def _fill_stack(stack_idx: int) -> None:
            plane_band = _radial_backproject_plane(
                radial_mask_mm,
                dense_map,
                _rows(int(stack_idx)),
                plane_row_slice=slice(int(t0), int(t1)),
            )
            if base == 'sagittal':
                block_out[:, int(stack_idx), :] = plane_band
            else:
                block_out[:, :, int(stack_idx)] = plane_band

        parallel_for_indices_chunked(
            int(stack_len),
            _fill_stack,
            max_workers=worker_count,
            desc=f'{desc} [t={t0}:{t1}]',
            show_progress=False,
            target_chunks_per_worker=4,
        )
        _emit_projection_block_callback(
            projection_block_callback,
            int(t0),
            block_out,
            desc=desc,
            required=True,
        )
    return SinkOnlyProjectionResult((t_dim, out_h, out_w))

if _numba is not None:
    # This tiny kernel intentionally avoids Numba's disk cache: the pipeline is often copied
    # to a new versioned filename/module, and a cached environment from the prior filename can
    # fail to import before the first projection.
    @_numba.njit(cache=False, nogil=True)  # type: ignore[misc]
    def _numba_or_tilted_radial_coordinates_into_packed(
        destination_flat: np.ndarray,
        ti: np.ndarray,
        yi: np.ndarray,
        xi: np.ndarray,
        out_h: int,
        packed_w: int,
    ) -> None:
        """Serial packed-bit OR in compiled code; repeated coordinates are intentional."""
        plane_stride = int(out_h) * int(packed_w)
        for index in range(int(ti.shape[0])):
            x = int(xi[index])
            destination_index = (
                int(ti[index]) * int(plane_stride)
                + int(yi[index]) * int(packed_w)
                + (x >> 3)
            )
            destination_flat[destination_index] |= np.uint8(1 << (7 - (x & 7)))
else:
    _numba_or_tilted_radial_coordinates_into_packed = None

def _or_tilted_radial_coordinates_into_packed(
    destination_flat: np.ndarray,
    ti: np.ndarray,
    yi: np.ndarray,
    xi: np.ndarray,
    *,
    out_h: int,
    packed_w: int,
) -> None:
    """OR final source coordinates into a C-order packed destination."""
    if _numba_or_tilted_radial_coordinates_into_packed is not None:
        _numba_or_tilted_radial_coordinates_into_packed(
            destination_flat,
            np.asarray(ti, dtype=np.int32),
            np.asarray(yi, dtype=np.int32),
            np.asarray(xi, dtype=np.int32),
            int(out_h),
            int(packed_w),
        )
        return
    packed_plane_stride = np.int64(int(out_h) * int(packed_w))
    packed_indices = ti.astype(np.int64, copy=False) * packed_plane_stride
    packed_indices += yi.astype(np.int64, copy=False) * np.int64(packed_w)
    packed_indices += (xi.astype(np.int64, copy=False) >> np.int64(3))
    bit_masks = np.left_shift(
        np.uint8(1),
        (np.int32(7) - (xi.astype(np.int32, copy=False) & np.int32(7))).astype(np.uint8),
    ).astype(np.uint8, copy=False)
    np.bitwise_or.at(destination_flat, packed_indices, bit_masks)

def _backproject_tilted_radial_volume_to_volume(
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool,
    reserve_bytes: int,
    workers: int,
    out_shape_tyx: Optional[Tuple[int, int, int]],
    known_row_occupancy: Optional[np.ndarray],
    known_slice_bboxes: Optional[np.ndarray],
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]],
    sink_only: bool,
) -> np.ndarray | SinkOnlyProjectionResult:
    """Compose Radial reconstruction and Tilted inverse projection into one destination.

    The retired implementation first wrote a dense ``(stack, plane_v, plane_u)`` tilted
    Cartesian volume, then read that roughly 25 GiB workspace a second time to scatter it
    into source geometry.  The dense path now gathers only foreground positions from one
    reconstructed radial frame and immediately scatters those coordinates into the final
    destination.  Sink-only callers use a source-space packed-bit accumulator (one eighth the
    uint8 size) and emit completed t-major blocks directly to the callback; neither mode ever
    materializes the tilted base stack.
    """
    if bool(sink_only) and projection_block_callback is None:
        raise ValueError(f'{desc}: sink_only=True requires projection_block_callback')
    if not is_tilted_radial_view(radial_view):
        raise ValueError(f'{desc}: composed tilted-Radial projection requires a tilted Radial view')

    tilted_source = radial_source_tilted_view(radial_view)
    target_shape = (
        (int(radial_view.full_t), int(radial_view.full_h), int(radial_view.full_w))
        if out_shape_tyx is None else tuple(int(v) for v in out_shape_tyx)
    )
    t_dim, out_h, out_w = (int(v) for v in target_shape)
    if min(t_dim, out_h, out_w) <= 0:
        raise ValueError(f'{desc}: invalid output shape {target_shape}')

    plan, plan_stats = build_radial_backprojection_plan(radial_view)
    _log_radial_backprojection_densification(desc, plan_stats)
    if not plan:
        _emit_projection_empty_range(
            projection_block_callback, 0, int(t_dim), (int(out_h), int(out_w)),
            desc=desc, required=bool(sink_only),
        )
        if bool(sink_only):
            return SinkOnlyProjectionResult(tuple(int(v) for v in target_shape))
        destination = allocate_workspace_array(
            shape=target_shape,
            dtype=np.uint8,
            path=out_path,
            desc=f'{desc} direct composed destination',
            prefer_memory=bool(prefer_memory),
            prefer_memfd=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
        flush_array(destination)
        return destination

    plane_h, plane_w = (int(tilted_source.src_h), int(tilted_source.src_w))
    stack_len = int(tilted_source.num_slices)
    grid = resolve_radial_processing_grid(radial_mask_mm, radial_view)
    dense_map = _radial_dense_map_for_processing(
        build_dense_radial_backprojection_map(
            radial_view, plan, out_shape_hw=(int(plane_h), int(plane_w)),
        ),
        grid,
    )
    valid_flat = np.flatnonzero(np.asarray(dense_map.valid_mask, dtype=bool).reshape(-1))
    valid_v = np.ascontiguousarray((valid_flat // int(plane_w)).astype(np.int32, copy=False))
    valid_u = np.ascontiguousarray((valid_flat % int(plane_w)).astype(np.int32, copy=False))
    radial_source_idx = np.ascontiguousarray(
        np.asarray(dense_map.source_idx_map, dtype=np.int32).reshape(-1)[valid_flat]
    )
    radial_u_idx = np.ascontiguousarray(
        np.asarray(dense_map.u_idx_map, dtype=np.int32).reshape(-1)[valid_flat]
    )

    row_occ: Optional[np.ndarray] = None
    if known_row_occupancy is not None:
        candidate = np.asarray(known_row_occupancy, dtype=bool).reshape(-1)
        if int(candidate.shape[0]) == int(grid.processing_h):
            row_occ = candidate
    # Validate metadata even though the direct gather currently uses row occupancy as its
    # high-value skip.  This catches stale descriptors at the same boundary as upright Radial.
    _validated_radial_slice_bboxes(
        known_slice_bboxes,
        int(np.asarray(radial_mask_mm).shape[0]),
        int(grid.processing_h),
        int(np.asarray(radial_mask_mm).shape[2]),
    )

    work_t = int(tilted_source.full_t)
    work_h = int(tilted_source.full_h)
    work_w = int(tilted_source.full_w)
    base_view = tilted_base_view_name(tilted_source)
    tan_alpha = float(math.tan(math.radians(float(tilted_source.tilt_angle_deg))))
    vertical = str(tilted_source.tilt_direction) == 'vertical'
    if not vertical and str(tilted_source.tilt_direction) != 'horizontal':
        raise ValueError(f'{desc}: unsupported tilt direction {tilted_source.tilt_direction!r}')
    axis_center = float(
        (int(tilted_source.src_h) - 1) / 2.0
        if vertical else (int(tilted_source.src_w) - 1) / 2.0
    )

    def _map_axis_to_out(values: np.ndarray, in_len: int, out_len: int) -> np.ndarray:
        if int(in_len) == int(out_len):
            return values.astype(np.int32, copy=False)
        mapped = (values.astype(np.int64, copy=False) * int(out_len)) // int(in_len)
        return np.minimum(mapped, int(out_len) - 1).astype(np.int32, copy=False)

    def _compose_frame_indices(
        frame_index: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return final source coordinates contributed by one tilted-Radial frame."""
        rows = _radial_processing_rows_for_output(grid, int(stack_len), int(frame_index))
        if int(rows.size) <= 0:
            return None
        if row_occ is not None and not bool(np.any(row_occ[rows])):
            return None
        hit = np.zeros((int(valid_flat.size),), dtype=bool)
        for processing_row in rows.tolist():
            if row_occ is not None and not bool(row_occ[int(processing_row)]):
                continue
            hit |= np.asarray(
                radial_mask_mm[
                    radial_source_idx,
                    int(processing_row),
                    radial_u_idx,
                ],
                dtype=np.uint8,
            ) != 0
        if not bool(np.any(hit)):
            return None
        selected = np.flatnonzero(hit)
        vv = valid_v[selected]
        uu = valid_u[selected]
        axis_coords = vv if vertical else uu
        stack_float = float(tilted_frame_center(tilted_source, int(frame_index))) + (
            float(tan_alpha) * (axis_coords.astype(np.float32, copy=False) - float(axis_center))
        )
        ss = np.rint(stack_float).astype(np.int32, copy=False)
        inside = (ss >= 0) & (ss < int(tilted_stack_axis_length(tilted_source)))
        if not bool(np.any(inside)):
            return None
        ss = ss[inside]
        vv = vv[inside]
        uu = uu[inside]
        if base_view == 'transverse':
            ti = _map_axis_to_out(ss, work_t, t_dim)
            yi = _map_axis_to_out(vv, work_h, out_h)
            xi = _map_axis_to_out(uu, work_w, out_w)
        elif base_view == 'sagittal':
            ti = _map_axis_to_out(vv, work_t, t_dim)
            yi = _map_axis_to_out(ss, work_h, out_h)
            xi = _map_axis_to_out(uu, work_w, out_w)
        elif base_view == 'coronal':
            ti = _map_axis_to_out(vv, work_t, t_dim)
            yi = _map_axis_to_out(uu, work_h, out_h)
            xi = _map_axis_to_out(ss, work_w, out_w)
        else:
            raise ValueError(f'{desc}: unsupported Tilted base {base_view!r}')
        return (
            np.ascontiguousarray(ti, dtype=np.int32),
            np.ascontiguousarray(yi, dtype=np.int32),
            np.ascontiguousarray(xi, dtype=np.int32),
        )

    worker_count = choose_slice_parallel_workers(int(workers), int(stack_len))
    avoided_base_bytes = array_nbytes(
        (int(tilted_source.num_slices), int(tilted_source.src_h), int(tilted_source.src_w)),
        np.uint8,
    )

    if not bool(sink_only):
        destination = allocate_workspace_array(
            shape=target_shape,
            dtype=np.uint8,
            path=out_path,
            desc=f'{desc} direct composed destination',
            prefer_memory=bool(prefer_memory),
            prefer_memfd=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
        print(
            f'{desc}: v16.1.3 direct tilted-Radial composition active; no '
            f'{avoided_base_bytes / GIB:.2f} GiB tilted base-stack intermediate is allocated.'
        )
        destination_flat = np.asarray(destination).reshape(-1)
        plane_stride = np.int64(int(out_h) * int(out_w))

        def _scatter_frame(frame_index: int) -> None:
            coordinates = _compose_frame_indices(int(frame_index))
            if coordinates is None:
                return
            ti, yi, xi = coordinates
            flat = ti.astype(np.int64, copy=False) * plane_stride
            flat += yi.astype(np.int64, copy=False) * np.int64(out_w)
            flat += xi.astype(np.int64, copy=False)
            # Same-byte stores are idempotent; this preserves the established Tilted
            # projector's parallel sparse-scatter contract.
            destination_flat[flat] = np.uint8(1)

        parallel_for_indices_chunked(
            int(stack_len),
            _scatter_frame,
            max_workers=int(worker_count),
            desc=f'{desc} [direct composed frames]',
            show_progress=True,
            target_chunks_per_worker=4,
        )
        flush_array(destination)
        if projection_block_callback is not None:
            block = max(1, _env_int('YOLO_TTA_PROJECTION_CALLBACK_BLOCK', 64))
            for t0 in range(0, int(t_dim), int(block)):
                t1 = min(int(t_dim), int(t0) + int(block))
                block_view = np.asarray(destination[int(t0):int(t1)])
                if bool(np.any(block_view)):
                    _emit_projection_block_callback(
                        projection_block_callback, int(t0), block_view,
                        desc=desc, required=False,
                    )
                else:
                    _emit_projection_empty_range(
                        projection_block_callback, int(t0), int(t1 - t0),
                        (int(out_h), int(out_w)), desc=desc, required=False,
                    )
        return destination

    # Sink-only mode commits through a packed source-space accumulator.  This preserves
    # exact union semantics while reducing the live destination from one byte to one bit per
    # voxel and lets the cvol writer receive final t-major blocks without a dense uint8 file.
    packed_w = int((int(out_w) + 7) // 8)
    packed_path = Path(out_path).with_name(Path(out_path).name + '.tilted_radial.bits.dat')
    packed_destination: Optional[np.ndarray] = None
    try:
        packed_destination = allocate_workspace_array(
            shape=(int(t_dim), int(out_h), int(packed_w)),
            dtype=np.uint8,
            path=packed_path,
            desc=f'{desc} direct composed packed sink',
            prefer_memory=bool(prefer_memory),
            prefer_memfd=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
        print(
            f'{desc}: v16.1.3 direct tilted-Radial packed sink active; no '
            f'{avoided_base_bytes / GIB:.2f} GiB tilted base stack and no '
            f'{array_nbytes(target_shape, np.uint8) / GIB:.2f} GiB uint8 source destination. '
            f'Packed source union={array_nbytes((t_dim, out_h, packed_w), np.uint8) / GIB:.2f} GiB.'
        )
        packed_flat = np.asarray(packed_destination).reshape(-1)
        sink_workers = max(
            1,
            min(
                int(worker_count),
                _env_int('YOLO_TTA_TILTED_RADIAL_SINK_WORKERS', int(worker_count)),
            ),
        )
        for coordinates in parallel_map_unordered(
            _compose_frame_indices,
            range(int(stack_len)),
            max_workers=int(sink_workers),
            max_pending=max(2, int(sink_workers)),
        ):
            if coordinates is None:
                continue
            ti, yi, xi = coordinates
            _or_tilted_radial_coordinates_into_packed(
                packed_flat, ti, yi, xi,
                out_h=int(out_h), packed_w=int(packed_w),
            )

        flush_array(packed_destination)
        target_block_bytes = max(
            16 * 1024 * 1024,
            int(max(16.0, _env_float('YOLO_TTA_TILTED_RADIAL_SINK_BLOCK_MIB', 256.0)) * 1024 * 1024),
        )
        callback_cap = max(1, _env_int('YOLO_TTA_PROJECTION_CALLBACK_BLOCK', 64))
        block_slices = max(
            1,
            min(
                int(callback_cap),
                int(target_block_bytes // max(1, int(out_h) * int(out_w))),
            ),
        )
        for t0 in range(0, int(t_dim), int(block_slices)):
            t1 = min(int(t_dim), int(t0) + int(block_slices))
            packed_block = np.asarray(packed_destination[int(t0):int(t1)], dtype=np.uint8)
            if not bool(np.any(packed_block)):
                _emit_projection_empty_range(
                    projection_block_callback, int(t0), int(t1 - t0),
                    (int(out_h), int(out_w)), desc=desc, required=True,
                )
                continue
            block_view = np.unpackbits(
                packed_block,
                axis=2,
                count=int(out_w),
                bitorder='big',
            ).astype(np.uint8, copy=False)
            _emit_projection_block_callback(
                projection_block_callback, int(t0), block_view,
                desc=desc, required=True,
            )
        return SinkOnlyProjectionResult(tuple(int(v) for v in target_shape))
    finally:
        if packed_destination is not None:
            close_memmap_array_without_flush(packed_destination)
        try:
            packed_path.unlink(missing_ok=True)
        except Exception:
            pass

def backproject_radial_volume_to_volume(
    radial_mask_mm: np.ndarray,
    radial_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    out_shape_tyx: Optional[Tuple[int, int, int]] = None,
    known_row_occupancy: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
    projection_block_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    sink_only: bool = False,
) -> np.ndarray | SinkOnlyProjectionResult:
    """Backproject a radial view-native mask stack into orthogonal (t, Y, X).

 ``out_shape_tyx`` backprojects directly into final source geometry. Transverse retains the
 optimized t-major implementation; sagittal/coronal build the dense circle map on their final
 ``(t,X)``/``(t,Y)`` planes. Tilted Radial views compose radial reconstruction with the
 Tilted shear and write final source geometry directly, without a dense tilted base stack.
 This terminal geometry map converts working-space circles into source-space ellipses when t
 was cube-resized.

 ``sink_only=True`` commits completed output-t blocks transactionally for every upright or
 tilted Cartesian Radial base; tilted-Radial uses one transient final destination but never
 materializes the retired base-stack intermediate."""
    if not is_radial_view(radial_view):
        raise ValueError('backproject_radial_volume_to_volume expects a radial view')

    if is_tilted_radial_view(radial_view):
        return _backproject_tilted_radial_volume_to_volume(
            radial_mask_mm, radial_view, out_path, desc,
            prefer_memory=bool(prefer_memory), reserve_bytes=int(reserve_bytes),
            workers=int(workers), out_shape_tyx=out_shape_tyx,
            known_row_occupancy=known_row_occupancy,
            known_slice_bboxes=known_slice_bboxes,
            projection_block_callback=projection_block_callback, sink_only=bool(sink_only),
        )
    if radial_base_view_name(radial_view) != 'transverse':
        return _backproject_cartesian_radial_generic(
            radial_mask_mm, radial_view, out_path, desc,
            prefer_memory=bool(prefer_memory), reserve_bytes=int(reserve_bytes),
            workers=int(workers), out_shape_tyx=out_shape_tyx,
            projection_block_callback=projection_block_callback, sink_only=bool(sink_only),
        )

    processing_grid = resolve_radial_processing_grid(radial_mask_mm, radial_view)
    native_work_t = int(radial_view.src_h)
    work_t = int(processing_grid.processing_h)
    work_h = int(radial_view.full_h)
    work_w = int(radial_view.full_w)
    if out_shape_tyx is None:
        t_dim, out_h, out_w = native_work_t, work_h, work_w
    else:
        t_dim, out_h, out_w = (int(v) for v in out_shape_tyx)

    if bool(sink_only) and projection_block_callback is None:
        raise ValueError(f'{desc}: sink_only=True requires projection_block_callback')
    vol_mm: Optional[np.ndarray] = None
    if not bool(sink_only):
        vol_mm = allocate_workspace_array(
            shape=(t_dim, out_h, out_w),
            dtype=np.uint8,
            path=out_path,
            desc=f'{desc} workspace',
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
        )
    else:
        print(
            f'{desc}: v13.3.17 C2 sink-only radial projection active; '
            f'skipping {array_nbytes((t_dim, out_h, out_w), np.uint8) / GIB:.2f} GiB dense workspace.'
        )

    plan, plan_stats = build_radial_backprojection_plan(radial_view)
    if bool(plan_stats.get('densified', 0.0)):
        print(
            f"{desc}: densifying radial backprojection for continuity "
            f"from {int(plan_stats['source_frames'])} source frame(s) at "
            f"{float(plan_stats['provided_spacing_deg']):.6g}° spacing to "
            f"{int(plan_stats['backprojection_angles'])} backprojection angle(s) at "
            f"{float(plan_stats['effective_spacing_deg']):.6g}° spacing "
            f"(full-coverage threshold {float(plan_stats['coverage_spacing_deg']):.6g}°)"
        )

    if not plan:
        _emit_projection_empty_range(
            projection_block_callback,
            0,
            int(t_dim),
            (int(out_h), int(out_w)),
            desc=desc,
            required=bool(sink_only),
        )
        if vol_mm is not None:
            flush_array(vol_mm)
            return vol_mm
        return SinkOnlyProjectionResult((int(t_dim), int(out_h), int(out_w)))

    dense_map = _radial_dense_map_for_processing(build_dense_radial_backprojection_map(
        radial_view, plan, out_shape_hw=None if out_shape_tyx is None else (out_h, out_w)
    ), processing_grid)
    valid = np.asarray(dense_map.valid_mask, dtype=bool)
    source_idx_map = np.asarray(dense_map.source_idx_map, dtype=np.int32)
    u_idx_map = np.asarray(dense_map.u_idx_map, dtype=np.int32)

    worker_count = choose_slice_parallel_workers(int(workers), int(t_dim))
    same_t_axis = bool(
        not processing_grid.reduced
        and int(native_work_t) == int(t_dim)
        and int(work_t) == int(t_dim)
    )

    # flatten the dense gather once. valid_pos holds the flat output positions
    # of in-ROI pixels; flat_src holds each one's flat (source_frame, u) index into a radial
    # cross-section. Per slice this turns the 2-array fancy gather + two masked passes into one
    # take + one indexed store, and empty radial cross-sections are skipped outright (the
    # nearly-empty bridge layers become almost free).
    u_len = int(radial_mask_mm.shape[2])
    valid_pos = np.flatnonzero(valid.reshape(-1))
    flat_src = (
        source_idx_map.reshape(-1)[valid_pos].astype(np.int64) * np.int64(u_len)
        + u_idx_map.reshape(-1)[valid_pos]
    )

    def _v_range_for_t(t_idx: int) -> Tuple[int, int]:
        if same_t_axis:
            return int(t_idx), int(t_idx) + 1
        mapped = _radial_processing_rows_for_output(
            processing_grid, int(t_dim), int(t_idx),
        )
        if mapped.size <= 0:
            return 0, 0
        return int(mapped[0]), int(mapped[-1]) + 1

    # device-union row occupancy (valid for the pre-interpolation layer only,
    # per the caller's contract) replaces both the GPU path's occupancy scan and the CPU
    # path's per-row strided np.any reads.
    row_occ = None
    if known_row_occupancy is not None:
        row_occ = np.asarray(known_row_occupancy, dtype=bool).reshape(-1)
        if int(row_occ.shape[0]) != int(work_t):
            row_occ = None
    radial_slice_bboxes = _validated_radial_slice_bboxes(
        known_slice_bboxes,
        int(radial_mask_mm.shape[0]),
        int(work_t),
        int(u_len),
    )

    gpu_done = False
    if gpu_backproject_enabled():
        try:
            if not processing_grid.reduced:
                gpu_done = _radial_backproject_gpu_resident(
                    radial_mask_mm, vol_mm, valid, source_idx_map, u_idx_map, _v_range_for_t,
                    int(t_dim), int(out_h), int(out_w), desc,
                    known_row_occupancy=row_occ,
                    projection_block_callback=projection_block_callback,
                    sink_only=bool(sink_only),
                )
            if not gpu_done:
                gpu_done = _radial_backproject_gpu_streaming(
                    radial_mask_mm, vol_mm, valid_pos, flat_src, _v_range_for_t,
                    int(t_dim), int(out_h), int(out_w), desc,
                    known_row_occupancy=row_occ,
                    known_slice_bboxes=radial_slice_bboxes,
                    projection_block_callback=projection_block_callback,
                    sink_only=bool(sink_only),
                )
        except Exception as exc:
            # Some GPU blocks may already have been indexed. The CPU retry rewrites all
            # output blocks, so retain projection correctness but discard that partial store.
            _abort_projection_block_callback(projection_block_callback, exc)
            if bool(sink_only):
                raise
            projection_block_callback = None
            print(f'{desc}: GPU backprojection failed ({exc}); using the CPU path.')
            gpu_done = False

    if not gpu_done:
        if row_occ is None:
            row_occ = _radial_row_occupancy(radial_mask_mm, desc)
        cpu_t_chunk = max(1, _env_int('YOLO_TTA_CPU_BACKPROJECT_T_CHUNK', 128))
        n_blocks = int(math.ceil(float(t_dim) / float(cpu_t_chunk)))
        stage_workers = max(1, min(8, int(worker_count), _cpu_count()))
        stage_pool = _acquire_parallel_pool(stage_workers)

        # Bound simultaneous vector blocks by an 8 GiB temporary budget. The old per-t
        # implementation launched many strided readers concurrently; two large t-major
        # blocks generally saturate memory bandwidth without recreating that contention.
        max_block_rows = int(math.ceil(float(cpu_t_chunk) * float(work_t) / float(max(1, t_dim)))) + 2
        per_block_bytes = (
            int(max_block_rows) * int(radial_mask_mm.shape[0]) * int(u_len)
            + int(max_block_rows) * int(valid_pos.shape[0])
        )
        block_workers = max(1, min(
            int(worker_count),
            int(n_blocks),
            2,
            max(1, int((8 * GIB) // max(1, int(per_block_bytes)))),
        ))

        def _backproject_t_block(block_idx: int) -> None:
            t0 = int(block_idx) * int(cpu_t_chunk)
            t1 = min(int(t_dim), int(t0) + int(cpu_t_chunk))
            v0 = int(_v_range_for_t(int(t0))[0])
            v1 = int(_v_range_for_t(int(t1 - 1))[1])
            if not bool(np.any(row_occ[int(v0):int(v1)])):
                _emit_projection_empty_range(
                    projection_block_callback,
                    int(t0),
                    int(t1 - t0),
                    (int(out_h), int(out_w)),
                    desc=desc,
                    required=bool(sink_only),
                )
                return
            t_major = np.empty(
                (int(v1 - v0), int(radial_mask_mm.shape[0]), int(u_len)), dtype=np.uint8,
            )
            _stage_radial_t_major_block(
                radial_mask_mm,
                int(v0),
                int(v1),
                t_major,
                slice_bboxes=radial_slice_bboxes,
                stage_pool=stage_pool,
                stage_workers=stage_workers,
            )
            gathered_rows = t_major.reshape(int(v1 - v0), -1).take(flat_src, axis=1)
            block_out = (
                np.asarray(vol_mm[int(t0):int(t1)])
                if vol_mm is not None
                else np.zeros((int(t1 - t0), int(out_h), int(out_w)), dtype=np.uint8)
            )
            out_flat = block_out.reshape(int(t1 - t0), -1)
            if same_t_axis:
                out_flat[:, valid_pos] = gathered_rows[:int(t1 - t0)]
                _emit_projection_block_callback(
                    projection_block_callback,
                    int(t0),
                    block_out,
                    desc=desc,
                    required=bool(sink_only),
                )
                return
            for t in range(int(t0), int(t1)):
                vs, ve = _v_range_for_t(int(t))
                local = gathered_rows[int(vs - v0):int(ve - v0)]
                if int(local.shape[0]) == 1:
                    reduced = local[0]
                else:
                    reduced = np.bitwise_or.reduce(local, axis=0)
                out_flat[int(t - t0), valid_pos] = reduced
            _emit_projection_block_callback(
                projection_block_callback,
                int(t0),
                block_out,
                desc=desc,
                required=bool(sink_only),
            )

        try:
            parallel_for_indices(
                int(n_blocks),
                _backproject_t_block,
                max_workers=int(block_workers),
                desc=f'{desc} [t-major x{int(cpu_t_chunk)}]',
                show_progress=True,
            )
        finally:
            _release_parallel_pool(stage_workers, stage_pool)

    if vol_mm is not None:
        flush_array(vol_mm)
        return vol_mm
    return SinkOnlyProjectionResult((int(t_dim), int(out_h), int(out_w)))

def backproject_tilted_volume_to_volume(
    tilted_mask_mm: np.ndarray,
    tilted_view: ViewInfo,
    out_path: Path,
    desc: str,
    *,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    out_shape_tyx: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Backproject a Tilted volume into the requested source geometry.
    
    Supports reduced processing rasters and direct output-geometry scattering without an intermediate full native view."""
    if not is_tilted_view(tilted_view):
        raise ValueError('backproject_tilted_volume_to_volume expects a Tilted View')

    work_t = int(tilted_view.full_t)
    work_h = int(tilted_view.full_h)
    work_w = int(tilted_view.full_w)

    src = np.asarray(tilted_mask_mm)
    if int(src.ndim) != 3 or int(src.shape[0]) != int(tilted_view.num_slices):
        raise ValueError(
            f'{desc}: tilted layer shape {tuple(src.shape)} must start with '
            f'{int(tilted_view.num_slices)} frames'
        )
    plane_h, plane_w = int(src.shape[1]), int(src.shape[2])
    native_plane = (int(tilted_view.src_h), int(tilted_view.src_w))
    reduced_processing = bool(
        delayed_native_expansion_enabled()
        and (int(plane_h), int(plane_w)) != native_plane
    )
    if reduced_processing and int(plane_h) != int(plane_w):
        raise ValueError(
            f'{desc}: delayed Tilted processing requires a square inference raster, '
            f'got {(int(plane_h), int(plane_w))}'
        )
    base_view = tilted_base_view_name(tilted_view)

    if reduced_processing and out_shape_tyx is not None:
        # Use the exact same terminal definition with and without --save nrrd:
        # reduced-view shear -> reduced orthogonal grid -> one orthogonal restore. Shear and
        # native-plane expansion do not commute at rounded stack boundaries, so expanding each
        # tilted frame first would make final masks depend on whether component NRRDs were saved.
        reduced_path = out_path.with_name(out_path.stem + '.d6_reduced_orthogonal.u8.dat')
        reduced_mm: Optional[np.ndarray] = None
        restored_mm: Optional[np.ndarray] = None
        restore_succeeded = False
        try:
            reduced_mm = backproject_tilted_volume_to_volume(
                tilted_mask_mm=src,
                tilted_view=tilted_view,
                out_path=reduced_path,
                desc=f'{desc} [reduced orthogonal projection]',
                prefer_memory=bool(prefer_memory),
                reserve_bytes=int(reserve_bytes),
                workers=int(workers),
                out_shape_tyx=None,
            )
            target_shape = tuple(int(v) for v in out_shape_tyx)
            restored_mm = allocate_workspace_array(
                shape=target_shape,
                dtype=np.uint8,
                path=out_path,
                desc=f'{desc} workspace',
                prefer_memory=bool(prefer_memory),
                reserve_bytes=int(reserve_bytes),
            )

            def _restore_tilted_slice(z_idx: int) -> None:
                restored_mm[int(z_idx), :, :] = _read_layer_slice_in_output_shape(
                    reduced_mm, target_shape, int(z_idx),
                )

            parallel_for_indices_chunked(
                int(target_shape[0]),
                _restore_tilted_slice,
                max_workers=choose_slice_parallel_workers(int(workers), int(target_shape[0])),
                desc=f'{desc} [terminal reduced-grid restore]',
                show_progress=True,
                target_chunks_per_worker=2,
            )
            flush_array(restored_mm)
            restore_succeeded = True
            return restored_mm
        finally:
            if reduced_mm is not None:
                close_memmap_array(reduced_mm)
            if restored_mm is not None and not bool(restore_succeeded):
                close_memmap_array(restored_mm)
            try:
                reduced_path.unlink(missing_ok=True)
            except Exception:
                pass

    if out_shape_tyx is None:
        if reduced_processing:
            if base_view == 'transverse':
                t_dim, out_h, out_w = int(work_t), int(plane_h), int(plane_w)
            elif base_view == 'sagittal':
                t_dim, out_h, out_w = int(plane_h), int(work_h), int(plane_w)
            elif base_view == 'coronal':
                t_dim, out_h, out_w = int(plane_h), int(plane_w), int(work_w)
            else:  # pragma: no cover
                raise ValueError(f'Unsupported Tilted View base: {base_view}')
        else:
            t_dim, out_h, out_w = work_t, work_h, work_w
    else:
        t_dim, out_h, out_w = (int(v) for v in out_shape_tyx)
    if t_dim <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError(
            f'Tilted view {tilted_view.name} has invalid output geometry '
            f'(t,Y,X)=({t_dim},{out_h},{out_w})'
        )

    vol_mm = allocate_workspace_array(
        shape=(t_dim, out_h, out_w),
        dtype=np.uint8,
        path=out_path,
        desc=f'{desc} workspace',
        prefer_memory=bool(prefer_memory),
        reserve_bytes=int(reserve_bytes),
    )

    tan_alpha = float(math.tan(math.radians(float(tilted_view.tilt_angle_deg))))
    if str(tilted_view.tilt_direction) == 'vertical':
        axis_center = float((int(tilted_view.src_h) - 1) / 2.0)
    elif str(tilted_view.tilt_direction) == 'horizontal':
        axis_center = float((int(tilted_view.src_w) - 1) / 2.0)
    else:  # pragma: no cover
        raise ValueError(f'Unsupported tilt direction: {tilted_view.tilt_direction}')

    stack_len = tilted_stack_axis_length(tilted_view)

    worker_count = choose_slice_parallel_workers(int(workers), int(tilted_view.num_slices))

    def _map_axis_to_out(idx_arr: np.ndarray, in_len: int, out_len: int) -> np.ndarray:
        # union-biased floor scaling from working-grid indices to source indices
        # (in_len >= out_len; the cube resize only grows axes). Identity when the axis is unscaled.
        if int(in_len) == int(out_len):
            return idx_arr
        mapped = (idx_arr.astype(np.int64, copy=False) * int(out_len)) // int(in_len)
        return np.minimum(mapped, int(out_len) - 1).astype(np.int32, copy=False)

    # per-frame invariants hoisted; the frame's emptiness test and nonzero run on
    # the raw uint8 plane (the old dtype=bool asarray made a full-plane cast COPY per frame);
    # the three mapped coordinate arrays collapse into ONE flat scatter index (built with
    # GIL-releasing int64 ufuncs) so the GIL-held portion is a single 1-array store.
    is_vertical_dir = str(tilted_view.tilt_direction) == 'vertical'
    vol_flat_scatter = np.asarray(vol_mm).reshape(-1)
    plane_stride = np.int64(int(out_h) * int(out_w))
    reduced_out_to_native: Optional[np.ndarray] = None
    if reduced_processing and out_shape_tyx is None:
        # Frame-invariant canonical-grid -> native-view transform. Building/inverting this
        # affine inside every one of ~3k frames per Tilted stack would serialize the otherwise
        # parallel projection on OpenCV/NumPy setup work.
        canonical = build_affine(
            view=str(tilted_view.name),
            src_w=int(tilted_view.src_w),
            src_h=int(tilted_view.src_h),
            out_size=int(plane_w),
            angle_deg=0.0,
            pad_mode=str(tilted_view.pad_mode),
        )
        reduced_out_to_native = np.asarray(canonical.M_out_to_src, dtype=np.float32)

    def _backproject_frame(frame_idx: int) -> None:
        frame_arr = np.asarray(src[int(frame_idx)])
        if not np.any(frame_arr):
            return
        vv, uu = np.nonzero(frame_arr)
        if vv.size <= 0:
            return

        frame_center = float(tilted_frame_center(tilted_view, int(frame_idx)))
        if reduced_processing and out_shape_tyx is None:
            # Convert only the coordinate used by the physical shear to native-view units;
            # the orthogonal in-plane coordinates themselves stay reduced.
            if reduced_out_to_native is None:  # pragma: no cover - guarded above
                raise RuntimeError(f'{desc}: reduced Tilted affine was not initialized')
            m = reduced_out_to_native
            native_u = m[0, 0] * uu.astype(np.float32, copy=False) + m[0, 1] * vv + m[0, 2]
            native_v = m[1, 0] * uu.astype(np.float32, copy=False) + m[1, 1] * vv + m[1, 2]
            axis_coords = native_v if is_vertical_dir else native_u
        else:
            axis_coords = vv if is_vertical_dir else uu
        stack_float = frame_center + tan_alpha * (axis_coords.astype(np.float32, copy=False) - axis_center)

        ss = np.rint(stack_float).astype(np.int32, copy=False)
        valid = (ss >= 0) & (ss < int(stack_len))
        if not np.any(valid):
            return

        ss_v = ss[valid]
        vv_v = vv[valid]
        uu_v = uu[valid]
        if reduced_processing and out_shape_tyx is None:
            # Reduced orthogonal projection. The stacking axis keeps working-grid pitch;
            # the two axes represented by the model plane retain inference pitch.
            if base_view == 'transverse':
                ti, yi, xi = ss_v, vv_v, uu_v
            elif base_view == 'sagittal':
                ti, yi, xi = vv_v, ss_v, uu_v
            elif base_view == 'coronal':
                ti, yi, xi = vv_v, uu_v, ss_v
            else:  # pragma: no cover
                raise ValueError(f'Unsupported Tilted View base: {base_view}')
        elif base_view == 'transverse':
            # base in-plane: horizontal X=uu, vertical Y=vv; stack t=ss
            ti = _map_axis_to_out(ss_v, work_t, t_dim)
            yi = _map_axis_to_out(vv_v, work_h, out_h)
            xi = _map_axis_to_out(uu_v, work_w, out_w)
        elif base_view == 'sagittal':
            # base in-plane: horizontal X=uu, vertical t=vv; stack Y=ss
            ti = _map_axis_to_out(vv_v, work_t, t_dim)
            yi = _map_axis_to_out(ss_v, work_h, out_h)
            xi = _map_axis_to_out(uu_v, work_w, out_w)
        elif base_view == 'coronal':
            # base in-plane: horizontal Y=uu, vertical t=vv; stack X=ss
            ti = _map_axis_to_out(vv_v, work_t, t_dim)
            yi = _map_axis_to_out(uu_v, work_h, out_h)
            xi = _map_axis_to_out(ss_v, work_w, out_w)
        else:  # pragma: no cover
            raise ValueError(f'Unsupported Tilted View base: {base_view}')
        flat = ti.astype(np.int64, copy=False) * plane_stride
        flat += yi.astype(np.int64, copy=False) * np.int64(out_w)
        flat += xi.astype(np.int64, copy=False)
        vol_flat_scatter[flat] = np.uint8(1)

    parallel_for_indices_chunked(
        int(tilted_view.num_slices),
        _backproject_frame,
        max_workers=worker_count,
        desc=desc,
        show_progress=True,
        target_chunks_per_worker=4,
    )

    flush_array(vol_mm)
    return vol_mm

@dataclass(frozen=True)
class ViewBackprojectionQueueJob:
    model_name: str
    view: ViewInfo
    native_source: np.ndarray
    out_path: Path
    desc: str
    min_radius: float = 0.0
    workers: int = 1
    # final source geometry to backproject directly into
    # (single resample). None keeps the historical working-geometry target.
    out_shape_tyx: Optional[Tuple[int, int, int]] = None

class HybridBackprojectionQueue:
    """Sequential Radial/Tilted backprojection queue with a full CPU fallback budget.

 Upright Radial views use orientation-aware dense or sink-only projection; transverse may also
 use the resident GPU backprojector. Tilted Radial jobs use the direct composed Radial/shear
 projector, while ordinary Tilted jobs use the shared CPU shear path. The historical
 class name is retained to avoid scheduler call-site churn."""

    def __init__(self, *, cpu_workers: int = 1) -> None:
        self.cpu_workers = max(1, int(cpu_workers))

    def _run_job(self, job: ViewBackprojectionQueueJob) -> Tuple[str, str, np.ndarray]:
        view_local = job.view
        if view_local.family == 'radial':
            projected = backproject_radial_volume_to_volume(
                radial_mask_mm=job.native_source,
                radial_view=view_local,
                out_path=job.out_path,
                desc=job.desc,
                prefer_memory=True,
                workers=int(job.workers),
                out_shape_tyx=job.out_shape_tyx,
            )
        elif is_tilted_view(view_local):
            projected = backproject_tilted_volume_to_volume(
                tilted_mask_mm=job.native_source,
                tilted_view=view_local,
                out_path=job.out_path,
                desc=job.desc,
                prefer_memory=True,
                workers=int(job.workers),
                out_shape_tyx=job.out_shape_tyx,
            )
        else:  # pragma: no cover
            raise ValueError(f'Unsupported queued backprojection view family: {view_local.family}')

        # min_radius is applied per view in each view's OWN native 2D slice plane BEFORE
        # backprojection, so it is NOT re-applied here; job.min_radius is not consumed.
        return str(job.model_name), str(view_local.name), projected

    def run(self, jobs: Sequence[ViewBackprojectionQueueJob]) -> List[Tuple[str, str, np.ndarray]]:
        results: List[Tuple[str, str, np.ndarray]] = []
        for job in jobs:
            if is_radial_view(job.view):
                if is_tilted_radial_view(job.view):
                    backend_note = 'direct composed tilted-Radial projection'
                elif radial_base_view_name(job.view) == 'transverse':
                    backend_note = 'transverse Radial GPU backprojection path (CPU fallback)'
                else:
                    backend_note = (
                        f'{radial_base_view_name(job.view)} Radial orientation-aware '
                        'dense/sink-only CPU path'
                    )
            else:
                backend_note = 'CPU tilted path'
            print(f'Backprojection queue: running {job.model_name}/{job.view.name} via {backend_note}')
            results.append(self._run_job(job))
        return results


# Late imports keep callable-only dependency cycles import-safe.
from ._latebind import bind_late_symbols as _bind_late_symbols

_bind_late_symbols(
    __name__,
    globals(),
    {
        "config": (
            "GIB",
        ),
        "cuda_backend": (
            "GpuRenderedYoloSource",
            "GpuTileRenderedYoloSource",
        ),
        "geometry": (
            "ViewInfo",
            "_affine2x3_to_3x3",
            "_cupy_external_stream",
            "build_affine",
            "build_radial_azimuths",
            "delayed_native_expansion_enabled",
            "is_radial_view",
            "is_tilted_radial_view",
            "is_tilted_view",
            "radial_base_view_name",
            "radial_plane_shape",
            "radial_source_tilted_view",
            "tilted_base_view_name",
            "tilted_frame_center",
            "tilted_stack_axis_length",
            "view_processing_plane_shape",
            "view_uses_inference_processing_grid",
        ),
        "inference": (
            "ModelInputChannelMismatchError",
            "ResidentRingUnitDescriptor",
            "_ResidentGpuPipelineSlot",
            "_cuda_graph_capture_context",
            "_resident_mask_kernels",
            "_split_segmentation_backend_outputs",
            "_warp_matrix_is_identity",
            "resident_trt_cuda_graphs_enabled",
            "resident_trt_native_warp_enabled",
            "resident_trt_ring_enabled",
        ),
        "outputs": (
            "_read_layer_slice_in_output_shape",
        ),
        "runtime": (
            "_acquire_parallel_pool",
            "_release_parallel_pool",
            "allocate_workspace_array",
            "array_nbytes",
            "choose_slice_parallel_workers",
            "close_memmap_array",
            "close_memmap_array_without_flush",
            "flush_array",
            "gpu_worker_aux_interpolation_pool",
            "parallel_for_indices",
            "parallel_for_indices_chunked",
            "parallel_map_unordered",
            "runtime_telemetry",
        ),
        "workspace": (
            "_cpu_count",
            "_env_flag",
            "_env_float",
            "_env_int",
            "proto_hole_treatment_mode",
            "proto_hole_treatment_radius",
            "v1613_d1_backprojection_overlap_enabled",
        ),
    },
)
