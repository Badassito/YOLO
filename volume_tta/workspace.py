"""Implementation subsystem extracted from the v17.0.5 volume TTA runtime.

This physical split intentionally preserves the original numerical and scheduling
behavior. Public coordination contracts live under ``inference_backends``.
"""

from __future__ import annotations

from ._stdlib import *
import numpy as np

def _read_meminfo_bytes() -> Dict[str, int]:
    info: Dict[str, int] = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            raw_val = parts[1].strip().split()[0]
            info[key] = int(raw_val) * 1024
    except Exception:
        pass
    return info

def _cgroup_read_int(path: Path) -> Optional[int]:
    try:
        raw = path.read_text().strip()
    except Exception:
        return None
    if not raw or raw == 'max':
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    # cgroup reports "unlimited" as a huge page-rounded sentinel (~2^63).
    if value <= 0 or value >= (1 << 60):
        return None
    return value

def _cgroup_reclaimable_file_bytes(node: Path, *, v2: bool) -> int:
    """Return clean inactive file-cache bytes reclaimable by this cgroup."""
    if v2:
        keys = ('inactive_file', 'file_dirty', 'file_writeback')
    else:
        keys = ('total_inactive_file', 'total_dirty', 'total_writeback')
    found: Dict[str, int] = {}
    try:
        for line in (node / 'memory.stat').read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in keys:
                try:
                    found[parts[0]] = int(parts[1])
                except Exception:
                    continue
    except Exception:
        return 0
    inactive = int(found.get(keys[0], 0))
    if inactive <= 0:
        return 0
    unreclaimable = int(found.get(keys[1], 0)) + int(found.get(keys[2], 0))
    return max(0, inactive - unreclaimable)

def _cgroup_memory_headroom_bytes() -> Optional[int]:
    """Return the tightest cgroup memory headroom across the process ancestry.

    Usage excludes reclaimable clean file cache; ``None`` means no finite cgroup limit.
    """
    try:
        cgroup_text = Path('/proc/self/cgroup').read_text()
    except Exception:
        return None
    headroom: Optional[int] = None
    for line in cgroup_text.splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        hierarchy_id, controllers, cg_path = parts
        if hierarchy_id == '0' and not controllers:
            # cgroup unified hierarchy.
            is_v2 = True
            base = Path('/sys/fs/cgroup')
            limit_name, usage_name = 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            # cgroup memory controller.
            is_v2 = False
            base = Path('/sys/fs/cgroup/memory')
            limit_name, usage_name = 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        node = base / cg_path.lstrip('/')
        while node == base or base in node.parents:
            limit = _cgroup_read_int(node / limit_name)
            if limit is not None:
                usage = int(_cgroup_read_int(node / usage_name) or 0)
                usage = max(0, usage - _cgroup_reclaimable_file_bytes(node, v2=is_v2))
                level_headroom = max(0, int(limit) - usage)
                headroom = level_headroom if headroom is None else min(headroom, level_headroom)
            if node == base:
                break
            node = node.parent
    return headroom

def available_anon_work_bytes() -> int:
    info = _read_meminfo_bytes()
    mem_avail = int(info.get('MemAvailable', 0))
    swap_free = int(info.get('SwapFree', 0))
    node_avail = max(0, mem_avail + swap_free)
    if _env_flag('YOLO_TTA_IGNORE_CGROUP_MEMORY_LIMIT', False):
        return node_avail
    cgroup_headroom = _cgroup_memory_headroom_bytes()
    if cgroup_headroom is None:
        return node_avail
    # Under a cgroup limit do not count node-wide swap: the cgroup charge is what the OOM killer
    # enforces (cluster cgroups commonly run with memory.swap.max=0 / memsw==mem), and blowing it
    # is an uncatchable SIGKILL that bypasses the MemoryError -> disk-memmap fallback, so err on
    # the tight side.
    return min(node_avail, int(cgroup_headroom))

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in ("", "0", "false", "no", "off")

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)

def v1613_fast_bundle_requested() -> bool:
    """Enable the command-specialized v16.1.3 pipeline unless explicitly disabled."""
    return _env_flag('YOLO_TTA_V1613_FAST_BUNDLE', True)

def v1613_fast_bundle_active() -> bool:
    """True after ``main`` proves the current command satisfies the fast-path contract."""
    return _env_flag('YOLO_TTA_V1613_BUNDLE_ACTIVE', False)

def v1613_d1_pipeline_active() -> bool:
    """Project, infer, treat 2D topology, and backproject inside each eligible GPU lease."""
    # ``main`` publishes the resolved state before spawning CUDA workers.
    if os.environ.get('YOLO_TTA_V1613_D1_PIPELINE_ACTIVE') is not None:
        return bool(
            v1613_fast_bundle_active()
            and _env_flag('YOLO_TTA_V1613_D1_PIPELINE_ACTIVE', False)
        )
    return bool(
        v1613_fast_bundle_active()
        and _env_flag('YOLO_TTA_V1613_D1_OWNER_PIPELINE', True)
    )

def v1613_d1_backprojection_overlap_enabled() -> bool:
    """Allow fallback completed-view backprojection to borrow an idle worker GPU."""
    return bool(
        v1613_fast_bundle_active()
        and not v1613_d1_pipeline_active()
        and _env_flag('YOLO_TTA_V1613_D1_BACKPROJECT_OVERLAP', True)
    )

def proto_hole_treatment_mode() -> str:
    """Resident-proto topology treatment: ``off`` or bounded binary ``closing``."""
    default = 'close' if v1613_fast_bundle_active() else 'off'
    raw = os.environ.get('YOLO_TTA_PROTO_HOLE_TREATMENT', default).strip().lower()
    aliases = {
        '': default, '0': 'off', 'false': 'off', 'none': 'off', 'disabled': 'off',
        '1': 'close', 'true': 'close', 'close': 'close', 'closing': 'close',
    }
    return aliases.get(raw, default)

def proto_hole_treatment_radius() -> int:
    """Square-neighborhood radius used by resident proto closing."""
    default = 2 if v1613_fast_bundle_active() else 0
    return max(0, min(8, _env_int('YOLO_TTA_PROTO_HOLE_RADIUS', default)))

def radial_source_mode() -> str:
    """Source sampling used by resident Radial kernels.

    ``texture_linear`` (default, v16.1.8) samples through the hardware-linear 3D texture:
    measured as fast as the pointer path on the standard command while interpolating all
    three spatial axes. ``nearest_xy_linear_t`` reads the canonical resident uint8 tensor
    directly, avoiding the second full-volume CUDA texture allocation, at the cost of
    nearest-neighbor in-plane (and sagittal/coronal stack-axis) sampling.
    """
    default = 'texture_linear'
    raw = os.environ.get('YOLO_TTA_RADIAL_SOURCE_MODE', default).strip().lower().replace('-', '_')
    aliases = {
        'texture': 'texture_linear', 'texture_linear': 'texture_linear',
        'hardware_linear': 'texture_linear', 'trilinear': 'texture_linear',
        'pointer': 'nearest_xy_linear_t', 'canonical': 'nearest_xy_linear_t',
        'nearest_xy': 'nearest_xy_linear_t', 'nearest_xy_linear_t': 'nearest_xy_linear_t',
    }
    return aliases.get(raw, default)

def tilted_inplane_linear_enabled() -> bool:
    """Bilinear in-plane sampling for Tilted FORWARD-pass inputs (v16.1.8 default).

    Every model-input Tilted renderer (fused CUDA kernel, resident Torch fallback, CPU
    grid renderer) samples the native tilted raster bilinearly when the composed output
    affine is not the identity, exactly as if the native frame were rendered and then
    warped with the Cartesian views' align_corners=False zero-padded bilinear warp. Each
    integer tap keeps its own sheared stack coordinate, so the native raster definition
    is unchanged. Mask backprojection keeps the exact nearest shear scatter, preserving
    the bit-for-bit layer-OR reconstruction contract. Set
    YOLO_TTA_TILTED_INPLANE_LINEAR=0 to restore v16.1.7 nearest-XY forward sampling."""
    return _env_flag('YOLO_TTA_TILTED_INPLANE_LINEAR', True)

_TILTED_IDENTITY_M = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

def _tilted_grid_is_identity(M_grid_to_src: np.ndarray, grid_h: int, grid_w: int, view: object) -> bool:
    """True when the output grid IS the native tilted raster (no in-plane resample)."""
    if int(grid_h) != int(view.src_h) or int(grid_w) != int(view.src_w):
        return False
    matrix = np.asarray(M_grid_to_src, dtype=np.float32).reshape(2, 3)
    return bool(np.allclose(matrix, _TILTED_IDENTITY_M, atol=1e-6))

def _slurm_allocated_cpu_count() -> Optional[int]:
    for env_name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        raw = os.environ.get(env_name, '').strip()
        if not raw:
            continue
        m = re.search(r'(\d+)', raw)
        if m is not None:
            try:
                return max(1, int(m.group(1)))
            except Exception:
                continue
    return None

def _cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, int(len(affinity)))
    except Exception:
        pass

    slurm_cpus = _slurm_allocated_cpu_count()
    if slurm_cpus is not None:
        return max(1, int(slurm_cpus))

    return max(1, int(os.cpu_count() or 1))

