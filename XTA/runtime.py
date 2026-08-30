"""Runtime lifecycle, workspaces, NUMA policy, executors, and observability."""

from __future__ import annotations

import atexit
import contextlib
import functools
import inspect
import json
import math
import mmap
import os
import queue
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import importlib.metadata as importlib_metadata
import multiprocessing as mp
from multiprocessing import reduction as mp_reduction
from collections import Counter
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
)
import numpy as np
from ._deps import cv2, tqdm

from .config import (
    GIB,
    SCRIPT_VERSION,
    SCRIPT_VERSION_COMPACT,
)

# Explicit lower-layer dependencies keep imports one-way.
from .workspace import (
    _cpu_count,
    _env_flag,
    _env_float,
    _env_int,
    available_anon_work_bytes,
)


if TYPE_CHECKING:
    from .geometry import (
        TILTED_VIEW_FAMILY,
        ViewInfo,
        is_radial_view,
        is_tilted_radial_view,
        is_tilted_view,
        radial_base_view_name,
        tilted_base_view_name,
    )
    from .interpolation import interpolate_view_volume_pass_inplace
    from .assembly import view_interpolation_wrap_axis

_NVIDIA_ML_PY_MODULE: Optional[object] = None

_NVIDIA_ML_PY_IMPORT_ERROR: Optional[BaseException] = None

def _load_nvidia_ml_py() -> object:
    """Load NVML bindings from the NVIDIA-maintained ``nvidia-ml-py`` distribution."""
    global _NVIDIA_ML_PY_MODULE, _NVIDIA_ML_PY_IMPORT_ERROR
    if _NVIDIA_ML_PY_MODULE is not None:
        return _NVIDIA_ML_PY_MODULE
    if _NVIDIA_ML_PY_IMPORT_ERROR is not None:
        raise ImportError('nvidia-ml-py is unavailable') from _NVIDIA_ML_PY_IMPORT_ERROR
    try:
        importlib_metadata.version('nvidia-ml-py')
        # The maintained distribution intentionally exports the ``pynvml`` import namespace.
        module = __import__('pynvml')
    except Exception as exc:
        _NVIDIA_ML_PY_IMPORT_ERROR = exc
        raise ImportError(
            'Install the NVIDIA-maintained NVML bindings with: pip install nvidia-ml-py'
        ) from exc
    _NVIDIA_ML_PY_MODULE = module
    return module

def _runtime_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_runtime_jsonable(v) for v in value[:32]]
    if isinstance(value, dict):
        return {str(k): _runtime_jsonable(v) for k, v in list(value.items())[:64]}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.dtype):
        return str(value)
    return repr(value)[:512]

class RuntimeTelemetry:
    """Low-overhead process-local phase and throughput telemetry."""

    def __init__(self) -> None:
        self.enabled = _env_flag('YOLO_TTA_TELEMETRY', True)
        self.lock = threading.RLock()
        self.started_ns = time.monotonic_ns()
        self.phase_ns: Counter[str] = Counter()
        self.phase_calls: Counter[str] = Counter()
        self.counters: Counter[str] = Counter()
        self.gauges: Dict[str, object] = {}
        self.fallbacks: Counter[str] = Counter()
        self._dirty = 0
        self._last_flush = time.monotonic()
        self.flush_seconds = max(2.0, _env_float('YOLO_TTA_TELEMETRY_FLUSH_SECONDS', 15.0))
        requested = os.environ.get('YOLO_TTA_TELEMETRY_PATH', '').strip()
        if requested:
            self.path = Path(requested)
        else:
            base = os.environ.get('SLURM_JOB_ID') or str(os.getpid())
            self.path = Path(tempfile.gettempdir()) / (
                f'gpt56-sol-ultra-v{SCRIPT_VERSION_COMPACT}-'
                f'telemetry-{base}-{os.getpid()}.jsonl'
            )

    @contextlib.contextmanager
    def span(self, name: str, **fields: object) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.monotonic_ns()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = time.monotonic_ns() - start
            with self.lock:
                self.phase_ns[str(name)] += int(elapsed)
                self.phase_calls[str(name)] += 1
                if failed:
                    self.counters[f'{name}.errors'] += 1
                for key, value in fields.items():
                    if isinstance(value, (int, float)):
                        self.counters[f'{name}.{key}'] += value
                self._dirty += 1
            self.maybe_flush()

    def add(self, name: str, value: object = 1) -> None:
        if not self.enabled:
            return
        try:
            amount: object = int(value)  # type: ignore[arg-type]
        except Exception:
            try:
                amount = float(value)  # type: ignore[arg-type]
            except Exception:
                return
        with self.lock:
            self.counters[str(name)] += amount  # type: ignore[operator]
            self._dirty += 1

    def gauge(self, name: str, value: object) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.gauges[str(name)] = _runtime_jsonable(value)
            self._dirty += 1

    def fallback(self, name: str, exc: Optional[BaseException] = None) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.fallbacks[str(name)] += 1
            if exc is not None:
                self.gauges[f'fallback.{name}.last_error'] = f'{type(exc).__name__}: {exc}'[:512]
            self._dirty += 1

    def snapshot(self, *, final: bool = False) -> Dict[str, object]:
        now_ns = time.monotonic_ns()
        with self.lock:
            phases: Dict[str, object] = {}
            for name, total_ns in self.phase_ns.items():
                calls = int(self.phase_calls.get(name, 0))
                seconds = float(total_ns) / 1e9
                phases[str(name)] = {
                    'calls': calls,
                    'seconds': seconds,
                    'mean_ms': (seconds * 1000.0 / calls) if calls else 0.0,
                }
            payload: Dict[str, object] = {
                'schema': f'gpt-5.6-sol-ultra-v{SCRIPT_VERSION}.telemetry.v1',
                'pid': os.getpid(),
                'monotonic_seconds': (now_ns - self.started_ns) / 1e9,
                'wall_time': time.time(),
                'final': bool(final),
                'phases': phases,
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'fallbacks': dict(self.fallbacks),
            }
            self._dirty = 0
        return payload

    def maybe_flush(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        with self.lock:
            due = bool(self._dirty >= 256 or (self._dirty and now - self._last_flush >= self.flush_seconds))
        if due:
            self.flush()

    def flush(self, *, final: bool = False) -> None:
        if not self.enabled:
            return
        payload = self.snapshot(final=bool(final))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
            with self.lock:
                self._last_flush = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.enabled = False
            try:
                print(f'[runtime telemetry disabled] {exc}', file=sys.stderr)
            except Exception:
                pass

class RuntimeSystemSampler:
    def __init__(self, telemetry: RuntimeTelemetry) -> None:
        self.telemetry = telemetry
        self.interval = max(1.0, _env_float('YOLO_TTA_TELEMETRY_SAMPLE_SECONDS', 5.0))
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.last: Optional[Dict[str, object]] = None
        self._nvml_ready = False

    @staticmethod
    def _read_cgroup_memory() -> Tuple[Optional[int], Optional[int]]:
        candidates = (
            (Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')),
            (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'), Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')),
        )
        for limit_path, current_path in candidates:
            try:
                raw = limit_path.read_text().strip()
                limit = None if raw == 'max' else int(raw)
                if limit is not None and int(limit) >= (1 << 60):
                    limit = None
                current = int(current_path.read_text().strip())
                return limit, current
            except Exception:
                continue
        return None, None

    @classmethod
    def _read_proc(cls) -> Dict[str, object]:
        result: Dict[str, object] = {'time': time.monotonic()}
        try:
            fields = Path('/proc/self/stat').read_text().split()
            result['cpu_ticks'] = int(fields[13]) + int(fields[14])
            result['rss_pages'] = int(fields[23])
        except Exception:
            pass
        try:
            for line in Path('/proc/self/io').read_text().splitlines():
                key, value = line.split(':', 1)
                result[key.strip()] = int(value.strip())
        except Exception:
            pass
        limit, current = cls._read_cgroup_memory()
        if current is not None:
            result['cgroup_memory_current'] = int(current)
        if limit is not None:
            result['cgroup_memory_limit'] = int(limit)
        return result

    def _sample_gpu(self) -> None:
        try:
            nvml = _load_nvidia_ml_py()
            if not self._nvml_ready:
                nvml.nvmlInit()
                self._nvml_ready = True
            util = []
            memory = []
            for index in range(int(nvml.nvmlDeviceGetCount())):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                usage = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                util.append({'gpu': int(usage.gpu), 'memory': int(usage.memory)})
                memory.append({'used': int(mem.used), 'total': int(mem.total)})
            self.telemetry.gauge('system.gpu_utilization', util)
            self.telemetry.gauge('system.gpu_memory', memory)
        except Exception:
            pass

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            now = self._read_proc()
            prior = self.last
            self.last = now
            try:
                page_size = int(os.sysconf('SC_PAGE_SIZE'))
            except Exception:
                page_size = 4096
            if 'rss_pages' in now:
                self.telemetry.gauge('system.rss_bytes', int(now['rss_pages']) * page_size)
            if 'cgroup_memory_current' in now:
                self.telemetry.gauge('system.cgroup_memory_current', int(now['cgroup_memory_current']))
            if prior is not None:
                dt = max(1e-9, float(now['time']) - float(prior['time']))
                try:
                    ticks = float(os.sysconf('SC_CLK_TCK'))
                except Exception:
                    ticks = 100.0
                if 'cpu_ticks' in now and 'cpu_ticks' in prior:
                    self.telemetry.gauge(
                        'system.process_cpu_cores',
                        (int(now['cpu_ticks']) - int(prior['cpu_ticks'])) / ticks / dt,
                    )
                for key, metric in (
                    ('read_bytes', 'system.disk_read_bytes_per_s'),
                    ('write_bytes', 'system.disk_write_bytes_per_s'),
                    ('rchar', 'system.logical_read_bytes_per_s'),
                    ('wchar', 'system.logical_write_bytes_per_s'),
                ):
                    if key in now and key in prior:
                        self.telemetry.gauge(metric, max(0, int(now[key]) - int(prior[key])) / dt)
            self._sample_gpu()
            self.telemetry.maybe_flush()

    def start(self) -> None:
        if not self.telemetry.enabled or not _env_flag('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', True):
            return
        self.last = self._read_proc()
        self.thread = threading.Thread(target=self._run, name='runtime-telemetry', daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=min(2.0, self.interval))
        if self._nvml_ready:
            try:
                _load_nvidia_ml_py().nvmlShutdown()
            except Exception:
                pass
            self._nvml_ready = False

class _TeeStream:
    def __init__(self, original: object, file_handle: object) -> None:
        self.original = original
        self.file_handle = file_handle
        self.lock = threading.Lock()

    def write(self, data: str) -> object:
        with self.lock:
            result = self.original.write(data)  # type: ignore[attr-defined]
            self.file_handle.write(data)  # type: ignore[attr-defined]
            return result

    def flush(self) -> None:
        with self.lock:
            self.original.flush()  # type: ignore[attr-defined]
            self.file_handle.flush()  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self.original, name)

_RUNTIME_OBSERVABILITY_LOCK = threading.Lock()

_RUNTIME_TELEMETRY: Optional[RuntimeTelemetry] = None

_RUNTIME_SYSTEM_SAMPLER: Optional[RuntimeSystemSampler] = None

_RUNTIME_STDIO_CAPTURE: Optional[object] = None

_RUNTIME_OBSERVABILITY_ATEXIT_REGISTERED = False

_RUNTIME_OBSERVABILITY_SHUTDOWN = False

_RUNTIME_TELEMETRY_DECORATED_SYMBOLS: List[str] = []

def _install_stdio_capture(telemetry: RuntimeTelemetry) -> Optional[object]:
    path = os.environ.get('YOLO_TTA_CAPTURE_STDIO_PATH', '').strip()
    if not path:
        return None
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open('a', encoding='utf-8', buffering=1)
        sys.stdout = _TeeStream(sys.stdout, handle)  # type: ignore[assignment]
        sys.stderr = _TeeStream(sys.stderr, handle)  # type: ignore[assignment]
        return handle
    except Exception as exc:
        telemetry.fallback('telemetry.stdio_capture', exc)
        return None

def _record_runtime_feature_gauges(telemetry: RuntimeTelemetry) -> None:
    telemetry.gauge('telemetry.decorated_symbols', sorted(_RUNTIME_TELEMETRY_DECORATED_SYMBOLS))
    # These are unconditional packaged contracts. Their owner modules depend on runtime,
    # so importing those modules here would invert the package dependency graph.
    telemetry.gauge('features', {
        'raw_bbox_restored_sparse_members': True,
        'crop_aware_low_quality_mirror': True,
        'slice_aligned_sparse_members': True,
        'owned_nrrd_member_transfer': True,
        'native_projection_callback': True,
        'native_projected_layer_materializer': True,
        'memfd_workspace_compatibility': bool(memfd_workspace_enabled()),
        'native_persistent_trt_ring': True,
    })

def initialize_runtime_observability() -> RuntimeTelemetry:
    """Initialize process-local telemetry explicitly; safe to call repeatedly."""
    global _RUNTIME_TELEMETRY, _RUNTIME_SYSTEM_SAMPLER, _RUNTIME_STDIO_CAPTURE
    global _RUNTIME_OBSERVABILITY_ATEXIT_REGISTERED
    telemetry = _RUNTIME_TELEMETRY
    if telemetry is not None:
        return telemetry
    with _RUNTIME_OBSERVABILITY_LOCK:
        telemetry = _RUNTIME_TELEMETRY
        if telemetry is not None:
            return telemetry
        telemetry = RuntimeTelemetry()
        _RUNTIME_TELEMETRY = telemetry
        _RUNTIME_STDIO_CAPTURE = _install_stdio_capture(telemetry)
        sampler = RuntimeSystemSampler(telemetry)
        _RUNTIME_SYSTEM_SAMPLER = sampler
        sampler.start()
        _record_runtime_feature_gauges(telemetry)
        if not _RUNTIME_OBSERVABILITY_ATEXIT_REGISTERED:
            atexit.register(shutdown_runtime_observability)
            _RUNTIME_OBSERVABILITY_ATEXIT_REGISTERED = True
        return telemetry

def shutdown_runtime_observability() -> None:
    """Stop process-local sampling and emit one final telemetry snapshot exactly once."""
    global _RUNTIME_OBSERVABILITY_SHUTDOWN
    with _RUNTIME_OBSERVABILITY_LOCK:
        if _RUNTIME_OBSERVABILITY_SHUTDOWN:
            return
        _RUNTIME_OBSERVABILITY_SHUTDOWN = True
        sampler = _RUNTIME_SYSTEM_SAMPLER
        telemetry = _RUNTIME_TELEMETRY
        capture = _RUNTIME_STDIO_CAPTURE
    if sampler is not None:
        try:
            sampler.close()
        except Exception:
            pass
    if telemetry is not None:
        try:
            telemetry.flush(final=True)
        except Exception:
            pass
    if capture is not None:
        try:
            capture.flush()
        except Exception:
            pass

def runtime_telemetry() -> RuntimeTelemetry:
    telemetry = _RUNTIME_TELEMETRY
    return telemetry if telemetry is not None else initialize_runtime_observability()

def runtime_telemetry_phase(phase: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorate a core function directly instead of installing a late monkeypatch."""
    def _decorate(original: Callable[..., object]) -> Callable[..., object]:
        qualified = str(getattr(original, '__qualname__', getattr(original, '__name__', phase)))
        _RUNTIME_TELEMETRY_DECORATED_SYMBOLS.append(f'{qualified}:{phase}')
        if inspect.isgeneratorfunction(original):
            @functools.wraps(original)
            def _generator(*args: object, **kwargs: object) -> Iterator[object]:
                with runtime_telemetry().span(str(phase)):
                    yield from original(*args, **kwargs)  # type: ignore[misc]
            return _generator

        @functools.wraps(original)
        def _wrapped(*args: object, **kwargs: object) -> object:
            with runtime_telemetry().span(str(phase)):
                return original(*args, **kwargs)
        return _wrapped
    return _decorate

def numa_enabled() -> bool:
    """Master switch for all NUMA handling (YOLO_TTA_NUMA=0 disables)."""
    return sys.platform.startswith('linux') and _env_flag('YOLO_TTA_NUMA', True)

def numa_worker_pin_enabled() -> bool:
    """Pin each GPU worker's threads to its GPU's NUMA node (=0 disables)."""
    return _env_flag('YOLO_TTA_NUMA_WORKER_PIN', True)

def numa_interleave_enabled() -> bool:
    """Mbind big shared allocations MPOL_INTERLEAVE (=0 disables)."""
    return _env_flag('YOLO_TTA_NUMA_INTERLEAVE', True)

def _parse_id_list(text: str) -> set:
    """Parse a kernel id-list string ('0-3,8,10-11') into a set of ints."""
    out: set = set()
    for part in str(text).strip().split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo_s, _, hi_s = part.partition('-')
            out.update(range(int(lo_s), int(hi_s) + 1))
        else:
            out.add(int(part))
    return out

_NUMA_TOPOLOGY_CACHE: Optional[Tuple[bool, Optional[Dict[str, object]]]] = None

_NUMA_TOPOLOGY_LOCK = threading.Lock()

def numa_topology() -> Optional[Dict[str, object]]:
    """Allocation-aware NUMA topology, or None when undiscoverable.

 Returns {'node_cpus': {node_id: set(ALLOCATED cpus on that node)}, # only non-empty nodes
 'cpu_to_node': {allocated cpu: node_id},
 'allowed_mems': set(node ids from Mems_allowed_list)}.
 Everything is intersected with this process's cpuset, so lopsided SLURM allocations are
 represented as-is. Cached per process (spawn children re-derive their own view)."""
    global _NUMA_TOPOLOGY_CACHE
    cached = _NUMA_TOPOLOGY_CACHE
    if cached is not None:
        return cached[1]
    with _NUMA_TOPOLOGY_LOCK:
        cached = _NUMA_TOPOLOGY_CACHE
        if cached is not None:
            return cached[1]
        topo: Optional[Dict[str, object]] = None
        try:
            if numa_enabled() and hasattr(os, 'sched_getaffinity'):
                allowed_cpus = {int(c) for c in os.sched_getaffinity(0)}
                node_root = '/sys/devices/system/node'
                node_cpus: Dict[int, set] = {}
                cpu_to_node: Dict[int, int] = {}
                for entry in sorted(os.listdir(node_root)):
                    m = re.fullmatch(r'node(\d+)', entry)
                    if m is None:
                        continue
                    node_id = int(m.group(1))
                    try:
                        with open(os.path.join(node_root, entry, 'cpulist'), 'r', encoding='ascii') as fh:
                            cpus = _parse_id_list(fh.read()) & allowed_cpus
                    except OSError:
                        continue
                    if cpus:
                        node_cpus[node_id] = cpus
                        for c in cpus:
                            cpu_to_node[int(c)] = node_id
                allowed_mems: set = set()
                try:
                    with open('/proc/self/status', 'r', encoding='ascii', errors='replace') as fh:
                        for line in fh:
                            if line.startswith('Mems_allowed_list:'):
                                allowed_mems = _parse_id_list(line.split(':', 1)[1])
                                break
                except OSError:
                    pass
                if not allowed_mems:
                    allowed_mems = set(node_cpus.keys())
                if node_cpus:
                    topo = {'node_cpus': node_cpus, 'cpu_to_node': cpu_to_node, 'allowed_mems': allowed_mems}
        except Exception:
            topo = None
        _NUMA_TOPOLOGY_CACHE = (True, topo)
        return topo

def _nvidia_gpu_inventory() -> List[Dict[str, object]]:
    """[{'bdf', 'uuid', 'minor'}] for every GPU the NVIDIA driver exposes via procfs."""
    entries: List[Dict[str, object]] = []
    root = '/proc/driver/nvidia/gpus'
    for name in sorted(os.listdir(root)):
        info_path = os.path.join(root, name, 'information')
        uuid: Optional[str] = None
        minor: Optional[int] = None
        try:
            with open(info_path, 'r', encoding='ascii', errors='replace') as fh:
                for line in fh:
                    if line.startswith('GPU UUID'):
                        uuid = line.split(':', 1)[1].strip()
                    elif line.startswith('Device Minor'):
                        try:
                            minor = int(line.split(':', 1)[1].strip())
                        except ValueError:
                            minor = None
        except OSError:
            continue
        entries.append({'bdf': str(name).lower(), 'uuid': uuid, 'minor': minor})
    return entries

def _numa_node_of_pci_device(bdf: str, topo: Dict[str, object]) -> Optional[int]:
    """NUMA node of a PCI device, restricted to nodes that hold allocated cpus."""
    node_cpus: Dict[int, set] = topo['node_cpus']  # type: ignore[assignment]
    base = f'/sys/bus/pci/devices/{str(bdf).lower()}'
    try:
        with open(base + '/numa_node', 'r', encoding='ascii') as fh:
            node = int(fh.read().strip())
        if node >= 0 and node in node_cpus:
            return node
    except (OSError, ValueError):
        pass
    # numa_node can be -1 (BIOS/kernel hides it); local_cpulist often still knows.
    try:
        with open(base + '/local_cpulist', 'r', encoding='ascii') as fh:
            local_cpus = _parse_id_list(fh.read())
        cpu_to_node: Dict[int, int] = topo['cpu_to_node']  # type: ignore[assignment]
        counts: Dict[int, int] = {}
        for c in local_cpus:
            nd = cpu_to_node.get(int(c))
            if nd is not None:
                counts[nd] = counts.get(nd, 0) + 1
        if counts:
            return max(counts.items(), key=lambda kv: kv[1])[0]
    except (OSError, ValueError):
        pass
    return None

def _resolve_gpu_numa_node(cvd_token: str, topo: Dict[str, object]) -> Optional[int]:
    """Map one CUDA_VISIBLE_DEVICES token (minor index or GPU UUID) to its NUMA node.

 Tokens come from the inherited CUDA_VISIBLE_DEVICES (SLURM sets minor indices or GPU
 UUIDs), i.e. the NVML/driver device domain — NOT torch's logical order — so they match
 the driver procfs 'Device Minor' / 'GPU UUID' fields directly. MIG instance tokens are
 not listed there; those resolve to None (worker stays unpinned)."""
    tok = str(cvd_token).strip()
    if not tok or tok.upper().startswith('MIG-'):
        return None
    try:
        inventory = _nvidia_gpu_inventory()
    except Exception:
        inventory = []
    match: Optional[Dict[str, object]] = None
    if tok.upper().startswith('GPU-'):
        for e in inventory:
            if str(e.get('uuid') or '').lower() == tok.lower():
                match = e
                break
    else:
        try:
            minor = int(tok)
        except ValueError:
            minor = None
        if minor is not None:
            for e in inventory:
                if e.get('minor') == minor:
                    match = e
                    break
    if match is not None:
        node = _numa_node_of_pci_device(str(match['bdf']), topo)
        if node is not None:
            return node
    # Fall back to NVML when driver procfs lacks the GPU mapping.
    try:
        nvml = _load_nvidia_ml_py()
        nvml.nvmlInit()
        try:
            if tok.upper().startswith('GPU-'):
                handle = nvml.nvmlDeviceGetHandleByUUID(tok.encode('ascii'))
            else:
                handle = nvml.nvmlDeviceGetHandleByIndex(int(tok))
            pci = nvml.nvmlDeviceGetPciInfo(handle)
            bus_id = pci.busId.decode('ascii', 'replace') if isinstance(pci.busId, bytes) else str(pci.busId)
            bdf = bus_id.strip().strip('\x00').lower()
            if len(bdf.split(':', 1)[0]) == 8:
                bdf = bdf[4:]
            return _numa_node_of_pci_device(bdf, topo)
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return None

def _physical_core_cpu_groups(cpus: Sequence[int]) -> Optional[List[List[int]]]:
    """Group allocated logical CPUs by Linux ``thread_siblings_list``.

 Each returned group is one physical core (restricted to ``cpus``). Within a group the
 representative hardware thread is first. Missing or asymmetric topology returns None so
 the node stays unpinned; guessing singleton cores could split two SMT siblings across GPUs."""
    allowed = {int(c) for c in cpus}
    if not allowed:
        return []
    siblings_by_cpu: Dict[int, Tuple[int, ...]] = {}
    try:
        for cpu in sorted(allowed):
            with open(
                f'/sys/devices/system/cpu/cpu{int(cpu)}/topology/thread_siblings_list',
                'r', encoding='ascii',
            ) as fh:
                siblings_all = _parse_id_list(fh.read())
            if int(cpu) not in siblings_all:
                return None
            siblings_by_cpu[int(cpu)] = tuple(sorted(int(c) for c in (siblings_all & allowed)))
    except (OSError, ValueError):
        return None

    for cpu, group in siblings_by_cpu.items():
        if not group or int(cpu) not in group:
            return None
        for peer in group:
            if siblings_by_cpu.get(int(peer)) != group:
                return None
    unique = sorted(set(siblings_by_cpu.values()), key=lambda g: (min(g), g))
    claimed: set = set()
    for group in unique:
        if claimed.intersection(group):
            return None
        claimed.update(group)
    if claimed != allowed:
        return None
    return [list(group) for group in unique]

def _flatten_physical_core_groups(core_groups: Sequence[Sequence[int]]) -> List[int]:
    """Representatives first, then SMT siblings, while retaining whole physical cores."""
    representatives = [int(group[0]) for group in core_groups if group]
    smt_siblings = [int(cpu) for group in core_groups if group for cpu in group[1:]]
    return representatives + smt_siblings

def gpu_feeder_reserved_physical_cores() -> int:
    """Exclusive physical cores reserved for each CUDA worker during inference."""
    return max(0, _env_int('YOLO_TTA_GPU_FEEDER_PHYSICAL_CORES', 4))

def plan_gpu_feeder_core_reservations(
    worker_tokens: Sequence[str],
    excluded_cpus: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    """Return disjoint whole-core feeder masks, preferring each GPU's measured NUMA node.

    Every selected physical core includes all allocated SMT siblings.  Reservations are
    process-exclusive during inference: the parent, OpenVINO instances, and other CUDA workers
    exclude these logical CPUs.  When a local node cannot supply the configured target, unused
    allocated cores from another node fill the deficit rather than overlapping reservations.
    """
    n = len(worker_tokens)
    target = int(gpu_feeder_reserved_physical_cores())
    reservations: List[List[List[int]]] = [[] for _ in range(n)]
    if n <= 0 or target <= 0:
        return [[] for _ in range(n)]

    try:
        allowed = {int(cpu) for cpu in os.sched_getaffinity(0)}
    except Exception:
        allowed = set(range(max(1, int(_cpu_count()))))
    allowed.difference_update(int(cpu) for cpu in (excluded_cpus or ()))
    if not allowed:
        print('Warning: [affinity] no allocated CPUs remain for dedicated GPU feeder cores.')
        return [[] for _ in range(n)]

    all_groups = _physical_core_cpu_groups(sorted(allowed))
    exact_topology = all_groups is not None
    if all_groups is None:
        # This fallback preserves exclusivity but cannot prove SMT completeness.  It is used
        # only when Linux thread_siblings_list is unavailable or inconsistent.
        all_groups = [[int(cpu)] for cpu in sorted(allowed)]
        print(
            'Warning: [affinity] physical-core topology is unavailable; dedicated GPU feeder '
            'reservations fall back to disjoint logical CPUs.'
        )

    # Preserve at least one whole physical core for the parent scheduler/result drain.
    # The target SLURM allocation has ample cores, so this guard only changes undersized jobs.
    max_assignable_groups = max(0, int(len(all_groups)) - 1)
    requested_groups = int(n) * int(target)
    if int(max_assignable_groups) < int(requested_groups):
        print(
            'Warning: [affinity] the allocation cannot provide every requested dedicated '
            f'GPU feeder core while retaining one parent core: requested={requested_groups}, '
            f'assignable={max_assignable_groups}.'
        )
    assigned_groups = 0

    topo = numa_topology()
    worker_nodes: List[Optional[int]] = [None] * n
    group_nodes: Dict[Tuple[int, ...], Optional[int]] = {}
    if topo is not None:
        cpu_to_node: Dict[int, int] = topo['cpu_to_node']  # type: ignore[assignment]
        worker_nodes = [_resolve_gpu_numa_node(str(token), topo) for token in worker_tokens]
        for group in all_groups:
            nodes = {cpu_to_node.get(int(cpu)) for cpu in group}
            nodes.discard(None)
            group_nodes[tuple(int(cpu) for cpu in group)] = (
                int(next(iter(nodes))) if len(nodes) == 1 else None
            )
    else:
        for group in all_groups:
            group_nodes[tuple(int(cpu) for cpu in group)] = None

    unused: List[List[int]] = [list(group) for group in all_groups]

    # Fair local allocation: one core per worker per round, then repeat up to the target.
    for node in sorted({int(node) for node in worker_nodes if node is not None}):
        worker_ids = [i for i, value in enumerate(worker_nodes) if value == node]
        for _round in range(target):
            for worker_id in worker_ids:
                if (
                    len(reservations[worker_id]) >= target
                    or int(assigned_groups) >= int(max_assignable_groups)
                ):
                    continue
                selected_pos = next(
                    (
                        pos for pos, group in enumerate(unused)
                        if group_nodes.get(tuple(int(cpu) for cpu in group)) == int(node)
                    ),
                    None,
                )
                if selected_pos is None:
                    break
                reservations[worker_id].append(unused.pop(int(selected_pos)))
                assigned_groups += 1

    # Resolve unmapped GPUs and local shortages from the remaining globally disjoint cores.
    for _round in range(target):
        for worker_id in range(n):
            if (
                len(reservations[worker_id]) >= target
                or not unused
                or int(assigned_groups) >= int(max_assignable_groups)
            ):
                continue
            reservations[worker_id].append(unused.pop(0))
            assigned_groups += 1

    flattened: List[List[int]] = []
    for worker_id, groups in enumerate(reservations):
        cpus = _flatten_physical_core_groups(groups)
        flattened.append(cpus)
        locality = (
            f'node {worker_nodes[worker_id]}'
            if worker_nodes[worker_id] is not None else 'NUMA node unresolved'
        )
        if len(groups) < target:
            print(
                f'Warning: [affinity] gpu-worker {worker_id} (token {worker_tokens[worker_id]}) '
                f'received {len(groups)}/{target} dedicated physical core(s); allocation exhausted.'
            )
        print(
            f'[affinity] gpu-worker {worker_id} (token {worker_tokens[worker_id]}): '
            f'{len(groups)} dedicated {"physical" if exact_topology else "logical"} core(s), '
            f'{len(cpus)} logical CPU(s), preferred {locality}; cpu_mask={cpus}'
        )
    return flattened

def plan_gpu_worker_affinity(
    worker_tokens: Sequence[str],
    excluded_cpus: Optional[Sequence[int]] = None,
    reserved_cpus_by_worker: Optional[Sequence[Sequence[int]]] = None,
) -> List[Optional[List[int]]]:
    """Per-worker CPU masks with exclusive feeder cores plus topology-local helpers.

    Dedicated feeder CPUs are removed from every shared/helper pool and added back only to
    their owning worker.  With NUMA pinning active, the remaining physical cores on a GPU's
    measured node are split among that node's workers.  Without NUMA discovery, workers share
    the non-reserved allocation while retaining mutually exclusive feeder masks.
    """
    n = len(worker_tokens)
    plan: List[Optional[List[int]]] = [None] * n
    if n == 0:
        return plan
    reserved: List[List[int]] = [
        [int(cpu) for cpu in (reserved_cpus_by_worker[i] if reserved_cpus_by_worker and i < len(reserved_cpus_by_worker) else ())]
        for i in range(n)
    ]
    all_reserved = {int(cpu) for values in reserved for cpu in values}
    excluded = {int(cpu) for cpu in (excluded_cpus or ())} | all_reserved

    try:
        allowed_all = {int(cpu) for cpu in os.sched_getaffinity(0)}
    except Exception:
        allowed_all = set(range(max(1, int(_cpu_count()))))
    shared_allowed = sorted(int(cpu) for cpu in (allowed_all - excluded))

    topo = numa_topology() if (numa_enabled() and numa_worker_pin_enabled()) else None
    if topo is None:
        if numa_enabled() and numa_worker_pin_enabled():
            print('[numa] topology-local helper pinning inactive: NUMA topology not discoverable')
        for i in range(n):
            combined = list(dict.fromkeys([*reserved[i], *shared_allowed]))
            plan[i] = combined or None
            print(
                f'[affinity] gpu-worker {i} (token {worker_tokens[i]}): '
                f'dedicated={len(reserved[i])} logical CPU(s), shared helpers={len(shared_allowed)}; '
                f'NUMA-local split unavailable.'
            )
        return plan

    node_cpus_raw: Dict[int, set] = topo['node_cpus']  # type: ignore[assignment]
    node_cpus: Dict[int, set] = {
        int(node): {int(cpu) for cpu in cpus if int(cpu) not in excluded}
        for node, cpus in node_cpus_raw.items()
    }
    node_of: List[Optional[int]] = []
    helper_groups_by_worker: List[List[List[int]]] = [[] for _ in range(n)]
    by_node: Dict[int, List[int]] = {}
    for i, tok in enumerate(worker_tokens):
        node = _resolve_gpu_numa_node(str(tok), topo)
        node_of.append(node)
        if node is not None and node in node_cpus:
            by_node.setdefault(int(node), []).append(i)

    min_cores = max(1, _env_int('YOLO_TTA_NUMA_WORKER_MIN_CORES', 4))
    for node, idxs in sorted(by_node.items()):
        core_groups = _physical_core_cpu_groups(sorted(node_cpus[int(node)]))
        if core_groups is None:
            print(
                f'Warning: [numa] node {int(node)} thread_siblings_list topology is unavailable '
                'or inconsistent; its workers receive the shared non-reserved helper pool.'
            )
            continue
        physical_share = len(core_groups) // len(idxs)
        if physical_share < min_cores:
            for i in idxs:
                helper_groups_by_worker[i] = [list(group) for group in core_groups]
            continue
        q, r = divmod(len(core_groups), len(idxs))
        pos = 0
        for j, i in enumerate(idxs):
            count = int(q + (1 if j < r else 0))
            helper_groups_by_worker[i] = [list(group) for group in core_groups[pos:pos + count]]
            pos += count

    for i, tok in enumerate(worker_tokens):
        helpers = _flatten_physical_core_groups(helper_groups_by_worker[i])
        if not helpers:
            # A resolved node can still have too few remaining cores for a clean split, or
            # thread_siblings discovery can fail after feeder removal. Fall back to that
            # node's non-reserved CPUs, then to the global non-reserved pool. No worker can
            # enter another GPU's dedicated feeder set.
            node_local = (
                sorted(int(cpu) for cpu in node_cpus.get(int(node_of[i]), set()))
                if node_of[i] is not None else []
            )
            helpers = node_local or list(shared_allowed)
        combined = list(dict.fromkeys([*reserved[i], *helpers]))
        plan[i] = combined or None
        print(
            f'[affinity] gpu-worker {i} (token {tok}): node={node_of[i]}, '
            f'dedicated={len(reserved[i])} logical CPU(s), helper={len(helpers)} logical CPU(s), '
            f'total_mask={len(combined)}.'
        )
    return plan

@dataclass(frozen=True)
class CpuInferenceInstancePlan:
    """One socket-local OpenVINO process and its complete logical CPU mask."""

    instance_id: int
    numa_nodes: Tuple[int, ...]
    cpus: Tuple[int, ...]
    physical_cores: int
    inference_threads: int

def _distribute_integer_budget(total: int, capacities: Sequence[int]) -> List[int]:
    caps = [max(0, int(value)) for value in capacities]
    if not caps or int(total) <= 0:
        return [0 for _ in caps]
    target = min(int(total), int(sum(caps)))
    allocated = [0 for _ in caps]
    positive = [index for index, cap in enumerate(caps) if cap > 0]
    for index in positive:
        if target <= 0:
            break
        allocated[index] = 1
        target -= 1
    while target > 0:
        candidates = [
            index for index, cap in enumerate(caps)
            if allocated[index] < cap
        ]
        if not candidates:
            break
        index = max(
            candidates,
            key=lambda item: (
                float(caps[item]) / float(max(1, allocated[item])),
                caps[item] - allocated[item],
                -item,
            ),
        )
        allocated[index] += 1
        target -= 1
    return allocated

def plan_openvino_cpu_instances(
    requested_instances: Optional[int],
    requested_threads: Optional[int],
    excluded_cpus: Optional[Sequence[int]] = None,
) -> List[CpuInferenceInstancePlan]:
    """Build socket-local CPU process masks for a non-SNC topology.

    ``auto`` reserves two physical cores per populated socket for the parent, GPU feeders,
    decode, and output. An explicit --cpu_threads value is a whole-job logical-thread budget
    and may consume that reserve.
    """
    excluded = {int(cpu) for cpu in (excluded_cpus or ())}
    try:
        allowed_cpus = sorted(
            int(cpu) for cpu in os.sched_getaffinity(0) if int(cpu) not in excluded
        )
    except Exception:
        allowed_cpus = [
            int(cpu) for cpu in range(max(1, int(_cpu_count()))) if int(cpu) not in excluded
        ]
    if not allowed_cpus:
        raise RuntimeError('No CPUs remain for OpenVINO after dedicated GPU feeder reservations')
    topo = numa_topology()
    node_entries: List[Tuple[int, List[List[int]]]] = []
    if topo is not None:
        node_map: Dict[int, set] = topo['node_cpus']  # type: ignore[assignment]
        for node, cpus in sorted(node_map.items()):
            eligible_cpus = sorted(int(cpu) for cpu in cpus if int(cpu) not in excluded)
            groups = _physical_core_cpu_groups(eligible_cpus)
            if groups:
                node_entries.append((int(node), groups))
    if not node_entries:
        groups = _physical_core_cpu_groups(allowed_cpus)
        if not groups:
            groups = [[int(cpu)] for cpu in allowed_cpus]
        node_entries = [(-1, groups)]

    instance_count = int(requested_instances or len(node_entries))
    instance_count = max(1, min(instance_count, sum(len(groups) for _node, groups in node_entries)))
    grouped_instances: List[Tuple[Tuple[int, ...], List[List[int]]]] = []
    if instance_count == 1:
        grouped_instances = [(
            tuple(int(node) for node, _groups in node_entries),
            [list(group) for _node, groups in node_entries for group in groups],
        )]
    elif instance_count == len(node_entries):
        grouped_instances = [
            ((int(node),), [list(group) for group in groups])
            for node, groups in node_entries
        ]
    else:
        all_groups: List[Tuple[int, List[int]]] = [
            (int(node), list(group))
            for node, groups in node_entries
            for group in groups
        ]
        q, r = divmod(len(all_groups), int(instance_count))
        position = 0
        for instance_id in range(int(instance_count)):
            count = int(q + (1 if instance_id < r else 0))
            subset = all_groups[position:position + count]
            position += count
            grouped_instances.append((
                tuple(dict.fromkeys(int(node) for node, _group in subset)),
                [list(group) for _node, group in subset],
            ))

    reserve_physical = max(0, _env_int('YOLO_TTA_CPU_SOCKET_RESERVE_CORES', 2))
    eligible_groups: List[Tuple[Tuple[int, ...], List[List[int]]]] = []
    for nodes, groups in grouped_instances:
        kept = list(groups)
        if requested_threads is None and len(kept) > reserve_physical:
            kept = kept[:len(kept) - reserve_physical]
        if not kept:
            kept = list(groups[:1])
        eligible_groups.append((nodes, kept))

    capacities = [len(_flatten_physical_core_groups(groups)) for _nodes, groups in eligible_groups]
    if requested_threads is None:
        quotas = list(capacities)
    else:
        quotas = _distribute_integer_budget(int(requested_threads), capacities)
    plans: List[CpuInferenceInstancePlan] = []
    for instance_id, ((nodes, groups), quota) in enumerate(zip(eligible_groups, quotas)):
        ordered = _flatten_physical_core_groups(groups)
        selected = ordered[:max(1, min(len(ordered), int(quota or 1)))]
        selected_set = set(selected)
        physical_count = sum(1 for group in groups if selected_set.intersection(group))
        plans.append(CpuInferenceInstancePlan(
            instance_id=int(instance_id),
            numa_nodes=tuple(int(node) for node in nodes),
            cpus=tuple(int(cpu) for cpu in selected),
            physical_cores=int(physical_count),
            inference_threads=int(len(selected)),
        ))
    return plans

def _sched_setaffinity_all_threads(cpus: Sequence[int]) -> bool:
    """Apply a CPU mask to EVERY thread of this process (Linux affinity is per-thread;
 os.sched_setaffinity(0,...) alone would only move the calling thread)."""
    if not hasattr(os, 'sched_setaffinity'):
        return False
    cpu_set = {int(c) for c in cpus}
    if not cpu_set:
        return False
    try:
        tids = sorted(int(t) for t in os.listdir('/proc/self/task'))
    except (OSError, ValueError):
        tids = [0]  # 0 == calling thread
    ok = False
    for tid in tids:
        try:
            os.sched_setaffinity(tid, cpu_set)
            ok = True
        except OSError:
            continue  # thread exited between listdir and the call
    return ok

_NUMA_MBIND_STATE = {'failed': False, 'announced': False}

def _numa_interleave_range(addr: int, nbytes: int, mems: Sequence[int]) -> bool:
    """Mbind(start, len, MPOL_INTERLEAVE, nodemask, maxnode, 0) via the raw syscall (no libnuma)."""
    if _NUMA_MBIND_STATE['failed']:
        return False
    try:
        machine = str(os.uname().machine)
    except Exception:
        _NUMA_MBIND_STATE['failed'] = True
        return False
    syscall_nr = {'x86_64': 237, 'aarch64': 235}.get(machine)
    nodes = sorted({int(m) for m in mems if int(m) >= 0})
    max_node = nodes[-1] if nodes else -1
    if syscall_nr is None or max_node < 0:
        _NUMA_MBIND_STATE['failed'] = True
        print(f'Warning: [numa] mbind unsupported here (machine={machine}, max node={max_node}); '
              f'page interleave disabled for this process.')
        return False
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        page = int(mmap.PAGESIZE)
        start = int(addr) & ~(page - 1)
        length = int(nbytes) + (int(addr) - start)
        # Linux's user ABI names this argument maxnode, but get_nodes decrements it before
        # copying the bitmap. Therefore highest node id N needs syscall maxnode=N+2. Size
        # the userspace object from that exact argument: maxnode=65 needs two 64-bit words,
        # never the single c_ulong used by the old code.
        bits_per_word = int(ctypes.sizeof(ctypes.c_ulong) * 8)
        maxnode_arg = int(max_node) + 2
        word_count = (maxnode_arg + bits_per_word - 1) // bits_per_word
        nodemask = (ctypes.c_ulong * int(word_count))()
        for node in nodes:
            word_index, bit_index = divmod(int(node), bits_per_word)
            nodemask[word_index] |= 1 << bit_index
        libc.syscall.restype = ctypes.c_long
        rc = libc.syscall(
            int(syscall_nr),
            ctypes.c_void_p(start),
            ctypes.c_size_t(length),
            ctypes.c_int(3),  # MPOL_INTERLEAVE
            ctypes.cast(nodemask, ctypes.POINTER(ctypes.c_ulong)),
            ctypes.c_ulong(maxnode_arg),
            ctypes.c_uint(0),
        )
        if int(rc) != 0:
            err = int(ctypes.get_errno())
            _NUMA_MBIND_STATE['failed'] = True
            print(f'Warning: [numa] mbind(MPOL_INTERLEAVE) failed (errno {err}); '
                  f'page interleave disabled for this process.')
            return False
        return True
    except Exception as exc:
        _NUMA_MBIND_STATE['failed'] = True
        print(f'Warning: [numa] mbind unavailable ({exc}); page interleave disabled for this process.')
        return False

def numa_interleave_memory(arr: object, desc: str = '') -> bool:
    """Round-robin a big shared buffer's pages across the allowed NUMA nodes.

 Called by the ALLOCATION paths right after creating a buffer and before first touch — the
 only moment placement can still be chosen (unprivileged MPOL_MF_MOVE cannot migrate shared
 multi-process pages later). Keyed on how the buffer was allocated, never on a mount point:
 anonymous RAM and tmpfs/shm mappings both take the policy; regular on-disk file mappings
 are silently unaffected. No-ops (returning False) when: NUMA handling is off, topology is
 unknown, only one memory node is allowed, the buffer is below
 YOLO_TTA_NUMA_INTERLEAVE_MIN_MIB, or this process's current affinity sits inside a single
 node (a pinned GPU worker's first-touch is already the better, local placement)."""
    if arr is None or not (numa_enabled() and numa_interleave_enabled()):
        return False
    topo = numa_topology()
    if topo is None:
        return False
    mems = sorted(int(m) for m in topo['allowed_mems'])  # type: ignore[call-overload]
    if len(mems) < 2:
        return False
    try:
        a = np.asarray(arr)
        if a.nbytes <= 0 or not a.flags['C_CONTIGUOUS']:
            return False
        min_bytes = max(0, _env_int('YOLO_TTA_NUMA_INTERLEAVE_MIN_MIB', 64)) * 1024 * 1024
        if int(a.nbytes) < int(min_bytes):
            return False
        if hasattr(os, 'sched_getaffinity'):
            cpu_to_node: Dict[int, int] = topo['cpu_to_node']  # type: ignore[assignment]
            nodes_now = {cpu_to_node.get(int(c)) for c in os.sched_getaffinity(0)}
            nodes_now.discard(None)
            if len(nodes_now) < 2:
                return False
        addr = int(a.ctypes.data)
        if addr == 0:
            return False
        ok = _numa_interleave_range(addr, int(a.nbytes), mems)
        if ok and not _NUMA_MBIND_STATE['announced']:
            _NUMA_MBIND_STATE['announced'] = True
            print(f'[numa] interleaving big shared allocations across nodes {mems} '
                  f'(first: {desc or "unnamed"}, {a.nbytes / GIB:.1f} GiB; YOLO_TTA_NUMA_INTERLEAVE=0 disables)')
        return ok
    except Exception:
        return False

def default_worker_budget() -> int:
    """Return the whole-job CPU concurrency target for mixed CPU/GPU/IO work.

    The main process subtracts per-GPU worker reservations while CUDA workers are active.
    """
    return max(1, int(_cpu_count()) * 2)

def gpu_worker_cpu_share(gpu_device_count: int) -> int:
    """Logical CPU share reserved for each CUDA inference worker process."""
    cpu = max(1, int(_cpu_count()))
    n = max(1, int(gpu_device_count))
    return max(1, cpu // n)

def main_process_worker_budget(gpu_device_count: int, gpu_worker_process_active: bool) -> int:
    """Main-process CPU budget while one-or-more CUDA worker processes are active."""
    total = int(default_worker_budget())
    if not bool(gpu_worker_process_active):
        return max(1, total)
    n = max(1, int(gpu_device_count))
    reserved = int(gpu_worker_cpu_share(n)) * n
    return max(1, total - reserved)

def tail_worker_budget_expansion_enabled() -> bool:
    """Reclaim the per-GPU CPU reservation after every CUDA worker has drained.

 Set YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND=0 to retain the inference-phase main-process
 worker budget during strictly tail-only stages."""
    return _env_flag('YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND', True)

def gpu_worker_direct_union_enabled() -> bool:
    """Allow disjoint angle-variant worker leases to write directly into that variant union."""
    return _env_flag('YOLO_TTA_GPU_WORKER_DIRECT_UNION', True)

def gpu_worker_fullframe_task_ranges(
    n_slices: int,
    slice_chunk: int,
) -> List[Tuple[int, int]]:
    """Return the profiled steady-state full-frame leases without tail adaptation."""
    total = max(1, int(n_slices))
    chunk = max(1, int(slice_chunk))
    return [
        (int(start), min(int(total), int(start) + int(chunk)))
        for start in range(0, int(total), int(chunk))
    ]

def gpu_worker_tail_split_point(
    slice_start: int,
    slice_count: int,
    inference_batch: int,
) -> Optional[int]:
    """Batch-aligned midpoint for a dispatch-time tail lease split."""
    if not _env_flag('YOLO_TTA_GPU_WORKER_ADAPTIVE_TAIL_LEASES', True):
        return None
    tail0 = int(slice_start)
    tail1 = int(slice_start) + max(0, int(slice_count))
    align = max(1, int(inference_batch))
    min_tail = max(align, _env_int('YOLO_TTA_GPU_WORKER_TAIL_MIN_SLICES', 128))
    midpoint = int(tail0) + ((int(tail1 - tail0) // 2) // int(align)) * int(align)
    if int(midpoint - tail0) >= int(min_tail) and int(tail1 - midpoint) >= int(min_tail):
        return int(midpoint)
    return None

def gpu_worker_target_lease_seconds() -> float:
    """Target render+TensorRT+resident-post duration for one full-frame lease."""
    value = _env_float('YOLO_TTA_GPU_WORKER_LEASE_TARGET_SECONDS', 2.0)
    return max(0.25, min(30.0, float(value)))

def gpu_worker_min_lease_slices() -> int:
    return max(1, _env_int('YOLO_TTA_GPU_WORKER_MIN_LEASE_SLICES', 32))

def gpu_worker_max_lease_slices() -> int:
    return max(gpu_worker_min_lease_slices(), _env_int('YOLO_TTA_GPU_WORKER_MAX_LEASE_SLICES', 128))

def gpu_worker_default_seconds_per_frame(view: 'ViewInfo') -> float:
    """Cold-start cost prior used until measured worker telemetry is available."""
    # Local import keeps the package dependency graph acyclic.
    from .geometry import (
        is_radial_view,
        is_tilted_radial_view,
        is_tilted_view,
    )

    if is_tilted_radial_view(view):
        default = 0.060
        env = 'YOLO_TTA_GPU_WORKER_DEFAULT_SEC_PER_FRAME_TILTED_RADIAL'
    elif is_radial_view(view):
        default = 0.050
        env = 'YOLO_TTA_GPU_WORKER_DEFAULT_SEC_PER_FRAME_RADIAL'
    elif is_tilted_view(view):
        default = 0.045
        env = 'YOLO_TTA_GPU_WORKER_DEFAULT_SEC_PER_FRAME_TILTED'
    else:
        default = 0.035
        env = 'YOLO_TTA_GPU_WORKER_DEFAULT_SEC_PER_FRAME_CARTESIAN'
    return max(1e-4, _env_float(env, default))

def gpu_worker_initial_lease_slices(view: 'ViewInfo', batch: int = 1) -> int:
    align = max(1, int(batch))
    estimate = int(round(gpu_worker_target_lease_seconds() / gpu_worker_default_seconds_per_frame(view)))
    estimate = max(gpu_worker_min_lease_slices(), min(gpu_worker_max_lease_slices(), estimate))
    estimate = max(align, (int(estimate) // align) * align)
    return int(estimate)

def gpu_worker_task_cost_key(task: Dict[str, object]) -> Tuple[object, ...]:
    # Local import keeps the package dependency graph acyclic.
    from .geometry import (
        ViewInfo,
        is_radial_view,
        radial_base_view_name,
        tilted_base_view_name,
    )

    view = task.get('view')
    if not isinstance(view, ViewInfo):
        return (str(task.get('kind', 'unknown')),)
    return (
        str(task.get('kind', 'unknown')),
        str(task.get('result_mode', 'file')),
        str(view.family),
        str(radial_base_view_name(view) if is_radial_view(view) else tilted_base_view_name(view)),
        int(task.get('out_size', 0)),
        int(getattr(view, 'src_h', 0)),
        int(getattr(view, 'src_w', 0)),
    )

def cpu_inference_supports_view(view: object) -> bool:
    """OpenVINO owns Cartesian and Tilted Cartesian work, never Radial work."""
    # Local import keeps the package dependency graph acyclic.
    from .geometry import (
        TILTED_VIEW_FAMILY,
        ViewInfo,
        is_radial_view,
    )

    return bool(
        isinstance(view, ViewInfo)
        and not is_radial_view(view)
        and str(view.family) in {'orthogonal', TILTED_VIEW_FAMILY}
    )

def cpu_inference_task_priority(task: Dict[str, object]) -> int:
    """Cartesian first (right-angle TTA first), then Tilted Cartesian."""
    # Local import keeps the package dependency graph acyclic.
    from .geometry import ViewInfo

    view = task.get('view')
    if not cpu_inference_supports_view(view):
        return 100
    assert isinstance(view, ViewInfo)
    tile_penalty = 1 if str(task.get('kind', '')) == 'tile' else 0
    if str(view.family) == 'orthogonal':
        angle = float(getattr(view, 'tta_angle_deg', 0.0)) % 90.0
        right_angle = bool(
            math.isclose(angle, 0.0, rel_tol=0.0, abs_tol=1e-7)
            or math.isclose(angle, 90.0, rel_tol=0.0, abs_tol=1e-7)
        )
        return int((0 if right_angle else 2) + tile_penalty)
    return int(4 + tile_penalty)

def cpu_worker_target_lease_seconds() -> float:
    # v17.0.3 retains the measured 10-second lease target from v17.0.2: it cut the bounded task window
    # from 8,083 to 3,447 and improved wall time from 19:31 to 17:51. Use that
    # demonstrated target by default; claim-time splitting still prevents a backend
    # from being forced to consume an arbitrarily oversized seed range.
    return max(0.5, min(30.0, _env_float('YOLO_TTA_CPU_WORKER_LEASE_TARGET_SECONDS', 10.0)))

def cpu_worker_min_lease_slices() -> int:
    return max(1, _env_int('YOLO_TTA_CPU_WORKER_MIN_LEASE_SLICES', 4))

def cpu_worker_max_lease_slices() -> int:
    return max(cpu_worker_min_lease_slices(), _env_int('YOLO_TTA_CPU_WORKER_MAX_LEASE_SLICES', 64))

def cpu_worker_default_seconds_per_frame(view: 'ViewInfo') -> float:
    default = 0.20 if str(view.family) == 'orthogonal' else 0.30
    env = (
        'YOLO_TTA_CPU_WORKER_DEFAULT_SEC_PER_FRAME_CARTESIAN'
        if str(view.family) == 'orthogonal' else
        'YOLO_TTA_CPU_WORKER_DEFAULT_SEC_PER_FRAME_TILTED'
    )
    return max(1e-4, _env_float(env, default))

def cpu_worker_initial_lease_slices(view: 'ViewInfo', batch: int = 1) -> int:
    align = max(1, int(batch))
    estimate = int(round(cpu_worker_target_lease_seconds() / cpu_worker_default_seconds_per_frame(view)))
    estimate = max(cpu_worker_min_lease_slices(), min(cpu_worker_max_lease_slices(), estimate))
    estimate = max(align, int(math.ceil(float(estimate) / float(align))) * align)
    return int(estimate)

HYBRID_DEFERRED_RESULT_MODE = 'hybrid_unclaimed'

def hybrid_cpu_reserved_view_count() -> int:
    """Number of ordered full-frame views reserved for sequential OpenVINO ownership.

    The default reserves the first transverse/sagittal/coronal TTA set. Remaining eligible
    views stay immediately available to CUDA D1. Set the value to zero to disable hybrid CPU
    full-frame reservation without disabling the OpenVINO backend itself.
    """
    return max(0, min(64, _env_int('YOLO_TTA_HYBRID_CPU_RESERVED_VIEW_COUNT', 3)))

def hybrid_gpu_stealback_min_cpu_samples() -> int:
    """Completed active-view OpenVINO leases required before early ETA assistance.

    Mandatory GPU work continues normally during this warmup. Once that work is exhausted,
    CUDA may assist immediately rather than idle, even if the sample floor was not reached.
    """
    return max(0, min(32, _env_int('YOLO_TTA_HYBRID_GPU_STEALBACK_MIN_CPU_SAMPLES', 2)))

def hybrid_gpu_stealback_enabled() -> bool:
    """Allow CUDA to assist the active CPU-owned view before mandatory GPU work drains."""
    return _env_flag('YOLO_TTA_HYBRID_GPU_STEALBACK', True)

def hybrid_gpu_stealback_eta_ratio() -> float:
    """Active CPU-view ETA must exceed this multiple of mandatory-GPU ETA."""
    return max(0.25, min(4.0, _env_float('YOLO_TTA_HYBRID_GPU_STEALBACK_ETA_RATIO', 1.0)))

def hybrid_gpu_stealback_min_lead_seconds() -> float:
    """Absolute CPU-over-GPU ETA lead required before early stealback."""
    return max(0.0, min(300.0, _env_float('YOLO_TTA_HYBRID_GPU_STEALBACK_MIN_LEAD_SECONDS', 5.0)))

def hybrid_gpu_stealback_max_fraction() -> float:
    """Maximum fraction of CUDA workers borrowed while mandatory GPU work remains."""
    return max(0.0, min(1.0, _env_float('YOLO_TTA_HYBRID_GPU_STEALBACK_MAX_FRACTION', 0.75)))

def hybrid_cpu_affinity_overlap_enabled() -> bool:
    """Share OpenVINO CPU masks with low-duty parent/CUDA helper threads in hybrid runs.

    GPU-only profiling for the target workload showed roughly five percent aggregate CPU duty.
    The v17.0.1 exclusive-mask policy nevertheless reduced each CUDA worker from roughly twenty
    helper threads to one and the parent pools from 160 workers to 16.  Keeping OpenVINO pinned
    socket-locally while allowing helper overlap preserves locality without assuming any GPU is
    attached to a particular NUMA node.  Set YOLO_TTA_HYBRID_CPU_AFFINITY_OVERLAP=0 to restore
    exclusive parent/GPU-helper masks.
    """
    return _env_flag('YOLO_TTA_HYBRID_CPU_AFFINITY_OVERLAP', True)

INTERPOLATION_PROCESS_WORKER_DEFAULT_CAP = 1

def _estimate_parent_view_postprocess_bytes(
    view: 'ViewInfo',
    *,
    nrrd_layers_enabled: bool,
    interpolation_enabled: bool,
) -> int:
    """Conservative live working-set estimate for one outer parent-view task.

 The baseline union already exists before admission, so this estimate covers the transient
 cleanup/projection/cvol buffers and the bounded slice bands used by the task. Interpolation
 has separate workspace admission, but reserving several GiB here prevents the outer pool
 from opening too many large view preparations while child passes are live."""
    plane_bytes = max(1, int(getattr(view, 'src_h', 1))) * max(1, int(getattr(view, 'src_w', 1)))
    slices = max(1, int(getattr(view, 'num_slices', 1)))
    view_bytes = int(plane_bytes) * int(slices)
    name = str(getattr(view, 'name', '')).lower()
    family = str(getattr(view, 'family', '')).lower()
    # Cleanup/cvol encoding is slice-bounded for Cartesian/Radial views.  Tilted
    # projection can additionally own a source-geometry output; tilted-Radial can
    # own both its reconstructed base stack and projected destination concurrently.
    estimate = max(1 * GIB, min(4 * GIB, int(view_bytes // 8) + 512 * 1024 * 1024))
    if 'tilted' in name and family == 'radial':
        estimate = max(int(estimate), int(view_bytes) * 2 + 4 * GIB)
    elif 'tilted' in name:
        estimate = max(int(estimate), int(view_bytes) + 4 * GIB)
    if bool(interpolation_enabled):
        estimate = max(int(estimate), int(view_bytes) * 2 + 4 * GIB)
    return max(512 * 1024 * 1024, min(96 * GIB, int(estimate)))

def resolve_parent_postprocess_worker_allocation(
    worker_budget: int,
    views: Sequence['ViewInfo'],
    *,
    nrrd_layers_enabled: bool,
    interpolation_enabled: bool,
) -> Tuple[int, int, int, int, int]:
    """Resolve independent parent-view and per-view slice concurrency.

    Defaults admit up to four views within CPU and anonymous-memory limits, then divide
    the worker budget across their nested slice pools. Interpolation enablement affects
    only the per-view memory estimate.
    """
    budget = max(1, int(worker_budget))
    view_list = list(views)
    view_count = max(1, len(view_list))
    per_view_bytes = max(
        [_estimate_parent_view_postprocess_bytes(
            view,
            nrrd_layers_enabled=bool(nrrd_layers_enabled),
            interpolation_enabled=bool(interpolation_enabled),
        ) for view in view_list]
        or [512 * 1024 * 1024]
    )
    reserve = int(max(0.0, _env_float('YOLO_TTA_PARENT_POSTPROCESS_RESERVE_GIB', 16.0)) * GIB)
    usable = max(0, int(available_anon_work_bytes()) - int(reserve))
    memory_cap = max(1, min(int(view_count), int(usable // max(1, int(per_view_bytes)))))
    cpu_cap = max(1, min(4, int(view_count), max(1, int(budget) // 16)))
    default_outer = max(1, min(int(view_count), int(cpu_cap), int(memory_cap)))
    requested_outer = max(1, _env_int('YOLO_TTA_PARENT_POSTPROCESS_WORKERS', int(default_outer)))
    outer_workers = max(1, min(int(view_count), int(budget), int(requested_outer)))
    default_slice_workers = max(1, int(budget) // max(1, int(outer_workers)))
    slice_workers = max(
        1,
        min(
            int(budget),
            _env_int('YOLO_TTA_PARENT_SLICE_WORKERS', int(default_slice_workers)),
        ),
    )
    return (
        int(outer_workers),
        int(slice_workers),
        int(per_view_bytes),
        int(memory_cap),
        int(default_outer),
    )

def resolve_parent_interpolation_worker_allocation(
    worker_budget: int,
    parent_postprocess_workers: int,
    *,
    interpolation_process_backend_active: bool,
) -> Tuple[int, int, int]:
    """Resolve per-parent interpolation workers from actual live overlap.

 The process-backend state is supplied explicitly for the current command. Merely having
 the backend installed/configured no longer divides worker budgets when ``--interpolation_distance 0``
 leaves no interpolation task capable of running."""
    budget = max(1, int(worker_budget))
    parent_workers = max(1, int(parent_postprocess_workers))
    default_overlap = max(
        1,
        min(
            parent_workers,
            INTERPOLATION_PROCESS_WORKER_DEFAULT_CAP
            if bool(interpolation_process_backend_active) else 1,
        ),
    )
    overlap = max(
        1,
        min(
            parent_workers,
            _env_int('YOLO_TTA_INTERPOLATION_CONCURRENT_PARENTS', default_overlap),
        ),
    )
    default_workers = max(1, budget // max(1, int(overlap)))
    resolved_workers = max(
        1,
        _env_int('YOLO_TTA_INTERPOLATION_TASK_WORKERS', default_workers),
    )
    return int(overlap), int(default_workers), int(resolved_workers)

def resolve_worker_count(requested: int, env_name: str, auto_value: int, max_tasks: Optional[int] = None) -> int:
    workers = int(requested)
    if workers <= 0:
        workers = _env_int(env_name, int(auto_value))
    workers = max(1, int(workers))
    if max_tasks is not None:
        workers = max(1, min(int(workers), int(max_tasks)))
    return workers

def array_nbytes(shape: Sequence[int], dtype: np.dtype | str | type) -> int:
    dtype_obj = np.dtype(dtype)
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total) * int(dtype_obj.itemsize)

_MEMFD_OWNER_LOCK = threading.RLock()

_MEMFD_OWNERS: Dict[str, Tuple[int, str]] = {}

_MEMFD_ATEXIT_REGISTERED = False

def memfd_workspace_enabled() -> bool:
    return bool(
        sys.platform.startswith('linux')
        and hasattr(os, 'memfd_create')
        and _env_flag('YOLO_TTA_MEMFD_WORKSPACES', True)
    )

def raw_store_memfd_enabled() -> bool:
    """Keep cvol/ctile payloads in bounded pathname storage."""
    return False

def _memfd_label(value: object) -> str:
    token = re.sub(r'[^A-Za-z0-9_.+-]+', '-', str(value)).strip('-')
    return (token or 'yolo-tta')[:200]

def _memfd_proc_path(fd: int, *, owner_pid: Optional[int] = None) -> Path:
    return Path(f'/proc/{int(os.getpid() if owner_pid is None else owner_pid)}/fd/{int(fd)}')

def _close_all_memfd_owners() -> None:
    with _MEMFD_OWNER_LOCK:
        entries = list(_MEMFD_OWNERS.items())
        _MEMFD_OWNERS.clear()
    for _key, (fd, _desc) in entries:
        try:
            os.close(int(fd))
        except OSError:
            pass

def _register_memfd_owner(key: object, fd: int, desc: str) -> None:
    global _MEMFD_ATEXIT_REGISTERED
    key_s = str(key)
    with _MEMFD_OWNER_LOCK:
        previous = _MEMFD_OWNERS.pop(key_s, None)
        _MEMFD_OWNERS[key_s] = (int(fd), str(desc))
        if not _MEMFD_ATEXIT_REGISTERED:
            atexit.register(_close_all_memfd_owners)
            _MEMFD_ATEXIT_REGISTERED = True
    if previous is not None:
        try:
            os.close(int(previous[0]))
        except OSError:
            pass

def _release_memfd_owner_key(key: object) -> bool:
    key_s = str(key)
    with _MEMFD_OWNER_LOCK:
        entry = _MEMFD_OWNERS.pop(key_s, None)
    if entry is None:
        return False
    try:
        os.close(int(entry[0]))
    except OSError:
        pass
    return True

def release_memfd_owners_under(root: Path) -> int:
    """Release memfd files represented by symlinks below ``root``."""
    try:
        root_abs = Path(root).absolute()
    except Exception:
        root_abs = Path(root)
    with _MEMFD_OWNER_LOCK:
        keys = list(_MEMFD_OWNERS)
    released = 0
    for key in keys:
        try:
            key_path = Path(key)
            if key_path == root_abs or root_abs in key_path.parents:
                released += int(_release_memfd_owner_key(key))
        except Exception:
            continue
    return int(released)

def _allocate_memfd_workspace_array(
    shape: Sequence[int],
    dtype: np.dtype | str | type,
    desc: str,
    *,
    initialize_zero: bool,
) -> np.memmap:
    if not memfd_workspace_enabled():
        raise OSError('memfd workspaces are disabled or unavailable')
    dtype_obj = np.dtype(dtype)
    shape_i = tuple(int(v) for v in shape)
    nbytes = int(array_nbytes(shape_i, dtype_obj))
    flags = int(getattr(os, 'MFD_CLOEXEC', 0)) | int(getattr(os, 'MFD_ALLOW_SEALING', 0))
    fd = os.memfd_create(_memfd_label(desc), flags=flags)
    try:
        os.ftruncate(int(fd), int(nbytes))
        proc_path = _memfd_proc_path(int(fd))
        mm = np.memmap(proc_path, dtype=dtype_obj, mode='r+', shape=shape_i)
        # Root memmaps may carry Python attributes.  They make ownership explicit and
        # allow close_memmap_array to close both the mapping and the parent keepalive fd.
        mm._workspace_memfd_owner_key = str(proc_path)  # type: ignore[attr-defined]
        mm._workspace_memfd_path = str(proc_path)  # type: ignore[attr-defined]
        mm._workspace_memfd_owner_fd = int(fd)  # type: ignore[attr-defined]
        mm._workspace_memfd_desc = str(desc)  # type: ignore[attr-defined]
        _register_memfd_owner(str(proc_path), int(fd), str(desc))
        if bool(initialize_zero) and int(nbytes) > 0:
            # A newly ftruncate'd memfd reads as zero.  Do not touch every page merely
            # to prove that fact; direct-union writers overwrite disjoint slice windows.
            pass
        numa_interleave_memory(mm, desc=desc)
        return mm
    except BaseException:
        try:
            os.close(int(fd))
        except OSError:
            pass
        raise

def _memfd_owner_key_from_array(arr: object) -> Optional[str]:
    current = arr
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        key = getattr(current, '_workspace_memfd_owner_key', None)
        if key:
            return str(key)
        current = getattr(current, 'base', None)
    return None

def _memfd_backing_path_from_array(arr: object) -> Optional[Path]:
    """Return the stable parent-owned /proc fd path for a memfd workspace.

    NumPy resolves ``memmap.filename`` through procfs and reports a pseudo-name such as
    ``/memfd:name (deleted)``. That string is descriptive but cannot be reopened by a
    spawned CUDA worker, so reopening must use the explicit owner path retained here.
    """
    current = arr
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        raw = getattr(current, '_workspace_memfd_path', None)
        if raw:
            return Path(str(raw))
        current = getattr(current, 'base', None)
    return None

def _memfd_owner_fd_for_path(path: object) -> Optional[int]:
    """Return the parent-owned memfd represented by ``path``, if any."""
    if path is None:
        return None
    raw = str(path)
    candidates = [raw]
    try:
        candidates.append(str(Path(raw).absolute()))
    except Exception:
        pass
    with _MEMFD_OWNER_LOCK:
        for key in candidates:
            entry = _MEMFD_OWNERS.get(str(key))
            if entry is not None:
                return int(entry[0])
    return None

def _duplicate_memfd_path_for_child(path: object) -> Optional[object]:
    """Create a multiprocessing descriptor-transfer handle for a memfd path."""
    fd = _memfd_owner_fd_for_path(path)
    if fd is None:
        return None
    return mp_reduction.DupFd(int(fd))

def _attach_memfd_transfers_to_task(task: Dict[str, object]) -> None:
    """Attach descriptor handles to one dispatch copy without changing canonical paths."""
    for path_field, handle_field in (
        ('source_volume_path', 'source_volume_fd'),
        ('result_mask_path', 'result_mask_fd'),
        ('result_conf_path', 'result_conf_fd'),
        ('canvas_path', 'canvas_fd'),
        ('d1_bitset_path', 'd1_bitset_fd'),
    ):
        raw_path = task.get(path_field)
        handle = _duplicate_memfd_path_for_child(raw_path)
        if handle is not None:
            task[handle_field] = handle
            task[f'{handle_field}_key'] = str(raw_path)

    native_resize = task.get('native_resize')
    if isinstance(native_resize, dict):
        native_copy = dict(native_resize)
        native_path = native_copy.get('path')
        handle = _duplicate_memfd_path_for_child(native_path)
        if handle is not None:
            native_copy['path_fd'] = handle
            native_copy['path_fd_key'] = str(native_path)
        task['native_resize'] = native_copy

def _detach_transferred_fd(handle: object) -> int:
    detach = getattr(handle, 'detach', None)
    if not callable(detach):
        raise TypeError(f'Invalid multiprocessing fd-transfer handle: {type(handle)!r}')
    fd = int(detach())
    if fd < 0:
        raise OSError(f'Invalid transferred file descriptor: {fd}')
    return fd

def _materialize_worker_task_memfd_paths(
    task: Dict[str, object],
    persistent_sources: Dict[str, int],
) -> List[int]:
    """Replace transferred handles with worker-local procfs paths.

    Source-volume descriptors are cached for the worker lifetime so the resident render
    engine sees one stable source identity. Result and canvas descriptors are task-local;
    dropping them after the task permits immediate sparse retirement in the parent.
    """
    transient_fds: List[int] = []
    # The caller can only close descriptors after this function returns.  Keep enough
    # transaction state here to roll back descriptors detached before a later handle
    # fails to materialize; otherwise the assignment at the call site never happens and
    # those descriptors are lost for the lifetime of the worker.
    persistent_keys_before = set(persistent_sources)

    def _resolve(
        holder: Dict[str, object],
        *,
        path_field: str,
        handle_field: str,
        persistent: bool,
    ) -> None:
        handle = holder.pop(handle_field, None)
        key_raw = holder.pop(f'{handle_field}_key', None)
        if handle is None:
            return
        key = str(key_raw or holder.get(path_field) or handle_field)
        received_fd = _detach_transferred_fd(handle)
        if persistent:
            cached = persistent_sources.get(key)
            if cached is None:
                persistent_sources[key] = int(received_fd)
                fd = int(received_fd)
            else:
                try:
                    os.close(int(received_fd))
                except OSError:
                    pass
                fd = int(cached)
        else:
            fd = int(received_fd)
            transient_fds.append(int(fd))
        holder[path_field] = str(_memfd_proc_path(int(fd), owner_pid=os.getpid()))

    try:
        _resolve(task, path_field='source_volume_path', handle_field='source_volume_fd', persistent=True)
        _resolve(task, path_field='result_mask_path', handle_field='result_mask_fd', persistent=False)
        _resolve(task, path_field='result_conf_path', handle_field='result_conf_fd', persistent=False)
        _resolve(task, path_field='canvas_path', handle_field='canvas_fd', persistent=False)
        _resolve(task, path_field='d1_bitset_path', handle_field='d1_bitset_fd', persistent=False)

        native_resize = task.get('native_resize')
        if isinstance(native_resize, dict):
            native_copy = dict(native_resize)
            # Keep the key beside the handle while reusing the generic resolver.
            if 'path_fd_key' in native_copy:
                native_copy['path_fd_key'] = native_copy.get('path_fd_key')
            _resolve(native_copy, path_field='path', handle_field='path_fd', persistent=True)
            task['native_resize'] = native_copy
        return transient_fds
    except BaseException:
        _close_fd_list(transient_fds)
        for key in list(set(persistent_sources) - persistent_keys_before):
            fd = persistent_sources.pop(key, None)
            if fd is not None:
                _close_fd_list((int(fd),))
        raise

def _close_fd_list(fds: Iterable[int]) -> None:
    for fd in list(fds):
        try:
            os.close(int(fd))
        except OSError:
            pass

def preflight_multiprocessing_payload(payload: object) -> None:
    """Synchronously prove that a queue payload can cross a process boundary.

    ``multiprocessing.Queue.put`` normally returns before its feeder thread pickles the
    object.  A serialization error would therefore be printed by that background thread
    after the scheduler had committed ownership and dispatch counters.  Use the same
    ForkingPickler up front so such errors remain ordinary, rollback-capable exceptions.
    """
    try:
        mp_reduction.ForkingPickler.dumps(payload)
    except BaseException as exc:
        raise TypeError(
            f'multiprocessing dispatch payload is not serializable: {type(exc).__name__}: {exc}'
        ) from exc

def _madvise_dontneed_array(arr: object) -> None:
    """Best-effort immediate page-cache/RAM release before a scratch mapping closes."""
    advice = getattr(mmap, 'MADV_DONTNEED', None)
    if advice is None or arr is None:
        return
    current = arr
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        mmap_obj = getattr(current, '_mmap', None)
        if mmap_obj is not None:
            try:
                mmap_obj.madvise(advice)
            except (AttributeError, OSError, ValueError, BufferError):
                pass
            return
        current = getattr(current, 'base', None)

def _create_memfd_backed_payload_path(path: Path, desc: str) -> int:
    """Create ``path`` as a symlink to a parent-owned memfd and return a writer fd."""
    if not raw_store_memfd_enabled():
        raise OSError('memfd cvol payloads are disabled')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _release_memfd_owner_key(str(path.absolute()))
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
    flags = int(getattr(os, 'MFD_CLOEXEC', 0)) | int(getattr(os, 'MFD_ALLOW_SEALING', 0))
    owner_fd = os.memfd_create(_memfd_label(desc), flags=flags)
    writer_fd: Optional[int] = None
    target = _memfd_proc_path(int(owner_fd))
    try:
        writer_fd = int(os.dup(int(owner_fd)))
        os.symlink(str(target), str(path))
        _register_memfd_owner(str(path.absolute()), int(owner_fd), str(desc))
        owner_fd = -1  # registry owns it now
        return int(writer_fd)
    except BaseException:
        if writer_fd is not None:
            try:
                os.close(int(writer_fd))
            except OSError:
                pass
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        if int(owner_fd) >= 0:
            try:
                os.close(int(owner_fd))
            except OSError:
                pass
        raise

@contextlib.contextmanager
def open_raw_store_payload_writer(
    path: Path,
    desc: str,
    *,
    force_path_backed: bool = False,
) -> Iterator[object]:
    """Open a cvol payload, using a path-compatible memfd only when explicitly enabled."""
    writer_fd: Optional[int] = None
    if raw_store_memfd_enabled() and not bool(force_path_backed):
        try:
            writer_fd = _create_memfd_backed_payload_path(Path(path), str(desc))
        except Exception as exc:
            runtime_telemetry().fallback('cvol.memfd', exc)
            writer_fd = None
    if writer_fd is not None:
        try:
            with os.fdopen(int(writer_fd), 'wb', buffering=0) as handle:
                yield handle
        finally:
            # fdopen owns writer_fd.  The separately registered owner fd remains live.
            pass
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('wb') as handle:
        yield handle

def flush_array(arr: object, *, force: bool = False) -> None:
    """Compatibility no-op for ephemeral scratch mappings.

    NumPy memmaps use shared mappings, so same-host readers observe dirty pages without
    synchronous ``msync``. Scratch files are retired rather than used as durability records;
    forcing writeback only adds I/O before cleanup.
    """
    return

@runtime_telemetry_phase('workspace.allocate')
def allocate_workspace_array(
    shape: Sequence[int],
    dtype: np.dtype | str | type,
    path: Optional[Path],
    desc: str,
    *,
    prefer_memory: bool = True,
    prefer_memfd: bool = False,
    reserve_bytes: int = 16 * GIB,
    reuse_existing: bool = False,
    initialize_zero: bool = True,
) -> np.ndarray:
    dtype_obj = np.dtype(dtype)
    need_bytes = array_nbytes(shape, dtype_obj)
    budget = workspace_budget_summary(need_bytes, reserve_bytes=reserve_bytes)
    use_in_memory = bool(prefer_memory) and should_use_in_memory_workspace(need_bytes, reserve_bytes=reserve_bytes)

    if use_in_memory:
        try:
            print(f"{desc}: in-memory ({budget})")
            shape_tuple = tuple(int(x) for x in shape)
            arr = (
                np.zeros(shape_tuple, dtype=dtype_obj)
                if bool(initialize_zero)
                else np.empty(shape_tuple, dtype=dtype_obj)
            )
            # big allocations are still untouched here (>32 MiB glibc allocations
            # are fresh private mmaps, and np.zeros defers to lazily-faulted zero pages), so the
            # interleave policy lands before first touch.
            numa_interleave_memory(arr, desc=desc)
            return arr
        except MemoryError:
            print(f"{desc}: in-memory allocation failed, falling back to disk ({budget})")

    if bool(prefer_memfd) and should_use_in_memory_workspace(need_bytes, reserve_bytes=reserve_bytes):
        try:
            mm = _allocate_memfd_workspace_array(
                shape=shape,
                dtype=dtype_obj,
                desc=desc,
                initialize_zero=bool(initialize_zero),
            )
            print(f"{desc}: memfd-backed shared RAM ({budget}) -> {_memfd_backing_path_from_array(mm)}")
            runtime_telemetry().add('workspace.memfd_bytes', int(need_bytes))
            return mm
        except (MemoryError, OSError, RuntimeError) as exc:
            runtime_telemetry().fallback('workspace.memfd', exc)
            print(f"{desc}: memfd allocation unavailable ({exc}); falling back to pathname storage ({budget})")

    if path is None:
        raise ValueError(f"{desc}: pathname fallback requires a filesystem path")

    path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and path.exists():
        print(f"{desc}: disk-backed reuse ({budget}) -> {path}")
        mm = np.memmap(path, dtype=dtype_obj, mode='r+', shape=tuple(int(x) for x in shape))
        numa_interleave_memory(mm, desc=desc)  # best-effort (existing pages stay put)
        return mm

    if path.exists():
        path.unlink()
    print(f"{desc}: disk-backed ({budget}) -> {path}")
    mm = np.memmap(path, dtype=dtype_obj, mode='w+', shape=tuple(int(x) for x in shape))
    numa_interleave_memory(mm, desc=desc)  #
    return mm

def _copy_workspace_array_cpu(
    dst: np.ndarray,
    src: np.ndarray,
    *,
    workers: int,
    desc: str,
) -> None:
    """Run the exact pre-DSA CPU implementation over the complete destination."""
    if src.ndim <= 1:
        np.copyto(dst, src)
        return

    total = int(src.shape[0])

    def _copy(idx: int) -> None:
        np.copyto(dst[int(idx)], src[int(idx)])

    parallel_for_indices(
        total,
        _copy,
        max_workers=choose_slice_parallel_workers(int(workers), total),
        desc=f'{desc} copy',
        show_progress=False,
    )

def _discard_failed_workspace_copy(dst: object, path: Optional[Path]) -> None:
    """Invalidate storage owned by ``copy_workspace_array`` after a failed copy."""
    owner_key = _memfd_owner_key_from_array(dst)
    root = _root_memmap_for_array(dst)
    filename = getattr(root, 'filename', None) if root is not None else None
    close_memmap_array_without_flush(dst)
    if owner_key is not None or path is None or not filename:
        return
    try:
        requested = Path(path).absolute()
        actual = Path(str(filename)).absolute()
        if actual == requested and requested.exists():
            requested.unlink()
    except OSError:
        pass

def _workspace_copy_fault_counts() -> Tuple[int, int]:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(usage.ru_minflt), int(usage.ru_majflt)
    except Exception:
        return 0, 0

def _record_dsa_copy_stats(
    stats: Dict[str, object],
    *,
    capabilities: Dict[str, object],
    elapsed_ns: int,
    cpu_ns: int,
    minor_faults: int,
    major_faults: int,
) -> None:
    telemetry = runtime_telemetry()
    for key in (
        'hardware_bytes',
        'descriptors',
        'submitted_descriptors',
        'batches',
        'queue_full_events',
        'partial_failures',
        'page_faults',
    ):
        try:
            telemetry.add(f'workspace.copy.dsa.{key}', int(stats.get(key, 0)))
        except Exception:
            pass
    for key in ('minor_page_faults', 'major_page_faults'):
        try:
            telemetry.add(f'workspace.copy.dsa.native_{key}', int(stats.get(key, 0)))
        except Exception:
            pass
    for key in ('submission_seconds', 'wait_seconds', 'total_seconds'):
        try:
            telemetry.add(f'workspace.copy.dsa.{key}', float(stats.get(key, 0.0)))
        except Exception:
            pass
    seconds = max(1e-12, float(elapsed_ns) / 1e9)
    copied = max(0, int(stats.get('hardware_bytes', 0)))
    telemetry.gauge('workspace.copy.dsa.last_gib_per_second', copied / GIB / seconds)
    telemetry.gauge('workspace.copy.dsa.last_cpu_seconds', max(0, int(cpu_ns)) / 1e9)
    telemetry.add('workspace.copy.dsa.minor_page_faults', max(0, int(minor_faults)))
    telemetry.add('workspace.copy.dsa.major_page_faults', max(0, int(major_faults)))
    telemetry.gauge('workspace.copy.dsa.work_queue', capabilities.get('work_queue'))
    telemetry.gauge('workspace.copy.dsa.work_queue_mode', capabilities.get('work_queue_mode'))
    telemetry.gauge('workspace.copy.dsa.numa_node', capabilities.get('numa_node'))
    telemetry.gauge('workspace.copy.dsa.caller_numa_node', capabilities.get('caller_numa_node'))
    telemetry.gauge('workspace.copy.dsa.block_on_fault', capabilities.get('block_on_fault'))
    telemetry.gauge('workspace.copy.dsa.queue_depth', stats.get('max_inflight'))

def copy_workspace_array(
    src: np.ndarray,
    path: Optional[Path],
    desc: str,
    *,
    prefer_memory: bool = True,
    prefer_memfd: bool = False,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
) -> np.ndarray:
    """Allocate and copy a workspace with optional, fail-closed Linux DSA offload."""
    # Importing this dependency-free control plane is cheap; its native extension is
    # still imported only when auto/dsa reaches an otherwise eligible copy.
    from . import intel_dsa

    backend = intel_dsa.requested_backend()
    minimum_bytes = intel_dsa.minimum_copy_bytes() if backend != 'cpu' else 0
    max_inflight = intel_dsa.requested_max_inflight() if backend != 'cpu' else 1
    requested_wq = intel_dsa.requested_work_queue() if backend != 'cpu' else None
    dst = allocate_workspace_array(
        shape=src.shape,
        dtype=src.dtype,
        path=path,
        desc=desc,
        prefer_memory=bool(prefer_memory),
        prefer_memfd=bool(prefer_memfd),
        reserve_bytes=int(reserve_bytes),
        initialize_zero=False,
    )
    if backend == 'cpu':
        telemetry = runtime_telemetry()
        telemetry.gauge('workspace.copy.requested_backend', 'cpu')
        telemetry.gauge('workspace.copy.selected_backend', 'cpu')
        _copy_workspace_array_cpu(dst, src, workers=int(workers), desc=desc)
        flush_array(dst)
        return dst

    telemetry = runtime_telemetry()
    telemetry.gauge('workspace.copy.requested_backend', backend)
    eligibility = intel_dsa.assess_copy_eligibility(
        src,
        dst,
        minimum_bytes=int(minimum_bytes),
    )
    telemetry.gauge('workspace.copy.source_backing', eligibility.source_backing)
    telemetry.gauge('workspace.copy.destination_backing', eligibility.destination_backing)
    telemetry.gauge('workspace.copy.destination_pages', 'unknown')
    if not eligibility.eligible:
        exc = intel_dsa.IntelDsaIneligible(eligibility.reasons)
        telemetry.gauge('workspace.copy.dsa.last_ineligibility_reasons', eligibility.reasons)
        if backend == 'dsa':
            telemetry.gauge('workspace.copy.selected_backend', 'none')
            _discard_failed_workspace_copy(dst, path)
            raise exc
        telemetry.fallback('workspace.copy.dsa.ineligible', exc)
        telemetry.gauge('workspace.copy.selected_backend', 'cpu')
        _copy_workspace_array_cpu(dst, src, workers=int(workers), desc=desc)
        flush_array(dst)
        return dst

    # Capability failures occur before native submission and are therefore safe initial
    # CPU fallbacks in auto mode. Execution failures take the stricter drained path below.
    try:
        manager = intel_dsa.get_manager()
        capabilities = manager.capabilities(work_queue=requested_wq)
    except intel_dsa.IntelDsaUnavailable as exc:
        if backend == 'dsa':
            telemetry.gauge('workspace.copy.selected_backend', 'none')
            _discard_failed_workspace_copy(dst, path)
            raise
        telemetry.fallback('workspace.copy.dsa.unavailable', exc)
        telemetry.gauge('workspace.copy.selected_backend', 'cpu')
        _copy_workspace_array_cpu(dst, src, workers=int(workers), desc=desc)
        flush_array(dst)
        return dst

    try:
        native_limit = max(1, int(capabilities.get('max_inflight', max_inflight)))
    except Exception:
        native_limit = int(max_inflight)
    effective_inflight = min(int(max_inflight), int(native_limit))
    before_minor, before_major = _workspace_copy_fault_counts()
    started_ns = time.monotonic_ns()
    started_cpu_ns = time.process_time_ns()
    try:
        stats = manager.copy(
            src,
            dst,
            capabilities=capabilities,
            max_inflight=effective_inflight,
            failure_cleanup=lambda: _discard_failed_workspace_copy(dst, path),
        )
    except intel_dsa.IntelDsaCopyError as exc:
        failed_elapsed_ns = int(time.monotonic_ns() - started_ns)
        failed_cpu_ns = int(time.process_time_ns() - started_cpu_ns)
        failed_minor, failed_major = _workspace_copy_fault_counts()
        failed_stats = dict(exc.stats)
        failed_stats.setdefault('max_inflight', int(effective_inflight))
        _record_dsa_copy_stats(
            failed_stats,
            capabilities=capabilities,
            elapsed_ns=failed_elapsed_ns,
            cpu_ns=failed_cpu_ns,
            minor_faults=max(0, int(failed_minor) - int(before_minor)),
            major_faults=max(0, int(failed_major) - int(before_major)),
        )
        telemetry.add('workspace.copy.dsa.failed_requests', 1)
        telemetry.gauge('workspace.copy.dsa.last_failure_drained', bool(exc.drained))
        telemetry.fallback('workspace.copy.dsa.execution', exc)
        # An undrained error is fatal even in auto: a late device write could race and
        # corrupt a CPU recovery copy. The failed destination is never handed to callers.
        if not bool(exc.drained):
            telemetry.gauge('workspace.copy.selected_backend', 'none')
            telemetry.gauge('workspace.copy.dsa.buffers_quarantined', True)
            raise
        if backend == 'dsa':
            telemetry.gauge('workspace.copy.selected_backend', 'none')
            _discard_failed_workspace_copy(dst, path)
            raise
        telemetry.add('workspace.copy.dsa.full_cpu_recopies', 1)
        telemetry.add('workspace.copy.dsa.full_cpu_recopy_bytes', int(eligibility.nbytes))
        telemetry.gauge('workspace.copy.selected_backend', 'cpu_recopy_after_dsa')
        try:
            _copy_workspace_array_cpu(dst, src, workers=int(workers), desc=desc)
            flush_array(dst)
            return dst
        except BaseException:
            _discard_failed_workspace_copy(dst, path)
            raise

    elapsed_ns = int(time.monotonic_ns() - started_ns)
    cpu_ns = int(time.process_time_ns() - started_cpu_ns)
    after_minor, after_major = _workspace_copy_fault_counts()
    stats = dict(stats)
    stats.setdefault('max_inflight', int(effective_inflight))
    _record_dsa_copy_stats(
        stats,
        capabilities=capabilities,
        elapsed_ns=elapsed_ns,
        cpu_ns=cpu_ns,
        minor_faults=max(0, int(after_minor) - int(before_minor)),
        major_faults=max(0, int(after_major) - int(before_major)),
    )
    telemetry.gauge('workspace.copy.dsa.buffers_quarantined', False)
    telemetry.gauge('workspace.copy.selected_backend', 'dsa')
    flush_array(dst)
    return dst

_PARALLEL_POOL_CACHE: Dict[int, List[ThreadPoolExecutor]] = {}

_PARALLEL_POOL_CACHE_LOCK = threading.Lock()

def _parallel_pool_cache_max_threads() -> int:
    return max(0, _env_int('YOLO_TTA_PARALLEL_POOL_CACHE_MAX_THREADS', 1024))

def _acquire_parallel_pool(workers: int) -> ThreadPoolExecutor:
    w = max(1, int(workers))
    with _PARALLEL_POOL_CACHE_LOCK:
        stack = _PARALLEL_POOL_CACHE.get(w)
        if stack:
            return stack.pop()
    return ThreadPoolExecutor(max_workers=w, thread_name_prefix=f'parallel-{w}')

def _release_parallel_pool(workers: int, pool: ThreadPoolExecutor) -> None:
    w = max(1, int(workers))
    with _PARALLEL_POOL_CACHE_LOCK:
        idle_threads = sum(int(size) * len(stack) for size, stack in _PARALLEL_POOL_CACHE.items())
        if idle_threads + w <= _parallel_pool_cache_max_threads():
            _PARALLEL_POOL_CACHE.setdefault(w, []).append(pool)
            return
    pool.shutdown(wait=False)

def shutdown_parallel_pool_cache() -> None:
    """Close every idle reusable pool at a top-level run boundary."""
    with _PARALLEL_POOL_CACHE_LOCK:
        pools = [pool for stack in _PARALLEL_POOL_CACHE.values() for pool in stack]
        _PARALLEL_POOL_CACHE.clear()
    for pool in pools:
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=True)

def _settle_parallel_futures(futures: Iterable[Future]) -> None:
    """Cancel-or-wait every future so a pool can safely return to the cache.

 Called on abandoned generators / error paths: pending futures either cancel (never ran)
 or finish; nothing of this call is left running when the pool is released."""
    remaining = [f for f in futures if f is not None]
    for fut in remaining:
        try:
            fut.cancel()
        except Exception:
            pass
    for fut in remaining:
        try:
            fut.exception()  # waits for completion; swallows task errors on cleanup paths
        except Exception:
            pass

def parallel_map_in_order(
    func: Callable[[int], object],
    items: Iterable[int],
    *,
    max_workers: int,
    max_pending: Optional[int] = None,
) -> Iterator[object]:
    workers = max(1, int(max_workers))
    if workers <= 1:
        for item in items:
            yield func(int(item))
        return

    pending_limit = max(workers, int(max_pending) if max_pending is not None else workers + 1)
    executor = _acquire_parallel_pool(workers)
    queue: List[Future] = []
    try:
        for item in items:
            queue.append(executor.submit(func, int(item)))
            if len(queue) >= pending_limit:
                fut = queue.pop(0)
                yield fut.result()
        while queue:
            fut = queue.pop(0)
            yield fut.result()
    finally:
        _settle_parallel_futures(queue)
        _release_parallel_pool(workers, executor)

def parallel_map_unordered(
    func: Callable[[int], object],
    items: Iterable[int],
    *,
    max_workers: int,
    max_pending: Optional[int] = None,
) -> Iterator[object]:
    """Bounded parallel map that yields results as soon as tasks complete.

 This is intended for variable-cost tasks whose output order does not affect the
 final binary result, such as interpolation endpoint seed planning. Unlike
 ``parallel_map_in_order``, one slow early item cannot block result consumption
 or prevent later items from being submitted once the pending bound is reached."""
    workers = max(1, int(max_workers))
    if workers <= 1:
        for item in items:
            yield func(int(item))
        return

    pending_limit = max(workers, int(max_pending) if max_pending is not None else workers + 1)
    iterator = iter(items)
    executor = _acquire_parallel_pool(workers)
    pending: set[Future] = set()
    try:
        def _submit_until_full() -> None:
            while len(pending) < pending_limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.add(executor.submit(func, int(item)))

        _submit_until_full()
        while pending:
            done, pending_remainder = wait(pending, return_when=FIRST_COMPLETED)
            pending = set(pending_remainder)
            for fut in done:
                yield fut.result()
            _submit_until_full()
    finally:
        _settle_parallel_futures(pending)
        _release_parallel_pool(workers, executor)

def parallel_for_indices(
    count: int,
    func: Callable[[int], None],
    *,
    max_workers: int,
    desc: str,
    show_progress: bool = True,
) -> None:
    total = max(0, int(count))
    if total <= 0:
        return

    workers = max(1, min(int(max_workers), total))
    if workers <= 1:
        iterable = tqdm(range(total), desc=desc) if show_progress else range(total)
        for idx in iterable:
            func(int(idx))
        return

    executor = _acquire_parallel_pool(workers)
    futures: List[Future] = []
    try:
        futures = [executor.submit(func, int(idx)) for idx in range(total)]
        if show_progress:
            with tqdm(total=total, desc=desc) as pbar:
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)
        else:
            for fut in as_completed(futures):
                fut.result()
    finally:
        _settle_parallel_futures(futures)
        _release_parallel_pool(workers, executor)

def choose_parallel_chunk_size(
    total_items: int,
    max_workers: int,
    *,
    target_chunks_per_worker: int = 4,
    min_chunk_size: int = 1,
    max_chunk_size: Optional[int] = None,
) -> int:
    total = max(0, int(total_items))
    workers = max(1, int(max_workers))
    if total <= 0:
        return max(1, int(min_chunk_size))

    denom = max(1, workers * max(1, int(target_chunks_per_worker)))
    chunk = max(int(min_chunk_size), int(math.ceil(float(total) / float(denom))))
    if max_chunk_size is not None:
        chunk = min(int(chunk), max(1, int(max_chunk_size)))
    return max(1, int(chunk))

def parallel_for_indices_chunked(
    count: int,
    func: Callable[[int], None],
    *,
    max_workers: int,
    desc: str,
    show_progress: bool = True,
    chunk_size: Optional[int] = None,
    target_chunks_per_worker: int = 4,
) -> None:
    total = max(0, int(count))
    if total <= 0:
        return

    workers = max(1, min(int(max_workers), total))
    if workers <= 1:
        iterable = tqdm(range(total), desc=desc) if show_progress else range(total)
        for idx in iterable:
            func(int(idx))
        return

    if chunk_size is None or int(chunk_size) <= 0:
        chunk = choose_parallel_chunk_size(
            total,
            workers,
            target_chunks_per_worker=int(target_chunks_per_worker),
            min_chunk_size=1,
        )
    else:
        chunk = max(1, int(chunk_size))

    ranges = [(int(start), int(min(total, start + chunk))) for start in range(0, total, chunk)]

    def _run_range(range_idx: int) -> int:
        start, stop = ranges[int(range_idx)]
        for idx in range(int(start), int(stop)):
            func(int(idx))
        return int(stop - start)

    executor = _acquire_parallel_pool(workers)
    futures: List[Future] = []
    try:
        futures = [executor.submit(_run_range, int(range_idx)) for range_idx in range(len(ranges))]
        if show_progress:
            with tqdm(total=total, desc=desc) as pbar:
                for fut in as_completed(futures):
                    pbar.update(int(fut.result()))
        else:
            for fut in as_completed(futures):
                fut.result()
    finally:
        _settle_parallel_futures(futures)
        _release_parallel_pool(workers, executor)

def workspace_anon_cap_bytes() -> int:
    """Return the optional anonymous-workspace cap.

 Task overrides remove the previous conservative default fractional cap. Anonymous workspaces are
 only capped when the user explicitly sets YOLO_TTA_MAX_ANON_WORKSPACE_GIB."""
    hard_cap_gib = max(0.0, _env_float('YOLO_TTA_MAX_ANON_WORKSPACE_GIB', 0.0))
    if hard_cap_gib <= 0.0:
        return 0
    return int(hard_cap_gib * GIB)

def workspace_budget_summary(required_bytes: int, reserve_bytes: int = 16 * GIB) -> str:
    avail = available_anon_work_bytes()
    cap = workspace_anon_cap_bytes()
    reserve = int(max(0, reserve_bytes))
    parts = [
        f'need={required_bytes / GIB:.1f} GiB',
        f'avail+swap={avail / GIB:.1f} GiB',
        f'reserve={reserve / GIB:.1f} GiB',
    ]
    if cap > 0:
        parts.append(f'anon-cap={cap / GIB:.1f} GiB')
    return ', '.join(parts)

def should_use_in_memory_workspace(required_bytes: int, reserve_bytes: int = 16 * GIB) -> bool:
    if int(required_bytes) <= 0:
        return False

    avail = available_anon_work_bytes()
    reserve = int(max(0, reserve_bytes))
    cap = workspace_anon_cap_bytes()

    if cap > 0 and int(required_bytes) > cap:
        return False
    return avail >= int(required_bytes) + reserve

def choose_slice_parallel_workers(requested_workers: int, num_items: int) -> int:
    return max(1, min(int(requested_workers), int(max(1, num_items))))

_MEMORY_BACKED_FSTYPES = ('tmpfs', 'ramfs', 'hugetlbfs')

_SCRATCH_DIR_IS_MEMORY_BACKED = False

_RUN_SCRATCH_CLEANUP_LOCK = threading.Lock()

_RUN_SCRATCH_CLEANUP_PATH: Optional[Path] = None

_RUN_SCRATCH_CLEANUP_KEEP = True

_RUN_SCRATCH_CLEANUP_REGISTERED = False

_RUN_TERMINATION_REQUESTED = threading.Event()

_RUN_TERMINATION_WATCHDOG_STARTED = False

_RUN_TERMINATION_SIGNAL = int(getattr(signal, 'SIGTERM', 15))

def _cleanup_registered_unique_run_scratch() -> None:
    """Remove this process's unique scratch tree on normal/handled termination.

    Only ``{stem}_{pid}_temp`` directories are eligible.  The conservative ownership
    check prevents an exit hook from recursively removing a caller-supplied scratch root
    or the shared default output directory.  SIGKILL remains inherently uncatchable.
    """
    global _RUN_SCRATCH_CLEANUP_PATH
    with _RUN_SCRATCH_CLEANUP_LOCK:
        path = _RUN_SCRATCH_CLEANUP_PATH
        if path is None or bool(_RUN_SCRATCH_CLEANUP_KEEP):
            return
        owned_suffix = f'_{int(os.getpid())}_temp'
        if not str(Path(path).name).endswith(owned_suffix):
            return
        _RUN_SCRATCH_CLEANUP_PATH = None
    try:
        release_memfd_owners_under(Path(path))
    except Exception:
        pass
    try:
        shutil.rmtree(Path(path), ignore_errors=True)
    except Exception:
        pass

def _force_termination_watchdog() -> None:
    """Bound graceful teardown so SLURM KillWait cannot strand PID scratch.

    Executor shutdown normally waits for active interpolation.  A pass can run for hours, so
    SIGTERM needs a finite grace period before terminating children, removing this run's
    conservatively-owned scratch tree, and exiting without Python's thread-join atexit phase.
    """
    _RUN_TERMINATION_REQUESTED.wait()
    signum = int(_RUN_TERMINATION_SIGNAL)
    grace_seconds = max(
        1.0, _env_float('YOLO_TTA_TERMINATION_GRACE_SECONDS', 20.0),
    )
    deadline = time.monotonic() + float(grace_seconds)
    while time.monotonic() < deadline:
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    try:
        children = list(mp.active_children())
    except Exception:
        children = []
    for child in children:
        try:
            child.terminate()
        except Exception:
            pass
    for child in children:
        try:
            child.join(timeout=0.5)
            if child.is_alive() and hasattr(child, 'kill'):
                child.kill()
        except Exception:
            pass
    _cleanup_registered_unique_run_scratch()
    os._exit(128 + int(signum))

def _run_scratch_sigterm_handler(signum: int, _frame: object) -> None:
    global _RUN_TERMINATION_SIGNAL
    _RUN_TERMINATION_SIGNAL = int(signum)
    _RUN_TERMINATION_REQUESTED.set()
    raise KeyboardInterrupt(f'received signal {int(signum)}')

def register_unique_run_scratch_cleanup(path: Path, *, keep_temp: bool) -> None:
    """Register conservative atexit/SIGTERM cleanup for one PID-owned scratch tree."""
    global _RUN_SCRATCH_CLEANUP_PATH, _RUN_SCRATCH_CLEANUP_KEEP, _RUN_SCRATCH_CLEANUP_REGISTERED
    global _RUN_TERMINATION_WATCHDOG_STARTED
    path_obj = Path(path)
    with _RUN_SCRATCH_CLEANUP_LOCK:
        _RUN_SCRATCH_CLEANUP_PATH = path_obj
        _RUN_SCRATCH_CLEANUP_KEEP = bool(keep_temp)
        if not _RUN_SCRATCH_CLEANUP_REGISTERED:
            atexit.register(_cleanup_registered_unique_run_scratch)
            _RUN_SCRATCH_CLEANUP_REGISTERED = True

    # SLURM normally sends SIGTERM before a forced SIGKILL. Convert that catchable signal
    # into Python unwinding so executor/sink finalizers and the atexit scratch cleanup run.
    if not bool(keep_temp) and str(path_obj.name).endswith(f'_{int(os.getpid())}_temp'):
        try:
            if not _RUN_TERMINATION_WATCHDOG_STARTED:
                watchdog = threading.Thread(
                    target=_force_termination_watchdog,
                    name='termination-watchdog',
                    daemon=True,
                )
                watchdog.start()
                _RUN_TERMINATION_WATCHDOG_STARTED = True
            signal.signal(signal.SIGTERM, _run_scratch_sigterm_handler)
            signal.signal(signal.SIGINT, _run_scratch_sigterm_handler)
        except (ValueError, OSError, AttributeError):
            pass

def _mount_fstype_for_path(path: Path) -> Optional[str]:
    """Filesystem type of the longest mount point containing ``path``, or None if unknown."""
    try:
        target = Path(path).resolve()
    except Exception:
        target = Path(path)
    try:
        lines = Path('/proc/mounts').read_text().splitlines()
    except Exception:
        return None
    best_depth = -1
    best_fstype: Optional[str] = None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            # proc/mounts octal-escapes spaces and friends in the mount point.
            mount_point = Path(parts[1].replace('\\040', ' '))
        except Exception:
            continue
        if target != mount_point and mount_point not in target.parents:
            continue
        depth = len(mount_point.parts)
        if depth > best_depth:
            best_depth = depth
            best_fstype = str(parts[2])
    return best_fstype

def path_is_memory_backed(path: Path) -> bool:
    """True when ``path`` lives on tmpfs/ramfs/hugetlbfs, i.e. its 'files' are RAM pages."""
    fstype = _mount_fstype_for_path(path)
    return bool(fstype is not None and str(fstype).lower() in _MEMORY_BACKED_FSTYPES)

def _filesystem_free_bytes(path: Path) -> int:
    try:
        return int(shutil.disk_usage(str(path)).free)
    except Exception:
        return 0

def scratch_dir_is_memory_backed() -> bool:
    """Whether the scratch root chosen by ``choose_scratch_dir`` is RAM rather than disk."""
    return bool(_SCRATCH_DIR_IS_MEMORY_BACKED)

def choose_scratch_dir(preferred: Optional[str], out_dir: Path, stem: str) -> Path:
    """Pick the bulk scratch directory.

 - ``--temp`` names a root and receives a unique ``{stem}_{pid}_temp`` child.
 - otherwise the default is exactly ``{output}/temp``."""
    global _SCRATCH_DIR_IS_MEMORY_BACKED

    if preferred:
        explicit_root = Path(preferred).expanduser()
        try:
            explicit_root = explicit_root.resolve()
        except Exception:
            pass
        try:
            explicit_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f'Unable to create --temp root {explicit_root}: {exc}') from exc
        if not explicit_root.is_dir() or not os.access(str(explicit_root), os.W_OK):
            raise RuntimeError(f'--temp root is not a writable directory: {explicit_root}')
        scratch_dir = explicit_root / f'{stem}_{os.getpid()}_temp'
        scratch_dir.mkdir(parents=True, exist_ok=True)
        _SCRATCH_DIR_IS_MEMORY_BACKED = bool(path_is_memory_backed(scratch_dir))
        return scratch_dir

    scratch_dir = Path(out_dir) / 'temp'
    scratch_dir.mkdir(parents=True, exist_ok=True)
    _SCRATCH_DIR_IS_MEMORY_BACKED = bool(path_is_memory_backed(scratch_dir))
    return scratch_dir

def estimate_voidfill_workspace_bytes(
    shape: Tuple[int, int, int],
    *,
    dtype: np.dtype | str | type = np.uint32,
) -> int:
    """Return bytes required for one full-volume label workspace.
    
    The global 3D void-fill path uses uint32; slice-local topology callers may request uint16 explicitly."""
    z_dim, h, w = shape
    return int(z_dim) * int(h) * int(w) * np.dtype(dtype).itemsize

def estimate_interpolation_workspace_bytes(
    shape: Tuple[int, int, int],
    *,
    label_dtype: np.dtype | str | type = np.uint32,
) -> int:
    z_dim, h, w = shape
    voxels = int(z_dim) * int(h) * int(w)
    return voxels * (np.dtype(label_dtype).itemsize + np.dtype(np.uint8).itemsize)

def expose_scratch_in_output(out_dir: Path, scratch_dir: Path) -> Path:
    """Expose the active scratch directory from the output tree when possible."""
    temp_entry = out_dir / 'temp'
    try:
        if temp_entry.exists() or temp_entry.is_symlink():
            if temp_entry.is_symlink() or temp_entry.is_file():
                temp_entry.unlink(missing_ok=True)
            elif temp_entry.resolve() != scratch_dir.resolve():
                shutil.rmtree(temp_entry, ignore_errors=True)
    except Exception:
        pass

    try:
        if temp_entry.resolve() == scratch_dir.resolve():
            return temp_entry
    except Exception:
        pass

    if temp_entry.exists() or temp_entry.is_symlink():
        return temp_entry

    try:
        os.symlink(str(scratch_dir), str(temp_entry), target_is_directory=True)
        return temp_entry
    except Exception:
        temp_entry.mkdir(parents=True, exist_ok=True)
        (temp_entry / 'SCRATCH_LOCATION.txt').write_text(str(scratch_dir) + '\n')
        return temp_entry

def _root_memmap_for_array(arr: object) -> Optional[np.memmap]:
    """Return the root memmap backing ``arr`` without materializing a copy."""
    current = arr
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, np.memmap):
            return current
        current = getattr(current, 'base', None)
    return None

def close_memmap_array(arr: object) -> None:
    if arr is None:
        return
    owner_key = _memfd_owner_key_from_array(arr)
    try:
        lazy_close = getattr(arr, 'close', None)
        if bool(getattr(arr, '_is_lazy_processing_cube', False)) and callable(lazy_close):
            try:
                lazy_close()
            except Exception:
                pass
            return
        flush_array(arr)
        if owner_key is not None:
            _madvise_dontneed_array(arr)
        root = _root_memmap_for_array(arr)
        if root is not None:
            mmap_obj = getattr(root, '_mmap', None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except (BufferError, OSError, ValueError):
                    pass
    finally:
        if owner_key is not None:
            _release_memfd_owner_key(owner_key)

def close_memmap_array_without_flush(arr: object) -> None:
    """Close a scratch mapping without forcing dirty pages to storage."""
    if arr is None:
        return
    owner_key = _memfd_owner_key_from_array(arr)
    try:
        if owner_key is not None:
            _madvise_dontneed_array(arr)
        root = _root_memmap_for_array(arr)
        if root is not None:
            mmap_obj = getattr(root, '_mmap', None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except (BufferError, OSError, ValueError):
                    pass
    finally:
        if owner_key is not None:
            _release_memfd_owner_key(owner_key)

def prediction_volume_build_flush_enabled() -> bool:
    """Return True to force flushing YOLO input volumes before inference."""
    return False

def prediction_hot_path_flush_enabled() -> bool:
    """Return True to force per-source prediction accumulation memmap flushes."""
    return False

_INTERPOLATION_PROCESS_EXECUTOR: Optional[ProcessPoolExecutor] = None

_INTERPOLATION_PROCESS_MAX_WORKERS = 0

_INTERPOLATION_PROCESS_WORKER = False

class _InterpolationPassAdmission:
    """Process-wide admission for every parent-scheduled interpolation pass.

    The lease surrounds process-backing conversion, auxiliary/dedicated submission, and the
    pass itself.  Consequently a queued tile pass cannot allocate its full-volume process
    input while another full-frame/auxiliary pass is still holding its planning heap.
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        self.max_concurrent = max(1, int(max_concurrent))
        self.active = 0
        self.active_estimated_bytes = 0
        self.condition = threading.Condition()

    @contextlib.contextmanager
    def reserve(self, desc: str, estimated_bytes: int) -> Iterator[float]:
        estimate = max(1, int(estimated_bytes))
        started_waiting = time.monotonic()
        announced = False
        with self.condition:
            while int(self.active) >= int(self.max_concurrent):
                if not announced:
                    print(
                        'Global interpolation admission: waiting for '
                        f'{desc} ({estimate / GIB:.1f} GiB structural estimate; '
                        f'{self.active}/{self.max_concurrent} pass slot(s) active, '
                        f'{self.active_estimated_bytes / GIB:.1f} GiB estimated active).'
                    )
                    announced = True
                self.condition.wait()
            waited_seconds = max(0.0, time.monotonic() - started_waiting)
            self.active += 1
            self.active_estimated_bytes += int(estimate)
        try:
            yield float(waited_seconds)
        finally:
            with self.condition:
                self.active = max(0, int(self.active) - 1)
                self.active_estimated_bytes = max(
                    0, int(self.active_estimated_bytes) - int(estimate)
                )
                self.condition.notify_all()

_INTERPOLATION_PASS_ADMISSION = _InterpolationPassAdmission(1)

def configure_interpolation_pass_admission(max_concurrent: int) -> None:
    """Configure global interpolation overlap before any parent pass is submitted."""
    global _INTERPOLATION_PASS_ADMISSION
    _INTERPOLATION_PASS_ADMISSION = _InterpolationPassAdmission(max(1, int(max_concurrent)))

def _globally_admitted_interpolation_pass(func: Callable[..., object]) -> Callable[..., object]:
    """Decorate the public interpolation entry point with one shared parent-side lease."""
    @functools.wraps(func)
    def admitted(mask_mm: np.ndarray, *args: object, **kwargs: object) -> object:
        # Process workers enter interpolate_view_volume_pass_inplace directly.  Every such
        # task already remains covered by the parent caller's lease while it waits for the
        # result, so a second process-local lease would provide no cross-process protection.
        if bool(_INTERPOLATION_PROCESS_WORKER):
            return func(mask_mm, *args, **kwargs)

        pass_tag = kwargs.get('pass_tag')
        if pass_tag is None and len(args) >= 3:
            pass_tag = args[2]
        view = kwargs.get('view')
        if view is None and len(args) >= 1:
            view = args[0]
        view_name = str(getattr(view, 'name', 'unknown-view'))
        desc = f'{view_name}/{pass_tag or "pass"}'

        # Five volume-equivalents cover the dense label-packing peak plus bridge canvas.
        # The topology-dependent planner heap is deliberately not presented as a byte bound;
        # default single-pass serialization is what prevents that unbounded term multiplying.
        volume_bytes = int(np.asarray(mask_mm).nbytes)
        structural_estimate = 5 * int(volume_bytes)
        if kwargs.get('bridge_delta_path') is not None:
            structural_estimate += int(volume_bytes)

        with _INTERPOLATION_PASS_ADMISSION.reserve(
            str(desc), int(structural_estimate),
        ) as waited_seconds:
            result = func(mask_mm, *args, **kwargs)

        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
        ):
            result[1].setdefault('global_admission_wait_seconds', float(waited_seconds))
            result[1].setdefault(
                'global_admission_structural_estimate_bytes', int(structural_estimate)
            )
            result[1].setdefault(
                'global_admission_pass_limit',
                int(_INTERPOLATION_PASS_ADMISSION.max_concurrent),
            )
        return result

    return admitted

def _sanitize_filesystem_token(value: object) -> str:
    token = re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(value).strip()).strip('_')
    return token or 'unnamed'

def interpolation_process_backend_enabled() -> bool:
    """Return whether interpolation passes use isolated process workers.

    Process isolation gives Python-heavy seed planning independent GILs and prevents it
    from starving main-process render threads. Set
    ``YOLO_TTA_INTERPOLATION_PROCESS_BACKEND=0`` to use the in-process path.
    """
    return _env_flag('YOLO_TTA_INTERPOLATION_PROCESS_BACKEND', True)

def interpolation_process_fallback_enabled() -> bool:
    """Allow an in-process fallback if a process interpolation task fails.

 The default is fail-fast so accidental reintroduction of the GIL bottleneck is
 visible. Set YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK=1 when completing a run is
 preferred over failing on a worker-process exception."""
    return _env_flag('YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK', False)

def interpolation_process_start_method() -> str:
    method = os.environ.get('YOLO_TTA_INTERPOLATION_PROCESS_START_METHOD', 'spawn').strip().lower()
    if method not in {'spawn', 'forkserver'}:
        raise ValueError(
            'YOLO_TTA_INTERPOLATION_PROCESS_START_METHOD must be spawn or forkserver; '
            f'{method or "<empty>"!r} is unsafe because it can inherit live thread pools '
            'and accelerator runtime state'
        )
    return method

def interpolation_process_cv2_threads() -> int:
    return max(1, _env_int('YOLO_TTA_INTERPOLATION_PROCESS_CV2_THREADS', 1))

def _interpolation_process_initializer() -> None:
    global _INTERPOLATION_PROCESS_WORKER
    initialize_runtime_observability()
    _INTERPOLATION_PROCESS_WORKER = True
    try:
        cv2.setNumThreads(int(interpolation_process_cv2_threads()))
    except Exception:
        pass


def interpolation_process_worker_active() -> bool:
    """Whether this interpreter is a dedicated interpolation worker process."""

    return bool(_INTERPOLATION_PROCESS_WORKER)

def create_interpolation_process_executor(max_workers: int) -> Optional[ProcessPoolExecutor]:
    if not interpolation_process_backend_enabled():
        return None
    workers = max(1, int(max_workers))
    start_method = interpolation_process_start_method()
    try:
        ctx = mp.get_context(start_method)
    except Exception:
        start_method = 'spawn'
        ctx = mp.get_context(start_method)
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_interpolation_process_initializer,
    )

def set_interpolation_process_executor(executor: Optional[ProcessPoolExecutor], max_workers: int = 0) -> None:
    global _INTERPOLATION_PROCESS_EXECUTOR, _INTERPOLATION_PROCESS_MAX_WORKERS
    _INTERPOLATION_PROCESS_EXECUTOR = executor
    _INTERPOLATION_PROCESS_MAX_WORKERS = max(0, int(max_workers))

def gpu_worker_aux_interpolation_enabled() -> bool:
    return _env_flag('YOLO_TTA_GPU_WORKER_AUX_INTERPOLATION', True)

class _GpuWorkerAuxInterpolationPool:
    """Route post-drain interpolation passes to explicitly leased warm GPU workers.

    Each worker owns a targeted task queue. The scheduler exposes no lease until every inference
    result has been collected, so an auxiliary pass can never occupy an interpreter that may need
    to feed CUDA again. Submission remains non-blocking; unavailable workers fall back to the
    dedicated interpolation process pool.
    """

    def __init__(self, task_queues: Dict[int, object]) -> None:
        queues = {int(worker_id): task_queue for worker_id, task_queue in task_queues.items()}
        if not queues:
            raise ValueError('GPU-worker aux interpolation requires at least one task queue')
        self._task_queues = queues
        self._worker_ids = tuple(sorted(queues))
        self._capacity = len(self._worker_ids)
        self._lock = threading.Lock()
        self._pending: Dict[int, Dict[str, object]] = {}
        self._leased: Dict[int, bool] = {}
        self._busy_workers: set[int] = set()
        self._next_task_id = 2_000_000_000
        self._cursor = 0
        self._failed_reason: Optional[str] = None
        self._announced = False

    @property
    def capacity(self) -> int:
        return int(self._capacity)

    def enable_worker(self, worker_id: int, *, allow_full_cpu_affinity: bool = False) -> bool:
        worker = int(worker_id)
        with self._lock:
            if self._failed_reason is not None or worker not in self._task_queues:
                return False
            changed = worker not in self._leased or bool(self._leased[worker]) != bool(allow_full_cpu_affinity)
            self._leased[worker] = bool(allow_full_cpu_affinity)
            announce = changed and not self._announced
            if announce:
                self._announced = True
        if announce:
            print(
                'GPU-worker auxiliary interpolation leases active after global inference drain: '
                'warm CUDA-worker interpreters may now run tail interpolation.'
            )
        return True

    def revoke_worker(self, worker_id: int) -> bool:
        """Revoke an idle lease; False means an auxiliary pass is still running there."""
        worker = int(worker_id)
        with self._lock:
            self._leased.pop(worker, None)
            return worker not in self._busy_workers

    def outstanding(self) -> int:
        with self._lock:
            return int(len(self._pending))

    def mark_failed(self, reason: str) -> None:
        with self._lock:
            if self._failed_reason is None:
                self._failed_reason = str(reason)
            entries = list(self._pending.values())
            self._pending.clear()
            self._leased.clear()
            self._busy_workers.clear()
        for entry in entries:
            if not entry.get('error'):
                entry['error'] = str(reason)
            entry['event'].set()  # type: ignore[union-attr]

    def try_submit(self, aux_kwargs: Dict[str, object]) -> Optional[Dict[str, object]]:
        with self._lock:
            if self._failed_reason is not None:
                return None
            candidates = [
                worker_id for worker_id in self._worker_ids
                if worker_id in self._leased and worker_id not in self._busy_workers
            ]
            if not candidates:
                return None
            start = int(self._cursor) % len(self._worker_ids)
            rank = {worker_id: (self._worker_ids.index(worker_id) - start) % len(self._worker_ids) for worker_id in candidates}
            worker_id = min(candidates, key=lambda value: rank[value])
            self._cursor = (self._worker_ids.index(worker_id) + 1) % len(self._worker_ids)
            task_id = int(self._next_task_id)
            self._next_task_id += 1
            handle: Dict[str, object] = {
                'event': threading.Event(), 'stats': None, 'error': None,
                'task_id': task_id, 'worker_id': int(worker_id),
            }
            self._pending[task_id] = handle
            self._busy_workers.add(int(worker_id))
            allow_full_cpu_affinity = bool(self._leased.get(int(worker_id), False))
            task_queue = self._task_queues[int(worker_id)]
        try:
            envelope = {
                'task_id': int(task_id),
                'task_type': 'interpolation_pass',
                'aux_kwargs': dict(aux_kwargs),
                'allow_full_cpu_affinity': bool(allow_full_cpu_affinity),
            }
            preflight_multiprocessing_payload(envelope)
            task_queue.put(envelope)
        except Exception:
            with self._lock:
                self._pending.pop(task_id, None)
                self._busy_workers.discard(int(worker_id))
            return None
        return handle

    def complete(
        self,
        task_id: int,
        worker_id: int,
        ok: bool,
        stats: Optional[Dict[str, object]],
        error: Optional[str],
    ) -> None:
        with self._lock:
            handle = self._pending.pop(int(task_id), None)
            self._busy_workers.discard(int(worker_id))
        if handle is None:
            return
        if bool(ok):
            handle['stats'] = dict(stats or {})
        else:
            handle['error'] = str(error or 'unknown GPU-worker aux interpolation failure')
        handle['event'].set()  # type: ignore[union-attr]

    def wait(self, handle: Dict[str, object], poll_seconds: float = 5.0) -> Dict[str, object]:
        event: threading.Event = handle['event']  # type: ignore[assignment]
        while not event.wait(timeout=float(poll_seconds)):
            with self._lock:
                failed = self._failed_reason
            if failed is not None:
                raise RuntimeError(f'GPU worker aux interpolation pool failed: {failed}')
        err = handle.get('error')
        if err:
            raise RuntimeError(str(err))
        stats = handle.get('stats')
        if not isinstance(stats, dict):
            raise RuntimeError('GPU worker aux interpolation returned no stats')
        return dict(stats)

_GPU_WORKER_AUX_INTERPOLATION_POOL: Optional[_GpuWorkerAuxInterpolationPool] = None

def set_gpu_worker_aux_interpolation_pool(pool: Optional[_GpuWorkerAuxInterpolationPool]) -> None:
    global _GPU_WORKER_AUX_INTERPOLATION_POOL
    _GPU_WORKER_AUX_INTERPOLATION_POOL = pool

def gpu_worker_aux_interpolation_pool() -> Optional[_GpuWorkerAuxInterpolationPool]:
    return _GPU_WORKER_AUX_INTERPOLATION_POOL

def reset_runtime_state_for_new_run() -> None:
    """Settle process-global runtime state before an embedded pipeline invocation.

    The command-line launcher normally executes once, but tests and Python callers may run
    the pipeline repeatedly in one interpreter.  Do not let an executor, auxiliary lease,
    termination flag, scratch registration, or cached thread pool from an earlier failed
    invocation become part of the next run.
    """
    global _INTERPOLATION_PROCESS_EXECUTOR, _INTERPOLATION_PROCESS_MAX_WORKERS
    global _INTERPOLATION_PROCESS_WORKER, _GPU_WORKER_AUX_INTERPOLATION_POOL
    global _INTERPOLATION_PASS_ADMISSION
    global _RUN_SCRATCH_CLEANUP_PATH, _RUN_SCRATCH_CLEANUP_KEEP
    global _SCRATCH_DIR_IS_MEMORY_BACKED

    executor = _INTERPOLATION_PROCESS_EXECUTOR
    aux_pool = _GPU_WORKER_AUX_INTERPOLATION_POOL
    _INTERPOLATION_PROCESS_EXECUTOR = None
    _INTERPOLATION_PROCESS_MAX_WORKERS = 0
    _INTERPOLATION_PROCESS_WORKER = False
    _GPU_WORKER_AUX_INTERPOLATION_POOL = None
    _INTERPOLATION_PASS_ADMISSION = _InterpolationPassAdmission(1)

    if aux_pool is not None:
        try:
            aux_pool.mark_failed('pipeline run boundary')
        except Exception:
            pass
    if executor is not None:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=True)

    _cleanup_registered_unique_run_scratch()
    with _RUN_SCRATCH_CLEANUP_LOCK:
        _RUN_SCRATCH_CLEANUP_PATH = None
        _RUN_SCRATCH_CLEANUP_KEEP = True
    _SCRATCH_DIR_IS_MEMORY_BACKED = False
    _RUN_TERMINATION_REQUESTED.clear()
    shutdown_parallel_pool_cache()
    # The optional native DSA binding owns process-local work-queue state. Importing
    # its Python control plane here does not load the native module, and close_manager
    # is a no-op unless a prior copy selected DSA.
    try:
        from . import intel_dsa
        intel_dsa.close_manager()
        runtime_telemetry().gauge('workspace.copy.dsa.buffers_quarantined', False)
    except Exception as exc:
        runtime_telemetry().fallback('workspace.copy.dsa.close', exc)

def _interpolation_array_backing_path(arr: object) -> Optional[Path]:
    if arr is None:
        return None
    try:
        arr_np = np.asarray(arr)
        memfd_path = _memfd_backing_path_from_array(arr)
        if memfd_path is not None and bool(arr_np.flags['C_CONTIGUOUS']):
            if memfd_path.stat().st_size >= int(arr_np.nbytes):
                return memfd_path
        if (
            isinstance(arr, np.memmap)
            # A sliced np.memmap is still an np.memmap and inherits the parent's
            # `offset` attribute verbatim (NOT adjusted for the slice), so offset==0
            # alone cannot prove the array starts at byte 0 of the file. Only the
            # root mapping has the raw mmap.mmap buffer as its `base`; views chain
            # to the parent ndarray instead.
            and isinstance(getattr(arr, 'base', None), mmap.mmap)
            and bool(arr_np.flags['C_CONTIGUOUS'])
            and int(getattr(arr, 'offset', 0) or 0) == 0
        ):
            filename = getattr(arr, 'filename', None)
            if not filename:
                return None
            path = Path(filename)
            # The child reopens by (path, shape): the file must actually hold the
            # full array starting at byte 0.
            if path.stat().st_size < int(arr_np.nbytes):
                return None
            return path
    except Exception:
        pass
    # Deliberately do not reuse a base memmap for ndarray views here. A child
    # process would need the view's byte offset and strides to reopen it exactly;
    # interpolation volumes are expected to be full C-contiguous arrays, so views
    # are copied into a dedicated process memmap instead.
    return None

def _ensure_process_backed_interpolation_volume(
    mask_mm: np.ndarray,
    *,
    work_dir: Path,
    pass_tag: str,
    workers: int,
) -> Tuple[np.ndarray, Path, bool]:
    """Return a memmap-backed volume that a process worker can reopen by path."""
    arr = np.asarray(mask_mm)
    if arr.ndim != 3:
        raise ValueError(f'Interpolation expects a 3D mask volume, got shape {arr.shape}')
    if arr.dtype != np.dtype(np.uint8):
        raise ValueError(f'Interpolation process backend expects uint8 mask volume, got dtype {arr.dtype}')

    backing_path = _interpolation_array_backing_path(mask_mm)
    if backing_path is not None:
        flush_array(mask_mm)
        return mask_mm, Path(backing_path), False

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    process_path = work_dir / f'{_sanitize_filesystem_token(pass_tag)}.process_input.u8.dat'
    process_mm = copy_workspace_array(
        arr,
        process_path,
        desc=f'Interpolation process input {pass_tag}',
        prefer_memory=False,
        workers=int(workers),
    )
    flush_array(process_mm)
    return process_mm, process_path, True

def _interpolation_process_entry(
    *,
    mask_path: str,
    mask_shape: Tuple[int, int, int],
    mask_dtype: str,
    work_dir: str,
    pass_tag: str,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_min_radius: float,
    keep_temp: bool,
    reserve_bytes: int,
    workers: int,
    wrap_axis: bool,
    bridge_delta_path: Optional[str] = None,
    bridge_component_dir: Optional[str] = None,
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    # Local import keeps the package dependency graph acyclic.
    from .interpolation import interpolate_view_volume_pass_inplace

    global _INTERPOLATION_PROCESS_WORKER
    _INTERPOLATION_PROCESS_WORKER = True
    try:
        cv2.setNumThreads(int(interpolation_process_cv2_threads()))
    except Exception:
        pass

    mask_mm = np.memmap(
        Path(mask_path),
        dtype=np.dtype(mask_dtype),
        mode='r+',
        shape=tuple(int(x) for x in mask_shape),
    )
    try:
        stats = interpolate_view_volume_pass_inplace(
            mask_mm=mask_mm,
            work_dir=Path(work_dir),
            pass_tag=str(pass_tag),
            max_slice_distance=int(max_slice_distance),
            search_angle_deg=float(search_angle_deg),
            interpolation_walk_back=int(interpolation_walk_back),
            interpolation_candidates=int(interpolation_candidates),
            interpolate_min_radius=float(interpolate_min_radius),
            keep_temp=bool(keep_temp),
            prefer_memory=True,
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
            wrap_axis=bool(wrap_axis),
            bridge_delta_path=Path(bridge_delta_path) if bridge_delta_path else None,
            bridge_component_dir=(
                Path(bridge_component_dir) if bridge_component_dir else None
            ),
            known_slice_any=known_slice_any,
            known_slice_bboxes=known_slice_bboxes,
        )
        stats = dict(stats)
        stats.update({
            'process_backend': 'process_pool_memmap',
            'process_pid': int(os.getpid()),
            'process_workers_inside_pass': int(workers),
            'process_memmap_path': str(mask_path),
        })
        flush_array(mask_mm)
        return stats
    finally:
        close_memmap_array(mask_mm)

@_globally_admitted_interpolation_pass
def interpolate_view_volume_pass_maybe_process(
    mask_mm: np.ndarray,
    view: 'ViewInfo',
    work_dir: Path,
    pass_tag: str,
    max_slice_distance: int,
    search_angle_deg: float,
    interpolation_walk_back: int,
    interpolation_candidates: int,
    interpolate_min_radius: float,
    keep_temp: bool = False,
    prefer_memory: bool = True,
    reserve_bytes: int = 16 * GIB,
    workers: int = 1,
    bridge_delta_path: Optional[Path] = None,
    bridge_component_dir: Optional[Path] = None,
    known_slice_any: Optional[np.ndarray] = None,
    known_slice_bboxes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Run one interpolation pass with mandatory family-defined boundary semantics.

    Radial and Tilted Radial views always wrap across their angular slice seam. Cartesian
    and Tilted Cartesian views never wrap. This is derived from ``view`` and has no CLI or
    environment override. The returned array may be a new process-shareable memmap.
    """
    # Local import keeps the package dependency graph acyclic.
    from .interpolation import interpolate_view_volume_pass_inplace
    from .assembly import view_interpolation_wrap_axis

    wrap_axis = view_interpolation_wrap_axis(view)
    executor = _INTERPOLATION_PROCESS_EXECUTOR
    if (
        _INTERPOLATION_PROCESS_WORKER
        or executor is None
        or not interpolation_process_backend_enabled()
    ):
        stats = interpolate_view_volume_pass_inplace(
            mask_mm=mask_mm,
            work_dir=work_dir,
            pass_tag=pass_tag,
            max_slice_distance=int(max_slice_distance),
            search_angle_deg=float(search_angle_deg),
            interpolation_walk_back=int(interpolation_walk_back),
            interpolation_candidates=int(interpolation_candidates),
            interpolate_min_radius=float(interpolate_min_radius),
            keep_temp=bool(keep_temp),
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
            wrap_axis=bool(wrap_axis),
            bridge_delta_path=bridge_delta_path,
            bridge_component_dir=bridge_component_dir,
            known_slice_any=known_slice_any,
            known_slice_bboxes=known_slice_bboxes,
        )
        stats = dict(stats)
        stats.setdefault('process_backend', 'disabled_or_unconfigured')
        return mask_mm, stats

    process_mm, process_path, copied_to_memmap = _ensure_process_backed_interpolation_volume(
        mask_mm,
        work_dir=Path(work_dir),
        pass_tag=str(pass_tag),
        workers=max(1, min(int(workers), int(mask_mm.shape[0]) if getattr(mask_mm, 'ndim', 0) else int(workers))),
    )
    fallback_enabled = bool(interpolation_process_fallback_enabled())
    worker_mm = process_mm
    worker_path = Path(process_path)
    staged_bridge_path: Optional[Path] = None
    staged_component_dir: Optional[Path] = None

    # A failed interpolation pass is allowed to have modified any byte before it raises.
    # When recovery is enabled, isolate that speculative write set in a private pathname
    # transaction.  The clean input is retained for the in-process retry, and an auxiliary
    # worker whose transport failed can continue touching only the abandoned transaction.
    if fallback_enabled:
        transaction_token = (
            f'{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}'
        )
        worker_path = Path(work_dir) / (
            f'{_sanitize_filesystem_token(pass_tag)}.{transaction_token}.fallback-stage.u8.dat'
        )
        worker_mm = copy_workspace_array(
            np.asarray(process_mm),
            worker_path,
            desc=f'Interpolation fallback transaction {pass_tag}',
            prefer_memory=False,
            workers=max(
                1,
                min(
                    int(workers),
                    int(mask_mm.shape[0]) if getattr(mask_mm, 'ndim', 0) else int(workers),
                ),
            ),
        )
        if bridge_delta_path is not None:
            staged_bridge_path = Path(work_dir) / (
                f'{_sanitize_filesystem_token(pass_tag)}.{transaction_token}.bridge-stage.u8.dat'
            )
        if bridge_component_dir is not None:
            staged_component_dir = Path(work_dir) / (
                f'{_sanitize_filesystem_token(pass_tag)}.{transaction_token}.component-stage'
            )

    shape = tuple(int(x) for x in np.asarray(worker_mm).shape)
    dtype_str = str(np.asarray(worker_mm).dtype)
    flush_array(process_mm)
    if worker_mm is not process_mm:
        flush_array(worker_mm)

    def _discard_speculative_worker_storage() -> None:
        if worker_mm is process_mm:
            return
        close_memmap_array(worker_mm)
        if not bool(keep_temp):
            try:
                Path(worker_path).unlink(missing_ok=True)
            except Exception:
                pass
            if staged_bridge_path is not None:
                try:
                    Path(staged_bridge_path).unlink(missing_ok=True)
                except Exception:
                    pass
            if staged_component_dir is not None:
                shutil.rmtree(staged_component_dir, ignore_errors=True)

    def _commit_speculative_worker_storage(stats: Dict[str, object]) -> np.ndarray:
        try:
            if worker_mm is not process_mm:
                total_slices = int(np.asarray(process_mm).shape[0])

                def _commit_slice(idx: int) -> None:
                    np.copyto(process_mm[int(idx)], worker_mm[int(idx)])

                parallel_for_indices(
                    total_slices,
                    _commit_slice,
                    max_workers=max(1, min(int(workers), total_slices)),
                    desc=f'Committing interpolation transaction {pass_tag}',
                    show_progress=False,
                )
                flush_array(process_mm)
            if staged_bridge_path is not None and bridge_delta_path is not None:
                if Path(staged_bridge_path).exists():
                    Path(bridge_delta_path).parent.mkdir(parents=True, exist_ok=True)
                    os.replace(Path(staged_bridge_path), Path(bridge_delta_path))
                    stats['bridge_delta_path'] = str(bridge_delta_path)
            if staged_component_dir is not None and bridge_component_dir is not None:
                target_root = Path(bridge_component_dir)
                target_root.mkdir(parents=True, exist_ok=True)
                rewritten: List[Dict[str, object]] = []
                for raw_entry in stats.get('bridge_component_deltas', []):
                    entry = dict(raw_entry)
                    source_path = Path(str(entry.get('path', '')))
                    target_path = target_root / source_path.name
                    if target_path.exists():
                        shutil.rmtree(target_path, ignore_errors=True)
                    if source_path.exists():
                        os.replace(source_path, target_path)
                    entry['path'] = str(target_path)
                    rewritten.append(entry)
                stats['bridge_component_deltas'] = rewritten
        finally:
            _discard_speculative_worker_storage()
        return process_mm

    def _fallback_in_process(backend_name: str) -> Dict[str, object]:
        # Close/unlink only the parent's speculative mapping. A transport-failed auxiliary
        # process may still hold its own mapping, but it cannot race this clean base volume.
        # Local import keeps the package dependency graph acyclic.
        from .interpolation import interpolate_view_volume_pass_inplace

        _discard_speculative_worker_storage()
        fallback_stats = interpolate_view_volume_pass_inplace(
            mask_mm=process_mm,
            work_dir=work_dir,
            pass_tag=pass_tag,
            max_slice_distance=int(max_slice_distance),
            search_angle_deg=float(search_angle_deg),
            interpolation_walk_back=int(interpolation_walk_back),
            interpolation_candidates=int(interpolation_candidates),
            interpolate_min_radius=float(interpolate_min_radius),
            keep_temp=bool(keep_temp),
            prefer_memory=bool(prefer_memory),
            reserve_bytes=int(reserve_bytes),
            workers=int(workers),
            wrap_axis=bool(wrap_axis),
            bridge_delta_path=bridge_delta_path,
            bridge_component_dir=bridge_component_dir,
            known_slice_any=known_slice_any,
            known_slice_bboxes=known_slice_bboxes,
        )
        result = dict(fallback_stats)
        result['process_backend'] = str(backend_name)
        return result

    pass_entry_kwargs: Dict[str, object] = dict(
        mask_path=str(worker_path),
        mask_shape=shape,
        mask_dtype=dtype_str,
        work_dir=str(work_dir),
        pass_tag=str(pass_tag),
        max_slice_distance=int(max_slice_distance),
        search_angle_deg=float(search_angle_deg),
        interpolation_walk_back=int(interpolation_walk_back),
        interpolation_candidates=int(interpolation_candidates),
        interpolate_min_radius=float(interpolate_min_radius),
        keep_temp=bool(keep_temp),
        reserve_bytes=int(reserve_bytes),
        workers=int(workers),
        wrap_axis=bool(wrap_axis),
        bridge_delta_path=(
            str(staged_bridge_path)
            if staged_bridge_path is not None else
            (str(bridge_delta_path) if bridge_delta_path is not None else None)
        ),
        bridge_component_dir=(
            str(staged_component_dir)
            if staged_component_dir is not None else
            (str(bridge_component_dir) if bridge_component_dir is not None else None)
        ),
        # small per-slice metadata arrays pickle across the process boundary.
        known_slice_any=known_slice_any,
        known_slice_bboxes=known_slice_bboxes,
    )

    # once inference has drained, the warm GPU worker processes serve
    # interpolation passes too — try them first (non-blocking), then the dedicated pool.
    aux_pool = gpu_worker_aux_interpolation_pool()
    if aux_pool is not None:
        aux_handle = aux_pool.try_submit(pass_entry_kwargs)
        if aux_handle is not None:
            try:
                stats = dict(aux_pool.wait(aux_handle))
            except Exception as exc:
                if not fallback_enabled:
                    raise RuntimeError(
                        f'GPU-worker aux interpolation failed for {pass_tag} at {process_path}. '
                        'Set YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK=1 to rerun failed passes in-process for recovery.'
                    ) from exc
                print(
                    f'Warning: GPU-worker aux interpolation failed for {pass_tag} ({exc}); '
                    'falling back to in-process interpolation because YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK=1.'
                )
                stats = _fallback_in_process('fallback_in_process_after_aux_failure')
                result_mm = process_mm
            else:
                result_mm = _commit_speculative_worker_storage(stats)
                # _interpolation_process_entry is shared with the ordinary process pool
                # and stamps that transport by default. This branch knows the actual
                # successful host; numerical CUDA use is reported independently by
                # interpolation_render_backend/gpu_interpolation_active.
                stats['process_backend'] = 'gpu_worker_aux_process'
            stats['process_memmap_copied_from_anonymous_array'] = bool(copied_to_memmap)
            stats['process_pool_workers'] = int(_INTERPOLATION_PROCESS_MAX_WORKERS)
            flush_array(result_mm)
            return result_mm, stats

    try:
        # Submission itself may fail when the pool is broken or shutting down. Keep it in
        # the same recovery transaction as Future.result() rather than leaking a raw error.
        fut = executor.submit(_interpolation_process_entry, **pass_entry_kwargs)
        stats = dict(fut.result())
    except Exception as exc:
        if not fallback_enabled:
            raise RuntimeError(
                f'Interpolation process worker failed for {pass_tag} at {process_path}. '
                'Set YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK=1 to rerun this pass in-process for recovery.'
            ) from exc
        print(
            f'Warning: interpolation process worker failed for {pass_tag} ({exc}); '
            'falling back to legacy in-process interpolation because YOLO_TTA_INTERPOLATION_PROCESS_FALLBACK=1.'
        )
        stats = _fallback_in_process('fallback_in_process_after_worker_failure')
        result_mm = process_mm
    else:
        result_mm = _commit_speculative_worker_storage(stats)

    stats.setdefault('process_backend', 'process_pool_memmap')
    stats['process_memmap_copied_from_anonymous_array'] = bool(copied_to_memmap)
    stats['process_pool_workers'] = int(_INTERPOLATION_PROCESS_MAX_WORKERS)
    flush_array(result_mm)
    return result_mm, stats
