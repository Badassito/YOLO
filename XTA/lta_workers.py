"""Spawn-safe persistent GPU workers for production LTA execution.

The coordinator sends only small, pickle-safe task descriptions through the
queues.  A worker writes its potentially large prediction to an artifact,
then returns the path and SHA256 identity.  CUDA and SAM are deliberately not
imported here: each child first narrows ``CUDA_VISIBLE_DEVICES`` and only then
imports the configured adapter module.

Adapter modules expose three intentionally small seams::

    predictor = factory(adapter_config)              # once per worker
    output = execute(predictor, task_kind, payload)   # once per task
    shutdown(predictor)                               # optional, once

``output`` may be a path string or a primitive mapping with
``artifact_path`` plus optional ``metrics`` and ``metadata`` mappings.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import multiprocessing
import os
import pickle
import queue
import sys
import time
import traceback as traceback_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

from .lta_cpu import bind_worker_cpu_environment, resolve_worker_cpu_budget
from .lta_telemetry import LtaExecutionTrace


_SCHEMA = "lta.worker/1"
_EVENT_POLL_SECONDS = 0.05
_TRACEBACK_LIMIT = 64 * 1024
_UNSET = object()


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    return resolved


def _device_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("execution_device_id must be an integer")
    if value < 0:
        raise ValueError("execution_device_id must be non-negative")
    return int(value)


def _worker_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("worker_index must be an integer")
    if value < 0:
        raise ValueError("worker_index must be non-negative")
    return int(value)


def _resolve_visible_device_tokens(
    device_ids: Sequence[int],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> dict[int, str]:
    """Map parent-logical CUDA indexes to allocation-safe visibility tokens."""

    source = os.environ if environ is None else environ
    raw = source.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not str(raw).strip():
        return {int(device): str(int(device)) for device in device_ids}
    tokens = tuple(token.strip() for token in str(raw).split(",") if token.strip())
    resolved: dict[int, str] = {}
    for device in device_ids:
        logical = int(device)
        if logical >= len(tokens):
            raise ValueError(
                f"logical CUDA device {logical} is outside parent CUDA_VISIBLE_DEVICES "
                f"with {len(tokens)} token(s)"
            )
        resolved[logical] = tokens[logical]
    return resolved


def _copy_primitive(value: object, *, name: str) -> object:
    """Validate and detach one JSON-like, spawn-safe value.

    Requiring exact builtin containers prevents a nominal ``Mapping`` from
    smuggling module objects, locks, tensors, or custom reduction code into a
    spawned process.  Tuples are retained because coordinate and shape data
    naturally use them in scheduler payloads.
    """

    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is list:
        return [
            _copy_primitive(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is tuple:
        return tuple(
            _copy_primitive(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if type(value) is dict:
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            copied[key] = _copy_primitive(item, name=f"{name}.{key}")
        return copied
    raise TypeError(
        f"{name} must contain only None, bool, int, float, str, list, tuple, "
        f"and string-keyed dict values; got {type(value).__name__}"
    )


def _primitive_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    # Convert the outer mapping before the strict recursive check so ordinary
    # read-only Mapping implementations remain convenient at the public API.
    copied = _copy_primitive(dict(value), name=name)
    assert isinstance(copied, dict)
    return copied


@dataclass(frozen=True)
class LtaWorkerInit:
    """Run-constant, dependency-free adapter description sent to every child."""

    adapter_module: str
    adapter_factory: str
    adapter_execute: str
    adapter_shutdown: Optional[str] = None
    adapter_config: Mapping[str, object] = field(default_factory=dict)
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "adapter_module", _nonempty_text(self.adapter_module, name="adapter_module")
        )
        object.__setattr__(
            self, "adapter_factory", _nonempty_text(self.adapter_factory, name="adapter_factory")
        )
        object.__setattr__(
            self, "adapter_execute", _nonempty_text(self.adapter_execute, name="adapter_execute")
        )
        if self.adapter_shutdown is not None:
            object.__setattr__(
                self,
                "adapter_shutdown",
                _nonempty_text(self.adapter_shutdown, name="adapter_shutdown"),
            )
        object.__setattr__(
            self,
            "adapter_config",
            _primitive_mapping(self.adapter_config, name="adapter_config"),
        )
        if self.schema != _SCHEMA:
            raise ValueError(f"unsupported LTA worker schema {self.schema!r}")
        # Keep spawn failures deterministic and parent-side.
        pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)


@dataclass(frozen=True)
class LtaWorkerTask:
    """One atomic, retryable LTA operation routed to a specific GPU worker."""

    work_id: str
    attempt_token: str
    kind: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("work_id", "attempt_token", "kind"):
            object.__setattr__(
                self, name, _nonempty_text(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "payload", _primitive_mapping(self.payload, name="payload")
        )
        pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)


@dataclass(frozen=True)
class LtaWorkerReady:
    execution_device_id: int
    worker_pid: int
    visible_device: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    worker_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_index", _worker_index(self.worker_index))
        object.__setattr__(
            self,
            "execution_device_id",
            _device_id(self.execution_device_id),
        )
        if isinstance(self.worker_pid, bool) or int(self.worker_pid) <= 0:
            raise ValueError("worker_pid must be positive")
        object.__setattr__(self, "worker_pid", int(self.worker_pid))
        object.__setattr__(
            self,
            "visible_device",
            _nonempty_text(self.visible_device, name="visible_device"),
        )
        object.__setattr__(
            self,
            "metadata",
            _primitive_mapping(self.metadata, name="ready metadata"),
        )


@dataclass(frozen=True)
class LtaWorkerResult:
    work_id: str
    attempt_token: str
    kind: str
    execution_device_id: int
    worker_pid: int
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    metrics: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    worker_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_index", _worker_index(self.worker_index))
        for name in ("work_id", "attempt_token", "kind", "artifact_path"):
            object.__setattr__(
                self, name, _nonempty_text(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "execution_device_id", _device_id(self.execution_device_id)
        )
        if isinstance(self.worker_pid, bool) or int(self.worker_pid) <= 0:
            raise ValueError("worker_pid must be positive")
        object.__setattr__(self, "worker_pid", int(self.worker_pid))
        digest = _nonempty_text(self.artifact_sha256, name="artifact_sha256").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("artifact_sha256 must be a 64-character hexadecimal SHA256")
        object.__setattr__(self, "artifact_sha256", digest)
        if isinstance(self.artifact_size_bytes, bool) or int(self.artifact_size_bytes) < 0:
            raise ValueError("artifact_size_bytes must be non-negative")
        object.__setattr__(self, "artifact_size_bytes", int(self.artifact_size_bytes))
        object.__setattr__(
            self, "metrics", _primitive_mapping(self.metrics, name="metrics")
        )
        object.__setattr__(
            self, "metadata", _primitive_mapping(self.metadata, name="metadata")
        )


@dataclass(frozen=True)
class LtaWorkerError:
    execution_device_id: int
    worker_pid: int
    phase: str
    error_type: str
    message: str
    traceback: str
    fatal: bool
    work_id: Optional[str] = None
    attempt_token: Optional[str] = None
    kind: Optional[str] = None
    worker_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_index", _worker_index(self.worker_index))


LtaWorkerEvent = Union[LtaWorkerReady, LtaWorkerResult, LtaWorkerError]


class LtaWorkerPoolError(RuntimeError):
    """Base error raised by the parent-side worker coordinator."""


class LtaWorkerStartupError(LtaWorkerPoolError):
    def __init__(self, event: LtaWorkerError):
        self.event = event
        super().__init__(
            f"LTA worker cuda:{event.execution_device_id} slot:{event.worker_index} failed during startup: "
            f"{event.error_type}: {event.message}"
        )


class LtaWorkerExecutionError(LtaWorkerPoolError):
    def __init__(self, event: LtaWorkerError):
        self.event = event
        work = f" work {event.work_id!r}" if event.work_id is not None else ""
        super().__init__(
            f"LTA worker cuda:{event.execution_device_id} slot:{event.worker_index}{work} failed: "
            f"{event.error_type}: {event.message}"
        )


class LtaWorkerDiedError(LtaWorkerPoolError):
    def __init__(self, device_id: int, pid: Optional[int], exitcode: Optional[int], *, worker_index: int = 0):
        self.device_id = int(device_id)
        self.worker_index = _worker_index(worker_index)
        self.pid = pid
        self.exitcode = exitcode
        super().__init__(
            f"LTA worker cuda:{device_id} slot:{worker_index} (pid={pid}) exited unexpectedly "
            f"with code {exitcode}"
        )


class StaleLtaWorkerResult(LtaWorkerPoolError):
    def __init__(self, result: LtaWorkerResult, expected_token: Optional[str]):
        self.result = result
        self.expected_token = expected_token
        super().__init__(
            f"stale LTA result for work {result.work_id!r}: attempt "
            f"{result.attempt_token!r}, expected {expected_token!r}"
        )


class LtaWorkerShutdownError(LtaWorkerPoolError):
    pass


def validate_result_for_attempt(
    result: LtaWorkerResult,
    expected_token: Optional[str],
) -> LtaWorkerResult:
    """Return ``result`` only when it belongs to the coordinator's live lease."""

    if not isinstance(result, LtaWorkerResult):
        raise TypeError("result must be an LtaWorkerResult")
    if expected_token is None:
        raise StaleLtaWorkerResult(result, None)
    expected = _nonempty_text(expected_token, name="expected_token")
    if result.attempt_token != expected:
        raise StaleLtaWorkerResult(result, expected)
    return result


@dataclass(frozen=True)
class _StopWorker:
    pass


_WORKER_PREDICTOR: object = _UNSET
_WORKER_EXECUTE: object = _UNSET
_WORKER_SHUTDOWN: object = _UNSET


def _resolve_adapter_symbol(module: object, name: str, *, role: str) -> Callable[..., object]:
    value = getattr(module, name, None)
    if not callable(value):
        module_name = getattr(module, "__name__", type(module).__name__)
        raise TypeError(f"LTA adapter {module_name}.{name} for {role} is not callable")
    return value


def _load_adapter_once(
    init: LtaWorkerInit,
) -> Tuple[object, Callable[..., object], Optional[Callable[..., object]]]:
    """Import the selected adapter and construct its predictor exactly once.

    This function is the module-level lazy seam used by subprocesses.  The
    environment binding happens in :func:`_worker_main` before this function
    can import an adapter (and, transitively, Torch or SAM).
    """

    global _WORKER_PREDICTOR, _WORKER_EXECUTE, _WORKER_SHUTDOWN
    if _WORKER_PREDICTOR is not _UNSET:
        assert callable(_WORKER_EXECUTE)
        shutdown = None if _WORKER_SHUTDOWN is _UNSET else _WORKER_SHUTDOWN
        assert shutdown is None or callable(shutdown)
        return _WORKER_PREDICTOR, _WORKER_EXECUTE, shutdown

    module = importlib.import_module(init.adapter_module)
    factory = _resolve_adapter_symbol(module, init.adapter_factory, role="factory")
    execute = _resolve_adapter_symbol(module, init.adapter_execute, role="execute")
    shutdown: Optional[Callable[..., object]] = None
    if init.adapter_shutdown is not None:
        shutdown = _resolve_adapter_symbol(
            module, init.adapter_shutdown, role="shutdown"
        )
    predictor = factory(dict(init.adapter_config))
    _WORKER_PREDICTOR = predictor
    _WORKER_EXECUTE = execute
    _WORKER_SHUTDOWN = shutdown if shutdown is not None else _UNSET
    return predictor, execute, shutdown


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _normalize_adapter_output(
    output: object,
) -> tuple[str, dict[str, object], dict[str, object]]:
    if isinstance(output, os.PathLike):
        artifact_path = os.fspath(output)
        metrics: Mapping[str, object] = {}
        metadata: Mapping[str, object] = {}
    elif isinstance(output, str):
        artifact_path = output
        metrics = {}
        metadata = {}
    elif isinstance(output, Mapping):
        unknown = set(output) - {"artifact_path", "metrics", "metadata"}
        if unknown:
            raise ValueError(
                "LTA adapter output has unsupported keys: " + ", ".join(sorted(map(str, unknown)))
            )
        artifact_path = output.get("artifact_path")  # type: ignore[assignment]
        metrics = output.get("metrics", {})  # type: ignore[assignment]
        metadata = output.get("metadata", {})  # type: ignore[assignment]
    else:
        raise TypeError(
            "LTA adapter execute must return a path or a mapping with artifact_path"
        )
    if not isinstance(artifact_path, (str, os.PathLike)):
        raise TypeError("LTA adapter artifact_path must be a path string")
    path_text = _nonempty_text(os.fspath(artifact_path), name="artifact_path")
    return (
        path_text,
        _primitive_mapping(metrics, name="adapter metrics"),
        _primitive_mapping(metadata, name="adapter metadata"),
    )


def _error_event(
    device_id: int,
    *,
    phase: str,
    exc: BaseException,
    fatal: bool,
    task: Optional[LtaWorkerTask] = None,
    worker_index: int = 0,
) -> LtaWorkerError:
    rendered_traceback = "".join(
        traceback_module.format_exception(type(exc), exc, exc.__traceback__)
    )
    if len(rendered_traceback) > _TRACEBACK_LIMIT:
        rendered_traceback = rendered_traceback[-_TRACEBACK_LIMIT:]
    return LtaWorkerError(
        execution_device_id=int(device_id),
        worker_pid=os.getpid(),
        phase=str(phase),
        error_type=type(exc).__name__,
        message=str(exc),
        traceback=rendered_traceback,
        fatal=bool(fatal),
        work_id=None if task is None else task.work_id,
        attempt_token=None if task is None else task.attempt_token,
        kind=None if task is None else task.kind,
        worker_index=worker_index,
    )


def _worker_main(
    device_id: int,
    visible_device_token: str,
    init: LtaWorkerInit,
    task_queue: object,
    event_queue: object,
    cpu_budget: Optional[Mapping[str, object]] = None,
    worker_index: int = 0,
) -> None:
    """Child entry point; keep imports above dependency-free and CUDA-neutral."""

    resolved_device = _device_id(device_id)
    resolved_index = _worker_index(worker_index)
    # Physical device N is intentionally exposed as the worker's sole logical
    # cuda:0.  This must precede adapter import because SAM imports Torch.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device_token)
    os.environ["LTA_EXECUTION_DEVICE_ID"] = str(resolved_device)
    os.environ["LTA_WORKER_INDEX"] = str(resolved_index)
    budget = dict(cpu_budget or resolve_worker_cpu_budget(1))
    bind_worker_cpu_environment(budget)
    trace = LtaExecutionTrace(init.adapter_config.get("trace_dir"), f"worker_{resolved_device}_slot{resolved_index}")
    trace.event("process_start", device_id=resolved_device, visible_device=visible_device_token,
                worker_index=resolved_index, cpu_budget=budget)

    predictor: object = _UNSET
    shutdown: Optional[Callable[..., object]] = None
    try:
        prematurely_loaded = tuple(
            name
            for name in (init.adapter_module, "torch", "sam3")
            if name in sys.modules
        )
        prematurely_loaded += tuple(
            name
            for name in sys.modules
            if name.startswith("torch.") or name.startswith("sam3.")
        )
        if prematurely_loaded:
            examples = ", ".join(sorted(set(prematurely_loaded))[:8])
            raise RuntimeError(
                "CUDA-sensitive modules were imported before the LTA worker "
                f"bound CUDA_VISIBLE_DEVICES: {examples}"
            )
        with trace.phase("adapter_startup", device_id=resolved_device, worker_index=resolved_index):
            predictor, execute, shutdown = _load_adapter_once(init)
        trace.event("process_ready", device_id=resolved_device, worker_index=resolved_index,
                    cpu_runtime=getattr(predictor, "cpu_budget", {}))
        event_queue.put(
            LtaWorkerReady(
                execution_device_id=resolved_device,
                worker_pid=os.getpid(),
                visible_device=os.environ["CUDA_VISIBLE_DEVICES"],
                worker_index=resolved_index,
                metadata={
                    "profile": getattr(predictor, "profile", {}),
                    "sam_runtime": getattr(predictor, "sam_runtime", {}),
                    "cpu_budget": budget,
                    "cpu_runtime": getattr(predictor, "cpu_budget", {}),
                    "constrained_batches": getattr(
                        predictor,
                        "constrained_batches",
                        None,
                    ),
                },
            )
        )
    except Exception as exc:
        trace.event("process_failed", phase="startup", error_type=type(exc).__name__, message=str(exc))
        event_queue.put(
            _error_event(
                resolved_device, phase="startup", exc=exc, fatal=True, worker_index=resolved_index,
            )
        )
        trace.close()
        return

    try:
        while True:
            with trace.phase("queue_wait", device_id=resolved_device, worker_index=resolved_index):
                message = task_queue.get()
            if isinstance(message, _StopWorker):
                break
            if not isinstance(message, LtaWorkerTask):
                exc = TypeError(
                    f"worker received unsupported task type {type(message).__name__}"
                )
                event_queue.put(
                    _error_event(
                        resolved_device, phase="execute", exc=exc, fatal=False, worker_index=resolved_index,
                    )
                )
                continue
            try:
                with trace.phase("execute", device_id=resolved_device, work_id=message.work_id,
                                 task_kind=message.kind, worker_index=resolved_index):
                    output = execute(predictor, message.kind, dict(message.payload))
                path_text, metrics, metadata = _normalize_adapter_output(output)
                artifact = Path(path_text).expanduser().resolve()
                if not artifact.is_file():
                    raise FileNotFoundError(
                        f"LTA adapter artifact is not a regular file: {artifact}"
                    )
                with trace.phase("manifest_hash", work_id=message.work_id) as details:
                    digest, size = _sha256_file(artifact)
                    details["stored_bytes"] = size
                event_queue.put(
                    LtaWorkerResult(
                        work_id=message.work_id,
                        attempt_token=message.attempt_token,
                        kind=message.kind,
                        execution_device_id=resolved_device,
                        worker_pid=os.getpid(),
                        artifact_path=str(artifact),
                        artifact_sha256=digest,
                        artifact_size_bytes=size,
                        metrics=metrics,
                        metadata=metadata,
                        worker_index=resolved_index,
                    )
                )
            except Exception as exc:
                trace.event("task_failed", work_id=message.work_id, error_type=type(exc).__name__,
                            message=str(exc))
                event_queue.put(
                    _error_event(
                        resolved_device,
                        phase="execute",
                        exc=exc,
                        fatal=False,
                        task=message,
                        worker_index=resolved_index,
                    )
                )
    finally:
        if shutdown is not None and predictor is not _UNSET:
            try:
                with trace.phase("adapter_shutdown", device_id=resolved_device):
                    shutdown(predictor)
            except Exception as exc:
                event_queue.put(
                    _error_event(
                        resolved_device,
                        phase="shutdown",
                        exc=exc,
                        fatal=True,
                        worker_index=resolved_index,
                    )
                )
        trace.event("process_stopped", device_id=resolved_device, worker_index=resolved_index, trace_write_errors=trace.write_errors)
        trace.close()


class LtaWorkerPool:
    """Bounded persistent process slots on each selected physical CUDA device.

    The coordinator API is intentionally single-threaded, matching
    :class:`XTA.lta_scheduler.LtaViewAffinityScheduler` ownership.  Retries may
    reuse a work id only with a fresh attempt token.  A completion carrying an
    older token is rejected before it can be handed to the scheduler.
    """

    def __init__(
        self,
        device_ids: Sequence[int],
        init: LtaWorkerInit,
        *,
        startup_timeout: Optional[float] = 120.0,
        workers_per_device: int = 1,
    ) -> None:
        if not isinstance(init, LtaWorkerInit):
            raise TypeError("init must be an LtaWorkerInit")
        devices = tuple(_device_id(value) for value in device_ids)
        if not devices:
            raise ValueError("device_ids must contain at least one CUDA device")
        if len(devices) != len(set(devices)):
            raise ValueError("device_ids must be unique")
        count = _worker_index(workers_per_device)
        if not 1 <= count <= 4:
            raise ValueError("workers_per_device must be between one and four")
        self.device_ids = devices
        self.workers_per_device = count
        self.worker_slots = tuple((device, index) for device in devices for index in range(count))
        self.cpu_budget = resolve_worker_cpu_budget(len(self.worker_slots))
        self.visible_device_tokens = _resolve_visible_device_tokens(devices)
        # Detach the exposed mapping again at the actual spawn boundary in
        # case a caller mutated the (necessarily pickleable) dataclass mapping
        # after construction.
        safe_init = LtaWorkerInit(
            adapter_module=init.adapter_module,
            adapter_factory=init.adapter_factory,
            adapter_execute=init.adapter_execute,
            adapter_shutdown=init.adapter_shutdown,
            adapter_config=init.adapter_config,
            schema=init.schema,
        )
        self.init = safe_init
        if safe_init.adapter_config.get("trace_dir"):
            print(
                "LTA worker CPU allocation: "
                f"devices={list(devices)} workers_per_device={count} allocated_cpus={self.cpu_budget['effective_cpu_count']} "
                f"native_threads_per_worker={self.cpu_budget['threads_per_worker']}",
                flush=True,
            )
        self.start_method = "spawn"
        self._context = multiprocessing.get_context(self.start_method)
        self._event_queue = self._context.Queue()
        self._task_queues = {
            slot: self._context.Queue() for slot in self.worker_slots
        }
        self._processes = {}
        self._expected_attempts: dict[str, str] = {}
        self._seen_attempts: set[tuple[str, str]] = set()
        self._attempt_routes: dict[tuple[str, str], tuple[tuple[int, int], str]] = {}
        self._closed = False
        self._closing = False
        self._atexit_registered = False
        self.ready_events: tuple[LtaWorkerReady, ...] = ()
        try:
            for device, index in self.worker_slots:
                process = self._context.Process(
                    target=_worker_main,
                    args=(
                        device,
                        self.visible_device_tokens[device],
                        safe_init,
                        self._task_queues[(device, index)],
                        self._event_queue,
                        self.cpu_budget,
                        index,
                    ),
                    name=f"lta-gpu-{device}-slot-{index}",
                    # SAM adapters remain free to create their own decoder or
                    # loader subprocesses.  The pool's atexit hook and explicit
                    # forced cleanup own termination instead of daemon rules.
                    daemon=False,
                )
                process.start()
                self._processes[(device, index)] = process
            self.ready_events = self._await_ready(startup_timeout)
            atexit.register(self._atexit_shutdown)
            self._atexit_registered = True
        except BaseException:
            self.shutdown(timeout=1.0, force=True)
            raise

    @property
    def pids(self) -> Mapping[int, int]:
        """Legacy replica-zero PID mapping; use pids_by_slot for every process."""
        return {
            device: int(process.pid)
            for (device, index), process in self._processes.items()
            if index == 0 and process.pid is not None
        }

    @property
    def pids_by_slot(self) -> Mapping[tuple[int, int], int]:
        return {
            slot: int(process.pid) for slot, process in self._processes.items()
            if process.pid is not None
        }

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("LTA worker pool is closed")

    def is_alive(self, execution_device_id: int, worker_index: int = 0) -> bool:
        slot = _device_id(execution_device_id), _worker_index(worker_index)
        if slot not in self._processes:
            raise ValueError(f"unknown LTA worker slot {slot}")
        return bool(self._processes[slot].is_alive())

    def check_liveness(self) -> None:
        """Raise immediately when any worker has left the running pool."""

        self._ensure_open()
        for device, index in self.worker_slots:
            process = self._processes[(device, index)]
            if not process.is_alive():
                # Refresh exitcode on platforms where is_alive joins a just-
                # exited child lazily.
                process.join(timeout=0)
                raise LtaWorkerDiedError(device, process.pid, process.exitcode, worker_index=index)

    def _validate_event_worker(self, event: LtaWorkerEvent) -> None:
        slot = event.execution_device_id, event.worker_index
        process = self._processes.get(slot)
        if process is None:
            raise LtaWorkerPoolError(f"event names unknown LTA worker slot {slot}")
        if event.worker_pid != process.pid:
            raise LtaWorkerPoolError(
                f"LTA worker slot {slot} event PID {event.worker_pid} differs from {process.pid}"
            )

    def _validate_attempt_route(self, event: LtaWorkerResult | LtaWorkerError) -> None:
        identity = event.work_id, event.attempt_token
        route = self._attempt_routes.get(identity)
        actual = (event.execution_device_id, event.worker_index), event.kind
        if route is None or actual != route:
            raise LtaWorkerPoolError(
                f"LTA work attempt {identity} arrived through route {actual}; expected {route}"
            )

    def _get_event(self, timeout: Optional[float]) -> LtaWorkerEvent:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            if deadline is None:
                slice_timeout = _EVENT_POLL_SECONDS
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    try:
                        event = self._event_queue.get_nowait()
                    except queue.Empty:
                        self.check_liveness()
                        raise TimeoutError("timed out waiting for an LTA worker event")
                    if not isinstance(
                        event, (LtaWorkerReady, LtaWorkerResult, LtaWorkerError)
                    ):
                        raise LtaWorkerPoolError(
                            "received unsupported LTA worker event "
                            f"{type(event).__name__}"
                        )
                    self._validate_event_worker(event)
                    return event
                slice_timeout = min(_EVENT_POLL_SECONDS, remaining)
            try:
                event = self._event_queue.get(timeout=slice_timeout)
            except queue.Empty:
                self.check_liveness()
                continue
            if not isinstance(event, (LtaWorkerReady, LtaWorkerResult, LtaWorkerError)):
                raise LtaWorkerPoolError(
                    f"received unsupported LTA worker event {type(event).__name__}"
                )
            self._validate_event_worker(event)
            return event

    def _await_ready(self, timeout: Optional[float]) -> tuple[LtaWorkerReady, ...]:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        ready: dict[tuple[int, int], LtaWorkerReady] = {}
        while len(ready) < len(self.worker_slots):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                event = self._get_event(remaining)
            except LtaWorkerDiedError as exc:
                raise LtaWorkerStartupError(
                    LtaWorkerError(
                        execution_device_id=exc.device_id,
                        worker_pid=exc.pid or 0,
                        phase="startup",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        traceback="",
                        fatal=True,
                        worker_index=exc.worker_index,
                    )
                ) from exc
            if isinstance(event, LtaWorkerError):
                raise LtaWorkerStartupError(event)
            if not isinstance(event, LtaWorkerReady):
                raise LtaWorkerStartupError(
                    LtaWorkerError(
                        execution_device_id=event.execution_device_id,
                        worker_pid=event.worker_pid,
                        phase="startup",
                        error_type="ProtocolError",
                        message="worker returned a result before becoming ready",
                        traceback="",
                        fatal=True,
                        worker_index=event.worker_index,
                    )
                )
            slot = event.execution_device_id, event.worker_index
            if slot not in self.worker_slots:
                raise LtaWorkerPoolError(
                    f"unknown worker cuda:{event.execution_device_id} became ready"
                )
            if slot in ready:
                raise LtaWorkerPoolError(
                    f"worker cuda:{event.execution_device_id} slot:{event.worker_index} became ready twice"
                )
            expected_visible = self.visible_device_tokens[event.execution_device_id]
            if event.visible_device != expected_visible:
                raise LtaWorkerPoolError(
                    f"worker cuda:{event.execution_device_id} exposed CUDA_VISIBLE_DEVICES="
                    f"{event.visible_device!r}; expected {expected_visible!r}"
                )
            ready[slot] = event
        return tuple(ready[slot] for slot in self.worker_slots)

    def submit(
        self,
        task: LtaWorkerTask,
        *,
        execution_device_id: int,
        worker_index: int = 0,
    ) -> None:
        """Queue ``task`` on one isolated slot of the selected physical device."""

        self._ensure_open()
        if not isinstance(task, LtaWorkerTask):
            raise TypeError("task must be an LtaWorkerTask")
        device = _device_id(execution_device_id)
        slot = device, _worker_index(worker_index)
        if slot not in self._task_queues:
            raise ValueError(f"unknown LTA worker slot {slot}")
        self.check_liveness()
        # Detach mutable caller dictionaries once more at the actual queue
        # boundary.  This also provides a final pickle-safety assertion.
        queued = LtaWorkerTask(
            work_id=task.work_id,
            attempt_token=task.attempt_token,
            kind=task.kind,
            payload=task.payload,
        )
        attempt_identity = (queued.work_id, queued.attempt_token)
        if attempt_identity in self._seen_attempts:
            raise ValueError(
                f"LTA attempt token {queued.attempt_token!r} was already submitted "
                f"for work {queued.work_id!r}"
            )
        self._seen_attempts.add(attempt_identity)
        self._expected_attempts[queued.work_id] = queued.attempt_token
        self._attempt_routes[attempt_identity] = slot, queued.kind
        self._task_queues[slot].put(queued)

    def wait_event(self, timeout: Optional[float] = None) -> LtaWorkerEvent:
        """Return the next raw event; primarily useful for diagnostics."""

        self._ensure_open()
        return self._get_event(timeout)

    def wait_result(self, timeout: Optional[float] = None) -> LtaWorkerResult:
        """Wait for one current-attempt completion or raise a typed worker error."""

        self._ensure_open()
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            event = self._get_event(remaining)
            if isinstance(event, LtaWorkerReady):
                raise LtaWorkerPoolError(
                    f"worker cuda:{event.execution_device_id} slot:{event.worker_index} emitted a duplicate ready event"
                )
            if isinstance(event, LtaWorkerError):
                if event.work_id is not None:
                    self._validate_attempt_route(event)
                    expected = self._expected_attempts.get(event.work_id)
                    if event.attempt_token != expected:
                        # A failed old lease must not fail its replacement.
                        continue
                    if expected == event.attempt_token:
                        self._expected_attempts.pop(event.work_id, None)
                raise LtaWorkerExecutionError(event)
            expected = self._expected_attempts.get(event.work_id)
            validate_result_for_attempt(event, expected)
            self._validate_attempt_route(event)
            self._expected_attempts.pop(event.work_id, None)
            return event

    def shutdown(self, *, timeout: float = 10.0, force: bool = True) -> tuple[int, ...]:
        """Request orderly adapter shutdown, then terminate lingering children.

        Returns unique physical execution-device ids that required forced termination.
        Calling this method more than once is harmless.
        """

        if self._closed:
            return ()
        self._closing = True
        for slot, process in self._processes.items():
            if process.is_alive():
                try:
                    self._task_queues[slot].put(_StopWorker())
                except Exception:
                    pass

        deadline = time.monotonic() + max(0.0, float(timeout))
        for process in self._processes.values():
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)

        forced = []
        for slot, process in self._processes.items():
            if process.is_alive():
                forced.append(slot)
                if force:
                    process.terminate()
        if forced and force:
            for slot in forced:
                process = self._processes[slot]
                process.join(timeout=2.0)
                if process.is_alive():
                    kill = getattr(process, "kill", None)
                    if callable(kill):
                        kill()
                        process.join(timeout=2.0)

        lingering = [
            slot
            for slot, process in self._processes.items()
            if process.is_alive()
        ]
        shutdown_errors: list[LtaWorkerError] = []
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, LtaWorkerError) and event.phase == "shutdown":
                shutdown_errors.append(event)
        for task_queue in self._task_queues.values():
            try:
                task_queue.close()
                task_queue.join_thread()
            except Exception:
                pass
        try:
            self._event_queue.close()
            self._event_queue.join_thread()
        except Exception:
            pass
        self._closed = True
        self._closing = False
        if self._atexit_registered:
            try:
                atexit.unregister(self._atexit_shutdown)
            except Exception:
                pass
            self._atexit_registered = False
        if lingering or (forced and not force):
            unresolved = lingering if lingering else forced
            raise LtaWorkerShutdownError(
                f"LTA workers did not exit cleanly: slots={unresolved}"
            )
        if shutdown_errors:
            detail = "; ".join(
                f"cuda:{event.execution_device_id} slot:{event.worker_index} {event.error_type}: {event.message}"
                for event in shutdown_errors
            )
            raise LtaWorkerShutdownError(f"LTA worker adapter shutdown failed: {detail}")
        return tuple(device for device in self.device_ids if any(slot[0] == device for slot in forced))

    def close(self) -> None:
        self.shutdown()

    def _atexit_shutdown(self) -> None:
        try:
            self.shutdown(timeout=0.25, force=True)
        except Exception:
            pass

    def __enter__(self) -> "LtaWorkerPool":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()


__all__ = (
    "LtaWorkerDiedError",
    "LtaWorkerError",
    "LtaWorkerEvent",
    "LtaWorkerExecutionError",
    "LtaWorkerInit",
    "LtaWorkerPool",
    "LtaWorkerPoolError",
    "LtaWorkerReady",
    "LtaWorkerResult",
    "LtaWorkerShutdownError",
    "LtaWorkerStartupError",
    "LtaWorkerTask",
    "StaleLtaWorkerResult",
    "validate_result_for_attempt",
)
