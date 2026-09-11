"""Allocation-aware CPU budgets for the independent LTA GPU workers."""

from __future__ import annotations

import os
import re
import sys
from functools import wraps
from typing import Mapping


_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _positive_prefix(value: object) -> int | None:
    match = re.match(r"^\s*([0-9]+)(?:\s*$|\s*\(|\s*,)", str(value or ""))
    if match is None:
        return None
    number = int(match.group(1))
    return number if number > 0 else None


def resolve_worker_cpu_budget(
    worker_count: int,
    *,
    environ: Mapping[str, str] | None = None,
    affinity_count: int | None = None,
    cpu_count: int | None = None,
) -> dict[str, object]:
    """Intersect affinity and Slurm limits, then divide the available threads.

    Affinity can expose an entire node even when Slurm grants fewer CPUs. An
    explicit lower native-thread limit is respected. The small per-frame CPU
    kernels do not need a whole-node OpenMP pool per GPU; cap each at four.
    """
    if isinstance(worker_count, bool) or int(worker_count) < 1:
        raise ValueError("worker_count must be positive")
    workers = int(worker_count)
    source = os.environ if environ is None else environ
    hardware = max(1, int(os.cpu_count() or 1) if cpu_count is None else int(cpu_count))
    if affinity_count is None and environ is None:
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pass
    constraints = {"os_cpu_count": hardware}
    if affinity_count is not None and int(affinity_count) > 0:
        constraints["process_affinity"] = int(affinity_count)
    for variable in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        value = _positive_prefix(source.get(variable))
        if value is not None:
            constraints[variable] = value
    effective = min(constraints.values())
    # Leave a coordinator CPU when the allocation can afford it.
    available = effective - 1 if effective > workers else effective
    threads = min(4, max(1, available // workers))
    inherited = [value for variable in _NATIVE_THREAD_VARIABLES
                 if (value := _positive_prefix(source.get(variable))) is not None]
    if inherited:
        threads = min(threads, min(inherited))
    return {
        "effective_cpu_count": effective,
        "worker_count": workers,
        "threads_per_worker": threads,
        "coordinator_cpu_reserve": max(0, effective - workers * threads),
        "minimum_worker_threads_exceed_allocation": workers > effective,
        "constraints": constraints,
        "inherited_thread_limit": min(inherited) if inherited else None,
        "policy": "allocation_intersection_shared_across_gpu_workers",
    }


def bind_worker_cpu_environment(budget: Mapping[str, object]) -> None:
    """Apply only inside the spawned child, before numerical-library imports."""
    threads = int(budget["threads_per_worker"])
    if threads < 1:
        raise ValueError("worker CPU thread count must be positive")
    for variable in _NATIVE_THREAD_VARIABLES:
        os.environ[variable] = str(threads)
    os.environ["LTA_CPU_THREADS"] = str(threads)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"


def configure_worker_runtime_threads(torch_module: object, cv2_module: object | None = None) -> dict[str, object]:
    """Set actual Torch/OpenCV limits before model construction in the child."""
    threads = _positive_prefix(os.environ.get("LTA_CPU_THREADS"))
    if threads is None:
        # Direct diagnostic factory calls still get a bounded allocation-aware
        # budget, without modifying the caller's environment.
        threads = int(resolve_worker_cpu_budget(1)["threads_per_worker"])
    torch_module.set_num_threads(threads)
    interop_error = None
    if int(torch_module.get_num_interop_threads()) != 1:
        try:
            torch_module.set_num_interop_threads(1)
        except RuntimeError as error:
            # A custom diagnostic may already have initialized inter-op work.
            # Preserve execution and expose the actual limit instead of lying.
            interop_error = str(error)
    if cv2_module is not None:
        cv2_module.setNumThreads(threads)
    return {
        "requested_native_threads": threads,
        "torch_num_threads": int(torch_module.get_num_threads()),
        "torch_num_interop_threads": int(torch_module.get_num_interop_threads()),
        "opencv_num_threads": None if cv2_module is None else int(cv2_module.getNumThreads()),
        "interop_configuration_error": interop_error,
        "native_environment": {key: os.environ.get(key) for key in _NATIVE_THREAD_VARIABLES},
    }


def host_native_thread_counts() -> dict[str, int | None]:
    result = {}
    for name, getter in (("cv2", "getNumThreads"), ("torch", "get_num_threads")):
        method = getattr(sys.modules.get(name), getter, None)
        result[name] = int(method()) if callable(method) else None
    return result


def bounded_lta_host_threads(function):
    """Limit nested native pools while LTA's explicit CPU workers share the node.

    OpenCV's default whole-node pool can otherwise compete with every GPU
    worker or multiply an explicit ThreadPoolExecutor's parallelism. Restore
    the embedding process's original library settings on every exit.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        import cv2
        libraries = [(cv2, "getNumThreads", "setNumThreads")]
        torch = sys.modules.get("torch")
        if torch is not None and callable(getattr(torch, "get_num_threads", None)):
            libraries.append((torch, "get_num_threads", "set_num_threads"))
        previous = []
        try:
            for module, getter, setter in libraries:
                old = int(getattr(module, getter)())
                previous.append((module, setter, old))
                getattr(module, setter)(1)
            return function(*args, **kwargs)
        finally:
            for module, setter, old in reversed(previous):
                getattr(module, setter)(old)
    return wrapped


__all__ = ("resolve_worker_cpu_budget", "bind_worker_cpu_environment", "configure_worker_runtime_threads",
           "bounded_lta_host_threads", "host_native_thread_counts")
