"""Hardware-gated smoke test and microbenchmark for optional Intel engines.

Nothing imports or probes an accelerator unless the operator names it with
``--backend``.  The ordinary test suite uses injectable fakes instead; run this
tool on the production Linux allocation after QATzip/QPL and the idxd work
queues have been provisioned.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _payload(size: int) -> bytes:
    seed = bytes(range(251)) + b"volume-tta-intel-selftest\x00"
    return (seed * ((int(size) + len(seed) - 1) // len(seed)))[: int(size)]


def _compression_backend(name: str, size: int, iterations: int, level: int) -> Dict[str, object]:
    from volume_tta import intel_compression

    capabilities = intel_compression.probe_capabilities(name)
    supported = tuple(int(value) for value in capabilities.get("supported_levels", (level,)))
    if int(level) not in supported:
        raise RuntimeError(
            f"{name} level {level} is unsupported; binding reports {supported}"
        )
    compressor = intel_compression.create_gzip_compressor(
        name,
        int(level),
        capabilities=capabilities,
    )
    payload_size = max(int(size), int(compressor.minimum_input_bytes))
    payload = _payload(payload_size)
    compressor.preflight_thread_state()
    output_bytes = 0
    started = time.perf_counter()
    try:
        for _ in range(int(iterations)):
            encoded = compressor(memoryview(payload))
            if gzip.decompress(encoded) != payload:
                raise RuntimeError(f"{name} gzip round trip changed the payload")
            output_bytes += len(encoded)
    finally:
        intel_compression.close_current_thread_state()
    elapsed = max(time.perf_counter() - started, 1e-12)
    native = intel_compression.native_stats(name, reset=False)
    if int(native.get("software_fallback_requests", 0)) != 0:
        raise RuntimeError(f"{name} reported software fallback: {native}")
    total_input = int(payload_size) * int(iterations)
    return {
        "backend": name,
        "capabilities": capabilities,
        "input_bytes": total_input,
        "output_bytes": int(output_bytes),
        "iterations": int(iterations),
        "elapsed_seconds": elapsed,
        "gib_per_second": total_input / (1024 ** 3) / elapsed,
        "compression_ratio": output_bytes / max(total_input, 1),
        "native_stats": native,
    }


def _dsa_backend(size: int, iterations: int) -> Dict[str, object]:
    import numpy as np

    from volume_tta import intel_dsa

    manager = intel_dsa.get_manager()
    capabilities = manager.capabilities(work_queue=intel_dsa.requested_work_queue())
    source = np.frombuffer(_payload(int(size)), dtype=np.uint8).copy()
    destination = np.empty_like(source)
    results = []
    started = time.perf_counter()
    try:
        for _ in range(int(iterations)):
            destination.fill(0)
            stats = manager.copy(
                source,
                destination,
                capabilities=capabilities,
                max_inflight=min(
                    int(intel_dsa.requested_max_inflight()),
                    int(capabilities.get("max_inflight", 1)),
                ),
            )
            if not np.array_equal(destination, source):
                raise RuntimeError("DSA copy changed the payload")
            results.append(stats)
    finally:
        intel_dsa.close_manager()
    elapsed = max(time.perf_counter() - started, 1e-12)
    total_input = int(source.nbytes) * int(iterations)
    return {
        "backend": "dsa",
        "capabilities": capabilities,
        "input_bytes": total_input,
        "iterations": int(iterations),
        "elapsed_seconds": elapsed,
        "gib_per_second": total_input / (1024 ** 3) / elapsed,
        "native_results": results,
    }


def _selected_backends(value: str) -> Iterable[str]:
    if value == "all":
        return ("qat", "iaa", "dsa")
    return (value,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=("qat", "iaa", "dsa", "all"),
        help="hardware engine to probe; no engine is selected implicitly",
    )
    parser.add_argument("--size-mib", type=float, default=16.0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--qat-level", type=int, default=1)
    parser.add_argument("--iaa-level", type=int, default=1)
    args = parser.parse_args()
    if not (args.size_mib > 0.0):
        parser.error("--size-mib must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    size = max(1, int(float(args.size_mib) * 1024 * 1024))
    report = {
        "pid": os.getpid(),
        "platform": sys.platform,
        "results": [],
    }
    for backend in _selected_backends(str(args.backend)):
        if backend == "dsa":
            result = _dsa_backend(size, int(args.iterations))
        else:
            level = args.qat_level if backend == "qat" else args.iaa_level
            result = _compression_backend(
                backend,
                size,
                int(args.iterations),
                int(level),
            )
        report["results"].append(result)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
