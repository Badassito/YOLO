"""Unprivileged CUDA-IPC/NVLink smoke test for v18.0.3 D1 owner groups.

Every spawned child narrows CUDA_VISIBLE_DEVICES to one inherited token, allocates the
same dedicated cudaMalloc-backed uint32 bitset used by D1 groups, and keeps it alive while
rank zero imports and OR-reduces the peer handles.  This isolates the most deployment-
sensitive part of the group path without loading a model or production volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _tokens(requested: int) -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return [str(index) for index in range(int(requested))]
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    if len(tokens) < int(requested):
        raise RuntimeError(
            f"requested {requested} GPUs but CUDA_VISIBLE_DEVICES exposes {len(tokens)}"
        )
    return tokens[: int(requested)]


def _pattern(rank: int, words: int) -> np.ndarray:
    indices = np.arange(int(words), dtype=np.uint32)
    return np.bitwise_xor(
        indices * np.uint32(0x9E3779B1),
        np.uint32((int(rank) + 1) * 0x01010101),
    )


def _worker(
    rank: int,
    token: str,
    words: int,
    result_queue: object,
    command_queue: object,
) -> None:
    allocation = None
    imports: List[object] = []
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(token)
        import cupy as cp  # type: ignore

        from XTA.cuda_d1 import (
            D1PartialBitsetArtifact,
            _CudaIpcImportedWords,
            _D1RawCudaAllocation,
        )

        cp.cuda.Device(0).use()
        allocation = _D1RawCudaAllocation(cp, int(words) * 4)
        host_pattern = _pattern(int(rank), int(words))
        allocation.array[:] = cp.asarray(host_pattern)
        cp.cuda.get_current_stream().synchronize()
        handle = bytes(cp.cuda.runtime.ipcGetMemHandle(int(allocation.pointer)))
        artifact = D1PartialBitsetArtifact(
            group_id="v1803-ipc-selftest",
            model_name="selftest",
            view_name="selftest",
            participant_rank=int(rank),
            participant_worker_id=int(rank),
            output_shape=(1, 1, int(words) * 32),
            word_count=int(words),
            covered_ranges=((int(rank), int(rank) + 1),),
            lease_token=f"selftest-{int(rank)}",
            transport="cuda_ipc",
            ipc_handle=handle,
        )
        result_queue.put({"type": "export", "rank": int(rank), "artifact": artifact.to_payload()})

        command = command_queue.get(timeout=180.0)
        if str(command.get("type")) == "reduce":
            payloads: Sequence[Dict[str, object]] = command["artifacts"]
            started = time.perf_counter()
            for payload in payloads:
                peer = D1PartialBitsetArtifact.from_payload(payload)
                if int(peer.participant_rank) == int(rank):
                    continue
                imported = _CudaIpcImportedWords(cp, peer)
                imports.append(imported)
                cp.bitwise_or(allocation.array, imported.array, out=allocation.array)
            cp.cuda.get_current_stream().synchronize()
            reduced = np.ascontiguousarray(cp.asnumpy(allocation.array), dtype=np.uint32)
            elapsed = max(0.0, time.perf_counter() - started)
            result_queue.put({
                "type": "reduced", "rank": int(rank), "elapsed_seconds": float(elapsed),
                "sum_u64": int(np.sum(reduced, dtype=np.uint64)),
                "first_words": [int(value) for value in reduced[:8]],
                "sha256": hashlib.sha256(reduced.view(np.uint8)).hexdigest(),
            })
            command = command_queue.get(timeout=180.0)
        if str(command.get("type")) != "release":
            raise RuntimeError(f"rank {rank} received unexpected command {command!r}")
        result_queue.put({"type": "released", "rank": int(rank)})
    except BaseException as exc:
        result_queue.put({
            "type": "error", "rank": int(rank), "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        for imported in reversed(imports):
            try:
                imported.close()
            except Exception:
                pass
        if allocation is not None:
            try:
                allocation.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--words", type=int, default=1 << 20)
    args = parser.parse_args()
    world_size = max(2, min(8, int(args.gpus)))
    words = max(1024, int(args.words))
    tokens = _tokens(world_size)
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    command_queues = [ctx.Queue() for _ in range(world_size)]
    processes = [
        ctx.Process(
            target=_worker,
            args=(rank, tokens[rank], words, result_queue, command_queues[rank]),
            name=f"v1803-d1-ipc-{rank}",
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    artifacts: Dict[int, Dict[str, object]] = {}
    reduced_message: Dict[str, object] | None = None
    released: set[int] = set()
    try:
        deadline = time.monotonic() + 240.0
        while len(artifacts) < world_size:
            timeout = max(0.1, deadline - time.monotonic())
            message = result_queue.get(timeout=timeout)
            if str(message.get("type")) == "error":
                raise RuntimeError(
                    f"rank {message.get('rank')} failed: {message.get('error')}\n"
                    f"{message.get('traceback')}"
                )
            if str(message.get("type")) != "export":
                raise RuntimeError(f"unexpected pre-reduction message {message!r}")
            artifacts[int(message["rank"])] = dict(message["artifact"])

        ordered = tuple(artifacts[index] for index in range(world_size))
        command_queues[0].put({"type": "reduce", "artifacts": ordered})
        reduced_message = result_queue.get(timeout=180.0)
        if str(reduced_message.get("type")) == "error":
            raise RuntimeError(
                f"rank 0 reduction failed: {reduced_message.get('error')}\n"
                f"{reduced_message.get('traceback')}"
            )
        if str(reduced_message.get("type")) != "reduced":
            raise RuntimeError(f"unexpected reduction message {reduced_message!r}")

        expected = _pattern(0, words)
        for rank in range(1, world_size):
            np.bitwise_or(expected, _pattern(rank, words), out=expected)
        expected_sum = int(np.sum(expected, dtype=np.uint64))
        expected_first = [int(value) for value in expected[:8]]
        expected_sha256 = hashlib.sha256(expected.view(np.uint8)).hexdigest()
        if (
            int(reduced_message.get("sum_u64", -1)) != expected_sum
            or list(reduced_message.get("first_words", ())) != expected_first
            or str(reduced_message.get("sha256", "")) != expected_sha256
        ):
            raise RuntimeError(
                f"D1 IPC OR mismatch: actual={reduced_message}, "
                f"expected_sum={expected_sum}, expected_first={expected_first}"
            )

        for command_queue in command_queues:
            command_queue.put({"type": "release"})
        while len(released) < world_size:
            message = result_queue.get(timeout=180.0)
            if str(message.get("type")) == "error":
                raise RuntimeError(
                    f"rank {message.get('rank')} release failed: {message.get('error')}\n"
                    f"{message.get('traceback')}"
                )
            if str(message.get("type")) == "released":
                released.add(int(message["rank"]))

        print(json.dumps({
            "status": "ok",
            "gpus": int(world_size),
            "words": int(words),
            "bytes_per_partial": int(words) * 4,
            "reduction_seconds": float(reduced_message.get("elapsed_seconds", 0.0)),
            "sum_u64": int(reduced_message["sum_u64"]),
            "tokens": tokens,
        }, sort_keys=True))
    finally:
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5.0)
        try:
            result_queue.close()
        except Exception:
            pass
        for command_queue in command_queues:
            try:
                command_queue.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
