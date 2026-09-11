from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
import pickle
import sys
import tempfile
import time
import unittest
import uuid
from unittest import mock
from pathlib import Path
from typing import Iterator

from XTA.lta_workers import (
    LtaWorkerDiedError,
    LtaWorkerExecutionError,
    LtaWorkerInit,
    LtaWorkerPool,
    LtaWorkerResult,
    LtaWorkerShutdownError,
    LtaWorkerTask,
    StaleLtaWorkerResult,
    validate_result_for_attempt,
)


_FAKE_ADAPTER_SOURCE = r'''
import json
import os
import time
from pathlib import Path

# Captured at module import, rather than at factory execution, so tests prove
# the worker narrowed visibility before it imported the adapter.
IMPORTED_VISIBLE_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
IMPORTED_OMP_THREADS = os.environ.get("OMP_NUM_THREADS")


def _append_log(path, value):
    # A per-process log avoids relying on cross-platform append atomicity; the
    # parent merges these tiny diagnostics after every worker has exited.
    worker_path = Path(str(path) + "." + str(os.getpid()))
    with worker_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def build_predictor(config):
    state = {
        "config": dict(config),
        "pid": os.getpid(),
        "visible": IMPORTED_VISIBLE_DEVICE,
        "calls": 0,
    }
    _append_log(
        config["log_path"],
        {
            "event": "factory",
            "pid": state["pid"],
            "import_visible": IMPORTED_VISIBLE_DEVICE,
            "current_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "import_omp_threads": IMPORTED_OMP_THREADS,
            "current_mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        },
    )
    return state


def execute_task(predictor, kind, payload):
    predictor["calls"] += 1
    _append_log(
        predictor["config"]["log_path"],
        {
            "event": "execute",
            "pid": os.getpid(),
            "kind": kind,
            "calls": predictor["calls"],
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    if kind == "crash":
        os._exit(int(payload.get("exit_code", 23)))
    if kind == "error":
        raise RuntimeError(str(payload.get("message", "fake adapter error")))
    if kind == "hang":
        time.sleep(float(payload.get("seconds", 30.0)))

    artifact = Path(payload["artifact_path"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "calls": predictor["calls"],
                "import_visible": IMPORTED_VISIBLE_DEVICE,
                "kind": kind,
                "payload_value": payload.get("value"),
                "pid": os.getpid(),
                "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "artifact_path": str(artifact),
        "metrics": {"calls": predictor["calls"]},
        "metadata": {
            "import_visible": IMPORTED_VISIBLE_DEVICE,
            "payload_value": payload.get("value"),
        },
    }


def close_predictor(predictor):
    _append_log(
        predictor["config"]["log_path"],
        {
            "event": "shutdown",
            "pid": os.getpid(),
            "calls": predictor["calls"],
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    if predictor["config"].get("shutdown_error"):
        raise RuntimeError(str(predictor["config"]["shutdown_error"]))
'''


@contextmanager
def _fake_adapter() -> Iterator[tuple[str, Path, Path]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        name = f"lta_fake_adapter_{uuid.uuid4().hex}"
        (root / f"{name}.py").write_text(_FAKE_ADAPTER_SOURCE, encoding="utf-8")
        log_path = root / "adapter.jsonl"
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        try:
            yield name, root, log_path
        finally:
            sys.path.remove(str(root))
            sys.modules.pop(name, None)


def _init(
    module_name: str,
    log_path: Path,
    *,
    shutdown_error: str | None = None,
) -> LtaWorkerInit:
    config = {"log_path": str(log_path), "nested": {"values": [1, 2, 3]}}
    if shutdown_error is not None:
        config["shutdown_error"] = shutdown_error
    return LtaWorkerInit(
        adapter_module=module_name,
        adapter_factory="build_predictor",
        adapter_execute="execute_task",
        adapter_shutdown="close_predictor",
        adapter_config=config,
    )


def _task(work_id: str, token: str, artifact: Path, **payload: object) -> LtaWorkerTask:
    return LtaWorkerTask(
        work_id=work_id,
        attempt_token=token,
        kind=str(payload.pop("kind", "track")),
        payload={"artifact_path": str(artifact), **payload},
    )


def _read_log(path: Path) -> list[dict[str, object]]:
    records = []
    for worker_path in sorted(path.parent.glob(path.name + ".*")):
        records.extend(
            json.loads(line)
            for line in worker_path.read_text(encoding="utf-8").splitlines()
        )
    return records


class LtaWorkerContractTests(unittest.TestCase):
    def test_cpu_budget_is_bound_before_import_without_modifying_parent_environment(self) -> None:
        with _fake_adapter() as (module_name, root, log_path):
            with mock.patch.dict(os.environ, {
                "SLURM_CPUS_PER_TASK": "8", "OMP_NUM_THREADS": "64", "MKL_NUM_THREADS": "64",
            }):
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                before = dict(os.environ)
                pool = LtaWorkerPool((0, 1, 2, 3), _init(module_name, log_path), startup_timeout=10)
                try:
                    self.assertEqual(dict(os.environ), before)
                    self.assertEqual(pool.cpu_budget["effective_cpu_count"], 8)
                    self.assertEqual(pool.cpu_budget["threads_per_worker"], 1)
                    self.assertTrue(all(event.metadata["cpu_budget"] == pool.cpu_budget for event in pool.ready_events))
                finally:
                    pool.shutdown(timeout=5)
            factories = [item for item in _read_log(log_path) if item["event"] == "factory"]
            self.assertEqual(len(factories), 4)
            self.assertTrue(all(item["import_omp_threads"] == "1" for item in factories))
            self.assertTrue(all(item["current_mkl_threads"] == "1" for item in factories))

    def test_init_and_task_are_detached_primitive_spawn_payloads(self) -> None:
        config = {"nested": {"values": [1, 2]}}
        payload = {"box": [1, 2, 3, 4]}
        init = LtaWorkerInit("adapter", "factory", "execute", adapter_config=config)
        task = LtaWorkerTask("work", "attempt", "track", payload)

        config["nested"]["values"].append(3)  # type: ignore[index, union-attr]
        payload["box"].append(5)

        self.assertEqual(init.adapter_config, {"nested": {"values": [1, 2]}})
        self.assertEqual(task.payload, {"box": [1, 2, 3, 4]})
        pickle.loads(pickle.dumps(init, protocol=pickle.HIGHEST_PROTOCOL))
        pickle.loads(pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL))

        with self.assertRaisesRegex(TypeError, "only None"):
            LtaWorkerTask("work", "attempt-2", "track", {"path": Path("bad")})

    def test_one_spawned_process_per_device_binds_before_import_and_reuses_predictor(self) -> None:
        with _fake_adapter() as (module_name, root, log_path):
            self.assertNotIn(module_name, sys.modules)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                pool = LtaWorkerPool(
                    (7, 3),
                    _init(module_name, log_path),
                    startup_timeout=10.0,
                )
            try:
                self.assertEqual(pool.start_method, "spawn")
                self.assertEqual(len(set(pool.pids.values())), 2)
                self.assertEqual(
                    [(event.execution_device_id, event.visible_device) for event in pool.ready_events],
                    [(7, "7"), (3, "3")],
                )

                pool.submit(_task("seven-a", "a1", root / "seven-a.json", value=11), execution_device_id=7)
                pool.submit(_task("three", "b1", root / "three.json", value=22), execution_device_id=3)
                pool.submit(_task("seven-b", "c1", root / "seven-b.json", value=33), execution_device_id=7)
                results = [pool.wait_result(timeout=10.0) for _ in range(3)]
            finally:
                pool.shutdown(timeout=5.0)

            by_work = {result.work_id: result for result in results}
            self.assertEqual(set(by_work), {"seven-a", "three", "seven-b"})
            self.assertEqual(by_work["seven-a"].execution_device_id, 7)
            self.assertEqual(by_work["seven-b"].execution_device_id, 7)
            self.assertEqual(by_work["three"].execution_device_id, 3)
            self.assertEqual(by_work["seven-a"].worker_pid, by_work["seven-b"].worker_pid)
            self.assertNotEqual(by_work["seven-a"].worker_pid, by_work["three"].worker_pid)
            for result in results:
                artifact = Path(result.artifact_path)
                self.assertEqual(result.artifact_size_bytes, artifact.stat().st_size)
                self.assertEqual(
                    result.artifact_sha256,
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                )

            records = _read_log(log_path)
            factories = [record for record in records if record["event"] == "factory"]
            self.assertEqual(len(factories), 2)
            self.assertEqual(
                {record["import_visible"] for record in factories}, {"7", "3"}
            )
            self.assertTrue(
                all(record["import_visible"] == record["current_visible"] for record in factories)
            )
            factory_counts: dict[int, int] = {}
            for record in factories:
                pid = int(record["pid"])
                factory_counts[pid] = factory_counts.get(pid, 0) + 1
            self.assertEqual(set(factory_counts.values()), {1})
            seven_calls = sorted(
                int(record["calls"])
                for record in records
                if record["event"] == "execute" and record["visible"] == "7"
            )
            self.assertEqual(seven_calls, [1, 2])

    def test_logical_devices_map_through_parent_cuda_visibility(self) -> None:
        with _fake_adapter() as (module_name, _root, log_path):
            with mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "GPU-alpha,GPU-beta,GPU-gamma"},
            ):
                pool = LtaWorkerPool(
                    (2, 0),
                    _init(module_name, log_path),
                    startup_timeout=10.0,
                )
            try:
                self.assertEqual(
                    pool.visible_device_tokens,
                    {2: "GPU-gamma", 0: "GPU-alpha"},
                )
                self.assertEqual(
                    [event.visible_device for event in pool.ready_events],
                    ["GPU-gamma", "GPU-alpha"],
                )
            finally:
                pool.shutdown(timeout=5.0)
            self.assertNotIn(module_name, sys.modules)

    def test_task_error_is_reported_without_killing_persistent_worker(self) -> None:
        with _fake_adapter() as (module_name, root, log_path):
            with LtaWorkerPool((4,), _init(module_name, log_path), startup_timeout=10.0) as pool:
                original_pid = pool.pids[4]
                pool.submit(
                    _task(
                        "bad",
                        "bad-attempt",
                        root / "unused.json",
                        kind="error",
                        message="controlled failure",
                    ),
                    execution_device_id=4,
                )
                with self.assertRaises(LtaWorkerExecutionError) as raised:
                    pool.wait_result(timeout=10.0)
                self.assertEqual(raised.exception.event.work_id, "bad")
                self.assertIn("controlled failure", raised.exception.event.message)
                self.assertTrue(pool.is_alive(4))

                pool.submit(
                    _task("good", "good-attempt", root / "good.json"),
                    execution_device_id=4,
                )
                result = pool.wait_result(timeout=10.0)
                self.assertEqual(result.work_id, "good")
                self.assertEqual(result.worker_pid, original_pid)

    def test_abrupt_child_crash_is_propagated_by_liveness_checks(self) -> None:
        with _fake_adapter() as (module_name, root, log_path):
            pool = LtaWorkerPool((2,), _init(module_name, log_path), startup_timeout=10.0)
            try:
                pool.submit(
                    _task(
                        "crash",
                        "attempt-crash",
                        root / "never.json",
                        kind="crash",
                        exit_code=37,
                    ),
                    execution_device_id=2,
                )
                with self.assertRaises(LtaWorkerDiedError) as raised:
                    pool.wait_result(timeout=10.0)
                self.assertEqual(raised.exception.device_id, 2)
                self.assertEqual(raised.exception.exitcode, 37)
            finally:
                pool.shutdown(timeout=0.1, force=True)

    def test_retry_token_rejects_late_result_before_new_attempt_settles(self) -> None:
        with _fake_adapter() as (module_name, root, log_path):
            with LtaWorkerPool((1,), _init(module_name, log_path), startup_timeout=10.0) as pool:
                pool.submit(
                    _task("same", "old-lease", root / "old.json", value="old"),
                    execution_device_id=1,
                )
                pool.submit(
                    _task("same", "new-lease", root / "new.json", value="new"),
                    execution_device_id=1,
                )
                with self.assertRaises(StaleLtaWorkerResult) as raised:
                    pool.wait_result(timeout=10.0)
                self.assertEqual(raised.exception.result.attempt_token, "old-lease")
                self.assertEqual(raised.exception.expected_token, "new-lease")

                current = pool.wait_result(timeout=10.0)
                self.assertIs(
                    validate_result_for_attempt(current, "new-lease"), current
                )

        with self.assertRaises(StaleLtaWorkerResult):
            validate_result_for_attempt(
                LtaWorkerResult(
                    work_id="same",
                    attempt_token="retired",
                    kind="track",
                    execution_device_id=1,
                    worker_pid=os.getpid(),
                    artifact_path=str(Path("artifact")),
                    artifact_sha256="0" * 64,
                    artifact_size_bytes=0,
                ),
                "live",
            )

    def test_orderly_shutdown_invokes_adapter_once_and_is_idempotent(self) -> None:
        with _fake_adapter() as (module_name, _root, log_path):
            pool = LtaWorkerPool((6,), _init(module_name, log_path), startup_timeout=10.0)
            pid = pool.pids[6]
            self.assertEqual(pool.shutdown(timeout=5.0), ())
            self.assertTrue(pool.closed)
            self.assertEqual(pool.shutdown(timeout=5.0), ())
            records = _read_log(log_path)
            shutdowns = [
                record
                for record in records
                if record["event"] == "shutdown" and int(record["pid"]) == pid
            ]
            self.assertEqual(len(shutdowns), 1)

    def test_orderly_shutdown_surfaces_adapter_cleanup_failure(self) -> None:
        with _fake_adapter() as (module_name, _root, log_path):
            pool = LtaWorkerPool(
                (6,),
                _init(
                    module_name,
                    log_path,
                    shutdown_error="controlled cleanup failure",
                ),
                startup_timeout=10.0,
            )
            with self.assertRaisesRegex(
                LtaWorkerShutdownError, "controlled cleanup failure"
            ):
                pool.shutdown(timeout=5.0)
            self.assertTrue(pool.closed)
            self.assertEqual(pool.shutdown(timeout=5.0), ())

    def test_shutdown_forces_cleanup_of_a_stuck_adapter(self) -> None:
        with _fake_adapter() as (module_name, root, log_path):
            pool = LtaWorkerPool((8,), _init(module_name, log_path), startup_timeout=10.0)
            pool.submit(
                _task(
                    "hang",
                    "hang-attempt",
                    root / "hang.json",
                    kind="hang",
                    seconds=30.0,
                ),
                execution_device_id=8,
            )
            # Wait until execute really started so the orderly sentinel is
            # behind the active call rather than racing it.
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if any(record.get("kind") == "hang" for record in _read_log(log_path)):
                    break
                time.sleep(0.02)
            else:
                self.fail("fake adapter did not start the hanging task")

            forced = pool.shutdown(timeout=0.05, force=True)
            self.assertEqual(forced, (8,))
            self.assertTrue(pool.closed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
