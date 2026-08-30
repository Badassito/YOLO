from __future__ import annotations

import multiprocessing as mp
import queue
import unittest

from XTA.inference_backends import DispatchSemantics, ExecutionTarget


def _echo_target(result_queue: object, target: ExecutionTarget) -> None:
    result_queue.put(
        {
            "is_target": isinstance(target, ExecutionTarget),
            "target_id": target.target_id,
            "world_size": target.world_size,
        }
    )


class SpawnContractTests(unittest.TestCase):
    def test_collective_target_round_trips_through_spawn(self) -> None:
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        target = ExecutionTarget(
            target_id="collective/slice-1",
            backend_id="future-collective",
            semantics=DispatchSemantics.COLLECTIVE,
            host_count=2,
            world_size=8,
            coordinator_rank=0,
            host_arches=("aarch64",),
        )
        process = context.Process(target=_echo_target, args=(result_queue, target))
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail("spawned contract process did not exit")
        self.assertEqual(process.exitcode, 0)
        try:
            result = result_queue.get(timeout=5)
        except queue.Empty as exc:
            self.fail(f"spawned contract process returned no result: {exc}")
        finally:
            result_queue.close()
            result_queue.join_thread()
        self.assertEqual(
            result,
            {"is_target": True, "target_id": "collective/slice-1", "world_size": 8},
        )


if __name__ == "__main__":
    unittest.main()
