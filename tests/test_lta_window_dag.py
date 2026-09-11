from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA.lta_execution import _drive_workers_to_fixed_point, _plan_initial_chains, _plan_window_tasks, _prime_authoritative_relay_destinations
from XTA.lta_propagation import LtaMaskSeed, run_mask_injected_session
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from XTA.lta_runtime import LtaTileGridPlan
from XTA.lta_sam import SamFramePrediction
from XTA.lta_scheduler import LtaViewAffinityScheduler
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_tiles import plan_tile_grid
from XTA.lta_worker_adapter import execute_worker_task
from XTA.lta_workers import LtaWorkerResult


class CpuWorkerPool:
    """Exercise the actual window worker protocol with independently ordered completions."""

    def __init__(self, order, devices, *, corrupt_dogfood=False):
        self.order = order
        self.pids = {device: 1000 + device for device in devices}
        self.pending = []
        self.finished = set()
        self.submitted = []
        self.maximum_active = 0
        self.parallel_directions = False
        self.corrupt_dogfood = corrupt_dogfood
        self.context = SimpleNamespace(predictor=object(), profile={"name": "cpu_protocol_test"}, sam_runtime={"distribution_version": "fake"}, constrained_batches=None)

    def submit(self, task, *, execution_device_id):
        predecessor = task.payload.get("predecessor_work_id")
        if predecessor and predecessor not in self.finished:
            raise AssertionError("dependent window dispatched before predecessor completion")
        self.pending.append((task, execution_device_id))
        self.submitted.append(task)
        self.maximum_active = max(self.maximum_active, len(self.pending))
        branches = {}
        for queued, _device in self.pending:
            branches.setdefault(queued.payload.get("chain_work_id"), set()).add(queued.payload["windows"][0]["branch"])
        self.parallel_directions |= any({"backward", "forward"} <= values for values in branches.values())

    def wait_result(self, timeout=None):
        if self.order == "roots_first":
            index = next((index for index, (task, _device) in enumerate(self.pending) if task.payload.get("window_index", 0) == 0), 0)
        else:
            index = -1
        task, device = self.pending.pop(index)
        output = execute_worker_task(self.context, task.kind, task.payload)
        artifact = Path(output["artifact_path"])
        manifest = json.loads(artifact.read_text())
        if self.corrupt_dogfood and manifest.get("dogfood_seed_artifacts"):
            path = Path(manifest["dogfood_seed_artifacts"][0]["path"])
            with path.open("ab") as handle:
                handle.write(b"corruption")
        self.finished.add(task.work_id)
        return LtaWorkerResult(
            work_id=task.work_id, attempt_token=task.attempt_token, kind=task.kind,
            execution_device_id=device, worker_pid=self.pids[device],
            artifact_path=str(artifact), artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            artifact_size_bytes=artifact.stat().st_size,
        )

    def check_liveness(self):
        pass


class LtaWindowDagTests(unittest.TestCase):
    def run_driver(self, root, *, windowed, order="roots_first", empty_backward=False, corrupt_dogfood=False):
        root.mkdir(parents=True)
        count = 120 if empty_backward else 59
        anchors = (50,) if empty_backward else (19, 37)
        tiles = plan_tile_grid(source_width=6, source_height=4, tile_size=4, tile_stride=2)
        grid = LtaTileGridPlan("s4_st2", 4, 2, tiles)
        view = SimpleNamespace(volume_id="volume", physical_view_id="transverse", runtime_view_id="transverse__tta_a0", frame_count=count, frame_height=4, frame_width=6, tile_grids=(grid,))
        cache_path = root / "cache.raw"
        np.zeros((count, 4, 6), dtype=np.uint8).tofile(cache_path)
        stat = cache_path.stat()
        cache = LtaPhysicalViewCacheRef(cache_path, (count, 4, 6), "uint8", "transverse", "c" * 64, stat.st_size, stat.st_mtime_ns)
        by_frame = {}
        for index, anchor in enumerate(anchors):
            mask = np.zeros((4, 4), dtype=bool)
            mask[1 + index, 3] = True
            lineage = LtaLineageId("volume", "transverse", view.runtime_view_id, f"object-{index}", tile_config_id=grid.config_id)
            by_frame[anchor] = (LtaMaskSeed(lineage, anchor, 0, mask, visited_tile_indices=(0,)),)
        with mock.patch("XTA.lta_execution._seeds_for_tile", side_effect=lambda *_args, **_kwargs: by_frame if _args[4] == 0 else {}), mock.patch("XTA.lta_execution.PRODUCTION_LTA_RELAY_MIN_PIXELS", 1):
            chains, inventory = _plan_initial_chains(SimpleNamespace(volume_id="volume"), (), view, cache, temp_root=root, conf=0.15, empty_frame_limit=30)
        tasks = _plan_window_tasks(chains) if windowed else chains
        scheduler = LtaViewAffinityScheduler((task.work for task in tasks), (0, 1, 2, 3), helper_queue_order="head", max_relay_generation=20)
        revisions = {}
        _prime_authoritative_relay_destinations(scheduler, tasks[0].work.view, inventory, revisions)
        pool = CpuWorkerPool(order, scheduler.device_ids, corrupt_dogfood=corrupt_dogfood)
        self.last_pool = pool
        union = np.zeros((count, 4, 6), dtype=np.uint8)

        def sam_adapter(_measured, _raw, **kwargs):
            session, prompt = kwargs["session"], kwargs["prompt_frame"]
            direction = kwargs["propagation_direction"]
            if direction == "both":
                frames = (*range(prompt, session.frame_stop), *range(prompt - 1, session.frame_start - 1, -1))
            elif direction == "forward":
                frames = range(prompt, session.frame_stop)
            else:
                frames = range(prompt, session.frame_start - 1, -1)
            for frame in frames:
                for object_id, seed in enumerate(kwargs["object_masks"]):
                    mask = np.zeros_like(seed) if empty_backward and frame == 36 else np.asarray(seed).copy()
                    kwargs["prediction_callback"](SamFramePrediction(session.sequence_id, session.session_index, frame, object_id, 1.0, mask, 0.9))
            return {"propagation": (), "seed_roundtrip_policy": "overlap-aware", "seed_roundtrip_passed": True, "anchor_integrity_passed": True}

        def propagate(measured, raw, **kwargs):
            return run_mask_injected_session(measured, raw, adapter=sam_adapter, **kwargs)

        # The predictor alone is synthetic. Rendering, seed artifacts, session
        # wrapper, sparse union transport, relays, scheduler and driver are real.
        with mock.patch("XTA.lta_propagation.run_mask_injected_session", side_effect=propagate), mock.patch("XTA.lta_execution.PRODUCTION_LTA_RELAY_MIN_PIXELS", 1):
            result = _drive_workers_to_fixed_point(
                scheduler=scheduler, pool=pool, initial=tasks, view_plan=view,
                cache_ref=cache, view_union=union, relay_mask_revisions=revisions,
                temp_root=root, conf=0.15, empty_frame_limit=30, worker_task_timeout=30,
            )
        return union, result, pool, scheduler

    def test_full_driver_window_dag_matches_whole_chains_across_completion_orders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, baseline_result, _baseline_pool, _ = self.run_driver(root / "whole", windowed=False)
            first, result, pool, scheduler = self.run_driver(root / "windows", windowed=True)
            reverse, reverse_result, _, _ = self.run_driver(root / "reverse", windowed=True, order="reverse")
            np.testing.assert_array_equal(first, baseline)
            np.testing.assert_array_equal(reverse, baseline)
            self.assertEqual(result[0], baseline_result[0])
            self.assertEqual(reverse_result[0], baseline_result[0])
            self.assertGreater(result[0], 0, "neighbor follow-up work was not exercised")
            self.assertEqual(pool.maximum_active, 4)
            self.assertTrue(pool.parallel_directions)
            self.assertTrue(all(task.kind == "propagation_window" for task in pool.submitted))
            self.assertTrue(all(len(task.payload["windows"]) == 1 for task in pool.submitted))
            self.assertTrue(all(task.payload["windows"][0]["frame_stop"] - task.payload["windows"][0]["frame_start"] <= 30 for task in pool.submitted))
            self.assertEqual(scheduler.queue_counts(), {"ready": 0, "blocked": 0, "active": 0, "completed_awaiting_commit": 0})
            self.assertEqual(result[3]["chain_count"], baseline_result[3]["chain_count"])
            self.assertGreater(result[3]["task_count"], baseline_result[3]["task_count"])

    def test_empty_boundary_cancels_only_its_direction_and_settles_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, _, _, _ = self.run_driver(root / "whole", windowed=False, empty_backward=True)
            candidate, result, pool, scheduler = self.run_driver(root / "windows", windowed=True, empty_backward=True)
            np.testing.assert_array_equal(candidate, baseline)
            self.assertGreaterEqual(result[3]["skipped_window_count"], 2)
            initial_tasks = [task for task in pool.submitted if task.payload["relay_generation"] == 0]
            self.assertFalse(any(task.payload["windows"][0]["branch"] == "backward" for task in initial_tasks))
            self.assertTrue(any(task.payload["windows"][0]["branch"] == "forward" for task in initial_tasks))
            self.assertEqual(scheduler.queue_counts()["blocked"], 0)

    def test_corrupt_dogfood_never_unlocks_children(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "digest changed"):
                self.run_driver(Path(folder) / "corrupt", windowed=True, corrupt_dogfood=True)
            self.assertTrue(all(task.payload["window_index"] == 0 for task in self.last_pool.submitted))

    def test_unsettled_graph_without_ready_or_active_work_fails_instead_of_spinning(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(LtaViewAffinityScheduler, "mark_projection_ready"), self.assertRaisesRegex(
                RuntimeError, "no active or ready windows; blocked=4",
            ):
                self.run_driver(Path(folder) / "stalled", windowed=True)
            self.assertEqual(self.last_pool.submitted, [])


if __name__ == "__main__":
    unittest.main()
