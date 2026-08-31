from __future__ import annotations

import importlib
import multiprocessing
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


ROOT = Path(__file__).resolve().parents[1]
install_stubs()

from XTA import pta_publication, pta_rendering, pta_workers
from XTA.pta_dataset import OutputCandidate


def _spawn_unpickle_worker_payload(
    payload_blob: bytes,
    result_queue: multiprocessing.Queue,
) -> None:
    """Spawn-safe target that proves payload loading never imports XTA.pta."""

    install_stubs()
    payload = pickle.loads(payload_blob)
    plan = payload["plans"][0]
    task = payload["tasks"][0]
    gpu_batch = payload["tasks"][1]
    candidate = task.items[0][1][0]
    modules = (
        type(plan).__module__,
        type(plan.view).__module__,
        type(plan.aff).__module__,
        type(plan.channel_variant).__module__,
        type(plan.tile_layout[0]).__module__,
        type(plan.tile_layout[0].cfg).__module__,
        type(candidate).__module__,
        type(task).__module__,
        type(gpu_batch).__module__,
        payload["initializer"].__module__,
        payload["process_entry"].__module__,
        payload["thread_entry"].__module__,
    )
    result_queue.put(
        {
            "pta_loaded": "XTA.pta" in sys.modules,
            "modules": modules,
        }
    )


def _representative_payload() -> dict[str, object]:
    identity = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    view = pta_rendering.ViewInfo(
        name="transverse",
        display_name="Transverse",
        family="transverse",
        num_slices=2,
        src_h=2,
        src_w=2,
        pad_mode="clamp",
        full_t=2,
        full_h=2,
        full_w=2,
    )
    affine = pta_rendering.AffineSpec(
        angle_deg=0.0,
        src_w=2,
        src_h=2,
        canvas_w=2,
        canvas_h=2,
        out_w=2,
        out_h=2,
        pad_off_x=0.0,
        pad_off_y=0.0,
        M_src_to_canvas=identity,
        M_canvas_to_src=identity,
        M_src_to_out=identity,
        M_out_to_src=identity,
    )
    tile = pta_rendering.RenderTileItem(
        cfg=pta_rendering.TileConfig(2, 2, "2x2"),
        x=0,
        y=0,
        tile_tag="tile",
        out_w=2,
        out_h=2,
        img_pattern="image_%04d.png",
        lbl_pattern="label_%04d.txt",
        overlay_path=None,
        label_enabled=False,
    )
    plan = pta_rendering.RenderPlan(
        view=view,
        aff=affine,
        tag="transverse",
        img_pattern="image_%04d.png",
        lbl_pattern="label_%04d.txt",
        overlay_path=None,
        label_enabled=False,
        tile_layout=(tile,),
        stats={},
    )
    candidate = OutputCandidate(
        order=0,
        volume_name="sample",
        parent_view_tag="transverse",
        output_tag="transverse",
        item_key="full",
        frame_idx=0,
        is_tile=False,
        label_enabled=False,
    )
    task = pta_workers.FrameRenderTask(
        plan_idx=0,
        frame_idx=0,
        items=(("full", (candidate,)),),
    )
    return {
        "plans": [plan],
        "tasks": [task, pta_workers.GpuFrameBatchTask((task,))],
        "initializer": pta_workers._render_worker_initializer,
        "process_entry": pta_workers._render_task_entry,
        "thread_entry": pta_workers._render_task_entry_thread,
    }


class PtaWorkerBoundaryTests(unittest.TestCase):
    def test_each_owner_first_imports_without_pta_or_accelerator_runtime(self) -> None:
        for module_name in (
            "XTA.pta_rendering",
            "XTA.pta_publication",
            "XTA.pta_workers",
        ):
            with self.subTest(module=module_name):
                program = (
                    "from tools.smoke_import import install_stubs; install_stubs(); "
                    "import importlib,sys; "
                    f"importlib.import_module({module_name!r}); "
                    "assert 'XTA.pta' not in sys.modules; "
                    "roots={name.split('.')[0] for name in sys.modules}; "
                    "assert not ({'torch','cupy','nvidia'} & roots), roots"
                )
                completed = subprocess.run(
                    [sys.executable, "-c", program],
                    cwd=ROOT,
                    env={**os.environ, "YOLO_TTA_TELEMETRY": "0"},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_pta_reexports_owner_objects_by_identity(self) -> None:
        from XTA import pta

        owner_names = {
            pta_rendering: (
                "TileConfig",
                "ChannelFormat",
                "ChannelVariant",
                "ViewInfo",
                "AffineSpec",
                "RenderTileItem",
                "RenderPlan",
                "RenderFrameSource",
                "render_plan_frame_source",
                "render_plan_frame_mask_source",
            ),
            pta_publication: (
                "OUTPUT_IMAGE_FORMATS",
                "PtaDatasetImageSink",
                "candidate_output_paths",
                "write_image",
                "write_selected_candidate_version",
            ),
            pta_workers: (
                "ArrayAllocator",
                "FrameRenderTask",
                "GpuFrameBatchTask",
                "PersistentRenderPool",
                "RenderPhaseHandle",
                "SharedBlock",
                "VolumeRenderProgress",
                "WarningLog",
                "build_phase_render_tasks",
                "drain_render_results",
                "execute_render_task",
                "set_worker_static_context",
            ),
        }
        for owner, names in owner_names.items():
            for name in names:
                with self.subTest(owner=owner.__name__, name=name):
                    self.assertIs(getattr(pta, name), getattr(owner, name))
        self.assertIs(pta._WORKER_STATIC, pta_workers._WORKER_STATIC)

    def test_payload_and_worker_targets_round_trip_through_spawn(self) -> None:
        payload_blob = pickle.dumps(
            _representative_payload(),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        child = context.Process(
            target=_spawn_unpickle_worker_payload,
            args=(payload_blob, result_queue),
        )
        child.start()
        child.join(timeout=30.0)
        if child.is_alive():
            child.terminate()
            child.join(timeout=10.0)
            self.fail("spawned PTA payload child did not exit")
        self.assertEqual(child.exitcode, 0)
        try:
            result = result_queue.get(timeout=5.0)
        except queue.Empty as exc:
            self.fail(f"spawned PTA payload child returned no result: {exc}")
        finally:
            result_queue.close()
            result_queue.join_thread()

        self.assertFalse(result["pta_loaded"])
        self.assertEqual(
            result["modules"],
            (
                "XTA.pta_rendering",
                "XTA.pta_rendering",
                "XTA.pta_rendering",
                "XTA.pta_rendering",
                "XTA.pta_rendering",
                "XTA.pta_rendering",
                "XTA.pta_dataset",
                "XTA.pta_workers",
                "XTA.pta_workers",
                "XTA.pta_workers",
                "XTA.pta_workers",
                "XTA.pta_workers",
            ),
        )

    def test_pool_uses_canonical_worker_initializer_and_process_entry(self) -> None:
        calls: dict[str, object] = {}

        class FakePool:
            def apply_async(self, function: object, *_args: object, **_kwargs: object) -> None:
                calls["entry"] = function

            def close(self) -> None:
                calls["closed"] = True

            def join(self) -> None:
                calls["joined"] = True

        class FakeContext:
            def Pool(self, **kwargs: object) -> FakePool:
                calls.update(kwargs)
                return FakePool()

        with tempfile.TemporaryDirectory() as temp_dir:
            pta_workers.set_worker_static_context(
                out_dir=Path(temp_dir),
                split_active=False,
                image_format="png",
                png_compression=1,
                jpeg_quality=95,
                jpeg_encode_backend="opencv",
                gpu_batch_size=1,
                gpu_render_threads=1,
                gpu_device_ids=(),
                augmentation=None,
            )
            with mock.patch.object(
                pta_workers.multiprocessing,
                "get_context",
                return_value=FakeContext(),
            ):
                pool = pta_workers.PersistentRenderPool(
                    backend="process",
                    workers=1,
                )
            handle = pta_workers.RenderPhaseHandle(
                payload_name="payload",
                payload_nbytes=1,
                payload_block=None,
                tasks=[object()],
                payload=None,
            )
            pool.submit_phase(handle, meta=(0, "A"))
            pool.close()

        self.assertIs(calls["initializer"], pta_workers._render_worker_initializer)
        self.assertIs(calls["entry"], pta_workers._render_task_entry)
        self.assertTrue(calls["closed"])
        self.assertTrue(calls["joined"])

    def test_initializer_resets_and_closes_every_worker_cache(self) -> None:
        stale_shm = mock.Mock()
        pta_workers._WORKER_PAYLOAD_CACHE["payload"] = {"gen": 1}
        pta_workers._WORKER_GEN_CACHE[1] = {"shms": [stale_shm]}
        pta_workers._WORKER_VOLUME_IDENTITY_BY_POINTER[123] = "stale"
        pta_workers._WORKER_GPU_DEVICE_ID = 4
        pta_workers._WORKER_GPU_RUNTIME = {"stale": True}
        pta_workers._WORKER_GPU_CODEC_WARNING_EMITTED = True
        pta_workers._WORKER_GPU_BATCH_CAP_WARNING_EMITTED = True

        pta_workers._render_worker_initializer()

        stale_shm.close.assert_called_once_with()
        self.assertEqual(pta_workers._WORKER_PAYLOAD_CACHE, {})
        self.assertEqual(pta_workers._WORKER_GEN_CACHE, {})
        self.assertEqual(pta_workers._WORKER_VOLUME_IDENTITY_BY_POINTER, {})
        self.assertIsNone(pta_workers._WORKER_GPU_DEVICE_ID)
        self.assertIsNone(pta_workers._WORKER_GPU_RUNTIME)
        self.assertFalse(pta_workers._WORKER_GPU_CODEC_WARNING_EMITTED)
        self.assertFalse(pta_workers._WORKER_GPU_BATCH_CAP_WARNING_EMITTED)


if __name__ == "__main__":
    unittest.main()
