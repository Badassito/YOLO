from __future__ import annotations

import hashlib
import json
import contextlib
import tempfile
import types
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    install_stubs()

from XTA.geometry import expand_views_into_tta_variants, get_view_infos
from XTA.lta_execution import (
    _accumulate_relay_seed_revision,
    _accumulate_worker_ready_audit,
    _accumulate_worker_audit,
    _aligned_annotations,
    _allocate_view_union,
    _authoritative_tile_owners,
    _hard_positive_volume,
    _aligned_known_background_frames,
    _known_background_audit,
    _new_worker_audit,
    _lineage_from_record,
    _plan_initial_chains,
    _plan_relay_generation,
    _relay_generation_bound,
    _seeds_for_tile,
    execute_lta_plan,
)
from XTA.lta_inputs import (
    AnnotationState,
    FrameAnnotation,
    LtaInputDiscovery,
    LtaVolumeSpec,
    PositiveExemplar,
    SourceRole,
    VolumeClass,
    YoloPolygon,
)
from XTA.lta_outputs import LtaArtifactReceipt, LtaLayerRecord, LtaRecompositionOp
from XTA.lta_postprocessing import LtaFinalNrrdArtifact
from XTA.lta_propagation import (
    LtaMaskSeed,
    LtaSeedProvenance,
    read_seed_artifact,
    write_seed_artifact,
)
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from XTA.lta_runtime import (
    LtaRunPlan,
    LtaRuntimeViewPlan,
    LtaTileGridPlan,
    LtaVolumePlan,
)
from XTA.lta_sam import LTA_MAX_NUM_OBJECTS, LocalSamBundle, plan_sam_sessions
from XTA.lta_scheduler import LtaSpatialRelayKey, LtaViewAffinityScheduler, LtaViewKey
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_tiles import plan_tile_grid
from XTA.lta_workers import LtaWorkerReady, LtaWorkerResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeWorkerPool:
    instances: list["_FakeWorkerPool"] = []

    def __init__(self, device_ids, _init, *, startup_timeout):
        self.device_ids = tuple(device_ids)
        self.pids = {device: 1000 + device for device in self.device_ids}
        self.pending = []
        self.used_devices = []
        self.wait_timeouts = []
        self.closed = False
        self.instances.append(self)

    def submit(self, task, *, execution_device_id):
        self.pending.append((task, int(execution_device_id)))
        self.used_devices.append(int(execution_device_id))

    def wait_result(self, timeout=None):
        self.wait_timeouts.append(timeout)
        task, device = self.pending.pop(0)
        payload = task.payload
        output = Path(payload["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        tile = payload["tile"]
        size = int(tile["size"])
        start = int(payload["output_frame_start"])
        stop = int(payload["output_frame_stop"])
        generation = int(payload["relay_generation"])
        union = np.zeros((stop - start, size, size), dtype=np.uint8)
        union[:, 1, 3] = 1
        union_path = output / "union.uint8.raw"
        union.tofile(union_path)
        relays = []
        if generation == 0 and int(payload["tile_index"]) == 0:
            source_seed = read_seed_artifact(
                payload["seed_artifact_path"],
                expected_sha256=payload["seed_artifact_sha256"],
            )[0]
            relay_mask = np.zeros((size, size), dtype=bool)
            relay_mask[1, 1] = True
            relay_seed = LtaMaskSeed(
                lineage=source_seed.lineage,
                frame_index=stop - 1,
                object_id=0,
                mask=relay_mask,
                provenance=LtaSeedProvenance.SPATIAL_RELAY,
                tracker_probability=0.9,
                relay_generation=1,
                visited_tile_indices=(0, 1),
            )
            relay_artifact = write_seed_artifact(output / "relay.npz", (relay_seed,))
            relays.append(
                {
                    "lineage": {
                        "volume_id": source_seed.lineage.volume_id,
                        "physical_view_id": source_seed.lineage.physical_view_id,
                        "runtime_view_id": source_seed.lineage.runtime_view_id,
                        "tile_config_id": source_seed.lineage.tile_config_id,
                        "lineage_id": source_seed.lineage.lineage_id,
                    },
                    "source_tile_index": 0,
                    "destination_tile_index": 1,
                    "frame_index": stop - 1,
                    "temporal_direction": "forward",
                    "generation": 1,
                    "visited_tile_indices": [0, 1],
                    "seed_artifact_path": str(relay_artifact.path),
                    "seed_artifact_sha256": relay_artifact.sha256,
                }
            )
        manifest = {
            "schema": "lta.propagation-chain/1",
            "status": "complete",
            "work_id": task.work_id,
            "sequence_id": payload["sequence_id"],
            "tile_index": payload["tile_index"],
            "tile": dict(tile),
            "relay_generation": generation,
            "output_frame_range": [start, stop],
            "union": {
                "path": str(union_path),
                "sha256": _sha256(union_path),
                "size_bytes": union_path.stat().st_size,
                "shape": list(union.shape),
                "dtype": "uint8",
            },
            "relays": relays,
            "windows": [],
            "profile": {
                "name": "fake",
                "execution_device_id": device,
            },
            "sam_runtime": {
                "distribution_version": "0.1.0",
                "package_tree_sha256": "f" * 64,
            },
            "constrained_batches": None,
            "relay_gate": {
                "minimum_overlap_pixels": int(payload["relay_min_pixels"]),
                "minimum_tracker_probability": float(
                    payload["relay_min_probability"]
                ),
            },
            "foreground_pixels": int(union.sum()),
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return LtaWorkerResult(
            work_id=task.work_id,
            attempt_token=task.attempt_token,
            kind=task.kind,
            execution_device_id=device,
            worker_pid=self.pids[device],
            artifact_path=str(manifest_path),
            artifact_sha256=_sha256(manifest_path),
            artifact_size_bytes=manifest_path.stat().st_size,
        )

    def check_liveness(self):
        return None

    def shutdown(self, *, timeout, force):
        self.closed = True


class _Finalization:
    def __init__(self, layers):
        self.terminal_union = np.logical_or.reduce(
            tuple(np.asarray(layer.volume) != 0 for layer in layers)
        ).astype(np.uint8)
        self.terminal_union_foreground_voxels = int(self.terminal_union.sum())
        self.postprocessing = {"execution_order": []}
        self.closed = False

    def close(self):
        self.closed = True


class LtaProductionExecutionTests(unittest.TestCase):
    def test_compact_worker_audit_retains_raw_and_wrapper_confidence_drops(self) -> None:
        audit = _new_worker_audit()
        _accumulate_worker_ready_audit(
            audit,
            LtaWorkerReady(
                execution_device_id=2,
                worker_pid=123,
                visible_device="GPU-uuid",
                metadata={
                    "profile": {"name": "h100", "capability": (9, 0)},
                    "sam_runtime": {
                        "distribution_version": "0.1.0",
                        "package_tree_sha256": "b" * 64,
                    },
                    "constrained_batches": None,
                },
            ),
        )
        result = LtaWorkerResult(
            work_id="work",
            attempt_token="attempt",
            kind="propagation_chain",
            execution_device_id=2,
            worker_pid=123,
            artifact_path="manifest.json",
            artifact_sha256="a" * 64,
            artifact_size_bytes=1,
        )
        _accumulate_worker_audit(
            audit,
            {
                "profile": {"name": "h100", "capability": [9, 0]},
                "sam_runtime": {
                    "distribution_version": "0.1.0",
                    "package_tree_sha256": "b" * 64,
                },
                "constrained_batches": None,
                "relay_gate": {
                    "minimum_overlap_pixels": 16,
                    "minimum_tracker_probability": 0.5,
                },
                "relays": (),
                "foreground_pixels": 4,
                "windows": (
                    {
                        "status": "complete",
                        "prediction_count": 3,
                        "retained_prediction_count": 0,
                        "dogfood_seed_count": 1,
                        "hole_fill_added_pixels": 2,
                        "adapter": {
                            "production_below_confidence_prediction_count": 1,
                            "tracker_confidence_filter": {
                                "below_threshold_object_observation_count": 2,
                            },
                        },
                    },
                ),
            },
            result,
        )

        self.assertEqual(audit["below_confidence_prediction_count"], 3)
        self.assertEqual(
            audit["profiles_by_device"],
            {"2": {"name": "h100", "capability": [9, 0]}},
        )
        self.assertEqual(audit["ready_worker_count"], 1)
        self.assertEqual(audit["chain_count"], 1)

    def test_known_background_labels_are_audited_without_subtracting_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self._plan(Path(temp_dir), (0,))
            source = plan.discovery.target_volumes[0]
            annotations = list(source.annotations)
            annotations[1] = replace(
                annotations[1],
                state=AnnotationState.KNOWN_BACKGROUND,
            )
            source = replace(source, annotations=tuple(annotations))
            frames = _aligned_known_background_frames(
                plan,
                source,
                tuple(item for item in annotations if item.polygons),
            )
            terminal = np.zeros((4, 4, 6), dtype=np.uint8)
            terminal[1, 2, 3] = 1
            audit = _known_background_audit(terminal, frames)

        self.assertEqual(frames, (1,))
        self.assertEqual(audit["policy"], "audit_only_no_subtraction")
        self.assertEqual(audit["frames_with_predicted_foreground"], 1)
        self.assertEqual(audit["predicted_foreground_pixels"], 1)
        self.assertTrue(bool(terminal[1, 2, 3]))

    def test_storage_preflight_fails_before_decode_when_scratch_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self._plan(Path(temp_dir), (0, 1, 2, 3))
            with (
                mock.patch("XTA.lta_execution.PRODUCTION_LTA_TILE_SIZE", 4),
                mock.patch(
                    "XTA.lta_execution.shutil.disk_usage",
                    return_value=types.SimpleNamespace(free=1),
                ),
                self.assertRaisesRegex(RuntimeError, "capacity preflight failed before decode"),
            ):
                execute_lta_plan(plan)
            self.assertFalse(plan.output_root.exists())
            self.assertFalse(plan.temp_root.exists())

    def test_overlapping_authoritative_polygon_has_one_direct_tile_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self._plan(Path(temp_dir), (0,))
            source = plan.discovery.target_volumes[0]
            view_plan = plan.volumes[0].runtime_views[0]
            grid = view_plan.tile_grids[0]
            polygon = YoloPolygon(
                class_id=0,
                row_index=7,
                points=((0.35, 0.25), (0.65, 0.25), (0.65, 0.75), (0.35, 0.75)),
                box_xyxy=(0.35, 0.25, 0.65, 0.75),
                box_cxcywh=(0.5, 0.5, 0.3, 0.5),
                normalized_area=0.15,
            )
            annotation = replace(
                source.annotations[0],
                polygons=(polygon,),
            )
            owners = _authoritative_tile_owners((annotation,), grid)
            seeded = [
                (tile_index, seed)
                for tile_index in range(len(grid.tiles))
                for seeds in _seeds_for_tile(
                    source,
                    (annotation,),
                    view_plan,
                    grid,
                    tile_index,
                    authoritative_tile_owners=owners,
                ).values()
                for seed in seeds
            ]

        self.assertEqual(len(seeded), 1)
        owner, seed = seeded[0]
        self.assertEqual((owner,), next(iter(owners.values())))
        self.assertTrue(seed.source_receipt["authoritative_tile_owner"])

        wide = YoloPolygon(
            class_id=0,
            row_index=8,
            points=((0.05, 0.25), (0.95, 0.25), (0.95, 0.75), (0.05, 0.75)),
            box_xyxy=(0.05, 0.25, 0.95, 0.75),
            box_cxcywh=(0.5, 0.5, 0.9, 0.5),
            normalized_area=0.45,
        )
        clipped_annotation = replace(annotation, polygons=(wide,))
        clipped_owners = _authoritative_tile_owners((clipped_annotation,), grid)
        self.assertEqual(next(iter(clipped_owners.values())), (0, 1))

    def test_complementary_later_relay_advances_a_monotone_mask_revision(self) -> None:
        lineage = LtaLineageId(
            "volume",
            "transverse",
            "transverse__tta_a0",
            "object",
            tile_config_id="s4_st2",
        )
        event = LtaSpatialRelayKey(
            view=LtaViewKey("volume", "transverse"),
            runtime_view_id=lineage.runtime_view_id,
            tile_config_id=lineage.tile_config_id,
            lineage_id=lineage.lineage_id,
            destination_tile_index=1,
            frame_index=7,
            temporal_direction="forward",
        )
        revisions = {}

        def seed(mask: np.ndarray, generation: int) -> LtaMaskSeed:
            return LtaMaskSeed(
                lineage=lineage,
                frame_index=7,
                object_id=0,
                mask=mask,
                provenance=LtaSeedProvenance.SPATIAL_RELAY,
                tracker_probability=0.8,
                relay_generation=generation,
                visited_tile_indices=(0, 1),
            )

        first_mask = np.zeros((4, 4), dtype=bool)
        first_mask[1, 1] = True
        complement = np.zeros((4, 4), dtype=bool)
        complement[1, 2] = True
        identity_fill = lambda mask, **_kwargs: types.SimpleNamespace(  # noqa: E731
            mask=np.asarray(mask, dtype=bool).copy(),
            receipt=types.SimpleNamespace(manifest_record=lambda: {"added_pixels": 0}),
        )
        with mock.patch(
            "XTA.lta_execution.fill_merged_seed_mask_holes_2d",
            side_effect=identity_fill,
        ):
            first = _accumulate_relay_seed_revision(
                event,
                seed(first_mask, 1),
                revisions,
            )
            second = _accumulate_relay_seed_revision(
                event,
                seed(complement, 2),
                revisions,
            )
            subset = _accumulate_relay_seed_revision(
                event,
                seed(first_mask, 3),
                revisions,
            )

        assert first is not None and second is not None
        self.assertNotEqual(first[1], second[1])
        np.testing.assert_array_equal(second[0].mask, first_mask | complement)
        self.assertEqual(second[0].source_receipt["new_foreground_pixels"], 1)
        self.assertIsNone(subset)

    def _plan(self, root: Path, devices: tuple[int, ...]) -> LtaRunPlan:
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "sam3.1.pt"
        checkpoint.write_bytes(b"checkpoint")
        video = root / "source.mkv"
        video.write_bytes(b"video")
        label0 = root / "source_0000.txt"
        label3 = root / "source_0003.txt"
        label0.write_text("0 0.05 0.25 0.20 0.25 0.20 0.50\n", encoding="utf-8")
        label3.write_text(label0.read_text(encoding="utf-8"), encoding="utf-8")
        polygon = YoloPolygon(
            class_id=0,
            row_index=0,
            points=((0.05, 0.25), (0.20, 0.25), (0.20, 0.50)),
            box_xyxy=(0.05, 0.25, 0.20, 0.50),
            box_cxcywh=(0.125, 0.375, 0.15, 0.25),
            normalized_area=0.01875,
        )
        annotations = tuple(
            FrameAnnotation(
                encoded_index=frame,
                frame_position=frame,
                state=(
                    AnnotationState.FOREGROUND
                    if frame in {0, 3}
                    else AnnotationState.UNKNOWN
                ),
                label_path=label0 if frame == 0 else label3 if frame == 3 else None,
                label_sha256="a" * 64 if frame in {0, 3} else None,
                polygons=(polygon,) if frame in {0, 3} else (),
            )
            for frame in range(4)
        )
        source = LtaVolumeSpec(
            source_role=SourceRole.TARGET,
            source_root=root,
            volume_id="input:source",
            stem="source",
            kind="video",
            media=(),
            video_path=video,
            video_sha256=None,
            video_identity_sha256="b" * 64,
            annotations=annotations,
            volume_class=VolumeClass.PARTIALLY_LABELED,
            encoded_indices=(0, 1, 2, 3),
            index_origin=0,
            frame_count=4,
            width=6,
            height=4,
            fps=1.0,
        )
        discovery = LtaInputDiscovery(
            input_path=video,
            target_volumes=(source,),
            exemplar_roots=(),
            exemplar_volumes=(),
            positive_pool=(),
            warnings=(),
        )
        physical = get_view_infos(
            T=4,
            H=4,
            W=6,
            cartesian_views=("transverse",),
            azimuthal_views=(),
            azimuthal_azimuth_angles=(),
            tilt_groups=(),
        )[0]
        runtime = expand_views_into_tta_variants((physical,), (0.0,))[0]
        grid = LtaTileGridPlan(
            config_id="s4_st2",
            tile_size=4,
            tile_stride=2,
            tiles=plan_tile_grid(
                source_width=6,
                source_height=4,
                tile_size=4,
                tile_stride=2,
            ),
        )
        view = LtaRuntimeViewPlan(
            volume_id=source.volume_id,
            physical_view_id="transverse",
            runtime_view_id=runtime.name,
            tta_angle_deg=0.0,
            frame_count=4,
            frame_height=4,
            frame_width=6,
            sessions=plan_sam_sessions("sequence", 4),
            tile_config_ids=(grid.config_id,),
            tile_grids=(grid,),
            encoded_frame_indices=(0, 1, 2, 3),
            raster_plan_digest="c" * 64,
            runtime_view=runtime,
        )
        volume_plan = LtaVolumePlan(
            volume_id=source.volume_id,
            stem="source",
            source_shape_tyx=(4, 4, 6),
            runtime_views=(view,),
        )
        bundle = LocalSamBundle(
            root=root,
            checkpoint_path=checkpoint,
            model_version="sam3.1",
            checkpoint_identity_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        )
        return LtaRunPlan(
            run_id=uuid.uuid4().hex,
            bundle=bundle,
            discovery=discovery,
            output_root=root / "output",
            temp_root=root / "output" / "temp" / "run",
            device_ids=devices,
            sam_execution="video",
            conf=0.15,
            save_tokens=("summary", "voxel_volume"),
            command=("xta", "--mode", "lta"),
            postprocessing={
                "keep_objects": 0,
                "enable_3d_void_fill": False,
                "gaussian_smoothing_enabled": False,
                "gaussian_sigma": 3.0,
                "gaussian_passes": 1,
            },
            volumes=(volume_plan,),
        )

    def _run(
        self,
        root: Path,
        devices: tuple[int, ...],
        *,
        preserve_temp: bool = False,
        cleanup_failure: bool = False,
        revalidation_failure: bool = False,
    ):
        pools = []

        def pool_factory(*args, **kwargs):
            pool = _FakeWorkerPool(*args, **kwargs)
            pools.append(pool)
            return pool

        def source_loader(source, *, path):
            volume = np.memmap(path, dtype=np.uint8, mode="w+", shape=(4, 4, 6))
            volume[:] = 0
            volume.flush()
            return volume

        def backproject(view_union, _view, out_path, _desc, **_kwargs):
            self.assertTrue(_kwargs["allow_transverse_passthrough"])
            result = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(4, 4, 6))
            result[:] = view_union
            result.flush()
            return result

        def finalizer(layers, **_kwargs):
            return _Finalization(layers)

        def nrrd_writer(output_dir, *, stem, terminal_union, **_kwargs):
            path = Path(output_dir) / f"{stem}_Global_final_output.seg.nrrd"
            path.write_bytes(np.asarray(terminal_union, dtype=np.uint8).tobytes())
            layer = LtaLayerRecord(
                layer_id="global_final_output",
                recomposition_op=LtaRecompositionOp.SELECT,
                source_role="global_final_output",
                volume=terminal_union,
            )
            return LtaFinalNrrdArtifact(
                layer=layer,
                receipt=LtaArtifactReceipt(
                    name="global_final_output.nrrd",
                    path=path,
                    sha256=_sha256(path),
                ),
            )

        plan = self._plan(root, devices)
        if preserve_temp and cleanup_failure:
            raise ValueError("test helper cleanup modes are mutually exclusive")
        if cleanup_failure:
            cleanup_context = mock.patch(
                "XTA.lta_execution.shutil.rmtree",
                side_effect=OSError("injected scratch cleanup failure"),
            )
        elif preserve_temp:
            cleanup_context = mock.patch("XTA.lta_execution.shutil.rmtree")
        else:
            cleanup_context = contextlib.nullcontext()
        input_revalidation_context = mock.patch(
            "XTA.lta_inputs.revalidate_lta_input_identities",
            side_effect=(
                RuntimeError("injected input revalidation failure")
                if revalidation_failure
                else None
            ),
        )
        with (
            mock.patch("XTA.lta_execution.PRODUCTION_LTA_TILE_SIZE", 4),
            mock.patch("XTA.lta_execution.LtaWorkerPool", side_effect=pool_factory),
            mock.patch("XTA.lta_execution._materialize_source_volume", side_effect=source_loader),
            mock.patch(
                "XTA.lta_execution.fill_completed_view_holes_2d_inplace",
                return_value=types.SimpleNamespace(manifest_record=lambda: {"added_pixels": 0}),
            ),
            mock.patch(
                "XTA.lta_execution.fill_merged_seed_mask_holes_2d",
                side_effect=lambda mask, **_kwargs: types.SimpleNamespace(
                    mask=np.asarray(mask, dtype=bool).copy(),
                    receipt=types.SimpleNamespace(
                        manifest_record=lambda: {"added_pixels": 0}
                    ),
                ),
            ),
            mock.patch(
                "XTA.assembly.project_view_volume_to_orthogonal_volume",
                side_effect=backproject,
            ),
            mock.patch("XTA.lta_execution.finalize_lta_native_union", side_effect=finalizer),
            mock.patch("XTA.lta_execution.write_global_final_output_nrrd", side_effect=nrrd_writer),
            mock.patch("XTA.lta_execution.revalidate_local_sam_bundle"),
            input_revalidation_context,
            cleanup_context,
        ):
            result = execute_lta_plan(plan, workers=1)
        self.assertTrue(result.manifest_path.is_file())
        self.assertTrue(result.final_nrrd_path.is_file())
        return result, pools[0], result.final_nrrd_path.read_bytes()

    def test_resized_index_aligned_exemplars_become_direct_mask_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = self._plan(root, (0,))
            source = plan.discovery.target_volumes[0]
            unknown = tuple(
                replace(
                    annotation,
                    state=AnnotationState.UNKNOWN,
                    label_path=None,
                    label_sha256=None,
                    polygons=(),
                )
                for annotation in source.annotations
            )
            unlabeled = replace(
                source,
                annotations=unknown,
                volume_class=VolumeClass.UNLABELED,
            )
            exemplar_image = root / "aligned_0001.png"
            exemplar_label = root / "aligned_0001.txt"
            exemplar_image.write_bytes(b"image")
            exemplar_label.write_text("label", encoding="utf-8")
            polygon = plan.discovery.target_volumes[0].annotations[0].polygons[0]
            exemplar = PositiveExemplar(
                exemplar_id="aligned",
                source_role=SourceRole.EXEMPLAR,
                source_root=root,
                volume_id="exemplar:aligned",
                volume_stem="aligned",
                volume_kind="sequence",
                encoded_frame_index=1,
                frame_position=0,
                media_path=exemplar_image,
                media_sha256=None,
                media_identity_sha256="e" * 64,
                label_path=exemplar_label,
                label_sha256="f" * 64,
                label_row_index=0,
                class_id=0,
                polygon=polygon.points,
                box_xyxy=polygon.box_xyxy,
                box_cxcywh=polygon.box_cxcywh,
                normalized_area=polygon.normalized_area,
                bundle_sha256="0" * 64,
                source_width=1008,
                source_height=1008,
            )
            discovery = replace(
                plan.discovery,
                target_volumes=(unlabeled,),
                positive_pool=(exemplar,),
            )
            aligned_plan = replace(plan, discovery=discovery, exemplar_index_origin=1)

            annotations = _aligned_annotations(aligned_plan, unlabeled)

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].frame_position, 0)
        self.assertEqual(annotations[0].polygons[0].points, polygon.points)

    def test_consumed_worker_artifacts_are_removed_and_cache_path_is_not_published(self) -> None:
        _FakeWorkerPool.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            result, _pool, _bytes = self._run(
                Path(temp_dir),
                (0, 1),
                preserve_temp=True,
            )
            temp_root = result.plan.temp_root
            self.assertTrue(temp_root.is_dir())
            self.assertEqual(list(temp_root.rglob("union.uint8.raw")), [])
            self.assertEqual(list(temp_root.rglob("manifest.json")), [])
            self.assertEqual(list(temp_root.rglob("*.npz")), [])
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            cache_record = manifest["execution"]["view_cache"]
            self.assertNotIn("path", cache_record)
            self.assertEqual(
                cache_record["lifecycle"],
                "ephemeral_removed_before_manifest_publication",
            )

    def test_fresh_sparse_workspaces_read_as_zero_without_eager_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = types.SimpleNamespace(frame_count=2, height=3, width=4)
            view = types.SimpleNamespace(frame_count=2, frame_height=3, frame_width=4)
            hard_positive = _hard_positive_volume(
                source,
                (),
                path=root / "hard-positive.raw",
            )
            view_union = _allocate_view_union(view, root / "view-union.raw")
            try:
                self.assertFalse(np.asarray(hard_positive, dtype=bool).any())
                self.assertFalse(np.asarray(view_union, dtype=bool).any())
            finally:
                for volume in (hard_positive, view_union):
                    mmap_obj = getattr(volume, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()

    def test_complete_manifest_is_not_published_before_scratch_cleanup(self) -> None:
        _FakeWorkerPool.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(OSError, "injected scratch cleanup failure"):
                self._run(root, (0,), cleanup_failure=True)
            self.assertFalse((root / "output" / "manifest.json").exists())

    def test_relay_safety_bound_fails_instead_of_discarding_new_mask_work(self) -> None:
        _FakeWorkerPool.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                mock.patch("XTA.lta_execution._relay_generation_bound", return_value=0),
                self.assertRaisesRegex(RuntimeError, "did not reach a mask fixed point"),
            ):
                self._run(root, (0,))
            self.assertFalse((root / "output" / "manifest.json").exists())

    def test_revalidation_precedes_every_public_artifact(self) -> None:
        _FakeWorkerPool.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(
                RuntimeError,
                "injected input revalidation failure",
            ):
                self._run(root, (0,), revalidation_failure=True)
            output = root / "output"
            self.assertEqual(list(output.glob("*.nrrd")), [])
            self.assertEqual(list(output.glob("*_summary.json")), [])
            self.assertEqual(list(output.glob("*_voxel_volume.json")), [])
            self.assertFalse((output / "manifest.json").exists())

    def test_manifest_replaces_preflight_schedule_with_actual_dynamic_dispatches(self) -> None:
        _FakeWorkerPool.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            result, pool, _bytes = self._run(Path(temp_dir), (0, 1, 2, 3))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        preflight = manifest["run_plan"]["device_schedule"]
        self.assertEqual(preflight["status"], "superseded")
        self.assertEqual(preflight["superseded_by"], "execution.device_schedule")
        self.assertNotIn("work", preflight)

        actual = manifest["execution"]["device_schedule"]
        work = actual["work"]
        self.assertEqual(actual["status"], "settled")
        self.assertTrue(actual["coordinator_selected"])
        self.assertEqual(actual["work_count"], len(pool.used_devices))
        self.assertEqual(len(work), len(pool.used_devices))
        self.assertEqual(len({item["work_id"] for item in work}), len(work))
        self.assertEqual(
            sorted(item["execution_device_id"] for item in work),
            sorted(pool.used_devices),
        )
        self.assertTrue(any(item["relay_generation"] > 0 for item in work))
        self.assertTrue(any(item["tail_assist"] for item in work))
        for item in work:
            self.assertEqual(
                item["tail_assist"],
                item["owner_device_id"] != item["execution_device_id"],
            )
            self.assertLess(item["frame_start"], item["frame_stop"])
            self.assertIn("batch-", item["work_id"])
        audit = manifest["execution"]["worker_audit"]
        self.assertEqual(audit["chain_count"], len(work))
        self.assertEqual(audit["retained_prediction_count"], 0)
        self.assertEqual(
            sorted(int(device) for device in audit["profiles_by_device"]),
            sorted(set(pool.used_devices)),
        )
        self.assertEqual(
            audit["relay_gate"],
            {
                "minimum_overlap_pixels": 16,
                "minimum_tracker_probability": 0.5,
            },
        )

    def test_dense_initial_and_relay_prompts_are_batched_before_worker_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = self._plan(root, (0,))
            view_plan = plan.volumes[0].runtime_views[0]
            grid = view_plan.tile_grids[0]
            temp_root = root / "dense-temp"
            temp_root.mkdir()
            cache_ref = types.SimpleNamespace(
                identity_sha256="cache-identity",
                payload=lambda: {
                    "path": str(root / "cache.raw"),
                    "shape": [4, 4, 6],
                    "dtype": "uint8",
                    "physical_view_id": "transverse",
                    "identity_sha256": "cache-identity",
                    "size_bytes": 96,
                    "mtime_ns": 0,
                },
            )
            mask_shape = (16, 16)
            seeds = tuple(
                LtaMaskSeed(
                    lineage=LtaLineageId(
                        volume_id=view_plan.volume_id,
                        physical_view_id=view_plan.physical_view_id,
                        runtime_view_id=view_plan.runtime_view_id,
                        tile_config_id=grid.config_id,
                        lineage_id=f"dense-{index:04d}",
                    ),
                    frame_index=0,
                    object_id=index,
                    mask=np.eye(
                        1,
                        mask_shape[0] * mask_shape[1],
                        index,
                        dtype=bool,
                    ).reshape(mask_shape),
                    visited_tile_indices=(0,),
                )
                for index in range(LTA_MAX_NUM_OBJECTS + 1)
            )

            with mock.patch(
                "XTA.lta_execution._seeds_for_tile",
                side_effect=lambda _source, _annotations, _view, _grid, tile_index, **_kwargs: (
                    {0: seeds} if int(tile_index) == 0 else {}
                ),
            ):
                initial, inventory = _plan_initial_chains(
                    plan.discovery.target_volumes[0],
                    (),
                    view_plan,
                    cache_ref,
                    temp_root=temp_root,
                    conf=0.15,
                    empty_frame_limit=30,
                )

            self.assertEqual(len(initial), 2)
            self.assertEqual(
                [
                    len(read_seed_artifact(chain.payload["seed_artifact_path"]))
                    for chain in initial
                ],
                [LTA_MAX_NUM_OBJECTS, 1],
            )
            self.assertTrue(initial[0].work.work_id.endswith("batch-0000-of-0002"))
            self.assertTrue(initial[1].work.work_id.endswith("batch-0001-of-0002"))
            self.assertEqual(
                [
                    seed.object_id
                    for seed in read_seed_artifact(
                        initial[1].payload["seed_artifact_path"]
                    )
                ],
                [0],
            )
            self.assertEqual(len(inventory[(grid.config_id, 0, 0)]), len(seeds))

            # Session planning must also split masks that SAM cannot represent
            # together, even when the object-count cap would permit one batch.
            overlap_mask = np.zeros(mask_shape, dtype=bool)
            overlap_mask[2:6, 3:7] = True
            overlapping_seeds = tuple(
                replace(seed, mask=overlap_mask.copy()) for seed in seeds[:2]
            )
            overlap_temp_root = root / "overlap-temp"
            overlap_temp_root.mkdir()
            with mock.patch(
                "XTA.lta_execution._seeds_for_tile",
                side_effect=lambda _source, _annotations, _view, _grid, tile_index, **_kwargs: (
                    {0: overlapping_seeds} if int(tile_index) == 0 else {}
                ),
            ):
                overlap_initial, _overlap_inventory = _plan_initial_chains(
                    plan.discovery.target_volumes[0],
                    (),
                    view_plan,
                    cache_ref,
                    temp_root=overlap_temp_root,
                    conf=0.15,
                    empty_frame_limit=30,
                )
            self.assertEqual(len(overlap_initial), 2)
            self.assertEqual(
                [
                    len(read_seed_artifact(chain.payload["seed_artifact_path"]))
                    for chain in overlap_initial
                ],
                [1, 1],
            )
            self.assertTrue(
                overlap_initial[0].work.work_id.endswith("batch-0000-of-0002")
            )
            self.assertTrue(
                overlap_initial[1].work.work_id.endswith("batch-0001-of-0002")
            )

            relay_records = tuple(
                {
                    "lineage": {
                        "volume_id": seed.lineage.volume_id,
                        "physical_view_id": seed.lineage.physical_view_id,
                        "runtime_view_id": seed.lineage.runtime_view_id,
                        "tile_config_id": seed.lineage.tile_config_id,
                        "lineage_id": seed.lineage.lineage_id,
                    },
                    "source_tile_index": 0,
                    "destination_tile_index": 1,
                    "frame_index": 1,
                    "temporal_direction": "forward",
                    "generation": 1,
                    "seed_artifact_path": str(root / f"unused-{index}.npz"),
                    "seed_artifact_sha256": "0" * 64,
                }
                for index, seed in enumerate(seeds)
            )

            def merged(group, *, generation, **_kwargs):
                record = group[0]
                lineage_index = int(
                    str(record["lineage"]["lineage_id"]).rsplit("-", 1)[-1]
                )
                return LtaMaskSeed(
                    lineage=_lineage_from_record(record["lineage"]),
                    frame_index=int(record["frame_index"]),
                    object_id=0,
                    mask=np.eye(
                        1,
                        mask_shape[0] * mask_shape[1],
                        lineage_index,
                        dtype=bool,
                    ).reshape(mask_shape),
                    provenance=LtaSeedProvenance.SPATIAL_RELAY,
                    relay_generation=generation,
                    visited_tile_indices=(0, 1),
                )

            scheduler = LtaViewAffinityScheduler(
                (chain.work for chain in initial),
                (0,),
                max_relay_generation=_relay_generation_bound(view_plan),
            )
            self.assertGreater(
                scheduler.max_relay_generation,
                len(grid.tiles) - 1,
            )
            with (
                mock.patch(
                    "XTA.lta_execution._verified_relay_artifact_paths",
                    return_value=(),
                ),
                mock.patch(
                    "XTA.lta_execution._select_and_merge_relay_group",
                    side_effect=merged,
                ),
                mock.patch(
                    "XTA.lta_execution.fill_merged_seed_mask_holes_2d",
                    side_effect=lambda value, **_kwargs: types.SimpleNamespace(
                        mask=np.asarray(value, dtype=bool).copy(),
                        receipt=types.SimpleNamespace(
                            manifest_record=lambda: {"added_pixels": 0}
                        ),
                    ),
                ),
            ):
                relays = _plan_relay_generation(
                    relay_records,
                    generation=1,
                    view_plan=view_plan,
                    cache_ref=cache_ref,
                    scheduler=scheduler,
                    relay_mask_revisions={},
                    temp_root=temp_root,
                    conf=0.15,
                    empty_frame_limit=30,
                    first_plan_order=len(initial),
                )

            self.assertEqual(len(relays), 2)
            self.assertEqual(
                [
                    len(read_seed_artifact(chain.payload["seed_artifact_path"]))
                    for chain in relays
                ],
                [LTA_MAX_NUM_OBJECTS, 1],
            )
            self.assertTrue(relays[0].work.work_id.endswith("batch-0000-of-0002"))
            self.assertTrue(relays[1].work.work_id.endswith("batch-0001-of-0002"))
            relay_seeds = tuple(
                seed
                for chain in relays
                for seed in read_seed_artifact(chain.payload["seed_artifact_path"])
            )
            self.assertEqual(
                [seed.lineage.lineage_id for seed in relay_seeds],
                [f"dense-{index:04d}" for index in range(len(seeds))],
            )
            self.assertEqual(
                [int(np.flatnonzero(seed.mask)[0]) for seed in relay_seeds],
                list(range(len(seeds))),
            )

            def merged_overlap(group, *, generation, **_kwargs):
                record = group[0]
                return LtaMaskSeed(
                    lineage=_lineage_from_record(record["lineage"]),
                    frame_index=int(record["frame_index"]),
                    object_id=0,
                    mask=overlap_mask.copy(),
                    provenance=LtaSeedProvenance.SPATIAL_RELAY,
                    relay_generation=generation,
                    visited_tile_indices=(0, 1),
                )

            overlap_scheduler = LtaViewAffinityScheduler(
                (chain.work for chain in overlap_initial),
                (0,),
                max_relay_generation=_relay_generation_bound(view_plan),
            )
            with (
                mock.patch(
                    "XTA.lta_execution._verified_relay_artifact_paths",
                    return_value=(),
                ),
                mock.patch(
                    "XTA.lta_execution._select_and_merge_relay_group",
                    side_effect=merged_overlap,
                ),
                mock.patch(
                    "XTA.lta_execution.fill_merged_seed_mask_holes_2d",
                    side_effect=lambda value, **_kwargs: types.SimpleNamespace(
                        mask=np.asarray(value, dtype=bool).copy(),
                        receipt=types.SimpleNamespace(
                            manifest_record=lambda: {"added_pixels": 0}
                        ),
                    ),
                ),
            ):
                overlap_relays = _plan_relay_generation(
                    relay_records[:2],
                    generation=1,
                    view_plan=view_plan,
                    cache_ref=cache_ref,
                    scheduler=overlap_scheduler,
                    relay_mask_revisions={},
                    temp_root=overlap_temp_root,
                    conf=0.15,
                    empty_frame_limit=30,
                    first_plan_order=len(overlap_initial),
                )
            self.assertEqual(len(overlap_relays), 2)
            self.assertEqual(
                [
                    len(read_seed_artifact(chain.payload["seed_artifact_path"]))
                    for chain in overlap_relays
                ],
                [1, 1],
            )
            self.assertEqual(
                [
                    read_seed_artifact(chain.payload["seed_artifact_path"])[
                        0
                    ].lineage.lineage_id
                    for chain in overlap_relays
                ],
                ["dense-0000", "dense-0001"],
            )
            self.assertTrue(
                overlap_relays[0].work.work_id.endswith("batch-0000-of-0002")
            )
            self.assertTrue(
                overlap_relays[1].work.work_id.endswith("batch-0001-of-0002")
            )

            # The production state-space bound admits a legitimate relay wave
            # deeper than a simple acyclic tile path.
            view_key = initial[0].work.view
            scheduler.mark_projection_ready(view_key, device_id=0)
            while True:
                claim = scheduler.claim(0)
                if claim is None:
                    break
                scheduler.complete(claim, claim.work.work_id)
                scheduler.drain_committable()
            scheduler.register_generation(view_key, 1, ())
            scheduler.seal_generation(view_key, 1)
            scheduler.register_generation(view_key, 2, ())
            scheduler.seal_generation(view_key, 2)
            self.assertTrue(scheduler.generation_settled(view_key, 2))

    def test_one_and_four_device_fixed_points_are_byte_identical(self) -> None:
        _FakeWorkerPool.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            one, one_pool, one_bytes = self._run(root / "one", (0,))
            four, four_pool, four_bytes = self._run(root / "four", (0, 1, 2, 3))

        self.assertEqual(one_bytes, four_bytes)
        self.assertEqual(one.relay_generations, 1)
        self.assertEqual(four.relay_generations, 1)
        self.assertEqual(set(one_pool.used_devices), {0})
        self.assertGreaterEqual(len(set(four_pool.used_devices)), 2)
        self.assertTrue(
            all(0.0 < value <= 4 * 60 * 60 for value in four_pool.wait_timeouts)
        )
        final = np.frombuffer(four_bytes, dtype=np.uint8).reshape(4, 4, 6)
        self.assertTrue(
            final[1:, 1, 5].all(),
            "relay did not reach the neighbor-only edge after its boundary frame",
        )


if __name__ == "__main__":
    unittest.main()
