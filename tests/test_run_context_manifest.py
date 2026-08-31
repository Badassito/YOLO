from __future__ import annotations

import enum
import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from XTA.unification.context import (
    activate_unified_launch,
    current_unified_launch,
)
from XTA.unification.manifest import json_compatible, write_json_manifest
from XTA.unification.tta_manifest import (
    assert_tta_artifacts_unchanged,
    build_tta_run_manifest,
    capture_tta_artifact_identities,
)


class _Choice(enum.Enum):
    VALUE = "value"


@dataclass(frozen=True)
class _Record:
    path: Path
    values: tuple[int, ...]


@dataclass(frozen=True)
class _View:
    name: str = "radial_transverse"
    physical_view_name: str = ""
    family: str = "radial"
    summary_family: str = "radial"
    display_name: str = "Radial Transverse"
    num_slices: int = 3
    src_h: int = 5
    src_w: int = 7
    pad_mode: str = "pad"
    azimuths_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    diameter: int = 7
    center_x: float = 3.0
    center_y: float = 2.0
    roi_radius: float = 3.0
    full_t: int = 3
    full_h: int = 5
    full_w: int = 7
    tilt_angle_deg: float = 0.0
    tilt_direction: str = ""
    tilt_frame_start: int = 0
    tilt_frame_stop: int = 3
    tilt_base_view: str = "transverse"
    horizontal_axis: str = "x"
    vertical_axis: str = "y"
    stack_axis: str = "t"
    radial_base_view: str = "transverse"
    radial_tilted_source: bool = False
    radial_source_view_name: str = ""
    radial_request_token: str = "transverse"
    tta_aug_id: str = ""
    tta_angle_deg: float = 0.0


@dataclass(frozen=True)
class _Channel:
    token: str = "C3S1"
    kind: str = "custom"
    channel_count: int = 3
    stride: int = 1
    offsets: tuple[int, ...] = (-1, 0, 1)


@dataclass(frozen=True)
class _Tile:
    config_id: str = "tile_4_2"
    tile_size: int = 4
    tile_stride: int = 2


@dataclass(frozen=True)
class _RadialRequest:
    view: str = "transverse"
    azimuth_angle: float | None = 45.0


class RunContextManifestTests(unittest.TestCase):
    def test_launch_context_nests_and_restores(self) -> None:
        self.assertIsNone(current_unified_launch())
        with activate_unified_launch(
            version="18.0.2",
            launcher="GPT-5.6-Sol-Ultra_v18.0.2_SLURM.py",
            mode="tta",
            mode_arguments=("--input", "volume.mkv"),
        ) as outer:
            self.assertIs(current_unified_launch(), outer)
            self.assertEqual(
                outer.command,
                (
                    "GPT-5.6-Sol-Ultra_v18.0.2_SLURM.py",
                    "--mode",
                    "tta",
                    "--input",
                    "volume.mkv",
                ),
            )
            with activate_unified_launch(
                version="18.0.2",
                launcher="GPT-5.6-Sol-Ultra_v18.0.2_SLURM.py",
                mode="pta",
                mode_arguments=("--input", "dataset"),
            ) as inner:
                self.assertIs(current_unified_launch(), inner)
            self.assertIs(current_unified_launch(), outer)
        self.assertIsNone(current_unified_launch())

    def test_manifest_conversion_and_atomic_output_are_deterministic(self) -> None:
        payload = {
            "choice": _Choice.VALUE,
            "record": _Record(Path("input.mkv"), (3, 1)),
            "unordered": {"b", "a"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "run_manifest.json"
            first = write_json_manifest(path, payload).read_bytes()
            second = write_json_manifest(path, payload).read_bytes()
            leftovers = tuple(Path(temp_dir).glob(".*.tmp"))

        self.assertEqual(first, second)
        self.assertEqual(leftovers, ())
        decoded = json.loads(first)
        self.assertEqual(decoded["choice"], "value")
        self.assertEqual(decoded["record"]["path"], "input.mkv")
        self.assertEqual(decoded["unordered"], ["a", "b"])

    def test_manifest_rejects_nonfinite_or_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            json_compatible({"bad": float("nan")})
        with self.assertRaises(TypeError):
            json_compatible({"bad": object()})

    def test_tta_manifest_captures_resolved_geometry_and_artifact_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mkv"
            model = root / "model.engine"
            source.write_bytes(b"source")
            model.write_bytes(b"model")
            identities = capture_tta_artifact_identities(
                input_path=source,
                gpu_model_path=model,
                cpu_model_path=None,
            )
            with activate_unified_launch(
                version="18.0.2",
                launcher="GPT-5.6-Sol-Ultra_v18.0.2_SLURM.py",
                mode="tta",
                mode_arguments=("--input", str(source)),
            ) as context:
                payload = build_tta_run_manifest(
                    launch_context=context,
                    pipeline_version="18.0.2",
                    resolved_config={"imgsz": 7, "angle": [0.0, 120.0]},
                    artifact_identities=identities,
                    source_shape_tyx=(3, 5, 7),
                    processing_shape_tyx=(3, 5, 7),
                    fps=24.0,
                    physical_views=(_View(),),
                    inference_views=(
                        _View(
                            name="radial_transverse__tta_a120",
                            physical_view_name="radial_transverse",
                            tta_aug_id="a120",
                            tta_angle_deg=120.0,
                        ),
                    ),
                    angles=(0.0, 120.0),
                    channel_format=_Channel(),
                    tile_configs=(_Tile(),),
                    radial_requests=(_RadialRequest(),),
                    radial_diameters=(7,),
                    radial_azimuth_angles=(45.0,),
                    backend={
                        "inference_devices": ["cuda:0", "cpu:0"],
                        "gpu_precision": "fp16",
                        "cpu_precision": "bf16",
                        "cpu_worker_ready_details": {
                            "0": {
                                "precision": "bf16",
                                "input_element_type": "f32",
                                "model_int8_quantized": False,
                            }
                        },
                        "physical_view_backend_ownership": [
                            {
                                "physical_view": "radial_transverse",
                                "contract": "hybrid_frame_partition",
                                "frames": {"cpu": 1, "gpu": 2},
                            }
                        ],
                    },
                    forward_sampling={
                        "radial_source_mode": "texture_linear",
                        "cube_t_axis_resize_backend": "trilinear",
                        "image_capture": "canonical_render_batch_tee",
                        "runtime_cuda_renderer_fallback_capture": (
                            "pending_gpu_qualification"
                        ),
                    },
                    prediction_processing={"owner": "tta_only"},
                    requested_outputs=("images", "voxel_volume"),
                    output_paths={
                        "run_manifest": root / "manifest.json",
                        "voxel_volume": root / "source_VoxelVolume.json",
                    },
                    output_metadata={
                        "voxel_volume": {
                            "schema": "xta.v18.voxel_count/1",
                            "voxel_count": 11,
                            "units": "foreground_voxels",
                            "shape_t_y_x": [3, 5, 7],
                        },
                        "model_input_images": {
                            "fanout": "canonical_render_batch_tee",
                            "device_resident_cuda_paths": "pending",
                        },
                    },
                )
            manifest_path = root / "manifest.json"
            write_json_manifest(manifest_path, payload)
            decoded = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["launcher"]["mode"], "tta")
        self.assertTrue(payload["inputs"]["captured_before_execution"])
        self.assertEqual(payload["inputs"]["source"]["size_bytes"], 6)
        gpu_model = payload["inputs"]["models"]["gpu"]
        self.assertEqual(gpu_model["requested_path"], str(model.resolve()))
        self.assertEqual(gpu_model["artifacts"][0]["size_bytes"], 5)
        self.assertEqual(
            gpu_model["artifacts"][0]["sha256"],
            hashlib.sha256(b"model").hexdigest(),
        )
        self.assertIsNone(payload["inputs"]["models"]["cpu"])
        self.assertEqual(payload["geometry"]["channel_format"]["direction"], "forward")
        self.assertEqual(payload["geometry"]["tta_angles_deg"], [0.0, 120.0])
        radial = payload["geometry"]["radial_groups"][0]
        self.assertEqual(radial["resolved_azimuth_angle_deg"], 45.0)
        self.assertEqual(
            radial["concrete_azimuth_vectors"][0]["azimuths_deg"],
            [0.0, 45.0, 90.0, 135.0],
        )
        self.assertTrue(
            payload["forward_sampling"][
                "same_backend_builtin_geometry_shared_with_pta"
            ]
        )
        self.assertEqual(
            decoded["forward_sampling"]["image_capture"],
            "canonical_render_batch_tee",
        )
        self.assertEqual(
            decoded["inference_backend"]["cpu_worker_ready_details"]["0"][
                "precision"
            ],
            "bf16",
        )
        self.assertEqual(
            decoded["inference_backend"]["physical_view_backend_ownership"][0][
                "frames"
            ],
            {"cpu": 1, "gpu": 2},
        )
        self.assertEqual(
            decoded["outputs"]["artifacts"]["voxel_volume"]["voxel_count"],
            11,
        )
        self.assertEqual(
            decoded["outputs"]["artifacts"]["model_input_images"]["fanout"],
            "canonical_render_batch_tee",
        )

    def test_openvino_identity_covers_xml_bin_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mkv"
            export = root / "openvino_export"
            export.mkdir()
            source.write_bytes(b"source")
            xml = export / "openvino_export.xml"
            weights = export / "openvino_export.bin"
            metadata = export / "metadata.json"
            xml.write_bytes(b"<net />")
            weights.write_bytes(b"weights")
            metadata.write_bytes(b'{"stride": 32}')

            identities = capture_tta_artifact_identities(
                input_path=source,
                gpu_model_path=None,
                cpu_model_path=export,
            )

        cpu_model = identities["models"]["cpu"]
        self.assertEqual(cpu_model["requested_path"], str(export.resolve()))
        artifacts = cpu_model["artifacts"]
        self.assertEqual(
            [Path(record["path"]).name for record in artifacts],
            ["openvino_export.xml", "openvino_export.bin", "metadata.json"],
        )
        self.assertEqual(
            [record["sha256"] for record in artifacts],
            [
                hashlib.sha256(b"<net />").hexdigest(),
                hashlib.sha256(b"weights").hexdigest(),
                hashlib.sha256(b'{"stride": 32}').hexdigest(),
            ],
        )
        for record in artifacts:
            self.assertIn("modified_time_ns", record)
            self.assertIn("change_or_creation_time_ns", record)
            self.assertIn("file_id", record)

    def test_completion_guard_detects_source_and_openvino_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mkv"
            export = root / "cpu_model"
            export.mkdir()
            xml = export / "cpu_model.xml"
            weights = export / "cpu_model.bin"
            source.write_bytes(b"source")
            xml.write_bytes(b"<net />")
            weights.write_bytes(b"weights")

            identities = capture_tta_artifact_identities(
                input_path=source,
                gpu_model_path=None,
                cpu_model_path=export,
            )
            assert_tta_artifacts_unchanged(identities)
            source.write_bytes(b"source changed")
            with self.assertRaisesRegex(RuntimeError, "source.mkv"):
                assert_tta_artifacts_unchanged(identities)

            source.write_bytes(b"source")
            identities = capture_tta_artifact_identities(
                input_path=source,
                gpu_model_path=None,
                cpu_model_path=export,
            )
            weights.write_bytes(b"weights changed")
            with self.assertRaisesRegex(RuntimeError, "cpu_model.bin"):
                assert_tta_artifacts_unchanged(identities)


if __name__ == "__main__":
    unittest.main()
