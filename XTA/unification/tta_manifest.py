"""Resolved TTA run-manifest construction for the unified v18 launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .context import UnifiedLaunchContext


TTA_RUN_MANIFEST_SCHEMA = "xta.v18.run_manifest/1"
TTA_VOXEL_COUNT_SCHEMA = "xta.v18.voxel_count/1"


_VIEW_FIELDS = (
    "name",
    "physical_view_name",
    "family",
    "summary_family",
    "display_name",
    "num_slices",
    "src_h",
    "src_w",
    "pad_mode",
    "azimuths_deg",
    "diameter",
    "center_x",
    "center_y",
    "roi_radius",
    "full_t",
    "full_h",
    "full_w",
    "tilt_angle_deg",
    "tilt_direction",
    "tilt_frame_start",
    "tilt_frame_stop",
    "tilt_base_view",
    "horizontal_axis",
    "vertical_axis",
    "stack_axis",
    "radial_base_view",
    "radial_tilted_source",
    "radial_source_view_name",
    "radial_request_token",
    "tta_aug_id",
    "tta_angle_deg",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(
    path: str | Path,
    *,
    content_digest: bool = False,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
        "change_or_creation_time_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "file_id": int(stat.st_ino),
    }
    if content_digest:
        if not resolved.is_file():
            raise ValueError(f"content identity requires a regular file: {resolved}")
        identity["sha256"] = _sha256_file(resolved)
    return identity


def _resolve_openvino_artifacts(path: str | Path) -> tuple[Path, ...]:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        if requested.suffix.lower() == ".xml":
            model_xml = requested
        elif requested.suffix.lower() == ".bin" and requested.with_suffix(".xml").is_file():
            model_xml = requested.with_suffix(".xml")
        else:
            raise ValueError(
                "OpenVINO model identity requires an IR XML/BIN file or export directory: "
                f"{requested}"
            )
    elif requested.is_dir():
        preferred = requested / f"{requested.name}.xml"
        xml_files = sorted(candidate for candidate in requested.glob("*.xml") if candidate.is_file())
        if preferred.is_file():
            model_xml = preferred
        elif len(xml_files) == 1:
            model_xml = xml_files[0]
        elif not xml_files:
            raise FileNotFoundError(f"no OpenVINO IR XML exists under {requested}")
        else:
            raise ValueError(
                f"OpenVINO export directory has multiple XML files: {requested}"
            )
    else:
        raise FileNotFoundError(requested)

    artifacts = [model_xml]
    model_bin = model_xml.with_suffix(".bin")
    if not model_bin.is_file():
        raise FileNotFoundError(
            f"OpenVINO IR weights file is missing for {model_xml}: {model_bin}"
        )
    artifacts.append(model_bin)
    for name in ("metadata.yaml", "metadata.yml", "metadata.json"):
        metadata = model_xml.parent / name
        if metadata.is_file():
            artifacts.append(metadata)
    return tuple(artifacts)


def capture_tta_artifact_identities(
    *,
    input_path: str | Path,
    gpu_model_path: str | Path | None,
    cpu_model_path: str | Path | None,
) -> dict[str, Any]:
    """Capture artifacts before decode/model execution begins.

    The potentially enormous source video uses a strong filesystem snapshot and
    is checked again at completion. Model artifacts are content-digested because
    OpenVINO's companion BIN can change independently of its XML.
    """

    models: dict[str, Any] = {"gpu": None, "cpu": None}
    if gpu_model_path is not None:
        gpu_path = Path(gpu_model_path).expanduser().resolve()
        if gpu_path.is_file():
            gpu_artifacts = (gpu_path,)
        elif gpu_path.is_dir():
            gpu_artifacts = tuple(
                sorted(candidate for candidate in gpu_path.rglob("*") if candidate.is_file())
            )
            if not gpu_artifacts:
                raise ValueError(f"GPU model directory contains no files: {gpu_path}")
        else:
            raise FileNotFoundError(gpu_path)
        models["gpu"] = {
            "requested_path": str(gpu_path),
            "artifacts": [
                _file_identity(artifact, content_digest=True)
                for artifact in gpu_artifacts
            ],
        }
    if cpu_model_path is not None:
        cpu_path = Path(cpu_model_path).expanduser().resolve()
        models["cpu"] = {
            "requested_path": str(cpu_path),
            "artifacts": [
                _file_identity(artifact, content_digest=True)
                for artifact in _resolve_openvino_artifacts(cpu_path)
            ],
        }
    return {
        "captured_before_execution": True,
        "source": _file_identity(input_path, content_digest=False),
        "models": models,
    }


def assert_tta_artifacts_unchanged(identities: Mapping[str, Any]) -> None:
    """Reject a success manifest when an input changed after its initial snapshot."""

    records: list[Mapping[str, Any]] = [identities["source"]]
    for model in identities["models"].values():
        if model is not None:
            records.extend(model["artifacts"])
    compared_fields = (
        "size_bytes",
        "modified_time_ns",
        "change_or_creation_time_ns",
        "device",
        "file_id",
    )
    for record in records:
        current = _file_identity(
            record["path"],
            content_digest="sha256" in record,
        )
        changed = [
            field for field in compared_fields
            if int(current[field]) != int(record[field])
        ]
        if "sha256" in record and str(current["sha256"]) != str(record["sha256"]):
            changed.append("sha256")
        if changed:
            raise RuntimeError(
                "TTA input/model artifact changed during execution; refusing a complete "
                f"manifest: path={record['path']}, fields={changed}"
            )


def _view_record(view: Any) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field in _VIEW_FIELDS:
        value = getattr(view, field)
        if field == "physical_view_name" and not str(value):
            value = getattr(view, "name")
        if isinstance(value, tuple):
            value = list(value)
        record[field] = value
    return record


def _channel_record(channel_format: Any) -> dict[str, Any]:
    return {
        "token": str(channel_format.token),
        "kind": str(channel_format.kind),
        "channel_count": int(channel_format.channel_count),
        "stride": int(channel_format.stride),
        "offsets": [int(value) for value in channel_format.offsets],
        "boundary_policy": "radial_wrap_mirror_u_cartesian_edge_clamp",
        "prediction_assignment": "center_slice_only",
        "direction": "forward",
    }


def build_tta_run_manifest(
    *,
    launch_context: UnifiedLaunchContext,
    pipeline_version: str,
    resolved_config: Mapping[str, Any],
    artifact_identities: Mapping[str, Any],
    source_shape_tyx: Sequence[int],
    processing_shape_tyx: Sequence[int],
    fps: float,
    physical_views: Sequence[Any],
    inference_views: Sequence[Any],
    angles: Sequence[float],
    channel_format: Any,
    tile_configs: Sequence[Any],
    radial_requests: Sequence[Any],
    radial_diameters: Sequence[int],
    radial_azimuth_angles: Sequence[float],
    backend: Mapping[str, Any],
    forward_sampling: Mapping[str, Any],
    prediction_processing: Mapping[str, Any],
    requested_outputs: Sequence[str],
    output_paths: Mapping[str, str | Path],
    output_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete success manifest without importing numerical runtimes."""

    if str(launch_context.mode) != "tta":
        raise ValueError("a TTA run manifest requires a TTA unified-launch context")

    physical_records = [_view_record(view) for view in physical_views]
    inference_records = [_view_record(view) for view in inference_views]
    radial_groups: list[dict[str, Any]] = []
    if not (
        len(radial_requests)
        == len(radial_diameters)
        == len(radial_azimuth_angles)
    ):
        raise ValueError("resolved radial request metadata has inconsistent lengths")
    for request, diameter, spacing in zip(
        radial_requests, radial_diameters, radial_azimuth_angles
    ):
        token = str(request.view)
        concrete = [
            {
                "view_name": str(record["name"]),
                "azimuths_deg": list(record["azimuths_deg"]),
            }
            for record in physical_records
            if str(record["radial_request_token"]) == token
        ]
        radial_groups.append(
            {
                "view": token,
                "requested_azimuth_angle_deg": (
                    "auto"
                    if request.azimuth_angle is None
                    else float(request.azimuth_angle)
                ),
                "diameter": int(diameter),
                "resolved_azimuth_angle_deg": float(spacing),
                "concrete_azimuth_vectors": concrete,
            }
        )

    paths = {str(key): str(Path(value)) for key, value in output_paths.items()}

    return {
        "schema": TTA_RUN_MANIFEST_SCHEMA,
        "status": "complete",
        "launcher": {
            "name": str(launch_context.launcher),
            "version": str(launch_context.version),
            "mode": str(launch_context.mode),
            "command": list(launch_context.command),
            "pipeline_version": str(pipeline_version),
        },
        "determinism_contract": {
            "scope": "TTA mode retains the implementation-defined TTA determinism contract.",
            "identity_components": [
                "launcher and pipeline version",
                "complete command and resolved configuration",
                "source and model artifact identities",
                "resolved physical geometry and TTA variants",
                "selected inference and forward-sampling backends",
            ],
            "file_identity": (
                "pre-execution source stat snapshot with post-run mutation check; "
                "content SHA-256 plus stat snapshots for every consumed model artifact"
            ),
        },
        "inputs": {
            **dict(artifact_identities),
            "source_shape_t_y_x": [int(value) for value in source_shape_tyx],
            "processing_shape_t_y_x": [
                int(value) for value in processing_shape_tyx
            ],
            "fps": float(fps),
        },
        "resolved_configuration": dict(resolved_config),
        "geometry": {
            "physical_views": physical_records,
            "tta_angles_deg": [float(value) for value in angles],
            "inference_view_variants": inference_records,
            "radial_groups": radial_groups,
            "tiles": [
                {
                    "config_id": str(config.config_id),
                    "tile_size": int(config.tile_size),
                    "tile_stride": int(config.tile_stride),
                }
                for config in tile_configs
            ],
            "channel_format": _channel_record(channel_format),
        },
        "forward_sampling": {
            "policy_contract": "ForwardSamplingPolicy",
            "same_backend_builtin_geometry_shared_with_pta": True,
            "pta_forward_backends_qualified": ["cpu"],
            "prediction_interpolation_is_separate": True,
            **dict(forward_sampling),
        },
        "inference_backend": dict(backend),
        "prediction_processing": dict(prediction_processing),
        "outputs": {
            "requested": [str(value) for value in requested_outputs],
            "paths": paths,
            "artifacts": dict(output_metadata or {}),
        },
    }


__all__ = (
    "TTA_RUN_MANIFEST_SCHEMA",
    "TTA_VOXEL_COUNT_SCHEMA",
    "assert_tta_artifacts_unchanged",
    "build_tta_run_manifest",
    "capture_tta_artifact_identities",
)
