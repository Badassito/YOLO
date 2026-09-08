"""Resolved TTA run-manifest construction for the unified launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .context import UnifiedLaunchContext


TTA_RUN_MANIFEST_SCHEMA = "xta.v21.run_manifest/1"
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
    "azimuthal_base_view",
    "azimuthal_tilted_source",
    "azimuthal_source_view_name",
    "azimuthal_request_token",
    "tta_aug_id",
    "tta_angle_deg",
)

_RADIAL_VIEW_FIELDS = (
    "radial_base_view", "radial_tilted_source", "radial_source_view_name",
    "radial_request_token", "radial_min_radius", "radial_max_radius", "radial_step",
    "radial_shell_start", "radial_radii", "radial_arc_origin", "radial_height_origin",
    "radial_patch_size", "radial_patch_index", "radial_height_index",
)


def radial_view_manifest_record(view: Any) -> dict[str, Any] | None:
    """Record the complete native shell trajectory without numerical imports."""
    if str(getattr(view, "family", "")) != "radial":
        return None
    record = {}
    for field in _RADIAL_VIEW_FIELDS:
        value = getattr(view, field)
        record[field] = list(value) if isinstance(value, tuple) else value
    return {
        **record,
        "source_shape_t_y_x": [int(view.full_t), int(view.full_h), int(view.full_w)],
        "center_x_y": [float(view.center_x), float(view.center_y)],
        "native_patch_shape_h_w": [int(view.src_h), int(view.src_w)],
        "tilt_angle_deg": float(view.tilt_angle_deg),
        "tilt_direction": str(view.tilt_direction),
        "slice_direction": "radius",
        "input_axes": ["azimuth_arc_length", "height"],
        "source_voxel_spacing": {"arc_length": 1.0, "height": 1.0},
        "angular_boundary": "periodic_0_360_degrees_without_reflection",
        "height_boundary": "unsheared_native_height_padding_zero",
        "source_boundary": "zero_extended_trilinear_intensity; nearest_categorical_outside_zero",
        "channel_boundary": "clamp_radius_within_patch_trajectory",
        "patch_frame_mapping": "radial_radii[frame_index] at fixed arc and height origins",
        "patch_is_tile": False,
        "tile_parent": "native_periodic_patch",
        "coverage_domain": "minimum_radius_to_largest_inscribed_cylinder; central_core_excluded",
    }


def radial_view_plan_metadata(view: Any) -> dict[str, Any]:
    """Bind shell geometry into plan identity while retaining other-family metadata."""
    record = radial_view_manifest_record(view)
    return {} if record is None else {"radial_shell": record}


_SPHERICAL_VIEW_FIELDS = (
    "spherical_face", "spherical_group", "spherical_request_tokens",
    "spherical_tilted_source", "spherical_min_radius", "spherical_max_radius",
    "spherical_step", "spherical_radii", "spherical_face_intervals",
    "spherical_patch_size", "spherical_u_origin", "spherical_v_origin",
    "spherical_patch_u", "spherical_patch_v", "spherical_rotation_xyz",
)


def spherical_view_manifest_record(view: Any) -> dict[str, Any] | None:
    """Record one QSC face-patch radius trajectory and its canonical cube."""
    if str(getattr(view, "family", "")) != "spherical":
        return None
    record: dict[str, Any] = {}
    for field in _SPHERICAL_VIEW_FIELDS:
        value = getattr(view, field)
        record[field] = list(value) if isinstance(value, tuple) else value
    return {
        **record,
        "source_shape_t_y_x": [int(view.full_t), int(view.full_h), int(view.full_w)],
        "center_x_y_t": [
            (int(view.full_w) - 1) / 2.0,
            (int(view.full_h) - 1) / 2.0,
            (int(view.full_t) - 1) / 2.0,
        ],
        "native_patch_shape_h_w": [int(view.src_h), int(view.src_w)],
        "tilt_angle_deg": float(view.tilt_angle_deg),
        "tilt_direction": str(view.tilt_direction),
        "projection": "quadrilateralized_spherical_cube_equal_area",
        "slice_direction": "radius",
        "input_axes": ["qsc_face_u", "qsc_face_v"],
        "face_grid": "fixed_outer_radius_endpoint_inclusive_n_plus_one_samples",
        "face_boundary": "inclusive_shared_face_edges; outside_face_patch_padding_zero",
        "source_boundary": "zero_extended_trilinear_intensity; nearest_categorical_outside_zero",
        "channel_boundary": "clamp_radius_within_face_patch_trajectory",
        "patch_frame_mapping": "spherical_radii[frame_index] at fixed QSC face and patch origins",
        "rotation_convention": "row_major_xyz_matrix_rotates_cube_directions_about_source_center",
        "patch_is_tile": False,
        "tile_parent": "native_qsc_face_patch",
        "coverage_domain": "minimum_radius_to_largest_inscribed_sphere; central_core_excluded",
    }


def spherical_view_plan_metadata(view: Any) -> dict[str, Any]:
    """Bind complete QSC geometry and cube rotation into raster-plan identity."""
    record = spherical_view_manifest_record(view)
    return {} if record is None else {"spherical_shell": record}


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
    radial = radial_view_manifest_record(view)
    if radial is not None:
        record.update(radial)
    spherical = spherical_view_manifest_record(view)
    if spherical is not None:
        record.update(spherical)
    return record


def _channel_record(channel_format: Any) -> dict[str, Any]:
    return {
        "token": str(channel_format.token),
        "kind": str(channel_format.kind),
        "channel_count": int(channel_format.channel_count),
        "stride": int(channel_format.stride),
        "offsets": [int(value) for value in channel_format.offsets],
        "boundary_policy": "azimuthal_wrap_mirror_u_radial_radius_clamp_spherical_radius_clamp_cartesian_edge_clamp",
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
    azimuthal_requests: Sequence[Any],
    azimuthal_diameters: Sequence[int],
    azimuthal_azimuth_angles: Sequence[float],
    backend: Mapping[str, Any],
    forward_sampling: Mapping[str, Any],
    prediction_processing: Mapping[str, Any],
    requested_outputs: Sequence[str],
    output_paths: Mapping[str, str | Path],
    output_metadata: Mapping[str, Any] | None = None,
    radial_requests: Sequence[Any] = (),
    spherical_requests: Sequence[Any] = (),
) -> dict[str, Any]:
    """Build the complete success manifest without importing numerical runtimes."""

    if str(launch_context.mode) != "tta":
        raise ValueError("a TTA run manifest requires a TTA unified-launch context")

    physical_records = [_view_record(view) for view in physical_views]
    inference_records = [_view_record(view) for view in inference_views]
    azimuthal_groups: list[dict[str, Any]] = []
    if not (
        len(azimuthal_requests)
        == len(azimuthal_diameters)
        == len(azimuthal_azimuth_angles)
    ):
        raise ValueError("resolved azimuthal request metadata has inconsistent lengths")
    for request, diameter, spacing in zip(
        azimuthal_requests, azimuthal_diameters, azimuthal_azimuth_angles
    ):
        token = str(request.view)
        concrete = [
            {
                "view_name": str(record["name"]),
                "azimuths_deg": list(record["azimuths_deg"]),
            }
            for record in physical_records
            if str(record["azimuthal_request_token"]) == token
        ]
        azimuthal_groups.append(
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

    radial_tokens = list(dict.fromkeys(
        [str(request.view) for request in radial_requests]
        + [str(record["radial_request_token"]) for record in physical_records
           if str(record["family"]) == "radial"]
    ))
    radial_groups = [
        {
            "view": token,
            "concrete_patch_trajectories": [
                record for record in physical_records
                if str(record["family"]) == "radial"
                and str(record["radial_request_token"]) == token
            ],
        }
        for token in radial_tokens
    ]

    spherical_records = [
        record for record in physical_records if str(record["family"]) == "spherical"
    ]
    spherical_group_ids = list(dict.fromkeys(
        str(record["spherical_group"]) for record in spherical_records
    ))
    spherical_groups = []
    for group_id in spherical_group_ids:
        trajectories = [
            record for record in spherical_records
            if str(record["spherical_group"]) == group_id
        ]
        spherical_groups.append({
            "group": group_id,
            "requested_views": list(dict.fromkeys(
                str(token) for record in trajectories
                for token in record["spherical_request_tokens"]
            )),
            "concrete_patch_trajectories": trajectories,
        })
    spherical_tokens = list(dict.fromkeys(
        [str(request.view) for request in spherical_requests]
        + [str(token) for record in spherical_records
           for token in record["spherical_request_tokens"]]
    ))
    spherical_request_records = [
        {
            "view": token,
            "canonical_groups": [
                group["group"] for group in spherical_groups
                if token in group["requested_views"]
            ],
        }
        for token in spherical_tokens
    ]

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
            "azimuthal_groups": azimuthal_groups,
            "radial_groups": radial_groups,
            "spherical_groups": spherical_groups,
            "spherical_requests": spherical_request_records,
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
    "radial_view_manifest_record",
    "radial_view_plan_metadata",
    "spherical_view_manifest_record",
    "spherical_view_plan_metadata",
)
