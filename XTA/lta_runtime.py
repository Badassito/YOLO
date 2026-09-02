"""Planning and execution boundary for the v19 LTA prototype.

The GPU-independent planner is complete and intentionally testable with fake
geometry compilers.  The public :func:`run` currently stops after preflight and
planning rather than publishing a false ``complete`` manifest before the SAM
execution loop is connected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple
import uuid

from .lta_config import LtaConfig
from .lta_inputs import (
    ImageMetadata,
    LtaInputDiscovery,
    VideoProbe,
    discover_lta_inputs,
)
from .lta_sam import (
    LTA_CHANNEL_POLICY,
    LTA_MAX_NUM_OBJECTS,
    LTA_MULTIPLEX_COUNT,
    LocalSamBundle,
    SamSessionPlan,
    plan_sam_sessions,
    resolve_local_sam_bundle,
)


class LtaPrototypeExecutionPending(RuntimeError):
    """Raised after a valid plan when GPU inference is not connected yet."""


@dataclass(frozen=True)
class LtaRuntimeViewPlan:
    volume_id: str
    physical_view_id: str
    runtime_view_id: str
    tta_angle_deg: float
    frame_count: int
    sessions: Tuple[SamSessionPlan, ...]
    tile_config_ids: Tuple[str, ...] = ()
    encoded_frame_indices: Tuple[Optional[int], ...] = ()
    raster_plan_digest: Optional[str] = None

    def manifest_record(self) -> dict[str, object]:
        return {
            "volume_id": self.volume_id,
            "physical_view_id": self.physical_view_id,
            "runtime_view_id": self.runtime_view_id,
            "tta_angle_deg": float(self.tta_angle_deg),
            "frame_count": int(self.frame_count),
            "sessions": [
                {
                    "session_index": int(session.session_index),
                    "frame_start": int(session.frame_start),
                    "frame_stop": int(session.frame_stop),
                }
                for session in self.sessions
            ],
            "tile_config_ids": list(self.tile_config_ids),
            "encoded_frame_indices": [
                None if value is None else int(value)
                for value in self.encoded_frame_indices
            ],
            "raster_plan_digest": self.raster_plan_digest,
        }


@dataclass(frozen=True)
class LtaVolumePlan:
    volume_id: str
    stem: str
    source_shape_tyx: Tuple[int, int, int]
    runtime_views: Tuple[LtaRuntimeViewPlan, ...]

    def manifest_record(self) -> dict[str, object]:
        return {
            "volume_id": self.volume_id,
            "stem": self.stem,
            "source_shape_tyx": [int(value) for value in self.source_shape_tyx],
            "runtime_views": [view.manifest_record() for view in self.runtime_views],
        }


@dataclass(frozen=True)
class LtaRunPlan:
    run_id: str
    bundle: LocalSamBundle
    discovery: LtaInputDiscovery
    output_root: Path
    temp_root: Path
    device_ids: Tuple[int, ...]
    sam_execution: str
    conf: float
    save_tokens: Tuple[str, ...]
    command: Tuple[str, ...]
    postprocessing: Mapping[str, object]
    volumes: Tuple[LtaVolumePlan, ...]

    def manifest_record(self) -> dict[str, object]:
        input_records = []
        for volume in self.discovery.all_volumes:
            input_records.append(
                {
                    "volume_id": volume.volume_id,
                    "source_role": volume.source_role.value,
                    "volume_class": volume.volume_class.value,
                    "video": (
                        None
                        if volume.video_path is None
                        else {
                            "path": str(volume.video_path),
                            "identity_sha256": volume.video_identity_sha256,
                        }
                    ),
                    "images": [
                        {
                            "encoded_index": int(media.encoded_index),
                            "path": str(media.path),
                            "identity_sha256": media.identity_sha256,
                            "content_sha256": media.sha256,
                        }
                        for media in volume.media
                    ],
                    "labels": [
                        {
                            "encoded_index": int(annotation.encoded_index),
                            "path": (
                                None
                                if annotation.label_path is None
                                else str(annotation.label_path)
                            ),
                            "sha256": annotation.label_sha256,
                            "state": annotation.state.value,
                        }
                        for annotation in volume.annotations
                    ],
                }
            )
        return {
            "mode": "lta",
            "run_id": self.run_id,
            "prototype_stage": "preflight_and_geometry_plan",
            "model": {
                "bundle_root": str(self.bundle.root),
                "checkpoint_path": str(self.bundle.checkpoint_path),
                "model_version": self.bundle.model_version,
                "checkpoint_identity_sha256": self.bundle.checkpoint_identity_sha256,
                "bpe_path": None if self.bundle.bpe_path is None else str(self.bundle.bpe_path),
            },
            "input": str(self.discovery.input_path),
            "exemplar_roots": [str(path) for path in self.discovery.exemplar_roots],
            "output_root": str(self.output_root),
            "temp_root": str(self.temp_root),
            "device_ids": [int(value) for value in self.device_ids],
            "sam_execution": self.sam_execution,
            "conf": float(self.conf),
            "channel_policy": LTA_CHANNEL_POLICY,
            "object_multiplex": {
                "max_num_objects": int(LTA_MAX_NUM_OBJECTS),
                "multiplex_count": int(LTA_MULTIPLEX_COUNT),
            },
            "command": list(self.command),
            "postprocessing": dict(self.postprocessing),
            "save": list(self.save_tokens),
            "positive_exemplar_count": len(self.discovery.positive_pool),
            "input_identities": input_records,
            "selected_positive_exemplars": [
                {
                    "exemplar_id": exemplar.exemplar_id,
                    "volume_id": exemplar.volume_id,
                    "media_path": str(exemplar.media_path),
                    "media_sha256": exemplar.media_sha256,
                    "media_identity_sha256": exemplar.media_identity_sha256,
                    "label_sha256": exemplar.label_sha256,
                }
                for exemplar in self.discovery.positive_pool
            ],
            "warnings": [
                {
                    "code": warning.code,
                    "message": warning.message,
                    "volume_id": warning.volume_id,
                }
                for warning in self.discovery.warnings
            ],
            "volumes": [volume.manifest_record() for volume in self.volumes],
        }


def probe_image_with_pillow(path: Path) -> ImageMetadata:
    """Read image dimensions without decoding the complete pixel payload."""

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - project dependency on production hosts
        raise RuntimeError("Pillow is required to inspect LTA image-stack dimensions") from exc
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise RuntimeError(f"Could not inspect LTA image {path}: {exc}") from exc
    return ImageMetadata(width=int(width), height=int(height))


def _one_frame_sessions(sequence_id: str, frame_count: int) -> Tuple[SamSessionPlan, ...]:
    return tuple(
        SamSessionPlan(
            sequence_id=sequence_id,
            session_index=index,
            frame_start=index,
            frame_stop=index + 1,
        )
        for index in range(int(frame_count))
    )


def _tile_config_ids(config: LtaConfig) -> Tuple[str, ...]:
    values = []
    for tile in config.tiles:
        config_id = getattr(tile, "config_id", None)
        values.append(
            str(config_id)
            if config_id is not None
            else f"s{int(tile.tile_size)}_st{int(tile.tile_stride)}"
        )
    return tuple(values)


def build_lta_run_plan(
    config: LtaConfig,
    *,
    video_probe: Optional[VideoProbe] = None,
    image_probe: Optional[Callable[[Path], ImageMetadata]] = None,
    discovery_fn: Callable[..., LtaInputDiscovery] = discover_lta_inputs,
    physical_compiler: Optional[Callable[..., object]] = None,
    variant_expander: Optional[Callable[..., Sequence[object]]] = None,
    run_id: Optional[str] = None,
    argv: Sequence[str] = (),
) -> LtaRunPlan:
    """Resolve local assets, labeled inputs, physical views, and SAM sessions."""

    if not isinstance(config, LtaConfig):
        raise TypeError("config must be an LtaConfig")
    bundle = resolve_local_sam_bundle(config.args.model)
    discovery = discovery_fn(
        config.args.input,
        config.exemplar_dirs,
        video_probe=video_probe,
        image_probe=image_probe or probe_image_with_pillow,
        require_positive=True,
    )

    if physical_compiler is None:
        from .unification.runtime import compile_physical_views

        physical_compiler = compile_physical_views
    if variant_expander is None:
        from .unification.contracts import PipelineMode
        from .unification.views import expand_view_variants

        variant_expander = lambda physical_views, angles: expand_view_variants(  # noqa: E731
            PipelineMode.LTA,
            physical_views,
            angles,
        )

    tiles = _tile_config_ids(config)
    resolved_run_id = str(run_id or uuid.uuid4().hex).strip()
    if not resolved_run_id:
        raise ValueError("run_id must not be empty")
    volume_plans = []
    from .unification.sampling import build_forward_raster_plan
    for volume in discovery.target_volumes:
        if volume.width is None or volume.height is None:
            raise RuntimeError(
                f"LTA planning is missing dimensions for target volume {volume.stem}"
            )
        shape = (int(volume.frame_count), int(volume.height), int(volume.width))
        compiled = physical_compiler(
            t_dim=shape[0],
            height=shape[1],
            width=shape[2],
            cartesian_views=config.cartesian_views,
            radial_requests=config.radial_requests,
            tilted_groups=config.tilted_groups,
            radial_native_raster=1008,
        )
        physical_views = tuple(getattr(compiled, "views"))
        variants = tuple(variant_expander(physical_views, config.angles))
        runtime_plans = []
        for variant in variants:
            physical = getattr(variant, "physical_view")
            runtime = getattr(variant, "runtime_view")
            angle = float(getattr(getattr(variant, "in_plane_variant"), "angle_deg"))
            frame_count = int(getattr(runtime, "num_slices"))
            if frame_count < 1:
                raise RuntimeError(
                    f"LTA runtime view {getattr(runtime, 'name', '<unknown>')} has no frames"
                )
            physical_name = str(getattr(physical, "name"))
            runtime_name = str(getattr(runtime, "name"))
            sequence_id = f"{volume.volume_id}::{runtime_name}"
            sessions = (
                plan_sam_sessions(sequence_id, frame_count)
                if str(config.args.sam_execution) == "video"
                else _one_frame_sessions(sequence_id, frame_count)
            )
            encoded_frame_indices: Tuple[Optional[int], ...] = (
                tuple(int(value) for value in volume.encoded_indices)
                if physical_name == "transverse"
                and len(volume.encoded_indices) == frame_count
                else tuple(None for _ in range(frame_count))
            )
            raster_plan = build_forward_raster_plan(
                mode="lta",
                physical_view_id=physical_name,
                angle_deg=angle,
                channel_token="RGB",
                channel_kind="rgb",
                channel_count=3,
                channel_stride=1,
                channel_offsets=(0, 0, 0),
                channel_direction="ascending",
                output_shape=(1008, 1008),
                metadata={
                    "runtime_view_id": runtime_name,
                    "source_shape_tyx": list(shape),
                    "channel_policy": LTA_CHANNEL_POLICY,
                    "runtime_kind": "fullframe_sam",
                },
            )
            runtime_plans.append(
                LtaRuntimeViewPlan(
                    volume_id=volume.volume_id,
                    physical_view_id=physical_name,
                    runtime_view_id=runtime_name,
                    tta_angle_deg=angle,
                    frame_count=frame_count,
                    sessions=sessions,
                    tile_config_ids=tiles,
                    encoded_frame_indices=encoded_frame_indices,
                    raster_plan_digest=str(raster_plan.digest),
                )
            )
        volume_plans.append(
            LtaVolumePlan(
                volume_id=volume.volume_id,
                stem=volume.stem,
                source_shape_tyx=shape,
                runtime_views=tuple(runtime_plans),
            )
        )

    output_root = Path(config.args.output).expanduser().resolve(strict=False)
    raw_temp = config.args.temp
    temp_base = (
        Path(raw_temp).expanduser().resolve(strict=False)
        if raw_temp
        else output_root / "temp"
    )
    temp_root = temp_base / f"lta_{resolved_run_id}"
    return LtaRunPlan(
        run_id=resolved_run_id,
        bundle=bundle,
        discovery=discovery,
        output_root=output_root,
        temp_root=temp_root,
        device_ids=tuple(int(value) for value in config.device_ids),
        sam_execution=str(config.args.sam_execution),
        conf=float(config.args.conf),
        save_tokens=tuple(config.save.tokens),
        command=tuple(str(value) for value in argv),
        postprocessing={
            "keep_objects": int(config.postprocessing.keep_objects),
            "enable_3d_void_fill": bool(config.postprocessing.enable_3d_void_fill),
            "gaussian_smoothing_enabled": bool(
                config.postprocessing.gaussian_smoothing_enabled
            ),
            "gaussian_sigma": float(config.postprocessing.gaussian_sigma),
            "gaussian_passes": int(config.postprocessing.gaussian_passes),
        },
        volumes=tuple(volume_plans),
    )


def run(config: LtaConfig, *, argv: Sequence[str] | None = None) -> LtaRunPlan:
    """Run the GPU-independent v19 prototype preflight and geometry planner.

    The returned plan is useful to tests and the upcoming SAM execution loop.
    A clear exception prevents users from mistaking successful planning for
    generated labels or a complete publication.
    """

    plan = build_lta_run_plan(config, argv=tuple(argv or ()))
    runtime_view_count = sum(len(volume.runtime_views) for volume in plan.volumes)
    session_count = sum(
        len(view.sessions)
        for volume in plan.volumes
        for view in volume.runtime_views
    )
    print(
        "v19 LTA prototype preflight complete: "
        f"volumes={len(plan.volumes)}, runtime_views={runtime_view_count}, "
        f"sessions={session_count}, positive_exemplars={len(plan.discovery.positive_pool)}, "
        f"devices={list(plan.device_ids)}"
    )
    raise LtaPrototypeExecutionPending(
        "v19 LTA prototype planning succeeded; SAM render/inference/backprojection "
        "execution is not connected yet, so no outputs or complete manifest were written"
    )


__all__ = (
    "LtaPrototypeExecutionPending",
    "LtaRunPlan",
    "LtaRuntimeViewPlan",
    "LtaVolumePlan",
    "build_lta_run_plan",
    "probe_image_with_pillow",
    "run",
)
