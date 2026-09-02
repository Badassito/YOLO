"""Resolved PTA configuration boundary for the execution engine.

The public parser produces :class:`PtaConfig`; this module converts that
immutable configuration into the execution namespace consumed by the dataset
engine. It deliberately owns internal scheduler defaults so the runtime no
longer parses or inherits options from an archived PTA launcher.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .pta_config import PtaConfig


def build_runtime_options(config: PtaConfig) -> argparse.Namespace:
    """Create the complete internal option set for one resolved PTA run."""

    public = config.args
    preprocessing = config.preprocessing
    save = config.save
    return argparse.Namespace(
        input=public.input,
        output=public.output,
        device=public.device,
        device_ids=config.device_ids,
        imgsz=public.imgsz,
        output_format=str(config.effective_output_format),
        channel_format=[str(config.channel_format.token)],
        force=public.force,
        gaussian_smoothing=(
            float(preprocessing.gaussian_sigma)
            if preprocessing.gaussian_smoothing_enabled
            else 0.0
        ),
        gaussian_smoothing_passes=(
            int(preprocessing.gaussian_passes)
            if preprocessing.gaussian_smoothing_enabled
            else 0
        ),
        save_images=save.enabled("images"),
        save_labels=save.enabled("labels"),
        save_nrrd=save.enabled("nrrd"),
        save_overlay=save.enabled("overlay"),
        voxel_volume=save.enabled("voxel_volume"),
        save_summary=save.enabled("summary"),
        train_split=public.train_split,
        split_method=public.split_method,
        background_percent=public.background_percent,
        augmentation=public.augmentation,
        augmentation_ratio=public.augmentation_ratio,
        augmentation_execution=public.augmentation_execution,
        offline_augmentation_backend=public.offline_augmentation_backend,
        gpu_batch_size=public.gpu_batch_size,
        workers=public.workers,
        frame_workers=public.frame_workers,
        png_compression=public.png_compression,
        overlay_tile_writer_limit=public.overlay_tile_writer_limit,
        overlay_workers=public.overlay_workers,
        overlay_pending_frames=public.overlay_pending_frames,
        worker_backend=public.worker_backend,
        pipeline_depth=public.pipeline_depth,
        jpeg_decode_backend=public.jpeg_decode_backend,
        jpeg_batch_size=public.jpeg_batch_size,
        jpeg_encode_backend=public.jpeg_encode_backend,
        tiff_encode_backend=public.tiff_encode_backend,
        jpeg_quality=public.jpeg_quality,
        topology_aware=public.topology_aware,
        # Resolved unified geometry is authoritative. These neutral values satisfy
        # the engine's internal shape checks and are never CLI aliases.
        enable_sagittal=False,
        enable_coronal=False,
        enable_radial=False,
        azimuth_angle=None,
        tilt_angle=["0"],
        tilt_direction=["vertical"],
        tile_size=(
            [str(tile.tile_size) for tile in config.tiles]
            if config.tiles
            else ["0"]
        ),
        tile_stride=(
            [str(tile.tile_stride) for tile in config.tiles]
            if config.tiles
            else None
        ),
        # Internal scheduler settings are not public flags.
        max_pending_frames=0,
        tile_task_chunk=1,
        aug_task_chunk=4,
        resume=False,
        _v18_requested_output_format=str(config.requested_output_format),
        _v18_config=config,
    )


def run(config: PtaConfig, *, argv: Sequence[str] | None = None) -> None:
    """Enter the PTA engine with an already validated configuration."""

    from . import pta

    arguments = None if argv is None else [str(value) for value in argv]
    pta.main(args=build_runtime_options(config), argv=arguments)


__all__ = ["build_runtime_options", "run"]
