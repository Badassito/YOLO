"""Dependency-light configuration for the unified v18 PTA mode.

Geometry token validation reuses the current TTA grammar, while mode-specific
defaults and output semantics remain here.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .config import (
    ChannelFormat,
    RadialViewRequest,
    TiltedViewGroup,
    resolve_cartesian_views,
    resolve_channel_format,
    resolve_radial_view_requests,
    resolve_tilted_view_groups,
)
from .unification.tiles import ResolvedTileGroup, resolve_tile_groups


PTA_SAVE_OPTION_TOKENS: Tuple[str, ...] = (
    "images",
    "labels",
    "nrrd",
    "overlay",
    "voxel_volume",
    "summary",
)


@dataclass(frozen=True)
class PtaSaveRequest:
    """Resolved PTA publication and diagnostic outputs."""

    tokens: Tuple[str, ...] = ()

    def enabled(self, token: str) -> bool:
        return str(token) in self.tokens


@dataclass(frozen=True)
class PreprocessingRequest:
    """Resolved PTA operations that run before physical view generation."""

    gaussian_smoothing_enabled: bool = False
    gaussian_sigma: float = 3.0
    gaussian_passes: int = 1


TileRequest = ResolvedTileGroup


@dataclass(frozen=True)
class PtaConfig:
    """Strict, fully resolved v18 PTA configuration."""

    args: argparse.Namespace
    channel_format: ChannelFormat
    preprocessing: PreprocessingRequest
    save: PtaSaveRequest
    cartesian_views: Tuple[str, ...]
    radial_requests: Tuple[RadialViewRequest, ...]
    tilted_groups: Tuple[TiltedViewGroup, ...]
    tiles: Tuple[TileRequest, ...]
    requested_output_format: str
    effective_output_format: str

    @property
    def has_physical_views(self) -> bool:
        if self.cartesian_views or self.tilted_groups:
            return True
        tilted_bases = {
            str(view)
            for group in self.tilted_groups
            for view in group.views
        }
        return any(
            not str(request.view).startswith("tilted_")
            or str(request.view)[len("tilted_") :] in tilted_bases
            for request in self.radial_requests
        )


def parse_output_image_format(value: str) -> str:
    """Canonicalize output-family aliases accepted by PTA."""

    token = str(value).strip().lower().lstrip(".")
    aliases = {
        "png": "png",
        "jpg": "jpg",
        "jpeg": "jpg",
        "tif": "tif",
        "tiff": "tif",
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "--output_format must be one of png, jpg/jpeg, or tif/tiff"
        ) from exc


class _SingleOccurrenceAction(argparse.Action):
    """Reject repeated scalar flags instead of silently accepting the last one."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:  # type: ignore[no-untyped-def]
        if getattr(self, "_seen", False):
            raise argparse.ArgumentError(
                self,
                f"{option_string or self.option_strings[0]} may be provided only once",
            )
        self._seen = True
        setattr(namespace, self.dest, values)


def _token_list(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else list(values)
    tokens: list[str] = []
    for raw in raw_values:
        tokens.extend(part for part in re.split(r"[,\s]+", str(raw).strip()) if part)
    return tokens


def resolve_pta_save_request(values: Sequence[str] | str | None) -> PtaSaveRequest:
    tokens: list[str] = []
    valid = set(PTA_SAVE_OPTION_TOKENS)
    for raw in _token_list(values):
        token = str(raw).lower()
        if token not in valid:
            expected = ", ".join(PTA_SAVE_OPTION_TOKENS)
            raise ValueError(
                f"--save values for PTA must be one or more of: {expected}; got {raw!r}"
            )
        if token in tokens:
            raise ValueError(f"--save contains duplicate PTA output {token!r}")
        tokens.append(token)
    return PtaSaveRequest(tokens=tuple(tokens))


def resolve_preprocessing_options(
    values: Sequence[str] | str | None,
) -> PreprocessingRequest:
    """Resolve the sole v18 PTA preprocessing operation.

    Absence disables smoothing. Once selected, omitted slots use the current
    TTA defaults of sigma 3 and one pass.
    """

    tokens = _token_list(values)
    if not tokens:
        return PreprocessingRequest()
    if len(tokens) != 1:
        raise ValueError("--preprocessing accepts gaussian_smoothing at most once")

    raw = tokens[0]
    slots = [part.strip() for part in str(raw).split(":")]
    if slots[0].lower() != "gaussian_smoothing":
        raise ValueError(
            "--preprocessing accepts only "
            "gaussian_smoothing[:STANDARD_DEVIATION][:SMOOTHING_PASSES]"
        )
    if len(slots) > 3:
        raise ValueError(
            f"--preprocessing {raw!r} must use "
            "gaussian_smoothing[:STANDARD_DEVIATION][:SMOOTHING_PASSES]"
        )
    try:
        sigma = float(slots[1]) if len(slots) >= 2 and slots[1] else 3.0
        passes = int(slots[2]) if len(slots) >= 3 and slots[2] else 1
    except Exception as exc:
        raise ValueError(f"--preprocessing {raw!r} has invalid Gaussian parameters") from exc
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            "--preprocessing gaussian_smoothing requires STANDARD_DEVIATION > 0"
        )
    if passes < 1:
        raise ValueError(
            "--preprocessing gaussian_smoothing requires SMOOTHING_PASSES >= 1"
        )
    return PreprocessingRequest(
        gaussian_smoothing_enabled=True,
        gaussian_sigma=float(sigma),
        gaussian_passes=int(passes),
    )


def resolve_tile_requests(values: Sequence[str] | str | None) -> Tuple[TileRequest, ...]:
    """Resolve the exact TTA TILE_SIZE:TILE_STRIDE grammar."""
    return tuple(resolve_tile_groups(values))


def build_pta_argparser(*, prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="v18 unified pretraining augmentation and dataset generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True, help="PTA input directory")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument(
        "--imgsz",
        default=0,
        type=int,
        help="Square emitted raster size; 0 preserves the native view size",
    )
    parser.add_argument(
        "--output_format",
        default="png",
        type=parse_output_image_format,
        metavar="{png,jpg,jpeg,tif,tiff}",
    )
    parser.add_argument(
        "--channel_format",
        default="gray",
        action=_SingleOccurrenceAction,
        metavar="{gray,RGB,CxSy}",
        help=(
            "Exactly one channel layout. PTA emits ascending and reversed custom "
            "channel orders when those layouts are distinct"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply PTA full-volume eligibility rules to partial inputs",
    )
    parser.add_argument(
        "--preprocessing",
        nargs="+",
        default=None,
        metavar="OPERATION[:PARAMETERS]",
        help=(
            "Pre-geometry operations. Currently accepts "
            "gaussian_smoothing[:STANDARD_DEVIATION][:SMOOTHING_PASSES]"
        ),
    )
    parser.add_argument(
        "--save",
        nargs="+",
        default=None,
        metavar="OUTPUT",
        help=(
            "PTA outputs: images, labels, nrrd, overlay, voxel_volume, summary. "
            "The normal dataset publication is --save images labels"
        ),
    )
    parser.add_argument("--train_split", default=None, type=float)
    parser.add_argument(
        "--split_method",
        default=None,
        choices=("volume", "view", "slice"),
    )
    parser.add_argument("--background_percent", default=1.0, type=float)

    parser.add_argument("--augmentation", default=None)
    parser.add_argument("--augmentation_ratio", default=1.0, type=float)
    parser.add_argument(
        "--augmentation_execution",
        default="deferred",
        choices=("deferred", "offline"),
    )
    parser.add_argument(
        "--offline_augmentation_backend",
        default="auto",
        choices=("auto", "cpu", "gpu"),
    )
    parser.add_argument(
        "--gpu_batch_size",
        default=32,
        type=int,
        help="Maximum offline GPU-policy candidate batch; runtime may lower it to fit free VRAM",
    )

    parser.add_argument(
        "--enable_cartesian",
        nargs="+",
        default=None,
        metavar="VIEW",
    )
    parser.add_argument(
        "--enable_radial",
        nargs="+",
        default=None,
        metavar="VIEWS[:AZIMUTH_ANGLE]",
    )
    parser.add_argument(
        "--enable_tilted",
        nargs="+",
        default=None,
        metavar="VIEW[:TILT_ANGLE[:TILT_DIRECTION]]",
    )
    parser.add_argument(
        "--enable_tile",
        nargs="+",
        default=None,
        metavar="TILE_SIZE:TILE_STRIDE",
    )

    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument(
        "--frame_workers",
        default=0,
        type=int,
        help=(
            "CPU render-worker budget; with offline GPU augmentation, one CUDA-owner process "
            "is used per visible device and this value controls its bounded CPU preparation threads"
        ),
    )
    parser.add_argument("--png_compression", default=1, type=int)
    parser.add_argument("--overlay_tile_writer_limit", default=64, type=int)
    parser.add_argument("--overlay_workers", default=0, type=int)
    parser.add_argument("--overlay_pending_frames", default=0, type=int)
    parser.add_argument(
        "--worker_backend",
        default="auto",
        choices=("auto", "process", "thread"),
    )
    parser.add_argument(
        "--pipeline_depth",
        default=2,
        type=int,
        choices=(1, 2),
    )
    parser.add_argument(
        "--jpeg_decode_backend",
        default="auto",
        choices=("auto", "nvjpeg", "opencv"),
    )
    parser.add_argument("--jpeg_batch_size", default=64, type=int)
    parser.add_argument(
        "--jpeg_encode_backend",
        default="auto",
        choices=("auto", "nvjpeg", "opencv"),
    )
    parser.add_argument("--jpeg_quality", default=95, type=int)
    parser.add_argument(
        "--topology_aware",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def resolve_pta_config(args: argparse.Namespace) -> PtaConfig:
    if int(args.imgsz) < 0:
        raise ValueError("--imgsz must be >= 0 in PTA mode")
    if not (0.0 <= float(args.background_percent) <= 1.0):
        raise ValueError("--background_percent must be in [0,1]")
    if args.train_split is not None and not (0.0 <= float(args.train_split) <= 1.0):
        raise ValueError("--train_split must be in [0,1]")
    if not math.isfinite(float(args.augmentation_ratio)) or float(args.augmentation_ratio) < 1.0:
        raise ValueError("--augmentation_ratio must be finite and >= 1")
    if float(args.augmentation_ratio) > 1.0 and not args.augmentation:
        raise ValueError("--augmentation_ratio > 1 requires --augmentation")
    if int(args.gpu_batch_size) <= 0:
        raise ValueError("--gpu_batch_size must be > 0")
    if int(args.workers) < 0 or int(args.frame_workers) < 0:
        raise ValueError("--workers and --frame_workers must be >= 0")
    if not 1 <= int(args.jpeg_quality) <= 100:
        raise ValueError("--jpeg_quality must be between 1 and 100")
    if int(args.jpeg_batch_size) <= 0:
        raise ValueError("--jpeg_batch_size must be > 0")
    if int(args.overlay_tile_writer_limit) <= 0:
        raise ValueError("--overlay_tile_writer_limit must be > 0")
    if int(args.overlay_workers) < 0 or int(args.overlay_pending_frames) < 0:
        raise ValueError("overlay worker settings must be >= 0")

    channel_format = resolve_channel_format(args.channel_format)
    requested_output_format = parse_output_image_format(args.output_format)
    effective_output_format = (
        "tif" if str(channel_format.kind) == "custom" else requested_output_format
    )
    if (
        effective_output_format == "png"
        and not 0 <= int(args.png_compression) <= 9
    ):
        raise ValueError("--png_compression must be between 0 and 9")
    return PtaConfig(
        args=args,
        channel_format=channel_format,
        preprocessing=resolve_preprocessing_options(args.preprocessing),
        save=resolve_pta_save_request(args.save),
        cartesian_views=tuple(resolve_cartesian_views(args.enable_cartesian)),
        radial_requests=tuple(resolve_radial_view_requests(args.enable_radial)),
        tilted_groups=tuple(resolve_tilted_view_groups(args.enable_tilted)),
        tiles=resolve_tile_requests(args.enable_tile),
        requested_output_format=str(requested_output_format),
        effective_output_format=str(effective_output_format),
    )


def parse_pta_args(
    argv: Optional[Sequence[str]] = None,
    *,
    prog: Optional[str] = None,
) -> PtaConfig:
    parser = build_pta_argparser(prog=prog)
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        return resolve_pta_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.error must terminate") from exc


__all__ = (
    "PTA_SAVE_OPTION_TOKENS",
    "PreprocessingRequest",
    "PtaConfig",
    "PtaSaveRequest",
    "TileRequest",
    "build_pta_argparser",
    "parse_output_image_format",
    "parse_pta_args",
    "resolve_preprocessing_options",
    "resolve_pta_config",
    "resolve_pta_save_request",
    "resolve_tile_requests",
)
