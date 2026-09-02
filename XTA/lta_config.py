"""Dependency-light configuration for the v19 label-time augmentation mode."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .config import (
    PostprocessingRequest,
    RadialViewRequest,
    TiltedViewGroup,
    resolve_cartesian_views,
    resolve_postprocessing_options,
    resolve_radial_view_requests,
    resolve_tilted_view_groups,
    resolve_tta_angles,
)
from .unification.tiles import ResolvedTileGroup, resolve_tile_groups


LTA_SAVE_OPTION_TOKENS: Tuple[str, ...] = (
    "nrrd",
    "images",
    "labels",
    "overlay",
    "voxel_volume",
    "summary",
)


@dataclass(frozen=True)
class LtaSaveRequest:
    """Resolved optional LTA publications; the run manifest remains mandatory."""

    tokens: Tuple[str, ...] = ()

    def enabled(self, token: str) -> bool:
        return str(token) in self.tokens


@dataclass(frozen=True)
class LtaConfig:
    """Strict public configuration passed to the future LTA runtime."""

    args: argparse.Namespace
    device_ids: Tuple[int, ...]
    exemplar_dirs: Tuple[str, ...]
    cartesian_views: Tuple[str, ...]
    radial_requests: Tuple[RadialViewRequest, ...]
    tilted_groups: Tuple[TiltedViewGroup, ...]
    tiles: Tuple[ResolvedTileGroup, ...]
    angles: Tuple[float, ...]
    save: LtaSaveRequest
    postprocessing: PostprocessingRequest

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


def _token_list(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else list(values)
    tokens: list[str] = []
    for raw in raw_values:
        tokens.extend(
            token for token in re.split(r"[,\s]+", str(raw).strip()) if token
        )
    return tokens


def resolve_lta_device_ids(values: Sequence[str] | str | None) -> Tuple[int, ...]:
    """Resolve the required logical CUDA-device list without a default."""

    resolved: list[int] = []
    for raw in _token_list(values):
        token = str(raw).strip().lower()
        if token.startswith("gpu:") or token.startswith("cuda:"):
            token = token.split(":", 1)[1].strip()
        if not token.isdigit():
            raise ValueError(
                "--device accepts non-negative logical CUDA indexes, for example "
                "--device 0 or --device 0,2"
            )
        device_id = int(token)
        if device_id not in resolved:
            resolved.append(device_id)
    if not resolved:
        raise ValueError("--device must select at least one logical CUDA index")
    return tuple(resolved)


def resolve_lta_save_request(
    values: Sequence[str] | str | None,
) -> LtaSaveRequest:
    tokens: list[str] = []
    valid = set(LTA_SAVE_OPTION_TOKENS)
    for raw in _token_list(values):
        token = str(raw).lower()
        if token not in valid:
            expected = ", ".join(LTA_SAVE_OPTION_TOKENS)
            raise ValueError(
                f"--save values for LTA must be one or more of: {expected}; got {raw!r}"
            )
        if token in tokens:
            raise ValueError(f"--save contains duplicate LTA output {token!r}")
        tokens.append(token)
    return LtaSaveRequest(tokens=tuple(tokens))


def build_lta_argparser(*, prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "v19 label-time augmentation planning prototype with local SAM visual "
            "exemplars and TTA-compatible inference views; publication is not connected yet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True, help="Target media file or volume directory")
    parser.add_argument("--output", required=True, help="Publication root")
    parser.add_argument(
        "--temp",
        default=None,
        help="Optional scratch root; the runtime creates a unique per-run directory beneath it",
    )
    parser.add_argument("--model", required=True, help="Local SAM model bundle")
    parser.add_argument(
        "--device",
        required=True,
        nargs="+",
        metavar="GPU_INDEX",
        help="One or more logical CUDA indexes into CUDA_VISIBLE_DEVICES; no default",
    )
    parser.add_argument(
        "--exemplar",
        nargs="+",
        default=None,
        metavar="LABELED_EXEMPLAR_DIR",
        help="Optional additional labeled exemplar directories",
    )

    parser.add_argument(
        "--enable_cartesian",
        nargs="+",
        default=None,
        metavar="VIEW",
        help="Inference-only Cartesian views: transverse, sagittal, coronal",
    )
    parser.add_argument(
        "--enable_radial",
        nargs="+",
        default=None,
        metavar="VIEWS[:AZIMUTH_ANGLE]",
        help="Inference-only structured Radial view groups",
    )
    parser.add_argument(
        "--enable_tilted",
        nargs="+",
        default=None,
        metavar="VIEW[:TILT_ANGLE[:TILT_DIRECTION]]",
        help="Inference-only structured Tilted view groups",
    )
    parser.add_argument(
        "--enable_tile",
        nargs="+",
        default=None,
        metavar="TILE_SIZE:TILE_STRIDE",
        help="Inference-only dense tile groups applied to selected parent views",
    )
    parser.add_argument(
        "--angle",
        nargs="+",
        default=["0,120,240"],
        metavar="DEG",
        help="TTA in-plane angles; comma-separated and whitespace-separated forms are accepted",
    )

    parser.add_argument(
        "--sam_execution",
        default="video",
        choices=("image", "video"),
        help="SAM image ablation or fixed-session video tracking for every runtime view",
    )
    parser.add_argument(
        "--conf",
        default=0.15,
        type=float,
        help="SAM instance-admission threshold with the existing XTA confidence semantics",
    )
    parser.add_argument(
        "--save",
        nargs="+",
        default=None,
        metavar="OUTPUT",
        help=(
            "Optional outputs: nrrd, images, labels, overlay, voxel_volume, summary. "
            "The machine-readable manifest is always required by the runtime"
        ),
    )
    parser.add_argument(
        "--postprocessing",
        nargs="+",
        default=None,
        metavar="OPERATION[:PARAMETERS]",
        help=(
            "Shared terminal-union operations: keep_objects[:N], 3d_void_fill, "
            "gaussian_smoothing[:SIGMA][:PASSES]"
        ),
    )
    return parser


def resolve_lta_config(args: argparse.Namespace) -> LtaConfig:
    for field_name in ("input", "output", "model"):
        if not str(getattr(args, field_name)).strip():
            raise ValueError(f"--{field_name} must not be empty")
    if not math.isfinite(float(args.conf)) or not 0.0 <= float(args.conf) <= 1.0:
        raise ValueError("--conf must be finite and in [0,1]")
    exemplar_dirs = tuple(str(value).strip() for value in (args.exemplar or ()))
    if any(not value for value in exemplar_dirs):
        raise ValueError("--exemplar values must not be empty")

    config = LtaConfig(
        args=args,
        device_ids=resolve_lta_device_ids(args.device),
        exemplar_dirs=exemplar_dirs,
        cartesian_views=tuple(resolve_cartesian_views(args.enable_cartesian)),
        radial_requests=tuple(resolve_radial_view_requests(args.enable_radial)),
        tilted_groups=tuple(resolve_tilted_view_groups(args.enable_tilted)),
        tiles=tuple(resolve_tile_groups(args.enable_tile)),
        angles=tuple(float(value) for value in resolve_tta_angles(args.angle)),
        save=resolve_lta_save_request(args.save),
        postprocessing=resolve_postprocessing_options(args.postprocessing),
    )
    if not config.has_physical_views:
        raise ValueError(
            "No LTA inference views are active. Enable at least one view with "
            "--enable_cartesian, --enable_tilted, or --enable_radial; "
            "--enable_tile does not create a parent view"
        )
    return config


def parse_lta_args(
    argv: Optional[Sequence[str]] = None,
    *,
    prog: Optional[str] = None,
) -> LtaConfig:
    parser = build_lta_argparser(prog=prog)
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        return resolve_lta_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.error must terminate") from exc


__all__ = (
    "LTA_SAVE_OPTION_TOKENS",
    "LtaConfig",
    "LtaSaveRequest",
    "build_lta_argparser",
    "parse_lta_args",
    "resolve_lta_config",
    "resolve_lta_device_ids",
    "resolve_lta_save_request",
)
