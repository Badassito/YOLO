"""Generate a full-volume mask with the experimental SAM 3.1 mask-seed path.

This harness deliberately lives outside the public LTA runtime.  It expands the
known-working authoritative-mask experiment in three ways:

* every overlapping native-resolution tile is eligible without a parent gate;
* every positive exemplar frame refreshes its tile, propagating both forward
  and backward over the nearest-exemplar temporal domain; and
* an optional bounded-memory mode carries the last per-object masks into the
  next overlapping chunk ("dogfooding" the tracker output).

The primary review artifacts are offset-aware, single-segment ``.seg.nrrd``
chunks.  A full native-resolution overlay is assembled from lossless tile-mask
videos after every tile completes.  This remains a diagnostic experiment over
pinned private SAM internals, not an approved LTA publication path.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for entry in (ROOT, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from lta_gpu_smoke import (  # noqa: E402
    _MeasuredPredictor,
    configure_constrained_gpu_batches,
    cuda_snapshot,
    find_case_video,
    install_sdpa_fallback,
    resolve_pinned_sam_runtime_provenance,
)
from XTA.lta_experimental import (  # noqa: E402
    resolve_mask_seed_capacity,
    run_mask_seed_session,
)
from XTA.lta_tiles import (  # noqa: E402
    TilePlan,
    eight_neighbor_graph,
    masks_for_tile,
    plan_tile_grid,
)
from XTA.lta_inputs import parse_yolo_segmentation_label, split_indexed_stem  # noqa: E402
from XTA.lta_sam import (  # noqa: E402
    LTA_MAX_NUM_OBJECTS,
    LTA_MULTIPLEX_COUNT,
    build_local_sam_predictor,
    resolve_local_sam_bundle,
)


EXPERIMENT_SCHEMA = "lta.full-volume-mask-seed/1"
DEFAULT_TILE_SIZE = 1008
DEFAULT_TILE_STRIDE = 756
DEFAULT_NRRD_GZIP_LEVEL = 3
DEFAULT_OVERLAY_ALPHA = 0.60
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class ExperimentSessionPlan:
    """An experiment-only session plan without the production 30-frame cap."""

    sequence_id: str
    session_index: int
    frame_start: int
    frame_stop: int

    def __post_init__(self) -> None:
        sequence_id = str(self.sequence_id).strip()
        if not sequence_id:
            raise ValueError("sequence_id must not be empty")
        if isinstance(self.session_index, bool) or int(self.session_index) < 0:
            raise ValueError("session_index must be >= 0")
        if isinstance(self.frame_start, bool) or int(self.frame_start) < 0:
            raise ValueError("frame_start must be >= 0")
        if isinstance(self.frame_stop, bool) or int(self.frame_stop) <= int(self.frame_start):
            raise ValueError("frame_stop must be greater than frame_start")
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "session_index", int(self.session_index))
        object.__setattr__(self, "frame_start", int(self.frame_start))
        object.__setattr__(self, "frame_stop", int(self.frame_stop))

    @property
    def frame_count(self) -> int:
        return int(self.frame_stop) - int(self.frame_start)


@dataclass(frozen=True)
class PositiveAnchor:
    """One labeled source frame used as an authoritative tracker refresh."""

    encoded_index: int
    frame_index: int
    image_path: Path
    label_path: Path
    label_sha256: str
    polygons: tuple[Any, ...]

    def __post_init__(self) -> None:
        if int(self.encoded_index) < 0 or int(self.frame_index) < 0:
            raise ValueError("anchor indexes must be non-negative")
        if not self.polygons:
            raise ValueError("a positive anchor must contain at least one polygon")


@dataclass(frozen=True)
class AnchorDomain:
    """Half-open temporal range owned by its nearest positive anchor."""

    anchor_frame: int
    frame_start: int
    frame_stop: int

    def __post_init__(self) -> None:
        if not int(self.frame_start) <= int(self.anchor_frame) < int(self.frame_stop):
            raise ValueError("anchor must fall inside its temporal domain")


@dataclass(frozen=True)
class WindowPlan:
    """One SAM session, including the frame shared with a dogfood neighbor."""

    branch: str
    ordinal: int
    frame_start: int
    frame_stop: int
    prompt_frame: int
    direction: str
    seed_kind: str

    def __post_init__(self) -> None:
        if str(self.branch) not in {"center", "backward", "forward"}:
            raise ValueError(f"unknown window branch: {self.branch}")
        if str(self.direction) not in {"both", "backward", "forward"}:
            raise ValueError(f"unknown propagation direction: {self.direction}")
        if str(self.seed_kind) not in {"authoritative", "dogfood"}:
            raise ValueError(f"unknown seed kind: {self.seed_kind}")
        if not int(self.frame_start) <= int(self.prompt_frame) < int(self.frame_stop):
            raise ValueError("window prompt must fall inside the frame range")

    @property
    def frame_count(self) -> int:
        return int(self.frame_stop) - int(self.frame_start)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def probe_video(path: Path) -> dict[str, Any]:
    """Read stable source geometry needed by the experiment planner."""

    import cv2

    path = Path(path).resolve(strict=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    try:
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width < 1 or height < 1 or frame_count < 1:
        raise RuntimeError(
            f"invalid video geometry {width}x{height}x{frame_count}: {path}"
        )
    if not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"invalid video frame rate {fps!r}: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def discover_positive_anchors(
    exemplar_root: Path,
    *,
    frame_count: int,
    index_origin: int = 1,
) -> tuple[PositiveAnchor, ...]:
    """Discover every nonempty indexed image/YOLO pair deterministically."""

    exemplar_root = Path(exemplar_root).resolve(strict=True)
    if not exemplar_root.is_dir():
        raise ValueError(f"--exemplar-root is not a directory: {exemplar_root}")
    images_by_stem: dict[str, Path] = {}
    for path in sorted(exemplar_root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in images_by_stem:
                raise ValueError(f"duplicate exemplar image stem: {path.stem}")
            images_by_stem[path.stem] = path.resolve()

    by_frame: dict[int, PositiveAnchor] = {}
    for label_path in sorted(exemplar_root.glob("*.txt")):
        digest, polygons = parse_yolo_segmentation_label(label_path)
        if not polygons:
            continue
        _stem, encoded_index = split_indexed_stem(label_path)
        if encoded_index is None:
            raise ValueError(f"positive exemplar has no encoded frame index: {label_path}")
        frame_index = int(encoded_index) - int(index_origin)
        if not 0 <= frame_index < int(frame_count):
            raise ValueError(
                f"positive exemplar index {encoded_index} maps outside [0,{frame_count}): "
                f"{label_path}"
            )
        image_path = images_by_stem.get(label_path.stem)
        if image_path is None:
            raise ValueError(f"positive exemplar has no stem-matched image: {label_path}")
        if frame_index in by_frame:
            raise ValueError(
                f"multiple positive exemplar pairs map to decoded frame {frame_index}"
            )
        by_frame[frame_index] = PositiveAnchor(
            encoded_index=int(encoded_index),
            frame_index=frame_index,
            image_path=image_path,
            label_path=label_path.resolve(),
            label_sha256=str(digest),
            polygons=tuple(polygons),
        )
    if not by_frame:
        raise ValueError(f"no positive exemplar pairs found under {exemplar_root}")
    return tuple(by_frame[index] for index in sorted(by_frame))


def discover_known_background_frames(
    exemplar_root: Path,
    *,
    frame_count: int,
    index_origin: int = 1,
) -> tuple[int, ...]:
    """Return explicit empty-label frames for validation, never suppression."""

    exemplar_root = Path(exemplar_root).resolve(strict=True)
    image_stems = {
        path.stem
        for path in exemplar_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    frames = []
    for label_path in sorted(exemplar_root.glob("*.txt")):
        _digest, polygons = parse_yolo_segmentation_label(label_path)
        if polygons:
            continue
        _stem, encoded_index = split_indexed_stem(label_path)
        if encoded_index is None:
            raise ValueError(f"known-background label has no encoded index: {label_path}")
        frame_index = int(encoded_index) - int(index_origin)
        if not 0 <= frame_index < int(frame_count):
            raise ValueError(
                f"known-background exemplar index {encoded_index} maps outside "
                f"[0,{frame_count}): {label_path}"
            )
        if label_path.stem not in image_stems:
            raise ValueError(f"known-background label has no stem-matched image: {label_path}")
        frames.append(frame_index)
    if len(frames) != len(set(frames)):
        raise ValueError("multiple known-background pairs map to one decoded frame")
    return tuple(sorted(frames))


def anchor_masks_for_tile(anchor: PositiveAnchor, tile: TilePlan) -> tuple[Any, ...]:
    """Rasterize every authoritative polygon portion visible in one tile."""

    return masks_for_tile(anchor.polygons, tile)


def plan_anchor_domains(
    anchor_frames: Iterable[int],
    *,
    frame_start: int,
    frame_stop: int,
) -> tuple[AnchorDomain, ...]:
    """Partition the selected range at midpoints between local exemplars."""

    frame_start = int(frame_start)
    frame_stop = int(frame_stop)
    if frame_start < 0 or frame_stop <= frame_start:
        raise ValueError("invalid selected frame range")
    frames = tuple(sorted(set(int(value) for value in anchor_frames)))
    if not frames:
        return ()
    if frames[0] < frame_start or frames[-1] >= frame_stop:
        raise ValueError("all domain anchors must fall inside the selected frame range")
    domains = []
    for index, anchor in enumerate(frames):
        start = (
            frame_start
            if index == 0
            else (int(frames[index - 1]) + int(anchor)) // 2 + 1
        )
        stop = (
            frame_stop
            if index + 1 == len(frames)
            else (int(anchor) + int(frames[index + 1])) // 2 + 1
        )
        domains.append(AnchorDomain(anchor, start, stop))
    if domains[0].frame_start != frame_start or domains[-1].frame_stop != frame_stop:
        raise RuntimeError("anchor-domain planning did not cover the selected range")
    for left, right in zip(domains, domains[1:]):
        if int(left.frame_stop) != int(right.frame_start):
            raise RuntimeError("anchor-domain planning produced a gap or overlap")
    return tuple(domains)


def plan_domain_windows(domain: AnchorDomain, chunk_frames: int) -> tuple[WindowPlan, ...]:
    """Plan one uncapped session or center-out dogfood sessions for a domain.

    ``chunk_frames=0`` removes the experiment-only cap and runs the complete
    nearest-exemplar domain in one SAM session.  Positive values must be at
    least two because chained windows overlap their prompt boundary by one
    frame.
    """

    chunk_frames = int(chunk_frames)
    domain_count = int(domain.frame_stop) - int(domain.frame_start)
    if chunk_frames < 0 or chunk_frames == 1:
        raise ValueError("--chunk-frames must be 0 (uncapped) or >= 2")
    if chunk_frames == 0 or domain_count <= chunk_frames:
        return (
            WindowPlan(
                branch="center",
                ordinal=0,
                frame_start=int(domain.frame_start),
                frame_stop=int(domain.frame_stop),
                prompt_frame=int(domain.anchor_frame),
                direction="both",
                seed_kind="authoritative",
            ),
        )

    left_budget = (chunk_frames - 1) // 2
    center_start = max(int(domain.frame_start), int(domain.anchor_frame) - left_budget)
    center_stop = min(int(domain.frame_stop), center_start + chunk_frames)
    if center_stop - center_start < chunk_frames:
        center_start = max(int(domain.frame_start), center_stop - chunk_frames)
    center = WindowPlan(
        branch="center",
        ordinal=0,
        frame_start=center_start,
        frame_stop=center_stop,
        prompt_frame=int(domain.anchor_frame),
        direction="both",
        seed_kind="authoritative",
    )

    backward = []
    boundary = int(center.frame_start)
    ordinal = 1
    while boundary > int(domain.frame_start):
        start = max(int(domain.frame_start), boundary - (chunk_frames - 1))
        backward.append(
            WindowPlan(
                branch="backward",
                ordinal=ordinal,
                frame_start=start,
                frame_stop=boundary + 1,
                prompt_frame=boundary,
                direction="backward",
                seed_kind="dogfood",
            )
        )
        boundary = start
        ordinal += 1

    forward = []
    boundary = int(center.frame_stop) - 1
    ordinal = 1
    while boundary + 1 < int(domain.frame_stop):
        stop = min(int(domain.frame_stop), boundary + chunk_frames)
        forward.append(
            WindowPlan(
                branch="forward",
                ordinal=ordinal,
                frame_start=boundary,
                frame_stop=stop,
                prompt_frame=boundary,
                direction="forward",
                seed_kind="dogfood",
            )
        )
        boundary = stop - 1
        ordinal += 1
    return (center, *backward, *forward)


def decode_rgb_tile_range(
    video_path: Path,
    frame_start: int,
    frame_stop: int,
    tile: TilePlan,
) -> list[Any]:
    """Decode an arbitrary half-open native tile range as RGB PIL frames."""

    import cv2
    from PIL import Image

    frame_start = int(frame_start)
    frame_stop = int(frame_stop)
    if frame_start < 0 or frame_stop <= frame_start:
        raise ValueError("invalid decode frame range")
    capture = cv2.VideoCapture(str(Path(video_path)))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    frames = []
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_start):
            raise RuntimeError(f"OpenCV could not seek to frame {frame_start}")
        reported = int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES))))
        if reported != frame_start:
            raise RuntimeError(
                f"OpenCV reported frame {reported} after seek to {frame_start}: {video_path}"
            )
        left, top, right, bottom = tile.xyxy
        for frame_index in range(frame_start, frame_stop):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"OpenCV stopped at frame {frame_index}: {video_path}")
            if frame.shape[1] != tile.source_width or frame.shape[0] != tile.source_height:
                raise RuntimeError(f"video dimensions changed inside tile range: {frame.shape}")
            if frame.ndim == 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError(f"unsupported decoded frame shape {frame.shape}")
            cropped = rgb[top:bottom, left:right]
            if cropped.shape != (tile.size, tile.size, 3):
                raise RuntimeError(f"decoded tile has wrong shape {cropped.shape}")
            frames.append(Image.fromarray(cropped.copy(), mode="RGB"))
    finally:
        capture.release()
    expected = frame_stop - frame_start
    if len(frames) != expected:
        raise RuntimeError(f"decoded {len(frames)} frames, expected {expected}")
    return frames


def _nrrd_float(value: float) -> str:
    return format(float(value), ".17g")


def _nrrd_vector(values: Sequence[float]) -> str:
    return "(" + ",".join(_nrrd_float(value) for value in values) + ")"


def _mask_extent(volume: Any) -> tuple[int, int, int, int, int, int]:
    import numpy as np

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"mask extent requires a 3-D volume; got {array.shape}")
    bounds: list[int] | None = None
    for z_index in range(int(array.shape[0])):
        ys, xs = np.nonzero(array[z_index])
        if not len(xs):
            continue
        current = [
            int(xs.min()),
            int(xs.max()),
            int(ys.min()),
            int(ys.max()),
            int(z_index),
            int(z_index),
        ]
        if bounds is None:
            bounds = current
        else:
            bounds[0] = min(bounds[0], current[0])
            bounds[1] = max(bounds[1], current[1])
            bounds[2] = min(bounds[2], current[2])
            bounds[3] = max(bounds[3], current[3])
            bounds[5] = int(z_index)
    if bounds is None:
        return 0, -1, 0, -1, 0, -1
    return tuple(bounds)  # type: ignore[return-value]


def write_offset_seg_nrrd(
    path: Path,
    volume_zyx: Any,
    *,
    tile: TilePlan,
    frame_start: int,
    full_frame_count: int,
    spacing_xyz: Sequence[float] = (1.0, 1.0, 1.0),
    segment_name: str = "SAM mask",
    gzip_level: int = DEFAULT_NRRD_GZIP_LEVEL,
) -> dict[str, Any]:
    """Write one cropped, world-offset binary segmentation for 3D Slicer.

    The source MKV has no patient-space transform, so the default geometry is
    explicitly voxel-index LPS with unit spacing.  Supplying physical spacing
    scales both directions and the chunk origin; every tile/chunk consequently
    shares one global coordinate system.
    """

    import numpy as np

    path = Path(path)
    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"NRRD volume must have shape (z,y,x); got {volume.shape}")
    depth, height, width = (int(value) for value in volume.shape)
    if depth < 1 or height != int(tile.size) or width != int(tile.size):
        raise ValueError(
            f"NRRD volume shape {volume.shape} does not match tile {tile.size}x{tile.size}"
        )
    spacing = tuple(float(value) for value in spacing_xyz)
    if len(spacing) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError("voxel spacing must contain three finite positive values")
    if not 0 <= int(frame_start) < int(full_frame_count):
        raise ValueError("NRRD frame start is outside the full source volume")
    if int(frame_start) + depth > int(full_frame_count):
        raise ValueError("NRRD chunk extends beyond the full source volume")
    if not 0 <= int(gzip_level) <= 9:
        raise ValueError("NRRD gzip level must be in [0,9]")

    extent = _mask_extent(volume)
    sx, sy, sz = spacing
    origin = (
        float(tile.left) * sx,
        float(tile.top) * sy,
        float(frame_start) * sz,
    )
    safe_name = str(segment_name).replace("\n", " ").replace("\r", " ")
    header_lines = [
        "NRRD0005",
        f"# Complete NRRD file generated by {Path(__file__).name}",
        "type: uint8",
        "dimension: 3",
        f"sizes: {width} {height} {depth}",
        "space: left-posterior-superior",
        "kinds: domain domain domain",
        "space directions: "
        + " ".join(
            (
                _nrrd_vector((sx, 0.0, 0.0)),
                _nrrd_vector((0.0, sy, 0.0)),
                _nrrd_vector((0.0, 0.0, sz)),
            )
        ),
        f"space origin: {_nrrd_vector(origin)}",
        "encoding: gzip",
        "endian: little",
        (
            "content: binary SAM tile chunk; exported_axes=(X,Y,frame); "
            f"global_offset=({tile.left},{tile.top},{int(frame_start)})"
        ),
        "Segmentation_ContainedRepresentationNames:=Binary labelmap|",
        "Segmentation_MasterRepresentation:=Binary labelmap",
        "Segmentation_SourceRepresentation:=Binary labelmap",
        (
            "Segmentation_ReferenceImageExtent:="
            f"0 {int(tile.source_width)-1} 0 {int(tile.source_height)-1} "
            f"0 {int(full_frame_count)-1}"
        ),
        (
            "Segmentation_ReferenceImageExtentOffset:="
            f"{int(tile.left)} {int(tile.top)} {int(frame_start)}"
        ),
        "Segment0_ID:=Segment_1",
        f"Segment0_Name:={safe_name}",
        "Segment0_NameAutoGenerated:=0",
        "Segment0_Color:=0.83 0.16 0.16",
        "Segment0_ColorAutoGenerated:=0",
        "Segment0_LabelValue:=1",
        "Segment0_Layer:=0",
        "Segment0_Extent:=" + " ".join(str(value) for value in extent),
        "Segment0_Tags:=TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|",
    ]
    header = ("\n".join(header_lines) + "\n\n").encode("ascii", errors="strict")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(header)
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=handle,
                compresslevel=int(gzip_level),
                mtime=0,
            ) as compressed:
                for z_index in range(depth):
                    slab = np.ascontiguousarray(volume[z_index] != 0, dtype=np.uint8)
                    compressed.write(slab.tobytes(order="C"))
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "shape_zyx": [depth, height, width],
        "global_offset_xyz": [int(tile.left), int(tile.top), int(frame_start)],
        "space_origin": list(origin),
        "space_directions": [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sz]],
        "segment_extent_local_x_y_z": list(extent),
        "nonzero_voxels": int(np.count_nonzero(volume)),
    }


def owned_frame_range(window: WindowPlan) -> tuple[int, int]:
    """Exclude the repeated dogfood context frame from a chunk's artifact."""

    if window.seed_kind == "authoritative":
        return int(window.frame_start), int(window.frame_stop)
    if window.direction == "backward":
        return int(window.frame_start), int(window.frame_stop) - 1
    if window.direction == "forward":
        return int(window.frame_start) + 1, int(window.frame_stop)
    raise RuntimeError("a dogfood window must be directional")


def _prediction_masks_at_frame(
    predictions: Iterable[Any],
    frame_index: int,
) -> tuple[Any, ...]:
    """Return nonempty per-object masks in stable object-ID order."""

    import numpy as np

    selected = sorted(
        (item for item in predictions if int(item.frame_index) == int(frame_index)),
        key=lambda item: int(item.object_id),
    )
    object_ids = [int(item.object_id) for item in selected]
    if len(object_ids) != len(set(object_ids)):
        raise RuntimeError(f"duplicate object prediction at frame {frame_index}")
    return tuple(
        np.asarray(item.binary_mask, dtype=bool).copy()
        for item in selected
        if bool(np.asarray(item.binary_mask, dtype=bool).any())
    )


def _union_predictions_into_store(
    predictions: Iterable[Any],
    *,
    store_zyx: Any,
    selected_frame_start: int,
    window: WindowPlan,
    tile_size: int,
) -> dict[str, Any]:
    """OR one chunk into the disk-backed tile result without a 3-D RAM copy."""

    import numpy as np

    seen: set[tuple[int, int]] = set()
    active_frames: set[int] = set()
    owned_active_frames: set[int] = set()
    object_ids: set[int] = set()
    owned_start, owned_stop = owned_frame_range(window)
    for item in predictions:
        frame_index = int(item.frame_index)
        object_id = int(item.object_id)
        identity = frame_index, object_id
        if identity in seen:
            raise RuntimeError(f"duplicate prediction for frame/object {identity}")
        seen.add(identity)
        if not int(window.frame_start) <= frame_index < int(window.frame_stop):
            raise RuntimeError(
                f"prediction frame {frame_index} lies outside window "
                f"[{window.frame_start},{window.frame_stop})"
            )
        binary = np.asarray(item.binary_mask, dtype=bool)
        if binary.shape != (int(tile_size), int(tile_size)):
            raise RuntimeError(
                f"prediction mask shape {binary.shape} != {(tile_size, tile_size)}"
            )
        target_index = frame_index - int(selected_frame_start)
        if not 0 <= target_index < int(store_zyx.shape[0]):
            raise RuntimeError(f"prediction frame {frame_index} lies outside tile store")
        if bool(binary.any()):
            active_frames.add(frame_index)
            if owned_start <= frame_index < owned_stop:
                store_zyx[target_index] |= binary
                owned_active_frames.add(frame_index)
        object_ids.add(object_id)
    return {
        "prediction_records": len(seen),
        "returned_object_ids": sorted(object_ids),
        "model_active_frames": sorted(active_frames),
        "owned_active_frames": sorted(owned_active_frames),
    }


def _window_artifact_name(
    *,
    tile_index: int,
    tile: TilePlan,
    anchor_frame: int,
    window: WindowPlan,
) -> str:
    owned_start, owned_stop = owned_frame_range(window)
    return (
        f"lta_tile{int(tile_index):02d}_x{tile.left:04d}-{tile.left+tile.size-1:04d}_"
        f"y{tile.top:04d}-{tile.top+tile.size-1:04d}_"
        f"z{owned_start:04d}-{owned_stop-1:04d}_"
        f"anchor{int(anchor_frame):04d}_{window.branch}{window.ordinal:02d}.seg.nrrd"
    )


def execute_window(
    measured: Any,
    predictor: Any,
    *,
    video_path: Path,
    tile_index: int,
    tile: TilePlan,
    anchor_frame: int,
    window: WindowPlan,
    seed_masks: Sequence[Any],
    selected_frame_start: int,
    tile_store_zyx: Any,
    nrrd_directory: Path,
    full_frame_count: int,
    spacing_xyz: Sequence[float],
    conf: float,
    session_index: int,
) -> tuple[dict[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    """Run, persist, and reduce one authoritative or dogfood session."""

    import numpy as np

    masks = tuple(np.asarray(mask, dtype=bool) for mask in seed_masks if bool(np.asarray(mask).any()))
    if not masks:
        raise ValueError("a tracker window requires at least one nonempty seed mask")
    if len(masks) > LTA_MAX_NUM_OBJECTS:
        raise ValueError(
            f"window seed count {len(masks)} exceeds SAM capacity {LTA_MAX_NUM_OBJECTS}"
        )
    ground_truth = np.logical_or.reduce(masks)
    resource = None
    result = None
    try:
        resource = decode_rgb_tile_range(
            video_path,
            int(window.frame_start),
            int(window.frame_stop),
            tile,
        )
        session = ExperimentSessionPlan(
            sequence_id=f"lta__full_tile_{int(tile_index):02d}",
            session_index=int(session_index),
            frame_start=int(window.frame_start),
            frame_stop=int(window.frame_stop),
        )
        result = run_mask_seed_session(
            measured,
            predictor,
            resource=resource,
            session=session,
            prompt_frame=int(window.prompt_frame),
            ground_truth=ground_truth,
            seed=None,
            object_masks=masks,
            conf=float(conf),
            propagation_mode="tracker-only",
        )
        predictions = tuple(result["propagation"])
        reduction = _union_predictions_into_store(
            predictions,
            store_zyx=tile_store_zyx,
            selected_frame_start=int(selected_frame_start),
            window=window,
            tile_size=int(tile.size),
        )
        # The seed is authoritative.  Apply it before publishing the owned
        # NRRD so the chunk, tile mask video, and final overlay cannot diverge
        # even if a future private tracker revision changes anchor handling.
        prompt_store_index = int(window.prompt_frame) - int(selected_frame_start)
        tile_store_zyx[prompt_store_index] |= ground_truth
        owned_start, owned_stop = owned_frame_range(window)
        if owned_stop <= owned_start:
            raise RuntimeError("dogfood context trimming left an empty owned range")
        owned_view = tile_store_zyx[
            owned_start - int(selected_frame_start) : owned_stop - int(selected_frame_start)
        ]
        nrrd_path = Path(nrrd_directory) / _window_artifact_name(
            tile_index=tile_index,
            tile=tile,
            anchor_frame=anchor_frame,
            window=window,
        )
        nrrd = write_offset_seg_nrrd(
            nrrd_path,
            owned_view,
            tile=tile,
            frame_start=owned_start,
            full_frame_count=int(full_frame_count),
            spacing_xyz=spacing_xyz,
            segment_name=(
                f"LTA SAM tile {tile_index:02d}, frames {owned_start}-{owned_stop-1}"
            ),
        )
        first_masks = _prediction_masks_at_frame(predictions, int(window.frame_start))
        last_masks = _prediction_masks_at_frame(predictions, int(window.frame_stop) - 1)
        record = {
            "anchor_frame": int(anchor_frame),
            "window": asdict(window),
            "session_frame_range": [int(window.frame_start), int(window.frame_stop)],
            "owned_frame_range": [owned_start, owned_stop],
            "input_seed_object_count": len(masks),
            "output_first_object_count": len(first_masks),
            "output_last_object_count": len(last_masks),
            "anchor_integrity_passed": bool(result["anchor_integrity_passed"]),
            "anchor_preview_propagation_iou": result[
                "anchor_preview_propagation_iou"
            ],
            "anchor_propagation_object_metrics": result[
                "anchor_propagation_object_metrics"
            ],
            "seed_object_metrics": result["seed_object_metrics"],
            "diagnostic_propagation_gate_passed": bool(
                result["diagnostic_propagation_gate_passed"]
            ),
            "propagation_response_count": int(result["propagation_response_count"]),
            **reduction,
            "nrrd": {
                **nrrd,
                "path": nrrd_path.name,
            },
        }
        return record, first_masks, last_masks
    finally:
        result = None
        resource = None
        gc.collect()


def write_tile_mask_video(
    path: Path,
    store_zyx: Any,
    *,
    fps: float,
) -> dict[str, Any]:
    """Encode one lossless full-length tile mask stream for overlay assembly."""

    import cv2
    import numpy as np

    path = Path(path)
    if len(store_zyx.shape) != 3:
        raise ValueError("tile mask store must have shape (frames,height,width)")
    frame_count, height, width = (int(value) for value in store_zyx.shape)
    if frame_count < 1 or height < 1 or width < 1:
        raise ValueError("tile mask store dimensions must be positive")
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("mask-video fps must be finite and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial{path.suffix}")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"FFV1"),
        float(fps),
        (width, height),
        isColor=False,
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the lossless FFV1 tile-mask video")
    foreground = 0
    try:
        for frame_index in range(frame_count):
            frame = np.asarray(store_zyx[frame_index] != 0, dtype=np.uint8) * 255
            foreground += int(np.count_nonzero(frame))
            writer.write(np.ascontiguousarray(frame))
    except BaseException:
        writer.release()
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    else:
        writer.release()
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("FFV1 tile-mask writer produced no payload")
    os.replace(temporary, path)
    return {
        "path": path.name,
        "codec": "ffv1",
        "lossless": True,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": float(fps),
        "foreground_voxels": foreground,
        "sha256": _sha256_file(path),
    }


def write_full_overlay(
    source_video: Path,
    *,
    tile_inputs: Sequence[tuple[TilePlan, Path]],
    frame_start: int,
    frame_stop: int,
    output_path: Path,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    ffmpeg_path: str | None = None,
    known_background_frames: Iterable[int] = (),
) -> dict[str, Any]:
    """Stream the parent-free tile union into one native H.264 review video."""

    import cv2
    import numpy as np

    if not tile_inputs:
        raise ValueError("overlay assembly requires at least one tile mask video")
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("overlay alpha must be in (0,1]")
    source_video = Path(source_video).resolve(strict=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the full-volume overlay")

    source = cv2.VideoCapture(str(source_video))
    masks = [(tile, cv2.VideoCapture(str(path))) for tile, path in tile_inputs]
    if not source.isOpened() or any(not capture.isOpened() for _tile, capture in masks):
        source.release()
        for _tile, capture in masks:
            capture.release()
        raise RuntimeError("OpenCV could not open the source or one tile-mask video")
    width = int(round(float(source.get(cv2.CAP_PROP_FRAME_WIDTH))))
    height = int(round(float(source.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    fps = float(source.get(cv2.CAP_PROP_FPS))
    expected = int(frame_stop) - int(frame_start)
    if expected < 1:
        source.release()
        for _tile, capture in masks:
            capture.release()
        raise ValueError("overlay frame range must be nonempty")
    if not source.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start)):
        source.release()
        for _tile, capture in masks:
            capture.release()
        raise RuntimeError(f"OpenCV could not seek source to frame {frame_start}")
    reported_source_frame = int(round(float(source.get(cv2.CAP_PROP_POS_FRAMES))))
    if reported_source_frame != int(frame_start):
        source.release()
        for _tile, capture in masks:
            capture.release()
        raise RuntimeError(
            f"OpenCV reported source frame {reported_source_frame} after seek to "
            f"{frame_start}: {source_video}"
        )

    temporary = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    log_path = output_path.with_suffix(output_path.suffix + ".ffmpeg.log")
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        format(fps, ".12g"),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv444p",
        str(temporary),
    ]
    process = None
    written = 0
    foreground = 0
    background_set = {
        int(value)
        for value in known_background_frames
        if int(frame_start) <= int(value) < int(frame_stop)
    }
    background_false_positive_pixels = 0
    background_false_positive_frames: list[dict[str, int]] = []
    try:
        with log_path.open("wb") as log_handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log_handle,
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg overlay encoder exposed no stdin")
            for offset in range(expected):
                ok, source_frame = source.read()
                if not ok or source_frame is None:
                    raise RuntimeError(
                        f"source decode stopped at frame {int(frame_start) + offset}"
                    )
                if source_frame.ndim == 2:
                    source_frame = cv2.cvtColor(source_frame, cv2.COLOR_GRAY2BGR)
                elif source_frame.ndim == 3 and source_frame.shape[2] == 1:
                    source_frame = cv2.cvtColor(source_frame, cv2.COLOR_GRAY2BGR)
                elif source_frame.ndim != 3 or source_frame.shape[2] != 3:
                    raise RuntimeError(f"unsupported source overlay frame {source_frame.shape}")
                union = np.zeros((height, width), dtype=bool)
                for tile, capture in masks:
                    mask_ok, mask_frame = capture.read()
                    if not mask_ok or mask_frame is None:
                        raise RuntimeError(
                            f"tile mask decode stopped at relative frame {offset}: {tile.xyxy}"
                        )
                    plane = mask_frame if mask_frame.ndim == 2 else mask_frame[..., 0]
                    if plane.shape != (tile.size, tile.size):
                        raise RuntimeError(
                            f"tile mask frame shape {plane.shape} != {(tile.size, tile.size)}"
                        )
                    left, top, right, bottom = tile.xyxy
                    union[top:bottom, left:right] |= plane > 127
                count = int(np.count_nonzero(union))
                foreground += count
                global_frame = int(frame_start) + offset
                if global_frame in background_set and count:
                    background_false_positive_pixels += count
                    background_false_positive_frames.append(
                        {"frame_index": global_frame, "mask_pixels": count}
                    )
                if count:
                    red = np.asarray((32, 32, 255), dtype=np.float32)
                    blended = np.rint(
                        source_frame[union].astype(np.float32) * (1.0 - alpha)
                        + red * alpha
                    ).astype(np.uint8)
                    source_frame[union] = blended
                process.stdin.write(np.ascontiguousarray(source_frame).tobytes(order="C"))
                written += 1
            process.stdin.close()
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg overlay encoder exited with {return_code}; see {log_path}"
            )
        if written != expected or not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("overlay encoder produced an incomplete artifact")
        os.replace(temporary, output_path)
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        source.release()
        for _tile, capture in masks:
            capture.release()
    return {
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "codec": "h264/libx264",
        "pixel_format": "yuv444p",
        "frame_start": int(frame_start),
        "frame_stop_exclusive": int(frame_stop),
        "frame_count": written,
        "width": width,
        "height": height,
        "fps": fps,
        "foreground_voxel_visits": foreground,
        "alpha": alpha,
        "ffmpeg_log": str(log_path),
        "known_background_validation": {
            "checked_frame_count": len(background_set),
            "false_positive_frame_count": len(background_false_positive_frames),
            "false_positive_pixels": background_false_positive_pixels,
            "false_positive_frames": background_false_positive_frames,
            "policy": "validation only; explicit empty labels never forced output to zero",
        },
    }


def _tile_directory_name(tile_index: int, tile: TilePlan) -> str:
    return (
        f"tile_{int(tile_index):02d}_x{tile.left:04d}-{tile.left+tile.size-1:04d}_"
        f"y{tile.top:04d}-{tile.top+tile.size-1:04d}"
    )


def _seed_masks_for_layout(
    anchor: PositiveAnchor,
    tile: TilePlan,
    seed_layout: str,
) -> tuple[Any, ...]:
    import numpy as np

    masks = anchor_masks_for_tile(anchor, tile)
    if not masks:
        return ()
    if str(seed_layout) == "instances":
        return masks
    if str(seed_layout) == "union":
        return (np.logical_or.reduce(masks),)
    raise ValueError(f"unknown seed layout: {seed_layout}")


def build_plan(
    *,
    video: Mapping[str, Any],
    bundle: Any,
    exemplar_root: Path,
    anchors: Sequence[PositiveAnchor],
    known_background_frames: Sequence[int],
    tiles: Sequence[TilePlan],
    selected_tile_indices: Sequence[int],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
    seed_layout: str,
    spacing_xyz: Sequence[float],
    conf: float,
    profile_requested: str,
) -> dict[str, Any]:
    """Build a JSON-stable execution contract before CUDA is touched."""

    selected = set(int(value) for value in selected_tile_indices)
    neighbor_graph = eight_neighbor_graph(tiles)
    anchor_records = [
        {
            "encoded_index": int(anchor.encoded_index),
            "decoded_frame_index": int(anchor.frame_index),
            "image_path": str(anchor.image_path),
            "label_path": str(anchor.label_path),
            "label_sha256": str(anchor.label_sha256),
            "polygon_count": len(anchor.polygons),
        }
        for anchor in anchors
        if int(frame_start) <= int(anchor.frame_index) < int(frame_stop)
    ]
    tile_records = []
    maximum_seed_objects = 0
    total_model_frame_visits = 0
    total_windows = 0
    for tile_index, tile in enumerate(tiles):
        if tile_index not in selected:
            continue
        local = []
        for anchor in anchors:
            if not int(frame_start) <= int(anchor.frame_index) < int(frame_stop):
                continue
            # Match runtime eligibility exactly.  A polygon bounding box can
            # cross a tile corner while the non-convex polygon has no raster
            # support there (an edge tile with no local anchor is a real example).
            intersecting = len(anchor_masks_for_tile(anchor, tile))
            if not intersecting:
                continue
            seed_objects = 1 if str(seed_layout) == "union" else intersecting
            maximum_seed_objects = max(maximum_seed_objects, seed_objects)
            local.append(
                {
                    "frame_index": int(anchor.frame_index),
                    "encoded_index": int(anchor.encoded_index),
                    "intersecting_polygon_count": int(intersecting),
                    "seed_object_count": int(seed_objects),
                }
            )
        domains = plan_anchor_domains(
            (item["frame_index"] for item in local),
            frame_start=int(frame_start),
            frame_stop=int(frame_stop),
        )
        domain_records = []
        for domain in domains:
            windows = plan_domain_windows(domain, int(chunk_frames))
            model_visits = sum(window.frame_count for window in windows)
            total_model_frame_visits += model_visits
            total_windows += len(windows)
            domain_records.append(
                {
                    "anchor_frame": int(domain.anchor_frame),
                    "frame_range": [int(domain.frame_start), int(domain.frame_stop)],
                    "windows": [asdict(window) for window in windows],
                    "model_frame_visits_including_context": int(model_visits),
                }
            )
        tile_records.append(
            {
                "tile_index": int(tile_index),
                "tile_xyxy": list(tile.xyxy),
                "directory": _tile_directory_name(tile_index, tile),
                "parent_gate": False,
                "neighbors": [
                    {
                        "tile_index": edge.destination_index,
                        "direction": edge.direction,
                        "overlap_xyxy": list(edge.overlap_xyxy),
                    }
                    for edge in neighbor_graph[tile_index]
                    if edge.destination_index in selected
                ],
                "anchors": local,
                "domains": domain_records,
                "complete_output_frame_count": int(frame_stop) - int(frame_start),
            }
        )

    if maximum_seed_objects < 1:
        raise ValueError("the selected tiles/frame range contain no positive exemplar masks")
    if maximum_seed_objects > LTA_MAX_NUM_OBJECTS:
        raise ValueError(
            f"one tile anchor requires {maximum_seed_objects} objects, exceeding "
            f"SAM's {LTA_MAX_NUM_OBJECTS}-object experiment cap"
        )
    payload = {
        "experiment_schema": EXPERIMENT_SCHEMA,
        "code_identity_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "XTA" / "lta_experimental.py",
                ROOT / "XTA" / "lta_tile_tracking.py",
                ROOT / "XTA" / "lta_tiles.py",
                TOOLS / "lta_gpu_smoke.py",
                ROOT / "XTA" / "lta_sam.py",
                ROOT / "XTA" / "lta_inputs.py",
            )
        },
        "diagnostic_only": True,
        "private_unstable_api": True,
        "video": dict(video),
        "model": {
            "bundle_root": str(bundle.root),
            "checkpoint_path": str(bundle.checkpoint_path),
            "model_version": str(bundle.model_version),
            "checkpoint_identity_sha256": str(bundle.checkpoint_identity_sha256),
        },
        "exemplar_root": str(Path(exemplar_root).resolve()),
        "positive_anchors": anchor_records,
        "known_background_frames": [
            int(value)
            for value in known_background_frames
            if int(frame_start) <= int(value) < int(frame_stop)
        ],
        "frame_range": [int(frame_start), int(frame_stop)],
        "tile_size": int(tiles[0].size),
        "tile_stride_requested": None,
        "selected_tile_count": len(tile_records),
        "seedable_tile_count": sum(bool(item["anchors"]) for item in tile_records),
        "unseeded_tile_indices": [
            int(item["tile_index"]) for item in tile_records if not item["anchors"]
        ],
        "full_grid_tile_count": len(tiles),
        "tiles": tile_records,
        "chunk_frames": int(chunk_frames),
        "uncapped_sam_domains": int(chunk_frames) == 0,
        "dogfood_overlap_frames": 0 if int(chunk_frames) == 0 else 1,
        "dogfood_context_is_excluded_from_next_owned_nrrd": True,
        "seed_layout": str(seed_layout),
        "conf": float(conf),
        "profile_requested": str(profile_requested),
        "maximum_seed_object_count": int(maximum_seed_objects),
        "model_object_capacity": int(resolve_mask_seed_capacity(maximum_seed_objects)),
        "total_planned_windows": int(total_windows),
        "total_planned_model_frame_visits": int(total_model_frame_visits),
        "voxel_spacing_xyz": [float(value) for value in spacing_xyz],
        "geometry_interpretation": (
            "index-coordinate LPS; MKV contains no patient-space origin/directions; "
            "video FPS is provenance and is not Z spacing"
        ),
        "authoritative_refresh_policy": (
            "each positive label owns the nearest-exemplar temporal domain; a fresh "
            "session replaces carried predictions at every domain anchor"
        ),
        "known_background_policy": (
            "empty exemplar labels are validation evidence only and never force output zero"
        ),
        "spatial_relay_adapter": {
            "module": "XTA.lta_tile_tracking",
            "topology": "symmetric_eight_neighbor",
            "outbound_seed": "last_active_overlap_frame_forward",
            "inbound_seed": "first_active_overlap_frame_backward",
            "execution_connected": False,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _selected_tile_indices(values: Sequence[int] | None, tile_count: int) -> tuple[int, ...]:
    if not values:
        return tuple(range(int(tile_count)))
    selected = tuple(sorted(set(int(value) for value in values)))
    invalid = [value for value in selected if not 0 <= value < int(tile_count)]
    if invalid:
        raise ValueError(f"--tile-index outside [0,{tile_count}): {invalid}")
    return selected


def _tile_plan_by_index(plan: Mapping[str, Any], tiles: Sequence[TilePlan]) -> dict[int, Any]:
    selected = {int(item["tile_index"]): item for item in plan["tiles"]}
    if any(index >= len(tiles) for index in selected):
        raise RuntimeError("serialized plan references a missing grid tile")
    return selected


def _safe_remove_partial_tile(path: Path, tiles_root: Path) -> None:
    """Remove only an explicitly named partial child while resuming a run."""

    path = Path(path).resolve(strict=True)
    root = Path(tiles_root).resolve(strict=True)
    if path.parent != root or not path.name.endswith(".partial"):
        raise RuntimeError(f"refusing to remove unsafe partial-tile path: {path}")
    shutil.rmtree(path)


def validate_completed_tile_artifacts(
    tile_directory: Path,
    summary: Mapping[str, Any],
) -> None:
    """Fail closed if a resume tile lost or changed any settled artifact."""

    tile_directory = Path(tile_directory).resolve(strict=True)

    def validate_receipt(receipt: Mapping[str, Any], *, label: str) -> None:
        relative = Path(str(receipt.get("path", "")))
        if not relative.name or relative.is_absolute() or relative.parent != Path("."):
            raise ValueError(f"completed tile has unsafe {label} path: {relative}")
        candidate = (tile_directory / relative).resolve(strict=True)
        if candidate.parent != tile_directory or not candidate.is_file():
            raise ValueError(f"completed tile {label} is missing: {candidate}")
        expected = str(receipt.get("sha256", ""))
        actual = _sha256_file(candidate)
        if len(expected) != 64 or actual != expected:
            raise ValueError(f"completed tile {label} checksum mismatch: {candidate}")

    mask_video = summary.get("tile_mask_video")
    if not isinstance(mask_video, Mapping):
        raise ValueError("completed tile has no mask-video receipt")
    validate_receipt(mask_video, label="mask video")
    windows = summary.get("executed_windows")
    if not isinstance(windows, list):
        raise ValueError("completed tile has no window receipts")
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping) or not isinstance(window.get("nrrd"), Mapping):
            raise ValueError(f"completed tile window {index} has no NRRD receipt")
        validate_receipt(window["nrrd"], label=f"window {index} NRRD")


def run_tile(
    measured: Any,
    predictor: Any,
    *,
    output_root: Path,
    plan_sha256: str,
    execution_sha256: str,
    execution_profile: Mapping[str, Any],
    video_path: Path,
    video_info: Mapping[str, Any],
    tile_index: int,
    tile: TilePlan,
    anchors: Sequence[PositiveAnchor],
    known_background_frames: Sequence[int],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
    seed_layout: str,
    spacing_xyz: Sequence[float],
    conf: float,
    resume: bool,
    artifacts_prevalidated: bool = False,
) -> dict[str, Any]:
    """Run one complete spatial tile as an atomic, resumable transaction."""

    import numpy as np

    tiles_root = Path(output_root) / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)
    directory_name = _tile_directory_name(tile_index, tile)
    complete = tiles_root / directory_name
    partial = tiles_root / f"{directory_name}.partial"
    if complete.is_dir():
        summary_path = complete / "tile_summary.json"
        if not resume or not summary_path.is_file():
            raise ValueError(f"completed tile output already exists: {complete}")
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(prior.get("plan_sha256")) != str(plan_sha256):
            raise ValueError(f"completed tile belongs to a different plan: {complete}")
        if str(prior.get("execution_sha256")) != str(execution_sha256):
            raise ValueError(
                f"completed tile belongs to a different execution profile: {complete}"
            )
        if not artifacts_prevalidated:
            validate_completed_tile_artifacts(complete, prior)
        return prior
    if partial.exists():
        if not resume:
            raise ValueError(f"partial tile output exists; pass --resume: {partial}")
        _safe_remove_partial_tile(partial, tiles_root)
    partial.mkdir(parents=False, exist_ok=False)

    local_anchors: list[tuple[PositiveAnchor, tuple[Any, ...]]] = []
    for anchor in anchors:
        if not int(frame_start) <= int(anchor.frame_index) < int(frame_stop):
            continue
        masks = _seed_masks_for_layout(anchor, tile, seed_layout)
        if masks:
            local_anchors.append((anchor, masks))
    domains = plan_anchor_domains(
        (anchor.frame_index for anchor, _masks in local_anchors),
        frame_start=int(frame_start),
        frame_stop=int(frame_stop),
    )
    anchors_by_frame = {int(anchor.frame_index): (anchor, masks) for anchor, masks in local_anchors}
    if len(anchors_by_frame) != len(local_anchors):
        raise RuntimeError("duplicate local anchor frame")

    selected_count = int(frame_stop) - int(frame_start)
    work_path = partial / "tile_union.u8.dat"
    store = np.memmap(
        work_path,
        dtype=np.uint8,
        mode="w+",
        shape=(selected_count, int(tile.size), int(tile.size)),
    )
    records: list[dict[str, Any]] = []
    halted: list[dict[str, Any]] = []
    session_index = 0
    active_error: BaseException | None = None
    try:
        for domain_index, domain in enumerate(domains):
            anchor, authoritative_masks = anchors_by_frame[int(domain.anchor_frame)]
            windows = plan_domain_windows(domain, int(chunk_frames))
            center = windows[0]
            print(
                f"tile {tile_index:02d}: anchor {domain.anchor_frame}, "
                f"center [{center.frame_start},{center.frame_stop})",
                file=sys.stderr,
                flush=True,
            )
            record, first_masks, last_masks = execute_window(
                measured,
                predictor,
                video_path=video_path,
                tile_index=tile_index,
                tile=tile,
                anchor_frame=int(anchor.frame_index),
                window=center,
                seed_masks=authoritative_masks,
                selected_frame_start=int(frame_start),
                tile_store_zyx=store,
                nrrd_directory=partial,
                full_frame_count=int(video_info["frame_count"]),
                spacing_xyz=spacing_xyz,
                conf=float(conf),
                session_index=session_index,
            )
            record["domain_index"] = int(domain_index)
            record["seed_label_path"] = str(anchor.label_path)
            record["seed_label_sha256"] = str(anchor.label_sha256)
            records.append(record)
            session_index += 1

            carry = first_masks
            for window in (item for item in windows[1:] if item.branch == "backward"):
                if not carry:
                    halted.append(
                        {
                            "anchor_frame": int(anchor.frame_index),
                            "branch": "backward",
                            "unreached_frame_range": [
                                int(domain.frame_start),
                                int(window.frame_stop) - 1,
                            ],
                            "reason": "empty dogfood boundary",
                        }
                    )
                    break
                print(
                    f"tile {tile_index:02d}: dogfood backward "
                    f"[{window.frame_start},{window.frame_stop})",
                    file=sys.stderr,
                    flush=True,
                )
                record, first_masks, _last_masks = execute_window(
                    measured,
                    predictor,
                    video_path=video_path,
                    tile_index=tile_index,
                    tile=tile,
                    anchor_frame=int(anchor.frame_index),
                    window=window,
                    seed_masks=carry,
                    selected_frame_start=int(frame_start),
                    tile_store_zyx=store,
                    nrrd_directory=partial,
                    full_frame_count=int(video_info["frame_count"]),
                    spacing_xyz=spacing_xyz,
                    conf=float(conf),
                    session_index=session_index,
                )
                record["domain_index"] = int(domain_index)
                records.append(record)
                carry = first_masks
                session_index += 1

            carry = last_masks
            for window in (item for item in windows[1:] if item.branch == "forward"):
                if not carry:
                    halted.append(
                        {
                            "anchor_frame": int(anchor.frame_index),
                            "branch": "forward",
                            "unreached_frame_range": [
                                int(window.frame_start) + 1,
                                int(domain.frame_stop),
                            ],
                            "reason": "empty dogfood boundary",
                        }
                    )
                    break
                print(
                    f"tile {tile_index:02d}: dogfood forward "
                    f"[{window.frame_start},{window.frame_stop})",
                    file=sys.stderr,
                    flush=True,
                )
                record, _first_masks, last_masks = execute_window(
                    measured,
                    predictor,
                    video_path=video_path,
                    tile_index=tile_index,
                    tile=tile,
                    anchor_frame=int(anchor.frame_index),
                    window=window,
                    seed_masks=carry,
                    selected_frame_start=int(frame_start),
                    tile_store_zyx=store,
                    nrrd_directory=partial,
                    full_frame_count=int(video_info["frame_count"]),
                    spacing_xyz=spacing_xyz,
                    conf=float(conf),
                    session_index=session_index,
                )
                record["domain_index"] = int(domain_index)
                records.append(record)
                carry = last_masks
                session_index += 1

        # Source labels are immutable hard positives even if a future tracker
        # revision stops returning its conditioning frame byte-for-byte.
        authoritative_voxels = 0
        for anchor, masks in local_anchors:
            union = np.logical_or.reduce(tuple(np.asarray(mask, dtype=bool) for mask in masks))
            store[int(anchor.frame_index) - int(frame_start)] |= union
            authoritative_voxels += int(np.count_nonzero(union))
        background_frames = [
            int(value)
            for value in known_background_frames
            if int(frame_start) <= int(value) < int(frame_stop)
        ]
        background_false_positives = []
        background_false_positive_pixels = 0
        for background_frame in background_frames:
            pixels = int(
                np.count_nonzero(store[background_frame - int(frame_start)])
            )
            if pixels:
                background_false_positive_pixels += pixels
                background_false_positives.append(
                    {"frame_index": background_frame, "mask_pixels": pixels}
                )
        store.flush()
        mask_video = write_tile_mask_video(
            partial / "tile_mask.mkv",
            store,
            fps=float(video_info["fps"]),
        )
        summary = {
            "status": "complete",
            "plan_sha256": str(plan_sha256),
            "execution_sha256": str(execution_sha256),
            "execution_profile": dict(execution_profile),
            "tile_index": int(tile_index),
            "tile_xyxy": list(tile.xyxy),
            "parent_gate": False,
            "frame_range": [int(frame_start), int(frame_stop)],
            "seed_layout": str(seed_layout),
            "positive_anchor_frames": [
                int(anchor.frame_index) for anchor, _masks in local_anchors
            ],
            "positive_anchor_count": len(local_anchors),
            "authoritative_hard_positive_voxels": authoritative_voxels,
            "known_background_validation": {
                "checked_frame_count": len(background_frames),
                "false_positive_frame_count": len(background_false_positives),
                "false_positive_pixels": background_false_positive_pixels,
                "false_positive_frames": background_false_positives,
                "policy": "validation only; explicit empty labels never forced output to zero",
            },
            "executed_window_count": len(records),
            "executed_windows": records,
            "halted_empty_dogfood_branches": halted,
            "tile_mask_video": mask_video,
        }
        _atomic_write_json(partial / "tile_summary.json", summary)
    except BaseException as exc:
        active_error = exc
        try:
            store.flush()
        except Exception:
            pass
        raise
    finally:
        mmap = getattr(store, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del store
        gc.collect()
    if active_error is not None:  # pragma: no cover - the except path re-raises
        raise active_error
    work_path.unlink(missing_ok=True)
    os.replace(partial, complete)
    final_summary = json.loads((complete / "tile_summary.json").read_text(encoding="utf-8"))
    return final_summary


def _resolve_profile(torch_module: Any, device: int, requested: str) -> dict[str, Any]:
    capability = tuple(int(value) for value in torch_module.cuda.get_device_capability(int(device)))
    profile = str(requested)
    if profile == "auto":
        profile = "h100" if capability[0] >= 9 else "egpu"
    if profile == "h100":
        if capability[0] < 9:
            raise ValueError(
                "--profile h100 requires CUDA capability 9.x or newer; "
                f"device {device} reports {capability[0]}.{capability[1]}"
            )
        return {
            "name": profile,
            "weight_storage": "float32",
            # External FA3 is optional and has no compatible prebuilt wheel in
            # the qualified Torch 2.13/CUDA 13 cluster environment. Building
            # its 293 CUDA translation units exhausted a 200 GiB SLURM step.
            # Use PyTorch SDPA, whose ordered backend policy still prefers its
            # built-in flash kernel before efficient attention and math.
            "use_fa3": False,
            # Real-valued RoPE is needed only for torch.compile compatibility;
            # this diagnostic harness deliberately runs compile=False.
            "use_rope_real": False,
            "constrained_batches": False,
            "sdpa_fallback": True,
            "attention_backend": "torch_sdpa_flash_efficient_math",
        }
    if profile == "egpu":
        return {
            "name": profile,
            "weight_storage": "bfloat16_egpu",
            "use_fa3": False,
            "use_rope_real": False,
            "constrained_batches": True,
            "sdpa_fallback": True,
            "attention_backend": "torch_sdpa_flash_efficient_math",
        }
    raise ValueError(f"unknown execution profile: {requested}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True, help="Local SAM 3.1 model bundle")
    parser.add_argument("--input-root", required=True, help="Directory containing the source MKV")
    parser.add_argument(
        "--exemplar-root",
        required=True,
        help="Full source image/YOLO pair directory; nonempty labels become refresh anchors",
    )
    parser.add_argument("--output", required=True, help="New output directory, or existing with --resume")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--profile", choices=("auto", "h100", "egpu"), default="auto")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=None)
    parser.add_argument("--exemplar-index-origin", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument(
        "--tile-index",
        type=int,
        action="append",
        default=None,
        help="Run only this row-major tile index (repeatable); default is the complete grid",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=48,
        help=(
            "SAM frames per bounded session including one overlap; 0 runs each complete "
            "nearest-exemplar domain uncapped (recommended only with ample host RAM)"
        ),
    )
    parser.add_argument(
        "--seed-layout",
        choices=("union", "instances"),
        default="union",
        help="Track one semantic union per tile anchor or one object per polygon",
    )
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument(
        "--voxel-spacing",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(1.0, 1.0, 1.0),
        help="Optional physical spacing; defaults to aligned voxel-index geometry",
    )
    parser.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.chunk_frames) < 0 or int(args.chunk_frames) == 1:
        raise ValueError("--chunk-frames must be 0 or >= 2")
    if not 0.0 <= float(args.conf) <= 1.0:
        raise ValueError("--conf must be in [0,1]")
    spacing = tuple(float(value) for value in args.voxel_spacing)
    if any(not math.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError("--voxel-spacing values must be finite and positive")

    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists() and any(output.iterdir()) and not (args.resume or args.plan_only):
        raise ValueError(f"--output must be new/empty unless --resume is used: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = resolve_local_sam_bundle(args.model)
    if bundle.model_version != "sam3.1":
        raise ValueError("the full-volume mask-seed experiment requires SAM 3.1")
    input_root = Path(args.input_root).expanduser().resolve(strict=True)
    exemplar_root = Path(args.exemplar_root).expanduser().resolve(strict=True)
    video_path = find_case_video(input_root, "direct")
    video = probe_video(video_path)
    frame_start = int(args.frame_start)
    frame_stop = int(video["frame_count"]) if args.frame_stop is None else int(args.frame_stop)
    if not 0 <= frame_start < frame_stop <= int(video["frame_count"]):
        raise ValueError(
            f"selected frame range [{frame_start},{frame_stop}) lies outside "
            f"[0,{video['frame_count']})"
        )
    anchors = discover_positive_anchors(
        exemplar_root,
        frame_count=int(video["frame_count"]),
        index_origin=int(args.exemplar_index_origin),
    )
    known_background_frames = discover_known_background_frames(
        exemplar_root,
        frame_count=int(video["frame_count"]),
        index_origin=int(args.exemplar_index_origin),
    )
    tiles = plan_tile_grid(
        source_width=int(video["width"]),
        source_height=int(video["height"]),
        tile_size=int(args.tile_size),
        tile_stride=int(args.tile_stride),
    )
    tile_indices = _selected_tile_indices(args.tile_index, len(tiles))
    plan = build_plan(
        video=video,
        bundle=bundle,
        exemplar_root=exemplar_root,
        anchors=anchors,
        known_background_frames=known_background_frames,
        tiles=tiles,
        selected_tile_indices=tile_indices,
        frame_start=frame_start,
        frame_stop=frame_stop,
        chunk_frames=int(args.chunk_frames),
        seed_layout=str(args.seed_layout),
        spacing_xyz=spacing,
        conf=float(args.conf),
        profile_requested=str(args.profile),
    )
    plan["tile_stride_requested"] = int(args.tile_stride)
    # The stride is part of the immutable plan identity; recompute after filling it.
    plan_without_id = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(plan_without_id, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    plan_path = output / "run_plan.json"
    if plan_path.is_file():
        prior_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if str(prior_plan.get("plan_sha256")) != str(plan["plan_sha256"]):
            raise ValueError("existing output run_plan.json does not match this invocation")
    else:
        _atomic_write_json(plan_path, plan)
    print(
        json.dumps(
            {
                "status": "plan_complete" if args.plan_only else "execution_planned",
                "plan_path": str(plan_path),
                "plan_sha256": str(plan["plan_sha256"]),
                "selected_tile_count": int(plan["selected_tile_count"]),
                "seedable_tile_count": int(plan["seedable_tile_count"]),
                "unseeded_tile_indices": list(plan["unseeded_tile_indices"]),
                "positive_anchor_count": len(plan["positive_anchors"]),
                "known_background_frame_count": len(plan["known_background_frames"]),
                "planned_windows": int(plan["total_planned_windows"]),
                "planned_model_frame_visits": int(
                    plan["total_planned_model_frame_visits"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.plan_only:
        return

    runtime = resolve_pinned_sam_runtime_provenance()
    pending: list[int] = []
    completed_summaries: dict[int, dict[str, Any]] = {}
    for tile_index in tile_indices:
        tile = tiles[int(tile_index)]
        completed_path = (
            output
            / "tiles"
            / _tile_directory_name(tile_index, tile)
            / "tile_summary.json"
        )
        if args.resume and completed_path.is_file():
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            if str(completed.get("plan_sha256")) != str(plan["plan_sha256"]):
                raise ValueError(f"completed tile belongs to a different plan: {completed_path}")
            validate_completed_tile_artifacts(completed_path.parent, completed)
            completed_summaries[int(tile_index)] = completed
        else:
            pending.append(int(tile_index))

    torch = None
    profile: dict[str, Any]
    execution_contract: dict[str, Any]
    execution_sha256: str
    device_summary: dict[str, Any]
    memory_events: list[dict[str, Any]] = []
    peak_allocated_mib: int | None = None
    predictor = None
    measured = None
    restore_sdpa = None
    constrained = None
    active_error: BaseException | None = None
    tile_summaries: list[dict[str, Any]] = []

    if pending:
        import torch as runtime_torch

        torch = runtime_torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for pending tile inference")
        if not 0 <= int(args.device) < int(torch.cuda.device_count()):
            raise ValueError(
                f"--device {args.device} is outside CUDA device_count="
                f"{torch.cuda.device_count()}"
            )
        torch.cuda.set_device(int(args.device))
        profile = _resolve_profile(torch, int(args.device), str(args.profile))
        if profile["name"] == "egpu" and not 2 <= int(args.chunk_frames) <= 48:
            raise ValueError(
                "the constrained eGPU profile requires --chunk-frames in [2,48]"
            )
        execution_contract = {
            "plan_sha256": str(plan["plan_sha256"]),
            "profile": dict(profile),
            "conf": float(args.conf),
            "construction_device": "meta",
            "compile": False,
            "warm_up": False,
            "async_loading_frames": False,
            "multiplex_count": int(LTA_MULTIPLEX_COUNT),
            "max_num_objects": int(plan["model_object_capacity"]),
        }
        execution_sha256 = hashlib.sha256(
            json.dumps(
                execution_contract,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for tile_index, completed in completed_summaries.items():
            if str(completed.get("execution_sha256")) != execution_sha256:
                raise ValueError(
                    f"completed tile {tile_index} used a different execution profile"
                )
        torch.cuda.reset_peak_memory_stats(int(args.device))
        memory_events = [cuda_snapshot(torch, int(args.device), "before_builder")]
        device_summary = {
            "index": int(args.device),
            "name": torch.cuda.get_device_name(int(args.device)),
            "capability": list(torch.cuda.get_device_capability(int(args.device))),
            "profile": profile,
            "inference_reused_without_cuda": False,
        }
        try:
            print(
                f"Loading SAM 3.1 once for {len(pending)} pending tile(s) with "
                f"profile={profile['name']}...",
                file=sys.stderr,
                flush=True,
            )
            predictor = build_local_sam_predictor(
                bundle,
                device_id=int(args.device),
                use_fa3=bool(profile["use_fa3"]),
                use_rope_real=bool(profile["use_rope_real"]),
                compile=False,
                warm_up=False,
                async_loading_frames=False,
                conf=float(args.conf),
                weight_storage=str(profile["weight_storage"]),
                max_num_objects=int(plan["model_object_capacity"]),
                construction_device="meta",
            )
            if profile["constrained_batches"]:
                constrained = configure_constrained_gpu_batches(predictor)
            if profile["sdpa_fallback"]:
                restore_sdpa = install_sdpa_fallback()
            measured = _MeasuredPredictor(
                predictor,
                torch,
                int(args.device),
                memory_events,
            )
            for tile_index in tile_indices:
                tile = tiles[int(tile_index)]
                print(
                    f"Starting tile {tile_index:02d}/{len(tiles)-1:02d} {tile.xyxy}",
                    file=sys.stderr,
                    flush=True,
                )
                tile_summary = run_tile(
                    measured,
                    predictor,
                    output_root=output,
                    plan_sha256=str(plan["plan_sha256"]),
                    execution_sha256=execution_sha256,
                    execution_profile=execution_contract,
                    video_path=video_path,
                    video_info=video,
                    tile_index=int(tile_index),
                    tile=tile,
                    anchors=anchors,
                    known_background_frames=known_background_frames,
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                    chunk_frames=int(args.chunk_frames),
                    seed_layout=str(args.seed_layout),
                    spacing_xyz=spacing,
                    conf=float(args.conf),
                    resume=bool(args.resume),
                    artifacts_prevalidated=True,
                )
                tile_summaries.append(tile_summary)
        except BaseException as exc:
            active_error = exc
            raise
        finally:
            cleanup_errors: list[tuple[str, Exception]] = []
            if restore_sdpa is not None:
                try:
                    restore_sdpa()
                except Exception as exc:
                    cleanup_errors.append(("SDPA restoration", exc))
            if predictor is not None:
                shutdown = getattr(predictor, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception as exc:
                        cleanup_errors.append(("predictor shutdown", exc))
            predictor = None
            measured = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                cleanup_errors.append(("CUDA garbage collection", exc))
            peak_allocated_mib = int(
                torch.cuda.max_memory_allocated(int(args.device)) // (1024 * 1024)
            )
            if active_error is not None:
                add_note = getattr(active_error, "add_note", None)
                if callable(add_note):
                    for boundary, cleanup_error in cleanup_errors:
                        add_note(
                            f"SAM {boundary} also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            elif cleanup_errors:
                raise cleanup_errors[0][1]
    else:
        if not completed_summaries:
            raise RuntimeError("no pending or completed tiles were found")
        execution_hashes = {
            str(item.get("execution_sha256")) for item in completed_summaries.values()
        }
        if len(execution_hashes) != 1 or "None" in execution_hashes:
            raise ValueError("completed tiles do not share one execution fingerprint")
        execution_sha256 = next(iter(execution_hashes))
        contracts = {
            json.dumps(item.get("execution_profile"), sort_keys=True, separators=(",", ":"))
            for item in completed_summaries.values()
        }
        if len(contracts) != 1:
            raise ValueError("completed tiles do not share one execution contract")
        execution_contract = json.loads(next(iter(contracts)))
        recomputed_execution_sha256 = hashlib.sha256(
            json.dumps(
                execution_contract,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if recomputed_execution_sha256 != execution_sha256:
            raise ValueError("completed tile execution fingerprint is corrupt")
        if str(execution_contract.get("plan_sha256")) != str(plan["plan_sha256"]):
            raise ValueError("completed tile execution contract references another plan")
        profile = dict(execution_contract["profile"])
        prior_summary_path = output / "summary.json"
        prior_summary = (
            json.loads(prior_summary_path.read_text(encoding="utf-8"))
            if prior_summary_path.is_file()
            else None
        )
        if (
            isinstance(prior_summary, Mapping)
            and str(prior_summary.get("plan_sha256")) == str(plan["plan_sha256"])
            and str(prior_summary.get("execution_sha256")) == execution_sha256
        ):
            device_summary = dict(prior_summary.get("device") or {})
            device_summary["finalization_reused_inference_without_cuda"] = True
            memory_events = list(prior_summary.get("memory_events") or [])
            previous_peak = prior_summary.get("peak_allocated_mib")
            peak_allocated_mib = (
                None if previous_peak is None else int(previous_peak)
            )
            previous_settings = prior_summary.get("settings")
            if isinstance(previous_settings, Mapping):
                constrained = previous_settings.get("constrained_batches")
        else:
            device_summary = {
                "index": None,
                "name": None,
                "capability": None,
                "profile": profile,
                "inference_provenance_unavailable": True,
                "finalization_reused_inference_without_cuda": True,
            }
        tile_summaries = [completed_summaries[int(index)] for index in tile_indices]

    overlay = None
    if not args.skip_overlay:
        tile_inputs = []
        for tile_index in tile_indices:
            tile = tiles[int(tile_index)]
            mask_path = (
                output
                / "tiles"
                / _tile_directory_name(tile_index, tile)
                / "tile_mask.mkv"
            )
            tile_inputs.append((tile, mask_path.resolve(strict=True)))
        overlay = write_full_overlay(
            video_path,
            tile_inputs=tile_inputs,
            frame_start=frame_start,
            frame_stop=frame_stop,
            output_path=output / "lta_full_overlay.mkv",
            alpha=float(args.overlay_alpha),
            known_background_frames=known_background_frames,
        )

    summary = {
        "status": "execution_complete",
        "experiment_schema": EXPERIMENT_SCHEMA,
        "diagnostic_only": True,
        "private_unstable_api": True,
        "plan_sha256": str(plan["plan_sha256"]),
        "execution_sha256": execution_sha256,
        "execution_profile": execution_contract,
        "run_plan": str(plan_path),
        "runtime": runtime,
        "device": device_summary,
        "settings": {
            "conf": float(args.conf),
            "chunk_frames": int(args.chunk_frames),
            "seed_layout": str(args.seed_layout),
            "multiplex_count": int(LTA_MULTIPLEX_COUNT),
            "max_num_objects": int(plan["model_object_capacity"]),
            "constrained_batches": constrained,
        },
        "tile_summaries": [
            str(
                output
                / "tiles"
                / _tile_directory_name(int(item["tile_index"]), tiles[int(item["tile_index"])])
                / "tile_summary.json"
            )
            for item in tile_summaries
        ],
        "overlay": overlay,
        "peak_allocated_mib": peak_allocated_mib,
        "memory_events": memory_events,
    }
    summary_path = output / "summary.json"
    summary["summary_path"] = str(summary_path)
    _atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
