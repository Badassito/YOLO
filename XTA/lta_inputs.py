"""Dependency-light input discovery for Label-Time Augmentation (LTA).

The discovery layer deliberately does not import :mod:`XTA.pta`, OpenCV, NumPy,
SAM, or any inference backend.  It describes flat image/video volumes, parses
class-0 YOLO segmentation labels, preserves explicit-background versus unknown
frames, and builds a deterministic positive-exemplar pool.  Decoding and model
runtime concerns belong to later LTA layers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)
VIDEO_EXTENSIONS = frozenset(
    {".mkv", ".mp4", ".mov", ".avi", ".mpg", ".mpeg", ".m4v", ".webm"}
)
UNSUPPORTED_LABEL_EXTENSIONS = frozenset({".nrrd", ".nhdr"})


class LtaInputError(ValueError):
    """Raised when an LTA input layout or annotation is invalid."""


class NoPositiveExemplarError(LtaInputError):
    """Raised when target and exemplar inputs contain no positive polygon."""


class SourceRole(str, Enum):
    """Provenance role for discovered media."""

    TARGET = "input"
    EXEMPLAR = "exemplar"


class VolumeClass(str, Enum):
    """Annotation coverage of a discovered volume."""

    FULLY_LABELED = "fully_labeled"
    PARTIALLY_LABELED = "partially_labeled"
    UNLABELED = "unlabeled"


class AnnotationState(str, Enum):
    """Meaning of one frame's YOLO label state."""

    FOREGROUND = "foreground"
    KNOWN_BACKGROUND = "known_background"
    UNKNOWN = "unknown"


class ExemplarPreferenceTier(IntEnum):
    """Preference order for choosing a positive prompt for a target session."""

    SAME_TARGET_SESSION = 0
    SAME_TARGET_VOLUME = 1
    OTHER_TARGET_VOLUME = 2
    EXTERNAL_EXEMPLAR = 3


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        if int(self.frame_count) <= 0:
            raise LtaInputError("Video frame_count must be positive")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise LtaInputError(f"Video {name} must be positive when supplied")
        if self.fps is not None and (
            not math.isfinite(float(self.fps)) or float(self.fps) <= 0.0
        ):
            raise LtaInputError("Video fps must be finite and positive when supplied")


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise LtaInputError("Image dimensions must be positive")


@dataclass(frozen=True)
class YoloPolygon:
    class_id: int
    row_index: int
    points: Tuple[Tuple[float, float], ...]
    box_xyxy: Tuple[float, float, float, float]
    box_cxcywh: Tuple[float, float, float, float]
    normalized_area: float


@dataclass(frozen=True)
class IndexedMedia:
    encoded_index: int
    frame_position: int
    path: Path
    sha256: Optional[str]
    identity_sha256: str
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass(frozen=True)
class FrameAnnotation:
    encoded_index: int
    frame_position: int
    state: AnnotationState
    label_path: Optional[Path]
    label_sha256: Optional[str]
    polygons: Tuple[YoloPolygon, ...]


@dataclass(frozen=True)
class LtaVolumeSpec:
    source_role: SourceRole
    source_root: Path
    volume_id: str
    stem: str
    kind: str
    media: Tuple[IndexedMedia, ...]
    video_path: Optional[Path]
    video_sha256: Optional[str]
    video_identity_sha256: Optional[str]
    annotations: Tuple[FrameAnnotation, ...]
    volume_class: VolumeClass
    encoded_indices: Tuple[int, ...]
    index_origin: Optional[int]
    frame_count: int
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None

    @property
    def positive_count(self) -> int:
        return sum(len(annotation.polygons) for annotation in self.annotations)

    @property
    def known_background_count(self) -> int:
        return sum(
            annotation.state is AnnotationState.KNOWN_BACKGROUND
            for annotation in self.annotations
        )

    @property
    def unknown_count(self) -> int:
        return sum(
            annotation.state is AnnotationState.UNKNOWN
            for annotation in self.annotations
        )

    @property
    def has_complete_label_coverage(self) -> bool:
        return self.volume_class is VolumeClass.FULLY_LABELED

    def annotation_for_index(self, encoded_index: int) -> FrameAnnotation:
        for annotation in self.annotations:
            if int(annotation.encoded_index) == int(encoded_index):
                return annotation
        raise KeyError(encoded_index)

    def media_for_index(self, encoded_index: int) -> IndexedMedia:
        for item in self.media:
            if int(item.encoded_index) == int(encoded_index):
                return item
        raise KeyError(encoded_index)


@dataclass(frozen=True)
class PositiveExemplar:
    exemplar_id: str
    source_role: SourceRole
    source_root: Path
    volume_id: str
    volume_stem: str
    volume_kind: str
    encoded_frame_index: int
    frame_position: int
    media_path: Path
    media_sha256: Optional[str]
    media_identity_sha256: str
    label_path: Path
    label_sha256: str
    label_row_index: int
    class_id: int
    polygon: Tuple[Tuple[float, float], ...]
    box_xyxy: Tuple[float, float, float, float]
    box_cxcywh: Tuple[float, float, float, float]
    normalized_area: float
    bundle_sha256: str
    source_width: Optional[int] = None
    source_height: Optional[int] = None

    @property
    def target_preference_capable(self) -> bool:
        return self.source_role is SourceRole.TARGET


@dataclass(frozen=True)
class RankedExemplar:
    exemplar: PositiveExemplar
    preference_tier: ExemplarPreferenceTier
    preference_reason: str
    deterministic_tie_break: str


@dataclass(frozen=True)
class LtaDiscoveryWarning:
    code: str
    message: str
    volume_id: Optional[str] = None


@dataclass(frozen=True)
class LtaInputDiscovery:
    input_path: Path
    target_volumes: Tuple[LtaVolumeSpec, ...]
    exemplar_roots: Tuple[Path, ...]
    exemplar_volumes: Tuple[LtaVolumeSpec, ...]
    positive_pool: Tuple[PositiveExemplar, ...]
    warnings: Tuple[LtaDiscoveryWarning, ...]

    @property
    def all_volumes(self) -> Tuple[LtaVolumeSpec, ...]:
        return self.target_volumes + self.exemplar_volumes


VideoProbe = Callable[[Path], object]
ImageProbe = Callable[[Path], object]


_INDEXED_STEM = re.compile(r"^(.*)_(\d+)$")
_ROBOFLOW_INDEXED_STEM = re.compile(
    r"^(.*)_(\d+)_png\.rf\.[^.]+$",
    re.IGNORECASE,
)


def split_indexed_stem(path: Path) -> Tuple[str, Optional[int]]:
    """Split the final ``_NNNN`` suffix used by flat LTA image stacks."""

    roboflow = _ROBOFLOW_INDEXED_STEM.match(path.stem)
    if roboflow is not None and roboflow.group(1):
        return roboflow.group(1), int(roboflow.group(2))
    match = _INDEXED_STEM.match(path.stem)
    if match is None or not match.group(1):
        return path.stem, None
    return match.group(1), int(match.group(2))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity_sha256(path: Path) -> str:
    """Hash stable discovery identity fields without reading the file payload."""

    stat = path.stat()
    payload = "\0".join(
        (
            _canonical_path_key(path),
            str(int(stat.st_size)),
            str(int(stat.st_mtime_ns)),
            str(int(stat.st_ctime_ns)),
            str(int(stat.st_dev)),
            str(int(stat.st_ino)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve())).replace("\\", "/")


def _volume_id(role: SourceRole, root: Path, stem: str) -> str:
    return f"{role.value}:{_canonical_path_key(root)}::{stem}"


def _parse_ratio(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text in {"N/A", "0/0"}:
        return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0.0:
            return None
        return float(numerator) / denominator_value
    return float(text)


def probe_video_with_ffprobe(path: Path) -> VideoMetadata:
    """Read video metadata with ffprobe without importing a decode runtime."""

    executable = shutil.which("ffprobe")
    if executable is None:
        raise LtaInputError(
            f"ffprobe is required to discover video frame bounds: {path}"
        )
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LtaInputError(f"Could not probe video {path}: {exc}") from exc
    try:
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            raise ValueError("no video stream")
        stream = streams[0]
        frame_count_raw = stream.get("nb_read_frames")
        if frame_count_raw in (None, "", "N/A"):
            raise ValueError("ffprobe returned no decoded-frame count")
        frame_count = int(frame_count_raw)
        width = int(stream["width"])
        height = int(stream["height"])
        fps = _parse_ratio(stream.get("avg_frame_rate")) or _parse_ratio(
            stream.get("r_frame_rate")
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LtaInputError(f"Invalid ffprobe metadata for {path}: {exc}") from exc
    return VideoMetadata(
        frame_count=int(frame_count),
        width=width,
        height=height,
        fps=fps,
    )


def parse_yolo_segmentation_label(path: Path) -> Tuple[str, Tuple[YoloPolygon, ...]]:
    """Parse one UTF-8 class-0 YOLO segmentation file.

    A zero-byte or whitespace-only file is an explicit known-background label.
    """

    payload = path.read_bytes()
    label_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LtaInputError(f"YOLO label is not UTF-8: {path}") from exc

    polygons = []
    for row_index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            class_id = int(tokens[0])
        except (IndexError, ValueError) as exc:
            raise LtaInputError(
                f"Invalid YOLO class at {path}:{row_index + 1}"
            ) from exc
        if class_id != 0:
            raise LtaInputError(
                f"LTA accepts only YOLO class 0; found {class_id} at "
                f"{path}:{row_index + 1}"
            )
        coordinate_tokens = tokens[1:]
        if len(coordinate_tokens) < 6 or len(coordinate_tokens) % 2:
            raise LtaInputError(
                f"YOLO segmentation row needs at least three coordinate pairs at "
                f"{path}:{row_index + 1}"
            )
        try:
            coordinates = tuple(float(token) for token in coordinate_tokens)
        except ValueError as exc:
            raise LtaInputError(
                f"Invalid YOLO coordinate at {path}:{row_index + 1}"
            ) from exc
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in coordinates
        ):
            raise LtaInputError(
                f"YOLO coordinates must be finite and normalized to [0, 1] at "
                f"{path}:{row_index + 1}"
            )
        points = tuple(zip(coordinates[0::2], coordinates[1::2]))
        xs = tuple(point[0] for point in points)
        ys = tuple(point[1] for point in points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        doubled_signed_area = sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
        )
        normalized_area = abs(float(doubled_signed_area)) * 0.5
        if (
            max_x - min_x <= 1e-12
            or max_y - min_y <= 1e-12
            or normalized_area <= 1e-12
        ):
            raise LtaInputError(
                f"Degenerate YOLO polygon at {path}:{row_index + 1}"
            )
        polygons.append(
            YoloPolygon(
                class_id=0,
                row_index=int(row_index),
                points=points,
                box_xyxy=(min_x, min_y, max_x, max_y),
                box_cxcywh=(
                    (min_x + max_x) * 0.5,
                    (min_y + max_y) * 0.5,
                    max_x - min_x,
                    max_y - min_y,
                ),
                normalized_area=normalized_area,
            )
        )
    return label_sha256, tuple(polygons)


def _normalize_video_metadata(value: object, path: Path) -> VideoMetadata:
    if isinstance(value, VideoMetadata):
        return value
    if isinstance(value, int):
        return VideoMetadata(frame_count=value)
    if isinstance(value, (tuple, list)):
        try:
            if len(value) == 1:
                return VideoMetadata(frame_count=int(value[0]))
            if len(value) == 4:
                return VideoMetadata(
                    frame_count=int(value[0]),
                    width=(int(value[1]) if value[1] is not None else None),
                    height=(int(value[2]) if value[2] is not None else None),
                    fps=(float(value[3]) if value[3] is not None else None),
                )
        except (TypeError, ValueError) as exc:
            raise LtaInputError(f"Invalid video probe result for {path}: {value!r}") from exc
    if isinstance(value, Mapping):
        try:
            frame_count_raw = (
                value["frame_count"]
                if "frame_count" in value
                else value["num_frames"]
            )
            return VideoMetadata(
                frame_count=int(frame_count_raw),
                width=(int(value["width"]) if value.get("width") is not None else None),
                height=(int(value["height"]) if value.get("height") is not None else None),
                fps=(float(value["fps"]) if value.get("fps") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LtaInputError(f"Invalid video probe result for {path}: {value!r}") from exc
    raise LtaInputError(f"Invalid video probe result for {path}: {value!r}")


def _normalize_image_metadata(value: object, path: Path) -> ImageMetadata:
    if isinstance(value, ImageMetadata):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return ImageMetadata(width=int(value[0]), height=int(value[1]))
        except (TypeError, ValueError) as exc:
            raise LtaInputError(f"Invalid image probe result for {path}: {value!r}") from exc
    if isinstance(value, Mapping):
        try:
            return ImageMetadata(width=int(value["width"]), height=int(value["height"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LtaInputError(f"Invalid image probe result for {path}: {value!r}") from exc
    raise LtaInputError(f"Invalid image probe result for {path}: {value!r}")


def _add_unique_indexed(
    groups: dict[str, dict[int, Path]],
    *,
    stem: str,
    index: int,
    path: Path,
    kind: str,
) -> None:
    existing = groups.setdefault(stem, {}).get(int(index))
    if existing is not None:
        raise LtaInputError(
            f"Duplicate {kind} frame for {stem}_{index}: "
            f"{existing.name}, {path.name}"
        )
    groups[stem][int(index)] = path


def _selected_labels_for_media(media_path: Path) -> Tuple[Path, ...]:
    parent = media_path.parent
    if media_path.suffix.lower() in IMAGE_EXTENSIONS:
        media_stem, media_index = split_indexed_stem(media_path)
        labels = []
        for path in parent.iterdir():
            if not path.is_file() or path.suffix.lower() != ".txt":
                continue
            label_stem, label_index = split_indexed_stem(path)
            if media_index is None:
                matches = label_index is None and label_stem == media_stem
            else:
                matches = label_stem == media_stem and label_index == media_index
            if matches:
                labels.append(path.resolve())
        return tuple(sorted(labels, key=lambda item: item.name))
    labels = []
    for path in parent.iterdir():
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue
        stem, _index = split_indexed_stem(path)
        if stem == media_path.stem or path.stem == media_path.stem:
            labels.append(path.resolve())
    return tuple(sorted(labels, key=lambda item: item.name))


def _resolve_video_origin(
    label_indices: Sequence[int],
    *,
    frame_count: int,
    stem: str,
) -> int:
    if not label_indices:
        return 0
    indices = tuple(sorted(int(value) for value in label_indices))
    zero_based = all(0 <= value < int(frame_count) for value in indices)
    one_based = all(1 <= value <= int(frame_count) for value in indices)
    if zero_based and one_based:
        raise LtaInputError(
            f"Sparse video label numbering for {stem} is ambiguous between zero- and "
            f"one-based indices: {list(indices)}. Include frame 0 or frame {frame_count} "
            "to make the origin explicit."
        )
    if not zero_based and not one_based:
        raise LtaInputError(
            f"Video label indices are out of range or mix index origins for {stem}: "
            f"frames={frame_count}, labels={list(indices)}"
        )
    return 0 if zero_based else 1


def _classify_coverage(label_count: int, frame_count: int) -> VolumeClass:
    if int(label_count) == 0:
        return VolumeClass.UNLABELED
    if int(label_count) == int(frame_count):
        return VolumeClass.FULLY_LABELED
    return VolumeClass.PARTIALLY_LABELED


def _build_annotations(
    *,
    encoded_indices: Sequence[int],
    labels_by_index: dict[int, Path],
) -> Tuple[FrameAnnotation, ...]:
    annotations = []
    for frame_position, encoded_index in enumerate(encoded_indices):
        label_path = labels_by_index.get(int(encoded_index))
        if label_path is None:
            annotations.append(
                FrameAnnotation(
                    encoded_index=int(encoded_index),
                    frame_position=int(frame_position),
                    state=AnnotationState.UNKNOWN,
                    label_path=None,
                    label_sha256=None,
                    polygons=(),
                )
            )
            continue
        label_sha256, polygons = parse_yolo_segmentation_label(label_path)
        annotations.append(
            FrameAnnotation(
                encoded_index=int(encoded_index),
                frame_position=int(frame_position),
                state=(
                    AnnotationState.FOREGROUND
                    if polygons
                    else AnnotationState.KNOWN_BACKGROUND
                ),
                label_path=label_path,
                label_sha256=label_sha256,
                polygons=polygons,
            )
        )
    return tuple(annotations)


def _discover_root(
    *,
    root: Path,
    role: SourceRole,
    selected_media: Optional[Path],
    video_probe: VideoProbe,
    image_probe: Optional[ImageProbe],
) -> Tuple[LtaVolumeSpec, ...]:
    if selected_media is None:
        files = tuple(sorted((path.resolve() for path in root.iterdir() if path.is_file()), key=lambda path: path.name))
        media_files = tuple(
            path
            for path in files
            if path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        )
        label_files = tuple(path for path in files if path.suffix.lower() == ".txt")
        unsupported_labels = tuple(
            path for path in files if path.suffix.lower() in UNSUPPORTED_LABEL_EXTENSIONS
        )
    else:
        media_files = (selected_media,)
        label_files = _selected_labels_for_media(selected_media)
        selected_stem, _selected_index = split_indexed_stem(selected_media)
        unsupported_labels = tuple(
            path.resolve()
            for path in root.iterdir()
            if path.is_file()
            and path.suffix.lower() in UNSUPPORTED_LABEL_EXTENSIONS
            and path.stem in {selected_media.stem, selected_stem}
        )
    if unsupported_labels:
        names = ", ".join(path.name for path in unsupported_labels[:12])
        raise LtaInputError(
            f"LTA accepts class-0 YOLO segmentation labels, not NRRD/NHDR: {names}"
        )
    if not media_files:
        raise LtaInputError(f"No supported image or video media found in {root}")

    indexed_images: dict[str, dict[int, Path]] = {}
    singleton_images: dict[str, Path] = {}
    videos: dict[str, Path] = {}
    for path in media_files:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            stem, index = split_indexed_stem(path)
            if index is None:
                existing = singleton_images.get(stem)
                if existing is not None:
                    raise LtaInputError(
                        f"Duplicate singleton image volume {stem}: "
                        f"{existing.name}, {path.name}"
                    )
                singleton_images[stem] = path
            else:
                _add_unique_indexed(
                    indexed_images,
                    stem=stem,
                    index=index,
                    path=path,
                    kind="image",
                )
        elif suffix in VIDEO_EXTENSIONS:
            existing = videos.get(path.stem)
            if existing is not None:
                raise LtaInputError(
                    f"Multiple video encodings found for volume {path.stem}: "
                    f"{existing.name}, {path.name}"
                )
            videos[path.stem] = path

    indexed_labels: dict[str, dict[int, Path]] = {}
    singleton_labels: dict[str, Path] = {}
    for path in label_files:
        stem, index = split_indexed_stem(path)
        if index is None:
            existing = singleton_labels.get(stem)
            if existing is not None:
                raise LtaInputError(
                    f"Duplicate unindexed label for {stem}: {existing.name}, {path.name}"
                )
            singleton_labels[stem] = path
        else:
            _add_unique_indexed(
                indexed_labels,
                stem=stem,
                index=index,
                path=path,
                kind="label",
            )

    media_stems = set(indexed_images) | set(singleton_images) | set(videos)
    for stem in sorted(media_stems):
        encodings = sum(
            int(stem in collection)
            for collection in (indexed_images, singleton_images, videos)
        )
        if encodings > 1:
            raise LtaInputError(
                f"Ambiguous media for volume {stem}: choose exactly one indexed image "
                "stack, singleton image, or video encoding"
            )
    orphan_indexed = sorted(set(indexed_labels) - media_stems)
    orphan_singleton = sorted(set(singleton_labels) - media_stems)
    if orphan_indexed or orphan_singleton:
        raise LtaInputError(
            "YOLO labels exist without matching media volume(s): "
            f"{(orphan_indexed + orphan_singleton)[:12]}"
        )
    if not media_stems:
        raise LtaInputError(f"No supported media volumes found in {root}")

    specs = []
    for stem in sorted(media_stems, key=lambda value: (value.casefold(), value)):
        volume_id = _volume_id(role, root, stem)
        if stem in singleton_images:
            if stem in indexed_labels:
                raise LtaInputError(
                    f"Indexed labels for singleton image volume {stem} have no matching frames"
                )
            labels_by_index = (
                {0: singleton_labels[stem]} if stem in singleton_labels else {}
            )
            image_path = singleton_images[stem]
            metadata = (
                _normalize_image_metadata(image_probe(image_path), image_path)
                if image_probe is not None
                else None
            )
            media = (
                IndexedMedia(
                    encoded_index=0,
                    frame_position=0,
                    path=image_path,
                    sha256=None,
                    identity_sha256=_file_identity_sha256(image_path),
                    width=(metadata.width if metadata is not None else None),
                    height=(metadata.height if metadata is not None else None),
                ),
            )
            encoded_indices = (0,)
            annotations = _build_annotations(
                encoded_indices=encoded_indices,
                labels_by_index=labels_by_index,
            )
            specs.append(
                LtaVolumeSpec(
                    source_role=role,
                    source_root=root,
                    volume_id=volume_id,
                    stem=stem,
                    kind="image",
                    media=media,
                    video_path=None,
                    video_sha256=None,
                    video_identity_sha256=None,
                    annotations=annotations,
                    volume_class=_classify_coverage(len(labels_by_index), 1),
                    encoded_indices=encoded_indices,
                    index_origin=0,
                    frame_count=1,
                    width=(metadata.width if metadata is not None else None),
                    height=(metadata.height if metadata is not None else None),
                    fps=None,
                )
            )
            continue

        if stem in indexed_images:
            if stem in singleton_labels:
                raise LtaInputError(
                    f"Unindexed label {singleton_labels[stem].name} is ambiguous for "
                    f"indexed image volume {stem}"
                )
            image_map = indexed_images[stem]
            labels_by_index = dict(indexed_labels.get(stem, {}))
            orphan_indices = sorted(set(labels_by_index) - set(image_map))
            if orphan_indices:
                raise LtaInputError(
                    f"Every YOLO label index must have a matching image for {stem}; "
                    f"orphan indices={orphan_indices}"
                )
            encoded_indices = tuple(sorted(image_map))
            media_items = []
            for frame_position, encoded_index in enumerate(encoded_indices):
                image_path = image_map[encoded_index]
                metadata = (
                    _normalize_image_metadata(image_probe(image_path), image_path)
                    if image_probe is not None
                    else None
                )
                media_items.append(
                    IndexedMedia(
                        encoded_index=int(encoded_index),
                        frame_position=int(frame_position),
                        path=image_path,
                        sha256=None,
                        identity_sha256=_file_identity_sha256(image_path),
                        width=(metadata.width if metadata is not None else None),
                        height=(metadata.height if metadata is not None else None),
                    )
                )
            annotations = _build_annotations(
                encoded_indices=encoded_indices,
                labels_by_index=labels_by_index,
            )
            dimensions = {
                (item.width, item.height)
                for item in media_items
                if item.width is not None and item.height is not None
            }
            if len(dimensions) > 1:
                detail = ", ".join(
                    f"{item.path.name}={item.width}x{item.height}"
                    for item in media_items
                )
                raise LtaInputError(
                    f"Indexed images in volume {stem} must have identical dimensions; "
                    f"{detail}"
                )
            specs.append(
                LtaVolumeSpec(
                    source_role=role,
                    source_root=root,
                    volume_id=volume_id,
                    stem=stem,
                    kind="sequence",
                    media=tuple(media_items),
                    video_path=None,
                    video_sha256=None,
                    video_identity_sha256=None,
                    annotations=annotations,
                    volume_class=_classify_coverage(
                        len(labels_by_index), len(encoded_indices)
                    ),
                    encoded_indices=encoded_indices,
                    index_origin=(
                        encoded_indices[0]
                        if encoded_indices
                        and encoded_indices
                        == tuple(range(encoded_indices[0], encoded_indices[0] + len(encoded_indices)))
                        and encoded_indices[0] in (0, 1)
                        else None
                    ),
                    frame_count=len(encoded_indices),
                    width=(media_items[0].width if media_items else None),
                    height=(media_items[0].height if media_items else None),
                    fps=None,
                )
            )
            continue

        if stem in singleton_labels:
            raise LtaInputError(
                f"Video labels must use a final '_NNNN' frame index: "
                f"{singleton_labels[stem].name}"
            )
        video_path = videos[stem]
        metadata = _normalize_video_metadata(video_probe(video_path), video_path)
        labels_by_index = dict(indexed_labels.get(stem, {}))
        origin = _resolve_video_origin(
            tuple(labels_by_index),
            frame_count=metadata.frame_count,
            stem=stem,
        )
        encoded_indices = tuple(
            range(int(origin), int(origin) + int(metadata.frame_count))
        )
        annotations = _build_annotations(
            encoded_indices=encoded_indices,
            labels_by_index=labels_by_index,
        )
        video_identity_sha256 = _file_identity_sha256(video_path)
        specs.append(
            LtaVolumeSpec(
                source_role=role,
                source_root=root,
                volume_id=volume_id,
                stem=stem,
                kind="video",
                media=(),
                video_path=video_path,
                # A whole-video content hash is deliberately deferred to the
                # manifest/decode layer. Discovery needs only a cheap mutation
                # identity; a container hash cannot identify one decoded frame.
                video_sha256=None,
                video_identity_sha256=video_identity_sha256,
                annotations=annotations,
                volume_class=_classify_coverage(
                    len(labels_by_index), metadata.frame_count
                ),
                encoded_indices=encoded_indices,
                index_origin=origin,
                frame_count=metadata.frame_count,
                width=metadata.width,
                height=metadata.height,
                fps=metadata.fps,
            )
        )
    return tuple(specs)


def _candidate_sort_key(candidate: PositiveExemplar) -> Tuple[object, ...]:
    return (
        0 if candidate.source_role is SourceRole.TARGET else 1,
        _canonical_path_key(candidate.source_root),
        candidate.volume_stem.casefold(),
        candidate.volume_stem,
        int(candidate.encoded_frame_index),
        int(candidate.label_row_index),
        candidate.exemplar_id,
    )


def _build_positive_pool(
    volumes: Iterable[LtaVolumeSpec],
) -> Tuple[PositiveExemplar, ...]:
    candidates = []
    image_hash_cache: dict[Path, str] = {}
    for volume in volumes:
        media_by_index = {item.encoded_index: item for item in volume.media}
        for annotation in volume.annotations:
            if not annotation.polygons:
                continue
            if annotation.label_path is None or annotation.label_sha256 is None:
                raise RuntimeError("Positive annotation is missing label provenance")
            if volume.kind == "video":
                if (
                    volume.video_path is None
                    or volume.video_identity_sha256 is None
                ):
                    raise RuntimeError("Positive video annotation is missing media provenance")
                media_path = volume.video_path
                media_sha256 = volume.video_sha256
                media_identity_sha256 = volume.video_identity_sha256
                source_width = volume.width
                source_height = volume.height
                frame_content_key = (
                    f"video:{volume.volume_id}:{media_identity_sha256}:"
                    f"{annotation.frame_position}"
                )
            else:
                media_item = media_by_index[annotation.encoded_index]
                media_path = media_item.path
                media_sha256 = image_hash_cache.get(media_path)
                if media_sha256 is None:
                    media_sha256 = _sha256_path(media_path)
                    image_hash_cache[media_path] = media_sha256
                media_identity_sha256 = media_item.identity_sha256
                source_width = media_item.width
                source_height = media_item.height
                frame_content_key = f"image:{media_sha256}"
            for polygon in annotation.polygons:
                canonical_polygon = ";".join(
                    f"{x.hex()},{y.hex()}" for x, y in polygon.points
                )
                canonical_box = ",".join(value.hex() for value in polygon.box_xyxy)
                # The conditioning unit is an image/frame plus its derived tight
                # positive box. Different source polygons that derive the same box
                # are deliberately one exemplar, while their full provenance remains
                # represented by the retained candidate's bundle hash.
                dedupe_material = f"{frame_content_key}\0{canonical_box}"
                exemplar_id = hashlib.sha256(
                    dedupe_material.encode("utf-8")
                ).hexdigest()
                media_bundle_identity = media_sha256 or media_identity_sha256
                bundle_material = (
                    f"{media_bundle_identity}\0{annotation.label_sha256}\0"
                    f"{annotation.encoded_index}\0{polygon.row_index}\0"
                    f"{canonical_polygon}"
                )
                bundle_sha256 = hashlib.sha256(
                    bundle_material.encode("utf-8")
                ).hexdigest()
                candidates.append(
                    PositiveExemplar(
                        exemplar_id=exemplar_id,
                        source_role=volume.source_role,
                        source_root=volume.source_root,
                        volume_id=volume.volume_id,
                        volume_stem=volume.stem,
                        volume_kind=volume.kind,
                        encoded_frame_index=annotation.encoded_index,
                        frame_position=annotation.frame_position,
                        media_path=media_path,
                        media_sha256=media_sha256,
                        media_identity_sha256=media_identity_sha256,
                        label_path=annotation.label_path,
                        label_sha256=annotation.label_sha256,
                        label_row_index=polygon.row_index,
                        class_id=polygon.class_id,
                        polygon=polygon.points,
                        box_xyxy=polygon.box_xyxy,
                        box_cxcywh=polygon.box_cxcywh,
                        normalized_area=polygon.normalized_area,
                        bundle_sha256=bundle_sha256,
                        source_width=source_width,
                        source_height=source_height,
                    )
                )
    ordered = sorted(candidates, key=_candidate_sort_key)
    deduplicated = []
    seen_ids = set()
    for candidate in ordered:
        if candidate.exemplar_id in seen_ids:
            continue
        seen_ids.add(candidate.exemplar_id)
        deduplicated.append(candidate)
    return tuple(deduplicated)


def rank_positive_exemplars_for_session(
    positive_pool: Sequence[PositiveExemplar],
    *,
    target_volume_id: str,
    directly_addressable_exemplar_ids: Sequence[str] = (),
    seed: int | str = 0,
) -> Tuple[RankedExemplar, ...]:
    """Rank positives deterministically for one target session.

    Same-session target polygons precede other polygons from the same target
    volume, then other target inputs, then external exemplar roots.
    """

    directly_addressable = frozenset(
        str(value) for value in directly_addressable_exemplar_ids
    )
    ranked = []
    for exemplar in positive_pool:
        if (
            exemplar.source_role is SourceRole.TARGET
            and exemplar.volume_id == target_volume_id
            and exemplar.exemplar_id in directly_addressable
        ):
            tier = ExemplarPreferenceTier.SAME_TARGET_SESSION
            reason = "geometry adapter marked this target polygon directly addressable"
        elif (
            exemplar.source_role is SourceRole.TARGET
            and exemplar.volume_id == target_volume_id
        ):
            tier = ExemplarPreferenceTier.SAME_TARGET_VOLUME
            reason = "target polygon belongs to the same target volume"
        elif exemplar.source_role is SourceRole.TARGET:
            tier = ExemplarPreferenceTier.OTHER_TARGET_VOLUME
            reason = "polygon comes from another target input volume"
        else:
            tier = ExemplarPreferenceTier.EXTERNAL_EXEMPLAR
            reason = "polygon comes from an external exemplar root"
        tie_break = hashlib.sha256(
            f"{seed}\0{target_volume_id}\0{exemplar.exemplar_id}".encode("utf-8")
        ).hexdigest()
        ranked.append(
            RankedExemplar(
                exemplar=exemplar,
                preference_tier=tier,
                preference_reason=reason,
                deterministic_tie_break=tie_break,
            )
        )
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                int(item.preference_tier),
                item.deterministic_tie_break,
                _candidate_sort_key(item.exemplar),
            ),
        )
    )


def discover_lta_inputs(
    input_arg: str | os.PathLike[str],
    exemplar_roots: Sequence[str | os.PathLike[str]] = (),
    *,
    video_probe: Optional[VideoProbe] = None,
    image_probe: Optional[ImageProbe] = None,
    require_positive: bool = True,
) -> LtaInputDiscovery:
    """Discover target and optional exemplar volumes for one LTA run."""

    input_path = Path(input_arg).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    selected_media: Optional[Path]
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise LtaInputError(f"Unsupported LTA input media file: {input_path}")
        selected_media = input_path
        target_root = input_path.parent
    elif input_path.is_dir():
        selected_media = None
        target_root = input_path
    else:
        raise LtaInputError(f"LTA input is neither a regular file nor directory: {input_path}")

    if isinstance(exemplar_roots, (str, os.PathLike)):
        exemplar_roots = (exemplar_roots,)

    resolved_exemplar_roots = []
    seen_roots = set()
    for raw_root in exemplar_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        if not root.is_dir():
            raise LtaInputError(f"--exemplar must be a directory: {root}")
        key = _canonical_path_key(root)
        if key in seen_roots:
            raise LtaInputError(f"Duplicate exemplar root: {root}")
        if key == _canonical_path_key(target_root):
            raise LtaInputError(
                f"Target and exemplar roots must be distinct to preserve provenance: {root}"
            )
        seen_roots.add(key)
        resolved_exemplar_roots.append(root)
    resolved_exemplar_roots.sort(key=_canonical_path_key)

    resolved_video_probe = video_probe or probe_video_with_ffprobe
    target_volumes = _discover_root(
        root=target_root,
        role=SourceRole.TARGET,
        selected_media=selected_media,
        video_probe=resolved_video_probe,
        image_probe=image_probe,
    )
    exemplar_volumes = tuple(
        volume
        for root in resolved_exemplar_roots
        for volume in _discover_root(
            root=root,
            role=SourceRole.EXEMPLAR,
            selected_media=None,
            video_probe=resolved_video_probe,
            image_probe=image_probe,
        )
    )
    positive_pool = _build_positive_pool(target_volumes + exemplar_volumes)
    if require_positive and not positive_pool:
        raise NoPositiveExemplarError(
            "LTA requires at least one valid class-0 YOLO segmentation polygon "
            "across target inputs and exemplar roots"
        )

    warnings = []
    for volume in target_volumes:
        if volume.volume_class is VolumeClass.FULLY_LABELED:
            warnings.append(
                LtaDiscoveryWarning(
                    code="fully_labeled_target",
                    message=(
                        f"{volume.stem} has explicit labels for every frame; LTA will "
                        "continue for comparison/regeneration"
                    ),
                    volume_id=volume.volume_id,
                )
            )
        elif volume.volume_class is VolumeClass.PARTIALLY_LABELED:
            warnings.append(
                LtaDiscoveryWarning(
                    code="partially_labeled_target",
                    message=(
                        f"{volume.stem} has sparse labels; missing frame labels remain unknown"
                    ),
                    volume_id=volume.volume_id,
                )
            )

    return LtaInputDiscovery(
        input_path=input_path,
        target_volumes=target_volumes,
        exemplar_roots=tuple(resolved_exemplar_roots),
        exemplar_volumes=exemplar_volumes,
        positive_pool=positive_pool,
        warnings=tuple(warnings),
    )


def revalidate_lta_input_identities(discovery: LtaInputDiscovery) -> None:
    """Fail if media/labels selected during discovery changed before publication."""

    if not isinstance(discovery, LtaInputDiscovery):
        raise TypeError("discovery must be an LtaInputDiscovery")
    for volume in discovery.all_volumes:
        if volume.video_path is not None:
            current = _file_identity_sha256(volume.video_path)
            if current != volume.video_identity_sha256:
                raise RuntimeError(f"LTA video changed during the run: {volume.video_path}")
        for media in volume.media:
            current = _file_identity_sha256(media.path)
            if current != media.identity_sha256:
                raise RuntimeError(f"LTA image changed during the run: {media.path}")
        for annotation in volume.annotations:
            if annotation.label_path is None:
                continue
            current = _sha256_path(annotation.label_path)
            if current != annotation.label_sha256:
                raise RuntimeError(
                    f"LTA YOLO label changed during the run: {annotation.label_path}"
                )
    for exemplar in discovery.positive_pool:
        if exemplar.media_sha256 is None:
            continue
        current = _sha256_path(exemplar.media_path)
        if current != exemplar.media_sha256:
            raise RuntimeError(
                f"LTA selected exemplar image changed during the run: {exemplar.media_path}"
            )


__all__ = (
    "AnnotationState",
    "ExemplarPreferenceTier",
    "FrameAnnotation",
    "IMAGE_EXTENSIONS",
    "ImageProbe",
    "ImageMetadata",
    "IndexedMedia",
    "LtaDiscoveryWarning",
    "LtaInputDiscovery",
    "LtaInputError",
    "LtaVolumeSpec",
    "NoPositiveExemplarError",
    "PositiveExemplar",
    "RankedExemplar",
    "SourceRole",
    "VIDEO_EXTENSIONS",
    "VideoProbe",
    "VideoMetadata",
    "VolumeClass",
    "YoloPolygon",
    "discover_lta_inputs",
    "parse_yolo_segmentation_label",
    "probe_video_with_ffprobe",
    "rank_positive_exemplars_for_session",
    "revalidate_lta_input_identities",
    "split_indexed_stem",
)
