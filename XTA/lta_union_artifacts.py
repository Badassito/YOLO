"""Bounded sparse LTA union storage with a verified legacy raw reader.

Only nonempty, tightly cropped frames are appended. Each row is independently
packed least-significant-bit first; absent logical frames are exactly zero.
The file digest covers stored bytes, never the potentially much larger logical
volume. Production consumers decode one crop at a time after full validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import operator
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

UNION_SCHEMA = "lta.union/2"
UNION_ENCODING = "cropped-row-packbits-little"
MAX_CHUNK_FRAMES = 30


def _index(value, name, *, minimum=0):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _shape(value, name="shape", length=3):
    result = tuple(_index(v, name, minimum=1) for v in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} positive dimensions")
    return result


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("union sha256 must be a lowercase hexadecimal SHA256")
    return value


def _file_digest(path, *, binary=False):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if binary and bool(np.any(np.frombuffer(block, dtype=np.uint8) > 1)):
                raise ValueError("legacy union contains nonbinary values")
            digest.update(block)
    return digest.hexdigest()


class LtaUnionWriter:
    """Append disjoint owned chunks of at most thirty binary tile frames."""

    def __init__(self, path: str | Path, *, shape: Sequence[int], frame_start: int):
        self.path = Path(path).resolve(strict=False)
        self.shape = _shape(shape)
        self.frame_start = _index(frame_start, "frame_start")
        self.stage = self.path.with_name("." + self.path.name + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self._stream = self.stage.open("xb")
        self._digest = hashlib.sha256()
        self._frames = []
        self._ranges = []
        self._size = 0
        self._foreground = 0
        self._finished = False

    def append_chunk(self, frame_start: int, array) -> None:
        if self._stream.closed:
            raise RuntimeError("union writer is closed")
        start = _index(frame_start, "chunk frame_start")
        values = np.asarray(array)
        if values.ndim != 3 or not 1 <= values.shape[0] <= MAX_CHUNK_FRAMES:
            raise ValueError("union chunks must contain one to thirty TYX frames")
        if tuple(values.shape[1:]) != self.shape[1:]:
            raise ValueError("union chunk tile shape differs from its logical shape")
        if values.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
            raise ValueError("union chunks must use uint8 or bool binary masks")
        if values.dtype == np.uint8 and bool(np.any(values > 1)):
            raise ValueError("union chunks must contain only zero and one")
        stop = start + int(values.shape[0])
        if not self.frame_start <= start < stop <= self.frame_start + self.shape[0]:
            raise ValueError("union chunk lies outside its logical frame range")
        if any(start < prior_stop and prior_start < stop for prior_start, prior_stop in self._ranges):
            raise ValueError("union chunks must own disjoint frame ranges")
        try:
            for ordinal, frame in enumerate(values):
                pixels = int(np.count_nonzero(frame))
                if not pixels:
                    continue
                ys = np.flatnonzero(np.any(frame, axis=1))
                xs = np.flatnonzero(np.any(frame, axis=0))
                top, bottom = int(ys[0]), int(ys[-1]) + 1
                left, right = int(xs[0]), int(xs[-1]) + 1
                crop = frame[top:bottom, left:right]
                packed = np.packbits(crop, axis=1, bitorder="little")
                data = packed.tobytes(order="C")
                if self._stream.write(data) != len(data):
                    raise OSError("incomplete sparse union payload write")
                self._digest.update(data)
                self._frames.append({
                    "frame_index": start + ordinal,
                    "offset_bytes": self._size,
                    "size_bytes": len(data),
                    "crop_yx": [top, left],
                    "shape_yx": [bottom - top, right - left],
                    "row_stride_bytes": int(packed.shape[1]),
                    "foreground_pixels": pixels,
                })
                self._size += len(data)
                self._foreground += pixels
            self._ranges.append((start, stop))
        except BaseException:
            self.abort()
            raise

    def finish(self) -> dict[str, object]:
        if self._stream.closed:
            raise RuntimeError("union writer is closed")
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            os.replace(self.stage, self.path)
            self._finished = True
        except BaseException:
            self.abort()
            raise
        return {"schema": UNION_SCHEMA, "encoding": UNION_ENCODING,
                "path": str(self.path), "sha256": self._digest.hexdigest(),
                "size_bytes": self._size, "dtype": "uint8", "shape": list(self.shape),
                "frame_start": self.frame_start, "foreground_pixels": self._foreground,
                "omitted_frames": "zero", "frames": [dict(frame) for frame in self._frames]}

    def abort(self) -> None:
        if not self._stream.closed:
            self._stream.close()
        if not self._finished:
            self.stage.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self._finished:
            self.abort()


@dataclass(frozen=True)
class _Frame:
    frame_index: int
    offset: int
    size: int
    top: int
    left: int
    height: int
    width: int
    stride: int
    foreground: int


@dataclass(frozen=True)
class ValidatedUnionArtifact:
    path: Path
    shape: tuple[int, int, int]
    frame_start: int
    encoding: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    frames: tuple[_Frame, ...]
    foreground_pixels: int | None

    def revalidate(self):
        stat = self.path.stat()
        if stat.st_size != self.size_bytes or stat.st_mtime_ns != self.mtime_ns:
            raise RuntimeError("union artifact changed after verification")


def _decode(data: bytes, frame: _Frame):
    packed = np.frombuffer(data, dtype=np.uint8).reshape(frame.height, frame.stride)
    remainder = frame.width % 8
    if remainder and bool(np.any(packed[:, -1] & (255 ^ ((1 << remainder) - 1)))):
        raise ValueError("sparse union row padding must be zero")
    decoded = np.unpackbits(packed, axis=1, count=frame.width, bitorder="little")
    if int(np.count_nonzero(decoded)) != frame.foreground:
        raise ValueError("sparse union foreground count differs from its payload")
    if not all(bool(np.any(edge)) for edge in (decoded[0], decoded[-1], decoded[:, 0], decoded[:, -1])):
        raise ValueError("sparse union crop must tightly bound its foreground")
    return decoded


def validate_union_artifact(descriptor: Mapping[str, object] | ValidatedUnionArtifact, *, frame_start=None):
    """Verify complete metadata and payload before a consumer mutates its view."""
    if isinstance(descriptor, ValidatedUnionArtifact):
        descriptor.revalidate()
        if frame_start is not None and descriptor.frame_start != _index(frame_start, "frame_start"):
            raise ValueError("union frame start differs from its manifest")
        return descriptor
    if not isinstance(descriptor, Mapping):
        raise TypeError("union descriptor must be a mapping")
    shape = _shape(descriptor["shape"])
    path = Path(str(descriptor["path"])).resolve(strict=True)
    if not path.is_file():
        raise ValueError("union artifact must be a regular file")
    stat = path.stat()
    digest = _digest(descriptor["sha256"])
    if descriptor.get("dtype", "uint8") != "uint8":
        raise ValueError("union logical dtype must be uint8")
    schema = descriptor.get("schema")
    if schema in (None, "lta.union/1"):
        if descriptor.get("encoding") not in (None, "raw-uint8"):
            raise ValueError("unsupported legacy union encoding")
        start = _index(0 if frame_start is None else frame_start, "frame_start")
        size = math.prod(shape)
        if stat.st_size != size or _index(descriptor.get("size_bytes", size), "size_bytes") != size:
            raise ValueError("legacy union byte length differs from logical shape")
        if _file_digest(path, binary=True) != digest:
            raise RuntimeError("union artifact SHA256 differs from its descriptor")
        result = ValidatedUnionArtifact(path, shape, start, "raw-uint8", digest, size, stat.st_mtime_ns, (), None)
    else:
        if schema != UNION_SCHEMA or descriptor.get("encoding") != UNION_ENCODING:
            raise ValueError("unsupported sparse union schema or encoding")
        if descriptor.get("omitted_frames") != "zero":
            raise ValueError("sparse union must declare omitted frames as zero")
        start = _index(descriptor["frame_start"], "frame_start")
        if frame_start is not None and start != _index(frame_start, "frame_start"):
            raise ValueError("union frame start differs from its manifest")
        size = _index(descriptor["size_bytes"], "size_bytes")
        if stat.st_size != size:
            raise ValueError("sparse union byte length differs from its descriptor")
        foreground = _index(descriptor["foreground_pixels"], "foreground_pixels")
        frames = []
        seen = set()
        expected_offset = 0
        raw_frames = descriptor["frames"]
        if not isinstance(raw_frames, (tuple, list)):
            raise TypeError("sparse union frames must be a list")
        for record in raw_frames:
            if not isinstance(record, Mapping):
                raise TypeError("sparse union frame must be a mapping")
            index = _index(record["frame_index"], "frame_index")
            if not start <= index < start + shape[0] or index in seen:
                raise ValueError("sparse union frame is duplicate or outside logical range")
            seen.add(index)
            crop = tuple(_index(v, "crop_yx") for v in record["crop_yx"])
            if len(crop) != 2:
                raise ValueError("crop_yx must contain top and left")
            height, width = _shape(record["shape_yx"], "shape_yx", 2)
            stride = _index(record["row_stride_bytes"], "row_stride_bytes", minimum=1)
            length = _index(record["size_bytes"], "frame size_bytes", minimum=1)
            offset = _index(record["offset_bytes"], "offset_bytes")
            pixels = _index(record["foreground_pixels"], "frame foreground_pixels", minimum=1)
            if crop[0] + height > shape[1] or crop[1] + width > shape[2]:
                raise ValueError("sparse union crop lies outside its logical tile")
            if stride != (width + 7) // 8 or length != height * stride or pixels > height * width:
                raise ValueError("sparse union packed dimensions or foreground count are invalid")
            if offset != expected_offset or offset + length > size:
                raise ValueError("sparse union payload offsets must be contiguous and in bounds")
            expected_offset += length
            frames.append(_Frame(index, offset, length, *crop, height, width, stride, pixels))
        if expected_offset != size or sum(frame.foreground for frame in frames) != foreground:
            raise ValueError("sparse union total size or foreground count is inconsistent")
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            for frame in frames:
                data = stream.read(frame.size)
                if len(data) != frame.size:
                    raise RuntimeError("sparse union payload ended before its indexed extent")
                actual.update(data)
                _decode(data, frame)
        if actual.hexdigest() != digest:
            raise RuntimeError("union artifact SHA256 differs from its descriptor")
        result = ValidatedUnionArtifact(path, shape, start, UNION_ENCODING, digest, size,
                                        stat.st_mtime_ns, tuple(frames), foreground)
    result.revalidate()
    return result


def iter_union_crops(descriptor, *, frame_start=None):
    """Yield global frame, tile-local (top,left), and one decoded nonempty crop."""
    artifact = validate_union_artifact(descriptor, frame_start=frame_start)
    if artifact.encoding == "raw-uint8":
        source = np.memmap(artifact.path, dtype=np.uint8, mode="r", shape=artifact.shape)
        try:
            for index in range(artifact.shape[0]):
                frame = source[index]
                if bool(np.any(frame > 1)):
                    raise ValueError("legacy union contains nonbinary values")
                if not bool(np.any(frame)):
                    continue
                ys = np.flatnonzero(np.any(frame, axis=1)); xs = np.flatnonzero(np.any(frame, axis=0))
                top, left = int(ys[0]), int(xs[0])
                yield artifact.frame_start + index, (top, left), np.asarray(frame[top:int(ys[-1])+1,left:int(xs[-1])+1]).copy()
        finally:
            source._mmap.close()
    else:
        with artifact.path.open("rb") as stream:
            for frame in artifact.frames:
                stream.seek(frame.offset)
                yield frame.frame_index, (frame.top, frame.left), _decode(stream.read(frame.size), frame)


def read_union_array(descriptor, *, frame_start=None, max_bytes=64 * 1024**2):
    """Diagnostic reconstruction with a strict allocation cap; not a production path."""
    artifact = validate_union_artifact(descriptor, frame_start=frame_start)
    if math.prod(artifact.shape) > _index(max_bytes, "max_bytes"):
        raise ValueError("logical union exceeds diagnostic reconstruction byte cap")
    output = np.zeros(artifact.shape, dtype=np.uint8)
    for frame, (top, left), crop in iter_union_crops(artifact):
        output[frame-artifact.frame_start,top:top+crop.shape[0],left:left+crop.shape[1]] |= crop
    return output


def logical_union_sha256(descriptor, *, frame_start=None):
    """Hash logical uint8 masks in TYX order for cross-format diagnostic parity."""
    artifact = validate_union_artifact(descriptor, frame_start=frame_start)
    if artifact.encoding == "raw-uint8":
        return artifact.sha256
    by_frame = {frame.frame_index: frame for frame in artifact.frames}
    digest = hashlib.sha256()
    plane = np.zeros(artifact.shape[1:], dtype=np.uint8)
    with artifact.path.open("rb") as stream:
        for index in range(artifact.frame_start, artifact.frame_start + artifact.shape[0]):
            plane.fill(0)
            frame = by_frame.get(index)
            if frame is not None:
                stream.seek(frame.offset)
                plane[frame.top:frame.top+frame.height,frame.left:frame.left+frame.width] = _decode(stream.read(frame.size), frame)
            digest.update(memoryview(plane))
    return digest.hexdigest()


def reduce_union_artifact_into_view(descriptor, *, view_union, tile_xyxy, frame_start, frame_stop=None):
    """Verify first, then OR only stored support into the private native view."""
    artifact = validate_union_artifact(descriptor, frame_start=frame_start)
    tile = tuple(_index(v, "tile_xyxy") for v in tile_xyxy)
    if len(tile) != 4:
        raise ValueError("tile_xyxy must contain four coordinates")
    x0,y0,x1,y1 = tile
    stop = artifact.frame_start + artifact.shape[0]
    if frame_stop is not None and _index(frame_stop, "frame_stop") != stop:
        raise ValueError("union frame stop differs from its manifest")
    if (getattr(view_union,"ndim",None) != 3 or view_union.dtype != np.uint8
            or not view_union.flags.writeable):
        raise ValueError("destination view must be a writable uint8 TYX array")
    if (artifact.shape[1:] != (y1-y0,x1-x0) or not 0<=x0<x1<=view_union.shape[2]
            or not 0<=y0<y1<=view_union.shape[1] or not 0<=artifact.frame_start<stop<=view_union.shape[0]):
        raise ValueError("union geometry lies outside its destination view")
    if artifact.encoding == "raw-uint8":
        from .lta_rendering import union_tile_chunk_into_view
        source = np.memmap(artifact.path,dtype=np.uint8,mode="r",shape=artifact.shape)
        try:
            union_tile_chunk_into_view(view_union,source,frame_start=artifact.frame_start,tile_xyxy=tile)
        finally:
            source._mmap.close()
    else:
        for frame,(top,left),crop in iter_union_crops(artifact):
            target=view_union[frame,y0+top:y0+top+crop.shape[0],x0+left:x0+left+crop.shape[1]]
            np.bitwise_or(target,crop,out=target)
    return {"path":str(artifact.path),"encoding":artifact.encoding,
            "logical_bytes":math.prod(artifact.shape),"stored_bytes":artifact.size_bytes,
            "indexed_frame_count":len(artifact.frames) if artifact.encoding==UNION_ENCODING else None,
            "foreground_pixels":artifact.foreground_pixels}


__all__ = ("LtaUnionWriter", "ValidatedUnionArtifact", "validate_union_artifact",
           "iter_union_crops", "read_union_array", "logical_union_sha256",
           "reduce_union_artifact_into_view", "UNION_SCHEMA", "UNION_ENCODING")
