"""Exact same-lineage mask and directional coverage for bounded LTA handoffs.

Packets contain cropped bitmaps and directed observed-frame intervals. The
disk-backed ledger stores adjacent-frame edges, so two observations that only
touch in frame space cannot invent a previously untraversed transition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import operator
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable, Mapping

import numpy as np

from .lta_tile_tracking import LtaLineageId


COVERAGE_SCHEMA = "lta.lineage-coverage/1"
_ENCODING = "cropped-flat-packbits-little"
_DIRECTIONS = {"forward": 1, "backward": -1}


def _index(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(operator.index(value))
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _shape(value: Iterable[int]) -> tuple[int, int]:
    shape = tuple(_index(item, "tile dimension", minimum=1) for item in value)
    if len(shape) != 2:
        raise ValueError("tile shape must contain exactly two positive dimensions")
    return shape


def _lineage(value: object) -> LtaLineageId:
    if not isinstance(value, LtaLineageId):
        raise TypeError("coverage requires an LtaLineageId")
    return value


def _binary(mask: object, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or any(value < 1 for value in array.shape):
        raise ValueError("coverage masks must have positive two-dimensional geometry")
    if shape is not None and tuple(array.shape) != shape:
        raise ValueError(f"coverage mask shape {array.shape} differs from tile shape {shape}")
    if array.dtype == np.bool_:
        return array
    if array.dtype != np.uint8 or bool(np.any(array > 1)):
        raise ValueError("coverage masks must be exact bool or binary uint8 masks")
    return array.view(np.bool_)


def _crop(mask: np.ndarray) -> tuple[int, int, int, int, bytes, int] | None:
    rows = np.flatnonzero(np.any(mask, axis=1))
    if not len(rows):
        return None
    columns = np.flatnonzero(np.any(mask, axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    crop = mask[top:bottom, left:right]
    return top, left, bottom - top, right - left, np.packbits(crop.reshape(-1), bitorder="little").tobytes(), int(np.count_nonzero(crop))


def _decode(crop: tuple[int, int, int, int, bytes, int]) -> np.ndarray:
    return np.unpackbits(np.frombuffer(crop[4], dtype=np.uint8), count=crop[2] * crop[3], bitorder="little").reshape(crop[2], crop[3]).view(np.bool_)


def _union_crops(left, right):
    top, column = min(left[0], right[0]), min(left[1], right[1])
    bottom = max(left[0] + left[2], right[0] + right[2])
    edge = max(left[1] + left[3], right[1] + right[3])
    mask = np.zeros((bottom - top, edge - column), dtype=np.bool_)
    for crop in (left, right):
        y, x = crop[0] - top, crop[1] - column
        mask[y:y + crop[2], x:x + crop[3]] |= _decode(crop)
    return top, column, bottom - top, edge - column, np.packbits(mask.reshape(-1), bitorder="little").tobytes(), int(np.count_nonzero(mask))


def _merge_ranges(ranges, *, touching: bool):
    merged = []
    for start, stop in sorted(ranges):
        joins = bool(merged) and (start <= merged[-1][1] if touching else start < merged[-1][1])
        if joins:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LtaCoverageBuilder:
    """One worker packet, retaining only nonempty cropped frame masks."""

    def __init__(self, tile_shape_hw: Iterable[int]):
        self.tile_shape_hw = _shape(tile_shape_hw)
        self._lineages: dict[str, LtaLineageId] = {}
        self._observed: dict[tuple[str, int], list[tuple[int, int]]] = {}
        self._masks: dict[tuple[str, int], tuple[int, int, int, int, bytes, int]] = {}
        self._empty_predictions = 0

    def _register(self, lineage: LtaLineageId) -> str:
        lineage = _lineage(lineage)
        token = lineage.token
        previous = self._lineages.get(token)
        if previous is not None and previous != lineage:
            raise ValueError("coverage lineage token collides with different metadata")
        self._lineages[token] = lineage
        return token

    def mark_observed(self, lineages, *, frame_start, frame_stop, prompt_frame, direction) -> None:
        start = _index(frame_start, "frame_start")
        stop = _index(frame_stop, "frame_stop", minimum=1)
        prompt = _index(prompt_frame, "prompt_frame")
        if not start <= prompt < stop:
            raise ValueError("coverage prompt must lie inside its observed frame range")
        if direction not in {"forward", "backward", "both"}:
            raise ValueError("coverage direction must be forward, backward, or both")
        values = tuple(_lineage(item) for item in lineages)
        if not values:
            raise ValueError("observed coverage requires at least one lineage")
        ranges = []
        if direction in {"forward", "both"}:
            ranges.append((1, prompt, stop))
        if direction in {"backward", "both"}:
            ranges.append((-1, start, prompt + 1))
        for lineage in values:
            token = self._register(lineage)
            for code, first, last in ranges:
                key = token, code
                # Frame intervals must overlap, not merely touch, before their
                # transition evidence can be combined.
                self._observed[key] = _merge_ranges((*self._observed.get(key, ()), (first, last)), touching=False)

    def add_prediction(self, lineage, frame_index, mask) -> None:
        frame = _index(frame_index, "frame_index")
        binary = _binary(mask, self.tile_shape_hw)
        token = self._register(lineage)
        incoming = _crop(binary)
        if incoming is None:
            self._empty_predictions += 1
            return
        key = token, frame
        self._masks[key] = incoming if key not in self._masks else _union_crops(self._masks[key], incoming)

    def stats(self) -> dict[str, int]:
        return {
            "lineage_count": len(self._lineages), "mask_count": len(self._masks),
            "directed_interval_count": sum(map(len, self._observed.values())),
            "stored_mask_bytes": sum(len(crop[4]) for crop in self._masks.values()),
            "empty_predictions": self._empty_predictions,
        }

    def write(self, output_dir, *, work_id, tile_index, tile_config_id) -> dict[str, object]:
        work = str(work_id).strip()
        config = str(tile_config_id).strip()
        tile = _index(tile_index, "tile_index")
        if not work or not config:
            raise ValueError("coverage work_id and tile_config_id must not be empty")
        if not self._lineages:
            raise ValueError("coverage packet has no lineages")
        if any(item.tile_config_id != config for item in self._lineages.values()):
            raise ValueError("coverage lineage tile configuration differs from the packet")
        ordered = sorted(self._lineages)
        by_token = {token: index for index, token in enumerate(ordered)}
        arrays = {}
        mask_rows = []
        for ordinal, ((token, frame), crop) in enumerate(sorted(self._masks.items())):
            y, x, height, width, packed, foreground = crop
            mask_rows.append((by_token[token], frame, y, x, height, width, foreground))
            arrays[f"mask_{ordinal:08d}"] = np.frombuffer(packed, dtype=np.uint8)
        observed = [
            (by_token[token], direction, start, stop)
            for (token, direction), ranges in sorted(self._observed.items())
            for start, stop in ranges
        ]
        metadata = {
            "schema": COVERAGE_SCHEMA, "encoding": _ENCODING,
            "work_id": work, "tile_index": tile, "tile_config_id": config,
            "tile_shape_hw": list(self.tile_shape_hw),
            "lineages": [asdict(self._lineages[token]) for token in ordered],
            "interval_semantics": "directed observed frame intervals; half-open",
        }
        arrays["masks"] = np.asarray(mask_rows, dtype=np.int64).reshape(-1, 7)
        arrays["observed"] = np.asarray(observed, dtype=np.int64).reshape(-1, 4)
        arrays["metadata"] = np.frombuffer(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"), dtype=np.uint8)
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / "lineage_coverage.npz"
        descriptor, name = tempfile.mkstemp(prefix=".lineage_coverage.", suffix=".partial", dir=folder)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return {
            "schema": COVERAGE_SCHEMA, "path": str(destination.resolve()),
            "sha256": _sha256(destination), "work_id": work,
            "tile_index": tile, "tile_config_id": config,
            "tile_shape_hw": list(self.tile_shape_hw), **self.stats(),
            "file_size_bytes": destination.stat().st_size,
        }


def _packet_crop(archive, row, ordinal, shape):
    lineage_index, frame, y, x, height, width, foreground = (int(value) for value in row)
    if y < 0 or x < 0 or height < 1 or width < 1 or y + height > shape[0] or x + width > shape[1]:
        raise ValueError("coverage crop lies outside its tile geometry")
    packed = archive[f"mask_{ordinal:08d}"]
    area = height * width
    if packed.dtype != np.uint8 or packed.ndim != 1 or len(packed) != (area + 7) // 8:
        raise ValueError("coverage crop packed storage is inconsistent")
    if area % 8 and int(packed[-1]) >> (area % 8):
        raise ValueError("coverage crop has nonzero padding bits")
    crop = y, x, height, width, packed.tobytes(), foreground
    mask = _decode(crop)
    if foreground <= 0 or int(np.count_nonzero(mask)) != foreground:
        raise ValueError("coverage crop foreground count is inconsistent")
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        raise ValueError("coverage crop is not tightly bounded")
    return crop


class LtaCoverageLedger:
    """SQLite coverage index with a bounded page cache and per-mask decoding."""

    def __init__(self, path, *, frame_count, _read_only=False):
        self.path = Path(path).resolve(strict=False)
        self.frame_count = _index(frame_count, "frame_count", minimum=1)
        self._read_only = bool(_read_only)
        self._closed = False
        self._audit = Counter()
        if self._read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self._db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path))
        try:
            self._db.execute("PRAGMA cache_size=-4096")
            self._db.execute("PRAGMA temp_store=FILE")
            tables = {row[0] for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and "coverage_meta" not in tables:
                raise ValueError("coverage path contains an unrelated SQLite database")
            if not tables:
                if self._read_only:
                    raise ValueError("coverage snapshot has no schema")
                self._db.executescript("""
                    CREATE TABLE coverage_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE packets (work_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
                    CREATE TABLE lineages (token TEXT PRIMARY KEY, metadata TEXT NOT NULL);
                    CREATE TABLE tiles (config TEXT NOT NULL, tile INTEGER NOT NULL,
                        height INTEGER NOT NULL, width INTEGER NOT NULL, PRIMARY KEY(config,tile));
                    CREATE TABLE masks (token TEXT NOT NULL, config TEXT NOT NULL, tile INTEGER NOT NULL,
                        frame INTEGER NOT NULL, y INTEGER NOT NULL, x INTEGER NOT NULL,
                        height INTEGER NOT NULL, width INTEGER NOT NULL, packed BLOB NOT NULL,
                        foreground INTEGER NOT NULL, PRIMARY KEY(token,config,tile,frame));
                    CREATE TABLE edges (token TEXT NOT NULL, config TEXT NOT NULL, tile INTEGER NOT NULL,
                        direction INTEGER NOT NULL, start INTEGER NOT NULL, stop INTEGER NOT NULL,
                        PRIMARY KEY(token,config,tile,direction,start));
                """)
                self._db.executemany("INSERT INTO coverage_meta VALUES (?,?)", (("schema", COVERAGE_SCHEMA), ("frame_count", str(self.frame_count))))
                self._db.commit()
            metadata = dict(self._db.execute("SELECT key,value FROM coverage_meta"))
            if metadata != {"schema": COVERAGE_SCHEMA, "frame_count": str(self.frame_count)}:
                raise ValueError("coverage database schema/frame count differs from this run")
        except BaseException:
            self._db.close()
            self._closed = True
            raise

    @property
    def closed(self):
        return self._closed

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("coverage ledger is closed")

    def close(self):
        if not self._closed:
            self._db.close()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_args):
        self.close()

    def snapshot(self, path):
        self._ensure_open()
        destination = Path(path).resolve(strict=False)
        if destination == self.path or destination.exists():
            raise FileExistsError("coverage snapshot destination must be new")
        if self._db.in_transaction:
            raise RuntimeError("coverage snapshot requires committed packet ingestion")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(destination))
        try:
            self._db.backup(target, pages=128)
            target.close()
            return LtaCoverageLedger(destination, frame_count=self.frame_count, _read_only=True)
        except BaseException:
            target.close()
            destination.unlink(missing_ok=True)
            raise

    def stats(self):
        self._ensure_open()
        masks, stored, foreground = self._db.execute("SELECT COUNT(*),COALESCE(SUM(length(packed)),0),COALESCE(SUM(foreground),0) FROM masks").fetchone()
        ranges, transitions = self._db.execute("SELECT COUNT(*),COALESCE(SUM(stop-start),0) FROM edges").fetchone()
        return {
            "packets": self._db.execute("SELECT COUNT(*) FROM packets").fetchone()[0],
            "lineages": self._db.execute("SELECT COUNT(*) FROM lineages").fetchone()[0],
            "masked_frames": masks, "stored_bytes": stored, "foreground_pixels": foreground,
            "directional_ranges": ranges, "covered_directional_transitions": transitions,
            "database_bytes": self.path.stat().st_size, "page_cache_limit_bytes": 4 * 1024 * 1024,
            "read_only": self._read_only, **dict(self._audit),
        }

    def ingest(self, receipt, *, expected_work_id, tile_index, tile_config_id,
               frame_start, frame_stop, expected_lineages=None):
        self._ensure_open()
        if self._read_only:
            raise RuntimeError("coverage snapshots are read-only")
        try:
            return self._ingest(receipt, expected_work_id=expected_work_id, tile_index=tile_index,
                                tile_config_id=tile_config_id, frame_start=frame_start,
                                frame_stop=frame_stop, expected_lineages=expected_lineages)
        except BaseException:
            self._audit["rejected_packets"] += 1
            raise

    def _ingest(self, receipt, *, expected_work_id, tile_index, tile_config_id,
                frame_start, frame_stop, expected_lineages):
        if not isinstance(receipt, Mapping) or receipt.get("schema") != COVERAGE_SCHEMA:
            raise ValueError("unsupported coverage packet receipt")
        work, config = str(expected_work_id).strip(), str(tile_config_id).strip()
        tile = _index(tile_index, "tile_index")
        start, stop = _index(frame_start, "frame_start"), _index(frame_stop, "frame_stop", minimum=1)
        if not work or not config or not 0 <= start < stop <= self.frame_count:
            raise ValueError("coverage packet expected identity/range is invalid")
        path = Path(str(receipt["path"])).resolve(strict=True)
        if not path.is_file() or path.stat().st_size != _index(receipt["file_size_bytes"], "coverage file size"):
            raise ValueError("coverage packet file size differs from receipt")
        digest = _sha256(path)
        if digest != str(receipt["sha256"]):
            raise RuntimeError("coverage packet digest changed")
        expected = None if expected_lineages is None else {_lineage(item) for item in expected_lineages}
        with np.load(path, allow_pickle=False) as archive:
            encoded = archive["metadata"]
            if encoded.dtype != np.uint8 or encoded.ndim != 1:
                raise ValueError("coverage metadata must use byte storage")
            metadata = json.loads(encoded.tobytes().decode("utf-8"))
            identity = (metadata.get("work_id"), _index(metadata["tile_index"], "packet tile_index"), metadata.get("tile_config_id"))
            if metadata.get("schema") != COVERAGE_SCHEMA or metadata.get("encoding") != _ENCODING or identity != (work, tile, config):
                raise ValueError("coverage packet metadata differs from expected identity")
            if metadata.get("interval_semantics") != "directed observed frame intervals; half-open":
                raise ValueError("coverage packet interval semantics are unsupported")
            if (receipt.get("work_id"), _index(receipt["tile_index"], "receipt tile_index"), receipt.get("tile_config_id")) != identity:
                raise ValueError("coverage receipt identity differs from its packet")
            shape = _shape(metadata["tile_shape_hw"])
            if list(shape) != receipt.get("tile_shape_hw"):
                raise ValueError("coverage receipt shape differs from its packet")
            lineages = tuple(LtaLineageId(**item) for item in metadata["lineages"])
            tokens = [item.token for item in lineages]
            if not lineages or tokens != sorted(set(tokens)) or any(item.tile_config_id != config for item in lineages):
                raise ValueError("coverage packet lineage identity/configuration is invalid")
            if expected is not None and not set(lineages).issubset(expected):
                raise ValueError("coverage packet contains an unexpected lineage")
            masks, observed = archive["masks"], archive["observed"]
            if masks.dtype != np.int64 or masks.ndim != 2 or masks.shape[1] != 7:
                raise ValueError("coverage mask index must be an Nx7 int64 array")
            if observed.dtype != np.int64 or observed.ndim != 2 or observed.shape[1] != 4:
                raise ValueError("coverage observed ranges must be an Nx4 int64 array")
            if (len(lineages), len(masks), len(observed)) != tuple(_index(receipt[key], key) for key in ("lineage_count", "mask_count", "directed_interval_count")):
                raise ValueError("coverage packet counts differ from receipt")
            expected_keys = {"metadata", "masks", "observed"} | {f"mask_{i:08d}" for i in range(len(masks))}
            if len(archive.files) != len(expected_keys) or set(archive.files) != expected_keys:
                raise ValueError("coverage packet has missing or unexpected arrays")
            previous = None
            packed_bytes = 0
            for ordinal, row in enumerate(masks):
                lineage_index, frame = int(row[0]), int(row[1])
                if not 0 <= lineage_index < len(lineages) or not start <= frame < stop:
                    raise ValueError("coverage mask lineage/frame is outside its expected window")
                key = lineage_index, frame
                if previous is not None and key <= previous:
                    raise ValueError("coverage mask index is not unique and ordered")
                previous = key
                packed_bytes += len(_packet_crop(archive, row, ordinal, shape)[4])
            if packed_bytes != _index(receipt["stored_mask_bytes"], "stored_mask_bytes"):
                raise ValueError("coverage packed byte count differs from receipt")
            for lineage_index, direction, first, last in observed:
                if not 0 <= lineage_index < len(lineages) or int(direction) not in (-1, 1) or not start <= first < last <= stop:
                    raise ValueError("coverage directed observation is outside its expected window")
            # All packet arrays have now been checked before the first SQL write.
            existing = self._db.execute("SELECT sha256 FROM packets WHERE work_id=?", (work,)).fetchone()
            if existing is not None:
                if existing[0] != digest:
                    raise ValueError("one coverage work_id changed its packet digest")
                self._audit["duplicate_packets"] += 1
                return {"duplicate": True, "mask_count": len(masks)}
            with self._db:
                prior_shape = self._db.execute("SELECT height,width FROM tiles WHERE config=? AND tile=?", (config, tile)).fetchone()
                if prior_shape is not None and tuple(prior_shape) != shape:
                    raise ValueError("coverage tile geometry changed across packets")
                lineage_metadata = [json.dumps(asdict(item), sort_keys=True, separators=(",", ":")) for item in lineages]
                for token, record in zip(tokens, lineage_metadata):
                    prior = self._db.execute("SELECT metadata FROM lineages WHERE token=?", (token,)).fetchone()
                    if prior is not None and prior[0] != record:
                        raise ValueError("coverage lineage token changed its exact identity")
                self._db.execute("INSERT OR IGNORE INTO tiles VALUES (?,?,?,?)", (config, tile, *shape))
                self._db.executemany("INSERT OR IGNORE INTO lineages VALUES (?,?)", zip(tokens, lineage_metadata))
                for ordinal, row in enumerate(masks):
                    token, frame = tokens[int(row[0])], int(row[1])
                    crop = _packet_crop(archive, row, ordinal, shape)
                    old = self._db.execute("SELECT y,x,height,width,packed,foreground FROM masks WHERE token=? AND config=? AND tile=? AND frame=?", (token, config, tile, frame)).fetchone()
                    if old is not None:
                        crop = _union_crops(tuple(old), crop)
                    self._db.execute("INSERT OR REPLACE INTO masks VALUES (?,?,?,?,?,?,?,?,?,?)", (token, config, tile, frame, *crop))
                for lineage_index, direction, first, last in observed:
                    edge_start, edge_stop = int(first), int(last) - 1
                    if edge_start == edge_stop:
                        continue
                    token = tokens[int(lineage_index)]
                    scope = token, config, tile, int(direction)
                    overlaps = self._db.execute("SELECT start,stop FROM edges WHERE token=? AND config=? AND tile=? AND direction=? AND stop>=? AND start<=?", (*scope, edge_start, edge_stop)).fetchall()
                    self._db.execute("DELETE FROM edges WHERE token=? AND config=? AND tile=? AND direction=? AND stop>=? AND start<=?", (*scope, edge_start, edge_stop))
                    merged_start = min((edge_start, *(value[0] for value in overlaps)))
                    merged_stop = max((edge_stop, *(value[1] for value in overlaps)))
                    self._db.execute("INSERT INTO edges VALUES (?,?,?,?,?,?)", (*scope, merged_start, merged_stop))
                self._db.execute("INSERT INTO packets VALUES (?,?)", (work, digest))
        return {"duplicate": False, "mask_count": len(masks), "directed_interval_count": len(observed)}

    def can_handoff(self, seed, *, tile_index, direction) -> bool:
        self._ensure_open()
        if direction not in _DIRECTIONS:
            raise ValueError("coverage handoff direction must be forward or backward")
        lineage = _lineage(seed.lineage)
        frame = _index(seed.frame_index, "seed frame_index")
        tile = _index(tile_index, "tile_index")
        if frame >= self.frame_count:
            raise ValueError("coverage seed frame is outside the run")
        candidate = _binary(seed.mask)
        self._audit["handoff_checks"] += 1
        if not bool(candidate.any()):
            self._audit["rejected_empty_candidate"] += 1
            return False
        scope = lineage.token, lineage.tile_config_id, tile
        geometry = self._db.execute("SELECT height,width FROM tiles WHERE config=? AND tile=?", scope[1:]).fetchone()
        if geometry is not None and tuple(candidate.shape) != tuple(geometry):
            raise ValueError("coverage candidate shape differs from its known tile")
        identity = self._db.execute("SELECT metadata FROM lineages WHERE token=?", (lineage.token,)).fetchone()
        record = json.dumps(asdict(lineage), sort_keys=True, separators=(",", ":"))
        if identity is None or identity[0] != record:
            self._audit["admitted_unmatched_lineage"] += 1
            return False
        covered = self._db.execute("SELECT y,x,height,width,packed,foreground FROM masks WHERE token=? AND config=? AND tile=? AND frame=?", (*scope, frame)).fetchone()
        if covered is None:
            self._audit["admitted_uncovered_frame"] += 1
            return False
        incoming = _crop(candidate)
        y, x, height, width, _packed, _count = incoming
        if y < covered[0] or x < covered[1] or y + height > covered[0] + covered[2] or x + width > covered[1] + covered[3]:
            self._audit["admitted_novel_pixels"] += 1
            return False
        existing_mask = _decode(tuple(covered))
        dy, dx = y - covered[0], x - covered[1]
        if bool(np.any(_decode(incoming) & ~existing_mask[dy:dy + height, dx:dx + width])):
            self._audit["admitted_novel_pixels"] += 1
            return False
        endpoint = frame == (self.frame_count - 1 if direction == "forward" else 0)
        if not endpoint:
            edge = frame if direction == "forward" else frame - 1
            traversed = self._db.execute("SELECT 1 FROM edges WHERE token=? AND config=? AND tile=? AND direction=? AND start<=? AND stop>? LIMIT 1", (*scope, _DIRECTIONS[direction], edge, edge)).fetchone()
            if traversed is None:
                self._audit["admitted_unobserved_direction"] += 1
                return False
        self._audit["covered_handoffs"] += 1
        if endpoint:
            self._audit["endpoint_handoffs"] += 1
        return True


__all__ = ("COVERAGE_SCHEMA", "LtaCoverageBuilder", "LtaCoverageLedger")
