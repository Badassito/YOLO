"""Encode owner bitsets directly as bounded, row-packed bbox payloads."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ._deps import _numba


@dataclass(frozen=True)
class PackedOwnerCrop:
    z: int
    y0: int
    y1: int
    x0: int
    x1: int
    foreground: int
    offset: int
    size: int


if _numba is not None:
    @_numba.njit(cache=True, nogil=True)
    def _packed_owner_metadata(words, height, width, first, count):
        meta = np.zeros((count, 5), np.int64)
        for zi in range(count):
            y0, y1, x0, x1, foreground = height, 0, width, 0, 0
            for y in range(height):
                start = ((first + zi) * height + y) * width
                stop = start + width
                at = start
                while at < stop:
                    shift = at % 32
                    bits = min(32 - shift, stop - at)
                    value = (np.uint64(words[at // 32]) >> np.uint64(shift)) & ((np.uint64(1) << np.uint64(bits)) - np.uint64(1))
                    if value:
                        # SWAR popcount: exact count without decoding 32 byte pixels.
                        v = value - ((value >> np.uint64(1)) & np.uint64(0x55555555))
                        v = (v & np.uint64(0x33333333)) + ((v >> np.uint64(2)) & np.uint64(0x33333333))
                        v = (v + (v >> np.uint64(4))) & np.uint64(0x0f0f0f0f)
                        foreground += np.int64(((v * np.uint64(0x01010101)) >> np.uint64(24)) & np.uint64(0xff))
                        lo, hi = 0, bits - 1
                        while not (value & (np.uint64(1) << np.uint64(lo))):
                            lo += 1
                        while not (value & (np.uint64(1) << np.uint64(hi))):
                            hi -= 1
                        x0 = min(x0, at - start + lo)
                        x1 = max(x1, at - start + hi + 1)
                        y0, y1 = min(y0, y), y + 1
                    at += bits
            if foreground:
                meta[zi, 0], meta[zi, 1] = y0, y1
                meta[zi, 2], meta[zi, 3] = x0, x1
                meta[zi, 4] = foreground
        return meta

    @_numba.njit(cache=True, nogil=True)
    def _packed_owner_encode(words, height, width, first, meta):
        total = 0
        for row in meta:
            total += (row[1] - row[0]) * ((row[3] - row[2] + 7) // 8)
        payload = np.empty(total, np.uint8)
        cursor = 0
        for zi in range(len(meta)):
            y0, y1, x0, x1, foreground = meta[zi]
            for y in range(y0, y1):
                for x in range(x0, x1, 8):
                    at = ((first + zi) * height + y) * width + x
                    shift = at % 32
                    value = np.uint64(words[at // 32]) >> np.uint64(shift)
                    bits = min(8, x1 - x)
                    if shift + bits > 32:
                        value |= np.uint64(words[at // 32 + 1]) << np.uint64(32 - shift)
                    payload[cursor] = np.uint8(value & ((np.uint64(1) << np.uint64(bits)) - np.uint64(1)))
                    cursor += 1
        return payload
else:
    _packed_owner_metadata = _packed_owner_encode = None


def encode_owner_packed_block(words, shape, first, count):
    """Return exact crop records and packed bytes, without a dense uint8 block."""
    depth, height, width = map(int, shape)
    first, count = int(first), int(count)
    if min(depth, height, width) <= 0 or first < 0 or count < 0 or first + count > depth:
        raise ValueError('Invalid packed publication geometry')
    if (not isinstance(words, np.ndarray) or words.dtype != np.uint32 or words.ndim != 1
            or not words.flags.c_contiguous or len(words) != (depth * height * width + 31) // 32):
        raise ValueError('Packed publication requires the exact contiguous source uint32 bitset')
    if _packed_owner_metadata is None:
        raise NotImplementedError('Packed publication requires Numba')
    meta = _packed_owner_metadata(words, height, width, first, count)
    payload = _packed_owner_encode(words, height, width, first, meta)
    records, cursor = [], 0
    for zi, row in enumerate(meta):
        y0, y1, x0, x1, foreground = map(int, row)
        size = (y1 - y0) * ((x1 - x0 + 7) // 8)
        records.append(PackedOwnerCrop(first + zi, y0, y1, x0, x1, foreground, cursor, size))
        cursor += size
    if cursor != len(payload):
        raise RuntimeError('Packed publication payload accounting mismatch')
    return records, payload
