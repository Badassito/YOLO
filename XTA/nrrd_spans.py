"""Stream native bbox layers without compressing their empty top/bottom rows."""
from __future__ import annotations

import numpy as np
import gzip
from functools import lru_cache


@lru_cache(maxsize=21)
def canonical_zero_member(size):
    """Small reusable gzip members: only powers of two through 1 MiB are legal."""
    size = int(size)
    if size <= 0 or size > 1024**2 or size & (size-1):
        raise ValueError('Canonical zero members require powers of two up to 1 MiB')
    return gzip.compress(bytes(size), compresslevel=9, mtime=0)


def stream_native_crop_spans(store, payload_writer, first, stop, member_bytes, sparse_consumer=None):
    """Emit the same C-order uint8 stream from bounded row bands and cached zeros.

    The caller proves native geometry, no dense-block observer, and a software
    member writer. Sparse observers still see each complete crop exactly once.
    """
    _, height, width = map(int, store.shape)
    rows = max(1, int(member_bytes) // max(1, width))
    zero_bytes = 0
    encoded_bytes = 0
    for z in range(int(first), int(stop)):
        crop_info = store.decode_slice_crop(z, dtype=np.uint8)
        if crop_info is None:
            zero_bytes += height * width
            continue
        y0, x0, y1, x1, crop = crop_info
        zero_bytes += y0 * width
        if zero_bytes:
            payload_writer.write_canonical_zeros(zero_bytes)
            zero_bytes = 0
        if sparse_consumer is not None:
            sparse_consumer(z, y0, x0, crop)
        for y in range(0, y1-y0, rows):
            count = min(rows, y1-y0-y)
            band = np.zeros((count, width), np.uint8)
            band[:, x0:x1] = crop[y:y+count]
            payload_writer.write_owned_known_nonzero(band)
            encoded_bytes += band.nbytes
        zero_bytes += (height-y1) * width
    if zero_bytes:
        payload_writer.write_canonical_zeros(zero_bytes)
    return int(encoded_bytes)
