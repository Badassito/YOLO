"""Small, persistent host-phase traces for diagnosing idle LTA GPU workers."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
from typing import Iterator


def lta_source_fingerprint() -> dict[str, object]:
    """Identify LTA modules and the shared decoder, including uncommitted edits."""
    root = Path(__file__).resolve().parent
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted((*root.glob('lta_*.py'), root / 'media.py'))}
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


class LtaExecutionTrace:
    """One JSONL stream per process/role; never initializes numerical runtimes.

    Times are host wall time, including any waits, not CUDA kernel durations.
    Buffered writes flush at least once per second when events are emitted.
    Diagnostics failure does not invalidate already computed masks.
    """
    def __init__(self, directory: str | Path | None, role: str):
        self.path: Path | None = None
        self._handle = None
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self.write_errors = 0
        self.events = 0
        self.role = re.sub(r'[^A-Za-z0-9_-]', '_', str(role))
        self.host = re.sub(r'[^A-Za-z0-9_-]', '_', socket.gethostname())
        if directory is not None:
            try:
                root = Path(directory)
                root.mkdir(parents=True, exist_ok=True)
                self.path = root / f'{self.role}-{self.host}-{os.getpid()}.jsonl'
                self._handle = self.path.open('a', encoding='utf-8', buffering=65536, newline='\n')
            except OSError:
                self.write_errors += 1

    def event(self, name: str, **fields: object) -> None:
        if self._handle is None:
            return
        now = time.monotonic()
        value = {**fields, "schema": "lta.host-phase/1", "event": str(name), "role": self.role,
                 "pid": os.getpid(), "thread_id": threading.get_ident(),
                 "time_unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                 "host": self.host}
        try:
            encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            with self._lock:
                if self._handle is None:
                    return
                self._handle.write(encoded + '\n')
                self.events += 1
                if now - self._last_flush >= 1.0 or name in {'process_ready', 'process_failed', 'phase_start'}:
                    self._handle.flush()
                    self._last_flush = now
        except (OSError, TypeError, ValueError):
            self.write_errors += 1

    @contextmanager
    def phase(self, name: str, **fields: object) -> Iterator[dict[str, object]]:
        details = dict(fields)
        if self._handle is None:
            yield details
            return
        start = time.monotonic()
        span_id = f'{os.getpid()}:{threading.get_ident()}:{time.monotonic_ns()}'
        self.event('phase_start', **{**details, 'phase': name, 'span_id': span_id})
        try:
            yield details
        except BaseException as error:
            self.event('phase_end', **{**details, 'phase': name, 'span_id': span_id, 'status': 'failed',
                                      'error_type': type(error).__name__, 'wall_seconds': time.monotonic() - start})
            raise
        else:
            self.event('phase_end', **{**details, 'phase': name, 'span_id': span_id, 'status': 'complete',
                                      'wall_seconds': time.monotonic() - start})

    def flush(self) -> None:
        if self._handle is not None:
            try:
                with self._lock:
                    self._handle.flush()
                    self._last_flush = time.monotonic()
            except OSError:
                self.write_errors += 1

    def close(self) -> None:
        if self._handle is not None:
            try:
                self.flush()
                self._handle.close()
            except OSError:
                self.write_errors += 1
            finally:
                self._handle = None


__all__ = ('LtaExecutionTrace', 'lta_source_fingerprint')
