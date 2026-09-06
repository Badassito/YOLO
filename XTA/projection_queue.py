"""Bounded ownership handoff from interpolation to immutable layer projection."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Condition
from time import perf_counter
from typing import Callable, Iterable, TypeVar

T = TypeVar('T')


class ComponentProjectionQueue:
    """Bound retained inputs and active scratch independently of parent preparation.

    Submission transfers ownership of an immutable input to its future. A full
    queue applies backpressure to the producer; it never drops a layer. One job
    larger than a byte limit may run alone, so unusual geometries still progress.
    Neither limit claims to bound other pipeline stages or OS page-cache residency.
    """

    def __init__(self, *, workers: int, max_pending: int,
                 max_source_bytes: int, max_working_bytes: int) -> None:
        self._condition = Condition()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix='component-projection',
        )
        self._max_pending = max(1, int(max_pending))
        self._max_source = max(1, int(max_source_bytes))
        self._max_working = max(1, int(max_working_bytes))
        self._closed = False
        self._aborted = False
        self._error: BaseException | None = None
        self._pending = self._source = self._active = self._working = 0
        self._stats = dict(submitted=0, completed=0, failed=0,
                           peak_pending=0, peak_source_bytes=0,
                           peak_active=0, peak_working_bytes=0,
                           submission_wait_seconds=0.0, working_wait_seconds=0.0)

    def submit(self, fn: Callable[..., T], /, *args,
               source_bytes: int, working_bytes: int, **kwargs) -> Future[T]:
        source_bytes, working_bytes = max(0, int(source_bytes)), max(0, int(working_bytes))
        started = perf_counter()
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError('Component projection failed') from self._error
                if self._closed:
                    raise RuntimeError('Component projection queue is closed')
                if (self._pending < self._max_pending and
                    (self._pending == 0 or self._source + source_bytes <= self._max_source)):
                    break
                self._condition.wait()
            self._pending += 1
            self._source += source_bytes
            self._stats['submitted'] += 1
            self._stats['peak_pending'] = max(self._stats['peak_pending'], self._pending)
            self._stats['peak_source_bytes'] = max(self._stats['peak_source_bytes'], self._source)
            self._stats['submission_wait_seconds'] += perf_counter() - started

        def run() -> T:
            started = perf_counter()
            with self._condition:
                while self._active and self._working + working_bytes > self._max_working:
                    if self._aborted:
                        raise RuntimeError('Component projection aborted before admission')
                    self._condition.wait()
                if self._aborted:
                    raise RuntimeError('Component projection aborted before admission')
                self._active += 1
                self._working += working_bytes
                self._stats['peak_active'] = max(self._stats['peak_active'], self._active)
                self._stats['peak_working_bytes'] = max(self._stats['peak_working_bytes'], self._working)
                self._stats['working_wait_seconds'] += perf_counter() - started
            try:
                return fn(*args, **kwargs)
            finally:
                with self._condition:
                    self._active -= 1
                    self._working -= working_bytes
                    self._condition.notify_all()

        def retired(future: Future) -> None:
            with self._condition:
                self._pending -= 1
                self._source -= source_bytes
                self._stats['completed'] += 1
                if not future.cancelled() and future.exception() is not None:
                    self._stats['failed'] += 1
                    self._error = future.exception()
                self._condition.notify_all()

        try:
            future = self._executor.submit(run)
        except BaseException:
            with self._condition:
                self._pending -= 1
                self._source -= source_bytes
                self._condition.notify_all()
            raise
        future.add_done_callback(retired)
        return future

    def abort(self) -> None:
        """Wake blocked parent submitters before joining the parent executor."""
        with self._condition:
            self._closed = self._aborted = True
            self._condition.notify_all()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        if cancel_futures:
            self.abort()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=bool(wait), cancel_futures=bool(cancel_futures))

    def snapshot(self) -> dict:
        with self._condition:
            return dict(self._stats, pending=self._pending, source_bytes=self._source,
                        active=self._active, working_bytes=self._working,
                        max_pending=self._max_pending, max_source_bytes=self._max_source,
                        max_working_bytes=self._max_working)


def prepared_view_waitables(parents: Iterable[Future]) -> list[Future]:
    """Wait on child publications after a parent releases its preparation thread.

    Keeping a completed parent in the wait set would spin the scheduler until
    its projection finished. A failed parent remains waitable so the next drain
    propagates its exception immediately.
    """
    waitables = []
    for parent in parents:
        if not parent.done() or parent.cancelled() or parent.exception() is not None:
            waitables.append(parent)
        else:
            waitables.extend(f for f in parent.result().pending_component_layers if not f.done())
    return waitables


def settle_prepared_view_components(prepared) -> bool:
    """Register every child atomically and in component order, or keep waiting."""
    children = prepared.pending_component_layers
    # Surface failures even if another component is still busy.
    for child in children:
        if child.done():
            child.result()
    if any(not child.done() for child in children):
        return False
    prepared.nrrd_layers.extend(child.result() for child in children)
    children.clear()
    return True
