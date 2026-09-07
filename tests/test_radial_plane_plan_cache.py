"""Shared plan builds preserve readonly ownership, independent keys and clear epochs."""
from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import gc
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np

from XTA import cylindrical_projection as cp


TIMEOUT = 5.0


class ObservedFuture(Future):
    def __init__(self):
        super().__init__()
        self.waiter_condition = threading.Condition()
        self.waiters = 0

    def result(self, timeout=None):
        with self.waiter_condition:
            self.waiters += 1
            self.waiter_condition.notify_all()
        return super().result(timeout)

    def wait_for_waiters(self, count):
        with self.waiter_condition:
            if not self.waiter_condition.wait_for(lambda: self.waiters >= count, TIMEOUT):
                raise AssertionError('shared-plan waiters did not arrive')


class InterruptedBuild(BaseException):
    pass


class RadialPlanePlanCacheTests(unittest.TestCase):
    def setUp(self):
        cp.clear_radial_plane_plan_cache()
        self.addCleanup(cp.clear_radial_plane_plan_cache)
        patch = mock.patch.object(cp, 'Future', ObservedFuture)
        patch.start()
        self.addCleanup(patch.stop)
        self.view = SimpleNamespace(
            radial_base_view='transverse', full_t=3, full_h=5, full_w=7,
            center_x=3.0, center_y=2.0, radial_min_radius=.75, radial_max_radius=2.0,
            radial_shell_start=0, num_slices=3, src_w=8, radial_arc_origin=0.0,
        )
        self.radii = np.asarray((.75, 1.5, 2.0))
        self.shape = (3, 5, 7)

    def call(self, view=None):
        return cp._radial_plane_plan(view or self.view, self.radii, self.shape)

    @staticmethod
    def plan(marker=1):
        arrays = (
            np.full(35, marker, np.int32), np.arange(36, dtype=np.uint32),
            np.zeros(35, np.int32),
        )
        for array in arrays:
            array.flags.writeable = False
        return cp.RadialPlanePlan(0, (5, 7), *arrays)

    def flight(self):
        with cp._PLANE_PLAN_LOCK:
            self.assertEqual(len(cp._PLANE_PLAN_INFLIGHT), 1)
            return next(iter(cp._PLANE_PLAN_INFLIGHT.values()))

    def test_identical_keys_share_one_build_and_one_readonly_cached_plan(self):
        entered, release = threading.Event(), threading.Event()
        built = self.plan()
        def build(*_args):
            entered.set()
            if not release.wait(TIMEOUT):
                raise AssertionError('build was not released')
            return built
        with mock.patch.object(cp, '_build_radial_plane_plan', side_effect=build) as builder, \
                ThreadPoolExecutor(max_workers=4) as pool:
            leader = pool.submit(self.call)
            try:
                self.assertTrue(entered.wait(TIMEOUT))
                flight = self.flight()
                followers = [pool.submit(self.call) for _ in range(3)]
                flight.wait_for_waiters(3)
                self.assertEqual(builder.call_count, 1)
                self.assertFalse(flight.cancel())
            finally:
                release.set()
            results = [task.result(TIMEOUT) for task in (leader, *followers)]
            self.assertEqual([hit for _, hit in results], [False, True, True, True])
            self.assertTrue(all(plan is built for plan, _ in results))
            self.assertEqual(cp._PLANE_PLAN_CACHE_SIZE, built.nbytes)
            self.assertFalse(cp._PLANE_PLAN_INFLIGHT)
            self.assertIs(self.call()[0], built)
            self.assertEqual(builder.call_count, 1)
            for array in (built.shell_index, built.column_offsets, built.native_columns):
                self.assertFalse(array.flags.writeable)

    def test_unrelated_keys_build_concurrently(self):
        together = threading.Barrier(2)
        other = SimpleNamespace(**{**vars(self.view), 'center_x': 3.5})
        def build(view, *_args):
            together.wait(TIMEOUT)
            return self.plan(1 if view is self.view else 2)
        with mock.patch.object(cp, '_build_radial_plane_plan', side_effect=build) as builder, \
                ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(self.call, view) for view in (self.view, other)]
            results = [task.result(TIMEOUT) for task in tasks]
        self.assertEqual(builder.call_count, 2)
        self.assertIsNot(results[0][0], results[1][0])
        self.assertEqual([hit for _, hit in results], [False, False])

    def test_failure_and_owner_cancellation_reach_waiters_and_allow_retry(self):
        for error in (RuntimeError('build failed'), CancelledError('owner cancelled'),
                      InterruptedBuild('owner interrupted'), cp._RadialPlanePlanTooLarge('too large')):
            with self.subTest(error=type(error).__name__):
                cp.clear_radial_plane_plan_cache()
                entered, release = threading.Event(), threading.Event()
                def build(*_args):
                    entered.set()
                    if not release.wait(TIMEOUT):
                        raise AssertionError('build was not released')
                    raise error
                with mock.patch.object(cp, '_build_radial_plane_plan', side_effect=build) as builder, \
                        ThreadPoolExecutor(max_workers=2) as pool:
                    leader = pool.submit(self.call)
                    try:
                        self.assertTrue(entered.wait(TIMEOUT))
                        flight = self.flight()
                        follower = pool.submit(self.call)
                        flight.wait_for_waiters(1)
                    finally:
                        release.set()
                    for task in (leader, follower):
                        with self.assertRaises(type(error)) as caught:
                            task.result(TIMEOUT)
                        self.assertIs(caught.exception, error)
                    self.assertEqual(builder.call_count, 1)
                self.assertFalse(cp._PLANE_PLAN_CACHE)
                self.assertFalse(cp._PLANE_PLAN_INFLIGHT)
                self.assertEqual(cp._PLANE_PLAN_CACHE_SIZE, 0)
                with mock.patch.object(cp, '_build_radial_plane_plan', return_value=self.plan()) as retry:
                    self.assertFalse(self.call()[1])
                    retry.assert_called_once()

    def test_abandoned_waiter_does_not_cancel_or_remove_shared_work(self):
        entered, release = threading.Event(), threading.Event()
        built = self.plan()
        def build(*_args):
            entered.set()
            if not release.wait(TIMEOUT):
                raise AssertionError('build was not released')
            return built
        with mock.patch.object(cp, '_build_radial_plane_plan', side_effect=build) as builder, \
                ThreadPoolExecutor(max_workers=3) as pool:
            leader = pool.submit(self.call)
            try:
                self.assertTrue(entered.wait(TIMEOUT))
                flight = self.flight()
                with mock.patch.object(flight, 'result', side_effect=CancelledError('wait abandoned')):
                    abandoned = pool.submit(self.call)
                    with self.assertRaises(CancelledError):
                        abandoned.result(TIMEOUT)
                self.assertIs(self.flight(), flight)
                self.assertFalse(flight.cancelled())
                follower = pool.submit(self.call)
                flight.wait_for_waiters(1)
            finally:
                release.set()
            self.assertIs(leader.result(TIMEOUT)[0], built)
            self.assertIs(follower.result(TIMEOUT)[0], built)
            self.assertEqual(builder.call_count, 1)

    def test_clear_detaches_old_build_without_losing_waiters_or_new_generation(self):
        for old_first in (True, False):
            with self.subTest(old_finishes_first=old_first):
                cp.clear_radial_plane_plan_cache()
                entered = [threading.Event(), threading.Event()]
                release = [threading.Event(), threading.Event()]
                built = [self.plan(1), self.plan(2)]
                counter, lock = [0], threading.Lock()
                def build(*_args):
                    with lock:
                        index = counter[0]
                        counter[0] += 1
                    entered[index].set()
                    if not release[index].wait(TIMEOUT):
                        raise AssertionError('build was not released')
                    return built[index]
                with mock.patch.object(cp, '_build_radial_plane_plan', side_effect=build), \
                        ThreadPoolExecutor(max_workers=3) as pool:
                    old_leader = pool.submit(self.call)
                    try:
                        self.assertTrue(entered[0].wait(TIMEOUT))
                        old_flight = self.flight()
                        old_waiter = pool.submit(self.call)
                        old_flight.wait_for_waiters(1)
                        cp.clear_radial_plane_plan_cache()
                        self.assertFalse(old_flight.done())
                        self.assertFalse(cp._PLANE_PLAN_CACHE)
                        new_leader = pool.submit(self.call)
                        self.assertTrue(entered[1].wait(TIMEOUT))
                        new_flight = self.flight()
                        self.assertIsNot(new_flight, old_flight)
                        first = 0 if old_first else 1
                        release[first].set()
                        (old_leader if old_first else new_leader).result(TIMEOUT)
                        if old_first:
                            self.assertIs(self.flight(), new_flight)
                            self.assertFalse(cp._PLANE_PLAN_CACHE)
                        else:
                            self.assertIs(self.call()[0], built[1])
                        release[1 - first].set()
                        self.assertIs(old_leader.result(TIMEOUT)[0], built[0])
                        self.assertIs(old_waiter.result(TIMEOUT)[0], built[0])
                        self.assertIs(new_leader.result(TIMEOUT)[0], built[1])
                        self.assertIs(self.call()[0], built[1])
                        self.assertEqual(counter[0], 2)
                        self.assertEqual(cp._PLANE_PLAN_CACHE_SIZE, built[1].nbytes)
                    finally:
                        for event in release:
                            event.set()

    def test_uncached_large_plan_still_shares_active_build(self):
        entered, release = threading.Event(), threading.Event()
        built = self.plan()
        def build(*_args):
            entered.set()
            if not release.wait(TIMEOUT):
                raise AssertionError('build was not released')
            return built
        with mock.patch.object(cp, '_PLANE_PLAN_CACHE_BYTES', 0), \
                mock.patch.object(cp, '_build_radial_plane_plan', side_effect=build) as builder, \
                ThreadPoolExecutor(max_workers=2) as pool:
            leader = pool.submit(self.call)
            try:
                self.assertTrue(entered.wait(TIMEOUT))
                flight = self.flight()
                follower = pool.submit(self.call)
                flight.wait_for_waiters(1)
            finally:
                release.set()
            self.assertIs(leader.result(TIMEOUT)[0], built)
            self.assertIs(follower.result(TIMEOUT)[0], built)
            self.assertEqual(builder.call_count, 1)
            self.assertFalse(cp._PLANE_PLAN_CACHE)
            self.assertFalse(self.call()[1])
            self.assertEqual(builder.call_count, 2)

    def test_completion_callbacks_run_outside_cache_lock(self):
        entered, release = threading.Event(), threading.Event()
        observed = []
        def build(*_args):
            entered.set()
            if not release.wait(TIMEOUT):
                raise AssertionError('build was not released')
            return self.plan()
        def inspect_lock(_flight):
            available = cp._PLANE_PLAN_LOCK.acquire(blocking=False)
            observed.append(available)
            if available:
                cp._PLANE_PLAN_LOCK.release()
        with mock.patch.object(cp, '_build_radial_plane_plan', side_effect=build), \
                ThreadPoolExecutor(max_workers=1) as pool:
            leader = pool.submit(self.call)
            try:
                self.assertTrue(entered.wait(TIMEOUT))
                self.flight().add_done_callback(inspect_lock)
            finally:
                release.set()
            leader.result(TIMEOUT)
        self.assertEqual(observed, [True])

    def test_clear_keeps_returned_plan_alive_but_does_not_retain_it(self):
        with mock.patch.object(cp, '_build_radial_plane_plan', side_effect=lambda *_: self.plan()):
            plan, _ = self.call()
        ref = weakref.ref(plan)
        cp.clear_radial_plane_plan_cache()
        self.assertIs(ref(), plan)
        np.testing.assert_array_equal(plan.shell_index, 1)
        self.assertFalse(plan.shell_index.flags.writeable)
        del plan
        gc.collect()
        self.assertIsNone(ref())

    def test_lru_byte_bound_and_returned_plan_lifetime_survive_eviction(self):
        views = [SimpleNamespace(**{**vars(self.view), 'center_x': value}) for value in (3.0, 3.5, 4.0)]
        plan_bytes = self.plan().nbytes
        with mock.patch.object(cp, '_PLANE_PLAN_CACHE_BYTES', 2 * plan_bytes), \
                mock.patch.object(cp, '_build_radial_plane_plan', side_effect=lambda *_: self.plan()) as builder:
            first, _ = self.call(views[0])
            evicted, _ = self.call(views[1])
            self.assertIs(self.call(views[0])[0], first)  # First becomes most recent.
            self.call(views[2])
            self.assertEqual(len(cp._PLANE_PLAN_CACHE), 2)
            self.assertEqual(cp._PLANE_PLAN_CACHE_SIZE, 2 * plan_bytes)
            self.assertIs(self.call(views[0])[0], first)
            rebuilt, hit = self.call(views[1])
            self.assertFalse(hit)
            self.assertIsNot(rebuilt, evicted)
            self.assertEqual(builder.call_count, 4)
            self.assertLessEqual(cp._PLANE_PLAN_CACHE_SIZE, 2 * plan_bytes)
            np.testing.assert_array_equal(evicted.shell_index, 1)
            self.assertFalse(evicted.shell_index.flags.writeable)


if __name__ == '__main__':
    unittest.main()
