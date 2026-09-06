"""Ownership, backpressure and dependency retirement without a CUDA device."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import unittest

from XTA.projection_queue import (
    ComponentProjectionQueue, prepared_view_waitables, settle_prepared_view_components,
)


class ProjectionQueueTests(unittest.TestCase):
    def queue(self, **kwargs):
        args = dict(workers=2, max_pending=2, max_source_bytes=10, max_working_bytes=10)
        args.update(kwargs)
        queue = ComponentProjectionQueue(**args)
        self.addCleanup(queue.shutdown, cancel_futures=True)
        self.addCleanup(queue.abort)
        return queue

    def test_submission_waits_for_source_credit_and_preserves_every_job(self):
        queue = self.queue()
        release, started, submitting = Event(), Event(), Event()
        self.addCleanup(release.set)
        def first():
            started.set()
            self.assertTrue(release.wait(5))
            return 'first'
        first_future = queue.submit(first, source_bytes=8, working_bytes=1)
        self.assertTrue(started.wait(2))
        with ThreadPoolExecutor(1) as producers:
            def submit():
                submitting.set()
                return queue.submit(lambda: 'second', source_bytes=8, working_bytes=1)
            submitted = producers.submit(submit)
            self.assertTrue(submitting.wait(2))
            try:
                self.assertFalse(submitted.done())
            finally:
                release.set()
            second_future = submitted.result(2)
        self.assertEqual(first_future.result(2), 'first')
        self.assertEqual(second_future.result(2), 'second')
        queue.shutdown()
        stats = queue.snapshot()
        self.assertEqual((stats['submitted'], stats['completed'], stats['pending']), (2, 2, 0))
        self.assertEqual(stats['peak_source_bytes'], 8)

    def test_oversized_working_job_runs_alone(self):
        queue = self.queue()
        release, entered, second_entered = Event(), Event(), Event()
        self.addCleanup(release.set)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        first_future = queue.submit(first, source_bytes=1, working_bytes=30)
        self.assertTrue(entered.wait(2))
        second_future = queue.submit(second_entered.set, source_bytes=1, working_bytes=2)
        try:
            self.assertFalse(second_entered.wait(0.05))
        finally:
            release.set()
        first_future.result(2)
        second_future.result(2)
        self.assertTrue(second_entered.is_set())
        self.assertEqual(queue.snapshot()['peak_working_bytes'], 30)

    def test_abort_wakes_blocked_parent_and_waiting_worker(self):
        queue = self.queue(max_pending=2)
        release, entered, submitting = Event(), Event(), Event()
        self.addCleanup(release.set)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        active = queue.submit(first, source_bytes=1, working_bytes=10)
        self.assertTrue(entered.wait(2))
        waiting = queue.submit(lambda: self.fail('aborted worker ran'), source_bytes=1, working_bytes=10)
        with ThreadPoolExecutor(1) as producers:
            def submit():
                submitting.set()
                return queue.submit(lambda: None, source_bytes=1, working_bytes=1)
            blocked = producers.submit(submit)
            self.assertTrue(submitting.wait(2))
            queue.abort()
            with self.assertRaisesRegex(RuntimeError, 'closed|failed'):
                blocked.result(2)
        with self.assertRaisesRegex(RuntimeError, 'aborted'):
            waiting.result(2)
        release.set()
        active.result(2)
        queue.shutdown()
        self.assertEqual(queue.snapshot()['pending'], 0)

    def test_projection_failure_returns_credit_and_rejects_more_inputs(self):
        queue = self.queue()
        def fail():
            raise ValueError('broken component')
        failed = queue.submit(fail, source_bytes=50, working_bytes=1)
        with self.assertRaisesRegex(ValueError, 'broken component'):
            failed.result(2)
        queue.shutdown()
        self.assertEqual(queue.snapshot()['source_bytes'], 0)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            queue.submit(lambda: None, source_bytes=1, working_bytes=1)

    def test_pending_limit_applies_even_to_zero_byte_components(self):
        queue = self.queue(max_pending=1)
        release, entered = Event(), Event()
        self.addCleanup(release.set)
        def first():
            entered.set()
            self.assertTrue(release.wait(5))
        future = queue.submit(first, source_bytes=0, working_bytes=0)
        self.assertTrue(entered.wait(2))
        queue.abort()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            queue.submit(lambda: None, source_bytes=0, working_bytes=0)
        release.set()
        future.result(2)


class PreparedViewHandoffTests(unittest.TestCase):
    def test_waits_on_children_and_preserves_deterministic_ref_order(self):
        first, second, parent = Future(), Future(), Future()
        prepared = SimpleNamespace(pending_component_layers=[first, second], nrrd_layers=['base'])
        self.assertEqual(prepared_view_waitables([parent]), [parent])
        parent.set_result(prepared)
        self.assertEqual(prepared_view_waitables([parent]), [first, second])
        second.set_result('second')
        self.assertFalse(settle_prepared_view_components(prepared))
        self.assertEqual(prepared.nrrd_layers, ['base'])
        self.assertEqual(prepared_view_waitables([parent]), [first])
        first.set_result('first')
        self.assertTrue(settle_prepared_view_components(prepared))
        self.assertEqual(prepared.nrrd_layers, ['base', 'first', 'second'])
        self.assertTrue(settle_prepared_view_components(prepared))
        self.assertEqual(prepared.nrrd_layers, ['base', 'first', 'second'])
        self.assertEqual(prepared_view_waitables([parent]), [])

    def test_child_error_is_visible_before_all_children_finish(self):
        first, second = Future(), Future()
        prepared = SimpleNamespace(pending_component_layers=[first, second], nrrd_layers=[])
        second.set_exception(ValueError('publication failed'))
        with self.assertRaisesRegex(ValueError, 'publication failed'):
            settle_prepared_view_components(prepared)
        self.assertEqual(prepared.nrrd_layers, [])

    def test_failed_parent_remains_waitable(self):
        failed = Future()
        failed.set_exception(ValueError('parent failed'))
        self.assertEqual(prepared_view_waitables([failed]), [failed])


if __name__ == '__main__':
    unittest.main()
