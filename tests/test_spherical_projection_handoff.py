"""Discarded CPU prefetch settles at a chunk boundary before GPU publication."""
from concurrent.futures import CancelledError
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import spherical_projection as sp
from XTA.spherical_geometry import build_spherical_view_infos


class SphericalProjectionHandoffTests(unittest.TestCase):
    def setUp(self):
        self.shape = (7, 9, 11)
        self.view = build_spherical_view_infos(*self.shape, targets=('transverse',),
            min_radius=.7, patch_size=15, tilted_views=())[0]
        self.source = np.ones((self.view.num_slices, 15, 15), np.uint8)
        self.radii, self.rotation, _, _ = sp._validate_spherical_projection(
            self.source, self.view, self.shape, None)

    def test_pre_cancelled_block_does_not_allocate_or_read(self):
        cancelled = threading.Event()
        cancelled.set()
        with (mock.patch.object(sp.np, 'empty', side_effect=AssertionError('allocated')),
              mock.patch.object(sp, '_pull_spherical_chunk', side_effect=AssertionError('read'))):
            with self.assertRaises(CancelledError):
                sp._project_spherical_block(self.source, self.view, self.radii,
                    self.rotation, self.shape, 0, 1, cancel_event=cancelled)

    def test_prefetch_stops_before_second_chunk_and_joins_on_promotion_or_sink_error(self):
        expected = sp._project_spherical_block(self.source, self.view, self.radii,
            self.rotation, self.shape, 0, self.shape[0])
        original_block, original_pull = sp._project_spherical_block, sp._pull_spherical_chunk
        for fail_sink in (False, True):
            with self.subTest(fail_sink=fail_sink):
                lock = threading.Lock()
                ready = threading.Event()
                started = set()
                calls = {1: [], 2: []}
                events = []
                live = 0
                seen = []
                result = np.zeros(self.shape, np.uint8)
                failure = ValueError('original sink failure')

                def block(*args, **kwargs):
                    nonlocal live
                    with lock:
                        live += 1
                        events.append(kwargs['cancel_event'])
                    try:
                        return original_block(*args, **kwargs)
                    finally:
                        with lock:
                            live -= 1

                def pull(*args, **kwargs):
                    z, first = args[5:7]
                    if z in calls:
                        with lock:
                            calls[z].append(first)
                            started.add(z)
                            if len(started) == 2:
                                ready.set()
                            cancel = events[0]
                        if first == 0 and not cancel.wait(3):
                            raise AssertionError('prefetch cancellation was not signalled')
                    return original_pull(*args, **kwargs)

                def gpu(z, count):
                    self.assertEqual(live, 0)
                    return expected[z:z + count].copy()

                def consume(z, array):
                    if z == 0:
                        self.assertTrue(ready.wait(3))
                        if fail_sink:
                            raise failure
                    seen.extend(range(z, z + len(array)))
                    result[z:z + len(array)] = array

                stage = SimpleNamespace(device_index=0, max_block_depth=2,
                    projector=SimpleNamespace(), project=mock.Mock(side_effect=gpu), close=mock.Mock())
                log = io.StringIO()
                with ExitStack() as stack:
                    stack.enter_context(redirect_stdout(log))
                    stack.enter_context(mock.patch.object(sp, '_try_spherical_cuda_stage', side_effect=[None, stage]))
                    stack.enter_context(mock.patch.object(sp, 'spherical_cuda_backproject_enabled', return_value=True))
                    stack.enter_context(mock.patch.object(sp, 'spherical_output_bounds', return_value=None))
                    stack.enter_context(mock.patch.object(sp, '_spherical_block_schedule', return_value=(1, 3)))
                    stack.enter_context(mock.patch.object(sp, '_PULL_CHUNK_VOXELS', 11))
                    stack.enter_context(mock.patch.object(sp, '_CUDA_RECHECK_SLICES', 1))
                    stack.enter_context(mock.patch.object(sp, '_project_spherical_block', side_effect=block))
                    stack.enter_context(mock.patch.object(sp, '_pull_spherical_chunk', side_effect=pull))
                    if fail_sink:
                        with self.assertRaises(ValueError) as caught:
                            sp.backproject_spherical_volume_to_volume(self.source, self.view,
                                Path('unused.dat'), 'cancel', workers=3, sink_only=True,
                                projection_block_callback=consume)
                        self.assertIs(caught.exception, failure)
                    else:
                        sp.backproject_spherical_volume_to_volume(self.source, self.view,
                            Path('unused.dat'), 'cancel', workers=3, sink_only=True,
                            projection_block_callback=consume)
                self.assertEqual(live, 0)
                self.assertEqual(calls, {1: [0], 2: [0]})
                self.assertTrue(events and all(event is events[0] for event in events))
                self.assertTrue(events[0].is_set())
                if fail_sink:
                    stage.project.assert_not_called()
                    stage.close.assert_not_called()
                    self.assertFalse(seen)
                else:
                    self.assertEqual(seen, list(range(self.shape[0])))
                    np.testing.assert_array_equal(result, expected)
                    stage.close.assert_called_once()
                    self.assertIn('cpu_cancelled_blocks=2', log.getvalue())
                    self.assertIn('admission_attempts=2', log.getvalue())
                    self.assertIn('cpu_reader_drain_s=', log.getvalue())


if __name__ == '__main__':
    unittest.main()
