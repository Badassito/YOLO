"""Windows affinity validates every live thread without weakening Linux behavior."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import runtime


class FakeProcess:
    def __init__(self):
        self.mask = 15
        self.masks = {11: 15, 22: 2}
        self.setter_calls = []
        self.fail_set = False
        self.clip_set = False
        self.birth_on_set = False

    def threads(self):
        return [SimpleNamespace(id=tid) for tid in self.masks]

    def cpu_affinity(self, values=None):
        if values is None:
            return [c for c in range(4) if self.mask & (1 << c)]
        self.setter_calls.append(tuple(values))
        if self.fail_set:
            self.fail_set = False
            raise PermissionError('denied')
        self.mask = sum(1 << c for c in values)
        if self.clip_set and len(self.setter_calls) == 1:
            self.mask = 1
        self.masks = {tid: self.mask for tid in self.masks}
        if self.birth_on_set:
            self.masks[33] = self.mask


class FakeApi:
    def __init__(self, process):
        self.process = process
        self.system = 15
        self.denied = None
        self.denied_after_set = None
        self.exit_on_query = None
        self.multi_group = False

    def process_masks(self):
        if self.multi_group:
            raise OSError('multiple processor groups')
        return self.process.mask, self.system

    def thread_mask(self, tid):
        if tid == self.exit_on_query:
            self.process.masks.pop(tid, None)
            self.exit_on_query = None
            raise OSError('thread exited')
        if tid == self.denied or (self.process.setter_calls and tid == self.denied_after_set):
            raise PermissionError('live thread denied')
        return self.process.masks[tid], 0

    def restore_thread_mask(self, tid, mask):
        if tid in self.process.masks:
            self.process.masks[tid] = mask


class WindowsAffinityContractTests(unittest.TestCase):
    def setUp(self):
        self.process = FakeProcess()
        self.api = FakeApi(self.process)

    def apply(self, values):
        return runtime._windows_setaffinity_all_threads(values, _process=self.process, _api=self.api)

    def test_process_mask_covers_existing_narrow_and_new_threads(self):
        self.process.birth_on_set = True
        self.assertTrue(self.apply([0, 2, 2]))
        self.assertEqual(self.process.mask, 5)
        self.assertEqual(self.process.masks, {11: 5, 22: 5, 33: 5})

    def test_invalid_or_system_disallowed_masks_never_mutate(self):
        self.api.system = 11  # CPU 2 is absent even though CPU 3 exists.
        for values in ([], [-1], [4], [2], [1.2], ['1']):
            with self.subTest(values=values):
                self.assertFalse(self.apply(values))
        self.assertFalse(self.process.setter_calls)

    def test_denied_live_thread_is_not_reported_as_partial_success(self):
        self.api.denied = 22
        self.assertFalse(self.apply([0]))
        self.assertFalse(self.process.setter_calls)

    def test_verification_failure_restores_process_and_original_thread_masks(self):
        self.api.denied_after_set = 22
        self.assertFalse(self.apply([0, 2]))
        self.assertEqual(self.process.mask, 15)
        self.assertEqual(self.process.masks, {11: 15, 22: 2})

    def test_silent_clipping_and_set_failure_return_false(self):
        self.process.clip_set = True
        self.assertFalse(self.apply([0, 2]))
        self.assertEqual(self.process.mask, 15)
        self.process.fail_set = True
        self.assertFalse(self.apply([1]))
        self.assertEqual(self.process.masks, {11: 15, 22: 2})

    def test_exited_thread_is_skipped_but_processor_groups_are_not_guessed(self):
        self.api.exit_on_query = 22
        self.assertTrue(self.apply([0, 2]))
        self.api.multi_group = True
        before = list(self.process.setter_calls)
        self.assertFalse(self.apply([0]))
        self.assertEqual(self.process.setter_calls, before)

    def test_dispatch_keeps_linux_per_thread_calls(self):
        fake_os = SimpleNamespace(
            name='posix', listdir=lambda _: ['42', '41'], sched_setaffinity=mock.Mock(),
        )
        with mock.patch.object(runtime, 'os', fake_os), mock.patch.object(runtime, '_windows_setaffinity_all_threads') as windows:
            self.assertTrue(runtime._sched_setaffinity_all_threads([1, 3]))
        windows.assert_not_called()
        self.assertEqual(fake_os.sched_setaffinity.call_args_list, [mock.call(41, {1, 3}), mock.call(42, {1, 3})])

    def test_windows_dispatches_to_verified_process_path(self):
        with mock.patch.object(runtime, 'os', SimpleNamespace(name='nt')), \
                mock.patch.object(runtime, '_windows_setaffinity_all_threads', return_value=True) as windows:
            self.assertTrue(runtime._sched_setaffinity_all_threads([2]))
        windows.assert_called_once_with([2])


@unittest.skipUnless(os.name == 'nt', 'Native Windows affinity smoke')
class WindowsAffinityNativeTests(unittest.TestCase):
    def test_temporary_subprocess_restricts_widens_and_restores_all_threads(self):
        code = r'''
import json, threading, psutil
from XTA.runtime import _WindowsAffinityApi, _sched_setaffinity_all_threads
p = psutil.Process()
api = _WindowsAffinityApi()
original = p.cpu_affinity()
mask = sum(1 << c for c in original)
stop = threading.Event()
ready = [threading.Event() for _ in range(4)]
threads = []
def run(event):
    event.set()
    stop.wait()
def start(event):
    t = threading.Thread(target=run, args=(event,))
    threads.append(t)
    t.start()
    assert event.wait(5)
def verify(cpus):
    expected = sum(1 << c for c in cpus)
    assert p.cpu_affinity() == sorted(cpus)
    tids = [t.id for t in p.threads()]
    assert tids and all(api.thread_mask(tid) == (expected, 0) for tid in tids)
    return len(tids)
try:
    for event in ready[:3]:
        start(event)
    api.restore_thread_mask(threads[0].native_id, 1 << original[0])
    target = original[-min(2, len(original)):]
    assert _sched_setaffinity_all_threads(target)
    checked = verify(target)
    start(ready[3])
    assert api.thread_mask(threads[3].native_id) == (sum(1 << c for c in target), 0)
    assert _sched_setaffinity_all_threads(original)
    checked_after = verify(original)
finally:
    assert _sched_setaffinity_all_threads(original)
    stop.set()
    for thread in threads:
        thread.join(5)
    assert p.cpu_affinity() == original
print(json.dumps({'original_cpus': original, 'restricted_cpus': target,
                  'verified_threads': checked, 'verified_threads_after_widen': checked_after,
                  'restored': True}))
'''
        completed = subprocess.run(
            [sys.executable, '-B', '-c', code], cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertTrue(result['restored'])
        self.assertGreaterEqual(result['verified_threads'], 4)
        print('Windows affinity smoke:', json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
