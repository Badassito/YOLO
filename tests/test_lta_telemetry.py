from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from XTA.lta_telemetry import LtaExecutionTrace, lta_source_fingerprint


class LtaTelemetryTests(unittest.TestCase):
    def test_phases_preserve_failures_and_carry_mutable_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = LtaExecutionTrace(temporary, 'coordinator')
            with trace.phase('reduce', work_id='window-1') as fields:
                fields['stored_bytes'] = 128
            with self.assertRaisesRegex(ValueError, 'bad shape'):
                with trace.phase('prepare'):
                    raise ValueError('bad shape')
            trace.close()
            records = [json.loads(line) for line in trace.path.read_text().splitlines()]
            self.assertEqual([record['event'] for record in records], ['phase_start', 'phase_end'] * 2)
            self.assertEqual(records[1]['stored_bytes'], 128)
            self.assertEqual(records[3]['status'], 'failed')
            self.assertEqual(records[3]['error_type'], 'ValueError')
            self.assertGreaterEqual(records[1]['wall_seconds'], 0)
            from tools.lta_trace_summary import summarize
            summary = summarize(Path(temporary))
            stream = summary['streams'][0]
            self.assertEqual(stream['phases']['reduce']['calls'], 1)
            self.assertEqual(stream['phases']['prepare']['failures'], 1)
            self.assertEqual(stream['unfinished_phases'], [])

    def test_start_is_readable_while_a_long_phase_is_still_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = LtaExecutionTrace(temporary, 'worker')
            with trace.phase('waiting_for_seed'):
                from tools.lta_trace_summary import summarize
                pending = summarize(Path(temporary))['streams'][0]['unfinished_phases']
                self.assertEqual([item['phase'] for item in pending], ['waiting_for_seed'])
            trace.close()

    def test_disabled_trace_has_no_output_and_source_fingerprint_is_stable(self):
        trace = LtaExecutionTrace(None, 'disabled')
        with trace.phase('work'):
            trace.event('done')
        trace.close()
        self.assertIsNone(trace.path)
        first = lta_source_fingerprint()
        self.assertEqual(first, lta_source_fingerprint())
        self.assertIn('lta_worker_adapter.py', first['files'])
        self.assertIn('media.py', first['files'])
        self.assertEqual(len(first['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
