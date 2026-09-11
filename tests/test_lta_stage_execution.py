from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import test_lta_execution as fixtures
from XTA.lta_postprocessing import LtaFinalizationOperations


def read_checkpoint(path: Path) -> np.ndarray:
    header, data = path.read_bytes().split(b'\n\n', 1)
    fields = dict(line.split(': ', 1) for line in header.decode().splitlines() if ': ' in line)
    if fields['encoding'] == 'gzip':
        data = gzip.decompress(data)
    xyz = tuple(int(value) for value in fields['sizes'].split())
    return np.frombuffer(data, dtype=np.uint8).reshape(tuple(reversed(xyz)))


def operations(*, fail_keep=False):
    def unexpected(*_args, **_kwargs):
        raise AssertionError('unrequested filter')

    def keep(volume, count, *_args, **_kwargs):
        from scipy.ndimage import label
        if fail_keep:
            raise RuntimeError('injected later-filter failure')
        labels, components = label(volume, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(labels.ravel(), minlength=components + 1)
        sizes[0] = 0
        kept = int(np.argmax(sizes))
        volume[:] = labels == kept
        return {'kept_objects': count, 'backend': 'independent_cpu_test'}

    return LtaFinalizationOperations(
        allocate_workspace_array=lambda *, shape, dtype, **_kwargs: np.zeros(shape, dtype=dtype),
        close_volume=lambda _volume: None,
        fill_3d_voids=unexpected,
        apply_gaussian_smoothing=unexpected,
        try_gpu_keep_objects=lambda *_args, **_kwargs: None,
        apply_cpu_keep_objects=keep,
    )


class LtaStageExecutionTests(unittest.TestCase):
    def test_default_saved_union_and_keep_stage_match_exported_masks_and_receipts(self):
        helper = fixtures.LtaProductionExecutionTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _pool, final_bytes = helper._run(
                root, (0,), postprocessing_override={'keep_objects': 1},
                finalization_operations=operations(),
            )
            manifest = json.loads(result.manifest_path.read_text())
            checkpoints = manifest['postprocessing_checkpoints']
            self.assertEqual([item['stage'] for item in checkpoints], [
                'before_postprocessing', 'after_keep_objects',
            ])
            masks = []
            for record in checkpoints:
                path = Path(record['path'])
                self.assertEqual(path.parent, result.manifest_path.parent)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['sha256'])
                self.assertEqual(record['recomposition_op'], 'select')
                self.assertFalse(record['metadata']['protected_foreground_restore_applied'])
                masks.append(read_checkpoint(path))
            before, after = masks
            self.assertEqual(before.shape, (4, 4, 6))
            self.assertTrue(np.all(after <= before))
            self.assertGreater(int(before.sum()), int(after.sum()))
            final = np.frombuffer(final_bytes, np.uint8).reshape(before.shape)
            self.assertTrue(np.all(after <= final))
            index_path = next(result.manifest_path.parent.glob('*_Postprocessing_checkpoints.json'))
            index = json.loads(index_path.read_text())
            self.assertTrue(index['checkpoint_sequence_complete'])
            self.assertEqual(index['requested_postprocessing'], {'keep_objects': 1})
            self.assertEqual(index['run_completion_marker'], 'manifest.json')
            self.assertEqual(index['checkpoints'], checkpoints)
            layer_ids = [item['layer_id'] for item in manifest['layers']]
            self.assertLess(layer_ids.index(checkpoints[-1]['layer_id']), layer_ids.index('global_final_output'))

    def test_completed_union_survives_later_filter_failure_without_complete_run_manifest(self):
        helper = fixtures.LtaProductionExecutionTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, 'later-filter failure'):
                helper._run(
                    root, (0,), postprocessing_override={'keep_objects': 1},
                    finalization_operations=operations(fail_keep=True),
                )
            output = root / 'output'
            saved = list(output.glob('*_Global_union_before_postprocessing.seg.nrrd'))
            self.assertEqual(len(saved), 1)
            self.assertGreater(int(read_checkpoint(saved[0]).sum()), 0)
            self.assertFalse((output / 'manifest.json').exists())
            self.assertFalse(list(output.glob('*_Global_final_output.seg.nrrd')))
            self.assertFalse(list(output.glob('*_Global_after_keep_objects.seg.nrrd')))
            index = json.loads(next(output.glob('*_Postprocessing_checkpoints.json')).read_text())
            self.assertFalse(index['checkpoint_sequence_complete'])
            self.assertEqual(index['requested_postprocessing'], {'keep_objects': 1})
            self.assertEqual(len(index['checkpoints']), 1)


if __name__ == '__main__':
    unittest.main()
