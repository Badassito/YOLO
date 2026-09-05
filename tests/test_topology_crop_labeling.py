"""Automatic sparse crop-CPU admission and retained labeling contracts."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from XTA import topology


class _CpuLabelingSelected(Exception):
    pass


class _GpuLabelingSelected(Exception):
    pass


class TopologyCropLabelingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((3, 20, 24), dtype=np.uint8)
        self.present = np.array([False, True, True])
        self.boxes = np.array([[0, 0, 0, 0], [2, 5, 4, 8], [3, 6, 5, 9]], dtype=np.int64)

    def test_metadata_validation_reads_no_mask_pixels_and_accepts_inactive_empty_rows(self):
        class MetadataOnlyVolume:
            shape = (3, 20, 24)

            def __array__(self, *args, **kwargs):
                raise AssertionError('metadata admission read mask pixels')

        boxes = self.boxes.copy()
        boxes[0] = -99
        self.assertTrue(topology._slice_label_metadata_is_bounded(MetadataOnlyVolume(), self.present, boxes))
        self.assertTrue(topology._slice_label_metadata_is_bounded(
            MetadataOnlyVolume(), np.zeros(3, dtype=bool), np.full((3, 4), -99, dtype=np.int64),
        ))

    def test_missing_malformed_and_out_of_bounds_metadata_are_not_eligible(self):
        invalid = [
            (None, self.boxes), (self.present, None),
            (self.present[:2], self.boxes), (self.present, self.boxes[:, :3]),
            (self.present.astype(np.uint8), self.boxes),
            (self.present, self.boxes.astype(np.float32)),
        ]
        for column, value in ((0, -1), (1, 21), (2, -1), (3, 25), (1, 2), (3, 4)):
            boxes = self.boxes.copy()
            boxes[1, column] = value
            invalid.append((self.present, boxes))
        for present, boxes in invalid:
            with self.subTest(present=present, boxes=boxes):
                self.assertFalse(topology._slice_label_metadata_is_bounded(self.mask, present, boxes))
        self.assertFalse(topology._slice_label_metadata_is_bounded(
            types.SimpleNamespace(shape=(3, 20)), self.present, self.boxes,
        ))

    def assert_selection(self, expected, *, metadata=True, cpu_policy=True, **overrides):
        kwargs = dict(
            prefer_memory=True, reserve_bytes=0, workers=1, compact_relabel=False,
            sparse_local_labels=True, known_slice_any=self.present if metadata else None,
            known_slice_bboxes=self.boxes if metadata else None,
        )
        kwargs.update(overrides)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=True), \
                mock.patch.object(topology, 'topology_sparse_cpu_labeling_enabled', return_value=cpu_policy), \
                mock.patch.object(topology, 'topology_sparse_cpu_max_coverage', return_value=0.5), \
                mock.patch.object(topology, 'should_use_in_memory_workspace', return_value=True), \
                mock.patch.object(topology, 'numa_interleave_memory'), \
                mock.patch.object(topology, '_try_label_slices_stage_a_gpu', side_effect=_GpuLabelingSelected), \
                mock.patch.object(topology, 'parallel_for_indices_chunked', side_effect=_CpuLabelingSelected), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(expected):
            topology.label_foreground_volume_streaming(self.mask, Path(directory) / 'labels', **kwargs)

    def test_valid_sparse_interpolation_metadata_automatically_selects_cpu(self):
        self.assert_selection(_CpuLabelingSelected)
        self.assert_selection(
            _CpuLabelingSelected,
            known_slice_any=np.zeros(3, dtype=bool), known_slice_bboxes=np.zeros((3, 4), dtype=np.int64),
        )

    def test_missing_invalid_or_high_coverage_metadata_retains_gpu_admission(self):
        self.assert_selection(_GpuLabelingSelected, metadata=False)
        boxes = self.boxes.copy()
        boxes[1, 1] = 21
        self.assert_selection(_GpuLabelingSelected, known_slice_bboxes=boxes)
        self.assert_selection(_GpuLabelingSelected, known_slice_bboxes=self.boxes.astype(float))
        self.assert_selection(_GpuLabelingSelected, known_slice_any=self.present.astype(np.uint8))
        self.assert_selection(
            _GpuLabelingSelected, known_slice_any=np.ones(3, dtype=bool),
            known_slice_bboxes=np.tile([0, 20, 0, 24], (3, 1)),
        )

    def test_explicit_cpu_preference_remains_available_outside_sparse_interpolation(self):
        self.assert_selection(_GpuLabelingSelected, sparse_local_labels=False)
        self.assert_selection(_GpuLabelingSelected, cpu_policy=False)
        self.assert_selection(
            _CpuLabelingSelected, sparse_local_labels=False, compact_relabel=True,
            prefer_crop_bounded_cpu_labeling=True,
        )

    def test_existing_paired_metadata_and_shape_errors_are_preserved(self):
        self.assert_selection(ValueError, known_slice_bboxes=None)
        self.assert_selection(ValueError, known_slice_bboxes=np.zeros((2, 4), dtype=np.int64))


@unittest.skipUnless(bool(getattr(topology.cv2, '__file__', None)), 'Native OpenCV unavailable')
class TopologyCropLabelingNumericalTests(unittest.TestCase):
    def test_real_sparse_topology_matches_independent_26_connected_graph_across_slabs_and_mirrored_seam(self):
        rng = np.random.default_rng(23003)
        mask = (rng.random((10, 11, 13)) < 0.065).astype(np.uint8)
        mask[0, 2, 3] = 1
        mask[-1, 2, -4] = 1
        mask[3:6, 7:9, 8:10] = 1  # foreground straddles a slab boundary
        present = np.any(mask, axis=(1, 2))
        boxes = np.zeros((len(mask), 4), dtype=np.int64)
        for z in np.flatnonzero(present):
            yy, xx = np.nonzero(mask[z])
            boxes[z] = (yy.min(), yy.max() + 1, xx.min(), xx.max() + 1)

        def normalize(labels):
            mapping = {0: 0}
            result = np.zeros(labels.shape, dtype=np.uint32)
            for flat_index, value in enumerate(labels.flat):
                value = int(value)
                if value not in mapping:
                    mapping[value] = len(mapping)
                result.flat[flat_index] = mapping[value]
            return result

        def graph(wrap):
            labels = np.zeros(mask.shape, dtype=np.uint32)
            label = 0
            depth, height, width = mask.shape
            for z, y, x in np.argwhere(mask):
                z, y, x = int(z), int(y), int(x)
                if labels[z, y, x]:
                    continue
                label += 1
                labels[z, y, x] = label
                stack = [(z, y, x)]
                while stack:
                    az, ay, ax = stack.pop()
                    for dz in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                bz, by, bx = az + dz, ay + dy, ax + dx
                                if wrap and (bz < 0 or bz >= depth):
                                    bz %= depth
                                    bx = width - 1 - bx
                                if not (0 <= bz < depth and 0 <= by < height and 0 <= bx < width):
                                    continue
                                if mask[bz, by, bx] and not labels[bz, by, bx]:
                                    labels[bz, by, bx] = label
                                    stack.append((bz, by, bx))
            return labels

        for wrap in (False, True):
            with self.subTest(wrap=wrap), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.dict(os.environ, {'YOLO_TTA_TOPOLOGY_SLAB_SLICES': '4'}), \
                    mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                stats = {}
                store, count, paths = topology.label_foreground_volume_streaming(
                    mask, Path(directory) / 'graph', prefer_memory=True, reserve_bytes=0,
                    wrap_axis=wrap, workers=3, compact_relabel=False, component_stats_out=stats,
                    known_slice_any=present, known_slice_bboxes=boxes, sparse_local_labels=True,
                    prefer_crop_bounded_cpu_labeling=False,
                )
                luts = stats['slice_local_luts']
                actual = np.stack([luts.lut_for(z)[np.asarray(store[z])] for z in range(len(mask))])
                expected = graph(wrap)
                self.assertEqual(count, int(expected.max()))
                np.testing.assert_array_equal(normalize(actual), normalize(expected))
                self.assertGreaterEqual(stats['topology_slab_count'], 3)


if __name__ == '__main__':
    unittest.main()
