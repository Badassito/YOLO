"""Exact component exports and bounded membership reads after bridge merge."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from XTA import interpolation, topology


class _RecordingMembership:
    def __init__(self, words: np.ndarray):
        self.words = words
        self.shape = words.shape
        self.dtype = words.dtype
        self.reads: list[tuple[object, int]] = []

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError('The exporter must not materialize the membership volume')

    def __getitem__(self, key: object) -> np.ndarray:
        result = self.words[key]
        self.reads.append((key, int(result.size)))
        return result


def _counts_and_bounds(words: np.ndarray, bit: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.count_nonzero(words & np.asarray(bit, dtype=words.dtype), axis=(1, 2)).astype(np.int64)
    bounds = np.zeros((words.shape[0], 4), dtype=np.int64)
    for z, plane in enumerate(words):
        yy, xx = np.nonzero(plane)
        if yy.size:
            # Deliberately conservative: covers other component bits and margin.
            bounds[z] = (
                max(0, int(yy.min()) - 1), max(0, int(xx.min()) - 1),
                min(words.shape[1], int(yy.max()) + 2), min(words.shape[2], int(xx.max()) + 2),
            )
    return counts, bounds


def _decode_store(path: Path) -> np.ndarray:
    store = interpolation.RawBBoxMaskStore.open(path, mmap_payload=True)
    try:
        return np.stack([store.decode_slice(z) for z in range(store.shape[0])])
    finally:
        store.close()


class InterpolationSparseExportTests(unittest.TestCase):
    def test_raw_and_packed_payloads_match_full_slice_oracle_at_every_word_width(self) -> None:
        for dtype, shift in ((np.uint8, 7), (np.uint16, 15), (np.uint32, 31), (np.uint64, 63)):
            for packed in (False, True):
                with self.subTest(dtype=dtype, packed=packed), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                    bit = 1 << shift
                    words = np.zeros((6, 48, 64), dtype=dtype)
                    words[0, :3, :5] |= np.asarray(bit, dtype=dtype)
                    words[2, 9:13, 12:18] |= np.asarray(bit, dtype=dtype)
                    words[2, 10, 15] |= np.asarray(1, dtype=dtype)
                    words[4, 44:48, 60:64] |= np.asarray(1, dtype=dtype)
                    # All component bits have already been cleared at pre-pass FG.
                    words[2, 10, 15] = 0
                    before = words.copy()
                    words.flags.writeable = False
                    counts, bounds = _counts_and_bounds(words, bit)
                    reader = _RecordingMembership(words)
                    expected = np.asarray(words & np.asarray(bit, dtype=dtype) != 0, dtype=np.uint8)

                    def encode(z: int) -> interpolation.RawBBoxSlicePayload:
                        result = interpolation._encode_component_membership_slice_payload(
                            z, reader, bit, added_by_slice=counts, rendered_paste_bboxes=bounds,
                            packbits_payload=packed,
                        )
                        reference = interpolation._encode_bool_mask_slice_payload(
                            z, expected[z], packbits_payload=packed,
                        )
                        self.assertEqual(result, reference)
                        return result

                    path = Path(tmp) / 'component.cvol'
                    stats = interpolation._write_raw_bbox_payload_store(
                        shape=words.shape, store_dir=path, encode_slice=encode,
                        format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT if packed else interpolation.CVOL_FORMAT,
                        desc='bounded component export', workers=2,
                    )
                    np.testing.assert_array_equal(_decode_store(path), expected)
                    np.testing.assert_array_equal(words, before)
                    expected_pixels = sum((int(b[2])-int(b[0]))*(int(b[3])-int(b[1]))
                                          for b, count in zip(bounds, counts) if count > 0)
                    self.assertEqual(sum(size for _key, size in reader.reads), expected_pixels)
                    self.assertEqual(len(reader.reads), int(np.count_nonzero(counts)))
                    self.assertLess(expected_pixels, words.size // 20)
                    self.assertEqual(stats['foreground_voxels'], int(expected.sum()))
                    self.assertEqual(stats['empty_slices'], 4)

    def test_zero_component_count_skips_nonempty_other_memberships_without_a_read(self) -> None:
        words = np.full((2, 13, 17), 2, dtype=np.uint8)
        reader = _RecordingMembership(words)
        result = interpolation._encode_component_membership_slice_payload(
            0, reader, 1, added_by_slice=np.zeros(2, dtype=np.int64), rendered_paste_bboxes=None,
        )
        self.assertTrue(result.is_empty)
        self.assertEqual(reader.reads, [])

    def test_invalid_metadata_and_incomplete_bounds_preserve_full_slice_export(self) -> None:
        words = np.zeros((3, 32, 48), dtype=np.uint16)
        words[1, 2:5, 3:6] = 1 << 10
        words[1, 25:27, 40:42] = 1 << 10
        counts, bounds = _counts_and_bounds(words, 1 << 10)
        expected = interpolation._encode_bool_mask_slice_payload(1, words[1] != 0)
        negative = counts.copy(); negative[1] = -1
        impossible = counts.copy(); impossible[1] = words.shape[1]*words.shape[2]+1
        inverted = bounds.copy(); inverted[1] = (5, 6, 2, 3)
        outside = bounds.copy(); outside[1] = (0, 0, 33, 48)
        narrow = bounds.copy(); narrow[1] = (2, 3, 5, 6)
        cases = [
            (None, bounds), (counts[:-1], bounds), (counts.astype(float), bounds),
            (negative, bounds), (impossible, bounds), (counts, None),
            (counts, bounds[:-1]), (counts, bounds.astype(float)),
            (counts, inverted), (counts, outside), (counts, narrow),
        ]
        for supplied_counts, supplied_bounds in cases:
            reader = _RecordingMembership(words)
            result = interpolation._encode_component_membership_slice_payload(
                1, reader, 1 << 10, added_by_slice=supplied_counts,
                rendered_paste_bboxes=supplied_bounds,
            )
            self.assertEqual(result, expected)
            self.assertEqual(reader.reads[-1][0], 1)
        # A valid but incomplete crop is detected by its count before replay.
        self.assertEqual(len(reader.reads), 1)  # Count exceeds narrow area: reject before reading.
        larger_narrow = bounds.copy(); larger_narrow[1] = (1, 2, 8, 10)
        reader = _RecordingMembership(words)
        result = interpolation._encode_component_membership_slice_payload(
            1, reader, 1 << 10, added_by_slice=counts, rendered_paste_bboxes=larger_narrow,
        )
        self.assertEqual(result, expected)
        self.assertEqual(len(reader.reads), 2)

    def test_actual_merge_and_export_cover_component_word_boundaries_and_empty_combinations(self) -> None:
        # Discovery is fixed so the real painter, clipping, membership layout,
        # pre-pass differencing and CVOL exporter can be compared deterministically.
        products = ((1, 8), (1, 9), (2, 8), (1, 17), (4, 8), (3, 11), (8, 8), (5, 13), (3, 43))
        for walk_count, candidate_count in products:
            total = walk_count*candidate_count
            with self.subTest(components=total), tempfile.TemporaryDirectory() as tmp, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                shape = (3, 9, 11)
                mask = np.zeros(shape, dtype=np.uint8)
                mask[0, 4, 5] = mask[2, 4, 5] = mask[1, 0, 0] = 1
                before = mask.copy()
                labels = np.zeros(shape, dtype=np.uint32)
                labels[0, 4, 5] = 1; labels[2, 4, 5] = 2
                plans = []
                expected = {}
                section = np.ones((3, 3), dtype=bool)
                sdf = np.ones((3, 3), dtype=np.float32)
                for index in range(total):
                    key = (index // candidate_count + 1, index % candidate_count + 1)
                    expected[key] = np.zeros(shape, dtype=np.uint8)
                    if index == 2:  # One unpainted combination must still have a store/header.
                        continue
                    y, x = ((0, 0), (4, 5), (8, 10))[index % 3]
                    plan = interpolation.SliceBridgeRenderPlan(
                        source_label=1, target_label=2, source_point=(0, y, x), target_point=(2, y, x),
                        source_anchor=(y, x), target_anchor=(y, x), steps=2, sign=1,
                        num_slices=shape[0], sdf0=sdf, sdf1=sdf,
                        interpolation_walk_back_index=key[0], interpolation_candidate_index=key[1],
                        cached_sections=[None, section, None],
                    )
                    plans.extend((plan, plan))  # Same component overlap must not double-count.
                    expected[key][1, max(0,y-1):min(shape[1],y+2), max(0,x-1):min(shape[2],x+2)] = 1
                    expected[key][before != 0] = 0
                result = interpolation.SliceSeedBridgePlanResult(
                    candidate_connections=len(plans), accepted_connections=len(plans), plans=plans,
                )
                seed = interpolation.SliceEndpointSeed(label=1, point=(0,4,5), direction_sign=1)
                root = Path(tmp)
                with (
                    mock.patch.object(topology, 'interpolation_skip_compact_relabel_enabled', return_value=False),
                    mock.patch.object(topology, 'label_foreground_volume_streaming', return_value=(labels, 2, [])),
                    mock.patch.object(interpolation, '_build_slice_endpoint_seeds', return_value=([seed], 1)),
                    mock.patch.object(interpolation, '_plan_slice_seed_bridges', return_value=result),
                    mock.patch.object(interpolation, 'should_use_in_memory_workspace', return_value=True),
                    mock.patch.object(interpolation, 'create_cuda_interpolation_renderer', return_value=(None, 'CPU test')),
                    mock.patch.object(interpolation, 'gpu_interpolation_required', return_value=False),
                ):
                    stats = interpolation.interpolate_view_volume_pass_inplace(
                        mask_mm=mask, work_dir=root/'work', pass_tag='pass1', max_slice_distance=2,
                        search_angle_deg=15, interpolation_walk_back=walk_count,
                        interpolation_candidates=candidate_count, interpolate_min_radius=0,
                        keep_temp=False, workers=2, bridge_component_dir=root/'components',
                    )
                entries = stats['bridge_component_deltas']
                self.assertEqual(len(entries), total)
                expected_word_count = 1 if total <= 64 else (total+63)//64
                self.assertEqual(stats['bridge_component_render_word_count'], expected_word_count)
                united = np.zeros(shape, dtype=np.uint8)
                for entry in entries:
                    key = (entry['walk_back_index'], entry['candidate_index'])
                    decoded = _decode_store(Path(entry['path']))
                    np.testing.assert_array_equal(decoded, expected[key])
                    self.assertEqual(entry['added_voxels'], int(expected[key].sum()))
                    united |= decoded
                np.testing.assert_array_equal(mask, before | united)
                self.assertEqual(stats['added_voxels'], int(united.sum()))
                self.assertFalse((root/'components'/'_membership').exists())


if __name__ == '__main__':
    unittest.main()
