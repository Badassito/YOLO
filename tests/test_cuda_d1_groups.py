from __future__ import annotations

import unittest

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA.cuda_d1 import (  # noqa: E402
    D1PartialBitsetArtifact,
    _normalize_slice_ranges,
    d1_reduce_word_arrays,
)


class D1GroupProtocolTests(unittest.TestCase):
    def test_partial_artifact_round_trip_preserves_opaque_ipc_handle(self) -> None:
        artifact = D1PartialBitsetArtifact(
            group_id="d1g-model-view",
            model_name="model",
            view_name="view",
            participant_rank=1,
            participant_worker_id=3,
            output_shape=(3, 5, 7),
            word_count=4,
            covered_ranges=((2, 4), (8, 10)),
            lease_token="lease-1",
            transport="cuda_ipc",
            ipc_handle=b"opaque-handle",
        )

        restored = D1PartialBitsetArtifact.from_payload(artifact.to_payload())

        self.assertEqual(restored, artifact)
        self.assertEqual(restored.byte_count, 16)

    def test_partial_artifact_rejects_invalid_shape_words_and_transport(self) -> None:
        base = dict(
            group_id="group",
            model_name="model",
            view_name="view",
            participant_rank=0,
            participant_worker_id=0,
            output_shape=(1, 1, 33),
            word_count=2,
            covered_ranges=((0, 1),),
            lease_token="lease",
            transport="cuda_ipc",
            ipc_handle=b"handle",
        )
        with self.assertRaisesRegex(ValueError, "word count"):
            D1PartialBitsetArtifact(**{**base, "word_count": 1})
        with self.assertRaisesRegex(ValueError, "missing its memory handle"):
            D1PartialBitsetArtifact(**{**base, "ipc_handle": b""})
        with self.assertRaisesRegex(ValueError, "unsupported D1 partial transport"):
            D1PartialBitsetArtifact(**{**base, "transport": "peer-magic"})

    def test_slice_range_normalization_coalesces_and_rejects_overlap(self) -> None:
        self.assertEqual(
            _normalize_slice_ranges(((4, 8), (0, 2), (2, 4)), total_slices=8),
            ((0, 8),),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            _normalize_slice_ranges(((0, 4), (3, 8)), total_slices=8)
        with self.assertRaisesRegex(ValueError, "exceeds depth"):
            _normalize_slice_ranges(((0, 9),), total_slices=8)

    def test_tree_or_matches_reference_for_odd_and_power_of_two_groups(self) -> None:
        rng = np.random.default_rng(20260831)
        for count in (1, 2, 3, 4, 5, 8):
            with self.subTest(count=count):
                partials = [
                    rng.integers(0, 2 ** 32, size=37, dtype=np.uint32)
                    for _ in range(count)
                ]
                originals = [partial.copy() for partial in partials]
                expected = np.bitwise_or.reduce(np.stack(partials, axis=0), axis=0)

                actual = d1_reduce_word_arrays(partials)

                np.testing.assert_array_equal(actual, expected)
                for partial, original in zip(partials, originals):
                    np.testing.assert_array_equal(partial, original)

    def test_tree_or_can_write_a_caller_owned_destination(self) -> None:
        partials = [
            np.asarray([0x1, 0x10, 0x100], dtype=np.uint32),
            np.asarray([0x2, 0x20, 0x200], dtype=np.uint32),
            np.asarray([0x4, 0x40, 0x400], dtype=np.uint32),
        ]
        destination = np.empty((3,), dtype=np.uint32)

        returned = d1_reduce_word_arrays(partials, destination=destination)

        self.assertIs(returned, destination)
        np.testing.assert_array_equal(
            destination, np.asarray([0x7, 0x70, 0x700], dtype=np.uint32),
        )

    def test_tree_or_rejects_dtype_and_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            d1_reduce_word_arrays(())
        with self.assertRaisesRegex(TypeError, "not uint32"):
            d1_reduce_word_arrays((np.zeros((2,), dtype=np.int32),))
        with self.assertRaisesRegex(ValueError, "shape"):
            d1_reduce_word_arrays((
                np.zeros((2,), dtype=np.uint32),
                np.zeros((3,), dtype=np.uint32),
            ))


if __name__ == "__main__":
    unittest.main()
