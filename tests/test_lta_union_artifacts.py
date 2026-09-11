from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from XTA.lta_union_artifacts import (
    LtaUnionWriter, UNION_ENCODING, iter_union_crops, logical_union_sha256,
    read_union_array, reduce_union_artifact_into_view, validate_union_artifact,
)


class UnionArtifactTests(unittest.TestCase):
    def test_out_of_order_chunks_pack_each_cropped_row_little_endian(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "union.bin"
            expected = np.zeros((1929, 5, 13), dtype=np.uint8)
            chunk = np.zeros((2, 5, 13), dtype=np.uint8)
            chunk[0, 1, 2] = chunk[0, 1, 10] = 1
            chunk[0, 3, 2] = chunk[0, 3, 10] = 1
            expected[100:102] = chunk
            earlier = np.zeros((1, 5, 13), dtype=bool)
            earlier[0, 4, 12] = True
            expected[7] = earlier[0]
            with LtaUnionWriter(path, shape=expected.shape, frame_start=0) as writer:
                writer.append_chunk(100, chunk)
                writer.append_chunk(7, earlier)
                writer.append_chunk(50, np.zeros((2, 5, 13), dtype=np.uint8))
                receipt = writer.finish()
            self.assertEqual(path.read_bytes(), b"\x01\x01\x00\x00\x01\x01\x01")
            self.assertEqual(receipt["size_bytes"], 7)
            self.assertEqual(receipt["foreground_pixels"], 5)
            self.assertEqual([row["frame_index"] for row in receipt["frames"]], [100, 7])
            self.assertEqual(receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            np.testing.assert_array_equal(read_union_array(receipt), expected)
            self.assertEqual(logical_union_sha256(receipt), hashlib.sha256(expected.tobytes()).hexdigest())
            self.assertEqual(len(list(iter_union_crops(receipt))), 2)

    def test_empty_full_depth_union_stores_no_payload_and_refuses_dense_reconstruction(self):
        with tempfile.TemporaryDirectory() as temp:
            with LtaUnionWriter(Path(temp) / "empty.bin", shape=(1929, 1008, 1008), frame_start=0) as writer:
                writer.append_chunk(300, np.zeros((1, 1008, 1008), dtype=np.uint8))
                receipt = writer.finish()
            self.assertEqual(receipt["size_bytes"], 0)
            self.assertEqual(receipt["frames"], [])
            self.assertEqual(receipt["sha256"], hashlib.sha256(b"").hexdigest())
            self.assertEqual(validate_union_artifact(receipt).shape, (1929, 1008, 1008))
            with self.assertRaisesRegex(ValueError, "byte cap"):
                read_union_array(receipt)

    def _small(self, root):
        masks = np.zeros((2, 4, 5), dtype=np.uint8)
        masks[0, 1, 2] = masks[0, 2, 3] = 1
        masks[1, 0, 0] = 1
        with LtaUnionWriter(root / "union.bin", shape=masks.shape, frame_start=3) as writer:
            writer.append_chunk(3, masks)
            receipt = writer.finish()
        return receipt, masks

    def test_verified_crop_reduction_preserves_existing_pixels_and_global_offsets(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, masks = self._small(Path(temp))
            target = np.zeros((8, 9, 12), dtype=np.uint8)
            target[3, 8, 11] = 1
            expected = target.copy()
            expected[3:5, 2:6, 4:9] |= masks
            audit = reduce_union_artifact_into_view(receipt, view_union=target,
                                                   tile_xyxy=(4, 2, 9, 6), frame_start=3, frame_stop=5)
            np.testing.assert_array_equal(target, expected)
            self.assertEqual(audit["encoding"], UNION_ENCODING)
            self.assertEqual(audit["logical_bytes"], 40)
            self.assertEqual(audit["indexed_frame_count"], 2)

    def test_malformed_metadata_is_rejected_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, _ = self._small(Path(temp))
            mutations = [
                lambda d: d.update(schema="unknown"),
                lambda d: d.update(omitted_frames="unknown"),
                lambda d: d.update(foreground_pixels=999),
                lambda d: d["frames"][0].update(offset_bytes=1),
                lambda d: d["frames"][0].update(crop_yx=[4, 0]),
                lambda d: d["frames"][0].update(row_stride_bytes=2),
                lambda d: d["frames"][1].update(frame_index=3),
                lambda d: d["frames"][0].update(frame_index=99),
            ]
            for mutate in mutations:
                bad = copy.deepcopy(receipt)
                mutate(bad)
                target = np.zeros((8, 9, 12), dtype=np.uint8)
                with self.subTest(descriptor=bad), self.assertRaises((ValueError, TypeError)):
                    reduce_union_artifact_into_view(bad, view_union=target,
                                                   tile_xyxy=(4, 2, 9, 6), frame_start=3, frame_stop=5)
                self.assertFalse(target.any())

    def test_digest_and_padding_corruption_fail_before_first_frame_is_applied(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, _ = self._small(Path(temp))
            path = Path(receipt["path"])
            original = path.read_bytes()
            # Opposite diagonal keeps tight bounds/count, exposing the digest gate.
            path.write_bytes(b"\x02\x01" + original[2:])
            target = np.zeros((8, 9, 12), dtype=np.uint8)
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                reduce_union_artifact_into_view(receipt, view_union=target,
                                               tile_xyxy=(4, 2, 9, 6), frame_start=3)
            self.assertFalse(target.any())
            # Corrupt only the final frame and update the digest: padding still rejects.
            padded = original[:-1] + b"\x81"
            path.write_bytes(padded)
            receipt["sha256"] = hashlib.sha256(padded).hexdigest()
            with self.assertRaisesRegex(ValueError, "padding"):
                reduce_union_artifact_into_view(receipt, view_union=target,
                                               tile_xyxy=(4, 2, 9, 6), frame_start=3)
            self.assertFalse(target.any())

    def test_legacy_raw_read_and_reduction_remain_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _compact, masks = self._small(root)
            path = root / "legacy.raw"
            masks.tofile(path)
            receipt = {"path": str(path), "shape": list(masks.shape),
                       "sha256": hashlib.sha256(masks.tobytes()).hexdigest(), "dtype": "uint8"}
            np.testing.assert_array_equal(read_union_array(receipt, frame_start=3), masks)
            target = np.zeros((8, 9, 12), dtype=np.uint8)
            reduce_union_artifact_into_view(receipt, view_union=target,
                                           tile_xyxy=(4, 2, 9, 6), frame_start=3, frame_stop=5)
            np.testing.assert_array_equal(target[3:5, 2:6, 4:9], masks)
            with self.assertRaisesRegex(ValueError, "frame stop"):
                reduce_union_artifact_into_view(receipt, view_union=target,
                                               tile_xyxy=(4, 2, 9, 6), frame_start=3, frame_stop=6)

    def test_writer_rejects_overlap_oversized_chunks_and_nonbinary_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "union.bin"
            writer = LtaUnionWriter(path, shape=(50, 2, 2), frame_start=0)
            writer.append_chunk(0, np.zeros((1, 2, 2), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                writer.append_chunk(0, np.zeros((1, 2, 2), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "thirty"):
                writer.append_chunk(1, np.zeros((31, 2, 2), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "zero and one"):
                writer.append_chunk(1, np.full((1, 2, 2), 2, dtype=np.uint8))
            writer.abort()
            self.assertFalse(path.exists())
            self.assertFalse(writer.stage.exists())


if __name__ == "__main__":
    unittest.main()
