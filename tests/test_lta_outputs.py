from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from XTA.lta_outputs import (
    LTA_MANIFEST_SCHEMA,
    LtaArtifactReceipt,
    LtaLayerRecord,
    LtaPublicationReceipt,
    compose_terminal_union,
    write_complete_lta_manifest,
)


class LtaOutputTests(unittest.TestCase):
    def test_role_aware_terminal_composition_is_ordered_and_binary(self) -> None:
        first = np.zeros((2, 3, 4), dtype=np.uint8)
        first[0, 1, 1] = 1
        bridge = np.zeros_like(first)
        bridge[1, 1, 2] = 9
        subtract = np.zeros_like(first)
        subtract[0, 1, 1] = 1
        checkpoint = np.zeros_like(first)
        checkpoint[0, 0, 0] = 1

        result = compose_terminal_union(
            (
                LtaLayerRecord("prediction", "union", "fullframe_sam", first),
                LtaLayerRecord("bridge", "union", "bridge", bridge),
                LtaLayerRecord(
                    "subtract",
                    "subtract_from_previous_checkpoint",
                    "audit_delta",
                    subtract,
                ),
                LtaLayerRecord("diagnostic", "none", "diagnostic", np.ones_like(first)),
                LtaLayerRecord("checkpoint", "select", "global_checkpoint", checkpoint),
            )
        )

        np.testing.assert_array_equal(result, checkpoint)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(result.flags.c_contiguous)

    def test_terminal_composition_rejects_shape_and_identity_drift(self) -> None:
        first = LtaLayerRecord("same", "union", "prediction", np.zeros((1, 2, 3)))
        duplicate = LtaLayerRecord("same", "union", "bridge", np.zeros((1, 2, 3)))
        mismatch = LtaLayerRecord("other", "union", "bridge", np.zeros((1, 2, 4)))

        with self.assertRaisesRegex(ValueError, "duplicate LTA layer_id"):
            compose_terminal_union((first, duplicate))
        with self.assertRaisesRegex(ValueError, "does not match"):
            compose_terminal_union((first, mismatch))

    def test_manifest_is_atomic_complete_and_caller_cannot_override_ownership(self) -> None:
        volume = np.zeros((1, 2, 3), dtype=np.uint8)
        layer = LtaLayerRecord(
            "transverse__tta_a0_fullframe_sam",
            "union",
            "fullframe_sam",
            volume,
            physical_view_id="transverse",
            runtime_view_id="transverse__tta_a0",
            tta_angle_deg=0.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "layer.seg.nrrd"
            artifact.write_bytes(b"settled-layer")
            artifact_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            path = Path(temp_dir) / "manifest.json"
            written = write_complete_lta_manifest(
                path,
                version="19.0.0",
                command=("xta", "--mode", "lta"),
                layers=(layer,),
                publication_receipt=LtaPublicationReceipt(
                    artifacts=(
                        LtaArtifactReceipt(
                            name="layer",
                            path=artifact,
                            sha256=artifact_digest,
                        ),
                    ),
                    terminal_union_shape_tyx=(1, 2, 3),
                    terminal_union_foreground_voxels=0,
                    source_revalidated=True,
                    model_revalidated=True,
                    layers_settled=True,
                ),
                payload={"status": "bad", "mode": "bad", "custom": 7},
            )
            record = json.loads(written.read_text(encoding="utf-8"))
            leftovers = list(path.parent.glob("*.assembling"))

        self.assertEqual(record["schema"], LTA_MANIFEST_SCHEMA)
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["mode"], "lta")
        self.assertEqual(record["pipeline_version"], "19.0.0")
        self.assertEqual(record["layers"][0]["shape_tyx"], [1, 2, 3])
        self.assertEqual(record["custom"], 7)
        self.assertTrue(record["publication_integrity"]["source_revalidated"])
        self.assertEqual(leftovers, [])

    def test_complete_manifest_rejects_unsettled_or_changed_artifacts(self) -> None:
        volume = np.zeros((1, 1, 1), dtype=np.uint8)
        layer = LtaLayerRecord("layer", "union", "sam", volume)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "layer.nrrd"
            artifact.write_bytes(b"first")
            wrong_digest = hashlib.sha256(b"other").hexdigest()
            with self.assertRaisesRegex(ValueError, "digest changed"):
                write_complete_lta_manifest(
                    root / "manifest.json",
                    version="19.0.0",
                    command=(),
                    layers=(layer,),
                    publication_receipt=LtaPublicationReceipt(
                        artifacts=(LtaArtifactReceipt("layer", artifact, wrong_digest),),
                        terminal_union_shape_tyx=(1, 1, 1),
                        terminal_union_foreground_voxels=0,
                        source_revalidated=True,
                        model_revalidated=True,
                        layers_settled=True,
                    ),
                    payload={},
                )


if __name__ == "__main__":
    unittest.main()
