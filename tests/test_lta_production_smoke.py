from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from XTA.lta_propagation import LtaMaskSeed, write_seed_artifact
from XTA.lta_tile_tracking import LtaLineageId


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "lta_production_smoke.py"
spec = importlib.util.spec_from_file_location("lta_production_smoke", TOOL_PATH)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class ProductionSmokeTests(unittest.TestCase):
    def _fixture(self, root: Path):
        cache = root / "cache.raw"
        np.zeros((59, 4, 4), dtype=np.uint8).tofile(cache)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        seed = LtaMaskSeed(
            lineage=LtaLineageId("fixture", "transverse", "runtime", "object"),
            frame_index=19,
            object_id=0,
            mask=mask,
        )
        artifact = write_seed_artifact(root / "seed.npz", (seed,))
        plan, seeds = tool.prepare_fixture(
            cache_path=cache,
            cache_shape=(59, 4, 4),
            seed_path=artifact.path,
            output_dir=root / "output",
            require_both_dogfood=True,
        )
        return plan, seeds

    def _manifest(self, root: Path, plan, seeds):
        union = np.zeros((59, 4, 4), dtype=np.uint8)
        union[:, 1, 1] = 1
        union_path = root / "union.raw"
        union.tofile(union_path)
        receipts = []
        for window in plan["payload"]["windows"]:
            provenance = "authoritative" if window["branch"] == "center" else "temporal_dogfood"
            receipts.append({
                "window": window,
                "status": "complete",
                "seed_partition_count": 1,
                "seed_partition_index": 0,
                "retained_prediction_count": 0,
                "hole_fill_added_pixels": 0,
                "adapter": {"canonical_predictions_retained": False, "prompt_provenance": [provenance]},
            })
        return {
            "schema": "lta.propagation-chain/1",
            "status": "complete",
            "output_frame_range": [0, 59],
            "windows": receipts,
            "union": {"path": str(union_path), "shape": [59, 4, 4], "dtype": "uint8", "sha256": tool._sha256_file(union_path)},
            "lineage_active_frame_ranges": {seeds[0].lineage.token: [[0, 59]]},
        }

    def test_relocated_fixture_plans_both_branches_and_owns_each_frame_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, _ = self._fixture(root)
            windows = plan["payload"]["windows"]
            self.assertEqual(
                [(w["branch"], w["frame_start"], w["frame_stop"], w["prompt_frame"]) for w in windows],
                [("center", 5, 35, 19), ("backward", 0, 6, 5), ("forward", 34, 59, 34)],
            )
            self.assertEqual(plan["frame_ownership_counts"], [1] * 59)
            self.assertEqual(plan["model_frame_visit_bound_per_seed"], 61)
            self.assertEqual(plan["payload"]["cache_ref"]["path"], str((root / "cache.raw").resolve()))
            self.assertEqual(plan["payload"]["seed_artifact_path"], str((root / "seed.npz").resolve()))
            with self.assertRaisesRegex(ValueError, "byte length"):
                tool.prepare_fixture(cache_path=root / "cache.raw", cache_shape=(58, 4, 4), seed_path=root / "seed.npz", output_dir=root / "output")

    def test_verification_rejects_halted_or_incorrectly_conditioned_dogfood(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, seeds = self._fixture(root)
            manifest = self._manifest(root, plan, seeds)
            self.assertEqual(len(tool.validate_chain(manifest, plan, seeds, require_all_active=True)["active_frames"]), 59)
            halted = copy.deepcopy(manifest)
            halted["windows"][1]["status"] = "halted_empty_dogfood_boundary"
            with self.assertRaisesRegex(RuntimeError, "backward window did not complete"):
                tool.validate_chain(halted, plan, seeds)
            wrong_prompt = copy.deepcopy(manifest)
            wrong_prompt["windows"][2]["adapter"]["prompt_provenance"] = ["authoritative"]
            with self.assertRaisesRegex(RuntimeError, "prompt provenance"):
                tool.validate_chain(wrong_prompt, plan, seeds)

    def test_verification_detects_lost_authoritative_pixels_even_with_valid_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, seeds = self._fixture(root)
            manifest = self._manifest(root, plan, seeds)
            path = Path(manifest["union"]["path"])
            union = np.fromfile(path, dtype=np.uint8).reshape(59, 4, 4)
            union[19, 1, 1] = 0
            union.tofile(path)
            manifest["union"]["sha256"] = tool._sha256_file(path)
            with self.assertRaisesRegex(RuntimeError, "authoritative foreground was lost"):
                tool.validate_chain(manifest, plan, seeds)

    def test_sparse_union_validation_keeps_the_legacy_logical_mask_hash(self):
        from XTA.lta_union_artifacts import LtaUnionWriter
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, seeds = self._fixture(root)
            manifest = self._manifest(root, plan, seeds)
            legacy = tool.validate_chain(manifest, plan, seeds)
            masks = np.fromfile(manifest["union"]["path"], dtype=np.uint8).reshape(59, 4, 4)
            with LtaUnionWriter(root / "packed.bin", shape=masks.shape, frame_start=0) as writer:
                writer.append_chunk(0, masks[:30])
                writer.append_chunk(30, masks[30:])
                manifest["union"] = writer.finish()
            compact = tool.validate_chain(manifest, plan, seeds)
            self.assertEqual(compact["union_sha256"], legacy["union_sha256"])
            self.assertEqual(compact["hard_positive_missing_pixels"], [0])
            self.assertEqual(compact["active_frames"], list(range(59)))
            self.assertLess(compact["stored_union_bytes"], masks.nbytes)


if __name__ == "__main__":
    unittest.main()
