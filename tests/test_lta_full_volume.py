from __future__ import annotations

import gzip
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "lta_full_volume.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("lta_full_volume", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LtaFullVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = _load_tool()

    def test_exemplar_inventory_maps_one_based_positive_and_empty_labels(self) -> None:
        positive_label = b"0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sample_0001.jpg").write_bytes(b"image placeholder")
            (root / "sample_0001.txt").write_bytes(positive_label)
            (root / "sample_0003.png").write_bytes(b"image placeholder")
            (root / "sample_0003.txt").write_bytes(b"")

            anchors = self.tool.discover_positive_anchors(
                root,
                frame_count=5,
                index_origin=1,
            )
            backgrounds = self.tool.discover_known_background_frames(
                root,
                frame_count=5,
                index_origin=1,
            )

            self.assertEqual(len(anchors), 1)
            self.assertEqual((anchors[0].encoded_index, anchors[0].frame_index), (1, 0))
            self.assertEqual(anchors[0].image_path.name, "sample_0001.jpg")
            self.assertEqual(len(anchors[0].polygons), 1)
            self.assertEqual(backgrounds, (2,))

            (root / "sample_0004.txt").write_bytes(positive_label)
            with self.assertRaisesRegex(ValueError, "positive exemplar has no stem-matched image"):
                self.tool.discover_positive_anchors(
                    root,
                    frame_count=5,
                    index_origin=1,
                )

    def test_experiment_session_and_uncapped_window_can_exceed_thirty_frames(self) -> None:
        session = self.tool.ExperimentSessionPlan("full-volume", 3, 7, 79)
        self.assertEqual(session.frame_count, 72)

        domain = self.tool.AnchorDomain(anchor_frame=41, frame_start=7, frame_stop=79)
        windows = self.tool.plan_domain_windows(domain, chunk_frames=0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].frame_count, 72)
        self.assertEqual(
            (
                windows[0].frame_start,
                windows[0].frame_stop,
                windows[0].prompt_frame,
                windows[0].seed_kind,
                windows[0].direction,
            ),
            (7, 79, 41, "authoritative", "both"),
        )

    def test_native_tile_grid_is_row_major_gapless_and_edge_pinned(self) -> None:
        tiles = self.tool.plan_tile_grid(
            source_width=3024,
            source_height=3064,
            tile_size=1008,
            tile_stride=756,
        )
        xs = (0, 756, 1512, 2016)
        ys = (0, 756, 1512, 2056)
        self.assertEqual(len(tiles), 16)
        self.assertEqual(
            [(tile.left, tile.top) for tile in tiles],
            [(x, y) for y in ys for x in xs],
        )
        self.assertTrue(
            all(
                tile.xyxy[2] <= tile.source_width
                and tile.xyxy[3] <= tile.source_height
                and tile.size == 1008
                for tile in tiles
            )
        )
        self.assertEqual(max(tile.xyxy[2] for tile in tiles), 3024)
        self.assertEqual(max(tile.xyxy[3] for tile in tiles), 3064)
        self.assertTrue(all(right - left <= 1008 for left, right in zip(xs, xs[1:])))
        self.assertTrue(all(bottom - top <= 1008 for top, bottom in zip(ys, ys[1:])))

    def test_build_plan_serializes_conf_and_requested_profile_in_identity(self) -> None:
        polygon = SimpleNamespace(
            box_xyxy=(0.1, 0.1, 0.4, 0.4),
            points=((0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)),
        )
        tile = self.tool.TilePlan(
            left=0,
            top=0,
            size=10,
            source_width=20,
            source_height=20,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            anchor = self.tool.PositiveAnchor(
                encoded_index=3,
                frame_index=2,
                image_path=root / "sample_0003.jpg",
                label_path=root / "sample_0003.txt",
                label_sha256="a" * 64,
                polygons=(polygon,),
            )
            bundle = SimpleNamespace(
                root=root / "bundle",
                checkpoint_path=root / "bundle" / "model.pt",
                model_version="sam-test",
                checkpoint_identity_sha256="b" * 64,
            )
            common = {
                "video": {
                    "path": str(root / "sample.mkv"),
                    "width": 20,
                    "height": 20,
                    "frame_count": 10,
                    "fps": 5.0,
                },
                "bundle": bundle,
                "exemplar_root": root,
                "anchors": (anchor,),
                "known_background_frames": (4, 12),
                "tiles": (tile,),
                "selected_tile_indices": (0,),
                "frame_start": 0,
                "frame_stop": 10,
                "chunk_frames": 0,
                "seed_layout": "union",
                "spacing_xyz": (1.0, 1.0, 1.0),
            }
            plan = self.tool.build_plan(
                **common,
                conf=0.37,
                profile_requested="constrained",
            )
            changed_conf = self.tool.build_plan(
                **common,
                conf=0.38,
                profile_requested="constrained",
            )
            changed_profile = self.tool.build_plan(
                **common,
                conf=0.37,
                profile_requested="balanced",
            )

        self.assertEqual(plan["conf"], 0.37)
        self.assertEqual(plan["profile_requested"], "constrained")
        self.assertEqual(plan["known_background_frames"], [4])
        self.assertEqual(len(plan["plan_sha256"]), 64)
        self.assertNotEqual(plan["plan_sha256"], changed_conf["plan_sha256"])
        self.assertNotEqual(plan["plan_sha256"], changed_profile["plan_sha256"])

    def test_anchor_domains_partition_range_at_nearest_anchor_midpoints(self) -> None:
        domains = self.tool.plan_anchor_domains(
            (31, 20, 10, 20),
            frame_start=0,
            frame_stop=40,
        )
        self.assertEqual(
            [
                (domain.anchor_frame, domain.frame_start, domain.frame_stop)
                for domain in domains
            ],
            [(10, 0, 16), (20, 16, 26), (31, 26, 40)],
        )
        self.assertEqual(domains[0].frame_start, 0)
        self.assertEqual(domains[-1].frame_stop, 40)
        self.assertTrue(
            all(left.frame_stop == right.frame_start for left, right in zip(domains, domains[1:]))
        )

    def test_bounded_windows_overlap_once_and_owned_ranges_cover_exactly_once(self) -> None:
        domain = self.tool.AnchorDomain(anchor_frame=50, frame_start=0, frame_stop=101)
        windows = self.tool.plan_domain_windows(domain, chunk_frames=10)
        center = windows[0]
        backward = [window for window in windows[1:] if window.branch == "backward"]
        forward = [window for window in windows[1:] if window.branch == "forward"]

        self.assertEqual(
            (center.frame_start, center.frame_stop, center.prompt_frame),
            (46, 56, 50),
        )
        self.assertTrue(all(1 < window.frame_count <= 10 for window in windows))

        previous = center
        for window in backward:
            overlap = set(range(previous.frame_start, previous.frame_stop)) & set(
                range(window.frame_start, window.frame_stop)
            )
            self.assertEqual(overlap, {window.prompt_frame})
            self.assertEqual(window.prompt_frame, previous.frame_start)
            previous = window

        previous = center
        for window in forward:
            overlap = set(range(previous.frame_start, previous.frame_stop)) & set(
                range(window.frame_start, window.frame_stop)
            )
            self.assertEqual(overlap, {window.prompt_frame})
            self.assertEqual(window.prompt_frame, previous.frame_stop - 1)
            previous = window

        ownership = {frame: 0 for frame in range(domain.frame_start, domain.frame_stop)}
        for window in windows:
            owned_start, owned_stop = self.tool.owned_frame_range(window)
            if window.seed_kind == "dogfood" and window.direction == "backward":
                self.assertEqual(
                    (owned_start, owned_stop),
                    (window.frame_start, window.frame_stop - 1),
                )
            elif window.seed_kind == "dogfood" and window.direction == "forward":
                self.assertEqual(
                    (owned_start, owned_stop),
                    (window.frame_start + 1, window.frame_stop),
                )
            for frame in range(owned_start, owned_stop):
                ownership[frame] += 1

        self.assertEqual(set(ownership.values()), {1})
        self.assertEqual(
            sum(window.frame_count for window in windows),
            (domain.frame_stop - domain.frame_start) + len(windows) - 1,
        )

    def test_union_skips_dogfood_context_but_writes_owned_frames(self) -> None:
        import numpy as np

        window = self.tool.WindowPlan(
            branch="forward",
            ordinal=1,
            frame_start=11,
            frame_stop=15,
            prompt_frame=11,
            direction="forward",
            seed_kind="dogfood",
        )
        context_prediction = np.ones((3, 3), dtype=bool)
        first_owned = np.zeros((3, 3), dtype=bool)
        first_owned[1, 1] = True
        second_owned_a = np.zeros((3, 3), dtype=bool)
        second_owned_a[0, 2] = True
        second_owned_b = np.zeros((3, 3), dtype=bool)
        second_owned_b[2, 0] = True
        predictions = (
            SimpleNamespace(frame_index=11, object_id=0, binary_mask=context_prediction),
            SimpleNamespace(frame_index=12, object_id=0, binary_mask=first_owned),
            SimpleNamespace(frame_index=13, object_id=0, binary_mask=second_owned_a),
            SimpleNamespace(frame_index=13, object_id=1, binary_mask=second_owned_b),
        )

        with tempfile.TemporaryDirectory() as folder:
            store = np.memmap(
                Path(folder) / "tile.u8.dat",
                mode="w+",
                dtype=np.uint8,
                shape=(5, 3, 3),
            )
            try:
                store[:] = 0
                store[1, 0, 0] = 1
                context_before = store[1].copy()
                summary = self.tool._union_predictions_into_store(
                    predictions,
                    store_zyx=store,
                    selected_frame_start=10,
                    window=window,
                    tile_size=3,
                )
                store.flush()

                np.testing.assert_array_equal(store[1], context_before)
                np.testing.assert_array_equal(store[2] != 0, first_owned)
                np.testing.assert_array_equal(
                    store[3] != 0,
                    second_owned_a | second_owned_b,
                )
                self.assertFalse(bool(store[4].any()))
            finally:
                store.flush()
                store._mmap.close()
            store._mmap.close()

        self.assertEqual(summary["prediction_records"], 4)
        self.assertEqual(summary["returned_object_ids"], [0, 1])
        self.assertEqual(summary["model_active_frames"], [11, 12, 13])
        self.assertEqual(summary["owned_active_frames"], [12, 13])

    def test_offset_nrrd_header_and_gzip_payload_use_xyz_with_x_fastest(self) -> None:
        import numpy as np

        tile = self.tool.TilePlan(
            left=5,
            top=7,
            size=4,
            source_width=20,
            source_height=18,
        )
        volume = np.zeros((3, 4, 4), dtype=np.int16)
        volume[0, 0, 0] = 1
        volume[0, 0, 3] = 2
        volume[1, 2, 1] = -3
        volume[2, 3, 2] = 4

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tile.seg.nrrd"
            result = self.tool.write_offset_seg_nrrd(
                path,
                volume,
                tile=tile,
                frame_start=6,
                full_frame_count=20,
                spacing_xyz=(0.5, 2.0, 4.0),
                segment_name="Tiny\ntile",
                gzip_level=1,
            )
            header_bytes, compressed_payload = path.read_bytes().split(b"\n\n", 1)

        header = header_bytes.decode("ascii")
        self.assertIn("type: uint8", header)
        self.assertIn("dimension: 3", header)
        self.assertIn("sizes: 4 4 3", header)
        self.assertIn("space directions: (0.5,0,0) (0,2,0) (0,0,4)", header)
        self.assertIn("space origin: (2.5,14,24)", header)
        self.assertIn("encoding: gzip", header)
        self.assertIn("endian: little", header)
        self.assertIn("Segmentation_ReferenceImageExtent:=0 19 0 17 0 19", header)
        self.assertIn("Segmentation_ReferenceImageExtentOffset:=5 7 6", header)
        self.assertIn("Segment0_Name:=Tiny tile", header)
        self.assertIn("Segment0_Extent:=0 3 0 3 0 2", header)

        expected = np.ascontiguousarray(volume != 0, dtype=np.uint8).tobytes(order="C")
        payload = gzip.decompress(compressed_payload)
        self.assertEqual(payload, expected)
        self.assertEqual(
            [index for index, value in enumerate(payload) if value],
            [0, 3, 25, 46],
        )
        self.assertEqual(result["shape_zyx"], [3, 4, 4])
        self.assertEqual(result["global_offset_xyz"], [5, 7, 6])
        self.assertEqual(result["space_origin"], [2.5, 14.0, 24.0])
        self.assertEqual(result["nonzero_voxels"], 4)

    def test_all_zero_nrrd_has_empty_local_extent_and_global_offset(self) -> None:
        import numpy as np

        tile = self.tool.TilePlan(
            left=13,
            top=17,
            size=4,
            source_width=40,
            source_height=50,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.seg.nrrd"
            result = self.tool.write_offset_seg_nrrd(
                path,
                np.zeros((2, 4, 4), dtype=np.uint8),
                tile=tile,
                frame_start=8,
                full_frame_count=25,
            )
            header_bytes, compressed_payload = path.read_bytes().split(b"\n\n", 1)

        header = header_bytes.decode("ascii")
        self.assertIn("space origin: (13,17,8)", header)
        self.assertIn("Segmentation_ReferenceImageExtentOffset:=13 17 8", header)
        self.assertIn("Segment0_Extent:=0 -1 0 -1 0 -1", header)
        self.assertEqual(gzip.decompress(compressed_payload), bytes(2 * 4 * 4))
        self.assertEqual(result["global_offset_xyz"], [13, 17, 8])
        self.assertEqual(result["segment_extent_local_x_y_z"], [0, -1, 0, -1, 0, -1])
        self.assertEqual(result["nonzero_voxels"], 0)

    def test_tiny_ffv1_tile_mask_video_round_trips_losslessly_when_available(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/NumPy are unavailable")
        if getattr(cv2, "__file__", None) is None:
            self.skipTest("real OpenCV is unavailable")

        masks = np.zeros((4, 8, 10), dtype=np.uint8)
        masks[0, 1:4, 2:5] = 1
        masks[1, ::2, ::3] = 1
        masks[2, 5:, 6:] = 1
        masks[3, :, 9] = 1

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.mkv"
            try:
                result = self.tool.write_tile_mask_video(path, masks, fps=7.5)
            except RuntimeError as exc:
                if "could not create" in str(exc).lower():
                    self.skipTest(f"OpenCV build has no FFV1 writer: {exc}")
                raise

            capture = cv2.VideoCapture(str(path))
            self.assertTrue(capture.isOpened())
            decoded = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    plane = frame if frame.ndim == 2 else frame[:, :, 0]
                    decoded.append(plane.copy())
            finally:
                capture.release()

        self.assertEqual(len(decoded), len(masks))
        for actual, expected in zip(decoded, masks):
            np.testing.assert_array_equal(actual, expected * 255)
        self.assertEqual(result["codec"], "ffv1")
        self.assertTrue(result["lossless"])
        self.assertEqual(result["frame_count"], 4)
        self.assertEqual(result["foreground_voxels"], int(np.count_nonzero(masks)))

    def test_completed_tile_artifacts_are_checksum_validated(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mask = root / "tile_mask.mkv"
            nrrd = root / "chunk.seg.nrrd"
            mask.write_bytes(b"lossless-mask")
            nrrd.write_bytes(b"nrrd-chunk")
            summary = {
                "tile_mask_video": {
                    "path": mask.name,
                    "sha256": self.tool._sha256_file(mask),
                },
                "executed_windows": [
                    {
                        "nrrd": {
                            "path": nrrd.name,
                            "sha256": self.tool._sha256_file(nrrd),
                        }
                    }
                ],
            }
            self.tool.validate_completed_tile_artifacts(root, summary)
            nrrd.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.tool.validate_completed_tile_artifacts(root, summary)

    def test_h100_profile_rejects_pre_hopper_device(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=lambda _device: (8, 9))
        )
        with self.assertRaisesRegex(ValueError, "capability 9"):
            self.tool._resolve_profile(fake_torch, 0, "h100")
        self.assertEqual(self.tool._resolve_profile(fake_torch, 0, "auto")["name"], "egpu")

        hopper = SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=lambda _device: (9, 0))
        )
        profile = self.tool._resolve_profile(hopper, 0, "h100")
        self.assertFalse(profile["use_fa3"])
        self.assertFalse(profile["use_rope_real"])
        self.assertTrue(profile["sdpa_fallback"])
        self.assertEqual(
            profile["attention_backend"], "torch_sdpa_flash_efficient_math"
        )


if __name__ == "__main__":
    unittest.main()
